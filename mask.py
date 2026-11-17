"""Save a cut-out mask of the person for every frame: <name>_mask.mp4.

White = person, black = room, soft edges kept. Same size, frame rate and frame
count as the clip.

  Mac:            Apple Vision VNGeneratePersonSegmentationRequest (.accurate),
                  in a small Swift program compiled once with swiftc.
  Windows/Linux:  MediaPipe Tasks ImageSegmenter: selfie multiclass model on
                  overlapping square tiles (selfie_segmenter.tflite is also downloaded).

ffmpeg decodes the frames in both cases (so phone rotation is handled) and
pipes raw RGB into the segmenter.

usage: .venv/bin/python mask.py [clip] [--seconds N]
"""
import argparse
import os
import platform
import struct
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from common import HERE, VideoWriter, count_frames, find_clip, model_path, probe, read_frames, stem

SWIFT_SRC = r'''
import Foundation
import Vision
import CoreVideo

// Reads raw RGB24 frames (W x H) from stdin. For each frame writes
// UInt32 mw, UInt32 mh, then mw*mh bytes of 8-bit person mask.
let args = CommandLine.arguments
let W = Int(args[1])!, H = Int(args[2])!
let frameBytes = W * H * 3
let input = FileHandle.standardInput
let output = FileHandle.standardOutput
let request = VNGeneratePersonSegmentationRequest()
request.qualityLevel = .accurate
request.outputPixelFormat = kCVPixelFormatType_OneComponent8

var pb: CVPixelBuffer?
CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA,
                    [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
let buffer = pb!

func readExactly(_ n: Int) -> Data? {
    var data = Data(capacity: n)
    while data.count < n {
        let chunk = input.readData(ofLength: n - data.count)
        if chunk.isEmpty { return nil }
        data.append(chunk)
    }
    return data
}

while let rgb = readExactly(frameBytes) {
    CVPixelBufferLockBaseAddress(buffer, [])
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    rgb.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        let s = src.bindMemory(to: UInt8.self)
        for y in 0..<H {
            let row = base + y * stride
            for x in 0..<W {
                let i = (y * W + x) * 3
                row[x*4+0] = s[i+2]; row[x*4+1] = s[i+1]; row[x*4+2] = s[i]; row[x*4+3] = 255
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])

    let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
    var mw: UInt32 = 0, mh: UInt32 = 0
    var bytes = Data()
    if (try? handler.perform([request])) != nil, let m = request.results?.first?.pixelBuffer {
        CVPixelBufferLockBaseAddress(m, .readOnly)
        let w = CVPixelBufferGetWidth(m), h = CVPixelBufferGetHeight(m)
        let st = CVPixelBufferGetBytesPerRow(m)
        let p = CVPixelBufferGetBaseAddress(m)!.assumingMemoryBound(to: UInt8.self)
        bytes.reserveCapacity(w * h)
        for y in 0..<h { bytes.append(p + y * st, count: w) }
        CVPixelBufferUnlockBaseAddress(m, .readOnly)
        mw = UInt32(w); mh = UInt32(h)
    }
    var header = Data()
    withUnsafeBytes(of: mw.littleEndian) { header.append(contentsOf: $0) }
    withUnsafeBytes(of: mh.littleEndian) { header.append(contentsOf: $0) }
    output.write(header)
    output.write(bytes)
}
'''


def apple_vision_masks(clip, info, seconds):
    build = os.path.join(HERE, ".build")
    os.makedirs(build, exist_ok=True)
    src = os.path.join(build, "personseg.swift")
    exe = os.path.join(build, "personseg")
    stale = not (os.path.exists(exe) and os.path.exists(src) and open(src).read() == SWIFT_SRC)
    if stale:
        with open(src, "w") as f:
            f.write(SWIFT_SRC)
        print("compiling Apple Vision helper with swiftc ...")
        subprocess.run(["swiftc", "-O", src, "-o", exe], check=True)
    w, h = info["width"], info["height"]
    proc = subprocess.Popen([exe, str(w), str(h)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    blank = np.zeros((h, w), np.uint8)
    for frame in read_frames(clip, w, h, seconds=seconds):
        proc.stdin.write(frame.tobytes())
        proc.stdin.flush()
        mw, mh = struct.unpack("<II", proc.stdout.read(8))
        if mw == 0:
            yield blank
            continue
        m = np.frombuffer(proc.stdout.read(mw * mh), np.uint8).reshape(mh, mw)
        yield np.asarray(Image.fromarray(m).resize((w, h), Image.BICUBIC))
    proc.stdin.close()
    proc.wait()


def tile_starts(long, short):
    """Square tiles of side `short` covering `long` with overlap."""
    n = int(np.ceil(long / short))
    if n == 1:
        return [0]
    return [int(round(k * (long - short) / (n - 1))) for k in range(n)]


def mediapipe_masks(clip, info, seconds):
    """MediaPipe selfie segmenter. Uses the multiclass selfie model (much better
    outlines than the plain one) on overlapping square tiles, so a tall phone
    frame isn't squashed into one 256x256 square."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    # The plain selfie_segmenter.tflite is kept as a fallback model on disk.
    model_path("selfie_segmenter.tflite")
    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path("selfie_multiclass_256x256.tflite")),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True, output_category_mask=False)
    w, h = info["width"], info["height"]
    fps = float(info["fps"])
    short, long = min(w, h), max(w, h)
    starts = tile_starts(long, short)
    # Tent weights along the long axis so tiles blend smoothly in the overlap.
    ramp = np.minimum(np.arange(short) + 1, short - np.arange(short)).astype(np.float32)
    ramp = np.minimum(ramp, short * 0.25)
    segs = [vision.ImageSegmenter.create_from_options(opts) for _ in starts]  # one per tile (VIDEO mode keeps state)
    try:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=seconds)):
            acc = np.zeros((h, w), np.float32)
            wsum = np.zeros((h, w), np.float32)
            for seg, st in zip(segs, starts):
                tile = frame[st:st + short] if h >= w else frame[:, st:st + short]
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(tile))
                res = seg.segment_for_video(img, int(round(i * 1000 / fps)))
                bg = np.squeeze(res.confidence_masks[0].numpy_view()).astype(np.float32)
                person = cv2.resize(1.0 - bg, (short, short), interpolation=cv2.INTER_LINEAR)
                if h >= w:
                    acc[st:st + short] += person * ramp[:, None]
                    wsum[st:st + short] += ramp[:, None]
                else:
                    acc[:, st:st + short] += person * ramp[None, :]
                    wsum[:, st:st + short] += ramp[None, :]
            p = acc / np.maximum(wsum, 1e-6)
            # The model never says exactly 0/1 (room ~0.08, me ~0.98): stretch to black/white.
            p = np.clip((p - 0.10) / 0.85, 0, 1)
            yield np.clip(p * 255 + 0.5, 0, 255).astype(np.uint8)
    finally:
        for seg in segs:
            seg.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--seconds", type=float, help="only the first N seconds (for tests)")
    ap.add_argument("--out", help="output path (default <name>_mask.mp4)")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    out = a.out or stem(clip) + "_mask.mp4"
    gen = apple_vision_masks if platform.system() == "Darwin" else mediapipe_masks
    print(f"{os.path.basename(clip)}: {info['width']}x{info['height']} @ {info['rate']} fps, "
          f"engine={'Apple Vision' if gen is apple_vision_masks else 'MediaPipe selfie multiclass (tiled)'}")
    wr = VideoWriter(out, info["width"], info["height"], info["rate"], pix="gray", crf=10)
    for m in gen(clip, info, a.seconds):
        wr.write(m)
        if wr.count % 100 == 0:
            print(f"  {wr.count} frames", file=sys.stderr)
    wr.close()
    print(f"wrote {out} ({wr.count} frames)")
    n = count_frames(out)
    if n != wr.count:
        print(f"WARNING: mask has {n} frames, expected {wr.count}")


if __name__ == "__main__":
    main()
