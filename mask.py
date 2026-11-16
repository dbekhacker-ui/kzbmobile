"""Save a cut-out mask of the person for every frame: <name>_mask.mp4.

White = person, black = room, soft edges kept. Same size, frame rate and frame
count as the clip.

  Mac:            Apple Vision VNGeneratePersonSegmentationRequest (.accurate),
                  in a small Swift program compiled once with swiftc.
  Windows/Linux:  MediaPipe Tasks ImageSegmenter with selfie_segmenter.tflite.

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


def mediapipe_masks(clip, info, seconds):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path("selfie_segmenter.tflite")),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True, output_category_mask=False)
    w, h = info["width"], info["height"]
    fps = float(info["fps"])
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=seconds)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            res = seg.segment_for_video(img, int(round(i * 1000 / fps)))
            conf = res.confidence_masks[-1].numpy_view()  # person probability, 0..1
            conf = np.squeeze(conf).astype(np.float32)
            if conf.shape != (h, w):
                conf = np.asarray(Image.fromarray(conf).resize((w, h), Image.BILINEAR))
            yield np.clip(conf * 255 + 0.5, 0, 255).astype(np.uint8)


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
          f"engine={'Apple Vision' if gen is apple_vision_masks else 'MediaPipe selfie segmenter'}")
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
