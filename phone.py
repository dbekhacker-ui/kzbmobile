"""Mask of the phone in my hand on every frame: <name>_phone.mp4.

The person segmenter sometimes drops the phone (its back is metal/glass, not
a person), so effects would show through it. This finds the phone with
MediaPipe's object detector (COCO "cell phone", efficientdet_lite2) and cuts
out its exact shape with MediaPipe's interactive segmenter (magic_touch),
pointed at the middle of the detected box. The cut-out is kept inside the
box (grown a little), and short detection gaps are bridged from the previous
box so a fast, blurry phone doesn't blink out.

White = phone, black = everything else. Same size, frame rate and frame
count as the clip. put_back.py adds it to the person mask automatically.

usage: .venv/bin/python phone.py [clip] [--seconds N]
"""
import argparse
import sys

import cv2
import numpy as np

from common import (MODEL_URLS, VideoWriter, count_frames, find_clip, model_path, probe,
                    read_frames, stem)

MODEL_URLS.setdefault("efficientdet_lite2.tflite",
                      "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite")
MODEL_URLS.setdefault("magic_touch.tflite",
                      "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite")
HOLD = 3  # frames to keep using the last box when detection drops out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    w, h, fps = info["width"], info["height"], float(info["fps"])

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from mediapipe.tasks.python.components.containers import keypoint as kp
    # Point-prompted segmenter: named "...Legacy" in current MediaPipe releases.
    from mediapipe.tasks.python.vision.interactive_segmenter_legacy import (
        InteractiveSegmenterLegacy as Seg, InteractiveSegmenterLegacyOptions as SegOptions,
        RegionOfInterest)

    det_opts = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path("efficientdet_lite2.tflite")),
        running_mode=vision.RunningMode.VIDEO, score_threshold=0.25, category_allowlist=["cell phone"],
        max_results=2)
    seg_opts = SegOptions(
        base_options=BaseOptions(model_asset_path=model_path("magic_touch.tflite")),
        output_confidence_masks=True, output_category_mask=False)
    out = a.out or stem(clip) + "_phone.mp4"

    # Pass 1: detect phone boxes.
    boxes = []
    with vision.ObjectDetector.create_from_options(det_opts) as det:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=a.seconds)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            res = det.detect_for_video(img, int(round(i * 1000 / fps)))
            if res.detections:
                bb = max(res.detections, key=lambda d: d.categories[0].score).bounding_box
                boxes.append(np.array([bb.origin_x, bb.origin_y, bb.width, bb.height], np.float64))
            else:
                boxes.append(None)
    found = sum(b is not None for b in boxes)

    # Pass 2: bridge gaps (fast, blurry swings) by carrying the box with the hand holding it.
    hands = load_hands(clip, w, h)
    filled = fill_gaps(boxes, hands)

    # Pass 3: cut out the phone's exact shape inside each box.
    # On bridged (guessed) boxes the cut-out must look like the phone: same
    # colour as on the last really-detected frame and a phone-like size;
    # otherwise it is dropped, so a wrong guess can never paste room over effects.
    wr = VideoWriter(out, w, h, info["rate"], pix="gray", crf=10)
    ref, rejected = None, 0
    with Seg.create_from_options(seg_opts) as seg:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=a.seconds)):
            m = np.zeros((h, w), np.uint8)
            box = filled[i] if i < len(filled) else None
            if box is not None:
                cut_phone(seg, mp, RegionOfInterest, kp, frame, box, m)
                sel = m > 127
                if sel.any():
                    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[sel].astype(np.float32).mean(0)
                    if boxes[i] is not None:
                        ref = lab
                    else:
                        area_ok = 0.25 <= sel.sum() / max(1.0, box[2] * box[3]) <= 1.3
                        if ref is None or not area_ok or np.linalg.norm(lab - ref) > 28:
                            m[:] = 0
                            rejected += 1
            wr.write(m)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} frames", file=sys.stderr)
    wr.close()
    n_fill = sum(1 for b, f in zip(boxes, filled) if b is None and f is not None)
    print(f"wrote {out}: {wr.count} frames, phone detected on {found}, bridged {n_fill - rejected} more "
          f"({rejected} bridged cut-outs rejected as not-phone), "
          f"{count_frames(out)} frames in file")


def load_hands(clip, w, h):
    import json
    import os
    hp = stem(clip) + "_hands.json"
    if not os.path.exists(hp):
        return None
    return [[np.array(hd["points"])[:, :2] * (w, h) for hd in f["hands"]] for f in json.load(open(hp))["frames"]]


def nearest_hand(hands, i, c):
    if hands is None or i >= len(hands) or not hands[i]:
        return None
    pts = min(hands[i], key=lambda p: np.linalg.norm(p.mean(0) - c))
    return pts.mean(0) if np.linalg.norm(pts.mean(0) - c) < 600 else None  # px: hand must be near the phone


def fill_gaps(boxes, hands, max_gap=24):
    """Fill runs of missing boxes (<= max_gap frames) between two detections.
    Each box keeps its offset from the hand that holds it; the offset and size
    blend from the box before the gap to the box after it."""
    out = list(boxes)
    idx = [i for i, b in enumerate(boxes) if b is not None]
    for a, b in zip(idx, idx[1:]):
        if b - a <= 1 or b - a > max_gap + 1:
            continue
        A, B = boxes[a], boxes[b]
        ca, cb = A[:2] + A[2:] / 2, B[:2] + B[2:] / 2
        ha, hb = nearest_hand(hands, a, ca), nearest_hand(hands, b, cb)
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            size = A[2:] * (1 - t) + B[2:] * t
            hk = nearest_hand(hands, k, ca * (1 - t) + cb * t) if ha is not None and hb is not None else None
            if hk is not None:
                c = hk + (ca - ha) * (1 - t) + (cb - hb) * t
            else:
                c = ca * (1 - t) + cb * t
            out[k] = np.array([c[0] - size[0] / 2, c[1] - size[1] / 2, size[0], size[1]])
    # Short dropouts at the end of a run: hold the last box a few frames.
    last, age = None, 99
    for i in range(len(out)):
        if out[i] is not None:
            last, age = out[i], 0
        elif last is not None and age < HOLD:
            out[i], age = last, age + 1
        else:
            age += 1
    return out


def cut_phone(seg, mp, RegionOfInterest, kp, frame, box, m):
    h, w = m.shape
    x, y, bw, bh = (float(v) for v in box)
    gx, gy = int(bw * 0.12) + 4, int(bh * 0.12) + 4
    x0, y0 = max(0, int(x) - gx), max(0, int(y) - gy)
    x1, y1 = min(w, int(x + bw) + gx), min(h, int(y + bh) + gy)
    crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
    if crop.size == 0 or x1 - x0 < 8 or y1 - y0 < 8:
        return
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=kp.NormalizedKeypoint(x=(x + bw / 2 - x0) / (x1 - x0),
                                                          y=(y + bh / 2 - y0) / (y1 - y0)))
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), roi)
    c = np.squeeze(r.confidence_masks[0].numpy_view()).astype(np.float32)
    c = cv2.resize(c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    # Keep only the blob under the centre point (no stray hits on the room).
    n, lab = cv2.connectedComponents((c > 0.5).astype(np.uint8))
    cy, cx = int((y + bh / 2) - y0), int((x + bw / 2) - x0)
    k = lab[min(max(cy, 0), lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)]
    if k == 0:
        return
    keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))
    # Never reach more than a hair past the box (no room spill).
    tx, ty = int(bw * 0.04) + 3, int(bh * 0.04) + 3
    box_m = np.zeros_like(keep)
    box_m[max(0, int(y) - ty - y0):max(0, int(y + bh) + ty - y0), max(0, int(x) - tx - x0):max(0, int(x + bw) + tx - x0)] = 1
    m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], np.clip(c * keep * box_m * 255 + 0.5, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    main()
