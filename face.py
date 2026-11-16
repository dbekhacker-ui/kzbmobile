"""Find the face on every frame with MediaPipe's face detector.

Saves <name>_face.json with one box per frame as fractions of the frame:
{"i", "t", "x", "y", "w", "h", "detected"} where x, y is the top-left corner
(x from the screen's left edge, y from the top). The box is smoothed over time
so it doesn't jitter; frames where the face wasn't found are filled from the
nearest detections ("detected": false). If no face is found at all the boxes
are null.

usage: .venv/bin/python face.py [clip] [--seconds N]
"""
import argparse
import json
import sys

import numpy as np

from common import find_clip, model_path, probe, read_frames, stem


def detect(clip, info, seconds):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path("blaze_face_short_range.tflite")),
        running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.5)
    w, h, fps = info["width"], info["height"], float(info["fps"])
    boxes, prev = [], None
    with vision.FaceDetector.create_from_options(opts) as det:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=seconds)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            res = det.detect_for_video(img, int(round(i * 1000 / fps)))
            cands = []
            for d in res.detections:
                b = d.bounding_box
                box = np.array([b.origin_x / w, b.origin_y / h, b.width / w, b.height / h])
                score = d.categories[0].score
                # Prefer the biggest face, and the one closest to last frame's face.
                rank = score * box[2] * box[3]
                if prev is not None:
                    rank /= 1 + 10 * np.linalg.norm((box[:2] + box[2:] / 2) - (prev[:2] + prev[2:] / 2))
                cands.append((rank, box))
            if cands:
                prev = max(cands, key=lambda c: c[0])[1]
                boxes.append(prev)
            else:
                boxes.append(None)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} frames", file=sys.stderr)
    return boxes, fps


def smooth(boxes, fps):
    n = len(boxes)
    found = np.array([b is not None for b in boxes])
    if not found.any():
        return None, found
    idx = np.arange(n)
    arr = np.zeros((n, 4))
    for k in range(4):  # fill gaps by linear interpolation, hold at the ends
        arr[:, k] = np.interp(idx, idx[found], np.array([b[k] for b in boxes if b is not None]))
    # Work on centre + size, then zero-phase Gaussian smoothing (~0.15 s).
    cs = np.column_stack([arr[:, 0] + arr[:, 2] / 2, arr[:, 1] + arr[:, 3] / 2, arr[:, 2], arr[:, 3]])
    sigma = max(1.0, 0.15 * fps)
    r = int(3 * sigma)
    kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    padded = np.pad(cs, ((r, r), (0, 0)), mode="edge")
    sm = np.column_stack([np.convolve(padded[:, k], kern / kern.sum(), mode="valid") for k in range(4)])
    out = np.column_stack([sm[:, 0] - sm[:, 2] / 2, sm[:, 1] - sm[:, 3] / 2, sm[:, 2], sm[:, 3]])
    return np.clip(out, 0, 1), found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    boxes, fps = detect(clip, info, a.seconds)
    sm, found = smooth(boxes, fps)
    frames = []
    for i in range(len(boxes)):
        if sm is None:
            frames.append(dict(i=i, t=round(i / fps, 4), x=None, y=None, w=None, h=None, detected=False))
        else:
            x, y, bw, bh = (round(float(v), 5) for v in sm[i])
            frames.append(dict(i=i, t=round(i / fps, 4), x=x, y=y, w=bw, h=bh, detected=bool(found[i])))
    out = a.out or stem(clip) + "_face.json"
    with open(out, "w") as f:
        json.dump(dict(width=info["width"], height=info["height"], fps=info["rate"],
                       frame_count=len(frames),
                       coords="fractions of the frame; x,y = top-left from screen left/top",
                       frames=frames), f)
    print(f"wrote {out}: {len(frames)} frames, face found on {int(found.sum())}")


if __name__ == "__main__":
    main()
