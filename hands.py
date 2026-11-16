"""Track up to 2 hands on every frame with MediaPipe's hand landmarker.

Saves <name>_hands.json. All coordinates are fractions of the frame
(x: 0 = left edge of the screen, 1 = right edge; y: 0 = top, 1 = bottom).

Per frame: {"i": frame index, "t": seconds, "hands": [ {
    "id": track id (stable while the hand stays in view),
    "points": 21 x [x, y, z]  (MediaPipe landmark order, z relative to wrist),
    "score": detection confidence 0..1,
    "hand": "left" | "right"   (the person's own hand),
    "side": "left" | "right"   (half of the screen it is on, as seen on screen),
    "filled": true if interpolated across a short gap } ]}

A second hand that sits on top of the first (a duplicate detection) is dropped.
Gaps of up to 3 frames are filled by interpolating the same hand.

MediaPipe labels handedness assuming a mirrored (selfie-preview) image. Phone
videos are saved un-mirrored, so by default the label is swapped to give the
person's real hand. Pass --mirrored if your clip is mirrored.

usage: .venv/bin/python hands.py [clip] [--seconds N] [--mirrored]
"""
import argparse
import json
import sys

import numpy as np

from common import find_clip, model_path, probe, read_frames, stem

MAX_GAP = 3


def bbox(p):
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def overlap(a, b):
    """Intersection over the smaller box: 1.0 when one hand sits on the other."""
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    small = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)) + 1e-9
    return iw * ih / small


def detect(clip, info, seconds, mirrored):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    w, h, fps = info["width"], info["height"], float(info["fps"])
    frames = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i, frame in enumerate(read_frames(clip, w, h, seconds=seconds)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            res = lm.detect_for_video(img, int(round(i * 1000 / fps)))
            hands = []
            for pts, hd in zip(res.hand_landmarks, res.handedness):
                p = np.array([[q.x, q.y, q.z] for q in pts], np.float64)
                label = hd[0].category_name.lower()  # 'left' / 'right' (mirrored assumption)
                if not mirrored:
                    label = "right" if label == "left" else "left"
                hands.append(dict(points=p, score=float(hd[0].score), hand=label))
            hands.sort(key=lambda d: -d["score"])
            # Drop a second hand that sits on top of the first.
            if len(hands) == 2 and overlap(hands[0]["points"][:, :2], hands[1]["points"][:, :2]) > 0.5:
                hands = hands[:1]
            frames.append(hands)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1} frames", file=sys.stderr)
    return frames


def track(frames):
    """Give each detection a stable id by matching to the previous frame."""
    next_id, last = 0, {}  # id -> (frame index, points, hand label)
    for i, hands in enumerate(frames):
        used = set()
        for hnd in hands:
            c = hnd["points"][:, :2].mean(0)
            best, bd = None, 0.25
            for tid, (j, p, lab) in last.items():
                if tid in used or i - j > MAX_GAP + 1:
                    continue
                d = np.linalg.norm(p[:, :2].mean(0) - c) + (0.1 if lab != hnd["hand"] else 0)
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best, next_id = next_id, next_id + 1
            used.add(best)
            hnd["id"] = best
            last[best] = (i, hnd["points"], hnd["hand"])


def fill_gaps(frames):
    """Interpolate a hand that disappears for up to MAX_GAP frames."""
    seen = {}  # id -> list of frame indices
    for i, hands in enumerate(frames):
        for hnd in hands:
            seen.setdefault(hnd["id"], []).append(i)
    for tid, idx in seen.items():
        for a, b in zip(idx, idx[1:]):
            if 1 < b - a <= MAX_GAP + 1:
                ha = next(x for x in frames[a] if x["id"] == tid)
                hb = next(x for x in frames[b] if x["id"] == tid)
                for k in range(a + 1, b):
                    if len(frames[k]) >= 2:
                        continue
                    t = (k - a) / (b - a)
                    p = ha["points"] * (1 - t) + hb["points"] * t
                    if frames[k] and overlap(frames[k][0]["points"][:, :2], p[:, :2]) > 0.5:
                        continue
                    frames[k].append(dict(points=p, score=min(ha["score"], hb["score"]),
                                          hand=ha["hand"], id=tid, filled=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--mirrored", action="store_true", help="clip is mirrored like a selfie preview")
    ap.add_argument("--out")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    frames = detect(clip, info, a.seconds, a.mirrored)
    track(frames)
    fill_gaps(frames)
    fps = float(info["fps"])
    out_frames = []
    for i, hands in enumerate(frames):
        rec = []
        for hnd in sorted(hands, key=lambda d: d["id"]):
            p = hnd["points"]
            rec.append(dict(id=hnd["id"], points=np.round(p, 5).tolist(), score=round(hnd["score"], 4),
                            hand=hnd["hand"], side="left" if p[:, 0].mean() < 0.5 else "right",
                            filled=bool(hnd.get("filled", False))))
        out_frames.append(dict(i=i, t=round(i / fps, 4), hands=rec))
    out = a.out or stem(clip) + "_hands.json"
    with open(out, "w") as f:
        json.dump(dict(width=info["width"], height=info["height"], fps=info["rate"],
                       frame_count=len(out_frames), mirrored=a.mirrored,
                       coords="fractions of the frame; x from screen left, y from top",
                       frames=out_frames), f)
    n = sum(len(f["hands"]) for f in out_frames)
    print(f"wrote {out}: {len(out_frames)} frames, {n} hand detections")


if __name__ == "__main__":
    main()
