"""One big word behind me at head height.

White Anton letters, ~4% margin each side, rising into place one at a time
(0.09 s apart, from ~1/3 of their height below, fading in with a little
overshoot), soft black shadow just under them, room dimmed to 78%, word grows
3% over the clip. I'm put back on top with put_back.py.

usage: .venv/bin/python text_behind.py [clip] [--word CLAUDE] [--seconds N] [--out text-behind.mp4]
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import HERE, VideoWriter, count_frames, find_clip, probe, stem
from put_back import PutBack

FONT = os.path.join(HERE, "fonts", "Anton-Regular.ttf")
MARGIN = 0.04          # each side
STAGGER = 0.09         # seconds between letters
RISE_TIME = 0.5        # seconds for one letter to settle
RISE_FROM = 1 / 3      # of letter height, below
GROW = 0.03            # word grows 3% over the clip
DIM = 0.78             # room brightness
SS = 2                 # supersampling for the letter sprites


def ease_out_back(p, c1=1.1):
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def letter_sprites(word, W):
    """Render each letter at supersampled size. Returns list of
    (alpha, shadow, x_left_px, top_px) in frame pixels at scale 1, plus cap height."""
    target = W * (1 - 2 * MARGIN)
    ref = ImageFont.truetype(FONT, 400)
    size = 400 * target / ref.getlength(word)
    font = ImageFont.truetype(FONT, size * SS)
    # Cap-height box of the whole word, so all letters share one baseline.
    x0, top, x1, bottom = font.getbbox(word)
    cap = (bottom - top) / SS
    pad = int(cap * 0.5 * SS)
    sprites = []
    for k, ch in enumerate(word):
        adv = font.getlength(word[:k])  # pen x of this letter (includes kerning)
        bx0, _, bx1, _ = font.getbbox(ch)
        w = int(bx1 - bx0) + 2 * pad
        h = int(bottom - top) + 2 * pad
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((pad - bx0, pad - top), ch, font=font, fill=255)
        a = np.asarray(im, np.float32) / 255.0
        # Soft shadow just under the letter.
        sh = cv2.GaussianBlur(a, (0, 0), cap * 0.035 * SS)
        M = np.float32([[1, 0, 0], [0, 1, cap * 0.03 * SS]])
        sh = cv2.warpAffine(sh, M, (w, h)) * 0.6
        # Sprite's top-left in frame pixels relative to the word's left edge / cap top.
        sprites.append((a, sh, (adv + bx0 - x0 - pad) / SS, -pad / SS))
    return sprites, cap, target


def head_height(face_json, H):
    data = json.load(open(face_json))
    ys = [f["y"] + f["h"] / 2 for f in data["frames"] if f["detected"]]
    if not ys:
        raise SystemExit("no face found - can't place the word at head height")
    ws = [f["w"] for f in data["frames"] if f["detected"]]
    return float(np.median(ys)) * H, float(np.median(ws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--word", default="CLAUDE")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--out", default=os.path.join(HERE, "text-behind.mp4"))
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    W, H, fps = info["width"], info["height"], float(info["fps"])
    total = count_frames(clip)
    dur = total / fps
    word = a.word.upper()

    yc, face_w = head_height(stem(clip) + "_face.json", H)
    sprites, cap, word_w = letter_sprites(word, W)
    print(f"word {word}: {word_w:.0f}px wide (face ~{face_w * W:.0f}px), cap {cap:.0f}px, centre y {yc:.0f}px")

    cx = W / 2
    wr = VideoWriter(a.out, W, H, info["rate"], audio_from=clip, seconds=a.seconds)
    for layer in PutBack(clip, seconds=a.seconds):
        t = layer.t
        s = 1 + GROW * min(1.0, t / dur)
        text = np.zeros((H, W), np.float32)
        shadow = np.zeros((H, W), np.float32)
        for k, (alpha, sh, dx, dy) in enumerate(sprites):
            p = np.clip((t - k * STAGGER) / RISE_TIME, 0, 1)
            if p <= 0:
                continue
            rise = (1 - ease_out_back(p)) * cap * RISE_FROM   # >0 below, <0 overshoot above
            opacity = ease_out_cubic(min(1.0, p / 0.6))
            # Word-space -> frame: scale about the word centre (cx, yc).
            left = cx + (dx - word_w / 2) * s
            top = yc + (dy - cap / 2) * s + rise * s
            M = np.float32([[s / SS, 0, left], [0, s / SS, top]])
            text = np.maximum(text, cv2.warpAffine(alpha, M, (W, H)) * opacity)
            shadow = np.maximum(shadow, cv2.warpAffine(sh, M, (W, H)) * opacity)
        bg = layer.frame.astype(np.float32) * DIM
        bg *= (1 - shadow)[..., None]                       # black shadow
        bg = bg * (1 - text)[..., None] + 255.0 * text[..., None]  # white letters
        wr.write(layer.over(np.clip(bg + 0.5, 0, 255).astype(np.uint8)))
        if wr.count % 100 == 0:
            print(f"  {wr.count}/{total} frames")
    wr.close()
    print(f"wrote {a.out}: {count_frames(a.out)} frames (clip has {total if not a.seconds else 'n/a for test'})")


if __name__ == "__main__":
    main()
