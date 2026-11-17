"""Big word behind me + animated captions of what I say + punch-in zooms.

Background word (always):
  White Anton across the frame at head height, ~4% margin each side, letters
  rise into place one at a time (0.09 s apart, from ~1/3 of their height
  below, fade in with a little overshoot), soft black shadow under them, room
  dimmed to 78%, word grows 3% over the clip. I'm put back on top with
  put_back.py, so my head covers the middle letters.

Captions (when a plan file is given; see make_plan / IMG_0811_plan.json):
  Same style (white Anton, soft black shadow), drawn in front, never over my
  face. Beat types:
    intro - words slide in from the left one by one, first word biggest,
            each next word smaller and further toward the centre
    line  - small lowercase caption, words rise in exactly when spoken
    hit   - a stressed word (TITAN, TYPE-C, 128 GB ...) pops in big,
            optional small sub-word under it
    hero  - the product name, biggest, letter-by-letter rise
  Zoom keyframes [[t, zoom], ...] make smooth push-ins and quick punches
  around my face; captions are drawn after the zoom so they stay sharp.

usage:
  .venv/bin/python text_behind.py [clip] --word "IPHONE 15 PRO"            # word only
  .venv/bin/python text_behind.py [clip] --plan IMG_0811_plan.json         # word + captions + zoom
  add --seconds N for a quick test of the first N seconds
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import HERE, VideoWriter, count_frames, find_clip, probe, stem
from put_back import PutBack, over

FONT = os.path.join(HERE, "fonts", "Anton-Regular.ttf")
SS = 2                 # supersampling for sprites

# background word
MARGIN = 0.04
STAGGER = 0.09
RISE_TIME = 0.5
RISE_FROM = 1 / 3
GROW = 0.03
DIM = 0.78

# captions (fractions of frame height unless noted)
LINE_SIZE = 0.056      # font size of normal caption words
LINE_Y = 0.79          # vertical centre of normal captions
HIT_SIZE = 0.155
HIT_Y = 0.71
HERO_Y = 0.71
INTRO_SIZE = 0.125
INTRO_BASE = 0.50      # baseline of the first intro word
INTRO_X = [0.05, 0.13, 0.22]   # fraction of width: each word starts further right (toward centre)
INTRO_SCALE = [1.0, 0.74, 0.56]
MAX_W = 0.88           # captions never wider than this fraction of the width
FACE_PAD = 0.05        # keep text this far (fraction of H) below/away from the face box


# ---------------------------------------------------------------- easing --
def ease_out_back(p, c1=1.1):
    p = np.clip(p, 0, 1)
    c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def ease_out_cubic(p):
    p = np.clip(p, 0, 1)
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    p = np.clip(p, 0, 1)
    return p ** 3


def smoother(p):
    p = np.clip(p, 0, 1)
    return p * p * p * (p * (p * 6 - 15) + 10)


# --------------------------------------------------------------- sprites --
_fonts, _sprites = {}, {}


def font(size):
    key = int(round(size * SS))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT, key)
    return _fonts[key]


class Sprite:
    """Text rendered once (supersampled) with its soft shadow.
    Positioned by its pen origin on the baseline; adv = advance width,
    cap = cap height (all in output pixels at scale 1)."""

    def __init__(self, text, size):
        f = font(size)
        x0, y0, x1, y1 = f.getbbox(text, anchor="ls")
        cap = -f.getbbox("H", anchor="ls")[1]
        pad = int(cap * 0.45) + 4
        w, h = int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad
        im = Image.new("L", (max(w, 1), max(h, 1)), 0)
        ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=f, fill=255, anchor="ls")
        self.alpha = np.asarray(im, np.float32) / 255.0
        sh = cv2.GaussianBlur(self.alpha, (0, 0), max(1.0, cap * 0.05))
        sh = cv2.warpAffine(sh, np.float32([[1, 0, 0], [0, 1, cap * 0.035]]), (im.width, im.height))
        self.shadow = np.clip(sh * 0.7, 0, 1)
        self.ox, self.oy = (x0 - pad) / SS, (y0 - pad) / SS
        self.adv = f.getlength(text) / SS
        self.cap = cap / SS


def sprite(text, size):
    key = (text, int(round(size * SS)))
    if key not in _sprites:
        _sprites[key] = Sprite(text, size)
    return _sprites[key]


class Draw:
    """One sprite placement: pen origin (x, baseline y), scale, opacity."""

    def __init__(self, sp, x, y, s=1.0, op=1.0):
        self.sp, self.x, self.y, self.s, self.op = sp, x, y, s, op

    def rect(self):
        sp, s = self.sp, self.s
        x0 = self.x + sp.ox * s
        y0 = self.y + sp.oy * s
        return x0, y0, x0 + sp.alpha.shape[1] / SS * s, y0 + sp.alpha.shape[0] / SS * s


def blit(text_c, shadow_c, d):
    """Warp a sprite into the canvases (only its bounding box)."""
    if d.op <= 0.003:
        return
    H, W = text_c.shape
    x0, y0, x1, y1 = d.rect()
    X0, Y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    X1, Y1 = min(W, int(np.ceil(x1)) + 1), min(H, int(np.ceil(y1)) + 1)
    if X1 <= X0 or Y1 <= Y0:
        return
    k = d.s / SS
    M = np.float32([[k, 0, x0 - X0], [0, k, y0 - Y0]])
    size = (X1 - X0, Y1 - Y0)
    a = cv2.warpAffine(d.sp.alpha, M, size, flags=cv2.INTER_LINEAR) * d.op
    sh = cv2.warpAffine(d.sp.shadow, M, size, flags=cv2.INTER_LINEAR) * d.op
    np.maximum(text_c[Y0:Y1, X0:X1], a, out=text_c[Y0:Y1, X0:X1])
    np.maximum(shadow_c[Y0:Y1, X0:X1], sh, out=shadow_c[Y0:Y1, X0:X1])


def apply_text(img, text_c, shadow_c):
    out = img.astype(np.float32)
    out *= (1 - shadow_c)[..., None]
    out = out * (1 - text_c)[..., None] + 255.0 * text_c[..., None]
    return out


# ------------------------------------------------------- background word --
def bg_letters(word, W):
    target = W * (1 - 2 * MARGIN)
    ref = ImageFont.truetype(FONT, 400)
    size = 400 * target / ref.getlength(word)
    f = font(size)
    left = f.getbbox(word, anchor="ls")[0] / SS
    out = []
    for k, ch in enumerate(word):
        if ch.isspace():
            continue
        pen = f.getlength(word[:k]) / SS - left
        out.append((sprite(ch, size), pen))
    return out, sprite("H", size).cap, target


def draw_bg_word(letters, cap, word_w, cx, cy, t, dur, W, H):
    text_c = np.zeros((H, W), np.float32)
    shadow_c = np.zeros((H, W), np.float32)
    s = 1 + GROW * min(1.0, t / dur)
    for k, (sp, pen) in enumerate(letters):
        p = (t - k * STAGGER) / RISE_TIME
        if p <= 0:
            continue
        rise = (1 - ease_out_back(p)) * cap * RISE_FROM
        op = ease_out_cubic(min(1.0, p / 0.6))
        x = cx + (pen - word_w / 2) * s
        y = cy + (cap / 2 + rise) * s
        blit(text_c, shadow_c, Draw(sp, x, y, s, op))
    return text_c, shadow_c


# --------------------------------------------------------------- captions --
def layout_words(words, size, W, cy):
    """Greedy-wrap words centred around cy. Returns [(sprite, x, baseline, t)]."""
    sps = [(sprite(w, size), t) for w, t in words]
    space = sprite("H", size).cap * 0.32
    lines, cur, cur_w = [], [], 0
    for sp, t in sps:
        add = sp.adv + (space if cur else 0)
        if cur and cur_w + add > MAX_W * W:
            lines.append((cur, cur_w))
            cur, cur_w = [], 0
            add = sp.adv
        cur.append((sp, t))
        cur_w += add
    if cur:
        lines.append((cur, cur_w))
    cap = sprite("H", size).cap
    gap = cap * 1.35
    top_base = cy + cap / 2 - gap * (len(lines) - 1) / 2
    out = []
    for li, (ln, lw) in enumerate(lines):
        x = W / 2 - lw / 2
        for sp, t in ln:
            out.append((sp, x, top_base + li * gap, t))
            x += sp.adv + space
    return out


def beat_draws(b, t, W, H):
    """Draw list for one beat at time t (empty when the beat isn't visible)."""
    end = b["end"]
    t0 = b.get("t", min(w[1] for w in b.get("words", [["", 0]])))
    if t < t0 - 0.01 or t > end + 0.4:
        return []
    out_p = (t - (end - 0.10)) / 0.14  # >0 while leaving; gone ~0.04 s after `end`
    leave_op = 1 - smoother(out_p) if out_p > 0 else 1.0
    draws = []
    kind = b["type"]

    if kind == "line":
        cap = sprite("H", LINE_SIZE * H).cap
        for sp, x, y, tw in layout_words(b["words"], LINE_SIZE * H, W, LINE_Y * H):
            p = (t - tw) / 0.32
            if p <= 0:
                continue
            rise = (1 - ease_out_back(p, 1.4)) * cap * 0.45
            op = ease_out_cubic(p / 0.6) * leave_op
            lift = cap * 0.25 * smoother(out_p) if out_p > 0 else 0
            draws.append(Draw(sp, x, y + rise - lift, 1.0, op))

    elif kind == "hit":
        size = HIT_SIZE * H * b.get("scale", 1.0)
        sp = sprite(b["text"], size)
        if sp.adv > MAX_W * W:
            size *= MAX_W * W / sp.adv
            sp = sprite(b["text"], size)
        p = (t - b["t"]) / 0.38
        s = (0.55 + 0.45 * ease_out_back(p, 1.8))
        s *= 1 + 0.035 * np.clip((t - b["t"]) / max(0.1, end - b["t"]), 0, 1)   # slow drift while held
        if out_p > 0:
            s *= 1 + 0.08 * smoother(out_p)
        op = ease_out_cubic(p / 0.45) * leave_op
        cy = HIT_Y * H
        draws.append(Draw(sp, W / 2 - sp.adv * s / 2, cy + sp.cap * s / 2, s, op))
        if b.get("sub"):
            sub_size = LINE_SIZE * H * 1.15
            ssp = sprite(b["sub"], sub_size)
            q = (t - b.get("sub_t", b["t"] + 0.15)) / 0.32
            if q > 0:
                rise = (1 - ease_out_back(q, 1.4)) * ssp.cap * 0.45
                y = cy + sp.cap / 2 + ssp.cap * 1.75 + rise
                draws.append(Draw(ssp, W / 2 - ssp.adv / 2, y, 1.0, ease_out_cubic(q / 0.6) * leave_op))

    elif kind == "hero":
        # Biggest thing on screen: split into two lines ("IPHONE" / "15 PRO"),
        # each as wide as allowed, letters rising in one after another.
        text = b["text"]
        parts = text.split()
        if len(parts) >= 2:
            cut = min(range(1, len(parts)), key=lambda c: abs(len(" ".join(parts[:c])) - len(" ".join(parts[c:]))))
            lines = [" ".join(parts[:cut]), " ".join(parts[cut:])]
        else:
            lines = [text]
        ref = font(100)
        size = min(100 * MAX_W * W / (ref.getlength(ln) / SS) for ln in lines)
        size = min(size, HIT_SIZE * H * 1.9)
        f = font(size)
        cap = sprite("H", size).cap
        gap = cap * 1.18
        s_all = 1.12 - 0.12 * ease_out_cubic((t - b["t"]) / 0.7)
        if out_p > 0:
            s_all *= 1 + 0.08 * smoother(out_p)
        cx, cy = W / 2, HERO_Y * H
        k_all = 0
        for li, ln in enumerate(lines):
            left = f.getbbox(ln, anchor="ls")[0] / SS
            width = f.getlength(ln) / SS
            base = cy + cap / 2 + (li - (len(lines) - 1) / 2) * gap
            for k, ch in enumerate(ln):
                if ch.isspace():
                    continue
                p = (t - b["t"] - k_all * 0.055) / 0.42
                k_all += 1
                if p <= 0:
                    continue
                pen = f.getlength(ln[:k]) / SS - left
                rise = (1 - ease_out_back(p, 1.6)) * cap * 0.5
                op = ease_out_cubic(p / 0.5) * leave_op
                x = cx + (pen - width / 2) * s_all
                y = cy + (base - cy + rise) * s_all
                draws.append(Draw(sprite(ch, size), x, y, s_all, op))

    elif kind == "intro":
        base = INTRO_BASE * H
        prev_cap = 0
        for k, (word, tw) in enumerate(b["words"]):
            size = INTRO_SIZE * H * INTRO_SCALE[min(k, len(INTRO_SCALE) - 1)]
            sp = sprite(word, size)
            if k > 0:
                base += prev_cap * 0.22 + sp.cap * 1.12
            prev_cap = sp.cap
            p = (t - tw) / 0.42
            if p <= 0:
                continue
            target = INTRO_X[min(k, len(INTRO_X) - 1)] * W
            start = -sp.adv - 30
            x = start + (target - start) * ease_out_back(p, 1.3)
            op = ease_out_cubic(p / 0.4)
            lp = (t - end - k * 0.06) / 0.28
            if lp > 0:
                x -= (target + sp.adv + 40) * ease_in_cubic(lp)
                op *= 1 - smoother(lp)
            draws.append(Draw(sp, x, base, 1.0, op))
    return draws


def avoid_face(draws, face, pad):
    """Shift a beat's draws down until they clear the face box (screen space)."""
    if not draws or face is None:
        return draws
    fx0, fy0, fx1, fy1 = face
    rects = [d.rect() for d in draws]
    x0 = min(r[0] for r in rects)
    x1 = max(r[2] for r in rects)
    y0 = min(r[1] for r in rects)
    y1 = max(r[3] for r in rects)
    if x1 < fx0 - pad or x0 > fx1 + pad or y1 < fy0 - pad or y0 > fy1 + pad:
        return draws
    dy = (fy1 + pad) - y0
    for d in draws:
        d.y += dy
    return draws


# ------------------------------------------------------------------- zoom --
def zoom_at(keys, t):
    if not keys:
        return 1.0
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, za), (tb, zb) in zip(keys, keys[1:]):
        if ta <= t <= tb:
            return za + (zb - za) * smoother((t - ta) / max(1e-6, tb - ta))
    return keys[-1][1]


def smooth_series(x, sigma):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--word", help="background word (default: from plan, else CLAUDE)")
    ap.add_argument("--plan", help="captions + zoom plan json")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--start", type=float, help="render from here (tests only; final render starts at 0)")
    ap.add_argument("--out", default=os.path.join(HERE, "text-behind.mp4"))
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    W, H, fps = info["width"], info["height"], float(info["fps"])
    total = count_frames(clip)
    dur = total / fps
    plan = json.load(open(a.plan)) if a.plan else {}
    word = (a.word or plan.get("word") or "CLAUDE").upper()
    beats = plan.get("beats", [])
    zkeys = plan.get("zoom", [])

    face = json.load(open(stem(clip) + "_face.json"))["frames"]
    det = [f for f in face if f["detected"]]
    if not det:
        raise SystemExit("no face found - can't place the word at head height")
    head_y = float(np.median([f["y"] + f["h"] / 2 for f in det])) * H
    fx = np.array([(f["x"] + f["w"] / 2) if f["x"] is not None else 0.5 for f in face]) * W
    fy = np.array([(f["y"] + f["h"] / 2) if f["y"] is not None else head_y / H for f in face]) * H
    # Zoom anchor: a little below the face, smoothed hard so the "camera" glides.
    ax = smooth_series(fx, 0.6 * fps)
    ay = smooth_series(fy + 0.10 * H, 0.6 * fps)

    letters, cap, word_w = bg_letters(word, W)
    print(f"word {word}: {word_w:.0f}px wide, cap {cap:.0f}px, centre y {head_y:.0f}px; "
          f"{len(beats)} caption beats, {len(zkeys)} zoom keys")

    wr = VideoWriter(a.out, W, H, info["rate"], audio_from=clip, seconds=a.seconds, start=a.start)
    for layer in PutBack(clip, seconds=a.seconds, start=a.start):
        t, i = layer.t, layer.i
        # 1) zoom the room and me together (scaling about an inside point keeps
        #    the crop inside the frame); the background word is NOT zoomed, so it
        #    always stays whole across the frame
        z = zoom_at(zkeys, t)
        cx, cy = float(ax[min(i, len(ax) - 1)]), float(ay[min(i, len(ay) - 1)])
        room, alpha, fg = layer.frame, layer.alpha, layer.fg
        if z > 1.0005:
            M = np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]])
            warp = lambda im, fl=cv2.INTER_CUBIC: cv2.warpAffine(im, M, (W, H), flags=fl, borderMode=cv2.BORDER_REFLECT)
            room, fg = warp(room), warp(fg)
            alpha = np.clip(warp(alpha, cv2.INTER_LINEAR), 0, 1)
        # 2) dimmed room + background word, then me on top
        tc, sc = draw_bg_word(letters, cap, word_w, W / 2, head_y, t, dur, W, H)
        bg = apply_text(room.astype(np.float32) * DIM, tc, sc)
        frame = over(fg, alpha, np.clip(bg + 0.5, 0, 255).astype(np.uint8))
        # 3) captions in screen space, kept off the face
        if beats:
            f = face[min(i, len(face) - 1)]
            fbox = None
            if f["x"] is not None:
                bx0, by0 = f["x"] * W, f["y"] * H
                bx1, by1 = bx0 + f["w"] * W, by0 + f["h"] * H
                fbox = (cx + (bx0 - cx) * z, cy + (by0 - cy) * z, cx + (bx1 - cx) * z, cy + (by1 - cy) * z)
            text_c = np.zeros((H, W), np.float32)
            shadow_c = np.zeros((H, W), np.float32)
            for b in beats:
                for d in avoid_face(beat_draws(b, t, W, H), fbox, FACE_PAD * H):
                    blit(text_c, shadow_c, d)
            if text_c.any() or shadow_c.any():
                frame = apply_text(frame, text_c, shadow_c)
        if frame.dtype != np.uint8:
            frame = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        wr.write(frame)
        if wr.count % 100 == 0:
            print(f"  {wr.count}/{total} frames")
    wr.close()
    n = count_frames(a.out)
    print(f"wrote {a.out}: {n} frames" + ("" if a.seconds else f" (clip has {total})"))


if __name__ == "__main__":
    main()
