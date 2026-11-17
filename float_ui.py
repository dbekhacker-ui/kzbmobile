"""Frosted-glass UI windows floating beside me + small captions + zooms + sound.

Windows (rounded panels: dark see-through fill over a blur of what is behind
them, thin light border, tilted a few degrees):
  profile   KZB Instagram-style profile window at head height on the left,
            BEHIND me, tucked so my head covers its right edge
  terminal  small terminal lower right, IN FRONT of me; it types the phone's
            specs at the moment I say them
  pill      notification pill lower left, IN FRONT of me
They pop in one after another with a little bounce, then bob gently, and leave
before the end. None of them (and no caption) ever touches my face box from
face.py: every frame each panel's real (rotated) outline is checked against
the face box and pushed away if needed.

Then the small captions of what I say (same style as text_behind.py), punch-in
zooms, a minimal end question, and sound effects under my voice.

Cuts: "cuts" in the plan drops whole frames (picture and sound together), with
a 15 ms audio crossfade and a zoom jump on each cut ("cut_zoom") so it reads as an edit.

Output (from <name>_plan.json):
  float-ui.mp4   my voice cleaned up by voice.py (no AI) + sound effects

usage: .venv/bin/python float_ui.py [clip] --plan 13_plan.json [--start S --seconds N]
"""
import argparse
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sfx
import text_behind as tb
from common import HERE, VideoWriter, count_frames, find_clip, probe, stem
from put_back import PutBack, over

FONTS = os.path.join(HERE, "fonts")
SS = 2                      # panels are drawn at 2x and scaled down (crisp edges)

# small captions (fractions of frame height) - same look as text_behind, just smaller
tb.LINE_SIZE = 0.034
tb.LINE_Y = 0.735
tb.HIT_SIZE = 0.066
tb.HIT_Y = 0.727
tb.HERO_Y = 0.727
tb.MAX_W = 0.80

GLASS_TINT = np.array([14, 16, 24], np.float32)
GLASS_TINT_A = 0.40
FACE_PAD = 14               # px between any panel and the face box


def F(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), int(round(size * SS)))


def ease_out_back(p, c1=1.7):
    return tb.ease_out_back(p, c1)


# ---------------------------------------------------------------- panels --
class Panel:
    """A rounded glass panel drawn in its own upright space (w x h design px)."""

    def __init__(self, w, h, radius, tilt, t_in, t_out, bob_amp, bob_period, phase, front):
        self.w, self.h, self.r = w, h, radius
        self.tilt, self.t_in, self.t_out = tilt, t_in, t_out
        self.bob_amp, self.bob_period, self.phase = bob_amp, bob_period, phase
        self.front = front
        W2, H2 = int(w * SS), int(h * SS)
        shape = Image.new("L", (W2, H2), 0)
        ImageDraw.Draw(shape).rounded_rectangle((0, 0, W2 - 1, H2 - 1), radius * SS, fill=255)
        self.shape = np.asarray(shape, np.float32) / 255
        border = Image.new("L", (W2, H2), 0)
        ImageDraw.Draw(border).rounded_rectangle((1, 1, W2 - 2, H2 - 2), radius * SS, outline=255,
                                                 width=max(2, int(1.4 * SS)))
        self.border = np.asarray(border, np.float32) / 255 * 0.42
        self._content_cache = {}

    def content(self, t):
        """RGBA uint8 (h*SS, w*SS, 4); subclasses draw their contents."""
        return np.zeros((int(self.h * SS), int(self.w * SS), 4), np.uint8)

    def anim(self, t):
        """(visible, scale, opacity, dy, dangle)."""
        if t < self.t_in:
            return False, 0, 0, 0, 0
        p = (t - self.t_in) / 0.5
        s = 0.62 + 0.38 * ease_out_back(min(p, 1.0), 1.9)
        op = tb.ease_out_cubic(min(1.0, p / 0.35))
        settle = tb.smoother((t - self.t_in) / 0.9)
        ph = 2 * np.pi * (t - self.t_in) / self.bob_period + self.phase
        dy = self.bob_amp * np.sin(ph) * settle
        da = 0.35 * np.sin(ph * 0.8 + 1.0) * settle
        if self.t_out is not None and t > self.t_out:
            q = (t - self.t_out) / 0.24
            if q >= 1:
                return False, 0, 0, 0, 0
            s *= 1 - 0.3 * tb.ease_in_cubic(q)
            op *= 1 - tb.smoother(q)
        return True, s, op, dy, da


def panel_matrix(p, cx, cy, s, angle):
    """Affine from panel sprite pixels (SS) to screen, centred at (cx, cy)."""
    th = np.deg2rad(angle)
    c, si = np.cos(th) * s / SS, np.sin(th) * s / SS
    W2, H2 = p.w * SS, p.h * SS
    return np.float32([[c, -si, cx - c * W2 / 2 + si * H2 / 2],
                       [si, c, cy - si * W2 / 2 - c * H2 / 2]])


def outline(p, M):
    W2, H2 = p.w * SS, p.h * SS
    pts = np.float32([[0, 0], [W2, 0], [W2, H2], [0, H2]])
    return pts @ M[:, :2].T + M[:, 2]


def box_hits(pts, face, pad=FACE_PAD):
    """Does the panel's real (rotated) outline touch the padded face box?
    Separating-axis test on the two convex shapes."""
    if face is None:
        return False
    fx0, fy0, fx1, fy1 = face
    box = np.float32([[fx0 - pad, fy0 - pad], [fx1 + pad, fy0 - pad], [fx1 + pad, fy1 + pad], [fx0 - pad, fy1 + pad]])
    for poly in (pts, box):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.float32([-e[1], e[0]])
            a, b = pts @ n, box @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def draw_glass(img, p, M, op, blur_src, t, shadow=True):
    """Composite one glass panel onto img (float32 HxWx3) in place."""
    H, W = img.shape[:2]
    pts = outline(p, M)
    m = 40
    X0, Y0 = int(max(0, pts[:, 0].min() - m)), int(max(0, pts[:, 1].min() - m))
    X1, Y1 = int(min(W, pts[:, 0].max() + m)), int(min(H, pts[:, 1].max() + m))
    if X1 <= X0 or Y1 <= Y0:
        return
    Mr = M.copy()
    Mr[:, 2] -= (X0, Y0)
    size = (X1 - X0, Y1 - Y0)
    a = cv2.warpAffine(p.shape, Mr, size, flags=cv2.INTER_LINEAR) * op
    roi = img[Y0:Y1, X0:X1]
    if shadow:
        sh = cv2.GaussianBlur(cv2.warpAffine(p.shape, Mr + np.float32([[0, 0, 0], [0, 0, 12]]), size), (0, 0), 16)
        roi *= (1 - 0.30 * sh * op)[..., None]
    frost = blur_src(X0, Y0, X1, Y1)
    glass = frost * (1 - GLASS_TINT_A) + GLASS_TINT * GLASS_TINT_A
    roi[:] = roi * (1 - a[..., None]) + glass * a[..., None]
    b = cv2.warpAffine(p.border, Mr, size, flags=cv2.INTER_LINEAR) * op
    roi[:] = roi * (1 - b[..., None]) + 255.0 * b[..., None]
    c = p.content(t)
    if c is not None:
        cw = cv2.warpAffine(c, Mr, size, flags=cv2.INTER_LINEAR).astype(np.float32)
        ca = cw[..., 3:4] / 255 * op
        roi[:] = roi * (1 - ca) + cw[..., :3] * ca


def blur_room_only(room, alpha, sigma):
    """Blur of the room with me cut out (no dark smear of my head in the glass)."""
    def f(x0, y0, x1, y1):
        m = int(3 * sigma)
        X0, Y0 = max(0, x0 - m), max(0, y0 - m)
        X1, Y1 = min(room.shape[1], x1 + m), min(room.shape[0], y1 + m)
        w = (1 - alpha[Y0:Y1, X0:X1]).astype(np.float32)
        num = cv2.GaussianBlur(room[Y0:Y1, X0:X1].astype(np.float32) * w[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
        out = num / np.maximum(den, 1e-3)
        # where there's no room at all nearby, fall back to the plain blur
        plain = cv2.GaussianBlur(room[Y0:Y1, X0:X1].astype(np.float32), (0, 0), sigma)
        k = np.clip(den / 0.15, 0, 1)
        out = out * k + plain * (1 - k)
        return out[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    return f


def blur_plain(img, sigma):
    def f(x0, y0, x1, y1):
        m = int(3 * sigma)
        X0, Y0 = max(0, x0 - m), max(0, y0 - m)
        X1, Y1 = min(img.shape[1], x1 + m), min(img.shape[0], y1 + m)
        out = cv2.GaussianBlur(img[Y0:Y1, X0:X1].astype(np.float32), (0, 0), sigma)
        return out[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    return f


def rgba_to_np(im):
    return np.asarray(im, np.uint8)


class ProfilePanel(Panel):
    """KZB profile window: the real round logo in a story ring, name, buttons,
    and one row of 3 real reel tiles from the Instagram profile (with their
    view counts) that appear one by one: soft fade, a little scale, blur to sharp."""
    GX, GY, GAP = 22, 264, 6
    REVEAL = 0.65

    def __init__(self, cfg, t_out):
        tw = (440 - 2 * self.GX - 2 * self.GAP) / 3
        th = tw * 4 / 3
        h = int(self.GY + th + 22)
        super().__init__(440, h, 30, -3.0, cfg["t_in"], t_out, 6, 3.4, 0.0, front=False)
        S = SS
        im = Image.new("RGBA", (440 * S, h * S), (0, 0, 0, 0))
        grad = np.linspace(22, 0, 140 * S)[:, None] * np.ones((1, 440 * S))
        top = np.zeros((140 * S, 440 * S, 4), np.uint8)
        top[..., :3] = 255
        top[..., 3] = grad.astype(np.uint8)
        im.alpha_composite(Image.fromarray(top))
        d = ImageDraw.Draw(im)
        d.text((24 * S, 20 * S), cfg["name"], font=F("Inter-Bold.otf", 22), fill=(255, 255, 255, 255))
        for k in range(3):
            d.ellipse(((392 + k * 10) * S, 32 * S, (396 + k * 10) * S, 36 * S), fill=(255, 255, 255, 220))
        d.line((0, 64 * S, 440 * S, 64 * S), fill=(255, 255, 255, 40), width=S)
        # avatar: the real KZB logo inside a story-style gradient ring
        ax, ay, ar = 72, 128, 48
        D = 2 * ar * S
        g = np.zeros((D, D, 4), np.uint8)
        yy, xx = np.mgrid[0:D, 0:D]
        tpos = ((xx + (D - yy)) / (2 * D))[..., None]
        c0, c1, c2 = np.array([254, 218, 117]), np.array([214, 41, 118]), np.array([150, 47, 191])
        g[..., :3] = np.where(tpos < 0.5, c0 + (c1 - c0) * (tpos / 0.5), c1 + (c2 - c1) * ((tpos - 0.5) / 0.5)).astype(np.uint8)
        ring = Image.new("L", (D * 4, D * 4), 0)
        ImageDraw.Draw(ring).ellipse((0, 0, D * 4 - 1, D * 4 - 1), fill=255)
        ImageDraw.Draw(ring).ellipse((5 * S * 4, 5 * S * 4, D * 4 - 5 * S * 4, D * 4 - 5 * S * 4), fill=0)
        g[..., 3] = np.asarray(ring.resize((D, D), Image.LANCZOS))
        im.alpha_composite(Image.fromarray(g), ((ax - ar) * S, (ay - ar) * S))
        inner = D - 2 * 8 * S
        logo = Image.open(os.path.join(HERE, cfg["logo"])).convert("RGBA").resize((inner, inner), Image.LANCZOS)
        im.alpha_composite(logo, ((ax - ar + 8) * S, (ay - ar + 8) * S))
        d = ImageDraw.Draw(im)
        d.text((142 * S, 104 * S), cfg["name"], font=F("Inter-SemiBold.otf", 25), fill=(255, 255, 255, 255))
        d.text((142 * S, 140 * S), cfg["bio"], font=F("Inter-Regular.otf", 18), fill=(205, 208, 215, 255))
        bw, bh, by = 190, 46, 196
        d.rounded_rectangle((22 * S, by * S, (22 + bw) * S, (by + bh) * S), 12 * S, fill=(0, 149, 246, 255))
        d.text(((22 + bw / 2) * S, (by + bh / 2) * S), cfg["follow"], font=F("Inter-SemiBold.otf", 18),
               fill=(255, 255, 255, 255), anchor="mm")
        d.rounded_rectangle((228 * S, by * S, (228 + bw) * S, (by + bh) * S), 12 * S, fill=(255, 255, 255, 36))
        d.text(((228 + bw / 2) * S, (by + bh / 2) * S), cfg["message"], font=F("Inter-SemiBold.otf", 18),
               fill=(255, 255, 255, 255), anchor="mm")
        # empty slots (shown until each reel appears)
        self.slots = []
        for k in range(3):
            x0 = self.GX + k * (tw + self.GAP)
            self.slots.append((x0, self.GY, tw, th))
            d.rounded_rectangle((x0 * S, self.GY * S, (x0 + tw) * S, (self.GY + th) * S), 8 * S, fill=(255, 255, 255, 18))
        self.base = im
        # the real tiles, rounded, at slot size
        self.tiles = []
        for k, tcfg in enumerate(cfg["tiles"][:3]):
            src = Image.open(os.path.join(HERE, tcfg["img"])).convert("RGB")
            tile = src.resize((int(tw * S), int(th * S)), Image.LANCZOS).convert("RGBA")
            m = Image.new("L", tile.size, 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, tile.size[0] - 1, tile.size[1] - 1), 8 * S, fill=255)
            tile.putalpha(m)
            self.tiles.append((tile, float(tcfg["t"])))

    def reveal_times(self):
        return [t for _, t in self.tiles]

    def content(self, t):
        prog = tuple(int(round(np.clip((t - tt) / self.REVEAL, 0, 1) * 30)) for _, tt in self.tiles)
        if prog in self._content_cache:
            return self._content_cache[prog]
        from PIL import ImageFilter
        im = self.base.copy()
        S = SS
        for (tile, _), q, (x0, y0, tw, th) in zip(self.tiles, prog, self.slots):
            if q == 0:
                continue
            e = float(tb.ease_out_cubic(q / 30))
            sc = 0.93 + 0.07 * e
            w2, h2 = max(2, int(tile.size[0] * sc)), max(2, int(tile.size[1] * sc))
            tt = tile.resize((w2, h2), Image.LANCZOS)
            if e < 0.999:
                tt = tt.filter(ImageFilter.GaussianBlur(float((1 - e) * 5 * S)))
                al = np.asarray(tt.getchannel("A"), np.float32) * e
                tt.putalpha(Image.fromarray(al.astype(np.uint8)))
            cx, cy = (x0 + tw / 2) * S, (y0 + th / 2) * S
            im.alpha_composite(tt, (int(cx - w2 / 2), int(cy - h2 / 2)))
        arr = rgba_to_np(im)
        self._content_cache[prog] = arr
        return arr


class TerminalPanel(Panel):
    CPS = 40   # typed characters per second

    def __init__(self, cfg, t_out):
        lines = cfg["lines"]
        h = 56 + 30 * len(lines) + 22
        super().__init__(380, h, 18, 4.0, cfg["t_in"], t_out, 7, 2.9, 1.7, front=True)
        self.lines = lines
        self.title = cfg["title"]
        self.mono = F("DejaVuSansMono.ttf", 18)
        self.mono_b = F("DejaVuSansMono-Bold.ttf", 18)

    def typed(self, t):
        out = []
        for item in self.lines:
            text, t0 = item[0], item[1]
            if text.startswith("@progress"):
                # a live progress bar: ["@progress label", t0, t1]
                if t >= t0:
                    t1 = item[2]
                    p = float(np.clip((t - t0) / (t1 - t0), 0, 1))
                    p = 1 - (1 - p) ** 1.6
                    n = int(round(p * 12))
                    out.append(f"{text.split(' ', 1)[1]:<9}" + "█" * n + "░" * (12 - n) + f" {int(p * 100):3d}%")
                continue
            n = int(max(0, (t - t0)) * self.CPS)
            if t >= t0:
                out.append(text[:max(1, n)] if n < len(text) else text)
        return tuple(out)

    def char_times(self):
        """Times of every typed character (for typing sounds)."""
        return [it[1] + k / self.CPS for it in self.lines if not it[0].startswith("@")
                for k in range(len(it[0])) if it[0][k] != " "]

    def content(self, t):
        typed = self.typed(t)
        cursor = int(t * 2.2) % 2 == 0
        key = (typed, cursor)
        if key in self._content_cache:
            return self._content_cache[key]
        S = SS
        im = Image.new("RGBA", (int(self.w * S), int(self.h * S)), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for k, col in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            cx = 22 + k * 20
            d.ellipse(((cx - 6) * S, 14 * S, (cx + 6) * S, 26 * S), fill=col + (230,))
        d.text((self.w / 2 * S, 20 * S), self.title, font=F("Inter-Regular.otf", 15), fill=(200, 203, 210, 255), anchor="mm")
        d.line((0, 40 * S, self.w * S, 40 * S), fill=(255, 255, 255, 34), width=S)
        y = 54
        end_xy = (18, 54)
        for line in typed:
            x = 18
            if line.startswith("$"):
                d.text((x * S, y * S), "$", font=self.mono_b, fill=(90, 225, 150, 255))
                d.text(((x + 20) * S, y * S), line[2:], font=self.mono, fill=(235, 238, 245, 255))
                w = 20 + self.mono.getlength(line[2:]) / S
            else:
                key_, val = line[:9], line[9:]
                d.text((x * S, y * S), key_, font=self.mono, fill=(140, 148, 165, 255))
                kw = self.mono.getlength(key_) / S
                good = "✓" in val or "100%" in val
                d.text(((x + kw) * S, y * S), val, font=self.mono_b, fill=(120, 230, 160, 255) if good else (255, 255, 255, 255))
                w = kw + self.mono_b.getlength(val) / S
            end_xy = (x + w + 4, y)
            y += 30
        if cursor:
            cx, cy = end_xy
            d.rectangle((cx * S, (cy + 2) * S, (cx + 10) * S, (cy + 22) * S), fill=(235, 238, 245, 200))
        arr = rgba_to_np(im)
        if len(self._content_cache) > 400:
            self._content_cache.clear()
        self._content_cache[key] = arr
        return arr


class PillPanel(Panel):
    def __init__(self, cfg, t_out):
        main_f = F("Inter-SemiBold.otf", 27)
        tw = main_f.getlength(cfg["text"]) / SS
        w = int(18 + 58 + 16 + max(tw, 200) + 30)
        super().__init__(w, 92, 46, -2.5, cfg["t_in"], t_out, 5, 3.1, 3.3, front=True)
        self.pulses = cfg.get("pulse", [])
        S = SS
        im = Image.new("RGBA", (w * S, 92 * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        # app icon: rounded square, blue->violet gradient, "K"
        ic = np.zeros((58 * S, 58 * S, 4), np.uint8)
        yy, xx = np.mgrid[0:58 * S, 0:58 * S]
        tpos = ((xx + yy) / (116 * S))[..., None]
        ic[..., :3] = (np.array([40, 120, 255]) * (1 - tpos) + np.array([150, 70, 255]) * tpos).astype(np.uint8)
        m = Image.new("L", (58 * S, 58 * S), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, 58 * S - 1, 58 * S - 1), 15 * S, fill=255)
        ic[..., 3] = np.asarray(m)
        im.alpha_composite(Image.fromarray(ic), (18 * S, 17 * S))
        d = ImageDraw.Draw(im)
        d.text(((18 + 29) * S, (17 + 29) * S), "K", font=F("Inter-Bold.otf", 30), fill=(255, 255, 255, 255), anchor="mm")
        d.text((92 * S, 18 * S), cfg["label"], font=F("Inter-Regular.otf", 16), fill=(200, 204, 212, 255))
        d.text((92 * S, 40 * S), cfg["text"], font=main_f, fill=(255, 255, 255, 255))
        self.static = rgba_to_np(im)

    def content(self, t):
        return self.static

    def pulse(self, t):
        s = 1.0
        for p in self.pulses:
            q = (t - p) / 0.35
            if 0 <= q <= 1:
                s += 0.06 * np.sin(np.pi * q) * (1 - q * 0.3)
        return s


class QuestionCard(Panel):
    def __init__(self, cfg, W):
        f1, f2 = F("Inter-SemiBold.otf", 40), F("Inter-Regular.otf", 25)
        w = int(min(W * 0.86, max(f1.getlength(cfg["text"]), f2.getlength(cfg["sub"])) / SS + 64))
        super().__init__(w, 132, 28, 0.0, cfg["t"], None, 3, 3.6, 0.5, front=True)
        S = SS
        im = Image.new("RGBA", (w * S, 132 * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((w / 2 * S, 52 * S), cfg["text"], font=f1, fill=(255, 255, 255, 255), anchor="mm")
        d.text((w / 2 * S, 95 * S), cfg["sub"], font=f2, fill=(205, 208, 215, 255), anchor="mm")
        self.static = rgba_to_np(im)

    def anim(self, t):
        if t < self.t_in:
            return False, 0, 0, 0, 0
        p = (t - self.t_in) / 0.45
        s = 0.94 + 0.06 * tb.ease_out_cubic(min(1.0, p))
        op = tb.ease_out_cubic(min(1.0, p / 0.8))
        return True, s, op, 0.0, 0.0

    def content(self, t):
        return self.static


# ------------------------------------------------------------- layout --
def place(panel, t, face, W, H, mode, anchor):
    """Screen matrix for a panel this frame, kept off the face box."""
    vis, s, op, dy, da = panel.anim(t)
    if not vis:
        return None, 0
    if isinstance(panel, PillPanel):
        s *= panel.pulse(t)
    ang = panel.tilt + da
    if mode == "profile":
        # Up-left of my face: its top-right corner goes behind my cap/head (above
        # the face box), so my head covers that edge; it never touches the face box.
        fx0, fy0, fw = anchor
        bottom = fy0 + 0.06 * fw               # bottom edge ~ level with the top of the face box
        right = fx0 + 0.42 * fw                # right edge reaches behind the cap
        fit = min(1.0, max(0.5, (bottom - 40) / (panel.h * 1.04)), (right - 24) / (panel.w * 1.04))
        s *= fit
        cx = right - panel.w * s / 2
        cy = bottom - panel.h * s / 2 + dy
    else:
        cx, cy = anchor[0], anchor[1] + dy
    M = panel_matrix(panel, cx, cy, s, ang)
    # never touch the face: push away (left for the profile, down for the rest)
    pad = 4 if mode == "profile" else FACE_PAD
    for k in range(80):
        pts = outline(panel, M)
        if not box_hits(pts, face, pad):
            break
        if mode == "profile":   # up first, then left if it runs out of room at the top
            M[:, 2] += (0, -4) if pts[:, 1].min() > 12 else (-4, 0)
        else:
            M[:, 2] += (0, 6)
    # keep inside the frame
    pts = outline(panel, M)
    M[0, 2] += max(0, 8 - pts[:, 0].min()) - max(0, pts[:, 0].max() - (W - 8))
    M[1, 2] += max(0, 8 - pts[:, 1].min()) - max(0, pts[:, 1].max() - (H - 8))
    return M, op


# ------------------------------------------------------------------ audio --
def read_pcm(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(sfx.SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def sfx_events(plan, term):
    w = plan["windows"]
    ev = [("hook", plan["hook_t"], -5.0)]
    ev += [("pop", w["pill"]["t_in"], -9.0, -0.4), ("ding", w["pill"]["t_in"] + 0.06, -12.0, -0.4)]
    ev += [("pop", w["profile"]["t_in"], -10.0, -0.5)]
    ev += [("pop", t["t"], -16.0, -0.5) for t in w["profile"].get("tiles", [])[1:]]
    ev += [("pop", w["terminal"]["t_in"], -10.0, 0.5)]
    ev += [("tick", t, -21.0, 0.45) for t in term.char_times()]
    keys = plan.get("zoom", [])
    for b in plan["beats"]:
        if b["type"] == "hit":
            ev.append(("pop", b["t"], -10.0))
            # a soft swish into the zoom punch that goes with the hit
            if any(abs(k[0] - b["t"]) < 0.35 for k in keys):
                ev.append(("whoosh", b["t"] - 0.02, -16.0))
    ev.append(("whoosh", w["t_out"] + 0.1, -17.0))
    ev.append(("chime", plan["question"]["t"], -7.0))
    return ev


def mix_and_mux(video_only, voice, sfx_track, out, limit=True):
    mix = voice + sfx_track
    peak = float(np.abs(mix).max())
    if peak > 0.97:   # soft safety only on the summed peaks
        mix = np.tanh(mix / 0.97) * 0.97 if limit else mix / peak * 0.97
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as fh:
        fh.write(mix.astype("<f4").tobytes())
        raw = fh.name
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_only,
                        "-f", "f32le", "-ar", str(sfx.SR), "-ac", "2", "-i", raw,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-movflags", "+faststart", out], check=True)
    finally:
        os.unlink(raw)


# ------------------------------------------------------------- cuts --
def cut_frames(plan, fps, total):
    """Source frames to drop (whole frames, so picture and sound stay in sync)."""
    ranges = []
    for a0, a1 in plan.get("cuts", []):
        r0, r1 = int(round(a0 * fps)), int(round(a1 * fps))
        if r1 > r0:
            ranges.append((r0, min(r1, total)))
    removed = set(i for r0, r1 in ranges for i in range(r0, r1))
    keep = [i for i in range(total) if i not in removed]
    return ranges, keep


def make_s2o(ranges, fps):
    secs = [(r0 / fps, r1 / fps) for r0, r1 in ranges]

    def s2o(t):
        return t - sum(min(max(t - a0, 0.0), a1 - a0) for a0, a1 in secs)
    return s2o


def remap_plan(plan, s2o, ranges, fps):
    """Plan times are written in the clip's own time; move them to the cut timeline,
    and add a zoom jump on every cut so the jump cut reads as an intended edit."""
    import copy
    p = copy.deepcopy(plan)
    p["hook_t"] = s2o(p["hook_t"])
    w = p["windows"]
    for k in ("pill", "profile", "terminal"):
        w[k]["t_in"] = s2o(w[k]["t_in"])
    w["t_out"] = s2o(w["t_out"])
    w["pill"]["pulse"] = [s2o(t) for t in w["pill"].get("pulse", [])]
    for tile in w["profile"].get("tiles", []):
        tile["t"] = s2o(tile["t"])
    w["terminal"]["lines"] = [[ln[0]] + [s2o(x) for x in ln[1:]] for ln in w["terminal"]["lines"]]
    for bt in p["beats"]:
        for k in ("t", "sub_t", "end"):
            if k in bt:
                bt[k] = s2o(bt[k])
        if "words" in bt:
            bt["words"] = [[wd, s2o(t)] for wd, t in bt["words"]]
    if p.get("question"):
        p["question"]["t"] = s2o(p["question"]["t"])
    keys = sorted([[s2o(t), z] for t, z in p["zoom"]])
    for (r0, r1), (za, zb) in zip(ranges, p.get("cut_zoom", [])):
        tc = s2o(r1 / fps)                     # first frame after the cut
        zprev = tb.zoom_at(keys, tc - 1 / fps)
        keys = [k for k in keys if not (tc - 0.06 <= k[0] <= tc + 1.35)]
        keys += [[tc - 0.002, zprev], [tc, za], [tc + 1.3, zb]]
        keys.sort()
    p["zoom"] = keys
    return p


def cut_audio(x, ranges, spf, n_out, xfade=0.015):
    """Drop the cut frames' samples; overlap-add a short equal-power crossfade at
    every join (using samples from just inside the cut), so there is no click and
    the length stays exactly n_out = kept frames x samples per frame."""
    need = (max((r1 for _, r1 in ranges), default=0) + 1) * spf
    if len(x) < need:
        x = np.pad(x, ((0, need - len(x)), (0, 0)))
    pieces, joins, prev = [], [], 0
    for r0, r1 in ranges:
        pieces.append(x[prev * spf:r0 * spf])
        joins.append((sum(len(q) for q in pieces), r0 * spf, r1 * spf))
        prev = r1
    pieces.append(x[prev * spf:])
    y = np.concatenate(pieces)
    h = int(xfade * sfx.SR / 2)
    th = np.linspace(0, np.pi / 2, 2 * h)[:, None]
    fo, fi = np.cos(th), np.sin(th)
    for j, a_end, b_start in joins:
        if j - h < 0 or j + h > len(y):
            continue
        y[j - h:j + h] = x[a_end - h:a_end + h] * fo + x[b_start - h:b_start + h] * fi
    if len(y) < n_out:
        y = np.pad(y, ((0, n_out - len(y)), (0, 0)))
    return y[:n_out]


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--start", type=float, help="tests: start here (clip time)")
    ap.add_argument("--seconds", type=float, help="tests: this many seconds of the clip")
    ap.add_argument("--out", default=os.path.join(HERE, "float-ui.mp4"))
    ap.add_argument("--video-only", action="store_true", help="tests: skip the audio versions")
    ap.add_argument("--audio-only", action="store_true",
                    help="reuse an already rendered <out>.video.mp4 and only build the two audio versions")
    ap.add_argument("--progress", help="write render progress JSON here (frames done/total, fps)")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    info = probe(clip)
    W, H, fps = info["width"], info["height"], float(info["fps"])
    total = count_frames(clip)
    src_plan = json.load(open(a.plan))
    ranges, keep = cut_frames(src_plan, fps, total)
    pos = {src: k for k, src in enumerate(keep)}
    s2o = make_s2o(ranges, fps)
    plan = remap_plan(src_plan, s2o, ranges, fps)
    wcfg = plan["windows"]
    beats, zkeys = plan["beats"], plan["zoom"]

    face = json.load(open(stem(clip) + "_face.json"))["frames"]
    kf = [face[min(i, len(face) - 1)] for i in keep]          # face per OUTPUT frame
    fxs = np.array([f["x"] if f["x"] is not None else 0.4 for f in kf]) * W
    fys = np.array([f["y"] if f["y"] is not None else 0.3 for f in kf]) * H
    fws = np.array([f["w"] if f["w"] is not None else 0.25 for f in kf]) * W
    fhs = np.array([f["h"] if f["h"] is not None else 0.15 for f in kf]) * H
    # smoothed in OUTPUT order, so windows glide across a cut instead of jumping
    ax = tb.smooth_series(fxs + fws / 2, 0.6 * fps)
    ay = tb.smooth_series(fys + fhs / 2 + 0.10 * H, 0.6 * fps)
    hx0 = tb.smooth_series(fxs, 0.25 * fps)
    hy0 = tb.smooth_series(fys, 0.5 * fps)
    hw = tb.smooth_series(fws, 0.5 * fps)

    t_out = wcfg.get("t_out")
    profile = ProfilePanel(wcfg["profile"], t_out + 0.09)
    term = TerminalPanel(wcfg["terminal"], t_out)
    pill = PillPanel(wcfg["pill"], t_out + 0.18)
    question = QuestionCard(plan["question"], W) if plan.get("question") else None

    tmp_video = os.path.splitext(a.out)[0] + ".video.mp4"
    if a.audio_only:
        return build_audio(a, clip, src_plan, plan, term, tmp_video, ranges, keep)
    wr = VideoWriter(tmp_video, W, H, info["rate"])
    import time
    t_start = time.time()
    for layer in PutBack(clip, seconds=a.seconds, start=a.start):
        if layer.i not in pos:
            continue                              # this frame was cut out
        o = pos[layer.i]
        t = o / fps
        # 1) zoom the room and me together
        z = tb.zoom_at(zkeys, t)
        cx, cy = float(ax[o]), float(ay[o])
        room, alpha, fg = layer.frame, layer.alpha, layer.fg
        if z > 1.0005:
            M = np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]])
            warp = lambda im, fl=cv2.INTER_CUBIC: cv2.warpAffine(im, M, (W, H), flags=fl, borderMode=cv2.BORDER_REFLECT)
            room, fg = warp(room), warp(fg)
            alpha = np.clip(warp(alpha, cv2.INTER_LINEAR), 0, 1)
        Z = lambda x, c: c + (x - c) * z
        fbox = None
        if kf[o]["x"] is not None:
            fbox = (Z(fxs[o], cx), Z(fys[o], cy), Z(fxs[o] + fws[o], cx), Z(fys[o] + fhs[o], cy))
        # 2) the profile window goes BEHIND me (frost = blur of the room only)
        scene = room.astype(np.float32)
        M, op = place(profile, t, fbox, W, H, "profile", (Z(hx0[o], cx), Z(hy0[o], cy), hw[o] * z))
        if M is not None:
            draw_glass(scene, profile, M, op, blur_room_only(room, alpha, 16), t, shadow=False)
        # 3) me on top
        scene = over(fg, alpha, np.clip(scene + 0.5, 0, 255).astype(np.uint8)).astype(np.float32)
        # 4) windows in front of me (frost = blur of whatever is behind, me included)
        base = scene.copy()
        for panel, anchor in ((term, (0.695 * W, 0.612 * H)), (pill, (0.05 * W + pill.w / 2, 0.800 * H))):
            M, op = place(panel, t, fbox, W, H, "front", anchor)
            if M is not None:
                draw_glass(scene, panel, M, op, blur_plain(base, 14), t)
        # 5) small captions, then the end question
        text_c = np.zeros((H, W), np.float32)
        shadow_c = np.zeros((H, W), np.float32)
        for b in beats:
            for d in tb.avoid_face(tb.beat_draws(b, t, W, H), fbox, 0.04 * H):
                tb.blit(text_c, shadow_c, d)
        if text_c.any() or shadow_c.any():
            scene = tb.apply_text(scene, text_c, shadow_c)
        if question is not None:
            M, op = place(question, t, fbox, W, H, "front", (W / 2, 0.62 * H))
            if M is not None:
                base = scene.copy()
                draw_glass(scene, question, M, op, blur_plain(base, 14), t)
        wr.write(np.clip(scene + 0.5, 0, 255).astype(np.uint8))
        if wr.count % 30 == 0:
            el = time.time() - t_start
            if a.progress:
                json.dump({"done": wr.count, "total": len(keep), "fps": wr.count / el}, open(a.progress, "w"))
            if wr.count % 100 == 0:
                print(f"  {wr.count}/{len(keep)} frames")
    wr.close()
    n = count_frames(tmp_video)
    print(f"video: {n} frames" + ("" if (a.seconds or a.start) else f" (clip {total} - cut {total - len(keep)} = {len(keep)})"))
    if a.video_only:
        os.replace(tmp_video, a.out)
        print("wrote", a.out, "(no audio, test)")
        return
    build_audio(a, clip, src_plan, plan, term, tmp_video, ranges, keep)


def build_audio(a, clip, src_plan, plan, term, tmp_video, ranges, keep):
    """The final audio: my voice cleaned by voice.py, cut exactly like the picture,
    with the sound effects under it."""
    fps = float(probe(clip)["fps"])
    spf = int(round(sfx.SR / fps))
    n_out = len(keep) * spf
    words = json.load(open(stem(clip) + "_words.json"))["words"]
    import voice as vmod
    clean, _ = vmod.enhance(clip)
    events = sfx_events(plan, term)
    voice_c = cut_audio(clean, ranges, spf, n_out)
    mix_and_mux(tmp_video, voice_c, sfx.render(events, n_out, sfx.speech_rms(clean, words)), a.out)
    print("wrote", a.out, "(cleaned voice + effects)")
    os.remove(tmp_video)


if __name__ == "__main__":
    main()
