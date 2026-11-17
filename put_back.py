"""Put the person back on top of whatever goes behind them.

Every effect uses this. It reads <name>_mask.mp4 (from mask.py) and, if
present, <name>_hands.json (from hands.py), and for each frame builds:

  alpha  float32 HxW, 0..1   how much of the person to show
  fg     float32 HxWx3       the person's colors, with the room cleaned
                             out of the soft edge

What it does to the raw mask:
  * Solid inside: deep inside the silhouette alpha ramps smoothly to 1, so
    nothing shows through a shirt or a blurry hand. Small see-through holes
    inside the body are filled. Holes inside a hand (from hands.py) are never
    filled, so gaps between fingers stay open; large holes of real room (an
    arm on a hip) are left alone too.
  * Hands: where the mask misses a hand, the palm and fingers from hands.py
    are drawn in, a little thinner than the real hand, so the patch never
    reaches past it.
  * Soft edge kept, never grown: outside the solid silhouette alpha is the
    raw mask value, untouched. The solid inside blends into the edge with a
    smooth ramp, so there is no hard step.
  * Edge colors: in the soft edge the old room color is pulled out by moving
    each pixel toward the person's own nearby colors, so no halo shows around
    hair and no new colors appear.

Use from an effect:

    from put_back import PutBack
    pb = PutBack(clip)                       # or PutBack(clip, seconds=3)
    for layer in pb:                         # one per frame, in order
        bg = make_background(layer.frame, layer.i)   # anything HxWx3 uint8
        out = layer.over(bg)                 # person composited on top
        writer.write(out)

Test from the command line (writes PNGs into checks/):

    .venv/bin/python put_back.py [clip] --at 0.5 1.5 2.5 --seconds 3 --bg ff00ff
"""
import argparse
import json
import os

import cv2
import numpy as np

from common import HERE, find_clip, probe, read_frames, stem

FINGERS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
PALM = [0, 1, 5, 9, 13, 17]
MIN_HAND_SCORE = 0.5


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hand_shapes(hands, w, h):
    """Return (patch, zone): patch = thin palm+finger fill (float 0..1),
    zone = area around each hand where holes must never be filled (bool)."""
    patch = np.zeros((h, w), np.uint8)
    zone = np.zeros((h, w), np.uint8)
    for hd in hands or []:
        if hd.get("score", 1) < MIN_HAND_SCORE:
            continue
        p = np.array(hd["points"], np.float64)[:, :2] * (w, h)
        size = np.linalg.norm(p[9] - p[0])  # wrist -> middle knuckle
        if size < 4:
            continue
        # Zone: convex hull of the hand, grown a bit.
        hull = cv2.convexHull(p.astype(np.float32)).astype(np.int32)
        one = np.zeros_like(zone)
        cv2.fillConvexPoly(one, hull, 1)
        zone |= cv2.dilate(one, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size * 0.3) | 1,) * 2))
        # Palm: knuckles + wrist, shrunk toward its centre so it stays inside.
        palm = p[PALM]
        c = palm.mean(0)
        palm = c + (palm - c) * 0.82
        cv2.fillPoly(patch, [np.round(palm * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
        # Fingers: bones drawn thinner than a real finger (~0.19 x size wide).
        thick = max(1, int(round(size * 0.11)))
        for f in FINGERS:
            chain = p[list(f[1:])]
            # Pull the last segment back so the patch stops short of the fingertip.
            chain = chain.copy()
            chain[-1] = chain[-2] + (chain[-1] - chain[-2]) * 0.6
            start = 1 if f[1] == 1 else 0  # thumb starts at its base joint
            for a, b in zip(chain[start:-1], chain[start + 1:]):
                cv2.line(patch, tuple(np.round(a * 4).astype(int)), tuple(np.round(b * 4).astype(int)),
                         255, thick, cv2.LINE_AA, shift=2)
    patch = cv2.GaussianBlur(patch, (0, 0), 1.0).astype(np.float32) / 255.0
    return patch, zone.astype(bool)


def _norm_blur(img, weight, sigma):
    """Average of img over pixels where weight>0, spread out by a Gaussian."""
    wsum = cv2.GaussianBlur(weight, (0, 0), sigma)
    acc = cv2.GaussianBlur(img * weight[..., None], (0, 0), sigma)
    return acc, wsum


def nearby_colors(img, weight, s):
    """Normalized-convolution color fill at two scales (fine first, coarse fallback)."""
    acc1, w1 = _norm_blur(img, weight, s)
    acc2, w2 = _norm_blur(img, weight, s * 5)
    c1 = acc1 / np.maximum(w1, 1e-6)[..., None]
    c2 = acc2 / np.maximum(w2, 1e-6)[..., None]
    k = _smoothstep(0.02, 0.15, w1)[..., None]
    return c1 * k + c2 * (1 - k)


def refine(frame, raw, hands=None):
    """frame: HxWx3 uint8, raw: HxW uint8 mask. Returns (alpha, fg) float32."""
    h, w = raw.shape
    S = min(h, w)
    img = frame.astype(np.float32)
    m = raw.astype(np.float32) / 255.0
    # Undo tiny codec noise at the extremes so black is really black.
    m = np.clip((m - 0.02) / 0.96, 0, 1)

    patch, zone = hand_shapes(hands, w, h)

    # --- holes inside the silhouette -------------------------------------
    core = (m >= 0.5).astype(np.uint8)
    r_close = max(1, int(S * 0.004))
    closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_close + 1,) * 2))
    max_room_hole = (0.06 * S) ** 2

    def holes(solid):
        """Enclosed background components of `solid` that should be made opaque."""
        n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - solid, connectivity=8)
        out = np.zeros((h, w), np.uint8)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if x == 0 or y == 0 or x + bw == w or y + bh == h:
                continue  # touches the frame edge: that's the room, not a hole
            comp = lab[y:y + bh, x:x + bw] == k
            if zone[y:y + bh, x:x + bw][comp].mean() > 0.2:
                continue  # inside/near a hand: keep gaps between fingers open
            if area > max_room_hole and m[y:y + bh, x:x + bw][comp].mean() < 0.15:
                continue  # big and clearly room (e.g. arm on hip): leave it
            out[y:y + bh, x:x + bw][comp] = 1
        return out

    # Holes in the raw core (specks), plus holes that only close up once
    # hairline leaks to the outside are sealed by a small closing.
    fill = holes(core)
    sealed = holes(closed)
    if sealed.any():  # grow back over what closing ate, but only into the hole
        sealed = cv2.dilate(sealed, np.ones((2 * r_close + 3,) * 2, np.uint8))
    fill = (fill | sealed) & (1 - core) & (1 - zone.astype(np.uint8))

    # --- solid inside, smooth ramp from the edge -------------------------
    solid_region = (core | fill).astype(np.uint8)
    dist = cv2.distanceTransform(solid_region, cv2.DIST_L2, 5)
    ramp = _smoothstep(0.0, max(2.0, S * 0.012), dist)
    alpha = m + (1 - m) * ramp
    # Filled holes are opaque (they are inside the body by construction).
    alpha = np.maximum(alpha, fill.astype(np.float32) * _smoothstep(0, 2.0, dist))
    # Hands the mask missed.
    alpha = np.maximum(alpha, patch)
    alpha = np.clip(alpha, 0, 1)
    # Drop faint specks floating in the room: soft blobs with no solid part.
    n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.03).astype(np.uint8), connectivity=8)
    if n > 2:
        has_core = np.zeros(n, bool)
        has_core[np.unique(lab[alpha >= 0.5])] = True
        has_core[0] = True
        alpha[~has_core[lab]] = 0

    # --- clean the old room out of the soft edge -------------------------
    sure_fg = (alpha >= 0.98).astype(np.float32)
    s = max(1.5, S * 0.005)
    near_fg = nearby_colors(img, sure_fg, s)
    edge = (alpha > 0.0) & (alpha < 0.98)
    # How much of the pixel is room: pull that share toward the person's colors.
    t = np.clip(1.0 - alpha, 0, 1) ** 0.6
    # Keep the pixel's own brightness detail (hair strands) on top of the pull.
    fg = img.copy()
    pulled = img + (near_fg - img) * t[..., None]
    fg[edge] = pulled[edge]
    # Never step outside the colors the person actually has nearby.
    lo = cv2.erode(np.where(sure_fg[..., None] > 0, img, 255.0), np.ones((5, 5), np.uint8))
    hi = cv2.dilate(np.where(sure_fg[..., None] > 0, img, 0.0), np.ones((5, 5), np.uint8))
    has = cv2.dilate(sure_fg, np.ones((5, 5), np.uint8)) > 0
    clamp = edge & has
    fg[clamp] = np.clip(fg[clamp], np.minimum(lo[clamp], near_fg[clamp]), np.maximum(hi[clamp], near_fg[clamp]))
    return alpha.astype(np.float32), fg.astype(np.float32)


def over(fg, alpha, bg):
    a = alpha[..., None]
    out = fg * a + bg.astype(np.float32) * (1 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Layer:
    def __init__(self, i, t, frame, alpha, fg, hands):
        self.i, self.t, self.frame, self.alpha, self.fg, self.hands = i, t, frame, alpha, fg, hands

    def over(self, bg):
        return over(self.fg, self.alpha, bg)


class PutBack:
    """Iterate the clip frame by frame with the person's alpha + cleaned colors.

    Besides <name>_mask.mp4 it uses, when present:
      <name>_hands.json  (hands.py)  hands filled in where the mask misses them
      <name>_phone.mp4   (phone.py)  the phone in my hand counts as me
    and it fills brief mask dropouts on fast motion: a pixel the mask lost in
    this frame is restored when BOTH the previous and next frame (warped to
    this one with optical flow) say it is me.
    """

    def __init__(self, clip, seconds=None, mask=None, hands=None, phone=None, temporal=True, start=None):
        self.clip = clip
        self.info = probe(clip)
        self.seconds = seconds
        self.start = start or 0.0
        self.temporal = temporal
        self.mask_path = mask or stem(clip) + "_mask.mp4"
        if not os.path.exists(self.mask_path):
            raise SystemExit(f"{self.mask_path} not found - run mask.py first")
        hp = hands or stem(clip) + "_hands.json"
        self.hands = json.load(open(hp))["frames"] if os.path.exists(hp) else []
        if not self.hands:
            print("note: no hands json, hand fill-in disabled (run hands.py)")
        pp = phone or stem(clip) + "_phone.mp4"
        self.phone_path = pp if os.path.exists(pp) else None

    def _raw(self):
        """Yield (frame, raw person mask incl. phone)."""
        w, h = self.info["width"], self.info["height"]
        kw = dict(seconds=self.seconds, start=self.start)
        frames = read_frames(self.clip, w, h, **kw)
        masks = read_frames(self.mask_path, w, h, pix="gray", **kw)
        if self.phone_path:
            phones = read_frames(self.phone_path, w, h, pix="gray", **kw)
        else:
            phones = iter(lambda: None, 0)
        for frame, m, ph in zip(frames, masks, phones):
            yield frame, (m if ph is None else np.maximum(m, ph))

    def _temporal(self, it):
        """Motion-compensated fill of one-frame dropouts (never removes anything)."""
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        w, h = self.info["width"], self.info["height"]
        sw, sh = w // 2, h // 2
        gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))

        def small_gray(f):
            return cv2.cvtColor(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)

        def warp(cur_g, nb_g, nb_m):
            flow = dis.calc(cur_g, nb_g, None)  # cur(x) ~ nb(x + flow)
            small = cv2.resize(nb_m, (sw, sh), interpolation=cv2.INTER_AREA)
            out = cv2.remap(small, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

        buf = []
        for frame, m in it:
            buf.append((frame, m, small_gray(frame)))
            if len(buf) == 1:
                yield frame, m                 # first frame: no previous one to agree with
            elif len(buf) == 3:
                (pf, pm, pg), (cf, cm, cg), (nf, nm, ng) = buf
                agree = np.minimum(warp(cg, pg, pm), warp(cg, ng, nm))
                yield cf, np.maximum(cm, agree)
                buf.pop(0)
        if len(buf) == 2:
            yield buf[1][0], buf[1][1]         # last frame: no next one

    def __iter__(self):
        fps = float(self.info["fps"])
        stream = self._temporal(self._raw()) if self.temporal else self._raw()
        first = int(round(self.start * fps))
        for i, (frame, raw) in enumerate(stream, start=first):
            hands = self.hands[i]["hands"] if i < len(self.hands) else []
            alpha, fg = refine(frame, raw, hands)
            yield Layer(i, i / fps, frame, alpha, fg, hands)


def parse_color(s):
    s = s.lstrip("#")
    return np.array([int(s[k:k + 2], 16) for k in (0, 2, 4)], np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Test: put the person over a flat color at given times.")
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--at", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--bg", default="ff00ff")
    ap.add_argument("--outdir", default=os.path.join(HERE, "checks"))
    a = ap.parse_args()
    clip = find_clip(a.clip)
    pb = PutBack(clip, seconds=a.seconds or max(a.at) + 0.2)
    fps = float(pb.info["fps"])
    want = {int(round(t * fps)): t for t in a.at}
    os.makedirs(a.outdir, exist_ok=True)
    color = parse_color(a.bg)
    for layer in pb:
        if layer.i in want:
            bg = np.empty_like(layer.frame)
            bg[:] = color
            path = os.path.join(a.outdir, f"over_{a.bg}_{want[layer.i]:.1f}s.png")
            cv2.imwrite(path, cv2.cvtColor(layer.over(bg), cv2.COLOR_RGB2BGR))
            print("wrote", path)
            if layer.i >= max(want):
                break


if __name__ == "__main__":
    main()
