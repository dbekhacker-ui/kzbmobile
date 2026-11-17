"""Small, clean UI sound effects synthesised from scratch (no samples, no AI,
no licensing issues), and a mixer that lays them under the voice.

Sounds (all mono float32 at SR):
  hook    - riser whoosh into a soft low thump (opening / sit-down moment)
  pop     - bubbly UI pop (a window or caption popping in)
  ding    - two-note notification bell (the notification pill)
  tick    - one keyboard tick (terminal typing, use per character)
  whoosh  - short air swish (zoom punches, things leaving)
  chime   - soft, low two-note chime (end question)

Levels: every sound is scaled relative to the voice, so effects sit under
the speech: `voice_rms` is the RMS of the speech parts, and each event
has a gain in dB relative to it (default -9 dB: clearly audible, never
louder than the voice).
"""
import numpy as np

SR = 44100
_rng = np.random.default_rng(7)


def _t(sec):
    return np.arange(int(sec * SR)) / SR


def _env(n, attack, decay):
    """Fast attack, exponential decay (time constant `decay` s) envelope of n samples."""
    a = max(1, int(attack * SR))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a) ** 2
    rest = np.arange(n - a) / SR
    e[a:] = np.exp(-rest / max(decay, 1e-4))
    return e


def _lowpass(x, cutoff):
    """One-pole low-pass (cutoff may be an array for sweeps)."""
    cutoff = np.broadcast_to(np.asarray(cutoff, float), x.shape)
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def _bandpass(x, lo, hi):
    return _lowpass(x, hi) - _lowpass(x, lo)


def _reverb(x, room=0.18, mix=0.18):
    """Tiny, smooth room: decaying-noise impulse response."""
    n = int(room * SR)
    ir = _rng.standard_normal(n) * np.exp(-np.arange(n) / (room * SR / 5))
    ir = _lowpass(ir, 6000)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    wet = np.convolve(x, ir)                       # len(x) + n - 1
    out = np.concatenate([x, np.zeros(n - 1)])
    return out * (1 - mix) + wet * mix


def _norm(x):
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def pop():
    t = _t(0.16)
    f = 260 + 640 * np.exp(-t / 0.018)          # quick downward chirp
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.002, 0.05)
    click = _bandpass(_rng.standard_normal(len(t)), 1500, 6000) * _env(len(t), 0.0005, 0.006)
    return _norm(_reverb(body + 0.35 * click, 0.12, 0.12))


def ding():
    t = _t(0.9)
    out = np.zeros(len(t))
    for start, f0 in ((0.0, 1318.5), (0.085, 1975.5)):   # E6 then B6
        s = int(start * SR)
        tt = t[: len(t) - s]
        tone = (np.sin(2 * np.pi * f0 * tt) + 0.25 * np.sin(2 * np.pi * f0 * 2.01 * tt)
                + 0.08 * np.sin(2 * np.pi * f0 * 3.0 * tt))
        out[s:] += tone * _env(len(tt), 0.002, 0.28)
    return _norm(_reverb(out, 0.3, 0.22))


def tick():
    t = _t(0.035)
    n = _bandpass(_rng.standard_normal(len(t)), 2500, 9000) * _env(len(t), 0.0003, 0.004)
    thock = np.sin(2 * np.pi * 180 * t) * _env(len(t), 0.0005, 0.008) * 0.5
    return _norm(n + thock)


def whoosh(dur=0.32, up=True):
    t = _t(dur)
    n = _rng.standard_normal(len(t))
    p = t / dur
    cut = 600 + 5200 * (p if up else 1 - p) ** 1.5
    shape = np.sin(np.pi * np.clip(p, 0, 1)) ** 1.6
    return _norm(_reverb(_lowpass(n, cut) * shape, 0.15, 0.15))


def hook():
    """Riser that lands on a soft low thump at 0.35 s (align `at` to the thump)."""
    rise = whoosh(0.38, up=True) * 0.8
    t = _t(0.7)
    f = 48 + 70 * np.exp(-t / 0.05)
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.003, 0.22)
    out = np.zeros(int(0.35 * SR) + len(t))
    out[: len(rise)] += rise[: len(out)]
    out[int(0.35 * SR):] += thump * 1.2
    return _norm(_reverb(out, 0.35, 0.2))


def chime():
    t = _t(1.4)
    out = np.zeros(len(t))
    for start, f0 in ((0.0, 659.3), (0.16, 987.8)):       # E5 then B5, soft
        s = int(start * SR)
        tt = t[: len(t) - s]
        tone = np.sin(2 * np.pi * f0 * tt) + 0.15 * np.sin(2 * np.pi * 2 * f0 * tt)
        out[s:] += tone * _env(len(tt), 0.01, 0.45)
    return _norm(_reverb(out, 0.45, 0.3))


SOUNDS = dict(pop=pop, ding=ding, tick=tick, whoosh=whoosh, hook=hook, chime=chime)
# Offset of the sound's "moment" inside the sample (so `at` lands on the beat).
LEAD = dict(hook=0.35, whoosh=0.20)


def speech_rms(voice, words=None, sr=SR):
    """RMS of the voice while speaking (word spans if given, else loud frames)."""
    v = voice if voice.ndim == 1 else voice.mean(1)
    if words:
        parts = [v[int(w["start"] * sr):int(w["end"] * sr)] for w in words if w["end"] > w["start"]]
        x = np.concatenate([p for p in parts if len(p)]) if parts else v
    else:
        hop = sr // 50
        fr = v[: len(v) // hop * hop].reshape(-1, hop)
        r = np.sqrt((fr ** 2).mean(1))
        x = fr[r > np.percentile(r, 60)].ravel()
    return float(np.sqrt((x ** 2).mean()) + 1e-9)


def render(events, n_samples, voice_rms, channels=2):
    """events: [(name, at_seconds, gain_db_vs_voice[, pan -1..1])] -> float32 (n, channels)."""
    out = np.zeros((n_samples, channels), np.float32)
    cache = {}
    for ev in events:
        name, at, gain_db = ev[0], ev[1], ev[2]
        pan = ev[3] if len(ev) > 3 else 0.0
        if name not in cache:
            cache[name] = SOUNDS[name]()
        s = cache[name]
        rms = np.sqrt((s[np.abs(s) > 0.05 * np.abs(s).max()] ** 2).mean())
        s = s * (voice_rms * 10 ** (gain_db / 20) / (rms + 1e-9))
        start = int((at - LEAD.get(name, 0.0)) * SR)
        a, b = max(0, start), min(n_samples, start + len(s))
        if b <= a:
            continue
        seg = s[a - start:b - start]
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        out[a:b, 0] += seg * l * np.sqrt(2)
        if channels > 1:
            out[a:b, 1] += seg * r * np.sqrt(2)
    return out
