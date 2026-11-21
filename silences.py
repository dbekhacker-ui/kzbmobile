"""Find what to cut so the video feels tight: the dead start before I speak, the
dead end after I finish, and pauses inside sentences where I lost my words.

Reads the clip's audio and <name>_words.json (transcribe.py). Prints a report and
writes <name>_cuts.json:
  {"intro": [0, t] or null,         # dead start (walking in / sitting down)
   "outro": [t, end] or null,        # dead end after the last word
   "pauses": [[a, b], ...],          # mid-speech pauses, already trimmed to leave
                                     # KEEP seconds of breath, frame-aligned
   "suggested_cuts": [...]}          # pauses only (intro/outro are for me to decide)

A pause counts when 50 ms loudness stays QUIET dB below the speech level for at
least MIN seconds between two words. Normal breaths between sentences are kept
(they are shorter than MIN, or I leave KEEP of them).

Before using a pause cut: transcribe the audio just before and after it on its own
(see the float-ui skill) - if a word is repeated around the pause, cut the repeat too.

usage: .venv/bin/python silences.py <clip> [--min 0.35] [--keep 0.15]
"""
import argparse
import json
import subprocess

import numpy as np

from common import find_clip, probe, stem


def loudness(clip, hop=0.05, sr=16000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    n = int(hop * sr)
    fr = a[: len(a) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9), hop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--min", type=float, default=0.35, help="shortest pause worth cutting (s)")
    ap.add_argument("--keep", type=float, default=0.15, help="breath to leave in each cut pause (s)")
    ap.add_argument("--quiet", type=float, default=16.0, help="dB below median speech that counts as silence")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    fps = float(probe(clip)["fps"])
    words = [w for w in json.load(open(stem(clip) + "_words.json"))["words"]
             if not w["w"].lower().startswith(("thank you", "subtitles"))]
    db, hop = loudness(clip)
    t = np.arange(len(db)) * hop
    speech = np.median(db[db > np.percentile(db, 40)])
    silent = db < speech - a.quiet

    first, last = words[0]["start"], words[-1]["end"]
    # speech really starts where loudness rises near the first word
    lead = np.where(~silent & (t > first - 0.6))[0]
    s0 = t[lead[0]] if len(lead) else first
    s0 = float(max(s0, first - 0.3))
    intro = [0.0, round(max(0.0, s0 - 0.12), 3)] if s0 > 0.6 else None
    tail = np.where(~silent & (t < last + 0.8))[0]
    s1 = float(t[tail[-1]] + hop if len(tail) else last)
    dur = len(db) * hop
    outro = [round(s1 + 0.35, 3), round(dur, 3)] if dur - s1 > 1.2 else None

    pauses = []
    k = 0
    while k < len(t):
        if silent[k] and s0 + 0.2 < t[k] < last:
            j = k
            while j < len(t) and silent[j]:
                j += 1
            a0, a1 = float(t[k]), float(t[j - 1] + hop)
            if a1 - a0 >= a.min:
                c0 = a0 + a.keep / 2
                c1 = a1 - a.keep / 2
                f0, f1 = int(np.ceil(c0 * fps)), int(np.floor(c1 * fps))
                if f1 > f0:
                    pauses.append([round(f0 / fps, 4), round(f1 / fps, 4)])
            k = j
        else:
            k += 1

    out = dict(intro=intro, outro=outro, pauses=pauses, suggested_cuts=pauses,
               speech_level_db=round(float(speech), 1))
    path = stem(clip) + "_cuts.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"speech level {speech:.1f} dB, silence below {speech - a.quiet:.1f} dB")
    print("dead start:", intro, " dead end:", outro)
    for p0, p1 in pauses:
        before = [w["w"] for w in words if w["end"] <= p0 + 0.1][-3:]
        after = [w["w"] for w in words if w["start"] >= p1 - 0.1][:3]
        print(f"pause {p0:6.2f}-{p1:6.2f}  ({p1 - p0:.2f} s cut)   ...{' '.join(before)} | {' '.join(after)}...")
    print("wrote", path)


if __name__ == "__main__":
    main()
