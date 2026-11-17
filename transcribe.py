"""Transcribe the clip's speech with per-word timings -> <name>_words.json.

Uses faster-whisper (large-v3 by default, downloaded once from HuggingFace into
models/whisper). The audio is only read, never changed.

usage: .venv/bin/python transcribe.py [clip] [--lang uz] [--model large-v3]
"""
import argparse
import json
import os
import subprocess

from common import MODELS, find_clip, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--lang", default=None, help="language code, e.g. uz (auto-detect if omitted)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--chunk", type=float, default=8.0, help="seconds per chunk (cut at quiet points)")
    ap.add_argument("--vad", action="store_true", help="skip silence with VAD (can drop quiet speech)")
    ap.add_argument("--prompt", default="",
                    help="spelling hints for names and terms (can make Whisper repeat them; keep short)")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    from faster_whisper import WhisperModel

    import numpy as np
    # Decode with ffmpeg to 16 kHz mono float samples (a copy; the clip is untouched).
    # Quiet phone audio confuses Whisper: bring this copy up to normal speech level first.
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vn", "-ac", "1",
                          "-af", "highpass=f=80,loudnorm=I=-18:TP=-2:LRA=11", "-ar", "16000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    wav = np.frombuffer(pcm, np.float32)
    model = WhisperModel(a.model, device="cpu", compute_type="int8",
                         download_root=os.path.join(MODELS, "whisper"))
    # Whisper loses the thread on long Uzbek stretches (repeats, wrong language),
    # so cut the speech into ~8 s chunks at the quietest moments and do each alone.
    sr = 16000
    hop = sr // 50                                           # 20 ms
    rms = np.sqrt(np.convolve(wav ** 2, np.ones(hop) / hop, mode="same")[::hop] + 1e-12)
    cuts, pos, n = [0], 0, len(wav) // hop
    while pos + 50 * a.chunk < n:
        lo, hi = pos + int(50 * a.chunk * 0.7), min(n, pos + int(50 * a.chunk * 1.3))
        pos = lo + int(np.argmin(rms[lo:hi]))
        cuts.append(pos)
    cuts.append(n)
    words, info = [], None
    for c0, c1 in zip(cuts, cuts[1:]):
        t0 = c0 * hop / sr
        segs, info = model.transcribe(wav[c0 * hop:c1 * hop], language=a.lang, word_timestamps=True,
                                      beam_size=5, initial_prompt=a.prompt or None, vad_filter=a.vad,
                                      condition_on_previous_text=False, temperature=0.0)
        for sg in segs:
            for w in sg.words:
                words.append(dict(w=w.word.strip(), start=round(t0 + w.start, 3), end=round(t0 + w.end, 3),
                                  p=round(w.probability, 3)))
            print(f"[{t0 + sg.start:6.2f}-{t0 + sg.end:6.2f}] {sg.text}")
    out = stem(clip) + "_words.json"
    json.dump(dict(language=info.language, language_probability=round(info.language_probability, 3),
                   words=words), open(out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(words)} words, language {info.language} ({info.language_probability:.2f})")


if __name__ == "__main__":
    main()
