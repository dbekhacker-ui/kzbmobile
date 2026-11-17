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
    ap.add_argument("--prompt", default="iPhone 15 Pro, titan, Type-C, abzor.",
                    help="spelling hints for names and terms")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    from faster_whisper import WhisperModel

    import numpy as np
    # Decode with ffmpeg to 16 kHz mono float samples (a copy; the clip is untouched).
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    wav = np.frombuffer(pcm, np.float32)
    if True:
        model = WhisperModel(a.model, device="cpu", compute_type="int8",
                             download_root=os.path.join(MODELS, "whisper"))
        segs, info = model.transcribe(wav, language=a.lang, word_timestamps=True, beam_size=5,
                                      initial_prompt=a.prompt, vad_filter=True)
        words = []
        for s in segs:
            for w in s.words:
                words.append(dict(w=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3),
                                  p=round(w.probability, 3)))
            print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text}")
    out = stem(clip) + "_words.json"
    json.dump(dict(language=info.language, language_probability=round(info.language_probability, 3),
                   words=words), open(out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(words)} words, language {info.language} ({info.language_probability:.2f})")


if __name__ == "__main__":
    main()
