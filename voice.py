"""Make my voice sound like a clip-on mic: clear, close and even - with classic
audio processing only (no AI voice, nothing re-synthesised).

Chain (ffmpeg filters):
  highpass 80 Hz        rumble / handling noise out
  afftdn                gentle spectral noise reduction (room hiss), measured noise floor
  EQ                    -3 dB at 300 Hz (boxy room), +3 dB at 4.5 kHz (presence),
                        +2 dB shelf above 10 kHz (air)
  deesser               tames sharp "s"
  acompressor           evens out loud/quiet words (3:1)
  loudnorm (2 passes)   -14 LUFS, true peak -1.5 dB (what Instagram plays at)

usage: .venv/bin/python voice.py [clip] [--out <name>_voice.wav]
Writes a 44.1 kHz stereo WAV of exactly the clip's audio length.
"""
import argparse
import json
import subprocess

import numpy as np

from common import find_clip, stem

CHAIN = ("highpass=f=80:poles=2,"
         "afftdn=nr=10:nf={nf}:tn=1,"
         "equalizer=f=300:t=q:w=1.0:g=-3,"
         "equalizer=f=4500:t=q:w=1.4:g=3,"
         "highshelf=f=10000:g=2,"
         "deesser=i=0.4:m=0.5:f=0.5,"
         "acompressor=threshold=-24dB:ratio=3:attack=6:release=120:makeup=3")


def read_audio(path, af=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn"]
    if af:
        cmd += ["-af", af]
    cmd += ["-ac", "2", "-ar", "44100", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32).reshape(-1, 2)


def noise_floor_db(clip):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", clip, "-vn", "-af", "astats=metadata=0", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    vals = [float(l.split(":")[-1]) for l in out.splitlines() if "Noise floor dB" in l]
    return max(-80.0, min(-20.0, vals[-1])) if vals else -50.0


def enhance(clip):
    nf = noise_floor_db(clip)
    pre = CHAIN.format(nf=round(nf, 1))
    # pass 1: measure loudness after the chain
    meas = subprocess.run(["ffmpeg", "-hide_banner", "-i", clip, "-vn", "-af",
                           pre + ",loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True).stderr
    js = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
    ln = ("loudnorm=I=-14:TP=-1.5:LRA=9:linear=true:"
          f"measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:"
          f"measured_thresh={js['input_thresh']}:offset={js['target_offset']}")
    y = read_audio(clip, pre + "," + ln + ",aresample=44100")
    n = len(read_audio(clip))
    if len(y) < n:
        y = np.pad(y, ((0, n - len(y)), (0, 0)))
    return y[:n], nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    clip = find_clip(a.clip)
    y, nf = enhance(clip)
    out = a.out or stem(clip) + "_voice.wav"
    import wave
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"wrote {out} (noise floor was {nf:.1f} dB)")


if __name__ == "__main__":
    main()
