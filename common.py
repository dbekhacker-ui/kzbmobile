"""Shared helpers: find the clip, probe it, decode frames with ffmpeg, write video.

ffmpeg does the decoding everywhere, so phone rotation metadata is applied
for us (ffmpeg auto-rotates by default) and every script sees upright frames.
"""
import glob
import json
import os
import subprocess
import urllib.request
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
# Files we generate ourselves; never treat them as "the clip".
DERIVED = ("_mask", "_out", "_test", "_preview", "_fx")

MODEL_URLS = {
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}


def model_path(name):
    """Download a MediaPipe model into models/ once, then reuse it."""
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        os.replace(tmp, path)
    return path


def find_clip(arg=None):
    if arg:
        return os.path.abspath(arg)
    found = [p for p in sorted(glob.glob(os.path.join(HERE, "*")))
             if p.lower().endswith(VIDEO_EXT)
             and not any(os.path.splitext(os.path.basename(p))[0].endswith(d) for d in DERIVED)]
    if not found:
        raise SystemExit("No video clip found in this folder. Put your clip here or pass its path.")
    if len(found) > 1:
        print("Several clips found, using", os.path.basename(found[0]), "(pass a path to choose)")
    return found[0]


def stem(clip):
    return os.path.splitext(clip)[0]


def probe(clip):
    """Return dict(width, height, fps (Fraction), rate (str), duration, rotation).

    width/height are the *displayed* size, i.e. after rotation.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
         "-show_streams", "-show_format", clip],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if "rotate" in s.get("tags", {}):
        rot = int(s["tags"]["rotate"])
    if rot % 180 != 0:
        w, h = h, w
    rate = s.get("r_frame_rate") or s.get("avg_frame_rate")
    fps = Fraction(rate)
    if fps == 0:
        fps = Fraction(s["avg_frame_rate"])
    dur = float(s.get("duration") or info["format"].get("duration") or 0)
    return dict(width=w, height=h, fps=fps, rate=f"{fps.numerator}/{fps.denominator}",
                duration=dur, rotation=rot)


def read_frames(clip, w, h, pix="rgb24", seconds=None, start=None):
    """Yield every decoded frame (upright) as a numpy array, one per source frame.
    start/seconds limit it to a section (for tests)."""
    ch = {"rgb24": 3, "gray": 1}[pix]
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", clip]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * ch
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            a = np.frombuffer(buf, np.uint8)
            yield a.reshape(h, w) if ch == 1 else a.reshape(h, w, 3)
    finally:
        if proc.poll() is None:
            proc.kill()  # stopped early: end ffmpeg quietly instead of a broken pipe
        proc.stdout.close()
        proc.wait()


class VideoWriter:
    """Pipe raw frames into ffmpeg. Constant frame rate = the clip's rate, so
    N frames in gives N frames out. Optionally copies the clip's audio untouched."""

    def __init__(self, path, w, h, rate, pix="rgb24", audio_from=None, seconds=None, crf=14, start=None):
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", rate, "-i", "-"]
        if audio_from:
            if start:
                cmd += ["-ss", str(start)]
            if seconds:
                cmd += ["-t", str(seconds)]
            cmd += ["-i", audio_from, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-r", rate, "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing video")


def count_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])
