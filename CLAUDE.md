# Video effects on my clips

The clip is the video file in this folder (if there isn't one, ask me for it).
Generated files sit next to it as `<name>_mask.mp4`, `<name>_hands.json`,
`<name>_face.json`, and effect outputs as `<name>_<effect>.mp4`.

## Setup (already done once)

- ffmpeg on PATH (Homebrew on a Mac, winget on Windows, apt on Linux).
- Python venv in `.venv` with `mediapipe numpy pillow` (OpenCV comes with mediapipe).
  Run everything with `.venv/bin/python` (Windows: `.venv\Scripts\python`).
- MediaPipe **Tasks API only** (`mediapipe.tasks.python.vision`). New versions have no `mp.solutions`.
- Models download once into `models/` (MediaPipe models from storage.googleapis.com; the
  Whisper model from HuggingFace, which only the **kzbmobile** cloud environment allows;
  download with `HF_HUB_DISABLE_XET=1`, afterwards run with `HF_HUB_OFFLINE=1`).
- `faster-whisper` is also installed in `.venv` (for `transcribe.py`).
- Linux containers may need `libegl1 libgles2` for MediaPipe to load.

## Helpers (run these before any effect, in this order)

| Script | Output | What it is |
|---|---|---|
| `mask.py [clip]` | `<name>_mask.mp4` | white = me, black = room, soft edges. Mac: Apple Vision person segmentation (accurate) via a swiftc-compiled helper; Windows/Linux: MediaPipe selfie multiclass model on overlapping square tiles (much cleaner than the plain selfie segmenter on tall phone video). |
| `hands.py [clip]` | `<name>_hands.json` | up to 2 hands per frame: 21 points, score, which hand (`hand`), which half of the screen (`side`), gaps ≤3 frames filled. |
| `face.py [clip]` | `<name>_face.json` | smoothed face box per frame. |
| `phone.py [clip]` | `<name>_phone.mp4` | the phone in my hand (object detector + point segmenter, gaps bridged along my hand, checked by colour). Run after hands.py. |
| `transcribe.py [clip] --lang uz` | `<name>_words.json` | what I say, word by word with times (faster-whisper large-v3). |
| `put_back.py` | (library) | puts me back on top of any background (uses mask + hands + phone, fills one-frame dropouts using motion). |

All take `--seconds N` to only do the first N seconds (for tests); effects and `PutBack` also
take `--start S` / `start=` to test a section in the middle. All coordinates in the
JSON files are **fractions of the frame**: x from the screen's left edge, y from the top.
`common.py` has the shared ffmpeg I/O (`probe`, `read_frames`, `VideoWriter`).

## Rules for every effect

1. **Keep my audio exactly as it is.** Copy the original audio stream (`-c:a copy`);
   never re-encode, resample, trim, or mix it. `VideoWriter(..., audio_from=clip)` does this.
2. **Keep the size, frame rate and frame count** of the clip. Decode with
   `common.read_frames` (one array per source frame, rotation applied) and write with
   `common.VideoWriter` at `probe(clip)["rate"]`. Check the output with
   `common.count_frames` against the clip.
3. **Put me back with `put_back.py`.** Build whatever goes behind me, then
   `layer.over(background)`. Don't composite with the raw mask, and don't grow,
   blur, or threshold the mask yourself.
   ```python
   from common import find_clip, probe, VideoWriter
   from put_back import PutBack
   clip = find_clip(); info = probe(clip)
   wr = VideoWriter(out, info["width"], info["height"], info["rate"], audio_from=clip)
   for layer in PutBack(clip):
       bg = ...                    # HxWx3 uint8, may use layer.frame, layer.t, layer.hands
       wr.write(layer.over(bg))
   wr.close()
   ```
4. **Draw any text with Pillow** (`PIL.ImageDraw`), never with ffmpeg's `drawtext`
   (some ffmpeg builds can't draw text).
5. **Never put anything over my face.** Use `<name>_face.json` and keep text, stickers,
   graphics and particles out of the face box (pad it a bit) on every frame.
   Things that go *behind* me are fine, since put_back keeps me on top.
6. **Left and right mean as seen on screen.** "Put it on the left" means the left side
   of the video as you watch it. In `<name>_hands.json`, `side` is screen side; `hand` is
   my actual hand (assumes an un-mirrored clip; run `hands.py --mirrored` for a mirrored one).

## Effects

- `/text-behind WORD` (skill in `.claude/skills/text-behind`) makes `text-behind.mp4` with
  `text_behind.py`: the big word behind me, captions from `<name>_plan.json`, zooms.

## Checking work

- Test on the first few seconds (`--seconds 3`) before rendering the whole clip.
- `put_back.py --at 0.5 1.5 2.5 --seconds 3 --bg ff00ff` puts me over flat magenta and
  saves PNGs to `checks/`. Look for see-through spots, room showing around or between
  fingers, and halos in the hair.
- Look at the frames you render before saying an effect is done.
