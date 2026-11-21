---
name: reels-qa
description: Final quality check of a finished Reel before sending it to the user - frame count/size/fps, loudness, cut joins, the first second, Instagram safe zones, face rule, captions, and a frame-by-frame look. Use at the end of /text-behind, /float-ui or any edit, or when the user asks "tekshir", "xato yo'qmi".
argument-hint: "path to the finished video and its clip"
---

# /reels-qa VIDEO CLIP

Run every check. Report each one as passed, or failed with what you saw. Fix the failures
before sending. Never call the video done without having looked at the frames.

## 1. Numbers (one Bash call)

- `ffprobe` both files: size and fps must match the clip. Frame count must equal the clip's,
  or the clip's minus the cut frames when the plan has cuts. Audio length must match the
  video within 1 frame.
- Loudness: `ffmpeg -i VIDEO -vn -af ebur128=peak=true -f null -`.
  - Cleaned voice: −14 ±0.5 LUFS, true peak ≤ −1 dB.
  - Untouched voice: the same LUFS as the clip.
- Every cut join: the largest sample step within ±9 ms of the join must be well under the
  99th-percentile step of the surrounding 1 s. If it isn't, that's a click.
- Effects vs voice: each effect sits 6–14 dB under the voice at that moment (hook ≤ 5 dB under).

## 2. Look (contact sheets, Read the images)

- `fps=0.5,scale=220:-2,tile=9x2` of the whole video: pacing, nothing stuck, the ending is clean.
- The first second, frames 0, 8, 15 and 30: is something already happening?
- 2 frames before and after every cut: a zoom jump hides it, and no overlay jumps.
- Each caption hit and the hero: readable, no overlap with panels or with each other.
- Full-resolution crops at the fastest hand and phone moves: no see-through, no room over
  the background text, no halo.

## 3. Rules

- Nothing touches the face box (the effect code enforces it; spot-check 3 frames).
- Instagram UI zones are empty: the bottom 15% and the right column at 0.88+ of the
  width, 0.55–0.9 of the height.
- If there is a cover: its text stays inside the centre 3:4 crop.

## 4. Report to the user (Uzbek, short)

- What was checked, with the key numbers: frames, LUFS, duration.
- Words that were guessed in the captions.
- Anything that can't be fixed, for example one frame of real camera blur.
- What to listen to with headphones, at timestamps, because I can't hear audio.
