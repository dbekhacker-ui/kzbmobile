---
name: float-ui
description: Make a "floating frosted-glass UI" video from the clip in this folder. A KZB Instagram-profile window tucked behind the person's head, a terminal that types the phone's specs and a notification pill float beside them, with small animated captions, punch-in zooms, sound effects under the voice, a minimal end question, and an optional cleaned-up voice version. Use when the user types /float-ui [PILL TEXT] or asks for this kind of video.
argument-hint: "PILL TEXT (e.g. iPhone 13 pro 128GB 77%)"
---

# /float-ui [PILL TEXT]

Make `float-ui.mp4` (my voice cleaned up by `voice.py` + sound effects; one version only)
from the clip in this folder with `float_ui.py`. Follow
`CLAUDE.md`:
- same size, fps and frame count;
- put me back with `put_back.py`;
- draw all text with Pillow;
- nothing touches my face box;
- left/right as seen on screen.

Mixing in sound effects, cleaning the voice and cutting out my stumbles are part of this
effect, by my request. Talk to me in the language I write in (usually Uzbek).

The notification pill text is `$ARGUMENTS`, exactly as I typed it. If I gave none, ask me.

## 1. Clip and helpers

- The clip is the video in this folder. If there are several, ask which one, and pass
  its path to every script.
- Run these, in parallel where possible:
  - `mask.py`, `hands.py` and `face.py`;
  - then `phone.py`, which needs `hands.json`;
  - `HF_HUB_OFFLINE=1 transcribe.py <clip> --lang uz`. It works in ~8 s chunks, because
    on long stretches Whisper repeats itself or switches language. Don't pass a
    `--prompt` with names: Whisper starts repeating them.
- Look at a contact sheet of the clip
  (`ffmpeg -vf "fps=0.5,scale=200:-2,tile=9x2"`), and read where my face and the phone
  usually are from `<name>_face.json` and `<name>_phone.mp4`.

## 2. Write `<name>_plan.json` (copy the layout of `13_plan.json`)

- `hook_t`: the moment I sit down or the action starts. The hook sound lands there.
- `windows`:
  - `pill.t_in`: right after the hook. The pill is the first thing seen, and the offer
    in the first second is the hook.
  - `profile.t_in`: ~0.3 s after the pill.
  - `terminal.t_in`: ~0.3 s after the profile.
  - `pill.pulse`: the times I say the numbers that are in the pill.
  - `profile.logo`: `assets/kzb_logo_round.png` (the real KZB Mobile logo, round).
  - `profile.tiles`: 3 real reels from my Instagram profile, cropped from my screenshots
    (`assets/tile*.png`, view counts kept), ordered by views so the strongest comes last.
    Each tile appears at its `t`: the first ~0.6 s after the window, then spread across the
    video (the last one just before the call to action).
  - `terminal.lines`: `$ kzb scan iphone` right after the pop-in, then
    `["@progress qidiruv", t0, t1]` running until I name the model, then
    `key(9 chars) value` lines exactly when I say each spec. Only facts I actually said.
  - `t_out`: ~1 s before my last sentence ends. The windows leave so the end is clean.
- `beats`: the captions, in corrected Uzbek, with Whisper's times kept, built the same
  way as for `/text-behind`:
  - `line`: small lowercase words;
  - `hit`: a stressed word or number, with `scale` 1.3 for the model name;
  - every beat ends before the next one starts.
  Write down every word you guessed, so you can tell me.
- `question`: a short, honest question that makes people comment, about the product in
  the video ("iPhone 13 Pro hali ham arziydimi?" with "Fikringizni izohda yozing"). Start
  it ~2 s before the end, after the windows have left. No emoji, no hype.
- `cuts`: [[start, end], ...] in clip seconds for pauses where I lost my words. Find them
  from 50 ms loudness (−45 dB or quieter for 0.3 s+) inside a sentence. Transcribe the pieces
  before and after separately to make sure no word is repeated (if one is, cut the repeat
  too). Leave ~0.15 s of the pause. Whole frames are dropped, with a 15 ms audio crossfade.
- `cut_zoom`: one [jump-to, settle-to] per cut, alternating in and out
  (e.g. [1.11, 1.06] then [1.0, 1.02]), so each jump cut reads as an intended edit.
- `zoom`: a slow push after the hook, a 0.2 s punch (1.10–1.16) on each hit, the deepest
  (≤1.16) on the model name, easing back, and 1.0 at the very end.

## 3. Test pieces, then render

- Use `float_ui.py <clip> --plan <plan> --start S --seconds N --video-only --out checks/x.mp4`
  on the opening, the model-name moment, the specs, any fast moves (phone raised to the
  face), and the ending. Pull frames and look at them. Check:
  - the profile window's corner sits behind my cap/head, and no panel or caption touches
    my face;
  - the terminal lines appear when I say them;
  - nothing hides the phone for long;
  - text stays out of Instagram's bottom ~15% and right-edge button column;
  - no letters or room show through my hands, neck or phone, and no halo in my hair.
- Do the full render without `--start`/`--seconds`; `--progress logs/progress.json` reports
  frames done and speed.

## 4. Checks before sending

- Size and fps equal to the clip; frame count = clip frames − cut frames.
- ~-14 LUFS, true peak ≤ -1 dB (`ffmpeg -af ebur128=peak=true`).
- Look at the frames just before and after every cut.
- Effects sit about 9–12 dB under the voice; the hook ~5 dB under. They must never be
  louder than my voice.
- Tell me which caption words you guessed. Tell me plainly that you can't listen to the
  audio, so I should listen to both versions before posting.
