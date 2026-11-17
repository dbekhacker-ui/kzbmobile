---
name: text-behind
description: Make a "text behind me" video from the clip in this folder - a big word behind the person at head height, animated captions of what they say (stressed words pop big, the product name biggest), and professional punch-in zooms. Use when the user types /text-behind WORD or asks for this kind of video with a new word.
argument-hint: "WORD (the big background word, e.g. IPHONE 15 PRO)"
---

# /text-behind WORD

Make `text-behind.mp4` from the clip in this folder: `WORD` big behind me, my speech as
animated captions in the same style, zooms on the important moments. Follow `CLAUDE.md`
(audio untouched, same size/fps/frame count, put me back with put_back.py, Pillow text,
nothing over my face, left/right as seen on screen). Talk to me in the language I write in
(usually Uzbek).

The background word is `$ARGUMENTS` in capitals. If I gave no word, ask me for one.

## 1. Clip and helpers

- The clip is the video in this folder. If there is none, ask me for it. If there are
  several, ask which one.
- Run each helper that has no output yet or whose output is older than the clip.
  Use `.venv/bin/python` and run them in parallel where possible:
  `mask.py`, `hands.py`, `face.py`, then `phone.py` (it needs `<name>_hands.json`), and
  `transcribe.py --lang uz` (or my language).
- `transcribe.py` needs the Whisper model from HuggingFace. The **kzbmobile** cloud
  environment allows it. If you need to download it, set `HF_HUB_DISABLE_XET=1`. Once it is
  in `models/whisper`, run with `HF_HUB_OFFLINE=1`.

## 2. Write `<name>_plan.json`

Copy the layout of `IMG_0811_plan.json`. Fill it in from `<name>_words.json`:

- `"word"`: the background word.
- **Fix the transcript.** Whisper's Uzbek is rough. Rewrite each word in correct Uzbek
  Latin spelling, but keep my own words (loanwords like "abzor" and "karobka" stay as I
  said them). Keep Whisper's start times. Write down every word you were unsure about, so
  you can tell me at the end.
- Beats (each one ends before the next starts):
  - `intro`: the opening phrase ("Bugungi abzorimiz ..."), two or three chunks. They slide
    in from the left one by one. The first is biggest and each next one is smaller and
    further toward the centre.
  - `hit`: every stressed word, i.e. a feature, material, port, number, spec, price or
    shop name (TITAN, TYPE-C, 128 GB, 78%, $550, KZB MOBILE). Write it in CAPS and
    optionally add a small `sub` word under it (with `sub_t`).
  - `hero`: the product name when I say it in full. It's the biggest beat, split over two
    lines.
  - `line`: everything else, in lowercase, one or two lines. Each word rises in exactly
    when I say it.
- `zoom`: keyframes `[t, z]` with smooth easing between them.
  - Slow push to ~1.07 over the intro.
  - A 0.2 s punch to 1.12–1.16 at each hit's start, easing back to ~1.05 over about a second.
  - A push to 1.18–1.20 through the hero, then back to 1.0.
  - Stay at 1.0 while I walk into or out of the frame.
  - Never go above 1.22. Keep about 1–3 s between punches so it doesn't feel jumpy.

## 3. Test before rendering everything

Render short pieces with
`text_behind.py --plan <name>_plan.json --start S --seconds N --out checks/segX.mp4`:
the intro, two hits, the hero, and every moment I move fast (the phone raised to my head,
quick hand moves). For each piece, pull frames with ffmpeg and look at them (`Read` the
PNGs). Check:

- captions don't overlap each other or my face, and are readable;
- the background word stays whole and my head covers the middle of it;
- no letters show through my hand, neck, phone or shirt, and no room shows over the
  letters in a gap that is really me (and the reverse: real room gaps, like between a
  raised arm and my head, show the letters);
- no halo around my hair or cap.

If the phone or hand breaks, fix `phone.py` or `put_back.py`, not the effect. To find
flicker frames, compare each mask frame with its neighbours (see the earlier session's
approach: pixels the median of t-1, t, t+1 has but t lacks).

## 4. Final render and checks

- Render the whole clip to `text-behind.mp4` (no `--start`/`--seconds`).
- Check that the frame count, size and fps match the clip (`common.count_frames`, ffprobe).
  Check the audio is byte-identical: md5 of
  `ffmpeg -i X -map 0:a -c copy -f data -` must be the same for the clip and the output.
- Look at a frame every ~3 s, plus each hit and the hero.
- Send me `text-behind.mp4`. Tell me which caption words you corrected or guessed, so I
  can fix them in the plan and you can re-render.
