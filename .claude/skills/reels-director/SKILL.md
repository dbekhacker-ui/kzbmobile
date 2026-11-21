---
name: reels-director
description: The playbook for every KZB Mobile Reels job - read it first whenever the user sends a new clip, asks for a video edit, a hook or script, posting advice, or says to make it "professional"/"viral". It holds what this account's audience responds to, the lessons from earlier edits (what broke and how it was fixed), the order of work, and the quality bar. Use together with /text-behind, /float-ui, /hook and /reels-qa.
---

# KZB Mobile Reels — director's playbook

The user runs KZB Mobile, a phone shop in Bulung'ur (Samarqand). Videos: the user sits at a
desk in the shop, holds a phone (usually a used iPhone), and talks in Uzbek about it for
~30 s. Reply in Uzbek (Latin), plainly, short sentences. Follow `CLAUDE.md` rules.

## 1. What this audience responds to (real numbers from the profile)

| Reel | Views | Why it worked |
|---|---|---|
| Customers at the counter, laptop shows KZB | 79.9K | real shop life, people, not an ad |
| "XR YOKI 11" (comparison) | 27.1K | a question viewers want to answer |
| iPhone 11 Pro at night, 😂 | 6.6K | mood, curiosity |
| "350.000" price in huge outline text | 4.3K | price shock |
| Plain product shots (13 Pro Max, 14 Pro…) | 1–3.8K | nothing to react to |
| Our IPHONE 15 PRO text-behind | 2.2K (first day) | strong look, but the first 2 s were walking in |

Rules that follow from this:
- **Questions and comparisons beat specs.** Open with a question or an X-or-Y choice.
- **Real people and real shop moments** beat studio-looking product shots.
- **A price shown big** is a hook on its own.
- **Never open on dead time.** The first word must land in the first 0.5 s. If the clip opens
  with walking in or sitting down, cut it with `silences.py` (`intro`), with the user's OK.
- **End with a question** that is easy to answer in one word ("Olarmidingiz?").

Posting: best 19:30–22:00, second best 12:30–14:00 (Asia/Tashkent). The first hour decides
reach: reply to comments fast, share to Stories, cross-post to Facebook. Exact hours are in
Instagram → Professional dashboard → Most active times; I can't read that, the user can.

## 2. Order of work for a new clip

1. Copy the clip in, `ffprobe` it, look at a contact sheet (`fps=0.5,tile=9x2`).
2. Helpers, in parallel where possible: `mask.py`, `hands.py`, `face.py`, then `phone.py`,
   and `transcribe.py --lang uz` (8 s chunks; never pass name hints in `--prompt`).
3. `silences.py <clip>` → `<name>_cuts.json`: dead start/end and stumble pauses. For each
   pause, transcribe the audio before and after it separately; if a word repeats, cut the
   repeat too. Ask before cutting the intro/outro (it changes the length).
4. Fix the transcript by hand into correct Uzbek Latin, keeping the user's own words
   (abzor, karobka, yomkost…). Keep a list of every guessed word to report.
5. Write the plan, test 3–6 s pieces with `--start/--seconds`, look at frames, then render.
6. `/reels-qa` before sending.
7. For long jobs, offer the live progress page (artifact with a `db` status doc).

## 3. Lessons learned (don't repeat these)

**Cut-out (mask)**
- MediaPipe's plain selfie model is blobby on 9:16 phone video. Use the multiclass model on
  overlapping square tiles (mask.py does). Remap: room ≈0.08, me ≈0.98.
- Fast phone/hand moves: the mask drops them for single frames. put_back fills one-frame
  dropouts from flow-warped neighbours, and phone.py adds the phone.
- phone.py bridges detector gaps along the hand, but **checks colour and size** on guessed
  boxes. An unchecked guess once pasted a piece of room between my arm and head.
- Hole filling must leave real gaps open (arm raised next to head, hand on hip) and never
  fill holes inside a hand.
- One frame of real camera motion blur on a swinging phone can't be removed. Say so.

**Text, layout and safe zones**
- Instagram covers: bottom ~15% (caption, username) and the right-edge column at 0.55–0.9 of
  the height (buttons). Keep text and windows out of both.
- The profile grid shows covers cropped to 3:4 around the centre: keep cover text inside
  the middle 75% of the height.
- Behind-me text: zoom the room and me, but **draw the background word unzoomed**, or the
  word gets cut at the sides.
- Captions: one beat ends before the next starts. A hit's sub-word can collide with the pill
  or terminal, so make it its own line beat instead.
- Nothing may touch the face box. Test the panel's real rotated outline, not its bounding box.

**Speech and transcripts**
- Whisper large-v3 drifts on long Uzbek speech: it repeats words, guesses the wrong
  language (Kazakh) or invents "Thank you for watching" over silence. Transcribe in ~8 s
  chunks, with loudness normalised first, VAD off, and `condition_on_previous_text=False`.
- Always tell the user which caption words were guessed.

**Audio**
- The user's raw voice is quiet (~−26.5 LUFS). voice.py brings it to −14 LUFS with
  classic processing only. Stronger denoise or gating barely helped (+1 dB) and risks a
  choppy sound, so it stays gentle.
- SFX sit 6–14 dB under the voice. The hook can land before speech starts. Measure levels
  with numbers; I can't listen, so ask the user to listen with headphones.
- Cuts: drop whole frames, use a 15 ms equal-power crossfade taken from inside the cut, and
  check that the join step is far below normal speech steps.
- A reverb or crossfade array off by one sample crashed a render once. Test the audio path
  on a short piece before a 20-minute render.

**Cuts that don't show**
- A zoom jump on every cut, alternating in and out ([1.11, 1.06] then [1.0, 1.02]), makes a
  jump cut read as an edit.
- Overlays run on output time, so windows and captions never jump at a cut. Smooth face
  tracks in output order.

**Process**
- A 20-minute render that fails at the audio step is expensive: keep `--audio-only` to
  rebuild sound on an existing video.
- Don't page through other long chat sessions for data (usage). The Instagram notes live on
  the user's PC (`instagram-kzbmobile.md`), reachable only if that computer is attached.

## 4. Quality bar ("wow")

A video is ready only when all of these hold:
- **Hook:** the first word in under 0.5 s, and a visual change by 1.0 s.
- **Pacing:** something new on screen every 2–3 s (caption beat, zoom, window, pop).
- **Look:** no edge halos, no see-through hands or phone, no text over the face, nothing in
  Instagram's UI zones.
- **Sound:** voice clear and even at −14 LUFS; effects audible but never louder than the voice.
- **End:** a one-line question, minimal, no hype, no emoji.
- **Report:** guessed words, anything I couldn't fix, and what the user should listen for.
