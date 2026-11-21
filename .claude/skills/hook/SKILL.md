---
name: hook
description: Write hooks and a full ~30 s spoken script in Uzbek for a KZB Mobile phone Reel, from the phone the user is selling today (model, storage, battery, condition, price). Use when the user says what phone they are filming, asks for a hook, "nima deyman", a script, or variants.
argument-hint: "phone model and anything known (storage, battery %, condition, price)"
---

# /hook PHONE

The phone is `$ARGUMENTS`. Write in Uzbek (Latin), the way the user talks: short, spoken,
friendly, no bookish words. Read `/reels-director` §1 first: questions and X-or-Y choices
work best for this audience, and plain spec lists don't.

## Facts

- Use only facts that are certain for this model (chip, camera, screen type, port), or
  facts the user gave. Write unknowns as placeholders: **[xotira] GB**, **[__]%**,
  **[NARX]**. Never invent a price, a battery %, or whether it still gets iOS updates.
- Name one or two honest downsides (e.g. LCD not OLED, Lightning not Type-C). Being honest
  builds trust, and the user's shop sells used phones, so trust is what sells.

## Output

1. **5 hook variants**, one per type: question/debate, price shock, comparison (X or Y),
   myth vs fact, and "who it's for". Give each one:
   - 🎙 the spoken line, which must work in under 3 s;
   - 📝 the on-screen text, at most 4 words, CAPS;
   - 🎬 what to do in the first second (phone towards the camera, two phones in two hands,
     tapping the phone on the desk…).
2. **Recommend one**, with one line on why, tied to the profile's numbers.
3. **The full script for the recommended hook**, in blocks with timings:
   HOOK (0–3 s) → JAVOB/why (3–12) → SHU TELEFON specs (12–22) → CHAQIRIQ: direktga /
   KZB Mobile'ga keling (22–27) → YAKUN: the same question back to the viewer (27–30).
   - Put the words that will become caption hits in **bold**.
   - Price comes late, after a 1 s pause, so people keep watching for it.
4. **Filming notes:**
   - Be already seated and speaking at frame 1; don't walk in.
   - Say the hook 2–3 different ways.
   - Hold the phone still for 1 s when showing it.
   - Say numbers clearly.
   - Stumbling is fine: `silences.py` and the cut tools will clean it up.

Keep the whole answer scannable. Don't repeat the playbook back to the user.
