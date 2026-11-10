---
name: sandglass
description: >-
  Sandglass: a cinematic data-film style where a subject is told as coloured sand in clear glass
  vessels shaped like real places, on a dark table, rendered in Blender Cycles at 4K. Each grain is a
  fixed amount (people, tonnes, speakers), colour is category, pile height is quantity, sand flies
  between vessels as flows. Diorama close-ups open, a still wide map holds while history plays, then a
  low golden-hour flight between mountains of sand, with an epic orchestral score and a calm narrator.
  First made for "200,000 years of human migration" (ancestry_sand/). Use whenever the user says
  Sandglass, "sandglass style", "the sand style" or "like the migration film", or wants a history, map,
  population, flow or any quantity over time and place turned into an animated film or YouTube video,
  even if they don't name the style; also for any part of one (scale, camera, light, music, narration,
  render package, listing, thumbnails).
---

# Sandglass

A way of turning a subject that changes across **places and time** into a physical, beautiful film:
the quantities become sand, the places become glass vessels, and time becomes sand pouring, flying and
piling up. Its first film: *200,000 years of human migration* (code in `ancestry_sand/`; read its
`README.md` for the full engineering story).

People share these films because they look like a real object someone built, not a chart, and because
the data stays honest. Keep both halves: the craft and the honesty.

## The signature (what makes a frame unmistakably Sandglass)

1. **A dark matte table** with the world (or the subject's map) laid out as **thick, clear, bevelled
   glass vessels**, one per region or category, standing on the real outlines.
2. **Sand as the data.** Small spherical grains, each a little different in shade and shine. One
   grain = a stated number of things, shown on screen ("1 grain ≈ 56,000 people").
3. **Colour = category** (12–15 distinct, named colours with a legend). Height and amount = quantity.
4. **Motion = flow.** Sand flies between vessels in arcs, streaked by motion blur.
5. **Mountains of poured sand** where things concentrate: messy pyramids with sand-pile slopes,
   shoulders and ridges, that spill over the glass at the coast. Never neat cylinders or towers.
6. **The camera's three acts:** diorama close-ups in the opening, a still wide map through the long
   middle, and a low, swooping flight between the mountains at the end (epic title-sequence energy),
   rising to a reveal of the whole map.
7. **Golden-hour light, deep grade:** a low warm sun for the finale, studio strips that make glass read
   as glass, AgX High Contrast with exposure -0.15.
8. **Sound:** an original epic orchestral score (a driving cello figure in 6/8 that builds to the full
   orchestra) under a calm, warm narrator (ElevenLabs "Lily").

Exact values for all of this (light angles, slopes, timings, fonts, colours) are in
`references/style.md`. Read it before designing a new film.

## Honesty rules (part of the style, not an add-on)

The look invites people to trust what they see, so the data has to deserve it:
- **Sand is proportional to the quantity**, with one scale for the whole film. If the scale must be
  compressed (the start of the migration film would have had 2 grains), compress it smoothly, put the
  current "1 grain ≈ N" on screen, and say so in the description.
- **Height must read as quantity.** Viewers equate a taller pile with more. Concentrations get a summit
  height that ranks them by size (height ∝ value^0.8 keeps the order without needles).
- **Cite sources** for every dataset and date; give ranges where the science is debated.
- **Say what is simplified** in the description and pinned comment (e.g. "regions show today's
  average", "city mountains borrow sand from the countryside so they can be seen").
- **Categories are labels for groups, not judgments.** For people, avoid anything that reads as race
  science, and keep the narration plain and humane when the history is violent.

## Making a new Sandglass film

Work with the user through these steps; show something (a still, a contact sheet, a sound clip) at
each step rather than describing it.

1. **Concept.** What is the quantity (the grains), what are the places (the vessels), what are the
   categories (the colours), and what is the time span? Write the one-line title in the house form:
   "N years of X" / "X, told in sand". Check the data exists, with sources.
2. **Story beats.** 8–14 beats with dates and on-screen times; a 3–4 minute film is the sweet spot
   (the migration film is 3:42). Time usually runs on a log clock so deep history moves fast.
3. **The sand budget.** Decide the scale: total grains today ~100k–150k for a world map (enough to
   fill, few enough to render), the compression exponent, and how concentrations (cities) become
   mountains within each region's budget.
4. **Build by adapting `ancestry_sand/`** (see `references/pipeline.md` for what each module does and
   what to change). Reuse the simulation, exporter, Blender scene builder, compositor and mixer; replace
   the story data.
5. **Look at it early and often.** Low-res CPU previews (`scripts/quick_render.py`), contact sheets
   (`scripts/contact_sheet.py`), and top-down / relief plots of the height field before any long render.
6. **Sound.** Generate the score in sections matched to the story beats; screen takes for dropouts and
   early endings (`scripts/screen_music.py`, `scripts/extend_score.py`); record the narration line by
   line; mix with `ancestry_sand/mix_audio.py` (it ducks the music and cleans clicks off clip ends).
7. **Deliver the package** the user renders on their own PC (4K Cycles .blend, transparent HUD .mov,
   compose .blend with relative paths, music + narration WAVs, captions .mov, SRTs, a README with a
   3-step quick start), dry-run the compose file inside a copy of the package, then zip it.
8. **Publish kit:** YouTube title, description (how to read it, chapters, what's simplified, sources),
   pinned comment, and three 1280x720 thumbnails for Test & Compare: new low, shallow-focus stills
   finished like photographs, with at most one small caption (`references/style.md` section 7; never
   big yellow headline type, which reads as an AI template).

## Lessons already paid for

These cost real time on the first film; `references/pipeline.md` has the details.
- Tall narrow towers look fake; slope-limited mountains with irregular flanks look like poured sand.
  Island cities need to spill over the glass or they become slabs.
- Raw renders look washed out next to graded thumbnails: bake the grade into the film.
- Music generators end early and sometimes drop out mid-piece: always screen takes by loudness per
  section and check the last 6 seconds.
- Text-to-speech clips can end with a stray click or a 6 kHz chirp after the speech: clean every clip.
- Watch homophones in narration ("too" heard as "two"); rephrase.
- Transparent 4K HUD overlays get huge if labels move every frame; hide moving labels or keep them few.
- Never name a TV show, film or composer in a music prompt: describe the style instead (it keeps the
  music original, and the music service refuses such prompts anyway).

## Pointers

- `references/style.md`: the visual, camera, light, typography and sound specification, with numbers.
- `references/pipeline.md`: module map of `ancestry_sand/`, how to adapt it, the delivery package,
  and the gotchas list.
- `scripts/`: preview renders, contact sheets, music screening, score extension, and thumbnails
  (`thumb_shot.py` renders stills from any angle, `thumb_finish.py` finishes them; fonts in `assets/fonts`).
