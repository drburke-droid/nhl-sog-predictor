# Sandglass style specification

The numbers below are what the first film (*200,000 years of human migration*) used. Treat them as
the house defaults: change them when a new subject needs it, and say why.

Units: the map is 1920 x 1080 "map px"; 100 map px = 1 Blender unit; the sand grid is 3-px cells
(640 x 360).

## 1. The set

| Element | Value |
|---|---|
| Table | dark matte, near-black, slight sheen; nothing else in frame |
| Vessels | thick bevelled clear glass on every outline, 14 map px tall (`WALL_H = 0.14`), one per region |
| Islands too thin to hold sand | thicken the outline just enough (`buffer`), never magnified insets (they confuse viewers) |
| Grain | sphere, radius 1.35–1.7 map px, a level every 0.55 px vertically (`GRAIN_DZ`), jittered ±0.65 px |
| Grain material | per-grain variation in shade, hue and roughness; slight subsurface translucency |
| Flying grains | same grains in flight, with a velocity attribute so Cycles motion blur streaks them |

## 2. The data grammar

- **One grain = one fixed amount**, shown top-left under the year: "1 grain ≈ 56,000 people".
- **Compression for deep time:** fill ∝ (quantity / quantity today)^0.38. The migration film runs
  from ~830 grains (1 grain ≈ 120 people) to ~145,000 (≈ 56,000). Say this in the description.
- **Before settled life, spread by habitable area**; after, by real density (towns blurred at
  2.5 / 5 / 16 cells), levelled off at 60 grains so only concentrations rise above the land.
- **Concentrations are mountains** (cities of 1M+ in the first film):
  - summit = 22 levels x (value / 5 million)^0.8 (Tokyo 109, New York 64, Calgary 9);
  - flanks at ~6 levels per cell (~48 degrees): bigger means a wider base;
  - profile (1 - d/R)^1.35, 2–4 shoulders (lesser summits at 0.3–0.65 R, 0.35–0.6 of the height),
    ridges and gullies (±22%, deepening downslope), grain roughness ±12%, the summit exact;
  - minimum base radius 2.2 cells, so small ones are mounds, not spikes;
  - coastal ones spill over the glass onto the table beside their own vessel;
  - at most 70% of a region's sand; past that they steepen rather than shrink;
  - on an already-high plain, still rise 45% of their height above it.
- **Colour palette** (one per category, with a legend bottom-left); the first film's 14:
  `#e8606b #b97a4c #f29a46 #f1cb5a #dfe5ee #93dbe8 #38bfae #62c486 #4d94e8 #7173e0 #c970c9 #b4dc48
  #9a6ae0 #ec5e93`. Keep neighbours on the map distinguishable; reserve near-white and pale blue for
  "special" categories (Neanderthal, Denisovan).

## 3. The camera (30 fps)

**Act 1, the dioramas (0:00–0:45).** Low close-ups over the first sparse grains, a look across to a
second place, back for the key event, then a pull-out that lands on the wide map by 0:30–0:45. Splines
through 6–8 keys, each ~6 s apart. Aperture about f/0.9 x min(max(7/distance, 1), 4): very shallow
depth of field close in, sharp when wide. The "share-worthy" moments live here, so open with one.

**Act 2, the still map.** The camera does not move while history plays (the story is the sand, and a
still camera renders faster and never shimmers). Region names fade in when it settles, and hide
whenever the camera moves.

**Act 3, the flight.** When the story reaches today: descend among the mountains and fly low through
the 20–25 most important places, title-sequence style:
- hop times 1.0 + 0.5 x distance, clamped to 1.7–3.2 s; ~1.9 s for long ocean crossings;
- height 0.8 x peak height + 0.45 units, clamped 0.65–1.25; tilt ~64 degrees; heading from the path's
  central difference (look ahead); lift over any peak in the way (clearance from the settled heights);
- a whip pan across the map's edge (e.g. the Pacific) with the cut at a half frame and quarter-frame
  camera keys, so motion blur hides the wrap;
- names appear as each place comes up, a third of the way up its mountain;
- end where the story began, then rise to reveal the whole map for the last ~5 s.

## 4. Light and grade

| | Value |
|---|---|
| Studio | soft bright strips for reflections and refraction; dimmed for the finale; off during the low flight (they only glint in the glass), back for the reveal |
| Golden sun | azimuth 290, elevation 30, colour (1.0, 0.58, 0.30), strength 7, angle 3; keyed on as the finale begins |
| Rim light | cool, fades out as the flight starts |
| Sun and lights in reflections | hidden from glossy and transmission rays (no glints on the glass) |
| View transform | AgX, look "AgX - High Contrast", exposure -0.15 |
| Compositor | bloom (size 8), gentle vignette |
| Motion blur | shutter 0.5, centred; resting sand excluded |
| Cycles | 64 samples adaptive (threshold 0.04), OIDN with albedo + normals, 6 bounces (6 transmission, 1 diffuse), no caustics, fixed seed, persistent data, 4K 3840x2160 |

## 5. On-screen text (HUD, drawn natively at 4K)

- Fonts: Inter Display SemiBold for the year, the title and place names; Inter Medium/Regular for the
  rest (`/usr/share/fonts/opentype/inter`).
- Top left: the year (large) with "years ago" / "CE" under it, then "1 grain ≈ N people".
- Bottom left: the legend, a small heading and one swatch per colour.
- Top right: a one-line sources note ("Ancestry from published ancient-DNA studies; population from
  long-run estimates.").
- Title: one line, centred, over a darkened opening, fading out at 4.2–5.8 s. House form:
  "200,000 years of human migration".
- Ink #e8eef8, muted #93a1b8; text has a soft dark halo so it reads on sand.
- Delivered as a transparent QuickTime Animation (.mov, qtrle argb) composited over the frames.

## 6. Sound

**Score** (ElevenLabs music, `composition_plan` with sections matched to the story beats,
`respect_sections_durations: true`, total exactly the film's length):
- global styles: "epic fantasy television main-title theme", "instrumental", "original melody",
  "driving cello ostinato in 6/8", "minor key", "dark, regal and adventurous", "relentless rhythmic
  pulse like turning clockwork", "orchestral strings, low brass, timpani and taiko", "builds layer by
  layer", "continuous music with no silences or pauses";
- negatives: vocals, lyrics, pop, EDM, rock, comedic, lo-fi, piano-led, ambient, silence;
- the arc: a lone cello figure at the start, a solo cello theme as things spread, full strings and
  horns in the middle, grave low brass at the dark turn, a timpani-roll build into "today", the full
  main-title statement (cellos and horns in unison, taiko, soaring violins, wordless choir) for the
  flight, a last chord and a cello echo for the reveal.
- An alternative, gentler score also exists (felt piano, strings, hand percussion: `music_plan.json`).

**Narration:** ElevenLabs `eleven_multilingual_v2`, voice Lily (`pFZP5JQG7iQjIQuC4Bku`), speed 1.0,
one request per line with `previous_text` / `next_text` for natural flow. ~300 words for 3:42, a line
every 5–10 s, plain words, numbers spelled out ("two percent"), no homophone traps.

**Mix:** both stems at -16 LUFS, peaks -1.5 dBTP; music 0.85, ducked to 0.27 from 0.3 s before to
0.6 s after each line, smoothed over 0.35 s; voice clips cleaned (stray end clicks silenced, 5 ms
fade-in, 30 ms fade-out). Also deliver music-only + short burned-in captions as an alternative cut.

## 7. Thumbnails and titles (1280 x 720, three for Test & Compare)

The film looks like a real object someone built; the thumbnail should look like a photograph of it.
Big all-caps headline type in yellow or gold with a dark glow reads as a template (viewers now take it
for an AI video): never use it. The first film's v1 thumbnails did; the v2 set replaced them.

- **Shoot new stills, don't grab film frames.** `scripts/thumb_shot.py` renders any frame of the film
  .blend from any camera pose (tx, ty, tilt, dist, heading) with depth of field: drafts at 25% / 24
  samples (~20 s each on CPU) to find the angle, finals at 50% / 96 samples (~4 min each).
  Low cameras (tilt 58-76) with f/0.9-2 read as macro photographs of a model; keep north roughly up
  so the map is recognised at a glance.
- **One idea, one point of interest**, readable at 360 px wide (the phone feed): a continent people
  recognise, one bright thing to look at (a spark of sand leaving Africa, a fan of streams).
- **Finish like a photo, not a poster** (`scripts/thumb_finish.py`): saturation x0.95, soft halation
  round the brightest highlights, natural vignette, slightly lifted blacks, fine film grain. Never
  push saturation or contrast.
- **Text: none, or one small factual caption** set like a museum label or photo caption: IBM Plex Mono
  (or Newsreader), sentence case, off-white (241, 236, 225), ~40 px, a tight soft shadow, no outline,
  no glow, in a corner. A date or a scale, not a slogan, and never the title again.
- **Titles:** sentence case, plain and specific, under 60 characters. Avoid colons, "told in",
  "the untold story", "journey", "epic", "like never before". A specific number ("145,000 grains")
  or the maker's voice ("I simulated...") reads as a person.
- The migration film's v2 set: the spark of the first band leaving Africa ("c. 55,000 years ago",
  frame 1105), emigrants streaming across the Atlantic ("1903", frame 4480), the whole world today
  ("1 grain = about 56,000 people", frame 6640), and a text-free macro of the first grains in southern
  Africa (frame 20). Poses are in `videos/ancestry/thumbnails/shots.json`.
