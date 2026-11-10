# Sandglass pipeline: building a film from `ancestry_sand/`

The first film's code is the template. Copy the folder for a new subject (e.g. `languages_sand/`) and
change the story data first; most of the machinery carries over unchanged.

## Contents
1. Module map
2. Adapting it to a new subject
3. Render and preview workflow
4. Sound workflow
5. The delivery package
6. Gotchas

## 1. Module map

| File | What it does | Change for a new subject? |
|---|---|---|
| `story.py` | all the data: categories and colours (`COMPONENTS`), vessels (`VESSELS`), populations by era, events, the clock (video s vs years), captions, title, `MEGACITIES` + `CITY_HILLS` and the mountain constants | **Yes: this is the subject** |
| `data/wup2018_cities_1m.csv` | UN 2025 urban agglomerations of 1M+ (force-added; `data/` is gitignored) | if mountains are cities |
| `vessels.py` | country outlines to glass vessel shapes (`codes`, `bbox`, `clip`, `buffer`) | rarely |
| `proj.py` | lon/lat to map px | rarely |
| `sand.py` | the sandpile simulation: per-cell stacks, targets (`cell_targets`), city mountains (`build_cities`, `city_rise`, spill), relocation, toppling, events | the target rules if the grammar changes |
| `camera.py` | the camera path: `OPENING` dioramas, `WIDE`, the generated flight (`CITIES`, `_build_tour`, `_clearance`, whip cut) | the opening shots and the flight's places |
| `film.py` | the painter: HUD (year, scale, legend, labels, sources), fonts, colours | text only |
| `export_blender.py` | runs the sim and writes grain records + flights for Blender, the 4K HUD .mov, captions .mov + SRT | no |
| `blender_scene.py` / `blender_film.py` | the Cycles scene: table, glass, Geometry Nodes point cloud driven by scene time (no Python at render), lights, grade, render settings | rarely |
| `blender_compose.py` | the compose .blend (VSE): frames + HUD + title + audio, relative paths | title text |
| `mix_audio.py` | masters the score, places and cleans the narration lines, ducks, writes WAV/MP3/SRT | no |
| `narration_script.json` | timed narration lines | **Yes** |
| `music_plan_epic.json` / `music_plan.json` | score section plans | section names and durations |

## 2. Adapting it to a new subject

1. **Categories** -> `COMPONENTS` (key, name, colour). 8–15 categories; check the colours against
   each other on the dark table.
2. **Places** -> `VESSELS` (countries grouped into regions). Fewer, larger vessels read better than
   many small ones; tiny islands need a `buffer`.
3. **Amounts over time** -> the population tables (per vessel, per era) and the world total that sets
   the scale. Keep the compression exponent so the opening is a sparse scatter that fills.
4. **Events** -> founding flights, admixture pulses, replacements, recolouring (see the event types in
   `story.py` and `Sand.run_events`). Each event names source, destination, time window and share.
5. **Clock** -> map story years to video seconds (log-ish for deep time). Captions and narration are
   timed in video seconds.
6. **Concentrations** -> `MEGACITIES` / `CITY_HILLS` or the subject's equivalent (ports, capitals,
   language hubs). Keep the mountain rules in `story.py` (`CITY_H5`, `CITY_EXP`, `CITY_SLOPE`,
   `CITY_MIN_R`, `CITY_SHARE`, `CITY_RELIEF`).
7. **Camera** -> the opening diorama keys and the flight's list of places (25 in the first film, in a
   route that crosses the map's edge once).
8. **Title, sources note, narration script, music plan.**

Before a full export, check the end state quickly: run the sim to "today" once, pickle it, and iterate
on `cell_targets` against the pickle (a few seconds per try), plotting a top-down height map and 3D
relief of key regions.

## 3. Render and preview workflow

```bash
cd <film>/
python3 export_blender.py --no-hud                      # sim -> grains.npz, flights.npz, data/tour_heights.npy (~10 min)
/opt/blender/blender-4.2.3-linux-x64/blender -b -P blender_film.py      # -> out/blender_film/every_grain_of_us.blend
python3 export_blender.py --hud-only --scale 2 --workers 3              # 4K HUD .mov (~50 min)
python3 export_blender.py --captions-only --scale 2                     # captions .mov + SRT
blender -b -P blender_compose.py                                        # compose .blend, relative paths
```

The camera reads `data/tour_heights.npy` at import, so always export (which saves the settled
heights) before building the .blend or the HUD.

Previews: `scripts/quick_render.py` renders chosen frames of the Film scene on the CPU at 25% and 16
samples (~15 s a frame); `scripts/contact_sheet.py` tiles them with labels. Check: the opening, the
wide map at "today", 4–6 flight frames, the reveal. Composite a HUD frame over a preview to check
labels sit on their mountains.

## 4. Sound workflow

1. Write the music plan (sections = story beats, durations summing to the film's length in ms).
2. Generate 2–3 takes: `POST https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192` with
   `{"composition_plan": ..., "model_id": "music_v1", "respect_sections_durations": true}` and header
   `xi-api-key: $ELEVEN_LABS_API`. Run them one or two at a time (more returns 429).
3. Screen: `python3 scripts/screen_music.py take*.mp3` prints mean loudness per section and any
   dropout. Reject takes whose climax (the flight) is not the loudest stretch or that dip mid-flight.
4. Takes often end 3–5 s early: `python3 scripts/extend_score.py take.mp3 out.wav 190 212 4.0`
   repeats the best-matching ~4 s stretch from that window so the natural ending lands at the end.
5. Narration: one TTS request per line (Lily, `eleven_multilingual_v2`), files `line_00.mp3` ...
6. `python3 mix_audio.py SCORE LINES_DIR OUT_DIR` writes the mastered score, the narration stem,
   the narrated mix (WAV + MP3) and the SRT.
7. Give the user MP3s to listen to; they judge the music, not the meters.

## 5. The delivery package (what the user renders)

One folder, zipped:
```
every_grain_of_us.blend            render: Film scene, 4K, frames -> //render4k/frame_####.png
every_grain_of_us_compose.blend    assemble: frames + HUD + title + sound -> //every_grain_of_us_4k.mp4
hud.mov                            transparent 4K on-screen text
*_score_with_narration.wav         default sound
*_score.wav, captions_4k.mov       the music-only + captions cut (muted strips in the compose file)
narration.srt, captions_short.srt  YouTube subtitles
render4k/                          empty
README.txt                         quick start (extract, render, compose), GPU setup, two-PC rendering, speed
```
Before zipping, copy the folder, drop 30 stand-in 4K frames into `render4k`, render that range of
the compose file, and check every strip resolves and the MP4 has picture, HUD, title and sound.

The user's machine: Windows, Blender 4.4+, RTX 5080 (OptiX), sometimes two PCs sharing one network
folder (Blender's placeholder files let them split frames). 4K frames take ~15–22 s each on a 5080.

## 6. Gotchas

- **Towers vs mountains.** Fixed-footprint towers make height honest but look artificial. Mountains
  need a slope limit, a minimum base, irregularity, a sand budget per region (they borrow from the
  countryside), and spill cells beyond the glass for coastal places.
- **Spill cells** join their city's vessel for the mountain alone (not `VCELLS`/`VALID`); toppling
  onto them is allowed only where a mountain stands; relocation uses land + spill.
- **Settle the budget each frame** (countryside + mountains = the region's sand) or a region's plain
  inflates.
- **Vectorise per-frame work**: 640 mountains as flat arrays with a per-cell max (`np.maximum.reduceat`)
  cost what 66 did in a loop.
- **HUD size**: qtrle stores changed pixels; moving labels on every frame took a 4K HUD to ~4 GB.
  Hide region labels while the camera moves (~250 MB).
- **Projection**: guard points behind the camera and normalise by `abs(P[2,3])`, or labels vanish.
- **Motion blur across a cut**: put the cut at a half frame with quarter-frame camera keys.
- **Light keyframes**: keep them in time order (a flight start before the golden-hour ramp broke it).
- **Windows paths**: save compose files with `relative_remap=False`, or Blender rewrites absolute paths.
- **ElevenLabs auth**: the key is in `$ELEVEN_LABS_API`, sent as `xi-api-key` (the proxy only covers
  elevenlabs.io, not api.elevenlabs.io).
- **Voice clips**: the generator can leave a 15–30 ms click or a 6 kHz chirp after the speech, and
  some clips start or stop on a loud sample: `mix_audio.clean_line` handles both without moving lines.
- **Music**: early endings, mid-piece dropouts, and an "ambient intro" if you let it: ask to "open
  immediately" and list "silence" among the negatives.
- **Container restarts** kill long jobs: run them with `nohup`, log to files, and make steps resumable.
- **Downloaded data** (e.g. the UN city table came from a public GitHub LFS mirror): verify its
  checksum and spot-check values against known figures before use.
- **Headless Blender saves scenes muted** (`scene.use_audio = True` means Playback > Mute): the render
  still has its sound, but the timeline plays silently. Set `scene.use_audio = False` before saving
  (`blender_compose.py` does).
- **A quiet opening disappears under the narration.** The epic take 3 opened 15-20 dB below its body
  (even with "opens immediately" in the prompt), and ducked under the voice the music seemed to start at
  0:30 (it also began with 0.7 s of digital silence). Measure 2-second levels of the first 45 s of every
  take and trim any leading silence; `mix_audio.py` lifts a quiet opening (`ride_intro`, to about 4 dB
  under the body) after normalising, leaving the rest bit-identical. To fix an MP4 already
  rendered, swap the sound in without re-rendering:
  `ffmpeg -i film.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -shortest out.mp4`.
