# How the B-2 hides from radar

## The explainer (`explainer/`)

`../media/b2_stealth/b2-stealth-explainer.mp4`: 4:00, 1920x1080 at 30 fps, narrated, captions burned in
(`explainer/b2-stealth-explainer.srt` for platforms that take a subtitle file).

Ten scenes: the hook (172 ft of wingspan that reads like a bird on a scope), radar 101 (pulse, echo, delay to range),
radar cross section (B-52 ~100 m², B-2 ~0.1 m² or less: public estimates, the real figure is classified), the four
tricks (shape, edge alignment, absorbent skin, hidden engines), why it works (detection range goes as RCS^1/4, so
/10,000 echo is /10 range), the limits (long-wavelength early-warning radar; route planning, quiet emissions, night)
and a recap.

* `narrate.py`: says every line with Piper (`en_US-ryan-high`, MIT; set `PIPER_VOICE` to the .onnx) and lays out
  the timeline (`timeline.json`): each scene's lead-in, its lines with 0.45 s between them, its tail.
* `voice_check.py`: reads every clip back with Whisper (small.en) to catch mispronunciations; four lines were
  re-taken this way (see its docstring), so running `narrate.py` again replaces clips that were checked.
* `scene.html`: every scene drawn on a canvas as a function of time (`render(t)`), cued off the narration's line
  times; the captions are drawn from the same timeline. `sfxEvents()` lists the sound cues (radar pings, echoes,
  swells) at the times the picture makes them.
* `render.js`: Playwright drives the page frame by frame into ffmpeg (`node render.js video f0 f1 out.mp4`, four
  chunks in parallel, ~14 min); `stills` for spot checks, `events` writes `sfx.json`.
* `mix.py`: the soundtrack: the narration at its timeline positions, a quiet ambient pad ducked under the voice, the
  synthesized sound cues. Mastered to -16 LUFS (two-pass loudnorm) when muxed.

```
python narrate.py                     # clips + timeline.json
node render.js events sfx.json
for i in 0 1 2 3; do node render.js video $((i*1798)) $(((i+1)*1798)) part$i.mp4 & done; wait
python mix.py                         # mix_raw.wav
ffmpeg -f concat -i parts.txt -c copy video.mp4   # then mux with mix_raw.wav through loudnorm
```

## Flybys with a real B-2 model (`flyby/`)

The B-2 flies past at real speed over real photographs, the way the Wright Flyer and the rest cross the desert road
in the speed-ladder film (`speed_ladder/`, branch `claude/wonderful-ritchie-uyd31i`): CG rendered in Blender over the
photograph's camera and light, its shadow on the ground, motion blur from a 1/100 s shutter, the speed in the corner,
its sound carried to the camera at the speed of sound. Picture 2.39:1 letterboxed in 1920x1080.

**The model**: FlightGear's B-2 Spirit by Markus Zojer (FGAddon, GPL v2+; `B-2.zip` in the Aircraft-2024 set on the
FlightGear mirrors), read into Blender by `ac3d.py` (from the speed ladder). `b2_model.py` builds it: gear up and every
door shut (the model's rest state), gear legs, tyres, lights, instruments and cockpit interior left out, the
flight-sim livery (white walkway outlines, "0088" serials, logos and credits in the textures) toned to 12% of its
contrast so the panel lines read only as faint shifts in the gray, satin paint, smoothed within surfaces and sharp
where panels meet. Real size: 52.2 m span, 21.2 m long. `look.py` renders it from five sides.

**Top-down over Whiteman AFB** (`topdown.py`; `../media/b2_stealth/b2_flyby_topdown_whiteman.mp4`, 6.2 s): the
B-2's home base, from the USDA's NAIP aerial photograph of 28 June 2022 (0.6 m, public domain; `fetch_naip.py`
reads the window out of Microsoft Planetary Computer's cloud-optimized copy). The picture is the flight line: the
runway, the infield, the row of B-2 hangar docks and the ramp, with a real B-2 parked on it. The camera hangs 900 m
up looking straight down (22.5 mm, 77 degrees across: the ground at 0.75 m a pixel, the photograph's own detail); the
B-2 passes 166 m below it, 734 m (2,400 ft) up, at 300 mph, heading WNW, straight away from the sun, so its shadow
runs 390 m ahead of it: the shadow glides over the infield first, the aircraft crosses in about two seconds and
overtakes its shadow on the screen (it is five times nearer the camera). The sun is the photograph's (117 degrees,
62 up: the hangars' shadows), the shadow is the mesh projected along it onto the ground with the sun's soft edge
(3.9 m), and the render's colour is matched to the real B-2 on the ramp (a nadir calibration render; mean sRGB 141,
146, 163), then desaturated a fifth.

**Sound** (`flyby_sound.py`): four F118 turbofans: the fans' whine (blade-passing tone, buzz-saw tones at shaft
multiples) loudest ahead, the jets' roar loudest 120-150 degrees aft, each sample carried to the camera at 343 m/s
(Doppler and swell by themselves), air absorption by range, a ground reflection for cameras on the ground.

```
sh flyby/fetch_assets.sh
cd flyby && python topdown.py plate && python topdown.py prep && python topdown.py render \
  && python topdown.py comp && python flyby_sound.py topdown && python topdown.py mux
```

**Side pass over the Bonneville Salt Flats** (`side.py`; `../media/b2_stealth/b2_flyby_side_bonneville.mp4`, 7.5 s):
the plate is the BLM's photograph of the flats and the Silver Island Mountains (public domain). The camera is matched
to it: level (the salt's far edge runs level, and on a flat plain that is eye level), ~30 degrees across (the
mountains' height for their distance, and the salt crust's size at the foot of the picture), the sun low (~15
degrees), right of and a little behind the camera (the crust's shadows, the lit faces of the range). The B-2 goes by
flat out, 630 mph (its published top speed), 65 ft up and 350 m out, crossing the picture in under a second, banked 20
degrees toward the camera (seen level from the ground a B-2 is edge-on, a sliver; banked, its planform reads), 1/200 s
shutter, its paint corrected by the top-down shot's calibration. The shot holds after it has gone while the roar
arrives (~1 s later) and the salt rolls up.

**The salt it tears up** (`debris.py`): at that height (under half its span) it rides in ground effect: ~150 tonnes of
lift pressing on the air between it and the crust, the air squirting out sideways and back, wingtip vortices hitting
the ground at once, the jets' wash coming down behind. A parcel model shaped to that, not a fluid simulation (the
speed ladder has one): 2,600 puffs of dust, each a clump of 24 sub-parcels so the cloud billows, launched from under
the track just behind the trailing edge, outward, up (a rooster tail and the vortices' columns go higher) and a little
forward, relaxing to a light breeze, rising while they mix, spreading, heavier grains settling; 2,600 crust chips
(20-80 cm) and 7,000 bits of grit thrown ballistically (up to ~15 m) and falling back. Each frame they are projected
through the plate's camera and splatted as soft blobs into optical depth, in front of the track and behind it (the
aircraft goes between), lit like the plate's own sunlit salt, bluer and darker where thick. The sound adds the salt:
a crunch and hiss from each stretch of ground as the aircraft passes over it and chips pattering down, each heard
after its travel time to the camera.

Composited on the photograph's 1280 px rendition (Wikimedia rate-limited the original during this session):
`SIDE_PLATE=bonneville_preview_1280.jpg python side.py comp`. With the 7198 px original in
`$B2_WORK/plates/bonneville_blm_28776016261.jpg`, `python side.py plate && python side.py comp && python side.py mux`
redoes the picture; the renders don't change (same framing).
