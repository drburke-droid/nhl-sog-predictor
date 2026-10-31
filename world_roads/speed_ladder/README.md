# How fast is fast: history's fastest machines across a desert road (experiment)

One steel ball the size of a bowling ball (20 cm, 33 kg) crosses a real desert highway from left to right, faster each
time, from 50 mph to the fastest thing ever built (Parker Solar Probe, 430,000 mph). The speed is in the corner. Only
the speed changes, so everything the ball does to the scene comes from the speed.

Experiments so far (`../media/speed_ladder/`): `test1_roll_and_orbital_pass.mp4`, the 50 mph roll and the 17,500 mph
pass in real time; `test2_with_slowmo_replay.mp4`, the same followed by the slow-motion replay of the pass, with sound.

## History's fastest machines fly past (the current direction)

The ball is replaced by five machines from the history of speed, each flying across the road at its real top speed,
20 m out (behind the cone row), a few metres up, left to right:

| Machine | Speed | Height | Crosses the picture in |
|---|---|---|---|
| Wright Flyer, 1903 | 30 mph (its airspeed on the first flight) | 3.8 m | 3 s |
| Spitfire, 1943 | 370 mph | 4 m | 0.25 s |
| Bell X-1, 1947 | Mach 1.06: 807 mph at sea level (the record's 700 mph was Mach 1.06 at 43,000 ft) | 6 m | 0.1 s |
| F-35 Lightning II, 2016 | Mach 1.6: 1,228 mph at sea level (its top speed is flown high up) | 6 m | 80 ms |
| X-15, 1967 | 4,520 mph | 6 m | 20 ms |
| Parker Solar Probe, 2024 | 430,000 mph | 5 m | 0.2 ms |

Models: FlightGear's Wright Flyer, Spitfire IX, Bell X-1, F-35 (its F-35B, in the AF-01 test livery, an F-35A; gear
and lift-fan doors shut, the difference doesn't show from the side) and X-15 (GPL; Aircraft-2024 from the FlightGear
mirrors), and
NASA's Parker Solar Probe (public domain, nasa/NASA-3D-Resources).

* `ac3d.py`: a reader for FlightGear's AC3D models (objects, transforms, UVs, smoothing, textures, transparency) into
  Blender, y-up turned z-up.
* `aircraft.py`: each machine built ready to fly: imported, nose along +x, landing gear up (wheels, legs and the parts
  that only show with the gear down hidden; the X-1's gear doors kept shut), the Spitfire's propeller from its own file
  on its hub, each propeller on a pivot so it can spin (the Flyer's two pushers counter-rotating at 350 rpm, the
  Spitfire's at 1,430). Rocket exhaust for the X-1's four chambers and the X-15's XLR99: emissive volumes, brightest
  at the nozzle, with shock diamonds.
* `flight.py`: where each machine is at every moment (the Flyer undulating in pitch, as it did), which frames of its
  shot need it rendered (any part of it, or its shadow, in the picture), and the slow-motion replay's clock.
* `render_flyby.py`: Blender renders over the plate's camera and light with the props in place, so its shadow falls
  across the road, the cones and the drums. 1/100 s exposure (90-degree shutter); the propellers get 17 motion steps
  so they blur round. Only the part of the picture the machine and its shadow touch is rendered; the rest comes from
  one static frame (`make_fly.fly` feathers the seam).
* `fly_sound.py`: each machine heard from the camera. Its sound is carried from where it left to the camera at
  343 m/s, splatted onto the arrival clock, so the Doppler shift and its swell, and above the speed of sound the
  silence before the Mach cone, come out by themselves. The Flyer's 12 hp four-cylinder (34 firings a second, open
  exhaust ports), valve clatter, chains and propellers; the Spitfire's Merlin V12 (300 firings a second, every
  multiple of 25 Hz present), its stub exhausts and four-bladed propeller. In the replay the real-time sound is played
  back as slowly as the pictures (16 times slower is four octaves down: each propeller blade becomes a thump).
* `make_fly.py`: composites each shot (and its replay) and joins it to its sound.

The film: `../media/speed_ladder/history_fastest_machines.mp4` (80 s, ending on the Parker Solar Probe) and `history_fastest_machines_no_parker.mp4` (67 s, ending on the X-15). The shots on their own: `fly1_wright_flyer.mp4`, `fly2_spitfire.mp4`, `fly3_bell_x1.mp4`, `fly4a_f35.mp4`, `fly4_x15.mp4`, `fly5_parker_finale.mp4`.

Two more cuts (`make_cuts.py`): `history_realtime.mp4` (37 s), every machine at real speed one after another, no slow motion: a 180-degree shutter (1/50 s) so each smears as far as it travels while the shutter is open (the X-15 40 m, wider than the picture: a faint streak across the sky in one frame), how long each was in the picture (2.9 s, 0.25 s, 0.11 s, 0.089 s, 0.023 s), and the shake and boom scaled to the pressure at the camera (~6, ~12 and ~20 kPa); `history_pegged16.mp4` (55 s), every replay at the same 16 times slower, from the replay renders. Title on every cut: "120 years of speed, flying past you".

`history_slow48.mp4` (62 s, `REPLAY48=1 make_cuts.py slow48`): the Wright Flyer at real speed, then the Spitfire, X-1,
F-35 and X-15 only in slow motion, every one 48 times slower (a camera at 1,200 frames a second), from replay renders
at that rate (`flight.REPLAY_SUF`). The slow-motion sign (`slowmo.py`) sits in the black band above the picture: a
stopwatch and a clock that runs at real time (so it crawls: the X-15's pass takes 23 ms of it), how many times slower
than real ("48x SLOWER THAN REAL", a bar from real to 48x, the frame rate a camera would need), and while ramping
the rate with one decimal (1.3x, 8.4x); each slowdown slams a big "48x SLOWER / slow motion" across the picture, which
cools a touch while slowed. `history_realtime.mp4` carries the same band reading REAL SPEED.

`history_race.mp4` (23 s, `RACE=1 make_race.py`, timeline in `race.py`): the Wright Flyer, the Spitfire, the F-35
and the X-15 all clear the right edge of the picture on the same frame (frame 421, within 1/1,200 s), so they start
staggered by their speeds. The Flyer comes in at real speed; at about 90% of the way across everything slows to 48x
(frame 59-79); the Spitfire comes in (frame 95) and is about halfway across, the Flyer at about 95%, when the F-35
comes in (frame 272); then the X-15 (frame 368, 45 ms before the finish). Each has its own lane (`flight.RACE_LANES`,
metres out from the camera and up): the Flyer 13 m / 1.85 m, the Spitfire 22 m / 6 m, the F-35 32 m / 5.2 m, the
X-15 48 m / 3.3 m, so they never meet and stand apart at the finish; renders, dust and shadows are redone in those
lanes (`RACE=1`). Each is in the picture for 2.30 s, 0.272 s, 0.125 s and 0.045 s. The supersonic ones pass in
silence; after the finish, as it ramps back to real speed, the F-35's boom reaches the camera 7 ms later and the
X-15's blast 115 ms later, each shaking it by its pressure there (~8 and ~11 kPa, scaled from the 20 m shots as
distance^-3/4). The X-1 is not in the race (it would fit between the Spitfire and the F-35); neither is the Parker
probe (it would cross the whole picture in about 0.2 ms, a fifth of one frame even at 48x).

`history_race_parker.mp4` (28 s, `RACE=1 PARKER=1 OUTSUF=_pk make_race_parker.py`; renders and dust for the frames
that differ from the race's, `parker_race.sh`): the race with the Parker Solar Probe in it. The same frames as the race
until 7.5 ms before the finish; then everything slows again, to 1,000 times slower (a camera at 25,000 frames a
second, with its own "1,000x SLOWER" while it slows), the planes all but stopped in the last few per cent of the
picture, and the probe comes in at the left and crosses in 31 frames: all five clear the right edge on the same
1/25,000 s. Its lane is 100 m out and 12 m up (`flight.PARKER_LANE`): its blast grows ~16 m in its first millisecond,
so nearer it would reach the planes before the finish; at 100 m it reaches the X-15's lane ~20 ms after the finish
and the camera 56 ms after. As in the finale it meets the air at the left edge (as if it appeared there) and is a
plasma bolt within 5 m; the air it shoves aside slows it, 192 km/s coming in, 90 km/s at the right edge. After the
finish each frame is 4.7% longer than the last until the blast reaches the camera: the column white-hot, then cooling
(by the time the blast is down to Mach 3 its shock heats the air only to ~800 K, so the glow is gone within ~10 ms),
the dust it raises from the desert rolling toward the camera in daylight, and the lens hit by the blast and the dust
(Mach 3, ten atmospheres). The F-35's boom arrives on the way, the X-15's never does; the probe itself is silent until
its blast arrives, cut off with the camera. Its light on everything else (`make_race_parker.py`): the ground lit to
hundreds of times its daylight, the air round the column glowing with the light it scatters (4e-5 per metre, clear
desert air), the planes lit by the glowing ground and air round them; the camera's exposure follows what its sensor
records (log mean, highlights clipped), so the column blows out white and the rest comes down to where it can be seen.

Two corrections to the finale's shock model (`parker_comp.py`), which show only this far out and this late: the
blast's radius is B x sqrt(its age there) rather than a steady paraboloid (as the plasma slows, the age is more than
the distance behind the head over v0: 30 ms on, twice the radius); and the air behind the shock is as hot as the
shock's speed makes it (the ionising-gas fit above ~4.5 km/s, an ideal gas's Rankine-Hugoniot below), and clear once
it has cooled below ~4,500 K. The finale's own frames predate both; within its 1.7 ms they change little.

`history_race_light.mp4` (40 s; `RACE=1 PARKER=1 OUTSUF=_pk make_race_light.py clean` composites the frames once
into a near-lossless file, then `make_race_light.py` draws the signs over them): the same race with a pulse of light
in it. Light goes round the world (40,075 km) in 0.1337 s, 7.5 times a second; counting back 17 laps from a pass 8
frames (320 us) after Parker comes in lands on the frame the Wright Flyer comes in, so while the Flyer crosses the
picture, light goes round 17 times, and its last lap comes round with Parker in the picture. It flies 250 m out, behind everything, each lap 3 m above the last, and each pass leaves a faint line:
a ladder building in the sky (markers: clear air keeps no trace of light gone by). Light crosses the 384 m-wide
picture there in 1.3 us, so a pass falls in one frame or none; the shutter is open a quarter of each frame, and catches
4 of the 16 passes it does not freeze on, as a streak brightest at the right (seen from the camera the pulse seems to
rush in at 2.5 times the speed of light and leave at 0.6, so it lingers on the right). At its first pass and its last,
time stops and the pulse crosses a million times slower (a camera at 25 million frames a second), rushing in and
slowing as it goes; the sound stops with it. The last time, Parker has been streaking in at 1,000x for 8 frames and
stops dead a third of the way across, and light races past it along the top of the ladder: the fastest machine ever
built, standing still beside light. While the probe is lit, the light's glow and lines (stand-ins and markers, not
something a camera could record) join the picture at the daylight exposure (`make_race_parker.frame`: they still go
behind everything, dimmed and bent by all that lies in front, but are not darkened with the exposure the probe's glare
forces), so the pulse and the ladder stay in view. `make_race_light.py reclean <old frame>` composites again only the
frames a move of the last pass changes (`LT_FORCE_FROM=k`: every frame from k on as well), the rest kept from the
clean file. The glow stands in for what a pulse scatters toward the camera, and light
does not follow the curve of the Earth by itself (round the equator in a fibre, a lap would take 0.20 s).
The signs keep apart the two ways time is slowed. Slow motion (real speed, 48x, 1,000x) is amber, on a scale from real
speed to 1,000x, with what each is for ("so we can see the planes", "so we can see Parker"). Time stopped is light's
colour: the badge reads TIME STOPPED, a dark band across the picture with a pause sign says what it is for, and as it
ends a play sign says what the race goes back to (real speed after the first pass, 1,000x after the last). The frozen
stretches before and after each pass are held longer than the pulse needs (their frames repeat back and forth: in a
few microseconds nothing moves but the grain; the clock keeps its 40 ns a frame), so there is time to read. Between
the clock and the badge, light's laps of the Earth: one pip for each of the 17, the newest flaring as it comes round.
Captions in the band below the picture (`captions.py`) say what is going on, light's part in light's colour.

`history_light_film.mp4` (6 min 15 s, narrated; `RACE=1 PARKER=1 OUTSUF=_pk make_light_film.py`): the whole thing as one film,
told toward light, the subject that holds it together. "How fast is light? Let's work up to it." Part 1, real speed
(`history_realtime.mp4`); part 2, 48x slower (`history_slow48.mp4`); part 3, the race with Parker
(`history_race_parker.mp4`); each opened by a card, with captions that keep light in view: how many times faster it
is than each machine, how far each goes in a second against light's 186,282 miles. Then what light is, how it comes
into the race, and how to read part 4: its speed with the machines' as bars drawn to scale (scaled to Parker the planes
barely show; scaled to light Parker is 0.8 px), round the Earth 7.5 times in a second at real speed, the plan over a
frame of the race (the ladder, one line a lap), and how time will run in part 4, drawn from its own frames (real
speed, time stopped, 48x, 1,000x, time stopped, the finish). Part 4, the race with light. The ending: how long each
would take to reach the Moon, 332 days for the Wright Flyer down to 33 minutes for Parker, while a pulse goes there on
screen in the 1.28 s light takes. The cards' sound is the road's wind; under the black ones, a low quiet chord.
Narration: Lily (`vo_script.py`, every line as spoken and as subtitled; `vo_tts.py` says them with ElevenLabs' Lily
when a key is in the environment, `vo_eleven/`, the voice in the film (`VO_DIR`), or Kokoro-82M's bf_lily offline,
`vo_lily/`) explains it all for anyone:
what each part shows, how the race is timed, that light is circling the Earth and each line is one more lap, and that
TIME STOPPED is really a million times slower. She is silent whenever anything flies at real speed (part 1, the
Wright Flyer in part 2, the races' first seconds), so those keep their own sound. Each line goes at its moment, or
after the one before it ends (`make_light_film.py plan` lists them); the cards last as long as she needs; a part that
ends before its line does holds a card or a black frame; part 4's frozen stretches are held as long as her
explanation (`make_light_film.py holds`, passed to `make_race_light.py` as `LT_HOLDS` with `LT_CAPTIONS=0` and
`LT_NAME=history_race_light_vo`). The film's own sound dips 10 dB under her voice (4 dB on the cards), and the
subtitles are her words, a sentence at a time.

Shock waves bend the light (`fly_fx.displacement`) as optics says they must: each shock shell is the edge of a weak
cylindrical lens. The density jump across it (Rankine-Hugoniot, from the pressure at that radius) raises the air's
refractive index by dn = 2.6e-4 x (density ratio - 1); a ray that passes a shell of radius R at distance d from its
axis is turned toward the axis by 2 dn d / sqrt(R^2 - d^2) (the cone's slant drops out), averaged over each pixel's
footprint. That confines the bending to a pixel or two along each shell's edge, where rays graze it, instead of
tearing the picture.

### The X-1, the X-15 and the finale

* The X-1 (`flyby_dust.py`, `fly_fx.py`): silent until its Mach cone reaches the camera 18 ms after it passes; the
  boom (about 15 kPa under it, the shock and its reflection; ~3 kPa at the camera) lifts a sheet of dust off the sand
  and gravel along its path. It barely rocks the cone (about a degree: the pressure jump pushes on a small object only
  for the millisecond it takes to wrap round it). In the replay (32 times slower) its bow and tail shocks bend the light
  where rays graze them, and its rocket's hot trail shimmers.
* The X-15 (`hyper.py`): at sea level, Mach 5.9 means 2.45 MPa of dynamic pressure, about 20 times what it was built
  for; it is shown intact, its leading edges glowing (900-2,300 K by how squarely they face the flow, a blackbody's
  visible brightness). Its drag leaves 3 MJ in every metre of air: a line blast that reaches the ground 14 ms after it
  passes and raises a wall of dust (`flyby_dust.py`); the camera is hit 57 ms after it passes.
* The finale's shock (`parker_comp.r_shock`): a bow shock 20 cm ahead of the heat shield (behind a shock this strong
  the gas is squeezed ~15-fold, so the gap is about a fifth of the shield's radius), curving back as a paraboloid and
  then following the blast-wave law, r^2 = (B^2/v) x distance behind, the same shock spreading as the probe's energy
  pushes it out. Freshly shocked air glows in a thin shell (~60,000 K at the nose, set by how fast the shock moves into
  the air; ~8,000 K where the blast front has slowed to 7 km/s) round a cooler interior; partly transparent, so the
  shell glows brightest at its edges and the probe shows through it; the dust the blast raises fills its base; light
  bends where rays graze the shock.
* The finale (`parker.py`, `parker_render.py`, `parker_comp.py`): the probe cannot survive sea-level air at 192 km/s.
  It boils away within ~25 microseconds, 5 m after meeting the air, and its vapour carries on as a bolt of plasma,
  slowed only as it shoves air aside, leaving 6-9e10 J in every metre (0.6 kilotons across the picture). The blast
  reaches the camera 1.8 ms later at Mach 18. In real time: one white frame, then no signal. The replay (20,000 times
  slower, then ramping up) shows the bolt, the column of air it leaves at tens of thousands of kelvin widening behind
  the blast front (optically thick, as bright as its temperature), the camera's exposure falling a hundredfold so the
  daylit scene sinks to silhouettes lit by the fireball, the ground erupting along the front as it races toward the
  lens. End card: light is still 1,560 times faster.
* `make_film.py`: title, the five shots cut together, one mix that keeps the escalation.


## The physics the escalation follows

Drag on the ball is the energy it dumps into the air per metre it travels: 0.5 rho v^2 Cd A. It grows with the square of
the speed; the dust, the debris and the blast all come from it (sea-level air, Cd 0.47 below sound, about 0.9 above).

| Speed | As fast as | Crosses 60 m in | Energy per metre | |
|---|---|---|---|---|
| 50 mph | a fast car | 2.7 s | 5 J | it has to roll: in the air it would fall 35 m before reaching the far side |
| 767 mph | sound | 0.17 s | 2 kJ | sonic boom, a dust trail |
| 2,000 mph | a rifle bullet | 0.07 s (2 frames) | 15 kJ | dust blasted up beside the path |
| 17,500 mph | orbit (the ISS) | 8 ms (under a frame) | 1 MJ, ~1/4 kg TNT | glows like a meteor; blast radius ~3 m |
| 430,000 mph | Parker Solar Probe | 0.3 ms | 650 MJ, ~150 kg TNT | the whole scene flattened |

Blast radius here is sqrt(energy per metre / atmospheric pressure), the scale of a cylindrical blast. Above about
1,000 mph the ball crosses between two frames, so each speed plays in real time (you see only what it leaves) and then
as a slow-motion replay; the fastest needs about 290,000 frames per second, which real high-speed cameras reach.

## The plate

`plate/road_pexels_11276661.mp4`: a 6-second drone shot down a straight two-lane desert road (Pexels 11276661, uploaded
at 2732x1148, 25 fps). The drone drifts up to 50 px.

* `stabilize.py`: every frame warped onto frame 74 with the homography of the ground (fits to 0.1 px; the sky follows
  within ~1 px). A locked-off shot.
* `clean.py`: paints out the pickup driving along the far-left road (with the same patch from the last frame, where it
  has gone), crops to the region valid in every frame (2.39:1) and scales to 1920x804.
* `comp.plate_loop()` loops the 149 frames forever, cross-fading the last second into the first.

## Matching the camera

`measure.py` fits the painted edge lines and finds the centre-line dashes; `camera.py` solves the camera from them. Lines
on the ground parallel to the road meet at the horizon's vanishing point (939.2, 372.9); a ground line at lateral
offset X runs at slope X/H whatever the lens, so the edge lines (slopes -2.905, +2.425) and 7.3 m between them (two
3.65 m lanes) give the camera's height: 1.37 m. The lens (f = 1250 px on the plate, ~24 mm equivalent) only changes
perspective within an object, not where things sit or how fast they cross; it makes the dashes 3.2 m long with 6.6 m
gaps. Pitch 1.24 deg down, yaw -0.82 deg, roll 0. The projected lane lines sit on the painted ones.

## Light

* Sun 50 deg right of the road, 20 deg up, soft (5 deg; thin cloud), golden; read from the stones on the left
  shoulder (lit on their right faces, short soft shadows to the left) and the lit face of the berm.
* `env.py`: an environment map for fill and reflections built from the plate itself (directions outside the camera's
  view fold back into it).
* Calibrated against the plate: strips of asphalt (albedo 0.05) and road paint (0.65) rendered on the road beside the
  real ones (`calib.py`) came out 2-4 times too bright and too blue with that sky and a 2.6 W/m^2 sun. Now the sky
  as seen in reflections stays as the plate shows it, but the light it sheds on diffuse surfaces is scaled to 0.32
  and warmed, and the sun is 1.3 W/m^2, golden (1, 0.78, 0.58): CG paint now matches the real paint at 6-13 m, and
  the sun gives about half the light on open ground. The ground under the shadow catcher bounces light at the
  asphalt's and gravel's dark albedo. The asphalt itself shows no cloud shadow: the plate/CG ratio holds from 4 to 25 m.
* Rendered with the Standard view transform, so render values match the plate's. `comp.plate_match` then gives every
  render the plate's look: a touch of its softness, its black level (0.0025), and its noise, measured frame to frame
  on the plate (sigma = 0.006 sqrt(value)), so CG isn't cleaner than the footage around it.

## CG

* `scene.py`: the Blender scene (camera, shadow-catching ground, sun, environment, steel).
* `roll.py`: test 1, the ball rolling across 12 m out at 50 mph (motion blur, 180-degree shutter).
* `blast.py`: test 2, the 17,500 mph pass 0.6 m up, 12 m out: the one-frame streak of a glowing ball crossing in 3 ms
  inside a 1/50 s exposure, and gravel and sand clods thrown from the path (Newtonian particles bouncing on the ground).
* `dust.py`, `blast_dust.py`: the dust. Blender's smoke solver (Mantaflow) crashes in the Python build of Blender
  available here, so this is a solver and volume renderer of its own: stable fluids (semi-Lagrangian advection, Jacobi
  pressure, vorticity confinement) with buoyant hot gas and an expanding core (about 3 m^2 per metre of path, the volume
  1 MJ of heat takes at atmospheric pressure); dust as 1.2 million parcels thrown off the ground by the blast, faster
  and thicker in clumps, relaxing to the air (fine dust) or flying ballistic (sand), less off the asphalt than the sand;
  ray-marched from the plate's camera with the sun scattered forward (Henyey-Greenstein), shadowed by the dust between
  it and the sun, plus sky light, and the dust's shadow on the ground.
* `comp.py`: composites in linear light: dust shadow on the plate, CG over it, dust over that, then what a camera adds:
  the flash of the pass, bloom, the shake when the blast reaches the camera (35 ms after the pass at 12 m), the plate's
  softness and grain, the label. 1920x1080, the 2.39:1 picture letterboxed.
* `make_test.py`: assembles the test clip.

## Props, and the 50 mph ball falling (test 3)

* `motion.py`: the 50 mph ball is thrown from far left, so it comes into view already falling (it drops 3.3 m crossing
  the frame), hits the road at x = 0.8 m at 9 m/s, bounces (restitution 0.4, the hit takes 15% of its speed and sets it
  spinning) and clips the cone standing on its line at the top of its hop. Also the blast's timing and push for each
  prop (below), the sign bending and the pickup rocking.
* (The flybys have the road to themselves: the car, the cone and the sign looked pasted in, so they were taken out.
  The builders stay in `props.py`.)
* `props.py`, `layout.py`: traffic cones and a 100 km/h sign built here, oil drums (Poly Haven `barrel_03`, CC0,
  weathered blue steel), and a car parked on the right shoulder: the BMW 1M from Blender's benchmark scene (CC0),
  scaled to its real 4.38 m, badges hidden. (A pickup built here from boxes looked fake.) Assets and licences in
  `assets/`.
* The blast reaches each prop when the shock does (`motion.t_hit`); the impulse per area falls as 1/distance from the
  path, ~200 Pa s at 2 m (rough, for a 1 MJ/m line blast). Cones within ~1.5 m of the path leave at 20-35 m/s and fly
  30-50 m; those 3-6 m away topple and tumble 2-8 m; the nearest drum is thrown 17 m, the others roll. The sign's post
  bends flat (the force on its panel is ~50 times what the post can carry). The car, 1.5 t with its rear 3 m from the
  path, only lurches (~0.25 m/s) and rocks on its springs, but its windows blow in.
* `rigid.py`: the tumbling, simulated here (contact points on each body's hull, spring-damper ground, Coulomb friction,
  air drag, 0.2 ms steps). Blender's Bullet threw a kicked cone 100 m straight up however it was set up.
* `render_props.py`: Blender renders of the props (and the 50 mph ball) for each shot, with z-depth; motion blur from
  keys at the real exposure's times. Where nothing moves one frame is shared; for the replay's slowed part (2.5 ms,
  props moving under 3 cm) that one render carries normal, position and diffuse-colour passes so the compositor relights
  it with the glowing ball (`comp.relight`).
* The dust renderer stops each ray at the props' depth (`dust.ray_tmax`), so dust behind a prop doesn't veil it, and the
  glow is hidden behind props standing between the camera and the path.
* `roll_dust.py` and `chunks.py`: the 50 mph impact (a faint puff of road dust, asphalt grit, a scuff left on the road);
  the car's tempered glass glinting as it falls.
* `sound.py` adds the ball rushing in and its thud, the cone it clips, every landing of every thrown cone (plastic) and
  drum (a clang), the windows shattering, the sign's twang, each delayed by its distance to the camera.

## The slow-motion replay (17,500 mph)

`replay_time.py` holds the clock and the physics. The ball crosses the 20 m in view in 2.5 ms, so the replay runs at
1/2000 speed (a camera at 50,000 frames per second, 20 microseconds a frame) while it crosses, then ramps back to real
speed (each frame's step growing by the same factor, 20 us to 40 ms over 60 frames), then half a second at real speed.
The label shows how much slower and the time since the replay began.

* The blast along the path is a strong cylindrical shock, R = (E/rho)^(1/4) sqrt(age), E = 1.08 MJ per metre: 1 m behind
  the ball in 0.5 ms, until it slows to the speed of sound at 2 ms and 1.37 m; then a sound wave that reaches the
  camera at 33 ms, when the camera shakes.
* The ball: white-hot inside a plasma sheath trailing back like a teardrop (a 7000 K shock layer). Lit as a point light
  of 200 W/sr in units where the sun is 2.6 W/m^2 (the real one would be ~25x brighter): a pool of light runs along the
  road under it. Its wake is the core of hot air, cooling from white through yellow and orange to dull red within a
  few milliseconds, widening as the shock does.
* Sparks: droplets of steel shed by the ball (it loses about a millimetre crossing the scene), slowed by the air within
  a few centimetres, cooling over 0.1-3 ms.
* The shock bends light where a ray grazes its front: the plate is displaced across the front (7 px while strong,
  fading as it weakens); the hot core shimmers.
* `replay_dust.py`: the same dust parcels as the real-time pass, each leaving the ground when the shock reaches it (the
  ball's passage plus the shock's travel to that spot). Behind a strong shock the air moves at hundreds of metres a
  second, so the dust first jumps up to ~0.2 m in about a millisecond, raggedly, more in some patches than others,
  then rises at its launch speed; kinematic until 3 ms, then the air solver takes over. The ball lights it.
* `replay_comp.py`: the plate frozen while time is slowed (it advances in real time), bent by the shock, the dust's
  shadow and the ball's light on the ground, gravel (ballistic, bouncing; drawn here rather than in Blender, so it
  can be shown at any instant), the dust, the glow and sparks, bloom, shake, grain, labels.
* `make_replay.py` composites every frame in parallel, joins the replay to the test clip and adds the sound.
* `sound.py`: the sound, synthesized. Desert wind; the ball rolling past; the crack of the shock arriving 35 ms after
  the pass (an N-wave), the boom and rumble, gravel landing over the next second or two, dust hiss. In the replay the
  world is muffled: a low drone and the plasma's sizzle swelling as the ball crosses, near-silence as the shock travels
  toward the camera, then a slowed, deeper boom when it arrives (replay frame 179, 37 ms of real time).

## Run

Work files go to `/home/user/lot`. Blender runs as the `bpy` module (`pip install bpy`, Python 3.13):

```
ffmpeg -i plate/road_pexels_11276661.mp4 /home/user/lot/frames/f%03d.png
python stabilize.py && python clean.py && python measure.py && python env.py
bpy-python roll.py                     # 37 frames, ~15 s each
python blast_dust.py 60                # dust, ~7 s a frame at full size
bpy-python blast.py 100 32             # streak and debris, ~25 s a frame
python make_test.py
python replay_dust.py                  # 210 replay frames, 6-35 s each
python make_replay.py                  # composite (3 processes), join, sound
```

Test 3 (props): `python rigid.py roll && python rigid.py blast`, then `render_all_props.sh` (bpy, ~40 s a frame, 193
frames) and `pipeline3.sh` (dust with the props' depth, `make_test3.py`, the replay, sound).

The film told toward light: `parker_race.sh` (the race's renders and dust in Parker's lane), then with
`RACE=1 PARKER=1 OUTSUF=_pk`: `make_race_parker.py`, `make_race_light.py clean` (~45 min, 3 processes),
`make_race_light.py` (the signs, ~5 min) and `make_light_film.py` (the film, from the four cuts and its own cards).

`resume.py` re-renders only the frames missing from a saved .blend (renders get interrupted when the container restarts).
