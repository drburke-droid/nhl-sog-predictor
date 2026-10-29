# How fast is fast: a steel ball across a desert road (experiment)

One steel ball the size of a bowling ball (20 cm, 33 kg) crosses a real desert highway from left to right, faster each
time, from 50 mph to the fastest thing ever built (Parker Solar Probe, 430,000 mph). The speed is in the corner. Only
the speed changes, so everything the ball does to the scene comes from the speed.

Experiments so far (`../media/speed_ladder/`): `test1_roll_and_orbital_pass.mp4`, the 50 mph roll and the 17,500 mph
pass in real time; `test2_with_slowmo_replay.mp4`, the same followed by the slow-motion replay of the pass, with sound.

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

* Sun 50 deg right of the road, 20 deg up, soft (4 deg; thin cloud), warm; read from the rocks' shading and the lit
  face of the berm. 2.6 W/m^2 puts ~45% of the light on open ground in the sun, the rest from the sky.
* `env.py`: an environment map for fill and reflections built from the plate itself (directions outside the camera's
  view fold back into it).
* Rendered with the Standard view transform, so render values match the plate's.

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

`resume.py` re-renders only the frames missing from a saved .blend (renders get interrupted when the container restarts).
