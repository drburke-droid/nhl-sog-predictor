# Powder mixing on a conveyor (Blender)

A looping, "how mixed is it?" animation: six dried powders (tiny cubes) sit in
lanes in a glass dish, ride a conveyor under a twin-paddle stand mixer, get
stirred for about two seconds, leave, and come round again. Each pass mixes
them a little more; an on-screen counter shows the pass number and a measured
"% mixed".

Nothing here is hand-animated. The grains are a real granular simulation, and
the percentage is measured from the simulated grains.

## Files

| File | What it does |
| --- | --- |
| `simulate.py` | Granular physics (position-based dynamics with friction, numba). Writes `cache/particles.pc2` plus `cache/meta.npz` / `meta.json` (animation channels, mixing curve). |
| `build_scene.py` | Builds the whole Blender scene from the cache (conveyor, glass dish, mixer, lights, camera, overlay), saves a `.blend`, and can render stills or the animation with Cycles. |

`cache/`, `renders/` and `.blend` files are generated, so they are git-ignored.

## Quick start

```bash
pip install bpy numba numpy          # or use Blender's own Python for build_scene.py
```

```bash
python simulate.py                   # ~64k grains, 10 passes, about 15 min on 4 cores
```

```bash
python build_scene.py --save mixer.blend --still 1 300 1100
```

```bash
python build_scene.py --anim --res 3840x2160 --samples 96 --gpu
```

```bash
ffmpeg -framerate 24 -i renders/frames/f_%04d.png -c:v libx264 -crf 16 -pix_fmt yuv420p mixer.mp4
```

Inside Blender instead of the `bpy` module:

```bash
blender -b -P build_scene.py -- --save mixer.blend
```

Then open `mixer.blend` and render as usual. The grains are one mesh of points
with a Mesh Cache modifier (the `.pc2` file, path stored relative to the
`.blend`) and a Geometry Nodes modifier that puts a cube on every point.

## Useful knobs

`simulate.py`

- `--grain 1.5` grain size in mm. Smaller is more powder-like and slower:
  1.5 is about 64k grains, 1.2 about 125k, 1.0 about 215k.
- `--passes 10` number of trips through the mixer.
- `--rev 1.0` paddle speed (rev/s) and `--t-mix 1.6` seconds under the mixer.
  Lower either one for a gentler mix that needs more passes.
- `--ingredients 6` number of lanes (up to 8 colours in the palette).
- `--alternate` reverse the paddles every pass. This mixes noticeably worse,
  because the reverse stroke partly undoes the previous one.
- `--set KEY=VALUE` override any geometry or timing value, e.g. `--set paddle_r=24`.

`build_scene.py`

- `--res 3840x2160`, `--samples 96`, `--gpu`, `--motion-blur`
- `--no-hud` drop the pass / % mixed overlay
- `--elev`, `--azim`, `--dist`, `--aperture` camera placement and depth of field
- `--start`, `--end` render a frame range. Rendering never overwrites existing
  frames, so it is resumable and several processes can share one job.

Colours are the `PALETTE` list at the top of `build_scene.py`.

## What "% mixed" means

The dish is cut into 7.5 mm sample columns. For each ingredient we take the
standard deviation of its share across the samples:

- `s0`: fully separated (where the video starts)
- `sr`: a perfectly random mix, which still varies a little from scoop to
  scoop because a scoop holds a finite number of grains
- `s`: now

`mixed = (s0 - s) / (s0 - sr)`, averaged over the six ingredients. 0% is the
untouched lanes; 100% means samples are indistinguishable from a random
shuffle. This is the standard-deviation form of the Lacey mixing index.
