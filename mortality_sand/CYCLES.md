# 2,000 years of catastrophe: the 4K Cycles project

This is the film as a Blender project, to render in Cycles at 3840×2160. Every visible grain of sand is a lit sphere standing in the glass vessels on the map table. The grains lifted away are glowing spheres rising out of the frame. The sand, text and voiceover match the painted film frame for frame.

The render settings are the migration film's ("Every grain of us"):

| Setting | Value |
|---|---|
| Resolution | 3840×2160, 30 fps |
| Device | Cycles on the GPU |
| Sampling | 64 adaptive samples, noise threshold 0.04 |
| Denoising | OpenImageDenoise, using albedo and normals |
| Light bounces | trimmed: 6 in all, 1 diffuse, 3 glossy, 6 through glass |
| Caustics | off |
| Noise seed | fixed |
| Grade | AgX High Contrast, exposure −0.15 |
| Motion blur | half-frame shutter |
| Lens | f/0.9 |
| Compositor | bloom and a vignette |
| Sand | each grain a little different in hue, shade and shine |
| Output | PNG frames in `render4k/` |

## What is here

| File | What it is |
|---|---|
| `two_thousand_years_of_catastrophe.blend` | The project. It has two scenes, `Film` and `Edit`. |
| `cache/` | The sand for every frame, and every lifted grain, read by the script inside the .blend. |
| `overlay_fill.mp4`, `overlay_matte.mp4` | The film's text, titles and labels in 4K, as a lossless layer for `Edit`. |
| `narration.m4a` | The voiceover. |

Keep everything together in one folder. Every path inside the .blend is relative to it. You need Blender 4.2 or newer. For an NVIDIA RTX 50-series card (5080, 5090), you need Blender 4.4 or newer.

## 1. Turn on your graphics card (once)

In Edit → Preferences → System → Cycles Render Devices, choose:
- OptiX for NVIDIA RTX cards;
- CUDA for older NVIDIA cards;
- HIP for AMD;
- Metal for Apple Silicon.

Then tick your card. If nothing is ticked, Cycles renders on the processor, which is about 20 times slower.

## 2. Render the frames (the long part)

1. Open `two_thousand_years_of_catastrophe.blend`.
2. When Blender says *"automatic execution of Python scripts in this file was disabled"*, click **Allow Execution**.
   - The script `sand_loader.py`, stored in the file, fills the grains for each frame from `cache/`. Without it there is no sand.
   - To trust the file every time instead: Preferences → Save & Load → **Auto Run Python Scripts**.
3. In the `Film` scene, choose Render → Render Animation (Ctrl+F12).
   - The frames go into a `render4k` folder next to the .blend, from `frame_0000.png` to `frame_8273.png`.
   - Leave about 85 GB free, about 10 MB a frame.
   - You can stop at any time (Esc, or close Blender) and start again: finished frames are skipped.

From the command line (`-y` allows the script; use CUDA, HIP or METAL in place of OPTIX to match your card):

```
blender -b two_thousand_years_of_catastrophe.blend -y -S Film -a -- --cycles-device OPTIX
```

**Two computers:** both must write into the same folder. Each then sees which frames the other has finished or started, because it leaves an empty placeholder file.
1. Share the first PC's project folder on the network.
2. On the second PC, run one line, using the first PC's IP address:

```
blender -b "\\192.168.1.50\cycles\two_thousand_years_of_catastrophe.blend" -y -S Film
  -o "\\192.168.1.50\cycles\render4k\frame_####" -s 4137 -a -- --cycles-device OPTIX
```

`-s 4137` starts it halfway through. Run it again without `-s` at the end to pick up whatever is left.

## 3. Assemble the film

When all 8,274 frames are done, switch to the `Edit` scene and choose Render → Render Animation.
- It lays the 4K text over the frames, adds the narration, and writes `two_thousand_years_of_catastrophe_4k.mp4`.
- The film runs 4:59. After the last rendered frame (`frame_8273.png`), the `Edit` scene holds it for 23 seconds
  under the closing words and the closing card, then fades to black. There is nothing more to render.
- This takes a few minutes.
- From the command line: `blender -b two_thousand_years_of_catastrophe.blend -y -S Edit -a`

## Before the long render

A frame in 1942, the busiest stretch, has 175,000 grains of sand, the glass and motion blur. At these settings in 4K it took 578 seconds (about 10 minutes) here on 4 CPU cores. A recent GPU is many times faster.

Try a few frames first. In the frame numbers below, Blender's frame N is the film's frame N:

| Frame | Year | What is happening |
|---|---|---|
| 511 | 541 | Plague of Justinian |
| 1350 | 1348 | Black Death |
| 1785 | 1520 | The Americas after 1492 |
| 3064 | 1812 | Napoleon in Russia |
| 4612 | 1916 | First World War |
| 6007 | 1942 | Stalingrad |
| 6302 | 1944 | Bengal famine, end of the Second World War |
| 6701 | 1960 | Great Leap Forward famine |
| 7663 | 2021 | COVID-19 |

```
blender -b two_thousand_years_of_catastrophe.blend -y -S Film -s 6007 -e 6037 -a -- --cycles-device OPTIX
```

**A quick 1080p draft:** set Output Properties → Resolution to 50%. The `Edit` scene scales draft frames up to 4K.

## What you can change

- **Look**
  - Materials: `Sand grain` and `Lifted grain`.
  - Lights: `key`, `fill`, `rim` and `top`.
  - Bloom and vignette: the `Film` scene's compositor.
  - Grade: Color Management.
- **Streaks and glow:** constants at the top of `sand_loader.py`, in the Text Editor:
  - `STREAK`: how long the rising grains' motion-blur streaks are.
  - `GLOW`: how brightly the rising grains shine.
  - `MIN_PX`: their smallest size on screen.
- **Camera:** this is the painted film's wide shot.
  - The text layer is drawn for this shot, so labels sit over their places only while the camera stays put.
  - For another shot, render `Film` without the text layer.

## How it was made

From the repository's `mortality_sand/` folder:

```
python export_cycles.py               # runs the simulation (sand.py): cache/ and the 4K text layer
blender -b -P blender_cycles.py       # builds two_thousand_years_of_catastrophe.blend
```
