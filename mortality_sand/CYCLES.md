# 2,000 years of catastrophe: the Cycles project

This folder holds the film as a Blender scene, to render in Cycles.

- **Everything is real geometry:** every visible grain of sand is a lit sphere, standing in the glass vessels on the map table. The grains lifted away are glowing spheres rising out of the frame.
- **Light is computed:** shadows, glass and glow are worked out by Cycles, not painted on.
- **The rest is the same as the painted film:** the same sand, frame for frame, the same text and the same voiceover.

## What is here

| File | What it is |
|---|---|
| `film_cycles.blend` | The project, for Blender 4.2 or later. It has two scenes, `Sand` and `Edit`. |
| `cache/` | The sand for every frame, and every lifted grain, read by the script inside the .blend. |
| `overlay_fill.mp4`, `overlay_matte.mp4` | The film's text, titles and labels, as a lossless layer for the `Edit` scene. |
| `narration.m4a` | The voiceover. |

Keep these files together. The paths inside the .blend are relative to the folder.

## Render it

1. **Open `film_cycles.blend` and allow scripts.**
   - Blender shows a bar: *"For security reasons, automatic execution of Python scripts in this file was disabled"*. Click **Allow Execution**.
   - To trust it every time instead: Preferences → Save & Load → **Auto Run Python Scripts**.
   - The script `sand_loader.py`, stored in the file, fills the grains for each frame from `cache/`. Without it the scene has no sand.
2. **Use your GPU.** In Preferences → System → Cycles Render Devices, choose CUDA, OptiX, HIP or Metal, and tick your card. The scene is already set to render on the GPU; with no GPU set up there, it renders on the CPU.
3. **Render the frames.** In the `Sand` scene, choose Render → Render Animation.
   - It writes 8,274 PNG frames to `frames/`, at 1920×1080 and 30 fps.
   - Finished frames are kept. If you stop, rendering again carries on where it left off.
   - Several computers can share the `frames/` folder: each skips the frames another has claimed.
4. **Put the film together.** Switch to the `Edit` scene, then Render → Render Animation.
   - It lays the text over the frames, adds the narration, and writes `two_thousand_years_of_catastrophe_cycles.mp4`.
   - This step takes a few minutes.

From a terminal (`-y` allows the script):

```
blender -b film_cycles.blend -y -S Sand -a                 # the frames
blender -b film_cycles.blend -y -S Sand -s 6000 -e 6030 -a # a stretch: Stalingrad, 1942
blender -b film_cycles.blend -y -S Edit -a                 # the finished film
```

## Before the long render

The full film is 8,274 frames. A frame in 1942, the busiest stretch (175,000 grains of sand, glass, motion blur), took 100 seconds here on 4 CPU cores at 32 samples, 1920×1080. A recent GPU is usually 5 to 20 times faster than that. Try a few frames first. The busiest moments are:

| Frame | Year | What is happening |
|---|---|---|
| 512 | 541 | Plague of Justinian |
| 1351 | 1348 | Black Death |
| 1786 | 1520 | The Americas after 1492 |
| 3065 | 1812 | Napoleon in Russia |
| 4613 | 1916 | First World War |
| 6008 | 1942 | Stalingrad |
| 6303 | 1944 | Bengal famine, end of the Second World War |
| 6702 | 1960 | Great Leap Forward famine |
| 7664 | 2021 | COVID-19 |

Ways to make it faster:
- **Samples:** Render Properties → Sampling. The default is 128 adaptive samples, with OpenImageDenoise. 64 is usually enough.
- **A draft:** set Output Properties → Resolution to 50%. The `Edit` scene scales draft frames up to 1080p.

## What you can change

- **Look**
  - Materials: `Sand grain` and `Lifted grain`.
  - Lights: `key`, `fill`, `rim` and `top`.
  - Glow and vignette: the `Sand` scene's compositor.
  - Colour: view transform AgX, Medium High Contrast.
- **Streaks and glow:** constants at the top of `sand_loader.py`, in the Text Editor:
  - `STREAK`: how long the rising grains' motion-blur streaks are.
  - `GLOW`: how brightly the rising grains shine.
  - `NEAR`: how close to the camera a rising grain can come before it is dropped.
- **Camera:** this is the painted film's wide shot.
  - The text layer is drawn for this shot, so labels sit over their places only while the camera stays put.
  - For a different shot, render the `Sand` scene without the text layer.

## How it was made

From the repository's `mortality_sand/` folder:

```
python export_cycles.py               # runs the simulation (sand.py) and writes cache/ and the text layer
blender -b -P blender_cycles.py       # builds film_cycles.blend
```
