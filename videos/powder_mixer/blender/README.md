# The Mix-Off: render it in Blender (4K, Cycles, GPU)

Everything needed to render the finished film is in this folder. Nothing has to
be simulated or installed besides Blender.

| File | What it is |
|---|---|
| `shot_Q1.blend` ... `shot_Q4.blend` | The four quarter-finals |
| `shot_S1.blend`, `shot_S2.blend` | The two semi-finals |
| `shot_F1.blend` | The final |
| `shot_X6.blend`, `shot_X5.blend` | The closing bowls (squares and smiley; logo and YouTube icon) |
| `the_mix_off_MASTER.blend` | Assembles the film: rendered frames + overlay + music |
| `overlay_multiply.mp4`, `overlay_add.mp4` | The scores, charts, captions and cards, as two overlay videos |
| `score.m4a` | The music |
| `render_all.bat`, `render_all.sh`, `use_gpu.py` | Optional: render everything in one go |

Keep all of them in the same folder.

## Steps

1. Install **Blender 5.3 or newer** (the files were saved with 5.3).
2. Turn on the GPU once: Edit > Preferences > System > Cycles Render Devices,
   pick OptiX (NVIDIA), HIP (AMD) or Metal (Mac) and tick your card.
3. Open each `shot_*.blend` and choose Render > Render Animation (Ctrl+F12).
   Frames are written to `frames/Q1/`, `frames/Q2/` and so on, next to the files.
4. Open `the_mix_off_MASTER.blend` and choose Render > Render Animation.
   It writes `the_mix_off_4K.mp4` (3840x2160, 24 fps, 3:54, with the music).

Or, instead of steps 3 and 4, double-click `render_all.bat` (Windows; edit the
Blender path at the top if yours differs) or run `./render_all.sh`.

Rendering can be stopped and restarted at any time: frames that already exist
are skipped. To redo a frame, delete its file.

## What is set up

- **Size.** Matches render at 3840x1680 (the picture area above the score
  strip), the closing bowls at 3840x2160. 3,231 frames in all.
- **Cycles.** GPU, up to 128 samples with adaptive sampling (noise threshold
  0.025), OpenImageDenoise with albedo and normal guides, caustics off, indirect
  light clamped. If it is too slow, lower Render > Sampling > Max Samples to 64
  before touching the resolution; the denoiser copes well.
- **Self-contained.** The grain animation is packed inside each shot file as a
  data texture (one row per frame, one pixel per grain), read by the Geometry
  Nodes modifier on the grains. There are no external caches or textures.
- **Metal.** Chrome, brushed steel (the trim band), turned steel (shaft and hub)
  and polished steel (tools) are separate materials with varying roughness; the
  enamel has a clear coat. Three reflection cards and a studio environment,
  visible only in reflections, give the metal something to reflect.

## Things you may want to change

- **The maker's name.** The trim band and the hub cap carry the lettering that
  came with the mixer model. To remove it, open the materials `BandLettering`
  and `CapLettering` in a shot file and set the value node named "Lettering"
  to 0. It has to be done in each of the seven match files.
- **1080p.** The master file expects 4K frames. If you want 1080p, ask for a
  1080p set rather than changing the resolution here.
- **Colours.** Each mixer's paint is the material `LEnamel` or `REnamel`.
- **Focus.** The camera focuses on the bowls, so the mixer heads are slightly soft, as in a
  photograph. For a sharper head, raise the camera's Depth of Field > F-Stop (or untick Depth of Field).

## How the overlay works

The overlay is not baked into the renders. The master file lays two ordinary
videos over the frames in the Video Sequencer: `overlay_multiply.mp4` (blend
mode Multiply: how much of the picture shows through) and `overlay_add.mp4`
(blend mode Add: the overlay itself). Together they act as an alpha overlay.

## Not checked

These files were built and test-rendered on a CPU. The GPU path (device
selection, GPU denoising) has not been run, and there is no render-time
estimate for your card: render ten frames of one shot first and multiply up.
