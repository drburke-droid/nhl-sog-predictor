EVERY GRAIN OF US: the full-Cycles Blender project
=================================================

What is in this folder
  every_grain_of_us.blend   the whole film (2:58, 5,340 frames at 30 fps), two scenes:
                              Film  table, glass, sand and flying sand, rendered by Cycles
                              Edit  joins the rendered frames, the on-screen text and the title
                                    into an MP4 (add the voice-over here too)
  hud.mov                   the on-screen text (year, "1 grain = N people", legend, labels),
                            transparent. Keep it next to the .blend.

You need Blender 4.2 or newer (free from blender.org). A graphics card makes a huge difference.


1. TURN ON YOUR GRAPHICS CARD (once)
   Edit > Preferences > System > Cycles Render Devices
     NVIDIA RTX: OptiX     older NVIDIA: CUDA     AMD: HIP     Apple Silicon Mac: Metal
   Tick your graphics card. (If nothing is ticked, Cycles renders on the processor, ~20x slower.)


2. RENDER THE FRAMES (the long part)
   Open every_grain_of_us.blend. The "Film" scene is open.
   Render > Render Animation (Ctrl+F12).
   Frames are written to a "render" folder next to the .blend (frame_0000.png ... frame_5339.png).

   You can stop at any time (Esc, or close Blender) and start again: finished frames are skipped.
   Several computers can render at once if they share the folder (each skips frames another has started).

   Command line (no window, a little faster):
     blender -b every_grain_of_us.blend -S Film -a -- --cycles-device OPTIX
   (use CUDA, HIP or METAL instead of OPTIX to match your card)


3. MAKE THE MP4
   Switch to the "Edit" scene (scene menu, top right of the window).
   Render > Render Animation. Out comes every_grain_of_us.mp4 next to the .blend (a few minutes).

   Change the title: in the Video Sequencer, click the "Title" strip, press N, and edit the text under
   Effect Strip. The subtitle is the "Subtitle" strip.
   Add the voice-over: Add > Sound, on a free channel, starting at frame 0. It is exported with the MP4.


SPEED AND QUALITY
   The project is set for good quality at a sensible speed:
     up to 64 samples per pixel, stopping early wherever the image is already clean (noise threshold 0.04),
     then denoised (OpenImageDenoise, using albedo and normals)
     light bounces trimmed to what glass and sand need (6 total, 6 through glass)
     no caustics, no motion blur, a fixed noise pattern so the still camera does not shimmer
     the sand as a point cloud (spheres traced directly, no meshes), only the grains that can be seen
   Before the full run, test one frame: go to a frame (e.g. 4600, among the towers) and press F12.

   To go faster:  Render Properties > Sampling > Render > Max Samples 32 (a little noisier), or
                  Output Properties > Resolution 50% for a draft of the whole film.
   To go cleaner: Max Samples 128 and Noise Threshold 0.02.

   Rough time per frame at the project settings, 1920x1080:
     4-core processor, no GPU     about 2 minutes        (about 7 days for the film)
     Apple M2/M3 Pro GPU          about 12 seconds       (about 18 hours)
     NVIDIA RTX 3060/4060         about 6 seconds        (about 9 hours)
     NVIDIA RTX 4080/4090         about 2-3 seconds      (about 3-4 hours)
   (Estimates; frames among the towers near the end are the slowest.)


THE STORY ON SCREEN
   The camera holds the whole map still for the first 2:23, then flies low among the population
   towers of Asia, across to Europe and Africa, and pulls back out for the closing ripple of light
   from East Africa. The timed script for the voice-over is grains_of_ancestry_narration.srt.
