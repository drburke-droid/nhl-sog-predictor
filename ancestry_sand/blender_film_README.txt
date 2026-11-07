EVERY GRAIN OF US: the full-Cycles Blender project (4K)
======================================================

What is in this folder
  every_grain_of_us.blend          RENDER: the whole film in Cycles (3:42, 6,660 frames at 30 fps,
                                   3840x2160), into a "render4k" folder here
  every_grain_of_us_compose.blend  ASSEMBLE: joins the frames, the on-screen text, the title and the
                                   sound into the finished MP4
  hud.mov                          the on-screen text in 4K (year, "1 grain = N people", legend, place and
                                   city names), transparent
  every_grain_of_us_score_with_narration.wav   music + Lily's voice-over (used by default)
  every_grain_of_us_score.wav                  music only (for the captions version)
  captions_4k.mov                  short captions, transparent (for the music-only version)
  narration.srt, captions_short.srt   subtitle files for YouTube
Keep everything together in this folder.

You need Blender 4.2 or newer (free from blender.org); for an NVIDIA RTX 50-series card (5080, 5090),
Blender 4.4 or newer. A graphics card makes a huge difference.

This version changes every frame: render it into a fresh folder (do not mix in frames from earlier
versions).


1. TURN ON YOUR GRAPHICS CARD (once)
   Edit > Preferences > System > Cycles Render Devices
     NVIDIA RTX: OptiX     older NVIDIA: CUDA     AMD: HIP     Apple Silicon Mac: Metal
   Tick your graphics card. (If nothing is ticked, Cycles renders on the processor, ~20x slower.)


2. RENDER THE FRAMES (the long part)
   Open every_grain_of_us.blend. The "Film" scene is open.
   Render > Render Animation (Ctrl+F12).
   Frames are written to a "render4k" folder next to the .blend (frame_0000.png ... frame_6659.png).
   Leave about 70 GB free for them (about 10 MB a frame).

   You can stop at any time (Esc, or close Blender) and start again: finished frames are skipped.

   Command line (no window, a little faster):
     blender -b every_grain_of_us.blend -S Film -a -- --cycles-device OPTIX
   (use CUDA, HIP or METAL instead of OPTIX to match your card)

   TWO COMPUTERS: both must write into the same folder, so each can see which frames the other has
   finished or started (it leaves an empty placeholder file). Share the first PC's project folder on
   the network, then on the second PC (Command Prompt, one line; use the first PC's IP address):
     blender -b "\\192.168.1.50\every_grain_of_us_blender\every_grain_of_us.blend" -S Film
       -o "\\192.168.1.50\every_grain_of_us_blender\render4k\frame_####" -s 3330 -a -- --cycles-device OPTIX
   (-s 3330 starts it halfway through; run it again without -s at the end to pick up what is left.)
   Or simply split the film: one PC renders frames 0-3329 and the other 3330-6659 (Output Properties >
   Frame Start / End), each into its own folder, and copy all the frames into one folder at the end.
   A frame that was being rendered when a computer stopped leaves an empty (0 KB) file behind: at the end,
   delete the 0 KB files and render again to fill those gaps.


3. MAKE THE MP4
   Once all 6,660 frames are in "render4k", open every_grain_of_us_compose.blend and
   Render > Render Animation. Out comes every_grain_of_us_4k.mp4 (4K, with sound) in this folder.

   Voice-over or captions: the narrated mix is on by default. For music with short captions instead,
   select "Music + narration" and press H (mute), select "Music only" and "Short captions" and press
   Alt+H (unmute), change the file name in Output Properties, and render again.
   Change the title: click the "Title" strip, press N, and edit the text under Effect Strip.


WHAT IS IN THE LOOK
   4K (3840x2160) throughout, with the on-screen text drawn natively at 4K.
   The opening: diorama close-ups, low over the first sparse grains of people in Africa, a look north to
     the Neanderthals of Europe, back to East Africa for the exit, then out to the whole map by 0:45.
   Motion blur: the migrating sand streaks through the air; the camera blurs as it moves.
   Sand tracks world population: a sparse scatter at first (1 grain ~ 120 people), filling as people
     multiply (1 grain ~ 56,000 today). The great cities rise as mountains of poured sand with
     sand-pile slopes: the bigger the city, the higher the summit and the wider the base (Tokyo is the
     tallest); coastal and island cities spill over the glass onto the sea. They rise with
     urbanization in the last century.
   The finale: down among the city mountains, then a low flight through the world's great cities
     (names appear as each comes up): Mumbai, Delhi, Dhaka, Bangkok, Singapore, Jakarta, Manila,
     Hong Kong, Shanghai, Beijing, Seoul, Tokyo, a whip across the Pacific to Los Angeles, Mexico City,
     New York, Lima, Sao Paulo, Buenos Aires, across the Atlantic to London, Paris, Istanbul, Moscow,
     Cairo, Lagos and Kinshasa, then up to reveal the whole world. A golden low sun lights the mountains.
   Richer sand: every grain a little different in shade, hue and shine, and slightly translucent.
   The grade: AgX with the High Contrast look, a touch darker (exposure -0.15), as in the thumbnails.

SPEED AND QUALITY
   The project is set for high quality at a sensible speed:
     up to 64 samples per pixel, stopping early wherever the image is already clean (noise threshold 0.04),
     then denoised (OpenImageDenoise, using albedo and normals)
     light bounces trimmed to what glass and sand need (6 total, 6 through glass)
     no caustics, a fixed noise pattern so the still camera does not shimmer
     the sand as a point cloud (spheres traced directly, no meshes), only the grains that can be seen
   Before the full run, test one frame: go to a frame (e.g. 5640, among the city mountains) and press F12.

   To go faster:  Render Properties > Sampling > Render > Max Samples 32 (a little noisier), or
                  Output Properties > Resolution 50% for a 1920x1080 draft of the whole film.
   To go cleaner: Max Samples 128 and Noise Threshold 0.02.

   Rough time per 4K frame (scaled from measurements on a processor and Blender's public benchmark):
     NVIDIA RTX 5090 / 4090         about 12-18 seconds   whole film about 22-33 hours
     NVIDIA RTX 5080 / 4080         about 15-22 seconds   about 28-41 hours
   Two computers sharing the job divide this between them. The low flight through the cities is
   the slowest part.


THE STORY ON SCREEN
   Diorama close-ups of the first people open the film; at 0:45 the camera settles on the whole map and
   holds it for 200,000 years, through the arrival of today's migrants. At 2:43 it comes down among the
   city mountains and flies through the world's great cities, ending in Africa, where the story began,
   and rises to reveal the whole world. The timed script for the voice-over is narration.srt.
