EVERY GRAIN OF US: the full-Cycles Blender project (4K), version 3.3
====================================================================

GETTING IT FROM GITHUB
  This folder is releases/every_grain_of_us_v3_3/ in the repository, on branch claude/sweet-faraday-uk5z62.
  Easiest: on the repository's GitHub page pick that branch, then Code > Download ZIP, and take this
  folder out of the zip. (Or download each file from the folder's page with the download button.)
  Keep every file together in one folder, as here.

QUICK START: extract this folder, render, assemble
  1. Extract the whole folder somewhere with about 75 GB free, and keep everything together in it.
  2. Open every_grain_of_us.blend, Render > Render Animation (frames go into "render4k" here).
  3. When all 6,660 frames are done, open every_grain_of_us_compose.blend, Render > Render Animation.
     Out comes every_grain_of_us_4k.mp4: the film, the on-screen text, the title, the music and
     Lily's voice-over, in 4K. Details for each step below.

What is in this folder
  every_grain_of_us.blend          RENDER: the whole film in Cycles (3:42, 6,660 frames at 30 fps,
                                   3840x2160), into a "render4k" folder here
  every_grain_of_us_compose.blend  ASSEMBLE: joins the frames, the on-screen text, the title and the
                                   sound into the finished MP4 (every path is relative to this folder)
  hud_part1.mov, hud_part2.mov, hud_part3.mov
                                   the on-screen text in 4K (year, "1 grain = N people", legend, place and
                                   city names), transparent; in three parts (frames 0-2099, 2100-5039,
                                   5040-6659) so each file stays under GitHub's 100 MB limit. The compose
                                   file plays them one after another.
  every_grain_of_us_score_with_narration.wav   the epic score + Lily's voice-over (used by default)
  every_grain_of_us_score.wav                  the epic score alone (for the captions version)
  streak_scale.mov                 the "1 streak = N people" line under the grain scale, transparent
  captions_4k.mov                  short captions, transparent (for the music-only version)
  narration.srt, captions_short.srt   subtitle files for YouTube
  render4k\                        the frames go here (empty apart from a note)
  alt_music_take2\                 the other take of the epic score (same two file names): to use it,
                                   copy both files over the ones beside the .blend files

You need Blender 4.2 or newer (free from blender.org); for an NVIDIA RTX 50-series card (5080, 5090),
Blender 4.4 or newer. A graphics card makes a huge difference.

From scratch: render into an empty render4k folder. Already rendered version 3.2? Keep frames 0-5039
and re-render only the ending: see "UPDATING FROM 3.2" below.

New in version 3.3: the music starts on the first frame (the score opened with 0.7 s of silence and a
very quiet first 30 seconds, so under the voice-over it seemed to start at 0:30), and the flight
through the cities is slower and easier to follow: 10 cities instead of 25 at about 40% of the speed
(Mumbai, Delhi, Dhaka, Bangkok, Hong Kong, Shanghai, Tokyo, a whip across the Pacific, Los Angeles,
Mexico City, New York), then a slow 13-second rise from New York to the whole world, arriving on the
final chord at 3:35 and held to the end (the old rise took 4.6 seconds). Frames 0-5039 are unchanged.

UPDATING FROM 3.2 (you have already rendered the frames)
  1. Copy these new files over the old ones in your folder: every_grain_of_us.blend, hud_part3.mov,
     every_grain_of_us_score_with_narration.wav, every_grain_of_us_score.wav,
     every_grain_of_us_compose.blend and README.txt.
  2. In render4k, delete frame_5040.png to frame_6659.png (sort by name, click frame_5040.png,
     Shift+click the last one, Delete). Keep frames 0000-5039: they have not changed.
  3. Open every_grain_of_us.blend, Render > Render Animation. Blender skips the frames you kept and
     renders only the 1,620 missing ones (the ending, 2:48-3:42).
  4. Open every_grain_of_us_compose.blend, Render > Render Animation, as before.

New in version 3.2 (the complete package): the cities are mountains of poured sand, every city of
1 million people or more (the bigger the city, the higher the summit and the wider the base; a
1-million city is a low mound; coastal and island cities spill over the glass); the whole film has a
deeper, richer grade, as in the thumbnails; the music is the epic score (a driving cello figure in
6/8 that builds to the full orchestra for the flight through the cities); and the clicks at the ends
of a few voice-over lines are gone.
Updated: every city follows its own population history (Chandler's census before 1950, the UN's figures
after), so the cities rise in their real order instead of all together.
Updated: the migration streams are now in proportion. A flying streak is a fifth of a grain, so each
stream flies five streaks for every grain that lands, the same for every migration (a flow twice as large is
twice as dense; before, every stream was drawn at about the same density whatever its size). The screen
says so under the grain scale: "1 streak = N people" (2,200 in the 1650s, 11,000 today), from the new
streak_scale.mov overlay.


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
   (Check first: render4k holds frame_0000.png to frame_6659.png, and no file is 0 KB.)

   ONLY WANT THE NEW SOUND on an MP4 you already made? Swap it in, no re-render (needs ffmpeg):
     ffmpeg -i every_grain_of_us_4k.mp4 -i every_grain_of_us_score_with_narration.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 320k -shortest every_grain_of_us_4k_v2.mp4

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
     multiply (1 grain ~ 56,000 today). Every city of 1 million people or more rises as a mountain
     of poured sand with sand-pile slopes: the bigger the city, the higher the summit and the wider the
     base (Tokyo is the tallest; a 1-million city is a low hill); coastal and island cities spill over
     the glass onto the sea. Each city rises with its own history: London is the tallest in 1900, New York
     and Tokyo by 1950, and Lagos, Dhaka and Kinshasa climb only after 1950.
   The finale: down among the city mountains, then a slow, low flight through the world's great cities
     (names appear as each comes up): Mumbai, Delhi, Dhaka, Bangkok, Hong Kong, Shanghai, Tokyo, a whip
     across the Pacific to Los Angeles, Mexico City and New York, then a slow rise over the Atlantic to
     the whole world, arriving on the final chord and held to the end. A golden low sun lights the mountains.
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
   city mountains and flies slowly through the great cities of Asia and the Americas, then rises from
   New York to the whole world, where Africa, where the story began, sits at the centre. The timed
   script for the voice-over is narration.srt.
