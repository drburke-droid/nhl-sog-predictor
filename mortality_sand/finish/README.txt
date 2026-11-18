FIX THE FROZEN ENDING AND REBUILD THE FILM
==========================================

What went wrong
  The film was built from the frames in render4k\. One or more frames from about 1980 (frame 7000)
  onward were missing, or were empty placeholders left by an interrupted render. ffmpeg stopped
  reading frames at the first gap and repeated the last picture to the end, while the text kept going.

How to fix it
  1. Extract this zip, then put fix_and_finish.bat and fix_frames.py into your project folder, the
     one that holds render4k, for example:
         C:\Users\jsbno\Downloads\two_thousand_years_cycles_4k
     (If they end up in a subfolder inside it, that works too.)
  2. Double-click fix_and_finish.bat. If Windows shows "Windows protected your PC", click
     More info, then Run anyway.

What the script does
  1. It finds Blender and ffmpeg. If ffmpeg is missing, it installs it.
  2. It checks every one of the 8,274 frames. It looks for missing files, empty placeholders,
     files cut off part-way, and frames of the wrong size.
  3. It deletes the broken frames and renders only those, plus any missing ones. It uses your
     .blend and your graphics card, and skips every good frame.
  4. It checks again. If anything is still missing, it stops and tells you, instead of building a
     broken film.
  5. It builds two_thousand_years_of_catastrophe_4k.mp4 with the frames, the text, the bar race,
     the held ending and the narration.

How long it takes
  If everything after 1980 is missing, that is about 1,250 frames. At the 4K settings, each frame
  takes about as long as your earlier frames did. You can close the window at any time and run the
  script again: it carries on where it stopped.

YouTube
  YouTube cannot swap the video file of an existing upload. Upload the new MP4 as a new video, with
  the same title, thumbnail, description and subtitles. Then delete or unlist the old one.
