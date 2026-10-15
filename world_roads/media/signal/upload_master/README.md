# 1080p upload masters

GitHub limits files to 100 MB, so each master is stored as byte-for-byte parts. Join them, then upload
the joined file. The SHA-256 of each joined file is in `SHA256.txt`.

## Video 1: one fire around the world (5:56)

* macOS / Linux: `cat signal_fires_world.mp4.part* > signal_fires_world.mp4`
* Windows (Command Prompt):
  `copy /b signal_fires_world.mp4.part0+signal_fires_world.mp4.part1 signal_fires_world.mp4`

A 720p preview that plays directly is at `../signal_fires_world_720p_preview.mp4`.

## Video 2: the analog backup (3:27)

* macOS / Linux: `cat signal_fires_ai.mp4.part* > signal_fires_ai.mp4`
* Windows (Command Prompt):
  `copy /b signal_fires_ai.mp4.part0+signal_fires_ai.mp4.part1 signal_fires_ai.mp4`

A 720p preview that plays directly is at `../signal_fires_ai_720p_preview.mp4`.
