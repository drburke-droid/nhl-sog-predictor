# Full 1080p upload master (9:07)

GitHub limits files to 100 MB, so the master is stored as three byte-for-byte parts.

Join them, then upload `signal_fires_full.mp4`:

* macOS / Linux: `cat signal_fires_full.mp4.part* > signal_fires_full.mp4`
* Windows (Command Prompt): `copy /b signal_fires_full.mp4.part0+signal_fires_full.mp4.part1+signal_fires_full.mp4.part2 signal_fires_full.mp4`

The SHA-256 of the joined file is in `SHA256.txt`. A 720p preview that plays directly is at
`../signal_fires_full_720p_preview.mp4`.
