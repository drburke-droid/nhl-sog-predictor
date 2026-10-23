# Your brain re-tunes "sharp": blur adaptation

A 7-minute video in two parts.

* **Part 1, the demo (2:34).** Viewers stare at a glowing dot while the left panel shows blurred photos and the right
  panel over-sharpened ones. After a minute, gold-framed test pairs show the *same* photo on both sides, and the left
  side looks sharper. This is the blur aftereffect of Webster, Georgeson & Webster (2002).
* **Part 2, the experiment (4:35).** Every eye blurs a point of light into its own smudge. Artal et al. (2004) cancelled
  that blur with adaptive optics, put it back either as each person's own smudge or the same smudge rotated, and all
  five people saw the rotated one as blurrier (they had to remove 20–40% of it to match).

Voice: Kokoro TTS, `bf_lily`. Music: the felt piano, pad and room from `../vernier/music.py`, re-arranged here.

## Deliverables

Rendered copies are committed in [`../media/blur_adaptation/`](../media/blur_adaptation/):

| File | What |
|---|---|
| `blur_adaptation.mp4` | Main cut (NASA photos), 1080p30, about 88 MB (two-pass re-encode of the CRF 12 master) |
| `blur_adaptation_fire.mp4` | "We Didn't Start the Fire" edition: Part 1 photos are public-domain pictures of things named in the song; the end card says so |
| `thumb_A.png`, `thumb_B.png`, `thumb_C.png` | A/B/C thumbnails (1280×720); `thumb_A_fire.png` is the Fire edition's A |
| `YOUTUBE.md`, `YOUTUBE_fire.md` | Titles, descriptions with chapters, tags |
| `fire_manifest.json` | Every Fire-edition photo: source page and the rights statement quoted from it |

The narration script is a Claude Doc ("Blur Adaptation Video Script"); `inputs/part1_script.json` and
`inputs/part2_script.json` are the copies the pipeline reads. `[0.8]` in a line is 0.8 s of silence.

## Run it

Needs ffmpeg, the Inter font (`/usr/share/fonts/opentype/inter`), Python 3.13 with numpy, scipy, Pillow, soundfile, and a
separate Python 3.12 venv for Kokoro (`pip install torch --index-url https://download.pytorch.org/whl/cpu kokoro soundfile`,
plus `apt install espeak-ng`). Renders go to `work/` (or `$BLUR_WORK`), which is git-ignored:

```
work/blur/src_images/      the NASA photos listed in inputs/nasa_images.json (images-api.nasa.gov, "~large.jpg")
work/v2/                   copy inputs/part1_script.json and inputs/part2_script.json here
work/fire/src_images/      the photos in inputs/fire/manifest.json;  copy inputs/fire/sets.json here
```

| Step | Command | Output |
|---|---|---|
| 1. Demo (Part 1 picture) | `python3 demo1_v2.py video` · `python3 demo1_v2.py stills` | `work/blur/demo1_v2.mp4`, stills |
| 2. Voice | `kvenv/bin/python tts2.py work/v2/part1_script.json work/v2/vo_p1 bf_lily` (same for part 2 → `vo_p2`) | one wav per line |
| 3. Part 2 picture | `python3 part2b.py video` (`stills` for a check; `tail E3` re-renders from scene E3 on) | `work/v2/part2_video.mp4` |
| 4. Audio + merge | `python3 audio_v2.py part1`, `part2`, `merge` | `work/v2/blur_adaptation_v2.mp4` |
| 5. Thumbnails | `python3 thumbs.py` | `work/v2/thumbs/` |

Fire edition: the same steps with `DEMO_SET=fire PART2_DIR=work/fire DEMO_VIDEO=work/fire/demo1_fire.mp4
FINAL_NAME=blur_adaptation_fire.mp4`, and `inputs/fire/part2_script.json` (it adds the end-card line E4).

## How it works

* **`demo1_v2.py`** draws the demo frame by frame (cyan adapting frames, gold test frames, a glowing fixation dot,
  bloom, the cone-mosaic background) and pipes raw frames to ffmpeg. Blur and sharpening are done in linear light.
* **`optics.py`** is the eye model: OSA-normalized Zernike wavefronts over a 6 mm pupil at 550 nm, point-spread
  functions by FFT (0.079 arcmin per pixel), rotated aberrations, the binary-noise stimulus, and convolution through a
  PSF. The three eyes are illustrative higher-order aberrations (about 0.26 µm RMS), not anyone's measurements. As in
  the paper, image quality (Strehl ratio) is the same at every rotation.
* **`part2b.py`** builds Part 2 scene by scene; each scene's length is its narration plus the script's pause. It reuses
  scene code from `part2.py` (the first cut).
* **`audio_v2.py`** re-times Part 1 without re-rendering it: it freezes the still cards (rules, reveal, control, end)
  long enough for each voice line, then mixes voice, the score and the test chimes, ducks the music under the voice,
  and normalizes the merged film to about −17 LUFS.

## Accuracy notes

* The experiment details (5 right eyes, 97-actuator mirror, 30 Hz wavefront sensor, 6 mm pupil, 500 ms presentations,
  rotations in 45° steps, matching factor 0.6–0.8, the 180° rotation still looking blurrier) are from Artal et al.
  (2004), J Vision 4(4): 281–287, doi:10.1167/4.4.4. Part 2's results bar shows only that 60–80% range; there are no
  per-person values.
* The blur aftereffect in Part 1 is Webster, Georgeson & Webster (2002), Nature Neuroscience 5: 839–840,
  doi:10.1038/nn906. How strongly a viewer sees it depends on screen size, distance and whether they keep fixating.
* The handwriting comparison and the "what you perceive" panel are labelled on screen as an analogy and an illustration.
