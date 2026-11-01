# How the B-2 hides from radar

## The explainer (`explainer/`)

`../media/b2_stealth/b2-stealth-explainer.mp4`: 4:00, 1920x1080 at 30 fps, narrated, captions burned in
(`explainer/b2-stealth-explainer.srt` for platforms that take a subtitle file).

Ten scenes: the hook (172 ft of wingspan that reads like a bird on a scope), radar 101 (pulse, echo, delay to range),
radar cross section (B-52 ~100 m², B-2 ~0.1 m² or less: public estimates, the real figure is classified), the four
tricks (shape, edge alignment, absorbent skin, hidden engines), why it works (detection range goes as RCS^1/4, so
/10,000 echo is /10 range), the limits (long-wavelength early-warning radar; route planning, quiet emissions, night)
and a recap.

* `narrate.py`: says every line with Piper (`en_US-ryan-high`, MIT; set `PIPER_VOICE` to the .onnx) and lays out
  the timeline (`timeline.json`): each scene's lead-in, its lines with 0.45 s between them, its tail.
* `voice_check.py`: reads every clip back with Whisper (small.en) to catch mispronunciations; four lines were
  re-taken this way (see its docstring), so running `narrate.py` again replaces clips that were checked.
* `scene.html`: every scene drawn on a canvas as a function of time (`render(t)`), cued off the narration's line
  times; the captions are drawn from the same timeline. `sfxEvents()` lists the sound cues (radar pings, echoes,
  swells) at the times the picture makes them.
* `render.js`: Playwright drives the page frame by frame into ffmpeg (`node render.js video f0 f1 out.mp4`, four
  chunks in parallel, ~14 min); `stills` for spot checks, `events` writes `sfx.json`.
* `mix.py`: the soundtrack: the narration at its timeline positions, a quiet ambient pad ducked under the voice, the
  synthesized sound cues. Mastered to -16 LUFS (two-pass loudnorm) when muxed.

```
python narrate.py                     # clips + timeline.json
node render.js events sfx.json
for i in 0 1 2 3; do node render.js video $((i*1798)) $(((i+1)*1798)) part$i.mp4 & done; wait
python mix.py                         # mix_raw.wav
ffmpeg -f concat -i parts.txt -c copy video.mp4   # then mux with mix_raw.wav through loudnorm
```
