# The video: "The Monopoly decisions you control"

Everything on screen is drawn by code in the style of the 1904 patent and 1930s paperwork (see
`../reports/Art direction.md`); nothing is Hasbro's. The voice-over is ElevenLabs.

| File | What it does |
|---|---|
| `script/narration_elevenlabs.txt` | The narration, one paragraph per scene |
| `tts.py` | Makes `audio/<scene>.mp3` with ElevenLabs (needs `ELEVEN_LABS_API`) |
| `align.py` | Times every narration word (`audio/<scene>.words.json`) so animation lands on the word (needs `pip install faster-whisper`) |
| `engine.py` | Frames, camera, paper grain, ffmpeg encoding, phrase cues (`scene.cue("...")`) |
| `kit.py` | Pages that slide in, and the elements on them: typed text, ledger bars, stamps, telegrams, drawings |
| `scenes_intro.py`, `scenes_a.py`, `scenes_b.py`, `scenes_c.py` | The 15 scenes |
| `sfx.py` | Synthesised sound effects (stamp, typewriter, paper, telegraph, pen, dice) |
| `render.py` | Renders scenes in parallel to `out/<scene>.mp4` |
| `assemble.py` | Joins the scenes with crossfades, mixes voice and effects (and `audio/music.mp3` if present), writes `out/monopoly_decisions.mp4` and `out/chapters.txt` |

```
python3 render.py -j 3            # every scene (about 25 minutes on 4 cores)
python3 render.py 07c             # one scene, after changing it
python3 scenes_b.py 07c_which_set_first --sheet   # contact sheet of key frames, to check layout quickly
python3 assemble.py --preview     # the finished video, plus a 720p copy
```

To change a line of narration: edit the paragraph, delete that scene's mp3, run `tts.py` and `align.py` for it, then
render the scene again and re-assemble. Cues follow the words, so the animation re-times itself.

Music: put a licensed track at `audio/music.mp3` and re-run `assemble.py`; it is looped and ducked under the voice.
