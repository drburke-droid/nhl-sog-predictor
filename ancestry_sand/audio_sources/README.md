# Audio sources for the migration film

The generated audio that cannot be regenerated identically (ElevenLabs output), kept so the sound can be
rebuilt at any time.

- `narration_lines/line_00.mp3` ... `line_20.mp3`: Lily's 21 narration lines (eleven_multilingual_v2, voice
  pFZP5JQG7iQjIQuC4Bku, speed 1.0), timed by `../narration_script.json`.
- `music/epic_take3.mp3` (the default score) and `music/epic_take2.mp3`: the epic 6/8 score (`../music_plan_epic.json`).
- `music/original_score_mastered.mp3`: the first, gentler score, already mastered to the film's length.

Rebuild the epic take 3 mix (as in the v3.2 package):

    python3 ../../.claude/skills/sandglass/scripts/extend_score.py music/epic_take3.mp3 /tmp/take3_222.wav 190 212 4.0 222
    python3 ../mix_audio.py /tmp/take3_222.wav narration_lines OUT_DIR

The .blend, hud.mov and captions are rebuilt from the code (README.md, "Render and preview workflow").
