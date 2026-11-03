Voice-over, one file per scene, named as in script/transcript.md (01_cold_open.mp3 ... 11_the_playbook.mp3).
Each scene re-times itself to its file automatically.

  tts.py    makes them with ElevenLabs (voice 1SM7GgM6IMuvQlz2BwM3, "Mark"); needs ELEVEN_LABS_API in the environment.
            Only scenes without a file are made, so delete a file to redo it:  python3 tts.py 04_buy_or_pass
  align.py  writes <scene>.words.json, the start and end of every narration word, so cues land on the word.
            Re-run it for any scene whose audio changes.  Needs: pip install faster-whisper
