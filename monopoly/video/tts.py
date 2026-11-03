"""Generate the voice-over with ElevenLabs, one file per scene, into audio/.

    ELEVEN_LABS_API=... python3 tts.py [scene ...]     # default: every scene that has no file yet
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from engine import AUDIO, SCENE_FILES, SCRIPT

VOICE = '1SM7GgM6IMuvQlz2BwM3'          # "Mark - Casual, Relaxed and Light"
MODEL = 'eleven_multilingual_v2'
SETTINGS = {'stability': 0.5, 'similarity_boost': 0.75, 'style': 0.1, 'use_speaker_boost': True}


def synth(i, paras, path):
    body = {'text': paras[i], 'model_id': MODEL, 'voice_settings': SETTINGS}
    if i > 0:
        body['previous_text'] = paras[i - 1]
    if i + 1 < len(paras):
        body['next_text'] = paras[i + 1]
    req = urllib.request.Request(
        'https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=mp3_44100_128' % VOICE,
        data=json.dumps(body).encode(), method='POST',
        headers={'xi-api-key': os.environ['ELEVEN_LABS_API'], 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                open(path, 'wb').write(r.read())
            return
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode(errors='replace')
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** (attempt + 1)); continue
            raise RuntimeError('%s: HTTP %d %s' % (os.path.basename(path), e.code, msg))


if __name__ == '__main__':
    os.makedirs(AUDIO, exist_ok=True)
    paras = open(SCRIPT).read().strip().split('\n\n')
    want = sys.argv[1:] or SCENE_FILES
    for i, name in enumerate(SCENE_FILES):
        path = os.path.join(AUDIO, name + '.mp3')
        if name not in want or (not sys.argv[1:] and os.path.exists(path)):
            continue
        synth(i, paras, path)
        d = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                                 capture_output=True, text=True).stdout)
        print('%-26s %6.1f s  (%d words, %.0f words a minute)' % (name, d, len(paras[i].split()), len(paras[i].split()) / d * 60), flush=True)
