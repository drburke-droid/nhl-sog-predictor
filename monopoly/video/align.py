"""Word timings for the voice-over: audio/<scene>.words.json, one [word, start, end] per narration word.

Whisper hears the file; its words are matched to the script's words, and any word it missed is placed
in proportion between its neighbours.  Scene.cue() uses these timings when they exist.

    python3 align.py [scene ...]
"""
import difflib
import json
import os
import re
import subprocess
import sys

import numpy as np

from engine import AUDIO, SCENE_FILES, narration


def norm(w):
    return re.sub(r"[^a-z0-9]", '', w.lower().replace('’', "'"))


def align(scene, model):
    path = os.path.join(AUDIO, scene + '.mp3')
    words = narration(scene).split()
    pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    segs, _ = model.transcribe(np.frombuffer(pcm, np.float32), word_timestamps=True, language='en', beam_size=5,
                               initial_prompt='Lizzie Magie. Monopoly. The Landlord\'s Game.')
    heard = [(w.word.strip(), w.start, w.end) for s in segs for w in s.words]
    a, b = [norm(w) for w in words], [norm(h[0]) for h in heard]
    times = [None] * len(words)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = list(heard[blk.b + k][1:])
    matched = sum(t is not None for t in times)
    # fill the gaps by character count between the neighbouring matched words
    end = heard[-1][2] if heard else 0.0
    i = 0
    while i < len(words):
        if times[i] is not None:
            i += 1; continue
        j = i
        while j < len(words) and times[j] is None:
            j += 1
        t0 = times[i - 1][1] if i > 0 else 0.0
        t1 = times[j][0] if j < len(words) else end
        chars = [len(w) + 1 for w in words[i:j]]
        tot, acc = sum(chars), 0
        for k in range(i, j):
            s = t0 + (t1 - t0) * acc / tot
            acc += chars[k - i]
            times[k] = [s, t0 + (t1 - t0) * acc / tot]
        i = j
    out = [[w, round(s, 3), round(e, 3)] for w, (s, e) in zip(words, times)]
    json.dump(out, open(os.path.join(AUDIO, scene + '.words.json'), 'w'), indent=0)
    return matched, len(words), ' '.join(h[0] for h in heard)


if __name__ == '__main__':
    from faster_whisper import WhisperModel
    model = WhisperModel('small.en', device='cpu', compute_type='int8')
    for scene in sys.argv[1:] or SCENE_FILES:
        m, n, heard = align(scene, model)
        print('%-26s matched %3d of %3d words' % (scene, m, n), flush=True)
        if scene == '01_cold_open':
            print('   heard:', heard)
