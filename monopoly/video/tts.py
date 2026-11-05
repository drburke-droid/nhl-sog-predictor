"""The voice-over, built from chunks so an edit only re-voices what changed.

script/narration_elevenlabs.txt holds one block per scene (blocks separated by a blank line) and one chunk per line:
usually a sentence, with very short sentences joined to the next. Each chunk's audio is kept in audio/chunks/, named
by a hash of its text and the voice settings, so:

    python3 tts.py               # voice only the chunks that have no audio yet, rebuild the scenes whose chunks changed
    python3 tts.py --dry-run     # list what would be voiced, and how many characters it would cost
    python3 tts.py --split REV   # one-off: cut the existing scene recordings into chunks (REV = the git revision whose
                                 # script matches them), so none of the current audio has to be voiced again

A scene's audio/<scene>.mp3 is its chunks joined end to end; audio/<scene>.chunks.json lists them. After a rebuild,
run align.py for the scenes it names so the animation follows the new timing.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from engine import AUDIO, SCENE_FILES, SCRIPT

VOICE = '1SM7GgM6IMuvQlz2BwM3'          # "Mark - Casual, Relaxed and Light"
MODEL = 'eleven_multilingual_v2'
SETTINGS = {'stability': 0.5, 'similarity_boost': 0.75, 'style': 0.1, 'use_speaker_boost': True}
RATE = 44100
PAD = 0.3            # silence kept at each end of a newly voiced chunk: half a typical pause between sentences
CHUNKS = os.path.join(AUDIO, 'chunks')
ABBREV = {'St.', 'Ave.', 'Mr.', 'Mrs.', 'No.', 'R.R.'}


# ---------------------------------------------------------------- the script
def blocks(text=None):
    """Scene name -> its chunks (lines), from the script."""
    text = open(SCRIPT).read() if text is None else text
    paras = [b for b in text.strip().split('\n\n')]
    return {s: [l.strip() for l in p.split('\n') if l.strip()] for s, p in zip(SCENE_FILES, paras)}


def sentences(par):
    words, out, cur = par.split(), [], []
    for w in words:
        cur.append(w)
        bare = w.rstrip('"”’')
        if bare.endswith(('.', '?', '!')) and not bare.endswith('..') and w not in ABBREV:
            out.append(' '.join(cur)); cur = []
    if cur:
        out.append(' '.join(cur))
    return out


def chunk(par, short=40):
    """Sentences, with any sentence under `short` characters joined to the one after it."""
    out, carry = [], ''
    for s in sentences(par):
        s = (carry + ' ' + s).strip() if carry else s
        if len(s) < short:
            carry = s
        else:
            out.append(s); carry = ''
    if carry:
        if out:
            out[-1] += ' ' + carry
        else:
            out.append(carry)
    return out


def key(text):
    return hashlib.sha1(('%s|%s|%s|%s' % (VOICE, MODEL, json.dumps(SETTINGS, sort_keys=True), text)).encode()).hexdigest()[:16]


def chunk_path(text):
    return os.path.join(CHUNKS, key(text) + '.mp3')


# ---------------------------------------------------------------- audio helpers
def decode(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(RATE), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def encode(x, path):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', '-', '-c:a', 'libmp3lame',
                    '-b:a', '192k', path], input=pcm, check=True)


def speech_rms(x):
    frames = x[: len(x) // 1024 * 1024].reshape(-1, 1024)
    r = np.sqrt((frames ** 2).mean(axis=1))
    loud = r[r > 0.02]
    return float(np.median(loud)) if len(loud) else 0.0


def trim(x, thresh=0.01):
    idx = np.nonzero(np.abs(x) > thresh)[0]
    if not len(idx):
        return x
    a, b = max(0, idx[0] - int(0.03 * RATE)), min(len(x), idx[-1] + int(0.08 * RATE))
    return x[a:b]


# ---------------------------------------------------------------- ElevenLabs
def synth(text, prev, nxt):
    body = {'text': text, 'model_id': MODEL, 'voice_settings': SETTINGS}
    if prev:
        body['previous_text'] = prev[-1000:]
    if nxt:
        body['next_text'] = nxt[:1000]
    req = urllib.request.Request(
        'https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=mp3_44100_128' % VOICE,
        data=json.dumps(body).encode(), method='POST',
        headers={'xi-api-key': os.environ['ELEVEN_LABS_API'], 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read(), r.headers.get('character-cost')
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode(errors='replace')
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** (attempt + 1)); continue
            raise RuntimeError('HTTP %d %s' % (e.code, msg))


def voice_chunk(text, prev, nxt, level):
    """Voice one chunk, trim its silence, pad it like a cut chunk, and match the scene's loudness."""
    tmp = os.path.join(CHUNKS, '_new.mp3')
    data, cost = synth(text, prev, nxt)
    open(tmp, 'wb').write(data)
    x = trim(decode(tmp))
    if level:
        r = speech_rms(x)
        if r > 0:
            x = x * (level / r)
    pad = np.zeros(int(PAD * RATE), np.float32)
    encode(np.concatenate([pad, x, pad]), chunk_path(text))
    os.remove(tmp)
    return cost


# ---------------------------------------------------------------- build
def manifest_path(scene):
    return os.path.join(AUDIO, scene + '.chunks.json')


def build(dry=False):
    os.makedirs(CHUNKS, exist_ok=True)
    sc = blocks()
    flat = [(s, i) for s in SCENE_FILES for i in range(len(sc[s]))]
    todo = [(s, i) for s, i in flat if not os.path.exists(chunk_path(sc[s][i]))]
    chars = sum(len(sc[s][i]) for s, i in todo)
    print('%d of %d chunks need voicing, %d characters' % (len(todo), len(flat), chars))
    for s, i in todo:
        print('  %-24s %s' % (s, sc[s][i][:90]))
    if dry:
        return []
    spent = 0
    for s, i in todo:
        lines = sc[s]
        k = SCENE_FILES.index(s)
        before = ' '.join(([' '.join(sc[SCENE_FILES[k - 1]])] if k else []) + lines[:i])
        after = ' '.join(lines[i + 1:] + ([' '.join(sc[SCENE_FILES[k + 1]])] if k + 1 < len(SCENE_FILES) else []))
        kept = [chunk_path(l) for l in lines if os.path.exists(chunk_path(l))]
        level = float(np.median([speech_rms(decode(p)) for p in kept[:6]])) if kept else 0.0
        cost = voice_chunk(lines[i], before, after, level)
        spent += int(cost) if cost and cost.isdigit() else len(lines[i])
        print('voiced %-22s %4s chars  %s' % (s, cost or len(lines[i]), lines[i][:70]), flush=True)
    rebuilt = []
    for s in SCENE_FILES:
        keys = [key(l) for l in sc[s]]
        mp = manifest_path(s)
        if os.path.exists(mp) and json.load(open(mp)) == keys and os.path.exists(os.path.join(AUDIO, s + '.mp3')):
            continue
        x = np.concatenate([decode(chunk_path(l)) for l in sc[s]])
        encode(x, os.path.join(AUDIO, s + '.mp3'))
        json.dump(keys, open(mp, 'w'), indent=0)
        rebuilt.append(s)
        print('rebuilt %-24s %.1f s' % (s, len(x) / RATE))
    if todo:
        print('characters charged: %d' % spent)
    if rebuilt:
        print('now run:  python3 align.py ' + ' '.join(rebuilt))
    return rebuilt


def split_existing(rev):
    """Cut each scene's current recording into chunks, using its word timings and the script it was recorded from."""
    os.makedirs(CHUNKS, exist_ok=True)
    rel = os.path.relpath(SCRIPT, subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip())
    old = subprocess.run(['git', 'show', '%s:%s' % (rev, rel)], capture_output=True, text=True, check=True).stdout
    old_paras = old.strip().split('\n\n')
    for s, par in zip(SCENE_FILES, old_paras):
        par = ' '.join(par.split('\n'))
        words = json.load(open(os.path.join(AUDIO, s + '.words.json')))
        if [w[0] for w in words] != par.split():
            print('skip %s: its word timings do not match the script at %s' % (s, rev)); continue
        x = decode(os.path.join(AUDIO, s + '.mp3'))
        parts, k = chunk(par), 0
        cuts = [0]
        for p in parts[:-1]:
            k += len(p.split())
            cuts.append(int((words[k - 1][2] + words[k][1]) / 2 * RATE))
        cuts.append(len(x))
        made = 0
        for p, a, b in zip(parts, cuts, cuts[1:]):
            if not os.path.exists(chunk_path(p)):
                encode(x[a:b], chunk_path(p)); made += 1
        if ' '.join(blocks()[s]) == par:  # the recording is current: record its chunks, keep its mp3 as is
            json.dump([key(p) for p in parts], open(manifest_path(s), 'w'), indent=0)
        print('%-24s %2d chunks cut (%d new)' % (s, len(parts), made))


if __name__ == '__main__':
    if '--split' in sys.argv:
        split_existing(sys.argv[sys.argv.index('--split') + 1])
    else:
        build(dry='--dry-run' in sys.argv)
