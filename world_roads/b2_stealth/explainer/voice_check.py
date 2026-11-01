"""Transcribe every narration clip with Whisper (faster-whisper small.en) and flag any that don't read back as written.
Piper samples each take at random, so a bad take is fixed by synthesizing again. Four lines were re-taken this way:
07 ("R-C-S"), 22 ("B-two's"), 37 and 38 (commas instead of full stops so "Shape" isn't swallowed)."""
import glob, json, os, re, subprocess
import numpy as np
from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
tl = json.load(open(os.path.join(HERE, 'timeline.json')))
want = {l['wav']: l['text'] for s in tl['scenes'] for l in s['lines']}
norm = lambda s: re.sub(r"[^a-z0-9' ]", '', s.lower().replace('-', '').replace(',', ' ')).split()
m = WhisperModel('small.en', device='cpu', compute_type='int8')
for f in sorted(glob.glob(os.path.join(HERE, 'audio', '*.wav'))):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'], capture_output=True).stdout
    segs, _ = m.transcribe(np.frombuffer(raw, dtype=np.float32), beam_size=3)
    got = ' '.join(s.text.strip() for s in segs)
    ok = norm(got) == norm(want[os.path.basename(f)])
    print('ok ' if ok else 'CHECK', os.path.basename(f), '|', got)
