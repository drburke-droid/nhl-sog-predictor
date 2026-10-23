"""Kokoro TTS with in-line pauses (run with the Python 3.12 venv).

    kvenv/bin/python tts2.py script.json outdir [voice]

script.json: [{"id": "A2", "text": "In a perfect eye... [0.8] all of its light..."}, ...]
`[1.0]` in the text = 1.0 s of silence at that spot. Each chunk is synthesized separately and trimmed of
its own leading/trailing silence, so the pauses are exactly what the script says. Writes outdir/<id>.wav
(24 kHz) and outdir/durations.json; unchanged lines are reused.
"""
import os
import re
import sys
import json
import hashlib
import numpy as np
import soundfile as sf
from kokoro import KPipeline

SR = 24000
src, outdir = sys.argv[1], sys.argv[2]
voice = sys.argv[3] if len(sys.argv) > 3 else "bf_lily"
os.makedirs(outdir, exist_ok=True)
lines = json.load(open(src))
dpath = os.path.join(outdir, "durations.json")
durs = json.load(open(dpath)) if os.path.exists(dpath) else {}
pipe = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")


def trim(a, thr=0.012, keep=0.035):
    idx = np.flatnonzero(np.abs(a) > thr)
    if not len(idx):
        return a[:0]
    k = int(keep * SR)
    return a[max(0, idx[0] - k): idx[-1] + k]


for ln in lines:
    key = hashlib.sha1(f"{voice}|{ln['text']}".encode()).hexdigest()[:12]
    wav = os.path.join(outdir, f"{ln['id']}.wav")
    if durs.get(ln["id"], {}).get("key") == key and os.path.exists(wav):
        continue
    parts = re.split(r"\[\s*([0-9.]+)\s*s?\s*\]", ln["text"])
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(np.zeros(int(float(p) * SR), np.float32))
        elif p.strip():
            a = np.concatenate([np.asarray(x) for _, _, x in pipe(p.strip(), voice=voice)])
            out.append(trim(a.astype(np.float32)))
    aud = np.concatenate(out)
    sf.write(wav, aud, SR)
    durs[ln["id"]] = {"key": key, "seconds": round(len(aud) / SR, 3), "text": ln["text"]}
    print(f"{ln['id']:6s} {len(aud) / SR:5.2f}s  {ln['text'][:72]}", flush=True)
json.dump(durs, open(dpath, "w"), indent=1)
