"""Narration with ElevenLabs: voice each timed cue of a VO script and lay the clips on the film's timeline.

    ELEVENLABS_API_KEY=... python narrate.py <VO_SCRIPT.md> <out dir> <film length s>   # -> <out dir>/narration.wav

The key is read from the environment (never from a file in the repository). A cue's time may carry tenths
("**2:10.4**"). SAY (set by the caller, {"written": "said"}) respells names for the voice only: the script keeps the
normal spelling, and the respelled text is what is sent and cached. Optional: ELEVENLABS_VOICE_ID (default
c6SfcYrb2t09NHXiT80T, the voice chosen for these films), ELEVENLABS_MODEL (default eleven_multilingual_v2). Each cue ("**m:ss** text") is
sent once and cached in <out dir>/voice/ by a hash of its text, voice and model, so re-runs cost nothing. A clip that
would run into the next cue pushes that cue later (reported). The assembly scripts mix narration.wav over the score
and dip the music by about 8 dB under the voice (duck_and_mix).
"""
import hashlib, json, os, re, subprocess, sys, time, urllib.error, urllib.request
import numpy as np

SR = 48000
VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "c6SfcYrb2t09NHXiT80T")      # the voice chosen for these films
MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
SAY = {}                                                                 # respellings for the voice, set by the caller


def say(text):
    for k, v in SAY.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


def cues(path):
    """[(seconds, text)] from '**m:ss** text' paragraphs; stage directions in *( )* are dropped"""
    s = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"\*\*(\d+):(\d\d(?:\.\d+)?)\*\*\s+(.+?)(?=\n\s*\n|\n##|\Z)", s, re.S):
        t = int(m.group(1)) * 60 + float(m.group(2))
        txt = re.sub(r"\*\(.*?\)\*", "", m.group(3), flags=re.S)
        txt = " ".join(txt.split())
        if txt:
            out.append((t, txt))
    return out


def cached(text, cache_dir):
    """the cached clip of a line (as respelled), or None"""
    h = hashlib.sha1(f"{VOICE}|{MODEL}|{say(text)}".encode()).hexdigest()[:16]
    mp3 = os.path.join(cache_dir, h + ".mp3")
    return mp3 if os.path.exists(mp3) and os.path.getsize(mp3) > 0 else None


def tts(text, cache_dir):
    hit = cached(text, cache_dir)
    if hit:
        return hit
    text = say(text)
    key = os.environ.get("ELEVENLABS_API_KEY")
    h = hashlib.sha1(f"{VOICE}|{MODEL}|{text}".encode()).hexdigest()[:16]
    mp3 = os.path.join(cache_dir, h + ".mp3")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is not set: add it to the environment's secrets and start a new session")
    body = json.dumps({"text": text, "model_id": MODEL,
                       "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.2,
                                          "use_speaker_boost": True}}).encode()
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128",
                                 data=body, method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})

    for attempt in range(5):                       # a refused or dropped request is retried (2, 4, 8, 16 s)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if attempt == 4 or e.code not in (401, 408, 409, 429, 500, 502, 503, 504):
                raise SystemExit(f"ElevenLabs refused a line: HTTP {e.code} {msg}")
            print(f"  ElevenLabs HTTP {e.code}, retrying: {msg[:160]}", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 4:
                raise
            print(f"  ElevenLabs request failed ({e}), retrying", flush=True)
        time.sleep(2 ** (attempt + 1))
    os.makedirs(cache_dir, exist_ok=True)
    open(mp3, "wb").write(data)
    return mp3


def decode(mp3):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def build(script, out_dir, total):
    cache = os.path.join(out_dir, "voice")
    track = np.zeros(int(total * SR) + SR, np.float32)
    free_at = 0.0
    for t, txt in cues(script):
        x = decode(tts(txt, cache))
        start = max(t + 0.15, free_at + 0.35)
        if start > t + 0.6:
            print(f"  cue at {int(t // 60)}:{t % 60:04.1f} starts {start - t:.1f} s late (the previous line ran long)", flush=True)
        i = int(start * SR)
        n = min(len(x), len(track) - i)
        track[i:i + n] += x[:n]
        free_at = start + len(x) / SR
    peak = np.abs(track).max() + 1e-9
    track = track / peak * 0.89
    from scipy.io import wavfile
    wavfile.write(os.path.join(out_dir, "narration.wav"), SR, (track[:int(total * SR)] * 32767).astype(np.int16))
    print("narration.wav", f"{len(cues(script))} cues", flush=True)


def duck_and_mix(music, voice_path, duck_db=18.0, voice_gain=1.0):
    """music (n x 2 float32 at SR) with the narration on top; the music dips by duck_db under the voice (18 dB keeps the
    voice about 15 dB above the score while it speaks; 8 dB left it only 4-6 dB above, too quiet to follow)"""
    from scipy.io import wavfile
    sr, v = wavfile.read(voice_path)
    v = v.astype(np.float32) / 32768.0
    n = len(music)
    v = np.pad(v, (0, max(0, n - len(v))))[:n]
    w = int(0.05 * SR)
    c = np.concatenate([[0.0], np.cumsum(v.astype(np.float64) ** 2)])
    idx = np.arange(n)
    lo, hi = np.clip(idx - w, 0, n), np.clip(idx + w, 0, n)
    rms = np.sqrt((c[hi] - c[lo]) / np.maximum(hi - lo, 1))
    active = (rms > 0.01).astype(np.float32)
    k = int(0.35 * SR)                                       # 0.35 s attack and release
    sm = np.convolve(active, np.ones(k, np.float32) / k, "same")
    gain = 10 ** (-duck_db * np.clip(sm * 1.6, 0, 1) / 20.0)
    out = music * gain[:, None] + (v * voice_gain)[:, None]
    return (out / (np.abs(out).max() + 1e-9) * 0.89).astype(np.float32)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], float(sys.argv[3]))
