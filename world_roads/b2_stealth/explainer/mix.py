"""Build the soundtrack: narration + ambient pad (ducked under voice) + synced sound effects."""
import json, os, subprocess, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 48000
tl = json.load(open(os.path.join(HERE, 'timeline.json')))
ev = json.load(open(os.path.join(HERE, 'sfx.json')))
D = tl['duration'] + 0.5
N = int(D * SR)
rng = np.random.default_rng(1)

def read_wav48(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-af', 'aresample=resampler=soxr',
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)

# ---------- narration ----------
voice = np.zeros(N)
for sc in tl['scenes']:
    for ln in sc['lines']:
        x = read_wav48(os.path.join(HERE, 'audio', ln['wav']))
        i = int((sc['start'] + ln['start'] + {'38_recap.wav': -0.2}.get(ln['wav'], 0.0)) * SR)
        voice[i:i + len(x)] += x[:max(0, N - i)]
voice *= 0.5 / (np.sqrt(np.mean(voice[voice != 0] ** 2)) * 4)  # speech RMS ~ -18 dBFS

# ---------- ambient pad ----------
t = np.arange(N) / SR
def note(f, amp, harm=(1, .45, .2), det=0.0025):
    out = np.zeros(N)
    for k, h in enumerate(harm, 1):
        for d in (-det, det):
            out += h * np.sin(2 * np.pi * f * k * (1 + d) * t + rng.uniform(0, 6.28))
    return amp * out
# slow chord changes (A minor family), crossfaded every 12 s
chords = [[110.0, 164.81, 220.0, 261.63], [87.31, 130.81, 174.61, 220.0], [98.0, 146.83, 196.0, 246.94], [110.0, 164.81, 220.0, 329.63]]
seg = 12.0
pad = np.zeros(N)
for ci, ch in enumerate(chords):
    # envelope: this chord is active for segments where (segment index % 4) == ci
    phase = (t / seg) % len(chords)
    d = np.minimum(np.abs(phase - ci), len(chords) - np.abs(phase - ci))
    env = np.clip(1.2 - d, 0, 1) ** 2  # smooth crossfade
    sig = sum(note(f, 1.0 / (1 + i * 0.6)) for i, f in enumerate(ch))
    pad += env * sig
drone = note(55.0, 1.2, harm=(1, .3), det=0.001)
shimmer = note(880.0, 0.06, harm=(1,), det=0.004) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 7.0))
pad = pad + drone + shimmer
lfo = 0.85 + 0.15 * np.sin(2 * np.pi * t / 9.0)
pad *= lfo
pad /= np.max(np.abs(pad))
# duck under the voice
def movavg(x, win):
    c = np.concatenate([[0.0], np.cumsum(x)])
    h = win // 2
    idx = np.arange(len(x))
    lo = np.clip(idx - h, 0, len(x)); hi = np.clip(idx + h, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
venv = movavg(np.abs(voice), int(0.25 * SR))
venv = movavg((venv > 0.01).astype(float), int(0.6 * SR))
duck = 1.0 - 0.45 * np.clip(venv, 0, 1)
fade = np.clip(t / 3.0, 0, 1) * np.clip((D - 0.5 - t) / 3.0, 0, 1)
music = pad * duck * fade * 0.11

# ---------- sound effects ----------
def env_exp(n, tau): return np.exp(-np.arange(n) / SR / tau)
def tone(f, dur, tau, amp, harm=((1, 1.0),)):
    n = int(dur * SR); tt = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * k * tt) for k, a in harm)
    att = np.clip(tt / 0.004, 0, 1)
    return amp * s * env_exp(n, tau) * att
def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR); y = np.zeros_like(x); acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y
def with_echoes(x, taps):
    out = np.zeros(len(x) + int(max(d for d, _ in taps) * SR) + 1); out[:len(x)] += x
    for d, g in taps:
        i = int(d * SR); out[i:i + len(x)] += g * x
    return out
SFX = {
    'ping': with_echoes(tone(1150, 1.0, 0.30, 0.22, ((1, 1.0), (2, 0.18), (3, 0.05))), [(0.19, 0.35), (0.38, 0.14)]),
    'echo': with_echoes(tone(880, 0.8, 0.22, 0.13), [(0.15, 0.25)]),
    'hit': tone(2100, 0.2, 0.04, 0.08),
    'blip': tone(1400, 0.5, 0.10, 0.08),
}
n = int(1.1 * SR); tt = np.arange(n) / SR
sw = lowpass(rng.standard_normal(n), 900) * (np.clip(tt / 0.85, 0, 1) ** 2) * np.clip((1.1 - tt) / 0.25, 0, 1)
SFX['swell'] = 0.16 * sw / np.max(np.abs(sw)) + 0.05 * np.sin(2 * np.pi * (60 * tt + 20 * tt ** 2)) * np.clip(tt / 0.9, 0, 1) * np.clip((1.1 - tt) / 0.25, 0, 1)
n = int(3.5 * SR); tt = np.arange(n) / SR
boom = np.sin(2 * np.pi * (48 * tt - 6 * tt ** 2)) * np.exp(-tt / 1.4) + 0.5 * lowpass(rng.standard_normal(n), 300) * np.exp(-tt / 0.5)
SFX['boom'] = 0.35 * boom / np.max(np.abs(boom)) * np.clip(tt / 0.01, 0, 1)

fx = np.zeros(N)
for e in ev:
    s = SFX[e['k']]; i = int(e['t'] * SR)
    if i >= N: continue
    fx[i:i + len(s)] += s[:N - i]

# ---------- mix (stereo, light width on pad) ----------
L = voice + fx + music * 0.9 + 0.1 * np.roll(music, int(0.011 * SR))
R = voice + fx + music * 0.9 + 0.1 * np.roll(music, int(0.017 * SR))
st = np.stack([L, R], axis=1)
peak = np.max(np.abs(st))
if peak > 0.95: st *= 0.95 / peak
pcm = (np.clip(st, -1, 1) * 32767).astype('<i2')
with wave.open(os.path.join(HERE, 'mix_raw.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('wrote mix_raw.wav', round(D, 2), 's  peak', round(float(peak), 3))
