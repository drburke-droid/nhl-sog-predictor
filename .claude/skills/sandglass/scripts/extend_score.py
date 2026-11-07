"""Lengthen a score that ends early by repeating a stretch of music where it loops cleanly.

    python3 extend_score.py IN.mp3 OUT.wav T_LO T_HI SECONDS [DURATION]
    e.g. extend_score.py take3.mp3 take3_222.wav 190 212 4.0 222

Searches insertion points t in [T_LO, T_HI] and lengths L near SECONDS for the place where the music
at t - L sounds most like the music at t (spectral match over five half-second windows), aligns the
join on the waveform, repeats [t - L, t] with a 60 ms crossfade, then trims or pads to DURATION with a
1.5 s fade. Prints the similarity (about 0.85+ is seamless; below 0.75, listen at the join).
"""
import subprocess, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
t_lo, t_hi, need = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
DUR = float(sys.argv[6]) if len(sys.argv) > 6 else 222.0
SR = 48000
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                     capture_output=True).stdout
x = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
mono = x.mean(1)
edges = np.geomspace(40, 8000, 49)


def spec(t, w=1.0):
    a = mono[int(t * SR):int((t + w) * SR)]
    s = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    f = np.fft.rfftfreq(len(a), 1 / SR)
    v = np.log1p(np.array([s[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]))
    return (v - v.mean()) / (v.std() + 1e-9)


best = None
for t_ins in np.arange(t_lo, t_hi, 0.25):
    ref = np.concatenate([spec(t_ins + d) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)])
    for L in np.arange(need - 0.35, need + 0.35, 0.01):
        c = np.concatenate([spec(t_ins - L + d) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)])
        sim = float(np.dot(ref, c) / len(ref))
        if best is None or sim > best[0]:
            best = (sim, L, t_ins)
sim, L, t_ins = best
a0, b0 = int((t_ins - L) * SR), int(t_ins * SR)
win = int(0.05 * SR)
seg = mono[b0:b0 + win]
lag = max(range(-int(0.015 * SR), int(0.015 * SR)), key=lambda k: float(np.dot(seg, mono[a0 + k:a0 + k + win])))
a0 += lag
xf = int(0.06 * SR)
ramp = np.linspace(0, 1, xf)[:, None]
head = x[:b0]
rep = x[a0:b0].copy()
rep[:xf] = head[-xf:] * (1 - ramp) + rep[:xf] * ramp
y = np.concatenate([head[:-xf], rep, x[b0:]])
n = int(DUR * SR)
y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros((n - len(y), 2), np.float32)])
fade = int(1.5 * SR)
y[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s16le", dst],
               input=y.astype(np.float32).tobytes(), check=True)
print(f"{src}: repeated {L:.3f}s ending at {t_ins:.2f}s (similarity {sim:.2f}, lag {lag / SR * 1000:.1f} ms) -> {dst}")
