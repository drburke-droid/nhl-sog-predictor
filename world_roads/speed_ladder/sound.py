"""A sound track for the test clip (roll, real-time pass, replay), synthesized.
  python sound.py total_seconds -> /home/user/lot/out/test2.wav

Real time: desert wind; the ball rolling past at 50 mph; at 17,500 mph the crack of the shock arriving 35 ms after the
pass (12 m at 343 m/s), the boom and rumble of the blast, gravel landing over the next second or two, the hiss of
dust. Replay: the world muffled; a low drone and the sizzle of the plasma swelling as the ball crosses; silence as the
shock travels toward the camera; a slowed, deeper boom when it arrives; the dust rumbling on."""
import numpy as np, math, sys, os
from scipy.signal import butter, sosfilt
import wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_time as T
SR = 48000
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 14.2
N = int(DUR * SR)
rng = np.random.default_rng(9)
out = np.zeros((N, 2), np.float64)
tt = np.arange(N) / SR

T_ROLL = 0.98                  # the ball passes the middle of the frame
T_PASS = (59 + 25) / 25.0      # the hypersonic pass (real time)
T_CRACK = T_PASS + 12 / 343.0
T_REPLAY = 145 / 25.0
TL = T.timeline()
T_BOOM2 = T_REPLAY + 179 / 25.0
T_END_A = T_REPLAY + sum(1 for _, s in TL if s == T.SLOW) / 25.0


def filt(x, kind, f, order=4):
    sos = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


def add(sig, t0, pan=0.0, gain=1.0):
    i0 = int(t0 * SR)
    if i0 >= N: return
    sig = sig[:max(0, N - max(i0, 0))] if i0 >= 0 else sig[-i0:][:N]
    i0 = max(i0, 0)
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    out[i0:i0 + len(sig), 0] += gain * l * sig
    out[i0:i0 + len(sig), 1] += gain * r * sig


def env(n, a, d):
    e = np.ones(n)
    na = max(1, int(a * SR)); e[:na] = np.linspace(0, 1, na)
    e[na:] = np.exp(-np.arange(n - na) / (d * SR))
    return e


# wind, ducked and muffled in the slow-motion part
w = filt(rng.standard_normal((N, 2)), "lowpass", 500)
w *= (0.6 + 0.4 * np.sin(2 * math.pi * 0.13 * tt + 1.0))[:, None]
duck = np.ones(N)
a, b = int(T_REPLAY * SR), int(T_BOOM2 * SR)
duck[a:b] = 0.35
duck[int((T_BOOM2 - 0.6) * SR):b] = np.linspace(0.35, 0.05, b - int((T_BOOM2 - 0.6) * SR))
out += 0.035 * w * duck[:, None]

# the roll: a steel ball on asphalt at 22 m/s, panned with it, louder as it nears
n = int(1.6 * SR)
x = np.linspace(-17.9, 17.9, n)             # its position over 1.6 s at 22.35 m/s
r = np.sqrt(x ** 2 + 12.0 ** 2)
roar = filt(rng.standard_normal(n), "bandpass", [70, 900]) * (12.0 / r) ** 1.5
grit = filt(rng.standard_normal(n), "bandpass", [1500, 6000]) * (12.0 / r) ** 2 * 0.35
seg = 8
for k in range(seg):                       # crude panning: eight segments
    s0, s1 = k * n // seg, (k + 1) * n // seg
    fade = np.ones(s1 - s0); fade[:200] = np.linspace(0, 1, 200); fade[-200:] = np.linspace(1, 0, 200)
    add((roar + grit)[s0:s1] * fade, T_ROLL - 0.8 + s0 / SR, pan=float(np.clip(x[(s0 + s1) // 2] / 10, -1, 1)), gain=0.35)

# the real-time pass: crack, boom, rumble, gravel, hiss
def crack(scale=1.0):
    n = int(0.012 * SR)
    t = np.arange(n) / SR
    nwave = np.where(t < 0.0015, 1 - 2 * t / 0.0015, 0.0)            # an N-wave, 1.5 ms
    burst = filt(rng.standard_normal(n), "highpass", 800) * np.exp(-t / 0.003)
    return scale * (nwave + 0.6 * burst)

def boom(f0, f1, dur, rumble_dur, lp=180):
    n = int(rumble_dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / (dur / 3))
    ph = 2 * math.pi * np.cumsum(f) / SR
    tone = np.sin(ph) * np.exp(-t / dur)
    rum = filt(rng.standard_normal(n), "lowpass", lp) * env(n, 0.01, rumble_dur / 3)
    return 0.9 * tone + 1.6 * rum

add(crack(1.0), T_CRACK, 0.0, 0.9)
add(boom(70, 32, 0.7, 3.0), T_CRACK + 0.004, 0.0, 0.8)
n = int(2.5 * SR)
for pan in (-0.4, 0.4):                    # dust hiss, decorrelated left and right
    add(filt(rng.standard_normal(n), "bandpass", [2000, 9000]) * env(n, 0.05, 0.9), T_CRACK + 0.05, pan, 0.08)

def patter(t0, dur, count, gain):
    for _ in range(count):
        tc = t0 + rng.gamma(3.0, dur / 4)
        m = int(0.004 * SR)
        click = filt(rng.standard_normal(m), "bandpass", [1200 + 3000 * rng.random(), 9000]) * np.exp(-np.arange(m) / (0.0008 * SR))
        add(click, tc, float(rng.uniform(-1, 1)), gain * rng.lognormal(0, 0.6))
patter(T_PASS + 0.45, 1.6, 260, 0.05)

# the replay
n = int((T_END_A - T_REPLAY) * SR)
t = np.arange(n) / SR
swell = np.exp(-((t - 68 / 25.0) / 1.6) ** 2)          # loudest as the ball passes mid-frame (replay frame 68)
drone = (np.sin(2 * math.pi * 41 * t) + 0.8 * np.sin(2 * math.pi * 43.3 * t) + 0.4 * np.sin(2 * math.pi * 82 * t)) * (0.3 + 0.7 * swell)
whoosh = filt(rng.standard_normal(n), "bandpass", [60, 500]) * swell
sizzle = filt(rng.standard_normal(n) * (rng.random(n) < 0.02), "bandpass", [1500, 5000]) * swell
for k in range(seg):
    s0, s1 = k * n // seg, (k + 1) * n // seg
    fade = np.ones(s1 - s0); fade[:400] = np.linspace(0, 1, 400); fade[-400:] = np.linspace(1, 0, 400)
    pan = float(np.clip((T.ball_x(TL[min(int((s0 + s1) / 2 / SR * 25), len(TL) - 1)][0])) / 10, -1, 1))
    add((0.25 * drone + 0.5 * whoosh + 0.6 * sizzle)[s0:s1] * fade, T_REPLAY + s0 / SR, pan, 0.5)
# the ramp toward the boom: a low rising rumble, then the slowed boom
n = int((T_BOOM2 - T_END_A) * SR)
rise = filt(rng.standard_normal(n), "lowpass", 120) * np.linspace(0.1, 0.6, n)
add(rise, T_END_A, 0.0, 0.5)
add(crack(0.5), T_BOOM2, 0.0, 0.5)
add(boom(40, 18, 1.6, 4.0, lp=110), T_BOOM2 + 0.01, 0.0, 1.1)
patter(T_BOOM2 + 0.3, 0.9, 120, 0.04)

# master: soft limit to -1 dBFS
peak = np.abs(out).max()
out = np.tanh(out / peak * 1.4) / np.tanh(1.4) * 0.89
pcm = (out * 32767).astype(np.int16)
with wave.open("/home/user/lot/out/test2.wav", "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
print("wav", DUR, "s; crack at", round(T_CRACK, 3), "slowed boom at", round(T_BOOM2, 2))
