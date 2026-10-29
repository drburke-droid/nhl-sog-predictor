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

# the 50 mph shot: the ball rushing in, the thud on the road, the cone clipped and clattering away
import motion as M
from layout import LAYOUT


def dist_pan(p):
    r = math.sqrt((p[0] + 0.04) ** 2 + p[1] ** 2 + (p[2] - 1.37) ** 2)
    return r, float(np.clip(p[0] / 10, -1, 1))


def thud(f0, dur, noise_lp, gain=1.0):
    n = int(dur * SR); t = np.arange(n) / SR
    body = np.sin(2 * math.pi * f0 * t * (1 - 0.3 * t / dur)) * np.exp(-t / (dur / 4))
    hit = filt(rng.standard_normal(n), "lowpass", noise_lp) * np.exp(-t / 0.012)
    return gain * (body + 0.8 * hit)


def plastic(gain=1.0):
    n = int(0.09 * SR); t = np.arange(n) / SR
    return gain * (filt(rng.standard_normal(n), "bandpass", [350, 2200]) * np.exp(-t / 0.02) + 0.5 * np.sin(2 * math.pi * 160 * t) * np.exp(-t / 0.03))


def clang(f0, gain=1.0, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for k, (ratio, dec, amp) in enumerate(((1.0, 0.9, 1.0), (2.32, 0.6, 0.6), (3.86, 0.4, 0.45), (5.1, 0.3, 0.3), (6.7, 0.2, 0.2))):
        x += amp * np.sin(2 * math.pi * f0 * ratio * t + k) * np.exp(-t / dec)
    x += 0.8 * filt(rng.standard_normal(n), "bandpass", [800, 5000]) * np.exp(-t / 0.01)
    return gain * x


def glass(gain=1.0, dur=0.6):
    n = int(dur * SR)
    x = filt(rng.standard_normal(n) * (rng.random(n) < 0.004), "bandpass", [3000, 11000])
    return gain * x * np.exp(-np.arange(n) / (0.18 * SR))


def impacts(sim, name, min_v=1.0):
    """(time, speed) of each landing of a simulated prop: where its fall turns into a bounce"""
    t, z = sim["t"], sim[name + "_loc"][:, 2]
    vz = np.gradient(z, t)
    out = []
    i = 1
    while i < len(t) - 5:
        if vz[i] < -min_v and vz[i + 3] > vz[i] + min_v:
            out.append((t[i], -vz[i], sim[name + "_loc"][i]))
            i += 40
        i += 1
    return out


# the ball rushing in from the left (air noise rising as it nears), the thud as it lands
n = int(1.1 * SR)
tt_ = np.arange(n) / SR
xs = M.ROLL_X0 + M.ROLL_V * tt_
rush = filt(rng.standard_normal(n), "bandpass", [200, 2500]) * (12.0 / np.sqrt(xs ** 2 + 144)) ** 2 * np.clip(tt_ / 0.3, 0, 1)
for k in range(8):
    s0, s1 = k * n // 8, (k + 1) * n // 8
    fade = np.ones(s1 - s0); fade[:300] = np.linspace(0, 1, 300); fade[-300:] = np.linspace(1, 0, 300)
    add(rush[s0:s1] * fade, s0 / SR, float(np.clip(xs[(s0 + s1) // 2] / 10, -1, 1)), 0.25)
r, pan = dist_pan((M.X_LAND, 12.0, 0.0))
add(thud(70, 0.35, 900, 1.0), M.T_LAND + r / 343, pan, 0.9)
add(filt(rng.standard_normal(int(0.25 * SR)), "bandpass", [1500, 7000]) * np.exp(-np.arange(int(0.25 * SR)) / (0.05 * SR)), M.T_LAND + r / 343, pan, 0.15)
sim_r = dict(np.load("/home/user/lot/props/sim_roll.npz"))
xc = LAYOUT["cone9"][1] - (M.R_BALL + 0.07)
tc = M.T_LAND + (xc - M.X_LAND) / M.VX1
r, pan = dist_pan((LAYOUT["cone9"][1], 12.0, 0.4))
add(plastic(1.0), tc + r / 343, pan, 0.6)
for te, v, p in impacts(sim_r, "cone9"):
    r, pan = dist_pan(p)
    add(plastic(min(1.0, v / 4)), te + r / 343, pan, 0.35 * 12 / r)
# the ball's second landing, off-screen right
add(thud(60, 0.3, 700, 0.5), M.T_LAND + 2 * M.VZ1 / 9.81 + 0.06, 1.0, 0.5)

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
sim_b = dict(np.load("/home/user/lot/props/sim_blast.npz"))


def prop_sounds(t_offset, t_max=9.0):
    """every landing of every thrown prop, delayed by its distance; t_offset maps real time to the track"""
    for name, (kind, *_r) in LAYOUT.items():
        if kind not in ("cone", "drum"): continue
        for te, v, p in impacts(sim_b, name):
            if te > t_max: continue
            r, pan = dist_pan(p)
            g = min(1.0, v / 5) * 12 / max(r, 6)
            add(clang(150 + 25 * (hash(name) % 4), g) if kind == "drum" else plastic(g), t_offset + te + r / 343, pan,
                0.45 if kind == "drum" else 0.35)
    r, pan = dist_pan((M.CAR["x"], M.CAR["y"], 1.4))
    add(glass(1.0), t_offset + M.t_hit(M.CAR["x"], M.CAR["d"]) + r / 343, pan, 0.35)
    r, pan = dist_pan((M.SIGN["x"], M.SIGN["y"], 1.5))
    add(clang(95, 0.8, 1.8), t_offset + M.t_hit(M.SIGN["x"], M.SIGN["d"]) + r / 343, pan, 0.25)


prop_sounds(T_PASS)

# the replay
seg = 8
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
# in the replay, a prop's landing plays when the replay's clock reaches it
for name, (kind, *_r) in LAYOUT.items():
    if kind not in ("cone", "drum"): continue
    for te, v, p in impacts(sim_b, name):
        idx = next((i for i, (t_, s_) in enumerate(TL) if t_ >= te), None)
        if idx is None: continue
        r, pan = dist_pan(p)
        g = min(1.0, v / 5) * 12 / max(r, 6)
        add(clang(150, g) if kind == "drum" else plastic(g), T_REPLAY + idx / 25.0 + r / 343, pan, 0.35)

# master: soft limit to -1 dBFS
peak = np.abs(out).max()
out = np.tanh(out / peak * 1.4) / np.tanh(1.4) * 0.89
pcm = (out * 32767).astype(np.int16)
with wave.open("/home/user/lot/out/test2.wav", "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes(pcm.tobytes())
print("wav", DUR, "s; crack at", round(T_CRACK, 3), "slowed boom at", round(T_BOOM2, 2))
