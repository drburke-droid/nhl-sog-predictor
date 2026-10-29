"""Sound for the flybys, synthesized. Each machine is heard from the camera as it really would be: what it emits at one
point arrives distance / 343 m/s later (which gives the Doppler shift), 1/distance loud, its highs soaked up by the air
with distance, panned by where it is. Desert wind under everything.
  python fly_sound.py wright -> /home/user/lot/out/fly_wright.wav (the shot's length)"""
import numpy as np, math, sys, os, wave
from scipy.signal import butter, sosfilt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, flight as F

SR = 48000
C_SOUND = 343.0
CAM = np.array([Cam.CAM_X, 0.0, Cam.H])


def filt(x, kind, f, order=4):
    return sosfilt(butter(order, f, btype=kind, fs=SR, output="sos"), x, axis=0)


def wander(n, rate, depth, rng):
    """slow random variation around 1"""
    k = max(4, int(n / SR * rate) + 4)
    pts = rng.standard_normal(k)
    return 1 + depth * np.interp(np.arange(n) / SR * rate, np.arange(k), pts)


def pulses(n, f, rng, jitter=0.03, f_wander=0.02):
    """times (s) of a train of events at about f per second, wandering and jittered"""
    out, t = [], 0.0
    fw = wander(n, 0.7, f_wander, rng)
    while t < n / SR:
        out.append(t)
        t += (1 / (f * fw[min(n - 1, int(t * SR))])) * (1 + jitter * rng.standard_normal())
    return np.array(out)


def stamp(sig, times, grain, amps):
    for t, a in zip(times, amps):
        i = int(t * SR)
        m = min(len(grain), len(sig) - i)
        if m > 0: sig[i:i + m] += a * grain[:m]


def wright_engine(dur, rng):
    """the Flyer at 1 m: its 12 hp four-cylinder engine (about 1,020 rpm, 34 firings a second, open exhaust ports),
    clattering valves, the chains to the propellers, and the two propellers (350 rpm, two blades each)"""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    sig = np.zeros(n)
    # exhaust: a sharp bark per firing, a band of noise plus a low thump, uneven from one to the next
    g = int(0.03 * SR); tg = np.arange(g) / SR
    fire = pulses(n, 34.0, rng)
    for t in fire:
        nb = filt(rng.standard_normal(g), "bandpass", [180, 2600], 2) * np.exp(-tg / 0.009)
        thump = np.sin(2 * math.pi * 72 * tg) * np.exp(-tg / 0.014)
        i = int(t * SR); m = min(g, n - i)
        if m > 0: sig[i:i + m] += rng.uniform(0.7, 1.25) * (nb[:m] * 1.4 + thump[:m] * 0.8)
    # valves: ticks, half the firing rate per cam, a little out of step
    c = int(0.003 * SR)
    tick = filt(rng.standard_normal(c), "bandpass", [2500, 7000], 2) * np.exp(-np.arange(c) / (0.0007 * SR))
    stamp(sig, pulses(n, 68.0, rng, 0.05), tick, rng.uniform(0.08, 0.2, 100000))
    # chains: a rattle with the links' rate in it
    ch = filt(rng.standard_normal(n), "bandpass", [1800, 6000], 2) * (0.6 + 0.4 * np.sin(2 * math.pi * 96 * tt) ** 2)
    sig += 0.06 * ch * wander(n, 3, 0.3, rng)
    # propellers: a soft whup at each blade's pass, the two a little apart in phase
    air = filt(rng.standard_normal(n), "lowpass", 380, 2)
    for ph in (0.0, 0.37):
        bl = 0.5 + 0.5 * np.cos(2 * math.pi * (2 * 350 / 60) * tt + 2 * math.pi * ph)
        sig += 0.55 * air * bl ** 6
    return sig


def merlin(dur, rng, rpm=3000, prop_rpm=1430):
    """a Spitfire at 1 m: the Rolls-Royce Merlin V12 at combat power (3,000 rpm: 300 firings a second; the small
    differences between cylinders put in every multiple of half the crank's rate, 25 Hz), its stub exhausts' roar,
    and the four-bladed propeller (about 1,430 rpm: a blade every 10.5 ms) buzzing with its tips near the speed of sound"""
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f0 = rpm / 120.0                                          # the half-order, 25 Hz
    fw = wander(n, 0.5, 0.004, rng)
    ph = 2 * math.pi * np.cumsum(f0 * fw) / SR
    sig = np.zeros(n)
    for k in range(1, 160):
        f = k * f0
        if f > 9000: break
        a = 0.25 * rng.uniform(0.3, 1.0) / (1 + (f / 900) ** 1.4)
        if k % 6 == 0: a *= 3.5                             # the firing rate's harmonics (every 150 Hz)
        if k % 12 == 0: a *= 1.6                            # 300 Hz and up
        sig += a * np.sin(k * ph + rng.uniform(0, 2 * math.pi))
    roar = filt(rng.standard_normal(n), "bandpass", [90, 4000], 2)
    sig += 1.3 * roar * (0.55 + 0.45 * np.sin(12 * ph) ** 2)
    php = 2 * math.pi * np.cumsum(4 * prop_rpm / 60 * fw) / SR
    saw = sum(np.sin(k * php) / k ** 1.1 for k in range(1, 40))
    sig += 1.6 * saw * (0.8 + 0.2 * wander(n, 2, 0.5, rng))
    return sig


SOURCES = {"wright": wright_engine, "spitfire": merlin}
LEVEL = {"wright": 2.2, "spitfire": 9.0}            # at 1 m, relative: a 12 hp engine against a 1,600 hp one


def path(name, te):
    m = F.MACHINES[name]
    return np.stack([m["speed"] * te, np.full(len(te), F.FLY_Y), np.full(len(te), m["alt"])], 1)


def heard(name, src, t_src0, n0, dur, over=None):
    """stereo at the camera for the shot's clock 0..dur; src emitted from flight time t_src0 on (sampled at SR); the
    flight's clock is the shot's minus n0 / 25.  Every bit of sound is carried from where and when it left to when it
    arrives (splatted onto the arrival clock), so the Doppler shift, its change of loudness, and above the speed of
    sound the silence before the Mach cone and the two branches after it, come out by themselves."""
    m = F.MACHINES[name]
    M = m["speed"] / C_SOUND
    over = over or int(min(16, max(2, math.ceil(2 * (1 + M)))))           # emission samples per output sample
    ns = len(src) * over
    te = t_src0 + np.arange(ns) / (SR * over)
    s = np.interp(np.arange(ns) / over, np.arange(len(src)), src)
    P = path(name, te)
    r = np.linalg.norm(P - CAM, axis=1)
    tau = te + r / C_SOUND + n0 / F.FPS                                    # arrival on the shot's clock
    k = np.clip((r - 15) / 120, 0, 1)
    s = (s * (1 - k) + filt(s, "lowpass", 1800, 2) * k) / r / over
    pan = np.clip((P[:, 0] - CAM[0]) / r, -1, 1) * 0.85
    n = int(dur * SR)
    out = np.zeros((n, 2))
    pos = tau * SR
    i = np.floor(pos).astype(np.int64); f = pos - i
    for ch, g in ((0, np.cos((pan + 1) * math.pi / 4)), (1, np.sin((pan + 1) * math.pi / 4))):
        for di, w in ((0, 1 - f), (1, f)):
            j = i + di
            ok = (j >= 0) & (j < n)
            out[:, ch] += np.bincount(j[ok], weights=(s * g * w)[ok], minlength=n)[:n]
    return out


def cone_arrival(name, n0):
    """when the Mach cone (or for a slower machine, its closest approach) reaches the camera, on the shot's clock"""
    m = F.MACHINES[name]
    te = np.linspace(-2, 2, 400001)
    P = path(name, te)
    tau = te + np.linalg.norm(P - CAM, axis=1) / C_SOUND
    i = int(np.argmin(tau))
    return float(tau[i] + n0 / F.FPS), float(np.linalg.norm(P[i] - CAM))


def wind(n, rng, gain=0.035):
    tt = np.arange(n) / SR
    w = filt(rng.standard_normal((n, 2)), "lowpass", 500)
    return gain * w * (0.6 + 0.4 * np.sin(2 * math.pi * 0.13 * tt + 1.0))[:, None]


def write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((x * 32767).astype("<i2").tobytes())


def shot(name, seed=3):
    rng = np.random.default_rng(seed)
    fr = F.frames(name)
    dur = len(fr) / F.FPS
    n0 = round(F.MACHINES[name]["pre"] * F.FPS)
    t_src0 = -n0 / F.FPS - 1.0                                    # emitted from a second before the shot begins
    src = SOURCES[name](dur + 2.0, rng)
    out = wind(int(dur * SR), rng) + LEVEL[name] * heard(name, src, t_src0, n0, dur)
    return out


def replay_audio(name, seed=5):
    """the replay's sound: what a microphone heard in real time, played back as slowly as the pictures (a recording
    at 16 times slower drops four octaves: the Merlin's 300 Hz firing becomes a 19 Hz growl, each propeller blade a
    separate thump), the wind barely moving"""
    rng = np.random.default_rng(seed)
    tl = F.replay(name)
    t0 = tl[0][0] - 0.3
    dur_real = tl[-1][0] - t0 + 0.1
    src = SOURCES[name](dur_real + 3.0, rng)
    h = LEVEL[name] * heard(name, src, t0 - 1.5, -t0 * F.FPS, dur_real)        # h[i]: flight time t0 + i / SR
    T = np.arange(int(len(tl) / F.FPS * SR)) / SR
    tk = np.interp(T * F.FPS, np.arange(len(tl)), [t for t, _ in tl])
    idx = (tk - t0) * SR
    out = np.stack([np.interp(idx, np.arange(len(h)), h[:, c]) for c in (0, 1)], 1)
    return out + wind(len(T), rng, 0.012)


def normalize(x, rms=0.16, win=0.25):
    """loudest quarter-second at the given RMS, peaks rounded off"""
    k = int(win * SR)
    loud = max(np.sqrt((x[i:i + k] ** 2).mean()) for i in range(0, len(x) - k, k // 2))
    return np.tanh(x * rms / loud * 1.2) / 1.2


if __name__ == "__main__":
    name = sys.argv[1]
    x = normalize(shot(name))
    if name in F.REPLAY:
        x = np.concatenate([x, normalize(replay_audio(name), rms=0.13)])
    print(name, "peak", round(float(np.abs(x).max()), 3))
    write(f"/home/user/lot/out/fly_{name}.wav", x)
