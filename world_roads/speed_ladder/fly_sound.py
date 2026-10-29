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


def rocket(dur, rng, peak=420.0, crackle=0.6):
    """a rocket engine at 1 m: broadband roar peaking where the jet's big eddies shed (a few hundred Hz for the X-1's
    small chambers), and the crackle of shocklets in the jet (sparse, sharp, one-sided pops)"""
    n = int(dur * SR)
    w = rng.standard_normal(n)
    roar = filt(w, "bandpass", [peak / 4, peak * 4], 1) + 0.5 * filt(w, "bandpass", [peak, peak * 12], 2)
    roar *= wander(n, 8, 0.15, rng)
    pops = np.zeros(n)
    k = rng.random(n) < 1500 / SR
    pops[k] = rng.exponential(1.0, k.sum()) ** 2
    pops = filt(pops, "highpass", 900, 2)
    return 1.4 * roar + crackle * pops


def n_wave(dp, length, rise=0.0002):
    """a sonic boom's N-wave: a jump up, a straight fall through zero to as far below, a jump back"""
    n = int((length + 2 * rise) * SR)
    t = np.arange(n) / SR
    x = np.where(t < rise, t / rise, np.where(t < length + rise, 1 - 2 * (t - rise) / length,
                                              np.maximum(-1 + (t - length - rise) / rise, 0) * 0 - 1))
    x[int((length + 2 * rise) * SR) - int(rise * SR):] *= np.linspace(1, 0, int(rise * SR))
    return dp * x


def xlr99(dur, rng):
    """the X-15's XLR99 (250 kN): a far bigger jet, so its roar sits lower"""
    return rocket(dur, rng, peak=160.0, crackle=0.9)


def f135(dur, rng):
    """the F-35's F135 in full afterburner (190 kN): a jet's roar, lower and broader than a rocket's, and the
    afterburner's crackle"""
    return rocket(dur, rng, peak=260.0, crackle=1.1)


SOURCES = {"wright": wright_engine, "spitfire": merlin, "x1": rocket, "f35": f135, "x15": xlr99}
LEVEL = {"wright": 2.2, "spitfire": 9.0, "x1": 14.0, "f35": 22.0, "x15": 30.0}  # at 1 m, relative: 12 hp against 1,600 hp ...
BOOM = {"x1": dict(dp=1.6, length=0.016), "f35": dict(dp=2.6, length=0.024)}             # relative to the sound levels above; N-wave length s


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
    if name in BOOM:
        out += boom(name, n0, int(dur * SR), rng)
    if name == "x15":
        out += blast_events(n0, int(dur * SR), rng)
    return out


# ---------------------------------------------------------------- the X-15's blast and what it throws
def place(out, sig, t0, pan=0.0, gain=1.0):
    i0 = int(t0 * SR)
    if i0 >= len(out) or i0 + len(sig) <= 0: return
    if i0 < 0: sig, i0 = sig[-i0:], 0
    m = min(len(sig), len(out) - i0)
    out[i0:i0 + m, 0] += gain * math.cos((pan + 1) * math.pi / 4) * sig[:m]
    out[i0:i0 + m, 1] += gain * math.sin((pan + 1) * math.pi / 4) * sig[:m]


def arrive(p, te):
    """(arrival time, pan, distance) at the camera of a sound made at p at flight time te"""
    d = float(np.linalg.norm(np.asarray(p, float) - CAM))
    return te + d / C_SOUND, float(np.clip((p[0] - CAM[0]) / 10, -1, 1)), d


def plastic(rng, gain=1.0):
    n = int(0.09 * SR); t = np.arange(n) / SR
    return gain * (filt(rng.standard_normal(n), "bandpass", [350, 2200]) * np.exp(-t / 0.02) +
                   0.5 * np.sin(2 * math.pi * 160 * t) * np.exp(-t / 0.03))


def clang(rng, f0, gain=1.0, dur=1.4):
    n = int(dur * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for k, (ratio, dec, amp) in enumerate(((1.0, 0.9, 1.0), (2.32, 0.6, 0.6), (3.86, 0.4, 0.45), (5.1, 0.3, 0.3))):
        x += amp * np.sin(2 * math.pi * f0 * ratio * t + k) * np.exp(-t / dec)
    return gain * (x + 0.8 * filt(rng.standard_normal(n), "bandpass", [800, 5000]) * np.exp(-t / 0.01))


def glass(rng, gain=1.0, dur=0.7):
    n = int(dur * SR)
    x = filt(rng.standard_normal(n) * (rng.random(n) < 0.005), "bandpass", [3000, 11000])
    return gain * x * np.exp(-np.arange(n) / (0.2 * SR))


def impacts(sim, name, min_v=1.0):
    """(time, speed, position) of each landing of a simulated prop: where its fall turns into a bounce"""
    t, z = sim["t"], sim[name + "_loc"][:, 2]
    vz = np.gradient(z, t)
    out, i = [], 1
    while i < len(t) - 5:
        if vz[i] < -min_v and vz[i + 3] > vz[i] + min_v:
            out.append((t[i], -vz[i], sim[name + "_loc"][i])); i += 40
        i += 1
    return out


def blast_events(n0, n, rng):
    """the blast reaching the camera (a crack, 80 Pa s: deafening, the boom and its rumble rolling off the land), the
    car's windows bursting, the sign's post folding, every landing of every thrown cone and drum, the gravel raining
    down, the dust hissing"""
    import hyper as Hy
    out = np.zeros((n, 2))
    off = n0 / F.FPS
    t_cam = float(Hy.t_hit(CAM[0], CAM[1], CAM[2])) + off
    m = int(0.02 * SR); tt = np.arange(m) / SR
    crack = np.where(tt < 0.004, 1 - 2 * tt / 0.004, 0.0) + 0.7 * filt(rng.standard_normal(m), "highpass", 700) * np.exp(-tt / 0.004)
    place(out, crack, t_cam, 0.0, 3.0)
    m = int(4.0 * SR); tt = np.arange(m) / SR
    f = 30 + 45 * np.exp(-tt / 0.25)
    rumble = 0.9 * np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.8) + \
        1.8 * filt(rng.standard_normal(m), "lowpass", 200) * np.exp(-tt / 1.3) * np.minimum(1, tt / 0.01)
    for pan in (-0.5, 0.5):
        place(out, rumble * rng.uniform(0.9, 1.1), t_cam + 0.003, pan, 1.6)
    from layout import LAYOUT
    if "car" in LAYOUT:
        car = (Hy.CAR["x"], Hy.CAR["y"], 1.0)
        ta, pan, d = arrive(car, Hy.first_hit())
        place(out, glass(rng), ta + 0.002, pan, 0.6 * 20 / d)
        place(out, clang(rng, 95, 0.5, 0.8), ta, pan, 20 / d)                 # the body slammed onto its stops
    if "sign" in LAYOUT:
        ta, pan, d = arrive((Hy.SIGN["x"], Hy.SIGN["y"], 2.0), float(Hy.t_hit(Hy.SIGN["x"], Hy.SIGN["y"], 2.0)))
        place(out, clang(rng, 180, 0.4, 1.0), ta + 0.03, pan, 12 / d)
    simf = "/home/user/lot/props/sim_x15.npz"
    sim = dict(np.load(simf)) if LAYOUT and os.path.exists(simf) else {}
    for name in [k[:-4] for k in sim if k.endswith("_loc")]:
        for te, v, pos in impacts(sim, name):
            ta, pan, d = arrive(pos, te)
            g = min(1.5, 0.12 * v) * 12 / max(d, 2.0)
            place(out, clang(rng, rng.uniform(70, 110), g, 1.0) if name.startswith("drum") else plastic(rng, g),
                  ta + off, pan)
    for _ in range(320):                                                      # gravel raining back down
        tc = t_cam + rng.gamma(3.0, 0.35)
        mm = int(0.004 * SR)
        click = filt(rng.standard_normal(mm), "bandpass", [1200 + 3000 * rng.random(), 9000]) * np.exp(-np.arange(mm) / (0.0008 * SR))
        place(out, click, tc, float(rng.uniform(-1, 1)), 0.06 * rng.lognormal(0, 0.6))
    m = int(3.0 * SR)
    for pan in (-0.4, 0.4):
        place(out, filt(rng.standard_normal(m), "bandpass", [2000, 9000]) * np.exp(-np.arange(m) / (1.0 * SR)), t_cam + 0.05, pan, 0.1)
    return out


def boom(name, n0, n, rng):
    """the Mach cone reaching the camera: the N-wave, its echo off the ground a moment later, then the rumble of the
    boom coming back off the land around"""
    b = BOOM[name]
    t_cone, r = cone_arrival(name, n0)
    out = np.zeros((n, 2))
    nw = n_wave(b["dp"], b["length"])
    for dt, g in ((0.0, 1.0), (0.0035, 0.8)):
        i = int((t_cone + dt) * SR)
        m = min(len(nw), n - i)
        if m > 0: out[i:i + m] += g * nw[:m, None] * np.array([0.9, 1.0])
    i = int(t_cone * SR)
    m = n - i
    if m > 0:
        tt = np.arange(m) / SR
        rum = filt(rng.standard_normal((m, 2)), "lowpass", 160, 2) * np.exp(-tt / 0.9)[:, None]
        rum += 0.4 * filt(rng.standard_normal((m, 2)), "bandpass", [200, 1200], 2) * np.exp(-tt / 0.25)[:, None]
        out[i:] += 0.9 * b["dp"] * rum * np.minimum(1, tt / 0.02)[:, None]
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
