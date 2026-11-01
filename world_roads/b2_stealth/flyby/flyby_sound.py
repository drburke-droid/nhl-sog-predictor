"""The B-2 heard from the camera: four F118 turbofans (no afterburner), carried to the camera at the speed of sound.

The sound leaving the aircraft at each instant is splatted onto the moment it arrives (emission time + distance / c),
as in the speed-ladder flybys (`speed_ladder/fly_sound.heard`), so the Doppler shift, the swell and the drop in pitch
as it passes all come out by themselves. Two sources with their own directivity:

* the fans' whine (blade-passing tone near 2.6 kHz with harmonics, and the buzz-saw tones at multiples of the shaft
  rate that a fan with supersonic blade tips makes): loudest ahead of the aircraft and fading behind it;
* the jets' roar (broadband, peaking a few hundred hertz): quiet ahead, loudest 120-150 degrees from the nose,
  so the full roar arrives just after the aircraft has gone by.

Air soaks up high frequencies with distance (a low-pass mixed in by range). On the ground (the side shot) the sound
also reaches the camera off the ground, a few milliseconds later: the two paths beating give a flyby's sweeping
"flanger" sound. Desert or airfield wind underneath.

    python flyby_sound.py topdown | side     -> $B2_WORK/<shot>/<shot>.wav
"""
import math, os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 48000
C_SOUND = 343.0


def shaped_noise(n, rng, fn):
    """white noise with spectrum magnitude fn(f)"""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    y = np.fft.irfft(X * fn(np.maximum(f, 1.0)), n)
    return y / (np.std(y) + 1e-12)


def lowpass(x, fc):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 4), len(x))


def wander(n, rng, rate, depth):
    k = max(4, int(n / SR * rate) + 4)
    pts = rng.standard_normal(k)
    return 1 + depth * np.interp(np.linspace(0, k - 3, n), np.arange(k), pts)


def jet_roar(n, rng):
    roar = shaped_noise(n, rng, lambda f: (f / 250.0) ** 1.0 / (1 + (f / 250.0) ** 2.6))
    rumble = shaped_noise(n, rng, lambda f: 1 / (1 + (f / 70.0) ** 3))
    crackle = shaped_noise(n, rng, lambda f: (f / 1500.0) / (1 + (f / 1500.0) ** 2.2))
    return (roar + 0.7 * rumble + 0.25 * crackle * wander(n, rng, 6, 0.6)) * wander(n, rng, 2, 0.15)


def fan_whine(n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    bpf = 2600.0 * wander(n, rng, 0.5, 0.004)
    ph = 2 * np.pi * np.cumsum(bpf) / SR
    for h, a in ((1, 1.0), (2, 0.45), (3, 0.15)):
        out += a * np.sin(h * ph + rng.uniform(0, 6.3))
    shaft = 125.0 * wander(n, rng, 0.4, 0.003)
    ps = 2 * np.pi * np.cumsum(shaft) / SR
    for k in range(6, 30):
        out += 0.18 * rng.uniform(0.2, 1.0) * np.sin(k * ps + rng.uniform(0, 6.3)) * wander(n, rng, 3, 0.3)
    hiss = shaped_noise(n, rng, lambda f: np.exp(-((np.log(f / 3500.0)) ** 2) / 0.5))
    return out / 3.0 + 0.35 * hiss


def directivity(theta, kind):
    """theta: angle between the nose and the line to the listener (0 ahead, pi behind)"""
    if kind == "fan":
        return 0.25 + 0.75 * np.clip(np.cos(theta / 2), 0, 1) ** 2
    return 0.12 + np.exp(-((theta - math.radians(135)) / math.radians(32)) ** 2)


def heard(src_fn, kinds, obs, dur, t_lead=3.0, ground=False, over=2, absorb=True):
    """stereo heard at obs over the shot's clock 0..dur.  src_fn(te) -> (positions (n,3), velocity (3,));
    kinds: [(signal sampled at SR from emission time -t_lead on, directivity kind, level)]"""
    ns = int((dur + t_lead) * SR)
    te = -t_lead + np.arange(ns * over) / (SR * over)
    P, vel = src_fn(te)
    vdir = vel / np.linalg.norm(vel)
    n = int(dur * SR)
    out = np.zeros((n, 2))
    images = [(P, 1.0)]
    if ground:
        Pg = P.copy(); Pg[:, 2] = -Pg[:, 2]
        images.append((Pg, 0.75))
    for Pk, g_img in images:
        d = obs[None, :] - Pk
        r = np.linalg.norm(d, axis=1)
        cos_th = (d @ vdir) / r
        theta = np.arccos(np.clip(cos_th, -1, 1))
        tau = te + r / C_SOUND
        pan = np.clip(-d[:, 0] / r, -1, 1) * 0.8           # source to the camera's left/right (x across the picture)
        k_air = np.clip((r - 100) / 900, 0, 1) if absorb else 0 * r
        for sig, kind, level in kinds:
            s = np.interp(np.arange(ns * over) / over, np.arange(len(sig)), sig)
            s_lp = np.interp(np.arange(ns * over) / over, np.arange(len(sig)), lowpass(sig, 1200.0))
            s = (s * (1 - k_air) + s_lp * k_air) * directivity(theta, kind) * level * g_img / r / over
            pos = tau * SR
            i = np.floor(pos).astype(np.int64); f = pos - i
            for ch, gch in ((0, np.cos((pan + 1) * math.pi / 4)), (1, np.sin((pan + 1) * math.pi / 4))):
                for di, w in ((0, 1 - f), (1, f)):
                    j = i + di; ok = (j >= 0) & (j < n)
                    out[:, ch] += np.bincount(j[ok], weights=(s * gch * w)[ok], minlength=n)[:n]
    return out


def wind(n, rng, gain):
    w = np.stack([shaped_noise(n, rng, lambda f: 1 / (1 + (f / 300.0) ** 2)) for _ in range(2)], 1)
    return gain * w * wander(n, rng, 0.3, 0.4)[:, None]


def write(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def shot_sound(shot, seed=7):
    rng = np.random.default_rng(seed)
    if shot == "topdown":
        import topdown as M
        S = M.S; v = M.v_ms()
        obs = np.array([0.0, 0.0, S["cam_h"]])
        src = lambda te: (np.stack([v * (te - S["t_pass"]), np.full(len(te), S["y_off"]), np.full(len(te), S["alt"])], 1),
                          np.array([v, 0, 0]))
        ground, amb = False, 0.006
    else:
        import side as M
        S = M.S; v = M.v_ms()
        obs = np.array([0.0, 0.0, S["cam_z"]])
        src = lambda te: (np.stack([v * (te - S["t_pass"]), np.full(len(te), S["dist"]), np.full(len(te), S["alt"])], 1),
                          np.array([v, 0, 0]))
        ground, amb = True, 0.012
    dur = S["dur"]; t_lead = 4.0
    ns = int((dur + t_lead) * SR)
    kinds = [(jet_roar(ns, rng), "jet", 1.0), (fan_whine(ns, rng), "fan", 0.55)]
    x = heard(src, kinds, obs, dur, t_lead, ground=ground)
    x /= np.max(np.abs(x)) + 1e-9
    x = 0.89 * x + wind(len(x), rng, amb)
    fade = np.clip(np.arange(len(x)) / (0.15 * SR), 0, 1) * np.clip((len(x) - np.arange(len(x))) / (0.3 * SR), 0, 1)
    x *= fade[:, None]
    out = os.path.join(M.OUT, S["name"] + ".wav")
    write(out, x * 0.9)
    print("wrote", out)


if __name__ == "__main__":
    shot_sound(sys.argv[1])
