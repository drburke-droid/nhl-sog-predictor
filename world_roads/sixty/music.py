"""The score: 60 beats a minute, so every beat is one second. A soft clock tick on each second (it lands with
the second hands on screen), a plucked lyre over a breathing pad in D dorian, and a small bell at each full
minute of the film. Everything is synthesized (additive partials, filtered noise, a synthetic room). No samples.

    python music.py            # writes out/score.wav for the timeline in assemble.py
"""
import os, sys, math
import numpy as np
from scipy.signal import fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SR = 48000
NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(note):
    n, acc, octv = note[0], note[1:-1], int(note[-1])
    k = NAMES[n] + {"": 0, "b": -1, "#": 1}[acc] + 12 * (octv + 1)
    return 440.0 * 2 ** ((k - 69) / 12)


def midi(note):
    return round(69 + 12 * math.log2(hz(note) / 440.0))


# ------------------------------------------------------------------ instruments
def lyre(f0, vel, seed=0, ring=4.0):
    """a plucked gut string: bright attack, upper partials die fast, a little body resonance"""
    rng = np.random.default_rng(seed)
    n = int(ring * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    pluck = 0.18                                          # plucked near the end: strong upper partials
    for k in range(1, 16):
        fk = k * f0 * (1 + 0.00012 * k * k)
        if fk > 9000:
            break
        amp = abs(math.sin(math.pi * k * pluck)) / k ** 1.05
        tau = 1.6 * (220 / f0) ** 0.4 / (1 + 0.35 * (k - 1))
        out += (amp * np.exp(-t / tau) * np.sin(2 * math.pi * fk * t + rng.uniform(0, 6.28))).astype(np.float32)
    out *= (1 - np.exp(-t / 0.0015)).astype(np.float32)
    m = int(0.012 * SR)
    out[:m] += rng.normal(0, 1, m).astype(np.float32) * np.exp(-np.arange(m) / (0.002 * SR)).astype(np.float32) * 0.05
    out = lfilter([0.5], [1, -0.5], out).astype(np.float32)
    return out * (0.3 + 0.7 * vel) ** 1.3


def pad_tone(f0, dur, fade_in=3.0, fade_out=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int((dur + fade_out) * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    for det in (-0.10, 0.0, 0.11):
        f = f0 * 2 ** (det / 12)
        for k in range(1, 6):
            out += (0.55 ** (k - 1) / k) * np.sin(2 * math.pi * f * k * t + rng.uniform(0, 6.28)).astype(np.float32)
    lfo = 1 + 0.12 * np.sin(2 * math.pi * 0.09 * t + rng.uniform(0, 6.28))
    env = np.clip(t / fade_in, 0, 1) ** 2 * np.clip((dur + fade_out - t) / fade_out, 0, 1) ** 1.5
    return lfilter([0.05], [1, -0.95], out * env * lfo).astype(np.float32)


def tick(hi=True, seed=0):
    """a clock tick: a tiny woody click"""
    rng = np.random.default_rng(seed)
    n = int(0.05 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 2600.0 if hi else 2150.0
    x = np.sin(2 * math.pi * f * t) * np.exp(-t / 0.004) + rng.normal(0, 1, n) * np.exp(-t / 0.0015) * 0.4
    x += 0.5 * np.sin(2 * math.pi * f * 0.47 * t) * np.exp(-t / 0.009)
    return (x * 0.5).astype(np.float32)


def bell(f0=880.0, seed=0):
    """a small bell: inharmonic partials, long ring"""
    rng = np.random.default_rng(seed)
    n = int(6.0 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    for r, a, tau in ((0.5, 0.5, 3.5), (1.0, 1.0, 2.6), (1.19, 0.5, 1.8), (1.56, 0.45, 1.4), (2.0, 0.35, 1.0), (2.51, 0.25, 0.7),
                      (3.01, 0.15, 0.5)):
        out += a * np.exp(-t / tau) * np.sin(2 * math.pi * f0 * r * t + rng.uniform(0, 6.28))
    out *= (1 - np.exp(-t / 0.002))
    return out.astype(np.float32)


def reverb_ir(rt60=2.6, length=4.0, seed=4):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    irs = []
    for ch in range(2):
        noise = rng.normal(0, 1, n)
        bright = noise * np.exp(-6.9 * t / (rt60 * 0.5))
        dark = lfilter([0.04], [1, -0.96], noise) * np.exp(-6.9 * t / rt60) * 6
        ir = bright * 0.4 + dark
        ir[: int(0.02 * SR)] = 0
        for d, g in ((0.013, 0.5), (0.019, 0.35), (0.027, 0.3), (0.035, 0.2)):
            ir[int((d + 0.002 * ch) * SR)] += g * 8
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    return np.stack(irs, 1).astype(np.float32)


# ------------------------------------------------------------------ the score
CH = {  # bass notes, melody tones, pad voicing
    "Dm":   (["D2", "A2"], ["D4", "F4", "A4", "C5", "D5", "E5"], ["D3", "A3", "C4", "F4"]),
    "C":    (["C2", "G2"], ["C4", "E4", "G4", "C5", "D5", "E5"], ["C3", "G3", "C4", "E4"]),
    "Bb":   (["Bb1", "F2"], ["D4", "F4", "A4", "Bb4", "D5", "F5"], ["Bb2", "F3", "A3", "D4"]),
    "F":    (["F2", "C3"], ["C4", "F4", "A4", "C5", "E5", "F5"], ["F3", "A3", "C4", "E4"]),
    "G":    (["G2", "D3"], ["D4", "G4", "B4", "D5", "E5"], ["G2", "D3", "B3", "E4"]),
    "Am":   (["A2", "E3"], ["E4", "A4", "C5", "E5", "G5"], ["A2", "E3", "C4", "G4"]),
    "Gm":   (["G2", "D3"], ["D4", "G4", "Bb4", "D5", "F5"], ["G2", "D3", "Bb3", "F4"]),
    "Asus": (["A2", "E3"], ["D4", "E4", "A4", "D5", "E5"], ["A2", "E3", "A3", "D4"]),
}

PLAN = {  # section: (chords, density)
    "open": (["Dm", "Dm", "Bb", "C", "Dm", "Asus"], 0.16),
    "tens": (["F", "C", "Dm", "Asus"], 0.20),
    "clay": (["Dm", "Bb", "F", "C", "Dm", "Gm", "Asus", "Dm"], 0.26),
    "why60": (["F", "C", "Dm", "Bb", "F", "G", "Asus"], 0.24),
    "sky": (["Dm", "C", "Bb", "Am", "Gm", "Asus"], 0.22),
    "egypt": (["Am", "Dm", "Bb", "Dm", "F", "C", "G", "Dm"], 0.22),
    "greece": (["Dm", "F", "C", "G", "Dm", "Bb", "Asus"], 0.28),
    "clocks": (["F", "C", "Bb", "Asus"], 0.24),
    "decimal": (["Dm", "Bb", "F", "C", "Gm", "Asus", "Dm"], 0.26),
    "others": (["F", "Am", "Bb", "C"], 0.24),
    "ladder": (["Dm", "C", "Bb", "G", "Asus"], 0.22),
    "close": (["Dm", "Bb", "Dm"], 0.10),
}


def plan(O, total, order):
    rows = []
    for k, s in enumerate(order):
        t0 = O[s]
        t1 = O[order[k + 1]] if k + 1 < len(order) else total
        chords, dens = PLAN[s]
        step = (t1 - t0) / len(chords)
        for j, c in enumerate(chords):
            rows.append((round(t0 + j * step), c, dens))
    rows.append((total, None, 0))
    return rows


def notes(rows, total, seed=60):
    """lyre notes on the one-second beat grid: (time, note, velocity)"""
    rng = np.random.default_rng(seed)
    ev = []
    prev = midi("A4")
    for (t0, c, dens), (t1, _, _) in zip(rows[:-1], rows[1:]):
        lh, rh, _ = CH[c]
        if t0 > 0.5:
            ev.append((t0, lh[0], 0.36))
            if t1 - t0 > 3:
                ev.append((t0 + 2, lh[1], 0.24))
        if t0 > total - 14 and c == "Dm":                    # the last chord, rolled and left to ring
            for k, n_ in enumerate(["D2", "A2", "F3", "A3", "D4", "E4", "A4"]):
                ev.append((t0 + 0.25 * k, n_, 0.36 - 0.025 * k))
            continue
        g = max(t0 + 1, 2)
        while g < t1 - 0.5:
            if rng.random() < dens:
                cand = [midi(n) for n in rh]
                w = np.array([math.exp(-abs(m - prev) / 3.0) * (0.4 if m == prev else 1.0) for m in cand])
                m = int(rng.choice(cand, p=w / w.sum()))
                ev.append((g, rh[cand.index(m)], float(rng.uniform(0.32, 0.55))))
                prev = m
                if rng.random() < 0.3:                      # an off-beat grace: half a second later
                    m2 = int(rng.choice(cand))
                    ev.append((g + 0.5, rh[cand.index(m2)], float(rng.uniform(0.22, 0.36))))
                g += int(rng.choice([1, 2, 2, 3]))
            else:
                g += 1
    return ev


def render(O, total, order, loud=None):
    rows = plan(O, total, order)
    ev = notes(rows, total)
    n = int((total + 6.0) * SR)
    Lc = np.zeros(n, np.float32)
    Rc = np.zeros(n, np.float32)
    for k, (t, name, v) in enumerate(ev):
        x = lyre(hz(name), v, seed=k)
        i = int(t * SR)
        x = x[: n - i]
        pan = float(np.clip((midi(name) - 62) / 36, -0.35, 0.35))
        Lc[i:i + len(x)] += x * math.sqrt(0.5 - pan)
        Rc[i:i + len(x)] += x * math.sqrt(0.5 + pan)
    P = np.zeros(n, np.float32)
    for (t0, c, _), (t1, _, _) in zip(rows[:-1], rows[1:]):
        if c is None:
            continue
        for j, name in enumerate(CH[c][2]):
            x = pad_tone(hz(name), t1 - t0 + 1.5, seed=j + int(t0))
            i = int(t0 * SR)
            x = x[: n - i]
            P[i:i + len(x)] += x
    P *= 0.16 / (np.abs(P).max() + 1e-9)
    mus = np.stack([Lc + P * 0.95, Rc + P * 1.05], 1)
    mus /= np.abs(mus).max() + 1e-9
    # the clock: a tick every second, louder where a clock is on screen
    T = np.zeros(n, np.float32)
    tk = [tick(True, 1), tick(False, 2)]
    for s in range(1, int(total)):
        lv = loud(s) if loud else 0.5
        x = tk[s % 2] * lv
        i = s * SR
        T[i:i + len(x)] += x[: n - i]
    B = np.zeros(n, np.float32)
    for m in range(60, int(total), 60):                      # a bell at each full minute of the film
        x = bell(hz("A5"), seed=m)
        i = m * SR
        B[i:i + len(x)] += x[: n - i]
    dry = mus * 0.62 + np.stack([T * 0.09, T * 0.09], 1) + np.stack([B * 0.05, B * 0.05], 1)
    ir = reverb_ir()
    wet = np.stack([fftconvolve(dry[:, 0], ir[:, 0])[:n], fftconvolve(dry[:, 1], ir[:, 1])[:n]], 1)
    mix = dry * 0.8 + wet * 0.45
    t = np.arange(n) / SR
    mix *= (np.clip(t / 0.5, 0, 1) * np.clip((total + 0.3 - t) / 3.0, 0, 1))[:, None]
    mix = mix[: int(total * SR)]
    return (mix / (np.abs(mix).max() + 1e-9) * 0.7).astype(np.float32), ev


def write(mix, path):
    import wave
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import assemble as A
    O, total = A.timeline()
    mix, ev = render(O, total, A.SECS, A.tick_level(O))
    os.makedirs(OUT, exist_ok=True)
    write(mix, os.path.join(OUT, "score.wav"))
    print(f"{len(ev)} notes, {total:.1f} s")
