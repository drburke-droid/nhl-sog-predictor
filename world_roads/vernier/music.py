"""The score: a sparse felt piano over a soft pad, in F major, in a small room. Everything is synthesized
(additive piano partials with string inharmonicity, felt damping and hammer noise; a filtered pad; a
synthetic reverb). No samples.

    python music.py            # writes out/score.wav for the timeline in assemble.py
"""
import os, sys, math, json
import numpy as np
from scipy.signal import fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SR = 48000

NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(note):
    """'A4', 'Bb2', 'F#3' -> Hz"""
    n, acc, octv = note[0], note[1:-1], int(note[-1])
    k = NAMES[n] + {"": 0, "b": -1, "#": 1}[acc] + 12 * (octv + 1)
    return 440.0 * 2 ** ((k - 69) / 12)


def midi(note):
    return round(69 + 12 * math.log2(hz(note) / 440.0))


# ------------------------------------------------------------------ instruments
def piano(f0, dur, vel, seed=0):
    """one felt-piano note: dur (s) until the damper falls, vel 0..1. Mono float32."""
    rng = np.random.default_rng(seed)
    ring = 3.5 if dur > 0.8 else 2.0
    n = int((dur + ring) * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    B = 0.00028 * (f0 / 261.6) ** 0.6                    # string stiffness: partials run sharp
    felt = 0.62 - 0.30 * vel                             # softer notes lose their upper partials faster
    rel = np.ones(n, np.float32)
    k_off = int(dur * SR)
    if k_off < n:
        rel[k_off:] = np.exp(-(t[k_off:] - dur) / 0.22)
    for k in range(1, 18):
        fk = k * f0 * math.sqrt(1 + B * k * k)
        if fk > 7000:
            break
        amp = k ** -1.15 * math.exp(-(k - 1) * felt)
        if k in (8, 9):                                  # the hammer strikes about 1/8 along: those partials are weak
            amp *= 0.35
        tau1 = 0.55 * (261.6 / f0) ** 0.5 / (1 + 0.10 * k)
        tau2 = 4.0 * (261.6 / f0) ** 0.65 / (1 + 0.06 * k)
        env = (0.62 * np.exp(-t / tau1) + 0.38 * np.exp(-t / tau2)).astype(np.float32)
        d = 0.06 + 0.02 * k                              # two strings, slightly apart: slow beating
        ph1, ph2 = rng.uniform(0, 2 * math.pi, 2)
        out += amp * env * 0.5 * (np.sin(2 * math.pi * fk * t + ph1) + np.sin(2 * math.pi * (fk + d) * t + ph2))
    out *= (1 - np.exp(-t / 0.0045)).astype(np.float32)  # felt: a soft attack
    out *= rel
    # the hammer and the action: a short, dull knock
    m = int(0.03 * SR)
    knock = rng.normal(0, 1, m).astype(np.float32) * np.exp(-np.arange(m) / (0.006 * SR)).astype(np.float32)
    knock = lfilter([0.08], [1, -0.92], knock).astype(np.float32)
    out[:m] += knock * 0.10
    # felt piano: muffled
    out = lfilter([0.35], [1, -0.65], out).astype(np.float32)
    return out * (0.25 + 0.75 * vel) ** 1.4


def pad_tone(f0, dur, fade_in=3.5, fade_out=3.5, seed=0):
    """a soft, slowly breathing pad tone: detuned, band-limited, lowpassed"""
    rng = np.random.default_rng(seed)
    n = int((dur + fade_out) * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, np.float32)
    for det in (-0.12, 0.0, 0.13):
        f = f0 * 2 ** (det / 12)
        for k in range(1, 7):
            out += (0.6 ** (k - 1) / k) * np.sin(2 * math.pi * f * k * t + rng.uniform(0, 6.28)).astype(np.float32)
    lfo = 1 + 0.12 * np.sin(2 * math.pi * 0.11 * t + rng.uniform(0, 6.28))
    env = np.clip(t / fade_in, 0, 1) ** 2 * np.clip((dur + fade_out - t) / fade_out, 0, 1) ** 1.5
    out = lfilter([0.06], [1, -0.94], out * env * lfo).astype(np.float32)
    return out


# ------------------------------------------------------------------ the room
def reverb_ir(rt60=2.4, length=3.6, seed=4):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    irs = []
    for ch in range(2):
        noise = rng.normal(0, 1, n)
        bright = noise * np.exp(-6.9 * t / (rt60 * 0.55))
        dark = lfilter([0.04], [1, -0.96], noise) * np.exp(-6.9 * t / rt60) * 6
        ir = bright * 0.45 + dark
        ir[: int(0.018 * SR)] = 0                        # pre-delay
        for d, g in ((0.011, 0.5), (0.017, 0.35), (0.023, 0.3), (0.031, 0.22)):
            ir[int((d + 0.002 * ch) * SR)] += g * 8
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    return np.stack(irs, 1).astype(np.float32)


# ------------------------------------------------------------------ the score
CH = {  # chord: left-hand root and fifth, right-hand chord tones (for melody), pad voicing
    "F":    (["F2", "C3"], ["A4", "C5", "E5", "F5", "G5", "A5"], ["F3", "A3", "C4", "E4"]),
    "Am":   (["A2", "E3"], ["G4", "A4", "C5", "E5", "G5"], ["A3", "C4", "E4", "G4"]),
    "Bb":   (["Bb1", "F2"], ["A4", "D5", "F5", "A5", "C5"], ["Bb2", "F3", "A3", "D4"]),
    "C":    (["C3", "G3"], ["G4", "C5", "D5", "E5", "G5"], ["C3", "G3", "D4", "E4"]),
    "Csus": (["C3", "G3"], ["G4", "C5", "D5", "F5", "G5"], ["C3", "G3", "C4", "F4"]),
    "Dm":   (["D2", "A2"], ["A4", "C5", "E5", "F5", "A5"], ["D3", "A3", "C4", "F4"]),
    "Gm":   (["G2", "D3"], ["Bb4", "D5", "F5", "G5", "A5"], ["G2", "D3", "Bb3", "F4"]),
    "F/A":  (["A2", "F3"], ["A4", "C5", "F5", "G5", "A5"], ["A2", "F3", "C4", "E4"]),
}


def plan(O, total):
    """(start, chord, density) for each chord, in video time, from the section offsets O"""
    rows = []

    def seq(t0, t1, chords, dens):
        step = (t1 - t0) / len(chords)
        for k, c in enumerate(chords):
            rows.append((t0 + k * step, c, dens))

    seq(0.0, O["eye"] + 10.0, ["F", "Bb"], 0.16)
    seq(O["retina"], O["retina"] + 49.0, ["Dm", "Bb", "F/A", "Gm", "Am", "Bb"], 0.24)
    seq(O["retina"] + 49.0, O["back"] + 6.0, ["Csus"], 0.06)            # "you shouldn't be able to see it"
    seq(O["back"] + 6.0, O["back"] + 14.0, ["C"], 0.14)
    seq(O["back"] + 14.0, O["smaller"], ["F"], 0.12)                     # "you can see it, can't you?"
    seq(O["smaller"], O["how"] - 0.5, ["Am", "Bb", "F", "C", "Am", "Bb", "Dm", "Csus"], 0.26)
    seq(O["how"], O["how"] + 49.0, ["Dm", "Bb", "F", "C", "Dm", "Gm"], 0.40)
    seq(O["how"] + 49.0, O["how"] + 62.0, ["Am", "Bb", "Csus"], 0.46)
    seq(O["how"] + 62.0, O["screen"], ["F"], 0.20)                       # "each cone is coarse..."
    seq(O["screen"], O["numbers"], ["Bb", "F/A", "Gm", "C"], 0.30)
    seq(O["numbers"], O["uses"], ["Am", "Dm", "Bb", "Csus"], 0.26)
    seq(O["uses"], O["close"], ["F", "Am", "Bb", "Gm", "Dm", "Bb", "Csus"], 0.28)
    seq(O["close"], O["close"] + 8.0, ["F"], 0.18)
    seq(O["close"] + 8.0, O["close"] + 12.0, ["Bb"], 0.15)
    seq(O["close"] + 12.0, total, ["F"], 0.0)
    rows.append((total, None, 0))
    return rows


def notes(O, total, seed=11):
    """the piano part: (time, note, duration, velocity)"""
    rng = np.random.default_rng(seed)
    rows = plan(O, total)
    ev = []
    # the opening motif, alone
    for t_, n_, d_, v_ in ((0.8, "A4", 2.4, 0.42), (3.0, "C5", 2.0, 0.38), (5.0, "F5", 3.0, 0.45), (8.6, "E5", 3.4, 0.36)):
        ev.append((t_, n_, d_, v_))
    prev = midi("F5")
    for (t0, c, dens), (t1, _, _) in zip(rows[:-1], rows[1:]):
        lh, rh, _ = CH[c]
        span = t1 - t0
        if t0 > 0.5:
            ev.append((t0, lh[0], min(span, 6.0), 0.32))
            if span > 3.0:
                ev.append((t0 + min(1.5, span / 3), lh[1], min(span - 1.5, 5.0), 0.22))
        if c == "F" and t0 > total - 12:                                  # the last chord, rolled and left to ring
            for k, n_ in enumerate(["F2", "C3", "A3", "E4", "G4", "C5"]):
                ev.append((t0 + 0.12 * k, n_, total - t0, 0.38 - 0.03 * k))
            continue
        g = t0 + 1.0 if t0 > 0.5 else 11.5
        while g < t1 - 0.6:
            if rng.random() < dens:
                cand = [midi(n) for n in rh]
                w = np.array([math.exp(-abs(m - prev) / 3.0) * (0.4 if m == prev else 1.0) for m in cand])
                m = int(rng.choice(cand, p=w / w.sum()))
                name = rh[cand.index(m)]
                dur_ = float(rng.choice([1.0, 1.5, 2.0, 3.0]))
                ev.append((g + rng.normal(0, 0.012), name, dur_, float(rng.uniform(0.28, 0.48))))
                prev = m
                g += float(rng.choice([1.0, 1.5, 2.0]))
            else:
                g += 0.5
    return ev, rows


def render(O, total):
    ev, rows = notes(O, total)
    n = int((total + 4.0) * SR)
    L = np.zeros(n, np.float32)
    Rr = np.zeros(n, np.float32)
    for k, (t, name, d, v) in enumerate(ev):
        f = hz(name)
        x = piano(f, d, v, seed=k)
        i = int(t * SR)
        x = x[: n - i]
        pan = np.clip((midi(name) - 60) / 40, -0.35, 0.35)
        L[i:i + len(x)] += x * math.sqrt(0.5 - pan)
        Rr[i:i + len(x)] += x * math.sqrt(0.5 + pan)
    # the pad: in from the retina onwards, out for the opening and the end
    P = np.zeros(n, np.float32)
    for (t0, c, _), (t1, _, _) in zip(rows[:-1], rows[1:]):
        if c is None or t0 < O["retina"] - 0.1:
            continue
        for j, name in enumerate(CH[c][2]):
            x = pad_tone(hz(name), t1 - t0 + 1.5, seed=j + int(t0))
            i = int(t0 * SR)
            x = x[: n - i]
            P[i:i + len(x)] += x
    P *= 0.10 / (np.abs(P).max() + 1e-9)
    dry = np.stack([L + P * 0.95, Rr + P * 1.05], 1)
    ir = reverb_ir()
    wet = np.stack([fftconvolve(dry[:, 0], ir[:, 0])[:n], fftconvolve(dry[:, 1], ir[:, 1])[:n]], 1)
    mix = dry * 0.78 + wet * 0.55
    # fade in from silence, out with the picture
    t = np.arange(n) / SR
    mix *= np.clip(t / 0.4, 0, 1)[:, None] * np.clip((total + 0.5 - t) / 3.5, 0, 1)[:, None]
    mix = mix[: int(total * SR)]
    return (mix / (np.abs(mix).max() + 1e-9) * 0.7).astype(np.float32), ev


def write(mix, path):
    import wave
    x = np.clip(mix, -1, 1)
    pcm = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import assemble as A
    O, total = A.timeline()
    mix, ev = render(O, total)
    write(mix, os.path.join(OUT, "score.wav"))
    print(f"{len(ev)} notes, {total:.1f} s")
