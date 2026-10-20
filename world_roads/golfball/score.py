"""The score: the vernier film's felt piano and pad (../vernier/music.py: additive piano partials, a filtered
pad, a synthetic room; no samples), with chord changes laid on this film's sections.

    python score.py            # writes out/score.wav for the timeline in assemble.py
"""
import os, sys, math, importlib.util
import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("vernier_music", os.path.join(os.path.dirname(HERE), "vernier", "music.py"))
V = importlib.util.module_from_spec(_spec)               # the vernier film's instruments
_spec.loader.exec_module(V)

OUT = os.path.join(HERE, "out")
SR = V.SR
CH = V.CH


def plan(O, total):
    """(start, chord, density) for each chord, in video time, from the section offsets O"""
    rows = []

    def seq(t0, t1, chords, dens):
        step = (t1 - t0) / len(chords)
        for k, c in enumerate(chords):
            rows.append((t0 + k * step, c, dens))

    seq(0.0, O["open"] + 15.0, ["F", "Bb"], 0.14)
    seq(O["open"] + 15.0, O["angle"], ["Csus"], 0.06)                   # "you shouldn't be able to see it"
    seq(O["angle"], O["chart"], ["F", "Am", "Bb", "C"], 0.22)
    seq(O["chart"], O["eye"], ["Dm", "Bb", "F", "Csus"], 0.22)
    seq(O["eye"], O["retina"], ["F/A"], 0.12)
    seq(O["retina"], O["retina"] + 47.0, ["Dm", "Bb", "F/A", "Gm", "Am", "Bb"], 0.24)
    seq(O["retina"] + 47.0, O["blur"], ["Csus"], 0.06)                  # "you'd never see it"
    seq(O["blur"], O["blur"] + 33.0, ["C", "F", "Bb", "F"], 0.30)       # the blur, and one cone's patch
    seq(O["blur"] + 33.0, O["blur"] + 66.0, ["Dm", "Bb", "F", "Gm", "C"], 0.36)
    seq(O["blur"] + 66.0, O["recipe"], ["F"], 0.16)                     # "it gets fainter"
    seq(O["recipe"], O["dark"], ["Am", "Bb", "F", "C", "Dm", "Bb", "Csus"], 0.28)
    seq(O["dark"], O["dark"] + 31.0, ["Dm", "Gm", "Bb", "C"], 0.26)
    seq(O["dark"] + 31.0, O["demo"], ["F", "Am", "Bb", "Gm", "Csus"], 0.30)
    seq(O["demo"], O["close"], ["F", "Bb", "Am", "Csus"], 0.10)          # sparse: look at the cards
    seq(O["close"], O["close"] + 8.0, ["F"], 0.18)
    seq(O["close"] + 8.0, O["close"] + 12.0, ["Bb"], 0.15)
    seq(O["close"] + 12.0, total, ["F"], 0.0)
    rows.append((total, None, 0))
    return rows


def render(O, total):
    V.plan = plan                                       # the vernier generator, on this film's chords
    ev, rows = V.notes(O, total, seed=23)
    n = int((total + 4.0) * SR)
    Lc = np.zeros(n, np.float32)
    Rc = np.zeros(n, np.float32)
    for k, (t, name, d, v) in enumerate(ev):
        x = V.piano(V.hz(name), d, v, seed=k)
        i = int(t * SR)
        x = x[: n - i]
        pan = np.clip((V.midi(name) - 60) / 40, -0.35, 0.35)
        Lc[i:i + len(x)] += x * math.sqrt(0.5 - pan)
        Rc[i:i + len(x)] += x * math.sqrt(0.5 + pan)
    # the pad: in from the retina onwards, out for the opening and the end
    Pd = np.zeros(n, np.float32)
    for (t0, c, _), (t1, _, _) in zip(rows[:-1], rows[1:]):
        if c is None or t0 < O["retina"] - 0.1 or t0 >= O["close"] + 12.0:
            continue
        for j, name in enumerate(CH[c][2]):
            x = V.pad_tone(V.hz(name), t1 - t0 + 1.5, seed=j + int(t0))
            i = int(t0 * SR)
            x = x[: n - i]
            Pd[i:i + len(x)] += x
    Pd *= 0.10 / (np.abs(Pd).max() + 1e-9)
    dry = np.stack([Lc + Pd * 0.95, Rc + Pd * 1.05], 1)
    ir = V.reverb_ir()
    wet = np.stack([fftconvolve(dry[:, 0], ir[:, 0])[:n], fftconvolve(dry[:, 1], ir[:, 1])[:n]], 1)
    mix = dry * 0.78 + wet * 0.55
    t = np.arange(n) / SR
    mix *= np.clip(t / 0.4, 0, 1)[:, None] * np.clip((total + 0.5 - t) / 3.5, 0, 1)[:, None]
    mix = mix[: int(total * SR)]
    return (mix / (np.abs(mix).max() + 1e-9) * 0.7).astype(np.float32), ev


write = V.write

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import assemble as A
    O, total = A.timeline()
    mix, ev = render(O, total)
    os.makedirs(OUT, exist_ok=True)
    write(mix, os.path.join(OUT, "score.wav"))
    print(f"{len(ev)} notes, {total:.1f} s")
