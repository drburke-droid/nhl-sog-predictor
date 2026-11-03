"""The score for "The 10 Biggest Statistical Outliers in Sports History".

Same instrument and room as the records film (felt piano, soft pad, synthesized, no samples; loaded from
../music.py, which loads ../vernier/music.py). D major, one bar = 3 s, so every 36 s countdown card is twelve bars:
a single low "chapter" chord under the rank number, then a walking arpeggio.

    python score.py          # writes out/score.wav for the timeline in assemble.py
"""
import os, sys, math, importlib.util
import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
_spec = importlib.util.spec_from_file_location("records_music", os.path.join(HERE, "..", "music.py"))
RM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RM)
SR, BAR, CH = RM.SR, RM.BAR, RM.CH
piano, pad_tone, reverb_ir, hz, midi, write = RM.piano, RM.pad_tone, RM.reverb_ir, RM.hz, RM.midi, RM.write

CARD_PROGS = [("D", "Bm", "G", "Asus"), ("Bm", "G", "D", "A"), ("G", "D/F#", "Em", "Asus"),
              ("D", "F#m", "G", "A"), ("Em", "G", "D", "Asus")]


def cycle(t0, n, prog, style):
    return [(t0 + i * BAR, prog[i % len(prog)], style) for i in range(n)]


def section_plan(O, total):
    """(start, chord, style) for every bar of the film, built from the section start times"""
    bars = []
    for name, t0 in O.items():
        if name == "cold":            # four numbers, the question, then the title
            bars += [(t0, "Bm", "pad"), (t0 + 3, "G", "pad"), (t0 + 6, "Em", "sparse"), (t0 + 9, "Asus", "sparse"),
                     (t0 + 12, "D", "title"), (t0 + 15, "D", "title"), (t0 + 18, "Bm", "half"), (t0 + 21, "Asus", "half")]
        elif name == "method":        # the Ledecky walk-through (0-51 s), then the three rules
            bars += cycle(t0, 17, ["D", "Bm", "G", "Asus"], "arp")
            bars += cycle(t0 + 51, 9, ["G", "D/F#", "Em", "Asus"], "half")
        elif name[0] == "c" and name[1:].isdigit():   # a countdown card
            r = int(name[1:])
            bars.append((t0, "Asus" if r > 1 else "D", "chapter" if r > 1 else "title"))
            bars += cycle(t0 + BAR, 11, CARD_PROGS[(10 - r) % len(CARD_PROGS)], "arp")
        elif name == "throws":
            bars += cycle(t0, 19, ["Em", "Bm", "G", "A"], "half")
        elif name == "close":
            bars += cycle(t0, 13, ["G", "D/F#", "Em", "Asus", "Bm", "G", "D", "A"], "half")
            bars += [(t0 + 41, "D", "title"), (t0 + 44, "G", "title"), (t0 + 47, "Bm", "half"), (t0 + 50, "G", "half"),
                     (t0 + 53, "Asus", "half")]
        elif name == "end":
            bars += [(t0, "D", "end"), (t0 + 6, "G", "pad"), (t0 + 9, "D", "end")]
    return sorted(bars, key=lambda b: b[0])


def render(O, total):
    bars = section_plan(O, total)
    ev = RM.notes(bars, total)
    n = int((total + 5.0) * SR)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    for k, (t, name, d, v) in enumerate(ev):
        x = piano(hz(name), d, v, seed=k)
        i = int(t * SR)
        x = x[: max(0, n - i)]
        pan = np.clip((midi(name) - 60) / 40, -0.35, 0.35)
        L[i:i + len(x)] += x * math.sqrt(0.5 - pan)
        R[i:i + len(x)] += x * math.sqrt(0.5 + pan)
    P = np.zeros(n, np.float32)
    for (t0, c, style), nxt in zip(bars, bars[1:] + [(total, None, None)]):
        if style == "chapter":
            continue
        for j, nm in enumerate(CH[c][2]):
            x = pad_tone(hz(nm), nxt[0] - t0 + 1.0, fade_in=1.2, fade_out=1.5, seed=j + int(t0 * 10))
            i = int(t0 * SR)
            x = x[: max(0, n - i)]
            P[i:i + len(x)] += x
    P *= 0.12 / (np.abs(P).max() + 1e-9)
    dry = np.stack([L + P * 0.95, R + P * 1.05], 1)
    ir = reverb_ir(rt60=2.0)
    wet = np.stack([fftconvolve(dry[:, 0], ir[:, 0])[:n], fftconvolve(dry[:, 1], ir[:, 1])[:n]], 1)
    mix = dry * 0.8 + wet * 0.5
    t = np.arange(n) / SR
    mix *= np.clip(t / 0.5, 0, 1)[:, None] * np.clip((total + 0.3 - t) / 4.0, 0, 1)[:, None]
    mix = mix[: int(total * SR)]
    return (mix / (np.abs(mix).max() + 1e-9) * 0.7).astype(np.float32)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import assemble
    O, total = assemble.timeline()
    os.makedirs(OUT, exist_ok=True)
    write(render(O, total), os.path.join(OUT, "score.wav"))
    print(f"{total:.1f} s")
