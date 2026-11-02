"""The score: a quiet stride. A felt piano over a soft pad in D major at 80 bpm (one bar = 3 s, so every event
card is exactly three bars), with a low "footfall" pulse; during the opening race the pulse quickens and
stops dead at the record's finish. Everything is synthesized (the felt-piano and pad models and the room
are the vernier film's, loaded from ../vernier/music.py). No samples.

    python music.py          # writes out/score.wav for the timeline in assemble.py
"""
import os, sys, math, importlib.util
import numpy as np
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
_spec = importlib.util.spec_from_file_location("vernier_music", os.path.join(HERE, "..", "vernier", "music.py"))
VM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VM)
SR = VM.SR
piano, pad_tone, reverb_ir, hz, midi, write = VM.piano, VM.pad_tone, VM.reverb_ir, VM.hz, VM.midi, VM.write

BAR = 3.0
BEAT = BAR / 4

CH = {  # left-hand root/fifth, right-hand arpeggio tones, pad voicing
    "D":    (["D2", "A2"], ["F#4", "A4", "D5", "E5", "F#5", "A5"], ["D3", "A3", "D4", "F#4"]),
    "Bm":   (["B1", "F#2"], ["F#4", "B4", "D5", "E5", "F#5"], ["B2", "F#3", "B3", "D4"]),
    "G":    (["G1", "D2"], ["G4", "B4", "D5", "E5", "F#5"], ["G2", "D3", "B3", "D4"]),
    "A":    (["A1", "E2"], ["E4", "A4", "C#5", "E5", "A5"], ["A2", "E3", "A3", "C#4"]),
    "Asus": (["A1", "E2"], ["E4", "A4", "D5", "E5", "A5"], ["A2", "E3", "A3", "D4"]),
    "Em":   (["E2", "B2"], ["E4", "G4", "B4", "D5", "E5"], ["E2", "B2", "G3", "D4"]),
    "F#m":  (["F#2", "C#3"], ["F#4", "A4", "C#5", "E5", "F#5"], ["F#2", "C#3", "A3", "E4"]),
    "D/F#": (["F#2", "D3"], ["F#4", "A4", "D5", "F#5", "A5"], ["F#2", "D3", "A3", "D4"]),
}
CARD_PROGS = [("D", "Bm", "G"), ("Em", "G", "Asus"), ("Bm", "G", "D"), ("G", "D/F#", "A"),
              ("D", "F#m", "G"), ("Bm", "Em", "A")]


def section_plan(O, total):
    """(start, chord, style) bars across the film, built from the sections' structure"""
    import film
    bars = []
    for name, t0 in O.items():
        sec = film.SECTIONS[name]()
        if name == "open":
            race = sec.parts[0]
            bars += [(t0, "Bm", "pad"), (t0 + 3.0, "G", "pad")]
            bars.append((t0 + race.t_go - 1.0, "race", (race.t_go - 1.0, race.t_freeze)))
            bars.append((t0 + race.t_freeze, "D/F#", "hold"))
            off = t0 + race.dur
            bars += [(off, "Em", "sparse"), (off + 3.0, "Asus", "sparse")]
            off += sec.parts[1].dur
            bars += [(off, "D", "title"), (off + 3.0, "D", "title")]
        elif name in ("how", "compare"):
            prog = ["D", "Bm", "G", "Asus"] if name == "how" else ["F#m", "G", "D", "A", "Bm", "G", "Em", "Asus"]
            n = int(sec.dur // BAR)
            bars += [(t0 + i * BAR, prog[i % len(prog)], "arp" if name == "how" else "half") for i in range(n)]
        elif name == "close":
            bars += [(t0, "G", "half"), (t0 + BAR * 2, "Asus", "half"), (t0 + BAR * 4, "D", "end")]
        else:                                                # a tour chapter
            k = 0
            for p in sec.parts:
                ps = t0 + sum(q.dur for q in sec.parts[:sec.parts.index(p)])
                if p.__class__.__name__ == "ChapterCard":
                    bars.append((ps, "Asus", "chapter"))
                else:
                    prog = CARD_PROGS[k % len(CARD_PROGS)]
                    k += 1
                    for i, c in enumerate(prog):
                        bars.append((ps + i * BAR, c, "arp"))
    return sorted(bars, key=lambda b: b[0])


def notes(bars, total, seed=7):
    rng = np.random.default_rng(seed)
    ev = []
    for (t0, c, style), nxt in zip(bars, bars[1:] + [(total, None, None)]):
        t1 = nxt[0]
        if c == "race":
            go_rel, freeze_rel = style
            go = t0 + 1.0
            fr = t0 - go_rel + freeze_rel
            # footfalls: quicken from a walk to a sprint, then stop dead at the finish
            t = go
            k = 0
            while t < fr - 0.05:
                ev.append((t, "D2" if k % 2 == 0 else "A1", 0.12, 0.30 + 0.18 * min(1, (t - go) / 4)))
                gap = 0.62 - 0.36 * min(1.0, (t - go) / 3.0)
                t += gap
                k += 1
            continue
        lh, rh, _ = CH[c]
        if style in ("pad", "sparse"):
            ev.append((t0 + 0.2, rh[rng.integers(1, len(rh))], 1.8, 0.28))
        elif style == "hold":
            for j, nm in enumerate(lh + rh[:3]):
                ev.append((t0 + 0.02 * j, nm, 2.6, 0.38))
        elif style == "title":
            ev.append((t0, lh[0], 2.8, 0.55))
            for j, nm in enumerate([rh[0], rh[2], rh[4]]):
                ev.append((t0 + 0.5 * j, nm, 1.6, 0.42))
        elif style == "chapter":
            ev.append((t0, lh[0], 2.8, 0.5))
            ev.append((t0 + 0.1, rh[-1], 2.2, 0.38))
        elif style == "arp":
            ev.append((t0, lh[0], BAR * 0.9, 0.42))                     # footfall on 1 and 3
            ev.append((t0 + 2 * BEAT, lh[1], BEAT * 1.6, 0.30))
            pat = [0, 2, 1, 3] if rng.random() < 0.5 else [1, 2, 3, 2]
            for i, j in enumerate(pat):
                if rng.random() < 0.12 and i:
                    continue
                ev.append((t0 + i * BEAT, rh[min(j, len(rh) - 1)], BEAT * 1.4, 0.26 + 0.06 * (i == 0)))
        elif style == "half":
            ev.append((t0, lh[0], BAR * 0.95, 0.40))
            ev.append((t0 + BEAT * 0.02, rh[1], BAR * 0.5, 0.26))
            ev.append((t0 + 2 * BEAT, rh[3 if len(rh) > 3 else -1], BAR * 0.45, 0.22))
        elif style == "end":
            ev.append((t0, lh[0], 5.0, 0.45))
            for j, nm in enumerate([rh[0], rh[2], rh[3]]):
                ev.append((t0 + 0.6 * j, nm, 4.0, 0.30))
    return ev


def render(O, total):
    bars = section_plan(O, total)
    ev = notes(bars, total)
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
    # the pad follows the chords (not under the race, where the footfalls carry it)
    P = np.zeros(n, np.float32)
    for (t0, c, style), nxt in zip(bars, bars[1:] + [(total, None, None)]):
        if c == "race" or style == "chapter":
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
    mix = render(O, total)
    write(mix, os.path.join(OUT, "score.wav"))
    print(f"{total:.1f} s")
