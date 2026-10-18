"""The rivers video's soundtrack, synthesized from scratch (no samples, no licensing issues).

The theme is pi: its digits 3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 ... played as steps up a D minor pentatonic
scale (0 = D, 1 = F, 2 = G, 3 = A, 4 = C, 5 = the D above, ...), on a soft mallet over Dm - Bb - F - C.
Pads, a flowing arpeggio, bass and drums come and go section by section, each section restarting the bar
grid so every change lands on a downbeat. Instruments not defined here come from flights/score.py.
"""
import math
import numpy as np
from scipy.signal import butter, sosfilt, oaconvolve
import score as SC

SR = SC.SR
BPM = 92.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(11)

POP = [(50, True), (46, False), (53, False), (48, False)]           # Dm Bb F  C   (roots, octave 3)
SUS = [(50, True), (46, False), (43, True), (45, False)]           # Dm Bb Gm A
PI_DIGITS = "31415926535897932384626433832795028841971693993751"
SCALE = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84]                  # D minor pentatonic, D4 up two octaves


def pi_phrase(start, beats=16):
    """sixteen digits of pi as quarter notes, the last one held"""
    out = []
    for k in range(beats - 2):
        out.append((k, SCALE[int(PI_DIGITS[start + k])], 1.0))
    out.append((beats - 2, SCALE[int(PI_DIGITS[start + beats - 2])], 2.0))
    return out


THEME_A = pi_phrase(0)
THEME_B = pi_phrase(16)

# section loudness (dB), so breakdowns pull back and drops hit; effects, impacts and the ending are exempt
STYLE_DB = {"intro": -4.0, "light": -2.0, "tension": -6.0, "full": 0.0, "full2": 0.5, "suspense": -7.0, "triumph": 0.5,
            "mellow": -3.5, "flow": -1.5}

STYLES = {
    #            pad  bass      drums      lead  arp    bright
    "intro":    (0.9, None,     None,      0.8,  None,  0.45),
    "light":    (0.7, "half",   "light",   0.9,  None,  0.6),
    "tension":  (0.9, "pedal",  "tick",    0.0,  "8",   0.35),
    "full":     (0.55, "oct",   "full",    1.0,  "16",  0.95),
    "full2":    (0.55, "oct",   "full",    1.0,  "16",  1.0),
    "suspense": (1.0, "pedal",  "heart",   0.0,  "4",   0.3),
    "triumph":  (0.9, "oct",    "light",   1.0,  "8",   0.85),
    "mellow":   (0.8, "half",   "light",   0.6,  None,  0.55),
    "flow":     (0.75, "half",  "tick",    0.85, "8",   0.6),
}


class Track(SC.Score):
    def __init__(self, total):
        super().__init__(total, [(0, 0.5), (total, 0.5)], [], [], intro_hit=False)
        for k in ("lead", "perc", "sfx", "crash"):
            self.stems[k] = np.zeros((2, self.N), np.float32)
        self.bright = [(0.0, 0.5)]
        self.level = [(0.0, 0.0)]
        self.cuts = []                    # (tape stop start, stop length, silent until)

    # ------------------------------------------------------------------ instruments
    def lead(self, t0, m, dur, vel=1.0, pan=0.08):
        """a soft mallet: a sine with a bright, quickly fading overtone and a little wood in the attack"""
        n = int((dur + 0.6) * SR)
        t = np.arange(n) / SR
        f = SC.mtof(m)
        sig = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 4.0 * f * t) * np.exp(-t / 0.05)
               + 0.18 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.3))
        env = np.minimum(1, t / 0.004) * np.exp(-t / (0.45 + 0.25 * dur))
        env *= np.where(t > dur, np.exp(-(t - dur) / 0.12), 1.0)
        self.add("lead", t0 * SR, sig * env * vel, pan=pan)

    def snare(self, t0, vel=1.0):
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        sos = butter(2, [1200, 6000], "bp", fs=SR, output="sos")
        sig = sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-t / 0.11) + 0.6 * np.sin(2 * np.pi * 186 * t) * np.exp(-t / 0.05)
        self.add("clap", t0 * SR, sig * vel, pan=-0.05)

    def wood(self, t0, vel=1.0, pan=0.25):
        n = int(0.12 * SR)
        t = np.arange(n) / SR
        sig = (np.sin(2 * np.pi * 1250 * t) + 0.45 * np.sin(2 * np.pi * 2410 * t)) * np.exp(-t / 0.028)
        self.add("perc", t0 * SR, sig * vel, pan=pan)

    def crash(self, t0, vel=1.0):
        n = int(3.0 * SR)
        t = np.arange(n) / SR
        sos = butter(2, 4500, "hp", fs=SR, output="sos")
        sig = sosfilt(sos, rng.normal(0, 1, (2, n)), axis=1) * np.exp(-t / 0.9) * np.minimum(1, t / 0.002)
        self.add("crash", t0 * SR, sig * vel)

    def bell(self, t0, m=81, vel=1.0):
        n = int(2.2 * SR)
        t = np.arange(n) / SR
        f = SC.mtof(m)
        sig = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.25)
               + 0.25 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.08)) * np.exp(-t / 0.7) * np.minimum(1, t / 0.002)
        self.add("sfx", t0 * SR, sig * vel)

    def section(self, t0, t1, style, key=0, prog=POP, theme_bars=0):
        """fill [t0, t1) in a style; the bar grid starts at t0"""
        pad, bass, drums, lead_v, arp, bright = STYLES[style]
        self.bright.append((t0, bright))
        self.level.append((t0, STYLE_DB[style]))
        nb = int(math.ceil((t1 - t0) / BAR - 1e-9))
        for b in range(nb):
            tb = t0 + b * BAR
            root, minor = prog[b % 4]
            root += key
            third = 3 if minor else 4
            chord = [root, root + third, root + 7]
            left = t1 - tb
            if pad and b % 1 == 0:
                d = min(BAR, left)
                self.pad_chord(tb, d, [root - 12, root, root + third, root + 7, root + 12], rel=0.9 if d > 1 else 0.3,
                               att=0.25, gain=pad)
            # bass
            bm = root - 12
            if bass == "half":
                for k in (0, 2):
                    if k * BEAT < left:
                        self.bass_note(tb + k * BEAT, min(2 * BEAT * 0.95, left - k * BEAT), bm, True)
            elif bass == "pedal":
                self.bass_note(tb, min(BAR * 0.98, left), (prog[0][0] + key) - 12, False)
            elif bass == "oct":
                for k, (o, ln) in enumerate(((0, .5), (12, .5), (0, .5), (0, .5), (12, .5), (0, .5), (7, .5), (12, .5))):
                    tt = k * BEAT / 2
                    if tt < left:
                        self.bass_note(tb + tt, min(ln * BEAT * 0.85, left - tt), bm + o, True)
            # drums
            for k in range(4):
                tt = k * BEAT
                if tt >= left:
                    break
                t = tb + tt
                if drums == "wood":
                    if k in (1, 3):
                        self.wood(t, 0.7)
                    if k == 0:
                        self.kick(t, 0.55)
                    self.hat(t + BEAT / 2, 0.25)
                elif drums == "light":
                    if k in (0, 2):
                        self.kick(t, 0.8)
                    if k in (1, 3):
                        self.snare(t, 0.55)
                    self.hat(t + BEAT / 2, 0.45)
                    self.hat(t, 0.25)
                elif drums == "tick":
                    self.hat(t, 0.35)
                    self.hat(t + BEAT / 2, 0.22)
                elif drums == "heart":
                    if k == 0:
                        self.kick(t, 0.75)
                        self.kick(t + 0.28, 0.5)
                    self.wood(t + BEAT / 2, 0.18, pan=-0.3)
                elif drums == "full":
                    self.kick(t, 1.0)
                    if k in (1, 3):
                        self.clap(t, 0.9)
                        self.snare(t, 0.35)
                    for q in range(4):
                        if tt + q * BEAT / 4 < left:
                            self.hat(t + q * BEAT / 4, (0.55 if q == 2 else 0.3) + 0.08 * rng.random(), open_=(k == 3 and q == 2))
            # arpeggio
            if arp:
                tones = [root + 12, root + 12 + third, root + 19, root + 24, root + 24 + third, root + 31]
                step = {"16": 0.25, "8": 0.5, "4": 1.0}[arp]
                pat = [0, 2, 4, 1, 3, 5, 2, 4, 0, 3, 5, 2, 4, 1, 3, 5]
                for q in range(int(4 / step)):
                    tt = q * step * BEAT
                    if tt < left:
                        self.arp_note(tb + tt, tones[pat[q % 16]], 0.55 + 0.4 * (q % 4 == 0), bright=min(1.0, bright + 0.1))
        # the theme on the lead (phrase A, then B; repeats)
        if lead_v > 0:
            n_phr = int(math.ceil((t1 - t0) / (4 * BAR)))
            for p in range(n_phr):
                ph = THEME_A if p % 2 == 0 else THEME_B
                for beat, m, ln in ph:
                    t = t0 + p * 4 * BAR + beat * BEAT
                    if t + 0.05 < t1:
                        self.lead(t, m + key, min(ln * BEAT * 0.9, t1 - t), lead_v)
                        if style == "full2":
                            self.lead(t, m + key + 12, min(ln * BEAT * 0.9, t1 - t), lead_v * 0.35, pan=-0.25)

    def tape_stop(self, t0, length, resume):
        self.cuts.append((t0, length, resume))

    # ------------------------------------------------------------------ mix
    def mixdown(self):
        S = self.stems
        t = np.arange(self.N) / SR
        bp = np.array(sorted(self.bright))
        br = np.interp(t, bp[:, 0], bp[:, 1])
        # steps between sections, smoothed over ~0.2 s
        k = int(0.2 * SR)
        br = np.convolve(br, np.ones(k) / k, mode="same")
        for c in range(2):
            S["pad"][c] = SC.svf_lp(S["pad"][c], 500 + 3000 * br, 0.8) * self.duck
            S["bass"][c] = SC.svf_lp(S["bass"][c], np.full(self.N, 520.0), 0.8) * (0.6 + 0.4 * self.duck)
            S["arp"][c] = SC.svf_lp(S["arp"][c], 1500 + 4500 * br, 0.9)
            S["lead"][c] = SC.svf_lp(S["lead"][c], 1800 + 3400 * br, 0.75)
        d = int(0.75 * BEAT * SR)                                # dotted-eighth echo on the lead and arp
        for name, fb in (("lead", 0.28), ("arp", 0.36)):
            irl = np.zeros(6 * d + 1); irr = np.zeros(6 * d + 1)
            irl[0] = irr[0] = 1.0
            for j in range(1, 6):
                (irl if j % 2 else irr)[j * d] = fb ** j
            m = S[name].mean(0)
            S[name] = np.stack([oaconvolve(m, irl)[:self.N], oaconvolve(m, irr)[:self.N]])
        target = {"pad": -25, "bass": -19, "arp": -29, "kick": -18, "hat": -31, "clap": -25, "fx": -21, "final": -19,
                  "lead": -20, "perc": -27, "sfx": -21, "crash": -27}
        sends = {"pad": 0.3, "arp": 0.4, "clap": 0.4, "fx": 0.5, "final": 0.6, "hat": 0.12, "lead": 0.22, "sfx": 0.25,
                 "crash": 0.3, "perc": 0.2}
        lp = np.array(sorted(self.level))
        steps = lp[np.searchsorted(lp[:, 0], t, side="right") - 1, 1]
        k = int(0.08 * SR)
        auto = 10 ** (np.convolve(steps, np.ones(k) / k, mode="same") / 20)
        mixed = np.zeros((2, self.N)); send = np.zeros((2, self.N))
        for name, x in S.items():
            act = np.abs(x).max(0) > 1e-4
            if not act.any():
                continue
            g = 10 ** (target[name] / 20) / max(np.sqrt(np.mean(x[:, act] ** 2)), 1e-9)
            gx = x * g if name in ("fx", "final", "sfx", "crash") else x * g * auto
            mixed += gx
            send += gx * sends.get(name, 0)
        n_ir = int(2.6 * SR)
        ti = np.arange(n_ir) / SR
        sos = butter(1, 3200, "lp", fs=SR, output="sos")
        ir = []
        for c in range(2):
            h = rng.normal(0, 1, n_ir) * np.exp(-ti / 0.15) * 0.5 + sosfilt(sos, rng.normal(0, 1, n_ir)) * np.exp(-ti / 0.5) * 1.5
            h = np.r_[np.zeros(int(0.02 * SR)), h][:n_ir]
            ir.append(h / np.sqrt(np.sum(h ** 2)))
        out = mixed + np.stack([oaconvolve(send[c], ir[c])[:self.N] for c in range(2)]) * 0.5
        out = sosfilt(butter(2, 28, "hp", fs=SR, output="sos"), out, axis=1)
        out = np.tanh(out * 1.5) / 1.5
        # tape stops: the music winds down like a record losing power, then silence until it restarts
        for t0, ln, resume in self.cuts:
            i0, i1, i2 = int(t0 * SR), int((t0 + ln) * SR), int(resume * SR)
            u = np.arange(i1 - i0) / (i1 - i0)
            pos = i0 + np.cumsum((1 - u) ** 2)
            for c in range(2):
                seg = np.interp(pos, np.arange(self.N), out[c])
                out[c, i0:i1] = seg * (1 - u ** 3)
                out[c, i1:i2] = 0
        fo = int(2.0 * SR)
        out[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
        out[:, :int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))
        return out / max(np.abs(out).max(), 1e-9) * 0.89

    def ending(self, t0, key=0):
        """the home chord, held to the end, and 3-1-4 on the mallet"""
        root = 50 + key
        hold = self.total - t0
        self.pad_chord(t0, hold - 1.0, [root - 12, root, root + 4, root + 7, root + 12, root + 16], rel=2.5, att=0.05,
                       gain=1.2, stem="final")
        self.bass_note(t0, hold - 0.6, root - 12, False)
        for k, d_ in enumerate("314"):
            self.lead(t0 + 0.0 + k * BEAT, SCALE[int(d_)] + 12 + key, 1.2, 0.7)


def main_track(O, total, marks):
    """the long video; O: section start times; marks: video times of the beats the music follows"""
    T = Track(total)
    T.section(0.0, O["einstein"], "intro")
    T.section(O["einstein"], O["stolum"], "mellow")
    T.section(O["stolum"], O["test"], "flow")
    go = marks["rest"]
    T.section(O["test"], go, "tension", prog=SUS)
    T.riser(go - 4.0, go)
    mid = go + 8 * BAR
    T.section(go, min(mid, marks["notpi"]), "full")
    if mid < marks["notpi"]:
        T.section(mid, marks["notpi"], "full2")
    T.tape_stop(marks["notpi"], 0.8, marks["notpi"] + 1.3)
    T.section(marks["notpi"] + 1.3, O["near"], "suspense", prog=SUS)
    T.section(O["near"], O["common"], "flow")
    T.riser(O["common"] - 3.0, O["common"])
    T.section(O["common"], O["end"], "full", key=0)
    T.section(O["end"], marks["final"], "triumph")
    T.ending(marks["final"])
    for t, sz in ((go, 0.8), (O["common"], 0.9)):
        T.impact(t, sz)
        T.crash(t, 1.0)
    T.bell(O["near"] + 0.5, 81)
    return T.mixdown()
