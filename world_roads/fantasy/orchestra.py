"""A small synthesized orchestra for a gridiron march: brass (trumpets, horns, trombones, tuba), strings, timpani,
military snare, concert bass drum and cymbals, mixed into a concert hall. Everything is synthesized from scratch: no
samples, no licensing.

* brass: three or four detuned band-limited saws per note; a pitch scoop into each note and a little vibrato on long
  ones; a low-pass that opens with the note's loudness (the brassy bloom), and an accent overshoot on hard attacks;
* strings: five detuned saws per note, sustained (slow bow) or marcato (short, accented), darker below;
* timpani: five inharmonic membrane modes with a slight downward pitch glide and a mallet thump; rolls;
* snare: a tuned shell and a burst of wire noise; rolls and flams;
* bass drum, crash cymbal and suspended-cymbal swell;
* a stereo hall (early reflections and a 2.6 s darkening tail), stem levels by loudness, glue compression, soft clip.
"""
import os, sys
import numpy as np
import numba as nb
from scipy.signal import oaconvolve, butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # svf_lp, mtof, the -14 LUFS writer  # noqa: E402

SR = SC.SR
mtof, svf_lp = SC.mtof, SC.svf_lp
rng = np.random.default_rng(1980)


@nb.njit(cache=True)
def saw_fm(freq, phase0):
    """a band-limited saw (polyBLEP) following a per-sample frequency"""
    n = len(freq)
    out = np.empty(n)
    ph = phase0
    for i in range(n):
        dt = freq[i] / 48000.0
        v = 2.0 * ph - 1.0
        if ph < dt:
            x = ph / dt
            v -= x + x - x * x - 1.0
        elif ph > 1.0 - dt:
            x = (ph - 1.0) / dt
            v -= x * x + x + x + 1.0
        out[i] = v
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return out


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "bp", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "hp", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "lp", fs=SR, output="sos"), x)


BRASS = {   # attack s, release s, brightness (cutoff in harmonics at full), cutoff ceiling Hz, voices, detune cents,
            # accent overshoot, filter resonance, drive at full (harmonics grow with loudness), stem
    "trumpet": dict(a=0.032, r=0.20, bright=10.0, ceil=8000, voices=3, det=7, blat=0.35, q=1.4, drive=2.2, stem="tpt"),
    "horn": dict(a=0.07, r=0.32, bright=6.5, ceil=3600, voices=4, det=8, blat=0.18, q=1.1, drive=1.2, stem="hrn"),
    "trombone": dict(a=0.05, r=0.24, bright=12.0, ceil=6500, voices=3, det=6, blat=0.4, q=1.7, drive=2.8, stem="tbn"),
    "tuba": dict(a=0.06, r=0.26, bright=4.0, ceil=1400, voices=2, det=5, blat=0.2, q=1.0, drive=1.0, stem="tbn"),
}
PAN = {"tpt": 0.28, "hrn": -0.32, "tbn": 0.12, "str_hi": -0.18, "str_lo": 0.2, "timp": -0.12, "snare": 0.06, "bd": 0.0,
       "cym": 0.15}
LEVEL = {"str_lo": -19.5, "str_hi": -23.0, "hrn": -20.0, "tpt": -21.5, "tbn": -21.0, "timp": -20.5, "snare": -23.0,
         "bd": -22.0, "cym": -27.0}
SEND = {"str_lo": 0.32, "str_hi": 0.42, "hrn": 0.42, "tpt": 0.32, "tbn": 0.3, "timp": 0.3, "snare": 0.2, "bd": 0.12,
        "cym": 0.3}


class Orchestra:
    def __init__(self, total):
        self.total = total
        self.N = int(np.ceil(total * SR))
        self.stems = {k: np.zeros((2, self.N), np.float32) for k in PAN}
        self.auto = [(0.0, 1.0), (total, 1.0)]       # the master dynamics: (time s, gain)

    def add(self, stem, t0, sig, pan=None, width=0.0):
        """mono signal into a stem at t0 s, panned (equal power); width decorrelates the two sides a little"""
        i0 = int(round(t0 * SR))
        if i0 >= self.N or i0 + len(sig) <= 0:
            return
        if i0 < 0:
            sig, i0 = sig[-i0:], 0
        n = min(len(sig), self.N - i0)
        p = PAN[stem] if pan is None else pan
        lg, rg = np.cos((p + 1) * np.pi / 4) * 1.414, np.sin((p + 1) * np.pi / 4) * 1.414
        self.stems[stem][0, i0:i0 + n] += sig[:n] * lg
        d = int(width * SR)
        if d:
            self.stems[stem][1, i0 + d:i0 + n] += sig[:n - d] * rg
        else:
            self.stems[stem][1, i0:i0 + n] += sig[:n] * rg

    # ------------------------------------------------------------------ brass
    def brass(self, t0, dur, m, vel=0.8, kind="horn", pan=None):
        p = BRASS[kind]
        n = int((dur + p["r"] + 0.05) * SR)
        t = np.arange(n) / SR
        f0 = float(mtof(m))
        a = p["a"] * (1.25 - 0.45 * vel)
        env = np.minimum(1.0, t / a) ** 1.5
        env *= 1 + p["blat"] * vel * np.exp(-np.maximum(t - a, 0) / 0.09) * (t > 0.5 * a)
        env *= np.where(t > dur, np.exp(-(t - dur) / (p["r"] / 4.6)), 1.0)
        env *= 0.35 + 0.65 * vel
        scoop = -32 * np.exp(-t / 0.035)
        vib = 9 * np.sin(2 * np.pi * 5.1 * t + rng.random() * 6) * np.clip((t - 0.35) / 0.4, 0, 1)
        out = np.zeros(n)
        for v in range(p["voices"]):
            cents = scoop + vib + p["det"] * (v - (p["voices"] - 1) / 2) + rng.normal(0, 2)
            s = saw_fm(f0 * 2 ** (cents / 1200), rng.random())
            d0 = int(rng.uniform(0, 0.012) * SR)
            out[d0:] += s[:n - d0]
        loud = np.minimum(1.0, env / (0.35 + 0.65 * vel)) * vel
        out = np.tanh(out / p["voices"] * (1 + p["drive"] * loud)) * p["voices"] / (1 + 0.4 * p["drive"] * vel)
        cut = np.minimum(f0 * (1.4 + p["bright"] * (0.3 + 0.7 * vel) * np.minimum(1.0, env)), p["ceil"])
        out = svf_lp(svf_lp(out, cut, p["q"]), cut * 1.7, 0.6)
        breath = bp(rng.normal(0, 1, n), min(f0 * 3, 5000), min(f0 * 6, 9000)) * np.exp(-t / 0.03) * 0.25 * vel
        self.add(p["stem"], t0, (out / p["voices"] + breath) * env, pan)

    # ------------------------------------------------------------------ strings
    def strings(self, t0, dur, m, vel=0.6, marc=False, stem=None, pan=None):
        stem = stem or ("str_lo" if m < 55 else "str_hi")
        rel = 0.07 if marc else 0.45
        n = int((dur + rel + 0.02) * SR)
        t = np.arange(n) / SR
        f0 = float(mtof(m))
        if marc:
            env = np.minimum(1.0, t / 0.008) * (0.55 + 0.45 * np.exp(-t / 0.07))
        else:
            env = np.sin(np.minimum(1.0, t / (0.16 - 0.06 * vel)) * np.pi / 2) ** 2
        env *= np.where(t > dur, np.exp(-(t - dur) / (rel / 4.6)), 1.0) * (0.3 + 0.7 * vel)
        vib = np.zeros(n) if marc else 7 * np.sin(2 * np.pi * 5.5 * t + rng.random() * 6) * np.clip((t - 0.2) / 0.5, 0, 1)
        out = np.zeros(n)
        for v in range(5):
            cents = vib + 6 * (v - 2) + rng.normal(0, 1.5)
            out += saw_fm(f0 * 2 ** (cents / 1200), rng.random())
        top = 5200 if marc else 3400
        cut = np.minimum(f0 * ((7 if marc else 4) + 6 * vel) * (0.6 + 0.4 * np.minimum(1, env / (0.3 + 0.7 * vel))), top)
        out = svf_lp(svf_lp(out, cut, 0.7), cut * 1.5, 0.6) / 5
        bow = hp(rng.normal(0, 1, n), 2500) * 0.012 * env
        self.add(stem, t0, out * env + bow, pan, width=0.0)

    # ------------------------------------------------------------------ percussion
    def timpani(self, t0, m, vel=0.8):
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        f0 = float(mtof(m)) * (1 + 0.035 * vel * np.exp(-t / 0.03))
        sig = np.zeros(n)
        for r, amp, dec in ((1.0, 1.0, 1.7), (1.5, 0.5, 1.0), (1.98, 0.36 * vel, 0.65), (2.44, 0.22 * vel, 0.45),
                            (2.9, 0.12 * vel, 0.3)):
            sig += amp * np.sin(2 * np.pi * np.cumsum(f0 * r) / SR + rng.random()) * np.exp(-t / dec)
        thump = lp(rng.normal(0, 1, n), 380) * np.exp(-t / 0.025) * 2.2
        self.add("timp", t0, (sig + thump) * vel * np.minimum(1, t / 0.002))

    def timp_roll(self, t0, t1, m, v0=0.2, v1=0.9):
        t = t0
        while t < t1:
            u = (t - t0) / max(t1 - t0, 1e-3)
            self.timpani(t, m, (v0 + (v1 - v0) * u) * rng.uniform(0.85, 1.0))
            t += rng.uniform(0.055, 0.068)

    def snare(self, t0, vel=0.8):
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        shell = (np.sin(2 * np.pi * 186 * t) + 0.5 * np.sin(2 * np.pi * 331 * t)) * np.exp(-t / 0.05)
        wires = bp(rng.normal(0, 1, n), 1600, 7000) * np.exp(-t / (0.09 + 0.08 * vel))
        crack = hp(rng.normal(0, 1, n), 4000) * np.exp(-t / 0.006)
        self.add("snare", t0, (0.55 * shell + wires + 0.6 * crack) * vel ** 1.3 * np.minimum(1, t / 0.0008))

    def snare_roll(self, t0, t1, v0=0.25, v1=0.9, rate=0.045):
        t = t0
        while t < t1:
            u = (t - t0) / max(t1 - t0, 1e-3)
            self.snare(t, (v0 + (v1 - v0) * u) * rng.uniform(0.8, 1.0))
            t += rate * rng.uniform(0.92, 1.08)

    def bass_drum(self, t0, vel=0.8):
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        f = 48 + 30 * np.exp(-t / 0.04)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55) + lp(rng.normal(0, 1, n), 200) * np.exp(-t / 0.06)
        self.add("bd", t0, sig * vel * np.minimum(1.0, t / 0.002))

    def crash(self, t0, vel=0.8):
        n = int(4.0 * SR)
        t = np.arange(n) / SR
        noise = rng.normal(0, 1, (2, n))
        onset = np.minimum(1.0, t / 0.004) ** 2                       # a few ms in: no codec-hostile spike
        sig = [(bp(noise[c], 2800, 11000) + 0.5 * bp(noise[c], 4200, 6500, 1)) * (0.6 * np.exp(-t / 0.16) + 0.5 * np.exp(-t / 1.8))
               * (1 + 0.15 * np.sin(2 * np.pi * 6.3 * t + c)) * onset for c in range(2)]
        i0 = int(t0 * SR)
        k = min(n, self.N - i0)
        if k > 0:
            self.stems["cym"][:, i0:i0 + k] += np.stack(sig)[:, :k] * vel

    def swell(self, t0, t1, vel=0.6):
        """a suspended cymbal rolled with soft mallets, rising into t1"""
        n = int((t1 - t0) * SR)
        if n <= 0:
            return
        u = np.linspace(0, 1, n)
        noise = rng.normal(0, 1, (2, n))
        sig = np.stack([bp(noise[c], 1500, 9000) * u ** 2.2 for c in range(2)]) * vel * 0.6
        i0 = int(t0 * SR)
        k = min(n, self.N - i0)
        self.stems["cym"][:, i0:i0 + k] += sig[:, :k]

    # ------------------------------------------------------------------ mix
    def hall(self):
        """a stereo concert-hall impulse: early reflections, then a tail darkening as it decays (T60 about 2.6 s)"""
        n = int(3.2 * SR)
        t = np.arange(n) / SR
        ir = np.zeros((2, n))
        for c in range(2):
            for d, g in ((0.011, 0.5), (0.019, 0.38), (0.027, 0.3), (0.041, 0.24), (0.058, 0.2), (0.073, 0.16)):
                ir[c, int((d + rng.uniform(-0.002, 0.002)) * SR)] += g * rng.choice([-1, 1])
            late = rng.normal(0, 1, n) * np.exp(-t / (2.6 / 6.9)) * np.clip((t - 0.03) / 0.05, 0, 1)
            bright = lp(late, 5000) * np.exp(-t / 0.5)
            dark = lp(late, 1800)
            ir[c] += (bright + dark) * 0.5
            ir[c] /= np.sqrt(np.sum(ir[c] ** 2))
        return ir

    def mix(self):
        mixed = np.zeros((2, self.N))
        send = np.zeros((2, self.N))
        for k, x in self.stems.items():
            act = np.abs(x).max(0) > 1e-4
            if not act.any():
                continue
            rms = np.sqrt(np.mean(x[:, act] ** 2))
            g = 10 ** (LEVEL[k] / 20) / max(rms, 1e-9)
            mixed += x * g
            send += x * g * SEND[k]
        ir = self.hall()
        wet = np.stack([oaconvolve(send[c], ir[c])[:self.N] for c in range(2)])
        out = mixed + wet * 0.75
        out = sosfilt(butter(2, 30, "hp", fs=SR, output="sos"), out, axis=1)
        out = sosfilt(butter(2, 17000, "lp", fs=SR, output="sos"), out, axis=1)
        # glue: a slow compressor on the sum (2:1 above -16 dBFS RMS); then the master dynamics; a gentle soft clip
        rms = np.sqrt(lp(np.mean(out ** 2, 0), 6) .clip(1e-12))
        over = np.maximum(20 * np.log10(rms) + 16, 0)
        out *= 10 ** (-over * 0.5 / 20)
        ta, ga = np.array(self.auto).T
        out *= np.interp(np.arange(self.N) / SR, ta, ga)
        out = np.tanh(out * 1.4) / 1.4
        fi, fo = int(0.005 * SR), int(1.5 * SR)
        out[:, :fi] *= np.linspace(0, 1, fi)
        out[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
        return out / max(np.abs(out).max(), 1e-9) * 0.89
