"""Original music for the flights videos, synthesized from scratch (no samples, no licensing issues).

D minor, 100 BPM, progression Dm-Bb-F-C. Detuned-saw pads, a pulsing bass, a plucked arpeggio
through a ping-pong delay, light drums, plus risers and impacts placed on the picture's key moments.
An energy curve per video decides what plays when. After "farther than the Sun" the music lifts a
whole step, and it resolves on a major chord when the plane lands.

    python score.py full    -> out/score_full.wav   (matches every_flight_light_speed_v2.mp4)
    python score.py short   -> out/score_short.wav  (matches every_flight_short.mp4)
"""
import os, sys, subprocess
import numpy as np
import numba as nb
from scipy.signal import oaconvolve, butter, sosfilt
from scipy.io import wavfile

SR = 48000
BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rng = np.random.default_rng(7)

# chords: (root midi for the arpeggio octave, minor?)
PROG = [(62, True), (58, False), (65, False), (60, False)]       # Dm Bb F C


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


@nb.njit(cache=True)
def saw(freq, n, phase0):
    out = np.empty(n)
    ph = phase0
    dt = freq / SR
    for i in range(n):
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


@nb.njit(cache=True)
def svf_lp(x, cutoff, q):
    y = np.empty_like(x)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for i in range(len(x)):
        g = np.tan(np.pi * min(cutoff[i], SR * 0.45) / SR)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        y[i] = v2
    return y


def env_adsr(n, a, r, hold_n):
    """attack a s, sustain to hold_n samples, release r s (exponential)."""
    t = np.arange(n) / SR
    e = np.minimum(1.0, t / max(a, 1e-4))
    e = np.sin(e * np.pi / 2) ** 2
    rel = t >= hold_n / SR
    e[rel] *= np.exp(-(t[rel] - hold_n / SR) / max(r / 4.6, 1e-4))
    return e


class Score:
    def __init__(self, total, energy, impacts, risers, transpose_at=None, final_at=None, intro_hit=True):
        self.total = total
        self.N = int(np.ceil(total * SR))
        self.energy_pts = np.array(energy, float)
        self.impacts = impacts
        self.risers = risers
        self.transpose_at = transpose_at
        self.final_at = final_at
        self.stems = {k: np.zeros((2, self.N), np.float32) for k in ("pad", "bass", "arp", "kick", "hat", "clap", "fx", "final")}
        self.duck = np.ones(self.N, np.float32)

    def e(self, t):
        return float(np.interp(t, self.energy_pts[:, 0], self.energy_pts[:, 1]))

    def tr(self, t):
        return 2 if (self.transpose_at is not None and t >= self.transpose_at - 0.2) else 0

    def add(self, stem, i0, sig, pan=0.0):
        i0 = int(i0)
        if i0 >= self.N:
            return
        n = min(sig.shape[-1], self.N - i0)
        if sig.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            self.stems[stem][0, i0:i0 + n] += sig[:n] * l * 1.414
            self.stems[stem][1, i0:i0 + n] += sig[:n] * r * 1.414
        else:
            self.stems[stem][:, i0:i0 + n] += sig[:, :n]

    # ------------------------------------------------------------------ instruments
    def pad_chord(self, t0, dur, notes, rel=2.4, att=0.9, gain=1.0, stem="pad"):
        n = int((dur + rel) * SR)
        hold = int(dur * SR)
        env = env_adsr(n, att, rel, hold)
        out = np.zeros((2, n))
        for j, m in enumerate(notes):
            f = mtof(m)
            for ch in range(2):
                for c in (-9.0, 0.0, 9.0):
                    det = 2 ** ((c + (5.0 if ch else -5.0)) / 1200)
                    out[ch] += saw(f * det, n, rng.random()) * (0.55 if c else 0.8)
        self.add(stem, t0 * SR, out * env * gain / len(notes))

    def bass_note(self, t0, dur, m, pulse):
        n = int((dur + 0.08) * SR)
        t = np.arange(n) / SR
        f = mtof(m)
        sig = np.sin(2 * np.pi * f * t) + 0.35 * saw(2 * f, n, 0.0)
        if pulse:
            env = np.minimum(1, t / 0.004) * (0.45 + 0.55 * np.exp(-t / 0.11))
        else:
            env = np.minimum(1, t / 0.05)
        env *= np.where(t > dur, np.exp(-(t - dur) / 0.02), 1.0)
        self.add("bass", t0 * SR, sig * env)

    def arp_note(self, t0, m, vel, bright):
        n = int(0.7 * SR)
        t = np.arange(n) / SR
        f = mtof(m)
        ph = 2 * np.pi * f * t
        sig = np.sin(ph) + 0.12 * np.sin(2 * ph) * bright + 0.04 * np.sin(3 * ph) * bright
        env = np.minimum(1, t / 0.002) * np.exp(-t / (0.08 + 0.08 * bright))
        self.add("arp", t0 * SR, sig * env * vel)

    def kick(self, t0, vel=1.0):
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        f = 46 + 80 * np.exp(-t / 0.035)
        ph = 2 * np.pi * np.cumsum(f) / SR
        sig = np.sin(ph) * np.exp(-t / 0.26) + rng.normal(0, 1, n) * np.exp(-t / 0.002) * 0.25
        self.add("kick", t0 * SR, sig * vel)
        i0 = int(t0 * SR)
        m = min(int(0.4 * SR), self.N - i0)
        if m > 0:
            tt = np.arange(m) / SR
            self.duck[i0:i0 + m] = np.minimum(self.duck[i0:i0 + m], 1 - 0.42 * np.exp(-tt / 0.13))

    def hat(self, t0, vel, open_=False):
        n = int(0.25 * SR)
        t = np.arange(n) / SR
        sos = butter(4, 7500, "hp", fs=SR, output="sos")
        sig = sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-t / (0.09 if open_ else 0.03))
        self.add("hat", t0 * SR, sig * vel, pan=0.3)

    def clap(self, t0, vel=1.0):
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        sos = butter(2, [900, 2800], "bp", fs=SR, output="sos")
        env = np.zeros(n)
        for d in (0.0, 0.011, 0.023):
            env += np.where(t >= d, np.exp(-(t - d) / 0.012), 0)
        env += np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.11), 0) * 0.6
        self.add("clap", t0 * SR, sosfilt(sos, rng.normal(0, 1, n)) * env * vel, pan=-0.1)

    def riser(self, ta, tb):
        n = int((tb - ta) * SR)
        if n <= 0:
            return
        u = np.linspace(0, 1, n)
        noise = rng.normal(0, 1, (2, n))
        cut = 250 * (9000 / 250) ** (u ** 1.6)
        sig = np.stack([svf_lp(noise[c], cut, 1.6) for c in range(2)])
        tone_f = 180 * (6.0 ** u)
        tone = np.sin(2 * np.pi * np.cumsum(tone_f) / SR) * 0.35
        amp = u ** 2.2
        self.add("fx", ta * SR, (sig + tone) * amp)

    def impact(self, t0, size=1.0):
        n = int(3.5 * SR)
        t = np.arange(n) / SR
        f = 34 + 30 * np.exp(-t / 0.15)
        boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
        sos = butter(2, 2500, "lp", fs=SR, output="sos")
        burst = sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-t / 0.35) * 0.6
        self.add("fx", t0 * SR, (boom + burst) * size)

    # ------------------------------------------------------------------ arrangement
    def compose(self):
        n_bars = int(np.ceil(self.total / BAR))
        end_play = self.final_at if self.final_at is not None else self.total
        pats16 = [[0, 2, 4, 5, 3, 5, 4, 2, 1, 3, 5, 4, 2, 4, 3, 1],
                  [0, 1, 2, 4, 2, 1, 3, 5, 0, 2, 3, 5, 4, 3, 2, 4],
                  [5, 4, 2, 3, 1, 2, 0, 2, 5, 3, 4, 2, 3, 1, 2, 4]]
        for b in range(n_bars):
            tb = b * BAR
            if tb >= end_play - 1e-6:
                break
            e = self.e(tb + 0.01)
            root, minor = PROG[(b // 2) % 4]
            root += self.tr(tb)
            third = 3 if minor else 4
            # pad: every 2 bars
            if b % 2 == 0:
                dur = min(2 * BAR, end_play - tb)
                pr = root - 12
                self.pad_chord(tb, dur, [pr - 12 + 7, pr, pr + third, pr + 7, pr + 12], rel=2.4 if dur > 1 else 0.6)
            # bass
            bm = root - 24
            if e >= 0.45:
                for k in range(8):
                    t = tb + k * BEAT / 2
                    if t < end_play:
                        self.bass_note(t, min(BEAT / 2 * 0.9, end_play - t), bm + (12 if k in (3, 7) and e > 0.8 else 0), True)
            else:
                self.bass_note(tb, min(BAR, end_play - tb) * 0.98, bm, False)
            # drums
            for k in range(4):
                t = tb + k * BEAT
                if t >= end_play:
                    break
                if e >= 0.5:
                    self.kick(t, 0.9 + 0.1 * (k == 0))
                if e >= 0.8 and k in (1, 3):
                    self.clap(t, 0.85)
                if e >= 0.62:
                    self.hat(t + BEAT / 2, 0.8 + 0.2 * rng.random(), open_=(e >= 0.85 and k == 3))
                    if e >= 0.85:
                        self.hat(t + BEAT / 4, 0.35 + 0.1 * rng.random())
                        self.hat(t + 3 * BEAT / 4, 0.35 + 0.1 * rng.random())
            # arpeggio
            if e >= 0.2:
                tones = [root, root + third, root + 7, root + 12, root + 12 + third, root + 19]
                pat = pats16[(b // 4) % len(pats16)]
                step = 1 if e >= 0.85 else 2
                for k in range(0, 16, step):
                    t = tb + k * BEAT / 4
                    if t >= end_play:
                        break
                    vel = (0.55 + 0.45 * (k % 4 == 0)) * (0.85 + 0.15 * rng.random())
                    self.arp_note(t, tones[pat[k]], vel, bright=min(1.0, 0.3 + e))
        for ta, tb in self.risers:
            self.riser(ta, tb)
        for t, s in self.impacts:
            self.impact(t, s)
        if self.final_at is not None:
            root = PROG[0][0] + self.tr(self.final_at)
            hold = self.total - self.final_at
            pr = root - 12
            self.pad_chord(self.final_at, hold - 1.0, [pr - 5, pr, pr + 4, pr + 7, pr + 12, pr + 16], rel=3.5, att=0.05,
                           gain=1.3, stem="final")
            self.pad_chord(self.final_at, hold - 1.0, [pr + 24, pr + 28, pr + 31], rel=3.5, att=1.2, gain=0.6, stem="final")
            self.bass_note(self.final_at, hold - 0.5, root - 24, False)

    # ------------------------------------------------------------------ mix
    def mix(self):
        S = self.stems
        t = np.arange(self.N) / SR
        e_curve = np.interp(t, self.energy_pts[:, 0], self.energy_pts[:, 1])
        # pad brightness follows the energy; sidechain pump on pad and bass
        cut = 600 + 2600 * e_curve
        for c in range(2):
            S["pad"][c] = svf_lp(S["pad"][c], cut, 0.8) * self.duck
            S["final"][c] = svf_lp(S["final"][c], np.full(self.N, 3800.0), 0.7)
            S["bass"][c] = svf_lp(S["bass"][c], np.full(self.N, 420.0), 0.75) * (0.55 + 0.45 * self.duck)
            S["arp"][c] = svf_lp(S["arp"][c], 1800 + 4200 * e_curve, 0.9)
        # ping-pong delay on the arpeggio (dotted eighth)
        d = int(0.75 * BEAT * SR)
        irl = np.zeros(7 * d + 1); irr = np.zeros(7 * d + 1)
        irl[0] = irr[0] = 1.0
        for k in range(1, 7):
            (irl if k % 2 else irr)[k * d] = 0.36 ** k
        arp_m = S["arp"].mean(0)
        S["arp"] = np.stack([oaconvolve(arp_m, irl)[:self.N], oaconvolve(arp_m, irr)[:self.N]])
        target = {"pad": -21, "bass": -19.5, "arp": -26.5, "kick": -18.5, "hat": -33, "clap": -28, "fx": -21, "final": -19}
        mixed = np.zeros((2, self.N))
        send = np.zeros((2, self.N))
        # dynamics: quieter breakdowns, louder peaks (impacts, risers and the final chord are exempt)
        auto = (10 ** (np.clip(-7.0 * (0.9 - e_curve), -6.0, 0.7) / 20)).astype(np.float32)
        sends = {"pad": 0.30, "arp": 0.45, "clap": 0.5, "fx": 0.55, "final": 0.6, "hat": 0.15}
        for k, x in S.items():
            act = np.abs(x).max(0) > 1e-4
            if not act.any():
                continue
            rms = np.sqrt(np.mean(x[:, act] ** 2))
            g = 10 ** (target[k] / 20) / max(rms, 1e-9)
            gx = x * g if k in ("fx", "final") else x * (g * auto)
            mixed += gx
            send += gx * sends.get(k, 0)
        # reverb: stereo synthetic hall (2 decaying noise bursts, the late one darker)
        n_ir = int(3.4 * SR)
        ti = np.arange(n_ir) / SR
        sos = butter(1, 3000, "lp", fs=SR, output="sos")
        ir = []
        for c in range(2):
            early = rng.normal(0, 1, n_ir) * np.exp(-ti / 0.18)
            late = sosfilt(sos, rng.normal(0, 1, n_ir)) * np.exp(-ti / 0.62) * 1.6
            h = np.r_[np.zeros(int(0.022 * SR)), (early * 0.5 + late)][:n_ir]
            ir.append(h / np.sqrt(np.sum(h ** 2)))
        wet = np.stack([oaconvolve(send[c], ir[c])[:self.N] for c in range(2)])
        out = mixed + wet * 0.55
        out = sosfilt(butter(2, 28, "hp", fs=SR, output="sos"), out, axis=1)
        out = np.tanh(out * 1.6) / 1.6
        # fade in/out
        fi = int(0.01 * SR); fo = int(2.5 * SR)
        out[:, :fi] *= np.linspace(0, 1, fi)
        out[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
        return out / max(np.abs(out).max(), 1e-9) * 0.89


def write(out, path):
    wav = path.replace(".wav", "_raw.wav")
    wavfile.write(wav, SR, (out.T * 32767).astype(np.int16))
    # loudness normalize for YouTube (-14 LUFS, -1.5 dBTP)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", str(SR), path],
                   check=True)
    os.remove(wav)
    print("wrote", path)


def full():
    L = np.load(os.path.join(os.path.dirname(OUT), "data", "flight_layers.npz"))
    t1, flown = L["leg_t1"], np.cumsum(L["legs"][:, 0] >= 0)
    DAY, CARD, T2 = 12.0, 2.5, float(t1[-1])
    TO = DAY + CARD                     # takeoff
    FIN = TO + T2                       # landing
    TOTAL = (int(DAY * 30) + int(CARD * 30) + int(np.ceil(T2 * 30)) + 1 + int(5.0 * 30)) / 30
    HALF = TO + float(t1[np.searchsorted(flown, flown[-1] / 2)])     # half the flights flown
    SUN = TO + 149.6e6 / 299_792.458                                  # distance flown = Earth-Sun
    print(f"takeoff {TO:.2f}  half {HALF:.2f}  sun {SUN:.2f}  landing {FIN:.2f}  total {TOTAL:.2f}")
    energy = [(0, 0.78), (DAY - 0.3, 0.78), (DAY, 0.3), (TO, 0.7), (TO + 40, 0.7), (TO + 41, 0.78),
              (TO + 150, 0.78), (TO + 151, 0.4), (TO + 240, 0.4), (TO + 250, 0.86),
              (TO + 440, 0.86), (TO + 441, 0.45), (SUN - 0.2, 0.45), (SUN, 0.9), (TO + 600, 0.9), (FIN, 1.0),
              (FIN + 0.01, 0.0), (TOTAL, 0.0)]
    impacts = [(0.0, 0.8), (TO, 1.0), (TO + 250, 0.8), (HALF, 0.9), (SUN, 1.2), (FIN, 1.4)]
    risers = [(DAY, TO), (TO + 240, TO + 250), (HALF - 8, HALF), (SUN - 14, SUN), (FIN - 8, FIN)]
    sc = Score(TOTAL, energy, impacts, risers, transpose_at=SUN, final_at=FIN)
    sc.compose()
    write(sc.mix(), os.path.join(OUT, "score_full.wav"))


def short():
    DAY, CARD, LIGHT, END = 7.0, 2.0, 24.0, 3.6
    TO = DAY + CARD
    FIN = TO + LIGHT
    TOTAL = DAY + CARD + LIGHT + END
    energy = [(0, 0.82), (DAY - 0.2, 0.82), (DAY, 0.3), (TO, 0.86), (FIN - 6, 0.9), (FIN, 1.0), (FIN + 0.01, 0), (TOTAL, 0)]
    impacts = [(0.0, 0.8), (TO, 1.0), (FIN, 1.4)]
    risers = [(DAY, TO), (FIN - 4, FIN)]
    sc = Score(TOTAL, energy, impacts, risers, final_at=FIN)
    sc.compose()
    write(sc.mix(), os.path.join(OUT, "score_short.wav"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"full": full, "short": short}[sys.argv[1]]()
