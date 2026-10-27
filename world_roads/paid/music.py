"""The races' score: a sports anthem, synthesized from scratch (no samples, no licensing).  python music.py [--career]

Brass, strings, timpani and drums built from the oscillators and filters of ../flights/score.py. 85.7 BPM in a
half-time feel: a beat is 0.7 s, a bar 2.8 s (two years at the steady pace; a paced race's years take whole
half-beats, so they still turn on the music). C major, I-V-vi-IV (C G Am F), a whole step up when free agency arrives.

* The title card: a timpani roll swelling into a trumpet fanfare; the board lands on a crash and an orchestral hit.
* The race: a driving sixteenth-note string ostinato, held horn and trombone chords, timpani on the downbeats; as the
  leader's pay climbs (its log, smoothed: the energy) toms, kick and snare join, then an eight-bar heroic theme in the
  horns, doubled by trumpets an octave up at the peaks, with crashes at the top of each phrase.
* The big moments: an orchestral hit (a brass stab, timpani, crash) at each caption that marks a record or a milestone,
  a crash at each new No. 1.
* The end: a snare and timpani roll into a last hit, then a held C major chord with a trumpet flourish under the summary.
"""
import os, sys
import numpy as np
from scipy.signal import oaconvolve, butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # noqa: E402  (oscillators, filters, the writer)

SR = SC.SR
BEAT = 0.7
BAR = 4 * BEAT
S16 = BEAT / 4
rng = np.random.default_rng(11)

PROG = [(48, (0, 4, 7)), (43, (0, 4, 7)), (45, (0, 3, 7)), (41, (0, 4, 7))]       # C G Am F: bass root, triad
# the theme: (sixteenth it starts on, sixteenths it lasts, midi), two four-bar halves over the progression
THEME = [
    [(0, 3, 72), (3, 1, 72), (4, 8, 79), (12, 4, 76)],
    [(0, 4, 74), (4, 2, 71), (6, 2, 74), (8, 8, 79)],
    [(0, 3, 69), (3, 1, 69), (4, 8, 76), (12, 4, 72)],
    [(0, 4, 77), (4, 4, 76), (8, 4, 74), (12, 2, 72), (14, 2, 74)],
    [(0, 6, 76), (6, 2, 74), (8, 4, 72), (12, 4, 67)],
    [(0, 3, 71), (3, 1, 72), (4, 4, 74), (8, 8, 79)],
    [(0, 6, 81), (6, 2, 79), (8, 4, 76), (12, 4, 72)],
    [(0, 4, 77), (4, 4, 81), (8, 8, 79)],
]
OSTINATO = [0, 0, 12, 0, 7, 0, 12, 7, 0, 0, 12, 0, 7, 12, 7, 12]                  # over the bar's root, in sixteenths
ACCENT = {0, 3, 6, 10, 13}


def lp(x, cut, q=0.7):
    return SC.svf_lp(np.ascontiguousarray(x, dtype=np.float64), np.broadcast_to(np.asarray(cut, float), x.shape).copy(), q)


class Anthem:
    def __init__(self, total, energy):
        self.total = total
        self.N = int(np.ceil(total * SR))
        self.E = np.array(energy, float)
        self.stems = {k: np.zeros((2, self.N), np.float32) for k in ("brass", "lead", "strings", "timp", "drums", "cym", "hit")}

    def e(self, t):
        return float(np.interp(t, self.E[:, 0], self.E[:, 1]))

    def add(self, stem, t0, sig, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= self.N or i0 < 0:
            return
        sig = sig.copy()
        k = min(480, sig.shape[-1])                   # every sound ends on a 10 ms fade: no clicks where it stops
        sig[..., -k:] *= np.linspace(1, 0, k)
        n = min(sig.shape[-1], self.N - i0)
        if sig.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            self.stems[stem][0, i0:i0 + n] += sig[:n] * l * 1.414
            self.stems[stem][1, i0:i0 + n] += sig[:n] * r * 1.414
        else:
            self.stems[stem][:, i0:i0 + n] += sig[:, :n]

    # ------------------------------------------------------------------ instruments
    def brass(self, t0, dur, m, vel=1.0, bright=1.0, stem="brass", voices=3, pan=0.0, att=0.035):
        """a brass section: detuned saws through a low-pass that opens with the attack (the 'blat') and settles; a
        slow vibrato on held notes"""
        n = int((dur + 0.5) * SR)
        t = np.arange(n) / SR
        f0 = SC.mtof(m)
        cut = np.minimum(SR * 0.45, f0 * (2.2 + 5.5 * bright * vel) * (1 + 1.4 * np.exp(-t / 0.07)) * np.minimum(1, t / att + 0.3))
        env = np.minimum(1, t / att) ** 1.5
        env *= np.where(t < dur, 1 - 0.18 * (1 - np.exp(-t / 0.25)), np.exp(-(t - dur) / 0.09) * (1 - 0.18 * (1 - np.exp(-dur / 0.25))))
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.3) / 0.3, 0, 1)
        out = np.zeros((2, n))
        for v in range(voices):
            det = 2 ** ((v - (voices - 1) / 2) * 0.07 / 12)
            sig = SC.saw(float(f0 * det), n, rng.random())
            if dur > 0.5:                              # vibrato by resampling, for held notes only
                idx = np.clip(np.cumsum(vib) - vib[0], 0, n - 1)
                sig = np.interp(idx, np.arange(n), sig)
            sig = lp(sig, cut, 0.9)
            p = pan + (v - (voices - 1) / 2) * 0.35
            l, r = np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)
            d = int(rng.uniform(0, 0.012) * SR)
            out[0, d:] += sig[:n - d] * l
            out[1, d:] += sig[:n - d] * r
        self.add(stem, t0, out * env * vel / voices)

    def string(self, t0, dur, m, vel=1.0, bright=1.0):
        n = int((dur + 0.25) * SR)
        t = np.arange(n) / SR
        f0 = SC.mtof(m)
        env = np.minimum(1, t / 0.006) * np.where(t < dur, 0.55 + 0.45 * np.exp(-t / 0.05), 0.55 * np.exp(-(t - dur) / 0.04))
        sig = sum(SC.saw(float(f0 * 2 ** (d / 1200)), n, rng.random()) for d in (-9, 0, 8))
        sig = lp(sig, min(SR * 0.45, 1500 + 2600 * bright), 0.8)
        self.add("strings", t0, sig * env * vel / 3, pan=rng.uniform(-0.35, 0.35))

    def timpani(self, t0, m, vel=1.0, click=0.8):
        n = int(4.5 * SR)
        t = np.arange(n) / SR
        f0 = SC.mtof(m) * (1 + 0.035 * np.exp(-t / 0.08))
        sig = np.zeros(n)
        for ratio, amp, dec in ((1.0, 1.0, 0.9), (1.504, 0.5, 0.55), (1.742, 0.3, 0.4), (2.0, 0.25, 0.35), (2.245, 0.15, 0.25)):
            sig += amp * np.sin(2 * np.pi * np.cumsum(f0 * ratio) / SR) * np.exp(-t / dec)
        click = lp(rng.normal(0, 1, n), 1800) * np.exp(-t / 0.012) * click
        self.add("timp", t0, (sig + click) * vel * np.minimum(1, t / 0.002), pan=-0.15)

    def timp_roll(self, ta, tb, m, v0=0.15, v1=1.0):
        t = ta
        while t < tb:
            u = (t - ta) / max(tb - ta, 1e-6)
            self.timpani(t, m, (v0 + (v1 - v0) * u ** 1.5) * rng.uniform(0.75, 1.0) * 0.55, click=0.2)
            t += 0.052

    def tom(self, t0, f, vel=1.0, pan=0.0):
        n = int(1.5 * SR)
        t = np.arange(n) / SR
        fr = f * (1 + 0.6 * np.exp(-t / 0.035))
        sig = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.3) + lp(rng.normal(0, 1, n), 2500) * np.exp(-t / 0.03) * 0.4
        self.add("drums", t0, sig * vel, pan=pan)

    def kick(self, t0, vel=1.0):
        n = int(1.4 * SR)
        t = np.arange(n) / SR
        f = 48 + 90 * np.exp(-t / 0.03)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28) + rng.normal(0, 1, n) * np.exp(-t / 0.002) * 0.3
        self.add("drums", t0, sig * vel)

    def snare(self, t0, vel=1.0):
        n = int(1.0 * SR)
        t = np.arange(n) / SR
        body = (np.sin(2 * np.pi * 185 * t) + 0.6 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.07)
        sos = butter(2, [1400, 9000], "bp", fs=SR, output="sos")
        noise = sosfilt(sos, rng.normal(0, 1, n)) * np.exp(-t / 0.16)
        self.add("drums", t0, (body * 0.6 + noise * 1.1) * vel, pan=0.08)

    def snare_roll(self, ta, tb, v0=0.15, v1=0.9):
        t, k = ta, 0
        while t < tb:
            u = (t - ta) / max(tb - ta, 1e-6)
            self.snare(t, (v0 + (v1 - v0) * u ** 2) * (0.85 if k % 2 else 1.0) * 0.6)
            t += S16 / 2
            k += 1

    def crash(self, t0, vel=1.0):
        n = int(5.5 * SR)
        t = np.arange(n) / SR
        sos = butter(2, 3200, "hp", fs=SR, output="sos")
        out = np.stack([sosfilt(sos, rng.normal(0, 1, n)) for _ in range(2)])
        shimmer = sum(np.sin(2 * np.pi * f * t + rng.random() * 6) for f in (3510, 4870, 6230, 7710)) * 0.08
        env = np.exp(-t / 1.1) * np.minimum(1, t / 0.003)
        self.add("cym", t0, (out + shimmer) * env * vel)

    def hit(self, t0, key, vel=1.0):
        """an orchestral hit: the tonic chord stabbed by all the brass, timpani, a crash, a low boom"""
        for m in (36, 48, 55, 60, 64, 67, 72, 76, 79):
            self.brass(t0, 0.34, m + key, vel * 0.9, bright=1.3, stem="hit", voices=2)
        self.timpani(t0, 36 + key + 7, vel)
        self.crash(t0, vel * 0.9)
        n = int(4.5 * SR)
        t = np.arange(n) / SR
        self.add("hit", t0, np.sin(2 * np.pi * np.cumsum(32 + 40 * np.exp(-t / 0.12)) / SR) * np.exp(-t / 0.8) * vel * 0.8)

    # ------------------------------------------------------------------ the arrangement
    def compose(self, race_at, end_at, lift_at, hits, crashes):
        """race_at: the board appears; end_at: the race ends (the summary follows); lift_at: the key goes up a step;
        hits: caption moments; crashes: new No. 1s"""
        # the title card: a timpani roll, the fanfare, the hit as the board lands
        self.timp_roll(0.2, race_at - BAR, 31, 0.1, 0.9)
        fan = [(0, 2, 67), (2, 1, 67), (3, 1, 67), (4, 4, 72), (8, 2, 67), (10, 2, 72), (12, 4, 76)]
        for s, d, m in fan:                           # the bar before the board lands
            self.brass(race_at - BAR + s * S16, d * S16 * 0.92, m, 0.8, 1.1, "lead", pan=0.1)
        self.hit(race_at, 0, 1.0)
        # the race, bar by bar
        nb = int(np.ceil((end_at - race_at) / BAR))
        for b in range(nb):
            tb = race_at + b * BAR
            e = self.e(tb + 0.05)
            key = 2 if tb >= lift_at - 0.05 else 0
            root, triad = PROG[b % 4]
            root += key
            end = min(end_at, tb + BAR)
            # strings: the ostinato, from the start; an octave up joins higher in the race
            for j, off in enumerate(OSTINATO):
                t = tb + j * S16
                if t >= end:
                    break
                acc = 1.0 if j in ACCENT else 0.62
                self.string(t, S16 * 0.8, root + 12 + off, (0.45 + 0.55 * e) * acc, bright=0.3 + 0.7 * e)
                if e > 0.62:
                    self.string(t, S16 * 0.8, root + 24 + off, (0.25 + 0.5 * e) * acc * 0.6, bright=0.5 + 0.5 * e)
            # held horns and trombones on the chord
            for m in triad:
                self.brass(tb, min(BAR, end - tb) * 0.96, root + 12 + m, 0.32 + 0.4 * e, 0.35 + 0.5 * e, "brass", voices=2,
                           pan=-0.25 if m else 0.2, att=0.12)
            self.brass(tb, min(BAR, end - tb) * 0.96, root, 0.4 + 0.4 * e, 0.4, "brass", voices=2, att=0.1)
            # timpani on the downbeat (and beat three in the busier stretches)
            self.timpani(tb, root - 12 if root - 12 >= 36 else root, 0.55 + 0.45 * e)
            if e > 0.5:
                self.timpani(tb + 2 * BEAT, root - 5 if root - 5 >= 36 else root + 7, 0.4 + 0.4 * e)
            # drums
            if e > 0.32:
                for j, f, v in ((10, 150, 0.6), (11, 120, 0.7), (13, 98, 0.85), (14, 82, 1.0)):
                    if tb + j * S16 < end:
                        self.tom(tb + j * S16, f, v * (0.4 + 0.6 * e), pan=0.3 - 0.2 * (j - 10))
            if e > 0.45:
                self.kick(tb, 0.9)
                self.kick(tb + 6 * S16, 0.6)
                if tb + 2 * BEAT < end:
                    self.snare(tb + 2 * BEAT, 0.55 + 0.45 * e)
            if e > 0.72:
                self.kick(tb + 2 * BEAT + 3 * S16, 0.5)
                if b % 2 == 1 and tb + 3.5 * BEAT < end:
                    self.snare_roll(tb + 3 * BEAT, tb + 4 * BEAT, 0.2, 0.7)
            if b % 4 == 0 and e > 0.55:
                self.crash(tb, 0.35 + 0.4 * e)
            # the theme, in eight-bar phrases, once the money is big enough
            phrase, half = (b // 8) % 2, b % 8
            if e > 0.5 and (phrase == 0 or e > 0.7):
                for s, d, m in THEME[half]:
                    t = tb + s * S16
                    if t >= end:
                        continue
                    dd = min(d * S16, end - t) * 0.95
                    self.brass(t, dd, m + key - 12, 0.55 + 0.35 * e, 0.6 + 0.4 * e, "lead", voices=3, pan=-0.15)   # horns
                    if e > 0.75:
                        self.brass(t, dd, m + key, 0.45 + 0.35 * e, 0.8 + 0.4 * e, "lead", voices=3, pan=0.2)     # trumpets
        for t in hits:
            if race_at + 0.5 < t < end_at - 0.5:
                self.hit(t, 2 if t >= lift_at else 0, 0.75)
        for t in crashes:
            if race_at + 0.5 < t < end_at - 0.5:
                self.crash(t, 0.5)
        # the end: a roll into the last hit, the held chord, a trumpet flourish
        self.snare_roll(end_at - BAR, end_at, 0.1, 0.95)
        self.timp_roll(end_at - BAR, end_at, 31 + 2, 0.1, 0.9)
        key = 2
        self.hit(end_at, key, 1.0)
        hold = self.total - end_at - 0.6
        for m in (36, 48, 55, 60, 64, 67, 72):
            self.brass(end_at + 0.05, hold * 0.8, m + key, 0.55, 0.7, "brass", voices=3, att=0.4)
        for s, d, m in ((0, 2, 72), (2, 2, 76), (4, 2, 79), (6, 10, 84)):
            self.brass(end_at + BEAT + s * S16, d * S16 * 0.95 if d < 10 else 2.4, m + key, 0.7, 1.1, "lead", pan=0.15)
        self.timp_roll(end_at + 0.6, end_at + 0.6 + 2.0, 31 + key, 0.5, 0.15)
        self.timpani(self.total - 3.0, 36 + key, 0.6)

    def mix(self):
        S = self.stems
        target = {"brass": -20.5, "lead": -19, "strings": -21.5, "timp": -21, "drums": -19.5, "cym": -30, "hit": -18}
        sends = {"brass": 0.35, "lead": 0.32, "strings": 0.28, "timp": 0.3, "drums": 0.18, "cym": 0.2, "hit": 0.4}
        t = np.arange(self.N) / SR
        e_curve = np.interp(t, self.E[:, 0], self.E[:, 1])
        auto = (10 ** (np.clip(-6.0 * (0.9 - e_curve), -5.0, 0.6) / 20)).astype(np.float32)
        mixed, send = np.zeros((2, self.N)), np.zeros((2, self.N))
        for k, x in S.items():
            act = np.abs(x).max(0) > 1e-4
            if not act.any():
                continue
            g = 10 ** (target[k] / 20) / max(np.sqrt(np.mean(x[:, act] ** 2)), 1e-9)
            gx = x * g * (1 if k == "hit" else auto)
            mixed += gx
            send += gx * sends[k]
        # a concert hall: early reflections and a darker late tail
        n_ir = int(2.9 * SR)
        ti = np.arange(n_ir) / SR
        sos = butter(1, 4200, "lp", fs=SR, output="sos")
        wet = []
        for c in range(2):
            early = rng.normal(0, 1, n_ir) * np.exp(-ti / 0.09)
            late = sosfilt(sos, rng.normal(0, 1, n_ir)) * np.exp(-ti / 0.55) * 1.5
            h = np.r_[np.zeros(int(0.018 * SR)), (early * 0.45 + late)][:n_ir]
            wet.append(oaconvolve(send[c], h / np.sqrt(np.sum(h ** 2)))[:self.N])
        out = mixed + np.stack(wet) * 0.5
        out = sosfilt(butter(2, 30, "hp", fs=SR, output="sos"), out, axis=1)
        out = np.tanh(out * 1.5) / 1.5
        fi, fo = int(0.01 * SR), int(2.2 * SR)
        out[:, :fi] *= np.linspace(0, 1, fi)
        out[:, -fo:] *= np.linspace(1, 0, fo) ** 1.4
        return out / max(np.abs(out).max(), 1e-9) * 0.89


def track():
    import video as V
    V.setup()
    total = V.total()
    at = lambda year: V.INTRO + V.D.secs(year)       # a year's moment in the film (a paced race's years vary)
    # energy: the leader's pay on a log scale, smoothed
    years = np.arange(V.FIRST, V.LAST + 1)
    lead = np.array([V.D.by[y][V.D.order[y][0]]["real"] if V.D.order.get(y) else np.nan for y in years])
    lead = np.where(np.isnan(lead), np.nanmin(lead), lead)
    ll = (np.log(lead) - np.log(lead.min())) / (np.log(lead.max()) - np.log(lead.min()))
    ll = np.convolve(np.pad(ll, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    end = V.INTRO + V.D.race
    E = [(0, 0.3), (V.INTRO, 0.3)] + [(at(y), 0.25 + 0.72 * v) for y, v in zip(years, ll)] + [(end, 0.97), (total, 0.97)]
    A = Anthem(total, E)
    A.compose(V.INTRO, end, at(1977) - 0.5 * V.PER_YEAR if V.D.secs(1977) > 0 else end,
              [c["T0"] for c in V.D.events if c.get("hit")], [V.INTRO + V.D.secs(y - 0.5) for y in V.D.leads])
    return A.mix()


if __name__ == "__main__":
    import video as V
    if "--career" in sys.argv or "--yearly" in sys.argv:
        V.configure("career" if "--career" in sys.argv else "annual")
    SC.write(track(), os.path.join(HERE, "out", f"{V.MODE['film']}_score.wav"))
    print("wrote", f"out/{V.MODE['film']}_score.wav")
