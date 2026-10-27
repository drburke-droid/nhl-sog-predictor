"""The race's score, synthesized from scratch with the instruments of ../flights/score.py (no samples, no licensing).

85.7 BPM: a beat is 0.7 s and a year 1.4 s, so every year turns on a beat (a bar is two years). C major
(Cmaj7 - Am7 - Fmaj7 - G), a whole step up when free agency arrives (1977); the energy follows the leader's pay (on a
log scale), from a sparse pulse in 1913 to the full groove of the 2020s; a bell at each milestone caption; a resolving
chord over the final standings.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # noqa: E402

SR = SC.SR
BPM = 60 / 0.7
BEAT = 60.0 / BPM
BAR = 4 * BEAT
MAJOR = [([60, 64, 67, 71], 48), ([57, 60, 64, 67], 45), ([53, 57, 60, 64], 41), ([55, 59, 62, 67], 43)]
rng = np.random.default_rng(7)


class Track(SC.Score):
    def bell(self, t0, m=84, vel=1.0, pan=0.0):
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        f = SC.mtof(m)
        sig = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.25)
               + 0.2 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.08)) * np.exp(-t / 0.8) * np.minimum(1, t / 0.002)
        self.add("fx", t0 * SR, sig * vel * 0.32, pan=pan)

    def compose_bars(self, up_at):
        pats = [[0, 2, 4, 2, 1, 3, 4, 3, 0, 2, 5, 4, 2, 3, 1, 2], [0, 1, 3, 4, 2, 4, 5, 3, 0, 2, 3, 5, 4, 2, 1, 3]]
        for b in range(int(np.ceil(self.total / BAR))):
            tb = b * BAR
            e = self.e(tb + 0.01)
            k = 2 if tb >= up_at - 0.05 else 0
            chord, root = MAJOR[(b // 2) % 4]
            chord, root = [c + k for c in chord], root + k
            end = min(self.total, tb + BAR)
            if b % 2 == 0:
                self.pad_chord(tb, min(2 * BAR, self.total - tb), [root, *chord], rel=2.4, gain=0.85)
            if e >= 0.45:
                for j in range(8):
                    t = tb + j * BEAT / 2
                    if t < end:
                        self.bass_note(t, BEAT / 2 * 0.85, root - 12 + (12 if (j in (3, 7) and e > 0.8) else 0), True)
            elif e >= 0.12:
                self.bass_note(tb, (end - tb) * 0.97, root - 12, False)
            for j in range(4):
                t = tb + j * BEAT
                if t >= end:
                    break
                if e >= 0.5:
                    self.kick(t, 0.85 + 0.15 * (j == 0))
                if e >= 0.78 and j in (1, 3):
                    self.clap(t, 0.8)
                if e >= 0.6:
                    self.hat(t + BEAT / 2, 0.75 + 0.2 * rng.random(), open_=(e >= 0.86 and j == 3))
            if e >= 0.2:
                tones = [chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[3] + 12, chord[1] + 24, chord[2] + 24]
                pat = pats[(b // 4) % 2]
                for j in range(0, 16, 1 if e >= 0.68 else 2):
                    t = tb + j * BEAT / 4
                    if t >= end:
                        break
                    self.arp_note(t, tones[pat[j]], (0.55 + 0.45 * (j % 4 == 0)) * (0.8 + 0.2 * rng.random()), bright=min(1.0, 0.25 + e))

    def ending(self, t0):
        hold = self.total - t0
        self.pad_chord(t0, hold - 1.0, [38, 50, 57, 61, 66, 69, 73], rel=3.5, att=0.08, gain=1.25, stem="final")
        self.pad_chord(t0, hold - 1.0, [74, 78, 81, 85], rel=3.5, att=1.2, gain=0.5, stem="final")
        self.bass_note(t0, hold - 0.5, 38, False)


def track():
    import video as V
    V.setup()
    total = V.total()
    at = lambda year: V.INTRO + (year - V.FIRST) * V.PER_YEAR
    # energy: the leader's pay on a log scale, smoothed
    years = np.arange(V.FIRST, V.LAST + 1)
    lead = np.array([V.D.by[y][V.D.order[y][0]]["real"] if V.D.order.get(y) else np.nan for y in years])
    lead = np.where(np.isnan(lead), np.nanmin(lead), lead)
    ll = (np.log(lead) - np.log(lead.min())) / (np.log(lead.max()) - np.log(lead.min()))
    ll = np.convolve(np.pad(ll, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    E = [(0, 0.1), (V.INTRO - 0.5, 0.16)] + [(at(y), 0.22 + 0.68 * v) for y, v in zip(years, ll)]
    E += [(total - V.OUTRO + 0.5, 0.9), (total - V.OUTRO + 1.4, 0.0), (total, 0.0)]
    caps = V.D.events
    T = Track(total, E, impacts=[(V.INTRO, 0.5)] + [(V.INTRO + (c["from"] - V.FIRST) * V.PER_YEAR, 0.35) for c in caps if c.get("hit")],
              risers=[(total - V.OUTRO - 2 * BAR, total - V.OUTRO + 0.05)])
    T.compose_bars(at(1977))
    for i, c in enumerate(caps):
        T.bell(V.INTRO + (c["from"] - V.FIRST) * V.PER_YEAR, 79 + (5 if i % 2 else 0), 0.8, -0.3 if i % 2 else 0.3)
    T.ending(total - V.OUTRO + 1.4)
    for t, s in T.impacts:
        T.impact(t, s)
    for ta, tb in T.risers:
        T.riser(ta, tb)
    SC.BEAT = BEAT
    return T.mix()


if __name__ == "__main__":
    SC.write(track(), os.path.join(HERE, "out", "score.wav"))
    print("wrote out/score.wav")
