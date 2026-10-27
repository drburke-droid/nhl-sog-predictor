"""The race's score, synthesized from scratch with the instruments of ../flights/score.py (no samples, no licensing).

85.7 BPM: a beat is 0.7 s, and every season starts on a beat (video.py gives each season 2 to 12 beats). A minor
(Am7 - Fmaj7 - Cmaj7 - G), a whole step up once the record season takes No. 1 for good; the energy follows how much
value each season puts on the board (on a log scale), so the years when the board fills and the record years drive and
the seasons that change nothing thin out; a bell at each caption, an impact at each new No. 1; a resolving chord over
the closing summary.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # noqa: E402

SR = SC.SR
BEAT = 0.7
BAR = 4 * BEAT
PROG = [([57, 60, 64, 67], 45), ([53, 57, 60, 64], 41), ([60, 64, 67, 71], 48), ([55, 59, 62, 67], 43)]
rng = np.random.default_rng(11)


class Track(SC.Score):
    def bell(self, t0, m=84, vel=1.0, pan=0.0):
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        f = SC.mtof(m)
        sig = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.25)
               + 0.2 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.08)) * np.exp(-t / 0.8) * np.minimum(1, t / 0.002)
        self.add("fx", t0 * SR, sig * vel * 0.32, pan=pan)

    def compose_bars(self, up_at, stop_at):
        pats = [[0, 2, 4, 2, 1, 3, 4, 3, 0, 2, 5, 4, 2, 3, 1, 2], [0, 1, 3, 4, 2, 4, 5, 3, 0, 2, 3, 5, 4, 2, 1, 3]]
        for b in range(int(np.ceil(stop_at / BAR))):
            tb = b * BAR
            e = self.e(tb + 0.01)
            k = 2 if (up_at is not None and tb >= up_at - 0.05) else 0
            chord, root = PROG[(b // 2) % 4]
            chord, root = [c + k for c in chord], root + k
            end = min(stop_at, tb + BAR)
            if b % 2 == 0:
                self.pad_chord(tb, min(2 * BAR, stop_at - tb), [root, *chord], rel=2.4, gain=0.85)
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
                if e >= 0.75 and j in (1, 3):
                    self.clap(t, 0.8)
                if e >= 0.58:
                    self.hat(t + BEAT / 2, 0.75 + 0.2 * rng.random(), open_=(e >= 0.85 and j == 3))
            if e >= 0.2:
                tones = [chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[3] + 12, chord[1] + 24, chord[2] + 24]
                pat = pats[(b // 4) % 2]
                for j in range(0, 16, 1 if e >= 0.66 else 2):
                    t = tb + j * BEAT / 4
                    if t >= end:
                        break
                    self.arp_note(t, tones[pat[j]], (0.55 + 0.45 * (j % 4 == 0)) * (0.8 + 0.2 * rng.random()), bright=min(1.0, 0.25 + e))

    def ending(self, t0, k):
        hold = self.total - t0
        self.pad_chord(t0, hold - 1.0, [x + k for x in (33, 45, 52, 57, 60, 64, 69)], rel=3.5, att=0.08, gain=1.25, stem="final")
        self.pad_chord(t0, hold - 1.0, [x + k for x in (69, 72, 76, 79)], rel=3.5, att=1.2, gain=0.5, stem="final")
        self.bass_note(t0, hold - 0.5, 33 + k, False)


def track():
    import video as V
    V.setup()
    D = V.D
    total = V.total()
    years = list(range(V.FIRST, V.LAST + 1))
    gain = np.array([D.board_gain(y) for y in years])
    ll = np.log1p(gain) / np.log1p(gain.max())
    E = [(0, 0.1), (V.INTRO - 0.7, 0.18)]
    for y, v in zip(years, ll):
        E += [(D.t0[y] + 0.01, 0.34 + 0.56 * v), (D.t0[y] + D.beats[y] * BEAT - 0.01, 0.34 + 0.56 * v)]
    E += [(D.race_end - 0.7, 0.9), (D.race_end + V.HOLD, 0.0), (total, 0.0)]
    # the lift: when the record season takes No. 1 for good
    lead = [(y, w, int(np.argmax(D.C[y][:, w]))) for y in years for w in range(D.weeks[y] + 1) if D.C[y][:, w].max() > 0]
    k = len(lead) - 1
    while k > 0 and lead[k - 1][2] == lead[-1][2]:
        k -= 1
    up = D.film_time(lead[k][0], lead[k][1]) if k > 0 else None
    caps = V.CAPS
    big = [c["from"] for c in caps if "for No. 1" in c["text"][0]]
    T = Track(total, E, impacts=[(V.INTRO, 0.5)] + [(t, 0.4) for t in big],
              risers=[(D.race_end - 2 * BAR, D.race_end + 0.05)])
    T.compose_bars(up, D.race_end + V.HOLD)
    for i, c in enumerate(caps):
        T.bell(c["from"], 79 + (5 if i % 2 else 0) + (2 if up is not None and c["from"] >= up else 0), 0.8, -0.3 if i % 2 else 0.3)
    T.ending(D.race_end + V.HOLD, 2 if up is not None else 0)
    for t, s in T.impacts:
        T.impact(t, s)
    for ta, tb in T.risers:
        T.riser(ta, tb)
    SC.BEAT = BEAT
    return T.mix()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    SC.write(track(), os.path.join(HERE, "out", "score.wav"))
