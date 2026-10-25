"""The traffic video's score, synthesized from scratch (no samples, no licensing issues).

C major at 96 BPM (Cmaj7 - Am7 - Fmaj7 - G), so every section, being a whole number of bars, starts on a
downbeat. An energy curve follows the story: a quiet hook, a pulsing 3 a.m. race, a long build as rush hour
fills the roads (a bell for each route that joins, a unison chime when the three times meet), calm cards, a
minor-key turn for Braess's paradox, and a resolution at the end. Instruments come from flights/score.py.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # noqa: E402
import look as L                                      # noqa: E402

SR = SC.SR
BPM = 96.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT                                        # 2.5 s
MAJOR = [([60, 64, 67, 71], 48), ([57, 60, 64, 67], 45), ([53, 57, 60, 64], 41), ([55, 59, 62, 67], 43)]   # Cmaj7 Am7 Fmaj7 G
MINOR = [([57, 60, 64, 67], 45), ([53, 57, 60, 65], 41), ([50, 53, 57, 62], 38), ([52, 56, 59, 64], 40)]   # Am7 F Dm E
rng = np.random.default_rng(5)


class Track(SC.Score):
    def bell(self, t0, m=84, vel=1.0, pan=0.0):
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        f = SC.mtof(m)
        sig = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.25)
               + 0.2 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.08)) * np.exp(-t / 0.8) * np.minimum(1, t / 0.002)
        self.add("fx", t0 * SR, sig * vel * 0.35, pan=pan)

    def compose_bars(self, plan):
        """plan(t) -> (energy, minor?, arp on?) for the bar starting at t"""
        n_bars = int(np.ceil(self.total / BAR))
        pats = [[0, 2, 4, 2, 1, 3, 4, 3, 0, 2, 5, 4, 2, 3, 1, 2], [0, 1, 3, 4, 2, 4, 5, 3, 0, 2, 3, 5, 4, 2, 1, 3]]
        for b in range(n_bars):
            tb = b * BAR
            e, minor, arp_on = plan(tb + 0.01)
            prog = MINOR if minor else MAJOR
            chord, root = prog[(b // 2) % 4]
            end = min(self.total, tb + BAR)
            if b % 2 == 0:
                self.pad_chord(tb, min(2 * BAR, self.total - tb), [root, chord[0], chord[1], chord[2], chord[3]], rel=2.4, gain=0.9)
            if e >= 0.45:
                for k in range(8):
                    t = tb + k * BEAT / 2
                    if t < end:
                        self.bass_note(t, BEAT / 2 * 0.85, root - 12 + (12 if (k in (3, 7) and e > 0.8) else 0), True)
            elif e >= 0.15:
                self.bass_note(tb, (end - tb) * 0.97, root - 12, False)
            for k in range(4):
                t = tb + k * BEAT
                if t >= end:
                    break
                if e >= 0.55:
                    self.kick(t, 0.85 + 0.15 * (k == 0))
                if e >= 0.8 and k in (1, 3):
                    self.clap(t, 0.8)
                if e >= 0.62:
                    self.hat(t + BEAT / 2, 0.75 + 0.2 * rng.random(), open_=(e >= 0.85 and k == 3))
            if arp_on and e >= 0.18:
                tones = [chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[3] + 12, chord[1] + 24, chord[2] + 24]
                pat = pats[(b // 4) % 2]
                step = 1 if e >= 0.7 else 2
                for k in range(0, 16, step):
                    t = tb + k * BEAT / 4
                    if t >= end:
                        break
                    vel = (0.55 + 0.45 * (k % 4 == 0)) * (0.8 + 0.2 * rng.random())
                    self.arp_note(t, tones[pat[k]], vel, bright=min(1.0, 0.25 + e))

    def ending(self, t0):
        hold = self.total - t0
        self.pad_chord(t0, hold - 1.0, [36, 48, 55, 59, 64, 67, 71], rel=3.5, att=0.08, gain=1.3, stem="final")
        self.pad_chord(t0, hold - 1.0, [72, 76, 79, 83], rel=3.5, att=1.2, gain=0.5, stem="final")
        self.bass_note(t0, hold - 0.5, 36, False)


def main_track(O, total):
    """O: section start times (seconds) from assemble.py"""
    import video as V
    E = [  # (time, energy)
        (0, 0.22), (O["hook"] + 18.5, 0.28), (O["hook"] + 19.5, 0.42),
        (O["light"], 0.4), (O["empty"], 0.5), (O["empty"] + V.Empty.GO, 0.66), (O["empty"] + 16.0, 0.66), (O["empty"] + 16.5, 0.42),
        (O["curve"], 0.22), (O["fill"], 0.3), (O["fill"] + 13.0, 0.5), (O["fill"] + 25.0, 0.64), (O["fill"] + 41.0, 0.92),
        (O["why"] - 0.1, 0.92), (O["why"], 0.3), (O["why"] + V.Why.MOVE0, 0.34), (O["why"] + V.Why.MOVE1, 0.5),
        (O["fine"] - 0.1, 0.5), (O["fine"], 0.2), (O["learn"], 0.42), (O["learn"] + V.Learn.R1, 0.78),
        (O["learn"] + V.Learn.R2 - 0.6, 0.78), (O["learn"] + V.Learn.R2, 0.4), (O["app"], 0.36), (O["app"] + 15.0, 0.5),
        (O["optimum"], 0.24), (O["myth"], 0.34), (O["myth"] + V.Myth.OPEN0, 0.48), (O["myth"] + V.Myth.SW0, 0.62),
        (O["myth"] + V.Myth.SW1, 0.84), (O["myth"] + V.Myth.SW1 + 7.0, 0.6), (O["myth"] + V.Myth.CHART0, 0.4), (O["outro"], 0.6), (O["outro"] + 15.0, 0.85), (total - 6.0, 0.85), (total - 5.9, 0.0), (total, 0.0)]
    E = sorted(E)
    impacts = [(O["empty"] + V.Empty.GO, 0.6), (O["fill"] + 41.0, 1.1), (O["myth"] + V.Myth.OPEN0, 0.8),
               (O["myth"] + V.Myth.SW1, 0.9), (O["outro"], 0.7)]
    risers = [(O["light"] + 11.0, O["empty"] + V.Empty.GO), (O["fill"] + 33.0, O["fill"] + 41.0),
              (O["myth"] + V.Myth.SW0, O["myth"] + V.Myth.SW1), (O["outro"] + 11.0, O["outro"] + 15.0)]
    T = Track(total, E, impacts, risers)
    minor_from, minor_to = O["myth"] + V.Myth.SW0, O["myth"] + V.Myth.CHART0     # the paradox, in a minor key

    def plan(t):
        e = float(np.interp(t, *zip(*E)))
        minor = minor_from <= t < minor_to
        arp = not (O["curve"] <= t < O["fill"] or O["fine"] <= t < O["learn"] or O["optimum"] <= t < O["myth"])
        return e, minor, arp
    T.compose_bars(plan)
    N = L.numbers()
    for i, m in enumerate((79, 83, 86)):                                     # each racer finishing at 3 a.m.
        T.bell(O["empty"] + V.Empty.GO + V.Empty.RACE * N["t0"][i] / 40.0, m, 0.8, pan=(i - 1) * 0.4)
    T.bell(O["fill"] + 13.0, 79, 1.0, -0.3)                                   # B joins
    T.bell(O["fill"] + 25.0, 83, 1.0, 0.3)                                    # C joins
    for m in (72, 84, 96):                                                    # one ETA: the three in unison
        T.bell(O["fill"] + 41.0, m, 1.0)
    for m in (72, 84):                                                        # the gap closes: equal again
        T.bell(O["why"] + V.Why.MOVE1, m, 0.7)
    T.bell(O["learn"] + V.Learn.R2 + V.Learn.SPD * (N["learn"]["calm"]["settle_day"] + 1), 84, 0.9)
    T.ending(total - 6.0)
    for t, s in T.impacts:
        T.impact(t, s)
    for ta, tb in T.risers:
        T.riser(ta, tb)
    SC.BEAT = BEAT                                                            # the arpeggio delay follows our tempo
    return T.mix()
