"""The rebuild video's score, synthesized from scratch (no samples, no licensing issues), with the instruments of
../flights/score.py, as in the traffic video.

C major at 96 BPM (Cmaj7 - Am7 - Fmaj7 - G), so every section, being a whole number of bars, starts on a downbeat. The
energy follows the story: a quiet hook over the season grid, a pulse through the tests, a minor-key turn while the
tear-down teams sink, bells as the verdicts land, and a resolution at the end.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "flights"))
import score as SC                                    # noqa: E402

SR = SC.SR
BPM = 96.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT                                        # 2.5 s
MAJOR = [([60, 64, 67, 71], 48), ([57, 60, 64, 67], 45), ([53, 57, 60, 64], 41), ([55, 59, 62, 67], 43)]   # Cmaj7 Am7 Fmaj7 G
MINOR = [([57, 60, 64, 67], 45), ([53, 57, 60, 65], 41), ([50, 53, 57, 62], 38), ([52, 56, 59, 64], 40)]   # Am7 F Dm E
rng = np.random.default_rng(11)


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
        pats = [[0, 2, 4, 2, 1, 3, 4, 3, 0, 2, 5, 4, 2, 3, 1, 2], [0, 1, 3, 4, 2, 4, 5, 3, 0, 2, 3, 5, 4, 2, 1, 3]]
        for b in range(int(np.ceil(self.total / BAR))):
            tb = b * BAR
            e, minor, arp_on = plan(tb + 0.01)
            chord, root = (MINOR if minor else MAJOR)[(b // 2) % 4]
            end = min(self.total, tb + BAR)
            if b % 2 == 0:
                self.pad_chord(tb, min(2 * BAR, self.total - tb), [root, *chord], rel=2.4, gain=0.9)
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
                for k in range(0, 16, 1 if e >= 0.7 else 2):
                    t = tb + k * BEAT / 4
                    if t >= end:
                        break
                    self.arp_note(t, tones[pat[k]], (0.55 + 0.45 * (k % 4 == 0)) * (0.8 + 0.2 * rng.random()), bright=min(1.0, 0.25 + e))

    def ending(self, t0):
        hold = self.total - t0
        self.pad_chord(t0, hold - 1.0, [36, 48, 55, 59, 64, 67, 71], rel=3.5, att=0.08, gain=1.3, stem="final")
        self.pad_chord(t0, hold - 1.0, [72, 76, 79, 83], rel=3.5, att=1.2, gain=0.5, stem="final")
        self.bass_note(t0, hold - 0.5, 36, False)


def main_track(O, total):
    """O: section start times (seconds) from assemble.py"""
    import video as V
    td = O["teardown"]
    E = sorted([(0, 0.16), (O["hook"] + 6.0, 0.26), (O["theory"], 0.28), (O["stories"], 0.38), (O["richer"], 0.48),
                (O["shifted"], 0.4), (O["winnow"], 0.46), (td, 0.34), (td + V.Teardown.T_REST, 0.42), (td + 14.4, 0.7),
                (td + 21.0, 0.5), (O["titles"], 0.5), (O["titles"] + V.Titles.SPLIT, 0.56), (O["verdict"], 0.6),
                (O["verdict"] + 12.0, 0.74), (O["outro"], 0.82), (total - 6.0, 0.82), (total - 5.9, 0.0), (total, 0.0)])
    impacts = [(O["richer"] + 1.0, 0.5), (td + V.Teardown.T_REST, 0.7), (O["outro"], 0.8)]
    risers = [(td + V.Teardown.T_REST + 0.5, td + 14.4), (O["verdict"] + 10.0, O["outro"])]
    T = Track(total, E, impacts, risers)
    minor_from, minor_to = td + V.Teardown.T_REST, td + 21.0          # the tear-downs sinking, in a minor key

    def plan(t):
        e = float(np.interp(t, *zip(*E)))
        arp = not (O["theory"] <= t < O["stories"])
        return e, minor_from <= t < minor_to, arp
    T.compose_bars(plan)
    T.bell(O["theory"] + 1.0, 79, 0.8, -0.3)                                # claim one
    T.bell(O["theory"] + 6.5, 84, 0.8, 0.3)                                 # claim two
    T.bell(O["shifted"] + 12.4, 76, 0.7)                                    # the effect is gone
    for m in (72, 84):                                                      # the same peak
        T.bell(td + 14.4, m, 0.8)
    T.bell(O["verdict"] + 1.0, 79, 0.9, -0.3)                               # half right
    T.bell(O["verdict"] + 6.5, 76, 0.9, 0.3)                                # mostly no
    T.ending(total - 6.0)
    for t, s in T.impacts:
        T.impact(t, s)
    for ta, tb in T.risers:
        T.riser(ta, tb)
    SC.BEAT = BEAT                                                          # the arpeggio delay follows our tempo
    return T.mix()
