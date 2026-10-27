"""The race's score: an original heroic march in the gridiron-film manner, for the synthesized orchestra of
orchestra.py (no samples, no licensing).

D minor, 85.7 BPM (a beat is 0.7 s, a bar 2.8 s; every season starts on a beat), the theme in two 8-bar halves:
  A  Dm  Bb  F  C | Dm  Gm  A  A
  B  Bb  F  Gm  A | Dm  C   A  Dm
* the title: a timpani roll and cymbal swell into a full-brass fanfare on the second bar;
* the explainer: a held, quiet underscore for reading (low strings, soft horns, a timpani stroke at each scene), and a
  snare and timpani roll into the race;
* the race: low strings driving eighths, the march on the snare, timpani on the downbeats, the theme on the horns;
  as a season puts more on the board, bass drum, trombone pads, upper strings and trumpets join; a snare roll ends
  each 8-bar phrase into a cymbal crash; the seasons that change nothing thin it to strings and snare;
* each new No. 1: a brass stab, crash, bass drum and timpani on the nearest beat;
* the key rises a whole step at the bar after the record season takes No. 1 for good;
* the summary: a timpani roll into the full orchestra on the major chord, held to the end.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orchestra as O                                 # noqa: E402

BEAT = 0.7
BAR = 4 * BEAT
CHORD = {"Dm": (38, (0, 3, 7)), "Bb": (34, (0, 4, 7)), "F": (41, (0, 4, 7)), "C": (36, (0, 4, 7)), "Gm": (43, (0, 3, 7)),
         "A": (45, (0, 4, 7)), "D": (38, (0, 4, 7))}
PROG = {"A": ("Dm", "Bb", "F", "C", "Dm", "Gm", "A", "A"), "B": ("Bb", "F", "Gm", "A", "Dm", "C", "A", "Dm")}
# the theme: (beat in the bar, MIDI note, beats held), bar by bar
THEME = {
    "A": [[(0, 62, 1.5), (1.5, 64, 0.5), (2, 65, 1), (3, 69, 1)], [(0, 74, 2), (2, 72, 1), (3, 70, 1)],
          [(0, 69, 1.5), (1.5, 67, 0.5), (2, 65, 1), (3, 72, 1)], [(0, 67, 3)],
          [(0, 62, 1.5), (1.5, 64, 0.5), (2, 65, 1), (3, 69, 1)], [(0, 70, 1), (1, 74, 1), (2, 79, 2)],
          [(0, 73, 1.5), (1.5, 74, 0.5), (2, 76, 2)], [(0, 69, 3)]],
    "B": [[(0, 77, 2), (2, 74, 1), (3, 70, 1)], [(0, 72, 2), (2, 69, 1), (3, 65, 1)],
          [(0, 67, 1), (1, 70, 1), (2, 74, 1), (3, 79, 1)], [(0, 76, 2), (2, 73, 2)],
          [(0, 74, 1.5), (1.5, 72, 0.5), (2, 69, 1), (3, 65, 1)], [(0, 67, 1), (1, 72, 1), (2, 76, 2)],
          [(0, 74, 2), (2, 73, 2)], [(0, 74, 4)]],
}
EXPLAIN = ("Dm", "Bb", "F", "C", "Dm", "Gm", "A", "A", "Dm", "Bb", "F", "C")      # the theme's harmony; C leads home to Dm


def tones(name, k=0):
    root, iv = CHORD[name]
    return root + k, [root + k + i for i in iv]


def snare_bar(o, tb, e, last):
    """the march on the snare: eighths when quiet, the full sixteenth pattern when busy; a roll closes a phrase"""
    pat = [1.0, 0, 0.45, 0.35, 0.8, 0, 0.45, 0, 1.0, 0, 0.45, 0.35, 0.8, 0.3, 0.55, 0.3]
    for j in range(16):
        if last and j >= 8:
            break
        v = pat[j]
        if v == 0 or (e < 0.55 and j % 2) or (e < 0.3 and j % 4):
            continue
        o.snare(tb + j * BEAT / 4, v * (0.45 + 0.5 * e) * O.rng.uniform(0.9, 1.0))
    if last:
        o.snare_roll(tb + 2 * BEAT, tb + BAR - 0.02, 0.3, 0.75 + 0.2 * e)


def track():
    import video as V
    V.setup()
    D = V.D
    total = V.total()
    o = O.Orchestra(total)
    years = list(range(V.FIRST, V.LAST + 1))
    gain = np.array([D.board_gain(y) for y in years])
    level = dict(zip(years, np.log1p(gain) / np.log1p(gain.max())))
    energy = lambda t: 0.3 + 0.65 * level[D.at(t)[0]]                     # the race's energy at film time t
    race0, end = V.INTRO, D.race_end + V.HOLD                               # bar 14 and the summary's bar
    nb = int(round(total / BAR))
    records = [c["from"] for c in V.CAPS if c.get("tag") == "NEW No. 1"]
    lift_bar = int(np.ceil(records[-1] / BAR)) if records else nb + 1
    key = lambda b: 2 if b >= lift_bar else 0

    # ---------------------------------------------------------------- the title: roll, swell, fanfare
    o.timp_roll(0.05, BAR - 0.05, 38, 0.12, 0.85)
    o.swell(0.0, BAR, 0.7)
    for j in range(16):
        o.strings(j * BEAT / 4, BEAT / 4 * 0.9, 38, 0.25 + 0.5 * j / 16, marc=True)
    t = BAR
    o.crash(t, 0.9)
    o.bass_drum(t, 1.0)
    o.timpani(t, 38, 1.0)
    for m in (62, 65, 69, 74):
        o.brass(t, BEAT * 0.55, m, 0.95, "trumpet")
    for m in (50, 57, 62, 65):
        o.brass(t, BEAT * 0.55, m, 0.95, "trombone")
    o.brass(t, BEAT * 0.9, 38, 0.95, "tuba")
    for beat, m, d in ((1, 69, 1 / 3), (4 / 3, 69, 1 / 3), (5 / 3, 69, 1 / 3), (2, 74, 1), (3, 77, 1), (4, 81, 3.5)):
        o.brass(t + beat * BEAT, d * BEAT * 0.92, m, 0.9, "trumpet")
    for beat, m, d in ((2, 62, 1), (3, 65, 1), (4, 69, 3.5)):
        o.brass(t + beat * BEAT, d * BEAT * 0.95, m, 0.8, "horn")
    for m in (50, 57, 65):                                               # under the fanfare: low brass and strings
        o.brass(t + BEAT, 3 * BEAT * 0.95, m, 0.75, "trombone")
    o.brass(t + BEAT, 3 * BEAT * 0.95, 38, 0.8, "tuba")
    for j in range(2, 8):
        o.strings(t + j * BEAT / 2, BEAT / 2 * 0.8, 38 + (12 if j in (3, 7) else 0), 0.8, marc=True, stem="str_lo")
        o.strings(t + j * BEAT / 2, BEAT / 2 * 0.8, 26 + (12 if j in (3, 7) else 0), 0.75, marc=True, stem="str_lo")
    o.snare_roll(t + 3 * BEAT, t + 4 * BEAT - 0.02, 0.3, 0.8, 0.042)
    o.timpani(t + 2 * BEAT, 45, 0.8)
    o.timpani(t + 4 * BEAT, 38, 0.9)
    o.crash(t + 4 * BEAT, 0.7)
    o.strings(2 * BAR, BAR * 1.2, 38, 0.6)
    o.strings(2 * BAR, BAR * 1.2, 50, 0.5)

    # ---------------------------------------------------------------- the explainer: a held underscore, then the roll
    for k, name in enumerate(EXPLAIN):
        b = 2 + k
        tb = b * BAR
        root, ch = tones(name)
        o.strings(tb, BAR * 0.98, root, 0.42)
        o.strings(tb, BAR * 0.98, root + 12, 0.35)
        if k >= 2:
            for m in ch:
                o.strings(tb, BAR * 0.98, m + 24, 0.22)
        if 8 <= k <= 11:                                                 # the theme's opening, softly, on a horn
            for beat, m, d in THEME["A"][k - 8]:
                o.brass(tb + beat * BEAT, d * BEAT * 0.95, m, 0.42, "horn")
        if 8 <= k <= 10:
            for j in range(4):
                o.snare(tb + j * BEAT, 0.22 + 0.05 * (k - 8))
    for t in V.scene_starts()[1:]:
        o.timpani(t, 38, 0.55)
    tb = race0 - BAR                                                     # into the race
    o.snare_roll(tb, race0 - 0.02, 0.15, 0.95, 0.042)
    o.timp_roll(tb + BAR / 2, race0 - 0.03, 45, 0.2, 0.85)
    o.swell(tb, race0, 0.8)

    # ---------------------------------------------------------------- the race: the march
    b0, b1 = int(round(race0 / BAR)), int(round(end / BAR))
    for b in range(b0, b1):
        tb = b * BAR
        e = energy(tb + 0.01)
        ph = (b - b0) // 8
        theme = "A" if ph % 2 == 0 else "B"
        k8 = (b - b0) % 8
        name = PROG[theme][k8]
        k = key(b)
        root, ch = tones(name, k)
        last = k8 == 7 or b == b1 - 1
        if k8 == 0 and (e > 0.45 or b == b0):
            o.crash(tb, 0.55 + 0.4 * e)
        # low strings: driving eighths (quarters when quiet), the octave on the and of two and four
        step = 1 if e >= 0.4 else 2
        for j in range(0, 8, step):
            m = root + (12 if j in (3, 7) else 0)
            o.strings(tb + j * BEAT / 2, BEAT / 2 * step * 0.8, m, 0.5 + 0.4 * e, marc=True, stem="str_lo")
            o.strings(tb + j * BEAT / 2, BEAT / 2 * step * 0.8, m - 12, 0.45 + 0.4 * e, marc=True, stem="str_lo")
        snare_bar(o, tb, e, last)
        o.timpani(tb, root, 0.55 + 0.4 * e)
        if e > 0.5:
            o.timpani(tb + 2 * BEAT, root + 7 - 12 if root + 7 > 50 else root + 7, 0.45 + 0.4 * e)
        if last:
            o.timp_roll(tb + 3 * BEAT, tb + BAR - 0.03, root, 0.3, 0.6 + 0.3 * e)
        if e > 0.58:
            o.bass_drum(tb, 0.7 + 0.25 * e)
            o.bass_drum(tb + 2 * BEAT, 0.55 + 0.25 * e)
        # the theme on the horns; trumpets double it an octave up when the board is busiest
        if e >= 0.36:
            for beat, m, d in THEME[theme][k8]:
                o.brass(tb + beat * BEAT, d * BEAT * 0.93, m + k, 0.5 + 0.45 * e, "horn")
                if e >= 0.74:
                    o.brass(tb + beat * BEAT, d * BEAT * 0.9, m + k + 12, 0.45 + 0.5 * e, "trumpet")
        # trombones and tuba: the chord on beats one and three
        if e >= 0.5:
            for beat in (0, 2):
                for m in (root + 12, ch[2] + 12, ch[1] + 24):
                    o.brass(tb + beat * BEAT, BEAT * 1.6, m, 0.4 + 0.45 * e, "trombone")
                o.brass(tb + beat * BEAT, BEAT * 1.6, root, 0.45 + 0.4 * e, "tuba")
        # upper strings hold the chord
        if e >= 0.55:
            for m in ch:
                o.strings(tb, BAR * 0.98, m + 24, 0.3 + 0.35 * e)

    # ---------------------------------------------------------------- each new No. 1: a stab on the nearest beat
    for t in records:
        tq = round(t / (BEAT / 2)) * BEAT / 2
        b = int(tq // BAR)
        root, ch = tones(PROG["A" if ((b - b0) // 8) % 2 == 0 else "B"][(b - b0) % 8], key(b))
        o.crash(tq, 1.0)
        o.bass_drum(tq, 1.0)
        o.timpani(tq, root, 1.0)
        for m in ch:
            o.brass(tq, BEAT * 0.5, m + 36, 1.0, "trumpet")
            o.brass(tq, BEAT * 0.5, m + 12, 1.0, "trombone")

    # ---------------------------------------------------------------- the summary: the major chord, held
    k = key(b1)
    root = 38 + k
    o.timp_roll(end - BAR / 2, end - 0.03, root, 0.3, 1.0)
    o.snare_roll(end - BAR / 2, end - 0.02, 0.3, 1.0, 0.04)
    hold = total - end - 1.6
    o.crash(end, 1.0)
    o.bass_drum(end, 1.0)
    o.timp_roll(end + 0.05, end + hold, root, 0.9, 0.25)
    for m in (root + 36, root + 40, root + 43, root + 48):
        o.brass(end, hold, m, 0.95, "trumpet")
    for m in (root + 24, root + 28, root + 31, root + 36):
        o.brass(end, hold, m, 0.9, "horn")
    for m in (root + 12, root + 19, root + 28):
        o.brass(end, hold, m, 0.9, "trombone")
    o.brass(end, hold, root, 0.9, "tuba")
    for m in (root + 36, root + 40, root + 43, root + 48):
        o.strings(end, hold, m, 0.75)
    o.strings(end, hold, root, 0.8)
    o.strings(end, hold, root + 12, 0.8)
    # the master dynamics: the fanfare 2 dB up, the explainer about 7.5 dB under the race for reading, a swell back into the
    # race, the race full (its arrangement carries its ups and downs), the finale full
    quiet = 0.42
    o.auto = [(0.0, 0.55), (BAR, 1.25), (2 * BAR, 1.15), (2 * BAR + 2.0, quiet), (race0 - BAR, quiet), (race0, 1.0),
              (total, 1.0)]
    return o.mix()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    O.SC.write(track(), os.path.join(HERE, "out", "score.wav"))
