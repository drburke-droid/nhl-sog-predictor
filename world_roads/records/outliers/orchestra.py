"""The score: an original heroic orchestral theme, written as MIDI timed to the film's events and played by
FluidSynth with the FluidR3 General MIDI soundfont (MIT licence) in a concert-hall reverb.

    python orchestra.py            # writes out/score.mid and out/score.wav for the timeline in assemble.py

Shape: the hook builds from low strings to a full-orchestra hit as the ten faces appear (17.4 s) and states the
theme under the title (21.8 s). Each countdown card opens on a hit under its rank number, drives with string
ostinato while the stacks build, crescendos into a hit as the score lands (18.2 s), then plays the theme; energy and
key rise towards #1. The explainer and the women's throws are calm. The close builds to the full theme at "Not a
ranking of the greatest athletes".
"""
import os, subprocess, sys, math
import numpy as np
import mido

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SR = 48000

# channel: (program, volume, reverb send)
CH = {"str": (0, 48, 100, 70), "ost": (1, 40, 92, 60), "cel": (2, 42, 96, 55), "cb": (3, 43, 92, 55),
      "hrn": (4, 60, 104, 75), "tpt": (5, 56, 92, 65), "tbn": (6, 57, 92, 65), "tba": (7, 58, 84, 55),
      "tmp": (8, 47, 112, 60), "drm": (9, 0, 100, 60), "chr": (10, 52, 78, 90), "hrp": (11, 46, 86, 80),
      "glk": (12, 9, 66, 90), "rev": (13, 119, 92, 70), "tko": (14, 116, 108, 60), "pno": (15, 0, 70, 70)}
CRASH, CRASH2, KICK, SNARE, TOM_L, TOM_M, RIDE = 49, 57, 35, 38, 41, 45, 51

# chords as semitones above the key's tonic: I, ii, iii, IV, V, vi, plus V7 and the borrowed bVI, bVII
CHORD = {"I": (0, 4, 7), "ii": (2, 5, 9), "iii": (4, 7, 11), "IV": (5, 9, 12), "V": (7, 11, 14), "vi": (9, 12, 16),
         "V7": (7, 11, 14, 17), "bVI": (8, 12, 15), "bVII": (10, 14, 17), "i": (0, 3, 7), "iv": (5, 8, 12),
         "VI": (8, 12, 15), "III": (3, 7, 10), "VII": (10, 14, 17), "v": (7, 10, 14)}
# the theme: (semitones from the tonic an octave above middle D, beats), two four-bar halves, and its chords
THEME = [[(0, 1.5), (-5, .5), (0, 1), (2, 1)], [(4, 2), (2, 1), (0, 1)], [(-3, 1.5), (-1, .5), (0, 1), (2, 1)], [(7, 4)],
         [(5, 1.5), (4, .5), (2, 1), (0, 1)], [(2, 1), (4, 1), (5, 1), (7, 1)], [(9, 1.5), (7, .5), (5, 1), (4, 1)],
         [(2, 2), (0, 2)]]
THEME_CH = [["I"], ["vi"], ["IV"], ["V"], ["ii"], ["V7"], ["IV"], ["V", "I"]]
FANFARE = [(0, .5), (0, .25), (0, .25), (7, 1), (4, .5), (7, .5), (12, 2)]


class Score:
    def __init__(self):
        self.ev = []                                              # (time s, channel, kind, a, b, note id)
        self.nid = 0
        self.dropped = []

    def note(self, ch, t, midi, dur, vel):
        c = CH[ch][0]
        vel = int(max(1, min(127, vel)))
        midi = int(midi)
        if not 20 <= midi <= 108:
            self.dropped.append((ch, round(t, 2), midi))
            return
        self.nid += 1
        self.ev.append((t, c, "on", midi, vel, self.nid))
        self.ev.append((t + max(0.03, dur - 0.012), c, "off", midi, 0, self.nid))

    def cc(self, ch, t, num, val):
        self.ev.append((t, CH[ch][0], "cc", num, int(max(0, min(127, val))), None))

    def drum(self, t, key, vel, dur=0.3):
        vel = int(max(1, min(127, vel)))
        self.nid += 1
        self.ev.append((t, 9, "on", key, vel, self.nid))
        self.ev.append((t + dur, 9, "off", key, 0, self.nid))

    def swell(self, ch, t0, t1, v0, v1, step=0.08):
        n = max(1, int((t1 - t0) / step))
        for i in range(n + 1):
            self.cc(ch, t0 + (t1 - t0) * i / n, 11, v0 + (v1 - v0) * i / n)

    def save(self, path):
        mf = mido.MidiFile(type=1, ticks_per_beat=960)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))  # 60 bpm: one beat = one second
        mf.tracks.append(meta)
        by = {}
        for name, (c, prog, vol, rev) in CH.items():
            tr = mido.MidiTrack()
            if c != 9:
                tr.append(mido.Message("program_change", channel=c, program=prog, time=0))
            tr.append(mido.Message("control_change", channel=c, control=7, value=vol, time=0))
            tr.append(mido.Message("control_change", channel=c, control=91, value=rev, time=0))
            tr.append(mido.Message("control_change", channel=c, control=11, value=100, time=0))
            by[c] = (tr, [])
        for e in self.ev:
            by[e[1]][1].append(e)
        order = {"off": 0, "cc": 1, "on": 2}
        for c, (tr, evs) in sorted(by.items()):
            evs = sorted(((int(round(max(0.0, e[0]) * 960)),) + e[1:] for e in evs), key=lambda e: (e[0], order[e[2]]))
            last = 0
            owner = {}                                            # pitch -> the note currently sounding it
            for tick, ch, kind, a, b, nid in evs:
                if kind == "off" and owner.get(a) != nid:
                    continue                                      # stale: a later note re-struck this pitch
                d = tick - last
                last = tick
                if kind == "on":
                    if a in owner:                                # re-strike: end the sounding note first
                        tr.append(mido.Message("note_off", channel=ch, note=a, velocity=0, time=d))
                        d = 0
                    owner[a] = nid
                    tr.append(mido.Message("note_on", channel=ch, note=a, velocity=b, time=d))
                elif kind == "off":
                    owner.pop(a, None)
                    tr.append(mido.Message("note_off", channel=ch, note=a, velocity=0, time=d))
                else:
                    tr.append(mido.Message("control_change", channel=ch, control=a, value=b, time=d))
            mf.tracks.append(tr)
        mf.save(path)


# ------------------------------------------------------------------ building blocks
def voices(key, chord, lo):
    """chord tones (semitones above key) placed at or above MIDI note lo"""
    out = []
    for s in CHORD[chord]:
        n = key + s
        while n < lo:
            n += 12
        while n >= lo + 12:
            n -= 12
        out.append(n)
    return sorted(out)


def pad(S, key, chord, t, dur, e, strings=True, horns=True, choir=False, low=True):
    if strings:
        for n in voices(key, chord, 62) + [voices(key, chord, 74)[-1]]:
            S.note("str", t, n, dur, 52 + 50 * e)
    if horns:
        for n in voices(key, chord, 55)[:3]:
            S.note("hrn", t, n, dur, 45 + 55 * e)
    if choir:
        for n in voices(key, chord, 60):
            S.note("chr", t, n, dur, 50 + 50 * e)
    if low:
        root = key + CHORD[chord][0]
        S.note("cel", t, (root % 12) + 36, dur, 55 + 50 * e)
        S.note("cb", t, (root % 12) + 24 + (12 if (root % 12) + 24 < 28 else 0), dur, 50 + 50 * e)


def ostinato(S, key, chord, t, bar, e, beats=4):
    """driving eighth notes in the strings: root, fifth, octave, fifth"""
    r = voices(key, chord, 50)[0]
    root = key + CHORD[chord][0]
    rr = (root % 12) + 48
    fifth = rr + 7
    pat = [rr, fifth, rr + 12, fifth] * beats
    step = bar / (2 * beats)
    for i, n in enumerate(pat[:2 * beats]):
        S.note("ost", t + i * step, n, step * 0.8, 58 + 52 * e + (10 if i % 2 == 0 else 0))
    if e > 0.65:                                                  # low strings pulse too
        for i in range(2 * beats):
            S.note("cel", t + i * step, rr - 12, step * 0.7, 52 + 50 * e)


def timpani(S, key, chord, t, bar, e, fill=False):
    root = key + CHORD[chord][0]
    n = (root % 12) + 38 if (root % 12) + 38 <= 50 else (root % 12) + 26
    beat = bar / 4
    S.note("tmp", t, n, beat, 70 + 50 * e)
    S.note("tmp", t + 2 * beat, n, beat, 60 + 45 * e)
    if fill:
        for i in range(4):
            S.note("tmp", t + 3 * beat + i * beat / 4, n + (7 if i % 2 else 0) - 12 * (n + 7 > 53), beat / 4, 70 + 40 * e)


def roll(S, t0, t1, v0, v1, timp_note=None, snare=True):
    n = max(2, int((t1 - t0) / 0.06))
    for i in range(n):
        tt = t0 + (t1 - t0) * i / n
        v = v0 + (v1 - v0) * (i / n) ** 1.5
        if snare:
            S.drum(tt, SNARE, v * 0.85, 0.05)
        if timp_note:
            S.note("tmp", tt, timp_note, 0.07, v)


def riser(S, key, t_hit, length, e):
    """the build into a hit: snare and timpani roll, a reverse cymbal, strings climbing"""
    t0 = t_hit - length
    roll(S, t0, t_hit - 0.02, 30, 95 + 30 * e, timp_note=(key % 12) + 38 - (12 if (key % 12) + 38 > 50 else 0) + 7)
    S.note("rev", t_hit - min(length, 2.4), 60, min(length, 2.4) - 0.02, 90 + 35 * e)
    steps = 16
    for i in range(steps):
        S.note("str", t0 + length * 0.5 + i * (length * 0.5 / steps), key + 12 + [0, 2, 4, 5, 7, 9, 11, 12][i % 8] + 12 * (i // 8),
               length * 0.5 / steps, 60 + 50 * e * i / steps)
    S.swell("hrn", t0, t_hit, 60, 120)
    S.swell("str", t0, t_hit, 70, 125)


def hit(S, key, chord, t, e, length=1.6):
    """a full-orchestra hit: crash, kick, taiko, timpani, brass chord"""
    S.drum(t, CRASH, 95 + 32 * e, 2.5)
    if e > 0.8:
        S.drum(t, CRASH2, 90 + 30 * e, 2.5)
    S.drum(t, KICK, 100 + 27 * e, 0.4)
    S.note("tko", t, 36, 1.0, 90 + 37 * e)
    root = key + CHORD[chord][0]
    S.note("tmp", t, (root % 12) + 38 - (12 if (root % 12) + 38 > 50 else 0), 1.2, 110 + 17 * e)
    for n in voices(key, chord, 48):
        S.note("tbn", t, n, length, 90 + 37 * e)
    S.note("tba", t, (root % 12) + 26 + (12 if (root % 12) + 26 < 28 else 0), length, 90 + 37 * e)
    for n in voices(key, chord, 60):
        S.note("hrn", t, n, length, 90 + 37 * e)
    S.cc("hrn", t, 11, 112)
    S.cc("str", t, 11, 112)


def theme(S, key, t, bar, bars, e, choir=False, octave=0):
    """the melody on trumpets (doubled by horns an octave down), with its chords, ostinato and timpani"""
    beat = bar / 4
    for i, b in enumerate(bars):
        tb = t + i * bar
        x = 0.0
        for semi, dur in THEME[b]:
            n = key + 24 + semi + octave
            S.note("tpt", tb + x * beat, n, dur * beat * 0.95, 82 + 45 * e)
            S.note("hrn", tb + x * beat, n - 12, dur * beat * 0.95, 75 + 45 * e)
            if e > 0.85:
                S.note("str", tb + x * beat, n + 12, dur * beat * 0.95, 70 + 40 * e)
            x += dur
        chords = THEME_CH[b]
        for j, ch in enumerate(chords):
            tc = tb + j * bar / len(chords)
            pad(S, key, ch, tc, bar / len(chords), e, strings=True, horns=False, choir=choir)
            for n in voices(key, ch, 48)[:2]:
                S.note("tbn", tc, n, bar / len(chords) * 0.9, 60 + 40 * e)
        ostinato(S, key, chords[0], tb, bar, e)
        timpani(S, key, chords[0], tb, bar, e, fill=(i == len(bars) - 1))
        S.drum(tb, KICK, 70 + 40 * e, 0.3)
        if e > 0.7:
            S.drum(tb + 2 * beat, TOM_L, 60 + 40 * e, 0.3)


def harp(S, key, chord, t, bar, e, beats=4):
    ns = voices(key, chord, 55) + voices(key, chord, 67)
    step = bar / (2 * beats)
    for i in range(2 * beats):
        S.note("hrp", t + i * step, ns[i % len(ns)], step * 3, 55 + 40 * e)


def sparkle(S, key, t0, t1, n, rs, e=0.5):
    scale = [0, 2, 4, 7, 9, 12, 14, 16, 19]
    for _ in range(n):
        tt = rs.uniform(t0, t1)
        S.note("glk", tt, key + 24 + scale[rs.integers(len(scale))], 0.6, 50 + 40 * e)


# ------------------------------------------------------------------ the sections
D = 50          # the key of D: MIDI 50 is the D below middle C; chords are voiced from here

def hook(S, t0):
    rs = np.random.default_rng(30)
    key = D
    bar = 2.4
    for i, ch in enumerate(["vi", "IV", "ii", "V"]):                # low and hushed under the title words
        pad(S, key, ch, t0 + i * bar, bar, 0.25, horns=(i > 1), choir=True)
    S.swell("str", t0, t0 + 9.6, 50, 85)
    sparkle(S, key, t0 + 2.4, t0 + 9.0, 22, rs, 0.35)                 # the scores falling into the pile
    for i, ch in enumerate(["vi", "IV", "I", "V"]):                 # the build: the pile, the tail, the ten
        tb = t0 + 9.6 + i * bar * 0.8
        pad(S, key, ch, tb, bar * 0.8, 0.45 + 0.1 * i, choir=True)
        ostinato(S, key, ch, tb, bar * 0.8, 0.45 + 0.1 * i)
        if i >= 1:
            timpani(S, key, ch, tb, bar * 0.8, 0.5 + 0.1 * i)
    riser(S, key, t0 + 17.4, 2.6, 0.9)
    hit(S, key, "I", t0 + 17.4, 1.0, 2.2)
    beat = 0.55
    x = 0
    for semi, dur in FANFARE:                                        # the call as the faces arrive
        S.note("tpt", t0 + 17.4 + x * beat, key + 24 + semi, dur * beat * 0.95, 118)
        S.note("hrn", t0 + 17.4 + x * beat, key + 12 + semi, dur * beat * 0.95, 105)
        x += dur
    pad(S, key, "I", t0 + 17.4, 2.2, 0.9, choir=True)
    pad(S, key, "vi", t0 + 19.6, 2.2, 0.85, choir=True)
    roll(S, t0 + 21.1, t0 + 21.78, 50, 110)
    hit(S, key, "I", t0 + 21.8, 1.0, 1.8)                            # the title
    theme(S, key, t0 + 21.8, 2.4, [0, 1, 2], 0.95, choir=True)
    pad(S, key, "V", t0 + 29.0, 1.4, 0.6, choir=True)


CARD_KEY = {10: -2, 9: 0, 8: -2, 7: 0, 6: 1, 5: 3, 4: 1, 3: 3, 2: 5, 1: 0}
PROGS = [["I", "vi", "IV", "V"], ["vi", "IV", "I", "V"], ["I", "iii", "IV", "V"], ["IV", "I", "vi", "V"]]


def card(S, t0, rank):
    e = 0.55 + 0.05 * (10 - rank)
    key = D + CARD_KEY[rank]
    bar = 18.2 / 8                                                   # the score lands on bar 8
    prog = PROGS[rank % len(PROGS)]
    hit(S, key, "I", t0 + 0.05, e * 0.8, 1.2)                         # the rank number
    pad(S, key, "I", t0 + 0.05, bar, e * 0.7)
    for i in range(1, 6):                                            # the stacks build, the curve draws
        ch = prog[(i - 1) % 4]
        tb = t0 + i * bar
        pad(S, key, ch, tb, bar, e * (0.55 + 0.06 * i), horns=(i >= 3))
        ostinato(S, key, ch, tb, bar, e * (0.5 + 0.08 * i))
        if i >= 3:
            timpani(S, key, ch, tb, bar, e * 0.7)
        if rank <= 3 and i == 5:
            sparkle(S, key, tb, tb + bar, 6, np.random.default_rng(rank), e)
    for i, ch in enumerate(["IV", "V"]):                             # the ruler climbs
        tb = t0 + (6 + i) * bar
        pad(S, key, ch, tb, bar, e * 0.85)
        ostinato(S, key, ch, tb, bar, e * 0.9)
    riser(S, key, t0 + 18.2, bar * 2 - 0.1, e)
    hit(S, key, "I", t0 + 18.2, e, 1.8)                               # the score lands
    halves = [0, 1, 2, 3] if rank % 2 == 0 else [4, 5, 6, 7]
    if rank == 1:
        halves = [0, 1, 2, 3, 4, 5, 6, 7]
    theme(S, key, t0 + 18.2, bar, halves, e, choir=(e >= 0.8))
    if rank != 1:                                                    # settle under the text
        for i, ch in enumerate(["IV", "V", "I"]):
            tb = t0 + 18.2 + (4 + i) * bar
            pad(S, key, ch, tb, bar * (1.4 if i == 2 else 1), e * (0.5 - 0.08 * i), horns=(i < 2))
            harp(S, key, ch, tb, bar, e * 0.6)
        S.swell("str", t0 + 27.3, t0 + 35.5, 100, 55)
        S.swell("hrn", t0 + 27.3, t0 + 35.5, 100, 50)


def calm(S, t0, dur, minor=False, events=()):
    """the explainer (and, in a minor key, the women's throws): harp and strings, quiet"""
    bar = 3.0 if not minor else 3.4
    key = D if not minor else D + 9 - 12                            # B minor
    prog = ["I", "vi", "IV", "V"] if not minor else ["i", "VI", "III", "VII"]
    n = int(math.ceil(dur / bar))                                     # fill the section; the next one overlaps
    S.cc("str", t0, 11, 80)
    S.cc("hrn", t0, 11, 80)
    for i in range(n):
        ch = prog[i % 4]
        tb = t0 + i * bar
        pad(S, key, ch, tb, bar, 0.32, horns=(i % 4 == 0))
        harp(S, key, ch, tb, bar, 0.4)
    for tt in events:                                                # a sparkle on a counted "5" or "1"
        sparkle(S, key, t0 + tt, t0 + tt + 1.2, 5, np.random.default_rng(int(tt * 10)), 0.6)
    S.cc("str", t0 + dur, 11, 100)
    S.cc("hrn", t0 + dur, 11, 100)


CLOSE_FINALE, CLOSE_DUR = 41.0, 58.0           # when "Not a ranking of the greatest athletes" lands, and the close's length


def close(S, t0, finale=None, dur=None):
    finale, dur = finale or CLOSE_FINALE, dur or CLOSE_DUR
    key = D
    bar = 2.4
    n = max(2, int((finale - 21.8) / bar + 1e-6))
    for i, ch in enumerate((["I", "IV", "vi", "V", "I", "IV", "ii", "V"] * 2)[:n]):   # the lists / the four
        tb = t0 + i * bar
        pad(S, key, ch, tb, bar, 0.55)
        ostinato(S, key, ch, tb, bar, 0.45)
    calm(S, t0 + n * bar, finale - 2.2 - n * bar)                    # reading time
    riser(S, key, t0 + finale, 3.0, 1.0)
    hit(S, key, "I", t0 + finale, 1.0, 2.0)                           # "Not a ranking of the greatest athletes"
    theme(S, key, t0 + finale, (dur - finale) / 8, list(range(8)), 1.0, choir=True)


def end(S, t0):
    key = D
    hit(S, key, "I", t0 + 0.05, 0.85, 3.0)
    pad(S, key, "I", t0, 6.0, 0.8, choir=True)
    for semi, at, d in ((7, 1.0, 1.2), (12, 2.4, 1.2), (14, 3.6, 0.6), (16, 4.2, 2.4)):
        S.note("hrn", t0 + at, key + 12 + semi, d, 95)
    pad(S, key, "IV", t0 + 6.0, 4.0, 0.5, choir=True)
    pad(S, key, "I", t0 + 10.0, 14.0, 0.4, choir=True)
    harp(S, key, "I", t0 + 10.0, 3.0, 0.4)
    S.swell("str", t0 + 12.0, t0 + 24.0, 100, 20)
    S.swell("chr", t0 + 12.0, t0 + 24.0, 100, 20)
    S.swell("hrn", t0 + 12.0, t0 + 24.0, 100, 20)


def build(O):
    S = Score()
    for name, t0 in O.items():
        if name == "hook":
            hook(S, t0)
        elif name[0] == "c" and name[1:].isdigit():
            card(S, t0, int(name[1:]))
        elif name == "curve":
            calm(S, t0, 44.0, events=(16.0, 30.0))
        elif name == "fair":
            calm(S, t0, 40.0)
        elif name == "throws":
            calm(S, t0, 58.0, minor=True)
        elif name == "sittler":
            calm(S, t0, 48.0, minor=True)
        elif name == "close":
            close(S, t0)
        elif name == "end":
            end(S, t0)
    return S


def render(O, total):
    """MIDI -> FluidSynth -> float32 stereo at SR, trimmed to the film"""
    os.makedirs(OUT, exist_ok=True)
    mid, wav = os.path.join(OUT, "score.mid"), os.path.join(OUT, "score_fluid.wav")
    build(O).save(mid)
    subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.45", "-r", str(SR), "-R", "1", "-C", "0",
                    "-o", "synth.reverb.room-size=0.82", "-o", "synth.reverb.damp=0.25", "-o", "synth.reverb.width=1.0",
                    "-o", "synth.reverb.level=0.75", "-o", "synth.polyphony=512", "-F", wav, SF2, mid], check=True)
    from scipy.io import wavfile
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    n = int(total * SR)
    x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
    t = np.arange(n) / SR
    x *= np.clip((total - t) / 2.5, 0, 1)[:, None]
    return (x / (np.abs(x).max() + 1e-9) * 0.89).astype(np.float32)


def write(x, path):
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import assemble
    O, total = assemble.timeline()
    S = build(O)
    if S.dropped:
        raise SystemExit(f"{len(S.dropped)} notes out of range, e.g. {S.dropped[:5]}")
    x = render(O, total)
    write(x, os.path.join(OUT, "score.wav"))
    print(f"{total:.1f} s, peak {np.abs(x).max():.2f}")
