"""
Original score for the knockout film, synthesized from scratch (no samples):
a marimba-like pattern over a soft pad, building round by round, with a bell
for each winner.  Reads the timeline written by make_film.py compose.

    python film_score.py mixing_knockout_timeline.json score.wav
"""
import json
import sys
import wave

import numpy as np

SR = 44100
BPM = 92.0
BEAT = 60.0 / BPM
# four-bar loop, one chord a bar: C  Am  F  G  (as MIDI notes)
CHORDS = [[48, 55, 60, 64, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60], [43, 50, 55, 59, 62]]
PATTERN = [0, 2, 3, 4, 3, 2, 3, 1]            # which chord tone on each eighth note


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def marimba(f, dur=0.9, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * 5.5) * (1 - np.exp(-t * 900))
    y = np.sin(2 * np.pi * f * t) + 0.32 * np.sin(2 * np.pi * 3.93 * f * t) * np.exp(-t * 14) \
        + 0.12 * np.sin(2 * np.pi * 9.5 * f * t) * np.exp(-t * 30)
    return amp * env * y


def bell(f, dur=2.6, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
            for r, a, d in ((1, 1.0, 1.6), (2.0, 0.5, 2.2), (2.76, 0.35, 3.0), (5.4, 0.18, 5.0)))
    return amp * y * (1 - np.exp(-t * 600))


def pad(f, dur, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / 0.5) * np.minimum(1, (dur - t) / 0.5)
    y = sum(np.sin(2 * np.pi * f * dt * t + ph) for dt, ph in ((0.997, 0.0), (1.0, 1.3), (1.004, 2.1)))
    return amp * env * y / 3


def shaker(dur=0.09, amp=1.0, rng=None):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0.0)                  # tilt the noise towards the top end
    return amp * x * np.exp(-np.arange(n) / SR * 55)


def put(buf, y, t0, pan=0.0):
    i = int(t0 * SR)
    if i >= buf.shape[1]:
        return
    n = min(len(y), buf.shape[1] - i)
    buf[0, i:i + n] += y[:n] * (0.5 - 0.5 * pan) * 1.4
    buf[1, i:i + n] += y[:n] * (0.5 + 0.5 * pan) * 1.4


def reverb(buf, seconds=1.6, mix=0.22, rng=None):
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 4.2)
    ir[: int(0.012 * SR)] = 0
    out = np.empty_like(buf)
    N = buf.shape[1] + n
    F = np.fft.rfft(ir, N)
    for c in range(2):
        wet = np.fft.irfft(np.fft.rfft(buf[c], N) * F, N)[: buf.shape[1]]
        out[c] = buf[c] + mix * wet / (np.abs(wet).max() + 1e-9) * np.abs(buf[c]).max()
    return out


def main(timeline_path, out_path):
    tl = json.load(open(timeline_path))
    total = tl["seconds"]
    rng = np.random.default_rng(3)
    buf = np.zeros((2, int((total + 3) * SR)))

    # how busy the music is, per segment: 0 cards, 1 quarter-finals, 2 semi-finals, 3 final
    def level(t):
        lv = 0
        for name, start, length in tl["segments"]:
            if start <= t < start + length:
                lv = {"Q": 1, "S": 2, "F": 3}.get(name[0], 0) if name[:2] in ("Q1", "Q2", "Q3", "Q4", "S1", "S2", "F1") else 0
        return lv

    n_eighths = int(total / (BEAT / 2))
    for k in range(n_eighths):
        t = k * BEAT / 2
        bar = (k // 8) % 4
        chord = CHORDS[bar]
        lv = level(t)
        step = k % 8
        if step == 0:                                            # pad and bass once a bar
            for m in chord[1:4]:
                put(buf, pad(hz(m), 4 * BEAT + 0.4, 0.075), t)
            if lv >= 2:
                put(buf, pad(hz(chord[0] - 12), 4 * BEAT, 0.22), t)
        if lv == 0:
            if step in (0, 3, 6):                                # sparse during the cards
                put(buf, marimba(hz(chord[PATTERN[step]] + 12), amp=0.20), t, pan=-0.2 + 0.4 * (step % 2))
        else:
            vel = 0.24 + 0.05 * (step % 4 == 0)
            put(buf, marimba(hz(chord[PATTERN[step]] + 12), amp=vel), t, pan=-0.3 if step % 2 else 0.3)
            if lv >= 3 and step % 2 == 0:
                put(buf, marimba(hz(chord[PATTERN[(step + 3) % 8]] + 24), 0.6, 0.11), t + BEAT / 4, pan=0.0)
            if lv >= 2 and step % 2 == 1:
                put(buf, shaker(amp=0.05, rng=rng), t, pan=0.15)
    # a bell when each match is decided (the result fades in 0.5 s before the segment's last 2.2 s)
    for name, start, length in tl["segments"]:
        if name in ("Q1", "Q2", "Q3", "Q4", "S1", "S2", "F1"):
            tb = start + length - 2.7
            put(buf, bell(hz(84), amp=0.16), tb)
            put(buf, bell(hz(79), amp=0.10), tb + 0.02, pan=0.3)
            if name == "F1":
                put(buf, bell(hz(91), amp=0.12), tb + 0.18, pan=-0.3)
    buf = reverb(buf, rng=rng)
    fade = np.ones(buf.shape[1])
    n_in, n_out = int(1.0 * SR), int(3.0 * SR)
    end = int(total * SR)
    fade[:n_in] = np.linspace(0, 1, n_in)
    fade[end - n_out:end] = np.linspace(1, 0, n_out)
    fade[end:] = 0
    buf = buf[:, :end + 1] * fade[:end + 1]
    buf = np.tanh(buf / np.abs(buf).max() * 1.4) * 0.8                # gentle limiting
    pcm = (buf.T * 32767).astype("<i2")
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", out_path, f"{total:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
