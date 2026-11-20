"""The score: a D-minor drone, a sub-bass hit when the Moon breaks, a pulse that builds through the rain,
and a cold, sparse end. Synthesized (the felt piano, pad and reverb come from ../../vernier/music.py).

python3 film/music.py             -> out/film_data/score_ladder.wav, timed to assemble.LADDER_CUT
python3 film/music.py seveneves   -> out/film_data/score.wav, timed to assemble.CUT
"""
import importlib.util
import os
import sys

import numpy as np
from scipy.signal import fftconvolve, lfilter, butter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location('vmusic', os.path.join(ROOT, '..', 'vernier', 'music.py'))
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
SR = vm.SR
sys.path.insert(0, HERE)
import assemble
import shots_meta


def timeline():
    t, marks = 0.0, []
    for item in assemble.CUT:
        if item[0] == 'card':
            d = item[1]
            name = 'card'
        else:
            name = item[0]
            d = shots_meta.FRAMES[name] / assemble.FPS
        marks.append((name, t, d))
        t += d
    return marks, t


def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    n = min(len(sig), len(buf) - i)
    if n > 0:
        buf[i:i + n] += gain * sig[:n]


def boom(dur=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 55 * np.exp(-t / 1.2) + 28
    ph = 2 * np.pi * np.cumsum(f) / SR
    env = (1 - np.exp(-t / 0.01)) * np.exp(-t / 1.6)
    noise = lfilter(*butter(2, 180 / (SR / 2)), np.random.default_rng(1).normal(0, 1, n))
    return (np.sin(ph) * 0.9 + noise * 0.25 * np.exp(-t / 0.5)) * env


def pulse(dur, bpm, gain_curve):
    n = int(dur * SR)
    out = np.zeros(n)
    beat = 60.0 / bpm
    k = 0
    while k * beat < dur:
        tt = np.arange(int(0.9 * SR)) / SR
        g = float(gain_curve(k * beat / dur))
        thump = np.sin(2 * np.pi * (48 * np.exp(-tt / 0.15) + 36) * tt) * np.exp(-tt / 0.22) * g
        add(out, thump, k * beat)
        k += 1
    return out


def shimmer(dur, curve, seed=2):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    b, a = butter(2, [3000 / (SR / 2), 9000 / (SR / 2)], btype='band')
    x = lfilter(b, a, x)
    env = np.array([curve(i / n) for i in range(0, n, 480)])
    env = np.interp(np.arange(n), np.arange(0, n, 480), env)
    return x * env * 0.08


def main():
    marks, total = timeline()
    T = {m[0] if m[0] != 'card' else f'card{i}': (m[1], m[2]) for i, m in enumerate(marks)}
    n = int((total + 4) * SR)
    dry = np.zeros(n)
    # the drone, all the way through
    for note, g in (('D2', 0.35), ('A2', 0.22), ('D3', 0.12)):
        add(dry, vm.pad_tone(vm.hz(note), total + 2, fade_in=4.0, fade_out=5.0, seed=hash(note) % 100), 0.0, g)
    # the Moon: three slow piano notes
    a0, ad = T['A_moon']
    for i, (note, v) in enumerate((('A4', 0.35), ('F4', 0.30), ('D4', 0.32))):
        add(dry, vm.piano(vm.hz(note), 1.6, v, seed=i), a0 + 0.6 + 1.5 * i, 0.5)
    # the break
    b0, bd = T['B_breakup']
    add(dry, boom(), b0, 1.1)
    for note, g in (('Bb2', 0.18), ('F3', 0.12), ('C4', 0.08)):
        add(dry, vm.pad_tone(vm.hz(note), bd + T['C_merge'][1], fade_in=1.5, fade_out=3.0), b0 + 0.3, g)
    # the merge: one low piano octave
    c0, cd = T['C_merge']
    add(dry, vm.piano(vm.hz('D2'), 2.5, 0.6, seed=7), c0 + cd * 0.5, 0.6)
    add(dry, vm.piano(vm.hz('D3'), 2.5, 0.45, seed=8), c0 + cd * 0.5, 0.5)
    # the wide shot and the rain: a pulse that quickens and grows, shimmer rising
    d0, dd = T['D_wide']
    e0, ed = T['E_rain']
    add(dry, pulse(dd, 54, lambda x: 0.25 + 0.2 * x), d0, 0.7)
    add(dry, pulse(ed, 66, lambda x: 0.45 + 0.4 * x), e0, 0.8)
    add(dry, shimmer(dd + ed, lambda x: x ** 2), d0, 1.0)
    for note, g in (('G2', 0.16), ('D3', 0.12), ('Bb3', 0.08)):
        add(dry, vm.pad_tone(vm.hz(note), ed + 1, fade_in=2.0, fade_out=2.0), e0, g)
    # the steam: everything falls away to one high, cold note
    f0, fd = T['F_steam']
    add(dry, vm.pad_tone(vm.hz('A5'), fd + 3, fade_in=1.5, fade_out=4.0), f0, 0.05)
    add(dry, vm.piano(vm.hz('D5'), 3.0, 0.3, seed=12), f0 + 1.0, 0.4)
    # the question card and the ladder: a slow rising arpeggio
    h0, hd = T['H_ladder']
    for i, note in enumerate(('D4', 'F4', 'A4', 'C5', 'E5')):
        add(dry, vm.piano(vm.hz(note), 1.2, 0.3 + 0.05 * i, seed=20 + i), h0 + 0.4 + i * hd / 5.2, 0.45)
    # mix: reverb, gentle limiter
    ir = vm.reverb_ir(rt60=3.2, length=4.5)
    wet = np.stack([fftconvolve(dry, ir[:, c])[:n] for c in range(2)], 1)
    mix = np.stack([dry, dry], 1) * 0.6 + wet * 0.55
    fade = np.ones(n)
    k = int(3 * SR)
    fade[-k:] = np.linspace(1, 0, k)
    mix *= fade[:, None]
    mix /= max(np.abs(mix).max() / 0.85, 1e-9)
    out = os.path.join(ROOT, 'out', 'film_data', 'score.wav')
    vm.write(mix.astype(np.float32), out)
    print('wrote', out, f'{total:.1f} s')




def ladder_score():
    """Score for the impact-ladder cut: drone; for each impact a hit whose depth grows with the size;
    a pulse under each Earth shot; a rising arpeggio under the summary."""
    t, marks = 0.0, []
    for item in assemble.LADDER_CUT:
        if item[0] in ('card', 'summary'):
            d = item[1]
            marks.append((item[0], t, d))
        else:
            d = shots_meta.FRAMES[item[0]] / assemble.FPS
            marks.append((item[0], t, d))
        t += d
    total = t
    n = int((total + 4) * SR)
    dry = np.zeros(n)
    for note, g in (('D2', 0.30), ('A2', 0.18)):
        add(dry, vm.pad_tone(vm.hz(note), total + 2, fade_in=4.0, fade_out=5.0, seed=3), 0.0, g)
    size_gain = {'I1_1m': 0.15, 'I2_60m': 0.3, 'I3_1km': 0.55, 'I4_10km': 0.8, 'I5_100km': 1.1}
    flash_at = {'I1_1m': 30 / 24, 'I2_60m': 30 / 24}
    for name, t0, d in marks:
        if name in size_gain:
            add(dry, boom(5.0), t0 + flash_at.get(name, 0.05), size_gain[name])
        if name.startswith('E'):
            k = {'E3_1km': 0.35, 'E4_10km': 0.6, 'E5_100km': 0.85}[name]
            add(dry, pulse(d, 60 + 10 * k, lambda x: k * (0.6 + 0.4 * x)), t0, 0.8)
            add(dry, shimmer(d, lambda x: k * (0.5 + 0.5 * x)), t0, 1.0)
        if name == 'H_ladder':
            for i, note in enumerate(('A4', 'F4', 'D4')):
                add(dry, vm.piano(vm.hz(note), 1.6, 0.32, seed=i), t0 + 0.8 + 1.6 * i, 0.5)
        if name == 'summary':
            for i, note in enumerate(('D4', 'F4', 'A4', 'C5', 'E5')):
                add(dry, vm.piano(vm.hz(note), 1.2, 0.3 + 0.05 * i, seed=20 + i), t0 + 0.4 + 1.5 * i, 0.45)
    ir = vm.reverb_ir(rt60=3.2, length=4.5)
    wet = np.stack([fftconvolve(dry, ir[:, c])[:n] for c in range(2)], 1)
    mix = np.stack([dry, dry], 1) * 0.6 + wet * 0.55
    fade = np.ones(n)
    k = int(3 * SR)
    fade[-k:] = np.linspace(1, 0, k)
    mix *= fade[:, None]
    mix /= max(np.abs(mix).max() / 0.85, 1e-9)
    out = os.path.join(ROOT, 'out', 'film_data', 'score_ladder.wav')
    vm.write(mix.astype(np.float32), out)
    print('wrote', out, f'{total:.1f} s')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'seveneves':
        main()
    else:
        ladder_score()
