"""Period sound effects, synthesised (no samples, nothing to license): rubber stamp, typewriter, paper, telegraph, pen, dice.

Each function returns a mono float32 array at RATE. events(scene) lists (time, kind) pairs for a scene, read from its elements.
"""
import numpy as np

RATE = 48000
R = np.random.default_rng(1935)


def _env(n, attack, decay):
    t = np.arange(n) / RATE
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _bandish(x, lo, hi):
    """Band-pass via FFT masking (fine for short sounds)."""
    f = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / RATE)
    f[(fr < lo) | (fr > hi)] = 0
    return np.fft.irfft(f, len(x)).astype(np.float32)


def thud():
    n = int(0.32 * RATE)
    t = np.arange(n) / RATE
    body = np.sin(2 * np.pi * (95 - 40 * t / t[-1]) * t) * _env(n, 0.002, 0.07)
    knock = _bandish(R.normal(0, 1, n), 200, 2500) * _env(n, 0.0005, 0.018)
    return (0.9 * body + 0.6 * knock).astype(np.float32)


def key():
    n = int(0.05 * RATE)
    click = _bandish(R.normal(0, 1, n), 1500, 9000) * _env(n, 0.0003, 0.006)
    t = np.arange(n) / RATE
    ping = np.sin(2 * np.pi * R.uniform(2300, 3100) * t) * _env(n, 0.0005, 0.01) * 0.25
    return ((click + ping) * R.uniform(0.6, 1.0)).astype(np.float32)


def swoosh(dur=0.6):
    n = int(dur * RATE)
    x = _bandish(R.normal(0, 1, n), 600, 6000)
    t = np.arange(n) / n
    return (x * np.sin(np.pi * t) ** 2 * 0.35).astype(np.float32)


def tap():
    n = int(0.09 * RATE)
    return (_bandish(R.normal(0, 1, n), 120, 1800) * _env(n, 0.001, 0.02) * 0.6).astype(np.float32)


def sounder():
    """A telegraph sounder: a sharp double clack."""
    n = int(0.12 * RATE)
    one = _bandish(R.normal(0, 1, int(0.03 * RATE)), 900, 5000) * _env(int(0.03 * RATE), 0.0002, 0.005)
    out = np.zeros(n, np.float32)
    out[:len(one)] += one
    j = int(R.uniform(0.05, 0.08) * RATE)
    out[j:j + len(one)] += 0.7 * one[:max(0, min(len(one), n - j))]
    return out


def scratch(dur=0.4):
    n = int(dur * RATE)
    x = _bandish(R.normal(0, 1, n), 3000, 12000)
    grain = 0.6 + 0.4 * np.abs(np.sin(np.arange(n) / RATE * 2 * np.pi * R.uniform(18, 30)))
    t = np.arange(n) / n
    return (x * grain * np.sin(np.pi * t) * 0.12).astype(np.float32)


def rattle(dur=1.5):
    out = np.zeros(int(dur * RATE) + RATE // 5, np.float32)
    t = 0.0
    while t < dur:
        k = tap() * R.uniform(0.5, 1.2)
        i = int(t * RATE)
        out[i:i + len(k)] += k[:len(out) - i]
        t += R.uniform(0.04, 0.16) * (1 + 2 * t / dur)
    return out


GAIN = {'thud': 0.55, 'key': 0.16, 'swoosh': 0.5, 'tap': 0.35, 'sounder': 0.22, 'scratch': 0.6, 'rattle': 0.4}
MAKE = {'thud': thud, 'key': key, 'swoosh': swoosh, 'tap': tap, 'sounder': sounder, 'scratch': scratch, 'rattle': rattle}


def events(scene):
    """(time, kind) pairs read off a scene's pages and elements."""
    import kit
    ev = []
    for k, p in enumerate(getattr(scene, 'pages', [])):
        if k and p.enter != 'cut':
            ev.append((p.t0, 'swoosh'))
        for e in p.els:
            if isinstance(e, kit.Stamp):
                ev.append((e.t0 + 0.1, 'thud'))
            elif isinstance(e, kit.Typed):
                for i, ch in enumerate(e.s):
                    if ch.strip():
                        ev.append((e.t0 + i / e.cps, 'key'))
            elif isinstance(e, kit.Telegram):
                ev.append((e.t0 + 0.3, 'tap'))
                n = sum(len(w) for w in e.words)
                ev += [(e.t0 + 0.35 + i / e.wps, 'sounder') for i in range(n)]
            elif isinstance(e, kit.Obj):
                ev.append((e.t0 + (e.dur * 0.75 if e.how == 'drop' else 0.0), 'tap' if e.how == 'drop' else 'swoosh'))
            elif isinstance(e, kit.Line):
                ev.append((e.t0, 'scratch'))
    ev += getattr(scene, 'extra_sfx', lambda: [])()
    return [(t, kind) for t, kind in ev if 0 <= t < scene.dur]


def render_track(timed, length):
    """Mix (absolute time, kind) events into one mono track of `length` seconds."""
    out = np.zeros(int(length * RATE) + RATE, np.float32)
    for t, kind in timed:
        s = MAKE[kind]() * GAIN[kind]
        i = int(t * RATE)
        if i >= len(out):
            continue
        out[i:i + len(s)] += s[:len(out) - i]
    return np.clip(out, -1, 1)
