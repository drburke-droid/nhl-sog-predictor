"""A meandering river on a flat plain: the Howard & Knutson (1984) centreline model that Stolum (1996) used.
Each point of the channel moves sideways at a rate set by the curvature just upstream of it (water carried
round a bend keeps eroding the outer bank downstream). When two loops touch, the neck is cut through and
the loop is left behind as an oxbow lake.

    python meander.py   -> data/meander_video.npz (centrelines over time, oxbows, sinuosity)
"""
import os, math
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
W = 1.0                    # channel width (lengths in widths)
DS = 0.5 * W               # spacing of centreline points
OMEGA, GAMMA = -1.0, 2.5   # Howard & Knutson weights: local term, upstream term
ALPHA = 0.3                # upstream decay, per width (2 Cf / depth)
KMIG = 0.1                 # migration rate per unit curvature
CUT = 1.2 * W              # neck cutoff distance
LOOP_MAX = 400 * W         # the longest loop a neck cutoff removes
ENDS = 10                  # points at each end that take no part in cutoffs
NOISE = 0.002 * W          # random bank disturbances per check


@nb.njit(cache=True)
def step(x, y, dt):
    n = len(x)
    # curvature (signed) and normals
    C = np.zeros(n); nx = np.zeros(n); ny = np.zeros(n)
    for i in range(1, n - 1):
        dx1, dy1 = x[i] - x[i - 1], y[i] - y[i - 1]
        dx2, dy2 = x[i + 1] - x[i], y[i + 1] - y[i]
        a1, a2 = math.atan2(dy1, dx1), math.atan2(dy2, dx2)
        da = a2 - a1
        while da > math.pi: da -= 2 * math.pi
        while da < -math.pi: da += 2 * math.pi
        l = 0.5 * (math.hypot(dx1, dy1) + math.hypot(dx2, dy2))
        C[i] = da / l
        tx, ty = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
        tl = math.hypot(tx, ty)
        nx[i], ny[i] = -ty / tl, tx / tl
    R0 = KMIG * C
    K = int(8.0 / ALPHA / DS)
    R1 = np.zeros(n)
    for i in range(1, n - 1):
        s = 0.0; g = 0.0
        for j in range(1, min(K, i) + 1):
            w = math.exp(-ALPHA * j * DS)
            s += R0[i - j] * w; g += w
        R1[i] = OMEGA * R0[i] + (GAMMA * s / g if g > 0 else 0.0)
    for i in range(1, n - 1):
        x[i] -= R1[i] * nx[i] * dt
        y[i] -= R1[i] * ny[i] * dt


def resample(x, y):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s = np.arange(0, d[-1], DS)
    s = np.r_[s, d[-1]] if d[-1] - s[-1] > 0.3 * DS else np.r_[s[:-1], d[-1]]
    return np.interp(s, d, x), np.interp(s, d, y)


@nb.njit(cache=True)
def find_cut(x, y, cut, skip, longest):
    """the first neck: two points closer than `cut` that are between `skip` and `longest` points apart
    along the channel (one loop, not the whole river folding back on itself)"""
    n = len(x)
    for i in range(10, n - 10):
        for j in range(i + skip, min(n - 10, i + longest)):
            if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < cut * cut:
                return i, j
    return -1, -1


def run(L=200.0, steps=60000, dt=0.05, seed=1, every=40):
    rng = np.random.default_rng(seed)
    x = np.arange(0, L + DS / 2, DS)
    y = np.zeros(len(x))
    for lam in rng.uniform(15, 60, 12):                       # a gently wavy start
        y += rng.normal(0, 0.15) * np.sin(2 * np.pi * x / lam + rng.uniform(0, 2 * np.pi))
    y -= np.linspace(y[0], y[-1], len(y))
    frames, oxbows, sin_t = [], [], []
    for it in range(steps):
        step(x, y, dt)
        if it % 5 == 0:
            x, y = resample(x, y)
            i, j = find_cut(x, y, CUT, int(4 * CUT / DS) + 4, int(LOOP_MAX / DS))
            while i >= 0:
                oxbows.append((it, np.c_[x[i:j + 1], y[i:j + 1]].astype(np.float32)))
                x, y = np.r_[x[:i + 1], x[j:]], np.r_[y[:i + 1], y[j:]]
                x, y = resample(x, y)
                i, j = find_cut(x, y, CUT, int(4 * CUT / DS) + 4, int(LOOP_MAX / DS))
            # small disturbances (a fallen tree, a slump) keep new bends starting
            y[1:-1] += rng.normal(0, NOISE, len(y) - 2)
        if it % every == 0:
            Ltot = np.hypot(np.diff(x), np.diff(y)).sum()
            sin_t.append(Ltot / math.hypot(x[-1] - x[0], y[-1] - y[0]))
            frames.append(np.c_[x, y].astype(np.float32))
    return frames, oxbows, np.array(sin_t)


def save_video_run(L=160.0, steps=300000, dt=0.1, seed=3, every=200):
    frames, oxbows, S = run(L=L, steps=steps, dt=dt, seed=seed, every=every)
    off = np.r_[0, np.cumsum([len(f) for f in frames])]
    ooff = np.r_[0, np.cumsum([len(o) for _, o in oxbows])]
    np.savez(os.path.join(DATA, "meander_video.npz"), xy=np.concatenate(frames), off=off, S=S, every=every, L=L,
             ox_it=np.array([it for it, _ in oxbows]), ox_xy=np.concatenate([o for _, o in oxbows]) if oxbows else np.zeros((0, 2)),
             ox_off=ooff)
    return S


if __name__ == "__main__":
    import time
    t0 = time.time()
    S = save_video_run()
    k = len(S)
    print(f"{k} frames, {time.time() - t0:.0f}s; sinuosity over the last two thirds: mean {S[k // 3:].mean():.3f}, "
          f"sd {S[k // 3:].std():.3f}")
