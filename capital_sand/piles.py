"""The shape of each person's gold: glass initials that fill first, then a mountain of poured balls that
overflows them, for any number of balls.

Packing is realistic: a cell is CELL map px square and one ball tall per level (DZ map px), so a pile of N
balls holds N cell-levels, and its flanks keep to about SLOPE levels per cell (a ~38 degree sand slope).
Small fortunes sit inside the letters; once the letters are full (to the glass's height, WALL_LV levels)
the rest piles up as a messy mountain centred on the initials (shoulders, ridges, grain-scale roughness),
spilling over the glass onto the table.

    pile = Pile(initials, centre=(x, y), n_max, seed)     # precomputes shapes for 0..n_max balls
    cells, heights = pile.shape(n)                        # float heights (levels) on pile.cells
"""
import numpy as np

CELL = 3.0            # map px
DZ = 2.4              # map px per level: balls of ~3 px diameter, packed
WALL_PX = 60.0        # glass initials' height, map px
WALL_LV = int(WALL_PX // DZ)
SLOPE = 1.0           # mountain flank: levels per cell (rise 2.4 px over 3 px: ~38 degrees)


class Pile:
    def __init__(self, ini, n_max, seed=0, grid=(1707, 960), n_steps=64):
        rng = np.random.default_rng(seed)
        self.ini = ini
        self.inside = np.ravel_multi_index(ini.inside, grid[::-1])     # cells inside the letters (flat index)
        self.cap = len(self.inside) * WALL_LV
        cx, cy = ini.centre
        r_max = (3 * max(n_max, 1) / (np.pi * SLOPE)) ** (1 / 3) * 1.35 + 4      # widest the mountain gets, cells
        ccx, ccy = cx / CELL, cy / CELL
        c0, c1 = int(max(ccx - r_max, 0)), int(min(ccx + r_max + 1, grid[0]))
        r0, r1 = int(max(ccy - r_max, 0)), int(min(ccy + r_max + 1, grid[1]))
        rr, cc = np.mgrid[r0:r1, c0:c1]
        win = (rr * grid[0] + cc).ravel()
        self.cells = np.union1d(win, self.inside)
        cr, ccol = np.divmod(self.cells, grid[0])
        self.ox, self.oy = (ccol + 0.5 - ccx).astype(np.float32), (cr + 0.5 - ccy).astype(np.float32)
        self.d = np.hypot(self.ox, self.oy)
        self.is_in = np.isin(self.cells, self.inside)
        ns = int(rng.integers(2, 5)); a = rng.uniform(0, 2 * np.pi, ns)
        self.sat = np.stack([np.cos(a), np.sin(a), rng.uniform(0.3, 0.65, ns), rng.uniform(0.35, 0.6, ns),
                             rng.uniform(0.3, 0.45, ns)], 1)
        ka = rng.uniform(0, 2 * np.pi, 3)
        self.wav = np.stack([np.cos(ka), np.sin(ka)], 1) * rng.uniform(5, 9, (3, 1))
        self.ph = rng.uniform(0, 2 * np.pi, 3)
        self.jit = rng.uniform(0.9, 1.1, len(self.cells))
        self.fill_jit = rng.uniform(-0.35, 0.35, len(self.cells))
        # precomputed shapes for n = 0 .. n_max (each holds exactly n balls); shape(n) interpolates between them
        lo = np.linspace(0, min(self.cap, n_max), 12)
        hi = np.geomspace(max(self.cap, 1) * 1.001, max(n_max, self.cap * 1.01) * 1.02, n_steps) if n_max > self.cap else []
        self.N = np.unique(np.r_[lo, hi])
        self.H = np.stack([self._solve(n) for n in self.N]).astype(np.float32)

    def _mountain(self, h):
        R = max(h / SLOPE, 1.0)
        u = self.d / R
        f = np.clip(1 - u, 0, 1) ** 1.35
        for dx, dy, dist, hs, rs in self.sat:
            us = np.hypot(self.ox - dx * dist * R, self.oy - dy * dist * R) / (rs * R)
            f = np.maximum(f, hs * np.clip(1 - us, 0, 1) ** 1.35)
        rid = np.cos(np.stack([self.ox / R, self.oy / R], 1) @ self.wav.T + self.ph).mean(1)
        return h * f * (1 + 0.22 * rid * np.minimum(u * 2.5, 1)) * np.where(u < 0.06, 1.0, self.jit)

    def _solve(self, n):
        if n <= self.cap:                                 # filling the letters, level with a little texture
            lvl = n / max(self.is_in.sum(), 1)
            hgt = np.where(self.is_in, np.clip(lvl + self.fill_jit * min(lvl, 1.5), 0, WALL_LV), 0.0)
            return hgt * (n / max(hgt.sum(), 1e-9)) if n > 0 else hgt
        base = np.where(self.is_in, float(WALL_LV), 0.0)
        lo, hi = 0.0, 4 * (3 * n / (np.pi * SLOPE)) ** (1 / 3) + WALL_LV
        for _ in range(22):                               # to ~1e-6 of the bracket; the total is then made exact
            h = (lo + hi) / 2
            tot = np.maximum(base, self._mountain(h)).sum()
            lo, hi = (h, hi) if tot < n else (lo, h)
        hgt = np.maximum(base, self._mountain(hi))
        return hgt * (n / hgt.sum())                      # exact

    def shape(self, n):
        n = float(np.clip(n, 0, self.N[-1]))
        k = int(np.clip(np.searchsorted(self.N, n) - 1, 0, len(self.N) - 2))
        w = (n - self.N[k]) / max(self.N[k + 1] - self.N[k], 1e-9)
        return self.cells, self.H[k] * (1 - w) + self.H[k + 1] * w
