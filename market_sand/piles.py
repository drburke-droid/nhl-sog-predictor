"""Organic sand piles: an irregular mountain for any number of grains, different for every company.

A pile is a height field around its centre, in units of its base radius R:
- an irregular footprint (the outline wobbles with 2-5 lobes);
- a main mountain with the sand-pile profile (1 - d)^1.35 (the Sandglass spec), its summit exact;
- 2-4 shoulders: lesser summits at 0.3-0.65 R, 0.35-0.6 of the height, which push the outline out;
- ridges and gullies running down the flanks (+-22%, deepening downslope);
- grain-scale roughness (+-12%) added by the renderer.
Small piles are nearly round heaps; the lobes, shoulders and ridges grow in as a pile grows (DEV), so a pile
looks poured rather than stamped. R is set so the pile holds exactly its grains:

    shape = Shape("standard_oil_nj")
    R = shape.radius(n)              # base radius (cells) for n grains
    H = shape.field(u, v, n)         # height / (A * R), with u, v in units of R (0 outside the pile)
    shape.extent(n)                  # the farthest the pile reaches from its centre (cells)
    z = A * R * H * LEVEL            # height in cells
"""
import hashlib
import numpy as np

A = 0.85          # summit height = A x R levels (main flanks about 1.1 levels per cell, a ~40 degree slope)
LEVEL = 0.8       # one level of packed grains, in cells
_G = np.linspace(-2.2, 2.2, 221)
_U, _V = np.meshgrid(_G, _G)
_DA = (_G[1] - _G[0]) ** 2


def _soft(d, e):
    """Distance with a rounded tip (no pin-point summits)."""
    return np.sqrt(d * d + e * e) - e


def dev(n):
    """How developed a pile of n grains is, 0 (a few grains: a round heap) to 1 (a mountain)."""
    return float(np.clip((np.log10(max(n, 1.0)) - 0.6) / 4.0, 0.0, 1.0))


class Shape:
    def __init__(self, key):
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        self.k = np.arange(2, 6)
        self.amp = rng.uniform(0.05, 0.12, 4) / np.sqrt(self.k / 2)
        self.ph = rng.uniform(0, 2 * np.pi, 4)
        self.off = rng.normal(0, 0.09, 2)                      # the summit sits a little off-centre
        self.sh = []
        for _ in range(int(rng.integers(2, 5))):
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0.35, 0.6)
            self.sh.append((d * np.cos(a), d * np.sin(a), rng.uniform(0.45, 0.7), rng.uniform(0.5, 0.75)))
        self.rm = (int(rng.integers(3, 6)), rng.uniform(0, 2 * np.pi), int(rng.integers(7, 11)), rng.uniform(0, 2 * np.pi))
        self.nz = [(rng.uniform(4, 9), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0)) for _ in range(5)]
        # volume and reach of the unit pile at a few stages of development
        self._g = np.linspace(0, 1, 6)
        self._K, self._B = [], []
        for g in self._g:
            H = self._field(_U, _V, g)
            self._K.append(H.sum() * _DA)
            self._B.append(np.hypot(_U, _V)[H > 1e-4].max())

    def _field(self, u, v, g):
        gg = 0.3 + 0.7 * g                                     # even a small heap is a little lopsided
        th = np.arctan2(v, u)
        rho = np.hypot(u, v)
        b = 1.0 + gg * (self.amp[:, None] * np.cos(self.k[:, None] * th.ravel()[None, :] + self.ph[:, None])).sum(0).reshape(th.shape)
        ox, oy = self.off * gg
        h = np.clip(1 - _soft(np.hypot(u - ox, v - oy), 0.05) / b, 0, None) ** 1.35
        for sx, sy, sr, sh in self.sh:                         # shoulders, merged smoothly (no creases)
            d = _soft(np.hypot(u - sx * g, v - sy * g), 0.09) / sr
            hs = g * sh * np.clip(1 - d, 0, None) ** 1.35
            h = (h ** 4 + hs ** 4) ** 0.25
        m1, p1, m2, p2 = self.rm
        ridge = 0.6 * np.sin(m1 * th + p1 + 2.5 * rho) + 0.4 * np.sin(m2 * th + p2 - 3.0 * rho)
        noise = sum(w * np.sin(f * (np.cos(a) * u + np.sin(a) * v) + p) for f, a, p, w in self.nz) / 3.0
        down = np.clip(rho, 0, 1.2)                            # gullies deepen downslope
        return np.clip(h * (1 + gg * down * (0.14 * ridge + 0.16 * noise)), 0, None)

    def K(self, n):
        return float(np.interp(dev(n), self._g, self._K))

    def radius(self, n):
        return (max(n, 0.0) / (A * self.K(n))) ** (1 / 3)

    def extent(self, n):
        return self.radius(n) * float(np.interp(dev(n), self._g, self._B))

    def radius_v(self, n):
        n = np.maximum(np.asarray(n, float), 0.0)
        g = np.clip((np.log10(np.maximum(n, 1.0)) - 0.6) / 4.0, 0, 1)
        return (n / (A * np.interp(g, self._g, self._K))) ** (1 / 3)

    def extent_v(self, n):
        n = np.maximum(np.asarray(n, float), 0.0)
        g = np.clip((np.log10(np.maximum(n, 1.0)) - 0.6) / 4.0, 0, 1)
        return self.radius_v(n) * np.interp(g, self._g, self._B)

    def field(self, u, v, n):
        return self._field(u, v, dev(n))
