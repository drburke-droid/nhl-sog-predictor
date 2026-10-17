"""Coverage maps: every road lights up the moment the photon first drives it.

Each road's length is binned to the pixel under its midpoint. Roads are sorted by when they were first
driven, so a frame at distance D just adds the roads first driven since the previous frame. A pixel's
brightness is the share of its road length driven so far, on top of a dim map of all roads.
"""
import numpy as np


class Coverage:
    def __init__(self, px, py, km, first, w, h):
        """px, py: pixel of each road (ints, may be off-screen); km: road length; first: km driven when first driven."""
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pix = (py * w + px)[ok].astype(np.int64)
        km, first = km[ok].astype(np.float64), first[ok].astype(np.float64)
        o = np.argsort(first, kind="stable")
        self.pix, self.km, self.first = pix[o], km[o], first[o]
        self.w, self.h = w, h
        self.total = np.bincount(pix, weights=km, minlength=w * h).reshape(h, w)
        self.done = np.zeros(w * h)
        self.i = 0

    def advance(self, D):
        """Add every road first driven at or before distance D; returns the roads added (their pixels)."""
        j = int(np.searchsorted(self.first, D, side="right"))
        if j > self.i:
            sl = slice(self.i, j)
            self.done += np.bincount(self.pix[sl], weights=self.km[sl], minlength=self.w * self.h)
            new = self.pix[sl]
            self.i = j
            return new
        return self.pix[:0]

    def fraction(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(self.total > 0, self.done.reshape(self.h, self.w) / self.total, 0.0)
        return f

    def density(self, gamma=0.45):
        t = self.total
        return (t / max(np.percentile(t[t > 0], 99), 1e-9)) ** gamma
