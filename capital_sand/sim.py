"""The table, frame by frame: every person's pile follows their fortune; new balls fall in from above,
departing balls roll across the table and off its edge, inheritances fly from one pile to another, and
philanthropy lifts away.

    sim = Table(people, grid)          # people: list of dict(key, pile=piles.Pile)
    for f in range(nframes):
        sim.step(f, counts, transfers)  # counts: balls per person this frame; transfers: [(i_from, i_to, n)]
    rec, fl = sim.records(), sim.flights()

Records are balls while they can be seen: (cell, level, first frame, last frame + 1). A ball buried under
others is not drawn (its record closes) and reopens if the pile shrinks back to it, so the renderer only
ever draws the surface. Flights carry the balls in motion; FLY_FRAC of the balls arriving or leaving get one
(the rest simply appear or vanish where the stream hides them), so the streams read true without millions of
flights.
"""
import numpy as np

RING = 64                # most levels of one column visible at once (deeper ones are hidden)
FLY_FRAC = 0.35          # share of arriving/departing balls drawn in flight
FALL_S = 0.9             # seconds a falling ball takes from above the frame to the pile


def _expand(cells, start, k):
    """Cells repeated k times with levels start, start+1, ..., start+k-1 (vectorised ranges)."""
    tot = int(k.sum())
    off = np.arange(tot) - np.repeat(np.cumsum(k) - k, k)
    return np.repeat(cells, k), np.repeat(start, k) + off


class Table:
    def __init__(self, people, grid, fps=30, seed=5, wall_cells=None):
        self.GW, self.GH = grid
        n = self.GW * self.GH
        self.people = people
        self.fps = fps
        self.rng = np.random.default_rng(seed)
        self.h = np.zeros(n, np.int32)              # balls per cell (levels)
        self.lo = np.zeros(n, np.int32)             # lowest visible level per cell
        self.U = self.rng.uniform(0, 1, n).astype(np.float32)   # fixed dither: a cell's fractional height rounds the same way every frame
        self.slot = -np.ones(n, np.int64)           # active cell -> row in the open-record ring
        self.ring = np.zeros((0, RING), np.int64)
        self.R = dict(cell=[], lev=[], tb=[], td=[])  # records (lists of numpy chunks)
        self._rec_n = 0
        self._td_fix = []                           # (record ids, frame) closing records
        self.F = dict(kind=[], cell=[], lev=[], t0=[], t1=[], a=[], b=[])
        self.owner = -np.ones(n, np.int32)
        self.wall = np.zeros(n, bool) if wall_cells is None else wall_cells

    # ------------------------------------------------------------------ shape of the whole table
    def target(self, counts):
        H = np.zeros(self.GW * self.GH, np.float32)
        own = -np.ones(self.GW * self.GH, np.int32)
        for i, (p, c) in enumerate(zip(self.people, counts)):
            if c <= 0.5:
                continue
            cells, hh = p["pile"].shape(c)
            m = hh > H[cells]
            H[cells[m]] = hh[m]; own[cells[m]] = i
        return np.floor(H + self.U).astype(np.int32), own

    def _slots(self, cells):
        new = np.unique(cells[self.slot[cells] < 0])
        if len(new):
            start = len(self.ring)
            self.ring = np.vstack([self.ring, -np.ones((len(new), RING), np.int64)])
            self.slot[new] = np.arange(start, start + len(new))
        return self.slot[cells]

    def _open(self, cells, levs, f):
        k = len(cells)
        ids = np.arange(self._rec_n, self._rec_n + k)
        self._rec_n += k
        self.R["cell"].append(cells.astype(np.int32)); self.R["lev"].append(levs.astype(np.int16))
        self.R["tb"].append(np.full(k, f, np.int32)); self.R["td"].append(np.full(k, -1, np.int32))
        rows = self._slots(cells)                    # (may grow self.ring: look it up after)
        self.ring[rows, levs % RING] = ids
        return ids

    def _close(self, cells, levs, f):
        ids = self.ring[self.slot[cells], levs % RING]
        self._td_fix.append((ids, f))

    def step(self, f, counts, transfers=()):
        Hq, own = self.target(counts)
        nb = np.minimum.reduce([np.roll(Hq.reshape(self.GH, self.GW), s, a) for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))]).ravel()
        lo = np.clip(np.minimum(nb, Hq) - 1, 0, None)
        lo = np.maximum(lo, Hq - RING + 1).clip(0)
        lo = np.where(self.wall, np.maximum(lo, 0), lo)
        ch = np.nonzero((Hq != self.h) | (lo != self.lo))[0]
        if len(ch):
            ho, lo_o, hn, lo_n = self.h[ch], self.lo[ch], Hq[ch], lo[ch]
            # levels leaving view: old window minus new window (two ranges), entering: new minus old
            def ranges(a0, a1, b0, b1):          # [a0,a1) minus [b0,b1) as two ranges
                return (a0, np.minimum(a1, np.maximum(b0, a0))), (np.maximum(a0, np.minimum(b1, a1)), a1)
            for (s0, s1), opening in ((r, False) for r in ranges(lo_o, ho, lo_n, hn)):
                k = np.maximum(s1 - s0, 0)
                if k.sum():
                    cells, levs = _expand(ch, s0, k)
                    self._close(cells, levs, f)
                    gone = levs >= np.repeat(hn, k)          # removed from the pile, not just buried
                    if gone.any():
                        self._depart(cells[gone], levs[gone], f, own, transfers)
            for (s0, s1) in ranges(lo_n, hn, lo_o, ho):
                k = np.maximum(s1 - s0, 0)
                if k.sum():
                    cells, levs = _expand(ch, s0, k)
                    self._open(cells, levs, f)
                    new = levs >= np.repeat(ho, k)           # newly added balls (not uncovered ones)
                    if new.any():
                        self._arrive(cells[new], levs[new], f, own)
        self.h, self.lo, self.owner = Hq, lo, own

    # ------------------------------------------------------------------ flights
    def _sample(self, k):
        return self.rng.random(k) < FLY_FRAC

    def _arrive(self, cells, levs, f, own):
        m = self._sample(len(cells))
        if not m.any():
            return
        c, l = cells[m], levs[m]
        dur = FALL_S * self.fps * self.rng.uniform(0.8, 1.2, len(c))
        self._flight(0, c, l, f - dur, f, own[c], np.zeros(len(c)))

    def _depart(self, cells, levs, f, own, transfers):
        m = self._sample(len(cells))
        if not m.any():
            return
        c, l = cells[m], levs[m]
        who = self.owner[c]
        kind = np.ones(len(c), np.int8)               # 1 = roll off the table
        dest = -np.ones(len(c), np.int32)
        for i_from, i_to, share, k in transfers:      # k: 2 = to another pile (inheritance), 3 = lift away (philanthropy)
            sel = (who == i_from) & (self.rng.random(len(c)) < share)
            kind[sel] = k; dest[sel] = i_to
        dur = np.where(kind == 1, self.rng.uniform(2.0, 3.5, len(c)), np.where(kind == 2, 1.6, 2.4)) * self.fps
        self._flight(kind, c, l, np.full(len(c), f, float), f + dur, who, dest)

    def _flight(self, kind, cells, levs, t0, t1, a, b):
        k = len(cells)
        self.F["kind"].append(np.broadcast_to(np.asarray(kind, np.int8), (k,)).copy())
        self.F["cell"].append(cells.astype(np.int32)); self.F["lev"].append(levs.astype(np.int16))
        full = lambda x, dt: np.broadcast_to(np.asarray(x, dt), (k,)).copy()
        self.F["t0"].append(full(t0, np.float32)); self.F["t1"].append(full(t1, np.float32))
        self.F["a"].append(full(a, np.int32)); self.F["b"].append(full(b, np.int32))

    # ------------------------------------------------------------------ results
    def records(self, last_frame):
        R = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.R.items()}
        for ids, f in self._td_fix:
            R["td"][ids] = f
        R["td"][R["td"] < 0] = last_frame + 1
        keep = R["td"] > R["tb"]
        return {k: v[keep] for k, v in R.items()}

    def flights(self):
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in self.F.items()}
