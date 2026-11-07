"""Sandpile simulation for "Grains of Ancestry".

The map is a grid of 3-px cells. A cell inside a vessel holds a stack of coloured grains (bottom to top).
Sand obeys a simple sandpile rule: where a pile stands two or more grains above its neighbour, the top
grain avalanches onto it; at a one-grain step it slides now and then. So poured sand heaps up where it
lands and spreads across the land.

Each vessel also has an exact gene-pool composition P (shares of each ancestry component). The grains
track it: growth copies grains in proportion to P, arrivals are poured in, displaced sand is lifted off
the surface, and drift recolours grains one by one.
"""
import math
import numpy as np
import cv2

from proj import W, H
import story as S
import vessels as V

CELL = 3
WC, HC = W // CELL, H // CELL
NCELL = WC * HC
import os as _os
TOWERS = _os.environ.get("SAND_TOWERS", "1") != "0"   # uncapped density: sand piles up in towers (SAND_TOWERS=0: capped)
HMAX = 600 if TOWERS else 96
HFULL = 4.0                 # mean grains per cell when a vessel is "full"
K = len(S.KEYS)
KI = {k: i for i, k in enumerate(S.KEYS)}
ARCH = np.array([k in S.ARCHAIC for k in S.KEYS])
ORDER = list(S.VESSELS.keys())
VI = {v: i for i, v in enumerate(ORDER)}
NV = len(ORDER)


# ============================================================== clock (shared with the painter)
CK_S = np.array([c[0] for c in S.CLOCK])
CK_U = np.log(np.array([c[1] for c in S.CLOCK], float) + 200.0)


def ybp_at(s):
    return max(0.0, math.exp(np.interp(s, CK_S, CK_U)) - 200.0)


def s_at(ybp):
    return float(np.interp(-math.log(ybp + 200.0), -CK_U, CK_S))


FILL_EXP = 0.38


def fill_at(ybp):
    """Share of the map's capacity in use, from world population: (people then / people today)^0.38. Sand
    tracks population, so the map starts as a sparse scatter and fills as people multiply; a grain stands
    for ~50 people 200,000 years ago, ~1,000 at the dawn of farming, ~55,000 today (the exponent keeps that
    range within reason: one grain per person-equivalent would leave 2 grains at the start)."""
    return float((world_pop(ybp) / world_pop(0.0)) ** FILL_EXP)


_WP = np.array(S.WORLD_POP, float)


def world_pop(ybp):
    return float(np.exp(np.interp(-ybp, -_WP[:, 0], np.log(_WP[:, 1]))))


def pop_share_hist(ybp):
    """Vessels' shares of world population from the historical tables (interpolated in time)."""
    yrs = np.array(S.POP_YEARS, float)
    vals = np.array([[max(x, 1e-4) for x in S.POP[v]] for v in ORDER])
    t = np.clip(ybp, 0, yrs[0])
    lv = np.array([np.interp(-t, -yrs, np.log(row)) for row in vals])
    p = np.exp(lv)
    return p / p.sum()


def smooth(v):
    v = min(1.0, max(0.0, v))
    return v * v * (3 - 2 * v)


# ============================================================== grid geometry
def build_grid():
    G = V.build(S.VESSELS)
    vid = np.full(NCELL, -1, np.int16)
    cx = (np.arange(WC) + 0.5) * CELL
    cy = (np.arange(HC) + 0.5) * CELL
    meta = {}
    for v in ORDER:
        g = G[v]
        m = np.zeros((H * 2, W * 2), np.uint8)
        ext = [np.round(e * 2 * 16).astype(np.int32) for e, _ in g["rings"]]
        cv2.fillPoly(m, ext, 255, cv2.LINE_AA, shift=4)
        holes = [np.round(h * 2 * 16).astype(np.int32) for _, hs in g["rings"] for h in hs]
        if holes:
            cv2.fillPoly(m, holes, 0, cv2.LINE_AA, shift=4)
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5) / 2.0
        d = dist[np.clip((cy * 2).astype(int), 0, H * 2 - 1)][:, np.clip((cx * 2).astype(int), 0, W * 2 - 1)]
        ok = (d >= 2.9).ravel() & (vid < 0)
        vid[ok] = VI[v]
        meta[v] = dict(anchor=np.array(g["anchor"], float), bbox=g["bbox"], geo=g["geo"],
                       inset=bool(S.VESSELS[v].get("inset")), frame=g.get("frame"),
                       mag=S.VESSELS[v]["inset"][0] if S.VESSELS[v].get("inset") else None,
                       inset_xf=g.get("inset_xf"))
    return vid, _neighbours(vid), meta


def _neighbours(vid):
    """The four neighbours of every cell that belong to the same vessel (-1 elsewhere)."""
    nb = np.full((4, NCELL), -1, np.int64)
    idx = np.arange(NCELL).reshape(HC, WC)
    for d, (dy, dx) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        t = np.full((HC, WC), -1, np.int64)
        ys, xs = slice(max(0, -dy), HC - max(0, dy)), slice(max(0, -dx), WC - max(0, dx))
        yt, xt = slice(max(0, dy), HC - max(0, -dy)), slice(max(0, dx), WC - max(0, -dx))
        t[ys, xs] = idx[yt, xt]
        t = t.ravel()
        same = (t >= 0) & (vid >= 0)
        same[same] = vid[t[same]] == vid[same]
        nb[d] = np.where(same, t, -1)
    return nb


VID, NB, META = build_grid()
VCELLS = [np.where(VID == i)[0] for i in range(NV)]
NCELLS_V = np.array([len(c) for c in VCELLS], float)
VALID = np.where(VID >= 0)[0]


def build_density():
    """Where people live today outside the great cities, per cell: town and city populations (Natural Earth
    populated places, without the megacities, which are towers of their own) spread into halos for the
    surrounding countryside, scaled to mean 1 within each vessel. Inset (magnified) vessels are uniform."""
    import json, os
    from proj import to_px
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ne_10m_populated_places_simple.geojson")
    if not os.path.exists(path):
        import subprocess
        subprocess.run(["curl", "-sS", "-o", path, "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                        "master/geojson/ne_10m_populated_places_simple.geojson"], check=True)
    f = json.load(open(path))["features"]
    lon = np.array([x["properties"]["longitude"] for x in f], float)
    lat = np.array([x["properties"]["latitude"] for x in f], float)
    pop = np.array([max(x["properties"].get("pop_max") or 0, 1000) for x in f], float)
    x, y = to_px(lon, lat)
    cl = [c[1:3] for c in list(S.MEGACITIES) + list(S.CITY_HILLS)]
    cx, cy = to_px(np.array([c[0] for c in cl]), np.array([c[1] for c in cl]))
    from scipy.spatial import cKDTree
    near_city = cKDTree(np.stack([cx, cy], 1)).query(np.stack([x, y], 1))[0] < 6.0
    pop[(pop >= 3e6) | near_city] = 0.0                         # the cities are mountains, not countryside
    ix = np.clip((x / CELL).astype(int), 0, WC - 1); iy = np.clip((y / CELL).astype(int), 0, HC - 1)
    g = np.zeros((HC, WC), np.float32)
    np.add.at(g, (iy, ix), pop)
    parts = [(2.5, 0.35), (5.0, 0.40), (16.0, 0.25)]            # (sigma in cells, weight)
    d = sum(w * cv2.GaussianBlur(g, (0, 0), sg) / max(cv2.GaussianBlur(g, (0, 0), sg).sum(), 1) for sg, w in parts)
    d = d.ravel() ** (1.0 if TOWERS else 0.7)
    out = np.ones(NCELL, np.float32)
    for i, v in enumerate(ORDER):
        c = VCELLS[i]
        if META[v]["inset"] or len(c) == 0:
            continue
        dv = d[c]
        dv = dv / max(dv.mean(), 1e-12)
        dv = np.clip(dv, 0.02, 400.0 if TOWERS else 6.0)   # cap: the densest places hold at most 6x the mean
        out[c] = dv / dv.mean()
    return out


DENS = build_density()


def build_cities():
    """Each city's mountain (the 66 megacities, then every other city of 1 million or more as a hill of
    the same kind): the cells within reach of its widest base (its own vessel's land, and the sea beside
    it), their offset from the summit (in cells), and its fixed irregularities: two to four shoulders
    (lesser summits on the flanks, like a city's satellite towns), a few broad ridges and gullies, and
    grain-scale roughness."""
    from proj import to_px
    out = []
    rng = np.random.default_rng(23)
    for name, lon, lat, mil in list(S.MEGACITIES) + list(S.CITY_HILLS):
        x, y = (float(np.ravel(v)[0]) for v in to_px(lon, lat))
        dv, j = _VALID_TREE.query((x, y))
        if dv > 15.0:                                           # nowhere near any land on the map
            continue
        vi = VID[VALID[j]]
        cells = VCELLS[vi]
        centre = cells[np.argmin(np.hypot(CELL_XY[cells, 0] - x, CELL_XY[cells, 1] - y))]   # nearest land
        reach = float(S.city_height(mil)) / S.CITY_SLOPE * 1.3 + 1.0
        cx, cy = centre % WC, centre // WC
        r = int(np.ceil(reach)) + 1
        gx, gy = np.meshgrid(np.arange(max(cx - r, 0), min(cx + r + 1, WC)), np.arange(max(cy - r, 0), min(cy + r + 1, HC)))
        box = (gy * WC + gx).ravel()
        box = box[np.hypot(*(CELL_XY[box] - CELL_XY[centre]).T) / CELL <= reach]
        land = box[VID[box] == vi]
        sea = box[VID[box] < 0]
        sea = sea[VID[VALID[_VALID_TREE.query(CELL_XY[sea])[1]]] == vi] if len(sea) else sea   # beside its own vessel
        cl = np.concatenate([land, sea])
        off = (CELL_XY[cl] - CELL_XY[centre]) / CELL
        ns = int(rng.integers(2, 5))
        ang = rng.uniform(0, 2 * np.pi, ns)
        sat = np.stack([np.cos(ang), np.sin(ang), rng.uniform(0.3, 0.65, ns),       # direction, distance (x R)
                        rng.uniform(0.35, 0.6, ns), rng.uniform(0.3, 0.45, ns)], 1)  # height (x summit), radius (x R)
        ka = rng.uniform(0, 2 * np.pi, 3)
        wav = np.stack([np.cos(ka), np.sin(ka)], 1) * rng.uniform(5.0, 9.0, (3, 1))  # ridges ~1/2 to 1/4 of R apart
        ph = rng.uniform(0, 2 * np.pi, 3)
        jit = rng.uniform(0.88, 1.12, len(cl))
        out.append(dict(name=name, vi=vi, cells=cl, off=off, dd=np.hypot(*off.T), sat=sat, wav=wav, ph=ph,
                        jit=jit, centre=centre, mil=mil))
    return out


def _flatten(cities):
    """The cities' cells as flat arrays (one entry per city cell), for computing every mountain at once."""
    n = len(cities)
    F = dict(city=np.concatenate([np.full(len(c["cells"]), i) for i, c in enumerate(cities)]),
             cell=np.concatenate([c["cells"] for c in cities]), off=np.concatenate([c["off"] for c in cities]),
             dd=np.concatenate([c["dd"] for c in cities]), jit=np.concatenate([c["jit"] for c in cities]),
             wav=np.stack([c["wav"] for c in cities]), ph=np.stack([c["ph"] for c in cities]),
             mil=np.array([c["mil"] for c in cities]), vi=np.array([c["vi"] for c in cities]),
             centre=np.array([c["centre"] for c in cities]))
    sat = np.zeros((n, 4, 5)); sat[:, :, 4] = 1.0                 # unused shoulders: zero height
    for i, c in enumerate(cities):
        sat[i, :len(c["sat"])] = c["sat"]
    F["sat"] = sat
    F["order"] = np.argsort(F["cell"], kind="stable")            # entries grouped by cell, for the per-cell max
    F["ucell"], F["starts"] = np.unique(F["cell"][F["order"]], return_index=True)
    return F



def build_affinity():
    """Log-weight of each ancestry component at each cell (S.ANCESTRY_GEO): where within its vessel a
    component's grains tend to sit."""
    from proj import to_px
    LA = np.zeros((NCELL, K), np.float32)
    for v, spots in S.ANCESTRY_GEO.items():
        i = VI[v]
        c = VCELLS[i]
        if not len(c):
            continue
        xy = CELL_XY[c]
        xf = META[v]["inset_xf"]
        for comps, lon, lat, sig, st in spots:
            ax, ay = to_px(lon, lat)
            p = np.array([float(ax), float(ay)])
            sp = sig * 5.6
            if xf is not None:                          # spot inside a magnified inset
                c0, c1, k = xf
                p = (p - np.asarray(c0)) * k + np.asarray(c1)
                sp *= k
            g = np.exp(-((xy[:, 0] - p[0]) ** 2 + (xy[:, 1] - p[1]) ** 2) / (2 * sp * sp))
            for comp in comps.split():
                LA[c, KI[comp]] += st * g
    return LA

GMAX = HFULL * NCELLS_V.sum()
CELL_XY = np.stack([(np.arange(NCELL) % WC + 0.5) * CELL, (np.arange(NCELL) // WC + 0.5) * CELL], 1)
from scipy.spatial import cKDTree
_VALID_TREE = cKDTree(CELL_XY[VALID])
CITIES = build_cities()
CF = _flatten(CITIES)
# Sand spilling over the glass: the cells beyond a vessel's wall that a coastal city's mountain spreads onto.
# They join the city's vessel for the mountain alone (VCELLS, VALID and the countryside stay on land).
SPILL = np.zeros(NCELL, bool)
for _c in CITIES:
    _sea = _c["cells"][VID[_c["cells"]] < 0]
    SPILL[_sea] = True
    VID[_sea] = _c["vi"]
NB = _neighbours(VID)
SCELLS = [np.concatenate([VCELLS[i], np.where(SPILL & (VID == i))[0]]) for i in range(NV)]   # land + spill
VTREE = [cKDTree(CELL_XY[c]) for c in VCELLS]
STREE = [cKDTree(CELL_XY[c]) for c in SCELLS]
LA = build_affinity()


# ============================================================== flight paths (map px)
def chaikin(pts, it=4):
    pts = np.asarray(pts, float)
    for _ in range(it):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


class Path:
    """Polyline through the route's waypoints; at the map edge it breaks and re-enters on the far side."""

    def __init__(self, p0, p1, wps):
        from proj import to_px
        if wps:
            x, y = to_px(np.array([w[0] for w in wps], float), np.array([w[1] for w in wps], float))
            raw = [np.array(p0)] + [np.array(v) for v in zip(x, y)] + [np.array(p1)]
            segs, cur = [], [raw[0]]
            for k in range(1, len(raw)):
                if abs(raw[k][0] - raw[k - 1][0]) > W * 0.5:
                    d0 = raw[k - 1] - raw[k - 2]
                    cur.append(raw[k - 1] + d0 / max(np.hypot(*d0), 1e-6) * 140)
                    segs.append(cur)
                    nxt = raw[k + 1] if k + 1 < len(raw) else raw[k]
                    d1 = nxt - raw[k]
                    cur = [raw[k] - d1 / max(np.hypot(*d1), 1e-6) * 140, raw[k]]
                else:
                    cur.append(raw[k])
            segs.append(cur)
            pieces = [chaikin(np.array(sg)) for sg in segs]
        else:
            p0, p1 = np.array(p0, float), np.array(p1, float)
            d = p1 - p0
            dist = np.hypot(*d)
            nrm = np.array([d[1], -d[0]]) / max(dist, 1e-6)
            if nrm[1] > 0:
                nrm = -nrm
            c = (p0 + p1) / 2 + nrm * dist * 0.12
            t = np.linspace(0, 1, 40)[:, None]
            pieces = [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1]
        self.pieces = pieces
        lens = [np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) for p in pieces]
        self.cum = lens
        self.total = sum(l[-1] for l in lens)

    def at(self, u):
        """Points at fractions u (array) of the way; also returns the piece index (for wrap breaks)."""
        d = np.asarray(u, float) * self.total
        out = np.zeros(d.shape + (2,))
        piece = np.zeros(d.shape, np.int32)
        acc = 0.0
        for k, (p, c) in enumerate(zip(self.pieces, self.cum)):
            sel = (d >= acc - 1e-9) & (d <= acc + c[-1] + 1e-9)
            if sel.any():
                dd = d[sel] - acc
                out[sel, 0] = np.interp(dd, c, p[:, 0])
                out[sel, 1] = np.interp(dd, c, p[:, 1])
                piece[sel] = k
            acc += c[-1]
        return out, piece


# ============================================================== the simulation
class Sand:
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.rng_fly = np.random.default_rng(seed + 101)     # flying grains beyond the old fixed-rate draw
        self.s = 0.0
        self.Ht = np.zeros(NCELL, np.float32)
        # each cell's target is a whole number of grains: the density target, dithered once, so sand
        # settles exactly and then lies still (no grains forever hopping between near-empty cells)
        self.U = np.random.default_rng(77).random(NCELL).astype(np.float32)
        self.Hq = np.zeros(NCELL, np.int32)
        self.stack = np.full((NCELL, HMAX), -1, np.int8)
        self.h = np.zeros(NCELL, np.int16)
        self.tchg = np.full(NCELL, -99.0, np.float32)       # last time the top grain changed
        self.counts = np.zeros((NV, K), np.int64)
        self.P = np.zeros((NV, K))
        self.fixed_fill = [None] * NV
        self.growth = [None] * NV
        self.replace = [None] * NV
        self.extinct = [None] * NV
        self.stir = np.zeros(NV)
        self.particles = []             # flying sand: dicts of arrays per emission batch
        self.splash = []                # (x, y, s0, colour)
        self.events = []
        for i, e in enumerate(S.EVENTS):
            e = dict(e, i=i, s0=s_at(e["t0"]), s1=s_at(e["t1"]), started=False, done=False)
            if e["kind"] in ("found", "flow"):
                self.prepare_route(e)
                if e["kind"] == "flow" and e["s1"] - e["s0"] < 2.0:
                    e["s1"] = e["s0"] + 2.0
            if e["kind"] in ("tint", "extinct") and e["s1"] - e["s0"] < 2.0:
                e["s1"] = e["s0"] + 2.0
            self.events.append(e)
        for v, (comp, fill) in S.INITIAL.items():
            i, c = VI[v], KI[comp]
            self.P[i, c] = 1.0
            if ARCH[c]:
                self.fixed_fill[i] = fill * fill_at(200_000)
            n = int(round((self.fixed_fill[i] if ARCH[c] else fill_at(200_000)) * HFULL * len(VCELLS[i])
                          * S.HABITABLE.get(v, 1.0)))
            cells = self.rng.choice(VCELLS[i], size=n)
            for _ in range(HMAX):
                cells = self.add(cells, np.full(len(cells), c, np.int8))
                if len(cells) == 0:
                    break
            self.cell_targets()
            for _ in range(60):
                self.topple(passes=1)
        self.first_seen = {}
        self._share_t = None
        self.Ht = np.zeros(NCELL, np.float32)

    # ---------------------------------------------------------- routes
    def prepare_route(self, e):
        src, dst = VI[e["src"]], VI[e["dst"]]
        a = META[e["src"]]["anchor"]
        b = META[e["dst"]]["anchor"]
        wps = S.ROUTES.get((e["src"], e["dst"]))
        from proj import to_px
        approach = np.array(to_px(*wps[-1]), float) if wps else a
        cells = VCELLS[dst]
        xy = CELL_XY[cells]
        entry = xy[np.argmin(np.hypot(*(xy - approach).T))]
        target = entry * 0.72 + b * 0.28
        if e.get("land"):                                   # a named landing place
            target = np.array(to_px(*e["land"]), float)
        pour = cells[np.argmin(np.hypot(*(xy - target).T))]
        e["pour"] = pour
        e["pour_xy"] = CELL_XY[pour]
        # landing cells ordered by distance from the pour point: arrivals spread outwards over the event
        e["by_dist"] = cells[np.argsort(np.hypot(*(xy - CELL_XY[pour]).T))]
        e["path"] = Path(a, CELL_XY[pour], wps)
        e["fly"] = float(min(4.2, 1.3 + e["path"].total / 520.0))

    # ---------------------------------------------------------- grain primitives (vectorised)
    def add(self, cells, cols):
        """Put grains on top of cells; returns the cells that could not take one this round."""
        cells = np.asarray(cells, np.int64)
        if len(cells) == 0:
            return cells
        uc, first = np.unique(cells, return_index=True)
        rest = np.delete(cells, first)
        cl = np.asarray(cols)[first]
        ok = self.h[uc] < HMAX
        uc, cl = uc[ok], cl[ok]
        self.stack[uc, self.h[uc]] = cl
        self.h[uc] += 1
        self.tchg[uc] = self.s
        np.add.at(self.counts, (VID[uc], cl.astype(np.int64)), 1)
        return rest

    def pop(self, cells):
        cells = np.unique(np.asarray(cells, np.int64))
        cells = cells[self.h[cells] > 0]
        self.h[cells] -= 1
        cl = self.stack[cells, self.h[cells]].astype(np.int64)
        self.stack[cells, self.h[cells]] = -1
        self.tchg[cells] = self.s
        np.add.at(self.counts, (VID[cells], cl), -1)
        return cells, cl

    def topple(self, passes=2):
        for _ in range(passes):
            for d in self.rng.permutation(4):
                nb = NB[d]
                src = np.where(nb >= 0)[0]
                ex = self.h - self.Hq
                diff = ex[src] - ex[nb[src]]
                r = self.rng.random(len(src))
                # only real slopes slide: a lone grain next to an empty cell stays put (no buzzing)
                mv = (diff >= 1.6) & (r < 0.85) & (self.h[src] > 0) \
                    & (self.h[nb[src]] < HMAX - 2) & (~SPILL[nb[src]] | (self.Hq[nb[src]] > 0))
                a = src[mv]
                if len(a) == 0:
                    continue
                b = nb[a]
                self.h[a] -= 1
                g = self.stack[a, self.h[a]].copy()
                self.stack[a, self.h[a]] = -1
                b2 = b
                self.stack[b2, self.h[b2]] = g
                self.h[b2] += 1
                self.tchg[a] = self.s
                self.tchg[b2] = self.s

    def relocate(self, v):
        """Spread sand toward where people live: a few grains above their cell's target move to a cell
        below target nearby (or, failing that, the nearest of a few anywhere in the vessel). Stops
        once every cell holds its target."""
        cells = SCELLS[v]
        hv, q = self.h[cells], self.Hq[cells]
        exc, dfc = cells[hv > q], cells[hv < q]
        if not len(exc) or not len(dfc):
            return
        m = min(len(exc), len(dfc), max(2, int(self.counts[v].sum() * 0.004)))
        src = self.rng.choice(exc, size=m, replace=False)
        dmask = np.zeros(NCELL, bool)
        dmask[dfc] = True
        dst = np.empty(m, np.int64)
        for i, (c, lst) in enumerate(zip(src, STREE[v].query_ball_point(CELL_XY[src], 30.0))):
            cand = cells[lst]
            cand = cand[dmask[cand]]
            if len(cand):
                dst[i] = cand[self.rng.integers(len(cand))]
            else:
                c8 = dfc[self.rng.integers(len(dfc), size=8)]
                dst[i] = c8[np.argmin(np.hypot(*(CELL_XY[c8] - CELL_XY[c]).T))]
        dst, k = np.unique(dst, return_index=True)
        _, cl = self.pop(src[k])
        self.add(dst[:len(cl)], cl.astype(np.int8))

    def occupied(self, v):
        c = VCELLS[v]
        return c[self.h[c] > 0]

    def recolor_sample(self, v, frm, to, n, top_ok=True):
        """Recolour up to n grains of colour frm in vessel v (random grains, any depth)."""
        occ = self.occupied(v)
        if len(occ) == 0 or n <= 0:
            return 0
        done = 0
        for _ in range(4):
            m = min(len(occ) * 2, max(64, n * 8))
            cells = self.rng.choice(occ, size=m)
            lev = (self.rng.random(m) * self.h[cells]).astype(np.int64)
            if not top_ok:
                lev = np.minimum(lev, np.maximum(self.h[cells] - 2, 0))
            hit = self.stack[cells, lev] == frm
            cells, lev = cells[hit], lev[hit]
            key = cells * HMAX + lev
            _, u = np.unique(key, return_index=True)
            cells, lev = cells[u][:n - done], lev[u][:n - done]
            self.stack[cells, lev] = to
            np.add.at(self.counts[v], frm, -len(cells))
            np.add.at(self.counts[v], to, len(cells))
            self.tchg[cells[lev == self.h[cells] - 1]] = self.s
            done += len(cells)
            if done >= n:
                break
        return done

    # ---------------------------------------------------------- populations
    def modern_P(self, v):
        rp = self.replace[v]
        return rp["P_new"] if rp is not None else self.P[v]

    @staticmethod
    def farm_w(ybp):
        w = float(np.clip((10_000 - ybp) / 8_000, 0, 1))
        return w * w * (3 - 2 * w)

    def city_rise(self, h, R, bgf):
        """How far every city's mountain stands above the countryside, per cell, for summit heights h and
        mean base radii R (cells) per city (h = 0: no mountain yet): a cone with slightly concave flanks,
        lesser summits on its shoulders, ridges and gullies that deepen down the slopes, and grain-scale
        roughness; the summit itself stands at exactly h. Where mountains overlap, the higher one counts."""
        rise = np.zeros(NCELL, np.float32)
        on = h > 0
        if not on.any():
            return rise
        every = on.all()
        act = slice(None) if every else on[CF["city"]]           # only the cities that have risen yet
        i, off, cell = CF["city"][act], CF["off"][act], CF["cell"][act]
        Ri = R[i]
        u = CF["dd"][act] / Ri
        f = np.clip(1 - u, 0, 1) ** 1.35
        sat = CF["sat"][i]
        for k in range(4):
            dx, dy, dist, hs, rs = sat[:, k].T
            us = np.hypot(off[:, 0] - dx * dist * Ri, off[:, 1] - dy * dist * Ri) / (rs * Ri)
            f = np.maximum(f, hs * np.clip(1 - us, 0, 1) ** 1.35)
        rid = np.cos(np.einsum("mj,mkj->mk", off / Ri[:, None], CF["wav"][i]) + CF["ph"][i]).mean(1)
        f = f * (1 + 0.22 * rid * np.minimum(u * 2.5, 1)) * np.where(u < 0.08, 1.0, CF["jit"][act])
        val = h[i] * f - bgf[cell]
        if every:                             # the highest per cell
            rise[CF["ucell"]] = np.maximum(np.maximum.reduceat(val[CF["order"]], CF["starts"]), 0.0)
        else:
            np.maximum.at(rise, cell, val.astype(np.float32))
        return rise

    def cell_targets(self):
        """Target pile height per cell. The great cities are mountains of poured sand: summit height for
        population (S.city_height, of the city as big as it was then), flanks at a sand-pile slope, so the
        bigger the city the wider its base; where a region's cities would take more than CITY_SHARE of its
        sand, its mountains steepen to fit. They rise out of the countryside wherever they stand above it.
        The rest of each vessel's grains spread where its people live (evenly before farming, by density
        once settled life takes over), levelled off at BACKGROUND_CAP."""
        ybp = ybp_at(self.s)
        w = self.farm_w(ybp)
        N = self.counts.sum(1).astype(float)
        uf = S.urban_frac(2025.0 - ybp)
        we = (1 - w) + w * DENS[VALID]
        vid = VID[VALID]
        bgf = np.zeros(NCELL, np.float32)
        rise = np.zeros(NCELL, np.float32)
        B = N.copy()
        cap = S.CITY_SHARE * N
        steep = getattr(self, "steep", None)  # per vessel: how much steeper than CITY_SLOPE its mountains stand
        steep = np.ones(NV) if steep is None else steep.copy()   # (carried from frame to frame: it changes slowly)
        for it in range(3):                   # countryside = what the city mountains do not take
            bg = B[vid] * we / np.maximum(np.bincount(vid, we, minlength=NV)[vid], 1e-9)
            for _ in range(6):                # level off at the cap, refill elsewhere
                over = bg > S.BACKGROUND_CAP
                if not over.any():
                    break
                ex = np.bincount(vid[over], bg[over] - S.BACKGROUND_CAP, minlength=NV)
                bg[over] = S.BACKGROUND_CAP
                room = np.where(over, 0.0, we)
                bg += ex[vid] * room / np.maximum(np.bincount(vid, room, minlength=NV)[vid], 1e-9)
            bgf[VALID] = bg
            if uf <= 0:
                break
            for k in range(3):                # steepen (or relax) until the mountains fit their share
                h = S.city_height(CF["mil"] * uf)
                h = np.where(h < 4.0, 0.0, np.maximum(h, bgf[CF["centre"]] + S.CITY_RELIEF * h))  # on a crowded plain, still a hill
                R = np.maximum(h / (S.CITY_SLOPE * steep[CF["vi"]]), S.CITY_MIN_R)
                rise = self.city_rise(h, R, bgf)
                extra = np.bincount(VID[rise > 0], rise[rise > 0], minlength=NV)
                new = np.maximum(steep * np.sqrt(np.maximum(extra, 1e-9) / np.maximum(cap, 1e-9)), 1.0)
                new = np.where(extra > 0, new, steep)
                if np.all(np.abs(new - steep) < 0.03 * steep) or k == 2:
                    break
                steep = new
            B = np.maximum(N - np.minimum(extra, cap), 0.0)
        if uf > 0:                            # settle the split exactly: countryside + mountains = the sand there is
            Bf = np.maximum(N - extra, 0.25 * N)
            k = Bf / np.maximum(np.bincount(vid, bgf[VALID], minlength=NV), 1e-9)
            bgf[VALID] *= np.minimum(k, 1.1)[vid]
        self.steep, self.bgf = steep, bgf
        H = np.minimum(bgf + rise, HMAX - 6)
        self.Ht = H
        self.Hq = np.floor(H + self.U).astype(np.int32)

    def base_grains(self, v, ybp):
        """Sand in proportion to population: spread by habitable area before farming, by the historical
        population tables after, blending between 10,000 years ago and 1 CE. Small peoples keep a
        minimum visible amount."""
        if self.fixed_fill[v] is not None:
            return self.fixed_fill[v] * HFULL * NCELLS_V[v] * S.HABITABLE.get(ORDER[v], 1.0)
        area = fill_at(ybp) * HFULL * NCELLS_V[v] * S.HABITABLE.get(ORDER[v], 1.0)
        w = self.farm_w(ybp)
        if w > 0:
            if self._share_t != ybp:
                self._share = pop_share_hist(ybp); self._share_t = ybp
            hist = fill_at(ybp) * GMAX * self._share[v]
        else:
            hist = area
        g = area * (1 - w) + hist * w
        if w > 0 and META[ORDER[v]]["inset"]:          # tiny island peoples stay visible
            g = max(g, S.MIN_VISIBLE * NCELLS_V[v])
        return g

    def target(self, v):
        ybp = ybp_at(self.s)
        base = self.base_grains(v, ybp)
        ex = self.extinct[v]
        if ex is not None:
            base *= max(0.0, 1 - (self.s - ex[0]) / max(ex[1] - ex[0], 0.5))
            return base
        gr = self.growth[v]
        if gr is None:
            return base
        if gr["t0"] is None:
            modern = float(gr["landed"])
        else:
            modern = min(base, gr["n0"] * math.exp(0.85 * (self.s - gr["t0"])))
        rp = self.replace[v]
        if rp is not None:
            A = rp["A0"] * max(0.0, 1 - (self.s - rp["s0"]) / (rp["s1"] - rp["s0"]))
            u = modern / max(modern + A, 1e-6)
            self.P[v] = rp["P_new"] * u + rp["P_old"] * (1 - u)
            if A <= 0 and modern >= base - 1:
                self.replace[v] = None
                self.P[v] = rp["P_new"].copy()
            return modern + A
        return modern

    def reconcile(self, v, dt):
        N = int(self.counts[v].sum())
        T = self.target(v)
        P = self.P[v]
        occ = self.occupied(v)
        if N < T - 1 and len(occ):
            # growth: new grains are copies, placed on piles of their own colour where possible
            k = int(min(T - N, max(12, 0.05 * N)))
            want = np.maximum(P * (N + k) - self.counts[v], 0)
            if want.sum() <= 0:
                want = P.copy()
            cols = self.rng.choice(K, size=k, p=want / want.sum())
            room = occ[self.h[occ] < HMAX - 2]
            crowded = len(room) < len(occ) * 0.5
            if len(room) == 0:
                room = occ
            tops = self.stack[room, self.h[room] - 1]
            cells = np.empty(k, np.int64)
            for c in np.unique(cols):
                sel = np.where(cols == c)[0]
                same = room[tops == c]
                pool = same if len(same) and self.rng.random() < 0.8 else room
                pw = (np.maximum(self.Hq[pool] - self.h[pool], 0) + 0.1) * np.exp(LA[pool, c])
                cells[sel] = self.rng.choice(pool, size=len(sel), p=pw / pw.sum())
            # some grains hop to a nearby spot: sand crosses small gaps (island chains) and fills out
            hop = np.where(self.rng.random(k) < (0.7 if crowded else 0.12))[0]
            if len(hop):
                near = VTREE[v].query_ball_point(CELL_XY[cells[hop]], 70.0 if crowded else 16.0)
                for j, lst in zip(hop, near):
                    if lst:
                        cells[j] = VCELLS[v][lst[int(self.rng.integers(len(lst)))]]
            self.add(cells, cols.astype(np.int8))
        elif N > T + 0.5 and len(occ):
            k = int(min(N - T, max(12, 0.05 * N)))
            if self.extinct[v] is not None:
                self.pop(self.rng.choice(occ, size=k))
            elif self.counts[v].sum() and (self.counts[v] - P * N).max() <= 0.75:
                ex = np.maximum(self.h[occ] - self.Hq[occ], 0) + 0.02
                self.pop(self.rng.choice(occ, size=min(k, len(occ)), p=ex / ex.sum()))
            else:
                ex = self.counts[v] - P * N
                tops = self.stack[occ, self.h[occ] - 1]
                over = occ[ex[tops] > 0]
                if len(over):
                    self.pop(self.rng.choice(over, size=min(k, len(over))))
        # composition: displaced sand is lifted off the top; otherwise grains recolour (drift, mixing)
        N = int(self.counts[v].sum())
        if N:
            d = self.counts[v] - P * N
            budget = max(2, int(N * 0.012 * dt * 30))
            for _ in range(3):
                hi, lo = int(np.argmax(d)), int(np.argmin(d))
                if d[hi] <= 0.75 or d[lo] >= -0.75 or budget <= 0:
                    break
                n = int(min(d[hi], -d[lo], budget))
                if self.replace[v] is not None and ARCH[hi]:
                    occ = self.occupied(v)
                    tops = self.stack[occ, self.h[occ] - 1]
                    cand = occ[tops == hi]
                    if len(cand):
                        self.pop(self.rng.choice(cand, size=min(n, len(cand))))
                    else:
                        self.recolor_sample(v, hi, lo, n, top_ok=False)
                else:
                    self.recolor_sample(v, hi, lo, n)
                budget -= n
                d = self.counts[v] - P * int(self.counts[v].sum())
        # sorting: random pairs of cells swap their top grains when each grain is more at home in the
        # other cell (within-region ancestry geography); totals are unchanged
        occ = self.occupied(v)
        if ORDER[v] in S.ANCESTRY_GEO and len(occ) > 8:
            m = min(len(occ) // 2 * 2, max(16, int(len(occ) * 0.04)) // 2 * 2)
            pick = self.rng.choice(occ, size=m, replace=False)
            a, b = pick[: m // 2], pick[m // 2:]
            ca = self.stack[a, self.h[a] - 1].astype(np.int64)
            cb = self.stack[b, self.h[b] - 1].astype(np.int64)
            gain = LA[b, ca] + LA[a, cb] - LA[a, ca] - LA[b, cb]
            sw = (gain > 0.02) & (ca != cb)
            a, b = a[sw], b[sw]
            self.stack[a, self.h[a] - 1] = cb[sw]
            self.stack[b, self.h[b] - 1] = ca[sw]
        # stirring: a top grain swaps with one deeper in its pile, or with a neighbour's top
        occ = self.occupied(v)
        if len(occ) > 4:
            m = int(len(occ) * (0.004 + 0.01 * self.stir[v]))
            if m:
                c = self.rng.choice(occ, size=m)
                c = np.unique(c)
                t = self.h[c] - 1
                lv = (self.rng.random(len(c)) * self.h[c]).astype(np.int64)
                a, b = self.stack[c, t].copy(), self.stack[c, lv].copy()
                self.stack[c, t], self.stack[c, lv] = b, a
                self.tchg[c[a != b]] = self.s
        self.stir[v] *= math.exp(-dt / 3.0)

    # ---------------------------------------------------------- events
    def start(self, e):
        e["started"] = True
        k = e["kind"]
        if k == "found":
            src, dst = VI[e["src"]], VI[e["dst"]]
            okc = np.array([(not ARCH[c]) and (e.get("only") is None or S.KEYS[c] in e["only"]) for c in range(K)])
            q = self.counts[src] * okc
            if q.sum() < 2 * e["n"]:
                q = self.modern_P(src) * okc
            cols = self.rng.choice(K, size=e["n"], p=q / q.sum())
            P = np.bincount(cols, minlength=K).astype(float)
            arch = self.modern_P(src) * ARCH
            P = P / P.sum() * (1 - arch.sum()) + arch
            if self.counts[dst].sum() > 0 and self.P[dst][ARCH].sum() > 0.5:
                keep = e.get("keep", 0.0)
                until = e.get("until") or e["t0"] * 0.85
                self.replace[dst] = dict(P_old=self.P[dst].copy(), P_new=P * (1 - keep) + self.P[dst] * keep,
                                         A0=float(self.counts[dst].sum()), s0=self.s,
                                         s1=max(s_at(until), self.s + 3.5))
                self.fixed_fill[dst] = None
            else:
                self.P[dst] = P
                self.fixed_fill[dst] = None
            self.extinct[dst] = None          # newcomers end any extinction still under way here
            self.growth[dst] = dict(n0=e["n"], t0=None, landed=0)
            e["cols"] = cols
            e["landed"] = 0
            self.emit(e, np.repeat(cols, 12), size=1.2, spread=0.7,         # only the founders themselves fly
                      keep=np.arange(0, 12 * len(cols), 12))
        elif k == "flow":
            src, dst = VI[e["src"]], VI[e["dst"]]
            e["Psrc"] = S.restrict(self.modern_P(src), e.get("only"))
            e["M"] = e["f"] * max(self.counts[dst].sum(), 10)      # grains poured in (and as many displaced)
            e["arr0"], e["arr1"] = e["s0"] + e["fly"], e["s1"] + e["fly"]
            e["poured"] = 0.0
            e["emit_acc"] = 0.0
        elif k == "tint":
            e["k"] = math.log(1 / 0.03) / max(e["s1"] - e["s0"], 1.0)
        elif k == "extinct":
            self.extinct[VI[e["jar"]]] = (e["s0"], e["s1"])

    def emit(self, e, cols, size=1.0, spread=1.0, keep=None, extra=None):
        """Launch flying grains from random spots in the source's sand toward the landing point. Every flying
        grain is one that lands (a flow's stream carries exactly the grains it pours in). The random draws are
        the same as when streams were drawn at a fixed rate (cols: that many), so the rest of the simulation is
        unchanged; keep: which of them really fly; extra: colours of more grains to fly (drawn separately)."""
        n = len(cols)
        src = VI[e["src"]]
        occ = self.occupied(src)
        start = CELL_XY[self.rng.choice(occ, size=n)] if len(occ) else np.tile(META[e["src"]]["anchor"], (n, 1))
        d = dict(ev=e["i"], col=np.asarray(cols, np.int64), t0=self.s + self.rng.random(n) * 0.25 * spread,
                 dur=e["fly"] * (0.88 + 0.24 * self.rng.random(n)), start=start,
                 end=e["pour_xy"] + self.rng.normal(0, 5.0, (n, 2)), size=size * (0.7 + 0.6 * self.rng.random(n)),
                 ph=self.rng.random((n, 6)) * 2 * math.pi, lat=self.rng.normal(0, 1, n),
                 amp=0.6 + 0.8 * self.rng.random(n), path=e["path"])
        if keep is not None:
            for k in ("col", "t0", "dur", "start", "end", "size", "ph", "lat", "amp"):
                d[k] = d[k][keep]
        if extra is not None and len(extra):
            r, m = self.rng_fly, len(extra)
            st = CELL_XY[r.choice(occ, size=m)] if len(occ) else np.tile(META[e["src"]]["anchor"], (m, 1))
            more = dict(col=np.asarray(extra, np.int64), t0=self.s + r.random(m) * 0.25 * spread,
                        dur=e["fly"] * (0.88 + 0.24 * r.random(m)), start=st,
                        end=e["pour_xy"] + r.normal(0, 5.0, (m, 2)), size=size * (0.7 + 0.6 * r.random(m)),
                        ph=r.random((m, 6)) * 2 * math.pi, lat=r.normal(0, 1, m), amp=0.6 + 0.8 * r.random(m))
            for k, v in more.items():
                d[k] = np.concatenate([d[k], v])
        if len(d["col"]):
            self.particles.append(d)

    def run_events(self, dt):
        for e in self.events:
            if e["done"]:
                continue
            if not e["started"]:
                if self.s < e["s0"]:
                    continue
                self.start(e)
            k = e["kind"]
            dst = VI.get(e.get("dst"))
            if k == "found":
                if self.s >= e["s0"] + e["fly"] * 0.95:
                    cells = e["by_dist"][:max(3, e["n"] // 2)]
                    cells = self.rng.choice(cells, size=e["n"])
                    self.add(cells, np.asarray(e["cols"], np.int8))
                    self.growth[dst]["landed"] = e["n"]
                    self.growth[dst]["t0"] = self.s
                    self.splash.append((e["pour_xy"], self.s, 1.0))
                    self.stir[dst] = 2.0
                    e["done"] = True
            elif k == "flow":
                if self.s <= e["s1"]:
                    # the visible stream: exactly the grains this flow pours in, one flying grain per landing grain
                    # (the old fixed-rate draw is still made, so the simulation's random stream is unchanged)
                    rate = float(np.clip(e["M"] / max(e["s1"] - e["s0"], 0.5) * 0.5, 120, 420))
                    e["emit_acc"] += rate * dt
                    n = int(e["emit_acc"])
                    e["fly_acc"] = e.get("fly_acc", 0.0) + e["M"] / max(e["s1"] - e["s0"], 0.5) * dt
                    want = int(e["fly_acc"])
                    e["fly_acc"] -= want
                    p_src = e["Psrc"] / e["Psrc"].sum()
                    if n:
                        e["emit_acc"] -= n
                        cols = self.rng.choice(K, size=n, p=p_src)
                        k = min(n, want)
                        self.emit(e, cols, keep=np.arange(k),
                                  extra=self.rng_fly.choice(K, size=want - k, p=p_src) if want > k else None)
                    elif want:
                        self.emit(e, np.zeros(0, np.int64), extra=self.rng_fly.choice(K, size=want, p=p_src))
                if e["arr0"] <= self.s:
                    span = max(e["arr1"] - e["arr0"], 0.5)
                    prog = min(1.0, (self.s - e["arr0"]) / span)
                    want = e["M"] * prog
                    m = int(want - e["poured"])
                    if m > 0:
                        # composition update for this slice of the flow
                        f0 = e["poured"] / e["M"]; f1 = (e["poured"] + m) / e["M"]
                        a0 = 1 - (1 - e["f"]) ** f0; a1 = 1 - (1 - e["f"]) ** f1
                        w = (a1 - a0) / max(1 - a0, 1e-9)
                        self.P[dst] = self.P[dst] * (1 - w) + e["Psrc"] * w
                        if self.replace[dst] is not None:
                            rp = self.replace[dst]
                            rp["P_new"] = rp["P_new"] * (1 - w) + e["Psrc"] * w
                        # pour: around the landing point, the circle widening as the flow goes on
                        nc = len(e["by_dist"])
                        sig = 6 + prog * nc * 0.55 * e.get("spread", 1.0)
                        idx = np.minimum((np.abs(self.rng.normal(0, sig, m))).astype(np.int64), nc - 1)
                        cols = self.rng.choice(K, size=m, p=e["Psrc"] / e["Psrc"].sum())
                        rest = self.add(e["by_dist"][idx], cols.astype(np.int8))
                        if len(rest):
                            self.add(self.rng.choice(e["by_dist"][:max(50, nc // 3)], size=len(rest)),
                                     cols[:len(rest)].astype(np.int8))
                        # as many grains make way elsewhere (the colours now over their share)
                        N = self.counts[dst].sum()
                        ex = self.counts[dst] - self.P[dst] * N
                        occ = self.occupied(dst)
                        far = occ[np.hypot(*(CELL_XY[occ] - e["pour_xy"]).T) > 25]
                        if len(far):
                            tops = self.stack[far, self.h[far] - 1]
                            over = far[ex[tops] > 0]
                            if len(over):
                                self.pop(self.rng.choice(over, size=min(m, len(over))))
                        e["poured"] += m
                        self.stir[dst] = min(self.stir[dst] + 0.15, 3.0)
                    if prog >= 1.0:
                        e["done"] = True
            elif k == "tint":
                v = VI[e["jar"]]
                tgt = KI[e["target"]]
                fac = math.exp(-e["k"] * dt)
                for sname in e["sources"]:
                    c = KI[sname]
                    m = self.P[v, c] if self.s >= e["s1"] else self.P[v, c] * (1 - fac)
                    self.P[v, c] -= m
                    self.P[v, tgt] += m
                e["done"] = self.s >= e["s1"]
            elif k == "extinct":
                v = VI[e["jar"]]
                if self.extinct[v] is None:                 # cleared by newcomers
                    e["done"] = True
                elif self.s >= e["s1"] + 1.0 and self.counts[v].sum() == 0:
                    self.extinct[v] = None
                    e["done"] = True

    def step(self, s_new):
        dt = s_new - self.s
        self.s = s_new
        self.run_events(dt)
        self.cell_targets()
        for v in range(NV):
            if self.counts[v].sum() or self.growth[v] is not None:
                self.reconcile(v, dt)
                self.relocate(v)
        self.topple(passes=2)
        self.particles = [p for p in self.particles if (p["t0"] + p["dur"]).max() + 0.3 > self.s]
        self.splash = [sp for sp in self.splash if self.s - sp[1] < 1.5]
        for c in range(K):
            if c not in self.first_seen and self.counts[:, c].sum() > 0:
                self.first_seen[c] = self.s

    # ---------------------------------------------------------- views for the painter
    def top_colors(self):
        top = np.full(NCELL, -1, np.int64)
        occ = self.h > 0
        top[occ] = self.stack[occ, self.h[occ] - 1]
        return top
