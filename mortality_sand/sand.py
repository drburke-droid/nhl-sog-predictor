"""Sandpile simulation for "2,000 years of catastrophe".

The map is a grid of story.CELL-px cells. A cell inside a vessel holds a stack of grains (bottom to top); one grain
is story.GRAIN people. Sand obeys a simple sandpile rule: where a pile stands two or more grains above its
neighbour, the top grain avalanches onto it. Each vessel's sand follows its population year by year
(build_data.py): new grains swell in where people live, the great cities rise as mountains of poured sand.

Excess deaths are lifted: every source in data/lifts.json (a catastrophe before 1950, a country's year
above its normal death rate after) takes its share of grains, year by year, off the top of the piles
where it struck (more from crowded places, in proportion to the people there). Each lifted grain is
handed to the painter as a particle that rises out of the frame, coloured by cause.
"""
import json, math, os
import numpy as np
import cv2

from proj import W, H, to_px
import story as S
import vessels as V

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CELL = S.CELL
GS = S.GS                        # grain size relative to the migration film's
WC, HC = W // CELL, H // CELL
NCELL = WC * HC
HMAX = 600
ORDER = list(S.VESSELS.keys())
VI = {v: i for i, v in enumerate(ORDER)}
NV = len(ORDER)
CI = {k: i for i, k in enumerate(S.CAUSE_KEYS)}
TOWERS = True                    # uncapped density: crowded places pile up
GRAIN_DZ = 0.55 * GS             # height of one grain level in a pile (map px)
PX_PER_DEG = 5.6                 # roughly, on this map: spreads given in degrees -> map px


# ============================================================== clock (shared with the painter)
CK_S = np.array(S.CLOCK["s"], float)
CK_Y = np.array(S.CLOCK["year"], float)


def year_at(s):
    """Calendar year (fractional) at video second s: year 1.0 until the title clears, 2026.0 at the end."""
    return float(np.interp(s, CK_S, CK_Y))


def s_at(year):
    return float(np.interp(year, CK_Y, CK_S))


def smooth(v):
    v = min(1.0, max(0.0, v))
    return v * v * (3 - 2 * v)


# ============================================================== population series
_SER = np.load(os.path.join(DATA, "series.npz"))
_YRS = _SER["years"].astype(float)
_POP = _SER["pop"]
assert list(_SER["vessels"]) == ORDER


def grains_at(year):
    """Each vessel's sand (grains) at a fractional calendar year: its population / GRAIN."""
    y = min(max(year, _YRS[0]), _YRS[-1])
    k = int(min(max(math.floor(y) - _YRS[0], 0), len(_YRS) - 2))
    w = y - _YRS[k]
    return (_POP[:, k] * (1 - w) + _POP[:, k + 1] * w) / S.GRAIN


# ============================================================== grid geometry
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
        ok = (d >= max(2.2, 2.9 * GS)).ravel() & (vid < 0)
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
    parts = [(7.5 / CELL, 0.35), (15.0 / CELL, 0.40), (48.0 / CELL, 0.25)]   # (sigma in cells, weight)
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
    for hi, (name, lon, lat, mil) in enumerate(list(S.MEGACITIES) + list(S.CITY_HILLS)):
        x, y = (float(np.ravel(v)[0]) for v in to_px(lon, lat))
        dv, j = _VALID_TREE.query((x, y))
        if dv > 15.0:                                           # nowhere near any land on the map
            continue
        vi = VID[VALID[j]]
        cells = VCELLS[vi]
        centre = cells[np.argmin(np.hypot(CELL_XY[cells, 0] - x, CELL_XY[cells, 1] - y))]   # nearest land
        reach = float(S.city_height(max(mil, CITY_M[hi].max()))) / S.CITY_SLOPE * 1.3 + 1.0   # its largest ever
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
        out.append(dict(name=name, hi=hi, vi=vi, cells=cl, off=off, dd=np.hypot(*off.T), sat=sat, wav=wav, ph=ph,
                        jit=jit, centre=centre, mil=mil))
    return out


def _flatten(cities):
    """The cities' cells as flat arrays (one entry per city cell), for computing every mountain at once."""
    n = len(cities)
    F = dict(city=np.concatenate([np.full(len(c["cells"]), i) for i, c in enumerate(cities)]),
             cell=np.concatenate([c["cells"] for c in cities]), off=np.concatenate([c["off"] for c in cities]),
             dd=np.concatenate([c["dd"] for c in cities]), jit=np.concatenate([c["jit"] for c in cities]),
             wav=np.stack([c["wav"] for c in cities]), ph=np.stack([c["ph"] for c in cities]),
             mil=np.array([c["mil"] for c in cities]), hi=np.array([c["hi"] for c in cities]),
             vi=np.array([c["vi"] for c in cities]),
             centre=np.array([c["centre"] for c in cities]))
    sat = np.zeros((n, 4, 5)); sat[:, :, 4] = 1.0                 # unused shoulders: zero height
    for i, c in enumerate(cities):
        sat[i, :len(c["sat"])] = c["sat"]
    F["sat"] = sat
    F["order"] = np.argsort(F["cell"], kind="stable")            # entries grouped by cell, for the per-cell max
    F["ucell"], F["starts"] = np.unique(F["cell"][F["order"]], return_index=True)
    return F



CELL_XY = np.stack([(np.arange(NCELL) % WC + 0.5) * CELL, (np.arange(NCELL) // WC + 0.5) * CELL], 1)
from scipy.spatial import cKDTree
_VALID_TREE = cKDTree(CELL_XY[VALID])
import cityhist
CITY_YEARS, CITY_M, CITY_INFO = cityhist.histories(list(S.MEGACITIES) + list(S.CITY_HILLS))   # millions by year
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
STREE = [cKDTree(CELL_XY[c]) for c in SCELLS]
RELOC_K = 40
SNBK = [t.query(CELL_XY[c], k=min(RELOC_K, len(c)))[1].reshape(len(c), -1) for t, c in zip(STREE, SCELLS)]


def build_countries():
    """The country (ISO3) under every vessel cell, for placing a country's excess deaths where its people
    live; and each country's centre on the map."""
    path = os.path.join(DATA, "country_cells.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["cc"], z["names"].tolist(), {k: tuple(v) for k, v in z["centre"].item().items()}
    countries = V.load_countries()
    names = sorted(countries)
    m = np.full((HC * 2, WC * 2), -1, np.int32)
    centre = {}
    order = sorted(names, key=lambda n: -countries[n].area)          # small countries drawn last, on top
    for n in order:
        g = countries[n]
        polys = V.split_cut(g)
        rings = []
        for p in polys:
            xy = np.array(p.exterior.coords)
            x, y = to_px(np.minimum(xy[:, 0], 189.999), np.maximum(xy[:, 1], -89.9))
            rings.append(np.round(np.stack([x, y], 1) / CELL * 2 * 16).astype(np.int32))
        if rings:
            cv2.fillPoly(m, rings, names.index(n), cv2.LINE_8, shift=4)
        c = g.representative_point()
        cx, cy = to_px(c.x, c.y)
        centre[S.NE_TO_ISO.get(n, n)] = (float(np.ravel(cx)[0]), float(np.ravel(cy)[0]))
    cc = m[1::2, 1::2].ravel()                                       # cell centres
    iso_names = [S.NE_TO_ISO.get(n, n) for n in names]
    np.savez_compressed(path, cc=cc, names=np.array(iso_names, object), centre=np.array(centre, object))
    return cc, iso_names, centre


CC, CC_NAMES, CC_CENTRE = build_countries()
CC_INDEX = {n: i for i, n in enumerate(CC_NAMES)}


def hash01(a, b, salt):
    """Stable pseudo-random numbers in [0, 1) from integer arrays (per grain jitter)."""
    h = (a.astype(np.uint64) * np.uint64(73856093)) ^ (b.astype(np.uint64) * np.uint64(19349663)) ^ np.uint64(salt * 83492791)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995)
    h = h ^ (h >> np.uint64(15))
    return (h % np.uint64(1 << 20)).astype(np.float64) / float(1 << 20)


def grain_xyz(cell, lev):
    """Where a grain is drawn (map px; z = the height of its centre), as the painter draws resting sand."""
    jx, jy, jr = hash01(cell, lev, 1), hash01(cell, lev, 2), hash01(cell, lev, 3)
    x = CELL_XY[cell, 0] + (jx - 0.5) * 1.3 * GS
    y = CELL_XY[cell, 1] + (jy - 0.5) * 1.3 * GS
    r = (1.35 + 0.35 * jr) * GS
    return x, y, lev * GRAIN_DZ + r, r


# ============================================================== the simulation
class Sand:
    def __init__(self, seed=11):
        self.rng = np.random.default_rng(seed)
        self.s = 0.0
        self.year = float(S.YEAR0)
        self.U = np.random.default_rng(77).random(NCELL).astype(np.float32)
        self.Hq = np.zeros(NCELL, np.int32)
        self.h = np.zeros(NCELL, np.int16)
        self.counts = np.zeros(NV, np.int64)
        self.particles = []            # batches of lifted grains (dicts of arrays)
        self.lifted = np.zeros(len(S.CAUSES))      # grains lifted so far, by cause
        self.recent = np.zeros(NV)     # grains lifted per vessel, decaying (what is happening where, for labels)
        self._load_lifts()
        # start settled: every vessel's sand in place at year 1
        self.counts[:] = np.round(grains_at(self.year)).astype(np.int64)
        self.cell_targets()
        self.h[:] = 0
        for v in range(NV):
            c = SCELLS[v]
            q = self.Hq[c]
            tot = int(q.sum())
            if tot == 0 or self.counts[v] == 0:
                continue
            want = int(self.counts[v])
            if tot != want:                                    # adjust the dither to hold exactly the count
                fr = self.Hq[c] - (self.Ht[c])
                order = np.argsort(fr if tot > want else -fr)
                d = abs(tot - want)
                idx = order[:min(d, len(order))]
                q = q.copy()
                q[idx] += -1 if tot > want else 1
                q = np.maximum(q, 0)
            self.h[c] = q
        self.counts = np.bincount(np.maximum(VID, 0), weights=self.h * (VID >= 0), minlength=NV).astype(np.int64)

    # ---------------------------------------------------------- lift sources
    def _load_lifts(self):
        L = json.load(open(os.path.join(DATA, "lifts.json")))
        self.L = L
        self.L_v = np.array([VI[l["vessel"]] for l in L])
        self.L_c = np.array([CI.get(l["cause"], CI["other"]) for l in L])
        self.L_y0 = np.array([l["y0"] for l in L], float)
        self.L_y1 = np.array([l["y1"] + 1 for l in L], float)          # exclusive end
        self.L_gpy = np.array([l["deaths"] for l in L]) / S.GRAIN / (self.L_y1 - self.L_y0)   # grains per year
        self.L_acc = self.rng.random(len(L)) * 0.999                     # dithered start: no lockstep rounding
        self.L_w = {}

    def lift_weights(self, i):
        """Candidate cells for lift i and their weights (before multiplying by the people there)."""
        if i in self.L_w:
            return self.L_w[i]
        l, v = self.L[i], self.L_v[i]
        cells = SCELLS[v]
        if l["kind"] == "spot":
            x, y = to_px(l["lon"], l["lat"])
            x, y = float(np.ravel(x)[0]), float(np.ravel(y)[0])
            sg = max(l["sigma"], 0.6) * PX_PER_DEG
            d2 = ((CELL_XY[cells] - (x, y)) ** 2).sum(1)
            w = np.exp(-d2 / (2 * sg * sg))
            if w.max() < 1e-3:                                       # the spot misses the glass: nearest land
                w = np.exp(-(np.sqrt(d2) - np.sqrt(d2.min())) ** 2 / (2 * 8.0 ** 2))
            ctr = (x, y)
        else:
            ci = CC_INDEX.get(l["iso"], -1)
            w = (CC[cells] == ci).astype(float) if ci >= 0 else np.zeros(len(cells))
            ctr = CC_CENTRE.get(l["iso"])
            if w.sum() == 0:                                         # a small island: nearest glass to it
                if ctr is None:
                    ctr = tuple(CELL_XY[cells].mean(0))
                d2 = ((CELL_XY[cells] - ctr) ** 2).sum(1)
                w = np.exp(-(np.sqrt(d2) - np.sqrt(d2.min())) ** 2 / (2 * 6.0 ** 2))
            else:
                ctr = tuple(CELL_XY[cells][w > 0].mean(0))
        keep = w > 1e-3
        out = (cells[keep], w[keep], np.array(ctr, float))
        self.L_w[i] = out
        return out

    def lift(self, i, k, dt):
        """Take k grains off the piles for lift i (from where it struck, in proportion to the people there)."""
        cells, w, ctr = self.lift_weights(i)
        hv = self.h[cells].astype(float)
        p = w * hv
        if p.sum() <= 0:
            return 0
        took_c, took_l = [], []
        k = int(min(k, hv.sum()))
        pick = self.rng.choice(len(cells), size=k, p=p / p.sum())
        cnt = np.bincount(pick, minlength=len(cells))
        cnt = np.minimum(cnt, self.h[cells])
        for r in range(int(cnt.max()) if len(cnt) else 0):
            sel = cells[cnt > r]
            self.h[sel] -= 1
            took_c.append(sel)
            took_l.append(self.h[sel].astype(np.int64))
        if not took_c:
            return 0
        c = np.concatenate(took_c); lv = np.concatenate(took_l)
        n = len(c)
        self.counts[self.L_v[i]] -= n
        x, y, z, r = grain_xyz(c, lv)
        self._new.append(dict(x=x, y=y, z=z, r=r, cx=np.full(n, ctr[0]), cy=np.full(n, ctr[1]),
                              t0=self.s - self.rng.random(n) * dt, cause=np.full(n, self.L_c[i]),
                              ph=self.rng.random((n, 3))))
        self.lifted[self.L_c[i]] += n
        self.recent[self.L_v[i]] += n
        return n

    def run_lifts(self, y0, y1, dt):
        ov = np.clip(np.minimum(y1, self.L_y1) - np.maximum(y0, self.L_y0), 0, None)
        act = np.where(ov > 0)[0]
        if not len(act):
            return
        self.L_acc[act] += self.L_gpy[act] * ov[act]
        for i in act:
            k = int(self.L_acc[i])
            if k >= 1:
                self.L_acc[i] -= k
                self.lift(i, k, dt)

    # ---------------------------------------------------------- grain moves
    def add(self, cells, col=0):
        cells = np.asarray(cells, np.int64)
        while len(cells):
            uc, first = np.unique(cells, return_index=True)
            rest = np.delete(cells, first)
            uc = uc[self.h[uc] < HMAX]
            self.h[uc] += 1
            np.add.at(self.counts, VID[uc], 1)
            cells = rest

    def pop(self, cells):
        cells = np.unique(np.asarray(cells, np.int64))
        cells = cells[self.h[cells] > 0]
        self.h[cells] -= 1
        np.add.at(self.counts, VID[cells], -1)
        return cells

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
                b2 = b
                self.h[b2] += 1

    def relocate(self, v):
        """Spread sand toward where people live: some grains above their cell's target move to a cell
        below target nearby (one of its RELOC_K nearest cells, else the nearest of a few anywhere in the
        vessel). Stops once every cell holds its target."""
        cells = SCELLS[v]
        hv, q = self.h[cells], self.Hq[cells]
        exc_i, dfc_i = np.where(hv > q)[0], np.where(hv < q)[0]
        if not len(exc_i) or not len(dfc_i):
            return
        m = min(len(exc_i), len(dfc_i), max(2, int(self.counts[v] * 0.004)), 4000)
        src = self.rng.choice(exc_i, size=m, replace=False)
        nbk = SNBK[v][src]                                      # (m, K) indices into cells
        deficit = hv < q
        pick = self.rng.integers(0, nbk.shape[1], (m, 6))
        cand = np.take_along_axis(nbk, pick, 1)
        good = deficit[cand]
        first = np.argmax(good, 1)
        dst = np.where(good.any(1), cand[np.arange(m), first], -1)
        miss = np.where(dst < 0)[0]
        if len(miss):
            c8 = dfc_i[self.rng.integers(len(dfc_i), size=(len(miss), 8))]
            dd = np.hypot(*(CELL_XY[cells[c8]] - CELL_XY[cells[src[miss]]][:, None, :]).transpose(2, 0, 1))
            dst[miss] = c8[np.arange(len(miss)), np.argmin(dd, 1)]
        dst, k = np.unique(dst, return_index=True)
        moved = self.pop(cells[src[k]])
        self.add(cells[dst[:len(moved)]])

    def reconcile(self, v, T):
        """Births and quiet losses: grains swell in where the pile is below its share of the people, or
        quietly go where it is above (slower growth, migration), until the vessel holds T grains."""
        N = int(self.counts[v])
        cells = SCELLS[v]
        if N < T - 0.5:
            k = int(min(round(T - N), max(6, 0.05 * N)))
            if k <= 0:
                return
            wgt = np.maximum(self.Hq[cells] - self.h[cells], 0).astype(float)
            if wgt.sum() <= 0:
                wgt = np.maximum(self.Ht[cells], 1e-3).astype(float)
            self.add(self.rng.choice(cells, size=k, p=wgt / wgt.sum()))
        elif N > T + 0.5:
            k = int(min(round(N - T), max(6, 0.05 * N)))
            occ = cells[self.h[cells] > 0]
            if k <= 0 or not len(occ):
                return
            ex = np.maximum(self.h[occ] - self.Hq[occ], 0) + 0.05
            self.pop(self.rng.choice(occ, size=min(k, len(occ)), p=ex / ex.sum(), replace=False))

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
        year = self.year
        w = 1.0
        N = self.counts.astype(float)
        mil_t = cityhist.at(CITY_YEARS, CITY_M, year)[CF["hi"]]             # each city's own population then
        uf = float((S.city_height(mil_t) >= 4.0 * S.HL).any())                      # any city big enough to show
        we = (1 - w) + w * DENS[VALID]
        vid = VID[VALID]
        bgf = np.zeros(NCELL, np.float32)
        rise = np.zeros(NCELL, np.float32)
        B = N.copy()
        cap = S.CITY_SHARE * N
        steep = getattr(self, "steep", None)  # per vessel: how much steeper than CITY_SLOPE its mountains stand
        steep = np.ones(NV) if steep is None else steep.copy()   # (carried from frame to frame: it changes slowly)
        warm = getattr(self, "steep", None) is not None
        for it in range(2 if warm else 3):    # countryside = what the city mountains do not take
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
            for k in range(1 if warm else 3):  # steepen (or relax) until the mountains fit their share
                h = S.city_height(mil_t)
                h = np.where(h < 4.0 * S.HL, 0.0, np.maximum(h, bgf[CF["centre"]] + S.CITY_RELIEF * h))  # on a crowded plain, still a hill
                R = np.maximum(h / (S.CITY_SLOPE * steep[CF["vi"]]), S.CITY_MIN_R)
                rise = self.city_rise(h, R, bgf)
                extra = np.bincount(VID[rise > 0], rise[rise > 0], minlength=NV)
                new = np.maximum(steep * np.sqrt(np.maximum(extra, 1e-9) / np.maximum(cap, 1e-9)), 1.0)
                new = np.where(extra > 0, new, steep)
                if np.all(np.abs(new - steep) < 0.03 * steep) or k == (0 if warm else 2):
                    if warm:
                        steep = steep + (new - steep) * 0.5      # settle over the next updates
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

    # ---------------------------------------------------------- one frame
    def step(self, s_new):
        dt = max(s_new - self.s, 1e-6)
        y0, y1 = self.year, year_at(s_new)
        self.s, self.year = s_new, y1
        self.recent *= math.exp(-dt / 1.5)
        if y1 > y0:
            self._new = []
            self.run_lifts(y0, y1, dt)
            if self._new:                       # one batch per frame (the painter works batch by batch)
                self.particles.append({k: np.concatenate([b[k] for b in self._new]) for k in self._new[0]})
            self._new = []
        T = grains_at(y1)
        self.nstep = getattr(self, "nstep", 0) + 1
        if self.nstep % TARGET_EVERY == 1 or TARGET_EVERY == 1:    # targets drift slowly: not every frame
            self.cell_targets()
        for v in range(NV):
            self.reconcile(v, T[v])
            self.relocate(v)
        self.topple(passes=2)
        self.particles = [p for p in self.particles if p["t0"].max() + LIFE > self.s]


TARGET_EVERY = 5  # frames between recomputing the pile shapes (cell_targets: the costly part)
LIFE = 5.0       # seconds a lifted grain is tracked (it is long out of the frame by then)
