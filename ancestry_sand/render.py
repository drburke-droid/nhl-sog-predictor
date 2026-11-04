"""Render "Grains of Ancestry": 200,000 years of human migration and mixing, as coloured sand poured into
glass vessels shaped like the land each population lives on.

    python render.py still 30 75.5 ...     # PNG stills at those video times (seconds) -> out/
    python render.py video                 # the full film -> out/grains_of_ancestry.mp4
    python render.py video --preview       # half-size, quick check -> out/preview.mp4
    python render.py report                # final ancestry of every vessel, to check against the sources

Each vessel has an exact gene-pool composition P (fractions of each ancestry component). The grains are
a physical picture of P: sand settles to the bottom of each glass shape, arrivals land on top, gaps
settle, jostling mixes the layers, and colour counts track P * N. Deterministic (fixed random seeds).
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from proj import W, H, to_px
import story as S
import vessels as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FPS = 30
SS = V.SS
R_GRAIN = V.R_GRAIN
FONT_DIR = "/usr/share/fonts/opentype/inter"
K = len(S.KEYS)
KI = {k: i for i, k in enumerate(S.KEYS)}
ARCH = np.array([k in S.ARCHAIC for k in S.KEYS])
MAX_FLYERS = 150            # grains drawn in flight per migration; each carries a share of the gene flow


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


COL = np.stack([hex_rgb(c[2]) for c in S.COMPONENTS])          # RGB 0..255
BG_TOP = hex_rgb("#070b14")
BG_BOT = hex_rgb("#0c1321")
LAND = hex_rgb("#121a29")
COAST = hex_rgb("#26354f")
INK = hex_rgb("#e8eef8")
MUTED = hex_rgb("#8d9bb3")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def smooth(v):
    v = min(1.0, max(0.0, v))
    return v * v * (3 - 2 * v)


# ============================================================== clock
CK_S = np.array([c[0] for c in S.CLOCK])
CK_U = np.log(np.array([c[1] for c in S.CLOCK], float) + 200.0)


def ybp_at(s):
    return max(0.0, math.exp(np.interp(s, CK_S, CK_U)) - 200.0)


def s_at(ybp):
    return float(np.interp(-math.log(ybp + 200.0), -CK_U, CK_S))


def fill_at(ybp):
    """Share of a vessel's capacity in use: population growth, very roughly."""
    pts = [(200_000, 0.40), (12_000, 0.40), (4_000, 0.66), (500, 0.78), (0, 0.97)]
    return float(np.interp(-ybp, [-p[0] for p in pts], [p[1] for p in pts]))


# ============================================================== map background
def build_background():
    path = os.path.join(HERE, "data", "ne_50m_land.geojson")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
        subprocess.run(["curl", "-sS", "-o", path, url], check=True)
    from shapely.geometry import shape
    gj = json.load(open(path))
    rings, holes = [], []
    for f in gj["features"]:
        for p in V.split_cut(shape(f["geometry"])):
            for ring, dst in [(p.exterior, rings)] + [(r, holes) for r in p.interiors]:
                xy = np.array(ring.coords)
                x, y = to_px(np.minimum(xy[:, 0], 189.999), np.maximum(xy[:, 1], -89.9))
                dst.append(np.round(np.stack([x, y], 1) * SS * 16).astype(np.int32))
    hh, ww = H * SS, W * SS
    t = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
    bg = (BG_TOP * (1 - t) + BG_BOT * t) * np.ones((1, ww, 1), np.float32)
    land = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(land, rings, 255, lineType=cv2.LINE_AA, shift=4)
    if holes:
        cv2.fillPoly(land, holes, 0, lineType=cv2.LINE_AA, shift=4)
    a = (land.astype(np.float32) / 255.0)[..., None]
    bg = bg * (1 - a) + LAND * a
    coast = np.zeros((hh, ww), np.uint8)
    cv2.polylines(coast, rings, True, 255, thickness=SS, lineType=cv2.LINE_AA, shift=4)
    c = (coast.astype(np.float32) / 255.0)[..., None] * 0.5
    bg = bg * (1 - c) + COAST * c
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    r2 = ((xx / ww - 0.5) / 0.75) ** 2 + ((yy / hh - 0.48) / 0.72) ** 2
    bg *= (1 - 0.35 * np.clip(r2, 0, 1.4) ** 1.5)[..., None]
    return np.clip(bg, 0, 255).astype(np.uint8)


# ============================================================== vessels
GEOM = V.build(S.VESSELS)


class Jar:
    def __init__(self, vid, spec, g):
        self.id, self.label, self.note = vid, spec["label"], spec.get("note")
        self.slots, self.up, self.nb = g["slots"], g["up"], g["nb"]
        self.cap = len(self.slots)
        self.cx, self.cy = g["anchor"]
        self.geo = g["geo"]
        self.bbox = g["bbox"]
        self.sprite = g["sprite"].astype(np.float32)
        self.sprite_off = g["sprite_off"]
        self.inset = bool(spec.get("inset"))
        self.P = np.zeros(K)
        self.L = []
        self.counts = np.zeros(K, np.int64)
        self.glass = 0.45            # displayed glass brightness (dim while empty)
        self.fixed_fill = None       # archaic occupants keep their own fill level
        self.growth = None           # dict(n0, t0) after a founding
        self.replace = None          # archaic sand being displaced by newcomers
        self.extinct = None
        self.boost = 0.0
        self.pending = 0


ORDER = list(S.VESSELS.keys())
JARS = {j: Jar(j, S.VESSELS[j], GEOM[j]) for j in ORDER}
JIDX = {j: i for i, j in enumerate(ORDER)}


# ============================================================== grains
MAXG = 60000


class Grains:
    def __init__(self):
        self.state = np.zeros(MAXG, np.int8)     # 0 free, 1 in a vessel, 2 flying, 3 fading
        self.x = np.zeros(MAXG, np.float32)
        self.y = np.zeros(MAXG, np.float32)
        self.col = np.zeros(MAXG, np.int16)
        self.prev = np.zeros(MAXG, np.int16)
        self.tc = np.full(MAXG, -99.0, np.float32)   # time of last colour change
        self.tb = np.full(MAXG, -99.0, np.float32)   # birth time (pop-in)
        self.tf = np.zeros(MAXG, np.float32)          # fade start
        self.jar = np.full(MAXG, -1, np.int16)
        self.free = list(range(MAXG - 1, -1, -1))
        self.f_path = np.zeros(MAXG, np.int32)
        self.f_s0 = np.zeros(MAXG, np.float32)
        self.f_dur = np.ones(MAXG, np.float32)
        self.f_off = np.zeros(MAXG, np.float32)
        self.f_dst = np.zeros(MAXG, np.int16)
        self.f_ev = np.zeros(MAXG, np.int32)
        self.f_start = np.zeros((MAXG, 2), np.float32)   # offset from the path start (lift-off point)
        self.f_end = np.zeros((MAXG, 2), np.float32)     # offset from the path end (landing slot)

    def new(self):
        return self.free.pop()

    def release(self, g):
        self.state[g] = 0
        self.free.append(g)


G = Grains()
SHADE = (0.80 + 0.32 * np.random.default_rng(3).random(MAXG)).astype(np.float32)


# ============================================================== flight paths
PATHS = []


def chaikin(pts, it=3):
    pts = np.asarray(pts, float)
    for _ in range(it):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def make_path(src, dst):
    """Screen-space path between two vessels' pour points, via the route's waypoints. Where a route
    crosses the map edge (Bering Strait / mid-Pacific) it leaves one side and re-enters the other."""
    a, b = JARS[src], JARS[dst]
    p0, p1 = np.array([a.cx, a.cy]), np.array([b.cx, b.cy])
    wps = S.ROUTES.get((src, dst))
    if wps:
        x, y = to_px(np.array([w[0] for w in wps], float), np.array([w[1] for w in wps], float))
        raw = [p0] + [np.array(v) for v in zip(x, y)] + [p1]
        segs, cur = [], [raw[0]]
        for k in range(1, len(raw)):
            if abs(raw[k][0] - raw[k - 1][0]) > W * 0.5:
                d0 = raw[k - 1] - raw[k - 2]
                cur.append(raw[k - 1] + d0 / max(np.hypot(*d0), 1e-6) * 90)
                segs.append(cur)
                nxt = raw[k + 1] if k + 1 < len(raw) else raw[k]
                d1 = nxt - raw[k]
                cur = [raw[k] - d1 / max(np.hypot(*d1), 1e-6) * 90, raw[k]]
            else:
                cur.append(raw[k])
        segs.append(cur)
        pieces = [chaikin(np.array(sg), 4) for sg in segs]
        pts = np.vstack(pieces)
        brk = np.zeros(len(pts) - 1, bool)
        n = 0
        for pc in pieces[:-1]:
            n += len(pc)
            brk[n - 1] = True
    else:
        d = p1 - p0
        dist = np.hypot(*d)
        nrm = np.array([d[1], -d[0]]) / max(dist, 1e-6)
        if nrm[1] > 0:
            nrm = -nrm                     # bulge upwards
        c = (p0 + p1) / 2 + nrm * dist * 0.22
        t = np.linspace(0, 1, 48)[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1
        brk = np.zeros(len(pts) - 1, bool)
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    seg[brk] = 0.0
    cum = np.concatenate([[0], np.cumsum(seg)])
    PATHS.append(dict(pts=pts, cum=cum, total=max(cum[-1], 1.0), brk=brk))
    return len(PATHS) - 1


def path_point(pi, u):
    p = PATHS[pi]
    d = u * p["total"]
    i = int(np.clip(np.searchsorted(p["cum"], d) - 1, 0, len(p["pts"]) - 2))
    seg = p["cum"][i + 1] - p["cum"][i]
    t = (d - p["cum"][i]) / seg if seg > 0 else 1.0
    a, b = p["pts"][i], p["pts"][i + 1]
    tang = b - a
    nrm = np.array([-tang[1], tang[0]]) / max(np.hypot(*tang), 1e-6)
    return a + (b - a) * t, nrm


def ease(u):
    return u * u * (3 - 2 * u) * 0.65 + u * 0.35


def flyer_pos(g, s):
    u = float(np.clip((s - G.f_s0[g]) / G.f_dur[g], 0, 1))
    pt, nrm = path_point(int(G.f_path[g]), ease(u))
    off = G.f_off[g] * math.sin(math.pi * u)
    w0 = 1 - smooth(u / 0.22)
    w1 = smooth((u - 0.72) / 0.28)
    return pt + nrm * off + G.f_start[g] * w0 + G.f_end[g] * w1


# ============================================================== the simulation
class Sim:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.s = 0.0
        self.events = []
        for i, e in enumerate(S.EVENTS):
            e = dict(e, i=i, s0=s_at(e["t0"]), s1=s_at(e["t1"]), started=False, done=False, launched=0)
            if e["kind"] in ("found", "flow"):
                e["path"] = make_path(e["src"], e["dst"])
                e["fly"] = min(3.6, 0.9 + PATHS[e["path"]]["total"] / 650.0)
                if e["kind"] == "flow" and e["s1"] - e["s0"] < 1.6:
                    e["s1"] = e["s0"] + 1.6
            if e["kind"] in ("tint", "extinct") and e["s1"] - e["s0"] < 2.0:
                e["s1"] = e["s0"] + 2.0
            self.events.append(e)
        self.guides = []
        for j, (comp, fill) in S.INITIAL.items():
            jar = JARS[j]
            c = KI[comp]
            jar.P[c] = 1.0
            if ARCH[c]:
                jar.fixed_fill = fill
            n = int(round((fill if ARCH[c] else fill_at(200_000)) * jar.cap))
            for _ in range(n):
                self.add_grain(jar, c, born=-99)
            jar.glass = 1.0
        self.first_seen = {}

    # ---- grain bookkeeping
    def add_grain(self, jar, c, pos=None, born=None):
        g = G.new()
        slot = len(jar.L)
        jar.L.append(g)
        G.state[g] = 1
        G.jar[g] = JIDX[jar.id]
        G.col[g] = c; G.prev[g] = c; G.tc[g] = -99
        G.tb[g] = self.s if born is None else born
        if pos is None:
            G.x[g], G.y[g] = jar.slots[slot]
        else:
            G.x[g], G.y[g] = pos
        jar.counts[c] += 1
        return g

    def remove_at(self, jar, i, settle=True):
        """Remove the grain in slot i: it fades, and the grains above it settle into the gap."""
        L = jar.L
        g = L[i]
        jar.counts[G.col[g]] -= 1
        G.state[g] = 3; G.tf[g] = self.s
        n = len(L)
        j = i
        if settle:
            while jar.up[j] >= 0 and jar.up[j] < n:
                k = jar.up[j]
                L[j] = L[k]
                j = k
        if j != n - 1:
            L[j] = L[n - 1]
        L.pop()

    def recolor(self, jar, g, c):
        jar.counts[G.col[g]] -= 1
        G.prev[g] = G.col[g]
        G.col[g] = c
        G.tc[g] = self.s
        jar.counts[c] += 1

    def pick_of_color(self, jar, c, lo=0, hi=None):
        """Index of a random grain of colour c, preferring the slot range [lo, hi)."""
        L = jar.L
        lo = max(0, lo)
        hi = len(L) if hi is None else min(hi, len(L))
        if hi > lo:
            for _ in range(24):
                i = int(self.rng.integers(lo, hi))
                if G.col[L[i]] == c:
                    return i
        if jar.counts[c] <= 0 or not L:
            return None
        for _ in range(60):
            i = int(self.rng.integers(len(L)))
            if G.col[L[i]] == c:
                return i
        ids = [i for i, g in enumerate(L) if G.col[g] == c]
        return ids[self.rng.integers(len(ids))] if ids else None

    def surface_pos(self, jar):
        if jar.L:
            span = max(4, len(jar.L) // 12)
            g = jar.L[len(jar.L) - 1 - int(self.rng.integers(min(span, len(jar.L))))]
            return np.array([G.x[g], G.y[g]])
        return np.array([jar.cx, jar.cy])

    @staticmethod
    def modern_P(jar):
        """The gene pool of the people who migrate: archaic sand still being displaced doesn't travel."""
        return jar.replace["P_new"] if jar.replace is not None else jar.P

    # ---- events
    def launch(self, e, c):
        src, dst = JARS[e["src"]], JARS[e["dst"]]
        g = G.new()
        G.state[g] = 2
        G.col[g] = c; G.prev[g] = c; G.tc[g] = -99; G.tb[g] = self.s
        G.f_path[g] = e["path"]; G.f_s0[g] = self.s
        G.f_dur[g] = e["fly"] * (0.92 + 0.16 * self.rng.random())
        G.f_off[g] = self.rng.normal(0, 2.5)
        G.f_dst[g] = JIDX[e["dst"]]
        G.f_ev[g] = e["i"]
        G.jar[g] = -1
        pts = PATHS[e["path"]]["pts"]
        G.f_start[g] = self.surface_pos(src) - pts[0]
        slot = len(dst.L) + dst.pending
        if slot >= dst.cap:
            slot = dst.cap - 1 - int(self.rng.integers(max(1, dst.cap // 10)))
        G.f_end[g] = dst.slots[slot] - pts[-1]
        dst.pending += 1
        G.x[g], G.y[g] = pts[0] + G.f_start[g]
        e["launched"] += 1

    def start_event(self, e):
        e["started"] = True
        k = e["kind"]
        if k == "found":
            src, dst = JARS[e["src"]], JARS[e["dst"]]
            okc = np.array([(not ARCH[c]) and (e.get("only") is None or S.KEYS[c] in e["only"]) for c in range(K)])
            cand = [g for g in src.L if okc[G.col[g]]]
            if len(cand) >= 2 * e["n"]:
                pick = self.rng.choice(len(cand), size=e["n"], replace=False)
                cols = [int(G.col[cand[k]]) for k in pick]
            else:                                   # source still tiny: draw from its gene pool
                q = self.modern_P(src) * okc
                cols = [int(c) for c in self.rng.choice(K, size=e["n"], p=q / q.sum())]
            P = np.zeros(K)
            for c in cols:
                P[c] += 1
            arch = self.modern_P(src) * ARCH
            P = P / P.sum() * (1 - arch.sum()) + arch
            if dst.L and dst.P[ARCH].sum() > 0.5:      # archaic people live here: displace, with mixing
                keep = e.get("keep", 0.0)
                P_new = P * (1 - keep) + dst.P * keep
                until = e.get("until") or e["t0"] * 0.85
                dst.replace = dict(P_old=dst.P.copy(), P_new=P_new, A0=len(dst.L), s0=self.s,
                                   s1=max(s_at(until), self.s + 3.0))
                dst.fixed_fill = None
            else:
                dst.P = P
                dst.extinct = None
                dst.fixed_fill = None
            dst.growth = dict(n0=e["n"], t0=None, landed=0)
            e["cols"] = cols
            self.guides.append([e["path"], self.s, self.s + e["fly"] + 0.6, e])
        elif k == "flow":
            src, dst = JARS[e["src"]], JARS[e["dst"]]
            N = max(len(dst.L), 2)
            f = e["f"]
            n_real = math.log(1 - f) / math.log(1 - 1.0 / N)
            n = int(max(3 if f >= 0.01 else 1, min(round(n_real), MAX_FLYERS)))
            e["n_total"] = n
            e["Psrc"] = S.restrict(self.modern_P(src), e.get("only"))
            e["qcounts"] = np.zeros(K)
            e["w"] = 1 - (1 - f) ** (1.0 / n)
            self.guides.append([e["path"], self.s, e["s1"] + e["fly"] + 0.6, e])
        elif k == "tint":
            e["k"] = math.log(1 / 0.03) / max(e["s1"] - e["s0"], 1.0)
        elif k == "extinct":
            JARS[e["jar"]].extinct = (e["s0"], e["s1"])

    def run_events(self, dt):
        for e in self.events:
            if e["done"]:
                continue
            if not e["started"]:
                if self.s < e["s0"]:
                    continue
                self.start_event(e)
            k = e["kind"]
            if k == "found":
                n = len(e["cols"])
                due = int(min(n, math.floor((self.s - e["s0"]) / 0.7 * n) + 1))
                while e["launched"] < due:
                    self.launch(e, e["cols"][e["launched"]])
                e["done"] = e["launched"] >= n
            elif k == "flow":
                prog = min(1.0, (self.s - e["s0"]) / max(e["s1"] - e["s0"], 1e-3))
                due = int(round(prog * e["n_total"]))
                while e["launched"] < due:
                    want = e["Psrc"] * (e["launched"] + 1) - e["qcounts"]
                    want[e["Psrc"] <= 0] = -1e9
                    c = int(np.argmax(want))
                    e["qcounts"][c] += 1
                    self.launch(e, c)
                e["done"] = e["launched"] >= e["n_total"]
            elif k == "tint":
                jar = JARS[e["jar"]]
                tgt = KI[e["target"]]
                fac = math.exp(-e["k"] * dt)
                for sname in e["sources"]:
                    c = KI[sname]
                    m = jar.P[c] if self.s >= e["s1"] else jar.P[c] * (1 - fac)
                    jar.P[c] -= m
                    jar.P[tgt] += m
                e["done"] = self.s >= e["s1"]
            elif k == "extinct":
                e["done"] = self.s >= e["s1"]

    def land(self, g):
        jar = JARS[ORDER[G.f_dst[g]]]
        e = self.events[G.f_ev[g]]
        jar.pending = max(0, jar.pending - 1)
        c = int(G.col[g])
        L = jar.L
        if len(L) >= jar.cap:
            ex = jar.counts - jar.P * (len(L) - 1)
            ex[jar.counts == 0] = -1e9
            i = self.pick_of_color(jar, int(np.argmax(ex)))
            self.remove_at(jar, i if i is not None else 0)
        G.state[g] = 1
        G.jar[g] = JIDX[jar.id]
        G.tb[g] = self.s
        L.append(g)
        jar.counts[c] += 1
        if e["kind"] == "flow":
            w = e["w"]
            jar.P = jar.P * (1 - w) + e["Psrc"] * w
            jar.boost = min(jar.boost + 0.2, 3.0)
            # this grain stands for several: pour its colour into the surface layer around it
            n = len(L)
            extra = int(round(w * n)) - 1
            span = max(20, n // 10)
            for _ in range(max(0, extra)):
                d = jar.counts - jar.P * n
                lo = int(np.argmin(d)); hi = int(np.argmax(d))
                if d[hi] < 0.5 or d[lo] > -0.5:
                    break
                i = self.pick_of_color(jar, hi, n - span, n)
                if i is None:
                    break
                self.recolor(jar, L[i], lo)
            tgt = min(self.n_target(jar), jar.cap - 1)
            if len(L) > tgt + 0.5 and len(L) > 1:
                ex = jar.counts - jar.P * (len(L) - 1)
                ex[jar.counts == 0] = -1e9
                i = self.pick_of_color(jar, int(np.argmax(ex)), 0, len(L) - span)
                if i is not None and i != len(L) - 1:
                    self.remove_at(jar, i)
        else:
            jar.boost = min(jar.boost + 0.1, 3.0)
            gr = jar.growth
            if gr is not None:
                gr["landed"] += 1
                if gr["t0"] is None:
                    gr["t0"] = self.s

    def n_target(self, jar):
        ybp = ybp_at(self.s)
        base = (jar.fixed_fill if jar.fixed_fill is not None else fill_at(ybp)) * jar.cap
        if jar.extinct is not None:
            s0, s1 = jar.extinct
            base *= max(0.0, 1 - (self.s - s0) / max(s1 - s0, 0.5))
        gr = jar.growth
        if gr is not None and jar.extinct is None:
            if gr["t0"] is None:
                modern = float(gr["landed"])
            else:
                modern = min(base, gr["n0"] * math.exp(0.8 * (self.s - gr["t0"])))
            rp = jar.replace
            if rp is not None:
                A = rp["A0"] * max(0.0, 1 - (self.s - rp["s0"]) / (rp["s1"] - rp["s0"]))
                u = modern / max(modern + A, 1e-6)
                jar.P = rp["P_new"] * u + rp["P_old"] * (1 - u)
                if A <= 0 and modern >= base - 1:
                    jar.replace = None
                    jar.P = rp["P_new"].copy()
                return min(modern + A, jar.cap)
            return min(modern, jar.cap)
        return min(base, jar.cap)

    def reconcile(self, jar, dt):
        L = jar.L
        tgt = self.n_target(jar)
        n = len(L)
        if n < math.floor(tgt) and n > 0:
            # growth: a new grain buds from an existing grain of the colour it copies
            for _ in range(min(int(math.floor(tgt)) - n, max(2, int(n * 0.05)))):
                n = len(L)
                want = jar.P * (n + 1) - jar.counts
                want[jar.P <= 0] = -1e9
                c = int(np.argmax(want))
                i = self.pick_of_color(jar, c)
                g0 = L[i if i is not None else n - 1]
                self.add_grain(jar, c, pos=(G.x[g0], G.y[g0]))
        elif n > math.ceil(tgt):
            for _ in range(min(n - int(math.ceil(tgt)), max(2, int(n * 0.04)))):
                if not L:
                    break
                if jar.extinct is not None:
                    self.remove_at(jar, len(L) - 1, settle=False)
                else:
                    ex = jar.counts - jar.P * (len(L) - 1)
                    ex[jar.counts == 0] = -1e9
                    i = self.pick_of_color(jar, int(np.argmax(ex)))
                    if i is not None:
                        self.remove_at(jar, i)
        # composition: displace (archaic sand) or recolour (drift) a few grains per frame
        n = len(L)
        if n:
            for _ in range(max(1, n // 60)):
                d = jar.counts - jar.P * n
                hi = int(np.argmax(d)); lo = int(np.argmin(d))
                if d[hi] <= 0.75 or d[lo] >= -0.75:
                    break
                i = self.pick_of_color(jar, hi)
                if i is None:
                    break
                if jar.replace is not None:
                    self.remove_at(jar, i)
                    n = len(L)
                else:
                    self.recolor(jar, L[i], lo)
        # jostling: neighbouring grains swap places, and now and then grains further apart
        if n > 4:
            m = self.rng.poisson((0.02 + 0.03 * jar.boost) * n)
            idx = self.rng.integers(0, n, size=m)
            far = self.rng.random(m) < 0.25
            reach = max(8, n // 8)
            for i, f in zip(idx.tolist(), far.tolist()):
                if f:
                    j = i + int(self.rng.integers(-reach, reach + 1))
                    if not 0 <= j < n:
                        continue
                else:
                    nbs = jar.nb[i]
                    if len(nbs) == 0:
                        continue
                    j = int(nbs[self.rng.integers(len(nbs))])
                    if j >= n:
                        continue
                L[i], L[j] = L[j], L[i]
        jar.boost *= math.exp(-dt / 2.5)
        if jar.extinct is not None and not L and self.s > jar.extinct[1]:
            jar.extinct = None
            jar.growth = None

    def step(self, s_new):
        dt = s_new - self.s
        self.s = s_new
        self.run_events(dt)
        for g in np.where(G.state == 2)[0]:
            if self.s >= G.f_s0[g] + G.f_dur[g]:
                self.land(g)
        for j in ORDER:
            jar = JARS[j]
            if jar.L or jar.growth is not None or jar.pending:
                self.reconcile(jar, dt)
            want = 1.0 if (jar.L or jar.pending) else 0.45
            jar.glass += (want - jar.glass) * (1 - math.exp(-dt / 0.6))
        for g in np.where((G.state == 3) & (self.s - G.tf > 0.6))[0]:
            G.release(g)
        a = 1 - math.exp(-dt / 0.10) if dt > 0 else 1.0
        for j in ORDER:
            jar = JARS[j]
            if jar.L:
                ids = np.array(jar.L, np.int32)
                tx, ty = jar.slots[:len(ids), 0], jar.slots[:len(ids), 1]
                G.x[ids] += (tx - G.x[ids]) * a
                G.y[ids] += (ty - G.y[ids]) * a
        for g in np.where(G.state == 2)[0]:
            G.x[g], G.y[g] = flyer_pos(g, self.s)
        for c in range(K):
            if c not in self.first_seen and any(JARS[j].counts[c] > 0 for j in ORDER):
                self.first_seen[c] = self.s


# ============================================================== drawing
class Painter:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.bg = build_background()                   # uint8 RGB at SS scale
        self.base_key = None
        self.base = None
        self.f_label = font("Inter-SemiBold.otf", 14)
        self.f_year = font("InterDisplay-SemiBold.otf", 58)
        self.f_year_sub = font("Inter-Medium.otf", 20)
        self.f_cap = font("Inter-Medium.otf", 29)
        self.f_leg = font("Inter-Medium.otf", 14)
        self.f_leg_h = font("Inter-SemiBold.otf", 12)
        self.f_title = font("InterDisplay-SemiBold.otf", 84)
        self.f_sub = font("Inter-Regular.otf", 28)
        self.f_note = font("Inter-Regular.otf", 15)
        self.cache = {}
        self.labels = {j: self.text_sprite(JARS[j].label, self.f_label, INK, halo=True) for j in ORDER}

    # ---- sprites
    def text_sprite(self, text, fnt, rgb, max_w=None, spacing=6, align="left", halo=False):
        key = (text, id(fnt), tuple(rgb), max_w, align, halo)
        if key in self.cache:
            return self.cache[key]
        lines = [text]
        if max_w:
            words, lines, cur = text.split(), [], ""
            for wd in words:
                t = (cur + " " + wd).strip()
                if fnt.getlength(t) > max_w and cur:
                    lines.append(cur); cur = wd
                else:
                    cur = t
            lines.append(cur)
        asc, desc = fnt.getmetrics()
        lh = asc + desc + spacing
        pad = 6 if halo else 2
        ww = int(max(fnt.getlength(l) for l in lines)) + 2 * pad
        hh = lh * len(lines) + 2 * pad
        im = Image.new("L", (ww, hh), 0)
        dr = ImageDraw.Draw(im)
        for i, l in enumerate(lines):
            x = pad if align == "left" else (ww - fnt.getlength(l)) / 2
            dr.text((x, pad + i * lh), l, font=fnt, fill=255)
        a = np.asarray(im, np.float32) / 255
        spr = np.zeros((hh, ww, 4), np.float32)
        spr[..., :3] = rgb
        spr[..., 3] = a
        if halo:      # dark halo so labels read over sand
            hal = np.clip(cv2.GaussianBlur(cv2.dilate(a, np.ones((3, 3), np.uint8)), (0, 0), 2.0) * 1.4, 0, 0.9)
            out_a = a + hal * (1 - a)
            spr[..., :3] = (rgb * a[..., None] + np.array([6, 9, 16], np.float32) * (hal * (1 - a))[..., None]) \
                / np.maximum(out_a, 1e-4)[..., None]
            spr[..., 3] = out_a
        self.cache[key] = spr
        return spr

    @staticmethod
    def blit(img, spr, x, y, alpha=1.0):
        """Alpha-composite an RGBA float sprite onto an RGB float image at (x, y)."""
        if alpha <= 0.003:
            return
        h, w = spr.shape[:2]
        H_, W_ = img.shape[:2]
        x, y = int(round(x)), int(round(y))
        xa, ya, xb, yb = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
        if xa >= xb or ya >= yb:
            return
        s = spr[ya - y:yb - y, xa - x:xb - x]
        a = s[..., 3:4] * alpha
        reg = img[ya:yb, xa:xb]
        reg *= (1 - a)
        reg += s[..., :3] * a

    @staticmethod
    def shade(img, x, y, w, h, a):
        if a <= 0.003:
            return
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))
        if x0 < x1 and y0 < y1:
            img[y0:y1, x0:x1] *= (1 - a)

    def base_layer(self):
        """Map + glass vessels (+ inset hairlines), cached while the vessels' brightness is unchanged."""
        key = tuple(int(round(JARS[j].glass * 12)) for j in ORDER)
        if key == self.base_key:
            return self.base
        img = self.bg.astype(np.float32)
        for j, q in zip(ORDER, key):
            jar = JARS[j]
            a = q / 12
            if jar.geo is not None:
                bx0, by0, bx1, by1 = jar.bbox
                qx = min(max(jar.geo[0], bx0 - 6), bx1 + 6)
                qy = min(max(jar.geo[1], by0 - 6), by1 + 6)
                col = tuple(float(v) for v in LAND * (1 - 0.45 * a) + MUTED * 0.45 * a)
                p1 = (int(jar.geo[0] * SS * 16), int(jar.geo[1] * SS * 16))
                cv2.line(img, (int(qx * SS * 16), int(qy * SS * 16)), p1, col, SS, cv2.LINE_AA, shift=4)
                cv2.circle(img, p1, int(2.6 * SS * 16), col, -1, cv2.LINE_AA, shift=4)
            ox, oy = jar.sprite_off
            self.blit(img, jar.sprite, ox, oy, a)
        self.base = np.clip(img, 0, 255).astype(np.uint8)
        self.base_key = key
        return self.base

    # ---- frame
    def frame(self, sim, s):
        hi = self.base_layer().copy()
        f16 = SS * 16
        # route guides (faint lines while a migration is under way)
        for pi, s_on, s_off, e in sim.guides:
            if e["kind"] == "flow":
                s_off = e["s1"] + e["fly"] + 0.5
            if s < s_on or s > s_off + 1.0:
                continue
            a = 0.22 * min(1.0, (s - s_on) / 0.5) * min(1.0, max(0.0, (s_off + 1.0 - s) / 1.0))
            col = tuple(float(v) for v in BG_BOT * (1 - a) + np.array([210, 225, 255]) * a)
            p = PATHS[pi]
            pts, cur, segs = p["pts"], [p["pts"][0]], []
            for k in range(len(pts) - 1):
                if p["brk"][k]:
                    segs.append(cur); cur = [pts[k + 1]]
                else:
                    cur.append(pts[k + 1])
            segs.append(cur)
            for sg in segs:
                if len(sg) > 1:
                    cv2.polylines(hi, [np.round(np.array(sg) * f16).astype(np.int32)], False, col, SS,
                                  cv2.LINE_AA, shift=4)

        glow = np.zeros((H // 2, W // 2, 3), np.float32)
        act = np.where(G.state > 0)[0]
        if len(act):
            st = G.state[act]
            c = G.col[act]
            tcol = np.clip((s - G.tc[act]) / 0.8, 0, 1)[:, None]
            rgb = COL[G.prev[act]] * (1 - tcol) + COL[c] * tcol
            r = np.full(len(act), R_GRAIN, np.float32)
            bscale = np.ones(len(act), np.float32)
            age = s - G.tb[act]
            young = (st == 1) & (age < 0.35)
            r[young] *= 0.4 + 0.6 * np.clip(age[young], 0, None) / 0.35
            fad = st == 3
            bscale[fad] = np.clip(1 - (s - G.tf[act][fad]) / 0.6, 0, 1)
            r[fad] *= 0.5 + 0.5 * bscale[fad]
            fly = st == 2
            r[fly] = 2.0
            arch = ARCH[c]
            tw = 0.5 + 0.5 * np.sin(s * 5.0 + act * 1.7)
            bscale[arch] *= 0.75 + 0.30 * tw[arch]
            rgb[~fly] *= SHADE[act][~fly, None]
            gi_extra = np.zeros(len(act), np.float32)
            tw0 = S.TODAY_S + 9.4
            if tw0 <= s <= tw0 + 6.5:          # closing ripple of light from East Africa
                e = JARS["EAF"]
                rad = (s - tw0) * 330.0
                amp = 0.85 * smooth((tw0 + 6.5 - s) / 1.5)
                dd = np.hypot(G.x[act] - e.cx, G.y[act] - e.cy)
                wv = np.exp(-((dd - rad) / 60.0) ** 2) * amp * (st == 1)
                rgb = rgb * (1 - wv[:, None]) + np.array([255, 244, 214], np.float32) * wv[:, None]
                gi_extra = wv
            col = rgb * bscale[:, None] + ((1 - bscale) * (~fly))[:, None] * LAND
            xs = (G.x[act] * f16).astype(np.int32).tolist()
            ys = (G.y[act] * f16).astype(np.int32).tolist()
            rs = (r * f16).astype(np.int32).tolist()
            cl = col.tolist()
            for i in range(len(act)):
                cv2.circle(hi, (xs[i], ys[i]), rs[i], cl[i], -1, cv2.LINE_AA, shift=4)
            gi = np.where(fly, 1.0, np.where(arch, 0.35 * tw, 0.0)) + 0.9 * gi_extra
            gsel = np.where(gi > 0.02)[0]
            if len(gsel):
                gx = (G.x[act][gsel] * 8).astype(np.int32).tolist()
                gy = (G.y[act][gsel] * 8).astype(np.int32).tolist()
                gr = (np.where(fly[gsel], 1.7, 1.2) * 16).astype(np.int32).tolist()
                gc = (rgb[gsel] * gi[gsel, None]).tolist()
                for i in range(len(gsel)):
                    cv2.circle(glow, (gx[i], gy[i]), gr[i], gc[i], -1, cv2.LINE_AA, shift=4)
        # comet trails behind flying grains
        for g in np.where(G.state == 2)[0]:
            u = float(np.clip((s - G.f_s0[g]) / G.f_dur[g], 0, 1))
            if u < 0.05:
                continue
            pa = flyer_pos(g, G.f_s0[g] + max(0.0, u - 0.06) * G.f_dur[g])
            pb = np.array([G.x[g], G.y[g]])
            if np.hypot(*(pb - pa)) > W * 0.3:
                continue
            rgb = COL[G.col[g]] * 0.6
            cv2.line(glow, (int(pa[0] * 8), int(pa[1] * 8)), (int(pb[0] * 8), int(pb[1] * 8)),
                     tuple(float(v) for v in rgb), 1, cv2.LINE_AA, shift=4)

        img = cv2.resize(hi, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.GaussianBlur(np.maximum(small - 60, 0), (0, 0), 3.0) * 0.35
        g1 = cv2.GaussianBlur(glow, (0, 0), 1.1)
        g2 = cv2.GaussianBlur(glow, (0, 0), 4.5)
        img += cv2.resize(bloom + g1 * 0.9 + g2 * 0.8, (W, H), interpolation=cv2.INTER_LINEAR)

        self.hud(img, sim, s)
        if self.scale != 1.0:
            img = cv2.resize(img, (int(W * self.scale), int(H * self.scale)), interpolation=cv2.INTER_AREA)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---- text and overlays
    def hud(self, img, sim, s):
        title_a = 1.0 - smooth((s - 4.2) / 1.6)
        for j in ORDER:
            jar = JARS[j]
            spr = self.labels[j]
            a = 0.35 + 0.5 * (jar.glass - 0.45) / 0.55
            if jar.inset:
                x, y = (jar.bbox[0] + jar.bbox[2]) / 2, jar.bbox[3] + 8
            else:
                x, y = jar.cx, jar.cy - spr.shape[0] / 2
            self.blit(img, spr, x - spr.shape[1] / 2, y, a * (1 - title_a * 0.6))
        # year
        ybp = ybp_at(s)
        if ybp >= 500:
            if ybp >= 10_000:
                v = int(round(ybp / 100.0) * 100)
            elif ybp >= 1_000:
                v = int(round(ybp / 10.0) * 10)
            else:
                v = int(round(ybp))
            big, small = f"{v:,}", "years ago"
        else:
            yr = int(round(2025 - ybp))
            big, small = (f"{yr}", "CE") if yr < 2024 else ("Today", "")
        spr = self.text_sprite(big, self.f_year, INK)
        self.blit(img, spr, 44, 26, 0.95)
        if small:
            self.blit(img, self.text_sprite(small, self.f_year_sub, MUTED), 48, 26 + spr.shape[0] + 2, 0.9)
        # legend
        lx, ly = 40, 700
        seen = sim.first_seen
        if seen:
            self.blit(img, self.text_sprite("ANCESTRY  (COLOUR OF SAND)", self.f_leg_h, MUTED), lx, ly - 24, 0.8)
        row = 0
        for c in range(K):
            if c not in seen:
                continue
            a = smooth((s - seen[c]) / 0.8)
            yy = ly + row * 19
            cv2.circle(img, ((lx + 7) * 16, (yy + 10) * 16), 6 * 16,
                       tuple(float(v) for v in LAND * (1 - a) + COL[c] * a), -1, cv2.LINE_AA, shift=4)
            self.blit(img, self.text_sprite(S.COMPONENTS[c][1], self.f_leg, INK), lx + 20, yy, 0.85 * a)
            row += 1
        # captions
        for (s0, s1, text) in CAP_SCHEDULE:
            if s0 - 0.5 <= s <= s1 + 0.5:
                a = smooth((s - s0 + 0.5) / 0.5) * smooth((s1 + 0.5 - s) / 0.5)
                spr = self.text_sprite(text, self.f_cap, INK, max_w=1300, align="center")
                x = W / 2 - spr.shape[1] / 2 + 90
                y = 985 - spr.shape[0] / 2
                self.shade(img, x - 30, y - 14, spr.shape[1] + 60, spr.shape[0] + 24, 0.5 * a)
                self.blit(img, spr, x, y, a)
        # title card
        if title_a > 0.003:
            self.shade(img, 0, 0, W, H, 0.55 * title_a)
            t = self.text_sprite("Grains of Ancestry", self.f_title, INK)
            self.blit(img, t, W / 2 - t.shape[1] / 2, 420, title_a)
            st = self.text_sprite("200,000 years of human migration and mixing, as sand", self.f_sub, MUTED)
            self.blit(img, st, W / 2 - st.shape[1] / 2, 530, title_a)
        # closing note
        if s > S.TODAY_S + 9.0:
            a = smooth((s - S.TODAY_S - 9.0) / 1.2)
            n = self.text_sprite("Proportions are approximate, simplified from published ancient-DNA studies. "
                                 "Vessels group neighbouring countries.", self.f_note, MUTED)
            self.blit(img, n, W - n.shape[1] - 40, 40, 0.8 * a)
        # highlight the islands of ancestry at the end
        t_i0, t_i1 = S.TODAY_S + 1.0, S.TODAY_S + 8.2
        if t_i0 <= s <= t_i1:
            a = smooth((s - t_i0) / 0.8) * smooth((t_i1 - s) / 0.8)
            for j in ("PNG", "ADS", "MDG"):
                bx0, by0, bx1, by1 = JARS[j].bbox
                rr = 1.0 + 0.04 * math.sin((s - t_i0) * 3)
                k = 0.7 * a
                col = tuple(float(v) for v in BG_BOT * (1 - k) + np.array([255, 235, 190]) * k)
                cv2.ellipse(img, (int((bx0 + bx1) / 2 * 16), int((by0 + by1) / 2 * 16)),
                            (int(((bx1 - bx0) / 2 + 26) * rr * 16), int(((by1 - by0) / 2 + 26) * rr * 16)),
                            0, 0, 360, col, 2, cv2.LINE_AA, shift=4)


# ============================================================== captions -> schedule
def caption_schedule():
    items = sorted((c[1], c[2]) if c[0] == "s" else (s_at(c[0]), c[1]) for c in S.CAPTIONS)
    out = []
    for i, (s0, text) in enumerate(items):
        s1 = s0 + max(3.6, len(text.split()) / 3.1)
        if i + 1 < len(items):
            s1 = min(s1, items[i + 1][0] - 1.0)
        else:
            s1 = S.DURATION - 0.6
        out.append((s0, s1, text))
    return out


CAP_SCHEDULE = caption_schedule()


# ============================================================== drivers
def cmd_still(args):
    os.makedirs(OUT, exist_ok=True)
    sim, painter = Sim(), Painter()
    want = sorted(float(v) for v in args.times)
    k = 0
    for f in range(int(round(S.DURATION * FPS)) + 1):
        s = f / FPS
        sim.step(s)
        while k < len(want) and want[k] <= s + 1e-9:
            p = os.path.join(OUT, f"still_{want[k]:07.2f}.png")
            cv2.imwrite(p, cv2.cvtColor(painter.frame(sim, s), cv2.COLOR_RGB2BGR))
            print("wrote", p, f"(ybp {ybp_at(s):,.0f})", flush=True)
            k += 1
        if k >= len(want):
            return


def render_chunk(job):
    """Worker: run the (deterministic) simulation from the start, paint frames [f0, f1) to a file."""
    f0, f1, path, scale, preview = job
    sim, painter = Sim(), Painter(scale)
    w, h = int(W * scale), int(H * scale)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast" if preview else "slow", "-crf", "24" if preview else "17",
                           "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f1):
        s = f / FPS
        sim.step(s)
        if f >= f0:
            ff.stdin.write(painter.frame(sim, s).tobytes())
            if (f - f0) % 300 == 0:
                print(f"  [{f0}-{f1}] frame {f}  s={s:6.1f}  {time.time() - t0:5.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    return path


def cmd_video(args):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    scale = 0.5 if args.preview else 1.0
    out = os.path.join(OUT, "preview.mp4" if args.preview else "grains_of_ancestry.mp4")
    n_frames = int(round(S.DURATION * FPS))
    nw = args.workers
    # later chunks re-simulate more frames before painting, so give them fewer frames to paint
    edges = [int(round(n_frames * (1 - (1 - k / nw) ** 1.0))) for k in range(nw + 1)]
    jobs = [(edges[k], edges[k + 1], os.path.join(OUT, f"chunk_{k}.mp4"), scale, args.preview) for k in range(nw)]
    t0 = time.time()
    with Pool(nw) as pool:
        parts = pool.map(render_chunk, jobs)
    lst = os.path.join(OUT, "chunks.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.basename(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    for p in parts:
        os.remove(p)
    os.remove(lst)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


def cmd_report(args):
    sim = Sim()
    for f in range(int(round(S.DURATION * FPS)) + 1):
        sim.step(f / FPS)
    print(f"{'vessel':22s}" + "".join(f"{k:>6s}" for k in S.KEYS) + "   grains/cap")
    for j in ORDER:
        jar = JARS[j]
        print(f"{jar.label:22s}" + "".join(f"{100 * p:6.1f}" for p in jar.P) + f"   {len(jar.L)}/{jar.cap}")
    print("captions:")
    for s0, s1, t in CAP_SCHEDULE:
        print(f"  {s0:6.1f}-{s1:6.1f}  {t[:70]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("still"); a.add_argument("times", nargs="+")
    b = sp.add_parser("video"); b.add_argument("--preview", action="store_true")
    b.add_argument("--workers", type=int, default=4)
    sp.add_parser("report")
    args = ap.parse_args()
    {"still": cmd_still, "video": cmd_video, "report": cmd_report}[args.cmd](args)
