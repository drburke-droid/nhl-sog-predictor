"""Fast X-ray city frames for the film.

With an orthographic camera, zooming and panning are 2D scales and shifts of the image. So the slab density
of every building is rendered once, at three scales ("plates"), and each frame samples the plate that suits
its zoom (area-averaged mip level + sub-pixel affine). Cleaned buildings are added to a second plate as the
light reaches them, so a frame costs one sample, not a re-render. The Empire State Building is drawn per
frame at the output camera (it is the hero, and it is partly clean for a while).

    R = CityRenderer(levels=("P1", "P2", "P3"))
    R.advance(L)                     # light time (s): add every building cleaned by then
    img = R.frame(cam)               # float RGB at 1920x1080, before bloom and tone mapping
"""
import os, math, time
import numpy as np
import cv2
import xray as X

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "nyc")
PD = os.path.join(HERE, "data", "plates")
PSI, EL = 34.0, 30.0
W, H = 1920, 1080
C_MS = 299792458.0

# plate: scale (px/m), size, and where the Empire State Building's foot lands (offsets in metres from it)
PLATES = {
    "P1": dict(s=2.5, PW=8400, PH=4800, ox=4200, oy=2900, dx=0.0, dy=0.0, lines=True),
    "P2": dict(s=0.6, PW=10800, PH=6400, ox=4600, oy=4000, dx=0.0, dy=0.0, lines=False),
    "P3": dict(s=0.12, PW=7200, PH=4400, ox=2600, oy=2600, dx=0.0, dy=0.0, lines=False),
}


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Route:
    def __init__(self, C):
        r = np.load(os.path.join(D, "route.npz"))
        self.order, self.ts, self.te = r["order"], r["t_start"], r["t_end"]
        self.rank = np.empty(len(self.order), np.int64)
        self.rank[self.order] = np.arange(len(self.order))
        self.te_sorted = self.te[self.order]
        self.cum_fa = np.concatenate([[0.0], np.cumsum(C.floor_area[self.order])])
        self.C = C

    def done_count(self, L):
        return int(np.searchsorted(self.te_sorted, L, side="right"))

    def progress(self, L):
        """(buildings done, floor area cleaned m^2) at light time L"""
        n = self.done_count(L)
        a = self.cum_fa[n]
        if n < len(self.order):
            b = self.order[n]
            if self.ts[b] <= L:
                a += self.C.floor_area[b] * (L - self.ts[b]) / (self.te[b] - self.ts[b])
        return n, float(a)

    def frontier(self, L):
        n = min(self.done_count(L), len(self.order) - 1)
        b = self.order[n]
        f = np.clip((L - self.ts[b]) / max(self.te[b] - self.ts[b], 1e-15), 0, 1)
        return b, f


class Plate:
    def __init__(self, C, R, name, esb):
        p = PLATES[name]
        self.name, self.s = name, p["s"]
        self.cam = X.Cam(PSI, EL, p["s"], C.cx[esb] + p["dx"], C.cy[esb] + p["dy"], 0.0, ox=p["ox"], oy=p["oy"],
                         W=p["PW"], H=p["PH"], ss=1)
        idx = X.visible(C, self.cam, margin=20)
        idx = idx[C.pb[idx] != esb]
        rk = R.rank[C.pb[idx]]
        o = np.argsort(rk, kind="stable")
        self.idx, self.rk = idx[o], rk[o]
        self.C = C
        self.lines = p["lines"]
        os.makedirs(PD, exist_ok=True)
        tp = os.path.join(PD, f"{name}_total.npy")
        if not os.path.exists(tp):
            t0 = time.time()
            tot = self._raster(self.idx, lines=False)
            np.save(tp, tot)
            if self.lines:
                np.save(os.path.join(PD, f"{name}_lines.npy"), self._raster(self.idx, lines=True))
            print(f"plate {name}: {len(self.idx):,} prisms, {time.time() - t0:.0f}s", flush=True)
        self.total = np.load(tp)
        self.ltotal = np.load(os.path.join(PD, f"{name}_lines.npy")) if self.lines else None
        self.clean = np.zeros_like(self.total)
        self.lclean = np.zeros_like(self.total) if self.lines else None
        self.n = 0                                        # buildings (route order) already added

    def _raster(self, idx, lines=False, out=None):
        """slab density (or roof outlines) of prisms idx onto the plate, using a local canvas per batch bbox"""
        C, cam = self.C, self.cam
        PH, PW = cam.H, cam.W
        if out is None:
            out = np.zeros((PH, PW), np.float32)
        if len(idx) == 0:
            return out
        # bbox in plate pixels
        u0, v0 = cam.project(C.pcx[idx], C.pcy[idx], C.pz0[idx])
        u1, v1 = cam.project(C.pcx[idx], C.pcy[idx], C.pz1[idx])
        ext = 120 * self.s + 4                             # half-size of a big footprint, px
        x0 = int(max(0, math.floor(min(u0.min(), u1.min()) - ext)))
        x1 = int(min(PW, math.ceil(max(u0.max(), u1.max()) + ext)))
        y0 = int(max(0, math.floor(min(v0.min(), v1.min()) - ext)))
        y1 = int(min(PH, math.ceil(max(v0.max(), v1.max()) + ext)))
        if x1 <= x0 or y1 <= y0:
            return out
        sub = X.Cam(PSI, EL, self.s, cam.t[0], cam.t[1], 0.0, ox=cam.o[0] - x0, oy=cam.o[1] - y0, W=x1 - x0, H=y1 - y0, ss=1)
        acc = out[y0:y1, x0:x1]
        canvas = np.zeros((y1 - y0, x1 - x0), np.float32)
        if lines:
            cv2.polylines(canvas, X._polys(C, sub, idx, C.pz1[idx]), True, 1.0, 1, cv2.LINE_8, shift=4)
            np.maximum(acc, canvas, out=acc)
            return out
        nfl = C.pnfl[idx]
        for j in range(int(nfl.max()) if len(nfl) else 0):
            k = idx[nfl > j]
            z = C.pz0[k] + j * (C.pz1[k] - C.pz0[k]) / C.pnfl[k]
            canvas[:] = 0
            cv2.fillPoly(canvas, X._polys(C, sub, k, z), 1.0, lineType=cv2.LINE_8, shift=4)
            acc += canvas
        return out

    def advance(self, n):
        """add buildings [self.n, n) in route order"""
        if n <= self.n:
            return
        a, b = np.searchsorted(self.rk, [self.n, n])
        k = self.idx[a:b]
        step = 6000
        for i in range(0, len(k), step):
            self._raster(k[i:i + step], out=self.clean)
            if self.lines:
                kk = k[i:i + step]
                self._raster(kk[self.C.pnfl[kk] > 0], lines=True, out=self.lclean)     # spires stay uncleaned
        self.n = n

    def sample(self, arr, cam):
        """arr (plate pixels) seen through output camera cam (same psi/elev, any scale/target): output-size image"""
        k = cam.s / self.s
        tu, tv = cam.project(np.array([self.cam.t[0]]), np.array([self.cam.t[1]]), np.array([0.0]))
        bx, by = tu[0] / cam.ss - k * self.cam.o[0], tv[0] / cam.ss - k * self.cam.o[1]
        m = max(0, int(math.floor(math.log2(1.0 / k)))) if k < 1 else 0
        f = 2 ** m
        Wo, Ho = cam.W, cam.H
        px0, px1 = (0 - bx) / k, (Wo - bx) / k
        py0, py1 = (0 - by) / k, (Ho - by) / k
        x0 = max(0, (int(math.floor(px0)) // f - 2) * f)
        y0 = max(0, (int(math.floor(py0)) // f - 2) * f)
        x1 = min(arr.shape[1], (int(math.ceil(px1)) // f + 3) * f)
        y1 = min(arr.shape[0], (int(math.ceil(py1)) // f + 3) * f)
        x1 = x0 + ((x1 - x0) // f) * f
        y1 = y0 + ((y1 - y0) // f) * f
        if x1 - x0 < f or y1 - y0 < f:
            return np.zeros((Ho, Wo), np.float32)
        crop = arr[y0:y1, x0:x1]
        small = crop if f == 1 else cv2.resize(crop, ((x1 - x0) // f, (y1 - y0) // f), interpolation=cv2.INTER_AREA)
        M = np.array([[k * f, 0, k * (x0 + (f - 1) / 2) + bx], [0, k * f, k * (y0 + (f - 1) / 2) + by]], np.float64)
        return cv2.warpAffine(small, M, (Wo, Ho), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def interp_log(s, table):
    """table: [(scale, value)] sorted by scale descending; value interpolated in log(scale)"""
    ls = math.log(s)
    xs = [math.log(a) for a, _ in table]
    vs = [np.asarray(v, np.float64) for _, v in table]
    fix = lambda r: float(r) if r.ndim == 0 else tuple(float(q) for q in r)
    if ls >= xs[0]:
        return fix(vs[0])
    if ls <= xs[-1]:
        return fix(vs[-1])
    for i in range(len(xs) - 1):
        if xs[i] >= ls >= xs[i + 1]:
            u = (ls - xs[i]) / (xs[i + 1] - xs[i])
            r = vs[i] * (1 - u) + vs[i + 1] * u
            return float(r) if r.ndim == 0 else tuple(float(q) for q in r)


# look vs zoom: (scale px/m, (dirty gain, dirty sat, clean gain, clean sat, roof-line gain))
LOOK = [(2.05, (0.10, 0.16, 0.10, 0.34, 0.15)),
        (1.0, (0.08, 0.45, 0.09, 0.65, 0.22)),
        (0.55, (0.05, 0.9, 0.09, 1.0, 0.22)),
        (0.25, (0.04, 0.9, 0.08, 1.0, 0.0)),
        (0.105, (0.035, 0.9, 0.07, 1.0, 0.0)),
        (0.036, (0.022, 0.9, 0.10, 1.0, 0.0)),
        (0.01, (0.05, 0.9, 0.25, 1.0, 0.0)),
        (0.003, (0.20, 0.9, 0.9, 1.0, 0.0)),
        (0.0009, (0.8, 0.9, 3.0, 1.0, 0.0))]


class CityRenderer:
    def __init__(self, levels=("P1", "P2", "P3")):
        self.C = X.City()
        self.R = Route(self.C)
        self.esb = self.C.esb
        self.plates = {nm: Plate(self.C, self.R, nm, self.esb) for nm in levels}
        self.esb_k = np.nonzero(self.C.pb == self.esb)[0]
        # the Empire State Building's floors, bottom to top, with cumulative share of its floor area
        sl = []
        for k in self.esb_k:
            ring = self.C.rxy[self.C.ps[k]:self.C.pe[k]][:-1]
            x, y = ring[:, 0], ring[:, 1]
            a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            for j in range(self.C.pnfl[k]):
                sl.append((self.C.pz0[k] + j * (self.C.pz1[k] - self.C.pz0[k]) / self.C.pnfl[k], k, j, a))
        sl.sort()
        self.esb_slabs = sl
        A = np.array([q[3] for q in sl])
        self.esb_cum = np.concatenate([[0], np.cumsum(A)]) / A.sum()
        self.esb_t0, self.esb_t1 = self.R.ts[self.esb], self.R.te[self.esb]

    # ------------------------------------------------------------- the Empire State Building
    def esb_state(self, L):
        """(level z below which ESB is clean, active slab index or None, fraction of that slab)"""
        if L <= self.esb_t0:
            return -1.0, None, 0.0
        if L >= self.esb_t1:
            return 1e9, None, 1.0
        q = (L - self.esb_t0) / (self.esb_t1 - self.esb_t0)
        i = int(np.searchsorted(self.esb_cum, q, side="right") - 1)
        i = min(i, len(self.esb_slabs) - 1)
        f = (q - self.esb_cum[i]) / max(self.esb_cum[i + 1] - self.esb_cum[i], 1e-12)
        return self.esb_slabs[i][0] - 0.01, i, f

    def esb_time_at_slab(self, i, f=0.0):
        q = self.esb_cum[i] + f * (self.esb_cum[i + 1] - self.esb_cum[i])
        return self.esb_t0 + q * (self.esb_t1 - self.esb_t0)

    def advance(self, L):
        n = self.R.done_count(L)
        for p in self.plates.values():
            p.advance(n)

    def plate_for(self, s):
        names = [nm for nm in ("P1", "P2", "P3") if nm in self.plates]
        best = None
        for nm in names:                                        # the coarsest plate still at least as fine as the output
            if self.plates[nm].s >= s * 0.999:
                best = nm
        return best or names[0]

    def layers(self, cam):
        """(dirty, clean, roof lines dirty, roof lines clean) at output size, crossfading plates near a switch"""
        s = cam.s
        nm = self.plate_for(s)
        out = self._layers_from(nm, cam)
        # blend toward the next coarser plate just above its scale, so the switch doesn't pop
        names = [n for n in ("P1", "P2", "P3") if n in self.plates]
        i = names.index(nm)
        if i + 1 < len(names):
            nxt = self.plates[names[i + 1]]
            u = (nxt.s * 1.35 - s) / (nxt.s * 0.35)            # 0 at 1.35x its scale, 1 at its scale
            if 0 < u < 1:
                o2 = self._layers_from(names[i + 1], cam)
                out = tuple(None if a is None else (a * (1 - u) + (b if b is not None else 0) * u) for a, b in zip(out, o2))
        return out

    def _layers_from(self, nm, cam):
        p = self.plates[nm]
        tot = p.sample(p.total, cam)
        cl = p.sample(p.clean, cam)
        if p.lines:
            lt = p.sample(p.ltotal, cam)
            lc = p.sample(p.lclean, cam)
            return np.maximum(tot - cl, 0), cl, np.maximum(lt - lc, 0), lc
        return np.maximum(tot - cl, 0), cl, None, None

    def frame(self, cam, L, esb_detail=True, look=None, nb=1.0, **esb_kw):
        """composite (float RGB, output size, linear before bloom); nb scales the surrounding city (0 = the
        Empire State Building alone, for cameras the plates weren't made for)"""
        img = np.zeros((cam.H, cam.W, 3), np.float32) + X.BG
        if nb > 0:
            gd, sd, gc, sc, lg = look or interp_log(cam.s, LOOK)
            dirty, clean, ld, lc = self.layers(cam)
            img += nb * (sd * (1 - np.exp(-dirty * gd)))[..., None] * X.CYAN
            img += nb * (sc * (1 - np.exp(-clean * gc)))[..., None] * X.GOLD * 1.25
            if ld is not None and lg > 0:
                img += nb * ((lg * np.minimum(ld, 1))[..., None] * X.CYAN + (lg * 1.3 * np.minimum(lc, 1))[..., None] * X.GOLD)
        img += self.esb_layer(cam, L, detail=esb_detail, **esb_kw)
        return img

    def esb_layer(self, cam, L, detail=True, above_gain=0.55, below_gain=0.85):
        """the Empire State Building at the output camera, drawn on a canvas just its size (supersampled when big)"""
        C = self.C
        out = np.zeros((cam.H, cam.W, 3), np.float32)
        ks = self.esb_k
        allxy = np.concatenate([C.rxy[C.ps[k]:C.pe[k]] for k in ks])
        zs = np.concatenate([np.full(C.pe[k] - C.ps[k], C.pz0[k]) for k in ks] + [np.full(C.pe[k] - C.ps[k], C.pz1[k]) for k in ks])
        uu, vv = cam.project(np.concatenate([allxy[:, 0]] * 2), np.concatenate([allxy[:, 1]] * 2), zs)
        uu, vv = uu / cam.ss, vv / cam.ss
        x0, x1 = int(max(0, uu.min() - 6)), int(min(cam.W, uu.max() + 7))
        y0, y1 = int(max(0, vv.min() - 6)), int(min(cam.H, vv.max() + 7))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return out
        ss = 2 if cam.s > 0.8 else 1
        c2 = X.Cam(math.degrees(cam.psi), math.degrees(cam.elev), cam.s, cam.t[0], cam.t[1], cam.t[2], ox=cam.o[0] - x0,
                   oy=cam.o[1] - y0, W=x1 - x0, H=y1 - y0, ss=ss)
        level = np.full(C.n, -1.0)
        level[self.esb], _, _ = self.esb_state(L)
        d0, c0 = X.slabs(C, c2, level, ks)
        sc = interp_log(cam.s, [(2.05, 1.0), (0.55, 0.0)])          # hero look up close, city look further out
        gd, sd, gc, scc, _ = interp_log(cam.s, LOOK)
        a = (1 - np.exp(-d0 * 0.25))[..., None] * X.CYAN * above_gain * sc + (sd * (1 - np.exp(-d0 * gd)))[..., None] * X.CYAN * (1 - sc)
        a += (1 - np.exp(-c0 * 0.30))[..., None] * X.GOLD * below_gain * sc + (scc * (1 - np.exp(-c0 * gc)))[..., None] * X.GOLD * 1.25 * (1 - sc)
        if detail and cam.s > 0.5:
            Lq = X.edges(C, c2, ks, level, verticals=True)
            a += ((0.30 * Lq[0])[..., None] * X.CYAN + (0.45 * Lq[1])[..., None] * X.GOLD) * sc
        out[y0:y1, x0:x1] = X.down(a, ss)[:y1 - y0, :x1 - x0]
        return out

    def frontier_point(self, L):
        """world (x, y, z) of the light at light time L"""
        C = self.C
        if self.esb_t0 < L < self.esb_t1:
            z, i, f = self.esb_state(L)
            return C.cx[self.esb], C.cy[self.esb], self.esb_slabs[i][0]
        b, f = self.R.frontier(L)
        return C.cx[b], C.cy[b], f * C.top[b]


def light_win(img, x, y, size=1.0, gain=1.0):
    """the white-hot light (as xray.light) added in place, computed only in a window around (x, y)"""
    H_, W_ = img.shape[:2]
    R = int(60 * size * 4) + 4
    x0, x1 = max(0, int(x) - R), min(W_, int(x) + R)
    y0, y1 = max(0, int(y) - R), min(H_, int(y) + R)
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    col = X.GOLD
    g = (np.exp(-d2 / (2 * (3.0 * size) ** 2)) * 7.0)[..., None] \
        + (np.exp(-d2 / (2 * (14 * size) ** 2)))[..., None] * (1.0 * col + 0.6) \
        + (np.exp(-d2 / (2 * (60 * size) ** 2)) * 0.35)[..., None] * col
    img[y0:y1, x0:x1] += g * gain
    return img


def finish(img, bloom=(0.22, 0.14, 0.08)):
    """bloom + tone map at output size -> uint8"""
    out = img.copy()
    for k, s in zip(bloom, (3, 12, 40)):
        out += k * cv2.GaussianBlur(img, (0, 0), s)
    return (np.clip(X.tonemap(out), 0, 1) * 255 + 0.5).astype(np.uint8)
