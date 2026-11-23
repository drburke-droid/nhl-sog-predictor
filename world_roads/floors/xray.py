"""X-ray 3/4 view of New York's buildings: every floor slab of every building, drawn as translucent glass.

Each building is one or more prisms: its footprint from the city's data (roof height, floor count), or,
where OpenStreetMap maps building parts (setbacks, podiums, spires), those parts. Every floor of every
prism is a slab; slabs are projected with an orthographic camera and summed, so dense blocks glow brighter,
like an X-ray. Cyan slabs are not yet cleaned, gold ones are.

    import xray; C = xray.City(); img = xray.render(C, cam, level)
"""
import os, math
import numpy as np
import cv2
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "nyc")
LON0, LAT0 = -73.95, 40.70
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110574.0
ESB_BIN = "1015862"

CYAN = np.array([0.30, 0.74, 1.00], np.float32)
GOLD = np.array([1.00, 0.70, 0.28], np.float32)
BG = np.array([0.010, 0.016, 0.032], np.float32)


class City:
    def __init__(self, use_parts=True):
        b = np.load(os.path.join(D, "buildings.npz"), allow_pickle=True)
        self.xy, self.off = b["xy"].astype(np.float64), b["off"]
        self.height, self.floors, self.area = b["height"], b["floors"].astype(np.int32), b["area"]
        self.floor_area, self.bin = b["floor_area"], b["bin"]
        self.cx, self.cy = b["cx"], b["cy"]
        self.n = len(self.floors)
        h = np.where(self.height > 1, self.height, self.floors * 3.4)
        self.top = h
        self.storey = np.clip(h / np.maximum(self.floors, 1), 2.6, 6.0)
        self.esb = int(np.nonzero(self.bin == ESB_BIN)[0][0])
        # prisms: ring (start, end into ring arrays), z0, z1, nfl, building index
        rs, re, z0, z1, nfl, bi = [], [], [], [], [], []
        rings_xy = [self.xy]
        n0 = len(self.xy)
        replaced = np.zeros(self.n, bool)
        if use_parts:
            P = self._parts()
            if P is not None:
                pxy, poff, pz0, pz1, pnf, pb = P
                replaced[np.unique(pb)] = True
                rings_xy.append(pxy)
                rs += list(poff[:-1] + n0); re += list(poff[1:] + n0)
                z0 += list(pz0); z1 += list(pz1); nfl += list(pnf); bi += list(pb)
        keep = np.nonzero(~replaced)[0]
        rs = np.concatenate([self.off[:-1][keep], np.array(rs, np.int64)])
        re = np.concatenate([self.off[1:][keep], np.array(re, np.int64)])
        z0 = np.concatenate([np.zeros(len(keep)), np.array(z0)])
        z1 = np.concatenate([self.top[keep], np.array(z1)])
        nfl = np.concatenate([self.floors[keep], np.array(nfl, np.int32)])
        bi = np.concatenate([keep, np.array(bi, np.int64)])
        self.rxy = np.concatenate(rings_xy)
        self.ps, self.pe, self.pz0, self.pz1, self.pnfl, self.pb = rs, re, z0, z1, nfl.astype(np.int32), bi
        self.pcx = np.array([self.rxy[a:b, 0].mean() for a, b in zip(rs, re)])
        self.pcy = np.array([self.rxy[a:b, 1].mean() for a, b in zip(rs, re)])

    def _parts(self):
        path = os.path.join(D, "parts.npz")
        if not os.path.exists(path):
            return None
        p = np.load(path, allow_pickle=True)
        xy, off = p["xy"], p["off"]
        mx, my = (xy[:, 0] - LON0) * KX, (xy[:, 1] - LAT0) * KY
        pxy = np.stack([mx, my], 1)
        npart = len(off) - 1
        polys = [shapely.Polygon(pxy[off[k]:off[k + 1]]) for k in range(npart)]
        pcx = np.array([pxy[off[k]:off[k + 1], 0].mean() for k in range(npart)])
        pcy = np.array([pxy[off[k]:off[k + 1], 1].mean() for k in range(npart)])
        parea = np.array([q.area for q in polys])
        # footprints near the parts
        x0, x1, y0, y1 = mx.min() - 50, mx.max() + 50, my.min() - 50, my.max() + 50
        near = np.nonzero((self.cx > x0) & (self.cx < x1) & (self.cy > y0) & (self.cy < y1))[0]
        fpolys = [shapely.Polygon(self.xy[self.off[i]:self.off[i + 1]]) for i in near]
        tree = shapely.STRtree(fpolys)
        pi, fi = tree.query(shapely.points(pcx, pcy), predicate="within")
        owner = np.full(npart, -1, np.int64)
        owner[pi] = near[fi]
        h, mh, nf, mf = p["height"], p["min_height"], p["num_floors"], p["min_floor"]
        out_k, out_z0, out_z1, out_nf = [], [], [], []
        for b in np.unique(owner[owner >= 0]):
            ks = np.nonzero(owner == b)[0]
            st = self.storey[b]
            # parts must cover most of the footprint at ground level, or the footprint is kept as is
            fp = shapely.Polygon(self.xy[self.off[b]:self.off[b + 1]])
            u = shapely.union_all([polys[k].buffer(0) for k in ks])
            if u.intersection(fp.buffer(0)).area < 0.6 * fp.area:
                continue
            top = np.where(np.isfinite(h[ks]), h[ks], np.where(np.isfinite(nf[ks]), nf[ks] * st, np.nan))
            bot = np.where(np.isfinite(mh[ks]), mh[ks], np.where(np.isfinite(mf[ks]), mf[ks] * st, np.nan))
            for j, k in enumerate(ks):
                if not np.isfinite(top[j]):
                    continue
                if not np.isfinite(bot[j]):
                    # stacked on the highest lower part that it sits over, else on the ground
                    cand = [top[i] for i, kk in enumerate(ks) if i != j and np.isfinite(top[i]) and top[i] < top[j] - 1
                            and polys[kk].buffer(0.5).contains(shapely.Point(pcx[k], pcy[k]))]
                    bot[j] = max(cand) if cand else 0.0
                z0, z1 = float(bot[j]), float(top[j])
                if z1 - z0 < 1.0:
                    continue
                if np.isfinite(nf[k]) and nf[k] > 0:
                    n = int(nf[k] - (mf[k] if np.isfinite(mf[k]) else 0)) if np.isfinite(mf[k]) or z0 < 1 else int(round((z1 - z0) / st))
                else:
                    n = int(round((z1 - z0) / st))
                if parea[k] < 150 and z0 > 150:
                    n = 0                                   # a spire or mast: no floors
                out_k.append(k); out_z0.append(z0); out_z1.append(z1); out_nf.append(max(0, n))
        if not out_k:
            return None
        ks = np.array(out_k)
        rings = [pxy[off[k]:off[k + 1]] for k in ks]
        roff = np.concatenate([[0], np.cumsum([len(r) for r in rings])])
        return np.concatenate(rings), roff, np.array(out_z0), np.array(out_z1), np.array(out_nf), owner[ks]


class Cam:
    """Orthographic 3/4 camera. psi: compass heading the camera looks toward (deg); elev: degrees above the
    horizon; s: px per metre (at output size); (tx, ty, tz) the world point at screen (ox, oy)."""

    def __init__(self, psi, elev, s, tx, ty, tz=0.0, ox=960, oy=540, W=1920, H=1080, ss=2):
        self.psi, self.elev, self.s = math.radians(psi), math.radians(elev), s
        self.t = (tx, ty, tz)
        self.o = (ox, oy)
        self.W, self.H, self.ss = W, H, ss
        self.f = (math.sin(self.psi), math.cos(self.psi))
        self.r = (math.cos(self.psi), -math.sin(self.psi))

    def project(self, x, y, z):
        """world -> supersampled pixels"""
        dx, dy, dz = x - self.t[0], y - self.t[1], z - self.t[2]
        u = self.o[0] + self.s * (dx * self.r[0] + dy * self.r[1])
        v = self.o[1] - self.s * ((dx * self.f[0] + dy * self.f[1]) * math.sin(self.elev) + dz * math.cos(self.elev))
        return u * self.ss, v * self.ss

    def depth(self, x, y):
        return (x - self.t[0]) * self.f[0] + (y - self.t[1]) * self.f[1]


def _polys(C, cam, idx, z):
    """projected rings of prisms idx at heights z (per prism), as int32 arrays (cv2 shift=4)"""
    if len(idx) == 0:
        return []
    starts, ends = C.ps[idx], C.pe[idx]
    lens = ends - starts
    rep = np.repeat(np.arange(len(idx)), lens)
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
    u, v = cam.project(C.rxy[pos, 0], C.rxy[pos, 1], z[rep])
    pts = np.stack([u * 16, v * 16], 1).astype(np.int32)
    return np.split(pts, np.cumsum(lens)[:-1])


def visible(C, cam, margin=200):
    """prisms whose base or top lands near the screen"""
    W, H = cam.W * cam.ss, cam.H * cam.ss
    m = margin * cam.ss
    u0, v0 = cam.project(C.pcx, C.pcy, C.pz0)
    u1, v1 = cam.project(C.pcx, C.pcy, C.pz1)
    ok = (np.maximum(u0, u1) > -m) & (np.minimum(u0, u1) < W + m) & (np.maximum(v0, v1) > -m) & (np.minimum(v0, v1) < H + m)
    return np.nonzero(ok)[0]


def slabs(C, cam, level, idx=None, only=None, max_groups=400):
    """Sum of projected floor slabs. level[b]: height (m) below which building b's floors are clean.
    Returns (dirty, clean) float32 density images at supersampled size."""
    W, H = cam.W * cam.ss, cam.H * cam.ss
    dirty = np.zeros((H, W), np.float32)
    clean = np.zeros((H, W), np.float32)
    if idx is None:
        idx = visible(C, cam)
    if only is not None:
        idx = idx[only(idx)]
    nfl = C.pnfl[idx]
    canvas = np.zeros((H, W), np.float32)
    for j in range(min(int(nfl.max()) if len(nfl) else 0, max_groups)):
        k = idx[nfl > j]
        z = C.pz0[k] + j * (C.pz1[k] - C.pz0[k]) / C.pnfl[k]
        cl = z < level[C.pb[k]] - 0.01
        for sel, acc in ((~cl, dirty), (cl, clean)):
            kk = k[sel]
            if len(kk) == 0:
                continue
            canvas[:] = 0
            cv2.fillPoly(canvas, _polys(C, cam, kk, z[sel]), 1.0, lineType=cv2.LINE_8, shift=4)
            acc += canvas
    return dirty, clean


def edges(C, cam, idx, level, floor_lines=True, verticals=True, roofs=True):
    """outline layer (float, 0..1 per line) for close shots: slab outlines, vertical corners, roof rims.
    Returns (dirty_lines, clean_lines)."""
    W, H = cam.W * cam.ss, cam.H * cam.ss
    out = [np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)]
    acc = [np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
    lw = max(1, cam.ss)
    if floor_lines:
        nfl = C.pnfl[idx]
        for j in range(int(nfl.max()) if len(nfl) else 0):
            k = idx[nfl > j]
            z = C.pz0[k] + j * (C.pz1[k] - C.pz0[k]) / C.pnfl[k]
            cl = z < level[C.pb[k]] - 0.01
            for c, sel in ((0, ~cl), (1, cl)):
                if sel.any():
                    out[c][:] = 0
                    cv2.polylines(out[c], _polys(C, cam, k[sel], z[sel]), True, 255, lw, cv2.LINE_AA, shift=4)
                    acc[c] += out[c].astype(np.float32) / 255
    if roofs:
        cl = C.pz1[idx] <= level[C.pb[idx]] + 0.01
        for c, sel in ((0, ~cl), (1, cl)):
            if sel.any():
                out[c][:] = 0
                cv2.polylines(out[c], _polys(C, cam, idx[sel], C.pz1[idx[sel]]), True, 255, lw, cv2.LINE_AA, shift=4)
                acc[c] += 1.6 * out[c].astype(np.float32) / 255
    if verticals:
        segs = [[], []]
        for k in idx:
            r = C.rxy[C.ps[k]:C.pe[k] - 1]
            if len(r) > 40:
                r = r[:: int(math.ceil(len(r) / 40))]
            lv = level[C.pb[k]]
            u0, v0 = cam.project(r[:, 0], r[:, 1], np.full(len(r), C.pz0[k]))
            zm = min(max(lv, C.pz0[k]), C.pz1[k])
            um, vm = cam.project(r[:, 0], r[:, 1], np.full(len(r), zm))
            u1, v1 = cam.project(r[:, 0], r[:, 1], np.full(len(r), C.pz1[k]))
            for a in range(len(r)):
                if zm > C.pz0[k]:
                    segs[1].append(((u0[a], v0[a]), (um[a], vm[a])))
                if zm < C.pz1[k]:
                    segs[0].append(((um[a], vm[a]), (u1[a], v1[a])))
        for c in (0, 1):
            if segs[c]:
                out[c][:] = 0
                pl = [np.array([[p[0] * 16, p[1] * 16], [q[0] * 16, q[1] * 16]], np.int32) for p, q in segs[c]]
                cv2.polylines(out[c], pl, False, 255, lw, cv2.LINE_AA, shift=4)
                acc[c] += 0.8 * out[c].astype(np.float32) / 255
    return acc


def bloom(img, ss=2, k=(0.22, 0.14, 0.08), s=(3, 12, 40)):
    out = img.copy()
    for kk, sg in zip(k, s):
        out += kk * cv2.GaussianBlur(img, (0, 0), sg * ss)
    return out


def tonemap(img):
    """filmic shoulder on brightness, hottest light goes white (as in the road films)"""
    lum = img.max(axis=2, keepdims=True)
    tm = 1 - np.exp(-lum * 1.6)
    out = img * (tm / np.maximum(lum, 1e-6))
    white = np.clip((lum - 1.3) / 3.0, 0, 1)
    out = out * (1 - white) + tm * white
    return np.clip(out, 0, 1)


def down(img, ss):
    if ss == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w // ss, h // ss), interpolation=cv2.INTER_AREA)


def compose(dirty, clean, gain=(0.10, 0.16), sat=(0.9, 1.0), lines=None, line_gain=(0.35, 0.55), extra=None, ss=2,
            bg=None, bloom_k=(0.22, 0.14, 0.08)):
    """density images -> RGB float (output size, tone-mapped 0..1)"""
    H, W = dirty.shape
    img = np.zeros((H, W, 3), np.float32) + (BG if bg is None else bg)
    img += (sat[0] * (1 - np.exp(-dirty * gain[0])))[..., None] * CYAN
    img += (sat[1] * (1 - np.exp(-clean * gain[1])))[..., None] * GOLD * 1.25
    if lines is not None:
        img += (line_gain[0] * lines[0])[..., None] * CYAN
        img += (line_gain[1] * lines[1])[..., None] * GOLD * 1.3
    if extra is not None:
        img += extra
    img = bloom(img, ss, k=bloom_k)
    return tonemap(down(img, ss))


def spot(H, W, x, y, s):
    yy, xx = np.ogrid[:H, :W]
    return np.exp(-(((xx - x) / s) ** 2 + ((yy - y) / s) ** 2) / 2).astype(np.float32)


def light(H, W, x, y, ss=2, size=1.0, streak=None, warm=True):
    """the white-hot light at (x, y) supersampled px; streak: (dx, dy) unit screen direction of travel"""
    g = np.zeros((H, W, 3), np.float32)
    col = GOLD if warm else np.array([1, 1, 1], np.float32)
    g += spot(H, W, x, y, 3.0 * ss * size)[..., None] * 7.0
    g += spot(H, W, x, y, 14 * ss * size)[..., None] * (1.0 * col + 0.6)
    g += spot(H, W, x, y, 60 * ss * size)[..., None] * 0.35 * col
    if streak is not None:
        dx, dy = streak
        yy, xx = np.ogrid[:H, :W]
        along = (xx - x) * dx + (yy - y) * dy
        perp = -(xx - x) * dy + (yy - y) * dx
        tail = np.exp(-(perp / (2.2 * ss * size)) ** 2 / 2) * np.where(along < 0, np.exp(along / (260 * ss * size)), np.exp(-(along / (6 * ss)) ** 2))
        g += tail[..., None].astype(np.float32) * (1.4 * col + 0.8)
    return g
