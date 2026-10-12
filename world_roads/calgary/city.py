"""Calgary at light speed: every public street inside city limits, driven by one point of light.

    python fetch_city.py && python fetch_boundary.py   # Overture data for the Calgary area
    python city.py                                    # graph + route + vertical Short + GIF

Same method as the world film at street scale: a greedy T-join plus Hierholzer Euler path per
piece, pieces ordered in rings spreading out from downtown, and Dijkstra deadheads between them.
"""
import os, sys, json, subprocess
import numpy as np
import cv2
import shapely
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import route as R  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
C_KMS = 299_792.458
STREETS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "unknown"]
WEIGHT = {"motorway": (1.0, 3), "trunk": (1.0, 3), "primary": (0.9, 3), "secondary": (0.85, 2),
          "tertiary": (0.75, 2), "unclassified": (0.55, 1), "residential": (0.5, 1),
          "living_street": (0.5, 1), "unknown": (0.45, 1)}
DOWNTOWN = (-114.0631, 51.0461)   # Centre Street & 8 Ave SW
LAT0 = 51.04
KX = np.radians(1) * 6371.0088 * np.cos(np.radians(LAT0))
KY = np.radians(1) * 6371.0088

VW, VH, FPS = 1080, 1920, 30
MAP_TOP, MAP_H = 440, 900
ROUTE_S = 12.0            # seconds of video for the whole walk
FD = "/usr/share/fonts/opentype/inter"


def xy(lon, lat):
    return (np.asarray(lon) - DOWNTOWN[0]) * KX, (np.asarray(lat) - DOWNTOWN[1]) * KY


def build():
    t = pq.read_table(os.path.join(DATA, "calgary_segments.parquet"))
    boundary = shapely.from_wkt(open(os.path.join(DATA, "calgary_boundary.wkt")).read())
    cls = np.array(t.column("class").to_pylist(), dtype=object)
    keep = np.isin(cls, STREETS)
    geoms = shapely.intersection(shapely.from_wkb(t.column("geometry").to_numpy(zero_copy_only=False)[keep]), boundary)
    cls = cls[keep]
    parts = shapely.get_parts(geoms, return_index=True)
    lines, gi = parts
    ok = shapely.get_type_id(lines) == 1
    lines, gi = lines[ok], gi[ok]
    coords, idx = shapely.get_coordinates(lines, return_index=True)
    q = np.round(coords * 2e5).astype(np.int64)            # ~0.5 m snapping merges shared connectors
    code = q[:, 0] * 10**9 + q[:, 1]
    nodes, inv = np.unique(code, return_inverse=True)
    same = idx[1:] == idx[:-1]
    a, b = inv[:-1][same], inv[1:][same]
    ecls = cls[gi[idx[:-1][same]]]
    good = a != b
    a, b, ecls = a[good], b[good], ecls[good]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    u, first = np.unique(lo * len(nodes) + hi, return_index=True)
    ea, eb, ecls = lo[first], hi[first], ecls[first]
    nlon = (nodes // 10**9) / 2e5
    nlat = (nodes % 10**9) / 2e5
    nx, ny = xy(nlon, nlat)
    elen = np.hypot(nx[ea] - nx[eb], ny[ea] - ny[eb])
    n = len(nodes)
    comp = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n)), directed=False)[1]
    mx, my = (nx[ea] + nx[eb]) / 2, (ny[ea] + ny[eb]) / 2
    key = np.hypot(mx, my)                                   # rings spreading out from downtown
    by = {c: float(elen[ecls == c].sum()) for c in STREETS}
    print("streets km:", round(elen.sum()), {k: round(v) for k, v in by.items()},
          "nodes", n, "edges", len(ea), "components", comp.max() + 1)
    return dict(nx=nx, ny=ny, ea=ea.astype(np.int64), eb=eb.astype(np.int64), elen=elen, key=key,
                comp=comp, ecls=ecls, boundary=boundary, by_class=by)


def walk(G, band=5.0):
    R.MATCH_KM, R.ALPHA = 3.0, 0.02
    ea, eb, elen, key, comp, nx, ny = (G[k] for k in ("ea", "eb", "elen", "key", "comp", "nx", "ny"))
    n = len(nx)
    indptr, inc = R.csr(n, ea, eb)
    bnd = np.floor(key / band).astype(np.int64)
    out_n, out_e = [np.array([int(ea[np.argmin(key)])])], [np.array([-2])]
    cur = int(out_n[0][0])
    target = np.full(n, -1, np.int64)
    for b in np.unique(bnd):
        be = np.nonzero(bnd == b)[0]
        bn, binv = np.unique(np.r_[ea[be], eb[be]], return_inverse=True)
        k = len(be)
        lab = connected_components(coo_matrix((np.ones(k), (binv[:k], binv[k:])), shape=(len(bn), len(bn))),
                                   directed=False)[1]
        elab = lab[binv[:k]]
        o = np.argsort(elab, kind="stable")
        cuts = np.r_[0, np.nonzero(np.diff(elab[o]))[0] + 1, k]
        pieces = [be[o[i:j]] for i, j in zip(cuts[:-1], cuts[1:])]
        pcomp = np.array([comp[ea[p[0]]] for p in pieces])
        pending = set(range(len(pieces)))
        target[bn] = lab
        while pending:
            if any(pcomp[p] == comp[cur] for p in pending):
                v, path = R.nearest_target(cur, indptr, inc, ea, eb, elen, target)
                if len(path):
                    pn = np.empty(len(path), np.int64); u = cur
                    for i, e in enumerate(path):
                        u = eb[e] if ea[e] == u else ea[e]; pn[i] = u
                    out_n.append(pn); out_e.append(path)
            else:
                cand = bn[np.isin(lab, list(pending))]
                v = int(cand[np.argmin((nx[cand] - nx[cur]) ** 2 + (ny[cand] - ny[cur]) ** 2)])
                out_n.append(np.array([v])); out_e.append(np.array([-1]))
            p = int(target[v])
            rn, re_ = R.euler_piece(pieces[p], v, ea, eb, elen, key, nx, ny)
            out_n.append(rn[1:]); out_e.append(re_[1:])
            cur = int(rn[-1]); pending.discard(p)
            target[bn[lab == p]] = -1
        target[bn] = -1
    rn, re_ = np.concatenate(out_n), np.concatenate(out_e)
    step = np.where(re_ >= 0, elen[np.maximum(re_, 0)], 0.0)
    cum = np.cumsum(step)
    seen = np.zeros(len(ea), bool); seen[re_[re_ >= 0]] = True
    assert seen.all()
    stats = {"street_km": float(elen.sum()), "walk_km": float(cum[-1]),
             "overhead": float(cum[-1] / elen.sum() - 1), "jumps": int((re_ == -1).sum()),
             "seconds_at_c": float(cum[-1] / C_KMS)}
    print(json.dumps(stats, indent=1))
    return rn, re_, cum, stats


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def sprite(sig, size):
    k = np.arange(size, dtype=np.float32) - size // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


def render(G, rn, re_, cum, stats):
    os.makedirs(OUT, exist_ok=True)
    nx, ny, ea, eb = G["nx"], G["ny"], G["ea"], G["eb"]
    bx0, by0, bx1, by1 = G["boundary"].bounds
    (x0, x1), (y0, y1) = [xy([bx0, bx1], [by0, by1])[0], xy([bx0, bx1], [by0, by1])[1]]
    sc = MAP_H / (y1 - y0)                                   # px per km
    MW = int(np.ceil((x1 - x0) * sc)); MH = MAP_H
    S = 2
    px = lambda X, Y: (np.c_[(X - x0) * sc, (y1 - Y) * sc] * S * 8).round().astype(np.int32)  # shift=3

    T = ROUTE_S * cum / cum[-1]
    Tprev = np.r_[0.0, T[:-1]]
    m = len(ea)
    tv = np.full(m, np.inf)
    st = np.nonzero(re_ >= 0)[0][::-1]
    tv[re_[st]] = 0.5 * (Tprev[st] + T[st])
    P = np.stack([px(nx[ea], ny[ea]), px(nx[eb], ny[eb])], 1)
    I = np.zeros((MH * S, MW * S), np.uint8)
    for c in sorted(STREETS, key=lambda c: WEIGHT[c][0]):
        sel = np.nonzero(G["ecls"] == c)[0]
        if len(sel):
            cv2.polylines(I, list(P[sel]), False, int(255 * WEIGHT[c][0]), WEIGHT[c][1], cv2.LINE_AA, shift=3)
    vt = np.full((MH * S, MW * S), np.inf, np.float32)
    fb = np.floor(tv * FPS * 4).astype(np.int64)
    order = np.argsort(-fb, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(fb[order]))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = order[a:b]
        cv2.polylines(vt, list(P[sel]), False, float(tv[sel].min()), 3, cv2.LINE_8, shift=3)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~np.isfinite(vt), return_distances=False, return_indices=True)
    vt = vt[idx[0], idx[1]]
    I = I.astype(np.float32) / 255
    # city-limits outline
    outline = np.zeros((MH * S, MW * S), np.uint8)
    bd = G["boundary"]
    for poly in getattr(bd, "geoms", [bd]):
        bx, byy = xy(*np.asarray(poly.exterior.coords).T)
        cv2.polylines(outline, [px(bx, byy)], True, 255, 2, cv2.LINE_AA, shift=3)
    outline = cv2.resize(outline, (MW, MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    CYAN = np.array([0.14, 0.66, 1.00], np.float32) * 0.6
    GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.95
    HOT = np.array([1.00, 0.93, 0.78], np.float32)
    dot = (sprite(1.6, 121)[..., None] * np.array([1, .97, .9]) * 2.6 + sprite(4.5, 121)[..., None] *
           np.array([1, .72, .3]) * 1.3 + sprite(16, 121)[..., None] * np.array([1, .52, .12]) * 0.45).astype(np.float32)
    rx, ry = (nx[rn] - x0) * sc, (y1 - ny[rn]) * sc

    total_ms = stats["seconds_at_c"] * 1000
    slow = ROUTE_S / stats["seconds_at_c"]
    hold0, hold1 = 1.0, 3.5
    n_frames = int((hold0 + ROUTE_S + hold1) * FPS)
    mx0 = (VW - MW) // 2
    hook = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(hook)
    for i, (txt, sz, col) in enumerate([("Every street in Calgary", 72, (240, 244, 250)),
                                        ("at the speed of light", 72, (240, 244, 250)),
                                        ("How long does it take?", 56, (255, 186, 80))]):
        d.text((VW // 2, 170 + [0, 88, 182][i]), txt, font=font("Inter-Bold.otf" if sz > 60 else "Inter-SemiBold.otf", sz),
               fill=col + (255,), anchor="ma")
    hook_a = np.asarray(hook).astype(np.float32) / 255
    f_clock = font("InterDisplay-Bold.otf", 120)
    path = os.path.join(OUT, "calgary_light_speed.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for f in range(n_frames):
        tf = f / FPS
        t = float(np.clip(tf - hold0, 0, ROUTE_S))
        end = tf > hold0 + ROUTE_S
        gold = (vt <= t).astype(np.float32)
        age = t - vt
        flash = np.where((age >= 0) & (age < 0.35), (1 - age / 0.35) ** 2, 0).astype(np.float32) * 1.6
        Ig = cv2.resize(I * gold, (MW, MH), interpolation=cv2.INTER_AREA)
        Ic = cv2.resize(I * (1 - gold), (MW, MH), interpolation=cv2.INTER_AREA)
        Fl = cv2.resize(I * flash, (MW, MH), interpolation=cv2.INTER_AREA)
        E = Ic[..., None] * CYAN + Ig[..., None] * GOLD + Fl[..., None] * HOT + outline[..., None] * np.array([.08, .14, .22], np.float32)
        if hold0 <= tf <= hold0 + ROUTE_S and t < ROUTE_S:
            i = min(max(int(np.searchsorted(T, t)), 1), len(T) - 1)
            u = 0 if T[i] <= Tprev[i] or re_[i] < 0 else (t - Tprev[i]) / (T[i] - Tprev[i])
            x = rx[i - 1] + u * (rx[i] - rx[i - 1]); y = ry[i - 1] + u * (ry[i] - ry[i - 1])
            xi, yi, h = int(round(x)), int(round(y)), 60
            ys0, ys1, xs0, xs1 = max(0, yi - h), min(MH, yi + h + 1), max(0, xi - h), min(MW, xi + h + 1)
            E[ys0:ys1, xs0:xs1] += dot[ys0 - (yi - h):ys1 - (yi - h), xs0 - (xi - h):xs1 - (xi - h)]
        if end:
            u = min(1.0, (tf - hold0 - ROUTE_S) / 1.5)
            E *= 1 + 0.35 * np.sin(np.pi * min(u * 1.2, 1.0))
        small = cv2.resize(E, (MW // 2, MH // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5) * 0.6 + cv2.GaussianBlur(small, (0, 0), 9) * 0.5, (MW, MH))
        img = 1 - np.exp(-(E + bloom) * 1.15)
        canvas = np.empty((VH, VW, 3), np.float32); canvas[:] = np.array([4, 7, 13], np.float32) / 255
        canvas[MAP_TOP:MAP_TOP + MH, mx0:mx0 + MW] = np.maximum(img, canvas[MAP_TOP:MAP_TOP + MH, mx0:mx0 + MW])
        # text
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        ms = t / ROUTE_S * total_ms
        d.text((VW // 2, 1365), f"{ms:.1f} ms", font=f_clock, fill=(250, 238, 212, 255), anchor="ma")
        d.text((VW // 2, 1500), f"slowed down {slow:,.0f}×  ·  {t / ROUTE_S * stats['walk_km']:,.0f} km driven",
               font=font("Inter-Medium.otf", 32), fill=(160, 178, 200, 255), anchor="ma")
        if end:
            u = min(1.0, (tf - hold0 - ROUTE_S - 0.6) / 0.5)
            if u > 0:
                for i, (txt, sz, col) in enumerate([(f"{total_ms:.0f} milliseconds.", 84, (255, 186, 80)),
                                                    ("Faster than a blink.", 60, (240, 244, 250)),
                                                    (f"{stats['street_km']:,.0f} km of Calgary streets", 40, (170, 186, 206))]):
                    d.text((VW // 2, 160 + [0, 104, 190][i]), txt,
                           font=font("Inter-Bold.otf" if sz > 50 else "Inter-Medium.otf", sz),
                           fill=col + (int(255 * u),), anchor="ma")
        d.text((VW // 2, 1556), "Public streets inside city limits (no alleys, parking lots or driveways)",
               font=font("Inter-Regular.otf", 22), fill=(110, 128, 150, 255), anchor="ma")
        d.text((VW // 2, 1586), "Road data \u00a9 OpenStreetMap contributors via Overture Maps",
               font=font("Inter-Regular.otf", 22), fill=(110, 128, 150, 255), anchor="ma")
        a = np.asarray(ov).astype(np.float32) / 255
        canvas = canvas * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
        if not end or (tf - hold0 - ROUTE_S) < 0.6:
            fade = 1.0 if not end else max(0.0, 1 - (tf - hold0 - ROUTE_S) / 0.6)
            canvas = canvas * (1 - hook_a[..., 3:] * fade) + hook_a[..., :3] * hook_a[..., 3:] * fade
        foot = "Streets inside city limits (excl. alleys, lots, driveways) · © OpenStreetMap contributors via Overture Maps"
        p.stdin.write((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    print("wrote", path)
    return path


if __name__ == "__main__":
    G = build()
    rn, re_, cum, stats = walk(G)
    json.dump({**stats, "by_class_km": G["by_class"]}, open(os.path.join(DATA, "calgary_stats.json"), "w"), indent=1)
    render(G, rn, re_, cum, stats)
