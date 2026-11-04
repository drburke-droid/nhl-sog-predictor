"""Vessel geometry: each population's vessel is the outline of its countries, made of glass.

For every vessel this builds (and caches in data/vessels_cache.pkl):
  - the outline in screen pixels (union of its countries; tiny islands as a magnified inset),
  - a glass sprite (RGBA, supersampled): rim, top-left highlight, faint tint,
  - the sand lattice: hex-packed grain slots inside the glass, ordered bottom-up (sand settles to the
    bottom of the shape), with the slot directly above each slot and each slot's near neighbours.
"""
import hashlib, json, math, os, pickle
import numpy as np
import cv2
from scipy.spatial import cKDTree

from proj import W, H, to_px

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SS = 2
D = 3.5            # grain spacing (px)
R_GRAIN = 1.70     # grain radius (px)
MARGIN = 1.0       # clearance between a grain's edge and the glass


def load_countries():
    path = os.path.join(DATA, "ne_10m_admin_0_countries.geojson")
    if not os.path.exists(path):
        import subprocess
        os.makedirs(DATA, exist_ok=True)
        url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
               "ne_10m_admin_0_countries.geojson")
        subprocess.run(["curl", "-sS", "-o", path, url], check=True)
    from shapely.geometry import shape
    gj = json.load(open(path))
    return {f["properties"]["ADM0_A3"]: shape(f["geometry"]) for f in gj["features"]}


def split_cut(g, cut=-170.0):
    """Polygons of g, with the part west of the map edge moved +360 so it projects on the right."""
    from shapely.geometry import box
    from shapely import affinity
    out = []
    for part, shift in ((g.intersection(box(-180, -90, cut, 90)), 360.0), (g.intersection(box(cut, -90, 180, 90)), 0.0)):
        if part.is_empty:
            continue
        part = affinity.translate(part, xoff=shift)
        for p in getattr(part, "geoms", [part]):
            if p.geom_type == "Polygon" and p.area > 0:
                out.append(p)
    return out


def vessel_polys(spec, countries):
    """Lon/lat polygons for one vessel spec."""
    from shapely.geometry import box
    lo0, lo1, la0, la1 = spec.get("bbox", (-400, 400, -90, 90))
    clip = spec.get("clip")
    polys = []
    for code in spec["codes"]:
        for p in split_cut(countries[code]):
            if clip and code in spec.get("clip_codes", spec["codes"]):   # (lon0, lat0, lon1, lat1)
                p = p.intersection(box(*clip))
                if p.is_empty:
                    continue
            for q in getattr(p, "geoms", [p]):
                if q.geom_type != "Polygon" or q.area <= 0:
                    continue
                rp = q.representative_point()
                lon = rp.x if rp.x <= 190 else rp.x - 360
                if not (lo0 <= lon <= lo1 and la0 <= rp.y <= la1):
                    continue
                if any(e[0] <= lon <= e[1] and e[2] <= rp.y <= e[3] for e in spec.get("exclude", [])):
                    continue
                polys.append(q)
    return polys


def project_polys(polys, spec):
    rings_px = []          # list of (exterior Nx2, [holes])
    for p in polys:
        def pr(ring):
            xy = np.array(ring.coords)
            x, y = to_px(np.minimum(xy[:, 0], 189.999), xy[:, 1])
            return np.stack([x, y], 1)
        rings_px.append((pr(p.exterior), [pr(r) for r in p.interiors]))
    inset = spec.get("inset")
    if inset:
        k, (clon, clat) = inset
        allp = np.vstack([r[0] for r in rings_px])
        c0 = (allp.min(0) + allp.max(0)) / 2
        c1 = np.array(to_px(clon, clat), float)
        rings_px = [((e - c0) * k + c1, [(h - c0) * k + c1 for h in hs]) for e, hs in rings_px]
    return rings_px


def raster(rings_px, ss):
    m = np.zeros((H * ss, W * ss), np.uint8)
    ext = [np.round(e * ss * 16).astype(np.int32) for e, _ in rings_px]
    cv2.fillPoly(m, ext, 255, lineType=cv2.LINE_AA, shift=4)
    holes = [np.round(h * ss * 16).astype(np.int32) for _, hs in rings_px for h in hs]
    if holes:
        cv2.fillPoly(m, holes, 0, lineType=cv2.LINE_AA, shift=4)
    return m


def glass_sprite(mask_ss):
    """RGBA float sprite (bbox-cropped) of a glass vessel from its supersampled mask."""
    ys, xs = np.nonzero(mask_ss)
    pad = 14 * SS
    y0, y1 = max(0, ys.min() - pad), min(mask_ss.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask_ss.shape[1], xs.max() + pad)
    m = mask_ss[y0:y1, x0:x1].astype(np.float32) / 255
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    er = cv2.erode(m, k(int(1.3 * SS)))
    rim = np.clip(m - er, 0, 1)
    halo = cv2.GaussianBlur(rim, (0, 0), 3.0 * SS) * (1 - m)
    sh = int(3 * SS)
    shifted = np.zeros_like(m); shifted[sh:, sh:] = m[:-sh, :-sh]
    spec = cv2.GaussianBlur(np.clip(m - shifted, 0, 1) * er, (0, 0), 1.0 * SS)
    shifted2 = np.zeros_like(m); shifted2[:-sh, :-sh] = m[sh:, sh:]
    shade = cv2.GaussianBlur(np.clip(m - shifted2, 0, 1) * er, (0, 0), 1.5 * SS)
    hh = m.shape[0]
    grad = np.linspace(1.0, 0.4, hh, dtype=np.float32)[:, None]
    layers = [  # (rgb, alpha) painted in order
        (np.array([150, 185, 235], np.float32), m * (0.05 + 0.05 * grad)),
        (np.array([10, 14, 24], np.float32), shade * 0.35),
        (np.array([170, 205, 255], np.float32), halo * 0.35),
        (np.array([235, 245, 255], np.float32), spec * 0.40),
        (np.array([205, 228, 255], np.float32), rim * 0.70),
    ]
    rgb = np.zeros(m.shape + (3,), np.float32)
    a = np.zeros(m.shape, np.float32)
    for c, al in layers:
        al = np.clip(al, 0, 1)
        rgb = rgb * (1 - al[..., None]) + c * al[..., None]     # premultiplied 'over'
        a = a + al * (1 - a)
    out = np.zeros(m.shape + (4,), np.float32)
    out[..., :3] = rgb / np.maximum(a, 1e-4)[..., None]
    out[..., 3] = a
    return out, (x0, y0)


def lattice(mask_ss, anchor_x):
    """Hex grain slots inside the mask (with clearance), bottom row first, each row from anchor_x out."""
    dist = cv2.distanceTransform((mask_ss > 127).astype(np.uint8), cv2.DIST_L2, 5) / SS
    ys, xs = np.nonzero(mask_ss > 127)
    xmin, xmax = xs.min() / SS, xs.max() / SS
    ymin, ymax = ys.min() / SS, ys.max() / SS
    dy = D * math.sqrt(3) / 2
    rows = []
    r = 0
    y = ymax - R_GRAIN - MARGIN
    while y > ymin:
        off = D / 2 if r % 2 else 0.0
        xr = np.arange(xmin + off, xmax + 1, D)
        yi = min(int(round(y * SS)), dist.shape[0] - 1)
        xi = np.clip(np.round(xr * SS).astype(int), 0, dist.shape[1] - 1)
        ok = dist[yi, xi] >= R_GRAIN + MARGIN
        xr = xr[ok]
        if len(xr):
            xr = xr[np.argsort(np.abs(xr - anchor_x) + 1e-3 * xr)]
            rows.append(np.stack([xr, np.full(len(xr), y)], 1))
        y -= dy
        r += 1
    pts = np.vstack(rows).astype(np.float32)
    tree = cKDTree(pts)
    n = len(pts)
    up = np.full(n, -1, np.int32)
    # the slot above: nearest slot one row up (within a grain's width)
    q = pts.copy(); q[:, 1] -= dy
    dd, ii = tree.query(q, k=1)
    sel = dd < D * 0.75
    up[sel] = ii[sel]
    up[up <= np.arange(n)] = -1          # must come later in the fill order
    nb = tree.query_ball_point(pts, D * 2.6)
    nb = [np.array([j for j in lst if j != i], np.int32) for i, lst in enumerate(nb)]
    return pts, up, nb


def build(specs):
    key = hashlib.md5((json.dumps(specs, sort_keys=True) + f"{D}{R_GRAIN}{MARGIN}{SS}v7-frame").encode()).hexdigest()
    cache = os.path.join(DATA, "vessels_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            c = pickle.load(f)
        if c.get("key") == key:
            return c["vessels"]
    countries = load_countries()
    out = {}
    for vid, spec in specs.items():
        polys = vessel_polys(spec, countries)
        if spec.get("buffer"):                 # round off island chains too thin to hold sand
            from shapely.ops import unary_union
            u = unary_union(polys).buffer(spec["buffer"], 24).buffer(-spec["buffer"] * 0.4, 24)
            polys = list(getattr(u, "geoms", [u]))
        rings = project_polys(polys, spec)
        m = raster(rings, SS)
        # anchor: deepest interior point of the biggest piece (where sand is poured in, and the label)
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5)
        ay, ax = np.unravel_index(np.argmax(dist), dist.shape)
        anchor = (ax / SS, ay / SS)
        if "anchor" in spec:
            anchor = tuple(float(v) for v in to_px(*spec["anchor"])) if not spec.get("inset") else anchor
        pts, up, nb = lattice(m, anchor[0])
        spr, off = glass_sprite(m)
        geo = spec.get("geo")
        extra = {}
        if spec.get("inset"):
            # the island at its true size (glass only), and the magnifier bubble around the inset
            true_spec = {k: v for k, v in spec.items() if k not in ("inset", "buffer")}
            extra["true_rings"] = project_polys(vessel_polys(true_spec, countries), true_spec)
            from shapely.geometry import box as sbox
            allp = np.vstack([e for e, _ in rings])
            frame = sbox(*allp.min(0), *allp.max(0)).buffer(9, 12)          # rounded-rectangle lens
            extra["frame"] = np.array(frame.exterior.coords)
            tp = np.vstack([e for e, _ in extra["true_rings"]])
            geo = None
            extra["true_center"] = tuple(float(v) for v in (tp.min(0) + tp.max(0)) / 2)
        out[vid] = dict(**extra,rings=rings, sprite=spr.astype(np.float16), sprite_off=off, slots=pts, up=up, nb=nb,
                        anchor=anchor, geo=tuple(float(v) for v in to_px(*geo)) if geo else None,
                        bbox=(float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())))
        print(f"  vessel {vid}: {len(pts)} grain slots", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(dict(key=key, vessels=out), f)
    return out
