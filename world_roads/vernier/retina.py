"""The retina, seen down a microscope: foveal cones packed in a honeycomb, and the image of the test line
lying across them.

Units on the retina are cone spacings (about 0.5 arcminute, 2.4 micrometres, at the very centre of the
fovea). The line's image is the printed line (its width in arcminutes for a phone at reading distance)
blurred by the eye's line spread function (Westheimer 1986).
"""
import math, functools
import numpy as np
import cv2
from look import W, H, S

CONE_ARCMIN = 0.5                 # centre-to-centre spacing of foveal cones
UM_PER_ARCMIN = 4.85              # on the retina (about 291 micrometres per degree)
PX_ARCSEC = 41.0                  # one 1080p pixel of this video, full screen on a phone at 30 cm
LINE_PX = 2.0                     # the test line's width, 1080p pixels


def lsf(x):
    """Westheimer's line spread function of the eye, x in arcmin (3 mm pupil)"""
    return 0.47 * np.exp(-3.3 * x * x) + 0.53 * np.exp(-0.93 * np.abs(x))


@functools.lru_cache(None)
def profile():
    """the line's image across its length: light (0..1) against distance from its middle, in cone spacings"""
    dx = 0.005
    x = np.arange(-30, 30 + dx / 2, dx)                    # arcmin
    half = LINE_PX * PX_ARCSEC / 60 / 2
    dark = (np.abs(x) <= half).astype(np.float64)
    k = lsf(x)
    k /= k.sum()
    blurred = np.convolve(dark, k, mode="same")
    light = 1 - 0.94 * blurred                             # a black line printed on white
    return x / CONE_ARCMIN, light


def light_at(d):
    """light at distance d (cone spacings) from the middle of the line's image"""
    xs, ls = profile()
    return np.interp(d, xs, ls)


def step_cones(step_px):
    return step_px * PX_ARCSEC / 60 / CONE_ARCMIN


@functools.lru_cache(None)
def cells():
    """each cone's outline: its Voronoi cell (cone spacings), shrunk toward the centre to leave a membrane"""
    from scipy.spatial import Voronoi
    p, _ = lattice()
    vor = Voronoi(p)
    polys = []
    for i in range(len(p)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            polys.append(None)
            continue
        v = vor.vertices[reg]
        polys.append(p[i] + (v - p[i]) * 0.91)
    return polys


@functools.lru_cache(None)
def lattice(n=60, seed=5, jitter=0.045):
    """cone centres (cone spacings) on a slightly irregular hexagonal lattice, rows along x"""
    rng = np.random.default_rng(seed)
    h = math.sqrt(3) / 2
    pts = []
    for r in range(-n, n + 1):
        for c in range(-n - 1, n + 2):
            pts.append((c + (r % 2) * 0.5, r * h))
    p = np.array(pts) + rng.normal(0, jitter, (len(pts), 2))
    tone = rng.normal(0, 1, len(p))
    return p, tone


class Scope:
    """a view of the mosaic: centre (u, v) in cone spacings, k = screen px per cone spacing"""

    def __init__(self, u=0.0, v=0.0, k=110.0):
        self.u, self.v, self.k = u, v, k

    def X(self, u):
        return (np.asarray(u) - self.u) * self.k + W / 2

    def Y(self, v):
        return (np.asarray(v) - self.v) * self.k + H / 2


def line_y(u, y0, step, xm=0.0):
    """the line's middle across the retina: y0 left of xm, y0 + step to the right"""
    return np.where(np.asarray(u) < xm, y0, y0 + step)


FIELD_CX, FIELD_CY, FIELD_R = 0.5, 0.455, 0.425          # the microscope's field: centre and radius (fractions of H, W)
Q = 2                                                      # the micrograph is worked at half resolution


HALF = LINE_PX * PX_ARCSEC / 60 / CONE_ARCMIN / 2         # half the printed line's width on the retina (cone spacings)
ROW = math.sqrt(3) / 2                                     # spacing between rows of cones


def sharp_at(d, aa=0.0):
    """the printed line's image with no blur at all: light (0..1) at distance d from its middle"""
    inside = np.clip((HALF - np.abs(d)) / aa + 0.5, 0, 1) if aa > 0 else (np.abs(d) <= HALF).astype(np.float64)
    return 1 - 0.94 * inside


def mosaic(scope, y0=0.0, step=0.0, lit=1.0, band=0.0, xm=0.0, field=True, glow=None, tint=None, centre=None, radius=None,
           sharp=0.0, binary=0.0):
    """the micrograph, uint8 RGB (H, W, 3).
    lit: 0 = cones shown plain; 1 = each cone shaded by the light its centre gets from the line's image.
    band: opacity of the line's image itself, drawn continuous over the cones.
    sharp: 0 = the line's image as the eye really forms it (blurred); 1 = with no blur at all.
    binary: 0 = each cone reports how much light it gets; 1 = "pixel logic": dark if the line covers its row's
    centre line, light if not (a cone's own centre sits within 0.05 of its row's, too little to see).
    glow: optional per-cone highlight (array over the lattice, 0..1) in red pencil colour.
    centre: where on screen (4K px) the scope's centre sits (default: the middle of the field);
    radius: the field's radius (4K px).
    """
    p, tone = lattice()
    w, h = W // Q, H // Q
    k = scope.k / Q
    ccx, ccy = centre if centre is not None else (W * FIELD_CX, H * FIELD_CY)
    ccx, ccy = ccx / Q, ccy / Q
    idx, label, cover, inner = layout(scope.u, scope.v, k, ccx, ccy)
    gap = np.array([44, 34, 38], np.float32)
    d = p[idx, 1] - line_y(p[idx, 0], y0, step, xm)
    L = light_at(d)
    if binary > 0:
        v_row = np.round(p[idx, 1] / ROW) * ROW
        L = L * (1 - binary) + sharp_at(v_row - line_y(p[idx, 0], y0, step, xm)) * binary
    plain = np.array([190, 160, 160], np.float32)
    bright = np.array([246, 218, 186], np.float32)
    dim = np.array([66, 50, 52], np.float32)
    cols = np.repeat(plain[None, :], len(idx), 0) * (1 + 0.05 * tone[idx])[:, None]
    if lit > 0:
        target = (dim[None, :] + (bright - dim)[None, :] * (L ** 1.8)[:, None]) * (1 + 0.03 * tone[idx])[:, None]
        cols = cols * (1 - lit) + target * lit
    if glow is not None:
        g = glow[idx][:, None]
        cols = cols * (1 - g) + np.array([212, 88, 66], np.float32)[None, :] * g
    if tint is not None:
        cols = tint(idx, cols)
    cols = np.vstack([cols * 0.82, gap[None, :]]).astype(np.float32)     # the last entry: no cone
    img = gap + (cols[label] - gap) * cover[..., None]
    # each cone brighter in its middle (the inner segment catching the light), softened by the optics
    img *= (1 + 0.22 * inner)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), max(0.6, k * 0.025))
    if band > 0:
        v = (np.arange(h, dtype=np.float32) - ccy) / k + scope.v
        xs = (np.arange(w, dtype=np.float32) - ccx) / k + scope.u
        prof = lambda dd: (light_at(dd) * (1 - sharp) + sharp_at(dd, 1.0 / k) * sharp).astype(np.float32)
        left = prof(v - y0)
        right = prof(v - y0 - step)
        Lm = np.where(xs[None, :] < xm, left[:, None], right[:, None])
        img *= (1 - band * (1 - Lm) * 0.9)[..., None]
    img += photo_grain()
    if field:
        img = microscope(img, (ccx, ccy, (radius or H * FIELD_R) / Q), Q)
    # finished at half resolution (large float arrays are slow to allocate), then scaled up
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)


@functools.lru_cache(4)
def layout(u, v, k, ccx, ccy):
    """the static part of a view: the cones in it, which cone covers each pixel, how much of each pixel the
    cones cover (anti-aliased, leaving the membranes), and the inner-segment highlight"""
    p, _ = lattice()
    w, h = W // Q, H // Q
    X = (p[:, 0] - u) * k + ccx
    Y = (p[:, 1] - v) * k + ccy
    m = (X > -k) & (X < w + k) & (Y > -k) & (Y < h + k)
    idx = np.nonzero(m)[0]
    polys = cells()
    label = np.full((h, w), len(idx), np.int32)
    cover = np.zeros((h, w), np.float32)
    inner = np.zeros((h, w), np.float32)
    for j, i in enumerate(idx):
        if polys[i] is None:
            continue
        q = polys[i]
        poly = np.round(np.c_[(q[:, 0] - u) * k + ccx, (q[:, 1] - v) * k + ccy] * 16).astype(np.int32)
        cv2.fillPoly(label, [poly], j, cv2.LINE_8, shift=4)
        cv2.fillPoly(cover, [poly], 1.0, cv2.LINE_AA, shift=4)
        cv2.circle(inner, (int(round(X[i] * 16)), int(round(Y[i] * 16))), int(round(0.29 * k * 16)), 1.0, -1, cv2.LINE_AA, shift=4)
    # pixels on a cone's anti-aliased rim but outside its hard fill take the nearest cone's colour
    rim = (label == len(idx)) & (cover > 0)
    if rim.any():
        lab = np.where(label == len(idx), 0, 255).astype(np.uint8)
        _, near = cv2.distanceTransformWithLabels(lab, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(label != len(idx))
        lut = np.zeros(near.max() + 1, np.int32)
        lut[near[ys, xs]] = label[ys, xs]
        label = np.where(rim, lut[near], label)
    inner = cv2.GaussianBlur(inner, (0, 0), max(0.6, k * 0.10))
    return idx, label, cover, inner


def screen_y(scope, v, cy=None):
    """screen y (4K px) of retina coordinate v"""
    return (np.asarray(v) - scope.v) * scope.k + (H * FIELD_CY if cy is None else cy)


def screen_x(scope, u, cx=None):
    return (np.asarray(u) - scope.u) * scope.k + (W * FIELD_CX if cx is None else cx)


@functools.lru_cache(None)
def photo_grain():
    """film grain for the half-resolution micrograph"""
    rng = np.random.default_rng(21)
    n = cv2.GaussianBlur(rng.normal(0, 1, (H // Q, W // Q)).astype(np.float32), (0, 0), 0.8)
    return (n / n.std() * 3.0)[..., None]


@functools.lru_cache(8)
def field_mask(cx=W * FIELD_CX, cy=H * FIELD_CY, R=H * FIELD_R, q=1):
    """the field's light (with its vignette) and its hard-ish edge, on a grid of (H/q, W/q); cx, cy, R in that grid's px"""
    yy, xx = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy) / R
    inside = np.clip((1 - rr) * R * q / 5 + 0.5, 0, 1)
    vig = 1 - 0.38 * np.clip(rr, 0, 1) ** 4
    return (inside * vig).astype(np.float32), inside.astype(np.float32)


OUTSIDE = np.array([19, 15, 14], np.float32)


def microscope(img, field=(W * FIELD_CX, H * FIELD_CY, H * FIELD_R), q=1):
    """the round field of view of a microscope, dark outside"""
    m, inside = field_mask(*[float(v) for v in field], q)
    return img * m[..., None] + OUTSIDE * (1 - inside[..., None])
