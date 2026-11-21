"""Clay tablets with cuneiform numbers pressed into them. Each tablet is a height map: a pillow of clay with a
little texture; every wedge is a pit pressed into it (a triangular pyramid for the head, a V-groove for a
vertical wedge's tail). The height map is lit from the upper left, so the marks read as real impressions.

Number signs follow the Old Babylonian sexagesimal system: a vertical wedge for 1, a corner wedge (Winkelhaken)
for 10, tens to the left of ones in each place. The way the wedges are grouped into rows is a common pattern,
stylised; scribes' hands varied.
"""
import math, functools
import numpy as np
import cv2
import skia

import look as L

ALBEDO = np.array([196, 150, 108], np.float32) / 255.0
LIGHT = np.array([-0.50, -0.62, 0.60], np.float32)
LIGHT /= np.linalg.norm(LIGHT)

ONES = {1: [1], 2: [2], 3: [3], 4: [3, 1], 5: [3, 2], 6: [3, 3], 7: [4, 3], 8: [4, 4], 9: [3, 3, 3]}
TENS = {1: [1], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2]}


def _tri_pit(hm, verts, depth, k=1.0):
    """a triangular pyramid pit (deepest at the incentre), merged into the depth map hm. verts in px."""
    v = np.asarray(verts, np.float64)
    x0, y0 = np.floor(v.min(0)).astype(int) - 2
    x1, y1 = np.ceil(v.max(0)).astype(int) + 3
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, hm.shape[1]), min(y1, hm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    # signed distance to each edge, positive inside (vertices taken counter-clockwise or clockwise: fix sign)
    area = (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1]) - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
    sgn = 1.0 if area > 0 else -1.0
    dmin = np.full(xx.shape, np.inf)
    per = 0.0
    for i in range(3):
        a, b = v[i], v[(i + 1) % 3]
        e = b - a
        Le = math.hypot(*e)
        per += Le
        d = sgn * ((xx - a[0]) * e[1] - (yy - a[1]) * e[0]) / Le * -1
        dmin = np.minimum(dmin, d)
    inr = 2 * abs(area) / 2 / per                       # inradius = area / semi-perimeter
    pit = depth * np.clip(dmin / inr, 0, 1) ** k
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], pit)


def _groove(hm, p0, p1, w0, w1, d0, d1):
    """a V-groove from p0 to p1 (px), half-width w0 -> w1, depth d0 -> d1"""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    pad = max(w0, w1) + 3
    x0, y0 = np.floor(np.minimum(p0, p1) - pad).astype(int)
    x1, y1 = np.ceil(np.maximum(p0, p1) + pad).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, hm.shape[1]), min(y1, hm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    e = p1 - p0
    Le = math.hypot(*e)
    u = ((xx - p0[0]) * e[0] + (yy - p0[1]) * e[1]) / Le ** 2
    v = ((xx - p0[0]) * -e[1] + (yy - p0[1]) * e[0]) / Le
    inside = (u >= 0) & (u <= 1)
    hw = w0 + (w1 - w0) * np.clip(u, 0, 1)
    d = d0 + (d1 - d0) * np.clip(u, 0, 1)
    pit = np.where(inside, d * np.clip(1 - np.abs(v) / hw, 0, 1), 0)
    # round the far end
    tip = (u > 1) & (np.hypot(xx - p1[0], yy - p1[1]) < w1)
    pit = np.where(tip, d1 * np.clip(1 - np.hypot(xx - p1[0], yy - p1[1]) / max(w1, 1e-3), 0, 1), pit)
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], pit)


def wedge(hm, x, y, u, f=1.0, ang=0.0, res=2.0):
    """a vertical wedge: head (top edge) centred on (x, y) in units, length u units, pointing down (ang: radians)"""
    if f <= 0:
        return
    ca, sa = math.cos(ang), math.sin(ang)

    def P(px, py):
        return ((x + px * ca - py * sa) * res, (y + px * sa + py * ca) * res)

    hw, hl = 0.30 * u, 0.42 * u
    D = 0.16 * u * res * f
    _tri_pit(hm, [P(-hw, 0), P(hw, 0), P(0, hl)], D, k=0.9)
    _groove(hm, P(0, hl * 0.55), P(0, u), 0.075 * u * res, 0.035 * u * res, D * 0.55, D * 0.18)


def corner(hm, x, y, u, f=1.0, res=2.0):
    """a corner wedge (Winkelhaken, '<'): centred on (x, y), size u units"""
    if f <= 0:
        return
    D = 0.20 * u * res * f
    _tri_pit(hm, [((x - 0.50 * u) * res, y * res), ((x + 0.42 * u) * res, (y - 0.44 * u) * res),
                  ((x + 0.42 * u) * res, (y + 0.44 * u) * res)], D, k=0.8)


# ------------------------------------------------------------------ numbers as wedge lists
def digit_marks(n, u=40.0):
    """marks for one sexagesimal digit 1..59: [(kind, dx, dy, size)] relative to the digit's left edge and middle,
    plus the digit's width. kind: 'w' wedge (dx, dy = head centre), 'c' corner (centre)."""
    marks, x = [], 0.0
    tens, ones = divmod(n, 10)
    if tens:
        rows = TENS[tens]
        cs = u * (0.92 if len(rows) == 1 else 0.64)
        for r, cnt in enumerate(rows):
            yc = (r - (len(rows) - 1) / 2) * cs * 1.0
            for k in range(cnt):
                marks.append(("c", x + cs * 0.5 + k * cs * 0.78, yc, cs))
        x += cs * 0.5 + (max(rows) - 1) * cs * 0.78 + cs * 0.55 + u * 0.12
    if ones:
        rows = ONES[ones]
        nr = len(rows)
        ln = u if nr == 1 else u * 0.52 if nr == 2 else u * 0.36
        gap = 0.10 * u
        tot = nr * ln + (nr - 1) * gap
        for r, cnt in enumerate(rows):
            y0 = -tot / 2 + r * (ln + gap)
            for k in range(cnt):
                marks.append(("w", x + ln * 0.31 + k * max(ln * 0.62, u * 0.30), y0, ln))
        x += ln * 0.31 * 2 + (max(rows) - 1) * max(ln * 0.62, u * 0.30)
    return marks, x


def number_marks(digits, x, y, u=40.0, gap=None):
    """a sexagesimal number (list of digits, most significant first; 0 leaves an empty space) laid out from (x, y).
    Returns (marks, width, digit spans)."""
    gap = 0.45 * u if gap is None else gap
    out, spans, cx = [], [], x
    for d in digits:
        if d == 0:
            spans.append((cx, cx + 0.7 * u))
            cx += 0.7 * u + gap
            continue
        ms, w = digit_marks(d, u)
        for kind, dx, dy, s in ms:
            out.append((kind, cx + dx, y + dy, s))
        spans.append((cx, cx + w))
        cx += w + gap
    return out, cx - gap - x, spans


# ------------------------------------------------------------------ the tablet
class Tablet:
    def __init__(self, w, h, shape="rect", seed=3, res=None, edge=26.0, colour=None):
        self.w, self.h, self.shape = w, h, shape
        self.res = res or L.S
        self.albedo = ALBEDO if colour is None else np.array(colour, np.float32) / 255.0
        r = self.res
        pw, ph = int(round(w * r)), int(round(h * r))
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32) / r
        if shape == "round":
            d = (min(w, h) / 2 - np.hypot(xx - w / 2, yy - h / 2) * np.where(True, 1, 1))
        else:
            rr = 30.0
            qx = np.abs(xx - w / 2) - (w / 2 - rr)
            qy = np.abs(yy - h / 2) - (h / 2 - rr)
            d = rr - (np.hypot(np.clip(qx, 0, None), np.clip(qy, 0, None)) + np.minimum(np.maximum(qx, qy), 0))
        self.mask = np.clip(d * r + 0.5, 0, 1).astype(np.float32)
        e = np.clip(d / edge, 0, 1)
        hm = edge * 0.55 * np.sqrt(1 - (1 - e) ** 2) * r
        rng = np.random.default_rng(seed)
        for sig, amp in ((60, 2.2), (18, 0.9), (4, 0.30), (1.2, 0.10)):
            n = rng.normal(0, 1, (ph, pw)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sig * r / 2)
            hm += n / (n.std() + 1e-9) * amp * r / 2
        self.base = hm.astype(np.float32)
        tex = rng.normal(0, 1, (ph, pw)).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.0 * r) * 0.35 + cv2.GaussianBlur(tex, (0, 0), 14 * r) * 2.0
        self.tex = (tex / (tex.std() + 1e-9) * 0.05).astype(np.float32)
        self.extra = []                                  # other carvings: (callable(hm, res))
        self._cache = {}

    def heights(self, marks):
        dep = np.zeros_like(self.base)                   # depth pressed in: each mark merges by max (a union of pits)
        for f in self.extra:
            f(dep, self.res)
        for m in marks:
            kind, x, y, s = m[:4]
            f = m[4] if len(m) > 4 else 1.0
            if kind == "w":
                wedge(dep, x, y, s, f, res=self.res)
            elif kind == "c":
                corner(dep, x, y, s, f, res=self.res)
        # clay pushed aside rises in a faint rim around each mark
        rim = cv2.GaussianBlur(dep, (0, 0), 2.0 * self.res) - dep
        return self.base - dep + 0.25 * np.clip(rim, 0, None)

    def image(self, marks, light=1.0):
        key = (tuple((m[0], round(m[1], 2), round(m[2], 2), round(m[3], 2), round(m[4] if len(m) > 4 else 1.0, 3)) for m in marks),
               round(light, 3))
        if key in self._cache:
            return self._cache[key]
        hm = self.heights(marks)
        gy, gx = np.gradient(hm)
        nz = np.ones_like(hm)
        nrm = np.sqrt(gx * gx + gy * gy + nz)
        shade = np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) / nrm, 0, 1) / LIGHT[2]
        cav = np.clip((self.base - hm) / (6 * self.res), 0, 1)          # darker deep in the marks
        lum = (0.30 + 0.70 * shade) * (1 - 0.30 * cav) * (1 + self.tex) * light
        # a warm lamp from the upper left, falling off across the tablet
        ph, pw = hm.shape
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        lamp = 1.08 - 0.22 * np.clip((xx / pw + yy / ph) / 2, 0, 1)
        rgb = np.clip(self.albedo[None, None, :] * (lum * lamp)[..., None], 0, 1)
        a = self.mask
        rgba = np.empty((ph, pw, 4), np.uint8)
        rgba[..., :3] = (rgb * a[..., None] * 255 + 0.5).astype(np.uint8)
        rgba[..., 3] = (a * 255 + 0.5).astype(np.uint8)
        img = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[key] = img
        return img

    def draw(self, c, x, y, marks, a=1.0, light=1.0, scale=1.0):
        """draw the tablet with its top-left at (x, y) in canvas units"""
        if a <= 0:
            return
        img = self.image(marks, light)
        c.save()
        c.translate(x, y)
        c.scale(scale, scale)
        # a soft shadow on the night
        sh = skia.Rect.MakeXYWH(14, 22, self.w, self.h)
        r = min(self.w, self.h) / 2 if self.shape == "round" else 30
        c.drawRoundRect(sh, r, r, L.paint((0, 0, 0), 0.55 * a, blur=22))
        c.scale(1 / self.res, 1 / self.res)
        p = skia.Paint(AntiAlias=True)
        p.setAlphaf(float(L.clamp(a)))
        c.drawImage(img, 0, 0, skia.SamplingOptions(skia.FilterMode.kLinear), p)
        c.restore()


def scored_line(p0, p1, width=1.6, depth=1.4):
    """a line scratched into the clay (units), for the square and diagonals on YBC 7289"""
    def f(hm, res):
        _groove(hm, (p0[0] * res, p0[1] * res), (p1[0] * res, p1[1] * res), width * res, width * res, depth * res, depth * res)
    return f
