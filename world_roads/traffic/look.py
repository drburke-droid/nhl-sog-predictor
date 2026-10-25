"""Shared look for the traffic-equilibrium video (1920x1080, 30 fps), in the style of the light-speed film:
the city's real streets as a glowing cyan network on near-black, routes as bright lines with bloom, Inter type.

    MapView      km -> pixel projection, the street network raster for that view (cached), route polylines
    Glow         an additive emission buffer with two-scale bloom
    chart/pill   PIL helpers for axes, dashed lines, ETA pills, letter badges
    Section      chunked, resumable rendering of one section to out/<name>.mp4
"""
import os, sys, json, math, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                            # fonts, captions, translucent-text fix  # noqa: E402
sys.path.insert(0, HERE)
from cities import C as CITYCONF, CITY                        # noqa: E402

DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
MEDIA = os.path.join(ROOT, "media", "traffic")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 8, 15], np.float32)
CYAN = np.array([0.14, 0.66, 1.00], np.float32) * 255          # the light-speed film's "not yet reached" roads
GOLD = (255, 186, 80)
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
# routes: gold, rose, violet (CVD-checked on this background), always with a letter badge as well
ROUTE = [(255, 178, 74), (255, 79, 139), (157, 140, 255)]
LET = "ABC"
DASH = [None, (16, 9), (4, 7)]                                   # chart line styles: solid / dashed / dotted
CLASS_W = {"motorway": (1.0, 2), "trunk": (0.9, 2), "primary": (0.75, 2), "secondary": (0.55, 1), "tertiary": (0.42, 1),
           "unclassified": (0.25, 1), "residential": (0.17, 1), "living_street": (0.14, 1)}


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


def numbers():
    return json.load(open(os.path.join(MEDIA, f"video_numbers_{CITY}.json")))


def routes_json():
    return json.load(open(os.path.join(MEDIA, f"routes_{CITY}.json")))


def graph():
    return dict(np.load(os.path.join(DATA, CITY, "graph.npz")))


# ------------------------------------------------------------------ map
class MapView:
    """x_km, y_km (east, north of the city's projection centre) -> screen pixels: centre (ox, oy) shows km point
    (kx, ky) at s px/km"""

    def __init__(self, ox, oy, s, kx=0.0, ky=0.0, name="full"):
        self.ox, self.oy, self.s, self.kx, self.ky, self.name = ox, oy, s, kx, ky, name

    @classmethod
    def fit(cls, box, name, fill=0.92, which=(0, 1, 2)):
        """the view that fits the main routes `which` (and both trip ends) into the screen box (x0, y0, x1, y1)"""
        R = routes_json()
        P = np.vstack([np.asarray(R["routes"][R["main"][i]]["detail"]["xy"]) for i in which])
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        s = fill * min((box[2] - box[0]) / (x1 - x0), (box[3] - box[1]) / (y1 - y0))
        return cls((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, round(float(s), 3), (x0 + x1) / 2, (y0 + y1) / 2, name)

    def px(self, x, y):
        return self.ox + (np.asarray(x) - self.kx) * self.s, self.oy - (np.asarray(y) - self.ky) * self.s

    def pts(self, xy):
        xy = np.asarray(xy, np.float64)
        X, Y = self.px(xy[:, 0], xy[:, 1])
        return np.c_[X, Y]

    def base(self):
        """the street network for this view, float32 HxWx3 on the 0..255 scale (cached)"""
        os.makedirs(DATA, exist_ok=True)
        path = os.path.join(DATA, CITY, f"base_{self.name}_{self.ox:.0f}_{self.oy:.0f}_{self.s}_{self.kx:.3f}_{self.ky:.3f}.npy")
        if os.path.exists(path):
            return np.load(path)
        g = graph()
        S = 2
        I = np.zeros((H * S, W * S), np.float32)
        X, Y = self.px(g["nx"], g["ny"])
        P = np.stack([np.c_[X[g["a"]], Y[g["a"]]], np.c_[X[g["b"]], Y[g["b"]]]], 1) * S * 16
        P = np.rint(P).astype(np.int32)
        for c, (wt, th) in sorted(CLASS_W.items(), key=lambda kv: kv[1][0]):
            sel = np.nonzero(g["ecls"] == c)[0]
            if len(sel):
                J = np.zeros_like(I)
                cv2.polylines(J, list(P[sel]), False, 1.0, th, cv2.LINE_AA, shift=4)
                I = np.maximum(I, J * wt)
        I = cv2.resize(I, (W, H), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(I, (0, 0), 3.0) * 0.55 + cv2.GaussianBlur(I, (0, 0), 10.0) * 0.35
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        vign = np.clip(1 - 0.38 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.75)) ** 2), 0.4, 1)
        img = BG[None, None] * vign[..., None] + (I * 0.62 + glow)[..., None] * CYAN[None, None] * 0.55
        img = img.astype(np.float32)
        np.save(path, img)
        return img


class Glow:
    """additive emission with two-scale bloom (0..255 scale)"""

    def __init__(self):
        self.e = np.zeros((H, W, 3), np.float32)

    def clear(self):
        self.e[:] = 0

    def line(self, pts, col, width=3, a=1.0):
        if a <= 0 or len(pts) < 2:
            return
        p = np.rint(np.asarray(pts) * 16).astype(np.int32)
        c = tuple(float(v) * a for v in col)
        cv2.polylines(self.e, [p], False, c, int(width), cv2.LINE_AA, shift=4)

    def dot(self, x, y, col, r=3, a=1.0):
        if a <= 0:
            return
        cv2.circle(self.e, (int(round(x * 16)), int(round(y * 16))), int(r * 16), tuple(float(v) * a for v in col), -1, cv2.LINE_AA, shift=4)

    def dots(self, P, col, r=3, a=1.0):
        for x, y in P:
            self.dot(x, y, col, r, a)

    def compose(self, base, bloom=1.0, dim=1.0):
        """lines cover the map under them (so their hue survives), and their bloom is added on top"""
        e = self.e
        small = cv2.resize(e, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        bl = cv2.addWeighted(cv2.GaussianBlur(small, (0, 0), 1.25), 0.45, cv2.GaussianBlur(small, (0, 0), 4.5), 0.4, 0)
        bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)
        c = cv2.split(e)
        m = cv2.min(cv2.max(cv2.max(c[0], c[1]), c[2]), 200.0)
        k = cv2.addWeighted(m, -0.9 / 200.0 * dim, m, 0, dim)          # dim * (1 - 0.9 * cover)
        img = cv2.multiply(base, cv2.merge([k, k, k]))
        img = cv2.addWeighted(img, 1.0, e, 0.92, 0)
        img = cv2.scaleAdd(bl, float(bloom), img)
        return cv2.convertScaleAbs(img)


def sprite(col, size=81, core=1.4, glow=4.5, halo=16):
    """the light-speed film's point of light, tinted: white-hot core, coloured glow and halo (0..255 additive)"""
    k = np.arange(size, dtype=np.float32) - size // 2
    r2 = k[None] ** 2 + k[:, None] ** 2
    c = np.array(col, np.float32) / 255
    out = (np.exp(-r2 / (2 * core ** 2))[..., None] * np.array([1.0, 0.97, 0.92]) * 2.4
           + np.exp(-r2 / (2 * glow ** 2))[..., None] * c * 1.1
           + np.exp(-r2 / (2 * halo ** 2))[..., None] * c * 0.35)
    return (out * 255).astype(np.float32)


def blit_add(img_f, spr, x, y, a=1.0):
    """add a sprite (float) centred at x, y into a float image"""
    h, w = spr.shape[:2]
    x0, y0 = int(round(x)) - w // 2, int(round(y)) - h // 2
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    img_f[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0] * a


def polyline_cum(P):
    d = np.hypot(*np.diff(P, axis=0).T)
    return np.r_[0, np.cumsum(d)]


def partial(P, cum, L):
    """the polyline up to arc length L"""
    if L <= 0:
        return P[:1]
    if L >= cum[-1]:
        return P
    i = int(np.searchsorted(cum, L))
    f = (L - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    return np.vstack([P[:i], P[i - 1] + f * (P[i] - P[i - 1])])


def at(P, cum, L):
    L = min(max(L, 0.0), cum[-1])
    i = int(np.clip(np.searchsorted(cum, L), 1, len(cum) - 1))
    f = (L - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
    return P[i - 1] + f * (P[i] - P[i - 1])


def resample(P, n):
    cum = polyline_cum(P)
    s = np.linspace(0, cum[-1], n)
    return np.array([at(P, cum, v) for v in s])


# ------------------------------------------------------------------ 2-D drawing (PIL)
def text(d, xy, s, size, col=INK, a=1.0, weight="SemiBold", anchor="la"):
    if a > 0 and s:
        d.text(xy, s, font=ST.font(size, weight), fill=tuple(col) + (int(255 * min(a, 1)),), anchor=anchor)


def dashed(d, pts, col, width, dash=None, a=1.0):
    """polyline with an optional (on, off) dash pattern in pixels"""
    if a <= 0 or len(pts) < 2:
        return
    fill = tuple(col) + (int(255 * min(a, 1)),)
    pts = [(float(x), float(y)) for x, y in pts]
    if dash is None:
        d.line(pts, fill=fill, width=width, joint="curve")
        return
    on, off = dash
    pos, draw = 0.0, True
    seg = [pts[0]]
    left = on
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        u = 0.0
        while L - u > 1e-9:
            step = min(left, L - u)
            u += step
            p = (x0 + (x1 - x0) * u / L, y0 + (y1 - y0) * u / L)
            if draw:
                seg.append(p)
            left -= step
            if left <= 1e-9:
                if draw and len(seg) > 1:
                    d.line(seg, fill=fill, width=width, joint="curve")
                draw = not draw
                left = on if draw else off
                seg = [p]
    if draw and len(seg) > 1:
        d.line(seg, fill=fill, width=width, joint="curve")


def badge(d, cx, cy, i, r=17, a=1.0):
    if a <= 0:
        return
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(ROUTE[i]) + (int(255 * a),))
    text(d, (cx, cy + 1), LET[i], int(r * 1.15), (10, 12, 20), a, "Bold", "mm")


def pill(d, x, y, i, main, sub=None, a=1.0, size=30, anchor="mm", strong=True, col=None, letter=None):
    """a rounded ETA label: letter badge, big text, optional small text; (x, y) is its anchor point.
    col / letter override route i's colour and letter (for paths that are not one of the three routes)"""
    if a <= 0:
        return (0, 0, 0, 0)
    f1 = ST.font(size, "Bold" if strong else "SemiBold")
    f2 = ST.font(max(20, int(size * 0.62)), "Medium")
    w1 = d.textlength(main, font=f1)
    w2 = d.textlength(sub, font=f2) if sub else 0
    r = int(size * 0.62)
    pad = int(size * 0.45)
    w = 2 * r + pad + w1 + (pad + w2 if sub else 0) + pad
    h = int(size * 1.6)
    x0 = x - w / 2 if anchor[0] == "m" else (x if anchor[0] == "l" else x - w)
    y0 = y - h / 2
    c = tuple(col or ROUTE[i])
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], h // 2, fill=ST.PANEL + (int(220 * a),), outline=c + (int(200 * a),), width=2)
    if letter is None:
        badge(d, x0 + h / 2, y0 + h / 2, i, r=int(h * 0.36), a=a)
    else:
        rr = int(h * 0.36)
        d.ellipse([x0 + h / 2 - rr, y0 + h / 2 - rr, x0 + h / 2 + rr, y0 + h / 2 + rr], fill=c + (int(255 * a),))
        text(d, (x0 + h / 2, y0 + h / 2 + 1), letter, int(rr * 1.15), (10, 12, 20), a, "Bold", "mm")
    text(d, (x0 + h / 2 + r + pad, y0 + h / 2 + 1), main, size, INK, a, "Bold" if strong else "SemiBold", "lm")
    if sub:
        text(d, (x0 + h / 2 + r + 2 * pad + w1, y0 + h / 2 + 2), sub, max(20, int(size * 0.62)), INK2, a, "Medium", "lm")
    return (x0, y0, x0 + w, y0 + h)


class Axes:
    """a plot box on the frame: data -> pixels, gridlines, ticks, labels"""

    def __init__(self, box, xr, yr):
        self.x0, self.y0, self.x1, self.y1 = box
        self.xr, self.yr = xr, yr

    def X(self, v):
        return self.x0 + (np.asarray(v, float) - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.x1 - self.x0)

    def Y(self, v):
        v = np.clip(np.asarray(v, float), self.yr[0] - 0.02 * (self.yr[1] - self.yr[0]), self.yr[1] + 0.02 * (self.yr[1] - self.yr[0]))
        return self.y1 - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.y1 - self.y0)

    def draw(self, d, xticks, yticks, xfmt=str, yfmt=str, xlabel="", ylabel="", a=1.0, size=22):
        if a <= 0:
            return
        for v in yticks:
            y = float(self.Y(v))
            d.line([(self.x0, y), (self.x1, y)], fill=(40, 52, 72, int(255 * a)), width=1)
            text(d, (self.x0 - 12, y), yfmt(v), size, MUTED, a, "Medium", "rm")
        d.line([(self.x0, self.y1), (self.x1, self.y1)], fill=(90, 106, 130, int(255 * a)), width=2)
        for v in xticks:
            x = float(self.X(v))
            d.line([(x, self.y1), (x, self.y1 + 7)], fill=(90, 106, 130, int(255 * a)), width=2)
            text(d, (x, self.y1 + 14), xfmt(v), size, MUTED, a, "Medium", "ma")
        if xlabel:
            text(d, ((self.x0 + self.x1) / 2, self.y1 + 50), xlabel, size, INK2, a, "Medium", "ma")
        if ylabel:
            text(d, (self.x0, self.y0 - 22), ylabel, size, INK2, a, "Medium", "ls")

    def series(self, d, xs, ys, i=None, col=None, width=4, dash="auto", a=1.0, clip=False):
        """clip=True: stop the line where it leaves the plot (instead of running along the edge)"""
        c = ROUTE[i] if col is None else col
        dash = DASH[i] if (dash == "auto" and i is not None) else (None if dash == "auto" else dash)
        seg = []
        for x, y in zip(xs, ys):
            if not np.isfinite(y) or (clip and y > self.yr[1]):
                if clip and np.isfinite(y) and seg:
                    seg.append((float(self.X(x)), float(self.Y(self.yr[1]))))
                dashed(d, seg, c, width, dash, a)
                seg = []
                continue
            seg.append((float(self.X(x)), float(self.Y(y))))
        dashed(d, seg, c, width, dash, a)


def caption_box(d, lines, a, y=None, cx=None, anchor="bottom"):
    ST.caption(d, lines, a, y=y, cx=cx, anchor=anchor)


def to_rgba(frame):
    return Image.fromarray(frame).convert("RGBA")


def finish(im):
    return np.asarray(im.convert("RGB"))


def overlay_frame(frame, draw_fn):
    """run draw_fn(ImageDraw) on an RGBA copy of the frame and return the composited uint8 array"""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    draw_fn(d, im)
    return np.asarray(im)


# ------------------------------------------------------------------ rendering
class Section:
    name, dur = "section", 1.0

    def frame(self, t):
        raise NotImplementedError

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            Image.fromarray(self.frame(still)).save(os.path.join(OUT, f"{self.name}_{still:05.1f}.png"))
            return
        CHUNK = 150
        cdir = os.path.join(OUT, "chunks")
        os.makedirs(cdir, exist_ok=True)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        parts = []
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            part = os.path.join(cdir, f"{self.name}_{c0:05d}.mp4")
            parts.append(part)
            if os.path.exists(part + ".ok"):
                continue
            tmp = part + ".tmp.mp4"
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                                  "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
            for i in range(c0, c1):
                p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            p.stdin.close()
            p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q)
            os.remove(q + ".ok")
        print("wrote", path, flush=True)
