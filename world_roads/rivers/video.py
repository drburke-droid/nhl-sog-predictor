"""Do rivers wind by pi? The video (1920x1080, 30 fps), caption-only.

    python video.py NAME [still times...]
    NAME: open, einstein, stolum, test, verdict, near, common, end
"""
import os, sys, json, math, time, subprocess, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                             # fonts, captions, translucent-text fix

DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
INK, INK2, MUTED, GOLD = ST.INK, ST.INK2, ST.MUTED, ST.GOLD
CYAN = (90, 200, 255)
CHORD = (255, 236, 200)                                         # the straight line
PI = math.pi
BG = (4, 7, 14)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def text(d, xy, s, size, col, a=1.0, weight="SemiBold", anchor="la"):
    if a > 0:
        d.text(xy, s, font=ST.font(size, weight), fill=tuple(col) + (int(255 * a),), anchor=anchor)


def km(x):
    return f"{x:,.0f} km"


@functools.lru_cache(None)
def rivers():
    meta = json.load(open(os.path.join(DATA, "vid.json")))
    z = np.load(os.path.join(DATA, "vid.npz"))
    return meta, {k: z[k] for k in z.files}


@functools.lru_cache(None)
def analysis():
    return json.load(open(os.path.join(DATA, "analysis.json")))


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def slerp(a, b, u):
    """great-circle interpolation between (lon, lat) a and b"""
    pa, pb = unit(*a), unit(*b)
    th = math.acos(min(1.0, max(-1.0, float(pa @ pb))))
    if th < 1e-9:
        return a
    p = (math.sin((1 - u) * th) * pa + math.sin(u * th) * pb) / math.sin(th)
    return math.degrees(math.atan2(p[1], p[0])), math.degrees(math.asin(max(-1, min(1, p[2]))))


class Section:
    name, dur = "section", 1.0

    def settle(self):
        """drop transient glow left by a jump in time (stills only)"""

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            if still >= 1 / FPS:
                self.frame(still - 1 / FPS)
                self.settle()
            Image.fromarray(self.frame(still)).save(os.path.join(OUT, f"{self.name}_{still:.1f}.png"))
            return
        # in chunks of CHUNK frames, so an interrupted render resumes; each resumed chunk first replays WARM frames
        # unrecorded, so anything that carries over from frame to frame does
        CHUNK, WARM = 150, 30
        cdir = os.path.join(OUT, "chunks")
        os.makedirs(cdir, exist_ok=True)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        parts = []
        prev_done = True
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            part = os.path.join(cdir, f"{self.name}_{c0:05d}.mp4")
            parts.append(part)
            if os.path.exists(part) and os.path.exists(part + ".ok"):
                prev_done = False
                continue
            if not prev_done:
                for i in range(max(0, c0 - WARM), c0):
                    self.frame(i / FPS)
            tmp = part + ".tmp.mp4"
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                                  "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
            for i in range(c0, c1):
                p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            p.stdin.close(); p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            prev_done = True
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q); os.remove(q + ".ok")
        print("wrote", path, flush=True)


# ================================================================== drawing rivers on the night globe
class Glow:
    """additive emission in two colours (rivers, straight lines), bloomed and tonemapped over a base image"""

    def __init__(self):
        self.A = np.zeros((H, W), np.float32)       # rivers (cyan)
        self.B = np.zeros((H, W), np.float32)       # straight lines (warm white)
        self.C = np.zeros((H, W), np.float32)       # highlights (gold)

    def poly(self, buf, pts, val, th=1, closed=False):
        if len(pts) < 2:
            return
        cv2.polylines(buf, [np.round(np.asarray(pts) * 16).astype(np.int32)], closed, float(val), th, cv2.LINE_AA, shift=4)

    def compose(self, img, exposure=1.2, bloom=1.0):
        E = (self.A[..., None] * (np.array(CYAN, np.float32) / 255) + self.B[..., None] * (np.array(CHORD, np.float32) / 255)
             + self.C[..., None] * (np.array(GOLD, np.float32) / 255))
        small = cv2.resize(E, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        bl = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5) * 0.7 + cv2.GaussianBlur(small, (0, 0), 9) * 0.5, (W, H))
        out = 1 - np.exp(-(img + E + bl * bloom) * exposure)
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


@functools.lru_cache(None)
def globe():
    import globe16 as G
    return G.Globe16()


class GlobeView:
    """camera on the globe: centre (lon, lat), radius in pixels, screen centre"""

    def __init__(self, lon, lat, rpx, cx=W / 2, cy=H / 2, g=None):
        self.lon, self.lat, self.rpx, self.cx, self.cy, self.g = lon, lat, rpx, cx, cy, g
        c = unit(lon, lat)
        e = np.array([-math.sin(math.radians(lon)), math.cos(math.radians(lon)), 0.0])
        self.c, self.e, self.n = c, e, np.cross(c, e)

    def base(self):
        return (self.g or globe()).render(self.lat, self.lon, self.rpx, self.cx, self.cy)

    def project(self, lonlat):
        P = unit(lonlat[:, 0], lonlat[:, 1])
        return np.c_[self.cx + (P @ self.e) * self.rpx, self.cy - (P @ self.n) * self.rpx], P @ self.c


def visible_runs(xy, z):
    """split a projected polyline into its runs on the visible hemisphere"""
    vis = z > 0.02
    if vis.all():
        return [xy]
    out, start = [], None
    for i, v in enumerate(vis):
        if v and start is None:
            start = i
        if (not v or i == len(vis) - 1) and start is not None:
            end = i + 1 if v else i
            if end - start >= 2:
                out.append(xy[start:end])
            start = None
    return out


def partial(xy, f):
    """the first fraction f of a polyline, by length"""
    if f >= 1:
        return xy
    if f <= 0 or len(xy) < 2:
        return xy[:1]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    L = cum[-1] * f
    j = int(np.searchsorted(cum, L) - 1)
    j = max(0, min(j, len(seg) - 1))
    u = (L - cum[j]) / max(seg[j], 1e-9)
    return np.r_[xy[:j + 1], [xy[j] + u * (xy[j + 1] - xy[j])]]


def river_xy(k, full=False):
    meta, Z = rivers()
    if full:
        return Z["f_xy"][Z["f_off"][k]:Z["f_off"][k + 1]].astype(np.float64)
    return Z["g_xy"][Z["g_off"][k]:Z["g_off"][k + 1]].astype(np.float64)


def fit_view(k, frac=0.62, cx=W / 2 + 180, cy=H / 2, rmax=5200):
    """a camera framing river k (its stem and straight line) in `frac` of the screen height"""
    meta, Z = rivers()
    pts = np.r_[river_xy(k), Z["c_xy"][k]]
    P = unit(pts[:, 0], pts[:, 1])
    m = P.mean(0)
    m /= np.linalg.norm(m)
    lon, lat = math.degrees(math.atan2(m[1], m[0])), math.degrees(math.asin(m[2]))
    v = GlobeView(lon, lat, 1.0)
    xy, _ = v.project(pts)
    ext = max(np.ptp(xy[:, 0]) / 1.5, np.ptp(xy[:, 1]), 1e-3)
    rpx = float(np.clip(frac * H / ext, 420, rmax))
    return lon, lat, rpx


def gauge(d, ratio, a, x0=1200, y0=962, w=600, mean=None, label="LENGTH ÷ STRAIGHT LINE, ALL RIVERS SO FAR"):
    """a number line from 1 to 4 with pi marked, a marker at the running ratio and its reading"""
    if a <= 0:
        return
    lo, hi = 1.0, 4.0
    X = lambda v: x0 + (v - lo) / (hi - lo) * w
    al = int(255 * a)
    d.rounded_rectangle([x0 - 34, y0 - 150, x0 + w + 34, y0 + 66], 18, fill=ST.PANEL + (int(205 * a),))
    text(d, (x0, y0 - 130), label, 19, MUTED, a)
    if ratio is not None:
        text(d, (x0, y0 - 104), f"{ratio:.2f}", 60, CYAN, a, "Bold")
    if mean is not None:
        text(d, (x0 + w, y0 - 92), "AVERAGE OF THE RATIOS", 17, MUTED, a, anchor="ra")
        text(d, (x0 + w, y0 - 70), f"{mean:.2f}", 30, INK2, a, "Bold", anchor="ra")
    d.line([(X(lo), y0), (X(hi), y0)], fill=INK2 + (al,), width=3)
    for v in np.arange(1.0, 4.01, 0.5):
        d.line([(X(v), y0 - 8), (X(v), y0 + 8)], fill=INK2 + (al,), width=2)
        if v == int(v):
            text(d, (X(v), y0 + 32), f"{v:.0f}", 22, INK2, a, anchor="mm")
    d.line([(X(PI), y0 - 22), (X(PI), y0 + 16)], fill=GOLD + (al,), width=4)
    text(d, (X(PI), y0 + 36), "π", 30, GOLD, a, "Bold", anchor="mm")
    if ratio is not None:
        xr = X(min(max(ratio, lo), hi))
        d.ellipse([xr - 12, y0 - 12, xr + 12, y0 + 12], fill=CYAN + (al,))


def river_card(d, m, a, x=80, y=150, n=None, prog=1.0):
    if a <= 0:
        return
    d.rounded_rectangle([x - 34, y - 36, x + 470, y + 430], 20, fill=ST.PANEL + (int(205 * a),))
    if n is not None:
        text(d, (x, y - 10), f"RIVER {n:,} OF 1,000", 20, MUTED, a)
    text(d, (x, y + 22), m["display"], 54 if len(m["display"]) < 15 else 42, INK, a, "Bold")
    text(d, (x, y + 112), "ALONG THE RIVER", 20, MUTED, a)
    text(d, (x, y + 138), km(m["L"] * min(1.0, prog * 2)), 44, CYAN, a, "Bold")
    if prog > 0.5:
        b = a * smooth((prog - 0.5) / 0.15)
        text(d, (x, y + 210), "STRAIGHT LINE, SOURCE TO MOUTH", 20, MUTED, b)
        text(d, (x, y + 236), km(m["D"]), 44, CHORD, b, "Bold")
    if prog > 0.8:
        b = a * smooth((prog - 0.8) / 0.12)
        text(d, (x, y + 308), "RATIO", 20, MUTED, b)
        text(d, (x, y + 330), f"{m['S']:.2f}", 64, GOLD if abs(m["S"] - PI) < 0.2 else INK, b, "Bold")


# ================================================================== the test: 1,000 rivers on the globe
class Test(Section):
    """the longest rivers one by one, then the rest of the 1,000 with the globe turning"""
    name = "test"
    FEATURED = [(0, 6.5), (1, 6.0), (2, 5.0), (3, 4.5), (4, 4.5), (5, 4.0), (6, 4.0), (7, 4.0), (9, 4.0), (16, 6.5)]
    DRAW = 2.6                     # seconds to draw a featured river and its straight line; the card then holds
    MOVE = 1.1                     # camera move into each featured river
    REST = 30.0                    # the other 990
    HOLD = 9.5

    def __init__(self):
        meta, Z = rivers()
        self.meta = meta
        t, self.slots = 3.0, []
        for k, d in self.FEATURED:
            self.slots.append((k, t, t + d))
            t += d
        self.t_rest = t
        self.t_hold = t + self.REST
        self.dur = self.t_hold + self.HOLD
        self.views = {k: fit_view(k) for k, _ in self.FEATURED}
        self.order = [k for k, _ in self.FEATURED] + [k for k in range(len(meta)) if k not in dict(self.FEATURED)]
        L = np.array([m["L"] for m in meta]); D = np.array([m["D"] for m in meta])
        self.cumL, self.cumD = np.cumsum(L[self.order]), np.cumsum(D[self.order])
        S = np.array([m["S"] for m in meta])
        self.cumS = np.cumsum(S[self.order])
        self.over = (88.0, 22.0, 500.0)        # the overview camera: lon, lat, rpx at the start of the rest

    def added(self, t):
        """how many rivers are in (in self.order), and the fraction drawn of the one being added"""
        for j, (k, a, b) in enumerate(self.slots):
            if t < b:
                return j, max(0.0, (t - a - self.MOVE * 0.6) / min(self.DRAW, b - a - self.MOVE * 0.6 - 0.6))
        if t < self.t_hold:
            u = (t - self.t_rest) / self.REST
            n = len(self.slots) + int((len(self.order) - len(self.slots)) * u ** 1.6)
            return n, 1.0
        return len(self.order), 1.0

    def camera(self, t):
        if t < self.t_rest:
            j = next(i for i, (k, a, b) in enumerate(self.slots) if t < b) if t < self.slots[-1][2] else len(self.slots) - 1
            k, a, b = self.slots[j]
            prev = self.views[self.slots[j - 1][0]] if j else (self.views[k][0] - 25, self.views[k][1], 420.0)
            u = smooth((t - a) / self.MOVE)
            lon, lat = slerp(prev[:2], self.views[k][:2], u)
            rpx = math.exp(math.log(prev[2]) + (math.log(self.views[k][2]) - math.log(prev[2])) * u)
            # a gentle drift while the river draws
            lon += 0.6 * (t - a)
            return lon, lat, rpx
        last = self.views[self.slots[-1][0]]
        u = smooth((t - self.t_rest) / 3.0)
        lon0, lat0 = slerp(last[:2], self.over[:2], u)
        rpx = math.exp(math.log(last[2]) + (math.log(self.over[2]) - math.log(last[2])) * u)
        spin = -408.0 * (t - self.t_rest) / (self.REST + self.HOLD)        # westward round the world, ending on Africa
        return lon0 + spin * smooth((t - self.t_rest) / 1.5), lat0, rpx

    def frame(self, t):
        lon, lat, rpx = self.camera(t)
        v = GlobeView(lon, lat, rpx, cx=W / 2 + 180 - (60 if t >= self.t_rest else 0) * smooth((t - self.t_rest) / 3.0))
        img = v.base()
        gl = Glow()
        n, f = self.added(t)
        th = 2
        for i in range(min(n + 1, len(self.order))):
            k = self.order[i]
            xy, z = v.project(river_xy(k))
            frac = f if i == n else 1.0
            if i == n and t >= self.t_rest:
                continue
            if frac <= 0:
                continue
            # featured: draw progressively; the river being drawn is brightest
            cur = i == n or (i < len(self.slots) and t < self.slots[i][2])
            pr = min(1.0, frac * 2)
            pts = partial(xy, pr)
            zz = z[:len(pts)]
            if len(zz) < len(pts):
                zz = np.r_[zz, z[len(pts) - 1:len(pts)]]
            age = 0.0
            if i >= len(self.slots):
                t_in = self.t_rest + self.REST * ((i - len(self.slots)) / (len(self.order) - len(self.slots))) ** (1 / 1.6)
                age = t - t_in
            val = 1.4 if cur else (0.75 + 1.2 * math.exp(-age / 0.6) if i >= len(self.slots) else 0.75)
            for run in visible_runs(pts, zz):
                gl.poly(gl.A, run, val, th + (1 if cur and rpx > 900 else 0))
            if i < len(self.slots) and frac > 0.5:
                c = self.meta[k]
                cxy, cz = v.project(rivers()[1]["c_xy"][k])
                cp = partial(cxy, smooth((frac - 0.5) / 0.3))
                bright = 1.1 if cur else 0.0
                if bright > 0:
                    for run in visible_runs(cp, cz[:len(cp)]):
                        gl.poly(gl.B, run, bright, 2)
                    if frac > 0.8:
                        for p in (cxy[0], cxy[-1]):
                            gl.poly(gl.B, [p - 1, p + 1], 2.0, 6)
        im = Image.fromarray(gl.compose(img))
        d = ImageDraw.Draw(im, "RGBA")
        cnt = min(n + (1 if f > 0.5 else 0), len(self.order))
        if cnt > 0:
            ratio = self.cumL[cnt - 1] / self.cumD[cnt - 1]
            mean = self.cumS[cnt - 1] / cnt
        else:
            ratio = mean = None
        a_g = ST.fade(t, 2.0, 1e9, 0.6, 0.6)
        gauge(d, ratio, a_g, mean=mean if cnt > 3 else None)
        if t < self.t_rest:
            j = n
            k, a, b = self.slots[min(j, len(self.slots) - 1)]
            ac = ST.fade(t, a + self.MOVE * 0.6, b, 0.4, 0.35)
            river_card(d, self.meta[k], ac, n=k + 1, prog=f)
        else:
            a = ST.fade(t, self.t_rest + 1.0, 1e9, 0.6, 0.6)
            text(d, (80, 110), "RIVERS", 22, MUTED, a)
            text(d, (80, 136), f"{cnt:,}", 72, INK, a, "Bold")
            text(d, (80, 236), "ALONG THE RIVERS", 22, MUTED, a)
            text(d, (80, 262), km(self.cumL[cnt - 1]), 44, CYAN, a, "Bold")
            text(d, (80, 330), "IN STRAIGHT LINES", 22, MUTED, a)
            text(d, (80, 356), km(self.cumD[cnt - 1]), 44, CHORD, a, "Bold")
        if t < 5.8:
            ST.caption(d, [("We traced the 1,000 longest rivers on Earth,", 34, INK, "SemiBold"),
                           ("source to mouth, in OpenStreetMap.", 34, INK, "SemiBold")], ST.fade(t, 0.2, 5.6, 0.5, 0.5), y=60)
        if t > self.t_hold + 0.5:
            ST.caption(d, [(f"All 1,000 together: {self.cumL[-1] / self.cumD[-1]:.2f}.", 40, INK, "Bold"),
                           ("Not π.", 40, GOLD, "Bold")], ST.fade(t, self.t_hold + 0.8, 1e9, 0.6, 0.6), y=60)
        return np.asarray(im)


# ================================================================== the verdict: all 1,000 ratios
def panel_bg():
    im = Image.new("RGB", (W, H), BG)
    return im


class Verdict(Section):
    """a histogram of the 1,000 ratios, building up, with pi marked"""
    name, dur = "verdict", 19.5

    def __init__(self):
        meta, _ = rivers()
        self.S = np.array([m["S"] for m in meta])
        self.A = analysis()
        self.edges = np.arange(1.0, 5.01, 0.1)
        self.counts = np.histogram(np.clip(self.S, 1.0, 4.999), self.edges)[0]

    def frame(self, t):
        im = panel_bg()
        d = ImageDraw.Draw(im, "RGBA")
        x0, x1, yb, ht = 220, 1720, 860, 520
        X = lambda v: x0 + (v - 1.0) / 4.0 * (x1 - x0)
        cmax = self.counts.max()
        grow = smooth(t / 3.0)
        a0 = ST.fade(t, 0.0, 1e9, 0.4, 0.4)
        text(d, (x0, 150), "THE 1,000 LONGEST RIVERS: LENGTH ÷ STRAIGHT LINE", 24, MUTED, a0)
        for i, c in enumerate(self.counts):
            lo, hi = self.edges[i], self.edges[i + 1]
            h = c / cmax * ht * grow
            near = abs((lo + hi) / 2 - PI) < 0.2
            col = GOLD if (near and t > 12.0) else CYAN
            d.rectangle([X(lo) + 2, yb - h, X(hi) - 2, yb], fill=col + (int(235 * a0),))
        d.line([(x0, yb), (x1, yb)], fill=INK2 + (int(255 * a0),), width=2)
        for v in range(1, 6):
            text(d, (X(v), yb + 36), f"{v}", 26, INK2, a0, anchor="mm")
        text(d, (X(5.0) + 22, yb + 36), "+", 26, INK2, a0, anchor="mm")
        ap = ST.fade(t, 3.0, 1e9, 0.5, 0.5)
        d.line([(X(PI), yb - ht - 40), (X(PI), yb)], fill=GOLD + (int(255 * ap),), width=4)
        text(d, (X(PI), yb - ht - 70), "π", 44, GOLD, ap, "Bold", anchor="mm")
        am = ST.fade(t, 5.0, 1e9, 0.5, 0.5)
        r = self.A["ratio_of_totals"]
        d.line([(X(r), yb - ht - 10), (X(r), yb)], fill=INK + (int(220 * am),), width=3)
        text(d, (X(r) + 16, yb - ht - 30), f"ALL TOGETHER {r:.2f}", 26, INK, am, "Bold", anchor="lm")
        text(d, (X(r) + 16, yb - ht + 4), f"median {self.A['median']:.2f}, average {self.A['mean']:.2f}", 22, INK2, am, anchor="lm")
        p25, p75 = self.A["percentiles"]["25"], self.A["percentiles"]["75"]
        aq = ST.fade(t, 8.5, 1e9, 0.5, 0.5)
        d.line([(X(p25), yb + 70), (X(p75), yb + 70)], fill=CYAN + (int(255 * aq),), width=6)
        text(d, ((X(p25) + X(p75)) / 2, yb + 100), f"half of all rivers: {p25:.1f} to {p75:.1f}", 24, CYAN, aq, anchor="mm")
        an = ST.fade(t, 12.0, 1e9, 0.5, 0.5)
        text(d, (X(PI) + 30, yb - ht + 60), f"{self.A['n_near']} rivers", 34, GOLD, an, "Bold")
        text(d, (X(PI) + 30, yb - ht + 104), "within 0.2 of π", 26, GOLD, an)
        return np.asarray(im)


# ================================================================== what the near-pi rivers share
class Common(Section):
    """four short charts: flatness, lowness, two routes to pi, where"""
    name = "common"
    PARTS = [("flat", 11.0), ("low", 10.0), ("routes", 14.0), ("where", 14.0)]

    def __init__(self):
        self.dur = sum(d for _, d in self.PARTS)
        self.A = analysis()
        meta, _ = rivers()
        self.meta = meta
        self.S = np.array([m["S"] for m in meta])
        self.M = np.array([m["M"] for m in meta]); self.C = np.array([m["C"] for m in meta])
        self.near = np.abs(self.S - PI) < 0.2

    def part(self, t):
        t0 = 0.0
        for nm, d in self.PARTS:
            if t < t0 + d:
                return nm, t - t0, d
            t0 += d
        return self.PARTS[-1][0], t - (t0 - self.PARTS[-1][1]), self.PARTS[-1][1]

    def frame(self, t):
        im = panel_bg()
        d = ImageDraw.Draw(im, "RGBA")
        nm, u, dd = self.part(t)
        a = ST.fade(u, 0.0, dd, 0.45, 0.45)
        getattr(self, "p_" + nm)(d, u, a)
        return np.asarray(im)

    def title(self, d, s, sub, a):
        text(d, (W / 2, 120), s, 48, INK, a, "Bold", anchor="mm")
        if sub:
            text(d, (W / 2, 180), sub, 28, INK2, a, anchor="mm")

    def p_flat(self, d, u, a):
        self.title(d, "The flatter the land, the closer to π", "length ÷ straight line, all rivers together, by how steeply they fall", a)
        B = self.A["by_gradient"]
        x0, yb, bw, gap, ht = 330, 880, 200, 70, 560
        Y = lambda v: yb - (v - 1.0) / (PI - 1.0) * ht
        d.line([(x0 - 40, Y(PI)), (x0 + 5 * (bw + gap), Y(PI))], fill=GOLD + (int(255 * a),), width=3)
        text(d, (x0 + 5 * (bw + gap) + 10, Y(PI)), "π", 40, GOLD, a, "Bold", anchor="lm")
        for i, b in enumerate(B[::-1]):                       # steep on the left, flat on the right
            g = smooth((u - 0.6 - 0.35 * i) / 0.8)
            v = 1.0 + (b["ratio"] - 1.0) * g
            x = x0 + i * (bw + gap)
            d.rectangle([x, Y(v), x + bw, yb], fill=CYAN + (int(230 * a),))
            if g > 0.9:
                text(d, (x + bw / 2, Y(v) - 30), f"{b['ratio']:.2f}", 36, INK, a, "Bold", anchor="mm")
            text(d, (x + bw / 2, yb + 36), b["label"], 26, INK2, a, anchor="mm")
            text(d, (x + bw / 2, yb + 70), f"{b['n']} rivers", 20, MUTED, a, anchor="mm")
        text(d, (x0 + 2.5 * (bw + gap) - gap / 2, yb + 120), "METRES OF FALL PER KILOMETRE OF RIVER", 22, MUTED, a, anchor="mm")
        d.line([(x0 - 40, yb), (x0 + 5 * (bw + gap) - gap + 40, yb)], fill=INK2 + (int(255 * a),), width=2)

    def p_low(self, d, u, a):
        self.title(d, f"The {self.A['n_near']} rivers nearest π run low", "", a)
        c = {x["feature"]: x for x in self.A["near_vs_rest"]}
        rows = [("share of the river below 200 m", f"{c['lowland_share']['near_median']:.0%}", f"{c['lowland_share']['rest_median']:.0%}"),
                ("median height above the sea", f"{c['median_elev_m']['near_median']:.0f} m", f"{c['median_elev_m']['rest_median']:.0f} m"),
                ("fall per km", f"{c['gradient_m_per_km']['near_median']:.2f} m", f"{c['gradient_m_per_km']['rest_median']:.2f} m")]
        text(d, (1180, 300), "NEAR π", 26, GOLD, a, "Bold", anchor="mm")
        text(d, (1500, 300), "THE OTHERS", 26, MUTED, a, "Bold", anchor="mm")
        for i, (lab, v1, v2) in enumerate(rows):
            b = a * smooth((u - 0.5 - 0.8 * i) / 0.5)
            y = 420 + i * 150
            text(d, (300, y), lab, 36, INK, b, anchor="lm")
            text(d, (1180, y), v1, 56, GOLD, b, "Bold", anchor="mm")
            text(d, (1500, y), v2, 56, INK2, b, "Bold", anchor="mm")
        text(d, (W / 2, 930), "medians", 22, MUTED, a, anchor="mm")

    def p_routes(self, d, u, a):
        self.title(d, "Two ways to reach π", "the extra length in wiggles under 10 km (meanders) and in bends over 10 km", a)
        x0, y0, sz = 560, 900, 640
        lo, hi = math.log(1.0), math.log(3.2)
        X = lambda m: x0 + (math.log(m) - lo) / (hi - lo) * sz
        Y = lambda c: y0 - (math.log(c) - lo) / (hi - lo) * sz
        al = int(255 * a)
        d.line([(x0, y0), (x0 + sz, y0)], fill=INK2 + (al,), width=2)
        d.line([(x0, y0), (x0, y0 - sz)], fill=INK2 + (al,), width=2)
        for v in (1, 1.5, 2, 3):
            text(d, (X(v), y0 + 30), f"×{v:g}", 22, INK2, a, anchor="mm")
            text(d, (x0 - 20, Y(v)), f"×{v:g}", 22, INK2, a, anchor="rm")
        text(d, (x0 + sz / 2, y0 + 70), "MEANDERS (wiggles under 10 km)", 22, MUTED, a, anchor="mm")
        text(d, (x0 - 60, y0 - sz - 40), "BENDS (over 10 km)", 22, MUTED, a, anchor="lm")
        # the curve meanders x bends = pi
        ap = a * smooth((u - 1.0) / 0.8)
        pts = [(X(m), Y(PI / m)) for m in np.exp(np.linspace(math.log(PI / 3.2), math.log(3.2), 80)) if 1 <= PI / m <= 3.2 and 1 <= m <= 3.2]
        if len(pts) > 1 and ap > 0:
            d.line(pts, fill=GOLD + (int(200 * ap),), width=3)
            text(d, (X(3.0) + 10, Y(PI / 3.0) - 26), "= π", 30, GOLD, ap, "Bold")
        ad = a * smooth((u - 0.4) / 0.8)
        for m, c, nr in zip(self.M, self.C, self.near):
            if not (1 <= m <= 3.2 and 1 <= c <= 3.2):
                continue
            x, y = X(m), Y(c)
            if nr:
                continue
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=CYAN + (int(110 * ad),))
        ag = a * smooth((u - 2.0) / 0.8)
        for m, c, nr in zip(self.M, self.C, self.near):
            if nr and 1 <= m <= 3.2 and 1 <= c <= 3.2:
                x, y = X(m), Y(c)
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=GOLD + (int(255 * ag),))
        al2 = a * smooth((u - 3.5) / 0.6)
        R = self.A["near_routes"]
        text(d, (1360, 360), f"{R['bends']} by big bends", 34, GOLD, al2, "Bold")
        text(d, (1360, 404), "Brahmaputra: round the Himalayas", 24, INK2, al2)
        text(d, (1360, 640), f"{R['meanders']} by meanders", 34, GOLD, al2, "Bold")
        text(d, (1360, 684), "Siberian tundra, Australian plains", 24, INK2, al2)

    def p_where(self, d, u, a):
        self.title(d, "Near the equator, and in the Arctic", "where the rivers nearest π reach the sea or end", a)
        # a plain equirectangular world, the 1,000 mouths as dots, the 36 in gold
        x0, y0, w, h = 260, 260, 1400, 600
        X = lambda lon: x0 + (lon + 180) / 360 * w
        Y = lambda lat: y0 + (75 - lat) / 135 * h
        tex = land_mask(w, h)
        al = int(255 * a)
        d.bitmap((x0, y0), tex, fill=(26, 34, 50, int(255 * a)))
        for lat in (0, 66.56):
            d.line([(x0, Y(lat)), (x0 + w, Y(lat))], fill=MUTED + (int(120 * a),), width=1)
        text(d, (x0 + w + 12, Y(0)), "equator", 20, MUTED, a, anchor="lm")
        text(d, (x0 + w + 12, Y(66.56)), "Arctic Circle", 20, MUTED, a, anchor="lm")
        ad = a * smooth((u - 0.4) / 0.6)
        for m, nr in zip(self.meta, self.near):
            if not nr:
                x, y = X(m["mouth"][0]), Y(m["mouth"][1])
                d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=CYAN + (int(140 * ad),))
        ag = a * smooth((u - 1.4) / 0.6)
        for m, nr in zip(self.meta, self.near):
            if nr:
                x, y = X(m["mouth"][0]), Y(m["mouth"][1])
                d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=GOLD + (int(255 * ag),))
        lat = np.abs(np.array([m["lat"] for m in self.meta]))
        eq = self.near[lat < 10].mean(); mid = self.near[(lat >= 40) & (lat < 60)].mean(); arc = self.near[lat >= 60].mean()
        at = a * smooth((u - 3.0) / 0.6)
        text(d, (W / 2, 920), f"within 10° of the equator: {eq:.0%} are near π.   40–60°: {mid:.0%}.   Beyond 60°: {arc:.0%}.",
             28, INK, at, anchor="mm")
        at2 = a * smooth((u - 5.5) / 0.6)
        text(d, (W / 2, 975), "Flat Arctic tundra rivers come closest of all: a typical ratio of 2.7.", 28, GOLD, at2, anchor="mm")


@functools.lru_cache(None)
def land_mask(w, h):
    """Natural Earth land as a 1-bit image, equirectangular, 75N to 60S"""
    gj = json.load(open(os.path.join(ROOT, "data", "ne", "ne_10m_land.geojson")))
    m = np.zeros((h, w), np.uint8)
    for f in gj["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for k, ring in enumerate(poly):
                c = np.asarray(ring)
                p = np.c_[(c[:, 0] + 180) / 360 * w, (75 - c[:, 1]) / 135 * h]
                cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 255 if k == 0 else 0, cv2.LINE_AA, shift=4)
    return Image.fromarray(m).convert("1")


# ================================================================== the rivers nearest pi
class Near(Section):
    """all 1,000 dim, the 36 nearest pi in gold; then five of them up close"""
    name = "near"
    INTRO = 8.5
    SPOT = 7.5
    PICKS = [("Brahmaputra", "The nearest to π of all 1,000. Mostly big bends:", "east behind the Himalayas, then south to the sea."),
             ("Lachlan", "Most of its extra length is in meanders,", "across the flat plains of New South Wales."),
             ("Messoyakha", "Meanders across the flat", "Siberian tundra to the Arctic Ocean."),
             ("Bengawan Solo", "Java's longest river,", "winding across its lowlands."),
             ("Sepik", "Winds through the swampy lowlands", "of New Guinea.")]

    def __init__(self):
        meta, Z = rivers()
        self.meta, self.Z = meta, Z
        self.near = [k for k, m in enumerate(meta) if abs(m["S"] - PI) < 0.2]
        by = {m["display"]: k for k, m in enumerate(meta)}
        self.spots = [(by[nm], l1, l2) for nm, l1, l2 in self.PICKS]
        self.views = {k: fit_view(k, frac=0.58, rmax=11000) for k, _, _ in self.spots}
        # sharper relief under the small rivers' close-ups
        import globe16 as G
        patches = []
        for k, _, _ in self.spots:
            xy = river_xy(k, full=True)
            if np.ptp(xy[:, 0]) < 12:
                lo0, lo1 = xy[:, 0].min() - 3, xy[:, 0].max() + 3
                la0, la1 = xy[:, 1].min() - 2, xy[:, 1].max() + 2
                patches.append((f"rivpi_{k}", round(la0, 1), round(la1, 1), round(lo0, 1), round(lo1, 1), 150))
        self.g = G.Globe16(patches)
        self.dur = self.INTRO + self.SPOT * len(self.spots) + 1.0
        self.over = (95.0, 22.0, 500.0)

    def camera(self, t):
        if t < self.INTRO:
            return self.over[0] - 2.0 * t, self.over[1], self.over[2]
        j = min(int((t - self.INTRO) / self.SPOT), len(self.spots) - 1)
        u = smooth((t - self.INTRO - j * self.SPOT) / 1.2)
        prev = self.views[self.spots[j - 1][0]] if j else (self.over[0] - 2.0 * self.INTRO, self.over[1], self.over[2])
        cur = self.views[self.spots[j][0]]
        lon, lat = slerp(prev[:2], cur[:2], u)
        rpx = math.exp(math.log(prev[2]) + (math.log(cur[2]) - math.log(prev[2])) * u)
        return lon, lat, rpx

    def frame(self, t):
        lon, lat, rpx = self.camera(t)
        v = GlobeView(lon, lat, rpx, cx=W / 2 + 180, g=self.g)
        img = v.base() * 0.85
        gl = Glow()
        j = min(int((t - self.INTRO) / self.SPOT), len(self.spots) - 1) if t >= self.INTRO else -1
        focus = self.spots[j][0] if j >= 0 else None
        uj = (t - self.INTRO - j * self.SPOT) if j >= 0 else 0.0
        dim = 0.32
        for k in range(len(self.meta)):
            if k in self.near:
                continue
            xy, z = v.project(river_xy(k))
            for run in visible_runs(xy, z):
                gl.poly(gl.A, run, dim, 1)
        for k in self.near:
            close = k == focus and rpx > 900
            xy, z = v.project(river_xy(k, full=close))
            val = 1.6 if k == focus else 0.9
            for run in visible_runs(xy, z):
                gl.poly(gl.C, run, val, 3 if k == focus else 2)
        if focus is not None and uj > 1.4:
            cxy, cz = v.project(self.Z["c_xy"][focus])
            cp = partial(cxy, smooth((uj - 1.4) / 0.8))
            for run in visible_runs(cp, cz[:len(cp)]):
                gl.poly(gl.B, run, 1.1, 2)
        im = Image.fromarray(gl.compose(img))
        d = ImageDraw.Draw(im, "RGBA")
        if t < self.INTRO:
            ST.caption(d, [(f"{len(self.near)} of the 1,000 come within 0.2 of π.", 38, INK, "Bold"),
                           ("Here they are.", 30, GOLD, "SemiBold")], ST.fade(t, 0.4, self.INTRO - 0.2, 0.5, 0.4), y=60)
        else:
            k, l1, l2 = self.spots[j]
            m = self.meta[k]
            a = ST.fade(uj, 0.9, self.SPOT, 0.4, 0.35)
            x, y = 80, 170
            d.rounded_rectangle([x - 34, y - 36, x + 520, y + 470], 20, fill=ST.PANEL + (int(205 * a),))
            text(d, (x, y - 10), f"RIVER {m['rank']:,} OF 1,000", 20, MUTED, a)
            text(d, (x, y + 22), m["display"], 52, INK, a, "Bold")
            text(d, (x, y + 104), "ALONG THE RIVER", 20, MUTED, a)
            text(d, (x, y + 128), km(m["L"]), 40, GOLD, a, "Bold")
            text(d, (x, y + 188), "STRAIGHT LINE", 20, MUTED, a)
            text(d, (x, y + 212), km(m["D"]), 40, CHORD, a, "Bold")
            text(d, (x, y + 272), "RATIO", 20, MUTED, a)
            text(d, (x, y + 294), f"{m['S']:.2f}", 64, GOLD, a, "Bold")
            text(d, (x, y + 390), l1, 24, INK2, a)
            text(d, (x, y + 422), l2, 24, INK2, a)
        return np.asarray(im)


# ================================================================== 1926: Einstein's tea leaves
TEA = (120, 70, 28)
SAND = (214, 190, 140)
BANK = (70, 58, 44)


def arrow(d, p0, p1, col, a, w=4, head=14):
    if a <= 0:
        return
    c = tuple(col) + (int(255 * a),)
    d.line([p0, p1], fill=c, width=w)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s_ in (-1, 1):
        q = (p1[0] - head * math.cos(ang + s_ * 0.45), p1[1] - head * math.sin(ang + s_ * 0.45))
        d.line([p1, q], fill=c, width=w)


def loop_arrows(d, cx, cy, rx, ry, phase, col, a, n=6, cw=True):
    """dashes moving round an ellipse: a circulation cell"""
    for k in range(n):
        u0 = (k / n + phase) % 1.0
        th0 = 2 * math.pi * u0 * (1 if cw else -1)
        th1 = th0 + (0.32 if cw else -0.32)
        p0 = (cx + rx * math.cos(th0), cy + ry * math.sin(th0))
        p1 = (cx + rx * math.cos(th1), cy + ry * math.sin(th1))
        arrow(d, p0, p1, col, a, 3, 10)


class Einstein(Section):
    name, dur = "einstein", 30.0
    T2 = 14.0                                     # the river bend

    def __init__(self):
        rng = np.random.default_rng(5)
        self.leaf_r = rng.uniform(0.25, 0.95, 46)
        self.leaf_a = rng.uniform(0, 2 * math.pi, 46)

    def frame(self, t):
        im = panel_bg()
        d = ImageDraw.Draw(im, "RGBA")
        a1 = ST.fade(t, 0.0, self.T2, 0.5, 0.5)
        a2 = ST.fade(t, self.T2, self.dur, 0.5, 0.5)
        text(d, (W / 2, 100), "1926: Albert Einstein, on tea leaves and rivers", 44, INK, ST.fade(t, 0.2, self.dur, 0.5, 0.5), "Bold", anchor="mm")
        if a1 > 0:
            self.teacup(d, t, a1)
        if a2 > 0:
            self.bend(d, t - self.T2, a2)
        return np.asarray(im)

    def teacup(self, d, t, a):
        cx, cy, R = 560, 500, 230
        al = int(255 * a)
        d.ellipse([cx - R - 18, cy - R - 18, cx + R + 18, cy + R + 18], fill=(225, 228, 235, al))
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=TEA + (al,))
        spin = 2.2 * (1 - math.exp(-max(t - 1.0, 0) / 0.6)) * math.exp(-max(t - 1.0, 0) / 9.0)
        rot = sum(0.0 for _ in ()) + 2.2 * 0.6 * (1 - math.exp(-max(t - 1.0, 0) / 0.6)) * 4
        ang = 1.6 * max(t - 1.0, 0) - 0.05 * max(t - 1.0, 0) ** 2
        for k in range(10):                                          # swirl streaks
            r = R * (0.2 + 0.075 * k)
            th = ang * (1.4 - 0.08 * k) + k
            d.arc([cx - r, cy - r, cx + r, cy + r], math.degrees(th), math.degrees(th) + 50, fill=(160, 100, 50, int(150 * a)), width=3)
        gather = smooth((t - 1.5) / 6.0)
        for r0, a0 in zip(self.leaf_r, self.leaf_a):
            r = R * r0 * (1 - 0.82 * gather * (0.75 + 0.25 * r0))
            th = a0 + ang * (1.6 - r / R)
            x, y = cx + r * math.cos(th), cy + r * math.sin(th)
            d.ellipse([x - 7, y - 4, x + 7, y + 4], fill=(30, 40, 20, al))
        text(d, (cx, cy + R + 50), "from above", 22, MUTED, a, anchor="mm")
        # cross-section, with the circulation that carries the leaves in
        x0, x1, yt, yb = 1040, 1640, 330, 710
        d.polygon([(x0, yt), (x1, yt), (x1 - 40, yb), (x0 + 40, yb)], fill=TEA + (int(200 * a),))
        d.line([(x0, yt - 10), (x0 + 40, yb), (x1 - 40, yb), (x1, yt - 10)], fill=(225, 228, 235, al), width=12)
        ph = max(t - 1.0, 0) * 0.25
        ca = a * smooth((t - 2.0) / 1.0)
        loop_arrows(d, (x0 + x1) / 2 - 150, (yt + yb) / 2, 120, 140, ph, INK, ca, cw=False)
        loop_arrows(d, (x0 + x1) / 2 + 150, (yt + yb) / 2, 120, 140, ph, INK, ca, cw=True)
        for k in range(9):
            u = (k + 0.5) / 9
            xg = x0 + 60 + u * (x1 - x0 - 120)
            xg = (x0 + x1) / 2 + (xg - (x0 + x1) / 2) * (1 - 0.85 * gather)
            d.ellipse([xg - 8, yb - 14, xg + 8, yb - 4], fill=(30, 40, 20, al))
        text(d, ((x0 + x1) / 2, yb + 50), "from the side", 22, MUTED, a, anchor="mm")
        ST.caption(d, [("Stir a cup of tea: the leaves end up in the middle, not at the rim.", 32, INK, "SemiBold"),
                       ("Friction slows the swirl at the bottom, and the water there drifts inwards, carrying them.", 28, INK2, "SemiBold")],
                   ST.fade(t, 0.8, 13.6, 0.5, 0.4))

    def bend(self, d, t, a):
        al = int(255 * a)
        g = smooth((t - 2.5) / 6.5)                                   # how far the bend has grown
        # top view of a bend: the channel is a band between two arcs; the outer bank moves out, a sand bar fills in
        cx, cy = 560, 700
        r_in, r_out = 150 - 10 * g, 270 + 70 * g
        d.pieslice([cx - r_out - 30, cy - r_out - 30, cx + r_out + 30, cy + r_out + 30], 180, 360, fill=BANK + (al,))
        d.pieslice([cx - r_out, cy - r_out, cx + r_out, cy + r_out], 180, 360, fill=(40, 90, 140, al))
        d.pieslice([cx - r_in - 60 * g, cy - r_in - 60 * g, cx + r_in + 60 * g, cy + r_in + 60 * g], 180, 360, fill=SAND + (int(255 * a * min(1, 0.25 + g)),))
        d.pieslice([cx - r_in, cy - r_in, cx + r_in, cy + r_in], 180, 360, fill=BG + (al,))
        d.rectangle([cx - r_out - 40, cy, cx + r_out + 40, cy + 40], fill=BG + (al,))
        ph = (t * 0.35) % 1.0
        for k in range(5):
            th = math.pi * (1 + (k / 5 + ph) % 1.0)
            rm = (r_in + 60 * g + r_out) / 2
            p0 = (cx + rm * math.cos(th), cy + rm * math.sin(th))
            p1 = (cx + rm * math.cos(th + 0.28), cy + rm * math.sin(th + 0.28))
            arrow(d, p0, p1, INK, a, 3, 10)
        text(d, (cx - r_out + 10, cy + 40), "outer bank: dug away", 22, (230, 150, 110), a)
        text(d, (cx, cy - 60), "inner bank:", 22, SAND, a * min(1, g * 3), anchor="mm")
        text(d, (cx, cy - 30), "sand piles up", 22, SAND, a * min(1, g * 3), anchor="mm")
        text(d, (cx + r_out - 10, cy + 40), "from above", 22, MUTED, a, anchor="ra")
        # cross-section at the bend: tilted surface, helical circulation
        x0, x1, yt, yb = 1040, 1700, 380, 660
        d.polygon([(x0, yt + 20), (x1, yt - 10), (x1 - 30, yb), (x0 + 200, yb), (x0 + 30, yt + 120)], fill=(40, 90, 140, al))
        d.polygon([(x0 - 40, yt + 20), (x0, yt + 20), (x0 + 30, yt + 120), (x0 + 200, yb), (x0 - 40, yb)],
                  fill=SAND + (int(255 * a * min(1, 0.3 + g)),))
        d.polygon([(x1, yt - 10), (x1 + 40, yt - 10), (x1 + 40, yb + 20), (x1 - 30, yb)], fill=BANK + (al,))
        loop_arrows(d, (x0 + x1) / 2 + 60, (yt + yb) / 2 + 20, 230, 100, t * 0.25, INK, a * smooth((t - 1.0) / 1.0), n=7, cw=True)
        text(d, (x0 + 40, yb + 50), "inner", 22, MUTED, a)
        text(d, (x1, yb + 50), "outer", 22, MUTED, a, anchor="ra")
        text(d, ((x0 + x1) / 2, yb + 90), "across the bend", 22, MUTED, a, anchor="mm")
        ST.caption(d, [("In a river bend the same swirl digs the outer bank and drops sand on the inner one.", 30, INK, "SemiBold"),
                       ("So every bend grows. Rivers meander.", 30, GOLD, "SemiBold")], ST.fade(t, 0.5, 15.5, 0.5, 0.4))
        text(d, (W - 60, H - 40), "A. Einstein, Die Naturwissenschaften 14, 223 (1926)", 20, MUTED, a, anchor="rm")


# ================================================================== 1996: Stolum's meandering model river
@functools.lru_cache(None)
def meander_run():
    z = np.load(os.path.join(DATA, "meander_video.npz"))
    return {k: z[k] for k in z.files}


class Stolum(Section):
    name, dur = "stolum", 36.0
    T0 = 1.0                                      # the river starts moving

    def __init__(self):
        self.R = meander_run()
        self.nf = len(self.R["off"]) - 1
        S = self.R["S"]
        self.mean = float(S[len(S) // 3:].mean())
        # the camera follows the meander belt as it widens and narrows (eased)
        xy, off = self.R["xy"], self.R["off"]
        x0 = np.array([np.percentile(xy[off[i]:off[i + 1], 0], 1) for i in range(self.nf)])
        x1 = np.array([np.percentile(xy[off[i]:off[i + 1], 0], 99) for i in range(self.nf)])
        y1 = np.array([np.percentile(np.abs(xy[off[i]:off[i + 1], 1]), 98) for i in range(self.nf)])
        sc = np.minimum(np.minimum(1750 / (x1 - x0), 230 / np.maximum(y1 * 1.15, 5.0)), 10.0)
        cx = (x0 + x1) / 2
        for i in range(1, self.nf):
            sc[i] = sc[i - 1] + (sc[i] - sc[i - 1]) * (0.3 if sc[i] < sc[i - 1] else 0.04)
            cx[i] = cx[i - 1] + (cx[i] - cx[i - 1]) * 0.04
        self.scale, self.cxs = sc, cx

    def fi(self, t):
        u = min(max((t - self.T0) / (self.dur - self.T0 - 1.0), 0.0), 1.0)
        return int(round((self.nf - 1) * u ** 1.25))

    def XY(self, p, f):
        return np.c_[W / 2 + (p[:, 0] - self.cxs[f]) * self.scale[f], 375 - p[:, 1] * self.scale[f]]

    def frame(self, t):
        f = self.fi(t)
        R = self.R
        gl = Glow()
        it_now = f * int(R["every"])
        for k in range(len(R["ox_it"])):
            age = (it_now - R["ox_it"][k]) / (int(R["every"]) * self.nf * 0.08)
            if age < 0:
                continue
            o = R["ox_xy"][R["ox_off"][k]:R["ox_off"][k + 1]]
            if age > 4:
                continue
            gl.poly(gl.B, self.XY(o, f), 0.4 * math.exp(-age), 2, closed=True)
            if age < 0.06:
                gl.poly(gl.C, self.XY(o, f), 1.5, 3, closed=True)       # a cutoff flashes as it happens
        c = R["xy"][R["off"][f]:R["off"][f + 1]]
        gl.poly(gl.A, self.XY(c, f), 1.25, 4)
        img = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32) / 255
        im = Image.fromarray(gl.compose(img))
        d = ImageDraw.Draw(im, "RGBA")
        a = ST.fade(t, 0.0, self.dur, 0.5, 0.5)
        text(d, (W / 2, 70), "1996: Hans-Henrik Stølum lets a model river meander across a flat plain", 36, INK, a, "Bold", anchor="mm")
        # the strip chart of length / straight line
        x0, x1, yb, yt = 260, 1660, 960, 760
        Y = lambda v: yb - (v - 1.0) / 3.5 * (yb - yt)
        al = int(255 * a)
        d.line([(x0, yb), (x1, yb)], fill=INK2 + (al,), width=2)
        for v in (1, 2, 3, 4):
            text(d, (x0 - 16, Y(v)), f"{v}", 20, MUTED, a, anchor="rm")
        d.line([(x0, Y(PI)), (x1, Y(PI))], fill=GOLD + (int(200 * a),), width=2)
        text(d, (x1 + 12, Y(PI)), "π", 30, GOLD, a, "Bold", anchor="lm")
        S = R["S"][:f + 1]
        if len(S) > 1:
            xs = x0 + np.arange(len(S)) / (self.nf - 1) * (x1 - x0)
            d.line(list(zip(xs, [Y(v) for v in np.clip(S, 1, 4.5)])), fill=CYAN + (al,), width=3)
        text(d, (x0, yt - 40), "LENGTH ÷ STRAIGHT LINE", 20, MUTED, a)
        text(d, (x0 + 300, yt - 50), f"{R['S'][f]:.2f}", 40, CYAN, a, "Bold")
        caps = [(0.5, 7.5, "Every bend grows, faster where it is sharpest."),
                (7.8, 15.8, "When two loops touch, the river cuts through, leaving an oxbow lake."),
                (16.1, 23.6, "The ratio climbs, collapses at each cutoff, and climbs again."),
                (23.9, self.dur, f"Stølum's model river hovered around π. This one averages {self.mean:.1f}.")]
        for c0, c1, s_ in caps:
            ca = ST.fade(t, c0, c1, 0.5, 0.4)
            if ca > 0:
                text(d, (W / 2, 690), s_, 30, INK, ca, "SemiBold", anchor="mm")
        text(d, (W - 60, H - 30), "H.-H. Stølum, Science 271, 1710 (1996); model of Howard & Knutson (1984)", 20, MUTED, a, anchor="rm")
        return np.asarray(im)


# ================================================================== the cold open: one river, measured
class Open(Section):
    """a meandering river (from the model) drawn, measured along its bends and in a straight line, and divided"""
    name, dur = "open", 22.0
    DRAW0, DRAW1, LINE0, LINE1, DIV = 0.6, 5.6, 6.0, 7.6, 8.4

    def __init__(self):
        R = meander_run()
        S = R["S"]
        k0 = len(S) // 3
        f = k0 + int(np.argmin(np.abs(S[k0:] - PI)))
        c = R["xy"][R["off"][f]:R["off"][f + 1]].astype(np.float64)
        self.S = float(S[f])
        sc = min(1500 / np.ptp(c[:, 0]), 520 / max(np.ptp(c[:, 1]), 1e-6))
        self.xy = np.c_[W / 2 + (c[:, 0] - (c[:, 0].min() + c[:, 0].max()) / 2) * sc,
                        420 - (c[:, 1] - (c[:, 1].min() + c[:, 1].max()) / 2) * sc]
        self.Lkm = 100.0 * self.S                     # a river 100 km from source to mouth, for scale

    def frame(self, t):
        gl = Glow()
        f = smooth((t - self.DRAW0) / (self.DRAW1 - self.DRAW0))
        pts = partial(self.xy, f)
        gl.poly(gl.A, pts, 1.3, 5)
        if f > 0:
            gl.poly(gl.A, [pts[-1] - 1, pts[-1] + 1], 3.0, 12)
        g = smooth((t - self.LINE0) / (self.LINE1 - self.LINE0))
        if g > 0:
            p0, p1 = self.xy[0], self.xy[-1]
            gl.poly(gl.B, [p0, p0 + (p1 - p0) * g], 1.2, 3)
        img = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32) / 255
        im = Image.fromarray(gl.compose(img))
        d = ImageDraw.Draw(im, "RGBA")
        a = ST.fade(t, 0.3, self.dur, 0.5, 0.5)
        text(d, (self.xy[0][0], self.xy[0][1] + 40), "source", 22, MUTED, a * min(1, f * 4), anchor="mm")
        text(d, (self.xy[-1][0], self.xy[-1][1] + 40), "mouth", 22, MUTED, a * smooth((f - 0.95) / 0.05), anchor="mm")
        y = 790
        text(d, (360, y), "ALONG THE RIVER", 22, MUTED, a)
        text(d, (360, y + 30), f"{self.Lkm * f:,.0f} km", 60, CYAN, a, "Bold")
        if t > self.LINE0:
            b = ST.fade(t, self.LINE0, 1e9, 0.4, 0.4)
            text(d, (760, y + 66), "÷", 60, INK2, b, "Bold", anchor="mm")
            text(d, (860, y), "STRAIGHT LINE, SOURCE TO MOUTH", 22, MUTED, b)
            text(d, (860, y + 30), f"{100.0 * g:,.0f} km", 60, CHORD, b, "Bold")
        if t > self.DIV:
            b = ST.fade(t, self.DIV, 1e9, 0.4, 0.4)
            text(d, (1400, y + 66), "=", 60, INK2, b, "Bold", anchor="mm")
            text(d, (1460, y + 30), f"{self.S:.2f}", 72, GOLD, b, "Bold")
        ST.caption(d, [("Some say that for rivers, on average, this ratio is π.", 36, INK, "SemiBold")],
                   ST.fade(t, 10.8, 16.6, 0.5, 0.4), y=60)
        at = ST.fade(t, 16.4, 1e9, 0.6, 0.6)
        if at > 0:
            d.rectangle([0, 0, W, H], fill=BG + (int(200 * at),))
            text(d, (W / 2, H / 2 - 30), "Do rivers wind by π?", 96, INK, at, "Bold", anchor="mm")
            text(d, (W / 2, H / 2 + 60), "We measured the 1,000 longest on Earth", 34, GOLD, at, "SemiBold", anchor="mm")
        return np.asarray(im)


# ================================================================== the end
class End(Section):
    name, dur = "end", 28.0

    def frame(self, t):
        A = analysis()
        im = panel_bg()
        d = ImageDraw.Draw(im, "RGBA")
        lines = [(0.3, f"The 1,000 longest rivers: {A['ratio_of_totals']:.2f}, not π.", 46, INK, "Bold"),
                 (3.3, "Stølum's π needs flat ground and endless time.", 34, INK2, "SemiBold"),
                 (6.8, "Real rivers have mountains to get round, rock to cut and a sea to reach.", 34, INK2, "SemiBold"),
                 (10.3, "The flatter the land, the closer they come: flat Arctic tundra rivers, about 2.7.", 34, INK2, "SemiBold"),
                 (14.3, "The nearest of all 1,000: the Brahmaputra, 3.16, by going round the Himalayas.", 34, GOLD, "SemiBold")]
        for k, (t0, s_, sz, col, wt) in enumerate(lines):
            a = ST.fade(t, t0, 20.5, 0.6, 0.6)
            text(d, (W / 2, 300 + k * 90 + (30 if k >= 1 else 0)), s_, sz, col, a, wt, anchor="mm")
        ac = ST.fade(t, 20.8, 1e9, 0.6, 0.6)
        credits = ["Rivers: OpenStreetMap contributors (ODbL), via Overture Maps.  Elevation: Terrarium tiles (Mapzen, AWS).",
                   "Countries and coasts: Natural Earth.  Method, data and every river: see the description."]
        for k, c in enumerate(credits):
            text(d, (W / 2, 470 + k * 50), c, 26, MUTED, ac, anchor="mm")
        text(d, (W / 2, 640), "Music: an original theme built from the digits of π.", 26, MUTED, ac, anchor="mm")
        return np.asarray(im)


SECTIONS = {"open": Open, "end": End, "stolum": Stolum, "einstein": Einstein, "test": Test, "verdict": Verdict, "common": Common, "near": Near}

if __name__ == "__main__":
    S = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            S.render(float(s))
    else:
        S.render()
