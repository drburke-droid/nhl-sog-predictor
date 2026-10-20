"""How far can you see a golf ball? The golf-ball film, section by section (4K, 30 fps), in the vernier
film's look (../vernier/look.py) and on its cone mosaic (../vernier/retina.py).

    python film.py NAME [still times...]
    NAME: open, angle, chart, eye, retina, blur, recipe, dark, demo, close
"""
import os, sys, math, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imaging as I                                     # sets up the vernier modules and look.OUT
import physics as P
import look as L
import retina as R
from look import W, H, S, Cam, smooth, ease, fade, caption, note, pencil, INK, INK_SOFT, RED, CREAM

GRAPHITE = (70, 66, 62)
CREAM_P = (236, 222, 200)
RED_P = (222, 84, 62)
RED_L = (250, 120, 92)
PX_ARCMIN = R.PX_ARCSEC / 60                            # one 1080p pixel, full screen on a phone at 30 cm (0.683′)
FOOT = "Sizes for a phone held at reading distance."


def units(arcmin):
    """an angle in 1080p pixels of this video, for a phone at reading distance"""
    return arcmin / PX_ARCMIN


def blackout(fr, a):
    if a <= 0:
        return fr
    return (fr.astype(np.float32) * (1 - a)).astype(np.uint8)


def mix(a, b, u):
    if u <= 0:
        return a
    if u >= 1:
        return b
    return cv2.addWeighted(a, 1 - u, b, u, 0)


def footnote(im, a, y=1050):
    if a > 0:
        L.text(ImageDraw.Draw(im), (1780, y), FOOT, 22, (150, 140, 130), a, kind="italic", anchor="rm")


def pen_bracket(x, y0, y1, side=1, tick=18):
    return [(x - side * tick, y0), (x, y0), (x, y1), (x - side * tick, y1)]


def circle_pts(cx, cy, r, n=72, a0=0.3):
    th = np.linspace(a0, a0 + 2 * math.pi + 0.25, n)
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]


def arrow_strokes(p0, p1, f, seed, head=16):
    """a pencil arrow from p0 to p1 (screen px), drawn to fraction f"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p0 - p1) / max(1e-6, np.hypot(*(p0 - p1)))
    heads = []
    for k, sg in enumerate((1, -1)):
        c_, s_ = math.cos(math.radians(30 * sg)), math.sin(math.radians(30 * sg))
        end = p1 + head * np.array([d[0] * c_ - d[1] * s_, d[0] * s_ + d[1] * c_])
        heads.append(([tuple(p1), tuple(end)], (f - 0.8) / 0.2, seed + 1 + k))
    return [([tuple(p0), tuple(p1)], f / 0.8, seed)] + heads


# ================================================================== the scenes: printed swatches with balls
def lin(rgb):
    return np.array(rgb, np.float32)


def srgb(lin_rgb):
    return tuple(int(v) for v in L.to_srgb(np.array(lin_rgb, np.float32)))


# linear-light colours with the luminance ratios of physics.SCENES (white ball Y = 0.88)
BALL_Y = 0.88
WHITE_BALL = lin([0.88, 0.88, 0.88])
SCENE = {
    "trees": dict(bg=lin([0.018, 0.034, 0.016]), ball=WHITE_BALL, name="dark trees"),
    "fairway": dict(bg=lin([0.080, 0.175, 0.050]), ball=WHITE_BALL, name="the fairway"),
    "sky": dict(bg=lin([0.200, 0.360, 0.680]), ball=WHITE_BALL, name="blue sky"),
    "overcast": dict(bg=lin([0.800, 0.800, 0.810]), ball=lin([0.44, 0.44, 0.44]), name="overcast sky"),
}


def Y(c):
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def disk_cov(X0, Y0, X1, Y1, cx, cy, r, n=12):
    """exact-ish area coverage (n×n samples per pixel) of a disk (screen px) over the pixel box"""
    xs = np.arange(X0, X1)[None, :, None, None] + (np.arange(n)[None, None, None, :] + 0.5) / n
    ys = np.arange(Y0, Y1)[:, None, None, None] + (np.arange(n)[None, None, :, None] + 0.5) / n
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    return inside.mean(axis=(2, 3)).astype(np.float32)


class Swatch(L.Card):
    """a printed card of one scene: a flat background with white balls at given distances (drawn to scale for
    a phone at reading distance, with exact coverage in linear light). dots: [(x, y, yards)]."""

    def __init__(self, scene, dots=(), x0=330, y0=330, x1=1590, y1=750, label=""):
        super().__init__(x0=x0, y0=y0, x1=x1, y1=y1, line_y=-999, step=0.0, label=label)
        self.scene = SCENE[scene]
        self.dots = list(dots)

    def draw(self, fr, cam, box=(0, 0, W, H), alpha=1.0, line=1.0, dots=1.0):
        face = srgb(self.scene["bg"])
        save = L.CARD
        L.CARD = face
        try:
            super().draw(fr, cam, box, alpha, line=0)
        finally:
            L.CARD = save
        if dots > 0:
            self.draw_dots(fr, cam, box, dots * alpha)

    def draw_dots(self, fr, cam, box, a):
        bx0, by0, bx1, by1 = box
        for x, y, yd in self.dots:
            r = units(float(P.ball_arcmin(yd))) / 2 * S * cam.z
            cx, cy = cam.X(x), cam.Y(y)
            X0, Y0 = int(max(bx0, math.floor(cx - r) - 1)), int(max(by0, math.floor(cy - r) - 1))
            X1, Y1 = int(min(bx1, math.ceil(cx + r) + 1)), int(min(by1, math.ceil(cy + r) + 1))
            if X1 <= X0 or Y1 <= Y0:
                continue
            cov = disk_cov(X0, Y0, X1, Y1, cx, cy, r, n=12 if r < 40 else 3) * a
            reg = fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0]
            fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0] = L.blend_lin(reg, cov, srgb(self.scene["ball"]))

    def line_box(self, cam, pad=30):
        return (int(cam.X(self.x0)), int(cam.Y(self.y0)), int(cam.X(self.x1)), int(cam.Y(self.y1)))

    def plate(self, im, cam, a=1.0):
        """the scene's name, printed small in the card's lower left corner"""
        if self.label and a > 0:
            col = (205, 200, 190) if Y(self.scene["bg"]) < 0.3 else INK_SOFT
            L.text(ImageDraw.Draw(im), (cam.X(self.x0 + 36) / S, cam.Y(self.y1 - 44) / S), self.label, 22 * cam.z, col, a, kind="italic")


def desk(cam, cards, lens=None, t=0.0, labels=1.0):
    """the desk under cam, with cards. lens: (world x, world y, radius in units, magnification, alpha)"""
    fr = L.paper(cam)
    for c in cards:
        if cam.Y(c.y1) > -200 and cam.Y(c.y0) < H + 200 and cam.X(c.x1) > -200 and cam.X(c.x0) < W + 200:
            c.draw(fr, cam)
    keeps = [c.line_box(cam) for c in cards if cam.Y(c.y1) > 0 and cam.Y(c.y0) < H]
    if lens is not None and lens[4] > 0:
        lx, ly, rad, mag, a = lens
        X, Y, Rp = cam.X(lx), cam.Y(ly), int(rad * S * cam.z)
        zm = cam.z * mag
        icam = Cam(lx - (X - W / 2) / (S * zm), ly - (Y - H / 2) / (S * zm), zm)
        box = (int(X) - Rp, int(Y) - Rp, int(X) + Rp, int(Y) + Rp)
        inner = L.paper(icam, box)
        for c in cards:
            c.draw(inner, icam, box)
        fr = L.loupe(fr, inner, X, Y, Rp, a)
    fr = finish_keep(fr, t, keeps)
    im = Image.fromarray(fr)
    for c in cards:
        if cam.Y(c.y1) > 0 and cam.Y(c.y0) < H:
            c.plate(im, cam, labels)
    return np.asarray(im).copy()


def finish_keep(fr, t, keeps):
    """lamp light over everything and film grain everywhere except the cards (several keep boxes)"""
    fr = cv2.multiply(fr, L.lamp_u8(), scale=1 / 200)
    clean = []
    for X0, Y0, X1, Y1 in keeps:
        X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
        if X1 > X0 and Y1 > Y0:
            clean.append((X0, Y0, X1, Y1, fr[Y0:Y1, X0:X1].copy()))
    pos, neg = L.grains()[int(round(t * L.FPS)) % 6]
    fr = cv2.subtract(cv2.add(fr, pos), neg)
    for X0, Y0, X1, Y1, c in clean:
        fr[Y0:Y1, X0:X1] = c
    return fr


def lens_path(t, t_in, t_out, x, y, dur=1.6, mag=7.0, rad=150):
    """the magnifier slides in from the lower right, rests over (x, y), and slides away"""
    off = np.array([760.0, 520.0])
    if t < t_in or t > t_out + dur:
        return None
    u = ease((t - t_in) / dur) if t < t_in + dur else 1 - ease((t - t_out) / dur) if t > t_out else 1.0
    px, py = np.array([x, y]) + off * (1 - u)
    return (px, py, rad, mag, 1.0)


# ================================================================== 1. a dot on a card
OPEN_YD = 400
OPEN_CARD = Swatch("trees", [(960.0, 540.0, OPEN_YD)], label="dark trees")
OPEN_HOME = Cam(960, 540, 1.0)


class Open(L.Section):
    name, dur = "open", 20.0
    LENS_IN, LENS_OUT = 8.2, 14.4

    def frame(self, t):
        cam = OPEN_HOME
        if t > 17.2:
            cam = Cam.lerp(OPEN_HOME, Cam(960, 540, 3.0), ease((t - 17.2) / 2.8))
        lens = lens_path(t, self.LENS_IN, self.LENS_OUT, 960.0, 540.0, mag=24.0)
        fr = desk(cam, [OPEN_CARD], lens=lens, t=t)
        im = Image.fromarray(fr)
        a_n = fade(t, 2.6, 8.0, 0.4, 0.6)
        if a_n > 0:
            note(im, (1150, 300), "a golf ball, 400 yards away", a_n, size=42, col=RED, anchor="ls")
        fr = np.asarray(im).copy()
        if a_n > 0:
            f = smooth((t - 2.6) / 0.8)
            fr = pencil(fr, arrow_strokes((cam.X(1200), cam.Y(312)), (cam.X(972), cam.Y(528)), f, 4), a=a_n)
        im = Image.fromarray(fr)
        caption(im, ["This white dot is a golf ball 400 yards away, against dark trees, drawn to scale."], fade(t, 1.0, 7.8, 0.7, 0.6))
        caption(im, ["Under a magnifier: it’s smaller than one pixel of this video."], fade(t, 9.6, 14.4, 0.6, 0.6))
        caption(im, ["On your retina, it’s smaller than a single cone.", "By the numbers, you shouldn’t be able to see it. But you can."],
                fade(t, 15.0, 19.6, 0.6, 0.5))
        footnote(im, fade(t, 1.0, 19.6, 0.6, 0.5))
        return blackout(np.asarray(im), 1 - smooth(t / 1.2))


# ================================================================== 2. the angle
class Angle(L.Section):
    """a notebook page: the ball, the eye, the angle; then a ladder of distances"""
    name, dur = "angle", 31.0
    EYE = np.array([250.0, 360.0])
    BALL = np.array([1650.0, 360.0])
    LADDER = [(100, 4.2), (160, 5.4), (320, 6.6), (640, 7.8)]

    def frame(self, t):
        cam = Cam(960, 540, 1.0)
        fr = L.finish(L.paper(cam), t)
        e, b = self.EYE, self.BALL
        rb = 30.0                                                # the ball, drawn (not to scale)
        st = []
        # the eye: an almond and an iris
        th = np.linspace(-0.9, 0.9, 40)
        upper = np.c_[e[0] - 60 + 120 * (th + 0.9) / 1.8, e[1] - 34 * np.cos(th * 1.7)]
        lower = np.c_[e[0] - 60 + 120 * (th + 0.9) / 1.8, e[1] + 34 * np.cos(th * 1.7)]
        st.append((upper * S, smooth((t - 0.4) / 0.6), 1))
        st.append((lower * S, smooth((t - 0.6) / 0.6), 2))
        st.append((circle_pts(e[0] + 12, e[1], 20) * S, smooth((t - 0.9) / 0.5), 3))
        st.append((circle_pts(b[0], b[1], rb) * S, smooth((t - 1.3) / 0.6), 4))
        fr = pencil(fr, st, col=GRAPHITE, width=3.0)
        lines = [([((e[0] + 40) * S, e[1] * S), (b[0] * S, (b[1] + sg * rb) * S)], smooth((t - 2.0 - 0.2 * k) / 1.2), 10 + k)
                 for k, sg in enumerate((-1, 1))]
        fr = pencil(fr, lines, col=RED, width=2.6)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        a1 = smooth((t - 3.4) / 0.6)
        L.text(d, (b[0], b[1] - 56), "1.68 in", 38, INK_SOFT, a1, kind="hand", anchor="ms")
        L.text(d, (b[0], b[1] + 64), "(42.67 mm)", 30, INK_SOFT, a1, kind="hand", anchor="mt")
        L.text(d, (e[0] + 150, e[1] - 30), "the angle", 34, RED, smooth((t - 3.8) / 0.6), kind="hand", anchor="ls")
        # the ladder: distance and angle
        y = 560.0
        for k, (yd, t0) in enumerate(self.LADDER):
            f = smooth((t - 9.0 - (t0 - 4.2) * 1.2) / 0.6)
            ang = float(P.ball_arcmin(yd))
            hi = yd == 160
            col = RED if hi else INK
            L.text(d, (660, y), f"{yd} yards", 44, col, f, kind="hand", anchor="rm")
            L.text(d, (720, y), {100: "1.6′", 160: "1′", 320: "0.5′", 640: "0.25′"}[yd], 44, col, f, kind="hand", anchor="lm")
            extra = {100: "", 160: "one arcminute", 320: "half an arcminute: one cone", 640: "a quarter"}[yd]
            L.text(d, (930, y), extra, 36, INK_SOFT if not hi else RED, f * smooth((t - 12.0 - k * 1.4) / 0.6), kind="hand", anchor="lm")
            y += 78
        caps = [(0.8, 8.4, ["A golf ball is 1.68 inches across.", "How big it looks depends on the angle it spans at your eye."]),
                (9.0, 18.6, ["At 160 yards it spans one arcminute: a sixtieth of a degree.",
                             "Twice as far, half the angle."]),
                (19.4, 30.4, ["At 320 yards, half an arcminute.",
                              "That’s the spacing of the cones at the very centre of your retina."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6))
        return blackout(np.asarray(im), 1 - smooth(t / 0.8))


# ================================================================== 3. the chart: a 20/20 E, and the ball at 160 yards
E_UNITS = units(5.0)                                     # a 20/20 E, 5′ tall, in 1080p pixels (7.3)
E_X, E_Y = 952.0, 540.0
DOT_X = E_X + 9.0


class Chart(L.Card):
    """a white card printed with a 20/20 E (black, real size for a phone at reading distance) and, beside it,
    a black dot the size of a golf ball at 160 yards"""

    def __init__(self):
        super().__init__(x0=330, y0=330, x1=1590, y1=750, line_y=-999, step=0.0, label="")

    def draw(self, fr, cam, box=(0, 0, W, H), alpha=1.0, line=1.0):
        super().draw(fr, cam, box, alpha, line=0)
        bx0, by0, bx1, by1 = box
        s = E_UNITS / 5
        x0, y0 = E_X - 2.5 * s, E_Y - 2.5 * s
        rects = [(x0, y0, x0 + s, y0 + 5 * s)] + [(x0, y0 + k * s, x0 + 5 * s, y0 + (k + 1) * s) for k in (0, 2, 4)]
        X0, Y0 = int(max(bx0, math.floor(cam.X(x0)) - 1)), int(max(by0, math.floor(cam.Y(y0)) - 1))
        X1, Y1 = int(min(bx1, math.ceil(cam.X(x0 + 5 * s)) + 1)), int(min(by1, math.ceil(cam.Y(y0 + 5 * s)) + 1))
        if X1 > X0 and Y1 > Y0:
            cov = np.zeros((Y1 - Y0, X1 - X0), np.float32)
            for a, b_, c, d in rects:
                cov = np.maximum(cov, L.rect_cov((X0, Y0, X1, Y1), cam.X(a), cam.Y(b_), cam.X(c), cam.Y(d)))
            reg = fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0]
            fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0] = L.blend_lin(reg, cov * alpha, INK)
        r = units(float(P.ball_arcmin(160))) / 2 * S * cam.z
        cx, cy = cam.X(DOT_X), cam.Y(E_Y)
        X0, Y0 = int(max(bx0, math.floor(cx - r) - 1)), int(max(by0, math.floor(cy - r) - 1))
        X1, Y1 = int(min(bx1, math.ceil(cx + r) + 1)), int(min(by1, math.ceil(cy + r) + 1))
        if X1 > X0 and Y1 > Y0:
            cov = disk_cov(X0, Y0, X1, Y1, cx, cy, r, n=12 if r < 40 else 3) * alpha
            reg = fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0]
            fr[Y0 - by0:Y1 - by0, X0 - bx0:X1 - bx0] = L.blend_lin(reg, cov, INK)

    def line_box(self, cam, pad=30):
        return (int(cam.X(self.x0)), int(cam.Y(self.y0)), int(cam.X(self.x1)), int(cam.Y(self.y1)))


CHART = Chart()


class ChartSec(L.Section):
    name, dur = "chart", 36.0
    LENS_IN, LENS_OUT = 7.0, 31.0
    MAG = 16.0

    def frame(self, t):
        cam = Cam(960, 540, 1.0)
        lx, ly = (E_X + DOT_X) / 2, E_Y
        lens = lens_path(t, self.LENS_IN, self.LENS_OUT, lx, ly, mag=self.MAG, rad=170)
        fr = desk(cam, [CHART], lens=lens, t=t)
        # pencil notes over the lens: the stroke's width and the dot's width, both one arcminute
        a_m = fade(t, 13.0, self.LENS_OUT, 0.5, 0.6)
        texts = []
        if a_m > 0 and lens is not None:
            X, Y = cam.X(lens[0]), cam.Y(lens[1])
            z = self.MAG * S
            s = E_UNITS / 5
            top = E_Y - 2.5 * s
            xs = X + (E_X + 2.5 * s - lx) * z + 26                 # right of the E's arms
            br1 = pen_bracket(xs, Y + (top - ly) * z, Y + (top + s - ly) * z, side=1, tick=14)
            yd_ = Y + (E_Y + float(units(P.ball_arcmin(160))) / 2 - ly) * z + 24
            dx0 = X + (DOT_X - float(units(P.ball_arcmin(160))) / 2 - lx) * z
            dx1 = X + (DOT_X + float(units(P.ball_arcmin(160))) / 2 - lx) * z
            br2 = [(dx0, yd_ + 14), (dx0, yd_), (dx1, yd_), (dx1, yd_ + 14)]
            fr = pencil(fr, [(br1, smooth((t - 13.0) / 0.6), 21), (br2, smooth((t - 19.6) / 0.6), 22)], col=RED, a=a_m, width=2.6)
            texts.append(((xs + 12) / S, (Y + (top + s / 2 - ly) * z) / S, "1′", smooth((t - 13.4) / 0.5), "lm"))
            texts.append((((dx0 + dx1) / 2) / S, (yd_ + 22) / S, "1′", smooth((t - 20.0) / 0.5), "mt"))
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for x_, y_, s_, a_, an in texts:
            L.text(d, (x_, y_), s_, 40, RED, a_ * a_m, kind="hand", anchor=an)
        a_l = fade(t, 2.2, self.LENS_IN + 1.0, 0.5, 0.5)
        if a_l > 0:
            note(im, (E_X - 30, E_Y - 60), "20/20 E", a_l, size=38, col=INK_SOFT, anchor="rs")
            note(im, (DOT_X + 30, E_Y + 64), "a golf ball at 160 yards", a_l, size=38, col=INK_SOFT, anchor="lt")
        caps = [(0.8, 6.8, ["On an eye chart, a 20/20 letter is 5 arcminutes tall. Here’s one, real size.",
                            "Beside it, a golf ball 160 yards away."]),
                (8.6, 18.4, ["Under the magnifier: each stroke of the E, and each gap, is one arcminute.",
                             "That’s the finest detail 20/20 vision is defined by."]),
                (19.2, 30.6, ["The golf ball at 160 yards is exactly as wide as one stroke."]),
                (31.4, 35.6, ["Now let’s look at both on your retina."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6))
        footnote(im, fade(t, 0.8, 30.6, 0.6, 0.6))
        return blackout(np.asarray(im), 1 - smooth(t / 0.8))


# ================================================================== 4. into the eye
EYE_C = np.array([1210.0, 540.0])
EYE_R = 250.0
FOVEA = EYE_C + np.array([EYE_R * 0.94, 0.0])
OBJ_X = 420.0
LENS_P = np.array([1035.0, 540.0])
PUPIL = 54.0


def arc(c, r, a0, a1, n=60, rx=None, ry=None):
    th = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + (rx or r) * np.cos(th), c[1] + (ry or r) * np.sin(th)]


def eye_strokes(cam):
    """the vernier film's eye sketch, looking at a letter instead of a screen"""
    X = lambda p: np.c_[cam.X(p[:, 0]), cam.Y(p[:, 1])]
    st = []
    st.append((X(arc(EYE_C, EYE_R, 200, 520, 120)), 0.3, 1.8, GRAPHITE, 3.0))
    st.append((X(arc(EYE_C + [-195.0, 0], 0, 120, 240, 30, rx=80, ry=98)), 1.6, 0.6, GRAPHITE, 3.0))
    st.append((X(arc(LENS_P, 0, 0, 360, 50, rx=26, ry=80)), 2.0, 0.7, GRAPHITE, 2.6))
    for sg, d in ((-1, 2.5), (1, 2.6)):
        st.append((X(np.array([[990.0, EYE_C[1] + sg * 86], [1004.0, EYE_C[1] + sg * PUPIL]])), d, 0.3, GRAPHITE, 3.0))
    st.append((X(arc(EYE_C, EYE_R * 0.94, -62, 62, 50)), 2.8, 0.9, (150, 70, 60), 3.4))
    src = np.array([OBJ_X + 6, 540.0])
    th = math.asin(50 / 98)
    for sg, d in ((-1, 3.8), (1, 3.9)):
        c_pt = np.array([1015 - 80 * math.cos(th), 540 + sg * 50])
        l_pt = LENS_P + [0, sg * 46]
        st.append((X(np.array([src, c_pt])), d, 0.55, RED, 2.6))
        st.append((X(np.array([c_pt, l_pt])), d + 0.55, 0.15, RED, 2.6))
        st.append((X(np.array([l_pt, FOVEA])), d + 0.7, 0.6, RED, 2.6))
    return st


class Eye(L.Section):
    name, dur = "eye", 11.0
    ZOOM = 7.0

    def cam(self, t):
        if t < self.ZOOM:
            return Cam(960, 540, 1.0)
        z1 = 10.0
        end = Cam(FOVEA[0], FOVEA[1] + (0.5 - R.FIELD_CY) * H / (S * z1), z1)
        return Cam.lerp(Cam(960, 540, 1.0), end, ease((t - self.ZOOM) / (self.dur - self.ZOOM)) ** 1.6)

    def frame(self, t):
        cam = self.cam(t)
        fr = L.finish(L.paper(cam), t)
        for pts, t0, d, col, wdt in eye_strokes(cam):
            f = (t - t0) / d
            if f > 0:
                fr = pencil(fr, [(pts, min(f, 1.0), int(t0 * 10))], col=col, width=wdt * cam.z ** 0.6)
        im = Image.fromarray(fr)
        a = fade(t, 1.0, self.ZOOM + 0.6, 0.4, 0.5)
        if a > 0 and cam.z < 1.5:
            L.text(ImageDraw.Draw(im), (cam.X(OBJ_X) / S, cam.Y(540) / S), "E", 60, INK, a, kind="serif", weight=700, anchor="mm")
            a2 = fade(t, 5.0, self.ZOOM + 0.6, 0.4, 0.5)
            note(im, (OBJ_X, 488), "the chart", a2, size=40, col=INK_SOFT, anchor="mb")
            note(im, (FOVEA[0] + 34, FOVEA[1] - 30), "fovea", a2, size=40, col=INK_SOFT)
            note(im, (1376, 744), "retina", a2, size=40, col=(150, 70, 60))
        caption(im, ["Your eye focuses the letter onto the retina, at the back,", "right in the centre, where you look."],
                fade(t, 0.6, self.ZOOM, 0.7, 0.5))
        fr = np.asarray(im)
        m = smooth((t - 8.6) / 2.0)
        if m > 0:
            fr = np.clip(fr * (1 - m) + R.microscope(fr.astype(np.float32)) * m, 0, 255).astype(np.uint8)
        return blackout(fr, 1 - smooth(t / 0.8))


# ================================================================== 5. on the retina, in pixel logic
K_RET = 120.0
TREES_BG = Y(SCENE["trees"]["bg"]) / BALL_Y              # the trees' light, as a share of the sunlit ball's (1/30)
C_TREES = 1 / TREES_BG - 1                               # the ball's Weber contrast against them (+29)
EXPO = 2.8                                               # the micrograph's exposure for the ball against the trees


@functools.lru_cache(None)
def ball_spot():
    """where to put the ball on the lattice: next to one cone, so that in pixel logic it covers four cone
    centres at 160 yards, one at 320 and none at 640 (checked against the jittered lattice)"""
    p, _ = R.lattice()
    best = None
    for i in np.argsort(np.hypot(p[:, 0] - 0.3, p[:, 1] - 0.2))[:12]:
        for ang in np.radians(np.arange(0, 360, 5)):
            for off in np.arange(0.30, 0.42, 0.01):
                c = p[i] + off * np.array([math.cos(ang), math.sin(ang)])
                dd = np.sort(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]))
                r160, r320, r640 = [float(P.ball_arcmin(y)) / 2 / I.U for y in (160, 320, 640)]
                if (dd < r160).sum() != 4 or (dd < r320).sum() != 1 or (dd < r640).sum() != 0:
                    continue
                margin = min(r160 - dd[3], dd[4] - r160, r320 - dd[0], dd[1] - r320, dd[0] - r640)
                if best is None or margin > best[0]:
                    best = (margin, c)
    return best[1]


BALL_C = ball_spot()


def ball_light(yd, sharp=1.0, aperture=True):
    """each cone's light (share of the sunlit ball's) for the ball against the trees, in pixel logic (sharp=1)
    or as the eye really forms it (sharp=0)"""
    def f(u, v):
        real = TREES_BG * (1 + C_TREES * I.ball_at(u, v, yd, aperture, *BALL_C))
        if sharp <= 0:
            return real
        on = I.ball_sharp_at(u, v, yd, *BALL_C)
        pix = TREES_BG + (1 - TREES_BG) * on
        return pix * sharp + real * (1 - sharp)
    return f


def ball_overlay(yd, sharp=1.0):
    """the ball's image itself (continuous), sharp or blurred"""
    rad = float(P.ball_arcmin(yd)) / 2 / I.U

    def f(u, v):
        r = np.hypot(u - BALL_C[0], v - BALL_C[1])
        sh = TREES_BG + (1 - TREES_BG) * np.clip((rad - r) * 12 + 0.5, 0, 1)
        if sharp >= 1:
            return sh
        real = TREES_BG * (1 + C_TREES * I.ball_at(u, v, yd, False, *BALL_C))
        return sh * sharp + real * (1 - sharp)
    return f


def yards_at(t, keys):
    """distance (yards) along keyframes [(t, yd)], easing geometrically between them"""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, y0), (t1, y1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0))
            return math.exp(math.log(y0) + (math.log(y1) - math.log(y0)) * u)
    return keys[-1][1]


class Retina(L.Section):
    """the E on the cones (pixel logic), the ball the size of a stroke, then the ball alone against the trees
    shrinking with distance: four cones, one, none"""
    name, dur = "retina", 62.0
    SWAP0, SWAP1 = 27.0, 29.0                          # the E gives way to the ball against the trees
    KEYS = [(36.0, 160), (38.4, 320), (44.6, 320), (47.0, 640)]

    def frame(self, t):
        sc = R.Scope(0.0, 0.0, K_RET)
        P_ = lambda u_, v_: (float(R.screen_x(sc, u_)), float(R.screen_y(sc, v_)))
        band = 0.32 * smooth((t - 0.8) / 2.0)
        sw = smooth((t - self.SWAP0) / (self.SWAP1 - self.SWAP0))
        yd = yards_at(t, self.KEYS)
        if sw < 1:
            fe = I.mosaic(sc, lambda u, v: I.e_sharp_at(u, v), overlay=lambda u, v: I.e_sharp_at(u, v), band=band, curve="print")
        if sw > 0:
            fb = I.mosaic(sc, ball_light(yd), overlay=ball_overlay(yd), band=0.32, exposure=EXPO)
        fr = fe if sw <= 0 else fb if sw >= 1 else mix(fe, fb, sw)
        fr = blackout(fr, 1 - smooth(t / 0.8))
        texts = []
        # the E: one stroke is two cones; the whole letter is ten
        a_e = fade(t, 9.4, self.SWAP0 + 0.4, 0.4, 0.6)
        if a_e > 0:
            x_b = P_(5.6, 0)[0]
            fr = pencil(fr, [(pen_bracket(x_b, P_(0, -5)[1], P_(0, -3)[1], side=1, tick=24), smooth((t - 9.4) / 0.6), 31),
                             (pen_bracket(P_(-5.9, 0)[0], P_(0, -5)[1], P_(0, 5)[1], side=-1, tick=24), smooth((t - 13.6) / 0.8), 32)],
                        col=RED_P, a=a_e, width=3.0)
            texts.append((W * R.FIELD_CX + H * R.FIELD_R + 30, P_(0, -4)[1], "1′: two cones", smooth((t - 9.8) / 0.5) * a_e, RED_L, "lm"))
            fr = pencil(fr, [([(x_b + 6, P_(0, -4)[1]), (W * R.FIELD_CX + H * R.FIELD_R + 18, P_(0, -4)[1])], smooth((t - 9.8) / 0.5), 34)],
                        col=RED_P, a=a_e, width=2.2)
            texts.append((W * R.FIELD_CX - H * R.FIELD_R - 30, P_(0, 0)[1], "5′: ten cones", smooth((t - 14.0) / 0.5) * a_e, RED_L, "rm"))
        # the ball at 160 yards, drawn over the E's middle arm
        a_c = fade(t, 18.4, self.SWAP0 + 0.4, 0.4, 0.6)
        if a_c > 0:
            r160 = float(P.ball_arcmin(160)) / 2 / I.U * sc.k
            cx, cy = P_(2.6, 0.0)
            fr = pencil(fr, [(circle_pts(cx, cy, r160), smooth((t - 18.4) / 0.9), 33)], col=RED_L, a=a_c, width=3.2)
            texts.append((cx, cy + r160 + 30, "the ball, 160 yd", smooth((t - 19.0) / 0.5) * a_c, RED_L, "mt"))
        # the ball's true outline, against the trees
        a_o = fade(t, self.SWAP1, self.dur, 0.6, 0.6)
        if a_o > 0:
            rr = float(P.ball_arcmin(yd)) / 2 / I.U * sc.k
            cx, cy = P_(*BALL_C)
            fr = pencil(fr, [(circle_pts(cx, cy, rr, n=90), 1.0, 41)], col=RED_L, a=a_o * 0.9, width=2.4)
            texts.append((cx + max(rr, 40) + 40, cy - 60, f"{int(round(yd / 10.0) * 10)} yards", a_o, RED_L, "lb"))
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for x_, y_, s_, a_, c_, an in texts:
            L.text(d, (x_ / S, y_ / S), s_, 40, c_, a_, kind="hand", anchor=an)
        caps = [(0.6, 8.6, ["Here is that 20/20 E on the cones at the centre of your retina, at the same scale."]),
                (9.2, 17.6, ["Each stroke is two cones wide. The whole letter spans about ten.",
                             "Suppose each cone works like a pixel: dark if the E covers its centre."]),
                (18.4, 26.8, ["A golf ball 160 yards away would fit inside one stroke."]),
                (29.4, 37.6, ["Now the ball alone, against dark trees. In pixel logic,",
                              "at 160 yards it lights the four cones whose centres it covers."]),
                (38.4, 46.2, ["At 320 yards, it’s one cone wide."]),
                (47.0, 54.6, ["At 640 yards, a quarter of a cone. It covers no cone’s centre.",
                              "If cones were pixels, you’d never see it."]),
                (55.2, 61.4, ["Yet against dark trees, a ball this far away is still easy to see."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=38)
        footnote(im, fade(t, 9.2, 26.8, 0.6, 0.6))
        return np.asarray(im)


# ================================================================== 6. the blur: no smaller, only fainter
BLUR_C = (W * 0.33, H * 0.45)
BLUR_R = H * 0.40
K_BLUR = 130.0
PLOT_X0, PLOT_X1 = 1250.0, 1800.0                        # the profile plots (1080p units)
PLOT_A0, PLOT_A1 = 140.0, 500.0                          # the extra light: top, axis
PLOT_B0, PLOT_B1 = 640.0, 820.0                          # the same, scaled to one height


@functools.lru_cache(64)
def spot_profile(yd):
    """the ball's image through a cone's aperture, change in light per unit contrast, against arcmin"""
    x = np.linspace(-3, 3, 241)
    return x, P.disk_image(np.abs(x), float(P.ball_arcmin(yd)), 1.0, I.APERTURE)


class Blur(L.Section):
    """the blur comes in; each cone's patch of the world; the ball recedes and only fades"""
    name, dur = "blur", 74.0
    BLUR0, BLUR1 = 3.0, 7.0                           # pixel logic to the real image
    PATCH = 18.0                                      # one cone's patch
    MOVE0, MOVE1 = 33.0, 36.0                         # the field slides left for the plot
    KEYS = [(0.0, 640), (28.6, 640), (31.0, 160), (40.0, 160), (44.0, 320), (49.0, 320), (53.0, 640), (58.0, 640), (62.0, 1280)]
    CURVES = ((160, 38.0), (320, 47.0), (640, 56.0), (1280, 63.0))      # each distance's curve appears
    NORM = 66.0

    def frame(self, t):
        u = ease((t - self.MOVE0) / (self.MOVE1 - self.MOVE0))
        cx = W * R.FIELD_CX + (BLUR_C[0] - W * R.FIELD_CX) * u
        cy = H * R.FIELD_CY + (BLUR_C[1] - H * R.FIELD_CY) * u
        rad = H * R.FIELD_R + (BLUR_R - H * R.FIELD_R) * u
        k = K_RET + (K_BLUR - K_RET) * u
        sc = R.Scope(BALL_C[0], BALL_C[1], k)
        X = lambda uu: (np.asarray(uu, np.float64) - sc.u) * sc.k + cx
        Yv = lambda vv: (np.asarray(vv, np.float64) - sc.v) * sc.k + cy
        yd = yards_at(t, self.KEYS)
        m = smooth((t - self.BLUR0) / (self.BLUR1 - self.BLUR0))
        glow = None
        a_p = fade(t, self.PATCH, self.MOVE0 - 0.6, 0.6, 0.8)
        p, _ = R.lattice()
        i0 = int(np.argmin(np.hypot(p[:, 0] - BALL_C[0], p[:, 1] - BALL_C[1])))
        if a_p > 0:
            glow = np.zeros(len(p), np.float32)
            glow[i0] = 0.55 * a_p
        fr = I.mosaic(sc, ball_light(yd, sharp=1 - m), overlay=ball_overlay(yd, sharp=1 - m), band=0.32, exposure=EXPO,
                      centre=(cx, cy), radius=rad, glow=glow)
        texts = []
        # one cone's patch of the world: a circle about one arcminute across, around the cone
        if a_p > 0:
            rp = 1.0 / 2 / I.U * sc.k                     # drawn at 1′ ("about 1′": the model's half-height width is 0.96′)
            fr = pencil(fr, [(circle_pts(X(p[i0, 0]), Yv(p[i0, 1]), rp, n=90), smooth((t - self.PATCH - 0.8) / 1.0), 51)],
                        col=CREAM_P, a=a_p, width=2.6)
            texts.append((X(p[i0, 0]) - rp - 30, Yv(p[i0, 1]) - rp - 10, "one cone’s patch: about 1′",
                          smooth((t - self.PATCH - 1.4) / 0.6) * a_p, CREAM_P, "rb"))
            a_b = a_p * smooth((t - 25.0) / 0.6)
            if a_b > 0:
                rb = float(P.ball_arcmin(160)) / 2 / I.U * sc.k
                fr = pencil(fr, [(circle_pts(X(p[i0, 0]), Yv(p[i0, 1]), rb, n=90, a0=2.0), smooth((t - 25.0) / 0.9), 52)],
                            col=RED_L, a=a_b, width=2.8)
                texts.append((X(p[i0, 0]) + rb + 30, Yv(p[i0, 1]) + rb + 10, "the ball at 160 yd", a_b, RED_L, "lt"))
        # the ball's distance, once the plot is up, and a 1′ ring around its spot
        a_d = fade(t, self.MOVE1, self.dur, 0.5, 0.8)
        if a_d > 0:
            fr = pencil(fr, [(circle_pts(X(BALL_C[0]), Yv(BALL_C[1]), 1.6 / I.U * sc.k, n=110, a0=1.0), 1.0, 55)],
                        col=CREAM_P, a=a_d * 0.55, width=2.0)
            texts.append((cx + rad * 0.74 + 20, cy + rad * 0.74 + 10, f"{int(round(yd / 10.0) * 10)} yards", a_d, CREAM_P, "lt"))
        # the plots: the extra light across the spot (through one cone's aperture) for each distance so far;
        # then the same curves scaled to one height
        a_pl = fade(t, self.MOVE1, self.dur, 0.6, 0.8)
        if a_pl > 0:
            peak = 0.2588
            PX_ = lambda x: (PLOT_X0 + (x + 3) / 6 * (PLOT_X1 - PLOT_X0)) * S
            PY_ = lambda v: (PLOT_A1 - v / peak * (PLOT_A1 - PLOT_A0)) * S
            axes = [([(PX_(-3), PY_(0)), (PX_(3), PY_(0))], smooth((t - self.MOVE1) / 0.8), 61),
                    ([(PX_(-3), PY_(0)), (PX_(-3), (PLOT_A0 - 20) * S)], smooth((t - self.MOVE1 - 0.3) / 0.8), 62)]
            fr = pencil(fr, axes, col=CREAM_P, a=a_pl, width=2.4)
            curves, leaders = [], []
            for j, (ydk, tk) in enumerate(self.CURVES):
                if t < tk:
                    continue
                x, pr = spot_profile(ydk)
                curves.append((np.c_[PX_(x), PY_(pr)], smooth((t - tk) / 1.2), 70 + j))
                lab = {160: "160 yd", 320: "320 yd: ¼ the light", 640: "640 yd: 1/16", 1280: "1,280 yd: 1/64"}[ydk]
                al = smooth((t - tk - 0.8) / 0.5) * a_pl
                if ydk <= 320:
                    texts.append((PX_(0.22), PY_(pr.max()) - 6, lab, al, RED_L, "lb"))
                else:
                    ly = PY_(0) - {640: 210, 1280: 110}[ydk]
                    xa = {640: 0.35, 1280: 0.75}[ydk]
                    leaders.append(([(PX_(1.55), ly), (PX_(xa), float(PY_(np.interp(xa, x, pr))) - 6)], smooth((t - tk - 0.6) / 0.5), 75 + j))
                    texts.append((PX_(1.62), ly, lab, al, RED_L, "lm"))
            fr = pencil(fr, curves, col=RED_L, a=a_pl, width=3.0)
            fr = pencil(fr, leaders, col=RED_L, a=a_pl * 0.8, width=1.8)
            texts.append((PX_(-3) - 14, (PLOT_A0 - 20) * S, "extra light", a_pl, CREAM_P, "rt"))
            texts.append((PX_(3), PY_(0) + 12, "across the spot", a_pl, CREAM_P, "rt"))
            # scaled to one height: the same spot, about 1′ wide
            a_n = a_pl * smooth((t - self.NORM) / 0.8)
            if a_n > 0:
                QY = lambda v: (PLOT_B1 - v * (PLOT_B1 - PLOT_B0)) * S
                st = [([(PX_(-3), QY(0)), (PX_(3), QY(0))], smooth((t - self.NORM) / 0.6), 90)]
                for j, (ydk, _) in enumerate(self.CURVES):
                    x, pr = spot_profile(ydk)
                    st.append((np.c_[PX_(x), QY(pr / pr.max())], smooth((t - self.NORM - 0.4 - 0.3 * j) / 1.0), 91 + j))
                fr = pencil(fr, st, col=CREAM_P, a=a_n, width=2.0)
                half = 0.5
                yb = QY(0.5)
                fr = pencil(fr, [([(PX_(-half) - 70, yb), (PX_(-half), yb)], smooth((t - self.NORM - 2.0) / 0.5), 96),
                                 ([(PX_(half), yb), (PX_(half) + 70, yb)], smooth((t - self.NORM - 2.0) / 0.5), 97)],
                            col=RED_L, a=a_n, width=2.4)
                texts.append((PX_(half) + 84, yb, "about 1′", a_n * smooth((t - self.NORM - 2.3) / 0.5), RED_L, "lm"))
                texts.append((PX_(-3), (PLOT_B0 - 30) * S, "the same curves, scaled to one height", a_n, CREAM_P, "lb"))
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for x_, y_, s_, a_, c_, an in texts:
            L.text(d, (x_ / S, y_ / S), s_, 38, c_, a_, kind="hand", anchor=an)
        caps = [(0.6, 9.6, ["So how? First, your eye isn’t a perfect lens.",
                            "It blurs every point of light into a small spot, about an arcminute wide."]),
                (10.2, 17.4, ["And a cone isn’t an on-off pixel. It reports how much light reaches it.",
                              "The ball lights a small patch of cones, a little."]),
                (18.2, 24.6, ["Turn it around: each cone collects light from a patch of the world",
                              "about one arcminute wide: its own width, spread by the blur."]),
                (25.4, 32.6, ["At 160 yards, the ball is as wide as one cone’s patch.", "Any farther, and the whole ball fits inside it."]),
                (37.0, 52.6, ["From there on, the ball keeps shrinking, but its image hardly does.",
                              "The image is the blur spot itself."]),
                (53.4, 66.0, ["It only gets fainter. Twice as far, a quarter of the light."]),
                (66.8, 73.4, ["It doesn’t get smaller. It gets fainter."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=38)
        return blackout(np.asarray(im), 1 - smooth(t / 0.6))


# ================================================================== 7. the recipe (spatial frequencies)
RX0, RX1 = 330.0, 1430.0                                 # the plot (1080p units): 1 to 100 cycles per degree
RY0, RY1 = 150.0, 690.0                                  # top, axis: four decades of strength
DECADES = 4.0


def rx(cpd):
    return (RX0 + np.log10(np.asarray(cpd, np.float64)) / 2 * (RX1 - RX0)) * S


def ry(logs):
    return (RY0 + np.clip(-np.asarray(logs, np.float64), 0, DECADES) / DECADES * (RY1 - RY0)) * S


@functools.lru_cache(None)
def recipe(yd):
    """the ball's spectrum: strength (relative to the 40-yard ball's at 0) against cycles per degree"""
    f = np.logspace(0, 2, 600)
    a40 = float(P.ball_arcmin(40)) ** 2
    amp = float(P.ball_arcmin(yd)) ** 2 / a40 * np.abs(P.jinc(float(P.ball_arcmin(yd)) * f / 60))
    return f, np.log10(np.maximum(amp, 1e-9))


class Recipe(L.Section):
    """the ball's spatial-frequency recipe against the range the eye can see: near, it has a shape there;
    far, it is flat there and only drops"""
    name, dur = "recipe", 54.0
    CURVES = [(40, 8.0), (80, 12.0), (160, 16.0), (320, 26.0), (640, 30.0)]
    SHIFT = 43.0                                         # the 640-yard curve lifted ×4 onto the 320-yard one

    def frame(self, t):
        cam = Cam(960, 540, 1.0)
        fr = L.paper(cam)
        # what the eye can see: a pale wash up to the analyzer CSF's cut-off
        cut = P.csf_cutoff()
        a_w = smooth((t - 3.0) / 1.0)
        if a_w > 0:
            x0, x1 = int(rx(1)), int(rx(cut))
            y0, y1 = int(RY0 * S), int(RY1 * S)
            reg = fr[y0:y1, x0:x1].astype(np.float32)
            fr[y0:y1, x0:x1] = (reg * (1 - 0.30 * a_w) + np.array([196, 184, 160], np.float32) * 0.30 * a_w).astype(np.uint8)
        fr = L.finish(fr, t)
        ax = [([(rx(1), RY1 * S), (rx(100), RY1 * S)], smooth((t - 0.6) / 0.8), 101),
              ([(rx(1), RY1 * S), (rx(1), (RY0 - 20) * S)], smooth((t - 0.9) / 0.8), 102)]
        ax += [([(rx(f), RY1 * S), (rx(f), (RY1 + 12) * S)], smooth((t - 1.4) / 0.4), 103 + k) for k, f in enumerate((1, 3, 10, 30, 100))]
        ax += [([(rx(cut), RY0 * S), (rx(cut), RY1 * S)], smooth((t - 3.2) / 0.8), 110)]
        fr = pencil(fr, ax, col=GRAPHITE, width=2.6)
        strokes = []
        for j, (yd, t0) in enumerate(self.CURVES):
            if t < t0:
                continue
            f, lg = recipe(yd)
            col = RED if yd in (320, 640) else GRAPHITE
            strokes.append((np.c_[rx(f), ry(lg)], smooth((t - t0) / 1.6), 120 + j, col))
        for pts, f_, seed, col in strokes:
            fr = pencil(fr, [(pts, f_, seed)], col=col, width=2.8)
        # the 640-yard curve, four times stronger, lands on the 320-yard curve
        a_s = fade(t, self.SHIFT, self.dur, 0.6, 0.6)
        if a_s > 0:
            f, lg = recipe(640)
            lift = math.log10(4) * smooth((t - self.SHIFT - 0.6) / 2.0)
            m = f <= cut
            fr = pencil(fr, [(np.c_[rx(f[m]), ry(lg[m] + lift)], 1.0, 140)], col=RED_L, a=a_s * 0.8, width=2.2)
            xa = rx(2.0)
            fr = pencil(fr, arrow_strokes((xa, ry(recipe(640)[1][0]) - 8), (xa, ry(recipe(320)[1][0]) + 10),
                                          smooth((t - self.SHIFT) / 0.6), 141, head=14), col=RED, a=a_s, width=2.4)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for k, fq in enumerate((1, 3, 10, 30, 100)):
            L.text(d, (rx(fq) / S, RY1 + 22), str(fq), 32, INK_SOFT, smooth((t - 1.6) / 0.5), kind="hand", anchor="mt")
        L.text(d, ((rx(1) + rx(100)) / 2 / S, RY1 + 64), "finer ripples (cycles per degree)", 34, INK_SOFT, smooth((t - 1.8) / 0.6),
               kind="hand", anchor="mt")
        L.text(d, (RX0 - 20, RY0 - 20), "strength", 34, INK_SOFT, smooth((t - 1.8) / 0.6), kind="hand", anchor="rt")
        a_c = smooth((t - 3.6) / 0.6)
        L.text(d, ((rx(1) + rx(cut)) / 2 / S, RY0 - 30), "what your eye can see", 36, INK_SOFT, a_c, kind="hand", anchor="ms")
        L.text(d, (rx(cut) / S + 10, RY0 + 6), f"≈ {cut:.0f} cycles per degree", 30, INK_SOFT, a_c, kind="hand", anchor="lt")
        for j, (yd, t0) in enumerate(self.CURVES):
            if t < t0:
                continue
            f, lg = recipe(yd)
            col = RED if yd in (320, 640) else INK_SOFT
            L.text(d, (rx(1.12) / S, ry(lg[0]) / S - 8), f"{yd} yd", 34, col, smooth((t - t0 - 1.0) / 0.5), kind="hand", anchor="lb")
        if a_s > 0:
            L.text(d, (rx(2.0) / S + 14, (ry(recipe(640)[1][0]) + ry(recipe(320)[1][0])) / 2 / S), "× 4", 36, RED,
                   a_s * smooth((t - self.SHIFT - 0.4) / 0.5), kind="hand", anchor="lm")
        caps = [(0.8, 7.6, ["Every image is a recipe of ripples of light: coarse ones and fine ones.",
                            "Here is the golf ball’s recipe, ripple by ripple."]),
                (8.2, 15.4, ["Up close, the recipe has a shape inside the range your eye can see.",
                             "That shape is what tells you the ball’s size."]),
                (16.2, 25.4, ["At 160 yards, the shape is slipping past the finest ripples you can see."]),
                (26.2, 33.6, ["By 320 yards, the recipe is flat across everything your eye can see."]),
                (34.4, 42.4, ["Farther away, it stays flat. It only drops: twice as far, a quarter as strong."]),
                (43.2, 53.4, ["To your eye, a ball at 640 yards is the same recipe as one at 320, a quarter as strong.",
                              "Nothing in it says how big the ball is. Only how bright."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6))
        return blackout(np.asarray(im), 1 - smooth(t / 0.8))


# ================================================================== 8. why a dark background
DARK_YD = 400
K_DARK = 74.0
DARK_FIELDS = [("trees", W * 0.19), ("sky", W * 0.5), ("overcast", W * 0.81)]
DARK_CY, DARK_R = H * 0.40, H * 0.255


def scene_contrast(key):
    sc_ = SCENE[key]
    return (Y(sc_["ball"]) - Y(sc_["bg"])) / Y(sc_["bg"])


@functools.lru_cache(None)
def dark_field(key):
    """the ball at 400 yards on one background, with the exposure set by the background (as the eye's own
    adaptation does): the background reads the same in all three, and what's left is the change"""
    C = scene_contrast(key)
    sc = R.Scope(BALL_C[0], BALL_C[1], K_DARK)
    cx = dict(DARK_FIELDS)[key]
    light = lambda u, v: 0.16 * (1 + C * I.ball_at(u, v, DARK_YD, True, *BALL_C))
    return I.mosaic(sc, light, exposure=1.0, centre=(cx, DARK_CY), radius=DARK_R)


def dark_fields():
    """all three fields in one frame"""
    out = np.full((H, W, 3), R.OUTSIDE.astype(np.uint8), np.uint8)
    for key, cx in DARK_FIELDS:
        f = dark_field(key)
        x0, x1 = int(cx - DARK_R - 4), int(cx + DARK_R + 4)
        out[:, x0:x1] = f[:, x0:x1]
    return out


BARS = [("overcast sky", -0.45, "overcast", None), ("a black ball, any sky", -1.0, None, None),
        ("blue sky", 1.5, "sky", None), ("the fairway", 5.0, "fairway", None), ("dark trees", 29.0, "trees", None)]


@functools.lru_cache(None)
def bar_yards():
    """the farthest a still ball is seen against each background, on a clear day (physics.py)"""
    out = []
    for name, bg, ball in P.SCENES:
        out.append((name, P.weber(ball, bg), round(P.max_yards_haze(ball, bg, "sky" in name) / 10) * 10))
    out.append(("a black ball, any sky", -1.0, round(P.max_yards_haze(0.0, 6000.0, True) / 10) * 10))
    return sorted(out, key=lambda r: r[2])


class Dark(L.Section):
    name, dur = "dark", 76.0
    PAGE = 31.0                                          # the fields give way to the notebook page

    def frame(self, t):
        if t < self.PAGE + 1.0:
            fr = dark_fields()
            fr = blackout(fr, 1 - smooth(t / 0.8))
            im = Image.fromarray(fr)
            d = ImageDraw.Draw(im)
            a_l = fade(t, 1.0, self.PAGE + 0.5, 0.6, 0.6)
            for (key, cx), lab in zip(DARK_FIELDS, ("dark trees", "blue sky", "overcast sky")):
                C = {"trees": 29.0, "sky": 1.5, "overcast": -0.45}[key]          # physics.SCENES
                L.text(d, (cx / S, (DARK_CY + DARK_R + 50) / S), lab, 40, CREAM_P, a_l, kind="hand", anchor="mt")
                pct = f"{C * 100:+,.0f}%".replace("-", "−")
                L.text(d, (cx / S, (DARK_CY + DARK_R + 160) / S), f"the ball: {pct}", 40, RED_L,
                       a_l * smooth((t - {"trees": 10.6, "sky": 13.0, "overcast": 21.0}[key]) / 0.6), kind="hand", anchor="mt")
            caps = [(0.8, 9.8, ["Your eye sets its exposure by the background, like a camera.",
                                "What it notices is change: brighter or darker than the surroundings, as a share of them."]),
                    (10.4, 20.2, ["Here’s the ball at 400 yards, each with the background at the same exposure.",
                                  "Against dark trees, the sunlit ball is 30 times as bright as what’s behind it."]),
                    (20.8, 30.6, ["Against an overcast sky, it’s darker than the sky.",
                                  "And a dark speck can never be more than 100% darker. A bright one can be thousands of percent brighter."])]
            for c0, c1, lines in caps:
                caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=36)
            fr = np.asarray(im)
            if t <= self.PAGE:
                return fr
            page = self.page(t)
            return mix(fr, page, smooth((t - self.PAGE) / 1.0))
        return self.page(t)

    def page(self, t):
        tp = t - self.PAGE
        cam = Cam(960, 540, 1.0)
        fr = L.finish(L.paper(cam), t)
        texts = []
        # the rule
        fr = pencil(fr, [([(150 * S, 168 * S), (560 * S, 172 * S)], smooth((tp - 1.2) / 0.6), 150)], col=GRAPHITE, width=2.6)
        # the bars: how far a still ball is seen
        rows = bar_yards()
        X0, PER = 560.0, 1.1                              # 1080p units per yard
        y = 330.0
        strokes = []
        for k, (name, C, yd) in enumerate(rows):
            t0 = 12.0 + 1.6 * k
            f = smooth((tp - t0) / 0.9)
            col = RED if C > 0 else GRAPHITE
            if f > 0:
                strokes.append(([(X0 * S, y * S), ((X0 + yd * PER * f) * S, y * S)], 1.0, 160 + k, col))
                strokes.append(([(X0 * S, (y - 14) * S), (X0 * S, (y + 14) * S)], 1.0, 170 + k, col))
                pct = f"{C * 100:+,.0f}%".replace("-", "−")
                texts.append(((X0 - 30, y), f"{name}  {pct}", f, INK_SOFT, "rm", 36))
                texts.append(((X0 + yd * PER + 22, y), f"{yd:,} yd", smooth((tp - t0 - 0.7) / 0.5), col, "lm", 38))
            y += 74
        for pts, f_, seed, col in strokes:
            fr = pencil(fr, [(pts, f_, seed)], col=col, width=5.0)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        L.text(d, (150, 160), "contrast × area ≥ about one cone", 54, INK, smooth((tp - 0.4) / 0.8), kind="hand", anchor="ls")
        L.text(d, (150, 228), "how far you could see a ball sitting still, on a clear day, if you knew where to look:", 32,
               INK_SOFT, smooth((tp - 10.6) / 0.8), kind="hand", anchor="ls")
        for xy, s_, a_, c_, an, sz in texts:
            L.text(d, xy, s_, sz, c_, a_, kind="hand", anchor=an)
        caps = [(0.8, 10.2, ["Measured thresholds give a simple rule: you can find a small spot",
                             "as long as its contrast times its area adds up to about one cone."]),
                (10.8, 21.6, ["A black ball against a bright sky is lost at about 310 yards, close to where it’s one cone wide.",
                              "A white ball against dark trees, thirty times brighter, goes about three times farther."]),
                (22.4, 33.4, ["That’s why a drive can vanish against an overcast sky,", "then turn up again on the fairway."]),
                (34.2, 44.4, ["And it’s why you can see stars. A star is thousands of times narrower than a cone,",
                              "but the night sky behind it is almost black."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(tp, c0, c1, 0.6, 0.6))
        return np.asarray(im)


# ================================================================== 9. try it: four scenes, four distances
DEMO_YDS = (100, 250, 500, 800)
DEMO_X = (560.0, 820.0, 1100.0, 1360.0)
DEMO_KEYS = ("trees", "fairway", "sky", "overcast")
DEMO_GAP = 820.0                                         # the cards' spacing down the desk
DEMO_CARDS = [Swatch(k, [(x, 540.0 + DEMO_GAP * j, yd) for x, yd in zip(DEMO_X, DEMO_YDS)], y0=330 + DEMO_GAP * j,
                     y1=750 + DEMO_GAP * j, label=SCENE[k]["name"]) for j, k in enumerate(DEMO_KEYS)]
DEMO_HOME = [Cam(960, 560 + DEMO_GAP * j, 1.0) for j in range(4)]


@functools.lru_cache(None)
def demo_seen():
    """for each card, whether a still ball at each distance is seen on a real sunny course (the clear-day
    limits of the bar chart)"""
    lim = {name: yd for name, _, yd in bar_yards()}
    return [[yd < lim[SCENE[k]["name"]] for yd in DEMO_YDS] for k in DEMO_KEYS]


class Demo(L.Section):
    name = "demo"
    PAN, HOLD = 2.6, 10.0
    MARKS = 3.6                                          # the ticks and crosses come in, card by card
    CAPS = {0: ["Golf balls at 100, 250, 500 and 800 yards against dark trees, drawn to scale.",
                "On a sunny course, a ball sitting still would be seen at all four."],
            1: ["On the fairway: the first three."],
            2: ["Against a blue sky: the first two."],
            3: ["Against an overcast sky: only the closest.",
                "Your screen is far dimmer than a sunny day, so the faintest ones are harder to see here."]}

    def __init__(self):
        self.dur = 4 * self.HOLD + 3 * self.PAN + 2.0

    def frame(self, t):
        seg = self.HOLD + self.PAN
        j = min(3, int(t // seg))
        u = t - j * seg
        if u > self.HOLD and j < 3:
            cam = Cam.lerp(DEMO_HOME[j], DEMO_HOME[j + 1], ease((u - self.HOLD) / self.PAN))
            v = None
        else:
            cam = DEMO_HOME[j]
            v = u
        fr = desk(cam, DEMO_CARDS, t=t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for k in range(4):
            c = DEMO_CARDS[k]
            if cam.Y(c.y1) < -100 or cam.Y(c.y0) > H + 100:
                continue
            for x, yd in zip(DEMO_X, DEMO_YDS):
                L.text(d, (cam.X(x) / S, cam.Y(c.y1 + 30) / S), f"{yd:,} yd", 34, INK_SOFT, 1.0, kind="hand", anchor="mt")
        if v is not None:
            end = self.HOLD - 0.3 if j < 3 else self.dur - j * seg - 0.4
            caption(im, self.CAPS[j], fade(v, 0.5, end, 0.6, 0.6))
        footnote(im, fade(t, 0.5, self.dur - 0.4, 0.6, 0.6))
        fr = np.asarray(im).copy()
        # pencil ticks under each ball, on the paper below the card; then, for a real sunny course, a tick
        # (seen) or a cross (lost) under each distance
        seen = demo_seen()
        for k in range(4):
            c = DEMO_CARDS[k]
            if cam.Y(c.y1) < -100 or cam.Y(c.y0) > H + 100:
                continue
            fr = pencil(fr, [([(cam.X(x), cam.Y(c.y1 + 26)), (cam.X(x), cam.Y(c.y1 + 8))], 1.0, 200 + 10 * k + q)
                             for q, x in enumerate(DEMO_X)], col=GRAPHITE, width=2.4)
            tk = k * seg + self.MARKS                     # this card's marks, once it's in view
            for q, x in enumerate(DEMO_X):
                f = smooth((t - tk - 0.45 * q) / 0.5)
                if f <= 0:
                    continue
                y = c.y1 + 96
                P_ = lambda dx, dy: (cam.X(x + dx), cam.Y(y + dy))
                if seen[k][q]:
                    fr = pencil(fr, [([P_(-13, -2), P_(-3, 10), P_(16, -16)], f, 300 + 10 * k + q)], col=RED, width=3.2)
                else:
                    fr = pencil(fr, [([P_(-11, -11), P_(11, 11)], f * 2, 340 + 10 * k + q),
                                     ([P_(11, -11), P_(-11, 11)], f * 2 - 1, 380 + 10 * k + q)], col=GRAPHITE, width=3.0)
        return blackout(fr, 1 - smooth(t / 0.8))


# ================================================================== 10. close
class Close(L.Section):
    name, dur = "close", 19.0

    def frame(self, t):
        cam = Cam.lerp(OPEN_HOME, Cam(960, 540, 0.6), ease((t - 8.0) / 9.0))
        fr = desk(cam, [OPEN_CARD], t=t)
        im = Image.fromarray(fr)
        caption(im, ["Past 160 yards, a golf ball is smaller than the finest detail 20/20 vision can resolve."], fade(t, 0.8, 8.0, 0.6, 0.6))
        caption(im, ["You don’t need to resolve it. You only need one blur spot", "to be a little brighter than the ones around it."],
                fade(t, 8.6, 16.4, 0.6, 0.6))
        return blackout(np.asarray(im), smooth((t - 16.6) / 2.4) + (1 - smooth(t / 0.8)))


SECTIONS = {"open": Open, "angle": Angle, "chart": ChartSec, "eye": Eye, "retina": Retina, "blur": Blur, "recipe": Recipe,
            "dark": Dark, "demo": Demo, "close": Close}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s_ in sys.argv[2:]:
            sec.render(float(s_))
    else:
        sec.render()
