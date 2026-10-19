"""Seeing smaller than your own pixels: the vernier film, section by section (4K, 30 fps).

    python film.py NAME [still times...]
    NAME: open, eye, retina, back, smaller, how, screen, numbers, uses, close
"""
import os, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L
from look import W, H, S, Cam, Card, smooth, ease, fade, caption, note, pencil, INK, INK_SOFT, RED, CREAM
import retina as R
import tv as T

GRAPHITE = (70, 66, 62)
PX_ARCSEC = R.PX_ARCSEC
CARDS = [Card(y0=330 + 900 * k, y1=750 + 900 * k, line_y=540 + 900 * k, step=st, label=f"Plate {k + 1}")
         for k, st in enumerate((0.5, 0.25, 0.125, 0.0))]
HOME = [Cam(960, 540 + 900 * k, 1.0) for k in range(4)]   # each card at its real size (exactly 1 unit = 2 px)


def blackout(fr, a):
    """fade to or from black (a = 1: black)"""
    if a <= 0:
        return fr
    return (fr.astype(np.float32) * (1 - a)).astype(np.uint8)


def mix(a, b, u):
    if u <= 0:
        return a
    if u >= 1:
        return b
    return cv2.addWeighted(a, 1 - u, b, u, 0)


# ------------------------------------------------------------------ the desk
def desk(cam, cards=CARDS, lens=None, ticks=None, t=0.0, plates=1.0):
    """the desk under cam. lens: (world x, world y, radius in units, magnification, alpha) of the hand
    magnifier, placed over that world point. ticks: (card index, drawn fraction) for two pencil marks
    pointing at the step from above and below."""
    fr = L.paper(cam)
    for c in cards:
        if cam.Y(c.y1) > -200 and cam.Y(c.y0) < H + 200:
            c.draw(fr, cam)
    keep = None
    for c in cards:
        if cam.Y(c.y1) > 0 and cam.Y(c.y0) < H:
            keep = c.line_box(cam)
    if ticks is not None:
        k, f = ticks
        c = cards[k]
        x = c.xm
        y = c.line_y
        fr = pencil(fr, [([(cam.X(x), cam.Y(y - 70)), (cam.X(x), cam.Y(y - 34))], f * 2, 31 + k),
                         ([(cam.X(x), cam.Y(y + 36 + c.step + c.th)), (cam.X(x), cam.Y(y + 72 + c.step + c.th))], f * 2 - 1, 37 + k)],
                    col=GRAPHITE, a=1.0, width=3.0)
    if lens is not None and lens[4] > 0:
        lx, ly, rad, mag, a = lens
        X, Y, Rp = cam.X(lx), cam.Y(ly), int(rad * S * cam.z)
        # the magnified view: world point (lx, ly) lands at the lens's centre on screen
        zm = cam.z * mag
        icam = Cam(lx - (X - W / 2) / (S * zm), ly - (Y - H / 2) / (S * zm), zm)
        box = (int(X) - Rp, int(Y) - Rp, int(X) + Rp, int(Y) + Rp)
        inner = L.paper(icam, box)
        for c in cards:
            c.draw(inner, icam, box)
        fr = L.loupe(fr, inner, X, Y, Rp, a)
    fr = L.finish(fr, t, keep=keep)
    im = Image.fromarray(fr)
    for c in cards:
        if cam.Y(c.y1) > 0 and cam.Y(c.y0) < H:
            c.plate(im, cam, plates)
    return np.asarray(im)


def lens_path(t, t_in, t_out, card, dur=1.6, mag=7.0, rad=150):
    """the magnifier slides in from the lower right, rests over the step, and slides away"""
    cx, cy = card.xm, card.line_y + card.step / 2
    off = np.array([760.0, 520.0])
    if t < t_in or t > t_out + dur:
        return None
    u = ease((t - t_in) / dur) if t < t_in + dur else 1 - ease((t - t_out) / dur) if t > t_out else 1.0
    x, y = np.array([cx, cy]) + off * (1 - u)
    return (x, y, rad, mag, 1.0)


def pen_bracket(x, y0, y1, side=1, tick=18):
    """a vertical bracket (screen px) from y0 to y1 at x, ticks pointing toward -side"""
    return [(x - side * tick, y0), (x, y0), (x, y1), (x - side * tick, y1)]


# ================================================================== 1. the line, and a magnifier
class Open(L.Section):
    name, dur = "open", 13.0
    LENS_IN, LENS_OUT = 1.6, 9.6

    def frame(self, t):
        c = CARDS[0]
        cam = HOME[0]
        if t > 10.4:                                             # push in toward the step, into the eye
            u = ease((t - 10.4) / 2.6)
            cam = Cam.lerp(HOME[0], Cam(c.xm, c.line_y, 4.5), u)
        lens = lens_path(t, self.LENS_IN, self.LENS_OUT, c)
        fr = desk(cam, lens=lens, t=t)
        im = Image.fromarray(fr)
        caption(im, ["There’s a step in the middle of this line."], fade(t, 3.4, 9.8, 0.7, 0.6))
        a_n = fade(t, 4.6, 9.8, 0.4, 0.6)
        if a_n > 0:
            note(im, (1254, 286), "half a pixel", a_n, size=44)
        fr = np.asarray(im).copy()
        if a_n > 0:
            f = smooth((t - 4.6) / 0.8)
            tip = np.array([1104.0, 428.0])
            back = np.array([0.77, -0.64])
            heads = []
            for k, sg in enumerate((1, -1)):
                c_, s_ = math.cos(math.radians(32 * sg)), math.sin(math.radians(32 * sg))
                end = tip + 17 * np.array([back[0] * c_ - back[1] * s_, back[0] * s_ + back[1] * c_])
                heads.append(([(cam.X(tip[0]), cam.Y(tip[1])), (cam.X(end[0]), cam.Y(end[1]))], (f - 0.85) / 0.15, 6 + k))
            fr = pencil(fr, [([(cam.X(1244), cam.Y(318)), (cam.X(1180), cam.Y(365)), (cam.X(tip[0]), cam.Y(tip[1]))], f, 4)]
                        + heads, a=a_n)
        return blackout(fr, 1 - smooth(t / 1.2))


# ================================================================== 2. into the eye
EYE_C = np.array([1210.0, 540.0])
EYE_R = 250.0
FOVEA = EYE_C + np.array([EYE_R * 0.94, 0.0])
SCREEN_X = 420.0
LENS = np.array([1035.0, 540.0])
PUPIL = 54.0


def arc(c, r, a0, a1, n=60, rx=None, ry=None):
    th = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + (rx or r) * np.cos(th), c[1] + (ry or r) * np.sin(th)]


def eye_strokes(cam):
    """the sketch, as (points in screen px, start time, duration, colour, width)"""
    X = lambda p: np.c_[cam.X(p[:, 0]), cam.Y(p[:, 1])]
    st = []
    st.append((X(arc(EYE_C, EYE_R, 200, 520, 120)), 0.3, 1.8, GRAPHITE, 3.0))                     # the eyeball
    st.append((X(arc(EYE_C + [-195.0, 0], 0, 120, 240, 30, rx=80, ry=98)), 1.6, 0.6, GRAPHITE, 3.0))   # cornea
    st.append((X(arc(LENS, 0, 0, 360, 50, rx=26, ry=80)), 2.0, 0.7, GRAPHITE, 2.6))                 # lens
    for sg, d in ((-1, 2.5), (1, 2.6)):                                                              # iris
        st.append((X(np.array([[990.0, EYE_C[1] + sg * 86], [1004.0, EYE_C[1] + sg * PUPIL]])), d, 0.3, GRAPHITE, 3.0))
    st.append((X(arc(EYE_C, EYE_R * 0.94, -62, 62, 50)), 2.8, 0.9, (150, 70, 60), 3.4))           # retina
    st.append((X(np.array([[SCREEN_X, 430.0], [SCREEN_X, 650.0]])), 3.2, 0.4, GRAPHITE, 4.0))       # the screen, edge on
    # the rays bend most at the cornea and a little more at the lens
    src = np.array([SCREEN_X + 6, 540.0])
    th = math.asin(50 / 98)
    for sg, d in ((-1, 3.8), (1, 3.9)):
        c_pt = np.array([1015 - 80 * math.cos(th), 540 + sg * 50])
        l_pt = LENS + [0, sg * 46]
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
        strokes = eye_strokes(cam)
        for pts, t0, d, col, wdt in strokes:
            f = (t - t0) / d
            if f > 0:
                fr = pencil(fr, [(pts, min(f, 1.0), int(t0 * 10))], col=col, width=wdt * cam.z ** 0.6)
        im = Image.fromarray(fr)
        a = fade(t, 5.0, self.ZOOM + 0.6, 0.4, 0.5)
        if a > 0 and cam.z < 1.5:
            note(im, (SCREEN_X, 418), "screen", a, size=40, col=INK_SOFT, anchor="mb")
            note(im, (SCREEN_X - 16, 536), "the line", a, size=40, col=INK_SOFT, anchor="rm")
            note(im, (FOVEA[0] + 34, FOVEA[1] - 30), "fovea", a, size=40, col=INK_SOFT)
            note(im, (1376, 744), "retina", a, size=40, col=(150, 70, 60))
        caption(im, ["Your eye focuses the line onto the retina, at the back,", "right in the centre, where you look."],
                fade(t, 0.6, self.ZOOM, 0.7, 0.5))
        fr = np.asarray(im)
        m = smooth((t - 8.6) / 2.0)                              # the microscope's field closes in
        if m > 0:
            fr = np.clip(fr * (1 - m) + R.microscope(fr.astype(np.float32)) * m, 0, 255).astype(np.uint8)
        return fr


# ================================================================== 3. the line on the retina
K_RET = 120.0                     # screen px per cone spacing
STEP_HALF = R.step_cones(0.5)


def _mark_cell(near=(1.4, -3.1)):
    p, _ = R.lattice()
    i = int(np.argmin(np.hypot(p[:, 0] - near[0], p[:, 1] - near[1])))
    return p[i], p[i] + (R.cells()[i] - p[i]) / 0.91         # the whole cell, out to the middle of the membrane


MARK_P, MARK_CELL = _mark_cell()


def cone_um(c):
    return c * R.CONE_ARCMIN * R.UM_PER_ARCMIN


class Retina(L.Section):
    name, dur = "retina", 50.0
    SPLIT0, SPLIT1 = 29.0, 31.5

    def frame(self, t):
        sc = R.Scope(0.0, STEP_HALF / 2, K_RET)
        band = smooth((t - 0.8) / 2.0)
        lit = smooth((t - 7.0) / 2.5)
        bnd = band * (1 - 0.7 * lit)
        u = smooth((t - self.SPLIT0) / (self.SPLIT1 - self.SPLIT0))
        if u <= 0:
            fr = R.mosaic(sc, y0=0.0, step=STEP_HALF, lit=lit, band=bnd)
        else:
            cx = W * (0.5 - 0.21 * u)
            rad = H * (R.FIELD_R - 0.085 * u)
            k = K_RET * (1 - 0.12 * u)
            left = R.mosaic(R.Scope(0.0, STEP_HALF / 2, k), y0=0.0, step=STEP_HALF, lit=lit, band=bnd,
                            centre=(cx, H * R.FIELD_CY), radius=rad)
            right = T.macro(0.0, STEP_HALF / 2 + 1.365, k, (W * 0.71, H * R.FIELD_CY), H * 0.34, y0=0.0, th=2.73, step=STEP_HALF)
            # the screen's field fades in over the retina's, which slides left out of its way
            _, inside = R.field_mask(W * 0.71, H * R.FIELD_CY, H * 0.34)
            a = (inside * u)[..., None]
            fr = np.clip(left * (1 - a) + right * a, 0, 255).astype(np.uint8)
        # red pencil: the middle of the line, with its step, and one cone traced beside it
        a_b = fade(t, 16.5, self.SPLIT0 + 0.3, 0.3, 0.6)
        P = lambda u_, v_: (float(R.screen_x(sc, u_)), float(R.screen_y(sc, v_)))
        if a_b > 0:
            trace = [P(-6.5, 0.0), P(-0.02, 0.0), P(0.02, STEP_HALF), P(6.5, STEP_HALF)]
            trace = np.concatenate([np.linspace(trace[i], trace[i + 1], 40) for i in range(3)])
            fr = pencil(fr, [(trace, smooth((t - 16.5) / 1.6), 51),
                             ([P(*q) for q in np.vstack([MARK_CELL, MARK_CELL[:1]])], smooth((t - 19.6) / 1.0), 53)],
                        col=(222, 84, 62), a=a_b, width=3.0)
        im = Image.fromarray(fr)
        if a_b > 0:
            x_, y_ = P(0.3, STEP_HALF + 0.22)
            note(im, (x_ / S, y_ / S), f"the step: {cone_um(STEP_HALF):.1f} µm", a_b * smooth((t - 17.6) / 0.6),
                 size=44, col=(236, 112, 90))
            x_, y_ = P(MARK_CELL[:, 0].max() + 0.25, MARK_P[1])
            note(im, (x_ / S, y_ / S), f"one cone: {cone_um(1.0):.1f} µm", a_b * smooth((t - 20.3) / 0.6),
                 size=44, col=(176, 52, 38), anchor="lm")
        if u > 0.5:
            d = ImageDraw.Draw(im)
            a = smooth((u - 0.5) / 0.5) * fade(t, 0, self.dur, 0, 0.6)
            L.text(d, (1920 * 0.29, 1080 * 0.835), "your retina, up close", 32, CREAM, a, kind="italic", anchor="mm")
            L.text(d, (1920 * 0.71, 1080 * 0.835), "a TV screen, up close", 32, CREAM, a, kind="italic", anchor="mm")
        caps = [(0.6, 7.6, ["This is how that line lands on your retina."]),
                (8.2, 16.0, ["The light falls on cones, packed in a honeycomb.", "Each one reports a single brightness."]),
                (16.6, 28.4, ["The step is narrower than one cone."]),
                (31.8, 40.6, ["A cone is a pixel of your eye, like a pixel on a TV."]),
                (41.0, 49.4, ["The step is smaller than a pixel.", "By that logic, you shouldn’t be able to see it."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=38)
        foot = fade(t, 17.0, 28.4, 0.6, 0.6)
        if foot > 0:
            L.text(ImageDraw.Draw(im), (1780, 1050), "Sizes for a phone held at reading distance.", 22, (150, 140, 130), foot,
                   kind="italic", anchor="rm")
        return np.asarray(im)


# ================================================================== 4. back to real size
class Back(L.Section):
    """one continuous zoom out: from the micrograph, through the card, to the line at its real size"""
    name, dur = "back", 24.0
    ZOOM, X0, X1 = 6.0, 1.4, 2.4               # the zoom's length; the micrograph dissolves into the card over X0..X1
    K_PER_Z = R.CONE_ARCMIN * 60 / PX_ARCSEC * S   # micrograph px per cone, for a desk zoom of 1

    def z(self, t):
        z0 = K_RET / self.K_PER_Z
        return z0 ** (1 - ease(t / self.ZOOM))

    def frame(self, t):
        c = CARDS[0]
        z = self.z(t)
        # keep the step where the micrograph had it (the field's centre), easing to the frame's centre
        off = (0.5 - R.FIELD_CY) * H / (S * z) * (1 - smooth((t - self.X1) / (self.ZOOM - self.X1)))
        cam = Cam(c.xm, c.line_y + c.th / 2 + c.step / 2 + off, z)
        d = desk(cam, ticks=(0, smooth((t - 6.6) / 1.2)) if t > 6.6 else None, t=t, plates=smooth((t - 4.0) / 1.5))
        if t < self.X1:
            # the micrograph, its dark surround giving way to the card, then dissolving into it
            u = ease(t / self.X1)
            field = (W * R.FIELD_CX, H * R.FIELD_CY, H * (R.FIELD_R + 0.1 * u))
            img = R.mosaic(R.Scope(0.0, STEP_HALF / 2, z * self.K_PER_Z), y0=0.0, step=STEP_HALF, lit=1.0, band=0.3 * (1 - u),
                           field=False)
            m, inside = R.field_mask(*field)
            w = smooth((t - 0.3) / 1.1)
            out = R.OUTSIDE * (1 - w) + d.astype(np.float32) * w
            fr = np.clip(img * m[..., None] + out * (1 - inside[..., None]), 0, 255).astype(np.uint8)
            d = mix(fr, d, smooth((t - self.X0) / (self.X1 - self.X0)))
        im = Image.fromarray(d)
        caption(im, ["Now, back to its real size.", "Look at the line, between the pencil marks."], fade(t, 6.4, 13.6, 0.6, 0.6))
        caption(im, ["You can see the step, can’t you?"], fade(t, 14.4, 23.6, 0.8, 0.6))
        return np.asarray(im)


# ================================================================== 5. smaller and smaller
class Smaller(L.Section):
    """pan down the stack of plates: a quarter, an eighth, and none"""
    name = "smaller"
    PAN = 3.0
    SEG = [(1, 22.0, ["A quarter of a pixel.", "On your retina, about a third of a cone."], 9.0),
           (2, 24.0, ["An eighth of a pixel: about a sixth of a cone.", "Some people can still see this one."], 10.0),
           (3, 14.0, ["And for comparison, a line with no step at all."], None)]

    def __init__(self):
        self.dur = sum(self.PAN + d for _, d, _, _ in self.SEG)

    def frame(self, t):
        t0 = 0.0
        prev = 0
        for k, d, lines, lens_at in self.SEG:
            if t < t0 + self.PAN + d:
                break
            t0 += self.PAN + d
            prev = k
        u = t - t0
        if u < self.PAN:
            cam = Cam.lerp(HOME[prev], HOME[k], ease(u / self.PAN))
            return desk(cam, t=t)
        v = u - self.PAN
        lens = None
        if lens_at is not None:
            lens = lens_path(v, lens_at, lens_at + 5.0, CARDS[k], mag={1: 12.0, 2: 20.0}[k])
        fr = desk(HOME[k], lens=lens, ticks=(k, smooth((v - 0.3) / 1.0)), t=t)
        im = Image.fromarray(fr)
        caption(im, lines, fade(v, 0.6, lens_at if lens_at else d - 0.4, 0.6, 0.6) if lens_at else fade(v, 0.6, d - 0.4, 0.6, 0.6))
        return np.asarray(im)


# ================================================================== 6. how
HOW_C = (W * 0.33, H * 0.45)                   # the field's centre (4K px) while the plot is up
HOW_R = H * 0.40
K_HOW = 130.0
HOW_V = 0.3                                    # the retina coordinate at the field's centre
PLOT_X0, PLOT_W = W * 0.615, W * 0.25          # the light plot: x = light, y = the same rows as the field
CREAM_P = (236, 222, 200)
RED_P = (222, 84, 62)
SQ3 = math.sqrt(3) / 2


def column(uc, rows=range(-6, 7)):
    """one column of cones across the line: in each row, the cone nearest u = uc"""
    p, _ = R.lattice()
    r = np.round(p[:, 1] / SQ3).astype(int)
    out = []
    for k in rows:
        m = np.nonzero(r == k)[0]
        out.append(m[np.argmin(np.abs(p[m, 0] - uc))])
    return np.array(out)


def fit_line(idx, read, guess=0.0):
    """where the line's middle must be for the eye's blur to best explain these readings (least squares)"""
    p, _ = R.lattice()
    ys = np.linspace(guess - 1.5, guess + 1.5, 1201)
    model = R.light_at(p[idx, 1][None, :] - ys[:, None])
    return ys[int(np.argmin(((model - read[None, :]) ** 2).sum(1)))]


def hexagon(i):
    p, _ = R.lattice()
    q = p[i] + (R.cells()[i] - p[i]) / 0.91
    return np.vstack([q, q[:1]])


COL_U = -1.5
COL = column(COL_U)
POOL_U = np.arange(-5.5, 6.0, 1.0)
POOL_COLS = [column(u, range(-5, 6)) for u in POOL_U]
POOL_NOISE = [np.random.default_rng(500 + j).normal(0, 0.10, len(c)) for j, c in enumerate(POOL_COLS)]


class How(L.Section):
    name, dur = "how", 66.0
    MOVE0, MOVE1 = 26.0, 29.0
    POOL = 41.0

    def state(self, t):
        u = ease((t - self.POOL) / 3.0)
        y0 = ease((t - self.MOVE0) / (self.MOVE1 - self.MOVE0)) / 3 * (1 - u)
        return y0, STEP_HALF * u, HOW_C[0] + (W * 0.5 - HOW_C[0]) * u

    def frame(self, t):
        y0, step, cx = self.state(t)
        sc = R.Scope(0.0, HOW_V, K_HOW)
        fr = R.mosaic(sc, y0=y0, step=step, lit=1.0, band=0.35, centre=(cx, HOW_C[1]), radius=HOW_R)
        fr = blackout(fr, 1 - smooth(t / 0.8))
        X = lambda u: (np.asarray(u, np.float64) - sc.u) * sc.k + cx
        Y = lambda v: (np.asarray(v, np.float64) - sc.v) * sc.k + HOW_C[1]
        PX = lambda Lv: PLOT_X0 + (np.asarray(Lv) - 0.35) / 0.65 * PLOT_W
        p, _ = R.lattice()
        texts = []
        a_plot = fade(t, 2.6, self.POOL + 0.4, 0.6, 0.8)
        if a_plot > 0:
            vs = np.linspace(-5.5, 5.5, 160)
            curve = np.c_[PX(R.light_at(vs - y0)), Y(vs)]
            axis = [(PLOT_X0 - 40, Y(-5.7)), (PLOT_X0 - 40, Y(5.7))]
            arrow = [(PX(0.4), Y(6.1)), (PX(1.0) + 30, Y(6.1))]
            head = [(PX(1.0) + 6, Y(6.1) - 16), (PX(1.0) + 30, Y(6.1)), (PX(1.0) + 6, Y(6.1) + 16)]
            fr = pencil(fr, [(axis, smooth((t - 2.6) / 0.8), 61), (arrow, smooth((t - 3.0) / 0.8), 62),
                             (head, smooth((t - 3.7) / 0.3), 63), (curve, smooth((t - 4.0) / 2.2), 64)],
                        col=CREAM_P, a=a_plot, width=2.6)
            texts.append(((PX(1.0) + 30) / S, Y(6.1) / S + 18, "more light", a_plot * smooth((t - 3.8) / 0.6), CREAM_P, "ra"))
        # one column of cones, outlined, and its readings as dots on the plot
        a_col = fade(t, 12.4, self.POOL + 0.4, 0.4, 0.8)
        if a_col > 0:
            fr = pencil(fr, [(np.c_[X(h[:, 0]), Y(h[:, 1])], smooth((t - 12.4 - 0.1 * j) / 0.5), 70 + j)
                             for j, h in enumerate(hexagon(i) for i in COL)], col=RED_P, a=a_col, width=2.4)
            Lc = R.light_at(p[COL, 1] - y0)
            dots = np.zeros(fr.shape[:2], np.float32)
            for j, i in enumerate(COL):
                ad = smooth((t - 14.6 - 0.16 * j) / 0.3)
                if ad > 0:
                    cv2.circle(dots, (int(PX(Lc[j]) * 16), int(Y(p[i, 1]) * 16)), int(13 * 16), ad, -1, cv2.LINE_AA, shift=4)
            if dots.any():
                dm = (dots * a_col)[..., None]
                fr = np.clip(fr * (1 - dm) + np.array(RED_P, np.float32) * dm, 0, 255).astype(np.uint8)
            # the middle of the pattern, and where it was before the line moved
            a_m = a_col * smooth((t - 18.6) / 0.6)
            if a_m > 0:
                est = fit_line(COL, Lc, y0)
                strokes = [([(PLOT_X0 - 70, Y(est)), (PX(1.0), Y(est))], smooth((t - 18.6) / 0.7), 80),
                           ([(X(COL_U - 0.9), Y(est)), (X(COL_U + 0.9), Y(est))], smooth((t - 18.9) / 0.5), 81)]
                fr = pencil(fr, strokes, col=(250, 120, 92), a=a_m, width=3.4)
                texts.append((PX(1.0) / S + 14, Y(est) / S, "middle", a_m * smooth((t - 19.2) / 0.5), (250, 120, 92), "lm"))
                a_g = a_m * smooth((t - self.MOVE0 - 0.2) / 0.6)
                if a_g > 0:
                    fr = pencil(fr, [([(PLOT_X0 - 70, Y(0.0)), (PX(1.0), Y(0.0))], 1.0, 82)], col=CREAM_P, a=a_g * 0.5, width=2.2)
                    a_br = a_m * smooth((t - self.MOVE1 - 0.3) / 0.6)
                    if a_br > 0:
                        xb = PLOT_X0 - 110
                        fr = pencil(fr, [(pen_bracket(xb, Y(0.0), Y(est), side=-1, tick=22), smooth((t - self.MOVE1 - 0.3) / 0.6), 83)],
                                    col=(250, 120, 92), a=a_br, width=2.6)
                        texts.append((xb / S - 14, Y(est / 2) / S, "⅓ cone", a_br, (250, 120, 92), "rm"))
        # every column along the line, each with its own rough answer, then the averages either side of the step
        if t > self.POOL + 3.0:
            ests = []
            strokes, outl = [], []
            for j, (uc, cidx) in enumerate(zip(POOL_U, POOL_COLS)):
                t_j = self.POOL + 3.4 + 0.42 * j
                truth = y0 + (step if uc > 0 else 0.0)
                read = R.light_at(p[cidx, 1] - truth) + POOL_NOISE[j]
                e = fit_line(cidx, read, truth)
                ests.append(e)
                f = smooth((t - t_j) / 0.4)
                for m_, i in enumerate(cidx):
                    if math.hypot(X(p[i, 0]) - cx, Y(p[i, 1]) - HOW_C[1]) > HOW_R - 0.75 * K_HOW:
                        continue                                  # only cones well inside the field
                    h = hexagon(i)
                    outl.append((np.c_[X(h[:, 0]), Y(h[:, 1])], f, 200 + 13 * j + m_))
                strokes.append(([(X(uc - 0.42), Y(e)), (X(uc + 0.42), Y(e))], smooth((t - t_j - 0.3) / 0.3), 300 + j))
            a_p = fade(t, self.POOL + 3.4, self.dur, 0.3, 0.8)
            fr = pencil(fr, outl, col=CREAM_P, a=a_p * 0.35, width=1.8)
            fr = pencil(fr, strokes, col=(250, 120, 92), a=a_p, width=3.2)
            ests = np.array(ests)
            mL, mR = ests[POOL_U < 0].mean(), ests[POOL_U > 0].mean()
            t_a = self.POOL + 3.4 + 0.42 * len(POOL_U) + 0.8
            trace = [(X(-6.0), Y(mL)), (X(-0.02), Y(mL)), (X(0.02), Y(mR)), (X(6.0), Y(mR))]
            trace = np.concatenate([np.linspace(trace[i], trace[i + 1], 40) for i in range(3)])
            fr = pencil(fr, [(trace, smooth((t - t_a) / 1.4), 330)], col=(255, 150, 120), a=a_p, width=3.6)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for x_, y_, s_, a_, c_, an in texts:
            L.text(d, (x_, y_), s_, 38, c_, a_, kind="hand", anchor=an)
        caps = [(1.0, 11.4, ["So how can you see it? It starts with blur.",
                             "Your eye spreads the line’s shadow over several rows of cones."]),
                (12.2, 23.6, ["Follow one column of cones across the line.", "Each cone reports how much light reaches it."]),
                (24.4, 39.8, ["Now move the line down by a third of a cone.",
                              "Every reading changes a little, and the middle of the pattern moves with it."]),
                (self.POOL + 3.0, 56.4, ["Your brain reads every column along the line and pools the answers.",
                                         "Left and right of the step, they differ. That difference is what you saw."]),
                (57.2, 65.4, ["Each cone is coarse. Comparing many of them is not."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=38)
        return np.asarray(im)


# ================================================================== 7. the screen does it in reverse
class Screen(L.Section):
    name, dur = "screen", 33.0
    K = 150.0
    C = (W * 0.5, H * 0.45)
    RAD = H * 0.40
    MOVES = (4.6, 6.9, 9.2, 11.5)               # the line moves down a quarter of a pixel at each
    STEP_T = 15.2                               # then: back up, with a half-pixel step on the right
    V0 = 1.5                                    # the pixel row at the field's centre

    def state(self, t):
        y0 = sum(0.25 * smooth((t - m) / 0.3) for m in self.MOVES)
        s = smooth((t - self.STEP_T) / 0.4)
        return y0 * (1 - s), 0.5 * s

    def frame(self, t):
        y0, step = self.state(t)
        fr = T.macro(0.0, self.V0, self.K, self.C, self.RAD, y0=y0, th=2.0, step=step, xm=0.0)
        fr = blackout(fr, 1 - smooth(t / 0.8))
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        a = fade(t, 2.2, self.dur, 0.6, 0.8)
        rows = np.arange(-1, 5)
        for side, off, an, yy in ((-1, 0.0, "rm", y0), (1, step, "lm", y0 + step)):
            lum = 1 - T.row_cover(rows.astype(np.float64), yy, 2.0)
            x_ = (self.C[0] + side * (self.RAD + 70)) / S
            L.text(d, (x_, (self.C[1] + (rows[0] - 0.9 - self.V0) * self.K) / S), "light", 34, (170, 160, 150), a, kind="hand", anchor=an)
            for r, lv in zip(rows, lum):
                yv = (self.C[1] + (r + 0.5 - self.V0) * self.K) / S
                L.text(d, (x_, yv), f"{int(round(lv * 100))}%", 40, CREAM if 0 < lv < 1 else (150, 140, 130), a, kind="hand", anchor=an)
        caps = [(0.8, 13.8, ["Screens play the same trick in reverse.",
                             "To move a line a quarter of a pixel, they share its darkness between the rows it covers."]),
                (15.8, 23.8, ["A half-pixel step is just a change in how bright the rows at its edges are."]),
                (24.4, 32.4, ["That’s how this video drew steps smaller than a pixel.",
                              "Your eye reads the shading, and finds the line."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6), col=CREAM, y=1000, size=38)
        return np.asarray(im)


# ================================================================== 8. the numbers, on a notebook page
def dim_line(x0, x1, y, tick=10):
    """a dimension line |---| in screen px"""
    return [[(x0, y - tick), (x0, y + tick)], [(x0, y), (x1, y)], [(x1, y - tick), (x1, y + tick)]]


class Numbers(L.Section):
    name, dur = "numbers", 36.0
    X0, PER = 700.0, 12.0                       # bars start here (1080p units); 12 units per arcsecond
    ROWS = [("cone spacing", 30, GRAPHITE, "30″", 2.0),
            ("plate 1", 20, RED, "20″", 3.6),
            ("plate 2", 10, RED, "10″", 4.4),
            ("plate 3", 5, RED, "5″", 5.2),
            ("most people", (5, 10), GRAPHITE, "5–10″", 13.4)]

    def frame(self, t):
        cam = Cam(960, 540, 1.0)
        fr = L.finish(L.paper(cam), t)
        y = 300.0
        texts = []
        for k, (lab, v, col, val, t0) in enumerate(self.ROWS):
            f = smooth((t - t0) / 0.7)
            if f <= 0:
                y += 82
                continue
            if isinstance(v, tuple):
                segs = dim_line(self.X0 * S, (self.X0 + v[0] * self.PER) * S, y * S)
                dash = [[((self.X0 + (v[0] + q) * self.PER) * S, y * S), ((self.X0 + (v[0] + q + 0.5) * self.PER) * S, y * S)]
                        for q in np.arange(0.4, v[1] - v[0], 1.0)]
                end = [[((self.X0 + v[1] * self.PER) * S, (y - 10) * S), ((self.X0 + v[1] * self.PER) * S, (y + 10) * S)]]
                segs = segs + dash + end
                xe = self.X0 + v[1] * self.PER
            else:
                segs = dim_line(self.X0 * S, (self.X0 + v * self.PER) * S, y * S)
                xe = self.X0 + v * self.PER
            fr = pencil(fr, [(sg, smooth((t - t0 - 0.1 * q) / 0.4), 400 + 10 * k + q) for q, sg in enumerate(segs)], col=col, width=3.0)
            texts.append(((self.X0 - 34, y), lab, f, INK_SOFT, "rm"))
            texts.append(((xe + 26, y), val, smooth((t - t0 - 0.5) / 0.5), col, "lm"))
            y += 82
        fr = pencil(fr, [([(150 * S, 168 * S), (430 * S, 172 * S)], smooth((t - 1.2) / 0.6), 450)], col=GRAPHITE, width=2.6)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        L.text(d, (150, 160), "hyperacuity", 64, INK, smooth((t - 0.4) / 0.8), kind="hand", anchor="ls")
        for xy, s_, a_, c_, an in texts:
            L.text(d, xy, s_, 40, c_, a_, kind="hand", anchor=an)
        a_n = smooth((t - 25.6) / 0.8)
        L.text(d, (self.X0, 740), "5″ ≈ a 0.5 mm pencil lead, 20 m away", 42, RED, a_n, kind="hand", anchor="lm")
        caps = [(0.8, 12.2, ["Seeing finer than the spacing of your own receptors has a name: hyperacuity.",
                             "Gerald Westheimer coined it in 1975."]),
                (13.0, 24.4, ["Most people can see a step of 5 to 10 arcseconds:",
                              "a sixth to a third of the gap between two cones."]),
                (25.2, 35.4, ["5 arcseconds is about the width of a pencil lead,", "seen from 20 metres away."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6))
        return blackout(np.asarray(im), 1 - smooth(t / 0.8))


# ================================================================== 9. where it's used
VCARD = Card(x0=240, y0=170, x1=1680, y1=560, line_y=-999, step=0.0)
ACARD = Card(x0=630, y0=1210, x1=1290, y1=1870, line_y=-999, step=0.0)
MM = 30.0                                       # vernier drawing: units per millimetre
MAIN_X0, BOUND = 330.0, 380.0                   # the main scale's zero; the line where the two scales meet
GRID_C, GRID_N, GRID_D = np.array([960.0, 1540.0]), 20, 28.0
BULGE_C, BULGE_S, BULGE_A = np.array([1018.0, 1492.0]), 62.0, 17.0


def ink_lines(fr, cam, lines, width=1.6, col=INK):
    """crisp printed lines (world polylines), anti-aliased"""
    m = np.zeros(fr.shape[:2], np.uint8)
    for pts in lines:
        P = np.c_[cam.X(np.asarray(pts)[:, 0]), cam.Y(np.asarray(pts)[:, 1])]
        cv2.polylines(m, [np.round(P * 16).astype(np.int32)], False, 255, max(1, int(round(width * S * cam.z))), cv2.LINE_AA, shift=4)
    a = (m.astype(np.float32) / 255)[..., None]
    return np.clip(fr * (1 - a) + np.array(col, np.float32) * a, 0, 255).astype(np.uint8)


def amsler(amt):
    """the grid's lines (world polylines), with a bulge of strength amt"""
    h = GRID_N * GRID_D / 2
    s = np.linspace(-h, h, 160)
    out = []
    for k in range(GRID_N + 1):
        c = -h + k * GRID_D
        for pts in (np.c_[s, np.full_like(s, c)], np.c_[np.full_like(s, c), s]):
            P = pts + GRID_C
            if amt > 0:
                dv = P - BULGE_C
                r = np.hypot(dv[:, 0], dv[:, 1])[:, None] / BULGE_S
                P = P + amt * BULGE_A * dv / BULGE_S * np.exp(-r ** 2 / 2)
            out.append(P)
    return out


class Uses(L.Section):
    name, dur = "uses", 53.0
    PAN0, PAN1 = 20.2, 22.8
    BEND0, BEND1 = 33.0, 35.6

    def frame(self, t):
        cam = Cam.lerp(Cam(960, 365, 1.0), Cam(960, 1540, 1.0), ease((t - self.PAN0) / (self.PAN1 - self.PAN0)))
        fr = L.paper(cam)
        VCARD.draw(fr, cam, line=0)
        ACARD.draw(fr, cam, line=0)
        # the vernier scale: main scale in mm above, the vernier (ten divisions in 9 mm) below, sliding to 23.4 mm
        z = 15.0 + 8.4 * ease((t - 3.0) / 4.2)
        main = []
        for m in range(0, 41):
            ln = 40 if m % 10 == 0 else 28 if m % 5 == 0 else 18
            main.append([(MAIN_X0 + m * MM, BOUND - ln), (MAIN_X0 + m * MM, BOUND)])
        main.append([(MAIN_X0 - 20, BOUND), (MAIN_X0 + 40 * MM + 20, BOUND)])
        vx0 = MAIN_X0 + z * MM
        vern = []
        for n in range(11):
            ln = 34 if n % 5 == 0 else 22
            vern.append([(vx0 + n * 0.9 * MM, BOUND), (vx0 + n * 0.9 * MM, BOUND + ln)])
        vern.append([(vx0 - 14, BOUND + 60), (vx0 + 9 * MM + 14, BOUND + 60)])
        fr = ink_lines(fr, cam, main + vern, width=1.4)
        # the Amsler grid and its centre dot
        amt = smooth((t - self.BEND0) / (self.BEND1 - self.BEND0))
        if cam.Y(ACARD.y0) < H:
            fr = ink_lines(fr, cam, amsler(amt), width=1.2)
            cv2.circle(fr, (int(cam.X(GRID_C[0]) * 16), int(cam.Y(GRID_C[1]) * 16)), int(5 * S * cam.z * 16), INK, -1, cv2.LINE_AA, shift=4)
        fr = L.finish(fr, t)
        # red pencil: the two marks that line up
        a_c = fade(t, 8.0, self.PAN0 + 0.6, 0.4, 0.6)
        if a_c > 0:
            xa = MAIN_X0 + 27 * MM
            ring = np.c_[cam.X(xa + 26 * np.cos(np.linspace(0.3, 6.6, 50))), cam.Y(BOUND + 34 * np.sin(np.linspace(0.3, 6.6, 50)))]
            fr = pencil(fr, [(ring, smooth((t - 8.0) / 0.8), 500)], col=RED, a=a_c, width=3.0)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        for m in range(0, 5):
            L.text(d, (cam.X(MAIN_X0 + m * 10 * MM) / S, cam.Y(BOUND - 50) / S), str(m), 30, INK, 1.0, anchor="ms")
        for n in (0, 5, 10):
            L.text(d, (cam.X(vx0 + n * 0.9 * MM) / S, cam.Y(BOUND + 66) / S), str(n), 24, INK, 1.0, anchor="mt")
        L.text(d, (cam.X(VCARD.x0 + 40) / S, cam.Y(VCARD.y0 + 50) / S), "cm", 24, INK_SOFT, 1.0, kind="italic")
        a_c2 = fade(t, 9.0, self.PAN0 + 0.6, 0.4, 0.6)
        L.text(d, (cam.X(MAIN_X0 + 27 * MM + 44) / S, cam.Y(BOUND + 120) / S), "23.4 mm", 44, RED, a_c2, kind="hand", anchor="lm")
        L.text(d, (cam.X(VCARD.x1) / S, cam.Y(VCARD.y1 + 52) / S), "Pierre Vernier, 1631", 40, INK_SOFT,
               fade(t, 1.4, self.PAN0 + 0.6, 0.6, 0.6), kind="hand", anchor="rm")
        L.text(d, (cam.X(ACARD.x1 + 40) / S, cam.Y(BULGE_C[1]) / S), "simulated", 38, INK_SOFT,
               fade(t, self.BEND1, self.dur, 0.6, 0.6), kind="hand", anchor="lm")
        caps = [(0.8, 9.8, ["Hyperacuity is why a vernier scale works.", "It reads a tenth of a millimetre by eye."]),
                (10.4, 19.8, ["You just spot which two marks line up.", "Here, the 4th: 23.4 mm."]),
                (23.2, 32.4, ["Eye doctors rely on it too. In an Amsler grid, you look at the centre dot,",
                              "and even a small bend in a line stands out."]),
                (33.2, 42.4, ["Bent or wavy lines can be an early sign of trouble in the macula.",
                              "Some home monitors for macular degeneration are built on hyperacuity."]),
                (43.2, 52.4, ["If straight lines ever look bent or wavy to you, see your optometrist."])]
        for c0, c1, lines in caps:
            caption(im, lines, fade(t, c0, c1, 0.6, 0.6))
        return np.asarray(im)


# ================================================================== 10. close
class Close(L.Section):
    name, dur = "close", 19.0
    PULL0, PULL1 = 7.6, 15.0

    def frame(self, t):
        cam = Cam.lerp(HOME[0], Cam(960, 1060, 0.55), ease((t - self.PULL0) / (self.PULL1 - self.PULL0)))
        fr = desk(cam, ticks=(0, 1.0), t=t)
        im = Image.fromarray(fr)
        caption(im, ["Your cones are about 30 arcseconds apart."], fade(t, 0.8, 7.6, 0.6, 0.6))
        caption(im, ["You saw a step a third of that. Maybe even a sixth."], fade(t, 8.2, 16.0, 0.6, 0.6))
        return blackout(np.asarray(im), smooth((t - 16.4) / 2.4))


SECTIONS = {"open": Open, "eye": Eye, "retina": Retina, "back": Back, "smaller": Smaller, "how": How,
            "screen": Screen, "numbers": Numbers, "uses": Uses, "close": Close}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s_ in sys.argv[2:]:
            sec.render(float(s_))
    else:
        sec.render()
