"""The look of the sixty film: brass lines engraved on a night sky, like an astrolabe. Cream serif captions,
gold for the sixties, lapis for Egypt's hours, clay for Babylon, cyan for anything decimal.

Drawing is Skia (anti-aliased paths); type is set with Pillow/RAQM (kerning, ligatures) and cached as images.
Geometry is in 1080p units; SIXTY_SCALE (default 2) sets pixels per unit, so 2 renders 4K and 1 a quick 1080p.
"""
import os, sys, math, time, subprocess, functools
import numpy as np
import skia
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONTS = os.path.join(HERE, "fonts")
S = float(os.environ.get("SIXTY_SCALE", "2"))
W, H, FPS = int(round(1920 * S)), int(round(1080 * S)), 30

NIGHT0 = (6, 8, 17)
NIGHT1 = (18, 24, 46)
GOLD = (224, 186, 112)
GOLD_DIM = (150, 124, 80)
CREAM = (241, 232, 213)
CREAM_DIM = (176, 168, 152)
CLAY = (214, 132, 88)
LAPIS = (104, 150, 232)
NIGHTBLUE = (44, 64, 128)
CYAN = (120, 214, 238)
RED = (226, 96, 80)
SUN = (255, 214, 140)


# ------------------------------------------------------------------ timing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def fade(t, t0, t1, fin=0.6, fout=0.6):
    if t < t0 or t > t1:
        return 0.0
    return float(min(1.0, (t - t0) / fin if fin else 1.0, (t1 - t) / fout if fout else 1.0))


def ramp(t, t0, dur):
    return ease((t - t0) / dur) if dur > 0 else float(t >= t0)


def lerp(a, b, u):
    return a + (b - a) * u


def mixc(c0, c1, u):
    return tuple(int(round(a + (b - a) * u)) for a, b in zip(c0, c1))


# ------------------------------------------------------------------ paint
def C4(col, a=1.0):
    return skia.Color4f(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, float(clamp(a)))


def paint(col, a=1.0, stroke=None, cap="round", blur=None, join="round"):
    p = skia.Paint()
    p.setAntiAlias(True)
    p.setColor4f(C4(col, a))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap({"round": skia.Paint.kRound_Cap, "butt": skia.Paint.kButt_Cap, "square": skia.Paint.kSquare_Cap}[cap])
        p.setStrokeJoin({"round": skia.Paint.kRound_Join, "miter": skia.Paint.kMiter_Join}[join])
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def line(c, x0, y0, x1, y1, col, a=1.0, w=1.5, glow=0.0, cap="round"):
    if a <= 0:
        return
    if glow > 0:
        c.drawLine(x0, y0, x1, y1, paint(col, a * glow, stroke=w * 4, blur=w * 2.5, cap=cap))
    c.drawLine(x0, y0, x1, y1, paint(col, a, stroke=w, cap=cap))


def poly(c, pts, col, a=1.0, w=1.5, closed=False, fill=False, glow=0.0):
    if a <= 0 or len(pts) < 2:
        return
    path = skia.Path()
    path.moveTo(*map(float, pts[0]))
    for p in pts[1:]:
        path.lineTo(*map(float, p))
    if closed:
        path.close()
    if fill:
        c.drawPath(path, paint(col, a))
        return
    if glow > 0:
        c.drawPath(path, paint(col, a * glow, stroke=w * 4, blur=w * 2.5))
    c.drawPath(path, paint(col, a, stroke=w))


def circle(c, x, y, r, col, a=1.0, w=None, glow=0.0):
    if a <= 0 or r <= 0:
        return
    if w is None:
        c.drawCircle(x, y, r, paint(col, a))
    else:
        if glow > 0:
            c.drawCircle(x, y, r, paint(col, a * glow, stroke=w * 4, blur=w * 2.5))
        c.drawCircle(x, y, r, paint(col, a, stroke=w))


def dot_glow(c, x, y, r, col, a=1.0, halo=4.0):
    """a soft point of light: radial falloff plus a crisp core"""
    if a <= 0:
        return
    R = r * halo
    sh = skia.GradientShader.MakeRadial((x, y), R, [C4(col, 0.55 * a).toColor(), C4(col, 0.12 * a).toColor(), C4(col, 0).toColor()],
                                        [0.0, 0.35, 1.0])
    p = skia.Paint(AntiAlias=True)
    p.setShader(sh)
    c.drawCircle(x, y, R, p)
    c.drawCircle(x, y, r, paint(mixc(col, (255, 255, 255), 0.5), a))


def sector(c, cx, cy, r0, r1, a0, a1, col, a=1.0):
    """filled annular sector, angles in degrees clockwise from 12 o'clock"""
    if a <= 0 or a1 <= a0:
        return
    path = skia.Path()
    s0, sw = a0 - 90.0, a1 - a0
    path.arcTo(skia.Rect.MakeLTRB(cx - r1, cy - r1, cx + r1, cy + r1), s0, sw, True)
    if r0 > 0:
        path.arcTo(skia.Rect.MakeLTRB(cx - r0, cy - r0, cx + r0, cy + r0), s0 + sw, -sw, False)
    else:
        path.lineTo(cx, cy)
    path.close()
    c.drawPath(path, paint(col, a))


def arc(c, cx, cy, r, a0, a1, col, a=1.0, w=1.5, glow=0.0):
    """stroked arc, degrees clockwise from 12 o'clock"""
    if a <= 0 or abs(a1 - a0) < 1e-6:
        return
    path = skia.Path()
    path.arcTo(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), a0 - 90.0, a1 - a0, True)
    if glow > 0:
        c.drawPath(path, paint(col, a * glow, stroke=w * 4, blur=w * 2.5, cap="butt"))
    c.drawPath(path, paint(col, a, stroke=w, cap="butt"))


def polar(cx, cy, r, deg):
    th = math.radians(deg - 90.0)
    return cx + r * math.cos(th), cy + r * math.sin(th)


def ticks(c, cx, cy, r, n, ln, col, a=1.0, w=1.2, frac=1.0, start=0.0, every=None, major=None, skip=None):
    """n radial ticks from r inward by ln (ln < 0: outward). frac: share of ticks drawn so far (clockwise)."""
    if a <= 0:
        return
    m = int(math.floor(n * frac + 1e-9))
    path = skia.Path()
    pathM = skia.Path()
    for k in range(m):
        if skip and skip(k):
            continue
        d = start + 360.0 * k / n
        L = ln
        target = path
        if major and k % major[0] == 0:
            L, target = major[1], pathM
        x0, y0 = polar(cx, cy, r, d)
        x1, y1 = polar(cx, cy, r - L, d)
        target.moveTo(x0, y0)
        target.lineTo(x1, y1)
    c.drawPath(path, paint(col, a, stroke=w, cap="butt"))
    if major:
        c.drawPath(pathM, paint(col, a, stroke=major[2] if len(major) > 2 else w * 1.6, cap="butt"))


# ------------------------------------------------------------------ type
FACES = {"serif": "SourceSerif4.ttf", "italic": "SourceSerif4-Italic.ttf", "sans": "Inter-Regular.otf",
         "sans-m": "Inter-Medium.otf", "sans-sb": "Inter-SemiBold.otf", "black": "Inter-Black.otf", "display": "Anton-Regular.ttf",
         "cjk": "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"}


@functools.lru_cache(None)
def pil_font(kind, px, weight):
    path = FACES[kind]
    path = path if path.startswith("/") else os.path.join(FONTS, path)
    f = ImageFont.truetype(path, px, layout_engine=ImageFont.Layout.RAQM)
    if kind in ("serif", "italic"):
        f.set_variation_by_axes([weight or 400, max(8, min(60, px / S * 0.6))])
    return f


@functools.lru_cache(maxsize=6000)
def _glyphs(s, kind, px, weight):
    """white text on transparent, premultiplied, with the baseline-left origin offset and the advance"""
    f = pil_font(kind, px, weight)
    l, t, r, b = f.getbbox(s, anchor="ls")
    pad = 3
    w, h = int(r - l) + 2 * pad, int(b - t) + 2 * pad
    m = Image.new("L", (max(w, 1), max(h, 1)), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), s, font=f, fill=255, anchor="ls")
    a = np.asarray(m)
    rgba = np.repeat(a[..., None], 4, axis=2).copy()
    img = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
    adv = f.getlength(s)
    asc, desc = f.getmetrics()
    cap = -f.getbbox("H", anchor="ls")[1]
    return img, l - pad, t - pad, adv, cap


def text_width(s, size, kind="serif", weight=None):
    px = max(4, int(round(size * S * 2)) / 2)
    return _glyphs(s, kind, int(px), weight)[3] / S * size * S / int(px)


def text(c, x, y, s, size, col, a=1.0, kind="serif", weight=None, anchor="ls"):
    """draw s with its anchor at (x, y) in canvas units. anchor: [l|m|r][s|m] (baseline or cap-middle)."""
    if a <= 0 or not s:
        return
    k = c.getTotalMatrix().getScaleX()
    px = max(4, int(round(size * k)))
    img, ox, oy, adv, cap = _glyphs(s, kind, px, weight)
    sc = size / px                                       # canvas units per glyph-image pixel
    dx = {"l": 0.0, "m": -adv / 2, "r": -adv}[anchor[0]]
    dy = {"s": 0.0, "m": cap / 2, "t": cap}[anchor[1]]
    c.save()
    c.translate(x, y)
    c.scale(sc, sc)
    p = skia.Paint(AntiAlias=True)
    p.setColorFilter(skia.ColorFilters.Blend(C4(col).toColor(), skia.BlendMode.kSrcIn))
    p.setAlphaf(float(clamp(a)))
    c.drawImage(img, ox + dx, oy + dy, skia.SamplingOptions(skia.FilterMode.kLinear), p)
    c.restore()


def caption(c, lines, a, x=150, y=972, size=40, col=CREAM):
    """documentary subtitles: left-aligned serif lines, bottom-left, over a soft shade"""
    if a <= 0:
        return
    if isinstance(lines, str):
        lines = [lines]
    lh = size * 1.34
    yy = y - (len(lines) - 1) * lh + (1 - a) * 8
    wmax = max(text_width(ln, size) for ln in lines)
    # a soft dark bed under the type, so it reads over any picture
    bed = skia.Rect.MakeLTRB(x - 60, yy - size * 1.3, x + wmax + 60, y + size * 0.75)
    c.drawRoundRect(bed, 40, 40, paint((3, 5, 12), 0.42 * a, blur=30))
    for ln in lines:
        text(c, x, yy, ln, size, col, a)
        yy += lh


def label(c, x, y, s, a, size=30, col=CREAM, anchor="ms", kind="serif", weight=None):
    text(c, x, y, s, size, col, a, kind=kind, weight=weight, anchor=anchor)


# ------------------------------------------------------------------ the sky
def _stars():
    rng = np.random.default_rng(60)
    n = 3400
    r = 2300 * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * math.pi, n)
    mag = rng.pareto(2.2, n)
    b = np.clip(0.18 + 0.22 * mag, 0.15, 1.0)
    size = np.clip(0.55 + 0.9 * mag, 0.55, 2.6)
    tint = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * math.pi, n)
    om = rng.uniform(1.0, 3.0, n)
    return r * np.cos(th), r * np.sin(th), b, size, tint, ph, om


STARS = _stars()


def stars(c, t=0.0, rot=0.0, pole=(960.0, 540.0), a=1.0, clip=None):
    """the star field rotated by rot (radians, clockwise on screen) about pole. clip(x, y) -> visibility 0..1"""
    if a <= 0:
        return
    sx, sy, b, size, tint, ph, om = STARS
    cr, sr = math.cos(rot), math.sin(rot)
    X = pole[0] + sx * cr - sy * sr
    Y = pole[1] + sx * sr + sy * cr
    vis = (X > -10) & (X < 1930) & (Y > -10) & (Y < 1090)
    idx = np.nonzero(vis)[0]
    tw = 1 + 0.18 * np.sin(om[idx] * t + ph[idx])
    for j, i in enumerate(idx):
        al = a * b[i] * tw[j]
        if clip is not None:
            al *= clip(X[i], Y[i])
        if al <= 0.01:
            continue
        col = (226 + 20 * tint[i], 230, 240 - 26 * tint[i]) if tint[i] > 0 else (238, 232 + 10 * tint[i], 228 + 20 * tint[i])
        col = tuple(int(v) for v in col)
        if size[i] > 1.6:
            dot_glow(c, X[i], Y[i], size[i] * 0.6, col, al * 0.9, halo=4.5)
        else:
            c.drawCircle(X[i], Y[i], size[i] * 0.62, paint(col, min(1.0, al)))


def sky(c, t=0.0, rot=0.0, pole=(960.0, 540.0), star_a=1.0, centre=(960, 600)):
    p = skia.Paint()
    p.setDither(True)
    p.setShader(skia.GradientShader.MakeRadial(centre, 1350, [C4(NIGHT1).toColor(), C4(NIGHT0).toColor()]))
    c.drawPaint(p)
    stars(c, t, rot, pole, star_a)


def vignette(c, a=0.5):
    p = skia.Paint()
    p.setDither(True)
    p.setShader(skia.GradientShader.MakeRadial((960, 540), 1150, [C4((0, 0, 0), 0).toColor(), C4((0, 0, 0), 0).toColor(),
                                                                  C4((0, 0, 0), a).toColor()], [0.0, 0.55, 1.0]))
    c.drawPaint(p)


def blackout(c, a):
    if a > 0:
        c.drawPaint(paint((0, 0, 0), a))


# ------------------------------------------------------------------ clocks
ROMAN = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]


def clock(c, cx, cy, r, h=None, m=None, s=None, a=1.0, n=12, numerals="roman", col=GOLD, minute_ticks=60,
          hand_a=(1.0, 1.0, 1.0), face=True, numeral_a=1.0, start_top=None):
    """an engraved clock face. h, m, s: hand angles as fractions of a turn (None hides a hand). n: hours on the dial."""
    if a <= 0:
        return
    if face:
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeRadial((cx, cy - r * 0.3), r * 1.4,
                                                   [C4((30, 38, 66), a).toColor(), C4((12, 16, 32), a).toColor()]))
        c.drawCircle(cx, cy, r, p)
    circle(c, cx, cy, r, col, a, w=2.2, glow=0.25)
    circle(c, cx, cy, r * 0.965, col, a * 0.6, w=1.0)
    if minute_ticks:
        ticks(c, cx, cy, r * 0.965, minute_ticks, r * 0.035, col, a * 0.85, w=1.0)
    ticks(c, cx, cy, r * 0.965, n, r * 0.075, col, a, w=2.2)
    if numerals and numeral_a > 0:
        for k in range(n):
            if numerals == "roman" and n == 12:
                s_ = ROMAN[k]
            elif numerals == "decimal":
                s_ = str(10 if k == 0 else k)
            elif numerals == "24":
                s_ = str(24 if k == 0 else k)
            else:
                s_ = str(n if k == 0 else k)
            x, y = polar(cx, cy, r * 0.79, 360.0 * k / n)
            text(c, x, y, s_, r * (0.13 if numerals == "roman" else 0.15), col, a * numeral_a, anchor="mm",
                 kind="serif" if numerals == "roman" else "sans-m")
    for frac, L, wd, ha in ((h, 0.52, 5.0, hand_a[0]), (m, 0.78, 3.4, hand_a[1])):
        if frac is None or ha <= 0:
            continue
        x, y = polar(cx, cy, r * L, 360.0 * frac)
        bx, by = polar(cx, cy, -r * 0.1, 360.0 * frac)
        line(c, bx, by, x, y, CREAM, a * ha, w=wd * r / 300, glow=0.25)
    if s is not None and hand_a[2] > 0:
        x, y = polar(cx, cy, r * 0.86, 360.0 * s)
        bx, by = polar(cx, cy, -r * 0.16, 360.0 * s)
        line(c, bx, by, x, y, RED, a * hand_a[2], w=1.6 * r / 300)
    circle(c, cx, cy, 6 * r / 300, CREAM, a)


# ------------------------------------------------------------------ frames and sections
_SURF = None


def canvas():
    """a fresh frame: returns (surface, canvas scaled to 1080p units)"""
    global _SURF
    if _SURF is None:
        _SURF = skia.Surface(W, H)
    c = _SURF.getCanvas()
    c.restoreToCount(1)
    c.resetMatrix()
    c.clear(skia.ColorBLACK)
    c.save()
    c.scale(S, S)
    return _SURF, c


def pixels(surf):
    return surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)


class Section:
    name, dur = "section", 1.0

    def draw(self, c, t):
        raise NotImplementedError

    def frame(self, t):
        surf, c = canvas()
        self.draw(c, t)
        vignette(c, 0.45)
        return pixels(surf)

    def _enc(self, path):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "14",
                                 "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            Image.fromarray(self.frame(still)[..., :3]).save(os.path.join(OUT, f"{self.name}_{still:05.1f}.png"))
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
            p = self._enc(tmp)
            for i in range(c0, c1):
                p.stdin.write(self.frame(i / FPS).tobytes())
            p.stdin.close(); p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q); os.remove(q + ".ok")
        print("wrote", path, flush=True)

    def patch(self, t0, t1):
        """re-render only the frames in [t0, t1) of an already rendered section"""
        path = os.path.join(OUT, f"{self.name}.mp4")
        tmp = path + ".patch.mp4"
        n = int(round(self.dur * FPS))
        dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgba", "-"], stdout=subprocess.PIPE)
        enc = self._enc(tmp)
        size = W * H * 4
        for i in range(n):
            buf = dec.stdout.read(size)
            if len(buf) < size:
                raise RuntimeError(f"{self.name}: only {i} frames in {path}")
            if t0 <= i / FPS < t1:
                buf = self.frame(i / FPS).tobytes()
            enc.stdin.write(buf)
        dec.stdout.close(); dec.wait()
        enc.stdin.close(); enc.wait()
        os.replace(tmp, path)
        print(f"patched {self.name} {t0}-{t1} s", flush=True)


def contact(sec, times, path, cols=4):
    """a contact sheet of stills, for checking a section at a glance"""
    ims = [Image.fromarray(sec.frame(t)[..., :3]).resize((480, 270), Image.LANCZOS) for t in times]
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 270))
    for k, im in enumerate(ims):
        d = ImageDraw.Draw(im)
        d.text((6, 4), f"{times[k]:.1f}s", fill=(255, 80, 80))
        sheet.paste(im, ((k % cols) * 480, (k // cols) * 270))
    sheet.save(path)
