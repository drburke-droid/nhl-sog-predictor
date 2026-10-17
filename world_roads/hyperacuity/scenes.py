"""Sections of the hyperacuity video (all at 4K, laid out in 1080p units).

    python scenes.py NAME            -> out/NAME.mp4
    python scenes.py NAME 3.5 ...    -> stills
"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
import ui
from ui import W, H, S, GOLD, INK, INK2, MUTED, RED, CYAN, smooth, fade, text, Quiz, magnify, captions

LINE_W = 4                                                     # stimulus line thickness (1080p px)


# ================================================================== the hook: a 1-pixel step
def hook():
    ys, x0, x1, xs, ans = [400, 560, 720], 300, 1620, 1190, 2

    def draw(p, i, odd):
        y = ys[i]
        if odd:
            p.rect(x0, y, xs, y + LINE_W)
            p.rect(xs, y + 1, x1, y + 1 + LINE_W)
        else:
            p.rect(x0, y, x1, y + LINE_W)

    def reveal(fr, u, a):
        g = smooth((u - 0.3) / 0.4) * a
        return magnify(fr, lambda q: draw(q, ans, True), (xs, ys[ans] + 2.5), (xs, 900), 110, 14, g,
                       [("a 1-pixel step", 34, GOLD, "Bold")])

    return Quiz("hook", "One of these lines has a 1-pixel step", "Find it", (x0 - 4, 380, x1 + 4, 740),
                ["A", "B", "C"], [(220, y + 2) for y in ys], (xs - 70, ys[ans] - 40, xs + 70, ys[ans] + 44), ans,
                draw, reveal, look=4.0, show=3.4)


# ================================================================== title, rules
class Title(ui.Section):
    name, dur = "title", 9.0

    def frame(self, t):
        fr = ui.blank()
        # a long line drifts in; a tiny kink in it is the logo
        a = fade(t, 2.6, self.dur, 0.6, 0.6)
        p = ui.Patch(300, 600, 1620, 640)
        p.rect(300, 616, 960, 616 + LINE_W)
        p.rect(960, 617, 1620, 617 + LINE_W)
        p.paste(fr, alpha=a * 0.9)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        caps = [(0.2, 2.6, [("Found it? Your eyes can do far better than one pixel.", 44, INK, "SemiBold")], {"anchor": "top", "y": 470})]
        for t0, t1, lines, kw in caps:
            ui.caption(d, lines, fade(t, t0, t1), **kw)
        text(d, (960, 470), "HYPERACUITY", 120, GOLD, a, "Bold")
        text(d, (960, 700), "How your eyes see smaller than a pixel", 44, INK, fade(t, 3.4, self.dur, 0.6, 0.6), "SemiBold")
        return np.asarray(im).copy()


class Rules(ui.Section):
    name, dur = "rules", 12.0
    items = [("Watch full screen", "the bigger the video, the bigger each pixel"),
             ("Set quality to 1080p or higher", "tap the gear icon, then Quality"),
             ("6 rounds: find the pair that's off", "look straight at each one; every offset is half the last"),
             ("Keep score", "and tell me your last round in the comments")]

    def frame(self, t):
        fr = ui.blank()
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a0 = fade(t, 0.2, self.dur, 0.4, 0.5)
        d.rounded_rectangle([420 * S, 170 * S, 1500 * S, 910 * S], 28 * S, fill=ui.PANEL + (int(215 * a0),))
        text(d, (480, 230), "THE TEST", 30, GOLD, a0, "Bold", "la")
        for k, (h, s) in enumerate(self.items):
            a = fade(t, 0.8 + k * 1.3, self.dur, 0.45, 0.5)
            y = 310 + k * 140
            text(d, (480, y), f"{k + 1}", 46, GOLD, a, "Bold", "la")
            text(d, (545, y + 2), h, 40, INK, a, "SemiBold", "la")
            text(d, (545, y + 56), s, 28, INK2, a, "Regular", "la")
        text(d, (480, 870), "1 pixel = one pixel of a 1080p frame", 22, MUTED, fade(t, 6.0, self.dur, 0.5, 0.5), "Regular", "la")
        return np.asarray(im).copy()


# ================================================================== straightness and tilt
def sdf_line(cx, cy, L, ang_deg, w=LINE_W / 2):
    c, s = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))

    def f(X, Y):
        u = (X - cx) * c + (Y - cy) * s
        v = -(X - cx) * s + (Y - cy) * c
        return np.maximum(np.abs(v) - w, np.abs(u) - L / 2)
    return f


def bent():
    ys, L, cx, sag, ans = [400, 560, 720], 1300, 960, 2.0, 0

    def draw(p, i, odd):
        y0, s_ = ys[i] + 2, (sag if odd else 0.0)

        def f(X, Y):
            yc = y0 - s_ * (1 - ((X - cx) / (L / 2)) ** 2)            # a parabola, s_ px high in the middle
            return np.maximum(np.abs(Y - yc) - LINE_W / 2, np.abs(X - cx) - L / 2)
        p.sdf(f)

    def reveal(fr, u, a):
        # the bend, drawn 20 times taller beside the real line
        g = smooth((u - 0.3) / 0.5) * a
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        xs = np.linspace(cx - L / 2, cx + L / 2, 200)
        yy = ys[ans] + 2 - 10 * sag * (1 - ((xs - cx) / (L / 2)) ** 2)
        d.line([(x * S, (y - 0) * S) for x, y in zip(xs, yy)], fill=GOLD + (int(220 * g),), width=3 * S)
        text(d, (cx, ys[ans] - 75), f"the bend, drawn 10× taller: the real one is {sag:g} pixels", 30, GOLD, g, "SemiBold")
        return np.asarray(im).copy()

    return Quiz("bent", f"One line bows by {sag:g} pixels in the middle", "Which one?", (cx - L / 2 - 4, 380, cx + L / 2 + 4, 740),
                ["A", "B", "C"], [(220, y + 2) for y in ys], (cx - L / 2 - 30, ys[ans] - 40, cx + L / 2 + 30, ys[ans] + 44),
                ans, draw, reveal, look=5.0, show=4.0)


def circle():
    xs, cy, R0, sq, ans = [480, 960, 1440], 560, 190, 0.015, 1

    def draw(p, i, odd):
        a_, b_ = (R0 * (1 + sq / 2), R0 * (1 - sq / 2)) if odd else (R0, R0)

        def f(X, Y):
            rho = np.sqrt(((X - xs[i]) / a_) ** 2 + ((Y - cy) / b_) ** 2)
            return np.abs((rho - 1) * (a_ + b_) / 2) - LINE_W / 2
        p.sdf(f)

    def reveal(fr, u, a):
        g = smooth((u - 0.3) / 0.5) * a
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        k = 12                                                       # the squash, exaggerated 12x
        a_, b_ = R0 * (1 + k * sq / 2), R0 * (1 - k * sq / 2)
        d.ellipse([(xs[ans] - a_) * S, (cy - b_) * S, (xs[ans] + a_) * S, (cy + b_) * S], outline=GOLD + (int(200 * g),), width=3 * S)
        text(d, (xs[ans], cy + R0 + 120), f"{sq * 100:.1f}% wider than tall: {2 * R0 * sq:.1f} pixels", 30, GOLD, g, "SemiBold")
        text(d, (xs[ans], cy + R0 + 160), "(drawn 12× more squashed)", 24, INK2, g, "Regular")
        return np.asarray(im).copy()

    return Quiz("circle", "One of these circles is squashed", "Which one?", (xs[0] - R0 - 10, cy - R0 - 10, xs[-1] + R0 + 10, cy + R0 + 10),
                ["A", "B", "C"], [(x, cy + R0 + 60) for x in xs], (xs[ans] - R0 - 30, cy - R0 - 30, xs[ans] + R0 + 30, cy + R0 + 30),
                ans, draw, reveal, look=5.0, show=4.0)


def tilt(diagonal):
    base = 45.0 if diagonal else 0.0
    xs, cy, L, tl, ans = [480, 960, 1440], 560, 380, 0.75, (0 if diagonal else 2)

    def draw(p, i, odd):
        p.sdf(sdf_line(xs[i], cy, L, base + (tl if odd else 0.0)))

    def reveal(fr, u, a):
        g = smooth((u - 0.3) / 0.5) * a
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        c, s = math.cos(math.radians(base)), math.sin(math.radians(base))
        d.line([((xs[ans] - c * L / 2) * S, (cy - s * L / 2) * S), ((xs[ans] + c * L / 2) * S, (cy + s * L / 2) * S)],
               fill=CYAN + (int(180 * g),), width=2 * S)
        text(d, (xs[ans], cy + 260), f"tilted {tl}°", 32, GOLD, g, "Bold")
        text(d, (xs[ans], cy + 300), "blue: where it would lie, parallel to the others", 24, INK2, g, "Regular")
        return np.asarray(im).copy()

    if diagonal:
        title, sub = "Now the same tilt, on a diagonal", "Harder?"
    else:
        title, sub = f"One line is tilted {tl}°", "Which one?"
    return Quiz("tilt_diag" if diagonal else "tilt", title, sub, (xs[0] - 220, cy - 220, xs[-1] + 220, cy + 220),
                ["A", "B", "C"], [(x, cy + 210) for x in xs], (xs[ans] - 170, cy - 170, xs[ans] + 170, cy + 170),
                ans, draw, reveal, look=4.0, show=4.0)


# ================================================================== cards, results, caliper, Amsler grid, outro
class Card(ui.Section):
    """Lines of text, appearing one after another. lines: [(t0, text, size, colour, weight)]"""

    def __init__(self, name, dur, lines, y0=440, gap=80):
        self.name, self.dur, self.lines, self.y0, self.gap = name, dur, lines, y0, gap

    def frame(self, t):
        fr = ui.blank()
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        for k, (t0, s_, size, col, wt) in enumerate(self.lines):
            text(d, (960, self.y0 + k * self.gap), s_, size, col, fade(t, t0, self.dur, 0.5, 0.45), wt)
        return np.asarray(im).copy()


PHONE_PX_ARCSEC = 40.0          # one 1080p pixel, video full screen on a phone at 30 cm
CONE_ARCSEC = 30.0              # foveal cone spacing
VERNIER_ARCSEC = 5.0            # a typical vernier threshold


class Results(ui.Section):
    name, dur = "results", 15.0
    OFF = [4, 2, 1, 0.5, 0.25, 0.125]

    def frame(self, t):
        fr = ui.blank()
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a0 = fade(t, 0.2, self.dur, 0.5, 0.5)
        text(d, (960, 110), "How far did you get?", 56, INK, a0, "Bold")
        text(d, (960, 170), f"on a phone at 30 cm, full screen, one pixel of this video is about {PHONE_PX_ARCSEC:.0f} arcseconds", 26, INK2, a0, "Regular")
        y = 250
        for k, o in enumerate(self.OFF):
            a = fade(t, 0.8 + 0.5 * k, self.dur, 0.4, 0.5)
            arc = o * PHONE_PX_ARCSEC
            lab = f"{ui.frac(o)} pixel" if o <= 1 else f"{o:g} pixels"
            text(d, (700, y), f"Round {k + 1}", 34, MUTED, a, "SemiBold", "lm")
            text(d, (900, y), lab, 40, INK, a, "Bold", "lm")
            text(d, (1180, y), f"≈ {arc:.0f}″", 40, GOLD, a, "Bold", "lm")
            if k == 2:                                             # the cone spacing falls between 1 and 1/2 pixel
                a_c = fade(t, 4.8, self.dur, 0.5, 0.5)
                yy = y + 52
                d.line([(640 * S, yy * S), (1400 * S, yy * S)], fill=CYAN + (int(200 * a_c),), width=2 * S)
                text(d, (1420, yy), f"cone spacing ≈ {CONE_ARCSEC:.0f}″", 26, CYAN, a_c, "SemiBold", "lm")
                y += 30
            y += 92
        text(d, (960, 880), "On a phone, everything below the blue line is a shift smaller than the gap between two cones.", 30, INK,
             fade(t, 6.0, self.dur, 0.5, 0.5), "SemiBold")
        text(d, (960, 940), "Comment the last round you got right, and what you watched on.", 32, GOLD, fade(t, 8.5, self.dur, 0.5, 0.5), "Bold")
        return np.asarray(im).copy()


class Caliper(ui.Section):
    name, dur = "caliper", 27.0
    MM = 40.0                                                     # 1080p units per millimetre
    X0, Y_MAIN = 140.0, 430.0

    def reading(self, t):
        keys = [(0, 12.0), (6.5, 12.0), (8.0, 12.3), (14.5, 12.3), (16.0, 12.7), (27.0, 12.7)]
        for (t0, r0), (t1, r1) in zip(keys[:-1], keys[1:]):
            if t <= t1:
                return r0 + (r1 - r0) * smooth((t - t0) / max(t1 - t0, 1e-6))
        return keys[-1][1]

    def frame(self, t):
        fr = ui.blank()
        a = fade(t, 0.2, self.dur, 0.5, 0.5)
        r = self.reading(t)
        M, X0, Y = self.MM, self.X0, self.Y_MAIN
        p = ui.Patch(X0 - 20, Y - 90, X0 + 41 * M, Y + 140)
        for mm in range(0, 41):                                    # the ruler
            L = 60 if mm % 10 == 0 else (42 if mm % 5 == 0 else 28)
            p.rect(X0 + mm * M - 1.5, Y - L, X0 + mm * M + 1.5, Y)
        p.rect(X0 - 10, Y - 2, X0 + 40 * M + 10, Y)
        xv = X0 + r * M                                             # the vernier: 10 marks over 9 mm
        for k in range(11):
            L = 50 if k in (0, 5, 10) else 34
            p.rect(xv + k * 0.9 * M - 1.5, Y + 6, xv + k * 0.9 * M + 1.5, Y + 6 + L)
        p.rect(xv - 10, Y + 6, xv + 9 * M + 10, Y + 8)
        p.paste(fr, alpha=a)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        for mm in range(0, 41, 10):
            text(d, (X0 + mm * M, Y - 90), f"{mm // 10}", 30, INK2, a, "SemiBold")
        text(d, (X0 + 40 * M + 40, Y - 30), "cm", 26, MUTED, a, "Regular", "lm")
        tenth = (r - math.floor(r)) * 10
        k = int(round(tenth)) % 10
        settled = abs(tenth - round(tenth)) < 0.02
        if settled:                                                 # the one pair that lines up
            xa = xv + k * 0.9 * M
            d.ellipse([(xa - 26) * S, (Y - 34) * S, (xa + 26) * S, (Y + 46) * S], outline=GOLD + (int(255 * a),), width=4 * S)
            text(d, (960, 660), f"{r:.1f} mm", 72, GOLD, a, "Bold")
            text(d, (960, 730), f"{math.floor(r):.0f} whole mm, and mark {k} lines up: {k} tenths", 30, INK2, a, "Regular")
        fr = np.asarray(im).copy()
        caps = [(0.3, 7.4, [("Engineers have used this for 400 years.", 38, INK, "SemiBold"),
                            ("The vernier scale, Pierre Vernier, 1631", 34, GOLD, "Bold")]),
                (8.2, 15.8, [("Ten marks on the slider span nine on the ruler,", 36, INK, "SemiBold"),
                             ("so at any position only one pair lines up.", 36, INK2, "Regular")]),
                (16.4, 26.6, [("You never measure the tenth of a millimetre.", 36, INK, "SemiBold"),
                              ("You just spot which pair lines up: the thing your eyes do best.", 36, GOLD, "SemiBold")])]
        return captions(fr, t, [(c[0], c[1], c[2], {}) for c in caps])


class Amsler(ui.Section):
    name, dur = "amsler", 36.0
    N, SQ, CX, CY = 20, 30.0, 960.0, 440.0

    def warp(self, X, Y, amp):
        """a local bulge, as in metamorphopsia"""
        dx, dy = X - (self.CX + 70), Y - (self.CY - 40)
        g = amp * np.exp(-(dx * dx + dy * dy) / (2 * 70.0 ** 2)) / 70.0
        return X + g * dx, Y + g * dy

    def frame(self, t):
        fr = ui.blank()
        a = fade(t, 0.2, self.dur, 0.6, 0.6)
        amp = 24.0 * smooth((t - 11.0) / 2.5) * (1 - smooth((t - 27.5) / 2.0))
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        half = self.N * self.SQ / 2
        s_ = np.linspace(-half, half, 400)
        for k in range(self.N + 1):
            c = -half + k * self.SQ
            for X, Y in ((self.CX + c + 0 * s_, self.CY + s_), (self.CX + s_, self.CY + c + 0 * s_)):
                Xw, Yw = self.warp(X, Y, amp)
                d.line([(x * S, y * S) for x, y in zip(Xw, Yw)], fill=(235, 238, 245, int(235 * a)), width=2 * S)
        d.ellipse([(self.CX - 6) * S, (self.CY - 6) * S, (self.CX + 6) * S, (self.CY + 6) * S], fill=GOLD + (int(255 * a),))
        if amp > 0.5:
            text(d, (self.CX + half + 40, self.CY - 40), "simulated", 26, MUTED, a * smooth((t - 11.0) / 1.0), "Regular", "lm")
        fr = np.asarray(im).copy()
        caps = [(0.4, 10.4, [("Eye doctors use this sensitivity too: the Amsler grid (Marc Amsler, 1947).", 36, INK, "SemiBold"),
                             ("Cover one eye and look at the dot. Every line should be straight.", 34, INK2, "Regular")]),
                (11.2, 21.6, [("Lines that look wavy, bent or missing can be an early sign", 36, INK, "SemiBold"),
                              ("of a problem in the macula, such as macular degeneration.", 36, INK, "SemiBold")]),
                (22.2, 28.6, [("Some at-home monitors for macular degeneration", 36, INK, "SemiBold"),
                              ("are built on this same hyperacuity.", 36, INK2, "Regular")]),
                (29.2, 35.6, [("If straight lines ever look bent,", 40, INK, "SemiBold"),
                              ("see your optometrist.", 48, GOLD, "Bold")])]
        return captions(fr, t, [(c[0], c[1], c[2], {}) for c in caps])


class Outro(ui.Section):
    name, dur = "outro", 18.0

    def frame(self, t):
        fr = ui.blank()
        a = fade(t, 0.2, self.dur, 0.6, 1.2)
        p = ui.Patch(300, 330, 1620, 360)
        p.rect(300, 340, 960, 340 + LINE_W)
        p.rect(960, 340.125, 1620, 340.125 + LINE_W)              # a 1/8-pixel step, to close
        p.paste(fr, alpha=a)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        text(d, (960, 200), "This line has a ⅛-pixel step in the middle.", 44, INK, a, "Bold")
        text(d, (960, 260), "Can you see it?", 36, GOLD, a, "SemiBold")
        text(d, (960, 440), "Comment your last round, and what you watched on.", 34, INK2, fade(t, 3.0, self.dur, 0.5, 1.2), "SemiBold")
        return np.asarray(im).copy()


SECTIONS = {"hook": hook, "title": Title, "rules": Rules, "bent": bent, "circle": circle,
            "tilt": lambda: tilt(False), "tilt_diag": lambda: tilt(True), "results": Results, "caliper": Caliper,
            "amsler": Amsler, "outro": Outro,
            "straight": lambda: Card("straight", 6.0, [(0.3, "It isn't just misalignment.", 54, INK, "Bold"),
                                                       (1.6, "Your eyes catch bends, squashes and tilts the same way.", 38, INK2, "SemiBold")]),
            "amblyopia": lambda: Card("amblyopia", 9.0, [(0.3, "One more thing eye doctors know:", 40, INK2, "SemiBold"),
                                                          (1.4, "in a lazy eye caused by an eye turn (strabismic amblyopia),", 40, INK, "SemiBold"),
                                                          (2.6, "vernier acuity suffers far more than letter-chart acuity.", 44, GOLD, "Bold")]),
            "oblique": lambda: Card("oblique", 7.0, [(0.3, "Same tilt, but the diagonal one is harder.", 50, INK, "Bold"),
                                                     (1.8, "Vision is sharpest for horizontal and vertical:", 38, INK2, "SemiBold"),
                                                     (2.6, "the 'oblique effect'.", 44, GOLD, "Bold")])}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
