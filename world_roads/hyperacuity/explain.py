"""Explainer sections: the scale of things (cones, a 20/20 letter, a phone pixel, the vernier limit), how a
screen draws between pixels, and a simulated retina that finds a line's centre to a fraction of a cone.

    python explain.py scale|aa|retina [still times...]
"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
import ui
from ui import W, H, S, GOLD, INK, INK2, MUTED, CYAN, smooth, fade, text, captions

CONE_ARCMIN = 0.5          # centre-to-centre spacing of foveal cones (arcmin)
VERNIER_ARCSEC = 5.0       # a typical vernier threshold (arcsec)
PHONE_PX_ARCSEC = 40.0     # one pixel of this video, full screen on a phone at 30 cm (arcsec)


def lsf(x):
    """Westheimer's line spread function of the human eye, x in arcmin."""
    return 0.47 * np.exp(-3.3 * x * x) + 0.53 * np.exp(-0.93 * np.abs(x))


def mosaic(x0, y0, x1, y1, sp, seed=3, jitter=0.03):
    """Hexagonal cone lattice (1080p units) with a little disorder."""
    rng = np.random.default_rng(seed)
    h = sp * math.sqrt(3) / 2
    pts = []
    r = 0
    y = y0
    while y < y1 + h:
        x = x0 + (r % 2) * sp / 2 - sp
        while x < x1 + sp:
            pts.append((x + rng.normal(0, jitter * sp), y + rng.normal(0, jitter * sp), r))
            x += sp
        y += h
        r += 1
    return np.array(pts)


def draw_cones(d, pts, sp, fill_lum=None, a=1.0, outline=(70, 82, 104)):
    rr = sp * 0.44
    for k, (x, y, _) in enumerate(pts):
        v = 0.0 if fill_lum is None else float(fill_lum[k])
        col = tuple(int(c) for c in (np.array([22, 26, 36]) * (1 - v) + np.array([255, 205, 120]) * v))
        d.ellipse([(x - rr) * S, (y - rr) * S, (x + rr) * S, (y + rr) * S], fill=col + (int(255 * a),),
                  outline=outline + (int(200 * a),), width=max(1, int(1.5 * S)))


# ================================================================== the scale of things
class Scale(ui.Section):
    name, dur = "scale", 40.0
    PX = 120.0                                    # 1080p units per arcminute

    def __init__(self):
        sp = CONE_ARCMIN * self.PX
        self.sp = sp
        self.pts = mosaic(-sp, -sp, W / S + sp, H / S + sp, sp)

    def frame(self, t):
        fr = ui.blank()
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a_m = fade(t, 0.2, self.dur, 1.0, 0.6) * 0.75
        draw_cones(d, self.pts, self.sp, a=a_m)
        P = self.PX
        # a 20/20 letter E: 5 arcmin tall, strokes of 1 arcmin
        a_e = fade(t, 6.0, 14.6, 0.6, 0.5)
        if a_e > 0:
            ex, ey = 520, 110
            for (cx, cy, w_, h_) in ((0, 0, 1, 5), (1, 0, 4, 1), (1, 2, 3, 1), (1, 4, 4, 1)):
                d.rectangle([(ex + cx * P) * S, (ey + cy * P) * S, (ex + (cx + w_) * P) * S, (ey + (cy + h_) * P) * S],
                            fill=(240, 244, 250, int(170 * a_e)), outline=GOLD + (int(255 * a_e),), width=2 * S)
            text(d, (ex + 2.5 * P, ey + 5 * P + 40), "a 20/20 letter: strokes 1 arcminute wide", 30, GOLD, a_e, "SemiBold")
        # one pixel of this video on a phone
        a_p = fade(t, 14.6, 22.6, 0.6, 0.5)
        if a_p > 0:
            s_ = PHONE_PX_ARCSEC / 60 * P
            px, py = 960, 400
            d.rectangle([(px - s_ / 2) * S, (py - s_ / 2) * S, (px + s_ / 2) * S, (py + s_ / 2) * S],
                        fill=(240, 244, 250, int(200 * a_p)), outline=CYAN + (int(255 * a_p),), width=2 * S)
            text(d, (px, py + s_ / 2 + 50), f"one pixel of this video on a phone: about {PHONE_PX_ARCSEC:.0f}″", 30, CYAN, a_p, "SemiBold")
        # the vernier limit: two bars offset by 5 arcsec
        a_v = fade(t, 22.6, self.dur, 0.6, 0.6)
        if a_v > 0:
            off = VERNIER_ARCSEC / 60 * P
            vx, vy, bw, bh = 1340, 400, 0.25 * P, 2.2 * P
            d.rectangle([(vx - bw / 2) * S, (vy - bh - 6) * S, (vx + bw / 2) * S, (vy - 6) * S], fill=GOLD + (int(255 * a_v),))
            d.rectangle([(vx - bw / 2 + off) * S, (vy + 6) * S, (vx + bw / 2 + off) * S, (vy + bh + 6) * S], fill=GOLD + (int(255 * a_v),))
            text(d, (vx, vy + bh + 50), f"a {VERNIER_ARCSEC:.0f}″ shift: visible", 30, GOLD, a_v, "SemiBold")
        fr = np.asarray(im).copy()
        caps = [
            (1.0, 6.0, [("The centre of your retina, magnified:", 40, INK, "SemiBold"),
                        (f"light sensors (cones) about {CONE_ARCMIN:g} arcminute apart", 40, GOLD, "Bold")]),
            (6.6, 14.4, [("Normal vision resolves detail about 1 arcminute across:", 36, INK, "SemiBold"),
                         ("two cones' worth", 36, INK2, "Regular")]),
            (15.0, 22.4, [("A pixel of this video on your phone is smaller than that,", 36, INK, "SemiBold"),
                          ("so 1 pixel was easy", 36, INK2, "Regular")]),
            (23.0, 31.0, [("Yet most people can see a misalignment of 5 to 10 arcseconds:", 36, INK, "SemiBold"),
                          ("a third to a sixth of the space between two cones", 36, GOLD, "Bold")]),
            (31.6, 39.6, [("Seeing finer than your own sensors has a name:", 36, INK, "SemiBold"),
                          ("hyperacuity", 52, GOLD, "Bold"),
                          ("named by Gerald Westheimer in 1975", 28, INK2, "Regular")]),
        ]
        return captions(fr, t, [(c[0], c[1], c[2], {}) for c in caps])


# ================================================================== how a screen draws between pixels
class AntiAlias(ui.Section):
    name, dur = "aa", 17.0
    N, P, X0, Y0 = 10, 130, 310, 300
    STEPS = [3.0, 3.25, 3.5, 3.75, 4.0]

    def pos(self, t):
        k = (t - 3.0) / 2.2
        i = int(np.clip(math.floor(k), 0, len(self.STEPS) - 1))
        j = min(i + 1, len(self.STEPS) - 1)
        u = smooth((k - i - 0.75) / 0.25) if k > i + 0.75 else 0.0
        return self.STEPS[0] if k < 0 else self.STEPS[i] + (self.STEPS[j] - self.STEPS[i]) * u

    def frame(self, t):
        fr = ui.blank()
        a = fade(t, 0.2, self.dur, 0.5, 0.5)
        p0 = self.pos(t)
        q = ui.Patch(0, 0, self.N, 1, s=1)
        q.rect(p0, 0, p0 + 2, 1)                     # a bar 2 pixels wide
        cov = q.cov[0]
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        lin = ui.BG_LIN[0] * (1 - cov) + cov * ui.FG_LUM
        g = ui.to_srgb(lin)
        for i in range(self.N):
            x = self.X0 + i * self.P
            v = int(g[i])
            d.rectangle([x * S, self.Y0 * S, (x + self.P - 6) * S, (self.Y0 + self.P) * S], fill=(v, v, v, int(255 * a)),
                        outline=(70, 82, 104, int(255 * a)), width=2 * S)
            hb = 200 * cov[i]
            d.rectangle([(x + 20) * S, (800 - hb) * S, (x + self.P - 26) * S, 800 * S], fill=GOLD + (int(230 * a),))
            text(d, (x + self.P / 2 - 3, 840), f"{cov[i] * 100:.0f}%", 26, INK2, a, "Regular")
        cx = self.X0 + (p0 + 1) * self.P - 3
        d.polygon([(cx * S, (self.Y0 - 14) * S), ((cx - 16) * S, (self.Y0 - 44) * S), ((cx + 16) * S, (self.Y0 - 44) * S)],
                  fill=CYAN + (int(255 * a),))
        text(d, (cx, self.Y0 - 70), f"line centre: {p0 + 1:.2f} px", 30, CYAN, a, "SemiBold")
        text(d, (self.X0 - 20, 700), "light in", 26, MUTED, a, "Regular", "rm")
        text(d, (self.X0 - 20, 735), "each pixel", 26, MUTED, a, "Regular", "rm")
        fr = np.asarray(im).copy()
        caps = [(0.4, 6.4, [("A screen can only light whole pixels.", 40, INK, "SemiBold"),
                            ("So how did it draw a line a quarter pixel over?", 36, INK2, "Regular")], {"anchor": "top", "y": 900 - 0}),
                (6.8, 16.6, [("It moves part of the light into the next pixel.", 40, INK, "SemiBold"),
                             ("Your eye reads that balance, and sees the line move.", 36, GOLD, "SemiBold")], {"anchor": "top", "y": 900})]
        return captions(fr, t, caps)


# ================================================================== the retina does the same
class Retina(ui.Section):
    name, dur = "retina", 36.0
    PX = 200.0                                    # 1080p units per arcminute
    XC, Y_GAP = 960.0, 350.0

    def __init__(self):
        sp = CONE_ARCMIN * self.PX
        self.sp = sp
        self.pts = mosaic(130, 90, 1790, 610, sp, seed=5)
        self.pts = self.pts[(self.pts[:, 1] > 90) & (self.pts[:, 1] < 610) & (self.pts[:, 0] > 120) & (self.pts[:, 0] < 1800)]
        self.upper = self.pts[:, 1] < self.Y_GAP
        xs = np.linspace(-3, 3, 6001)
        self.grid = xs

    def delta(self, t):
        """the lower half's shift, arcsec"""
        if t < 6:
            return 0.0
        if t < 12:
            return 30.0 * smooth((t - 6) / 3)
        if t < 19:
            return 30.0 - (30.0 - VERNIER_ARCSEC) * smooth((t - 14) / 3)
        return VERNIER_ARCSEC

    def excitations(self, dlt):
        x = (self.pts[:, 0] - self.XC) / self.PX                  # arcmin from the upper line
        shift = np.where(self.upper, 0.0, dlt / 60)
        return lsf(x - shift)

    def estimate(self, e, mask):
        """the line position that best explains these cones' signals (least squares over a fine grid)"""
        x = (self.pts[mask, 0] - self.XC) / self.PX
        near = np.abs(x) < 2.0
        xs = self.grid[(self.grid > -0.6) & (self.grid < 0.6)]
        err = ((lsf(x[near][None, :] - xs[:, None]) - e[mask][near][None, :]) ** 2).sum(1)
        return xs[int(np.argmin(err))]

    def frame(self, t):
        fr = ui.blank()
        a = fade(t, 0.2, self.dur, 0.6, 0.6)
        dlt = self.delta(t)
        e = self.excitations(dlt)
        # the light falling on the retina: the blurred image of the two half-lines
        x0, y0, x1, y1 = 120, 60, 1800, 640
        X = (np.arange(int(x0 * S), int(x1 * S)) / S - self.XC) / self.PX
        Y = np.arange(int(y0 * S), int(y1 * S)) / S
        prof_u, prof_l = lsf(X), lsf(X - dlt / 60)
        img = np.where((Y < self.Y_GAP)[:, None], prof_u[None, :], prof_l[None, :])
        glow = (img[..., None] * np.array([255, 170, 70]) * 0.30 * a).astype(np.float32)
        sub = fr[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)].astype(np.float32)
        fr[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)] = np.clip(sub + glow, 0, 255).astype(np.uint8)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        draw_cones(d, self.pts, self.sp, e ** 1.3 * smooth((t - 2.0) / 1.0), a)
        d.line([(x0 * S, self.Y_GAP * S), (x1 * S, self.Y_GAP * S)], fill=(90, 100, 120, int(120 * a)), width=2 * S)
        # what the cones say
        a_e = fade(t, 8.0, self.dur, 0.8, 0.6)
        if a_e > 0:
            eu, el = self.estimate(e, self.upper), self.estimate(e, ~self.upper)
            est = (el - eu) * 60
            for yy, xm, col, lab in ((700, eu, INK2, "upper line"), (760, el, GOLD, "lower line")):
                xp = self.XC + xm * self.PX * 6                        # positions drawn 6x enlarged
                d.line([((self.XC - 330) * S, yy * S), ((self.XC + 330) * S, yy * S)], fill=(60, 70, 90, int(255 * a_e)), width=2 * S)
                d.polygon([(xp * S, (yy - 4) * S), ((xp - 14) * S, (yy - 28) * S), ((xp + 14) * S, (yy - 28) * S)], fill=col + (int(255 * a_e),))
                text(d, (self.XC - 350, yy - 8), lab, 26, col, a_e, "SemiBold", "rm")
            text(d, (self.XC + 350, 715), f"shift worked out from the cones: {est:.1f}″", 30, GOLD, a_e, "Bold", "lm")
            text(d, (self.XC + 350, 755), f"(true shift {dlt:.1f}″; marks drawn 6× farther apart)", 24, INK2, a_e, "Regular", "lm")
        fr = np.asarray(im).copy()
        caps = [(0.4, 7.6, [("Your eye has the same problem: its 'pixels' are cones.", 38, INK, "SemiBold"),
                            ("And its optics blur even a fine line across several of them.", 34, INK2, "Regular")]),
                (8.2, 18.0, [("Shift the lower line, and the balance between neighbouring cones shifts too.", 36, INK, "SemiBold"),
                             ("Compare the two rows, and the shift can be worked out far finer than the cones.", 32, INK2, "Regular")]),
                (19.0, 27.0, [(f"A {VERNIER_ARCSEC:.0f}-arcsecond shift: a sixth of the space between cones.", 38, GOLD, "Bold"),
                              ("The information is there. Your brain reads it.", 34, INK, "SemiBold")]),
                (27.6, 35.6, [("Here a little blur helps: it spreads each line over several cones.", 34, INK, "SemiBold"),
                              ("A line thinner than a cone could move inside one, unnoticed.", 34, INK2, "Regular")])]
        return captions(fr, t, [(c[0], c[1], c[2], {}) for c in caps])


SECTIONS = {"scale": Scale, "aa": AntiAlias, "retina": Retina}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
