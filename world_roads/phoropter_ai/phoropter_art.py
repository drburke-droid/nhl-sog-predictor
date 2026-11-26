"""A phoropter as the examiner sees it, drawn in code so every part can move with the exam.

The patient's right eye is on the examiner's left. Each lens battery has: a large knurled sphere wheel on the
outside, a sphere-power window above the aperture, a cylinder-power knob and window below, an axis scale ring
around the aperture (TABO: 0 at the examiner's right, 90 up), an auxiliary-lens knob on top (O = open,
OC = occluded, R = retinoscopy, P = polaroid), and a Jackson cross-cylinder on a swing arm that parks above the
aperture and swings in front of it. Rotary prisms swing in for binocular balance. The central bridge carries the
PD scale, a levelling vial, the forehead-rest knob and the hanging arm.

    draw(im, cx, cy, s, eyes=[(S, C, axis), (S, C, axis)], occluded=None|0|1, cur=0|1|None, jcc=None|(eye, angle),
         prism=False)
"""
import math
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import ui

SS = 2                                     # supersampling
U_W, U_H = 1060, 660                       # drawing units: x -530..530, y -390..270
OX, OY = 530, 390
AP_X, AP_Y, AP_R = 172, 20, 54             # aperture centre (x = +/-AP_X) and radius
RING_R = 84

BODY_HI, BODY_LO = (226, 230, 236), (176, 183, 193)
EDGE = (72, 80, 92)
DARK, DARK2 = (40, 44, 52), (62, 68, 78)
METAL_HI = (245, 247, 250)
GLASS = (12, 28, 40)
SCALE = (238, 238, 232)
INKD = (28, 30, 34)
RED_DOT, WHITE_DOT = (220, 56, 48), (246, 246, 246)
GOLD = ui.GOLD


def _catmull(pts, n=14):
    """Closed Catmull-Rom spline through pts."""
    out = []
    m = len(pts)
    for i in range(m):
        p0, p1, p2, p3 = pts[(i - 1) % m], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    return out


class Canvas:
    """Drawing in phoropter units on a supersampled RGBA patch."""

    def __init__(self, s):
        self.k = s * SS
        self.im = Image.new("RGBA", (int(U_W * self.k), int(U_H * self.k)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def p(self, x, y):
        return ((x + OX) * self.k, (y + OY) * self.k)

    def poly(self, pts, fill=None, outline=None, width=1):
        self.d.polygon([self.p(*q) for q in pts], fill=fill, outline=outline, width=max(1, int(width * self.k)))

    def circle(self, x, y, r, fill=None, outline=None, width=1):
        (a, b), (c, d) = self.p(x - r, y - r), self.p(x + r, y + r)
        self.d.ellipse([a, b, c, d], fill=fill, outline=outline, width=max(1, int(width * self.k)) if outline else 0)

    def line(self, pts, fill, width=1):
        self.d.line([self.p(*q) for q in pts], fill=fill, width=max(1, int(width * self.k)))

    def rect(self, x0, y0, x1, y1, r, fill=None, outline=None, width=1):
        (a, b), (c, d) = self.p(x0, y0), self.p(x1, y1)
        self.d.rounded_rectangle([a, b, c, d], int(r * self.k), fill=fill, outline=outline,
                                 width=max(1, int(width * self.k)) if outline else 0)

    def text(self, x, y, s, size, fill, weight="SemiBold", anchor="mm"):
        self.d.text(self.p(x, y), s, font=ui.font(max(1, size * self.k / 1.0), weight) if False else
                    ui.ST.font(max(6, int(size * self.k)), weight), fill=fill, anchor=anchor)

    def gradient_fill(self, pts, top, bottom):
        """Fill a polygon with a vertical gradient (via a mask)."""
        mask = Image.new("L", self.im.size, 0)
        ImageDraw.Draw(mask).polygon([self.p(*q) for q in pts], fill=255)
        h = self.im.size[1]
        g = np.linspace(0, 1, h)[:, None]
        col = (np.array(top)[None, None, :] * (1 - g[..., None]) + np.array(bottom)[None, None, :] * g[..., None])
        grad = Image.fromarray(np.repeat(col, self.im.size[0], axis=1).astype(np.uint8), "RGB").convert("RGBA")
        self.im.paste(grad, (0, 0), mask)
        self.d = ImageDraw.Draw(self.im, "RGBA")


def knurled(c, x, y, r, teeth=64, body=DARK2, rim=DARK):
    c.circle(x, y, r, fill=rim)
    for i in range(teeth):
        t = 2 * math.pi * i / teeth
        c.line([(x + (r - 5) * math.cos(t), y + (r - 5) * math.sin(t)), (x + r * math.cos(t), y + r * math.sin(t))],
               (120, 126, 136, 255), 1.4)
    c.circle(x, y, r - 7, fill=body)
    c.circle(x - r * 0.25, y - r * 0.3, r * 0.35, fill=(255, 255, 255, 22))


def battery_outline(sg):
    pts = [(70, -150), (190, -168), (300, -215), (420, -190), (492, -90), (500, 40), (470, 150), (380, 225),
           (240, 250), (110, 222), (66, 150), (58, 40)]
    return _catmull([(sg * x, y) for x, y in pts])


@lru_cache(8)
def static_layer(s):
    """Everything that never moves, at supersampled resolution (RGBA)."""
    c = Canvas(s)
    # hanging arm and top bar
    c.rect(-22, -392, 22, -150, 8, fill=(150, 158, 170, 255), outline=EDGE + (255,), width=2)
    c.rect(-96, -248, 96, -214, 14, fill=(190, 196, 205, 255), outline=EDGE + (255,), width=2)
    for sg in (-1, 1):
        out = battery_outline(sg)
        c.gradient_fill(out, BODY_HI, BODY_LO)
        c.poly(out, outline=EDGE + (255,), width=2.5)
    # central bridge
    bridge = _catmull([(-78, -170), (78, -170), (92, -40), (84, 170), (40, 210), (-40, 210), (-84, 170), (-92, -40)])
    c.gradient_fill(bridge, (206, 211, 219), (160, 168, 179))
    c.poly(bridge, outline=EDGE + (255,), width=2.5)
    # levelling vial
    c.rect(-46, -150, 46, -126, 10, fill=(214, 232, 206, 255), outline=EDGE + (255,), width=1.5)
    c.circle(2, -138, 8, fill=(250, 252, 240, 255))
    c.line([(-14, -150), (-14, -126)], EDGE + (255,), 1)
    c.line([(18, -150), (18, -126)], EDGE + (255,), 1)
    # PD scale
    c.rect(-56, -96, 56, -62, 6, fill=(30, 34, 40, 255))
    for i, v in enumerate(range(54, 75, 2)):
        x = -50 + i * 10
        c.line([(x, -94), (x, -84 if v % 10 == 0 else -88)], (230, 230, 230, 255), 1.2)
        if v % 10 == 0:
            c.text(x, -73, str(v), 9, (230, 230, 230, 255), "Medium")
    c.text(0, -106, "PD", 10, EDGE + (255,), "SemiBold")
    knurled(c, 0, -20, 26)
    # forehead rest knob
    knurled(c, 0, 150, 34)
    c.text(0, 102, "FOREHEAD", 9, EDGE + (255,), "SemiBold")
    for sg in (-1, 1):
        # sphere wheel (outer side)
        knurled(c, sg * 362, -20, 118, teeth=110, body=(78, 84, 96, 255), rim=(46, 50, 58, 255))
        for i in range(12):
            t = 2 * math.pi * i / 12
            c.line([(sg * 362 + 60 * math.cos(t), -20 + 60 * math.sin(t)),
                    (sg * 362 + 96 * math.cos(t), -20 + 96 * math.sin(t))], (150, 156, 168, 200), 2)
        c.circle(sg * 362, -20, 44, fill=(56, 60, 70, 255), outline=(120, 126, 136, 255), width=2)
        c.text(sg * 362, -20, "SPH", 14, (210, 214, 222, 255), "Bold")
        # strong-sphere lever
        c.rect(sg * 300 - 16, -196, sg * 300 + 16, -150, 8, fill=DARK + (255,))
        # cylinder knob
        knurled(c, sg * 262, 168, 40, teeth=56)
        c.text(sg * 262, 168, "CYL", 12, (210, 214, 222, 255), "Bold")
        # axis knob
        knurled(c, sg * 96, 170, 20, teeth=36)
        # axis scale ring around the aperture
        x0 = sg * AP_X
        c.circle(x0, AP_Y, RING_R + 4, fill=(120, 128, 140, 255))
        c.circle(x0, AP_Y, RING_R, fill=SCALE + (255,))
        for deg in range(0, 360, 5):
            t = math.radians(deg)
            long = deg % 30 == 0
            r0 = RING_R - (12 if long else 6)
            c.line([(x0 + r0 * math.cos(t), AP_Y - r0 * math.sin(t)),
                    (x0 + RING_R * math.cos(t), AP_Y - RING_R * math.sin(t))], INKD + (255,), 1.6 if long else 1)
        for deg in (0, 30, 60, 90, 120, 150, 180):
            t = math.radians(deg)
            r0 = RING_R - 21
            c.text(x0 + r0 * math.cos(t), AP_Y - r0 * math.sin(t), str(deg), 9, INKD + (255,), "SemiBold")
        c.circle(x0, AP_Y, AP_R + 3, fill=(60, 66, 76, 255))
        # sphere window above, cylinder window below the aperture
        c.rect(x0 + sg * 78 - 40, -152, x0 + sg * 78 + 40, -120, 6, fill=(22, 24, 28, 255),
               outline=(110, 116, 126, 255), width=1.5)
        c.rect(sg * 262 - 40, 108, sg * 262 + 40, 136, 6, fill=(22, 24, 28, 255), outline=(110, 116, 126, 255),
               width=1.5)
        # auxiliary lens knob
        kx = x0 + sg * 40
        c.circle(kx, -200, 28, fill=(70, 76, 88, 255), outline=(130, 136, 146, 255), width=2)
        for lab, ang in (("O", 90), ("OC", 30), ("R", 150), ("P", 210), ("PH", 330)):
            t = math.radians(ang)
            c.text(kx + 43 * math.cos(t), -200 - 43 * math.sin(t), lab, 9, EDGE + (255,), "Bold")
    return c


def draw(im, cx, cy, s, eyes, occluded=None, cur=None, jcc=None, prism=False, a=1.0):
    """Composite the phoropter onto PIL image im, centred on the bridge at (cx, cy), s = pixels per unit."""
    c = Canvas(s)
    c.im = static_layer(s).im.copy()
    c.d = ImageDraw.Draw(c.im, "RGBA")
    for e, sg in ((0, -1), (1, 1)):                      # right eye on the examiner's left
        x0 = sg * AP_X
        S, C, ax = eyes[e]
        # what is behind the aperture
        if occluded == e:
            c.circle(x0, AP_Y, AP_R, fill=(8, 9, 11, 255))
            c.text(x0, AP_Y, "OC", 16, (90, 96, 106, 255), "Bold")
        else:
            c.circle(x0, AP_Y, AP_R, fill=GLASS + (255,))
            c.circle(x0 - 16, AP_Y - 18, 22, fill=(255, 255, 255, 26))
            if C < 0:
                t = math.radians(ax)
                dx, dy = math.cos(t) * (AP_R - 4), -math.sin(t) * (AP_R - 4)
                c.line([(x0 - dx, AP_Y - dy), (x0 + dx, AP_Y + dy)], GOLD + (235,), 3)
        # axis pointer on the scale ring
        if C < 0:
            t = math.radians(ax)
            for sgn in (1, -1):
                px, py = x0 + sgn * (RING_R + 6) * math.cos(t), AP_Y - sgn * (RING_R + 6) * math.sin(t)
                c.circle(px, py, 6, fill=(200, 40, 40, 255))
        # aux knob pointer: O (open) or OC (occluded)
        ang = math.radians(30 if occluded == e else 90)
        kx = x0 + sg * 40
        c.line([(kx, -200), (kx + 22 * math.cos(ang), -200 - 22 * math.sin(ang))], (240, 240, 240, 255), 3)
        # power windows
        c.text(x0 + sg * 78, -136, ui.fmt_d(S), 15, (120, 240, 160, 255), "Bold")
        c.text(sg * 262, 122, ui.fmt_d(C, False) if C < 0 else "0.00", 13, (120, 240, 160, 255), "Bold")
        # tested eye
        if cur == e and occluded is not None:
            c.circle(x0, AP_Y, RING_R + 12, outline=GOLD + (230,), width=4)
        # Jackson cross-cylinder: parked above the aperture, or swung in front of it
        pivot = (x0 - sg * 44, -142)                       # hinge at the top inner corner of the battery
        if jcc is not None and jcc[0] == e:
            jx, jy, jr = x0, AP_Y, AP_R + 10
        else:
            jx, jy, jr = x0 - sg * 44, -252, 26              # parked, swung up
        c.line([pivot, (jx, jy)], (90, 96, 108, 255), 7)
        c.circle(pivot[0], pivot[1], 8, fill=DARK + (255,))
        c.circle(jx, jy, jr + 5, fill=(52, 56, 64, 255))
        c.circle(jx, jy, jr, fill=(170, 196, 210, 170) if (jcc is not None and jcc[0] == e) else (150, 170, 184, 255))
        if jcc is not None and jcc[0] == e:
            t = math.radians(jcc[1])
            for sgn in (1, -1):
                for f in (0.38, 0.7):
                    px, py = jx + sgn * math.cos(t) * jr * f, jy - sgn * math.sin(t) * jr * f
                    c.circle(px, py, 7.5, fill=RED_DOT + (255,))
                    qx, qy = jx + sgn * math.cos(t + math.pi / 2) * jr * f, jy - sgn * math.sin(t + math.pi / 2) * jr * f
                    c.circle(qx, qy, 7.5, fill=WHITE_DOT + (255,), outline=(90, 90, 90, 255), width=1)
        # rotary prism (balance)
        if prism:
            c.circle(x0, AP_Y, AP_R + 14, outline=(210, 214, 222, 255), width=6)
            for i in range(0, 360, 15):
                t = math.radians(i)
                c.line([(x0 + (AP_R + 9) * math.cos(t), AP_Y + (AP_R + 9) * math.sin(t)),
                        (x0 + (AP_R + 17) * math.cos(t), AP_Y + (AP_R + 17) * math.sin(t))], INKD + (255,), 1.2)
            c.text(x0, AP_Y + (AP_R - 14) * (1 if e == 0 else -1), "3Δ " + ("UP" if e == 0 else "DN"), 11,
                   (230, 230, 230, 255), "Bold")
    small = c.im.resize((int(U_W * s), int(U_H * s)), Image.LANCZOS)
    if a < 1:
        alpha = small.split()[3].point(lambda v: int(v * a))
        small.putalpha(alpha)
    im.paste(small, (int(cx - OX * s), int(cy - OY * s)), small)


if __name__ == "__main__":
    import os
    im, d = ui.new_frame()
    draw(im, 960, 330, 0.95, [(-2.75, -0.75, 4), (-3.0, -0.5, 2)], occluded=1, cur=0, jcc=(0, 49))
    im.save(os.path.join(ui.OUT, "test_phoropter_art.png"))
    im, d = ui.new_frame()
    draw(im, 960, 330, 0.95, [(-2.25, -0.75, 174), (-2.25, -0.75, 7)], occluded=None, cur=None, prism=True)
    im.save(os.path.join(ui.OUT, "test_phoropter_art2.png"))
