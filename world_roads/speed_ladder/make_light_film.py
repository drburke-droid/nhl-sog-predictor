"""The whole film, told toward light: how fast is light, worked up to through 120 years of machines on the desert road.
  RACE=1 PARKER=1 OUTSUF=_pk python make_light_film.py -> /home/user/lot/out/history_light_film.mp4
  ... make_light_film.py test <part> -> a few frames of one part as PNG (out/test_film)

An opening; part 1, every machine at real speed (history_realtime.mp4); part 2, 48 times slower (history_slow48.mp4);
part 3, the race (history_race_parker.mp4); each with captions in the band below the picture that keep light in view
(how many times faster it is; how far it goes in a second). Then what light is and how it comes into the race: its
speed drawn to scale against the machines, round the Earth 7.5 times a second, the plan (a pulse lapping the Earth,
a line in the sky per lap) and how time will run (slow motion, and twice time stopped to watch light pass). Part 4,
the race with light (history_race_light.mp4, its own signs and captions). The ending: how long each would take to
reach the Moon, light going there in real time."""
import math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, fly_sound as S, slowmo as SM, captions as CAP, make_cuts as MC, make_film as MFi

OUT = "/home/user/lot/out"
FPS = 25
WHITE, DIM, CYAN, ACCENT = SM.WHITE, SM.DIM, SM.CYAN, SM.ACCENT
INK = (220, 226, 234)                       # the machines in the explainer's charts: one quiet colour, light in cyan
LIGHT_MPH = 670616629
MACHINES = [("Wright Flyer", 30), ("Spitfire", 370), ("Bell X-1", 807), ("F-35", 1228), ("X-15", 4520),
            ("Parker Solar Probe", 430000)]
MOON_MI = 238855


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def fade(t, t0, t1, d=0.4):
    """1 between t0 and t1, ramping over d seconds at each end"""
    return min(ease((t - t0) / d), ease((t1 - t) / d))


# ---- drawing

def text(d, xy, s, size, weight="SemiBold", col=WHITE, a=1.0, anchor="mm"):
    """s with [bracketed] runs in cyan; xy its anchor point (mm: centred)"""
    if a <= 0: return
    f = C.font(size, weight)
    runs = CAP._runs(s)
    w = sum(d.textlength(t, font=f) for t, _ in runs)
    x, y = xy
    x = x - w / 2 if anchor[0] == "m" else x - w if anchor[0] == "r" else x
    for t, c in runs:
        c = col if c == WHITE else c
        d.text((x, y), t, font=f, fill=c + (int(255 * a),), anchor="l" + anchor[1])
        x += d.textlength(t, font=f)


def glow_text(img, xy, s, size, weight, col, a, radius=22):
    """big words with a glow round them (light)"""
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    text(ImageDraw.Draw(lay), xy, s, size, weight, col, a)
    g = lay.filter(ImageFilter.GaussianBlur(radius))
    img.alpha_composite(g); img.alpha_composite(g); img.alpha_composite(lay)


def canvas(bg=None):
    im = Image.fromarray(bg if bg is not None else np.zeros((C.H, C.W, 3), np.uint8)).convert("RGBA")
    return im, Image.new("RGBA", im.size, (0, 0, 0, 0))


def done(im, ov):
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


_ROAD = {}


def road(n, dim=0.45):
    """the empty road (as the film's title), darkened under the words"""
    if n not in _ROAD:
        if len(_ROAD) > 170: _ROAD.clear()
        st = C.props("/home/user/lot/render/fly_wright/static.npz")
        img = C.over_props(C.plate_loop(n), C.plate_match(st, 300 + n))
        img = C.grain(C.bloom(img), 300 + n)
        o = np.zeros((C.H, C.W, 3), np.uint8)
        o[C.TOP:C.TOP + C.PH] = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
        _ROAD[n] = o
    return (_ROAD[n].astype(np.float32) * dim).astype(np.uint8)


# ---- cards

def opening(T, dur, n0=0):
    """T: when each of the voice's lines starts (s); dur: how long the card lasts"""
    nf = int(round(dur * FPS))
    out = []
    for i in range(nf):
        t = i / FPS
        a = min(1.0, (nf - 1 - i) / 10)
        im, ov = canvas(road(n0 + i, 0.5 + 0.2 * (1 - ease(t / 1.5))))
        glow_text(im, (C.W / 2, C.H / 2 - 60), "How fast is [light]?", 124, "Black", WHITE, ease((t - 0.3) / 0.8) * a, 16)
        out.append(done(im, ov))
    return out


def part_card(num, title, sub, n0=0, seconds=3.8):
    nf = int(seconds * FPS)
    out = []
    for i in range(nf):
        t = i / FPS
        a = min(ease(t / 0.4), (nf - 1 - i) / 10)
        im, ov = canvas(road(n0 + i, 0.4))
        d = ImageDraw.Draw(ov)
        text(d, (C.W / 2, C.H / 2 - 120), f"PART {num} OF 4", 30, "Bold", CYAN, a)
        text(d, (C.W / 2, C.H / 2 - 30), title, 112, "Black", WHITE, a)
        if sub: text(d, (C.W / 2, C.H / 2 + 80), sub, 40, "Medium", WHITE, ease((t - 0.5) / 0.5) * a)
        out.append(done(im, ov))
    return out


def light_intro(T, dur):
    """Parker, then light"""
    nf = int(round(dur * FPS))
    tl = T["e2"] - 0.3
    out = []
    for i in range(nf):
        t = i / FPS
        im, ov = canvas()
        d = ImageDraw.Draw(ov)
        text(d, (C.W / 2, C.H / 2 - 30), "Parker is the fastest machine ever built.", 60, "Bold", WHITE,
             fade(t, T["e1"], tl - 0.2))
        text(d, (C.W / 2, C.H / 2 + 50), "Light is in another league.", 40, "Medium", DIM, fade(t, T["e1"] + 2.8, tl - 0.2))
        a = fade(t, tl, dur + 1, 0.5) * min(1.0, (nf - 1 - i) / 8)
        glow_text(im, (C.W / 2, C.H / 2 - 110), "LIGHT", 210, "Black", CYAN, a, 28)
        text(d, (C.W / 2, C.H / 2 + 60), "670,616,629 mph", 76, "Bold", WHITE, fade(t, tl + 0.4, dur + 1))
        text(d, (C.W / 2, C.H / 2 + 150), "186,282 miles every second", 40, "Medium", DIM, fade(t, T["e2"] + 3.0, dur + 1))
        out.append(done(im, ov) if a > 0 or t < 3 else done(im, ov))
    for i in range(1, 9):                               # last words fade with the rest
        out[-i] = (out[-i].astype(np.float32) * (i / 9)).astype(np.uint8)
    return out


def speed_chart(T, dur):
    """the speeds as bars to scale: first against Parker (the planes barely show), then light comes in and the scale
    pulls out until it fits, and Parker shrinks to less than a pixel"""
    nf = int(round(dur * FPS))
    tl = T["e5"] + 1.1                                  # light comes in as she says so
    X0, X1 = 640, 1760
    Wd = X1 - X0
    y0, step, bh = 300, 82, 30
    out = []
    for i in range(nf):
        t = i / FPS
        grow = ease(t / 1.3)
        u = ease((t - tl - 0.6) / 2.4)
        vmax = math.exp(math.log(430000) * (1 - u) + math.log(LIGHT_MPH) * u)
        im, ov = canvas()
        d = ImageDraw.Draw(ov)
        a_all = min(1.0, (nf - 1 - i) / 10)
        text(d, (C.W / 2, 170), "Speeds, drawn to scale", 56, "Bold", WHITE, a_all)
        for r, (name, v) in enumerate(MACHINES + [("Light", LIGHT_MPH)]):
            y = y0 + r * step
            is_light = name == "Light"
            ra = ease((t - tl) / 0.4) * a_all if is_light else a_all
            if ra <= 0: continue
            text(d, (X0 - 30, y), name, 34, "SemiBold", WHITE, ra, "rm")
            w = Wd * v / vmax * (ease((t - tl) / 0.6) if is_light else grow)
            w_draw = min(w, C.W - X0 + 10)
            col = CYAN if is_light else INK
            if w_draw >= 8:
                d.rounded_rectangle((X0, y - bh / 2, X0 + w_draw, y + bh / 2), 4, fill=col + (int(255 * ra),))
            elif w_draw > 0:                            # thinner than a pixel: as faint as its share of one
                d.rectangle((X0, y - bh / 2, X0 + max(1, w_draw), y + bh / 2),
                            fill=col + (int(255 * ra * min(1.0, w_draw)),))
            label = f"{v:,} mph"
            if is_light:
                f = C.font(30, "Bold")
                lw = d.textlength(label, font=f)
                xl = min(X0 + w_draw - 18, C.W - 60) - lw
                if xl > X0 + 10: d.text((xl, y), label, font=f, fill=(6, 14, 28, int(255 * ra)), anchor="lm")
            else:
                f = C.font(28, "Medium")
                lw = d.textlength(label, font=f)
                if X0 + w_draw + 16 + lw > C.W - 40:         # no room past the bar's end: inside it
                    d.text((X0 + w_draw - 16 - lw, y), label, font=C.font(28, "Bold"), fill=(6, 14, 28, int(255 * ra)),
                           anchor="lm")
                else:
                    text(d, (X0 + max(w_draw, 0) + 16, y), label, 28, "Medium", DIM, ra, "lm")
        d.line((X0, y0 - 40, X0, y0 + 6 * step + 40), fill=DIM + (int(140 * a_all),), width=2)
        yb = y0 + 6 * step + 64
        text(d, (X0, yb), "0", 24, "Medium", DIM, a_all, "mm")
        text(d, (X1, yb), f"{vmax:,.0f} mph", 24, "Medium", DIM, a_all, "rm")
        d.line((X1, yb - 26, X1, yb - 14), fill=DIM + (int(140 * a_all),), width=2)
        out.append(done(im, ov))
    return out


def _earth(R):
    """a dark blue disc lit from the upper left, with a faint grid"""
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32) / R
    rr = np.sqrt(xx ** 2 + yy ** 2)
    inside = rr <= 1
    zz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
    lit = np.clip(0.35 + 0.65 * (-0.5 * xx - 0.5 * yy + 0.7 * zz), 0.12, 1)
    rgb = np.stack([18 + 30 * lit, 52 + 80 * lit, 100 + 120 * lit], -1)
    a = np.clip((1 - rr) * R, 0, 1) * inside
    im = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(im)
    for k in range(-2, 3):                                  # parallels and meridians
        yk = R + k * R * 0.38
        hw = R * math.sqrt(max(0.0, 1 - (k * 0.38) ** 2))
        d.line((R - hw, yk, R + hw, yk), fill=(150, 200, 255, 50), width=2)
        d.ellipse((R - R * abs(k) * 0.38, 0, R + R * abs(k) * 0.38, 2 * R), outline=(150, 200, 255, 40), width=2)
    return im


def earth_laps(T, dur):
    """round the Earth 7.5 times in a second, at real speed (twice), the laps and the second counted"""
    nf = int(round(dur * FPS))
    cx, cy, R = C.W / 2, 560, 200
    Ro = R + 40
    globe = _earth(R)
    whirls = [(T["e7"] + 0.4, T["e7"] + 1.4), (T["e8"] + 0.2, T["e8"] + 1.2)]
    out = []
    for i in range(nf):
        t = i / FPS
        a_all = min(1.0, (nf - 1 - i) / 10)
        u = 0.0
        for w0, w1 in whirls:
            if t >= w0: u = min(1.0, (t - w0) / (w1 - w0))
        moving = any(w0 <= t < w1 + 1 / FPS for w0, w1 in whirls)
        laps = 7.5 * u
        im, ov = canvas()
        im.alpha_composite(globe, (int(cx - R), int(cy - R)))
        d = ImageDraw.Draw(ov)
        d.ellipse((cx - Ro, cy - Ro, cx + Ro, cy + Ro), outline=CYAN + (int(50 * a_all),), width=2)
        ang = 2 * math.pi * laps - math.pi / 2
        if moving:                                      # the trail: as far as it went since the last frame, fading
            span = min(2 * math.pi, 7.5 * 2 * math.pi / FPS * 1.6)
            for s in range(60):
                f0 = s / 60
                a0 = ang - span * f0
                x, y = cx + Ro * math.cos(a0), cy + Ro * math.sin(a0)
                rr = 10 * (1 - f0) + 3
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(170, 225, 255, int(230 * (1 - f0) ** 1.2 * a_all)))
        x, y = cx + Ro * math.cos(ang), cy + Ro * math.sin(ang)
        for rr, al in ((34, 50), (20, 110), (11, 255)):
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(225, 245, 255, int(al * a_all)))
        text(d, (C.W / 2, 175), "In one second, [light could go round the Earth 7½ times].", 48, "Bold", WHITE,
             ease((t - 0.2) / 0.5) * a_all)
        text(d, (C.W / 2 + 470, cy - 40), f"{u:.2f} s", 64, "Bold", WHITE, a_all, "lm")
        text(d, (C.W / 2 + 470, cy + 20), "real time", 28, "Medium", DIM, a_all, "lm")
        lap_txt = "7½" if u >= 1 else f"{laps:.1f}"
        text(d, (C.W / 2 - 470, cy - 40), lap_txt, 64, "Bold", CYAN, a_all, "rm")
        text(d, (C.W / 2 - 470, cy + 20), "laps of the Earth", 28, "Medium", DIM, a_all, "rm")
        out.append(done(im, ov))
    return out


def plan(bg, T, dur):
    """over a frame of the race with light, the ladder of lines it leaves: what part 4 will show"""
    import make_race_light as L
    import camera as Cam
    nf = int(round(dur * FPS))
    rows = []
    for n in range(0, 17):
        (u, v) = L.uv(np.array([Cam.W * 0.5]), L.Z0 + L.DZ * n)[0]
        rows.append(C.TOP + v)
    out = []
    for i in range(nf):
        t = i / FPS
        a_all = min(1.0, ease(t / 0.4), (nf - 1 - i) / 10)
        im, ov = canvas((bg.astype(np.float32) * (1 - 0.25 * ease(t / 0.6))).astype(np.uint8))
        d = ImageDraw.Draw(ov)
        d.rectangle((0, 0, C.W, C.TOP), fill=(0, 0, 0, 255))
        d.rectangle((0, C.TOP + C.PH, C.W, C.H), fill=(0, 0, 0, 255))
        text(d, (C.W / 2, C.TOP / 2), "Next: the same race, with a pulse of [light] in it.", 48, "Bold", WHITE, a_all)
        a1 = ease((t - T["e12"]) / 0.5) * a_all        # on the left, in clear sky: what the lines are
        top, bot = rows[-1], rows[0]
        if a1 > 0:
            d.rounded_rectangle((28, (top + bot) / 2 - 52, 330, (top + bot) / 2 + 52), 10, fill=(4, 8, 16, int(150 * a1)))
            text(d, (48, (top + bot) / 2 - 20), "one line for each", 30, "SemiBold", WHITE, a1, "lm")
            text(d, (48, (top + bot) / 2 + 20), "lap of the Earth", 30, "SemiBold", CYAN, a1, "lm")
        a2 = ease((t - T["e12"] - 2.0) / 0.5) * a_all
        if a2 > 0:
            for r, lab in ((rows[0], "where it set off"), (rows[-1], "lap 16")):
                f = C.font(28, "SemiBold")
                w = d.textlength(lab, font=f)
                d.rounded_rectangle((28, r - 22, 60 + w, r + 22), 10, fill=(4, 8, 16, int(150 * a2)))
                text(d, (44, r), lab, 28, "SemiBold", WHITE, a2, "lm")
                d.line((68 + w, r, 380, r), fill=CYAN + (int(255 * a2),), width=3)
                d.polygon([(380, r - 9), (398, r), (380, r + 9)], fill=CYAN + (int(255 * a2),))
        a3 = ease((t - T["e12"] - 4.0) / 0.5) * a_all
        text(d, (C.W / 2, rows[-1] - 60), "250 m away, each lap 3 m above the last", 28, "Medium", WHITE, a3)
        out.append(done(im, ov))
    return out


def time_plan(T, dur):
    """how time runs in part 4, drawn as its own frames go: real speed, the first stop, the planes at 48x, 1,000x,
    the second stop, Parker and the finish"""
    import make_race_light as L
    hA, hB = L.HOLDS[0], L.HOLDS[L.N_PASS - 1]
    segs = [("real", 15), ("stop", hA[0] + 35 + hA[1]), ("real", 45), ("slow48", 353), ("slow1000", 27),
            ("stop", hB[0] + 35 + hB[1]), ("slow1000", 34), ("after", 96)]
    total = sum(n for _, n in segs)
    X0, X1, y, h = 150, 1770, 560, 54
    cols = {"real": (210, 214, 222), "slow48": ACCENT, "slow1000": (255, 140, 30), "after": (255, 140, 30), "stop": CYAN}
    xs, x = [], X0
    for kind, n in segs:
        w = (X1 - X0) * n / total
        xs.append((kind, x, x + w)); x += w
    nf = int(round(dur * FPS))
    out = []
    for i in range(nf):
        t = i / FPS
        a_all = min(1.0, (nf - 1 - i) / 10)
        head = X0 + (X1 - X0) * ease((t - T["e14"]) / 3.5)
        im, ov = canvas()
        d = ImageDraw.Draw(ov)
        text(d, (C.W / 2, 175), "How time runs in part 4", 56, "Bold", WHITE, ease(t / 0.4) * a_all)
        text(d, (C.W / 2, 245), "the sign at the top right always says which", 32, "Medium", DIM, ease((t - 0.3) / 0.4) * a_all)
        for kind, xa, xb in xs:
            if head <= xa: continue
            xe = min(xb, head)
            if xe - 2 > xa + 2:
                d.rectangle((xa + 2, y - h / 2, xe - 2, y + h / 2), fill=cols[kind] + (int(255 * a_all),))
            if kind == "stop" and xe - xa > 60:
                SM._pause(d, (xa + xb) / 2 - 16, y, 30, (6, 14, 28, int(255 * a_all)))
        if head < X1:
            d.ellipse((head - 9, y - 9, head + 9, y + 9), fill=WHITE + (int(255 * a_all),))
        lab = lambda xa, xb: (xa + xb) / 2
        k = {i_: s for i_, s in enumerate(xs)}
        notes = [(k[0][1], k[2][2], "REAL SPEED", "above", (210, 214, 222), "m"),
                 (k[3][1], k[3][2], "48× SLOWER", "above", ACCENT, "m"),
                 (k[4][1], k[7][2], "1,000× SLOWER", "above", (255, 140, 30), "m"),
                 (k[1][1], k[1][2], "TIME STOPPED: light sets off", "below", CYAN, "l"),
                 (k[5][1], k[5][2], "TIME STOPPED: light's 17th lap", "below", CYAN, "r")]
        for xa, xb, s, where, col, al in notes:
            a = ease((head - xa) / 80) * a_all
            if a <= 0: continue
            yy = y - h / 2 - 34 if where == "above" else y + h / 2 + 34
            x = lab(xa, xb) if al == "m" else xa if al == "l" else xb
            text(d, (x, yy), s, 28, "Bold", col, a, al + "m")
        xf = k[6][2]                                     # the finish
        af = ease((head - xf) / 40) * a_all
        if af > 0:
            d.line((xf, y - h / 2 - 8, xf, y + h / 2 + 70), fill=WHITE + (int(255 * af),), width=3)
            text(d, (xf, y + h / 2 + 96), "finish", 28, "Bold", WHITE, af)
        text(d, (C.W / 2, 790), "TIME STOPPED = slowed down a million times", 36, "Bold", CYAN,
             ease((t - T["e16"] - 2.5) / 0.5) * a_all)
        out.append(done(im, ov))
    return out


def fmt_trip(hours):
    if hours >= 48: return f"{hours / 24:,.0f} days" if hours / 24 >= 3 else f"{hours / 24:.1f} days"
    if hours >= 1: return f"{hours:.0f} hours"
    if hours * 60 >= 1: return f"{hours * 60:.0f} minutes"
    return f"{hours * 3600:.1f} seconds"


def moon(T, dur, d_m2):
    """how long each would take to reach the Moon; light going there at its real speed, Earth and Moon to scale"""
    nf = int(round(dur * FPS))
    ex, mx, yd = 200, 200 + 1520, 330
    re_, rm = 1520 * 3959 / MOON_MI, 1520 * 1080 / MOON_MI
    rows = MACHINES + [("Light", LIGHT_MPH)]
    trip = MOON_MI / (LIGHT_MPH / 3600)
    t1 = T["m4"] + 1.0                                  # "Watch." and it goes
    trips = [(t1, t1 + trip), (t1 + trip + 1.2, t1 + 2 * trip + 1.2)]
    out = []
    for i in range(nf):
        t = i / FPS
        a_all = min(1.0, (nf - 1 - i) / 12)
        im, ov = canvas()
        d = ImageDraw.Draw(ov)
        text(d, (C.W / 2, 150), "Getting to the Moon", 56, "Bold", WHITE, ease(t / 0.4) * a_all)
        text(d, (C.W / 2, 215), f"{MOON_MI:,} miles  ·  Earth and Moon to scale", 30, "Medium", DIM, ease(t / 0.4) * a_all)
        d.ellipse((ex - re_, yd - re_, ex + re_, yd + re_), fill=(40, 100, 190, int(255 * a_all)))
        d.ellipse((mx - rm, yd - rm, mx + rm, yd + rm), fill=(190, 190, 185, int(255 * a_all)))
        d.line((ex + re_ + 8, yd, mx - rm - 8, yd), fill=DIM + (int(60 * a_all),), width=1)
        for k, (name, v) in enumerate(rows):
            t_row = T["m3"] if name == "Light" else T["m2"] + k * d_m2 / 6.3
            a = ease((t - t_row) / 0.35) * a_all
            yy = 420 + k * 72
            is_light = name == "Light"
            text(d, (C.W / 2 - 30, yy), name, 38, "SemiBold", WHITE, a, "rm")
            text(d, (C.W / 2 + 30, yy), fmt_trip(MOON_MI / v), 38, "Bold", CYAN if is_light else INK, a, "lm")
        for t0, t1 in trips:                            # light, Earth to Moon, in real time
            if t0 <= t < t1 + 0.6:
                u = min(1.0, (t - t0) / (t1 - t0))
                x = ex + re_ + (mx - rm - ex - re_) * u
                d.line((max(ex + re_, x - 260), yd, x, yd), fill=(200, 235, 255, int(150 * a_all)), width=3)
                for rr, al in ((18, 60), (10, 140), (5, 255)):
                    d.ellipse((x - rr, yd - rr, x + rr, yd + rr), fill=(225, 245, 255, int(al * a_all)))
                text(d, (x, yd - 46), f"{min(t - t0, t1 - t0):.2f} s", 30, "Bold", CYAN, a_all)
        out.append(done(im, ov))
    return out


def last_card(T, dur):
    nf = int(round(dur * FPS))
    out = []
    for i in range(nf):
        t = i / FPS
        a = min(ease(t / 0.6), (nf - 1 - i) / 20)
        im, ov = canvas()
        glow_text(im, (C.W / 2, C.H / 2 - 50), "[Light]: 670,616,629 mph.", 84, "Black", WHITE, a, 14)
        text(ImageDraw.Draw(ov), (C.W / 2, C.H / 2 + 60), "Nothing with mass can ever catch it.", 44, "SemiBold",
             DIM, ease((t - 1.2) / 0.6) * a)
        out.append(done(im, ov))
    return out


# ---- Lily's voice

import re, wave
import vo_script as V

VO_DIR = os.environ.get("VO_DIR", "/home/user/lot/vo/lily")
NO_SUB = {"o1", "e1", "e2", "m5"}                       # their words are on the card already
GAP, LEAD, TAIL = 0.35, 0.5, 0.9                        # s between lines, before the first, after the last
CROSS = 35                                              # frames a frozen pass takes to cross (make_race_light)


def vo_len(name):
    with wave.open(f"{VO_DIR}/{name}.wav") as w:
        return w.getnframes() / w.getframerate()


def frames_at(seconds):
    return int(round(seconds * FPS))


_VO = {}


def vo_audio(name):
    """a line at the film's rate, at one loudness"""
    if name not in _VO:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{VO_DIR}/{name}.wav", "-f", "f32le", "-ac", "2", "-ar",
                              str(S.SR), "-"], capture_output=True, check=True).stdout
        a = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
        voiced = a[np.abs(a[:, 0]) > 0.02]
        rms = np.sqrt(np.mean(voiced ** 2)) if len(voiced) else 0.1
        _VO[name] = a * min(0.10 / rms, 0.8 / np.abs(a).max())
    return _VO[name]


def chain(names, t0=LEAD, at=None):
    """the lines one after another: {name: start (s)} and when the last one ends; at: {name: earliest start}"""
    T, t = {}, t0
    for n in names:
        t = max(t, (at or {}).get(n, 0.0))
        T[n] = t
        t += vo_len(n) + GAP
    return T, t - GAP


def light_holds():
    """frames time stands still before and after each frozen pass in part 4, as long as Lily needs: before the
    first, f1; across and after it, f2 then f3; the same for the last (f6; f7, f8)"""
    pre_a = max(65, frames_at(0.3 + vo_len("f1") + 0.5))
    post_a = max(40, frames_at(vo_len("f2") - CROSS / FPS + GAP + vo_len("f3") + 0.5))
    pre_b = max(65, frames_at(0.3 + vo_len("f6") + 0.5))
    post_b = max(40, frames_at(vo_len("f7") - CROSS / FPS + GAP + vo_len("f8") + 0.3))
    return pre_a, post_a, pre_b, post_b


os.environ.setdefault("LT_HOLDS", ",".join(str(h) for h in light_holds()))   # before make_race_light is imported


# ---- the parts, from their cuts

PART1_SILENT = [(100, 170, "Real speed: what a normal camera sees, 25 frames a second."),   # no voice over the flights
                (170, 261, "Wright Flyer, 1903. [Light is 22 million times faster.]"),
                (261, 362, "Spitfire, 1943. [Light is 1.8 million times faster.]"),
                (362, 475, "Bell X-1, 1947, the first faster than sound. [Light: 831,000 times faster.]"),
                (475, 588, "F-35, 2016. [Light is 546,000 times faster.]"),
                (588, 714, "X-15, 1967, the fastest crewed aircraft. [Light: 148,000 times faster.]"),
                (714, 764, "Parker Solar Probe, the fastest machine ever built. Watch closely...")]


def part4_anchors():
    """part 4's frames shown, and where its lines go in them"""
    import make_race_light as L
    js, seq = L.sequence()
    first = {}
    for o, (j, clock, phase) in enumerate(seq):
        k = js[j][0]
        if phase is None: first.setdefault(k, o)
        else: first.setdefault((phase[0], phase[1]), o)
    A, B = 0, L.N_PASS - 1
    cues = [(first[(A, "pre")] + 8, "f1"), (first[(A, "cross")], "f2"), (first[(A, "cross")], "f3"),
            (first[80], "f4"), (first[330], "f5"),
            (first[(B, "pre")] + 8, "f6"), (first[(B, "cross")], "f7"), (first[(B, "cross")], "f8"),
            (first[474] + 3, "f9"), (first[474] + 3, "f10")]
    return len(seq), cues, first[(B, "pre")]


def part_plan(num):
    """(cut, first frame, end frame, [(frame, line)] wanted, the frame held if the lines need longer)"""
    if num == 1: return "history_realtime", 100, 922, [(775, "p1e")], 880
    if num == 2:
        return ("history_slow48", 112, 1544, [(300, "p2a"), (800, "p2b"), (1075, "p2c"), (1300, "p2d"), (1440, "p2e")],
                1490)
    if num == 3: return "history_race_parker", 0, 600, [(78, "p3a"), (252, "p3b"), (392, "p3c"), (506, "p3d")], 599
    n_seq, cues, _ = part4_anchors()
    return "history_race_light_vo", 0, n_seq + 30 + 112, cues, n_seq + 30 + 60


def place(k0, k1, wanted, hold_at):
    """each line at its frame, or after the one before it ends; and the frames to hold at hold_at so the last one
    is done before the part is (offsets from k0)"""
    out, end = [], -10 ** 9
    for k, name in wanted:
        f = max(k - k0, end + frames_at(GAP))
        out.append((f, name))
        end = f + frames_at(vo_len(name))
    extra = max(0, end + frames_at(TAIL) - (k1 - k0))
    return out, {hold_at: extra}


def frames_of(name, k0, k1):
    """frames k0..k1 of a cut, one at a time (a whole cut in memory would not fit)"""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{OUT}/{name}.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, k = C.W * C.H * 3, 0
    while k1 is None or k < k1:
        b = p.stdout.read(n)
        if len(b) < n: break
        if k >= k0: yield np.frombuffer(b, np.uint8).reshape(C.H, C.W, 3)
        k += 1
    p.stdout.close(); p.wait()


def sound_of(name, k0, k1, holds):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{OUT}/{name}.mp4", "-f", "f32le", "-ac", "2", "-ar",
                          str(S.SR), "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    per = S.SR / FPS
    a = a[int(k0 * per):int(k1 * per)].copy()
    a = np.pad(a, ((0, max(0, int((k1 - k0) * per) - len(a))), (0, 0)))
    for k, extra in sorted(holds.items(), reverse=True):        # held frames: silence
        i = int((k + 1 - k0) * per)
        a = np.concatenate([a[:i], np.zeros((int(extra * per), 2)), a[i:]])
    edge = int(0.01 * S.SR)                             # no clicks at the joins
    a[:edge] *= np.linspace(0, 1, edge)[:, None]; a[-edge:] *= np.linspace(1, 0, edge)[:, None]
    return a


def part_frames(name, k0, k1, holds, silent):
    for i, f in enumerate(frames_of(name, k0, k1)):
        k = k0 + i
        f = CAP.draw(f, *CAP.at(silent, k)) if silent else f
        for _ in range(1 + holds.get(k, 0)): yield f


def pad(n_frames, seed=3):
    """under the black explainer cards: a low, quiet chord that breathes"""
    n = int(n_frames / FPS * S.SR)
    t = np.arange(n) / S.SR
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2))
    for f, g in ((55.0, 1.0), (82.41, 0.6), (110.0, 0.5), (164.81, 0.3), (220.0, 0.15)):
        for ch, det in ((0, 0.997), (1, 1.003)):
            ph = rng.uniform(0, 2 * np.pi)
            x[:, ch] += g * np.sin(2 * np.pi * f * det * t + ph) * (0.75 + 0.25 * np.sin(2 * np.pi * 0.09 * t + ph))
    x *= 0.02
    e = min(n // 2, int(0.6 * S.SR))
    x[:e] *= np.linspace(0, 1, e)[:, None]; x[-e:] *= np.linspace(1, 0, e)[:, None]
    return x


def wind(n_frames, seed=1):
    a = S.wind(int(n_frames / FPS * S.SR), np.random.default_rng(seed), 0.035)
    e = int(0.3 * S.SR)
    a[:e] *= np.linspace(0, 1, e)[:, None]; a[-e:] *= np.linspace(1, 0, e)[:, None]
    return a


def plan_bg():
    """a frame of the race with light before any sign, the ladder up to lap 16 (from the clean file)"""
    import make_race_light as L
    js, seq = L.sequence()
    j = next(i for i, (k, extra, s, c) in enumerate(js) if extra is None and k == 360)
    for i, fr in enumerate(L.clean_frames()):
        if i == j: return fr.copy()


# ---- the film

def card(name, make, lines, dur_min=0.0, at=None, t0=LEAD, tail=TAIL, bed="pad"):
    T, end = chain(lines, t0, at)
    dur = max(dur_min, end + tail)
    return dict(name=name, n=frames_at(dur), cues=[(frames_at(T[l]), l) for l in lines],
                frames=lambda: iter(make(T, dur)), bed=bed, duck=4.0)


def part_piece(num):
    name, k0, k1, wanted, hold_at = part_plan(num)
    cues, holds = place(k0, k1, wanted, hold_at)
    silent = PART1_SILENT if num == 1 else []
    return dict(name=f"part{num}", n=(k1 - k0) + sum(holds.values()), cues=cues,
                frames=lambda: part_frames(name, k0, k1, holds, silent),
                bed=lambda: sound_of(name, k0, k1, holds), duck=10.0)


def chart_card():
    T, _ = chain(["e3", "e4", "e5"], 0.4)
    T6 = max(T["e5"] + vo_len("e5") + GAP, T["e5"] + 2.3)            # as Parker shrinks
    return card("chart", speed_chart, ["e3", "e4", "e5", "e6"], at={"e6": T6}, t0=0.4, tail=1.0)


def moon_card():
    T, end = chain(["m1", "m2", "m3", "m4"], 0.4)
    trips_end = T["m4"] + 1.0 + 2 * MOON_MI / (LIGHT_MPH / 3600) + 1.2
    d2 = vo_len("m2")
    return card("moon", lambda T_, dur: moon(T_, dur, d2), ["m1", "m2", "m3", "m4"], dur_min=trips_end + 0.8, t0=0.4)


def film():
    return [card("opening", lambda T, d: opening(T, d, 0), ["o1", "o2"], dur_min=6.5, t0=0.6, bed="wind"),
            card("card1", lambda T, d: part_card(1, "REAL SPEED", "", 0, d), ["c1"], 3.8, bed="wind"),
            part_piece(1),
            card("card2", lambda T, d: part_card(2, "48× SLOWER", "", 0, d), ["c2"], 3.8, bed="wind"),
            part_piece(2),
            card("card3", lambda T, d: part_card(3, "THE RACE", "", 0, d), ["c3"], 3.8, bed="wind"),
            part_piece(3),
            card("light", light_intro, ["e1", "e2"], at={"e2": 0.4 + vo_len("e1") + 0.7}, t0=0.4),
            chart_card(),
            card("earth", earth_laps, ["e7", "e8", "e9"], t0=0.4),
            card("plan", lambda T, d: plan(plan_bg(), T, d), ["e10", "e11", "e12", "e13"], t0=0.4),
            card("time", time_plan, ["e14", "e15", "e16", "e17"], t0=0.4),
            card("card4", lambda T, d: part_card(4, "THE RACE, WITH LIGHT", "", 0, d), ["c4"], 3.4, bed="wind"),
            part_piece(4),
            moon_card(),
            card("end", last_card, ["m5"], t0=0.6, tail=1.8)]


def subtitles(placed):
    """what she says, shown in the band below the picture a sentence at a time, each for its share of the line"""
    out = []
    for f0, name in placed:
        if name in NO_SUB: continue
        sents = [x for x in re.split(r"(?<=[.?!])\s+", V.LINES[name][1]) if x]
        merged = []
        for x in sents:                                 # a short sentence rides with the next
            if merged and len(merged[-1]) < 28: merged[-1] += " " + x
            else: merged.append(x)
        n = frames_at(vo_len(name))
        tot = sum(len(x) for x in merged)
        f = f0
        for x in merged:
            g = f + n * len(x) / tot
            out.append((int(f), int(g) + 6, x))
            f = g
    return out


def mix(pieces, placed, total):
    """the pieces' own sound, lowered under her voice (10 dB in the parts, 4 dB under the cards), and her voice"""
    per = S.SR / FPS
    bed, depth = [], []
    for p in pieces:
        m = int(p["n"] * per)
        a = p["bed"]() if callable(p["bed"]) else wind(p["n"]) if p["bed"] == "wind" else pad(p["n"])
        bed.append(np.pad(a, ((0, max(0, m - len(a))), (0, 0)))[:m])
        depth.append(np.full(m, p["duck"]))
    bed, depth = np.concatenate(bed), np.concatenate(depth)
    n = len(bed)
    vo = np.zeros((n, 2))
    on = np.zeros(n)
    for f0, name in placed:
        a = vo_audio(name)
        i = int(f0 * per)
        j = min(n, i + len(a))
        vo[i:j] += a[:j - i]
        on[max(0, i - int(0.15 * S.SR)):min(n, j + int(0.35 * S.SR))] = 1
    k = int(0.15 * S.SR)                                # the ducking eases in and out
    on = np.convolve(on, np.ones(k) / k, mode="same")
    g = 1 - on * (1 - 10 ** (-depth / 20))
    x = bed * g[:, None] + vo
    ax = np.abs(x)                                      # a soft ceiling
    return np.where(ax < 0.9, x, np.sign(x) * (0.9 + 0.1 * np.tanh((ax - 0.9) / 0.1)))


if __name__ == "__main__" and sys.argv[1:2] == ["holds"]:            # for make_race_light.py's film version
    print(os.environ["LT_HOLDS"])
elif __name__ == "__main__" and sys.argv[1:2] == ["plan"]:           # where every line falls
    t = 0
    for p in film():
        for f, name in p["cues"]:
            print(f"{(t + f) / FPS:7.2f} s  {p['name']:8s} {name:4s} {vo_len(name):5.2f} s  {V.LINES[name][1][:70]}")
        t += p["n"]
    print(f"{t / FPS:.1f} s in all")
elif __name__ == "__main__" and sys.argv[1:2] == ["test"]:
    os.makedirs(f"{OUT}/test_film", exist_ok=True)
    import cv2
    for p in film():
        if p["name"] not in sys.argv[2:]: continue
        fr = list(p["frames"]())
        for i in range(0, len(fr), max(1, len(fr) // 8)):
            cv2.imwrite(f"{OUT}/test_film/{p['name']}_{i:04d}.png", fr[i][..., ::-1])
        print(p["name"], len(fr), "frames, planned", p["n"])
elif __name__ == "__main__":
    pieces = film()
    placed, t = [], 0
    for p in pieces:
        placed += [(t + f, name) for f, name in p["cues"]]
        t += p["n"]
    subs = subtitles(placed)
    sink = MC.Sink(os.environ.get("FILM_NAME", "history_light_film"))
    for p in pieces:
        n0 = sink.n
        for f in p["frames"]():
            if sink.n - n0 >= p["n"]: break
            sink.add([CAP.draw(f, *CAP.at(subs, sink.n))] if CAP.at(subs, sink.n)[0] else [f])
        while sink.n - n0 < p["n"]:                    # a maker one frame short: hold its last
            sink.add([f])
        print(p["name"], p["n"] / FPS, "s", flush=True)
    sink.close(mix(pieces, placed, t))
