"""Three routes, one ETA: the video (1920x1080, 30 fps, caption-only). Every number comes from scenario.py.

    python video.py NAME [still times...]
    NAME: hook, light, empty, curve, fill, wardrop, learn, app, optimum, braess, outro
"""
import os, sys, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

import look as L
from look import W, H, FPS, INK, INK2, MUTED, GOLD, ROUTE, smooth, lerp, text
from cities import C as CITY_C
import model as M
import style as ST

N = L.numbers()
R = L.routes_json()
MAIN = [R["routes"][i] for i in R["main"]]
T0, CAP = N["t0"], N["cap"]
NAMES = N["names"]
XF = 0.5
BAR = 2.5                                   # 96 BPM: sections are whole bars long (+ the cross-fade)
GOLDISH = (255, 214, 140)
FILM = os.path.join(L.ROOT, "media", "world_roads_light_speed.mp4")


def bars(n):
    return n * BAR + XF


IN = CITY_C["names_in"]                                    # route names as they read mid-sentence
SIDES = N["sides"]
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()
Y0 = int(5 * np.floor((min(T0) - 3) / 5))                   # chart floors and ceilings that fit this city's times
Y1 = int(5 * np.ceil((N["ue"]["level"] + 3) / 5))
DMAX = int(1000 * np.ceil((N["d_rush"] + 400) / 1000))


ONE_IN = round(1 / N["learn"]["calm"]["params"]["replan"])


def final_cap():
    """the learning run's last word: close but not identical, or all the same once rounded to the minute"""
    ex, r = N["eta_final_exact"], N["eta_final"]
    if len(set(r)) == 1:
        return [cap(f"Not exactly equal ({ex[0]:.1f}, {ex[1]:.1f} and {ex[2]:.1f} min),", 36),
                cap(f"but rounded to the minute, all three say {r[0]} min.", 40, GOLD)]
    return [cap(f"Close, not identical: {r[0]}, {r[1]} and {r[2]} minutes.", 40, GOLD)]


def cap1(s):
    return s[:1].upper() + s[1:]


def wrap(s, size, width, weight="Medium"):
    """split text into lines no wider than `width` pixels"""
    f = ST.font(size, weight)
    out, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = t
    return out + ([cur] if cur else [])


def fmt_int(v):
    return f"{int(round(v)):,}"


def cars_h(v):
    return "1 car/h" if int(round(v)) == 1 else f"{fmt_int(v)} cars/h"


def caps(d, t, items):
    """items: (t0, t1, lines, kwargs) with lines [(text, size, colour, weight)]"""
    for it in items:
        a = ST.fade(t, it[0], it[1])
        if a > 0:
            ST.caption(d, it[2], a, **(it[3] if len(it) > 3 else {}))


def cap(s, size=40, col=INK, w="SemiBold"):
    return (s, size, col, w)


def header(d, s, sub=None, a=1.0, x=70, y=56):
    text(d, (x, y), s, 34, INK, a, "SemiBold")
    if sub:
        text(d, (x, y + 46), sub, 24, MUTED, a, "Medium")


def note(d, s, a=1.0, x=70, y=H - 40, anchor="ls", size=19):
    text(d, (x, y), s, size, MUTED, a, "Regular", anchor)


# ------------------------------------------------------------------ the map scene
RIGHT = L.MapView.fit((950, 60, 1860, 900), "right")      # beside a text column, clear of the bottom captions
LEFT = L.MapView.fit((150, 120, 1000, 880), "left")        # beside a chart
CENTER = L.MapView.fit((300, 80, 1620, 1000), "center")
BADGE_AT = [0.42, 0.47, 0.40]                     # where each route's letter sits along it


def shade(view):
    """dim the street network under the text column (right view) or the chart column (left view)"""
    x = np.arange(W, dtype=np.float32)
    if view.name == "right":
        m = 0.22 + 0.78 * np.clip((x - 640) / 300, 0, 1)
    elif view.name == "left":
        m = 0.22 + 0.78 * np.clip((1060 - x) / 160, 0, 1)
    else:
        m = np.ones(W, np.float32)
    return m[None, :, None]


def dash_polys(P, cum, on, off):
    out, s = [], 0.0
    while s < cum[-1]:
        a, b = s, min(s + on, cum[-1])
        i0, i1 = np.searchsorted(cum, a), np.searchsorted(cum, b)
        seg = [L.at(P, cum, a)] + [P[k] for k in range(i0, i1)] + [L.at(P, cum, b)]
        out.append(np.array(seg))
        s += on + off
    return out


class Scene:
    def __init__(self, view):
        self.v = view
        self.base = (view.base() * shade(view)).astype(np.float32)
        self.P = [view.pts(r["detail"]["xy"]) for r in MAIN]
        self.cum = [L.polyline_cum(p) for p in self.P]
        self.tmin = [np.array(r["detail"]["tmin"]) for r in MAIN]
        self.O = view.px(R["origin"]["x"], R["origin"]["y"])
        self.D = view.px(R["dest"]["x"], R["dest"]["y"])
        self.dash = [dash_polys(P, c, 9, 9) for P, c in zip(self.P, self.cum)]
        self.glow = L.Glow()
        self.badge_at = [L.at(self.P[i], self.cum[i], BADGE_AT[i] * self.cum[i][-1]) for i in range(3)]

    def px_at_minute(self, i, m):
        Ls = np.interp(m, self.tmin[i], self.cum[i])
        return L.at(self.P[i], self.cum[i], Ls), Ls

    def route(self, i, width=4, a=1.0, upto=None, dashed=False):
        if dashed:
            for seg in self.dash[i]:
                self.glow.line(seg, ROUTE[i], width, a)
            return
        P = self.P[i] if upto is None else L.partial(self.P[i], self.cum[i], upto)
        self.glow.line(P, ROUTE[i], width, a)

    def pins(self, d, a=1.0, labels=True):
        if a <= 0:
            return
        for (x, y), nm, k in ((self.O, N["origin"], 0), (self.D, N["dest"], 1)):
            d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=(8, 12, 20, int(235 * a)), outline=(240, 244, 250, int(255 * a)), width=4)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(240, 244, 250, int(255 * a)))
            if labels:
                lab = f"{nm} (start)" if k == 0 else f"{nm} (destination)"
                tw = d.textlength(lab, font=ST.font(24, "SemiBold"))
                side = CITY_C["pin_labels"][k]                # l / r / t / b of the pin
                cx = {"l": x - 36 - tw / 2, "r": x + 36 + tw / 2}.get(side, x)
                cy = {"t": y - 44, "b": y + 44}.get(side, y)
                cx = min(max(cx, tw / 2 + 24), W - tw / 2 - 24)           # keep the label on screen
                d.rounded_rectangle([cx - tw / 2 - 12, cy - 20, cx + tw / 2 + 12, cy + 20], 12, fill=ST.PANEL + (int(190 * a),))
                text(d, (cx, cy), lab, 24, INK, a, "SemiBold", "mm")

    def badges(self, d, a=1.0, which=(0, 1, 2)):
        for i in which:
            x, y = self.badge_at[i]
            d.ellipse([x - 21, y - 21, x + 21, y + 21], fill=(8, 12, 20, int(230 * a)))
            L.badge(d, x, y, i, r=17, a=a)

    def compose(self, dim=1.0, extra=None):
        img = self.glow.compose(self.base, dim=dim)
        if extra is not None:
            img = np.clip(img.astype(np.float32) + extra, 0, 255).astype(np.uint8)
        return img


def eta_rows(d, x, y, rows, a=1.0, size=30, gap=None):
    """stacked ETA pills: rows of (route, main text, small text)"""
    gap = gap or int(size * 2.15)
    for k, (i, main, sub) in enumerate(rows):
        L.pill(d, x, y + k * gap, i, main, sub, a=a, size=size, anchor="lm")


class Cars:
    """dots for the vehicles on each route: count = flow x trip time (vehicles on the road at once), speed =
    route length / trip time (a time-lapse); new dots slot in between old ones, so counts change smoothly"""
    PHI = (np.arange(600) * 0.6180339887) % 1.0

    def __init__(self, scene, state, t_end, per=25.0, lap=0.25):
        self.s, self.state, self.per = scene, state, per
        ts = np.arange(0, t_end + 2 / FPS, 1 / FPS)
        speed = np.array([[1.0 / max(1e-6, T * lap) for T in state(t)[1]] for t in ts])
        self.ts, self.u = ts, np.cumsum(speed, axis=0) / FPS

    def draw(self, t, a=1.0, r=2.8):
        f, T = self.state(t)
        k = min(len(self.ts) - 1, int(round(t * FPS)))
        for i in range(3):
            if f[i] <= 0.5 or not np.isfinite(T[i]):
                continue
            n = int(min(len(self.PHI), round(f[i] * T[i] / 60.0 / self.per)))
            if n <= 0:
                continue
            s = (self.PHI[:n] + self.u[k, i]) % 1.0
            Ls = s * self.s.cum[i][-1]
            xs = np.interp(Ls, self.s.cum[i], self.s.P[i][:, 0])
            ys = np.interp(Ls, self.s.cum[i], self.s.P[i][:, 1])
            col = tuple(min(255, c * 0.6 + 110) for c in ROUTE[i])
            self.s.glow.dots(zip(xs, ys), col, r, a)


def flow_width(f):
    return int(round(2 + 7 * min(1.0, f / 2600.0)))


# ================================================================== 1. the hook
class Hook(L.Section):
    name, dur = "hook", bars(10)

    def __init__(self):
        self.sc = Scene(RIGHT)
        ue = N["ue"]
        self.cars = Cars(self.sc, lambda t: (ue["flows"], N["learn"]["calm"]["times"][-1]), self.dur)
        self.spr = [L.sprite(c) for c in ROUTE]

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        grow = smooth((t - 1.4) / 3.6)
        for i in range(3):
            sc.route(i, 4, 0.95, upto=grow * sc.cum[i][-1])
        if t > 5.0:
            self.cars.draw(t, a=smooth((t - 5.0) / 1.5))
        fade = smooth(t / 1.4)
        img = sc.compose(dim=fade).astype(np.float32)
        if 1.4 < t < 5.4:
            for i in range(3):
                x, y = L.at(sc.P[i], sc.cum[i], grow * sc.cum[i][-1])
                L.blit_add(img, self.spr[i], x, y, 1.0 - smooth((t - 4.6) / 0.8))
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, smooth((t - 0.6) / 0.6))
        ab = smooth((t - 5.0) / 0.6)
        sc.badges(d, ab)
        x = 70
        lines = [("You're not the only one", INK), ("who constantly gets 3 or more", INK), ("completely different routes", INK),
                 ("to your destination", INK), ("with the same ETA.", GOLD)]
        for k, (s, col) in enumerate(lines):
            a = smooth((t - 6.6 - 0.9 * k) / 0.5) * (1 - smooth((t - 18.6) / 0.6))
            text(d, (x, 120 + k * 76), s, 58, col, a, "Bold")
        text(d, (x, 560), "RUSH HOUR", 22, MUTED, ab, "SemiBold")
        eta_rows(d, x, 628, [(i, f"{N['eta_final'][i]} min", NAMES[i]) for i in range(3)], a=ab, size=32)
        a2 = ST.fade(t, 12.8, 18.6, 0.5, 0.6)
        for k, s_ in enumerate(wrap(f"{N['trip']}: {SIDES[0]}, {SIDES[1]}, or {SIDES[2]}. All within a minute.", 28, 780)):
            text(d, (x, 860 + 38 * k), s_, 28, INK2, a2, "Medium")
        a3 = ST.fade(t, 19.2, 1e9, 0.5, 0.5)
        text(d, (x, 150), "Coincidence?", 76, INK, a3, "Bold")
        a4 = ST.fade(t, 20.6, 1e9, 0.5, 0.5)
        text(d, (x, 270), "Not really. Thousands of drivers,", 38, INK2, a4, "SemiBold")
        text(d, (x, 322), "each taking the fastest route they know,", 38, INK2, a4, "SemiBold")
        text(d, (x, 374), "push those times together.", 38, GOLD, a4, "SemiBold")
        note(d, f"Rush-hour times: a toy traffic model on real {N['city_name']} roads (details in the description)", smooth((t - 5.5) / 0.8))
        return np.asarray(im)


# ================================================================== 2. callback to the light-speed film
class Clip:
    """frames of an existing video, read in order (seeks when asked out of order)"""

    def __init__(self, path, t0):
        self.path, self.t0, self.p, self.nxt = path, t0, None, -1

    def get(self, i):
        if self.p is None or i != self.nxt:
            if self.p is not None:
                self.p.kill()
            self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.t0 + i / FPS:.4f}", "-i", self.path, "-f", "rawvideo",
                                       "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = self.p.stdout.read(W * H * 3)
        self.nxt = i + 1
        if len(buf) < W * H * 3:
            return np.zeros((H, W, 3), np.uint8)
        return np.frombuffer(buf, np.uint8).reshape(H, W, 3)


class Light(L.Section):
    name, dur = "light", bars(6)
    START = 26.0                                      # North America turning gold, the dot crossing the east

    def __init__(self):
        self.clip = Clip(FILM, self.START)

    def frame(self, t):
        fr = self.clip.get(int(round(t * FPS))).astype(np.float32) * 0.88
        im = Image.fromarray(fr.astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        up = dict(y=790)                                   # above the film's own legend and credits
        caps(d, t, [(0.4, 6.2, [cap("In an earlier video, a point of light drove the world's roads.")], up),
                    (6.6, 10.6, [cap("No traffic, no red lights: about 4 minutes 40 seconds, by our estimate.")], up),
                    (11.0, 15.6, [cap("Down here, we share the road. And that changes everything.")], up)])
        return np.asarray(im)


# ================================================================== 3. 3 a.m.: the race on empty roads
class Empty(L.Section):
    name, dur = "empty", bars(10)
    GO, RACE = 2.0, 13.5                              # race starts; seconds of video for 40 minutes of driving

    def __init__(self):
        self.sc = Scene(RIGHT)
        self.spr = [L.sprite(c) for c in ROUTE]

    def minute(self, t):
        return max(0.0, (t - self.GO) / self.RACE * 40.0)

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        m = self.minute(t)
        heads = []
        for i in range(3):
            sc.route(i, 2, 0.32)
            p, Ls = sc.px_at_minute(i, min(m, T0[i]))
            sc.route(i, 5, 1.0, upto=Ls)
            heads.append(p)
        img = sc.compose().astype(np.float32)
        for i in range(3):
            if m < T0[i] + 0.3:
                L.blit_add(img, self.spr[i], heads[i][0], heads[i][1], 1.0)
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0)
        header(d, "3:00 a.m.", "Empty roads, the speed limit all the way")
        mm = min(m, max(T0))
        text(d, (80, 260), "DRIVING TIME", 24, MUTED, 1.0, "SemiBold")
        text(d, (80, 290), f"{int(mm):d}:{int((mm % 1) * 60):02d}", 96, GOLD, 1.0, "Bold")
        for i in range(3):
            y = 470 + i * 92
            done = m >= T0[i]
            L.badge(d, 104, y, i, r=22)
            text(d, (144, y), NAMES[i], 32, INK if done else INK2, 1.0, "SemiBold", "lm")
            text(d, (144, y + 36), N["sides"][i], 22, MUTED, 1.0, "Medium", "lm")
            if done:
                a = smooth((m - T0[i]) / 0.8)
                text(d, (800, y), f"{T0[i]:.0f} min", 40, INK, a, "Bold", "rm")
        caps(d, t, [(0.3, 5.6, [cap("At 3 a.m., the same trip on the same three routes.")]),
                    (6.0, 12.6, [cap("No traffic, no red lights: real roads at the posted speed limits.")]),
                    (16.4, 20.8, [cap(f"{cap1(IN[0])} wins by {WORDS[round(T0[1] - T0[0])]} minutes.")]),
                    (21.2, 25.6, [cap("So why do the three tie at rush hour?", 40, GOLD)])])
        note(d, "Road data © OpenStreetMap contributors, via Overture Maps · speed limits from OpenStreetMap, typical limits where unmapped")
        return np.asarray(im)


# ================================================================== 4. the congestion curve
class Curve(L.Section):
    name, dur = "curve", bars(8)

    def __init__(self):
        self.ax = L.Axes((190, 190, 1080, 840), (0, 3000), (Y0, Y0 + 50))
        self.f = np.linspace(0, 3000, 301)
        self.T = [np.array([M.bpr(T0[i], CAP[i], x) for x in self.f]) for i in range(3)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = np.clip(1 - 0.38 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.75)) ** 2), 0.4, 1)
        self.bg = (L.BG[None, None] * v[..., None]).astype(np.uint8)

    def frame(self, t):
        im = Image.fromarray(self.bg.copy())
        d = ImageDraw.Draw(im, "RGBA")
        ax = self.ax
        a0 = smooth(t / 0.8)
        ax.draw(d, range(0, 3001, 500), range(Y0, Y0 + 51, 10), lambda v: f"{v:,}", lambda v: f"{v}", "Cars per hour on the route", "Trip time (minutes)", a0)
        u = smooth((t - 1.2) / 7.0)
        k = int(u * 300)
        ax.series(d, self.f[:k + 1], self.T[0][:k + 1], i=0, width=5, a=a0)
        if 0 < k:
            x, y = float(ax.X(self.f[k])), float(ax.Y(self.T[0][k]))
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=ROUTE[0] + (255,), outline=(8, 12, 20, 255), width=3)
            if t < 12.5:
                text(d, (x - 16, y - 18), f"{fmt_int(self.f[k])} cars/h: {self.T[0][k]:.0f} min", 26, INK, smooth((t - 1.2) / 0.5), "SemiBold", "rs")
        ab = smooth((t - 12.6) / 1.0)
        for i in (1, 2):
            ax.series(d, self.f, self.T[i], i=i, width=4, a=ab, clip=True)
        for i in range(3):
            a = 1.0 if i == 0 else ab
            j = int(np.nonzero(self.T[i] <= Y0 + 50)[0][-1])
            bx, by = float(ax.X(self.f[j])), float(ax.Y(self.T[i][j]))
            if i == 0:
                bx, by = ax.x1 + 30, by
            else:
                bx, by = bx + 26, by + 6
            L.badge(d, bx, by, i, r=16, a=a * a0 if i else a0 * smooth((t - 8.0) / 0.5))
        x0 = 1190
        text(d, (x0, 230), "Every car slows", 54, INK, ST.fade(t, 1.0, 1e9), "Bold")
        text(d, (x0, 296), "every other car.", 54, INK, ST.fade(t, 1.0, 1e9), "Bold")
        text(d, (x0, 392), "A little at first.", 42, INK2, ST.fade(t, 5.0, 1e9), "SemiBold")
        text(d, (x0, 446), "Then a lot.", 42, GOLD, ST.fade(t, 7.0, 1e9), "SemiBold")
        af = ST.fade(t, 9.0, 1e9)
        text(d, (x0, 560), f"trip time = {T0[0]:.0f} min × (1 + 0.15 × (cars ÷ {CAP[0]:,.0f})⁴)", 27, INK, af, "Medium")
        text(d, (x0, 606), "The BPR curve (US Bureau of Public Roads, 1964):", 22, MUTED, af, "Regular")
        text(d, (x0, 636), "a standard simplification in traffic planning.", 22, MUTED, af, "Regular")
        text(d, (x0, 666), "Our capacities (cars per hour) are made up.", 22, MUTED, af, "Regular")
        text(d, (x0, 760), "Each route has its own curve.", 36, INK, ab, "SemiBold")
        return np.asarray(im)


# ================================================================== 5. rush hour builds
# demand keeps creeping up while each "joins" caption shows, so the new route visibly starts to fill
FILL_SCHED = [(0.0, 0.0), (3.0, 0.0), (13.0, N["joins"][1]), (18.0, N["joins"][1] + 220), (25.0, N["joins"][2]),
              (30.0, N["joins"][2] + 220), (41.0, N["d_rush"]), (60.0, N["d_rush"])]


def demand_at(t, sched=FILL_SCHED):
    for (ta, da), (tb, db) in zip(sched[:-1], sched[1:]):
        if ta <= t <= tb:
            return lerp(da, db, smooth((t - ta) / (tb - ta)) if tb > ta else 1.0)
    return sched[-1][1]


class Fill(L.Section):
    name, dur = "fill", bars(20)

    def __init__(self):
        self.sc = Scene(LEFT)
        self.dg = np.linspace(0, DMAX, 601)
        self.ue = [M.parallel_ue(T0, CAP, x) for x in self.dg]
        self.Tg = np.array([u["times"] for u in self.ue])
        self.cars = Cars(self.sc, self.state, self.dur)
        self.ax = L.Axes((1170, 440, 1830, 770), (0, DMAX), (Y0, Y1))

    def state(self, t):
        D = demand_at(t)
        u = M.parallel_ue(T0, CAP, D)
        return u["flows"], u["times"]

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        D = demand_at(t)
        u = M.parallel_ue(T0, CAP, D)
        for i in range(3):
            if u["used"][i]:
                sc.route(i, flow_width(u["flows"][i]), 0.95)
            else:
                sc.route(i, 2, 0.4, dashed=True)
        self.cars.draw(t)
        im = Image.fromarray(sc.compose())
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0, labels=False)
        sc.badges(d)
        eta_rows(d, 1110, 150, [(i, f"{u['times'][i]:.1f} min", NAMES[i] + (f" · {cars_h(u['flows'][i])}" if u["used"][i] else " · nobody yet"))
                               for i in range(3)], size=30, gap=64)
        header(d, "Rush hour builds", "toy traffic model · each dot ≈ 25 cars on the road")
        ax = self.ax
        ax.draw(d, range(0, DMAX + 1, 1000), range(Y0, Y1 + 1, 5), lambda v: f"{v:,}", str, "Drivers making this trip (per hour)", "Trip time on each route (min)")
        k = int(np.searchsorted(self.dg, D))
        for i in range(3):
            j = N["joins"][i]
            idle = self.dg <= j + 1e-9
            used = self.dg >= j - 1e-9
            m = np.arange(len(self.dg)) <= k
            ax.series(d, self.dg[idle & m], self.Tg[idle & m, i], i=i, width=2, a=0.45, dash=(3, 7))
            ax.series(d, self.dg[used & m], self.Tg[used & m, i] + (i - 1) * 0.12, i=i, width=4)   # braided, so all three show
        for i in range(3):
            x, y = float(ax.X(D)), float(ax.Y(u["times"][i]))
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=ROUTE[i] + (255,), outline=(8, 12, 20, 255), width=3)
        for i, tj in ((1, 13.0), (2, 25.0)):
            a = smooth((t - tj) / 0.6)
            x, y = float(ax.X(N["joins"][i])), float(ax.Y(T0[i]))
            text(d, (x + 14, y + 20) if i == 1 else (x - 14, y - 26), f"{LET(i)} joins at {fmt_int(N['joins'][i])}", 22, INK2, a, "Medium", "la" if i == 1 else "rs")
        text(d, (1830, 372), f"{fmt_int(D)} drivers per hour", 32, GOLD, 1.0, "Bold", "rs")
        caps(d, t, [(3.0, 12.4, [cap(f"As more people make the trip, they all take {IN[0]} at first. It slows down.", 34)]),
                    (13.0, 17.8, [cap(f"At {fmt_int(N['joins'][1])} drivers an hour, {IN[0]} is as slow as {IN[1]} is when empty.", 34),
                                  cap("Now drivers start taking that route too.", 34, INK2)]),
                    (25.0, 29.8, [cap(f"At {fmt_int(N['joins'][2])}, {IN[2]} is just as quick. It fills up too.", 34)]),
                    (30.4, 40.4, [cap("From here on, all three slow down together.", 34)]),
                    (41.2, 50.6, [cap("Three completely different routes. One ETA.", 44, GOLD, "Bold")])], )
        return np.asarray(im)


def LET(i):
    return L.LET[i]


# ================================================================== 6. Wardrop's principle
class Wardrop(L.Section):
    name, dur = "wardrop", bars(8)

    def __init__(self):
        sc = Scene(CENTER)
        b = sc.base * 0.35
        self.bg = cv2.GaussianBlur(b, (0, 0), 4).astype(np.uint8)

    def frame(self, t):
        im = Image.fromarray(self.bg.copy())
        d = ImageDraw.Draw(im, "RGBA")
        a0 = ST.fade(t, 0.2, 1e9)
        text(d, (W / 2, 210), "John Glen Wardrop, 1952", 30, MUTED, a0, "SemiBold", "ma")
        text(d, (W / 2, 290), "Traffic settles where no driver", 62, INK, a0, "Bold", "ma")
        text(d, (W / 2, 372), "can get there sooner by switching alone.", 62, INK, a0, "Bold", "ma")
        a1 = ST.fade(t, 5.5, 1e9)
        a2 = ST.fade(t, 8.0, 1e9)
        d.rounded_rectangle([420, 520, 1500, 720], 26, fill=ST.PANEL + (int(210 * max(a1, 0.001)),))
        text(d, (480, 575), "Routes people use", 40, INK, a1, "SemiBold", "lm")
        text(d, (1440, 575), "the same travel time", 40, GOLD, a1, "Bold", "rm")
        text(d, (480, 665), "Routes nobody uses", 40, INK, a2, "SemiBold", "lm")
        text(d, (1440, 665), "slower", 40, INK2, a2, "Bold", "rm")
        a3 = ST.fade(t, 11.5, 1e9)
        text(d, (W / 2, 800), "Game theorists call this a Nash equilibrium: a state nobody wants to leave on their own.", 32, INK2, a3, "Medium", "ma")
        text(d, (W / 2, 1010), "J. G. Wardrop (1952), Some theoretical aspects of road traffic research, Proc. Institution of Civil Engineers", 20, MUTED, a0, "Regular", "ma")
        return np.asarray(im)


# ================================================================== 7. drivers learning, day by day
class Learn(L.Section):
    name, dur = "learn", bars(18)
    R1, R2, SPD = 5.0, 21.5, 0.7                     # run starts; seconds of video per day

    def __init__(self):
        self.sc = Scene(LEFT)
        self.runs = [N["learn"]["chase"], N["learn"]["calm"]]
        self.cars = Cars(self.sc, self.state, self.dur)

    def run_day(self, t):
        if t < self.R2 - 0.6:
            return 0, min(20.0, max(0.0, (t - self.R1) / self.SPD))
        return 1, min(30.0, max(0.0, (t - self.R2) / self.SPD))

    def state(self, t):
        r, day = self.run_day(t)
        run = self.runs[r]
        n = len(run["flows"]) - 1
        d0 = min(int(day), n)
        d1 = min(d0 + 1, n)
        u = smooth(min(1.0, (day - d0) / 0.5))
        f = [lerp(a, b, u) for a, b in zip(run["flows"][d0], run["flows"][d1])]
        return f, [M.bpr(T0[i], CAP[i], f[i]) for i in range(3)]

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        r, day = self.run_day(t)
        f, T = self.state(t)
        for i in range(3):
            if f[i] > 1:
                sc.route(i, flow_width(f[i]), 0.95)
            else:
                sc.route(i, 2, 0.4, dashed=True)
        self.cars.draw(t)
        im = Image.fromarray(sc.compose())
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0, labels=False)
        sc.badges(d)
        eta_rows(d, 1110, 150, [(i, f"{T[i]:.0f} min" if T[i] <= 120 else "2 h+", f"{NAMES[i]} · {cars_h(f[i])}") for i in range(3)], size=30, gap=64)
        sub = ["half of all drivers switch to yesterday's quickest route", f"1 in {ONE_IN} rethink each day; nobody switches to save under a minute"][r]
        header(d, f"Day {int(day) + 1}", sub)
        run = self.runs[r]
        days = 21 if r == 0 else 31
        ax = L.Axes((1170, 440, 1830, 770), (1, days), (Y0, 120))
        a_ax = 1 - smooth((t - (self.R2 - 1.2)) / 0.5) if r == 0 else smooth((t - (self.R2 - 0.6)) / 0.5)
        ax.draw(d, range(1, days + 1, 5), range(Y0, 121, 20 if Y0 % 20 == 0 else 15), str, str, "Day", "Trip time on each route (min)", a_ax)
        k = min(int(day), len(run["times"]) - 1)
        xs = np.arange(1, k + 2)
        for i in range(3):
            ys = [run["times"][j][i] if run["flows"][j][i] > 1 else np.nan for j in range(k + 1)]
            ax.series(d, xs, ys, i=i, width=4, a=a_ax)
        ue = N["ue"]["level"]
        y = float(ax.Y(ue))
        L.dashed(d, [(ax.x0, y), (ax.x1, y)], INK2, 2, (8, 6), a_ax)
        text(d, (ax.x1, y - 10), f"equilibrium {ue:.1f} min", 22, INK2, a_ax, "Medium", "rs")
        text(d, (ax.x1, ax.y0 - 40), "lines above 120 min are cut off", 19, MUTED, a_ax if r == 0 else 0, "Regular", "rs")
        caps(d, t, [(0.3, 4.6, [cap("Nobody solves this equation. Drivers find it by trial and error.", 36)]),
                    (5.4, 11.4, [cap("If half of all drivers jump to yesterday's quickest route every morning...", 36)]),
                    (11.8, 19.8, [cap("...the jam just moves from road to road.", 36)]),
                    (21.6, 28.2, [cap(f"If 1 driver in {ONE_IN} rethinks each day, and nobody switches to save under a minute...", 34)]),
                    (28.6, 35.6, [cap(f"...the times settle within a minute of each other in about {N['learn']['calm']['settle_day'] + 1} days.", 36)]),
                    (36.0, 45.6, final_cap())])
        note(d, "A toy model of day-to-day learning; real drivers are more varied, and many never switch.")
        return np.asarray(im)


# ================================================================== 8. what the app shows
class App(L.Section):
    name, dur = "app", bars(14)

    def __init__(self):
        self.sc = Scene(RIGHT)
        v = self.sc.v
        vs = R["via_sample"]
        order = np.argsort([-x["minutes"] for x in vs])             # slowest drawn first, fastest on top
        self.via = [(vs[k]["minutes"], v.pts(vs[k]["xy"])) for k in order]
        self.vt = np.array([x[0] for x in self.via])
        self.top = sorted(range(len(R["routes"])), key=lambda k: R["routes"][k]["minutes"])[:3]
        self.topP = [v.pts(R["routes"][k]["xy"]) for k in self.top]
        self.topT = [R["routes"][k]["minutes"] for k in self.top]
        lo, hi = int(np.floor(self.vt.min())) - 1, int(np.ceil(self.vt.max())) + 1
        self.bins = np.arange(lo, hi + 1)
        self.cnt = np.histogram(self.vt, self.bins)[0]
        self.ymax = int(np.ceil(self.cnt.max() / 20) * 20)
        self.ax = L.Axes((110, 290, 880, 720), (lo, hi), (0, self.ymax))
        self.acc, self.acc_n = np.zeros((H, W), np.float32), 0

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        n = int(len(self.via) * smooth((t - 3.0) / 10.0))
        hi = smooth((t - 15.0) / 1.0)
        if n < self.acc_n:
            self.acc[:], self.acc_n = 0, 0
        for k in range(self.acc_n, n):                     # density: how many routes pass through each pixel
            P = self.via[k][1]
            x0, y0 = np.maximum(0, np.floor(P.min(0)).astype(int) - 3)
            x1, y1 = np.minimum([W, H], np.ceil(P.max(0)).astype(int) + 4)
            tmp = np.zeros((y1 - y0, x1 - x0), np.float32)
            p = np.rint((P - [x0, y0]) * 16).astype(np.int32)
            cv2.polylines(tmp, [p], False, 1.0, 2, cv2.LINE_AA, shift=4)
            self.acc[y0:y1, x0:x1] += tmp
        self.acc_n = n
        heat = (1 - np.exp(-self.acc / 6.0))[..., None] * np.array([255, 236, 205], np.float32) * (0.85 - 0.55 * hi)
        sc.glow.e += heat
        for k in range(3):
            sc.glow.line(self.topP[k], (255, 240, 210), 4, hi)
        im = Image.fromarray(sc.compose(dim=1.0 - 0.35 * hi))
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0, labels=False)
        header(d, "Every way across town", f"{len(self.via)} routes, each through a different junction · 3 a.m. times")
        ax = self.ax
        shown = self.vt[:n]
        cnt = np.histogram(shown, self.bins)[0] if n else np.zeros(len(self.bins) - 1)
        step = 10 if self.bins[-1] - self.bins[0] > 30 else 5
        ax.draw(d, range(int(np.ceil(self.bins[0] / step) * step), int(self.bins[-1]) + 1, step), range(0, self.ymax + 1, 20),
                lambda v: f"{v} min", str, "Trip time at 3 a.m.", "Number of routes")
        bw = (ax.x1 - ax.x0) / (len(self.bins) - 1)
        for j, c in enumerate(cnt):
            if c <= 0:
                continue
            x0 = ax.x0 + j * bw + 2
            y = float(ax.Y(min(c, self.ymax)))
            d.rounded_rectangle([x0, y, x0 + bw - 4, ax.y1], 4, fill=(110, 170, 230, int(230 * (1 - 0.5 * hi))))
        for k in range(3):
            x = float(ax.X(self.topT[k]))
            a = hi
            d.line([(x, ax.y0 + 40), (x, ax.y1)], fill=(255, 240, 210, int(255 * a)), width=3)
        text(d, (float(ax.X(self.topT[0])) + 10, ax.y0 + 30), "the app's three", 24, GOLDISH, hi, "SemiBold", "ls")
        for k in range(3):
            P = self.topP[k]
            x, y = L.at(P, L.polyline_cum(P), (0.30 + 0.17 * k) * L.polyline_cum(P)[-1])
            lab = f"{round(self.topT[k]):.0f} min"
            tw = d.textlength(lab, font=ST.font(30, "Bold"))
            bx = x + 30 if k else x - 30 - tw - 28
            d.rounded_rectangle([bx, y - 24, bx + tw + 28, y + 24], 24, fill=ST.PANEL + (int(220 * hi),), outline=(255, 240, 210, int(200 * hi)), width=2)
            text(d, (bx + 14, y), lab, 30, INK, hi, "Bold", "lm")
        caps(d, t, [(0.3, 3.8, [cap("There's a second reason the ETAs match.")]),
                    (4.2, 12.6, [cap(f"There are hundreds of ways from {N['origin']} to {N['dest']}. At 3 a.m. they take {N['app']['via_min']:.0f} to {N['app']['via_max']:.0f} minutes.", 34)]),
                    (15.2, 23.4, [cap("Your app shows you only the fastest few. Those are close together by definition,", 34),
                                  cap(f"even on empty roads: {round(self.topT[0])}, {round(self.topT[1])} and {round(self.topT[2])} minutes.", 34, GOLDISH)]),
                    (24.0, 35.6, [cap("Traffic pulls the routes people use together. The app hides the rest.", 36),
                                  cap("Both make the ETAs on your screen match.", 36, GOLD)])])
        return np.asarray(im)


# ================================================================== 9. everyone for themselves vs the best for everyone
class Optimum(L.Section):
    name, dur = "optimum", bars(10)

    def __init__(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = np.clip(1 - 0.38 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.75)) ** 2), 0.4, 1)
        self.bg = (L.BG[None, None] * v[..., None]).astype(np.uint8)

    def panel(self, d, x0, title, flows, times, a, avg):
        text(d, (x0 + 330, 200), title, 38, INK, a, "Bold", "ma")
        top = int(1000 * np.ceil(max(N["ue"]["flows"] + N["so"]["flows"]) * 1.08 / 1000))
        ax = L.Axes((x0 + 60, 300, x0 + 600, 700), (0, 3), (0, top))
        for v in range(0, top + 1, 1000):
            y = float(ax.Y(v))
            d.line([(ax.x0, y), (ax.x1, y)], fill=(40, 52, 72, int(255 * a)), width=1)
            text(d, (ax.x0 - 10, y), f"{v:,}", 20, MUTED, a, "Medium", "rm")
        text(d, (ax.x0, ax.y0 - 26), "cars per hour", 20, INK2, a, "Medium", "ls")
        for i in range(3):
            cx = float(ax.X(i + 0.5))
            y = float(ax.Y(flows[i]))
            d.rounded_rectangle([cx - 50, y, cx + 50, ax.y1], 6, fill=ROUTE[i] + (int(235 * a),))
            text(d, (cx, y - 12), fmt_int(flows[i]), 24, INK, a, "SemiBold", "ms")
            L.badge(d, cx, ax.y1 + 34, i, r=18, a=a)
            text(d, (cx, ax.y1 + 78), f"{times[i]:.1f} min", 28, INK, a, "Bold", "ma")
        text(d, (x0 + 330, 860), f"average trip {avg:.1f} min", 30, GOLD, a, "SemiBold", "ma")

    def frame(self, t):
        im = Image.fromarray(self.bg.copy())
        d = ImageDraw.Draw(im, "RGBA")
        ue, so = N["ue"], N["so"]
        self.panel(d, 200, "Everyone for themselves", ue["flows"], ue["times"], ST.fade(t, 0.3, 1e9), ue["avg"])
        self.panel(d, 1060, "A perfect traffic controller", so["flows"], so["times"], ST.fade(t, 5.8, 1e9), so["avg"])
        caps(d, t, [(0.4, 5.4, [cap("Is everyone choosing for themselves the best for everyone?", 36)]),
                    (5.8, 12.6, [cap("Not quite. A controller would send a few more drivers the slower way, so the rest save more.", 32)]),
                    (13.0, 19.6, [cap(f"Average trip: {ue['avg']:.1f} → {so['avg']:.1f} minutes. Here, choosing for yourself costs just {100 * (N['poa'] - 1):.1f}%.", 34)]),
                    (20.0, 25.6, [cap("Economists call that gap the price of anarchy. Sometimes it's much bigger.", 34, GOLD)])],)
        note(d, f"Toy model at rush hour ({fmt_int(N['d_rush'])} drivers per hour). The gap peaks at {100 * (N['poa_max']['value'] - 1):.0f}% in lighter traffic.")
        return np.asarray(im)


# ================================================================== 10. Braess's paradox
class Braess(L.Section):
    name, dur = "braess", bars(18)
    NODES = {"S": (430, 520), "A": (960, 250), "B": (960, 790), "T": (1490, 520)}
    PATHS = {"S-A-T": "SAT", "S-B-T": "SBT", "S-A-B-T": "SABT"}
    OPEN0, SW0, SW1 = 12.0, 16.0, 26.0

    def __init__(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = np.clip(1 - 0.38 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.52) / (H * 0.75)) ** 2), 0.4, 1)
        self.bg = (L.BG[None, None] * v[..., None]).astype(np.float32)
        self.glow = L.Glow()
        self.poly = {k: self.path_pts(v) for k, v in self.PATHS.items()}
        self.cum = {k: L.polyline_cum(p) for k, p in self.poly.items()}
        ts = np.arange(0, self.dur + 2 / FPS, 1 / FPS)
        sp = np.array([[1.0 / (self.times(t)[k] * 0.11) for k in self.PATHS] for t in ts])
        self.u = np.cumsum(sp, axis=0) / FPS

    def path_pts(self, nodes):
        P = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            (x0, y0), (x1, y1) = self.NODES[a], self.NODES[b]
            for u in np.linspace(0, 1, 40):
                P.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
        return np.array(P)

    def x_short(self, t):
        return 4000.0 * smooth((t - self.SW0) / (self.SW1 - self.SW0))

    def flows(self, t):
        x = self.x_short(t)
        return {"S-A-T": (4000 - x) / 2, "S-B-T": (4000 - x) / 2, "S-A-B-T": x}

    def times(self, t):
        f = self.flows(t)
        sa = f["S-A-T"] + f["S-A-B-T"]
        bt = f["S-B-T"] + f["S-A-B-T"]
        return {"S-A-T": sa / 100 + 45, "S-B-T": 45 + bt / 100, "S-A-B-T": sa / 100 + bt / 100}

    def frame(self, t):
        g = self.glow
        g.clear()
        f, T = self.flows(t), self.times(t)
        conn = smooth((t - self.OPEN0) / 1.5)
        a0 = smooth(t / 0.8)
        edges = [("S", "A", "cars ÷ 100 min"), ("A", "T", "45 min"), ("S", "B", "45 min"), ("B", "T", "cars ÷ 100 min")]
        for a, b, _ in edges:
            g.line([self.NODES[a], self.NODES[b]], (60, 90, 130), 26, a0 * 0.8)
        if conn > 0:
            (x0, y0), (x1, y1) = self.NODES["A"], self.NODES["B"]
            g.line([(x0, y0), (x0, y0 + (y1 - y0) * conn)], (255, 214, 140), 14, 0.9)
        k = min(len(self.u) - 1, int(round(t * FPS)))
        for j, p in enumerate(self.PATHS):
            n = int(round(f[p] * T[p] / 60 / 30))
            if n <= 0:
                continue
            s = (Cars.PHI[:n] + self.u[k, j]) % 1.0
            Ls = s * self.cum[p][-1]
            xs = np.interp(Ls, self.cum[p], self.poly[p][:, 0])
            ys = np.interp(Ls, self.cum[p], self.poly[p][:, 1])
            col = (255, 230, 190) if p == "S-A-B-T" else (200, 225, 255)
            g.dots(zip(xs, ys), col, 4, a0)
        img = g.compose(self.bg)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        for a, b, lab in edges:
            (x0, y0), (x1, y1) = self.NODES[a], self.NODES[b]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            L_ = math.hypot(x1 - x0, y1 - y0)
            nx, ny = -(y1 - y0) / L_, (x1 - x0) / L_
            if nx * (mx - 960) + ny * (my - 520) < 0:
                nx, ny = -nx, -ny
            text(d, (mx + nx * 80, my + ny * 80), lab, 30, INK2, a0, "SemiBold", "mm")
        if conn > 0:
            text(d, (990, 520), "shortcut: 0 min", 30, GOLDISH, conn, "Bold", "lm")
        for nm, (x, y) in self.NODES.items():
            d.ellipse([x - 34, y - 34, x + 34, y + 34], fill=(240, 244, 250, int(255 * a0)))
            text(d, (x, y + 2), nm, 36, (8, 12, 20), a0, "Bold", "mm")
        text(d, (self.NODES["S"][0], self.NODES["S"][1] + 60), "start", 24, MUTED, a0, "Medium", "ma")
        text(d, (self.NODES["T"][0], self.NODES["T"][1] + 60), "destination", 24, MUTED, a0, "Medium", "ma")
        # path board
        x0 = 1580
        text(d, (x0, 160), "4,000 DRIVERS", 24, MUTED, a0, "SemiBold")
        rows = [("S-A-T", "via A"), ("S-B-T", "via B"), ("S-A-B-T", "via the shortcut")]
        for r, (p, lab) in enumerate(rows):
            if p == "S-A-B-T" and conn <= 0:
                continue
            a = a0 if p != "S-A-B-T" else conn
            y = 210 + r * 96
            text(d, (x0, y), lab, 28, INK, a, "SemiBold")
            text(d, (x0, y + 36), f"{fmt_int(f[p])} drivers · {T[p]:.0f} min", 26, GOLDISH if f[p] > 0.5 else MUTED, a, "Medium")
        avg = sum(f[p] * T[p] for p in f) / 4000
        text(d, (x0, 520), "AVERAGE TRIP", 24, MUTED, a0, "SemiBold")
        text(d, (x0, 552), f"{avg:.0f} min", 72, GOLD, a0, "Bold")
        caps(d, t, [(0.3, 4.0, [cap("Sometimes the price of choosing for yourself is much higher.", 36)]),
                    (4.4, 11.6, [cap("4,000 drivers, two routes. They split evenly, and every trip takes 65 minutes.", 36)]),
                    (12.0, 15.8, [cap("Now add a shortcut: a free road from A to B.", 36)]),
                    (16.2, 25.8, [cap("Each driver who takes it gets there sooner, so more and more drivers switch...", 34)]),
                    (26.2, 31.0, [cap("...until everyone takes the shortcut. Now every trip takes 80 minutes.", 36)]),
                    (31.4, 38.0, [cap("A new road made everyone 15 minutes slower. Closing it would get everyone home sooner.", 34, GOLD)]),
                    (38.4, 45.6, [cap("Braess's paradox (Dietrich Braess, 1968). It can happen, but it isn't the rule:", 34),
                                  cap("with 2,000 drivers, the same shortcut saves everyone 15 minutes.", 34, INK2)])])
        note(d, "The classic textbook example; link times are toy numbers.", a0)
        return np.asarray(im)


# ================================================================== 11. the end
class Outro(L.Section):
    name, dur = "outro", bars(10)

    def __init__(self):
        self.sc = Scene(RIGHT)
        self.cars = Cars(self.sc, lambda t: (N["learn"]["calm"]["flows"][-1], N["learn"]["calm"]["times"][-1]), self.dur)

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        for i in range(3):
            sc.route(i, flow_width(N["ue"]["flows"][i]), 0.95)
        self.cars.draw(t)
        img = sc.compose(dim=1.0 - 0.6 * smooth((t - 22.5) / 2.5))
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0)
        sc.badges(d)
        x = 70
        a1 = ST.fade(t, 0.5, 1e9)
        text(d, (x, 110), "So when your app offers", 52, INK, a1, "Bold")
        text(d, (x, 178), "three completely different", 52, INK, a1, "Bold")
        text(d, (x, 246), "routes with the same ETA:", 52, INK, a1, "Bold")
        eta_rows(d, x, 360, [(i, f"{N['eta_final'][i]} min", NAMES[i]) for i in range(3)], a=a1, size=28)
        a2 = ST.fade(t, 5.5, 1e9)
        text(d, (x, 540), "you're not the only one.", 52, GOLD, a2, "Bold")
        a3 = ST.fade(t, 9.0, 1e9)
        for k, s in enumerate(["Thousands of drivers, each taking the fastest", "way they know, made those times match. And", "the app only showed you the close calls."]):
            text(d, (x, 640 + 44 * k), s, 32, INK2, a3, "Medium")
        a4 = ST.fade(t, 15.5, 1e9)
        text(d, (x, 820), "Try it yourself: the traffic lab is linked below.", 32, INK, a4, "SemiBold")
        a5 = ST.fade(t, 19.0, 1e9)
        for k, s in enumerate(["Roads © OpenStreetMap contributors, via Overture Maps. Free-flow times: real roads at the speed limit.",
                               f"Traffic: a toy model with made-up capacities, not {N['city_name']} traffic data. Music: original, synthesized for this video."]):
            text(d, (x, 990 + 30 * k), s, 20, MUTED, a5, "Regular")
        return np.asarray(im)


SECTIONS = {c.name: c for c in (Hook, Light, Empty, Curve, Fill, Wardrop, Learn, App, Optimum, Braess, Outro)}
ORDER = ["hook", "light", "empty", "curve", "fill", "wardrop", "learn", "app", "optimum", "braess", "outro"]

if __name__ == "__main__":
    s = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for v in sys.argv[2:]:
            s.render(still=float(v))
    else:
        s.render()
