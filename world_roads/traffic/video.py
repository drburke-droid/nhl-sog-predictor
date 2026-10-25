"""Three routes, one ETA: the video (1920x1080, 30 fps, caption-only). Every number comes from scenario.py.

    python video.py NAME [still times...]
    NAME: hook, light, empty, curve, fill, why, fine, learn, app, optimum, myth, outro
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


# ================================================================== 6. why the times end up equal
class Why(L.Section):
    name, dur = "why", bars(12)
    MOVE0, MOVE1 = 5.0, 12.5                         # drivers switching from the unbalanced start to the equilibrium

    def __init__(self):
        self.sc = Scene(RIGHT)
        self.f0, self.f1 = N["why_start"]["flows"], N["ue"]["flows"]
        self.cars = Cars(self.sc, self.state, self.dur)

    def state(self, t):
        u = smooth((t - self.MOVE0) / (self.MOVE1 - self.MOVE0))
        f = [lerp(a, b, u) for a, b in zip(self.f0, self.f1)]
        return f, [M.bpr(T0[i], CAP[i], f[i]) for i in range(3)]

    def frame(self, t):
        sc = self.sc
        sc.glow.clear()
        f, T = self.state(t)
        for i in range(3):
            sc.route(i, flow_width(f[i]), 0.95)
        self.cars.draw(t)
        im = Image.fromarray(sc.compose())
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0)
        sc.badges(d)
        x, a0 = 70, smooth(t / 0.6)
        text(d, (x, 96), "Why do the times", 56, INK, a0, "Bold")
        text(d, (x, 162), "end up equal?", 56, INK, a0, "Bold")
        text(d, (x, 270), "RUSH HOUR · TOY MODEL", 22, MUTED, a0, "SemiBold")
        for i in range(3):
            y = 336 + i * 66
            L.pill(d, x, y, i, f"{T[i]:.0f} min", f"{NAMES[i]} · {cars_h(f[i])}", a=a0, size=30, anchor="lm")
            dlt = f[i] - self.f0[i]
            if abs(dlt) > 20 and t < 17.0:
                text(d, (x + 560, y), f"{'+' if dlt > 0 else '−'}{fmt_int(abs(dlt))} cars/h", 24, GOLDISH if dlt > 0 else INK2,
                     smooth((t - self.MOVE0) / 1.0) * (1 - smooth((t - 16.4) / 0.6)), "SemiBold", "lm")
        aw = ST.fade(t, 17.0, 1e9)
        d.rounded_rectangle([x - 10, 530, x + 830, 810], 22, fill=ST.PANEL + (int(210 * aw),))
        text(d, (x + 20, 562), "WARDROP'S PRINCIPLE (1952)", 22, MUTED, aw, "SemiBold")
        text(d, (x + 20, 602), "Traffic settles where no driver can get", 34, INK, aw, "Bold")
        text(d, (x + 20, 646), "there sooner by switching on their own.", 34, INK, aw, "Bold")
        text(d, (x + 20, 718), "Routes people use", 30, INK, aw, "SemiBold", "lm")
        text(d, (x + 800, 718), "the same time", 30, GOLD, aw, "Bold", "rm")
        text(d, (x + 20, 772), "Routes nobody uses", 30, INK, aw, "SemiBold", "lm")
        text(d, (x + 800, 772), "slower", 30, INK2, aw, "Bold", "rm")
        ts = N["why_start"]["times"]
        caps(d, t, [(0.4, 4.8, [cap(f"Suppose {IN[0]} showed {ts[0]:.0f} minutes, and the other two {ts[1]:.0f}.", 36)]),
                    (5.2, 11.8, [cap("Drivers leave the slower routes for the quicker one.", 34),
                                 cap("It slows down as they arrive, and the others speed up.", 34, INK2)]),
                    (12.2, 16.6, [cap("They meet in the middle. Any gap left over would tempt more drivers to switch, so it closes.", 32)]),
                    (17.0, 23.6, [cap("That's Wardrop's principle: a state no driver can improve on by switching alone.", 34)]),
                    (24.0, 30.6, [cap("Game theorists call it a Nash equilibrium. Nobody plans it: it's what's left", 34),
                                  cap("once everyone has picked the quickest route they know.", 34, INK2)])])
        return np.asarray(im)


class Fine(L.Section):
    name, dur = "fine", bars(12)
    AT = [1.0, 7.5, 14.0, 20.5]

    def __init__(self):
        sc = Scene(CENTER)
        self.bg = cv2.GaussianBlur(sc.base * 0.3, (0, 0), 4).astype(np.uint8)

    def frame(self, t):
        im = Image.fromarray(self.bg.copy())
        d = ImageDraw.Draw(im, "RGBA")
        a0 = smooth(t / 0.6)
        text(d, (W / 2, 96), "The fine print", 60, INK, a0, "Bold", "ma")
        text(d, (W / 2, 176), "\u201cthe same ETA\u201d is true only in a narrow sense", 30, INK2, a0, "Medium", "ma")
        items = [("It only applies to routes people actually use.",
                  f"At 3 a.m. everyone takes {IN[0]} ({T0[0]:.0f} min). The other two stay slower ({T0[1]:.0f} and {T0[2]:.0f}), and nothing evens them out."),
                 ("It's about the same trip, at the same time.",
                  "A driver heading somewhere else, or you an hour earlier, faces a different balance."),
                 ("\u201cEqual\u201d means close enough.",
                  "Nobody switches to save 30 seconds, ETAs are predictions, and apps round to the minute."),
                 ("It isn't the best outcome for everyone.",
                  "Each driver picks what's quickest for them. As we'll see, that can leave everyone slower.")]
        for k, (head, body) in enumerate(items):
            a = smooth((t - self.AT[k]) / 0.6)
            y = 280 + k * 175
            d.rounded_rectangle([280, y - 14, 1640, y + 136], 20, fill=ST.PANEL + (int(200 * a),))
            d.ellipse([310, y + 8, 360, y + 58], fill=GOLD + (int(255 * a),))
            text(d, (335, y + 34), str(k + 1), 30, (10, 12, 20), a, "Bold", "mm")
            text(d, (390, y + 10), head, 38, INK, a, "Bold")
            for j, line in enumerate(wrap(body, 28, 1210)):
                text(d, (390, y + 66 + 34 * j), line, 28, INK2, a, "Medium")
        return np.asarray(im)


# ================================================================== 7. drivers learning, day by day
class Learn(L.Section):
    name, dur = "learn", bars(20)
    R1, D2, D3, SPD1 = 5.0, 11.0, 17.0, 0.4          # the chasing run: day 1 held, day 2 held, then 0.4 s a day
    R2, SPD = 25.5, 0.55                             # the calm run: start, seconds of video per day

    def __init__(self):
        self.sc = Scene(LEFT)
        self.runs = [N["learn"]["chase"], N["learn"]["calm"]]
        self.cars = Cars(self.sc, self.state, self.dur)

    def run_day(self, t):
        if t < self.R2 - 0.6:
            day = 0.0 if t < self.D2 else min(1.0, (t - self.D2) / 0.6) if t < self.D3 else 1.0 + (t - self.D3) / self.SPD1
            return 0, min(20.0, day)
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
        sub = ["half of everyone switches to yesterday's quickest route", f"1 in {ONE_IN} try another route each day; nobody switches to save under a minute"][r]
        n = int(day) + 1
        header(d, f"Day {n}" + (f" · {('Monday', 'Tuesday')[n - 1]}" if r == 0 and n <= 2 else ""), sub)
        run = self.runs[r]
        if r == 0:                                       # where the crowd goes next, and where it just went
            k0 = int(day)
            yq = int(np.argmin(run["times"][max(0, k0 - 1)]))
            lab = "quickest today" if k0 == 0 else "yesterday's quickest"
            at = ST.fade(t, self.R1 + 1.0, self.R2 - 1.2)
            text(d, (1600, 150 + 64 * yq), "← " + lab, 22, GOLDISH, at, "SemiBold", "lm")
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
        c0 = [round(v) for v in self.runs[0]["times"][0]]
        caps(d, t, [(0.3, 4.6, [cap("Nobody works this balance out. So how do drivers find it?", 36),
                                cap("Imagine a week when it hasn't settled yet.", 36, INK2)]),
                    (5.0, 10.6, [cap(f"Say on Monday most drivers take {IN[0]}: it's the quickest on empty roads.", 34),
                                 cap(f"So it jams: {c0[0]} minutes, while {IN[1]} takes {c0[1]}.", 34, INK2)]),
                    (11.0, 16.6, [cap("Drivers find out their route wasn't the quickest. If half of everyone", 34),
                                  cap(f"switches to {IN[1]} on Tuesday, it jams instead.", 34, INK2)]),
                    (17.0, 24.6, [cap("Day after day, the crowd chases yesterday's quickest route,", 34),
                                  cap("and the jam just moves from road to road.", 34, INK2)]),
                    (25.5, 32.6, [cap(f"Real drivers are calmer. Say only 1 in {ONE_IN} tries another route each day,", 34),
                                  cap("and nobody switches to save under a minute...", 34, INK2)]),
                    (33.0, 41.6, [cap(f"...then the times creep together, and settle within a minute in about {N['learn']['calm']['settle_day'] + 1} days.", 34)]),
                    (42.0, 50.4, final_cap())])
        note(d, "A toy model of day-to-day learning; real drivers are more varied, and many never switch.")
        return np.asarray(im)


# ================================================================== 8. what the app shows
class App(L.Section):
    name, dur = "app", bars(15)
    HI0 = 13.5                                        # the spaghetti dims and our three routes come back

    def __init__(self):
        self.sc = Scene(RIGHT)
        v = self.sc.v
        vs = R["via_sample"]
        order = np.argsort([-x["minutes"] for x in vs])             # slowest drawn first, fastest on top
        self.via = [(vs[k]["minutes"], v.pts(vs[k]["xy"])) for k in order]
        self.vt = np.array([x[0] for x in self.via])
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
        hi = smooth((t - self.HI0) / 1.0)
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
        heat = (1 - np.exp(-self.acc / 6.0))[..., None] * np.array([255, 236, 205], np.float32) * (0.85 - 0.6 * hi)
        sc.glow.e += heat
        for i in range(3):                                 # our three, the same routes as all along
            sc.route(i, 5, 0.95 * hi)
        im = Image.fromarray(sc.compose(dim=1.0 - 0.35 * hi))
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0, labels=False)
        sc.badges(d, a=hi)
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
            d.rounded_rectangle([x0, y, x0 + bw - 4, ax.y1], 4, fill=(110, 170, 230, int(230 * (1 - 0.55 * hi))))
        for i in range(3):                                 # where our three sit among them
            x = float(ax.X(T0[i]))
            yb = ax.y0 + (32, 80, 32)[i]                   # staggered so close times don't collide
            d.line([(x, yb + 18), (x, ax.y1)], fill=tuple(ROUTE[i]) + (int(255 * hi),), width=4)
            L.badge(d, x, yb, i, r=17, a=hi)
            text(d, (x + (-24 if i == 1 else 24), yb + 1), f"{T0[i]:.0f} min", 22, INK, hi, "Bold", "rm" if i == 1 else "lm")
        caps(d, t, [(0.3, 3.8, [cap("There's a second reason the ETAs match.")]),
                    (4.2, 12.6, [cap(f"There are hundreds of ways from {N['origin']} to {N['dest']}. At 3 a.m. they take {N['app']['via_min']:.0f} to {N['app']['via_max']:.0f} minutes.", 34)]),
                    (13.0, 20.6, [cap("Your app never shows you hundreds. It offers a few that are quick and", 34),
                                  cap("genuinely different, like our three: the quickest through each crossing.", 34, INK2)]),
                    (21.0, 28.6, [cap("The slow ones, and near-copies of quicker ones, never make the list.", 34),
                                  cap("So the routes you see are already the close calls.", 34, INK2)]),
                    (29.0, 37.6, [cap("Traffic pulls the routes people use together. The app hides the rest.", 36),
                                  cap("Both make the ETAs on your screen match.", 36, GOLD)])])
        return np.asarray(im)


# ================================================================== 9. everyone for themselves vs the best for everyone
class Optimum(L.Section):
    name, dur = "optimum", bars(12)

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
                    (13.0, 19.6, [cap("Why? Joining a busy road slows everyone behind you, and nobody counts that cost.", 34)]),
                    (20.0, 25.4, [cap(f"Here the gap is small: {ue['avg']:.1f} → {so['avg']:.1f} minutes on average, just {100 * (N['poa'] - 1):.1f}%.", 34)]),
                    (25.8, 30.6, [cap("Economists call it the price of anarchy. A new road can make it much bigger.", 34, GOLD)])],)
        note(d, f"Toy model at rush hour ({fmt_int(N['d_rush'])} drivers per hour). The gap peaks at {100 * (N['poa_max']['value'] - 1):.0f}% in lighter traffic.")
        return np.asarray(im)


# ================================================================== 10. Braess's paradox, on the map
def piece(P, cum, L0, L1):
    """the polyline between arc lengths L0 and L1"""
    i0, i1 = int(np.searchsorted(cum, L0)), int(np.searchsorted(cum, L1))
    return np.vstack([L.at(P, cum, L0)] + [P[k] for k in range(i0, i1)] + [L.at(P, cum, L1)])


class Neck:
    """the lane-drop diagram: 4 lanes squeeze into 2. A toy queue, simulated once for every frame: cars stop
    behind the car ahead, and the merge lets one car through every 1/MU seconds, whichever has waited longest"""
    X0, XM, XT, X1, CY, LW = 110, 470, 560, 860, 500, 34
    V, GAP, MU, CW, CH = 220.0, 27.0, 5.0, 22, 13
    RED = (255, 96, 86)

    def __init__(self, t0, t1, heavy0, lo=3.0, hi=9.0):
        self.t0 = t0
        dt, t = 1.0 / FPS, t0 - 4.0                       # start early so the road is busy when it fades in
        cars, self.frames, nxt, acc, k = [], [], -1e9, 0.0, 0
        while t < t1 + dt:
            acc += lerp(lo, hi, smooth((t - heavy0) / 1.0)) * dt
            while acc >= 1.0:
                ln = (0, 2, 1, 3)[k % 4]
                last = [c for c in cars if c["lane"] == ln and not c["out"]]
                if last and min(c["x"] for c in last) < self.X0 - 20 + self.GAP:
                    break                                 # the queue has backed up to the start
                cars.append(dict(lane=ln, x=self.X0 - 20.0, out=False, v=self.V, wait=None))
                acc -= 1.0
                k += 1
            fronts = []
            for ln in range(4):
                q = sorted((c for c in cars if c["lane"] == ln and not c["out"]), key=lambda c: -c["x"])
                for j, c in enumerate(q):
                    lim = q[j - 1]["x"] - self.GAP if j else self.XM
                    x1 = min(c["x"] + self.V * dt, lim)
                    if not j and c["x"] + self.V * dt >= self.XM:
                        c["wait"] = t if c["wait"] is None else c["wait"]
                        fronts.append(c)
                    c["v"], c["x"] = (x1 - c["x"]) / dt, max(c["x"], x1)
            if fronts and t >= nxt:
                c = min(fronts, key=lambda c: c["wait"])
                c["out"], nxt = True, t + 1.0 / self.MU
            for side in (0, 1):
                q = sorted((c for c in cars if c["out"] and (c["lane"] >= 2) == side), key=lambda c: -c["x"])
                for j, c in enumerate(q):
                    x1 = c["x"] + self.V * dt
                    if j:
                        x1 = min(x1, q[j - 1]["x"] - self.GAP)
                    c["v"], c["x"] = (x1 - c["x"]) / dt, max(c["x"], x1)
            cars = [c for c in cars if c["x"] < self.X1 + 20]
            self.frames.append([(c["x"], self.y(c), 1.0 - min(1.0, c["v"] / self.V), not c["out"]) for c in cars])
            t += dt
        waits = [any(q and s > 0.6 for _, _, s, q in fr) for fr in self.frames]
        self.queue_at = next((t0 - 4.0 + k * dt for k, w in enumerate(waits) if w and t0 - 4.0 + k * dt > heavy0), t1)

    def y(self, c):
        y0 = self.CY + (c["lane"] - 1.5) * self.LW
        y1 = self.CY + (0.5 if c["lane"] >= 2 else -0.5) * self.LW
        return lerp(y0, y1, smooth((c["x"] - self.XM) / (self.XT - self.XM)))

    def half(self, x):
        return lerp(2 * self.LW, self.LW, min(1.0, max(0.0, (x - self.XM) / (self.XT - self.XM))))

    def draw(self, d, t, a, heavy):
        if a <= 0:
            return
        cy = self.CY
        d.rounded_rectangle([70, 300, 900, 800], 22, fill=ST.PANEL + (int(215 * a),))
        text(d, (100, 336), "WHAT'S A BOTTLENECK?", 22, MUTED, a, "SemiBold")
        xs = [self.X0 - 30, self.XM, self.XT, self.X1 + 30]
        top = [(x, cy - self.half(x)) for x in xs]
        bot = [(x, cy + self.half(x)) for x in xs[::-1]]
        d.polygon(top + bot, fill=(36, 42, 56, int(255 * a)))
        for P in (top, [(x, cy + self.half(x)) for x in xs]):
            d.line(P, fill=(190, 198, 214, int(230 * a)), width=3)
        for off in (-1, 1):                                       # the lanes that end
            L.dashed(d, [(self.X0 - 30, cy + off * self.LW), (self.XM, cy + off * self.LW)], (150, 160, 180), 2, (12, 10), a)
        L.dashed(d, [(self.X0 - 30, cy), (self.X1 + 30, cy)], (150, 160, 180), 2, (12, 10), a)
        text(d, ((self.X0 + self.XM) / 2, cy - 2 * self.LW - 24), "4 LANES", 22, INK2, a, "SemiBold", "mm")
        text(d, ((self.XT + self.X1) / 2, cy - self.LW - 24), "2 LANES", 22, INK2, a, "SemiBold", "mm")
        bxm = (self.XM + self.XT) / 2
        d.polygon([(bxm, cy + 2 * self.LW + 10), (bxm - 9, cy + 2 * self.LW + 24), (bxm + 9, cy + 2 * self.LW + 24)], fill=GOLDISH + (int(255 * a),))
        text(d, (bxm, cy + 2 * self.LW + 40), "THE BOTTLENECK", 20, GOLDISH, a, "Bold", "mm")
        k = max(0, min(len(self.frames) - 1, int(round((t - self.t0 + 4.0) * FPS))))
        waiting = 0
        for x, y, stop, queued in self.frames[k]:
            ca = a * min(1.0, max(0.0, (x - self.X0 + 30) / 30.0)) * min(1.0, max(0.0, (self.X1 + 30 - x) / 30.0))
            col = tuple(int(lerp(c0, c1, stop)) for c0, c1 in zip((236, 242, 252), self.RED))
            d.rounded_rectangle([x - self.CW / 2, y - self.CH / 2, x + self.CW / 2, y + self.CH / 2], 4, fill=col + (int(255 * ca),))
            waiting += queued and stop > 0.6
        lab, col, body = (("HEAVY TRAFFIC", self.RED, "More cars arrive than 2 lanes can pass. The queue grows.") if heavy else
                          ("LIGHT TRAFFIC", (130, 226, 170), "Fewer cars arrive than 2 lanes can pass. No waiting."))
        text(d, (100, 662), lab, 26, col, a, "Bold", "lm")
        text(d, (100, 704), body, 26, INK, a, "Medium", "lm")
        text(d, (100, 754), "Cars waiting", 24, INK2, a, "Medium", "lm")
        bw = min(560, waiting * 14)
        d.rounded_rectangle([260, 742, 260 + max(bw, 6), 766], 8, fill=self.RED + (int(230 * a * (waiting > 0)),))
        text(d, (280 + bw, 754), str(waiting), 24, INK, a, "Bold", "lm")


class Myth(L.Section):
    name, dur = "myth", bars(40)
    NECK0, HEAVY0, NECK1 = 6.0, 14.8, 26.0                     # the lane-drop diagram: in, traffic gets heavy, out
    HI0, OPEN0, NIGHT0, FIRST0, SW0, SW1, CHART0 = 26.0, 41.0, 48.0, 54.0, 60.0, 70.0, 92.0
    PER = 15.0                                                  # vehicles per dot

    def __init__(self):
        m = self.m = N["myth"]
        cfg = m["cfg"]
        sc = self.sc = Scene(RIGHT)
        X, Y = self.X, self.Y = cfg["from_route"], cfg["to_route"]
        bx, by = m["bottleneck_from"], m["bottleneck_to"]
        lx0, lx1 = np.interp([bx[1], bx[2]], sc.tmin[X], sc.cum[X])
        ly0, ly1 = np.interp([by[1], by[2]], sc.tmin[Y], sc.cum[Y])
        self.bn = [piece(sc.P[X], sc.cum[X], lx0, lx1), piece(sc.P[Y], sc.cum[Y], ly0, ly1)]
        self.bn_names = [bx[0], by[0]]
        self.Pp, self.Qp = L.at(sc.P[X], sc.cum[X], lx1), L.at(sc.P[Y], sc.cum[Y], ly0)
        bore = np.linspace(self.Pp, self.Qp, 60)
        self.bore, self.bore_cum = bore, L.polyline_cum(bore)
        tunnel = np.vstack([L.partial(sc.P[X], sc.cum[X], lx1), bore, piece(sc.P[Y], sc.cum[Y], ly0, sc.cum[Y][-1])])
        self.paths = [sc.P[X], tunnel, sc.P[Y]]
        self.pcum = [L.polyline_cum(p) for p in self.paths]
        self.cols = [ROUTE[X], GOLDISH, ROUTE[Y]]
        self.dash = dash_polys(bore, self.bore_cum, 14, 10)
        self.bt0 = [bx[2] - bx[1], by[2] - by[1]]                # each bottleneck's time when empty
        self.bcap = [cfg["cap_from"], cfg["cap_to"]]
        self.neck = Neck(self.NECK0, self.NECK1 + 1.0, self.HEAVY0)
        ts = np.arange(0, self.dur + 2 / FPS, 1 / FPS)
        sp = np.array([[1.0 / max(1e-6, T * 0.25) for T in self.state(t)[1]] for t in ts])
        self.u = np.cumsum(sp, axis=0) / FPS

    def state(self, t):
        """flows and trip times on (old route X, the tunnel route, old route Y)"""
        st = self.m["steps"]
        u = smooth((t - self.SW0) / (self.SW1 - self.SW0)) * (len(st) - 1)
        k = min(int(u), len(st) - 2)
        w = u - k
        f = [lerp(a, b, w) for a, b in zip(st[k]["flows"], st[k + 1]["flows"])]
        T = [lerp(a, b, w) for a, b in zip(st[k]["times"], st[k + 1]["times"])]
        return f, T

    def bottleneck_times(self, f):
        """minutes to get through each bottleneck: the first carries old route X and the tunnel route, the second
        the tunnel route and old route Y"""
        return [M.bpr(self.bt0[0], self.bcap[0], f[0] + f[1]), M.bpr(self.bt0[1], self.bcap[1], f[1] + f[2])]

    def frame(self, t):
        sc, m = self.sc, self.m
        g = sc.glow
        g.clear()
        f, T = self.state(t)
        opened = smooth((t - self.OPEN0) / 2.0)
        aside = smooth((t - 1.0) / 1.5)
        other = 3 - self.X - self.Y
        sc.route(other, 2, 0.95 * (1 - aside) + 0.1 * aside)
        sc.route(self.X, flow_width(f[0] + f[1]), 0.95)
        sc.route(self.Y, flow_width(f[1] + f[2]), 0.95)
        hi = smooth((t - self.HI0) / 1.0)
        pulse = 0.75 + 0.25 * math.sin(2 * math.pi * t / 1.6)
        for P in self.bn:
            g.line(P, (255, 246, 230), 11, 0.9 * hi * pulse)
        if opened > 0:
            reach = opened * self.bore_cum[-1]
            for seg in self.dash:
                if np.hypot(*(seg[0] - self.bore[0])) <= reach:
                    g.line(seg, GOLDISH, 6, 1.0)
        k = min(len(self.u) - 1, int(round(t * FPS)))
        for j in range(3):
            if f[j] <= 0.5:
                continue
            n = int(min(len(Cars.PHI), round(f[j] * T[j] / 60.0 / self.PER)))
            s_ = (Cars.PHI[:n] + self.u[k, j]) % 1.0
            Ls = s_ * self.pcum[j][-1]
            xs = np.interp(Ls, self.pcum[j], self.paths[j][:, 0])
            ys = np.interp(Ls, self.pcum[j], self.paths[j][:, 1])
            col = tuple(min(255, c * 0.6 + 110) for c in self.cols[j])
            g.dots(zip(xs, ys), col, 3.0, 1.0)
        im = Image.fromarray(sc.compose())
        d = ImageDraw.Draw(im, "RGBA")
        sc.pins(d, 1.0)
        sc.badges(d, which=(self.X, self.Y))
        # labels on the map, with the time it takes to get through each bottleneck
        for P, nm, side, tb in zip(self.bn, self.bn_names, (1, -1), self.bottleneck_times(f)):
            x, y = P[len(P) // 2]
            lab = f"BOTTLENECK · {nm} · {tb:.0f} min"
            tw = d.textlength(lab, font=ST.font(22, "SemiBold"))
            bx = x + 30 if side > 0 else x - 30 - tw - 24
            bx = min(max(bx, 960), W - tw - 40)
            d.rounded_rectangle([bx, y - 50, bx + tw + 24, y - 14], 12, fill=ST.PANEL + (int(220 * hi),), outline=(255, 246, 230, int(160 * hi)), width=1)
            text(d, (bx + 12, y - 32), lab, 22, INK, hi, "SemiBold", "lm")
        if opened > 0:
            mx, my = self.bore[len(self.bore) // 2]
            lab = f"MYTHICAL TUNNEL · {m['cfg']['minutes']:.0f} min, never any traffic"
            tw = d.textlength(lab, font=ST.font(24, "Bold"))
            d.rounded_rectangle([mx - tw / 2 - 14, my + 18, mx + tw / 2 + 14, my + 58], 14, fill=ST.PANEL + (int(225 * opened),), outline=GOLDISH + (int(220 * opened),), width=2)
            text(d, (mx, my + 38), lab, 24, GOLDISH, opened, "Bold", "mm")
        # the bottleneck diagram, then the board
        an = smooth((t - self.NECK0) / 0.6) * (1 - smooth((t - self.NECK1) / 0.6))
        self.neck.draw(d, t, an, t >= self.neck.queue_at - 0.3)          # "heavy" once the queue starts
        x, a0 = 70, smooth(t / 0.6)
        ab0 = smooth((t - self.NECK1 - 0.3) / 0.6)
        text(d, (x, 96), "Can a new road make", 52, INK, a0, "Bold")
        text(d, (x, 158), "everyone slower?", 52, INK, a0, "Bold")
        text(d, (x, 262), f"THOUGHT EXPERIMENT · {fmt_int(m['cfg']['demand'])} DRIVERS AN HOUR", 22, MUTED, a0, "SemiBold")
        rows = [(self.X, f[0], T[0]), (None, f[1], T[1]), (self.Y, f[2], T[2])]
        y = 326
        for j, (i, fl, tm) in enumerate(rows):
            if i is None:
                if opened <= 0:
                    continue
                if t < self.FIRST0:
                    main, sub = f"{m['night'][1]:.0f} min", "via the tunnel, at 3 a.m."
                    a = opened * smooth((t - self.NIGHT0) / 0.6)
                    if t < self.NIGHT0:
                        main, sub, a = "—", "via the tunnel", opened
                elif t < self.SW0:
                    main, sub, a = f"{m['lone']:.0f} min", "the first driver to try it", 1.0
                else:
                    main, sub, a = f"{tm:.0f} min", f"via the tunnel · {cars_h(fl)}", 1.0
                L.pill(d, x, y, 0, main, sub, a=a, size=30, anchor="lm", col=GOLDISH, letter="T")
            else:
                L.pill(d, x, y, i, f"{tm:.0f} min", f"{NAMES[i]} route · {cars_h(fl)}", a=ab0, size=30, anchor="lm")
            y += 66
        avg = sum(a_ * b_ for a_, b_ in zip(f, T)) / max(1.0, sum(f))
        ab = ab0 * (1 - smooth((t - self.CHART0) / 0.6))
        text(d, (x, 560), "AVERAGE TRIP", 22, MUTED, ab, "SemiBold")
        text(d, (x, 590), f"{avg:.0f} min", 80, GOLD, ab, "Bold")
        if t > self.SW1:
            xw = x + d.textlength(f"{avg:.0f} min", font=ST.font(80, "Bold")) + 28
            text(d, (xw, 640), f"+{m['opened']['avg'] - m['closed']['avg']:.0f} min for everyone", 34, INK, ab * smooth((t - self.SW1) / 0.8), "Bold", "lm")
        ac = smooth((t - self.CHART0) / 0.6)
        if ac > 0:
            cv = np.array(m["curve"])
            ax = L.Axes((150, 545, 860, 770), (0, 6000), (0, 80))
            hurt = cv[cv[:, 2] > cv[:, 1] + 0.05, 0]
            if len(hurt):
                d.rectangle([float(ax.X(hurt.min())), ax.y0, float(ax.X(hurt.max())), ax.y1], fill=(255, 186, 80, int(40 * ac)))
                text(d, (float(ax.X((hurt.min() + hurt.max()) / 2)), ax.y0 + 8), "the tunnel makes everyone slower", 20, GOLDISH, ac, "SemiBold", "ma")
            ax.draw(d, range(0, 6001, 2000), range(0, 81, 20), lambda v: f"{v:,}", str, "Drivers per hour", "Average trip (min)", ac, size=20)
            ax.series(d, cv[:, 0], cv[:, 1], col=INK2, width=3, dash=(8, 6), a=ac)
            ax.series(d, cv[:, 0], cv[:, 2], col=GOLDISH, width=4, dash=None, a=ac)
            for k, (lab, col, dsh) in enumerate((("without the tunnel", INK2, (8, 6)), ("with the tunnel", GOLDISH, None))):
                ly = ax.y1 - 56 + 26 * k
                L.dashed(d, [(ax.x1 - 250, ly), (ax.x1 - 210, ly)], col, 3, dsh, ac)
                text(d, (ax.x1 - 200, ly), lab, 19, INK2, ac, "Medium", "lm")
            xx = float(ax.X(m["cfg"]["demand"]))
            d.line([(xx, ax.y0), (xx, ax.y1)], fill=(240, 244, 250, int(200 * ac)), width=2)
        before, after = m["closed"]["avg"], m["opened"]["avg"]
        bn0, bn1 = self.bn_names
        bc = [round(v) for v in self.bottleneck_times(m["closed"]["flows"])]
        bo = [round(v) for v in self.bottleneck_times(m["opened"]["flows"])]
        caps(d, t, [(0.3, 5.6, [cap("Can a new road make everyone slower? Here's a thought experiment on the same map.", 34)]),
                    (6.0, 11.6, [cap("First, what's a bottleneck? A place where the road narrows:", 34),
                                 cap("say, 4 lanes squeeze into 2.", 34, INK2)]),
                    (12.0, 16.6, [cap("In light traffic, 2 lanes cope: cars get through at full speed.", 34)]),
                    (17.0, 25.6, [cap("In heavy traffic, more cars arrive than 2 lanes can pass.", 34),
                                  cap("A queue builds, and the wait keeps growing.", 34, INK2)]),
                    (26.0, 33.6, [cap("Our thought experiment gives each route one bottleneck like that:", 34),
                                  cap(f"the {bn0} and the {bn1}. Say the rest never jams.", 32, INK2)]),
                    (34.0, 40.6, [cap(f"With {fmt_int(m['cfg']['demand'])} drivers an hour, they split so both routes take {before:.0f} minutes.", 34)]),
                    (41.0, 47.6, [cap(f"Now dig a mythical tunnel from the end of the {bn0}", 34),
                                  cap(f"straight to the {bn1}: {m['cfg']['minutes']:.0f} minutes, never any traffic.", 34, GOLDISH)]),
                    (48.0, 53.6, [cap(f"At 3 a.m. it's a real shortcut: {m['night'][1]:.0f} minutes instead of {m['night'][2]:.0f}.", 34)]),
                    (54.0, 59.6, [cap(f"At rush hour, the first driver to try it arrives in {m['lone']:.0f} minutes, not {before:.0f}.", 34),
                                  cap("So others follow...", 34, INK2)]),
                    (60.0, 69.6, [cap("...and every one of them goes through both bottlenecks, so both queues grow.", 32),
                                  cap(f"{bn0}: {bc[0]} → {bo[0]} min. {bn1}: {bc[1]} → {bo[1]} min.", 32, INK2)]),
                    (71.0, 77.6, [cap(f"When the switching stops, every route takes {after:.0f} minutes:", 34),
                                  cap(f"{after - before:.0f} minutes longer than before the tunnel, for everyone.", 34, GOLD)]),
                    (78.0, 84.6, [cap("And nobody can do better by going back: both old routes now share", 34),
                                  cap("a jammed bottleneck with the tunnel route.", 34, INK2)]),
                    (85.0, 91.6, [cap("Each driver who switched saved time, but slowed everyone in two bottlenecks.", 32),
                                  cap(f"Close the tunnel, and everyone arrives {after - before:.0f} minutes sooner.", 32, INK2)]),
                    (92.0, 99.8, [cap("This is Braess's paradox (Dietrich Braess, 1968). It isn't the rule:", 34),
                                  cap("with fewer drivers the same tunnel helps. It depends on the bottlenecks.", 32, INK2)])])
        note(d, "Thought experiment: the tunnel is imaginary, and the bottlenecks, their capacities and the driver numbers are made up.", a0)
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
        text(d, (x, 820), "And beware the shortcut that everyone takes.", 32, INK, a4, "SemiBold")
        a5 = ST.fade(t, 19.0, 1e9)
        for k, s in enumerate(["Roads © OpenStreetMap contributors, via Overture Maps. Free-flow times: real roads at the speed limit.",
                               f"Traffic: a toy model with made-up capacities, not {N['city_name']} traffic data. Music: original, synthesized for this video."]):
            text(d, (x, 990 + 30 * k), s, 20, MUTED, a5, "Regular")
        return np.asarray(im)


SECTIONS = {c.name: c for c in (Hook, Light, Empty, Curve, Fill, Why, Fine, Learn, App, Optimum, Myth, Outro)}
ORDER = ["hook", "light", "empty", "curve", "fill", "why", "fine", "learn", "app", "optimum", "myth", "outro"]

if __name__ == "__main__":
    s = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for v in sys.argv[2:]:
            s.render(still=float(v))
    else:
        s.render()
