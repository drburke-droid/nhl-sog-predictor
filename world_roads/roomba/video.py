"""The Roomba photon video (1920x1080, 30 fps), caption-only.

    python video.py NAME [still times...]     NAME: cal_intro, rule, cal_run, compare, usa_intro, usa_run, last, outro
"""
import os, sys, json, math, time, subprocess, functools
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                             # fonts, captions, translucent-text fix
import maps
import cover as CV

DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
INK, INK2, MUTED, GOLD, CYAN, RED = ST.INK, ST.INK2, ST.MUTED, ST.GOLD, (90, 200, 255), ST.RED
CAL_SEED = 31                                                   # the walk nearest the median of 200
CAL_PLANNED_S = 0.03714518787706265                            # the Calgary film's planned route (calgary_stats.json)
CAL_X = 1180                                                    # Calgary map centre on screen (map left of the HUD)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fmt_light(s):
    if s < 1e-3:
        return f"{s * 1e6:,.0f} µs"
    if s < 1:
        return f"{s * 1e3:,.1f} ms"
    if s < 60:
        return f"{s:,.2f} s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)} h {int(m):02d} min {int(sec):02d} s" if h else f"{int(m)} min {int(sec):02d} s"


def fmt_hm(s):
    h, m = divmod(int(round(s / 60)), 60)
    return f"{h} h {m:02d} min"


def text(d, xy, s, size, col, a=1.0, weight="SemiBold", anchor="la"):
    if a > 0:
        d.text(xy, s, font=ST.font(size, weight), fill=tuple(col) + (int(255 * a),), anchor=anchor)


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
        path = os.path.join(OUT, f"{self.name}.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                              "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            if i % 150 == 0:
                print(f"{self.name}: frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close(); p.wait()
        print("wrote", path, flush=True)


def calgary_stats():
    """medians (and walk counts) of the Calgary cover times, from the 200-walk run when it is there"""
    js = json.load(open(os.path.join(DATA, "calgary_cover.json")))
    out = dict(median_s_nonback=js["median_s_nonback"], median_s_plain=js["median_s_plain"], n_nonback=8, n_plain=8)
    mp = os.path.join(DATA, "calgary_cover_many.json")
    if os.path.exists(mp):
        many = json.load(open(mp))
        for k in ("nonback", "plain"):
            if "seconds_" + k in many:
                out["median_s_" + k] = float(np.median(many["seconds_" + k]))
                out["n_" + k] = len(many["seconds_" + k])
    return out


# ================================================================== Calgary
class CalRun(Section):
    """Calgary from the first metre to the last street. Phase 'intro' is the slow start (captions set the
    scene); phase 'run' is the rest, accelerating, with the HUD and milestones."""

    def __init__(self, phase):
        self.phase = phase
        self.C = maps.Calgary(seed=CAL_SEED, cy=-1.5, ox=CAL_X)
        self.T_km = self.C.total_km
        self.net_km = float(self.C.elen.sum())
        log = np.load(os.path.join(DATA, f"calgary_log_{CAL_SEED}.npy"))
        self.log = log
        self.D0 = 90.0                                        # km driven during the intro (0.3 ms of light)
        if phase == "intro":
            self.name, self.dur = "cal_intro", 15.0
        else:
            self.name, self.dur = "cal_run", 56.0
            self.C.map.advance(self.D0)
            self.C.k = self.C.step_at(self.D0)
        self.lastD = -1
        # milestones (fraction of street length driven)
        f = log[:, 3] / self.net_km
        self.ms = []
        for q, lab in ((0.5, "Half of Calgary's streets"), (0.9, "90% of streets"), (0.99, "99% of streets"), (0.999, "99.9% of streets")):
            k = int(np.searchsorted(f, q))
            self.ms.append((lab, float(log[min(k, len(log) - 1), 1])))

    def settle(self):
        self.C.trail[:] = 0

    def D(self, t):
        if self.phase == "intro":
            u = t / self.dur
            return self.D0 * (math.exp(5.0 * u) - 1) / (math.exp(5.0) - 1)
        u = min(t / (self.dur - 6.0), 1.0)
        return self.D0 + (self.T_km - self.D0) * u ** 3.4

    def frame(self, t):
        D = self.D(t)
        if D < self.lastD:                                     # stills out of order: rebuild state
            self.__init__(self.phase)
        self.lastD = D
        fr = self.C.frame(D)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        sec = D / CV.C_KMS
        driven = self.C.map.done.sum() / self.C.map.total.sum()
        a_h = ST.fade(t, 0.8 if self.phase == "intro" else 0.0, self.dur + 9, 0.6, 0.6)
        x0 = 60
        text(d, (x0, 60), "CALGARY · EVERY STREET", 26, MUTED, a_h, "SemiBold")
        text(d, (x0, 100), "LIGHT TIME", 22, MUTED, a_h)
        text(d, (x0, 128), fmt_light(sec), 84, GOLD, a_h, "Bold")
        text(d, (x0, 250), "STREETS DRIVEN", 22, MUTED, a_h)
        text(d, (x0, 278), f"{driven * 100:.2f}%" if driven < 0.9999 else f"{driven * 100:.3f}%", 64, INK, a_h, "Bold")
        text(d, (x0, 380), "DISTANCE DRIVEN", 22, MUTED, a_h)
        text(d, (x0, 408), f"{D:,.0f} km", 48, INK, a_h, "Bold")
        if D > self.net_km:
            text(d, (x0, 470), f"{D / self.net_km:,.1f}× the city's {self.net_km:,.0f} km of streets", 24, INK2, a_h, "Regular")
        if self.phase == "run":
            self.chart(d, t, D, a_h)
            # the last street: a ring when it is found
            if t > self.dur - 6.0:
                e = int(np.argmax(self.C.first))
                hx, hy = self.C.mx[e], self.C.my[e]
                g = smooth((t - self.dur + 6.0) / 0.5) * ST.fade(t, 0, self.dur, 0, 0.5)
                r = 22 + 60 * (1 - smooth((t - self.dur + 6.0) / 0.5))
                d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=GOLD + (int(255 * g),), width=4)
                lab = f"the last street ({self.C.elen[e] * 1000:.0f} m)"
                tw = d.textlength(lab, font=ST.font(28))
                d.rounded_rectangle([hx + 30, hy - 24, hx + 54 + tw, hy + 24], 12, fill=ST.PANEL + (int(215 * g),))
                text(d, (hx + 42, hy), lab, 28, GOLD, g, "SemiBold", "lm")
        fr = np.asarray(im).copy()
        return ST.draw_captions(fr, t, self.caps(t))

    def chart(self, d, t, D, a):
        """streets driven vs light time, filling in as the walk goes"""
        x0, y0, w, h = 60, 600, 520, 300
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 70, 90, int(200 * a)), width=2)
        T = self.log[-1, 1]
        k = int(np.searchsorted(self.log[:, 1], D))
        xs = x0 + self.log[:k + 1, 1] / T * w
        ys = y0 + h - self.log[:k + 1, 3] / self.net_km * h
        if k > 1:
            d.line(list(zip(xs, ys)), fill=GOLD + (int(255 * a),), width=3)
        text(d, (x0, y0 + h + 12), "0", 20, MUTED, a)
        text(d, (x0 + w, y0 + h + 12), f"{T / CV.C_KMS:.2f} s", 20, MUTED, a, anchor="ra")
        text(d, (x0 + 8, y0 + 6), "100%", 20, MUTED, a)
        text(d, (x0, y0 - 34), "streets driven, over time", 22, INK2, a)

    def caps(self, t):
        if self.phase == "intro":
            return [(0.6, 5.2, [("A photon leaves downtown Calgary at the speed of light.", 40, INK, "SemiBold")], {"anchor": "top", "cx": CAL_X}),
                    (5.6, 10.0, [("At every intersection, it picks a road at random.", 40, INK, "SemiBold")], {"anchor": "top", "cx": CAL_X}),
                    (10.4, 14.8, [("How long until it has driven every street?", 44, GOLD, "Bold")], {"anchor": "top", "cx": CAL_X})]
        caps = []
        Ts = [(lab, self.t_of(D)) for lab, D in self.ms]
        for (lab, tt), nxt in zip(Ts, Ts[1:] + [("", self.dur - 6.0)]):
            secs = self.D_to_s(tt)
            caps.append((tt, min(tt + 3.4, nxt[1] - 0.2), [(f"{lab}: {fmt_light(secs)}", 40, GOLD, "Bold")], {"anchor": "top", "cx": CAL_X}))
        tail = 1 - self.ms[2][1] / self.T_km                      # share of the trip after 99% of the streets
        caps.append((self.dur - 5.6, self.dur, [(f"Every street: {fmt_light(self.T_km / CV.C_KMS)}", 46, GOLD, "Bold"),
                                                (f"{tail * 100:.0f}% of the trip went to finding the last 1% of streets", 30, INK2, "Regular")],
                     {"anchor": "bottom", "cx": CAL_X}))
        return caps

    def t_of(self, Dq):
        """video time at which D(t) reaches Dq"""
        u = ((Dq - self.D0) / (self.T_km - self.D0)) ** (1 / 3.4)
        return u * (self.dur - 6.0)

    def D_to_s(self, tt):
        return self.D(tt) / CV.C_KMS


# ================================================================== the rule
class Rule(Section):
    name, dur = "rule", 13.0

    def frame(self, t):
        fr = np.zeros((H, W, 3), np.uint8); fr[:] = (5, 8, 15)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        cx, cy, L = 960, 470, 300
        a = ST.fade(t, 0.2, self.dur, 0.5, 0.5)
        roads = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in roads:
            d.line([(cx, cy), (cx + dx * L, cy + dy * L)], fill=(70, 90, 120, int(255 * a)), width=14)
        # arriving from the left
        u = smooth((t - 0.6) / 2.0)
        px = cx - L + L * u
        d.ellipse([px - 14, cy - 14, px + 14, cy + 14], fill=(255, 245, 220, int(255 * a)))
        if t > 2.8:
            g = smooth((t - 2.8) / 0.6) * a
            for dx, dy, lx, ly in ((1, 0, L * 0.5, -48), (0, -1, 48, -L * 0.5), (0, 1, 48, L * 0.5)):
                ex, ey = cx + dx * L * 0.62, cy + dy * L * 0.62
                d.line([(cx + dx * 40, cy + dy * 40), (ex, ey)], fill=GOLD + (int(230 * g),), width=6)
                text(d, (cx + lx, cy + ly), "⅓", 44, GOLD, g, "Bold", "mm")
            d.line([(cx - 40, cy), (cx - L * 0.62, cy)], fill=RED + (int(200 * g),), width=6)
            qx, qy, q = cx - L * 0.36, cy - 52, 16
            d.line([(qx - q, qy - q), (qx + q, qy + q)], fill=RED + (int(255 * g),), width=7)
            d.line([(qx - q, qy + q), (qx + q, qy - q)], fill=RED + (int(255 * g),), width=7)
            text(d, (cx - L * 0.5, cy + 40), "the way it came in", 28, RED, g, "SemiBold", "ma")
        fr = np.asarray(im).copy()
        caps = [(0.4, 6.2, [("The rule: at every intersection, take any road", 40, INK, "SemiBold"),
                            ("except the one you came in on. Every choice equally likely.", 36, INK2, "Regular")]),
                (6.6, 12.6, [("At a dead end, turn around. Like a Roomba.", 40, INK, "SemiBold"),
                             ("No map, no memory, no plan.", 36, GOLD, "SemiBold")])]
        return ST.draw_captions(fr, t, caps)


# ================================================================== compare
class Compare(Section):
    name, dur = "compare", 14.0

    def frame(self, t):
        js = calgary_stats()
        roomba = js["median_s_nonback"]
        plain = js["median_s_plain"]
        rows = [("Planned route that drives every street", CAL_PLANNED_S, CYAN, "from my Calgary video"),
                (f"Roomba photon (typical of {js['n_nonback']} runs)", roomba, GOLD, "never turns back except at dead ends"),
                (f"Plain random walk (can turn back anywhere; typical of {js['n_plain']} runs)", plain, RED, "")]
        fr = np.zeros((H, W, 3), np.uint8); fr[:] = (5, 8, 15)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a0 = ST.fade(t, 0.2, self.dur, 0.5, 0.5)
        text(d, (960, 120), "Every street in Calgary, at the speed of light", 50, INK, a0, "Bold", "ma")
        x0, xmax = 260, 1660
        lo, hi = math.log10(0.01), math.log10(100)
        for k, (lab, s, col, sub) in enumerate(rows):
            a = ST.fade(t, 1.0 + k * 1.6, self.dur, 0.5, 0.5)
            y = 300 + k * 190
            g = smooth((t - 1.0 - k * 1.6) / 1.0)
            wbar = (math.log10(s) - lo) / (hi - lo) * (xmax - x0) * g
            d.rounded_rectangle([x0, y, x0 + max(wbar, 4), y + 70], 10, fill=col + (int(220 * a),))
            text(d, (x0, y - 44), lab, 32, INK, a, "SemiBold")
            text(d, (x0 + wbar + 20, y + 35), fmt_light(s), 44, col, a, "Bold", "lm")
            if sub:
                text(d, (x0, y + 84), sub, 24, INK2, a, "Regular")
        text(d, (960, 940), "(bar lengths on a log scale)", 22, MUTED, a0, "Regular", "ma")
        fr = np.asarray(im).copy()
        caps = [(6.4, self.dur, [(f"Random is about {roomba / CAL_PLANNED_S:,.0f}× slower than a plan.", 40, GOLD, "Bold"),
                                 (f"Allow U-turns and it's {plain / roomba:,.0f}× slower again.", 34, INK2, "Regular")])]
        return ST.draw_captions(fr, t, caps)


# ================================================================== USA
USA_SEED = int(os.environ.get("USA_SEED", "0"))
USA_SCALE, USA_OX, USA_OY = 0.335, 996, 621
# the last road and where it is (Overture names at both of the road's ends; see where.py)
LAST_PLACE = {0: ("Beach Road", "Manasota Key, Englewood Beach, Florida"), 1: ("Sheldrake Court", "near Beaufort, North Carolina"),
              2: ("Cowshed Road", "Queenstown, Virginia"), 3: ("Nellie Lane", "near Port Angeles, Washington")}

LAST_NOTE = {0: "all on one barrier island"}


def usa_data():
    js = json.load(open(os.path.join(DATA, "usa_cover.json")))
    w = [r for r in js["walks"] if r["seed"] == USA_SEED][0]
    allw = js["walks"][:]
    mp = os.path.join(DATA, "usa_cover_more.json")
    if os.path.exists(mp):
        allw += json.load(open(mp))["walks"]
    secs = sorted(r["seconds"] for r in allw if r["done"])
    return js["meta"], w, np.load(os.path.join(DATA, f"usa_log_{USA_SEED}.npy")), secs


class USARun(Section):
    """The whole lower 48, from Kansas to the last road (stops one road short; 'last' takes it from there)."""

    def __init__(self, phase):
        self.phase = phase
        self.meta, self.w, self.log, self.secs = usa_data()
        self.T = self.w["km"]
        self.m = self.meta["roads"]
        self.U = None
        if phase == "intro":
            self.name, self.dur = "usa_intro", 10.0
        else:
            self.name, self.dur = "usa_run", 50.0
        f = self.log[:, 3] / self.meta["km"]
        self.ms = []
        for q, lab in ((0.5, "Half of America's road length"), (0.9, "90% of road length"), (0.99, "99%"), (0.999, "99.9%"), (0.9999, "99.99%")):
            k = int(np.searchsorted(f, q))
            self.ms.append((lab, float(self.log[min(k, len(self.log) - 1), 1])))
        self.lastD = -1

    def load(self):
        if self.U is None:
            self.U = maps.USA(seed=USA_SEED, scale=USA_SCALE, ox=USA_OX, oy=USA_OY)
            self.Dend = usa_tail(self.T)["Ds"]                        # where 'last' takes over
            self.sx, self.sy = self.U.to_px(*self.meta["start_lonlat"])

    P = 3.3
    RUN = 46.0

    def settle(self):
        if self.U is not None and hasattr(self.U, "flash"):
            self.U.flash[:] = 0

    def D(self, t):
        if self.phase == "intro":
            return 0.0
        u = min(t / self.RUN, 1.0)
        return self.Dend * u ** self.P

    def t_of(self, Dq):
        return (Dq / self.Dend) ** (1 / self.P) * self.RUN

    def frame(self, t):
        self.load()
        D = self.D(t)
        if D < self.lastD:
            self.U.map.reset(); self.U.flash[:] = 0
        self.lastD = D
        fr = self.U.frame(D)
        if self.phase == "intro":                             # the dim map fades in
            a = smooth(t / 2.0)
            fr = (maps.BG + (fr.astype(np.float32) - maps.BG) * a).astype(np.uint8)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        left = self.m - self.U.map.i
        a_h = ST.fade(t, 3.0 if self.phase == "intro" else 0.0, 1e9, 0.6, 0.6)
        self.hud(d, D, left, a_h)
        if self.phase == "intro":
            g = ST.fade(t, 4.5, 1e9, 0.5, 0.5)
            r = 18 + 30 * (1 - smooth((t - 4.5) / 0.6))
            d.ellipse([self.sx - r, self.sy - r, self.sx + r, self.sy + r], outline=GOLD + (int(255 * g),), width=4)
            lab = "start: the centre of the lower 48, near Lebanon, Kansas"
            tw = d.textlength(lab, font=ST.font(26))
            d.rounded_rectangle([self.sx + 34, self.sy - 22, self.sx + 58 + tw, self.sy + 22], 12, fill=ST.PANEL + (int(215 * g),))
            text(d, (self.sx + 46, self.sy), lab, 26, GOLD, g, "SemiBold", "lm")
        else:
            self.chart(d, D, a_h)
        fr = np.asarray(im).copy()
        return ST.draw_captions(fr, t, self.caps())

    def hud(self, d, D, left, a):
        x0, y0 = 60, 772
        text(d, (x0, y0), "LIGHT TIME", 22, MUTED, a)
        text(d, (x0, y0 + 26), fmt_light(D / CV.C_KMS) if D > 0 else "0 s", 64, GOLD, a, "Bold")
        text(d, (x0, y0 + 112), "ROADS NOT YET DRIVEN", 22, MUTED, a)
        text(d, (x0, y0 + 138), f"{left:,}", 52, INK if left > 1 else GOLD, a, "Bold")
        text(d, (x0, y0 + 210), f"{D:,.0f} km driven" if D < 1e9 else f"{D / 1e9:,.2f} billion km driven", 28, INK2, a, "Regular")

    def chart(self, d, D, a):
        """roads not yet driven (log scale) vs light time"""
        x0, y0, w, h = 1570, 740, 250, 220
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 70, 90, int(200 * a)), width=2)
        k = int(np.searchsorted(self.log[:, 1], D))
        L = np.maximum(self.m - self.log[:k + 1, 2], 1)
        xs = x0 + self.log[:k + 1, 1] / self.T * w
        ys = y0 + h - np.log10(L) / 8 * h
        if k > 1:
            d.line(list(zip(xs, ys)), fill=GOLD + (int(255 * a),), width=3)
        text(d, (x0, y0 - 32), "roads not yet driven", 22, INK2, a)
        for e in (0, 4, 8):
            text(d, (x0 + w + 8, y0 + h - e / 8 * h), ("1", "10,000", "100 M")[e // 4], 18, MUTED, a, "Regular", "lm")
        text(d, (x0, y0 + h + 10), "0", 18, MUTED, a, "Regular")
        text(d, (x0 + w, y0 + h + 10), fmt_light(self.T / CV.C_KMS).split(" min")[0] + " min", 18, MUTED, a, "Regular", "ra")

    def caps(self):
        top = {"anchor": "top", "y": 18}
        if self.phase == "intro":
            return [(0.4, 4.6, [("Now: every road in the contiguous United States.", 40, INK, "SemiBold"),
                                (f"{self.meta['km'] / 1e6:.1f} million km of road, {self.m / 1e6:.1f} million segments", 32, INK2, "Regular")], top),
                    (5.0, 10.0, [("Same photon. Same rule.", 42, GOLD, "Bold")], top)]
        caps = []
        Ts = [(lab, self.t_of(D), D) for lab, D in self.ms]
        for (lab, tt, D), nxt in zip(Ts, Ts[1:] + [("", self.RUN + 1.0, 0)]):
            caps.append((tt, min(tt + 3.2, nxt[1] - 0.2), [(f"{lab}: {fmt_light(D / CV.C_KMS)}", 40, GOLD, "Bold")], top))
        left = len(usa_tail(self.T)["late"])
        caps.append((self.RUN + 0.3, self.dur, [(f"{left:,} roads left." if left > 1 else "One road left.", 46, GOLD, "Bold")], top))
        return caps


@functools.lru_cache(maxsize=4)
def usa_tail(T):
    """The walk's ending, step by step (exact): the final burst of new roads and the slow-motion window.
    Returns dict(Ds: slow-motion start km, Da: first road of the final burst, late: roads first driven after Ds
    with their km, steps: (road, km after) for every step after Ds)."""
    tl = np.load(os.path.join(DATA, f"usa_tail_{USA_SEED}.npz"))
    road, km, new = tl["road"], tl["km"], tl["new"]
    assert abs(float(tl["total"]) - T) < 1e-3 * max(1.0, T / 1e9)
    nk = km[new]
    gaps = np.diff(nk)
    big = np.nonzero(gaps > TAIL_GAP_KM)[0]                    # the final burst starts after the last long gap
    b0 = int(big[-1]) + 1 if len(big) else 0
    Da = float(nk[b0])
    k_a = int(np.searchsorted(km, Da))                       # the step that first enters the burst
    k_s = max(0, k_a - TAIL_PRE_STEPS)
    Ds = float(km[k_s - 1]) if k_s > 0 else float(km[0]) - 1e-6
    sel = new & (km > Ds)
    return dict(Ds=Ds, Da=Da, n_burst=int(len(nk) - b0), late=road[sel], late_km=km[sel], steps=(road[k_s:], km[k_s:]))


TAIL_GAP_KM = 200.0
TAIL_PRE_STEPS = 60


class Last(Section):
    """Zoom from the whole country to the last dark roads, then watch the photon finish them in slow motion."""
    name, dur = "last", 20.0
    Z0, Z1 = 1.8, 8.8                        # zoom window
    S0, S1 = 9.8, 15.0                       # slow motion: the photon drives from Ds to the end
    ARRIVE = 15.0

    def __init__(self, scale0=USA_SCALE, ox=USA_OX, oy=USA_OY, cx=W / 2 + 120, fit=(1500, 880)):
        self.fit = fit
        self.meta, self.w, self.log, self.secs = usa_data()
        self.T = self.w["km"]
        self.tail = usa_tail(self.T)
        self.scale0, self.ox, self.oy, self.cx = scale0, ox, oy, cx
        self.Z = None

    def load(self):
        if self.Z is not None:
            return
        tl = self.tail
        self.Z = maps.Zoom(tl["late"])
        Z = self.Z
        self.late = tl["late"]
        self.late_km = tl["late_km"]
        self.LE = Z.ends(self.late)
        # the slow-motion path: node after every step (the walk's start node of the window is whichever end fits)
        r, k = tl["steps"]
        ea, eb = Z.ea[r].astype(np.int64), Z.eb[r].astype(np.int64)
        for n0 in (ea[0], eb[0]):
            nodes = [n0]
            ok = True
            for i in range(len(r)):
                c = nodes[-1]
                if ea[i] == c:
                    nodes.append(eb[i])
                elif eb[i] == c:
                    nodes.append(ea[i])
                else:
                    ok = False
                    break
            if ok:
                break
        assert ok, "path does not chain"
        nodes = np.array(nodes)
        self.px, self.py = Z.x[nodes].astype(np.float64), Z.y[nodes].astype(np.float64)
        self.pk = np.r_[k[0] - Z.road_km[r[0]], k]                   # km at each node
        # final view: the late roads and the window's path, with a margin
        xs = np.r_[self.LE[0], self.LE[2], self.px]; ys = np.r_[self.LE[1], self.LE[3], self.py]
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        Z.tx, Z.ty = (bx0 + bx1) / 2, (by0 + by1) / 2                # zoom about the box centre
        self.s1 = min(self.fit[0] / max(bx1 - bx0, 0.3), self.fit[1] / max(by1 - by0, 0.3), 4000.0)
        self.p0 = (self.ox + Z.tx * self.scale0, self.oy - Z.ty * self.scale0)
        self.last = int(self.late[-1])

    def view(self, t):
        v = smooth((t - self.Z0) / (self.Z1 - self.Z0))
        s = self.scale0 * (self.s1 / self.scale0) ** v
        c = smooth((t - self.Z0) / ((self.Z1 - self.Z0) * 0.6))
        px = self.p0[0] + (self.cx - self.p0[0]) * c
        py = self.p0[1] + (H / 2 + 20 - self.p0[1]) * c
        return s, px, py

    def Dt(self, t):
        """km driven at video time t: held at Ds, then the slow-motion window, linear in distance"""
        u = min(max((t - self.S0) / (self.S1 - self.S0), 0.0), 1.0)
        return self.tail["Ds"] + (self.T - self.tail["Ds"]) * u

    def map_frame(self, t):
        """the map (no text): returns (frame, world->screen, km driven, roads still dark, head xy)"""
        self.load()
        s, px, py = self.view(t)
        Z = self.Z
        dens, f = Z.density(s, px, py)
        img = maps.BG + dens[..., None] * maps.GOLD
        D = self.Dt(t)
        th = max(2, int(round(1 + 2 * min(3.0, math.log2(max(s, 40) / 40)))))
        x0, y0 = f(self.LE[0], self.LE[1]); x1, y1 = f(self.LE[2], self.LE[3])
        lit = self.late_km <= D + 1e-9
        glow = np.zeros((H, W), np.float32)
        for i in range(len(self.late)):
            p, q = (int(x0[i]), int(y0[i])), (int(x1[i]), int(y1[i]))
            col = maps.GOLD if lit[i] else maps.DIM * 2.2
            cv2.line(img, p, q, tuple(float(c) for c in col), th, cv2.LINE_AA)
            if lit[i]:                                                    # fresh roads glow for ~0.4 s of video
                age = (D - self.late_km[i]) / (self.T - self.tail["Ds"]) * (self.S1 - self.S0)
                if i == len(self.late) - 1:
                    age = max(t - self.ARRIVE, 0) * 0.35
                cv2.line(glow, p, q, float(math.exp(-age / 0.4)), th + 2, cv2.LINE_AA)
        # the photon: recent path and head
        head = None
        if self.S0 - 0.3 <= t <= self.S1 + 0.05:
            j = int(np.searchsorted(self.pk, D))
            j = min(max(j, 1), len(self.pk) - 1)
            a = (D - self.pk[j - 1]) / max(self.pk[j] - self.pk[j - 1], 1e-12)
            hx = self.px[j - 1] + (self.px[j] - self.px[j - 1]) * a
            hy = self.py[j - 1] + (self.py[j] - self.py[j - 1]) * a
            sx, sy = f(np.r_[self.px[max(0, j - 40):j], hx], np.r_[self.py[max(0, j - 40):j], hy])
            pts = np.c_[sx, sy].astype(np.int32)
            tr = np.zeros((H, W), np.float32)
            n = len(pts)
            for i in range(n - 1):
                cv2.line(tr, tuple(pts[i]), tuple(pts[i + 1]), float(0.25 + 0.75 * (i + 1) / n), th, cv2.LINE_AA)
            glow += tr * 0.7
            head = (float(sx[-1]), float(sy[-1]))
            hd = np.zeros((H, W), np.float32)
            cv2.circle(hd, (int(head[0]), int(head[1])), max(3, th), 1.0, -1, cv2.LINE_AA)
            glow += cv2.GaussianBlur(hd, (0, 0), 3) * 4 + cv2.GaussianBlur(hd, (0, 0), 12) * 6
        if t > self.ARRIVE:                                              # the last road: a burst
            fl = math.exp(-(t - self.ARRIVE) / 0.9)
            g = np.zeros((H, W), np.float32)
            cv2.line(g, (int(x0[-1]), int(y0[-1])), (int(x1[-1]), int(y1[-1])), 1.0, th + 2, cv2.LINE_AA)
            glow += (cv2.GaussianBlur(g, (0, 0), 8) * 4 + cv2.GaussianBlur(g, (0, 0), 40) * 30) * fl
        gl = np.minimum(glow, 3.0)[..., None] * maps.HOT * 0.6
        sa = 1 - smooth((t - self.Z0) / 2.5)                            # the pocket's blue spark, until the zoom resolves it
        if sa > 0:                                                         # same pixels as the run's sparks
            mx, my = np.round((x0 + x1) / 2).astype(int), np.round((y0 + y1) / 2).astype(int)
            ok = (~lit) & (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
            m = np.zeros((H, W), np.float32)
            m[my[ok], mx[ok]] = 1.0
            gl = gl + maps.spark(m) * sa
        fr = maps.tonemap(img, gl)
        dark = int((~lit).sum())
        lc = ((x0[-1] + x1[-1]) / 2, (y0[-1] + y1[-1]) / 2)
        return fr, f, D, dark, head, s, lc

    def frame(self, t):
        fr, f, D, dark, head, s, lc = self.map_frame(t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        # ring: the dark pocket while zooming; the last road after
        ga = ST.fade(t, 0.4, self.Z1 + 0.6, 0.4, 0.5) * (0.65 + 0.35 * math.sin(t * 5))
        bx = f(self.Z.tx, self.Z.ty)
        rr = 46
        d.ellipse([bx[0] - rr, bx[1] - rr, bx[0] + rr, bx[1] + rr], outline=GOLD + (int(255 * ga),), width=4)
        road, place = LAST_PLACE[USA_SEED]
        ap = ST.fade(t, self.Z1 - 1.2, 1e9, 0.5, 0.5)
        tw = d.textlength(place, font=ST.font(30))
        d.rounded_rectangle([60, 40, 60 + tw + 28, 92], 12, fill=ST.PANEL + (int(215 * ap),))
        text(d, (74, 66), place, 30, GOLD, ap, "SemiBold", "lm")
        if t > self.ARRIVE:
            gl = ST.fade(t, self.ARRIVE, 1e9, 0.3, 0.3)
            lab = f"the last road: {road}"
            tw = d.textlength(lab, font=ST.font(30))
            d.rounded_rectangle([lc[0] + 30, lc[1] - 26, lc[0] + 58 + tw, lc[1] + 26], 12, fill=ST.PANEL + (int(215 * gl),))
            text(d, (lc[0] + 44, lc[1]), lab, 30, GOLD, gl, "SemiBold", "lm")
        self.scalebar(d, s, ST.fade(t, 1.0, 1e9, 0.5, 0.5))
        sl = (self.T - self.tail["Ds"]) / CV.C_KMS / (self.S1 - self.S0)          # light-seconds per video second
        if self.S0 - 0.5 < t < self.S1 + 0.5:
            a = ST.fade(t, self.S0 - 0.5, self.S1 + 0.5, 0.4, 0.4)
            lab = f"slow motion: {1 / sl:,.0f}× slower than light"
            tw = d.textlength(lab, font=ST.font(28))
            d.rounded_rectangle([W - 80 - tw - 28, 40, W - 80, 92], 12, fill=ST.PANEL + (int(215 * a),))
            text(d, (W - 94, 66), lab, 28, INK, a, "SemiBold", "rm")
        x0, y0 = 60, 772                                                    # where the run's HUD was
        text(d, (x0, y0), "LIGHT TIME", 22, MUTED, 1.0)
        sec = D / CV.C_KMS
        h_, rem = divmod(sec, 3600); m_, s_ = divmod(rem, 60)
        text(d, (x0, y0 + 26), f"{int(h_)} h {int(m_):02d} min {s_:08.5f} s", 56, GOLD, 1.0, "Bold")
        text(d, (x0, y0 + 112), "ROADS NOT YET DRIVEN", 22, MUTED, 1.0)
        text(d, (x0, y0 + 138), f"{dark:,}", 52, GOLD, 1.0, "Bold")
        fr = np.asarray(im).copy()
        return ST.draw_captions(fr, t, self.caps())

    def scalebar(self, d, s, a):
        for L in (1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05):
            if L * s <= 240:
                break
        x1, y = W - 80, H - 70
        x0 = x1 - L * s
        d.line([(x0, y), (x1, y)], fill=INK + (int(230 * a),), width=4)
        for x in (x0, x1):
            d.line([(x, y - 10), (x, y + 10)], fill=INK + (int(230 * a),), width=4)
        text(d, ((x0 + x1) / 2, y - 18), f"{L:g} km" if L >= 1 else f"{L * 1000:.0f} m", 24, INK, a, "SemiBold", "md")

    def caps(self):
        bot = {"anchor": "bottom"}
        T = self.T / CV.C_KMS
        road, place = LAST_PLACE[USA_SEED]
        n = len(self.tail["late"])
        last_m = float(self.Z.road_km[self.last]) * 1000 if self.Z is not None else 0
        first = (f"{n:,} roads still dark, {LAST_NOTE.get(USA_SEED, 'all in one corner of the map')}." if n > 1
                 else "One road still dark.")
        top = {"anchor": "top", "y": 18}
        return [(0.3, self.Z0 + 2.4, [(first, 42, INK, "SemiBold")], top),
                (self.Z0 + 2.8, self.S0 - 0.3, [("Islands and peninsulas have few ways in.", 40, INK, "SemiBold"),
                                               ("A random walker can miss them for hours.", 34, INK2, "Regular")],
                 {"anchor": "top", "y": 110}),
                (self.ARRIVE, self.dur, [(f"The last road: {fmt_light(T)}", 46, GOLD, "Bold"),
                 (f"{last_m:.0f} m of {road}, after {self.T / 1e9:.2f} billion km of driving.", 30, INK2, "Regular")],
                 {"anchor": "top", "y": 110})]


# ================================================================== outro
def usa_gold():
    """the whole map, every road driven (cached): the backdrop for the runs map"""
    p = os.path.join(OUT, f"usa_gold_{USA_SEED}.png")
    if not os.path.exists(p):
        U = maps.USA(seed=USA_SEED, scale=USA_SCALE, ox=USA_OX, oy=USA_OY)
        U.map.advance(np.inf)
        Image.fromarray(maps.tonemap(U.map.image(gamma=0.42, dim=1.5, fgamma=4.0))).save(p)
    return np.asarray(Image.open(p).convert("RGB"))


class Runs(Section):
    """Where the last road was, in every run."""
    name, dur = "runs", 11.0

    def __init__(self):
        self.meta, self.w, self.log, self.secs = usa_data()
        js = json.load(open(os.path.join(DATA, "usa_cover.json")))["walks"]
        mp = os.path.join(DATA, "usa_cover_more.json")
        if os.path.exists(mp):
            js += json.load(open(mp))["walks"]
        self.walks = sorted(js, key=lambda w: w["seed"])
        lon, lat = np.load(os.path.join(DATA, "usa_lon.npy"), mmap_mode="r"), np.load(os.path.join(DATA, "usa_lat.npy"), mmap_mode="r")
        ea, eb = np.load(os.path.join(DATA, "usa_ea.npy"), mmap_mode="r"), np.load(os.path.join(DATA, "usa_eb.npy"), mmap_mode="r")
        e = np.array([w["last"] for w in self.walks])
        lo = (lon[ea[e]].astype(np.float64) + lon[eb[e]]) / 2
        la = (lat[ea[e]].astype(np.float64) + lat[eb[e]]) / 2
        x, y = maps.albers(lo, la)
        self.px, self.py = USA_OX + x * USA_SCALE, USA_OY - y * USA_SCALE
        self.bg = None

    def frame(self, t):
        if self.bg is None:
            self.bg = (maps.BG + (usa_gold().astype(np.float32) - maps.BG) * 0.42).astype(np.uint8)
        a0 = smooth(t / 0.6)
        fr = (maps.BG + (self.bg.astype(np.float32) - maps.BG) * a0).astype(np.uint8)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        for i, w in enumerate(self.walks):
            g = smooth((t - 1.2 - 0.22 * i) / 0.3)
            if g <= 0:
                continue
            x, y = self.px[i], self.py[i]
            r = 9 + 22 * (1 - g)
            d.ellipse([x - r, y - r, x + r, y + r], outline=GOLD + (int(255 * g),), width=3)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 245, 220, int(255 * g)))
            if w["seed"] == USA_SEED:
                lab = "the run you just watched"
                tw = d.textlength(lab, font=ST.font(24))
                d.rounded_rectangle([x + 18, y - 20, x + 40 + tw, y + 20], 10, fill=ST.PANEL + (int(215 * g),))
                text(d, (x + 29, y), lab, 24, GOLD, g, "SemiBold", "lm")
        fr = np.asarray(im).copy()
        s = self.secs
        caps = [(0.4, 5.6, [(f"I ran it {len(s)} times. Typically {fmt_hm(float(np.median(s)))} of light,", 40, INK, "SemiBold"),
                            (f"from {fmt_hm(s[0])} to {fmt_hm(s[-1])}.", 34, INK2, "Regular")], {"anchor": "top", "y": 18}),
                (6.0, self.dur, [("Each dot is where a run's last road was.", 40, INK, "SemiBold"),
                                 ("Most are on the coast: islands, peninsulas, bayous.", 34, GOLD, "SemiBold")], {"anchor": "top", "y": 18})]
        return ST.draw_captions(fr, t, caps)


class Outro(Section):
    name, dur = "outro", 20.0
    AU = 149_597_870.7
    PLANETS = (("Mercury", 0.387), ("Venus", 0.723), ("Earth", 1.0), ("Mars", 1.524), ("Jupiter", 5.203),
               ("Saturn", 9.537), ("Uranus", 19.19), ("Neptune", 30.07))

    def __init__(self):
        self.meta, self.w, self.log, self.secs = usa_data()

    def frame(self, t):
        fr = np.zeros((H, W, 3), np.uint8); fr[:] = (5, 8, 15)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a = ST.fade(t, 0.2, 13.0, 0.6, 0.6)
        x0, x1, y = 150, 1780, 600
        kpx = (x1 - x0) / (31 * self.AU)
        d.line([(x0, y), (x1, y)], fill=(60, 70, 90, int(255 * a)), width=2)
        d.ellipse([x0 - 26, y - 26, x0 + 26, y + 26], fill=(255, 210, 120, int(255 * a)))
        text(d, (x0, y + 44), "Sun", 26, INK2, a, "SemiBold", "ma")
        for i, (nm, au) in enumerate(self.PLANETS):
            x = x0 + au * self.AU * kpx
            r = 9 if au > 4 else 5
            d.ellipse([x - r, y - r, x + r, y + r], fill=(150, 170, 200, int(255 * a)))
            if au > 4:
                text(d, (x, y + 44), nm, 26, INK2, a, "SemiBold", "ma")
                lt = au * self.AU / CV.C_KMS
                text(d, (x, y + 78), "light: " + (f"{lt / 60:.0f} min" if lt < 3600 else f"{int(lt // 3600)} h {int(lt % 3600 // 60)} min"), 22, MUTED, a, "Regular", "ma")
        D = self.w["km"]
        g = smooth((t - 1.2) / 3.0)
        xe = x0 + D * kpx * g
        d.line([(x0 + 26, y), (xe, y)], fill=GOLD + (int(255 * a),), width=8)
        if g > 0:
            d.ellipse([xe - 12, y - 12, xe + 12, y + 12], fill=(255, 245, 220, int(255 * a)))
        ga = a * smooth((t - 4.2) / 0.5)
        text(d, (xe, y - 40), f"{D / 1e9:.2f} billion km: the photon's drive", 30, GOLD, ga, "Bold", "md")
        fr = np.asarray(im).copy()
        s = self.secs
        med, lo, hi = float(np.median(s)), s[0], s[-1]
        caps = [(0.6, 6.4, [("In a straight line, that drive would take you past Saturn.", 40, INK, "SemiBold")], {"anchor": "top", "y": 250}),
                (6.8, 12.8, [(f"Driving each road exactly once would take light {self.meta['km'] / CV.C_KMS:.0f} seconds.", 40, INK, "SemiBold"),
                             (f"At random, it drives every kilometre about {round(self.w['x_network'], -1):.0f} times over.", 32, INK2, "Regular")],
                 {"anchor": "top", "y": 250}),
                (13.4, 20.0, [("What should the photon try next?", 54, GOLD, "Bold"),
                              ("Tell me in the comments.", 36, INK2, "Regular")], {"anchor": "top", "y": 420})]
        return ST.draw_captions(fr, t, caps)


SECTIONS = {"cal_intro": lambda: CalRun("intro"), "cal_run": lambda: CalRun("run"), "rule": Rule, "compare": Compare,
            "usa_intro": lambda: USARun("intro"), "usa_run": lambda: USARun("run"), "last": Last, "runs": Runs, "outro": Outro}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
