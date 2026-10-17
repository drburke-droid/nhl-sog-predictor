"""How long would a Roomba take to drive every road? The video (1920x1080, 30 fps), caption-only.

    python video.py NAME [still times...]
    NAME: room, rule, nyc_intro, nyc_run, usa_intro, usa_run, last, runs, payoff
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
        # in chunks of CHUNK frames, so an interrupted render resumes; each resumed chunk first replays WARM frames
        # unrecorded, so trails and glows carry over as in an unbroken render
        CHUNK, WARM = 150, 30
        cdir = os.path.join(OUT, "chunks")
        os.makedirs(cdir, exist_ok=True)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        parts = []
        prev_done = True
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            part = os.path.join(cdir, f"{self.name}_{c0:05d}.mp4")
            parts.append(part)
            if os.path.exists(part) and os.path.exists(part + ".ok"):
                prev_done = False
                continue
            if not prev_done:
                for i in range(max(0, c0 - WARM), c0):
                    self.frame(i / FPS)
            tmp = part + ".tmp.mp4"
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                                  "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
            for i in range(c0, c1):
                p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            p.stdin.close(); p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            prev_done = True
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q); os.remove(q + ".ok")
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


# ================================================================== the cold open: a Roomba in a room
class RoomScene(Section):
    name, dur = "room", 16.0

    def __init__(self):
        import room as RM
        self.RM = RM
        rw, rh = RM.RW * RM.PX_M, RM.RH * RM.PX_M
        self.ox, self.oy = int(960 - rw / 2), int(468 - rh / 2)
        self.R = RM.Room(self.ox, self.oy, W, H)
        im = Image.new("RGB", (W, H), (5, 8, 15))
        d = ImageDraw.Draw(im)
        px = lambda x, y: (self.ox + x * RM.PX_M, self.oy + y * RM.PX_M)
        d.rectangle([*px(0, 0), *px(RM.RW, RM.RH)], fill=(18, 22, 32))
        for k in range(1, int(RM.RW / 0.3)):                                  # floorboards
            x = self.ox + k * 0.3 * RM.PX_M
            d.line([(x, self.oy), (x, self.oy + rh)], fill=(22, 27, 38), width=2)
        for nm, (x0, y0, x1, y1) in RM.BOXES:
            d.rounded_rectangle([*px(x0, y0), *px(x1, y1)], 10, fill=(40, 46, 62), outline=(86, 96, 118), width=3)
            d.text(px((x0 + x1) / 2, (y0 + y1) / 2), nm, font=ST.font(24), fill=(130, 140, 160), anchor="mm")
        x, y, r = RM.PLANT
        d.ellipse([*px(x - r, y - r), *px(x + r, y + r)], fill=(34, 64, 48), outline=(70, 120, 90), width=3)
        d.rectangle([*px(0, 0), *px(RM.RW, RM.RH)], outline=(150, 160, 180), width=8)
        self.base = np.asarray(im).astype(np.float32)

    def sim_t(self, t):
        if t < 4.0:
            return 3.0 * t                                 # 3x real time: a couple of bumps
        u = min((t - 4.0) / 10.5, 1.0)
        return 12.0 + (1800.0 - 12.0) * u ** 2.4           # then a time-lapse to 30 minutes

    def frame(self, t):
        st = self.sim_t(t)
        k = self.R.advance(st)
        cov = (self.R.cov.astype(bool) & self.R.free).astype(np.float32)
        img = self.base * (1 - 0.42 * cov[..., None]) + np.array(GOLD, np.float32) * 0.42 * cov[..., None]
        # the last few seconds of path, brighter
        j0 = max(0, k - int(25.0 / self.R.dt))
        tr = np.zeros((H, W), np.float32)
        if k > j0:
            cv2.polylines(tr, [self.R.P[j0:k + 1].astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
        img = img + tr[..., None] * np.array([255, 235, 190], np.float32) * 0.5
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        hd = self.R.heading(k)
        self.last_heading = hd if hd is not None else getattr(self, "last_heading", 0.0)
        maps.draw_roomba(im, self.R.P[k][0], self.R.P[k][1], self.last_heading, diam=int(2 * self.RM.R * self.RM.PX_M))
        d = ImageDraw.Draw(im, "RGBA")
        a = ST.fade(t, 0.6, 1e9, 0.5, 0.5)
        mm, ss = divmod(int(st), 60)
        x0 = self.ox + self.RM.RW * self.RM.PX_M + 40
        text(d, (x0, self.oy), "CLEANING TIME", 22, MUTED, a)
        text(d, (x0, self.oy + 26), f"{mm}:{ss:02d}", 56, GOLD, a, "Bold")
        text(d, (x0, self.oy + 110), "FLOOR COVERED", 22, MUTED, a)
        text(d, (x0, self.oy + 136), f"{self.R.covered() * 100:.0f}%", 56, INK, a, "Bold")
        sp = "real time × 3" if t < 4.0 else "time-lapse"
        text(d, (x0, self.oy + 220), sp, 24, INK2, a, "Regular")
        fr = np.asarray(im.convert("RGB")).copy()
        caps = [(0.4, 4.6, [("The first Roombas had no map.", 44, INK, "SemiBold")]),
                (4.9, 9.9, [("Drive straight. Bump into something. Turn a random way. Repeat.", 40, INK, "SemiBold")]),
                (10.2, 15.8, [("Given time, it covers nearly the whole floor.", 40, INK, "SemiBold"),
                              ("The last few percent take the longest.", 34, GOLD, "SemiBold")])]
        return ST.draw_captions(fr, t, caps)


# ================================================================== the rule
class Rule(Section):
    name, dur = "rule", 13.0

    def frame(self, t):
        fr = np.zeros((H, W, 3), np.uint8); fr[:] = (5, 8, 15)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        cx, cy, L = 960, 520, 300
        a = ST.fade(t, 0.2, self.dur, 0.5, 0.5)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.line([(cx, cy), (cx + dx * L, cy + dy * L)], fill=(70, 90, 120, int(255 * a)), width=26)
        # the road up ends in a dead end
        d.line([(cx - 40, cy - L), (cx + 40, cy - L)], fill=(150, 160, 180, int(255 * a)), width=8)
        text(d, (cx + 56, cy - L), "dead end", 28, INK2, a, "SemiBold", "lm")
        if 2.8 < t < 6.8:
            g = smooth((t - 2.8) / 0.6) * a * (1 - smooth((t - 6.2) / 0.6))
            for dx, dy, lx, ly in ((1, 0, L * 0.55, -56), (0, -1, 56, -L * 0.45), (0, 1, 56, L * 0.55)):
                ex, ey = cx + dx * L * 0.62, cy + dy * L * 0.62
                d.line([(cx + dx * 46, cy + dy * 46), (ex, ey)], fill=GOLD + (int(230 * g),), width=6)
                text(d, (cx + lx, cy + ly), "⅓", 44, GOLD, g, "Bold", "mm")
            d.line([(cx - 46, cy), (cx - L * 0.62, cy)], fill=RED + (int(200 * g),), width=6)
            qx, qy, q = cx - L * 0.36, cy - 60, 16
            d.line([(qx - q, qy - q), (qx + q, qy + q)], fill=RED + (int(255 * g),), width=7)
            d.line([(qx - q, qy + q), (qx + q, qy - q)], fill=RED + (int(255 * g),), width=7)
            text(d, (cx - L * 0.5, cy + 48), "the way it came in", 28, RED, g, "SemiBold", "ma")
        # the Roomba: in from the left, picks the road up, meets the dead end, turns round, comes back and goes right
        keys = [(0.6, (cx - L, cy), 0.0), (2.6, (cx, cy), 0.0), (4.6, (cx, cy), 0.0), (5.0, (cx, cy), -math.pi / 2),
                (7.4, (cx, cy - L + 30), -math.pi / 2), (8.0, (cx, cy - L + 30), math.pi / 2), (10.0, (cx, cy), math.pi / 2),
                (10.4, (cx, cy), 0.0), (12.4, (cx + L, cy), 0.0)]
        if t <= keys[0][0]:
            pos, ang = keys[0][1], keys[0][2]
        elif t >= keys[-1][0]:
            pos, ang = keys[-1][1], keys[-1][2]
        else:
            for (t0, p0, a0), (t1, p1, a1) in zip(keys, keys[1:]):
                if t0 <= t <= t1:
                    u = smooth((t - t0) / (t1 - t0))
                    pos = (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
                    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                    ang = a0 + da * u
                    break
        maps.draw_roomba(im, pos[0], pos[1], ang, diam=56)
        fr = np.asarray(im.convert("RGB")).copy()
        caps = [(0.4, 6.4, [("On streets, the same idea: at every intersection, take any road", 40, INK, "SemiBold"),
                            ("except the one you came in on. Every choice equally likely.", 34, INK2, "Regular")]),
                (6.8, 12.8, [("Dead end? Turn around.", 40, INK, "SemiBold"),
                             ("No map. No memory. No plan.", 34, GOLD, "SemiBold")])]
        return ST.draw_captions(fr, t, caps)


# ================================================================== a city, at the speed of light
NYC = dict(prefix="nyc", seed=30, km_px=21.0, cx=0.5, cy=-6.7, ox=1180, oy=540, title="NEW YORK CITY · EVERY STREET",
           start="Times Square", last_street="Beach 6th Street", last_place="Far Rockaway, Queens",
           last_note="at the far end of the Rockaway peninsula", half="Half of New York's streets")


def city_stats(prefix):
    js = json.load(open(os.path.join(DATA, f"{prefix}_cover_many.json")))
    s = js["seconds_nonback"]
    return dict(median_s=float(np.median(s)), n=len(s), min_s=float(min(s)), max_s=float(max(s)))


class CityRun(Section):
    """A city from the first metre to the last street. Phase 'intro' is the slow start (the question);
    phase 'run' is the rest, accelerating, with the HUD and milestones."""

    def __init__(self, phase, cfg=NYC):
        self.phase, self.cfg = phase, cfg
        c = cfg
        self.C = maps.City(c["prefix"], c["seed"], c["km_px"], c["cx"], c["cy"], c["ox"], c["oy"])
        self.T_km = self.C.total_km
        self.net_km = float(self.C.elen.sum())
        self.log = np.load(os.path.join(DATA, f"{c['prefix']}_log_{c['seed']}.npy"))
        self.D0 = 60.0                                        # km driven during the intro
        self.name = f"{c['prefix']}_{phase}"
        self.dur = 15.0 if phase == "intro" else 56.0
        if phase == "run":
            self.C.map.advance(self.D0)
            self.C.k = self.C.step_at(self.D0)
        self.lastD = -1
        f = self.log[:, 3] / self.net_km
        self.ms = []
        for q, lab in ((0.5, c["half"]), (0.9, "90% of streets"), (0.99, "99% of streets"), (0.999, "99.9% of streets")):
            k = int(np.searchsorted(f, q))
            self.ms.append((lab, float(self.log[min(k, len(self.log) - 1), 1])))
        bt = self.C.block_t
        self.gap_D = float(bt[-3])                            # two blocks left from here
        e = int(np.argmax(self.C.first))
        self.last_seg = e
        self.last_block_m = float(self.C.elen[self.C.blk == self.C.blk[e]].sum()) * 1000
        self.T_end = self.dur - 6.0

    def settle(self):
        self.C.trail[:] = 0

    def D(self, t):
        if self.phase == "intro":
            u = t / self.dur
            return self.D0 * (math.exp(5.0 * u) - 1) / (math.exp(5.0) - 1)
        u = min(t / self.T_end, 1.0)
        return self.D0 + (self.T_km - self.D0) * u ** 3.4

    def t_of(self, Dq):
        u = ((Dq - self.D0) / (self.T_km - self.D0)) ** (1 / 3.4)
        return u * self.T_end

    def frame(self, t):
        D = self.D(t)
        if D < self.lastD:                                     # stills out of order: rebuild state
            self.__init__(self.phase, self.cfg)
        self.lastD = D
        fr = self.C.frame(D)
        if self.phase == "intro":
            fr = (maps.BG + (fr.astype(np.float32) - maps.BG) * smooth(t / 1.5)).astype(np.uint8)
        im = Image.fromarray(fr)
        hx, hy, ang = self.C.head(D)
        maps.draw_roomba(im, hx, hy, ang, diam=26)
        d = ImageDraw.Draw(im, "RGBA")
        sec = D / CV.C_KMS
        left = self.C.blocks_left(D)
        a_h = ST.fade(t, 1.0 if self.phase == "intro" else 0.0, 1e9, 0.6, 0.6)
        x0 = 60
        text(d, (x0, 60), self.cfg["title"], 26, MUTED, a_h, "SemiBold")
        text(d, (x0, 100), "LIGHT TIME", 22, MUTED, a_h)
        text(d, (x0, 128), fmt_light(sec), 84, GOLD, a_h, "Bold")
        text(d, (x0, 250), "BLOCKS NOT YET DRIVEN", 22, MUTED, a_h)
        text(d, (x0, 278), f"{left:,}", 64, INK if left > 0 else GOLD, a_h, "Bold")
        text(d, (x0, 380), "DISTANCE DRIVEN", 22, MUTED, a_h)
        text(d, (x0, 408), f"{D:,.0f} km", 48, INK, a_h, "Bold")
        if D > self.net_km:
            text(d, (x0, 470), f"{D / self.net_km:,.1f}× the city's {self.net_km:,.0f} km of streets", 24, INK2, a_h, "Regular")
        if self.phase == "intro":
            sx, sy = self.C.X[self.C.start], self.C.Y[self.C.start]
            g = ST.fade(t, 1.0, 6.0, 0.5, 0.6)
            d.ellipse([sx - 30, sy - 30, sx + 30, sy + 30], outline=GOLD + (int(255 * g),), width=4)
            lab = f"start: {self.cfg['start']}"
            tw = d.textlength(lab, font=ST.font(28))
            d.rounded_rectangle([sx + 40, sy - 24, sx + 64 + tw, sy + 24], 12, fill=ST.PANEL + (int(215 * g),))
            text(d, (sx + 52, sy), lab, 28, GOLD, g, "SemiBold", "lm")
        else:
            self.chart(d, D, a_h)
            if t > self.T_end:                                # the last block: a ring when it is found
                e = self.last_seg
                lx, ly = self.C.mx[e], self.C.my[e]
                g = smooth((t - self.T_end) / 0.5) * ST.fade(t, 0, self.dur, 0, 0.5)
                r = 22 + 60 * (1 - smooth((t - self.T_end) / 0.5))
                d.ellipse([lx - r, ly - r, lx + r, ly + r], outline=GOLD + (int(255 * g),), width=4)
                lab = f"the last block: {self.cfg['last_street']}"
                tw = d.textlength(lab, font=ST.font(28))
                bx = lx - 34 - tw - 24                                    # label to the left (the city edge is on the right)
                d.rounded_rectangle([bx, ly - 24, lx - 30, ly + 24], 12, fill=ST.PANEL + (int(215 * g),))
                text(d, (bx + 12, ly), lab, 28, GOLD, g, "SemiBold", "lm")
        fr = np.asarray(im.convert("RGB")).copy()
        return ST.draw_captions(fr, t, self.caps())

    def chart(self, d, D, a):
        """blocks not yet driven (log scale) vs light time"""
        x0, y0, w, h = 60, 600, 520, 300
        if not hasattr(self, "curve"):
            Ds = np.linspace(0, self.T_km, 600)
            self.curve = (Ds, np.maximum(self.C.nblk - np.searchsorted(self.C.block_t, Ds, side="right"), 1))
        Ds, L = self.curve
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 70, 90, int(200 * a)), width=2)
        k = int(np.searchsorted(Ds, D))
        top = math.log10(self.C.nblk)
        xs = x0 + Ds[:k + 1] / self.T_km * w
        ys = y0 + h - np.log10(L[:k + 1]) / top * h
        if k > 1:
            d.line(list(zip(xs, ys)), fill=GOLD + (int(255 * a),), width=3)
        text(d, (x0, y0 + h + 12), "0", 20, MUTED, a)
        text(d, (x0 + w, y0 + h + 12), f"{self.T_km / CV.C_KMS:.2f} s", 20, MUTED, a, anchor="ra")
        text(d, (x0 + w + 8, y0), f"{self.C.nblk:,}", 18, MUTED, a, "Regular", "lm")
        text(d, (x0 + w + 8, y0 + h), "1", 18, MUTED, a, "Regular", "lm")
        text(d, (x0, y0 - 34), "blocks not yet driven (log scale)", 22, INK2, a)

    def caps(self):
        cxk = {"cx": self.cfg["ox"]}
        top = dict(anchor="top", **cxk)
        if self.phase == "intro":
            return [(0.5, 5.0, [("How long would a Roomba take to drive down", 40, INK, "SemiBold"),
                                ("every street in New York City?", 40, INK, "SemiBold")], top),
                    (5.4, 9.6, [("Not cleaning them. Just driving down each one.", 40, INK, "SemiBold")], top),
                    (10.0, 14.8, [("And so we can watch it: at the speed of light.", 44, GOLD, "Bold")], top)]
        caps = []
        Ts = [(lab, self.t_of(D), D) for lab, D in self.ms]
        tg = self.t_of(self.gap_D)
        for (lab, tt, D), nxt in zip(Ts, Ts[1:] + [("", tg, 0)]):
            caps.append((tt, min(tt + 3.4, nxt[1] - 0.2), [(f"{lab}: {fmt_light(D / CV.C_KMS)}", 40, GOLD, "Bold")], top))
        gap = (self.T_km - self.gap_D)
        caps.append((tg + 0.2, self.T_end - 0.3, [("Two blocks left.", 44, GOLD, "Bold"),
                     (f"Finding them takes {gap / self.T_km:.0%} of the whole trip.", 32, INK2, "Regular")], top))
        st = city_stats(self.cfg["prefix"])
        caps.append((self.T_end + 0.4, self.dur, [(f"Every street in New York: {fmt_light(self.T_km / CV.C_KMS)}", 46, GOLD, "Bold"),
                     (f"The last block: {self.cfg['last_street']}, {self.cfg['last_place']},", 30, INK2, "Regular"),
                     (f"{self.cfg['last_note']}. (Typical of {st['n']} runs: {fmt_light(st['median_s'])}.)", 30, INK2, "Regular")],
                     top))
        return caps


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
        left = self.U.blocks_left(D)
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
        text(d, (x0, y0 + 112), "BLOCKS NOT YET DRIVEN", 22, MUTED, a)
        text(d, (x0, y0 + 138), f"{left:,}", 52, INK if left > 1 else GOLD, a, "Bold")
        text(d, (x0, y0 + 210), f"{D:,.0f} km driven" if D < 1e9 else f"{D / 1e9:,.2f} billion km driven", 28, INK2, a, "Regular")

    def chart(self, d, D, a):
        """blocks not yet driven (log scale) vs light time"""
        x0, y0, w, h = 1570, 740, 250, 220
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(60, 70, 90, int(200 * a)), width=2)
        if not hasattr(self, "curve"):
            Ds = np.linspace(0, self.T, 800)
            self.curve = (Ds, np.maximum(self.U.nblk - np.searchsorted(self.U.block_t, Ds, side="right"), 1))
        Ds, L = self.curve
        k = int(np.searchsorted(Ds, D))
        xs = x0 + Ds[:k + 1] / self.T * w
        ys = y0 + h - np.log10(L[:k + 1]) / 8 * h
        if k > 1:
            d.line(list(zip(xs, ys)), fill=GOLD + (int(255 * a),), width=3)
        text(d, (x0, y0 - 32), "blocks not yet driven", 22, INK2, a)
        for e in (0, 4, 8):
            text(d, (x0 + w + 8, y0 + h - e / 8 * h), ("1", "10,000", "100 M")[e // 4], 18, MUTED, a, "Regular", "lm")
        text(d, (x0, y0 + h + 10), "0", 18, MUTED, a, "Regular")
        text(d, (x0 + w, y0 + h + 10), fmt_light(self.T / CV.C_KMS).split(" min")[0] + " min", 18, MUTED, a, "Regular", "ra")

    def caps(self):
        top = {"anchor": "top", "y": 18}
        if self.phase == "intro":
            nb = self.U.nblk if self.U is not None else 0
            return [(0.4, 4.6, [("Now: every road in the contiguous United States.", 40, INK, "SemiBold"),
                                (f"{self.meta['km'] / 1e6:.1f} million km of road, {nb / 1e6:.1f} million blocks", 32, INK2, "Regular")], top),
                    (5.0, 10.0, [("Same Roomba. Same rule. Still at the speed of light.", 42, GOLD, "Bold")], top)]
        caps = []
        Ts = [(lab, self.t_of(D), D) for lab, D in self.ms]
        for (lab, tt, D), nxt in zip(Ts, Ts[1:] + [("", self.RUN + 1.0, 0)]):
            caps.append((tt, min(tt + 3.2, nxt[1] - 0.2), [(f"{lab}: {fmt_light(D / CV.C_KMS)}", 40, GOLD, "Bold")], top))
        left = self.U.blocks_left(self.Dend) if self.U is not None else 0
        caps.append((self.RUN + 0.3, self.dur, [(f"{left:,} blocks left." if left > 1 else "One block left.", 46, GOLD, "Bold")], top))
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


@functools.lru_cache(maxsize=2)
def late_block_count(T):
    """how many blocks were still undriven when the slow motion starts"""
    late = usa_tail(T)["late"]
    return len(np.unique(np.load(os.path.join(DATA, "usa_block.npy"), mmap_mode="r")[late]))


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
        blk = np.load(os.path.join(DATA, "usa_block.npy"), mmap_mode="r")[self.late]
        ub, inv = np.unique(blk, return_inverse=True)
        bf = np.full(len(ub), -np.inf)
        np.maximum.at(bf, inv, self.late_km)
        self.late_block_t, self.n_late_blocks = np.sort(bf), len(ub)
        self.last_block_m = float(Z.road_km[self.late[blk == blk[-1]]].sum()) * 1000
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
            head = (float(sx[-1]), float(sy[-1]), math.atan2(sy[-1] - sy[-2], sx[-1] - sx[-2]) if len(sx) > 1 else 0.0)
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
        dark = self.n_late_blocks - int(np.searchsorted(self.late_block_t, D + 1e-9, side="right"))
        lc = ((x0[-1] + x1[-1]) / 2, (y0[-1] + y1[-1]) / 2)
        return fr, f, D, dark, head, s, lc

    def frame(self, t):
        fr, f, D, dark, head, s, lc = self.map_frame(t)
        im = Image.fromarray(fr)
        if head is not None:
            maps.draw_roomba(im, head[0], head[1], head[2], diam=30)
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
            lab = f"the last block: {road}"
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
        text(d, (x0, y0 + 112), "BLOCKS NOT YET DRIVEN", 22, MUTED, 1.0)
        text(d, (x0, y0 + 138), f"{dark:,}", 52, GOLD, 1.0, "Bold")
        fr = np.asarray(im.convert("RGB")).copy()
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
        n = self.n_late_blocks if self.Z is not None else 0
        last_m = self.last_block_m if self.Z is not None else 0
        first = (f"{n:,} blocks still dark, {LAST_NOTE.get(USA_SEED, 'all in one corner of the map')}." if n > 1
                 else "One block still dark.")
        top = {"anchor": "top", "y": 18}
        return [(0.3, self.Z0 + 2.4, [(first, 42, INK, "SemiBold")], top),
                (self.Z0 + 2.8, self.S0 - 0.3, [("Islands and peninsulas have few ways in.", 40, INK, "SemiBold"),
                                               ("A Roomba with no map can miss them for hours.", 34, INK2, "Regular")],
                 {"anchor": "top", "y": 110}),
                (self.ARRIVE, self.dur, [(f"Every road in America: {fmt_light(T)}", 46, GOLD, "Bold"),
                 (f"The last block: {last_m:.0f} m of {road}, after {self.T / 1e9:.2f} billion km.", 30, INK2, "Regular"),
                 ("That's farther than Saturn is from the Sun.", 30, INK2, "Regular")],
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
                (6.0, self.dur, [("Each dot is where a run's last block was.", 40, INK, "SemiBold"),
                                 ("Most are on the coast: islands, peninsulas, bayous.", 34, GOLD, "SemiBold")], {"anchor": "top", "y": 18})]
        return ST.draw_captions(fr, t, caps)


class Payoff(Section):
    """Light speed vs a real Roomba."""
    name, dur = "payoff", 21.0
    V = 0.5                       # m/s: a Roomba's top drive speed (the 600-series / Create 2 drive limit)
    YEAR = 365.25 * 86400

    def __init__(self):
        meta, w, log, self.secs = usa_data()
        self.usa_s = float(np.median(self.secs))
        self.nyc = city_stats("nyc")

    def years(self, light_s):
        return light_s * CV.C_KMS * 1000 / self.V / self.YEAR

    def frame(self, t):
        fr = np.zeros((H, W, 3), np.uint8); fr[:] = (5, 8, 15)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        cols = (1000, 1500)
        a1 = ST.fade(t, 0.4, 1e9, 0.5, 0.5)
        a2 = ST.fade(t, 6.6, 1e9, 0.5, 0.5)
        for x, nm in zip(cols, ("NEW YORK", "AMERICA")):
            text(d, (x, 210), nm, 34, MUTED, a1, "SemiBold", "ma")
        text(d, (180, 330), "at the speed of light", 36, INK2, a1, "SemiBold", "lm")
        text(d, (cols[0], 330), fmt_light(self.nyc["median_s"]), 64, GOLD, a1, "Bold", "mm")
        text(d, (cols[1], 330), fmt_hm(self.usa_s), 64, GOLD, a1, "Bold", "mm")
        text(d, (180, 520), "at a real Roomba's", 36, INK2, a2, "SemiBold", "lm")
        text(d, (180, 566), "top speed (0.5 m/s)", 36, INK2, a2, "SemiBold", "lm")
        g = smooth((t - 7.2) / 2.5)
        ny, us = self.years(self.nyc["median_s"]) * g, self.years(self.usa_s) * g
        text(d, (cols[0], 540), f"{ny:,.0f} years", 64, INK, a2, "Bold", "mm")
        text(d, (cols[1], 540), f"{round(us, -3):,.0f} years" if g >= 1 else f"{us:,.0f} years", 64, INK, a2, "Bold", "mm")
        d.line([(180, 430), (1760, 430)], fill=(50, 60, 80, int(255 * a1)), width=2)
        # a Roomba setting off, at (scaled) Roomba pace
        rx = 180 + 60 * max(t - 7.0, 0)
        if t > 6.6:
            maps.draw_roomba(im, rx, 700, 0.0, diam=48)
        fr = np.asarray(im.convert("RGB")).copy()
        caps = [(0.4, 6.2, [("Those are light-speed times: typical of all the runs.", 40, INK, "SemiBold")]),
                (6.6, 12.4, [("A real Roomba is a little slower.", 40, INK, "SemiBold")]),
                (12.8, 17.0, [("And that's just driving down each road.", 40, INK, "SemiBold"),
                              ("To clean one? A two-lane road is about 20 Roombas wide.", 34, INK2, "Regular")]),
                (17.3, 21.0, [("What should the Roomba try next?", 50, GOLD, "Bold"),
                              ("Tell me in the comments.", 34, INK2, "Regular")])]
        return ST.draw_captions(fr, t, caps)


SECTIONS = {"room": RoomScene, "rule": Rule, "nyc_intro": lambda: CityRun("intro"), "nyc_run": lambda: CityRun("run"),
            "usa_intro": lambda: USARun("intro"), "usa_run": lambda: USARun("run"), "last": Last, "runs": Runs, "payoff": Payoff}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
