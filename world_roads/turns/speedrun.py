"""LIGHT SPEEDRUN: a photon picks a random road at every intersection on every street in North America,
attempt after attempt, chasing a personal best from LA to New York. Vertical Short.

The attempts are the real simulated walks from full_photon.py (seeds 0..199, in order); each run is
replayed exactly (same random sequence) so the times, splits and record progression are genuine.

    python speedrun.py prep       # background map + node->pixel table + PB runs (cached)
    python speedrun.py still 30   # one frame at t = 30 s
    python speedrun.py            # -> out/speedrun.mp4 (with music)
"""
import os, sys, json, subprocess
import numpy as np
import numba as nb
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "flights"))
import full_photon as FP  # noqa: E402
import lost_photon as LP  # noqa: E402

DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
FD = "/usr/share/fonts/opentype/inter"
VW, VH, FPS = 1080, 1920, 30
S = 2
MAP_TOP, MAP_W = 236, 1060
LON0, LAT0 = -98.0, 42.0
C = LP.C_KMS
INK, INK2, MUTED = (240, 244, 250), (160, 178, 200), (110, 128, 150)
GOLD_T, GREEN, RED = (255, 186, 80), (90, 220, 120), (255, 92, 92)
CYAN = np.array([0.14, 0.66, 1.00], np.float32) * 0.40
GOLD = np.array([1.00, 0.52, 0.10], np.float32)
HOT = np.array([1.00, 0.93, 0.80], np.float32)
SPLITS = ["Escape LA (100 km)", "Rockies (105°W)", "Mississippi (90°W)", "New York"]
PB_SEEDS = [0, 50, 101, 148, 181, 193]          # attempts 1, 51, 102, 149, 182, 194


def font(n, s, tab=False):
    return ImageFont.truetype(os.path.join(FD, n), s, layout_engine=ImageFont.Layout.RAQM)


def laea(lon, lat):
    lo, la = np.radians(np.asarray(lon, np.float64) - LON0), np.radians(np.asarray(lat, np.float64))
    p0 = np.radians(LAT0)
    k = np.sqrt(2 / (1 + np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo)))
    return LP.R * k * np.cos(la) * np.sin(lo), LP.R * k * (np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo))


def fit():
    G = LP.load()                                           # same continental extent as the full network
    x, y = laea(G["lon"], G["lat"])
    sc = (MAP_W - 20) / (x.max() - x.min())
    MH = int(np.ceil((y.max() - y.min()) * sc)) + 20
    return sc, MH, 10 - x.min() * sc, 10 + y.max() * sc


SC, MH, OX, OY = None, None, None, None


def setup():
    global SC, MH, OX, OY
    cache = os.path.join(DATA, "speedrun_fit.json")
    if os.path.exists(cache):
        SC, MH, OX, OY = json.load(open(cache))
    else:
        SC, MH, OX, OY = fit()
        json.dump([SC, MH, OX, OY], open(cache, "w"))


def to_px(lon, lat):
    x, y = laea(lon, lat)
    return x * SC + OX, -y * SC + OY


# ------------------------------------------------------------------------------------------ walking
@nb.njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@nb.njit(cache=True)
def advance(indptr, adj, alen, goal, pix, far, lonn, heat, recent, st, kmv, km_target, splits):
    """Continue a walk until km_target (or arrival). st = [prev, cur, steps, done]; kmv = [km].
    Identical RNG use to full_photon.walks(), so a seeded run reproduces the same walk."""
    prev, cur, steps = st[0], st[1], st[2]
    km = kmv[0]
    while km < km_target:
        if goal[cur]:
            st[3] = 1
            if splits[3] < 0:
                splits[3] = km
            break
        a = indptr[cur]; b = indptr[cur + 1]
        d = b - a
        n_ex = d
        for j in range(a, b):
            if adj[j] == prev:
                n_ex -= 1
                break
        if n_ex <= 0:
            j_pick = a
        else:
            k = np.random.randint(n_ex)
            j_pick = a
            skipped = False
            for j in range(a, b):
                if not skipped and adj[j] == prev and n_ex < d:
                    skipped = True
                    continue
                if k == 0:
                    j_pick = j
                    break
                k -= 1
        km += alen[j_pick]
        prev = cur; cur = adj[j_pick]
        steps += 1
        p = pix[cur]
        if p >= 0:
            heat[p] += 1.0
            recent[p] = 1.0
        if splits[0] < 0 and far[cur]:
            splits[0] = km
        if splits[1] < 0 and lonn[cur] > -105.0:
            splits[1] = km
        if splits[2] < 0 and lonn[cur] > -90.0:
            splits[2] = km
    st[0] = prev; st[1] = cur; st[2] = steps
    kmv[0] = km


class World:
    def __init__(self):
        setup()
        self.F = FP.load_fast()
        F = self.F
        self.W2, self.H2 = MAP_W * S, MH * S
        cache = os.path.join(DATA, "speedrun_pix.npy")
        if os.path.exists(cache):
            self.pix = np.load(cache)
        else:
            pix = np.empty(len(F["lon"]), np.int32)
            for i in range(0, len(pix), 5_000_000):
                X, Y = to_px(F["lon"][i:i + 5_000_000], F["lat"][i:i + 5_000_000])
                xi, yi = np.floor(X * S).astype(np.int64), np.floor(Y * S).astype(np.int64)
                ok = (xi >= 0) & (xi < self.W2) & (yi >= 0) & (yi < self.H2)
                pix[i:i + 5_000_000] = np.where(ok, yi * self.W2 + xi, -1)
            np.save(cache, pix)
            self.pix = pix
        self.far = (LP.gc_km(F["lon"].astype(np.float64), F["lat"].astype(np.float64), *LP.LA) > 100.0)
        self.lonn = F["lon"]
        self.bg = self.background()

    def background(self):
        cache = os.path.join(DATA, "speedrun_bg.npy")
        if os.path.exists(cache):
            return np.load(cache)
        F = self.F
        dens = np.bincount(self.pix[self.pix >= 0], minlength=self.W2 * self.H2).astype(np.float32).reshape(self.H2, self.W2)
        # long rural roads would be dotted at this scale: draw them as lines
        src = np.repeat(np.arange(len(F["lon"]), dtype=np.int64), np.diff(F["indptr"]))
        long_ = np.nonzero((F["alen"] > 2.0) & (src < F["adj"]))[0]
        a, b = src[long_], F["adj"][long_].astype(np.int64)
        ax, ay = to_px(F["lon"][a], F["lat"][a]); bx, by = to_px(F["lon"][b], F["lat"][b])
        P = np.rint(np.stack([np.c_[ax, ay], np.c_[bx, by]], 1) * S * 8).astype(np.int32)
        lines = np.zeros((self.H2, self.W2), np.uint8)
        for i in range(0, len(P), 500_000):
            cv2.polylines(lines, list(P[i:i + 500_000]), False, 140, 1, cv2.LINE_AA, shift=3)
        I = np.maximum(np.clip(np.log1p(dens) / np.log1p(60), 0, 1), lines.astype(np.float32) / 255)
        np.save(cache, I.astype(np.float32))
        return I.astype(np.float32)

    def new_run(self, seed):
        seed_rng(seed)
        return dict(seed=seed, st=np.array([-1, int(self.F["start"]), 0, 0], np.int64), km=np.zeros(1),
                    splits=np.full(4, -1.0), heat=np.zeros(self.W2 * self.H2, np.float32),
                    recent=np.zeros(self.W2 * self.H2, np.float32))

    def step(self, run, km_target):
        F = self.F
        advance(F["indptr"], F["adj"], F["alen"], F["goal"], self.pix, self.far, self.lonn, run["heat"], run["recent"],
                run["st"], run["km"], km_target, run["splits"])


def prep_pb_runs(W):
    """Full runs for the personal bests: final heat (for the montage) and splits (for comparisons)."""
    out = {}
    for s in PB_SEEDS:
        path = os.path.join(DATA, f"speedrun_pb_{s}.npz")
        if os.path.exists(path):
            z = np.load(path); out[s] = dict(heat=z["heat"], splits=z["splits"], km=float(z["km"]))
            continue
        r = W.new_run(s)
        W.step(r, 1e18)
        h = cv2.resize(r["heat"].reshape(W.H2, W.W2), (MAP_W, MH), interpolation=cv2.INTER_AREA)
        np.savez(path, heat=h.astype(np.float32), splits=r["splits"], km=r["km"][0])
        out[s] = dict(heat=h, splits=r["splits"].copy(), km=float(r["km"][0]))
        print("pb run", s, r["km"][0] / C, r["splits"] / C, flush=True)
    return out


# ------------------------------------------------------------------------------------------ timeline
def load_attempts():
    d = json.load(open(os.path.join(DATA, "full_photon_fast.json")))
    w = sorted(d["walks"], key=lambda x: x["seed"])
    return np.array([x["seconds"] for x in w]), d["summary"]


def build_timeline(times):
    """List of (start, end, kind, info) segments."""
    seg = []
    t = 0.0
    def add(dur, kind, **info):
        nonlocal t
        seg.append((t, t + dur, kind, info)); t += dur
    add(3.2, "hook")
    add(times[0] / 3.0, "run", attempt=1, seed=0, speed=3)
    add(1.3, "result", attempt=1)
    add(times[1] / 40.0, "run", attempt=2, seed=1, speed=40)
    add(1.0, "result", attempt=2)
    add(times[2] / 80.0, "run", attempt=3, seed=2, speed=80)
    add(1.0, "result", attempt=3)
    # montage of attempts 4..193: quick ticks, holding on each new personal best
    pbs = {51, 102, 149, 182}
    sched = []
    tm = t
    for a in range(4, 194):
        dur = 1.0 if a in pbs else 0.055
        sched.append((tm, a, a in pbs)); tm += dur
    add(tm - t, "montage", sched=sched)
    add(1.4, "wr_intro", attempt=194)
    add(times[193], "run", attempt=194, seed=193, speed=1)
    add(2.0, "result", attempt=194, wr=True)
    add(5.0, "end")
    return seg, t


def fmt(sec, dec=2):
    if sec is None or sec < 0:
        return "–"
    m, s = divmod(sec, 60)
    if dec == 0:
        return f"{int(m)}:{int(round(s)):02d}" if m >= 1 else f"{int(round(s))} s"
    return f"{int(m)}:{s:0{3 + dec}.{dec}f}" if m >= 1 else f"{s:.{dec}f}"


# ------------------------------------------------------------------------------------------ render
class Renderer:
    def __init__(self):
        self.W = World()
        self.pb = prep_pb_runs(self.W)
        self.times, self.summary = load_attempts()
        self.seg, self.total = build_timeline(self.times)
        self.mx0 = (VW - MAP_W) // 2
        k = np.arange(121, dtype=np.float32) - 60
        r2 = k[None] ** 2 + k[:, None] ** 2
        self.dot = (np.exp(-r2 / (2 * 1.8 ** 2))[..., None] * np.array([1, .97, .92]) * 3.0 +
                    np.exp(-r2 / (2 * 5 ** 2))[..., None] * np.array([1, .75, .35]) * 1.4 +
                    np.exp(-r2 / (2 * 18 ** 2))[..., None] * np.array([1, .55, .15]) * 0.5).astype(np.float32)
        bg = cv2.resize(self.W.bg, (MAP_W, MH), interpolation=cv2.INTER_AREA)
        self.bg = bg[..., None] * CYAN
        self.f = dict(title=font("Inter-Black.otf", 70), sub=font("Inter-SemiBold.otf", 32), sub2=font("Inter-Medium.otf", 27),
                      att=font("Inter-Bold.otf", 40), clock=font("InterDisplay-Bold.otf", 118), pb=font("Inter-SemiBold.otf", 32),
                      split=font("Inter-SemiBold.otf", 31), splitt=font("Inter-Bold.otf", 31), badge=font("Inter-Bold.otf", 26),
                      big=font("Inter-Black.otf", 84), mid=font("Inter-Bold.otf", 44), body=font("Inter-Medium.otf", 34),
                      small=font("Inter-Regular.otf", 21))
        cx, cy = to_px(np.array([-121.0]), np.array([17.5]))
        self.clk = (40, int(MAP_TOP + cy[0]) - 60)
        self.la = to_px(np.array([LP.LA[0]]), np.array([LP.LA[1]]))
        self.ny = to_px(np.array([LP.NYC[0]]), np.array([LP.NYC[1]]))
        self.run = None
        self.run_seed = None
        self.pb_splits = None

    # ---- map composition
    def map_img(self, heat2=None, heat1=None, recent2=None, dot=None, dim=1.0, flash=0.0):
        E = self.bg.copy()
        if heat2 is not None:
            h = cv2.resize(heat2.reshape(self.W.H2, self.W.W2), (MAP_W, MH), interpolation=cv2.INTER_AREA)
        else:
            h = heat1
        if h is not None:
            g = np.clip(np.log1p(h * 3) / np.log1p(40), 0, 1)
            E = E * (1 - 0.6 * g[..., None]) + g[..., None] * GOLD * (0.55 + flash)
        if recent2 is not None:
            r = cv2.resize(recent2.reshape(self.W.H2, self.W.W2), (MAP_W, MH), interpolation=cv2.INTER_AREA)
            E += np.clip(r * 3, 0, 1.5)[..., None] * HOT
        E *= dim
        if dot is not None:
            x, y = dot
            xi, yi, hh = int(round(x)), int(round(y)), 60
            ys0, ys1, xs0, xs1 = max(0, yi - hh), min(MH, yi + hh + 1), max(0, xi - hh), min(MAP_W, xi + hh + 1)
            if ys1 > ys0 and xs1 > xs0:
                E[ys0:ys1, xs0:xs1] += self.dot[ys0 - (yi - hh):ys1 - (yi - hh), xs0 - (xi - hh):xs1 - (xi - hh)]
        small = cv2.resize(E, (MAP_W // 2, MH // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2) * 0.5 + cv2.GaussianBlur(small, (0, 0), 8) * 0.45, (MAP_W, MH))
        return 1 - np.exp(-(E + bloom) * 1.25)

    def compose(self, mapimg, draw):
        canvas = np.empty((VH, VW, 3), np.float32); canvas[:] = np.array([4, 7, 13], np.float32) / 255
        canvas[MAP_TOP:MAP_TOP + MH, self.mx0:self.mx0 + MAP_W] = np.maximum(mapimg, canvas[MAP_TOP:MAP_TOP + MH, self.mx0:self.mx0 + MAP_W])
        im = Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8))
        d = ImageDraw.Draw(im, "RGBA")
        draw(d)
        return np.asarray(im)

    # ---- HUD pieces
    def header(self, d):
        d.text((VW // 2, 46), "LIGHT SPEEDRUN", font=self.f["title"], fill=GOLD_T, anchor="ma")
        d.text((VW // 2, 134), "Los Angeles → New York at the speed of light", font=self.f["sub"], fill=INK, anchor="ma")
        d.text((VW // 2, 180), "Rule: a random road at every intersection · every street", font=self.f["sub2"], fill=INK2, anchor="ma")
        for (x, y), nm, anc in [(self.la, "LA", "ra"), (self.ny, "NYC", "la")]:
            X, Y = self.mx0 + x[0], MAP_TOP + y[0]
            d.ellipse([X - 8, Y - 8, X + 8, Y + 8], outline=INK + (230,), width=3)
            d.text((X + (-14 if anc == "ra" else 14), Y + 6), nm, font=self.f["badge"], fill=INK, anchor=anc)

    def clock_block(self, d, attempt, t, speed=None, pb=None, color=(250, 238, 212)):
        x, y = self.clk
        if speed is not None:
            lab = "REAL TIME" if speed == 1 else f"SPED UP ×{speed}"
            w = d.textlength(lab, font=self.f["badge"]) + 36
            d.rounded_rectangle([x, y - 104, x + w, y - 58], 23, fill=(220, 40, 40, 235) if speed == 1 else (60, 80, 110, 235))
            d.text((x + w / 2, y - 81), lab, font=self.f["badge"], fill=(255, 255, 255), anchor="mm")
        if attempt is not None:
            d.text((x, y - 48), f"ATTEMPT {attempt}", font=self.f["att"], fill=INK, anchor="la")
        d.text((x - 6, y), fmt(t, 2), font=self.f["clock"], fill=color, anchor="la", features=["tnum"])
        if pb is not None:
            d.text((x, y + 142), f"Personal best  {fmt(pb, 2)}", font=self.f["pb"], fill=GOLD_T, anchor="la", features=["tnum"])

    def splits_panel(self, d, cur, pbs, show_delta=True):
        y0 = MAP_TOP + MH + 18
        d.rounded_rectangle([40, y0 - 8, VW - 40, y0 + 4 * 54 + 8], 18, fill=(10, 16, 28, 210))
        for i, nm in enumerate(SPLITS):
            y = y0 + 12 + i * 54
            d.text((66, y), nm, font=self.f["split"], fill=INK, anchor="la")
            c = cur[i] if cur is not None else -1
            p = pbs[i] if pbs is not None else -1
            if c >= 0:
                d.text((800, y), fmt(c / C), font=self.f["splitt"], fill=INK, anchor="ra", features=["tnum"])
                if show_delta and p >= 0:
                    dl = (c - p) / C
                    d.text((VW - 66, y), ("+" if dl >= 0 else "−") + fmt(abs(dl)), font=self.f["splitt"],
                           fill=RED if dl >= 0 else GREEN, anchor="ra", features=["tnum"])
            elif p >= 0:
                d.text((800, y), fmt(p / C), font=self.f["split"], fill=MUTED, anchor="ra", features=["tnum"])

    def histogram(self, d, done_times, pb):
        y0 = MAP_TOP + MH + 10
        x0, x1, hgt = 70, VW - 70, 150
        d.rounded_rectangle([40, y0 - 2, VW - 40, y0 + 232], 18, fill=(10, 16, 28, 210))
        d.text((66, y0 + 10), f"Finish times, {len(done_times)} attempts", font=self.f["split"], fill=INK, anchor="la")
        edges = np.logspace(np.log10(9), np.log10(900), 31)
        cnt = np.histogram(np.clip(done_times, 9, 899), edges)[0] if len(done_times) else np.zeros(30)
        bw = (x1 - x0) / 30
        top = max(cnt.max(), 8)
        for i, c in enumerate(cnt):
            if c:
                hh = hgt * c / top
                col = GOLD_T if edges[i + 1] <= pb * 1.0001 else (90, 150, 210)
                d.rectangle([x0 + i * bw + 2, y0 + 56 + hgt - hh, x0 + (i + 1) * bw - 2, y0 + 56 + hgt], fill=col)
        for v, lab in [(10, "10 s"), (60, "1 min"), (300, "5 min")]:
            xx = x0 + (np.log10(v) - np.log10(9)) / (np.log10(900) - np.log10(9)) * (x1 - x0)
            d.text((xx, y0 + 56 + hgt + 6), lab, font=self.f["small"], fill=MUTED, anchor="ma")

    # ---- frames
    def frame(self, t):
        seg = next(s for s in self.seg if s[0] <= t < s[1]) if t < self.total else self.seg[-1]
        t0, t1, kind, info = seg
        u = t - t0
        times = self.times
        if kind == "hook":
            img = self.map_img(dim=1.0)
            def draw(d):
                self.header(d)
                a = min(1.0, u / 0.4)
                d.rounded_rectangle([70, MAP_TOP + 330, VW - 70, MAP_TOP + 640], 26, fill=(4, 8, 16, int(215 * a)))
                d.text((VW // 2, MAP_TOP + 362), "A photon flips a coin", font=self.f["mid"], fill=INK + (int(255 * a),), anchor="ma")
                d.text((VW // 2, MAP_TOP + 420), "at every intersection.", font=self.f["mid"], fill=INK + (int(255 * a),), anchor="ma")
                d.text((VW // 2, MAP_TOP + 500), "How fast can it reach New York?", font=self.f["body"], fill=GOLD_T + (int(255 * a),), anchor="ma")
                d.text((VW // 2, MAP_TOP + 560), "Shortest possible drive: 0.0145 s", font=self.f["sub2"], fill=INK2 + (int(255 * a),), anchor="ma")
                self.splits_panel(d, None, None)
            return self.compose(img, draw)
        if kind == "run":
            seed, speed, att = info["seed"], info["speed"], info["attempt"]
            if self.run_seed != seed:
                self.run = self.W.new_run(seed); self.run_seed = seed
                self.pb_splits = self.pb[0]["splits"] if att <= 3 and att > 1 else (self.pb[181]["splits"] if att == 194 else None)
            r = self.run
            light = u * speed
            r["recent"] *= 0.45
            self.W.step(r, light * C)
            done = r["st"][3] == 1
            tt = r["km"][0] / C
            cur = int(r["st"][1])
            dot = (to_px(self.W.F["lon"][cur:cur + 1], self.W.F["lat"][cur:cur + 1]))
            dot = (dot[0][0], dot[1][0])
            img = self.map_img(heat2=r["heat"], recent2=r["recent"], dot=dot)
            pbv = None if att == 1 else (times[0] if att <= 3 else times[181])
            def draw(d):
                self.header(d)
                self.clock_block(d, att, tt, speed=speed, pb=pbv)
                self.splits_panel(d, r["splits"], self.pb_splits)
            return self.compose(img, draw)
        if kind == "result":
            att = info["attempt"]
            r = self.run
            if r["st"][3] != 1:                                 # finish the run exactly (last frame lands just short)
                self.W.step(r, 1e18)
            tt = r["km"][0] / C
            wr = info.get("wr", False)
            new_pb = att == 1 or wr
            img = self.map_img(heat2=r["heat"], flash=0.5 * max(0, 1 - u / 0.6) if new_pb else 0)
            pbv = times[0] if att <= 3 else times[193]
            def draw(d):
                self.header(d)
                self.clock_block(d, att, tt, pb=pbv, color=GOLD_T if new_pb else RED)
                self.splits_panel(d, r["splits"], None if att == 1 else self.pb_splits)
                a = min(1.0, u / 0.15)
                if new_pb:
                    lab = "NEW RECORD!" if wr else "FIRST RUN: NEW PB"
                    d.rounded_rectangle([90, MAP_TOP + 470, VW - 90, MAP_TOP + 590], 24, fill=(4, 8, 16, int(220 * a)))
                    d.text((VW // 2, MAP_TOP + 486), lab, font=self.f["big"], fill=GOLD_T + (int(255 * a),), anchor="ma")
                else:
                    d.rounded_rectangle([140, MAP_TOP + 480, VW - 140, MAP_TOP + 580], 24, fill=(4, 8, 16, int(220 * a)))
                    d.text((VW // 2, MAP_TOP + 496), f"No PB (+{fmt(tt - times[0])})", font=self.f["mid"], fill=RED + (int(255 * a),), anchor="ma")
            return self.compose(img, draw)
        if kind == "montage":
            sched = info["sched"]
            k = max(i for i, s in enumerate(sched) if s[0] <= t) if t >= sched[0][0] else 0
            tm, att, is_pb = sched[k]
            done_att = att                                      # attempts 1..att finished
            done_times = times[:done_att]
            pb_idx = int(np.argmin(done_times))
            pb_seed = pb_idx
            flash = 0.0
            if is_pb:
                flash = 0.6 * max(0.0, 1 - (t - tm) / 0.5)
            heat = self.pb[pb_seed]["heat"] if pb_seed in self.pb else None
            img = self.map_img(heat1=heat, flash=flash, dim=0.9)
            def draw(d):
                self.header(d)
                self.clock_block(d, att, times[att - 1], speed=None, pb=float(done_times.min()),
                                 color=GOLD_T if is_pb else (200, 205, 215))
                self.histogram(d, done_times, float(done_times.min()))
                if is_pb:
                    a = min(1.0, (t - tm) / 0.12)
                    d.rounded_rectangle([110, MAP_TOP + 470, VW - 110, MAP_TOP + 590], 24, fill=(4, 8, 16, int(220 * a)))
                    d.text((VW // 2, MAP_TOP + 486), f"NEW PB  {fmt(times[att - 1])}", font=self.f["big"],
                           fill=GOLD_T + (int(255 * a),), anchor="ma", features=["tnum"])
            return self.compose(img, draw)
        if kind == "wr_intro":
            img = self.map_img(heat1=self.pb[181]["heat"], dim=0.55)
            def draw(d):
                self.header(d)
                self.clock_block(d, 194, 0.0, speed=1, pb=times[181])
                self.splits_panel(d, None, self.pb[181]["splits"])
                a = min(1.0, u / 0.2)
                d.rounded_rectangle([90, MAP_TOP + 440, VW - 90, MAP_TOP + 640], 24, fill=(4, 8, 16, int(225 * a)))
                d.text((VW // 2, MAP_TOP + 460), "ATTEMPT 194", font=self.f["big"], fill=INK + (int(255 * a),), anchor="ma")
                d.text((VW // 2, MAP_TOP + 566), "in real time. Watch the clock.", font=self.f["body"], fill=GOLD_T + (int(255 * a),), anchor="ma")
            return self.compose(img, draw)
        # end card
        img = self.map_img(heat1=self.pb[193]["heat"], dim=0.45)
        sm = self.summary
        def draw(d):
            self.header(d)
            a = min(1.0, u / 0.4)
            d.rounded_rectangle([60, MAP_TOP + 200, VW - 60, MAP_TOP + 1010], 28, fill=(4, 8, 16, int(225 * a)))
            rows = [("RECORD", self.f["sub"], INK2, 0), (f"{fmt(times[193])} s", self.f["big"], GOLD_T, 46),
                    ("attempt 194 of 200", self.f["body"], INK, 150),
                    (f"Typical attempt: {fmt(sm['median_s'], 0)}", self.f["body"], INK, 270),
                    (f"Worst attempt: {fmt(sm['max_s'], 0)}", self.f["body"], INK, 326),
                    ("Following the GPS instead: 0.0145 s", self.f["body"], INK, 382),
                    (f"Every street in North America: {sm['network_km'] / 1e6:.1f} million km", self.f["sub2"], INK2, 500),
                    (f"{sm['junctions'] / 1e6:.0f} million intersections, one coin flip each", self.f["sub2"], INK2, 545),
                    ("Simulation · road data © OpenStreetMap contributors via Overture Maps", self.f["small"], MUTED, 650)]
            for txt, f_, col, dy in rows:
                d.text((VW // 2, MAP_TOP + 236 + dy), txt, font=f_, fill=col + (int(255 * a),), anchor="ma")
            self.splits_panel(d, self.pb[193]["splits"], self.pb[181]["splits"])
        return self.compose(img, draw)


def music(R, path):
    import score as SC_
    ev = []
    for t0, t1, kind, info in R.seg:
        ev.append((t0, kind, info))
    tot = R.total
    run_starts = [s[0] for s in R.seg if s[2] == "run"]
    mont = next(s for s in R.seg if s[2] == "montage")
    wr = next(s for s in R.seg if s[2] == "run" and s[3]["attempt"] == 194)
    pb_times = [tm for tm, a, p in mont[3]["sched"] if p]
    energy = [(0, 0.55), (run_starts[0], 0.8), (mont[0], 0.88), (mont[1], 0.88), (mont[1] + 0.01, 0.35),
              (wr[0], 0.95), (wr[1], 1.0), (wr[1] + 0.01, 0.0), (tot, 0.0)]
    impacts = [(run_starts[0], 0.9)] + [(t, 0.9) for t in pb_times] + [(wr[0], 1.1), (wr[1], 1.4)]
    risers = [(mont[1], wr[0]), (wr[1] - 3.0, wr[1])]
    sc = SC_.Score(tot, energy, impacts, risers, transpose_at=wr[0], final_at=wr[1])
    sc.compose()
    SC_.write(sc.mix(), path)


def render(still=None):
    os.makedirs(OUT, exist_ok=True)
    R = Renderer()
    print(f"timeline {R.total:.1f} s", flush=True)
    if still is not None:
        # play runs up to the still time for correct state
        for f in range(int(still * FPS) + 1):
            t = f / FPS
            seg = next(s for s in R.seg if s[0] <= t < s[1])
            if seg[2] == "run" or f == int(still * FPS):
                img = R.frame(t)
        Image.fromarray(img).save(os.path.join(OUT, "speedrun_still.png"))
        return
    vid = os.path.join(OUT, "_speedrun_video.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", vid],
                         stdin=subprocess.PIPE)
    n = int(round(R.total * FPS))
    for f in range(n):
        p.stdin.write(R.frame(f / FPS).tobytes())
        if f % 150 == 0:
            print(f"frame {f}/{n}", flush=True)
    p.stdin.close(); p.wait()
    wav = os.path.join(OUT, "speedrun_score.wav")
    music(R, wav)
    final = os.path.join(OUT, "speedrun.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    print("wrote", final)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        W = World(); prep_pb_runs(W)
    elif len(sys.argv) > 2 and sys.argv[1] == "still":
        render(float(sys.argv[2]))
    else:
        render()
