"""Vertical Short: the lost photon. One typical coin-flip walk from LA to New York, in real time.

    python lost_photon_render.py            -> out/lost_photon.mp4
    python lost_photon_render.py still 8    -> out/lost_photon_still.png (t = 8 s)
The walk shown is the one closest to the median of the 2,000 simulated in lost_photon.py.
"""
import os, sys, json, subprocess
import numpy as np
import cv2
import shapely
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lost_photon as LP  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
NE = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")
FD = "/usr/share/fonts/opentype/inter"
VW, VH, FPS = 1080, 1920, 30
S = 2                                   # supersample for the map layers
MAP_TOP, MAP_W = 236, 1060
LON0, LAT0 = -98.0, 42.0                # projection centre
HOOK, ARRIVE_HOLD, END = 4.2, 1.6, 4.6
EARTH_KM = 40_075.0
CAR_KMH = 100.0

INK = (240, 244, 250)
INK2 = (160, 178, 200)
MUTED = (110, 128, 150)
GOLD_T = (255, 186, 80)
CYAN = np.array([0.14, 0.66, 1.00], np.float32) * 0.42
GOLD = np.array([1.00, 0.50, 0.09], np.float32)
HOT = np.array([1.00, 0.93, 0.80], np.float32)


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def laea(lon, lat):
    """Lambert azimuthal equal-area, km."""
    lo, la = np.radians(np.asarray(lon) - LON0), np.radians(np.asarray(lat))
    p0 = np.radians(LAT0)
    k = np.sqrt(2 / (1 + np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo)))
    x = LP.R * k * np.cos(la) * np.sin(lo)
    y = LP.R * k * (np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo))
    return x, y


def pick_walk():
    d = json.load(open(os.path.join(DATA, "lost_photon_sim.json")))
    s = np.array([w["seconds"] for w in d["walks"]])
    i = int(np.argmin(np.abs(s - np.median(s))))
    return d["summary"], d["walks"][i]


def shortest_path(G):
    m = len(G["ea"])
    A = coo_matrix((np.r_[G["elen"], G["elen"]], (np.r_[G["ea"], G["eb"]], np.r_[G["eb"], G["ea"]])),
                   shape=(G["n"], G["n"])).tocsr()
    dist, pred = dijkstra(A, indices=G["start"], return_predecessors=True)
    goal = np.nonzero(G["goal"])[0]
    t = int(goal[np.argmin(dist[goal])])
    path = [t]
    while path[-1] != G["start"]:
        path.append(int(pred[path[-1]]))
    return np.array(path[::-1]), float(dist[t])


class Scene:
    def __init__(self):
        self.summary, self.walk = pick_walk()
        G = LP.load()
        self.G = G
        path = os.path.join(DATA, f"lost_photon_path_{self.walk['seed']}.npy")
        if not os.path.exists(path):
            LP.record(self.walk["seed"])
        self.path = np.load(path)
        # projection fit
        x, y = laea(G["lon"], G["lat"])
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        self.sc = (MAP_W - 20) / (x1 - x0)
        self.MW = MAP_W
        self.MH = int(np.ceil((y1 - y0) * self.sc)) + 20
        self.ox, self.oy = 10 - x0 * self.sc, 10 + y1 * self.sc
        self.px = (x * self.sc + self.ox)
        self.py = (-y * self.sc + self.oy)
        self.P2 = np.rint(np.c_[self.px, self.py] * S * 8).astype(np.int32)       # shift=3
        # roads (static), weighted by class
        cls_w = {0: (1.0, 3), 1: (0.85, 3), 2: (0.7, 2), 3: (0.55, 2), 4: (0.42, 2)}
        H2, W2 = self.MH * S, self.MW * S
        I = np.zeros((H2, W2), np.uint8)
        ecls = G["ecls"]
        for c in sorted(cls_w, key=lambda c: cls_w[c][0]):
            e = np.nonzero(ecls == c)[0]
            segs = np.stack([self.P2[G["ea"][e]], self.P2[G["eb"][e]]], 1)
            cv2.polylines(I, list(segs), False, int(255 * cls_w[c][0]), cls_w[c][1], cv2.LINE_AA, shift=3)
        self.I2 = I.astype(np.float32) / 255
        self.land = self._land()
        # walk timing (real time)
        cum = np.r_[0.0, np.cumsum(self._step_km())]
        self.tw = cum / LP.C_KMS                      # seconds at c for each path node
        self.walk_s = float(self.tw[-1])
        self.cum_km = cum
        # pass-through events
        names, place_of = LP.place_index(G)
        p = place_of[self.path]
        enter = np.nonzero((p[1:] >= 0) & (p[1:] != p[:-1]))[0] + 1
        self.place_names = names
        self.enter_t = self.tw[enter]
        self.enter_p = p[enter]
        totals = np.bincount(self.enter_p, minlength=len(names))
        show = [i for i in np.argsort(-totals) if totals[i] > 0 and names[i] not in ("Los Angeles",)][:6]
        if names.index("Calgary") not in show and totals[names.index("Calgary")] > 0:
            show[-1] = names.index("Calgary")
        self.show = sorted(show, key=lambda i: -totals[i])
        self.totals = totals
        # shortest route for the hook
        self.sp, self.sp_km = shortest_path(G)
        self.H = np.zeros((H2, W2), np.float32)       # heat: frames with a visit, per pixel
        self.wake = np.zeros((H2, W2), np.float32)
        self.i_done = 0
        k = np.arange(121, dtype=np.float32) - 60
        r2 = k[None] ** 2 + k[:, None] ** 2
        self.dot = (np.exp(-r2 / (2 * 1.8 ** 2))[..., None] * np.array([1, .97, .92]) * 3.0 +
                    np.exp(-r2 / (2 * 5 ** 2))[..., None] * np.array([1, .75, .35]) * 1.4 +
                    np.exp(-r2 / (2 * 18 ** 2))[..., None] * np.array([1, .55, .15]) * 0.5).astype(np.float32)
        self.stats = dict(seed=self.walk["seed"], walk_s=self.walk_s, walk_km=float(cum[-1]),
                          steps=int(len(self.path) - 1), shortest_km=self.sp_km,
                          shortest_ms=self.sp_km / LP.C_KMS * 1000,
                          times_slower=self.walk_s / (self.sp_km / LP.C_KMS),
                          around_earth=float(cum[-1] / EARTH_KM),
                          car_years=float(cum[-1] / CAR_KMH / 24 / 365.25),
                          passes={names[i]: int(totals[i]) for i in range(len(names))},
                          sim=self.summary)

    def _step_km(self):
        G = self.G
        a, b = self.path[:-1], self.path[1:]
        return LP.gc_km(G["lon"][a], G["lat"][a], G["lon"][b], G["lat"][b])

    def _land(self):
        import json as js
        gj = js.load(open(NE))
        box = shapely.box(-180, -5, -20, 85)
        H2, W2 = self.MH * S, self.MW * S
        m = np.zeros((H2, W2), np.uint8)
        for f in gj["features"]:
            geom = shapely.geometry.shape(f["geometry"])
            if not geom.intersects(box):
                continue
            geom = geom.intersection(box)
            for poly in getattr(geom, "geoms", [geom]):
                if poly.geom_type != "Polygon":
                    continue
                for ring in [poly.exterior] + list(poly.interiors):
                    c = np.asarray(ring.coords)
                    x, y = laea(c[:, 0], c[:, 1])
                    pts = np.rint(np.c_[x * self.sc + self.ox, -y * self.sc + self.oy] * S * 8).astype(np.int32)
                    cv2.fillPoly(m, [pts], 255 if ring is poly.exterior else 0, cv2.LINE_AA, shift=3)
        land = cv2.resize(m, (self.MW, self.MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        edge = cv2.morphologyEx((land > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
        return land, edge

    def advance(self, t):
        """Accumulate the walk up to walk-time t (seconds at c)."""
        i1 = int(np.searchsorted(self.tw, t, side="right")) - 1
        self.wake *= 0.5
        if i1 > self.i_done:
            seg = self.P2[self.path[self.i_done:i1 + 1]]
            mask = np.zeros(self.H.shape, np.uint8)
            cv2.polylines(mask, [seg], False, 1, 3, cv2.LINE_8, shift=3)
            mf = mask.astype(np.float32)
            self.H += mf
            self.wake = np.maximum(self.wake, mf)
            self.i_done = i1
        return i1

    def map_frame(self, i_dot, gain=1.0, extra=None, flash=0.0):
        I2 = self.I2
        heat = self.H
        g = np.clip(np.log1p(heat) / np.log1p(30), 0, 1)
        lit = heat > 0
        gold2 = I2 * np.where(lit, 0.32 + 0.5 * g, 0)
        cyan2 = I2 * (~lit)
        hot2 = I2 * self.wake * 1.4 + (self.wake > 0) * 0.25 * self.wake
        dn = lambda a: cv2.resize(a, (self.MW, self.MH), interpolation=cv2.INTER_AREA)
        land, coast = self.land
        E = (dn(cyan2)[..., None] * CYAN + dn(gold2)[..., None] * GOLD + dn(hot2)[..., None] * HOT)
        E += land[..., None] * np.array([0.020, 0.032, 0.050], np.float32) + coast[..., None] * np.array([.03, .05, .08], np.float32)
        if extra is not None:
            E += extra
        E *= gain
        if i_dot is not None:
            n = self.path[i_dot]
            self._sprite(E, self.px[n], self.py[n], 1.0 + flash)
        small = cv2.resize(E, (self.MW // 2, self.MH // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2) * 0.55 + cv2.GaussianBlur(small, (0, 0), 8) * 0.45,
                           (self.MW, self.MH))
        return 1 - np.exp(-(E + bloom) * 1.2)

    def _sprite(self, E, x, y, amp):
        xi, yi, h = int(round(x)), int(round(y)), 60
        ys0, ys1, xs0, xs1 = max(0, yi - h), min(self.MH, yi + h + 1), max(0, xi - h), min(self.MW, xi + h + 1)
        if ys1 > ys0 and xs1 > xs0:
            E[ys0:ys1, xs0:xs1] += amp * self.dot[ys0 - (yi - h):ys1 - (yi - h), xs0 - (xi - h):xs1 - (xi - h)]


def text_centered(d, y, txt, f, col, a=255):
    d.text((VW // 2, y), txt, font=f, fill=tuple(col) + (int(a),), anchor="ma")


def fmt_clock(t):
    return f"{t:6.3f} s".strip()


def render(still_t=None):
    sc = Scene()
    os.makedirs(OUT, exist_ok=True)
    json.dump(sc.stats, open(os.path.join(DATA, "lost_photon_video_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in sc.stats.items() if k != "sim"}, indent=1), flush=True)
    W_S = sc.walk_s
    total = HOOK + W_S + ARRIVE_HOLD + END
    n_frames = int(round(total * FPS))
    mx0 = (VW - sc.MW) // 2
    mb = MAP_TOP + sc.MH                       # map bottom
    names = sc.place_names
    f_title = font("Inter-Bold.otf", 64)
    f_sub = font("Inter-SemiBold.otf", 46)
    f_clock = font("InterDisplay-Bold.otf", 118)
    f_mid = font("Inter-Medium.otf", 36)
    f_small = font("Inter-Medium.otf", 30)
    f_city = font("Inter-SemiBold.otf", 22)
    f_foot = font("Inter-Regular.otf", 22)
    # shortest-route raster for the hook
    spx = sc.P2[sc.sp]
    sp_mask = np.zeros(sc.H.shape, np.uint8)
    cv2.polylines(sp_mask, [spx], False, 255, 5, cv2.LINE_AA, shift=3)
    sp_small = cv2.resize(sp_mask, (sc.MW, sc.MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sp_cum = np.r_[0, np.cumsum(np.hypot(*np.diff(np.c_[sc.px[sc.sp], sc.py[sc.sp]], axis=0).T))]
    sp_frac = np.zeros((sc.MH, sc.MW), np.float32)
    # each map pixel of the route gets the fraction of the route at which it is reached
    lab = np.full(sc.H.shape, np.inf, np.float32)
    for k in range(len(sc.sp) - 1, 0, -1):
        cv2.line(lab, tuple(spx[k - 1]), tuple(spx[k]), float(sp_cum[k] / sp_cum[-1]), 7, cv2.LINE_8, shift=3)
    lab = cv2.resize(np.where(np.isfinite(lab), lab, 9).astype(np.float32), (sc.MW, sc.MH), interpolation=cv2.INTER_NEAREST)
    la_xy = (sc.px[sc.G["start"]], sc.py[sc.G["start"]])
    ny_n = sc.sp[-1]
    ny_xy = (sc.px[ny_n], sc.py[ny_n])
    city_xy = {nm: laea(*LP.PLACES[nm]) for nm in names}
    city_xy = {nm: (x * sc.sc + sc.ox, -y * sc.sc + sc.oy) for nm, (x, y) in city_xy.items()}
    st = sc.stats
    global CL_X, CL_Y
    cx, cy = laea(-121.0, 17.0)                       # Pacific, off Mexico
    CL_X, CL_Y = 40, int(MAP_TOP - cy * sc.sc + sc.oy) - 40
    path = os.path.join(OUT, "lost_photon.mp4")
    if still_t is None:
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                              "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                             stdin=subprocess.PIPE)
    frames = range(n_frames) if still_t is None else [int(round(still_t * FPS))]
    if still_t is not None:                        # replay frame by frame so the wake is right
        for ff in range(int(round(still_t * FPS))):
            sc.advance(float(np.clip(ff / FPS - HOOK, 0, sc.walk_s)))
    last_flash = {}
    for f in frames:
        tf = f / FPS
        tw = float(np.clip(tf - HOOK, 0, W_S))
        phase = "hook" if tf < HOOK else ("walk" if tf < HOOK + W_S else ("arrive" if tf < HOOK + W_S + ARRIVE_HOLD else "end"))
        extra = None
        i_dot = None
        flash = 0.0
        if phase == "hook":
            u = np.clip((tf - 1.2) / 0.9, 0, 1)            # direct route drawn 1.2..2.1 s
            fade = 1.0 if tf < 3.4 else max(0.0, 1 - (tf - 3.4) / 0.6)
            vis = (lab <= u).astype(np.float32) * sp_small * fade
            extra = vis[..., None] * (GOLD * 1.2 + HOT * 0.6)
            i_dot = 0
        else:
            i_dot = sc.advance(tw)
            if phase != "walk":
                i_dot = len(sc.path) - 1
                flash = max(0.0, 3.0 * (1 - (tf - HOOK - W_S) / 0.8))
        gain = 1.0
        if phase == "end":
            gain = 1 - 0.55 * min(1.0, (tf - HOOK - W_S - ARRIVE_HOLD) / 0.5)
        img = sc.map_frame(i_dot, gain=gain, extra=extra, flash=flash)
        canvas = np.empty((VH, VW, 3), np.float32); canvas[:] = np.array([4, 7, 13], np.float32) / 255
        canvas[MAP_TOP:mb, mx0:mx0 + sc.MW] = np.maximum(img, canvas[MAP_TOP:mb, mx0:mx0 + sc.MW])
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        # map labels
        for nm, (x, y) in [("Los Angeles", la_xy), ("New York", ny_xy)]:
            d.ellipse([mx0 + x - 7, MAP_TOP + y - 7, mx0 + x + 7, MAP_TOP + y + 7], outline=INK + (230,), width=2)
            if nm == "Los Angeles":
                d.text((mx0 + x - 14, MAP_TOP + y + 8), nm, font=f_city, fill=INK + (235,), anchor="ra")
            else:
                d.text((mx0 + x + 14, MAP_TOP + y + 6), nm, font=f_city, fill=INK + (235,), anchor="la")
        # counters for pass-through cities
        seen_now = np.bincount(sc.enter_p[sc.enter_t <= tw], minlength=len(names)) if phase != "hook" else np.zeros(len(names), int)
        for i in sc.show:
            nm = names[i]
            x, y = city_xy[nm]
            recent = phase == "walk" and np.any((sc.enter_p == i) & (sc.enter_t <= tw) & (sc.enter_t > tw - 0.12))
            r = 9 if recent else 4
            d.ellipse([mx0 + x - r, MAP_TOP + y - r, mx0 + x + r, MAP_TOP + y + r],
                      outline=(GOLD_T if recent else INK2) + (255 if recent else 120,), width=2)
        # top text
        if phase == "hook":
            text_centered(d, 56, "Los Angeles to New York", f_title, INK)
            text_centered(d, 134, "at the speed of light", f_title, INK)
            if tf > 1.2:
                a = min(1.0, (tf - 1.2) / 0.3) * 255
                d.text((CL_X, CL_Y - 60), "Shortest drive:", font=f_mid, fill=GOLD_T + (int(a),), anchor="la")
                d.text((CL_X, CL_Y - 10), f"{st['shortest_ms']:.0f} milliseconds", font=font("Inter-Bold.otf", 64),
                       fill=GOLD_T + (int(a),), anchor="la")
            if tf > 2.5:
                a = min(1.0, (tf - 2.5) / 0.3) * 255
                text_centered(d, mb + 40, "Now flip a coin at every intersection.", f_sub, INK, a)
        else:
            text_centered(d, 56, "A photon that flips a coin", f_title, INK)
            text_centered(d, 134, "at every intersection", f_title, INK)
        # clock + counters
        if phase != "hook":
            km = float(np.interp(tw, sc.tw, sc.cum_km))
            if phase == "walk":
                d.rounded_rectangle([CL_X, CL_Y - 62, CL_X + 196, CL_Y - 14], 24, fill=(220, 40, 40, 235))
                d.text((CL_X + 98, CL_Y - 38), "REAL TIME", font=font("Inter-Bold.otf", 28), fill=(255, 255, 255, 255), anchor="mm")
            d.text((CL_X - 6, CL_Y), f"{tw:.3f} s", font=f_clock, fill=(250, 238, 212, 255), anchor="la")
            d.text((CL_X, CL_Y + 150), f"{km:,.0f} km driven", font=f_small, fill=INK2 + (255,), anchor="la")
            d.text((CL_X, CL_Y + 192), f"{km / EARTH_KM:,.0f}\u00d7 around the Earth", font=f_small, fill=INK2 + (255,), anchor="la")
            y0 = mb + 14
            text_centered(d, y0, "Drove through", f_small, MUTED)
            fb = font("Inter-Bold.otf", 30)
            for k, i in enumerate(sc.show):
                ci, ri = k % 3, k // 3
                xc = 180 + ci * 360
                y = y0 + 46 + ri * 46
                a_txt, b_txt = names[i] + "  ", f"{seen_now[i]}\u00d7"
                wa, wb = d.textlength(a_txt, font=f_small), d.textlength(f"{sc.totals[i]}\u00d7", font=fb)
                x0 = xc - (wa + wb) / 2
                d.text((x0, y), a_txt, font=f_small, fill=INK + (255,), anchor="la")
                d.text((x0 + wa, y), b_txt, font=fb, fill=GOLD_T + (255,), anchor="la")
        if phase in ("arrive", "end"):
            u = min(1.0, (tf - HOOK - W_S) / 0.35)
            d.rounded_rectangle([90, 300, VW - 90, 400], 20, fill=(4, 8, 16, int(200 * u)))
            text_centered(d, 318, "Arrived in New York", font("Inter-Bold.otf", 52), GOLD_T, 255 * u)
        if phase == "end":
            u = min(1.0, (tf - HOOK - W_S - ARRIVE_HOLD) / 0.5)
            box = (60, MAP_TOP + 60, VW - 60, MAP_TOP + 60 + 690)
            d.rounded_rectangle(box, 28, fill=(6, 10, 18, int(225 * u)))
            lines = [(f"{st['walk_s']:.1f} seconds", font("InterDisplay-Bold.otf", 104), GOLD_T, 0),
                     (f"{st['times_slower']:,.0f}× longer than the", f_sub, INK, 150),
                     (f"shortest drive ({st['shortest_ms']:.0f} ms)", f_sub, INK, 210),
                     (f"In a car at {CAR_KMH:.0f} km/h, nonstop:", f_mid, INK2, 320),
                     (f"{st['car_years']:,.1f} years", font("Inter-Bold.otf", 64), INK, 372),
                     (f"We sent {st['sim']['n']:,} lost photons.", f_mid, INK2, 492),
                     (f"Luckiest: {st['sim']['min_s']:.1f} s   Unluckiest: {st['sim']['max_s'] / 60:.0f} min", f_mid, INK, 542),
                     ("This one was typical (the median).", f_small, MUTED, 610)]
            for txt, fnt, col, dy in lines:
                text_centered(d, box[1] + 40 + dy, txt, fnt, col, 255 * u)
        d.text((VW // 2, VH - 92), "Highways and main roads of North and Central America (2.2 million km)",
               font=f_foot, fill=MUTED + (255,), anchor="ma")
        d.text((VW // 2, VH - 62), "Simulation · Road data © OpenStreetMap contributors via Overture Maps",
               font=f_foot, fill=MUTED + (255,), anchor="ma")
        a = np.asarray(ov).astype(np.float32) / 255
        canvas = canvas * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
        frame = (np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)
        if still_t is not None:
            Image.fromarray(frame).save(os.path.join(OUT, "lost_photon_still.png"))
            return
        p.stdin.write(frame.tobytes())
        if f % 60 == 0:
            print(f"frame {f}/{n_frames}", flush=True)
    p.stdin.close(); p.wait()
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        render(float(sys.argv[2]))
    else:
        render()
