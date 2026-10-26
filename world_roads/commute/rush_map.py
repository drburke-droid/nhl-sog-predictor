"""Rush hour across North America, a map in the light-speed film's look:  python rush_map.py [still HOUR ...]
-> ../media/commute/rush_map.mp4 (1920x1080, 30 fps), or rush_map_<HOUR>.png stills

The clock runs from 5 a.m. to midnight, 4 s an hour, and it is the same local time in every metro, as if North
America had one time zone. Each metro's highways glow by its city-wide congestion at that local time: cyan while the
roads are free, gold as they fill, white-hot at the worst. Congestion is the TomTom Traffic Index's extra travel time
over empty roads, metro area, Tue-Thu averages of 2025 (tomtom_weekhour.json, tomtom.py). Each hourly value sits at
the middle of its hour, joined by a monotone cubic (PCHIP), so the light never overshoots the data.

Base map: Natural Earth 10 m land, lakes, borders and roads (../data/ne, as ../run_all.sh fetches), in a Lambert
conformal conic projection centred on the metros. Look: the palette, background, bloom and tone curve of ../render.py
and ../na_wide.py.
"""
import os, sys, json
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(1, ROOT)
from render import CYAN, GOLD, HOT, FPS, ffmpeg_proc_size
from metros import METROS

NE = os.path.join(ROOT, "data", "ne")
MEDIA = os.path.join(ROOT, "media", "commute")
CACHE = os.path.join(HERE, "data", "rush_map_layers.npz")
FD = "/usr/share/fonts/opentype/inter"
OW, OH, SS = 1920, 1080, 2                              # frame, and the supersample lines and land are drawn at
H0, H1, SEC_PER_HOUR, HOLD0, HOLD1 = 5.0, 24.0, 4.0, 1.5, 2.5
NFRAMES = int(round((HOLD0 + (H1 - H0) * SEC_PER_HOUR + HOLD1) * FPS))
OCEAN = np.array([0.008, 0.016, 0.035], np.float32)     # ../render.py's background
LANDC = np.array([0.030, 0.050, 0.088], np.float32)
ROAD_W = {"Major Highway": 1.0, "Beltway": 1.0, "Bypass": 0.9, "Secondary Highway": 0.72, "Road": 0.5}
SIG_ROAD = 30.0                                         # px (about 140 km): how far out a metro's highways light up
MIDWEEK = ["TUESDAY", "WEDNESDAY", "THURSDAY"]
# label placement, (dx, dy, anchor) from the office; crowded places set by hand
LABEL = {"Los Angeles": (-16, 0, "rm"), "Inland Empire": (6, 22, "mt"), "San Francisco Bay Area": (-16, 0, "rm"),
         "Seattle": (-16, 4, "rm"), "Vancouver": (-16, -4, "rm"), "Philadelphia": (14, 8, "lm"), "New York": (14, -2, "lm"),
         "Boston": (14, 0, "lm"), "Washington": (12, 14, "lm"), "Montreal": (12, -8, "lm"), "Toronto": (-10, -14, "rm"),
         "Detroit": (0, 16, "mt"), "Chicago": (-16, 0, "rm"), "Houston": (14, 4, "lm"), "Dallas–Fort Worth": (-16, 0, "rm"),
         "Monterrey": (16, 0, "lm"), "Guadalajara": (-16, 0, "rm"), "Mexico City": (16, 6, "lm"), "Miami": (14, 0, "lm"),
         "Atlanta": (16, 0, "lm"), "Phoenix": (14, 0, "lm"), "Calgary": (-14, 0, "rm"), "Edmonton": (14, 0, "lm")}
SHORT = {"San Francisco Bay Area": "Bay Area"}


# ------------------------------------------------------------------ projection: Lambert conformal conic
PHI1, PHI2, PHI0, LAM0, R = map(np.radians, (20.0, 50.0, 37.0, -97.0, 1.0))
N_ = np.log(np.cos(PHI1) / np.cos(PHI2)) / np.log(np.tan(np.pi / 4 + PHI2 / 2) / np.tan(np.pi / 4 + PHI1 / 2))
F_ = np.cos(PHI1) * np.tan(np.pi / 4 + PHI1 / 2) ** N_ / N_
RHO0 = F_ / np.tan(np.pi / 4 + PHI0 / 2) ** N_


def lcc(lon, lat):
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.clip(np.asarray(lat, float), -80, 89.5))
    rho = F_ / np.tan(np.pi / 4 + phi / 2) ** N_
    t = N_ * (((lam - LAM0 + np.pi) % (2 * np.pi)) - np.pi)
    return rho * np.sin(t), RHO0 - rho * np.cos(t)


def fit():
    """scale and origin that put the metros between y = 190 and 960, centred on x = 860"""
    x, y = lcc([m[4][2] for m in METROS], [m[4][1] for m in METROS])
    s = (960 - 190) / (y.max() - y.min())
    return s, 860 - s * (x.max() + x.min()) / 2, 190 + s * y.max()


S_, X0_, Y0_ = fit()


def px(lon, lat):
    x, y = lcc(lon, lat)
    return X0_ + S_ * x, Y0_ - S_ * y


# ------------------------------------------------------------------ base map
def rings(name, keep=lambda p: True):
    """projected rings of a Natural Earth layer, on the supersampled canvas; rings across the projection's seam or
    far outside the frame are dropped"""
    for f in json.load(open(os.path.join(NE, name + ".geojson")))["features"]:
        g, p = f["geometry"], f["properties"]
        if g is None or not keep(p):
            continue
        polys = {"Polygon": [g["coordinates"]], "MultiPolygon": g["coordinates"],
                 "LineString": [[g["coordinates"]]], "MultiLineString": [[c] for c in g["coordinates"]]}[g["type"]]
        for poly in polys:
            out = []
            for ring in poly:
                a = np.asarray(ring, float)
                x, y = px(a[:, 0], a[:, 1])
                if len(x) > 1 and np.abs(np.diff(x)).max() > OW:
                    continue
                if x.max() < -200 or x.min() > OW + 200 or y.max() < -200 or y.min() > OH + 200:
                    continue
                out.append(np.round(np.c_[x, y] * SS * 16).astype(np.int32))
            if out:
                yield p, out


def build():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    big = (OH * SS, OW * SS)
    land = np.zeros(big, np.uint8)
    for _, rs in rings("ne_10m_land"):
        cv2.fillPoly(land, rs, 255, cv2.LINE_AA, shift=4)
    for _, rs in rings("ne_10m_lakes"):
        cv2.fillPoly(land, rs, 0, cv2.LINE_AA, shift=4)

    def lines(name, keep, weight, thick):
        img = np.zeros(big, np.float32)
        for p, rs in rings(name, keep):
            cv2.polylines(img, rs, False, float(weight(p)), thick, cv2.LINE_AA, shift=4)
        return cv2.resize(img, (OW, OH), interpolation=cv2.INTER_AREA)

    roads = lines("ne_10m_roads", lambda p: p.get("type") in ROAD_W, lambda p: ROAD_W[p["type"]], 2)
    borders = lines("ne_10m_admin_0_boundary_lines_land", lambda p: True, lambda p: 1.0, 2)
    states = lines("ne_50m_admin_1_states_provinces_lines", lambda p: True, lambda p: 1.0, 1)
    land1 = cv2.resize(land, (OW, OH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    coast = cv2.morphologyEx(land, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    coast = cv2.resize(coast, (OW, OH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    vign = 1 - 0.45 * (((xx - OW / 2) / (OW * 0.62)) ** 2 + ((yy - OH * 0.52) / (OH * 0.75)) ** 2)
    bg = OCEAN[None, None] * np.clip(vign, 0.35, 1)[..., None] + land1[..., None] * (LANDC - OCEAN)[None, None]
    bg += coast[..., None] * np.array([0.035, 0.07, 0.12], np.float32)
    bg += cv2.GaussianBlur(coast, (0, 0), 6)[..., None] * np.array([0.01, 0.025, 0.05], np.float32)
    bg += borders[..., None] * np.array([0.05, 0.07, 0.10], np.float32) + states[..., None] * np.array([0.018, 0.026, 0.04], np.float32)
    L = dict(bg=bg.astype(np.float32), roads=roads.astype(np.float32))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **L)
    return L


# ------------------------------------------------------------------ traffic
def congestion():
    """{rank: f(hour) -> % extra travel time}, Tue-Thu, from the hourly values at the middle of each hour"""
    W = json.load(open(os.path.join(MEDIA, "tomtom_weekhour.json")))["cities"]
    out = {}
    for rank, name, cc, tz, office, key in METROS:
        s = W[key]["areas"]["metro"]
        p = np.array([np.mean([s[f"{d}-{h}"][2] for d in MIDWEEK]) for h in range(24)])
        x = np.arange(-3, 27) + 0.5
        out[rank] = PchipInterpolator(x, p[np.arange(-3, 27) % 24])
    return out


def ramp(a):
    """colour for a congestion level a (1 = 100 % extra time): cyan, gold, white-hot"""
    stops = [(0.0, CYAN * 0.7), (0.15, CYAN), (0.45, GOLD), (0.7, GOLD), (1.0, HOT)]
    a = float(np.clip(a, 0, 1.0))
    for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
        if a <= a1:
            u = (a - a0) / (a1 - a0)
            return c0 * (1 - u) + c1 * u
    return HOT


def amp(a):
    return float(np.clip((a - 0.05) / 0.95, 0, 1.1)) ** 1.15


def hour_of(f):
    return H0 + np.clip(f / FPS - HOLD0, 0, (H1 - H0) * SEC_PER_HOUR) / SEC_PER_HOUR


def clock(h):
    m = int(h * 60 + 1e-6) // 5 * 5
    hh, mm = divmod(m, 60)
    return f"{(hh - 1) % 12 + 1}:{mm:02d} " + ("a.m." if hh % 24 < 12 else "p.m.")


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def tone(c):
    return tuple(int(v) for v in np.clip((1 - np.exp(-np.asarray(c) * 1.6)) * 255, 0, 255))


class Map:
    def __init__(self):
        L = build()
        self.bg, roads = L["bg"], L["roads"]
        self.base = roads[..., None] * CYAN * 0.30
        self.cong = congestion()
        self.cities = []
        r = int(3 * SIG_ROAD)
        for rank, name, cc, tz, (o, lat, lon), key in METROS:
            x, y = px(lon, lat)
            xi, yi = int(round(x)), int(round(y))
            y0, y1, x0, x1 = max(yi - r, 0), min(yi + r + 1, OH), max(xi - r, 0), min(xi + r + 1, OW)
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d2 = (gx - x) ** 2 + (gy - y) ** 2
            self.cities.append(dict(rank=rank, name=name, x=x, y=y, box=(y0, y1, x0, x1), d2=d2,
                                    road=roads[y0:y1, x0:x1] * np.exp(-d2 / (2 * SIG_ROAD ** 2))))
        hours = np.linspace(H0, H1, 400)
        self.avg_h = hours
        self.avg = np.mean([self.cong[c["rank"]](hours) for c in self.cities], axis=0)
        self.F = dict(title=font("Inter-SemiBold.otf", 46), sub=font("Inter-Regular.otf", 25), small=font("Inter-Medium.otf", 15),
                      clock=font("InterDisplay-Medium.otf", 66), note=font("Inter-Regular.otf", 17), foot=font("Inter-Regular.otf", 14),
                      label=font("Inter-Medium.otf", 15), row=font("Inter-Medium.otf", 21), val=font("InterDisplay-Medium.otf", 21),
                      tick=font("Inter-Regular.otf", 13))

    def frame(self, h):
        E = self.base.copy()
        levels = {}
        for c in self.cities:
            a = max(float(self.cong[c["rank"]](h)), 0.0) / 100
            levels[c["name"]] = a
            col, k = ramp(a), amp(a)
            y0, y1, x0, x1 = c["box"]
            sig = 3.0 + 8.0 * k
            w = c["road"] * (0.10 + 1.35 * k) + np.exp(-c["d2"] / (2 * sig ** 2)) * (0.10 + 0.75 * k)
            E[y0:y1, x0:x1] += w[..., None] * col
            E[y0:y1, x0:x1] += (np.exp(-c["d2"] / (2 * 1.4 ** 2)) * 0.6)[..., None] * np.array([0.8, 0.9, 1.0], np.float32)
        small = cv2.resize(E, (OW // 2, OH // 2), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 3.0)
        tiny = cv2.resize(small, (OW // 8, OH // 8), interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (OW // 2, OH // 2))
        bloom = cv2.resize(cv2.addWeighted(b1, 0.35, b2, 0.45, 0), (OW, OH))
        img = 1 - np.exp(-(self.bg + E + bloom) * 1.15)
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        self.hud(im, h, levels)
        return np.asarray(im)

    def hud(self, im, h, levels):
        F = self.F
        ov = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for c in self.cities:
            dx, dy, anchor = LABEL[c["name"]]
            k = amp(levels[c["name"]])
            al = int(120 + 135 * min(k * 1.6, 1))
            d.text((c["x"] + dx, c["y"] + dy), SHORT.get(c["name"], c["name"]), font=F["label"], anchor=anchor,
                   fill=(206, 218, 236, al), stroke_width=3, stroke_fill=(2, 5, 12, int(al * 0.8)))
        d.text((58, 40), "Rush hour across North America", font=F["title"], fill=(236, 242, 250, 255))
        d.text((60, 100), "A typical Tuesday to Thursday in 23 metros, each on its own local clock", font=F["sub"],
               fill=(150, 176, 205, 255))
        # worst traffic now
        x0, y0 = 1560, 560
        d.text((x0, y0), "WORST TRAFFIC NOW", font=F["small"], fill=(130, 150, 175, 255))
        d.text((x0 + 300, y0), "EXTRA TIME", font=F["small"], fill=(130, 150, 175, 255), anchor="ra")
        for i, (name, a) in enumerate(sorted(levels.items(), key=lambda kv: -kv[1])[:5]):
            y = y0 + 32 + i * 42
            d.text((x0, y), SHORT.get(name, name), font=F["row"], fill=(236, 242, 250, 255))
            d.text((x0 + 300, y), f"+{a * 100:.0f}%", font=F["val"], fill=(246, 236, 214, 255), anchor="ra")
            d.rectangle((x0, y + 28, x0 + int(300 * min(a, 1.05) / 1.05), y + 31), fill=tone(ramp(a) * 1.4) + (255,))
        # clock
        right = OW - 60
        d.text((right, 896), "LOCAL TIME, EVERY CITY", font=F["small"], fill=(130, 150, 175, 255), anchor="ra")
        d.text((right, 912), clock(h), font=F["clock"], fill=(246, 236, 214, 255), anchor="ra")
        d.text((right, 1000), "as if North America had one time zone", font=F["note"], fill=(160, 178, 200, 255), anchor="ra")
        # the day so far: average congestion of the 23 metros
        tx0, tx1, ty0, ty1 = 60, 600, 930, 1005
        xs = tx0 + (self.avg_h - H0) / (H1 - H0) * (tx1 - tx0)
        ys = ty1 - self.avg / 80 * (ty1 - ty0)
        d.line(list(zip(xs, ys)), fill=(70, 95, 125, 255), width=2)
        k = int(np.searchsorted(self.avg_h, h, "right"))
        if k > 1:
            d.line(list(zip(xs[:k], ys[:k])), fill=tone(GOLD * 1.5) + (255,), width=3)
        d.line([(tx0, ty1), (tx1, ty1)], fill=(60, 80, 105, 255), width=1)
        for hh, lab in ((6, "6 a.m."), (9, "9"), (12, "noon"), (15, "3"), (18, "6 p.m."), (21, "9"), (24, "midnight")):
            x = tx0 + (hh - H0) / (H1 - H0) * (tx1 - tx0)
            d.line([(x, ty1), (x, ty1 + 4)], fill=(90, 110, 135, 255), width=1)
            d.text((x, ty1 + 6), lab, font=F["tick"], fill=(120, 140, 165, 255), anchor="ma")
        px_, py_ = tx0 + (h - H0) / (H1 - H0) * (tx1 - tx0), ty1 - float(np.interp(h, self.avg_h, self.avg)) / 80 * (ty1 - ty0)
        d.ellipse((px_ - 5, py_ - 5, px_ + 5, py_ + 5), fill=(255, 240, 215, 255))
        d.text((tx0, ty0 - 22), "AVERAGE OF THE 23 METROS", font=F["small"], fill=(130, 150, 175, 255))
        d.text((60, 1046), "Congestion: TomTom Traffic Index, extra travel time over empty roads, metro areas, Tue-Thu "
               "averages of 2025 · Map: Natural Earth", font=F["foot"], fill=(120, 140, 165, 255))
        im.paste(ov, (0, 0), ov)


M = None


def init():
    global M
    M = Map()


def render(f):
    return M.frame(float(hour_of(f))).tobytes()


def main():
    build()
    os.makedirs(MEDIA, exist_ok=True)
    path = os.path.join(MEDIA, "rush_map.mp4")
    p = ffmpeg_proc_size(path, OW, OH)
    with Pool(os.cpu_count(), initializer=init) as pool:
        for i, fr in enumerate(pool.imap(render, range(NFRAMES), chunksize=8)):
            p.stdin.write(fr)
            if i % 300 == 0:
                print(f"  frame {i}/{NFRAMES}", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", path, NFRAMES, "frames")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        init()
        for s in sys.argv[2:]:
            dst = os.path.join(MEDIA, f"rush_map_{s.replace('.', 'h')}.png")
            cv2.imwrite(dst, M.frame(float(s))[..., ::-1])
            print("still", dst)
    else:
        main()
