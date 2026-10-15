"""Vertical Short: signal fires from Plymouth around the world, until the North Atlantic stops them,
then the fire ships that close the loop.

    python render_fires.py still 20    -> out/fires_still.png
    python render_fires.py             -> out/signal_fires.mp4 (with music)
"""
import os, sys, json, math, subprocess
import numpy as np
import cv2
import shapely
from shapely.strtree import STRtree
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
FD = "/usr/share/fonts/opentype/inter"
VW, VH, FPS = 1080, 1920, 30
GCY = 1010                              # globe centre on screen
MIN_PER_RELAY = 5.0
INK, INK2, MUTED = (240, 244, 250), (160, 178, 200), (110, 128, 150)
GOLD_T, RED = (255, 186, 80), (255, 110, 90)
FIRE = np.array([1.0, 0.55, 0.15], np.float32)
CORE = np.array([1.0, 0.92, 0.75], np.float32)
SHIP = np.array([1.0, 0.80, 0.45], np.float32)
R_EARTH = 6371.0088


PEAKS = {"Muztagh Ata, in China's Pamir mountains": (38.276, 75.116), "Ismoil Somoni Peak, Tajikistan": (38.942, 72.019),
         "Mont Blanc": (45.833, 6.865), "Mount Ararat": (39.702, 44.298), "Mount Foraker": (62.961, -151.399),
         "Mount Blackburn": (61.731, -143.433), "Denali": (63.069, -151.007), "Mount Logan": (60.567, -140.405),
         "Mount Saint Elias": (60.292, -140.931), "Mount Bona": (61.385, -141.749), "Kongur Tagh, China": (38.594, 75.313),
         "Lenin Peak": (39.343, 72.877), "Khan Tengri": (42.211, 80.175), "Belukha": (49.807, 86.590),
         "Klyuchevskaya Sopka": (56.056, 160.642), "Gunnbjørn Fjeld": (68.919, -29.898), "Damavand": (35.952, 52.110),
         "Mount Elbrus": (43.355, 42.439), "Alam-Kuh": (36.379, 50.960), "Sabalan": (38.266, 47.835),
         "Kangchenjunga": (27.703, 88.148), "Mount Everest": (27.988, 86.925), "La Soufrière, Guadeloupe": (16.044, -61.663),
         "Mount Scenery, Saba": (17.635, -63.237), "Pico do Príncipe": (1.578, 7.381), "Mount Damavand": (35.952, 52.110)}


def peak_name(lat, lon, km=4.0, default=None):
    for nm, (la, lo) in PEAKS.items():
        if gc_km((lat, lon), (la, lo)) < km:
            return nm
    return default


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s, layout_engine=ImageFont.Layout.RAQM)


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gc_km(a, b):
    return 2 * R_EARTH * np.arcsin(np.clip(np.linalg.norm(unit(*a) - unit(*b)) / 2, 0, 1))


def dest(lat, lon, brg, d_km):
    """Point(s) d_km from (lat, lon) along bearing(s) brg (radians)."""
    p1, l1, dl = math.radians(lat), math.radians(lon), np.asarray(d_km) / R_EARTH
    p2 = np.arcsin(np.sin(p1) * np.cos(dl) + np.cos(p1) * np.sin(dl) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(dl) * np.cos(p1), np.cos(dl) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2) * np.ones_like(dl * brg), (np.degrees(l2) + 540) % 360 - 180


def bearing(a, b):
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    return math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))


class Globe:
    def __init__(self):
        # the world texture (globe_texture_world.py: corrected Greenland ice sheet), at half resolution
        tex = cv2.imread(os.path.join(DATA, "globe_tex_world.png"))
        tex = cv2.resize(tex, (tex.shape[1] // 2, tex.shape[0] // 2), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        self.tex = tex
        self.TH, self.TW = tex.shape[:2]

    def basis(self, lat0, lon0):
        c = unit(lat0, lon0)                              # towards the viewer
        east = np.array([-math.sin(math.radians(lon0)), math.cos(math.radians(lon0)), 0.0])
        north = np.cross(c, east)
        return c, east, north

    def render(self, lat0, lon0, rpx, h=VH, w=VW, cy=GCY):
        c, e, n = self.basis(lat0, lon0)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        x = (xs - w / 2) / rpx
        y = -(ys - cy) / rpx
        r2 = x * x + y * y
        inside = r2 <= 1
        z = np.sqrt(np.clip(1 - r2, 0, 1))
        P = x[..., None] * e + y[..., None] * n + z[..., None] * c
        lat = np.degrees(np.arcsin(np.clip(P[..., 2], -1, 1)))
        lon = np.degrees(np.arctan2(P[..., 1], P[..., 0]))
        mx = ((lon + 180) / 360 * self.TW).astype(np.float32)
        my = ((90 - lat) / 180 * self.TH).astype(np.float32)
        img = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        limb = np.clip(1 - z, 0, 1)
        img = img * (0.85 + 0.25 * z[..., None]) * inside[..., None]
        # atmosphere glow at the limb
        rr = np.sqrt(r2)
        halo = np.exp(-np.maximum(rr - 1, 0) * rpx / 22) * (rr > 0.985)
        img += (halo + limb ** 6 * 0.6 * inside)[..., None] * np.array([0.10, 0.22, 0.45], np.float32) * 0.6
        return img

    def project(self, lat, lon, lat0, lon0, rpx, w=VW, cy=GCY):
        c, e, n = self.basis(lat0, lon0)
        P = unit(np.asarray(lat), np.asarray(lon))
        return w / 2 + (P @ e) * rpx, cy - (P @ n) * rpx, P @ c


def sprite(sig, size):
    k = np.arange(size, dtype=np.float32) - size // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


class Film:
    def __init__(self):
        L = json.load(open(os.path.join(DATA, "loop.json")))
        self.L = L
        st = L["stations"]
        self.st = st
        self.lat = np.array([s["lat"] for s in st]); self.lon = np.array([s["lon"] for s in st])
        self.kind = [s["kind"] for s in st]
        self.leg = [s["leg"] for s in st]
        self.n_out = sum(1 for s in st if s["leg"] == "out")
        self.globe = Globe()
        gj = json.load(open(os.path.join(ROOT, "data", "ne", "ne_50m_admin_0_countries.geojson")))
        polys = [shapely.geometry.shape(f["geometry"]) for f in gj["features"]]
        names = [f["properties"]["NAME"] for f in gj["features"]]
        tree = STRtree(polys)
        self.country = []
        for s in st:
            if s["kind"] == "ship":
                self.country.append("North Atlantic (fire ship)")
                continue
            p = shapely.Point(s["lon"], s["lat"])
            hit = tree.query(p, predicate="intersects")
            nm = names[hit[0]] if len(hit) else names[tree.nearest(p)]
            self.country.append({"United States of America": "USA (Alaska)" if s["lon"] < -140 else "USA",
                                 "Faeroe Is.": "Faroe Islands"}.get(nm, nm))
        self.hop_km = np.r_[0, [gc_km((self.lat[i - 1], self.lon[i - 1]), (self.lat[i], self.lon[i])) for i in range(1, len(st))]]
        self.hz = L.get("verified_heights_m", [s_["h_m"] for s_ in st])
        no = self.n_out
        hi = int(np.argmax([self.hz[i] if self.kind[i] == "fire" else 0 for i in range(no)]))
        lg = int(np.argmax(self.hop_km[:no]))
        br = next(i for i in range(1, no) if self.country[i - 1] == "Russia" and self.country[i].startswith("USA"))
        nm = lambda i: peak_name(self.lat[i], self.lon[i], default=f"a summit in {self.country[i]}")
        self.events = {hi: [(f"Highest fire: {self.hz[hi]:,.0f} m", "call", GOLD_T), (nm(hi), "call2", INK)],
                       br: [("The Bering Strait", "call", GOLD_T), (f"Russia → Alaska in one {self.hop_km[br]:.0f} km jump", "call2", INK)],
                       lg: [(f"Longest sightline: {self.hop_km[lg]:.0f} km", "call", GOLD_T), (f"{nm(lg - 1)} → {nm(lg)}", "call2", INK)]}
        self.gap = self.gap_numbers()
        self.timeline()
        self.f = dict(title=font("Inter-Black.otf", 60), sub=font("Inter-Medium.otf", 30), big=font("InterDisplay-Bold.otf", 120),
                      lab=font("Inter-SemiBold.otf", 28), stat=font("Inter-Medium.otf", 34), call=font("Inter-Bold.otf", 46),
                      call2=font("Inter-Medium.otf", 32), small=font("Inter-Regular.otf", 21))
        self.spr = {s: sprite(s, int(s * 8) | 1) for s in (1.2, 2.5, 6.0, 14.0)}

    def gap_numbers(self):
        import chain as CH
        last = self.st[self.n_out - 1]
        pk = np.load(os.path.join(DATA, "peaks_hr.npy"))
        m = (pk[:, 0] > 63.2) & (pk[:, 0] < 66.7) & (pk[:, 1] > -24.6) & (pk[:, 1] < -13.4)
        d = np.array([gc_km((last["lat"], last["lon"]), (a, b)) for a, b in pk[m][:, :2]])
        rs = CH.reach(self.hz[self.n_out - 1] / 1000 + CH.PYRE_KM) + CH.reach(pk[m][:, 2] / 1000 + CH.PYRE_KM)
        j = int(np.argmin(d - rs))
        ra = float(CH.reach(self.hz[self.n_out - 1] / 1000 + CH.PYRE_KM))
        return dict(dist=float(d[j]), reach=float(rs[j]), short=float(d[j] - rs[j]), h=float(self.hz[self.n_out - 1]),
                    a=(last["lat"], last["lon"]), b=tuple(map(float, pk[m][j, :2])), ra=ra, rb=float(rs[j]) - ra)

    # ------------------------------------------------------------------ timing
    def timeline(self):
        n = len(self.st)
        t = np.zeros(n)
        HOOK = 3.5
        cur = HOOK
        for i in range(n):
            if i == 0:
                t[i] = cur
                continue
            leg = self.leg[i]
            if leg == "out":
                k = i
                dt = 0.55 if k < 6 else (0.24 if k < self.n_out - 6 else 0.4)
                if (i - 1) in self.events:
                    dt += 2.0                                        # linger on a callout
                cur += dt
            else:
                if i == self.n_out:
                    cur += 4.2                                       # the fire dies... then the ships
                cur += 0.32 if self.kind[i] == "ship" else 0.38
            t[i] = cur
        self.t_lit = t
        self.t_dead = t[self.n_out - 1] + 0.6
        self.t_ships = t[self.n_out] - 0.4
        self.t_loop = t[-1]
        self.total = self.t_loop + 6.5

    # ------------------------------------------------------------------ camera
    def camera(self, t):
        lit = np.nonzero(self.t_lit <= t)[0]
        k = lit[-1] if len(lit) else 0
        # follow a weighted average of the last few lit fires (smooth), via unit vectors
        idx = np.arange(max(0, k - 5), k + 1)
        if k + 1 < len(self.st) and t > self.t_lit[k]:
            u = min(1.0, (t - self.t_lit[k]) / max(self.t_lit[k + 1] - self.t_lit[k], 1e-3))
            idx = np.r_[idx, k + 1]
            wts = np.r_[np.linspace(0.3, 1, len(idx) - 1), u]
        else:
            wts = np.linspace(0.3, 1, len(idx))
        v = (unit(self.lat[idx], self.lon[idx]) * wts[:, None]).sum(0)
        v /= np.linalg.norm(v)
        lat0, lon0 = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
        lat0 = max(min(lat0, 72), 20)
        rpx = 2300
        if t < 3.5:
            rpx = 3600 - 1300 * (t / 3.5) ** 2
        if self.t_dead <= t < self.t_ships + 1.0:                   # pull back to show the gap
            u = min(1.0, (t - self.t_dead) / 2.0)
            u = u * u * (3 - 2 * u)
            gap = unit(np.array([66.0]), np.array([-14.0]))[0]
            v2 = v * (1 - u) + gap * u
            v2 /= np.linalg.norm(v2)
            lat0, lon0 = math.degrees(math.asin(v2[2])), math.degrees(math.atan2(v2[1], v2[0]))
            rpx = 2300 - 500 * u
        elif t >= self.t_ships + 1.0 and t < self.t_loop + 0.5:
            rpx = 1800
        elif t >= self.t_loop + 0.5:                                # final spin around the whole loop
            u = min(1.0, (t - self.t_loop - 0.5) / 5.5)
            u = u * u * (3 - 2 * u)
            rpx = 1800 - 1300 * min(1.0, u * 2.5)
            lon0 = -4.0 + 360 * u
            lat0 = 50 - 5 * math.sin(math.pi * u)
        return lat0, lon0, rpx

    # ------------------------------------------------------------------ frame
    def frame(self, t):
        lat0, lon0, rpx = self.camera(t)
        img = self.globe.render(lat0, lon0, rpx)
        X, Y, Z = self.globe.project(self.lat, self.lon, lat0, lon0, rpx)
        E = np.zeros_like(img)
        scale = np.clip(rpx / 2300, 0.45, 1.6)
        lit = self.t_lit <= t
        dead = self.t_dead <= t < self.t_ships + 0.2
        # sightlines
        lines = np.zeros((VH, VW), np.float32)
        hot = np.zeros((VH, VW), np.float32)
        for i in range(1, len(self.st)):
            if not lit[i] or Z[i] < 0.05 or Z[i - 1] < 0.05:
                continue
            age = t - self.t_lit[i]
            p0 = (int(X[i - 1] * 16), int(Y[i - 1] * 16)); p1 = (int(X[i] * 16), int(Y[i] * 16))
            cv2.line(lines, p0, p1, 0.55, 1, cv2.LINE_AA, shift=4)
            if age < 0.6:
                cv2.line(hot, p0, p1, float(1.6 * (1 - age / 0.6)), 2, cv2.LINE_AA, shift=4)
        E += lines[..., None] * FIRE * 0.8 + hot[..., None] * CORE
        self.gap_lbl = None
        if dead:
            self.draw_gap(E, t, lat0, lon0, rpx)
        # fires
        rng = np.random.default_rng(int(t * FPS))
        for i in range(len(self.st)):
            if Z[i] < 0.02:
                continue
            x, y = X[i], Y[i]
            if not (-50 < x < VW + 50 and -50 < y < VH + 50):
                continue
            if lit[i]:
                age = t - self.t_lit[i]
                flick = 0.85 + 0.25 * math.sin(t * 9 + i * 1.7) + 0.08 * rng.standard_normal()
                amp = flick * (1 + 2.5 * math.exp(-age / 0.25))
                if dead and i == self.n_out - 1:
                    amp *= 1 + 0.6 * math.sin(t * 14)
                col = SHIP if self.kind[i] == "ship" else FIRE
                self.blit(E, x, y, self.spr[2.5] * amp * 1.4, CORE)
                self.blit(E, x, y, self.spr[6.0] * amp * 0.7 * scale, col)
                self.blit(E, x, y, self.spr[14.0] * amp * 0.22 * scale, col)
            elif self.leg[i] == "out" or t >= self.t_ships:
                self.blit(E, x, y, self.spr[1.2] * 0.25, np.array([0.5, 0.55, 0.65], np.float32))   # unlit pyre
        small = cv2.resize(E, (VW // 2, VH // 2), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 10) * 0.5, (VW, VH))
        out = 1 - np.exp(-(img + E + bloom) * 1.15)
        frame = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        self.hud(ImageDraw.Draw(frame, "RGBA"), t, X, Y, Z)
        return np.asarray(frame)

    def draw_gap(self, E, t, lat0, lon0, rpx):
        """The two horizons that fail to touch: each summit's sight range, and the red gap between them."""
        g = self.gap
        u = min(1.0, (t - self.t_dead) / 0.8) * min(1.0, (self.t_ships + 0.2 - t) / 0.3)
        layer = np.zeros((VH, VW), np.float32)
        th = np.linspace(0, 2 * np.pi, 241)
        for (la, lo), r in ((g["a"], g["ra"]), (g["b"], g["rb"])):
            plat, plon = dest(la, lo, th, r)
            X, Y, Z = self.globe.project(plat, plon, lat0, lon0, rpx)
            pts = np.c_[X * 16, Y * 16].astype(np.int32)
            for k in range(0, len(th) - 1, 2):                       # dashed
                if Z[k] > 0.05 and Z[k + 1] > 0.05:
                    cv2.line(layer, tuple(pts[k]), tuple(pts[k + 1]), 0.9, 2, cv2.LINE_AA, shift=4)
        E += layer[..., None] * FIRE * 0.7 * u
        # the gap itself, along the line between the summits
        brg = bearing(g["a"], g["b"])
        f = np.linspace(g["ra"], g["dist"] - g["rb"], 24)
        plat, plon = dest(g["a"][0], g["a"][1], brg, f)
        X, Y, Z = self.globe.project(plat, plon, lat0, lon0, rpx)
        red = np.zeros((VH, VW), np.float32)
        pulse = 0.8 + 0.35 * math.sin(t * 8)
        pts = np.c_[X * 16, Y * 16].astype(np.int32)
        cv2.polylines(red, [pts], False, 1.0, 5, cv2.LINE_AA, shift=4)
        E += red[..., None] * np.array([1.0, 0.18, 0.12], np.float32) * 1.6 * pulse * u
        # Iceland's summit: an unlit pyre
        X, Y, Z = self.globe.project(np.array([g["b"][0]]), np.array([g["b"][1]]), lat0, lon0, rpx)
        self.blit(E, X[0], Y[0], self.spr[1.2] * 0.6 * u, np.array([0.7, 0.75, 0.85], np.float32))
        mid = len(f) // 2
        Xm, Ym, _ = self.globe.project(plat[mid:mid + 1], plon[mid:mid + 1], lat0, lon0, rpx)
        self.gap_lbl = (float(Xm[0]), float(Ym[0]), u)

    def blit(self, E, x, y, spr, col):
        h = spr.shape[0] // 2
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - h), min(VH, yi + h + 1), max(0, xi - h), min(VW, xi + h + 1)
        if y1 > y0 and x1 > x0:
            E[y0:y1, x0:x1] += spr[y0 - (yi - h):y1 - (yi - h), x0 - (xi - h):x1 - (xi - h), None] * col

    # ------------------------------------------------------------------ text
    def hud(self, d, t, X, Y, Z):
        f = self.f
        c = lambda y, s, fn, col, a=255: d.text((VW // 2, y), s, font=fn, fill=tuple(col) + (int(a),), anchor="ma")
        c(60, "How many signal fires", f["title"], INK)
        c(130, "to circle the Earth?", f["title"], INK)
        c(214, "Mountaintop to mountaintop, line of sight only", f["sub"], INK2)
        lit = np.nonzero(self.t_lit <= t)[0]
        k = int(lit[-1]) if len(lit) else -1
        n_fire = sum(1 for i in range(min(k + 1, len(self.st) - 1)) if self.kind[i] == "fire")   # Plymouth counts once
        n_ship = sum(1 for i in range(k + 1) if self.kind[i] == "ship")
        km = float(self.hop_km[:k + 1].sum()) if k >= 0 else 0
        hrs = max(k, 0) * MIN_PER_RELAY / 60
        y0 = 1560
        d.rounded_rectangle([40, y0 - 10, VW - 40, y0 + 290], 24, fill=(6, 10, 18, 205))
        d.text((80, y0 + 6), "FIRES LIT", font=f["lab"], fill=MUTED, anchor="la")
        d.text((72, y0 + 38), f"{n_fire}", font=f["big"], fill=GOLD_T, anchor="la", features=["tnum"])
        if t >= self.t_ships:
            d.text((560, y0 + 6), "FIRE SHIPS", font=f["lab"], fill=MUTED, anchor="la")
            d.text((552, y0 + 38), f"{n_ship}", font=f["big"], fill=(255, 214, 150), anchor="la", features=["tnum"])
        else:
            d.text((560, y0 + 6), "NOW", font=f["lab"], fill=MUTED, anchor="la")
            nm = self.country[k] if k >= 0 else "England"
            fs = f["call"] if d.textlength(nm, font=f["call"]) < 440 else f["call2"]
            d.text((560, y0 + 52), nm, font=fs, fill=INK, anchor="la")
        d.text((80, y0 + 196), f"{km:,.0f} km of sightlines", font=f["stat"], fill=INK2, anchor="la")
        d.text((80, y0 + 240), f"≈ {int(hrs)} h {int((hrs % 1) * 60):02d} min at {MIN_PER_RELAY:.0f} min per relay",
               font=f["stat"], fill=INK2, anchor="la")
        # callouts
        def box(lines, y=300, a=255):
            hgt = 30 + 58 * len(lines)
            d.rounded_rectangle([70, y, VW - 70, y + hgt], 22, fill=(6, 10, 18, int(215 * a / 255)))
            for j, (s, fn, col) in enumerate(lines):
                c(y + 22 + 58 * j, s, fn, col, a)
        if t < 3.5:
            a = 255 * min(1, t / 0.4)
            box([("1588: beacon fires warn England", f["call2"], INK), ("of the Spanish Armada.", f["call2"], INK),
                 ("What if the signal kept going?", f["call"], GOLD_T)], a=a)
        for i, lines in self.events.items():
            if self.t_lit[i] <= t < self.t_lit[i] + 2.3:
                a = 255 * min(1, (t - self.t_lit[i]) / 0.25, (self.t_lit[i] + 2.3 - t) / 0.3)
                box([(s_, f[fn], col) for s_, fn, col in lines], a=a)
        if self.t_dead <= t < self.t_ships:
            a = 255 * min(1, (t - self.t_dead) / 0.4)
            g = self.gap
            box([("The fire dies in Greenland.", f["call"], RED),
                 (f"Iceland is {g['dist']:.0f} km away. Both horizons", f["call2"], INK),
                 (f"together reach {g['reach']:.0f} km: {g['short']:.0f} km short.", f["call2"], INK)], a=a)
        if self.gap_lbl is not None:
            x, y, u = self.gap_lbl
            txt = f"{self.gap['short']:.0f} km gap"
            w = d.textlength(txt, font=f["stat"])
            x = min(max(x + 26, 50), VW - 50 - w)
            d.rounded_rectangle([x - 12, y - 26, x + w + 12, y + 26], 14, fill=(6, 10, 18, int(200 * u)))
            d.text((x, y), txt, font=f["stat"], fill=RED + (int(255 * u),), anchor="lm")
        if self.t_ships <= t < self.t_ships + 2.6:
            a = 255 * min(1, (t - self.t_ships) / 0.3)
            box([("Close the loop with fire ships", f["call"], GOLD_T), ("a fire 30 m up the mast, seen 21 km", f["call2"], INK)], a=a)
        if t >= self.t_loop:
            a = 255 * min(1, (t - self.t_loop) / 0.5)
            tot = len(self.st) - 1
            box([("Back in Plymouth", f["call"], GOLD_T),
                 (f"{sum(1 for s in self.kind[:-1] if s == 'fire')} fires + {self.kind.count('ship')} ships = {tot} relays", f["call2"], INK),
                 (f"{self.hop_km.sum():,.0f} km, ≈ {tot * MIN_PER_RELAY / 60:.1f} hours around the world", f["call2"], INK)], a=a)
        d.text((VW // 2, VH - 44), "Elevation: Terrarium (AWS). Earth curvature + normal refraction. A thought experiment.",
               font=f["small"], fill=MUTED, anchor="ma")


def render(still=None):
    os.makedirs(OUT, exist_ok=True)
    F = Film()
    print(f"film {F.total:.1f} s, {len(F.st)} stations", flush=True)
    if still is not None:
        Image.fromarray(F.frame(still)).save(os.path.join(OUT, "fires_still.png"))
        return
    vid = os.path.join(OUT, "_fires.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", vid],
                         stdin=subprocess.PIPE)
    n = int(round(F.total * FPS))
    for i in range(n):
        p.stdin.write(F.frame(i / FPS).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close(); p.wait()
    import score as SC
    out_t = F.t_lit[:F.n_out]
    bering = [F.t_lit[i] for i in range(1, F.n_out) if F.country[i - 1] == "Russia" and F.country[i].startswith("USA")]
    energy = [(0, 0.55), (3.5, 0.8), (F.t_dead - 0.01, 0.95), (F.t_dead, 0.25), (F.t_ships, 0.25), (F.t_ships + 0.5, 0.88),
              (F.t_loop, 1.0), (F.t_loop + 0.01, 0.0), (F.total, 0.0)]
    impacts = [(3.5, 0.9)] + [(b, 1.0) for b in bering] + [(F.t_dead, 1.2), (F.t_ships + 0.5, 0.9), (F.t_loop, 1.4)]
    risers = [(F.t_dead - 2.5, F.t_dead), (F.t_ships - 2.0, F.t_ships + 0.5), (F.t_loop - 3.0, F.t_loop)]
    sc = SC.Score(F.total, energy, impacts, risers, transpose_at=F.t_ships + 0.5, final_at=F.t_loop)
    sc.compose()
    wav = os.path.join(OUT, "fires_score.wav")
    SC.write(sc.mix(), wav)
    final = os.path.join(OUT, "signal_fires.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    print("wrote", final)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "still":
        render(float(sys.argv[2]))
    else:
        render()
