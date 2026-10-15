"""Globe sections of the long video (16:9): hook, history, rules, the journey around the world,
the most-countries tour, outro. Rendered through longform.Section.

    python long_globe.py journey            -> out/long/journey.mp4
    python long_globe.py journey 40 80.5    -> stills
"""
import os, sys, json, math
import numpy as np
import cv2
import shapely
from PIL import Image, ImageDraw
import style as ST
import globe16 as G
from globe16 import W, H, unit, slerp_ll, gc_km, FIRE, CORE, SHIP
from longform import Section, smooth, DATA, OUT, FPS
from render_fires import dest, bearing, MIN_PER_RELAY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ease(u):
    return smooth(u)


class Cam:
    """Keyframed camera: [(t, lat, lon, rpx)], great-circle moves with eased timing, log-zoom."""
    def __init__(self, keys):
        self.k = keys

    def __call__(self, t):
        k = self.k
        if t <= k[0][0]:
            return k[0][1:]
        for a, b in zip(k[:-1], k[1:]):
            if t <= b[0]:
                u = ease((t - a[0]) / max(b[0] - a[0], 1e-6))
                la, lo = slerp_ll((a[1], a[2]), (b[1], b[2]), u)
                r = math.exp(math.log(a[3]) + (math.log(b[3]) - math.log(a[3])) * u)
                return la, lo, r
        return k[-1][1:]


def hud_text(fr, items):
    """items: [(x, y, text, size, colour, weight, anchor, alpha)]"""
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    for x, y, s, size, col, wt, anc, a in items:
        if a > 0:
            d.text((x, y), s, font=ST.font(size, wt), fill=tuple(col) + (int(255 * a),), anchor=anc)
    return np.asarray(im)


def labels(fr, labs):
    """Small station labels with a dark backing: [(x, y, text, alpha)]"""
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    f = ST.font(26, "SemiBold")
    for x, y, s, a in labs:
        if a <= 0:
            continue
        tw = d.textlength(s, font=f)
        d.rounded_rectangle([x + 14, y - 18, x + 30 + tw, y + 18], 9, fill=ST.PANEL + (int(170 * a),))
        d.text((x + 22, y), s, font=f, fill=ST.INK + (int(255 * a),), anchor="lm")
    return np.asarray(im)


# ====================================================================== hook
class Hook(Section):
    name, dur = "hook", 13.0

    def __init__(self):
        self.g = G.Globe16()
        L = json.load(open(os.path.join(DATA, "loop.json")))
        st = L["stations"]
        self.lat = np.array([s["lat"] for s in st]); self.lon = np.array([s["lon"] for s in st])
        self.kind = [s["kind"] for s in st]
        n = len(st)
        self.t_lit = 0.8 + np.arange(n) / (n - 1) * 6.5

    def frame(self, t):
        lat0, lon0, rpx = 62.0, -10 + 16 * t, 500 + 5 * t
        cy = 640
        img = self.g.render(lat0, lon0, rpx, cy=cy)
        X, Y, Z = self.g.project(self.lat, self.lon, lat0, lon0, rpx, cy=cy)
        Ly = G.Layer()
        for i in range(len(self.lat)):
            if self.t_lit[i] > t or Z[i] < 0.02:
                continue
            if i and Z[i - 1] > 0.02:
                Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 0.5)
            age = t - self.t_lit[i]
            Ly.fire(X[i], Y[i], 0.55 * (1 + 2.0 * math.exp(-age / 0.25)), 0.8, SHIP if self.kind[i] == "ship" else FIRE)
        fr = Ly.compose(img)
        a = ST.fade(t, 1.6, self.dur - 0.3, 0.6, 0.6)
        fr = hud_text(fr, [(W / 2, 80, "How many signal fires would it take", 64, ST.INK, "Bold", "ma", a),
                           (W / 2, 162, "to send a message around the world?", 64, ST.INK, "Bold", "ma", a)])
        fin = smooth(t / 0.8)
        return (fr * fin).astype(np.uint8) if fin < 1 else fr


# ====================================================================== history
LOULON = (37.47, 34.62)
GREAT_PALACE = (41.006, 28.977)
BADALING = (40.359, 116.016)
PLYMOUTH = (50.364, -4.142)


class History(Section):
    name, dur = "history", 72.0

    def __init__(self):
        self.g = G.Globe16(patches=[("aegean", 36.5, 41.5, 21.0, 28.5, 150)])
        hj = json.load(open(os.path.join(DATA, "history.json")))["aeschylus"]
        self.ae = hj
        self.ae_t = 10.0 + np.arange(len(hj)) * 1.35
        self.cam = Cam([(0, 32, 30, 560), (6.0, 32, 30, 600), (9.0, 38.9, 24.6, 9000), (25.5, 38.9, 24.6, 9200),
                        (28.5, 39.3, 31.6, 4300), (38.5, 39.3, 31.6, 4500), (42.0, 40.3, 112.0, 2400),
                        (48.5, 40.3, 112.0, 2600), (52.0, 52.0, -2.6, 4600), (60.0, 52.0, -2.6, 4800), (72, 52.0, -2.6, 4900)])
        self.caps = [
            (0.4, 6.4, [("Before the telegraph, the fastest way", 46, ST.INK, "SemiBold"),
                        ("to send news across a country was fire.", 46, ST.INK, "SemiBold")]),
            (9.6, 21.0, [("458 BC: in Aeschylus's play Agamemnon,", 40, ST.INK, "SemiBold"),
                         ("a chain of beacons carries news of Troy's fall to Argos in one night", 32, ST.INK2, "Regular")]),
            (21.4, 27.6, [("We checked it against real terrain: every hop works", 42, ST.GOLD, "Bold"),
                          (f"8 hops, {sum(h.get('hop_km', 0) for h in self.ae):,.0f} km", 32, ST.INK2, "Regular")]),
            (29.2, 38.8, [("9th century: nine beacons linked the Byzantine frontier", 40, ST.INK, "SemiBold"),
                          ("to Constantinople, reportedly delivering warnings within about an hour", 32, ST.INK2, "Regular")]),
            (42.4, 49.2, [("On China's Great Wall, beacon towers sent warnings", 40, ST.INK, "SemiBold"),
                          ("with smoke by day and fire by night", 32, ST.INK2, "Regular")]),
            (52.4, 60.0, [("1588: the Spanish Armada is sighted off Cornwall", 40, ST.INK, "SemiBold"),
                          ("and beacons carry the alarm across England", 32, ST.INK2, "Regular")]),
        ]
        self.lotr_bg = None
        p = os.path.join(OUT, "lotr_bg.png")
        if os.path.exists(p):
            self.lotr_bg = cv2.imread(p)[..., ::-1].astype(np.float32)

    def frame(self, t):
        if t >= 61.0 and self.lotr_bg is not None:
            return self.lotr(t)
        lat0, lon0, rpx = self.cam(t)
        img = self.g.render(lat0, lon0, rpx)
        Ly = G.Layer()
        labs = []
        # Aeschylus
        a_ae = ST.fade(t, 9.0, 28.0, 0.5, 0.8)
        if a_ae > 0:
            la = np.array([h["lat"] for h in self.ae]); lo = np.array([h["lon"] for h in self.ae])
            X, Y, Z = self.g.project(la, lo, lat0, lon0, rpx)
            for i in range(len(self.ae)):
                if t < self.ae_t[i]:
                    continue
                age = t - self.ae_t[i]
                if i:
                    Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 0.6 * a_ae, 2)
                Ly.fire(X[i], Y[i], a_ae * 0.8 * (1 + 2.5 * math.exp(-age / 0.3)), 1.4)
                labs.append((X[i], Y[i], self.ae[i]["name"], a_ae * min(1, age / 0.3)))
        # Byzantium
        a_by = ST.fade(t, 28.5, 41.0, 0.6, 0.8)
        if a_by > 0:
            X, Y, Z = self.g.project(np.array([LOULON[0], GREAT_PALACE[0]]), np.array([LOULON[1], GREAT_PALACE[1]]), lat0, lon0, rpx)
            u = smooth((t - 30.0) / 4.0)
            n = 40
            for k in range(int(n * u)):
                if k % 2 == 0:
                    f0, f1 = k / n, (k + 1) / n
                    p0 = slerp_ll(LOULON, GREAT_PALACE, f0); p1 = slerp_ll(LOULON, GREAT_PALACE, f1)
                    x0, y0, _ = self.g.project(np.array([p0[0]]), np.array([p0[1]]), lat0, lon0, rpx)
                    x1, y1, _ = self.g.project(np.array([p1[0]]), np.array([p1[1]]), lat0, lon0, rpx)
                    Ly.line((x0[0], y0[0]), (x1[0], y1[0]), 0.7 * a_by, 2)
            Ly.fire(X[0], Y[0], a_by * 1.0, 1.3)
            labs.append((X[0], Y[0], "Loulon, the Cilician Gates", a_by))
            if u >= 1:
                Ly.fire(X[1], Y[1], a_by * 1.0, 1.3)
                labs.append((X[1], Y[1], "Constantinople", a_by))
        # Great Wall
        a_gw = ST.fade(t, 42.0, 51.0, 0.6, 0.8)
        if a_gw > 0:
            X, Y, Z = self.g.project(np.array([BADALING[0]]), np.array([BADALING[1]]), lat0, lon0, rpx)
            Ly.fire(X[0], Y[0], a_gw * (0.9 + 0.1 * math.sin(t * 9)), 1.3)
            labs.append((X[0], Y[0], "Badaling, Great Wall", a_gw))
        # Armada
        a_ar = ST.fade(t, 52.0, 62.0, 0.6, 0.8)
        if a_ar > 0:
            X, Y, Z = self.g.project(np.array([PLYMOUTH[0]]), np.array([PLYMOUTH[1]]), lat0, lon0, rpx)
            Ly.fire(X[0], Y[0], a_ar * (0.9 + 0.1 * math.sin(t * 9)), 1.3)
            labs.append((X[0], Y[0], "Plymouth", a_ar))
        fr = Ly.compose(img)
        fr = labels(fr, labs)
        fr = ST.draw_captions(fr, t, self.caps)
        fin = smooth(t / 0.6)
        return (fr * fin).astype(np.uint8) if fin < 1 else fr

    def lotr(self, t):
        a = smooth((t - 61.0) / 1.0)
        bg = self.lotr_bg * (0.75 + 0.05 * (t - 61) / 11)
        z = 1 + 0.03 * (t - 61) / 11                                     # slow push-in
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        bg = cv2.warpAffine(bg, M, (W, H))
        if a < 1:
            lat0, lon0, rpx = self.cam(t)
            prev = self.g.render(lat0, lon0, rpx) * 255
            bg = prev * (1 - a) + bg * a
        fr = np.clip(bg, 0, 255).astype(np.uint8)
        caps = [(61.8, 67.6, [("And in Tolkien's The Return of the King,", 40, ST.INK, "SemiBold"),
                              ("beacons on the peaks of the White Mountains call Rohan to Gondor's aid", 32, ST.INK2, "Regular")]),
                (68.0, 72.0, [("So how does a chain of beacons work?", 46, ST.GOLD, "Bold")])]
        return ST.draw_captions(fr, t, caps)


# ====================================================================== the journey (the loop)
class Journey(Section):
    name = "journey"

    def __init__(self):
        from render_fires import Film
        F = Film()
        self.F = F
        self.g = G.Globe16()
        self.lat, self.lon, self.kind, self.leg = F.lat, F.lon, F.kind, F.leg
        self.n_out, self.country, self.hop_km, self.gap = F.n_out, F.country, F.hop_km, F.gap
        n = len(self.lat)
        first = {}
        for i, c in enumerate(self.country):
            first.setdefault(c, i)
        cnt = lambda c: sum(1 for i in range(n - 1) if self.country[i] == c and self.kind[i] == "fire")
        from render_fires import peak_name
        no = self.n_out
        hz = F.hz
        at = lambda name: next((i for i in range(no) if peak_name(self.lat[i], self.lon[i]) == name), None)
        nm = lambda i: peak_name(self.lat[i], self.lon[i], default=f"a summit in {self.country[i]}")
        hi = int(np.argmax([hz[i] if self.kind[i] == "fire" else 0 for i in range(no)]))
        lg = int(np.argmax(self.hop_km[:no]))
        br = next(i for i in range(1, no) if self.country[i - 1] == "Russia" and self.country[i].startswith("USA"))
        ev = {}
        mb = at("Mont Blanc")
        if mb is not None:
            ev[mb] = [("Mont Blanc", 44, ST.GOLD, "Bold"), (f"the fire from earlier, seen {self.hop_km[mb]:.0f} km away in {self.country[mb - 1]}", 30, ST.INK2, "Regular")]
        ar = at("Mount Ararat")
        if ar is not None:
            ev[ar] = [(f"Mount Ararat, {hz[ar]:,.0f} m", 44, ST.GOLD, "Bold"), ("eastern Turkey", 30, ST.INK2, "Regular")]
        ev[hi] = [(f"Highest fire: {hz[hi]:,.0f} m", 44, ST.GOLD, "Bold"), (nm(hi), 30, ST.INK2, "Regular")]
        if "Russia" in first:
            ev[first["Russia"]] = [(f"Russia: {cnt('Russia')} fires across Siberia", 44, ST.GOLD, "Bold"), ("from the Altai to the Bering Sea", 30, ST.INK2, "Regular")]
        ev[br] = [("The Bering Strait", 44, ST.GOLD, "Bold"), (f"Russia to Alaska in one {self.hop_km[br]:.0f} km jump", 30, ST.INK2, "Regular")]
        ev[lg] = [(f"Longest sightline: {self.hop_km[lg]:.0f} km", 44, ST.GOLD, "Bold"), (f"{nm(lg - 1)} to {nm(lg)}", 30, ST.INK2, "Regular")]
        if "Canada" in first:
            ev[first["Canada"]] = [(f"Canada: {cnt('Canada')} fires", 44, ST.GOLD, "Bold"), ("along the Rockies and across the Arctic islands", 30, ST.INK2, "Regular")]
        if "Greenland" in first:
            ev[first["Greenland"]] = [("Greenland", 44, ST.GOLD, "Bold"), (f"{cnt('Greenland')} fires, up onto the ice sheet", 30, ST.INK2, "Regular")]
        self.events = ev
        # timeline
        t = np.zeros(n)
        cur = 5.0
        for i in range(n):
            if i == 0:
                t[i] = cur
                continue
            if self.leg[i] == "out":
                dt = 0.9 if i < 6 else (0.62 if i < self.n_out - 6 else 0.8)
                if (i - 1) in self.events:
                    dt += 2.6
                cur += dt
            else:
                if i == self.n_out:
                    cur += 7.5                                   # the fire dies... then the ships
                cur += 0.45 if self.kind[i] == "ship" else 0.6
                if self.country[i - 1] != self.country[i] and self.kind[i] == "fire":
                    cur += 0.9
            t[i] = cur
        self.t_lit = t
        self.t_dead = t[self.n_out - 1] + 0.7
        self.t_ships = t[self.n_out] - 0.5
        self.t_loop = t[-1]
        self.dur = self.t_loop + 11.0
        print(f"journey {self.dur:.1f} s", flush=True)

    def camera(self, t):
        lit = np.nonzero(self.t_lit <= t)[0]
        k = lit[-1] if len(lit) else 0
        idx = np.arange(max(0, k - 5), k + 1)
        if k + 1 < len(self.lat) and t > self.t_lit[k]:
            u = min(1.0, (t - self.t_lit[k]) / max(self.t_lit[k + 1] - self.t_lit[k], 1e-3))
            idx = np.r_[idx, k + 1]
            wts = np.r_[np.linspace(0.3, 1, len(idx) - 1), u]
        else:
            wts = np.linspace(0.3, 1, len(idx))
        v = (unit(self.lat[idx], self.lon[idx]) * wts[:, None]).sum(0)
        v /= np.linalg.norm(v)
        lat0, lon0 = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
        lat0 = max(min(lat0, 70), 25)
        rpx = 1750
        if t < 5.0:
            rpx = 4200 - 2450 * smooth(t / 5.0)
        if self.t_dead <= t < self.t_ships + 1.0:
            u = smooth((t - self.t_dead) / 2.0)
            gap = unit(66.5, -16.0)
            v2 = v * (1 - u) + gap * u
            v2 /= np.linalg.norm(v2)
            lat0, lon0 = math.degrees(math.asin(v2[2])), math.degrees(math.atan2(v2[1], v2[0]))
            rpx = 1750 - 250 * u
        elif self.t_ships + 1.0 <= t < self.t_loop + 0.8:
            rpx = 1500
        elif t >= self.t_loop + 0.8:
            u = smooth((t - self.t_loop - 0.8) / 8.0)
            rpx = 1500 * (1 - u) + 520 * u
            lon0 = lon0 + 300 * u
            lat0 = lat0 * (1 - u) + 60 * u
        return lat0, lon0, rpx

    def frame(self, t):
        lat0, lon0, rpx = self.camera(t)
        img = self.g.render(lat0, lon0, rpx)
        X, Y, Z = self.g.project(self.lat, self.lon, lat0, lon0, rpx)
        Ly = G.Layer()
        lit = self.t_lit <= t
        dead = self.t_dead <= t < self.t_ships + 0.2
        sc = float(np.clip(rpx / 1750, 0.5, 1.4))
        for i in range(1, len(self.lat)):
            if lit[i] and Z[i] > 0.03 and Z[i - 1] > 0.03:
                Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 0.55)
                age = t - self.t_lit[i]
                if age < 0.6:
                    Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 1.6 * (1 - age / 0.6), 2)
        self.gap_lbl = None
        if dead:
            self.draw_gap(Ly, t, lat0, lon0, rpx)
        for i in range(len(self.lat)):
            if Z[i] < 0.02 or not (-60 < X[i] < W + 60 and -60 < Y[i] < H + 60):
                continue
            if lit[i]:
                age = t - self.t_lit[i]
                amp = (0.85 + 0.2 * math.sin(t * 9 + i * 1.7)) * (1 + 2.5 * math.exp(-age / 0.25))
                if dead and i == self.n_out - 1:
                    amp *= 1 + 0.6 * math.sin(t * 14)
                Ly.fire(X[i], Y[i], amp * 0.9, sc, SHIP if self.kind[i] == "ship" else FIRE)
            elif self.leg[i] == "out" or t >= self.t_ships:
                Ly.blit(X[i], Y[i], 0.9, 0.25, (0.5, 0.55, 0.65))
        fr = Ly.compose(img)
        return self.hud(fr, t)

    def draw_gap(self, Ly, t, lat0, lon0, rpx):
        g = self.gap
        u = min(1.0, (t - self.t_dead) / 0.8) * min(1.0, (self.t_ships + 0.2 - t) / 0.3)
        th = np.linspace(0, 2 * np.pi, 241)
        for (la, lo), r in ((g["a"], g["ra"]), (g["b"], g["rb"])):
            plat, plon = dest(la, lo, th, r)
            X, Y, Z = self.g.project(plat, plon, lat0, lon0, rpx)
            for k in range(0, len(th) - 1, 2):
                if Z[k] > 0.05 and Z[k + 1] > 0.05:
                    Ly.line((X[k], Y[k]), (X[k + 1], Y[k + 1]), 0.9 * u, 2)
        brg = bearing(g["a"], g["b"])
        f = np.linspace(g["ra"], g["dist"] - g["rb"], 24)
        plat, plon = dest(g["a"][0], g["a"][1], brg, f)
        X, Y, Z = self.g.project(plat, plon, lat0, lon0, rpx)
        pulse = 0.8 + 0.35 * math.sin(t * 8)
        red = np.zeros((H, W), np.float32)
        cv2.polylines(red, [np.c_[X * 16, Y * 16].astype(np.int32)], False, 1.0, 5, cv2.LINE_AA, shift=4)
        Ly.E += red[..., None] * np.array([1.0, 0.18, 0.12], np.float32) * 1.6 * pulse * u
        Xb, Yb, _ = self.g.project(np.array([g["b"][0]]), np.array([g["b"][1]]), lat0, lon0, rpx)
        Ly.blit(Xb[0], Yb[0], 1.3, 0.6 * u, (0.7, 0.75, 0.85))
        mid = len(f) // 2
        self.gap_lbl = (float(X[mid]), float(Y[mid]), u)

    def hud(self, fr, t):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        lit = np.nonzero(self.t_lit <= t)[0]
        k = int(lit[-1]) if len(lit) else -1
        n = len(self.lat)
        n_fire = sum(1 for i in range(min(k + 1, n - 1)) if self.kind[i] == "fire")
        n_ship = sum(1 for i in range(k + 1) if self.kind[i] == "ship")
        km = float(self.hop_km[:k + 1].sum()) if k >= 0 else 0
        hrs = max(k, 0) * MIN_PER_RELAY / 60
        x0, y0 = 40, 850
        d.rounded_rectangle([x0, y0, x0 + 720, y0 + 190], 22, fill=ST.PANEL + (200,))
        d.text((x0 + 30, y0 + 20), "FIRES LIT", font=ST.font(22, "SemiBold"), fill=ST.MUTED, anchor="la")
        d.text((x0 + 24, y0 + 44), f"{n_fire}", font=ST.font(92, "Bold"), fill=ST.GOLD, anchor="la", features=["tnum"])
        if t >= self.t_ships:
            d.text((x0 + 250, y0 + 20), "FIRE SHIPS", font=ST.font(22, "SemiBold"), fill=ST.MUTED, anchor="la")
            d.text((x0 + 244, y0 + 44), f"{n_ship}", font=ST.font(92, "Bold"), fill=(255, 214, 150), anchor="la", features=["tnum"])
        d.text((x0 + 470, y0 + 20), "NOW", font=ST.font(22, "SemiBold"), fill=ST.MUTED, anchor="la")
        nm = self.country[k] if k >= 0 else "England"
        nm = {"North Atlantic (fire ship)": "Fire ship"}.get(nm, nm)
        fs = ST.font(40, "SemiBold") if d.textlength(nm, font=ST.font(40, "SemiBold")) < 230 else ST.font(28, "SemiBold")
        d.text((x0 + 470, y0 + 56), nm, font=fs, fill=ST.INK, anchor="la")
        d.text((x0 + 30, y0 + 150), f"{km:,.0f} km of sightlines  ·  ≈ {int(hrs)} h {int((hrs % 1) * 60):02d} min at {MIN_PER_RELAY:.0f} min per relay",
               font=ST.font(26, "Regular"), fill=ST.INK2, anchor="la")
        d.text((40, 36), "AROUND THE WORLD FROM PLYMOUTH", font=ST.font(22, "SemiBold"), fill=ST.MUTED, anchor="la")
        caps = []
        if t < 5.0:
            caps.append((0.3, 4.9, [("Start: Plymouth, England", 44, ST.GOLD, "Bold"),
                                    ("where the Armada beacons burned in 1588", 30, ST.INK2, "Regular")], {"anchor": "top"}))
        for i, lines in self.events.items():
            caps.append((self.t_lit[i], self.t_lit[i] + 2.9, lines, {"anchor": "top"}))
        g = self.gap
        caps.append((self.t_dead, self.t_ships, [("The fire dies in Greenland", 46, ST.RED, "Bold"),
                                                 (f"Iceland is {g['dist']:.0f} km away. Both horizons together reach {g['reach']:.0f} km:", 30, ST.INK, "Regular"),
                                                 (f"{g['short']:.0f} km short.", 30, ST.INK, "Regular")], {"anchor": "top"}))
        caps.append((self.t_ships, self.t_ships + 3.2, [("Close the loop with fire ships", 44, ST.GOLD, "Bold"),
                                                        ("a fire 30 m up the mast, seen from 21 km", 30, ST.INK2, "Regular")], {"anchor": "top"}))
        isl = [i for i in range(self.n_out, n - 1) if self.kind[i] == "fire" and self.country[i] != self.country[i - 1]]
        for i in isl:
            nm = self.country[i]
            c = sum(1 for j in range(self.n_out, n - 1) if self.country[j] == nm and self.kind[j] == "fire")
            caps.append((self.t_lit[i], self.t_lit[i] + 1.6, [(f"{nm}: {c} fire{'s' if c > 1 else ''}", 40, ST.GOLD, "Bold")], {"anchor": "top"}))
        tot = n - 1
        n_f = sum(1 for s in self.kind[:-1] if s == "fire")
        caps.append((self.t_loop, self.dur + 1, [("Back in Plymouth", 52, ST.GOLD, "Bold"),
                                                 (f"{n_f} fires + {self.kind.count('ship')} fire ships = {tot} relays", 34, ST.INK, "SemiBold"),
                                                 (f"{self.hop_km.sum():,.0f} km, about {tot * MIN_PER_RELAY / 60:.1f} hours around the world", 30, ST.INK2, "Regular")],
                     {"anchor": "top"}))
        for c in caps:
            a = ST.fade(t, c[0], c[1])
            if a > 0:
                ST.caption(d, c[2], a, **c[3])
        if self.gap_lbl is not None:
            x, y, u = self.gap_lbl
            txt = f"{self.gap['short']:.0f} km gap"
            w = d.textlength(txt, font=ST.font(30))
            x = min(max(x + 26, 50), W - 50 - w)
            d.rounded_rectangle([x - 12, y - 24, x + w + 12, y + 24], 12, fill=ST.PANEL + (int(200 * u),))
            d.text((x, y), txt, font=ST.font(30), fill=ST.RED + (int(255 * u),), anchor="lm")
        out = np.asarray(im)
        if t > self.dur - 0.8:
            out = (out * (1 - smooth((t - self.dur + 0.8) / 0.8))).astype(np.uint8)
        return out


# ====================================================================== rules
class Rules(Section):
    name, dur = "rules", 17.0

    def __init__(self):
        self.g = G.Globe16()

    def frame(self, t):
        u = smooth(t / self.dur)
        lat0, lon0, rpx = 47.0, -4.142 + 18 * (1 - u), 1100 + 500 * u
        cx = W * 0.70
        img = self.g.render(lat0, lon0, rpx, cx=cx)
        X, Y, Z = self.g.project(np.array([PLYMOUTH[0]]), np.array([PLYMOUTH[1]]), lat0, lon0, rpx, cx=cx)
        Ly = G.Layer()
        Ly.fire(X[0], Y[0], 1.0 + 0.15 * math.sin(t * 9), 1.3)
        fr = Ly.compose(img)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([70, 120, 900, 960], 28, fill=ST.PANEL + (215,))
        a0 = ST.fade(t, 0.3, self.dur, 0.5, 0.6)
        d.text((120, 170), "THE CHALLENGE", font=ST.font(30, "Bold"), fill=ST.GOLD + (int(255 * a0),), anchor="la")
        rules = [("Start in Plymouth,", "where England's Armada beacons were lit"),
                 ("Fires only on real mountaintops", "a bonfire 10 m tall on the summit"),
                 ("Each fire must see the next one", "over the curve of the Earth and every ridge between"),
                 ("All the way around the world", "and back to Plymouth"),
                 ("As few fires as possible", "")]
        for k, (a, b) in enumerate(rules):
            al = ST.fade(t, 1.0 + k * 1.6, self.dur, 0.5, 0.6)
            y = 250 + k * 135
            d.text((120, y), f"{k + 1}", font=ST.font(46, "Bold"), fill=ST.GOLD + (int(255 * al),), anchor="la")
            d.text((185, y + 2), a, font=ST.font(40, "SemiBold"), fill=ST.INK + (int(255 * al),), anchor="la")
            if b:
                d.text((185, y + 54), b, font=ST.font(28, "Regular"), fill=ST.INK2 + (int(255 * al),), anchor="la")
        al = ST.fade(t, 9.5, self.dur, 0.5, 0.6)
        d.text((120, 905), "Terrain: Terrarium elevation tiles · every sightline re-checked every 150 m",
               font=ST.font(22, "Regular"), fill=ST.MUTED + (int(255 * al),), anchor="la")
        out = np.asarray(im)
        f = smooth(t / 0.6) * (1 - smooth((t - self.dur + 0.7) / 0.7))
        return (out * f).astype(np.uint8) if f < 1 else out


# ====================================================================== the most-countries tour
class Tour(Section):
    name = "tour"
    INTRO, OUTRO = 16.0, 12.0
    NOTES = {
        "Netherlands": ("Saba: the Netherlands' highest mountain", "is in the Caribbean"),
        "France": ("Guadeloupe", "an overseas part of France"),
        "Vatican": ("Vatican City", "a fire on Vatican Hill"),
        "Monaco": ("Monaco", "a fire on its 161 m high point"),
        "San Marino": ("San Marino", "Monte Titano"),
        "Liechtenstein": ("Liechtenstein", "a summit safely inside its borders"),
        "São Tomé & Príncipe": ("São Tomé & Príncipe", "island-hopping in the Gulf of Guinea"),
        "Yemen": ("Across the Bab-el-Mandeb", "from Africa into Arabia"),
        "Morocco": ("The Strait of Gibraltar", "from Europe into Africa"),
        "Sri Lanka": ("Sri Lanka", "across the Palk Strait"),
        "Singapore": ("Singapore", "its 163 m high point, Bukit Timah, would do"),
        "Australia": ("Australia", "island by island across the Torres Strait"),
        "Japan": ("Japan", "across the strait from Sakhalin"),
        "Cyprus": ("Cyprus", "seen across the sea from Turkey"),
        "Malta": ("Malta", "seen from Sicily"),
    }

    def __init__(self):
        self.g = G.Globe16()
        T = json.load(open(os.path.join(DATA, "tour.json")))
        self.T = T
        self.lat = np.array(T["lat"]); self.lon = np.array(T["lon"])
        cj = json.load(open(os.path.join(DATA, "countries.json")))
        self.names, counts = cj["names"], cj["counts"]
        cf = T.get("counts_for", T["cid"])
        n = len(self.lat)
        seen, self.new_at = set(), {}
        for i in range(n):
            c = cf[i]
            if c >= 0 and counts[c] and c not in seen:
                seen.add(c); self.new_at[i] = c
        self.n_countries = len(seen)
        self.reach_set = seen
        # timeline
        t = np.zeros(n)
        cur = self.INTRO
        self.notes = {}
        for i in range(n):
            if i:
                cur += 0.42 if i in self.new_at else 0.075
                if (i - 1) in self.notes:
                    cur += 2.0
            t[i] = cur
            if i in self.new_at and self.names[self.new_at[i]] in self.NOTES:
                self.notes[i] = self.NOTES[self.names[self.new_at[i]]]
        self.notes[0] = (f"Start: {self.names[self.new_at.get(0, cf[0])]}", "")
        self.t_lit = t
        self.t_end = t[-1]
        self.dur = self.t_end + self.OUTRO
        self.hop_km = np.r_[0, [gc_km((self.lat[i - 1], self.lon[i - 1]), (self.lat[i], self.lon[i])) for i in range(1, n)]]
        # smoothed camera path (unit vectors per frame)
        fr_t = np.arange(int(self.dur * FPS) + 2) / FPS
        k = np.clip(np.searchsorted(t, fr_t, side="right") - 1, 0, n - 1)
        u = np.clip((fr_t - t[k]) / np.maximum(np.r_[np.diff(t), 1][k], 1e-3), 0, 1)
        P = unit(self.lat, self.lon)
        k2 = np.minimum(k + 1, n - 1)
        tgt = P[k] * (1 - u[:, None]) + P[k2] * u[:, None]
        sig = int(1.1 * FPS)
        ker = np.exp(-np.arange(-3 * sig, 3 * sig + 1) ** 2 / (2 * sig * sig)); ker /= ker.sum()
        pad = np.pad(tgt, ((3 * sig, 3 * sig), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(3)], 1)
        self.cam_v = sm / np.linalg.norm(sm, axis=1, keepdims=True)
        # country outlines for tints
        names10, geoms, _ = __import__("world").countries()
        self.polys = {}
        for g in set(self.new_at.values()) | set(self.unreachable_ids(geoms)):
            gg = geoms[g].simplify(0.03)
            parts = [gg] if gg.geom_type == "Polygon" else list(gg.geoms)
            self.polys[g] = [np.asarray(pp.exterior.coords) for pp in parts if pp.area > 0.0005 or len(parts) < 4]
        print(f"tour {n} fires, {self.n_countries} countries, {self.dur:.1f} s", flush=True)

    def unreachable_ids(self, geoms):
        cj = json.load(open(os.path.join(DATA, "countries.json")))
        return [g for g in range(len(cj["names"])) if cj["counts"][g] and g not in self.reach_set]

    def camera(self, t):
        if t < self.INTRO:
            u = smooth((t - 9.0) / 6.0)
            v0 = unit(15.0, 20.0 + 6 * t)
            v1 = self.cam_v[0]
            v = v0 * (1 - u) + v1 * u
            v /= np.linalg.norm(v)
            rpx = 520 * (1 - u) + 1250 * u
        elif t > self.t_end + 0.5:
            u = smooth((t - self.t_end - 0.5) / 7.0)
            v = self.cam_v[min(int(t * FPS), len(self.cam_v) - 1)]
            rpx = 1250 * (1 - u) + 500 * u
            lat0, lon0 = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
            return lat0 * (1 - u) + 12 * u, lon0 - 60 * u, rpx
        else:
            v = self.cam_v[min(int(t * FPS), len(self.cam_v) - 1)]
            rpx = 1250
        return math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0])), rpx

    def tint(self, img, lat0, lon0, rpx, ids, col, alpha):
        if not ids:
            return img
        c, e, nn = self.g.basis(lat0, lon0)
        mask = np.zeros((H, W), np.float32)
        for g in ids:
            for ring in self.polys.get(g, []):
                P = unit(ring[:, 1], ring[:, 0])
                z = P @ c
                if (z < 0).all():
                    continue
                x = P @ e; y = P @ nn
                r = np.hypot(x, y)
                back = z < 0
                x = np.where(back, x / np.maximum(r, 1e-9), x); y = np.where(back, y / np.maximum(r, 1e-9), y)
                pts = np.c_[W / 2 + x * rpx, H / 2 - y * rpx]
                if (pts[:, 0] < -W).all() or (pts[:, 0] > 2 * W).all() or (pts[:, 1] < -H).all() or (pts[:, 1] > 2 * H).all():
                    continue
                pts = np.clip(pts, -4 * W, 5 * W)
                cv2.fillPoly(mask, [(pts * 16).astype(np.int32)], float(alpha.get(g, 1.0)) if isinstance(alpha, dict) else float(alpha),
                             cv2.LINE_AA, shift=4)
        return img * (1 - mask[..., None] * 0.55) + mask[..., None] * np.array(col, np.float32) * 0.55

    def frame(self, t):
        lat0, lon0, rpx = self.camera(t)
        img = self.g.render(lat0, lon0, rpx)
        # unreachable countries (red) during the intro and at the end
        unreach = self.unreachable_ids(None)
        a_red = max(ST.fade(t, 4.0, self.INTRO - 0.5, 1.0, 1.0), ST.fade(t, self.t_end + 1.0, self.dur + 5, 1.5, 1.0))
        if a_red > 0:
            img = self.tint(img, lat0, lon0, rpx, unreach, (0.75, 0.16, 0.12), a_red)
        lit = self.t_lit <= t
        visited = {c: self.t_lit[i] for i, c in self.new_at.items() if lit[i]}
        if visited:
            al = {c: min(1.0, 0.45 + 0.9 * math.exp(-(t - tv) / 0.5)) for c, tv in visited.items()}
            img = self.tint(img, lat0, lon0, rpx, list(visited), (0.95, 0.68, 0.25), al)
        X, Y, Z = self.g.project(self.lat, self.lon, lat0, lon0, rpx)
        Ly = G.Layer()
        idx = np.nonzero(lit)[0]
        for i in idx[1:]:
            if Z[i] > 0.03 and Z[i - 1] > 0.03:
                Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 0.5)
        for i in idx:
            if Z[i] < 0.02 or not (-40 < X[i] < W + 40 and -40 < Y[i] < H + 40):
                continue
            age = t - self.t_lit[i]
            amp = (0.8 + 0.15 * math.sin(t * 9 + i)) * (1 + 2.5 * math.exp(-age / 0.2))
            Ly.fire(X[i], Y[i], amp * 0.7, 0.8)
        fr = Ly.compose(img)
        return self.hud(fr, t, lit)

    def hud(self, fr, t, lit):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        k = int(np.nonzero(lit)[0][-1]) if lit.any() else -1
        n_c = sum(1 for i in self.new_at if lit[i])
        a_h = ST.fade(t, self.INTRO - 1.0, self.dur + 5, 0.6, 0.6)
        if a_h > 0:
            x0, y0 = 40, 850
            d.rounded_rectangle([x0, y0, x0 + 600, y0 + 190], 22, fill=ST.PANEL + (int(200 * a_h),))
            d.text((x0 + 30, y0 + 20), "COUNTRIES", font=ST.font(22, "SemiBold"), fill=ST.MUTED + (int(255 * a_h),), anchor="la")
            d.text((x0 + 24, y0 + 44), f"{n_c}", font=ST.font(92, "Bold"), fill=ST.GOLD + (int(255 * a_h),), anchor="la", features=["tnum"])
            d.text((x0 + 24 + d.textlength(f"{n_c}", font=ST.font(92, "Bold")) + 10, y0 + 92), f"/ {self.n_countries}",
                   font=ST.font(40, "SemiBold"), fill=ST.INK2 + (int(255 * a_h),), anchor="la")
            d.text((x0 + 330, y0 + 20), "FIRES", font=ST.font(22, "SemiBold"), fill=ST.MUTED + (int(255 * a_h),), anchor="la")
            d.text((x0 + 324, y0 + 44), f"{max(k + 1, 0)}", font=ST.font(92, "Bold"), fill=(255, 214, 150, int(255 * a_h)), anchor="la", features=["tnum"])
            km = float(self.hop_km[:k + 1].sum()) if k >= 0 else 0
            d.text((x0 + 30, y0 + 150), f"{km:,.0f} km of sightlines", font=ST.font(26, "Regular"), fill=ST.INK2 + (int(255 * a_h),), anchor="la")
            # the last few countries reached
            recent = [(i, c) for i, c in sorted(self.new_at.items()) if lit[i]][-6:]
            for j, (i, c) in enumerate(reversed(recent)):
                age = t - self.t_lit[i]
                a = a_h * max(0.25, 1 - j * 0.15) * min(1, age / 0.25)
                y = 900 - j * 46 + (0 if age > 0.25 else 20 * (1 - age / 0.25))
                d.text((W - 60, y), self.names[c], font=ST.font(34 if j == 0 else 28, "SemiBold"),
                       fill=(ST.GOLD if j == 0 else ST.INK) + (int(255 * a),), anchor="ra")
            d.text((40, 36), "ONE CHAIN, AS MANY COUNTRIES AS POSSIBLE", font=ST.font(22, "SemiBold"), fill=ST.MUTED + (int(255 * a_h),), anchor="la")
        caps = [(0.5, 8.8, [("Part 2: one chain of fires", 50, ST.INK, "Bold"),
                            ("through as many of the world's 195 countries as possible", 34, ST.INK2, "Regular")], {"anchor": "top"}),
                (9.2, self.INTRO - 0.3, [(f"{195 - self.n_countries} countries are out of reach of any fire", 44, ST.RED, "Bold"),
                                         ("Iceland, New Zealand, Madagascar, Cabo Verde, Mauritius and the Pacific islands", 30, ST.INK2, "Regular")],
                 {"anchor": "top"})]
        for i, (a, b) in self.notes.items():
            lines = [(a, 42, ST.GOLD, "Bold")] + ([(b, 30, ST.INK2, "Regular")] if b else [])
            caps.append((self.t_lit[i], self.t_lit[i] + 2.3, lines, {"anchor": "top"}))
        caps.append((self.t_end + 0.8, self.dur + 5, [(f"{self.n_countries} countries, {len(self.lat)} fires", 52, ST.GOLD, "Bold"),
                                                       (f"{self.names[self.T['counts_for'][0]] if 'counts_for' in self.T else ''} to {self.names[self.new_at[max(self.new_at)]]}, "
                                                        f"{float(self.hop_km.sum()):,.0f} km of sightlines", 32, ST.INK, "SemiBold")],
                     {"anchor": "top"}))
        for c in caps:
            a = ST.fade(t, c[0], c[1])
            if a > 0:
                ST.caption(d, c[2], a, **c[3])
        out = np.asarray(im)
        f = smooth(t / 0.6) * (1 - smooth((t - self.dur + 0.8) / 0.8))
        return (out * f).astype(np.uint8) if f < 1 else out


# ====================================================================== outro
class Outro(Section):
    name, dur = "outro", 20.0

    def __init__(self):
        self.g = G.Globe16()
        L = json.load(open(os.path.join(DATA, "loop.json")))
        st = L["stations"]
        self.la1 = np.array([x["lat"] for x in st]); self.lo1 = np.array([x["lon"] for x in st])
        self.kind = [x["kind"] for x in st]
        T = json.load(open(os.path.join(DATA, "tour.json")))
        self.la2 = np.array(T["lat"]); self.lo2 = np.array(T["lon"])
        cj = json.load(open(os.path.join(DATA, "countries.json")))
        cf = T.get("counts_for", T["cid"])
        self.n_c = len({c for c in cf if c >= 0 and cj["counts"][c]})
        self.n_fire = sum(1 for k in self.kind[:-1] if k == "fire"); self.n_ship = self.kind.count("ship")

    def frame(self, t):
        lat0, lon0, rpx = 28.0, 10 + 9 * t, 470
        img = self.g.render(lat0, lon0, rpx, cy=630)
        Ly = G.Layer()
        for la, lo, col, amp in ((self.la2, self.lo2, (1.0, 0.78, 0.35), 0.35), (self.la1, self.lo1, (1.0, 0.5, 0.12), 0.55)):
            X, Y, Z = self.g.project(la, lo, lat0, lon0, rpx, cy=630)
            for i in range(1, len(la)):
                if Z[i] > 0.02 and Z[i - 1] > 0.02:
                    Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), amp)
            for i in range(len(la)):
                if Z[i] > 0.02:
                    Ly.blit(X[i], Y[i], 0.9, amp * 1.2, col)
        fr = Ly.compose(img)
        caps = [(1.0, self.dur, [(f"Around the world: {self.n_fire} fires + {self.n_ship} fire ships", 40, (255, 160, 70), "Bold"),
                                 (f"Through {self.n_c} countries: {len(self.la2)} fires", 40, ST.GOLD, "Bold"),
                                 ("The Earth's curve keeps 18 countries out of reach", 30, ST.INK2, "Regular")], {"anchor": "top"})]
        fr = ST.draw_captions(fr, t, caps)
        f = smooth(t / 0.8) * (1 - smooth((t - self.dur + 2.0) / 2.0))
        return (fr * f).astype(np.uint8) if f < 1 else fr


SECTIONS = {"hook": Hook, "history": History, "journey": Journey, "rules": Rules, "tour": Tour, "outro": Outro}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
