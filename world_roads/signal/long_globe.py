"""Globe sections of the two long videos (16:9), rendered through longform.Section.
  Video 1 (around the world): hook, history, rules, journey, outro
  Video 2 (if an AI took over every phone, how fast could fire warn every capital?): cold, plan, wave, outro2

    python long_globe.py journey            -> out/long/journey.mp4
    python long_globe.py journey 40 80.5    -> stills
"""
import os, sys, json, math
import numpy as np
import cv2
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
    title = "THE CHALLENGE"
    rules = [("Start in Plymouth,", "where England's Armada beacons were lit"),
             ("Fires only on real mountaintops", "a bonfire 10 m tall on the summit"),
             ("Each fire must see the next one", "over the curve of the Earth and every ridge between"),
             ("All the way around the world", "and back to Plymouth"),
             ("As few fires as possible", "")]
    focus, lit = PLYMOUTH, True

    def __init__(self):
        self.g = G.Globe16()

    def frame(self, t):
        u = smooth(t / self.dur)
        lat0, lon0, rpx = 47.0, self.focus[1] + 18 * (1 - u), 1100 + 500 * u
        cx = W * 0.70
        img = self.g.render(lat0, lon0, rpx, cx=cx)
        X, Y, Z = self.g.project(np.array([self.focus[0]]), np.array([self.focus[1]]), lat0, lon0, rpx, cx=cx)
        self.focus_xy = (float(X[0]), float(Y[0]))
        Ly = G.Layer()
        if self.lit:
            Ly.fire(X[0], Y[0], 1.0 + 0.15 * math.sin(t * 9), 1.3)
        else:                                                         # the pyre, built and waiting
            Ly.blit(X[0], Y[0], 1.3, 0.5, (0.6, 0.65, 0.8))
        fr = Ly.compose(img)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([70, 120, 900, 960], 28, fill=ST.PANEL + (215,))
        a0 = ST.fade(t, 0.3, self.dur, 0.5, 0.6)
        d.text((120, 170), self.title, font=ST.font(30, "Bold"), fill=ST.GOLD + (int(255 * a0),), anchor="la")
        rules = self.rules
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


# ====================================================================== part 2: the analog backup
FAMOUS = {"London", "Paris", "Berlin", "Madrid", "Rome", "Moscow", "Cairo", "Beijing", "Tokyo", "New Delhi", "Washington, D.C.",
          "Ottawa", "Mexico City", "Brasília", "Buenos Aires", "Canberra", "Pretoria", "Nairobi", "Abuja", "Riyadh", "Tehran",
          "Ankara", "Kyiv", "Seoul", "Jakarta", "Bangkok", "Lima", "Bogotá", "Santiago", "Kinshasa", "Addis Ababa", "Islamabad",
          "Ulaanbaatar", "Hanoi", "Manila", "Havana", "Caracas", "Athens", "Stockholm", "Warsaw", "Vatican City", "Kabul"}
CALLOUTS = ["London", "Paris", "Moscow", "Cairo", "New Delhi", "Beijing", "Tokyo", "Washington, D.C.", "Mexico City",
            "Brasília", "Canberra"]


def glitch(t, seed=0):
    """'NO SIGNAL': TV static with torn, colour-split bands."""
    rng = np.random.default_rng(int(t * 30) + seed)
    small = (rng.random((H // 3, W // 3)) * 255).astype(np.uint8)
    st = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32) * 0.55
    img = np.repeat(st[..., None], 3, -1)
    for _ in range(7):
        y0 = int(rng.integers(0, H - 40)); hh = int(rng.integers(6, 60)); sh = int(rng.integers(-120, 120))
        img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], sh, axis=1)
    img[..., 0] = np.roll(img[..., 0], 6, axis=1); img[..., 2] = np.roll(img[..., 2], -6, axis=1)
    img[::3] *= 0.75
    fr = np.clip(img, 0, 255).astype(np.uint8)
    im = Image.fromarray(fr)
    d = ImageDraw.Draw(im, "RGBA")
    jx = int(rng.integers(-6, 7))
    d.rectangle([W / 2 - 330, H / 2 - 80, W / 2 + 330, H / 2 + 80], fill=(0, 0, 0, 215))
    d.text((W / 2 + jx, H / 2), "NO SIGNAL", font=ST.font(96, "Bold"), fill=(240, 240, 240, 255), anchor="mm")
    return np.asarray(im)


STRAITS = [("the Strait of Gibraltar", 35.95, -5.6), ("the Bosporus", 41.1, 29.05), ("the Dardanelles", 40.2, 26.4),
           ("the Strait of Sicily", 37.2, 11.3), ("the Bab-el-Mandeb", 12.6, 43.4), ("the Strait of Hormuz", 26.6, 56.3),
           ("the Bering Strait", 65.8, -169.0), ("the Darién", 8.0, -77.3), ("the Torres Strait", -10.2, 142.2),
           ("the Suez Canal", 30.5, 32.4), ("the Caribbean islands", 11.5, -61.5)]


def continent(cname, lat, lon, gj_cont):
    """Continent of a fire: Natural Earth's, by country, with Russia split at the Urals, European Turkey and the Sinai."""
    if cname == "Russia":
        return "Asia" if lon > 60.0 else "Europe"
    if cname == "Turkey" and lon < 29.2 and lat > 40.0:
        return "Europe"
    if cname == "Egypt" and lon > 32.4 and lat < 31.4:
        return "Asia"
    return gj_cont


def crossings(F):
    """[(relay, (continent, 'across the Strait of Gibraltar' / 'from Israel'))]: where the fire first reaches each continent."""
    names = json.load(open(os.path.join(DATA, "countries.json")))["names"]
    gj = json.load(open(os.path.join(ROOT, "data", "ne", "ne_10m_admin_0_countries.geojson")))
    gc = [f["properties"]["CONTINENT"] for f in gj["features"]]
    cont = [continent(names[f["cid"]], f["lat"], f["lon"], gc[f["cid"]]) if f["cid"] >= 0 else None for f in F]
    seen, out = {cont[0]}, []
    for i in sorted(range(len(F)), key=lambda i: F[i]["hop"]):
        c, p = cont[i], F[i]["parent"]
        if c in seen or c not in ("Africa", "Asia", "North America", "South America", "Oceania", "Europe") or p < 0:
            continue
        seen.add(c)
        mid = slerp_ll((F[p]["lat"], F[p]["lon"]), (F[i]["lat"], F[i]["lon"]), 0.5)
        near = [(gc_km(mid, (a, b)), nm) for nm, a, b in STRAITS]
        dk, nm = min(near)
        if names[F[p]["cid"]] == names[F[i]["cid"]] == "Russia":
            how = "across the Urals"
        else:
            how = f"across {nm}" if dk < 300 else f"from {names[F[p]['cid']]}"
        out.append((int(F[i]["hop"]), (c, how)))
    return out


class Alert(Section):
    """If an AI took over every phone, cable, satellite and radio: how long to warn every capital by fire?"""
    name = "alert"
    GLITCH, SETUP_END, START_END = 2.6, 15.5, 25.0
    END_HOLD = 17.0

    def __init__(self):
        self.g = G.Globe16()
        T = json.load(open(os.path.join(DATA, "capitals_tree.json")))
        st = json.load(open(os.path.join(DATA, "capitals_start.json")))
        self.start = (st["lat"], st["lon"])
        F = T["fires"]
        self.lat = np.array([f["lat"] for f in F]); self.lon = np.array([f["lon"] for f in F])
        self.hop = np.array([f["hop"] for f in F]); self.par = np.array([f["parent"] for f in F])
        self.caps = T["capitals"]; self.unreach = T["unreachable"]
        self.mpr = T["min_per_relay"]
        self.max_hop = int(self.hop.max())
        # relay k is lit at time tk(k): slow while the camera is close on Britain, then steady
        self.INTRO = self.START_END
        k = np.arange(self.max_hop + 2)
        self.tk = self.INTRO + np.where(k < 10, k * 0.9, 9.0 + (k - 10) * 0.5)
        self.t_end = float(self.tk[self.max_hop])
        self.t_out = self.t_end + self.END_HOLD                    # end, in this section's own time
        self.dur = self.t_out
        self.t_fire = self.tk[self.hop]
        self.km = float(sum(gc_km((self.lat[i], self.lon[i]), (self.lat[p], self.lon[p])) for i, p in enumerate(self.par) if p >= 0))
        last = self.caps[-1]
        self.last = last
        # captions during the wave: the fire reaching each new continent first, then headline capitals,
        # never closer than 3.4 s to each other
        self.crossings, t_prev = [], -10.0                   # [(caption time, relay, (continent, how))], queued
        for k, info in crossings(F):
            ts = max(float(self.tk[k]), t_prev + 3.0)
            self.crossings.append((ts, k, info)); t_prev = ts
        placed = [ts for ts, _, _ in self.crossings] + [self.t_end]
        byname = {c["capital"]: c for c in self.caps}
        self.callouts = []                                    # [(time, [capitals warned on that relay])]
        for nm in CALLOUTS:
            c = byname.get(nm)
            if c is None or any(c in g for _, g in self.callouts):
                continue
            tc = self.tk[c["hop"]]
            if all(abs(tc - q) >= 3.4 for q in placed) and tc < self.t_end - 4:
                same = [byname[o] for o in CALLOUTS if o in byname and o != nm and byname[o]["hop"] == c["hop"]]
                self.callouts.append((tc, [c] + same[:1])); placed.append(tc)
        self.callouts.sort(key=lambda x: x[0])
        # camera: centre of the fires lit in the last ~20 relays, smoothed
        fr_t = np.arange(int(self.t_out * FPS) + 2) / FPS
        P = unit(self.lat, self.lon)
        tgt = np.zeros((len(fr_t), 3))
        S0 = unit(*self.start)
        for j, t in enumerate(fr_t):
            kk = np.searchsorted(self.tk, t, side="right") - 1
            if kk < 1:
                tgt[j] = S0; continue
            m = (self.hop <= kk) & (self.hop > kk - 20)
            wts = np.exp(-(kk - self.hop[m]) / 8.0)
            v = (P[m] * wts[:, None]).sum(0)
            tgt[j] = v / max(np.linalg.norm(v), 1e-9)
        sig = int(1.6 * FPS)
        ker = np.exp(-np.arange(-3 * sig, 3 * sig + 1) ** 2 / (2 * sig * sig)); ker /= ker.sum()
        pad = np.pad(tgt, ((3 * sig, 3 * sig), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(3)], 1)
        self.cam_v = sm / np.linalg.norm(sm, axis=1, keepdims=True)
        print(f"alert: {len(F)} fires, {len(self.caps)} capitals, last {last['capital']} at {last['hop']} relays, "
              f"{self.t_out:.1f} s", flush=True)

    def clock(self, t):
        if t < self.INTRO:
            return 0.0
        k = np.searchsorted(self.tk, t, side="right") - 1
        k = min(k, self.max_hop)
        if k < self.max_hop:
            k = k + min(1.0, (t - self.tk[k]) / (self.tk[k + 1] - self.tk[k]))
        return k * self.mpr

    def camera(self, t):
        if t < self.SETUP_END:
            return 25.0, -20 + 8 * t, 520
        if t < self.INTRO:
            u = smooth((t - self.SETUP_END) / 4.0)
            la, lo = slerp_ll((25.0, -20 + 8 * self.SETUP_END), self.start, u)
            return la, lo, math.exp(math.log(520) + (math.log(2600) - math.log(520)) * u)
        v = self.cam_v[min(int(t * FPS), len(self.cam_v) - 1)]
        la, lo = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
        k = (t - self.INTRO) / max(self.t_end - self.INTRO, 1)
        rpx = 2600 * math.exp(-(t - self.INTRO) / 6.0) + 680 * (1 - math.exp(-(t - self.INTRO) / 6.0))
        if t > self.t_end:
            u = smooth((t - self.t_end - 1.0) / 6.0)
            rpx = rpx * (1 - u) + 470 * u
            la = la * (1 - u) + 15 * u
            lo = lo - 70 * u
        return la, lo, rpx

    def frame(self, t):
        if t < self.GLITCH:
            fr = glitch(t)
            if t > self.GLITCH - 0.4:
                fr = (fr * (1 - smooth((t - self.GLITCH + 0.4) / 0.4))).astype(np.uint8)
            return fr
        lat0, lon0, rpx = self.camera(t)
        img = self.g.render(lat0, lon0, rpx)
        X, Y, Z = self.g.project(self.lat, self.lon, lat0, lon0, rpx)
        Ly = G.Layer()
        lit = self.t_fire <= t
        sc = float(np.clip(rpx / 1750, 0.45, 1.3))
        for i in np.nonzero(lit)[0]:
            p = self.par[i]
            if p >= 0 and Z[i] > 0.02 and Z[p] > 0.02:
                Ly.line((X[p], Y[p]), (X[i], Y[i]), 0.45)
        for i in np.nonzero(lit)[0]:
            if Z[i] < 0.02 or not (-40 < X[i] < W + 40 and -40 < Y[i] < H + 40):
                continue
            age = t - self.t_fire[i]
            amp = (0.75 + 0.15 * math.sin(t * 9 + i)) * (1 + 2.5 * math.exp(-age / 0.25))
            Ly.fire(X[i], Y[i], amp * 0.7, sc)
        if self.INTRO - 3.5 < t < self.INTRO:                       # Bletchley Park waits, pyre ready
            Xs, Ys, _ = self.g.project(np.array([self.start[0]]), np.array([self.start[1]]), lat0, lon0, rpx)
            Ly.blit(Xs[0], Ys[0], 1.3, 0.5, (0.6, 0.65, 0.8))
        fr = Ly.compose(img)
        return self.hud(fr, t, lat0, lon0, rpx)

    def hud(self, fr, t, lat0, lon0, rpx):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        c_e, e_e, n_e = self.g.basis(lat0, lon0)
        clock_min = self.clock(t)

        def proj(la, lo):
            p = unit(la, lo)
            return W / 2 + (p @ e_e) * rpx, H / 2 - (p @ n_e) * rpx, p @ c_e
        # capitals: rings (waiting), gold flash (warned), red (unreachable)
        a_caps = ST.fade(t, self.START_END - 1.5, self.t_out + 5, 1.0, 0.5)
        for c in self.caps:
            x, y, z = proj(c["lat"], c["lon"])
            if z < 0.05 or not (0 < x < W and 0 < y < H) or a_caps <= 0:
                continue
            tc = self.tk[c["hop"]]
            if t < tc:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(220, 228, 240, int(150 * a_caps)), width=2)
                continue
            age = t - tc
            r = 6 + 16 * math.exp(-age / 0.35)
            d.ellipse([x - r, y - r, x + r, y + r], outline=ST.GOLD + (int(255 * a_caps * min(1, 0.45 + math.exp(-age / 0.6))),), width=3)
            if c["capital"] in FAMOUS and age < 3.0:
                a = a_caps * min(1, age / 0.2) * min(1, (3.0 - age) / 0.5)
                txt = f"{c['capital']}  {int(c['hours'])} h {int(round(c['hours'] % 1 * 60)):02d}"
                f = ST.font(24, "SemiBold")
                tw = d.textlength(txt, font=f)
                d.rounded_rectangle([x + 12, y - 16, x + 26 + tw, y + 16], 8, fill=ST.PANEL + (int(180 * a),))
                d.text((x + 19, y), txt, font=f, fill=ST.INK + (int(255 * a),), anchor="lm")
        a_red = ST.fade(t, self.t_end + 1.0, self.t_out + 5, 0.8, 0.5)
        for c in self.unreach:
            x, y, z = proj(c["lat"], c["lon"])
            if a_red > 0 and z > 0.05 and 0 < x < W and 0 < y < H:
                col = ST.RED + (int(255 * a_red),)
                d.line([(x - 8, y - 8), (x + 8, y + 8)], fill=col, width=3); d.line([(x - 8, y + 8), (x + 8, y - 8)], fill=col, width=3)
        # the HUD
        a_h = ST.fade(t, self.START_END - 1.0, self.t_out + 5, 0.6, 0.5)
        if a_h > 0:
            d.text((40, 30), "NO PHONES  ·  NO INTERNET  ·  NO RADIO", font=ST.font(22, "SemiBold"), fill=ST.RED + (int(230 * a_h),), anchor="la")
            blink = 0.5 + 0.5 * math.sin(t * 5)
            d.ellipse([42, 70, 56, 84], fill=(255, 150, 60, int(255 * a_h * blink)))
            d.text((66, 77), "NETWORK: FIRE", font=ST.font(22, "SemiBold"), fill=ST.GOLD + (int(230 * a_h),), anchor="lm")
            x0, y0 = 40, 830
            d.rounded_rectangle([x0, y0, x0 + 760, y0 + 210], 22, fill=ST.PANEL + (int(205 * a_h),))
            hh, mm = int(clock_min // 60), int(clock_min % 60)
            d.text((x0 + 30, y0 + 20), "TIME SINCE THE FIRST FIRE", font=ST.font(22, "SemiBold"), fill=ST.MUTED + (int(255 * a_h),), anchor="la")
            d.text((x0 + 24, y0 + 46), f"T+ {hh:02d}:{mm:02d}", font=ST.font(96, "Bold"), fill=ST.GOLD + (int(255 * a_h),), anchor="la", features=["tnum"])
            n_c = sum(1 for c in self.caps if self.tk[c["hop"]] <= t)
            d.text((x0 + 470, y0 + 20), "CAPITALS WARNED", font=ST.font(22, "SemiBold"), fill=ST.MUTED + (int(255 * a_h),), anchor="la")
            d.text((x0 + 466, y0 + 46), f"{n_c}", font=ST.font(96, "Bold"), fill=ST.INK + (int(255 * a_h),), anchor="la", features=["tnum"])
            d.text((x0 + 470, y0 + 160), f"of {len(self.caps)} reachable", font=ST.font(26, "Regular"), fill=ST.INK2 + (int(255 * a_h),), anchor="la")
            n_f = int((self.t_fire <= t).sum())
            d.text((x0 + 30, y0 + 160), f"{n_f:,} bonfires lit", font=ST.font(26, "Regular"), fill=ST.INK2 + (int(255 * a_h),), anchor="la")
            # board: the latest capitals warned
            recent = [c for c in self.caps if self.tk[c["hop"]] <= t][-7:]
            for j, c in enumerate(reversed(recent)):
                age = t - self.tk[c["hop"]]
                a = a_h * max(0.25, 1 - j * 0.13) * min(1, age / 0.25)
                y = 980 - j * 44
                txt = f"{c['capital']}   {int(c['hours'])} h {int(round(c['hours'] % 1 * 60)):02d}"
                d.text((W - 50, y), txt, font=ST.font(30 if j == 0 else 26, "SemiBold"), fill=(ST.GOLD if j == 0 else ST.INK) + (int(255 * a),), anchor="ra")
        caps = [
            (2.9, 8.9, [("Imagine an AI takes over every phone,", 44, ST.INK, "SemiBold"),
                        ("every satellite, every cable and every radio on Earth", 44, ST.INK, "SemiBold")], {}),
            (9.2, 15.2, [("How fast could we warn every country", 44, ST.INK, "SemiBold"),
                         ("with the one network it can't hack?", 44, ST.GOLD, "Bold")], {}),
            (15.6, 20.4, [("The first fire: Bletchley Park", 46, ST.GOLD, "Bold"),
                          ("home of Colossus, the first programmable electronic computer,", 30, ST.INK2, "Regular"),
                          ("and of the first AI Safety Summit. Fitting.", 30, ST.INK2, "Regular")], {}),
            (20.6, self.INTRO + 2.5, [("Every keeper knows the plan:", 40, ST.INK, "SemiBold"),
                                      ("see a fire, unplug everything, light your own", 40, ST.GOLD, "Bold"),
                                      ("5 minutes per relay · the clock stops when the last capital sees a fire", 28, ST.INK2, "Regular")], {}),
        ]
        for tc, g in self.callouts:
            h = g[0]["hours"]
            who = " and ".join(c["capital"] for c in g) + (" are" if len(g) > 1 else " is")
            caps.append((tc, tc + 2.6, [(f"{who} warned", 44, ST.GOLD, "Bold"),
                                        (f"after {int(h)} h {int(round(h % 1 * 60)):02d} min", 30, ST.INK2, "Regular")], {"anchor": "top"}))
        for ts, k, (cont, how) in self.crossings:
            m = k * self.mpr
            caps.append((ts, ts + 2.8, [(f"The fire reaches {cont}", 44, ST.INK, "Bold"),
                                        (f"{how} · T+ {int(m // 60)} h {int(m % 60):02d}", 30, ST.INK2, "Regular")], {"anchor": "top"}))
        L = self.last
        caps.append((self.t_end - 0.2, self.t_end + 4.6, [(f"The last capital: {L['capital']}, {L['country']}", 44, ST.GOLD, "Bold"),
                                                          (f"after {int(L['hours'])} h {int(round(L['hours'] % 1 * 60)):02d} min", 30, ST.INK2, "Regular")],
                     {"anchor": "top"}))
        nu = len(self.unreach)
        names_u = [c["country"] for c in self.unreach]
        pick = [n for n in ("Iceland", "New Zealand", "Sri Lanka", "Madagascar") if n in names_u]
        caps.append((self.t_end + 5.0, self.t_end + 10.0, [(f"{nu} capitals never get the message", 44, ST.RED, "Bold"),
                                                           (", ".join(pick) + f" and {nu - len(pick)} more island nations: you're on your own", 30, ST.INK2, "Regular"),
                                                           ("no fire can see that far across the sea, over the curve of the Earth", 26, ST.MUTED, "Regular")],
                     {"anchor": "top"}))
        hh, mm = int(self.last["hours"]), int(round(self.last["hours"] % 1 * 60))
        caps.append((self.t_end + 10.4, self.t_out + 5, [(f"Every reachable capital warned in {hh} h {mm:02d} min", 46, ST.GOLD, "Bold"),
                                                       (f"{len(self.lat):,} bonfires · {self.km:,.0f} km of sightlines", 32, ST.INK, "SemiBold"),
                                                       ("Total bandwidth: one bit", 30, ST.INK2, "Regular")], {"anchor": "top"}))
        for c in caps:
            a = ST.fade(t, c[0], c[1])
            if a > 0:
                ST.caption(d, c[2], a, **c[3])
        out = np.asarray(im)
        f = smooth((t - self.GLITCH) / 0.5) * (1 - smooth((t - self.t_out + 0.8) / 0.8))
        return (out * f).astype(np.uint8) if f < 1 else out


class Cold(Alert):
    """Video 2's cold open: NO SIGNAL, then the premise over a spinning globe."""
    name = "cold"

    def __init__(self):
        super().__init__()
        self.t_out = self.dur = self.SETUP_END + 0.5


class Plan(Rules):
    name, dur = "plan", 16.0
    title = "THE BACKUP PLAN"
    rules = [("First fire: Bletchley Park", "where the first AI Safety Summit met in 2023"),
             ("Fires on real hilltops", "10 m bonfires on summits, plus one in every capital"),
             ("Each fire must see a burning one", "over the curve of the Earth and every ridge between"),
             ("5 minutes to react", "from seeing a fire to lighting your own"),
             ("No phones, no radio, no boats", "the clock stops when the last capital sees a fire")]
    lit = False

    def __init__(self):
        super().__init__()
        st = json.load(open(os.path.join(DATA, "capitals_start.json")))
        self.focus = (st["lat"], st["lon"])

    def frame(self, t):
        fr = Rules.frame(self, t)
        a = ST.fade(t, 1.0, self.dur - 0.7, 0.5, 0.5)
        return ST.ring_label(fr, *self.focus_xy, "Bletchley Park", a) if a > 0 else fr


class Wave(Alert):
    """Video 2's main event: from the flight to Bletchley Park to the last capital."""
    name = "wave"

    def __init__(self):
        super().__init__()
        self.LEAD = self.SETUP_END
        self.dur = self.t_out - self.LEAD

    def frame(self, t):
        fr = Alert.frame(self, t + self.LEAD)
        f = smooth(t / 0.6)
        return (fr * f).astype(np.uint8) if f < 1 else fr


# ====================================================================== outro
class Outro(Section):
    """Video 1's ending: the loop turning on the globe, then a teaser for Video 2."""
    name, dur = "outro", 20.0

    def __init__(self):
        self.g = G.Globe16()
        L = json.load(open(os.path.join(DATA, "loop.json")))
        st = L["stations"]
        self.la1 = np.array([x["lat"] for x in st]); self.lo1 = np.array([x["lon"] for x in st])
        self.kind = [x["kind"] for x in st]
        self.n_fire = sum(1 for k in self.kind[:-1] if k == "fire"); self.n_ship = self.kind.count("ship")

    def frame(self, t):
        lat0, lon0, rpx = 28.0, 10 + 9 * t, 470
        img = self.g.render(lat0, lon0, rpx, cy=630)
        Ly = G.Layer()
        X, Y, Z = self.g.project(self.la1, self.lo1, lat0, lon0, rpx, cy=630)
        for i in range(1, len(X)):
            if Z[i] > 0.02 and Z[i - 1] > 0.02:
                Ly.line((X[i - 1], Y[i - 1]), (X[i], Y[i]), 0.55)
        for i in range(len(X)):
            if Z[i] > 0.02:
                Ly.blit(X[i], Y[i], 0.9, 0.65, (1.0, 0.5, 0.12))
        fr = Ly.compose(img)
        caps = [(1.0, 7.8, [(f"Around the world: {self.n_fire} fires + {self.n_ship} fire ships", 46, (255, 160, 70), "Bold"),
                            ("every sightline checked against real terrain", 30, ST.INK2, "Regular")], {"anchor": "top"}),
                (8.3, self.dur, [("Next: an AI takes over every phone on Earth.", 44, ST.INK, "SemiBold"),
                                 ("How fast could bonfires warn every country?", 44, ST.GOLD, "Bold")], {"anchor": "top"})]
        fr = ST.draw_captions(fr, t, caps)
        f = smooth(t / 0.8) * (1 - smooth((t - self.dur + 2.0) / 2.0))
        return (fr * f).astype(np.uint8) if f < 1 else fr


class Outro2(Section):
    """Video 2's ending: the whole broadcast tree, the capitals it can't reach, and a link back to Video 1."""
    name, dur = "outro2", 20.0

    def __init__(self):
        self.g = G.Globe16()
        T = json.load(open(os.path.join(DATA, "capitals_tree.json")))
        F = T["fires"]
        self.la = np.array([f["lat"] for f in F]); self.lo = np.array([f["lon"] for f in F])
        self.par = np.array([f["parent"] for f in F])
        self.T = T
        self.km = float(sum(gc_km((self.la[i], self.lo[i]), (self.la[p], self.lo[p])) for i, p in enumerate(self.par) if p >= 0))

    def frame(self, t):
        lat0, lon0, rpx = 22.0, 10 + 9 * t, 470
        img = self.g.render(lat0, lon0, rpx, cy=630)
        Ly = G.Layer()
        X, Y, Z = self.g.project(self.la, self.lo, lat0, lon0, rpx, cy=630)
        for i, p in enumerate(self.par):
            if p >= 0 and Z[i] > 0.02 and Z[p] > 0.02:
                Ly.line((X[p], Y[p]), (X[i], Y[i]), 0.4)
        for i in range(len(X)):
            if Z[i] > 0.02:
                Ly.blit(X[i], Y[i], 0.9, 0.4, (1.0, 0.78, 0.35))
        fr = Ly.compose(img)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        c_e, e_e, n_e = self.g.basis(lat0, lon0)
        for c in self.T["capitals"]:
            p = unit(c["lat"], c["lon"])
            x, y = W / 2 + (p @ e_e) * rpx, 630 - (p @ n_e) * rpx
            if p @ c_e > 0.05:
                d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=ST.GOLD + (200,), width=2)
        for c in self.T["unreachable"]:
            p = unit(c["lat"], c["lon"])
            x, y = W / 2 + (p @ e_e) * rpx, 630 - (p @ n_e) * rpx
            if p @ c_e > 0.05:
                d.line([(x - 7, y - 7), (x + 7, y + 7)], fill=ST.RED + (230,), width=3)
                d.line([(x - 7, y + 7), (x + 7, y - 7)], fill=ST.RED + (230,), width=3)
        fr = np.asarray(im)
        last = self.T["capitals"][-1]
        hh, mm = int(last["hours"]), int(round(last["hours"] % 1 * 60))
        caps = [(1.0, 7.8, [(f"{len(self.T['capitals'])} capitals warned in {hh} h {mm:02d} min", 46, ST.GOLD, "Bold"),
                            (f"{len(self.la):,} bonfires · {self.km:,.0f} km of sightlines · zero electricity", 30, ST.INK2, "Regular")],
                 {"anchor": "top"}),
                (8.3, self.dur, [("Good luck hacking a bonfire.", 50, ST.GOLD, "Bold"),
                                 ("Next: one fire, all the way around the world", 32, ST.INK2, "Regular")], {"anchor": "top"})]
        fr = ST.draw_captions(fr, t, caps)
        f = smooth(t / 0.8) * (1 - smooth((t - self.dur + 2.0) / 2.0))
        return (fr * f).astype(np.uint8) if f < 1 else fr


SECTIONS = {"hook": Hook, "history": History, "journey": Journey, "rules": Rules, "alert": Alert, "outro": Outro,
            "cold": Cold, "plan": Plan, "wave": Wave, "outro2": Outro2}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
