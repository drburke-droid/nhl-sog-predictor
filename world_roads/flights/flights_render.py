"""Render "Every flight on Earth": a 24-hour day of flights, then one plane flying them all at light speed.

    python schedule.py && python itinerary.py
    python flights_render.py prepare
    python flights_render.py still 1 20.5      # part 1 at 20.5 h UTC   |  still 2 300  -> part 2 at 300 s
    python flights_render.py video --workers 3

Part 1 compresses 24 h (UTC) into DAY_S seconds. Part 2 runs in real time: a light-speed plane flies
every flight of that day back to back (190.8 M km -> 10 min 37 s). Same map, palette and type as the
road film.
"""
import os, sys, json, subprocess, argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from proj import to_px, W, H  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FPS = 30
DAY_S = 12.0                 # part 1 length
CARD_S = 2.5                 # transition card
END_S = 5.0                  # end card (room for the music to resolve)
HOOK_S = 2.6                 # opening hook line over the day map
C_KMS = 299_792.458
NPTS = 48
FD = "/usr/share/fonts/opentype/inter"
CYAN = np.array([0.14, 0.66, 1.00], np.float32)
GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.92
HOT = np.array([1.00, 0.93, 0.78], np.float32)
FLY = np.array([0.55, 0.85, 1.00], np.float32)


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


# ----------------------------------------------------------------------------------------- prepare
def slerp_paths(lat1, lon1, lat2, lon2, n=NPTS):
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    a, b = xyz(lat1, lon1), xyz(lat2, lon2)
    ang = np.arccos(np.clip((a * b).sum(-1), -1, 1))[:, None, None]
    t = np.linspace(0, 1, n)[None, :, None]
    s = np.sin(ang)
    p = np.where(s > 1e-9, (np.sin((1 - t) * ang) * a[:, None] + np.sin(t * ang) * b[:, None]) / np.where(s > 1e-9, s, 1),
                 a[:, None] + 0 * t)
    la = np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1)))
    lo = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return la, lo


def prepare():
    S = dict(np.load(os.path.join(DATA, "schedule.npz")))
    I = dict(np.load(os.path.join(DATA, "itinerary.npz")))
    lat, lon = S["lat"], S["lon"]
    src, dst = S["src"], S["dst"]
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    code = a * 100000 + b
    pairs, pid = np.unique(code, return_inverse=True)
    pa, pb = pairs // 100000, pairs % 100000
    la, lo = slerp_paths(lat[pa], lon[pa], lat[pb], lon[pb])
    px, py = to_px(lo, la)
    P = np.stack([px, py], -1).astype(np.float32)                    # (pairs, NPTS, 2)
    brk = np.abs(np.diff(P[..., 0], axis=1)) > W / 2                 # antimeridian crossings
    # per-pair pixel footprint (anti-aliased) for the gold accumulation
    flights_per_pair = np.bincount(pid, minlength=len(pairs)).astype(np.float32)
    pix_idx, pix_w, starts = [], [], [0]
    for k in range(len(pairs)):
        pts = P[k]
        cuts = np.r_[0, np.nonzero(brk[k])[0] + 1, NPTS]
        x0, y0 = np.floor(pts.min(0)).astype(int) - 2
        x1, y1 = np.ceil(pts.max(0)).astype(int) + 3
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
        canvas = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.uint8)
        polys = [np.round((pts[c0:c1] - [x0, y0]) * 16).astype(np.int32) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 - c0 > 1]
        cv2.polylines(canvas, polys, False, 255, 1, cv2.LINE_AA, shift=4)
        yy, xx = np.nonzero(canvas)
        pix_idx.append(((yy + y0) * W + (xx + x0)).astype(np.int32)); pix_w.append(canvas[yy, xx].astype(np.float32) / 255)
        starts.append(starts[-1] + len(yy))
    pix_idx = np.concatenate(pix_idx); pix_w = np.concatenate(pix_w); starts = np.array(starts, np.int64)
    pair_of_px = np.repeat(np.arange(len(pairs)), np.diff(starts))
    total = np.bincount(pix_idx, weights=pix_w * flights_per_pair[pair_of_px], minlength=W * H).astype(np.float32)
    # airports
    ax, ay = to_px(lon, lat)
    # part-2 timeline
    legs, leg_km = I["legs"], I["leg_km"]
    leg_t1 = np.cumsum(leg_km) / C_KMS
    leg_t0 = np.r_[0.0, leg_t1[:-1]]
    leg_pair = np.where(legs[:, 0] >= 0, pid[np.maximum(legs[:, 0], 0)], -1)
    # repositioning legs need their own paths
    dead = np.nonzero(legs[:, 0] < 0)[0]
    dla, dlo = slerp_paths(lat[legs[dead, 1]], lon[legs[dead, 1]], lat[legs[dead, 2]], lon[legs[dead, 2]])
    dpx, dpy = to_px(dlo, dla)
    DP = np.stack([dpx, dpy], -1).astype(np.float32)
    leg_path = np.where(legs[:, 0] >= 0, leg_pair, -1)
    dead_slot = np.full(len(legs), -1); dead_slot[dead] = np.arange(len(dead))
    leg_rev = (legs[:, 1] != pa[np.maximum(leg_pair, 0)]) & (legs[:, 0] >= 0)
    bg = np.load(os.path.join(os.path.dirname(HERE), "data", "layers.npz"))["bg"]
    np.savez(os.path.join(DATA, "flight_layers.npz"), P=P, brk=brk, pid=pid, pa=pa, pb=pb,
             pix_idx=pix_idx, pix_w=pix_w, pstarts=starts, total=total, fpp=flights_per_pair,
             ax=ax.astype(np.float32), ay=ay.astype(np.float32), src=src, dst=dst, dep=S["dep"], dur=S["dur"], km=S["km"],
             legs=legs, leg_km=leg_km, leg_t0=leg_t0, leg_t1=leg_t1, leg_pair=leg_pair, leg_rev=leg_rev,
             dead_slot=dead_slot, DP=DP, bg=bg)
    print("pairs", len(pairs), "pixels", len(pix_idx), "part-2 seconds", round(float(leg_t1[-1]), 2))


# ----------------------------------------------------------------------------------------- renderer
def sprite(sig, n):
    k = np.arange(n, dtype=np.float32) - n // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


class FlightRenderer:
    def __init__(self):
        L = dict(np.load(os.path.join(DATA, "flight_layers.npz")))
        self.L = L
        st = json.load(open(os.path.join(DATA, "itinerary_stats.json")))
        self.T2 = float(L["leg_t1"][-1])
        self.n_flights = int(st["flights"])
        self.total_km = float(st["total_km"])
        self.flight_km = float(st["flight_km"])
        # part-2 static network brightness
        tot = L["total"]
        self.Inet = (1 - np.exp(-tot / 9.0)).astype(np.float32)
        self.acc = np.zeros(W * H, np.float32)
        self.flash = np.zeros(W * H, np.float32)
        self.k_done = 0
        self.dot = (sprite(1.6, 121)[..., None] * np.array([1, .97, .9]) * 2.6 + sprite(4.5, 121)[..., None] *
                    np.array([1, .72, .3]) * 1.25 + sprite(18, 121)[..., None] * np.array([1, .52, .12]) * 0.45).astype(np.float32)
        self.f_title = font("Inter-SemiBold.otf", 46)
        self.f_sub = font("Inter-Regular.otf", 25)
        self.f_clock = font("InterDisplay-Medium.otf", 66)
        self.f_small = font("Inter-Medium.otf", 15)
        self.f_dist = font("Inter-Regular.otf", 17)
        self.f_cap = font("Inter-Regular.otf", 14)
        self.f_card = font("Inter-SemiBold.otf", 54)
        self.f_card2 = font("Inter-Regular.otf", 28)
        self.ab = np.bincount(L["src"], minlength=len(L["ax"])).astype(np.float32)   # departures per airport per day
        dep2 = np.r_[L["dep"] - 24.0, L["dep"]]
        self.d_order = np.argsort(dep2)
        self.d_sorted = dep2[self.d_order]
        self.d_src = np.r_[L["src"], L["src"]][self.d_order]
        self.d_dur = np.r_[L["dur"], L["dur"]][self.d_order]
        self.d_fid = np.r_[np.arange(len(L["src"])), np.arange(len(L["src"]))][self.d_order]
        self.maxdur = float(L["dur"].max())

    # ------------------------------------------------------------- helpers
    def path_points(self, pair, u, rev):
        """positions at fractions u (n, k) along pair paths."""
        P = self.L["P"]
        uu = np.where(rev[:, None], 1 - u, u) * (NPTS - 1)
        i0 = np.clip(np.floor(uu).astype(int), 0, NPTS - 2)
        f = (uu - i0)[..., None]
        p0 = P[pair[:, None], i0]; p1 = P[pair[:, None], i0 + 1]
        valid = ~self.L["brk"][pair[:, None], i0]              # segment does not wrap the map edge
        return p0 + (p1 - p0) * f, valid

    def text(self, img, items):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        for it in items:
            d.text(*it[:2], **it[2])
        a = np.asarray(im).astype(np.float32) / 255
        ys, xs = np.nonzero(a[..., 3] > 0)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            reg = img[y0:y1, x0:x1]; aa = a[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = reg * (1 - aa[..., 3:]) + aa[..., :3] * aa[..., 3:]
        return img

    def finish(self, E, extra_text, dim=0.0):
        small = cv2.resize(E, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 2.2)
        tiny = cv2.resize(small, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (W // 2, H // 2))
        bloom = cv2.resize(cv2.addWeighted(b1, 0.55, b2, 0.65, 0), (W, H))
        img = cv2.add(cv2.add(self.L["bg"], E), bloom)
        cv2.multiply(img, -1.15, dst=img); cv2.exp(img, dst=img); cv2.subtract((1.0, 1.0, 1.0, 0.0), img, dst=img)
        if dim > 0:
            img *= (1 - dim)
        img = self.text(img, extra_text)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def captions(self, lines):
        return [((60, 1010 + i * 20), s, dict(font=self.f_cap, fill=(120, 140, 165, 255))) for i, s in enumerate(lines)]

    # ------------------------------------------------------------- part 1: one day
    def frame_day(self, h):
        L = self.L
        E = np.zeros((H, W, 3), np.float32)
        lo = np.searchsorted(self.d_sorted, h - self.maxdur)
        hi = np.searchsorted(self.d_sorted, h, "right")
        cand = np.arange(lo, hi)
        dep = self.d_sorted[cand]; dur = self.d_dur[cand]
        act = cand[h < dep + dur]
        fid = self.d_fid[act]
        u = (h - self.d_sorted[act]) / self.d_dur[act]
        pair = L["pid"][fid]
        rev = L["src"][fid] != L["pa"][pair]
        # faint full arcs of everything in the air
        arcs = np.zeros((H, W), np.uint8)
        polys = []
        brk = L["brk"]
        P = L["P"]
        for pr in np.unique(pair):
            pts = np.round(P[pr] * 4).astype(np.int32)
            if brk[pr].any():
                cuts = np.r_[0, np.nonzero(brk[pr])[0] + 1, NPTS]
                polys += [pts[c0:c1] for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 - c0 > 1]
            else:
                polys.append(pts)
        cv2.polylines(arcs, polys, False, 255, 1, cv2.LINE_AA, shift=2)
        E += arcs.astype(np.float32)[..., None] / 255 * (CYAN * 0.10)
        # comets: last ~35 min of each flight's path, brightening to the head
        tail = np.clip(0.6 / self.d_dur[act], 0, 1)
        k = 7
        us = np.clip(u[:, None] - tail[:, None] * (1 - np.linspace(0, 1, k))[None], 0, 1)
        pts, valid = self.path_points(pair, us, rev)
        ok = valid.all(1) & (np.abs(np.diff(pts[..., 0], axis=1)).max(1) < W / 2)
        pts = np.round(pts[ok] * 4).astype(np.int32)
        com = np.zeros((H, W), np.uint8)
        for j in range(k - 1):
            cv2.polylines(com, list(pts[:, j:j + 2]), False, int(60 + 195 * (j + 1) / (k - 1)), 1, cv2.LINE_AA, shift=2)
        E += com.astype(np.float32)[..., None] / 255 * FLY * 0.95
        # airports: pulse with recent departures (last 40 min), faint base dot for every airport
        r0 = np.searchsorted(self.d_sorted, h - 0.66); r1 = hi
        recent = self.d_src[r0:r1]
        age = (h - self.d_sorted[r0:r1]) / 0.66
        glow = np.zeros(W * H, np.float32)
        ax, ay = L["ax"], L["ay"]
        aidx = (np.clip(ay.astype(int), 0, H - 1) * W + np.clip(ax.astype(int), 0, W - 1))
        np.add.at(glow, aidx[recent], (1 - age) ** 2 * 0.35)
        base = np.zeros(W * H, np.float32)
        np.maximum.at(base, aidx, np.minimum(0.08 + self.ab / 600, 0.5))
        g = glow.reshape(H, W) + base.reshape(H, W)
        g = cv2.GaussianBlur(g, (0, 0), 1.2) * 6 + cv2.GaussianBlur(g, (0, 0), 4) * 10
        E += np.minimum(g, 3)[..., None] * np.array([0.75, 0.9, 1.0], np.float32)
        # departure rings at busy airports (one ring per departure in the last 20 min)
        rr0 = np.searchsorted(self.d_sorted, h - 0.33)
        ring = np.zeros((H, W), np.uint8)
        for s_, t_ in zip(self.d_src[rr0:hi][::2], self.d_sorted[rr0:hi][::2]):
            v = (h - t_) / 0.33
            cv2.circle(ring, (int(ax[s_] * 16), int(ay[s_] * 16)), int((2 + 14 * v) * 16), int(110 * (1 - v)), 1, cv2.LINE_AA, shift=4)
        E += ring.astype(np.float32)[..., None] / 255 * FLY * 0.6
        # HUD
        in_air = len(act)
        today = L["dep"] <= h
        flown_km = float(np.sum(np.clip((h - L["dep"]) / L["dur"], 0, 1) * L["km"]))
        hh, mm = int(h) % 24, int((h % 1) * 60)
        txt = [((58, 40), "Every flight on Earth, one day", dict(font=self.f_title, fill=(236, 242, 250, 255))),
               ((60, 100), f"~{self.n_flights:,} scheduled flights · 24 hours in {DAY_S:.0f} seconds", dict(font=self.f_sub, fill=(150, 176, 205, 255))),
               ((W - 60, 896), "TIME (UTC)", dict(font=self.f_small, fill=(130, 150, 175, 255), anchor="ra")),
               ((W - 60, 912), f"{hh:02d}:{mm:02d}", dict(font=self.f_clock, fill=(246, 236, 214, 255), anchor="ra")),
               ((W - 60, 996), f"{in_air:,} flights in the air · {int(today.sum()):,} departed today", dict(font=self.f_dist, fill=(160, 178, 200, 255), anchor="ra")),
               ((W - 60, 1022), f"{flown_km / 1e6:.1f} million km flown today", dict(font=self.f_dist, fill=(160, 178, 200, 255), anchor="ra"))]
        txt += self.captions(["Modeled schedule: real airports and airline routes (OpenFlights), frequencies and times estimated to total ~109,000 flights/day.",
                              "Flight paths drawn as great circles. All figures are estimates."])
        return E, txt

    # ------------------------------------------------------------- part 2: light-speed plane
    def reset_to(self, t):
        L = self.L
        k = int(np.searchsorted(L["leg_t1"], t, "right"))
        self.acc[:] = 0
        self.flash[:] = 0
        self._add_legs(0, k, flash=False)
        self.k_done = k

    def _add_legs(self, a, b, flash=True):
        L = self.L
        pairs = L["leg_pair"][a:b]
        pairs = pairs[pairs >= 0]
        if len(pairs) == 0:
            return
        st = L["pstarts"]
        if len(pairs) > 2000:      # bulk path (chunk start)
            cnt = np.bincount(pairs, minlength=len(st) - 1).astype(np.float32)
            reps = np.repeat(cnt, np.diff(st))
            self.acc += np.bincount(L["pix_idx"], weights=L["pix_w"] * reps, minlength=W * H).astype(np.float32)
            return
        idx = np.concatenate([L["pix_idx"][st[p]:st[p + 1]] for p in pairs])
        w = np.concatenate([L["pix_w"][st[p]:st[p + 1]] for p in pairs])
        np.add.at(self.acc, idx, w)
        if flash:
            np.add.at(self.flash, idx, w)

    def frame_light(self, t, decay):
        L = self.L
        k = int(np.searchsorted(L["leg_t1"], t, "right"))
        self.flash *= decay
        if k > self.k_done:
            self._add_legs(self.k_done, k)
            self.k_done = k
        frac = np.minimum(self.acc / np.maximum(L["total"], 1e-6), 1.0)
        I = self.Inet
        Ig = (I * frac).reshape(H, W); Ic = (I * (1 - frac)).reshape(H, W)
        Fl = np.minimum(self.flash, 2.5).reshape(H, W) * 0.9
        E = cv2.merge([cv2.scaleAdd(Fl, float(HOT[c]), cv2.addWeighted(Ic, float(CYAN[c] * 0.45), Ig, float(GOLD[c] * 0.85), 0)) for c in range(3)])
        # plane position and trail (the last ~0.06 s of legs)
        i = min(int(np.searchsorted(L["leg_t1"], t, "left")), len(L["leg_t1"]) - 1)
        x, y, _ = self._leg_point(i, np.clip((t - L["leg_t0"][i]) / max(L["leg_t1"][i] - L["leg_t0"][i], 1e-9), 0, 1))
        j0 = max(0, int(np.searchsorted(L["leg_t1"], t - 0.06, "left")))
        tr = np.zeros((H, W), np.uint8)
        for j in range(j0, i + 1):
            u1 = 1.0 if j < i else np.clip((t - L["leg_t0"][j]) / max(L["leg_t1"][j] - L["leg_t0"][j], 1e-9), 0, 1)
            us = np.linspace(0, u1, 24)
            q = [self._leg_point(j, uu) for uu in us]
            pts = np.array([(a, b) for a, b, _ in q], np.float32)
            okp = np.array([c for _, _, c in q])
            seg_ok = (np.abs(np.diff(pts[:, 0])) < W / 2) & okp[1:] & okp[:-1]
            age = (t - L["leg_t1"][j]) / 0.06 if j < i else 0.0
            val = int(255 * (1 - 0.8 * np.clip(age, 0, 1)) * (0.45 if L["legs"][j, 0] < 0 else 1.0))
            p16 = np.round(pts * 16).astype(np.int32)
            cv2.polylines(tr, [p16[q:q + 2] for q in range(len(p16) - 1) if seg_ok[q]], False, val, 1, cv2.LINE_AA, shift=4)
        E += tr.astype(np.float32)[..., None] / 255 * np.array([1.0, 0.85, 0.55], np.float32) * 1.3
        n = 121; h2 = n // 2
        xi, yi = int(round(x)), int(round(y))
        a0, a1, b0, b1 = max(yi - h2, 0), min(yi + h2 + 1, H), max(xi - h2, 0), min(xi + h2 + 1, W)
        if a1 > a0 and b1 > b0:
            E[a0:a1, b0:b1] += self.dot[a0 - (yi - h2):a1 - (yi - h2), b0 - (xi - h2):b1 - (xi - h2)]
        fin = np.clip((t - (self.T2 - 2.0)) / 2.0, 0, 1) ** 2
        if fin > 0:
            E *= 1 + 0.45 * fin
        flown = int(np.sum(L["legs"][:k, 0] >= 0))
        km = float(L["leg_t1"][k - 1] * C_KMS) if k else 0.0
        tt = min(t, self.T2)
        mm, ss = int(tt // 60), int(tt % 60)
        txt = [((58, 40), "One plane at the speed of light", dict(font=self.f_title, fill=(236, 242, 250, 255))),
               ((60, 100), f"flying every one of today's {self.n_flights:,} flights, back to back", dict(font=self.f_sub, fill=(150, 176, 205, 255))),
               ((W - 60, 896), "ELAPSED (REAL TIME)", dict(font=self.f_small, fill=(130, 150, 175, 255), anchor="ra")),
               ((W - 60, 912), f"{mm}:{ss:02d}", dict(font=self.f_clock, fill=(246, 236, 214, 255), anchor="ra")),
               ((W - 60, 996), f"{flown:,} of {self.n_flights:,} flights flown", dict(font=self.f_dist, fill=(160, 178, 200, 255), anchor="ra")),
               ((W - 60, 1022), f"{km / 1e6:.1f} million km at 299,792 km/s", dict(font=self.f_dist, fill=(160, 178, 200, 255), anchor="ra"))]
        txt += self.captions(["Each flight departs from the airport where the last one landed. 3,150 empty repositioning legs add ~1% distance.",
                              "Modeled schedule from OpenFlights routes. Thought experiment: all figures are estimates."])
        return E, txt

    def _leg_point(self, j, u):
        L = self.L
        p = L["leg_pair"][j]
        if p >= 0:
            pts = L["P"][p]
            if L["leg_rev"][j]:
                u = 1 - u
        else:
            pts = L["DP"][L["dead_slot"][j]]
        x = u * (NPTS - 1); i0 = min(int(x), NPTS - 2); f = x - i0
        if abs(pts[i0 + 1, 0] - pts[i0, 0]) > W / 2:            # wraps the map edge: snap to an endpoint
            q = pts[i0] if f < 0.5 else pts[i0 + 1]
            return float(q[0]), float(q[1]), False
        q = pts[i0] + (pts[i0 + 1] - pts[i0]) * f
        return float(q[0]), float(q[1]), True


# ----------------------------------------------------------------------------------------- timeline
def timeline():
    """Return list of (segment, param) per output frame."""
    R_T2 = float(np.load(os.path.join(DATA, "flight_layers.npz"))["leg_t1"][-1])
    n1 = int(DAY_S * FPS); n_card = int(CARD_S * FPS); n2 = int(np.ceil(R_T2 * FPS)) + 1; n_end = int(END_S * FPS)
    fr = [("day", 24.0 * f / (n1 - 1)) for f in range(n1)]
    fr += [("card", f / n_card) for f in range(n_card)]
    fr += [("light", min(f / FPS, R_T2)) for f in range(n2)]
    fr += [("end", f / n_end) for f in range(n_end)]
    return fr


def render_range(args):
    f0, f1, path = args
    cv2.setNumThreads(2)
    R = FlightRenderer()
    fr = timeline()
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-tune", "film",
                          "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    decay = 0.88
    # warm up part-2 state
    first_light = next((i for i in range(f0, f1) if fr[i][0] in ("light", "end")), None)
    if first_light is not None:
        t0 = fr[first_light][1] if fr[first_light][0] == "light" else R.T2
        R.reset_to(max(t0 - 1.0, 0) - 1e-9)
        for tt in np.arange(max(t0 - 1.0, 0), t0, 1 / FPS):
            R.flash *= decay; k = int(np.searchsorted(R.L["leg_t1"], tt, "right"))
            if k > R.k_done:
                R._add_legs(R.k_done, k); R.k_done = k
    last_day = None
    for i in range(f0, f1):
        kind, v = fr[i]
        if kind == "day":
            E, txt = R.frame_day(v)
            tv = v / 24.0 * DAY_S
            if tv < HOOK_S:
                a = 1.0 if tv < HOOK_S - 0.6 else (HOOK_S - tv) / 0.6
                hook = [((W // 2, 450), "Every flight on Earth in one day", dict(font=R.f_card, fill=(236, 242, 250, int(255 * a)), anchor="ma")),
                        ((W // 2, 528), "then one plane flies them all at the speed of light", dict(font=R.f_card2, fill=(255, 186, 80, int(255 * a)), anchor="ma"))]
                img = R.finish(E, txt + hook, dim=0.4 * a)
            else:
                img = R.finish(E, txt)
            last_day = (E, txt)
        elif kind == "card":
            if last_day is None:
                last_day = R.frame_day(24.0)
            E, txt = last_day
            a = min(1.0, v / 0.25)
            card = [((W // 2, 470), "Now: one plane at the speed of light", dict(font=R.f_card, fill=(255, 186, 80, int(255 * a)), anchor="ma")),
                    ((W // 2, 548), f"flying every one of those {R.n_flights:,} flights, back to back", dict(font=R.f_card2, fill=(236, 242, 250, int(255 * a)), anchor="ma")),
                    ((W // 2, 596), "How long does it take?", dict(font=R.f_card2, fill=(150, 176, 205, int(255 * a)), anchor="ma"))]
            img = R.finish(E, txt[:0] + card, dim=0.6 * a)
        elif kind == "light":
            E, txt = R.frame_light(v, decay)
            img = R.finish(E, txt)
        else:
            E, txt = R.frame_light(R.T2, decay)
            a = min(1.0, v / 0.3)
            mm, ss = int(R.T2 // 60), int(R.T2 % 60)
            txt = txt + [((W // 2, 470), f"{mm} minutes {ss} seconds.", dict(font=R.f_card, fill=(255, 186, 80, int(255 * a)), anchor="ma")),
                         ((W // 2, 548), f"Every flight of the day, {R.total_km / 1e6:.0f} million km, at light speed.", dict(font=R.f_card2, fill=(236, 242, 250, int(255 * a)), anchor="ma"))]
            img = R.finish(E, txt, dim=0.45 * a)
        p.stdin.write(img.tobytes())
        if (i - f0) % 600 == 0:
            print(f"  {os.path.basename(path)} {i - f0}/{f1 - f0}", flush=True)
    p.stdin.close(); p.wait()
    return path


def video(workers):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    fr = timeline()
    n = len(fr)
    cost = np.cumsum([3.2 if k == "day" else 1.0 for k, _ in fr])    # day frames are ~3x slower
    cuts = np.r_[0, np.searchsorted(cost, cost[-1] * np.arange(1, workers) / workers), n].astype(int)
    jobs = [(int(a), int(b), os.path.join(OUT, f"_part{i}.mp4")) for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
    with Pool(workers) as pool:
        parts = pool.map(render_range, jobs)
    lst = os.path.join(OUT, "_parts.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
    final = os.path.join(OUT, "every_flight_light_speed_v2.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", final], check=True)
    for q in parts:
        os.remove(q)
    os.remove(lst)
    print("wrote", final, f"{n / FPS:.1f}s")


def stills(args):
    os.makedirs(OUT, exist_ok=True)
    R = FlightRenderer()
    for part, v in zip(args[0::2], args[1::2]):
        if int(part) == 1:
            E, txt = R.frame_day(float(v))
        else:
            R.reset_to(float(v) - 0.5)
            for tt in np.arange(float(v) - 0.5, float(v) + 1e-9, 1 / FPS):
                E, txt = R.frame_light(float(tt), 0.88)
        img = R.finish(E, txt)
        path = os.path.join(OUT, f"still_p{part}_{v}.png")
        cv2.imwrite(path, img[..., ::-1]); print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "video", "still"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "video":
        video(a.workers)
    else:
        stills(a.args)
