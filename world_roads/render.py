"""Render "All world roads at light speed" (280 s, 1920x1080, 30 fps) by streaming frames to FFmpeg.

    python render.py prepare                 # build cached raster layers from graph + route
    python render.py video  [--workers 3]    # full 8,400-frame film -> out/world_roads_light_speed.mp4
    python render.py preview                 # ~15 s time-compressed preview -> out/world_roads_preview.mp4
    python render.py still 120.5 [...]       # PNG stills at given times (seconds)

Everything is deterministic. Per frame only *changes* are computed: road pixels are pre-sorted by
the time the dot first reaches them, so turning roads gold is an incremental scatter-add.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from proj import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FPS = 30
DURATION = 280.0
NFRAMES = int(DURATION * FPS)            # 8400
C_KMS = 299_792.458
EST_TOTAL_KM = DURATION * C_KMS          # 83.94 M km implied by 280 s at c

CYAN = np.array([0.14, 0.66, 1.00], np.float32)
GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.92
HOT = np.array([1.00, 0.93, 0.78], np.float32)
CLASS_W = {0: 1.0, 1: 0.95, 2: 0.80, 3: 0.62, 4: 0.48}   # motorway..tertiary line intensity
CLASS_T = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1}                  # line thickness on the 2x canvas
FLASH_S = 0.9            # newly visited roads glow white-hot for this long
TRAIL_S = 0.12           # visible motion trail length (seconds of route)
FONT_DIR = "/usr/share/fonts/opentype/inter"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


# --------------------------------------------------------------------------- prepare
def prepare():
    g = dict(np.load(os.path.join(DATA, "graph.npz")))
    r = dict(np.load(os.path.join(DATA, "route.npz")))
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    rstats = json.load(open(os.path.join(DATA, "route_stats.json")))
    ea, eb, ecls = g["ea"], g["eb"], g["ecls"]
    m = len(ea)
    edge, cum = r["edge"], r["cum"]
    T = DURATION * cum / cum[-1]                       # route time at the END of each step
    Tprev = np.r_[0.0, T[:-1]]
    tv = np.full(m, np.inf)
    steps = np.nonzero(edge >= 0)[0][::-1]             # reverse so the first visit wins
    tv[edge[steps]] = 0.5 * (Tprev[steps] + T[steps])
    assert np.isfinite(tv).all()

    S = 2
    vt2 = np.full((H * S, W * S), np.inf, np.float32)
    I8 = np.zeros((H * S, W * S), np.uint8)
    P = np.stack([np.c_[g["nx"][ea], g["ny"][ea]], np.c_[g["nx"][eb], g["ny"][eb]]], 1)
    P = np.rint(P).astype(np.int32)                    # 4x grid == 2x canvas with shift=1
    # intensity (anti-aliased), dim classes first so bright classes win
    for c in sorted(CLASS_W, key=lambda c: CLASS_W[c]):
        sel = np.nonzero(ecls == c)[0]
        if len(sel):
            cv2.polylines(I8, list(P[sel]), False, int(255 * CLASS_W[c]), CLASS_T[c], cv2.LINE_AA, shift=1)
    # visit time, drawn latest-first so earliest visit wins
    fb = np.floor(tv * FPS).astype(np.int64)
    order = np.argsort(-fb, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(fb[order]))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = order[a:b]
        val = float(tv[sel].min())
        for th in (2, 1):
            s2 = sel[[CLASS_T[c] == th for c in ecls[sel]]]
            if len(s2):
                cv2.polylines(vt2, list(P[s2]), False, val, th, cv2.LINE_8, shift=1)
    Imaj = I8.astype(np.float32) / 255.0

    d = np.load(os.path.join(DATA, "density.npz"))["dens"]
    land2 = cv2.imread(os.path.join(DATA, "land2x.png"), 0)
    minor = d[1] + d[2]
    minor = cv2.GaussianBlur(minor, (0, 0), 0.6)
    Imin = 0.36 * np.clip(np.log1p(minor / 1.5) / np.log1p(400 / 1.5), 0, 1) ** 1.15
    Imin *= (land2 > 0) | (Imaj > 0)
    I2 = np.maximum(Imaj, Imin)
    # every lit pixel without its own visit time inherits the nearest routed road's time
    missing = ~np.isfinite(vt2)
    idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    vt_full = vt2[idx[0], idx[1]]
    lit = I2 > 0.012
    yy, xx = np.nonzero(lit)
    pv = vt_full[yy, xx].astype(np.float32)
    pi = I2[yy, xx].astype(np.float32) / (S * S)
    p1 = ((yy // S) * W + (xx // S)).astype(np.int32)
    o = np.argsort(pv, kind="stable")
    pv, pi, p1 = pv[o], pi[o], p1[o]
    Itot = np.bincount(p1, weights=pi, minlength=W * H).astype(np.float32)

    # city sparkle candidates: brightest local maxima of minor-road density
    dens1 = cv2.resize(minor, (W, H), interpolation=cv2.INTER_AREA)
    mx = ndimage.maximum_filter(dens1, 9)
    cand = np.flatnonzero((dens1 == mx) & (dens1 > np.percentile(dens1[dens1 > 0], 99.3)))
    rng = np.random.default_rng(7)
    cand = cand[np.argsort(-dens1.ravel()[cand])][:1400]
    sp_phase = rng.uniform(0, 2 * np.pi, len(cand)).astype(np.float32)
    sp_freq = rng.uniform(0.6, 1.8, len(cand)).astype(np.float32)
    sp_amp = (0.25 + 0.75 * (dens1.ravel()[cand] / dens1.ravel()[cand].max()) ** 0.5).astype(np.float32)
    sp_vt = vt_full[(cand // W) * S, (cand % W) * S].astype(np.float32)

    # background: near-black ocean, dark navy land, faint coastline
    land1 = cv2.resize(land2, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    yy1, xx1 = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1 - 0.45 * (((xx1 - W / 2) / (W * 0.62)) ** 2 + ((yy1 - H * 0.52) / (H * 0.75)) ** 2)
    ocean = np.array([0.008, 0.016, 0.035], np.float32)
    landc = np.array([0.030, 0.050, 0.088], np.float32)
    bg = ocean[None, None] * np.clip(vign, 0.35, 1)[..., None] + land1[..., None] * (landc - ocean)[None, None]
    coast = cv2.morphologyEx(land2, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    coast = cv2.resize(coast, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    bg += coast[..., None] * np.array([0.035, 0.07, 0.12], np.float32)
    bg += cv2.GaussianBlur(coast, (0, 0), 6)[..., None] * np.array([0.01, 0.025, 0.05], np.float32)

    # route geometry (1x pixels)
    rx = (g["nx"][r["node"]] / 4).astype(np.float32)
    ry = (g["ny"][r["node"]] / 4).astype(np.float32)
    jumps = np.nonzero(edge == -1)[0]
    jd = np.hypot(rx[jumps] - rx[jumps - 1], ry[jumps] - ry[jumps - 1])
    jsel = jumps[jd > 18]
    np.savez(os.path.join(DATA, "layers.npz"), pv=pv, pi=pi, p1=p1, Itot=Itot, bg=bg.astype(np.float32),
             sp_idx=cand.astype(np.int32), sp_phase=sp_phase, sp_freq=sp_freq, sp_amp=sp_amp, sp_vt=sp_vt,
             rx=rx, ry=ry, rt=T.astype(np.float64), redge=edge.astype(np.int64),
             j_t=T[jsel], j_x0=rx[jsel - 1], j_y0=ry[jsel - 1], j_x1=rx[jsel], j_y1=ry[jsel],
             j_d=np.hypot(rx[jsel] - rx[jsel - 1], ry[jsel] - ry[jsel - 1]))
    meta = {"routed_km": float(np.load(os.path.join(DATA, "graph.npz"))["elen"].sum()),
            "walk_km": float(cum[-1]), "overhead": rstats["repeat_overhead"], "jumps": rstats["jumps"],
            "lit_pixels": int(len(pv))}
    json.dump(meta, open(os.path.join(DATA, "layers_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------- renderer
def sprite(sig, size):
    k = np.arange(size, dtype=np.float32) - size // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


class Renderer:
    def __init__(self, preview=False):
        L = dict(np.load(os.path.join(DATA, "layers.npz")))
        self.L = L
        self.meta = json.load(open(os.path.join(DATA, "layers_meta.json")))
        self.Ig = np.zeros(W * H, np.float32)
        self.k = 0
        self.preview = preview
        # dot sprite: white-hot core, golden glow, wide orange halo, faint anamorphic streak
        n = 161
        core = sprite(1.5, n)[..., None] * np.array([1.0, 0.97, 0.90]) * 2.6
        glow = sprite(4.5, n)[..., None] * np.array([1.0, 0.72, 0.30]) * 1.25
        halo = sprite(18, n)[..., None] * np.array([1.0, 0.52, 0.12]) * 0.42
        k = np.arange(n, dtype=np.float32) - n // 2
        streak = (np.exp(-(k[:, None] ** 2) / (2 * 1.2 ** 2)) * np.exp(-np.abs(k[None]) / 28))[..., None] \
            * np.array([1.0, 0.75, 0.4]) * 0.35
        self.dot = (core + glow + halo + streak).astype(np.float32)
        self.static = self._static_overlay()
        self.f_clock = font("InterDisplay-Medium.otf", 66)
        self.f_small = font("Inter-Medium.otf", 15)
        self.f_dist = font("Inter-Regular.otf", 17)
        self.f_jump = font("Inter-Medium.otf", 14)
        rng = np.random.default_rng(1234)
        # one static dither texture (breaks up banding in the dark gradients without costing bitrate)
        self.grain = [cv2.merge([rng.uniform(-0.004, 0.004, (H, W)).astype(np.float32)] * 3)]
        self.nframe = 0

    # ---- static text (title etc.)
    def _static_overlay(self):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((58, 40), "All world roads at light speed", font=font("Inter-SemiBold.otf", 46), fill=(236, 242, 250, 255))
        d.text((60, 100), "Estimated total duration: 4m 40s", font=font("Inter-Regular.otf", 25), fill=(150, 176, 205, 255))
        fs = font("Inter-Regular.otf", 14)
        x0, y0 = 60, 962
        lines = [
            "Stylized visualization of a thought experiment — not a verified or optimal all-road route. All figures are estimates.",
            f"Routed: real motorway–tertiary road geometry (≈{self.meta['routed_km']/1e6:.1f} M km, generalized); "
            "fainter minor roads light up by proximity.",
            "Rings mark hypothetical instantaneous jumps between disconnected road networks; sea crossings add no distance.",
            "Road data © OpenStreetMap contributors, via Overture Maps (release 2026-09-23.1). Land: Natural Earth.",
        ]
        for i, s in enumerate(lines):
            d.text((x0, y0 + i * 21), s, font=fs, fill=(120, 140, 165, 255))
        # legend
        lx, ly = 62, 930
        fl = font("Inter-Regular.otf", 15)
        d.line([(lx, ly + 9), (lx + 26, ly + 9)], fill=(60, 185, 255, 255), width=3)
        d.text((lx + 36, ly), "not yet reached", font=fl, fill=(140, 165, 190, 255))
        d.line([(lx + 170, ly + 9), (lx + 196, ly + 9)], fill=(255, 175, 60, 255), width=3)
        d.text((lx + 206, ly), "visited", font=fl, fill=(140, 165, 190, 255))
        if self.preview:
            d.text((W - 60, 40), "PREVIEW · time-compressed", font=font("Inter-SemiBold.otf", 18),
                   fill=(255, 190, 90, 255), anchor="ra")
        a = np.asarray(im).astype(np.float32) / 255
        out = []
        lab, n = ndimage.label(ndimage.binary_dilation(a[..., 3] > 0, iterations=12))
        for sl in ndimage.find_objects(lab):
            ys, xs = sl
            b = (xs.start, ys.start, xs.stop, ys.stop)
            aa = a[ys, xs]
            out.append((b, np.ascontiguousarray(aa[..., :3] * aa[..., 3:]), np.ascontiguousarray(aa[..., 3:])))
        return out

    def _dynamic_text(self, t, jump_labels):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        tt = min(t, DURATION)
        mm, ss = int(tt // 60), int(tt % 60)
        right = W - 60
        d.text((right, 896), "ELAPSED", font=self.f_small, fill=(130, 150, 175, 255), anchor="ra")
        # monospaced digits: draw each glyph centred in a fixed cell to avoid jitter
        s = f"{mm}:{ss:02d}"
        cell = 40
        x = right - cell * (len(s) - 1) - 18
        for ch in s:
            w = cell if ch != ":" else 18
            d.text((x + w / 2, 912), ch, font=self.f_clock, fill=(246, 236, 214, 255), anchor="ma")
            x += w
        frac = tt / DURATION
        d.text((right, 1000), f"≈ {frac * EST_TOTAL_KM / 1e6:.1f} M km hypothetical distance at c",
               font=self.f_dist, fill=(160, 178, 200, 255), anchor="ra")
        bw = 300
        d.rectangle([right - bw, 1030, right, 1032], fill=(40, 55, 75, 255))
        d.rectangle([right - bw, 1030, right - bw + int(bw * frac), 1032], fill=(255, 176, 66, 255))
        for (x, y, a) in jump_labels:
            txt = "hypothetical jump · no road"
            tw = d.textlength(txt, font=self.f_jump)
            tx = x + 16 if x + 16 + tw < W - 24 else x - 16 - tw
            ty = y - 26 if y > 200 else y + 12
            d.text((tx, ty), txt, font=self.f_jump, fill=(225, 235, 250, int(255 * a)))
        bb = im.getbbox()
        return im, bb

    def reset_to(self, t):
        L = self.L
        k = int(np.searchsorted(L["pv"], t, "right"))
        self.Ig = np.bincount(L["p1"][:k], weights=L["pi"][:k], minlength=W * H).astype(np.float32)
        self.k = k

    def dot_pos(self, t):
        rt, rx, ry = self.L["rt"], self.L["rx"], self.L["ry"]
        i = int(np.searchsorted(rt, t, "left"))
        i = min(max(i, 1), len(rt) - 1)
        t0, t1 = rt[i - 1], rt[i]
        if self.L["redge"][i] < 0 or t1 <= t0:
            return float(rx[i]), float(ry[i]), i
        u = (t - t0) / (t1 - t0)
        return float(rx[i - 1] + u * (rx[i] - rx[i - 1])), float(ry[i - 1] + u * (ry[i] - ry[i - 1])), i

    def frame(self, t, time_scale=1.0):
        L = self.L
        # gold accumulation (incremental)
        k = int(np.searchsorted(L["pv"], t, "right"))
        if k > self.k:
            np.add.at(self.Ig, L["p1"][self.k:k], L["pi"][self.k:k])
            self.k = k
        elif k < self.k:
            self.reset_to(t)
        Ig = self.Ig
        Ic = L["Itot"] - Ig
        # white-hot wake of recently visited roads
        flash_s = FLASH_S * time_scale
        kf = int(np.searchsorted(L["pv"], t - flash_s, "right"))
        F = np.zeros(W * H, np.float32)
        if k > kf:
            age = (t - L["pv"][kf:k]) / flash_s
            np.add.at(F, L["p1"][kf:k], L["pi"][kf:k] * (1 - age) ** 2 * 2.2)
        Ic2, Ig2, F2 = Ic.reshape(H, W), Ig.reshape(H, W), F.reshape(H, W)
        E = cv2.merge([cv2.scaleAdd(F2, float(HOT[c]), cv2.addWeighted(Ic2, float(CYAN[c] * 0.56), Ig2, float(GOLD[c]), 0))
                       for c in range(3)])
        # finale: whole network brightens over the last two seconds
        fin = np.clip((t - (DURATION - 2.0)) / 2.0, 0, 1) ** 2
        if fin > 0:
            E *= 1 + 0.5 * fin
        # city sparkle (restrained)
        sp = L["sp_idx"]
        tw = (0.5 + 0.5 * np.sin(L["sp_freq"] * t * 2 * np.pi / 3 + L["sp_phase"])) ** 6 * L["sp_amp"] * 0.55
        visited = (L["sp_vt"] <= t)[:, None]
        col = np.where(visited, GOLD[None] * 1.0 + 0.2, CYAN[None] * 0.7)
        Ef = E.reshape(-1, 3)
        np.add.at(Ef, sp, tw[:, None] * col)
        # dot, trail, jump rings are drawn straight into E (inside small bounding boxes)
        D = E
        x, y, i = self.dot_pos(t)
        trail_s = TRAIL_S * time_scale
        i0 = int(np.searchsorted(L["rt"], t - trail_s, "left"))
        if i0 < i:
            seg_x = np.r_[L["rx"][i0:i], x]; seg_y = np.r_[L["ry"][i0:i], y]
            bx0, by0 = max(int(seg_x.min()) - 3, 0), max(int(seg_y.min()) - 3, 0)
            bx1, by1 = min(int(seg_x.max()) + 4, W), min(int(seg_y.max()) + 4, H)
            tr = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            seg_x = seg_x - bx0; seg_y = seg_y - by0
            jm = np.r_[L["redge"][i0 + 1:i] < 0, False] if i - i0 > 1 else np.zeros(1, bool)
            nseg = len(seg_x) - 1
            if nseg > 0:
                pts = np.round(np.c_[seg_x, seg_y] * 16).astype(np.int32)
                n_bins = 6
                for b in range(n_bins):
                    a_, b_ = b * nseg // n_bins, (b + 1) * nseg // n_bins
                    if b_ <= a_:
                        continue
                    sel = [pts[j:j + 2] for j in range(a_, b_) if not (j < len(jm) and jm[j])]
                    if sel:
                        cv2.polylines(tr, sel, False, int(60 + 195 * (b + 1) / n_bins), 1, cv2.LINE_AA, shift=4)
                trf = tr.astype(np.float32) / 255
                D[by0:by1, bx0:bx1] += trf[..., None] * np.array([1.0, 0.82, 0.5], np.float32) * 1.4
        self._add_sprite(D, x, y, self.dot)
        # jump rings
        jt = L["j_t"]
        win = 0.7 * max(time_scale, 1.0)
        a, b = np.searchsorted(jt, t - win), np.searchsorted(jt, t, "right")
        labels = []
        ring = np.zeros((H, W), np.uint8) if b > a else None
        for j in range(a, b):
            u = (t - jt[j]) / win
            big = min(1.0, L["j_d"][j] / 300)
            for (cx, cy, base, amp) in ((L["j_x0"][j], L["j_y0"][j], 4, 0.35), (L["j_x1"][j], L["j_y1"][j], 3, 0.8)):
                rad = base + (12 + 22 * big) * (1 - (1 - u) ** 2)
                amp = amp * (0.5 + 0.5 * big)
                cv2.circle(ring, (int(cx * 16), int(cy * 16)), int(rad * 16), int(255 * amp * (1 - u) ** 1.5),
                           1, cv2.LINE_AA, shift=4)
            if L["j_d"][j] > 140:
                labels.append((L["j_x1"][j], L["j_y1"][j], (1 - u) ** 0.7))
        # keep only the most recent label (they can overlap during island hopping)
        labels = labels[-1:]
        if b > a:
            ys, xs = np.nonzero(ring)
            if len(ys):
                D[ys, xs] += ring[ys, xs, None].astype(np.float32) / 255 * np.array([0.85, 0.92, 1.0], np.float32) * 0.9
        # final pulse around the finishing point
        if fin > 0:
            u = fin
            ring2 = np.zeros((H, W), np.uint8)
            for rr, aa in ((14 + 80 * u, 1.0), (8 + 44 * u, 0.6)):
                cv2.circle(ring2, (int(x * 16), int(y * 16)), int(rr * 16), int(255 * aa * (1 - 0.6 * u)), 2,
                           cv2.LINE_AA, shift=4)
            ys, xs = np.nonzero(ring2)
            D[ys, xs] += ring2[ys, xs, None].astype(np.float32) / 255 * GOLD * 1.2
            self._add_sprite(D, x, y, self.dot * (1.5 * u))
        em = E
        # bloom: two blur scales combined at half resolution, upsampled once
        small = cv2.resize(em, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 2.2)
        tiny = cv2.resize(small, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        b2 = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
        bloom = cv2.resize(cv2.addWeighted(b1, 0.55, b2, 0.65, 0), (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.add(cv2.add(L["bg"], em), bloom)
        cv2.multiply(img, -1.15, dst=img)
        cv2.exp(img, dst=img)
        cv2.subtract((1.0, 1.0, 1.0, 0.0), img, dst=img)
        # overlays (only inside their bounding boxes)
        for (x0, y0, x1, y1), srgb, sa in self.static:
            reg = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = reg * (1 - sa) + srgb
        dyn, bb = self._dynamic_text(t, labels)
        if bb:
            x0, y0, x1, y1 = bb
            a = np.asarray(dyn.crop(bb)).astype(np.float32) / 255
            reg = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = reg * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
        img = cv2.add(img, self.grain[self.nframe % len(self.grain)])
        self.nframe += 1
        return cv2.convertScaleAbs(img, alpha=255.0, beta=0.5)

    @staticmethod
    def _add_sprite(D, x, y, spr):
        n = spr.shape[0]; h = n // 2
        xi, yi = int(round(x)), int(round(y))
        x0, y0 = xi - h, yi - h
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + n), min(H, y0 + n)
        if dx1 > dx0 and dy1 > dy0:
            D[dy0:dy1, dx0:dx1] += spr[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]


# --------------------------------------------------------------------------- encode
def ffmpeg_proc(path, crf=18, preset="slow"):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-tune", "film", "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def frame_time(f):
    # frame 0 shows 0:00 and the last frame shows exactly 4:40 (route complete)
    return DURATION * f / (NFRAMES - 1)


def render_range(args):
    f0, f1, path = args
    R = Renderer()
    R.reset_to(frame_time(f0) - 1e-9)
    p = ffmpeg_proc(path)
    t0 = time.time()
    for f in range(f0, f1):
        p.stdin.write(R.frame(frame_time(f)).tobytes())
        if (f - f0) % 300 == 0:
            el = time.time() - t0
            print(f"  [{os.path.basename(path)}] frame {f - f0}/{f1 - f0}  {el:.0f}s", flush=True)
    p.stdin.close(); p.wait()
    return path


def video(workers):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    cuts = np.linspace(0, NFRAMES, workers + 1).astype(int)
    jobs = [(int(a), int(b), os.path.join(OUT, f"_part{i}.mp4")) for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
    with Pool(workers) as pool:
        parts = pool.map(render_range, jobs)
    lst = os.path.join(OUT, "_parts.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
    final = os.path.join(OUT, "world_roads_light_speed.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", final], check=True)
    for p in parts:
        os.remove(p)
    os.remove(lst)
    print("wrote", final)


def preview(seconds=15.0):
    os.makedirs(OUT, exist_ok=True)
    n = int(seconds * FPS)
    scale = DURATION / seconds
    R = Renderer(preview=True)
    p = ffmpeg_proc(os.path.join(OUT, "world_roads_preview.mp4"), crf=20, preset="medium")
    for f in range(n):
        t = DURATION * f / (n - 1)
        p.stdin.write(R.frame(t, time_scale=scale / 4).tobytes())
    p.stdin.close(); p.wait()
    print("wrote preview")


def stills(times):
    os.makedirs(OUT, exist_ok=True)
    R = Renderer()
    for t in times:
        R.reset_to(t - 0.5)
        for tt in np.linspace(t - 0.5, t, 4):
            img = R.frame(float(tt))
        path = os.path.join(OUT, f"still_{int(t // 60)}m{int(t % 60):02d}s.png")
        cv2.imwrite(path, img[..., ::-1])
        print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "video", "preview", "still"])
    ap.add_argument("times", nargs="*", type=float)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "video":
        video(a.workers)
    elif a.cmd == "preview":
        preview()
    else:
        stills(a.times or [0, 40, 80, 120, 160, 200, 240, 279.9])
