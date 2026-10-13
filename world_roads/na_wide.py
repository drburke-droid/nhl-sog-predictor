"""Wide (1920x1080) North America cut of the road film, built from 2x-resolution layers.

    python na_wide.py            -> out/na_wide.mp4   (real time 0:00-0:50 + end card)
    python na_wide.py still 30   -> PNG at t = 30 s
"""
import os, sys, json, subprocess
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import render
from render import CLASS_W, CLASS_T, CYAN, GOLD, HOT, DATA, OUT, FPS, EST_TOTAL_KM
from proj import W, H

S = 2                                    # supersample of the source canvas
CX0, CY0, CW, CH = 140, 336, 1440, 810   # crop on the 2x canvas (= 70..790 x 168..573 at 1x)
OW, OH = 1920, 1080
T_END = 50.0
END_S = 3.0
DURATION = render.DURATION
FD = "/usr/share/fonts/opentype/inter"


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def build():
    cache = os.path.join(DATA, "na_wide_layers.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    g = dict(np.load(os.path.join(DATA, "graph.npz")))
    r = dict(np.load(os.path.join(DATA, "route.npz")))
    ea, eb, ecls = g["ea"], g["eb"], g["ecls"]
    edge, cum = r["edge"], r["cum"]
    T = DURATION * cum / cum[-1]
    Tprev = np.r_[0.0, T[:-1]]
    tv = np.full(len(ea), np.inf)
    st = np.nonzero(edge >= 0)[0][::-1]
    tv[edge[st]] = 0.5 * (Tprev[st] + T[st])
    P = np.rint(np.stack([np.c_[g["nx"][ea], g["ny"][ea]], np.c_[g["nx"][eb], g["ny"][eb]]], 1)).astype(np.int32)
    I8 = np.zeros((H * S, W * S), np.uint8)
    for c in sorted(CLASS_W, key=lambda c: CLASS_W[c]):
        sel = np.nonzero(ecls == c)[0]
        cv2.polylines(I8, list(P[sel]), False, int(255 * CLASS_W[c]), CLASS_T[c], cv2.LINE_AA, shift=1)
    vt2 = np.full((H * S, W * S), np.inf, np.float32)
    fb = np.floor(tv * FPS).astype(np.int64)
    order = np.argsort(-fb, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(fb[order]))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = order[a:b]
        cv2.polylines(vt2, list(P[sel]), False, float(tv[sel].min()), 2, cv2.LINE_8, shift=1)
    Imaj = I8.astype(np.float32) / 255
    d = np.load(os.path.join(DATA, "density.npz"))["dens"]
    land2 = cv2.imread(os.path.join(DATA, "land2x.png"), 0)
    minor = cv2.GaussianBlur(d[1] + d[2], (0, 0), 0.6)
    Imin = 0.36 * np.clip(np.log1p(minor / 1.5) / np.log1p(400 / 1.5), 0, 1) ** 1.15
    Imin *= (land2 > 0) | (Imaj > 0)
    I2 = np.maximum(Imaj, Imin)
    sl = (slice(CY0, CY0 + CH), slice(CX0, CX0 + CW))
    vt_c, I_c = vt2[sl], I2[sl]
    miss = ~np.isfinite(vt_c)
    idx = ndimage.distance_transform_edt(miss, return_distances=False, return_indices=True)
    vt_c = vt_c[idx[0], idx[1]]
    lit = I_c > 0.012
    yy, xx = np.nonzero(lit)
    pv = vt_c[yy, xx].astype(np.float32); pi = I_c[yy, xx].astype(np.float32)
    p1 = (yy * CW + xx).astype(np.int32)
    o = np.argsort(pv, kind="stable")
    L = dict(pv=pv[o], pi=pi[o], p1=p1[o],
             Itot=np.bincount(p1, weights=pi, minlength=CW * CH).astype(np.float32))
    bg = np.load(os.path.join(DATA, "layers.npz"))["bg"]
    L["bg"] = cv2.resize(bg, (W * S, H * S), interpolation=cv2.INTER_LINEAR)[sl].astype(np.float32)
    np.savez(cache, **L)
    return L


class NARenderer:
    def __init__(self):
        self.L = build()
        self.R = render.Renderer()          # route geometry, dot timing, sprites
        self.Ig = np.zeros(CW * CH, np.float32)
        self.k = 0
        n = 161
        k = np.arange(n, dtype=np.float32) - n // 2
        r2 = k[None] ** 2 + k[:, None] ** 2
        self.dot = (np.exp(-r2 / (2 * 2.0 ** 2))[..., None] * np.array([1, .97, .9]) * 2.6 +
                    np.exp(-r2 / (2 * 6.0 ** 2))[..., None] * np.array([1, .72, .3]) * 1.25 +
                    np.exp(-r2 / (2 * 24 ** 2))[..., None] * np.array([1, .52, .12]) * 0.42).astype(np.float32)

    def reset_to(self, t):
        L = self.L
        k = int(np.searchsorted(L["pv"], t, "right"))
        self.Ig = np.bincount(L["p1"][:k], weights=L["pi"][:k], minlength=CW * CH).astype(np.float32)
        self.k = k

    def to_crop(self, x, y):
        return x * S - CX0, y * S - CY0

    def frame(self, t, end_u=None):
        L, RL = self.L, self.R.L
        k = int(np.searchsorted(L["pv"], t, "right"))
        if k > self.k:
            np.add.at(self.Ig, L["p1"][self.k:k], L["pi"][self.k:k]); self.k = k
        kf = int(np.searchsorted(L["pv"], t - 0.9, "right"))
        F = np.zeros(CW * CH, np.float32)
        if k > kf:
            age = (t - L["pv"][kf:k]) / 0.9
            np.add.at(F, L["p1"][kf:k], L["pi"][kf:k] * (1 - age) ** 2 * 1.2)
        Ic = (L["Itot"] - self.Ig).reshape(CH, CW); Ig = self.Ig.reshape(CH, CW); F2 = F.reshape(CH, CW)
        E = cv2.merge([cv2.scaleAdd(F2, float(HOT[c]), cv2.addWeighted(Ic, float(CYAN[c] * 0.30), Ig, float(GOLD[c] * 0.52), 0))
                       for c in range(3)])
        if end_u is None:
            x, y, i = self.R.dot_pos(t)
            i0 = int(np.searchsorted(RL["rt"], t - 0.12, "left"))
            if i0 < i:
                xs, ys = self.to_crop(np.r_[RL["rx"][i0:i], x], np.r_[RL["ry"][i0:i], y])
                jm = np.r_[RL["redge"][i0 + 1:i] < 0, False] if i - i0 > 1 else np.zeros(1, bool)
                tr = np.zeros((CH, CW), np.uint8)
                pts = np.round(np.c_[xs, ys] * 16).astype(np.int32)
                nseg = len(pts) - 1
                for b in range(6):
                    a_, b_ = b * nseg // 6, (b + 1) * nseg // 6
                    segs = [pts[j:j + 2] for j in range(a_, b_) if not (j < len(jm) and jm[j])]
                    if segs:
                        cv2.polylines(tr, segs, False, int(60 + 195 * (b + 1) / 6), 2, cv2.LINE_AA, shift=4)
                E += tr.astype(np.float32)[..., None] / 255 * np.array([1.0, 0.82, 0.5], np.float32) * 1.3
            cx, cy = self.to_crop(x, y)
            n = self.dot.shape[0]; h = n // 2
            xi, yi = int(round(cx)), int(round(cy))
            a0, a1, b0, b1 = max(yi - h, 0), min(yi + h + 1, CH), max(xi - h, 0), min(xi + h + 1, CW)
            if a1 > a0 and b1 > b0:
                E[a0:a1, b0:b1] += self.dot[a0 - (yi - h):a1 - (yi - h), b0 - (xi - h):b1 - (xi - h)]
            # jump rings
            jt = RL["j_t"]
            a_, b_ = np.searchsorted(jt, t - 0.7), np.searchsorted(jt, t, "right")
            if b_ > a_:
                ring = np.zeros((CH, CW), np.uint8)
                for j in range(a_, b_):
                    u = (t - jt[j]) / 0.7
                    for (px, py, amp) in ((RL["j_x0"][j], RL["j_y0"][j], 0.35), (RL["j_x1"][j], RL["j_y1"][j], 0.8)):
                        qx, qy = self.to_crop(px, py)
                        if 0 <= qx < CW and 0 <= qy < CH:
                            rad = 6 + 50 * (1 - (1 - u) ** 2)
                            cv2.circle(ring, (int(qx * 16), int(qy * 16)), int(rad * 16), int(255 * amp * (1 - u) ** 1.5), 2, cv2.LINE_AA, shift=4)
                E += ring.astype(np.float32)[..., None] / 255 * np.array([0.85, 0.92, 1.0], np.float32) * 0.9
        small = cv2.resize(E, (CW // 2, CH // 2), interpolation=cv2.INTER_AREA)
        b1_ = cv2.GaussianBlur(small, (0, 0), 3.0)
        tiny = cv2.resize(small, (CW // 8, CH // 8), interpolation=cv2.INTER_AREA)
        b2_ = cv2.resize(cv2.GaussianBlur(tiny, (0, 0), 3.0), (CW // 2, CH // 2))
        bloom = cv2.resize(cv2.addWeighted(b1_, 0.35, b2_, 0.45, 0), (CW, CH))
        img = 1 - np.exp(-(L["bg"] + E + bloom) * 1.15)
        img = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_CUBIC)
        if end_u is not None:
            img *= 1 - 0.55 * min(1.0, end_u / 0.4)
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((58, 40), "Every road in North America at light speed", font=font("Inter-SemiBold.otf", 46), fill=(236, 242, 250))
        d.text((60, 100), "Real time · the first minute of a 4 min 40 s trip around the world", font=font("Inter-Regular.otf", 25), fill=(150, 176, 205))
        tt = min(t, T_END)
        right = OW - 60
        d.text((right, 896), "ELAPSED", font=font("Inter-Medium.otf", 15), fill=(130, 150, 175), anchor="ra")
        d.text((right, 912), f"{int(tt // 60)}:{int(tt % 60):02d}", font=font("InterDisplay-Medium.otf", 66), fill=(246, 236, 214), anchor="ra")
        d.text((right, 1000), f"≈ {tt / DURATION * EST_TOTAL_KM / 1e6:.1f} M km hypothetical distance at c",
               font=font("Inter-Regular.otf", 17), fill=(160, 178, 200), anchor="ra")
        d.text((60, 1040), "Road data © OpenStreetMap contributors via Overture Maps · Thought experiment: all figures are estimates.",
               font=font("Inter-Regular.otf", 14), fill=(120, 140, 165))
        if end_u is not None:
            a = int(255 * min(1.0, end_u / 0.4))
            ov = Image.new("RGBA", (OW, OH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            od.text((OW // 2, 440), "North America: about a minute.", font=font("Inter-Bold.otf", 64), fill=(255, 186, 80, a), anchor="ma")
            od.text((OW // 2, 530), "Every road on Earth: 4 minutes 40 seconds.", font=font("Inter-Medium.otf", 40), fill=(236, 242, 250, a), anchor="ma")
            im.paste(ov, (0, 0), ov)
        return np.asarray(im)


def main():
    R = NARenderer()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "na_wide.mp4")
    p = render.ffmpeg_proc_size(path, OW, OH)
    n_main = int(T_END * FPS) + 1
    for f in range(n_main):
        p.stdin.write(R.frame(f / FPS).tobytes())
    for f in range(int(END_S * FPS)):
        p.stdin.write(R.frame(T_END, end_u=f / FPS).tobytes())
    p.stdin.close(); p.wait()
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        R = NARenderer()
        for s in sys.argv[2:]:
            t = float(s)
            R.reset_to(t - 1.0)
            for tt in np.arange(t - 1.0, t, 1 / FPS):
                R.frame(float(tt))
            cv2.imwrite(os.path.join(OUT, f"na_still_{s}.png"), R.frame(t)[..., ::-1])
            print("still", s)
    else:
        main()
