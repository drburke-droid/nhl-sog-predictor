"""Zoomed views for the coast film: the same map drawn straight from the network's vectors through a camera
(centre and zoom over the world map), so the film can open on one river in Calgary and pull back to the
whole world. Close in, a river is drawn as water between two banks: the bank the fish swims up turns gold
on the way up, the other on the way back down.

    python zoom.py prepare     -> data/zoom_r{RES}.npz (world-pixel vertices, times; fine fish path near Calgary)
"""
import os, sys, math, time
import numpy as np
import numba as nb
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cproj import W, H, RES, to_px  # noqa: E402
import render as R  # noqa: E402

DATA = R.DATA
C_KMS = R.C_KMS
ZFILE = os.path.join(DATA, f"zoom_r{RES}.npz")
CALGARY_BOX = (-117.5, 49.0, -110.0, 53.2)         # where the close-up views can look


@nb.njit(cache=True)
def draw_view(wx, wy, segj, tfw, tbw, kind, cx, cy, zoom, t, flash_t, bank_px, SS, Ic, Ig, F, Wt):
    """Rasterise every segment in view on a supersampled canvas. Screen = (world - centre) * zoom + half
    screen. Rivers: two banks offset bank_px either side (0 = one line); the right bank of the drawing
    direction lights at the forward pass, the left one at the backward pass. Shores: one line."""
    Hh, Ww = Ic.shape
    hx, hy = Ww / (2.0 * SS), Hh / (2.0 * SS)
    margin = 4.0 / zoom
    xmin, xmax = cx - hx / zoom - margin, cx + hx / zoom + margin
    ymin, ymax = cy - hy / zoom - margin, cy + hy / zoom + margin
    inten = (1.0, 0.8, 0.6, 0.6)
    for k in range(len(segj)):
        j = segj[k]
        x0w, y0w, x1w, y1w = wx[j], wy[j], wx[j + 1], wy[j + 1]
        if (x0w < xmin and x1w < xmin) or (x0w > xmax and x1w > xmax) or (y0w < ymin and y1w < ymin) or (y0w > ymax and y1w > ymax):
            continue
        if abs(x1w - x0w) > Ww / (4.0 * SS):
            continue                                  # crosses the map's edge
        X0 = ((x0w - cx) * zoom + hx) * SS
        Y0 = ((y0w - cy) * zoom + hy) * SS
        X1 = ((x1w - cx) * zoom + hx) * SS
        Y1 = ((y1w - cy) * zoom + hy) * SS
        kd = kind[k]
        I = inten[kd]
        river = kd >= 2
        nbank = 2 if (river and bank_px > 0.05) else 1
        dx, dy = X1 - X0, Y1 - Y0
        L = math.hypot(dx, dy)
        if L > 0:
            nx_, ny_ = -dy / L, dx / L                 # right of the drawing direction (y down)
        else:
            nx_, ny_ = 0.0, 0.0
        for b in range(nbank):
            if nbank == 1:
                off = 0.0
                if river:
                    ta, tb2 = min(tfw[k], tbw[k]), max(tfw[k], tbw[k])
                    if t >= tb2:
                        g, c = 1.0, 0.0
                    elif t >= ta:
                        g, c = 0.5, 0.0
                    else:
                        g, c = 0.0, 1.0
                    tp = ta if t < tb2 else tb2
                else:
                    tp = min(tfw[k], tbw[k])
                    g, c = (1.0, 0.0) if t >= tp else (0.0, 1.0)
            else:
                off = bank_px * SS * (1.0 if b == 0 else -1.0)
                tp = tfw[k] if b == 0 else tbw[k]
                g, c = (1.0, 0.0) if t >= tp else (0.0, 1.0)
            fl = 0.0
            if flash_t > 0 and t >= tp and t - tp < flash_t:
                u = (t - tp) / flash_t
                fl = (1 - u) * (1 - u) * 2.4
            ax, ay = X0 + nx_ * off, Y0 + ny_ * off
            n = int(max(abs(dx), abs(dy))) + 1
            th = 2 if (kd <= 1 or zoom > 3.0) else 1
            if zoom > 3.0 and Ww > 5000:
                th = 3                                # 4K close-ups: keep the lines' width on screen
            for i in range(n + 1):
                x = int(ax + dx * i / n + 0.5)
                y = int(ay + dy * i / n + 0.5)
                for oy in range(th):
                    for ox in range(th):
                        xx, yy = x + ox - th // 2, y + oy - th // 2
                        if 0 <= xx < Ww and 0 <= yy < Hh:
                            if I * c > Ic[yy, xx]:
                                Ic[yy, xx] = I * c
                            if I * g > Ig[yy, xx]:
                                Ig[yy, xx] = I * g
                            if I * fl > F[yy, xx]:
                                F[yy, xx] = I * fl
            # water between the banks
            if nbank == 2 and b == 0:
                wwid = int(2 * bank_px * SS)
                for i in range(n + 1):
                    x = int(X0 + dx * i / n + 0.5)
                    y = int(Y0 + dy * i / n + 0.5)
                    for oy in range(-wwid // 2, wwid // 2 + 1):
                        for ox in range(-wwid // 2, wwid // 2 + 1):
                            xx, yy = x + ox, y + oy
                            if 0 <= xx < Ww and 0 <= yy < Hh:
                                Wt[yy, xx] = 1.0


def prepare():
    t0 = time.time()
    z = np.load(os.path.join(DATA, "network.npz"))
    gxy, goff, kind = z["gxy"], z["goff"], z["kind"]
    tz = np.load(os.path.join(DATA, "tour.npz"))
    tf, tb = tz["tf"] / C_KMS, tz["tb"] / C_KMS
    wx, wy = to_px(gxy[:, 0], gxy[:, 1], 1)
    ek = np.repeat(kind, np.diff(goff))
    last = np.zeros(len(gxy), bool)
    last[goff[1:] - 1] = True
    segj = np.nonzero(~last)[0]
    # untraced (unreached) segments keep inf times
    s_tf = tf[segj].astype(np.float32)
    s_tb = tb[segj + 1].astype(np.float32)
    s_k = ek[segj].astype(np.int8)
    # shores are walked one way: both "times" are that pass
    one = s_k <= 1
    tt = np.minimum(s_tf, s_tb)
    s_tf = np.where(one, tt, s_tf)
    s_tb = np.where(one, tt, s_tb)
    # the close-up region (Calgary to the Rockies)
    x0, y0, x1, y1 = CALGARY_BOX
    inbox = (gxy[segj, 0] > x0) & (gxy[segj, 0] < x1) & (gxy[segj, 1] > y0) & (gxy[segj, 1] < y1)
    loc = segj[inbox]
    # fine path near the start and the end: every vertex pass, with the side of the river it is on
    T = np.r_[tf, tb]
    V = np.r_[np.arange(len(tf)), np.arange(len(tb))]
    side = np.r_[np.zeros(len(tf), np.int8), np.ones(len(tb), np.int8)]
    total = float(tz["total"]) / C_KMS
    ok = np.isfinite(T) & ((T < 1.5) | (T > total - 1.5))
    T, V, side = T[ok], V[ok], side[ok]
    o = np.argsort(T, kind="stable")
    T, V, side = T[o], V[o], side[o]
    river_v = ek[V] >= 2
    np.savez(ZFILE, wx=wx, wy=wy, segj=segj.astype(np.int64), s_tf=s_tf, s_tb=s_tb, s_k=s_k,
             loc_segj=loc.astype(np.int64), loc_tf=s_tf[inbox], loc_tb=s_tb[inbox], loc_k=s_k[inbox],
             fp_t=T, fp_x=wx[V], fp_y=wy[V], fp_side=side, fp_river=river_v, total=total)
    print(f"zoom data: {len(segj):,} segments, {len(loc):,} near Calgary, fine path {len(T):,} ({time.time() - t0:.0f}s)")


class Zoom:
    def __init__(self, bg_world):
        Z = dict(np.load(ZFILE))
        self.Z = Z
        self.bg = bg_world

    def emission(self, t, cx, cy, zoom, flash_t, bank_px, local=None):
        Z = self.Z
        SS = 2
        Ic = np.zeros((H * SS, W * SS), np.float32)
        Ig = np.zeros_like(Ic); F = np.zeros_like(Ic); Wt = np.zeros_like(Ic)
        if local is None:
            local = zoom > 60
        if local:
            draw_view(Z["wx"], Z["wy"], Z["loc_segj"], Z["loc_tf"], Z["loc_tb"], Z["loc_k"], cx, cy, zoom, t, flash_t,
                      bank_px, SS, Ic, Ig, F, Wt)
        else:
            draw_view(Z["wx"], Z["wy"], Z["segj"], Z["s_tf"], Z["s_tb"], Z["s_k"], cx, cy, zoom, t, flash_t,
                      bank_px, SS, Ic, Ig, F, Wt)
        def down(a):
            return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        Ic, Ig, F, Wt = down(Ic), down(Ig), down(F), down(Wt)
        # thin lines on the 2x canvas average down; the world raster keeps intensity per 1x pixel the same way
        # close in, single lines would read fainter than the dense world map: lift them as the camera closes in
        lift = (1.0 + 1.3 * min(1.0, max(0.0, math.log(max(zoom, 1.0)) / math.log(40.0)))) * R.LINE_GAIN
        E = cv2.merge([F * float(R.HOT[c]) + (Ic * float(R.CYAN[c] * 0.42) + Ig * float(R.GOLD[c])) * lift for c in range(3)])
        # background: the world background through the same camera, river water on top
        M = np.array([[zoom, 0, W / 2 - cx * zoom], [0, zoom, H / 2 - cy * zoom]], np.float32)
        bg = cv2.warpAffine(self.bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        water = np.array([0.010, 0.030, 0.070], np.float32)
        bg = bg * (1 - Wt[..., None]) + Wt[..., None] * water
        return E, bg

    def fish(self, t):
        """fish position (world px), drawing direction, and which side of its river it swims on"""
        Z = self.Z
        ft = Z["fp_t"]
        i = int(np.searchsorted(ft, t, "left"))
        i = min(max(i, 1), len(ft) - 1)
        t0, t1 = ft[i - 1], ft[i]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        x = Z["fp_x"][i - 1] + u * (Z["fp_x"][i] - Z["fp_x"][i - 1])
        y = Z["fp_y"][i - 1] + u * (Z["fp_y"][i] - Z["fp_y"][i - 1])
        # direction of travel over a short look-around
        a, b = max(i - 3, 0), min(i + 2, len(ft) - 1)
        dx, dy = Z["fp_x"][b] - Z["fp_x"][a], Z["fp_y"][b] - Z["fp_y"][a]
        return float(x), float(y), float(dx), float(dy), bool(Z["fp_river"][i])

    def trail(self, t, t_back):
        Z = self.Z
        ft = Z["fp_t"]
        a, b = int(np.searchsorted(ft, t - t_back)), int(np.searchsorted(ft, t))
        return Z["fp_x"][a:b], Z["fp_y"][a:b], ft[a:b]


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
