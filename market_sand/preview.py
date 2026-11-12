"""Flat top-down preview of the whole film (layout, clock, camera, husks and flows) before any Blender work.

    python3 preview.py still 1602.8 1720.5 1911.5 1989.9 2026.8   # PNGs -> out/still_<year>.png
    python3 preview.py sheet                                       # 20 moments -> out/contact_sheet.png
    python3 preview.py video [--fps 24] [--size 1280x720] [--workers 4]   # -> out/preview.mp4

What it shows:
- the nested grid of plots (layout.py), oldest top left, newest bottom right, with canals between them;
  plots of companies still to come and vacant plots are faint, so the grid reads from the start;
- each company's sand as an organic pile (piles.py) in the middle of its plot, coloured by industry and lit
  from the upper left; close up it is drawn grain by grain;
- a faint outline on the plot at the widest the pile has been (its tide mark), so a husk shows how big it
  once was; a company that has died keeps a dark plot, its name and its years;
- pale, uncoloured sand drifting down the canals toward the lower right (the market): sparse while there is
  little money on the table, denser as the total grows;
- grains lifting from the canal into a growing company and draining back from a shrinking one (pale, taking
  the company's colour as they land); grains flying straight from one company to another only for a real
  deal (merger, purchase, break-up), in the giving company's colour;
- the camera frames every company that has mattered so far: it opens close on the VOC in the top left
  corner and only ever pulls back, ending on the whole grid.
"""
import argparse, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import layout, piles, story
from clock import Clock, FINALE
from story import Story, INDUSTRIES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIR = "/usr/share/fonts/opentype/inter"
WATER = np.array([7, 9, 12], np.float32)
STREAM = np.array([205, 190, 150], np.float32)       # pale uncoloured sand: money not in any company
LAND = (25, 27, 32)
LAND_DEAD = (17, 18, 21)
LAND_FUTURE = (13, 15, 18)
INK, MUTED = (232, 238, 248), (147, 161, 184)
FLY_S = 0.9           # seconds a grain is in flight
MIN_VIEW = 26.0       # narrowest view, cells across
SAFE = (0.20, 0.17, 0.97, 0.95)   # where the content may sit (fractions of the frame: x0, y0, x1, y1)
LIGHT = np.array([-0.45, -0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)
TITLE = "424 years of the world's biggest companies"
OPEN_ON = "voc"


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), max(6, int(size)))


def hash2(i, j, seed=0.0):
    return np.modf(np.abs(np.sin(i * 12.9898 + j * 78.233 + seed * 3.17) * 43758.5453))[0]


def smooth(x, n):
    """Centred Gaussian smoothing over about n samples (edges padded)."""
    if n < 2:
        return x
    k = np.exp(-0.5 * (np.arange(-2 * n, 2 * n + 1) / (n / 2)) ** 2); k /= k.sum()
    pad = np.r_[np.full(2 * n, x[0]), x, np.full(2 * n, x[-1])]
    return np.convolve(pad, k, mode="valid")


class Film:
    def __init__(self, fps=24):
        self.fps = fps
        self.S = S = Story()
        self.C = C = Clock(S.now)
        self.ids = sorted(S.T, key=lambda k: S.start[k])
        self.shape = [piles.Shape(k) for k in self.ids]
        ext = {k: self.shape[i].extent(S.peak_grains[k]) for i, k in enumerate(self.ids)}
        self.L = layout.build([dict(id=k, first=S.start[k]) for k in self.ids], ext)
        self.nf = int(math.ceil(C.duration * fps))
        self.years = np.array([min(C.year(f / fps), S.now) for f in range(self.nf)])
        self.G = np.array([[S.grains_one(k, y) for k in self.ids] for y in self.years], np.float32)
        self.PK = np.maximum.accumulate(self.G, axis=0)                     # the most each has held so far
        self.EXT = np.stack([self.shape[i].extent_v(self.G[:, i]) for i in range(len(self.ids))], 1)
        self.col = np.array([hexrgb(INDUSTRIES[S.companies[k]["industry"]][1]) for k in self.ids])
        self.pos = np.array([self.L.lots[k][:2] for k in self.ids])
        self.half = np.array([self.L.lots[k][2] for k in self.ids])
        self._camera()
        self._flows()

    # ---- camera: frame everything that has mattered so far; it only ever pulls back ----
    def _camera(self):
        G = self.G
        gmax = G.max(axis=1, keepdims=True)
        relevant = G >= np.maximum(0.5, 0.04 * gmax)
        relevant[:, self.ids.index(OPEN_ON)] = True
        seen = np.maximum.accumulate(relevant, axis=0)
        reach = np.maximum(np.maximum.accumulate(self.EXT, axis=0), 2.0) * 1.15 + 1.5
        x0 = np.where(seen, self.pos[:, 0] - reach, np.inf).min(axis=1)
        x1 = np.where(seen, self.pos[:, 0] + reach, -np.inf).max(axis=1)
        y0 = np.where(seen, self.pos[:, 1] - reach, np.inf).min(axis=1)
        y1 = np.where(seen, self.pos[:, 1] + reach * 1.3, -np.inf).max(axis=1)     # room for names below
        sw, sh = SAFE[2] - SAFE[0], SAFE[3] - SAFE[1]
        w = np.maximum.accumulate(np.maximum(np.maximum((x1 - x0) / sw, (y1 - y0) / sh * 16 / 9), MIN_VIEW))
        cx = (x0 + x1) / 2 - w * ((SAFE[0] + SAFE[2]) / 2 - 0.5)
        cy = (y0 + y1) / 2 - w * 9 / 16 * ((SAFE[1] + SAFE[3]) / 2 - 0.5)
        n_fin = int(FINALE * self.fps)                                       # finale: ease out to the whole grid
        whole_w = max(self.L.width * 1.03 / sw, self.L.height * 1.03 / sh * 16 / 9)
        e = np.clip((np.arange(self.nf) - (self.nf - n_fin)) / (n_fin * 0.4), 0, 1)
        e = e * e * (3 - 2 * e)
        wcx = self.L.width / 2 - whole_w * ((SAFE[0] + SAFE[2]) / 2 - 0.5)
        wcy = self.L.height / 2 - whole_w * 9 / 16 * ((SAFE[1] + SAFE[3]) / 2 - 0.5)
        lw = np.log(w) * (1 - e) + np.log(whole_w) * e
        cx, cy = cx * (1 - e) + wcx * e, cy * (1 - e) + wcy * e
        n = int(3.0 * self.fps)
        self.cam = np.stack([smooth(cx, n), smooth(cy, n), np.exp(smooth(lw, n))], axis=1)

    # ---- flows: grains lifting from the canal, draining to it, and flying between companies in deals ----
    def _flows(self):
        S, fps = self.S, self.fps
        idx = {k: i for i, k in enumerate(self.ids)}
        quay = np.c_[self.pos[:, 0] + self.half + layout.GAP / 2, self.pos[:, 1]]     # the canal on the plot's right
        rec = []      # (frame, src_x, src_y, dst_x, dst_y, kind, company index)
        acc = np.zeros(len(self.ids))
        handed = {idx[k] for k in S.hands_over}
        for f in range(1, self.nf):
            d = self.G[f] - self.G[f - 1]
            acc += np.minimum(5.0, 0.35 * np.abs(d) ** 0.5) * (np.abs(d) > 0.02)
            for i in np.nonzero(acc >= 1)[0]:
                n = int(acc[i]); acc[i] -= n
                if d[i] < 0 and i in handed and self.years[f] >= S.end[self.ids[i]]:
                    continue                      # its sand is going to the buyer (drawn below)
                for _ in range(n):
                    rec.append((f, *quay[i], *self.pos[i], 0, i) if d[i] > 0 else (f, *self.pos[i], *quay[i], 1, i))
        for ev in S.events:
            if ev["type"] not in ("merger", "acquisition", "split", "spinoff"):
                continue
            src = [idx[k] for k in ev["from_ids"].split(";") if k in idx]
            dst = [idx[k] for k in ev["to_ids"].split(";") if k in idx]
            if not src or not dst:
                continue
            f0 = int(self.C.seconds(ev["t"]) * fps)
            for j in range(int(2.0 * fps)):
                for s in src:
                    for t in dst:
                        for _ in range(3):
                            rec.append((f0 + j, *self.pos[s], *self.pos[t], 2, s))
        rec.sort(key=lambda r: r[0])
        self.fl = np.array(rec, np.float64).reshape(-1, 7)

    # ---- one frame ----
    def frame(self, f, size=(1280, 720)):
        Wp, Hp = size
        cx, cy, w = self.cam[f]
        sc = Wp / w
        X0, Y0 = cx - w / 2, cy - Hp / sc / 2
        t_film = f / self.fps
        year = self.years[f]
        G = self.G[f]
        stream = float(np.clip((G.sum() / self.G.sum(axis=1).max()) ** (1 / 3), 0.05, 0.8))

        def to_px(x, y):
            return (x - X0) * sc, (y - Y0) * sc

        # water, with pale sand drifting down the canals toward the lower right
        img = np.empty((Hp, Wp, 3), np.float32); img[:] = WATER
        WX, WY = np.meshgrid(X0 + (np.arange(Wp) + 0.5) / sc, Y0 + (np.arange(Hp) + 0.5) / sc)
        along = (WX + WY) * 0.7071 - t_film * 2.0
        across = (WX - WY) * 0.7071
        gi, gj = np.floor(along * 1.1), np.floor(across * 1.1)
        fu, fv = np.modf(along * 1.1 + 1e5)[0] - 0.5, np.modf(across * 1.1 + 1e5)[0] - 0.5
        rr = (fu ** 2 + fv ** 2) / 0.16
        dots = hash2(gi, gj, 7.0) < 0.03 + 0.32 * stream
        if sc > 2.5:
            dots &= rr < 1
            k = (0.35 + 0.5 * np.sqrt(np.clip(1 - rr, 0, 1)))[dots, None]
        else:
            k = 0.4
        vis_k = float(np.clip(sc / 3.0, 0.25, 1.0))
        img[dots] = img[dots] * (1 - 0.6 * vis_k) + STREAM[None, :] * 0.55 * vis_k * k

        # plots: faint for vacant ones and companies still to come; land for the living; dark for husks
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil, "RGBA")

        def plot(x, y, hh, fill, outline=None):
            a, b = to_px(x - hh, y - hh)
            c, d = to_px(x + hh, y + hh)
            if c < 0 or a > Wp or d < 0 or b > Hp or c - a < 1:
                return
            dr.rounded_rectangle([a, b, c, d], radius=max(0, (c - a) * 0.06), fill=fill, outline=outline,
                                 width=max(1, int(sc * 0.15)) if outline else 0)

        for x, y, hh in self.L.vacant:
            plot(x, y, hh, LAND_FUTURE)
        vis = []
        for i, k in enumerate(self.ids):
            x, y = self.pos[i]; hh = self.half[i]
            if year < self.S.start[k] - story.RAMP:
                plot(x, y, hh, LAND_FUTURE)
                continue
            dead = year > self.S.end[k]
            plot(x, y, hh, LAND_DEAD if dead else LAND, (34, 37, 44) if not dead else None)
            px, py = to_px(x, y)
            e = max(self.EXT[f, i], 1.0) * sc * 1.2 + 4
            if px + e > 0 and px - e < Wp and py + e > 0 and py - e < Hp:
                vis.append(i)
        img = np.asarray(pil, np.float32).copy()

        # tide marks: the outline of the widest the pile has been, on the plot
        for i in vis:
            pk, n = float(self.PK[f, i]), float(G[i])
            if pk < 1 or pk < n * 1.12:
                continue
            sh = self.shape[i]
            Rp = sh.radius(pk); E = sh.extent(pk) * sc
            if E < 3:
                continue
            px, py = to_px(*self.pos[i])
            i0, i1 = max(0, int(px - E - 2)), min(Wp, int(px + E + 3))
            j0, j1 = max(0, int(py - E - 2)), min(Hp, int(py + E + 3))
            if i0 >= i1 or j0 >= j1:
                continue
            U, V = np.meshgrid((X0 + (np.arange(i0, i1) + 0.5) / sc - self.pos[i][0]) / Rp,
                               (Y0 + (np.arange(j0, j1) + 0.5) / sc - self.pos[i][1]) / Rp)
            Hk = sh.field(U, V, pk)
            eps = min(0.2, (1.6 / max(Rp * sc, 1e-6)) ** 1.35)
            ring = (Hk > 0) & (Hk < eps)
            sub = img[j0:j1, i0:i1]
            sub[ring] = sub[ring] * 0.55 + self.col[i] * 0.45 * 0.6

        # shadows, then the piles: an organic height field, lit from the upper left, grain by grain close up
        for i in vis:
            n = float(G[i])
            if n < 0.5:
                continue
            sh = self.shape[i]
            R = sh.radius(n)
            E = sh.extent(n) * sc
            px, py = to_px(*self.pos[i])
            col = self.col[i]
            if E < 1.2:
                ii, jj = int(px), int(py)
                if 0 <= ii < Wp and 0 <= jj < Hp:
                    img[jj, ii] = img[jj, ii] * 0.3 + col * 0.7
                continue
            off = 0.5 * E
            i0, i1 = max(0, int(px - E - 1)), min(Wp, int(px + E + off + 2))
            j0, j1 = max(0, int(py - E - 1)), min(Hp, int(py + E + off + 2))
            if i0 >= i1 or j0 >= j1:
                continue
            DX, DY = np.meshgrid(X0 + (np.arange(i0, i1) + 0.5) / sc - self.pos[i][0],
                                 Y0 + (np.arange(j0, j1) + 0.5) / sc - self.pos[i][1])
            H = sh.field(DX / R, DY / R, n)
            Hs = sh.field((DX - off / sc) / R, (DY - off * 0.8 / sc) / R, n)          # shadow: the pile shifted
            sub = img[j0:j1, i0:i1]
            sub *= (1 - 0.5 * np.clip(Hs * 6, 0, 1))[..., None]
            inside = H > 0
            if not inside.any():
                continue
            z = piles.A * R * H * piles.LEVEL
            # grains on a hex grid anchored to the plot (they don't swim as the camera moves)
            vv = DY / 0.866
            row = np.floor(vv)
            uu = DX + 0.5 * (row % 2)
            gi = np.floor(uu)
            rough = (hash2(gi, row, i + 0.5) - 0.5) * 0.24 * piles.LEVEL * np.clip(H * 4, 0, 1)   # grain roughness
            gy, gx = np.gradient(z + rough * float(np.clip(1.5 - sc / 4, 0, 1)), 1 / sc)
            nrm = np.stack([-gx, -gy, np.ones_like(z)], -1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
            lit = np.clip(nrm @ LIGHT, 0, 1)
            fu, fv = uu - gi - 0.5, vv - row - 0.5
            r2 = 4 * (fu ** 2 + fv ** 2)
            sph = np.stack([2 * fu, 2 * fv, np.sqrt(np.clip(1 - r2, 0, 1))], -1)
            lam = np.clip(sph @ LIGHT, 0, 1) * (r2 < 1)
            var = 0.86 + 0.28 * hash2(gi, row, i)
            wgr = float(np.clip((sc - 2.5) / 6.0, 0, 1))
            shade = (1 - wgr) * (0.25 + 0.85 * lit) + wgr * (0.10 + 0.95 * lam * (0.45 + 0.7 * lit))
            shade = shade * var * (0.92 + 0.16 * np.clip(H, 0, 1))
            a = np.clip(H * R * sc * 0.6, 0, 1)
            sub[:] = sub * (1 - a[..., None]) + (col[None, None, :] * shade[..., None]) * a[..., None]

        # flying grains
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil, "RGBA")
        fly_f = int(FLY_S * self.fps)
        if len(self.fl):
            lo = np.searchsorted(self.fl[:, 0], f - fly_f, side="right")
            hi = np.searchsorted(self.fl[:, 0], f, side="right")
            for f0, sx, sy, tx, ty, kind, ci in self.fl[lo:hi]:
                p = (f - f0) / fly_f
                e = p * p * (3 - 2 * p)
                L_ = math.hypot(tx - sx, ty - sy) + 1e-6
                jit = hash2(f0, ci, kind) - 0.5
                nx, ny = -(ty - sy) / L_, (tx - sx) / L_
                bulge = math.sin(math.pi * p) * (0.18 * L_ + 1.5) * (0.6 + jit)
                x = sx + (tx - sx) * e + nx * bulge + jit * 0.8
                y = sy + (ty - sy) * e + ny * bulge - math.sin(math.pi * p) * 1.2
                ppx, ppy = to_px(x, y)
                rad = max(0.8, 0.45 * sc * (1 + 0.35 * math.sin(math.pi * p)))
                c = self.col[int(ci)]
                c = STREAM * (1 - e) + c * e if kind == 0 else (c * (1 - e) + STREAM * e if kind == 1 else c)
                dr.ellipse([ppx - rad, ppy - rad, ppx + rad, ppy + rad], fill=tuple(int(v) for v in c) + (235,))

        # names under the piles: the 12 biggest living companies, any that are big on screen, and big husks
        fs = max(10, Hp / 54)
        f_name, f_small = font("Inter-Medium.otf", fs), font("Inter-Regular.otf", fs * 0.82)
        order = np.argsort(-G)
        rank = {int(i): r + 1 for r, i in enumerate(order[:10]) if G[i] >= 1}
        top12 = {int(i) for i in order[:12] if G[i] >= 1}
        cands = []
        for i in vis:
            k = self.ids[i]
            dead = year > self.S.end[k]
            e_now = max(float(self.EXT[f, i]), 2.0) * sc
            husk_big = dead and float(self.PK[f, i]) > 0 and self.shape[i].extent(float(self.PK[f, i])) * sc > 18
            if (not dead and G[i] >= 1 and (e_now > 26 or i in top12)) or husk_big or (sc > 20 and G[i] >= 0.5):
                cands.append((-(G[i] if not dead else 1e-3 * self.PK[f, i]), i))
        boxes = []

        def free(bb):
            return all(bb[2] < o[0] or bb[0] > o[2] or bb[3] < o[1] or bb[1] > o[3] for o in boxes)

        for _, i in sorted(cands):
            k = self.ids[i]
            px, py = to_px(*self.pos[i])
            dead = year > self.S.end[k]
            label = self.S.companies[k]["label"]
            if i in rank and rank[i] <= 5:
                label = f"{rank[i]}  {label}"
            reach = self.EXT[f, i] if not dead else self.shape[i].extent(float(self.PK[f, i]))
            y = min(py + max(float(reach), 2.0) * sc + 3, py + self.half[i] * sc - fs * (2.2 if dead else 1.2))
            fnt = f_name if not dead else f_small
            lines = [(label, fnt, (MUTED[0], MUTED[1], MUTED[2], 160) if dead else INK + (235,))]
            if dead:
                lines.append((f"{int(self.S.start[k])}–{int(float(self.S.end[k]))}", f_small, MUTED + (130,)))
            bbs = [dr.textbbox((px, y + j * fs * 1.05), t, font=fn, anchor="ma") for j, (t, fn, _) in enumerate(lines)]
            bb = (min(b[0] for b in bbs) - 3, bbs[0][1] - 1, max(b[2] for b in bbs) + 3, bbs[-1][3] + 1)
            if not free(bb):
                continue
            boxes.append(bb)
            for j, (t, fn, c) in enumerate(lines):
                dr.text((px, y + j * fs * 1.05), t, font=fn, fill=c, anchor="ma")

        # HUD
        m = Hp * 0.045
        dr.text((m, m * 0.8), f"{int(math.floor(year))}", font=font("InterDisplay-SemiBold.otf", Hp / 11), fill=INK)
        dr.text((m, m * 0.8 + Hp / 9.2), f"1 grain = ${story.GRAIN_USD_M:,.0f} million in today's dollars",
                font=font("Inter-Medium.otf", Hp / 40), fill=MUTED)
        dr.text((Wp - m, m * 0.9), "Flat preview: layout, clock and camera only", font=font("Inter-Regular.otf", Hp / 48),
                fill=MUTED + (160,), anchor="ra")
        present = sorted({self.S.companies[self.ids[i]]["industry"] for i in vis
                          if G[i] >= 0.5 or year > self.S.end[self.ids[i]]}, key=list(INDUSTRIES).index)
        ly = Hp - m - len(present) * Hp / 34
        for ind in present:
            lab, hx = INDUSTRIES[ind]
            dr.ellipse([m, ly + 3, m + Hp / 60, ly + 3 + Hp / 60], fill=tuple(int(v) for v in hexrgb(hx)))
            dr.text((m + Hp / 40, ly), lab, font=font("Inter-Regular.otf", Hp / 46), fill=MUTED)
            ly += Hp / 34
        if 0.4 < t_film < 6.0:
            a = int(255 * min(1, (t_film - 0.4) / 0.8, (5.8 - t_film) / 0.8))
            dr.text((Wp / 2, Hp * 0.86), TITLE, font=font("InterDisplay-SemiBold.otf", Hp / 18), fill=INK + (a,), anchor="mm")
        return pil


FILM, SIZE = None, None


def _worker_init(size):
    global SIZE
    SIZE = size


def _render(f):
    return FILM.frame(f, SIZE).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "sheet", "video", "info"])
    ap.add_argument("years", nargs="*", type=float)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    os.makedirs(OUT, exist_ok=True)
    film = Film(a.fps)
    if a.mode == "info":
        L = film.L
        print(f"{len(film.ids)} companies, {film.nf} frames, {film.C.duration:.1f} s; grid {L.M} x {L.rows} large plots "
              f"of {L.c_top:.0f} cells ({L.width:.0f} x {L.height:.0f}), {len(L.vacant)} vacant plots")
        for f in range(0, film.nf, int(10 * a.fps)):
            print(f"  {f / a.fps:6.1f} s  {film.years[f]:7.1f}  view {film.cam[f][2]:7.1f} cells")
        return
    if a.mode == "still":
        for y in a.years:
            f = min(int(round(film.C.seconds(y) * a.fps)), film.nf - 1)
            film.frame(f, size).save(os.path.join(OUT, f"still_{y:.1f}.png"))
            print("wrote", os.path.join(OUT, f"still_{y:.1f}.png"))
        return
    if a.mode == "sheet":
        n, tw = 20, 480
        th = tw * 9 // 16
        sheet = Image.new("RGB", (tw * 4, th * 5), (0, 0, 0))
        for j, f in enumerate(np.linspace(int(3 * a.fps), film.nf - 1, n).astype(int)):
            sheet.paste(film.frame(f, (tw * 2, th * 2)).resize((tw, th), Image.LANCZOS), ((j % 4) * tw, (j // 4) * th))
        sheet.save(os.path.join(OUT, "contact_sheet.png"))
        print("wrote", os.path.join(OUT, "contact_sheet.png"))
        return
    import multiprocessing as mp
    path = os.path.join(OUT, "preview.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium", path],
                          stdin=subprocess.PIPE)
    global FILM
    FILM = film                                   # forked workers inherit it
    with mp.get_context("fork").Pool(a.workers, initializer=_worker_init, initargs=(size,)) as pool:
        for j, buf in enumerate(pool.imap(_render, range(film.nf), chunksize=8)):
            ff.stdin.write(buf)
            if j % (a.fps * 20) == 0:
                print(f"  frame {j}/{film.nf}", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
