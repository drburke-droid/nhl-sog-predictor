"""Flat top-down preview of the whole film (layout, clock, camera, husks and flows) before any Blender work.

    python3 preview.py still 1602.3 1720.5 1911.5 1989.9 2026.8   # PNGs -> out/still_<year>.png
    python3 preview.py sheet                                       # 20 moments -> out/contact_sheet.png
    python3 preview.py video [--fps 24] [--size 960x540] [--workers 4]   # -> out/preview.mp4

What it shows:
- each company's sand as a cone on its own island (pedestal), coloured by industry; when the camera is
  close, the cone is drawn as individual grains;
- a faint ring on the table at the widest the pile has ever been (its tide mark), so a husk shows how
  big it once was; a company that has died keeps a dark pedestal and a dimmed name;
- canals between the islands with a stream of pale, uncoloured sand (the market); it is thin while there
  is little money on the table and widens as the total grows;
- grains lifting from the canal into a growing company and draining back from a shrinking one (pale, they
  take the company's colour as they land); grains flying straight from one company to another only for a
  real deal (a merger, a purchase, a break-up), in the giving company's colour;
- the camera: it frames every company that has mattered so far, so it starts close on the first grains in
  Amsterdam and only ever pulls back, keeping the husks in view.
"""
import argparse, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import layout, story
from clock import Clock, FINALE
from story import Story, radius, INDUSTRIES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT_DIR = "/usr/share/fonts/opentype/inter"
TABLE = np.array([12, 13, 16], np.float32)
CHANNEL = np.array([6, 7, 9], np.float32)
STREAM = np.array([205, 190, 150], np.float32)       # pale uncoloured sand: money not in any company
INK, MUTED = (232, 238, 248), (147, 161, 184)
PED_R = 1.6           # pedestal radius, cells
FLY_S = 0.9           # seconds a grain is in flight
MIN_VIEW = 24.0       # narrowest view, cells across
SAFE = (0.20, 0.17, 0.97, 0.95)   # where the content may sit (fractions of the frame: x0, y0, x1, y1)
LIGHT = np.array([-0.45, -0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)
TITLE = "424 years of the world's biggest companies"


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
        comps = [dict(id=k, district=S.companies[k]["district"], first=S.start[k]) for k in self.ids]
        self.L = layout.build(comps, {k: float(radius(S.peak_grains[k])) for k in self.ids})
        self.nf = int(math.ceil(C.duration * fps))
        self.years = np.array([min(C.year(f / fps), S.now) for f in range(self.nf)])
        self.G = np.array([[S.grains_one(k, y) for k in self.ids] for y in self.years], np.float32)
        self.col = np.array([hexrgb(INDUSTRIES[S.companies[k]["industry"]][1]) for k in self.ids])
        self.pos = np.array([self.L.lots[k][:2] for k in self.ids])
        self.lot_r = np.array([self.L.lots[k][2] for k in self.ids])
        self._camera()
        self._flows()

    # ---- camera: frame everything that has mattered so far; it only ever pulls back ----
    def _camera(self):
        G = self.G
        gmax = G.max(axis=1, keepdims=True)
        relevant = (G >= np.maximum(0.5, 0.03 * gmax))
        seen = np.maximum.accumulate(relevant, axis=0)
        rmax = np.maximum.accumulate(radius(G), axis=0)
        reach = np.maximum(rmax, PED_R * 2) * 1.35 + 2.0
        x0 = np.where(seen, self.pos[:, 0] - reach, np.inf).min(axis=1)
        x1 = np.where(seen, self.pos[:, 0] + reach, -np.inf).max(axis=1)
        y0 = np.where(seen, self.pos[:, 1] - reach, np.inf).min(axis=1)
        y1 = np.where(seen, self.pos[:, 1] + reach * 1.15, -np.inf).max(axis=1)   # room for names below
        ok = np.isfinite(x0)
        first = np.argmax(ok)
        for a in (x0, x1, y0, y1):
            a[~ok] = a[first]
        # the content box must fit the safe area: x from SAFE[0] to SAFE[2], y from SAFE[1] to SAFE[3]
        # (fractions of the frame), clear of the year at top left and the legend at bottom left
        sw, sh = SAFE[2] - SAFE[0], SAFE[3] - SAFE[1]
        w = np.maximum.accumulate(np.maximum(np.maximum((x1 - x0) / sw, (y1 - y0) / sh * 16 / 9), MIN_VIEW))
        h = w * 9 / 16
        cx = (x0 + x1) / 2 - w * ((SAFE[0] + SAFE[2]) / 2 - 0.5)
        cy = (y0 + y1) / 2 - h * ((SAFE[1] + SAFE[3]) / 2 - 0.5)
        # finale: ease out to the whole table
        n_fin = int(FINALE * self.fps)
        whole_w = max(self.L.width / (SAFE[2] - SAFE[0]), self.L.height / (SAFE[3] - SAFE[1]) * 16 / 9)
        e = np.clip((np.arange(self.nf) - (self.nf - n_fin)) / (n_fin * 0.4), 0, 1)
        e = e * e * (3 - 2 * e)
        lw = np.log(w) * (1 - e) + np.log(whole_w) * e
        wcx = self.L.width / 2 - whole_w * ((SAFE[0] + SAFE[2]) / 2 - 0.5)
        wcy = self.L.height / 2 - whole_w * 9 / 16 * ((SAFE[1] + SAFE[3]) / 2 - 0.5)
        cx = cx * (1 - e) + wcx * e
        cy = cy * (1 - e) + wcy * e
        n = int(3.0 * self.fps)
        self.cam = np.stack([smooth(cx, n), smooth(cy, n), np.exp(smooth(lw, n))], axis=1)

    # ---- flows: grains lifting from the canal, draining to it, and flying between companies in deals ----
    def _flows(self):
        S, L, fps = self.S, self.L, self.fps
        idx = {k: i for i, k in enumerate(self.ids)}
        quay = np.array([(x, y + r + layout.CANAL * 0.5) for x, y, r in (L.lots[k] for k in self.ids)])
        rec = []      # (frame, src_x, src_y, dst_x, dst_y, kind, company index)
        acc = np.zeros(len(self.ids))
        handed = {idx[k] for k in self.ids if S.companies[k]["fate"] in story.HANDOVER}
        for f in range(1, self.nf):
            d = self.G[f] - self.G[f - 1]
            want = np.minimum(5.0, 0.35 * np.abs(d) ** 0.5) * (np.abs(d) > 0.02)
            acc += want
            for i in np.nonzero(acc >= 1)[0]:
                n = int(acc[i]); acc[i] -= n
                if d[i] < 0 and i in handed and self.years[f] >= S.end[self.ids[i]]:
                    continue                      # its sand is going to the buyer (drawn below)
                for _ in range(n):
                    if d[i] > 0:
                        rec.append((f, *quay[i], *self.pos[i], 0, i))
                    else:
                        rec.append((f, *self.pos[i], *quay[i], 1, i))
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
                        if (f0 + j) % max(1, len(dst) // 4) == 0:
                            for _ in range(3):
                                rec.append((f0 + j, *self.pos[s], *self.pos[t], 2, s))
        rec.sort(key=lambda r: r[0])
        self.fl = np.array(rec, np.float64).reshape(-1, 7)

    # ---- one frame ----
    def frame(self, f, size=(960, 540)):
        Wp, Hp = size
        cx, cy, w = self.cam[f]
        sc = Wp / w
        h = Hp / sc
        X0, Y0 = cx - w / 2, cy - h / 2
        img = np.empty((Hp, Wp, 3), np.float32); img[:] = TABLE
        t_film = f / self.fps
        year = self.years[f]
        G = self.G[f]
        total = G.sum()
        peak_total = self.G.sum(axis=1).max()
        stream_frac = float(np.clip((total / peak_total) ** (1 / 3), 0.06, 0.8))

        def to_px(x, y):
            return (x - X0) * sc, (y - Y0) * sc

        # canals, with a stream of pale sand down the middle
        for ax0, ay0, ax1, ay1, cw in self.L.canals:
            px0, py0 = to_px(ax0, ay0); px1, py1 = to_px(ax1, ay1)
            i0, i1 = max(0, int(px0)), min(Wp, int(math.ceil(px1)))
            j0, j1 = max(0, int(py0)), min(Hp, int(math.ceil(py1)))
            if i0 >= i1 or j0 >= j1:
                continue
            img[j0:j1, i0:i1] = CHANNEL
            vertical = (ay1 - ay0) > (ax1 - ax0)
            wx = X0 + (np.arange(i0, i1) + 0.5) / sc
            wy = Y0 + (np.arange(j0, j1) + 0.5) / sc
            WX, WY = np.meshgrid(wx, wy)
            if vertical:
                across, along = WX - (ax0 + ax1) / 2, WY - t_film * 2.5
                half = (ax1 - ax0) / 2
            else:
                across, along = WY - (ay0 + ay1) / 2, WX - t_film * 2.5
                half = (ay1 - ay0) / 2
            band = np.abs(across) < half * stream_frac
            gi, gj = np.floor(along * 1.6), np.floor(across * 1.6)
            dots = band & (hash2(gi, gj, 7.0) < 0.45)
            fu, fv = np.modf(along * 1.6)[0] - 0.5, np.modf(across * 1.6 + 1000)[0] - 0.5
            sph = np.sqrt(np.clip(1 - 4 * (fu ** 2 + fv ** 2), 0, 1))
            k = 0.35 + 0.35 * sph if sc * 0.6 > 2 else 0.5
            sub = img[j0:j1, i0:i1]
            sub[dots] = STREAM * (k[dots, None] if np.ndim(k) else k) * 0.75
            sub[band & ~dots] = sub[band & ~dots] * 0.6 + STREAM * 0.12

        # pedestals and tide marks (drawn with PIL on an overlay later), piles in numpy now
        vis = []
        for i, k in enumerate(self.ids):
            px, py = to_px(*self.pos[i])
            reach = max(radius(self.S.peak_until(k, year)), PED_R) * sc + 4
            if px + reach < 0 or px - reach > Wp or py + reach < 0 or py - reach > Hp:
                continue
            if year < self.S.start[k] - story.RAMP:
                continue
            vis.append(i)
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil, "RGBA")
        for i in vis:
            k = self.ids[i]
            px, py = to_px(*self.pos[i])
            dead = year > self.S.end[k]
            pr = PED_R * sc
            dr.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(16, 18, 22, 255) if dead else (30, 34, 42, 255),
                       outline=(120, 130, 150, 90 if dead else 170), width=max(1, int(sc * 0.12)))
            tide = radius(self.S.peak_until(k, year))
            if tide > radius(G[i]) + 0.4 and tide * sc > 2:
                c = tuple(int(v) for v in self.col[i])
                rr = tide * sc
                dr.ellipse([px - rr, py - rr, px + rr, py + rr], outline=c + (70,), width=max(1, int(sc * 0.18)))
        img = np.asarray(pil, np.float32).copy()

        for i in vis:                      # shadows: a soft dark patch thrown down-right, longer for taller piles
            r = float(radius(G[i]))
            if G[i] < 0.5 or r * sc < 2:
                continue
            px, py = to_px(*self.pos[i])
            off = 0.55 * r * sc
            R = r * sc * 1.15
            i0, i1 = max(0, int(px + off - R)), min(Wp, int(px + off + R) + 1)
            j0, j1 = max(0, int(py + off * 0.8 - R)), min(Hp, int(py + off * 0.8 + R) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            DX, DY = np.meshgrid(np.arange(i0, i1) + 0.5 - px - off, np.arange(j0, j1) + 0.5 - py - off * 0.8)
            dd = np.hypot(DX, DY) / R
            img[j0:j1, i0:i1] *= (1 - 0.55 * np.clip(1.25 - dd, 0, 1))[..., None]

        for i in vis:
            n = G[i]
            if n < 0.5:
                continue
            r = float(radius(n))
            px, py = to_px(*self.pos[i])
            col = self.col[i]
            if r * sc < 1.2:
                ii, jj = int(px), int(py)
                if 0 <= ii < Wp and 0 <= jj < Hp:
                    img[jj, ii] = col
                continue
            R = r * sc + 1
            i0, i1 = max(0, int(px - R)), min(Wp, int(px + R) + 1)
            j0, j1 = max(0, int(py - R)), min(Hp, int(py + R) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            wx = X0 + (np.arange(i0, i1) + 0.5) / sc - self.pos[i][0]
            wy = Y0 + (np.arange(j0, j1) + 0.5) / sc - self.pos[i][1]
            DX, DY = np.meshgrid(wx, wy)
            d = np.hypot(DX, DY) + 1e-6
            a = np.clip((r - d) * sc, 0, 1)
            if not a.any():
                continue
            g = 0.8 * story.SLOPE
            nrm = np.stack([g * DX / d, g * DY / d, np.ones_like(d)], -1)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
            cone = np.clip(nrm @ LIGHT, 0, 1)
            hgt = np.clip((r - d) / max(r, 1e-6), 0, 1)
            # grains on a hex grid anchored to the island (so they don't swim as the camera moves)
            v = DY / 0.866
            row = np.floor(v)
            u = DX + 0.5 * (row % 2)
            gi = np.floor(u)
            fu, fv = u - gi - 0.5, v - row - 0.5
            rr = 4 * (fu ** 2 + fv ** 2)
            sz = np.sqrt(np.clip(1 - rr, 0, 1))
            sph = np.stack([2 * fu, 2 * fv, sz], -1)
            lam = np.clip(sph @ LIGHT, 0, 1) * (rr < 1)
            var = 0.88 + 0.24 * hash2(gi, row, i)
            wgr = float(np.clip((sc - 2.5) / 6.0, 0, 1))
            shade = (1 - wgr) * (0.35 + 0.75 * cone) + wgr * (0.12 + 0.95 * lam * (0.55 + 0.6 * cone))
            shade = shade * var * (0.9 + 0.2 * hgt)
            c = col[None, None, :] * shade[..., None]
            sub = img[j0:j1, i0:i1]
            sub[:] = sub * (1 - a[..., None]) + c * a[..., None]

        # flying grains
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(pil, "RGBA")
        fly_f = int(FLY_S * self.fps)
        if len(self.fl):
            lo = np.searchsorted(self.fl[:, 0], f - fly_f, side="right")
            hi = np.searchsorted(self.fl[:, 0], f, side="right")
            for row_ in self.fl[lo:hi]:
                f0, sx, sy, tx, ty, kind, ci = row_
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
                col = self.col[int(ci)]
                if kind == 0:      # from the canal: pale, taking the colour as it lands
                    c = STREAM * (1 - e) + col * e
                elif kind == 1:    # back to the canal: losing its colour
                    c = col * (1 - e) + STREAM * e
                else:
                    c = col
                dr.ellipse([ppx - rad, ppy - rad, ppx + rad, ppy + rad], fill=tuple(int(v) for v in c) + (235,))

        # names: districts when wide, companies under their piles
        fs = max(10, Hp / 54)
        f_name, f_small = font("Inter-Medium.otf", fs), font("Inter-Regular.otf", fs * 0.82)
        order = np.argsort(-G)
        rank = {int(i): r + 1 for r, i in enumerate(order[:10]) if G[i] >= 1}
        for i in vis:
            k = self.ids[i]
            px, py = to_px(*self.pos[i])
            r_now = max(float(radius(G[i])), PED_R)
            dead = year > self.S.end[k]
            big = G[i] >= 1 and (r_now * sc > 10 or i in rank)
            husk_seen = dead and radius(self.S.peak_grains[k]) * sc > 8
            if not (big or husk_seen or sc > 14):
                continue
            label = self.S.companies[k]["label"]
            if i in rank and rank[i] <= 5:
                label = f"{rank[i]}  {label}"
            y = py + r_now * sc + 3
            c = (MUTED[0], MUTED[1], MUTED[2], 150) if dead else INK + (235,)
            dr.text((px, y), label, font=f_name if not dead else f_small, fill=c, anchor="ma")
            if dead:
                c2 = self.S.companies[k]
                end = int(float(self.S.end[k]))
                dr.text((px, y + fs * 1.05), f"{int(self.S.start[k])}–{end}", font=f_small, fill=MUTED + (120,), anchor="ma")
        for name, (dx0, dy0, dx1, dy1) in self.L.districts.items():
            ppx, ppy = to_px((dx0 + dx1) / 2, dy0)
            if sc < 3.5 and 0 < ppx < Wp and -20 < ppy < Hp:
                dr.text((ppx, ppy - 4), layout.DISTRICTS[name]["label"].upper(), font=font("Inter-SemiBold.otf", fs * 0.9),
                        fill=MUTED + (int(200 * np.clip((3.5 - sc) / 2, 0, 1)),), anchor="md")

        # HUD
        m = Hp * 0.045
        dr.text((m, m * 0.8), f"{int(math.floor(year))}", font=font("InterDisplay-SemiBold.otf", Hp / 11), fill=INK)
        dr.text((m, m * 0.8 + Hp / 9.2), f"1 grain = ${story.GRAIN_USD_M:,.0f} million in today's dollars",
                font=font("Inter-Medium.otf", Hp / 40), fill=MUTED)
        dr.text((Wp - m, m * 0.9), "Flat preview: layout, clock and camera only", font=font("Inter-Regular.otf", Hp / 48),
                fill=MUTED + (160,), anchor="ra")
        present = sorted({self.S.companies[self.ids[i]]["industry"] for i in vis if G[i] >= 0.5 or year > self.S.end[self.ids[i]]},
                         key=list(INDUSTRIES).index)
        ly = Hp - m - len(present) * Hp / 34
        for ind in present:
            lab, hx = INDUSTRIES[ind]
            dr.ellipse([m, ly + 3, m + Hp / 60, ly + 3 + Hp / 60], fill=tuple(int(v) for v in hexrgb(hx)))
            dr.text((m + Hp / 40, ly), lab, font=font("Inter-Regular.otf", Hp / 46), fill=MUTED)
            ly += Hp / 34
        if t_film < 6.0:
            a = int(255 * min(1, (t_film - 0.4) / 0.8, (5.8 - t_film) / 0.8)) if t_film > 0.4 else 0
            if a > 0:
                dr.text((Wp / 2, Hp * 0.16), TITLE, font=font("InterDisplay-SemiBold.otf", Hp / 18), fill=INK + (a,), anchor="mm")
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
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    os.makedirs(OUT, exist_ok=True)
    film = Film(a.fps)
    if a.mode == "info":
        print(f"{len(film.ids)} companies, {film.nf} frames, {film.C.duration:.1f} s; table {film.L.width:.0f} x {film.L.height:.0f} cells")
        for f in range(0, film.nf, int(10 * a.fps)):
            print(f"  {f / a.fps:6.1f} s  {film.years[f]:7.1f}  view {film.cam[f][2]:6.1f} cells")
        return
    if a.mode == "still":
        for y in a.years:
            f = int(round(film.C.seconds(y) * a.fps))
            film.frame(min(f, film.nf - 1), size).save(os.path.join(OUT, f"still_{y:.1f}.png"))
            print("wrote", os.path.join(OUT, f"still_{y:.1f}.png"))
        return
    if a.mode == "sheet":
        n, tw = 20, 480
        th = tw * 9 // 16
        sheet = Image.new("RGB", (tw * 4, th * 5), (0, 0, 0))
        fr = np.linspace(int(3 * a.fps), film.nf - 1, n).astype(int)
        for j, f in enumerate(fr):
            im = film.frame(f, (tw * 2, th * 2)).resize((tw, th), Image.LANCZOS)
            sheet.paste(im, ((j % 4) * tw, (j // 4) * th))
        sheet.save(os.path.join(OUT, "contact_sheet.png"))
        print("wrote", os.path.join(OUT, "contact_sheet.png"))
        return
    import multiprocessing as mp
    path = os.path.join(OUT, "preview.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", path],
                          stdin=subprocess.PIPE)
    global FILM
    FILM = film                                   # forked workers inherit it
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_worker_init, initargs=(size,)) as pool:
        for j, buf in enumerate(pool.imap(_render, range(film.nf), chunksize=8)):
            ff.stdin.write(buf)
            if j % (a.fps * 20) == 0:
                print(f"  frame {j}/{film.nf}", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
