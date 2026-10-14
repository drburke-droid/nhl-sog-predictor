"""Vertical Short: every street in 48 cities at the speed of light, one city after another.

    python race.py               -> out/city_race.mp4, out/city_race_board.png
    python race.py still 20      -> out/city_race_still.png (t = 20 s)
Cities are run in order of lap time (fastest first); the grid scrolls as rows finish and ends on
the full board. All laps share one slow-motion factor, so their on-screen lengths compare honestly.
"""
import os, sys, json, subprocess
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cities import CITIES, RADIUS_KM, slug  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
FD = "/usr/share/fonts/opentype/inter"
STREETS = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
           "living_street", "unknown"]
WEIGHT = [(1.0, 3), (1.0, 3), (0.9, 3), (0.85, 2), (0.75, 2), (0.55, 1), (0.5, 1), (0.5, 1), (0.45, 1)]
C_KMS = 299_792.458

VW, VH, FPS = 1080, 1920, 30
COLS, CW, D = 3, 360, 296             # columns, cell width, map diameter (px)
CELL_H = 418
VIEW_TOP, VIEW_BOT = 318, 1716        # scrolling viewport
SS = 2                                # supersample for city layers
INTRO, GAP, SCROLL_S = 1.6, 0.10, 0.45
TO_BOARD, BOARD_HOLD = 1.1, 5.0
TARGET_RACE_S = 50.0

INK = (240, 244, 250)
INK2 = (160, 178, 200)
MUTED = (110, 128, 150)
GOLD_T = (255, 186, 80)
BG = np.array([4, 7, 13], np.float32) / 255
CYAN = np.array([0.14, 0.66, 1.00], np.float32) * 0.55
GOLD = np.array([1.00, 0.56, 0.13], np.float32) * 0.95
HOT = np.array([1.00, 0.93, 0.78], np.float32)


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def sprite(sig, size):
    k = np.arange(size, dtype=np.float32) - size // 2
    return np.exp(-(k[None] ** 2 + k[:, None] ** 2) / (2 * sig * sig))


DOT = (sprite(1.3, 61)[..., None] * np.array([1, .97, .9]) * 2.6 + sprite(3.5, 61)[..., None] *
       np.array([1, .72, .3]) * 1.3 + sprite(11, 61)[..., None] * np.array([1, .52, .12]) * 0.45).astype(np.float32)


def tone(E):
    small = cv2.resize(E, (E.shape[1] // 2, E.shape[0] // 2), interpolation=cv2.INTER_AREA)
    bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.0) * 0.55 + cv2.GaussianBlur(small, (0, 0), 7) * 0.45,
                       (E.shape[1], E.shape[0]))
    return 1 - np.exp(-(E + bloom) * 1.15)


class City:
    def __init__(self, name):
        z = np.load(os.path.join(DATA, f"{slug(name)}.npz"))
        self.stats = json.load(open(os.path.join(DATA, f"{slug(name)}.json")))
        self.name = name
        self.lap_s = self.stats["seconds_at_c"]
        nx, ny, ea, eb, ecls = z["nx"], z["ny"], z["ea"], z["eb"], z["ecls"]
        rn, re_, cum = z["rn"], z["re"], z["cum"]
        sc = D * SS / (2 * RADIUS_KM)
        N = D * SS
        X = (nx + RADIUS_KM) * sc; Y = (RADIUS_KM - ny) * sc
        P = np.rint(np.stack([np.c_[X[ea], Y[ea]], np.c_[X[eb], Y[eb]]], 1) * 8).astype(np.int32)
        I = np.zeros((N, N), np.uint8)
        for c in np.argsort([w for w, _ in WEIGHT], kind="stable"):
            sel = np.nonzero(ecls == c)[0]
            if len(sel):
                cv2.polylines(I, list(P[sel]), False, int(255 * WEIGHT[c][0]), WEIGHT[c][1], cv2.LINE_AA, shift=3)
        T = cum / cum[-1]
        Tprev = np.r_[0.0, T[:-1]]
        tv = np.full(len(ea), np.inf)
        st = np.nonzero(re_ >= 0)[0][::-1]
        tv[re_[st]] = 0.5 * (Tprev[st] + T[st])
        vt = np.full((N, N), np.inf, np.float32)
        fb = np.floor(tv * 400).astype(np.int64)
        order = np.argsort(-fb, kind="stable")
        bounds = np.r_[0, np.nonzero(np.diff(fb[order]))[0] + 1, len(order)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            sel = order[a:b]
            cv2.polylines(vt, list(P[sel]), False, float(tv[sel].min()), 3, cv2.LINE_8, shift=3)
        idx = ndimage.distance_transform_edt(~np.isfinite(vt), return_distances=False, return_indices=True)
        self.vt = vt[idx[0], idx[1]]
        self.I = I.astype(np.float32) / 255
        ring = np.zeros((N, N), np.uint8)
        cv2.circle(ring, (N // 2 * 8, N // 2 * 8), (N // 2 - 2) * 8, 255, 2, cv2.LINE_AA, shift=3)
        self.ring = cv2.resize(ring, (D, D), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        self.T, self.Tprev, self.re = T, Tprev, re_
        self.rx, self.ry = X[rn] / SS, Y[rn] / SS
        self.cyan_img = self.frame(-1.0, 0.0)
        self.gold_img = self.frame(2.0, 0.0)

    def frame(self, u, flash_w, dot=False):
        """u = fraction of the lap done; flash_w = flash length in lap fractions."""
        gold = (self.vt <= u).astype(np.float32)
        dn = lambda a: cv2.resize(a, (D, D), interpolation=cv2.INTER_AREA)
        E = dn(self.I * (1 - gold))[..., None] * CYAN + dn(self.I * gold)[..., None] * GOLD
        if flash_w > 0:
            age = u - self.vt
            fl = np.where((age >= 0) & (age < flash_w), (1 - age / flash_w) ** 2, 0).astype(np.float32) * 1.6
            E += dn(self.I * fl)[..., None] * HOT
        E += self.ring[..., None] * np.array([.10, .16, .25], np.float32)
        if dot and 0 <= u < 1:
            x, y = self.dot_pos(u)
            xi, yi, h = int(round(x)), int(round(y)), 30
            ys0, ys1, xs0, xs1 = max(0, yi - h), min(D, yi + h + 1), max(0, xi - h), min(D, xi + h + 1)
            E[ys0:ys1, xs0:xs1] += DOT[ys0 - (yi - h):ys1 - (yi - h), xs0 - (xi - h):xs1 - (xi - h)]
        return tone(E)

    def dot_pos(self, u):
        T, Tp = self.T, self.Tprev
        i = min(max(int(np.searchsorted(T, u)), 1), len(T) - 1)
        f = 0.0 if T[i] <= Tp[i] or self.re[i] < 0 else (u - Tp[i]) / (T[i] - Tp[i])
        return self.rx[i - 1] + f * (self.rx[i] - self.rx[i - 1]), self.ry[i - 1] + f * (self.ry[i] - self.ry[i - 1])


def fmt_ms(s):
    return f"{s * 1000:.1f} ms"


class Race:
    def __init__(self):
        names = [n for n, _ in CITIES if os.path.exists(os.path.join(DATA, f"{slug(n)}.npz"))]
        laps = {n: json.load(open(os.path.join(DATA, f"{slug(n)}.json")))["seconds_at_c"] for n in names}
        self.names = sorted(names, key=lambda n: laps[n])                  # fastest first
        print("loading", len(self.names), "cities", flush=True)
        self.cities = [City(n) for n in self.names]
        self.lap = np.array([c.lap_s for c in self.cities])
        self.slow = round(TARGET_RACE_S / self.lap.sum())
        self.dur = self.lap * self.slow
        self.t0 = INTRO + np.r_[0, np.cumsum(self.dur + GAP)[:-1]]          # start of each lap (video s)
        self.race_end = self.t0[-1] + self.dur[-1] + GAP
        self.total = self.race_end + TO_BOARD + BOARD_HOLD
        self.rows = int(np.ceil(len(self.cities) / COLS))
        self.max_scroll = max(0, self.rows * CELL_H + 20 - (VIEW_BOT - VIEW_TOP))
        self.f = dict(title=font("Inter-Bold.otf", 58), sub=font("Inter-Medium.otf", 30),
                      name=font("Inter-SemiBold.otf", 30), ms=font("Inter-Bold.otf", 36),
                      ms_dim=font("Inter-Medium.otf", 30), total=font("InterDisplay-Bold.otf", 64),
                      small=font("Inter-Medium.otf", 26), foot=font("Inter-Regular.otf", 21))
        self.stats = dict(slow=self.slow, total_light_s=float(self.lap.sum()),
                          cities=[dict(name=c.name, lap_ms=c.lap_s * 1000, street_km=c.stats["street_km"],
                                       walk_km=c.stats["walk_km"]) for c in self.cities])

    def active(self, t):
        k = int(np.searchsorted(self.t0, t, side="right")) - 1
        return k

    def scroll(self, t):
        """Scroll so the active row is the second visible row; eased at each row change."""
        s = 0.0
        for r in range(2, self.rows):
            k = r * COLS
            if k >= len(self.cities):
                break
            u = np.clip((t - self.t0[k] + SCROLL_S * 0.5) / SCROLL_S, 0, 1)
            s += CELL_H * (u * u * (3 - 2 * u))
        return min(s, self.max_scroll)

    def cell_xy(self, k, scroll):
        r, c = divmod(k, COLS)
        return c * CW + (CW - D) // 2, int(round(VIEW_TOP + 18 + r * CELL_H - scroll))

    def grid_frame(self, t):
        canvas = np.empty((VH, VW, 3), np.float32); canvas[:] = BG
        k_act = self.active(t)
        sc = self.scroll(t)
        texts = []
        for k, cty in enumerate(self.cities):
            x, y = self.cell_xy(k, sc)
            if y + D + 120 < VIEW_TOP - 40 or y > VIEW_BOT + 20:
                continue
            if k < k_act or (k == k_act and t >= self.t0[k] + self.dur[k]):
                img, state = cty.gold_img, "done"
                if t < self.t0[k] + self.dur[k] + 0.35:            # finish pulse
                    img = np.clip(img * (1 + 0.5 * (1 - (t - self.t0[k] - self.dur[k]) / 0.35)), 0, 1)
            elif k == k_act:
                u = (t - self.t0[k]) / self.dur[k]
                img, state = cty.frame(u, 0.12 / self.dur[k], dot=True), "run"
            else:
                img, state = cty.cyan_img * 0.8, "wait"
            y0, y1 = max(y, 0), min(y + D, VH)
            if y1 > y0:
                canvas[y0:y1, x:x + D] = np.maximum(canvas[y0:y1, x:x + D], img[y0 - y:y1 - y])
            texts.append((k, x, y, state))
        # fade cells out under the header and footer
        ramp = np.ones(VH, np.float32)
        ramp[:VIEW_TOP] = 0
        ramp[VIEW_TOP:VIEW_TOP + 60] = np.linspace(0, 1, 60)
        ramp[VIEW_BOT - 60:VIEW_BOT] = np.linspace(1, 0, 60)
        ramp[VIEW_BOT:] = 0
        canvas = BG + (canvas - BG) * ramp[:, None, None]
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        for k, x, y, state in texts:
            cx = x + D // 2
            d.text((cx, y + D + 14), self.cities[k].name, font=self.f["name"],
                   fill=(INK if state != "wait" else INK2) + (255,), anchor="ma")
            if state == "done":
                d.text((cx, y + D + 56), fmt_ms(self.lap[k]), font=self.f["ms"], fill=GOLD_T + (255,), anchor="ma")
            elif state == "run":
                d.text((cx, y + D + 56), fmt_ms((t - self.t0[k]) / self.slow), font=self.f["ms"],
                       fill=(250, 238, 212, 255), anchor="ma")
        ova = np.asarray(ov).astype(np.float32) / 255
        ova[..., 3] *= ramp[:, None]
        canvas = canvas * (1 - ova[..., 3:]) + ova[..., :3] * ova[..., 3:]
        done_s = float(self.lap[:max(k_act, 0)].sum()) + (min((t - self.t0[k_act]) / self.slow, self.lap[k_act]) if k_act >= 0 else 0)
        return canvas, done_s, k_act

    def chrome(self, canvas, t, done_s, k_act, board=0.0):
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        n = len(self.cities)
        d.text((VW // 2, 52), f"Every street in {n} cities", font=self.f["title"], fill=INK + (255,), anchor="ma")
        d.text((VW // 2, 124), "at the speed of light", font=self.f["title"], fill=INK + (255,), anchor="ma")
        d.text((VW // 2, 208), f"All public streets within {RADIUS_KM:.0f} km of downtown", font=self.f["sub"],
               fill=INK2 + (255,), anchor="ma")
        d.text((VW // 2, 252), f"Slowed down {self.slow}×  ·  fastest city first", font=self.f["small"],
               fill=MUTED + (255,), anchor="ma")
        if board < 1:
            a = int(255 * (1 - board))
            d.text((VW // 2, VIEW_BOT + 22), f"{done_s:.3f} s", font=self.f["total"], fill=(250, 238, 212, a), anchor="ma")
            lab = f"total light time  ·  city {min(max(k_act, 0) + 1, n)} of {n}"
            d.text((VW // 2, VIEW_BOT + 104), lab, font=self.f["small"], fill=INK2 + (a,), anchor="ma")
        d.text((VW // 2, VH - 58), "Road data © OpenStreetMap contributors via Overture Maps · laps include backtracking",
               font=self.f["foot"], fill=MUTED + (255,), anchor="ma")
        a = np.asarray(ov).astype(np.float32) / 255
        return canvas * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]

    # ---- final board: every city at once, 6 x 8
    def board(self):
        BC, BW = 6, 176
        bd = 134
        BH = 178
        top = 296
        canvas = np.empty((VH, VW, 3), np.float32); canvas[:] = BG
        ov = Image.new("RGBA", (VW, VH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        fn, fm = font("Inter-SemiBold.otf", 19), font("Inter-Bold.otf", 21)
        x_off = (VW - BC * BW) // 2
        for k, cty in enumerate(self.cities):
            r, c = divmod(k, BC)
            x = x_off + c * BW + (BW - bd) // 2; y = top + r * BH
            img = cv2.resize(cty.gold_img, (bd, bd), interpolation=cv2.INTER_AREA)
            canvas[y:y + bd, x:x + bd] = np.maximum(canvas[y:y + bd, x:x + bd], img)
            d.text((x + bd // 2, y + bd + 6), cty.name, font=fn, fill=INK + (255,), anchor="ma")
            d.text((x + bd // 2, y + bd + 28), f"{self.lap[k] * 1000:.0f} ms", font=fm, fill=GOLD_T + (255,), anchor="ma")
        y_end = top + int(np.ceil(len(self.cities) / BC)) * BH
        d.text((VW // 2, y_end + 18), f"All {len(self.cities)} cities: {self.lap.sum():.2f} seconds",
               font=font("Inter-Bold.otf", 52), fill=GOLD_T + (255,), anchor="ma")
        d.text((VW // 2, y_end + 80), f"Fastest: {self.cities[0].name} {self.lap[0] * 1000:.0f} ms   ·   "
               f"Slowest: {self.cities[-1].name} {self.lap[-1] * 1000:.0f} ms",
               font=font("Inter-Medium.otf", 28), fill=INK2 + (255,), anchor="ma")
        a = np.asarray(ov).astype(np.float32) / 255
        return canvas * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]

    def frame(self, t):
        if t < self.race_end:
            canvas, done_s, k = self.grid_frame(max(t, 0))
            if t < INTRO:
                canvas, done_s, k = self.grid_frame(0.0)
                done_s, k = 0.0, 0
            return self.chrome(canvas, t, done_s, k)
        u = min(1.0, (t - self.race_end) / TO_BOARD)
        u = u * u * (3 - 2 * u)
        g, _, _ = self.grid_frame(self.race_end - 1e-6)
        if not hasattr(self, "_board"):
            self._board = self.board()
        canvas = g * (1 - u) + self._board * u
        return self.chrome(canvas, t, float(self.lap.sum()), len(self.cities) - 1, board=u)


def render_chunk(args):
    i, n_chunks = args
    R = Race()
    n = int(round(R.total * FPS))
    f0, f1 = n * i // n_chunks, n * (i + 1) // n_chunks
    part = os.path.join(OUT, f"_race_part{i}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                          "-pix_fmt", "yuv420p", part], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(to8(R.frame(f / FPS)).tobytes())
        if (f - f0) % 150 == 0:
            print(f"chunk {i}: frame {f - f0}/{f1 - f0}", flush=True)
    p.stdin.close(); p.wait()
    return part


def to8(c):
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def main(still=None, n_chunks=4):
    os.makedirs(OUT, exist_ok=True)
    R = Race()
    json.dump(R.stats, open(os.path.join(DATA, "race_stats.json"), "w"), indent=1)
    print(f"slow {R.slow}x, light {R.lap.sum():.3f} s, video {R.total:.1f} s", flush=True)
    if still is not None:
        Image.fromarray(to8(R.frame(still))).save(os.path.join(OUT, "city_race_still.png"))
        return
    Image.fromarray(to8(R.chrome(R.board(), R.total, float(R.lap.sum()), len(R.cities) - 1, board=1.0))).save(
        os.path.join(OUT, "city_race_board.png"))
    del R
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(n_chunks) as ex:
        parts = list(ex.map(render_chunk, [(i, n_chunks) for i in range(n_chunks)]))
    lst = os.path.join(OUT, "_race_parts.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(p)}'\n" for p in parts))
    path = os.path.join(OUT, "city_race.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", path], check=True)
    for p in parts + [lst]:
        os.remove(p)
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        main(float(sys.argv[2]))
    else:
        main()
