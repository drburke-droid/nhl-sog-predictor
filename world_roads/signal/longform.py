"""The long (16:9) signal-fire video, section by section. Each section renders to out/long/<name>.mp4;
assemble.py joins them and adds the score.

    python longform.py physics          -> out/long/physics.mp4
    python longform.py physics 12.0     -> a still at t = 12 s (out/long/physics_12.0.png)
"""
import os, sys, json, math, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
import style as ST
import chain as CH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out", "long")
W, H, FPS = 1920, 1080, 30
BG = (5, 8, 15)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Section:
    name, dur = "section", 1.0

    def frame(self, t):
        raise NotImplementedError

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            cv2.imwrite(os.path.join(OUT, f"{self.name}_{still:.1f}.png"), self.frame(still)[..., ::-1])
            return
        path = os.path.join(OUT, f"{self.name}.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", path],
                             stdin=subprocess.PIPE)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            if i % 150 == 0:
                print(f"{self.name}: frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close(); p.wait()
        print("wrote", path, flush=True)


# ====================================================================== the physics explainer
class Physics(Section):
    """Side view of the curved Earth (vertical exaggeration EX; straight lines stay straight)."""
    name, dur = "physics", 52.0
    SS = 2                                           # supersampling

    def __init__(self):
        L = json.load(open(os.path.join(DATA, "loop.json")))
        st, hz = L["stations"], L["verified_heights_m"]
        from render_fires import peak_name, gc_km
        fire = [i for i in range(1, len(st)) if st[i]["kind"] == "fire" and st[i - 1]["kind"] == "fire"]
        lg = max(fire, key=lambda i: gc_km((st[i - 1]["lat"], st[i - 1]["lon"]), (st[i]["lat"], st[i]["lon"])))
        a, b = st[lg - 1], st[lg]
        self.prof_names = (peak_name(a["lat"], a["lon"], default="Fire A"), peak_name(b["lat"], b["lon"], default="Fire B"))
        self.ha, self.hb = hz[lg - 1] / 1000 + CH.PYRE_KM, hz[lg] / 1000 + CH.PYRE_KM
        import edges as ED
        mos, row0, Wm = ED.load_global()
        pa, pb = CH.xyz(np.array([a["lat"]]), np.array([a["lon"]]))[0], CH.xyz(np.array([b["lat"]]), np.array([b["lon"]]))[0]
        th = math.acos(min(1.0, float(pa @ pb)))
        self.D = th * CH.R
        f = np.linspace(0, 1, 1200)
        p = (np.sin((1 - f) * th)[:, None] * pa + np.sin(f * th)[:, None] * pb) / math.sin(th)
        self.ps = f * self.D
        self.pt = np.array([CH.terrain(mos, row0, Wm, *q) for q in p])        # km
        self.pt[0] = self.ha - CH.PYRE_KM; self.pt[-1] = self.hb - CH.PYRE_KM
        line = self.ha + (self.hb - self.ha) * f
        bulge = self.ps * (self.D - self.ps) / (2 * CH.REFF)
        clear = line - (self.pt - bulge)
        inner = (self.ps > 2) & (self.ps < self.D - 2)
        k = np.argmin(np.where(inner, clear, 1e9))
        self.min_clear_m, self.min_at = float(clear[k] * 1000), float(self.ps[k])
        self.bulge_mid = self.D ** 2 / (8 * CH.REFF)
        print(f"profile {self.D:.1f} km, bulge {self.bulge_mid:.2f} km, min clearance {self.min_clear_m:.0f} m at {self.min_at:.0f} km", flush=True)
        from render_fires import Film
        self.gap = Film().gap                                  # the North Atlantic gap, for the two-fire beat
        self.T = dict(title=0.0, grow=3.0, two=17.0, real=32.0, end=self.dur)

    # --- drawing helpers on a supersampled PIL canvas
    def canvas(self):
        im = Image.new("RGB", (W * self.SS, H * self.SS), BG)
        return im, ImageDraw.Draw(im, "RGBA")

    def finish(self, im):
        return np.asarray(im.resize((W, H), Image.LANCZOS))

    def mountain(self, x_km, h_km, half_w_km, xs, ys_fn):
        """Polygon of a smooth mountain standing on the curved surface."""
        u = np.linspace(-1, 1, 60)
        xx = x_km + u * half_w_km
        hh = h_km * (np.cos(u * math.pi / 2) ** 1.6)
        return [(xs(x), ys_fn(x, hv)) for x, hv in zip(xx, hh)]

    def frame(self, t):
        T = self.T
        if t < T["two"]:
            return self.beat_horizon(t)
        if t < T["real"]:
            return self.beat_two(t)
        return self.beat_real(t)

    def setup_curve(self, x_obs, kx, ex, x0=140, y0=720):
        S = self.SS
        ky = kx * ex
        xs = lambda x: (x0 + (x - 0) * kx) * S
        surf = lambda x: -((x - x_obs) ** 2) / (2 * CH.REFF)
        ys_fn = lambda x, h=0.0: (y0 - (surf(x) + h) * ky) * S
        return xs, ys_fn, surf

    def draw_earth(self, d, xs, ys_fn, x_from, x_to):
        S = self.SS
        xx = np.linspace(x_from, x_to, 300)
        pts = [(xs(x), ys_fn(x)) for x in xx]
        poly = pts + [(pts[-1][0], H * S + 50), (pts[0][0], H * S + 50)]
        d.polygon(poly, fill=(20, 32, 52, 255))
        d.line(pts, fill=(70, 110, 160, 255), width=3 * S)

    def fire(self, d, x, y, r=10, a=1.0):
        S = self.SS
        for rr, al in ((r * 3.2, 0.10), (r * 2.0, 0.25), (r * 1.2, 0.6), (r * 0.6, 1.0)):
            col = (255, 150, 60) if rr > r else (255, 235, 190)
            d.ellipse([x - rr * S, y - rr * S, x + rr * S, y + rr * S], fill=col + (int(255 * al * a),))

    def text(self, d, x, y, s, size, col=ST.INK, weight="SemiBold", anchor="la", a=1.0):
        f = ST.font(size * self.SS, weight)
        d.text((x * self.SS, y * self.SS), s, font=f, fill=tuple(col) + (int(255 * a),), anchor=anchor)

    def head(self, d, t, t0, t1, lines):
        a = ST.fade(t, t0, t1, 0.4, 0.4)
        if a > 0:
            self.text(d, W / 2, 70, lines[0], 52, ST.INK, "Bold", "ma", a)
            if len(lines) > 1:
                self.text(d, W / 2, 140, lines[1], 32, ST.INK2, "Regular", "ma", a)

    def beat_horizon(self, t):
        T = self.T
        im, d = self.canvas()
        S = self.SS
        kx, ex = 2.6, 22.0
        x_obs = 70.0
        xs, ys_fn, surf = self.setup_curve(x_obs, kx, ex, y0=620)
        self.draw_earth(d, xs, ys_fn, -60, 700)
        u = smooth((t - T["grow"] - 0.5) / 10.0)
        h = 0.10 * (48.0 ** u) if t > T["grow"] else 0.10                # 100 m -> 4,800 m, exponential
        rch = float(CH.reach(h + CH.PYRE_KM))
        poly = self.mountain(x_obs, h, 35 + 12 * h, xs, ys_fn)
        d.polygon(poly, fill=(42, 56, 78, 255), outline=(90, 120, 160, 255))
        top = (xs(x_obs), ys_fn(x_obs, h + CH.PYRE_KM))
        a_in = ST.fade(t, T["grow"], T["two"], 0.6, 0.5)
        # horizon ray: from the fire, tangent to the surface at x_obs + reach
        xh = x_obs + rch
        hp = (xs(xh), ys_fn(xh))
        ext = 1.0 + 220 / max(rch, 1)
        far = (top[0] + (hp[0] - top[0]) * ext, top[1] + (hp[1] - top[1]) * ext)
        d.line([top, hp], fill=(255, 186, 80, int(230 * a_in)), width=3 * S)
        for k in range(14):                                              # dashed beyond the horizon
            f0, f1 = 1 + (ext - 1) * k / 14, 1 + (ext - 1) * (k + 0.5) / 14
            d.line([(top[0] + (hp[0] - top[0]) * f0, top[1] + (hp[1] - top[1]) * f0),
                    (top[0] + (hp[0] - top[0]) * f1, top[1] + (hp[1] - top[1]) * f1)], fill=(255, 186, 80, int(110 * a_in)), width=2 * S)
        d.ellipse([hp[0] - 7 * S, hp[1] - 7 * S, hp[0] + 7 * S, hp[1] + 7 * S], fill=(255, 186, 80, int(255 * a_in)))
        self.fire(d, top[0], top[1], 9)
        self.text(d, hp[0] / S + 14, hp[1] / S - 52, "horizon", 30, ST.GOLD, "SemiBold", "la", a_in)
        self.text(d, hp[0] / S + 14, hp[1] / S - 18, f"{rch:,.0f} km", 30, ST.INK, "Regular", "la", a_in)
        # counter
        self.text(d, 140, 860, f"Fire at {h * 1000:,.0f} m", 44, ST.INK, "Bold", "la", a_in)
        self.text(d, 140, 918, f"sees {rch:,.0f} km to the horizon", 36, ST.INK2, "Regular", "la", a_in)
        self.head(d, t, 0.0, T["two"] - 0.2, ["How far can a fire be seen?",
                                               "The Earth curves away. From a beach you can see about 5 km."])
        self.text(d, W - 60, H - 46, "Vertical scale exaggerated · includes normal atmospheric refraction", 22, ST.MUTED,
                  "Regular", "ra", ST.fade(t, 0, self.dur))
        return self.finish(im)

    def beat_two(self, t):
        T = self.T
        im, d = self.canvas()
        S = self.SS
        kx, ex = 4.2, 34.0
        g = self.gap                                  # the numbers of the North Atlantic gap (render_fires.gap_numbers)
        h1 = g["h"] / 1000
        r1 = float(CH.reach(h1 + CH.PYRE_KM)); r2 = g["rb"]
        h2 = r2 ** 2 / (2 * CH.REFF) - CH.PYRE_KM
        u = smooth((t - T["two"] - 1.5) / 9.0)
        dist = (g["dist"] - 100) + 100 * u
        x_mid = 0.0                                   # surface tangent half-way between the fires
        xs, ys_fn, surf = self.setup_curve(x_mid, kx, ex, x0=W / 2, y0=640)
        self.draw_earth(d, xs, ys_fn, -260, 260)
        x1, x2 = -dist / 2, dist / 2
        for xm, hm, w in ((x1, h1, 55), (x2, h2, 22)):
            d.polygon(self.mountain(xm, hm, w, xs, ys_fn), fill=(42, 56, 78, 255), outline=(90, 120, 160, 255))
        A = (xs(x1), ys_fn(x1, h1 + CH.PYRE_KM)); B = (xs(x2), ys_fn(x2, h2 + CH.PYRE_KM))
        a_in = ST.fade(t, T["two"], T["real"], 0.6, 0.5)
        for top, xh in ((A, x1 + r1), (B, x2 - r2)):
            hp = (xs(xh), ys_fn(xh))
            d.line([top, hp], fill=(255, 186, 80, int(170 * a_in)), width=2 * S)
            d.ellipse([hp[0] - 6 * S, hp[1] - 6 * S, hp[0] + 6 * S, hp[1] + 6 * S], fill=(255, 186, 80, int(255 * a_in)))
        ok = dist <= r1 + r2
        col = (120, 220, 140) if ok else (255, 110, 90)
        d.line([A, B], fill=col + (int(240 * a_in),), width=3 * S)
        if not ok:
            xx = np.linspace(x1, x2, 400)
            ly = (h1 + CH.PYRE_KM + surf(x1)) + ((h2 + CH.PYRE_KM + surf(x2)) - (h1 + CH.PYRE_KM + surf(x1))) * (xx - x1) / dist
            below = ly < surf(xx)
            if below.any():
                xb = xx[below]
                d.line([(xs(xb[0]), ys_fn(xb[0], ly[below][0] - surf(xb[0]))), (xs(xb[-1]), ys_fn(xb[-1], ly[below][-1] - surf(xb[-1])))],
                       fill=(255, 110, 90, int(255 * a_in)), width=8 * S)
        self.fire(d, A[0], A[1], 9)
        self.fire(d, B[0], B[1], 7, 1.0 if ok else 0.3)
        gap = dist - (r1 + r2)
        self.text(d, 110, 860, f"Distance {dist:,.0f} km", 44, ST.INK, "Bold", "la", a_in)
        self.text(d, 110, 918, f"Horizons together reach {r1:,.0f} + {r2:,.0f} = {r1 + r2:,.0f} km", 36, ST.INK2, "Regular", "la", a_in)
        msg = "They can see each other" if ok else f"{gap:,.0f} km short: the Earth is in the way"
        self.text(d, W - 110, 880, msg, 40, (120, 220, 140) if ok else ST.RED, "Bold", "ra", a_in)
        self.head(d, t, T["two"], T["real"] - 0.2, ["Two fires see each other only if their horizons overlap",
                                                     f"A fire at {h1 * 1000:,.0f} m and one at {h2 * 1000:,.0f} m"])
        self.text(d, W - 60, H - 46, "Vertical scale exaggerated · includes normal atmospheric refraction", 22, ST.MUTED, "Regular", "ra", 1.0)
        return self.finish(im)

    def beat_real(self, t):
        T = self.T
        im, d = self.canvas()
        S = self.SS
        x0, x1 = 110, W - 110
        kx = (x1 - x0) / self.D
        ex = 22.0
        ky = kx * ex
        y0 = 700
        bulge = self.ps * (self.D - self.ps) / (2 * CH.REFF)
        X = (x0 + self.ps * kx) * S
        Y = (y0 - (self.pt - bulge) * ky) * S
        sea = (y0 - (0 - bulge) * ky) * S
        u = smooth((t - T["real"] - 0.3) / 4.0)
        n = max(2, int(len(X) * u))
        pts = list(zip(X[:n], Y[:n]))
        d.polygon(list(zip(X, sea)) + [(X[-1], H * S + 50), (X[0], H * S + 50)], fill=(12, 20, 36, 255))
        d.line(list(zip(X, sea)), fill=(50, 80, 120, 200), width=2 * S)
        if n > 2:
            d.polygon(pts + [(X[n - 1], H * S + 50), (X[0], H * S + 50)], fill=(36, 50, 72, 255))
            d.line(pts, fill=(110, 140, 180, 255), width=2 * S)
        A = (X[0], (y0 - self.ha * ky) * S); B = (X[-1], (y0 - self.hb * ky) * S)
        a_line = smooth((t - T["real"] - 4.5) / 1.5)
        if a_line > 0:
            mid = (A[0] + (B[0] - A[0]) * a_line, A[1] + (B[1] - A[1]) * a_line)
            d.line([A, mid], fill=(255, 186, 80, 240), width=3 * S)
        self.fire(d, A[0], A[1], 9)
        if a_line >= 1:
            self.fire(d, B[0], B[1], 9)
        a_in = ST.fade(t, T["real"], self.dur, 0.6, 0.6)
        self.text(d, A[0] / S, A[1] / S - 64, self.prof_names[0], 32, ST.INK, "SemiBold", "la", a_in)
        self.text(d, B[0] / S, B[1] / S - 64, self.prof_names[1], 32, ST.INK, "SemiBold", "ra", a_in)
        # the bulge: straight chord between the two sea-level points vs the curved sea level
        a_b = smooth((t - T["real"] - 6.5) / 1.0)
        if a_b > 0:
            for k in range(60):
                xa = X[0] + (X[-1] - X[0]) * k / 60; xb_ = X[0] + (X[-1] - X[0]) * (k + 0.5) / 60
                d.line([(xa, y0 * S), (xb_, y0 * S)], fill=(160, 195, 235, int(150 * a_b)), width=2 * S)
            seapts = list(zip(X, sea))
            for k in range(0, len(seapts) - 8, 16):
                d.line(seapts[k:k + 9], fill=(120, 170, 230, int(220 * a_b)), width=2 * S)
            xm = (x0 + self.D / 2 * kx) * S
            ys_ = (y0 + self.bulge_mid * ky) * S
            d.line([(xm, y0 * S), (xm, ys_)], fill=(160, 195, 235, int(230 * a_b)), width=3 * S)
            self.text(d, xm / S, y0 - 330, f"The Earth bulges {self.bulge_mid:.1f} km between them", 36, (170, 205, 240), "SemiBold", "ma", a_b)
            self.text(d, xm / S, y0 - 284, "...and the sightline still clears every ridge", 32, ST.INK2, "Regular", "ma", a_b)
        a_c = smooth((t - T["real"] - 10.0) / 1.0)
        if a_c > 0:
            self.text(d, W / 2, y0 - 220, "Every sightline in this video was checked like this", 34, ST.GOLD, "SemiBold", "ma", a_c)
        self.head(d, t, T["real"], self.dur, [f"The longest sightline on our route: {self.D:.0f} km",
                                               f"{self.prof_names[0]} to {self.prof_names[1]} · real elevation data"])
        self.text(d, W - 60, H - 46, "Vertical scale exaggerated 22× · includes normal atmospheric refraction", 22, ST.MUTED, "Regular", "ra", 1.0)
        out = self.finish(im)
        fade_out = 1 - smooth((t - (self.dur - 0.8)) / 0.8)
        return (out * fade_out).astype(np.uint8) if fade_out < 1 else out


SECTIONS = {"physics": Physics}

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
