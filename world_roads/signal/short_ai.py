"""Vertical Short for Video 2: an AI takes every phone, cable and satellite; bonfires from Bletchley Park
warn every capital they can reach (capitals_tree.py).

    python short_ai.py still 20     -> out/short_ai_20.0.png
    python short_ai.py              -> out/signal_fires_ai_short.mp4 (with music)
"""
import os, sys, json, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw
import style as ST
import globe16 as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "flights"))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
VW, VH, FPS = 1080, 1920, 30
G.W, G.H = VW, VH                                  # the globe renderer and its layers, in portrait
CX, CY = VW / 2, 1010
GL, SPIN_END, FIRE0 = 1.8, 7.6, 10.6               # static, premise, the flight to Bletchley Park
END_HOLD = 12.5
FAMOUS = {"London", "Paris", "Berlin", "Rome", "Moscow", "Cairo", "Beijing", "Tokyo", "New Delhi", "Washington, D.C.",
          "Ottawa", "Mexico City", "Brasília", "Buenos Aires", "Canberra", "Pretoria", "Nairobi", "Seoul", "Jakarta", "Lima"}


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def hm(hours):
    return f"{int(hours)} h {int(round(hours % 1 * 60)):02d} min"


def glitch(t):
    rng = np.random.default_rng(int(t * 30) + 3)
    small = (rng.random((VH // 3, VW // 3)) * 255).astype(np.uint8)
    img = np.repeat(cv2.resize(small, (VW, VH), interpolation=cv2.INTER_NEAREST).astype(np.float32)[..., None] * 0.55, 3, -1)
    for _ in range(9):
        y0 = int(rng.integers(0, VH - 60)); hh = int(rng.integers(8, 90)); sh = int(rng.integers(-90, 90))
        img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], sh, axis=1)
    img[..., 0] = np.roll(img[..., 0], 6, axis=1); img[..., 2] = np.roll(img[..., 2], -6, axis=1)
    img[::3] *= 0.75
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    jx = int(rng.integers(-6, 7))
    d.rectangle([VW / 2 - 330, VH / 2 - 85, VW / 2 + 330, VH / 2 + 85], fill=(0, 0, 0, 215))
    d.text((VW / 2 + jx, VH / 2), "NO SIGNAL", font=ST.font(100, "Bold"), fill=(240, 240, 240, 255), anchor="mm")
    return np.asarray(im)


class Short:
    def __init__(self):
        self.g = G.Globe16()
        T = json.load(open(os.path.join(DATA, "capitals_tree.json")))
        st = json.load(open(os.path.join(DATA, "capitals_start.json")))
        self.start = (st["lat"], st["lon"])
        F = T["fires"]
        self.lat = np.array([f["lat"] for f in F]); self.lon = np.array([f["lon"] for f in F])
        self.hop = np.array([f["hop"] for f in F]); self.par = np.array([f["parent"] for f in F])
        self.caps, self.unreach = T["capitals"], T["unreachable"]
        self.mpr = T["min_per_relay"]
        self.max_hop = int(self.hop.max())
        k = np.arange(self.max_hop + 2)
        self.tk = FIRE0 + np.where(k < 10, k * 0.45, 4.5 + (k - 10) * 0.17)
        self.t_end = float(self.tk[self.max_hop])
        self.total = self.t_end + END_HOLD
        self.t_fire = self.tk[self.hop]
        self.last = self.caps[-1]
        # camera: the centre of the fires lit in the last ~20 relays, smoothed
        fr_t = np.arange(int(self.total * FPS) + 2) / FPS
        P = G.unit(self.lat, self.lon)
        S0 = G.unit(*self.start)
        tgt = np.zeros((len(fr_t), 3))
        for j, t in enumerate(fr_t):
            kk = np.searchsorted(self.tk, t, side="right") - 1
            if kk < 1:
                tgt[j] = S0; continue
            m = (self.hop <= kk) & (self.hop > kk - 20)
            v = (P[m] * np.exp(-(kk - self.hop[m]) / 8.0)[:, None]).sum(0)
            tgt[j] = v / max(np.linalg.norm(v), 1e-9)
        sig = int(1.2 * FPS)
        ker = np.exp(-np.arange(-3 * sig, 3 * sig + 1) ** 2 / (2 * sig * sig)); ker /= ker.sum()
        pad = np.pad(tgt, ((3 * sig, 3 * sig), (0, 0)), mode="edge")
        sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(3)], 1)
        self.cam_v = sm / np.linalg.norm(sm, axis=1, keepdims=True)

    def clock(self, t):
        if t < FIRE0:
            return 0.0
        k = min(np.searchsorted(self.tk, t, side="right") - 1, self.max_hop)
        if k < self.max_hop:
            k = k + min(1.0, (t - self.tk[k]) / (self.tk[k + 1] - self.tk[k]))
        return k * self.mpr

    def camera(self, t):
        spin = (25.0, -20 + 8 * SPIN_END)
        if t < SPIN_END:
            return 25.0, -20 + 8 * t, 470
        if t < FIRE0:
            u = smooth((t - SPIN_END) / (FIRE0 - SPIN_END - 0.4))
            la, lo = G.slerp_ll(spin, self.start, u)
            return la, lo, math.exp(math.log(470) + (math.log(2400) - math.log(470)) * u)
        v = self.cam_v[min(int(t * FPS), len(self.cam_v) - 1)]
        la, lo = math.degrees(math.asin(v[2])), math.degrees(math.atan2(v[1], v[0]))
        e = 1 - math.exp(-(t - FIRE0) / 3.0)
        rpx = 2400 * (1 - e) + 820 * e
        if t > self.t_end:
            u = smooth((t - self.t_end - 0.5) / 4.0)
            rpx = rpx * (1 - u) + 500 * u
            la = la * (1 - u) + 12 * u
            lo = lo - 50 * u
        return la, lo, rpx

    def frame(self, t):
        if t < GL:
            fr = glitch(t)
            f = 1 - smooth((t - GL + 0.3) / 0.3)
            return (fr * f).astype(np.uint8) if f < 1 else fr
        lat0, lon0, rpx = self.camera(t)
        img = self.g.render(lat0, lon0, rpx, cx=CX, cy=CY)
        X, Y, Z = self.g.project(self.lat, self.lon, lat0, lon0, rpx, cx=CX, cy=CY)
        Ly = G.Layer()
        lit = np.nonzero(self.t_fire <= t)[0]
        sc = float(np.clip(rpx / 1500, 0.45, 1.3))
        for i in lit:
            p = self.par[i]
            if p >= 0 and Z[i] > 0.02 and Z[p] > 0.02:
                Ly.line((X[p], Y[p]), (X[i], Y[i]), 0.45)
        for i in lit:
            if Z[i] > 0.02 and -40 < X[i] < VW + 40 and -40 < Y[i] < VH + 40:
                age = t - self.t_fire[i]
                Ly.fire(X[i], Y[i], 0.7 * (0.75 + 0.15 * math.sin(t * 9 + i)) * (1 + 2.5 * math.exp(-age / 0.2)), sc)
        if FIRE0 - 2.5 < t < FIRE0:
            Xs, Ys, _ = self.g.project(np.array([self.start[0]]), np.array([self.start[1]]), lat0, lon0, rpx, cx=CX, cy=CY)
            Ly.blit(Xs[0], Ys[0], 1.3, 0.5, (0.6, 0.65, 0.8))
        fr = self.hud(Ly.compose(img), t, lat0, lon0, rpx)
        f = smooth((t - GL) / 0.4) * (1 - smooth((t - self.total + 0.6) / 0.6))
        return (fr * f).astype(np.uint8) if f < 1 else fr

    def hud(self, fr, t, lat0, lon0, rpx):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        c_e, e_e, n_e = self.g.basis(lat0, lon0)

        def proj(la, lo):
            p = G.unit(la, lo)
            return CX + (p @ e_e) * rpx, CY - (p @ n_e) * rpx, p @ c_e
        a_caps = ST.fade(t, FIRE0 - 1.0, self.total + 5, 0.8, 0.5)
        for c in self.caps:
            x, y, z = proj(c["lat"], c["lon"])
            if a_caps <= 0 or z < 0.05 or not (0 < x < VW and 0 < y < VH):
                continue
            tc = self.tk[c["hop"]]
            if t < tc:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(220, 228, 240, int(150 * a_caps)), width=2)
                continue
            age = t - tc
            r = 6 + 16 * math.exp(-age / 0.3)
            d.ellipse([x - r, y - r, x + r, y + r], outline=ST.GOLD + (int(255 * a_caps * min(1, 0.45 + math.exp(-age / 0.5))),), width=3)
            if c["capital"] in FAMOUS and age < 1.6:
                a = a_caps * min(1, age / 0.15) * min(1, (1.6 - age) / 0.3)
                f = ST.font(30, "SemiBold")
                tw = d.textlength(c["capital"], font=f)
                d.rounded_rectangle([x + 12, y - 20, x + 30 + tw, y + 20], 9, fill=ST.PANEL + (int(180 * a),))
                d.text((x + 21, y), c["capital"], font=f, fill=ST.INK + (int(255 * a),), anchor="lm")
        a_red = ST.fade(t, self.t_end + 3.5, self.total + 5, 0.6, 0.5)
        for c in self.unreach:
            x, y, z = proj(c["lat"], c["lon"])
            if a_red > 0 and z > 0.05 and 0 < x < VW and 0 < y < VH:
                col = ST.RED + (int(255 * a_red),)
                d.line([(x - 9, y - 9), (x + 9, y + 9)], fill=col, width=4); d.line([(x - 9, y + 9), (x + 9, y - 9)], fill=col, width=4)
        # the HUD: clock and count up top, the latest capitals below the globe
        a_h = ST.fade(t, FIRE0 - 0.8, self.t_end + 3.2, 0.5, 0.5)
        if a_h > 0:
            d.text((VW / 2, 150), "NO PHONES · NO INTERNET · NO RADIO", font=ST.font(30, "SemiBold"), fill=ST.RED + (int(230 * a_h),), anchor="ma")
            cm = self.clock(t)
            d.text((VW / 2, 200), f"T+ {int(cm // 60):02d}:{int(cm % 60):02d}", font=ST.font(140, "Bold"), fill=ST.GOLD + (int(255 * a_h),),
                   anchor="ma", features=["tnum"])
            n_c = sum(1 for c in self.caps if self.tk[c["hop"]] <= t)
            d.text((VW / 2, 380), f"{n_c} of {len(self.caps)} capitals warned", font=ST.font(44, "SemiBold"), fill=ST.INK + (int(255 * a_h),),
                   anchor="ma", features=["tnum"])
            recent = [c for c in self.caps if self.tk[c["hop"]] <= t][-4:]
            for j, c in enumerate(reversed(recent)):
                age = t - self.tk[c["hop"]]
                a = a_h * max(0.3, 1 - j * 0.22) * min(1, age / 0.2)
                d.text((VW / 2, 1600 + j * 56), f"{c['capital']}  ·  {hm(c['hours'])}", font=ST.font(40 if j == 0 else 34, "SemiBold"),
                       fill=(ST.GOLD if j == 0 else ST.INK) + (int(255 * a),), anchor="ma")
        L = self.last
        nu = len(self.unreach)
        caps = [(GL + 0.2, 4.9, [("An AI just took over", 58, ST.INK, "Bold"), ("every phone, cable", 58, ST.INK, "Bold"),
                                 ("and satellite on Earth", 58, ST.INK, "Bold")], {"y": 230}),
                (5.0, SPIN_END + 0.3, [("How fast could", 58, ST.INK, "Bold"), ("BONFIRES", 84, ST.GOLD, "Bold"),
                                       ("warn every country?", 58, ST.INK, "Bold")], {"y": 230}),
                (SPIN_END + 0.4, FIRE0 + 0.6, [("First fire:", 44, ST.INK2, "SemiBold"), ("Bletchley Park", 62, ST.GOLD, "Bold"),
                                               ("see a fire, light your own", 40, ST.INK, "SemiBold")], {"y": 1560}),
                (self.t_end + 0.2, self.t_end + 3.3, [("Last capital:", 44, ST.INK2, "SemiBold"), (L["capital"], 62, ST.GOLD, "Bold"),
                                                      (hm(L["hours"]), 48, ST.INK, "SemiBold")], {"y": 1500}),
                (self.t_end + 3.5, self.t_end + 7.4, [(f"{len(self.caps)} capitals in {hm(L['hours'])}", 52, ST.GOLD, "Bold"),
                                                      (f"{len(self.lat):,} bonfires", 44, ST.INK, "SemiBold"),
                                                      (f"{nu} capitals: too far across the sea", 40, ST.RED, "SemiBold")], {"y": 200}),
                (self.t_end + 7.6, self.total + 1, [("Good luck hacking", 64, ST.GOLD, "Bold"), ("a bonfire.", 64, ST.GOLD, "Bold"),
                                                    ("Full video on the channel", 40, ST.INK2, "SemiBold")], {"y": 200})]
        for c in caps:
            a = ST.fade(t, c[0], c[1])
            if a > 0:
                ST.caption(d, c[2], a, W=VW, H=VH, **c[3])
        return np.asarray(im)


def render(still=None):
    os.makedirs(OUT, exist_ok=True)
    S = Short()
    print(f"short {S.total:.1f} s: {len(S.lat)} fires, last {S.last['capital']} at {S.t_end:.1f} s", flush=True)
    if still is not None:
        for s in still:
            cv2.imwrite(os.path.join(OUT, f"short_ai_{s:.1f}.png"), S.frame(s)[..., ::-1])
        return
    vid = os.path.join(OUT, "_short_ai.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", vid],
                         stdin=subprocess.PIPE)
    n = int(round(S.total * FPS))
    for i in range(n):
        p.stdin.write(np.ascontiguousarray(S.frame(i / FPS)).tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{n}", flush=True)
    p.stdin.close(); p.wait()
    import score as SC
    from assemble import static, AF
    te = S.t_end
    energy = [(0, 0.0), (GL, 0.0), (GL + 0.05, 0.6), (FIRE0 - 0.1, 0.6), (FIRE0, 0.85), (te - 10, 0.95), (te, 1.0),
              (te + 3.4, 0.6), (te + 7.6, 0.8), (S.total, 0.0)]
    impacts = [(GL, 1.1), (5.0, 0.6), (FIRE0, 1.1), (te, 1.3), (te + 7.6, 0.9)]
    risers = [(FIRE0 - 2.5, FIRE0), (te - 3.0, te)]
    sc = SC.Score(S.total, energy, impacts, risers, transpose_at=FIRE0, final_at=te + 7.6, intro_hit=False)
    sc.compose()
    mix = sc.mix()
    mix = mix + static(mix.shape[1], GL, SC.SR)
    wav = os.path.join(OUT, "short_ai_score.wav")
    SC.write(mix, wav)
    final = os.path.join(OUT, "signal_fires_ai_short.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", AF, "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(vid)
    print("wrote", final)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "still":
        render([float(a) for a in sys.argv[2:]])
    else:
        render()
