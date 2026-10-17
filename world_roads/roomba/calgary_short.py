"""Vertical Short (1080x1920): a Roomba with no map drives every street in Calgary, at the speed of light.

    python calgary_short.py [still times...]   -> out/calgary_short_video.mp4, then out/calgary_short.mp4 (with score)
"""
import os, sys, math, time, subprocess
import numpy as np
from PIL import Image, ImageDraw
import video as V
import maps
import cover as CV
from video import ST, text, smooth, fmt_light, GOLD, INK, INK2, MUTED

SW, SH, FPS = 1080, 1920, 30
KM_PX = 33.0
CX, CY, OX, OY = -0.9, -1.75, 540, 950          # map centre (km) on screen
RUN0, RUN1 = 1.5, 34.0                         # the run, video seconds
DUR = 42.0
YEAR = 365.25 * 86400


class CalgaryShort:
    def __init__(self):
        self.C = maps.City("calgary", V.CAL_SEED, KM_PX, CX, CY, OX, OY, w=SW, h=SH)
        self.T = self.C.total_km
        self.med = V.calgary_stats()["median_s_nonback"]
        self.net_km = float(self.C.elen.sum())
        self.lastD = -1

    def D(self, t):
        u = min(max((t - RUN0) / (RUN1 - RUN0), 0.0), 1.0)
        return self.T * u ** 3.2

    def frame(self, t):
        D = self.D(t)
        if D < self.lastD:
            self.__init__()
        self.lastD = D
        fr = self.C.frame(D)
        if t < 1.2:
            fr = (maps.BG + (fr.astype(np.float32) - maps.BG) * smooth(t / 1.2)).astype(np.uint8)
        im = Image.fromarray(fr)
        hx, hy, ang = self.C.head(D)
        if t < RUN1 + 0.3:
            maps.draw_roomba(im, hx, hy, ang, diam=34)
        d = ImageDraw.Draw(im, "RGBA")
        # top
        a0 = ST.fade(t, 0.2, RUN1 - 1.0, 0.4, 0.5)
        for i, (s, size, col) in enumerate((("A Roomba with no map.", 56, INK), ("Every street in Calgary.", 56, INK),
                                            ("At the speed of light.", 56, GOLD))):
            text(d, (SW / 2, 70 + 76 * i), s, size, col, a0 * smooth((t - 0.3 - 0.9 * i) / 0.4), "Bold", "ma")
        if t > RUN1:
            e = int(np.argmax(self.C.first))
            lx, ly = self.C.mx[e], self.C.my[e]
            g = smooth((t - RUN1) / 0.5)
            r = 26 + 60 * (1 - g)
            d.ellipse([lx - r, ly - r, lx + r, ly + r], outline=GOLD + (int(255 * g),), width=5)
            yrs = self.med * CV.C_KMS * 1000 / 0.5 / YEAR
            rows = [(f"Every street: {fmt_light(self.T / CV.C_KMS)}", 64, GOLD, RUN1 + 0.2),
                    (f"(a typical run: the median of 200 is {fmt_light(self.med)})", 36, INK2, RUN1 + 0.6),
                    (f"At a real Roomba's top speed: {yrs:.0f} years", 44, INK, RUN1 + 2.0),
                    ("Every road in America? Full video on the channel", 36, GOLD, RUN1 + 4.0)]
            for i, (s, size, col, t0) in enumerate(rows):
                text(d, (SW / 2, 60 + [0, 84, 150, 220][i]), s, size, col, smooth((t - t0) / 0.4), "Bold" if i != 1 else "Regular", "ma")
        # HUD
        y = 1660
        text(d, (270, y), "LIGHT TIME", 30, MUTED, 1.0, "SemiBold", "ma")
        text(d, (270, y + 40), fmt_light(D / CV.C_KMS) if D > 0 else "0 s", 70, GOLD, 1.0, "Bold", "ma")
        left = self.C.blocks_left(D)
        text(d, (810, y), "BLOCKS NOT YET DRIVEN", 30, MUTED, 1.0, "SemiBold", "ma")
        text(d, (810, y + 40), f"{left:,}", 70, INK if left else GOLD, 1.0, "Bold", "ma")
        return np.asarray(im)

    def render(self, stills=None):
        os.makedirs(V.OUT, exist_ok=True)
        if stills:
            for s in stills:
                if s >= 1 / FPS:
                    self.frame(s - 1 / FPS)
                    self.C.trail[:] = 0
                Image.fromarray(self.frame(s)).save(os.path.join(V.OUT, f"calgary_short_{s:.1f}.png"))
            return
        path = os.path.join(V.OUT, "calgary_short_video.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SW}x{SH}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                              "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        n = int(round(DUR * FPS))
        t0 = time.time()
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            if i % 150 == 0:
                print(f"calgary short: frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close(); p.wait()
        import assemble as A
        A.mux_short(path, os.path.join(V.OUT, "calgary_short.mp4"), go=RUN0, arrive=RUN1, end=RUN1 + 4.0)


if __name__ == "__main__":
    CalgaryShort().render([float(x) for x in sys.argv[1:]])
