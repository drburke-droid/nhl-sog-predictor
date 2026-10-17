"""The vertical Short (1080x1920): how long would a Roomba take to drive every road in America? At light speed.

    python short.py [still times...]   -> out/roomba_short_video.mp4 (no audio; assemble_short adds the score)
"""
import os, sys, math, time, subprocess
import numpy as np
from PIL import Image, ImageDraw
import video as V
import maps
import cover as CV
from video import ST, text, smooth, fmt_light, fmt_hm, GOLD, INK, INK2, MUTED

SW, SH, FPS = 1080, 1920, 30
SCALE = 0.215
OX, OY = 970, 568                    # the USA map centred on the 1920x1080 working canvas
CROP_X, MAP_Y = 420, 400             # working canvas x 420..1500 -> short at y 400..1480
HOOK, RUN = 2.5, 27.5                # the run: HOOK .. HOOK+RUN
LAST0 = HOOK + RUN                   # the zoom and the slow motion (Last's own clock)
END = LAST0 + 19.0                   # the answer card
DUR = END + 6.0


class Short:
    def __init__(self):
        self.run = V.USARun("run")
        self.run.RUN = RUN
        self.last = V.Last(scale0=SCALE, ox=OX, oy=OY, cx=960, fit=(980, 680))
        self.U = None
        self.lastD = -1

    def usa(self):
        if self.U is None:
            self.U = maps.USA(seed=V.USA_SEED, scale=SCALE, ox=OX, oy=OY)
            self.run.U = self.U
            self.run.Dend = V.usa_tail(self.run.T)["Ds"]
        return self.U

    # the map shows between the title and the HUD, feathered at both ends
    _y = np.arange(MAP_Y, MAP_Y + 1080)
    BAND = np.clip(np.minimum((_y - 500) / 80, (1380 - _y) / 80), 0, 1)[:, None, None].astype(np.float32)

    def canvas(self, fr=None):
        c = np.zeros((SH, SW, 3), np.uint8); c[:] = (5, 8, 15)
        if fr is not None:
            bg = np.array([5, 8, 15], np.float32)
            c[MAP_Y:MAP_Y + 1080] = (bg + (fr[:, CROP_X:CROP_X + SW].astype(np.float32) - bg) * self.BAND).astype(np.uint8)
        return c

    def frame(self, t):
        if t < LAST0:
            U = self.usa()
            D = 0.0 if t < HOOK else self.run.D(t - HOOK)
            if D < self.lastD:
                U.map.reset(); U.flash[:] = 0
            self.lastD = D
            fr = U.frame(D)
            if t < HOOK:
                fr = (maps.BG + (fr.astype(np.float32) - maps.BG) * smooth(t / 1.0)).astype(np.uint8)
            left = U.blocks_left(D)
            head = None
        elif t < END:
            if self.U is not None:                              # free the country map before the zoom loads
                self.U = None; self.run.U = None
            fr, f, D, dark, head, s, lc = self.last.map_frame(t - LAST0)
            left = dark
        else:
            fr, D, left, head = None, self.last.T, 0, None
        im = Image.fromarray(self.canvas(fr))
        if head is not None:                                    # the Roomba, in Short coordinates
            maps.draw_roomba(im, head[0] - CROP_X, head[1] + MAP_Y, head[2], diam=34)
        d = ImageDraw.Draw(im, "RGBA")
        a_end = smooth((t - END) / 0.6)
        if t < END:
            self.hud(d, t, D, left)
            self.top(d, t)
        else:
            self.answer(d, t - END)
        return np.asarray(im)

    def hud(self, d, t, D, left):
        y = 1400
        text(d, (SW / 2, y), "LIGHT TIME", 30, MUTED, 1.0, "SemiBold", "ma")
        sec = D / CV.C_KMS
        if t >= LAST0 + self.last.S0 - 0.5:
            h_, rem = divmod(sec, 3600); m_, s_ = divmod(rem, 60)
            s = f"{int(h_)} h {int(m_):02d} min {s_:06.3f} s"
        else:
            s = fmt_light(sec) if D > 0 else "0 s"
        text(d, (SW / 2, y + 40), s, 76 if len(s) < 16 else 62, GOLD, 1.0, "Bold", "ma")
        text(d, (SW / 2, y + 160), "BLOCKS NOT YET DRIVEN", 30, MUTED, 1.0, "SemiBold", "ma")
        text(d, (SW / 2, y + 200), f"{left:,}", 76, INK if left > 1 else GOLD, 1.0, "Bold", "ma")

    def top(self, d, t):
        def block(lines, a, y0=150):
            y = y0
            for s, size, col, wt in lines:
                text(d, (SW / 2, y), s, size, col, a, wt, "ma")
                y += int(size * 1.3)
        a = ST.fade(t, 0.0, LAST0 - 0.2, 0.01, 0.4)
        block([("How long would a Roomba", 56, INK, "Bold"), ("take to drive every road", 56, INK, "Bold"),
               ("in America?", 56, INK, "Bold"), ("(at the speed of light)", 44, GOLD, "SemiBold")], a, 260)
        L = self.last
        tl = V.usa_tail(L.T)
        n = V.late_block_count(L.T)
        block([(f"{n:,} blocks left," if n > 1 else "One block left,", 56, INK, "Bold"),
               (V.LAST_NOTE.get(V.USA_SEED, "all in one corner of the map") if n > 1 else "", 44, INK2, "SemiBold")],
              ST.fade(t, LAST0 + 0.2, LAST0 + L.S0, 0.4, 0.4), 300)
        sl = (L.T - tl["Ds"]) / CV.C_KMS / (L.S1 - L.S0)
        block([("Slow motion", 56, INK, "Bold"), (f"{1 / sl:,.0f}× slower than light", 44, INK2, "SemiBold")],
              ST.fade(t, LAST0 + L.S0, LAST0 + L.ARRIVE, 0.4, 0.3), 300)
        road, place = V.LAST_PLACE[V.USA_SEED]
        block([("The last block", 60, GOLD, "Bold"), (road, 44, INK, "SemiBold"), (place, 36, INK2, "Regular")],
              ST.fade(t, LAST0 + L.ARRIVE, END, 0.3, 0.4), 280)

    def answer(self, d, u):
        L = self.last
        meta, w, log, secs = V.usa_data()
        rows = [("Every road in America", 52, INK, "SemiBold"), (fmt_hm(L.T / CV.C_KMS), 120, GOLD, "Bold"),
                ("at the speed of light", 44, INK2, "SemiBold"),
                (f"typical of {len(secs)} runs: {fmt_hm(float(np.median(secs)))}", 36, INK2, "Regular"), ("", 30, INK, "Regular"),
                ("At a real Roomba's top speed:", 44, INK, "SemiBold"),
                (f"{round(float(np.median(secs)) * CV.C_KMS * 1000 / 0.5 / (365.25 * 86400), -3):,.0f} years", 84, INK, "Bold"),
                ("", 30, INK, "Regular"), ("Full video on the channel", 40, GOLD, "SemiBold")]
        y = 420
        for i, (s, size, col, wt) in enumerate(rows):
            text(d, (SW / 2, y), s, size, col, smooth((u - 0.15 * i) / 0.4), wt, "ma")
            y += int(size * 1.35) + 10

    def render(self, stills=None):
        os.makedirs(V.OUT, exist_ok=True)
        if stills:
            for s in stills:
                self.frame(s - 1 / FPS)                         # as in the running video: drop the jump's glow
                if self.U is not None and hasattr(self.U, "flash"):
                    self.U.flash[:] = 0
                Image.fromarray(self.frame(s)).save(os.path.join(V.OUT, f"short_{s:.1f}.png"))
            return
        path = os.path.join(V.OUT, "roomba_short_video.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SW}x{SH}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                              "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        n = int(round(DUR * FPS))
        t0 = time.time()
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            if i % 150 == 0:
                print(f"short: frame {i}/{n} {time.time() - t0:.0f}s", flush=True)
        p.stdin.close(); p.wait()
        print("wrote", path, flush=True)


if __name__ == "__main__":
    Short().render([float(x) for x in sys.argv[1:]])
