"""Vertical Short (1080x1920) of "Every flight on Earth": the day in 7 s, then the light-speed plane
sped up so its 10 min 36 s fits in 24 s. Frames come from flights_render.FlightRenderer without its
on-screen text; captions are re-set at Shorts size.

    python flights_short.py            -> out/every_flight_short.mp4
    python flights_short.py still 30   -> out/short_still.png (t = 30 s)
"""
import os, sys, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flights_render as FR  # noqa: E402

OUT = FR.OUT
FPS = 30
VW, VH = 1080, 1920
DAY_S, CARD_S, LIGHT_S, END_S = 7.0, 2.0, 24.0, 3.6
CROP = (40, 40, 1840, 960)                       # x, y, w, h on the 1920x1080 map
MW = VW
MH = int(round(CROP[3] * VW / CROP[2]))
MAP_Y = 640
BG = np.array([4, 7, 13], np.uint8)
FD = "/usr/share/fonts/opentype/inter"
INK, INK2, MUTED, GOLD = (240, 244, 250), (160, 178, 200), (110, 128, 150), (255, 186, 80)


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


F = dict(title=font("Inter-Bold.otf", 66), sub=font("Inter-Medium.otf", 36), clock=font("InterDisplay-Bold.otf", 150),
         label=font("Inter-SemiBold.otf", 28), stat=font("Inter-Medium.otf", 34), card=font("Inter-Bold.otf", 64),
         card2=font("Inter-Medium.otf", 38), foot=font("Inter-Regular.otf", 22), badge=font("Inter-Bold.otf", 28))


def timeline(T2):
    n1, nc, n2, ne = (int(round(s * FPS)) for s in (DAY_S, CARD_S, LIGHT_S, END_S))
    fr = [("day", 24.0 * f / (n1 - 1)) for f in range(n1)]
    fr += [("card", f / nc) for f in range(nc)]
    fr += [("light", T2 * f / (n2 - 1)) for f in range(n2)]
    fr += [("end", f / ne) for f in range(ne)]
    return fr


def compose(map_rgb, draw_fn):
    x, y, w, h = CROP
    m = cv2.resize(map_rgb[y:y + h, x:x + w], (MW, MH), interpolation=cv2.INTER_AREA)
    canvas = np.empty((VH, VW, 3), np.uint8)
    canvas[:] = np.median(np.r_[m[:3], m[-3:]].reshape(-1, 3), axis=0).astype(np.uint8)   # match the map's ocean
    canvas[MAP_Y:MAP_Y + MH] = m
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im, "RGBA")
    draw_fn(d)
    d.text((VW // 2, VH - 120), "Modeled schedule from real airports and airline routes (OpenFlights).",
           font=F["foot"], fill=MUTED, anchor="ma")
    d.text((VW // 2, VH - 90), "Flight paths drawn as great circles. All figures are estimates.",
           font=F["foot"], fill=MUTED, anchor="ma")
    return np.asarray(im)


def ctext(d, y, s, f, col, a=255):
    d.text((VW // 2, y), s, font=f, fill=tuple(col) + (int(a),), anchor="ma")


def render_range(args):
    f0, f1, path = args
    cv2.setNumThreads(2)
    R = FR.FlightRenderer()
    L = R.L
    fr = timeline(R.T2)
    speed = R.T2 / LIGHT_S
    decay = 0.80
    p = None
    if path:
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                              "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    first = next((i for i in range(f0, f1) if fr[i][0] in ("light", "end")), None)
    if first is not None:
        t0 = fr[first][1] if fr[first][0] == "light" else R.T2
        R.reset_to(max(t0 - 3.0, 0) - 1e-9)
    last_day = None
    out = None
    for i in range(f0, f1):
        kind, v = fr[i]
        if kind == "day":
            E, _ = R.frame_day(v)
            img = R.finish(E, [])
            last_day = E
            in_air = int(np.sum((L["dep"] <= v) & (v < L["dep"] + L["dur"])) +
                         np.sum((L["dep"] - 24 <= v) & (v < L["dep"] - 24 + L["dur"])))
            today = int(np.sum(L["dep"] <= v))
            hh, mm = int(v) % 24, int((v % 1) * 60)

            def draw(d, hh=hh, mm=mm, in_air=in_air, today=today):
                ctext(d, 150, "Every flight on Earth", F["title"], INK)
                ctext(d, 236, "in one day", F["title"], INK)
                ctext(d, 340, f"~{R.n_flights:,} flights · 24 hours in {DAY_S:.0f} seconds", F["sub"], INK2)
                ctext(d, MAP_Y + MH + 40, "TIME (UTC)", F["label"], MUTED)
                ctext(d, MAP_Y + MH + 76, f"{hh:02d}:{mm:02d}", F["clock"], (250, 238, 212))
                ctext(d, MAP_Y + MH + 260, f"{in_air:,} flights in the air right now", F["stat"], INK2)
                ctext(d, MAP_Y + MH + 308, f"{today:,} departed so far today", F["stat"], INK2)
            out = compose(img, draw)
        elif kind == "card":
            if last_day is None:
                last_day, _ = R.frame_day(24.0)
            a = min(1.0, v / 0.2)
            img = R.finish(last_day, [], dim=0.7 * a)

            def draw(d, a=a):
                ctext(d, 190, "Now: one plane", F["card"], GOLD, 255 * a)
                ctext(d, 276, "at the speed of light", F["card"], GOLD, 255 * a)
                ctext(d, MAP_Y + MH + 60, f"flying all {R.n_flights:,} flights,", F["card2"], INK, 255 * a)
                ctext(d, MAP_Y + MH + 112, "one after another.", F["card2"], INK, 255 * a)
                ctext(d, MAP_Y + MH + 210, "How long does it take?", F["card2"], INK2, 255 * a)
            out = compose(img, draw)
        else:
            t = v if kind == "light" else R.T2
            E, _ = R.frame_light(t, decay)
            k = int(np.searchsorted(L["leg_t1"], t, "right"))
            flown = int(np.sum(L["legs"][:k, 0] >= 0))
            km = float(L["leg_t1"][k - 1] * FR.C_KMS) if k else 0.0
            a = min(1.0, v / 0.3) if kind == "end" else 0.0
            img = R.finish(E, [], dim=0.5 * a)
            mm, ss = int(min(t, R.T2) // 60), int(min(t, R.T2) % 60)

            def draw(d, mm=mm, ss=ss, flown=flown, km=km, a=a, kind=kind):
                ctext(d, 150, "One plane at", F["title"], INK)
                ctext(d, 236, "the speed of light", F["title"], INK)
                ctext(d, 340, f"flying every one of today's {R.n_flights:,} flights", F["sub"], INK2)
                if kind == "light":
                    w = 300
                    d.rounded_rectangle([VW // 2 - w // 2, MAP_Y + MH + 26, VW // 2 + w // 2, MAP_Y + MH + 76], 25,
                                        fill=(200, 60, 40, 235))
                    ctext(d, MAP_Y + MH + 36, f"SPED UP {speed:.0f}×", F["badge"], (255, 255, 255))
                else:
                    ctext(d, MAP_Y + MH + 40, "REAL TIME", F["label"], MUTED)
                ctext(d, MAP_Y + MH + 86, f"{mm}:{ss:02d}", F["clock"], (250, 238, 212))
                ctext(d, MAP_Y + MH + 270, f"{flown:,} of {R.n_flights:,} flights flown", F["stat"], INK2)
                ctext(d, MAP_Y + MH + 318, f"{km / 1e6:,.1f} million km", F["stat"], INK2)
                if kind == "end":
                    d.rounded_rectangle([70, MAP_Y + 120, VW - 70, MAP_Y + MH - 120], 26, fill=(6, 10, 18, int(215 * a)))
                    ctext(d, MAP_Y + 170, "10 minutes 36 seconds.", font("Inter-Bold.otf", 70), GOLD, 255 * a)
                    ctext(d, MAP_Y + 268, "Every flight of the day,", F["card2"], INK, 255 * a)
                    ctext(d, MAP_Y + 318, f"{R.total_km / 1e6:.0f} million km, at light speed.", F["card2"], INK, 255 * a)
            out = compose(img, draw)
        if p is None:
            return out
        p.stdin.write(out.tobytes())
        if (i - f0) % 150 == 0:
            print(f"{os.path.basename(path)} {i - f0}/{f1 - f0}", flush=True)
    p.stdin.close(); p.wait()
    return path


def main():
    import json
    T2 = float(np.load(os.path.join(FR.DATA, "flight_layers.npz"))["leg_t1"][-1])
    fr = timeline(T2)
    n = len(fr)
    n_day = sum(1 for k, _ in fr if k in ("day", "card"))
    mid = n_day + (n - n_day) // 2
    jobs = [(0, n_day // 2, os.path.join(OUT, "_s0.mp4")), (n_day // 2, n_day, os.path.join(OUT, "_s1.mp4")),
            (n_day, mid, os.path.join(OUT, "_s2.mp4")), (mid, n, os.path.join(OUT, "_s3.mp4"))]
    from multiprocessing import Pool
    with Pool(len(jobs)) as pool:
        parts = pool.map(render_range, jobs)
    lst = os.path.join(OUT, "_s.txt")
    open(lst, "w").write("".join(f"file '{os.path.basename(q)}'\n" for q in parts))
    final = os.path.join(OUT, "every_flight_short.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", final], check=True)
    for q in parts + [lst]:
        os.remove(q)
    print("wrote", final, f"{n / FPS:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "still":
        T2 = float(np.load(os.path.join(FR.DATA, "flight_layers.npz"))["leg_t1"][-1])
        fr = timeline(T2)
        f = int(round(float(sys.argv[2]) * FPS))
        img = render_range((f, f + 1, None))
        Image.fromarray(img).save(os.path.join(OUT, "short_still.png"))
    else:
        main()
