"""Do the to-scale golf balls survive YouTube? Two seconds of the opening card and of each demo card, as the film shows it (moving film
grain on the desk around the card), are encoded as the master, then scaled and re-encoded the ways YouTube
might stream them (two-pass VP9 like YouTube's own, plus a fast one-pass VP9 as a worst case, H.264 and AV1),
decoded at six moments, and each ball measured.

What matters for seeing a ball this small is the light it adds (its contrast times its area; ACCURACY.md
§4), not its exact shape. So each ball is measured as the extra light in a 7×7 window around it (1080p px,
about 4.8′ at reading distance), in linear light, against the median of a ring around the window, and
divided by what was drawn: 1.00 means the encode kept all of the ball's light.

    python codec_check.py     -> table, and out/codec_check.json
"""
import os, sys, subprocess, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import film
import look as L
import physics as P

TMP = os.path.join(L.OUT, "codec")
VARIANTS = [  # name, width, codec args (None: the master itself), two-pass
    ("master 2160p (HEVC crf 24)", 3840, None, False),
    ("2160p VP9 12 Mb/s", 3840, ["-c:v", "libvpx-vp9", "-b:v", "12M", "-deadline", "good", "-cpu-used", "4", "-row-mt", "1"], True),
    ("1080p VP9 2 Mb/s", 1920, ["-c:v", "libvpx-vp9", "-b:v", "2000k", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1"], True),
    ("1080p VP9 1.2 Mb/s", 1920, ["-c:v", "libvpx-vp9", "-b:v", "1200k", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1"], True),
    ("1080p VP9 fast 1.5 Mb/s", 1920, ["-c:v", "libvpx-vp9", "-b:v", "1500k", "-deadline", "realtime", "-cpu-used", "8"], False),
    ("1080p H.264 2.5 Mb/s", 1920, ["-c:v", "libx264", "-b:v", "2500k", "-preset", "medium"], False),
    ("1080p AV1 1.2 Mb/s", 1920, ["-c:v", "libsvtav1", "-b:v", "1200k", "-preset", "6", "-svtav1-params", "verbosity=0"], False),
    ("720p VP9 0.8 Mb/s", 1280, ["-c:v", "libvpx-vp9", "-b:v", "800k", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1"], True),
]
SAMPLES = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9)


def ball_light(img, width, x, y):
    """extra light around a ball at 1080p point (x, y) (screen coordinates), in linear light × 1080p px²"""
    a = L.to_lin(np.asarray(img.convert("RGB"))).astype(np.float64)
    a = a @ np.array([0.2126, 0.7152, 0.0722])
    k = width / 1920
    X, Y = x * k, y * k
    win = (slice(int(round(Y - 3.5 * k)), int(round(Y + 3.5 * k))), slice(int(round(X - 3.5 * k)), int(round(X + 3.5 * k))))
    ring = a[int(Y - 7 * k):int(Y + 7 * k), int(X - 7 * k):int(X + 7 * k)].copy()
    c = ring.shape[0] // 2
    ring[c - int(4 * k):c + int(4 * k), c - int(4 * k):c + int(4 * k)] = np.nan
    bg = np.nanmedian(ring)
    return float((a[win] - bg).sum() / (k * k))


def drawn_light(card, yd):
    sc = card.scene
    return (film.Y(sc["ball"]) - film.Y(sc["bg"])) * float(film.units(P.ball_arcmin(yd))) ** 2 * np.pi / 4


def main():
    os.makedirs(TMP, exist_ok=True)
    res = {}
    cards = [(film.OPEN_CARD, film.OPEN_HOME, "opening card")] + [(c, film.DEMO_HOME[j], c.scene["name"]) for j, c in enumerate(film.DEMO_CARDS)]
    for j, (card, cam, key) in enumerate(cards):
        pts = [(float(cam.X(x) / L.S), float(cam.Y(y) / L.S), yd) for x, y, yd in card.dots]
        master = os.path.join(TMP, f"card{j}_master.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{L.W}x{L.H}", "-r", "30",
                              "-i", "-", "-c:v", "libx265", "-preset", "medium", "-crf", "24", "-x265-params", "log-level=error",
                              "-pix_fmt", "yuv420p", master], stdin=subprocess.PIPE)
        png = os.path.join(TMP, f"card{j}.png")
        for i in range(60):
            fr = film.desk(cam, [card] if card is film.OPEN_CARD else film.DEMO_CARDS, t=i / 30)
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
            if i == 0:
                Image.fromarray(fr).save(png)
        p.stdin.close(); p.wait()
        img = Image.open(png)
        res.setdefault("drawn (PNG)", {})[key] = {str(yd): [round(ball_light(img, 3840, x, y) / drawn_light(card, yd), 3)] for x, y, yd in pts}
        for name, w, args, two in VARIANTS:
            src = master
            if args:
                src = os.path.join(TMP, f"card{j}_{'_'.join(name.split()[:3])}.mkv")
                base = ["ffmpeg", "-y", "-v", "error", "-i", master, "-vf", f"scale={w}:-2:flags=lanczos", "-pix_fmt", "yuv420p"] + args
                if two:
                    subprocess.run(base + ["-pass", "1", "-passlogfile", src, "-f", "null", "-"], check=True)
                    subprocess.run(base + ["-pass", "2", "-passlogfile", src, src], check=True)
                else:
                    subprocess.run(base + [src], check=True)
            vals = {str(yd): [] for _, _, yd in pts}
            for ss in SAMPLES:
                dec = os.path.join(TMP, "dec.png")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-i", src, "-frames:v", "1", dec], check=True)
                img = Image.open(dec)
                for x, y, yd in pts:
                    vals[str(yd)].append(round(ball_light(img, w, x, y) / drawn_light(card, yd), 3))
            res.setdefault(name, {})[key] = vals
            print(f"{key:13s} {name:26s} " + "  ".join(f"{yd}: {np.median(v):.2f} ({min(v):.2f}..{max(v):.2f})"
                                                       for yd, v in vals.items()), flush=True)
    json.dump(res, open(os.path.join(L.OUT, "codec_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
