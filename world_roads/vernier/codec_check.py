"""Do the plates' sub-pixel steps survive YouTube? Each plate's frame is encoded the way YouTube might
(downscaled, then VP9 / H.264 / AV1 at streaming bitrates), decoded, and the line's dark centre measured
on each side of the step.

    python codec_check.py     -> table: the measured step (1080p px) for each plate and encode
"""
import os, sys, subprocess, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L
import film

TMP = os.path.join(L.OUT, "codec")
VARIANTS = [  # name, width, codec args (None: the master itself)
    ("master 2160p (x264 crf 20)", 3840, None),
    ("2160p VP9 8 Mb/s", 3840, ["-c:v", "libvpx-vp9", "-b:v", "8M", "-deadline", "realtime", "-cpu-used", "8"]),
    ("1080p VP9 1.5 Mb/s", 1920, ["-c:v", "libvpx-vp9", "-b:v", "1500k", "-deadline", "realtime", "-cpu-used", "8"]),
    ("1080p H.264 2.5 Mb/s", 1920, ["-c:v", "libx264", "-b:v", "2500k", "-preset", "fast"]),
    ("1080p AV1 1 Mb/s", 1920, ["-c:v", "libsvtav1", "-b:v", "1000k", "-preset", "10"]),
    ("720p VP9 0.8 Mb/s", 1280, ["-c:v", "libvpx-vp9", "-b:v", "800k", "-deadline", "realtime", "-cpu-used", "8"]),
    ("480p VP9 0.4 Mb/s", 854, ["-c:v", "libvpx-vp9", "-b:v", "400k", "-deadline", "realtime", "-cpu-used", "8"]),
]


def measure(png, width, card):
    """the step: (dark centre right of the step) - (dark centre left of it), in 1080p px. Each column's
    darkness is taken against that column's own background, in linear light."""
    a = L.to_lin(np.asarray(Image.open(png).convert("RGB"))).mean(-1)
    k = width / 1920
    yc = 540 + card.th / 2 + card.step / 2                 # the line's middle on screen (1080p units)
    rows = np.arange(int((yc - 14) * k), int((yc + 14) * k))
    far = np.r_[np.arange(int((yc - 30) * k), int((yc - 18) * k)), np.arange(int((yc + 18) * k), int((yc + 30) * k))]
    out = []
    for x0, x1 in ((card.lx0 + 40, card.xm - 40), (card.xm + 40, card.lx1 - 40)):
        cols = slice(int(x0 * k), int(x1 * k))
        bg = np.median(a[far, cols], axis=0)
        dark = np.clip(bg[None, :] - a[rows, cols], 0, None).mean(1)
        y = (rows + 0.5) / k
        out.append((dark * y).sum() / dark.sum())
    return out[1] - out[0]


SAMPLES = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9)            # decoded frames measured (s)


def main():
    """two seconds of each plate as the film shows it (moving film grain on the desk), encoded, decoded at six
    moments; the step-free plate 4 is the control, and its reading is subtracted"""
    os.makedirs(TMP, exist_ok=True)
    res = {}
    for k, card in enumerate(film.CARDS):
        master = os.path.join(TMP, f"plate{k + 1}_master.mp4")
        p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{L.W}x{L.H}", "-r", "30",
                              "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", master],
                             stdin=subprocess.PIPE)
        for i in range(60):
            fr = film.desk(film.HOME[k], ticks=(k, 1.0), t=i / 30)
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
            if i == 0:
                png = os.path.join(TMP, f"plate{k + 1}.png")
                Image.fromarray(fr).save(png)
        p.stdin.close()
        p.wait()
        res.setdefault("drawn (PNG)", {})[str(card.step)] = [round(measure(png, 3840, card), 4)]
        for name, w, args in VARIANTS:
            src = master
            if args:
                src = os.path.join(TMP, f"plate{k + 1}_{name.split()[0]}_{name.split()[1]}.mkv")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", master, "-vf", f"scale={w}:-2:flags=lanczos",
                                "-pix_fmt", "yuv420p"] + args + [src], check=True)
            vals = []
            for ss in SAMPLES:
                dec = os.path.join(TMP, "dec.png")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-i", src, "-frames:v", "1", dec], check=True)
                vals.append(round(measure(dec, w, card), 4))
            res.setdefault(name, {})[str(card.step)] = vals
            print(f"plate {k + 1} ({card.step} px)  {name:28s} " + " ".join(f"{v:+.3f}" for v in vals), flush=True)
    # against the control
    table = {}
    for name, d in res.items():
        ctl = float(np.median(d["0.0"]))
        table[name] = {st: {"median": round(float(np.median(v)) - ctl, 3), "min": round(min(v) - ctl, 3), "max": round(max(v) - ctl, 3)}
                       for st, v in d.items() if st != "0.0"}
        table[name]["control"] = round(ctl, 3)
    json.dump({"raw": res, "vs_control": table}, open(os.path.join(L.OUT, "codec_check.json"), "w"), indent=1)
    for name, d in table.items():
        print(f"{name:28s} " + "  ".join(f"{st}: {v['median']:.3f} ({v['min']:.3f}..{v['max']:.3f})"
                                          for st, v in d.items() if st != "control"))


if __name__ == "__main__":
    main()
