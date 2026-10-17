"""Do the sub-pixel offsets survive YouTube? Encode a round's frame the way YouTube might (downscale, then
VP9 / H.264 / AV1 at streaming bitrates), decode it, and measure where each bar's light is centred.

    python codec_check.py     -> table: measured offset of the shifted pair, and the largest false offset
                                 among the three aligned pairs (the noise floor), in 1080p pixels
"""
import os, subprocess, json
import numpy as np
from PIL import Image
import ui
from rounds import Rounds, OFFSETS, ANSWERS, SIGNS, XS, YC, BAR_H, GAP, ROUND, T_INTRO

TMP = os.path.join(ui.OUT, "codec")
VARIANTS = [  # name, width, codec args
    ("master 2160p (x264 crf 14)", 3840, None),
    ("2160p VP9 8 Mb/s", 3840, ["-c:v", "libvpx-vp9", "-b:v", "8M", "-deadline", "realtime", "-cpu-used", "8"]),
    ("1080p VP9 1.5 Mb/s", 1920, ["-c:v", "libvpx-vp9", "-b:v", "1500k", "-deadline", "realtime", "-cpu-used", "8"]),
    ("1080p H.264 2.5 Mb/s", 1920, ["-c:v", "libx264", "-b:v", "2500k", "-preset", "fast"]),
    ("1080p AV1 1 Mb/s", 1920, ["-c:v", "libsvtav1", "-b:v", "1000k", "-preset", "10"]),
    ("720p VP9 0.8 Mb/s", 1280, ["-c:v", "libvpx-vp9", "-b:v", "800k", "-deadline", "realtime", "-cpu-used", "8"]),
    ("480p VP9 0.4 Mb/s", 854, ["-c:v", "libvpx-vp9", "-b:v", "400k", "-deadline", "realtime", "-cpu-used", "8"]),
]


def measure(png, width):
    a = ui.to_lin(np.asarray(Image.open(png).convert("RGB"))).mean(-1)
    k = width / 1920
    bg = np.median(a)
    out = []
    for cx in XS:
        c = []
        for y0, y1 in ((YC - GAP / 2 - BAR_H + 15, YC - GAP / 2 - 15), (YC + GAP / 2 + 15, YC + GAP / 2 + BAR_H - 15)):
            rows = a[int(y0 * k):int(y1 * k), int((cx - 14) * k):int((cx + 14) * k)]
            p = np.clip(rows.mean(0) - bg, 0, None)
            x = np.arange(len(p)) + 0.5 + int((cx - 14) * k)
            c.append((p * x).sum() / p.sum())
        out.append((c[1] - c[0]) / k)
    return out


def main():
    os.makedirs(TMP, exist_ok=True)
    sec = Rounds()
    res = {}
    for r in (3, 4, 5):                                   # 1/2, 1/4, 1/8 px
        t = r * ROUND + T_INTRO + 2.0
        png = os.path.join(TMP, f"r{r}.png")
        Image.fromarray(sec.frame(t)).save(png)
        master = os.path.join(TMP, f"r{r}_master.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", png, "-t", "2", "-r", "30", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", master], check=True)
        for name, w, args in VARIANTS:
            src = master
            if args:
                enc = os.path.join(TMP, f"r{r}_{w}.mkv")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", master, "-vf", f"scale={w}:-2:flags=lanczos",
                                "-pix_fmt", "yuv420p"] + args + [enc], check=True)
                src = enc
            out_png = os.path.join(TMP, f"r{r}_{w}_dec.png")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.5", "-i", src, "-frames:v", "1", out_png], check=True)
            m = measure(out_png, w)
            ans = ANSWERS[r]
            true = SIGNS[r] * OFFSETS[r]
            noise = max(abs(v) for i, v in enumerate(m) if i != ans)
            res.setdefault(name, {})[OFFSETS[r]] = (m[ans], noise)
            print(f"{name:28s} offset {ui.frac(OFFSETS[r]):>3s} px: measured {m[ans]:+.3f} (true {true:+.3f}), "
                  f"aligned pairs read up to {noise:.3f}", flush=True)
    json.dump({k: {str(o): v for o, v in d.items()} for k, d in res.items()}, open(os.path.join(ui.OUT, "codec_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
