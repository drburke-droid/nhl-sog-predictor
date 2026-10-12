"""Vertical 1080x1920 YouTube Shorts cut from the same renderer.

    python shorts.py            # renders all three into out/shorts/
    python shorts.py world      # or just one: world | americas | finale

Each Short reuses render.Renderer without its 16:9 text overlays, crops a fixed (static) map region,
and adds large phone-readable type. Shorts UI covers roughly the bottom 380 px and right 140 px, so
the clock and captions stay above and left of that.
"""
import os, sys, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import render
from render import DURATION, FPS, EST_TOTAL_KM

VW, VH = 1080, 1920
OUT = os.path.join(render.OUT, "shorts")
FD = "/usr/share/fonts/opentype/inter"
BG = (4, 7, 13)

SHORTS = {
    # name: crop (x0, y0, x1, y1) in 1080p map pixels, film-time range, playback speed, hook lines, end lines
    "world": dict(crop=(130, 150, 1900, 960), t0=0.0, t1=DURATION, speed=280 / 45,
                  hook=["Every road on Earth", "at the speed of light.", "How long does it take?"],
                  end=["4 minutes 40 seconds.", "Full real-time version", "on my channel"]),
    "americas": dict(crop=(110, 170, 735, 575), t0=0.0, t1=50.0, speed=1.0,
                     hook=["Driving every road", "in North America", "at light speed"],
                     end=["The whole world takes 4:40.", "Full video on my channel"]),
    "finale": dict(crop=(1270, 480, 1880, 895), t0=250.0, t1=DURATION, speed=1.0,
                   hook=["The last 30 seconds", "of driving every road", "on Earth at light speed"],
                   end=["Every road. 4 min 40 s.", "Full video on my channel"]),
}


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def text_layer(lines, y, sizes, colors, alpha=1.0):
    im = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for ln, sz, col in zip(lines, sizes, colors):
        f = font("Inter-Bold.otf" if sz >= 60 else "Inter-SemiBold.otf", sz)
        d.text((VW // 2, y), ln, font=f, fill=col + (int(255 * alpha),), anchor="ma")
        y += int(sz * 1.22)
    return im


def composite(base, im):
    a = np.asarray(im).astype(np.float32) / 255
    if a[..., 3].max() == 0:
        return base
    out = base.astype(np.float32) / 255
    out = out * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]
    return (out * 255 + 0.5).astype(np.uint8)


def make(name):
    cfg = SHORTS[name]
    os.makedirs(OUT, exist_ok=True)
    R = render.Renderer()
    R.static = []                                   # drop the 16:9 title/caption overlays
    R._dynamic_text = lambda t, labels: (None, None)
    x0, y0, x1, y1 = cfg["crop"]
    cw, ch = x1 - x0, y1 - y0
    mw = VW - 40
    mh = int(round(mw * ch / cw))
    my = 830 - mh // 2                               # map band centred between hook and clock
    t0, t1, speed = cfg["t0"], cfg["t1"], cfg["speed"]
    n_main = int(round((t1 - t0) / speed * FPS))
    n_end = int(3.0 * FPS)
    ts = speed
    R.reset_to(max(t0 - 1.0, 0) - 1e-9)
    for tt in np.linspace(max(t0 - 1.0, 0), t0, 8)[:-1]:   # warm up the trail/flash state
        R.frame(float(tt), time_scale=max(1.0, ts / 3))

    hook = text_layer(cfg["hook"], 190, [74, 74, 58], [(240, 244, 250), (240, 244, 250), (255, 186, 80)])
    sub = "sped up {:.0f}×".format(speed) if speed > 1.01 else "real time"
    path = os.path.join(OUT, f"short_{name}.mp4")
    p = render.ffmpeg_proc_size(path, VW, VH)
    last = None
    f_clock = font("InterDisplay-Bold.otf", 150)
    f_small = font("Inter-Medium.otf", 36)
    for f in range(n_main + n_end):
        if f < n_main:
            t = min(t0 + (f / max(n_main - 1, 1)) * (t1 - t0), DURATION)
            img = R.frame(t, time_scale=max(1.0, ts / 3))
            last = (img, t)
        else:
            img, t = last
        canvas = np.empty((VH, VW, 3), np.uint8); canvas[:] = BG
        crop = img[y0:y1, x0:x1]
        canvas[my:my + mh, 20:20 + mw] = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_CUBIC)
        im = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        mm, ss = int(t // 60), int(t % 60)
        cy = 1215
        d.text((VW // 2, cy), f"{mm}:{ss:02d}", font=f_clock, fill=(250, 238, 212, 255), anchor="ma")
        d.text((VW // 2, cy + 175), f"≈ {t / DURATION * EST_TOTAL_KM / 1e6:.1f} million km at light speed",
               font=f_small, fill=(170, 186, 206, 255), anchor="ma")
        bw = 640; bx = (VW - bw) // 2; by = cy + 240
        d.rectangle([bx, by, bx + bw, by + 6], fill=(40, 55, 75, 255))
        d.rectangle([bx, by, bx + int(bw * t / DURATION), by + 6], fill=(255, 176, 66, 255))
        d.text((VW // 2, by + 26), sub, font=font("Inter-Regular.otf", 30), fill=(120, 140, 165, 255), anchor="ma")
        canvas = composite(canvas, im)
        # hook text for the first part, end card at the end
        if f < n_main:
            canvas = composite(canvas, hook)
        else:
            u = min(1.0, (f - n_main) / (0.4 * FPS))
            dim = canvas.astype(np.float32) * (1 - 0.55 * u)
            canvas = dim.astype(np.uint8)
            endl = text_layer(cfg["end"], 220, [76] + [56] * (len(cfg["end"]) - 1),
                              [(255, 186, 80)] + [(240, 244, 250)] * (len(cfg["end"]) - 1), alpha=u)
            canvas = composite(canvas, endl)
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
    print("wrote", path, f"{(n_main + n_end) / FPS:.1f}s")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or SHORTS):
        make(nm)
