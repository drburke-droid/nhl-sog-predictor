"""A vertical Short (1080x1920, ~57 s) cut from the finished film, to send viewers to the full video.

    python3 make_short.py FILM.mp4 RELEASE_DIR OUT.mp4

FILM.mp4      the finished film (any size 16:9; a 1280x720 copy works)
RELEASE_DIR   releases/every_grain_of_us_v3_3 (the 4K HUD overlays: the year panel is taken from them, sharp)

Four clips, cut where no narration line is running, joined by 0.4 s dissolves (50 s in all):
  A 2:22.3-2:38.6  sand streaming across the Atlantic, 1780s-1950s ("Settlers transform..." "Modern migration...")
  B 2:41.0-3:01.0  "Today, almost every vessel..." then down among the mountains: Mumbai, Delhi, Dhaka, Bangkok
  C 3:20.0-3:24.2  Mexico City to New York
  D 3:30.0-3:40.7  the rise to the whole world (the open sea after New York left out), "...the same sand.
                   Every person alive descends from those first African grains."
Layout: a 4:5 crop of the film (1080x1350; ~1.9x of a 720p copy), a blurred copy of the shot filling the frame
behind it, the headline on top in the thumbnails' big condensed type, the film's own year panel top-left of the
picture until the flight, and during the final reveal the picture opens out to the full 16:9 frame; a call to the full film at
the end.
"""
import os, subprocess, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FILM, REL, OUT = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "..", "..", ".claude", "skills", "sandglass", "assets", "fonts")
W, H, FPS = 1080, 1920, 30
PIC_Y = 370                                 # top of the picture (4:5, 1080x1350, until the reveal)
PIC_H = 1350
X_FADE = 12                                 # dissolve frames (0.4 s)
# (source start, end, crop centre x, crop centre y, crop height) in 1280x720 source px; the crop is 4:5
CLIPS = [(142.3, 158.6, 500.0, 352.0, 620.0), (161.0, 181.0, 640.0, 360.0, 720.0), (200.0, 204.2, 640.0, 360.0, 720.0),
         (210.0, 220.7, 640.0, 360.0, 720.0)]
PANEL_OFF = (163.5, 164.7)                  # the year panel steps aside as the camera drops into the flight
OPEN = (209.6, 211.8)                       # in clip D: the picture opens out to the full frame as the camera rises

PAPER = (250, 248, 242, 255)
TEAL = (132, 236, 216, 255)


def head(size):
    return ImageFont.truetype(os.path.join(FONTS, "barlow-condensed-latin-800-normal.woff"), size)


def text_layer(lines, y0, gap=6, shadow=0.75):
    """RGBA layer with centred lines [(text, size, colour)], a soft tight shadow, no outline."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
    y = y0
    for s, size, col in lines:
        f = head(size)
        while d.textlength(s, font=f) > W - 90:
            size -= 2; f = head(size)
        d.text((W / 2, y), s, font=f, fill=col, anchor="mt")
        ds.text((W / 2, y + 3), s, font=f, fill=255, anchor="mt")
        y += f.getbbox(s, anchor="mt")[3] + gap
    from PIL import ImageFilter
    sh = sh.filter(ImageFilter.GaussianBlur(9)).point(lambda v: int(v * shadow))
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    base.paste((0, 0, 0, 255), (0, 0), sh)
    base.alpha_composite(lay)
    return np.asarray(base).astype(np.float32) / 255.0


TOP = text_layer([("200,000 YEARS OF", 124, PAPER), ("HUMAN MIGRATION", 124, TEAL)], 120)
CTA = text_layer([("WATCH THE FULL FILM", 92, TEAL), ("on the channel", 54, PAPER)], 1385)


def over(img, lay, a=1.0, x=0, y=0):
    """Alpha-composite float RGBA `lay` onto float RGB `img` at (x, y), times a."""
    h, w = lay.shape[:2]
    reg = img[y:y + h, x:x + w]
    al = lay[..., 3:4] * a
    reg[:] = reg * (1 - al) + lay[..., :3] * al


def reader(path, start, dur, vf, w, h, ch):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path, "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "rgb24" if ch == 3 else "rgba", "-"], stdout=subprocess.PIPE)
    n = w * h * ch
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, ch)
    p.wait()


def hud_panel(t0, n):
    """The film's year panel at film times t0.., from the 4K HUD parts (with alpha), scaled for the Short."""
    f0 = int(round(t0 * FPS))
    out = []
    for name, a, b in (("hud_part1.mov", 0, 2100), ("hud_part2.mov", 2100, 5040), ("hud_part3.mov", 5040, 6660)):
        lo, hi = max(f0, a), min(f0 + n, b)
        if lo >= hi:
            continue
        vf = f"select=between(n\\,{lo - a}\\,{hi - a - 1}),crop=760:420:0:0,scale=494:273:flags=area"
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(REL, name), "-vf", vf, "-vsync", "0",
                            "-f", "rawvideo", "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
        out.append(np.frombuffer(p, np.uint8).reshape(-1, 273, 494, 4))
    return np.concatenate(out)[:n].astype(np.float32) / 255.0


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def compose(fr, t, clip, panel):
    """One Short frame from a 1280x720 film frame at film time t."""
    _, _, cx, cy, ch = clip
    src = fr.astype(np.float32) / 255.0
    # background: the same shot, enlarged to fill, very soft and dark
    small = cv2.resize(src, (64, 36), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    bw = 36 * W / H
    bg = small[:, int(32 - bw / 2):int(32 + bw / 2) + 1]
    img = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC) * 0.42
    # the picture: a 4:5 crop that opens out to the full 16:9 frame during the reveal
    u = ease((t - OPEN[0]) / (OPEN[1] - OPEN[0])) if OPEN[0] - 1 < t else 0.0
    c_h = ch + (720 - ch) * u
    c_w = ch * 0.8 + (1280 - ch * 0.8) * u
    cx, cy = cx + (640 - cx) * u, cy + (360 - cy) * u
    x0 = int(round(min(max(cx - c_w / 2, 0), 1280 - c_w)))
    y0 = int(round(min(max(cy - c_h / 2, 0), 720 - c_h)))
    crop = src[y0:y0 + int(round(c_h)), x0:x0 + int(round(c_w))]
    ph = int(round(W * crop.shape[0] / crop.shape[1]))
    pic = cv2.resize(crop, (W, ph), interpolation=cv2.INTER_LANCZOS4)
    py = PIC_Y + (PIC_H - ph) // 2
    # a soft shadow under the picture, then the picture
    m = np.zeros((H, W), np.float32); m[py:py + ph] = 1
    img *= 1 - 0.5 * cv2.GaussianBlur(m, (0, 0), 18)[..., None]
    img[py:py + ph] = pic
    # the film's year panel, top-left of the picture, until the flight (where it would hide the city names)
    a = 1.0 - ease((t - PANEL_OFF[0]) / (PANEL_OFF[1] - PANEL_OFF[0]))
    if a > 0.005:
        over(img, panel, a, 16, py + 6)
    over(img, TOP, 1.0)
    return img


def clip_frames(k):
    s0, s1 = CLIPS[k][:2]
    n = int(round((s1 - s0) * FPS))
    panel = hud_panel(s0, n)
    for i, fr in enumerate(reader(FILM, s0, s1 - s0, "scale=1280:720", 1280, 720, 3)):
        if i >= n:
            break
        t = s0 + i / FPS
        yield compose(fr, t, CLIPS[k], panel[min(i, len(panel) - 1)])


def audio_graph(total):
    """The clips' sound, trimmed and joined with the same 0.4 s crossfades, faded out at the end."""
    g = "".join(f"[1:a]atrim={c[0]}:{c[1]},asetpts=PTS-STARTPTS[a{i}];" for i, c in enumerate(CLIPS))
    prev = "a0"
    for i in range(1, len(CLIPS)):
        g += f"[{prev}][a{i}]acrossfade=d=0.4[x{i}];"
        prev = f"x{i}"
    return g + f"[{prev}]afade=t=out:st={total / FPS - 0.6:.3f}:d=0.6[a]"


def main():
    total = sum(int(round((c[1] - c[0]) * FPS)) for c in CLIPS) - X_FADE * (len(CLIPS) - 1)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-i", FILM, "-filter_complex",
         audio_graph(total),
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", OUT], stdin=subprocess.PIPE)
    written, tail = 0, []
    for k in range(len(CLIPS)):
        frames = clip_frames(k)
        if tail:                                   # dissolve from the last clip's tail
            for j, prev in enumerate(tail):
                cur = next(frames)
                a = (j + 1) / (len(tail) + 1)
                enc.stdin.write((np.clip(prev * (1 - a) + cur * a, 0, 1) * 255).astype(np.uint8).tobytes())
                written += 1
            tail = []
        buf = []
        for img in frames:
            buf.append(img)
            if len(buf) > X_FADE:
                out = buf.pop(0)
                last = k == len(CLIPS) - 1
                if last and written >= total - 4 * FPS:          # the call to the full film, last 4 s
                    out = out.copy()
                    over(out, CTA, ease((written - (total - 4 * FPS)) / 12))
                enc.stdin.write((np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes())
                written += 1
        if k < len(CLIPS) - 1:
            tail = buf
        else:
            for out in buf:
                out = out.copy()
                over(out, CTA, 1.0)
                enc.stdin.write((np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes())
                written += 1
    enc.stdin.close(); enc.wait()
    print(f"wrote {OUT}: {written} frames ({written / FPS:.1f} s)")


if __name__ == "__main__":
    main()
