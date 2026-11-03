"""A vertical Short (1080x1920): the five planes at real speed, one after the other, slowest first. No light, no slow
motion: 100+ years of aviation as a camera at the side of the road would see it.
  python make_short_planes.py -> /home/user/lot/out/planes_real_speed_SHORT.mp4

Each pass is composited again without the film's signs (make_cuts.rt_frames, labels left off), from a little before
the plane comes in to just after its boom and dust; the sound is the real-speed cut's (history_realtime.mp4) over the
same frames. The picture sits in the middle of the frame (the middle 70% of its width), a blurred copy behind it;
above it the year, the name and the speed; below it how long the plane was in the picture."""
import os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, flight as F, make_fly as MF, make_cuts as MC, fly_fx as FX, fly_sound as S

OUT = "/home/user/lot/out"
VW, VH, FPS = 1080, 1920, 25
WHITE, DIM, AMBER = (255, 255, 255), (205, 210, 218), (255, 186, 60)
# name, shown name, year, mph, window (s, on the plane's own clock), its first frame in history_realtime.mp4
PLANES = [("wright", "WRIGHT FLYER", 1903, 30, (-2.4, 2.0), 100),
          ("spitfire", "SPITFIRE", 1943, 370, (-1.4, 1.0), 261),
          ("x1", "BELL X-1", 1947, 807, (-1.4, 1.8), 362),
          ("f35", "F-35 LIGHTNING II", 2016, 1228, (-1.4, 1.8), 475),
          ("x15", "X-15", 1967, 4520, (-1.4, 2.8), 588)]
CROP_W = 1100                                       # the middle of the picture's 1920 px
PIC_W, PIC_H = VW, int(round(C.PH * VW / CROP_W))
PIC_Y = 640


def font(size, weight="Black"):
    return C.font(size, weight)


def clean_frames(name, i0, i1):
    """the plane's real-speed frames i0..i1, without labels"""
    amp = MC.SHAKE_PER_KPA * MC.KPA.get(name, 0)
    fr = F.frames(name)
    for n, t in fr[i0:i1]:
        img = MF.compose(name, n, t, C.plate_loop(n), expo=MC.EXPO_RT)
        f = MC.finish(img, 7000 + n, [], 1.0, FX.shake_t(name, t) if name in MC.KPA else -1, amp)
        yield f[C.TOP:C.TOP + C.PH]


def text(d, xy, s, size, col, weight="Black", a=1.0, anchor="mm", stroke=0):
    if a <= 0: return
    d.text(xy, s, font=font(size, weight), fill=col + (int(255 * a),), anchor=anchor,
           stroke_width=stroke, stroke_fill=(0, 0, 0, int(160 * a)))


def backdrop(pic):
    """the empty road, blurred and dark, filling the frame behind the picture"""
    h = VH
    w = int(pic.shape[1] * h / pic.shape[0])
    bg = cv2.resize(pic, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)[:, (w - VW) // 2:(w - VW) // 2 + VW]
    return (bg.astype(np.float32) * 0.32).astype(np.uint8)


def vertical(pic, top_lines, bottom_lines, bg):
    """the picture in the middle, the backdrop filling the rest, words above and below"""
    x0 = (pic.shape[1] - CROP_W) // 2
    mid = cv2.resize(pic[:, x0:x0 + CROP_W], (PIC_W, PIC_H), interpolation=cv2.INTER_AREA)
    canvas = bg.copy()
    canvas[PIC_Y:PIC_Y + PIC_H] = mid
    im = Image.fromarray(canvas).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle((0, PIC_Y - 2, VW, PIC_Y), fill=(255, 255, 255, 40))
    d.rectangle((0, PIC_Y + PIC_H, VW, PIC_Y + PIC_H + 2), fill=(255, 255, 255, 40))
    for args in top_lines + bottom_lines: text(d, *args)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def fmt_in_picture(s):
    return f"{s:.1f} s" if s >= 1 else f"{s:.2f} s" if s >= 0.1 else f"{s:.3f} s"


def pass_frames(k, plane):
    name, shown, year, mph, (w0, w1), _ = plane
    fr = F.frames(name)
    t0 = fr[0][1]
    i0, i1 = int(round((w0 - t0) * FPS)), int(round((w1 - t0) * FPS))
    dur, t_exit = MC.crossing(name)
    out, bg = [], None
    for j, pic in enumerate(clean_frames(name, i0, i1)):
        if bg is None: bg = backdrop(pic)               # its first frame: the road before the plane comes
        t = fr[i0 + j][1]
        a = min(1.0, (j + 1) / 6)
        top = []
        if k == 0 and j < 40:                        # the opening seconds: what this is
            b = min(1.0, (j + 1) / 5, (40 - j) / 6)
            top += [((VW / 2, 300), "100+ YEARS", 118, WHITE, "Black", b, "mm", 4),
                    ((VW / 2, 425), "OF AVIATION", 118, WHITE, "Black", b, "mm", 4),
                    ((VW / 2, 540), "AT REAL SPEED", 62, AMBER, "Black", b, "mm", 3)]
            b = 0.0
        else:
            b = a if k else min(1.0, (j - 39) / 6)
            top += [((VW / 2, 260), str(year), 170, AMBER, "Black", b, "mm", 4),
                    ((VW / 2, 420), shown, 84 if len(shown) < 12 else 70, WHITE, "Black", b, "mm", 4),
                    ((VW / 2, 530), f"{mph:,} MPH", 64, WHITE, "Bold", b, "mm", 3)]
        if k or j >= 40: top += [((VW / 2, 150), "REAL SPEED  ·  NO SLOW MOTION", 34, DIM, "Bold", 0.9, "mm", 2)]
        bottom = []
        c = min(1.0, max(0.0, (t - t_exit - 0.15) / 0.25))
        if c > 0:
            bottom += [((VW / 2, PIC_Y + PIC_H + 60), "in the picture for", 46, WHITE, "SemiBold", c, "mm", 2),
                       ((VW / 2, PIC_Y + PIC_H + 145), fmt_in_picture(dur), 96, AMBER, "Black", c, "mm", 3)]
        out.append(vertical(pic, top, bottom, bg))
    return out, (i0, i1)


def end_card(last_pic, n=88):
    """the fastest crewed plane ever flown is from 1967"""
    out, bg = [], backdrop(last_pic)
    for j in range(n):
        a = min(1.0, (j + 1) / 8)
        lines = [((VW / 2, 330), "THE FASTEST PLANE", 92, WHITE, "Black", a, "mm", 4),
                 ((VW / 2, 440), "EVER FLOWN", 92, WHITE, "Black", a, "mm", 4),
                 ((VW / 2, 550), "WITH A PILOT", 92, WHITE, "Black", a, "mm", 4),
                 ((VW / 2, PIC_Y + PIC_H + 85), "IS FROM 1967.", 104, AMBER, "Black", min(1.0, max(0.0, (j - 18) / 8)),
                  "mm", 4),
                 ((VW / 2, PIC_Y + PIC_H + 180), "X-15  ·  4,520 MPH", 52, WHITE, "Bold", min(1.0, max(0.0, (j - 30) / 8)),
                  "mm", 3)]
        out.append(vertical(last_pic, lines, [], bg))
    return out


def sound(ranges, n_end):
    """the real-speed cut's sound over the same frames, joined without clicks; the end card quiet"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{OUT}/history_realtime.mp4", "-f", "f32le", "-ac", "2",
                          "-ar", str(S.SR), "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)
    per = S.SR / FPS
    segs = []
    for f0, f1 in ranges:
        s = a[int(f0 * per):int(f1 * per)].copy()
        e = int(0.012 * S.SR)
        s[:e] *= np.linspace(0, 1, e)[:, None]; s[-e:] *= np.linspace(1, 0, e)[:, None]
        segs.append(s)
    tail = segs[-1][-int(0.4 * S.SR):].copy() * np.linspace(1, 0, int(0.4 * S.SR))[:, None]
    segs.append(np.concatenate([tail, np.zeros((int(n_end * per) - len(tail), 2))]))
    return np.concatenate(segs)


if __name__ == "__main__":
    name_out = "planes_real_speed_SHORT"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r",
                          str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                          f"{OUT}/{name_out}_video.mp4"], stdin=subprocess.PIPE)
    ranges, last = [], None
    for k, plane in enumerate(PLANES):
        frames, (i0, i1) = pass_frames(k, plane)
        for f in frames: p.stdin.write(f.tobytes())
        ranges.append((plane[5] + i0, plane[5] + i1))
        last = list(clean_frames(plane[0], i1 - 1, i1))[0]
        print(plane[0], len(frames), "frames", flush=True)
    end = end_card(last)
    for f in end: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    S.write(f"{OUT}/{name_out}.wav", np.clip(sound(ranges, len(end)), -1, 1))      # already through the cut's mic
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{OUT}/{name_out}_video.mp4", "-i", f"{OUT}/{name_out}.wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", f"{OUT}/{name_out}.mp4"], check=True)
    os.remove(f"{OUT}/{name_out}_video.mp4")
    print(name_out, "done", flush=True)
