"""The cover for the aviation Short (1080x1920): the title, then the five planes stacked as a timeline, each in a strip
cut from the film around it, with its year and speed.
  python short_cover.py <out.png>"""
import subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw
import comp as C

OUT = "/home/user/lot/out"
VW, VH = 1080, 1920
WHITE, AMBER = (255, 255, 255), (255, 186, 60)
# the plane's frame: (cut, frame, flame side to keep: the jets' afterburners trail to the left)
PLANES = [("wright", 85, 1903, "WRIGHT FLYER", 30, False, (0, 1920)),
          ("history_race_parker", 190, 1943, "SPITFIRE", 370, False, (0, 1920)),
          ("history_slow48", 885, 1947, "BELL X-1", 807, True, (0, 1920)),
          ("history_race_parker", 345, 2016, "F-35", 1228, True, (0, 1920)),
          ("history_race_parker", 392, 1967, "X-15", 4520, True, (300, 1250))]   # the X-15 in the middle
TOP_H, GAP = 430, 10
STRIP_H = (VH - TOP_H - 40 - 4 * GAP) // 5


def frame(cut, n):
    if cut == "wright":                                    # the flyby at real speed, composited again without signs
        import make_short_planes as SP
        return next(SP.clean_frames("wright", n, n + 1))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{OUT}/{cut}.mp4", "-vf", f"select=eq(n\\,{n})", "-vsync", "0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(C.H, C.W, 3)[C.TOP:C.TOP + C.PH].copy()


def plane_box(pic, light=False, xr=(0, 1920)):
    """the plane in the sky (between columns xr): its dark pixels (or, for the pale Wright Flyer, what stands out
    from the sky)"""
    sky = pic[:360].astype(np.float32) / 255
    sky[:, :xr[0]] = sky[:, xr[1]:] = 1.0 if not light else 0.0
    lum = sky.mean(-1)
    if light:
        blur = cv2.GaussianBlur(lum, (0, 0), 25)
        mask = np.abs(lum - blur) > 0.12
    else:
        mask = lum < 0.33
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask[:, :xr[0]] = 0; mask[:, xr[1]:] = 0
    if light:                                              # a biplane is struts and wires: join its pieces
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[i, :4]
    return x, y, w, h


def strip(cut, n, flame, light, xr):
    pic = frame(cut, n)
    x, y, w, h = plane_box(pic, light, xr)
    if flame: x, w = x - int(0.45 * w), int(w * 1.45)       # the afterburner behind it
    cw = int(min(C.W, max(w * 1.25, h * 1.15 * VW / STRIP_H, 760)))     # the whole plane, height too
    ch = int(round(cw * STRIP_H / VW))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - cw / 2, 0, C.W - cw))
    y0 = int(np.clip(cy - ch / 2, 0, C.PH - ch))
    return cv2.resize(pic[y0:y0 + ch, x0:x0 + cw], (VW, STRIP_H), interpolation=cv2.INTER_AREA)


def main(path):
    base = frame("history_race_parker", 5)                 # the empty road, behind the title
    h = int(round(C.PH * VW / C.W))                        # the picture across the width, cut to the title band
    top = cv2.resize(base, (VW, h))[(h - TOP_H) // 2:(h - TOP_H) // 2 + TOP_H]
    canvas = np.zeros((VH, VW, 3), np.uint8)
    canvas[:TOP_H] = (cv2.GaussianBlur(top, (0, 0), 3).astype(np.float32) * 0.45).astype(np.uint8)
    ys = []
    for k, (cut, n, year, name, mph, flame, xr) in enumerate(PLANES):
        y = TOP_H + 20 + k * (STRIP_H + GAP)
        canvas[y:y + STRIP_H] = strip(cut, n, flame, name == "WRIGHT FLYER", xr)
        ys.append(y)
    im = Image.fromarray(canvas).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    sh = dict(stroke_width=5, stroke_fill=(0, 0, 0, 190))
    d.text((VW / 2, 115), "100+ YEARS", font=C.font(124, "Black"), fill=WHITE + (255,), anchor="mm", **sh)
    d.text((VW / 2, 245), "OF AVIATION", font=C.font(124, "Black"), fill=WHITE + (255,), anchor="mm", **sh)
    d.text((VW / 2, 362), "AT REAL SPEED", font=C.font(70, "Black"), fill=AMBER + (255,), anchor="mm", **sh)
    for y, (cut, n, year, name, mph, flame, xr) in zip(ys, PLANES):
        d.rectangle((0, y + STRIP_H - 96, VW, y + STRIP_H), fill=(0, 0, 0, 90))
        d.text((32, y + STRIP_H - 18), str(year), font=C.font(92, "Black"), fill=AMBER + (255,), anchor="ls", **sh)
        d.text((VW - 32, y + STRIP_H - 26), f"{mph:,} MPH", font=C.font(58, "Black"), fill=WHITE + (255,),
               anchor="rs", **sh)
    Image.alpha_composite(im, ov).convert("RGB").save(path)


if __name__ == "__main__":
    main(sys.argv[1])
