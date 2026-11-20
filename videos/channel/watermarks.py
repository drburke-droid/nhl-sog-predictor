"""Video watermark options (YouTube: 150x150 PNG, shown small in the bottom-right corner of every video; on desktop it
turns into a Subscribe button on hover).

    python watermarks.py <dark 1920x1080 frame> <bright image>
    -> watermarks/wm_*.png (150x150, transparent) and watermarks/watermarks_preview.jpg (each one on a dark film
       frame and a bright one, roughly at the size YouTube shows it)
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_branding as B  # noqa: E402

OUT = os.path.join(HERE, "watermarks")
S = 600                      # drawn at 4x, saved at 150
NAVY = np.array([11, 22, 38], np.float32) / 255
CYAN = np.array([70, 178, 255], np.float32) / 255


def blank():
    return np.zeros((S, S, 4), np.float32)


def over(dst, rgb, a):
    """composite colour rgb with coverage a onto RGBA float dst"""
    a = a[..., None]
    out_a = a + dst[..., 3:] * (1 - a)
    dst[..., :3] = (np.asarray(rgb) * a + dst[..., :3] * dst[..., 3:] * (1 - a)) / np.maximum(out_a, 1e-6)
    dst[..., 3:] = out_a


def place(al, cx=S / 2, cy=S / 2):
    m = np.zeros((S, S), np.float32)
    h, w = al.shape
    y0, x0 = int(cy - h / 2), int(cx - w / 2)
    m[y0:y0 + h, x0:x0 + w] = al
    return m


def shadow(m, sigma=14, k=0.75):
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma) * 1.6, 0, 1) * k


def disc(r=S / 2 - 6):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    return np.clip(r - np.hypot(xx - S / 2 + 0.5, yy - S / 2 + 0.5) + 0.5, 0, 1)


def gold_fill(m):
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    return (B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t), m


def wm_white(M):
    d = blank()
    over(d, (1, 1, 1), place(B.mark_alpha(M, 560)))
    return d


def wm_white_shadow(M):
    d = blank()
    m = place(B.mark_alpha(M, 520))
    over(d, (0, 0, 0), shadow(m))
    over(d, (1, 1, 1), m)
    return d


def wm_gold_shadow(M):
    d = blank()
    m = place(B.mark_alpha(M, 520))
    over(d, (0, 0, 0), shadow(m))
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None]
    rgb = B.GOLD_TOP * (1 - t[..., None]) + B.GOLD_BOT * t[..., None]
    a = m[..., None]
    out_a = a + d[..., 3:] * (1 - a)
    d[..., :3] = (np.broadcast_to(rgb, (S, S, 3)) * a + d[..., :3] * d[..., 3:] * (1 - a)) / np.maximum(out_a, 1e-6)
    d[..., 3:] = out_a
    return d


def wm_badge(M):
    """the gold-on-navy channel picture as a round badge: reads as a button"""
    d = blank()
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rr = np.clip(np.hypot(xx - S / 2, yy - S / 2) / (S / 2), 0, 1)
    navy = B.NAVY_IN * (1 - rr[..., None]) + B.NAVY_OUT * rr[..., None]
    dk = disc()
    d[..., :3] = navy
    d[..., 3] = dk
    ring = np.clip(dk - disc(S / 2 - 22), 0, 1)
    over(d, B.GOLD_BOT * 0.95, ring * 0.9)
    m = place(B.mark_alpha(M, 400), cy=S / 2 + 4)
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    rgb = B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t
    a = m[..., None]
    d[..., :3] = rgb * a + d[..., :3] * (1 - a)
    return d


def wm_split(M):
    d = blank()
    m = place(B.mark_alpha(M, 520))
    over(d, (0, 0, 0), shadow(m))
    x = np.arange(S)[None, :, None]
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    gold = B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t
    rgb = np.where(x < S / 2, np.broadcast_to(CYAN, (S, S, 3)), gold)
    a = m[..., None]
    out_a = a + d[..., 3:] * (1 - a)
    d[..., :3] = (rgb * a + d[..., :3] * d[..., 3:] * (1 - a)) / np.maximum(out_a, 1e-6)
    d[..., 3:] = out_a
    return d


def wm_tv(M):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    gold = (255, 196, 84, 255)
    dr.line([(230, 60), (300, 128)], fill=gold, width=22)
    dr.line([(370, 60), (300, 128)], fill=gold, width=22)
    dr.rounded_rectangle([40, 128, 560, 500], radius=70, fill=(11, 22, 38, 235), outline=gold, width=26)
    dr.rounded_rectangle([150, 500, 200, 550], radius=10, fill=gold)
    dr.rounded_rectangle([400, 500, 450, 550], radius=10, fill=gold)
    d = np.asarray(im).astype(np.float32) / 255
    sh = shadow(d[..., 3], 10, 0.6)
    base = blank()
    over(base, (0, 0, 0), sh)
    a = d[..., 3:]
    out_a = a + base[..., 3:] * (1 - a)
    base[..., :3] = (d[..., :3] * a + base[..., :3] * base[..., 3:] * (1 - a)) / np.maximum(out_a, 1e-6)
    base[..., 3:] = out_a
    m = place(B.mark_alpha(M, 330), cy=318)
    t = np.linspace(0, 1, S, dtype=np.float32)[:, None, None]
    rgb = B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t
    base[..., :3] = rgb * m[..., None] + base[..., :3] * (1 - m[..., None])
    return base


def save(d, name):
    im = Image.fromarray(B.to8(d), "RGBA").resize((150, 150), Image.LANCZOS)
    im.save(os.path.join(OUT, name), optimize=True)
    return im


def preview(items, dark_path, bright_path):
    dark = Image.open(dark_path).convert("RGB").resize((640, 360), Image.LANCZOS)
    bright = Image.open(bright_path).convert("RGB").resize((640, 360), Image.LANCZOS)
    f = ImageFont.truetype(os.path.join(B.INTER, "Inter-SemiBold.otf"), 22)
    rows = []
    for name, im in items:
        row = Image.new("RGB", (640 * 2 + 30, 360 + 40), (255, 255, 255))
        for k, bg in enumerate((dark, bright)):
            fr = bg.copy().convert("RGBA")
            # YouTube draws the watermark in the bottom-right corner, small: about 1/17 of the player's width here
            sz = 40
            wm = im.resize((sz, sz), Image.LANCZOS)
            fr.alpha_composite(wm, (640 - sz - 14, 360 - sz - 14))
            # a 3x zoom of the corner, so the detail is visible
            crop = fr.crop((640 - 120, 360 - 90, 640, 360)).resize((240, 180), Image.LANCZOS)
            fr.alpha_composite(crop.convert("RGBA"), (16, 16))
            ImageDraw.Draw(fr).rectangle([16, 16, 256, 196], outline=(255, 255, 255), width=2)
            row.paste(fr.convert("RGB"), (k * 670, 0))
        ImageDraw.Draw(row).text((8, 366), name, font=f, fill=(20, 20, 30))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(os.path.join(OUT, "watermarks_preview.jpg"), quality=90)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    M = B.mark_mask()
    items = [("1. white (minimal)", save(wm_white(M), "wm_white.png")),
             ("2. white with a soft shadow (reads on bright frames too)", save(wm_white_shadow(M), "wm_white_shadow.png")),
             ("3. gold with a soft shadow", save(wm_gold_shadow(M), "wm_gold_shadow.png")),
             ("4. round badge: the channel picture (looks like a button)", save(wm_badge(M), "wm_badge.png")),
             ("5. split blue and gold, with a shadow", save(wm_split(M), "wm_split.png")),
             ("6. waiting-room TV", save(wm_tv(M), "wm_tv.png"))]
    preview(items, sys.argv[1], sys.argv[2])
    print(sorted(os.listdir(OUT)))
