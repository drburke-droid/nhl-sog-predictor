"""More channel-art directions for Calgary Vision Centre (alternatives to make_branding.py's gold-on-navy set).

    python ideas.py <dir with the coast film's 4K clean frames>
        (RES=2 python film.py clean 12.6 77.0 123.6 in world_roads/coast; frames are out/clean_*.png)
    -> ideas/logo_*.png (800x800), ideas/banner_*.jpg (2560x1440), ideas/logos_sheet.jpg, ideas/banners_sheet.jpg

Logos: gold (the current one), inverse (navy on a gold disc), split (blue and gold, the films' "not yet" and
"done"), globe (the half-gold globe), chart (the mark as the top letter of an eye chart), tv (an old
waiting-room television).
Banners: gold world, frontier (half blue, half gold), calgary (the Bow and Elbow from the film's opening),
chart (an eye chart whose last line is the joke), tv (the waiting room).
"""
import os, sys, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_branding as B  # noqa: E402

OUT = os.path.join(HERE, "ideas")
NAVY = np.array([11, 22, 38], np.float32) / 255
CYAN = np.array([70, 178, 255], np.float32) / 255
W, H = 2560, 1440


def anton(s):
    return ImageFont.truetype(os.path.join(B.FONTS, "Anton-Regular.ttf"), s)


def inter(s, w="SemiBold"):
    return ImageFont.truetype(os.path.join(B.INTER, f"Inter-{w}.otf"), s)


def solid_mark(alpha, rgb):
    return np.concatenate([np.broadcast_to(np.asarray(rgb, np.float32), alpha.shape + (3,)), alpha[..., None]], -1)


def load_frame(path, w, h):
    fr = cv2.imread(path)[..., ::-1].astype(np.float32) / 255
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)


def save_png(a, name):
    Image.fromarray(B.to8(a)).save(os.path.join(OUT, name), optimize=True)


# ------------------------------------------------------------------ logos (800 x 800, shown as a circle)
def logo_gold(M):
    c = B.radial_bg(800, 800, cy=0.5, r=0.72)
    al = B.mark_alpha(M, 560)
    B.paste(c, B.gold_mark(al), 120, (800 - al.shape[0]) // 2 + 6, glow=0.55, glow_sigma=22)
    return c


def logo_inverse(M):
    yy, xx = np.mgrid[0:800, 0:800].astype(np.float32)
    t = (yy / 800)[..., None]
    c = B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t
    c *= (1 - 0.18 * np.clip(np.hypot(xx - 400, yy - 380) / 420, 0, 1) ** 2)[..., None]
    al = B.mark_alpha(M, 560)
    B.paste(c, solid_mark(al, NAVY), 120, (800 - al.shape[0]) // 2 + 6)
    return c


def logo_split(M):
    c = B.radial_bg(800, 800, cy=0.5, r=0.72)
    al = B.mark_alpha(M, 560)
    h, w = al.shape
    x = np.arange(w)[None, :, None]
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    gold = B.GOLD_TOP * (1 - t) + B.GOLD_BOT * t
    rgb = np.where(x < w / 2, np.broadcast_to(CYAN, (h, w, 3)), gold)
    B.paste(c, np.concatenate([rgb, al[..., None]], -1), 120, (800 - h) // 2 + 6, glow=0.35, glow_sigma=20)
    return c


def logo_globe(M, frames):
    import thumbs as T
    T.TW, T.TH, T.S = 400, 400, 2
    g = dict(cx=200, cy=200, r=200, lon0=40.0, lat0=20.0)
    fr = np.asarray(Image.open(os.path.join(frames, "clean_0077.00.png")).convert("RGB")).astype(np.float32) / 255
    c = T.globe(fr, g)
    lum = c.max(axis=2, keepdims=True)
    c = c * ((1 - np.exp(-lum * 1.5)) / np.maximum(lum, 1e-6)) * 0.62
    al = B.mark_alpha(M, 520)
    sh = np.zeros((800, 800), np.float32)
    y0 = (800 - al.shape[0]) // 2
    sh[y0:y0 + al.shape[0], 140:140 + al.shape[1]] = al
    c *= (1 - 0.65 * cv2.GaussianBlur(sh, (0, 0), 16))[..., None]
    B.paste(c, solid_mark(al, (1.0, 1.0, 1.0)), 140, y0)
    return c


def logo_chart(M):
    c = np.ones((800, 800, 3), np.float32) * np.array([0.97, 0.96, 0.93], np.float32)
    al = B.mark_alpha(M, 420)
    B.paste(c, solid_mark(al, (0.05, 0.05, 0.07)), 190, 150)
    im = Image.fromarray(B.to8(c))
    d = ImageDraw.Draw(im)
    rows = [("F  P", 92, 430), ("T  O  Z", 62, 545), ("L  P  E  D", 42, 625), ("P  E  C  F  D", 28, 684)]
    for txt, s, y in rows:
        d.text((400, y), txt, font=inter(s, "Bold"), fill=(15, 15, 20), anchor="ma")
    d.line([(250, 735), (550, 735)], fill=(200, 40, 40), width=4)
    return np.asarray(im).astype(np.float32) / 255


def logo_tv(M):
    c = B.radial_bg(800, 800, cy=0.5, r=0.75)
    im = Image.fromarray(B.to8(c)).convert("RGBA")
    d = ImageDraw.Draw(im)
    gold = (255, 196, 84, 255)
    d.line([(330, 150), (400, 228)], fill=gold, width=12)
    d.line([(470, 150), (400, 228)], fill=gold, width=12)
    d.rounded_rectangle([120, 228, 680, 620], radius=70, outline=gold, width=20)
    d.rounded_rectangle([250, 620, 290, 660], radius=8, fill=gold)
    d.rounded_rectangle([510, 620, 550, 660], radius=8, fill=gold)
    c = np.asarray(im.convert("RGB")).astype(np.float32) / 255
    scr = np.zeros((800, 800), np.float32)
    scr[255:594, 147:654] = 1
    yy = np.arange(800)[:, None]
    c += (scr * (0.035 + 0.02 * (yy % 6 < 3)))[..., None] * np.array([0.3, 0.5, 1.0])
    al = B.mark_alpha(M, 380)
    B.paste(c, B.gold_mark(al), 210, 424 - al.shape[0] // 2, glow=0.5, glow_sigma=16)
    return c


# ------------------------------------------------------------------ banners (2560 x 1440)
def words(im, x, y, name_col=(255, 255, 255), tag_col=(255, 205, 120), sub_col=(196, 206, 222), shadow=True,
          name="CALGARY VISION CENTRE", tag="Actually interesting waiting room videos",
          sub="Eye science  ·  Light-speed maps of the whole world  ·  Calgary, Alberta", size=112):
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.text((x, y - 128 * size / 112), name, font=anton(size), fill=name_col + (255,))
    d.text((x + 4, y + 22), tag, font=inter(46, "MediumItalic"), fill=tag_col + (255,))
    if sub:
        d.text((x + 4, y + 92), sub, font=inter(30, "Regular"), fill=sub_col + (255,))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(lay.getchannel("A").filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.8)))
        im.alpha_composite(sh)
    im.alpha_composite(lay)


def text_width(size=112):
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    return max(d.textlength("CALGARY VISION CENTRE", font=anton(size)),
               d.textlength("Actually interesting waiting room videos", font=inter(46, "MediumItalic")),
               d.textlength("Eye science  ·  Light-speed maps of the whole world  ·  Calgary, Alberta", font=inter(30, "Regular")))


def map_banner(M, frame, dim=0.62, band_dim=0.40, mark="gold"):
    c = frame.copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    band = np.exp(-((yy - H / 2) / 260) ** 2) * np.exp(-((xx - W / 2) / 900) ** 2)
    c = c * (dim - band_dim * band[..., None]) + B.radial_bg(W, H, cy=0.5, r=0.9) * 0.35
    mw = 360
    al = B.mark_alpha(M, mw)
    x0 = int((W - (mw + 64 + text_width())) / 2)
    B.paste(c, B.gold_mark(al), x0, int(H / 2 - al.shape[0] / 2), glow=0.45, glow_sigma=20)
    im = Image.fromarray(B.to8(c)).convert("RGBA")
    words(im, x0 + mw + 64, H / 2)
    return im


def banner_chart(M):
    paper = np.array([246, 244, 238])
    im = Image.new("RGBA", (W, H), tuple(paper) + (255,))
    d = ImageDraw.Draw(im)
    ink = (18, 18, 24, 255)
    # outside the band every device shows (TV only): the top of the chart and its smallest lines
    top = [("E", 230, 40), ("F  P", 150, 300)]
    for txt, s, y in top:
        d.text((W / 2, y), txt, font=inter(s, "Bold"), fill=ink, anchor="ma")
    for i, (txt, s) in enumerate([("D  E  F  P  O  T  E  C", 34), ("L  E  F  O  D  P  C  T", 26), ("F  D  P  L  T  C  E  O", 20)]):
        d.text((W / 2, 980 + 60 * i), txt, font=inter(s, "Bold"), fill=ink, anchor="ma")
    for i, (lab, y) in enumerate((("20/200", 150), ("20/100", 360), ("20/40", 598), ("20/20", 845), ("20/10", 1000))):
        d.text((180, y), lab, font=inter(30, "Medium"), fill=(150, 150, 150, 255))
    # the band: mark, name, tagline, and the line you can't read on a phone
    al = B.mark_alpha(M, 250)
    a = np.asarray(im.convert("RGB")).astype(np.float32) / 255
    B.paste(a, solid_mark(al, (0.07, 0.07, 0.09)), int(W / 2 - 125), 540)
    im = Image.fromarray(B.to8(a)).convert("RGBA")
    d = ImageDraw.Draw(im)
    d.text((W / 2, 690), "CALGARY VISION CENTRE", font=anton(96), fill=ink, anchor="ma")
    d.text((W / 2, 812), "Actually interesting waiting room videos", font=inter(44, "MediumItalic"), fill=(170, 110, 20, 255), anchor="ma")
    d.text((W / 2, 885), "if you can read this line on your phone, you probably don't need us. watch anyway.",
           font=inter(22, "Medium"), fill=(60, 60, 70, 255), anchor="ma")
    # the red and green bars every chart ends with
    d.rectangle([W / 2 - 520, 1180, W / 2, 1300], fill=(200, 40, 40, 255))
    d.rectangle([W / 2, 1180, W / 2 + 520, 1300], fill=(30, 140, 70, 255))
    for k, (x, txt) in enumerate(((W / 2 - 260, "O  C"), (W / 2 + 260, "C  O"))):
        d.text((x, 1205), txt, font=inter(56, "Bold"), fill=(10, 10, 10, 255), anchor="ma")
    return im


def banner_tv(M, frame):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall = np.array([24, 34, 52], np.float32) / 255
    c = wall * (0.55 + 0.45 * np.exp(-((xx - 1500) / 1100) ** 2 - ((yy - 640) / 700) ** 2))[..., None]
    c[1110:] *= 0.55                                      # floor
    # the TV on the wall, showing the gold world
    tw, th = 680, 383
    tx, ty = 1700 - tw // 2, 720 - th // 2
    scr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    glow = np.zeros((H, W), np.float32)
    glow[ty:ty + th, tx:tx + tw] = 1
    c += cv2.GaussianBlur(glow, (0, 0), 70)[..., None] * np.array([0.30, 0.20, 0.06]) * 0.9
    c[ty - 18:ty + th + 18, tx - 18:tx + tw + 18] = 0.03
    c[ty:ty + th, tx:tx + tw] = scr * 1.1
    im = Image.fromarray(B.to8(c)).convert("RGBA")
    d = ImageDraw.Draw(im)
    d.rectangle([tx, ty + th - 64, tx + tw, ty + th], fill=(10, 14, 22, 215))
    lab = "NOW SHOWING:"
    d.text((tx + 20, ty + th - 48), lab, font=inter(24, "Bold"), fill=(255, 196, 84, 255))
    d.text((tx + 32 + d.textlength(lab, font=inter(24, "Bold")), ty + th - 48), "every shore on Earth at light speed",
           font=inter(24, "Medium"), fill=(235, 240, 248, 255))
    # waiting-room chairs (TV and desktop only)
    for i in range(7):
        x = 260 + i * 330
        d.rounded_rectangle([x, 1150, x + 250, 1200], radius=16, fill=(58, 74, 100, 255))
        d.rounded_rectangle([x + 10, 1010, x + 240, 1150], radius=22, fill=(48, 62, 86, 255))
        d.rectangle([x + 30, 1200, x + 44, 1290], fill=(36, 46, 64, 255))
        d.rectangle([x + 206, 1200, x + 220, 1290], fill=(36, 46, 64, 255))
    # name on the left wall
    a = np.asarray(im.convert("RGB")).astype(np.float32) / 255
    al = B.mark_alpha(M, 220)
    B.paste(a, B.gold_mark(al), 545, 540, glow=0.4, glow_sigma=16)
    im = Image.fromarray(B.to8(a)).convert("RGBA")
    d = ImageDraw.Draw(im)
    d.text((545, 668), "CALGARY", font=anton(96), fill=(255, 255, 255, 255))
    d.text((545, 790), "VISION CENTRE", font=anton(56), fill=(255, 255, 255, 255))
    d.text((549, 872), "Actually interesting waiting room videos", font=inter(33, "MediumItalic"),
           fill=(255, 205, 120, 255))
    return im


def sheet_logos(items):
    cell = 300
    sheet = Image.new("RGB", (cell * len(items), cell + 60), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    mask = Image.new("L", (800, 800), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 799, 799], fill=255)
    for i, (name, a) in enumerate(items):
        im = Image.fromarray(B.to8(a)).convert("RGBA")
        im.putalpha(mask)
        sheet.paste(im.resize((260, 260), Image.LANCZOS), (i * cell + 20, 16), im.resize((260, 260), Image.LANCZOS))
        small = im.resize((64, 64), Image.LANCZOS)
        sheet.paste(small, (i * cell + 216, 218), small)
        d.text((i * cell + cell / 2, cell + 22), f"{i + 1}. {name}", font=inter(26, "SemiBold"), fill=(20, 20, 30), anchor="ma")
    sheet.save(os.path.join(OUT, "logos_sheet.jpg"), quality=92)


def sheet_banners(items):
    w, h = 1280, 720
    sheet = Image.new("RGB", (w * 2 + 30, (h + 70) * ((len(items) + 1) // 2) + 10), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(items):
        x, y = (i % 2) * (w + 30), (i // 2) * (h + 70)
        sm = im.convert("RGB").resize((w, h), Image.LANCZOS)
        dd = ImageDraw.Draw(sm)
        dd.rectangle([(w - 773) // 2, (h - 212) // 2, (w + 773) // 2, (h + 212) // 2], outline=(255, 90, 90), width=2)
        sheet.paste(sm, (x, y))
        d.text((x + 8, y + h + 12), f"{i + 1}. {name}   (red box: what every device shows)", font=inter(30, "SemiBold"), fill=(20, 20, 30))
    sheet.save(os.path.join(OUT, "banners_sheet.jpg"), quality=88)


if __name__ == "__main__":
    frames = sys.argv[1]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "world_roads", "coast"))
    os.makedirs(OUT, exist_ok=True)
    M = B.mark_mask()
    logos = [("gold", logo_gold(M)), ("inverse", logo_inverse(M)), ("split", logo_split(M)),
             ("globe", logo_globe(M, frames)), ("eye chart", logo_chart(M)), ("waiting-room TV", logo_tv(M))]
    for name, a in logos:
        save_png(a, "logo_" + {"eye chart": "eye_chart", "waiting-room TV": "waiting_room_tv"}.get(name, name) + ".png")
    sheet_logos(logos)
    gold = load_frame(os.path.join(frames, "clean_0123.60.png"), W, H)
    half = load_frame(os.path.join(frames, "clean_0077.00.png"), W, H)
    city = load_frame(os.path.join(frames, "clean_0012.60.png"), W, H)
    banners = [("gold world", map_banner(M, gold)),
               ("frontier: half blue, half gold", map_banner(M, half, dim=0.75, band_dim=0.45)),
               ("Calgary's rivers", map_banner(M, city * 1.6, dim=0.8, band_dim=0.45)),
               ("eye chart", banner_chart(M)),
               ("waiting-room TV", banner_tv(M, gold))]
    for name, im in banners:
        fn = {"gold world": "gold", "frontier: half blue, half gold": "frontier", "Calgary's rivers": "calgary_rivers",
              "eye chart": "eye_chart", "waiting-room TV": "waiting_room_tv"}[name]
        im.convert("RGB").save(os.path.join(OUT, f"banner_{fn}.jpg"), quality=92, optimize=True)
    sheet_banners(banners)
    print(sorted(os.listdir(OUT)))
