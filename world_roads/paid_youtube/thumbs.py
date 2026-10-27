"""Thumbnails for the best-paid athletes race (paid/, film media/paid/best_paid_athletes.mp4), 1280x720, drawn at
1920x1080 and scaled down:

    python thumbs.py [PAID]      PAID: that race's data, its media/paid directory (default ../media/paid)

    A  "Same dollars": Babe Ruth's best season against Michael Jordan's 1997-98, both in 2025 dollars
    B  "Still No. 1": Jordan's 1997-98 over the best-paid quarterbacks of today
    C  "Highest-paid ever?": Ruth and Jordan, the question without the answer

Every figure comes from that race's season_top.csv. The photos are in photos/ (CREDITS.md: Ruth 1921, Bain News
Service, Library of Congress, no known restrictions; Jordan 1997, Steve Lipofsky, CC BY-SA 3.0). Each is cut out once
with rembg (pip install rembg) and the cutout kept beside it.
-> ../media/paid_youtube/best_paid_thumb_A.png, _B.png, _C.png
"""
import os, sys, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "media", "paid_youtube")
PHOTOS = os.path.join(HERE, "photos")
W, H = 1920, 1080

BG = (5, 7, 13)                                       # the film's colours (paid/video.py)
GOLD = (255, 186, 80)
LEAGUE = {"MLB": (57, 135, 229), "NBA": (217, 89, 38), "NFL": (25, 158, 112), "NHL": (150, 156, 168)}
INK, INK2 = (244, 246, 250), (168, 176, 192)
FD = "/usr/share/fonts/opentype/inter"


def font(size, weight="Black", display=True):
    return ImageFont.truetype(os.path.join(FD, f"Inter{'Display' if display else ''}-{weight}.otf"), size)


def board(paid):
    """the final board and every season's figure in 2025 dollars, from season_top.csv"""
    rows = list(csv.DictReader(open(os.path.join(paid, "season_top.csv"))))
    last = max(int(r["year"]) for r in rows)
    final = sorted((r for r in rows if int(r["year"]) == last), key=lambda r: int(r["rank"]))
    best = {}
    for r in rows:
        k = (r["player"], int(r["season"]))
        best[k] = max(best.get(k, 0.0), float(r["real_2025"]))
    return final, best


def money(v):
    return f"${v / 1e6:.1f}M"


def season(r):
    """a board row's season as the film names it: basketball and hockey seasons span two years"""
    y = int(r["season"])
    return f"{y - 1}-{y % 100:02d}" if r["league"] in ("NBA", "NHL") else f"{y}"


def background():
    """the film's near-black, lifted a little toward the middle"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((x - W / 2) / W, (y - H * 0.45) / H)
    lift = np.clip(1.0 - r * 1.25, 0, 1) ** 1.6
    a = np.array(BG, np.float32)[None, None] + lift[..., None] * np.array([18, 22, 34], np.float32)[None, None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGBA")


def glow(im, cx, cy, rx, ry, col, alpha):
    """a soft coloured light behind a figure"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    g = np.exp(-(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)) * alpha
    layer = Image.new("RGBA", (W, H), col + (0,))
    layer.putalpha(Image.fromarray((g * 255).clip(0, 255).astype(np.uint8)))
    im.alpha_composite(layer)


def cutout(name, erase=()):
    """photos/<name>.jpg without its background, cached as photos/<name>_cut.png; erase: (box, level) pairs, in the
    photo's pixels, for what the cutout kept that isn't the player: inside box, whatever is darker than level goes"""
    path = os.path.join(PHOTOS, f"{name}_cut.png")
    if not os.path.exists(path):
        from rembg import remove, new_session
        cut = remove(Image.open(os.path.join(PHOTOS, f"{name}.jpg")).convert("RGB"),
                     session=new_session("isnet-general-use"))
        luma = np.array(cut.convert("L"), np.float32)
        gone = np.zeros(luma.shape, np.float32)
        for (x0, y0, x1, y1), level in erase:
            gone[y0:y1, x0:x1] = luma[y0:y1, x0:x1] < level
        gone = np.array(Image.fromarray((gone * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5)), np.float32)
        a = np.array(cut.split()[3], np.float32) * (1 - gone / 255)
        cut.putalpha(Image.fromarray(a.astype(np.uint8)))
        cut.save(path)
    return Image.open(path).convert("RGBA")


def figure(cut, box, height, mono=False, fade_left=0.0):
    """a cutout cropped to box and scaled to height; mono: black and white, warmed a little, for the 1920s;
    fade_left: the share of its width over which it fades in from the left (where the photo's own frame cuts it)"""
    crop = cut.crop(box)
    k = height / crop.height
    crop = crop.resize((round(crop.width * k), height), Image.LANCZOS)
    a = crop.split()[3]
    if mono:
        g = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
        crop = ImageOps.colorize(g, (12, 12, 16), (250, 244, 232)).convert("RGBA")
    if fade_left:
        ramp = np.clip(np.arange(crop.width) / (fade_left * crop.width), 0, 1) ** 1.5
        a = Image.fromarray((np.array(a, np.float32) * ramp[None]).astype(np.uint8))
    crop.putalpha(a)
    return crop


def place(im, crop, x, y, col, rim=10, alpha=0.95):
    """a figure with its top-left at x, y, lit at the edge in col"""
    a = crop.split()[3]
    edge = a.filter(ImageFilter.MaxFilter(rim | 1)).filter(ImageFilter.GaussianBlur(rim * 1.6))
    lit = Image.new("RGBA", crop.size, col + (0,))
    lit.putalpha(edge.point(lambda v: int(v * alpha)))
    im.alpha_composite(lit, (x, y))
    im.alpha_composite(crop, (x, y))


def shade_bottom(im, y0, strength=0.85):
    """darken toward the bottom edge, so text there reads over the figures"""
    g = np.clip((np.arange(H, dtype=np.float32) - y0) / (H - y0), 0, 1) ** 1.3 * strength
    a = np.repeat(g[:, None], W, 1)
    layer = Image.new("RGBA", (W, H), BG + (0,))
    layer.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    im.alpha_composite(layer)


def text(im, xy, s, f, fill, anchor="la", shadow=14, sa=0.75):
    """text with a soft dark shadow under it"""
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((xy[0], xy[1] + shadow // 3), s, font=f, fill=(0, 0, 0, int(255 * sa)), anchor=anchor)
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(shadow)))
    ImageDraw.Draw(im).text(xy, s, font=f, fill=fill, anchor=anchor)


def bar(im, x0, y, x1, h, col, r=10):
    """a board bar: the colour with a lit top edge, as in the film"""
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((x0, y, x1, y + h), r, fill=col)
    top = tuple(min(255, int(c * 1.25 + 30)) for c in col)
    d.rounded_rectangle((x0, y, x1, y + max(3, h // 9)), r // 2, fill=top)


def save(im, tag):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"best_paid_thumb_{tag}.png")
    im.convert("RGB").resize((1280, 720), Image.LANCZOS).save(out, optimize=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB")


def thumb_a(final, best, ruth, jordan):
    """Ruth's best season against Jordan's 1997-98, the same dollars"""
    r = max(((k, v) for k, v in best.items() if k[0] == "Babe Ruth"), key=lambda kv: kv[1])
    j = final[0]
    im = background()
    glow(im, 380, 560, 520, 520, LEAGUE["MLB"], 0.55)
    glow(im, 1520, 520, 640, 640, LEAGUE["NBA"], 0.75)
    place(im, ruth(820), 40, H - 820, LEAGUE["MLB"])
    jf = jordan(1060)
    place(im, jf, W - jf.width + 20, H - 1060 + 40, LEAGUE["NBA"])
    shade_bottom(im, 560, 0.92)
    text(im, (W / 2, 40), "PAID IN TODAY'S DOLLARS", font(78), GOLD, anchor="ma")
    text(im, (60, 760), f"RUTH {r[0][1]}", font(64, "ExtraBold"), INK2)
    text(im, (60, 830), money(r[1]), font(190), INK)
    jv, f = money(float(j["real_2025"])), font(300)
    x0 = W - 50 - ImageDraw.Draw(im).textlength(jv, font=f)
    text(im, (x0 + 12, 650), f"JORDAN {season(j)}", font(64, "ExtraBold"), INK2)
    text(im, (W - 50, 720), jv, f, GOLD, anchor="ra", shadow=22, sa=0.9)
    save(im, "A")


def thumb_b(final, best, ruth, jordan):
    """Jordan's 1997-98 still on top of today's quarterbacks"""
    im = background()
    glow(im, 1500, 500, 700, 700, LEAGUE["NBA"], 0.8)
    jf = jordan(1180)
    place(im, jf, W - jf.width + 20, H - 1180 + 70, LEAGUE["NBA"])
    text(im, (70, 40), "STILL", font(200), INK)
    text(im, (60, 210), "#1", font(330), GOLD)
    rows = [r for k, r in enumerate(final) if r["player"] not in {q["player"] for q in final[:k]}][:3]  # one each
    top = float(rows[0]["real_2025"])
    for i, r in enumerate(rows):
        y = 640 + i * 136
        x1 = 70 + 1000 * float(r["real_2025"]) / top
        bar(im, 70, y, x1, 104, LEAGUE[r["league"]])
        ImageDraw.Draw(im).text((96, y + 52), f"{r['player'].split()[-1].upper()} {season(r)}", font=font(58, "Black"),
                                fill=(255, 255, 255), anchor="lm")
        ImageDraw.Draw(im).text((x1 - 24, y + 52), money(float(r["real_2025"])), font=font(58, "Black"),
                                fill=tuple(int(c * 0.25) for c in LEAGUE[r["league"]]), anchor="rm")
    save(im, "B")


def thumb_c(final, best, ruth, jordan):
    """the question: who was paid the most, Ruth's era or Jordan's, or today's?"""
    im = background()
    glow(im, 380, 560, 560, 560, LEAGUE["MLB"], 0.6)
    glow(im, 1540, 540, 620, 620, LEAGUE["NBA"], 0.75)
    place(im, ruth(900), 0, H - 900, LEAGUE["MLB"])
    jf = jordan(1040)
    place(im, jf, W - jf.width + 20, H - 1040 + 40, LEAGUE["NBA"])
    shade_bottom(im, 700, 0.9)
    text(im, (W / 2 + 10, 150), "?", font(560), GOLD, anchor="ma")
    text(im, (W / 2, 830), "HIGHEST-PAID EVER", font(120), INK, anchor="ma")
    text(im, (W / 2, 975), "MLB · NBA · NFL · NHL  ·  IN TODAY'S DOLLARS", font(56, "ExtraBold"), INK2, anchor="ma")
    save(im, "C")


def main():
    paid = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "media", "paid")
    final, best = board(paid)
    rc, jc = cutout("ruth", RUTH_ERASE), cutout("jordan")
    ruth = lambda h: figure(rc, RUTH_BOX, h, mono=True)
    jordan = lambda h: figure(jc, JORDAN_BOX, h, fade_left=0.2)
    for f in (thumb_a, thumb_b, thumb_c):
        f(final, best, ruth, jordan)


RUTH_BOX = (40, 240, 600, 1000)                       # cap to knees (photos/ruth_cut.png, 696x1024)
RUTH_ERASE = [((148, 620, 193, 690), 160),           # the teammate behind Ruth's left forearm
              ((150, 760, 200, 1001), 158)]           # and his left leg (Ruth's own trousers are lighter)
JORDAN_BOX = (0, 30, 1200, 1762)                      # the whole frame (photos/jordan_cut.png, 1200x1762)

if __name__ == "__main__":
    main()
