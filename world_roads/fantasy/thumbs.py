"""Thumbnails for the fantasy race, 1280x720, drawn at 1920x1080 and scaled down:  python thumbs.py

    A  "0.1 points apart": the two best seasons, Faulk 2000 and Tomlinson 2006, a tenth of a point apart
    B  "Who's No. 1?": five famous seasons, their values hidden
    C  "Fantasy points lie": the intro's example, total points against points over replacement

Every number comes from ../media/fantasy/seasons.csv and summary.json, as in the film.
-> ../media/fantasy/fantasy_thumb_A.png, _B.png, _C.png
"""
import os, sys, csv, json
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L                                      # noqa: E402

MEDIA = os.path.join(os.path.dirname(HERE), "media", "fantasy")
INK, INK2, MUTED, GOLD = L.INK, L.INK2, L.MUTED, L.GOLD
SUM = json.load(open(os.path.join(HERE, "summary.json")))
SEASONS = {(r["player"], int(r["season"])): r for r in csv.DictReader(open(os.path.join(MEDIA, "seasons.csv")))}


def last(name):
    return name.split()[-1].upper()


def save(im, tag):
    out = os.path.join(MEDIA, f"fantasy_thumb_{tag}.png")
    im.resize((1280, 720), Image.LANCZOS).save(out, optimize=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.2f} MB")


def big_bar(im, d, y, h, x1, r, value, col, glow=0.0):
    if glow:
        L.glow(im, (90, y, x1, y + h), col, glow, 26)
    L.bar(im, 90, y, x1, h, col)
    d.text((126, y + h / 2 + 4), f"{last(r['player'])} '{r['season'] % 100:02d}", font=L.bc(int(h * 0.72), "Black"),
           fill=(255, 255, 255), anchor="lm")
    d.text((x1 - 28, y + h / 2 + 4), value, font=L.bc(int(h * 0.62), "BlackItalic"),
           fill=tuple(int(c * 0.28) for c in col), anchor="rm")


def thumb_a():
    """the two best seasons, a tenth of a point apart"""
    im = L.background(grain=False)
    d = ImageDraw.Draw(im, "RGBA")
    a, b = SUM["first"], SUM["second"]
    d.text((86, 190), "THE GREATEST", font=L.bc(178, "BlackItalic"), fill=INK, anchor="ls")
    d.text((86, 360), "FANTASY SEASON EVER", font=L.bc(178, "BlackItalic"), fill=INK, anchor="ls")
    for k, r in enumerate((a, b)):
        y = 460 + 210 * k
        big_bar(im, d, y, 175, 1330 - 2 * k, r, f"{r['vorp']:.1f}", L.POSC[r["pos"]], glow=0.6 if k == 0 else 0.0)
    gap = a["vorp"] - b["vorp"]
    d.text((1625, 730), f"{gap:.1f}", font=L.bc(360, "Black"), fill=GOLD, anchor="ms")
    d.text((1625, 820), "POINTS APART", font=L.bc(68, "ExtraBold"), fill=INK, anchor="ms")
    d.text((92, 980), "EVERY SEASON SINCE 1980, RANKED", font=L.bc(58, "SemiBold"), fill=INK2, anchor="ls")
    save(im, "A")


def thumb_b():
    """five famous seasons, values hidden: who is No. 1?"""
    im = L.background(grain=False)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((90, 330), "WHO'S", font=L.bc(250, "BlackItalic"), fill=INK, anchor="ls")
    d.text((90, 590), "#1?", font=L.bc(330, "BlackItalic"), fill=GOLD, anchor="ls")
    d.text((96, 700), "THE GREATEST FANTASY SEASONS", font=L.bc(56, "ExtraBold"), fill=INK, anchor="ls")
    d.text((96, 770), "EVERY SEASON SINCE 1980, RANKED", font=L.bc(44, "SemiBold"), fill=INK2, anchor="ls")
    picks = sorted([(SUM["first"]["player"], SUM["first"]["season"]), (SUM["second"]["player"], SUM["second"]["season"]),
                    ("Christian McCaffrey", 2019), ("Priest Holmes", 2002), ("Cooper Kupp", 2021)], key=lambda p: p[1])
    for k, (p, y) in enumerate(picks):
        r = SEASONS[p, y]
        col = L.POSC[r["pos"]]
        yy = 150 + 150 * k
        L.bar(im, 940, yy, 1660, 112, col)
        d.text((972, yy + 60), f"{last(p)} '{y % 100:02d}", font=L.bc(84, "Black"), fill=(255, 255, 255), anchor="lm")
        d.text((1700, yy + 60), "?", font=L.bc(110, "Black"), fill=GOLD, anchor="lm")
    save(im, "B")


def thumb_c():
    """total points against points over replacement, from the intro's example"""
    im = L.background(grain=False)
    d = ImageDraw.Draw(im, "RGBA")
    x = SUM["example"]
    d.text((86, 210), "FANTASY POINTS", font=L.bc(200, "BlackItalic"), fill=INK, anchor="ls")
    d.text((86 + d.textlength("FANTASY POINTS ", font=L.bc(200, "BlackItalic")), 210), "LIE", font=L.bc(200, "BlackItalic"),
           fill=GOLD, anchor="ls")
    k = 1180 / max(x["qb"]["points"], x["rb"]["points"])
    for n, r in enumerate((x["qb"], x["rb"])):
        y = 330 + 300 * n
        col = L.POSC[r["pos"]]
        x0, xf, x1 = 90, 90 + r["free"] * k, 90 + r["points"] * k
        d.text((x0, y + 4), f"{r['player'].upper()}  ·  {r['points']} PTS", font=L.bc(64, "ExtraBold"), fill=INK, anchor="ls")
        L.bar(im, x0, y + 30, x1, 150, col)
        L.bar(im, x0, y + 30, xf, 150, (78, 88, 108))
        d.text(((x0 + xf) / 2, y + 108), f"FREE {r['pos']}: {r['free']}", font=L.bc(56, "Bold"), fill=INK, anchor="mm")
        best = r["value"] == max(x["qb"]["value"], x["rb"]["value"])
        d.text((x1 + 30, y + 108), f"+{r['value']}", font=L.bc(150 if best else 110, "Black"),
               fill=GOLD if best else INK2, anchor="lm")
    save(im, "C")


if __name__ == "__main__":
    thumb_a()
    thumb_b()
    thumb_c()
