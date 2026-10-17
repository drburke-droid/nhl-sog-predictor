"""Three thumbnails for A/B/C testing (1280x720), built on frames of the video itself.

    python thumbs.py     -> out/thumb_A.png (America half lit), thumb_B.png (Calgary vs America),
                            thumb_C.png (the last road)
Needs the stills: python video.py usa_run 14 ; python video.py cal_run 53 ; python video.py last 15.6
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import video as V

TW, TH = 1280, 720
OUT = V.OUT
ST = V.ST


def f(n, w="Bold"):
    return ST.font(n, w)


def headline(d, xy, s, size, col, anchor="la"):
    d.text(xy, s, font=f(size), fill=col, stroke_width=max(5, size // 16), stroke_fill=(0, 0, 0), anchor=anchor)


def still(name):
    p = os.path.join(OUT, f"{name}.png")
    assert os.path.exists(p), f"render {name} first (see the docstring)"
    return Image.open(p).convert("RGB")


def clean(im, boxes):
    """paint over the video's own text (HUD, captions) with the background colour"""
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, fill=(5, 8, 15))
    return im


def thumb_A():
    """America mid-run: gold where the photon has been, blue where it hasn't."""
    im = clean(still("usa_run_14.0"), [(0, 0, 1920, 90), (0, 760, 700, 1080), (1540, 700, 1920, 1000)])
    im = im.crop((60, 40, 1860, 1053)).resize((TW, TH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    headline(d, (40, 470), "EVERY ROAD", 120, V.GOLD)
    headline(d, (44, 600), "AT RANDOM", 84, (255, 255, 255))
    headline(d, (TW - 40, 40), "AT LIGHT SPEED", 54, (255, 255, 255), "ra")
    im.save(os.path.join(OUT, "thumb_A.png"))


def thumb_B():
    """Same rule, two maps: Calgary in a second, America in hours."""
    cal = clean(still("cal_run_53.0"), [(0, 0, 640, 1080), (500, 940, 1920, 1080), (500, 0, 1920, 150)]).crop((730, 0, 1630, 1080))
    usa = clean(still("usa_run_46.5"), [(0, 0, 1920, 90), (0, 760, 700, 1080), (1540, 700, 1920, 1000)]).crop((170, 120, 1790, 1060))
    im = Image.new("RGB", (TW, TH), (5, 8, 15))
    im.paste(cal.resize((400, 480), Image.LANCZOS), (10, 150))
    im.paste(usa.resize((840, 487), Image.LANCZOS), (430, 150))
    d = ImageDraw.Draw(im)
    js = V.calgary_stats()
    meta, w, log, secs = V.usa_data()
    headline(d, (210, 30), "CALGARY", 56, (255, 255, 255), "ma")
    headline(d, (850, 30), "AMERICA", 56, (255, 255, 255), "ma")
    headline(d, (210, 590), f"{js['median_s_nonback']:.1f} s", 96, V.GOLD, "ma")
    headline(d, (850, 590), V.fmt_hm(w["seconds"]).replace(" min", "m").replace(" h", "h"), 96, V.GOLD, "ma")
    im.save(os.path.join(OUT, "thumb_B.png"))


def thumb_C():
    """The last road, lit."""
    im = clean(still("last_15.6"), [(0, 0, 1920, 100), (0, 100, 1920, 200), (0, 760, 760, 1080), (1500, 980, 1920, 1080)])
    im = im.crop((240, 60, 1680, 870)).resize((TW, TH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    meta, w, log, secs = V.usa_data()
    headline(d, (40, 40), "THE LAST ROAD", 104, V.GOLD)
    headline(d, (44, 170), "IN AMERICA", 64, (255, 255, 255))
    headline(d, (40, TH - 40), V.fmt_hm(w["seconds"]) + " at light speed", 58, (255, 255, 255), "ls")
    im.save(os.path.join(OUT, "thumb_C.png"))


if __name__ == "__main__":
    thumb_A(); thumb_B(); thumb_C()
    print("wrote", [os.path.join(OUT, f"thumb_{k}.png") for k in "ABC"])
