"""Three thumbnail options (1280x720) for the knockout film, built from the
simulation caches and one rendered still.

    python make_thumbs.py path/to/1080p_still.png
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

import film_compose as F
import make_film as MF

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
INK, PAPER = (16, 18, 22), (252, 252, 251)
BLUE, ORANGE, YELLOW = (42, 120, 214), (235, 104, 52), (250, 196, 40)


def headline(d, xy, s, size, fill=(255, 255, 255), stroke=INK, anchor="la", width=10):
    d.text(xy, s, font=F.font(size, "Black", True), fill=fill, anchor=anchor, stroke_width=width, stroke_fill=stroke)


def pill(d, xy, s, size, fill, ink=(255, 255, 255), anchor="l"):
    f = F.font(size, "Bold")
    w = int(d.textlength(s, font=f))
    x, y = xy
    if anchor == "r":
        x -= w + 2 * size
    elif anchor == "m":
        x -= (w + 2 * size) // 2
    d.rounded_rectangle((x, y, x + w + 2 * size, y + int(size * 1.9)), int(size * 0.95), fill=fill)
    d.text((x + size, y + int(size * 0.95)), s, font=f, fill=ink, anchor="lm")


def main(still):
    out = os.path.join(HERE, "thumbnails")
    os.makedirs(out, exist_ok=True)
    ms = {m["key"]: m for m in MF.bracket()}
    run = lambda load, tool: F.Run(MF.run_dir(load, tool))
    score = lambda key: {ms[key]["a"]: ms[key]["ra"], ms[key]["b"]: ms[key]["rb"]}     # by tool, whichever side it stood on

    # A: the 3-D still, spoon (left) against the stand mixer (right); render it with
    #    bowl_scene.py --left cache/film/stripes_spoon --right cache/film/stripes_paddle_large
    sc = score("S1")
    im = Image.open(still).convert("RGB")
    w, h = im.size
    im = im.crop((int(w * 0.08), int(h * 0.13), int(w * 0.92), int(h * 0.13) + int(w * 0.84 * H / W))).resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    headline(d, (W // 2, 26), "SPOON vs STAND MIXER", 82, anchor="ma")
    pill(d, (150, 606), f"{100 * sc['spoon']:.0f}% MIXED", 44, F.tool_rgb("spoon"))
    pill(d, (W - 150, 606), f"{100 * sc['paddle_large']:.0f}% MIXED", 44, F.tool_rgb("paddle_large"), anchor="r")
    im.save(os.path.join(out, "mix_off_thumb_A.png"))

    # B: eight top-down bowls five seconds in
    im = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(im)
    order = ["spoon", "whisk", "fork", "paddle_large", "wire_whip", "paddle_small", "twin_beaters", "spoon_centre"]
    s = 250
    for k, tool in enumerate(order):
        r = run("wedges", tool)
        x = 46 + (k % 4) * (s + 62)
        y = 150 + (k // 4) * (s + 34)
        im.paste(Image.new("RGB", (s + 12, s + 12), INK), (x - 6, y - 6))
        d.ellipse((x - 6, y - 6, x + s + 5, y + s + 5), fill=(60, 62, 68))
        disc = r.disc(int(5.0 * 24), s)
        im.paste(disc, (x, y), disc)
    headline(d, (W // 2, 14), "8 WAYS TO MIX.  1 WINNER.", 88, fill=YELLOW, stroke=INK, anchor="ma", width=6)
    im.save(os.path.join(out, "mix_off_thumb_B.png"))

    # C: same spoon, two techniques
    sc = score("Q1")
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    s = 470
    for tool, x in (("spoon", 70), ("spoon_centre", W - 70 - s)):
        r = run("wedges", tool)
        d.ellipse((x - 8, 196, x + s + 7, 196 + s + 15), fill=(214, 211, 204))
        disc = r.disc(r.n - 1, s)
        im.paste(disc, (x, 204), disc)
    headline(d, (W // 2, 12), "SAME SPOON.", 120, fill=INK, stroke=PAPER, anchor="ma", width=0)
    pill(d, (70 + s // 2, 150), f"{100 * sc['spoon']:.0f}% MIXED", 38, F.tool_rgb("spoon"), anchor="m")
    pill(d, (W - 70 - s // 2, 150), f"{100 * sc['spoon_centre']:.0f}% MIXED", 38, F.tool_rgb("spoon_centre"), anchor="m")
    headline(d, (W // 2, H // 2 + 30), "vs", 90, fill=INK, stroke=PAPER, anchor="mm", width=8)
    im.save(os.path.join(out, "mix_off_thumb_C.png"))
    print("wrote 3 thumbnails to", out)


if __name__ == "__main__":
    main(sys.argv[1])
