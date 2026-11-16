"""Three thumbnail options (1280x720) for the knockout film, built from rendered
stills (the commands that make each are in the comments below).

    python make_thumbs.py A_still.png C_still.png B_tiles_folder
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


def main(still, still_c, tiles):
    out = os.path.join(HERE, "thumbnails")
    os.makedirs(out, exist_ok=True)
    ms = {m["key"]: m for m in MF.bracket()}
    run = lambda load, tool: F.Run(MF.run_dir(load, tool))
    score = lambda key: {ms[key]["a"]: ms[key]["ra"], ms[key]["b"]: ms[key]["rb"]}     # by tool, whichever side it stood on

    # A: the 3-D still, spoon (left) against the stand mixer's paddle (right); render it 16:9 with
    #    bowl_scene.py --left cache/film/stripes_spoon --right cache/film/stripes_paddle_large --no-hud
    #                  --still 150 --res 1920x1080 --samples 40 --elev 31 --dist 820 --target 0,20,95
    sc = score("S1")
    im = Image.open(still).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    headline(d, (W // 2, 20), "SPOON vs STAND MIXER", 82, anchor="ma")
    pill(d, (70, 612), f"{100 * sc['spoon']:.0f}% MIXED", 44, F.tool_rgb("spoon"))
    pill(d, (W - 70, 612), f"{100 * sc['paddle_large']:.0f}% MIXED", 44, F.tool_rgb("paddle_large"), anchor="r")
    im.save(os.path.join(out, "mix_off_thumb_A.png"))

    # B: the eight methods five seconds in, as rendered bowls.  One still per tool in tiles/<tool>/still_0120.png:
    #    bowl_scene.py --left cache/film/wedges_<tool> --no-hud --bare --still 120 --res 900x810 --samples 40
    #                  --elev 62 --dist 245 --target 0,-6,8 --out tiles/<tool>
    im = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(im)
    order = ["spoon", "whisk", "fork", "paddle_large", "wire_whip", "paddle_small", "twin_beaters", "spoon_centre"]
    tw, th, gap, x0, y0 = 296, 266, 16, 24, 136
    mask = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, tw - 1, th - 1), 20, fill=255)
    for k, tool in enumerate(order):
        tile = Image.open(os.path.join(tiles, tool, "still_0120.png")).convert("RGB")
        w, h = tile.size                                   # crop in on the bowl, keeping the tile's 10:9 shape
        tile = tile.crop((int(w * 0.11), int(h * 0.075), int(w * 0.89), int(h * 0.855))).resize((tw, th), Image.LANCZOS)
        im.paste(tile, (x0 + (k % 4) * (tw + gap), y0 + (k // 4) * (th + gap)), mask)
    headline(d, (W // 2, 14), "8 WAYS TO MIX.  1 WINNER.", 88, fill=YELLOW, stroke=INK, anchor="ma", width=6)
    im.save(os.path.join(out, "mix_off_thumb_B.png"))

    # C: same spoon, two techniques, as a rendered still of the two bowls near the end of the match:
    #    bowl_scene.py --left cache/film/wedges_spoon --right cache/film/wedges_spoon_centre --no-hud --bare
    #                  --still 340 --res 1920x1080 --samples 64 --elev 62 --dist 455 --target 0,24,6
    sc = score("Q1")
    im = Image.open(still_c).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    headline(d, (W // 2, 4), "SAME SPOON.", 128, anchor="ma", width=12)
    headline(d, (W // 2, H // 2 + 60), "vs", 84, anchor="mm", width=9)
    pill(d, (70, 612), f"{100 * sc['spoon']:.0f}% MIXED", 44, F.tool_rgb("spoon"))
    pill(d, (W - 70, 612), f"{100 * sc['spoon_centre']:.0f}% MIXED", 44, F.tool_rgb("spoon_centre"), anchor="r")
    im.save(os.path.join(out, "mix_off_thumb_C.png"))
    print("wrote 3 thumbnails to", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
