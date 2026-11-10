"""Variations on thumbnail A ("Embers"): a Cycles photograph of the film's scene, an editorial serif in warm
off-white on the darker side, a small caption in ember coral. One function, several layouts.

    python3 compose_a.py RENDER_DIR OUTDIR
RENDER_DIR holds render_embers_orange_1944.png, render_europe_low_1944.png and render_europe_map_1944.png
(thumb_render.py with T_EMBER=1; the commands are in its docstring).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose as C                                   # fonts (fetched on first use), grain, shadow, colours

W, H = 1280, 720
INK = C.INK_LIGHT
CORAL = (240, 138, 100)


def darken(im, left=0.62, strength=0.58, top=0.0):
    """The photograph's own falloff, stronger where the type sits (the left, and optionally the top)."""
    a = np.asarray(im).astype(np.float32)
    xx = np.linspace(0, 1, W)[None, :, None]
    a *= (1 - strength) + strength * np.clip(xx / left, 0, 1) ** 1.4
    if top:
        yy = np.linspace(0, 1, H)[:, None, None]
        a *= (1 - top) + top * np.clip(yy / 0.35, 0, 1) ** 1.2
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def layout(photo, lines, caption=None, x=62, y=84, step=86, cap_y=44, left=0.62, strength=0.58, top=0.0):
    """lines: (text, size, italic) top to bottom; the caption sits above them."""
    im = Image.open(photo).convert("RGB").resize((W, H), Image.LANCZOS)
    im = C.grain(darken(im, left, strength, top), 3.5)
    fonts = [(t, C.serif(sz, italic=it), sz) for t, sz, it in lines]

    def draw(d, fill=None):
        yy = y
        for t, f, sz in fonts:
            d.text((x, yy), t, font=f, fill=fill if fill is not None else INK)
            yy += int(step * sz / 90)
    C.soft_shadow(im, draw, radius=18, opacity=170)
    draw(ImageDraw.Draw(im))
    if caption:
        ImageDraw.Draw(im).text((x + 3, cap_y), caption, font=C.inter(21, "Inter-SemiBold.otf"), fill=CORAL)
    return im


def main(rd, out):
    os.makedirs(out, exist_ok=True)
    emb = os.path.join(rd, "render_embers_orange_1944.png")
    low = os.path.join(rd, "render_europe_low_1944.png")
    mp = os.path.join(rd, "render_europe_map_1944.png")
    cap = "2,000 YEARS  ·  640 MILLION LIVES"
    designs = {
        # the original A, with ember-coloured sparks
        "A1_embers_orange": (emb, [("Every one of your", 90, False), ("ancestors", 90, False), ("survived this.", 90, True)],
                             dict(caption=cap)),
        # Europe, 1944: the same words
        "A2_europe_survived": (low, [("Every one of your", 84, False), ("ancestors", 84, False), ("survived this.", 84, True)],
                               dict(caption="EUROPE, 1944  ·  640 MILLION LIVES IN 2,000 YEARS", y=82, left=0.55,
                                    strength=0.62)),
        # Europe: closer to home
        "A3_europe_great_grandparents": (low, [("Your great-", 92, False), ("grandparents", 92, False),
                                               ("lived through this.", 92, True)],
                                         dict(caption="EUROPE, 1944", y=82, left=0.58, strength=0.62)),
        # the map view, a dateline
        "A4_map_1944": (mp, [("1944.", 150, False), ("Every one of your ancestors", 56, False),
                             ("survived this.", 56, True)],
                        dict(caption=None, y=60, step=88, left=0.55, strength=0.6)),
        # the map view, the page's words in this style
        "A5_map_still_here": (mp, [("640 million", 98, False), ("lifted away.", 98, True), ("You're still here.", 60, True)],
                              dict(caption="2,000 YEARS OF CATASTROPHE, IN SAND", y=82, step=92, left=0.55, strength=0.62)),
    }
    for name, (photo, lines, kw) in designs.items():
        if not os.path.exists(photo):
            print("missing", photo)
            continue
        im = layout(photo, lines, **kw)
        im.save(os.path.join(out, f"thumb_{name}.jpg"), quality=92)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
