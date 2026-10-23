"""YouTube thumbnails (1280x720) for the blur-adaptation video, Light Speed / hyperacuity style."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo1_v2 as D
import part2 as P
import optics as O

OUT = f"{P.BASE}/v2/thumbs"
os.makedirs(OUT, exist_ok=True)
W, H = D.W, D.H
F = D.F
from PIL import ImageFont
H1 = ImageFont.truetype(f"{D.FONT_DIR}/InterDisplay-Black.otf", 150)
H2 = ImageFont.truetype(f"{D.FONT_DIR}/InterDisplay-Black.otf", 112)
SUB = ImageFont.truetype(f"{D.FONT_DIR}/InterDisplay-SemiBold.otf", 62)
GOLDT = (255, 190, 90)


def outline_text(d, xy, s, font, fill, anchor="la", stroke=8):
    d.text(xy, s, font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(5, 8, 14))


def save(img, name):
    im = Image.fromarray(D.to8(img)).resize((1280, 720), Image.LANCZOS)
    im.save(f"{OUT}/{name}.png")


# A: the identical pair, gold frames: "ONE OF THESE WILL LOOK SHARPER"
def thumb_a():
    img = D.card_base(0.0).copy()
    S, y = 760, 290
    xl, xr = W // 2 - 20 - S, W // 2 + 20
    em = np.zeros((H, W, 3), np.float32)
    D.frame_rect(em, xl, y, S, S - 20, 10, D.GOLD)
    D.frame_rect(em, xr, y, S, S - 20, 10, D.GOLD)
    img = img + em + D.bloom(em) * 1.3
    ph = D.disp(D.load("61c-14-008"))
    ph = np.asarray(Image.fromarray((ph * 255).astype(np.uint8)).resize((S, S)), np.float32)[: S - 20] / 255
    img[y:y + S - 20, xl:xl + S] = ph
    img[y:y + S - 20, xr:xr + S] = ph
    D.add_dot(img, True, 1.4, W // 2, y + (S - 20) // 2, 1.6)
    ov = D.overlay(lambda d: (outline_text(d, (W // 2, 140), "ONE WILL LOOK SHARPER", H2, (255, 255, 255), "mm"),
                              outline_text(d, (W // 2, 245), "(they're identical)", SUB, GOLDT, "mm", 6)))
    save(D.comp(img, ov), "thumb_A")


# B: perfect eye point vs your smudge: "THIS IS YOUR BLUR"
def thumb_b():
    img = D.card_base(0.0).copy()
    A_ = P.Assets.__new__(P.Assets)
    WA = O.wavefront(O.EYES["A"])
    smudge = P.to_tile(O.heat(O.psf(WA, 14), gain=1.3), 720)
    point = P.to_tile(O.heat(O.psf(np.zeros_like(WA), 14), gain=1.3), 300)
    em = np.zeros((H, W, 3), np.float32)
    D.frame_rect(em, 1120, 150, 720, 720, 8, D.GOLD)
    D.frame_rect(em, 620, 570, 300, 300, 4, np.array([0.8, 0.86, 0.95], np.float32))
    img = img + em + D.bloom(em)
    img[150:870, 1120:1840] += smudge * 1.15
    img[570:870, 620:920] += point * 1.3
    ov = D.overlay(lambda d: (outline_text(d, (80, 150), "THIS IS", H1, (255, 255, 255)),
                              outline_text(d, (80, 310), "YOUR BLUR", H1, GOLDT),
                              d.text((770, 950), "a perfect eye", font=SUB, fill=(210, 222, 238), anchor="mm"),
                              d.text((1480, 950), "your eye", font=SUB, fill=GOLDT, anchor="mm"),
                              d.text((1020, 720), "→", font=H2, fill=(150, 176, 205), anchor="mm")))
    save(D.comp(img, ov), "thumb_B")


# C: bold statement over the cone mosaic band (hyperacuity thumb C style)
def thumb_c():
    img = D.card_base(1.0).copy()
    ov = D.overlay(lambda d: (outline_text(d, (90, 150), "YOU SEE", H1, (255, 255, 255)),
                              outline_text(d, (90, 315), "SHARPER", H1, GOLDT),
                              outline_text(d, (90, 480), "THAN YOUR", H1, (255, 255, 255)),
                              outline_text(d, (90, 645), "EYES ALLOW", H1, (255, 255, 255)),
                              d.text((96, 880), "your brain hides your eye's blur", font=SUB, fill=(210, 222, 238))))
    save(D.comp(img, ov), "thumb_C")


thumb_a(); thumb_b(); thumb_c()
print("ok")
