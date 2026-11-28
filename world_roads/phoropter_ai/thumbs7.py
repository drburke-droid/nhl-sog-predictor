"""Thumbnails for the standalone eye-exam film (film7.py), 1280x720.

    python thumbs7.py      -> out/thumb7_A.png (the textbook's steps vs the AI's), thumb7_B.png (day one vs after
                              practice, 100 patients each), thumb7_C.png ("1 or 2?" with the green glow)
"""
import os, json
from PIL import Image, ImageDraw
import ui
import clinic_ui as CU
from thumbs import shadow_text
from thumbs3 import canvas, W, H
from steps7 import STEP_COL

HERE = ui.HERE
TB_COL, AI_COL = (0xE7, 0x66, 0x24), (0x33, 0x8C, 0xED)
OUT_COL = dict(CU.OUT_COL, walked=(150, 70, 80))


def chip(d, x, y, lab, s=64):
    d.rounded_rectangle([x, y, x + s, y + s], int(s * 0.22), fill=STEP_COL[lab] + (255,))
    d.text((x + s / 2, y + s / 2 + 2), lab, font=ui.ST.font(int(s * 0.62), "Bold"), fill=ui.BG, anchor="mm")


def thumb_A():
    """The textbook's step order and the AI's, as numbered chips."""
    im, d = canvas()
    shadow_text(d, (640, 18), "AN AI LEARNED THE EYE EXAM", 66, ui.INK)
    rows = [("TEXTBOOK", TB_COL, ["1", "2", "4", "5", "6", "2", "3"]),
            ("AI", AI_COL, ["1", "2", "3", "1", "6", "S", "5", "6"])]
    for j, (lab, col, seq) in enumerate(rows):
        y = 195 + j * 230
        shadow_text(d, (190, y + 26), lab, 56, col)
        for k, s in enumerate(seq):
            chip(d, 380 + k * 104, y, s, 90)
    shadow_text(d, (640, 650), "same tools, its own order", 40, ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb7_A.png"))


def grid(d, x0, y0, per100, cell=34, gap=5):
    cols = []
    for key, n in zip(("happy", "annoyed", "recheck", "review", "walked"), per100):
        cols += [OUT_COL[key]] * n
    cols = (cols + [OUT_COL["walked"]] * 100)[:100]
    for m, c in enumerate(cols):
        x = x0 + (m % 10) * (cell + gap)
        y = y0 + (m // 10) * (cell + gap)
        d.rounded_rectangle([x, y, x + cell, y + cell], 5, fill=tuple(c) + (255,))


def thumb_B():
    """100 patients on day one and after 2.4 million practice patients."""
    st = json.load(open(os.path.join(HERE, "data", "montage7.json")))
    first, last = st[0]["per100"], st[-1]["per100"]
    im, d = canvas()
    shadow_text(d, (640, 14), "2.4 MILLION PATIENTS LATER", 66, ui.INK)
    grid(d, 95, 200, first)
    grid(d, 795, 200, last)
    shadow_text(d, (290, 112), "DAY ONE", 52, OUT_COL["review"])
    shadow_text(d, (990, 112), "AFTER PRACTICE", 52, OUT_COL["happy"])
    shadow_text(d, (290, 612), f"{first[0]} happy, {first[4]} walk out", 34, ui.INK2)
    shadow_text(d, (990, 612), f"{last[0]} happy of 100", 34, ui.INK2)
    d.polygon([(545, 375), (655, 375), (655, 345), (725, 395), (655, 445), (655, 415), (545, 415)], fill=ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb7_B.png"))


def thumb_C():
    """“Which is clearer, 1 or 2?”: two charts, one glowing."""
    import film7
    rows = film7.answer_cases()
    sec = film7.PatientAnswers7()
    r = rows[0]
    imgs, which = sec.chart_pair(r["S"], r["line"])
    im, d = canvas()
    shadow_text(d, (640, 16), "“WHICH IS CLEARER, 1 OR 2?”", 64, ui.INK)
    for j, img in enumerate(imgs):
        crop = Image.fromarray(img).convert("RGB")
        crop = crop.crop((0, int(crop.size[1] * 0.2), crop.size[0], int(crop.size[1] * 0.68)))
        sc = min(420 / crop.size[0], 470 / crop.size[1])
        src = crop.resize((int(crop.size[0] * sc), int(crop.size[1] * sc)), Image.LANCZOS)
        x0, y0 = 150 + j * 560, 150
        if j == 1:
            for k in range(10):
                d.rounded_rectangle([x0 - 8 - 4 * k, y0 - 8 - 4 * k, x0 + src.size[0] + 8 + 4 * k,
                                     y0 + src.size[1] + 8 + 4 * k], 12 + 4 * k,
                                    outline=(110, 230, 140, int(255 * (0.95 - 0.09 * k))), width=4)
        im.paste(src, (x0, y0))
        d = ImageDraw.Draw(im, "RGBA")
        ui.badge(d, x0 + 44, y0 + 44, str(j + 1), ui.GOLD, r=36)
    shadow_text(d, (640, 640), "how an AI learned the textbook's question", 40, ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb7_C.png"))


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    thumb_A()
    thumb_B()
    thumb_C()
    print("thumbnails written")
