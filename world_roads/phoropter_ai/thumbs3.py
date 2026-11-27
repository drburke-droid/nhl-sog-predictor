"""Thumbnails for the clinic video (1280x720).

    python thumbs3.py      -> out/thumb3_A.png, out/thumb3_B.png, out/thumb3_C.png
"""
import os, json
import numpy as np
from PIL import Image, ImageDraw
import ui
import clinic_ui as CU
from thumbs import shadow_text

W, H = 1280, 720
HERE = ui.HERE
OUTC = [CU.OUT_COL["happy"], CU.OUT_COL["annoyed"], CU.OUT_COL["recheck"], CU.OUT_COL["review"], (120, 60, 70)]


def canvas():
    im = Image.new("RGB", (W, H), ui.BG)
    return im, ImageDraw.Draw(im, "RGBA")


def thumb_A():
    """The real phoropter, colour-coded, and the premise."""
    im, d = canvas()
    st = dict(S=[0.0, 0.0], C=[0.0, 0.0], ax=[180, 180], phase=0)
    CU.draw_phoropter(im, 250, 0, 0.53, None, hot=("jcc", 0))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 470, W, H], fill=ui.BG + (210,))
    shadow_text(d, (640, 488), "DAY ONE AT", 74, ui.INK)
    shadow_text(d, (640, 576), "AI OPTOMETRY", 104, ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb3_A.png"))


def grid(d, x0, y0, counts, n=300, cols=20, cell=16):
    c = np.array(counts, float)
    c = np.round(c / c.sum() * n).astype(int)
    c[0] += n - c.sum()
    colors = np.repeat(np.arange(5), c)
    for j in range(n):
        r_, c_ = divmod(j, cols)
        d.rectangle([x0 + c_ * cell, y0 + r_ * cell, x0 + c_ * cell + cell - 3, y0 + r_ * cell + cell - 3],
                    fill=OUTC[colors[j]])


def thumb_B(R, first=None, tag="thumb3_B"):
    """Day one vs graduation: 300 patients each, one square per patient."""
    im, d = canvas()
    first = first or R["stages"][0]["sampled_outcomes"]
    last = R["ai"]["outcomes"]
    grid(d, 80, 196, first, cols=15, cell=25)
    grid(d, 825, 196, last, cols=15, cell=25)
    shadow_text(d, (640, 18), "AN AI OPENED AN EYE CLINIC", 66, ui.INK)
    shadow_text(d, (267, 118), "DAY ONE", 52, CU.OUT_COL["review"])
    shadow_text(d, (1012, 124), f"{R['meta']['exams'] / 1e6:.1f} MILLION PATIENTS LATER", 34, CU.OUT_COL["happy"])
    d.polygon([(500, 420), (720, 420), (720, 385), (790, 445), (720, 505), (720, 470), (500, 470)], fill=ui.GOLD)
    im.save(os.path.join(ui.OUT, f"{tag}.png"))


def thumb_C(R, tag="thumb3_C"):
    """AI vs the textbook: minutes and happy patients."""
    im, d = canvas()
    ai, tb = R["ai"]["summary"], R["textbook"]["summary"]
    shadow_text(d, (640, 26), "AI vs THE TEXTBOOK", 90, ui.INK)
    for x, lab, s, col in ((330, "TEXTBOOK", tb, (0xE7, 0x66, 0x24)), (950, "AI", ai, (0x33, 0x8C, 0xED))):
        d.rounded_rectangle([x - 260, 170, x + 260, 680], 30, fill=(8, 10, 16, 235), outline=col + (255,), width=8)
        shadow_text(d, (x, 192), lab, 60, col)
        m_, s_ = divmod(int(round(s["minutes"] * 60)), 60)
        shadow_text(d, (x, 290), f"{m_}:{s_:02d}", 120, ui.INK)
        d.text((x, 450), "minutes in the chair", font=ui.ST.font(30, "SemiBold"), fill=ui.INK2, anchor="mm")
        shadow_text(d, (x, 500), f"{100 * s['happy']:.0f}%", 84, CU.OUT_COL["happy"])
        d.text((x, 630), "happy patients", font=ui.ST.font(30, "SemiBold"), fill=ui.INK2, anchor="mm")
    im.save(os.path.join(ui.OUT, f"{tag}.png"))


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    thumb_A()
    p = os.path.join(HERE, "data", "results_f.json")
    if os.path.exists(p):
        R = json.load(open(p))
        thumb_B(R)
        thumb_C(R)
        g = os.path.join(HERE, "data", "results_g.json")
        if os.path.exists(g):
            G = json.load(open(g))
            thumb_B(G, first=R["stages"][0]["sampled_outcomes"], tag="thumb4_B")
            thumb_C(G, tag="thumb4_C")
    print("thumbnails written")
