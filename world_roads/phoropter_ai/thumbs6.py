"""Thumbnails for the fourth clinic video (env6: the Borish method, a realistic patient), 1280x720.

    python thumbs6.py      -> out/thumb6_A.png, thumb6_C.png, thumb6_D.png
"""
import os, json
from PIL import Image, ImageDraw
import ui
import clinic_ui as CU
import clinic_ui5 as CU5
import sequence5 as SQ
from thumbs import shadow_text
from thumbs3 import canvas, W, H

HERE = ui.HERE
TR = os.path.join(HERE, "data", "traces")
TB_COL, AI_COL = (0xE7, 0x66, 0x24), (0x33, 0x8C, 0xED)


def thumb_A(R):
    """Before and after practice in the new clinic: astigmatic eyes with the axis within 10 degrees."""
    im, d = canvas()
    shadow_text(d, (640, 20), "IT FOUND THE CROSS-CYLINDER", 72, ui.INK)
    b, a = R["day1"]["axis"]["within_10"], R["ai"]["axis"]["within_10"]
    for x, lab, v, col in ((330, "BEFORE", b, CU.OUT_COL["review"]), (950, "AFTER PRACTICE", a, CU.OUT_COL["happy"])):
        d.rounded_rectangle([x - 260, 170, x + 260, 640], 30, fill=(8, 10, 16, 235), outline=col + (255,), width=8)
        shadow_text(d, (x, 196), lab, 52, col)
        shadow_text(d, (x, 300), f"{100 * v:.0f}%", 170, ui.INK)
        d.text((x, 560), "astigmatism axis within 10°", font=ui.ST.font(32, "SemiBold"), fill=ui.INK2, anchor="mm")
    d.polygon([(600, 405), (660, 405), (660, 375), (712, 420), (660, 465), (660, 435), (600, 435)], fill=ui.GOLD)
    shadow_text(d, (640, 660), "no smiles, rough answers, a few remarks", 34, ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb6_A.png"))


def thumb_C(R):
    """AI vs the Borish method: minutes in the chair and happy patients."""
    im, d = canvas()
    ai, tb = R["ai"]["summary"], R["textbook"]["summary"]
    shadow_text(d, (640, 26), "AI vs THE BORISH METHOD", 74, ui.INK)
    for x, lab, s, col in ((330, "BORISH", tb, TB_COL), (950, "AI", ai, AI_COL)):
        d.rounded_rectangle([x - 260, 170, x + 260, 680], 30, fill=(8, 10, 16, 235), outline=col + (255,), width=8)
        shadow_text(d, (x, 192), lab, 60, col)
        m_, s_ = divmod(int(round(s["minutes"] * 60)), 60)
        shadow_text(d, (x, 290), f"{m_}:{s_:02d}", 120, ui.INK)
        d.text((x, 450), "minutes in the chair", font=ui.ST.font(30, "SemiBold"), fill=ui.INK2, anchor="mm")
        shadow_text(d, (x, 500), f"{100 * s['happy']:.0f}%", 84, CU.OUT_COL["happy"])
        d.text((x, 630), "happy patients", font=ui.ST.font(30, "SemiBold"), fill=ui.INK2, anchor="mm")
    im.save(os.path.join(ui.OUT, "thumb6_C.png"))


def thumb_D(R, title="AN AI WORKED OUT ITS OWN EYE EXAM"):
    """The Borish method's steps and the AI's for the same patient: coloured bars, one per step, same scale."""
    im, d = canvas()
    shadow_text(d, (640, 14), title, 58, ui.INK)
    cols = []
    for x, name, col, tr_name in ((70, "BORISH", TB_COL, "m_borish"), (690, "AI", AI_COL, "m_ai")):
        tr = json.load(open(os.path.join(TR, f"{tr_name}.json")))
        cols.append((x, name, col, tr, SQ.visits(SQ.steps_of(tr))))
    rh = min(30, min((H - 16 - 180 - 32 * len(v)) / sum(len(s) for _, s in v) for *_, v in cols))
    for x, name, col, tr, vis in cols:
        m_, s_ = divmod(int(round(tr["steps"][-1]["secs"])), 60)
        shadow_text(d, (x + 260, 100), f"{name}  {m_}:{s_:02d}", 50, col)
        y = 180
        for stg, sts in vis:
            lab = ("RIGHT EYE", "LEFT EYE", "BOTH EYES")[stg]
            d.text((x, y + 2), lab, font=ui.ST.font(17, "Bold"), fill=ui.GOLD if stg < 2 else (120, 200, 255))
            y += 22
            for st in sts:
                c = CU5.GROUP_COL[SQ.KIND_GROUP[st["kind"]]]
                d.rounded_rectangle([x, y, x + 520, y + rh - 4], 5, fill=tuple(c) + (255,))
                y += rh
            y += 10
    im.save(os.path.join(ui.OUT, "thumb6_D.png"))


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    R = json.load(open(os.path.join(HERE, "data", "results_m.json")))
    thumb_A(R)
    thumb_C(R)
    thumb_D(R)
    print("thumbnails written")
