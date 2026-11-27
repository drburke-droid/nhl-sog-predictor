"""Thumbnails for the third clinic video (no mentor, plano start), 1280x720.

    python thumbs5.py      -> out/thumb5_B.png, thumb5_C.png, thumb5_D.png
"""
import os, json
from PIL import Image, ImageDraw
import ui
import clinic_ui as CU
import clinic_ui5 as CU5
import sequence5 as SQ
from thumbs import shadow_text
from thumbs3 import canvas, thumb_B, thumb_C, W, H

HERE = ui.HERE
TR = os.path.join(HERE, "data", "traces")
TB_COL, AI_COL = (0xE7, 0x66, 0x24), (0x33, 0x8C, 0xED)


def thumb_D(R):
    """The textbook's steps and the AI's for the same patient: two columns of coloured bars, one per step."""
    im, d = canvas()
    shadow_text(d, (640, 14), "AN AI WORKED OUT ITS OWN EYE EXAM", 58, ui.INK)
    cols = []
    for x, name, col, tr_name in ((70, "TEXTBOOK", TB_COL, "h_textbook"), (690, "AI", AI_COL, "h_ai")):
        tr = json.load(open(os.path.join(TR, f"{tr_name}.json")))
        cols.append((x, name, col, tr, SQ.visits(SQ.steps_of(tr))))
    # one row height for both columns (same scale), fitted to the longer one
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
    im.save(os.path.join(ui.OUT, "thumb5_D.png"))


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    R = json.load(open(os.path.join(HERE, "data", "results_h.json")))
    thumb_B(R, tag="thumb5_B")
    thumb_C(R, tag="thumb5_C")
    thumb_D(R)
    print("thumbnails written")
