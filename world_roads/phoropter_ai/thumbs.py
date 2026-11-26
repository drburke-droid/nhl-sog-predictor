"""YouTube thumbnails (1280x720).

    python thumbs.py      -> out/thumb_A.png, out/thumb_B.png, out/thumb_C.png
"""
import os, json
import numpy as np
from PIL import Image, ImageDraw
import ui
import phoropter_art as PA
from optics import to_pv

W, H = 1280, 720
HERE = ui.HERE


def canvas():
    im = Image.new("RGB", (W, H), ui.BG)
    return im, ImageDraw.Draw(im, "RGBA")


def shadow_text(d, xy, s, size, col, anchor="ma", weight="Black"):
    f = ui.ST.font(size, weight) if os.path.exists(os.path.join(ui.ST.FD, f"Inter-{weight}.otf")) else ui.ST.font(size, "Bold")
    x, y = xy
    for dx, dy in ((4, 4), (3, 3)):
        d.text((x + dx, y + dy), s, font=f, fill=(0, 0, 0, 200), anchor=anchor)
    d.text((x, y), s, font=f, fill=col, anchor=anchor)


def thumb_A():
    """The phoropter, a cross-cylinder flipped in, and the premise."""
    im, d = canvas()
    PA.draw(im, 640, 262, 0.96, [(-2.25, -1.00, 170), (-2.50, -0.75, 10)], occluded=1, cur=0, jcc=(0, 215))
    d = ImageDraw.Draw(im, "RGBA")
    shadow_text(d, (640, 512), "I GAVE AN AI", 84, ui.INK)
    shadow_text(d, (640, 598), "A PHOROPTER", 104, ui.GOLD)
    im.save(os.path.join(ui.OUT, "thumb_A.png"))


def chart_tile(Me, J0, J45, pupil=4.6, w=420, h=470):
    img = ui._view(round(Me, 3), round(J0, 3), round(J45, 3), pupil, 6)
    img = Image.fromarray(img).crop((40, 150, 460, 620)).resize((w, h), Image.LANCZOS)
    return img


def thumb_B(R):
    """AI vs my routine: the score."""
    im, d = canvas()
    ai = R["ai"]["summary"]["within_025"]
    me = R["mine"]["summary"]["within_025"]
    blurry = chart_tile(-0.75, 0.25, -0.1)
    sharp = chart_tile(0.0, 0.0, 0.0)
    im.paste(blurry, (70, 150))
    im.paste(sharp, (790, 150))
    d = ImageDraw.Draw(im, "RGBA")
    shadow_text(d, (640, 26), "AI vs MY EYE EXAM", 86, ui.INK)
    for x, lab, v, col in ((280, "ME", me, (0x00, 0xA8, 0x88)), (1000, "AI", ai, (0x33, 0x8C, 0xED))):
        d.rounded_rectangle([x - 170, 470, x + 170, 690], 26, fill=(8, 10, 16, 235), outline=col + (255,), width=6)
        shadow_text(d, (x, 486), lab, 64, col)
        shadow_text(d, (x, 560), f"{100 * v:.0f}%", 104, ui.INK)
    d.text((640, 600), "of eyes within 0.25 D", font=ui.ST.font(30, "SemiBold"), fill=ui.INK2, anchor="mm")
    d.text((640, 640), "5,000 simulated patients", font=ui.ST.font(26, "Medium"), fill=ui.MUTED, anchor="mm")
    im.save(os.path.join(ui.OUT, "thumb_B.png"))


def thumb_C():
    """It found the axis by itself: the phoropter with the cylinder axis line, and the before/after chart."""
    im, d = canvas()
    E = to_pv(-2.25, -1.00, 170)
    L = to_pv(-2.50, -0.75, 4)
    before = chart_tile(E[0] - L[0], E[1] - L[1], E[2] - L[2], w=380, h=430)
    after = chart_tile(0.0, 0.0, 0.0, w=380, h=430)
    im.paste(before, (60, 230))
    im.paste(after, (840, 230))
    d = ImageDraw.Draw(im, "RGBA")
    d.polygon([(470, 445), (800, 445), (800, 420), (860, 470), (800, 520), (800, 495), (470, 495)], fill=ui.GOLD + (255,))
    shadow_text(d, (640, 30), "THE AI LEARNED", 80, ui.INK)
    shadow_text(d, (640, 118), "THE CROSS-CYLINDER", 84, ui.GOLD)
    d.text((250, 680), "14° off axis", font=ui.ST.font(34, "SemiBold"), fill=ui.INK2, anchor="mm")
    d.text((1030, 680), "after", font=ui.ST.font(34, "SemiBold"), fill=ui.INK2, anchor="mm")
    im.save(os.path.join(ui.OUT, "thumb_C.png"))


if __name__ == "__main__":
    os.makedirs(ui.OUT, exist_ok=True)
    thumb_A()
    thumb_C()
    p = os.path.join(HERE, "data", "results_e.json")
    if os.path.exists(p):
        thumb_B(json.load(open(p)))
    print("thumbnails written")
