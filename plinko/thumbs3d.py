"""Thumbnails for the Blender edition, built from real composited frames (1280x720).

    python3 thumbs3d.py   -> ../videos/plinko/thumbnails/plinko_blender_thumb_A.png, _B.png
"""
import os

import cairo
import cv2
import numpy as np

import composite3d as C
import draw as D

OUTDIR = "../videos/plinko/thumbnails"


def frame(shot, t):
    s = C.shot_by_name(shot)
    return C.render(s, int(round(t * C.FPS)) + 1)


def finish(img, texts, name, side="left", extra=None):
    a = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    buf = np.zeros((C.OH, C.OW, 4), np.uint8)
    buf[..., :3] = a
    buf[..., 3] = 255
    s = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, C.OW, C.OH)
    ctx = cairo.Context(s)
    x0, x1 = (0, 1000) if side == "left" else (C.OW, 1170)
    g = cairo.LinearGradient(x0, 0, x1, 0)
    g.add_color_stop_rgba(0, 0.02, 0.025, 0.035, 0.92)
    g.add_color_stop_rgba(1, 0.02, 0.025, 0.035, 0.0)
    ctx.rectangle(min(x0, x1), 0, abs(x1 - x0), C.OH)
    ctx.set_source(g)
    ctx.fill()
    if extra:
        extra(ctx)
    for (s_, x, y, size, col, wt) in texts:
        D.text(ctx, s_, x, y, size, col, 1.0, wt)
    s.flush()
    out = np.ndarray((C.OH, C.OW, 4), np.uint8, s.get_data())[..., :3]
    os.makedirs(OUTDIR, exist_ok=True)
    cv2.imwrite(f"{OUTDIR}/{name}.png", cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA))
    print("wrote", name)


if __name__ == "__main__":
    # A: the twin chips and their trails, split into $100 and $10,000 (no captions from the film)
    sh, t = C.shot_by_name("twins"), 13.5
    fr = int(round(t * C.FPS)) + 1
    img, crop = C.base_frame(sh, fr, t)
    img = img * C.vignette()
    cm = C.cam(sh["cam"])
    gsurf, gctx = C.glow_layer()
    C.ov_twins(cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, C.OW, C.OH)), gctx, t, sh, cm, crop)
    img = C.add_glow(img, gsurf)

    def pills(ctx):
        for rec in (C.PATHS["twin_a"], C.PATHS["twin_b"]):
            k = int(round(rec[-1, 1] / C.W))
            x, y = cm.px(C.board_world(k * C.W, C.B.last_row_y() + 0.2 * C.W, 0.05), crop)
            D.pill(ctx, D.SLOT_LABELS[k - 1], x, y - 60, 44, D.GOLD if k == 5 else D.INK)
    finish(img, [("0.001 mm", 80, 380, 170, D.GOLD, "black"), ("apart.", 80, 550, 170, D.INK, "black")],
           "plinko_blender_thumb_A", extra=pills)
    # B: the held chip, no captions, and the question beside it
    sh = C.shot_by_name("question")
    img, _ = C.base_frame(sh, 121, 4.0)
    img = img * C.vignette() * 0.9
    finish(img, [("Could a robot", 1250, 300, 90, D.INK, "black"), ("win Plinko", 1250, 405, 90, D.INK, "black"),
                 ("every time?", 1250, 510, 90, D.GOLD, "black")], "plinko_blender_thumb_B", side="right")
