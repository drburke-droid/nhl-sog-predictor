"""Thumbnails for the chaos cut (1280x720), from the hook's last frame without its captions.

    python3 thumbs_chaos.py   -> ../videos/plinko/thumbnails/plinko_chaos_thumb_A.png, _B.png
"""
import cairo
import numpy as np

import composite3d as C3
import composite_chaos as CC
import draw as D
import thumbs3d as T


def hook_frame(t=15.5):
    sh = CC.shot_by_name("hook")
    fr = int(round(t * CC.FPS)) + 1
    img, crop = C3.base_frame(sh, fr, t)
    img = img * C3.vignette()
    cm = C3.cam(sh["cam"])
    gsurf, gctx = C3.glow_layer()
    CC.ov_hook(cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, CC.OW, CC.OH)), gctx, t, sh, cm, crop)
    img = C3.add_glow(img, gsurf)
    rep = CC.HP["hook_replay"]

    def pills(ctx):
        CC.board_label(ctx, cm, crop, 5, "$10,000", True, 1.0, -64, 46)
        k = int(round(rep[-1, 1] / CC.W))
        CC.board_label(ctx, cm, crop, k, D.SLOT_LABELS[k - 1], False, 1.0, -64, 46)
    return img, pills


if __name__ == "__main__":
    img, pills = hook_frame()
    T.finish(img.copy(), [("Same drop.", 80, 400, 150, D.INK, "black"), ("Different", 80, 560, 150, D.GOLD, "black"),
                          ("prize.", 80, 710, 150, D.GOLD, "black")], "plinko_chaos_thumb_A", extra=pills)
    T.finish(img.copy(), [("Matched to", 80, 360, 120, D.INK, "black"), ("an atom.", 80, 490, 120, D.INK, "black"),
                          ("Still missed.", 80, 640, 120, D.GOLD, "black")], "plinko_chaos_thumb_B", extra=pills)
