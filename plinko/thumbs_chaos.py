"""Thumbnails for the chaos cut (1280x720), from the film's own frames without their captions.

    python3 thumbs_chaos.py   -> ../videos/plinko/thumbnails/plinko_chaos_thumb_A.png, _B.png, _C.png

A and B: the hook's last frame (a $10,000 drop and its atom-close replay in $500).
C: the warm-board pair (the same release point, the board 0.00006 C warmer: $10,000 and $0).
"""
import cairo
import numpy as np

import composite3d as C3
import composite_chaos as CC
import draw as D
import thumbs3d as T


def shot_frame(name, t, overlay):
    """The composited frame of a shot at time t, with only its glowing trails (captions drawn off-screen)."""
    sh = CC.shot_by_name(name)
    fr = int(round(t * CC.FPS)) + 1
    img, crop = C3.base_frame(sh, fr, t)
    img = img * C3.vignette()
    cm = C3.cam(sh["cam"])
    gsurf, gctx = C3.glow_layer()
    overlay(cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, CC.OW, CC.OH)), gctx, t, sh, cm, crop)
    return C3.add_glow(img, gsurf), cm, crop


def labels(cm, crop, win, rep):
    """Prize pills over the two landing slots, stacked when the slots are neighbours."""
    def draw(ctx):
        kw, kr = int(round(win[-1, 1] / CC.W)), int(round(rep[-1, 1] / CC.W))
        adjacent = abs(kw - kr) == 1
        CC.board_label(ctx, cm, crop, kw, "$10,000", True, 1.0, -150 if adjacent else -64, 46)
        CC.board_label(ctx, cm, crop, kr, D.SLOT_LABELS[kr - 1], False, 1.0, -64, 46)
    return draw


if __name__ == "__main__":
    img, cm, crop = shot_frame("hook", 16.0, CC.ov_hook)
    pills = labels(cm, crop, CC.HP["hook_win"], CC.HP["hook_replay"])
    T.finish(img.copy(), [("Same release point.", 80, 360, 84, D.INK, "black"),
                          ("Different", 80, 540, 160, D.GOLD, "black"), ("prize.", 80, 700, 160, D.GOLD, "black")],
             "plinko_chaos_thumb_A", extra=pills)
    T.finish(img.copy(), [("Matched to", 80, 360, 120, D.INK, "black"), ("an atom.", 80, 490, 120, D.INK, "black"),
                          ("Still missed.", 80, 640, 120, D.GOLD, "black")], "plinko_chaos_thumb_B", extra=pills)
    img, cm, crop = shot_frame("warm", 15.5, CC.ov_warm)
    pills = labels(cm, crop, CC.HP["warm_win"], CC.HP["warm_replay"])
    T.finish(img.copy(), [("0.00006 °C", 80, 360, 128, D.GOLD, "black"), ("warmer:", 80, 490, 112, D.INK, "black"),
                          ("$10,000", 80, 650, 128, D.INK, "black"), ("became $0.", 80, 790, 128, D.INK, "black")],
             "plinko_chaos_thumb_C", extra=pills)
