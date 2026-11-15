"""Thumbnails for the chaos cut (1280x720), from the film's own frames without their captions.

    python3 thumbs_chaos.py   -> ../videos/plinko/thumbnails/plinko_chaos_thumb_A.png, _B.png, _C.png

A and B: the hook's last frame (a $10,000 drop and its atom-close replay in $500).
C: the warm-board pair (the same release point, the board 0.00006 C warmer: $10,000 and $0).
D: a photo of a Plinko board (the official home game), "The Price Is Right Plinko Home Game" by TaurusEmerald,
   CC BY-SA 4.0 (Wikimedia Commons; see ../videos/plinko/thumbnails/sources/ATTRIBUTION.md). D adapts it, so D is
   CC BY-SA 4.0 too, and the description must carry the attribution.
"""
import cairo
import cv2
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


def thumb_d():
    OW, OH = CC.OW, CC.OH
    photo = cv2.imread("../videos/plinko/thumbnails/sources/The_Price_Is_Right_Plinko_Home_Game_960px.jpg")
    crop = photo[110:1180, 110:870]                       # the board, from its sign to the prize panel
    crop = cv2.resize(crop, (int(crop.shape[1] * OH / crop.shape[0]), OH), interpolation=cv2.INTER_AREA)
    bg = np.zeros((OH, OW, 3), np.float32)
    bg[:] = (10 / 255, 12 / 255, 16 / 255)
    img = bg.copy()
    x0 = OW - crop.shape[1] - 40
    img[:, x0:x0 + crop.shape[1]] = crop / 255.0                    # finish() takes BGR, as cv2 does
    ramp = np.clip((np.arange(OW) - (x0 - 10)) / 140.0, 0, 1)[None, :, None]
    img = img * ramp + bg * (1 - ramp)
    T.finish(img, [("Same release point.", 80, 380, 84, D.INK, "black"), ("Different", 80, 560, 160, D.GOLD, "black"),
                   ("prize.", 80, 720, 160, D.GOLD, "black"), ("Why? Chaos theory.", 84, 850, 64, D.INK, "black")],
             "plinko_chaos_thumb_D")


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
    thumb_d()
