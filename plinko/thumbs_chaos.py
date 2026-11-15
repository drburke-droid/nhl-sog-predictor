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


SRC = "../videos/plinko/thumbnails/sources/The_Price_Is_Right_Plinko_Home_Game_960px.jpg"
CUTOUT = "../videos/plinko/thumbnails/sources/The_Price_Is_Right_Plinko_Home_Game_cutout.png"


def make_cutout():
    """The board alone, background transparent: GrabCut from a box around the board, largest piece, holes
    filled, smoothed, pulled in 1 px off the background fringe, feathered. Saved as RGBA (CC BY-SA 4.0, like
    its source)."""
    im = cv2.imread(SRC)
    mask = np.zeros(im.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(1)
    cv2.grabCut(im, mask, (105, 105, 770, 1080), bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    def largest(m):
        _, lab, st, _ = cv2.connectedComponentsWithStats(m)
        return np.where(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    fg = largest(fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fg = cv2.drawContours(np.zeros_like(fg), cnts, -1, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg = largest(cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k))
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = smooth_bottom(im, fg)
    alpha = cv2.GaussianBlur(fg, (0, 0), 1.2)
    x, y, w, h = cv2.boundingRect(fg)
    out = np.dstack([im, alpha])[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6]
    cv2.imwrite(CUTOUT, out)
    return out


def smooth_bottom(im, fg, rim=7, corner=41):
    """Replace the base (feet and the uneven rim under the prize panel) with one smooth curve: fit a parabola to
    the bottom of the coloured panel, cut everything more than `rim` px below it, then round the corners where
    the curve meets the sides (an opening, applied to the bottom of the board only)."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    ys_, xs_ = np.nonzero(fg)
    y_bot = ys_.max()
    band = slice(y_bot - 200, y_bot + 1)
    xs, ys = [], []
    for x in range(fg.shape[1]):
        col = np.nonzero((hsv[band, x, 1] > 90) & (fg[band, x] > 0))[0]
        if len(col):
            xs.append(x)
            ys.append(band.start + col.max())
    xs, ys = np.array(xs, float), np.array(ys, float)
    lo, hi = np.percentile(xs, 12), np.percentile(xs, 88)               # the panel's straight run, not its ends
    sel = (xs >= lo) & (xs <= hi)
    for _ in range(3):                                                   # robust: drop columns far off the fit
        c = np.polyfit(xs[sel], ys[sel], 2)
        r = ys - np.polyval(c, xs)
        sel &= np.abs(r) < 6
    curve = np.polyval(c, np.arange(fg.shape[1])) + rim
    yy = np.arange(fg.shape[0])[:, None]
    cut = np.where(yy <= curve[None, :], fg, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (corner, corner))
    opened = cv2.morphologyEx(cut, cv2.MORPH_OPEN, k)
    low = yy >= y_bot - 160
    return np.where(low, opened, cut).astype(np.uint8)


def thumb_d():
    """A photo of a real Plinko board, cut out of its background, beside the film's line."""
    OW, OH = CC.OW, CC.OH
    cut = cv2.imread(CUTOUT, cv2.IMREAD_UNCHANGED)
    if cut is None:
        cut = make_cutout()
    H = 1000
    cut = cv2.resize(cut, (int(cut.shape[1] * H / cut.shape[0]), H), interpolation=cv2.INTER_AREA)
    rgb, a = cut[..., :3] / 255.0, cut[..., 3:4] / 255.0
    yy, xx = np.mgrid[0:OH, 0:OW]
    x0, y0 = OW - cut.shape[1] - 110, (OH - H) // 2 + 10
    cx, cy = x0 + cut.shape[1] / 2, OH / 2
    glow = np.exp(-(((xx - cx) / 620.0) ** 2 + ((yy - cy) / 520.0) ** 2))[..., None]
    img = np.empty((OH, OW, 3), np.float32)
    img[:] = (0.04, 0.045, 0.06)                                     # BGR, as cv2 and finish() use
    img += glow * np.array([0.10, 0.12, 0.20])                       # warm light behind the board
    sh = np.zeros((OH, OW), np.float32)                              # soft shadow, down and to the left
    sh[y0 + 18:y0 + 18 + H, x0 - 22:x0 - 22 + cut.shape[1]] = a[..., 0]
    sh = cv2.GaussianBlur(sh, (0, 0), 22)[..., None]
    img *= 1 - 0.65 * sh
    reg = img[y0:y0 + H, x0:x0 + cut.shape[1]]
    img[y0:y0 + H, x0:x0 + cut.shape[1]] = rgb * a + reg * (1 - a)
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
