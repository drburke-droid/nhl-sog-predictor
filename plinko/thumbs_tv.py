"""Thumbnails on a broadcast frame of the TV board (1280x720): our simulated chip paths traced onto its pegs.

    python3 thumbs_tv.py FRAME.png   -> ../videos/plinko/thumbnails/tv/plinko_chaos_thumb_<text>4.png

FRAME is a screenshot of the 1980s Plinko set (545x316, board left of centre, stairs at right). It is not kept
in this repository, and neither are the thumbnails made from it (videos/plinko/thumbnails/tv/ is git-ignored).

The board face is mapped by a homography from board coordinates (board.py, metres) to the frame, fitted to the
frame's bottom row of pegs (8 pins read off a high-pass view), its peg lattice and the face's side edges. The
sim paths are drawn through it, carried on down into the frame's deeper slots, with our rendered chips warped
onto the board plane where they come to rest.
Text  A, B, C  as in thumbs_chaos.py;  D  a short version: "Same release point. $10,000 / $0."
Paths A and B: the opening drop and its atom-close replay ($10,000 and $500); C and D: a pair from the
warm-board experiment ($10,000 and $0), with a ring where the paths part.
"""
import os
import sys

import cairo
import cv2
import numpy as np

import board as B
import composite3d as C3
import composite_chaos as CC
import draw as D
import thumbs_chaos as TC

P, RS = B.SLOT_PITCH, B.ROW_SPACING
OW, OH = C3.OW, C3.OH
OUTDIR = "../videos/plinko/thumbnails/tv"
# board coordinates -> frame pixels: bottom-row pegs at x = 1.5 and 8.5 pitches, top-row pegs at the same x
SRC = np.float32([[1.5 * P, 12 * RS], [8.5 * P, 12 * RS], [1.5 * P, 0], [8.5 * P, 0]])
DST = np.float32([[174.0, 233.0], [273.7, 244.3], [229.7, 84.2], [327.5, 82.5]])
CROP = (6, 3, 545, 306)             # drops the pillarbox edge at left and the player's scrubber dot at the bottom
REST = 12 * RS + 2.0 * P            # the frame's slots are deeper than the modern board's: rest chips lower


def base(frame):
    x0, y0, x1, y1 = CROP
    im = cv2.imread(frame)[y0:y1, x0:x1]
    im = cv2.fastNlMeansDenoisingColored(im, None, 3, 3, 5, 15)
    im = cv2.resize(im, (OW, OH), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
    soft = cv2.GaussianBlur(im, (0, 0), 3)
    im = np.clip(im + 0.6 * (im - soft), 0, 1)                       # gentle unsharp mask after the 3.6x upscale
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.12, 0, 1)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sx, sy = OW / (x1 - x0), OH / (y1 - y0)
    A = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    H = A @ cv2.getPerspectiveTransform(SRC, DST)                    # board metres -> output pixels
    return im, H


def to_px_fn(H):
    return lambda x, y: cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0].astype(np.float64)


def extend(rec):
    """The sim path, then straight down the slot centre to the frame's deeper resting point."""
    k = int(round(rec[-1, 1] / P))
    xy = rec[:, 1:3].copy()
    tail = np.linspace(0, 1, 12)[1:, None] * (np.array([k * P, REST]) - xy[-1]) + xy[-1]
    return np.vstack([xy, tail]), np.array([k * P, REST])


def paste_chip(img, name, H, c, ang):
    ch = cv2.imread(TC.CHIPS[name], cv2.IMREAD_UNCHANGED)
    ys, xs = np.nonzero(ch[..., 3] > 10)
    ch = ch[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    n = ch.shape[0]
    r = B.CHIP_DIAMETER / 2
    ca, sa = np.cos(ang), np.sin(ang)
    # sprite pixels -> board metres (rotated, centred on c) -> output pixels
    S = np.array([[2 * r / n * ca, -2 * r / n * sa, c[0] - r * ca + r * sa],
                  [2 * r / n * sa, 2 * r / n * ca, c[1] - r * sa - r * ca], [0, 0, 1]])
    w = cv2.warpPerspective(ch, H @ S, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = w[..., 3:4]
    return w[..., :3] * a + img * (1 - a)


def trails(img, H, recs):
    """Solid paths with a dark edge and a soft shadow: additive glow vanishes on the white board."""
    shadow = np.zeros(img.shape[:2], np.float32)
    lines = []
    for xy, col in recs:
        pts = np.round(cv2.perspectiveTransform(np.float32(xy)[None], H)[0] * 4).astype(np.int32)   # shift=2
        cv2.polylines(shadow, [pts], False, 1.0, 26, cv2.LINE_AA, 2)
        lines.append((pts, col))
    img = img * (1 - 0.45 * cv2.GaussianBlur(shadow, (0, 0), 7)[..., None])
    for pts, col in lines:
        bgr = np.array(col[::-1], np.float32)
        for wd, c in ((15, bgr * 0.35), (10, bgr), (3, bgr * 0.35 + 0.65)):
            layer = np.zeros(img.shape[:2], np.float32)
            cv2.polylines(layer, [pts], False, 1.0, wd, cv2.LINE_AA, 2)
            img = img * (1 - layer[..., None]) + c * layer[..., None]
    return np.clip(img, 0, 1)


def labels(H, win, rep, wc, rc, split_t, size=56):
    """A ring where the paths part, a dot at the shared release point, and prize pills set off the board
    (left of it for the replay, right of it for the winner) with leader lines to the chips."""
    to_px = to_px_fn(H)
    kw, kr = int(round(win[-1, 1] / P)), int(round(rep[-1, 1] / P))
    r_px = float(np.hypot(*(to_px(wc[0] + B.CHIP_DIAMETER / 2, wc[1]) - to_px(*wc))))

    def draw(ctx):
        q = to_px(*rep[int(round(split_t * 60)), 1:3])
        for wd, col in ((9, (0.03, 0.03, 0.05, 0.85)), (4, (1, 1, 1, 1))):
            ctx.new_path()
            ctx.arc(q[0], q[1], r_px * 2.4, 0, 2 * np.pi)
            ctx.set_source_rgba(*col)
            ctx.set_line_width(wd)
            ctx.stroke()
        s0 = to_px(*win[0, 1:3])
        for rad, col in ((11, (0.03, 0.03, 0.05, 1)), (7, (1, 1, 1, 1))):
            ctx.new_path()
            ctx.arc(s0[0], s0[1], rad, 0, 2 * np.pi)
            ctx.set_source_rgba(*col)
            ctx.fill()
        for c, k, gold, anchor in ((rc, kr, False, (330, 905)), (wc, kw, True, (1110, 905))):
            chip = to_px(*c)
            lab = "$10,000" if gold else D.SLOT_LABELS[k - 1]
            w = D.text_width(ctx, lab, size, "black")
            x = anchor[0] - w / 2 if not gold else anchor[0] + w / 2
            side = 1 if not gold else -1
            ctx.new_path()
            ctx.move_to(x + side * (w / 2 + size * 0.2), anchor[1])
            ctx.line_to(chip[0] - side * r_px * 1.1, chip[1])
            for wd, col in ((7, (0.03, 0.03, 0.05, 0.85)), (3.5, (1, 1, 1, 0.95))):
                ctx.set_source_rgba(*col)
                ctx.set_line_width(wd)
                ctx.stroke_preserve()
            ctx.new_path()
            D.pill(ctx, lab, x, anchor[1], size, D.GOLD if gold else D.INK)
    return draw


def scrim(img, x_from=1190, x_full=1340, dark=0.82):
    """Darken the stairs at right so the text reads; the board stays bright."""
    xx = np.arange(OW, dtype=np.float32)
    k = np.clip((xx - x_from) / (x_full - x_from), 0, 1)[None, :, None]
    k = k * k * (3 - 2 * k)
    return img * (1 - dark * k) + 0.012 * dark * k


def picture(frame, win, rep, split_t):
    img, H = base(frame)
    wxy, wc = extend(win)
    rxy, rc = extend(rep)
    img = trails(img, H, [(rxy[:-4], D.S1), (wxy[:-4], D.GOLD)])
    img = paste_chip(img, "blue", H, rc, rep[-1, 3])
    img = paste_chip(img, "red", H, wc, win[-1, 3])
    img = scrim(img)
    return img, labels(H, win, rep, wc, rc, split_t)


TEXTS = dict(TC.TEXTS)
TEXTS["D"] = [("Same release point.", 84, D.INK), ("$10,000", 190, D.GOLD), ("or $0.", 190, D.INK)]


def save(img, texts, name, extra):
    a = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    buf = np.zeros((OH, OW, 4), np.uint8)
    buf[..., :3] = a
    buf[..., 3] = 255
    s = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, OW, OH)
    ctx = cairo.Context(s)
    if extra:
        extra(ctx)
    for (s_, x, y, size, col, wt) in texts:
        D.text(ctx, s_, x, y, size, col, 1.0, wt)
    s.flush()
    out = np.ndarray((OH, OW, 4), np.uint8, s.get_data())[..., :3]
    os.makedirs(OUTDIR, exist_ok=True)
    cv2.imwrite(f"{OUTDIR}/{name}.png", cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA))
    print("wrote", name)


if __name__ == "__main__":
    frame = sys.argv[1]
    wt = np.load("results/chaos2_warm_thumb.npz")                  # parts earlier and wider (chaos2.py)
    pics = {"hook": picture(frame, CC.HP["hook_win"], CC.HP["hook_replay"], CC.FILM["hook_split_t"]),
            "warm": picture(frame, wt["win"], wt["replay"], float(wt["split_t"]))}
    for tk, lines in TEXTS.items():
        img, pills = pics["warm" if tk in "CD" else "hook"]
        save(img.copy(), TC.layout(lines, OW - 1330 - 50, x=1330), f"plinko_chaos_thumb_{tk}4", pills)
