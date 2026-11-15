"""Thumbnails for the chaos cut (1280x720): every pairing of three lines of text with three pictures.

    python3 thumbs_chaos.py   -> ../videos/plinko/thumbnails/plinko_chaos_thumb_<text><picture>.png (9 files)

Text  A  "Same release point. Different prize."
      B  "Match release point to an atom, different result."
      C  "0.00006 °C warmer: $10,000 became $0."
Picture 1  the hook's last frame: a $10,000 drop and its atom-close replay in $500 (prize pills shown, except
           with text C, which names $0)
        2  the warm-board pair: the same release point, the board 0.00006 C warmer, $10,000 and $0, with a ring
           where the paths part
           (picture 3 with text C uses another pair from the same experiment, results/chaos2_warm_thumb.npz, that
           also lands in $0 but parts earlier and swings up to 54 cm apart, so the split is easy to see)
        3  a photo of a real Plinko board (the official home game), cut out of its background (smooth curved
           base), with the film's two paths drawn on its own pegs and our chips in the slots: "The Price Is Right Plinko Home Game" by TaurusEmerald, CC BY-SA 4.0
           (Wikimedia Commons; see
           ../videos/plinko/thumbnails/sources/ATTRIBUTION.md). The *3 thumbnails adapt it, so they are CC BY-SA 4.0
           too, and a description that uses one must carry the attribution.
Text is sized down where needed so it never reaches the board.
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


def pill_layer(to_px, win, rep, size, split_t=None, dy=-64):
    """Prize pills for the two landing slots, and (when split_t is given) a ring where the two paths part. Pills for
    neighbouring slots sit side by side, each pushed out away from the other, with a leader line to its chip."""
    W_ = CC.W
    kw, kr = int(round(win[-1, 1] / W_)), int(round(rep[-1, 1] / W_))
    r_px = float(np.hypot(*(to_px(W_ + C3.B.CHIP_DIAMETER / 2, 1.0) - to_px(W_, 1.0))))

    def draw(ctx):
        if split_t is not None:
            q = to_px(*rep[int(round(split_t * 60)), 1:3])
            for rad, wd, al in ((r_px * 1.9, 9, 0.25), (r_px * 1.9, 3.5, 1.0)):
                ctx.new_path()
                ctx.arc(q[0], q[1], rad, 0, 2 * np.pi)
                ctx.set_source_rgba(1, 1, 1, al)
                ctx.set_line_width(wd)
                ctx.stroke()
        y_lab = C3.B.last_row_y() + 0.2 * W_
        if abs(kw - kr) == 1:
            d = 1 if kw > kr else -1
            for k, rec, lab, gold, side in ((kw, win, "$10,000", True, d), (kr, rep, D.SLOT_LABELS[kr - 1], False, -d)):
                chip = to_px(*rec[-1, 1:3])
                q = to_px(k * W_ + side * 1.05 * W_, y_lab) + np.array([0.0, dy])
                ctx.new_path()
                ctx.move_to(q[0] - side * 10, q[1] + 8)
                ctx.line_to(chip[0] + side * r_px * 0.7, chip[1] - r_px * 0.7)
                ctx.set_source_rgba(1, 1, 1, 0.9)
                ctx.set_line_width(2.5)
                ctx.stroke()
                D.pill(ctx, lab, q[0], q[1], size, D.GOLD if gold else D.INK)
        else:
            for k, lab, gold in ((kw, "$10,000", True), (kr, D.SLOT_LABELS[kr - 1], False)):
                q = to_px(k * W_, y_lab) + np.array([0.0, dy])
                D.pill(ctx, lab, q[0], q[1], size, D.GOLD if gold else D.INK)
    return draw


def labels(cm, crop, win, rep, split_t=None):
    return pill_layer(lambda x, y: np.array(cm.px(C3.board_world(x, y, 0.03), crop)), win, rep, 46, split_t)


SRC = "../videos/plinko/thumbnails/sources/The_Price_Is_Right_Plinko_Home_Game_960px.jpg"
CHIPS = {"red": "blender/assets/thumb/chip_red.png", "blue": "blender/assets/thumb/chip_blue.png"}


def photo_pegs(im):
    """Peg centres in the photo: small bluish-white dots on the cream board, below the logo."""
    b, g, r = [im[..., i].astype(int) for i in range(3)]
    m = (((b - r) > 6) & (b > 170)).astype(np.uint8) * 255
    _, _, st, cen = cv2.connectedComponentsWithStats(m)
    return np.array([c for c, q in zip(cen[1:], st[1:]) if 8 <= q[cv2.CC_STAT_AREA] <= 160 and 240 < c[1] < 980
                     and 180 < c[0] < 790])


def board_to_photo(im):
    """Homography from board coordinates (m) to photo pixels: the four corner pegs first, then every peg."""
    pts = photo_pegs(im)
    pts = pts[np.argsort(pts[:, 1])]
    rows, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] - cur[-1][1] > 12:
            rows.append(np.array(cur))
            cur = [p]
        else:
            cur.append(p)
    rows.append(np.array(cur))
    while len(rows[0]) < 5:                                   # stray hits on the logo above the first row
        rows.pop(0)
    assert len(rows) == 13, f"expected 13 peg rows, found {len(rows)}"
    W_, Hs = CC.W, C3.B.ROW_SPACING
    r0, r10 = rows[0][np.argsort(rows[0][:, 0])], rows[10][np.argsort(rows[10][:, 0])]
    assert len(r0) == 8 and len(r10) == 8, "the corner rows must be complete"
    src = np.float32([[1.5 * W_, 0], [8.5 * W_, 0], [1.5 * W_, 10 * Hs], [8.5 * W_, 10 * Hs]])
    dst = np.float32([r0[0], r0[-1], r10[0], r10[-1]])
    H0 = cv2.getPerspectiveTransform(src, dst)
    pegs, _ = C3.B.pegs()
    proj = cv2.perspectiveTransform(pegs[None].astype(np.float32), H0)[0]
    bs, ps = [], []
    for q in pts:
        d = np.hypot(*(proj - q).T)
        k = int(np.argmin(d))
        if d[k] < 15:
            bs.append(pegs[k])
            ps.append(q)
    H, _ = cv2.findHomography(np.float32(bs), np.float32(ps), cv2.RANSAC, 4.0)
    err = np.hypot(*(cv2.perspectiveTransform(np.float32(bs)[None], H)[0] - np.float32(ps)).T)
    print(f"photo board: {len(bs)} pegs matched, fit error {np.sqrt((err ** 2).mean()):.1f} px")
    return H


def board_alpha(im):
    """The board alone: GrabCut from a box around it, the largest piece with its holes filled, smoothed, pulled in
    1 px off the background fringe, its base replaced by one smooth curve (no feet), feathered. 0..1, photo size."""
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
    return cv2.GaussianBlur(fg, (0, 0), 1.2).astype(np.float32) / 255.0


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


def photo_frame(win, rep, split_t=None, H_board=1000):
    """The real board from the photo, cut out of its background, with the film's two paths drawn on its own pegs:
    one release point, two glowing trails, the chips in their slots and a faint chip where both were let go; set
    on the film's dark backdrop with a warm light behind it and a soft shadow."""
    OW, OH = CC.OW, CC.OH
    im = cv2.imread(SRC)
    Hm = board_to_photo(im)
    ph = im.astype(np.float32) / 255.0

    def proj(xy):                                                      # board metres -> photo pixels
        return cv2.perspectiveTransform(np.float32(xy)[None], Hm)[0]
    glow = np.zeros_like(ph)
    core = np.zeros_like(ph)
    for rec, col in ((rep, D.S1), (win, D.GOLD)):
        pts = proj(rec[:, 1:3]).astype(np.int32)
        bgr = np.array(col[::-1], np.float32)
        cv2.polylines(glow, [pts], False, tuple(float(v) for v in bgr), 10, cv2.LINE_AA)
        cv2.polylines(core, [pts], False, tuple(float(v) * 0.4 + 0.6 for v in bgr), 3, cv2.LINE_AA)
    ph = np.clip(ph + 0.9 * cv2.GaussianBlur(glow, (0, 0), 9) + 0.85 * core, 0, 1)

    def paste(name, xy_board, ang, alpha=1.0):
        ch = cv2.imread(CHIPS[name], cv2.IMREAD_UNCHANGED)
        ys, xs = np.nonzero(ch[..., 3] > 10)
        ch = ch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        c = proj([xy_board])[0]
        e = proj([[xy_board[0] + C3.B.CHIP_DIAMETER / 2, xy_board[1]]])[0]
        d = 2 * float(np.hypot(*(e - c)))
        ch = cv2.resize(ch, (int(d), int(d)), interpolation=cv2.INTER_AREA)
        M_ = cv2.getRotationMatrix2D((d / 2, d / 2), -np.degrees(ang), 1.0)
        ch = cv2.warpAffine(ch, M_, (int(d), int(d)), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        x, y = int(round(c[0] - d / 2)), int(round(c[1] - d / 2))
        a = ch[..., 3:4] / 255.0 * alpha
        reg = ph[y:y + ch.shape[0], x:x + ch.shape[1]]
        ph[y:y + ch.shape[0], x:x + ch.shape[1]] = ch[..., :3] / 255.0 * a + reg * (1 - a)
    paste("red", win[0, 1:3], win[0, 3], 0.55)                     # where both were let go
    paste("blue", rep[-1, 1:3], rep[-1, 3])
    paste("red", win[-1, 1:3], win[-1, 3])
    # cut the board out and set it on the backdrop
    al = board_alpha(im)
    ys, xs = np.nonzero(al > 0.5)
    bx0, by0, bx1, by1 = xs.min() - 6, ys.min() - 6, xs.max() + 7, ys.max() + 7
    s = H_board / (by1 - by0)
    size = (int(round((bx1 - bx0) * s)), H_board)
    rgb = cv2.resize(ph[by0:by1, bx0:bx1], size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(al[by0:by1, bx0:bx1], size, interpolation=cv2.INTER_AREA)[..., None]
    w_ = size[0]
    x0, y0 = OW - w_ - 110, (OH - H_board) // 2 + 10
    yy, xx = np.mgrid[0:OH, 0:OW]
    lightg = np.exp(-(((xx - (x0 + w_ / 2)) / 620.0) ** 2 + ((yy - OH / 2) / 520.0) ** 2))[..., None]
    img = np.empty((OH, OW, 3), np.float32)
    img[:] = (0.04, 0.045, 0.06)                                     # BGR, as cv2 and finish() use
    img += lightg * np.array([0.10, 0.12, 0.20])                     # warm light behind the board
    sh = np.zeros((OH, OW), np.float32)                              # soft shadow, down and to the left
    sh[y0 + 18:y0 + 18 + H_board, x0 - 22:x0 - 22 + w_] = a[..., 0]
    img *= 1 - 0.65 * cv2.GaussianBlur(sh, (0, 0), 22)[..., None]
    reg = img[y0:y0 + H_board, x0:x0 + w_]
    img[y0:y0 + H_board, x0:x0 + w_] = rgb * a + reg * (1 - a)
    to_canvas = lambda q: (q - np.array([bx0, by0])) * s + np.array([x0, y0])
    pills = pill_layer(lambda x, y: to_canvas(proj([[x, y]])[0]), win, rep, 40, split_t, dy=-50 * s)
    return img, pills, x0


TEXTS = {
    "A": [("Same release point.", 84, D.INK), ("Different", 160, D.GOLD), ("prize.", 160, D.GOLD)],
    "B": [("Match release point", 96, D.INK), ("to an atom,", 120, D.INK), ("different", 150, D.GOLD),
          ("result.", 150, D.GOLD)],
    "C": [("0.00006 \u00b0C", 128, D.GOLD), ("warmer:", 112, D.INK), ("$10,000", 128, D.INK),
          ("became $0.", 128, D.INK)],
}


def layout(lines, max_w, x=80, cy=540):
    """Stack the lines, centred on cy; one common scale so the widest line fits in max_w."""
    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8))
    widest = max(D.text_width(ctx, s_, size, "black") / max_w for s_, size, _ in lines)
    k = min(1.0, 1.0 / widest)
    sizes = [size * k for _, size, _ in lines]
    steps = [0.0] + [0.98 * b + 16 * k for b in sizes[1:]]
    total = sum(steps) + 0.72 * sizes[0]
    y = cy - total / 2 + 0.72 * sizes[0]
    out = []
    for (s_, _, col), size, step in zip(lines, sizes, steps):
        y += step
        out.append((s_, x, y, size, col, "black"))
    return out


if __name__ == "__main__":
    pics = {}
    img, cm, crop = shot_frame("hook", 16.0, CC.ov_hook)
    pics["1"] = (img, labels(cm, crop, CC.HP["hook_win"], CC.HP["hook_replay"]), 860)
    img, cm, crop = shot_frame("warm", 15.5, CC.ov_warm)
    pics["2"] = (img, labels(cm, crop, CC.HP["warm_win"], CC.HP["warm_replay"], CC.ONE["pair_split_t"]), 860)
    wt = np.load("results/chaos2_warm_thumb.npz")                  # a pair that parts earlier and wider (chaos2.py)
    photo = {"hook": photo_frame(CC.HP["hook_win"], CC.HP["hook_replay"]),
             "warm": photo_frame(wt["win"], wt["replay"], split_t=float(wt["split_t"]))}
    for tk, lines in TEXTS.items():
        img, pills, x0 = photo["warm" if tk == "C" else "hook"]
        pics["3"] = (img, pills, x0 - 120)
        for pk, (img, pills, max_w) in pics.items():
            extra = None if (tk == "C" and pk == "1") else pills       # don't show $500 beside "became $0"
            T.finish(img.copy(), layout(lines, max_w), f"plinko_chaos_thumb_{tk}{pk}", extra=extra)
