"""Thumbnails for the chaos cut (1280x720): every pairing of three lines of text with three pictures of our board.

    python3 thumbs_chaos.py   -> ../videos/plinko/thumbnails/plinko_chaos_thumb_<text><picture>.png (9 files)

Text  A  "Same release point. Different prize."
      B  "Match release point to an atom, different result."
      C  "0.00006 °C warmer: $10,000 became $0."
Picture 1  the hook's last frame: a $10,000 drop and its atom-close replay in $500 (prize pills shown, except
           with text C, which names $0)
        2  the warm-board pair: the same release point, the board 0.00006 C warmer, $10,000 and $0, with a ring
           where the paths part
        3  our board up close (lower half, front camera): the paths glowing on its pins, rendered chips
           (blender/render_thumb_chips.py) in their slots, the ring where the paths part. A and B use the hook
           pair; C uses another pair from the same warm-board experiment (results/chaos2_warm_thumb.npz) that
           also lands in $0 but parts earlier and swings up to 54 cm apart, so the split is easy to see.
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


CHIPS = {"red": "blender/assets/thumb/chip_red.png", "blue": "blender/assets/thumb/chip_blue.png"}
CLOSE = (0.54, 0.553, 0.548)           # front camera, zoomed on the lower board: the split, the slots, the prize panel


def closeup(win, rep, split_t=None, crop=CLOSE):
    """Our board up close (the front plate, zoomed on its lower half): the two paths glowing on its own pins,
    rendered chips in their slots, the ring where the paths part."""
    cm = C3.cam("front")
    img = C3.crop_img(C3.plate("front", "set"), crop).astype(np.float32)
    to_px = lambda x, y: np.array(cm.px(C3.board_world(x, y, 0.03), crop))

    def paste(name, xy, ang):
        ch = cv2.imread(CHIPS[name], cv2.IMREAD_UNCHANGED)
        ys, xs = np.nonzero(ch[..., 3] > 10)
        ch = ch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        c = to_px(*xy)
        d = 2 * float(np.hypot(*(to_px(xy[0] + C3.B.CHIP_DIAMETER / 2, xy[1]) - c)))
        ch = cv2.resize(ch, (int(d), int(d)), interpolation=cv2.INTER_AREA)
        M_ = cv2.getRotationMatrix2D((d / 2, d / 2), -np.degrees(ang), 1.0)
        ch = cv2.warpAffine(ch, M_, (int(d), int(d)), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        x, y = int(round(c[0] - d / 2)), int(round(c[1] - d / 2))
        a = ch[..., 3:4] / 255.0
        reg = img[y:y + ch.shape[0], x:x + ch.shape[1]]
        img[y:y + ch.shape[0], x:x + ch.shape[1]] = ch[..., :3] / 255.0 * a + reg * (1 - a)
    paste("blue", rep[-1, 1:3], rep[-1, 3])
    paste("red", win[-1, 1:3], win[-1, 3])
    cov = C3.crop_img(C3.plate("front", "cover"), crop)
    img = cov[..., :3] + img * (1 - cov[..., 3:4])
    glow = np.zeros_like(img)
    core = np.zeros_like(img)
    for rec, col in ((rep, D.S1), (win, D.GOLD)):
        pts = np.array([to_px(x, y) for x, y in rec[:, 1:3]]).astype(np.int32)
        bgr = np.array(col[::-1], np.float32)
        cv2.polylines(glow, [pts], False, tuple(float(v) for v in bgr), 12, cv2.LINE_AA)
        cv2.polylines(core, [pts], False, tuple(float(v) * 0.4 + 0.6 for v in bgr), 4, cv2.LINE_AA)
    img = np.clip(img + 0.9 * cv2.GaussianBlur(glow, (0, 0), 10) + 0.85 * core, 0, 1) * C3.vignette()
    x0 = to_px(0.5 * C3.W - 0.17, C3.B.last_row_y())[0]             # the outer edge of the striped side frame
    return img, pill_layer(to_px, win, rep, 46, split_t, dy=-70), int(x0)


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
    close = {"hook": closeup(CC.HP["hook_win"], CC.HP["hook_replay"], CC.FILM["hook_split_t"]),
             "warm": closeup(wt["win"], wt["replay"], float(wt["split_t"]))}
    for tk, lines in TEXTS.items():
        img, pills, x0 = close["warm" if tk == "C" else "hook"]
        pics["3"] = (img, pills, x0 - 80 - 60)
        for pk, (img, pills, max_w) in pics.items():
            extra = None if (tk == "C" and pk == "1") else pills       # don't show $500 beside "became $0"
            T.finish(img.copy(), layout(lines, max_w), f"plinko_chaos_thumb_{tk}{pk}", extra=extra)
