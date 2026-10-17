"""The vernier test: six rounds, each half the offset of the last (4 px down to 1/8 px of a 1080p frame).

Each round shows four bar pairs, A-D. In three the lower bar sits exactly under the upper one; in one it is
shifted sideways. A countdown, then the answer, with a magnifier showing the real 1080p pixels.

    python rounds.py                 -> out/rounds.mp4
    python rounds.py still 7.5 ...   -> stills
"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw
import ui
from ui import W, H, S, GOLD, INK, INK2, MUTED, smooth, fade, text

OFFSETS = [4, 2, 1, 0.5, 0.25, 0.125]            # 1080p pixels
ANSWERS = [2, 0, 3, 1, 3, 0]                    # which of A-D is shifted
SIGNS = [1, -1, 1, -1, -1, 1]                   # right or left
XS = [420, 780, 1140, 1500]                     # stimulus centres (1080p units)
YC = 430
BAR_W, BAR_H, GAP = 4, 110, 4
T_INTRO, T_LOOK, T_REVEAL, T_END = 1.4, 5.0, 3.6, 0.5
LOUPE_R = 140
ROUND = T_INTRO + T_LOOK + T_REVEAL + T_END


def vernier(p, cx, d):
    """Upper bar on whole pixels at cx; lower bar shifted by d."""
    x0 = cx - BAR_W / 2
    p.rect(x0, YC - GAP / 2 - BAR_H, x0 + BAR_W, YC - GAP / 2)
    p.rect(x0 + d, YC + GAP / 2, x0 + d + BAR_W, YC + GAP / 2 + BAR_H)


def magnifier(fr, d, cx, center, radius, mag, label, a):
    """Nearest-neighbour view of the 1080p pixels around the gap, with a pixel grid and the offset marked."""
    if a <= 0:
        return fr
    half = radius / mag                                           # 1080p px shown either side
    q = ui.Patch(cx - half - 1, YC - half - 1, cx + half + 1, YC + half + 1, s=1)
    vernier(q, cx, d)
    src = q.srgb()
    k = int(round(mag * S))
    big = np.kron(src, np.ones((k, k, 1), np.uint8))
    for g in range(0, big.shape[0], k):                           # pixel grid
        big[g, :] = (big[g, :].astype(np.int16) * 0.55 + 18).astype(np.uint8)
    for g in range(0, big.shape[1], k):
        big[:, g] = (big[:, g].astype(np.int16) * 0.55 + 18).astype(np.uint8)
    # crop so that (cx, YC) sits at the loupe centre
    ox = int(round((cx - q.X0) * k)) - int(radius * S)
    oy = int(round((YC - q.Y0) * k)) - int(radius * S)
    R = int(radius * S)
    crop = big[oy:oy + 2 * R, ox:ox + 2 * R]
    im = Image.fromarray(fr)
    mask = Image.new("L", (2 * R, 2 * R), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 2 * R - 1, 2 * R - 1], fill=int(255 * a))
    X, Y = int(center[0] * S) - R, int(center[1] * S) - R
    im.paste(Image.fromarray(crop), (X, Y), mask)
    dr = ImageDraw.Draw(im, "RGBA")
    col = GOLD + (int(255 * a),)
    dr.ellipse([X, Y, X + 2 * R, Y + 2 * R], outline=col, width=5 * S)
    # mark the two right edges, and the offset between them
    xe1 = X + R + (BAR_W / 2) * k
    xe2 = X + R + (BAR_W / 2 + d) * k
    yy = Y + R
    for xe, y_0, y_1 in ((xe1, Y + 0.12 * R, yy - GAP / 2 * k), (xe2, yy + GAP / 2 * k, Y + 1.88 * R)):
        dr.line([(xe, y_0), (xe, y_1)], fill=(90, 200, 255, int(220 * a)), width=2 * S)
    l1, _, l2 = label.partition("\n")
    f1, f2 = ui.font(40, "Bold"), ui.font(28, "Regular")
    wmax = max(dr.textlength(l1, font=f1), dr.textlength(l2, font=f2) if l2 else 0)
    tx = min(max(X + R, wmax / 2 + 30 * S), W - wmax / 2 - 30 * S)          # keep the label on screen
    dr.text((tx, Y + 2 * R + 22 * S), l1, font=f1, fill=col, anchor="ma")
    if l2:
        dr.text((tx, Y + 2 * R + 72 * S), l2, font=f2, fill=INK2 + (int(255 * a),), anchor="ma")
    return np.asarray(im).copy()


class Rounds(ui.Section):
    name = "rounds"

    def __init__(self, offsets=OFFSETS):
        self.offsets = offsets
        self.dur = ROUND * len(offsets)
        self.sfx = []                                             # (time, "tick" | "ding")
        for r in range(len(offsets)):
            t0 = r * ROUND
            self.sfx += [(t0 + T_INTRO + k, "tick") for k in range(5)] + [(t0 + T_INTRO + T_LOOK, "ding")]

    def frame(self, t):
        r = min(int(t // ROUND), len(self.offsets) - 1)
        u = t - r * ROUND
        d_px, ans, sg = self.offsets[r], ANSWERS[r], SIGNS[r]
        fr = ui.blank()
        a_stim = fade(u, 0.3, ROUND - 0.15, 0.4, 0.35)
        if a_stim > 0:
            p = ui.Patch(XS[0] - 40, YC - GAP / 2 - BAR_H - 2, XS[-1] + 40, YC + GAP / 2 + BAR_H + 2)
            for i, cx in enumerate(XS):
                vernier(p, cx, sg * d_px if i == ans else 0.0)
            p.paste(fr, alpha=a_stim)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        a_t = fade(u, 0.0, ROUND - 0.15, 0.4, 0.35)
        text(d, (960, 92), f"ROUND {r + 1} OF {len(self.offsets)}", 30, MUTED, a_t, "SemiBold")
        size = f"{ui.frac(d_px)} pixel" if d_px <= 1 else f"{d_px:g} pixels"
        text(d, (960, 150), f"One pair is off by {size}", 54, INK, a_t, "Bold")
        t_rev = T_INTRO + T_LOOK
        for i, cx in enumerate(XS):
            hit = u >= t_rev and i == ans
            col = GOLD if hit else INK2
            text(d, (cx, YC + BAR_H + 70), "ABCD"[i], 46, col, a_stim, "Bold")
            if hit:                                                # the answer's ring
                g = smooth((u - t_rev) / 0.35)
                rr = (BAR_H + 34) * S * (1.25 - 0.25 * g)
                d.ellipse([cx * S - rr * 0.62, YC * S - rr, cx * S + rr * 0.62, YC * S + rr],
                          outline=GOLD + (int(255 * g * a_stim),), width=5 * S)
        # countdown: a ring that empties, with the seconds left
        if T_INTRO <= u < t_rev:
            left = t_rev - u
            cxr, cyr, rr = 1710 * S, 120 * S, 52 * S
            d.ellipse([cxr - rr, cyr - rr, cxr + rr, cyr + rr], outline=(60, 70, 90, 255), width=8 * S)
            d.arc([cxr - rr, cyr - rr, cxr + rr, cyr + rr], -90, -90 + 360 * left / T_LOOK, fill=GOLD + (255,), width=8 * S)
            text(d, (1710, 122), f"{math.ceil(left)}", 50, INK, 1.0, "Bold")
        fr = np.asarray(im).copy()
        if u >= t_rev:
            a_m = smooth((u - t_rev - 0.25) / 0.4) * fade(u, 0, ROUND - 0.15, 0.1, 0.35)
            cx = XS[ans]
            label = f"{ui.frac(d_px)} px {'right' if sg > 0 else 'left'}"
            if d_px < 1:
                label += "\nthe edge pixel just glows dimmer"
            fr = magnifier(fr, sg * d_px, cx, (cx, 820), LOUPE_R, LOUPE_R / (GAP / 2 + 7), label, a_m)
            im = Image.fromarray(fr)
            text(ImageDraw.Draw(im, "RGBA"), (960, 222), f"Answer: {'ABCD'[ans]}", 48, GOLD,
                 smooth((u - t_rev) / 0.3) * a_t, "Bold")
            fr = np.asarray(im).copy()
        return fr


if __name__ == "__main__":
    sec = Rounds()
    if len(sys.argv) > 2 and sys.argv[1] == "still":
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
