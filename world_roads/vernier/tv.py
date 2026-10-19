"""A TV screen up close: square pixels, each a red, a green and a blue stripe in a black matrix. A black
horizontal line is drawn on it exactly as a screen draws one: each row of pixels is dimmed by the share
of it the line covers, so a line that sits between rows dims both a little.
"""
import math, functools
import numpy as np
import cv2
from look import W, H
import retina as R

Q = 2
STRIPES = np.array([[255, 52, 40], [60, 236, 80], [64, 100, 255]], np.float32)


def row_cover(rows, y0, th):
    """share of each pixel row [r, r+1) covered by the line [y0, y0+th)"""
    return np.clip(np.minimum(rows + 1, y0 + th) - np.maximum(rows, y0), 0, 1)


def macro(u0, v0, k, centre, radius, y0=0.0, th=2.73, step=0.0, xm=0.0, glow=1.0, labels=False):
    """the screen up close in a round field: (u0, v0) the pixel coordinate at the field's centre, k screen px
    per pixel. Returns uint8 (H, W, 3); outside the field is dark."""
    w, h = W // Q, H // Q
    kk = k / Q
    cx, cy = centre[0] / Q, centre[1] / Q
    img = np.zeros((h, w, 3), np.float32)
    c0 = int(math.floor(u0 - (cx + kk) / kk)) - 1
    c1 = int(math.ceil(u0 + (w - cx + kk) / kk)) + 1
    r0 = int(math.floor(v0 - (cy + kk) / kk)) - 1
    r1 = int(math.ceil(v0 + (h - cy + kk) / kk)) + 1
    rows = np.arange(r0, r1, dtype=np.float64)
    left = 1 - row_cover(rows, y0, th)
    right = 1 - row_cover(rows, y0 + step, th)
    sw = kk / 3
    for ci in range(c0, c1):
        lum = left if ci + 0.5 < xm else right
        x = (ci - u0) * kk + cx
        for ri, r in enumerate(range(r0, r1)):
            y = (r - v0) * kk + cy
            b = lum[ri]
            if b <= 0.003:
                continue
            for s in range(3):
                xa = x + s * sw + sw * 0.16
                xb = x + (s + 1) * sw - sw * 0.16
                cv2.rectangle(img, (int(round(xa * 8)), int(round((y + kk * 0.10) * 8))),
                              (int(round(xb * 8)), int(round((y + kk * 0.90) * 8))),
                              [float(c) * b for c in STRIPES[s]], -1, cv2.LINE_AA, shift=3)
    # the phosphor glow and a little lens softness
    soft = cv2.GaussianBlur(img, (0, 0), max(0.6, kk * 0.03))
    bloom = cv2.GaussianBlur(img, (0, 0), max(1.0, kk * 0.18))
    img = soft * 0.85 + bloom * 0.45 * glow
    m, inside = R.field_mask(cx, cy, radius / Q, Q)
    img = img * (1 - 0.3 / 0.38 * (1 - m / np.maximum(inside, 1e-6)))[..., None] * inside[..., None] \
        + R.OUTSIDE * (1 - inside[..., None])
    # finished at half resolution, then scaled up
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
