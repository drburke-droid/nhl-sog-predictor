"""Measure the road markings on the reference plate frame: edge lines (least squares), centre-line dash ends."""
import cv2, numpy as np, json
im = cv2.imread("/home/user/lot/plate/p074.png")
g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
bg = cv2.medianBlur(g.astype(np.uint8), 61).astype(np.float32)
white = ((g - bg > 35) & (hsv[..., 1] < 60)).astype(np.uint8)
white[:395] = 0
n, lab, st, cen = cv2.connectedComponentsWithStats(white)
big = sorted(range(1, n), key=lambda i: -st[i, 4])[:2]
fits = {}
for i in big:
    ys, xs = np.where(lab == i)
    keep = ys > 410
    # centre of the stripe in each row, then x = a*y + b
    rows = np.unique(ys[keep])
    cx = np.array([xs[keep][ys[keep] == r].mean() for r in rows])
    a, b = np.polyfit(rows, cx, 1)
    res = cx - (a * rows + b)
    side = "left" if a < 0 else "right"
    fits[side] = (a, b)
    print(side, "x = %.5f*y + %.2f" % (a, b), "rows", rows.min(), rows.max(), "rms", res.std())
(aL, bL), (aR, bR) = fits["left"], fits["right"]
yv = (bR - bL) / (aL - aR); xv = aL * yv + bL
print("vanishing point", xv, yv)
# centre line: through the vanishing point and the near dash centre
ys, xs = np.where(white[520:640, 900:1000] > 0); xs = xs + 900; ys = ys + 520
xn, yn = xs.mean(), ys.mean()
ac = (xn - xv) / (yn - yv)
prof = []
for y in np.arange(yv + 3, 804, 0.25):
    x = xv + ac * (y - yv)
    # brightness across a few pixels around the line, relative to the asphalt beside it
    halfw = max(1.5, 0.012 * (y - yv))
    xi = np.arange(x - halfw, x + halfw + 0.01, 0.5)
    v = cv2.remap(g, xi.astype(np.float32)[None], np.full((1, len(xi)), y, np.float32), cv2.INTER_LINEAR)[0].mean()
    side = cv2.remap(g, np.float32([[x - 6 * halfw, x + 6 * halfw]]), np.full((1, 2), y, np.float32), cv2.INTER_LINEAR)[0].mean()
    prof.append((y, v - side))
prof = np.array(prof)
on = prof[:, 1] > 0.5 * np.percentile(prof[prof[:, 0] > 450, 1], 97)
edges = np.flatnonzero(np.diff(on.astype(int)))
segs = []
for k in range(0, len(edges) - 1):
    if on[edges[k] + 1]:
        y0, y1 = prof[edges[k] + 1, 0], prof[edges[k + 1], 0]
        if y1 - y0 > 0.6: segs.append((round(y0, 2), round(y1, 2)))
print("centre line x = %.2f + %.5f*(y-%.2f)" % (xv, ac, yv))
print("dashes (y top, y bottom):", segs)
json.dump(dict(left=fits["left"], right=fits["right"], vp=[xv, yv], centre=[ac], dashes=segs), open("/home/user/lot/src/marks.json", "w"))
