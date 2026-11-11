"""Top-down pictures of finished tournament runs: one row per run, one tile per 2.5 s."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools import mix_curve

HERE = os.path.dirname(os.path.abspath(__file__))
COL = np.array([[210, 54, 43], [242, 179, 29], [98, 163, 59], [44, 107, 209], [244, 239, 230], [91, 52, 32]], np.uint8)


def tile(p, ing, R, d, px=3.0):
    S = int(2 * R * px) + 2
    img = np.full((S, S, 3), 252, np.uint8)
    zb = np.full((S, S), -1.0)
    o = np.argsort(p[:, 2])
    x = ((p[o, 0] + R) * px).astype(int)
    y = ((R - p[o, 1]) * px).astype(int)
    r = max(int(d * px / 2), 1)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx = np.clip(x + dx, 0, S - 1); yy = np.clip(y + dy, 0, S - 1)
            m = p[o, 2] > zb[yy, xx]
            img[yy[m], xx[m]] = COL[ing[o]][m]
            zb[yy[m], xx[m]] = p[o, 2][m]
    return Image.fromarray(img)


def main():
    work = os.path.join(HERE, "cache", "tournament")
    names = sys.argv[2:] or sorted(n for n in os.listdir(work) if os.path.exists(os.path.join(work, n, "meta.json")))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "experiments", "tournament_topviews.png")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    rows = []
    for n in names:
        m = json.load(open(os.path.join(work, n, "meta.json")))
        z = np.load(os.path.join(work, n, "meta.npz"))
        if "snaps" not in z:
            continue
        tiles = [tile(s, z["ingredient"], m["bowl"]["R"], m["grain"]) for s in z["snaps"]]
        w, h = tiles[0].size
        row = Image.new("RGB", (230 + (w + 6) * len(tiles), h + 8), (252, 252, 251))
        dr = ImageDraw.Draw(row)
        dr.text((8, h // 2 - 22), m["label"], fill=(11, 11, 11), font=font)
        dr.text((8, h // 2), f"{m['load']}, {m['grain']} mm, x{m['speed']:g}", fill=(82, 81, 78), font=font)
        dr.text((8, h // 2 + 22), f"{100 * mix_curve(z)[-1]:.0f}% mixed", fill=(82, 81, 78), font=font)
        for k, t in enumerate(tiles):
            row.paste(t, (230 + (w + 6) * k, 4))
        rows.append(row)
    W = max(r.size[0] for r in rows)
    sheet = Image.new("RGB", (W, sum(r.size[1] for r in rows)), (252, 252, 251))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.size[1]
    sheet.save(out)
    print("wrote", out, sheet.size)


if __name__ == "__main__":
    main()
