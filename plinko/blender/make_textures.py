"""Textures for the Blender scene (run with the system python): the plexiglass cover's lattice mask and the
chip face. Both are drawn from board.py so they line up with the pins exactly.

    python3 blender/make_textures.py   -> blender/assets/cover_mask.png, chip_face.png
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import board as B  # noqa: E402

W, H = B.SLOT_PITCH, B.ROW_SPACING
# cover extent in board coordinates (metres): wall to wall, from just above the top row to the divider tops
X0, X1 = 0.5 * W, 9.5 * W
Y0, Y1 = -0.45 * W, B.last_row_y() + 0.55 * W
PPM = 2600.0                                   # pixels per metre (2.6 px per mm)
BAR = 0.024                                    # bar width, metres


def px(x, y):
    return ((x - X0) * PPM, (y - Y0) * PPM)


def cover_mask():
    from PIL import Image, ImageDraw, ImageFilter
    w, h = int((X1 - X0) * PPM), int((Y1 - Y0) * PPM)
    im = Image.new("L", (w, h), 0)            # 0 = hole, 255 = acrylic
    d = ImageDraw.Draw(im)
    pts, rows = B.pegs()
    nodes = [tuple(p) for p in pts]
    # wall tooth apexes act as lattice nodes too
    for side in (-1, 1):
        for (x, y) in B.wall_polyline(side)[1:-1]:
            nodes.append((x, y))
    lw = int(BAR * PPM)
    for i, (xa, ya) in enumerate(nodes):
        for (xb, yb) in nodes[i + 1:]:
            dist = math.hypot(xa - xb, ya - yb)
            if abs(dist - W) < 0.02 * W:       # neighbours on the triangular lattice
                d.line([px(xa, ya), px(xb, yb)], fill=255, width=lw)
    for (x, y) in nodes:
        cx, cy = px(x, y)
        r = lw * 0.9
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    for (x, y) in [tuple(p) for p in pts]:       # pin heads poke through the cover
        cx, cy = px(x, y)
        r = B.PEG_DIAMETER * 0.75 * PPM
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    # solid border band along the walls and the top
    d.rectangle([0, 0, w, int(0.06 * PPM)], fill=255)
    for side in (-1, 1):
        poly = B.wall_polyline(side)
        edge = [px(x, y) for (x, y) in poly]
        outer = 0 if side < 0 else w
        d.polygon([(outer, -10)] + edge + [(outer, h + 10)], fill=255)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    im.save(os.path.join(HERE, "assets", "cover_mask.png"), optimize=True)
    print("cover mask", im.size)


def chip_face():
    from PIL import Image, ImageDraw
    n = 1024
    im = Image.new("RGB", (n, n), (176, 18, 22))
    d = ImageDraw.Draw(im)
    c = n / 2
    for rr, col in ((0.47, (236, 236, 232)), (0.43, (176, 18, 22))):   # a thin painted ring
        d.ellipse([c - rr * n, c - rr * n, c + rr * n, c + rr * n], fill=col)
    for k in range(3):                         # three spokes, so spin shows
        a = -math.pi / 2 + k * 2 * math.pi / 3
        x0, y0 = c + math.cos(a) * 0.12 * n, c + math.sin(a) * 0.12 * n
        x1, y1 = c + math.cos(a) * 0.33 * n, c + math.sin(a) * 0.33 * n
        d.line([(x0, y0), (x1, y1)], fill=(236, 236, 232), width=int(0.045 * n))
        for (x, y) in ((x0, y0), (x1, y1)):
            r = 0.0225 * n
            d.ellipse([x - r, y - r, x + r, y + r], fill=(236, 236, 232))
    r = 0.06 * n
    d.ellipse([c - r, c - r, c + r, c + r], fill=(236, 236, 232))
    im.save(os.path.join(HERE, "assets", "chip_face.png"))
    print("chip face", im.size)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
    cover_mask()
    chip_face()
