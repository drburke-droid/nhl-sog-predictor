"""Glass initials: a person's initials drawn in a heavy typeface, turned into glass vessel walls and the
sand cells inside the letters.

Map units are map px (the table is W x H map px; 100 map px = 1 Blender unit). The sand lives on a grid of
CELL-px cells. A set of initials is drawn with its centre at (cx, cy) and capitals `cap` map px tall.

    m = initials("EM", cx, cy, cap=120)
    m.walls      # glass: list of polygons, each a list of rings [[x, y], ...] (outer first, then holes)
    m.inside     # (rows, cols) of the sand cells inside the letters
    m.bbox       # (x0, y0, x1, y1) in map px
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
SS = 4                      # supersampling of the glyph raster (sub-pixel smooth outlines)
WALL_T = 4.0                # glass thickness, map px
GRAIN_R = 1.5               # grain radius, map px (sand keeps this far from the glass)


@dataclass
class Initials:
    text: str
    walls: list
    inside: tuple
    bbox: tuple
    centre: tuple


def _raster(text, cap, tracking):
    """The initials as a supersampled mask (uint8), with the size of the ink box in map px."""
    size = int(round(cap * SS / 0.727))          # Inter's capital height is ~0.727 em
    f = ImageFont.truetype(FONT, size)
    gap = int(round(tracking * cap * SS))
    widths = [f.getbbox(ch)[2] - f.getbbox(ch)[0] for ch in text]
    W = sum(widths) + gap * (len(text) - 1) + 8 * SS
    H = int(cap * SS * 2.0)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    x = 4 * SS
    for ch, w in zip(text, widths):
        bb = f.getbbox(ch)
        d.text((x - bb[0], int(cap * SS * 1.5)), ch, font=f, fill=255, anchor="ls")
        x += w + gap
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 127)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def initials(text, cx, cy, cap=120.0, tracking=0.10, cell=3):
    a = _raster(text, cap, tracking)
    h, w = a.shape
    x0, y0 = cx - w / SS / 2, cy - h / SS / 2
    ink = (a > 127).astype(np.uint8)
    t = int(round(WALL_T * SS / 2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    pad = 4 * t
    inkp = np.pad(ink, pad)
    band = cv2.dilate(inkp, k) - cv2.erode(inkp, k)          # the glass: a band along every outline
    cs, hier = cv2.findContours(band, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    to_map = lambda c: [[round(x0 + (p[0][0] - pad) / SS, 2), round(y0 + (p[0][1] - pad) / SS, 2)] for p in c]
    walls = []
    if hier is not None:
        for i, hh in enumerate(hier[0]):
            if hh[3] != -1:                                   # holes are added to their parent
                continue
            c = cv2.approxPolyDP(cs[i], 0.6, True)
            rings = [to_map(c)]
            j = hh[2]
            while j != -1:
                rings.append(to_map(cv2.approxPolyDP(cs[j], 0.6, True)))
                j = hier[0][j][0]
            walls.append(rings)
    # sand cells: centre inside the ink, at least half the glass + a grain radius from the outline
    dist = cv2.distanceTransform(inkp, cv2.DIST_L2, 5) / SS
    r0, r1 = int(np.floor(y0 / cell)), int(np.ceil((y0 + h / SS) / cell))
    c0, c1 = int(np.floor(x0 / cell)), int(np.ceil((x0 + w / SS) / cell))
    rr, cc = np.mgrid[r0:r1, c0:c1]
    px = ((cc + 0.5) * cell - x0) * SS + pad
    py = ((rr + 0.5) * cell - y0) * SS + pad
    ok = (px >= 0) & (py >= 0) & (px < inkp.shape[1]) & (py < inkp.shape[0])
    dd = np.zeros_like(px, dtype=np.float32)
    dd[ok] = dist[py[ok].astype(int), px[ok].astype(int)]
    keep = dd >= WALL_T / 2 + GRAIN_R
    return Initials(text, walls, (rr[keep], cc[keep]), (x0, y0, x0 + w / SS, y0 + h / SS), (cx, cy))


if __name__ == "__main__":
    m = initials("EM", 500, 300)
    print(m.text, "walls", len(m.walls), "rings", sum(len(p) for p in m.walls), "inside cells", len(m.inside[0]), "bbox", m.bbox)
