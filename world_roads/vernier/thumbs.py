"""Three thumbnails (1280x720), in the film's own look.

    python thumbs.py
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L
from look import W, H, S, Cam, CREAM, RED, INK, pencil
import retina as R
import tv as T
import film

OUT = L.OUT


def small(fr):
    return Image.fromarray(cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA))


def headline(im, xy, s, size, col, kind="serif", anchor="la", shadow=None):
    """text on the 1280x720 thumbnail (xy in its px)"""
    d = ImageDraw.Draw(im)
    f = L.font(kind, size / S)                            # size in thumbnail px
    if shadow:
        d.text((xy[0] + 2, xy[1] + 3), s, font=f, fill=shadow, anchor=anchor)
    d.text(xy, s, font=f, fill=col, anchor=anchor)


def thumb_a():
    """the card, and the magnifier over the step"""
    c = film.CARDS[0]
    cam = Cam(c.xm - 250, c.line_y + 60, 1.25)
    fr = film.desk(cam, lens=(c.xm, c.line_y + 1.25, 150, 24.0, 1.0), t=0.0, plates=0.0)
    im = small(fr)
    headline(im, (64, 30), "Half a pixel.", 100, INK)
    headline(im, (70, 560), "and you can see it", 76, RED, kind="hand")
    return im


def thumb_b():
    """the retina under the microscope, the line's middle traced across it, with its step"""
    sc = R.Scope(0.0, 0.34, 170.0)
    cx, cy, rad = W * 0.68, H * 0.5, H * 0.55
    fr = R.mosaic(sc, y0=0.0, step=film.STEP_HALF, lit=1.0, band=0.25, centre=(cx, cy), radius=rad)
    X = lambda u: (u - sc.u) * sc.k + cx
    Y = lambda v: (v - sc.v) * sc.k + cy
    tr = [(X(-6.6), Y(0)), (X(-0.02), Y(0)), (X(0.02), Y(film.STEP_HALF)), (X(6.6), Y(film.STEP_HALF))]
    tr = np.concatenate([np.linspace(tr[i], tr[i + 1], 40) for i in range(3)])
    fr = pencil(fr, [(tr, 1.0, 51)], col=(236, 92, 66), width=5.0)
    im = small(fr)
    headline(im, (46, 230), "Smaller", 80, CREAM)
    headline(im, (46, 318), "than", 80, CREAM)
    headline(im, (46, 406), "one cone.", 80, CREAM)
    headline(im, (50, 530), "you still see it", 58, (250, 120, 92), kind="hand")
    return im


def thumb_c():
    """your retina and a TV screen, side by side"""
    k = film.K_RET * 1.15
    left = R.mosaic(R.Scope(0.0, film.STEP_HALF / 2, k), y0=0.0, step=film.STEP_HALF, lit=1.0, band=0.25,
                    centre=(W * 0.27, H * 0.53), radius=H * 0.36)
    right = T.macro(0.0, film.STEP_HALF / 2 + 1.365, k, (W * 0.73, H * 0.53), H * 0.36, y0=0.0, th=2.73, step=film.STEP_HALF)
    xx = np.arange(W)
    fr = np.ascontiguousarray(np.where((xx < W * 0.5)[None, :, None], left, right).astype(np.uint8))
    im = small(fr)
    headline(im, (640, 26), "Your eye has pixels too.", 78, CREAM, anchor="ma")
    headline(im, (346, 642), "your retina", 50, (230, 220, 205), kind="hand", anchor="ma")
    headline(im, (934, 642), "a TV screen", 50, (230, 220, 205), kind="hand", anchor="ma")
    return im


if __name__ == "__main__":
    for name, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        p = os.path.join(OUT, f"vernier_thumb_{name}.png")
        f().save(p)
        print("wrote", p)
