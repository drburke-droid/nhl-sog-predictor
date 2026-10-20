"""Three thumbnails (1280x720), in the film's own look.

    python thumbs.py
"""
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import film
import imaging as I
import physics as P
import look as L
import retina as R
from look import W, H, S, Cam, CREAM, RED, INK, pencil

OUT = L.OUT


def small(fr):
    return Image.fromarray(cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA))


def headline(im, xy, s, size, col, kind="serif", anchor="la", shadow=None):
    """text on the 1280x720 thumbnail (xy in its px)"""
    d = ImageDraw.Draw(im)
    f = L.font(kind, size / S)
    if shadow:
        d.text((xy[0] + 2, xy[1] + 3), s, font=f, fill=shadow, anchor=anchor)
    d.text(xy, s, font=f, fill=col, anchor=anchor)


def thumb_a():
    """the dark-trees card, and the magnifier over the ball 400 yards away"""
    cam = Cam(960 - 260, 540 + 40, 1.25)
    fr = film.desk(cam, [film.OPEN_CARD], lens=(960.0, 540.0, 160, 30.0, 1.0), t=0.0, labels=0.0)
    im = small(fr)
    headline(im, (60, 34), "A golf ball, 400 yards away.", 76, INK)
    headline(im, (66, 590), "smaller than one cone. you still see it", 58, RED, kind="hand")
    return im


def thumb_b():
    """the retina under the microscope: the blur spot of a ball 320 yards away, against dark trees, and the
    ball itself (one cone wide) as a red ring"""
    sc = R.Scope(film.BALL_C[0], film.BALL_C[1], 190.0)
    cx, cy, rad = W * 0.68, H * 0.5, H * 0.56
    fr = I.mosaic(sc, film.ball_light(320, sharp=0.0), exposure=film.EXPO, centre=(cx, cy), radius=rad)
    rr = float(P.ball_arcmin(320)) / 2 / I.U * sc.k
    fr = pencil(fr, [(film.circle_pts(cx, cy, rr, n=60), 1.0, 51)], col=(236, 92, 66), width=4.0)
    im = small(fr)
    headline(im, (46, 200), "It doesn’t", 76, CREAM)
    headline(im, (46, 286), "get smaller.", 76, CREAM)
    headline(im, (50, 400), "it gets fainter", 64, (250, 120, 92), kind="hand")
    return im


def thumb_c():
    """the 20/20 E on the cones, and a golf ball at 160 yards drawn in one stroke"""
    sc = R.Scope(0.0, 0.0, 128.0)
    cx, cy, rad = W * 0.66, H * 0.5, H * 0.47
    fr = I.mosaic(sc, lambda u, v: I.e_sharp_at(u, v), overlay=lambda u, v: I.e_sharp_at(u, v), band=0.3, curve="print",
                  centre=(cx, cy), radius=rad)
    r160 = float(P.ball_arcmin(160)) / 2 / I.U * sc.k
    fr = pencil(fr, [(film.circle_pts(cx + 2.6 * sc.k, cy, r160), 1.0, 33)], col=(250, 120, 92), width=5.0)
    im = small(fr)
    headline(im, (40, 190), "160 yards:", 72, CREAM)
    headline(im, (40, 272), "one stroke", 72, CREAM)
    headline(im, (40, 354), "of the 20/20 E", 72, CREAM)
    headline(im, (44, 470), "you can still see it at 600", 54, (250, 120, 92), kind="hand")
    return im


if __name__ == "__main__":
    for name, f in (("A", thumb_a), ("B", thumb_b), ("C", thumb_c)):
        p = os.path.join(OUT, f"golfball_thumb_{name}.png")
        f().save(p)
        print("wrote", p)
