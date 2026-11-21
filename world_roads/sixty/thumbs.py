"""Three thumbnails (1280x720), in the film's own look. Drawn at 4K and scaled down.

    python thumbs.py
"""
import os, sys
os.environ.setdefault("SIXTY_SCALE", "2")
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L
from look import text, GOLD, CREAM, CREAM_DIM, LAPIS, CLAY, CYAN, RED
import clay
import film

OUT = L.OUT


def finish(surf, name):
    fr = L.pixels(surf)[..., :3]
    im = cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA)
    path = os.path.join(OUT, name)
    Image.fromarray(im).save(path)
    print("wrote", path)


def thumb_a():
    """the clock, and the question"""
    surf, c = L.canvas()
    L.sky(c, 0.0, pole=(960, -300), centre=(1300, 540))
    cx, cy, r = 1480, 540, 400
    L.clock(c, cx, cy, r, (10 + 10 / 60) / 12, 8 / 60, 37 / 60, a=1.0)
    L.ticks(c, cx, cy, r * 0.965, 60, r * 0.05, CLAY, 1.0, w=3.0)
    text(c, 90, 400, "WHY 60?", 172, GOLD, 1.0, kind="black", anchor="ls")
    text(c, 90, 640, "WHY 24?", 172, LAPIS, 1.0, kind="black", anchor="ls")
    text(c, 96, 780, "and not 100?", 84, CREAM, 1.0, kind="italic", anchor="ls")
    L.vignette(c, 0.4)
    finish(surf, "why60_thumb_A.png")


def thumb_b():
    """Babylon's square root of 2 beside a clock: the same sixties"""
    surf, c = L.canvas()
    L.sky(c, 0.0, pole=(960, -300), centre=(1300, 540))
    sec = film.Clay()
    d = film.YBC[2]
    marks = sec.ybc_top + sec.ybc_bot + sec.ybc_30
    sc = 1.36
    sec.ybc.draw(c, 1130, 160, marks, 1.0, scale=sc)
    text(c, 80, 300, "YOUR CLOCK", 138, CREAM, 1.0, kind="black", anchor="ls")
    text(c, 80, 470, "RUNS ON", 138, CREAM, 1.0, kind="black", anchor="ls")
    text(c, 80, 640, "BABYLONIAN", 138, CLAY, 1.0, kind="black", anchor="ls")
    text(c, 80, 810, "MATH", 138, CLAY, 1.0, kind="black", anchor="ls")
    text(c, 86, 930, "3,700-year-old tablet: √2 in base 60", 50, CREAM_DIM, 1.0, kind="italic", anchor="ls")
    L.vignette(c, 0.35)
    finish(surf, "why60_thumb_B.png")


def thumb_c():
    """the French decimal clock: noon at 5"""
    surf, c = L.canvas()
    L.sky(c, 0.0, pole=(960, -300), centre=(1300, 540))
    cx, cy, r = 1380, 540, 420
    L.clock(c, cx, cy, r, 0.5, 0.0, 0.0, a=1.0, n=10, numerals="decimal", minute_ticks=100, col=CYAN,
            hand_a=(1.0, 1.0, 0.0))
    text(c, 90, 330, "10 HOURS", 170, CYAN, 1.0, kind="black", anchor="ls")
    text(c, 90, 520, "A DAY", 170, CYAN, 1.0, kind="black", anchor="ls")
    text(c, 96, 690, "France tried it in 1793.", 66, CREAM, 1.0, kind="sans-sb", anchor="ls")
    text(c, 96, 790, "Noon was 5 o'clock.", 66, CREAM_DIM, 1.0, kind="italic", anchor="ls")
    L.vignette(c, 0.4)
    finish(surf, "why60_thumb_C.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    thumb_a()
    thumb_b()
    thumb_c()
