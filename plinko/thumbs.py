"""YouTube thumbnails (1280x720) -> media/thumbnails/. Drawn from the same simulator paths as the film."""
import math
import os

import cairo

import board as B
import draw as D
import render as R


def canvas():
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1080)
    c = cairo.Context(s)
    D.background(c, 1920, 1080)
    return s, c


def save(s, name):
    os.makedirs("media/thumbnails", exist_ok=True)
    big = f"media/thumbnails/{name}_1920.png"
    s.write_to_png(big)
    from PIL import Image
    Image.open(big).resize((1280, 720), Image.LANCZOS).save(f"media/thumbnails/{name}.png")
    os.remove(big)


def thumb_a(d):
    """Rain of chips over the board, big question."""
    s, c = canvas()
    v = D.fit_view(1100, 30, 1880, 1050, margin_bottom_labels=R.LBL)
    D.draw_board(c, v, label_slots=False)
    lx, ly = v.p(5 * B.SLOT_PITCH, B.last_row_y() + B.FLOOR_BELOW_LAST_ROW)
    D.text(c, "$10,000", lx, ly + 44, 34, D.GOLD, 1, "semibold", "center")
    pos, _, _ = R.rain_state(d, 9.0)
    for p in pos:
        D.draw_chip(c, v, p[0], p[1], p[2], alpha=0.9)
    D.text(c, "WHERE", 80, 300, 190, D.INK, 1, "black")
    D.text(c, "DO YOU", 80, 480, 190, D.INK, 1, "black")
    D.text(c, "DROP IT?", 80, 660, 190, D.GOLD, 1, "black")
    D.text(c, "I rebuilt the board in a physics engine", 86, 780, 46, D.INK2, 1, "medium")
    save(s, "plinko_thumb_A")


def thumb_b(d):
    """The twin chips: 0.001 mm apart, different slots."""
    s, c = canvas()
    v = D.fit_view(1100, 30, 1880, 1050, margin_bottom_labels=R.LBL)
    D.draw_board(c, v, label_slots=False)
    lx, ly = v.p(5 * B.SLOT_PITCH, B.last_row_y() + B.FLOOR_BELOW_LAST_ROW)
    D.text(c, "$10,000", lx, ly + 44, 34, D.GOLD, 1, "semibold", "center")
    ra, rb = d["anim"]["twin_a"], d["anim"]["twin_b"]
    for rec, col in ((rb, D.S1), (ra, D.CHIP)):
        c.set_line_width(7)
        c.set_source_rgba(*D.rgba(col, 0.8))
        for i in range(len(rec)):
            x, y = v.p(rec[i, 1], rec[i, 2])
            (c.move_to if i == 0 else c.line_to)(x, y)
        c.stroke()
        D.draw_chip(c, v, rec[-1, 1], rec[-1, 2], rec[-1, 3], color=col)
    D.text(c, "0.001 mm", 80, 330, 200, D.GOLD, 1, "black")
    D.text(c, "apart.", 80, 520, 200, D.INK, 1, "black")
    D.text(c, "Different slots.", 86, 650, 84, D.INK, 1, "black")
    D.text(c, "So where should you drop a Plinko chip?", 86, 770, 44, D.INK2, 1, "medium")
    save(s, "plinko_thumb_B")


if __name__ == "__main__":
    R.set_format("wide")
    d = R.data()
    thumb_a(d)
    thumb_b(d)
    print("thumbnails written")
