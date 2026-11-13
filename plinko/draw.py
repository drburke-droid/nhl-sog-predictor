"""Cairo drawing helpers: the board to scale, chips, text, simple charts. Dark studio look."""
import math

import cairo
import numpy as np

import board as B

# ---------------------------------------------------------------- palette (dark surface)
BG0 = (0.055, 0.063, 0.082)       # #0e1015
BG1 = (0.090, 0.102, 0.129)
INK = (0.941, 0.937, 0.925)       # #f0efec text primary
INK2 = (0.765, 0.761, 0.718)      # #c3c2b7 text secondary
INK3 = (0.55, 0.55, 0.52)         # muted
GRID = (0.20, 0.21, 0.23)
BOARD = (0.925, 0.890, 0.820)     # cream backboard
BOARD_EDGE = (0.70, 0.66, 0.60)
WALL = (0.985, 0.975, 0.955)
PIN = (0.36, 0.37, 0.40)
PIN_HI = (0.80, 0.81, 0.84)
CHIP = (0.890, 0.286, 0.282)      # #e34948
CHIP_RIM = (0.86, 0.87, 0.89)
GOLD = (0.929, 0.631, 0.0)        # #eda100 the $10k accent
# chart series (validated dark-mode categorical slots 1..3)
S1 = (0.224, 0.529, 0.898)        # #3987e5 blue
S2 = (0.851, 0.349, 0.149)        # #d95926 orange
S3 = (0.098, 0.620, 0.439)        # #199e70 aqua

SLOT_LABELS = ["$100", "$500", "$1,000", "$0", "$10,000", "$0", "$1,000", "$500", "$100"]
SLOT_FILL = [(0.20, 0.30, 0.42), (0.22, 0.36, 0.52), (0.25, 0.44, 0.64), (0.16, 0.16, 0.17),
             (0.55, 0.40, 0.05), (0.16, 0.16, 0.17), (0.25, 0.44, 0.64), (0.22, 0.36, 0.52),
             (0.20, 0.30, 0.42)]


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], a)


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def fade(t, t0, t1, d=0.5):
    """1 inside [t0, t1], ramps over d seconds at both ends."""
    if t < t0 or t > t1:
        return 0.0
    return min(1.0, (t - t0) / d if d > 0 else 1.0, (t1 - t) / d if d > 0 else 1.0)


def new_surface(w, h):
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    return s, cairo.Context(s)


def background(ctx, w, h):
    g = cairo.RadialGradient(w * 0.5, h * 0.45, 0, w * 0.5, h * 0.45, max(w, h) * 0.75)
    g.add_color_stop_rgb(0, *BG1)
    g.add_color_stop_rgb(1, *BG0)
    ctx.set_source(g)
    ctx.paint()


# ---------------------------------------------------------------- text
def text(ctx, s, x, y, size, color=INK, alpha=1.0, weight="regular", align="left", family="Inter",
         anchor="baseline"):
    fam = family
    wt = cairo.FONT_WEIGHT_BOLD if weight in ("bold", "black", "semibold") else cairo.FONT_WEIGHT_NORMAL
    if weight == "black":
        fam = "Inter Display Black"
    elif weight == "semibold":
        fam = "Inter SemiBold"
        wt = cairo.FONT_WEIGHT_NORMAL
    elif weight == "medium":
        fam = "Inter Medium"
    ctx.select_font_face(fam, cairo.FONT_SLANT_NORMAL, wt)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    fx = ctx.font_extents()
    if align == "center":
        x -= ext.x_advance / 2
    elif align == "right":
        x -= ext.x_advance
    if anchor == "middle":
        y += (fx[0] - fx[1]) / 2 * 0.9
    elif anchor == "top":
        y += fx[0] * 0.78
    ctx.set_source_rgba(*rgba(color, alpha))
    ctx.move_to(x, y)
    ctx.show_text(s)
    return ext.x_advance


def text_width(ctx, s, size, weight="regular", family="Inter"):
    fam = family
    wt = cairo.FONT_WEIGHT_BOLD if weight in ("bold", "black", "semibold") else cairo.FONT_WEIGHT_NORMAL
    if weight == "black":
        fam = "Inter Display Black"
    elif weight == "semibold":
        fam, wt = "Inter SemiBold", cairo.FONT_WEIGHT_NORMAL
    elif weight == "medium":
        fam = "Inter Medium"
    ctx.select_font_face(fam, cairo.FONT_SLANT_NORMAL, wt)
    ctx.set_font_size(size)
    return ctx.text_extents(s).x_advance


def wrap(ctx, s, size, max_w, weight="regular"):
    words, lines, cur = s.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if text_width(ctx, t, size, weight) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def paragraph(ctx, s, x, y, size, max_w, color=INK, alpha=1.0, weight="regular", lh=1.32, align="left"):
    lines = wrap(ctx, s, size, max_w, weight)
    for i, ln in enumerate(lines):
        text(ctx, ln, x, y + i * size * lh, size, color, alpha, weight, align)
    return y + len(lines) * size * lh


def rounded_rect(ctx, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def column(ctx, x, base_y, w, h, color, alpha=1.0, r=4):
    """Bar growing up from base_y, 4px rounded data end, square at the baseline."""
    if h <= 0.5:
        return
    r = min(r, w / 2, h)
    ctx.new_path()
    ctx.move_to(x, base_y)
    ctx.line_to(x, base_y - h + r)
    ctx.arc(x + r, base_y - h + r, r, math.pi, 1.5 * math.pi)
    ctx.arc(x + w - r, base_y - h + r, r, 1.5 * math.pi, 2 * math.pi)
    ctx.line_to(x + w, base_y)
    ctx.close_path()
    ctx.set_source_rgba(*rgba(color, alpha))
    ctx.fill()


def caption_box(ctx, w, h, lines, t_in, t_out, t, size=34, y_frac=0.885, max_w=None):
    a = fade(t, t_in, t_out, 0.4)
    if a <= 0:
        return
    max_w = max_w or w * 0.8
    allw = []
    for ln in lines:
        allw += wrap(ctx, ln, size, max_w)
    lh = size * 1.3
    y = h * y_frac - (len(allw) - 1) * lh
    for i, ln in enumerate(allw):
        text(ctx, ln, w / 2, y + i * lh, size, INK, a, "medium", "center")


# ---------------------------------------------------------------- board
class View:
    """Maps board metres to screen pixels. (cx, top) is where board x = 5 pitches, y = y_top lands."""

    def __init__(self, cx, top_px, scale, y_top=-0.55 * B.SLOT_PITCH):
        self.cx, self.top, self.s, self.y_top = cx, top_px, scale, y_top

    def p(self, x, y):
        return self.cx + (x - 5 * B.SLOT_PITCH) * self.s, self.top + (y - self.y_top) * self.s

    def L(self, d):
        return d * self.s

    def height_px(self):
        return (B.last_row_y() + B.FLOOR_BELOW_LAST_ROW - self.y_top) * self.s

    def width_px(self):
        return 9 * B.SLOT_PITCH * self.s


def fit_view(x0, y0, x1, y1, margin_bottom_labels=0.0):
    """Largest view whose playfield (plus slot labels) fits the screen box."""
    W = B.SLOT_PITCH
    y_top = -0.55 * W
    total_h = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW - y_top + margin_bottom_labels
    s = min((x1 - x0) / (9.6 * W), (y1 - y0) / total_h)
    return View((x0 + x1) / 2, y0 + ((y1 - y0) - total_h * s) / 2, s, y_top)


PEG_XY, PEG_ROW = B.pegs()
WALLS = [B.wall_polyline(-1), B.wall_polyline(1)]
DIVS = B.divider_segments()


def draw_board(ctx, v, alpha=1.0, slot_fill=None, label_slots=True, pins_alpha=1.0, highlight=None,
               label_size=None, show_values=True):
    W = B.SLOT_PITCH
    y_top = v.y_top
    y_floor = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW
    # backboard
    x0, y0 = v.p(0.5 * W, y_top)
    x1, y1 = v.p(9.5 * W, y_floor)
    pad = v.L(0.12 * W)
    rounded_rect(ctx, x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, v.L(0.06 * W))
    ctx.set_source_rgba(*rgba(BOARD_EDGE, alpha))
    ctx.fill()
    g = cairo.LinearGradient(0, y0, 0, y1)
    g.add_color_stop_rgba(0, *rgba(mix(BOARD, (1, 1, 1), 0.25), alpha))
    g.add_color_stop_rgba(1, *rgba(BOARD, alpha))
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.set_source(g)
    ctx.fill()
    # slots
    ydiv = B.last_row_y() + B.DIVIDER_TOP_BELOW_LAST_ROW
    for k in range(9):
        fill = (slot_fill or SLOT_FILL)[k]
        a = alpha * (0.95 if highlight is None or highlight == k + 1 else 0.35)
        sx0, sy0 = v.p((k + 0.5) * W, ydiv)
        sx1, sy1 = v.p((k + 1.5) * W, y_floor)
        ctx.rectangle(sx0, sy0, sx1 - sx0, sy1 - sy0)
        ctx.set_source_rgba(*rgba(fill, a))
        ctx.fill()
    # walls (sawtooth), clipped to the backboard
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.clip()
    for side, poly in zip((-1, 1), WALLS):
        outer = (0.5 * W - 0.12 * W) if side < 0 else (9.5 * W + 0.12 * W)
        ctx.new_path()
        ctx.move_to(*v.p(outer, poly[0][1]))
        for (x, y) in poly:
            ctx.line_to(*v.p(x, y))
        ctx.line_to(*v.p(outer, poly[-1][1]))
        ctx.close_path()
        ctx.set_source_rgba(*rgba(WALL, alpha))
        ctx.fill_preserve()
        ctx.set_source_rgba(*rgba(BOARD_EDGE, alpha * 0.9))
        ctx.set_line_width(max(1.0, v.L(0.004)))
        ctx.stroke()
    ctx.restore()
    # dividers
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for (x, ya, _, yb) in DIVS:
        ctx.move_to(*v.p(x, ya))
        ctx.line_to(*v.p(x, yb))
    ctx.set_source_rgba(0.25, 0.25, 0.27, alpha)
    ctx.set_line_width(max(1.5, v.L(B.DIVIDER_THICKNESS)))
    ctx.stroke()
    # pins
    rp = max(2.2, v.L(B.PEG_DIAMETER / 2) * 1.3)
    for (x, y) in PEG_XY:
        px, py = v.p(x, y)
        ctx.arc(px + rp * 0.25, py + rp * 0.35, rp * 1.05, 0, 2 * math.pi)
        ctx.set_source_rgba(0, 0, 0, 0.18 * alpha * pins_alpha)
        ctx.fill()
        gr = cairo.RadialGradient(px - rp * 0.35, py - rp * 0.35, rp * 0.1, px, py, rp)
        gr.add_color_stop_rgba(0, *rgba(PIN_HI, alpha * pins_alpha))
        gr.add_color_stop_rgba(1, *rgba(PIN, alpha * pins_alpha))
        ctx.arc(px, py, rp, 0, 2 * math.pi)
        ctx.set_source(gr)
        ctx.fill()
    # slot labels under the board
    if label_slots and show_values:
        size = label_size or max(12, v.L(0.30 * W))
        for k in range(9):
            lx, ly = v.p((k + 1) * W, y_floor)
            col = GOLD if k == 4 else INK
            a = alpha if highlight is None or highlight == k + 1 else alpha * 0.4
            text(ctx, SLOT_LABELS[k], lx, ly + pad + size * 1.15, size, col, a, "semibold", "center")


def label_size_for(v):
    return max(12, v.L(0.30 * B.SLOT_PITCH))


def draw_chip(ctx, v, x, y, ang=0.0, alpha=1.0, color=CHIP, ghost=False, r_scale=1.0):
    px, py = v.p(x, y)
    R = v.L(B.CHIP_DIAMETER / 2) * r_scale
    if ghost:
        ctx.arc(px, py, R, 0, 2 * math.pi)
        ctx.set_source_rgba(*rgba(color, alpha))
        ctx.fill()
        return
    ctx.arc(px + R * 0.06, py + R * 0.1, R * 1.02, 0, 2 * math.pi)
    ctx.set_source_rgba(0, 0, 0, 0.25 * alpha)
    ctx.fill()
    ctx.arc(px, py, R, 0, 2 * math.pi)
    ctx.set_source_rgba(*rgba(CHIP_RIM, alpha))
    ctx.fill()
    ctx.arc(px, py, R * 0.86, 0, 2 * math.pi)
    ctx.set_source_rgba(*rgba(color, alpha))
    ctx.fill()
    # an off-centre mark so spin is visible
    ctx.set_line_width(max(1.0, R * 0.12))
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for k in range(3):
        a = ang + k * 2 * math.pi / 3
        ctx.move_to(px + math.cos(a) * R * 0.30, py + math.sin(a) * R * 0.30)
        ctx.line_to(px + math.cos(a) * R * 0.66, py + math.sin(a) * R * 0.66)
    ctx.set_source_rgba(1, 1, 1, 0.85 * alpha)
    ctx.stroke()
    ctx.arc(px, py, R * 0.14, 0, 2 * math.pi)
    ctx.set_source_rgba(1, 1, 1, 0.85 * alpha)
    ctx.fill()


def dim_line(ctx, xa, ya, xb, yb, label, size=22, alpha=1.0, color=INK, side=1, offset=0.0, tick=8):
    """Engineering-style dimension line with end ticks and a label."""
    ctx.set_source_rgba(*rgba(color, alpha))
    ctx.set_line_width(1.5)
    ctx.move_to(xa, ya)
    ctx.line_to(xb, yb)
    ctx.stroke()
    ang = math.atan2(yb - ya, xb - xa)
    nx, ny = -math.sin(ang), math.cos(ang)
    for (x, y) in ((xa, ya), (xb, yb)):
        ctx.move_to(x - nx * tick, y - ny * tick)
        ctx.line_to(x + nx * tick, y + ny * tick)
    ctx.stroke()
    mx, my = (xa + xb) / 2, (ya + yb) / 2
    if abs(ang) < 0.3 or abs(abs(ang) - math.pi) < 0.3:   # horizontal
        text(ctx, label, mx, my - 12 * side + (size if side < 0 else 0), size, color, alpha, "semibold",
             "center")
    else:
        text(ctx, label, mx + 14 * side, my, size, color, alpha, "semibold",
             "left" if side > 0 else "right", anchor="middle")


def frame_to_bytes(surface):
    surface.flush()
    return bytes(surface.get_data())


def pill(ctx, s, x, y, size, color=INK, bg=BG0, alpha=1.0, weight="black"):
    """Text centred on (x, y) on a dark rounded pill: readable over the cream board."""
    w = text_width(ctx, s, size, weight)
    pad = size * 0.35
    rounded_rect(ctx, x - w / 2 - pad, y - size * 0.62 - pad * 0.6, w + 2 * pad, size * 1.24 + pad * 1.2, size * 0.3)
    ctx.set_source_rgba(*rgba(bg, 0.88 * alpha))
    ctx.fill()
    text(ctx, s, x, y, size, color, alpha, weight, "center", anchor="middle")
