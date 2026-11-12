"""The table: a nested grid of square plots, oldest company top left, newest bottom right.

The table is a grid of large squares, filled along diagonals from the top left corner to the bottom right
(like a wave spreading from the corner), so at any moment the filled part is a compact triangle the camera
can frame, and at the end the newest giants sit in the bottom right corner with the companies of the middle
years in the centre. Unused large squares go to the other two corners. A large square that holds
small companies is split into four, and those into four again, as far as needed: every company gets the
smallest square its pile will ever need, in the order companies first appear, reading each split square the
same way (top left, top right, bottom left, bottom right). So the first companies (the VOC, the East India
Company, the WIC, Hudson's Bay) share a small block in the top left corner, the 1720 bubble companies sit a
few plots on, and today's giants each fill a whole large square. All canals are straight and line up.

When a company needs a bigger plot than the next free one, the cursor moves on to the next free plot of
that size, and the small plots skipped stay empty (vacant lots). The number of large squares across is
chosen so the finished grid fits a 16:9 frame.

    L = build(companies, extent)     # companies: list of dict(id, first); extent: {id: widest reach, cells}
    L.lots[id] -> (cx, cy, half)     # plot centre and half its side (land, without the canal), in cells
    L.vacant -> [(cx, cy, half)]     # empty plots
"""
import math

GAP = 3.0            # canal width between plots, cells
PAD = 1.06           # a plot is this much wider than the widest its pile ever gets
MIN_HALF = 2.5       # the smallest half-plot, cells
TARGET = 1.75        # width / height of the finished grid (the frame's safe area)
J = 7                # a large square can be split down to 2^7 = 128 plots a side


class Layout:
    pass


def _morton(m):
    """Finest-level offset inside a large square -> (x, y) in finest plots."""
    x = y = 0
    for l in range(J):
        q = (m >> (2 * (J - 1 - l))) & 3
        x = (x << 1) | (q & 1)
        y = (y << 1) | (q >> 1)
    return x, y


def _place(order, need, c_top):
    """Each company's (start offset, level) along the Morton cursor; the skipped ranges."""
    p, out, skipped = 0, {}, []
    for k in order:
        j = 0
        while j < J and c_top / 2 ** (j + 1) >= need[k]:
            j += 1
        span = 4 ** (J - j)
        q = -(-p // span) * span                   # next free plot of that size
        if q > p:
            skipped.append((p, q))
        out[k] = (q, j)
        p = q + span
    return out, skipped, p


def _nodes(a, b):
    """Split the finest-slot range [a, b) into aligned squares: [(start, level)]."""
    res = []
    while a < b:
        j = 0
        while (a % 4 ** (J - j)) or a + 4 ** (J - j) > b:
            j += 1
        res.append((a, j))
        a += 4 ** (J - j)
    return res


def build(companies, extent):
    order = [c["id"] for c in sorted(companies, key=lambda c: c["first"])]
    need = {k: 2 * max(extent.get(k, 0) * PAD, MIN_HALF) + GAP for k in order}
    c_top = max(need.values())
    placed, skipped, end = _place(order, need, c_top)
    n_big = -(-end // 4 ** J)
    best = None
    for M in range(2, 30):
        for R in range(1, 30):
            if M * R < n_big or M * R > n_big + max(M, R):
                continue
            score = abs(math.log((M / R) / TARGET)) + 0.02 * (M * R - n_big)
            if best is None or score < best[0]:
                best = (score, M, R)
    _, M, R = best
    # large squares in diagonal order; the spare ones are the cells farthest from the main diagonal
    cells = sorted(((i, j) for i in range(R) for j in range(M)), key=lambda c: (c[0] / R + c[1] / M, c[0]))
    spare = M * R - n_big
    off = sorted(cells, key=lambda c: -abs(c[0] / max(R - 1, 1) - c[1] / max(M - 1, 1)))[:spare]
    seq = [c for c in cells if c not in off]
    fine = c_top / 2 ** J

    def square(start, j):
        b, m = divmod(start, 4 ** J)
        fx, fy = _morton(m)
        size = c_top / 2 ** j
        r_, c_ = seq[b]
        x0 = c_ * c_top + fx * fine
        y0 = r_ * c_top + fy * fine
        return x0 + size / 2, y0 + size / 2, (size - GAP) / 2

    L = Layout()
    L.M, L.c_top, L.rows = M, c_top, R
    L.lots = {k: square(*placed[k]) for k in order}
    L.vacant = [square(s, j) for a, b in skipped for s, j in _nodes(a, b)]
    L.vacant += [square(s, j) for s, j in _nodes(end, n_big * 4 ** J)]      # the rest of the last large square
    for r_, c_ in off:                                                       # the spare large squares
        L.vacant.append(((c_ + 0.5) * c_top, (r_ + 0.5) * c_top, (c_top - GAP) / 2))
    L.level = {k: placed[k][1] for k in order}
    L.x0, L.y0, L.x1, L.y1 = 0.0, 0.0, M * c_top, R * c_top
    L.width, L.height = L.x1, L.y1
    return L
