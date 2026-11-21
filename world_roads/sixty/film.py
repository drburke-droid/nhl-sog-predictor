"""Why 60? Why 24? The film, section by section (4K, 30 fps).

    python film.py NAME [still times...]
    NAME: open, tens, clay, why60, sky, egypt, greece, clocks, decimal, others, ladder, close
"""
import os, sys, math
import numpy as np
import skia

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import look as L
from look import (smooth, ease, fade, ramp, lerp, clamp, mixc, text, caption, label, line, poly, circle, arc, sector,
                  ticks, polar, paint, GOLD, GOLD_DIM, CREAM, CREAM_DIM, CLAY, LAPIS, NIGHTBLUE, CYAN, RED, SUN)
import clay


def caps(c, t, rows):
    for t0, t1, lines in rows:
        caption(c, lines, fade(t, t0, t1, 0.6, 0.6))


def tick_frac(T, snap=0.14):
    """a clock's second count with a quick, eased jump at each whole second"""
    k = math.floor(T)
    return k + smooth((T - k) / snap)


# ------------------------------------------------------------------ hands
def _capsule(x, y, ang, ln, w):
    """a capsule from (x, y) going ang degrees from straight up (clockwise), length ln, width w"""
    p = skia.Path()
    p.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-w / 2, -ln - w / 2, w / 2, w / 2), w / 2, w / 2))
    m = skia.Matrix()
    m.setRotate(ang)
    m.postTranslate(x, y)
    p.transform(m)
    return p


class Hand:
    """a stylised open hand, palm towards us, fingers up. thumb='right' or 'left'. Fingers are listed
    outside-in: pinky, ring, middle, index, thumb. Each finger has segment boundaries (fractions of its length)."""

    def __init__(self, cx, cy, sc=1.0, thumb="right"):
        sg = 1 if thumb == "right" else -1
        self.fingers = []
        spec = [(-64, -6, 118, 31), (-22, -2, 150, 35), (20, 0, 164, 36), (62, 4, 146, 35)]
        for fx, fa, fl, fw in spec:
            bx, by = cx + sg * fx * sc, cy - 40 * sc
            self.fingers.append(dict(x=bx, y=by, ang=sg * fa, ln=fl * sc, w=fw * sc, segs=(0.0, 0.40, 0.70, 1.0)))
        self.fingers.append(dict(x=cx + sg * 70 * sc, y=cy + 52 * sc, ang=sg * 52, ln=112 * sc, w=40 * sc, segs=(0.0, 0.5, 1.0)))
        palm = skia.Path()
        palm.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - 88 * sc, cy - 62 * sc, cx + 88 * sc, cy + 120 * sc), 46 * sc, 46 * sc))
        out = palm
        for f in self.fingers:
            out = skia.Op(out, _capsule(f["x"], f["y"], f["ang"], f["ln"], f["w"]), skia.PathOp.kUnion_PathOp)
        self.outline = out
        self.sc = sc

    def point(self, f, u):
        """a point u (0..1) along finger f's axis"""
        a = math.radians(f["ang"])
        return f["x"] + math.sin(a) * f["ln"] * u, f["y"] - math.cos(a) * f["ln"] * u

    def draw(self, c, a=1.0, col=GOLD, creases=0.0, fill=0.07):
        if a <= 0:
            return
        c.drawPath(self.outline, paint((20, 26, 48), a * 0.9))
        c.drawPath(self.outline, paint(col, a * fill))
        c.drawPath(self.outline, paint(col, a * 0.25, stroke=6 * self.sc, blur=4 * self.sc))
        c.drawPath(self.outline, paint(col, a, stroke=2.0 * self.sc))
        if creases > 0:
            for f in self.fingers:
                for u in f["segs"][1:-1]:
                    x, y = self.point(f, u)
                    a_ = math.radians(f["ang"])
                    dx, dy = math.cos(a_) * f["w"] * 0.32, math.sin(a_) * f["w"] * 0.32
                    line(c, x - dx, y - dy, x + dx, y + dy, col, a * creases * 0.8, w=1.4 * self.sc)

    def light_finger(self, c, k, a, col=GOLD):
        if a <= 0:
            return
        f = self.fingers[k]
        p = _capsule(f["x"], f["y"], f["ang"], f["ln"], f["w"])
        p = skia.Op(p, self.outline, skia.PathOp.kIntersect_PathOp)
        c.drawPath(p, paint(col, 0.35 * a))
        c.drawPath(p, paint(col, 0.5 * a, stroke=8 * self.sc, blur=6 * self.sc))

    def light_segment(self, c, k, j, a, col=GOLD):
        """light segment j (0 = nearest the palm) of finger k"""
        if a <= 0:
            return
        f = self.fingers[k]
        u0, u1 = f["segs"][j], f["segs"][j + 1]
        x0, y0 = self.point(f, u0 + 0.04)
        x1, y1 = self.point(f, min(u1, 0.97) - 0.04)
        line(c, x0, y0, x1, y1, col, 0.45 * a, w=f["w"] * 0.62, cap="butt")
        mx, my = self.point(f, (u0 + u1) / 2)
        L.dot_glow(c, mx, my, 5 * self.sc, CREAM, a, halo=4)


# ================================================================== 1. open
TEN_ROWS = [("1 km", "1,000 m"), ("1 m", "100 cm"), ("1 kg", "1,000 g"), ("1 dollar", "100 cents")]
TIME_ROWS = [("1 minute", "60", "seconds"), ("1 hour", "60", "minutes"), ("1 day", "24", "hours")]


class Open(L.Section):
    name, dur = "open", 28.0

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.004, pole=(960, -300))
        cx, cy, r = 960, 420, 250
        T = 11 * 3600 + 59 * 60 + 44 + t                         # 12:00:00 lands at t = 16
        sec = tick_frac(T)
        hfr, mfr, sfr = (T / 3600 % 12) / 12, (T / 60 % 60) / 60, (sec % 60) / 60
        dim = 1 - 0.55 * smooth((t - 22.4) / 1.2)
        L.clock(c, cx, cy, r, hfr, mfr, sfr, a=dim)
        # the columns: tens on the left, time on the right
        a_cols = 1 - smooth((t - 15.6) / 1.0)
        for k, (lhs, rhs) in enumerate(TEN_ROWS):
            a = fade(t, 1.6 + 0.6 * k, 16.6, 0.5, 0.01) * a_cols
            y = 316 + 72 * k
            text(c, 440, y, lhs + " = ", 34, CREAM_DIM, a, kind="sans", anchor="rs")
            text(c, 440, y, rhs, 34, CYAN, a, kind="sans-m", anchor="ls")
        for k, (lhs, num, unit) in enumerate(TIME_ROWS):
            a = fade(t, 8.6 + 1.0 * k, 16.6, 0.5, 0.01) * a_cols
            y = 330 + 84 * k
            x = 1310
            text(c, x, y, lhs + " = ", 34, CREAM_DIM, a, kind="sans", anchor="ls")
            x += L.text_width(lhs + " = ", 34, "sans")
            text(c, x, y, num, 44, GOLD, a, kind="sans-sb", anchor="ls")
            x += L.text_width(num, 44, "sans-sb") + 10
            text(c, x, y, unit, 34, CREAM_DIM, a, kind="sans", anchor="ls")
        # where they come from: the clock's 60 marks (Mesopotamia), its hours (Egypt)
        a_m = fade(t, 16.4, 23.2, 0.8, 0.8)
        if a_m > 0:
            ticks(c, cx, cy, r * 0.965, 60, r * 0.05, CLAY, a_m, w=2.4)
            L.arc(c, cx, cy, r * 1.06, 0, 360, CLAY, a_m * 0.5, w=1.2)
            text(c, cx - r - 60, cy - 40, "60s", 46, CLAY, a_m, kind="sans-sb", anchor="rs")
            text(c, cx - r - 60, cy + 6, "Mesopotamia", 30, CREAM, a_m, anchor="rs")
            a_e = fade(t, 17.2, 23.2, 0.8, 0.8)
            for k in range(12):
                x, y = polar(cx, cy, r * 0.79, 30 * k)
                L.dot_glow(c, x, y, 0.1, LAPIS, a_e * 0.8, halo=r * 0.9)
                text(c, x, y, L.ROMAN[k], r * 0.13, mixc(LAPIS, CREAM, 0.35), a_e, anchor="mm")
            text(c, cx + r + 60, cy - 40, "12 + 12 hours", 46, LAPIS, a_e, kind="sans-sb", anchor="ls")
            text(c, cx + r + 60, cy + 6, "Egypt", 30, CREAM, a_e, anchor="ls")
        a_t = fade(t, 23.2, 28.5, 1.0, 0.01)
        if a_t > 0:
            text(c, 960, 806, "Why 60? Why 24?", 92, CREAM, a_t, weight=560, anchor="ms")
            text(c, 960, 866, "and why is almost everything else counted in tens?", 32, CREAM_DIM, a_t, kind="italic", anchor="ms")
        caps(c, t, [(1.0, 7.6, ["Almost everything we measure, we count in tens."]),
                    (8.2, 15.2, ["Except time: 60 seconds, 60 minutes, 24 hours."]),
                    (15.8, 22.6, ["The sixties come from ancient Mesopotamia, the 24 from ancient Egypt.",
                                  "It took Greek astronomers to put them together."])])
        L.blackout(c, 1 - smooth(t / 1.4))


# ================================================================== 2. tens
def counter_box(c, x, y, w, h, v0, v1, u, a=1.0, col=CREAM, size=110, fmt=str):
    """a digit window rolling from v0 to v1 (u: 0..1)"""
    if a <= 0:
        return
    r = skia.Rect.MakeXYWH(x, y, w, h)
    c.drawRoundRect(r, 18, 18, paint((14, 19, 38), a))
    c.drawRoundRect(r, 18, 18, paint(GOLD_DIM, a, stroke=2))
    c.save()
    c.clipRect(r, doAntiAlias=True)
    off = ease(u) * h
    ym = y + h / 2
    if v0 != v1 and u > 0:
        text(c, x + w / 2, ym - off, fmt(v0), size, col, a, kind="sans-m", anchor="mm")
        text(c, x + w / 2, ym + h - off, fmt(v1), size, col, a, kind="sans-m", anchor="mm")
    else:
        text(c, x + w / 2, ym, fmt(v1 if u >= 1 else v0), size, col, a, kind="sans-m", anchor="mm")
    c.restore()


class Tens(L.Section):
    name, dur = "tens", 23.0
    COUNT0 = 1.0                                                  # fingers light from here, every 0.5 s

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.004, pole=(960, -300))
        # ten fingers
        a_h = fade(t, 0.2, 8.4, 0.8, 0.8)
        if a_h > 0:
            hl, hr = Hand(700, 520, 1.15, thumb="right"), Hand(1220, 520, 1.15, thumb="left")
            hl.draw(c, a_h)
            hr.draw(c, a_h)
            n = int(clamp((t - self.COUNT0) / 0.5 + 1, 0, 10))
            order = [(hl, 0), (hl, 1), (hl, 2), (hl, 3), (hl, 4), (hr, 4), (hr, 3), (hr, 2), (hr, 1), (hr, 0)]
            for k in range(n):
                h_, f_ = order[k]
                h_.light_finger(c, f_, a_h * smooth((t - self.COUNT0 - 0.5 * k) / 0.2))
            if n > 0:
                text(c, 960, 190, str(n), 96, GOLD, a_h, kind="sans-sb", anchor="ms")
        # the place-value window: 8, 9, 10 ...
        a_b = fade(t, 8.4, 23.5, 0.8, 0.01)
        if a_b > 0:
            bx, by, bw, bh = 960 - 250, 300, 230, 250
            if t < 15.2:
                seq, t0, step = [8, 9, 10], 9.4, 1.6
                k = int(clamp((t - t0) / step, 0, len(seq) - 1))
                u = clamp((t - t0 - k * step) / 0.35)
                v_prev = seq[max(k - 1, 0)]
                v = seq[k]
                if k == 0:
                    v_prev, u = v, 1
                tens0, ones0, tens1, ones1 = divmod(v_prev, 10) + divmod(v, 10)
                counter_box(c, bx, by, bw, bh, tens0, tens1, u if tens0 != tens1 else 1, a_b, col=GOLD if v >= 10 else CREAM_DIM)
                counter_box(c, bx + bw + 40, by, bw, bh, ones0, ones1, u if ones0 != ones1 else 1, a_b)
                text(c, bx + bw / 2, by + bh + 56, "tens", 36, CREAM_DIM, a_b, anchor="ms")
                text(c, bx + bw * 1.5 + 40, by + bh + 56, "ones", 36, CREAM_DIM, a_b, anchor="ms")
            else:
                a2 = smooth((t - 15.2) / 0.6)
                seq = [57, 58, 59, 60, 61]
                t0, step = 16.6, 1.0
                k = int(clamp((t - t0) / step, 0, len(seq) - 1))
                u = clamp((t - t0 - k * step) / 0.35)
                v_prev, v = seq[max(k - 1, 0)], seq[k]
                if k == 0:
                    v_prev, u = v, 1
                s0, o0 = divmod(v_prev, 60)
                s1, o1 = divmod(v, 60)
                counter_box(c, bx, by, bw, bh, s0, s1, u if s0 != s1 else 1, a_b, col=CLAY if v >= 60 else CREAM_DIM)
                counter_box(c, bx + bw + 40, by, bw, bh, o0, o1, u if o0 != o1 else 1, a_b)
                text(c, bx + bw / 2, by + bh + 56, "sixties", 36, CLAY, a_b * a2, anchor="ms")
                text(c, bx + bw * 1.5 + 40, by + bh + 56, "ones", 36, CREAM_DIM, a_b, anchor="ms")
                if v >= 60:
                    text(c, 960, by + bh + 130, f"= {v}", 40, CREAM, a_b * smooth((t - t0 - 3 * step - 0.3) / 0.4), kind="sans", anchor="ms")
        caps(c, t, [(0.8, 7.6, ["We count in tens because we have ten fingers."]),
                    (8.2, 14.8, ["Run out of fingers and you start a new column:", "9, then 10. One ten, no ones."]),
                    (15.4, 22.8, ["About 4,000 years ago, scribes in Mesopotamia", "started their new column at 60 instead."])])


# ================================================================== 3. clay
def stylus(c, x, y, lift, a=1.0):
    """a reed stylus seen from above, its cut tip at (x, y); lift 0 = pressing, 1 = raised"""
    if a <= 0:
        return
    ang = math.radians(-38)
    dx, dy = math.cos(ang), math.sin(ang)
    ln = 900
    sh = 10 + 70 * lift
    for pass_ in ("shadow", "reed"):
        ox, oy = (sh * 0.7, sh) if pass_ == "shadow" else (0, -lift * 18)
        sc = 1 + 0.07 * lift if pass_ == "reed" else 1.0
        x0, y0 = x + ox, y + oy
        x1, y1 = x0 + dx * ln * sc, y0 + dy * ln * sc
        nx, ny = -dy, dx
        w0, w1 = 9 * sc, 15 * sc
        pts = [(x0 + nx * w0 * 0.2, y0 + ny * w0 * 0.2), (x0 + dx * 40 + nx * w0, y0 + dy * 40 + ny * w0), (x1 + nx * w1, y1 + ny * w1),
               (x1 - nx * w1, y1 - ny * w1), (x0 + dx * 40 - nx * w0, y0 + dy * 40 - ny * w0), (x0 - nx * w0 * 0.2, y0 - ny * w0 * 0.2)]
        if pass_ == "shadow":
            path = skia.Path()
            path.addPoly([skia.Point(*p) for p in pts], True)
            c.drawPath(path, paint((0, 0, 0), 0.45 * a, blur=6 + 10 * lift))
        else:
            path = skia.Path()
            path.addPoly([skia.Point(*p) for p in pts], True)
            p = skia.Paint(AntiAlias=True)
            p.setShader(skia.GradientShader.MakeLinear([skia.Point(x0 + nx * 14, y0 + ny * 14), skia.Point(x0 - nx * 14, y0 - ny * 14)],
                                                       [L.C4((232, 208, 150), a).toColor(), L.C4((150, 118, 70), a).toColor()]))
            c.drawPath(path, p)
            # nodes on the reed
            for u in (0.45, 0.82):
                xa, ya = x0 + dx * ln * sc * u, y0 + dy * ln * sc * u
                w_ = w0 + (w1 - w0) * u
                line(c, xa + nx * w_, ya + ny * w_, xa - nx * w_, ya - ny * w_, (120, 92, 52), a * 0.8, w=2.5)


def press_plan(marks, t0, gap=0.42, first_lead=0.9):
    """times at which each mark is pressed"""
    return [(t0 + k * gap, m) for k, m in enumerate(marks)]


def pressed(plan, t, dur=0.14):
    out = []
    for tp, m in plan:
        f = smooth((t - tp) / dur)
        if f > 0:
            out.append(tuple(m) + (f,))
    return out


def stylus_state(plan, t, ox, oy, lead=0.7, leave=0.8):
    """(x, y, lift, alpha) of the stylus following a press plan; tablet offset (ox, oy)"""
    if not plan:
        return None
    pts = []
    for tp, m in plan:
        kind, x, y, s = m[:4]
        if kind == "w":
            pts.append((tp, ox + x, oy + y + 0.12 * s))
        else:
            pts.append((tp, ox + x + 0.1 * s, oy + y))
    t_first, t_last = pts[0][0], pts[-1][0]
    if t < t_first - lead or t > t_last + leave:
        return None
    if t < t_first:
        u = (t - (t_first - lead)) / lead
        x, y = pts[0][1] + 260 * (1 - ease(u)), pts[0][2] - 200 * (1 - ease(u))
        return x, y, 1 - ease(u) * 0.9 + 0.0, smooth(u * 2)
    for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
        if ta <= t < tb:
            v = (t - ta) / (tb - ta)
            if v < 0.3:
                return xa, ya, 0.1 * (1 - smooth(v / 0.15)) if v < 0.15 else 0.0, 1.0
            w = (v - 0.3) / 0.7
            return lerp(xa, xb, ease(w)), lerp(ya, yb, ease(w)), 0.35 * math.sin(math.pi * w) + 0.1 * smooth((w - 0.7) / 0.3), 1.0
    u = (t - t_last) / leave
    if u < 0.25:
        return pts[-1][1], pts[-1][2], 0.0, 1.0
    w = (u - 0.25) / 0.75
    return pts[-1][1] + 300 * ease(w), pts[-1][2] - 240 * ease(w), ease(w), 1 - smooth(w)


TAB_A = (370, 140, 1180, 430)                                       # x, y, w, h
YBC = (330, 130, 560)                                               # x, y, diameter


def _numbers_row(items, y, u, x0, gap, centre=None):
    """lay out several numbers in a row: returns (marks per item, centre x per item). centre: centre the row there."""
    if centre is not None:
        tot = sum(clay.number_marks(d, 0, 0, u=u)[1] for d in items) + gap * (len(items) - 1)
        x0 = centre - tot / 2
    out, cxs, x = [], [], x0
    for digs in items:
        ms, w, _ = clay.number_marks(digs, x, y, u=u)
        out.append(ms)
        cxs.append(x + w / 2)
        x += w + gap
    return out, cxs


class Clay(L.Section):
    name, dur = "clay", 56.0

    def __init__(self):
        x, y, w, h = TAB_A
        self.tab = clay.Tablet(w, h, seed=5)
        self.tab2 = clay.Tablet(w, h, seed=8)
        # tablet 1: 1, 10, 23, 59
        self.row1, self.cx1 = _numbers_row([[1], [10], [23], [59]], h * 0.47, 76, 0, 100, centre=w / 2)
        self.plan1 = (press_plan(self.row1[0], 2.6) + press_plan(self.row1[1], 9.0) +
                      press_plan(self.row1[2], 15.6, gap=0.3) + press_plan(self.row1[3], 17.8, gap=0.22))
        # tablet 2: 1,15 (= 75), then a lone wedge
        ms, wd, spans = clay.number_marks([1, 15], 200, h * 0.47, u=84, gap=110)
        self.n75, self.spans75 = ms, spans
        self.plan2 = press_plan(ms, 25.0, gap=0.34)
        self.lone, _, _ = clay.number_marks([1], 880, h * 0.47, u=84)
        self.plan3 = press_plan(self.lone, 33.6)
        # YBC 7289
        d = YBC[2]
        self.ybc = clay.Tablet(d, d, shape="round", seed=11, edge=40, colour=(186, 146, 104))
        R, cx, cy = 185, d / 2, d / 2
        V = [(cx - R, cy), (cx, cy - R), (cx + R, cy), (cx, cy + R)]
        self.ybc.extra = [clay.scored_line(V[i], V[(i + 1) % 4], 1.5, 1.6) for i in range(4)] + \
                         [clay.scored_line(V[0], V[2], 1.4, 1.5), clay.scored_line(V[1], V[3], 1.4, 1.5)]
        top, w1, sp1 = clay.number_marks([1, 24, 51, 10], 0, cy - 30, u=34, gap=16)
        bot, w2, sp2 = clay.number_marks([42, 25, 35], 0, cy + 34, u=34, gap=16)
        thirty, w3, _ = clay.number_marks([30], 0, 0, u=34)
        self.ybc_top = [(k, x + cx - w1 / 2, y, s) for k, x, y, s in top]
        self.ybc_bot = [(k, x + cx - w2 / 2, y, s) for k, x, y, s in bot]
        self.ybc_30 = [(k, x + cx - R * 0.62 - w3 / 2 - 22, y + cy - R * 0.5 - 22, s) for k, x, y, s in thirty]
        self.sp_top = [(a + cx - w1 / 2, b + cx - w1 / 2) for a, b in sp1]

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        x, y, w, h = TAB_A
        # --- tablet 1
        a1 = fade(t, 0.0, 23.6, 1.0, 0.8)
        if a1 > 0:
            self.tab.draw(c, x, y, pressed(self.plan1, t), a1)
            labs = [("1", 2.8), ("10", 9.2), ("23", 16.8), ("59", 20.2)]
            for k, (s_, t_) in enumerate(labs):
                text(c, x + self.cx1[k], y + h + 70, s_, 48, GOLD, a1 * smooth((t - t_) / 0.5), kind="sans-m", anchor="ms")
            sub = [(16.8, "2 tens + 3 ones"), (20.2, "5 tens + 9 ones")]
            for k, (t_, s_) in enumerate(sub):
                text(c, x + self.cx1[2 + k], y + h + 112, s_, 26, CREAM_DIM, a1 * smooth((t - t_) / 0.5), anchor="ms")
            st = stylus_state(self.plan1, t, x, y)
            if st:
                stylus(c, st[0], st[1], st[2], st[3] * a1)
        # --- tablet 2: 75 = 1,15; then a lone wedge
        a2 = fade(t, 23.4, 41.0, 0.8, 0.9)
        if a2 > 0:
            self.tab2.draw(c, x, y, pressed(self.plan2, t) + pressed(self.plan3, t), a2)
            a_l = a2 * smooth((t - 26.6) / 0.6)
            (s0, s1), (u0, u1) = self.spans75
            text(c, x + (s0 + s1) / 2, y + h + 64, "1 × 60", 40, CLAY, a_l, kind="sans-m", anchor="ms")
            text(c, x + (u0 + u1) / 2, y + h + 64, "15", 40, CREAM, a_l, kind="sans-m", anchor="ms")
            text(c, x + (s0 + u1) / 2, y + h + 118, "= 75", 40, GOLD, a2 * smooth((t - 27.6) / 0.6), kind="sans-sb", anchor="ms")
            a_z = a2 * smooth((t - 34.6) / 0.6)
            lx = x + 880 + 30
            for k, s_ in enumerate(["1?", "60?", "3,600?"]):
                text(c, lx, y + h + 64 + 46 * k, s_, 38, CREAM if k == 0 else CLAY, a_z * smooth((t - 34.8 - 0.7 * k) / 0.4),
                     kind="sans-m", anchor="ms")
            st = stylus_state(self.plan2 + self.plan3, t, x, y)
            if st and t < 30.5:
                stylus(c, st[0], st[1], st[2], st[3] * a2)
            st = stylus_state(self.plan3, t, x, y)
            if st and t > 30.5:
                stylus(c, st[0], st[1], st[2], st[3] * a2)
        # --- YBC 7289
        a3 = fade(t, 41.0, 56.5, 1.0, 0.01)
        if a3 > 0:
            bx, by, d = YBC
            marks = self.ybc_top + self.ybc_bot + self.ybc_30
            self.ybc.draw(c, bx, by, marks, a3)
            text(c, bx + d / 2, by + d + 50, "YBC 7289, Yale Babylonian Collection", 24, CREAM_DIM, a3, kind="italic", anchor="ms")
            # the reading
            a_r = a3 * smooth((t - 42.4) / 0.6)
            cols = [CREAM, CLAY, GOLD, LAPIS]
            for k, (s0, s1) in enumerate(self.sp_top):
                yy = by + d / 2 - 30 - 24
                line(c, bx + s0, yy - 2, bx + s1, yy - 2, cols[k], a_r * 0.9, w=2.6, glow=0.4)
            X0, Y0 = 1030, 320
            dig = ["1", "24", "51", "10"]
            xx = X0
            for k, s_ in enumerate(dig):
                text(c, xx, Y0, s_, 60, cols[k], a_r, kind="sans-sb", anchor="ls")
                xx += L.text_width(s_, 60, "sans-sb") + 8
                if k == 0:
                    text(c, xx, Y0, ";", 60, CREAM_DIM, a_r, kind="sans", anchor="ls")
                    xx += L.text_width(";", 60, "sans") + 14
                elif k < 3:
                    text(c, xx, Y0, ",", 60, CREAM_DIM, a_r, kind="sans", anchor="ls")
                    xx += L.text_width(",", 60, "sans") + 14
            a_e = a3 * smooth((t - 47.0) / 0.6)
            terms = [("1", CREAM), (" + 24/60", CLAY), (" + 51/60²", GOLD), (" + 10/60³", LAPIS)]
            xx = X0
            for s_, col in terms:
                text(c, xx, Y0 + 100, s_, 40, col, a_e, kind="sans-m", anchor="ls")
                xx += L.text_width(s_, 40, "sans-m")
            text(c, X0, Y0 + 170, "= 1.414212 96…", 46, CREAM, a3 * smooth((t - 48.2) / 0.6), kind="sans-m", anchor="ls")
            text(c, X0, Y0 + 240, "√2 = 1.414213 56…", 46, GOLD, a3 * smooth((t - 50.4) / 0.6), kind="sans-m", anchor="ls")
            text(c, X0, Y0 + 300, "off by 0.000 000 6", 32, CREAM_DIM, a3 * smooth((t - 51.4) / 0.6), kind="italic", anchor="ls")
        caps(c, t, [(0.6, 7.6, ["They wrote by pressing a cut reed into soft clay."]),
                    (8.2, 14.8, ["One wedge meant 1. A corner wedge meant 10."]),
                    (15.4, 22.8, ["Any number up to 59 is a pile of tens and ones."]),
                    (23.6, 31.8, ["At 60 a new column starts, just as ours does at 10.", "1 sixty and 15 ones: 75."]),
                    (32.4, 40.4, ["There was no zero yet. A lone wedge could mean 1, or 60, or 3,600:",
                                  "the reader worked it out from context."]),
                    (41.4, 48.6, ["A school tablet from about 1700 BC gives the square root of 2", "as 1;24,51,10."]),
                    (49.2, 55.8, ["That's right to within six ten-millionths."])])


# ================================================================== 4. why 60
DIVS = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30]


def bead_ring(c, cx, cy, r, n, groups, a, cols=(GOLD, CLAY), br=7.0, bad=None, spokes=True):
    """n beads on a ring, coloured in `groups` equal groups (0: plain); bad: indices of leftover beads (red)"""
    if a <= 0:
        return
    circle(c, cx, cy, r, GOLD_DIM, a * 0.5, w=1.2)
    size = n // groups if groups else n
    for k in range(n):
        x, y = polar(cx, cy, r, 360.0 * (k + 0.5) / n)
        if bad and k in bad:
            col = RED
        elif groups:
            col = cols[(k // size) % 2]
        else:
            col = CREAM_DIM
        circle(c, x, y, br, col, a)
    if groups and spokes:
        for g in range(groups):
            d = 360.0 * g * size / n
            x0, y0 = polar(cx, cy, r * 0.25, d)
            x1, y1 = polar(cx, cy, r * 1.12, d)
            line(c, x0, y0, x1, y1, CREAM, a * 0.55, w=1.4)


class Why60(L.Section):
    name, dur = "why60", 48.0
    D0, DSTEP = 9.0, 0.88

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        cx, cy, r = 640, 430, 250
        a_r = fade(t, 0.3, 33.4, 0.8, 0.8)
        if a_r > 0:
            k = int((t - self.D0) // self.DSTEP)
            clockmode = smooth((t - 25.0) / 0.8)
            if 0 <= k < len(DIVS) and t < 18.4:
                d = DIVS[k]
                u = clamp((t - self.D0 - k * self.DSTEP) / 0.2)
                bead_ring(c, cx, cy, r, 60, d, a_r * (0.4 + 0.6 * u))
                text(c, cx, cy - 6, f"÷ {d}", 84, GOLD, a_r, kind="sans-sb", anchor="ms")
                text(c, cx, cy + 56, f"{60 // d} each", 40, CREAM, a_r, kind="sans", anchor="ms")
            elif t >= 18.4 and clockmode <= 0:
                bead_ring(c, cx, cy, r, 60, 0, a_r)
                text(c, cx, cy + 20, "60", 100, GOLD, a_r * 0.6, kind="sans-sb", anchor="ms")
            elif clockmode > 0:
                bead_ring(c, cx, cy, r, 60, 0, a_r * (1 - clockmode))
                L.clock(c, cx, cy, r, None, 20 / 60, None, a=a_r * clockmode, hand_a=(0, 1, 0), numeral_a=0.0)
                sector(c, cx, cy, 0, r * 0.95, 0, 120 * smooth((t - 26.0) / 1.2), GOLD, a_r * clockmode * 0.32)
                text(c, cx, cy + r + 70, "⅓ hour = 20 minutes", 44, GOLD, a_r * smooth((t - 26.6) / 0.6), kind="sans-m", anchor="ms")
            else:
                bead_ring(c, cx, cy, r, 60, 0, a_r)
                text(c, cx, cy + 20, "60", 100, GOLD, a_r * 0.6, kind="sans-sb", anchor="ms")
        # ten
        a_t = fade(t, 18.4, 25.2, 0.7, 0.6)
        if a_t > 0:
            tx, ty, tr = 1380, 430, 150
            ph = [(18.6, 2, None), (20.0, 5, None), (21.6, 3, {9})]
            cur = [p for p in ph if t >= p[0]]
            if cur:
                t0, d, bad = cur[-1]
                if bad:
                    bead_ring(c, tx, ty, tr, 10, 3, a_t, br=9, bad=bad)
                    text(c, tx, ty - 4, "÷ 3", 64, CREAM, a_t, kind="sans-sb", anchor="ms")
                    text(c, tx, ty + 44, "3.333…", 34, RED, a_t, kind="sans", anchor="ms")
                else:
                    bead_ring(c, tx, ty, tr, 10, d, a_t, br=9)
                    text(c, tx, ty - 4, f"÷ {d}", 64, GOLD, a_t, kind="sans-sb", anchor="ms")
                    text(c, tx, ty + 44, f"{10 // d} each", 34, CREAM, a_t, kind="sans", anchor="ms")
            else:
                bead_ring(c, tx, ty, tr, 10, 0, a_t, br=9)
        # a third of a metre
        a_m = fade(t, 25.4, 33.4, 0.7, 0.8)
        if a_m > 0:
            x0, x1, y = 1100, 1780, 470
            line(c, x0, y, x1, y, CREAM, a_m, w=2)
            for k in range(101):
                h = 26 if k % 10 == 0 else 14 if k % 5 == 0 else 8
                xx = x0 + (x1 - x0) * k / 100
                line(c, xx, y, xx, y - h, CREAM_DIM, a_m, w=1.2)
                if k % 10 == 0:
                    text(c, xx, y + 34, str(k), 20, CREAM_DIM, a_m, kind="sans", anchor="ms")
            u = smooth((t - 26.6) / 1.2)
            xm = x0 + (x1 - x0) * (100 / 3) / 100
            c.drawRect(skia.Rect.MakeLTRB(x0, y - 40, x0 + (xm - x0) * u, y - 2), paint(CYAN, 0.22 * a_m))
            line(c, xm, y - 70, xm, y + 4, RED, a_m * u, w=2.4)
            text(c, xm, y - 86, "33.333… cm", 40, RED, a_m * smooth((t - 27.8) / 0.5), kind="sans-m", anchor="ms")
            text(c, (x0 + x1) / 2, y + 100, "⅓ metre", 44, CYAN, a_m * smooth((t - 27.8) / 0.5), kind="sans-m", anchor="ms")
        # the knuckle theory
        a_k = fade(t, 33.6, 48.5, 0.8, 0.01)
        if a_k > 0:
            hl, hr = Hand(640, 520, 1.2, thumb="right"), Hand(1280, 520, 1.2, thumb="left")
            hl.draw(c, a_k, creases=0.6)
            hr.draw(c, a_k, creases=1.0)
            T0, step = 35.0, 0.135
            n = int(clamp((t - T0) / step + 1, 0, 60))
            if t >= T0:
                cur = n - 1
                dz, inner = divmod(cur, 12)
                fk, sj = divmod(inner, 3)
                finger = 3 - fk                                   # index finger first, then middle, ring, pinky
                seg = 2 - sj                                      # tip first
                hr.light_segment(c, finger, seg, a_k)
                done = n // 12 if n < 60 else 5
                order = [3, 2, 1, 0, 4]
                for k in range(done):
                    hl.light_finger(c, order[k], a_k)
                text(c, 1280, 200, f"{n}", 80, GOLD, a_k, kind="sans-sb", anchor="ms")
                text(c, 640, 200, f"{done} dozen" + ("s" if done != 1 else ""), 48, CLAY, a_k * (1 if done else 0.0), kind="sans-m", anchor="ms")
            text(c, 960, 860 - 40, "12 × 5 = 60", 56, GOLD, a_k * smooth((t - 43.6) / 0.6), kind="sans-sb", anchor="ms")
            text(c, 960, 140, "one theory", 30, CREAM_DIM, a_k * smooth((t - 34.0) / 0.6), kind="italic", anchor="ms")
        caps(c, t, [(0.6, 8.0, ["Why 60? Nobody knows for sure.", "But 60 is very easy to share."]),
                    (8.6, 17.8, ["60 splits evenly into 2, 3, 4, 5, 6, 10, 12, 15, 20 or 30 equal parts."]),
                    (18.4, 24.6, ["10 splits evenly only into 2 or 5."]),
                    (25.2, 32.8, ["A third of an hour is 20 minutes.", "A third of a metre is 33.333… centimetres."]),
                    (33.6, 41.4, ["One theory: point your thumb at the 12 finger bones of one hand,",
                                  "and count each dozen on the other hand's five fingers."]),
                    (42.0, 47.6, ["12 × 5 = 60. A nice story, but there's no ancient evidence for it."])])


# ================================================================== 5. the Babylonian sky
class Sky(L.Section):
    name, dur = "sky", 46.0
    CX, CY, R = 960, 450, 330

    def rot(self, t):
        """the sky's turn so far (radians): brisk, then slow while we watch one UŠ go by"""
        w0, w1 = math.radians(7.0), math.radians(0.25)
        k = smooth((t - 22.0) / 2.0) - smooth((t - 33.0) / 2.0)
        # integrate the speed numerically (cheap: 0.05 s steps)
        n = int(t / 0.05)
        acc = 0.0
        for i in range(n):
            tt = i * 0.05
            kk = smooth((tt - 22.0) / 2.0) - smooth((tt - 33.0) / 2.0)
            acc += (w0 + (w1 - w0) * kk) * 0.05
        return acc

    def draw(self, c, t):
        cx, cy, R = self.CX, self.CY, self.R
        rot = self.rot(t)
        L.sky(c, t, rot=-rot, pole=(cx, cy), centre=(cx, cy))
        a = fade(t, 0.4, 46.5, 0.8, 0.01)
        circle(c, cx, cy, R, GOLD, a * 0.9, w=1.8, glow=0.3)
        circle(c, cx, cy, 5, CREAM, a)
        # 12 double hours
        a12 = smooth((t - 8.6) / 1.0) * a
        for k in range(12):
            ak = a12 * smooth((t - 8.8 - 0.25 * k) / 0.4)
            if ak <= 0:
                continue
            sector(c, cx, cy, R * 0.72, R, 30 * k, 30 * (k + 1), CLAY, ak * (0.20 if k % 2 == 0 else 0.08))
            x0, y0 = polar(cx, cy, R * 0.72, 30 * k)
            x1, y1 = polar(cx, cy, R * 1.0, 30 * k)
            line(c, x0, y0, x1, y1, CLAY, ak, w=1.6)
            lab = str(k + 1) if t < 33.6 else "30°"
            x, y = polar(cx, cy, R + 42, 30 * k + 15)
            text(c, x, y, lab, 30, CLAY if t < 33.6 else GOLD, ak * (1 - fade(t, 33.0, 34.2, 0.6, 0.6)), kind="sans-m", anchor="mm")
        # 360 UŠ
        a360 = smooth((t - 16.6) / 0.6) * a
        if a360 > 0:
            fr = clamp((t - 16.8) / 3.0)
            ticks(c, cx, cy, R, 360, 12, GOLD, a360, w=0.9, frac=fr, major=(30, 30, 2.0))
        # one UŠ: one degree of the sky's turn = 4 minutes
        a1 = fade(t, 24.2, 33.0, 0.8, 0.8)
        if a1 > 0:
            sector(c, cx, cy, 0, R * 1.0, -0.5, 0.5, GOLD, a1 * 0.6)
            x0, y0 = polar(cx, cy, R * 1.02, 0)
            text(c, x0, y0 - 18, "1 UŠ", 38, GOLD, a1, kind="sans-sb", anchor="ms")
            # a bright star crossing that degree, with a clock in sky minutes
            sx, sy = polar(cx, cy, R * 0.86, 0)
            ph = math.degrees(rot) - math.degrees(self.rot(25.4))
            # the star sits on the ring and turns with the sky (anticlockwise), starting just past the mark
            ang = 0.9 - ph
            if -3 < ang < 4:
                px, py = polar(cx, cy, R * 0.86, ang)
                L.dot_glow(c, px, py, 3.2, (255, 244, 220), a1, halo=6)
            mins = clamp((0.5 - ang) / 1.0) * 4.0
            text(c, cx + 170, cy - R * 0.62, f"{int(mins)}:{int(mins % 1 * 60):02d}", 46, CREAM, a1, kind="sans-m", anchor="ls")
            text(c, cx + 170, cy - R * 0.62 + 34, "minutes of sky time", 22, CREAM_DIM, a1, anchor="ls")
            text(c, cx, cy + 30, "1 UŠ = 1° = 4 minutes", 40, GOLD, a1 * smooth((t - 26.4) / 0.6), kind="sans-m", anchor="ms")
        # centre labels
        a_d = fade(t, 9.0, 24.0, 0.6, 0.6)
        text(c, cx, cy - 10, "1 day", 44, CREAM, a_d, kind="sans-m", anchor="ms")
        text(c, cx, cy + 34, "= 12 bēru", 32, CLAY, a_d * smooth((t - 9.6) / 0.5), kind="sans", anchor="ms")
        text(c, cx, cy + 72, "= 360 UŠ", 32, GOLD, a_d * smooth((t - 17.4) / 0.5), kind="sans", anchor="ms")
        a_z = fade(t, 34.2, 46.5, 0.8, 0.01)
        text(c, cx, cy - 4, "360°", 64, GOLD, a_z, kind="sans-sb", anchor="ms")
        text(c, cx, cy + 40, "12 signs × 30°", 30, CREAM, a_z, kind="sans", anchor="ms")
        caps(c, t, [(0.6, 7.8, ["Babylonian astronomers worked in sixties too.", "They watched the sky turn, once a day."]),
                    (8.4, 15.8, ["They divided the day into 12 “double hours”,", "called bēru."]),
                    (16.4, 23.6, ["Each double hour was 30 UŠ:", "360 in a day."]),
                    (24.2, 32.8, ["One UŠ was 4 of our minutes,", "the time the sky takes to turn one degree."]),
                    (33.6, 45.4, ["Later they split the sky's great circle the same way: 12 signs of 30 degrees.",
                                  "Greek astronomers took that 360° circle from Babylon. We still use it."])])


# ================================================================== 6. Egypt
HOR = 760.0                                                         # the horizon
POLE_E = (-840.0, 240.0)                                            # north celestial pole, off to the left (facing east)
EAST = (960.0, HOR)
R_EQ = math.hypot(EAST[0] - POLE_E[0], EAST[1] - POLE_E[1])
TH_E = math.atan2(EAST[1] - POLE_E[1], EAST[0] - POLE_E[0])
DECAN_STEP = math.radians(6.1)                                      # 10 degrees of sky between decans, on screen
THEBES_DAY = {"summer": 13.74, "winter": 10.53}                     # hours of daylight at 25.7 N (see ACCURACY.md)


def ground(c, a=1.0, day=0.0):
    """the desert horizon with two distant pyramids"""
    path = skia.Path()
    path.moveTo(0, HOR + 6)
    for x in range(0, 1921, 40):
        path.lineTo(x, HOR + 4 * math.sin(x / 210.0) + 3 * math.sin(x / 77.0 + 1))
    path.lineTo(1920, 1080)
    path.lineTo(0, 1080)
    path.close()
    col = mixc((7, 8, 14), (58, 46, 34), day)
    c.drawPath(path, paint(col, a))
    for ax, ay, b0, b1 in ((330, 688, 240, 420), (500, 716, 430, 570)):
        tri = skia.Path()
        tri.addPoly([skia.Point(b0, HOR + 2), skia.Point(ax, ay), skia.Point(b1, HOR + 2)], True)
        c.drawPath(tri, paint(mixc((10, 11, 19), (84, 66, 46), day), a))
        if day > 0:
            half = skia.Path()
            half.addPoly([skia.Point(ax, ay), skia.Point(b1, HOR + 2), skia.Point(ax + (b1 - ax) * 0.25, HOR + 2)], True)
            c.drawPath(half, paint((40, 30, 22), a * day * 0.6))
    line(c, 0, HOR + 1, 1920, HOR + 1, GOLD_DIM, a * 0.35, w=1.0)


def horizon_glow(c, col, a):
    p = skia.Paint()
    p.setDither(True)
    p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, HOR), skia.Point(0, HOR - 420)],
                                               [L.C4(col, a).toColor(), L.C4(col, 0).toColor()]))
    c.drawRect(skia.Rect.MakeLTRB(0, 0, 1920, HOR + 10), p)


def _decan_shapes():
    rng = np.random.default_rng(36)
    out = []
    for k in range(40):
        n = int(rng.integers(3, 6))
        pts = rng.normal(0, 1, (n, 2)) * np.array([18, 12])
        out.append((pts, rng.uniform(0.6, 1.0, n)))
    return out


DECANS = _decan_shapes()


def day_ring(c, cx, cy, r, span, a, labels=True, rin=None, cols=(GOLD, LAPIS)):
    """a 24-hour ring: 12 day hours over the top (span degrees of daylight), 12 night hours below"""
    rin = r * 0.70 if rin is None else rin
    d0 = -span / 2
    for k in range(12):
        sector(c, cx, cy, rin, r, d0 + span * k / 12, d0 + span * (k + 1) / 12, cols[0], a * (0.42 if k % 2 == 0 else 0.26))
    ns = 360 - span
    for k in range(12):
        sector(c, cx, cy, rin, r, span / 2 + ns * k / 12, span / 2 + ns * (k + 1) / 12, cols[1], a * (0.42 if k % 2 == 0 else 0.26))
    for k in range(12):
        for base, sp in ((d0, span), (span / 2, ns)):
            d = base + sp * k / 12
            x0, y0 = polar(cx, cy, rin, d)
            x1, y1 = polar(cx, cy, r, d)
            line(c, x0, y0, x1, y1, CREAM, a * 0.7, w=1.2)
    circle(c, cx, cy, r, CREAM, a * 0.5, w=1.2)
    circle(c, cx, cy, rin, CREAM, a * 0.5, w=1.2)


class Egypt(L.Section):
    name, dur = "egypt", 56.0
    RISE0 = 17.9                                                    # the first night decan rises

    def phi(self, t):
        """the sky's turn so far about the pole (radians): slow, then a time-lapse for the night hours"""
        w0, w1 = math.radians(0.9), math.radians(7.6)
        n = int(t / 0.05)
        acc = 0.0
        for i in range(n):
            tt = i * 0.05
            k = smooth((tt - 16.6) / 1.0) * (1 - smooth((tt - 27.8) / 1.0))
            acc += (w0 + (w1 - w0) * k) * 0.05
        return acc

    def night(self, c, t, a):
        ph = self.phi(t)
        p0 = self.phi(self.RISE0)
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, HOR)],
                                                   [L.C4(L.NIGHT0).toColor(), L.C4((22, 30, 58)).toColor()]))
        p.setDither(True)
        c.drawPaint(p)
        L.stars(c, t, rot=-ph, pole=POLE_E, a=a, clip=lambda x, y: clamp((HOR - 4 - y) / 30))
        horizon_glow(c, (40, 52, 96), 0.35 * a)
        # the band the decans travel along
        a_b = a * fade(t, 8.6, 27.6, 1.0, 0.8) * 0.5
        if a_b > 0:
            pts = []
            for j in range(80):
                ang = TH_E - math.radians(40) * j / 79
                pts.append((POLE_E[0] + R_EQ * math.cos(ang), POLE_E[1] + R_EQ * math.sin(ang)))
            poly(c, pts, LAPIS, a_b * 0.5, w=34)
            poly(c, pts, LAPIS, a_b, w=1.0)
        # decans: on the sky's equator, 6.1 screen-degrees apart; decan k reaches the horizon when ph = p0 + k step
        risen = 0
        for k in range(-5, 16):
            ang = TH_E - (ph - p0 - k * DECAN_STEP)
            x = POLE_E[0] + R_EQ * math.cos(ang)
            y = POLE_E[1] + R_EQ * math.sin(ang)
            if y > HOR + 30 or x > 2000 or y < -40:
                continue
            pts, br = DECANS[k + 5]
            vis = clamp((HOR - 4 - y) / 30)
            hour = 0 <= k < 12
            if hour and y < HOR - 4:
                risen = max(risen, k + 1)
            glow = 1.0 + 1.4 * math.exp(-max(0.0, HOR - y) / 40.0) if hour else 1.0
            for (dx, dy), b in zip(pts, br):
                if y + dy < HOR - 2:
                    L.dot_glow(c, x + dx, y + dy, 2.6 * b, (255, 240, 214), a * vis * b * min(glow, 1.6) * (1.0 if hour else 0.55), halo=5)
            if hour:
                al = a * vis * smooth((HOR - y) / 60) * (1 - smooth((t - 27.4) / 0.8))
                text(c, x + 46, y + 4, str(k + 1), 36, mixc(LAPIS, CREAM, 0.3), al, kind="sans-m", anchor="lm")
        ground(c, a)
        a_c = a * fade(t, 17.6, 28.2, 0.5, 0.6)
        if a_c > 0 and risen > 0:
            text(c, 1700, 150, str(risen), 96, LAPIS, a_c, kind="sans-sb", anchor="rs")
            text(c, 1700, 190, "hours of night", 28, CREAM, a_c, anchor="rs")
        text(c, 960, HOR + 50, "east", 24, CREAM_DIM, a * fade(t, 1.0, 27.0, 0.8, 0.8), kind="italic", anchor="ms")
        # 36 ten-day weeks
        a_s = a * fade(t, 8.4, 17.2, 0.6, 0.8)
        if a_s > 0:
            x0, y0 = 352, 96
            for k in range(41):
                epa = k >= 36
                w_ = 14 if epa else 26
                x = x0 + (k * 31 if not epa else 36 * 31 + (k - 36) * 18)
                on = smooth((t - 9.0 - 0.09 * k) / 0.25)
                col = CLAY if epa else LAPIS
                c.drawRoundRect(skia.Rect.MakeXYWH(x, y0, w_, 30), 4, 4, paint(col, a_s * (0.18 + 0.6 * on)))
            text(c, 960, y0 + 72, "36 decans × 10 days = 360 days,  + 5 extra days = 365", 30, CREAM, a_s * smooth((t - 12.8) / 0.6),
                 kind="sans", anchor="ms")

    def daytime(self, c, t, a):
        u = clamp((t - 29.0) / 7.2)                                 # sunrise to sunset
        sky_top = mixc((34, 52, 96), (64, 118, 186), math.sin(math.pi * u))
        sky_hor = mixc((150, 104, 92), (176, 196, 214), math.sin(math.pi * u) ** 0.5)
        p = skia.Paint()
        p.setDither(True)
        p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, HOR)],
                                                   [L.C4(sky_top, a).toColor(), L.C4(sky_hor, a).toColor()]))
        c.drawPaint(p)
        # the sun's path, facing south: 12 parts, the first and last are twilight
        cx, rx, ry = 960, 760, 470
        def P(v):
            return cx - rx * math.cos(math.pi * v), HOR - ry * math.sin(math.pi * v)
        for k in range(12):
            tw = k in (0, 11)
            pts = [P(k / 12 + (1 / 12) * j / 16) for j in range(17)]
            passed = smooth((u - k / 12) * 12 / 0.4) if u > k / 12 else 0.0
            col = (196, 170, 230) if tw else GOLD
            poly(c, pts, col, a * (0.45 + 0.55 * passed), w=3.0 if not tw else 2.0, glow=0.3 * passed)
            x, y = P(k / 12)
            dx, dy = x - cx, y - HOR
            nn = math.hypot(dx / rx ** 2, dy / ry ** 2)
            ox, oy = dx / rx ** 2 / nn, dy / ry ** 2 / nn
            line(c, x - ox * 10, y - oy * 10, x + ox * 10, y + oy * 10, CREAM, a * 0.7, w=1.6)
            mx, my = P((k + 0.5) / 12)
            mdx, mdy = mx - cx, my - HOR
            nm = math.hypot(mdx / rx ** 2, mdy / ry ** 2)
            lx, ly = mx + mdx / rx ** 2 / nm * 34, my + mdy / ry ** 2 / nm * 34
            lab = "twilight" if tw else str(k)
            text(c, lx, ly, lab, 20 if tw else 28, col, a * (0.4 + 0.6 * passed), kind="italic" if tw else "sans-m", anchor="mm")
        x, y = P(-0.02 + 1.04 * u)
        if y < HOR + 30:
            L.dot_glow(c, x, y, 22, SUN, a, halo=4.5)
        ground(c, a, day=math.sin(math.pi * u) * 0.8 + 0.2)
        # the shadow clock (inset, bottom right): a bar with a raised crossbar at the sun's end
        a_i = a * fade(t, 29.4, 37.2, 0.6, 0.6)
        if a_i > 0:
            gx0, gx1, gy, hp = 1250, 1800, 1000, 64
            c.drawRoundRect(skia.Rect.MakeLTRB(gx0 - 40, gy - 150, gx1 + 40, gy + 46), 18, 18, paint((8, 10, 20), a_i * 0.78))
            text(c, (gx0 + gx1) / 2, gy - 112, "shadow clock", 28, CREAM, a_i, kind="italic", anchor="ms")
            morning = u < 0.5
            elev = math.radians(max(2.0, 84 * math.sin(math.pi * u)))
            post = gx0 if morning else gx1
            sgn = 1 if morning else -1
            ln = min(gx1 - gx0 - 10, hp / math.tan(elev) * 2.4)
            c.drawRect(skia.Rect.MakeLTRB(gx0, gy, gx1, gy + 16), paint((150, 120, 80), a_i))
            c.drawRect(skia.Rect.MakeLTRB(post - 7, gy - hp, post + 7, gy), paint((180, 146, 96), a_i))
            c.drawRect(skia.Rect.MakeLTRB(post - 20, gy - hp - 8, post + 20, gy - hp), paint((180, 146, 96), a_i))
            c.drawRect(skia.Rect.MakeLTRB(min(post, post + sgn * ln), gy, max(post, post + sgn * ln), gy + 16), paint((30, 22, 16), a_i * 0.85))
            for hh in range(1, 6):
                v = hh / 12 if morning else 1 - hh / 12
                e = math.radians(max(2.0, 84 * math.sin(math.pi * v)))
                xx = post + sgn * min(gx1 - gx0 - 10, hp / math.tan(e) * 2.4)
                line(c, xx, gy - 6, xx, gy + 22, CREAM, a_i * 0.85, w=1.8)
        a_c = a * fade(t, 29.0, 37.0, 0.5, 0.6)
        n = min(10, max(0, int(u * 12) - 0)) if u > 1 / 12 else 0
        if a_c > 0:
            text(c, 1700, 150, str(min(n, 10)), 96, GOLD, a_c, kind="sans-sb", anchor="rs")
            text(c, 1700, 190, "hours of daylight", 28, CREAM, a_c, anchor="rs")
            text(c, 1700, 226, "+ 2 of twilight", 24, (196, 170, 230), a_c * smooth((u - 0.9) / 0.1), kind="italic", anchor="rs")

    def draw(self, c, t):
        a_n = 1 - smooth((t - 27.8) / 1.2)
        a_d = smooth((t - 27.8) / 1.2) * (1 - smooth((t - 36.8) / 1.0))
        if a_n > 0:
            self.night(c, t, 1.0)
        if a_d > 0:
            c.saveLayerAlpha(None, int(255 * a_d))
            self.daytime(c, t, 1.0)
            c.restore()
        a_r = smooth((t - 36.8) / 1.0)
        if a_r > 0:
            c.saveLayerAlpha(None, int(255 * a_r))
            L.sky(c, t, rot=t * 0.003, pole=(960, -300))
            cx, cy, r = 960, 430, 270
            s_ = ease((t - 45.0) / 2.0) - ease((t - 49.4) / 2.4) * 0
            span = 180.0
            if t > 45.0:
                span = lerp(180.0, THEBES_DAY["summer"] * 15, ease((t - 45.0) / 2.0))
            if t > 49.6:
                span = lerp(THEBES_DAY["summer"] * 15, THEBES_DAY["winter"] * 15, ease((t - 49.6) / 2.2))
            day_ring(c, cx, cy, r, span, 1.0)
            ticks(c, cx, cy, r + 26, 24, -10, CREAM_DIM, 0.6 * smooth((t - 45.0) / 1.0), w=1.2)
            x, y = polar(cx, cy, r + 46, 0)
            text(c, cx - r - 50, cy - 70, "12 hours", 44, GOLD, 1.0, kind="sans-sb", anchor="rs")
            text(c, cx - r - 50, cy - 30, "of day", 30, CREAM, 1.0, anchor="rs")
            text(c, cx - r - 50, cy + 120, "12 hours", 44, LAPIS, 1.0, kind="sans-sb", anchor="rs")
            text(c, cx - r - 50, cy + 160, "of night", 30, CREAM, 1.0, anchor="rs")
            text(c, cx, cy + 22, "24", 110, CREAM, 1.0 - smooth((t - 44.6) / 0.6), kind="sans-sb", anchor="ms")
            # seasonal hours
            a_s = smooth((t - 45.2) / 0.8)
            if a_s > 0:
                summer = 1 - smooth((t - 49.6) / 1.4)
                dh = (THEBES_DAY["summer"] if summer > 0.5 else THEBES_DAY["winter"]) * 5
                nh = (24 - (THEBES_DAY["summer"] if summer > 0.5 else THEBES_DAY["winter"])) * 5
                name = "Midsummer" if summer > 0.5 else "Midwinter"
                text(c, cx + r + 60, cy - 70, f"{name} at Thebes", 34, CREAM, a_s, anchor="ls")
                text(c, cx + r + 60, cy - 18, f"a day hour ≈ {dh:.0f} min", 34, GOLD, a_s, kind="sans-m", anchor="ls")
                text(c, cx + r + 60, cy + 28, f"a night hour ≈ {nh:.0f} min", 34, LAPIS, a_s, kind="sans-m", anchor="ls")
            c.restore()
        caps(c, t, [(0.6, 7.8, ["Meanwhile, Egyptian priests told time at night", "by watching stars rise."]),
                    (8.4, 16.8, ["They tracked 36 star groups, the decans.",
                                 "Each one marked a ten-day week of their 365-day year."]),
                    (17.4, 27.2, ["About 12 of them rose one after another in the full dark of night,", "so the star tables split the night into 12 hours."]),
                    (28.4, 36.4, ["By day, one ancient text counts 10 hours on a shadow clock,", "plus an hour of twilight at each end: 12."]),
                    (37.4, 44.2, ["12 hours of day + 12 hours of night = 24."]),
                    (44.8, 55.4, ["But these hours stretched and shrank with the seasons:",
                                  "each was a twelfth of that day's daylight, or of its night."])])


# ================================================================== 7. Greece: minutes and seconds
RULER_Y, RULER_W = 520.0, 1400.0
FOCUS = [0.5, 37.5 / 60, 37 / 60 + 21.5 / 3600, 37 / 60 + 21 / 3600 + 44.5 / 216000]
ZOOMS = [(29.0, 32.2), (38.4, 41.6), (45.6, 48.8)]                 # (start, end) of each x60 zoom


def zoom_state(t):
    Z, ctr = 1.0, FOCUS[0]
    for j, (t0, t1) in enumerate(ZOOMS):
        if t < t0:
            break
        u = clamp((t - t0) / (t1 - t0))
        Z0, Z1 = 60.0 ** j, 60.0 ** (j + 1)
        Z = Z0 * (Z1 / Z0) ** ease(u)
        s0 = (FOCUS[j + 1] - FOCUS[j]) * RULER_W * Z0               # where the target sits on screen at the start
        s = s0 * (1 - ease(u))
        ctr = FOCUS[j + 1] - s / (RULER_W * Z)
    return Z, ctr


def ruler(c, t, a):
    Z, ctr = zoom_state(t)
    X = lambda p: 960 + (p - ctr) * RULER_W * Z
    x0, x1 = max(-20.0, X(0.0)), min(1940.0, X(1.0))
    if x1 > x0:
        line(c, x0, RULER_Y, x1, RULER_Y, CREAM, a, w=2.0, cap="butt")
    for j in range(5):
        n = 60 ** j
        sp = RULER_W * Z / n
        if sp < 3:
            break
        al = a * clamp((sp - 3) / 8)
        h = min(120.0, 8 + 24 * math.log10(sp))
        p0 = ctr + (-30 - 960) / (RULER_W * Z)
        p1 = ctr + (1950 - 960) / (RULER_W * Z)
        k0, k1 = max(0, math.floor(p0 * n)), min(n, math.ceil(p1 * n))
        path = skia.Path()
        for k in range(k0, k1 + 1):
            if j > 0 and k % 60 == 0:
                continue
            xx = X(k / n)
            path.moveTo(xx, RULER_Y)
            path.lineTo(xx, RULER_Y - h)
        c.drawPath(path, paint(GOLD if j else CREAM, al, stroke=1.6 if j < 2 else 1.3, cap="butt"))
    return X


def bracket(c, xa, xb, y, s, a, col=CREAM, size=34):
    if a <= 0:
        return
    xa, xb = max(xa, 40), min(xb, 1880)
    poly(c, [(xa, y - 12), (xa, y), (xb, y), (xb, y - 12)], col, a, w=1.6)
    text(c, (xa + xb) / 2, y + 46, s, size, col, a, kind="sans-m", anchor="ms")


class Greece(L.Section):
    name, dur = "greece", 54.0

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        # Egypt's 24 and Babylon's 60 slide together
        a_r = fade(t, 0.3, 16.6, 0.8, 0.8)
        if a_r > 0:
            m = ease((t - 4.4) / 2.6)
            r = lerp(170, 240, m)
            ex, bx = lerp(560, 960, m), lerp(1360, 960, m)
            cy = 440
            span = lerp(206.0, 180.0, ease((t - 10.2) / 2.0)) if t < 12.5 else 180.0
            day_ring(c, ex, cy, r, span, a_r * (1 - 0.15 * m))
            ticks(c, bx, cy, r + 2, 60, -16, CLAY, a_r, w=1.6)
            circle(c, bx, cy, r + 2, CLAY, a_r * 0.6, w=1.2)
            text(c, ex, cy + r + 66, "Egypt: 24 hours", 34, LAPIS, a_r * (1 - smooth((t - 3.8) / 0.8)), kind="sans-m", anchor="ms")
            text(c, bx, cy + r + 66, "Babylon: sixties", 34, CLAY, a_r * (1 - smooth((t - 3.8) / 0.8)), kind="sans-m", anchor="ms")
            text(c, 960, cy - 4, "Greek astronomy", 40, CREAM, a_r * smooth((t - 6.6) / 0.8), anchor="ms")
            a_h = a_r * smooth((t - 9.0) / 0.6)
            text(c, 960, cy + 40, "seasonal hours" if t < 11.4 else "24 equal hours", 28,
                 GOLD, a_h * (1 - fade(t, 10.6, 12.2, 0.4, 0.4)), kind="italic", anchor="ms")
            if t > 11.0:
                for k in range(24):
                    x, y = polar(960, cy, r + 40, 15 * k)
                    text(c, x, y, str(k if k else 24), 20, CREAM_DIM, a_r * smooth((t - 11.6 - 0.03 * k) / 0.4), kind="sans", anchor="mm")
        # the hour, cut into sixtieths, and sixtieths of those
        a_u = fade(t, 16.4, 54.5, 0.8, 0.01)
        if a_u > 0:
            draw_in = smooth((t - 16.6) / 2.0)
            X = ruler(c, t, a_u * draw_in)
            Z, ctr = zoom_state(t)
            # highlight the unit we are about to zoom into
            for j, (h0, h1) in enumerate(((24.6, 38.8), (33.8, 46.0), (43.0, 54.5))):
                ah = a_u * fade(t, h0, h1, 0.6, 0.8)
                if ah > 0:
                    n = 60 ** (j + 1)
                    p = FOCUS[j + 1] - 0.5 / n
                    xa, xb = X(p), X(p + 1 / n)
                    c.drawRect(skia.Rect.MakeLTRB(xa, RULER_Y - 90, xb, RULER_Y), paint(GOLD, 0.30 * ah))
            # brackets naming what fills the screen
            names = [(17.4, 29.4, 0, "1 hour"), (31.8, 38.8, 1, "1 minute = 60 seconds"), (41.2, 46.0, 2, "1 second = 60 thirds"),
                     (48.4, 54.5, 3, "1 third = 60 fourths")]
            for t0, t1, j, s_ in names:
                ab = a_u * fade(t, t0, t1, 0.6, 0.5)
                if ab > 0:
                    n = 60 ** j
                    p = FOCUS[j] - 0.5 / n if j else 0.0
                    bracket(c, X(p), X(p + 1 / n), RULER_Y + 40, s_, ab, CREAM)
            # the Latin
            for t0, t1, latin, gloss, word in ((24.8, 33.2, "pars minuta prima", "“first small part”", "minute"),
                                               (34.0, 42.4, "pars minuta secunda", "“second small part”", "second")):
                al = a_u * fade(t, t0, t1, 0.6, 0.6)
                if al > 0:
                    text(c, 960, 210, latin, 64, CREAM, al, kind="italic", anchor="ms")
                    text(c, 960, 262, gloss, 32, CREAM_DIM, al * smooth((t - t0 - 0.8) / 0.6), anchor="ms")
                    text(c, 960, 352, "→  " + word, 60, GOLD, al * smooth((t - t0 - 3.0) / 0.6), kind="sans-sb", anchor="ms")
            al = a_u * fade(t, 43.0, 54.5, 0.6, 0.01)
            text(c, 960, 210, "thirds, fourths…", 64, CREAM, al, kind="italic", anchor="ms")
            text(c, 960, 262, "al-Bīrūnī, about AD 1000", 32, CREAM_DIM, al * smooth((t - 43.8) / 0.6), anchor="ms")
        caps(c, t, [(0.6, 7.8, ["Greek astronomers brought the two together:", "Egypt's hours and Babylon's sixties."]),
                    (8.4, 15.8, ["In the 2nd century BC, Hipparchus used 24 equal hours", "for his calculations, whatever the season."]),
                    (16.4, 23.8, ["And astronomers wrote fractions the Babylonian way:", "in sixtieths, and sixtieths of sixtieths."]),
                    (24.4, 32.8, ["In medieval Latin, the first cut was the pars minuta prima,", "the “first small part”: the minute."]),
                    (33.4, 42.0, ["The next was the pars minuta secunda,", "the “second small part”: the second."]),
                    (42.6, 53.4, ["Some kept going. Around AD 1000, al-Bīrūnī gave times", "in hours, minutes, seconds, thirds and fourths."])])


# ================================================================== 8. clocks
class Clocks(L.Section):
    name, dur = "clocks", 37.0
    CX, CY, R = 960, 360, 240

    def clock_time(self, t):
        """the displayed time (s): a time-lapse that slows to real time once there's a second hand"""
        n = int(t / 0.05)
        acc = 3 * 3600.0
        for i in range(n):
            tt = i * 0.05
            sp = 600.0 * (1 - smooth((tt - 16.4) / 1.0)) + 30.0 * (smooth((tt - 16.4) / 1.0) - smooth((tt - 25.0) / 0.8)) \
                 + 1.0 * smooth((tt - 25.0) / 0.8)
            acc += sp * 0.05
        return acc

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        cx, cy, R = self.CX, self.CY, self.R
        a_s = fade(t, 0.3, 8.6, 0.8, 0.8)
        if a_s > 0:
            span = 180 + 26 * math.sin(t * 0.9)
            day_ring(c, cx, cy, R, span, a_s)
            text(c, cx, cy + 12, "seasonal hours", 36, CREAM, a_s, kind="italic", anchor="ms")
        a_c = smooth((t - 8.2) / 1.0)
        if a_c > 0:
            T = self.clock_time(t)
            if t >= 26.0:                                       # real time from here: tick on the whole second
                T = math.ceil(self.clock_time(26.0)) - 26.0 + tick_frac(t)
            ha = (T / 3600 % 12) / 12
            ma = (T / 60 % 60) / 60
            sa = (T % 60) / 60
            L.clock(c, cx, cy, R, ha, ma, sa, a=a_c, hand_a=(1.0, smooth((t - 16.8) / 1.2), smooth((t - 25.6) / 1.0)),
                    minute_ticks=60 if t > 16 else 0)
            # the pendulum: a seconds pendulum, one swing per second
            a_p = smooth((t - 17.2) / 1.2)
            if a_p > 0:
                px, py = cx, cy + R + 14
                th = math.radians(9.0) * math.cos(math.pi * (t - 0.0))
                lp = 230
                bx, by = px + lp * math.sin(th), py + lp * math.cos(th)
                line(c, px, py, bx, by, GOLD, a_p, w=2.6)
                L.dot_glow(c, bx, by, 18, GOLD, a_p * 0.6, halo=2.2)
                circle(c, bx, by, 18, GOLD, a_p, w=2.4)
                circle(c, px, py, 5, CREAM, a_p)
        for t0, t1, big, small in ((8.6, 16.6, "1300s", "tower clocks: one hand"), (16.8, 25.4, "1656", "Huygens' pendulum clock"),
                                  (25.6, 37.5, "c. 1670", "seconds pendulum")):
            al = fade(t, t0, t1, 0.6, 0.6)
            text(c, 1560, 300, big, 76, GOLD, al, kind="sans-sb", anchor="ms")
            text(c, 1560, 350, small, 30, CREAM, al, anchor="ms")
        caps(c, t, [(0.6, 7.8, ["For centuries, most people lived by seasonal hours,", "read from sundials, water clocks and bells."]),
                    (8.4, 15.8, ["From the 1300s, mechanical clocks in European towns struck equal hours.",
                                 "Most had a single hand."]),
                    (16.4, 24.8, ["Minute hands only caught on after Christiaan Huygens' pendulum clock of 1656.",
                                  "It cut a good clock's error from about 15 minutes a day to about 15 seconds."]),
                    (25.4, 36.0, ["Second hands followed, with the seconds pendulum, from about 1670.",
                                  "Sixty minutes of sixty seconds were now built into the clock."])])


# ================================================================== 9. French decimal time
def hms(T):
    T = int(math.floor(T + 1e-6)) % 86400
    return f"{T // 3600:02d}:{T // 60 % 60:02d}:{T % 60:02d}"


def dec_hms(T):
    d = (T % 86400) / 86400 * 100000                                  # decimal seconds since midnight
    d = int(math.floor(d + 1e-6))
    return f"{d // 10000}:{d // 100 % 100:02d}:{d % 100:02d}"


class Decimal(L.Section):
    name, dur = "decimal", 55.0
    LAPSE0, LAPSE1 = 26.0, 32.0                                     # whole seconds, so the ticks stay on the beat

    def T(self, t):
        """clock time (s after midnight): real time, a time-lapse from midnight to noon, then real time again"""
        if t < self.LAPSE0:
            return t
        if t < self.LAPSE1:
            u = ease((t - self.LAPSE0) / (self.LAPSE1 - self.LAPSE0))
            return lerp(self.LAPSE0, 43200.0, u)
        return 43200.0 + (t - self.LAPSE1)

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        T = self.T(t)
        lapse = self.LAPSE0 < t < self.LAPSE1
        lx, rx, cy, r = 560, 1360, 390, 230
        # ordinary clock
        Ts = T if lapse else tick_frac(T)
        L.clock(c, lx, cy, r, (T / 3600 % 12) / 12, (T / 60 % 60) / 60, (Ts % 60) / 60, a=fade(t, 0.2, 55.5, 0.8, 0.01))
        text(c, lx, cy - r - 40, "our time", 32, CREAM, 1.0, anchor="ms")
        text(c, lx, cy + r + 64, hms(T), 50, CREAM, fade(t, 1.0, 55.5, 0.6, 0.01), kind="sans-m", anchor="ms")
        # decimal clock
        a_d = smooth((t - 8.6) / 1.0) * (1 - 0.6 * smooth((t - 35.4) / 1.4))
        if a_d > 0:
            D = (T % 86400) / 86400                                   # fraction of the day
            Dsec = D * 100000
            ds = Dsec if lapse else tick_frac(Dsec)
            L.clock(c, rx, cy, r, D * 10 % 10 / 10, (D * 1000 % 100) / 100, (ds % 100) / 100, a=a_d, n=10, numerals="decimal",
                    minute_ticks=100, col=CYAN)
            text(c, rx, cy - r - 40, "French decimal time, 1793", 32, CYAN, a_d, anchor="ms")
            text(c, rx, cy + r + 64, dec_hms(T), 50, CYAN, a_d, kind="sans-m", anchor="ms")
        # conversions
        for k, (s_, t0) in enumerate((("1 decimal hour = 2 h 24 min", 16.8), ("1 decimal minute = 1 min 26.4 s", 18.2),
                                      ("1 decimal second = 0.864 s", 19.6))):
            al = fade(t, t0, 25.4, 0.5, 0.6)
            text(c, 960, 790 + 44 * k, s_, 32, CREAM, al, kind="sans", anchor="ms")
        al = fade(t, 31.6, 34.4, 0.5, 0.6)
        text(c, 960, 810, "noon", 40, GOLD, al, kind="italic", anchor="ms")
        # suspended
        a_x = fade(t, 35.4, 55.5, 0.8, 0.01)
        if a_x > 0:
            c.save()
            c.rotate(-12, rx, cy)
            c.drawRoundRect(skia.Rect.MakeLTRB(rx - 270, cy - 46, rx + 270, cy + 46), 8, 8, paint(RED, a_x * 0.85, stroke=4))
            text(c, rx, cy + 11, "SUSPENDED  7 APRIL 1795", 30, RED, a_x, kind="sans-sb", anchor="ms")
            c.restore()
        caps(c, t, [(0.6, 7.8, ["In 1793, revolutionary France tried to make time", "decimal too, like its new metre."]),
                    (8.4, 15.8, ["10 hours a day,", "100 minutes an hour, 100 seconds a minute."]),
                    (16.4, 24.8, ["A decimal hour lasted 2 hours 24 minutes, a decimal minute 1 min 26.4 s,",
                                  "and a decimal second 0.864 s."]),
                    (25.4, 33.6, ["Midnight was 0 o'clock.", "Noon was 5."]),
                    (34.2, 42.6, ["It didn't last. In April 1795, France suspended decimal time,",
                                  "in the same law that brought in the metric system."]),
                    (43.2, 54.2, ["Changing meant new clocks, new habits, new tables for navigators and astronomers.",
                                  "And 24 and 60 split into halves, thirds and quarters. The old units won."])])


# ================================================================== 10. other days
def ring_babylon(c, cx, cy, r, a):
    for k in range(12):
        sector(c, cx, cy, r * 0.62, r, 30 * k, 30 * (k + 1), CLAY, a * (0.32 if k % 2 == 0 else 0.14))
    ticks(c, cx, cy, r, 360, r * 0.08, CLAY, a, w=0.7)
    circle(c, cx, cy, r, CLAY, a * 0.7, w=1.2)


def ring_china(c, cx, cy, r, a, n_ke=100):
    for k in range(12):
        sector(c, cx, cy, r * 0.78, r, 30 * k - 15, 30 * k + 15, RED, a * (0.34 if k % 2 == 0 else 0.14))
    ticks(c, cx, cy, r * 0.74, n_ke, r * 0.12, CREAM, a, w=0.9)
    circle(c, cx, cy, r, RED, a * 0.7, w=1.2)


def ring_india(c, cx, cy, r, a):
    for k in range(60):
        sector(c, cx, cy, r * 0.62, r, 6 * k, 6 * (k + 1), GOLD, a * (0.34 if k % 2 == 0 else 0.12))
    circle(c, cx, cy, r, GOLD, a * 0.7, w=1.2)


def ring_france(c, cx, cy, r, a):
    for k in range(10):
        sector(c, cx, cy, r * 0.62, r, 36 * k, 36 * (k + 1), CYAN, a * (0.34 if k % 2 == 0 else 0.12))
    ticks(c, cx, cy, r * 0.62, 100, -r * 0.06, CYAN, a, w=0.8)
    circle(c, cx, cy, r, CYAN, a * 0.7, w=1.2)


class Others(L.Section):
    name, dur = "others", 39.0

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        xs = [260, 610, 960, 1310, 1660]
        cy, r = 420, 128
        china96 = smooth((t - 15.0) / 1.2)
        focus = {2: (9.2, 18.8), 3: (19.4, 27.0)}
        rows = [("Babylon", "12 bēru, 360 UŠ", CLAY), ("Egypt", "12 + 12 seasonal hours", LAPIS),
                ("China", "12 shí + 100 kè" if china96 < 0.5 else "12 shí + 96 kè", RED), ("India", "60 ghaṭikā", GOLD),
                ("France, 1793", "10 decimal hours", CYAN)]
        for k, (name, sub, col) in enumerate(rows):
            a = smooth((t - 0.8 - 0.7 * k) / 0.8)
            dim = 1.0
            if 9.2 < t < 27.0:
                dim = 0.3
                for j, (f0, f1) in focus.items():
                    if f0 <= t <= f1 and j == k:
                        dim = 1.0
            a *= dim * (1 - smooth((t - 38.0) / 1.0) * 0)
            cx = xs[k]
            if k == 0:
                ring_babylon(c, cx, cy, r, a)
            elif k == 1:
                day_ring(c, cx, cy, r, 206, a, rin=r * 0.62)
            elif k == 2:
                ring_china(c, cx, cy, r, a * (1 - china96) if china96 < 1 else 0)
                if china96 > 0:
                    ring_china(c, cx, cy, r, a * china96, n_ke=96)
            elif k == 3:
                ring_india(c, cx, cy, r, a)
            else:
                ring_france(c, cx, cy, r, a)
            text(c, cx, cy - r - 36, name, 34, col, a, kind="sans-sb", anchor="ms")
            text(c, cx, cy + r + 52, sub, 24, CREAM, a, kind="sans", anchor="ms")
        # China close-up notes
        a_c = fade(t, 10.0, 18.8, 0.6, 0.6)
        if a_c > 0:
            text(c, 960, 712, "刻", 72, RED, a_c * china96, kind="cjk", anchor="ms")
            text(c, 960, 756, "kè: today, a quarter of an hour", 26, CREAM, a_c * china96, kind="italic", anchor="ms")
        a_i = fade(t, 20.0, 27.0, 0.6, 0.6)
        if a_i > 0:
            text(c, 1310, 712, "1 ghaṭikā = 24 min", 28, GOLD, a_i, kind="sans-m", anchor="ms")
            text(c, 1310, 750, "60 vighaṭikā of 24 s", 24, CREAM, a_i, kind="sans", anchor="ms")
        caps(c, t, [(0.6, 8.6, ["Other cultures cut the day their own ways."]),
                    (9.2, 18.6, ["For over 2,000 years China counted 100 kè a day, alongside 12 double hours.",
                                 "In the 1600s calendar reformers made it 96, so a kè became 15 minutes."]),
                    (19.2, 26.6, ["Indian astronomy split the day into 60 ghaṭikā of 24 minutes,", "and each of those into 60: base 60 again."]),
                    (27.2, 37.8, ["But the Greek blend is the one that spread: through Arabic and Latin astronomy,",
                                  "then Europe's clocks, to the rest of the world."])])


# ================================================================== 11. the ladder of time units
LADDER = [("1 millennium", "×10", CYAN), ("1 century", "×10", CYAN), ("1 decade", "×10", CYAN), ("1 year", "×365¼", CREAM_DIM),
          ("1 day", "×24", GOLD), ("1 hour", "×60", GOLD), ("1 minute", "×60", GOLD), ("1 second", "×1,000", CYAN),
          ("1 millisecond", "×1,000", CYAN), ("1 microsecond", "×1,000", CYAN), ("1 nanosecond", None, None)]


def globe(c, cx, cy, r, lon0, a, band=None, tilt=22.0):
    """a wire globe: meridians every 15 degrees, a few parallels; band: (lon_a, lon_b) to fill"""
    if a <= 0:
        return
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeRadial((cx - r * 0.35, cy - r * 0.4), r * 1.6,
                                               [L.C4((40, 54, 96), a).toColor(), L.C4((10, 14, 30), a).toColor()]))
    c.drawCircle(cx, cy, r, p)
    b = math.radians(tilt)

    def P(lat, lon):
        la, lo = math.radians(lat), math.radians(lon - lon0)
        x, y, z = math.cos(la) * math.sin(lo), math.sin(la), math.cos(la) * math.cos(lo)
        y2 = y * math.cos(b) - z * math.sin(b)
        z2 = y * math.sin(b) + z * math.cos(b)
        return cx + r * x, cy - r * y2, z2

    if band:
        pts = [P(lat, band[0]) for lat in range(-90, 91, 3)] + [P(lat, band[1]) for lat in range(90, -91, -3)]
        path = skia.Path()
        vis = [(x, y) for x, y, z in pts if z > -0.02]
        if len(vis) > 2:
            path.addPoly([skia.Point(x, y) for x, y in vis], True)
            c.save()
            c.clipPath(skia.Path().addCircle(cx, cy, r), doAntiAlias=True)
            c.drawPath(path, paint(GOLD, a * 0.35))
            c.restore()
    for lon in range(0, 360, 15):
        seg = []
        for lat in range(-90, 91, 3):
            x, y, z = P(lat, lon)
            if z > 0:
                seg.append((x, y))
            else:
                if len(seg) > 1:
                    poly(c, seg, GOLD, a * 0.75, w=1.2)
                seg = []
        if len(seg) > 1:
            poly(c, seg, GOLD, a * 0.75, w=1.2)
    for lat in (-60, -30, 0, 30, 60):
        seg = []
        for lon in range(0, 361, 4):
            x, y, z = P(lat, lon)
            if z > 0:
                seg.append((x, y))
            else:
                if len(seg) > 1:
                    poly(c, seg, CREAM_DIM, a * (0.7 if lat == 0 else 0.35), w=1.0)
                seg = []
        if len(seg) > 1:
            poly(c, seg, CREAM_DIM, a * (0.7 if lat == 0 else 0.35), w=1.0)
    circle(c, cx, cy, r, GOLD, a, w=1.8, glow=0.3)


class Ladder(L.Section):
    name, dur = "ladder", 47.0

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.003, pole=(960, -300))
        # the caesium count
        a_a = fade(t, 0.3, 10.4, 0.8, 0.8)
        if a_a > 0:
            ax, ay = 560, 420
            for k in range(3):
                c.save()
                c.rotate(60 * k + t * 20, ax, ay)
                c.drawOval(skia.Rect.MakeLTRB(ax - 190, ay - 62, ax + 190, ay + 62), paint(CYAN, a_a * 0.6, stroke=1.6))
                c.restore()
            L.dot_glow(c, ax, ay, 16, CREAM, a_a, halo=4)
            text(c, ax, ay + 240, "caesium-133", 36, CYAN, a_a, kind="sans-m", anchor="ms")
            u = smooth((t - 1.6) / 6.0)
            n = int(round(9192631770 * u))
            text(c, 1340, 420, f"{n:,}", 72, CREAM, a_a, kind="sans-sb", anchor="ms")
            text(c, 1340, 478, "cycles of its radiation = 1 second", 30, CREAM_DIM, a_a * smooth((t - 7.6) / 0.6), anchor="ms")
        # the ladder
        a_l = fade(t, 10.2, 33.6, 0.8, 0.8)
        if a_l > 0:
            y0, dy, x = 92, 66, 900
            for k, (name, f, col) in enumerate(LADDER):
                y = y0 + dy * k
                if k <= 3:
                    grp, tg = "top", 18.2
                elif k <= 7:
                    grp, tg = "mid", 25.2
                else:
                    grp, tg = "bottom", 10.6
                hi = smooth((t - tg) / 0.8)
                base = 0.35
                al = a_l * (base + (1 - base) * hi)
                text(c, x, y, name, 34, CREAM, al, kind="sans-m", anchor="rs")
                if f:
                    fcol = col
                    text(c, x + 60, y + dy / 2, f, 30, fcol, a_l * (0.35 + 0.65 * max(hi, smooth((t - [18.2, 25.2, 10.6][["top", "mid", "bottom"].index(grp if k != 7 else "bottom")]) / 0.8))),
                         kind="sans-sb", anchor="lm")
                    line(c, x + 40, y + 8, x + 40, y + dy - 30, CREAM_DIM, al * 0.6, w=1.2)
            # brackets
            for t0, k0, k1, s_, col in ((10.6, 7, 10, "decimal", CYAN), (18.2, 0, 3, "decimal", CYAN), (25.2, 4, 7, "24 and 60", GOLD)):
                ab = a_l * smooth((t - t0 - 0.4) / 0.6)
                if ab > 0:
                    xa = x + 230
                    ya, yb = y0 + dy * k0 - 30, y0 + dy * k1 + 6
                    poly(c, [(xa - 12, ya), (xa, ya), (xa, yb), (xa - 12, yb)], col, ab, w=1.8)
                    text(c, xa + 24, (ya + yb) / 2, s_, 36, col, ab, kind="sans-sb", anchor="lm")
        # the turning Earth
        a_g = fade(t, 33.6, 47.5, 0.8, 0.01)
        if a_g > 0:
            lon0 = (t - 33.6) * 12.0
            globe(c, 760, 430, 280, lon0, a_g, band=(0, 15))
            text(c, 1180, 380, "360° ÷ 24 h", 54, GOLD, a_g, kind="sans-sb", anchor="ls")
            text(c, 1180, 450, "= 15° an hour", 54, GOLD, a_g * smooth((t - 35.2) / 0.6), kind="sans-sb", anchor="ls")
            text(c, 1180, 510, "1° every 4 minutes", 34, CREAM, a_g * smooth((t - 36.4) / 0.6), anchor="ls")
        caps(c, t, [(0.6, 9.6, ["Since 1967 the second has been defined by an atom:",
                                "9,192,631,770 cycles of the radiation from caesium-133."]),
                    (10.2, 17.6, ["Below the second, time went decimal after all:", "milliseconds, microseconds, nanoseconds."]),
                    (18.2, 24.6, ["Above the year, too: decades, centuries, millennia."]),
                    (25.2, 32.8, ["The 24 and the sixties survive in between,", "on the scale of a human day."]),
                    (33.6, 45.8, ["And on the globe: Earth turns 360° in 24 hours, 15° an hour.",
                                  "That's why time zones are roughly 15° wide."])])


# ================================================================== 12. close
class Close(L.Section):
    name, dur = "close", 25.0

    def draw(self, c, t):
        L.sky(c, t, rot=t * 0.004, pole=(960, -300))
        cx, cy, r = 960, 430, 270
        T = 11 * 3600 + 59 * 60 + 47 + t
        sec = tick_frac(T)
        L.clock(c, cx, cy, r, (T / 3600 % 12) / 12, (T / 60 % 60) / 60, (sec % 60) / 60, a=1.0)
        notes = [(1.4, (cx - r - 40, cy - 150), "hours, 12 + 12: Egypt", LAPIS, "rs", polar(cx, cy, r * 0.8, -60)),
                 (3.4, (cx + r + 40, cy - 150), "sixties: Mesopotamia", CLAY, "ls", polar(cx, cy, r * 0.97, 45)),
                 (5.4, (cx - r - 40, cy + 170), "“minute”, “second”: medieval Latin", CREAM, "rs", polar(cx, cy, r * 0.6, 225)),
                 (10.6, (cx + r + 40, cy + 170), "1 s = 9,192,631,770 caesium cycles", CYAN, "ls", polar(cx, cy, r * 0.7, 135))]
        a_all = 1 - smooth((t - 18.4) / 1.4)
        for t0, (x, y), s_, col, anc, tip in notes:
            a = smooth((t - t0) / 0.7) * a_all
            if a <= 0:
                continue
            text(c, x, y, s_, 30, col, a, kind="sans-m", anchor=anc[0] + "m")
            sx = x + (12 if anc[0] == "l" else -12)
            line(c, sx, y + 4, tip[0], tip[1], col, a * 0.7, w=1.4)
            circle(c, tip[0], tip[1], 4, col, a)
        caps(c, t, [(0.6, 9.8, ["So every clock face is a fossil: Egypt's hours, counted in Babylon's sixties,",
                                "joined by Greek astronomers and named in medieval Latin,"]),
                    (10.4, 17.4, ["kept in step, today, by a caesium atom."])])
        a_e = fade(t, 18.6, 25.5, 1.0, 0.01)
        text(c, 960, 880, "Why 60? Why 24?", 52, CREAM, a_e * (1 - smooth((t - 22.6) / 1.6)), weight=540, anchor="ms")
        L.blackout(c, smooth((t - 22.4) / 2.4))


# ================================================================== main
def _sections():
    return {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, L.Section) and v is not L.Section}


SECTIONS = None


def get(name):
    global SECTIONS
    if SECTIONS is None:
        SECTIONS = {cls.name: cls for cls in _sections().values()}
    return SECTIONS[name]


if __name__ == "__main__":
    sec = get(sys.argv[1])()
    if len(sys.argv) > 2 and sys.argv[2] == "sheet":
        ts = [float(x) for x in sys.argv[3:]] or list(np.arange(0.5, sec.dur, sec.dur / 16))
        os.makedirs(L.OUT, exist_ok=True)
        L.contact(sec, ts, os.path.join(L.OUT, f"sheet_{sec.name}.jpg"))
    elif len(sys.argv) > 2:
        for s_ in sys.argv[2:]:
            sec.render(float(s_))
    else:
        sec.render()
