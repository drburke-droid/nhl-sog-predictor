"""
Compositor for the knockout film (make_film.py compose): lays each match's 3-D
frames over a data strip (names, live % mixed, top-down view of each bowl, live
chart) and cuts them together with title, round, bracket and takeaway cards.

Everything on screen that is a number comes from the simulation caches.
"""
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools import TOOL_COLOR, TOOLS, fork_visual, mix_curve, shape_geometry

W, H = 1280, 720
PIC_H = 560                       # 3-D picture on top, data strip below
FPS = 24
PAPER, INK, MUTED, TRACK, HAIR = (252, 252, 251), (20, 23, 28), (92, 90, 86), (222, 219, 212), (228, 226, 220)
SIDE = [(42, 120, 214), (235, 104, 52)]          # left / right contestant
GRAIN = np.array([[210, 54, 43], [242, 179, 29], [98, 163, 59], [44, 107, 209], [244, 239, 230], [91, 52, 32],
                  [181, 38, 110], [43, 43, 48]], np.uint8)
ROUND_TEXT = {
    "wedges": ("QUARTER-FINALS", "Six powders, loaded as pie wedges", "Everyone gets the same bowl and 15 seconds."),
    "stripes": ("SEMI-FINALS", "New problem: stripes", "Now the powders have to cross the bowl, not just go round it."),
    "rings": ("THE FINAL", "Hardest load: rings", "Going round and round does nothing. Powder has to move in and out."),
}
_fonts = {}

# Everything below is laid out for a 1280 x 720 frame.  K scales it: 1.5 gives
# 1920 x 1080, 3 gives 3840 x 2160.  Layout code keeps using 1280 x 720 units.
K = 1.0


def set_scale(k):
    global K
    K = float(k)


def spx(v):
    """Layout units to output pixels."""
    return int(round(v * K))


def new_canvas():
    return Image.new("RGB", (spx(W), spx(H)), PAPER)


class Draw:
    """ImageDraw that takes layout units: positions, widths and radii are scaled by K."""

    def __init__(self, im):
        self.d = ImageDraw.Draw(im)

    def line(self, xy, fill=None, width=1, joint=None):
        self.d.line([v * K for v in xy], fill=fill, width=max(1, spx(width)), joint=joint)

    def rounded_rectangle(self, box, radius, fill=None, outline=None, width=1):
        self.d.rounded_rectangle([v * K for v in box], radius * K, fill=fill, outline=outline, width=max(1, spx(width)))

    def textlength(self, s, font):
        return self.d.textlength(s, font=font) / K

    def text(self, xy, s, font=None, fill=None, anchor=None):
        self.d.text((xy[0] * K, xy[1] * K), s, font=font, fill=fill, anchor=anchor)


def font(size, weight="Regular", display=False):
    key = (spx(size), weight, display)
    if key not in _fonts:
        base = "/usr/share/fonts/opentype/inter/Inter%s-%s.otf" % ("Display" if display else "", weight)
        fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight in ("Bold", "ExtraBold", "Black", "SemiBold") else "")
        _fonts[key] = ImageFont.truetype(base if os.path.exists(base) else fb, spx(size))
    return _fonts[key]


def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def short(tool):
    return TOOLS[tool]["label"]


def tool_rgb(tool):
    h = TOOL_COLOR[tool].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


_icons = {}


def tool_icon(tool, size):
    """Silhouette of a tool in its identity colour, as an RGBA image `size` layout units square."""
    key = (tool, spx(size))
    if key in _icons:
        return _icons[key]
    ss = 4
    S = spx(size) * ss
    col = tool_rgb(tool) + (255,)
    polys, holes, lines = [], [], []
    parts = TOOLS[tool]["parts"]
    gap = 0.0
    for k, p in enumerate(parts):
        g = shape_geometry(p["shape"], p["p"])
        H = g["H"]
        dx = (k - (len(parts) - 1) / 2) * 24.0
        if p["shape"] == "fork":
            polys.append([(x + dx, z) for x, z in fork_visual(H + 30)])
        elif g["kind"] in ("outline", "spoon"):
            polys.append([(x + dx, z) for x, z in g["outer"]])
            holes += [[(x + dx, z) for x, z in h] for h in g["holes"]]
            lines.append(([(dx, H - 1), (dx, H + 22)], 3.4))
        else:
            for w in g["wires"]:
                lines.append(([(x + dx, z) for x, _, z in w], 1.5))
            lines.append(([(dx, H - 1), (dx, H + 16)], 3.4))
    pts = [q for poly in polys for q in poly] + [q for ln, _ in lines for q in ln]
    xs, zs = [q[0] for q in pts], [q[1] for q in pts]
    sc = 0.9 * S / max(max(xs) - min(xs), max(zs) - min(zs))
    cx, cz = (max(xs) + min(xs)) / 2, (max(zs) + min(zs)) / 2
    T = lambda q: (S / 2 + (q[0] - cx) * sc, S / 2 - (q[1] - cz) * sc)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for poly in polys:
        d.polygon([T(q) for q in poly], fill=col)
    for h in holes:
        d.polygon([T(q) for q in h], fill=(0, 0, 0, 0))
    for ln, w in lines:
        d.line([T(q) for q in ln], fill=col, width=max(2, int(round(w * sc))), joint="curve")
    _icons[key] = im.resize((spx(size), spx(size)), Image.LANCZOS)
    return _icons[key]


def paste_icon(canvas, tool, size, x, y):
    ic = tool_icon(tool, size)
    canvas.paste(ic, (spx(x), spx(y)), ic)


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        z = np.load(os.path.join(d, "meta.npz"))
        self.mix = np.clip(mix_curve(z), 0, 1)
        self.count, self.t, self.ingr = z["count"], z["t"], z["ingredient"]
        self.n, self.N = self.meta["n_frames"], self.meta["n_grains"]
        self.pos = np.memmap(os.path.join(d, "particles.pc2"), dtype="<f4", mode="r", offset=32).reshape(self.n, self.N, 3)
        self.R, self.d = self.meta["bowl"]["R"], self.meta["grain"]
        self.label, self.unit = self.meta["label"], self.meta["unit"]

    def disc(self, i, size):
        """Top-down picture of the bowl at frame i, as an RGBA disc."""
        ss = 2
        size = spx(size)
        S = size * ss
        p = np.asarray(self.pos[i])
        px = S / (2 * self.R)
        img = np.zeros((S, S, 3), np.uint8)
        img[:] = (238, 236, 230)
        zb = np.full((S, S), -1.0, np.float32)
        x = ((p[:, 0] + self.R) * px).astype(np.int32)
        y = ((self.R - p[:, 1]) * px).astype(np.int32)
        r = max(int(round(self.d * px / 2)), 1)
        col = GRAIN[self.ingr]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xx = np.clip(x + dx, 0, S - 1); yy = np.clip(y + dy, 0, S - 1)
                m = p[:, 2] > zb[yy, xx]
                # several grains can land on one pixel: let the highest win
                order = np.argsort(p[m, 2])
                img[yy[m][order], xx[m][order]] = col[m][order]
                zb[yy[m][order], xx[m][order]] = p[m, 2][order]
        im = Image.fromarray(img).resize((size, size), Image.LANCZOS).convert("RGBA")
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).ellipse((2, 2, S - 3, S - 3), fill=255)
        im.putalpha(mask.resize((size, size), Image.LANCZOS))
        return im


def paste_disc(canvas, disc, x, y, ring=TRACK):
    """Paste a disc (already in output pixels) with its ring at layout position x, y."""
    s = disc.size[0]
    x, y, e = spx(x), spx(y), spx(3)
    d = ImageDraw.Draw(canvas)
    d.ellipse((x - e, y - e, x + s + e - 1, y + s + e - 1), fill=ring)
    canvas.paste(disc, (x, y), disc)


# ---------------------------------------------------------------------------
# match frames
# ---------------------------------------------------------------------------
def chart(a, b, i, n, w=300, h=118, seconds=15.0):
    ss = 2                                  # drawn at twice the output size, then reduced
    q = ss * K
    im = Image.new("RGB", (int(round(w * q)), int(round(h * q))), PAPER)
    d = ImageDraw.Draw(im)
    lw = max(1, int(round(q)))
    x0, x1, y0, y1 = 34 * q, (w - 18) * q, 8 * q, (h - 20) * q
    X = lambda t: x0 + (x1 - x0) * min(t / seconds, 1.0)
    Y = lambda v: y1 - (y1 - y0) * v
    for v, lab in ((0, "0"), (0.5, "50"), (0.9, "90"), (1.0, "")):
        d.line((x0, Y(v), x1, Y(v)), fill=HAIR if v != 0.9 else (190, 187, 180), width=lw)
        if lab:
            text(d, (x0 - 6 * q, Y(v)), lab, font(10 * ss, "Medium"), MUTED, "rm")
    for s in (0, 5, 10, 15):
        text(d, (X(s), y1 + 5 * q), f"{s} s", font(10 * ss, "Medium"), MUTED, "ma")
    for run in (a, b):
        col = tool_rgb(run.meta["tool"])
        k = min(i, run.n - 1)
        pts = [(X(run.t[j]), Y(run.mix[j])) for j in range(0, k + 1)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=int(round(3 * q)), joint="curve")
        px, py = pts[-1]
        d.ellipse((px - 4 * q, py - 4 * q, px + 4 * q, py + 4 * q), fill=col, outline=PAPER, width=lw)
    return im.resize((spx(w), spx(h)), Image.LANCZOS)


def count_word(run, i):
    k = int(np.floor(run.count[min(i, run.n - 1)]))
    unit = run.unit[:-1] if k == 1 and run.unit.endswith("s") else run.unit
    return f"{k} {unit}"


PLATE = None        # None: real pictures.  (r, g, b): a flat colour instead, for the overlay plates.


def full_picture(path):
    """A full-frame rendered shot (the closing section)."""
    if PLATE is not None:
        return Image.new("RGB", (spx(W), spx(H)), PLATE)
    return Image.open(path).convert("RGB").resize((spx(W), spx(H)), Image.LANCZOS)


def picture(path):
    """Load a rendered frame and fit it to the picture area (centre crop)."""
    if PLATE is not None:
        return Image.new("RGB", (spx(W), spx(PIC_H)), PLATE)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    th = int(round(w * PIC_H / W))
    if h > th:
        top = (h - th) // 2
        im = im.crop((0, top, w, top + th))
    return im.resize((spx(W), spx(PIC_H)), Image.LANCZOS)


def match_frame(m, a, b, i, frames_dir, winner=None, badge=0.0):
    """One composed frame.  winner/badge: after the bell, 0..1 fade-in of the result."""
    n = min(a.n, b.n)
    k = min(i, n - 1)
    canvas = new_canvas()
    canvas.paste(picture(os.path.join(frames_dir, f"f_{k + 1:04d}.png")), (0, 0))
    canvas.info["pic"] = (m["key"], k + 1)          # which rendered frame is under this one
    d = Draw(canvas)
    d.line((0, PIC_H, W, PIC_H), fill=HAIR, width=1)
    # header pill on the picture
    title = m["name"].upper()
    sub = f"loaded as {ROUND_NAMES[m['load']]}"
    tw = int(d.textlength(title, font=font(17, "Bold")))
    sw = int(d.textlength(sub, font=font(14, "Medium")))
    clock = f"{min(a.t[k], 15.0):4.1f} s"
    pw = tw + sw + 150
    px = (W - pw) // 2
    d.rounded_rectangle((px, 18, px + pw, 56), 19, fill=(255, 255, 255), outline=HAIR)
    text(d, (px + 20, 37), title, font(17, "Bold"), INK, "lm")
    text(d, (px + 20 + tw + 14, 37), sub, font(14, "Medium"), MUTED, "lm")
    text(d, (px + pw - 20, 37), clock, font(17, "SemiBold"), INK, "rm")
    cap = CAPTIONS.get(m["key"])
    tsec = a.t[k]
    if cap and 2.5 <= tsec <= 13.0 and winner is None:
        cw = int(d.textlength(cap, font=font(17, "Medium")))
        cx = (W - cw) // 2
        d.rounded_rectangle((cx - 18, PIC_H - 52, cx + cw + 18, PIC_H - 16), 18, fill=(255, 255, 255), outline=HAIR)
        text(d, (W // 2, PIC_H - 34), cap, font(17, "Medium"), INK, "mm")
    # data strip
    ds = 128
    for side, run in ((0, a), (1, b)):
        col = tool_rgb(run.meta["tool"])
        lose = winner is not None and winner != side
        dx = 22 if side == 0 else W - 22 - ds
        paste_disc(canvas, run.disc(k, ds), dx, PIC_H + 16)
        tx = dx + ds + 22 if side == 0 else dx - 22
        anc = "l" if side == 0 else "r"
        ink = MUTED if lose and badge > 0.5 else INK
        paste_icon(canvas, run.meta["tool"], 30, tx if side == 0 else tx - 30, PIC_H + 12)
        text(d, (tx + 38 if side == 0 else tx - 38, PIC_H + 20), run.label.upper(), font(15, "Bold"), ink, anc + "a")
        pct = f"{100 * run.mix[k]:.0f}%"
        text(d, (tx, PIC_H + 40), pct, font(58, "Bold", True), ink, anc + "a")
        bw = 250
        bx0 = tx if side == 0 else tx - bw
        d.rounded_rectangle((bx0, PIC_H + 110, bx0 + bw, PIC_H + 118), 4, fill=TRACK)
        fill = max(int(bw * run.mix[k]), 8)
        d.rounded_rectangle((bx0, PIC_H + 110, bx0 + fill, PIC_H + 118), 4, fill=col if not (lose and badge > 0.5) else (170, 168, 162))
        text(d, (tx, PIC_H + 127), count_word(run, k), font(14, "Medium"), MUTED, anc + "a")
        if winner == side and badge > 0:
            wx = tx + (150 if side == 0 else -150)
            lab = "WINNER"
            lw = int(d.textlength(lab, font=font(15, "Bold")))
            box = (wx, PIC_H + 58, wx + lw + 28, PIC_H + 88) if side == 0 else (wx - lw - 28, PIC_H + 58, wx, PIC_H + 88)
            c = tuple(int(PAPER[q] + (col[q] - PAPER[q]) * badge) for q in range(3))
            d.rounded_rectangle(box, 15, fill=c)
            text(d, ((box[0] + box[2]) // 2, PIC_H + 73), lab, font(15, "Bold"), (255, 255, 255), "mm")
    ch = chart(a, b, k, n)
    canvas.paste(ch, (spx((W - 300) // 2), spx(PIC_H + 18)))
    text(d, (W // 2 - 150 + 34, PIC_H + 10), "% MIXED", font(10, "Bold"), MUTED, "la")
    return canvas


ROUND_NAMES = {"wedges": "pie wedges", "stripes": "stripes", "rings": "rings"}


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------
def card(lines, discs=None, footer=None):
    """Simple centred text card.  lines: list of (text, size, weight, colour, gap-after)."""
    im = new_canvas()
    d = Draw(im)
    total = sum(sz + gap for _, sz, _, _, gap in lines) + (240 if discs else 0)
    y = (H - total) // 2
    if discs:
        n = len(discs)
        s = int(round(discs[0][0].size[0] / K))
        x = (W - (n * s + (n - 1) * 70)) // 2
        for dimg, cap in discs:
            paste_disc(im, dimg, x, y)
            text(d, (x + s // 2, y + s + 12), cap, font(17, "SemiBold"), INK, "ma")
            x += s + 70
        y += 190 + 50
    for s, sz, wt, col, gap in lines:
        text(d, (W // 2, y), s, font(sz, wt, sz >= 40), col, "ma")
        y += sz + gap
    if footer:
        text(d, (W // 2, H - 40), footer, font(14, "Medium"), MUTED, "mm")
    return im


def champion_card(m, runs):
    """The winner of the final, in its colour."""
    win = m["winner"]
    lose = m["b"] if win == m["a"] else m["a"]
    rw, rl = (m["ra"], m["rb"]) if win == m["a"] else (m["rb"], m["ra"])
    im = new_canvas()
    d = Draw(im)
    text(d, (W // 2, 150), "CHAMPION", font(26, "Bold"), MUTED, "mm")
    paste_icon(im, win, 150, W // 2 - 75, 190)
    text(d, (W // 2, 410), short(win).replace(", by hand", "").upper(), font(96, "Black", True), tool_rgb(win), "mm")
    text(d, (W // 2, 500), f"{100 * rw:.0f}% mixed in the final, against the {short(lose).lower()}'s {100 * rl:.0f}%",
         font(24, "Medium"), MUTED, "mm")
    return im


def contestants_card():
    im = new_canvas()
    d = Draw(im)
    text(d, (W // 2, 70), "The contestants", font(40, "Bold", True), INK, "mm")
    cols = [("BY HAND", 110, [("spoon", "Spoon", "stirred in a spiral through the whole bowl"),
                              ("fork", "Fork", "the same spiral"),
                              ("whisk", "Whisk", "the same spiral"),
                              ("spoon_centre", "Spoon twirled in the middle", "spins on the spot")]),
            ("BY MACHINE", 680, [("paddle_large", "Stand mixer, large paddle", "orbits the bowl while it spins"),
                                 ("paddle_small", "Stand mixer, small paddle", "same motion, a much smaller paddle"),
                                 ("wire_whip", "Stand mixer, wire whip", "same motion, wires instead of a blade"),
                                 ("twin_beaters", "Hand mixer", "twin beaters carried round the bowl")])]
    for head, x, rows in cols:
        text(d, (x, 150), head, font(15, "Bold"), MUTED, "la")
        d.line((x, 176, x + 460, 176), fill=HAIR, width=2)
        y = 200
        for tool, name, how in rows:
            paste_icon(im, tool, 54, x, y - 2)
            text(d, (x + 72, y), name, font(25, "SemiBold"), INK, "la")
            text(d, (x + 72, y + 34), how, font(18, "Medium"), MUTED, "la")
            y += 104
    text(d, (W // 2, H - 72), "Every tool hangs from the same stand, so only the tool and the way it moves differ.",
         font(17, "Medium"), INK, "mm")
    text(d, (W // 2, H - 44), "Hand tools: 1.5 strokes a second.  Stand mixer: 0.8 orbits a second.  Hand mixer: 5 turns a second.",
         font(15, "Medium"), MUTED, "mm")
    return im


# ---------------------------------------------------------------------------
def compose(args, matches):
    from make_film import run_dir
    set_scale(getattr(args, "scale", 1.0))
    if getattr(args, "partial", False):
        matches = [m for m in matches if m["winner"] is not None]
    runs = {}
    for m in matches:
        for t in (m["a"], m["b"]):
            runs[(m["load"], t)] = Run(run_dir(m["load"], t))
    ref = runs[("wedges", "spoon")]

    # frames are streamed straight into ffmpeg; only the last one is kept (for cross-fades)
    global PLATE
    PLATE = getattr(args, "plate", None)
    codec = (["-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast"] if PLATE is not None       # lossless, for the plates
             else ["-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{spx(W)}x{spx(H)}", "-r", str(FPS),
           "-i", "-"] + codec + [args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    state = dict(n=0, last=None)
    timeline = []       # [segment name, start second, length in seconds]
    pics = []           # per output frame: (shot, frame number) of the rendered picture under it, or None

    def emit(im, repeat=1):
        raw = im.tobytes()
        for _ in range(repeat):
            enc.stdin.write(raw)
        state["n"] += repeat
        state["last"] = im
        pics.extend([im.info.get("pic")] * repeat)

    def fade(a, b, k, pic):
        im = Image.blend(a, b, k)
        if pic:
            im.info["pic"] = pic
        return im

    def begin(name, first):
        last = state["last"]
        if last is not None:
            pa, pb = last.info.get("pic"), first.info.get("pic")
            if pa and pb:
                # picture to picture: dip through the paper colour, so that no frame ever
                # holds two rendered pictures (the Blender master lays one under each frame)
                paper = new_canvas()
                for k in range(5):
                    emit(fade(last, paper, (k + 1) / 5, pa))
                for k in range(5):
                    emit(fade(paper, first, (k + 1) / 6, pb))
            else:
                for k in range(10):
                    emit(fade(last, first, (k + 1) / 11, pa or pb))
        timeline.append([name, round(state["n"] / FPS, 2), 0.0])

    def end():
        timeline[-1][2] = round(state["n"] / FPS - timeline[-1][1], 2)

    def still(name, im, seconds):
        begin(name, im)
        emit(im, int(round(seconds * FPS)))
        end()

    # ---- opening ------------------------------------------------------------
    # cold open: a few seconds of the semi-final under the question the film answers
    tm = next((m for m in matches if m["key"] == "S1"), matches[0])
    tdir = os.path.join(args.frames, tm["key"], "frames")
    t0, t1 = 36, 150
    have = [i for i in range(t0, t1) if os.path.exists(os.path.join(tdir, f"f_{i + 1:04d}.png"))
            and os.path.getsize(os.path.join(tdir, f"f_{i + 1:04d}.png")) > 0]
    if have:
        def teaser(i):
            im = new_canvas()
            im.paste(picture(os.path.join(tdir, f"f_{i + 1:04d}.png")), (0, 0))
            im.info["pic"] = (tm["key"], i + 1)
            d = Draw(im)
            d.line((0, PIC_H, W, PIC_H), fill=HAIR, width=1)
            text(d, (W // 2, PIC_H + 58), "What is the best way to mix dry ingredients?", font(44, "Bold", True), INK, "mm")
            text(d, (W // 2, PIC_H + 116), "We simulated eight of them, grain by grain.", font(22, "Medium"), MUTED, "mm")
            return im
        begin("cold_open", teaser(have[0]))
        for i in have:
            emit(teaser(i))
        end()
    still("title", card([("THE MIX-OFF", 92, "Black", INK, 22),
                         ("Eight ways to mix dry ingredients. One bowl. Fifteen seconds each.", 26, "Medium", MUTED, 0)],
                        footer=f"A grain-by-grain physics simulation: {ref.N:,} grains per bowl"), 3.5)
    i50 = int(np.argmax(ref.mix >= 0.5))
    still("scoring", card([("How it is scored", 40, "Bold", INK, 16),
                           ("Cut the bowl into small scoops and count the colours in each.", 22, "Medium", MUTED, 8),
                           ("100% means every scoop looks like a perfect shuffle.", 22, "Medium", MUTED, 0)],
                          discs=[(ref.disc(0, 170), "0% mixed"), (ref.disc(i50, 170), "50%"),
                                 (ref.disc(ref.n - 1, 170), f"{100 * ref.mix[-1]:.0f}%")]), 5.0)
    still("contestants", contestants_card(), 5.5)

    # ---- matches ------------------------------------------------------------
    last_round = None
    for idx, m in enumerate(matches):
        a, b = runs[(m["load"], m["a"])], runs[(m["load"], m["b"])]
        if m["round"] != last_round:
            h1, h2, h3 = ROUND_TEXT[m["load"]]
            still("round_" + m["load"], card([(h1, 64, "Black", INK, 18), (h2, 30, "SemiBold", INK, 12),
                                              (h3, 22, "Medium", MUTED, 0)], discs=[(a.disc(0, 170), "")]), 3.5)
            last_round = m["round"]
        fdir = os.path.join(args.frames, m["key"], "frames")
        n = min(a.n, b.n)
        if getattr(args, "partial", False):             # preview cut from whatever has been rendered
            have = 0
            while have < n and os.path.exists(os.path.join(fdir, f"f_{have + 1:04d}.png")) and \
                    os.path.getsize(os.path.join(fdir, f"f_{have + 1:04d}.png")) > 0:
                have += 1
            n = min(n, have, args.partial)
            if n < 2:
                continue
        win = 0 if m["winner"] == m["a"] else 1
        first = match_frame(m, a, b, 0, fdir)
        begin(m["key"], first)
        emit(first)
        for i in range(1, n):
            emit(match_frame(m, a, b, i, fdir))
        for k in range(12):
            emit(match_frame(m, a, b, n - 1, fdir, win, (k + 1) / 12))
        emit(match_frame(m, a, b, n - 1, fdir, win, 1.0), int(2.2 * FPS))
        end()
        print("composed", m["key"], f"{state['n'] / FPS:.1f} s", flush=True)

    # ---- closing ------------------------------------------------------------
    champ = matches[-1]["winner"]
    still("champion", champion_card(matches[-1], runs), 4.5)
    for k in range(len(TAKEAWAYS)):
        still(f"takeaway{k + 1}", card(TAKEAWAYS[k]), 4.5 if k < 3 else 5.0)
    still("research", card([("What published experiments say", 34, "Bold", INK, 18),
                            ("In industrial powder blenders, mixing depends on the number of turns, not the speed.", 21, "Medium", MUTED, 8),
                            ("Brone, Alexander & Muzzio, AIChE Journal, 1998", 15, "Medium", MUTED, 26),
                            ("In bladed mixers, two or three blades mix better than one or four.", 21, "Medium", MUTED, 8),
                            ("Boonkanokwong et al., Powder Technology, 2016", 15, "Medium", MUTED, 26),
                            ("We found no published measurements of spoons or whisks in dry powder.", 21, "Medium", INK, 0)]), 5.0)
    still("caveats", card([("What this is, and is not", 34, "Bold", INK, 18),
                           ("A physics simulation of dry, free-flowing grains of equal size.", 21, "Medium", MUTED, 8),
                           ("No clumping, no dust cloud, no butter. Machine speeds are typical low settings.", 21, "Medium", MUTED, 8),
                           ("Every number on screen was measured from the simulated grains.", 21, "Medium", MUTED, 0)]), 3.5)

    # ---- and now, some satisfying blending -------------------------------------
    from make_film import SATISFYING, art_dir
    from patterns import ART
    shots = []
    for key, left, right in SATISFYING:
        fdir = os.path.join(args.frames, key, "frames")
        n = 0
        while os.path.exists(os.path.join(fdir, f"f_{n + 1:04d}.png")) and os.path.getsize(os.path.join(fdir, f"f_{n + 1:04d}.png")) > 0:
            n += 1
        if getattr(args, "partial", False):
            n = min(n, args.partial)
        if n >= 2:
            shots.append((key, fdir, n, left, right))
    if shots:
        still("satisfying_title", card([("And now, some satisfying blending.", 54, "Bold", INK, 16),
                                        ("Same grains, same physics. No scores.", 24, "Medium", MUTED, 0)]), 2.6)
        for key, fdir, n, left, right in shots:
            tool_name = lambda t: "Spoon" if t == "spoon_circle" else short(t).replace(", by hand", "")
            labels = [f"{ART[art][0]}  \u00b7  {tool_name(tool)}" for art, tool in (left, right)]

            def shot(i):
                im = full_picture(os.path.join(fdir, f"f_{i + 1:04d}.png"))
                im.info["pic"] = (key, i + 1)
                d = Draw(im)
                for side, lab in enumerate(labels):
                    lw = d.textlength(lab, font=font(17, "SemiBold"))
                    x0 = 24 if side == 0 else W - 24 - lw - 36
                    d.rounded_rectangle((x0, H - 58, x0 + lw + 36, H - 22), 18, fill=(255, 255, 255), outline=HAIR)
                    text(d, (x0 + 18, H - 40), lab, font(17, "SemiBold"), INK, "lm")
                return im

            first = shot(0)
            begin(key, first)
            emit(first, 42)                              # hold the untouched picture long enough to read it
            for i in range(1, n):
                emit(shot(i))
            emit(state["last"], 10)
            end()
            print("composed", key, f"{state['n'] / FPS:.1f} s", flush=True)
        still("end", card([("THE MIX-OFF", 76, "Black", INK, 20),
                           ("Method, code and data are linked in the description.", 22, "Medium", MUTED, 0)],
                          footer="Kitchen Inc is a made-up brand. Not sponsored by or affiliated with YouTube or any appliance maker."), 3.0)
    enc.stdin.close()
    enc.wait()
    json.dump(pics, open(os.path.splitext(args.out)[0] + "_pics.json", "w"))
    json.dump(dict(fps=FPS, seconds=round(state["n"] / FPS, 2), segments=timeline,
                   matches=[{k: m[k] for k in ("key", "name", "load", "a", "b", "ra", "rb", "winner")} for m in matches]),
              open(os.path.splitext(args.out)[0] + "_timeline.json", "w"), indent=1)
    print(f"wrote {args.out}: {state['n']} frames, {state['n'] / FPS:.1f} s")


TAKEAWAYS = [
    [("1. Coverage beats power", 44, "Bold", INK, 20),
     ("A tool only mixes where it sweeps.", 24, "Medium", MUTED, 10),
     ("The spoon twirled in the middle never touched most of the bowl.", 24, "Medium", MUTED, 0)],
    [("2. Count strokes, not seconds", 44, "Bold", INK, 20),
     ("A spoon or whisk needed 11 to 20 strokes to reach 90%, depending on the load.", 24, "Medium", MUTED, 10),
     ("A spoon stirred twice as fast needed the same number of strokes.", 24, "Medium", MUTED, 0)],
    [("3. How you load the bowl matters", 44, "Bold", INK, 20),
     ("Wedges and stripes are easy. Rings are hard: powder has to move in and out.", 24, "Medium", MUTED, 10),
     ("Rings are also where the stand mixer did its best work: 94% in 20 seconds.", 24, "Medium", MUTED, 0)],
    [("4. Does grain size or speed change the winner?", 40, "Bold", INK, 20),
     ("Grain size: no. From 1.4 to 2.4 mm grains, the spoon and whisk stayed on top.", 23, "Medium", MUTED, 10),
     ("Speed: not for hand tools. At double speed the stand mixer's paddle", 23, "Medium", MUTED, 4),
     ("improved to 97% and the hand mixer dropped to 54%.", 23, "Medium", MUTED, 0)],
]


# one line per match, written from what the simulation shows (see the top-down views)
CAPTIONS = {
    "Q1": "Same spoon. The only difference is where it goes.",
    "Q2": "Both stir the middle. The outer ring just rides round.",
    "Q3": "The beaters churn the centre. Whole wedges circle the rim.",
    "Q4": "The small paddle never reaches the core in the middle.",
    "S1": "The spoon folds the stripes into a spiral. The paddle leaves the rim behind.",
    "S2": "Same path, but the fork is narrower and the rim lags behind.",
    "F1": "Rings: the outermost band is the last to go.",
}
