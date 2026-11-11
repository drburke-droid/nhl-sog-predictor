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

from tools import TOOLS, mix_curve

W, H = 1280, 720
PIC_H = 560                       # 3-D picture on top, data strip below
FPS = 24
PAPER, INK, MUTED, TRACK, HAIR = (252, 252, 251), (20, 23, 28), (92, 90, 86), (222, 219, 212), (228, 226, 220)
SIDE = [(42, 120, 214), (235, 104, 52)]          # left / right contestant
GRAIN = np.array([[210, 54, 43], [242, 179, 29], [98, 163, 59], [44, 107, 209], [244, 239, 230], [91, 52, 32]], np.uint8)
ROUND_TEXT = {
    "wedges": ("QUARTER-FINALS", "Six powders, loaded as pie wedges", "Everyone gets the same bowl and 15 seconds."),
    "stripes": ("SEMI-FINALS", "New problem: stripes", "Now the powders have to cross the bowl, not just go round it."),
    "rings": ("THE FINAL", "Hardest load: rings", "Going round and round does nothing. Powder has to move in and out."),
}
_fonts = {}


def font(size, weight="Regular", display=False):
    key = (size, weight, display)
    if key not in _fonts:
        base = "/usr/share/fonts/opentype/inter/Inter%s-%s.otf" % ("Display" if display else "", weight)
        fb = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if weight in ("Bold", "ExtraBold", "Black", "SemiBold") else "")
        _fonts[key] = ImageFont.truetype(base if os.path.exists(base) else fb, size)
    return _fonts[key]


def text(d, xy, s, f, fill=INK, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def short(tool):
    return TOOLS[tool]["label"]


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
    s = disc.size[0]
    d = ImageDraw.Draw(canvas)
    d.ellipse((x - 3, y - 3, x + s + 2, y + s + 2), fill=ring)
    canvas.paste(disc, (x, y), disc)


# ---------------------------------------------------------------------------
# match frames
# ---------------------------------------------------------------------------
def chart(a, b, i, n, w=300, h=118, seconds=15.0):
    ss = 2
    im = Image.new("RGB", (w * ss, h * ss), PAPER)
    d = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 34 * ss, (w - 18) * ss, 8 * ss, (h - 20) * ss
    X = lambda t: x0 + (x1 - x0) * min(t / seconds, 1.0)
    Y = lambda v: y1 - (y1 - y0) * v
    for v, lab in ((0, "0"), (0.5, "50"), (0.9, "90"), (1.0, "")):
        d.line((x0, Y(v), x1, Y(v)), fill=HAIR if v != 0.9 else (190, 187, 180), width=ss)
        if lab:
            text(d, (x0 - 6 * ss, Y(v)), lab, font(10 * ss, "Medium"), MUTED, "rm")
    for s in (0, 5, 10, 15):
        text(d, (X(s), y1 + 5 * ss), f"{s} s", font(10 * ss, "Medium"), MUTED, "ma")
    for run, col in ((a, SIDE[0]), (b, SIDE[1])):
        k = min(i, run.n - 1)
        pts = [(X(run.t[j]), Y(run.mix[j])) for j in range(0, k + 1)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3 * ss, joint="curve")
        px, py = pts[-1]
        d.ellipse((px - 4 * ss, py - 4 * ss, px + 4 * ss, py + 4 * ss), fill=col, outline=PAPER, width=ss)
    return im.resize((w, h), Image.LANCZOS)


def count_word(run, i):
    k = int(np.floor(run.count[min(i, run.n - 1)]))
    unit = run.unit[:-1] if k == 1 and run.unit.endswith("s") else run.unit
    return f"{k} {unit}"


def picture(path):
    """Load a rendered frame and fit it to the picture area (centre crop)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    th = int(round(w * PIC_H / W))
    if h > th:
        top = (h - th) // 2
        im = im.crop((0, top, w, top + th))
    return im.resize((W, PIC_H), Image.LANCZOS)


def match_frame(m, a, b, i, frames_dir, winner=None, badge=0.0):
    """One composed frame.  winner/badge: after the bell, 0..1 fade-in of the result."""
    n = min(a.n, b.n)
    k = min(i, n - 1)
    canvas = Image.new("RGB", (W, H), PAPER)
    canvas.paste(picture(os.path.join(frames_dir, f"f_{k + 1:04d}.png")), (0, 0))
    d = ImageDraw.Draw(canvas)
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
    # data strip
    ds = 128
    for side, run in ((0, a), (1, b)):
        col = SIDE[side]
        lose = winner is not None and winner != side
        dx = 22 if side == 0 else W - 22 - ds
        paste_disc(canvas, run.disc(k, ds), dx, PIC_H + 16)
        tx = dx + ds + 22 if side == 0 else dx - 22
        anc = "l" if side == 0 else "r"
        ink = MUTED if lose and badge > 0.5 else INK
        text(d, (tx, PIC_H + 20), run.label.upper(), font(15, "Bold"), ink, anc + "a")
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
    canvas.paste(ch, ((W - ch.size[0]) // 2, PIC_H + 18))
    text(d, (W // 2 - 150 + 34, PIC_H + 10), "% MIXED", font(10, "Bold"), MUTED, "la")
    return canvas


ROUND_NAMES = {"wedges": "pie wedges", "stripes": "stripes", "rings": "rings"}


# ---------------------------------------------------------------------------
# cards
# ---------------------------------------------------------------------------
def card(lines, discs=None, footer=None):
    """Simple centred text card.  lines: list of (text, size, weight, colour, gap-after)."""
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    total = sum(sz + gap for _, sz, _, _, gap in lines) + (240 if discs else 0)
    y = (H - total) // 2
    if discs:
        n = len(discs)
        s = discs[0][0].size[0]
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


def bracket_card(matches, runs, highlight=None, title="THE DRAW"):
    """The knockout tree, filled in as far as it has been played."""
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    text(d, (W // 2, 44), title, font(30, "Bold", True), INK, "mm")
    cols = [110, 500, 860]
    bw, bh = 310, 34
    pos = {}
    by_round = {r: [m for m in matches if m["round"] == r] for r in range(3)}
    ys = {0: [150 + k * 132 for k in range(4)], 1: [216, 480], 2: [348]}
    for r in range(3):
        slots = by_round[r] or []
        n_expected = [4, 2, 1][r]
        for k in range(n_expected):
            y = ys[r][k]
            m = slots[k] if k < len(slots) else None
            played = m is not None and highlight is not None and (m["round"], m["k"]) <= highlight
            for j, key in enumerate(("a", "b")):
                yy = y + j * (bh + 6)
                box = (cols[r], yy, cols[r] + bw, yy + bh)
                name = short(m[key]) if m else ""
                won = played and m["winner"] == m[key]
                lost = played and not won
                d.rounded_rectangle(box, 8, fill=(255, 255, 255), outline=INK if won else HAIR, width=2 if won else 1)
                if name:
                    text(d, (box[0] + 12, yy + bh // 2), name, font(15, "Bold" if won else "Medium"), MUTED if lost else INK, "lm")
                    if played:
                        v = m["ra"] if key == "a" else m["rb"]
                        text(d, (box[2] - 12, yy + bh // 2), f"{100 * v:.0f}%", font(15, "Bold" if won else "Medium"),
                             MUTED if lost else INK, "rm")
            pos[(r, k)] = (cols[r] + bw, y + bh + 3)
            if r > 0:                      # connectors from the two feeder matches
                for src in (2 * k, 2 * k + 1):
                    sx, sy = pos[(r - 1, src)]
                    tx, ty = cols[r], y + bh + 3
                    mx = (sx + tx) // 2
                    d.line((sx, sy, mx, sy, mx, ty, tx, ty), fill=TRACK, width=2)
    for r, (lab, load) in enumerate((("QUARTER-FINALS", "pie wedges"), ("SEMI-FINALS", "stripes"), ("FINAL", "rings"))):
        text(d, (cols[r], 96), lab, font(14, "Bold"), INK, "la")
        text(d, (cols[r], 116), load, font(13, "Medium"), MUTED, "la")
    return im


def hold(im, seconds):
    return [im] * int(round(seconds * FPS))


def fade(a, b, n=10):
    return [Image.blend(a, b, (k + 1) / (n + 1)) for k in range(n)]


# ---------------------------------------------------------------------------
def compose(args, matches):
    from make_film import run_dir
    runs = {}
    for m in matches:
        for t in (m["a"], m["b"]):
            runs[(m["load"], t)] = Run(run_dir(m["load"], t))
    ref = runs[("wedges", "spoon")]
    n_grains = ref.N

    seq = []            # list of frames (PIL images); consecutive segments are cross-faded

    def add(frames, cut=False):
        if seq and not cut:
            seq.extend(fade(seq[-1], frames[0]))
        seq.extend(frames)

    # ---- opening ------------------------------------------------------------
    add(hold(card([("THE MIX-OFF", 92, "Black", INK, 22),
                   ("Eight ways to mix dry ingredients. One bowl. Fifteen seconds each.", 26, "Medium", MUTED, 0)],
                  footer=f"A grain-by-grain physics simulation: {n_grains:,} grains per bowl"), 3.5))
    i50 = int(np.argmax(ref.mix >= 0.5))
    add(hold(card([("How it is scored", 40, "Bold", INK, 16),
                   ("Cut the bowl into small scoops and count the colours in each.", 22, "Medium", MUTED, 8),
                   ("100% means every scoop looks like a perfect shuffle.", 22, "Medium", MUTED, 0)],
                  discs=[(ref.disc(0, 170), "0% mixed"), (ref.disc(i50, 170), "50%"), (ref.disc(ref.n - 1, 170), f"{100*ref.mix[-1]:.0f}%")]), 5.0))
    add(hold(bracket_card(matches, runs, None), 4.0))

    # ---- matches ------------------------------------------------------------
    last_round = None
    for m in matches:
        a, b = runs[(m["load"], m["a"])], runs[(m["load"], m["b"])]
        if m["round"] != last_round:
            h1, h2, h3 = ROUND_TEXT[m["load"]]
            add(hold(card([(h1, 64, "Black", INK, 18), (h2, 30, "SemiBold", INK, 12), (h3, 22, "Medium", MUTED, 0)],
                          discs=[(a.disc(0, 170), "")]), 3.5))
            last_round = m["round"]
        fdir = os.path.join(args.frames, m["key"], "frames")
        n = min(a.n, b.n)
        win = 0 if m["winner"] == m["a"] else 1
        frames = [match_frame(m, a, b, i, fdir) for i in range(n)]
        add(frames)
        for k in range(12):
            seq.append(match_frame(m, a, b, n - 1, fdir, win, (k + 1) / 12))
        seq.extend(hold(match_frame(m, a, b, n - 1, fdir, win, 1.0), 2.2))
        print("composed", m["key"], len(seq), flush=True)
        is_round_end = m is matches[-1] or matches[matches.index(m) + 1]["round"] != m["round"]
        if is_round_end and m["round"] < 2:
            add(hold(bracket_card(matches, runs, (m["round"], m["k"]), "THE DRAW"), 3.5))

    # ---- closing ------------------------------------------------------------
    f = matches[-1]
    champ = f["winner"]
    add(hold(bracket_card(matches, runs, (2, 0), f"CHAMPION: {short(champ).upper()}"), 4.5))
    add(hold(card(TAKEAWAYS[0]), 4.5))
    add(hold(card(TAKEAWAYS[1]), 4.5))
    add(hold(card(TAKEAWAYS[2]), 4.5))
    add(hold(card([("What this is, and is not", 34, "Bold", INK, 18),
                   ("A physics simulation of dry, free-flowing grains of equal size.", 21, "Medium", MUTED, 8),
                   ("No clumping, no dust cloud, no butter. Machine speeds are typical low settings.", 21, "Medium", MUTED, 8),
                   ("Every number on screen was measured from the simulated grains.", 21, "Medium", MUTED, 0)]), 5.0))

    # ---- encode ---------------------------------------------------------------
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for im in seq:
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()
    print(f"wrote {args.out}: {len(seq)} frames, {len(seq)/FPS:.1f} s")


TAKEAWAYS = [
    [("1. Coverage beats power", 44, "Bold", INK, 20),
     ("A tool only mixes where it sweeps.", 24, "Medium", MUTED, 10),
     ("The spoon twirled in the middle never touched most of the bowl.", 24, "Medium", MUTED, 0)],
    [("2. Count strokes, not seconds", 44, "Bold", INK, 20),
     ("A spoon or whisk needs about a dozen strokes around the whole bowl.", 24, "Medium", MUTED, 10),
     ("A spoon stirred twice as fast needed the same number of strokes.", 24, "Medium", MUTED, 0)],
    [("3. How you load the bowl matters", 44, "Bold", INK, 20),
     ("Side by side is easy. Rings are hard: powder has to move in and out.", 24, "Medium", MUTED, 10),
     ("That is the one job where the stand mixer shines.", 24, "Medium", MUTED, 0)],
]
