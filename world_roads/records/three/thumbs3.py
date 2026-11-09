"""Thumbnails for film C (1280×720, three for YouTube's thumbnail test), in the films' look: dark surface, gold, the
athletes' team colours.

    python thumbs3.py c     # -> ../../../videos/sports/thumbnails/season_outliers_thumb_{D,E,F}.png
                            #    (THUMBS_FIRST_SET=1 also remakes A-C, the first style)

A: the five, rising above the elite curve. B: DiMaggio's 56 against the longest streak by anyone else that year (24).
C: Gretzky's 163 assists, more than anyone else's points (141).
D: every season-leading performance as a dot, Wilt alone at 7. E: a seismograph trace through every season on the five's
charts, by year; the five are the spikes. F: the five off the end of a bell curve.
"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]] + (sys.argv[1:] or ["c"])
WHICH = sys.argv[1]
sys.argv = [sys.argv[0], "x"]
sys.path.insert(0, HERE)
import cfilm                                                # noqa: E402  (palettes, photo folders, facts)
import scenes as SC                                         # noqa: E402
from look import W, H, BG, INK, INK2, MUTED, GOLD, new_fig, text, title, plt, px, outlined   # noqa: E402
from PIL import Image                                       # noqa: E402

DST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "videos", "sports", "thumbnails"))
CARDS = cfilm.CARDS
CURVE = "#9a9890"


def canvas(bg=None):
    fig = new_fig(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    if bg is not None:
        ax.imshow(bg, extent=(0, W, H, 0), zorder=0, interpolation="bilinear")
    return fig, ax


def save(fig, name):
    os.makedirs(DST, exist_ok=True)
    tmp = os.path.join(cfilm.OUT, f"_thumb_{name}.png")
    fig.savefig(tmp, dpi=100, facecolor=BG)
    plt.close(fig)
    out = os.path.join(DST, f"season_outliers_thumb_{name}.png")
    Image.open(tmp).convert("RGB").resize((1280, 720), Image.LANCZOS).save(out)
    print(out)


def card_plate(rank):
    """the card's background with the athlete's portrait (as on the countdown cards)"""
    c = CARDS[rank]
    pk = c.get("palette", rank)
    bg = SC.background(pk).astype(np.float32)
    por = SC.portrait_layer(c["photo"], pk).astype(np.float32)
    a = por[..., 3:4] / 255.0
    return (bg * (1 - a) + por[..., :3] * a).astype(np.uint8), SC.PALETTE[pk]


def thumb_a():
    """the five rising above the elite curve, lowest score on the left"""
    fig, ax = canvas()
    rng = np.random.default_rng(5)
    xs = np.linspace(60, W - 60, 46)
    curve = lambda x: 905 - 40 * np.sin((x - 60) / (W - 120) * math.pi)          # noqa: E731
    ax.plot(xs, curve(xs), color=CURVE, lw=px(5), zorder=2)
    for x in np.linspace(80, W - 80, 58):
        ax.plot([x], [curve(x) + rng.normal(0, 22)], marker="o", ms=px(10), color=SC.BLUE if hasattr(SC, "BLUE") else "#3b82f6",
                mec="none", alpha=0.75, zorder=1)
    text(fig, W - 70, 985, "the elite curve", size=36, color=INK2, ha="right")
    order = sorted(CARDS, key=lambda r: CARDS[r]["z_e3"])                       # Ichiro ... Wilt
    big = {"ichiro": "262", "rodman": "18.3", "gretzky": "163", "dimaggio": "56", "chamberlain": "50.4"}
    for i, r in enumerate(order):
        c = CARDS[r]
        pk = c.get("palette", r)
        x = 215 + i * 372
        yc = 905 - c["z_e3"] * 82                                               # higher score, higher up
        d = 290
        ax.plot([x, x], [curve(x) - 6, yc + d / 2 + 80], color=GOLD, lw=px(4), ls=(0, (2, 2)), zorder=3)
        ax.imshow(SC.face_disc(c["photo"], pk, d), extent=(x - d / 2, x + d / 2, yc + d / 2, yc - d / 2), zorder=5)
        title(fig, x, yc + d / 2 + 72, big[c["photo"]], size=68, color=SC.PALETTE[pk]["accent"], ha="center",
              path_effects=outlined(6))
    title(fig, W / 2, 128, "5 seasons nobody came close to", size=104, color=INK, ha="center")
    save(fig, "A")


def thumb_big(rank, number, unit, versus, who, name):
    """one athlete on their card plate: the number, and what the next best managed"""
    plate, pal = card_plate(rank)
    fig, ax = canvas(plate)
    title(fig, 96, 420, number, size=300, color=pal["accent"])
    text(fig, 106, 525, unit, size=78, color=INK, weight="semibold")
    for i, line in enumerate(versus.split("\n")):                  # one call per line: no overlap
        title(fig, 104, 690 + 96 * i, line, size=84, color=GOLD)
    text(fig, 106, 1010, who, size=40, color=INK2)
    save(fig, name)


def first_set():
    thumb_a()
    thumb_big(2, "56", "games in a row", "Anyone else\nthat year: 24", "Joe DiMaggio, 1941", "B")
    thumb_big(3, "163", "assists", "More than anyone\nelse's points", "Wayne Gretzky, 1985–86", "C")


# ------------------------------------------------------------------ a second style: the gap itself (review, 9 November)
from PIL import ImageDraw, ImageFilter, ImageFont           # noqa: E402

FONTS = "/usr/share/fonts/opentype/inter/"


def font(weight, size):
    return ImageFont.truetype(FONTS + {"black": "InterDisplay-Black.otf", "bold": "InterDisplay-Bold.otf",
                                       "semi": "InterDisplay-SemiBold.otf", "med": "InterDisplay-Medium.otf"}[weight], size)


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


GOLD_RGB, INK_RGB, MUTED_RGB = rgb(GOLD), (240, 238, 232), (138, 143, 152)


def base(glow_at=None, glow_color=GOLD_RGB, strength=110, radius=420):
    im = Image.new("RGBA", (W, H), (11, 14, 19, 255))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = 1.0 - 0.35 * np.clip(np.hypot((xx - W / 2) / W, (yy - H / 2) / H) * 1.6, 0, 1)        # a soft vignette
    arr = np.asarray(im).astype(np.float32)
    arr[..., :3] *= v[..., None]
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if glow_at:
        glow(im, glow_at, radius, glow_color, strength)
    return im


def glow(im, at, r, color, a):
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse((at[0] - r, at[1] - r, at[0] + r, at[1] + r), fill=color + (a,))
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(r * 0.45)))


def disc(im, key, pk, cx, cy, d, ring=GOLD_RGB, halo=True):
    if halo:
        glow(im, (cx, cy), int(d * 0.75), ring, 120)
    f = Image.fromarray(np.asarray(SC.face_disc(key, pk, d)))
    im.alpha_composite(f, (int(cx - d / 2), int(cy - d / 2)))
    ImageDraw.Draw(im).ellipse((cx - d / 2 - 4, cy - d / 2 - 4, cx + d / 2 + 4, cy + d / 2 + 4), outline=ring + (255,),
                               width=8)


def dashed(dr, p0, p1, color, width=6, dash=18, gap=12):
    x0, y0 = p0
    x1, y1 = p1
    n = math.hypot(x1 - x0, y1 - y0)
    k = 0.0
    while k < n:
        a, b = k / n, min(n, k + dash) / n
        dr.line((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), fill=color, width=width)
        k += dash + gap


def finish(im, name):
    os.makedirs(DST, exist_ok=True)
    out = os.path.join(DST, f"season_outliers_thumb_{name}.png")
    im.convert("RGB").resize((1280, 720), Image.LANCZOS).save(out)
    print(out)


def thumb_d():
    """NOT EVEN CLOSE: every other season-leading performance as a dot, piled up; Wilt alone, far to the right"""
    sc = cfilm.FC["scale"]
    X = lambda z: 110 + (z + 3.0) * 158                     # noqa: E731
    im = base(glow_at=(int(X(7.05)), 470), strength=95, radius=380)
    dr = ImageDraw.Draw(im)
    rng = np.random.default_rng(7)
    edges, counts = sc["hist_edges"], list(sc["hist_counts"])
    film_z = sorted(c["z_e3"] for c in CARDS.values())    # the histogram stops at 4; the 25 beyond are in the tail
    pitch, rad, base_y = 8, 3.4, 958
    for i, n in enumerate(counts):
        cols = int((X(edges[i + 1]) - X(edges[i])) // pitch)
        for k in range(n):
            col, row = k % cols, k // cols
            x = X(edges[i]) + 4 + col * pitch + rng.uniform(-0.6, 0.6)
            y = base_y - row * pitch
            shade = 0.55 + 0.45 * rng.random()
            dr.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(int(70 * shade + 30), int(110 * shade + 30), int(170 * shade + 40), 255))
    rows = []                                               # x positions already used, per row
    for t in sorted(sc["tail"], key=lambda t: t["z"]):      # the other 20 past 4 (Vance 5.83, Curry, Ruth...), one dot each
        if t["top10"]:
            continue
        x = X(t["z"])
        row = 0
        while row < len(rows) and any(abs(x - u) < pitch for u in rows[row]):
            row += 1
        if row == len(rows):
            rows.append([])
        rows[row].append(x)
        y = base_y - row * pitch
        dr.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(100, 140, 210, 255))
    for z in film_z[:-1]:                                   # the other four of the five, small and gold
        x = X(z)
        dr.ellipse((x - 9, base_y - 9, x + 9, base_y + 9), fill=GOLD_RGB + (255,))
    wx = X(7.05)
    glow(im, (wx, base_y), 40, GOLD_RGB, 200)
    dr = ImageDraw.Draw(im)
    dr.ellipse((wx - 15, base_y - 15 - 22, wx + 15, base_y + 15 - 22), fill=GOLD_RGB + (255,))    # DiMaggio's, stacked
    dr.ellipse((wx - 15, base_y - 15, wx + 15, base_y + 15), fill=GOLD_RGB + (255,))
    dashed(dr, (wx, 640), (wx, base_y - 50), GOLD_RGB + (255,), width=5)
    disc(im, "chamberlain", 2, wx, 470, 330)
    dr = ImageDraw.Draw(im)
    ay = 455                                                # the gap: from the pile's peak to Wilt
    dashed(dr, (X(0.0), ay), (wx - 200, ay), GOLD_RGB + (255,), width=7, dash=26, gap=14)
    dr.polygon([(wx - 178, ay), (wx - 214, ay - 18), (wx - 214, ay + 18)], fill=GOLD_RGB + (255,))
    dr.polygon([(X(0.0) - 22, ay), (X(0.0) + 14, ay - 18), (X(0.0) + 14, ay + 18)], fill=GOLD_RGB + (255,))
    dr.text(((X(0.0) + wx - 200) / 2, ay - 22), "7 standard deviations", font=font("bold", 54), fill=GOLD_RGB, anchor="ms")
    dr.text((80, 50), "NOT EVEN", font=font("black", 176), fill=INK_RGB, anchor="lt")
    dr.text((80, 225), "CLOSE", font=font("black", 176), fill=GOLD_RGB, anchor="lt")
    dr.text((X(0.6), 1030), f"{sc['n'] - 5:,} other season-leading performances", font=font("semi", 40), fill=MUTED_RGB,
            anchor="ms")
    dr.text((wx, 1030), "Wilt, 1962", font=font("bold", 44), fill=GOLD_RGB, anchor="ms")
    finish(im, "D")


def thumb_e():
    """5 SEISMIC SEASONS: one trace through every season on the five's own graphs, in time order (each season's best by
    everyone else, 1926-2019, as standard deviations above its own season's elite curve), with the five as the spikes.
    Up is standard deviations, the same for every point, so the spikes are to scale."""
    X = lambda yr: 80 + (yr - 1925) * (1870 - 80) / (2020 - 1925)      # noqa: E731
    y0, sy = 840, 80
    Y = lambda z: y0 - sy * z                               # noqa: E731
    im = base()
    for c in CARDS.values():                                # a warm glow behind each spike
        glow(im, (int(X(c["year"])), int(Y(c["z_e3"]))), 190, GOLD_RGB, 70)
    dr = ImageDraw.Draw(im)
    for z in range(-1, 8):                                  # faint scale lines, one per standard deviation
        dr.line((70, Y(z), 1880, Y(z)), fill=(255, 255, 255, 34 if z else 70), width=2)
        dr.text((52, Y(z)), str(z), font=font("bold", 30), fill=GOLD_RGB if z >= 5 else MUTED_RGB, anchor="rm")

    rng = np.random.default_rng(3)
    pts, spikes = [], {}
    for c in CARDS.values():
        ru = c["ruler"]
        tr = np.array(ru["trend"])
        for s_, v_ in ru["window"]:
            pts.append((s_, rng.random(), (v_ - np.interp(s_, tr[:, 0], tr[:, 1])) / ru["step"], None))
        pts.append((c["year"], 0.5, c["z_e3"], c))          # the athlete's own season, mid-year
    pts.sort(key=lambda p: (p[0], p[1]))
    xy, per_year = [], {}
    for yr, _, z, c in pts:
        per_year.setdefault(yr, []).append((z, c))
    for yr in sorted(per_year):
        row = per_year[yr]
        for i, (z, c) in enumerate(row):
            x = X(yr) + (i + 0.5) / len(row) * (X(yr + 1) - X(yr)) - (X(yr + 1) - X(yr)) / 2
            xy.append((x, Y(z), c))
            if c:
                spikes[c["photo"]] = (len(xy) - 1, c)
    line = [(x, y) for x, y, _ in xy]
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).line(line, fill=(110, 170, 255, 255), width=10, joint="curve")
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(7)))
    dr = ImageDraw.Draw(im)
    dr.line(line, fill=(200, 222, 255, 255), width=3, joint="curve")
    D = 150
    shift = {"gretzky": (-20, -12), "rodman": (20, 6)}       # these two are seven years apart: keep the discs clear
    for key, (i, c) in spikes.items():
        seg = line[max(0, i - 1): i + 2]
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).line(seg, fill=GOLD_RGB + (255,), width=16, joint="curve")
        im.alpha_composite(g.filter(ImageFilter.GaussianBlur(6)))
        ImageDraw.Draw(im).line(seg, fill=(255, 214, 120, 255), width=5, joint="curve")
    for key, (i, c) in spikes.items():
        x, y = line[i]
        dx, dy = shift.get(key, (0, 0))
        disc(im, key, c.get("palette", c["rank"]), x + dx, y + dy, D, halo=False)
        ImageDraw.Draw(im).text((x - 14, y + dy + D / 2 + 16), str(c["year"]), font=font("bold", 34),
                                fill=GOLD_RGB, anchor="rt")
    dr = ImageDraw.Draw(im)
    fl = font("semi", 32)
    w = fl.getlength("the elite curve")
    dr.rounded_rectangle((1880 - w - 18, Y(0) - 26, 1892, Y(0) + 24), radius=10, fill=(11, 14, 19, 235),
                         outline=(255, 255, 255, 70), width=2)
    dr.text((1880 - w / 2 - 3, Y(0)), "the elite curve", font=fl, fill=(200, 204, 210), anchor="mm")
    f = font("black", 136)
    x = 70
    for word, col in (("5 ", INK_RGB), ("SEISMIC ", GOLD_RGB), ("SEASONS", INK_RGB)):
        dr.text((x, 34), word, font=f, fill=col, anchor="lt")
        x += f.getlength(word)
    dr.text((70, 1056), "Every season on their five charts, by year: standard deviations above the elite curve",
            font=font("semi", 31), fill=MUTED_RGB, anchor="ls")
    finish(im, "E")


def thumb_f():
    """OFF THE BELL CURVE: a bell curve that runs out by three standard deviations; the five sit out past five"""
    X = lambda z: 360 + z * 182                             # noqa: E731
    base_y, peak = 930, 520
    im = base(glow_at=(int(X(6.5)), 520), strength=80, radius=520)
    zz = np.linspace(-1.9, 7.6, 600)
    yy = base_y - peak * np.exp(-zz ** 2 / 2)
    fill = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(fill).polygon([(X(-1.9), base_y)] + [(X(z), y) for z, y in zip(zz, yy)] + [(X(7.6), base_y)],
                                 fill=(57, 135, 229, 120))
    im.alpha_composite(fill)
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).line([(X(z), y) for z, y in zip(zz, yy)], fill=(120, 180, 255, 255), width=14)
    im.alpha_composite(g.filter(ImageFilter.GaussianBlur(10)))
    dr = ImageDraw.Draw(im)
    dr.line([(X(z), y) for z, y in zip(zz, yy)], fill=(230, 240, 255, 255), width=6)
    dr.line((X(-1.9), base_y, X(7.6), base_y), fill=(150, 156, 166, 255), width=4)
    for k in range(0, 8):
        dr.line((X(k), base_y, X(k), base_y + 18), fill=(150, 156, 166, 255), width=4)
        dr.text((X(k), base_y + 26), f"{k}", font=font("bold", 40), fill=MUTED_RGB if k < 5 else GOLD_RGB, anchor="mt")
    dr.text((X(-1.9), base_y + 80), "standard deviations above the best of their era", font=font("semi", 38),
            fill=MUTED_RGB, anchor="lt")
    order = sorted(CARDS.values(), key=lambda c: c["z_e3"])        # Ichiro, Rodman, Gretzky, DiMaggio, Wilt
    spots = {"ichiro": 745, "rodman": 540, "gretzky": 745, "dimaggio": 520, "chamberlain": 300}
    nudge = {"ichiro": -40, "rodman": -50, "gretzky": 10, "dimaggio": 120, "chamberlain": -60}
    for c in order:
        z, key = c["z_e3"], c["photo"]
        x = X(z)
        cx, cy = x + nudge.get(key, 0), spots[key]
        d = 190 if key != "chamberlain" else 240
        dr = ImageDraw.Draw(im)
        dr.line((cx, cy + d / 2, x, base_y - 12), fill=GOLD_RGB + (255,), width=4)
        dr.ellipse((x - 11, base_y - 11, x + 11, base_y + 11), fill=GOLD_RGB + (255,))
        disc(im, key, c.get("palette", c["rank"]), cx, cy, d, halo=False)
    dr = ImageDraw.Draw(im)
    dr.text((70, 50), "OFF THE", font=font("black", 150), fill=INK_RGB, anchor="lt")
    dr.text((70, 200), "BELL CURVE", font=font("black", 150), fill=GOLD_RGB, anchor="lt")
    finish(im, "F")


if __name__ == "__main__":
    if WHICH != "c":
        raise SystemExit("only film C's thumbnails so far")
    if os.environ.get("THUMBS_FIRST_SET"):
        first_set()                                         # A-C: faces and numbers (the first style)
    thumb_d()
    thumb_e()
    thumb_f()
