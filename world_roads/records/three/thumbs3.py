"""Thumbnails for film C (1280×720, three for YouTube's thumbnail test), in the films' look: dark surface, gold, the
athletes' team colours.

    python thumbs3.py c     # -> ../../../videos/sports/thumbnails/season_outliers_thumb_{A,B,C}.png

A: the five, rising above the elite curve. B: DiMaggio's 56 against the longest streak by anyone else that year (24).
C: Gretzky's 163 assists, more than anyone else's points (141).
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


if __name__ == "__main__":
    if WHICH != "c":
        raise SystemExit("only film C's thumbnails so far")
    thumb_a()
    thumb_big(2, "56", "games in a row", "Anyone else\nthat year: 24", "Joe DiMaggio, 1941", "B")
    thumb_big(3, "163", "assists", "More than anyone\nelse's points", "Wayne Gretzky, 1985–86", "C")
