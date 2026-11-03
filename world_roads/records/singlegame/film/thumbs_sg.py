"""Thumbnails for the single-day film: A, Wilt's 100 against Kobe's 81; B, the four that stayed in the top ten.

    python thumbs_sg.py      # -> ../../../../videos/sports/thumbnails/single_day_thumb_{A,B}.png
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sdfilm                                              # noqa: E402  (sets up palettes and photo folders)
import scenes as SC                                        # noqa: E402
from look import W, H, INK, INK2, GOLD, new_fig, text, title, plt, BG   # noqa: E402
from PIL import Image                                      # noqa: E402

DST = os.path.join(HERE, "..", "..", "..", "..", "videos", "sports", "thumbnails")


def save(fig, name):
    os.makedirs(DST, exist_ok=True)
    tmp = os.path.join(sdfilm.look.OUT, f"_thumb_{name}.png")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    fig.savefig(tmp, dpi=100, facecolor=BG)
    plt.close(fig)
    Image.open(tmp).convert("RGB").resize((1280, 720), Image.LANCZOS).save(os.path.join(DST, f"single_day_thumb_{name}.png"))


def canvas():
    fig = new_fig(1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    return fig, ax


def thumb_a():
    fig, ax = canvas()
    for (key, pk, big, who), x in ((("chamberlain", "warriors", "100", "Wilt, 1962"), 120),
                                  (("bryant", "lakers", "81", "Kobe, 2006"), 1000)):
        ax.imshow(SC.face_disc(key, pk, 380), extent=(x, x + 380, 640, 260), zorder=3)
        title(fig, x + 410, 520, big, size=230, color=SC.PALETTE[pk]["accent"])
        text(fig, x + 420, 610, who, size=50, color=INK2, weight="semibold")
    title(fig, W / 2, 170, "One game. Off the charts.", size=96, color=INK, ha="center")
    text(fig, W / 2, 900, "The 10 biggest single-day outliers in sports history", size=48, color=GOLD, ha="center", weight="semibold")
    save(fig, "A")


def thumb_b():
    fig, ax = canvas()
    items = [("beamon", "usa", "8.90 m", "Beamon"), ("chamberlain", "warriors", "100", "Chamberlain"),
             ("zelezny", "czech", "98.48 m", "Železný"), ("bryant", "lakers", "81", "Bryant")]
    for i, (key, pk, big, who) in enumerate(items):
        x, y = 70 + (i % 2) * 930, 230 + (i // 2) * 410
        ax.imshow(SC.face_disc(key, pk, 330), extent=(x, x + 330, y + 330, y), zorder=3)
        title(fig, x + 360, y + 200, big, size=150, color=INK)
        text(fig, x + 366, y + 280, who, size=48, color=SC.PALETTE[pk]["accent"], weight="semibold")
    title(fig, W / 2, 150, "Which single day stood furthest out?", size=80, color=GOLD, ha="center")
    save(fig, "B")


if __name__ == "__main__":
    thumb_a()
    thumb_b()
    print(os.path.abspath(DST))
