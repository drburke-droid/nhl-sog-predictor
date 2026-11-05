"""The 10 Biggest Statistical Outliers in Sports History: the sections of the film.

    python film.py <section> [scale]     # cold curve method fair scale c10 ... c1 throws close end -> out/<section>.mp4
    python film.py stills                # a still of every section at its busiest moment -> out/stills/

Every number on screen comes from facts.json (python facts.py), built from the locked model E3_main_env of the
follow-up analysis. The look (surface, type, gold and blue) is the records film's (../look.py).
"""
import json, os, sys, math

import numpy as np
import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, REC)
import look                                                    # noqa: E402
from look import (W, H, BG, INK, INK2, MUTED, GRID, AXIS, GOLD, BLUE, GREY, px, fs, clamp, ease, ease_out, fade,  # noqa: E402
                  ax_px, text, title, style_axes, ring, outlined)
from matplotlib.ticker import MaxNLocator, FuncFormatter        # noqa: E402

look.OUT = os.path.join(HERE, "out")
F = json.load(open(os.path.join(HERE, "facts.json"), encoding="utf-8"))
CARDS = {c["rank"]: c for c in F["cards"]}
TREND = "#9a9890"                                              # the trend of everyone else (neutral ink, not a series colour)


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def para(fig, x, y, s, width=46, size=27, color=INK2, alpha=1.0, lh=1.32, weight="regular"):
    """a wrapped paragraph; returns the y below it"""
    for i, line in enumerate(wrap(s, width)):
        text(fig, x, y + i * size * lh, line, size=size, color=color, alpha=alpha, weight=weight)
    return y + len(wrap(s, width)) * size * lh


def fmt_value(cat, v):
    if cat in ("ath_M_JT", "ath_M_LJ"):
        return f"{v:.2f} m"
    if cat in ("nba_ppg", "nba_rpg"):
        return f"{v:.1f}"
    return f"{v:.0f}"


def unit_word(cat):
    return {"mlb_h": "hits", "mlb_hr": "home runs", "nba_rpg": "rebounds a game", "nba_3pm": "three-pointers",
            "mlb_so": "strikeouts", "ath_M_JT": "", "nhl_assists": "assists", "mlb_streak": "games",
            "nba_ppg": "points a game", "ath_M_LJ": ""}[cat]


ENV_LABEL = {"mlb_h": "League hits per team per game", "mlb_hr": "League home runs per team per game",
             "nba_rpg": "League rebounds per team per game", "nba_3pm": "League three-pointers per team per game",
             "mlb_so": "League strikeouts per team per game", "nhl_assists": "League assists per team per game",
             "nba_ppg": "League points per team per game"}

SPORT_WORD = {"MLB": "Baseball", "NBA": "Basketball", "NHL": "Hockey", "NFL": "Football", "Athletics": "Athletics",
              "Swimming": "Swimming"}

RATE_WORD = {"mlb_h": "hit rate", "mlb_hr": "home-run rate", "nba_rpg": "rebounding rate", "nba_3pm": "three-point rate",
             "mlb_so": "strikeout rate", "nhl_assists": "assist rate", "nba_ppg": "scoring rate"}
RULER_UNIT = {"mlb_h": "hits", "mlb_hr": "home runs", "nba_rpg": "rebounds", "nba_3pm": "threes", "mlb_so": "strikeouts",
              "ath_M_JT": "m", "nhl_assists": "assists", "mlb_streak": "games", "nba_ppg": "points", "ath_M_LJ": "cm"}

# portraits (Wikimedia Commons, free licences; photos.py) and their credit lines
PHOTO_KEY = {10: "ichiro", 9: "ruth", 8: "rodman", 7: "curry", 6: "vance", 5: "zelezny", 4: "gretzky", 3: "dimaggio",
             2: "chamberlain", 1: "beamon"}
CREDITS = json.load(open(os.path.join(HERE, "photos", "credits.json"), encoding="utf-8"))
_PH = {}


def photo(fig, key, x0, y0, x1, y1, alpha=1.0, credit=True, credit_size=14):
    if alpha <= 0.001:
        return
    if key not in _PH:
        from PIL import Image
        import scenes as _SC
        _PH[key] = np.asarray(Image.open(_SC.photo_path(key)).convert("RGB"))
    ax = ax_px(fig, x0, y0, x1, y1)
    ax.set_axis_off()
    ax.imshow(_PH[key], alpha=clamp(alpha), interpolation="antialiased", aspect="auto")
    if credit:
        text(fig, x1, y1 + 20, CREDITS[key]["short"], size=credit_size, color=MUTED, ha="right", alpha=alpha)


def sig(v, n):
    """v to n significant figures, as plain text (23.0 -> '23', 5.586 -> '5.6')"""
    d = max(0, n - 1 - int(math.floor(math.log10(abs(v))))) if v else 0
    out = f"{round(v, d):.{d}f}"
    return out.rstrip("0").rstrip(".") if "." in out and d > 0 and float(out) == round(v, d) and out.endswith("0") and n > 2 else out


CM_CATS = ("ath_M_LJ", "ath_W_TJ", "ath_M_PV")


def ruler_numbers(c):
    """strings for '(value - curve) / step = score', rounded so the arithmetic on screen gives the score on screen"""
    r, cat = c["ruler"], c["category"]
    k = 100.0 if cat in CM_CATS else 1.0                      # steps of a few centimetres: shown in centimetres
    x, b, st = r["x"] * k, r["base"] * k, r["step"] * k
    best = None
    for db in range(0, 4):
        for ds in range(0, 4):
            bb, ss = round(b, db), round(st, ds)
            if ss > 0 and abs(round((x - bb) / ss, 2) - c["z_e3"]) < 1e-9 and (best is None or db + ds < best[0]):
                best = (db + ds, db, ds)
    assert best, cat
    dx = 0 if float(x).is_integer() else (2 if cat.startswith("ath_") else 1)          # marks keep both decimals
    return f"{x:.{dx}f}", f"{b:.{best[1]}f}", f"{st:.{best[2]}f}"


def card_lines(c):
    """three context lines per card, every number from facts.json"""
    r, ex = c["rank"], F["extras"]
    ru = c["runner_up"]
    pr = c["previous_record"]
    nr = c["next_record"]
    if r == 10:
        return [f"262 hits broke George Sisler's record of {float(pr['value']):.0f}, set in {pr['year']}.",
                f"Next best in 2004: {ru['name']}, {float(ru['value']):.0f}.",
                "Other seasons are converted to 2004's league hit rate, the way old prices are adjusted for inflation."]
    if r == 9:
        return [f"His own record from the year before was {float(pr['value']):.0f}.",
                f"Next best in 1920: {ru['name']}, {float(ru['value']):.0f}. The average team hit {ex['mlb_1920_avg_team_hr']:.0f}.",
                f"Without the league adjustment his score is {c['z_main_no_env']:.1f}: home runs were that rare in 1920."]
    if r == 8:
        return [f"Next best that season: {ru['name']}, {float(ru['value']):.1f}.",
                f"His 18.7 a year earlier scores lower ({F['extras']['rodman_1992_z']:.2f}) once league rebounding is allowed for.",
                "He led the league in rebounding seven seasons running, 1991–92 to 1997–98."]
    if r == 7:
        return [f"The previous record was his own: {float(pr['value']):.0f}, the season before.",
                f"Next best: teammate {ru['name']}, {float(ru['value']):.0f}.",
                "The score allows for the league's three-point rate, which has kept rising since."]
    if r == 6:
        return [f"Next best in either league: {ru['name']}, {float(ru['value']):.0f}.",
                f"Strikeouts were scarce: about {ex['mlb_1924_so_per_team_game']:.1f} per team per game.",
                f"Without the league adjustment his score is {c['z_main_no_env']:.1f}; scarcity is what lifts it."]
    if r == 5:
        return ["98.48 m with the javelin redesigned in 1986 — still the world record.",
                f"Next best in 1996: {ru['name']}, {float(ru['value']):.2f} m.",
                f"It broke his own record of {float(pr['value']):.2f} m, set in {pr['year']}."]
    if r == 4:
        nxt = ex["nhl_1986_points_leaders"][1]
        return [f"Next best in assists: {ru['name']}, {float(ru['value']):.0f}.",
                f"His assists alone topped every other player's points: {nxt[0].split()[-1]} had {float(nxt[1]):.0f}.",
                f"Scoring was near its peak, which brings his score down from {c['z_main_no_env']:.2f} to {c['z_e3']:.2f}."]
    if r == 3:
        ls = ex["longest_streak_since_1941"]
        return [f"The record before: {float(pr['value']):.0f} games, {pr['holder']}, {pr['year']}.",
                f"Longest by anyone else in 1941: {float(ru['value']):.0f} games.",
                f"The closest since: {ls[2]}'s {float(ls[0]):.0f}, in {ls[1]}."]
    if r == 2:
        return [f"Next best: {ru['name']}, {float(ru['value']):.1f} — {50.4 - float(ru['value']):.1f} points behind.",
                f"He played {ex['wilt_minutes_per_game']:.1f} minutes a game, overtimes included.",
                f"Teams averaged {ex['nba_points_per_team_game_1962']:.1f} points; adjusting for that, {c['z_main_no_env']:.2f} becomes {c['z_e3']:.2f}."]
    if r == 1:
        return [f"One jump, not a season total: {100 * (8.90 - float(pr['value'])):.0f} cm beyond the world record.",
                f"Next best that year: {float(ru['value']):.2f} m. The record stood until {nr['holder']}'s {float(nr['value']):.2f} m in {nr['year']}.",
                "It came at 2,240 m altitude, with a +2.0 m/s tailwind, the legal maximum."]
    return []


SECTIONS = {}


def section(fn):
    SECTIONS[fn.__name__] = fn
    return fn


# ------------------------------------------------------------------ cold open
class ColdOpen:
    dur = 24.0
    ITEMS = [("8.90 m", "long jump · Bob Beamon · 1968"), ("50.4", "points a game · Wilt Chamberlain · 1961–62"),
             ("56", "games in a row with a hit · Joe DiMaggio · 1941"), ("163", "assists · Wayne Gretzky · 1985–86")]

    def draw(self, fig, t):
        pos = [(540, 360), (1380, 360), (540, 720), (1380, 720)]
        out = fade(t, 0, 11.6, 0, 0.6)
        for i, ((big, lab), (x, y)) in enumerate(zip(self.ITEMS, pos)):
            a = out * fade(t, 0.6 + 1.5 * i, 99, 0.7, 0)
            title(fig, x, y, big, size=128, color=INK, ha="center", alpha=a)
            text(fig, x, y + 58, lab, size=27, color=INK2, ha="center", alpha=a)
        text(fig, W / 2, 985, "Which of these stood furthest above everyone else of its time?", size=36, color=INK,
             ha="center", alpha=out * fade(t, 7.0, 99, 0.8, 0), weight="medium")
        a = fade(t, 12.2, self.dur, 1.0, 0.7)
        title(fig, W / 2, 400, "The 10 Biggest", size=104, ha="center", alpha=a)
        title(fig, W / 2, 520, "Statistical Outliers", size=104, ha="center", alpha=a)
        title(fig, W / 2, 640, "in Sports History", size=104, ha="center", alpha=a, color=GOLD)
        text(fig, W / 2, 745, "74 categories · 6 sports · every season measured against its own era", size=31, color=INK2,
             ha="center", alpha=a * fade(t, 13.4, self.dur, 1.0, 0.7))


@section
def cold():
    return ColdOpen()


# ------------------------------------------------------------------ the hook: 4,940 scores, ten that broke the curve
class Hook:
    dur = 30.0
    ORDER = [3, 7, 1, 9, 5, 2, 10, 6, 4, 8]                 # faces in a teaser order, not rank order

    def __init__(self):
        import scenes as SC
        self.SC = SC
        sc = F["scale"]
        self.sc = sc
        self.edges = sc["hist_edges"]
        self.counts = sc["hist_counts"]
        self.tail = sc["tail"]
        self.cmax = max(self.counts)
        rs = np.random.default_rng(4940)
        p = np.array(self.counts, float) / sum(self.counts)
        self.parts = [(int(rs.choice(len(p), p=p)), float(rs.uniform(2.4, 8.2)), float(rs.uniform(-0.18, 0.18)))
                      for _ in range(320)]
        h = {}
        self.tpos = []
        for d in sorted(self.tail, key=lambda d: d["z"]):
            b = math.floor(d["z"] / 0.25) * 0.25
            k = h.get(b, 0)
            h[b] = k + 1
            self.tpos.append((d, b + 0.125, k))
        self.N = F.get("n_cards", 10)                       # a five-card film passes n_cards and hook_order
        if F.get("hook_order"):
            self.ORDER = F["hook_order"]
        self.faces = {r: SC.face_disc(CARDS[r].get("photo") or PHOTO_KEY[r], CARDS[r].get("palette", r), 128)
                      for r in range(1, self.N + 1)}
        self.words = F.get("hook_words", {})
        self.slot = {r: (W / 2 + (i - (self.N - 1) / 2) * 158, 250) for i, r in enumerate(self.ORDER)}
        self.rank_of = {CARDS[r]["athlete"]: r for r in range(1, self.N + 1)}

    def X(self, z):
        return 200 + (z + 4.3) / 12.6 * 1520

    def Y(self, n):
        return 900 - n / self.cmax * 360

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.4, 0.8)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ta = al * fade(t, 0.2, 15.6, 0.6, 0.6)
        title(fig, W / 2, 170, self.words.get("headline", f"{self.sc['n']:,} season-leading performances."), size=70,
              ha="center", alpha=ta)
        text(fig, W / 2, 228, self.words.get("sub", "74 categories · six sports · 1876 to 2026"), size=30, color=INK2, ha="center",
             alpha=ta * fade(t, 1.4, 99, 0.6, 0))
        ca = al * fade(t, 2.4, 21.6, 0.6, 0.8)                  # the chart
        if ca > 0:
            ax.plot([self.X(-4.3), self.X(8.3)], [900, 900], color="#5a5f66", lw=px(1.5), alpha=ca)
            for z in range(-4, 9, 2):
                ax.text(self.X(z), 930, f"{z}", ha="center", va="center", fontsize=fs(19), color=MUTED, alpha=ca)
            ax.text(self.X(8.3), 962, "score: steps above everyone else of their era", ha="right", va="center",
                    fontsize=fs(18), color=MUTED, alpha=ca)
            g = ease(clamp((t - 2.6) / 6.0))
            for lo, n in zip(self.edges[:-1], self.counts):
                if n:
                    h = (900 - self.Y(n)) * g
                    ax.add_patch(matplotlib.patches.Rectangle((self.X(lo) + 2, 900 - h), self.X(lo + 0.5) - self.X(lo) - 4, h,
                                                              color=BLUE, alpha=0.85 * ca, lw=0, zorder=2))
            for b, t0, jit in self.parts:                       # the rain
                u = (t - t0) / 0.55
                if 0 <= u <= 1:
                    lo = self.edges[b]
                    x = self.X(lo + 0.25 + jit)
                    y = 300 + (self.Y(self.counts[b] * g) - 300) * u * u
                    ax.plot([x], [y], marker="o", ms=px(5), color=BLUE, alpha=ca * (1 - 0.3 * u), zorder=3, mec="none")
            ma = ca * fade(t, 10.0, 99, 0.6, 0)
            if ma > 0:
                xm = self.X(self.sc["median"])
                ax.plot([xm, xm], [self.Y(self.cmax) - 30, 900], color=TREND, lw=px(2.5), ls=(0, (3, 2)), alpha=ma, zorder=4)
            tt = ca * fade(t, 12.4, 99, 0.6, 0)
            fly = ease(clamp((t - 16.0) / 2.0))
            for d, zc, k in self.tpos:
                x0, y0 = self.X(zc), 888 - 22 * k
                if d["top10"]:
                    r = self.rank_of.get(d["name"])
                    if r is None:
                        continue
                    glow_a = fade(t, 15.0, 99, 0.5, 0)
                    tx, ty = self.slot[r]
                    x, y = x0 + (tx - x0) * fly, y0 + (ty + 70 - y0) * fly - 160 * math.sin(math.pi * fly)
                    if fly < 0.98:
                        ax.plot([x], [y], marker="o", ms=px(11 + 6 * glow_a), color=GOLD, alpha=tt, zorder=6,
                                mec="#ffe9b0", mew=px(1.5 * glow_a))
                elif tt > 0:
                    ax.plot([x0], [y0], marker="o", ms=px(9), color=BLUE, alpha=tt, zorder=5, mec="none")
        caps = [(7.2, 10.0, self.words.get("graded", "Each one graded against everyone else of its era.")),
                (10.0, 12.4, "Most land near zero. Leading your league is normal."),
                (12.4, 15.6, f"Only {self.sc['at_least_5']} of {self.sc['n']:,} reach 5."),
                (18.0, 21.6, self.words.get("ten", "Ten broke the curve."))]
        for t0, t1, msg in caps:
            a = al * fade(t, t0, t1, 0.4, 0.4)
            if a > 0:
                title(fig, W / 2, 1030 if t0 < 18 else 470, msg, size=40 if t0 < 18 else 64, ha="center",
                      color=INK if t0 < 18 else GOLD, alpha=a)
        fa = al * fade(t, 17.4, 99, 0.4, 0)                     # the faces
        if fa > 0:
            for r, (x, y) in self.slot.items():
                pop = ease_out(clamp((t - 17.4 - 0.06 * self.ORDER.index(r)) / 0.5))
                if pop > 0:
                    d = 128 * (0.6 + 0.4 * pop)
                    ax.imshow(self.faces[r], extent=(x - d / 2, x + d / 2, y + d / 2, y - d / 2), alpha=fa * pop, zorder=8)
        ha = al * fade(t, 21.8, 99, 0.7, 0)                     # the title
        if ha > 0:
            title(fig, W / 2, 560, self.words.get("title1", "The 10 Biggest"), size=104, ha="center", alpha=ha)
            title(fig, W / 2, 680, self.words.get("title2", "Statistical Outliers"), size=104, ha="center", alpha=ha)
            title(fig, W / 2, 800, self.words.get("title3", "in Sports History"), size=104, ha="center", color=GOLD, alpha=ha)
            text(fig, W / 2, 900, self.words.get("countdown", "Counting down from ten"), size=32, color=INK2, ha="center",
                 alpha=ha * fade(t, 24.5, 99, 0.6, 0))


@section
def hook():
    return Hook()


# ------------------------------------------------------------------ how the score works, 1: grading on a curve
class Curve:
    """the analogy: a class test. 95 in a class averaging 70 (give or take 5) is five steps up; in a class
    averaging 90 it is one. Illustrative numbers, not data."""
    dur = 44.0

    def __init__(self):
        rng = np.random.default_rng(11)
        z = rng.normal(size=31)
        self.zs = (z - z.mean()) / z.std()
        self.A = self.layout(70.0)
        self.B = self.layout(90.0)

    def layout(self, mean):
        xs = np.minimum(np.round((mean + 5.0 * self.zs) / 2.0) * 2.0, 100.0)
        pts, h = [], {}
        for x in list(xs) + [95.0]:                      # the gold student last, on top of their column
            k = h.get(x, 0)
            h[x] = k + 1
            pts.append((x, 0.7 + 1.05 * k))
        return np.array(pts)

    def chart(self, fig, t, al):
        ax = ax_px(fig, 150, 330, 1100, 900)
        style_axes(ax, grid_y=False)
        ax.set_yticks([])
        ax.set_xlim(37, 103)
        ax.set_ylim(0, 17)
        ax.set_xticks(range(40, 101, 10))
        for lab in ax.get_xticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        ax.text(0, -0.13, "test score", transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
        mv = ease(clamp((t - 26.0) / 3.5))                 # the class moves from average 70 to average 90
        P = self.A * (1 - mv) + self.B * mv
        mean = 70.0 + 20.0 * mv
        pop = clamp((t - 2.0) / 3.0)
        n = len(P) - 1
        for i, (x, y) in enumerate(P[:-1]):
            a = al * clamp(pop * n - i * 0.9)
            if a > 0:
                ring(ax, x, y, BLUE, size=13, alpha=a, z=5)
        ma = al * fade(t, 6.5, 99, 0.8, 0)
        if ma > 0:
            ax.plot([mean, mean], [0, 12.2], color=TREND, lw=px(3), alpha=ma, zorder=3)
            ax.text(mean, 12.6, f"class average {mean:.0f}", ha="center", fontsize=fs(21), color=INK2, alpha=ma)
        ba = al * fade(t, 9.0, 99, 0.8, 0)
        if ba > 0:
            ax.axvspan(mean - 5, mean + 5, ymin=0, ymax=12.2 / 17, color=TREND, alpha=0.16 * ba, lw=0, zorder=1)
            if t < 26.0:
                ax.annotate("", (mean + 5, 10.3), (mean, 10.3), arrowprops=dict(arrowstyle="<->", color=INK2, lw=px(2)),
                            alpha=ba)
                ax.text(mean + 5.8, 10.3, "one step: 5 points", ha="left", va="center", fontsize=fs(19), color=INK2,
                        alpha=ba)
        ga = al * fade(t, 15.0, 99, 0.6, 0)
        if ga > 0:
            gx, gy = P[-1]
            ring(ax, gx, gy, GOLD, size=17, alpha=ga, z=8)
            ax.annotate("95", (gx, gy), xytext=(px(14), px(6)), textcoords="offset points", fontsize=fs(23), color=INK,
                        weight="semibold", alpha=ga, zorder=9, path_effects=outlined(6))
            # the ruler: from the class average to 95, one tick per 5 points
            for t0, m0 in ((16.0, 70.0), (30.0, 90.0)):
                ra = ga * fade(t, t0, 26.0 if m0 == 70.0 else 99, 0.4, 0.8)
                if ra <= 0:
                    continue
                prog = ease(clamp((t - t0) / 3.5))
                end = m0 + (95.0 - m0) * prog
                yr = 14.6
                ax.plot([m0, end], [yr, yr], color=GOLD, lw=px(3), alpha=ra, zorder=7)
                ax.plot([95, 95], [gy + 0.6, yr], color=GOLD, lw=px(1.5), ls=(0, (2, 2)), alpha=ra * prog, zorder=6)
                for k in range(0, int((end - m0) / 5.0 + 1e-6) + 1):
                    xk = m0 + 5.0 * k
                    ax.plot([xk, xk], [yr - 0.45, yr + 0.45], color=GOLD, lw=px(2.5), alpha=ra, zorder=7)
                    if k:
                        ax.text(xk - 2.5, yr + 0.75, f"{k}", ha="center", fontsize=fs(19), color=GOLD, alpha=ra)
        return ax

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE SCORE WORKS", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "Grading on a curve", size=58, alpha=al)
        self.chart(fig, t, al)
        caps = [(1.5, "Picture a class test. Most of the class scores around 70, give or take about 5 points."),
                (9.0, "Call that usual give-or-take one step."),
                (15.0, "A student who scores 95 is 25 points above the class average: five steps. On our scale, that's "
                       "a score of 5."),
                (26.0, "Now put the same 95 in a class that averages 90. It's one step above: a score of 1."),
                (35.0, "Same number, different class. So every result in this film is graded against the class of its "
                       "own era.")]
        y = 330
        for t0, s in caps:
            a = al * fade(t, t0, 99, 0.8, 0)
            if a > 0:
                y = para(fig, 1190, y, s, width=40, size=27, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                         alpha=a) + 22
        # the counter
        if t > 16.0:
            v = 5.0 * ease(clamp((t - 16.0) / 3.5)) if t < 28.0 else 1.0 * ease(clamp((t - 30.0) / 3.5))
            ca = al * (fade(t, 16.0, 26.0, 0.5, 0.6) + fade(t, 30.0, 99, 0.5, 0))
            title(fig, 1190, 1035, f"{v:.0f}" if v >= 1 or t > 33.5 else "0", size=72, color=GOLD, alpha=ca)
            word = "step" if f"{v:.0f}" == "1" and t > 28.0 else "steps"
            text(fig, 1260, 1030, f"{word} above the class average", size=24, color=INK2, alpha=ca)


@section
def curve():
    return Curve()


# ------------------------------------------------------------------ how the score works, 2: the class in sport (Ledecky 2016)
class Method:
    dur = 52.0

    def __init__(self):
        m = F["method_example"]
        self.m = m
        self.pts = [(p["season"], p["time"]) for p in m["window"]]
        self.curve = np.array(m["trend_time"])
        self.z = float(m["z"])
        self.sp_base, self.sp_scale = m["base_speed"], m["scale_speed"]

    def chart(self, fig, t, al):
        m = self.m
        ax = ax_px(fig, 150, 330, 1100, 930)
        style_axes(ax)
        times = [p[1] for p in self.pts] + [m["x_time"], m["baseline_time"]]
        ax.set_xlim(2000.3, 2026.7)
        ax.set_ylim(max(times) + 3.0, min(times) - 3.0)            # faster is up
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v // 60)}:{v % 60:04.1f}"))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        ax.text(0, 1.04, "Women's 800 m freestyle · ▲ faster", transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
        sweep = 2001 + (2026.9 - 2001) * ease((t - 2.5) / 9.0)
        for s, v in self.pts:
            if s <= sweep:
                pop = ease_out((sweep - s) / 1.5)
                ring(ax, s, v, BLUE, size=12 * (0.5 + 0.5 * pop), alpha=al * pop, z=5)
        tc = clamp((t - 16.0) / 5.0)
        if tc > 0:
            k = max(2, int(len(self.curve) * tc))
            ax.plot(self.curve[:k, 0], self.curve[:k, 1], color=TREND, lw=px(3), alpha=al, zorder=4)
        ba = fade(t, 24.0, 99, 1.2, 0) * al
        if ba > 0:
            x = self.curve[:, 0]
            sp = 800 / self.curve[:, 1]
            ax.fill_between(x, 800 / (sp - self.sp_scale), 800 / (sp + self.sp_scale), color=TREND, alpha=0.16 * ba,
                            lw=0, zorder=2)
        la = fade(t, 32.0, 99, 0.8, 0) * al
        if la > 0:
            ring(ax, 2016, m["x_time"], GOLD, size=17, alpha=la, z=8)
            ax.annotate("Katie Ledecky, 2016: 8:04.79", (2016, m["x_time"]), xytext=(px(16), px(10)),
                        textcoords="offset points", fontsize=fs(23), color=INK, alpha=la, zorder=9,
                        path_effects=outlined(6), weight="semibold")
            prog = ease(clamp((t - 33.0) / 4.0))
            top = self.sp_base + (self.z * self.sp_scale) * prog
            ax.plot([2016, 2016], [800 / self.sp_base, 800 / top], color=GOLD, lw=px(2.5), alpha=la, ls=(0, (2, 2)), zorder=7)
            for k in range(1, int(self.z * prog) + 1):
                yk = 800 / (self.sp_base + k * self.sp_scale)
                ax.plot([2015.6, 2016.4], [yk, yk], color=GOLD, lw=px(2), alpha=la, zorder=7)
                ax.text(2015.3, yk, f"{k}", ha="right", va="center", fontsize=fs(17), color=GOLD, alpha=la, zorder=7)
        return ax

    def draw(self, fig, t):
        m = self.m
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE SCORE WORKS", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "In sport, the class is everyone else", size=58, alpha=al)
        photo(fig, "ledecky", 1648, 24, 1824, 244, alpha=al * fade(t, 30.0, 99, 0.8, 0))
        self.chart(fig, t, al)
        caps = [(1.5, "Every season's leader, in 74 categories across six sports, is graded against a class: the best "
                      "of everyone else in each of the 15 seasons before and after."),
                (8.0, "Example: the women's 800 m freestyle. Each blue dot is the fastest time that season by anyone "
                      "except Katie Ledecky."),
                (16.0, "The curve is the class average, moving with the times."),
                (24.0, f"The band is the usual give-or-take. Here, one step is about {m['step_seconds']:.1f} seconds."),
                (32.0, f"Ledecky's 8:04.79 in 2016 beat the curve by {m['gap_seconds']:.1f} seconds: {self.z:.1f} steps. "
                       f"Her score is {self.z:.1f}."),
                (41.0, f"That ranks {m['rank_e3']}th of everything we measured. The ten in this film stood out even "
                       "further.")]
        y = 330
        for t0, s in caps:
            a = al * fade(t, t0, 99, 0.8, 0)
            if a > 0:
                y = para(fig, 1190, y, s, width=40, size=26, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                         alpha=a) + 20
        if t > 33.0:
            cnt = self.z * ease(clamp((t - 33.0) / 4.0))
            ca = al * fade(t, 33.0, 99, 0.5, 0)
            title(fig, 150, 1040, f"{cnt:.1f}", size=64, color=GOLD, alpha=ca)
            text(fig, 270, 1035, "steps above the curve", size=24, color=INK2, alpha=ca)


@section
def method():
    return Method()


# ------------------------------------------------------------------ how the score works, 3: three rules that keep it fair
class Fair:
    dur = 40.0

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE SCORE WORKS", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, F.get("fair_title", "Three rules that keep it fair"), size=58, alpha=al)
        ex = F["extras"]
        rules = F.get("fair_rules") or [(1.0, "Adjusted for inflation",
                  "When every team scores more, each point is worth a little less, like money in an inflated economy. "
                  "So in team sports, everyone else's numbers are converted into the athlete's own season before they're "
                  f"compared. Wilt's 50.4 came when teams averaged {ex['nba_points_per_team_game_1962']:.1f} points a "
                  "game. How much to convert is estimated from the data."),
                 (12.0, "Graded against rivals, not yourself",
                  "An athlete's own other seasons are left out of the class. Otherwise Gretzky's 1985–86 would be graded "
                  "against Gretzky."),
                 (20.0, "One entry each",
                  "Each athlete appears once, with their highest-scoring season. A few entries are a single mark or a "
                  "streak rather than a season total: Beamon's one jump, DiMaggio's 56 games.")]
        y = 270
        tight = len(rules) > 3                                  # four rules: a smaller body so they all fit
        for i, (t0, head, body) in enumerate(rules):
            a = al * fade(t, t0, 99, 0.8, 0)
            title(fig, 150, y + 52, f"{i + 1}", size=64 if not tight else 56, color=GOLD, alpha=a)
            title(fig, 250, y + 40, head, size=40 if not tight else 34, alpha=a)
            y = para(fig, 250, y + (88 if not tight else 76), body, width=78 if not tight else 98,
                     size=28 if not tight else 23, color=INK2, alpha=a) + (34 if not tight else 22)
        para(fig, 250, max(y + 10, 900), "The score measures distance from the class, not talent, and it isn't a "
                                         "probability. The order depends on these choices: more on that at the end.",
             width=86, size=24, color=MUTED, alpha=al * fade(t, 28.0 if not tight else 33.0, 99, 0.8, 0))


@section
def fair():
    return Fair()


# ------------------------------------------------------------------ how the score works, 4: what's a good score?
class Scale:
    dur = 34.0

    def __init__(self):
        self.s = F["scale"]
        extra = sorted((d["name"], d["year"]) for d in self.s["tail"] if d["z"] >= 5 and not d["top10"])
        assert extra == [("Bob Feller", 1946), ("Dennis Rodman", 1992)], extra

    def chart(self, fig, t, al):
        s = self.s
        ax = ax_px(fig, 150, 330, 1100, 900)
        style_axes(ax)
        ax.set_xlim(-4.3, 8.3)
        ax.set_ylim(0, 1080)
        ax.set_xticks(range(-4, 9))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        ax.text(0, 1.04, "number of season leaders, by score", transform=ax.transAxes, fontsize=fs(19), color=MUTED,
                alpha=al)
        ax.text(1, -0.11, "score: steps above the curve", transform=ax.transAxes, ha="right", fontsize=fs(19), color=MUTED,
                alpha=al)
        g = ease(clamp((t - 2.0) / 4.0))
        edges = s["hist_edges"]
        for lo, n in zip(edges[:-1], s["hist_counts"]):
            if n:
                ax.bar(lo + 0.25, n * g, width=0.44, color=BLUE, alpha=al * 0.9, lw=0, zorder=3)
        ma = al * fade(t, 7.0, 99, 0.8, 0)
        if ma > 0:
            ax.plot([s["median"]] * 2, [0, 1040], color=TREND, lw=px(2.5), ls=(0, (3, 2)), alpha=ma, zorder=4)
            ax.text(s["median"], 1050, "a typical league leader", ha="center", fontsize=fs(21), color=INK2, alpha=ma)
        ta = al * fade(t, 15.0, 99, 0.8, 0)
        if ta > 0:
            ax.plot([4, 4], [0, 520], color=GRID, lw=px(2), zorder=2)
            ax.text(4.1, 540, "beyond 4: one dot per season", fontsize=fs(19), color=INK2, alpha=ta)
            h = {}
            for d in sorted(s["tail"], key=lambda d: d["z"]):
                b = math.floor(d["z"] / 0.25) * 0.25
                k = h.get(b, 0)
                h[b] = k + 1
                gold = d["top10"] and t >= 21.0
                ring(ax, b + 0.125, 22 + 36 * k, GOLD if gold else BLUE, size=10, alpha=ta, z=6)
        la = al * fade(t, 21.5, 99, 0.8, 0)
        if la > 0:
            top = [d["z"] for d in s["tail"] if d["top10"]]
            ax.plot([min(top), max(top)], [300, 300], color=GOLD, lw=px(2), alpha=la)
            for xv in (min(top), max(top)):
                ax.plot([xv, xv], [280, 320], color=GOLD, lw=px(2), alpha=la)
            ax.text((min(top) + max(top)) / 2, 335, "the ten in this film", ha="center", fontsize=fs(21), color=GOLD,
                    weight="semibold", alpha=la)
        return ax

    def draw(self, fig, t):
        s = self.s
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE SCORE WORKS", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "What's a good score?", size=58, alpha=al)
        self.chart(fig, t, al)
        caps = [(1.0, f"We graded all {s['n']:,} season leaders in the data this way."),
                (7.0, "Leading your league is normal: somebody does it every season. A typical leader scores about 0, "
                      "right on the curve."),
                (15.0, f"Three in four score below 1. Only {s['at_least_4']} reach 4."),
                (21.0, f"Only {s['at_least_5']} reach 5: the ten athletes in this film, plus Rodman's 1991–92 and Bob "
                       "Feller's 1946."),
                (28.0, "Here they are, from ten to one.")]
        y = 330
        for t0, c_ in caps:
            a = al * fade(t, t0, 99, 0.8, 0)
            if a > 0:
                y = para(fig, 1190, y, c_, width=40, size=27, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                         alpha=a) + 22


@section
def scale():
    return Scale()


# ------------------------------------------------------------------ the countdown cards
class Card:
    """one countdown card, in the colours of the athlete's team (or country), with a sport-specific chart"""
    dur = 36.0

    def __init__(self, rank):
        import scenes as SC
        self.SC = SC
        self.c = CARDS[rank]
        pk = self.c.get("palette", rank)
        self.pal = SC.PALETTE[pk]
        self.acc = self.pal["accent"]
        self.first = F.get("n_cards", 10)                   # the first card shown explains how to read the chart
        self.scene = SC.make_scene(self.c, explain=(rank == self.first))
        self.lines = card_lines(self.c)
        self.nums = ruler_numbers(self.c)
        key = self.c.get("photo") or PHOTO_KEY[rank]
        bg = SC.background(pk).astype(np.float32)
        por = SC.portrait_layer(key, pk).astype(np.float32)
        scr = SC.scrim().astype(np.float32)

        def over(base, layer):
            a = layer[..., 3:4] / 255.0
            return base * (1 - a) + layer[..., :3] * a
        self.plate0 = over(bg, scr)                              # without the portrait
        self.plate1 = over(over(bg, por), scr)                   # with it
        self.base_rgb = np.array(SC.BG, np.float32)
        self.credit = CREDITS.get(key, {}).get("short", "")

    def plate(self, fig, t, al):
        pa = fade(t, 0.3, 99, 1.0, 0)
        img = self.plate0 * (1 - pa) + self.plate1 * pa
        if al < 0.999:
            img = self.base_rgb * (1 - al) + img * al
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(img.astype(np.uint8), extent=(0, W, H, 0), interpolation="nearest", zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    def draw(self, fig, t):
        c, acc = self.c, self.acc
        al = fade(t, 0, self.dur, 0.6, 0.7)
        self.plate(fig, t, al)
        title(fig, 96, 230, f"{c['rank']}", size=190, color=acc, alpha=al)
        nx = 96 + (150 if c["rank"] < 10 else 240)
        title(fig, nx, 128, c["athlete"], size=66, alpha=al)
        text(fig, nx, 180, f"{c['title']} · {c['season']} · {self.pal['team']}", size=28, color=INK2, alpha=al)
        text(fig, nx, 232, c["achievement"], size=40, color=acc, weight="semibold", alpha=al)
        text(fig, 1900, 1068, self.credit, size=13, color=MUTED, ha="right", alpha=al * fade(t, 0.3, 99, 1.0, 0))
        self.scene.draw(fig, t, al)
        prog = ease(clamp((t - 14.2) / 4.0))
        sa = al * fade(t, 14.2, 99, 0.6, 0)
        if sa > 0:
            xs, bs, ss = self.nums
            u = RULER_UNIT[c["category"]]
            r = c["ruler"]
            if c.get("friendly"):
                friendly = c["friendly"]
            elif c["category"] in ("ath_M_LJ", "ath_W_TJ", "ath_M_PV"):
                friendly = f"Curve about {r['base']:.2f} m · one step about {100 * r['step']:.1f} cm"
            else:
                friendly = f"Curve about {sig(r['base'], 3)} {u} · one step about {sig(r['step'], 2)} {u}"
            text(fig, 1230, 556, "OUTLIER SCORE", size=21, color=INK2, weight="semibold", alpha=sa)
            title(fig, 1230, 662, f"{c['z_e3'] * prog:.2f}", size=116, color=acc, alpha=sa)
            text(fig, 1230, 702, "steps above the curve", size=24, color=INK, alpha=sa)
            text(fig, 1230, 738, friendly, size=21, color=INK2, alpha=sa)
            cm = " (in centimetres)" if c["category"] in CM_CATS else ""
            arith = c.get("arith") or f"({xs} − {bs}) ÷ {ss} = {c['z_e3']:.2f}{cm}"
            text(fig, 1230, 768, arith, size=21, color=MUTED,
                 alpha=sa * fade(t, 18.2, 99, 0.6, 0))
        y = 812
        for i, line in enumerate(self.lines):
            a = al * fade(t, 19.5 + 5.0 * i, 99, 0.8, 0)
            y = para(fig, 1230, y, line, width=44, size=23, color=INK if i == 0 else INK2, alpha=a) + 10
        if c["rank"] == self.first:
            text(fig, 640, 1050, "Historical statistical dominance model: ranks depend on category, comparison window and "
                                 "era adjustment. Not a probability or a measure of ability.", size=16, color=MUTED,
                 ha="center", alpha=al * fade(t, 1.0, 10.5, 0.8, 0.8))


for _r in range(10, 0, -1):
    SECTIONS[f"c{_r}"] = (lambda r: (lambda: Card(r)))(_r)


# ------------------------------------------------------------------ the women's throws
class Throws:
    dur = 58.0

    def __init__(self):
        self.sp = F["womens_throws"]["ath_W_SP"]
        self.dt = F["womens_throws"]["ath_W_DT"]
        self.leads = np.array(self.sp["season_leads"], float)
        self.rec = self.sp["record_line"]

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.7)
        sp, dt = self.sp, self.dt
        text(fig, 96, 92, "WHY THE WOMEN'S THROWS AREN'T ON THIS LIST", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "When the outlier is an era", size=58, alpha=al)
        ax = ax_px(fig, 150, 300, 1100, 930)
        style_axes(ax)
        ax.set_xlim(1964, 2027)
        ax.set_ylim(19.0, 23.2)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        ax.text(0, 1.04, "Women's shot put: the longest throw of each season (metres)", transform=ax.transAxes,
                fontsize=fs(19), color=MUTED, alpha=al)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        band = al * fade(t, 17.0, 99, 1.0, 0)
        if band > 0:
            ax.axvspan(1979.5, 1989.5, color=GOLD, alpha=0.10 * band, lw=0, zorder=1)
            ax.text(1984.5, 23.05, "1980s", ha="center", va="top", fontsize=fs(20), color=INK2, alpha=band)
        ax.axhline(22.0, color=AXIS, lw=px(1.5), ls=(0, (4, 3)), zorder=2, alpha=al)
        ax.text(2026.5, 22.04, "22 m", ha="right", va="bottom", fontsize=fs(18), color=MUTED, alpha=al)
        sweep = 1964 + (2027.5 - 1964) * ease((t - 1.5) / 11.0)
        m = self.leads[:, 0] <= sweep
        if m.sum() > 1:
            ax.plot(self.leads[m, 0], self.leads[m, 1], color=BLUE, lw=px(2.5), alpha=al, zorder=4)
            ax.scatter(self.leads[m, 0], self.leads[m, 1], s=px(8) ** 2, color=BLUE, edgecolors=BG, linewidths=px(1.5),
                       alpha=al, zorder=5)
        rx = [p[0] for p in self.rec if p[0] <= sweep]
        ry = [p[1] for p in self.rec if p[0] <= sweep]
        if rx:
            ax.step(rx + [min(sweep, 2027)], ry + [ry[-1]], where="post", color=GOLD, lw=px(3.5), alpha=al, zorder=6)
        if sweep > 1988:
            ax.annotate("world record, 22.63 m (1987)", (1990, 22.63), xytext=(0, px(10)), textcoords="offset points",
                        fontsize=fs(21), color=INK, alpha=al * fade(t, 9, 99, 0.6, 0), path_effects=outlined(6))
        best20 = max(v for y_, v in sp["season_leads"] if y_ >= 2020)
        caps = [(1.5, "By one measure these are the most extreme numbers in the whole dataset."),
                (8.0, f"No woman has thrown 22 m since {sp['last_year_at_or_beyond_threshold']}; the best this decade is "
                      f"{best20:.2f} m. The best discus since 2000 is {76.80 - dt['best_since_2000'][0]:.2f} m short "
                      f"of the 1988 record."),
                (19.0, f"But in the 1980s, marks like these were routine at the top: "
                       f"{sp['marks_at_or_beyond_threshold_by_decade']['1980']} shot puts of 22 m or more, and "
                       f"{dt['marks_at_or_beyond_threshold_by_decade']['1980']} discus throws of 72 m or more by "
                       f"{dt['athletes_at_or_beyond_threshold_by_decade']['1980']} women."),
                (30.0, f"Against their own era, the record-holders' best seasons score "
                       f"{sp['record_holder_best_season']['z']:.1f} (shot) and {dt['record_holder_best_season']['z']:.1f} "
                       f"(discus). The top ten starts at {CARDS[10]['z_e3']:.2f}."),
                (40.0, "So they were measured like every other event and don't make the list: the outlier is the era, "
                       "not one athlete.")]
        y = 315
        for t0, s in caps:
            a = al * fade(t, t0, 99, 0.8, 0)
            if a > 0:
                y = para(fig, 1190, y, s, width=44, size=24, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                         alpha=a) + 18
        para(fig, 150, 1000, "Random out-of-competition drug testing began in athletics in the late 1980s. This film does "
                             "not single anyone out.", width=86, size=20, color=MUTED, alpha=al * fade(t, 48.0, 99, 0.8, 0))


@section
def throws():
    return Throws()


# ------------------------------------------------------------------ the close
class Close:
    dur = 58.0

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.8)
        R = F["robustness"]
        a1 = fade(t, 0, 21.0, 0.6, 0.7) * al
        if a1 > 0:
            text(fig, 96, 92, "HOW SURE IS THIS ORDER?", size=21, color=MUTED, weight="semibold", alpha=a1)
            title(fig, 96, 160, "Four that never left the top ten", size=58, alpha=a1)
            four = [CARDS[1], CARDS[2], CARDS[3], CARDS[4]]
            for i, c in enumerate(four):
                x = 96 + i * 440
                a = a1 * fade(t, 1.0 + 0.8 * i, 99, 0.6, 0)
                photo(fig, PHOTO_KEY[c["rank"]], x, 200, x + 112, 340, alpha=a, credit=False)
                title(fig, x, 395, c["athlete"], size=40, alpha=a)
                text(fig, x, 438, f"{c['achievement']}, {c['season']}", size=24, color=INK2, alpha=a)
                text(fig, x, 476, f"#{c['rank']} here", size=24, color=GOLD, alpha=a, weight="semibold")
            para(fig, 96, 590, f"In all {R['n_specs']} versions of the method we tested (comparison windows of 7 to 20 "
                               "seasons, straight or curved trends, with or without the league-scoring adjustment), "
                               "these four stayed in the top ten. Their order did not.", width=84, size=32, color=INK,
                 alpha=a1 * fade(t, 4.5, 99, 0.8, 0))
            para(fig, 96, 815, "When the comparison seasons are resampled, the other six places change hands often.",
                 width=90, size=27, color=INK2, alpha=a1 * fade(t, 11.0, 99, 0.8, 0))
        a2 = fade(t, 21.5, 40.5, 0.6, 0.7) * al
        if a2 > 0:
            text(fig, 96, 92, "WHAT CHANGES THE ORDER", size=21, color=MUTED, weight="semibold", alpha=a2)
            title(fig, 96, 160, "Same data, different questions", size=58, alpha=a2)
            ar = R["all_seasons_rank"]
            ne = R["no_env_rank"]
            pts = [f"Count the athlete's own other seasons in the comparison, and Gretzky falls to #{ar['Wayne Gretzky']}: "
                   f"his own great seasons become the bar. DiMaggio rises to #{ar['Joe DiMaggio']}.",
                   f"Drop the league-scoring adjustment, and Dan Marino's 48 touchdown passes (1984) and Katie Ledecky's "
                   f"2016 enter the top ten (#{ne['Dan Marino']} and #{ne['Katie Ledecky']}).",
                   "Places 5 to 10 shuffle under almost any change."]
            y = 330
            for i, s in enumerate(pts):
                a = a2 * fade(t, 22.5 + 5.0 * i, 99, 0.8, 0)
                title(fig, 100, y + 6, "•", size=56, color=GOLD, alpha=a)
                y = para(fig, 150, y, s, width=76, size=33, color=INK, alpha=a) + 46
        a3 = fade(t, 41.0, self.dur, 0.8, 0.8) * al
        if a3 > 0:
            title(fig, W / 2, 400, "Not a ranking of the greatest athletes.", size=60, ha="center", alpha=a3)
            title(fig, W / 2, 500, "A ranking of how far one result stood", size=60, ha="center", alpha=a3 * fade(t, 42.5, 99, 0.8, 0))
            title(fig, W / 2, 580, "from everyone else of its time.", size=60, ha="center", color=GOLD,
                  alpha=a3 * fade(t, 42.5, 99, 0.8, 0))
            text(fig, W / 2, 720, "Which definition would you choose?", size=36, color=INK2, ha="center",
                 alpha=a3 * fade(t, 46.0, 99, 0.8, 0))


@section
def close():
    return Close()


class End:
    dur = 26.0

    def draw(self, fig, t):
        self.credits(fig, t)
        al = fade(t, 0, 12.5, 0.8, 0.8)
        if al <= 0:
            return
        text(fig, 96, 120, "SOURCES AND METHOD", size=21, color=MUTED, weight="semibold", alpha=al)
        lines = [
            ("Data", "Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro Football Archives, nflverse, "
                     "World Athletics, World Aquatics, alltime-athletics.com (Peter Larsson), Retrosheet."),
            ("Model", f"{R_N()} season-leading performances in 74 categories. Each is compared with the best of everyone "
                      "else in the 15 seasons before and after, around a curved trend, with a league-scoring adjustment "
                      "for team sports (estimated from the data). Seasons are labelled by the year they end."),
            ("Caveats", "Scores are statistical distances, not probabilities or measures of ability. Player minutes, "
                        "possessions and ice time are not standardised. A correction for the number of teams was "
                        "tested and left out as unreliable."),
            ("Retrosheet", "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
                           "Interested parties may contact Retrosheet at www.retrosheet.org."),
        ]
        y = 200
        for i, (h, s) in enumerate(lines):
            a = al * fade(t, 0.4 + 0.5 * i, 99, 0.6, 0)
            text(fig, 96, y + 26, h, size=26, color=GOLD, weight="semibold", alpha=a)
            y = para(fig, 330, y, s, width=92, size=26, color=INK2, alpha=a) + 34


def _credits(self, fig, t):
    al = fade(t, 13.0, self.dur, 0.8, 1.2)
    if al <= 0:
        return
    text(fig, 96, 120, "PHOTOGRAPHS", size=21, color=MUTED, weight="semibold", alpha=al)
    order = ["beamon", "chamberlain", "dimaggio", "gretzky", "zelezny", "vance", "curry", "rodman", "ruth", "ichiro"]
    for i, k in enumerate(order):
        cr = CREDITS[k]
        col, row = i % 2, i // 2
        x, y = 96 + col * 880, 200 + row * 118
        a = al * fade(t, 13.3 + 0.12 * i, 99, 0.5, 0)
        photo(fig, k, x, y, x + 72, y + 90, alpha=a, credit=False)
        text(fig, x + 96, y + 30, cr["athlete"], size=24, color=INK, weight="semibold", alpha=a)
        lic = cr["license"] if cr["license"] != "Public domain" else "public domain"
        text(fig, x + 96, y + 62, f"{cr['author']} · {lic}", size=20, color=INK2, alpha=a)
        text(fig, x + 96, y + 88, cr["file"][:70], size=16, color=MUTED, alpha=a)
    para(fig, 96, 920, "All from Wikimedia Commons (commons.wikimedia.org): cropped, backgrounds removed, toned in team colours. Licences: "
                       "creativecommons.org/licenses (CC BY 2.0, CC BY-SA 4.0, CC0) or public domain. Full credits and links "
                       "are in the description.", width=120, size=20, color=MUTED, alpha=al)


End.credits = _credits


def R_N():
    return f"{F['robustness']['n_scored']:,}"


@section
def end():
    return End()


ORDER = ["hook"] + [f"c{r}" for r in range(10, 0, -1)] + ["curve", "fair", "throws", "close", "end"]


def stills():
    os.makedirs(os.path.join(look.OUT, "stills"), exist_ok=True)
    picks = {"hook": [6.0, 14.0, 17.0, 19.5, 27.0], "curve": [24.0, 40.0], "fair": [34.0],
             "throws": [52.0], "close": [12.0, 34.0, 50.0], "end": [10.0, 22.0]}
    for r in range(10, 0, -1):
        picks[f"c{r}"] = [33.0]
    for name in ORDER:
        sec = SECTIONS[name]()
        for t in picks.get(name, [sec.dur / 2]):
            p = os.path.join(look.OUT, "stills", f"{ORDER.index(name):02d}_{name}_{t:05.1f}.png")
            look.still(sec, t, p)
            print(p, flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "stills":
        stills()
    else:
        sc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print(look.render_section(which, SECTIONS[which](), scale=sc))
