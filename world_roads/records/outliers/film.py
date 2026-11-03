"""The 10 Biggest Statistical Outliers in Sports History: the sections of the film.

    python film.py <section> [scale]     # cold method c10 ... c1 throws close end -> out/<section>.mp4
    python film.py stills                # a still of every section at its busiest moment -> out/stills/

Every number on screen comes from facts.json (python facts.py), built from the locked model E3_main_env of the
follow-up analysis. The look (surface, type, gold and blue) is the records film's (../look.py).
"""
import json, os, sys, math

import numpy as np

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


def card_lines(c):
    """three context lines per card, every number from facts.json"""
    r, ex = c["rank"], F["extras"]
    ru = c["runner_up"]
    pr = c["previous_record"]
    nr = c["next_record"]
    if r == 10:
        return [f"262 hits broke George Sisler's record of {float(pr['value']):.0f}, set in {pr['year']}.",
                f"Next best in 2004: {ru['name']}, {float(ru['value']):.0f}.",
                "The score also accounts for how often hitters got hits league-wide."]
    if r == 9:
        return [f"His own record from the year before was {float(pr['value']):.0f}.",
                f"Next best in 1920: {ru['name']}, {float(ru['value']):.0f}. The average team hit {ex['mlb_1920_avg_team_hr']:.0f}.",
                f"Without the league adjustment his score is {c['z_main_no_env']:.1f}: home runs were that rare in 1920."]
    if r == 8:
        return [f"Next best that season: {ru['name']}, {float(ru['value']):.1f}.",
                "His 18.7 the season before is higher in raw terms; against league-wide rebounding, 1992–93 stands further out.",
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


# ------------------------------------------------------------------ the method, with Ledecky 2016
class Method:
    dur = 78.0

    def __init__(self):
        m = F["method_example"]
        self.m = m
        self.pts = [(p["season"], p["time"]) for p in m["window"]]
        self.curve = np.array(m["trend_time"])
        self.z = float(m["z"])
        # the band and the ruler in time units: speed = 800 / time; unit = scale in speed
        self.sp_base, self.sp_scale = m["base_speed"], m["scale_speed"]

    def chart(self, fig, t, al):
        m = self.m
        ax = ax_px(fig, 150, 300, 1100, 930)
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
        # dots sweep in
        sweep = 2001 + (2026.9 - 2001) * ease((t - 2.5) / 9.0)
        for s, v in self.pts:
            if s <= sweep:
                pop = ease_out((sweep - s) / 1.5)
                ring(ax, s, v, BLUE, size=12 * (0.5 + 0.5 * pop), alpha=al * pop, z=5)
        # the trend
        tc = clamp((t - 16.0) / 6.0)
        if tc > 0:
            k = max(2, int(len(self.curve) * tc))
            ax.plot(self.curve[:k, 0], self.curve[:k, 1], color=TREND, lw=px(3), alpha=al, zorder=4)
        # the band: +/- one unit of scatter (in speed), shown in time
        ba = fade(t, 26.0, 99, 1.2, 0) * al
        if ba > 0:
            x = self.curve[:, 0]
            sp = 800 / self.curve[:, 1]
            ax.fill_between(x, 800 / (sp - self.sp_scale), 800 / (sp + self.sp_scale), color=TREND, alpha=0.16 * ba,
                            lw=0, zorder=2)
        # Ledecky and the ruler
        la = fade(t, 36.0, 99, 0.8, 0) * al
        if la > 0:
            ring(ax, 2016, m["x_time"], GOLD, size=17, alpha=la, z=8)
            ax.annotate("Katie Ledecky, 2016: 8:04.79", (2016, m["x_time"]), xytext=(px(16), px(10)),
                        textcoords="offset points", fontsize=fs(23), color=INK, alpha=la, zorder=9,
                        path_effects=outlined(6), weight="semibold")
            prog = clamp((t - 38.0) / 5.0)
            top = self.sp_base + (self.z * self.sp_scale) * prog
            ax.plot([2016, 2016], [800 / self.sp_base, 800 / top], color=GOLD, lw=px(2.5), alpha=la, ls=(0, (2, 2)), zorder=7)
            for k in range(1, int(self.z * prog) + 1):
                yk = 800 / (self.sp_base + k * self.sp_scale)
                ax.plot([2015.6, 2016.4], [yk, yk], color=GOLD, lw=px(2), alpha=la, zorder=7)
        return ax

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE LIST WAS BUILT", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "Each season, measured against its own era", size=58, alpha=al)
        ca = fade(t, 0, 50.5, 0, 0.8)
        if ca > 0:
            self.chart(fig, t, al * ca)
            x0 = 1190
            caps = [(1.5, "Every season's leader in 74 categories, across six sports, is compared with the best of "
                          "everyone else in the 15 seasons before and after."),
                    (8.0, "Example: the women's 800 m freestyle. Each blue dot is the fastest time by anyone except "
                          "Katie Ledecky that season."),
                    (16.0, "A curve follows how the event changed over those seasons."),
                    (26.0, "The usual scatter of the dots around the curve becomes the unit of measurement."),
                    (36.0, f"Her 2016 swim, 8:04.79, sits {self.z:.1f} of those units above her era."),
                    (44.0, f"That ranks her {F['method_example']['rank_e3']}th. These are the ten that stood out further.")]
            y = 330
            for t0, s in caps:
                a = al * ca * fade(t, t0, 99, 0.8, 0)
                if a > 0:
                    y = para(fig, x0, y, s, width=40, size=27, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                             alpha=a) + 22
            if t > 38:
                cnt = self.z * ease(clamp((t - 38.0) / 5.0))
                title(fig, 1190, 1035, f"{cnt:.2f}", size=64, color=GOLD, alpha=al * ca * fade(t, 38, 99, 0.5, 0))
                text(fig, 1370, 1030, "units above the curve", size=24, color=INK2, alpha=al * ca * fade(t, 38, 99, 0.5, 0))
        ra = fade(t, 51.0, self.dur, 0.8, 0.6)
        if ra > 0:
            rules = [(51.5, "1", "Team sports are also measured against the league's scoring level: 50 points a game means "
                                 "more when teams average 100 than when they average 120."),
                     (57.5, "2", "An athlete's own other seasons are left out of the comparison, so this measures "
                                 "dominance over rivals."),
                     (63.5, "3", "Each athlete appears once. Seasons are labelled by the year they end, and some entries "
                                 "are a single mark or a streak rather than a season total.")]
            y = 330
            for t0, n, s in rules:
                a = ra * fade(t, t0, 99, 0.8, 0)
                title(fig, 150, y + 40, n, size=64, color=GOLD, alpha=a)
                y = para(fig, 250, y, s, width=62, size=33, color=INK, alpha=a) + 48
            para(fig, 250, max(y, 860), "The score is a statistical distance, not a probability, and the order depends on "
                                        "these choices. More on that at the end.", width=80, size=24, color=MUTED,
                 alpha=ra * fade(t, 69.5, 99, 0.8, 0))


@section
def method():
    return Method()


# ------------------------------------------------------------------ the countdown cards
class Card:
    dur = 36.0

    def __init__(self, rank):
        self.c = CARDS[rank]
        c = self.c
        self.share = c.get("share")
        if self.share:                                    # team sports: share of an average team's total
            self.pts = [tuple(p) for p in self.share["window"]]
            self.curve = np.array(self.share["trend"])
            self.x = self.share["x"]
        else:
            self.pts = [(p["season"], p["value"]) for p in c["window"] if p["holder"] != "list cut-off"]
            self.curve = np.array(c["trend_raw_units"])
            self.x = c["x_raw"]
        self.lines = card_lines(c)

    def chart(self, fig, t, al):
        c = self.c
        ax = ax_px(fig, 150, 330, 1150, 930)
        style_axes(ax)
        ys = [v for _, v in self.pts] + [self.x] + list(self.curve[:, 1])
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08
        x0 = min(min(s for s, _ in self.pts), c["year"]) - 0.8
        x1 = max(max(s for s, _ in self.pts), c["year"]) + 0.8
        ax.set_xlim(x0, x1)
        ax.set_ylim(lo - pad, hi + pad * 1.6)
        if self.share:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{100 * v:.0f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_value(c["category"], v).replace(" m", "")))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        unit = {"ath_M_JT": "metres", "ath_M_LJ": "metres", "mlb_streak": "consecutive games with a hit"}.get(c["category"],
                                                                                                         unit_word(c["category"]))
        if self.share:
            lab = f"{unit}, as a {self.share['label']} (league average)"
        else:
            note = f" (other seasons scaled to {c['games_per_team']:.0f} games)" if c["games_per_team"] else ""
            lab = f"{unit}{note}"
        ax.text(0, 1.04, lab, transform=ax.transAxes, fontsize=fs(19), color=MUTED, alpha=al)
        sweep = x0 + (x1 + 1 - x0) * ease((t - 1.2) / 9.5)
        for s, v in self.pts:
            if s <= sweep and s != c["year"]:
                pop = ease_out((sweep - s) / 1.4)
                ring(ax, s, v, BLUE, size=12 * (0.5 + 0.5 * pop), alpha=al * pop, z=5)
        # the runner-up in the season itself, smaller
        tc = clamp((t - 11.0) / 3.0)
        if tc > 0:
            k = max(2, int(len(self.curve) * tc))
            ax.plot(self.curve[:k, 0], self.curve[:k, 1], color=TREND, lw=px(3), alpha=al, zorder=4)
        ga = fade(t, 14.2, 99, 0.6, 0) * al
        if ga > 0:
            base = float(np.interp(c["year"], self.curve[:, 0], self.curve[:, 1]))
            prog = ease_out(clamp((t - 14.2) / 1.6))
            yy = base + (self.x - base) * prog
            ax.plot([c["year"], c["year"]], [base, yy], color=GOLD, lw=px(2.5), ls=(0, (2, 2)), alpha=ga, zorder=6)
            ring(ax, c["year"], yy, GOLD, size=19, alpha=ga, z=8)
            if prog > 0.95:
                lab = (f"{c['athlete']}, {c['season']}: {c['achievement'].split(' ')[0]} = {100 * self.x:.0f}%" if self.share
                       else f"{c['athlete']}, {c['season']}: {fmt_value(c['category'], self.x)}")
                left = c["year"] > (x0 + x1) / 2
                ax.annotate(lab, (c["year"], self.x), xytext=(px(-18 if left else 18), px(4)), textcoords="offset points",
                            ha="right" if left else "left", va="center", fontsize=fs(23), color=INK, weight="semibold",
                            alpha=ga, zorder=9, path_effects=outlined(6))
        return ax

    def env_inset(self, fig, t, al):
        c = self.c
        if not c.get("league_env"):
            return
        a = al * fade(t, 22.0, 99, 0.8, 0)
        if a <= 0:
            return
        ax = ax_px(fig, 1250, 905, 1830, 1000)
        ax.set_facecolor("none")
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        e = np.array(c["league_env"], float)
        ax.plot(e[:, 0], e[:, 1], color=INK2, lw=px(2), alpha=a)
        ev = c["league_env_at_season"]
        ring(ax, c["year"], ev, GOLD, size=10, alpha=a, z=5)
        ax.set_xlim(c["year"] - 15.6, c["year"] + 15.6)
        text(fig, 1250, 895, f"{ENV_LABEL[c['category']]}, {c['year'] - 15}–{min(c['year'] + 15, 2026)}", size=19,
             color=MUTED, alpha=a)

    def draw(self, fig, t):
        c = self.c
        al = fade(t, 0, self.dur, 0.6, 0.7)
        title(fig, 96, 230, f"{c['rank']}", size=190, color=GOLD, alpha=al)
        nx = 96 + (150 if c["rank"] < 10 else 240)
        title(fig, nx, 128, c["athlete"], size=66, alpha=al)
        text(fig, nx, 180, f"{c['title']} · {SPORT_WORD[c['sport']]} · {c['season']}", size=28, color=INK2, alpha=al)
        text(fig, nx, 232, c["achievement"], size=40, color=INK, weight="semibold", alpha=al)
        # legend row, under the chart (glyphs keep the marks round)
        la = al * fade(t, 1.0, 99, 0.6, 0)
        lx = 150
        for lab, col, glyph in (("everyone else's best, each season", BLUE, "●"), ("their trend", TREND, "—"),
                                (f"{c['athlete'].split()[-1]}, {c['season']}", GOLD, "●")):
            text(fig, lx, 1022, glyph, size=24, color=col, alpha=la)
            text(fig, lx + 34, 1022, lab, size=21, color=INK2, alpha=la)
            lx += 34 + len(lab) * 11.0 + 48
        self.chart(fig, t, al)
        # the score
        sa = al * fade(t, 16.0, 99, 0.6, 0)
        if sa > 0:
            text(fig, 1250, 360, "OUTLIER SCORE", size=21, color=MUTED, weight="semibold", alpha=sa)
            v = c["z_e3"] * ease(clamp((t - 16.0) / 3.0))
            title(fig, 1250, 470, f"{v:.2f}", size=120, color=GOLD, alpha=sa)
            text(fig, 1250, 512, "typical spreads above everyone else's trend", size=22, color=INK2, alpha=sa)
            if c["env_adjusted"]:
                text(fig, 1250, 545, "the score uses an estimated version of the league adjustment", size=20, color=MUTED, alpha=sa)
        y = 610
        for i, line in enumerate(self.lines):
            a = al * fade(t, 19.5 + 5.0 * i, 99, 0.8, 0)
            y = para(fig, 1250, y, line, width=40, size=26, color=INK if i == 0 else INK2, alpha=a) + 16
        self.env_inset(fig, t, al)
        if c["rank"] == 10:
            text(fig, W / 2, 1050, "Historical statistical dominance model: ranks depend on category, comparison window and "
                                   "era adjustment. Not a probability or a measure of ability.", size=19, color=MUTED,
                 ha="center", alpha=al * fade(t, 1.0, 12.0, 0.8, 0.8))


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
                title(fig, x, 360, c["athlete"], size=40, alpha=a)
                text(fig, x, 405, f"{c['achievement']}, {c['season']}", size=24, color=INK2, alpha=a)
                text(fig, x, 445, f"#{c['rank']} here", size=24, color=GOLD, alpha=a, weight="semibold")
            para(fig, 96, 560, f"In all {R['n_specs']} versions of the method we tested (comparison windows of 7 to 20 "
                               "seasons, straight or curved trends, with or without the league-scoring adjustment), "
                               "these four stayed in the top ten. Their order did not.", width=84, size=32, color=INK,
                 alpha=a1 * fade(t, 4.5, 99, 0.8, 0))
            para(fig, 96, 790, "When the comparison seasons are resampled, the other six places change hands often.",
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
    dur = 16.0

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.8, 1.2)
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


def R_N():
    return f"{F['robustness']['n_scored']:,}"


@section
def end():
    return End()


ORDER = ["cold", "method"] + [f"c{r}" for r in range(10, 0, -1)] + ["throws", "close", "end"]


def stills():
    os.makedirs(os.path.join(look.OUT, "stills"), exist_ok=True)
    picks = {"cold": [5.5, 18.0], "method": [30.0, 47.0, 72.0], "throws": [52.0], "close": [12.0, 34.0, 50.0], "end": [10.0]}
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
        scale = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print(look.render_section(which, SECTIONS[which](), scale=scale))
