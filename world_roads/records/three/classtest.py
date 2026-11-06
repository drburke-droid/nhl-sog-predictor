"""Film A's "grading on a curve", built the way every card is: a class test with one student's 95, the five best scores
by everyone else in that class and in each of the 15 classes before and after, a year with an easier test scaled to
match, the curve through all those top fives, one standard deviation (3 points), and the 95 measured against it:
(95 - 80) / 3 = 5. Then the same 95 among top fives around 92: one standard deviation. Illustrative numbers, not data.

Timed to the narration by script_a.py (timed_a.json): spec["at"] holds when each step starts, spec["caps"] the captions.
"""
import numpy as np

import film as FM
from look import (W, H, INK, INK2, MUTED, GOLD, BLUE, px, fs, clamp, ease, ease_out, fade, ax_px, text, title,  # noqa
                  ring, outlined)

TREND = FM.TREND
SD, CURVE0, LIFT, STAR = 3.0, 80.0, 12.0, 95.0          # one standard deviation, the curve this year, the "what if", the 95
EASY_YEAR, EASY_BY = 7, 9.0                             # the class with the easier test, and how much easier


def curve(x):
    return CURVE0 + 0.05 * x                                # the top fives drift up a little with better teaching


class ClassTest:
    def __init__(self, spec):
        self.spec = spec
        self.at = spec["at"]
        self.dur = spec["dur"]
        rng = np.random.default_rng(80)
        self.years = list(range(-15, 16))
        res = np.clip(rng.normal(size=(31, 5)), -2.2, 2.2)
        res = (res - res.mean()) / res.std()                # exactly one standard deviation of 3 points around the curve
        self.top5 = {y: sorted(curve(y) + SD * res[i]) for i, y in enumerate(self.years)}
        cls = np.clip(rng.normal(62.0, 8.0, size=22), 38.0, 76.0)
        self.rest = sorted(cls)                             # the rest of this year's class, below the top five
        self.jit = rng.uniform(-0.38, 0.38, size=len(self.rest))

    def X(self, y):
        return y

    def chart(self, fig, t, al):
        at = self.at
        ax = ax_px(fig, 150, 300, 1100, 900)
        ax.set_xlim(-16.6, 16.6)
        ax.set_ylim(36, 102)
        ax.set_axis_off()
        ax.plot([-16.4, 16.4], [37.5, 37.5], color="#5a5f66", lw=px(1.5), alpha=al, zorder=3)
        for v in range(40, 101, 10):
            ax.plot([-16.4, 16.4], [v, v], color="#ffffff", alpha=0.06 * al, lw=px(1), zorder=1)
            ax.text(-16.9, v, f"{v}", ha="right", va="center", fontsize=fs(17), color=MUTED, alpha=al)
        for y, lab in ((-15, "15 years\nbefore"), (-10, "−10"), (-5, "−5"), (0, "this\nyear"), (5, "+5"), (10, "+10"),
                       (15, "15 years\nafter")):
            ax.text(y, 35.2, lab, ha="center", va="top", fontsize=fs(17), color=MUTED if y else INK2, alpha=al,
                    linespacing=1.0)
        ax.text(-16.4, 104.5, "Test scores: each dot is one of the five best by everyone else in that year's class", ha="left",
                va="bottom", fontsize=fs(19), color=INK2, alpha=al)
        lift = LIFT * ease(clamp((t - at["lift"] - 1.0) / 3.5))              # the "what if": every other class higher
        # this year's class: the whole class, then only its five best (the 95 left out)
        ca = al * fade(t, 0.8, 99, 0.6, 0)
        rest_a = ca * (1 - 0.85 * ease(clamp((t - at["top5"]) / 1.5)))
        for k, (v, j) in enumerate(zip(self.rest, self.jit)):
            a = rest_a * clamp((t - 0.8) * 8 - k * 0.35)
            if a > 0:
                ring(ax, j, v, "#6f757d", size=8.5, alpha=a, z=4, ew=1.0)
        for k, v in enumerate(self.top5[0]):
            a = ca * clamp((t - 0.8) * 8 - (len(self.rest) + k) * 0.35)
            if a > 0:
                hi = ease(clamp((t - at["top5"]) / 1.0))
                dx = [0, -0.3, 0.3, -0.6, 0.6][k]
                ring(ax, dx * hi, v + lift, BLUE if hi > 0.5 else "#6f757d", size=8.5 + 2.5 * hi, alpha=a, z=5, ew=1.2)
        sa = al * fade(t, at["s95"], 99, 0.6, 0)
        if sa > 0:
            ring(ax, 0, STAR, GOLD, size=16, alpha=sa, z=9)
            ax.annotate("95", (0, STAR), xytext=(px(14), px(4)), textcoords="offset points", fontsize=fs(24), color=INK,
                        weight="semibold", alpha=sa, zorder=10, path_effects=outlined(6))
        la = al * fade(t, at["top5"] + 0.6, at["others"] + 1.0, 0.5, 0.6)
        if la > 0:
            ax.text(1.2, np.mean(self.top5[0]), "the five best\nby everyone else", ha="left", va="center", fontsize=fs(19),
                    color=INK, alpha=la, weight="semibold", path_effects=outlined(6), zorder=10)
        # the other 30 classes, sweeping out from this year
        sweep = 16.0 * ease(clamp((t - at["others"]) / 5.0))
        for y in self.years:
            if y == 0 or abs(y) > sweep:
                continue
            pop = ease_out(clamp((sweep - abs(y)) / 1.5))
            vs = self.top5[y]
            if y == EASY_YEAR:                                              # the easier test: shown high, then scaled
                if t < at["easy"]:
                    continue
                drop = ease(clamp((t - at["easy"] - 2.0) / 2.0))
                vs = [v + EASY_BY * (1 - drop) for v in vs]
                pop = ease_out(clamp((t - at["easy"]) / 0.6))
            for k, v in enumerate(vs):
                dx = [0, -0.3, 0.3, -0.6, 0.6][k]
                ring(ax, y + dx, v + lift, "#b8bec6", size=10.0 * (0.5 + 0.5 * pop), alpha=0.9 * al * pop, z=4, ew=1.2)
        ea = al * fade(t, at["easy"], at["band"], 0.5, 0.6)
        if ea > 0:
            drop = ease(clamp((t - at["easy"] - 2.0) / 2.0))
            yv = np.mean(self.top5[EASY_YEAR]) + EASY_BY * (1 - drop)
            ax.text(EASY_YEAR + 1.0, yv + 4.5, "an easier test" if drop < 0.5 else "scaled to match", ha="left",
                    va="center", fontsize=fs(19), color=INK, alpha=ea, weight="semibold", path_effects=outlined(6),
                    zorder=10)
        # the curve, then one standard deviation either side of it
        xs = np.linspace(-15.5, 15.5, 120)
        tc = clamp((t - at["curve"]) / 3.0)
        if tc > 0:
            k = max(2, int(len(xs) * tc))
            ax.plot(xs[:k], curve(xs[:k]) + lift, color=TREND, lw=px(3.2), alpha=al * 0.95, zorder=6)
            if tc >= 1:
                ax.text(15.9, curve(15.5) + lift, "the curve", ha="left", va="center", fontsize=fs(18), color=INK2,
                        alpha=al, path_effects=outlined(5), zorder=10)
        ba = al * fade(t, at["band"], 99, 1.0, 0)
        if ba > 0:
            ax.fill_between(xs, curve(xs) + lift - SD, curve(xs) + lift + SD, color=TREND, alpha=0.16 * ba, lw=0, zorder=2)
            if t < at["star"]:
                xm = -15.9                                  # at the left edge, clear of the dots
                ax.annotate("", (xm, curve(xm) + SD), (xm, curve(xm) - SD), arrowprops=dict(arrowstyle="<->", color=INK,
                                                                                              lw=px(2)), alpha=ba, zorder=10)
                ax.text(-15.4, curve(xm) + SD + 6.5, "the band: one standard deviation\neither side, about 3 points",
                        ha="left", va="center", fontsize=fs(19), color=INK, alpha=ba, weight="semibold",
                        path_effects=outlined(6), zorder=10)
        # the ruler: from the curve up to the 95, one tick per standard deviation
        if t >= at["star"]:
            base = curve(0) + lift
            prog = ease(clamp((t - at["star"] - 0.8) / 3.5)) if t < at["lift"] else 1.0
            top = base + (STAR - base) * prog
            ax.plot([0, 0], [base, top], color=GOLD, lw=px(2.5), ls=(0, (2, 2)), alpha=al, zorder=7)
            for kk in range(1, int((top - base) / SD + 1e-6) + 1):
                yk = base + kk * SD
                ax.plot([-0.75, 0.75], [yk, yk], color=GOLD, lw=px(2.4), alpha=al, zorder=8, path_effects=outlined(4))
                ax.text(-1.0, yk, f"{kk}", ha="right", va="center", fontsize=fs(17), color=GOLD, weight="semibold",
                        alpha=al, zorder=8, path_effects=outlined(5))
            ax.plot([-0.75, 0.75], [base, base], color=TREND, lw=px(2.4), alpha=al, zorder=8, path_effects=outlined(4))
        return ax

    def draw(self, fig, t):
        at = self.at
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW THE SCORE WORKS", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "Grading on a curve", size=58, alpha=al)
        self.chart(fig, t, al)
        caps = self.spec["caps"]
        starts = [0.0] + list(self.spec.get("pages", [at["star"]]))   # pages of captions (film A: the class, the score)
        ends = starts[1:] + [99.0]
        cur = max((c[0] for c in caps if c[0] <= t), default=None)
        for p0, p1 in zip(starts, ends):
            y = 330
            for t0, s in caps:
                if not p0 <= t0 < p1:
                    continue
                a = al * fade(t, t0, p1, 0.8, 0.6)
                if a > 0:
                    y = FM.para(fig, 1190, y, s, width=40, size=26, color=INK if t0 == cur else INK2, alpha=a) + 20
        # the counter
        if t > at["star"] + 0.8:
            lifting = t >= at["lift"] + 1.0
            v = (STAR - curve(0)) / SD * ease(clamp((t - at["star"] - 0.8) / 3.5))
            if lifting:
                v = (STAR - curve(0) - LIFT) / SD + (1 - ease(clamp((t - at["lift"] - 1.0) / 3.5))) * \
                    (LIFT / SD)
            ca = al * fade(t, at["star"] + 0.8, 99, 0.5, 0)
            title(fig, 150, 1040, f"{v:.0f}", size=72, color=GOLD, alpha=ca)
            word = "standard deviation" if f"{v:.0f}" == "1" else "standard deviations"
            text(fig, 225, 1035, f"{word} above {self.spec.get('curve_word', 'the curve')}", size=24, color=INK2, alpha=ca)


class ClassTestBest(ClassTest):
    """film C's class test: only the single best score by everyone else counts, in this class and in each of the 15
    classes before and after (each class drawn whole, then only its best kept); the curve through those bests is the
    elite curve. spec["at"] adds avg (the class average, crossed out) and bests (each class's best lit)."""

    N = 24                                                  # students in a class

    def __init__(self, spec):
        super().__init__(spec)
        rng = np.random.default_rng(1)
        res = np.clip(rng.normal(size=31), -2.2, 2.2)
        res = (res - res.mean()) / res.std()
        res[15] = 0.15                                      # this year's best by everyone else: about 80
        self.best = {y: float(curve(y) + SD * res[i]) for i, y in enumerate(self.years)}
        self.cls = {}
        for y in self.years:
            v = rng.normal(61.0, 7.5, size=self.N - 1)
            v = np.minimum(v, self.best[y] - rng.uniform(1.5, 5.0, size=self.N - 1))   # everyone else below the best
            self.cls[y] = (np.clip(v, 36.0, 99.0), rng.uniform(-0.36, 0.36, size=self.N - 1))
        self.avg0 = float(np.mean(list(self.cls[0][0]) + [self.best[0]]))

    def chart(self, fig, t, al):
        at = self.at
        ax = ax_px(fig, 150, 300, 1100, 900)
        ax.set_xlim(-16.6, 16.6)
        ax.set_ylim(36, 102)
        ax.set_axis_off()
        ax.plot([-16.4, 16.4], [37.5, 37.5], color="#5a5f66", lw=px(1.5), alpha=al, zorder=3)
        for v in range(40, 101, 10):
            ax.plot([-16.4, 16.4], [v, v], color="#ffffff", alpha=0.06 * al, lw=px(1), zorder=1)
            ax.text(-16.9, v, f"{v}", ha="right", va="center", fontsize=fs(17), color=MUTED, alpha=al)
        u = self.spec.get("unit", "year")                   # film C's narration says semesters
        for y, lab in ((-15, f"15 {u}s\nbefore"), (-10, "−10"), (-5, "−5"), (0, f"this\n{u}"), (5, "+5"), (10, "+10"),
                       (15, f"15 {u}s\nafter")):
            ax.text(y, 35.2, lab, ha="center", va="top", fontsize=fs(17), color=MUTED if y else INK2, alpha=al,
                    linespacing=1.0)
        head = f"Test scores: each column is one {u}'s class" if t < at["bests"] + 1.0 else \
            f"Test scores: each blue dot is the best score by everyone else in that {u}'s class"
        ax.text(-16.4, 104.5, head, ha="left", va="bottom", fontsize=fs(19), color=INK2, alpha=al)
        lift = LIFT * ease(clamp((t - at["lift"] - 1.0) / 3.5))
        dim_rest = 1.0 - 0.8 * ease(clamp((t - at["bests"]) / 1.5)) - 0.2 * ease(clamp((t - at["star"]) / 1.5))
        # this year's class, whole: the 95, everyone else, and their best
        ca = al * fade(t, 0.8, 99, 0.6, 0)
        v0, j0 = self.cls[0]
        for k, (v, j) in enumerate(zip(v0, j0)):
            a = ca * clamp((t - 0.8) * 8 - k * 0.35) * (1.0 - 0.75 * ease(clamp((t - at["top5"] - 1.0) / 1.5))) * \
                (dim_rest if t > at["bests"] else 1.0)
            if a > 0:
                ring(ax, j, v, "#6f757d", size=8.0, alpha=a, z=4, ew=1.0)
        hi = ease(clamp((t - at["top5"]) / 1.0))
        b0 = self.best[0] + lift
        ring(ax, 0, b0, BLUE if hi > 0.5 else "#6f757d", size=8.0 + 3.5 * hi, alpha=ca, z=6, ew=1.2)
        la = al * fade(t, at["top5"] + 0.4, at["others"] + 0.5, 0.5, 0.6)
        if la > 0:
            ax.annotate(f"the best by everyone else: {self.best[0]:.0f}", (0, b0), xytext=(px(26), px(-6)),
                        textcoords="offset points", ha="left", va="center", fontsize=fs(19), color=INK, weight="semibold",
                        alpha=la, path_effects=outlined(6), zorder=10)
        aa = al * fade(t, at["avg"], at["others"] + 0.3, 0.5, 0.6)          # the class average: not what counts
        if aa > 0:
            ax.plot([-1.4, 1.4], [self.avg0, self.avg0], color=INK2, lw=px(2), ls=(0, (3, 2)), alpha=aa, zorder=7)
            ax.text(1.7, self.avg0, f"class average ({self.avg0:.0f}): not used", ha="left", va="center", fontsize=fs(19),
                    color=INK2, alpha=aa, path_effects=outlined(6), zorder=10)
        sa = al * fade(t, at["s95"], 99, 0.6, 0)
        if sa > 0:
            ring(ax, 0, STAR, GOLD, size=16, alpha=sa, z=9)
            ax.annotate("95", (0, STAR), xytext=(px(14), px(4)), textcoords="offset points", fontsize=fs(24), color=INK,
                        weight="semibold", alpha=sa, zorder=10, path_effects=outlined(6))
        # the 30 other classes, whole, sweeping out from this year; then only each class's best is kept
        sweep = 16.0 * ease(clamp((t - at["others"]) / 5.0))
        lit = ease(clamp((t - at["bests"]) / 1.5))
        for y in self.years:
            if y == 0 or abs(y) > sweep:
                continue
            pop = ease_out(clamp((sweep - abs(y)) / 1.5))
            up = 0.0
            if y == EASY_YEAR:                                              # the easier test: shown high, then scaled
                if t < at["easy"]:
                    continue
                up = EASY_BY * (1 - ease(clamp((t - at["easy"] - 2.0) / 2.0)))
                pop = ease_out(clamp((t - at["easy"]) / 0.6))
            vs, js = self.cls[y]
            ra = al * pop * 0.55 * dim_rest
            if ra > 0.01:
                for v, j in zip(vs, js):
                    ring(ax, y + j, v + up, "#6f757d", size=6.5, alpha=ra, z=3, ew=0.8)
            ring(ax, y, self.best[y] + up + lift, BLUE if lit > 0.5 else "#8a9098", size=7.0 + 3.5 * lit, alpha=al * pop,
                 z=5, ew=1.2)
        ea = al * fade(t, at["easy"], at["band"], 0.5, 0.6)
        if ea > 0:
            drop = ease(clamp((t - at["easy"] - 2.0) / 2.0))
            yv = self.best[EASY_YEAR] + EASY_BY * (1 - drop)
            ax.text(EASY_YEAR + 0.8, yv + 4.5, "an easier test" if drop < 0.5 else "scaled to match", ha="left",
                    va="center", fontsize=fs(19), color=INK, alpha=ea, weight="semibold", path_effects=outlined(6),
                    zorder=10)
        # the elite curve, then one standard deviation either side of it
        xs = np.linspace(-15.5, 15.5, 120)
        tc = clamp((t - at["curve"]) / 3.0)
        if tc > 0:
            k = max(2, int(len(xs) * tc))
            ax.plot(xs[:k], curve(xs[:k]) + lift, color=TREND, lw=px(3.2), alpha=al * 0.95, zorder=6)
            if tc >= 1:
                ax.annotate("the elite curve", (15.5, curve(15.5) + lift), xytext=(0, px(12)), textcoords="offset points",
                            ha="right", va="bottom", fontsize=fs(18), color=INK2, alpha=al, path_effects=outlined(5),
                            zorder=10)
        ba = al * fade(t, at["band"], 99, 1.0, 0)
        if ba > 0:
            ax.fill_between(xs, curve(xs) + lift - SD, curve(xs) + lift + SD, color=TREND, alpha=0.16 * ba, lw=0, zorder=2)
            if t < at["star"]:
                xm = -15.9
                ax.annotate("", (xm, curve(xm) + SD), (xm, curve(xm) - SD), arrowprops=dict(arrowstyle="<->", color=INK,
                                                                                              lw=px(2)), alpha=ba, zorder=10)
                ax.text(-15.4, curve(xm) + SD + 6.5, "the band: one standard deviation\neither side, about 3 points",
                        ha="left", va="center", fontsize=fs(19), color=INK, alpha=ba, weight="semibold",
                        path_effects=outlined(6), zorder=10)
        if t >= at["star"]:                                                 # the ruler: one tick per standard deviation
            base = curve(0) + lift
            prog = ease(clamp((t - at["star"] - 0.8) / 3.5)) if t < at["lift"] else 1.0
            top = base + (STAR - base) * prog
            ax.plot([0, 0], [base, top], color=GOLD, lw=px(2.5), ls=(0, (2, 2)), alpha=al, zorder=7)
            for kk in range(1, int((top - base) / SD + 1e-6) + 1):
                yk = base + kk * SD
                ax.plot([-0.75, 0.75], [yk, yk], color=GOLD, lw=px(2.4), alpha=al, zorder=8, path_effects=outlined(4))
                ax.text(-1.0, yk, f"{kk}", ha="right", va="center", fontsize=fs(17), color=GOLD, weight="semibold",
                        alpha=al, zorder=8, path_effects=outlined(5))
            ax.plot([-0.75, 0.75], [base, base], color=TREND, lw=px(2.4), alpha=al, zorder=8, path_effects=outlined(4))
        return ax
