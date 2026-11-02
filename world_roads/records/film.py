"""The records film, section by section.

    python film.py still <section> <t> [scale]     # one frame to out/still_<section>_<t>.png
    python film.py <section> [scale]               # render the section to out/<section>.mp4

Sections are listed in SECTIONS; each has .dur (s) and .draw(fig, t).
"""
import os, sys, re, math, datetime as dt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory, offset_copy
from look import *                                                       # noqa: F403
import analysis as A

REC, GAMES = A.load_records(), A.load_games()
NOW = dt.date(2026, 11, 1)


def yearf(d):
    return d.year + (d.timetuple().tm_yday - 0.5) / 365.25


def clean_name(s):
    s = re.sub(r"\s*\(\s*[A-Z]{3}\s*\)", "", s)
    s = re.sub(r"\s*\(n[ée]e [^)]*\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def tick_fmt(ev):
    if ev in A.FIELD:
        return FuncFormatter(lambda v, p: f"{v:.2f} m" if v < 10 else f"{v:.1f} m" if v < 30 else f"{v:.0f} m")
    def f(v, p):
        if v < 60:
            return f"{v:.1f}" if v >= 20 else f"{v:.2f}".rstrip("0").rstrip(".") if False else f"{v:.1f}"
        m, s = divmod(v, 60)
        return f"{int(m)}:{int(round(s)):02d}" if v >= 600 else f"{int(m)}:{s:04.1f}"
    return FuncFormatter(f)


class EventData:
    """everything an event card needs, precomputed"""

    def __init__(self, sex, ev):
        self.sex, self.ev = sex, ev
        self.field = ev in A.FIELD
        line, rows, base = A.event_stats(sex, ev, REC, GAMES)
        self.line = [r for r in line if not r["equalled"]]
        self.rows, self.base = rows, base
        self.old = [r for r in REC.get((sex, "JT_old"), []) if not r["equalled"]] if ev == "JT" else []
        first_games = min([r["year"] for r in rows] or [2024])
        first_rec = yearf(self.line[0]["date"]) if self.line else first_games
        self.x0 = max(min(first_games, first_rec) - 3, 1890 if sex == "M" else 1918)
        if ev == "JT":
            self.x0 = max(min(first_games, yearf(self.old[0]["date"])) - 3, 1890 if sex == "M" else 1918)
        self.x1 = 2027.5
        # step arrays for the record (x in fractional years)
        self.rx = np.array([yearf(r["date"]) for r in self.line])
        self.ry = np.array([r["mark"] for r in self.line])
        self.rkind = [r["kind"] for r in self.line]
        n_light = sum(1 for k in self.rkind if k in ("hand", "pre"))
        self.auto_from = yearf(self.line[n_light]["date"]) if 0 < n_light < len(self.line) else None
        # y range from the record, winners and medians in view (outlier finishers may fall outside)
        vals = [r["mark"] for r in self.line if yearf(r["date"]) >= self.x0]
        vals += [x["median"] for x in rows if x["median"]] + [x["winner"] for x in rows if x["winner"]]
        vals += [r["mark"] for r in self.old if yearf(r["date"]) >= self.x0]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        self.ylim = (lo - pad, hi + pad)
        # percentage series
        b = base
        self.base_year = b["year"] if b else None
        if b:
            p0r, p0f = A.perf(ev, b["record"]), A.perf(ev, b["median"])
            self.pr_x = [yearf(b["date"])]
            self.pr_y = [0.0]
            for r in self.line:
                yx = yearf(r["date"])
                if self.pr_x[0] < yx <= yearf(last_date(rows)):
                    self.pr_x.append(yx)
                    self.pr_y.append(A.perf(ev, r["mark"]) / p0r - 1)
            self.pf = [(x["year"] + 0.6, A.perf(ev, x["median"]) / p0f - 1) for x in rows
                       if x["median"] and x["year"] >= b["year"]]
            last = [x for x in rows if x["median"] and x["record"]][-1]
            self.pct_end = yearf(last["date"])
            self.rec_impr, self.field_impr = last["rec_impr"], last["field_impr"]
            self.gap_last, self.gap0 = last["gap"], b["gap"]
            ys = self.pr_y + [p for _, p in self.pf]
            lo, hi = min(ys + [0]), max(ys)
            pad = (hi - lo) * 0.12 + 0.002
            self.plim = (lo - pad * 0.6, hi + pad)

    def record_at(self, year):
        i = np.searchsorted(self.rx, year, side="right") - 1
        return self.line[i] if i >= 0 else None


TITLE_EV = A.NAMES
SEXW = {"M": "Men's", "W": "Women's"}


def last_date(rows):
    return [x for x in rows if x["median"] and x["record"]][-1]["date"]


def pct(v, digits=1):
    return f"{v * 100:+.{digits}f}%".replace("+", "+").replace("-", "−")


class EventCard:
    def __init__(self, sex, ev, chapter="", callout="", dur=9.0, fade_in=0.45, fade_out=0.45):
        self.d = EventData(sex, ev)
        self.chapter, self.callout = chapter, callout
        self.dur, self.fin, self.fout = dur, fade_in, fade_out
        self.sweep = (0.35, dur - 2.6)

    def year_at(self, t):
        a, b = self.sweep
        return self.d.x0 + (2026.9 - self.d.x0) * ease((t - a) / (b - a))

    # layers (the explainer overrides these to build the chart up in steps)
    def rec_year(self, t):
        return self.year_at(t)

    def field_year(self, t):
        return self.year_at(t)

    def pct_year(self, t):
        return self.year_at(t)

    def dim_dots(self, t):
        return 1.0

    def caption(self, t):
        return [(self.callout, fade(t, 0.8, self.dur, 0.6, self.fout))] if self.callout else []

    def labels_alpha(self, t):
        return fade(t, self.sweep[1] - 0.2, 99, 0.6, 0)

    def counter_alpha(self, t):
        return fade(t, 0.2, self.sweep[1] + 0.8, 0.3, 0.8)

    def draw(self, fig, t):
        d = self.d
        al = fade(t, 0, self.dur, self.fin, self.fout)
        yr = self.rec_year(t)
        fyr = self.field_year(t)
        # heading
        text(fig, 96, 92, f"{self.chapter}".upper(), size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, f"{SEXW[d.sex]} {TITLE_EV[d.ev].lower() if d.ev in A.FIELD else TITLE_EV[d.ev]}",
              size=62, alpha=al)
        for line, a in self.caption(t):
            text(fig, 96, 212, line, size=28, color=INK2, alpha=al * a)
        # left: marks
        ax = ax_px(fig, 150, 270, 1080, 940)
        style_axes(ax)
        ax.set_xlim(d.x0, d.x1)
        ax.set_ylim(*d.ylim)
        if not d.field:
            ax.invert_yaxis()
        ax.yaxis.set_major_formatter(tick_fmt(d.ev))
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        ax.text(0, 1.035, "▲ faster" if not d.field else ("▲ higher" if d.ev in ("HJ", "PV") else "▲ farther"),
                transform=ax.transAxes, fontsize=fs(18), color=MUTED, alpha=al)
        # Olympic top eights
        dim = self.dim_dots(t)
        for x in d.rows:
            gx = x["year"] + 0.6
            if gx > fyr or not x["marks"]:
                continue
            pop = ease_out((fyr - gx) / 2.5)
            ms = [m for m in x["marks"] if d.ylim[0] <= m <= d.ylim[1]]
            # the column joins the marks near the middle of the final; a runner who pulled up stays a lone dot
            ref = x["median"] or (x["marks"][len(x["marks"]) // 2] if x["marks"] else None)
            tol = 0.08 if d.field else 0.03
            near = [m for m in ms if ref and abs(A.perf(d.ev, m) / A.perf(d.ev, ref) - 1) <= tol]
            if len(near) >= 2:
                ax.plot([gx, gx], [min(near), max(near)], color=BLUE, lw=px(2), alpha=0.45 * al * pop * dim, zorder=3,
                        solid_capstyle="round")
            ax.scatter([gx] * len(ms), ms, s=px(7) ** 2, color=BLUE, alpha=0.55 * al * pop * dim, linewidths=0,
                       zorder=4)
            if x["median"]:
                ring(ax, gx, x["median"], BLUE, size=13 * (0.6 + 0.4 * pop), alpha=al * pop, z=6)
        # old javelin (context only)
        if d.old:
            ox = np.array([yearf(r["date"]) for r in d.old])
            oy = np.array([r["mark"] for r in d.old])
            end = yearf(A.IMPLEMENT_FROM[(d.sex, "JT")])
            m = ox <= yr
            if m.any():
                xs = np.append(ox[m], min(yr, end))
                ys = np.append(oy[m], oy[m][-1])
                ax.step(xs, ys, where="post", color=GOLD, lw=px(2.5), alpha=0.35 * al, zorder=7)
                if yr > end:
                    ax.annotate("old javelin", (end, oy[-1]), xytext=(px(8), 0), textcoords="offset points",
                                ha="left", va="center", fontsize=fs(18), color=MUTED,
                                alpha=al * fade(yr, end, 9999, 1.5, 0))
        # record line
        m = d.rx <= yr
        if m.any():
            xs = np.append(d.rx[m], yr)
            ys = np.append(d.ry[m], d.ry[m][-1])
            kinds = [k for k, mm in zip(d.rkind, m) if mm]
            # hand-timed / pre-ratification part lighter
            n_light = sum(1 for k in kinds if k in ("hand", "pre"))
            if n_light:
                sw = d.rx[n_light] if n_light < len(d.rx) and d.rx[n_light] <= yr else yr
                xl = np.append(d.rx[:n_light], sw)
                yl = np.append(d.ry[:n_light], d.ry[n_light - 1])
                ax.step(xl, yl, where="post", color=GOLD, lw=px(3), alpha=0.55 * al, zorder=8)
            if n_light < len(xs) - 1:
                ax.step(xs[n_light:], ys[n_light:], where="post", color=GOLD, lw=px(3.5), alpha=al, zorder=8)
            cur = d.record_at(yr)
            if cur:
                ring(ax, yr, cur["mark"], GOLD, size=12, alpha=al, z=9)
                lab = f"{A.fmt(d.ev, cur['mark'])}  {clean_name(cur['athlete'])}, {cur['date'].year}"
                ax.annotate(lab, (yr, cur["mark"]), xytext=(px(-14), px(16)), textcoords="offset points",
                            ha="right", va="bottom", fontsize=fs(20), color=INK, alpha=al, zorder=10,
                            path_effects=outlined(6))
        # year counter
        text(fig, 1080, 160, f"{int(min(yr, 2026))}", size=62, color=INK2, weight="medium", ha="right",
             alpha=al * self.counter_alpha(t), family="Inter Display")
        # right: % better
        if d.base_year:
            pyr = self.pct_year(t)
            if pyr > d.base_year - 2 or self.pct_axes_alpha(t) > 0:
                self.draw_pct(fig, t, pyr, al * self.pct_axes_alpha(t))

    def pct_axes_alpha(self, t):
        return 1.0

    def draw_pct(self, fig, t, yr, al):
        d = self.d
        ax = ax_px(fig, 1240, 270, 1610, 940)
        style_axes(ax)
        ax.set_xlim(d.base_year - 2, 2026.5)
        ax.set_ylim(*d.plim)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v * 100:+.0f}%" if abs(v) > 1e-9 else "0"))
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        ax.spines["bottom"].set_alpha(al)
        unit = "speed" if not d.field else ("height" if d.ev in ("HJ", "PV") else "distance")
        ax.text(0, 1.035, f"% better than {d.base_year} ({unit}), to Paris 2024", transform=ax.transAxes,
                fontsize=fs(18), color=MUTED, alpha=al)
        prx, pry = np.array(d.pr_x), np.array(d.pr_y)
        yr = min(yr, d.pct_end)
        m = prx <= yr
        if m.any():
            xs, ys = np.append(prx[m], yr), np.append(pry[m], pry[m][-1])
            sw = d.auto_from
            if sw and sw > xs[0]:
                k = np.searchsorted(xs, sw)
                ax.step(np.append(xs[:k], min(sw, yr)), np.append(ys[:k], ys[k - 1]), where="post", color=GOLD,
                        lw=px(3), alpha=0.55 * al, zorder=5)
                if yr > sw:
                    ax.step(xs[k:], ys[k:], where="post", color=GOLD, lw=px(3.5), alpha=al, zorder=5)
            else:
                ax.step(xs, ys, where="post", color=GOLD, lw=px(3.5), alpha=al, zorder=5)
        pts = [(x, y) for x, y in d.pf if x <= yr]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=px(3), alpha=al, zorder=6)
            for x, y in pts:
                ring(ax, x, y, BLUE, size=10, alpha=al, z=7)
        # end labels once the sweep is done: number in ink, the series named in its colour
        la = al * self.labels_alpha(t)
        if la > 0:
            ry, fy = d.rec_impr, d.field_impr
            sep = (d.plim[1] - d.plim[0]) * 0.075
            if abs(ry - fy) < sep:
                mid = (ry + fy) / 2
                ry, fy = (mid + sep / 2, mid - sep / 2) if ry >= fy else (mid - sep / 2, mid + sep / 2)
            for v, y, name, c in ((d.rec_impr, ry, "record", GOLD), (d.field_impr, fy, "field", BLUE)):
                ax.annotate(pct(v), (1.0, y), xycoords=("axes fraction", "data"), xytext=(px(22), 0),
                            textcoords="offset points", color=INK, fontsize=fs(27), fontweight="bold", va="center",
                            alpha=la, annotation_clip=False)
                tr = blended_transform_factory(ax.transAxes, ax.transData)
                if name == "field":
                    ax.plot([1.0], [y], transform=offset_copy(tr, fig, x=px(142), units="points"), marker="o",
                            ms=px(13), color=c, mec=BG, mew=px(2), alpha=la, clip_on=False)
                else:
                    ax.annotate("", (1.0, y), xycoords=offset_copy(tr, fig, x=px(154), units="points"),
                                xytext=(1.0, y), textcoords=offset_copy(tr, fig, x=px(132), units="points"),
                                arrowprops=dict(arrowstyle="-", color=c, lw=px(4), alpha=la, shrinkA=0, shrinkB=0),
                                annotation_clip=False)
                ax.annotate(name, (1.0, y), xycoords=("axes fraction", "data"), xytext=(px(160), 0),
                            textcoords="offset points", color=INK2, fontsize=fs(21), va="center", alpha=la,
                            annotation_clip=False)


class Legend:
    """the key under the event cards"""
    @staticmethod
    def draw(fig, al):
        y = 1022
        ax = ax_px(fig, 0, 0, W, H)
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.plot([150, 190], [y - 8, y - 8], color=GOLD, lw=px(3.5), alpha=al)
        text(fig, 202, y, "world record (lighter: hand-timed or before ratification)", size=19, color=INK2, alpha=al)
        ax.scatter([830], [y - 8], s=px(7) ** 2, color=BLUE, alpha=0.55 * al, linewidths=0)
        text(fig, 846, y, "Olympic top eight", size=19, color=INK2, alpha=al)
        ax.scatter([1050], [y - 8], s=px(13) ** 2, color=BLUE, edgecolors=BG, linewidths=px(2), alpha=al)
        text(fig, 1068, y, "the field: median finalist", size=19, color=INK2, alpha=al)


class Card(EventCard):
    def draw(self, fig, t):
        super().draw(fig, t)
        Legend.draw(fig, fade(t, 0, self.dur, self.fin, self.fout))


class Seq:
    """sub-sections played one after another inside one rendered section"""

    def __init__(self, parts):
        self.parts = parts
        self.starts = np.cumsum([0] + [p.dur for p in parts[:-1]])
        self.dur = float(sum(p.dur for p in parts))

    def draw(self, fig, t):
        i = max(0, np.searchsorted(self.starts, t, side="right") - 1)
        self.parts[i].draw(fig, t - self.starts[i])


class ChapterCard:
    def __init__(self, name, sub, dur=3.0):
        self.name, self.sub, self.dur = name, sub, dur

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.5, 0.5)
        title(fig, 960, 520, self.name, size=96, ha="center", alpha=al)
        text(fig, 960, 590, self.sub, size=30, color=INK2, ha="center", alpha=al * fade(t, 0.3, self.dur, 0.5, 0.5))


class Caption:
    """centred lines of text, each with its own time window"""

    def __init__(self, lines, dur):
        self.lines, self.dur = lines, dur

    def draw(self, fig, t):
        for (t0, t1, y, s, size, color, weight) in self.lines:
            a = fade(t, t0, t1, 0.5, 0.5)
            if a > 0:
                if weight == "title":
                    title(fig, 960, y, s, size=size, color=color, ha="center", alpha=a)
                else:
                    text(fig, 960, y, s, size=size, color=color, weight=weight, ha="center", alpha=a)


TRACK = "#2b1f1d"
LANE = "#cfc9c0"


class Race:
    """two runners at their average speeds: a field-coloured one and a record-coloured one"""

    def __init__(self):
        g = A.games_for("M", "100m", GAMES, None)[1896]
        self.burke = (g["winner"], g["winner_mark"])
        cur = [r for r in REC[("M", "100m")] if not r["equalled"]][-1]
        self.bolt = (clean_name(cur["athlete"]), cur["mark"], cur["date"].year)
        self.t_go = 5.0
        self.t_freeze = self.t_go + self.bolt[1]
        self.dur = self.t_freeze + 4.2
        self.gap_m = 100 - 100 * self.bolt[1] / self.burke[1]

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        a1 = fade(t, 0, 4.6, 0.6, 0.6)
        title(fig, 960, 470, "Athens, 1896", size=80, ha="center", alpha=a1)
        text(fig, 960, 540, f"The first Olympic 100 metres. {self.burke[0]} wins in {self.burke[1]:.1f} seconds.",
             size=32, color=INK2, ha="center", alpha=a1 * fade(t, 0.8, 4.6, 0.6, 0.6))
        at = fade(t, 4.0, self.dur, 0.8, 0.6)
        if at <= 0:
            return
        ax = ax_px(fig, 0, 0, W, H)
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        x0, x1 = 200, 1720                                   # start and finish lines (100 m)
        lanes = [(430, BLUE, f"{self.burke[0]}", f"1896 Olympic champion · {self.burke[1]:.1f}", self.burke[1]),
                 (570, GOLD, f"{self.bolt[0]}", f"world record, {self.bolt[2]} · {self.bolt[1]:.2f}", self.bolt[1])]
        ax.add_patch(matplotlib.patches.Rectangle((120, 430), 1680, 280, color=TRACK, alpha=at, zorder=1))
        for y in (430, 570, 710):
            ax.plot([120, 1800], [y, y], color=LANE, lw=px(2), alpha=0.55 * at, zorder=2)
        ax.plot([x1, x1], [430, 710], color=INK, lw=px(4), alpha=at, zorder=2)
        ax.plot([x0, x0], [430, 710], color=LANE, lw=px(2), alpha=0.6 * at, zorder=2)
        rt = min(max(t - self.t_go, 0), self.bolt[1])         # race clock, frozen at the record's finish
        for top, c, name, sub, T in lanes:
            y = top + 96
            x = x0 + (x1 - x0) * min(rt / T, 1.0)
            trail = 70 * (1 if 0 < rt < T and t < self.t_freeze else 0)
            if trail:
                ax.plot([x - trail, x], [y, y], color=c, lw=px(10), alpha=0.25 * at, solid_capstyle="round", zorder=3)
            ax.scatter([x], [y], s=px(26) ** 2, color=c, edgecolors=BG, linewidths=px(3), alpha=at, zorder=4)
            text(fig, x0 + 28, top + 38, name, size=26, weight="semibold", alpha=at)
            text(fig, x0 + 28, top + 66, sub, size=20, color=INK2, alpha=at)
        text(fig, 1720, 400, f"{rt:5.2f} s", size=40, color=INK, weight="medium", ha="right", alpha=at,
             family="Inter Display")
        if t >= self.t_freeze:
            fa = fade(t, self.t_freeze, self.dur, 0.4, 0.6)
            xb = x0 + (x1 - x0) * self.bolt[1] / self.burke[1]
            yb = 430 + 62
            ax.annotate("", (x1 - 4, yb), (xb + 20, yb), arrowprops=dict(arrowstyle="<->", color=INK, lw=px(2.5)),
                        alpha=fa)
            text(fig, (xb + x1) / 2, yb - 14, f"{self.gap_m:.0f} metres", size=30, weight="bold", ha="center",
                 alpha=fa)
            text(fig, 960, 830, f"When the world record holder finishes, the first Olympic champion is "
                 f"{self.gap_m:.0f} metres back.", size=30, color=INK, ha="center", alpha=fa)
            text(fig, 960, 875, "Both at their average speeds.", size=21, color=MUTED, ha="center", alpha=fa)


def all_events():
    out = []
    for sex in ("M", "W"):
        for ev in A.ORDER:
            if (sex == "M" and ev == "100mH") or (sex == "W" and ev == "110mH"):
                continue
            out.append((sex, ev))
    return out


_ED = {}


def ED(sex, ev):
    if (sex, ev) not in _ED:
        _ED[(sex, ev)] = EventData(sex, ev)
    return _ED[(sex, ev)]


class Dumbbell:
    """record vs field improvement for every event of one sex: running on the left, jumps and throws right"""

    def __init__(self, sex, dur=13.0, heading="", sub=""):
        self.sex, self.dur, self.heading, self.sub = sex, dur, heading, sub
        evs = [e for s_, e in all_events() if s_ == sex]
        self.run = [ED(sex, e) for e in evs if e not in A.FIELD and ED(sex, e).base_year]
        self.fld = [ED(sex, e) for e in evs if e in A.FIELD and ED(sex, e).base_year]

    def panel(self, fig, t, items, x0, x1, top, al, label):
        n = len(items)
        row = 62
        ax = ax_px(fig, x0 + 250, top, x1 - 290, top + row * n)
        ax.set_axis_off()
        vmax = max(max(d.rec_impr, d.field_impr) for d in items)
        step = 5 if vmax < 0.35 else 20
        vmax = math.ceil(vmax * 100 / step) * step / 100
        ax.set_xlim(0, vmax)
        ax.set_ylim(n - 0.5, -0.5)
        for g in np.arange(0, vmax + 1e-9, step / 100):
            ax.plot([g, g], [-0.5, n - 0.5], color=GRID, lw=1.0, zorder=0, alpha=al)
            ax.text(g, n - 0.25, f"{g * 100:.0f}%" if g else "0", ha="center", va="top", fontsize=fs(18), color=INK2,
                    alpha=al)
        text(fig, x0, top - 34, label, size=24, weight="semibold", alpha=al)
        text(fig, x1 - 170, top - 34, "record", size=19, color=INK2, ha="right", alpha=al)
        text(fig, x1 - 60, top - 34, "field", size=19, color=INK2, ha="right", alpha=al)
        for i, d in enumerate(items):
            a = al * ease_out((t - 0.8 - 0.12 * i) / 0.6)
            if a <= 0:
                continue
            y = top + row * i + row / 2 + 9
            nm = A.SHORT[d.ev] + (f"  since {d.base_year}" if d.base_year != (1912 if d.sex == "M" else 0) else "")
            text(fig, x0, y, A.SHORT[d.ev], size=22, alpha=a)
            if d.base_year != 1912:
                text(fig, x0 + 136, y, f"from {d.base_year}", size=16, color=MUTED, alpha=a)
            ax.plot([d.rec_impr, d.field_impr], [i, i], color=GREY, lw=px(3), alpha=a, zorder=2,
                    solid_capstyle="round")
            ax.plot([d.rec_impr], [i], marker="D", ms=px(14), color=GOLD, mec=BG, mew=px(2), alpha=a, zorder=4)
            ax.plot([d.field_impr], [i], marker="o", ms=px(15), color=BLUE, mec=BG, mew=px(2), alpha=a, zorder=5)
            text(fig, x1 - 170, y, pct(d.rec_impr, 1), size=21, ha="right", alpha=a,
                 weight="bold" if d.rec_impr > d.field_impr else "regular")
            text(fig, x1 - 60, y, pct(d.field_impr, 1), size=21, ha="right", alpha=a,
                 weight="bold" if d.field_impr >= d.rec_impr else "regular")

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "EVERY EVENT", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, self.heading, size=56, alpha=al)
        text(fig, 96, 212, self.sub, size=27, color=INK2, alpha=al * fade(t, 0.6, self.dur, 0.6, 0.6))
        self.panel(fig, t, self.run, 96, 930, 320, al, "Running (speed)")
        self.panel(fig, t, self.fld, 1010, 1840, 320, al, "Jumps and throws (distance, height)")
        # key
        ax = ax_px(fig, 0, 0, W, H)
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.plot([110], [1012], marker="D", ms=px(14), color=GOLD, mec=BG, mew=px(2), alpha=al)
        text(fig, 130, 1020, "world record", size=19, color=INK2, alpha=al)
        ax.plot([300], [1012], marker="o", ms=px(15), color=BLUE, mec=BG, mew=px(2), alpha=al)
        text(fig, 320, 1020, "the field (median Olympic finalist), to the last Games", size=19, color=INK2, alpha=al)
        text(fig, 1840, 1020, "Jumps and throws scale with the square of speed, so their percentages run about twice as large.",
             size=17, color=MUTED, ha="right", alpha=al)


class Explainer(Card):
    """the men's 100 m card, built up layer by layer with captions"""

    def __init__(self):
        super().__init__("M", "100m", chapter="How to read the charts", dur=40.0, fade_in=0.8, fade_out=0.8)
        d = self.d
        self.T = {"rec": (1.2, 8.0), "field": (10.0, 13.0), "median": (16.0, 17.5), "pct": (21.5, 28.5)}
        self.lines = [
            (0.9, 9.6, "The gold line is the world record: every time it was broken, from the first known marks to today."),
            (9.8, 15.6, "Each blue column is one Olympic final: the top eight finishers, at every Games."),
            (15.8, 21.2, "The big dot is the middle of each final, the median finalist. That is the field."),
            (21.4, 29.6, f"On the right, both are measured against {d.base_year}: how much faster than back then."),
            (29.8, 35.8, f"Since {d.base_year} the record has got {d.rec_impr * 100:.1f}% faster. "
                         f"The field, {d.field_impr * 100:.1f}%."),
            (36.0, 40.0, "Up is better on every chart. Now, every event."),
        ]

    def _sweep(self, t, key, y0=None):
        a, b = self.T[key]
        y0 = self.d.x0 if y0 is None else y0
        return y0 + (2026.9 - y0) * ease((t - a) / (b - a))

    def rec_year(self, t):
        return self._sweep(t, "rec")

    def field_year(self, t):
        return self._sweep(t, "field")

    def pct_year(self, t):
        return self._sweep(t, "pct", self.d.base_year - 2)

    def pct_axes_alpha(self, t):
        return fade(t, self.T["pct"][0] - 0.8, 999, 0.8, 0)

    def dim_dots(self, t):
        return 1.0 - 0.7 * fade(t, self.T["median"][0], self.T["pct"][0], 0.8, 0.8)

    def labels_alpha(self, t):
        return fade(t, self.T["pct"][1], 999, 0.6, 0)

    def counter_alpha(self, t):
        return fade(t, self.T["rec"][0], self.T["rec"][1] + 1.0, 0.3, 0.8)

    def caption(self, t):
        return [(s, fade(t, a, b, 0.5, 0.5)) for a, b, s in self.lines]


SECTIONS = {}


def section(fn):
    SECTIONS[fn.__name__] = fn
    return fn


@section
def open():
    race = Race()
    cap = Caption([(0.2, 2.8, 540, "But a world record is one person, on one day.", 40, INK, "regular"),
                   (3.0, 5.6, 540, "What about everyone else?", 40, INK, "regular")], 6.0)
    ttl = Caption([(0.3, 5.4, 500, "The Record and the Rest", 104, INK, "title"),
                   (0.9, 5.4, 580, "130 years of world records, against the Olympic field", 32, INK2, "regular")], 5.8)
    return Seq([race, cap, ttl])


@section
def how():
    return Explainer()


class GapChart:
    """the field as a percentage of the record, Games by Games, every event of one sex (grey), some picked out"""

    def __init__(self, sex, highlight=(), dur=14.0, heading="", sub="", note="", x0=None):
        self.sex, self.dur, self.heading, self.sub, self.note = sex, dur, heading, sub, note
        self.items = [ED(sex, e) for s_, e in all_events() if s_ == sex and ED(sex, e).base_year]
        self.highlight = list(highlight)
        self.x0 = x0 or (1910 if sex == "M" else 1926)
        # the median gap across events, Games by Games (only Games where at least half the events ran)
        by = {}
        for d in self.items:
            for x in d.rows:
                if x["gap"] and x["year"] >= d.base_year:
                    by.setdefault(x["year"], []).append(x["gap"])
        self.med = sorted((y, float(np.median(v))) for y, v in by.items() if len(v) >= max(3, len(self.items) // 3))

    def series(self, d):
        return [(x["year"], x["gap"]) for x in d.rows if x["gap"] and x["year"] >= d.base_year]

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "HOW CLOSE DOES THE FIELD GET?", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, self.heading, size=56, alpha=al)
        text(fig, 96, 212, self.sub, size=27, color=INK2, alpha=al * fade(t, 0.6, self.dur, 0.6, 0.6))
        ax = ax_px(fig, 170, 290, 1560, 940)
        style_axes(ax)
        ax.set_xlim(self.x0, 2026)
        lo = min(g for d in self.items for _, g in self.series(d))
        ax.set_ylim(min(0.80, math.floor(lo * 50) / 50 - 0.005), 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v * 100:.0f}%"))
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.axhline(1.0, color=GOLD, lw=px(2.5), alpha=0.8 * al, zorder=2)
        ax.text(self.x0 + 1, 0.993, "the world record at the time", color=INK2, fontsize=fs(19), va="top", alpha=al)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_alpha(al)
        yr = self.x0 + (2026 - self.x0) * ease((t - 0.8) / 6.0)
        for d in self.items:
            pts = [(y, g) for y, g in self.series(d) if y <= yr]
            if len(pts) > 1 and d.ev not in self.highlight:
                ax.plot(*zip(*pts), color=GREY, lw=px(1.6), alpha=0.6 * al, zorder=3)
        hl_a = al * fade(t, 7.2, 999, 0.8, 0)
        ends = []
        for d in self.items:
            if d.ev in self.highlight:
                pts = [(y, g) for y, g in self.series(d) if y <= yr]
                if len(pts) > 1:
                    ax.plot(*zip(*pts), color=GREY, lw=px(1.6), alpha=0.6 * al * (1 - hl_a), zorder=3)
                    ax.plot(*zip(*pts), color=BLUE, lw=px(3), alpha=hl_a, zorder=5)
                    ends.append([pts[-1][1], pts[-1], A.SHORT[d.ev]])
        # end labels pushed apart so they never overlap
        ends.sort(key=lambda e: e[0])
        gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.035
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < gap:
                ends[i][0] = ends[i - 1][0] + gap
        for ly, pt, name in ends:
            ax.annotate(name, (pt[0], ly), xytext=(px(10), 0), textcoords="offset points", color=INK,
                        fontsize=fs(20), va="center", alpha=hl_a, path_effects=outlined(5), annotation_clip=False)
        med = [(y, g) for y, g in self.med if y <= yr]
        ma = al * (1 - hl_a * 0.6) if self.highlight else al
        if len(med) > 1:
            ax.plot(*zip(*med), color=INK, lw=px(4), alpha=ma, zorder=6)
            ax.annotate(f"median of all events  {med[-1][1] * 100:.1f}%", med[-1], xytext=(px(12), px(-2)),
                        textcoords="offset points", color=INK, fontsize=fs(21), fontweight="semibold", va="center",
                        alpha=ma, annotation_clip=False, path_effects=outlined(5))
        if self.note:
            text(fig, 170, 1020, self.note, size=19, color=MUTED, alpha=al * fade(t, 2.0, self.dur, 0.6, 0.6))


class RecordAges:
    """when each current world record was set"""

    def __init__(self, dur=13.0, cut=1990, heading="", sub=""):
        self.dur, self.cut, self.heading, self.sub = dur, cut, heading, sub
        self.cols = {}
        for sex in ("M", "W"):
            items = []
            for s_, e in all_events():
                if s_ != sex:
                    continue
                line = [r for r in REC[(sex, e)] if not r["equalled"]]
                cur = line[-1]
                items.append((yearf(cur["date"]), A.SHORT[e], clean_name(cur["athlete"]), A.fmt(e, cur["mark"]),
                              cur["status"]))
            self.cols[sex] = sorted(items)

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.6)
        text(fig, 96, 92, "THE RECORDS TODAY", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, self.heading, size=56, alpha=al)
        text(fig, 96, 212, self.sub, size=27, color=INK2, alpha=al * fade(t, 0.6, self.dur, 0.6, 0.6))
        for k, (sex, x0, x1) in enumerate((("M", 96, 930), ("W", 1010, 1840))):
            items = self.cols[sex]
            text(fig, x0, 300, "Men" if sex == "M" else "Women", size=24, weight="semibold", alpha=al)
            ax = ax_px(fig, x0 + 150, 320, x1 - 300, 320 + 40 * len(items))
            ax.set_axis_off()
            ax.set_xlim(1980, 2027)
            ax.set_ylim(len(items) - 0.5, -0.5)
            for g in (1980, 1990, 2000, 2010, 2020):
                ax.plot([g, g], [-0.5, len(items) - 0.5], color=GRID, lw=1.0, zorder=0, alpha=al)
                ax.text(g, len(items) - 0.3, str(g), ha="center", va="top", fontsize=fs(17), color=INK2, alpha=al)
            for i, (yx, ev, who, mark, st) in enumerate(items):
                a = al * ease_out((t - 0.8 - 0.08 * i - 0.6 * k) / 0.5)
                if a <= 0:
                    continue
                old = yx < self.cut
                y = 320 + 40 * i + 20 + 8
                text(fig, x0, y, ev, size=20, alpha=a)
                ax.barh([i], [2026.84 - yx], left=[yx], height=0.56, color=GOLD, alpha=a * (1.0 if old else 0.35),
                        zorder=2)
                text(fig, x1 - 290, y, f"{int(yx)}", size=19, color=INK if old else INK2, alpha=a,
                     weight="bold" if old else "regular")
                text(fig, x1 - 230, y, f"{who}", size=17, color=INK2, alpha=a)


CALLOUTS = {
    ("M", "100m"): "Since 1912 the record is 10.6% faster and the field 10.8%. They have moved together.",
    ("W", "100m"): "Florence Griffith Joyner's 10.49 has stood since 1988. No Olympic field has come within 3.5% of it.",
    ("M", "200m"): "Usain Bolt's 19.19 pulled the record away: since 1952 the record has gained 7.3%, the field 4.5%.",
    ("W", "200m"): "Griffith Joyner's 21.34 is also from 1988. Since 1948 the field has gained more than the record.",
    ("M", "400m"): "From 47.8 in 1912 to Wayde van Niekerk's 43.03 in 2016. The Paris field ran at 98.2% of it.",
    ("W", "400m"): "Marita Koch's 47.60 dates from 1985. Since 1964 the field has gained 10.6%, the record 9.0%.",
    ("M", "110mH"): "Ja'Kobe Tharp ran 12.75 in June 2026, breaking a record that had stood since 2012.",
    ("W", "100mH"): "Masai Russell ran 12.09 in August 2026. Since 1972 the field has gained more than the record.",
    ("M", "400mH"): "Karsten Warholm's 45.94 won Tokyo. Alison dos Santos ran 45.80 in August 2026.",
    ("W", "400mH"): "Sydney McLaughlin-Levrone broke it six times from 2021 to 2024, the last in the Paris final.",
    ("M", "800m"): "The tightest of all: the Paris field ran at 99.1% of David Rudisha's record speed.",
    ("W", "800m"): "Jarmila Kratochvílová's 1:53.28 is from 1983: the oldest record in this film.",
    ("M", "1500m"): "Hicham El Guerrouj's 3:26.00 has stood since 1998. The Paris field ran at 98.6% of it.",
    ("W", "1500m"): "Faith Kipyegon has lowered the record three times since 2023.",
    ("M", "5000m"): "Olympic distance finals are tactical: the Paris field ran at 95% of record speed.",
    ("W", "5000m"): "Olympic since 1996. The field (+4.5%) and the record (+4.3%) have moved almost exactly together.",
    ("M", "10000m"): "Since 1912 the field (+21.1%) has gained more than the record (+18.3%).",
    ("W", "10000m"): "Since 1988 the record has gained 4.6%, the Olympic field 2.6%.",
    ("M", "HJ"): "Javier Sotomayor's 2.45 m has stood since 1993. Since 1912 the field has gained more than the record.",
    ("W", "HJ"): "Yaroslava Mahuchikh's 2.10 m in 2024 broke a record that had stood for 37 years.",
    ("M", "PV"): "Armand Duplantis has broken it 15 times since 2020, once in the Paris final.",
    ("W", "PV"): "Olympic since 2000. Yelena Isinbayeva's 5.06 m (2009) still stands.",
    ("M", "LJ"): "Bob Beamon's 8.90 in 1968 added 55 cm in one jump. Mike Powell's 8.95 has stood since 1991.",
    ("W", "LJ"): "Galina Chistyakova's 7.52 m is from 1988. The Paris field jumped 90% of it.",
    ("M", "TJ"): "Since 1912 the field has gained 24%, the record 18%. Jonathan Edwards's 18.29 m is from 1995.",
    ("W", "TJ"): "Olympic since 1996, and the record has moved just 1.5%: Yulimar Rojas's 15.74 m (2022).",
    ("M", "SP"): "Since 1912 the field has gained 63%, the record 52%. Ryan Crouser's 23.56 m is from 2023.",
    ("W", "SP"): "Natalya Lisovskaya's 22.63 m is from 1987. The Paris field threw 85% of it.",
    ("M", "DT"): "Jürgen Schult's 74.08 m stood from 1986 to 2024. Mykolas Alekna has broken it three times since.",
    ("W", "DT"): "The widest gap of any event: the Paris field threw 84.9% of Gabriele Reinsch's 76.80 m (1988).",
    ("M", "JT"): "A new javelin from 1986. Jan Železný's 98.48 m (1996) is 10 m beyond the Paris median.",
    ("W", "JT"): "The only event whose field went backwards: the Paris median threw shorter than Sydney 2000's.",
    ("M", "HT"): "Yuriy Sedykh's 86.74 m (1986) is the oldest men's record. The field has gained 67% since 1920.",
    ("W", "HT"): "Olympic since 2000. Anita Włodarczyk's 82.98 m (2016) still stands.",
}

CHAPTERS = {
    "sprints": ("Sprints", "100, 200 and 400 metres", ["100m", "200m", "400m"]),
    "hurdles": ("Hurdles", "110 m (men), 100 m (women) and 400 m hurdles", ["110mH", "400mH"]),
    "distance": ("Middle and long distance", "800 to 10,000 metres", ["800m", "1500m", "5000m", "10000m"]),
    "jumps": ("Jumps", "High jump, pole vault, long jump, triple jump", ["HJ", "PV", "LJ", "TJ"]),
    "throws": ("Throws", "Shot put, discus, javelin, hammer", ["SP", "DT", "JT", "HT"]),
}
CARD_DUR = 9.0


def chapter(key):
    name, sub, evs = CHAPTERS[key]
    parts = [ChapterCard(name, sub, 3.0)]
    for ev in evs:
        for sex in ("M", "W"):
            e = "100mH" if (ev == "110mH" and sex == "W") else ev
            parts.append(Card(sex, e, chapter=name, callout=CALLOUTS[(sex, e)], dur=CARD_DUR))
    return Seq(parts)


for _k in CHAPTERS:
    SECTIONS[_k] = (lambda k: (lambda: chapter(k)))(_k)


def _count(sex):
    items = [ED(s_, e) for s_, e in all_events() if s_ == sex and ED(s_, e).base_year]
    return sum(d.rec_impr > d.field_impr for d in items), len(items)


@section
def compare():
    rm, nm = _count("M")
    rw, nw = _count("W")
    gm = GapChart("M", heading="Men: the field keeps pace",
                  sub="The middle of an Olympic final has stayed within about 4% of the world record since 1912.")
    gw = GapChart("W", highlight=("400m", "800m", "SP", "DT"), heading="Women: the field fell back",
                  sub="Closest to the record around 1980; further away since the records of the 1980s.",
                  note="Most of the 1980s records were set before random out-of-competition drug testing became routine.",
                  dur=15.0)
    ages = RecordAges(heading="How old the records are",
                      sub="Seven of the women's records were set in 1983–88. One of the men's is from the 1980s.")
    return Seq([
        Dumbbell("M", heading="Men: the record against the field",
                 sub=f"How much better than in 1912. The record gained more in {rm} of {nm} events, the field in {nm - rm}."),
        Dumbbell("W", heading="Women: the record against the field",
                 sub=f"How much better than at each event's first Games. The record gained more in {rw} of {nw}."),
        gm, gw, ages])


@section
def close():
    gm = GapChart("M")
    first, last = gm.med[0], gm.med[-1]
    return Caption([
        (0.4, 7.4, 470, f"The middle of the men's Olympic final ran, jumped and threw at", 40, INK, "regular"),
        (0.4, 7.4, 530, f"{first[1] * 100:.1f}% of the world record in {first[0]}, and {last[1] * 100:.1f}% in {last[0]}.",
         40, INK, "regular"),
        (7.8, 13.6, 480, "The record is one person, on one day.", 44, INK, "regular"),
        (8.6, 13.6, 545, "The field is everyone who made the final. Mostly, it has kept up.", 44, INK, "regular"),
        (14.0, 21.0, 470, "The Record and the Rest", 64, INK, "title"),
        (14.4, 21.0, 540, "World records: World Athletics progressions, via Wikipedia (November 2026).", 24, INK2, "regular"),
        (14.4, 21.0, 578, "Olympic results: Olympedia. The field is the median of each final's top eight.", 24, INK2, "regular"),
        (14.4, 21.0, 616, "Every frame drawn in code. Method and sources in the description.", 24, INK2, "regular"),
    ], 21.0)


if __name__ == "__main__":
    if sys.argv[1] == "still":
        name, t = sys.argv[2], float(sys.argv[3])
        scale = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        os.makedirs(OUT, exist_ok=True)
        still(SECTIONS[name](), t, os.path.join(OUT, f"still_{name}_{t:g}.png"), scale)
    else:
        name = sys.argv[1]
        scale = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print(render_section(name, SECTIONS[name](), scale=scale))
