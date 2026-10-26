"""Win now, or tear it down? The video (1920x1080, 30 fps, captions). Every number comes from results.json and the
tables beside it (analyse.py, build.py).

    python video.py NAME [still times...]
    NAME: hook, theory, stories, richer, shifted, winnow, teardown, titles, verdict, outro
"""
import sys
import numpy as np
from PIL import Image, ImageDraw

import look as L
from look import W, H, INK, INK2, MUTED, CYAN, GOLD, ROSE, HOT, font, fade, smooth, cap

R = L.results()
XF, BAR = 0.5, 2.5                          # 96 BPM: sections are whole bars long (+ the cross-fade)
TH = R["thirds"]
H1, CTRL, H2, CH = R["h1"], R["controls"], R["h2"], R["champions"]


def bars(n):
    return n * BAR + XF


def pct(v, d=0):
    return f"{100 * v:.{d}f}%"


def nth(v):
    n = int(round(100 * v))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rho(k, src=None):
    return (src or H1)[k]["rho"]


GRID_BOX = (330, 172, 1780, 808)
_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        _GRID = L.stripes(GRID_BOX)
    return _GRID


def draw_grid(im, t_cols, a=1.0, labels=True):
    """the season grid, revealed column by column up to t_cols (columns, fractional), at opacity a"""
    g, geo, cw = grid()
    arr = np.array(g)
    x = np.arange(arr.shape[1])
    ramp = np.clip(t_cols - x / cw, 0, 1) * a
    arr[..., 3] = (arr[..., 3] * ramp[None, :]).astype(np.uint8)
    layer = Image.fromarray(arr)
    im.paste(layer, (GRID_BOX[0], GRID_BOX[1]), layer)
    if labels and a > 0:
        d = ImageDraw.Draw(im, "RGBA")
        for lg in L.LEAGUES:
            y0, y1 = geo[lg]
            d.text((GRID_BOX[0] - 24, (y0 + y1) / 2), L.NAME[lg], font=font(30, "Bold"), fill=INK + (int(230 * a),), anchor="rm")
        for yr in (2001, 2005, 2010, 2015, 2020, 2025):
            xc = GRID_BOX[0] + (yr - 2001 + 0.5) * cw
            d.text((xc, GRID_BOX[3] + 12), str(yr), font=font(22, "Regular"), fill=MUTED + (int(230 * a),), anchor="ma")


class Hook(L.Section):
    name, dur = "hook", bars(8)

    def frame(self, t):
        im = L.canvas()
        draw_grid(im, 25 * smooth(t / 6.0) * 1.08)
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0.8, self.dur + 1, 0.8)
        d.text((W / 2, 46), "Win now, or tear it down?", font=font(58, "Bold"), fill=INK + (int(255 * a),), anchor="ma")
        a2 = fade(t, 2.0, self.dur + 1, 0.8)
        d.text((W / 2, 122), "Every NHL, NBA, MLB and NFL franchise, every season since 2000-01  ·  brighter is a better finish, white a title",
               font=font(25, "Regular"), fill=INK2 + (int(255 * a2),), anchor="ma")
        cap(d, ["Some fan bases won't sit through a rebuild."], fade(t, 7.0, 13.0))
        cap(d, ["Does that trap their teams,", "and does tearing it down pay off?"], fade(t, 13.4, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Theory(L.Section):
    name, dur = "theory", bars(7)

    def frame(self, t):
        im = L.canvas()
        draw_grid(im, 26, a=0.18, labels=False)
        d = ImageDraw.Draw(im, "RGBA")
        d.text((W / 2, 170), "THE THEORY", font=font(30, "Bold"), fill=GOLD + (int(255 * fade(t, 0.3, self.dur + 1)),), anchor="ma")
        for i, (t0, head, body) in enumerate(((1.0, "1   Rich, rabid markets can't stomach a rebuild.",
                                              "So owners trade picks and prospects to win now."),
                                             (6.5, "2   Teams that can tear it down climb higher later,",
                                              "at the cost of lower lows."))):
            a = fade(t, t0, self.dur + 1, 0.6)
            y = 300 + i * 230
            d.text((W / 2, y), head, font=font(48, "SemiBold"), fill=INK + (int(255 * a),), anchor="ma")
            d.text((W / 2, y + 70), body, font=font(36, "Regular"), fill=INK2 + (int(255 * a),), anchor="ma")
        a = fade(t, 12.0, self.dur + 1, 0.6)
        d.text((W / 2, 830), "Tested on 124 franchises and 3,026 seasons, 2000-01 to 2024-25,", font=font(30, "Regular"),
               fill=INK2 + (int(255 * a),), anchor="ma")
        d.text((W / 2, 874), "with Forbes values, every draft pick and every NFL trade.", font=font(30, "Regular"),
               fill=INK2 + (int(255 * a),), anchor="ma")
        return np.asarray(im.convert("RGB"))


STORIES = [("mlb", "NYY", "New York Yankees"), ("mlb", "LAD", "Los Angeles Dodgers"), ("nhl", "PIT", "Pittsburgh Penguins"),
           ("mlb", "HOU", "Houston Astros"), ("nhl", "EDM", "Edmonton Oilers"), ("nba", "PHI", "Philadelphia 76ers")]


class Stories(L.Section):
    name, dur = "stories", bars(8)

    def __init__(self):
        S = L.rows("seasons.csv")
        self.rows = []
        for lg, team, nm in STORIES:
            mine = [s for s in S if s["league"] == lg and s["team"] == team]
            b3 = sum(int(s["rank"]) > int(s["teams"]) - 3 for s in mine)
            ti = sum(int(s["champion"]) for s in mine)
            self.rows.append((lg, nm, mine, b3, ti))

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        d.text((W / 2, 60), "Stories cut both ways", font=font(52, "Bold"), fill=INK + (int(255 * fade(t, 0.3, self.dur + 1)),), anchor="ma")
        x0, cw, rh, y0 = 560, 34, 46, 200
        for i, (lg, nm, mine, b3, ti) in enumerate(self.rows):
            a = fade(t, 0.8 + 0.5 * i + (2.5 if i >= 2 else 0) + (2.5 if i >= 4 else 0), self.dur + 1, 0.5)
            y = y0 + i * (rh + 28) + (24 if i >= 2 else 0) + (24 if i >= 4 else 0)
            d.text((x0 - 24, y + rh / 2), nm, font=font(30, "SemiBold"), fill=INK + (int(255 * a),), anchor="rm")
            for s in mine:
                c = L.season_col(lg, s["season"]) - 2001
                if 0 <= c < 25:
                    col = HOT if s["champion"] == "1" else L.rank_colour(float(s["rank_pct"]))
                    d.rectangle([x0 + c * cw + 2, y, x0 + (c + 1) * cw - 2, y + rh], fill=col + (int(255 * a),))
                    if int(s["rank"]) > int(s["teams"]) - 3:
                        d.rectangle([x0 + c * cw + 2, y + rh + 4, x0 + (c + 1) * cw - 2, y + rh + 9], fill=ROSE + (int(255 * a),))
                    if s["champion"] == "1":
                        xc = x0 + (c + 0.5) * cw
                        d.polygon([(xc, y - 14), (xc + 6, y - 8), (xc, y - 2), (xc - 6, y - 8)], fill=HOT + (int(255 * a),))
            txt = f"{b3} bottom-three finish{'es' if b3 != 1 else ''} · {ti} title{'s' if ti != 1 else ''}"
            d.text((x0 + 25 * cw + 24, y + rh / 2), txt, font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="lm")
        ak = fade(t, 1.0, self.dur + 1)
        d.text((x0, y0 - 40), "2001", font=font(20, "Regular"), fill=MUTED + (int(255 * ak),), anchor="lm")
        d.text((x0 + 25 * cw, y0 - 40), "2025", font=font(20, "Regular"), fill=MUTED + (int(255 * ak),), anchor="rm")
        d.rectangle([x0 + 6 * cw, y0 - 48, x0 + 6 * cw + 30, y0 - 43], fill=ROSE + (int(255 * ak),))
        d.text((x0 + 6 * cw + 40, y0 - 40), "bottom three in the league", font=font(20, "Regular"), fill=MUTED + (int(255 * ak),), anchor="lm")
        xc = x0 + 15 * cw
        d.polygon([(xc, y0 - 49), (xc + 6, y0 - 43), (xc, y0 - 37), (xc - 6, y0 - 43)], fill=HOT + (int(255 * ak),))
        d.text((xc + 14, y0 - 40), "title", font=font(20, "Regular"), fill=MUTED + (int(255 * ak),), anchor="lm")
        cap(d, ["Never bottomed out, and won.", "Tore it down, and won.  Tore it down, and didn't."], fade(t, 6.0, 13.0))
        cap(d, ["For every Penguins there's an Oilers.", "So: what do 3,026 seasons say?"], fade(t, 13.4, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Richer(L.Section):
    name, dur = "richer", bars(8)
    BASE, SCALE, GX = 720, 3300, (560, 960, 1360)

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        L.title(d, "Do valuable teams avoid the bottom?", fade(t, 0.2, self.dur + 1),
                "Each league's franchises in thirds by Forbes value, their seasons in their current market")
        a0 = fade(t, 0.6, self.dur + 1)
        d.line([(300, self.BASE), (1620, self.BASE)], fill=MUTED + (int(200 * a0),), width=2)
        g1, g2 = smooth((t - 1.0) / 1.5), smooth((t - 10.0) / 1.5)
        for gx, th, lab in zip(self.GX, TH, ("Poorest third", "Middle third", "Richest third")):
            d.text((gx, self.BASE + 18), lab, font=font(30, "SemiBold"), fill=INK + (int(255 * a0),), anchor="ma")
            for k, (key, col, dx, g) in enumerate((("bottom3", ROSE, -70, g1), ("top3", GOLD, 70, g2))):
                v = th[key]
                h = v * self.SCALE * g
                L.bar(d, gx + dx - 56, self.BASE, gx + dx + 56, self.BASE - h, col, a=1.0 if g > 0 else 0)
                if g > 0.6:
                    d.text((gx + dx, self.BASE - h - 12), pct(v, 1), font=font(30, "SemiBold"), fill=INK + (int(255 * min(1, (g - 0.6) / 0.4)),), anchor="mb")
            at = fade(t, 13.5, self.dur + 1)
            d.text((gx, self.BASE + 62), f"{th['titles']} titles", font=font(28, "Regular"), fill=GOLD + (int(255 * at),), anchor="ma")
        for col, lab, a in ((ROSE, "bottom-three finishes", fade(t, 1.4, self.dur + 1)), (GOLD, "top-three finishes", fade(t, 10.4, self.dur + 1))):
            y = 236 + (0 if col == ROSE else 44)
            d.rectangle([300, y - 12, 324, y + 12], fill=col + (int(255 * a),))
            d.text((336, y), lab, font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="lm")
        p, r = TH[0], TH[2]
        cap(d, ["Yes. The richest third bottomed out less often:", f"{pct(r['bottom3'], 1)} of seasons, against {pct(p['bottom3'], 1)} for the poorest."],
            fade(t, 3.0, 9.6))
        cap(d, ["But they also finished top three nearly twice as often,", f"and won {r['titles']} titles to {p['titles']}."], fade(t, 11.0, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Shifted(L.Section):
    name, dur = "shifted", bars(8)
    X0, X1, BASE, SCALE = 360, 1360, 760, 3000

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        L.title(d, "Shifted, not steadier", fade(t, 0.2, self.dur + 1), "Share of seasons by finish in the league, poorest and richest thirds")
        a0 = fade(t, 0.5, self.dur + 1)
        d.line([(self.X0 - 40, self.BASE), (self.X1 + 40, self.BASE)], fill=MUTED + (int(200 * a0),), width=2)
        for v in (0.05, 0.10, 0.15):
            y = self.BASE - v * self.SCALE
            d.line([(self.X0 - 40, y), (self.X1 + 40, y)], fill=(40, 52, 72, int(255 * a0)), width=1)
            d.text((self.X0 - 56, y), pct(v), font=font(22, "Regular"), fill=MUTED + (int(255 * a0),), anchor="rm")
        d.text((self.X0, self.BASE + 16), "worst tenth", font=font(26, "Regular"), fill=INK2 + (int(255 * a0),), anchor="ma")
        d.text((self.X1, self.BASE + 16), "best tenth", font=font(26, "Regular"), fill=INK2 + (int(255 * a0),), anchor="ma")
        d.text(((self.X0 + self.X1) / 2, self.BASE + 16), "finish in the league", font=font(26, "Regular"), fill=MUTED + (int(255 * a0),), anchor="ma")
        G = L.Glow()
        xs = np.linspace(self.X0, self.X1, 10)
        for k, (th, col, t0) in enumerate(((TH[0], CYAN, 1.0), (TH[2], GOLD, 4.0))):
            u = smooth((t - t0) / 2.0)
            if u <= 0:
                continue
            ys = self.BASE - np.array(th["deciles"]) * self.SCALE
            n = 1 + u * 9
            m = int(np.floor(n))
            pts = list(zip(xs[:m], ys[:m]))
            if m < 10:
                f = n - m
                pts.append((xs[m - 1] + (xs[m] - xs[m - 1]) * f, ys[m - 1] + (ys[m] - ys[m - 1]) * f))
            G.line(pts, col, 5)
            for x, y in pts[:m]:
                G.dot(x, y, col, 7)
            if u >= 1:
                lab = ("poorest third" if k == 0 else "richest third") + f"  ·  spread {th['sd_rank']:.3f}"
                d.text((self.X1 + 40, ys[-1]), lab, font=font(26, "SemiBold"), fill=col + (int(255 * fade(t, t0 + 2, self.dur + 1)),), anchor="lm")
        im = G.over(im)
        d = ImageDraw.Draw(im, "RGBA")
        cap(d, ["Their seasons spread just as widely.", "The whole curve slides toward the top."], fade(t, 6.5, 12.0))
        cap(d, ["Compare teams of equal quality and the effect is gone:",
                f"{rho('value_z~tank_rate_lz'):+.2f} becomes {CTRL['value_z~tank_rate|mean_rank']['rho']:+.2f}. Rich teams aren't avoiding rebuilds; they're better."],
            fade(t, 12.4, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Winnow(L.Section):
    name, dur = "winnow", bars(8)
    ZERO, SCALE = 1300, 420

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        L.title(d, "Do they play win-now?", fade(t, 0.2, self.dur + 1), "Spearman correlation across franchises, within each league")
        items = [("Older rosters", "in bigger markets, at equal quality", CTRL["market_z~age_z|mean_rank"], "Yes", GOLD, 1.0),
                 ("Fewer of their own first-round picks", "in richer teams, at equal quality", CTRL["value_z~firsts_per_draft|mean_rank"], "A little", GOLD, 5.0),
                 ("NFL first-rounders traded away", "in richer teams", H1["value_z~net_firsts_traded_away_lz"], "No sign", MUTED, 9.0)]
        a0 = fade(t, 0.6, self.dur + 1)
        d.line([(self.ZERO, 250), (self.ZERO, 760)], fill=MUTED + (int(200 * a0),), width=2)
        for v, lab in ((-0.5, "-0.5"), (0.0, "0"), (0.5, "+0.5")):
            d.text((self.ZERO + v * self.SCALE, 772), lab, font=font(22, "Regular"), fill=MUTED + (int(255 * a0),), anchor="ma")
        for i, (head, sub, s, verdict, col, t0) in enumerate(items):
            a = fade(t, t0, self.dur + 1, 0.5)
            y = 330 + i * 160
            d.text((160, y - 22), head, font=font(36, "SemiBold"), fill=INK + (int(255 * a),), anchor="lm")
            d.text((160, y + 24), sub, font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="lm")
            g = smooth((t - t0 - 0.3) / 1.0)
            v = s["rho"] * g
            x1 = self.ZERO + v * self.SCALE
            if g > 0:
                d.rounded_rectangle([min(self.ZERO, x1), y - 20, max(self.ZERO, x1), y + 20], 6, fill=col + (int(255 * a),))
                d.text((x1 + (14 if v >= 0 else -14), y), f"{s['rho']:+.2f}", font=font(30, "SemiBold"), fill=INK + (int(255 * a),),
                       anchor="lm" if v >= 0 else "rm")
            av = fade(t, t0 + 1.4, self.dur + 1)
            d.rounded_rectangle([1660, y - 24, 1800, y + 24], 24, outline=col + (int(255 * av),), width=3)
            d.text((1730, y), verdict, font=font(26, "SemiBold"), fill=col + (int(255 * av),), anchor="mm")
        cap(d, ["Valuable teams do lean win-now, a little.",
                f"But it's market size more than fan passion (the premium's roster-age link: {CTRL['premium_z~age_z|mean_rank']['rho']:+.2f})."],
            fade(t, 12.0, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Teardown(L.Section):
    name, dur = "teardown", bars(12)
    X0, X1, Y0, Y1 = 240, 1340, 230, 790
    T_FORK, T_REST = 4.0, 11.0

    def x(self, k):
        return self.X0 + (k + 2) / 12 * (self.X1 - self.X0)

    def y(self, p):
        return self.Y1 - p * (self.Y1 - self.Y0)

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        L.title(d, "Does tearing it down pay?", fade(t, 0.2, self.dur + 1),
                "From a season in the middle of the pack: average finish in the league, the next ten seasons", y=50)
        a0 = fade(t, 0.6, self.dur + 1)
        for p, lab in ((0.0, "worst"), (0.25, "25th"), (0.5, "50th"), (0.75, "75th"), (1.0, "best")):
            y = self.y(p)
            d.line([(self.X0, y), (self.X1, y)], fill=(40, 52, 72, int(255 * a0)), width=1)
            d.text((self.X0 - 16, y), lab, font=font(22, "Regular"), fill=MUTED + (int(255 * a0),), anchor="rm")
        for k in range(-2, 11, 2):
            d.text((self.x(k), self.Y1 + 14), ("+" if k > 0 else "") + str(k), font=font(22, "Regular"), fill=MUTED + (int(255 * a0),), anchor="ma")
        d.text(((self.X0 + self.X1) / 2, self.Y1 + 46), "seasons from the mediocre one", font=font(24, "Regular"), fill=MUTED + (int(255 * a0),), anchor="ma")
        d.line([(self.x(0), self.Y0), (self.x(0), self.Y1)], fill=MUTED + (int(160 * a0),), width=2)
        af = fade(t, self.T_FORK, self.dur + 1)
        d.rectangle([self.x(0.5), self.Y0, self.x(2.5), self.Y1], fill=(255, 255, 255, int(12 * af)))
        d.text((self.x(1.5), self.Y0 - 10), "the fork", font=font(22, "Regular"), fill=MUTED + (int(255 * af),), anchor="mb")
        G = L.Glow()
        n = {"teardown": H2["all"]["n_teardown"], "stayed": H2["all"]["n_stayed"]}
        for key, col in (("stayed", CYAN), ("teardown", ROSE)):
            c = R["curves"][key]
            u = 2 + 2 * smooth((t - 1.5) / 2.5) + 8 * smooth((t - self.T_REST) / 6.0)     # how far along (index into the curve)
            m = int(np.floor(u))
            pts = [(self.x(k - 2), self.y(c[k])) for k in range(m + 1)]
            if m < 12:
                f = u - m
                pts.append((self.x(m - 2 + f), self.y(c[m] + (c[m + 1] - c[m]) * f)))
            G.line(pts, col, 5)
            G.dot(*pts[-1], col, 8)
            if u > 3.5:
                lab = ("tore it down" if key == "teardown" else "stayed in the middle") + f"  ({n[key]})"
                k = 3 if key == "teardown" else 5
                d.text((self.x(k) + 14, self.y(c[k + 2]) + (34 if key == "teardown" else -34)), lab, font=font(24, "SemiBold"),
                       fill=col + (int(255 * min(1, u - 3.5)),), anchor="lm")
        im = G.over(im)
        d = ImageDraw.Draw(im, "RGBA")
        a = H2["all"]
        stats = [("Worst season", "low", 17.5), ("Best season, years 3-10", "peak", 21.0), ("Decade average", "mean", 24.5)]
        for i, (lab, key, t0) in enumerate(stats):
            s = fade(t, t0, self.dur + 1, 0.5)
            y = 300 + i * 150
            d.text((1480, y), lab, font=font(26, "Regular"), fill=INK2 + (int(255 * s),), anchor="lm")
            d.text((1480, y + 44), nth(a[key]["teardown"]), font=font(44, "Bold"), fill=ROSE + (int(255 * s),), anchor="lm")
            d.text((1620, y + 44), "vs", font=font(28, "Regular"), fill=MUTED + (int(255 * s),), anchor="lm")
            d.text((1680, y + 44), nth(a[key]["stayed"]), font=font(44, "Bold"), fill=CYAN + (int(255 * s),), anchor="lm")
        d.text((1480, 260), "PERCENTILE IN THE LEAGUE", font=font(18, "SemiBold"), fill=MUTED + (int(255 * fade(t, 17.5, self.dur + 1)),), anchor="lm")
        cap(d, ["Start in the middle of the pack.", f"{n['teardown']} times a team tore it down; {n['stayed']} times it stayed put."], fade(t, 1.0, 7.8))
        cap(d, ["The tear-downs sank much lower..."], fade(t, 8.2, 14.0))
        cap(d, ["...and climbed back no higher."], fade(t, 14.4, 21.4))
        cap(d, ["Ten years of lower lows, for the same peak."], fade(t, 21.8, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Titles(L.Section):
    name, dur = "titles", bars(10)
    SPLIT = 11.5

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        a = H2["all"]
        if t < self.SPLIT + 0.6:
            s = fade(t, 0.2, self.SPLIT + 0.4, 0.5, 0.6)
            L.title(d, "What about championships?", s, "A title within ten years of the fork")
            for lab, v, col, x in (("tore it down", a["title"]["teardown"], ROSE, 700), ("stayed in the middle", a["title"]["stayed"], CYAN, 1220)):
                d.text((x, 330), pct(v), font=font(140, "Bold"), fill=col + (int(255 * s),), anchor="ma")
                d.text((x, 500), lab, font=font(32, "SemiBold"), fill=INK2 + (int(255 * s),), anchor="ma")
            ci = a["title"]["ci95"]
            d.text((W / 2, 590), f"A gap well within chance (95% interval {ci[0] * 100:+.0f} to {ci[1] * 100:+.0f} points)",
                   font=font(28, "Regular"), fill=MUTED + (int(255 * s),), anchor="ma")
            sb = fade(t, 5.5, self.SPLIT + 0.4, 0.5, 0.6)
            nb = H2["nba"]["title"]
            d.text((W / 2, 680), f"NBA only: {pct(nb['teardown'])} against {pct(nb['stayed'])}", font=font(40, "SemiBold"), fill=GOLD + (int(255 * sb),), anchor="ma")
            cap(d, ["A slightly better title shot, and mostly in one league:", "the NBA, where one draft pick can carry a franchise."], fade(t, 6.0, self.SPLIT + 0.2))
        else:
            tt = t - self.SPLIT
            s = fade(tt, 0.3, self.dur + 1, 0.5)
            L.title(d, "Did champions tear it down first?", s, "Champions since 2010 with a bottom-three finish in the decade before, against all teams")
            for i, lg in enumerate(L.LEAGUES):
                c = CH[lg]
                y = 330 + i * 130
                d.text((330, y + 16), L.NAME[lg], font=font(36, "Bold"), fill=INK + (int(255 * s),), anchor="rm")
                g = smooth((tt - 0.8 - 0.3 * i) / 1.0)
                for k, (v, col, lab) in enumerate(((c["tore_first"], GOLD, f"{c['titles']} champions"), (c["all_teams"], CYAN, "all teams"))):
                    x1 = 370 + v * 1300 * g
                    yy = y + k * 38
                    d.rounded_rectangle([370, yy, max(371, x1), yy + 30], 6, fill=col + (int(255 * s),))
                    if g > 0.5:
                        d.text((x1 + 14, yy + 15), f"{pct(v)}  {lab}", font=font(24, "SemiBold" if k == 0 else "Regular"),
                               fill=(INK if k == 0 else INK2) + (int(255 * min(1, (g - 0.5) * 2)),), anchor="lm")
            cap(d, ["Champions weren't more likely to have torn down first,", f"except, perhaps, in hockey (p = {CH['nhl']['p']:.2f})."],
                fade(tt, 3.5, self.dur - self.SPLIT - 0.2))
        return np.asarray(im.convert("RGB"))


class Verdict(L.Section):
    name, dur = "verdict", bars(8)

    def frame(self, t):
        im = L.canvas()
        d = ImageDraw.Draw(im, "RGBA")
        L.title(d, "The verdict", fade(t, 0.2, self.dur + 1))
        cards = [("1   Rich teams can't stomach a rebuild", "Half right", GOLD, 1.0,
                  ["They bottom out less, because they're better.", "Older rosters: a big-market habit.", "A few more first-rounders traded away."]),
                 ("2   Tear-downs climb higher later", "Mostly no", ROSE, 6.5,
                  ["Lower lows, every time.", "The same peaks.", "A better title shot, mostly in the NBA."])]
        for i, (head, verdict, col, t0, lines) in enumerate(cards):
            a = fade(t, t0, self.dur + 1, 0.6)
            x0 = 140 + i * 840
            d.rounded_rectangle([x0, 230, x0 + 800, 800], 28, fill=L.PANEL + (int(220 * a),), outline=col + (int(160 * a),), width=3)
            d.text((x0 + 50, 280), head, font=font(32, "SemiBold"), fill=INK2 + (int(255 * a),), anchor="lm")
            d.text((x0 + 50, 380), verdict, font=font(76, "Bold"), fill=col + (int(255 * a),), anchor="lm")
            for k, ln in enumerate(lines):
                ak = fade(t, t0 + 1.2 + 0.6 * k, self.dur + 1, 0.5)
                d.text((x0 + 50, 500 + k * 80), "·  " + ln, font=font(31, "Regular"), fill=INK + (int(255 * ak),), anchor="lm")
        cap(d, ["Money buys a better team. A tear-down buys a lottery ticket."], fade(t, 13.0, self.dur - 0.2))
        return np.asarray(im.convert("RGB"))


class Outro(L.Section):
    name, dur = "outro", bars(5)

    def frame(self, t):
        im = L.canvas()
        draw_grid(im, 26, a=0.22 * fade(t, 0.0, self.dur + 1, 1.0), labels=False)
        d = ImageDraw.Draw(im, "RGBA")
        a = fade(t, 0.4, self.dur + 1)
        d.text((W / 2, 360), "Win now, or tear it down?", font=font(64, "Bold"), fill=INK + (int(255 * a),), anchor="ma")
        d.text((W / 2, 460), "Be good. The rest is a lottery.", font=font(40, "SemiBold"), fill=GOLD + (int(255 * fade(t, 1.6, self.dur + 1)),), anchor="ma")
        b = fade(t, 3.0, self.dur + 1)
        for k, ln in enumerate(("Seasons, drafts and ages: Sports Reference (NHL, NBA, MLB) and nflverse (NFL), 2000-01 to 2024-25.",
                                "Values: Forbes, via Wikipedia's league lists. Markets: US and Canadian metro populations.",
                                "Methods and code: world_roads/rebuild/README.md")):
            d.text((W / 2, 640 + k * 44), ln, font=font(26, "Regular"), fill=INK2 + (int(255 * b),), anchor="ma")
        return np.asarray(im.convert("RGB"))


SECTIONS = {c.name: c for c in (Hook, Theory, Stories, Richer, Shifted, Winnow, Teardown, Titles, Verdict, Outro)}
ORDER = list(SECTIONS)

if __name__ == "__main__":
    sec = SECTIONS[sys.argv[1]]()
    if len(sys.argv) > 2:
        for s in sys.argv[2:]:
            sec.render(float(s))
    else:
        sec.render()
