"""The best-paid athletes, 1913-2025, every salary in 2025 dollars: a bar-chart race (1920x1080, 30 fps), in two cuts

    python video.py [--career | --yearly] still 1930.0 1998.5 ...   -> out/seasons_still_1930.0.png ... (t in years;
                                                                       < FIRST is the title)
    python video.py [--career | --yearly] render [jobs]             -> out/seasons.mp4, careers.mp4 or race.mp4

The single-season race: the ten best-paid seasons to date, by salary in 2025 dollars, each season its own bar (Michael
Jordan's 1996-97 and 1997-98 are two), staying until ten better ones come along (../media/paid/season_top.csv,
records.py). The career race (--career): the ten biggest careers to date each year, every season's pay in 2025 dollars
added up, retired players keeping their totals until they are passed (../media/paid/career_top.csv, lifetime.py). The
yearly race (--yearly): the ten best-paid of each year alone (../media/paid/salaries.csv).

Bars slide past each other as the ranks change, grow (and in the yearly race shrink) with the pay, slide in from below
the board and out of it. Each bar carries the player (in the single-season race, and the season), his team and league
(in the career race, for a retired player, his league and seasons), the figure in 2025 dollars, and under it the money
as it was paid. The scale follows the
leader. In the single-season and yearly races, while the scale is small, a dashed line marks today's MLB minimum salary ($760,000 in
2025): the stars of the 1920s and 1930s earned, in today's money, about what a rookie on the minimum makes now. In the
career race a career more than a fifth of whose money is estimated (lifetime.py) carries a "≈".

Colour encodes the league (MLB blue, NBA orange, NFL green, NHL grey: three hues validated for colour-blind
separation on this background, the fourth league neutral) and every bar also names its league in text.
"""
import os, sys, csv, json, time, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, translucent text  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "paid")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 7, 13], np.float32)                 # #05070d
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
GOLD = (255, 186, 80)
LEAGUE = {"MLB": (57, 135, 229), "NBA": (217, 89, 38), "NFL": (25, 158, 112), "NHL": (150, 156, 168)}
font = ST.font

FIRST, LAST = 1913, 2025
N = 10                                                # bars on the board
PER_YEAR = 1.4                                        # seconds a year takes
INTRO, OUTRO = 5.6, 11.2                              # whole bars of the score (2.8 s)
MIN_2025 = 760_000                                    # MLB minimum salary, 2025
X0, X1 = 560, 1500                                    # bar baseline and the leader's bar end
MODES = {
    "records": dict(data="season_top.csv", captions="captions.json", film="seasons", title="The best-paid seasons ever",
                    subtitle="Single-season salaries in 2025 dollars, the ten best to date · MLB, NBA, NFL, NHL",
                    intro=f"{FIRST} to {LAST}, every salary in 2025 dollars: a season stays until ten better ones come along"),
    "annual": dict(data="salaries.csv", captions="captions_yearly.json", film="race", title="The best-paid athletes",
                   subtitle="Salaries in 2025 dollars · MLB, NBA, NFL, NHL",
                   intro=f"{FIRST} to {LAST}, every salary in 2025 dollars"),
    "career": dict(data="career_top.csv", captions="captions_career.json", film="careers", title="The biggest career earnings",
                   subtitle="Career salaries to date, in 2025 dollars · MLB, NBA, NFL, NHL",
                   intro=f"{FIRST} to {LAST}, every season's salary in 2025 dollars, added up"),
}
NAME = "records"
MODE = MODES[NAME]
CAREER, RECORDS = False, True
TWO_YEAR = {"NBA", "ABA", "BAA", "NBL", "NHL", "WHA"}            # circuits whose seasons span two calendar years


def configure(mode):
    global NAME, MODE, CAREER, RECORDS
    NAME, MODE, CAREER, RECORDS = mode, MODES[mode], mode == "career", mode == "records"


def season(row):
    """a season as it is named: 1997-98 in basketball and hockey, 1998 in baseball and football"""
    y = int(row["season"])
    return f"{y - 1}-{str(y)[2:]}" if row["circuit"] in TWO_YEAR else str(y)


Y0, PITCH, BAR = 186, 68, 48                          # first row's top, row spacing, bar thickness


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def money(v, short=True):
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e7:
        return f"${v / 1e6:.0f}M" if short and v >= 1e8 else f"${v / 1e6:.1f}M"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e4:
        return f"${v / 1e3:.0f}K" if short else f"${v:,.0f}"
    return f"${v:,.0f}"


class Data:
    def __init__(self):
        self.by = defaultdict(dict)                   # year -> key -> row
        for r in csv.DictReader(open(os.path.join(MEDIA, MODE["data"]))):
            y = int(r["year"])
            if CAREER:                                # a career to date: its total in both dollars
                r["real"], r["nominal"] = float(r["cum_real"]), float(r["cum_nominal"])
                r["est_share"] = float(r["est_share"])
            else:
                r["real"], r["nominal"] = float(r["real_2025"]), float(r["nominal"])
            k = r["league"] + ":" + r["pid"] + (":" + r["season"] if RECORDS else "")   # in the records, each season its own
            if FIRST <= y <= LAST:
                self.by[y][k] = r
        # ties keep one order (by name), so two equal salaries never trade places
        self.order = {y: sorted(d, key=lambda k: (-d[k]["real"], d[k]["player"])) for y, d in self.by.items()}
        self.cov = {int(r["year"]): r for r in csv.DictReader(open(os.path.join(MEDIA, "coverage.csv")))}
        cap = os.path.join(HERE, MODE["captions"])
        self.events = json.load(open(cap)) if os.path.exists(cap) else []
        self.places()

    def top(self, y, n):
        return self.order.get(y, [])[:n]

    def floor(self, y):
        """where a bar with no salary in year y (arriving or gone) is ranked: under the board's 12th"""
        o = self.order.get(y, [])
        return 0.8 * self.by[y][o[min(len(o), N + 2) - 1]]["real"] if o else 0.0

    def places(self):
        """every bar's place on the board at every frame. Between two years the board is sorted by salaries moving in a
        straight line from one year's to the next's, over the middle 60% of the year, so a bar passes another when its
        pay does and each year's board stands exactly as ranked on the year itself; one arriving rises from under the
        board, one leaving sinks under it. The places are then eased in time (a Gaussian of 0.06 years), so a pass takes
        about a third of a second."""
        n = int(round((LAST - FIRST) * PER_YEAR * FPS)) + 1
        self.ts = FIRST + np.arange(n) / FPS / PER_YEAR
        keys = sorted({k for y in self.order for k in self.top(y, N + 3)})
        self.ix = {k: i for i, k in enumerate(keys)}
        self.keys = keys
        R = np.full((len(keys), n), float(OFF), np.float32)
        for j, t in enumerate(self.ts):
            y0 = int(np.floor(t))
            u, y1 = min(1.0, max(0.0, (t - y0 - 0.2) / 0.6)), min(LAST, y0 + 1)
            on = []
            for k in set(self.top(y0, N + 3)) | set(self.top(y1, N + 3)):
                a, b = self.by[y0].get(k), self.by[y1].get(k)
                v0 = a["real"] if a else self.floor(y0)
                v1 = b["real"] if b else self.floor(y1)
                on.append((-(v0 + (v1 - v0) * u), (a or b)["player"], k))
            for r, (_, _, k) in enumerate(sorted(on)):
                R[self.ix[k], j] = min(r, OFF)
        s = 0.06 * PER_YEAR * FPS
        h = int(3 * s)
        ker = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
        ker /= ker.sum()
        self.pos = np.array([np.convolve(np.pad(r, h, mode="edge"), ker, mode="valid") for r in R], np.float32)

    def row(self, k, y, other):
        r = self.by[y].get(k) or self.by[other].get(k)
        for d in range(1, 3):
            r = r or self.by.get(y - d, {}).get(k) or self.by.get(y + d, {}).get(k)
        return r

    def state(self, t):
        """the bars on the board at time t (in years): key -> (place, salary in 2025 dollars, row it is labelled from)"""
        j = int(round((min(max(t, FIRST), LAST) - FIRST) * PER_YEAR * FPS))
        y0 = int(np.floor(t))
        u, y1 = t - y0, min(LAST, y0 + 1)
        out = {}
        for i in np.nonzero(self.pos[:, j] < N - 0.15)[0]:
            k = self.keys[i]
            a, b = self.by[y0].get(k), self.by[y1].get(k)
            if a and b:
                v = a["real"] + (b["real"] - a["real"]) * u
            else:                                     # arriving or leaving: its one salary
                v = (a or b or self.row(k, y0, y1))["real"]
            row = (b if u >= 0.5 and b else a) or b or self.row(k, y0, y1)
            out[k] = (float(self.pos[i, j]), v, row)
        return out


OFF = N + 3                                           # the place of a bar that is not near the board


D = None


def scale_track():
    """the axis maximum at every frame: the leader's value with room for its label, smoothed both ways in time"""
    n = int(round((LAST - FIRST) * PER_YEAR * FPS)) + 1
    ts = FIRST + np.arange(n) / FPS / PER_YEAR
    lead = []
    for t in ts:
        on = [v for r, v, _ in D.state(t).values() if r < N - 0.5]
        lead.append(max(on) if on else 1.0)
    lead = np.array(lead)
    k = int(0.9 * FPS)
    pad = np.pad(lead, k, mode="edge")
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.2)) ** 2)
    ker /= ker.sum()
    sm = np.convolve(pad, ker, mode="valid")
    return np.maximum(sm, lead * 0.97)                # never let the leader run off the scale


SCALE = None


def scale_at(t):
    i = int(round((min(max(t, FIRST), LAST) - FIRST) * PER_YEAR * FPS))
    return SCALE[min(i, len(SCALE) - 1)]


_BG = None


def canvas():
    global _BG
    if _BG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = 1 - 0.35 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.8)) ** 2)
        _BG = (BG[None, None] * np.clip(v, 0.5, 1)[..., None]).astype(np.uint8)
    return Image.fromarray(_BG.copy())


def ticks(d, xmax, a=1.0):
    """gridlines and labels for the scale, nice steps fading in and out as the scale moves"""
    steps = [m * 10 ** e for e in range(3, 9) for m in (1, 2, 2.5, 5)]
    seen = {}
    for s in steps:
        dens = xmax / s
        al = smooth((dens - 2.2) / 1.2) * (1 - smooth((dens - 6.5) / 2.0))
        if al <= 0.02:
            continue
        v = s
        while v <= xmax * 1.001:
            seen[v] = max(seen.get(v, 0), al)
            v += s
    px = {v: X0 + (X1 - X0) * v / xmax for v in seen}
    al0 = dict(seen)
    for v in seen:                                    # a tick crowding a stronger one gives way to it
        near = [abs(px[v] - px[u]) for u in al0 if al0[u] > al0[v]]
        seen[v] *= smooth((min(near, default=1e9) - 75) / 45)
    for v, al in seen.items():
        x = px[v]
        if x > X1 + 150:
            continue
        al *= a
        d.line([(x, Y0 - 14), (x, Y0 + N * PITCH - 8)], fill=(40, 52, 72, int(140 * al)), width=1)
        lab = f"${v / 1e6:g}M" if v >= 1e6 else f"${v / 1e3:g}K"
        d.text((x, Y0 - 22), lab, font=font(20, "Regular"), fill=MUTED + (int(255 * al),), anchor="mb")


def board(im, t, a=1.0, hold=False):
    d = ImageDraw.Draw(im, "RGBA")
    xmax = scale_at(t)
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    ticks(d, xmax, a)
    st = D.state(t)
    # today's minimum salary, while the scale is small enough for it to mean something; its dashes break around labels
    xm = xs(MIN_2025)
    am = a * smooth((xm - X0 - 40) / 60) * (1 - smooth((t - 1975) / 4)) * (not CAREER)
    if am > 0.01:
        gaps = []
        for r, v, row in st.values():
            x1 = max(X0 + 6, xs(v)) + 8
            w = max(d.textlength(money(v, short=False), font=font(28, "Bold")), d.textlength(under(row), font=font(19, "Regular")))
            if x1 <= xm <= x1 + w + 12:
                gaps.append((Y0 + r * PITCH - 4, Y0 + r * PITCH + BAR + 4))
        for y in range(Y0 - 6, Y0 + N * PITCH - 8, 14):
            if not any(g0 < y + 7 and y < g1 for g0, g1 in gaps):
                d.line([(xm, y), (xm, y + 7)], fill=GOLD + (int(200 * am),), width=2)
        d.text((xm, Y0 + N * PITCH + 18), "today's MLB minimum, $760K", font=font(20, "Regular"), fill=GOLD + (int(230 * am),), anchor="ms")
    # longest first, so a shorter bar passing a longer one stays in view on top of it; while two bars overlap, the
    # lower-paid one's labels give way
    for k, (r, v, row) in sorted(st.items(), key=lambda kv: (-kv[1][1], kv[1][0])):
        al = a * smooth((N - 0.15 - r) / 0.7)
        if al <= 0.01:
            continue
        near = min((abs(r - r2) for k2, (r2, v2, _) in st.items() if k2 != k and (v2, -r2) > (v, -r)), default=9)
        tl = al * smooth((near - 0.3) / 0.4)
        y = Y0 + r * PITCH
        col = LEAGUE[row["league"]]
        x1 = max(X0 + 6, xs(v))
        d.rounded_rectangle([X0, y, x1, y + BAR], 4, fill=col + (int(255 * al),))
        d.rectangle([X0, y, min(x1, X0 + 6), y + BAR], fill=col + (int(255 * al),))
        if r < 0.5 and not hold:                     # the leader
            d.rounded_rectangle([X0 - 1, y - 1, x1 + 1, y + BAR + 1], 5, outline=GOLD + (int(150 * al * (1 - r * 2)),), width=2)
        if tl <= 0.01:
            continue
        nx = X0 - 18
        if RECORDS:                                  # the season, after the name
            d.text((nx, y + BAR / 2 - 1), season(row), font=font(23, "Regular"), fill=INK2 + (int(255 * tl),), anchor="rs")
            nx -= d.textlength(season(row), font=font(23, "Regular")) + 10
        d.text((nx, y + BAR / 2 - 1), row["player"], font=font(29, "SemiBold"), fill=INK + (int(255 * tl),), anchor="rs")
        d.text((X0 - 18, y + BAR / 2 + 15), who(row), font=font(19, "Regular"), fill=INK2 + (int(235 * tl),), anchor="rm")
        approx = "≈" if CAREER and row["est_share"] > 0.2 else ""
        d.text((x1 + 14, y + BAR / 2 - 9), approx + money(v, short=False), font=font(28, "Bold"), fill=INK + (int(255 * tl),), anchor="lm")
        d.text((x1 + 14, y + BAR / 2 + 15), under(row), font=font(19, "Regular"), fill=MUTED + (int(235 * tl),), anchor="lm")
    return d


def who(row):
    """under the name: team and league; in the career race, a retired player's league and seasons"""
    if not CAREER:
        return f"{row['team']} · {row['circuit']}"
    if row["active"] == "1":
        return f"{row['team']} · {row['league']} · since {row['first']}"
    return f"{row['league']} · {row['first']}–{row['last']}"


def under(row):
    """under the figure: the money as it was paid"""
    if RECORDS:
        return f"{money(row['nominal'], short=False)} then"
    if not CAREER:
        return f"{money(row['nominal'], short=False)} in {row['year']}"
    return f"{row['seasons']} seasons · {money(row['nominal'], short=False)} as paid"


def header(d, a=1.0):
    d.text((90, 62), MODE["title"], font=font(46, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((90, 104), MODE["subtitle"], font=font(25, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
    x = W - 90
    for lg in ("NHL", "NFL", "NBA", "MLB"):
        w = d.textlength(lg, font=font(22, "SemiBold"))
        d.text((x, 62), lg, font=font(22, "SemiBold"), fill=INK + (int(255 * a),), anchor="rs")
        d.rounded_rectangle([x - w - 30, 44, x - w - 10, 64], 4, fill=LEAGUE[lg] + (int(255 * a),))
        x -= w + 58


def year_label(d, t, a=1.0):
    y = int(np.floor(min(max(t, FIRST), LAST) + 0.5))
    d.text((W - 90, H - 78), str(y), font=font(132, "Bold"), fill=INK + (int(235 * a),), anchor="rs")
    f = float(D.cov[y]["to_2025"])
    d.text((W - 92, H - 40), f"$1 in {y} = ${f:,.2f} in 2025", font=font(24, "Regular"), fill=INK2 + (int(255 * a),), anchor="rs")


def caption(d, t, a=1.0):
    for e in D.events:
        t0, t1 = e["from"], e["to"]
        al = a * smooth((t - t0) / 0.35) * (1 - smooth((t - t1) / 0.35))
        if al <= 0.01:
            continue
        lines = e["text"] if isinstance(e["text"], list) else [e["text"]]
        y = H - 96
        for i, line in enumerate(lines):
            d.text((90, y + i * 38), line, font=font(30 if i == 0 else 24, "SemiBold" if i == 0 else "Regular"),
                   fill=(INK if i == 0 else INK2) + (int(255 * al),), anchor="ls")


def frame(T):
    """T: seconds from the start of the film"""
    im = canvas()
    if T < INTRO:
        d = ImageDraw.Draw(im, "RGBA")
        a = smooth(T / 0.8) * (1 - smooth((T - INTRO + 1.0) / 0.8))
        d.text((W / 2, 420), MODE["title"], font=font(72, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((W / 2, 500), MODE["intro"], font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((W / 2, 600), "MLB · NBA · NFL · NHL", font=font(26, "SemiBold"), fill=MUTED + (int(255 * a),), anchor="mm")
        return np.asarray(im)
    t = FIRST + (T - INTRO) / PER_YEAR
    hold = t >= LAST
    t = min(t, LAST)
    a = smooth((T - INTRO) / 0.8)
    end = total()
    dim = 1 - 0.94 * smooth((T - (end - OUTRO + 1.6)) / 0.8)      # the board steps back for the summary
    d = board(im, t, a * dim, hold)
    header(d, a)
    year_label(d, t, a * (dim - 0.06) / 0.94)          # gone under the sources line
    caption(d, FIRST + (T - INTRO) / PER_YEAR, a)
    if T > end - OUTRO + 1.6:
        summary(d, T - (end - OUTRO + 1.6))
    return np.asarray(im)


def summary(d, s):
    """over the final standings: the first star against the last, in the same money"""
    a = smooth(s / 0.8) * (1 - smooth((s - (OUTRO - 2.6)) / 0.8))
    if a <= 0.01:
        return
    then, now = SUMMARY["then"], SUMMARY["now"]
    cx = W / 2
    head = {"career": "Career earnings, in 2025 dollars", "records": "The best-paid seasons, in 2025 dollars"}
    d.text((cx, 300), head.get(NAME, "In 2025 dollars"), font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for i, (r, x) in enumerate(((then, cx - 380), (now, cx + 380))):
        if CAREER:
            sub, low = who(r), under(r)
        elif RECORDS:
            sub, low = f"{r['team']} · {r['circuit']} · {season(r)}", f"{money(r['nominal'], short=False)} in {r['season']} dollars"
        else:
            sub, low = f"{r['team']} · {r['circuit']} · {r['year']}", f"{money(r['nominal'], short=False)} in {r['year']} dollars"
        d.text((x, 400), r["player"], font=font(44, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 452), sub, font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((x, 560), money(r["real"], short=False), font=font(86, "Bold"), fill=(GOLD if i else INK) + (int(255 * a),), anchor="mm")
        d.text((x, 640), low, font=font(26, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")
    d.text((cx, 560), "→", font=font(64, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    d.text((cx, 740), f"{now['real'] / then['real']:.0f} times as much", font=font(40, "SemiBold"), fill=INK + (int(255 * a),), anchor="mm")
    if SUMMARY.get("line"):
        d.text((cx, 830), SUMMARY["line"], font=font(28, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    lines = SOURCES[NAME]
    for i, line in enumerate(lines):
        d.text((cx, 996 + 28 * i - 14 * (len(lines) - 2)), line, font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")


SOURCES = {"annual": ("Salaries: Baseball Reference (Michael Haupert's research of the Hall of Fame's contract cards), Lahman database,",
                      "Basketball Reference, OverTheCap, CapWages, published contracts · CPI-U: Federal Reserve Bank of Minneapolis"),
           "records": ("Salaries: Baseball Reference (Michael Haupert's research of the Hall of Fame's contract cards), Lahman database,",
                       "Basketball Reference, OverTheCap, CapWages, published contracts · CPI-U: Federal Reserve Bank of Minneapolis",
                       "Baseball seasons before 1913 count too, converted with the Minneapolis Fed's CPI estimates"),
           "career": ("Salaries: Baseball Reference (Michael Haupert's research of the Hall of Fame's contract cards), Lahman database,",
                      "Basketball Reference, OverTheCap (cash paid), CapWages, published contracts · CPI-U: Federal Reserve Bank of Minneapolis",
                      "Careers count seasons before 1913 too; a season between two documented salaries is estimated (≈: over a fifth estimated)")}
SUMMARY = {}


def total():
    return INTRO + (LAST - FIRST) * PER_YEAR + OUTRO


def setup():
    global D, SCALE
    D = Data()
    SCALE = scale_track()
    rows = [r for d in D.by.values() for r in d.values()]
    ruth = [r for r in rows if r["player"] == "Babe Ruth"]
    SUMMARY["now"] = D.by[LAST][D.order[LAST][0]]
    # Babe Ruth: his best year, or his whole career
    SUMMARY["then"] = max(ruth, key=lambda r: (r["real"], int(r["year"]))) if ruth else D.by[FIRST][D.order[FIRST][0]]
    if CAREER:                                        # the longest time at No. 1
        lead = [(y, D.order[y][0]) for y in range(FIRST, LAST + 1) if D.order.get(y)]
        runs, start = [], 0
        for i in range(1, len(lead) + 1):
            if i == len(lead) or lead[i][1] != lead[start][1]:
                runs.append((i - start, lead[start][0], lead[i - 1][0], lead[start][1]))
                start = i
        n, y0, y1, k = max(runs)
        SUMMARY["line"] = f"Longest at No. 1: {D.by[y1][k]['player']}, {n} years ({y0}–{y1})"
    elif RECORDS:                                     # Ruth's best season against the record; the best of the last year
        last = [r for r in D.by[LAST].values() if int(r["season"]) == LAST]
        best = max(last, key=lambda r: r["real"]) if last else None
        if best is not None and best is not SUMMARY["now"]:
            SUMMARY["line"] = (f"The best of {LAST}: {best['player']}, {best['team']} · {best['circuit']}, "
                               f"{money(best['real'], short=False)}")
    else:                                             # the record, if the last year's leader doesn't hold it
        rec = max(rows, key=lambda r: r["real"])
        if rec is not SUMMARY["now"]:
            SUMMARY["line"] = (f"The record: {rec['player']}, {rec['team']} · {rec['circuit']}, {rec['year']}: "
                               f"{money(rec['real'], short=False)} ({money(rec['nominal'], short=False)} then)")


def render_range(c0, c1, part, mode):
    configure(mode)
    setup()
    tmp = part + ".tmp.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-tune", "animation",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(c0, c1):
        p.stdin.write(np.ascontiguousarray(frame(i / FPS)).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, part)


def render(jobs=4):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    n = int(round(total() * FPS))
    CH = 300
    film = MODE["film"]
    mode = NAME
    parts = [(c0, min(n, c0 + CH), os.path.join(OUT, "chunks", f"{film}_{c0:05d}.mp4")) for c0 in range(0, n, CH)]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(render_range, *p, mode) for p in parts if not os.path.exists(p[2])]
        for f in futs:
            f.result()
    print(f"{n} frames in {time.time() - t0:.0f} s", flush=True)
    lst = os.path.join(OUT, "chunks", f"{film}.txt")
    open(lst, "w").write("".join(f"file '{p[2]}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(OUT, f"{film}.mp4")], check=True)
    for p in parts:
        os.remove(p[2])
    print("wrote", os.path.join(OUT, f"{film}.mp4"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    if args and args[0] in ("--career", "--yearly"):
        configure({"--career": "career", "--yearly": "annual"}[args[0]])
        args = args[1:]
    if args[0] == "still":
        setup()
        for a in args[1:]:
            t = float(a)
            T = INTRO + (t - FIRST) * PER_YEAR if t >= FIRST else t
            Image.fromarray(frame(T)).save(os.path.join(OUT, f"{MODE['film']}_still_{t:06.1f}.png"))
            print("still", t)
    else:
        render(int(args[1]) if len(args) > 1 else 4)
