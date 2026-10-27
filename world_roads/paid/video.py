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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, translucent text  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "paid")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 7, 13], np.float32)                 # #05070d
BG_RGB = (5, 7, 13)
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
GOLD = (255, 186, 80)
LEAGUE = {"MLB": (57, 135, 229), "NBA": (217, 89, 38), "NFL": (25, 158, 112), "NHL": (150, 156, 168)}
font = ST.font
_DF = {}


def dfont(size, weight="Bold"):
    """Inter Display, cut for headline sizes"""
    if (size, weight) not in _DF:
        _DF[(size, weight)] = ImageFont.truetype(os.path.join(ST.FD, f"InterDisplay-{weight}.otf"), size,
                                                 layout_engine=ImageFont.Layout.RAQM)
    return _DF[(size, weight)]


TNUM = ["tnum"]                                       # figures of one width, so changing numbers don't jitter

FIRST, LAST = 1913, 2025
N = 10                                                # bars on the board
PER_YEAR = 1.4                                        # seconds a year takes (unless the race paces itself)
HALF = 0.35                                           # half a beat of the score
PACE = [1, 2, 3, 4, 5, 6, 7, 7, 8]                    # half-beats a year takes in a paced race, by how much its board changes
INTRO, OUTRO = 5.6, 11.2                              # whole bars of the score (2.8 s)
MIN_2025 = 760_000                                    # MLB minimum salary, 2025
X0, X1 = 560, 1500                                    # bar baseline and the leader's bar end
MODES = {
    "records": dict(data="season_top.csv", captions="captions.json", film="seasons", title="The best-paid seasons ever", paced=True,
                    layout="panel",
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


# two layouts: "classic" (the board fills the frame, the year and captions below it) and "panel" (a slimmer board, the
# year at the top right, captions in a band, league revenue in a panel along the bottom)
GEOM = {"classic": dict(Y0=186, PITCH=68, BAR=48, NAME=29, SEASON=23, SUB=19, VAL=28, UNDER=19, TICK=20),
        "panel": dict(Y0=178, PITCH=52, BAR=38, NAME=25, SEASON=20, SUB=16, VAL=25, UNDER=16, TICK=18)}
G = GEOM["panel"]
PANEL = True


def configure(mode):
    global NAME, MODE, CAREER, RECORDS, G, PANEL, Y0, PITCH, BAR
    NAME, MODE, CAREER, RECORDS = mode, MODES[mode], mode == "career", mode == "records"
    PANEL = MODE.get("layout") == "panel"
    G = GEOM["panel" if PANEL else "classic"]
    Y0, PITCH, BAR = G["Y0"], G["PITCH"], G["BAR"]


def season(row):
    """a season as it is named: 1997-98 in basketball and hockey, 1998 in baseball and football"""
    y = int(row["season"])
    return f"{y - 1}-{str(y)[2:]}" if row["circuit"] in TWO_YEAR else str(y)


Y0, PITCH, BAR = G["Y0"], G["PITCH"], G["BAR"]        # first row's top, row spacing, bar thickness (configure sets them)


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
        self.pace()
        self.windows()
        self.places()
        self.leads = [y for y in range(FIRST + 1, LAST + 1) if self.top(y, 1) != self.top(y - 1, 1)]
        self.rev = load_revenue() if PANEL else {}

    def pulse(self, T):
        """0 to 1: the leader's halo flares as a new No. 1 reaches the top, and fades over a second"""
        p = 0.0
        for y in self.leads:
            dt = T - (INTRO + self.secs(y - 0.5))
            if -0.3 < dt < 3:
                p = max(p, smooth((dt + 0.3) / 0.4) * np.exp(-max(0.0, dt - 0.1) / 0.9))
        return p

    def pace(self):
        """the seconds each year takes, from the year before to it: PER_YEAR throughout, or, in a race that paces itself,
        quick where the board stands still and slower where it churns. A year's change is the number of entries that
        join the top ten, two more for a new No. 1, one more for a caption; it buys whole half-beats of the score (PACE),
        so the years still turn on the music: 0.35 s for a year that changes nothing, up to 2.8 s for the busiest."""
        capt = {int(round(e["from"] + 0.5)) for e in self.events}
        steps = [0.0]
        for y in range(FIRST + 1, LAST + 1):
            if MODE.get("paced"):
                before, now = self.top(y - 1, N), self.top(y, N)
                change = len(set(now) - set(before)) + 2 * (now[:1] != before[:1]) + (y in capt)
                steps.append(HALF * PACE[min(change, len(PACE) - 1)])
            else:
                steps.append(PER_YEAR)
        self.knot_y = np.arange(FIRST, LAST + 1, dtype=float)
        self.knot_s = np.cumsum(steps)
        self.race = float(self.knot_s[-1])            # seconds from the first year's board to the last's

    def secs(self, t):
        """seconds into the race at time t (in years)"""
        return float(np.interp(t, self.knot_y, self.knot_s))

    def year_of(self, s):
        return np.interp(s, self.knot_s, self.knot_y)

    def windows(self):
        """each caption's time on screen, in seconds: from its moment, for as long as captions.py gave it at the steady
        pace (3.6 s); one due sooner makes room down to 2.1 s, then waits"""
        last = None
        for e in self.events:
            e["T0"] = INTRO + self.secs(e["from"])
            e["T1"] = e["T0"] + (e["to"] - e["from"]) * PER_YEAR
            if last and e["T0"] < last["T1"] + 0.42:
                last["T1"] = max(last["T0"] + 2.1, e["T0"] - 0.42)
                shift = max(0.0, last["T1"] + 0.42 - e["T0"])
                e["T0"], e["T1"] = e["T0"] + shift, e["T1"] + shift
            last = e

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
        board, one leaving sinks under it. The places are then eased in time (a Gaussian of 0.084 s, on screen), so a
        pass takes about a third of a second."""
        n = int(round(self.race * FPS)) + 1
        self.ts = self.year_of(np.arange(n) / FPS)    # each frame's time, in years
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
        j = min(len(self.ts) - 1, int(round(self.secs(t) * FPS)))
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
REV_FILE = os.path.join(MEDIA, "revenue.csv")
RX0, RX1 = 168, 1640                                  # the revenue panel's plot: left and right
RY0, RY1 = 868, 1004                                  # its top and its baseline


GAP = 3                                               # a stretch of more years than this between documented ones is drawn dotted


def load_revenue():
    """league -> (years, revenue in 2025 dollars, whether the step into each year crosses a long undocumented stretch),
    from revenue.py's table"""
    out = defaultdict(list)
    if not os.path.exists(REV_FILE):
        return {}
    for r in csv.DictReader(open(REV_FILE)):
        out[r["league"]].append((int(r["year"]), float(r["real_2025"]), r.get("documented", "1") == "1"))
    rev = {}
    for lg, v in out.items():
        ys = [y for y, _, _ in v]
        docs = [y for y, _, d in v if d]
        dotted = []
        for y in ys:
            a = max([x for x in docs if x < y], default=y)
            b = min([x for x in docs if x >= y], default=y)
            dotted.append(b - a > GAP)
        rev[lg] = (np.array(ys, float), np.array([x for _, x, _ in v]), dotted)
    return rev


def rev_at(lg, t):
    ys, vs = D.rev[lg][:2]
    return float(np.interp(t, ys, vs)) if ys[0] <= t else None


def rev_money(v):
    return f"${v / 1e9:.1f}B" if v >= 1e9 else f"${v / 1e6:.0f}M" if v >= 1e6 else f"${v / 1e3:.0f}K"


def rev_scale_track():
    """the revenue axis's maximum at every frame: the largest revenue drawn so far, eased in time"""
    top, m = [], 1e6
    for t in D.ts:
        for lg in D.rev:
            v = rev_at(lg, t)
            if v is not None:
                m = max(m, v)
        top.append(m)
    top = np.array(top)
    k = int(0.9 * FPS)
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.2)) ** 2)
    ker /= ker.sum()
    return np.maximum(np.convolve(np.pad(top, k, mode="edge"), ker, mode="valid"), top) * 1.1


REVSCALE = None


def revenue(im, t, a):
    """league revenue, in 2025 dollars, drawn year by year as the race runs: one line per league (dotted across a stretch
    of more than three years with no figure), its head labelled; a
    gold dot on a league's line in each year one of its players set the salary record"""
    if not D.rev or a <= 0.01:
        return
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([70, 814, W - 70, 1052], 14, fill=(255, 255, 255, int(7 * a)), outline=(255, 255, 255, int(18 * a)))
    d.text((96, 846), "League revenue, in 2025 dollars", font=font(19, "SemiBold"), fill=INK2 + (int(255 * a),), anchor="ls")
    legend(d, W - 96, 846, a, size=17, swatch="line")
    ymax = REVSCALE[min(len(REVSCALE) - 1, int(round(D.secs(t) * FPS)))]
    xs = lambda y: RX0 + (RX1 - RX0) * (y - FIRST) / (LAST - FIRST)
    ys = lambda v: RY1 - (RY1 - RY0) * v / ymax
    start = min(D.rev[lg][0][0] for lg in D.rev)       # the first year with a figure on record
    before = 1 - smooth((t - start + 0.6) / 0.6)       # until then, the axis of years and a note
    if before > 0.01:
        d.text(((RX0 + RX1) / 2, (RY0 + RY1) / 2), f"League revenue figures on record begin in {int(start)}", font=font(19, "Regular"),
               fill=MUTED + (int(200 * a * before),), anchor="mm")
    # the revenue axis: nice steps, fading in and out as it grows
    seen = {}
    for st in [m * 10 ** e for e in range(5, 12) for m in (1, 2, 5)]:
        dens = ymax / st
        al = smooth((dens - 1.6) / 0.8) * (1 - smooth((dens - 4.2) / 1.2))
        v = st
        while al > 0.02 and v <= ymax:
            seen[v] = max(seen.get(v, 0), al)
            v += st
    al0 = dict(seen)
    for v in seen:                                    # a tick crowding a stronger one gives way to it
        near = [abs(ys(v) - ys(u)) for u in al0 if al0[u] > al0[v]]
        seen[v] *= smooth((min(near, default=1e9) - 22) / 14)
    for v, al in seen.items():
        y, al = ys(v), al * (1 - before)
        d.line([(RX0, y), (RX1, y)], fill=(40, 52, 72, int(110 * al * a)), width=1)
        d.text((RX0 - 12, y), rev_money(v).replace(".0B", "B"), font=font(15, "Regular"), fill=MUTED + (int(255 * al * a),),
               anchor="rm", features=TNUM)
    d.line([(RX0, RY1), (RX1, RY1)], fill=(70, 84, 106, int(200 * a)), width=1)
    for y in range(1920, LAST + 1, 10):
        d.line([(xs(y), RY1), (xs(y), RY1 + 5)], fill=(70, 84, 106, int(200 * a)), width=1)
        d.text((xs(y), RY1 + 22), str(y), font=font(15, "Regular"), fill=MUTED + (int(255 * a),), anchor="ms", features=TNUM)
    # the lines, drawn at twice the size and scaled down so they are smooth
    S, ox, oy = 2, RX0 - 20, RY0 - 24
    layer = Image.new("RGBA", ((RX1 - ox + 30) * S, (RY1 - oy + 12) * S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    heads = []
    for lg in ("MLB", "NHL", "NBA", "NFL"):
        if lg not in D.rev or D.rev[lg][0][0] > t:
            continue
        yrs, vals, dotted = D.rev[lg]
        hv = rev_at(lg, t)
        pts = [(xs(y), ys(v), dot) for y, v, dot in zip(yrs, vals, dotted) if y <= t]
        if t > yrs[len(pts) - 1]:                     # the head, between two years
            pts.append((xs(t), ys(hv), dotted[min(len(pts), len(dotted) - 1)]))
        pts = [((x - ox) * S, (y - oy) * S, dot) for x, y, dot in pts]
        run = [pts[0][:2]]                            # solid runs as lines; a long undocumented stretch as faint dots
        for (x0, y0, _), (x1, y1, dot) in zip(pts, pts[1:]):
            if dot:
                if len(run) > 1:
                    ld.line(run, fill=LEAGUE[lg] + (255,), width=3 * S, joint="curve")
                n = max(1, int(np.hypot(x1 - x0, y1 - y0) / (9 * S)))
                for i in range(n):
                    px, py = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
                    ld.ellipse([px - 1.6 * S, py - 1.6 * S, px + 1.6 * S, py + 1.6 * S], fill=LEAGUE[lg] + (170,))
                run = [(x1, y1)]
            else:
                run.append((x1, y1))
        if len(run) > 1:
            ld.line(run, fill=LEAGUE[lg] + (255,), width=3 * S, joint="curve")
        heads.append([ys(hv), lg, xs(t), hv])
    layer = layer.resize((layer.width // S, layer.height // S), Image.LANCZOS)
    if a < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    im.paste(layer, (ox, oy), layer)
    # the record markers
    for y in D.leads:
        lg = D.by[y][D.order[y][0]]["league"]
        if lg in D.rev and y - 0.5 <= t and D.rev[lg][0][0] <= y:
            al = a * smooth((t - (y - 0.5)) / 0.4)
            cx, cy = xs(y), ys(rev_at(lg, y))
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=GOLD + (int(240 * al),), outline=BG_RGB + (int(255 * al),), width=1)
    # the heads, their labels kept apart
    heads.sort()
    for i in range(1, len(heads)):
        heads[i][0] = max(heads[i][0], heads[i - 1][0] + 18)
    for i in range(len(heads) - 2, -1, -1):
        heads[i][0] = min(heads[i][0], heads[i + 1][0] - 18)
    for ly, lg, hx, hv in heads:
        hy = ys(hv)
        d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=LEAGUE[lg] + (int(255 * a),), outline=BG_RGB + (int(255 * a),), width=2)
        d.text((hx + 12, ly), f"{lg} {rev_money(hv)}", font=font(15, "SemiBold"), fill=INK + (int(255 * a),), anchor="lm", features=TNUM)


def scale_track():
    """the axis maximum at every frame: the leader's value with room for its label, smoothed both ways in time"""
    lead = []
    for t in D.ts:
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
    i = int(round(D.secs(t) * FPS))
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
    steps = [m * 10 ** e for e in range(3, 9) for m in (1, 2, 5)]         # 1-2-5: one family of ticks at a time
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
        d.text((x, Y0 - 22), lab, font=font(G["TICK"], "Regular"), fill=MUTED + (int(255 * al),), anchor="mb")


_GRAD = {}


def bar_fill(col, w, h):
    """a bar's face: from 80% of its colour at the baseline to all of it at the end, a faint light along the top edge"""
    if (col, h) not in _GRAD:
        g = np.linspace(0.8, 1.0, 256)[None, :, None] * np.array(col, np.float32)[None, None, :]
        strip = np.repeat(g, h, axis=0)
        strip[:2] = np.minimum(255, strip[:2] * 1.16)
        strip[-1:] *= 0.86
        _GRAD[(col, h)] = Image.fromarray(strip.astype(np.uint8))
    return _GRAD[(col, h)].resize((max(1, int(w)), h), Image.BILINEAR)


def draw_bar(im, x0, y, x1, col, al):
    w, yi = int(round(x1 - x0)), int(round(y))
    mask = Image.new("L", (w, BAR), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w - 1, BAR - 1], 5, fill=int(255 * al))
    md.rectangle([0, 0, min(w - 1, 6), BAR - 1], fill=int(255 * al))     # the baseline end stays square
    im.paste(bar_fill(col, w, BAR), (int(x0), yi), mask)


def glow(im, x0, y, x1, col, strength):
    """a soft halo of colour around a bar"""
    if strength <= 0.01:
        return
    pad = 28
    w, h = int(x1 - x0) + 2 * pad, BAR + 2 * pad
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).rounded_rectangle([pad, pad + 2, w - pad, h - pad - 2], 8, fill=int(255 * min(1.0, strength)))
    im.paste(col, (int(x0) - pad, int(round(y)) - pad), layer.filter(ImageFilter.GaussianBlur(13)))


def board(im, t, a=1.0, hold=False, T=None):
    d = ImageDraw.Draw(im, "RGBA")
    xmax = scale_at(t)
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    ticks(d, xmax, a)
    st = D.state(t)
    # the places, numbered
    for i in range(N):
        d.text((118, Y0 + i * PITCH + BAR / 2), str(i + 1), font=dfont(G["NAME"] - 4, "Medium"), fill=MUTED + (int(150 * a),),
               anchor="rm", features=TNUM)
    # today's minimum salary, while the scale is small enough for it to mean something; its dashes break around labels
    xm = xs(MIN_2025)
    am = a * smooth((xm - X0 - 40) / 60) * (1 - smooth((t - 1975) / 4)) * (not CAREER)
    if am > 0.01:
        gaps = []
        for r, v, row in st.values():
            x1 = max(X0 + 6, xs(v)) + 8
            w = max(d.textlength(money(v, short=False), font=font(G["VAL"], "Bold")), d.textlength(under(row), font=font(G["UNDER"], "Regular")))
            if x1 <= xm <= x1 + w + 12:
                gaps.append((Y0 + r * PITCH - 4, Y0 + r * PITCH + BAR + 4))
        for y in range(Y0 - 6, Y0 + N * PITCH - 8, 14):
            if not any(g0 < y + 7 and y < g1 for g0, g1 in gaps):
                d.line([(xm, y), (xm, y + 7)], fill=GOLD + (int(200 * am),), width=2)
        d.text((xm, Y0 + N * PITCH + 16), "today's MLB minimum, $760K", font=font(G["TICK"], "Regular"), fill=GOLD + (int(230 * am),), anchor="ms")
    # the leader's halo, brighter for a moment when a new No. 1 arrives
    pulse = D.pulse(T) if T is not None else 0.0
    for k, (r, v, row) in st.items():
        if r < 0.5 and not hold:
            al = a * (1 - r * 2)
            glow(im, X0, Y0 + r * PITCH, max(X0 + 6, xs(v)), LEAGUE[row["league"]], al * 0.42)
            glow(im, X0, Y0 + r * PITCH, max(X0 + 6, xs(v)), GOLD, al * 0.75 * pulse)
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
        draw_bar(im, X0, y, x1, col, al)
        if r < 0.5 and not hold:                     # the leader
            d.rounded_rectangle([X0 - 1, y - 1, x1 + 1, y + BAR + 1], 6, outline=GOLD + (int((150 + 100 * pulse) * al * (1 - r * 2)),), width=2)
        if tl <= 0.01:
            continue
        nx = X0 - 18
        if RECORDS:                                  # the season, after the name
            d.text((nx, y + BAR / 2 - 1), season(row), font=font(G["SEASON"], "Regular"), fill=INK2 + (int(255 * tl),), anchor="rs")
            nx -= d.textlength(season(row), font=font(G["SEASON"], "Regular")) + 9
        d.text((nx, y + BAR / 2 - 1), row["player"], font=font(G["NAME"], "SemiBold"), fill=INK + (int(255 * tl),), anchor="rs")
        d.text((X0 - 18, y + BAR / 2 + G["SUB"] - 3), who(row), font=font(G["SUB"], "Regular"), fill=INK2 + (int(235 * tl),), anchor="rm")
        approx = "≈" if CAREER and row["est_share"] > 0.2 else ""
        d.text((x1 + 14, y + BAR / 2 - G["VAL"] / 3), approx + money(v, short=False), font=font(G["VAL"], "Bold"), fill=INK + (int(255 * tl),),
               anchor="lm", features=TNUM)
        d.text((x1 + 14, y + BAR / 2 + G["UNDER"] - 3), under(row), font=font(G["UNDER"], "Regular"), fill=MUTED + (int(235 * tl),),
               anchor="lm", features=TNUM)
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


def legend(d, x, y, a, size=22, swatch="square"):
    """the leagues' colours, right-aligned at x"""
    for lg in ("NHL", "NFL", "NBA", "MLB"):
        w = d.textlength(lg, font=font(size, "SemiBold"))
        d.text((x, y), lg, font=font(size, "SemiBold"), fill=INK + (int(255 * a),), anchor="rs")
        if swatch == "square":
            d.rounded_rectangle([x - w - size - 8, y - size + 4, x - w - 10, y + 2 - 2], 4, fill=LEAGUE[lg] + (int(255 * a),))
        else:                                         # a stroke, for lines
            d.rounded_rectangle([x - w - 34, y - size / 2 - 1, x - w - 10, y - size / 2 + 3], 2, fill=LEAGUE[lg] + (int(255 * a),))
        x -= w + (58 if swatch == "square" else 56)


def header(d, a=1.0):
    if PANEL:
        d.rounded_rectangle([64, 28, 70, 108], 3, fill=GOLD + (int(255 * a),))
        d.text((90, 66), MODE["title"], font=dfont(48, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
        d.text((91, 102), MODE["subtitle"], font=font(22, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
        return
    d.text((90, 62), MODE["title"], font=font(46, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((90, 104), MODE["subtitle"], font=font(25, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
    legend(d, W - 90, 62, a)


def year_label(d, t, a=1.0):
    y = int(np.floor(min(max(t, FIRST), LAST) + 0.5))
    f = float(D.cov[y]["to_2025"])
    if not PANEL:
        d.text((W - 90, H - 78), str(y), font=font(132, "Bold"), fill=INK + (int(235 * a),), anchor="rs")
        d.text((W - 92, H - 40), f"$1 in {y} = ${f:,.2f} in 2025", font=font(24, "Regular"), fill=INK2 + (int(255 * a),), anchor="rs")
        return
    # the year at the top right, the new one rolling up into place as the old one leaves
    p = smooth((D.secs(t) - D.secs(y - 0.5)) / 0.24) if y > FIRST else 1.0
    big = dfont(104, "Bold")
    if p < 1:
        d.text((W - 90, 108 - 26 * p), str(y - 1), font=big, fill=INK + (int(235 * a * (1 - p)),), anchor="rs", features=TNUM)
    d.text((W - 90, 108 + 26 * (1 - p)), str(y), font=big, fill=INK + (int(235 * a * p),), anchor="rs", features=TNUM)
    for yy, al in ((y - 1, 1 - p), (y, p)):          # its line under it changes with it
        if al > 0.01 and yy >= FIRST:
            d.text((W - 92, 140), f"$1 in {yy} = ${float(D.cov[yy]['to_2025']):,.2f} in 2025", font=font(19, "Regular"),
                   fill=INK2 + (int(255 * a * al),), anchor="rs", features=TNUM)


def caption(d, T, a=1.0):
    """T: seconds from the start of the film"""
    for e in D.events:
        al = a * smooth((T - e["T0"]) / 0.5) * (1 - smooth((T - e["T1"]) / 0.5))
        if al <= 0.01:
            continue
        lines = e["text"] if isinstance(e["text"], list) else [e["text"]]
        if PANEL:                                     # in the band between the board and the revenue panel
            y0 = Y0 + N * PITCH + 52
            d.rounded_rectangle([90, y0 - 26, 95, y0 + 4 + 30 * (len(lines) - 1)], 2, fill=GOLD + (int(230 * al),))
            for i, line in enumerate(lines):
                d.text((112, y0 + i * 32), line, font=font(26 if i == 0 else 21, "SemiBold" if i == 0 else "Regular"),
                       fill=(INK if i == 0 else INK2) + (int(255 * al),), anchor="ls")
            continue
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
        if PANEL:                                     # a title card: the span, the title over a gold rule, the leagues
            lift = 18 * (1 - smooth(T / 1.2))
            d.text((W / 2, 372 + lift), f"{FIRST} – {LAST}", font=font(24, "SemiBold"), fill=GOLD + (int(255 * a),), anchor="mm")
            d.text((W / 2, 452 + lift), MODE["title"], font=dfont(86, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
            d.rounded_rectangle([W / 2 - 60, 516 + lift, W / 2 + 60, 520 + lift], 2, fill=GOLD + (int(255 * a),))
            for i, line in enumerate(("Every salary in 2025 dollars.", "A season stays on the board until ten better ones come along.")):
                d.text((W / 2, 572 + 40 * i + lift), line, font=font(30, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
            ws = [26 + d.textlength(lg, font=font(24, "SemiBold")) for lg in ("MLB", "NBA", "NFL", "NHL")]
            x = W / 2 - (sum(ws) + 44 * 3) / 2
            for lg, w in zip(("MLB", "NBA", "NFL", "NHL"), ws):
                d.ellipse([x, 696, x + 16, 712], fill=LEAGUE[lg] + (int(255 * a),))
                d.text((x + 26, 704), lg, font=font(24, "SemiBold"), fill=INK + (int(230 * a),), anchor="lm")
                x += w + 44
            return np.asarray(im)
        d.text((W / 2, 420), MODE["title"], font=font(72, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((W / 2, 500), MODE["intro"], font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((W / 2, 600), "MLB · NBA · NFL · NHL", font=font(26, "SemiBold"), fill=MUTED + (int(255 * a),), anchor="mm")
        return np.asarray(im)
    hold = T - INTRO >= D.race
    t = float(D.year_of(min(T - INTRO, D.race)))
    a = smooth((T - INTRO) / 0.8)
    end = total()
    dim = 1 - 0.94 * smooth((T - (end - OUTRO + 1.6)) / 0.8)      # the board steps back for the summary
    if PANEL:
        revenue(im, t, a * dim)
    d = board(im, t, a * dim, hold, T)
    header(d, a)
    year_label(d, t, a * (dim - 0.06) / 0.94)          # gone under the sources line
    caption(d, T, a)
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
    return INTRO + D.race + OUTRO


def setup():
    global D, SCALE, REVSCALE
    D = Data()
    SCALE = scale_track()
    REVSCALE = rev_scale_track() if D.rev else None
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
    if D is None:
        setup()
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
            T = INTRO + D.secs(t) if t >= FIRST else t
            Image.fromarray(frame(T)).save(os.path.join(OUT, f"{MODE['film']}_still_{t:06.1f}.png"))
            print("still", t)
    else:
        render(int(args[1]) if len(args) > 1 else 4)
