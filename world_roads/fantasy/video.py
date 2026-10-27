"""Fantasy football's most valuable seasons since 1980: a bar-chart race of the all-time best single seasons, in points
over a replacement player (1920x1080, 30 fps), 12-team PPR.

    python video.py still 1984.5 2006.99 ...   -> out/still_1984.50.png ... (t in seasons: 2006.5 is mid-2006;
                                                  < 1980 is the title, > 2026 the summary)
    python video.py timeline                   seconds per season
    python video.py render [jobs]              -> out/race.mp4 (chunks rendered in parallel)

The board holds the twelve most valuable single seasons so far (../media/fantasy/seasons.csv, weekly.csv, built by
vorp.py); each bar is a player's season. While a season is played its bars grow toward their totals (outlined), and a
great one climbs onto the board and stays until better seasons push it off. Under the board, the season's best so far
when it is not on the board, with the place it would hold. Each bar carries the player, his position, team and year,
the value, and the season's stat line once it is over. The scale follows the leader.

The film opens with a 42-second explainer in plain words: why total points mislead (a real season where a quarterback
outscored a running back who was worth more), what "points over replacement" means (the waiver-wire player you could
have had for free), who that free player is in a 12-team league, why missed games count zero, and how to read the chart.

Colour encodes the position (QB magenta, RB blue, WR yellow, TE green: validated together for colour-blind separation on
this background) and every bar also names its position in text.

Seasons do not all take the same time: a season runs 2 to 12 beats of the score (1.4 to 8.4 s) by how much value its
seasons put on the board and how often bars passed each other, so the early years, when the board fills, run longest
and a season that changes nothing passes in 1.4 s. Within a season the weeks take its first 80%; the final standings
hold for the rest.
"""
import os, sys, csv, json, time, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, translucent text  # noqa: E402
from vorp import REPL, FIRST, LAST, CANDIDATES, weeks  # replacement ranks, years  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "fantasy")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 7, 13], np.float32)                 # #05070d
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
POSC = {"QB": (213, 81, 129), "RB": (57, 135, 229), "WR": (201, 133, 0), "TE": (0, 131, 0)}
font = ST.font

N = 12                                                # bars on the board: one per team in the league
BEAT = 0.7                                            # 85.7 BPM
# the intro, in plain words: the question, what "over replacement" means, who the free player is, missed games, how
# to read the chart (beats each)
SCENES = (("title", 5), ("question", 11), ("value", 14), ("free", 11), ("missed", 7), ("watch", 12))
INTRO, OUTRO = sum(b for _, b in SCENES) * BEAT, 16 * BEAT    # 42 s, 11.2 s
HOLD = 4 * BEAT                                       # the final board holds 2.8 s before the summary
WEEKS_FRAC = 0.8                                      # the weeks take this much of a season; the final standings the rest
X0, X1 = 560, 1460                                    # bar baseline and the leader's bar end
Y0, PITCH, BAR = 178, 54, 40                          # first row's top, row spacing, bar thickness
CHASE = Y0 + N * PITCH + 12                           # the season's best, when it is not on the board
OFF = N + 3                                           # the place of a bar that is not near the board


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def num(v):
    return f"{v:,.0f}"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def statline(r):
    """the season's line, by position"""
    i = lambda k: int(round(float(r[k])))
    if r["pos"] == "QB":
        return f"{i('pass_yds'):,} pass yds · {i('pass_td')} TD passes"
    if r["pos"] == "RB":
        return f"{i('rush_yds'):,} rush yds · {i('rec')} catches · {i('rush_td') + i('rec_td')} TD"
    return f"{i('rec')} catches · {i('rec_yds'):,} yds · {i('rec_td')} TD"


class Data:
    def __init__(self):
        self.all = defaultdict(list)                  # season -> every season's VORP (for all-time places)
        self.rep = defaultdict(dict)
        rows = []
        for r in csv.DictReader(open(os.path.join(MEDIA, "seasons.csv"))):
            r["season"], r["vorp"], r["points"] = int(r["season"]), float(r["vorp"]), float(r["points"])
            r["games"], r["season_rank"] = int(r["games"]), int(r["season_rank"])
            self.all[r["season"]].append(r["vorp"])
            if r["season_rank"] <= CANDIDATES:
                rows.append(r)
        for r in csv.DictReader(open(os.path.join(MEDIA, "replacement.csv"))):
            self.rep[int(r["season"])][r["pos"]] = float(r["replacement"])
        wk = defaultdict(list)                        # (pid, season) -> [(week, vorp_week)]
        for r in csv.DictReader(open(os.path.join(MEDIA, "weekly.csv"))):
            wk[r["player_id"], int(r["season"])].append((int(r["week"]), float(r["vorp_week"])))
        # the bars: seasons that could reach the board, in name order (a stable sort then breaks ties by name)
        self.e = sorted(rows, key=lambda r: (r["player"], r["season"]))
        self.weeks = {y: weeks(y) for y in range(FIRST, LAST + 1)}
        self.season_of = np.array([r["season"] for r in self.e])
        # value at every week boundary of every season: C[y] is (bars, weeks + 1); a season grows in its own year
        # (evenly over the games he played from 1999, over the season's weeks before), holds after, is 0 before
        self.C = {}
        final = np.array([r["vorp"] for r in self.e])
        for y in range(FIRST, LAST + 1):
            nw = self.weeks[y]
            c = np.where(self.season_of[:, None] < y, final[:, None], 0.0) * np.ones((1, nw + 1))
            for i in np.nonzero(self.season_of == y)[0]:
                r = self.e[i]
                if r["source"] == "nflverse":
                    inc = np.zeros(nw + 1)
                    for w, v in wk[r["player_id"], y]:
                        inc[min(w, nw)] += v
                    c[i] = np.cumsum(inc)
                    c[i] *= final[i] / c[i][-1] if c[i][-1] else 1.0      # ends exactly on the season's total
                else:
                    c[i] = final[i] * np.arange(nw + 1) / nw
            self.C[y] = c
        self.done = {y: np.sort(np.concatenate([self.all[s] for s in range(FIRST, y)]))[::-1] if y > FIRST else np.zeros(0)
                     for y in range(FIRST, LAST + 2)}
        self.timeline()
        self.track()

    # ------------------------------------------------------------------ time
    def board_gain(self, y):
        """value put on the board in season y: the change of the bars on it at the season's start or end"""
        a, b = self.C[y][:, 0], self.C[y][:, -1]
        on = {i for i in np.argsort(-a, kind="stable")[:N] if a[i] > 0} | set(np.argsort(-b, kind="stable")[:N])
        return float(sum(abs(b[i] - a[i]) for i in on))

    def passes(self, y):
        """how many times a bar moves up the board in season y (sampled 48 times a week)"""
        c, n, prev = self.C[y], 0, None
        for w in range(self.weeks[y]):
            for u in np.linspace(0, 1, 48, endpoint=False):
                v = c[:, w] + (c[:, w + 1] - c[:, w]) * u
                o = [i for i in np.argsort(-v, kind="stable")[:N] if v[i] > 0]
                if prev is not None:
                    pr = {i: k for k, i in enumerate(prev)}
                    n += sum(1 for k, i in enumerate(o) if i not in pr or pr[i] > k)
                prev = o
        return n

    def timeline(self):
        """beats per season: 2 (nothing moves) to 12, from the value put on the board (150 points a beat) and the
        passes on it (no more than 4 a second where the cap allows); the last season, the finish, at least 6"""
        self.beats = {}
        for y in range(FIRST, LAST + 1):
            b = int(np.clip(max(2 + round(self.board_gain(y) / 150), np.ceil(self.passes(y) / (4 * BEAT))), 2, 12))
            self.beats[y] = max(b, 6) if y == LAST else b
        self.t0 = {}
        t = INTRO
        for y in range(FIRST, LAST + 1):
            self.t0[y] = t
            t += self.beats[y] * BEAT
        self.race_end = t

    def at(self, T):
        """film time -> (season, fraction of the season's slot)"""
        if T < INTRO:
            return FIRST, 0.0
        for y in range(LAST, FIRST - 1, -1):
            if T >= self.t0[y]:
                return y, min(1.0, (T - self.t0[y]) / (self.beats[y] * BEAT))
        return FIRST, 0.0

    def film_time(self, y, week):
        """film time at which season y's week ends (week 0: the season's start; week = weeks: its final standings)"""
        return self.t0[y] + self.beats[y] * BEAT * WEEKS_FRAC * week / self.weeks[y]

    def season_t(self, s):
        """'2006.5' (mid-season, by slot fraction) -> film time"""
        y = int(np.floor(s))
        if y < FIRST:
            return max(0.0, s - (FIRST - 1)) * INTRO
        if y > LAST:
            return self.race_end + (s - y) * OUTRO
        return self.t0[y] + (s - y) * self.beats[y] * BEAT

    def values(self, T):
        y, f = self.at(T)
        wf = min(1.0, f / WEEKS_FRAC) * self.weeks[y]
        k = int(np.floor(wf))
        c = self.C[y]
        if k >= self.weeks[y]:
            return c[:, -1], y, self.weeks[y], f
        u = wf - k
        return c[:, k] + (c[:, k + 1] - c[:, k]) * u, y, wf, f

    def place(self, v, y, vals):
        """the all-time place a value would hold now: behind every finished season and this season's bars above it"""
        before = int(np.sum(self.done[y] > v))
        now = int(np.sum(vals[self.season_of == y] > v))
        return before + now + 1

    def track(self):
        """every bar's place at every frame of the race (ranked by value, ties by name; a season with no value is off
        the board), eased in time with a Gaussian of 0.1 s so a pass takes about a third of a second; the leader's
        value for the scale"""
        n = int(round(self.race_end * FPS)) + 1
        R = np.full((len(self.e), n), float(OFF), np.float32)
        lead = np.zeros(n)
        for j in range(n):
            v = self.values(j / FPS)[0]
            o = np.argsort(-v, kind="stable")[:OFF]
            o = o[v[o] > 0]
            R[o, j] = np.arange(len(o))
            lead[j] = v[o[0]] if len(o) else 0.0
        s = 0.1 * FPS
        h = int(3 * s)
        ker = np.exp(-0.5 * (np.arange(-h, h + 1) / s) ** 2)
        ker /= ker.sum()
        self.near = np.nonzero((R < OFF).any(1))[0]
        self.pos = {int(i): np.convolve(np.pad(R[i], h, mode="edge"), ker, mode="valid") for i in self.near}
        k = int(0.9 * FPS)
        ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.2)) ** 2)
        ker /= ker.sum()
        sm = np.convolve(np.pad(lead, k, mode="edge"), ker, mode="valid")
        self.scale = np.maximum(np.maximum(sm, lead * 0.97), 60.0)

    def frame_ix(self, T):
        return int(round(min(max(T, 0), self.race_end) * FPS))

    def at_pos(self, i, j):
        return float(self.pos[i][min(j, len(self.pos[i]) - 1)]) if i in self.pos else float(OFF)


D = None
CAPS = []
SUMMARY = {}


def total():
    return D.race_end + OUTRO


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
    steps = [m * 10 ** e for e in range(1, 5) for m in (1, 2, 5)]
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
        d.text((x, Y0 - 22), num(v), font=font(20, "Regular"), fill=MUTED + (int(255 * al),), anchor="ms")


def bar(d, r, v, yy, x1, al, tl, live, final, sub=None):
    """one season's bar and its labels; live: this season, outlined"""
    col = POSC[r["pos"]]
    d.rounded_rectangle([X0, yy, x1, yy + BAR], 4, fill=col + (int(255 * al),))
    d.rectangle([X0, yy, min(x1, X0 + 6), yy + BAR], fill=col + (int(255 * al),))
    if live:
        d.rounded_rectangle([X0 - 2, yy - 2, x1 + 2, yy + BAR + 2], 5, outline=INK + (int(200 * al),), width=2)
    if tl <= 0.01:
        return
    d.text((X0 - 18, yy + BAR / 2 - 9), r["player"], font=font(26, "SemiBold"), fill=INK + (int(255 * tl),), anchor="rm")
    d.text((X0 - 18, yy + BAR / 2 + 13), f"{r['pos']} · {r['team']} · {r['season']}", font=font(18, "Regular"),
           fill=(INK if live else INK2) + (int(235 * tl),), anchor="rm")
    d.text((x1 + 14, yy + BAR / 2 - 7), num(v), font=font(26, "Bold"), fill=INK + (int(255 * tl),), anchor="lm")
    sub = sub or (statline(r) if final else "so far")
    d.text((x1 + 14, yy + BAR / 2 + 14), sub, font=font(18, "Regular"), fill=MUTED + (int(235 * tl),), anchor="lm")


def board(im, T, a=1.0, hold=False):
    d = ImageDraw.Draw(im, "RGBA")
    j = D.frame_ix(T)
    xmax = D.scale[min(j, len(D.scale) - 1)]
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    ticks(d, xmax, a)
    vals, y, wf, f = D.values(min(T, D.race_end))
    final = wf >= D.weeks[y]
    st = {}
    for i in D.near:
        r = D.at_pos(int(i), j)
        if r < N - 0.15 and vals[i] > 0:
            st[int(i)] = (r, float(vals[i]))
    # longest first, so a shorter bar passing a longer one stays in view on top of it; while two bars overlap, the
    # lower one's labels give way
    for i, (r, v) in sorted(st.items(), key=lambda kv: (-kv[1][1], kv[1][0])):
        al = a * smooth((N - 0.15 - r) / 0.7) * smooth(v / 4)
        if al <= 0.01:
            continue
        near = min((abs(r - r2) for i2, (r2, v2) in st.items() if i2 != i and (v2, -r2) > (v, -r)), default=9)
        live = D.season_of[i] == y and not hold
        bar(d, D.e[i], v, Y0 + r * PITCH, max(X0 + 6, xs(v)), al, al * smooth((near - 0.3) / 0.4), live,
            final or D.season_of[i] < y)
    # the season's best so far, under the board while it is not on it
    if not hold:
        mine = np.nonzero((D.season_of == y) & (vals > 0))[0]
        if len(mine):
            i = int(mine[np.argmax(vals[mine])])
            show = a * smooth((D.at_pos(i, j) - (N - 0.15)) / 0.7)
            if show > 0.01:
                d.line([(X0 - 330, CHASE - 7), (X1 + 260, CHASE - 7)], fill=(40, 52, 72, int(200 * show)), width=1)
                v = float(vals[i])
                place = D.place(v, y, vals)
                bar(d, D.e[i], v, CHASE, max(X0 + 6, xs(v)), show, show, True, final,
                    sub=f"{y}'s best{' so far' if not final else ''}: No. {place} all-time")
    return d


def header(d, a=1.0):
    d.text((90, 62), "Fantasy football's most valuable seasons", font=font(46, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((90, 104), f"The 12 best single seasons since {FIRST}, in points over a replacement player · 12-team PPR",
           font=font(25, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
    x = W - 90
    for p in ("TE", "WR", "RB", "QB"):
        w = d.textlength(p, font=font(22, "SemiBold"))
        d.text((x, 62), p, font=font(22, "SemiBold"), fill=INK + (int(255 * a),), anchor="rs")
        d.rounded_rectangle([x - w - 30, 44, x - w - 10, 64], 4, fill=POSC[p] + (int(255 * a),))
        x -= w + 58
    d.text((W - 90, 100), "outlined: this season", font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="rs")


def season_label(d, T, a=1.0):
    vals, y, wf, f = D.values(min(T, D.race_end))
    d.text((W - 90, H - 86), str(y), font=font(132, "Bold"), fill=INK + (int(235 * a),), anchor="rs")
    wk = "Final" if wf >= D.weeks[y] else f"Week {max(1, int(np.ceil(wf)))}"
    d.text((W - 92, H - 46), wk, font=font(28, "SemiBold"), fill=INK2 + (int(255 * a),), anchor="rs")
    rep = D.rep[y]
    line = "   ".join(f"{p}{REPL[p]} {num(rep[p])}" for p in REPL)
    d.text((X0, CHASE + BAR + 36), f"Replacement in {y}, season points:   {line}", font=font(20, "Regular"),
           fill=MUTED + (int(255 * a),), anchor="ls")


def caption(d, T, a=1.0):
    for e in CAPS:
        t0, t1 = e["from"], e["to"]
        al = a * smooth((T - t0) / 0.35) * (1 - smooth((T - t1) / 0.35))
        if al <= 0.01:
            continue
        y = H - 82
        for i, line in enumerate(e["text"]):
            d.text((90, y + i * 36), line, font=font(29 if i == 0 else 23, "SemiBold" if i == 0 else "Regular"),
                   fill=(INK if i == 0 else INK2) + (int(255 * al),), anchor="ls")


def scene_at(T):
    """the intro scene at film time T: (name, seconds into it, its length)"""
    t = 0.0
    for name, b in SCENES:
        if T < t + b * BEAT or name == SCENES[-1][0]:
            return name, T - t, b * BEAT
        t += b * BEAT


def scene_starts():
    return list(np.cumsum([0] + [b * BEAT for _, b in SCENES[:-1]]))


def lines(d, rows, y, al, size=31, col=INK2, weight="Regular", gap=46):
    for k, line in enumerate(rows):
        d.text((W / 2, y + gap * k), line, font=font(size, weight), fill=col + (int(255 * al),), anchor="mm")


def example_bars(d, al, split=0.0, value_al=0.0):
    """the intro's example: two seasons' points, then (split) the part the best free player would have scored"""
    x = SUMMARY.get("example")
    if not x or al <= 0.01:
        return
    k = 860 / max(x["qb"]["points"], x["rb"]["points"])
    for n, (r, y) in enumerate(((x["qb"], 430), (x["rb"], 540))):
        col = POSC[r["pos"]]
        x0, xf, x1 = 640, 640 + r["free"] * k, 640 + r["points"] * k
        d.rounded_rectangle([x0, y, x1, y + 64], 6, fill=col + (int(255 * al * (1 - 0.75 * split)),))
        if split > 0.01:
            d.rounded_rectangle([xf, y, x1, y + 64], 6, fill=col + (int(255 * al * split),))
            d.rectangle([xf, y, min(x1, xf + 8), y + 64], fill=col + (int(255 * al * split),))
            d.rounded_rectangle([x0, y, xf, y + 64], 6, fill=(70, 78, 92, int(255 * al * split)))
            d.text(((x0 + xf) / 2, y + 32), f"best free {r['pos']}: {r['free']}", font=font(22, "SemiBold"),
                   fill=INK + (int(255 * al * smooth((split - 0.5) / 0.5)),), anchor="mm")
        d.text((x0 - 22, y + 20), r["player"], font=font(32, "SemiBold"), fill=INK + (int(255 * al),), anchor="rm")
        d.text((x0 - 22, y + 50), f"{r['pos']} · {r['team']} · {x['season']}", font=font(21, "Regular"),
               fill=INK2 + (int(255 * al),), anchor="rm")
        pa = al * (1 - value_al)
        d.text((x1 + 18, y + 32), f"{r['points']} points", font=font(30, "Bold"), fill=INK + (int(255 * pa),), anchor="lm")
        va = al * value_al
        d.text((x1 + 18, y + 20), f"+{r['value']}", font=font(30, "Bold"), fill=INK + (int(255 * va),), anchor="lm")
        d.text((x1 + 18, y + 50), "over replacement", font=font(21, "Regular"), fill=INK2 + (int(255 * va),), anchor="lm")


def intro(d, T):
    name, s, dur = scene_at(T)
    a = smooth(s / 0.5) * (1 - smooth((s - dur + 0.6) / 0.5))
    rv = lambda t0: a * smooth((s - t0) / 0.6)                     # an element revealed t0 seconds into the scene
    x = SUMMARY.get("example") or {}
    q, b = x.get("qb", {}), x.get("rb", {})
    if name == "title":
        d.text((W / 2, 470), "Fantasy football's most valuable seasons", font=font(72, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        lines(d, [f"The best single seasons since {FIRST} · 12-team PPR league"], 556, rv(0.6), size=34)
    elif name == "question":
        lines(d, ["Who had the best fantasy season ever?"], 290, a, size=54, col=INK, weight="Bold")
        lines(d, [f"Total points can't settle it. In {x.get('season')}, {q.get('player')} outscored {b.get('player')}:"], 370, rv(0.8))
        example_bars(d, rv(1.6))
        lines(d, [f"Yet {b.get('player', '').split()[-1]} did more to win fantasy leagues. Why?"], 740, rv(4.0), size=40,
              col=INK, weight="SemiBold")
    elif name == "value":
        lines(d, ["Because a good quarterback was easy to find"], 230, a, size=54, col=INK, weight="Bold")
        lines(d, ["Every season, useful players sit unclaimed on the waiver wire. A player is only worth",
                  "what he scores beyond the best one his manager could have picked up for free."], 310, rv(0.6))
        example_bars(d, a, split=smooth((s - 2.2) / 1.2), value_al=smooth((s - 3.6) / 0.6))
        la = rv(2.8)
        d.rounded_rectangle([560, 676, 584, 700], 4, fill=(70, 78, 92, int(255 * la)))
        d.text((596, 688), "what the best free player at his position scored", font=font(23, "Regular"),
               fill=INK2 + (int(255 * la),), anchor="lm")
        d.rounded_rectangle([1150, 676, 1162, 700], 3, fill=POSC["QB"] + (int(255 * la),))
        d.rounded_rectangle([1164, 676, 1176, 700], 3, fill=POSC["RB"] + (int(255 * la),))
        d.text((1188, 688), "what the player added: points over replacement", font=font(23, "Regular"),
               fill=INK2 + (int(255 * la),), anchor="lm")
        lines(d, [f"{b.get('player', '').split()[-1]} added {b.get('value')}, {q.get('player', '').split()[-1]} {q.get('value')}."
                  " That margin is what this race measures."], 800, rv(5.4), size=36, col=INK, weight="SemiBold")
    elif name == "free":
        lines(d, ["Who is the free player?"], 230, a, size=54, col=INK, weight="Bold")
        lines(d, ["A 12-team league starts a QB, 2 RBs, 3 WRs, a TE and a flex (an RB, WR or TE).",
                  "Once every team's starters are taken, the best player left is about the:"], 310, rv(0.6))
        why = {"QB": "12 starters, plus bye-week fill-ins", "RB": "24 starters, plus flex spots",
               "WR": "36 starters, plus flex spots", "TE": "12 starters, plus the odd flex"}
        for n, p in enumerate(REPL):
            ca = rv(1.6 + 0.4 * n)
            cx = W / 2 + (n - 1.5) * 380
            d.rounded_rectangle([cx - 130, 450, cx + 130, 456], 3, fill=POSC[p] + (int(255 * ca),))
            d.text((cx, 530), ordinal(REPL[p]), font=font(80, "Bold"), fill=INK + (int(255 * ca),), anchor="mm")
            d.text((cx, 602), f"best {p}", font=font(32, "SemiBold"), fill=INK + (int(255 * ca),), anchor="mm")
            d.text((cx, 645), why[p], font=font(22, "Regular"), fill=MUTED + (int(255 * ca),), anchor="mm")
        lines(d, ["Every season is measured against that year's free players,",
                  "so a 1985 season and a 2025 season are each judged by the standards of their day."], 760, rv(4.0))
    elif name == "missed":
        lines(d, ["Missed games count zero"], 410, a, size=54, col=INK, weight="Bold")
        lines(d, ["When a player sits out, his manager starts a free player that week,",
                  "so the missed game neither helps nor hurts him."], 500, rv(0.6))
        lines(d, ["300 points in 12 games beats 300 points in 16."], 660, rv(2.0), size=40, col=INK, weight="SemiBold")
    else:                                                          # how to read the chart
        lines(d, ["How to read the chart"], 250, a, size=54, col=INK, weight="Bold")
        rows = ("Each bar is one player's season, coloured by position", "The board holds the 12 most valuable seasons so far",
                "Outlined bars are the season being played, growing week by week",
                "Under the board: this season's best so far, and where it would rank")
        for n, text in enumerate(rows):
            ra = rv(0.8 + 0.9 * n)
            y = 380 + 120 * n
            ix = 470
            if n == 0:
                for m, p in enumerate(POSC):
                    d.rounded_rectangle([ix, y - 30 + 16 * m, ix + 150 - 22 * m, y - 18 + 16 * m], 3, fill=POSC[p] + (int(255 * ra),))
                cx = 660
                for p in POSC:
                    d.rounded_rectangle([cx, y + 30, cx + 18, y + 48], 4, fill=POSC[p] + (int(255 * ra),))
                    d.text((cx + 26, y + 39), p, font=font(22, "SemiBold"), fill=INK2 + (int(255 * ra),), anchor="lm")
                    cx += 26 + d.textlength(p, font=font(22, "SemiBold")) + 28
            elif n == 1:
                for m in range(4):
                    d.rounded_rectangle([ix, y - 30 + 16 * m, ix + 160 - 26 * m, y - 18 + 16 * m], 3, fill=POSC["RB"] + (int(255 * ra),))
            elif n == 2:
                d.rounded_rectangle([ix, y - 16, ix + 120, y + 16], 4, fill=POSC["WR"] + (int(255 * ra),))
                d.rounded_rectangle([ix - 2, y - 18, ix + 122, y + 18], 5, outline=INK + (int(220 * ra),), width=2)
            else:
                d.line([(ix - 10, y - 26), (ix + 170, y - 26)], fill=(40, 52, 72, int(255 * ra)), width=2)
                d.rounded_rectangle([ix, y - 14, ix + 80, y + 14], 4, fill=POSC["RB"] + (int(255 * ra),))
                d.rounded_rectangle([ix - 2, y - 16, ix + 82, y + 16], 5, outline=INK + (int(220 * ra),), width=2)
            d.text((660, y - (8 if n == 0 else 0)), text, font=font(30, "Regular"), fill=INK + (int(255 * ra),), anchor="lm")


def summary(d, s):
    """over the final standings: the top two seasons, the board's make-up, the best at the other positions"""
    a = smooth(s / 0.8) * (1 - smooth((s - (OUTRO - HOLD - 0.8)) / 0.8))
    if a <= 0.01:
        return
    cx = W / 2
    d.text((cx, 230), f"The most valuable seasons, {FIRST}–{LAST}", font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for r, x in ((SUMMARY["first"], cx - 400), (SUMMARY["second"], cx + 400)):
        d.text((x, 330), r["player"], font=font(44, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 382), f"{r['pos']} · {r['team']} · {r['season']}", font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((x, 486), f"{r['vorp']:,.1f}", font=font(96, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 560), r["line"], font=font(26, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")
    for k, key in enumerate(("gap", "mix", "others")):
        d.text((cx, 670 + 64 * k), SUMMARY[key], font=font(36 if k == 0 else 28, "SemiBold" if k == 0 else "Regular"),
               fill=(INK if k == 0 else INK2) + (int(255 * a),), anchor="mm")
    for i, line in enumerate(SOURCES):
        d.text((cx, 980 + 27 * i), line, font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")


SOURCES = ("Data: Pro Football Reference season tables 1980–1998 (via fantasydatapros/data), nflverse weekly player stats 1999–2025",
           "PPR: 1 per catch, 1 per 10 rushing or receiving yards, 1 per 25 passing yards, 6 per TD (4 passing), −2 per interception",
           "or lost fumble (estimated before 1994) · two-point conversions and return TDs only from 1999 · regular seasons")


def frame(T):
    """T: seconds from the start of the film"""
    im = canvas()
    if T < INTRO - 0.2:
        intro(ImageDraw.Draw(im, "RGBA"), T)
        return np.asarray(im)
    a = smooth((T - INTRO + 0.2) / 0.8)
    hold = T >= D.race_end
    end = total()
    dim = 1 - 0.94 * smooth((T - (end - OUTRO + HOLD)) / 0.8)     # the board steps back for the summary
    d = board(im, T, a * dim, hold)
    header(d, a)
    season_label(d, T, a * max(0.0, dim - 0.06) / 0.94)
    caption(d, T, a)
    if T > end - OUTRO + HOLD:
        summary(d, T - (end - OUTRO + HOLD))
    return np.asarray(im)


def setup():
    global D, CAPS
    D = Data()
    p = os.path.join(HERE, "captions.json")
    CAPS = []
    if os.path.exists(p):
        for c in json.load(open(p)):
            t0 = D.film_time(c["season"], c["week"]) + c.get("delay", 0.0)
            CAPS.append(dict(c, **{"from": t0}))
        CAPS.sort(key=lambda c: c["from"])
        # each shows 3.6 s; when the next is due sooner, it makes room, down to 2.4 s, and the next waits for the rest
        for k, c in enumerate(CAPS):
            if k and CAPS[k - 1]["to"] + 0.3 > c["from"]:
                CAPS[k - 1]["to"] = max(CAPS[k - 1]["from"] + 2.4, c["from"] - 0.3)
                c["from"] = max(c["from"], CAPS[k - 1]["to"] + 0.3)
            c["to"] = c["from"] + 3.6
        if CAPS:
            CAPS[-1]["to"] = min(CAPS[-1]["to"], D.race_end + HOLD - 0.2)
    s = os.path.join(HERE, "summary.json")
    if os.path.exists(s):
        SUMMARY.update(json.load(open(s)))


def render_range(c0, c1, part):
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
    setup()
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    n = int(round(total() * FPS))
    CH = 300
    parts = [(c0, min(n, c0 + CH), os.path.join(OUT, "chunks", f"race_{c0:05d}.mp4")) for c0 in range(0, n, CH)]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(render_range, *p) for p in parts if not os.path.exists(p[2])]
        for f in futs:
            f.result()
    print(f"{n} frames in {time.time() - t0:.0f} s", flush=True)
    lst = os.path.join(OUT, "chunks", "race.txt")
    open(lst, "w").write("".join(f"file '{p[2]}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(OUT, "race.mp4")], check=True)
    for p in parts:
        os.remove(p[2])
    print("wrote", os.path.join(OUT, "race.mp4"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "still":
        setup()
        for a in sys.argv[2:]:
            s = float(a)
            T = D.season_t(s)
            Image.fromarray(frame(T)).save(os.path.join(OUT, f"still_{s:07.2f}.png"))
            print("still", s, f"T={T:.1f}s")
    elif sys.argv[1] == "timeline":
        setup()
        for y in range(FIRST, LAST + 1):
            print(y, D.beats[y], f"{D.t0[y]:.1f}s", f"gain {D.board_gain(y):.0f}")
        print(f"race ends {D.race_end:.1f}s, film {total():.1f}s")
    else:
        render(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
