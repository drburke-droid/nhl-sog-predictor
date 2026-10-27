"""Fantasy football's most valuable seasons since 1980: a bar-chart race of the all-time best single seasons, in points
over a replacement player (1920x1080, 30 fps), 12-team PPR.

    python video.py still 1984.5 2006.99 ...   -> out/still_1984.50.png ... (t in seasons: 2006.5 is mid-2006;
                                                  < 1980 is the title, > 2026 the summary)
    python video.py timeline                   seconds per season
    python video.py render [jobs]              -> out/race.mp4 (chunks rendered in parallel)

The board holds the twelve most valuable single seasons so far (../media/fantasy/seasons.csv, weekly.csv, built by
vorp.py); each bar is a player's season. While a season is played its bars grow toward their totals (glowing), and a
great one climbs onto the board and stays until better seasons push it off. Under the board, the season's best so far
when it is not on the board, with the place it would hold. Each bar carries the player, his position, team and year,
the value, and the season's stat line once it is over. The scale follows the leader.

The film opens with a 39-second explainer in plain words: the hook, why total points mislead (a real season where a
quarterback outscored a running back who was worth more), what "points over replacement" means (the waiver-wire player
you could have had for free), who that free player is in a 12-team league, and how to read the race. The look (look.py):
a lit navy field, Barlow Condensed display type, bars with a sheen and the year in the tip, rank lanes, a glow on the
season being played, a gold flash and tag when a new No. 1 arrives, lower-third caption cards, and a 1980-2025 timeline
with a gold dot where each record fell.

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
import style as ST                                    # translucent text  # noqa: E402
import look as L                                      # background, type, bars, glow  # noqa: E402
from vorp import REPL, FIRST, LAST, CANDIDATES, weeks  # replacement ranks, years  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "fantasy")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
POSC = L.POSC

N = 12                                                # bars on the board: one per team in the league
BEAT = 0.7                                            # 85.7 BPM
# the intro, in plain words: the hook, the question, what "over replacement" means, who the free player is, how to
# read the race (beats each)
SCENES = (("title", 8), ("question", 11), ("value", 14), ("free", 11), ("watch", 12))
INTRO, OUTRO = sum(b for _, b in SCENES) * BEAT, 16 * BEAT    # 39.2 s, 11.2 s
HOLD = 4 * BEAT                                       # the final board holds 2.8 s before the summary
WEEKS_FRAC = 0.8                                      # the weeks take this much of a season; the final standings the rest
X0, X1 = 610, 1440                                    # bar baseline and the leader's bar end
Y0, PITCH, BAR = 176, 52, 38                          # first row's top, row spacing, bar thickness
CHASE = Y0 + N * PITCH + 20                           # the season's best, when it is not on the board
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
RECORDS = []                                          # film times at which a new No. 1 arrives
RECORD_SEASONS = []                                   # (season, film time) of each, for the timeline


def total():
    return D.race_end + OUTRO


def canvas():
    return L.background()


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
        d.line([(x, Y0 - 12), (x, Y0 + N * PITCH - 10)], fill=L.GRID + (int(120 * al),), width=1)
        d.text((x, Y0 - 20), num(v), font=L.inter(18, "Medium"), fill=MUTED + (int(255 * al),), anchor="ms")


def lanes(d, a):
    """the twelve places: a faint lane each and its number, No. 1 in gold"""
    for k in range(N):
        y = Y0 + k * PITCH
        d.rounded_rectangle([96, y - 3, W - 90, y + BAR + 3], 10, fill=(255, 255, 255, int(7 * a)))
        d.text((138, y + BAR / 2 + 1), str(k + 1), font=L.bc(34, "Bold"),
               fill=(L.GOLD if k == 0 else MUTED) + (int(255 * a * (1 if k == 0 else 0.8)),), anchor="mm")


def season_bar(im, d, r, v, yy, x1, al, tl, live, final, sub=None, flash=0.0):
    """one season's bar and its labels: the year in the bar's tip, a glow while it is being played"""
    col = POSC[r["pos"]]
    if live:
        L.glow(im, (X0, yy, x1, yy + BAR), col, 0.55 * al, 14)
    if flash > 0.01:
        L.glow(im, (X0, yy, x1, yy + BAR), L.GOLD, 0.9 * flash, 20)
    L.bar(im, X0, yy, x1, BAR, col, al)
    if x1 - X0 > 120:
        d.text((x1 - 12, yy + BAR / 2 + 1), str(r["season"]), font=L.bc(30, "BlackItalic"),
               fill=tuple(int(c * 0.32) for c in col) + (int(230 * al),), anchor="rm")
    if tl <= 0.01:
        return
    d.text((X0 - 18, yy + BAR / 2 - 7), r["player"].upper(), font=L.bc(31, "Bold"), fill=INK + (int(255 * tl),), anchor="rm")
    d.text((X0 - 18, yy + BAR / 2 + 15), f"{r['pos']} · {r['team']}", font=L.inter(16, "Medium"),
           fill=INK2 + (int(235 * tl),), anchor="rm")
    d.text((x1 + 14, yy + BAR / 2 - 6), num(v), font=L.bc(36, "Bold"), fill=INK + (int(255 * tl),), anchor="lm")
    sub = sub or (statline(r) if final else "so far")
    d.text((x1 + 15, yy + BAR / 2 + 16), sub, font=L.inter(15, "Regular"), fill=MUTED + (int(240 * tl),), anchor="lm")
    if flash > 0.01:
        w = d.textlength(num(v), font=L.bc(36, "Bold"))
        L.pill(d, x1 + 26 + w, yy + BAR / 2 - 6, "NEW No. 1", L.GOLD, size=20, alpha=tl * flash)


def board(im, T, a=1.0, hold=False):
    d = ImageDraw.Draw(im, "RGBA")
    j = D.frame_ix(T)
    xmax = D.scale[min(j, len(D.scale) - 1)]
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    lanes(d, a)
    ticks(d, xmax, a)
    vals, y, wf, f = D.values(min(T, D.race_end))
    final = wf >= D.weeks[y]
    flash = max((smooth((T - t) / 0.2) * (1 - smooth((T - t - 1.8) / 0.6)) for t in RECORDS if t <= T + 0.2), default=0.0)
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
        season_bar(im, d, D.e[i], v, Y0 + r * PITCH, max(X0 + 6, xs(v)), al, al * smooth((near - 0.3) / 0.4), live,
                   final or D.season_of[i] < y, flash=flash * smooth((0.5 - r) / 0.5))
    # the season's best so far, under the board while it is not on it
    if not hold:
        mine = np.nonzero((D.season_of == y) & (vals > 0))[0]
        if len(mine):
            i = int(mine[np.argmax(vals[mine])])
            show = a * smooth((D.at_pos(i, j) - (N - 0.15)) / 0.7)
            if show > 0.01:
                for x in range(100, W - 90, 14):
                    d.line([(x, CHASE - 10), (x + 6, CHASE - 10)], fill=L.GRID + (int(200 * show),), width=1)
                d.text((100, CHASE + BAR / 2 - 7), "THIS SEASON'S BEST", font=L.bc(20, "SemiBold"),
                       fill=MUTED + (int(255 * show),), anchor="lm")
                d.text((100, CHASE + BAR / 2 + 14), "not on the board yet" if not final else "missed the board",
                       font=L.inter(14, "Regular"), fill=MUTED + (int(200 * show),), anchor="lm")
                v = float(vals[i])
                place = D.place(v, y, vals)
                season_bar(im, d, D.e[i], v, CHASE, max(X0 + 6, xs(v)), show, show, True, final,
                           sub=f"No. {place} all-time{' so far' if not final else ''}")
    return d


def header(d, a=1.0):
    d.text((90, 78), "THE GREATEST FANTASY SEASONS", font=L.bc(58, "ExtraBold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((92, 112), f"The 12 most valuable single seasons since {FIRST}, in points over replacement · 12-team PPR",
           font=L.inter(20, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
    x = W - 90
    for p in ("TE", "WR", "RB", "QB"):
        w = d.textlength(p, font=L.bc(24, "Bold"))
        d.text((x, 74), p, font=L.bc(24, "Bold"), fill=INK + (int(255 * a),), anchor="rs")
        d.ellipse([x - w - 26, 56, x - w - 10, 72], fill=POSC[p] + (int(255 * a),))
        x -= w + 50
    d.text((W - 90, 110), "glowing: the season being played", font=L.inter(17, "Regular"), fill=MUTED + (int(255 * a),), anchor="rs")


def season_label(d, T, a=1.0):
    vals, y, wf, f = D.values(min(T, D.race_end))
    fy = L.bc(176, "BlackItalic")
    d.text((W - 86, 1012), str(y), font=fy, fill=INK + (int(240 * a),), anchor="rs")
    yw = d.textlength(str(y), font=fy)
    wk = "FINAL" if wf >= D.weeks[y] else f"WEEK {max(1, int(np.ceil(wf)))}"
    d.text((W - 92, 872), wk, font=L.bc(28, "Bold"), fill=(L.GOLD if wk == "FINAL" else INK2) + (int(255 * a),), anchor="rs")
    x0, x1 = W - 92 - yw * 0.92, W - 92
    d.rounded_rectangle([x0, 1018, x1, 1022], 2, fill=L.GRID + (int(200 * a),))
    d.rounded_rectangle([x0, 1018, x0 + (x1 - x0) * min(1.0, wf / D.weeks[y]), 1022], 2, fill=INK + (int(230 * a),))
    rep = D.rep[y]
    line = "    ".join(f"{p}{REPL[p]} {num(rep[p])}" for p in REPL)
    d.text((X0, 892), f"Free player at each position in {y}:    {line}", font=L.inter(17, "Regular"),
           fill=MUTED + (int(255 * a),), anchor="ls")


def timeline(d, T, a=1.0):
    """1980 to 2025 along the bottom: where the race is, and a gold dot where each record fell"""
    x0, x1, yy = 96, W - 92, 1050
    pos = lambda t: x0 + (x1 - x0) * (t - FIRST) / (LAST + 1 - FIRST)
    y, f = D.at(min(T, D.race_end))
    now = pos(y + f) if T < D.race_end else x1
    d.line([(x0, yy), (x1, yy)], fill=L.GRID + (int(200 * a),), width=2)
    d.line([(x0, yy), (now, yy)], fill=INK2 + (int(200 * a),), width=2)
    for dec in range(FIRST, LAST + 1, 5):
        x = pos(dec)
        d.line([(x, yy - 5), (x, yy + 5)], fill=L.GRID + (int(255 * a),), width=1)
        if dec % 10 == 0:
            d.text((x, yy + 20), str(dec), font=L.inter(14, "Medium"), fill=MUTED + (int(255 * a),), anchor="mm")
    for s, t in RECORD_SEASONS:
        if t <= T:
            x = pos(s)
            d.ellipse([x - 5, yy - 5, x + 5, yy + 5], fill=L.GOLD + (int(255 * a),))
    d.ellipse([now - 7, yy - 7, now + 7, yy + 7], fill=INK + (int(255 * a),))


def caption(im, d, T, a=1.0):
    """a lower third: an accent stripe, a tag, the headline and a line under it"""
    for e in CAPS:
        t0, t1 = e["from"], e["to"]
        al = a * smooth((T - t0) / 0.35) * (1 - smooth((T - t1) / 0.35))
        if al <= 0.01:
            continue
        acc = tuple(e.get("accent") or INK2)
        head, sub = e["text"][0], e["text"][1] if len(e["text"]) > 1 else ""
        tag = e.get("tag", "")
        fh, fs = L.bc(34, "Bold"), L.inter(20, "Regular")
        tw = (d.textlength(tag, font=L.bc(19, "Bold")) + 32) if tag else 0
        w = max(tw + d.textlength(head, font=fh), d.textlength(sub, font=fs)) + 60
        dy = 10 * (1 - smooth((T - t0) / 0.35))
        x, y = 90, 912 + dy
        d.rounded_rectangle([x, y, x + w, y + 96], 12, fill=L.PANEL + (int(215 * al),))
        d.rounded_rectangle([x, y, x + 7, y + 96], 3, fill=acc + (int(255 * al),))
        if tag:
            L.pill(d, x + 26, y + 32, tag, acc, size=19, alpha=al)
        d.text((x + 26 + tw, y + 33), head, font=fh, fill=INK + (int(255 * al),), anchor="lm")
        d.text((x + 28, y + 70), sub, font=fs, fill=INK2 + (int(255 * al),), anchor="lm")


def scene_at(T):
    """the intro scene at film time T: (name, seconds into it, its length)"""
    t = 0.0
    for name, b in SCENES:
        if T < t + b * BEAT or name == SCENES[-1][0]:
            return name, T - t, b * BEAT
        t += b * BEAT


def scene_starts():
    return list(np.cumsum([0] + [b * BEAT for _, b in SCENES[:-1]]))


def lines(d, rows, y, al, size=30, col=INK2, face=None, gap=44, rise=0.0):
    f = face or L.inter(size, "Regular")
    for k, line in enumerate(rows):
        d.text((W / 2, y + gap * k + 18 * rise), line, font=f, fill=col + (int(255 * al),), anchor="mm")


def heading(d, text, y, al, rise=0.0, size=66):
    d.text((W / 2, y + 18 * rise), text, font=L.bc(size, "ExtraBold"), fill=INK + (int(255 * al),), anchor="mm")


def example_bars(im, d, al, split=0.0, value_al=0.0, y0=450):
    """the intro's example: two seasons' points, then (split) the part the best free player would have scored"""
    x = SUMMARY.get("example")
    if not x or al <= 0.01:
        return
    k = 860 / max(x["qb"]["points"], x["rb"]["points"])
    for n, r in enumerate((x["qb"], x["rb"])):
        y = y0 + 110 * n
        col = POSC[r["pos"]]
        x0, xf, x1 = 660, 660 + r["free"] * k, 660 + r["points"] * k
        if split < 0.999:
            L.bar(im, x0, y, x1, 64, col, al * (1 - split))
        if split > 0.001:
            L.bar(im, x0, y, x1, 64, col, al * split)
            L.bar(im, x0, y, xf, 64, (78, 88, 108), al * split)
            d.text(((x0 + xf) / 2, y + 33), f"BEST FREE {r['pos']}: {r['free']}", font=L.bc(26, "Bold"),
                   fill=INK + (int(255 * al * smooth((split - 0.5) / 0.5)),), anchor="mm")
        d.text((x0 - 22, y + 22), r["player"].upper(), font=L.bc(40, "Bold"), fill=INK + (int(255 * al),), anchor="rm")
        d.text((x0 - 22, y + 52), f"{r['pos']} · {r['team']} · {x['season']}", font=L.inter(19, "Medium"),
               fill=INK2 + (int(255 * al),), anchor="rm")
        pa = al * (1 - value_al)
        d.text((x1 + 18, y + 32), f"{r['points']} pts", font=L.bc(40, "Bold"), fill=INK + (int(255 * pa),), anchor="lm")
        va = al * value_al
        d.text((x1 + 18, y + 22), f"+{r['value']}", font=L.bc(44, "Bold"), fill=L.GOLD + (int(255 * va),), anchor="lm")
        d.text((x1 + 20, y + 54), "over replacement", font=L.inter(18, "Regular"), fill=INK2 + (int(255 * va),), anchor="lm")


def intro(im, d, T):
    name, s, dur = scene_at(T)
    a = smooth(s / 0.5) * (1 - smooth((s - dur + 0.6) / 0.5))
    rv = lambda t0: a * smooth((s - t0) / 0.6)                     # an element revealed t0 seconds into the scene
    up = lambda t0: 1 - smooth((s - t0) / 0.6)                     # ... rising into place as it appears
    x = SUMMARY.get("example") or {}
    q, b = x.get("qb", {}), x.get("rb", {})
    last = lambda r: r.get("player", "").split()[-1]
    if name == "title":
        for k, (word, y) in enumerate((("THE GREATEST", 420), ("FANTASY SEASON EVER", 560))):
            d.text((W / 2, y + 30 * up(0.15 * k)), word, font=L.bc(150, "BlackItalic"),
                   fill=INK + (int(255 * rv(0.15 * k)),), anchor="mm")
        sw = 760 * smooth((s - 0.6) / 0.8)
        d.rounded_rectangle([W / 2 - sw, 640, W / 2 + sw, 646], 3, fill=L.GOLD + (int(255 * a),))
        lines(d, [f"Every season since {FIRST}, ranked by how much it really helped win fantasy leagues"], 712, rv(1.0),
              rise=up(1.0))
        lines(d, ["12-TEAM PPR LEAGUE"], 768, rv(1.4), face=L.bc(26, "SemiBold"), col=MUTED, rise=up(1.4))
        f, g = SUMMARY.get("first"), SUMMARY.get("second")
        if f and g:
            gap = f["vorp"] - g["vorp"]
            hook = "DECIDED BY A TENTH OF A POINT" if round(gap, 1) == 0.1 else f"DECIDED BY {gap:.1f} POINTS"
            lines(d, [hook], 850, rv(2.0), face=L.bc(40, "BlackItalic"), col=L.GOLD, rise=up(2.0))
    elif name == "question":
        heading(d, "WHO HAD THE BEST SEASON EVER?", 270, a, up(0))
        lines(d, [f"Total points can't settle it. In {x.get('season')}, {q.get('player')} outscored {b.get('player')}:"],
              350, rv(0.8), rise=up(0.8))
        example_bars(im, d, rv(1.6), y0=440)
        lines(d, [f"Yet {last(b)} did more to win fantasy leagues. Why?"], 740, rv(4.0), face=L.bc(48, "Bold"), col=INK,
              rise=up(4.0))
    elif name == "value":
        heading(d, "BECAUSE A GOOD QB WAS EASY TO FIND", 230, a, up(0))
        lines(d, ["Every season, useful players sit unclaimed on the waiver wire. A player is only worth",
                  "what he scores beyond the best one his manager could have picked up for free."], 312, rv(0.6), rise=up(0.6))
        example_bars(im, d, a, split=smooth((s - 2.2) / 1.2), value_al=smooth((s - 3.6) / 0.6), y0=430)
        la = rv(2.8)
        L.bar(im, 560, 684, 588, 22, (78, 88, 108), la)
        d.text((600, 695), "what the best free player at his position scored", font=L.inter(21, "Regular"),
               fill=INK2 + (int(255 * la),), anchor="lm")
        L.bar(im, 1120, 684, 1134, 22, POSC["QB"], la)
        L.bar(im, 1136, 684, 1150, 22, POSC["RB"], la)
        d.text((1162, 695), "what the player added: points over replacement", font=L.inter(21, "Regular"),
               fill=INK2 + (int(255 * la),), anchor="lm")
        lines(d, [f"{last(b)} added {b.get('value')}, {last(q)} {q.get('value')}. That margin is what this race measures."],
              800, rv(5.4), face=L.bc(44, "Bold"), col=INK, rise=up(5.4))
    elif name == "free":
        heading(d, "WHO IS THE FREE PLAYER?", 230, a, up(0))
        lines(d, ["A 12-team league starts a QB, 2 RBs, 3 WRs, a TE and a flex (an RB, WR or TE).",
                  "Once every team's starters are taken, the best player left is about the:"], 312, rv(0.6), rise=up(0.6))
        why = {"QB": "12 starters, plus bye-week fill-ins", "RB": "24 starters, plus flex spots",
               "WR": "36 starters, plus flex spots", "TE": "12 starters, plus the odd flex"}
        for n, p in enumerate(REPL):
            ca, ru = rv(1.6 + 0.35 * n), up(1.6 + 0.35 * n)
            cx = W / 2 + (n - 1.5) * 390
            y = 410 + 24 * ru
            d.rounded_rectangle([cx - 175, y, cx + 175, y + 270], 18, fill=L.PANEL + (int(220 * ca),))
            d.rounded_rectangle([cx - 175, y, cx + 175, y + 8], 4, fill=POSC[p] + (int(255 * ca),))
            d.text((cx, y + 105), ordinal(REPL[p]), font=L.bc(118, "Black"), fill=INK + (int(255 * ca),), anchor="mm")
            d.text((cx, y + 190), f"BEST {p}", font=L.bc(38, "Bold"), fill=POSC[p] + (int(255 * ca),), anchor="mm")
            d.text((cx, y + 235), why[p], font=L.inter(19, "Regular"), fill=MUTED + (int(255 * ca),), anchor="mm")
        lines(d, ["Every season is measured against that year's free players,",
                  "so 1985 and 2025 are each judged by the standards of their day."], 780, rv(4.0), rise=up(4.0))
    else:                                                          # how to read the race
        heading(d, "HOW TO READ THE RACE", 240, a, up(0))
        rows = ("Each bar is one player's season, coloured by position", "The board holds the 12 most valuable seasons so far",
                "Glowing bars are the season being played, growing week by week",
                "Under the board: this season's best so far, and where it would rank")
        for n, text in enumerate(rows):
            ra, ru = rv(0.8 + 0.9 * n), up(0.8 + 0.9 * n)
            y = 370 + 125 * n + 16 * ru
            ix = 470
            if n == 0:
                for m, p in enumerate(POSC):
                    L.bar(im, ix, y - 34 + 17 * m, ix + 160 - 24 * m, 13, POSC[p], ra)
                cx = 660
                for p in POSC:
                    d.ellipse([cx, y + 26, cx + 16, y + 42], fill=POSC[p] + (int(255 * ra),))
                    d.text((cx + 24, y + 35), p, font=L.bc(24, "Bold"), fill=INK2 + (int(255 * ra),), anchor="lm")
                    cx += 24 + d.textlength(p, font=L.bc(24, "Bold")) + 30
            elif n == 1:
                for m in range(4):
                    L.bar(im, ix, y - 34 + 17 * m, ix + 170 - 28 * m, 13, POSC["RB"], ra)
            elif n == 2:
                L.glow(im, (ix, y - 18, ix + 130, y + 18), POSC["WR"], 0.7 * ra, 14)
                L.bar(im, ix, y - 18, ix + 130, 36, POSC["WR"], ra)
            else:
                for xx in range(ix - 10, ix + 180, 14):
                    d.line([(xx, y - 30), (xx + 6, y - 30)], fill=L.GRID + (int(255 * ra),), width=2)
                L.glow(im, (ix, y - 14, ix + 90, y + 14), POSC["RB"], 0.6 * ra, 12)
                L.bar(im, ix, y - 14, ix + 90, 28, POSC["RB"], ra)
            d.text((660, y - (8 if n == 0 else 0)), text, font=L.inter(30, "Regular"), fill=INK + (int(255 * ra),), anchor="lm")


def summary(d, s):
    """over the final standings: the top two seasons, the gap between them, the board's make-up"""
    a = smooth(s / 0.8) * (1 - smooth((s - (OUTRO - HOLD - 0.8)) / 0.8))
    if a <= 0.01:
        return
    cx = W / 2
    first, second = SUMMARY["first"], SUMMARY["second"]
    d.text((cx, 220), "THE GREATEST FANTASY SEASON EVER", font=L.bc(64, "ExtraBold"), fill=INK + (int(255 * a),), anchor="mm")
    d.text((cx, 276), f"{FIRST}–{LAST}, in points over replacement", font=L.inter(24, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for k, (r, x) in enumerate(((first, cx - 430), (second, cx + 430))):
        d.text((x, 370), f"No. {k + 1}", font=L.bc(30, "Bold"), fill=(L.GOLD if k == 0 else MUTED) + (int(255 * a),), anchor="mm")
        d.text((x, 420), r["player"].upper(), font=L.bc(56, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 466), f"{r['pos']} · {r['team']} · {r['season']}", font=L.inter(24, "Medium"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((x, 572), f"{r['vorp']:,.1f}", font=L.bc(150, "Black"), fill=(L.GOLD if k == 0 else INK) + (int(255 * a),), anchor="mm")
        d.text((x, 664), r["line"], font=L.inter(23, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")
    gap = first["vorp"] - second["vorp"]
    L.pill(d, cx, 572, f"{gap:.1f} APART" if gap < 10 else f"{num(gap)} APART", L.GOLD, size=32, anchor="m", alpha=a, pad=16)
    d.text((cx, 770), SUMMARY["mix"], font=L.inter(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    if SUMMARY.get("others"):
        d.text((cx, 816), SUMMARY["others"], font=L.inter(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for i, line in enumerate(SOURCES):
        d.text((cx, 960 + 26 * i), line, font=L.inter(17, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")


SOURCES = ("Data: Pro Football Reference season tables 1980–1998 (via fantasydatapros/data), nflverse weekly player stats 1999–2025",
           "PPR: 1 per catch, 1 per 10 rushing or receiving yards, 1 per 25 passing yards, 6 per TD (4 passing), −2 per interception",
           "or lost fumble (estimated before 1994) · two-point conversions and return TDs only from 1999 · regular seasons")


def frame(T):
    """T: seconds from the start of the film"""
    im = canvas()
    if T < INTRO - 0.2:
        intro(im, ImageDraw.Draw(im, "RGBA"), T)
        return np.asarray(im)
    a = smooth((T - INTRO + 0.2) / 0.8)
    hold = T >= D.race_end
    end = total()
    dim = 1 - 0.985 * smooth((T - (end - OUTRO + HOLD)) / 0.8)    # the board steps back for the summary
    d = board(im, T, a * dim, hold)
    header(d, a)
    season_label(d, T, a * max(0.0, dim - 0.015) / 0.985)
    timeline(d, T, a * max(0.0, dim - 0.015) / 0.985)
    caption(im, d, T, a)
    if T > end - OUTRO + HOLD:
        summary(d, T - (end - OUTRO + HOLD))
    return np.asarray(im)


def setup():
    global D, CAPS
    D = Data()
    RECORDS.clear()
    RECORD_SEASONS.clear()
    p = os.path.join(HERE, "captions.json")
    CAPS = []
    if os.path.exists(p):
        for c in json.load(open(p)):
            t0 = D.film_time(c["season"], c["week"]) + c.get("delay", 0.0)
            CAPS.append(dict(c, **{"from": t0}))
            if c.get("tag") == "NEW No. 1":
                RECORDS.append(t0)
                RECORD_SEASONS.append((c["season"] + WEEKS_FRAC * c["week"] / D.weeks[c["season"]], t0))
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
