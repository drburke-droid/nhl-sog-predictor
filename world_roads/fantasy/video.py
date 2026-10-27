"""Fantasy football's most valuable players since 2000: a bar-chart race of career value over replacement (1920x1080,
30 fps), 12-team PPR.

    python video.py still 2000.5 2006.99 ...   -> out/still_2000.50.png ... (t in seasons: 2006.5 is mid-2006;
                                                  < 2000 is the title, > 2026 the summary)
    python video.py render [jobs]              -> out/race.mp4 (chunks rendered in parallel)

The twelve players with the most career VORP (../media/fantasy/seasons.csv, weekly.csv, built by vorp.py): bars slide
past each other as the ranks change and grow week by week as each season's value accrues. Each bar carries the player,
his position and team, his career VORP, and what this season has added (or, once his career is over, its span; the bar
then fades). The scale follows the leader.

Colour encodes the position (QB magenta, RB blue, WR yellow, TE green: validated together for colour-blind separation on
this background) and every bar also names its position in text.

Seasons do not all take the same time: a season runs 2 to 12 beats of the score (1.4 to 8.4 s) by how much value the
players on the board added in it and how often they passed each other, so the busy early 2000s run longest and the
seasons in which the board stands still pass quickly. Within a season the weeks take its first 80%; the final standings hold for the rest.
"""
import os, sys, csv, json, time, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, translucent text  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "fantasy")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 7, 13], np.float32)                 # #05070d
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
POSC = {"QB": (213, 81, 129), "RB": (57, 135, 229), "WR": (201, 133, 0), "TE": (0, 131, 0)}
REPL = {"QB": 13, "RB": 36, "WR": 36, "TE": 13}
font = ST.font

FIRST, LAST = 2000, 2025
N = 12                                                # bars on the board: one per team in the league
BEAT = 0.7                                            # 85.7 BPM
INTRO, OUTRO = 12 * BEAT, 16 * BEAT                   # 8.4 s, 11.2 s
WEEKS_FRAC = 0.8                                      # the weeks take this much of a season; the final standings the rest
X0, X1 = 560, 1460                                    # bar baseline and the leader's bar end
Y0, PITCH, BAR = 178, 58, 42                          # first row's top, row spacing, bar thickness
OFF = N + 3                                           # the place of a bar that is not near the board

TEAM = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
        "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
        "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "STL": "Rams", "LAC": "Chargers",
        "SD": "Chargers", "LV": "Raiders", "OAK": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots",
        "NO": "Saints", "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
        "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Washington"}


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def num(v):
    return f"{v:,.0f}"


class Data:
    def __init__(self):
        self.sea = {}                                 # (pid, season) -> row
        for r in csv.DictReader(open(os.path.join(MEDIA, "seasons.csv"))):
            for k in ("ppr", "vorp", "career_vorp", "replacement"):
                r[k] = float(r[k])
            r["season"] = int(r["season"])
            self.sea[r["player_id"], r["season"]] = r
        self.rep = defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(MEDIA, "replacement.csv"))):
            self.rep[int(r["season"])][r["pos"]] = float(r["replacement"])
        wk = defaultdict(list)                        # (pid, season) -> [(week, team, vorp_week)]
        name = {}
        for r in csv.DictReader(open(os.path.join(MEDIA, "weekly.csv"))):
            wk[r["player_id"], int(r["season"])].append((int(r["week"]), r["team"], float(r["vorp_week"])))
            name[r["player_id"]] = r["player"]
        self.wk = wk
        self.weeks = {y: (18 if y >= 2021 else 17) for y in range(FIRST, LAST + 1)}
        # players with any value, in name order (a stable sort then breaks ties by name)
        self.pids = sorted({p for p, _ in wk}, key=lambda p: (name[p], p))
        self.name = name
        self.ix = {p: i for i, p in enumerate(self.pids)}
        self.played = defaultdict(list)               # pid -> seasons with a game
        for (p, y) in self.sea:
            if p in self.ix:
                self.played[p].append(y)
        for p in self.played:
            self.played[p].sort()
        # cumulative VORP at every week boundary of every season: C[y] is (players, weeks + 1)
        P = len(self.pids)
        base = np.zeros(P)
        self.C = {}
        for y in range(FIRST, LAST + 1):
            nw = self.weeks[y]
            inc = np.zeros((P, nw + 1))
            for (p, yy), rows in wk.items():
                if yy == y:
                    for w, _, v in rows:
                        inc[self.ix[p], min(w, nw)] += v
            self.C[y] = base[:, None] + np.cumsum(inc, axis=1)
            base = self.C[y][:, -1].copy()
        self.timeline()
        self.track()

    # ------------------------------------------------------------------ time
    def board_gain(self, y):
        """value added in season y by the players on the board at its start or its end"""
        a, b = self.C[y][:, 0], self.C[y][:, -1]
        on = {i for i in np.argsort(-a, kind="stable")[:N] if a[i] > 0} | set(np.argsort(-b, kind="stable")[:N])
        return float(sum(b[i] - a[i] for i in on))

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
        """beats per season: 2 (nothing moves) to 12, from the value added on the board (150 points a beat) and the
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

    def track(self):
        """every bar's place at every frame of the race (ranked by value, ties by name; a player with no value is off
        the board), eased in time with a Gaussian of 0.1 s so a pass takes about a third of a second; and the leader's
        value for the scale"""
        n = int(round(self.race_end * FPS)) + 1
        P = len(self.pids)
        R = np.full((P, n), float(OFF), np.float32)
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
        # the scale: the leader's value with room for his label, smoothed both ways in time, never under 60
        k = int(0.9 * FPS)
        ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.2)) ** 2)
        ker /= ker.sum()
        sm = np.convolve(np.pad(lead, k, mode="edge"), ker, mode="valid")
        self.scale = np.maximum(np.maximum(sm, lead * 0.97), 60.0)

    def frame_ix(self, T):
        return int(round(min(max(T, 0), self.race_end) * FPS))

    # ------------------------------------------------------------------ labels
    def label(self, i, y, wf):
        """position, team, this season's value so far (None if he has no game this season), career span, and how far
        his bar has faded (1 once his last season is over)"""
        p = self.pids[i]
        rows = self.wk.get((p, y))
        if rows:
            done = [r for r in rows if r[0] <= max(1, int(np.ceil(wf)))]
            team = (done or rows)[-1][1]
            pos = self.sea[p, y]["pos"]
            this = self.C[y][i, min(self.weeks[y], int(np.floor(wf)))] - self.C[y][i, 0]
            if wf < self.weeks[y]:
                k = int(np.floor(wf))
                this += (self.C[y][i, k + 1] - self.C[y][i, k]) * (wf - k)
        else:                                         # no game this season: the team he played the most games for
            last = max((s for s in self.played[p] if s < y), default=None)
            nxt = min((s for s in self.played[p] if s > y), default=None)
            src = last if last is not None else nxt
            games = defaultdict(int)
            for s in self.played[p]:
                if s <= src:
                    for w, t, _ in self.wk.get((p, s), []):
                        games[t] += 1
            team = max(games, key=lambda t: (games[t], t)) if games else self.sea[p, src]["team"]
            pos = self.sea[p, src]["pos"]
            this = None
        seen = [s for s in self.played[p] if s <= y]
        span = (seen[0], seen[-1]) if seen else None
        return pos, TEAM.get(team, team), this, span

    def faded(self, i, y, f):
        last = self.played[self.pids[i]][-1]
        if last >= LAST:
            return 0.0
        return smooth(((y - last - 1) + f) / 0.25) if y > last else 0.0


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


def board(im, T, a=1.0, hold=False):
    d = ImageDraw.Draw(im, "RGBA")
    j = D.frame_ix(T)
    xmax = D.scale[min(j, len(D.scale) - 1)]
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    ticks(d, xmax, a)
    vals, y, wf, f = D.values(min(T, D.race_end))
    st = {}
    for i in D.near:
        r = float(D.pos[int(i)][min(j, len(D.pos[int(i)]) - 1)])
        if r < N - 0.15 and vals[i] > 0:
            st[int(i)] = (r, float(vals[i]))
    # longest first, so a shorter bar passing a longer one stays in view on top of it; while two bars overlap, the
    # lower one's labels give way
    for i, (r, v) in sorted(st.items(), key=lambda kv: (-kv[1][1], kv[1][0])):
        al = a * smooth((N - 0.15 - r) / 0.7) * smooth(v / 4)
        if al <= 0.01:
            continue
        near = min((abs(r - r2) for i2, (r2, v2) in st.items() if i2 != i and (v2, -r2) > (v, -r)), default=9)
        tl = al * smooth((near - 0.3) / 0.4)
        pos, team, this, span = D.label(i, y, wf)
        fd = D.faded(i, y, f)
        yy = Y0 + r * PITCH
        col = POSC[pos]
        x1 = max(X0 + 6, xs(v))
        ba = al * (1 - 0.55 * fd)
        d.rounded_rectangle([X0, yy, x1, yy + BAR], 4, fill=col + (int(255 * ba),))
        d.rectangle([X0, yy, min(x1, X0 + 6), yy + BAR], fill=col + (int(255 * ba),))
        if r < 0.5 and not hold:                     # the leader
            d.rounded_rectangle([X0 - 1, yy - 1, x1 + 1, yy + BAR + 1], 5, outline=INK + (int(110 * al * (1 - r * 2)),), width=2)
        if tl <= 0.01:
            continue
        ta = tl * (1 - 0.3 * fd)
        d.text((X0 - 18, yy + BAR / 2 - 10), D.name[D.pids[i]], font=font(27, "SemiBold"), fill=INK + (int(255 * ta),), anchor="rm")
        d.text((X0 - 18, yy + BAR / 2 + 14), f"{pos} · {team}", font=font(18, "Regular"), fill=INK2 + (int(235 * ta),), anchor="rm")
        d.text((x1 + 14, yy + BAR / 2 - 8), num(v), font=font(27, "Bold"), fill=INK + (int(255 * ta),), anchor="lm")
        if this is not None:
            sub = f"+{num(this)} in {y}"
        elif span:
            back = any(s > y for s in D.played[D.pids[i]])     # a season off, not the end of his career
            sub = f"no games in {y}" if back else f"{span[0]}–{span[1]}"
        else:
            sub = ""
        d.text((x1 + 14, yy + BAR / 2 + 15), sub, font=font(18, "Regular"), fill=MUTED + (int(235 * ta),), anchor="lm")
    return d


def header(d, a=1.0):
    d.text((90, 62), "Fantasy football's most valuable players", font=font(46, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((90, 104), "Career points over a replacement player since 2000 · 12-team PPR", font=font(25, "Regular"),
           fill=INK2 + (int(255 * a),), anchor="ls")
    x = W - 90
    for p in ("TE", "WR", "RB", "QB"):
        w = d.textlength(p, font=font(22, "SemiBold"))
        d.text((x, 62), p, font=font(22, "SemiBold"), fill=INK + (int(255 * a),), anchor="rs")
        d.rounded_rectangle([x - w - 30, 44, x - w - 10, 64], 4, fill=POSC[p] + (int(255 * a),))
        x -= w + 58
    d.text((W - 90, 100), "faded bar: career over", font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="rs")


def season_label(d, T, a=1.0):
    vals, y, wf, f = D.values(min(T, D.race_end))
    d.text((W - 90, H - 86), str(y), font=font(132, "Bold"), fill=INK + (int(235 * a),), anchor="rs")
    wk = "Final" if wf >= D.weeks[y] else f"Week {max(1, int(np.ceil(wf)))}"
    d.text((W - 92, H - 46), wk, font=font(28, "SemiBold"), fill=INK2 + (int(255 * a),), anchor="rs")
    rep = D.rep[y]
    line = "   ".join(f"{p}{REPL[p]} {num(rep[p])}" for p in ("QB", "RB", "WR", "TE"))
    d.text((X0, Y0 + N * PITCH + 46), f"Replacement level in {y}, season PPR points:   {line}", font=font(20, "Regular"),
           fill=MUTED + (int(255 * a),), anchor="ls")


def caption(d, T, a=1.0):
    for e in CAPS:
        t0, t1 = e["from"], e["to"]
        al = a * smooth((T - t0) / 0.35) * (1 - smooth((T - t1) / 0.35))
        if al <= 0.01:
            continue
        lines = e["text"]
        y = H - 96
        for i, line in enumerate(lines):
            d.text((90, y + i * 38), line, font=font(30 if i == 0 else 24, "SemiBold" if i == 0 else "Regular"),
                   fill=(INK if i == 0 else INK2) + (int(255 * al),), anchor="ls")


def intro(d, T):
    a = smooth(T / 0.8) * (1 - smooth((T - INTRO + 0.9) / 0.8))
    cx = W / 2
    d.text((cx, 330), "Fantasy football's most valuable players", font=font(68, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
    d.text((cx, 410), f"Career value over replacement, {FIRST} to {LAST} · 12-team PPR league", font=font(34, "Regular"),
           fill=INK2 + (int(255 * a),), anchor="mm")
    b = a * smooth((T - 1.6) / 0.8)
    lines = ("Each season, a player scores his PPR points minus the replacement level at his position:",
             "what the best player left on waivers scored. In a 12-team league that is the",
             "13th-best QB and TE and the 36th-best RB and WR. A season below that counts zero.")
    for k, line in enumerate(lines):
        d.text((cx, 540 + 46 * k), line, font=font(29, "Regular"), fill=INK2 + (int(255 * b),), anchor="mm")
    x = cx - 2 * 150 + 75
    for p in ("QB", "RB", "WR", "TE"):
        d.rounded_rectangle([x - 52, 742, x - 30, 764], 4, fill=POSC[p] + (int(255 * b),))
        d.text((x - 20, 762), f"{p}{REPL[p]}", font=font(26, "SemiBold"), fill=INK + (int(255 * b),), anchor="ls")
        x += 150


def summary(d, s):
    """over the final standings: the leader against the closest active player, and the board's make-up"""
    a = smooth(s / 0.8) * (1 - smooth((s - (OUTRO - 2.4)) / 0.8))
    if a <= 0.01:
        return
    cx = W / 2
    lead, chase = SUMMARY["lead"], SUMMARY["chase"]
    d.text((cx, 250), f"Career points over replacement, {FIRST}–{LAST}", font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for i, (r, x) in enumerate(((lead, cx - 400), (chase, cx + 400))):
        d.text((x, 350), r["player"], font=font(44, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 402), f"{r['pos']} · {r['teams']} · {r['span']}", font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((x, 506), num(r["vorp"]), font=font(96, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 580), r["note"], font=font(26, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")
    d.text((cx, 690), SUMMARY["gap"], font=font(36, "SemiBold"), fill=INK + (int(255 * a),), anchor="mm")
    d.text((cx, 760), SUMMARY["mix"], font=font(28, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for i, line in enumerate(SOURCES):
        d.text((cx, 996 + 28 * i), line, font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")


SOURCES = ("Data: nflverse weekly player stats and rosters, regular seasons 2000–2025 · PPR: 1 per catch, 1 per 10 rushing or",
           "receiving yards, 1 per 25 passing yards, 6 per TD (4 passing), 2 per two-point conversion, −2 per interception or lost fumble")


def frame(T):
    """T: seconds from the start of the film"""
    im = canvas()
    if T < INTRO - 0.2:
        intro(ImageDraw.Draw(im, "RGBA"), T)
        return np.asarray(im)
    a = smooth((T - INTRO + 0.2) / 0.8)
    hold = T >= D.race_end
    end = total()
    dim = 1 - 0.94 * smooth((T - (end - OUTRO + 1.6)) / 0.8)      # the board steps back for the summary
    d = board(im, T, a * dim, hold)
    header(d, a)
    season_label(d, T, a * max(0.0, dim - 0.06) / 0.94)
    caption(d, T, a)
    if T > end - OUTRO + 1.6:
        summary(d, T - (end - OUTRO + 1.6))
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
        CAPS[-1]["to"] = min(CAPS[-1]["to"], D.race_end + 1.4)
    s = json.load(open(os.path.join(HERE, "summary.json"))) if os.path.exists(os.path.join(HERE, "summary.json")) else None
    if s:
        SUMMARY.update(s)


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
