"""Trades and free agents by time horizon: what a move gives a team now and later, and what it costs:  python horizon.py
-> ../media/rebuild/trade_horizon.csv, team_horizon.csv

A star is a star whatever he is paid, but a 33-year-old in the last year of his deal and a 25-year-old with four
cheap years left are not the same asset, and a trade that sheds salary or matches it is not a sale. So every player,
prospect and pick that moves (NBA, NHL, MLB; the NFL has no production measure here, see moves.py) gets a projected
value, in stars (moves.py), for each of the six seasons from the move, counted only while the team that got him had
him, and split into *now* (the season of the move, what is left of it for a mid-season deal, and the next) and
*later* (the four seasons after that):

* A player with a track record (a season in the league in the last two): his last three seasons weighted 5-4-3 (a
  shortened season scaled to a full one), carried forward by his league's aging curve, fitted on every player and
  season 2001-2025 (value j seasons on, 0 if he is out of the league, against the track record going in, a straight
  line for each age band and j).
* A prospect (no track record, and young; or a minor leaguer named in the trade): what traded prospects of his age in
  his league went on to give the teams that got them, on average, season by season.
* A pick: what picks in that round of the league's draft (2000-2025) went on to produce, season by season after the
  draft.
* Control: a veteran's projection counts while he stayed with the team that got him (until he first played for
  someone else), which stands in for the years of contract or control it bought; a rental counts for the rest of the
  season, a player with four years of control for four. A prospect's or pick's average already holds that.
* Cost: each player's salary that season as a share of the league's average payroll (NBA 2001-2025, NHL cap hits
  2010-2025, MLB 2001-2016), so a team that sheds salary without taking value back can be told from one that sells.

team_horizon.csv: per franchise and season, value traded in and out, now and later; free agents signed and lost, now
and later; salary traded in and out.
"""
import os, csv, re
from collections import defaultdict
import numpy as np
from build import DATA, OUT, page, table, text
from moves import COVER, PROSPECT_AGE, pid_of

LEAGUES = ["nba", "nhl", "mlb"]
AHEAD = 6                                   # seasons projected: the season of the move and five more
NOW = 2                                     # of which 'now': the first two
BANDS = [(0, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 99)]
LEFT = dict(nba={10: 0.9, 11: 0.8, 12: 0.65, 1: 0.5, 2: 0.35, 3: 0.2, 4: 0.05, 5: 0.0},     # share of the season still to
            nhl={10: 0.9, 11: 0.8, 12: 0.65, 1: 0.5, 2: 0.35, 3: 0.2, 4: 0.05, 5: 0.0},     # play after a mid-season move
            mlb={4: 0.9, 5: 0.75, 6: 0.6, 7: 0.4, 8: 0.25, 9: 0.1, 10: 0.0})
F = lambda v: float(v) if v not in ("", None) else None
NHL_CAP = {2006: 39.0, 2007: 44.0, 2008: 50.3, 2009: 56.7, 2010: 56.8, 2011: 59.4, 2012: 64.3, 2013: 70.2}  # $M, upper limit


def band(age):
    return next(i for i, (a, b) in enumerate(BANDS) if a <= age <= b)


# shortened seasons, scaled to a full one: MLB 2020 (60 games), NHL 2012-13 (48), 2019-20 (about 70), 2020-21 (56),
# NBA 2011-12 (66), 2019-20 (about 70), 2020-21 (72)
SHORT = {("mlb", 2020): 162 / 60, ("nhl", 2013): 82 / 48, ("nhl", 2020): 82 / 70, ("nhl", 2021): 82 / 56,
         ("nba", 2012): 82 / 66, ("nba", 2020): 82 / 70, ("nba", 2021): 82 / 72}
WEIGHTS = (5, 4, 3)                         # the last three seasons, most recent first (as in Tango's Marcel)


class League:
    def __init__(self, lg, rows):
        self.lg = lg
        self.v, self.age, self.teams, self.sal = {}, {}, {}, {}
        for r in rows:
            k = (r["pid"], int(r["season"]))
            self.v[k] = float(r["stars"]) * SHORT.get((lg, k[1]), 1.0)
            if r["age"]:
                self.age[k] = int(float(r["age"]))
            self.teams[k] = r["teams"].split("|") if r["teams"] else []
            if r["salary_musd"]:
                self.sal[k] = float(r["salary_musd"])
        self.seasons = sorted({s for _, s in self.v})
        self.index = {s: i for i, s in enumerate(self.seasons)}
        self.last = max(self.seasons)
        self.by_player = defaultdict(list)
        for pid, s in self.v:
            self.by_player[pid].append(s)
        self.fit()

    def exists(self, s):
        return s in self.index                           # the NHL lost 2004-05

    def before(self, s, n=3):
        """the league's n seasons before s"""
        i = self.index.get(s, len([x for x in self.seasons if x < s]))
        return self.seasons[max(0, i - n):i][::-1]

    def base(self, pid, s):
        """his track record going into season s: the last three seasons weighted 5-4-3 (those he played), and his age
        in the season before s; None unless he played in one of the two seasons before"""
        prev = self.before(s)
        got = [(w, self.v[(pid, b)], b) for w, b in zip(WEIGHTS, prev) if (pid, b) in self.v]
        if not got or got[0][2] not in prev[:2]:
            return None
        ages = [(b, self.age[(pid, b)]) for _, _, b in got if (pid, b) in self.age]
        if not ages:
            return None
        b0, a0 = ages[0]
        return sum(w * v for w, v, _ in got) / sum(w for w, _, _ in got), a0 + (len([x for x in self.seasons if b0 < x < s]))

    def fit(self):
        """value j seasons from s against the track record going in, by age band: a + b * base, with 0 for a player out
        of the league (fitted on every player and season with a track record)"""
        pairs = defaultdict(list)
        for pid, ss in self.by_player.items():
            for s in sorted(set(ss) | {x + 1 for x in ss}):
                if not self.exists(s):
                    continue
                bs = self.base(pid, s)
                if bs is None:
                    continue
                x, age = bs
                for j in range(AHEAD):
                    t = s + j
                    if t > self.last or not self.exists(t):
                        continue
                    pairs[(band(age), j)].append((x, self.v.get((pid, t), 0.0)))
        self.coef = {}
        for key, xy in pairs.items():
            x, y = np.array(xy).T
            b, a = np.polyfit(x, y, 1) if len(xy) > 30 else (0.0, float(np.mean(y)))
            self.coef[key] = (a, b)

    def project(self, pid, s):
        """projected stars in seasons s .. s+AHEAD-1, or None if he has no track record"""
        bs = self.base(pid, s)
        if bs is None:
            return None
        x, age = bs
        return [sum(self.coef.get((band(age), j), (0.0, 0.0)) * np.array([1.0, x])) for j in range(AHEAD)]

    def tenure(self, pid, team, s):
        """1 for each season from s on that he was the team's (absent seasons count if he came back to it)"""
        w, pending = [], 0
        for t in range(s, s + AHEAD):
            if t > self.last:                            # beyond the data: assume he stayed
                w += [1.0] * (pending + 1)
                pending = 0
                continue
            tm = self.teams.get((pid, t))
            if tm is None:
                pending += 1
                continue
            if team in tm:
                w += [1.0] * pending + [1.0]
                pending = 0
            else:
                break
        w += [0.0] * (AHEAD - len(w))
        return w[:AHEAD]

    def salary(self, pid, s):
        return self.sal.get((pid, s), self.sal.get((pid, s - 1)))


def expected_prospects(L, moves):
    """what traded prospects gave the teams that got them, by age band, season by season from the trade"""
    acc = defaultdict(lambda: np.zeros(AHEAD))
    n = defaultdict(int)
    for m in moves:
        if m["asset"] != "prospect" or int(m["season"]) + AHEAD - 1 > L.last:
            continue
        s, pid = int(m["season"]), m["pid"]
        if pid and L.project(pid, s) is not None:     # he has a track record: projected, not averaged
            continue
        key = band(int(float(m["age"]))) if m["age"] else "unknown"
        got = np.zeros(AHEAD)
        if pid:
            w = L.tenure(pid, m["to_team"], s)
            got = np.array([L.v.get((pid, s + j), 0.0) * w[j] for j in range(AHEAD)])
        acc[key] += got
        n[key] += 1
    # a prospect named without an age is one the data never saw play (Retrosheet ids, NHL names unmatched): count
    # them as the busts of every age band, in proportion, so each band's average is over all prospects of that age
    known = sum(v for k, v in n.items() if k != "unknown")
    unk = n.pop("unknown", 0)
    acc.pop("unknown", None)
    out = {k: acc[k] / (n[k] + unk * n[k] / known) for k in n}
    out["unknown"] = sum(acc.values()) / (known + unk)
    return out


def expected_picks(L):
    """what each round of the draft produced, by season after the draft (the draft of June Y plays from Y+1 on)"""
    acc = defaultdict(lambda: np.zeros(AHEAD + 1))
    n = defaultdict(int)
    if L.lg == "mlb":                                    # MLB trades carry no picks here
        return {}
    for y in range(2000, L.last - AHEAD + 1):
        h = page("sr", L.lg, f"draft_{y}.html")
        m = re.search(r'<table[^>]*id="stats".*?</table>', h, re.S)
        rnd = 1
        for cls, body in re.findall(r"<tr([^>]*)>(.*?)</tr>", m.group(0), re.S)[1:]:
            if "thead" in cls or "over_header" in cls:
                r = re.search(r"Round (\d+)", body)
                rnd = int(r.group(1)) if r else rnd
                continue
            cells = dict(re.findall(r'<t[hd][^>]*data-stat="([^"]+)"[^>]*>(.*?)</t[hd]>', body, re.S))
            if not text(cells.get("pick_overall", "")).isdigit():
                continue
            pid = pid_of(cells.get("player", "") or cells.get("player_name", ""))
            acc[min(rnd, 7)] += np.array([L.v.get((pid, y + j), 0.0) if pid else 0.0 for j in range(1, AHEAD + 2)])
            n[min(rnd, 7)] += 1
    return {r: acc[r] / n[r] for r in acc}


def left(lg, phase, date):
    if phase != "in" or not date or date[5:7] in ("", "00"):
        return 1.0
    return LEFT[lg].get(int(date[5:7]), 1.0)


def value_of(L, m, prospects, picks, team):
    """projected stars for `team` (who got the asset), seasons s .. s+5"""
    s = int(m["season"])
    if m["asset"] == "pick":
        if not picks:
            return np.zeros(AHEAD)
        rnd = int(m["pick_round"] or 0) or 4
        e = picks.get(min(rnd, max(picks)), np.zeros(AHEAD + 1))
        y = int(m["pick_year"])
        return np.array([e[t - y - 1] if 0 <= t - y - 1 <= AHEAD else 0.0 for t in range(s, s + AHEAD)])
    pid = m["pid"]
    proj = L.project(pid, s) if pid else None
    if proj is None:
        age = int(float(m["age"])) if m["age"] else None
        if age is not None and age > PROSPECT_AGE[L.lg] + 2:   # a veteran without a recent season: nothing expected
            return np.zeros(AHEAD)
        e = prospects.get(band(age) if age is not None else "unknown")
        return np.array(e) if e is not None else np.zeros(AHEAD)
    return np.array(proj) * np.array(L.tenure(pid, team, s))


def main():
    players = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, "players.csv"))):
        players[r["league"]].append(r)
    moves = list(csv.DictReader(open(os.path.join(OUT, "moves.csv"))))
    pay = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, "payroll.csv"))):
        pay[(r["league"], int(r["season"]))].append(float(r["payroll_musd"]))
    avg_pay = {k: np.mean(v) for k, v in pay.items()}
    for y, cap in NHL_CAP.items():                       # NHL payroll is rebuilt from 2014; before, the cap stands in
        avg_pay.setdefault(("nhl", y), cap)
    assets, agg = [], defaultdict(lambda: defaultdict(float))
    for lg in LEAGUES:
        L = League(lg, players[lg])
        M = [m for m in moves if m["league"] == lg]
        prospects = expected_prospects(L, [m for m in M if m["kind"] == "trade"])
        picks = expected_picks(L)
        print(lg, "prospects (stars, six seasons):", {k: round(float(v.sum()), 2) for k, v in sorted(prospects.items(), key=str)})
        print(lg, "picks by round (stars, seasons 1-6 after the draft):", {k: round(float(v[:AHEAD].sum()), 2) for k, v in sorted(picks.items())})
        for m in M:
            s = int(m["season"])
            if m["kind"] == "trade" and m["asset"] not in ("player", "prospect", "pick"):
                continue
            v = value_of(L, m, prospects, picks, m["to_team"])
            v[0] *= left(lg, m["phase"], m["date"])
            now, later = float(v[:NOW].sum()), float(v[NOW:].sum())
            sal = L.salary(m["pid"], s) if m["pid"] and m["asset"] != "pick" else None
            ap = avg_pay.get((lg, s)) or avg_pay.get((lg, s - 1))
            sal_share = sal / ap if sal is not None and ap else None
            assets.append(dict(league=lg, season=s, date=m["date"], phase=m["phase"], kind=m["kind"], trade_id=m["trade_id"],
                               from_team=m["from_team"], to_team=m["to_team"], asset=m["asset"], pid=m["pid"], name=m["name"],
                               age=m["age"], stars_last=m["stars"], now=round(now, 3), later=round(later, 3),
                               salary_share=round(sal_share, 4) if sal_share is not None else "",
                               pick_year=m["pick_year"], pick_round=m["pick_round"]))
            pre = "trade" if m["kind"] == "trade" else "fa"
            for team, side in ((m["to_team"], "in"), (m["from_team"], "out")):
                if not team:
                    continue
                a = agg[(lg, s, team)]
                a[f"{pre}_now_{side}"] += now
                a[f"{pre}_later_{side}"] += later
                if pre == "trade" and sal_share is not None:
                    a[f"salary_{side}"] += sal_share
                    a["salary_known"] = 1
    cols = list(assets[0])
    with open(os.path.join(OUT, "trade_horizon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(assets)
    print(f"trade_horizon.csv: {len(assets)} rows")
    seasons = list(csv.DictReader(open(os.path.join(OUT, "seasons.csv"))))
    keys = ["trade_now_in", "trade_now_out", "trade_later_in", "trade_later_out", "fa_now_in", "fa_now_out",
            "fa_later_in", "fa_later_out", "salary_in", "salary_out"]
    out = []
    for r in seasons:
        lg, s, t = r["league"], int(r["season"]), r["team"]
        if lg not in LEAGUES:
            continue
        a = agg[(lg, s, t)]
        has_tr = COVER["trades"][lg][0] <= s <= COVER["trades"][lg][1]
        has_fa = COVER["fa"][lg][0] <= s <= COVER["fa"][lg][1]
        has_sal = dict(nba=(2001, 2025), nhl=(2010, 2025), mlb=(2001, 2016))[lg]
        has_sal = has_sal[0] <= s <= has_sal[1]
        row = dict(league=lg, season=s, team=t)
        for k in keys:
            ok = has_sal if k.startswith("salary") else has_tr if k.startswith("trade") else has_fa
            row[k] = round(a[k], 3) if ok else ""
        out.append(row)
    with open(os.path.join(OUT, "team_horizon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"team_horizon.csv: {len(out)} rows")


if __name__ == "__main__":
    main()
