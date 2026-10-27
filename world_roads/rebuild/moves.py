"""Payroll, free agents and trades, per franchise and season:  python moves.py
-> ../media/rebuild/players.csv, moves.csv, payroll.csv, team_moves.csv

* players.csv: every player's value each season (with the teams he played for, and his salary where known: NBA
  2001-2025, NHL 2010-2025 cap hits, MLB 2001-2016, NFL cap hits), in his league's own currency (NBA win shares, NHL
  point shares, goalies scaled to skaters by each group's star line; MLB wins above replacement; NFL: his cap hit as
  a share of the salary cap, the only measure on hand), and in *stars*: that value over the league's star line, the
  top 5 % of all player-seasons 2001-2025 (1.0 = an all-star season).
* moves.csv: every player and pick traded (NBA 2001-2025, NHL 2001-2025, MLB 2001-2021, NFL 2002-2024) and every
  free agent signed from another team (NBA 2001-2025, NHL 2010-2025, MLB 2001-2021, NFL 2013-2024), with the player's
  track record (his value the season before, in stars; an NFL free agent's new deal as a share of the cap) and age.
* payroll.csv: each roster's payroll (NBA, team pages, and MLB, team salaries, 2001-2025; NHL 2014-2025, rebuilt from
  every contract's cap hit and who played where; NFL 2013-2024, cap charges) and its ratio to the league's average
  that season.
* team_moves.csv: per franchise and season, the stars traded in and out, the picks and prospects traded in and out,
  the stars signed and lost in free agency, and payroll against the league.

Seasons as in seasons.csv: NHL and NBA by the year they end, MLB and NFL by the year they are played. A move counts
for the season it prepares or happens in: NHL and NBA moves from mid-June on (the draft, free agency) count for the
next season; MLB moves in November and December for the next year.
"""
import os, re, csv, json, html, zipfile, difflib, unicodedata, warnings
from collections import defaultdict
from statistics import median
import numpy as np
from teams import FRANCHISES, code_map
from build import DATA, OUT, page, text, table, team_code, write

YEARS = range(2001, 2026)
STAR_Q = 0.95                      # the star line: this quantile of every player-season's value in the league
PROSPECT_AGE = dict(nba=23, nhl=23, mlb=25, nfl=24)
PICK_W = {1: 1.0, 2: 0.35}         # future assets: a first-round pick counts 1, a second 0.35, later rounds 0.1,
OTHER_PICK, PROSPECT_W = 0.1, 0.5  # a prospect (young, no track record yet) 0.5
COVER = dict(trades=dict(nba=(2001, 2025), nhl=(2001, 2025), mlb=(2001, 2021), nfl=(2002, 2024)),
             fa=dict(nba=(2001, 2025), nhl=(2010, 2025), mlb=(2001, 2021), nfl=(2013, 2024)),
             payroll=dict(nba=(2001, 2025), nhl=(2014, 2025), mlb=(2001, 2025), nfl=(2013, 2024)))
MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                          "September", "October", "November", "December"])}


# ---------------------------------------------------------------- names, dates, seasons

def key(name):
    """a name folded for matching: Mikkel Bødker, Mikkel Boedker -> 'mikkelbodker'"""
    s = name.lower()
    for a, b in (("ø", "o"), ("æ", "ae"), ("ß", "ss"), ("ł", "l"), ("đ", "d")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("oe", "o"), ("ae", "a"), ("ue", "u")):
        s = s.replace(a, b)
    return s


def season_of(league, y, m, d=1):
    """the season a move on this date counts for"""
    if league in ("nba", "nhl"):
        return y + 1 if (m, d) >= (6, 15) else y
    if league == "mlb":
        return y + 1 if m >= 11 else y
    return y


def phase_of(league, m):
    if league in ("nba", "nhl"):
        return "in" if m >= 10 or m <= 5 else "off"
    if league == "mlb":
        return "in" if 4 <= m <= 10 else "off"
    return "in" if m >= 9 else "off"


# ---------------------------------------------------------------- player values

class Players:
    """value[(pid, season)] = total; stints[(pid, season)] = [(team, games, value)] in the order played"""

    def __init__(self, league):
        self.league, self.value, self.age, self.stints, self.name = league, {}, {}, defaultdict(list), {}
        self.by_key, self.goalies = defaultdict(set), set()

    def add(self, pid, name, season, age, team, games, value, total_row=False):
        k = (pid, season)
        self.name[pid] = name
        self.by_key[key(name)].add(pid)
        if age is not None:
            self.age[k] = age
        if total_row:
            self.value[k] = value
        else:
            self.stints[k].append((team, games, value))

    def finish(self):
        for k, st in self.stints.items():
            self.value.setdefault(k, sum(v for _, _, v in st))
        line = lambda vals: sorted(vals)[int(STAR_Q * (len(vals) - 1))]
        if self.goalies:                      # goalies against goalies: point shares rate them far above skaters
            f = line([v for (p, _), v in self.value.items() if p not in self.goalies]) / line(
                [v for (p, _), v in self.value.items() if p in self.goalies])
            for k in self.value:
                if k[0] in self.goalies:
                    self.value[k] *= f
                    self.stints[k] = [(t, g, v * f) for t, g, v in self.stints.get(k, [])]
        self.star = line(list(self.value.values()))
        self.seasons = defaultdict(set)
        for pid, s in self.value:
            self.seasons[pid].add(s)

    def prev(self, pid, season):
        """his track record before a season: the season before (the one before that if he missed it)"""
        for s in (season - 1, season - 2) if self.league != "nhl" else (season - 1, season - 2, season - 3):
            if (pid, s) in self.value:
                return self.value[(pid, s)]
        return 0.0

    def age_at(self, pid, season):
        for d in range(0, 6):
            for s in (season - d, season + d):
                if (pid, s) in self.age:
                    return self.age[(pid, s)] + (season - s)
        return None

    def last_team(self, pid, season):
        st = self.stints.get((pid, season))
        return st[-1][0] if st else None

    def teams_in(self, pid, season):
        return {t for t, _, _ in self.stints.get((pid, season), [])}

    def find(self, name, season, teams=(), window=3):
        """the player this name means: one who was with one of these teams around then, else the only one by the name
        who played around then"""
        cands = self.by_key.get(key(name), set())
        if not cands:
            return None
        near = [p for p in cands if any(abs(s - season) <= window for s in self.seasons.get(p, ()))]
        fit = [p for p in near if any(set(teams) & self.teams_in(p, s) for s in range(season - 1, season + 2))]
        if len(fit) == 1:
            return fit[0]
        if len(near) == 1:
            return near[0]
        if fit:
            return fit[0]
        return None

    def fuzzy(self, name, season, teams):
        """a spelling the exact match missed (Ozolinsh, Ozoliņš): the closest name among those teams' players then"""
        pool = {p for (p, s), st in self.stints.items() if abs(s - season) <= 1 and any(t in teams for t, _, _ in st)}
        k = key(name)
        best = max(pool, key=lambda p: difflib.SequenceMatcher(None, k, key(self.name[p])).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, k, key(self.name[best])).ratio() >= 0.85:
            return best
        return None

    def match(self, name, season, teams=(), window=3):
        return self.find(name, season, teams, window) or (self.fuzzy(name, season, teams) if teams else None)


def pid_of(cell):
    m = re.search(r"/players/\w/([a-z0-9.'-]+)\.s?html", cell)
    return m.group(1) if m else None


def num(cell, default=0.0):
    t = text(cell).replace(",", "").replace("$", "")
    try:
        return float(t)
    except ValueError:
        return default


def sr_players(league, cm):
    P = Players(league)
    specs = dict(nba=[("advanced_{y}", "advanced", "games", "ws")],
                 nhl=[("misc_{y}", None, "games", "ps"), ("goalies_{y}", "goalie_stats", "goalie_games", "gps")],
                 mlb=[("value_batting_{y}", "players_value_batting", "b_pa", "b_war"),
                      ("value_pitching_{y}", "players_value_pitching", "p_ip", "p_war")])[league]
    for y in YEARS:
        if league == "nhl" and y == 2005:
            continue
        acc = defaultdict(lambda: [None, None, 0.0, False, []])   # pid -> name, age, total, has_total, stints
        for fname, tid, gcol, vcol in specs:
            h = page("sr", league, fname.format(y=y) + ".html").replace("<!--", "").replace("-->", "")
            if tid is None:
                tid = next(t for t in re.findall(r'<table[^>]*id="([^"]+)"', h) if "post" not in t)
            for r in table(h, tid):
                pid = pid_of(r.get("name_display", ""))
                if not pid:
                    continue
                team = text(r.get("team_name_abbr", ""))
                a = acc[pid]
                a[0] = text(r["name_display"]).rstrip("*#+ ")
                age = text(r.get("age", ""))
                a[1] = int(age) if age.isdigit() else a[1]
                v = num(r.get(vcol, ""))
                if vcol == "gps":
                    P.goalies.add(pid)
                if re.fullmatch(r"TOT|\d+TM", team):
                    a[2] += v
                    a[3] = True
                elif team in cm:
                    a[4].append((cm[team], num(r.get(gcol, "")), v))
        for pid, (name, age, tot, has_tot, stints) in acc.items():
            merged = {}
            for t, g, v in stints:                   # the batting and pitching rows of one stint are one stint
                if t in merged:
                    merged[t] = (merged[t][0] + g, merged[t][1] + v)
                else:
                    merged[t] = (g, v)
            for t, (g, v) in merged.items():
                P.add(pid, name, y, age, t, g, v)
            if has_tot or not stints:
                P.add(pid, name, y, age, None, 0, tot if has_tot else 0.0, total_row=True)
    P.finish()
    return P


# ---------------------------------------------------------------- NBA

def nba_moves(P):
    cm = code_map("nba")
    trades, fas = [], []
    for y in YEARS:
        h = page("sr", "nba", f"transactions_{y}.html")
        seg = h[h.find("page_index"):]
        for date, body in re.findall(r"<li><span>([^<]+)</span>(.*?)</li>", seg, re.S):
            m = re.match(r"(\w+) (\d+), (\d{4})", date.strip())
            if not m:
                continue
            mo, dd, yy = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
            s, ph = season_of("nba", yy, mo, dd), phase_of("nba", mo)
            for k, p in enumerate(re.findall(r"<p>(.*?)</p>", body, re.S)):
                if " traded " in p:
                    tid = f"{yy}{mo:02d}{dd:02d}-{k}"
                    for frm, to, items in nba_trade_parts(p):
                        if frm in cm and to in cm:
                            for it in items:
                                trades.append(dict(it, trade_id=tid, season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=ph,
                                                   from_team=cm[frm], to_team=cm[to]))
                elif re.search(r"\bsigned\b", p) and "re-signed" not in p and "extension" not in p:
                    to = re.search(r'data-attr-to="(\w+)"', p)
                    pid = pid_of(p)
                    if to and pid and to.group(1) in cm:
                        fas.append(dict(season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=ph, to_team=cm[to.group(1)], pid=pid))
    return trades, fas


def nba_items(t):
    """players and picks named in a stretch of a trade sentence"""
    t = re.sub(r"\([^()]*\)", "", t)                       # '(OG Anunoby was later selected)' and other notes
    items = [dict(asset="player", pid=pid_of(a)) for a in re.findall(r'<a href="/players/[^"]+"', t)]
    for yr, rd in re.findall(r"(\d{4}) (\d)(?:st|nd|rd|th) round draft pick", t):
        items.append(dict(asset="pick", pick_year=int(yr), pick_round=int(rd)))
    return items


def nba_trade_parts(p):
    """(from, to, items) for each leg: 'the A traded X to the B for Y' or '...; the B traded Z to the C; ...'"""
    out = []
    for leg in re.split(r";", p):
        frm = re.search(r'data-attr-from="(\w+)"', leg)
        to = re.search(r'data-attr-to="(\w+)"', leg)
        if not (frm and to):
            continue
        a, b = leg.find("traded", frm.end()), to.start()
        if a < 0:
            continue
        out.append((frm.group(1), to.group(1), nba_items(leg[a:b])))
        back = re.search(r"</a>\s*for\b(.*)", leg[to.end():], re.S)
        if back:
            out.append((to.group(1), frm.group(1), nba_items(back.group(1))))
    return out


def nba_payroll():
    """each roster's payroll, and each player's salary (the team pages' salary tables)"""
    cm = code_map("nba")
    out, sal = {}, defaultdict(float)
    for y in YEARS:
        for f in os.listdir(os.path.join(DATA, "sr", "nba")):
            m = re.fullmatch(rf"team_([A-Z]{{3}})_{y}\.html", f)
            if not m or m.group(1) not in cm:
                continue
            h = page("sr", "nba", f).replace("<!--", "").replace("-->", "")
            rows = [r for r in table(h, "salaries2") if "salary" in r and "$" in r["salary"]]
            for r in rows:
                if pid_of(r.get("player", "")):
                    sal[(pid_of(r["player"]), y)] += num(r["salary"]) / 1e6
            tot = sum(num(r["salary"]) for r in rows)
            if tot:
                out[(y, cm[m.group(1)])] = tot / 1e6
    return out, dict(sal)


# ---------------------------------------------------------------- NHL

NHL_NAMES = {"Mighty Ducks of Anaheim": "ANA", "Anaheim Mighty Ducks": "ANA", "Phoenix Coyotes": "UTA",
             "Arizona Coyotes": "UTA", "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA", "Atlanta Thrashers": "WPG",
             "Montréal Canadiens": "MTL", "Vegas Golden Knights": "VEG"}


def nhl_team(name):
    name = html.unescape(name).strip()
    for code, forbes, *_ in FRANCHISES["nhl"]:
        if forbes == name:
            return code
    return NHL_NAMES.get(name)


def nhl_trades(P):
    out = []
    for f in sorted(os.listdir(os.path.join(DATA, "nhltt"))):
        h = page("nhltt", f)
        for k, tb in enumerate(re.findall(r'<table class="trade">(.*?)</table>', h, re.S)):
            heads = re.findall(r"trade_list_by_team/[^\"]+\">([^<]+)</a>\s*acquire", tb)
            date = re.search(r'<td class="date"[^>]*>([^<]+)</td>', tb)
            if len(heads) != 2 or not date:
                continue
            m = re.match(r"(\w+) (\d+), (\d{4})", date.group(1).strip())
            mo, dd, yy = MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
            s, ph = season_of("nhl", yy, mo, dd), phase_of("nhl", mo)
            a, b = nhl_team(heads[0]), nhl_team(heads[1])
            if not (a and b):
                continue
            sides = ([], [])
            for row in re.findall(r"<tr>(.*?)</tr>", tb, re.S)[1:]:
                cells = [c for cls, c in re.findall(r'<td([^>]*)>(.*?)</td>', row, re.S) if "logo" not in cls and "date" not in cls]
                for i, c in enumerate(cells[:2]):
                    sides[i].append(text(c))
            tid = f"{f[:-5]}-{k}"
            for got, gave, items in ((a, b, sides[0]), (b, a, sides[1])):
                for it in items:
                    pk = re.search(r"(\d{4})\s+(?:conditional\s+)?(\d)(?:st|nd|rd|th)\s+round", it, re.I)
                    later = re.search(r"to be named later \(([^)#]+)\)", it)
                    if pk:
                        rec = dict(asset="pick", pick_year=int(pk.group(1)), pick_round=int(pk.group(2)))
                    elif re.search(r"\bpick\b", it, re.I) and not re.search(r"return of|not exercised", it, re.I):
                        yr = re.search(r"(\d{4})", it)
                        rec = dict(asset="pick", pick_year=int(yr.group(1)) if yr else s, pick_round=0)
                    elif later:
                        rec = dict(asset="player", pid=P.match(later.group(1).strip(), s, (gave, got)), name=later.group(1).strip())
                    elif not it or re.search(r"future considerations|cash|rights|waiver|loan|expansion draft|return of|"
                                             r"to be named|not exercised", it, re.I):
                        continue
                    else:
                        rec = dict(asset="player", pid=P.match(it, s, (gave, got)), name=it)
                    out.append(dict(rec, trade_id=tid, season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=ph, from_team=gave, to_team=got))
    return out


def capwages():
    out = []
    for f in sorted(os.listdir(os.path.join(DATA, "capwages"))):
        h = page("capwages", f)
        d = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))["props"]["pageProps"]
        out += d["signings"]
    return out


def nhl_fa_payroll(P):
    cm = dict(code_map("nhl"), VGK="VEG")
    contracts = defaultdict(list)             # pid -> [(start season, end season, cap hit, team, type, signed season)]
    fas, miss = [], 0
    for c in capwages():
        team = cm.get(c["teamTricode"])
        m = re.match(r"(\w{3})\w*\.? (\d+), (\d{4})", c["date"])
        if not team or not m:
            continue
        mo = next(v for k, v in MONTHS.items() if k.startswith(m.group(1)))
        yy, dd = int(m.group(3)), int(m.group(2))
        last, first = (c["player"].split(", ", 1) + [""])[:2]
        name = f"{first} {last}".strip()
        s = season_of("nhl", yy, mo, dd)
        pid = P.match(name, s, (team,), window=4)
        if not pid:
            miss += 1
            continue
        n = int(c["length"] or 1)
        listed = c["capHit"].strip()
        cap = float(re.sub(r"[^\d.]", "", listed) or 0) / 1e6 if listed.startswith("$") else 0.0
        total = float(re.sub(r"[^\d.]", "", c.get("value") or "") or 0) / 1e6
        if total and not 0 < cap <= total:    # a cap hit that is not money ('NHL LOCKOUT / 24% ROLLBACK'), or more
            cap = total / n                   # than the whole deal ($17.9M for a $925,000 one-year deal): its average
        start = s + 1 if "Ext" in c["contractType"] else s          # an extension starts the season after
        contracts[pid].append([start, n, cap, c["contractType"], f"{yy}{mo:02d}{dd:02d}"])
        if c["contractType"] in ("Std", "35+ Contract") and COVER["fa"]["nhl"][0] <= s:
            prev = P.last_team(pid, s - 1) or (P.last_team(pid, s - 2) if s - 1 == 2005 else None)
            if prev and prev != team:
                fas.append(dict(season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=phase_of("nhl", mo), to_team=team, pid=pid))
    # a contract runs until the next one starts; each season's cap hit goes to the teams he played for, by games
    pay, sal = defaultdict(float), {}
    for pid, cs in contracts.items():
        cs.sort(key=lambda c: c[4])
        runs = []                             # [start, end, cap]
        for start, n, cap, typ, _ in cs:
            if runs and start <= runs[-1][1]:
                if "Ext" in typ:              # an extension follows the deal it extends
                    start = runs[-1][1] + 1
                else:                         # a new deal (after a buyout, a release) replaces the old one
                    runs[-1][1] = start - 1
            runs.append([start, start + n - 1, cap])
        for start, end, cap in runs:
            for s in range(start, end + 1):
                sal[(pid, s)] = cap
                st = P.stints.get((pid, s), [])
                g = sum(x[1] for x in st)
                for t, gm, _ in st:
                    pay[(s, t)] += cap * (gm / g if g else 1 / len(st))
    print(f"  nhl: {miss} signings not matched to a player")
    return fas, dict(pay), sal


# ---------------------------------------------------------------- MLB

def lahman(name):
    import rdata
    warnings.simplefilter("ignore")
    import tarfile
    tf = tarfile.open(os.path.join(DATA, "lahman", "Lahman_14.0-0.tar.gz"))
    raw = tf.extractfile(f"Lahman/data/{name}.RData").read()
    tmp = os.path.join(DATA, "lahman", f"{name}.RData")
    open(tmp, "wb").write(raw)
    return rdata.read_rda(tmp)[name]


def mlb_moves(P):
    cm = code_map("mlb")
    teams = lahman("Teams")
    retro = {r.teamIDretro: cm[r.teamIDBR] for r in teams.itertuples() if r.yearID >= 1998 and r.teamIDBR in cm}
    people = lahman("People")
    bb = {r.retroID: r.bbrefID for r in people.itertuples() if isinstance(r.retroID, str)}
    born = {r.bbrefID: r.birthYear for r in people.itertuples() if isinstance(r.bbrefID, str)}
    trades, fas = [], []
    z = zipfile.ZipFile(os.path.join(DATA, "retrosheet", "tranDB.zip"))
    for row in csv.reader(z.read("tran.txt").decode("latin-1").splitlines()):
        date, tid, player, typ, frm, to = row[0], row[5], row[6].strip(), row[7].strip(), row[8].strip(), row[10].strip()
        yy, mo, dd = int(date[:4]), int(date[4:6]), int(date[6:8])
        if not (2000 <= yy <= 2021):
            continue
        s = season_of("mlb", yy, mo or 1, dd or 1) if mo else yy
        ph = phase_of("mlb", mo) if mo else "off"
        if not (COVER["trades"]["mlb"][0] <= s <= COVER["trades"]["mlb"][1]):
            continue
        if typ == "T" and frm in retro and to in retro:
            pid = bb.get(player)
            rec = dict(asset="player", pid=pid, name=player) if pid else (dict(asset="prospect", name=player) if player else None)
            if rec:
                if pid and born.get(pid):
                    rec["age"] = s - int(born[pid])
                trades.append(dict(rec, trade_id=f"r{tid}", season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=ph,
                                   from_team=retro[frm], to_team=retro[to]))
        elif typ == "F" and to in retro and bb.get(player):
            fas.append(dict(season=s, date=f"{yy}-{mo:02d}-{dd:02d}", phase=ph, to_team=retro[to], pid=bb[player]))
    return trades, fas, born


def mlb_payroll():
    """the team's salaries as Baseball Reference totals them (the team batting value table), every season"""
    cm = code_map("mlb")
    out = {}
    for y in YEARS:
        h = page("sr", "mlb", f"value_batting_{y}.html").replace("<!--", "").replace("-->", "")
        for r in table(h, "teams_value_batting"):
            t = team_code(r.get("team_name", ""))
            if t in cm and num(r.get("Salary", "")):
                out[(y, cm[t])] = num(r["Salary"]) / 1e6
    return out


def mlb_salaries():
    """each player's salary, 2001-2016 (the Lahman database; Baseball Reference no longer lists them)"""
    bb = {r.playerID: r.bbrefID for r in lahman("People").itertuples() if isinstance(r.bbrefID, str)}
    out = defaultdict(float)
    for r in lahman("Salaries").itertuples():
        if 2001 <= r.yearID <= 2016 and r.playerID in bb:
            out[(bb[r.playerID], r.yearID)] += r.salary / 1e6
    return dict(out)


def lahman_payroll():
    """the Lahman database's salaries (to 2016), to check the one above"""
    cm = code_map("mlb")
    teams = lahman("Teams")
    br = {(r.yearID, r.teamID): r.teamIDBR for r in teams.itertuples()}
    out = defaultdict(float)
    for r in lahman("Salaries").itertuples():
        t = br.get((r.yearID, r.teamID))
        if 2001 <= r.yearID <= 2016 and t in cm:
            out[(r.yearID, cm[t])] += r.salary / 1e6
    return dict(out)


# ---------------------------------------------------------------- NFL

def nfl_team(nick):
    for code, name, *_ in FRANCHISES["nfl"]:
        if name.split()[-1] == nick:
            return code
    return {"Redskins": "WAS", "Washington": "WAS", "Football Team": "WAS"}.get(nick)


def nfl_all():
    """players (cap share each season), trades, free agents and payroll from the OverTheCap contracts"""
    import pyarrow.parquet as pq
    cm = code_map("nfl")
    rows = pq.read_table(os.path.join(DATA, "nflverse", "historical_contracts.parquet")).to_pylist()
    best = {}
    for c in rows:                            # each contract carries the player's whole history; keep the fullest
        k = c["otc_id"]
        if k not in best or len(c["season_history"] or []) > len(best[k]["season_history"] or []):
            best[k] = c
    born = {}
    for r in csv.DictReader(open(os.path.join(DATA, "nflverse", "players.csv"), encoding="utf8")):
        if r["otc_id"] and r["birth_date"][:4].isdigit():
            born[r["otc_id"]] = int(r["birth_date"][:4])
    P = Players("nfl")
    pay, caps, fas, sal = defaultdict(float), defaultdict(list), [], defaultdict(float)
    for oid, c in best.items():
        pid = str(oid)
        dob = re.search(r"(\d{4})$", c.get("date_of_birth") or "")
        dob = int(dob.group(1)) if dob else born.get(pid)
        hist = {}
        for s in c["season_history"] or []:
            if not (s["year"] or "").isdigit() or not nfl_team(s["team"] or ""):
                continue
            y, t = int(s["year"]), nfl_team(s["team"])
            if s.get("cap_number") and s.get("cap_percent"):
                caps[y].append(s["cap_number"] / s["cap_percent"])
            pct = (s.get("cap_percent") or 0) * 100
            hist[y] = t
            if 2001 <= y <= 2025:
                P.add(pid, c["player"], y, (y - dob) if dob else None, t, 1, pct)
                sal[(pid, y)] += s.get("cap_number") or 0
                if COVER["payroll"]["nfl"][0] <= y <= COVER["payroll"]["nfl"][1]:
                    pay[(y, t)] += s.get("cap_number") or 0
        for ch in c["contract_history"] or []:
            t, y = nfl_team(ch["team"] or ""), ch["year_signed"]
            if ch["contract_type"] == "UFA" and t and y and COVER["fa"]["nfl"][0] <= y <= COVER["fa"]["nfl"][1]:
                prev = hist.get(y - 1)
                if prev and prev != t:
                    fas.append(dict(season=y, date=str(y), phase="off", to_team=t, pid=pid, apy=ch["apy"] or 0))
    P.finish()
    for f in fas:                             # an NFL free agent: what the new deal pays, as a share of that year's cap
        f["pct"] = 100 * f.pop("apy") / median(caps[f["season"]])
    pid_by_pfr = {}
    for r in csv.DictReader(open(os.path.join(DATA, "nflverse", "players.csv"), encoding="utf8")):
        if r["pfr_id"] and r["otc_id"]:
            pid_by_pfr[r["pfr_id"]] = r["otc_id"]
    trades = []
    for r in csv.DictReader(open(os.path.join(DATA, "nflverse", "trades.csv"))):
        if r["gave"] not in cm or r["received"] not in cm:
            continue
        s = int(r["season"])
        mo = int(r["trade_date"][5:7]) if r["trade_date"] else 4
        base = dict(trade_id=r["trade_id"], season=s, date=r["trade_date"], phase=phase_of("nfl", mo),
                    from_team=cm[r["gave"]], to_team=cm[r["received"]])
        if r["pick_season"]:
            trades.append(dict(base, asset="pick", pick_year=int(r["pick_season"]), pick_round=int(r["pick_round"] or 7)))
        elif r["pfr_id"]:
            trades.append(dict(base, asset="player", pid=pid_by_pfr.get(r["pfr_id"]), name=r["pfr_name"]))
    return P, trades, fas, dict(pay), dict(sal)


# ---------------------------------------------------------------- together

def value_rows(P, rows, born=None):
    """each moved player's track record (stars) and age; picks and prospects as future assets"""
    lg = P.league
    for r in rows:
        if r.get("asset") == "player":
            pid = r.get("pid")
            v = P.prev(pid, r["season"]) if pid else 0.0
            if pid and lg == "nfl":                            # NFL: his cap share that season, else the last
                v = P.value.get((pid, r["season"]), v)
            elif pid and r["phase"] == "in":                   # mid-season: also what he'd done for that team so far
                v = max([v] + [x for t, _, x in P.stints.get((pid, r["season"]), []) if t == r["from_team"]])
            age = r.get("age") or (P.age_at(pid, r["season"]) if pid else None)
            r["stars"] = round(v / P.star, 3)
            r["age"] = age
            r["name"] = P.name.get(pid, r.get("name", ""))
            young = (age is None and not pid and lg in ("nhl", "mlb")) or (age is not None and age <= PROSPECT_AGE[lg])
            if r["stars"] < 0.25 and young:
                r["asset"] = "prospect"
        if r.get("asset") == "prospect":
            r["future"] = PROSPECT_W
        elif r.get("asset") == "pick":
            r["future"] = PICK_W.get(r["pick_round"], OTHER_PICK)
        else:
            r.setdefault("future", 0.0)
        r.setdefault("stars", 0.0)
    return rows


def fa_rows(P, rows):
    out = []
    for r in rows:
        prev = P.last_team(r["pid"], r["season"] - 1) or P.last_team(r["pid"], r["season"] - 2)
        if P.league == "nfl" or (prev and prev != r["to_team"]):
            v = r.pop("pct") if "pct" in r else P.prev(r["pid"], r["season"])
            out.append(dict(r, kind="fa", asset="player", from_team=r.get("from_team") or prev,
                            stars=round(v / P.star, 3), age=P.age_at(r["pid"], r["season"]), name=P.name.get(r["pid"], "")))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    players, moves, payroll = [], [], {}
    for lg in ("nba", "nhl", "mlb", "nfl"):
        print(lg, flush=True)
        born = None
        if lg == "nfl":
            P, trades, fas, pay, sal = nfl_all()
        else:
            P = sr_players(lg, dict(code_map(lg), **({"VGK": "VEG"} if lg == "nhl" else {})))
            if lg == "nba":
                trades, fas = nba_moves(P)
                pay, sal = nba_payroll()
            elif lg == "nhl":
                trades = nhl_trades(P)
                fas, pay, sal = nhl_fa_payroll(P)
            else:
                trades, fas, born = mlb_moves(P)
                pay, sal = mlb_payroll(), mlb_salaries()
        lo, hi = COVER["payroll"][lg]
        payroll[lg] = {k: v for k, v in pay.items() if lo <= k[0] <= hi}
        lo, hi = COVER["trades"][lg]
        tr = value_rows(P, [dict(t, kind="trade", league=lg) for t in trades if lo <= t["season"] <= hi], born)
        lo, hi = COVER["fa"][lg]
        fa = [dict(f, league=lg) for f in fa_rows(P, fas) if lo <= f["season"] <= hi]
        unmatched = sum(1 for t in tr if t["asset"] in ("player", "prospect") and not t.get("pid"))
        print(f"  star line {P.star:.2f}; {len(tr)} trade assets ({unmatched} players unmatched), {len(fa)} free agents, "
              f"{len(payroll[lg])} payrolls", flush=True)
        for (pid, s), v in P.value.items():
            players.append(dict(league=lg, season=s, pid=pid, name=P.name.get(pid, ""), age=P.age.get((pid, s)),
                                team=P.last_team(pid, s), value=round(v, 3), stars=round(v / P.star, 3),
                                teams="|".join(t for t, _, _ in P.stints.get((pid, s), [])),
                                salary_musd=round(sal[(pid, s)], 4) if (pid, s) in sal else ""))
        moves += tr + fa
    cols = ["league", "season", "date", "phase", "kind", "trade_id", "from_team", "to_team", "asset", "pid", "name", "age",
            "stars", "future", "pick_year", "pick_round"]
    write("moves.csv", [{c: m.get(c, "") for c in cols} for m in moves])
    write("players.csv", sorted(players, key=lambda r: (r["league"], r["season"], -r["value"])))
    pr = []
    for lg, pay in payroll.items():
        by = defaultdict(list)
        for (s, t), v in pay.items():
            by[s].append(v)
        for (s, t), v in sorted(pay.items()):
            pr.append(dict(league=lg, season=s, team=t, payroll_musd=round(v, 2), payroll_rel=round(v / (sum(by[s]) / len(by[s])), 4)))
    write("payroll.csv", pr)
    lah, mine = lahman_payroll(), {(r["season"], r["team"]): r["payroll_musd"] for r in pr if r["league"] == "mlb"}
    ks = [k for k in lah if k in mine]
    print(f"  MLB payroll against the Lahman salaries, 2001-2016: r = {np.corrcoef([lah[k] for k in ks], [mine[k] for k in ks])[0, 1]:.3f} ({len(ks)} team-seasons)")
    write("team_moves.csv", team_moves(moves, pr))


def team_moves(moves, pr):
    seasons = list(csv.DictReader(open(os.path.join(OUT, "seasons.csv"))))
    pay = {(r["league"], r["season"], r["team"]): r for r in pr}
    agg = defaultdict(lambda: defaultdict(float))
    for m in moves:
        for team, sign in ((m["to_team"], 1), (m["from_team"], -1)):
            if not team:
                continue
            a = agg[(m["league"], int(m["season"]), team)]
            if m["kind"] == "trade":
                side = "in" if sign > 0 else "out"
                a[f"trade_stars_{side}"] += m["stars"]
                a[f"trade_future_{side}"] += m["future"]
                if m["phase"] == "off":
                    a["trade_net_off"] += sign * m["stars"]
                if m["asset"] == "pick" and m["pick_round"] == 1:
                    a[f"firsts_{side}"] += 1
                if m["asset"] == "player" and m["stars"] >= 1:
                    a[f"stars_{side}"] += 1                        # all-star-level players
            else:
                a["fa_stars_in" if sign > 0 else "fa_stars_out"] += m["stars"]
    out = []
    for r in seasons:
        lg, s, t = r["league"], int(r["season"]), r["team"]
        a = agg[(lg, s, t)]
        row = dict(league=lg, season=s, team=t)
        has_tr = COVER["trades"][lg][0] <= s <= COVER["trades"][lg][1]
        has_fa = COVER["fa"][lg][0] <= s <= COVER["fa"][lg][1]
        for c in ("trade_stars_in", "trade_stars_out", "trade_future_in", "trade_future_out", "firsts_in", "firsts_out",
                  "stars_in", "stars_out"):
            row[c] = round(a[c], 3) if has_tr else ""
        row["trade_net"] = round(a["trade_stars_in"] - a["trade_stars_out"], 3) if has_tr else ""
        row["future_net"] = round(a["trade_future_in"] - a["trade_future_out"], 3) if has_tr else ""
        row["trade_net_off"] = round(a["trade_net_off"], 3) if has_tr else ""
        for c in ("fa_stars_in", "fa_stars_out"):
            row[c] = round(a[c], 3) if has_fa else ""
        row["fa_net"] = round(a["fa_stars_in"] - a["fa_stars_out"], 3) if has_fa else ""
        p = pay.get((lg, s, t))
        row["payroll_musd"] = p["payroll_musd"] if p else ""
        row["payroll_rel"] = p["payroll_rel"] if p else ""
        out.append(row)
    return out


if __name__ == "__main__":
    main()
