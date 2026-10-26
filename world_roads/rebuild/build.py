"""Tidy tables from the raw pages (fetch.py):  python build.py
-> ../media/rebuild/seasons.csv, drafts.csv, nfl_trades.csv, teams.csv

* seasons.csv: one row per franchise and season, 2000-01 to 2024-25 (2005 in the NHL was lost to the lockout; NFL
  and MLB seasons by calendar year, NHL and NBA by the year they end): wins, losses, the win share the league ranks
  by (NHL points %, NFL ties as half a win), the roster's average age (NHL, NBA; MLB batters and pitchers averaged),
  playoffs, championship, and the season's rank within the league as a percentile (0 the worst team, 1 the best).
* drafts.csv: every NHL, NBA and NFL draft pick 2000-2025, with its round and the franchise that made it.
* nfl_trades.csv: the draft picks in NFL trades since 2002 (nflverse), by franchise, given or received.
* teams.csv: each franchise's Forbes value (latest edition on Wikipedia's per-league lists), revenue, and metro
  population (2025 estimate for US metro areas, 2021 census for Canadian ones).
"""
import os, re, csv, html, json
from collections import defaultdict
from teams import FRANCHISES, code_map, info

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(os.path.dirname(HERE), "media", "rebuild")
YEARS = range(2001, 2026)


def page(*p):
    return open(os.path.join(DATA, *p), encoding="utf-8", errors="replace").read()


def text(cell):
    return re.sub(r"\[[^\]]*\]", "", html.unescape(re.sub(r"<[^>]+>", "", cell))).strip()


def table(h, tid):
    """rows of a Sports Reference table (also when it sits inside an HTML comment) as [{data-stat: cell html}]"""
    m = re.search(rf'<table[^>]*id="{tid}".*?</table>', h, re.S)
    if not m:
        return []
    rows = []
    for r in re.findall(r"<tr([^>]*)>(.*?)</tr>", m.group(0), re.S):
        cells = dict(re.findall(r'<t[hd][^>]*data-stat="([^"]+)"[^>]*>(.*?)</t[hd]>', r[1], re.S))
        cells["_class"] = r[0]
        rows.append(cells)
    return rows


def team_code(cell):
    m = re.search(r"/teams/([A-Z]{2,3})/", cell)
    return m.group(1) if m else None


def finals_winner(h):
    """the champion from a league page's playoffs table: the winner of the first series listed (the final)"""
    m = re.search(r'<table[^>]*id="all_playoffs".*?</table>', h, re.S) or re.search(r'id="all_playoffs".*?</table>', h, re.S)
    if not m:
        return None
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.S):
        if "Final" in r and "/teams/" in r:
            return team_code(r)
    return None


def nhl_nba(league):
    cm = code_map(league)
    rows = []
    for y in YEARS:
        name = f"{league.upper()}_{y}.html"
        if league == "nhl" and y == 2005:
            continue
        h = page("sr", league, name)
        champ = cm.get(finals_winner(h))
        tid, tcol = ("stats", "team_name") if league == "nhl" else ("advanced-team", "team")
        for r in table(h, tid):
            code = team_code(r.get(tcol, ""))
            if not code or code not in cm:
                continue
            w, l = int(text(r["wins"])), int(text(r["losses"]))
            if league == "nhl":
                ol = int(text(r.get("losses_ot", "0")) or 0) + int(text(r.get("ties", "0")) or 0)
                pct = float(text(r["points_pct"]))
                age = float(text(r["average_age"]))
            else:
                ol, pct, age = 0, w / (w + l), float(text(r["age"]))
            rows.append(dict(league=league, season=y, team=cm[code], code=code, wins=w, losses=l, other=ol, pct=round(pct, 4),
                             age=age, playoffs=int("*" in text(r.get(tcol, ""))), champion=int(cm[code] == champ)))
    return rows


def mlb():
    cm = code_map("mlb")
    champs = mlb_champions()
    rows = []
    for y in YEARS:
        p = {team_code(r.get("team_name", "")) or text(r.get("team_name", "")): r for r in table(page("sr", "mlb", f"pitching_{y}.html"), "teams_standard_pitching")}
        b = {team_code(r.get("team_name", "")) or text(r.get("team_name", "")): r for r in table(page("sr", "mlb", f"batting_{y}.html"), "teams_standard_batting")}
        for key, r in p.items():
            code = key if key in cm else None
            if code is None:
                continue
            w, l = int(text(r["W"] if "W" in r else r["wins"])), int(text(r["L"] if "L" in r else r["losses"]))
            ages = [float(text(x["age_pitch" if x is r else "age_bat"])) for x in (r, b.get(key)) if x is not None]
            rows.append(dict(league="mlb", season=y, team=cm[code], code=code, wins=w, losses=l, other=0, pct=round(w / (w + l), 4),
                             age=round(sum(ages) / len(ages), 2), playoffs=None, champion=int(champs.get(y) == cm[code])))
    return rows


def mlb_champions():
    """World Series winners by year, from Wikipedia's list (data/champions/world_series.html), as franchise codes"""
    names = {name: code for code, name, *_ in FRANCHISES["mlb"]}
    names.update({"Florida Marlins": "MIA", "Anaheim Angels": "LAA", "Cleveland Indians": "CLE", "Oakland Athletics": "ATH"})
    out = {}
    h = page("champions", "world_series.html")
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        c = [text(x) for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
        if c and re.fullmatch(r"(19|20)\d\d", c[0]) and 2001 <= int(c[0]) <= 2025:
            for cell in c[1:3]:
                for name, code in names.items():
                    if cell.startswith(name):
                        out.setdefault(int(c[0]), code)
                        break
                if int(c[0]) in out:
                    break
    return out


def nfl():
    cm = code_map("nfl")
    g = list(csv.DictReader(open(os.path.join(DATA, "nflverse", "games.csv"))))
    rec = defaultdict(lambda: [0, 0, 0])
    playoffs, champs = set(), {}
    for r in g:
        if not r["result"] or not (2000 <= int(r["season"]) <= 2024):
            continue
        s, home, away, res = int(r["season"]), cm[r["home_team"]], cm[r["away_team"]], float(r["result"])
        if r["game_type"] == "REG":
            for team, sign in ((home, 1), (away, -1)):
                k = rec[(s, team)]
                k[0 if res * sign > 0 else 1 if res * sign < 0 else 2] += 1
        else:
            playoffs.update({(s, home), (s, away)})
            if r["game_type"] == "SB":
                champs[s] = home if res > 0 else away
    rows = []
    for (s, team), (w, l, t) in sorted(rec.items()):
        rows.append(dict(league="nfl", season=s, team=team, code=team, wins=w, losses=l, other=t,
                         pct=round((w + 0.5 * t) / (w + l + t), 4), age=None, playoffs=int((s, team) in playoffs),
                         champion=int(champs.get(s) == team)))
    return rows


def percentiles(rows):
    """each season's rank within its league, 0 for the worst record and 1 for the best (ties share a rank)"""
    by = defaultdict(list)
    for r in rows:
        by[(r["league"], r["season"])].append(r)
    for grp in by.values():
        vals = sorted(r["pct"] for r in grp)
        n = len(grp)
        for r in grp:
            lo, hi = vals.index(r["pct"]), n - 1 - vals[::-1].index(r["pct"])
            r["rank_pct"] = round(((lo + hi) / 2) / (n - 1), 4)
            r["rank"] = n - hi                     # 1 = best
            r["teams"] = n


def drafts():
    out = []
    for league in ("nhl", "nba"):
        cm = code_map(league)
        for y in range(2000, 2026):
            h = page("sr", league, f"draft_{y}.html")
            m = re.search(r'<table[^>]*id="stats".*?</table>', h, re.S)
            rnd = 1
            for cls, body in re.findall(r"<tr([^>]*)>(.*?)</tr>", m.group(0), re.S)[1:]:
                if "thead" in cls or "over_header" in cls:
                    r = re.search(r"Round (\d+)", body)
                    rnd = int(r.group(1)) if r else rnd
                    continue
                cells = dict(re.findall(r'<t[hd][^>]*data-stat="([^"]+)"[^>]*>(.*?)</t[hd]>', body, re.S))
                code = team_code(cells.get("team_name", "") or cells.get("team_id", ""))
                pick = text(cells.get("pick_overall", ""))
                if code in cm and pick.isdigit():
                    out.append(dict(league=league, year=y, round=rnd, pick=int(pick), team=cm[code]))
    cm = code_map("nfl")
    for r in csv.DictReader(open(os.path.join(DATA, "nflverse", "draft_picks.csv"))):
        if 2000 <= int(r["season"]) <= 2025 and r["team"] in cm:
            out.append(dict(league="nfl", year=int(r["season"]), round=int(r["round"]), pick=int(r["pick"]), team=cm[r["team"]]))
    return out


def nfl_trades():
    cm = code_map("nfl")
    out = []
    for r in csv.DictReader(open(os.path.join(DATA, "nflverse", "trades.csv"))):
        if r.get("pick_season") and r["gave"] in cm and r["received"] in cm:
            out.append(dict(trade_id=r["trade_id"], season=r["season"], date=r.get("trade_date", ""), gave=cm[r["gave"]],
                            received=cm[r["received"]], pick_season=r["pick_season"], pick_round=r.get("pick_round", ""),
                            pick_number=r.get("pick_number", "")))
    return out


def teams():
    """Forbes value and revenue (first table of Wikipedia's per-league page), metro population"""
    metro = metros()
    out = []
    for league, fr in FRANCHISES.items():
        h = page("val", f"forbes_{league.upper()}.html")
        t = re.findall(r"<table[^>]*wikitable.*?</table>", h, re.S)[0]
        rows = [[text(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)] for r in re.findall(r"<tr.*?</tr>", t, re.S)]
        head = rows[0]
        vals = {r[head.index("Team")]: r for r in rows[1:]}
        for code, name, m, old, moved in fr:
            r = vals[name]
            out.append(dict(league=league, team=code, name=name, metro=m, moved=moved or "",
                            value_musd=money(r[head.index("Value")]), revenue_musd=money(r[head.index("Revenue")]),
                            metro_pop=metro[m]))
    return out


def money(s):
    s = s.replace("\n", " ").replace(",", "")
    v = float(re.search(r"[\d.]+", s).group(0))
    return round(v * 1000 if "billion" in s else v, 1)


CANADA = {"Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton", "Ottawa", "Winnipeg"}
US_HINT = {"Columbus": "Columbus, OH", "Portland": "Portland–Vancouver", "Jacksonville": "Jacksonville, FL", "Washington": "Washington–Arlington",
           "Kansas City": "Kansas City, MO", "Charlotte": "Charlotte–Concord", "Miami": "Miami–Fort Lauderdale", "Dallas": "Dallas–Fort Worth",
           "Cleveland": "Cleveland, OH", "Phoenix": "Phoenix–Mesa", "Buffalo": "Buffalo–Cheektowaga", "Las Vegas": "Las Vegas–Henderson"}


def metros():
    us = {}
    h = page("metro", "us_msa.html")
    t = re.findall(r"<table[^>]*wikitable.*?</table>", h, re.S)[0]
    for r in re.findall(r"<tr.*?</tr>", t, re.S)[1:]:
        c = [text(x) for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
        if len(c) > 2 and c[1].replace(",", "").isdigit():
            us[c[0]] = int(c[1].replace(",", ""))
    ca = {}
    h = page("metro", "ca_cma.html")
    t = re.findall(r"<table[^>]*wikitable.*?</table>", h, re.S)[0]
    for r in re.findall(r"<tr.*?</tr>", t, re.S)[1:]:
        c = [text(x) for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
        if len(c) > 5 and c[5].replace(",", "").isdigit():
            ca[c[2]] = int(c[5].replace(",", ""))
    out = {}
    for m in {f[2] for fr in FRANCHISES.values() for f in fr}:
        if m in CANADA:
            hits = [k for k in ca if k.startswith(m)]
        else:
            key = US_HINT.get(m, m)
            hits = [k for k in us if k.startswith(key)]
        assert len(hits) == 1, (m, hits)
        out[m] = (ca if m in CANADA else us)[hits[0]]
    return out


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    seasons = nhl_nba("nhl") + nhl_nba("nba") + mlb() + nfl()
    percentiles(seasons)
    write("seasons.csv", seasons)
    write("drafts.csv", drafts())
    write("nfl_trades.csv", nfl_trades())
    write("teams.csv", teams())


if __name__ == "__main__":
    main()
