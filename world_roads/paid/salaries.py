"""Every salary we can document, 1913-2025, in its own year's dollars and in 2025's:  python salaries.py
-> ../media/paid/salaries.csv, top.csv (the 15 best-paid each year), coverage.csv

* MLB: each candidate's salary table on Baseball Reference (Michael Haupert's research of the Hall of Fame's contract
  cards for the early years; deferred money at its present value when signed, as Baseball Reference lists it, so
  Shohei Ohtani's 2024 deal counts at $28.2 million a year, not $70 million); the Lahman database's salaries for
  1985-2016 fill in anyone without a page.
* NBA: every roster's salary table on Basketball Reference, seasons 1984-85 to 2024-25 (a season counts for the year
  it ends; a player traded in mid-season, his larger figure).
* NFL: OverTheCap's contracts (nflverse): a season's pay is the average per year of the contract he was playing
  under, the way NFL pay is reported (a signing bonus spread over the deal, not counted in the year it was paid).
* NHL: cap hits (each contract's average per year, CapWages), 2010 on.
* Before those series begin (the NBA before 1984-85, the NFL before 1983 or a player OverTheCap lacks, the NHL before
  2010), the stars' contracts
  as documented in published sources: curated.csv, each row with its source and a quote.

Inflation: the annual average consumer price index (CPI-U) from the Federal Reserve Bank of Minneapolis' table,
1913-2025; a salary in year y is worth salary * CPI(2025) / CPI(y) in 2025 dollars.
"""
import os, re, sys, csv, html, json, unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
REBUILD = os.path.join(ROOT, "rebuild")
OUT = os.path.join(ROOT, "media", "paid")
sys.path.insert(0, REBUILD)
from build import table, text                                   # noqa: E402
from candidates import lahman                                   # noqa: E402

FIRST, LAST = 1913, 2025
TWO = {"Red Sox", "White Sox", "Blue Jays", "Trail Blazers", "Maple Leafs", "Red Wings", "Golden Knights",
       "Blue Jackets", "Football Team"}


def nickname(team):
    """'New York Yankees' -> 'Yankees', 'Boston Red Sox' -> 'Red Sox'"""
    team = html.unescape(team).replace("\xa0", " ").strip()
    team = re.sub(r" of Anaheim$", "", team)
    w = team.split()
    if len(w) >= 2 and " ".join(w[-2:]) in TWO:
        return " ".join(w[-2:])
    return w[-1] if w else team


def key(name):
    s = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def money(s):
    s = text(s).replace(",", "")
    m = re.search(r"\$?\s*([\d.]+)", s)
    return float(m.group(1)) if m else None


def cpi():
    h = open(os.path.join(DATA, "cpi.html"), encoding="utf8", errors="replace").read()
    out = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S):
        c = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if c and re.fullmatch(r"\d{4}", c[0]) and re.fullmatch(r"[\d.]+", c[1]):
            out[int(c[0])] = float(c[1])
    return out


def page(*p):
    return open(os.path.join(*p), encoding="utf8", errors="replace").read()


# ---------------------------------------------------------------- MLB

def mlb():
    rows = []
    d = os.path.join(DATA, "sr", "mlb", "players")
    have = set()
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        h = page(d, f).replace("<!--", "").replace("-->", "")
        if not h:
            continue
        pid = f[:-6]
        m = re.search(r"<h1[^>]*>(.*?)</h1>", h, re.S)
        name = text(m.group(1)) if m else pid
        for r in table(h, "br-salaries"):
            y = text(r.get("year_ID", ""))
            sal = money(r.get("Salary", ""))
            note = text(r.get("notes", ""))
            if not y.isdigit() or sal is None or not (FIRST <= int(y) <= LAST):
                continue
            if re.search(r"per month|per week|per game", note, re.I) or sal < 100:
                continue                                   # a rate, not a season's salary
            rows.append(dict(year=int(y), league="MLB", pid=pid, player=name, team=nickname(text(r.get("team_name", ""))),
                             nominal=sal, source="Baseball Reference", basis="salary", note=note[:120]))
            have.add((pid, int(y)))
    # the Lahman salaries for 1985-2016, for anyone the pages don't cover
    people = lahman("People")
    names = {r.playerID: (r.bbrefID, f"{r.nameFirst} {r.nameLast}") for r in people.itertuples()}
    teams = lahman("Teams")
    tname = {(r.yearID, r.teamID): r.name for r in teams.itertuples()}
    for r in lahman("Salaries").itertuples():
        bb, nm = names.get(r.playerID, (r.playerID, r.playerID))
        if (bb, r.yearID) in have or not (FIRST <= r.yearID <= LAST):
            continue
        rows.append(dict(year=r.yearID, league="MLB", pid=bb, player=nm, team=nickname(tname.get((r.yearID, r.teamID), r.teamID)),
                         nominal=float(r.salary), source="Lahman database", basis="salary", note=""))
    return best(rows)


def best(rows):
    """one row per player and year: the largest (a player traded mid-season appears on two teams' tables)"""
    by = {}
    for r in rows:
        k = (r["league"], r["pid"], r["year"])
        if k not in by or r["nominal"] > by[k]["nominal"]:
            by[k] = r
    return list(by.values())


# ---------------------------------------------------------------- NBA

def nba():
    rows = []
    for base in (os.path.join(DATA, "sr", "nba"), os.path.join(REBUILD, "data", "sr", "nba")):
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            m = re.fullmatch(r"team_([A-Z]{3})_(\d{4})\.html", f)
            if not m:
                continue
            h = page(base, f).replace("<!--", "").replace("-->", "")
            t = re.search(r"<title>\s*\d{4}-\d{2}\s+(.*?)\s+Roster", h)
            team = nickname(t.group(1)) if t else m.group(1)
            for r in table(h, "salaries2"):
                pm = re.search(r"/players/\w/([a-z0-9.'-]+)\.html", r.get("player", ""))
                sal = money(r.get("salary", ""))
                if pm and sal:
                    rows.append(dict(year=int(m.group(2)), league="NBA", pid=pm.group(1), player=text(r["player"]), team=team,
                                     nominal=sal, source="Basketball Reference", basis="salary", note=""))
    return best(rows)


# ---------------------------------------------------------------- NFL

def nfl():
    import pyarrow.parquet as pq
    d = pq.read_table(os.path.join(REBUILD, "data", "nflverse", "historical_contracts.parquet")).to_pylist()
    full = {}
    for c in d:                        # every contract row carries the player's whole history; keep the fullest
        if c["otc_id"] not in full or len(c["contract_history"] or []) > len(full[c["otc_id"]]["contract_history"] or []):
            full[c["otc_id"]] = c
    rows = []
    for oid, c in full.items():
        played = {(int(s["year"]), s["team"]) for s in c["season_history"] or [] if (s["year"] or "").isdigit()
                  and ((s.get("cash_paid") or 0) > 0 or (s.get("cap_number") or 0) > 0)}
        deals = sorted([h for h in c["contract_history"] or [] if h["year_signed"] and h["apy"]], key=lambda h: h["year_signed"])
        for y, team in played:
            if not (1983 <= y <= LAST):
                continue
            on = [h for h in deals if h["team"] == team and h["year_signed"] <= y <= h["year_signed"] + max(1, h["yrs"] or 1) - 1]
            if on:
                h = on[-1]
                rows.append(dict(year=y, league="NFL", pid=str(oid), player=c["player"], team=team, nominal=h["apy"] * 1e6,
                                 source="OverTheCap (nflverse)", basis="contract average",
                                 note=f"{h['yrs']} years, ${h['total']:.1f}M, signed {h['year_signed']}" if h.get("total") else ""))
    return best(rows)


# ---------------------------------------------------------------- NHL

def nhl():
    sys.path.insert(0, REBUILD)
    from teams import FRANCHISES
    names = {code: nickname(name) for code, name, *_ in FRANCHISES["nhl"]}
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, "media", "rebuild", "players.csv"))):
        if r["league"] == "nhl" and r["salary_musd"] and int(r["season"]) >= 2010:
            rows.append(dict(year=int(r["season"]), league="NHL", pid=r["pid"], player=r["name"], team=names.get(r["team"], r["team"]),
                             nominal=float(r["salary_musd"]) * 1e6, source="CapWages", basis="cap hit (contract average)", note=""))
    return best(rows)


# ---------------------------------------------------------------- curated, before the series

def curated(covered):
    """documented contracts before each league's series begins; a row the series already has is left to the series"""
    rows = []
    p = os.path.join(HERE, "curated.csv")
    if not os.path.exists(p):
        return rows
    for r in csv.DictReader(open(p, encoding="utf8")):
        y, sal = int(r["year_end"]), float(r["salary_usd"])
        lg = r["league"].upper()
        lg = {"ABA": "NBA", "BAA": "NBA", "NBL": "NBA", "AFL": "NFL", "AAFC": "NFL", "WFL": "NFL", "USFL": "NFL", "WHA": "NHL"}.get(lg, lg)
        if (lg, key(r["player"]), y) in covered:
            continue
        rows.append(dict(year=y, league=lg, circuit=r["league"].upper(), pid="c:" + key(r["player"]), player=r["player"],
                         team=nickname(r["team"].split("/")[-1]), nominal=sal, source=r["source_url"], basis=r["basis"],
                         note=r.get("notes", "")[:160]))
    return best(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = cpi()
    rows = mlb() + nba() + nfl() + nhl()
    covered = {(r["league"], key(r["player"]), r["year"]) for r in rows}
    rows += curated(covered)
    for r in rows:
        r.setdefault("circuit", r["league"])
        r["real_2025"] = round(r["nominal"] * c[LAST] / c[r["year"]], 0)
        r["nominal"] = round(r["nominal"], 0)
    rows.sort(key=lambda r: (r["year"], -r["real_2025"]))
    cols = ["year", "league", "circuit", "player", "team", "nominal", "real_2025", "basis", "source", "pid", "note"]
    with open(os.path.join(OUT, "salaries.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: r[k] for k in cols} for r in rows])
    top, cov = [], []
    for y in range(FIRST, LAST + 1):
        R = [r for r in rows if r["year"] == y]
        for i, r in enumerate(R[:15]):
            top.append(dict(year=y, rank=i + 1, **{k: r[k] for k in ("league", "circuit", "player", "team", "nominal", "real_2025", "basis", "pid")}))
        n = defaultdict(int)
        for r in R:
            n[r["league"]] += 1
        cov.append(dict(year=y, cpi=c[y], to_2025=round(c[LAST] / c[y], 3), **{lg: n[lg] for lg in ("MLB", "NBA", "NFL", "NHL")}))
    for name, data in (("top.csv", top), ("coverage.csv", cov)):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"salaries.csv: {len(rows)} rows; top.csv: {len(top)}; CPI 1913 {c[FIRST]}, 2025 {c[LAST]}")


if __name__ == "__main__":
    main()
