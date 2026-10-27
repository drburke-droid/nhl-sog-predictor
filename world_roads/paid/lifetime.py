"""Career earnings in 2025 dollars, for the career race:  python lifetime.py
-> ../media/paid/careers.csv (every season counted, its pay in both dollars, documented or estimated)
   ../media/paid/career_top.csv (the 40 biggest careers to date, every year 1913-2025, retired players included)

A career's earnings are the sum of its seasons' pay, each season converted to 2025 dollars with its own year's CPI.
The pay of a season is the yearly race's figure (salaries.py), with two differences, because a career adds up the money
as it was paid:
* NFL: the cash paid that season (OverTheCap's season histories, 1983 on: salary and every bonus in the season it was
  paid), not the contract's average per year.
* Careers that began before 1913 count from their first season: Baseball Reference's salaries before 1913, converted
  with the Minneapolis Fed's CPI estimates for 1800-1912 (spliced to the CPI-U at 1913).
NHL cap hits count from 2005-06, the first season after the lockout; the yearly race starts them in 2009-10, when every
contract is in CapWages, but a career needs the long contracts already running before then.

A season a player played without a documented salary, between two seasons with one, is estimated: his salary rises
(or falls) by the same percentage each season from one documented season to the next. (A straight line would put too
much in the seasons just before a big raise: O.J. Simpson's 1973-75, between $53,750 and $833,333.) MLB seasons played come from the Lahman database's appearances
(the established major leagues, 1871 on; not the Negro leagues, which Lahman now includes but whose pay is not on
record); in the other leagues the seasons between two documented ones, before each league's own
series begins (NBA 1984-85, NFL 1983, NHL 2005-06), are taken as played, except the lockout season 2004-05 and the
seasons the research found a player sat out or wasn't paid for (research/spans_*.csv). Seasons
before a player's first documented salary or after his last are left out, not guessed. Estimated seasons are marked,
and each career's estimated share is kept.
"""
import os, re, sys, csv, html
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import salaries as S                                            # noqa: E402
from salaries import key, nickname, page                        # noqa: E402
from candidates import lahman                                   # noqa: E402

FIRST, LAST = S.FIRST, S.LAST
SERIES = {"NBA": 1985, "NFL": 1983, "NHL": 2006}                # where each league's own series begins
KEEP = 40


def cpi():
    """the CPI-U from 1913, and the Minneapolis Fed's estimates before it, spliced at 1913"""
    c = S.cpi()
    early = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page(S.DATA, "cpi_1800.html"), re.S):
        x = [html.unescape(re.sub(r"<[^>]+>", "", v)).strip() for v in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if x and re.fullmatch(r"\d{4}", x[0]) and re.fullmatch(r"[\d.]+", x[1]):
            early[int(x[0])] = float(x[1])
    for y in range(1800, FIRST):
        c[y] = early[y] * c[FIRST] / early[FIRST]
    return c


MAJOR = {"NL", "AL", "AA", "UA", "PL", "FL", "NA"}     # the Negro leagues' seasons, which have no salaries on record, aside


def mlb_played():
    """(bbref id) -> {season: team nickname}, the seasons of every player in the established major leagues (Lahman
    appearances)"""
    bb = {r.playerID: r.bbrefID for r in lahman("People").itertuples()}
    team = {(r.yearID, r.teamID): nickname(r.name) for r in lahman("Teams").itertuples()}
    out = defaultdict(dict)
    for r in lahman("Appearances").itertuples():
        if r.lgID in MAJOR:
            out[bb.get(r.playerID, r.playerID)][r.yearID] = team.get((r.yearID, r.teamID), r.teamID)
    return out


def nfl_cash():
    import pyarrow.parquet as pq
    d = pq.read_table(os.path.join(S.REBUILD, "data", "nflverse", "historical_contracts.parquet")).to_pylist()
    full = {}
    for c in d:                        # every contract row carries the player's whole history; keep the fullest
        if c["otc_id"] not in full or len(c["season_history"] or []) > len(full[c["otc_id"]]["season_history"] or []):
            full[c["otc_id"]] = c
    rows = []
    for oid, c in full.items():
        for s in c["season_history"] or []:
            if (s["year"] or "").isdigit() and (s.get("cash_paid") or 0) > 0 and 1920 <= int(s["year"]) <= LAST:   # not the "Total" row
                rows.append(dict(year=int(s["year"]), league="NFL", pid=str(oid), player=c["player"], team=s["team"],
                                 nominal=s["cash_paid"] * 1e6, source="OverTheCap (nflverse)", basis="cash paid", note=""))
    return S.best(rows)


def nhl_all():
    sys.path.insert(0, S.REBUILD)
    from teams import FRANCHISES
    names = {code: nickname(name) for code, name, *_ in FRANCHISES["nhl"]}
    rows = []
    for r in csv.DictReader(open(os.path.join(S.ROOT, "media", "rebuild", "players.csv"))):
        if r["league"] == "nhl" and r["salary_musd"] and int(r["season"]) >= SERIES["NHL"]:
            rows.append(dict(year=int(r["season"]), league="NHL", pid=r["pid"], player=r["name"], team=names.get(r["team"], r["team"]),
                             nominal=float(r["salary_musd"]) * 1e6, source="CapWages", basis="cap hit (contract average)", note=""))
    return S.best(rows)


def merge_curated(rows):
    """a documented contract (pid 'c:name') joins the same player's series career, matched by name in the same league"""
    series = defaultdict(set)
    for r in rows:
        if not r["pid"].startswith("c:"):
            series[(r["league"], key(r["player"]))].add(r["pid"])
    span = defaultdict(list)
    for r in rows:
        span[(r["league"], r["pid"])].append(r["year"])
    for r in rows:
        if r["pid"].startswith("c:"):
            cands = series.get((r["league"], key(r["player"])), set())
            last = max(span[(r["league"], r["pid"])])
            near = sorted(cands, key=lambda p: abs(min(span[(r["league"], p)]) - last))
            if near and abs(min(span[(r["league"], near[0])]) - last) <= 4:
                r["pid"] = near[0]
    return rows


GROUP = {"ABA": "NBA", "BAA": "NBA", "NBL": "NBA", "AFL": "NFL", "AAFC": "NFL", "WFL": "NFL", "USFL": "NFL", "WHA": "NHL"}


def not_played():
    """seasons inside a career the player didn't play or wasn't paid for (research/spans_*.csv): (league, name) -> years"""
    out, stints = defaultdict(set), defaultdict(list)
    for lg in ("nba", "nfl", "nhl"):
        p = os.path.join(HERE, "research", f"spans_{lg}.csv")
        for r in csv.DictReader(open(p, encoding="utf8")) if os.path.exists(p) else []:
            group = GROUP.get(r["league"].strip().upper(), r["league"].strip().upper())
            for y in (r.get("not_played") or "").replace(",", ";").split(";"):
                if y.strip().isdigit():
                    out[(group, key(r["player"]))].add(int(y))
            stints[(group, key(r["player"]))].append((int(r["first_year_end"]), int(r["last_year_end"])))
    for k, ss in stints.items():                      # the years between two stints (retired, then back)
        for y in range(min(a for a, b in ss), max(b for a, b in ss) + 1):
            if not any(a <= y <= b for a, b in ss):
                out[k].add(y)
    return out


def estimate(rows, played):
    """the seasons played between two documented ones, their salary by equal percentage steps between"""
    skip = not_played()
    by = defaultdict(dict)
    for r in rows:
        by[(r["league"], r["pid"])][r["year"]] = r
    out = []
    for (lg, pid), d in by.items():
        ys = sorted(d)
        if lg == "MLB":
            gaps = [y for y in played.get(pid, {}) if ys[0] < y < ys[-1] and y not in d]
        else:
            out_ = skip.get((lg, key(d[ys[0]]["player"])), set()) | skip.get((lg, key(d[ys[-1]]["player"])), set())
            gaps = [y for y in range(ys[0] + 1, ys[-1]) if y not in d and y < SERIES[lg] and y not in out_
                    and not (lg == "NHL" and y == 2005)]
        for y in gaps:
            a = max(v for v in ys if v < y)
            b = min(v for v in ys if v > y)
            ra, rb = d[a], d[b]               # the same raise, in percent, each season from one documented salary to the next
            nom = ra["nominal"] * (rb["nominal"] / ra["nominal"]) ** ((y - a) / (b - a))
            team = played.get(pid, {}).get(y) if lg == "MLB" else (ra["team"] if ra["team"] == rb["team"] else "")
            out.append(dict(year=y, league=lg, circuit=ra.get("circuit", lg) if ra.get("circuit") == rb.get("circuit") else lg,
                            pid=pid, player=ra["player"], team=team or ra["team"], nominal=nom, source="estimated",
                            basis=f"estimated between {a} and {b}", note=""))
    return out


def main():
    os.makedirs(S.OUT, exist_ok=True)
    c = cpi()
    rows = S.mlb(first=1871) + S.nba() + nfl_cash() + nhl_all()
    covered = {(r["league"], key(r["player"]), r["year"]) for r in rows}
    rows += S.curated(covered)
    for r in rows:
        r.setdefault("circuit", r["league"])
    rows = S.best(merge_curated(rows))
    played = mlb_played()
    rows += estimate(rows, played)
    for r in rows:
        r.setdefault("circuit", r["league"])
        r["estimated"] = int(r["source"] == "estimated")
        r["real_2025"] = round(r["nominal"] * c[LAST] / c[r["year"]], 0)
        r["nominal"] = round(r["nominal"], 0)
    # careers, season by season: a player's name and league from his latest season, his total to date
    by = defaultdict(list)
    for r in rows:
        by[(r["league"], r["pid"])].append(r)
    career = []
    for k, rs in by.items():
        rs.sort(key=lambda r: r["year"])
        cr = cn = est = 0.0
        for i, r in enumerate(rs):
            cr += r["real_2025"]
            cn += r["nominal"]
            est += r["real_2025"] * r["estimated"]
            r.update(cum_real=cr, cum_nominal=cn, seasons=i + 1, first=rs[0]["year"], est_share=round(est / cr, 3) if cr else 0)
        career.append(rs)
    cols = ["year", "league", "circuit", "player", "team", "nominal", "real_2025", "estimated", "basis", "source", "pid",
            "seasons", "cum_nominal", "cum_real", "est_share"]
    with open(os.path.join(S.OUT, "careers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rs in sorted(career, key=lambda rs: -rs[-1]["cum_real"]):
            w.writerows([{k: r[k] for k in cols} for r in rs])
    # the biggest careers to date, each year: a retired player keeps his total
    import heapq
    # the seasons a career spans, for its label: MLB's from the appearances (a buyout paid after the last season is money,
    # not a season played), the other leagues' from their pay
    span = {}
    for rs in career:
        k = (rs[0]["league"], rs[0]["pid"])
        paid = {r["year"] for r in rs}
        span[k] = set(played.get(k[1], {})) or paid if k[0] == "MLB" else paid
        span[k] = (span[k], min(span[k] | paid))
    when = defaultdict(list)
    for r in rows:
        when[max(r["year"], FIRST - 1)].append(r)
    state, top = {}, []
    for y in range(FIRST - 1, LAST + 1):
        for r in sorted(when[y], key=lambda r: r["year"]):
            state[(r["league"], r["pid"])] = r
        if y < FIRST:
            continue
        best = heapq.nlargest(KEEP, state.values(), key=lambda r: (r["cum_real"], r["player"]))
        for i, r in enumerate(best):
            ys, first = span[(r["league"], r["pid"])]
            top.append(dict(year=y, rank=i + 1, league=r["league"], circuit=r["circuit"], player=r["player"], team=r["team"],
                            pid=r["pid"], first=first, last=max([v for v in ys if v <= y] or [r["year"]]), active=int(y in ys),
                            seasons=sum(v <= y for v in ys), cum_nominal=round(r["cum_nominal"]), cum_real=round(r["cum_real"]),
                            est_share=r["est_share"]))
    with open(os.path.join(S.OUT, "career_top.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0]))
        w.writeheader()
        w.writerows(top)
    n_est = sum(r["estimated"] for r in rows)
    print(f"careers.csv: {len(rows)} seasons ({n_est} estimated), {len(career)} careers; career_top.csv: {len(top)} rows")


if __name__ == "__main__":
    main()
