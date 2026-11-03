"""Write outputs/sources_and_corrections.csv: every data repair, verified fact and unresolved warning, with its
source, the original and revised values, and its measured effect on the scores.

    python analysis/corrections_log.py      # from followup/

Status values:
  applied        repaired in data/v2 (the original files are untouched); used by every 'corrected' variant
  quarantined    left out of data/v2 (not re-dated or guessed)
  verified       a fact checked against the source (no data change)
  reviewed       a flag examined and found to be correct as is, or a deliberate rule
  unresolved     could not be settled from the sources at hand; listed with its likely effect
Record-level impact = the largest change in any season leader's z in that category between the original
inputs and data/v2, both scored with the original method (window 15, OLS, first-listed leader), and the
category-seasons that moved by more than 0.001.
"""
import csv, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gsmodel as G                                       # noqa: E402

F = os.path.dirname(HERE)
AT = "http://www.alltime-athletics.com/{}.htm"
AT_PAGE = {("M", "PV"): "mpoleok", ("W", "PV"): "wpoleok", ("W", "HT"): "whammok", ("M", "LJ"): "mlongok",
           ("M", "TJ"): "mtripok", ("W", "LJ"): "wlongok", ("M", "100m"): "m_100ok", ("M", "200m"): "m_200ok",
           ("M", "110mH"): "m_110hok", ("W", "100m"): "w_100ok"}
WA = ("https://api.worldaquatics.com/fina/rankings/swimming?gender={g}&distance={d}&stroke={s}&poolConfiguration=LCM"
      "&year={y}&timesMode=BEST_TIMES&pageSize=100")


def impacts():
    D0, D2 = G.Data("original"), G.Data("v2")
    a = {(r["cat"], r["year"]): r for r in G.score(D0, G.Spec())}
    b = {(r["cat"], r["year"]): r for r in G.score(D2, G.Spec())}
    per = defaultdict(list)
    for k in set(a) | set(b):
        za, zb = a.get(k, {}).get("z"), b.get(k, {}).get("z")
        if za is None or zb is None:
            per[k[0]].append((k[1], za, zb))
        elif abs(za - zb) > 1e-3:
            per[k[0]].append((k[1], za, zb))
    def fmt(cat):
        ch = per.get(cat, [])
        if not ch:
            return "no score changes"
        ch.sort(key=lambda t: -abs((t[2] or 0) - (t[1] or 0)))
        return f"{len(ch)} season(s) changed; largest: " + "; ".join(
            f"{y}: {('%.2f' % za) if za is not None else 'none'} -> {('%.2f' % zb) if zb is not None else 'none'}"
            for y, za, zb in ch[:3])
    return fmt


def main():
    fmt = impacts()
    rows = []
    add = lambda **k: rows.append(k)
    log = list(csv.DictReader(open(os.path.join(F, "data", "v2", "corrections_applied.csv"), encoding="utf-8")))
    # C2 future dates
    for r in [x for x in log if x["code"] == "C2"]:
        sex, ev = r["category"].split("_")[1], r["category"].split("_", 2)[2]
        add(code="C2", status="quarantined", sport="Athletics", category=r["category"], season=r["season"], item=r["name"],
            original_value=f"{r['value']} dated {r['season']}", revised_value="mark left out of data/v2 (not re-dated)",
            evidence=f"The source line itself carries the impossible year (cached all-time list, {sex} {ev}); the parser copied it.",
            source_url=AT.format(AT_PAGE.get((sex, ev), "")), record_level_impact=fmt(r["category"]))
    # C1 wind prefixes (root cause + each name in an original top ten)
    add(code="C1", status="applied", sport="Athletics", category="(15 rows in 10 events)", season="", item="wind written with a decimal comma or two decimals",
        original_value="'+1,7 Simon Ehammer'", revised_value="'Simon Ehammer'",
        evidence="alltime.py WIND pattern [+-]?\\d{1,2}\\.\\d does not match '+1,7' or '+2.81', so the reading stayed in the name; marks are unaffected. Also created 19 duplicate athlete entries in the deeper season lists.",
        source_url=AT.format("m_100ok"), record_level_impact="none at window 15 for any season leader (names were never a leader's comparator)")
    for r in [x for x in log if x["code"] in ("C1", "C1dup")]:
        add(code=r["code"], status="applied", sport="Athletics", category=r["category"], season=r["season"], item=r["name"],
            original_value=r["name"], revised_value=r["name"].split(" ", 1)[1] if r["code"] == "C1" else "duplicate entry removed",
            evidence="wind reading parsed into the athlete's name", source_url="", record_level_impact=fmt(r["category"]))
    # C3 swimming: one name, two World Aquatics personIds in one season (evidence read from the cached lists)
    ev3 = {
        "Gabriel Santos": ("applied", "BRA both; identical 48.11; born 1996-05-04 / 1987-05-04 (same day and month)"),
        "Ben Armbruster": ("applied", "AUS both; 50.58 / 50.59; born 2002-05-10 / 2002-10-03 (day and month swapped)"),
        "Hannah Casey": ("applied", "AUS both; second record's birth date 2006-04-27 is a placeholder shared by 3+ names"),
        "Brittany Castelluzzo": ("applied", "AUS both; second record's birth date 2006-04-27 is a placeholder shared by 3+ names"),
        "Iona Anderson": ("applied", "AUS both; both born 2005"),
        "Liu Xiaoyu": ("applied", "CHN both; born 1991-06-25 / 1988-06-25 (same day and month)"),
        "Miho Takahashi": ("applied", "JPN both; identical 4:38.57 in 2018 (the same swim listed twice)"),
        "Maggie Bowen": ("applied", "USA both; no birth date on either record"),
        "Wang Guoyue": ("unresolved", "CHN both; one record has no birth date (other 1999): merged, could be two swimmers"),
        "Liu Jing": ("unresolved", "CHN both; one record has no birth date (other 1990): merged, could be two swimmers"),
    }
    for r in [x for x in log if x["code"] == "C3"]:
        parts = r["category"].split("_")
        st = {"fr": "FREESTYLE", "ba": "BACKSTROKE", "br": "BREASTSTROKE", "bu": "BUTTERFLY", "me": "MEDLEY"}[parts[3]]
        stt, ev = ev3.get(r["name"], ("applied", "same name, two personIds"))
        add(code="C3", status=stt, sport="Swimming", category=r["category"], season=r["season"], item=r["name"],
            original_value=f"listed twice (second entry {r['value']})", revised_value="one entry (the faster)",
            evidence=ev, source_url=WA.format(g="M" if parts[1] == "M" else "F", d=parts[2], s=st, y=r["season"]),
            record_level_impact=fmt(r["category"]))
    for cat, y, nm, ev in [("swim_W_100_br", "2013-2014", "Sophie Taylor", "GBR (born 1996-02-02) and AUS (born 1996-10-13) in the same seasons: two swimmers"),
                           ("swim_W_200_fr", "2011", "Chen Qian", "CHN; born 1993-04-16 and 1995-02-16; different times (1:57.37, 1:58.24): treated as two swimmers"),
                           ("swim_W_100_br", "2002", "Rhiannon Leier", "CAN; born 1977-05-30 and 1985-11-08: treated as two swimmers")]:
        add(code="C3x", status="reviewed", sport="Swimming", category=cat, season=y, item=nm, original_value="one name",
            revised_value="kept as two swimmers", evidence=ev, source_url="", record_level_impact=fmt(cat))
    # C4 spelling variants, C5 inconsistent birth dates
    for r in [x for x in log if x["code"] == "C4"]:
        add(code="C4", status="applied", sport="Athletics", category=r["category"], season="", item=r["name"],
            original_value="two name keys", revised_value="one athlete",
            evidence="same date of birth in the all-time list or World Athletics list and a near-identical spelling (twins excluded: Alvin/Calvin Harrison, Tia/Tina Clayton)",
            source_url="", record_level_impact=fmt(r["category"]))
    for r in [x for x in log if x["code"] == "C5"]:
        add(code="C5", status="unresolved", sport="Athletics", category=r["category"], season="", item=r["name"],
            original_value="one name", revised_value="kept as one person (as in the original)",
            evidence="dates of birth differ between sources by under five years with the same nation; may be one person with a source error or two people",
            source_url="", record_level_impact=fmt(r["category"]))
    # C6 football ID crosswalk
    n6 = sum(1 for x in log if x["code"] == "C6")
    add(code="C6", status="applied", sport="NFL", category="all six", season="1998/1999", item=f"{n6} players across the source boundary (e.g. Brett Favre, Jerry Rice, Emmitt Smith)",
        original_value="Pro Football Archives ID to 1998, nflverse ID from 1999", revised_value="one identity",
        evidence="same name, top-40 seasons on both sides of the boundary (archive seasons to 1996-98 and nflverse seasons from 1999-2001)",
        source_url="https://www.profootballarchives.com/ ; https://github.com/nflverse/nflverse-data", record_level_impact="no score changes once linked (unlinked IDs would have counted a player's own seasons as rivals)")
    # stable IDs separating different people the original merged
    for sport, cat, item, ev, url in [
        ("NFL", "nfl_rush_yds", "Adrian Peterson (Bears, 2002-09) and Adrian Peterson (Vikings, 2007-21)", "two Pro Football Archives / nflverse IDs", "https://www.pro-football-reference.com/"),
        ("NBA", "nba_ppg etc.", "Eddie Johnson (johnsed02) and Eddie Johnson (johnsed03)", "two Reference IDs, overlapping careers", "https://www.basketball-reference.com/players/j/johnsed02.html"),
        ("NBA", "nba_apg", "Mike Dunleavy Sr. and Jr.", "two Reference IDs", "https://www.basketball-reference.com/players/d/dunlemi01.html"),
        ("NHL", "nhl_*", "Syl Apps Sr. and Jr.", "two Reference IDs", "https://www.hockey-reference.com/players/a/appssy01.html"),
        ("Athletics", "ath_M_200m", "Wallace Spearmon Sr. (b. 1962) and Jr. (b. 1984)", "dates of birth 22 years apart", AT.format("m_200ok")),
        ("Athletics", "ath_M_100m", "Calvin Smith (b. 1961) and Calvin Smith (b. 1987)", "dates of birth 26 years apart", AT.format("m_100ok"))]:
        add(code="ID", status="applied", sport=sport, category=cat, season="", item=item,
            original_value="one name key (the original left both out as 'the performer')", revised_value="two athletes",
            evidence=ev, source_url=url, record_level_impact=fmt(cat) if not cat.endswith(("etc.", "_*")) else "see the per-category rows")
    # original-name aliases across leader boards (review round 2): one Reference ID under two names
    v2rows = [r for r in csv.DictReader(open(os.path.join(F, "data", "v2", "season_lists_v2.csv"), encoding="utf-8"))
              if r["orig_rank"] and r["category"].startswith("mlb_")]
    al = defaultdict(lambda: defaultdict(set))
    for r in v2rows:
        al[r["athlete_id"]][G.norm(r["name_raw"])].add((r["category"], r["season"]))
    for aid, names in sorted(al.items()):
        if len(names) > 1:
            same_season = sorted({s for n in names for (_, s) in names[n]} & set.intersection(*[{s for (_, s) in v} for v in names.values()]))
            add(code="ALIAS", status="applied", sport="MLB", category="several", season=";".join(same_season),
                item=" / ".join(sorted(names)), original_value="two name keys (two athletes in the original's one-entry-per-athlete-season step)",
                revised_value=f"one athlete ({aid})",
                evidence="Baseball-Reference shows the same player ID under two display names on different leader boards",
                source_url=f"https://www.baseball-reference.com/players/{aid[4]}/{aid[4:]}.shtml",
                record_level_impact=(("both names appear in top-ten lists of " + ", ".join(same_season) + "; the original split him into two "
                                      "athlete-seasons only where both names LED a category in one season (outputs/identity_crosswalk_log.csv: "
                                      "Aaron 1956, Witt 2025). Exclusion within a category was unaffected (one name per category).")
                                     if same_season else "names never share a season; exclusion within a category unaffected"))
    add(code="C7", status="applied", sport="NBA", category="nba_apg", season="1947-1948", item="BAA assists, 1946-47 and 1947-48",
        original_value="league assists 6.9 and 7.0 per team-game", revised_value="seasons left out of assist comparisons",
        evidence="18.6 per team-game in 1948-49: a 2.7-fold jump in one season reflects how assists were credited, not play; dividing by it made Ernie Calverley 1946-47 look like the 2nd-greatest season of all",
        source_url="https://www.basketball-reference.com/leagues/NBA_stats_per_game.html", record_level_impact=fmt("nba_apg"))
    add(code="TIES", status="applied", sport="all", category="237 category-seasons", season="", item="ties for first",
        original_value="only the first-listed co-leader scored", revised_value="every tied leader scored (4,943 scores, from 4,659)",
        evidence="row order decided which co-leader was scored; tests/test_reproduction.py checks the new rule is order-free",
        source_url="", record_level_impact="285 additional season-leader scores")
    for y, h, r in ((1919, 4, 3), (1932, 9, 8), (1935, 8, 9)):
        add(code="TEAMS", status="applied", sport="NHL", category="handoff field model", season=y, item="number of NHL teams",
            original_value=str(h), revised_value=str(r),
            evidence="team codes in Hockey-Reference's skater table for the season (1918-19 Ottawa, Montreal, Toronto; 1931-32 Ottawa sat out; 1934-35 St. Louis Eagles)",
            source_url=f"https://www.hockey-reference.com/leagues/NHL_{y}_skaters.html", record_level_impact="handoff field model only")
    add(code="PACE", status="verified", sport="NBA", category="nba_ppg", season=1962, item="1961-62 pace, 126.2 vs 131.1",
        original_value="both quoted as 'pace'", revised_value="126.2 = unweighted mean of the nine teams' estimated paces; 131.1 = Philadelphia Warriors' own estimated pace",
        evidence="Basketball-Reference's league-average table publishes pace only from 1973-74 (turnovers and offensive rebounds recorded); earlier team paces are estimates in each season's Advanced table (range 122.9-131.1 in 1961-62)",
        source_url="https://www.basketball-reference.com/leagues/NBA_1962.html ; https://www.basketball-reference.com/leagues/NBA_stats_per_game.html",
        record_level_impact="see nba_points_denominator_effect.csv (Wilt 1961-62 z: 7.73 none, 6.05 mean team pace, 5.75 shot opportunities, 4.48 league points)")
    add(code="QUAL", status="reviewed", sport="NBA", category="nba_ppg", season=1962, item="Elgin Baylor 38.3 ppg in 48 games",
        original_value="not in the qualified list", revised_value="unchanged",
        evidence="Basketball-Reference's per-game leaders require a minimum number of games; a qualification rule, not an error",
        source_url="https://www.basketball-reference.com/leagues/NBA_1962_leaders.html", record_level_impact="none")
    add(code="MOSS", status="unresolved", sport="NFL", category="nfl_rec_yds", season=1933, item="Paul Moss receiving yards",
        original_value="383 (Pro Football Archives)", revised_value="unchanged",
        evidence="Wikipedia's leader list has 283; not settled", source_url="https://www.profootballarchives.com/1933nflpit.html",
        record_level_impact="does not affect any top-200 season")
    add(code="ENVJUMP", status="reviewed", sport="NHL/MLB/NFL/NBA", category="environment denominators", season="",
        item="other one-season jumps in league rates (>40%)",
        original_value="NHL 1929-30, MLB 1877-78/1884/1887/1893/1919-21, NFL 1933, NBA threes 1995", revised_value="kept",
        evidence="each matches a documented rule change (forward pass 1929-30; four strikes 1887; pitching distance 1893; live ball; Lake Front Park ground rule 1884; NFL passing rule 1933; shorter three-point line 1994-97): real environment, which the adjustment is meant to absorb",
        source_url="", record_level_impact="none (no data change)")
    add(code="NAMES", status="reviewed", sport="all", category="handoff possible_name_collisions.txt", season="",
        item="62 similar-name pairs flagged by the handoff",
        original_value="name-based identity", revised_value="stable IDs where the source has one",
        evidence="pairs such as Joe/Dom DiMaggio, Ted/Ken Williams, Ryan/Dean Crouser, Gianmarco/Marco Tamberi are different people and were already distinct; spelling variants (Malachowski, Zagorniy/Zagornyi, Ta Lou/Ta Lou-Smith) are merged by C4",
        source_url="", record_level_impact="see C4 rows")
    out = os.path.join(F, "outputs", "sources_and_corrections.csv")
    cols = ["code", "status", "sport", "category", "season", "item", "original_value", "revised_value", "evidence",
            "source_url", "record_level_impact"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r["status"] for r in rows))


if __name__ == "__main__":
    main()
