"""Whose Baseball Reference pages to read for salaries:  python candidates.py  -> data/candidates.txt

Anyone who could have been among the game's best-paid in some season: each season's 30 most valuable players by WAR
(1913-1984) and 40 most valuable (2014-2025), every All-Star (1933-1984, 2014 on) and every Hall of Fame player (the Lahman
database), and the 120 best-paid each season 2010-2016 (the Lahman salaries, whose contracts ran on past 2016). The
Lahman salaries cover 1985-2016 completely, so the pages matter for the years before and after.
"""
import os, re, sys, csv, tarfile, warnings
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
REBUILD = os.path.join(ROOT, "rebuild")
sys.path.insert(0, REBUILD)
from build import table, text                                   # noqa: E402  (the rebuild study's page helpers)


def lahman(name):
    import rdata
    warnings.simplefilter("ignore")
    tmp = os.path.join(DATA, "lahman", f"{name}.RData")
    if not os.path.exists(tmp):
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        tf = tarfile.open(os.path.join(REBUILD, "data", "lahman", "Lahman_14.0-0.tar.gz"))
        open(tmp, "wb").write(tf.extractfile(f"Lahman/data/{name}.RData").read())
    return rdata.read_rda(tmp)[name]


def value_page(y, kind):
    for base in (os.path.join(DATA, "sr", "mlb"), os.path.join(REBUILD, "data", "sr", "mlb")):
        p = os.path.join(base, f"value_{kind}_{y}.html")
        if os.path.exists(p):
            return open(p, encoding="utf8", errors="replace").read().replace("<!--", "").replace("-->", "")
    return None


def war(y):
    """{bbref id: WAR} for a season, batting and pitching together"""
    out = defaultdict(float)
    for kind, col in (("batting", "b_war"), ("pitching", "p_war")):
        h = value_page(y, kind)
        if not h:
            continue
        for r in table(h, f"players_value_{kind}"):
            m = re.search(r"/players/\w/([a-z0-9.'-]+)\.shtml", r.get("name_display", ""))
            team = text(r.get("team_name_abbr", ""))
            if m and not re.fullmatch(r"\d+TM|TOT", team):
                try:
                    out[m.group(1)] += float(text(r.get(col, "")) or 0)
                except ValueError:
                    pass
    return out


def main():
    """candidates in the order to fetch them: Hall of Famers, then players with two or more All-Star games, then the
    best-paid of 2010-2016 and the most valuable of 2014-2025, then the rest (so a first draft can start early)"""
    people = lahman("People")
    bb = {r.playerID: r.bbrefID for r in people.itertuples() if isinstance(r.bbrefID, str)}
    hof = lahman("HallOfFame")
    tier = {}

    def add(ids, t):
        for i in ids:
            tier[i] = min(tier.get(i, 9), t)
    add({bb[r.playerID] for r in hof.itertuples() if r.inducted == "Y" and r.category == "Player" and r.playerID in bb}, 0)
    asg = lahman("AllstarFull")
    games = defaultdict(int)
    for r in asg.itertuples():
        if (r.yearID <= 1984 or r.yearID >= 2014) and r.playerID in bb:
            games[bb[r.playerID]] += 1
    add({p for p, n in games.items() if n >= 2}, 1)
    sal = lahman("Salaries")
    for y in range(2010, 2017):
        s = sal[sal.yearID == y].sort_values("salary", ascending=False).head(120)
        add({bb[p] for p in s.playerID if p in bb}, 2)
    for y in list(range(1913, 1985)) + list(range(2014, 2026)):
        w = war(y)
        add(set(sorted(w, key=w.get, reverse=True)[:30 if y < 1985 else 40]), 2 if y >= 2014 else 3)
    add(set(games), 3)
    ids = sorted(tier, key=lambda i: (tier[i], i))
    open(os.path.join(DATA, "candidates.txt"), "w").write("\n".join(ids) + "\n")
    print(f"{len(ids)} candidates:", {t: sum(v == t for v in tier.values()) for t in sorted(set(tier.values()))})


if __name__ == "__main__":
    main()
