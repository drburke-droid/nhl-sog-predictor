"""Does a rich, impatient fan base stop a team from rebuilding, and does rebuilding pay?  python analyse.py
-> ../media/rebuild/results.json, team_level.csv, events.csv

Every measure is taken within its league (z-scores across the league's franchises, or a season's rank in the league as
a percentile, 0 the worst and 1 the best), so the four leagues pool.

H1, who rebuilds: each franchise's Forbes value, its market (metro population), and its premium (value beyond what the
market predicts, the residual of log value on log population within the league), against how it builds, over its
seasons in its current market (at least 10): how often it finishes in the bottom three; its roster's age against the
league that season (NHL, NBA, MLB); its own first-round picks made per draft (NHL, NBA, NFL); NFL first-round picks
traded away, net. Spearman correlations pooled over the leagues, with p-values from 10,000 shuffles within league.

Controls: each H1 association again with the franchise's average rank held fixed (Spearman partial correlation,
ranks residualised on the average rank within league), since a valuable team is usually also a good one.

H2, does it pay: from mediocrity (a season in the middle of the league, ranked 30-70 %, with no bottom-three finish in
the three seasons up to it), the teams that tore down (a bottom-three finish within the next two seasons) against the
teams that stayed in the middle (both next seasons ranked 25-75 %), over the following ten seasons: the peak (best rank
in years 3-10), a top-three finish, a title, the low (worst rank in years 1-10) and the average. Confidence intervals by
resampling franchises (2,000 times), since a franchise can start several windows. The same again over a grid of
definitions (bottom three or five, within two or three seasons, over eight or ten). And the champions: how many had a
bottom-three finish in the ten seasons before their title, against all teams in the same seasons (titles from the
eleventh season of the data on), with p from drawing one team per title season at random (20,000 times).
"""
import os, csv, json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "media", "rebuild")
LEAGUES = ["nhl", "nba", "mlb", "nfl"]
RNG = np.random.default_rng(20261026)
MID, STAY, HORIZON, BEFORE, AFTER = (0.30, 0.70), (0.25, 0.75), 10, 3, 2


def load(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def z(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def bottom3(s):
    return int(s["rank"]) > int(s["teams"]) - 3


def top3(s):
    return int(s["rank"]) <= 3


def team_level(seasons, teams, drafts, trades):
    """one row per franchise: owner/fan measures and how it builds, over its seasons in its current market"""
    rows = []
    for lg in LEAGUES:
        T = [t for t in teams if t["league"] == lg]
        lv, lp = np.log([float(t["value_musd"]) for t in T]), np.log([float(t["metro_pop"]) for t in T])
        b = np.polyfit(lp, lv, 1)
        prem = lv - np.polyval(b, lp)
        S = [s for s in seasons if s["league"] == lg]
        ages = defaultdict(list)
        for s in S:
            if s["age"]:
                ages[s["season"]].append(float(s["age"]))
        amean = {k: (np.mean(v), np.std(v)) for k, v in ages.items()}
        for t, v_z, m_z, p_z in zip(T, z(lv), z(lp), z(prem)):
            first = int(t["moved"]) if t["moved"] else 0
            mine = [s for s in S if s["team"] == t["team"] and int(s["season"]) >= first]
            if len(mine) < 10:
                continue
            r = dict(league=lg, team=t["team"], name=t["name"], seasons=len(mine), first=first, value_z=v_z, market_z=m_z, premium_z=p_z,
                     value_musd=float(t["value_musd"]), metro_pop=int(t["metro_pop"]),
                     tank_rate=np.mean([bottom3(s) for s in mine]), top_rate=np.mean([top3(s) for s in mine]),
                     titles=sum(int(s["champion"]) for s in mine), playoff_rate=np.mean([int(s["playoffs"]) for s in mine if s["playoffs"]]) if lg != "mlb" else np.nan,
                     sd_rank=np.std([float(s["rank_pct"]) for s in mine]), mean_rank=np.mean([float(s["rank_pct"]) for s in mine]))
            if amean and all(s["age"] for s in mine):
                r["age_z"] = np.mean([(float(s["age"]) - amean[s["season"]][0]) / amean[s["season"]][1] for s in mine])
            if lg in ("nhl", "nba", "nfl"):
                lo = max(2000, first - (0 if lg == "nfl" else 1))
                years = sorted({int(d["year"]) for d in drafts if d["league"] == lg and int(d["year"]) >= lo})
                years = [y for y in years if any(int(s["season"]) in (y, y + 1) for s in mine)]
                r["firsts_per_draft"] = sum(1 for d in drafts if d["league"] == lg and d["team"] == t["team"] and d["round"] == "1"
                                            and int(d["year"]) in years) / len(years)
            if lg == "nfl":
                lo = first or 2002
                given = sum(1 for x in trades if x["gave"] == t["team"] and x["pick_round"] == "1" and int(x["season"]) >= lo)
                got = sum(1 for x in trades if x["received"] == t["team"] and x["pick_round"] == "1" and int(x["season"]) >= lo)
                r["net_firsts_traded_away"] = (given - got) / len({int(s["season"]) for s in mine if int(s["season"]) >= lo})
            rows.append(r)
    for key in ("tank_rate", "top_rate", "sd_rank", "mean_rank", "titles", "age_z", "firsts_per_draft", "net_firsts_traded_away", "playoff_rate"):
        for lg in LEAGUES:
            R = [r for r in rows if r["league"] == lg and key in r and not np.isnan(r[key])]
            if len(R) > 3:
                for r, v in zip(R, z([r[key] for r in R])):
                    r[key + "_lz"] = v
    return rows


def assoc(rows, x, y, n=10000):
    """pooled Spearman correlation of two within-league measures, p from shuffling x within each league"""
    R = [r for r in rows if x in r and y in r and not np.isnan(r[y])]
    if len(R) < 8:
        return None
    xs, ys = np.array([r[x] for r in R]), np.array([r[y] for r in R])
    rho = spearmanr(xs, ys)[0]
    groups = [np.array([i for i, r in enumerate(R) if r["league"] == lg]) for lg in LEAGUES]
    groups = [g for g in groups if len(g)]
    null = np.empty(n)
    for k in range(n):
        xp = xs.copy()
        for g in groups:
            xp[g] = xs[RNG.permutation(g)]
        null[k] = spearmanr(xp, ys)[0]
    per = {lg: round(float(spearmanr([r[x] for r in R if r["league"] == lg], [r[y] for r in R if r["league"] == lg])[0]), 3)
           for lg in LEAGUES if sum(r["league"] == lg for r in R) > 5}
    return dict(rho=round(float(rho), 3), p=round(float(np.mean(np.abs(null) >= abs(rho))), 4), n=len(R), by_league=per)


def thirds(seasons, rows):
    """each league's franchises split into thirds by value: how their seasons spread over the league's ranks"""
    third = {}
    for lg in LEAGUES:
        R = sorted([r for r in rows if r["league"] == lg], key=lambda r: r["value_z"])
        for i, r in enumerate(R):
            third[(lg, r["team"])] = 3 * i // len(R)
    first = {(r["league"], r["team"]): r.get("first", 0) for r in rows}
    out = []
    for k in range(3):
        X = [s for s in seasons if third.get((s["league"], s["team"])) == k and int(s["season"]) >= first[(s["league"], s["team"])]]
        rp = np.array([float(s["rank_pct"]) for s in X])
        out.append(dict(third=["poorest", "middle", "richest"][k], seasons=len(X), mean_rank=round(float(rp.mean()), 3),
                        sd_rank=round(float(rp.std()), 3), bottom3=round(float(np.mean([bottom3(s) for s in X])), 3),
                        top3=round(float(np.mean([top3(s) for s in X])), 3), titles=sum(int(s["champion"]) for s in X),
                        deciles=np.round(np.histogram(rp, bins=10, range=(0, 1))[0] / len(rp), 4).tolist()))
    return out


def partial(rows, x, y, c, n=10000):
    """Spearman partial correlation of x and y with c held fixed (ranks residualised on c's ranks within league)"""
    R = [r for r in rows if all(k in r and not np.isnan(r[k]) for k in (x, y, c))]

    def res(k):
        out = np.empty(len(R))
        for lg in LEAGUES:
            idx = [i for i, r in enumerate(R) if r["league"] == lg]
            if idx:
                a, b = rankdata([R[i][k] for i in idx]), rankdata([R[i][c] for i in idx])
                out[idx] = a - np.polyval(np.polyfit(b, a, 1), b)
        return out

    rx, ry = res(x), res(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    groups = [g for g in (np.array([i for i, r in enumerate(R) if r["league"] == lg]) for lg in LEAGUES) if len(g)]
    null = np.empty(n)
    for k in range(n):
        xp = rx.copy()
        for g in groups:
            xp[g] = rx[RNG.permutation(g)]
        null[k] = np.corrcoef(xp, ry)[0, 1]
    return dict(rho=round(float(rho), 3), p=round(float(np.mean(np.abs(null) >= abs(rho))), 4), n=len(R))


def champions(seasons, look=10, n=20000):
    """champions with a bottom-three finish in the previous `look` seasons, against every team in those seasons"""
    by = defaultdict(list)
    for s in seasons:
        by[(s["league"], s["team"])].append(s)
    rows = []
    for (lg, team), S in by.items():
        S.sort(key=lambda s: int(s["season"]))
        for i in range(look, len(S)):
            rows.append(dict(league=lg, team=team, season=int(S[i]["season"]), champ=int(S[i]["champion"]),
                             tore=int(any(bottom3(x) for x in S[i - look:i]))))
    out = {}
    for lg in ["all"] + LEAGUES:
        R = [r for r in rows if lg == "all" or r["league"] == lg]
        C = [r for r in R if r["champ"]]
        pool = defaultdict(list)
        for r in R:
            pool[(r["league"], r["season"])].append(r["tore"])
        keys = [(r["league"], r["season"]) for r in C]
        ch = float(np.mean([r["tore"] for r in C]))
        sims = np.array([np.mean([pool[k][RNG.integers(len(pool[k]))] for k in keys]) for _ in range(n)])
        out[lg] = dict(titles=len(C), tore_first=round(ch, 3), all_teams=round(float(np.mean([r["tore"] for r in R])), 3),
                       p=round(float(np.mean(sims >= ch)), 4), champions=[[r["team"], r["season"], r["tore"]] for r in sorted(C, key=lambda r: r["season"])])
    return out


def events(seasons):
    """windows from mediocrity: what each team did next, and what came of it"""
    by = defaultdict(list)
    for s in seasons:
        by[(s["league"], s["team"])].append(s)
    out = []
    for (lg, team), S in by.items():
        S.sort(key=lambda s: int(s["season"]))
        for i in range(BEFORE - 1, len(S) - HORIZON):
            s = S[i]
            rp = float(s["rank_pct"])
            if not (MID[0] <= rp <= MID[1]) or any(bottom3(x) for x in S[i - BEFORE + 1:i + 1]):
                continue
            nxt, win = S[i + 1:i + 1 + AFTER], S[i + 1:i + 1 + HORIZON]
            if any(bottom3(x) for x in nxt):
                path = "teardown"
            elif all(STAY[0] <= float(x["rank_pct"]) <= STAY[1] for x in nxt):
                path = "stayed"
            else:
                path = "other"
            later = win[AFTER:]
            out.append(dict(league=lg, team=team, season=int(s["season"]), path=path,
                            peak=max(float(x["rank_pct"]) for x in later), top3=int(any(top3(x) for x in later)),
                            title=int(any(int(x["champion"]) for x in later)), low=min(float(x["rank_pct"]) for x in win),
                            mean=float(np.mean([float(x["rank_pct"]) for x in win])),
                            curve=[float(x["rank_pct"]) for x in S[i - 2:i + 1 + HORIZON]]))
    return out


def compare(ev, league=None, n=2000):
    E = [e for e in ev if e["path"] in ("teardown", "stayed") and (league is None or e["league"] == league)]
    keys = ("peak", "top3", "title", "low", "mean")
    res = {"n_teardown": sum(e["path"] == "teardown" for e in E), "n_stayed": sum(e["path"] == "stayed" for e in E)}
    for k in keys:
        a = np.mean([e[k] for e in E if e["path"] == "teardown"])
        b = np.mean([e[k] for e in E if e["path"] == "stayed"])
        res[k] = dict(teardown=round(float(a), 3), stayed=round(float(b), 3), diff=round(float(a - b), 3))
    teams = sorted({(e["league"], e["team"]) for e in E})
    idx = defaultdict(list)
    for i, e in enumerate(E):
        idx[(e["league"], e["team"])].append(i)
    boot = {k: [] for k in keys}
    for _ in range(n):
        pick = [i for t in RNG.choice(len(teams), len(teams)) for i in idx[teams[t]]]
        B = [E[i] for i in pick]
        for k in keys:
            a = [e[k] for e in B if e["path"] == "teardown"]
            b = [e[k] for e in B if e["path"] == "stayed"]
            if a and b:
                boot[k].append(np.mean(a) - np.mean(b))
    for k in keys:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        res[k]["ci95"] = [round(float(lo), 3), round(float(hi), 3)]
    return res


def main():
    seasons, teams, drafts, trades = load("seasons.csv"), load("teams.csv"), load("drafts.csv"), load("nfl_trades.csv")
    T = team_level(seasons, teams, drafts, trades)
    h1 = {}
    for x in ("value_z", "market_z", "premium_z"):
        for y in ("tank_rate_lz", "age_z_lz", "firsts_per_draft_lz", "net_firsts_traded_away_lz", "sd_rank_lz", "top_rate_lz", "mean_rank_lz", "titles_lz"):
            a = assoc(T, x, y)
            if a:
                h1[f"{x}~{y}"] = a
    h2_team = {f"tank_rate_lz~{y}": assoc(T, "tank_rate_lz", y) for y in ("top_rate_lz", "titles_lz", "sd_rank_lz", "mean_rank_lz")}
    controls = {f"{x}~{y}|mean_rank": partial(T, x, y, "mean_rank") for x in ("value_z", "market_z", "premium_z")
                for y in ("tank_rate", "age_z", "firsts_per_draft", "sd_rank", "top_rate", "titles")}
    ev = events(seasons)
    h2 = {"all": compare(ev), **{lg: compare(ev, lg) for lg in LEAGUES}}
    global bottom3, HORIZON, AFTER
    keep = (bottom3, HORIZON, AFTER)
    grid = {}
    for kb in (3, 5):
        for after in (2, 3):
            for horizon in (8, 10):
                bottom3 = (lambda k: lambda s: int(s["rank"]) > int(s["teams"]) - k)(kb)
                HORIZON, AFTER = horizon, after
                e = events(seasons)
                grid[f"bottom{kb}_within{after}_over{horizon}"] = {"all": compare(e, n=500), "nba": compare(e, "nba", n=500)}
    bottom3, HORIZON, AFTER = keep
    champs = champions(seasons)
    th = thirds(seasons, T)
    curves = {p: np.mean([e["curve"] for e in ev if e["path"] == p], axis=0).round(4).tolist() for p in ("teardown", "stayed")}

    print("H1: owner and fan measures against how franchises build (pooled Spearman, p by shuffling within league)")
    for k, a in h1.items():
        print(f"  {k:42s} rho {a['rho']:+.2f}  p {a['p']:.3f}  n {a['n']:3d}   " + " ".join(f"{lg} {v:+.2f}" for lg, v in a["by_league"].items()))
    print("\nH1 with the franchise's average rank held fixed")
    for k, a in controls.items():
        print(f"  {k:42s} rho {a['rho']:+.2f}  p {a['p']:.3f}  n {a['n']:3d}")
    print("\nH2, franchise level: tanking against peaks")
    for k, a in h2_team.items():
        print(f"  {k:42s} rho {a['rho']:+.2f}  p {a['p']:.3f}  n {a['n']:3d}   " + " ".join(f"{lg} {v:+.2f}" for lg, v in a["by_league"].items()))
    print("\nH2, from mediocrity: tore down vs stayed in the middle (next 10 seasons)")
    for g, r in h2.items():
        print(f"  {g:4s} n {r['n_teardown']:3d} / {r['n_stayed']:3d}  " + "  ".join(
            f"{k} {r[k]['teardown']:.2f} vs {r[k]['stayed']:.2f} ({r[k]['diff']:+.2f} [{r[k]['ci95'][0]:+.2f},{r[k]['ci95'][1]:+.2f}])" for k in ("peak", "top3", "title", "low", "mean")))
    print("\nH2 over other definitions: tore down minus stayed (peak, title; NBA title)")
    for k, g in grid.items():
        a, b = g["all"], g["nba"]
        print(f"  {k:24s} n {a['n_teardown']:3d}/{a['n_stayed']:3d}  peak {a['peak']['diff']:+.2f} {a['peak']['ci95']}  title {a['title']['diff']:+.2f} "
              f"{a['title']['ci95']}  low {a['low']['diff']:+.2f}  mean {a['mean']['diff']:+.2f}  | NBA title {b['title']['diff']:+.2f} {b['title']['ci95']}")
    print("\nfranchises by value, thirds of each league: how their seasons spread")
    for t in th:
        print(f"  {t['third']:8s} {t['seasons']} seasons: mean rank {t['mean_rank']:.2f}, spread {t['sd_rank']:.3f}, bottom three {t['bottom3']:.1%}, "
              f"top three {t['top3']:.1%}, {t['titles']} titles")
    print("\nchampions with a bottom-three finish in the ten seasons before the title, against all teams")
    for lg, c in champs.items():
        print(f"  {lg:4s} {c['titles']:2d} titles: {c['tore_first']:.0%} against {c['all_teams']:.0%}, p {c['p']:.3f}")
    json.dump(dict(h1=h1, controls=controls, thirds=th, h2_team=h2_team, h2=h2, grid=grid, champions=champs, curves=curves,
                   params=dict(mid=MID, stay=STAY, horizon=HORIZON, before=BEFORE, after=AFTER)),
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    with open(os.path.join(OUT, "team_level.csv"), "w", newline="") as f:
        keys = sorted({k for r in T for k in r}, key=lambda k: (k not in ("league", "team", "name"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in T])
    with open(os.path.join(OUT, "events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in ev[0] if k != "curve"])
        w.writeheader()
        w.writerows([{k: v for k, v in e.items() if k != "curve"} for e in ev])


if __name__ == "__main__":
    main()
