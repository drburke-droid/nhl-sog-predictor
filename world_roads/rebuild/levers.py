"""Payroll, free agents and trades: who pulls which lever, what it buys, and whether selling pays, league by league:
python levers.py  -> ../media/rebuild/levers.json, lever_events.csv

Every team-season from moves.py (team_moves.csv) with its finish (seasons.csv). In stars (1.0 = an all-star-level
season, the league's top 5 % of player-seasons): stars traded in and out, judged by each player's season before (and,
mid-season, what he had done that season for the team that traded him); future assets traded in and out (a first-round
pick 1, a second 0.35, later picks 0.1, a prospect, young and without a track record, 0.5); stars signed from other
teams in free agency and lost to them. Payroll as a share of the league's average that season.

* A *sell* season: at least half a star traded away, net, and at least half a future asset back (star out, prospect
  in). A *buy* season: the reverse (prospects and picks out, at least half a star in).
* profiles: per league, how unequal payrolls are, how often teams sell and buy, and what each lever goes with: the
  season's rank against payroll (and against payroll within a franchise, its own richer and poorer years), and the
  change in rank from the season before against stars traded in the offseason (deadline deals follow the season as
  much as they shape it) and signed, net. Confidence intervals by resampling
  franchises (2,000 times).
* who: per league, each franchise's Forbes value, market and premium (as in analyse.py) against how it builds over its
  seasons in its current market: average payroll, sell and buy rates, stars signed per season, firsts traded away per
  season; Spearman, p from 10,000 shuffles; and with the franchise's average rank held fixed.
* bad years: after a season in the bottom third of the league, what a team does in the next two (sells, buys, signs,
  cuts payroll), by the franchise's value: the richest third of each league against the poorest third.
* teardown: from mediocrity (ranked 30-70 %), what a team did over the next two seasons, ranked within its league
  (each league trades on its own scale): *sold*, the fifth that traded away the most stars, net, taking future assets
  back (star out, prospect in); *bought*, the fifth that added the most stars, net, by trade and free agency; and
  *neither*, the middle (30-70 %) of both. Then over the ten seasons after: the peak (best rank in years 3-10), a
  top-three finish, a title, the low (worst rank in years 1-10), the average. Per league; again by payroll (the fifth
  that cut the most against the league, the fifth that added the most; NBA and MLB, the leagues with payroll long
  enough); and over a grid (fifths or thirds, eight or ten seasons). Sellers are often teams already sliding, so each
  gap also comes adjusted: least squares with the starting rank, the trend into it (the change in rank over the two
  seasons before) and the roster's age held fixed.
* champions: each champion's standing among that season's teams (0.5 typical) in stars traded, net, over seasons t-10
  to t-3; in stars traded and signed, net, over t-2 to t; and in payroll (every title season with payroll); p from a
  sign test.

By time horizon (horizon.py; NBA, NHL, MLB), the same questions with trades valued by when their value arrives and
how long the team kept it, salary set apart: a season or window that gave up present value (projected over the season
and the next) and came out ahead in future value (the four after) tore down; one that gave up as much with nothing
back while shedding salary dumped; one that added present value paying in future value went for it. Who does each
(per franchise), what rich and poor teams do after a bad year, what came of each from mediocrity, how the champions
traded before their titles, and how the first classification's sellers and buyers sort under this one.
"""
import os, csv, json
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "media", "rebuild")
LEAGUES = ["nhl", "nba", "mlb", "nfl"]
RNG = np.random.default_rng(20261027)
SELL, BUY = 0.5, 0.5                    # a sell / buy season: half a star out / in, net, half a future asset the other way
MID, HORIZON, AFTER, Q = (0.30, 0.70), 10, 2, 0.2
F = lambda v: float(v) if v not in ("", None) else np.nan


def load(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def rows():
    """team-seasons with finish, moves and payroll, in order, each with the season before"""
    seasons = load("seasons.csv")
    tm = {(r["league"], r["season"], r["team"]): r for r in load("team_moves.csv")}
    teams = {(r["league"], r["team"]): r for r in load("teams.csv")}
    out = defaultdict(list)
    for s in seasons:
        m = tm.get((s["league"], s["season"], s["team"]), {})
        r = dict(league=s["league"], team=s["team"], season=int(s["season"]), rank=F(s["rank_pct"]), age=F(s["age"]),
                 champion=int(s["champion"]), top3=int(s["rank"]) <= 3, bottom3=int(s["rank"]) > int(s["teams"]) - 3,
                 moved=teams[(s["league"], s["team"])]["moved"])
        for k in ("trade_net", "future_net", "trade_net_off", "trade_stars_in", "trade_stars_out", "trade_future_in", "trade_future_out",
                  "fa_stars_in", "fa_stars_out", "fa_net", "payroll_rel", "firsts_in", "firsts_out", "stars_in", "stars_out"):
            r[k] = F(m.get(k))
        r["sell"] = float(r["trade_net"] <= -SELL and r["future_net"] >= SELL) if not np.isnan(r["trade_net"]) else np.nan
        r["buy"] = float(r["trade_net"] >= BUY and r["future_net"] <= -BUY) if not np.isnan(r["trade_net"]) else np.nan
        out[(s["league"], s["team"])].append(r)
    ages = defaultdict(list)                  # roster age against the league that season
    for S in out.values():
        for r in S:
            ages[(r["league"], r["season"])].append(r["age"])
    for S in out.values():
        for r in S:
            a = np.array(ages[(r["league"], r["season"])], float)
            r["age_z"] = (r["age"] - np.nanmean(a)) / np.nanstd(a) if not np.isnan(r["age"]) else np.nan
    for S in out.values():
        S.sort(key=lambda r: r["season"])
        for a, b in zip(S, S[1:]):
            b["prev_rank"] = a["rank"]
            b["d_rank"] = b["rank"] - a["rank"]
            b["d_pay"] = b["payroll_rel"] - a["payroll_rel"]
        S[0]["prev_rank"] = S[0]["d_rank"] = S[0]["d_pay"] = np.nan
    return out


def clusters(R):
    idx = defaultdict(list)
    for i, r in enumerate(R):
        idx[(r["league"], r["team"])].append(i)
    return list(idx.values())


def boot(R, stat, n=2000):
    """a statistic of team-season rows, and its 95 % interval from resampling franchises"""
    est = stat(R)
    cl = clusters(R)
    vals = []
    for _ in range(n):
        pick = [i for c in RNG.integers(0, len(cl), len(cl)) for i in cl[c]]
        v = stat([R[i] for i in pick])
        if v is not None and not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(est=round(float(est), 3), ci95=[round(float(lo), 3), round(float(hi), 3)], n=len(R))


def rho(x, y):
    def f(R):
        a = np.array([r[x] for r in R], float)
        b = np.array([r[y] for r in R], float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        return spearmanr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan
    return f


def within(R, x):
    """x as the franchise's own deviation from its average"""
    by = defaultdict(list)
    for r in R:
        if not np.isnan(r[x]):
            by[(r["league"], r["team"])].append(r[x])
    for r in R:
        v = by.get((r["league"], r["team"]))
        r[x + "_w"] = r[x] - np.mean(v) if v and not np.isnan(r[x]) else np.nan


def profiles(T):
    out = {}
    for lg in LEAGUES:
        R = [r for S in T.values() for r in S if r["league"] == lg]
        within(R, "payroll_rel")
        within(R, "rank")
        P = [r for r in R if not np.isnan(r["payroll_rel"])]
        Tr = [r for r in R if not np.isnan(r["trade_net"])]
        Fa = [r for r in R if not np.isnan(r["fa_net"])]
        cv = [np.std([r["payroll_rel"] for r in P if r["season"] == s]) for s in sorted({r["season"] for r in P})]
        top = [r for r in P if r["payroll_rel"] >= np.percentile([q["payroll_rel"] for q in P if q["season"] == r["season"]], 80)]
        out[lg] = dict(
            seasons=dict(payroll=[min(r["season"] for r in P), max(r["season"] for r in P)] if P else None,
                         trades=[min(r["season"] for r in Tr), max(r["season"] for r in Tr)],
                         fa=[min(r["season"] for r in Fa), max(r["season"] for r in Fa)]),
            payroll_spread=round(float(np.median(cv)), 3) if cv else None,
            top_fifth_payroll_avg_rank=round(float(np.mean([r["rank"] for r in top])), 3) if top else None,
            sell_rate=round(float(np.mean([r["sell"] for r in Tr])), 3),
            buy_rate=round(float(np.mean([r["buy"] for r in Tr])), 3),
            stars_traded_per_season=round(float(np.mean([r["trade_stars_in"] for r in Tr])), 3),
            stars_signed_per_season=round(float(np.mean([r["fa_stars_in"] for r in Fa])), 3) if Fa else None,
            payroll_vs_rank=boot(P, rho("payroll_rel", "rank")) if P else None,
            payroll_vs_rank_within=boot(P, rho("payroll_rel_w", "rank_w")) if P else None,
            d_payroll_vs_d_rank=boot([r for r in P if not np.isnan(r["d_pay"])], rho("d_pay", "d_rank")) if P else None,
            trade_net_vs_d_rank=boot([r for r in Tr if not np.isnan(r["d_rank"])], rho("trade_net_off", "d_rank")),
            fa_net_vs_d_rank=boot([r for r in Fa if not np.isnan(r["d_rank"])], rho("fa_net", "d_rank")) if Fa else None,
            sell_next_rank=boot([r for r in Tr if not np.isnan(r["d_rank"])], lambda R: np.mean([r["d_rank"] for r in R if r["sell"] == 1]) - np.mean([r["d_rank"] for r in R if r["sell"] == 0])),
        )
    return out


def team_measures(T):
    """per franchise over its seasons in its current market: value, market, premium (within league), how it builds"""
    teams = load("teams.csv")
    out = []
    for lg in LEAGUES:
        TT = [t for t in teams if t["league"] == lg]
        lv, lp = np.log([float(t["value_musd"]) for t in TT]), np.log([float(t["metro_pop"]) for t in TT])
        prem = lv - np.polyval(np.polyfit(lp, lv, 1), lp)
        zs = lambda v: (v - v.mean()) / v.std()
        for t, v, m, p in zip(TT, zs(lv), zs(lp), zs(prem)):
            S = [r for r in T[(lg, t["team"])] if not t["moved"] or r["season"] >= int(t["moved"])]
            row = dict(league=lg, team=t["team"], value=v, market=m, premium=p, mean_rank=np.mean([r["rank"] for r in S]))
            for k, col in (("payroll", "payroll_rel"), ("sell_rate", "sell"), ("buy_rate", "buy"),
                           ("signed_stars", "fa_stars_in"), ("firsts_out", "firsts_out"), ("traded_stars_in", "trade_stars_in")):
                vals = [r[col] for r in S if not np.isnan(r[col])]
                row[k] = float(np.mean(vals)) if len(vals) >= 8 else np.nan
            out.append(row)
    return out


def corr_test(R, x, y, c=None, n=10000):
    R = [r for r in R if not np.isnan(r[y])]
    if len(R) < 8 or np.std([r[y] for r in R]) == 0:        # MLB trades no picks
        return None
    xs, ys = rankdata([r[x] for r in R]), rankdata([r[y] for r in R])
    if c:
        cs = rankdata([r[c] for r in R])
        xs = xs - np.polyval(np.polyfit(cs, xs, 1), cs)
        ys = ys - np.polyval(np.polyfit(cs, ys, 1), cs)
    r0 = np.corrcoef(xs, ys)[0, 1]
    null = np.array([np.corrcoef(RNG.permutation(xs), ys)[0, 1] for _ in range(n)])
    return dict(rho=round(float(r0), 3), p=round(float(np.mean(np.abs(null) >= abs(r0))), 4), n=len(R))


def who(T):
    M = team_measures(T)
    out = {}
    for lg in LEAGUES:
        R = [r for r in M if r["league"] == lg]
        out[lg] = {f"{x}~{y}": dict(raw=corr_test(R, x, y), at_equal_rank=corr_test(R, x, y, "mean_rank"))
                   for x in ("value", "market", "premium")
                   for y in ("payroll", "sell_rate", "buy_rate", "signed_stars", "firsts_out", "traded_stars_in")}
    return out, M


def bad_years(T, M):
    """after a bottom-third season, the next two: rich (top third of value in the league) against poor (bottom third)"""
    tier = {}
    for lg in LEAGUES:
        R = sorted([r for r in M if r["league"] == lg], key=lambda r: r["value"])
        n = len(R)
        for i, r in enumerate(R):
            tier[(lg, r["team"])] = "poor" if i < n / 3 else "rich" if i >= n - n / 3 else "middle"
    out = {}
    for lg in LEAGUES:
        E = []
        for (l, team), S in T.items():
            if l != lg:
                continue
            moved = next((r["moved"] for r in S if r["moved"]), "")
            for i in range(1, len(S) - 2):
                if S[i - 1]["rank"] > 1 / 3 or (moved and S[i - 1]["season"] < int(moved)):
                    continue
                nx = S[i:i + 2]
                e = dict(league=lg, team=team, tier=tier[(lg, team)],
                         sold=max(r["sell"] for r in nx), bought=max(r["buy"] for r in nx),
                         signed=sum(r["fa_stars_in"] for r in nx), net_stars=sum(r["trade_net"] for r in nx),
                         d_pay=S[i + 1]["payroll_rel"] - S[i - 1]["payroll_rel"])
                E.append(e)
        res = {}
        for k in ("sold", "bought", "signed", "net_stars", "d_pay"):
            ok = [e for e in E if not np.isnan(e[k]) and e["tier"] != "middle"]
            if len(ok) < 20:
                continue
            res[k] = dict(rich=round(float(np.mean([e[k] for e in ok if e["tier"] == "rich"])), 3),
                          poor=round(float(np.mean([e[k] for e in ok if e["tier"] == "poor"])), 3),
                          diff=boot(ok, lambda R, k=k: np.mean([e[k] for e in R if e["tier"] == "rich"]) - np.mean([e[k] for e in R if e["tier"] == "poor"])))
        res["n"] = dict(rich=sum(e["tier"] == "rich" for e in E), poor=sum(e["tier"] == "poor" for e in E))
        out[lg] = res
    return out


def lever_events(T, horizon=HORIZON, mid=MID):
    """windows from mediocrity: what the team did over the next two seasons (stars traded, future assets traded, stars
    signed, payroll), and what came of it over the next ten"""
    out = []
    for (lg, team), S in T.items():
        for i in range(len(S) - horizon):
            s = S[i]
            if not (mid[0] <= s["rank"] <= mid[1]):
                continue
            nx, win = S[i + 1:i + 1 + AFTER], S[i + 1:i + 1 + horizon]
            if any(np.isnan(r["trade_net"]) for r in nx) or win[-1]["season"] - s["season"] > horizon + 1:
                continue
            later = win[AFTER:]
            out.append(dict(league=lg, team=team, season=s["season"], start=s["rank"], age_z=s["age_z"],
                            trend=s["rank"] - S[i - 2]["rank"] if i >= 2 else np.nan,
                            traded=round(sum(r["trade_net"] for r in nx), 3), future=round(sum(r["future_net"] for r in nx), 3),
                            signed=round(sum(0 if np.isnan(r["fa_net"]) else r["fa_net"] for r in nx), 3),
                            d_pay=round(nx[-1]["payroll_rel"] - s["payroll_rel"], 4),
                            peak=max(r["rank"] for r in later), top3=int(any(r["top3"] for r in later)),
                            title=int(any(r["champion"] for r in later)), low=min(r["rank"] for r in win),
                            mean=float(np.mean([r["rank"] for r in win])),
                            curve=[r["rank"] for r in S[max(0, i - 2):i + 1 + horizon]]))
    return out


def paths(ev, q=Q, by="trades"):
    """within each league: 'sold', the fifth that traded away the most talent and took future assets back; 'bought',
    the fifth that added the most, by trade and free agency; 'neither', the middle (30-70 %) of both. By payroll:
    'sold', the fifth that cut the most against the league, 'bought' the fifth that added the most."""
    for lg in LEAGUES:
        E = [e for e in ev if e["league"] == lg]
        if by == "payroll":
            E = [e for e in E if not np.isnan(e["d_pay"])]
            if len(E) < 40:
                continue
            d = np.array([e["d_pay"] for e in E])
            lo, hi, m1, m2 = np.quantile(d, [q, 1 - q, 0.3, 0.7])
            for e in E:
                e["path_pay"] = ("sold" if e["d_pay"] <= min(lo, 0) else "bought" if e["d_pay"] >= max(hi, 0) else
                                 "neither" if m1 <= e["d_pay"] <= m2 else "other")
            continue
        tr = np.array([e["traded"] for e in E])
        add = np.array([e["traded"] + e["signed"] for e in E])
        t_lo = np.quantile(tr, q)
        a_hi, a1, a2 = np.quantile(add, [1 - q, 0.3, 0.7])
        t1, t2 = np.quantile(tr, [0.3, 0.7])
        for e in E:
            a = e["traded"] + e["signed"]
            e["path"] = ("sold" if e["traded"] <= min(t_lo, -0.05) and e["future"] > 0 else "bought" if a >= max(a_hi, 0.05) else
                         "neither" if a1 <= a <= a2 and t1 <= e["traded"] <= t2 else "other")
    return ev


def compare(ev, a, b, league=None, n=2000, key="path"):
    E = [e for e in ev if e.get(key) in (a, b) and (league is None or e["league"] == league)]
    res = {f"n_{a}": sum(e[key] == a for e in E), f"n_{b}": sum(e[key] == b for e in E)}
    if min(res.values()) < 5:
        return res
    for k in ("peak", "top3", "title", "low", "mean"):
        stat = lambda R, k=k: (np.mean([e[k] for e in R if e[key] == a]) - np.mean([e[k] for e in R if e[key] == b])
                               if any(e[key] == a for e in R) and any(e[key] == b for e in R) else np.nan)
        res[k] = dict({a: round(float(np.mean([e[k] for e in E if e[key] == a])), 3),
                       b: round(float(np.mean([e[k] for e in E if e[key] == b])), 3)}, diff=boot(E, stat, n))
    return res


def adjusted(ev, a, b, league=None, n=1000, key="path"):
    """the a-b gap with the starting rank, the trend into it (rank change over the two seasons before) and roster age
    (not in the NFL) held fixed: least squares within league, CI from resampling franchises"""
    E = [e for e in ev if e.get(key) in (a, b) and (league is None or e["league"] == league) and not np.isnan(e["trend"])]
    if sum(e[key] == a for e in E) < 5 or sum(e[key] == b for e in E) < 5:
        return None

    def fit(R, k):
        X = [[float(e[key] == a), e["start"], e["trend"]] + ([0.0 if np.isnan(e["age_z"]) else e["age_z"]] if league != "nfl" else [])
             + ([float(e["league"] == l) for l in LEAGUES[1:]] if league is None else []) for e in R]
        X = np.column_stack([np.ones(len(R)), np.array(X)])
        y = np.array([e[k] for e in R], float)
        return np.linalg.lstsq(X, y, rcond=None)[0][1]
    out = {}
    for k in ("peak", "title", "low", "mean"):
        out[k] = boot(E, lambda R, k=k: fit(R, k) if len({e[key] for e in R}) == 2 else np.nan, n)
    out["groups"] = {p: dict(start=round(float(np.mean([e["start"] for e in E if e[key] == p])), 3),
                             trend=round(float(np.mean([e["trend"] for e in E if e[key] == p])), 3),
                             age_z=round(float(np.nanmean([e["age_z"] for e in E if e[key] == p])), 3) if league != "nfl" else None)
                     for p in (a, b)}
    return out


def teardown(T):
    ev = paths(paths(lever_events(T)), by="payroll")
    res = {"main": {lg or "all": dict(sold_vs_neither=compare(ev, "sold", "neither", lg),
                                      bought_vs_neither=compare(ev, "bought", "neither", lg),
                                      sold_vs_bought=compare(ev, "sold", "bought", lg),
                                      adjusted=dict(sold_vs_neither=adjusted(ev, "sold", "neither", lg),
                                                    bought_vs_neither=adjusted(ev, "bought", "neither", lg)))
                    for lg in LEAGUES + [None]}}
    res["payroll"] = {lg or "all": dict(cut_vs_neither=compare(ev, "sold", "neither", lg, key="path_pay"),
                                        added_vs_neither=compare(ev, "bought", "neither", lg, key="path_pay"),
                                        cut_vs_added=compare(ev, "sold", "bought", lg, key="path_pay"))
                      for lg in ("nba", "mlb", None)}
    grid = []
    for q in (0.2, 1 / 3):
        for hz in (8, 10):
            e = paths(lever_events(T, hz), q)
            for lg in LEAGUES + [None]:
                for a, b in (("sold", "neither"), ("bought", "neither"), ("sold", "bought")):
                    c = compare(e, a, b, lg, n=500)
                    grid.append(dict(groups=round(q, 3), horizon=hz, league=lg or "all", comparison=f"{a}_vs_{b}",
                                     n_a=c.get(f"n_{a}"), n_b=c.get(f"n_{b}"),
                                     **{k: c[k]["diff"]["est"] for k in ("peak", "title", "low", "mean") if k in c},
                                     **{k + "_ci": c[k]["diff"]["ci95"] for k in ("peak", "title") if k in c}))
    res["grid"] = grid
    return res, ev


def champions(T, look=10):
    """what each champion had done before its title (stars traded, net, in seasons t-10 to t-3; stars traded and signed,
    net, in t-2 to t; payroll that season), against every other team in the same seasons"""
    out = {}
    for lg in LEAGUES:
        champs, base = [], []
        for (l, team), Ss in T.items():
            if l != lg:
                continue
            for i, r in enumerate(Ss):
                early = [x for x in Ss[max(0, i - look):max(0, i - 2)] if not np.isnan(x["trade_net"])]
                late = [x for x in Ss[max(0, i - 2):i + 1] if not np.isnan(x["trade_net"])]
                if len(early) < look - 3 or len(late) < 3:
                    continue
                rec = dict(season=r["season"], pay=r["payroll_rel"],
                           early=sum(x["trade_net"] for x in early),
                           late=sum(x["trade_net"] + (0 if np.isnan(x["fa_net"]) else x["fa_net"]) for x in late))
                (champs if r["champion"] else base).append(rec)
        if not champs:
            continue
        res = dict(n=len(champs), seasons=[min(c["season"] for c in champs), max(c["season"] for c in champs)])
        # payroll: every title season with payroll, whether or not the trades reach back ten years
        pay = [(r["season"], r["payroll_rel"], r["champion"]) for (l, _), Ss in T.items() if l == lg for r in Ss
               if not np.isnan(r["payroll_rel"])]
        pct = [float(np.mean([p < c[1] for p in [q[1] for q in pay if q[0] == c[0] and not q[2]]])) for c in pay if c[2]]
        if pct:
            from scipy.stats import binomtest
            above = sum(p > 0.5 for p in pct)
            res["pay"] = dict(percentile=round(float(np.mean(pct)), 3), n=len(pct), above_half=above,
                              p=round(float(binomtest(above, len(pct)).pvalue), 4),
                              seasons=[min(q[0] for q in pay if q[2]), max(q[0] for q in pay if q[2])])
        for k in ("early", "late"):
            pct = []
            for c in champs:
                same = np.array([b[k] for b in base if b["season"] == c["season"] and not np.isnan(b[k])])
                if not np.isnan(c[k]) and len(same):
                    pct.append(float(np.mean(same < c[k]) + 0.5 * np.mean(same == c[k])))
            # each champion's standing among that season's other teams (0.5 = typical), and p from a sign test on it
            if pct:
                from scipy.stats import binomtest
                above = sum(p > 0.5 for p in pct)
                res[k] = dict(percentile=round(float(np.mean(pct)), 3), n=len(pct), above_half=above,
                              p=round(float(binomtest(above, len(pct)).pvalue), 4))
        out[lg] = res
    return out


# ---------------------------------------------------------------- by time horizon (horizon.py): NBA, NHL, MLB

HL = ["nhl", "nba", "mlb"]
DUMP = 0.10                             # a salary dump: a tenth of the league's average payroll shed, net, in trades


def h_rows(T):
    """each team-season's moves by horizon (stars): traded and signed, net, now and later; salary traded, net"""
    th = {(r["league"], int(r["season"]), r["team"]): r for r in load("team_horizon.csv")}
    for (lg, team), S in T.items():
        for r in S:
            h = th.get((lg, r["season"], team), {})
            g = lambda k: F(h.get(k, ""))
            tn = g("trade_now_in") - g("trade_now_out") if h.get("trade_now_in", "") != "" else np.nan
            tl = g("trade_later_in") - g("trade_later_out") if h.get("trade_now_in", "") != "" else np.nan
            fn = g("fa_now_in") - g("fa_now_out") if h.get("fa_now_in", "") != "" else 0.0
            fl = g("fa_later_in") - g("fa_later_out") if h.get("fa_now_in", "") != "" else 0.0
            r.update(h_trade_now=tn, h_trade_later=tl, h_now=tn + fn, h_later=tl + fl,
                     h_salary=g("salary_in") - g("salary_out") if h.get("salary_in", "") != "" else np.nan)
    for lg in HL:                       # a season traded present for future (or the reverse): the league's fifth
        R = [r for (l, _), S in T.items() if l == lg for r in S if not np.isnan(r["h_now"])]
        lo, hi = np.quantile([r["h_now"] for r in R], [Q, 1 - Q])
        for r in R:
            r["pf"] = float(r["h_now"] <= min(lo, 0) and r["h_later"] > 0)
            r["fp"] = float(r["h_now"] >= max(hi, 0) and r["h_later"] < 0)
    for (lg, _), S in T.items():
        for r in S:
            r.setdefault("pf", np.nan)
            r.setdefault("fp", np.nan)
    return T


def h_events(T, horizon=HORIZON, mid=MID, q=Q):
    """windows from mediocrity, by what the next two seasons did to the present and the future (within league):
    tore_down: the fifth that gave up the most present value, net, and came out ahead in future value;
    dumped / shed: gave up as much present value with no future back, shedding a tenth of an average payroll or more
    in salary (dumped) or not (shed: stars walking, deals for nothing); went_for_it: the fifth that added the most
    present value, paying for it in future value; added: added as much and lost no future; neither: the middle
    (30-70 %) of both"""
    ev = []
    for (lg, team), S in T.items():
        if lg not in HL:
            continue
        for i in range(len(S) - horizon):
            s = S[i]
            if not (mid[0] <= s["rank"] <= mid[1]):
                continue
            nx, win = S[i + 1:i + 1 + AFTER], S[i + 1:i + 1 + horizon]
            if any(np.isnan(r["h_trade_now"]) for r in nx) or win[-1]["season"] - s["season"] > horizon + 1:
                continue
            sal = [r["h_salary"] for r in nx]
            later = win[AFTER:]
            ev.append(dict(league=lg, team=team, season=s["season"], start=s["rank"], age_z=s["age_z"],
                           trend=s["rank"] - S[i - 2]["rank"] if i >= 2 else np.nan,
                           now=sum(r["h_now"] for r in nx), fut=sum(r["h_later"] for r in nx),
                           salary=np.nan if any(np.isnan(x) for x in sal) else sum(sal),
                           peak=max(r["rank"] for r in later), top3=int(any(r["top3"] for r in later)),
                           title=int(any(r["champion"] for r in later)), low=min(r["rank"] for r in win),
                           mean=float(np.mean([r["rank"] for r in win]))))
    for lg in HL:
        E = [e for e in ev if e["league"] == lg]
        now, fut = np.array([e["now"] for e in E]), np.array([e["fut"] for e in E])
        lo, hi, n1, n2 = np.quantile(now, [q, 1 - q, 0.3, 0.7])
        f1, f2 = np.quantile(fut, [0.3, 0.7])
        for e in E:
            if e["now"] <= min(lo, 0):
                e["hpath"] = ("tore_down" if e["fut"] > 0 else
                              "dumped" if not np.isnan(e["salary"]) and e["salary"] <= -DUMP else "shed")
            elif e["now"] >= max(hi, 0):
                e["hpath"] = "went_for_it" if e["fut"] < 0 else "added"
            else:
                e["hpath"] = "neither" if n1 <= e["now"] <= n2 and f1 <= e["fut"] <= f2 else "other"
    return ev


def h_teardown(T):
    ev = h_events(T)
    pairs = (("tore_down", "neither"), ("went_for_it", "neither"), ("tore_down", "went_for_it"), ("dumped", "neither"),
             ("shed", "neither"), ("added", "neither"))
    res = {"main": {lg or "all": {f"{a}_vs_{b}": compare(ev, a, b, lg, key="hpath") for a, b in pairs}
                    for lg in HL + [None]}}
    for lg in HL + [None]:
        res["main"][lg or "all"]["adjusted"] = {f"{a}_vs_{b}": adjusted(ev, a, b, lg, key="hpath")
                                                for a, b in pairs[:3]}
    res["counts"] = {lg: dict(Counter(e["hpath"] for e in ev if e["league"] == lg)) for lg in HL}
    grid = []
    for q in (0.2, 1 / 3):
        for hz in (8, 10):
            e = h_events(T, hz, q=q)
            for lg in HL + [None]:
                for a, b in pairs[:3]:
                    c = compare(e, a, b, lg, n=500, key="hpath")
                    grid.append(dict(groups=round(q, 3), horizon=hz, league=lg or "all", comparison=f"{a}_vs_{b}",
                                     n_a=c.get(f"n_{a}"), n_b=c.get(f"n_{b}"),
                                     **{k: c[k]["diff"]["est"] for k in ("peak", "title", "low", "mean") if k in c},
                                     **{k + "_ci": c[k]["diff"]["ci95"] for k in ("peak", "title") if k in c}))
    res["grid"] = grid
    return res, ev


def h_who(T, M):
    """per franchise (current market, at least eight seasons): how often it traded present for future (pf) and future
    for present (fp), and salary traded, net, per season (a share of the league's average payroll)"""
    rows_ = []
    for m in M:
        if m["league"] not in HL:
            continue
        t = next(t for t in load_teams() if t["league"] == m["league"] and t["team"] == m["team"])
        S = [r for r in T[(m["league"], m["team"])] if not t["moved"] or r["season"] >= int(t["moved"])]
        S = [r for r in S if not np.isnan(r["h_trade_now"])]
        sal = [r["h_salary"] for r in S if not np.isnan(r["h_salary"])]
        enough = len(S) >= 8
        rows_.append(dict(m, pf_rate=float(np.mean([r["pf"] for r in S])) if enough else np.nan,
                          fp_rate=float(np.mean([r["fp"] for r in S])) if enough else np.nan,
                          salary_traded=float(np.mean(sal)) if len(sal) >= 8 else np.nan))
    out = {}
    for lg in HL:
        R = [r for r in rows_ if r["league"] == lg]
        out[lg] = {f"{x}~{y}": dict(raw=corr_test(R, x, y), at_equal_rank=corr_test(R, x, y, "mean_rank"))
                   for x in ("value", "market", "premium") for y in ("pf_rate", "fp_rate", "salary_traded")}
    return out


_TEAMS = []


def load_teams():
    if not _TEAMS:
        _TEAMS.extend(load("teams.csv"))
    return _TEAMS


def h_bad_years(T, M):
    """after a bottom-third season, the next two: present traded for future, future for present, present and future
    value moved (stars), salary traded; rich (top third of the league by value) against poor (bottom third)"""
    tier = {}
    for lg in HL:
        R = sorted([r for r in M if r["league"] == lg], key=lambda r: r["value"])
        for i, r in enumerate(R):
            tier[(lg, r["team"])] = "poor" if i < len(R) / 3 else "rich" if i >= len(R) - len(R) / 3 else "middle"
    out = {}
    for lg in HL:
        E = []
        for (l, team), S in T.items():
            if l != lg:
                continue
            moved = next((r["moved"] for r in S if r["moved"]), "")
            for i in range(1, len(S) - 2):
                if S[i - 1]["rank"] > 1 / 3 or (moved and S[i - 1]["season"] < int(moved)):
                    continue
                nx = S[i:i + 2]
                if any(np.isnan(r["h_trade_now"]) for r in nx):
                    continue
                sal = [r["h_salary"] for r in nx]
                E.append(dict(league=lg, team=team, tier=tier[(lg, team)], pf=max(r["pf"] for r in nx), fp=max(r["fp"] for r in nx),
                              now=sum(r["h_now"] for r in nx), fut=sum(r["h_later"] for r in nx),
                              salary=np.nan if any(np.isnan(x) for x in sal) else sum(sal)))
        res = {}
        for k in ("pf", "fp", "now", "fut", "salary"):
            ok = [e for e in E if not np.isnan(e[k]) and e["tier"] != "middle"]
            if len(ok) < 20 or not any(e["tier"] == "rich" for e in ok) or not any(e["tier"] == "poor" for e in ok):
                continue
            res[k] = dict(rich=round(float(np.mean([e[k] for e in ok if e["tier"] == "rich"])), 3),
                          poor=round(float(np.mean([e[k] for e in ok if e["tier"] == "poor"])), 3),
                          diff=boot(ok, lambda R, k=k: np.mean([e[k] for e in R if e["tier"] == "rich"]) - np.mean([e[k] for e in R if e["tier"] == "poor"])
                                    if any(e["tier"] == "rich" for e in R) and any(e["tier"] == "poor" for e in R) else np.nan))
        res["n"] = dict(rich=sum(e["tier"] == "rich" for e in E), poor=sum(e["tier"] == "poor" for e in E))
        out[lg] = res
    return out


def h_champions(T, look=10):
    """each champion's standing among that season's teams in present and future value moved, net, over t-10 to t-3
    and over t-2 to t (0 the least, 1 the most)"""
    out = {}
    from scipy.stats import binomtest
    for lg in HL:
        champs, base = [], []
        for (l, team), Ss in T.items():
            if l != lg:
                continue
            for i, r in enumerate(Ss):
                early = [x for x in Ss[max(0, i - look):max(0, i - 2)] if not np.isnan(x["h_trade_now"])]
                late = [x for x in Ss[max(0, i - 2):i + 1] if not np.isnan(x["h_trade_now"])]
                if len(early) < look - 3 or len(late) < 3:
                    continue
                rec = dict(season=r["season"], early_now=sum(x["h_now"] for x in early), early_fut=sum(x["h_later"] for x in early),
                           late_now=sum(x["h_now"] for x in late), late_fut=sum(x["h_later"] for x in late))
                (champs if r["champion"] else base).append(rec)
        res = dict(n=len(champs))
        for k in ("early_now", "early_fut", "late_now", "late_fut"):
            pct = [float(np.mean([b[k] < c[k] for b in base if b["season"] == c["season"]])) for c in champs]
            above = sum(p > 0.5 for p in pct)
            res[k] = dict(percentile=round(float(np.mean(pct)), 3), above_half=above, p=round(float(binomtest(above, len(pct)).pvalue), 4))
        out[lg] = res
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return clean(o.item())
    return o


def main():
    T = rows()
    res = {"definitions": dict(sell=f"trade_net <= -{SELL} stars and future_net >= {SELL}", buy=f"trade_net >= {BUY} and future_net <= -{BUY}",
                               sold=f"from rank {MID}: the league's bottom {Q:.0%} of stars traded, net, over the next {AFTER} seasons, with future assets back",
                               bought=f"the top {Q:.0%} of stars traded and signed, net", neither="the middle 30-70 % of both")}
    print("profiles", flush=True)
    res["profiles"] = profiles(T)
    print("who", flush=True)
    res["who"], M = who(T)
    print("bad years", flush=True)
    res["bad_years"] = bad_years(T, M)
    print("teardown", flush=True)
    res["teardown"], ev = teardown(T)
    res["champions"] = champions(T)
    print("by horizon", flush=True)
    h_rows(T)
    res["horizon"] = dict(definitions=dict(
        present_for_future=f"a season in the league's bottom {Q:.0%} of present value moved (stars traded and signed, net, "
                           "projected over the season and the next) that came out ahead in future value (the four after)",
        future_for_present=f"the top {Q:.0%} of present value moved, paid for with future value",
        dumped=f"a window that gave up as much present value with no future back and shed {DUMP:.0%} of an average payroll"))
    res["horizon"]["teardown"], hev = h_teardown(T)
    res["horizon"]["who"] = h_who(T, M)
    res["horizon"]["bad_years"] = h_bad_years(T, M)
    res["horizon"]["champions"] = h_champions(T)
    old = {(e["league"], e["team"], e["season"]): e["path"] for e in ev}       # the first classification, by stars last season
    res["horizon"]["reclassified"] = {lg: {p: dict(Counter(e["hpath"] for e in hev if e["league"] == lg and old.get((lg, e["team"], e["season"])) == p))
                                           for p in ("sold", "bought")} for lg in HL}
    with open(os.path.join(OUT, "horizon_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hev[0]))
        w.writeheader()
        w.writerows(hev)
    json.dump(clean(res), open(os.path.join(OUT, "levers.json"), "w"), indent=1)
    with open(os.path.join(OUT, "lever_events.csv"), "w", newline="") as f:
        cols = list(dict.fromkeys(k for e in ev for k in e if k != "curve"))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: e.get(k, "") for k in cols} for e in ev])
    print("wrote levers.json, lever_events.csv")


if __name__ == "__main__":
    main()
