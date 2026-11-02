"""League context (environment denominators, team counts, tail scales) for the model variants.

Environment denominators are league per-team-game rates of the same statistic (data/context/*_league.csv):
a category's value divided by its league rate asks "how many team-games' worth of league-average output",
i.e. dominance relative to the scoring/opportunity environment. Categories with no defensible league
denominator are left unadjusted and listed in UNADJUSTED.
"""
import csv, math, os
from collections import defaultdict

import numpy as np

import gsmodel as G

CTX = os.path.join(G.DATA, "context")

# category -> (league file, column, kind); kind 'ratio' divides the value, 'era' uses league ERA / ERA
ENV_MAP = {
    "nba_ppg": ("nba", "pts_per_team_g", "ratio"),
    "nba_rpg": ("nba", "trb_per_team_g", "ratio"),
    "nba_apg": ("nba", "ast_per_team_g", "ratio"),
    "nba_3pm": ("nba", "fg3_per_team_g", "ratio"),
    "nhl_goals": ("nhl", "goals_per_team_g", "ratio"),
    "nhl_assists": ("nhl", "assists_per_team_g", "ratio"),
    "nhl_points": ("nhl", "points_per_team_g", "ratio"),
    "mlb_hr": ("mlb", "hr_per_team_g", "ratio"),
    "mlb_h": ("mlb", "h_per_team_g", "ratio"),
    "mlb_rbi": ("mlb", "rbi_per_team_g", "ratio"),
    "mlb_sb": ("mlb", "sb_per_team_g", "ratio"),
    "mlb_so": ("mlb", "so_pitch_per_team_g", "ratio"),
    "mlb_ba": ("mlb", "ba", "ratio"),
    "mlb_era": ("mlb", "era", "era"),
    "nfl_pass_yds": ("nfl", "pass_yds_per_team_g", "ratio"),
    "nfl_pass_td": ("nfl", "pass_td_per_team_g", "ratio"),
    "nfl_rush_yds": ("nfl", "rush_yds_per_team_g", "ratio"),
    "nfl_rec_yds": ("nfl", "rec_yds_per_team_g", "ratio"),
    "nfl_int": ("nfl", "int_per_team_g", "ratio"),
    "nfl_sacks": ("nfl", "sacks_per_team_g", "ratio"),
}
UNADJUSTED = {
    "mlb_w": "wins by a pitcher: no league denominator separates a pitcher's opportunity (starts, rotation size, "
             "complete-game norms) from his ability with the data in hand",
    "mlb_streak": "hitting streak: needs an at-bat/plate-appearance opportunity model (per-game hit probability), "
                  "not a league rate",
    "athletics/swimming": "individual sports: treated with participation-depth and technology variants, "
                          "not a league scoring denominator",
}
NREF = {"NBA": 30, "NHL": 32, "MLB": 30, "NFL": 32}


def read(name):
    p = os.path.join(CTX, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def league_tables():
    t = {}
    for lg in ("nba", "nhl", "mlb", "nfl"):
        t[lg] = {int(r["season"]): r for r in read(f"{lg}_league.csv")}
    return t


def env_series(alt=None):
    """category -> {season: denominator}. alt overrides (cat -> column) e.g. {'nba_ppg': 'pace_team_mean'}"""
    L = league_tables()
    out = {}
    for cat, (lg, col, kind) in ENV_MAP.items():
        col = (alt or {}).get(cat, col)
        s = {}
        for y, r in L[lg].items():
            if col == "shot_opps":
                f, ft = fnum(r.get("fga_per_team_g")), fnum(r.get("fta_per_team_g"))
                v = f + 0.44 * ft if f and ft else None
            else:
                v = fnum(r.get(col))
            if v and v > 0 and y not in G.DEFINITION_BREAKS.get(cat, ()):
                s[y] = v
        out[cat] = s
    return out


def teams():
    L = league_tables()
    return {sp: {y: fnum(r.get("teams")) for y, r in L[lg].items() if fnum(r.get("teams"))}
            for sp, lg in (("NBA", "nba"), ("NHL", "nhl"), ("MLB", "mlb"), ("NFL", "nfl"))}


def rank_tail_beta(data, cats, envs=None, smooth=7):
    """the handoff's experimental tail scale: per season, OLS slope of ranks 2-10 (values, divided by the
    environment where given) on -ln(rank - 0.5); clipped at 0; median over seasons within +/-smooth"""
    raw = {}
    for c in cats:
        for y, lst in data.seasons[c].items():
            e = (envs or {}).get(c, {}).get(y, 1.0) if envs else 1.0
            if len(lst) < 6 or (envs and c in envs and y not in envs[c]):
                continue
            x = np.array([o.perf / e for o in lst[1:]])
            a = np.array([-math.log(i - 0.5) for i in range(2, len(x) + 2)])
            raw[(c, y)] = max(0.0, float(np.polyfit(a, x, 1)[0]))
    beta = defaultdict(dict)
    for (c, y) in raw:
        near = [b for (cc, yy), b in raw.items() if cc == c and abs(yy - y) <= smooth]
        beta[c][y] = float(np.median(near)) if len(near) >= 4 else float(np.median([v for (cc, _), v in raw.items() if cc == c]))
    return raw, dict(beta)


def make(data, env="none", field=0.0, alt=None, gamma=None, beta_env=True, sports=("NBA", "NHL", "MLB", "NFL")):
    """a Context for gsmodel.score. env/field cover the four leagues (where a denominator exists)."""
    E = env_series(alt)
    cats = [c for c in E if data.meta.get(c) and data.meta[c]["sport"] in sports]
    ctx = G.Context(env={c: E[c] for c in cats}, teams=teams(), gamma=gamma or {},
                    nref={s: NREF[s] for s in sports}, env_cats=cats,
                    kind={c: ENV_MAP[c][2] for c in cats})
    if field:
        _, beta = rank_tail_beta(data, cats, envs=({c: E[c] for c in cats} if (env != "none" and beta_env) else None))
        ctx.beta = beta
    return ctx
