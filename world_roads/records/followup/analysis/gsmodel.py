"""Core model for the greatest-seasons follow-up audit. numpy only.

Every option defaults to the original published method, so Spec() reproduces the original scores exactly
(tests/test_reproduction.py checks all 18,561 pipeline scores).

Season convention: a season is stored as the calendar year in which it ends (NBA/NHL 1985-86 -> 1986).
"""
import csv, math, os, re, unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# corrected data only: seasons whose statistic was recorded under a different definition, so they are not
# comparable with the rest of the category (logged as C7 in the corrections log)
DEFINITION_BREAKS = {"nba_apg": {1947, 1948}}     # BAA assists credited far more sparingly (league 6.9-7.0 per
                                                  # team-game in 1946-48, 18.6 in 1948-49)

TRANS = str.maketrans({"ł": "l", "Ł": "L", "ß": "ss", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "æ": "ae",
                       "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D"})


def norm(name):
    """the original pipeline's name key (accents, case and punctuation removed)"""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]", "", s.lower()).split())


def norm2(name):
    """transliteration-aware name key (ł->l, ß->ss, ...) used for display-level matching in v2"""
    return norm(name.translate(TRANS))


# ------------------------------------------------------------------ data
@dataclass
class Entry:
    id: str          # identity used for self-exclusion
    name: str        # display name
    perf: float      # direction-adjusted performance (bigger is better)
    raw: str         # the value as published
    games: float     # games per team (totals only) or nan


class Data:
    def __init__(self, version="original"):
        self.version = version
        self.cats = {r["category"]: r for r in csv.DictReader(open(os.path.join(DATA, "original", "categories.csv"),
                                                                   encoding="utf-8"))}
        self.seasons = defaultdict(dict)          # cat -> year -> [Entry] in rank order
        if version == "original":
            by = defaultdict(list)
            for r in csv.DictReader(open(os.path.join(DATA, "original", "season_lists.csv"), encoding="utf-8")):
                by[(r["category"], int(r["season"]))].append(r)
            for (c, y), rs in by.items():
                rs.sort(key=lambda r: int(r["rank"]))
                self.seasons[c][y] = [Entry(norm(r["name_raw"]), r["name_display"], float(r["performance"]),
                                            r["raw_value"], float(r["games_per_team"]) if r["games_per_team"] else math.nan)
                                      for r in rs]
        elif version == "v2":
            by = defaultdict(list)
            for r in csv.DictReader(open(os.path.join(DATA, "v2", "season_lists_v2.csv"), encoding="utf-8")):
                if r["v2_rank"]:
                    by[(r["category"], int(r["season"]))].append(r)
            for (c, y), rs in by.items():
                rs.sort(key=lambda r: int(r["v2_rank"]))
                self.seasons[c][y] = [Entry(r["athlete_id"], r["name_display"], float(r["performance"]), r["raw_value"],
                                            float(r["games_per_team"]) if r["games_per_team"] else math.nan)
                                      for r in rs[:10]]
        else:
            raise ValueError(version)
        self.meta = {}
        for c, m in self.cats.items():
            self.meta[c] = {
                "sport": m["sport"], "title": m["title"],
                "short": {int(t) for t in m["short_seasons_left_out"].split()}
                         | (DEFINITION_BREAKS.get(c, set()) if version == "v2" else set()),
                "shift": {int(t) for t in m["level_shift_seasons"].split()},
                "scaled": m["is_total_scaled_by_schedule"] == "True",
                "cut": float(m["cutoff_performance"]) if m["cutoff_performance"] else None,
                "first": int(m["first_season"]), "start": int(m["performances_count_from"]),
            }


# ------------------------------------------------------------------ specification
@dataclass(frozen=True)
class Spec:
    window: int = 15              # seasons either side
    trend: str = "ols"            # ols | flat | huber | quad | quad_safe (quad, but ols near the data's ends)
    exclusion: str = "peer"       # peer: leave out all the performer's seasons; tested: only the tested one
    ties: str = "first"           # first: score the first-listed leader (original); all: every tied leader
    shift: bool = True            # swimming 2008-09 level shift
    env: str = "none"             # none | ratio | power (league environment, where supplied) | depth
                                  # (individual sports: value / median of that season's ranks 2-10)
    field: float = 0.0            # strength of the proposed offset beta*ln(Nref/N) (0 = none)
    field_tail: str = "gumbel"    # gumbel: beta*ln(Nref/N); normal: sigma*(sqrt(2lnNref)-sqrt(2lnN))
    min_real: int = 8
    label: str = ""

    def but(self, **kw):
        return replace(self, **kw)


# ------------------------------------------------------------------ the measure
def gumbel_T(values, x):
    v = np.asarray(values, float)
    if len(v) < 6 or v.std(ddof=1) <= 0:
        return None
    beta = v.std(ddof=1) * math.sqrt(6) / math.pi
    mu = v.mean() - 0.5772 * beta
    z = (x - mu) / beta
    if z > 700:
        return float("inf")
    return 1 / max(1 - math.exp(-math.exp(-z)), 1e-300)


def _design(yrs, y, shift, trend):
    cols = [np.ones_like(yrs)]
    if trend in ("ols", "huber", "quad"):
        cols.append(yrs - y)
    if trend == "quad":
        cols.append((yrs - y) ** 2)
    d = np.isin(yrs, list(shift)).astype(float) if shift else np.zeros_like(yrs)
    use_shift = bool(shift) and d.sum() > 0 and len(yrs) - d.sum() >= 8
    if use_shift:
        cols.append(d)
    return np.column_stack(cols), use_shift


def fit_baseline(pts, y, shift=(), trend="ols"):
    """-> (baseline at y, residuals, scale, n_params). OLS/flat/quad: scale = sqrt(SSR/(n-k)).
    huber: iteratively reweighted least squares (c = 1.345), scale = 1.4826 * MAD of residuals."""
    yrs = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    if trend == "quad_safe":          # quadratic only with at least 5 comparison seasons on each side
        trend = "quad" if min((yrs < y).sum(), (yrs > y).sum()) >= 5 else "ols"
    X, use_shift = _design(yrs, y, shift, trend)
    k = X.shape[1]
    if len(v) - k < 2:
        return None
    coef = np.linalg.lstsq(X, v, rcond=None)[0]
    if trend == "huber":
        for _ in range(50):
            r = v - X @ coef
            s = 1.4826 * np.median(np.abs(r - np.median(r))) or (np.std(r) or 1.0)
            u = np.abs(r) / (1.345 * s)
            w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
            new = np.linalg.lstsq(X * np.sqrt(w)[:, None], v * np.sqrt(w), rcond=None)[0]
            if np.allclose(new, coef, atol=1e-10, rtol=1e-10):
                coef = new
                break
            coef = new
    resid = v - X @ coef
    base = coef[0] + (coef[-1] if use_shift and y in shift else 0.0)
    if trend == "huber":
        scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    else:
        scale = math.sqrt(float(resid @ resid) / (len(v) - k))
    return base, resid, scale, k


# ------------------------------------------------------------------ transforms (environment, field size)
class Context:
    """per-category environment series env[cat][year] (league rate in the category's own units, per team-game
    for totals) and team counts teams[sport][year]; elasticities gamma[cat] for env='power'; tail scales
    beta[cat][year] for the proposed field offset; Nref[sport]."""

    def __init__(self, env=None, teams=None, gamma=None, beta=None, nref=None, env_cats=None, kind=None):
        self.env = env or {}
        self.teams = teams or {}
        self.gamma = gamma or {}
        self.beta = beta or {}
        self.nref = nref or {}
        self.env_cats = set(env_cats) if env_cats is not None else set(self.env)
        self.kind = kind or {}           # cat -> 'ratio' (value / env) or 'era' (league ERA / ERA)


def depth_level(lst):
    """median of ranks 2-10 of a season list (the field's depth below the leader)"""
    vals = [e.perf for e in lst[1:10]]
    return float(np.median(vals)) if len(vals) >= 5 else None


def transform(v, cat, sport, y, spec, ctx):
    """value in the comparison scale for season y, or None when a required input is missing"""
    if spec.env == "depth":
        if cat in ctx.env_cats:
            d = ctx.env.get(cat, {}).get(y)
            return None if not d else v / d
        return v
    if spec.env != "none" and cat in ctx.env_cats:
        e = ctx.env.get(cat, {}).get(y)
        if e is None or e <= 0:
            return None
        g = ctx.gamma.get(cat, 1.0) if spec.env == "power" else 1.0
        if ctx.kind.get(cat) == "era":           # perf is minus the ERA: league ERA / ERA, an ERA+-style ratio
            v = e ** g / (-v)
        else:
            v = v / e ** g
    if spec.field and sport in ctx.nref and cat in ctx.beta:
        N, b = ctx.teams.get(sport, {}).get(y), ctx.beta[cat].get(y)
        if N is None or b is None:
            return None
        if spec.field_tail == "gumbel":
            v = v + spec.field * b * math.log(ctx.nref[sport] / N)
        else:   # normal-tail growth of the expected maximum, with sigma matched to beta at N
            sig = b * math.sqrt(2 * math.log(max(N, 2)))
            v = v + spec.field * sig * (math.sqrt(2 * math.log(ctx.nref[sport])) - math.sqrt(2 * math.log(max(N, 2))))
    return v


# ------------------------------------------------------------------ scoring
def leaders(lst, ties):
    if not lst:
        return []
    if ties == "first":
        return [lst[0]]
    top = lst[0].perf
    return [e for e in lst if e.perf >= top - 1e-12]


def score(data, spec=Spec(), ctx=None, cats=None, details=False):
    """every season-leading performance -> dict(cat, sport, year, id, name, raw, x, base, scale, z, T, n, real)"""
    ctx = ctx or Context()
    out = []
    for cat in (cats or data.meta):
        m = data.meta[cat]
        S = data.seasons[cat]
        shift = m["shift"] if spec.shift else set()
        for y in sorted(S):
            if y in m["short"] or y < m["start"]:
                continue
            for lead in leaders(S[y], spec.ties):
                gy = lead.games if m["scaled"] else None
                x = transform(lead.perf, cat, m["sport"], y, spec, ctx)
                if x is None:
                    continue
                pts, real, pids = [], 0, []
                for s in range(y - spec.window, y + spec.window + 1):
                    if s in m["short"] or s not in S:
                        continue
                    if spec.exclusion == "peer" or s == y:
                        o = next((e for e in S[s] if e.id != lead.id), None)
                    else:
                        o = S[s][0]
                    if o is not None:
                        p = transform(o.perf, cat, m["sport"], s, spec, ctx)
                        if p is None:
                            continue
                        if m["scaled"] and not (spec.env == "depth" and cat in ctx.env_cats):
                            p = p * gy / o.games
                        pts.append((s, p))
                        pids.append(o.id)
                        real += 1
                    elif m["cut"] is not None and s >= m["first"]:
                        c = transform(m["cut"], cat, m["sport"], s, spec, ctx)
                        if c is not None:
                            pts.append((s, c))
                            pids.append("cutoff")
                if real < spec.min_real:
                    continue
                fb = fit_baseline(pts, y, shift, spec.trend)
                if fb is None:
                    continue
                base, resid, scale, k = fb
                if not scale > 0:
                    continue
                z = (x - base) / scale
                before = sum(1 for s, _ in pts if s < y)
                after = sum(1 for s, _ in pts if s > y)
                rec = {"cat": cat, "sport": m["sport"], "title": m["title"], "year": y, "id": lead.id,
                       "name": lead.name, "raw": lead.raw, "x": x, "base": base, "scale": scale, "z": z,
                       "T": gumbel_T(resid, x - base), "n": len(pts), "real": real,
                       "before": before, "after": after, "edge": min(before, after) < 5}
                if details:
                    rec["pts"] = pts
                    rec["pids"] = pids             # who holds each comparison value ("cutoff" = list cut-off)
                    rec["resid"] = resid
                out.append(rec)
    return out


def athlete_seasons(recs):
    """one entry per athlete-season (the category where it scores highest), ranked by z"""
    best = {}
    for r in recs:
        k = (r["sport"], r["id"], r["year"])
        if k not in best or r["z"] > best[k]["z"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: -r["z"])


def by_athlete(ranked):
    """each athlete's best season (the page's top-ten view)"""
    seen, out = set(), []
    for r in ranked:
        k = (r["sport"], r["id"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out
