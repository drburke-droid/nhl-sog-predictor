"""Cross-category tail calibration (a sensitivity model) and resampling uncertainty.

calibrate(recs, q, family): z is not equally rare in every category, so each leader's z is converted to
  p = P(a season leader of this category exceeds z), leaving the leader being calibrated out of the fit.
  Above the category's q-quantile u of leader z (default q = 0.75):
    family="exponential" (PRIMARY): p = (k/n) exp(-(z - u)/s), s the mean excess, partially pooled across
                         categories on the log scale (DerSimonian-Laird; se of log s = 1/sqrt(k));
    family="gpd"        (SENSITIVITY): p = (k/n) (1 + xi (z - u)/s)^(-1/xi), xi the bounded MLE shape
                         (-0.5..0.5) partially pooled across categories, s refitted with xi fixed. With xi < 0
                         the GPD has an upper endpoint u - s/xi; a leader beyond it gets p = 0 and is flagged
                         'beyond_support' (evidence against the GPD fit, not a probability of zero).
  Below u, p is the empirical share of the category's leaders at or above z.
  The exponential tail is the Gumbel-domain assumption the original z already makes; the pooled GPD shape
  is negative (lighter than exponential), so where the GPD is right the exponential p is conservative.
  Both are model extrapolations, NOT observed frequencies.

p_interval(...): 90% interval for p from resampling the category's other leaders (refits u, s).
tail_coverage(...): leave-one-out check of the tail model: share of all leaders with p_LOO <= t vs t.
empirical_null(...): model-light reference. For every candidate, each non-target comparison season s
  (|s - y| >= 2) is withheld with its neighbours and predicted from the rest of the same window; the
  standardised errors are what z looks like for an ordinary season's best. Counts of null values at or above
  a leader's z (excluding nulls held by that athlete) bound p only down to about 1 / (distinct seasons).

bootstrap(...): moving-block bootstrap of comparison seasons (blocks of 3 seasons, circular, per category).
  Each replicate reweights the comparison seasons, refits, rescores and reranks. The output is a FREQUENCY
  CONDITIONAL ON THE MODEL: the share of refits in which an athlete's best season is in the top ten, not a
  probability that the season is 'truly' top ten. With B replicates its Monte Carlo SE is sqrt(f(1-f)/B).
  model_averaged_bootstrap(...) additionally draws the specification (window, trend, environment) at random
  from a pre-declared set in every replicate.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

import gsmodel as G


# ------------------------------------------------------------------ GPD with partially pooled shape
def gpd_nll(params, y, xi=None):
    if xi is None:
        ls, xi = params
    else:
        ls = params[0]
    s = math.exp(ls)
    t = 1 + xi * y / s
    if np.any(t <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return len(y) * ls + y.sum() / s
    return len(y) * ls + (1 + 1 / xi) * np.log(t).sum()


def gpd_fit(y, xi=None):
    y = np.asarray(y, float)
    s0 = max(float(y.mean()), 1e-3)
    if xi is None:
        best = None
        for x0 in (-0.2, 0.0, 0.2, 0.5):
            r = minimize(gpd_nll, [math.log(s0), x0], args=(y,), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        ls, xh = best.x
        # standard error of xi from a numerical Hessian
        h = 1e-4
        f = lambda a, b: gpd_nll([a, b], y)
        H = np.zeros((2, 2))
        p = [ls, xh]
        for i in range(2):
            for j in range(2):
                pp = list(p); pm = list(p); mp = list(p); mm = list(p)
                pp[i] += h; pp[j] += h; pm[i] += h; pm[j] -= h; mp[i] -= h; mp[j] += h; mm[i] -= h; mm[j] -= h
                H[i, j] = (f(*pp) - f(*pm) - f(*mp) + f(*mm)) / (4 * h * h)
        try:
            se = math.sqrt(max(np.linalg.inv(H)[1, 1], 1e-6))
        except np.linalg.LinAlgError:
            se = 0.5
        return math.exp(ls), xh, se
    r = minimize(gpd_nll, [math.log(s0)], args=(y, xi), method="Nelder-Mead")
    return math.exp(r.x[0]), xi, None


def gpd_sf(y, s, xi):
    if y <= 0:
        return 1.0
    if abs(xi) < 1e-8:
        return math.exp(-y / s)
    t = 1 + xi * y / s
    return 0.0 if t <= 0 else t ** (-1 / xi)


def gpd_shape(y):
    """bounded MLE of the GPD shape (xi in [-0.5, 1]) with the asymptotic standard error (1 + xi)/sqrt(k);
    a diagnostic of whether a category's leader excesses are lighter (xi < 0) or heavier (xi > 0) than exponential"""
    y = np.asarray(y, float)
    best = None
    for xi in np.linspace(-0.5, 1.0, 151):
        r = minimize(gpd_nll, [math.log(max(float(y.mean()), 1e-3))], args=(y, xi), method="Nelder-Mead")
        if best is None or r.fun < best[0]:
            best = (r.fun, xi)
    xi = float(best[1])
    return xi, (1 + xi) / math.sqrt(len(y))


def dl_pool(est, se):
    """DerSimonian-Laird random-effects pooling -> (mean, tau2, shrunken estimates)"""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    mu_fe = float((w * est).sum() / w.sum())
    Q = float((w * (est - mu_fe) ** 2).sum())
    tau2 = max(0.0, (Q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = float((ws * est).sum() / ws.sum())
    shr = (est / se ** 2 + mu / max(tau2, 1e-12)) / (1 / se ** 2 + 1 / max(tau2, 1e-12))
    return mu, tau2, shr


def _fit_categories(by, q, min_exc):
    fits = {}
    for c, rs in by.items():
        z = np.array([r["z"] for r in rs])
        u = float(np.quantile(z, q))
        exc = z[z > u] - u
        if len(exc) >= min_exc:
            xi, xse = gpd_shape(exc)
            fits[c] = {"u": u, "n": len(z), "k": len(exc), "scale": float(exc.mean()), "xi_hat": xi, "se": xse}
    cs = list(fits)
    mu_s, tau2_s, shr_s = dl_pool([math.log(fits[c]["scale"]) for c in cs], [1 / math.sqrt(fits[c]["k"]) for c in cs])
    mu_x, tau2_x, shr_x = dl_pool([fits[c]["xi_hat"] for c in cs], [fits[c]["se"] for c in cs])
    for c, ls, xs in zip(cs, shr_s, shr_x):
        fits[c]["scale_pooled"] = math.exp(float(ls))
        fits[c]["xi_pooled"] = float(np.clip(xs, -0.5, 0.5))
    return fits, {"xi_pooled_mean": mu_x, "tau2": tau2_x, "log_scale_mean": mu_s, "log_scale_tau2": tau2_s}


def _tail_p(z_target, others, f, pool, family):
    """p for one leader from its category's other leaders' z (target left out); -> (p, beyond_support)"""
    u = f["u"]
    if z_target <= u:
        return float(np.sum(np.append(others, z_target) >= z_target)) / (len(others) + 1), False
    exc = others[others > u] - u
    k = len(exc) + 1
    if family == "exponential":
        loo = float(exc.mean()) if len(exc) else f["scale"]
        w_c, w_0 = max(len(exc), 1), 1 / max(pool["log_scale_tau2"], 1e-12)
        ls = (math.log(max(loo, 1e-6)) * w_c + pool["log_scale_mean"] * w_0) / (w_c + w_0)
        return (k / (len(others) + 1)) * math.exp(-(z_target - u) / math.exp(ls)), False
    xi = f["xi_pooled"]
    s, _, _ = gpd_fit(exc, xi=xi) if len(exc) >= 3 else (f["scale"], xi, None)
    y = z_target - u
    if xi < 0 and y >= -s / xi:
        return 0.0, True
    return (k / (len(others) + 1)) * gpd_sf(y, s, xi), False


def calibrate(recs, q=0.75, min_exc=6, family="exponential"):
    by = defaultdict(list)
    for r in recs:
        by[r["cat"]].append(r)
    fits, pool = _fit_categories(by, q, min_exc)
    out = []
    for c, rs in by.items():
        z = np.array([r["z"] for r in rs])
        f = fits.get(c)
        for i, r in enumerate(rs):
            others = np.delete(z, i)
            if f is None:
                p, beyond = float(np.sum(z >= r["z"])) / len(z), False
            else:
                p, beyond = _tail_p(r["z"], others, f, pool, family)
            r2 = dict(r)
            r2["p_cal"] = max(float(p), 1e-300)
            r2["beyond_support"] = beyond
            r2["cal"] = -math.log10(r2["p_cal"])
            out.append(r2)
    summary = {**pool, "fits": fits, "family": family, "q": q, "n_scored": len(recs)}
    return out, summary


def p_interval(recs, cat, z_target, q=0.75, B=500, seed=3, family="exponential"):
    """90% interval for one leader's p: resample its category's other leaders (with replacement), refit"""
    by = defaultdict(list)
    for r in recs:
        by[r["cat"]].append(r)
    zc = np.array([r["z"] for r in by[cat]])
    others = np.delete(zc, int(np.argmin(np.abs(zc - z_target))))
    _, pool = _fit_categories(by, q, 6)
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(B):
        o = rng.choice(others, size=len(others), replace=True)
        u = float(np.quantile(np.append(o, z_target), q))
        exc = o[o > u] - u
        if len(exc) < 3:
            continue
        f = {"u": u, "scale": float(exc.mean()), "xi_pooled": 0.0}
        if family == "gpd":
            f["xi_pooled"] = float(np.clip(dl_pool([gpd_shape(exc)[0]], [0.3])[0], -0.5, 0.5))
        p, _ = _tail_p(z_target, o, f, pool, family)
        ps.append(p)
    return float(np.percentile(ps, 5)), float(np.percentile(ps, 95))


def tail_coverage(recs, q=0.75, family="exponential", ts=(0.05, 0.02, 0.01, 0.005, 0.002, 0.001)):
    """leave-one-out check: if the tail model is calibrated, a share t of all leaders should get p_LOO <= t"""
    cal, _ = calibrate(recs, q=q, family=family)
    p = np.array([r["p_cal"] for r in cal])
    return [{"family": family, "q": q, "t": t, "expected_count": round(t * len(p), 1),
             "observed_count": int((p <= t).sum()), "observed_share": round(float((p <= t).mean()), 5)} for t in ts]


def empirical_null(data, spec, ctx=None):
    """standardised errors of withheld non-target comparison seasons (see module docstring) -> list of
    (category, season, holder_id, e)"""
    recs = G.score(data, spec, ctx=ctx, details=True)
    out = []
    for r in recs:
        m = data.meta[r["cat"]]
        shift = m["shift"] if spec.shift else set()
        for (s, v), pid in zip(r["pts"], r["pids"]):
            if abs(s - r["year"]) < 2 or pid == "cutoff":
                continue
            keep = [(t, w) for t, w in r["pts"] if abs(t - s) > 1]
            if len(keep) < 8:
                continue
            fb = G.fit_baseline(keep, s, shift, spec.trend)
            if fb is None or not fb[2] > 0:
                continue
            out.append((r["cat"], s, pid, (v - fb[0]) / fb[2]))
    return out


# ------------------------------------------------------------------ block bootstrap
def block_weights(seasons, L, rng):
    """multiplicity of each season under a circular moving-block bootstrap of the season sequence"""
    n = len(seasons)
    cnt = np.zeros(n)
    k = 0
    while k < n:
        st = rng.integers(0, n)
        for j in range(L):
            if k >= n:
                break
            cnt[(st + j) % n] += 1
            k += 1
    return dict(zip(seasons, cnt))


def refit_weighted(pts, y, x, wts, shift, trend):
    yrs = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    m = np.array([wts.get(int(t), 0.0) for t in yrs])
    keep = m > 0
    if keep.sum() < 6:
        return None
    yrs, v, m = yrs[keep], v[keep], m[keep]
    if trend == "quad_safe":
        trend = "quad" if min((yrs < y).sum(), (yrs > y).sum()) >= 5 else "ols"
    X, use_shift = G._design(yrs, y, shift, "ols" if trend == "huber" else trend)
    k = X.shape[1]
    if m.sum() - k < 2:
        return None
    sw = np.sqrt(m)
    coef = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)[0]
    r = v - X @ coef
    sc = math.sqrt(float((m * r * r).sum()) / (m.sum() - k))
    base = coef[0] + (coef[-1] if use_shift and y in shift else 0.0)
    return (x - base) / sc if sc > 0 else None


def bootstrap(data, spec, ctx=None, B=300, L=3, seed=7, top=10):
    recs = G.score(data, spec, ctx=ctx, details=True)
    rng = np.random.default_rng(seed)
    cats = sorted({r["cat"] for r in recs})
    hits, ranks = defaultdict(int), defaultdict(list)
    zs = defaultdict(list)
    for b in range(B):
        W = {c: block_weights(sorted(data.seasons[c]), L, rng) for c in cats}
        rb = []
        for r in recs:
            m = data.meta[r["cat"]]
            z = refit_weighted(r["pts"], r["year"], r["x"], W[r["cat"]], m["shift"] if spec.shift else set(), spec.trend)
            if z is not None:
                rb.append({**r, "z": z})
        ranked = G.by_athlete(G.athlete_seasons(rb))
        for i, r in enumerate(ranked[:200]):
            k = (r["sport"], r["id"])
            ranks[k].append(i + 1)
            zs[k].append(r["z"])
            if i < top:
                hits[k] += 1
    return hits, ranks, zs, B


def model_averaged_bootstrap(data, specs, B=600, L=3, seed=17, top=10):
    """specs: list of (spec, ctx). Each replicate draws one specification uniformly and one block resample."""
    rng = np.random.default_rng(seed)
    pre = [(spec, G.score(data, spec, ctx=ctx, details=True)) for spec, ctx in specs]
    cats = sorted({r["cat"] for _, recs in pre for r in recs})
    hits, ranks = defaultdict(int), defaultdict(list)
    for b in range(B):
        spec, recs = pre[int(rng.integers(0, len(pre)))]
        W = {c: block_weights(sorted(data.seasons[c]), L, rng) for c in cats}
        rb = []
        for r in recs:
            m = data.meta[r["cat"]]
            z = refit_weighted(r["pts"], r["year"], r["x"], W[r["cat"]], m["shift"] if spec.shift else set(), spec.trend)
            if z is not None:
                rb.append({**r, "z": z})
        ranked = G.by_athlete(G.athlete_seasons(rb))
        for i, r in enumerate(ranked[:200]):
            k = (r["sport"], r["id"])
            ranks[k].append(i + 1)
            if i < top:
                hits[k] += 1
    return hits, ranks, B
