"""Cross-category tail calibration and resampling uncertainty.

calibrate(recs): z is not equally rare in every category (hitting-streak leaders are often 3+ sd above
  trend; points-per-game leaders rarely are). For each category, a generalized Pareto distribution (GPD)
  is fitted to the season leaders' z above the category's 75th percentile. The shape xi is partially pooled
  across categories (precision-weighted random-effects, DerSimonian-Laird), the scale is then refitted with
  the pooled shape, and each leader gets
      p = P(a season leader of this category exceeds z)   (leave-one-out for the leader being calibrated)
  reported as -log10(p). This is a model-based tail probability within the category, not an observed
  frequency; it is meant to put categories on one footing, and its uncertainty is large at the extremes.

bootstrap(...): moving-block bootstrap of comparison seasons (blocks of 3 seasons, circular, per category),
  which keeps neighbouring seasons together. Each replicate reweights every window's comparison seasons by
  how often they were drawn, refits the baseline and scale, rescores every leader, and reranks. Output:
  P(top ten) and 5-95% rank ranges per athlete (each athlete's best season, as on the page).
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


def calibrate(recs, q=0.75, min_exc=6):
    """exponential (Gumbel-domain) tail above each category's 75th percentile of leader z, with the scale
    partially pooled across categories on the log scale (se of log scale = 1/sqrt(k)); the leader being
    calibrated is left out of its own category's scale. GPD shapes are reported as a diagnostic only."""
    by = defaultdict(list)
    for r in recs:
        by[r["cat"]].append(r)
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
        fits[c]["xi_pooled"] = float(xs)
    out = []
    for c, rs in by.items():
        z = np.array([r["z"] for r in rs])
        f = fits.get(c)
        for r in rs:
            if f is None or r["z"] <= f["u"]:
                p = float(np.sum(z >= r["z"])) / len(z)
            else:
                exc = z[z > f["u"]] - f["u"]
                k = len(exc)
                loo = (exc.sum() - (r["z"] - f["u"])) / max(k - 1, 1)          # leave the leader out
                # shrink the left-out scale toward the pooled mean with the same weights as dl_pool
                w_c = (k - 1) / 1.0
                w_0 = 1 / max(tau2_s, 1e-12)
                ls = (math.log(max(loo, 1e-6)) * w_c + mu_s * w_0) / (w_c + w_0)
                p = (f["k"] / f["n"]) * math.exp(-(r["z"] - f["u"]) / math.exp(ls))
            r2 = dict(r)
            r2["p_cal"] = max(float(p), 1e-300)
            r2["cal"] = -math.log10(r2["p_cal"])
            out.append(r2)
    summary = {"xi_pooled_mean": mu_x, "tau2": tau2_x, "log_scale_mean": mu_s, "log_scale_tau2": tau2_s, "fits": fits}
    return out, summary


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
