"""Traffic equilibrium toy models: pure functions, no I/O. The lab (lab/src/model/*.ts) mirrors these.

Units: travel times in minutes, flows, capacities and demand in vehicles per hour (veh/h) over one
steady peak hour. Every value is a toy-model value, not a measurement.

  * bpr / msc                      BPR travel time and its marginal social cost
  * parallel_ue / parallel_so      exact user equilibrium and system optimum on independent parallel routes
  * learning                       a day-to-day route-choice process (behavioural, NOT an equilibrium solver)
  * network_ue                     path-equilibration user equilibrium on a small network (Braess)
  * app_shown                      which of many candidate routes a route-suggesting app would display
"""
import math

ALPHA, BETA = 0.15, 4.0
EPS_FLOW = 1e-9            # a route carrying less than this (veh/h) counts as unused ...
USED_SHARE = 1e-4          # ... as does one carrying under 0.01 % of all trips


# ------------------------------------------------------------------ congestion curve
def bpr(t0, cap, f, alpha=ALPHA, beta=BETA):
    """BPR travel time (min) of a route with free-flow time t0 (min) and capacity cap (veh/h) at flow f (veh/h)."""
    if f <= 0:
        return float(t0)
    if cap <= 0:
        return math.inf
    return t0 * (1 + alpha * (f / cap) ** beta)


def msc(t0, cap, f, alpha=ALPHA, beta=BETA):
    """Marginal social cost d(f T(f))/df: the driver's own time plus the delay they add to everyone else."""
    if f <= 0:
        return float(t0)
    if cap <= 0:
        return math.inf
    return t0 * (1 + alpha * (1 + beta) * (f / cap) ** beta)


def _flow_at(level, t0, cap, a, beta):
    """Flow at which a route's cost t0 (1 + a x^beta) reaches `level` (0 below free flow)."""
    if cap <= 0 or level <= t0:
        return 0.0
    return cap * ((level / t0 - 1) / a) ** (1 / beta)


def _open(t0, cap):
    return [i for i in range(len(t0)) if cap[i] > 0 and t0[i] > 0 and math.isfinite(t0[i])]


def _solve_level(t0, cap, D, a, beta):
    """Common cost level `lam` with sum_i flow_i(lam) = D, by bisection; returns (lam, flows)."""
    n = len(t0)
    ok = _open(t0, cap)
    if D <= 0 or not ok:
        return (min((t0[i] for i in ok), default=math.nan), [0.0] * n)
    if a <= 0:                                  # flat costs: everyone on the cheapest open route(s)
        best = min(t0[i] for i in ok)
        tied = [i for i in ok if t0[i] <= best * (1 + 1e-12)]
        cs = sum(cap[i] for i in tied)
        f = [0.0] * n
        for i in tied:
            f[i] = D * cap[i] / cs
        return best, f
    lo = min(t0[i] for i in ok)
    hi = lo * 2
    total = lambda lam: sum(_flow_at(lam, t0[i], cap[i], a, beta) for i in ok)
    for _ in range(400):
        if total(hi) >= D:
            break
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < D:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    lam = 0.5 * (lo + hi)
    f = [0.0] * n
    for i in ok:
        f[i] = _flow_at(lam, t0[i], cap[i], a, beta)
    s = sum(f)
    if s > 0:                                   # remove the last ~1e-12 of bisection error so flows sum to D
        f = [x * D / s for x in f]
    return lam, f


def summarize(t0, cap, f, alpha=ALPHA, beta=BETA):
    """Times, totals and the equilibrium diagnostic for any assignment f (veh/h on each route)."""
    n = len(t0)
    ok = set(_open(t0, cap))
    T = [bpr(t0[i], cap[i], f[i], alpha, beta) if i in ok else math.inf for i in range(n)]
    D = sum(f)
    # under 0.01 % of all trips (a fraction of a vehicle in a learning run) counts as unused
    used = [i for i in range(n) if f[i] > max(EPS_FLOW, USED_SHARE * D)]
    total = sum(f[i] * T[i] for i in range(n) if f[i] > 0)
    best = min((T[i] for i in ok), default=math.nan)
    # the most any one driver could save by switching alone (0 at an exact user equilibrium) ...
    gain = max((T[i] for i in used), default=best) - best if used else 0.0
    # ... and what the average driver could save (the relative gap times the average trip)
    avg = total / D if D > 0 else math.nan
    return dict(flows=list(f), times=T, used=[i in used for i in range(n)], open=[i in ok for i in range(n)],
                demand=D, total_veh_min=total, avg_min=avg, max_gain_min=max(0.0, gain),
                avg_gain_min=(max(0.0, avg - best) if D > 0 else 0.0))


def parallel_ue(t0, cap, D, alpha=ALPHA, beta=BETA):
    """Wardrop user equilibrium on independent parallel routes: used routes share one travel time `lam`,
    unused routes are no faster (their free-flow time is at least `lam`)."""
    lam, f = _solve_level(t0, cap, D, alpha, beta)
    out = summarize(t0, cap, f, alpha, beta)
    out["level"] = lam if D > 0 else math.nan
    return out


def parallel_so(t0, cap, D, alpha=ALPHA, beta=BETA):
    """System optimum: equal marginal social costs on used routes. Times are the BPR times, not the MSC."""
    lam, f = _solve_level(t0, cap, D, alpha * (1 + beta), beta)
    out = summarize(t0, cap, f, alpha, beta)
    out["level"] = lam if D > 0 else math.nan
    out["msc"] = [msc(t0[i], cap[i], f[i], alpha, beta) for i in range(len(t0))]
    return out


def price_of_anarchy(ue, so):
    """TotalTime_UE / TotalTime_SO, or nan when undefined (no demand)."""
    if not (so["total_veh_min"] > 0):
        return math.nan
    return ue["total_veh_min"] / so["total_veh_min"]


def joins_at(t0, cap, alpha=ALPHA, beta=BETA, system=False):
    """Demand (veh/h) at which each route starts to carry traffic (0 for the fastest; inf if closed)."""
    a = alpha * (1 + beta) if system else alpha
    ok = _open(t0, cap)
    out = []
    for j in range(len(t0)):
        if j not in ok:
            out.append(math.inf)
            continue
        out.append(sum(_flow_at(t0[j], t0[i], cap[i], a, beta) for i in ok if i != j))
    return out


def presets(t0, cap, alpha=ALPHA, beta=BETA):
    """Demand presets computed from the model, so they never claim more convergence than it gives.
    low: only the fastest route used. moderate: halfway between the 2nd and 3rd route joining.
    near: demand at which the last route's free-flow time is 0.5 min above the shared time (all within
    a minute, yet the last route still carries nobody). heavy: every route used, deep in congestion."""
    ok = _open(t0, cap)
    if not ok:
        return dict(low=0.0, moderate=0.0, near=0.0, heavy=0.0)
    j = sorted(joins_at(t0, cap, alpha, beta)[i] for i in ok)
    total_cap = sum(cap[i] for i in ok)
    second = j[1] if len(j) > 1 else total_cap
    last = j[-1] if len(j) > 1 else total_cap
    third = j[2] if len(j) > 2 else last
    slow = max(ok, key=lambda i: t0[i])
    lam_near = max(t0[slow] - 0.5, min(t0[i] for i in ok) + 0.01)
    near = sum(_flow_at(lam_near, t0[i], cap[i], alpha, beta) for i in ok)
    return dict(low=round(0.6 * second), moderate=round(0.5 * (second + third)), near=round(near),
                heavy=round(max(1.25 * last, 1.1 * total_cap)))


# ------------------------------------------------------------------ day-to-day learning
def mulberry32(seed):
    """Tiny seeded PRNG, identical to the lab's, so a seed gives the same run in both."""
    s = seed & 0xFFFFFFFF

    def rnd():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = s
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def gauss(rnd):
    u = max(rnd(), 1e-12)
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rnd())


def choice_matrix(T, ok, theta, sigma):
    """P[i][j]: share of re-planning drivers now on route i who pick route j.
    A driver stays unless another route looks better by more than `theta` minutes; `sigma` (min) spreads
    perceptions (logit). sigma = 0: everyone who switches picks the best-looking route."""
    n = len(T)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if i not in ok:
            continue
        cost = {j: T[j] - (theta if j == i else 0.0) for j in ok}
        if sigma > 0:
            m = min(cost.values())
            w = {j: math.exp(-(c - m) / sigma) for j, c in cost.items()}
            s = sum(w.values())
            for j in ok:
                P[i][j] = w[j] / s
        else:
            m = min(cost.values())
            if cost[i] <= m + 1e-12:
                P[i][i] = 1.0                    # current route is (one of) the best: stay
            else:
                best = [j for j in ok if cost[j] <= m + 1e-12]
                for j in best:
                    P[i][j] = 1.0 / len(best)
    return P


def learning(t0, cap, D, days=60, start=None, replan=0.2, delay=1, theta=0.0, sigma=0.0, noise=0.0,
             seed=1, alpha=ALPHA, beta=BETA):
    """Day-to-day route choice. Each day a share `replan` of every route's drivers reconsiders, using the
    travel times observed `delay` days earlier (each multiplied by 1 + noise * N(0,1), a seeded day-to-day
    disturbance such as weather or incidents). Mass moves between routes, so vehicles are never created or
    destroyed and no flow goes negative. This is a behavioural sketch with no guarantee of convergence.
    Returns per-day flows, (experienced) times and the max unilateral gain."""
    n = len(t0)
    ok = _open(t0, cap)
    if start is None:                            # most drivers try the fastest empty road
        fastest = min(ok, key=lambda i: t0[i]) if ok else 0
        start = [0.0] * n
        for i in ok:
            start[i] = 0.9 if i == fastest else 0.1 / max(1, len(ok) - 1)
        if len(ok) == 1:
            start[fastest] = 1.0
    s = sum(start[i] for i in ok)
    f = [(start[i] / s * D if (i in ok and s > 0) else 0.0) for i in range(n)]
    rnd = mulberry32(seed)
    hist_f, hist_T, hist_obs, hist_gain, hist_avg = [], [], [], [], []
    for day in range(days + 1):
        T = [bpr(t0[i], cap[i], f[i], alpha, beta) if i in ok else math.inf for i in range(n)]
        obs = [T[i] * max(0.05, 1 + noise * gauss(rnd)) if i in ok else math.inf for i in range(n)]
        hist_f.append(list(f))
        hist_T.append(T)
        hist_obs.append(obs)
        sm = summarize(t0, cap, f, alpha, beta)
        hist_gain.append(sm["max_gain_min"])
        hist_avg.append(sm["avg_gain_min"])
        if day == days:
            break
        seen = hist_obs[max(0, len(hist_obs) - delay)]
        P = choice_matrix(seen, ok, theta, sigma)
        g = [(1 - replan) * f[j] for j in range(n)]
        for i in ok:
            for j in ok:
                g[j] += replan * f[i] * P[i][j]
        s = sum(g)
        f = [max(0.0, x) * (D / s) if s > 0 else 0.0 for x in g]
    return dict(flows=hist_f, times=hist_T, observed=hist_obs, max_gain=hist_gain, avg_gain=hist_avg)


# ------------------------------------------------------------------ networks (Braess)
def braess_network(connector=True, D=4000.0, slope=100.0, const=45.0, conn_cost=0.0):
    """The classic four-node example: S->A and B->T take flow/slope minutes, A->T and S->B `const` minutes,
    the optional one-way connector A->B `conn_cost` minutes. Links: (from, to, a, b) with time a + b * flow."""
    links = [("S", "A", 0.0, 1 / slope), ("A", "T", const, 0.0), ("S", "B", const, 0.0), ("B", "T", 0.0, 1 / slope)]
    if connector:
        links.append(("A", "B", conn_cost, 0.0))
    return dict(links=links, origin="S", dest="T", demand=D)


def simple_paths(links, o, d):
    out, adj = [], {}
    for k, (u, v, *_) in enumerate(links):
        adj.setdefault(u, []).append((v, k))

    def walk(u, seen, path):
        if u == d:
            out.append(list(path))
            return
        for v, k in adj.get(u, []):
            if v not in seen:
                walk(v, seen | {v}, path + [k])
    walk(o, {o}, [])
    return out


def network_ue(net, iters=10000, tol=1e-9):
    """User equilibrium on a small network with link times a + b * flow (b >= 0), over every simple path.
    Pairwise path equilibration: start everyone on the free-flow shortest path, then repeatedly move flow
    from the slowest used path to the fastest one until their times are equal (a bisection per move). Each
    move lowers Beckmann's objective, so this converges for any increasing link costs.
    Returns path flows/times, link flows/times, total vehicle-minutes and the max unilateral gain."""
    links, D = net["links"], float(net["demand"])
    paths = simple_paths(links, net["origin"], net["dest"])
    m, P = len(links), len(paths)
    inc = [[k in p for k in range(m)] for p in paths]

    def load(pf):
        x = [0.0] * m
        for p, q in zip(paths, pf):
            for k in p:
                x[k] += q
        return x

    def ptimes(x):
        t = [links[k][2] + links[k][3] * x[k] for k in range(m)]
        return [sum(t[k] for k in p) for p in paths]
    pf = [0.0] * P
    if D > 0 and P:
        c = ptimes([0.0] * m)
        pf[min(range(P), key=lambda i: (c[i], i))] = D
    for _ in range(iters):
        if D <= 0 or P < 2:
            break
        x = load(pf)
        c = ptimes(x)
        used = [i for i in range(P) if pf[i] > 0]
        w = max(used, key=lambda i: c[i])
        bst = min(range(P), key=lambda i: (c[i], i))
        if c[w] - c[bst] <= tol * max(1.0, c[bst]):
            break
        # moving d from w to bst changes their time difference linearly (links on both paths cancel)
        slope = sum(links[k][3] * (1 if inc[w][k] else 0) + links[k][3] * (1 if inc[bst][k] else 0)
                    - 2 * links[k][3] * (1 if (inc[w][k] and inc[bst][k]) else 0) for k in range(m))
        d = pf[w] if slope <= 0 else min(pf[w], (c[w] - c[bst]) / slope)
        pf[w] -= d
        pf[bst] += d
    x = load(pf)
    t = [links[k][2] + links[k][3] * x[k] for k in range(m)]
    c = ptimes(x)
    total = sum(x[k] * t[k] for k in range(m))
    used = [q > 1e-9 * max(D, 1) for q in pf]
    best = min(c) if c else math.nan
    gain = max((ci for ci, u in zip(c, used) if u), default=best) - best if any(used) else 0.0
    names = ["-".join([net["origin"]] + [links[k][1] for k in p]) for p in paths]
    return dict(paths=names, path_flows=pf, path_times=c, link_flows=x, link_times=t, used=used,
                total_veh_min=total, avg_min=(total / D if D > 0 else math.nan), max_gain_min=max(0.0, gain))


# ------------------------------------------------------------------ what the app shows
def app_shown(times, k=3, max_detour=None):
    """Indices of the routes an app would list: the k fastest, optionally only those within `max_detour`
    (a fraction, e.g. 0.25 = 25 % slower than the fastest). A stylized rule, not any real app's logic."""
    order = sorted(range(len(times)), key=lambda i: (times[i], i))
    best = times[order[0]] if order else math.nan
    shown = [i for i in order[:k] if max_detour is None or times[i] <= best * (1 + max_detour)]
    return shown


def random_routes(n, best=24.0, spread=1.0, seed=1):
    """n candidate route times with no traffic equilibrium at all: the fastest is `best` min and the others
    are uniformly up to (1 + spread) x best. Seeded."""
    rnd = mulberry32(seed)
    t = [best] + [best * (1 + spread * rnd()) for _ in range(max(0, n - 1))]
    return t
