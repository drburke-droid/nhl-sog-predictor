"""Every number in the video, computed from model.py and the city's routes:  python scenario.py
-> ../media/traffic/video_numbers_<city>.json   (TRAFFIC_CITY=nyc|calgary, default nyc)

The free-flow times are measured on real roads (city_routes.py). Everything about traffic is a toy model:
the capacities are made up, the congestion curve is BPR, and the three routes are treated as independent.
"""
import os, json
import numpy as np
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "traffic")
CAP = [2000.0, 1500.0, 1100.0]          # veh/h, illustrative (the lab's default capacities)
from cities import C, CITY                # noqa: E402
D_RUSH = C["d_rush"]                    # veh/h making this kind of trip at the peak, illustrative
D_NIGHT = 300.0
START = [0.6, 0.2, 0.2]                 # first day: most drivers try the route that is fastest when empty
RUN_CHASE = dict(replan=0.5)            # half of all drivers jump to yesterday's quickest route
RUN_CALM = C["calm"]                    # a few reconsider each day; nobody switches to save under a minute
NAMES, SIDES = C["names"], C["sides"]


def load_routes():
    return json.load(open(os.path.join(MEDIA, f"routes_{CITY}.json")))


def myth_network(R, cfg, D, tunnel=True):
    """Braess on the map. Two of the three routes, each split into a bottleneck (a named road, BPR with a made-up
    capacity) and stretches that traffic does not slow; the mythical tunnel joins the end of one bottleneck to the
    start of the other. Nodes: S start, P tunnel entrance, Q tunnel exit, T destination."""
    X, Y = (R["routes"][R["main"][cfg[k]]]["detail"] for k in ("from_route", "to_route"))
    bx = next(r for r in X["roads"] if r[0] == cfg["from_road"])
    by = next(r for r in Y["roads"] if r[0] == cfg["to_road"])
    TX, TY = X["tmin"][-1], Y["tmin"][-1]
    L = [{"from": "S", "to": "P", "plus": bx[1], "t0": bx[2] - bx[1], "cap": cfg["cap_from"]},
         {"from": "P", "to": "T", "t0": TX - bx[2], "cap": None},
         {"from": "S", "to": "Q", "t0": by[1], "cap": None},
         {"from": "Q", "to": "T", "t0": by[2] - by[1], "cap": cfg["cap_to"], "plus": TY - by[2]}]
    if tunnel:
        L.append({"from": "P", "to": "Q", "t0": cfg["minutes"], "cap": None})
    return dict(links=L, origin="S", dest="T", demand=D), bx, by


def myth_times(R, cfg, fX, fN, fY):
    """trip times on the old route X, the tunnel route and the old route Y for given flows"""
    net, _, _ = myth_network(R, cfg, 1.0)
    L = net["links"]
    t = lambda l, x: l.get("plus", 0.0) + (l["t0"] if l["cap"] is None else M.bpr(l["t0"], l["cap"], x))
    sp, pt, sq, qt, pq = (t(L[0], fX + fN), t(L[1], fX), t(L[2], fY), t(L[3], fY + fN), t(L[4], fN))
    return sp + pt, sp + pq + qt, sq + qt


def myth(R):
    cfg = C.get("myth")
    if not cfg:
        return None
    D = cfg["demand"]
    net, bx, by = myth_network(R, cfg, D)
    closed = M.network_ue(myth_network(R, cfg, D, tunnel=False)[0])
    opened = M.network_ue(net)
    fo = dict(zip(opened["paths"], opened["path_flows"]))
    fc = dict(zip(closed["paths"], closed["path_flows"]))
    start = (fc["S-P-T"], 0.0, fc["S-Q-T"])
    end = (fo["S-P-T"], fo["S-P-Q-T"], fo["S-Q-T"])
    steps = []
    for u in np.linspace(0, 1, 41):                       # drivers moving from the old routes to the tunnel route
        f = tuple(a + (b - a) * u for a, b in zip(start, end))
        steps.append(dict(flows=f, times=myth_times(R, cfg, *f)))
    curve = []
    for d in np.linspace(100, 6000, 60):
        curve.append([float(d), M.network_ue(myth_network(R, cfg, d, False)[0])["avg_min"], M.network_ue(myth_network(R, cfg, d)[0])["avg_min"]])
    night = myth_times(R, cfg, 0, 0, 0)
    return dict(cfg=cfg, bottleneck_from=bx, bottleneck_to=by, closed=dict(flows=start, avg=closed["avg_min"]),
                opened=dict(flows=end, avg=opened["avg_min"], times=myth_times(R, cfg, *end)),
                lone=myth_times(R, cfg, start[0], 1e-6, start[2])[1], night=night, steps=steps, curve=curve)


def why_start(t0, fast=45.0):
    """an unbalanced moment for the 'why equal?' demonstration: route A shows `fast` minutes and the other two a
    common slower time, with the same rush-hour drivers as the equilibrium"""
    fA = M._flow_at(fast, t0[0], CAP[0], M.ALPHA, M.BETA)
    lo, hi = fast, 300.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        rest = sum(M._flow_at(mid, t0[i], CAP[i], M.ALPHA, M.BETA) for i in (1, 2))
        lo, hi = (mid, hi) if fA + rest < D_RUSH else (lo, mid)
    f = [fA] + [M._flow_at(lo, t0[i], CAP[i], M.ALPHA, M.BETA) for i in (1, 2)]
    return dict(flows=f, times=[M.bpr(t0[i], CAP[i], f[i]) for i in range(3)])


def numbers():
    R = load_routes()
    main = [R["routes"][i] for i in R["main"]]
    t0 = [r["minutes"] for r in main]
    ue = M.parallel_ue(t0, CAP, D_RUSH)
    so = M.parallel_so(t0, CAP, D_RUSH)
    joins = M.joins_at(t0, CAP)
    ds = np.linspace(50, 8000, 160)
    poa = [M.price_of_anarchy(M.parallel_ue(t0, CAP, d), M.parallel_so(t0, CAP, d)) for d in ds]
    k = int(np.argmax(poa))
    chase = M.learning(t0, CAP, D_RUSH, days=40, start=START, **RUN_CHASE)
    calm = M.learning(t0, CAP, D_RUSH, days=40, start=START, **RUN_CALM)
    settle = next(d for d in range(41) if all(max(T) - min(T) < 1.0 for T in calm["times"][d:]))   # first day from which all stay within a minute
    br_c = M.network_ue(M.braess_network(False))
    br_o = M.network_ue(M.braess_network(True))
    br_other = {str(D): [M.network_ue(M.braess_network(False, D))["avg_min"], M.network_ue(M.braess_network(True, D))["avg_min"]]
                for D in (2000, 3000, 9000, 10000)}
    alts = [r["minutes"] for r in R["routes"]]
    via = np.array([v["minutes"] for v in R["via_sample"]])
    out = dict(
        city=CITY, city_name=C["name"], trip=C["trip"], names=NAMES, sides=SIDES,
        crossings=[r.get("crossing") for r in main], t0=t0, km=[r["km"] for r in main], via=[r["via"] for r in main], cap=CAP,
        main_overlap=R["main_overlap"], origin=R["origin"]["name"], dest=R["dest"]["name"],
        d_rush=D_RUSH, d_night=D_NIGHT, joins=joins,
        ue=dict(level=ue["level"], flows=ue["flows"], times=ue["times"], avg=ue["avg_min"], total=ue["total_veh_min"]),
        so=dict(flows=so["flows"], times=so["times"], avg=so["avg_min"], total=so["total_veh_min"]),
        poa=M.price_of_anarchy(ue, so), poa_max=dict(value=poa[k], demand=float(ds[k])),
        learn=dict(start=START, chase=dict(params=RUN_CHASE, times=chase["times"], flows=chase["flows"]),
                   calm=dict(params=RUN_CALM, times=calm["times"], flows=calm["flows"], settle_day=settle)),
        eta_final=[int(round(t)) for t in calm["times"][-1]], eta_final_exact=calm["times"][-1],
        braess=dict(closed=dict(paths=br_c["paths"], flows=br_c["path_flows"], times=br_c["path_times"], avg=br_c["avg_min"]),
                    opened=dict(paths=br_o["paths"], flows=br_o["path_flows"], times=br_o["path_times"], avg=br_o["avg_min"]),
                    other_demands=br_other),
        myth=myth(R), why_start=why_start(t0),
        app=dict(top3=sorted(alts)[:3], n_alternatives=len(alts), via_n=len(via), via_min=float(via.min()), via_max=float(via.max()),
                 via_median=float(np.median(via))),
    )
    return out


if __name__ == "__main__":
    out = numbers()
    json.dump(out, open(os.path.join(MEDIA, f"video_numbers_{CITY}.json"), "w"), indent=1)
    p = {k: v for k, v in out.items() if k not in ("learn",)}
    print(json.dumps(p, indent=1)[:3000])
    print("calm settles (all within 1 min) from day", out["learn"]["calm"]["settle_day"], "final", [round(t, 2) for t in out["learn"]["calm"]["times"][-1]])
