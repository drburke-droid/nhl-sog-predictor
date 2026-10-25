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
