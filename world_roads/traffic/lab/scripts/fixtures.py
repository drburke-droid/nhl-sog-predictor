"""Reference values from model.py for the lab's parity test:  python scripts/fixtures.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import model as M

T0, CAP = [12.0, 17.0, 22.0], [2000.0, 1500.0, 1100.0]
out = dict(ue=[], so=[], learning=[], braess=[], presets=M.presets(T0, CAP), joins=M.joins_at(T0, CAP))
for D in (0, 800, 2600, 4200, 9000):
    out["ue"].append(dict(D=D, flows=M.parallel_ue(T0, CAP, D)["flows"], times=M.parallel_ue(T0, CAP, D)["times"]))
    out["so"].append(dict(D=D, flows=M.parallel_so(T0, CAP, D)["flows"]))
for kw in (dict(replan=0.2), dict(replan=0.5, sigma=1.0, noise=0.15, seed=11, delay=2), dict(replan=0.3, theta=1.5)):
    r = M.learning(T0, CAP, 4200, days=40, **kw)
    out["learning"].append(dict(kw=kw, flows=r["flows"][-1], gain=r["max_gain"][-1], day10=r["flows"][10]))
for conn in (False, True):
    for D in (2000, 4000, 6000):
        r = M.network_ue(M.braess_network(conn, D))
        out["braess"].append(dict(conn=conn, D=D, paths=r["paths"], flows=r["path_flows"], avg=r["avg_min"]))
json.dump(out, open(os.path.join(HERE, "..", "src", "model", "fixtures.json"), "w"), indent=0)
print("ok")
