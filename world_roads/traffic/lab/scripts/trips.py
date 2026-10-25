"""The real trips the lab offers (free-flow times only):  python scripts/trips.py -> src/trips/trips.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TRAFFIC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, TRAFFIC)
from cities import CITIES  # noqa: E402

out = {}
for city, c in CITIES.items():
    p = os.path.join(os.path.dirname(TRAFFIC), "media", "traffic", f"routes_{city}.json")
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    main = [R["routes"][i] for i in R["main"]]
    out[city] = dict(name=c["name"], trip=c["trip"], origin=R["origin"]["name"], dest=R["dest"]["name"],
                     main=[dict(name=n, minutes=r["minutes"], km=r["km"], via=r["via"]) for n, r in zip(c["names"], main)],
                     alternatives=[r["minutes"] for r in R["routes"]], via_sample=[v["minutes"] for v in R["via_sample"]],
                     d_rush=c["d_rush"])
json.dump(out, open(os.path.join(HERE, "..", "src", "trips", "trips.json"), "w"))
print({k: [m["minutes"] for m in v["main"]] for k, v in out.items()})
