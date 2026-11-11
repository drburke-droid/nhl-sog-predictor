"""
Head-to-head runs for the kitchen tools (bowl_sim.py), plus the sensitivity
sets: does the ranking change with grain size, with speed, or with how the
bowl was loaded?

    python tournament.py --set base            # the eight contestants
    python tournament.py --set grain speed load
    python tournament.py --collect             # rebuild the CSV from finished runs

Writes experiments/tournament.csv (one row per run) and tournament_curves.npz.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tools import CONTESTANTS, mix_curve

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["spoon", "whisk", "fork", "paddle_large", "twin_beaters"]


def runs_for(sets):
    """(run name, set, tool, extra args, seconds)."""
    out = []
    if "base" in sets:
        out += [(f"base_{t}", "base", t, [], 20.0) for t in CONTESTANTS]
    if "grain" in sets:
        out += [(f"grain2.4_{t}", "grain 2.4 mm", t, ["--grain", "2.4"], 20.0) for t in CONTESTANTS]
        out += [(f"grain1.4_{t}", "grain 1.4 mm", t, ["--grain", "1.4"], 20.0) for t in CONTESTANTS]
    if "speed" in sets:                       # same number of strokes at half and double speed
        out += [(f"speed2_{t}", "double speed", t, ["--speed", "2"], 10.0) for t in CONTESTANTS]
        if "half" in sets:                    # slow (40 s runs); not part of the default set
            out += [(f"speed0.5_{t}", "half speed", t, ["--speed", "0.5"], 40.0) for t in KEY]
    if "load" in sets:
        for ld in ("stripes", "rings"):
            out += [(f"load_{ld}_{t}", f"loaded as {ld}", t, ["--load", ld], 20.0) for t in CONTESTANTS]
        out += [("base_spoon_circle", "base", "spoon_circle", [], 20.0)]
    return out


def run_one(r, work, force):
    name, _, tool, extra, seconds = r
    out = os.path.join(work, name)
    if os.path.exists(os.path.join(out, "meta.json")) and not force:
        return name
    cmd = [sys.executable, os.path.join(HERE, "bowl_sim.py"), "--tool", tool, "--no-cache", "--out", out,
           "--seconds", str(seconds)] + extra
    with open(os.path.join(work, name + ".log"), "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{name} failed, see {work}/{name}.log")
    return name


def collect(work):
    rows, curves = [], {}
    for name in sorted(os.listdir(work)):
        mp = os.path.join(work, name, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        z = np.load(os.path.join(work, name, "meta.npz"))
        mix = mix_curve(z)
        curves[name + "_t"], curves[name + "_mix"], curves[name + "_count"] = z["t"], mix, z["count"]
        row = dict(run=name, tool=m["tool"], label=m["label"], unit=m["unit"], grain_mm=m["grain"],
                   speed=m["speed"], load=m["load"], grains=m["n_grains"], seconds=m["seconds"],
                   strokes=round(float(z["count"][-1]), 1), final_percent=round(100 * float(mix[-1]), 1))
        for lvl in (50, 80, 90, 95):
            i = np.where(mix >= lvl / 100)[0]
            row[f"s_to_{lvl}"] = round(float(z["t"][i[0]]), 1) if len(i) else ""
            row[f"count_to_{lvl}"] = round(float(z["count"][i[0]]), 1) if len(i) else ""
        rows.append(row)
    out_dir = os.path.join(HERE, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tournament.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(out_dir, "tournament_curves.npz"), **curves)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=["base"], choices=("base", "grain", "speed", "half", "load"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--work", default=os.path.join(HERE, "cache", "tournament"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect", action="store_true", help="only rebuild the CSV")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    if not args.collect:
        with ThreadPoolExecutor(args.jobs) as ex:
            for n in ex.map(lambda r: run_one(r, args.work, args.force), runs_for(args.set)):
                print("done", n, flush=True)
    for r in collect(args.work):
        print(f"{r['run']:28s} {r['final_percent']:5.1f}%  90% at {r['s_to_90'] or '-':>5} s / {r['count_to_90'] or '-':>5} {r['unit']}")


if __name__ == "__main__":
    main()
