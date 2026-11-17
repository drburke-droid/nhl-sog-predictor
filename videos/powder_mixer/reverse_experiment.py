"""
Can you un-mix by stirring backwards?

Each run stirs forward for a set time, then retraces the tool's path exactly in
reverse until the tool is back where it started.  If mixing were reversible the
bowl would return to its starting pattern (as a viscous liquid does in G.I.
Taylor's demonstration); the runs measure how much of it actually comes back.

    python reverse_experiment.py                 # all sets, about an hour on 4 cores
    python reverse_experiment.py --set tools     # one set
    python plot_reverse.py

Forward time is given in hand strokes (1.5 strokes a second), so every tool
gets the same number of seconds; the tool's own count is recorded too.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STROKE = 1 / 1.5          # seconds per hand stroke
TOOLS6 = ["spoon", "whisk", "fork", "paddle_large", "twin_beaters", "spoon_centre"]
SETS = ("tools", "load", "grain", "repeat", "small", "solver")


def cases(sets):
    out = []
    if "tools" in sets:       # how many strokes before turning back, for each tool
        out += [("tools", t, "wedges", 1.8, n, 7) for t in TOOLS6 for n in (1, 2, 4, 8)]
        out += [("tools", t, "wedges", 1.8, 16, 7) for t in ("spoon", "whisk")]
    if "load" in sets:        # does the starting arrangement matter?
        out += [("load", t, load, 1.8, n, 7) for load in ("stripes", "rings") for t in ("spoon", "whisk", "paddle_large")
                for n in (2, 8)]
    if "grain" in sets:       # finer and coarser grains
        out += [("grain", t, "wedges", g, n, 7) for g in (1.4, 2.4) for t in ("spoon", "whisk") for n in (2, 8)]
    if "repeat" in sets:      # same case, different random fill: how much is noise?
        out += [("repeat", "spoon", "wedges", 1.8, n, 8) for n in (2, 8)]
    if "small" in sets:       # barely disturbed: a quarter and half a stroke
        out += [("small", t, "wedges", 1.8, n, 7) for t in ("spoon", "whisk", "paddle_large") for n in (0.25, 0.5)]
    if "solver" in sets:      # half the solver time step: is the irreversibility physical or numerical?
        out += [("solver", "spoon", "wedges", 1.8, n, 7, 0.5) for n in (1, 2)]
    return out


def name(c):
    _, tool, load, grain, n, seed = c[:6]
    return f"{tool}_{load}_g{grain:g}_n{n:g}" + ("" if seed == 7 else f"_s{seed}") + (f"_dt{c[6]:g}" if len(c) > 6 else "")


def run(c, root):
    _, tool, load, grain, n, seed = c[:6]
    out = os.path.join(root, name(c))
    if not os.path.exists(os.path.join(out, "meta.json")):
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "bowl_sim.py"), "--tool", tool, "--load", load, "--grain", str(grain),
               "--seed", str(seed), "--reverse-after", str(n * STROKE), "--no-cache", "--out", out]
        if len(c) > 6:
            cmd += ["--max-dt", str(c[6])]
        with open(out + ".log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("done", name(c), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--root", default=os.path.join(HERE, "cache", "reverse"))
    a = ap.parse_args()
    todo = cases(a.set)
    todo.sort(key=lambda c: -c[4] * (1.8 / c[3]) ** 3)             # longest first
    with ThreadPoolExecutor(a.jobs) as ex:
        dirs = list(ex.map(lambda c: run(c, a.root), todo))

    rows, curves = [], {}
    for c, d in zip(todo, dirs):
        m = json.load(open(os.path.join(d, "meta.json")))
        z = np.load(os.path.join(d, "meta.npz"))
        fps, fm = m["fps"], m["frame_mid"]
        f_in, f_out = int(round(0.5 * fps)), int(round((m["seconds"] - 0.5) * fps))     # tool fully down ... about to lift
        mix = z["mix"]
        rows.append(dict(set=c[0], tool=c[1], label=m["label"], load=c[2], grain=c[3], strokes=c[4], seed=c[5],
                         dt_ms=round(m["dt_ms"], 2),
                         tool_count=round(m["count_mid"], 2), unit=m["unit"], n_grains=m["n_grains"],
                         mix_turn=round(100 * mix[fm], 1), mix_peak=round(100 * mix.max(), 1),
                         mix_after_plunge=round(100 * mix[f_in], 1), mix_before_lift=round(100 * mix[f_out], 1),
                         mix_end=round(100 * mix[-1], 1),
                         moved_turn_mm=round(m["moved_mid_median"], 1), moved_end_mm=round(m["moved_end_median"], 1),
                         way_back=round(1 - m["moved_end_median"] / m["moved_mid_median"], 3),
                         home_fraction=round(m["home_fraction"], 3)))
        curves[name(c)] = dict(t=[round(float(v), 4) for v in z["t"]], count=[round(float(v), 3) for v in z["count"]],
                               mix=[round(float(v), 4) for v in mix], frame_mid=fm)
    rows.sort(key=lambda r: (r["set"], r["tool"], r["load"], r["grain"], r["strokes"], r["seed"]))
    os.makedirs(os.path.join(HERE, "experiments"), exist_ok=True)
    path = os.path.join(HERE, "experiments", "reverse.csv")
    old = []
    if os.path.exists(path) and set(a.set) != set(SETS):      # keep sets not re-run
        old = [r for r in csv.DictReader(open(path)) if r["set"] not in a.set]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old + rows)
    cpath = os.path.join(HERE, "experiments", "reverse_curves.json")
    allc = json.load(open(cpath)) if os.path.exists(cpath) else {}
    allc.update(curves)
    json.dump(allc, open(cpath, "w"))
    print("wrote", path, "and", cpath)


if __name__ == "__main__":
    main()
