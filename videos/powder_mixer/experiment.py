"""
Mixer design experiment: how does "% mixed" grow with paddle revolutions for
different paddle counts, shapes and ways of driving them?

Runs simulate.py once per design (in parallel), then writes

  experiments/mixing_vs_revolutions.csv   one row per design per pass
  experiments/curves.npz                  full per-frame curves

    python experiment.py                  # all designs, ~36k grains each
    python experiment.py --grain 1.5 --passes 20 --jobs 4

The solver time step matters: with a 4 ms step the grains jitter enough to mix
themselves, roughly doubling the first-pass figure.  The default 1 ms limit
(--max-dt) is where the answer stops changing much.
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

# name -> (label, extra simulate.py arguments)
DESIGNS = {
    "twin_gate":      ("Twin gate paddles (the film)", []),
    "twin_gate_seed": ("Twin gate paddles, different random fill", ["--seed", "8"]),
    "twin_plate":     ("Twin solid plates", ["--style", "plate"]),
    "twin_fork":      ("Twin forks (3 tines a side)", ["--style", "fork"]),
    "single_small":   ("One gate paddle, same size", ["--paddles", "1"]),
    "single_wide":    ("One wide gate paddle, dish to match",
                       ["--paddles", "1", "--set", "paddle_r=43", "--set", "tray_rc=45",
                        "--set", "mix_reach=30"]),
    "twin_alternate": ("Twin gate, reversing every pass", ["--alternate"]),
    "twin_half_speed": ("Twin gate, half speed, same revolutions",
                        ["--rev", "0.5", "--t-mix", "3.2", "--set", "t_plunge=0.7",
                         "--set", "t_lift=0.7", "--set", "t_spin=0.9"]),
    "twin_double_speed": ("Twin gate, double speed, same revolutions",
                          ["--rev", "2.0", "--t-mix", "0.8", "--set", "t_plunge=0.175",
                           "--set", "t_lift=0.175", "--set", "t_spin=0.225"]),
}


def run_one(name, args):
    out = os.path.join(args.work, name)
    if os.path.exists(os.path.join(out, "meta.json")) and not args.force:
        return name
    cmd = [sys.executable, os.path.join(HERE, "simulate.py"), "--no-cache", "--out", out,
           "--grain", str(args.grain), "--passes", str(args.passes), "--rev", "1.0",
           "--max-dt", str(args.max_dt)] + DESIGNS[name][1]
    with open(os.path.join(args.work, name + ".log"), "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{name} failed, see {args.work}/{name}.log")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grain", type=float, default=1.8)
    ap.add_argument("--passes", type=int, default=16)
    ap.add_argument("--max-dt", type=float, default=1.0, help="solver time step limit in ms")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--work", default=os.path.join(HERE, "cache", "experiment_converged"))
    ap.add_argument("--only", nargs="*", help="subset of design names")
    ap.add_argument("--force", action="store_true", help="re-run designs that already have results")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    names = args.only or list(DESIGNS)
    with ThreadPoolExecutor(args.jobs) as ex:
        for n in ex.map(lambda n: run_one(n, args), names):
            print("done", n, flush=True)

    out_dir = os.path.join(HERE, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    curves = {}
    with open(os.path.join(out_dir, "mixing_vs_revolutions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "label", "grains", "pass", "paddle_revolutions", "percent_mixed"])
        for n in names:
            meta = json.load(open(os.path.join(args.work, n, "meta.json")))
            z = np.load(os.path.join(args.work, n, "meta.npz"))
            curves[n + "_revs"], curves[n + "_mix"] = z["revs"], z["mix"]
            for p, (rv, m) in enumerate(zip(meta["revs_after_pass"], meta["mix_after_pass"])):
                w.writerow([n, DESIGNS[n][0], meta["n_grains"], p, f"{rv:.2f}", f"{100 * m:.1f}"])
    np.savez_compressed(os.path.join(out_dir, "curves.npz"), **curves)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
