"""
The knockout film: eight ways of mixing dry ingredients, seven matches.

Each round loads the bowl differently, so every match is new footage and the
winner has to cope with a new problem: pie wedges, then stripes, then rings.
Who goes through is decided by the simulations, not written in here.

    python make_film.py sims        # simulate every match (about 30 min on 4 cores)
    python make_film.py render      # Blender frames for every match, no overlay
    python make_film.py compose     # overlays, cards, bracket -> mixing_knockout.mp4

All three stages skip work that is already done, so they can be re-run.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tools import TOOLS, mix_curve

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "film")
SECONDS = 15.0
ROUNDS = [("Quarter-final", "wedges", "pie wedges"), ("Semi-final", "stripes", "stripes"), ("Final", "rings", "rings")]
QUARTERS = [("spoon", "spoon_centre"), ("paddle_large", "wire_whip"), ("whisk", "twin_beaters"), ("fork", "paddle_small")]


def run_dir(load, tool):
    return os.path.join(CACHE, f"{load}_{tool}")


def result(load, tool):
    """Final % mixed of a finished run, or None."""
    d = run_dir(load, tool)
    if not os.path.exists(os.path.join(d, "meta.json")):
        return None
    return float(mix_curve(np.load(os.path.join(d, "meta.npz")))[-1])


def bracket():
    """Matches known so far: list of dicts (round index, number, load, a, b, winner or None)."""
    matches, pairs = [], QUARTERS
    for r, (rname, load, _) in enumerate(ROUNDS):
        winners = []
        for k, (a, b) in enumerate(pairs):
            ra, rb = result(load, a), result(load, b)
            w = None if ra is None or rb is None else (a if ra >= rb else b)
            name = rname if len(pairs) == 1 else f"{rname} {k + 1}"
            matches.append(dict(round=r, k=k, name=name, load=load, a=a, b=b, ra=ra, rb=rb, winner=w,
                                key=f"{'QSF'[r]}{k + 1}"))
            winners.append(w)
        if None in winners or len(winners) == 1:
            break
        pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
    return matches


def stage_sims(args):
    os.makedirs(CACHE, exist_ok=True)

    def one(job):
        load, tool = job
        d = run_dir(load, tool)
        if os.path.exists(os.path.join(d, "meta.json")):
            return
        cmd = [sys.executable, os.path.join(HERE, "bowl_sim.py"), "--tool", tool, "--load", load,
               "--seconds", str(SECONDS), "--out", d]
        with open(d + ".log", "w") as log:
            if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"simulation failed: {d}.log")

    while True:
        todo = [(m["load"], t) for m in bracket() for t in (m["a"], m["b"]) if result(m["load"], t) is None]
        if not todo:
            break
        with ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(one, todo))
    for m in bracket():
        print(f"{m['name']:16s} {m['load']:8s} {TOOLS[m['a']]['label']} {100*m['ra']:.0f}%  vs  "
              f"{TOOLS[m['b']]['label']} {100*m['rb']:.0f}%  ->  {TOOLS[m['winner']]['label']}", flush=True)


def frames_done(out, n):
    fd = os.path.join(out, "frames")
    if not os.path.isdir(fd):
        return 0
    return sum(os.path.getsize(os.path.join(fd, f)) > 0 for f in os.listdir(fd))


def stage_render(args):
    """Render every match in bracket order, waiting for simulations still running."""
    import time
    while True:
        ms = bracket()
        pending = None
        for m in ms:
            if m["winner"] is None:                       # its simulations are not finished yet
                pending = "wait"
                break
            n = json.load(open(os.path.join(run_dir(m["load"], m["a"]), "meta.json")))["n_frames"]
            if frames_done(os.path.join(args.frames, m["key"]), n) < n:
                pending = m
                break
        if pending is None:
            if len(ms) == 7:
                break
            pending = "wait"
        if pending == "wait":
            time.sleep(20)
            continue
        m = pending
        out = os.path.join(args.frames, m["key"])
        fd = os.path.join(out, "frames")
        if os.path.isdir(fd):                             # placeholders left by an interrupted render
            for f in os.listdir(fd):
                if os.path.getsize(os.path.join(fd, f)) == 0:
                    os.remove(os.path.join(fd, f))
        print(f"rendering {m['key']}: {TOOLS[m['a']]['label']} vs {TOOLS[m['b']]['label']}", flush=True)
        cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, "bowl_scene.py"),
               "--left", run_dir(m["load"], m["a"]), "--right", run_dir(m["load"], m["b"]),
               "--no-hud", "--anim", "--res", args.res, "--samples", str(args.samples), "--out", out]
        with open(os.path.join(args.frames, m["key"] + ".log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("render complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sims", "render", "compose", "bracket"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--frames", default=os.path.join(HERE, "renders", "film"))
    ap.add_argument("--res", default="960x420")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "mixing_knockout.mp4"))
    args = ap.parse_args()
    if args.stage == "sims":
        stage_sims(args)
    elif args.stage == "render":
        os.makedirs(args.frames, exist_ok=True)
        stage_render(args)
    elif args.stage == "bracket":
        for m in bracket():
            print(m)
    else:
        import film_compose
        film_compose.compose(args, bracket())


if __name__ == "__main__":
    main()
