"""
The knockout film: eight ways of mixing dry ingredients, seven matches.

Each round loads the bowl differently, so every match is new footage and the
winner has to cope with a new problem: pie wedges, then stripes, then rings.
Who goes through is decided by the simulations, not written in here.

    python make_film.py sims        # simulate every match (about 30 min on 4 cores)
    python make_film.py render      # Blender frames for every match, no overlay
    python make_film.py compose     # overlays, cards, bracket -> mixing_knockout.mp4 (silent)
    python make_film.py finish      # synthesize the score, mix to -16 LUFS -> the_mix_off_PREVIEW_720p.mp4

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
# Which bowl the eventual winner of each match stands at, so that one side does not win every time.
# The bowls are independent simulations, so the side changes nothing but the picture.
WINNER_SIDE = "LRRLRLR"
# the closing "satisfying blending" minute: two bowls a shot, each (pixel art, tool)
SATISFYING = [("X1", ("checkerboard", "spoon"), ("squares", "paddle_small")),
              ("X2", ("face", "whisk"), ("heart", "paddle_large")),
              ("X3", ("rocket", "fork"), ("alien", "twin_beaters")),
              ("X4", ("floppy", "spoon_circle"), ("potion", "wire_whip"))]
ART_SECONDS, ART_GRAIN = 12.0, 1.5
STEEP = ["--bare", "--elev", "66", "--dist", "455", "--target", "0,0,8"]     # camera for those shots


def run_dir(load, tool):
    return os.path.join(CACHE, f"{load}_{tool}")


def art_dir(art, tool):
    return os.path.join(CACHE, f"art_{art}_{tool}")


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
            if w is not None and (w == a) != (WINNER_SIDE[len(matches)] == "L"):
                a, b, ra, rb = b, a, rb, ra
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

    def one_art(job):
        art, tool = job
        d = art_dir(art, tool)
        if os.path.exists(os.path.join(d, "meta.json")):
            return
        cmd = [sys.executable, os.path.join(HERE, "bowl_sim.py"), "--tool", tool, "--load", "art:" + art,
               "--grain", str(ART_GRAIN), "--seconds", str(ART_SECONDS), "--out", d]
        with open(d + ".log", "w") as log:
            if subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"simulation failed: {d}.log")

    while True:
        todo = [(m["load"], t) for m in bracket() for t in (m["a"], m["b"]) if result(m["load"], t) is None]
        if not todo:
            break
        with ThreadPoolExecutor(args.jobs) as ex:
            list(ex.map(one, todo))
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(one_art, [j for _, a, b in SATISFYING for j in (a, b)]))
    for m in bracket():
        print(f"{m['name']:16s} {m['load']:8s} {TOOLS[m['a']]['label']} {100*m['ra']:.0f}%  vs  "
              f"{TOOLS[m['b']]['label']} {100*m['rb']:.0f}%  ->  {TOOLS[m['winner']]['label']}", flush=True)


def frames_done(out, n):
    fd = os.path.join(out, "frames")
    if not os.path.isdir(fd):
        return 0
    return sum(os.path.getsize(os.path.join(fd, f)) > 0 for f in os.listdir(fd))


def blender_cmd(args):
    """How to run bowl_scene.py: with the bpy module, or inside a Blender install."""
    script = os.path.join(HERE, "bowl_scene.py")
    cmd = [args.blender, "-b", "-P", script, "--"] if args.blender else [sys.executable, "-W", "ignore", script]
    return cmd, (["--gpu"] if args.gpu else [])


def render_jobs():
    """(key, left cache, right cache, extra scene arguments, which --res to use) for every shot of the film."""
    jobs = [(m["key"], run_dir(m["load"], m["a"]), run_dir(m["load"], m["b"]), [], "res") for m in bracket()]
    if len(jobs) == 7:
        jobs += [(key, art_dir(*a), art_dir(*b), STEEP, "sat_res") for key, a, b in SATISFYING]
    return jobs


def stage_render(args):
    """Render every shot in order, waiting for simulations that are still running."""
    import time
    total = 7 + len(SATISFYING)
    cmd0, gpu = blender_cmd(args)
    while True:
        jobs = render_jobs()
        pending = None
        for key, left, right, extra, res in jobs:
            metas = [os.path.join(d, "meta.json") for d in (left, right)]
            if not all(os.path.exists(p) for p in metas):
                pending = "wait"
                break
            n = min(json.load(open(p))["n_frames"] for p in metas)
            if frames_done(os.path.join(args.frames, key), n) < n:
                pending = (key, left, right, extra, res)
                break
        if pending is None:
            if len(jobs) == total:
                break
            pending = "wait"
        if pending == "wait":
            time.sleep(20)
            continue
        key, left, right, extra, res = pending
        out = os.path.join(args.frames, key)
        fd = os.path.join(out, "frames")
        if os.path.isdir(fd):                             # placeholders left by an interrupted render
            for f in os.listdir(fd):
                if os.path.getsize(os.path.join(fd, f)) == 0:
                    os.remove(os.path.join(fd, f))
        print(f"rendering {key}", flush=True)
        cmd = cmd0 + ["--left", left, "--right", right, "--no-hud", "--anim", "--res", getattr(args, res),
                      "--samples", str(args.samples), "--out", out] + extra + gpu
        with open(os.path.join(args.frames, key + ".log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("render complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sims", "render", "compose", "finish", "bracket"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--frames", default=os.path.join(HERE, "renders", "film"))
    ap.add_argument("--res", default="960x420", help="render size of the matches (picture area is 16:7)")
    ap.add_argument("--sat-res", default="854x480", help="render size of the closing full-frame shots (16:9)")
    ap.add_argument("--gpu", action="store_true", help="render on the GPU")
    ap.add_argument("--blender", help="path to a Blender executable; default is the bpy Python module")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "mixing_knockout.mp4"))
    ap.add_argument("--final", default=os.path.join(HERE, "the_mix_off_PREVIEW_720p.mp4"))
    ap.add_argument("--crf", type=int, default=22, help="quality of the finished file (lower is larger)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="output size of compose: 1 = 1280x720, 1.5 = 1920x1080, 3 = 3840x2160")
    ap.add_argument("--partial", type=int, default=0, metavar="N",
                    help="compose a preview: at most N frames per match, from whatever is rendered")
    args = ap.parse_args()
    if args.stage == "sims":
        stage_sims(args)
    elif args.stage == "render":
        os.makedirs(args.frames, exist_ok=True)
        stage_render(args)
    elif args.stage == "finish":
        base = os.path.splitext(args.out)[0]
        wav = base + "_score.wav"
        subprocess.run([sys.executable, os.path.join(HERE, "film_score.py"), base + "_timeline.json", wav], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out, "-i", wav, "-c:v", "libx264", "-crf", str(args.crf),
                        "-preset", "slow", "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.final], check=True)
        print("wrote", args.final, f"{os.path.getsize(args.final) / 1e6:.0f} MB")
    elif args.stage == "bracket":
        for m in bracket():
            print(m)
    else:
        import film_compose
        film_compose.compose(args, bracket())


if __name__ == "__main__":
    main()
