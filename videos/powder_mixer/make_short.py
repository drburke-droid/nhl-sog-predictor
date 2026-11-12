"""
A vertical (1080x1920) Short from one match of the knockout: the two bowls
stacked, each rendered on its own so it stays sharp.

    python make_short.py render      # two single-bowl renders (about 75 min on 4 cores)
    python make_short.py compose     # -> same_spoon_SHORT.mp4
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

import film_compose as F
import make_film as MF

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
PANEL_H, PANEL_Y = 780, (214, 1004)
SHORTS = {
    "Q1": dict(headline="SAME SPOON.", sub="Same bowl. Two ways to stir. 15 seconds.",
               labels=("STIRRED IN A SPIRAL", "TWIRLED IN THE MIDDLE"),
               closing=("A tool only mixes", "where it sweeps."), out="same_spoon_two_ways_SHORT.mp4"),
}


def stage_render(args):
    m = {mm["key"]: mm for mm in MF.bracket()}[args.match]
    for side in ("a", "b"):
        out = os.path.join(args.frames, args.match, side)
        n = json.load(open(os.path.join(MF.run_dir(m["load"], m[side]), "meta.json")))["n_frames"]
        if MF.frames_done(out, n) >= n:
            continue
        cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, "bowl_scene.py"), "--left", MF.run_dir(m["load"], m[side]),
               "--no-hud", "--anim", "--res", "720x520", "--samples", str(args.samples), "--dist", "410", "--elev", "35",
               "--target", "0,12,60",
               "--out", out]
        with open(out.rstrip("/") + ".log", "w") if os.path.isdir(os.path.dirname(out)) else open(os.devnull, "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("short render complete", flush=True)


def frame(spec, runs, dirs, i, winner=None, badge=0.0):
    im = Image.new("RGB", (W, H), F.PAPER)
    d = ImageDraw.Draw(im)
    F.text(d, (W // 2, 26), spec["headline"], F.font(118, "Black", True), F.INK, "ma")
    F.text(d, (W // 2, 160), spec["sub"], F.font(30, "Medium"), F.MUTED, "ma")
    n = min(r.n for r in runs)
    k = min(i, n - 1)
    for side, (run, fdir, y) in enumerate(zip(runs, dirs, PANEL_Y)):
        col = F.tool_rgb(run.meta["tool"])
        pic = Image.open(os.path.join(fdir, "frames", f"f_{k + 1:04d}.png")).convert("RGB").resize((W, PANEL_H), Image.LANCZOS)
        im.paste(pic, (0, y))
        lose = winner is not None and winner != side
        # label
        lab = spec["labels"][side]
        lw = int(d.textlength(lab, font=F.font(30, "Bold")))
        d.rounded_rectangle((28, y + 24, 28 + lw + 48, y + 84), 30, fill=(255, 255, 255))
        F.text(d, (52, y + 54), lab, F.font(30, "Bold"), F.INK, "lm")
        # score card
        pct = f"{100 * run.mix[k]:.0f}%"
        d.rounded_rectangle((W - 300, y + 24, W - 28, y + 168), 28, fill=(255, 255, 255))
        F.text(d, (W - 164, y + 82), pct, F.font(96, "Bold", True), F.MUTED if lose and badge > 0.5 else F.INK, "mm")
        d.rounded_rectangle((W - 276, y + 140, W - 52, y + 152), 6, fill=F.TRACK)
        d.rounded_rectangle((W - 276, y + 140, W - 276 + max(int(224 * run.mix[k]), 12), y + 152), 6,
                            fill=(170, 168, 162) if lose and badge > 0.5 else col)
        if winner == side and badge > 0:
            c = tuple(int(255 + (col[q] - 255) * badge) for q in range(3))
            d.rounded_rectangle((W - 300, y + 184, W - 28, y + 244), 30, fill=c)
            F.text(d, (W - 164, y + 214), "WINNER", F.font(32, "Bold"), (255, 255, 255), "mm")
        # top-down view
        disc = run.disc(k, 230)
        F.paste_disc(im, disc, 30, y + PANEL_H - 260, ring=(255, 255, 255))
    F.text(d, (40, H - 66), f"{min(runs[0].t[k], 15.0):4.1f} s", F.font(54, "Bold", True), F.INK, "lm")
    F.text(d, (W - 40, H - 66), "% mixed: how evenly the six powders are spread", F.font(26, "Medium"), F.MUTED, "rm")
    return im


def stage_compose(args):
    spec = SHORTS[args.match]
    m = {mm["key"]: mm for mm in MF.bracket()}[args.match]
    runs = [F.Run(MF.run_dir(m["load"], m[s])) for s in ("a", "b")]
    dirs = [os.path.join(args.frames, args.match, s) for s in ("a", "b")]
    out = os.path.join(HERE, spec["out"])
    silent = out.replace(".mp4", "_silent.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
           "-i", "-", "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count = 0

    def emit(im, repeat=1):
        nonlocal count
        raw = im.tobytes()
        for _ in range(repeat):
            enc.stdin.write(raw)
        count += repeat

    n = min(r.n for r in runs)
    win = 0 if m["winner"] == m["a"] else 1
    emit(frame(spec, runs, dirs, 0), 18)                      # a beat on the untouched bowls
    for i in range(n):
        emit(frame(spec, runs, dirs, i))
    for k in range(12):
        emit(frame(spec, runs, dirs, n - 1, win, (k + 1) / 12))
    last = frame(spec, runs, dirs, n - 1, win, 1.0)
    emit(last, 60)
    card = Image.new("RGB", (W, H), F.PAPER)
    d = ImageDraw.Draw(card)
    for j, line in enumerate(spec["closing"]):
        F.text(d, (W // 2, H // 2 - 90 + j * 120), line, F.font(100, "Black", True), F.INK, "mm")
    for k in range(10):
        emit(Image.blend(last, card, (k + 1) / 11))
    emit(card, 66)
    enc.stdin.close()
    enc.wait()
    seconds = count / 24
    t_bell = (18 + n + 12) / 24
    tl = dict(fps=24, seconds=seconds, segments=[["intro", 0.0, 0.75], ["Q1", 0.75, t_bell + 2.7 - 0.75],
                                                 ["closing", t_bell + 2.7, seconds - t_bell - 2.7]])
    tlp = out.replace(".mp4", "_timeline.json")
    json.dump(tl, open(tlp, "w"))
    wav = out.replace(".mp4", "_score.wav")
    subprocess.run([sys.executable, os.path.join(HERE, "film_score.py"), tlp, wav], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    for p in (silent, wav, tlp):
        os.remove(p)
    print(f"wrote {out}: {seconds:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("render", "compose"))
    ap.add_argument("--match", default="Q1", choices=sorted(SHORTS))
    ap.add_argument("--frames", default=os.path.join(HERE, "renders", "short"))
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.frames, a.match), exist_ok=True)
    (stage_render if a.stage == "render" else stage_compose)(a)
