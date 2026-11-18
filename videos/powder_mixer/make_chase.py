"""
"Can it reach 100% mixed?"  One mixer, one whisk, stirring for a minute, with
the score, a stroke counter and how the score is graded on screen.

    python bowl_sim.py --tool whisk --load wedges --seconds 60 --out cache/chase/whisk_60
    python bowl_scene.py --left cache/chase/whisk_60 --no-hud --res 2220x2160 --samples 128 --elev 34 --dist 372 \
        --target 8,8,68 --final //frames/CH/f_ --save cache/chase/shot_CH.blend
    ./render_chase.sh                       # 722 frames at 1110x1080, about 5 hours on 4 cores
    python make_chase.py                    # -> the_chase_for_100_1080p.mp4

The first 24 seconds of stirring play in real time; the rest is shown at six
times speed.  The finish line is the score of a perfect shuffle of the same
grains, which is not exactly 100: chance makes it 99.5 to 100.5 (95% range of
400 shuffles; see perfect_shuffle_band()).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw

import film_compose as F
from bowl_sim import mix_index

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_FRAMES, STEP = 576, 6                       # must match render_chase.sh
BAND = (99.5, 100.5)
PIC_W = 740                                      # layout units; the picture is PIC_W x 720
X0, X1 = 778, 1240                               # the panel
GREEN, GREEN_BG = (0, 122, 61), (214, 240, 224)


def perfect_shuffle_band(run, raw0, n=400, seed=1):
    """Score of the same grains with their colours dealt at random: mean and 95% range."""
    rng = np.random.default_rng(seed)
    P = np.asarray(run.pos[-1], np.float64)
    v = [100 * (mix_index(P, rng.permutation(run.ingr), run.R, 6) - raw0) / (1 - raw0) for _ in range(n)]
    return float(np.mean(v)), [float(x) for x in np.percentile(v, [2.5, 97.5])]


def gridded_disc(run, i, size):
    disc = run.disc(i, size)
    grid = Image.new("RGBA", disc.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    px = disc.size[0] / (2 * run.R)
    k = F.SAMPLE_CELL
    while k < 2 * run.R:
        v = int(round(k * px))
        g.line((v, 0, v, disc.size[1]), fill=(20, 23, 28, 170), width=max(1, F.spx(1.0)))
        g.line((0, v, disc.size[0], v), fill=(20, 23, 28, 170), width=max(1, F.spx(1.0)))
        k += F.SAMPLE_CELL
    grid.putalpha(ImageChops.multiply(grid.getchannel("A"), disc.getchannel("A")))
    return Image.alpha_composite(disc, grid)


def chart(score, count, i, w, h, cmax, col):
    ss = 2
    q = ss * F.K
    im = Image.new("RGB", (int(round(w * q)), int(round(h * q))), F.PAPER)
    d = ImageDraw.Draw(im)
    lw = max(1, int(round(q)))
    x0, x1, y0, y1 = 30 * q, (w - 8) * q, 8 * q, (h - 22) * q
    X = lambda c: x0 + (x1 - x0) * min(c / cmax, 1.0)
    Y = lambda v: y1 - (y1 - y0) * min(max(v, 0), 101) / 101
    d.rectangle((x0, Y(BAND[1]), x1, Y(BAND[0])), fill=GREEN_BG)
    for v in (0, 50, 100):
        d.line((x0, Y(v), x1, Y(v)), fill=F.HAIR if v != 100 else GREEN, width=lw)
        F.text(d, (x0 - 6 * q, Y(v)), str(v), F.font(10 * ss, "Medium"), F.MUTED, "rm")
    for c in range(0, int(cmax) + 1, 30):
        F.text(d, (X(c), y1 + 5 * q), str(c), F.font(10 * ss, "Medium"), F.MUTED, "ma")
    F.text(d, (x1, y1 + 5 * q), "strokes", F.font(10 * ss, "Medium"), F.MUTED, "ra")
    pts = [(X(count[j]), Y(score[j])) for j in range(0, i + 1)]
    if len(pts) > 1:
        d.line(pts, fill=col, width=int(round(2.6 * q)), joint="curve")
    px, py = pts[-1]
    d.ellipse((px - 4 * q, py - 4 * q, px + 4 * q, py + 4 * q), fill=col, outline=F.PAPER, width=lw)
    return im.resize((F.spx(w), F.spx(h)), Image.LANCZOS)


def frame(run, score, i, pic, info, speed=1):
    im = F.new_canvas()
    im.paste(pic, (0, 0))
    d = F.Draw(im)
    col = F.tool_rgb("whisk")
    v, strokes, secs = float(score[i]), float(run.count[i]), float(min(run.t[i], run.meta["seconds"]))
    if speed > 1:                                # fast-forward badge on the picture
        lab = f"{speed}x speed"
        lw = d.textlength(lab, F.font(20, "Bold"))
        d.rounded_rectangle((24, 24, 24 + lw + 40, 66), 21, fill=(255, 255, 255), outline=F.HAIR)
        F.text(d, (44, 45), lab, F.font(20, "Bold"), F.INK, "lm")
    F.text(d, (X0, 34), "ONE WHISK. ONE BOWL. KEEP STIRRING.", F.font(14, "Bold"), F.MUTED, "la")
    F.text(d, (X0, 56), "Can it reach 100% mixed?", F.font(35, "Bold", True), F.INK, "la")
    # the score
    pct = f"{v:.1f}%"
    F.text(d, (X0 - 4, 226), pct, F.font(118, "Bold", True), F.INK, "ls")
    F.text(d, (X0 + d.textlength(pct, F.font(118, "Bold", True)) + 10, 226), "mixed", F.font(28, "SemiBold"), F.MUTED, "ls")
    d.rounded_rectangle((X0, 246, X1, 258), 6, fill=F.TRACK)
    d.rounded_rectangle((X0, 246, X0 + max((X1 - X0) * min(v, 100) / 100, 12), 258), 6, fill=col)
    # the last five points, with the band a perfect shuffle scores in
    F.text(d, (X0, 282), "THE LAST FIVE POINTS", F.font(13, "Bold"), F.MUTED, "la")
    g0, g1, lo, hi = X0, X1, 95.0, 101.0
    GX = lambda s: g0 + (g1 - g0) * (min(max(s, lo), hi) - lo) / (hi - lo)
    d.rounded_rectangle((g0, 306, g1, 326), 10, fill=F.TRACK)
    d.rounded_rectangle((GX(BAND[0]), 306, GX(BAND[1]), 326), 0, fill=GREEN_BG)
    d.line((GX(100), 302, GX(100), 330), fill=GREEN, width=2)
    for s in (95, 96, 97, 98, 99, 100):
        F.text(d, (GX(s), 334), str(s), F.font(12, "Medium"), GREEN if s == 100 else F.MUTED, "ma")
    if v >= lo:
        cx = GX(v)
        d.rounded_rectangle((cx - 9, 303, cx + 9, 329), 9, fill=col, outline=(255, 255, 255), width=2)
    note = "A perfect shuffle scores in the green band: 99.5 to 100.5."
    if info["touch_frame"] is not None and i >= info["touch_frame"]:
        note = f"In the green band on stroke {info['touch_stroke']}. A perfect shuffle scores 99.5 to 100.5."
    F.text(d, (X0, 356), note, F.font(14, "Medium"), GREEN, "la")
    # counters
    for k, (lab, val) in enumerate((("STROKES", f"{int(strokes)}"), ("SECONDS", f"{secs:.1f}"))):
        x = X0 + k * 150
        F.text(d, (x, 396), lab, F.font(13, "Bold"), F.MUTED, "la")
        F.text(d, (x, 456), val, F.font(46, "Bold", True), F.INK, "ls")
    # top-down view under the grading grid, and the curve so far
    F.paste_disc(im, gridded_disc(run, i, 150), X0, 478)
    ch = chart(score, run.count, i, X1 - X0 - 168, 156, info["cmax"], col)
    im.paste(ch, (F.spx(X0 + 168), F.spx(476)))
    # how it is graded
    F.text(d, (X0, 646), "HOW WE GRADE IT", F.font(13, "Bold"), F.MUTED, "la")
    F.text(d, (X0, 667), f"Cut the bowl into {info['scoops']} scoops. Count the six colours in each.", F.font(15, "Medium"), F.INK, "la")
    F.text(d, (X0, 688), "0% = as loaded.  100% = the scoops match a perfect shuffle.", F.font(15, "Medium"), F.INK, "la")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(HERE, "cache", "chase", "whisk_60"))
    ap.add_argument("--frames", default=os.path.join(HERE, "renders", "chase", "CH", "frames"))
    ap.add_argument("--out", default=os.path.join(HERE, "the_chase_for_100_1080p.mp4"))
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--stills", help="write a few test frames to this folder instead of the film")
    a = ap.parse_args()
    F.set_scale(a.scale)
    run = F.Run(a.run)
    z = np.load(os.path.join(a.run, "meta.npz"))
    score = 100 * np.asarray(z["mix"], float)                    # not clipped: a shuffle can score just over 100
    raw0 = float(z["mix_raw"][0])
    mean, band = perfect_shuffle_band(run, raw0)
    print(f"perfect shuffle: mean {mean:.2f}, 95% range {band[0]:.2f} to {band[1]:.2f}")
    p0 = np.asarray(run.pos[0])
    nc = int(np.ceil(2 * run.R / F.SAMPLE_CELL))
    cells = np.bincount(np.clip(((p0[:, 0] + run.R) / F.SAMPLE_CELL).astype(int), 0, nc - 1) * nc
                        + np.clip(((p0[:, 1] + run.R) / F.SAMPLE_CELL).astype(int), 0, nc - 1), minlength=nc * nc)
    touch = int(np.argmax(score >= BAND[0])) if (score >= BAND[0]).any() else None
    stirring = run.t < run.meta["seconds"] - 0.6
    after = stirring & (np.arange(run.n) >= (touch or 0))
    info = dict(scoops=int((cells > 0.35 * cells.max()).sum()), cmax=float(run.count.max()), touch_frame=touch,
                touch_stroke=int(np.ceil(run.count[touch])) if touch is not None else None)
    seq = [(f, 1) for f in range(0, REAL_FRAMES)] + [(f, STEP) for f in range(REAL_FRAMES + STEP - 1, run.n, STEP)]
    pic_size = (F.spx(PIC_W), F.spx(F.H))

    def picture(f):
        return Image.open(os.path.join(a.frames, f"f_{f + 1:04d}.png")).convert("RGB").resize(pic_size, Image.LANCZOS)

    if a.stills:
        os.makedirs(a.stills, exist_ok=True)
        for f, sp in ((0, 1), (120, 1), (300, 1), (touch, 1), (seq[-40][0], STEP)):
            p = os.path.join(a.frames, f"f_{f + 1:04d}.png")
            pic = picture(f) if os.path.exists(p) and os.path.getsize(p) else Image.new("RGB", pic_size, (70, 72, 78))
            frame(run, score, f, pic, info, sp).save(os.path.join(a.stills, f"chase_{f:04d}.png"))
        return

    silent = os.path.splitext(a.out)[0] + "_silent.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{F.spx(F.W)}x{F.spx(F.H)}", "-r", "24",
           "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-crf", str(a.crf),
           "-preset", "slow", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", silent]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0

    def emit(im, repeat=1):
        nonlocal n
        raw = im.tobytes()
        for _ in range(repeat):
            enc.stdin.write(raw)
        n += repeat

    first = frame(run, score, 0, picture(0), info)
    emit(first, 24)                                              # a beat on the untouched bowl
    t_touch = None
    last = first
    for f, sp in seq:
        last = frame(run, score, f, picture(f), info, sp)
        if touch is not None and t_touch is None and f >= touch:
            t_touch = n / 24
        emit(last)
    emit(last, 36)
    lo, hi = float(score[after].min()), float(score[after].max())
    lines = [(f"{info['touch_stroke']} strokes to reach a perfect shuffle's score.", 44, "Bold", F.INK, 22),
             (f"The next {run.count.max() - info['touch_stroke']:.0f} strokes changed nothing: it stayed between {lo:.1f}% and {hi:.1f}%,",
              24, "Medium", F.MUTED, 8),
             (f"averaging {score[after].mean():.1f}%. Once it is as good as a shuffle, more stirring cannot improve it.", 24, "Medium", F.MUTED, 0)]
    card = F.card(lines)
    for k in range(10):
        emit(Image.blend(last, card, (k + 1) / 11))
    emit(card, int(5.0 * 24))
    enc.stdin.close()
    enc.wait()
    seconds = n / 24
    t_card = seconds - 5.0 - 10 / 24
    bell = (t_touch or 0) + 2.7                                  # film_score rings 2.7 s before a match segment ends
    tl = dict(fps=24, seconds=seconds, segments=[["F1", 0.0, bell], ["X1", bell, t_card - bell], ["closing", t_card, seconds - t_card]])
    tlp = os.path.splitext(a.out)[0] + "_timeline.json"
    json.dump(tl, open(tlp, "w"))
    wav = os.path.splitext(a.out)[0] + "_score.wav"
    subprocess.run([sys.executable, os.path.join(HERE, "film_score.py"), tlp, wav], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", a.out],
                   check=True)
    for p in (silent, wav, tlp):
        os.remove(p)
    print(f"wrote {a.out}: {seconds:.1f} s; in the band on stroke {info['touch_stroke']}; afterwards {lo:.2f} to {hi:.2f}, "
          f"mean {score[after].mean():.2f}")


if __name__ == "__main__":
    main()
