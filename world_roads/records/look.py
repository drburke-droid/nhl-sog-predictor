"""The look of the records film: a dark track-at-night surface, Inter type, the world record in gold and
the Olympic field in blue (the reference palette's dark steps, validated as a pair on this surface:
CVD dE 27.2, normal-vision dE 30.2, both >= 3:1 contrast). Geometry is in 1080p units.

Frames are drawn with matplotlib (Agg) and piped to ffmpeg, several workers at once.
"""
import os, sys, math, subprocess, functools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30

BG = "#0f1215"
INK = "#f0efec"
INK2 = "#b9b8b0"
MUTED = "#7d7c76"
GRID = "#262a30"
AXIS = "#3a3f46"
GOLD = "#c98500"          # the world record
BLUE = "#3987e5"          # the Olympic field
GREY = "#565b63"          # context lines (other events)

plt.rcParams.update({
    "font.family": "Inter", "font.size": 20, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.major.size": 0, "ytick.major.size": 0, "savefig.facecolor": BG,
})


def px(x):
    """1080p units -> figure points (the figure is 19.2 x 10.8 in at 100 dpi, so 1 unit = 0.72 pt)"""
    return x * 0.72


def fs(units):
    """font size in 1080p pixel units (cap height-ish) -> points"""
    return units * 0.72


def clamp(x, a=0.0, b=1.0):
    return min(max(x, a), b)


def ease(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def fade(t, t0, t1, fin=0.5, fout=0.5):
    if t < t0 or t > t1:
        return 0.0
    a = 1.0 if fin <= 0 else (t - t0) / fin
    b = 1.0 if fout <= 0 else (t1 - t) / fout
    return clamp(min(a, b))


def new_fig(scale=1):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100 * scale, facecolor=BG)
    return fig


def ax_px(fig, x0, y0, x1, y1):
    """an axes placed by its 1080p pixel box (y measured from the top)"""
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], facecolor="none")


def text(fig, x, y, s, size=28, color=INK, weight="regular", ha="left", va="baseline", alpha=1.0,
         family="Inter", **kw):
    """text placed in 1080p pixels from the top-left"""
    if alpha <= 0.001:
        return None
    return fig.text(x / W, 1 - y / H, s, fontsize=fs(size), color=color, fontweight=weight, ha=ha, va=va,
                    alpha=clamp(alpha), family=family, **kw)


def title(fig, x, y, s, size=64, color=INK, alpha=1.0, ha="left", va="baseline", **kw):
    return text(fig, x, y, s, size=size, color=color, weight="bold", family="Inter Display", alpha=alpha,
                ha=ha, va=va, **kw)


def style_axes(ax, grid_y=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", labelsize=fs(19), pad=px(8), colors=INK2)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=1.0)
        ax.set_axisbelow(True)


def ring(ax, x, y, color, size=10, alpha=1.0, z=5, edge=BG, ew=2.0):
    """a marker with a surface-coloured ring so overlapping marks stay separate"""
    ax.scatter([x], [y], s=(px(size)) ** 2, color=color, edgecolors=edge, linewidths=px(ew), alpha=alpha,
               zorder=z)


def outlined(lw=4):
    return [pe.withStroke(linewidth=px(lw), foreground=BG)]


def frame_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3]


# ------------------------------------------------------------------ rendering harness
def render_section(name, section, scale=1, workers=4, crf=16):
    """render a section (object with .dur and .draw(fig, t)) to out/<name>.mp4 using several processes"""
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    n = int(round(section.dur * FPS))
    chunks = [(i * n // workers, (i + 1) * n // workers) for i in range(workers)]
    parts = [os.path.join(OUT, f"{name}.part{i}.mp4") for i in range(workers)]
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_render_chunk, args=(section, a, b, p, scale, crf)) for (a, b), p in zip(chunks, parts)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker failed for {name}")
    lst = os.path.join(OUT, f"{name}.parts.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.basename(p)}'\n")
    dst = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", dst],
                   check=True, cwd=OUT)
    for p in parts:
        os.remove(p)
    os.remove(lst)
    return dst


def _render_chunk(section, a, b, path, scale, crf):
    w, h = W * scale, H * scale
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                           "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                           "-colorspace", "bt709", path], stdin=subprocess.PIPE)
    fig = new_fig(scale)
    for i in range(a, b):
        fig.clf()
        section.draw(fig, i / FPS)
        ff.stdin.write(np.ascontiguousarray(frame_rgb(fig)).tobytes())
    ff.stdin.close()
    ff.wait()
    plt.close(fig)


def still(section, t, path, scale=1):
    fig = new_fig(scale)
    section.draw(fig, t)
    fig.savefig(path, dpi=100 * scale, facecolor=BG)
    plt.close(fig)
