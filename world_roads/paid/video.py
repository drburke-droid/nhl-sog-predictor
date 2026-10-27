"""The best-paid athletes, 1913-2025, every salary in 2025 dollars: a bar-chart race (1920x1080, 30 fps)

    python video.py still 1930.0 1998.5 ...   -> out/still_1930.0.png ... (t in years; < FIRST is the title)
    python video.py render [jobs]             -> out/race.mp4 (chunks rendered in parallel)

The ten best-paid each year, by salary in 2025 dollars (../media/paid/salaries.csv): bars slide past each other as
the ranks change, grow and shrink with the pay, slide in from below the board and out of it. Each bar carries the
player, his team and league, the salary in 2025 dollars, and what it was in its own year's money. The scale follows the
leader. While it is small, a dashed line marks today's MLB minimum salary ($760,000 in 2025): the stars of the 1920s
and 1930s earned, in today's money, about what a rookie on the minimum makes now.

Colour encodes the league (MLB blue, NBA orange, NFL green, NHL grey: three hues validated for colour-blind
separation on this background, the fourth league neutral) and every bar also names its league in text.
"""
import os, sys, csv, json, time, subprocess
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, translucent text  # noqa: E402

MEDIA = os.path.join(ROOT, "media", "paid")
OUT = os.path.join(HERE, "out")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 7, 13], np.float32)                 # #05070d
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
GOLD = (255, 186, 80)
LEAGUE = {"MLB": (57, 135, 229), "NBA": (217, 89, 38), "NFL": (25, 158, 112), "NHL": (150, 156, 168)}
font = ST.font

FIRST, LAST = 1913, 2025
N = 10                                                # bars on the board
PER_YEAR = 1.4                                        # seconds a year takes
INTRO, OUTRO = 5.6, 11.2                              # whole bars of the score (2.8 s)
MIN_2025 = 760_000                                    # MLB minimum salary, 2025
X0, X1 = 560, 1500                                    # bar baseline and the leader's bar end
Y0, PITCH, BAR = 186, 68, 48                          # first row's top, row spacing, bar thickness


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def money(v, short=True):
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"
    if v >= 1e7:
        return f"${v / 1e6:.0f}M" if short and v >= 1e8 else f"${v / 1e6:.1f}M"
    if v >= 1e6:
        return f"${v / 1e6:.2f}M"
    if v >= 1e4:
        return f"${v / 1e3:.0f}K" if short else f"${v:,.0f}"
    return f"${v:,.0f}"


class Data:
    def __init__(self):
        self.by = defaultdict(dict)                   # year -> pid -> row
        for r in csv.DictReader(open(os.path.join(MEDIA, "salaries.csv"))):
            y = int(r["year"])
            r["real"], r["nominal"] = float(r["real_2025"]), float(r["nominal"])
            self.by[y][r["league"] + ":" + r["pid"]] = r
        # ties keep one order from year to year (by name), so two equal salaries don't swap places
        self.order = {y: sorted(d, key=lambda k: (-d[k]["real"], d[k]["player"])) for y, d in self.by.items()}
        self.rank = {y: {k: i for i, k in enumerate(o)} for y, o in self.order.items()}
        self.cov = {int(r["year"]): r for r in csv.DictReader(open(os.path.join(MEDIA, "coverage.csv")))}
        self.events = json.load(open(os.path.join(HERE, "captions.json"))) if os.path.exists(os.path.join(HERE, "captions.json")) else []

    def top(self, y, n):
        return self.order.get(y, [])[:n]

    def slot(self, y, k, other):
        """a bar's place in year y: its rank; below the board if it has no salary that year (arriving or gone),
        spaced by its rank in the other year so that no two share a slot"""
        r = self.rank.get(y, {}).get(k)
        if r is not None:
            return min(r, N + 8)
        return N + 1.5 + self.rank.get(other, {}).get(k, N) * 0.35

    def state(self, t):
        """every bar on or near the board at time t (in years): key -> (rank, real value, row of the nearest year)"""
        y0 = int(np.floor(t))
        u = t - y0
        y1 = min(LAST, y0 + 1)
        keys = set(self.top(y0, N + 2)) | set(self.top(y1, N + 2))
        out = {}
        mv = smooth((u - 0.08) / 0.84)               # ranks move through the middle of the year
        for k in keys:
            a, b = self.by[y0].get(k), self.by[y1].get(k)
            r0, r1 = self.slot(y0, k, y1), self.slot(y1, k, y0)
            v0 = a["real"] if a else b["real"]
            v1 = b["real"] if b else a["real"]
            row = (b if u >= 0.5 and b else a) or b
            out[k] = (r0 + (r1 - r0) * mv, v0 + (v1 - v0) * u, row, int(round(t)) if (b if u >= 0.5 else a) else (y1 if b else y0))
        return out


D = None


def scale_track():
    """the axis maximum at every frame: the leader's value with room for its label, smoothed both ways in time"""
    n = int(round((LAST - FIRST) * PER_YEAR * FPS)) + 1
    ts = FIRST + np.arange(n) / FPS / PER_YEAR
    lead = []
    for t in ts:
        st = D.state(t)
        on = [v for r, v, _, _ in st.values() if r < N]
        lead.append(max(on) if on else 1.0)
    lead = np.array(lead)
    k = int(0.9 * FPS)
    pad = np.pad(lead, k, mode="edge")
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.2)) ** 2)
    ker /= ker.sum()
    sm = np.convolve(pad, ker, mode="valid")
    return np.maximum(sm, lead * 0.97)                # never let the leader run off the scale


SCALE = None


def scale_at(t):
    i = int(round((min(max(t, FIRST), LAST) - FIRST) * PER_YEAR * FPS))
    return SCALE[min(i, len(SCALE) - 1)]


_BG = None


def canvas():
    global _BG
    if _BG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = 1 - 0.35 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.8)) ** 2)
        _BG = (BG[None, None] * np.clip(v, 0.5, 1)[..., None]).astype(np.uint8)
    return Image.fromarray(_BG.copy())


def ticks(d, xmax, a=1.0):
    """gridlines and labels for the scale, nice steps fading in and out as the scale moves"""
    steps = [m * 10 ** e for e in range(3, 9) for m in (1, 2, 2.5, 5)]
    seen = {}
    for s in steps:
        dens = xmax / s
        al = smooth((dens - 2.2) / 1.2) * (1 - smooth((dens - 6.5) / 2.0))
        if al <= 0.02:
            continue
        v = s
        while v <= xmax * 1.001:
            seen[v] = max(seen.get(v, 0), al)
            v += s
    for v, al in seen.items():
        x = X0 + (X1 - X0) * v / xmax
        if x > X1 + 150:
            continue
        al *= a
        d.line([(x, Y0 - 14), (x, Y0 + N * PITCH - 8)], fill=(40, 52, 72, int(140 * al)), width=1)
        lab = f"${v / 1e6:g}M" if v >= 1e6 else f"${v / 1e3:g}K"
        d.text((x, Y0 - 22), lab, font=font(20, "Regular"), fill=MUTED + (int(255 * al),), anchor="mb")


def board(im, t, a=1.0, hold=False):
    d = ImageDraw.Draw(im, "RGBA")
    xmax = scale_at(t)
    xs = lambda v: X0 + (X1 - X0) * v / xmax
    ticks(d, xmax, a)
    # today's minimum salary, while the scale is small enough for it to mean something
    xm = xs(MIN_2025)
    am = a * smooth((xm - X0 - 40) / 60) * (1 - smooth((t - 1975) / 4))
    if am > 0.01:
        for y in range(Y0 - 6, Y0 + N * PITCH - 8, 14):
            d.line([(xm, y), (xm, y + 7)], fill=GOLD + (int(200 * am),), width=2)
        d.text((xm + 8, Y0 + N * PITCH - 16), "today's MLB minimum, $760K", font=font(20, "Regular"), fill=GOLD + (int(230 * am),), anchor="ls")
    st = D.state(t)
    for k, (r, v, row, yr) in sorted(st.items(), key=lambda kv: -kv[1][0]):
        al = a * smooth((N + 0.45 - r) / 0.9)
        if al <= 0.01:
            continue
        y = Y0 + r * PITCH
        col = LEAGUE[row["league"]]
        x1 = max(X0 + 6, xs(v))
        d.rounded_rectangle([X0, y, x1, y + BAR], 4, fill=col + (int(255 * al),))
        d.rectangle([X0, y, min(x1, X0 + 6), y + BAR], fill=col + (int(255 * al),))
        if r < 0.5 and not hold:                     # the leader
            d.rounded_rectangle([X0 - 1, y - 1, x1 + 1, y + BAR + 1], 5, outline=GOLD + (int(150 * al * (1 - r * 2)),), width=2)
        d.text((X0 - 18, y + BAR / 2 - 11), row["player"], font=font(29, "SemiBold"), fill=INK + (int(255 * al),), anchor="rm")
        d.text((X0 - 18, y + BAR / 2 + 15), f"{row['team']} · {row['circuit']}", font=font(19, "Regular"), fill=INK2 + (int(235 * al),), anchor="rm")
        d.text((x1 + 14, y + BAR / 2 - 9), money(v, short=False), font=font(28, "Bold"), fill=INK + (int(255 * al),), anchor="lm")
        d.text((x1 + 14, y + BAR / 2 + 15), f"{money(row['nominal'], short=False)} in {row['year']}", font=font(19, "Regular"),
               fill=MUTED + (int(235 * al),), anchor="lm")
    return d


def header(d, a=1.0):
    d.text((90, 62), "The best-paid athletes", font=font(46, "Bold"), fill=INK + (int(255 * a),), anchor="ls")
    d.text((90, 104), "Salaries in 2025 dollars · MLB, NBA, NFL, NHL", font=font(25, "Regular"), fill=INK2 + (int(255 * a),), anchor="ls")
    x = W - 90
    for lg in ("NHL", "NFL", "NBA", "MLB"):
        w = d.textlength(lg, font=font(22, "SemiBold"))
        d.text((x, 62), lg, font=font(22, "SemiBold"), fill=INK + (int(255 * a),), anchor="rs")
        d.rounded_rectangle([x - w - 30, 44, x - w - 10, 64], 4, fill=LEAGUE[lg] + (int(255 * a),))
        x -= w + 58


def year_label(d, t, a=1.0):
    y = int(round(min(max(t, FIRST), LAST)))
    d.text((W - 90, H - 78), str(y), font=font(132, "Bold"), fill=INK + (int(235 * a),), anchor="rs")
    f = float(D.cov[y]["to_2025"])
    d.text((W - 92, H - 40), f"$1 in {y} = ${f:,.2f} in 2025", font=font(24, "Regular"), fill=INK2 + (int(255 * a),), anchor="rs")


def caption(d, t, a=1.0):
    for e in D.events:
        t0, t1 = e["from"], e["to"]
        al = a * smooth((t - t0) / 0.35) * (1 - smooth((t - t1) / 0.35))
        if al <= 0.01:
            continue
        lines = e["text"] if isinstance(e["text"], list) else [e["text"]]
        y = H - 96
        for i, line in enumerate(lines):
            d.text((90, y + i * 38), line, font=font(30 if i == 0 else 24, "SemiBold" if i == 0 else "Regular"),
                   fill=(INK if i == 0 else INK2) + (int(255 * al),), anchor="ls")


def frame(T):
    """T: seconds from the start of the film"""
    im = canvas()
    if T < INTRO:
        d = ImageDraw.Draw(im, "RGBA")
        a = smooth(T / 0.8) * (1 - smooth((T - INTRO + 1.0) / 0.8))
        d.text((W / 2, 420), "The best-paid athletes", font=font(72, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((W / 2, 500), f"{FIRST} to {LAST}, every salary in 2025 dollars", font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((W / 2, 600), "MLB · NBA · NFL · NHL", font=font(26, "SemiBold"), fill=MUTED + (int(255 * a),), anchor="mm")
        return np.asarray(im)
    t = FIRST + (T - INTRO) / PER_YEAR
    hold = t >= LAST
    t = min(t, LAST)
    a = smooth((T - INTRO) / 0.8)
    end = total()
    dim = 1 - 0.72 * smooth((T - (end - OUTRO + 1.6)) / 0.8)      # the board steps back for the summary
    d = board(im, t, a * dim, hold)
    header(d, a)
    year_label(d, t, a * dim)
    caption(d, FIRST + (T - INTRO) / PER_YEAR, a)
    if T > end - OUTRO + 1.6:
        summary(d, T - (end - OUTRO + 1.6))
    return np.asarray(im)


def summary(d, s):
    """over the final standings: the first best-paid star against the last, in the same money"""
    a = smooth(s / 0.8) * (1 - smooth((s - (OUTRO - 2.6)) / 0.8))
    if a <= 0.01:
        return
    then, now = SUMMARY["then"], SUMMARY["now"]
    cx = W / 2
    d.text((cx, 300), "In the same money", font=font(34, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    for i, (r, x) in enumerate(((then, cx - 380), (now, cx + 380))):
        d.text((x, 400), r["player"], font=font(44, "Bold"), fill=INK + (int(255 * a),), anchor="mm")
        d.text((x, 452), f"{r['team']} · {r['circuit']} · {r['year']}", font=font(26, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
        d.text((x, 560), money(r["real"], short=False), font=font(86, "Bold"), fill=(GOLD if i else INK) + (int(255 * a),), anchor="mm")
        d.text((x, 640), f"{money(r['nominal'], short=False)} in {r['year']} dollars", font=font(26, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")
    d.text((cx, 560), "→", font=font(64, "Regular"), fill=INK2 + (int(255 * a),), anchor="mm")
    d.text((cx, 740), f"{now['real'] / then['real']:.0f} times as much", font=font(40, "SemiBold"), fill=INK + (int(255 * a),), anchor="mm")
    d.text((cx, 1010), SOURCES, font=font(19, "Regular"), fill=MUTED + (int(255 * a),), anchor="mm")


SOURCES = ("Salaries: Baseball Reference (Michael Haupert's research of the Hall of Fame's contract cards), Lahman database, "
           "Basketball Reference, OverTheCap, CapWages, published contracts · CPI-U: Federal Reserve Bank of Minneapolis")
SUMMARY = {}


def total():
    return INTRO + (LAST - FIRST) * PER_YEAR + OUTRO


def setup():
    global D, SCALE
    D = Data()
    SCALE = scale_track()
    # the summary: Babe Ruth's best year, and the best-paid of the last year
    ruth = [r for d in D.by.values() for r in d.values() if r["player"] == "Babe Ruth"]
    SUMMARY["then"] = max(ruth, key=lambda r: r["real"]) if ruth else D.by[FIRST][D.order[FIRST][0]]
    SUMMARY["now"] = D.by[LAST][D.order[LAST][0]]
    for r in SUMMARY.values():
        r["year"] = int(r["year"])


def render_range(c0, c1, part):
    setup()
    tmp = part + ".tmp.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-tune", "animation",
                          "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    for i in range(c0, c1):
        p.stdin.write(np.ascontiguousarray(frame(i / FPS)).tobytes())
    p.stdin.close()
    p.wait()
    os.replace(tmp, part)


def render(jobs=4):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    n = int(round(total() * FPS))
    CH = 300
    parts = [(c0, min(n, c0 + CH), os.path.join(OUT, "chunks", f"race_{c0:05d}.mp4")) for c0 in range(0, n, CH)]
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        futs = [ex.submit(render_range, *p) for p in parts if not os.path.exists(p[2])]
        for f in futs:
            f.result()
    print(f"{n} frames in {time.time() - t0:.0f} s", flush=True)
    lst = os.path.join(OUT, "chunks", "race.txt")
    open(lst, "w").write("".join(f"file '{p[2]}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(OUT, "race.mp4")], check=True)
    for p in parts:
        os.remove(p[2])
    print("wrote", os.path.join(OUT, "race.mp4"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "still":
        setup()
        for a in sys.argv[2:]:
            t = float(a)
            T = INTRO + (t - FIRST) * PER_YEAR if t >= FIRST else t
            Image.fromarray(frame(T)).save(os.path.join(OUT, f"still_{t:06.1f}.png"))
            print("still", t)
    else:
        render(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
