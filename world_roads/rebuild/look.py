"""Shared look for the rebuild video (1920x1080, 30 fps), in the style of the light-speed film: near-black navy, cyan
for the field, gold for the best, glowing lines, Inter type (captions and fonts from ../signal/style.py).

    stripes()    every franchise's seasons as a grid of cells, coloured by finish (white for a title)
    Glow         lines and dots drawn with a soft bloom
    bars/axes    chart helpers on a PIL ImageDraw
    Section      chunked, resumable rendering of one section to out/<name>.mp4
"""
import os, sys, csv, json, time, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "signal"))
import style as ST                                    # fonts, captions, translucent-text fix  # noqa: E402

OUT = os.path.join(HERE, "out")
MEDIA = os.path.join(ROOT, "media", "rebuild")
W, H, FPS = 1920, 1080, 30
BG = np.array([5, 8, 15], np.float32)
CYAN, GOLD, HOT = (36, 168, 255), (255, 186, 80), (255, 238, 210)
ROSE, VIOLET = (255, 79, 139), (157, 140, 255)        # with gold, the traffic video's CVD-checked set on this background
INK, INK2, MUTED = ST.INK, ST.INK2, ST.MUTED
PANEL = ST.PANEL
LEAGUES = ["nhl", "nba", "mlb", "nfl"]
NAME = {"nhl": "NHL", "nba": "NBA", "mlb": "MLB", "nfl": "NFL"}
font, fade = ST.font, ST.fade


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def results():
    return json.load(open(os.path.join(MEDIA, "results.json")))


def rows(name):
    return list(csv.DictReader(open(os.path.join(MEDIA, name))))


_BG = None


def background():
    global _BG
    if _BG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = 1 - 0.35 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.8)) ** 2)
        _BG = (BG[None, None] * np.clip(v, 0.5, 1)[..., None]).astype(np.float32)
    return _BG


def canvas():
    """an RGB frame: Pillow blends translucent drawing only into RGB images (into RGBA it writes the colour as is)"""
    return Image.fromarray(background().astype(np.uint8))


class Glow:
    """additive emission with a two-scale bloom, laid over a frame"""

    def __init__(self):
        self.E = np.zeros((H, W, 3), np.float32)

    def line(self, pts, col, width=4, a=1.0):
        if len(pts) < 2 or a <= 0:
            return
        layer = np.zeros((H, W), np.uint8)
        cv2.polylines(layer, [np.round(np.array(pts) * 16).astype(np.int32)], False, 255, width, cv2.LINE_AA, shift=4)
        self.E += (layer.astype(np.float32) / 255 * a)[..., None] * np.array(col, np.float32)

    def dot(self, x, y, col, r=7, a=1.0):
        layer = np.zeros((H, W), np.uint8)
        cv2.circle(layer, (int(x * 16), int(y * 16)), int(r * 16), 255, -1, cv2.LINE_AA, shift=4)
        self.E += (layer.astype(np.float32) / 255 * a)[..., None] * np.array(col, np.float32)

    def over(self, im):
        small = cv2.resize(self.E, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 3.0)
        b2 = cv2.GaussianBlur(cv2.resize(small, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        bloom = cv2.resize(cv2.addWeighted(b1, 0.5, cv2.resize(b2, (W // 2, H // 2)), 0.6, 0), (W, H))
        base = np.asarray(im.convert("RGB")).astype(np.float32)
        out = np.clip(base + self.E + bloom * 0.9, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


def rank_colour(p):
    """a season's finish (0 the worst in the league, 1 the best): dark navy through cyan to gold"""
    stops = [(0.0, (16, 26, 48)), (0.45, (22, 86, 150)), (0.7, (40, 160, 235)), (0.9, (240, 170, 70)), (1.0, GOLD)]
    for (a, c0), (b, c1) in zip(stops, stops[1:]):
        if p <= b:
            return lerp(c0, c1, (p - a) / (b - a))
    return stops[-1][1]


def season_col(league, season):
    """seasons by the year they end: NFL and MLB seasons by their year, the NFL's moved on one (2000 ends in 2001)"""
    return int(season) + (1 if league == "nfl" else 0)


def stripes(box, order=None, gap=14, cellgap=4):
    """the grid image (RGBA, box-sized) and row geometry: a row per franchise, richest first in each league"""
    x0, y0, x1, y1 = box
    S, T = rows("seasons.csv"), {(t["league"], t["team"]): t for t in rows("teams.csv")}
    by = {}
    for s in S:
        by.setdefault((s["league"], s["team"]), []).append(s)
    leagues = order or LEAGUES
    teams = {lg: sorted({k[1] for k in by if k[0] == lg}, key=lambda t: -float(T[(lg, t)]["value_musd"])) for lg in leagues}
    n = sum(len(v) for v in teams.values())
    rh = (y1 - y0 - gap * (len(leagues) - 1)) / n
    cw = (x1 - x0) / 25
    im = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    geo, y = {}, 0.0
    for lg in leagues:
        top = y
        for t in teams[lg]:
            geo[(lg, t)] = (y0 + y, y0 + y + rh)
            for s in by[(lg, t)]:
                c = season_col(lg, s["season"]) - 2001
                if 0 <= c < 25:
                    col = HOT if s["champion"] == "1" else rank_colour(float(s["rank_pct"]))
                    d.rectangle([c * cw + cellgap / 2, y + 0.5, (c + 1) * cw - cellgap / 2, y + rh - 0.5], fill=col + (255,))
            y += rh
        geo[lg] = (y0 + top, y0 + y)
        y += gap
    return im, geo, cw


def bar(d, x0, y0, x1, y1, col, a=1.0, r=6):
    """a bar from baseline y0 up to y1 (y1 < y0), rounded at the data end"""
    if a <= 0 or abs(y0 - y1) < 1:
        return
    top, bot = min(y0, y1), max(y0, y1)
    d.rounded_rectangle([x0, top, x1, bot], r, fill=tuple(col) + (int(255 * a),))
    d.rectangle([x0, bot - r, x1, bot], fill=tuple(col) + (int(255 * a),))


def title(d, text, a, sub=None, y=64):
    d.text((W / 2, y), text, font=font(54, "Bold"), fill=INK + (int(255 * a),), anchor="ma")
    if sub:
        d.text((W / 2, y + 74), sub, font=font(28, "Regular"), fill=INK2 + (int(255 * a),), anchor="ma")


def cap(d, lines, a, y=None):
    """a caption panel: a list of strings (first line large) or of (text, size, colour, weight)"""
    L = [(x, 40 if i == 0 else 32, INK if i == 0 else INK2, "SemiBold" if i == 0 else "Regular") if isinstance(x, str) else x
         for i, x in enumerate(lines)]
    ST.caption(d, L, a, y=y)


class Section:
    name, dur = "section", 1.0

    def frame(self, t):
        raise NotImplementedError

    def render(self, still=None):
        os.makedirs(OUT, exist_ok=True)
        if still is not None:
            Image.fromarray(self.frame(still)).save(os.path.join(OUT, f"{self.name}_{still:05.1f}.png"))
            return
        CHUNK = 150
        cdir = os.path.join(OUT, "chunks")
        os.makedirs(cdir, exist_ok=True)
        n = int(round(self.dur * FPS))
        t0 = time.time()
        parts = []
        for c0 in range(0, n, CHUNK):
            c1 = min(n, c0 + CHUNK)
            part = os.path.join(cdir, f"{self.name}_{c0:05d}.mp4")
            parts.append(part)
            if os.path.exists(part + ".ok"):
                continue
            tmp = part + ".tmp.mp4"
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                  "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                                  "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
            for i in range(c0, c1):
                p.stdin.write(np.ascontiguousarray(self.frame(i / FPS)).tobytes())
            p.stdin.close()
            p.wait()
            os.replace(tmp, part)
            open(part + ".ok", "w").close()
            print(f"{self.name}: frames {c0}-{c1}/{n} {time.time() - t0:.0f}s", flush=True)
        lst = os.path.join(cdir, f"{self.name}.txt")
        open(lst, "w").write("".join(f"file '{q}'\n" for q in parts))
        path = os.path.join(OUT, f"{self.name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", path], check=True)
        for q in parts:
            os.remove(q)
            os.remove(q + ".ok")
        print("wrote", path, flush=True)
