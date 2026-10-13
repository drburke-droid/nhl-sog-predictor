"""One blink vs. light driving every street in Calgary, side by side.

    python blink.py            -> out/blink_vs_calgary.mp4 and out/blink_vs_calgary.gif
    python blink.py still 5    -> PNG at 5 s

The face is the 3D model from the neurogaze gaze simulator (drburke-droid/neurogaze), captured
headless at 41 lid closures with neurogaze_capture.mjs into data/eye_frames/.
Timing: a blink ~300 ms (lid closes in ~100 ms, stays shut ~50 ms, reopens over ~150 ms).
Light covers every Calgary street (with backtracking) in stats["seconds_at_c"] (~37 ms),
so it laps the city ~8 times per blink. Shown once at real speed, then 30x slower.
"""
import os, sys, json, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import city

OUT = city.OUT
CW, CH, FPS = 1200, 800, 30
BLINK = (0.100, 0.150, 0.300)          # closed-by, open-from, open-by (seconds)
SLOW = 30.0
PANEL_Y0, PANEL_Y1 = 130, 690
FD = "/usr/share/fonts/opentype/inter"
BG = np.array([4, 7, 13], np.float32) / 255
CYAN = np.array([0.14, 0.66, 1.00], np.float32)
GOLD = np.array([1.00, 0.56, 0.13], np.float32)
HOT = np.array([1.00, 0.93, 0.78], np.float32)


def font(n, s):
    return ImageFont.truetype(os.path.join(FD, n), s)


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def lid(tau):
    """0 = open, 1 = closed, for real time tau (s) since the blink began."""
    c1, o0, o1 = BLINK
    if tau <= 0 or tau >= o1:
        return 0.0
    if tau < c1:
        return ease(tau / c1)
    if tau < o0:
        return 1.0
    return 1 - ease((tau - o0) / (o1 - o0))


# ------------------------------------------------------------------ Calgary layers
def calgary_layers():
    G = city.build()
    rn, re_, cum, stats = city.walk(G)
    nx, ny, ea, eb = G["nx"], G["ny"], G["ea"], G["eb"]
    bx0, by0, bx1, by1 = G["boundary"].bounds
    (x0, x1), (y0, y1) = [city.xy([bx0, bx1], [by0, by1])[0], city.xy([bx0, bx1], [by0, by1])[1]]
    MH = PANEL_Y1 - PANEL_Y0
    sc = MH / (y1 - y0)
    MW = int(np.ceil((x1 - x0) * sc))
    S = 2
    px = lambda X, Y: (np.c_[(X - x0) * sc, (y1 - Y) * sc] * S * 8).round().astype(np.int32)
    frac = cum / cum[-1]
    fprev = np.r_[0.0, frac[:-1]]
    tv = np.full(len(ea), np.inf)
    st = np.nonzero(re_ >= 0)[0][::-1]
    tv[re_[st]] = 0.5 * (fprev[st] + frac[st])
    P = np.stack([px(nx[ea], ny[ea]), px(nx[eb], ny[eb])], 1)
    I = np.zeros((MH * S, MW * S), np.uint8)
    for c in sorted(city.STREETS, key=lambda c: city.WEIGHT[c][0]):
        sel = np.nonzero(G["ecls"] == c)[0]
        if len(sel):
            cv2.polylines(I, list(P[sel]), False, int(255 * city.WEIGHT[c][0]), max(1, city.WEIGHT[c][1] - 1), cv2.LINE_AA, shift=3)
    vt = np.full((MH * S, MW * S), np.inf, np.float32)
    fb = np.floor(tv * 400).astype(np.int64)
    order = np.argsort(-fb, kind="stable")
    bounds = np.r_[0, np.nonzero(np.diff(fb[order]))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = order[a:b]
        cv2.polylines(vt, list(P[sel]), False, float(tv[sel].min()), 2, cv2.LINE_8, shift=3)
    idx = ndimage.distance_transform_edt(~np.isfinite(vt), return_distances=False, return_indices=True)
    vt = vt[idx[0], idx[1]]
    outline = np.zeros((MH * S, MW * S), np.uint8)
    bd = G["boundary"]
    for poly in getattr(bd, "geoms", [bd]):
        bx, byy = city.xy(*np.asarray(poly.exterior.coords).T)
        cv2.polylines(outline, [px(bx, byy)], True, 255, 2, cv2.LINE_AA, shift=3)
    return dict(I=I.astype(np.float32) / 255, vt=vt, outline=cv2.resize(outline, (MW, MH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255,
                MW=MW, MH=MH, lap_s=stats["seconds_at_c"], stats=stats)


def calgary_panel(C, tau):
    """tau = real seconds since the blink began; laps run only during the blink."""
    lap = C["lap_s"]
    t = float(np.clip(tau, 0, BLINK[2]))
    k = int(t // lap); u = (t % lap) / lap
    I, vt = C["I"], C["vt"]
    if tau <= 0:
        gold = np.zeros_like(I); flash = np.zeros_like(I)
    elif k == 0:
        gold = (vt <= u).astype(np.float32)
        age = u - vt
        flash = np.where((age >= 0) & (age < 0.18), (1 - age / 0.18) ** 2, 0).astype(np.float32)
    else:
        gold = np.ones_like(I)
        age = u - vt
        flash = np.where((age >= 0) & (age < 0.18), (1 - age / 0.18) ** 2, 0).astype(np.float32)
        if tau >= BLINK[2]:
            flash[:] = 0
    MW, MH = C["MW"], C["MH"]
    Ig = cv2.resize(I * gold, (MW, MH), interpolation=cv2.INTER_AREA)
    Ic = cv2.resize(I * (1 - gold), (MW, MH), interpolation=cv2.INTER_AREA)
    Fl = cv2.resize(I * flash, (MW, MH), interpolation=cv2.INTER_AREA)
    E = Ic[..., None] * CYAN * 0.6 + Ig[..., None] * GOLD * 0.95 + Fl[..., None] * HOT * 1.6 \
        + C["outline"][..., None] * np.array([0.08, 0.14, 0.22], np.float32)
    laps = 0.0 if tau <= 0 else min(tau, BLINK[2]) / lap
    return E, laps


# ------------------------------------------------------------------ the eye (glowing line art)
def eye_panel(c, w=520, h=560):
    S = 2
    W2, H2 = w * S, h * S
    cx, cy = W2 / 2, H2 / 2 + 10 * S
    ew, eh_u, eh_l = 210 * S, 105 * S, 62 * S
    xs = np.linspace(-ew, ew, 160)
    s = np.sqrt(np.clip(1 - (xs / ew) ** 2, 0, 1))
    up = cy - eh_u * s ** 0.9
    lo = cy + eh_l * s ** 0.9
    lid_y = up * (1 - c) + (lo - 2 * S) * c
    pts = lambda ys: np.c_[cx + xs, ys]
    sclera = np.zeros((H2, W2), np.uint8)
    cv2.fillPoly(sclera, [np.round(np.r_[pts(lid_y), pts(lo)[::-1]] * 8).astype(np.int32)], 255, cv2.LINE_AA, shift=3)
    vis = sclera.astype(np.float32) / 255
    E = np.zeros((H2, W2, 3), np.float32)
    E += vis[..., None] * np.array([0.30, 0.34, 0.40], np.float32)
    # iris + pupil, clipped to the visible eye
    iris = np.zeros((H2, W2), np.uint8)
    r_i = 64 * S
    cv2.circle(iris, (int(cx * 8), int((cy - 6 * S) * 8)), int(r_i * 8), 255, -1, cv2.LINE_AA, shift=3)
    yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
    rr = np.hypot(xx - cx, yy - (cy - 6 * S)) / r_i
    ang = np.arctan2(yy - (cy - 6 * S), xx - cx)
    tex = 0.75 + 0.25 * np.sin(ang * 38) * np.clip(rr, 0, 1)
    irisf = iris.astype(np.float32) / 255 * vis
    col = (np.array([0.10, 0.45, 0.70], np.float32) * (1 - rr[..., None] * 0.5) + np.array([0.55, 0.40, 0.12], np.float32) * np.clip(0.6 - rr, 0, 1)[..., None])
    E = E * (1 - irisf[..., None]) + col * tex[..., None] * irisf[..., None] * 1.4
    pupil = np.zeros((H2, W2), np.uint8)
    cv2.circle(pupil, (int(cx * 8), int((cy - 6 * S) * 8)), int(26 * S * 8), 255, -1, cv2.LINE_AA, shift=3)
    pf = pupil.astype(np.float32) / 255 * vis
    E = E * (1 - pf[..., None])
    hl = np.zeros((H2, W2), np.uint8)
    cv2.circle(hl, (int((cx + 20 * S) * 8), int((cy - 28 * S) * 8)), int(10 * S * 8), 255, -1, cv2.LINE_AA, shift=3)
    E += (hl.astype(np.float32) / 255 * vis)[..., None] * 2.0
    # lid edge, lower lid, crease and lashes (glowing lines)
    lines = np.zeros((H2, W2), np.uint8)
    cv2.polylines(lines, [np.round(pts(lid_y) * 8).astype(np.int32)], False, 255, 3 * S, cv2.LINE_AA, shift=3)
    cv2.polylines(lines, [np.round(pts(lo) * 8).astype(np.int32)], False, 170, 2 * S, cv2.LINE_AA, shift=3)
    crease = (up - 34 * S * s) * (1 - c) + (lid_y - 22 * S * s) * c      # the crease drops with the lid
    cv2.polylines(lines, [np.round(pts(crease) * 8).astype(np.int32)], False, 90, 2 * S, cv2.LINE_AA, shift=3)
    for i in range(12, 148, 11):
        x0, y0 = cx + xs[i], lid_y[i]
        tilt = (xs[i] / ew) * 0.9
        dy = -(22 * S) * (1 - c) + (14 * S) * c
        cv2.line(lines, (int(x0 * 8), int(y0 * 8)), (int((x0 + tilt * 16 * S) * 8), int((y0 + dy) * 8)), 220, 2 * S, cv2.LINE_AA, shift=3)
    E += (lines.astype(np.float32) / 255)[..., None] * np.array([0.95, 0.80, 0.55], np.float32) * 0.9
    return cv2.resize(E, (w, h), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ 3D face frames (neurogaze model)
EYE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eye_frames")
_EYE = {}


def face_frame(c):
    """Blend the two captured lid positions around closure c (0 open .. 1 closed)."""
    if not _EYE:
        files = sorted(f for f in os.listdir(EYE_DIR) if f.endswith(".png"))
        _EYE["f"] = [cv2.imread(os.path.join(EYE_DIR, f))[..., ::-1].astype(np.float32) / 255 for f in files]
        h, w = _EYE["f"][0].shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        _EYE["mask"] = (np.clip(edge / 36.0, 0, 1) ** 1.5)[..., None]           # soft rectangular feather
    fr = _EYE["f"]
    x = float(np.clip(c, 0, 1)) * (len(fr) - 1)
    i0 = int(np.floor(x)); i1 = min(i0 + 1, len(fr) - 1); u = x - i0
    return fr[i0] * (1 - u) + fr[i1] * u, _EYE["mask"]


# ------------------------------------------------------------------ compose
def compose(C, tau, mode, card=None):
    img = np.zeros((CH, CW, 3), np.float32) + BG
    ex = 40
    E, laps = calgary_panel(C, tau)
    mx = 620 + (540 - C["MW"]) // 2
    img[PANEL_Y0:PANEL_Y0 + C["MH"], mx:mx + C["MW"]] += E
    small = cv2.resize(img, (CW // 2, CH // 2), interpolation=cv2.INTER_AREA)
    bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3) * 0.5 + cv2.GaussianBlur(small, (0, 0), 10) * 0.4, (CW, CH))
    img = 1 - np.exp(-(img + bloom) * 1.1)
    face, m = face_frame(lid(tau))
    fh, fw = face.shape[:2]
    fy = PANEL_Y0 + ((PANEL_Y1 - PANEL_Y0) - fh) // 2
    fx = 30
    reg = img[fy:fy + fh, fx:fx + fw]
    img[fy:fy + fh, fx:fx + fw] = reg * (1 - m) + face * m
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    d.text((CW // 2, 28), "One blink vs. light driving every street in Calgary", font=font("Inter-SemiBold.otf", 38), fill=(236, 242, 250), anchor="ma")
    d.text((CW // 2, 80), mode, font=font("Inter-Bold.otf", 24), fill=(255, 186, 80), anchor="ma")
    ms = max(0.0, min(tau, BLINK[2])) * 1000
    d.text((300, 705), "One blink", font=font("Inter-SemiBold.otf", 30), fill=(236, 242, 250), anchor="ma")
    d.text((300, 745), f"{ms:5.0f} ms of ~300 ms", font=font("Inter-Regular.otf", 24), fill=(160, 178, 200), anchor="ma")
    d.text((890, 705), "Light, every street in Calgary", font=font("Inter-SemiBold.otf", 30), fill=(236, 242, 250), anchor="ma")
    d.text((890, 745), f"laps of the city: {laps:.1f}   ({C['lap_s'] * 1000:.0f} ms each)", font=font("Inter-Regular.otf", 24), fill=(160, 178, 200), anchor="ma")
    if card:
        a = int(255 * card[0])
        ov = Image.new("RGBA", (CW, CH), (4, 7, 13, int(0.72 * a))); od = ImageDraw.Draw(ov)
        for i, (txt, sz, col, bold) in enumerate(card[1]):
            od.text((CW // 2, 300 + i * 74), txt, font=font("Inter-Bold.otf" if bold else "Inter-Medium.otf", sz), fill=col + (a,), anchor="ma")
        im.paste(ov, (0, 0), ov)
    return np.asarray(im)


def timeline():
    fr = []
    # A: real speed (blink starts at 0.6 s)
    for f in range(int(1.6 * FPS)):
        t = f / FPS
        fr.append((t - 0.6, "REAL SPEED", None))
    # card: now slower
    for f in range(int(1.4 * FPS)):
        a = min(1.0, f / 6) * min(1.0, (1.4 * FPS - f) / 6)
        fr.append((BLINK[2] + 0.01, "REAL SPEED", (a, [("Too fast to see.", 54, (236, 242, 250), True),
                                                    ("Now 30× slower.", 44, (255, 186, 80), True)])))
    # B: 30x slower, with a short hold before and after
    n = int((BLINK[2] * SLOW + 1.2) * FPS)
    for f in range(n):
        tau = (f / FPS - 0.4) / SLOW
        fr.append((tau, "30× SLOWER", None))
    # C: end card
    for f in range(int(3.0 * FPS)):
        a = min(1.0, f / 8)
        fr.append((BLINK[2] + 0.01, "30× SLOWER", (a, [("In one blink, light could drive", 46, (236, 242, 250), True),
                                                         ("every street in Calgary 8 times.", 46, (255, 186, 80), True),
                                                         ("Blink ~300 ms · Calgary at light speed: 37 ms", 28, (160, 178, 200), False)])))
    return fr


def main():
    C = calgary_layers()
    os.makedirs(OUT, exist_ok=True)
    mp4 = os.path.join(OUT, "blink_vs_calgary.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{CW}x{CH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    for tau, mode, card in timeline():
        p.stdin.write(compose(C, tau, mode, card).tobytes())
    p.stdin.close(); p.wait()
    gif = os.path.join(OUT, "blink_vs_calgary.gif")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf",
                    "fps=15,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                    "-loop", "0", gif], check=True)
    print("wrote", mp4, gif, "laps per blink", BLINK[2] / C["lap_s"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        C = calgary_layers()
        fr = timeline()
        for s in sys.argv[2:]:
            i = min(int(float(s) * FPS), len(fr) - 1)
            tau, mode, card = fr[i]
            cv2.imwrite(os.path.join(OUT, f"blink_{s}.png"), compose(C, tau, mode, card)[..., ::-1])
            print("still", s)
    else:
        main()
