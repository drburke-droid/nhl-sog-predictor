"""Course scenes in the poster look: sky, tree lines and fairway, drawn as flat layered shapes in late
afternoon light, under grain.

    Shot(...)        a camera down the fairway (perspective), and the ball's flight in it
    telephoto(...)   the broadcast camera's view: sky over a far tree line, the fairway's mowing stripes
"""
import math, functools
import numpy as np
import cv2
import look2 as S2
from look2 import W, H, K

SKY_HI = (58, 98, 150)
SKY_MID = (138, 170, 196)
SKY_LO = (236, 214, 184)
HAZE = (176, 182, 170)
TREE_DEEP = (17, 28, 21)
TREE_RIM = (104, 110, 58)
FAIR_1 = (96, 134, 58)
FAIR_2 = (80, 117, 49)
ROUGH_1 = (58, 88, 41)


def vgrad3(c0, c1, c2, y0, y1, y2):
    """sky: c0 at row y0, c1 at y1, c2 at y2 (units), smooth between"""
    y = np.arange(H, dtype=np.float32) / K
    out = np.zeros((H, 1, 3), np.float32)
    a = np.clip((y - y0) / (y1 - y0), 0, 1)[:, None]
    b = np.clip((y - y1) / (y2 - y1), 0, 1)[:, None]
    a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    c0, c1, c2 = (np.array(c, np.float32) for c in (c0, c1, c2))
    out[:, 0, :] = np.where((y < y1)[:, None], c0 * (1 - a) + c1 * a, c1 * (1 - b) + c2 * b)
    return out


@functools.lru_cache(None)
def noise(seed, sigma):
    rng = np.random.default_rng(seed)
    q = max(1, int(sigma // 3))
    n = rng.normal(0, 1, (H // q + 2, W // q + 2)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma * K / q)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) if q > 1 else n[:H, :W]
    return n / n.std()


def treeline(seed, base, top_lo, top_hi, density=1.0, conifer=0.3, x0=-120, x1=2040, scale=1.0):
    """a tree line's silhouette (mask 0..1) and its sunlit rim. Broadleaf crowns are clusters of leaf clumps;
    conifers are ragged tiers. The canopy is closed down to the base, so there are no gaps."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.uint8)
    put = lambda x, y, r: cv2.circle(m, (int(round(x * K * 16)), int(round(y * K * 16))), max(1, int(round(r * K * 16))),
                                     255, -1, cv2.LINE_AA, shift=4)
    x = x0
    while x < x1:
        h = rng.uniform(top_lo, top_hi)
        top = base - h
        if rng.random() < conifer:
            wdt = rng.uniform(0.30, 0.42) * h
            tiers = int(rng.integers(7, 12))
            pts_l, pts_r = [], []
            for k in range(tiers + 1):
                u = k / tiers
                yy = top + h * u
                ww = wdt / 2 * u ** 0.8 * rng.uniform(0.8, 1.15)
                pts_l.append((x - ww, yy + rng.uniform(0, 4) * scale))
                pts_r.append((x + ww, yy + rng.uniform(0, 4) * scale))
                if k < tiers:
                    u2 = (k + 0.6) / tiers
                    w2 = wdt / 2 * u2 ** 0.8 * rng.uniform(0.5, 0.7)
                    pts_l.append((x - w2, top + h * u2))
                    pts_r.append((x + w2, top + h * u2))
            poly = [(x, top)] + pts_r + [(x + wdt * 0.4, base + 30), (x - wdt * 0.4, base + 30)] + pts_l[::-1]
            cv2.fillPoly(m, [np.round(np.array(poly) * K * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            x += wdt * rng.uniform(0.35, 0.7) / density
        else:
            cw = h * rng.uniform(0.55, 0.95)                    # crown width
            ch = h * rng.uniform(0.55, 0.8)                     # crown height
            cy = top + ch / 2
            for _ in range(int(40 * scale + 30)):
                a = rng.uniform(0, 2 * math.pi)
                rr = math.sqrt(rng.uniform(0, 1))
                px_ = x + math.cos(a) * rr * cw / 2
                py_ = cy + math.sin(a) * rr * ch / 2
                r = rng.uniform(0.06, 0.15) * h * (1.15 - 0.5 * rr)
                put(px_, py_, r)
            cv2.rectangle(m, (int((x - cw * 0.38) * K), int(cy * K)), (int((x + cw * 0.38) * K), int((base + 30) * K)), 255, -1)
            x += cw * rng.uniform(0.45, 0.8) / density
    # close the canopy's lower part, so there is no sky between trunks
    cv2.rectangle(m, (0, int((base - top_lo * 0.45) * K)), (W, int((base + 30) * K)), 255, -1)
    mask = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 0.5 * K)
    # the sunlit rim (sun high on the left): inside the silhouette but not inside it shifted down-right
    sh = np.roll(np.roll(mask, int(6 * K * scale), 0), int(4 * K * scale), 1)
    rim = cv2.GaussianBlur(np.clip(mask - sh, 0, 1), (0, 0), 1.0 * K)
    return mask, rim * mask


def streak_clouds(fr, seed, y_lo=120, y_hi=420, n=5):
    """soft, flat-bottomed streaks of cloud lit warm from below"""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(100, 1820), rng.uniform(y_lo, y_hi)
        L, T = rng.uniform(200, 520), rng.uniform(10, 26)
        e = np.zeros((H, W), np.float32)
        cv2.ellipse(e, (int(cx * K), int(cy * K)), (int(L * K), int(T * K)), rng.uniform(-3, 3), 0, 360, 1.0, -1, cv2.LINE_AA)
        e = cv2.GaussianBlur(e, (0, 0), 9 * K)
        m = np.maximum(m, e * rng.uniform(0.35, 0.6))
    m *= np.clip(0.6 + 0.6 * noise(seed + 40, 14), 0, 1.2)
    col = np.array([246, 226, 206], np.float32)
    f = fr.astype(np.float32)
    return (f * (1 - m[..., None]) + col * m[..., None]).astype(np.uint8)


class Shot:
    """a camera behind the tee looking down the fairway: eye 1.6 m up, `back` metres behind the tee, focal
    length f (px of a 1080p frame), horizon row. Metres downrange z, sideways x, up y."""

    def __init__(self, f=2500.0, horizon=700.0, back=10.0, eye=1.6, cx=960.0):
        self.f, self.horizon, self.back, self.eye, self.cx = f, horizon, back, eye, cx

    def P(self, x, y, z):
        d = z + self.back
        return self.cx + self.f * x / d, self.horizon - self.f * (y - self.eye) / d

    def ground_y(self, z):
        return self.horizon + self.f * self.eye / (z + self.back)


def flight(t, carry=250.0, apex=30.0, start_x=1.6, end_x=-7.0):
    """the ball's position (x, y, z metres) at fraction t of its flight: it slows as it goes (z), climbs to
    its apex at about 55% of the carry and comes down more steeply (y), and curves gently (x)"""
    t = np.asarray(t, np.float64)
    z = carry * (1 - (1 - t) ** 1.45)
    s = z / carry
    g = lambda s_: s_ * (1 - s_) * (1 + 0.6 * s_)
    gs = np.linspace(0, 1, 401)
    y = apex * g(s) / g(gs).max()
    x = start_x + (end_x - start_x) * s ** 1.6
    return x, y, z


def course(shot, seed=1, trees_z=310.0, tree_h=(16, 28), back_trees=True, clouds=False, sky=None, light=1.0):
    """the scene for a camera down the fairway"""
    hz = shot.horizon
    c0, c1, c2 = sky or (SKY_HI, SKY_MID, SKY_LO)
    glow_y = shot.ground_y(trees_z) - tree_h[1] * shot.f / (trees_z + shot.back) * 0.9   # warm light just over the trees
    fr = np.broadcast_to(vgrad3(c0, c1, c2, 0, glow_y * 0.62, glow_y + 30), (H, W, 3)).astype(np.float32)
    fr = fr * (1 + 0.012 * noise(seed + 20, 40))[..., None]
    fr = np.clip(fr, 0, 255).astype(np.uint8)
    if clouds:
        fr = streak_clouds(fr, seed + 30, y_lo=hz * 0.15, y_hi=hz * 0.62)
    base = shot.ground_y(trees_z)
    sc = shot.f / (trees_z + shot.back)                         # px per metre at the trees
    if back_trees:
        mb, _ = treeline(seed + 5, shot.ground_y(trees_z + 60) - 2, tree_h[0] * sc * 0.75, tree_h[1] * sc * 0.85,
                         density=1.6, conifer=0.4, scale=0.8)
        hazec = (np.array(TREE_DEEP, np.float32) * 0.45 + np.array(c2, np.float32) * 0.55).astype(int)
        fr = S2.over(fr, mb, tuple(hazec))
    mf, rf = treeline(seed, base, tree_h[0] * sc, tree_h[1] * sc, density=1.0, conifer=0.3)
    fr = S2.over(fr, mf, TREE_DEEP)
    fr = S2.over(fr, rf, TREE_RIM, 0.55 * light)
    f = fr.astype(np.float32)
    f *= (1 + 0.10 * noise(seed + 3, 1.6) * mf + 0.06 * noise(seed + 4, 6) * mf)[..., None]
    fr = np.clip(f, 0, 255).astype(np.uint8)
    # the ground: rough under the trees, then the fairway's 10-yard mowing stripes
    g0 = int(math.floor(base * K))
    fr[g0:] = ROUGH_1
    edge = shot.ground_y(trees_z - 12)
    zs = np.arange(0, trees_z - 12, 9.144)
    for k, z0 in enumerate(zs):
        ya, yb = shot.ground_y(min(z0 + 9.144, trees_z - 12)), shot.ground_y(z0)
        if yb < edge:
            continue
        cv2.rectangle(fr, (0, int(round(ya * K))), (W, int(round(yb * K))), (FAIR_1, FAIR_2)[k % 2], -1)
    yy = np.arange(H, dtype=np.float32)[:, None] / K
    shade = np.clip((yy - edge) / 40, 0, 1)
    f = fr.astype(np.float32)
    f[g0:] *= (0.94 + 0.06 * shade[g0:])[..., None] * (1 + 0.035 * noise(seed + 7, 3))[g0:, :, None]
    # a long shadow from the trees across the far fairway
    sh = np.clip((edge + 26 - yy) / 26, 0, 1) * (yy > base)
    f *= (1 - 0.28 * sh)[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def tracer(fr, shot, t0, t1, carry=250.0, apex=30.0, width=5.0, a=1.0, glow=True):
    """the broadcast shot tracer from flight fraction t0 to t1"""
    ts = np.linspace(t0, t1, 300)
    x, y, z = flight(ts, carry, apex)
    pts = np.array([shot.P(xx, yy, zz) for xx, yy, zz in zip(x, y, z)])
    pts = pts[(pts[:, 1] < 1100)]
    if len(pts) < 2:
        return fr, None
    if glow:
        g = np.zeros(fr.shape[:2], np.float32)
        cv2.polylines(g, [np.round(pts * K * 16).astype(np.int32)], False, 1.0, int(width * 3 * K), cv2.LINE_AA, shift=4)
        g = cv2.GaussianBlur(g, (0, 0), 6 * K)
        fr = np.clip(fr.astype(np.float32) + g[..., None] * np.array([110, 40, 20], np.float32) * a, 0, 255).astype(np.uint8)
    fr = S2.marker(fr, pts, S2.TRACER, width=width, wobble=0.0, a=a)
    return fr, pts[-1]
