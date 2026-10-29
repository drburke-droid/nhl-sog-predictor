"""Composite the Parker finale.
  python parker_comp.py -> /home/user/lot/out/frames_parker/f####.png, and /home/user/lot/out/fly_parker.mp4

Real time: the empty road for two seconds; the frame in which the probe arrives is nothing but white (it crosses in a
fifth of a millisecond and lights everything); the camera is gone before the next one (the blast reaches it 1.8 ms
later): no signal. Then the replay, 20,000 times slower (500,000 frames a second): the probe meeting the air at the left,
its heat shield white-hot, boiling away within about 5 m into a bolt of plasma that crosses the picture leaving a
column of air at tens of thousands of kelvin, widening as the blast does (parker.py); the ground erupting where the
blast reaches it, a wall of dust and vapour racing outward lit by the fireball, toward the camera; the light so intense
that the camera's exposure falls a hundredfold and the daylit scene sinks to a silhouette. It ends when the blast
reaches the lens."""
import numpy as np, math, os, sys, cv2, subprocess
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as Cam, comp as C, dust as D, flight as F, fly_fx as FX, parker as P
from scipy.ndimage import gaussian_filter

W, H = Cam.W, Cam.H_IMG
R_ = "/home/user/lot/render/parker"
OUT = "/home/user/lot/out"
FR = f"{OUT}/frames_parker"
WHITE, DIM = (255, 255, 255), (214, 220, 228)
P0 = 40                                   # plate frame the replay freezes on

# per-pixel geometry: closest approach of each ray to the path, where along it, and the ground point it sees
DMIN, XS, GX, GY = FX.geometry(P.PZ)
R0 = D.Renderer(D.Grid(h=1.0), scale=1.0)
dirs, org = R0.dirs, R0.org
tg = np.where(dirs[..., 2] < -1e-6, -org[2] / np.minimum(dirs[..., 2], -1e-6), np.inf)
GROUND = np.isfinite(tg)
GP = org + np.where(GROUND, tg, 0)[..., None] * dirs
FWD = Cam.R()[2]
COLDEPTH = ((np.stack([XS, np.full_like(XS, P.PY), np.full_like(XS, P.PZ)], -1) - P.CAM_P) @ FWD).astype(np.float32)

_BB = [(1000, (1.0, 0.18, 0.0)), (1500, (1.0, 0.39, 0.06)), (2000, (1.0, 0.54, 0.16)), (3000, (1.0, 0.71, 0.42)),
       (4000, (1.0, 0.82, 0.64)), (5500, (1.0, 0.94, 0.88)), (6500, (1.0, 1.0, 1.0)), (10000, (0.79, 0.86, 1.0)),
       (20000, (0.67, 0.77, 1.0)), (40000, (0.62, 0.73, 1.0))]


def bb(T):
    """blackbody colour (linear, normalised) at T K"""
    T = np.clip(np.asarray(T, float), 1000, 40000)
    lt = np.log(T); ks = np.log([k for k, _ in _BB]); cs = np.array([c for _, c in _BB])
    return np.stack([np.interp(lt, ks, cs[:, i]) for i in range(3)], -1)


def vis(T):
    """visible radiance of a blackbody at T K, against a sunlit white surface (~0.8): 70 at 2,300 K, climbing as
    exp(-24,000 K / T) below 6,000 K and roughly as T above"""
    T = np.asarray(T, float)
    lo = 60.0 * np.exp(24000 / 2300 - 24000 / np.minimum(T, 6000.0))
    return np.where(T > 6000, 60.0 * math.exp(24000 / 2300 - 4.0) * T / 6000.0, lo)


def core(x, t):
    """radius of the luminous region: everything inside the blast front (shocked to thousands of kelvin); behind the
    point where the probe met the air the blast closes in a rounded end"""
    x = np.asarray(x, float)
    R = P.shock_r(np.maximum(x, P.X_IN), t)
    behind = np.maximum(P.X_IN - x, 0)
    return 0.08 + np.sqrt(np.maximum(R ** 2 - behind ** 2, 0))


def lum_temperature(a):
    """K of the glowing air a seconds after the plasma passed: tens of thousands at first; behind a front still moving
    at Mach 18 when it reaches the camera, shocked air stays at 6,000-8,000 K (ionisation holds it there)"""
    return np.clip(P.temperature(a), 5500.0, 40000.0)


def head(t):
    """(x, radius) of the plasma bolt: the probe's vapour, its cloud swelling as the probe boils away"""
    return float(P.x_head(t)), 0.2 + 0.8 * min(t / P.T_GONE, 1.0)


T_HEAD = 30000.0
_mottle = None


def column(t):
    """radiance of the blast's glowing air (optically thick: as bright as its temperature, brightest toward its
    edge where the line of sight runs longest through the shocked shell, mottled) and the plasma bolt's head"""
    global _mottle
    if _mottle is None:
        rng = np.random.default_rng(12)
        _mottle = []
        for _ in range(6):                                  # turbulent brightness: smooth, several scales
            m = 0
            for cells, amp in ((48, 0.5), (20, 0.3), (8, 0.2)):
                m = m + amp * cv2.GaussianBlur(cv2.resize(rng.random((H // cells + 2, W // cells + 2)).astype(np.float32),
                                                          (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), cells / 3)
            _mottle.append(((m - m.mean()) / (m.std() + 1e-6)).astype(np.float32))
    a = t - P.t_at(np.maximum(XS, P.X_IN))
    on = (a > 0) & (core(XS, t) > 0.081)
    img = np.zeros((H, W, 3), np.float32)
    if on.any():
        aa = np.where(on, a, 1e-3)
        Rr = core(XS, t)
        q = DMIN / Rr
        shell = np.clip(1 - q ** 20, 0, 1) * (0.8 + 0.2 * np.clip(q, 0, 1) ** 4)
        ph = t / 40e-6
        k0 = int(ph) % 6; f = ph % 1
        mot = _mottle[k0] * (1 - f) + _mottle[(k0 + 1) % 6] * f
        T = lum_temperature(aa)
        L = vis(T) * shell * np.clip(1 + 0.12 * mot, 0.6, 1.4)
        img += (np.where(on, L, 0)[..., None] * bb(T)).astype(np.float32)
    if t > 0:
        xh, rh = head(t)
        uv, z = Cam.project([[xh, P.PY, P.PZ]])
        if z[0] > 0:
            r_px = Cam.F * rh / z[0]
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            dx = xx - uv[0, 0]
            trail = 3.0 if t > P.T_GONE else 0.25 + 2.75 * t / P.T_GONE        # a cap on the heat shield, then a bolt
            sx = np.where(dx < 0, trail * r_px, r_px)
            d2 = (dx / sx) ** 2 + ((yy - uv[0, 1]) / r_px) ** 2
            img += (float(vis(T_HEAD)) * np.exp(-0.5 * d2 ** 2))[..., None] * bb(T_HEAD)
    return img


def lights(t, n=28):
    """the column and the bolt as point lights [((x, y, z), rgb)], in the daylight's units (sun 1.3 W/m^2): a stretch
    of optically thick column of radius r and radiance L shines with intensity L * 2r * length"""
    out = []
    if t <= 0: return out
    xh, rh = head(t)
    xs = np.linspace(P.X_IN, min(xh, 40.0), n)
    dx = xs[1] - xs[0]
    for x in xs:
        a = t - float(P.t_at(x))
        if a <= 0: continue
        T = float(lum_temperature(a))
        c = bb(T) * float(vis(T)) * 2 * float(core(x, t)) * dx
        out.append(((float(x), P.PY, P.PZ), tuple(float(v) for v in c)))
    c = bb(T_HEAD) * float(vis(T_HEAD)) * math.pi * rh ** 2 * 2
    out.append(((xh, P.PY, P.PZ), tuple(float(v) for v in c)))
    return out


def ground_light(ls):
    """irradiance on the ground the plate shows, from point lights, relative to the daylight's (0.8)"""
    E = np.zeros((H, W, 3), np.float32)
    for (lx, ly, lz), c in ls:
        L = np.array([lx, ly, lz], np.float32) - GP
        d2 = (L ** 2).sum(-1) + 0.05
        cos = np.clip(L[..., 2] / np.sqrt(d2), 0, 1)
        E += (cos / d2)[..., None] * np.array(c, np.float32)
    return np.where(GROUND[..., None], E / 0.8, 0)


def exposure(img):
    """the camera's exposure, metered on the whole picture: it holds the brightest part just below white, so the
    daylit scene sinks as the fireball grows"""
    m = float(np.percentile(img.max(-1), 99.5))
    return min(1.0, 1.1 / max(m, 1e-9))


def displacement(t):
    """the blast's front bending light where rays graze it"""
    Rr = (core(XS, t) - 0.08).astype(np.float32)
    on = Rr > 0.2
    w = 0.08 + 0.03 * Rr
    q = (DMIN - Rr) / w
    disp = np.where(on, 14.0 * (-q * np.exp(0.5 - 0.5 * q * q)), 0).astype(np.float32)
    return disp * GX, disp * GY


# ---------------------------------------------------------------- the ground erupting: a wall of dust and vapour
VIS = D.Grid(x0=-16, x1=17, y0=1.0, y1=36.0, z1=10, h=0.15)
REN = D.Renderer(VIS, scale=0.5, kappa=1.6, albedo=(0.4, 0.3, 0.22), sun_irr=1.3, sun_col=(1.0, 0.78, 0.58))
_noise = None


def wall_density(t):
    global _noise
    X, Y, Z = VIS.coords()
    if _noise is None: _noise = D.noise3(VIS.n, 6, 31, 3).astype(np.float32)
    Rr = core(X, t) - 0.08
    rg = np.sqrt(np.maximum(Rr ** 2 - P.PZ ** 2, 0))                 # how far out along the ground the blast is
    rho = np.abs(Y - P.PY)
    front = rg * 0.97
    w = 0.2 + 0.05 * rg
    h = 0.4 + 0.12 * rg
    shell = np.exp(-((rho - front) / w) ** 2) * (Z < h * (0.5 + 1.0 * _noise)) * (rho < rg * 1.02)
    inner = 0.0 * Z
    vapour = np.hypot(Y - P.PY, Z - P.PZ) < 0.5 * Rr                       # the hottest inner core: no dust left
    return (30.0 * (shell + inner) * (0.4 + _noise) * (rg > 0) * ~vapour).astype(np.float32)


def wall(t, ls):
    if P.shock_r(np.array([0.0]), t)[0] < P.PZ: return None
    VIS.d[:] = gaussian_filter(wall_density(t), 0.8)
    if VIS.d.max() < 1e-3: return None
    # a light stands for a stretch of column: soften it by that stretch and the column's radius
    soft = [((x, y, z), c, max(2.0, float(core(np.array([x]), t)[0]))) for (x, y, z), c in ls]
    rgb, tr, gs = D.render_l(REN, soft)
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
    return up(rgb), up(tr)[..., None]


def replay_frame(job):
    n, t, slow = job
    path = f"{FR}/r{n:04d}.png"
    if os.path.exists(path): return
    ls = lights(t)
    plate = C.plate_loop(P0)
    plate = plate * (1 + ground_light(ls))
    pp = f"{R_}/p{n:04d}.npz"
    p = C.props(pp if os.path.exists(pp) else f"{R_}/static.npz")
    img = C.over_props(plate, C.plate_match(p, 7000 + n))
    if ls and p is not None: img = img + C.relight(p, ls)
    col = column(t)
    if p is not None:                                                    # what stands in front of the glowing air
        Rr = core(XS, t)
        entry = COLDEPTH - np.sqrt(np.maximum(Rr ** 2 - DMIN ** 2, 0))
        hide = (p["a"][..., 0] > 0.5) & (p["depth"] < entry)
        col = col * (~hide)[..., None]
    img = img + col
    wl = wall(t, ls)
    if wl is not None: img = img * wl[1] + wl[0]
    img = FX.warp(img, displacement(t))
    g = exposure(img)
    img = img * g
    img = img / (1 + img / 24.0)                                        # what the sensor can hold, before the bloom
    # the last frames: the blast at the lens
    k = np.clip((t - (P.T_CAMERA - 0.12e-3)) / 0.12e-3, 0, 1)
    img = img * (1 - k) + 12.0 * k
    lines = [("430,000 mph", 64, "Bold", WHITE), ("Parker Solar Probe, 2024", 30, "Medium", DIM),
             ("its top speed, here at sea level", 24, "Medium", DIM),
             (f"replay, {1 / slow:,.0f}× slower  ·  t = {t * 1e6:,.0f} µs" if t < 1e-3 else
              f"replay, {1 / slow:,.0f}× slower  ·  t = {t * 1e3:.2f} ms", 24, "Medium", DIM)]
    img = C.bloom(img, thresh=0.8, sigma=(6, 24, 80), gain=(0.6, 0.45, 0.35))
    img = C.grain(img, 9000 + n)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    out = C.label(out, lines, 1.0 - k, corner="bl")
    cv2.imwrite(path, out[..., ::-1])


def lost(seconds=1.6):
    """no signal: black, with the camera's last frame breaking up for a moment"""
    frames = []
    for i in range(int(seconds * F.FPS)):
        out = np.zeros((C.H, C.W, 3), np.uint8)
        if i < 3:
            rng = np.random.default_rng(i)
            noise = (rng.random((C.PH // 8, C.W // 8)) * 255 * (0.6 - 0.2 * i)).astype(np.uint8)
            out[C.TOP:C.TOP + C.PH] = cv2.resize(noise, (C.W, C.PH), interpolation=cv2.INTER_NEAREST)[..., None]
        frames.append(out)
    return frames


def realtime():
    """two seconds of the road; then the frame the probe arrives in, all white; then nothing"""
    n0 = round(F.MACHINES["parker"]["pre"] * F.FPS)
    st = C.props(f"{R_}/static.npz")
    out = []
    lines = [("430,000 mph", 64, "Bold", WHITE), ("Parker Solar Probe, 2024", 30, "Medium", DIM),
             ("its top speed, here at sea level", 24, "Medium", DIM)]
    for n in range(n0):
        img = C.over_props(C.plate_loop(n), C.plate_match(st, 6000 + n))
        img = C.grain(C.bloom(img), 6000 + n)
        pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
        o = np.zeros((C.H, C.W, 3), np.uint8); o[C.TOP:C.TOP + C.PH] = pic
        out.append(C.label(o, lines, min(1.0, n / 8), corner="bl"))
    o = np.zeros((C.H, C.W, 3), np.uint8); o[C.TOP:C.TOP + C.PH] = 255
    out.append(o)
    return out + lost(1.4) + card("The camera's last 1.7 milliseconds", "replayed 20,000 times slower", 2.6)


def card(text, sub, seconds, fade=0.5):
    frames = []
    nf = int(seconds * F.FPS)
    for i in range(nf):
        a = min(1.0, i / (fade * F.FPS), (nf - 1 - i) / (fade * F.FPS))
        im = np.zeros((C.H, C.W, 3), np.uint8)
        from PIL import Image, ImageDraw
        pil = Image.fromarray(im); d = ImageDraw.Draw(pil)
        for txt, size, weight, y, col in ((text, 58, "Bold", C.H // 2 - 70, WHITE), (sub, 34, "Medium", C.H // 2 + 20, DIM)):
            f = C.font(size, weight)
            w = d.textlength(txt, font=f)
            d.text(((C.W - w) / 2, y), txt, font=f, fill=tuple(int(c * a) for c in col))
        frames.append(np.asarray(pil))
    return frames


def sound(n_rt, n_lost, n_replay, n_after):
    """real time: wind, then at the white frame a crack too loud to record, cut off as the camera dies; silence. The
    replay: a deep swell and the crackle of the blast as it comes on, cut off at the lens; silence after"""
    import fly_sound as S
    sr = S.SR
    rng = np.random.default_rng(21)
    N = int((n_rt + n_lost + n_replay + n_after) / F.FPS * sr)
    out = np.zeros((N, 2))
    i_flash = int(n_rt / F.FPS * sr)
    out[:i_flash] += S.wind(i_flash, rng, 0.035)
    m = int(0.03 * sr); tt = np.arange(m) / sr
    crack = np.clip(3.0 * (1 - tt / 0.03) * (1 + 0.6 * rng.standard_normal(m)), -1, 1)
    out[i_flash:i_flash + m] += crack[:, None]
    i0 = int((n_rt + n_lost) / F.FPS * sr); i1 = i0 + int(n_replay / F.FPS * sr)
    tl = P.replay()
    T = np.arange(i1 - i0) / sr
    tk = np.interp(T * F.FPS, np.arange(len(tl)), [t for t, _ in tl])
    r = P.shock_r(np.zeros_like(tk), tk) / 20.3
    swell = np.clip(r, 0, 1) ** 2
    hum = S.filt(rng.standard_normal((len(T), 2)), "lowpass", 90, 2) * 3.0
    crk = S.filt(rng.standard_normal((len(T), 2)) * (rng.random((len(T), 2)) < 0.01), "bandpass", [300, 3000], 2) * 4.0
    drone = 0.25 * np.sin(2 * math.pi * np.cumsum(28 + 20 * swell) / sr)
    rep = (hum * (0.15 + 0.85 * swell[:, None]) + crk * swell[:, None] + drone[:, None] * (0.3 + swell[:, None]))
    rep *= np.minimum(1, T / 0.3)[:, None]
    out[i0:i1] += 0.3 * rep
    return np.tanh(out * 1.2) / 1.2


if __name__ == "__main__":
    os.makedirs(FR, exist_ok=True)
    tl = P.replay()
    which = sys.argv[1:] or ["all"]
    jobs = [(n, t, s) for n, (t, s) in enumerate(tl)]
    if which != ["all"]: jobs = [j for j in jobs if str(j[0]) in which]
    with Pool(2) as pool: pool.map(replay_frame, jobs)
    if which == ["all"]:
        rt = realtime()
        after = lost(1.2) + card("At 430,000 mph, nothing in this picture survives.",
                                 "Light is still 1,560 times faster.", 4.5)
        k = 0
        for im in rt:
            cv2.imwrite(f"{FR}/f{k:04d}.png", im[..., ::-1]); k += 1
        n_rt = round(F.MACHINES["parker"]["pre"] * F.FPS) + 1
        for n in range(len(tl)):
            cv2.imwrite(f"{FR}/f{k:04d}.png", cv2.imread(f"{FR}/r{n:04d}.png")); k += 1
        for im in after:
            cv2.imwrite(f"{FR}/f{k:04d}.png", im[..., ::-1]); k += 1
        import fly_sound as S
        S.write(f"{OUT}/fly_parker.wav", sound(n_rt, len(rt) - n_rt, len(tl), len(after)))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{FR}/f%04d.png", "-i",
                        f"{OUT}/fly_parker.wav", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-b:a", "192k", "-shortest", f"{OUT}/fly_parker.mp4"], check=True)
    print("done", len(jobs))
