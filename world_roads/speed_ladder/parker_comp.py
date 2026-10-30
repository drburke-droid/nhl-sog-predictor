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
DMIN, XS, GX, GY = FX.geometry(P.PZ, P.PY)
PX_M = FX._geo[(P.PZ, P.PY, "px")]                  # m of closest approach per pixel
R0 = D.Renderer(D.Grid(h=1.0), scale=1.0)
dirs, org = R0.dirs, R0.org
tg = np.where(dirs[..., 2] < -1e-6, -org[2] / np.minimum(dirs[..., 2], -1e-6), np.inf)
GROUND = np.isfinite(tg)
GP = org + np.where(GROUND, tg, 0)[..., None] * dirs
FWD = Cam.R()[2]
# the height of each ray's closest approach to the path (for the dust raised inside the blast, near the ground)
_oy, _oz = org[1] - P.PY, org[2] - P.PZ
_ts = np.clip(-(_oy * dirs[..., 1] + _oz * dirs[..., 2]) / (dirs[..., 1] ** 2 + dirs[..., 2] ** 2), 0, tg)
ZC = (org[2] + _ts * dirs[..., 2]).astype(np.float32)
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


# ---------------------------------------------------------------- the shock around the probe and the blast behind it
# The bow shock stands DELTA ahead of the heat shield (a flat disk 1.15 m in radius; behind a shock this strong the
# gas is squeezed ~15-fold, so the gap is about a fifth of the radius) and curves back as a paraboloid (radius of
# curvature ~2.5 m, as for a blunt flat face). Further back it follows the blast-wave law, r^2 = (B^2 / v) * distance:
# the same shock, now spreading as the energy the probe leaves in each metre pushes it out (parker.py). The two meet
# smoothly: r^2 = (B^2/v) u + (5.0 - B^2/v) L (1 - exp(-u / L)), u the distance behind the shock's nose.
DELTA, BULGE_L, NOSE = 0.2, 0.5, 5.0


def r_shock(x, t):
    """radius of the shock (bow shock, then blast front) around the path at x, time t; 0 ahead of it. Behind the point
    where the probe met the air it closes in a rounded end"""
    x = np.asarray(x, float)
    xh = float(P.x_head(t))
    if t <= 0: return np.zeros_like(x)
    xe = np.maximum(x, P.X_IN)
    u = np.maximum(xh + DELTA - xe, 0)
    B2, vh = P.blast(xe) ** 2, head_speed(t)
    # the blast's r^2 = B^2 x its age there (as the probe slows, the age is more than u / v0: in the race, where the
    # plasma has slowed to a tenth of its speed 30 ms on, a steady paraboloid would be half the size)
    r2 = B2 * age(xe, t, u) + np.maximum(NOSE - B2 / vh, 0) * BULGE_L * (1 - np.exp(-u / BULGE_L))
    behind = np.maximum(P.X_IN - x, 0)
    return np.sqrt(np.maximum(r2 - behind ** 2, 0))


def head_speed(t):
    return P.V0 / (1 + (float(P.x_head(t)) - P.X_IN) / P.LS)


def age(xe, t, u):
    """s since the head passed xe; ahead of the head (the bow shock's nose, u < DELTA) as if it had"""
    return np.where(u > DELTA, np.maximum(t - P.t_at(xe), 0), u / P.V0)


def core(x, t):                                     # kept for the dust wall and the shimmer: the shock's radius
    return 0.08 + r_shock(x, t)


def shell_temperature(x, t):
    """K of the air just behind the shock: set by how fast the shock moves into it. At the nose, 192 km/s: ~60,000 K;
    the blast front still moving at 7 km/s gives ~8,000 K (energy going into ionising the air holds it down)"""
    x = np.asarray(x, float)
    r = np.maximum(r_shock(x, t), 1e-3)
    xh = float(P.x_head(t))
    u = np.maximum(xh + DELTA - np.maximum(x, P.X_IN), 0)
    B2, vh = P.blast(np.maximum(x, P.X_IN)) ** 2, head_speed(t)
    drdt = (B2 + vh * np.maximum(NOSE - B2 / vh, 0) * np.exp(-u / BULGE_L)) / (2 * r)  # outward, at fixed x
    slope = drdt / vh                                                                # dr/du
    un = drdt / np.sqrt(1 + slope ** 2)                                              # speed into the shock
    # fast, it is held down by ionising the air; slower (the blast front far out, ~1 km/s at 100 m) an ideal gas's
    # shock: Mach 3 heats it only to ~800 K, and it no longer glows
    M2 = (un / 343.0) ** 2
    rh = 288.0 * (2.8 * M2 - 0.4) * (0.4 * M2 + 2) / (5.76 * np.maximum(M2, 1.0))
    return np.clip(np.minimum(8000.0 * (un / 7000.0) ** 0.6, rh), 300.0, 60000.0)


def inner_temperature(a):
    """K of the gas inside the envelope, a seconds after the head passed: it expands and cools behind the shock"""
    return np.clip(P.temperature(a), 800.0, 40000.0)


def opacity(T):
    return np.clip((np.asarray(T, np.float32) - 1500.0) / 3000.0, 0.0, 1.0)


_mottle = None


def envelope(t):
    """the shock envelope's light along each ray, and how much of what lies inside and behind it shows through.
    A thin shell of freshly shocked air (an eighth of the radius) round a dimmer interior; partly transparent, so the
    edges, where the line of sight runs longest through the shell, glow brightest: what a bow shock looks like.
    -> (radiance rgb, transmission to the probe inside, transmission to what is behind)"""
    global _mottle
    rs = r_shock(XS, t).astype(np.float32)
    on = rs > 0.02
    Lrgb = np.zeros((H, W, 3), np.float32)
    T_in = np.ones((H, W), np.float32); T_bg = np.ones((H, W), np.float32)
    if not on.any(): return Lrgb, T_in, T_bg
    if _mottle is None:                                  # streaks along the flow in the turbulent gas behind
        rng = np.random.default_rng(12)
        _mottle = []
        for _ in range(6):
            m = cv2.GaussianBlur(cv2.resize(rng.standard_normal((H // 6, W // 60)).astype(np.float32), (W, H),
                                            interpolation=cv2.INTER_CUBIC), (0, 0), 3)
            _mottle.append(m / (m.std() + 1e-6))
    d = DMIN
    ri = rs * 0.9
    c_shell = np.where(d < ri, np.sqrt(np.maximum(rs ** 2 - d ** 2, 0)) - np.sqrt(np.maximum(ri ** 2 - d ** 2, 0)),
                       np.sqrt(np.maximum(rs ** 2 - d ** 2, 0)))                 # each side, front and back
    c_in = 2 * np.sqrt(np.maximum(ri ** 2 - d ** 2, 0))
    kap = 1.6 / np.maximum(rs, 0.05)
    a = np.maximum(t - P.t_at(np.maximum(XS, P.X_IN)), 1e-8)
    Ts = shell_temperature(XS, t); Tin = inner_temperature(a)
    # hot air is murky (ionised), cool air clear: the shell and the gas inside thin out as they cool below ~4,500 K
    tf = np.exp(-kap * opacity(Ts) * c_shell); ti = np.exp(-0.12 * kap * opacity(Tin) * c_in)
    ph = t / 40e-6; k0 = int(ph) % 6; f = ph % 1
    mot = _mottle[k0] * (1 - f) + _mottle[(k0 + 1) % 6] * f
    turb = 1 + 0.05 * mot * np.clip(a / 20e-6, 0, 1)
    Ss = (vis(Ts) * turb)[..., None] * bb(Ts)
    Si = (vis(Tin) * turb)[..., None] * bb(Tin)
    # where the blast has swept the ground, the dust it raised fills the bottom of the envelope: darker, browner,
    # thicker there (it rises as the gas inside does)
    dusty = np.clip(1 - ZC / (0.6 + 0.25 * rs), 0, 1) * np.clip((rs - P.PZ) / 1.5, 0, 1) * (c_in > 0)
    Si = Si * (1 - 0.75 * dusty)[..., None] * np.array([1.0, 0.8, 0.62], np.float32)
    Ss = Ss * (1 - 0.45 * dusty)[..., None]
    ti = ti * np.exp(-4.0 * dusty)
    L = Ss * (1 - tf)[..., None] * (1 + (tf * ti)[..., None]) + Si * ((1 - ti) * tf)[..., None]
    Lrgb = np.where(on[..., None], L, 0).astype(np.float32)
    T_in = np.where(on, tf * np.sqrt(ti), 1).astype(np.float32)
    T_bg = np.where(on, tf * tf * ti, 1).astype(np.float32)
    return Lrgb, T_in, T_bg


def lights(t, n=28):
    """the envelope as point lights [((x, y, z), rgb, radius)] in the daylight's units (sun 1.3 W/m^2): a stretch of
    shell of radius r and radiance L shines with intensity ~ L * 2r * length"""
    out = []
    if t <= 0: return out
    xh = float(P.x_head(t))
    xs = np.linspace(P.X_IN, min(xh, P.X_OUT), n)
    dx = xs[1] - xs[0] if n > 1 else 1.0
    for x in xs:
        r = float(r_shock(np.array([x]), t)[0])
        if r <= 0: continue
        T = float(shell_temperature(np.array([x]), t)[0])
        c = bb(T) * float(vis(T)) * 2 * r * dx * 0.22
        out.append(((float(x), P.PY, P.PZ), tuple(float(v) for v in c), max(1.0, r)))
    return out


def ground_light(ls):
    """irradiance on the ground the plate shows, from point lights, relative to the daylight's (0.8)"""
    E = np.zeros((H, W, 3), np.float32)
    for (lx, ly, lz), c, *_ in ls:
        L = np.array([lx, ly, lz], np.float32) - GP
        d2 = (L ** 2).sum(-1) + 0.05
        cos = np.clip(L[..., 2] / np.sqrt(d2), 0, 1)
        E += (cos / d2)[..., None] * np.array(c, np.float32)
    return np.where(GROUND[..., None], E / 0.8, 0)


def exposure(img):
    """the camera's exposure, metered on the whole picture: it holds the brightest part just below white, so the
    daylit scene sinks as the fireball grows"""
    m = float(np.percentile(img.max(-1), 99.97))
    return min(1.0, 1.2 / max(m, 1e-9))


def displacement(t):
    """the shock bending light where rays graze it: the air behind it is many times denser"""
    Rr = r_shock(XS, t).astype(np.float32)
    on = Rr > 0.02
    w = 0.03 + 0.03 * Rr
    q = (DMIN - Rr) / w
    # no more than the band can take without folding over (its steepest slope is 1.65 amp / width in px): a shell
    # thinner than the pixels it covers (far out, as in the race) cannot be resolved
    amp = np.minimum(12.0, 0.48 * w / PX_M)
    disp = np.where(on, amp * (-q * np.exp(0.5 - 0.5 * q * q)), 0).astype(np.float32)
    return disp * GX, disp * GY


# ---------------------------------------------------------------- the ground erupting: a wall of dust and vapour
VIS = D.Grid(x0=-16, x1=17, y0=1.0, y1=36.0, z1=10, h=0.15) if not F.PARKER else \
      D.Grid(x0=P.X_IN - 4, x1=P.X_FINISH + 8, y0=1.0, y1=P.PY + 40, z1=24, h=0.5)    # the race: 100 m out
REN = D.Renderer(VIS, scale=0.5, kappa=1.6, albedo=(0.5, 0.38, 0.27), phase_g=0.72, sun_irr=1.3,
                 sun_col=(1.0, 0.78, 0.58))
_noise = None


def wall_density(t):
    global _noise
    X, Y, Z = VIS.coords()
    if _noise is None: _noise = D.noise3(VIS.n, 6 if not F.PARKER else 10, 31, 3 if not F.PARKER else 4).astype(np.float32)
    Rr = core(X, t) - 0.08
    rg = np.sqrt(np.maximum(Rr ** 2 - P.PZ ** 2, 0))                 # how far out along the ground the blast is
    rho = np.abs(Y - P.PY)
    front = rg * 0.97
    w = 0.2 + 0.05 * rg
    h = 0.4 + 0.12 * rg
    top = h * (0.5 + 1.0 * _noise)
    shell = np.exp(-((rho - front) / w) ** 2) * np.clip((top - Z) / (0.5 * top + 0.1), 0, 1) ** 1.5 * (rho < rg * 1.02)
    inner = 0.12 * (rho < front) * np.clip(1 - Z / (0.8 * top + 0.1), 0, 1)      # what it raised hangs behind
    vapour = np.hypot(Y - P.PY, Z - P.PZ) < 0.5 * Rr                       # the hottest inner core: no dust left
    return (30.0 * (shell + inner) * (0.4 + _noise) * (rg > 0) * ~vapour).astype(np.float32)


def wall(t, ls):
    if P.shock_r(np.array([P.X_IN + 2.0 if F.PARKER else 0.0]), t)[0] < P.PZ: return None
    VIS.d[:] = gaussian_filter(wall_density(t), 0.8)
    if VIS.d.max() < 1e-3: return None
    # a light stands for a stretch of column: soften it by that stretch and the column's radius
    rgb, tr, gs = D.render_l(REN, [(p_, tuple(3.0 * v for v in c), max(2.0, r)) for p_, c, r in ls])
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
    if ls and p is not None: img = img + C.relight(p, [(q_, c) for q_, c, _ in ls])
    L, T_in, T_bg = envelope(t)
    if p is not None:
        # the probe sits inside the envelope (seen through its near side); anything nearer than the envelope is in
        # front of it; the plate is behind it
        Rr = r_shock(XS, t)
        entry = COLDEPTH - np.sqrt(np.maximum(Rr ** 2 - DMIN ** 2, 0))
        solid = p["a"][..., 0] > 0.5
        front = solid & (p["depth"] < entry)
        inside = solid & ~front
        T = np.where(inside, T_in, T_bg)
        img = np.where(front[..., None], img, img * T[..., None] + L)
    else:
        img = img * T_bg[..., None] + L
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
