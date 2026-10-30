"""The race with the Parker Solar Probe (race.py, PARKER=1): the race as before until its last few milliseconds, when
everything slows again, to 1,000 times slower, and the probe comes in at the left, 100 m out, and crosses with the
others: all five clear the right edge on the same 1/25,000 s. Then its blast spreads toward the camera, which it
reaches 56 ms after the finish.
  RACE=1 PARKER=1 OUTSUF=_pk python make_race_parker.py -> /home/user/lot/out/history_race_parker.mp4

Parker here is the finale's (parker.py, parker_comp.py) in its own lane: it meets the air at the left edge (as if it
appeared there) and is a plasma bolt within 5 m; the air it shoves aside slows it, 192 km/s coming in, 90 km/s at the
right edge. Its blast reaches the ground under it 0.5 ms after it passes, the X-15's lane ~20 ms after the finish,
the camera 56 ms after; nothing is touched before the finish.

Its light (parker_comp.lights: the column as point lights, by its temperature) on everything else:
* the ground: hundreds of times the daylight on it (parker_comp.ground_light);
* the air: it scatters a little of that light toward the camera (clear desert air, 4e-5 per metre, about 100 km
  visibility; taken as scattering evenly in all directions), so the sky round the column glows;
* the machines: lit by the glowing ground and air round them: half of what they see is ground (albedo 0.1), half
  glowing air; taken as a boost of their rendered colour (no new renders); the sun's shadows dim only the daylight;
and the camera's exposure follows what its sensor records (an average, log mean, of the picture with what is too
bright for it clipped, as a camera's metering sees it), so the white-hot column blows out and the rest comes down
to where it can be seen; the lens spreads a little of the column's light over the picture (veiling glare, 0.2%)."""
import cv2, numpy as np, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
assert os.environ.get("RACE") == "1" and os.environ.get("PARKER") == "1" and os.environ.get("OUTSUF") == "_pk", \
    "run with RACE=1 PARKER=1 OUTSUF=_pk"
import comp as C, flight as F, fly_fx as FX, race as RC, slowmo as SM, parker as P, parker_comp as PC
import fly_sound as S, make_race as MR, make_cuts as MC

WHITE, DIM = MR.WHITE, MR.DIM
BETA = 4e-5                 # m^-1, what clear desert air scatters
ALB_G = 0.1                 # the ground's albedo (asphalt, sand)
SUB = 4                     # the air's glow at a quarter of the resolution: it is smooth
TOP = 1000.0                # the slowest the cut goes: the far end of the sign's scale
_d = PC.dirs[::SUB, ::SUB]
_rn = np.linalg.norm(_d, axis=-1)
DIRS = (_d / _rn[..., None]).reshape(-1, 3)
REACH = np.where(np.isfinite(PC.tg[::SUB, ::SUB]), PC.tg[::SUB, ::SUB] * _rn, 3000.0).reshape(-1)
ORG = np.asarray(PC.org, float)
GDIST = np.where(PC.GROUND, PC.tg * np.linalg.norm(PC.dirs, axis=-1), np.inf).astype(np.float32)
K_PLATE = float(np.median(C.plate_loop(0)[PC.GROUND].mean(-1)) * math.pi / (ALB_G * 0.8))   # plate units per unit
METER0 = None
LIGHT = None                # make_race_light.py: f(k, extra) -> the light's layer, far behind everything (plate units)


def air_glow(ls):
    """the light the air scatters toward the camera along each ray (to the ground, or 3 km), from the column's
    lights: beta / 4 pi x the integral of each light's irradiance along the ray (plate units)"""
    out = np.zeros((len(DIRS), 3))
    for (lx, ly, lz), c, r in ls:
        v = np.array([lx, ly, lz]) - ORG
        s0 = DIRS @ v
        dm = np.maximum(np.sqrt(np.maximum(v @ v - s0 ** 2, 0)), r)
        out += ((np.arctan((REACH - s0) / dm) + np.arctan(s0 / dm)) / dm)[:, None] * np.array(c)
    g = (out * BETA / (4 * math.pi) * K_PLATE).reshape(_d.shape[0], _d.shape[1], 3).astype(np.float32)
    return cv2.resize(g, (C.W, C.PH), interpolation=cv2.INTER_LINEAR)


def boost(E, glow):
    """times its daylight, the light on a machine's sides: pi x the mean radiance round it (half ground, half air)"""
    Lg = ALB_G * np.median(E[PC.GROUND], axis=0) * 0.8 / math.pi
    La = np.median(glow[~PC.GROUND], axis=0) / K_PLATE
    return (math.pi * 0.5 * (Lg + La) / 0.8).astype(np.float32)


def log_mean(lum):
    return float(np.exp(np.log(np.minimum(lum, 1.0) + 1e-4).mean()))


def metered(img):
    """the exposure that brings the log mean of what the sensor records (clipped at white) back to the daylight
    scene's"""
    global METER0
    if METER0 is None: METER0 = log_mean(C.plate_loop(0)[::4, ::4].mean(-1))
    lum = img[::4, ::4].mean(-1)
    lo, hi = -14.0, 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if log_mean(lum * math.exp(mid)) < METER0 else (lo, mid)
    return math.exp(hi)


def glare(img, gains=((5, 0.001), (19, 0.0007), (50, 0.0003))):
    """the lens's veiling glare: 0.2% of the light spread wide (sigma 40, 150, 400 px), worked at 1/8 size"""
    small = cv2.resize(img, (C.W // 8, C.PH // 8), interpolation=cv2.INTER_AREA)
    g = sum(w * cv2.GaussianBlur(small, (0, 0), s_) for s_, w in gains)
    return cv2.resize(g, (C.W, C.PH), interpolation=cv2.INTER_LINEAR)


def frame(k, extra=None):
    if isinstance(k, tuple): k, extra = k
    tl = RC.timeline()
    T, slow = tl[k]
    expo = 0.01 * slow
    tp = RC.own("parker", T)                     # the probe's own time: 0 when it meets the air at the left edge
    img = C.plate_loop(int((T - tl[0][0]) * F.FPS))
    if LIGHT is not None: img = img + LIGHT(k, extra)
    on = tp > 0
    day_frac, B = 1.0, 1.0
    if on:
        ls = PC.lights(tp)
        E = PC.ground_light(ls)
        glow = air_glow(ls)
        lit = img * (1 + E) + glow
        day_frac = np.clip(img.mean(-1, keepdims=True) / np.maximum(lit.mean(-1, keepdims=True), 1e-6), 0, 1)
        B = 1 + boost(E, glow)
        L, T_in, T_bg = PC.envelope(tp)
        img = FX.warp(lit * T_bg[..., None] + L, PC.displacement(tp))
        w = PC.wall(tp, ls)
        if w is not None: img = img * w[1] + w[0]
    for name in MR.FAR_FIRST:
        t = RC.own(name, T)
        dl = MR.dust(name, k)
        if dl is not None: img = img * (1 - C.SUN_FRACTION * (1 - dl[2]) * day_frac)
        p = MR.load(name, k)
        if p is not None:
            p = C.plate_match(p, 5000 + 7919 * MR.FAR_FIRST.index(name) + k)
            if on:
                solid = (p["depth"] < GDIST - 1.0)[..., None]
                p = dict(p, rgb=np.where(solid, p["rgb"] * B, p["rgb"]), a=np.where(solid, p["a"], p["a"] * day_frac))
            img = C.over_props(img, p)
        g = FX.wake_glow(name, t, expo)
        if g is not None: img = img + g
        if dl is not None: img = img * dl[1] + dl[0] * B
        img = FX.warp(img, FX.displacement(name, t))
    if on:
        img = img * metered(img)
        img = img + glare(img)
        hi = np.clip(img - 0.8, 0, 2.2)                                 # the sensor's own halo round what it clips
        for s_, w in ((6, 0.3), (24, 0.2), (80, 0.1)): img = img + w * cv2.GaussianBlur(hi, (0, 0), s_)
    else:
        img = C.bloom(img)
    for i, name in enumerate(("f35", "x15")):
        st = MR.shake_time(name, T)
        if st >= 0: img = C.shake(img, st, amp=MC.SHAKE_PER_KPA * MR.KPA[name], decay=4.0, seed=3 + i)
    wk = white(k)
    if wk > 0:                                  # the blast at the lens: Mach 3 by now, ten atmospheres, and the dust
        img = C.shake(img, wk * 0.02, amp=60.0, decay=0.0, seed=11)
        img = img * (1 - wk) + np.array([0.035, 0.026, 0.018], np.float32) * wk
    img = C.grain(img, 300 + k)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return out


def white(k):
    """the last two frames: the blast, carrying the dust it raised, reaching the lens"""
    tl = RC.timeline()
    T_cam, dt = tl[-1][0], 2 * (tl[-1][0] - tl[-2][0])
    return float(np.clip((tl[k][0] - (T_cam - dt)) / dt, 0, 1))


def labels(fr, k):
    T = RC.timeline()[k][0]
    lines = []
    for name in RC.ORDER + ["parker"]:
        if RC.own(name, T) < RC.entry_time(name) - 1e-9: continue
        m = F.MACHINES[name]
        year, title = m["title"].split(" ", 1)
        lines.append((f"{m['mph']:,} mph  ·  {title}, {year}", 30, "SemiBold", WHITE))
    if T > 1e-4:
        lines.append(("all five cleared the frame on the same 1/25,000 s", 26, "Medium", DIM))
    return C.label(fr, lines, 1.0 - white(k), corner="bl") if lines else fr


def audio(tl, n_after):
    """the race's sound on this timeline (make_race.audio: each machine's on its own clock, played back as the frames
    go); the probe is silent until its blast reaches the camera: a crack too loud to record, cut off as the camera
    goes; then silence"""
    a = MR.audio(tl)
    sr = S.SR
    m = int(0.03 * sr)
    i0 = max(0, len(a) - int(2 / F.FPS * sr))
    rng = np.random.default_rng(21)
    tt = np.arange(m) / sr
    crack = np.clip(3.0 * (1 - tt / 0.03) * (1 + 0.6 * rng.standard_normal(m)), -1, 1)
    a[i0:] *= np.linspace(1, 0.2, len(a) - i0)[:, None]
    a = np.concatenate([a, np.zeros((int(n_after / F.FPS * sr), 2))])
    a[len(a) - int(n_after / F.FPS * sr):][:m] += crack[:, None]
    return a


def overlay(fr, k):
    """the labels and the slow-motion signs, on a composited frame"""
    tl = RC.timeline()
    full_slow = next(j for j, (T, s) in enumerate(tl) if abs(s - RC.SLOW) < 1e-9)
    k_ramp = next(j for j in range(full_slow, len(tl)) if tl[j][1] < RC.SLOW - 1e-12)   # the second slowdown
    T, slow = tl[k]
    fr = labels(fr, k)
    amt = min(1.0, math.log(1 / slow) / math.log(48.0)) if slow < 0.99 else 0.0
    fr = SM.badge(SM.tint(fr, amt), slow, T - tl[0][0], top=TOP)
    fr = SM.slam(fr, 48, k - full_slow + 4)
    return SM.slam(fr, 1000, k - k_ramp, n_hold=12, n_out=10)          # gone before Parker comes in


def ending():
    return PC.lost(1.2) + PC.card("At 430,000 mph, nothing in this picture survives.",
                                  "Light is still 1,560 times faster.", 4.5)


def splice(k0, k1):
    """frames k0..k1 composited afresh, the rest as they are in the finished cut; re-encoded"""
    import subprocess
    from multiprocessing import Pool
    tl, after = RC.timeline(), ending()
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        fresh = dict(zip(range(k0, k1 + 1), pool.map(frame, range(k0, k1 + 1))))
    src = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{MC.OUT}/history_race_parker.mp4", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    sink = MC.Sink("history_race_parker_fix")
    for i in range(len(tl) + len(after)):
        fr = np.frombuffer(src.stdout.read(C.W * C.H * 3), np.uint8).reshape(C.H, C.W, 3)
        sink.add([overlay(fresh[i], i) if k0 <= i <= k1 else fr])
    src.wait()
    sink.close(MC.mic(audio(tl, len(after)), 1.6))
    for ext in ("mp4", "wav"):
        os.replace(f"{MC.OUT}/history_race_parker_fix.{ext}", f"{MC.OUT}/history_race_parker.{ext}")


if __name__ == "__main__":
    from multiprocessing import Pool
    tl = RC.timeline()
    if sys.argv[1:2] == ["splice"]:
        splice(int(sys.argv[2]), int(sys.argv[3])); sys.exit()
    ks_all = [int(a) for a in sys.argv[1:]] or list(range(len(tl)))
    if sys.argv[1:]:                                       # a few frames, to look at
        os.makedirs("/home/user/lot/out/test_pk", exist_ok=True)
    else:
        sink = MC.Sink("history_race_parker")
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        for c0 in range(0, len(ks_all), 24):
            ks = ks_all[c0:c0 + 24]
            out = [overlay(fr, k) for k, fr in zip(ks, pool.map(frame, ks))]
            if sys.argv[1:]:
                for k, fr in zip(ks, out): cv2.imwrite(f"/home/user/lot/out/test_pk/f{k:04d}.png", fr[..., ::-1])
            else:
                sink.add(out)
            print("frames", ks[-1], flush=True)
    if not sys.argv[1:]:
        after = ending()
        sink.add(after)
        sink.close(MC.mic(audio(tl, len(after)), 1.6))
