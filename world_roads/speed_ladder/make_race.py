"""The final cut: the race to the right edge (race.py). The Wright Flyer comes in at real speed; as it nears the far
side everything slows to 48 times slower, and the Spitfire, the F-35 and the X-15 come in one after another, each
at its own real speed, so that all four clear the right edge on the same frame. Then the booms reach the camera
while it ramps back to real speed.
  RACE=1 python make_race.py -> /home/user/lot/out/history_race.mp4

Layers, farthest first (the lanes never cross, so that is the right order everywhere): the X-15, the F-35, the
Spitfire, the Wright Flyer; each over the plate with its shadow, its dust (flyby_dust.py, in its lane), the wake's
glow, and the picture behind it bent by its shock waves. Peak pressure at the camera scales from the 20 m shots as
distance^-3/4: the F-35's boom ~8 kPa, the X-15's blast ~10 kPa; each shakes the camera when it arrives."""
import cv2, numpy as np, os, subprocess, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, flight as F, fly_fx as FX, fly_sound as S, make_fly as MF, race as RC, slowmo as SM
import make_cuts as MC

R_ = "/home/user/lot/render"
SUF = os.environ.get("OUTSUF", "")                  # _pk: the race with Parker (make_race_parker.py)
OUT = "/home/user/lot/out"
WHITE, DIM = MF.WHITE, MF.DIM
FAR_FIRST = ["x15", "f35", "spitfire", "wright"]
D_SHOT = math.hypot(F.FLY_Y, 6.0 - 1.37)            # the camera from the path in the shots: 20 m out, 6 m up


def dist(name):
    y, z = F.RACE_LANES[name]
    return math.hypot(y, z - 1.37)


KPA = {n: MC.KPA[n] * (D_SHOT / dist(n)) ** 0.75 for n in ("f35", "x15")}


def load(name, k):
    p = f"{R_}/fly_{name}_race{SUF}/p{k:04d}.npz"
    if not os.path.exists(p): return None
    st = C.props(f"{R_}/fly_{name}/static.npz")
    d = np.load(p)
    out = C.props(p)
    if "rect" in d:
        x0, y0, x1, y1 = [int(v) for v in d["rect"]]
        h, w = out["a"].shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        big = np.full_like(xx, 1e4)
        dd = np.minimum.reduce([xx - x0 if x0 > 0 else big, x1 - 1 - xx if x1 < w else big,
                                yy - y0 if y0 > 0 else big, y1 - 1 - yy if y1 < h else big])
        kk = np.clip(dd / MF.FEATHER, 0, 1)[..., None]
        for key in ("rgb", "a"):
            out[key] = out[key] * kk + st[key] * (1 - kk)
    return out


def dust(name, k):
    p = f"{R_}/fly_{name}_race_dust{SUF}"
    return C.dust(f"{p}/d{k:04d}.png", f"{p}/g{k:04d}.png")


def shake_time(name, T):
    """race time since the boom reached the camera"""
    if name not in shake_time.c:
        n0 = round(F.MACHINES[name]["pre"] * F.FPS)
        shake_time.c[name] = S.cone_arrival(name, n0)[0] - n0 / F.FPS - RC.exit_time(name)
    return T - shake_time.c[name]


shake_time.c = {}


def frame(k):
    tl = RC.timeline()
    T, slow = tl[k]
    expo = 0.01 * slow
    img = C.plate_loop(int((T - tl[0][0]) * F.FPS))
    for name in FAR_FIRST:
        t = RC.own(name, T)
        dl = dust(name, k)
        if dl is not None: img = img * (1 - C.SUN_FRACTION * (1 - dl[2]))
        p = load(name, k)
        if p is not None: img = C.over_props(img, C.plate_match(p, 5000 + 7919 * FAR_FIRST.index(name) + k))
        glow = FX.wake_glow(name, t, expo)
        if glow is not None: img = img + glow
        if dl is not None: img = img * dl[1] + dl[0]
        img = FX.warp(img, FX.displacement(name, t))
    img = C.bloom(img)
    for i, name in enumerate(("f35", "x15")):
        st = shake_time(name, T)
        if st >= 0: img = C.shake(img, st, amp=MC.SHAKE_PER_KPA * KPA[name], decay=4.0, seed=3 + i)
    img = C.grain(img, 300 + k)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return out


def roll(k):
    """the machines in the race so far, bottom left, each line coming up as it comes in"""
    T = RC.timeline()[k][0]
    lines = []
    for name in RC.ORDER:
        if RC.own(name, T) < RC.entry_time(name) - 1e-9: continue
        m = F.MACHINES[name]
        year, title = m["title"].split(" ", 1)
        lines.append((f"{m['mph']:,} mph  ·  {title}, {year}", 30, "SemiBold", WHITE))
    return lines


def labels(fr, k):
    lines = roll(k)
    T = RC.timeline()[k][0]
    if T > 0.004:                                      # all gone: the finish
        lines = lines + [("all four cleared the frame on the same 1/1,200 s", 26, "Medium", DIM)]
    return C.label(fr, lines, 1.0, corner="bl") if lines else fr


def audio(tl):
    """every machine's sound on its own clock, mixed on the race's, then played back as the frames go"""
    rng = np.random.default_rng(3)
    T0, T1 = tl[0][0] - 0.5, tl[-1][0] + 0.5
    n = int((T1 - T0) * S.SR)
    mix = S.wind(n, rng)
    for name in RC.ORDER:
        t_a = T0 + RC.exit_time(name)             # its own time when the race's sound begins
        dur = T1 - T0
        src = S.SOURCES[name](dur + 2.0, rng)
        # heard(): its clock is flight time + n0 / 25; this n0 starts it at t_a
        n0p = -t_a * F.FPS
        sh = S.LEVEL[name] * S.heard(name, src, t_a - 1.0, n0p, dur)
        k_p = KPA.get(name, 1.0) / MC.KPA.get(name, 1.0)
        if name in S.BOOM: sh += k_p * S.boom(name, n0p, n, rng)
        if name == "x15": sh += k_p * S.blast_events(n0p, n, rng)
        mix[:len(sh)] += sh[:n]
    # played back as the frames go: screen time -> race time
    ts = np.array([T for T, _ in tl])
    tau = np.arange(int(len(tl) / F.FPS * S.SR)) / S.SR
    Tk = np.interp(tau * F.FPS, np.arange(len(tl)), ts)
    pos = (Tk - T0) * S.SR
    out = np.stack([np.interp(pos, np.arange(n), mix[:, c]) for c in (0, 1)], 1)
    out = S.filt(out, "highpass", 25, 4)          # slowed 48x most of it falls below hearing: keep what can be heard
    slowk = np.interp(tau * F.FPS, np.arange(len(tl)), [min(1.0, math.log(1 / s) / math.log(48)) for _, s in tl])
    rum = S.filt(np.random.default_rng(17).standard_normal((len(out), 2)), "bandpass", [30, 110], 2)
    rum *= 0.012 / (np.sqrt((rum ** 2).mean()) + 1e-9)
    return out * (1 - 0.45 * slowk)[:, None] + rum * slowk[:, None]     # the slowed sound a little under, as in the replays


if __name__ == "__main__":
    from multiprocessing import Pool
    tl = RC.timeline()
    fin = RC.finish_frame()
    slow_from = next(k for k, (T, s) in enumerate(tl) if s < 0.99)
    full_slow = next(k for k, (T, s) in enumerate(tl) if abs(s - RC.SLOW) < 1e-9)
    sink = MC.Sink("history_race")
    with Pool(3) as pool:
        for k0 in range(0, len(tl), 24):
            ks = list(range(k0, min(len(tl), k0 + 24)))
            raw = pool.map(frame, ks)
            out = []
            for k, fr in zip(ks, raw):
                T, slow = tl[k]
                fr = labels(fr, k)
                amt = min(1.0, math.log(1 / slow) / math.log(48.0)) if slow < 0.99 else 0.0
                fr = SM.badge(SM.tint(fr, amt), slow, T - tl[0][0])
                fr = SM.slam(fr, 48, k - full_slow + 4)
                out.append(fr)
            sink.add(out)
            print("frames", ks[-1], flush=True)
    sink.close(MC.mic(audio(tl), 1.6))
