"""The race with Parker and light (make_race_parker.py, plus light): a pulse of light sets off as the Wright Flyer
comes in, and goes round the world 17 times before the finish, crossing the picture each time it comes round.
  RACE=1 PARKER=1 OUTSUF=_pk python make_race_light.py -> /home/user/lot/out/history_race_light.mp4

Light goes round the equator (40,075 km) in 0.1337 s: 7.5 times a second. Counting back 17 laps from a pass just
before Parker comes in lands on the frame the Wright Flyer comes in; so while the Flyer crosses the picture, light goes
round 17 times.
* Where: 250 m out, behind everything; each lap 3 m above the last (a helix round the Earth: that adds nothing
  measurable to a lap), so each pass leaves its own line, a ladder building in the sky. The lines are markers (clear
  air keeps no trace of light that has gone by), faint.
* What the camera catches: light crosses the 384 m-wide picture there in 1.3 us, far less than any frame, so a pass is
  in one frame or none: the shutter is open a quarter of each frame (1/100 s at real speed), so it catches 4 of the 17
  passes it does not freeze on, as a streak across the picture, brightest on the right (it takes the light longer to
  reach the camera from the right of the picture... the left: the pulse coming toward the camera seems to rush,
  going away to dawdle, and the streak is brightest where it seems slowest).
* Its first pass and its last: time stops (the race frozen in its frame) and the pulse crosses a million times slower
  (a camera at 25 million frames a second), seen as it reaches the camera: rushing in at ~2.5 times the speed of light
  and slowing to ~0.6 as it leaves (the light from its far side has farther to come). A pulse is seen only by the light
  it scatters toward the camera; the glow stands in for that.
Light does not follow the curve of the Earth by itself (in a fibre round the equator a lap would take 0.20 s)."""
import cv2, numpy as np, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_race_parker as MP
import comp as C, flight as F, race as RC, slowmo as SM, camera as Cam, fly_sound as S, make_cuts as MC

C_LIGHT = 299792458.0
LAP = 40075e3 / C_LIGHT
YL, Z0, DZ = 250.0, 6.0, 3.0                # the light's lane: distance out, first lap's height, rise per lap
N_PASS = 18                                 # passes: the first, then one at the end of each of 17 laps
FREEZE = 1e-6                               # the frozen passes: a million times slower
TOP = 1e6
COL = np.array([0.82, 0.9, 1.0], np.float32)
tl = RC.timeline()
TS = np.array([T for T, _ in tl])
K_IN = next(k for k, (T, s) in enumerate(tl) if RC.own("parker", T) >= 0)
K_LAST = K_IN - 3                           # its last pass: three frames (120 us) before Parker comes in
PASS_T = [TS[K_LAST] - (N_PASS - 1 - n) * LAP for n in range(N_PASS)]
K_FIRST = int(round(np.interp(PASS_T[0], TS, np.arange(len(TS)))))
FROZEN = {0: K_FIRST, N_PASS - 1: K_LAST}   # pass -> the frame time stops in
PRE, POST = 34, 10                          # frozen frames before the pulse comes in (the sign's time) and after


def edge_x(u, z):
    lo, hi = -1e4, 1e4
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Cam.project((mid, YL, z))[0][0, 0] < u else (lo, mid)
    return (lo + hi) / 2


CAM = np.array([Cam.CAM_X, 0.0, Cam.H])
XA, XB = edge_x(-40.0, Z0) , edge_x(Cam.W + 40.0, Z0)       # where a pass begins and ends, just off the picture


def seen_at(x, z):
    """camera time (from the pulse at XA) at which the pulse at x is seen"""
    return (x - XA) / C_LIGHT + np.linalg.norm(np.stack([x - CAM[0], np.full_like(x, YL), np.full_like(x, z - CAM[2])]),
                                               axis=0) / C_LIGHT


def x_seen(tau, z):
    """where the pulse is seen at camera time tau"""
    lo, hi = np.full_like(tau, XA - 50), np.full_like(tau, XB + 50)
    for _ in range(50):
        mid = (lo + hi) / 2
        early = seen_at(mid, z) < tau
        lo, hi = np.where(early, mid, lo), np.where(early, hi, mid)
    return (lo + hi) / 2


def uv(x, z):
    return Cam.project(np.stack([x, np.full_like(x, YL), np.full_like(x, z)], -1))[0]


_rows = np.arange(C.PH, dtype=np.float32)[:, None]
_cols = np.arange(C.W, dtype=np.float32)[None, :]


def line(n, u_end=None, peak=0.12, sigma=1.3, weight=None):
    """the line pass n leaves (to column u_end), anti-aliased, with a soft glow: (row slice, its band of the layer)"""
    z = Z0 + DZ * n
    (u0, v0), (u1, v1) = uv(np.array([XA, XB]), z)
    v = v0 + (v1 - v0) * (_cols - u0) / (u1 - u0)
    r0 = max(0, int(min(v0, v1) - 12 * sigma)); r1 = min(C.PH, int(max(v0, v1) + 12 * sigma) + 1)
    d2 = (_rows[r0:r1] - v) ** 2
    prof = peak * np.exp(-0.5 * d2 / sigma ** 2) + 0.25 * peak * np.exp(-0.5 * d2 / (5 * sigma) ** 2)
    if weight is not None: prof = prof * weight
    if u_end is not None: prof = prof * np.clip(u_end - _cols + 0.5, 0, 1)
    return slice(r0, r1), prof[..., None] * COL


def dwell(z):
    """how long the pulse seems to stay at each column: d(seen time)/du, normalised to 1 at the middle"""
    xs = np.linspace(XA, XB, 400)
    us = uv(xs, z)[:, 0]
    tau = seen_at(xs, z)
    d = np.gradient(tau, us)
    w = np.interp(_cols[0], us, d)
    return (w / np.interp(C.W / 2, us, d))[None, :]


def ball(u, v, streak_u0, peak=6.0):
    """the pulse: a glowing ball, smeared along its path over the exposure"""
    uu = np.clip(_cols, min(streak_u0, u), max(streak_u0, u))
    d2 = (_cols - uu) ** 2 + (_rows - v) ** 2
    return (peak * np.exp(-0.5 * d2 / 3.0 ** 2) + 0.5 * np.exp(-0.5 * d2 / 18.0 ** 2) +
            0.08 * np.exp(-0.5 * d2 / 60.0 ** 2))[..., None] * COL


def layer(k, extra=None):
    """everything of the light's in frame k: the lines of the passes so far, the streak of a pass the shutter catches;
    extra = (pass, camera time) in a frozen frame: the pulse and the line it is drawing"""
    T, slow = tl[k]
    out = np.zeros((C.PH, C.W, 3), np.float32)
    frozen_n = extra[0] if extra else None
    for n, Tn in enumerate(PASS_T):
        if n == frozen_n:
            z = Z0 + DZ * n
            tau = extra[1]
            if tau <= seen_at(np.array([XA]), z)[0]: continue
            x = x_seen(np.array([tau]), z)[0]
            x0 = x_seen(np.array([max(tau - 0.25 * FREEZE / F.FPS, 0)]), z)[0]
            (u, v), (u0, _) = uv(np.array([x, x0]), z)
            rs, band = line(n, u_end=u); out[rs] += band
            if XA < x < XB: out += ball(u, v, u0)
            continue
        if Tn > T + 1e-12 and not (extra and n < frozen_n): continue
        rs, band = line(n); out[rs] += band
        expo = 0.01 * slow
        if n not in FROZEN and abs(Tn - T) <= expo / 2:                  # caught: the streak, this frame only
            rs, band = line(n, peak=1.4, sigma=1.6, weight=dwell(Z0 + DZ * n)); out[rs] += band
    return out


def laps_done(T):
    return sum(1 for Tn in PASS_T if Tn <= T + 1e-12)


def labels(fr, k, extra=None):
    T = tl[k][0]
    lines = []
    for name in RC.ORDER + ["parker"]:
        if RC.own(name, T) < RC.entry_time(name) - 1e-9: continue
        m = F.MACHINES[name]
        year, title = m["title"].split(" ", 1)
        lines.append((f"{m['mph']:,} mph  ·  {title}, {year}", 30, "SemiBold", MP.WHITE))
    p = laps_done(T) if not extra else extra[0] + (extra[1] > seen_at(np.array([XA]), Z0 + DZ * extra[0])[0])
    if p:
        n = p - 1
        what = "setting off round the world" if n == 0 else f"{n} lap{'s' if n > 1 else ''} of the Earth"
        lines.append((f"{C_LIGHT * 3600 / 1609.344:,.0f} mph  ·  light, {what}", 30, "SemiBold", (200, 225, 255)))
    if T > 1e-4:
        lines.append(("all five cleared the frame on the same 1/25,000 s", 26, "Medium", MP.DIM))
    return C.label(fr, lines, 1.0 - MP.white(k), corner="bl") if lines else fr


def jobs():
    """[(k, extra, slow, clock)] in the order shown: the race's frames, with time stopped at the first pass and the
    last: PRE frames, the pulse crossing, POST frames, all in the one frame of the race"""
    out = []
    for k, (T, s) in enumerate(tl):
        for n, kf in FROZEN.items():
            if k != kf: continue
            Tn, dt = PASS_T[n], FREEZE / F.FPS
            t0, t1 = seen_at(np.array([XA, XB]), Z0 + DZ * n)        # the pulse first seen, last seen
            for j in range(-PRE, int(math.ceil((t1 - t0) / dt)) + POST):
                out.append((k, (n, t0 + j * dt), FREEZE, Tn - tl[0][0] + j * dt))
        out.append((k, None, s, T - tl[0][0]))
    return out


def audio(js, n_after):
    """the race's sound (make_race_parker.audio), with silence while time stands still"""
    a = MP.audio(tl, n_after)
    sr = S.SR
    per = sr / F.FPS
    segs, i_race = [], 0
    run = 0
    for k, extra, s, clock in js:
        if extra is None:
            run += 1
        else:
            if run:
                segs.append(a[int(i_race * per):int((i_race + run) * per)]); i_race += run; run = 0
            segs.append(np.zeros((int(per), 2)))
    segs.append(a[int(i_race * per):])
    return np.concatenate(segs)


if __name__ == "__main__":
    from multiprocessing import Pool
    MP.LIGHT = layer
    js = jobs()
    full_slow = next(k for k, (T, s) in enumerate(tl) if abs(s - RC.SLOW) < 1e-9)
    full_deep = next(k for k, (T, s) in enumerate(tl) if abs(s - RC.DEEP) < 1e-9)
    k_ramp = next(k for k in range(full_slow, len(tl)) if tl[k][1] < RC.SLOW - 1e-12)   # the second slowdown
    pick = [int(a) for a in sys.argv[1:]]
    if pick: js = [js[i] for i in pick]
    if not pick: sink = MC.Sink("history_race_light")
    os.makedirs("/home/user/lot/out/test_lt", exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        for c0 in range(0, len(js), 24):
            chunk = js[c0:c0 + 24]
            raw = pool.map(MP.frame, [(k, extra) for k, extra, s, clock in chunk])
            out = []
            for (k, extra, s, clock), fr in zip(chunk, raw):
                fr = labels(fr, k, extra)
                amt = min(1.0, math.log(1 / s) / math.log(48.0)) if s < 0.99 else 0.0
                fr = SM.badge(SM.tint(fr, amt), s, clock, top=TOP)
                if extra is None:
                    fr = SM.slam(fr, 48, k - full_slow + 4)
                    fr = SM.slam(fr, 1000, k - k_ramp, n_hold=12, n_out=10)    # gone before Parker comes in
                else:
                    jj = round((extra[1] - seen_at(np.array([XA]), Z0 + DZ * extra[0])[0]) / (FREEZE / F.FPS)) + PRE
                    fr = SM.slam(fr, 1e6, jj, n_hold=16)
                out.append(fr)
            if pick:
                for i, fr in zip(pick[c0:c0 + 24], out): cv2.imwrite(f"/home/user/lot/out/test_lt/j{i:04d}.png", fr[..., ::-1])
            else:
                sink.add(out)
            print("jobs", c0 + len(chunk), "of", len(js), flush=True)
    if not pick:
        after = MP.PC.lost(1.2) + MP.PC.card("Light went round the world 17 times",
                                             "while the Wright Flyer crossed the picture.", 4.5)
        sink.add(after)
        sink.close(MC.mic(audio(js, len(after)), 1.6))
