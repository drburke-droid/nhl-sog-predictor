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
import captions as CAP

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
K_LAST = K_IN + int(os.environ.get("LT_LAST", "8"))   # its last pass: 8 frames (320 us) after Parker comes in, so
                                            # time stops with Parker in the picture and light passes it


def passes(k_last):
    """the passes' times (17 laps back from the last, in frame k_last), the frame the first falls in, and pass -> the
    frame time stops in"""
    pt = [TS[k_last] - (N_PASS - 1 - n) * LAP for n in range(N_PASS)]
    k_first = int(round(np.interp(pt[0], TS, np.arange(len(TS)))))
    return pt, k_first, {0: k_first, N_PASS - 1: k_last}


PASS_T, K_FIRST, FROZEN = passes(K_LAST)
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


def passes_seen(k, extra):
    """passes the camera has seen by this frame (the frozen ones once the pulse comes into the picture)"""
    if not extra: return laps_done(tl[k][0])
    return extra[0] + (extra[1] > seen_at(np.array([XA]), Z0 + DZ * extra[0])[0])


def labels(fr, k, extra=None):
    T = tl[k][0]
    lines = []
    for name in RC.ORDER + ["parker"]:
        if RC.own(name, T) < RC.entry_time(name) - 1e-9: continue
        m = F.MACHINES[name]
        year, title = m["title"].split(" ", 1)
        lines.append((f"{m['mph']:,} mph  ·  {title}, {year}", 30, "SemiBold", MP.WHITE))
    p = passes_seen(k, extra)
    if p:
        lines.append((f"{C_LIGHT * 3600 / 1609.344:,.0f} mph  ·  light", 30, "SemiBold", SM.CYAN))
    if T > 1e-4:
        lines.append(("all five cleared the frame on the same 1/25,000 s", 26, "Medium", MP.DIM))
    return C.label(fr, lines, 1.0 - MP.white(k), corner="bl") if lines else fr


def jobs(pt=None, frozen=None):
    """[(k, extra, slow, clock)] in the order shown: the race's frames, with time stopped at the first pass and the
    last: PRE frames, the pulse crossing, POST frames, all in the one frame of the race"""
    pt, frozen = pt or PASS_T, frozen or FROZEN
    out = []
    for k, (T, s) in enumerate(tl):
        for n, kf in frozen.items():
            if k != kf: continue
            Tn, dt = pt[n], FREEZE / F.FPS
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


CLEAN = "/home/user/lot/out/race_light_clean.mp4"     # the composited frames, before any sign or label


def clean():
    """the slow part, done once: the composited frames into a near-lossless file that the signs are drawn over"""
    from multiprocessing import Pool
    import subprocess
    MP.LIGHT = layer
    js = jobs()
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C.W}x{C.H}",
                          "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt",
                          "yuv420p", CLEAN + ".part.mp4"], stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        for n, fr in enumerate(pool.imap(MP.frame, [(k, extra) for k, extra, s, clock in js], chunksize=2)):
            p.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
            if (n + 1) % 24 == 0: print("clean", n + 1, "of", len(js), flush=True)
    p.stdin.close(); p.wait()
    os.replace(CLEAN + ".part.mp4", CLEAN)
    print("clean done", len(js), flush=True)


HOLD_PRE, HOLD_POST = 65, 40      # frames time stands still before a frozen pass comes in, and after it has gone
HOLDS = {0: (HOLD_PRE, HOLD_POST), N_PASS - 1: (HOLD_PRE, HOLD_POST)}
if os.environ.get("LT_HOLDS"):     # "pre,post,pre,post": held longer, for a voice to say what is going on
    _h = [int(x) for x in os.environ["LT_HOLDS"].split(",")]
    HOLDS = {0: (_h[0], _h[1]), N_PASS - 1: (_h[2], _h[3])}
RESUME = 100                      # frames at most the "back to" sign shows, at the end of the stretch after a pass


def sequence():
    """the frames shown, [(job, clock, phase)]: the jobs in order, but with the frozen stretch before each frozen pass
    held for HOLD_PRE frames and the one after for HOLD_POST (their frames repeat back and forth: in a few
    microseconds nothing moves but the grain), long enough to read what is going on; the clock keeps its 40 ns a
    frame throughout. phase: None for the race's own frames, else (pass, "pre" | "cross" | "post", frames into it)"""
    js = jobs()
    dt = FREEZE / F.FPS
    out, i = [], 0
    while i < len(js):
        k, extra, s, clock = js[i]
        if extra is None:
            out.append((i, clock, None)); i += 1; continue
        run = [j for j in range(i, len(js)) if js[j][1] is not None and js[j][1][0] == extra[0]]
        pre, cross, post = run[:PRE], run[PRE:len(run) - POST], run[len(run) - POST:]
        bounce = lambda seq, m: seq[m % (2 * len(seq) - 2) if m % (2 * len(seq) - 2) < len(seq)
                                    else 2 * len(seq) - 2 - m % (2 * len(seq) - 2)]
        c0, c1 = js[cross[0]][3], js[cross[-1]][3]
        h_pre, h_post = HOLDS[extra[0]]
        for m in range(h_pre):
            out.append((bounce(pre[::-1], h_pre - 1 - m), c0 - (h_pre - m) * dt, (extra[0], "pre", m)))
        for m, j in enumerate(cross):
            out.append((j, js[j][3], (extra[0], "cross", m)))
        for m in range(h_post):
            out.append((bounce(post, m), c1 + (m + 1) * dt, (extra[0], "post", m)))
        i = run[-1] + 1
    return js, out


def lap_counter(frame8, laps, flash=0.0, alpha=1.0):
    """in the band above the picture, between the clock and the rate: light's laps of the Earth so far, one pip
    for each of the 17, the newest flaring as it comes round"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame8).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    cy = SM.BAND // 2 + 4
    n_laps = N_PASS - 1
    pip, gap = 16, 9
    x0 = 560
    r = 22                                              # the Earth: a disc with its equator and a meridian
    gx, gy = x0 + r, cy
    d.ellipse((gx - r, gy - r, gx + r, gy + r), outline=SM.CYAN + (a,), width=3)
    d.line((gx - r, gy, gx + r, gy), fill=SM.CYAN + (a,), width=2)
    d.ellipse((gx - r * 0.45, gy - r, gx + r * 0.45, gy + r), outline=SM.CYAN + (int(a * 0.7),), width=2)
    tx = gx + r + 18
    f1, f2 = C.font(24, "Bold"), C.font(24, "Medium")
    head = "LIGHT"
    d.text((tx, cy - 38), head, font=f1, fill=SM.CYAN + (a,))
    w = d.textlength(head + "  ", font=f1)
    what = "setting off round the Earth" if laps == 0 else f"{laps} lap{'s' if laps != 1 else ''} of the Earth"
    d.text((tx + w, cy - 38), what, font=f2, fill=SM.WHITE + (a,))
    for n in range(n_laps):
        x = tx + n * (pip + gap)
        box = (x, cy + 4, x + pip, cy + 4 + pip)
        if n < laps:
            d.ellipse(box, fill=SM.CYAN + (a,))
        else:
            d.ellipse(box, outline=SM.DIM + (int(a * 0.55),), width=2)
    if flash > 0 and laps > 0:
        x = tx + (laps - 1) * (pip + gap) + pip / 2
        rr = pip / 2 + 10 * flash
        d.ellipse((x - rr, cy + 4 + pip / 2 - rr, x + rr, cy + 4 + pip / 2 + rr), fill=(235, 248, 255, int(a * flash)))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def cues(js, seq):
    """the captions, on the frames shown"""
    first = {}
    for o, (j, clock, phase) in enumerate(seq):
        k = js[j][0]
        if phase is None: first.setdefault(k, o)
        else: first.setdefault((phase[0], phase[1]), o)
    A, B = 0, N_PASS - 1
    o = lambda k: first[k]
    laps_at = lambda k: laps_done(tl[k][0]) - 1
    k_x15 = next(k for k, (T, s) in enumerate(tl) if RC.own("x15", T) >= RC.entry_time("x15") - 1e-9)
    k_fin = next(k for k, (T, s) in enumerate(tl) if T >= -1e-12)
    return [
        (first[(A, "pre")], first[(A, "cross")], "Time has stopped. Watch for a pulse of [light]."),
        (first[(A, "cross")], first[(A, "post")] + HOLDS[A][1], "[Light crosses the whole picture in 1.3 millionths of a second.]"),
        (o(K_FIRST), o(K_FIRST + 95), "Off round the Earth: [7½ laps a second], one line in the sky per lap."),
        (o(130), o(215), "Slowed 48×, [light still laps the Earth every 6½ seconds]."),
        (o(k_x15), o(k_x15 + 40), f"[{laps_at(k_x15)} laps of the Earth] so far, and the finish is close."),
        (o(K_IN - 18), first[(B, "pre")], "Here comes Parker: 430,000 mph."),
        (first[(B, "pre")], first[(B, "cross")], "Time stops again: Parker stands still, but [light doesn't]."),
        (first[(B, "cross")], first[(B, "post")] + HOLDS[B][1],
         "[Its 17th lap: light races past Parker as if it were standing still.]"),
        (o(K_LAST), o(k_fin), "Parker carries on to the finish."),
        (o(k_fin), o(k_fin + 50), "All five finish together, in the same 1/25,000 s."),
        (o(k_fin + 50), o(len(tl) - 1) + 1 + 30, "Even Parker is [1,560 times slower than light]."),     # into the black
    ]


def audio_seq(js, seq, n_after):
    """the race's sound (make_race_parker.audio), with silence while time stands still"""
    a = MP.audio(tl, n_after)
    per = S.SR / F.FPS
    segs, i_race, run = [], 0, 0
    for j, clock, phase in seq:
        if phase is None:
            run += 1
            continue
        if run:
            segs.append(a[int(i_race * per):int((i_race + run) * per)]); i_race += run; run = 0
        segs.append(np.zeros((int(per), 2)))
    segs.append(a[int(i_race * per):])
    return np.concatenate(segs)


def clean_frames():
    """the clean file's frames, in job order"""
    import subprocess
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", CLEAN, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    n = C.W * C.H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(C.H, C.W, 3)
    p.wait()


def overlay(pick=None):
    """the signs, labels and captions over the clean frames: the finished cut (or, pick: those frames shown, as PNG)"""
    js, seq = sequence()
    full_slow = next(k for k, (T, s) in enumerate(tl) if abs(s - RC.SLOW) < 1e-9)
    k_ramp = next(k for k in range(full_slow, len(tl)) if tl[k][1] < RC.SLOW - 1e-12)   # the second slowdown
    cq = cues(js, seq) if os.environ.get("LT_CAPTIONS", "1") == "1" else []     # LT_CAPTIONS=0: the film's own
    src, cache, last = clean_frames(), {}, -1
    sink = None if pick else MC.Sink(os.environ.get("LT_NAME", "history_race_light"))
    os.makedirs("/home/user/lot/out/test_lt", exist_ok=True)
    laps_prev, since = 0, 99
    for o, (j, clock, phase) in enumerate(seq):
        while last < j:
            last += 1; cache[last] = next(src)
        for x in [x for x in cache if x < j - 90]: del cache[x]
        k, extra, s, _ = js[j]
        laps = max(0, passes_seen(k, extra) - 1)
        since = 0 if laps != laps_prev else since + 1
        laps_prev = laps
        if pick is not None and o not in pick: continue
        fr = labels(cache[j].copy(), k, extra)
        amt = min(1.0, math.log(1 / s) / math.log(48.0)) if s < 0.99 else 0.0
        stopped = phase is not None
        fr = SM.badge(SM.tint(fr, amt), s, clock, top=1000.0, stopped=stopped)
        if passes_seen(k, extra) >= 1:
            fr = lap_counter(fr, laps, flash=max(0.0, 1 - since / 8), alpha=1.0 - MP.white(k))
        if phase is None:
            fr = SM.slam(fr, 48, k - full_slow + 4, sub="so we can see the planes")
            fr = SM.slam(fr, 1000, k - k_ramp, n_hold=12, n_out=8, sub="so we can see Parker")   # gone before time stops
        else:
            n, part, m = phase
            if part == "pre":
                fr = SM.banner(fr, "TIME STOPPED", "a million times slower, to watch light set off" if n == 0
                               else "a million times slower: Parker stands still, light doesn't", m, n_hold=40, n_out=10)
            elif part == "post":
                h_post = HOLDS[n][1]
                L = min(h_post, RESUME)                  # the last L frames before the race takes up again
                fr = SM.banner(fr, "BACK TO REAL SPEED" if n == 0 else "BACK TO 1,000× SLOWER",
                               "the race carries on where it stopped" if n == 0 else "Parker carries on to the finish",
                               m - (h_post - L), mark="play", n_in=5, n_hold=L - 13, n_out=8, size=96)
        fr = CAP.draw(fr, *CAP.at(cq, o))
        if pick is not None:
            cv2.imwrite(f"/home/user/lot/out/test_lt/o{o:04d}.png", fr[..., ::-1])
        else:
            sink.add([fr])
        if (o + 1) % 100 == 0: print("overlay", o + 1, "of", len(seq), flush=True)
    if pick is None:
        lost = [CAP.draw(f, *CAP.at(cq, len(seq) + i)) for i, f in enumerate(MP.PC.lost(1.2))]
        after = lost + MP.PC.card("Light went round the world 17 times", "while the Wright Flyer crossed the picture.", 4.5)
        sink.add(after)
        sink.close(MC.mic(audio_seq(js, seq, len(after)), 1.6))


def reclean(old_last):
    """after the last pass has moved (from frame old_last to K_LAST): only the frames that change composited again
    (the new frozen stretch, and the race's frames whose lines or caught streaks differ), the rest taken from the
    clean file as it is"""
    from multiprocessing import Pool
    import subprocess
    MP.LIGHT = layer
    opt, _, ofrozen = passes(old_last)
    old, new = jobs(opt, ofrozen), jobs()
    dt = FREEZE / F.FPS

    def key(k, extra):
        if extra is None: return ("race", k)
        return ("frozen", k, extra[0], round((extra[1] - seen_at(np.array([XA]), Z0 + DZ * extra[0])[0]) / dt))

    def state(k, pt, frozen):
        T, slow = tl[k]
        seen = [n for n, Tn in enumerate(pt) if Tn <= T + 1e-12]
        return seen, [n for n in seen if n not in frozen and abs(pt[n] - T) <= 0.005 * slow]

    where = {key(k, e): i for i, (k, e, s, c) in enumerate(old)}
    src = []
    for k, e, s, c in new:
        i = where.get(key(k, e))
        if i is not None and e is None and state(k, opt, ofrozen) != state(k, PASS_T, FROZEN): i = None
        if k >= int(os.environ.get("LT_FORCE_FROM", 10 ** 6)): i = None     # these frames composited again anyway
        src.append(i)
    todo = [j for j, i in enumerate(src) if i is None]
    print("composite again:", len(todo), "of", len(new), flush=True)
    with Pool(int(os.environ.get("NPROC", "3"))) as pool:
        made = dict(zip(todo, pool.map(MP.frame, [(new[j][0], new[j][1]) for j in todo], chunksize=1)))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C.W}x{C.H}",
                          "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "fast", "-pix_fmt",
                          "yuv420p", CLEAN + ".part.mp4"], stdin=subprocess.PIPE)
    reader, cache, last = clean_frames(), {}, -1
    need = {}
    for j, i in enumerate(src):
        if i is not None: need[i] = need.get(i, 0) + 1
    for j, i in enumerate(src):
        if i is None:
            fr = made.pop(j)
        else:
            while last < i:
                last += 1; f = next(reader)
                if last in need: cache[last] = f
            fr = cache[i]
            need[i] -= 1
            if not need[i]: del cache[i]
        p.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
    p.stdin.close(); p.wait()
    for _ in reader: pass
    os.replace(CLEAN + ".part.mp4", CLEAN)
    print("reclean done", len(new), flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["reclean"]:     # reclean <old last-pass frame> (LT_FORCE_FROM=k)
    reclean(int(sys.argv[2]))
elif __name__ == "__main__" and sys.argv[1:2] == ["clean"]:
    clean()
elif __name__ == "__main__" and sys.argv[1:2] == ["seq"]:          # where things fall in the frames shown
    js, seq = sequence()
    print(len(seq), "frames shown")
    for o, (k0, k1, text) in enumerate(cues(js, seq)): print(k0, k1, text)
elif __name__ == "__main__":
    overlay([int(a) for a in sys.argv[1:]] or None)
