"""Other cuts of the film, from the same renders.
  python make_cuts.py realtime -> history_realtime.mp4: every machine at real speed, one after the other, no slow motion
  python make_cuts.py pegged   -> history_pegged16.mp4: each machine at real speed, then its replay, every replay at the
                                  same rate (16 times slower, a camera at 400 frames a second)
  REPLAY48=1 python make_cuts.py slow48 -> history_slow48.mp4: the Wright Flyer at real speed, then every other machine
                                  only in slow motion, all 48 times slower (a camera at 1,200 frames a second; the replay
                                  renders at that rate, flight.REPLAY_SUF), with the slow-motion sign (slowmo.py)

Real time at 25 frames a second hides how much faster the fast ones are: the F-35 is in the picture for 89 ms (two or
three frames), the X-15 for 23 ms (one). Without bending the physics, these cuts make the difference visible:
* a 180-degree shutter (1/50 s, the usual for 25 fps), so each machine smears as far as it really travels while the
  shutter is open: the X-1 7 m, the F-35 11 m, the X-15 40 m (wider than the picture) (render_flyby.py EXPO=0.02);
* how long each was in the picture, shown once it has gone;
* the shake and the sound scaled to the pressure each makes at the camera: the X-1's boom ~6 kPa (169 dB), the F-35's
  ~12 kPa (175 dB), the X-15's blast ~20 kPa (180 dB); the camera's microphone overloads harder the louder it is.
The pegged replays reuse the replay renders: the Spitfire's are already at 16x; the X-1's every second frame, the
F-35's every third; the X-15's (rendered 200x slower) three consecutive frames averaged, the exposure a 400 fps camera
would give (1/1,600 s). At 16x the Parker probe is still a single white frame."""
import cv2, numpy as np, os, subprocess, sys, shutil, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, flight as F, fly_fx as FX, fly_sound as S, make_fly as MF, make_film as MFi, parker_comp as PC
import slowmo as SM
R_ = "/home/user/lot/render"
OUT = "/home/user/lot/out"
WHITE, DIM = MF.WHITE, MF.DIM
ORDER = ["wright", "spitfire", "x1", "f35", "x15"]
KPA = {"x1": 5.9, "f35": 11.8, "x15": 20.0}             # peak pressure at the camera
SHAKE_PER_KPA = 1.5                                       # px of jolt per kPa on the camera
PEG = 1 / 16
EXPO_RT = 0.02


import functools


@functools.lru_cache(maxsize=None)
def crossing(name):
    """seconds the machine itself (not its shadow) is in the picture, and when it leaves"""
    m = F.MACHINES[name]
    L = 90.0 / m["speed"]
    ts = np.linspace(-L, L, 40001)
    vis = []
    for t in ts:
        c, _ = F.pose(name, t)
        uv, z = C_project(F._corners(c, (m["box"][0] * 0.9, m["box"][1] * 0.3, m["box"][2] * 0.6)))
        vis.append(uv[:, 0].max() > 0 and uv[:, 0].min() < C.W)
    i = np.where(vis)[0]
    return ts[i[-1]] - ts[i[0]], ts[i[-1]]


def C_project(P):
    import camera as Cam
    return Cam.project(P)


def fmt_s(s):
    return f"{s:.1f} s" if s >= 1 else (f"{s:.2f} s" if s >= 0.1 else f"{s:.3f} s")


_orig_fly = MF.fly


def fly_s50(name, n, replay=False):
    """the real-time renders with the 180-degree shutter where there are any"""
    p = f"{R_}/fly_{name}_s50/p{n:04d}.npz"
    if not replay and os.path.exists(p):
        return C.props(p)
    return _orig_fly(name, n, replay)


MF.fly = fly_s50


def finish(img, seed, lines, label_alpha, shake_t, amp):
    img = C.bloom(img)
    if shake_t >= 0 and amp > 0:
        img = C.shake(img, shake_t, amp=amp, decay=4.0, seed=3)
    img = C.grain(img, seed)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return C.label(out, lines, label_alpha, corner="bl") if lines else out


def rt_frames(name, seed0=0):
    """the machine at real speed: the shot's frames, recomposited"""
    dur, t_exit = crossing(name)
    out = []
    amp = SHAKE_PER_KPA * KPA.get(name, 0)
    for n, t in F.frames(name):
        img = MF.compose(name, n, t, C.plate_loop(n), expo=EXPO_RT)
        lines = MF.lines(name)
        a = np.clip((t - t_exit) / 0.3, 0, 1)
        if a > 0: lines = lines + [(f"in the picture for {fmt_s(dur)}", 26, "Medium", DIM)]
        la = [min(1.0, n / 8)] if True else None
        out.append(finish(img, seed0 + 7000 + n, lines, min(1.0, n / 8), FX.shake_t(name, t) if name in KPA else -1, amp))
    return out


def peg_times(name):
    """(t, [replay frame indices to average]) at 16x slower, from the replay's own frames, while they allow it"""
    tl = F.replay(name)
    ts = np.array([t for t, _ in tl])
    step = PEG / F.FPS
    out, t = [], ts[0]
    avg = 1
    while True:
        i = int(np.argmin(np.abs(ts - t)))
        # tight while the machine is in the picture; once it has gone only the dust moves, slowly, so the nearest
        # replay frame will do (the camera's shake, worked out for every frame's own time, stays smooth) until the
        # shock has reached the camera and shaken it for a moment
        if F.in_view(name, t, 0.0):
            if abs(ts[i] - t) > 0.05 * step: break
        else:
            t_end = (-FX.shake_t(name, 0.0) + 0.15) if name in KPA else crossing(name)[1] + 0.06
            if t > t_end or abs(ts[i] - t) > 2.5 * step or t > ts[-1]: break
        idx = [j for j in range(i - avg // 2, i - avg // 2 + avg) if 0 <= j < len(ts)]
        out.append((t, idx))
        t += step
    return out


def exposure_blur(name, p):
    """smear the render along the flight to the exposure a 400 fps camera would give (1/1,600 s), when it was rendered
    for a faster one (the X-15's replay, at 200x slower): a continuous blur, as the real exposure would be"""
    if p is None or F.REPLAY[name] >= PEG / 3: return p
    m = F.MACHINES[name]
    px = m["speed"] * (0.01 * PEG - 0.01 * F.REPLAY[name]) * 1250.0 / F.FLY_Y
    k = max(1, int(round(px)))
    ker = np.ones((1, k), np.float32) / k
    return dict(p, rgb=cv2.filter2D(p["rgb"], -1, ker), a=cv2.filter2D(p["a"], -1, ker)[..., None]
                if p["a"].ndim == 3 else cv2.filter2D(p["a"], -1, ker))


def peg_frames(name, seed0=0):
    out = []
    amp = SHAKE_PER_KPA * KPA.get(name, 0)
    t_first = F.replay(name)[0][0]
    tl = F.replay(name)
    for k, (t, idx) in enumerate(peg_times(name)):
        bg = C.plate_loop(60 + int((t - t_first) * F.FPS))
        prev = MF.fly
        MF.fly = lambda nm, n, replay=False: exposure_blur(nm, prev(nm, n, replay))
        try:
            imgs = [MF.compose(name, j, tl[j][0], bg, replay=True, expo=0.01 * PEG) for j in idx]
        finally:
            MF.fly = prev
        img = sum(imgs) / len(imgs)
        lines = MF.lines(name) + [(f"replay, {1 / PEG:.0f}× slower", 26, "Medium", DIM)]
        out.append(finish(img, seed0 + 9000 + k, lines, min(1.0, k / 6), FX.shake_t(name, t) if name in KPA else -1, amp))
    return out


def peg_audio(name, n_frames, seed=7):
    """the real-time sound played back 16 times slower, over the pegged frames"""
    rt = S.shot(name, seed=seed)                   # on the shot's clock
    n0 = round(F.MACHINES[name]["pre"] * F.FPS)
    ts = [t for t, _ in peg_times(name)][:n_frames]
    T = np.arange(int(n_frames / F.FPS * S.SR)) / S.SR
    tk = np.interp(T * F.FPS, np.arange(len(ts)), ts) + n0 / F.FPS
    out = np.stack([np.interp(tk * S.SR, np.arange(len(rt)), rt[:, c]) for c in (0, 1)], 1)
    rum = S.filt(np.random.default_rng(17).standard_normal((len(T), 2)), "bandpass", [30, 110], 2)
    rum *= 0.012 / (np.sqrt((rum ** 2).mean()) + 1e-9)
    return out + rum


def slowed(frames8, tl, t0, lead=0, slam_at=0):
    """the slow-motion sign on finished frames: tl their [(t, slow)], t0 when the clock starts; the first `lead`
    frames hold the first moment while the sign comes up"""
    out = []
    for k, im in enumerate(frames8):
        t, slow = tl[max(0, k - lead)] if k >= lead else (tl[0][0], tl[0][1])
        amt = min(1.0, math.log(1 / slow) / math.log(48.0)) if slow < 0.99 else 0.0
        im = SM.badge(SM.tint(im, amt), slow, t - t0)
        out.append(SM.slam(im, 1 / tl[0][1], k - slam_at) if tl[0][1] < 0.99 else im)
    return out


def r48(name, seed0=0, lead=26):
    """the machine only in slow motion (REPLAY48=1: its replay, 48x slower while it crosses, then ramping back to real
    speed while its dust settles), held a moment first while the sign comes up; and its sound, played back as slowly"""
    tl = F.replay(name)
    assert abs(tl[0][1] - 1 / 48) < 1e-9, "run with REPLAY48=1"
    amp = SHAKE_PER_KPA * KPA.get(name, 0)
    t_first = tl[0][0]
    raw = []
    for n, (t, slow) in enumerate(tl):
        bg = C.plate_loop(60 + int((t - t_first) * F.FPS))
        img = MF.compose(name, n, t, bg, replay=True, expo=0.01 * slow)
        raw.append(finish(img, seed0 + 9000 + n, MF.lines(name), 1.0, FX.shake_t(name, t) if name in KPA else -1, amp))
    frames = slowed([raw[0]] * lead + raw, tl, t_first, lead=lead)
    return frames, slow_audio(name, [t for t, _ in tl], lead)


def slow_audio(name, ts, lead):
    """the real-time sound played back as the frames go (ts: each frame's flight time), after `lead` frames of hush"""
    rt = S.shot(name, seed=7)
    n0 = round(F.MACHINES[name]["pre"] * F.FPS)
    n = len(ts)
    T = np.arange(int(n / F.FPS * S.SR)) / S.SR
    tk = np.interp(T * F.FPS, np.arange(n), ts) + n0 / F.FPS
    a = np.stack([np.interp(tk * S.SR, np.arange(len(rt)), rt[:, c]) for c in (0, 1)], 1)
    a = S.filt(a, "highpass", 25, 4)              # slowed 48x most of it falls below hearing: keep what can be heard
    a = np.concatenate([np.zeros((int(lead / F.FPS * S.SR), 2)), a])
    rum = S.filt(np.random.default_rng(17).standard_normal((len(a), 2)), "bandpass", [30, 110], 2)
    rum *= 0.012 / (np.sqrt((rum ** 2).mean()) + 1e-9)
    return a + rum


def parker_rt():
    """two seconds of road, the white frame, then nothing (from the finale's own frames)"""
    n0 = round(F.MACHINES["parker"]["pre"] * F.FPS)
    fr = f"{OUT}/frames_parker"
    ims = [cv2.imread(f"{fr}/f{k:04d}.png")[..., ::-1] for k in range(n0 + 1)]
    return ims + PC.lost(1.4)


class Sink:
    """frames straight into the encoder, a segment at a time (frame files for a whole cut would fill the disk)"""
    def __init__(self, name):
        self.name, self.n = name, 0
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C.W}x{C.H}",
                                   "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                                   f"{OUT}/{name}_video.mp4"], stdin=subprocess.PIPE)

    def add(self, frames):
        for im in frames:
            self.p.stdin.write(np.ascontiguousarray(im, np.uint8).tobytes()); self.n += 1

    def close(self, audio):
        self.p.stdin.close(); self.p.wait()
        S.write(f"{OUT}/{self.name}.wav", audio)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{OUT}/{self.name}_video.mp4", "-i", f"{OUT}/{self.name}.wav",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", f"{OUT}/{self.name}.mp4"], check=True)
        os.remove(f"{OUT}/{self.name}_video.mp4")
        print(self.name, self.n / 25, "s")


def write(frames, audio, name):
    fr = f"{OUT}/frames_{name}"
    shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
    for k, im in enumerate(frames):
        cv2.imwrite(f"{fr}/f{k:05d}.png", im[..., ::-1])
    S.write(f"{OUT}/{name}.wav", audio)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{fr}/f%05d.png", "-i", f"{OUT}/{name}.wav",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    f"{OUT}/{name}.mp4"], check=True)
    print(name, len(frames) / 25, "s")


def mic(x, gain):
    """the camera's microphone: the mix at one gain, overloading (softly) on what is too loud for it"""
    return np.tanh(x * gain * 1.2) / 1.2 * 0.95


def segment_audio(name, n):
    a = S.shot(name)
    m = int(n / F.FPS * S.SR)
    return np.pad(a, ((0, max(0, m - len(a))), (0, 0)))[:m]


def parker_audio(n_road, n_after):
    sr = S.SR
    rng = np.random.default_rng(21)
    a = S.wind(int(n_road / F.FPS * sr), rng, 0.035)
    m = int(0.03 * sr); tt = np.arange(m) / sr
    crack = 9.0 * (1 - tt / 0.03) * (1 + 0.6 * rng.standard_normal(m))
    return np.concatenate([a, np.stack([crack, crack], 1), np.zeros((int(n_after / F.FPS * sr) - m, 2))])


def slow48():
    MFi.TITLE = ("120 years of speed,", "flying past you,", "48 times slower")
    out = Sink("history_slow48")
    title = MFi.title_frames(4.5)
    out.add(title)
    audio = [S.wind(int(len(title) / 25 * S.SR), np.random.default_rng(1), 0.035)]
    f = rt_frames("wright")                    # too slow to slow down: at real speed, with the sign saying so
    ts = [(t, 1.0) for _, t in F.frames("wright")]
    out.add(slowed(f, ts, ts[0][0])); audio.append(segment_audio("wright", len(f)))
    for name in ORDER[1:]:
        fr, au = r48(name)
        out.add(fr); audio.append(au * 0.6)
        print(name, "done", flush=True)
    card = PC.card(*MFi.END_X15, 4.5)
    out.add(card); audio.append(np.zeros((int(len(card) / 25 * S.SR), 2)))
    out.close(mic(np.concatenate(audio), 1.6))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "slow48":
        slow48(); sys.exit()
    if which == "realtime":                             # with the sign: REAL SPEED, and the clock
        out = Sink("history_realtime")
        title = MFi.title_frames()
        out.add(title)
        audio = [S.wind(int(len(title) / 25 * S.SR), np.random.default_rng(1), 0.035)]
        for name in ORDER:
            f = rt_frames(name)
            ts = [(t, 1.0) for _, t in F.frames(name)]
            out.add(slowed(f, ts, ts[0][0])); audio.append(segment_audio(name, len(f)))
        n0 = round(F.MACHINES["parker"]["pre"] * F.FPS)
        pr = parker_rt()
        out.add(slowed(pr, [(k / 25, 1.0) for k in range(len(pr))], 0.0))
        audio.append(parker_audio(n0 + 1, len(pr) - n0 - 1))
        card = PC.lost(0.4) + PC.card("At 430,000 mph, nothing in this picture survives.", "Light is still 1,560 times faster.", 4.5)
        out.add(card); audio.append(np.zeros((int(len(card) / 25 * S.SR), 2)))
        out.close(mic(np.concatenate(audio), 1.6))
        sys.exit()
    title = MFi.title_frames()
    frames = list(title)
    audio = [S.wind(int(len(title) / 25 * S.SR), np.random.default_rng(1), 0.035)]
    for name in ORDER:
        f = rt_frames(name)
        frames += f; audio.append(segment_audio(name, len(f)))
        if which == "pegged" and name != "wright":
            p = peg_frames(name)
            frames += p; audio.append(peg_audio(name, len(p)) * 0.6)
    n0 = round(F.MACHINES["parker"]["pre"] * F.FPS)
    pr = parker_rt()
    frames += pr; audio.append(parker_audio(n0 + 1, len(pr) - n0 - 1))
    if which == "pegged":                               # the probe at 16x: still one white frame, then nothing
        road = cv2.imread(f"{OUT}/frames_parker/f{n0 - 1:04d}.png")[..., ::-1].copy()
        lines = [("430,000 mph", 64, "Bold", WHITE), ("Parker Solar Probe, 2024", 30, "Medium", DIM),
                 ("its top speed, here at sea level", 24, "Medium", DIM), ("replay, 16× slower", 26, "Medium", DIM)]
        road = C.label(road, lines, 1.0, corner="bl")
        white = np.zeros_like(road); white[C.TOP:C.TOP + C.PH] = 255
        peg = [road] * 12 + [white] + PC.lost(1.2)
        frames += peg; audio.append(parker_audio(12, len(peg) - 12) * 0.5)
    card = PC.lost(0.4) + PC.card("At 430,000 mph, nothing in this picture survives.", "Light is still 1,560 times faster.", 4.5)
    frames += card; audio.append(np.zeros((int(len(card) / 25 * S.SR), 2)))
    mix = np.concatenate(audio)
    write(frames, mic(mix, 1.6), "history_" + ("realtime" if which == "realtime" else "pegged16"))
