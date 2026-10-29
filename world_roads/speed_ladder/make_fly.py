"""Composite the flybys over the plate and join them with their sound.
  python make_fly.py wright [...] -> /home/user/lot/out/fly_<name>.mp4"""
import cv2, numpy as np, os, subprocess, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, flight as F, fly_fx as FX
R_ = "/home/user/lot/render"
OUT = "/home/user/lot/out"
WHITE, DIM = (255, 255, 255), (214, 220, 228)
FEATHER = 80                     # px over which a partial render blends into the static frame at its edges


def fly(name, n, replay=False):
    """the CG for frame n: the render where the machine is in the picture (feathered into the static frame at the edges
    of the part rendered), else the static frame"""
    st = C.props(f"{R_}/fly_{name}/static.npz")
    p = f"{R_}/fly_{name}{'_replay' if replay else ''}/p{n:04d}.npz"
    if not os.path.exists(p): return st
    d = np.load(p)
    out = C.props(p)
    if "rect" in d:
        x0, y0, x1, y1 = [int(v) for v in d["rect"]]
        h, w = out["a"].shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.minimum.reduce([xx - x0 if x0 > 0 else np.full_like(xx, 1e4), x1 - 1 - xx if x1 < w else np.full_like(xx, 1e4),
                                  yy - y0 if y0 > 0 else np.full_like(yy, 1e4), y1 - 1 - yy if y1 < h else np.full_like(yy, 1e4)])
        k = np.clip(dist / FEATHER, 0, 1)[..., None]
        for key in ("rgb", "a"):
            out[key] = out[key] * k + st[key] * (1 - k)
    return out


def finish(img, seed, lines, label_alpha=1.0, fl=0.0, shake_t=-1.0, shake_amp=10.0):
    img = C.flash(img, fl)
    img = C.bloom(img)
    img = C.shake(img, shake_t, amp=shake_amp, seed=3)      # one shake, continuous from frame to frame
    img = C.grain(img, seed)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return C.label(out, lines, label_alpha, corner="bl") if lines else out


# a word on how the speed was set where the real record was flown high up, in thinner air
NOTE = {"x1": "Mach 1.06, its record (700 mph at 43,000 ft); here at sea level",
        "x15": "its record speed (flown 20 miles up), here at sea level"}


def lines(name):
    m = F.MACHINES[name]
    year, title = m["title"].split(" ", 1)
    out = [(f"{m['mph']:,} mph", 64, "Bold", WHITE), (f"{title}, {year}", 30, "Medium", DIM)]
    if name in NOTE: out.append((NOTE[name], 24, "Medium", DIM))
    return out


def replay_lines(name, slow):
    s = f"replay, {1 / slow:,.0f}× slower" if slow < 0.95 else "replay, real speed"
    return lines(name) + [(s, 26, "Medium", DIM)]


def compose(name, n, t, bg, replay=False, expo=0.01):
    """plate, dust shadow, CG, dust, bent by the shocks"""
    dl = FX.dust_layer(name, n, replay)
    if dl is not None: bg = bg * (1 - C.SUN_FRACTION * (1 - dl[2]))
    img = C.over_props(bg, C.plate_match(fly(name, n, replay), 5000 + n + (100000 if replay else 0)))
    glow = FX.wake_glow(name, t, expo)
    if glow is not None: img = img + glow
    if dl is not None: img = img * dl[1] + dl[0]
    return FX.warp(img, FX.displacement(name, t))


def one(job):
    name, n, plate0, path = job
    t = F.frames(name)[n][1]
    img = compose(name, n, t, C.plate_loop(plate0 + n))
    out = finish(img, 7000 + n, lines(name), label_alpha=min(1.0, n / 8), shake_t=FX.shake_t(name, t),
                 shake_amp=FX.FX.get(name, {}).get("shake", 0.0))
    cv2.imwrite(path, out[..., ::-1])


def one_replay(job):
    """the replay: the plate frozen while time crawls (it moves on in real time)"""
    name, n, t, slow, t_first, plate0, path = job
    img = compose(name, n, t, C.plate_loop(plate0 + int((t - t_first) * F.FPS)), replay=True, expo=0.01 * slow)
    out = finish(img, 9000 + n, replay_lines(name, slow), label_alpha=min(1.0, n / 6), shake_t=FX.shake_t(name, t),
                 shake_amp=FX.FX.get(name, {}).get("shake", 0.0))
    cv2.imwrite(path, out[..., ::-1])


def shot(name, plate0=0):
    fr = f"{OUT}/frames_{name}"
    os.makedirs(fr, exist_ok=True)
    jobs = [(name, n, plate0, f"{fr}/f{n:04d}.png") for n, _ in F.frames(name)]
    with Pool(3) as p: p.map(one, jobs)
    if name in F.REPLAY:                                     # the replay follows straight on
        tl = F.replay(name)
        k0 = len(jobs)
        rj = [(name, n, t, slow, tl[0][0], plate0 + 60, f"{fr}/f{k0 + n:04d}.png") for n, (t, slow) in enumerate(tl)]
        with Pool(3) as p: p.map(one_replay, rj)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly_sound.py"), name], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{fr}/f%04d.png", "-i", f"{OUT}/fly_{name}.wav",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    f"{OUT}/fly_{name}.mp4"], check=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        shot(name)
