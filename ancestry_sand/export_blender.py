"""Export the whole film for a full Cycles render in Blender (blender_film.py then builds the .blend).

    python export_blender.py [--no-hud | --hud-only] [--scale 2] [--workers 4]   # -> out/blender_film/

grains.npz   every grain the film shows, as records: where it sits, its radius, the frames it is visible
             (tb <= frame < td), and its colour, easing from c0 to c1 (c = c1 + (c0 - c1) * 0.82^(frame - tc));
             new grains on top of a pile swell in from frame ts. Geometry Nodes keeps the records alive at
             the current frame, so the .blend renders the film with no Python at render time.
             Only grains that can be seen are kept: the top layers of each pile and the sides of towers.
flights.npz  every flying grain: launch frame, flight length, colour, size, and its path sampled at
             K points (positions in Blender units, plus whether each step stays on the map).
hud.mov      the year counter, legend, labels and closing highlights as a transparent video
             (QuickTime Animation, alpha), laid over the render in the .blend's "Edit" scene (the title is
             a text strip there).
Colours are linear RGB. Units: 1 Blender unit = 100 map px.
"""
import os, sys, time, subprocess
import numpy as np
import cv2
import story as S
import sand as Sd
from film import Painter, FPS, COL, GRAIN_DZ, hash01, W, H

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.environ.get("SAND_BLENDER_OUT", os.path.join(HERE, "out", "blender_film"))
KS = 40                       # samples along each flight path
EASE = 0.82                   # colour easing per frame (the compositor's 0.18 step)


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


class Records:
    def __init__(self, cap=2_000_000):
        self.n = 0
        self.a = dict(cell=np.zeros(cap, np.int32), lev=np.zeros(cap, np.int16), c0=np.zeros((cap, 3), np.float32),
                      c1=np.zeros((cap, 3), np.float32), tb=np.zeros(cap, np.float32), td=np.zeros(cap, np.float32),
                      tc=np.zeros(cap, np.float32), ts=np.zeros(cap, np.float32))

    def add(self, **kw):
        m = len(kw["cell"])
        if self.n + m > len(self.a["cell"]):
            for k, v in self.a.items():
                self.a[k] = np.concatenate([v, np.zeros_like(v)])
        ids = np.arange(self.n, self.n + m)
        for k, v in kw.items():
            self.a[k][ids] = v
        self.n += m
        return ids

    def colour_at(self, ids, f):
        a = self.a
        w = EASE ** np.maximum(f - a["tc"][ids], 0)[:, None]
        return a["c1"][ids] + (a["c0"][ids] - a["c1"][ids]) * w


def hud_rgba(pt, sim, s):
    """The HUD as straight-alpha RGBA: drawn once over black and once over white."""
    B = np.zeros((H * pt.k, W * pt.k, 3), np.float32)
    Wt = np.full((H * pt.k, W * pt.k, 3), 255.0, np.float32)
    pt.hud(B, sim, s, title=False)
    pt.hud(Wt, sim, s, title=False)
    a = np.clip(1 - (Wt - B).mean(2) / 255.0, 0, 1)
    a = np.maximum(a, np.clip(B.max(2) / 255.0, 0, 1))          # additive glows (the highlight rings)
    rgb = np.clip(B / np.maximum(a, 1e-4)[..., None], 0, 255)
    return np.dstack([rgb[..., ::-1], a[..., None] * 255]).astype(np.uint8)    # BGRA for cv2


def hud_chunk(job):
    """HUD frames f0..f1-1 at scale k into one QuickTime Animation file (re-simulating from the start)."""
    f0, f1, k, path = job
    sim, pt = Sd.Sand(), Painter(hud_scale=k)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
                           "-s", f"{W * k}x{H * k}", "-r", str(FPS), "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
                           "-g", "300", path], stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(f1):
        s = f / FPS
        sim.step(s)
        if f >= f0:
            pt.set_camera(s)
            ff.stdin.write(hud_rgba(pt, sim, s).tobytes())
            if (f - f0) % 300 == 0:
                print(f"  hud [{f0}-{f1}] frame {f}  {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close(); ff.wait()
    return path


def export_hud(k=1, workers=4):
    """hud.mov: the HUD for the whole film, k x 1920x1080, rendered in parallel chunks and joined."""
    from multiprocessing import Pool
    n = int(os.environ.get("SAND_EXPORT_FRAMES", round(S.DURATION * FPS)))
    edges = np.linspace(0, n, workers + 1).round().astype(int)
    jobs = [(int(edges[i]), int(edges[i + 1]), k, os.path.join(OUTD, f"hud_part{i}.mov")) for i in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        parts = pool.map(hud_chunk, jobs)
    lst = os.path.join(OUTD, "hud_parts.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{os.path.basename(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    os.path.join(OUTD, "hud.mov")], check=True)
    for p in parts + [lst]:
        os.remove(p)
    print(f"wrote hud.mov ({W * k}x{H * k}) in {time.time() - t0:.0f}s")


def main():
    os.makedirs(OUTD, exist_ok=True)
    sim, pt = Sd.Sand(), Painter()
    n = int(os.environ.get("SAND_EXPORT_FRAMES", round(S.DURATION * FPS)))     # (fewer: quick tests)
    R = Records()
    tint = pt.cell_tint.ravel()
    pkey = np.zeros(0, np.int64); pci = np.zeros(0, np.int64); prec = np.zeros(0, np.int64)
    flights = dict(t0=[], dur=[], col=[], size=[], pos=[], ok=[])
    uk = np.linspace(0.0, 1.0, KS)
    t0 = time.time()
    for f in range(n):
        s = f / FPS
        sim.step(s)
        # flying grains: sample each new batch's paths once
        for p in sim.particles:
            if p.get("_exported"):
                continue
            p["_exported"] = True
            m = len(p["t0"])
            idx = np.arange(m)
            X = np.zeros((m, KS, 3), np.float32); ok = np.zeros((m, KS), bool); pc = np.zeros((m, KS), np.int64)
            for k, u in enumerate(uk):
                x, y, z, okk, piece = Painter.flight_pos(p, idx, np.full(m, u))
                X[:, k] = np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, (z + 2.0) / 100.0], 1)
                ok[:, k], pc[:, k] = okk, piece
            seg = np.zeros((m, KS), bool)
            seg[:, :-1] = ok[:, :-1] & ok[:, 1:] & (pc[:, :-1] == pc[:, 1:])
            flights["t0"].append(p["t0"] * FPS); flights["dur"].append(p["dur"] * FPS)
            flights["col"].append(srgb_to_linear(COL[p["col"]])); flights["size"].append(p["size"])
            flights["pos"].append(X); flights["ok"].append(seg)
        # resting grains: what the compositor would draw this frame, as records
        ph = pt.prev_hall.copy()
        had = pt.vis is not None
        pt.update_disp(sim)
        V = pt.vis
        if V is None:
            key = np.zeros(0, np.int64); cell = lev = ci = key
        else:
            key, cell, lev = V["key"], V["cell"], V["lev"]
            ci = sim.stack[cell, lev].astype(np.int64)
        if len(pkey) and len(key):
            j = np.clip(np.searchsorted(pkey, key), 0, len(pkey) - 1)
            hit = pkey[j] == key
        else:
            j = np.zeros(len(key), np.int64); hit = np.zeros(len(key), bool)
        same = hit & (pci[j] == ci) if len(pkey) else hit
        chg = hit & ~same
        new = ~hit
        rec = np.zeros(len(key), np.int64)
        rec[same] = prec[j[same]]
        cont = np.zeros(len(pkey), bool)
        cont[j[same]] = True
        R.a["td"][prec[~cont]] = f
        tgt = srgb_to_linear(COL[ci] * tint[cell][:, None]) if len(key) else np.zeros((0, 3), np.float32)
        if chg.any():
            old = prec[j[chg]]
            rec[chg] = R.add(cell=cell[chg], lev=lev[chg], c0=R.colour_at(old, f - 1), c1=tgt[chg],
                             tb=np.full(chg.sum(), f), tc=np.full(chg.sum(), f - 1), ts=R.a["ts"][old])
        if new.any():
            fresh = new & (lev >= ph[cell]) & had
            rec[new] = R.add(cell=cell[new], lev=lev[new], c0=tgt[new], c1=tgt[new], tb=np.full(new.sum(), f),
                             tc=np.full(new.sum(), f), ts=np.where(fresh[new], f, -1e6))
        R.a["td"][rec] = n                     # open until closed
        pkey, pci, prec = key, ci, rec
        if f % 300 == 0:
            print(f"frame {f}/{n}  records {R.n}  visible {len(key)}  {time.time() - t0:.0f}s", flush=True)

    m = R.n
    a = {k: v[:m] for k, v in R.a.items()}
    cell, lev = a["cell"].astype(np.int64), a["lev"].astype(np.int64)
    jx, jy, jr = hash01(cell, lev, 1), hash01(cell, lev, 2), hash01(cell, lev, 3)
    x = Sd.CELL_XY[cell, 0] + (jx - 0.5) * 1.3
    y = Sd.CELL_XY[cell, 1] + (jy - 0.5) * 1.3
    pos = np.stack([x / 100.0 - 9.6, 5.4 - y / 100.0, lev * GRAIN_DZ / 100.0], 1).astype(np.float32)
    np.savez_compressed(os.path.join(OUTD, "grains.npz"), pos=pos, rad=((1.35 + 0.35 * jr) / 100.0).astype(np.float32),
                        c0=a["c0"], c1=a["c1"], tb=a["tb"], td=a["td"], tc=a["tc"], ts=a["ts"], frames=n,
                        eaf=np.asarray(Sd.META["EAF"]["anchor"], np.float32))
    fl = {k: np.concatenate(v).astype(np.float32 if k != "ok" else bool) for k, v in flights.items()}
    np.savez_compressed(os.path.join(OUTD, "flights.npz"), ks=KS, **fl)
    print(f"wrote {m} grain records, {len(fl['t0'])} flying grains in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    arg = lambda name, d: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d
    if "--hud-only" not in sys.argv:
        main()
    if "--no-hud" not in sys.argv:
        export_hud(k=arg("--scale", 1), workers=arg("--workers", 4))
