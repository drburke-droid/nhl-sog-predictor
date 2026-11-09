"""Export the film for Blender Cycles: the sand frame by frame, the lifted grains and the on-screen text,
into cycles/cache/, ready for blender_cycles.py (which builds cycles/film_cycles.blend).

    python export_cycles.py                 # the whole film (about 30 min)
    python export_cycles.py --to 20         # the first 20 s (a quick test)

The simulation is the one film.py paints (sand.py), run once from the start. Writes:
  cycles/cache/static.npz        the grid: cell centres, neighbours, per-cell tint; colours; camera; timing
  cycles/cache/sand/NNN.npz      pile heights per cell, CHUNK frames per file (a full first frame, then
                                 only the cells that changed: about 60 a frame)
  cycles/cache/lifted.npz        every lifted grain: when, from where, toward which place, its cause
  cycles/overlay_fill.mp4        the on-screen text and titles, premultiplied colour (lossless RGB)
  cycles/overlay_matte.mp4       1 - their opacity, grey (lossless): the frame is multiplied by this,
                                 then the fill added
  cycles/narration.m4a           the voiceover (from out/<film>_narrated.mp4, when it has been made)
"""
import argparse, os, subprocess, time
import numpy as np

import story as S
import sand as Sd
import film as F
import camera as C
from proj import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "cycles")
CHUNK = 150
FPS = F.FPS


def ffmpeg_writer(path, gray=False):
    """Lossless H.264 (RGB, or one grey plane for the matte): the text layer is mostly still, so it stays
    small, and Blender's sequencer reads it back exactly."""
    enc = (["-c:v", "libx264", "-qp", "0", "-pix_fmt", "gray"] if gray else ["-c:v", "libx264rgb", "-qp", "0"])
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *enc, "-preset", "slow", "-g", "300", path],
                            stdin=subprocess.PIPE)


def overlay(painter, sim, s):
    """The film's text layer as premultiplied colour and opacity, by drawing it over black and over
    white: every text operation is 'over', so out = bg * (1 - a) + c."""
    blk = np.zeros((H, W, 3), np.float32)
    wht = np.full((H, W, 3), 255.0, np.float32)
    painter.hud(blk, sim, s)
    painter.hud(wht, sim, s)
    a = np.clip(1.0 - (wht - blk).mean(2, keepdims=True) / 255.0, 0.0, 1.0)
    fill = np.clip(blk, 0, 255).astype(np.uint8)
    matte = np.clip((1.0 - a[..., 0]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return fill, matte


def static(painter):
    loc, r, u, f = C.basis(C.WIDE)
    return dict(
        cell_xy=Sd.CELL_XY.astype(np.float32), nb=Sd.NB.astype(np.int32),
        tint=painter.cell_tint.ravel().astype(np.float32),
        sand_srgb=F.SANDCOL.astype(np.float32), cause_srgb=F.CCOL.astype(np.float32),
        cause_keys=np.array(S.CAUSE_KEYS),
        gs=np.float32(Sd.GS), grain_dz=np.float32(Sd.GRAIN_DZ), cell=np.int32(Sd.CELL),
        fps=np.int32(FPS), life=np.float32(Sd.LIFE), tg=np.float32(F.TG), chunk=np.int32(CHUNK),
        cam_loc=loc, cam_r=r, cam_u=u, cam_f=f, focal_px=np.float32(C.FOCAL_PX), lens=np.float32(C.LENS),
        width=np.int32(W), height=np.int32(H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", type=float, default=None, help="stop after this many seconds (a test)")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    n = int(round((args.to if args.to else S.DURATION) * FPS))
    cache = os.path.join(OUTD, "cache")
    os.makedirs(os.path.join(cache, "sand"), exist_ok=True)

    sim = Sd.Sand()
    painter = F.Painter(sim=sim)
    st = static(painter)
    st["nframes"] = np.int32(n)
    np.savez_compressed(os.path.join(cache, "static.npz"), **st)

    if not args.no_overlay:
        wf = ffmpeg_writer(os.path.join(OUTD, "overlay_fill.mp4"))
        wm = ffmpeg_writer(os.path.join(OUTD, "overlay_matte.mp4"), gray=True)
    births, seen = [], set()
    base = prev = None
    idx, val, off = [], [], [0]
    t0 = time.time()
    for f in range(n):
        s = f / FPS
        sim.step(s)
        for p in sim.particles:                        # each batch of lifted grains, once, when it is born
            if id(p) not in seen:
                seen.add(id(p))
                births.append(p)
        h = sim.h.astype(np.int16)
        if f % CHUNK == 0:
            base, prev, idx, val, off = h.copy(), h.copy(), [], [], [0]
        else:
            ch = np.nonzero(h != prev)[0].astype(np.int32)
            idx.append(ch); val.append(h[ch]); off.append(off[-1] + len(ch))
            prev = h.copy()
        if f % CHUNK == CHUNK - 1 or f == n - 1:
            np.savez_compressed(os.path.join(cache, "sand", f"{f // CHUNK:03d}.npz"), base=base,
                                idx=np.concatenate(idx) if idx else np.zeros(0, np.int32),
                                val=np.concatenate(val) if val else np.zeros(0, np.int16),
                                off=np.array(off, np.int64))
        if not args.no_overlay:
            fill, matte = overlay(painter, sim, s)
            wf.stdin.write(fill.tobytes())
            wm.stdin.write(matte.tobytes())
        if f % 300 == 0:
            print(f"  frame {f}/{n}  year {sim.year:7.1f}  {time.time() - t0:5.0f}s", flush=True)
    if not args.no_overlay:
        for w in (wf, wm):
            w.stdin.close(); w.wait()

    # every lifted grain, sorted by the moment it was taken
    keys = ["t0", "x", "y", "z", "r", "cx", "cy", "cause", "ph", "mv"]
    L = {k: np.concatenate([b[k] for b in births]) for k in keys} if births else {k: np.zeros(0) for k in keys}
    o = np.argsort(L["t0"], kind="stable")
    L = {k: (v[o].astype(np.int8) if k == "cause" else v[o].astype(np.float32)) for k, v in L.items()}
    np.savez_compressed(os.path.join(cache, "lifted.npz"), **L)

    narrated = os.path.join(HERE, "out", F.FILM_NAME + "_narrated.mp4")
    if os.path.exists(narrated):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", narrated, "-vn", "-c:a", "copy",
                        os.path.join(OUTD, "narration.m4a")], check=True)
    print(f"exported {n} frames, {len(L['t0'])} lifted grains in {time.time() - t0:.0f}s -> {OUTD}")


if __name__ == "__main__":
    main()
