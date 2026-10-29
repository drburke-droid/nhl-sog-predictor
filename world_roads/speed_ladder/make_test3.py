"""Test 3, with the props: 50 mph (the ball falls in, bounces off the road, clips a cone), then 17,500 mph in real time
(the blast throws the cones and drums, bends the sign flat, rocks the parked car and blows in its windows); the replay
follows (replay_comp.py).  -> /home/user/lot/out/test3_a.mp4"""
import cv2, numpy as np, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C, camera as Cam, motion as M, chunks
R_ = "/home/user/lot/render"
OUT = "/home/user/lot/out"; FR = f"{OUT}/frames3"
os.makedirs(FR, exist_ok=True)
WHITE, DIM = (255, 255, 255), (214, 220, 228)
n_out = 0


def put(img):
    global n_out
    cv2.imwrite(f"{FR}/t{n_out:05d}.png", img[..., ::-1]); n_out += 1


def finish(img, seed, lines, shake_t=-1.0, shake_amp=10.0, fl=0.0, label_alpha=1.0):
    img = C.flash(img, fl)
    img = C.bloom(img)
    img = C.shake(img, shake_t, amp=shake_amp, seed=seed)
    img = C.grain(img, seed)
    pic = (C.srgb(img) * 255 + 0.5).astype(np.uint8)
    out = np.zeros((C.H, C.W, 3), np.uint8)
    out[C.TOP:C.TOP + C.PH] = pic
    return C.label(out, lines, label_alpha)


def roll_shot():
    land, _ = Cam.project([[M.X_LAND, 12.0, 0.0]])
    ex, _ = Cam.project([[M.X_LAND + 0.16, 12.0, 0.0]]); ey, _ = Cam.project([[M.X_LAND, 12.12, 0.0]])
    rx, ry = abs(ex[0, 0] - land[0, 0]), max(2.0, abs(ey[0, 1] - land[0, 1]))
    for n in range(59):
        t = n / 25.0
        bg = C.scuff(C.plate_loop(n), land[0], rx, ry, t - M.T_LAND)
        img = C.over_props(bg, C.props(f"{R_}/props_roll/p{n:04d}.npz"))
        img = C.over(img, chunks.grit(t, 0.02), soften=0)
        dl = C.dust(f"{R_}/roll_dust/d{n:04d}.png", f"{R_}/roll_dust/g{n:04d}.png")
        if dl is not None: img = img * dl[1] + dl[0]
        put(finish(img, n, [("50 mph", 64, "Bold", WHITE), ("a fast car", 30, "Medium", DIM)], label_alpha=min(1.0, n / 6)))


def blast_shot():
    for n in range(86):
        t, cg = (n - 25) / 25.0, n - 17
        bg = C.plate_loop(40 + n)
        dl = C.dust(f"{R_}/dust_blast/d{cg:04d}.png", f"{R_}/dust_blast/g{cg:04d}.png") if cg >= 9 else None
        if dl is not None: bg = bg * (1 - C.SUN_FRACTION * (1 - dl[2]))
        p = C.props(f"{R_}/props_blast/p{n:04d}.npz") or C.props(f"{R_}/props_blast/static.npz")
        img = C.over_props(bg, p)
        img = C.over(img, C.cg(f"{R_}/blast_100/b{cg:04d}.png") if cg >= 8 else None)
        img = img + chunks.shards(t, 0.02)
        if dl is not None: img = img * dl[1] + dl[0]
        fl = {25: 1.1, 26: 0.2}.get(n, 0.0)
        put(finish(img, 1000 + n, [("17,500 mph", 64, "Bold", WHITE), ("orbital speed, like the Space Station", 30, "Medium", DIM)],
                   shake_t=t - 0.035, shake_amp=16.0, fl=fl, label_alpha=min(1.0, n / 6)))


if __name__ == "__main__":
    which = sys.argv[1:] or ["roll", "blast"]
    if "roll" in which: roll_shot()
    if "blast" in which: blast_shot()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{FR}/t%05d.png", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", f"{OUT}/test3_a.mp4"], check=True)
    print("frames", n_out)
