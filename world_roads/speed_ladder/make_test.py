"""Assemble the first experiment: the 50 mph roll, then the 17,500 mph pass in real time.  -> /home/user/lot/out/test1.mp4"""
import cv2, numpy as np, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comp as C
OUT = "/home/user/lot/out"; os.makedirs(f"{OUT}/frames", exist_ok=True)
WHITE, DIM = (255, 255, 255), (214, 220, 228)
n = 0
def put(img):
    global n
    cv2.imwrite(f"{OUT}/frames/t{n:05d}.png", img[..., ::-1]); n += 1

def shot(cg_dir, frames, lead, tail, title, sub, flash_at=None, shock_at=None, plate0=0, cg_name="r", dust_dir=None,
         shake_amp=10.0):
    total = lead + frames + tail
    for i in range(total):
        k = min(max(i - lead, 0), frames - 1)
        layer = C.cg(f"{cg_dir}/{cg_name}{k:04d}.png") if cg_dir else None
        dl = C.dust(f"{dust_dir}/d{k:04d}.png", f"{dust_dir}/g{k:04d}.png") if dust_dir else None
        k = i - lead
        fl = 0.0
        if flash_at is not None:
            fl = {0: 1.1, 1: 0.2}.get(k - flash_at, 0.0)
        st = (k - shock_at) / 25 if shock_at is not None else -1
        a = min(1.0, i / 6) if i < 6 else 1.0
        put(C.frame(C.plate_loop(plate0 + i), layer, fl=fl, shake_t=st, seed=i, dust_layer=dl, shake_amp=shake_amp,
                    lines=[(title, 64, "Bold", WHITE), (sub, 30, "Medium", DIM)], label_alpha=a))

if __name__ == "__main__":
    which = sys.argv[1:] or ["roll", "blast"]
    if "roll" in which:
        shot("/home/user/lot/render/roll", 37, 10, 12, "50 mph", "a fast car", plate0=0)
    if "blast" in which:
        shot("/home/user/lot/render/blast_100", 69, 17, 0, "17,500 mph", "orbital speed, like the Space Station", flash_at=8,
             shock_at=9, plate0=40, cg_name="b", dust_dir="/home/user/lot/render/dust_blast", shake_amp=16.0)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "25", "-i", f"{OUT}/frames/t%05d.png", "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{OUT}/test1.mp4"], check=True)
    print("frames", n)
