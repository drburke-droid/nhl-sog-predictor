"""Build the desk's paper texture (out/paper_tex.png) once. Same recipe as ../vernier/look.paper_tex, but the
widest blurs are worked on a coarser grid and scaled up: the same look in seconds instead of most of an hour.

    python make_paper.py
"""
import os, sys, math
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import imaging  # noqa: F401  (points look.OUT here)
import look as L


def main():
    path = os.path.join(L.OUT, "paper_tex.png")
    os.makedirs(L.OUT, exist_ok=True)
    x0, y0, x1, y1 = L.WORLD
    tw, th = int((x1 - x0) * L.TEX_S), int((y1 - y0) * L.TEX_S)
    rng = np.random.default_rng(17)
    lum = np.zeros((th, tw), np.float32)
    for sig, amp in ((420, 0.012), (140, 0.010), (40, 0.008), (6, 0.008), (1.2, 0.014)):
        q = max(1, int(sig // 8))                     # work wide blurs on a grid q times coarser
        n = rng.normal(0, 1, (th // q + 1, tw // q + 1)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sig / q)
        n = cv2.resize(n, (tw, th), interpolation=cv2.INTER_CUBIC) if q > 1 else n[:th, :tw]
        lum += n / (n.std() + 1e-9) * amp
    fib = np.zeros((th, tw), np.float32)
    for _ in range(9000):
        x, y = rng.uniform(0, tw), rng.uniform(0, th)
        ang = rng.uniform(0, math.pi)
        Ln = rng.uniform(15, 90)
        bend = rng.normal(0, 0.4)
        pts = []
        for k in range(8):
            u = k / 7
            a_ = ang + bend * (u - 0.5)
            pts.append((x + math.cos(a_) * Ln * (u - 0.5), y + math.sin(a_) * Ln * (u - 0.5)))
        cv2.polylines(fib, [np.round(np.array(pts) * 4).astype(np.int32)], False, float(rng.choice([-1, 1]) * rng.uniform(0.3, 1)),
                      1, cv2.LINE_AA, shift=2)
    lum += cv2.GaussianBlur(fib, (0, 0), 0.6) * 0.045
    base = np.array(L.PAPER, np.float32)
    tex = np.clip(base[None, None, :] * (1 + lum[..., None]) + np.array([2, 0, -3], np.float32) * lum[..., None] * 20, 0, 255)
    cv2.imwrite(path, tex.astype(np.uint8)[..., ::-1])
    print("wrote", path, tex.shape)


if __name__ == "__main__":
    main()
