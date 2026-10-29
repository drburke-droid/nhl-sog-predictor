"""The puff where the 50 mph ball hits the road (x = 0.8 m, t = 0.931 s; 9 m/s straight down on asphalt): road dust
splashed out low and sideways, a little up, slowed by the air within a fraction of a second, settling.
  python roll_dust.py -> /home/user/lot/render/roll_dust/d####.png, g####.png (shot frames)"""
import numpy as np, math, os, sys, time, cv2
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust as D, motion as M, blast_dust as B
OUT = "/home/user/lot/render/roll_dust"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(12)
N = 80_000
ang = rng.uniform(0, 2 * math.pi, N)
sp = rng.lognormal(math.log(2.5), 0.5, N)
up = np.abs(rng.normal(0.9, 0.6, N))
V0 = np.stack([np.cos(ang) * sp + 0.25 * M.VX1 * rng.random(N), np.sin(ang) * sp, up], 1)   # carried a bit forward
P0 = np.stack([M.X_LAND + rng.normal(0, 0.05, N), 12.0 + rng.normal(0, 0.05, N), np.full(N, 0.01)], 1)
TAU, VS = 0.12, 0.25
w = rng.lognormal(0, 1.4, N).astype(np.float32)
vis = D.Grid(x0=-1.5, x1=4.5, y0=9.5, y1=14.5, z1=1.6, h=0.03)
R = D.Renderer(vis, scale=1.0, kappa=1.3, albedo=(0.30, 0.27, 0.24))
UNIT = 0.00022
for n in range(59):
    t = n / 25.0
    a = t - M.T_LAND
    path_d = f"{OUT}/d{n:04d}.png"
    if a <= 0:
        continue
    k = 1 - math.exp(-a / TAU)
    P = (P0 + V0 * TAU * k + np.array([0, 0, -VS]) * a).astype(np.float32)
    P[:, 2] = np.maximum(P[:, 2], 0.0)
    vis.d[:] = 0
    fade = math.exp(-a / 1.5)
    D.splat(P, (w * UNIT * fade / vis.h ** 3 * 1e-3).astype(np.float32), np.ones(N, np.bool_), vis.d, vis.o, vis.h)
    vis.d[:] = gaussian_filter(vis.d, 0.7)
    rgb, tr, gs = D.render_l(R, [], tmax=B.props_tmax(R, n, "roll"))
    cv2.imwrite(path_d, np.clip(rgb[..., ::-1] * 32767.5, 0, 65535).astype(np.uint16))
    cv2.imwrite(f"{OUT}/g{n:04d}.png", np.clip(np.stack([np.zeros_like(tr), gs, tr], -1) * 65535, 0, 65535).astype(np.uint16))
    print("roll dust", n, f"T.min {tr.min():.3f}", flush=True)
