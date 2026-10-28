"""The dust of the 17,500 mph pass, simulated and rendered for every frame after it.
  python blast_dust.py [frames]  -> /home/user/lot/render/dust_blast/d####.png (premultiplied dust, linear x2)
                                    and g####.png (transmittance, ground shadow)

The ball passes at t = 0, 0.6 m up, along y = 12 m. The drag puts about 1 MJ into every metre of air along its path
(rough estimate; see the README). Modelled as: a core of air heated and expanding (about 3 m^2 per metre of path, the
volume 1 MJ of heat takes at atmospheric pressure), which then rises; dust and sand thrown off the ground under it by
the blast, faster the closer to the path; sand lifted later wherever the wind along the ground stays strong. Less
dust comes off the asphalt than off the sand."""
import numpy as np, math, os, sys, time, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dust as D
OUT = "/home/user/lot/render/dust_blast"
os.makedirs(OUT, exist_ok=True)
FPS, SUB = 25, 3
K = 8                                              # the frame of the pass (as in blast.py)
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PY, PZ = 12.0, 0.6
SCALE = float(os.environ.get("DUST_SCALE", "1.0"))
rng = np.random.default_rng(7)

# air
air = D.Grid(x0=-17, x1=17, y0=4, y1=22, z1=10, h=0.15)
sim = D.Sim(air, alpha=0.0, beta=6.0, eps=6.0, drag=0.2, cool=0.35, settle=0.0, iters=80)
sim.salt = (1e9, 0.0, 0.0)                         # lifting from the ground is done with parcels below
X, Y, Z = air.coords()
r2 = (Y - PY) ** 2 + (Z - PZ) ** 2
along = D.noise3(air.n, 9, 11, 3)                  # the blast is not the same strength all along the path
lump = 0.4 + 1.2 * along
air.T[:] = (2.0 * np.exp(-r2 / (2 * 0.9 ** 2)) * lump).astype(np.float32)
SIG, AREA, TAU = 0.55, 3.0, 0.05
core = (np.exp(-r2 / (2 * SIG ** 2)) / (2 * math.pi * SIG ** 2) * lump).astype(np.float32)
air.u[:] += (4.0 * (D.noise3(air.n, 5, 12, 2) - 0.5) * np.exp(-r2 / 8)).astype(np.float32)
air.v[:] += (3.0 * (D.noise3(air.n, 5, 14, 2) - 0.5) * np.exp(-r2 / 8)).astype(np.float32)
air.w[:] += (3.0 * (D.noise3(air.n, 5, 13, 2) - 0.5) * np.exp(-r2 / 8)).astype(np.float32)
dyA = Y - PY
air.w[:] += (8.0 * np.exp(-dyA ** 2 / (2 * 1.3 ** 2)) * np.exp(-Z / 3.0) * lump).astype(np.float32)
air.v[:] += (8.0 * np.tanh(dyA / 1.0) * np.exp(-dyA ** 2 / (2 * 2.5 ** 2)) * np.exp(-Z / 0.8) * lump).astype(np.float32)

# where dust comes from: asphalt gives less than sand and gravel
def avail(x):
    a = np.ones_like(x)
    a[(x > -4.2) & (x < 3.5)] = 0.55
    a[(x > -6.2) & (x <= -4.2)] = 0.7
    return a

# a 2D noise over the ground: clumps of dust, thrown harder where they are thicker
from scipy.ndimage import zoom
def ground_noise(seed):
    r = np.random.default_rng(seed)
    out = 0
    for cells, amp in ((0.5, 0.35), (1.5, 0.4), (4.0, 0.25)):
        nx, ny = int(36 / cells) + 4, int(10 / cells) + 4
        out = out + amp * zoom(r.random((nx, ny)), (360 / nx * 1.0001, 100 / ny * 1.0001), order=3)[:360, :100]
    out = (out - out.mean()) / out.std()
    return out
GN = ground_noise(21)                               # 0.1 m cells over x -18..18, y 7..17
def gn(x, y):
    i = np.clip(((x + 18) / 0.1).astype(int), 0, 359); j = np.clip(((y - 7) / 0.1).astype(int), 0, 99)
    return GN[i, j]

def emit(n, speed, tau, vs, spread):
    x = rng.uniform(-17, 17, n)
    keep = rng.random(n) < avail(x)
    x = x[keep]; n = len(x)
    dy = rng.normal(0, spread, n)
    y = PY + dy; z = rng.uniform(0, 0.15, n)
    c = gn(x, y)
    xi = np.clip(((x - air.o[0]) / air.h).astype(int), 0, air.n[0] - 1)
    st = lump[xi, int((PY - air.o[1]) / air.h), 3]
    sp = speed * np.exp(-np.abs(dy) / 1.5) * st * np.exp(0.25 * c) * rng.lognormal(0, 0.3, n)
    upw = 1.0 + 0.35 * rng.standard_normal(n)
    dirn = np.stack([0.2 * rng.standard_normal(n), 0.7 * dy, upw], 1)
    dirn /= np.linalg.norm(dirn, axis=1, keepdims=True)
    P = np.stack([x, y, z], 1).astype(np.float32)
    V = (dirn * sp[:, None]).astype(np.float32)
    w = np.exp(0.8 * c)                                 # clumpy: some patches carry far more dust
    return P, V, np.full(n, tau, np.float32), np.full(n, vs, np.float32), w.astype(np.float32)

fine = emit(1_200_000, 32.0, 0.32, 0.08, 1.4)
sand = emit(150_000, 13.0, 0.9, 8.0, 1.0)
P = np.concatenate([fine[0], sand[0]]); V = np.concatenate([fine[1], sand[1]])
tau = np.concatenate([fine[2], sand[2]]); vs = np.concatenate([fine[3], sand[3]])
mass = np.concatenate([fine[4], 0.35 * sand[4]]).astype(np.float32)
alive = np.ones(len(P), np.bool_)
print("parcels", len(P))

# what is rendered: parcels splatted on a finer grid
vis = D.Grid(x0=-16, x1=16, y0=5, y1=20, z1=9, h=0.1)
R = D.Renderer(vis, scale=SCALE, kappa=1.3)
UNIT = 0.11                                        # density per parcel per cell volume (tuned by eye)

def save(f, rgb, tr, gs):
    cv2.imwrite(f"{OUT}/d{f:04d}.png", np.clip(rgb[..., ::-1] * 32767.5, 0, 65535).astype(np.uint16))
    g2 = np.stack([np.zeros_like(tr), gs, tr], -1)
    cv2.imwrite(f"{OUT}/g{f:04d}.png", np.clip(g2 * 65535, 0, 65535).astype(np.uint16))

t = 0.0; dt = 1.0 / (FPS * SUB)
for j in range(1, NF + 1):
    t0 = time.time()
    for s in range(SUB):
        sim.src = (core * (AREA / TAU) * math.exp(-t / TAU)).astype(np.float32) if t < 6 * TAU else None
        sim.step(dt)
        D.move_parcels(P, V, tau, vs, alive, air.u, air.v, air.w, air.o, air.h, dt, 0.3)
        t += dt
    vis.d[:] = 0
    D.splat(P, mass * UNIT / (vis.h ** 3) * 1e-3, alive, vis.d, vis.o, vis.h)
    from scipy.ndimage import gaussian_filter
    vis.d[:] = gaussian_filter(vis.d, 0.6)
    rgb, tr, gs = R.render()
    save(K + j, rgb, tr, gs)
    print(f"frame {K + j} t={t:.3f}s alive {alive.sum()} maxd {vis.d.max():.2f} T.min {tr.min():.3f}  {time.time() - t0:.1f}s", flush=True)
