"""Render the props, and the 50 mph ball, over the plate's camera and light, frame by frame and resumable.
  bpy-python render_props.py roll|blast|replay [first last]
-> /home/user/lot/render/props_<seq>/p####.npz: rgba (linear, premultiplied), depth (m)

Where nothing moves the frame is rendered once and shared (static.npz). The replay's slowed part, 2.5 ms in which the
props move less than 3 cm, uses that one render, with normal, position and diffuse-colour passes so the compositor can
relight it with the glowing ball. Motion blur: each moving prop is keyed at the frame and a quarter frame either side,
with the times of the real exposure (half the frame's real interval, a 180-degree shutter)."""
import bpy, sys, os, math, time
import numpy as np
from mathutils import Quaternion, Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, layout as L, props as P, motion as M, replay_time as T

SEQ = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else sys.argv[1]
args = [a for a in sys.argv[1:] if a not in ("--",)]
rng_ = [int(a) for a in args[1:3]] if len(args) >= 3 else None
OUT = f"/home/user/lot/render/props_{SEQ}"
os.makedirs(OUT, exist_ok=True)
SAMPLES = int(os.environ.get("SAMPLES", "32"))
PASS = 25                                            # the frame of the pass in the real-time shot


def frames():
    """[(frame, real time t, exposure s, static?)]"""
    if SEQ == "roll":
        return [(n, n / 25.0, 0.02, False) for n in range(59)]
    if SEQ == "blast":
        return [(n, (n - PASS) / 25.0, 0.02, n < PASS) for n in range(86)]
    out = []
    for n, (t, slow) in enumerate(T.timeline()):
        out.append((n, t, 0.5 / (25 * slow), slow >= T.SLOW))
    return out


s = S.reset(samples=SAMPLES)
s.render.use_motion_blur, s.render.motion_blur_shutter = True, 0.5
vl = s.view_layers[0]
vl.use_pass_z = True
vl.use_pass_normal = True
vl.use_pass_position = True
vl.use_pass_diffuse_color = True
s.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
s.render.image_settings.color_depth = "16"
objs = L.build()
for o in objs.values():
    o.rotation_mode = "QUATERNION"
rest = {k: (o.location.copy(), o.rotation_quaternion.copy()) for k, o in objs.items()}
ball = S.ball(s, M.R_BALL) if SEQ == "roll" else None
if ball: ball.rotation_mode = "XYZ"
glass = [o for o in bpy.data.objects if o.get("glass")]
glassmat = glass[0].data.materials[0] if glass else None
lights = []
for i in range(16):
    l = bpy.data.lights.new(f"pl{i}", "POINT"); l.energy = 0.0; l.shadow_soft_size = 0.1
    o = bpy.data.objects.new(f"pl{i}", l); s.collection.objects.link(o); lights.append(o)

sims = {}
for sc in ("roll", "blast"):
    p = f"/home/user/lot/props/sim_{sc}.npz"
    if os.path.exists(p): sims[sc] = dict(np.load(p))
SIM = sims.get("roll" if SEQ == "roll" else "blast")


def pose(name, t):
    """location and quaternion of a simulated prop at time t"""
    tt = SIM["t"]
    k = (t - tt[0]) * 1000.0
    if k <= 0: return SIM[name + "_loc"][0], SIM[name + "_quat"][0]
    k = min(k, len(tt) - 1.001)
    i = int(k); f = k - i
    loc = SIM[name + "_loc"][i] * (1 - f) + SIM[name + "_loc"][i + 1] * f
    q0, q1 = Quaternion(SIM[name + "_quat"][i]), Quaternion(SIM[name + "_quat"][i + 1])
    return loc, q0.slerp(q1, f)


def place(n, t, expo):
    """key every moving thing at n - 1/4, n, n + 1/4 with the times of the exposure"""
    for o in list(objs.values()) + ([ball] if ball else []):
        o.animation_data_clear()
    for df, dt in ((-0.25, -expo / 2), (0.0, 0.0), (0.25, expo / 2)):
        tk = t + dt
        for name, o in objs.items():
            kind = L.LAYOUT[name][0]
            if kind in ("cone", "drum") and name + "_loc" in SIM:
                loc, q = pose(name, tk)
                o.location = Vector(loc); o.rotation_quaternion = q
            elif kind == "sign" and SEQ != "roll":
                o.location = rest[name][0]
                o.rotation_quaternion = Quaternion((1, 0, 0), M.sign_bend(tk)) @ rest[name][1]
            elif kind == "car" and SEQ != "roll":
                slide, pitch, heave = M.car_motion(tk)
                l0 = rest[name][0]
                o.location = Vector((l0.x, l0.y + slide, l0.z + heave))
                o.rotation_quaternion = Quaternion((1, 0, 0), -pitch) @ rest[name][1]
            else:
                continue
            o.keyframe_insert("location", frame=n + df); o.keyframe_insert("rotation_quaternion", frame=n + df)
        if ball:
            p, ang = M.ball(tk)
            ball.location = p; ball.rotation_euler = (0.0, ang, 0.0)
            ball.keyframe_insert("location", frame=n + df); ball.keyframe_insert("rotation_euler", frame=n + df)
    for o in list(objs.values()) + ([ball] if ball else []):
        ad = o.animation_data
        if ad and ad.action:
            for fc in getattr(ad.action, "fcurves", []):
                for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    broken = SEQ != "roll" and M.car_glass_broken(t)
    for g in glass:
        g.data.materials[0] = P.shattered() if broken else glassmat
    # light: the glowing ball and its wake in the replay; the streak's flash in the real-time shot
    ls = []
    if SEQ == "replay" and t > -2e-4 and t < 0.012:
        ls = T.lights(t)
    if SEQ == "blast" and n == PASS:
        ls = [((x, T.PY, T.PZ), (30 * 0.85, 30 * 0.9, 30.0)) for x in range(-14, 15, 2)]
    for i, o in enumerate(lights):
        if i < len(ls):
            (x, y, z), c = ls[i]
            inten = max(c)
            o.location = (x, y, z)
            o.data.energy = 4 * math.pi * inten                 # W/sr in the compositor's units -> watts
            o.data.color = tuple(ci / inten for ci in c)
        else:
            o.data.energy = 0.0
    s.frame_set(n)


def save(path, exr, passes=False):
    """Combined and depth always; normal, position and diffuse colour (for relighting) when asked"""
    import OpenEXR
    f = OpenEXR.File(exr)
    ch = {k: np.asarray(c.pixels) for part in f.parts for k, c in part.channels.items()}
    out = {"rgba": ch["ViewLayer.Combined"].astype(np.float16),
           "depth": np.minimum(ch["ViewLayer.Depth.Z"], 6e4).astype(np.float16)}
    if passes:
        out["normal"] = np.stack([ch[f"ViewLayer.Normal.{a}"] for a in "XYZ"], -1).astype(np.float16)
        out["position"] = np.clip(np.stack([ch[f"ViewLayer.Position.{a}"] for a in "XYZ"], -1), -6e4, 6e4).astype(np.float16)
        out["diffcol"] = ch["ViewLayer.Diffuse Color"].astype(np.float16)
    np.savez_compressed(path, **out)
    os.remove(exr)


todo = frames()
if rng_: todo = [x for x in todo if rng_[0] <= x[0] <= rng_[1]]
for n, t, expo, static in todo:
    path = f"{OUT}/static.npz" if static else f"{OUT}/p{n:04d}.npz"
    if os.path.exists(path): continue
    t0 = time.time()
    place(n, t, expo)
    s.render.filepath = f"{OUT}/tmp_{n:04d}.exr"
    bpy.ops.render.render(write_still=True)
    save(path, s.render.filepath, passes=static)
    print(f"{SEQ} {n} t={t * 1e3:.2f} ms {'static ' if static else ''}{time.time() - t0:.1f}s", flush=True)
print("ALLDONE", SEQ)
