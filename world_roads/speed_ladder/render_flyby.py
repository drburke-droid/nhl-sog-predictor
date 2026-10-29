"""Render one machine's flyby over the plate's camera and light, with the props standing in the road, frame by frame
and resumable.
  bpy-python render_flyby.py wright [first last]
  bpy-python render_flyby.py spitfire replay [first last]
-> /home/user/lot/render/fly_<name>[_replay]/p####.npz: rgba (linear, premultiplied), depth (m); static.npz for the frames where
neither the machine nor its shadow is in the picture. Only the part of the picture the machine and its shadow touch is
rendered; the rest is taken from the static frame.

Motion blur: the machine and its propellers are keyed at the frame and a quarter frame either side, at the times of a
1/100 s exposure (90-degree shutter at 25 fps); the propellers get enough motion steps to blur round in an arc."""
import bpy, sys, os, math, time
import numpy as np
from mathutils import Vector, Euler, Quaternion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, layout as L, aircraft as A, flight as F, props as PR
from exr import save

args = [a for a in sys.argv[1:] if a != "--"]
args = args[args.index(os.path.basename(__file__)) + 1:] if os.path.basename(__file__) in args else args
NAME = args[0]
REPLAY = len(args) > 1 and args[1] == "replay"
args = [NAME] + args[1 + REPLAY:]
rng_ = [int(a) for a in args[1:3]] if len(args) >= 3 else None
STATIC = f"/home/user/lot/render/fly_{NAME}/static.npz"
OUT = f"/home/user/lot/render/fly_{NAME}" + ("_replay" if REPLAY else "")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(STATIC), exist_ok=True)
SAMPLES = int(os.environ.get("SAMPLES", "48"))
EXPO = 0.01

s = S.reset(samples=SAMPLES)
s.render.use_motion_blur, s.render.motion_blur_shutter = True, 0.25
vl = s.view_layers[0]
vl.use_pass_z = True
s.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
s.render.image_settings.color_depth = "16"
objs = L.build()
rest = {k: (o.location.copy(), o.rotation_quaternion.copy()) for k, o in objs.items()}
glass = [o for o in bpy.data.objects if o.get("glass")]
glassmat = glass[0].data.materials[0] if glass else None
# the props the blast moves (the X-15): cones and drums from rigid.py, the sign and the car from hyper.py
SIMF = f"/home/user/lot/props/sim_{NAME}.npz"
SIM = dict(np.load(SIMF)) if NAME == "x15" and os.path.exists(SIMF) else None
if SIM is not None:
    import hyper as Hy
root, pivots = A.build(NAME)
root.rotation_mode = "XYZ"
for o in [root] + [p for p, _ in pivots] + [c for p, _ in pivots for c in p.children]:
    o.cycles.motion_steps = 5          # 2^(5-1)+1 = 17 transforms across the shutter
plumes = A.plume(root, NAME)
A.heat(NAME)

# a sample of the machine's own vertices (relative to its centre), to find the part of the picture it and its shadow
# cover more tightly than its bounding box does
bpy.context.view_layer.update()
_pts = []
for o in bpy.data.objects:
    if o.type != "MESH" or o.hide_render or o is root: continue
    par = o
    while par is not None and par is not root: par = par.parent
    if par is None: continue
    nv = len(o.data.vertices)
    _pts += [tuple(o.matrix_world @ o.data.vertices[i].co) for i in range(0, nv, max(1, nv // 400))]
LOCAL = np.array(_pts, float)
import camera as Cam, scene_sun as SS
SUN = np.array(SS.direction())


def tight_rect(t, expo, margin=48):
    out = None
    for dt in (-expo / 2, 0.0, expo / 2):
        (x, y, z), (roll, pitch, yaw) = F.pose(NAME, t + dt)
        Rm = np.array(Euler((roll, -pitch, yaw), "XYZ").to_matrix())
        P = LOCAL @ Rm.T + np.array([x, y, z])
        for Q in (P, P - np.outer(P[:, 2] / SUN[2], SUN)):
            uv, zc = Cam.project(Q)
            ok = zc > 0.1
            if ok.sum() < len(zc): return None            # partly beside the camera: fall back to the box
            r = (uv[:, 0].min() - margin, uv[:, 1].min() - margin, uv[:, 0].max() + margin, uv[:, 1].max() + margin)
            if r[2] < 0 or r[0] > Cam.W or r[3] < 0 or r[1] > Cam.H_IMG: continue
            out = r if out is None else (min(out[0], r[0]), min(out[1], r[1]), max(out[2], r[2]), max(out[3], r[3]))
    if out is None: return None
    return (max(0, int(out[0])), max(0, int(out[1])), min(Cam.W, int(out[2]) + 1), min(Cam.H_IMG, int(out[3]) + 1))


def pose(name, t):
    tt = SIM["t"]
    k = min(max((t - tt[0]) * 1000.0, 0.0), len(tt) - 1.001)
    i = int(k); f = k - i
    loc = SIM[name + "_loc"][i] * (1 - f) + SIM[name + "_loc"][i + 1] * f
    return loc, Quaternion(SIM[name + "_quat"][i]).slerp(Quaternion(SIM[name + "_quat"][i + 1]), f)


def moving(t, expo):
    return SIM is not None and t + expo / 2 >= Hy.first_hit() - 0.001


def place_props(n, t, expo):
    for o in objs.values(): o.animation_data_clear()
    for df, dt in ((-0.125, -expo / 2), (0.0, 0.0), (0.125, expo / 2)):
        tk = t + dt
        for name, o in objs.items():
            kind = L.LAYOUT[name][0]
            l0, q0 = rest[name]
            if kind in ("cone", "drum") and name + "_loc" in SIM:
                loc, q = pose(name, tk)
                o.location = Vector(loc); o.rotation_quaternion = q
            elif kind == "sign":
                o.location = l0; o.rotation_quaternion = Quaternion((1, 0, 0), Hy.sign_bend(tk)) @ q0
            elif kind == "car":
                slide, pitch, heave, roll = Hy.car_motion(tk)
                o.location = Vector((l0.x, l0.y - slide, l0.z + heave))
                o.rotation_quaternion = Quaternion((1, 0, 0), -pitch) @ Quaternion((0, 1, 0), roll) @ q0
            o.keyframe_insert("location", frame=n + df); o.keyframe_insert("rotation_quaternion", frame=n + df)
    for o in objs.values():
        ad = o.animation_data
        if ad and ad.action:
            for fc in getattr(ad.action, "fcurves", []):
                for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    for g in glass:
        g.data.materials[0] = PR.shattered() if Hy.car_glass_broken(t) else glassmat


def place(n, t, show=True, expo=EXPO):
    if SIM is not None: place_props(n, t, expo)
    for o in [root] + [p for p, _ in pivots]:
        o.animation_data_clear()
    for df, dt in ((-0.125, -expo / 2), (0.0, 0.0), (0.125, expo / 2)):
        (x, y, z), (roll, pitch, yaw) = F.pose(NAME, t + dt)
        root.location = (x, y, z) if show else (0, -500, -500)
        root.rotation_euler = Euler((roll, -pitch, yaw), "XYZ")
        root.keyframe_insert("location", frame=n + df); root.keyframe_insert("rotation_euler", frame=n + df)
        for p, w in pivots:
            p.rotation_euler.x = w * (t + dt)
            p.keyframe_insert("rotation_euler", frame=n + df)
    for o in [root] + [p for p, _ in pivots]:
        ad = o.animation_data
        if ad and ad.action:
            for fc in getattr(ad.action, "fcurves", []):
                for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    s.frame_set(n)


def border(rect):
    s.render.use_border = rect is not None
    s.render.use_crop_to_border = False
    if rect is None: return
    x0, y0, x1, y1 = rect
    W, H = s.render.resolution_x, s.render.resolution_y
    s.render.border_min_x, s.render.border_max_x = x0 / W, x1 / W
    s.render.border_min_y, s.render.border_max_y = 1 - y1 / H, 1 - y0 / H


# [(frame, t, exposure)]: real time, or the replay's high-speed camera (its exposure a quarter of its frame interval)
todo = [(n, t, EXPO) for n, t in F.frames(NAME)] if not REPLAY else \
       [(n, t, EXPO * slow) for n, (t, slow) in enumerate(F.replay(NAME))]
if rng_: todo = [x for x in todo if rng_[0] <= x[0] <= rng_[1]]
if not os.path.exists(STATIC):                            # the static frame first: the others borrow from it
    todo = [next((n, t, EXPO) for n, t in F.frames(NAME) if not F.in_view(NAME, t, EXPO))] + todo
for n, t, expo in todo:
    rect = F.screen_rect(NAME, t, expo)
    if rect:
        tr = tight_rect(t, expo)
        if tr: rect = tr
    if moving(t, expo):                                   # the blast has the props on the move: everything
        rect = (0, 0, s.render.resolution_x, s.render.resolution_y)
    path = f"{OUT}/p{n:04d}.npz" if rect else STATIC
    if os.path.exists(path): continue
    t0 = time.time()
    if rect and (rect[2] - rect[0]) * (rect[3] - rect[1]) > 0.7 * s.render.resolution_x * s.render.resolution_y:
        rect = (0, 0, s.render.resolution_x, s.render.resolution_y)
    place(n, t, F.screen_rect(NAME, t, expo) is not None, expo)
    border(rect if rect and rect != (0, 0, s.render.resolution_x, s.render.resolution_y) else None)
    s.render.filepath = f"{OUT}/tmp_{n:04d}.exr"
    bpy.ops.render.render(write_still=True)
    save(path, s.render.filepath)
    if s.render.use_border:
        st, d = dict(np.load(STATIC)), dict(np.load(path))
        x0, y0, x1, y1 = rect
        for k in ("rgba", "depth"):
            st[k][y0:y1, x0:x1] = d[k][y0:y1, x0:x1]
        np.savez_compressed(path, **st, rect=np.array(rect))
    print(f"{NAME} {n} t={t:.4f} s {rect} {time.time() - t0:.1f}s", flush=True)
print("ALLDONE", NAME)
