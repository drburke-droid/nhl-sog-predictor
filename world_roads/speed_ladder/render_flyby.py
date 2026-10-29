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
from mathutils import Vector, Euler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, layout as L, aircraft as A, flight as F
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
L.build()
root, pivots = A.build(NAME)
root.rotation_mode = "XYZ"
for o in [root] + [p for p, _ in pivots] + [c for p, _ in pivots for c in p.children]:
    o.cycles.motion_steps = 5          # 2^(5-1)+1 = 17 transforms across the shutter


def place(n, t, show=True, expo=EXPO):
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
    path = f"{OUT}/p{n:04d}.npz" if rect else STATIC
    if os.path.exists(path): continue
    t0 = time.time()
    if rect and (rect[2] - rect[0]) * (rect[3] - rect[1]) > 0.7 * s.render.resolution_x * s.render.resolution_y:
        rect = (0, 0, s.render.resolution_x, s.render.resolution_y)
    place(n, t, rect is not None, expo)
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
