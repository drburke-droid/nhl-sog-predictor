"""Render the moving chips for one shot, alone, over a transparent background.

Everything static is a holdout (so pins and walls still hide the chips behind them) and the plexiglass cover
is left out (the compositor lays the cover plate on top). The camera matches the compositor's crop of the
plate exactly: a crop of scale s centred at (cx, cy) is the same camera with lens / s and a lens shift.
Each frame renders only the screen rectangle around the chips.

    blender -b --factory-startup -P blender/render_chips.py -- <shot> [first_frame last_frame]
Writes out/chips/<shot>/f0001.png ... (only frames where a chip is on screen) and frames.json.
"""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cams as C  # noqa: E402
import scene as S  # noqa: E402

ROOT = os.path.dirname(HERE)
argv = sys.argv[sys.argv.index("--") + 1:]
shot_name = argv[0]
meta = json.load(open(os.path.join(ROOT, "out/anim3d/shots.json")))
paths = np.load(os.path.join(ROOT, "out/anim3d/paths.npz"))
shot = [s for s in meta["shots"] if s["name"] == shot_name][0]
FPS, HZ = meta["fps"], meta["hz"]
nframes = int(round(shot["dur"] * FPS))
f_first = int(argv[1]) if len(argv) > 1 else 1
f_last = int(argv[2]) if len(argv) > 2 else nframes
outdir = os.path.join(ROOT, "out", "chips", shot_name)
os.makedirs(outdir, exist_ok=True)

M, rig, signs = S.build()
S.render_settings(samples=16, res=C.OUT_RES, motion_blur=True)
sc = bpy.context.scene
sc.render.film_transparent = True
sc.render.image_settings.color_mode = "RGBA"
sc.frame_start, sc.frame_end = 1, nframes
for o in list(bpy.data.objects):
    if o.type in ("MESH", "FONT"):
        o.is_holdout = True
bpy.data.objects["cover"].hide_render = True
cam, tgt, spec = C.setup(shot["cam"])


# ------------------------------------------------------------------ camera: match the plate crop over time
def crop_at(t):
    ks = shot["crop"]
    if len(ks) == 1 or t <= ks[0][0]:
        return ks[0][1:]
    for a, b in zip(ks[:-1], ks[1:]):
        if t <= b[0]:
            f = (t - a[0]) / (b[0] - a[0])
            f = f * f * (3 - 2 * f)
            return [a[i] + (b[i] - a[i]) * f for i in (1, 2, 3)]
    return ks[-1][1:]


PW, PH = C.PLATE_RES
lens0, fstop0 = cam.data.lens, cam.data.dof.aperture_fstop
sx0, sy0 = cam.data.shift_x, cam.data.shift_y          # the plate camera's own lens shift
for fr in range(1, nframes + 2):
    s, cx, cy = crop_at((fr - 1) / FPS)
    cam.data.lens = lens0 / s
    cam.data.shift_x = (sx0 + cx - 0.5) / s
    cam.data.shift_y = (sy0 + (0.5 - cy) * (PH / PW)) / s
    cam.data.dof.aperture_fstop = fstop0 / s
    for path_ in ("lens", "shift_x", "shift_y"):
        cam.data.keyframe_insert(path_, frame=fr)
    cam.data.dof.keyframe_insert("aperture_fstop", frame=fr)

# ------------------------------------------------------------------ chips
space = bpy.data.objects.new("ChipSpace", None)
sc.collection.objects.link(space)
space.matrix_world = S.rig_matrix()
DEPTH = S.T_CHIP / 2 + 0.0015
chips = []


def keys(obj, path_, idx, frames, values, interp="LINEAR"):
    # let Blender create the action, its slot and the curve (4.4+ actions need a slot), then bulk-load points
    obj.keyframe_insert(path_, index=idx, frame=float(frames[0]))
    fc = obj.animation_data.action.fcurves.find(path_, index=idx)
    fc.keyframe_points.clear()
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames))
    co[0::2], co[1::2] = frames, values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [bpy.types.Keyframe.bl_rna.properties["interpolation"]
                                                     .enum_items[interp].value] * len(frames))
    fc.update()


slow = shot.get("slowmo", 1.0)
for i, c in enumerate(shot["chips"]):
    ob = S.new_chip(f"chip{i:03d}", M, c.get("face", "chip_face"))
    ob.parent = space
    ob.rotation_mode = "ZYX"
    if c.get("path"):
        p = paths[c["path"]]
        t_shot = c["t0"] + p[:, 0] / slow
        fr = 1 + t_shot * FPS
        if c.get("hold"):                       # sitting at the release point until t0
            fr = np.concatenate([[1.0], fr])
            p = np.concatenate([p[:1], p])
        x, y, ang = p[:, 1], p[:, 2], np.unwrap(p[:, 3])
        keys(ob, "location", 0, fr, x - 5 * S.W)
        keys(ob, "location", 1, fr, np.full_like(x, -DEPTH - c.get("depth_offset", 0.0)))
        keys(ob, "location", 2, fr, -y)
        keys(ob, "rotation_euler", 0, fr, np.full_like(x, math.pi / 2))
        keys(ob, "rotation_euler", 2, fr, -ang)
        t_on = c["appear"] if c.get("appear") is not None else c["t0"]
        t_off = t_shot[-1] + c.get("linger", 0.3)
    else:
        x, y, ang = c["still"]
        ob.location = (x - 5 * S.W, -DEPTH, -y)
        ob.rotation_euler = (math.pi / 2, 0, -ang)
        t_on, t_off = c.get("appear", 0.0), 1e9
    f_on, f_off = 1 + t_on * FPS, 1 + t_off * FPS
    vis_f = [0.0, max(0.0, f_on - 0.01), f_on, f_off, f_off + 0.01]
    vis_v = [0.0, 0.0, 1.0, 1.0, 0.0]
    for ax in range(3):
        keys(ob, "scale", ax, vis_f, vis_v, "CONSTANT")
    chips.append(ob)

# ------------------------------------------------------------------ render, frame by frame, chips' box only
log = {}
t_all = time.time()
static = all(not c.get("path") for c in shot["chips"]) and len(shot["crop"]) == 1
if static and shot["chips"]:           # nothing moves: one frame serves the whole shot
    f_on = 1 + int(math.ceil(min(c.get("appear", 0.0) for c in shot["chips"]) * FPS))
    f_first = f_last = max(f_first, f_on)
for fr in range(f_first, f_last + 1):
    sc.frame_set(fr)
    xs, ys = [], []
    for sub in (-0.3, 0.0, 0.3):              # the shutter is open from -0.25 to +0.25 of a frame
        tt = fr + sub
        sc.frame_set(int(math.floor(tt)), subframe=tt - math.floor(tt))
        for ob in chips:
            if ob.matrix_world.to_scale().x < 0.5:
                continue
            ctr = ob.matrix_world.translation
            for d in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)):
                q = world_to_camera_view(sc, cam, ctr + Vector(d) * S.R_CHIP * 1.2)
                xs.append(q.x)
                ys.append(q.y)
    sc.frame_set(fr)
    if not xs:
        continue
    m = 0.02
    x0, x1 = max(0.0, min(xs) - m), min(1.0, max(xs) + m)
    y0, y1 = max(0.0, min(ys) - m * 1.78), min(1.0, max(ys) + m * 1.78)
    if x1 <= 0 or x0 >= 1 or y1 <= 0 or y0 >= 1 or x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        continue
    sc.render.use_border = True
    sc.render.use_crop_to_border = False
    sc.render.border_min_x, sc.render.border_max_x = x0, x1
    sc.render.border_min_y, sc.render.border_max_y = y0, y1
    sc.render.filepath = os.path.join(outdir, f"f{fr:04d}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    log[fr] = [x0, x1, y0, y1, round(time.time() - t0, 2)]
    print(f"CHIPS {shot_name} {fr}/{nframes} {time.time() - t0:.1f}s area {(x1-x0)*(y1-y0):.3f}", flush=True)
if static and log:
    import shutil
    shutil.copy(os.path.join(outdir, f"f{f_first:04d}.png"), os.path.join(outdir, "static.png"))
    log["static_from"] = f_first
old = json.load(open(os.path.join(outdir, "frames.json"))) if os.path.exists(os.path.join(outdir, "frames.json")) else {}
old.update({str(k): v for k, v in log.items()})
json.dump(old, open(os.path.join(outdir, "frames.json"), "w"))
print(f"CHIPS_DONE {shot_name} {len(log)} frames in {time.time() - t_all:.0f}s", flush=True)
