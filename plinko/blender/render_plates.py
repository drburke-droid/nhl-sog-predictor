"""Render the static plates for one camera: the set without the cover, and the cover alone (RGBA).

    blender -b --factory-startup -P blender/render_plates.py -- front [samples]
Writes out/plates/<cam>_set.png, <cam>_cover.png and <cam>.json (camera + projected points).
"""
import json
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cams as C  # noqa: E402
import scene as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
name = argv[0]
samples = int(argv[1]) if len(argv) > 1 else 32
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.path.join(root, "out", "plates")
os.makedirs(outdir, exist_ok=True)

M, rig, signs = S.build()
S.render_settings(samples=samples, res=C.PLATE_RES, motion_blur=False)
sc = bpy.context.scene
sc.cycles.adaptive_threshold = 0.03
cam, tgt, spec = C.setup(name)
bpy.context.view_layer.update()

from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Vector  # noqa: E402

info = S.camera_info(cam)
pts = S.world_points()
# reference projections, so the compositor can check its own projection maths against Blender's
probe = [pts["drop_gaps"][4], pts["slot_centres"][0], pts["field_br"], pts["pins"][50]]
info["probe_world"] = probe
info["probe_ndc"] = [list(world_to_camera_view(sc, cam, Vector(p)))[:2] for p in probe]
info["points"] = pts
json.dump(info, open(os.path.join(outdir, f"{name}.json"), "w"))

cover = bpy.data.objects["cover"]
t0 = time.time()
cover.hide_render = True
sc.render.film_transparent = False
sc.render.filepath = os.path.join(outdir, f"{name}_set.png")
bpy.ops.render.render(write_still=True)
print(f"PLATE {name} set {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
cover.hide_render = False
for o in bpy.data.objects:
    if o.type in ("MESH", "FONT") and o.name != "cover":
        o.is_holdout = True
sc.render.film_transparent = True
sc.render.image_settings.color_mode = "RGBA"
sc.render.filepath = os.path.join(outdir, f"{name}_cover.png")
bpy.ops.render.render(write_still=True)
print(f"PLATE {name} cover {time.time() - t0:.0f}s", flush=True)
