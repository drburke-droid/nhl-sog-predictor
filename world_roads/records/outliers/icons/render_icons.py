"""Render the sport-icon models (supplied .blend files) to transparent PNG sprites with Blender's Cycles.

    python -I render_icons.py <models dir> <out dir> [name:file:azimuth:elevation:pixels ...]

Each file is opened with scripts disabled. Existing cameras and lights are removed; an orthographic camera is aimed
at the meshes' bounding box from the given azimuth and elevation (degrees), with a soft three-light rig and a
transparent background.
"""
import math, os, sys
import bpy
from mathutils import Vector

src, dst = sys.argv[1], sys.argv[2]
KEY, FILL, RIM = 6.0, 1.8, 4.5          # light power per square metre of distance (W/m^2): a soft studio look
jobs = [a.split(":") for a in sys.argv[3:]]
os.makedirs(dst, exist_ok=True)


def bbox():
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))
    for o in bpy.context.scene.objects:
        if o.type in ("MESH", "FONT", "CURVE") and o.visible_get():
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                lo = Vector(map(min, lo, w))
                hi = Vector(map(max, hi, w))
    return lo, hi


def area(name, loc, energy, size, target):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.size = energy, size
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


for name, fname, az, el, pix in jobs:
    bpy.ops.wm.open_mainfile(filepath=os.path.join(src, fname), load_ui=False, use_scripts=False)
    sc = bpy.context.scene
    for o in list(sc.objects):
        if o.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(o, do_unlink=True)
    lo, hi = bbox()
    ctr, ext = (lo + hi) / 2, (hi - lo)
    size = max(ext)
    az, el = math.radians(float(az)), math.radians(float(el))
    d = Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)))
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = size * 1.12
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    cam.location = ctr + d * size * 6
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    k = size * 4
    for nm, v, e, s_ in (("key", Vector((-0.8, -1.0, 1.1)), KEY, 3), ("fill", Vector((1.1, -0.6, 0.3)), FILL, 4),
                         ("rim", Vector((0.2, 1.2, 0.9)), RIM, 2)):
        dist = (v * k).length
        area(nm, ctr + v * k, e * dist * dist, size * s_, ctr)
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.055, 0.06, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    sc.world = w
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 48
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True
    px = int(pix)
    sc.render.resolution_x = sc.render.resolution_y = px
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "AgX"
    sc.render.filepath = os.path.join(dst, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", name, flush=True)
