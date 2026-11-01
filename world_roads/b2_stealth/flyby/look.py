"""Look renders of the built B-2 (top, three-quarter, side, front, underside): python look.py OUTDIR"""
import bpy, math, os, sys
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2_model

out = sys.argv[-1]; os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
root = b2_model.build()
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 48; sc.cycles.device = "CPU"; sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.42, 0.55, 0.75, 1); bg.inputs[1].default_value = 0.45
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 4.0; sun.data.angle = math.radians(1.5)
sun.rotation_euler = (math.radians(35), math.radians(15), math.radians(-40))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50
views = {"top": (0, 0, 1), "q34": (0.75, -0.8, 0.45), "side": (0.02, -1, 0.06), "front": (1, 0.0, 0.1),
         "under": (0.3, -0.5, -0.8)}
for tag, d in views.items():
    d = Vector(d).normalized()
    cam.location = d * 70
    cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y" if tag != "top" else "X").to_euler()
    sc.render.filepath = os.path.join(out, f"look_{tag}.png")
    bpy.ops.render.render(write_still=True)
