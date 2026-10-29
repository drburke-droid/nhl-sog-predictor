"""Calibration: strips of asphalt (albedo 0.05) and road paint (0.65) rendered on the road beside the real ones, to
compare with the plate (read the result with the numbers in the README). -> /home/user/lot/render/calib.npz"""
import bpy, sys, os, math
sys.path.insert(0, "/home/user/lot/src")
import scene as S, layout as L
from exr import save
s = S.reset(samples=48)
s.view_layers[0].use_pass_z = True
s.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
objs = L.build()
def patch(name, x0, x1, y0, y1, col, rough):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x0, y0, 0.002), (x1, y0, 0.002), (x1, y1, 0.002), (x0, y1, 0.002)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new(name, me); s.collection.objects.link(o)
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*col, 1); p.inputs["Roughness"].default_value = rough
    me.materials.append(m)
# asphalt and white paint test patches, side by side, near (left lane) and farther
patch("paintstrip", -1.3, -1.0, 4.0, 30.0, (0.65, 0.65, 0.65), 0.6)
patch("asphstrip", -2.6, -1.8, 4.0, 30.0, (0.05, 0.05, 0.05), 0.8)
s.render.filepath = "/home/user/lot/render/calib.exr"
bpy.ops.render.render(write_still=True)
save("/home/user/lot/render/calib.npz", s.render.filepath)
