"""The Blender scene that matches the plate: camera from camera.py, a shadow-catching ground, the low hazy sun and the
plate-built environment (env.py) for fill and reflections. Rendered with a transparent film and composited over the
plate in comp.py."""
import bpy, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as C

from scene_sun import SUN_AZ, SUN_EL   # degrees: to the right of the road direction, above the horizon
import os
# Calibrated against the plate (calib.py): test patches of asphalt and road paint rendered on the road beside the real
# ones came out 2-4 times too bright and too blue with the plate-built sky at full strength. The sky as seen in
# reflections stays as the plate shows it; the light it sheds on diffuse surfaces is scaled down and warmed, and the
# sun is golden, giving about half the light on open ground (the stones on the shoulder are lit on their right faces
# and throw short soft shadows to the left).
SUN_STRENGTH = float(os.environ.get("SUN_STRENGTH", "1.3"))      # W/m^2
SUN_ANGLE = 5.0                                                 # degrees: thin cloud softens the shadows
SUN_COLOR = (1.0, 0.78, 0.58)
SKY_DIFFUSE = float(os.environ.get("SKY_DIFFUSE", "0.32"))       # the sky's light on diffuse surfaces, relative
SKY_TINT = (1.1, 1.0, 0.85)
GROUND_ALBEDO = (0.075, 0.065, 0.058)                           # dark asphalt and darker gravel, for bounce light


def reset(samples=64, scale=100, fps=25):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    s.cycles.denoiser = "OPENIMAGEDENOISE"
    s.render.film_transparent = True
    s.render.resolution_x, s.render.resolution_y = C.W, C.H_IMG
    s.render.resolution_percentage = scale
    s.render.fps = fps
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.render.image_settings.color_depth = "16"
    s.cycles.max_bounces = 6
    s.render.threads_mode = "AUTO"
    camera(s); world(s); sun(s); ground(s)
    return s


def camera(s):
    b = C.blender()
    cam = bpy.data.cameras.new("cam")
    cam.lens, cam.sensor_width, cam.sensor_fit = b["lens"], b["sensor"], "HORIZONTAL"
    cam.shift_x, cam.shift_y = b["shift_x"], b["shift_y"]
    cam.clip_start, cam.clip_end = 0.05, 5000
    o = bpy.data.objects.new("cam", cam)
    o.location = b["loc"]
    o.rotation_euler = [math.radians(a) for a in b["rot_deg"]]
    s.collection.objects.link(o)
    s.camera = o


def world(s):
    w = bpy.data.worlds.new("env")
    s.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.load("/home/user/lot/env.hdr")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    # reflections see the sky as the plate does; diffuse light from it is scaled and warmed (see SKY_DIFFUSE)
    full = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(tex.outputs["Color"], full.inputs["Color"])
    tint = nt.nodes.new("ShaderNodeMix"); tint.data_type = "RGBA"; tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], tint.inputs["A"])
    tint.inputs["B"].default_value = (*(SKY_DIFFUSE * c for c in SKY_TINT), 1.0)
    dim = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(tint.outputs["Result"], dim.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Diffuse Ray"], mix.inputs["Fac"])
    nt.links.new(full.outputs["Background"], mix.inputs[1])
    nt.links.new(dim.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def sun(s):
    l = bpy.data.lights.new("sun", "SUN")
    l.energy, l.angle = SUN_STRENGTH, math.radians(SUN_ANGLE)
    l.color = SUN_COLOR
    o = bpy.data.objects.new("sun", l)
    # a sun lamp shines along its local -Z; point it from (az, el) toward the scene
    az, el = math.radians(SUN_AZ), math.radians(SUN_EL)
    d = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))     # toward the sun
    import mathutils
    o.rotation_euler = mathutils.Vector(d).to_track_quat("Z", "Y").to_euler()
    s.collection.objects.link(o)


def ground(s):
    me = bpy.data.meshes.new("ground")
    me.from_pydata([(-200, -5, 0), (200, -5, 0), (200, 600, 0), (-200, 600, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("ground", me)
    o.is_shadow_catcher = True
    m = bpy.data.materials.new("groundmat")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*GROUND_ALBEDO, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    me.materials.append(m)
    s.collection.objects.link(o)
    return o


def steel(name="steel", rough=0.22):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.56, 0.57, 0.58, 1)
    p.inputs["Metallic"].default_value = 1.0
    # brushed, slightly scuffed: roughness varies with a noise
    n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = 40
    r = nt.nodes.new("ShaderNodeMapRange")
    r.inputs["To Min"].default_value, r.inputs["To Max"].default_value = rough * 0.6, rough * 1.6
    nt.links.new(n.outputs["Fac"], r.inputs["Value"])
    nt.links.new(r.outputs["Result"], p.inputs["Roughness"])
    return m


def ball(s, radius=0.15, name="ball"):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius)
    o = bpy.context.active_object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(steel())
    return o
