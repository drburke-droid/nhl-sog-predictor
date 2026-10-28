"""The Blender scene that matches the plate: camera from camera.py, a shadow-catching ground, the low hazy sun and the
plate-built environment (env.py) for fill and reflections. Rendered with a transparent film and composited over the
plate in comp.py."""
import bpy, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera as C

SUN_AZ, SUN_EL = 50.0, 20.0         # degrees: to the right of the road direction, above the horizon
SUN_STRENGTH, SUN_ANGLE = 2.6, 4.0  # W/m^2, degrees (thin cloud softens the shadows)


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
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def sun(s):
    l = bpy.data.lights.new("sun", "SUN")
    l.energy, l.angle = SUN_STRENGTH, math.radians(SUN_ANGLE)
    l.color = (1.0, 0.93, 0.84)
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
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.15, 0.11, 1)
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
