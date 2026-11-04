"""Blender (Cycles) scene: the world map as a dark table with a glass vessel standing on every outline.

    blender -b -P blender_scene.py -- --samples 384 [--test]

Renders two plates at 1920x1080 into data/: plate_glass.png (table + glass) and plate_bare.png (table
only), both 16-bit. Their ratio and difference are the glass's tint and highlights, composited over the
sand each frame. Also writes data/camera.json: the 3x4 matrix projecting table points (map px, height px)
to screen px.
Units: 1 Blender unit = 100 map pixels; map pixel (x, y) -> (x/100 - 9.6, 5.4 - y/100, 0).
"""
import json, math, os, sys
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 256
TEST = "--test" in argv
SAND = argv[argv.index("--sand") + 1] if "--sand" in argv else None       # render this sand state
OUTP = argv[argv.index("--out") + 1] if "--out" in argv else None
# --view X Y ZOOM TILT: a close-up looking at map pixel (X, Y), ZOOM times closer, tilted TILT degrees
VIEW = [float(v) for v in argv[argv.index("--view") + 1:argv.index("--view") + 5]] if "--view" in argv else None
# --poses plate_poses.json --outdir DIR [--range A B] [--bare-samples N]: glass plates for the moving camera
POSES = argv[argv.index("--poses") + 1] if "--poses" in argv else None
OUTDIR = argv[argv.index("--outdir") + 1] if "--outdir" in argv else os.path.join(HERE, "data", "plates")
RANGE = [int(v) for v in argv[argv.index("--range") + 1:argv.index("--range") + 3]] if "--range" in argv else None
BARE_SAMPLES = int(argv[argv.index("--bare-samples") + 1]) if "--bare-samples" in argv else 24
FAST = "--fast" in argv          # fewer bounces, no refractive caustics (keyframe plates)
WALL_H = 0.14            # wall height (14 map px)
BEVEL = 0.009


def px2w(x, y, z=0.0):
    return Vector((x / 100.0 - 9.6, 5.4 - y / 100.0, z))


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24 if TEST else SAMPLES
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = 16
scene.cycles.transmission_bounces = 16
scene.cycles.glossy_bounces = 8
scene.cycles.caustics_reflective = False
scene.cycles.caustics_refractive = True
scene.cycles.blur_glossy = 1.0
if FAST:
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 10
    scene.cycles.glossy_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.caustics_refractive = False
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "16"
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.render.threads_mode = "AUTO"

# ---------------------------------------------------------------- world: dark for the camera and the
# table, but a studio of soft bright strips for reflections and refraction (what makes glass read as glass)
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
bg = wn["Background"]
out = wn["World Output"]
dark = wn.new("ShaderNodeBackground")
dark.inputs["Color"].default_value = (0.006, 0.008, 0.014, 1)
dark.inputs["Strength"].default_value = 1.0
# reflection studio: brightness by elevation (horizon glow) and azimuth (softbox strips)
geo = wn.new("ShaderNodeTexCoord")
sep = wn.new("ShaderNodeSeparateXYZ")
wl.new(geo.outputs["Generated"], sep.inputs["Vector"])
grad = wn.new("ShaderNodeValToRGB")          # elevation: z of the view direction
wl.new(sep.outputs["Z"], grad.inputs["Fac"])
grad.color_ramp.elements[0].position = 0.45
grad.color_ramp.elements[0].color = (0.02, 0.025, 0.035, 1)
grad.color_ramp.elements[1].position = 0.75
grad.color_ramp.elements[1].color = (0.9, 0.95, 1.0, 1)
e = grad.color_ramp.elements.new(0.95)
e.color = (0.25, 0.28, 0.33, 1)
wave = wn.new("ShaderNodeTexWave")            # vertical strips around the horizon
wave.wave_type = "RINGS"
wave.inputs["Scale"].default_value = 1.4
wave.inputs["Distortion"].default_value = 0.0
wave.inputs["Detail"].default_value = 0.0
wl.new(geo.outputs["Generated"], wave.inputs["Vector"])
strip = wn.new("ShaderNodeValToRGB")
wl.new(wave.outputs["Fac"], strip.inputs["Fac"])
strip.color_ramp.elements[0].position = 0.55
strip.color_ramp.elements[0].color = (0.25, 0.25, 0.25, 1)
strip.color_ramp.elements[1].position = 0.9
strip.color_ramp.elements[1].color = (1.6, 1.6, 1.6, 1)
mul = wn.new("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
mul.inputs["Factor"].default_value = 1.0
wl.new(grad.outputs["Color"], mul.inputs[6])
wl.new(strip.outputs["Color"], mul.inputs[7])
wl.new(mul.outputs[2], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 2.2
lp = wn.new("ShaderNodeLightPath")
mix = wn.new("ShaderNodeMixShader")
gl = wn.new("ShaderNodeMath"); gl.operation = "MAXIMUM"
wl.new(lp.outputs["Is Glossy Ray"], gl.inputs[0])
wl.new(lp.outputs["Is Transmission Ray"], gl.inputs[1])
wl.new(gl.outputs[0], mix.inputs["Fac"])
wl.new(dark.outputs[0], mix.inputs[1])
wl.new(bg.outputs[0], mix.inputs[2])
wl.new(mix.outputs[0], out.inputs["Surface"])

# ---------------------------------------------------------------- the table (map texture)
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
table = bpy.context.object
table.scale = (19.2 * 8, 10.8 * 8, 1)       # large: low camera angles never see its edge
mat = bpy.data.materials.new("Table")
mat.use_nodes = True
nt = mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(os.path.join(DATA, "table.png"))
tex.extension = "EXTEND"
tc = nt.nodes.new("ShaderNodeTexCoord")
mp = nt.nodes.new("ShaderNodeMapping")
# object coords (-0.5..0.5) -> map uv: the map covers the central 19.2 x 10.8 of the plane
mp.inputs["Scale"].default_value = (8, 8, 1)
mp.inputs["Location"].default_value = (0.5, 0.5, 0)
nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.42
bsdf.inputs["Specular IOR Level"].default_value = 0.35
table.data.materials.append(mat)

# ---------------------------------------------------------------- glass
glass = bpy.data.materials.new("Glass")
glass.use_nodes = True
g = glass.node_tree.nodes["Principled BSDF"]
g.inputs["Base Color"].default_value = (0.93, 0.97, 1.0, 1)
g.inputs["Transmission Weight"].default_value = 1.0
g.inputs["Roughness"].default_value = 0.015
g.inputs["IOR"].default_value = 1.50
g.inputs["Coat Weight"].default_value = 0.25
g.inputs["Coat Roughness"].default_value = 0.02

walls = json.load(open(os.path.join(DATA, "walls.json")))
glass_objs = []
for vid, polys in walls.items():
    cu = bpy.data.curves.new(f"wall_{vid}", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = WALL_H / 2 - BEVEL
    cu.bevel_depth = BEVEL
    cu.bevel_resolution = 3
    for rings in polys:
        for ring in rings:
            pts = ring[:-1] if ring[0] == ring[-1] else ring
            sp = cu.splines.new("POLY")
            sp.points.add(len(pts) - 1)
            for k, (x, y) in enumerate(pts):
                v = px2w(x, y)
                sp.points[k].co = (v.x, v.y, 0, 1)
            sp.use_cyclic_u = True
    ob = bpy.data.objects.new(f"wall_{vid}", cu)
    ob.location.z = WALL_H / 2
    ob.data.materials.append(glass)
    scene.collection.objects.link(ob)
    glass_objs.append(ob)

# ---------------------------------------------------------------- lights: big softbox upper left, rim lights
def area(name, loc, size, energy, color, target=(0, 0, 0)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    scene.collection.objects.link(ob)
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


area("key", (-14, 10, 14), 10, 2600, (1.0, 0.96, 0.9))
area("fill", (12, -12, 10), 14, 700, (0.75, 0.85, 1.0))
area("rim", (0, 16, 4), 18, 1800, (0.85, 0.9, 1.0))
area("top", (0, 0, 22), 8, 500, (1, 1, 1))

# ---------------------------------------------------------------- camera: gently tilted, framing the map
cam_d = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_d)
scene.collection.objects.link(cam)
scene.camera = cam
target = Vector((0, 0.35, 0))
tilt = math.radians(22)
dist = 26.0
cam.location = target + Vector((0, -math.sin(tilt) * dist, math.cos(tilt) * dist))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_d.sensor_fit = "HORIZONTAL"
cam_d.sensor_width = 36
bpy.context.view_layer.update()

from bpy_extras.object_utils import world_to_camera_view


def fits(lens):
    cam_d.lens = lens
    bpy.context.view_layer.update()
    pts = [px2w(x, y) for x in (0, 1920) for y in (70, 880)]
    ok = True
    for p in pts:
        c = world_to_camera_view(scene, cam, p)
        ok &= 0.005 < c.x < 0.995 and 0.0 < c.y < 1.0
    return ok


lo, hi = 10.0, 120.0
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if fits(mid) else (lo, mid)
cam_d.lens = lo
bpy.context.view_layer.update()

# projection matrix: (map px x, map px y, height px) -> screen px
import numpy as np
samples, scr = [], []
for x in np.linspace(0, 1920, 7):
    for y in np.linspace(0, 1080, 5):
        for z in (0.0, 40.0):
            c = world_to_camera_view(scene, cam, px2w(x, y, z / 100.0))
            samples.append((x, y, z))
            scr.append((c.x * 1920, (1 - c.y) * 1080))
A = []
for (x, y, z), (u, v) in zip(samples, scr):
    A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
    A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
_, _, vt = np.linalg.svd(np.array(A))
P = vt[-1].reshape(3, 4)
P /= P[2, 3]
json.dump(dict(P=P.tolist(), lens=cam_d.lens), open(os.path.join(DATA, "camera.json"), "w"))
print("lens", cam_d.lens)

# ---------------------------------------------------------------- optional: real sand, as spheres
if VIEW:
    vx, vy, zoom, vtilt = VIEW
    tgt = px2w(vx, vy)
    d = dist / zoom
    t = math.radians(vtilt)
    cam.location = tgt + Vector((0, -math.sin(t) * d, math.cos(t) * d))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_d.dof.use_dof = True
    cam_d.dof.focus_distance = d
    cam_d.dof.aperture_fstop = 0.9
    bpy.context.view_layer.update()
if SAND:
    z = np.load(SAND)
    n = len(z["x"])
    me = bpy.data.meshes.new("sand")
    co = np.stack([z["x"] / 100.0 - 9.6, 5.4 - z["y"] / 100.0, z["z"] / 100.0], 1).astype(np.float32)
    me.vertices.add(n)
    me.vertices.foreach_set("co", co.ravel())
    ca = me.attributes.new("col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.concatenate([z["c"], np.ones((n, 1), np.float32)], 1).ravel())
    ra = me.attributes.new("rad", "FLOAT", "POINT")
    ra.data.foreach_set("value", (z["r"] / 100.0).astype(np.float32))
    ob = bpy.data.objects.new("sand", me)
    scene.collection.objects.link(ob)
    sm = bpy.data.materials.new("Sand")
    sm.use_nodes = True
    sb = sm.node_tree.nodes["Principled BSDF"]
    at = sm.node_tree.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "col"
    sm.node_tree.links.new(at.outputs["Color"], sb.inputs["Base Color"])
    sb.inputs["Roughness"].default_value = 0.5
    sb.inputs["Specular IOR Level"].default_value = 0.45
    sb.inputs["Subsurface Weight"].default_value = 0.08
    sb.inputs["Coat Weight"].default_value = 0.15
    ng = bpy.data.node_groups.new("ToPoints", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    rad = ng.nodes.new("GeometryNodeInputNamedAttribute")
    rad.data_type = "FLOAT"
    rad.inputs["Name"].default_value = "rad"
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = sm
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
    ng.links.new(m2p.outputs["Points"], setm.inputs["Geometry"])
    ng.links.new(setm.outputs["Geometry"], go.inputs[0])
    mod = ob.modifiers.new("pts", "NODES")
    mod.node_group = ng
    scene.render.filepath = OUTP
    bpy.ops.render.render(write_still=True)
    print("done")
    raise SystemExit

# ---------------------------------------------------------------- plates along the camera path
if POSES:
    import time
    from mathutils import Matrix
    sys.path.insert(0, HERE)
    import camera as CM
    poses = json.load(open(POSES))["poses"]
    ids = list(range(len(poses)))
    if RANGE:
        ids = ids[RANGE[0]:RANGE[1]]
    os.makedirs(OUTDIR, exist_ok=True)
    scene.render.use_persistent_data = True          # keep the BVH between keyframes
    cam_d.dof.use_dof = False
    cam_d.lens = CM.LENS

    def place(pose):
        loc, r, u, f = CM.basis(pose)
        m = Matrix.Identity(4)
        for i in range(3):
            m[i][0], m[i][1], m[i][2], m[i][3] = r[i], u[i], -f[i], loc[i]
        cam.matrix_world = m
        bpy.context.view_layer.update()

    for kind in ("glass", "bare"):
        for ob in glass_objs:
            ob.hide_render = kind == "bare"
        for i in ids:
            # pose 0 is the wide shot, held for a third of the film: more samples
            scene.cycles.samples = (max(SAMPLES, 160) if i == 0 else SAMPLES) if kind == "glass" else BARE_SAMPLES
            path = os.path.join(OUTDIR, f"{kind[0]}_{i:03d}.png")
            if os.path.exists(path):
                continue
            place(poses[i])
            t0 = time.time()
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print(f"PLATE {kind} {i} {time.time() - t0:.1f}s", flush=True)
    print("PLATES_DONE")
    raise SystemExit

# ---------------------------------------------------------------- render both plates
suffix = "_test" if TEST else ""
scene.render.filepath = os.path.join(DATA, f"plate_glass{suffix}.png")
bpy.ops.render.render(write_still=True)
for ob in glass_objs:
    ob.hide_render = True
scene.render.filepath = os.path.join(DATA, f"plate_bare{suffix}.png")
bpy.ops.render.render(write_still=True)
print("done")
