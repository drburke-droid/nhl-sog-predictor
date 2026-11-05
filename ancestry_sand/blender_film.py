"""Build the Cycles project for the whole film (run export_blender.py first):

    blender -b -P blender_film.py                  # -> out/blender_film/every_grain_of_us.blend

Two scenes in one .blend:
  Film  the table, the glass, the sand and the flying sand, rendered by Cycles to render/frame_####.png.
        The sand is a point cloud of every grain the film ever shows; Geometry Nodes keeps the grains
        alive at the current frame and eases their colours, so it renders with no Python at render time.
        Settings are tuned for speed (adaptive sampling, denoising, fewer bounces, no caustics).
  Edit  the video editor: the rendered frames, the HUD overlay (hud.mov: year, legend, labels) and the
        title as editable text strips, written to every_grain_of_us.mp4. Add the voice-over here.
"""
import importlib.util, json, math, os, sys
import numpy as np
import bpy
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTD = os.environ.get("SAND_BLENDER_OUT", os.path.join(HERE, "out", "blender_film"))
BLEND = os.path.join(OUTD, "every_grain_of_us.blend")
FONT_DIR = "/usr/share/fonts/opentype/inter"
FPS = 30

spec = importlib.util.spec_from_file_location("blender_scene", os.path.join(HERE, "blender_scene.py"))
BS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BS)                      # table, glass, lights, studio world, camera
import camera as CM
import story as S

scene, cam, cam_d = BS.scene, BS.cam, BS.cam_d
scene.name = "Film"
G = np.load(os.path.join(OUTD, "grains.npz"))
FL = np.load(os.path.join(OUTD, "flights.npz"))
NFRAMES = int(G["frames"])
KS = int(FL["ks"])


# ================================================================ node helpers
def sock(socks, name):
    """The socket called `name` (by identifier first, then the first enabled one by name)."""
    for s in socks:
        if s.identifier == name:
            return s
    for s in socks:
        if s.name == name and s.enabled:
            return s
    raise KeyError(name)


class Tree:
    def __init__(self, nt):
        self.nt = nt

    def node(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def val(self, sock_, v):
        """Connect a socket or set a constant."""
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock_)
        else:
            sock_.default_value = v

    def math(self, op, a, b=0.0, c=0.0, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.val(n.inputs[0], a); self.val(n.inputs[1], b); self.val(n.inputs[2], c)
        return n.outputs[0]

    def cmp(self, op, a, b):
        n = self.node("FunctionNodeCompare", data_type="FLOAT", operation=op)
        self.val(sock(n.inputs, "A"), a); self.val(sock(n.inputs, "B"), b)
        return n.outputs["Result"]

    def boolean(self, op, a, b=None):
        n = self.node("FunctionNodeBooleanMath", operation=op)
        self.val(n.inputs[0], a)
        if b is not None:
            self.val(n.inputs[1], b)
        return n.outputs[0]

    def attr(self, name, kind="FLOAT"):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=kind)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def mix(self, kind, fac, a, b):
        n = self.node("ShaderNodeMix", data_type=kind)
        suf = {"RGBA": "Color", "VECTOR": "Vector", "FLOAT": "Float"}[kind]
        self.val(sock(n.inputs, "Factor_Float"), fac)
        self.val(sock(n.inputs, f"A_{suf}"), a); self.val(sock(n.inputs, f"B_{suf}"), b)
        return sock(n.outputs, f"Result_{suf}")

    def store(self, geo, name, kind, value):
        n = self.node("GeometryNodeStoreNamedAttribute", data_type=kind, domain="POINT")
        self.link(geo, n.inputs["Geometry"])
        n.inputs["Name"].default_value = name
        self.val(sock(n.inputs, "Value"), value)
        return n.outputs["Geometry"]


def geo_tree(name):
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    T = Tree(ng)
    return ng, T, T.node("NodeGroupInput"), T.node("NodeGroupOutput")


def point_mesh(name, co, attrs):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", np.ascontiguousarray(co, np.float32).ravel())
    for k, v in attrs.items():
        v = np.asarray(v, np.float32)
        if v.ndim == 2:
            a = me.attributes.new(k, "FLOAT_COLOR", "POINT")
            a.data.foreach_set("color", np.concatenate([v, np.ones((len(v), 1), np.float32)], 1).ravel())
        else:
            a = me.attributes.new(k, "FLOAT", "POINT")
            a.data.foreach_set("value", v)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    return ob


def sand_material(name, emit_scale):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    col = nt.nodes.new("ShaderNodeAttribute"); col.attribute_name = "col"; col.attribute_type = "GEOMETRY"
    emi = nt.nodes.new("ShaderNodeAttribute"); emi.attribute_name = "emit"; emi.attribute_type = "GEOMETRY"
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = emit_scale
    nt.links.new(col.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(col.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(emi.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Specular IOR Level"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.15
    return m


# ================================================================ the resting sand
sand = point_mesh("Sand", G["pos"], dict(rad=G["rad"], tb=G["tb"], td=G["td"], tc=G["tc"], ts=G["ts"],
                                         c0=G["c0"], c1=G["c1"]))
ng, T, gi, go = geo_tree("SandOverTime")
frame = T.node("GeometryNodeInputSceneTime").outputs["Frame"]
pts = T.node("GeometryNodeMeshToPoints", mode="VERTICES")
T.link(gi.outputs[0], pts.inputs["Mesh"])
alive = T.boolean("AND", T.cmp("GREATER_EQUAL", frame, T.attr("tb")), T.cmp("LESS_THAN", frame, T.attr("td")))
dele = T.node("GeometryNodeDeleteGeometry", domain="POINT")
T.link(pts.outputs["Points"], dele.inputs["Geometry"])
T.link(T.boolean("NOT", alive), dele.inputs["Selection"])
# new grains on top swell in: scale 0.3 -> 1 over ~3 frames
scale = T.math("MULTIPLY_ADD", T.math("SUBTRACT", frame, T.attr("ts")), 0.22, 0.3, clamp=True)
rad = T.math("MULTIPLY", T.attr("rad"), scale)
srad = T.node("GeometryNodeSetPointRadius")
T.link(dele.outputs["Geometry"], srad.inputs["Points"]); T.link(rad, srad.inputs["Radius"])
cz = T.node("ShaderNodeCombineXYZ"); T.link(rad, cz.inputs["Z"])
spos = T.node("GeometryNodeSetPosition")
T.link(srad.outputs["Points"], spos.inputs["Geometry"]); T.link(cz.outputs[0], spos.inputs["Offset"])
# colours ease toward their new value: c = c1 + (c0 - c1) * 0.82^(frame - tc)
w = T.math("POWER", 0.82, T.math("MAXIMUM", T.math("SUBTRACT", frame, T.attr("tc")), 0.0))
col = T.mix("RGBA", w, T.attr("c1", "FLOAT_COLOR"), T.attr("c0", "FLOAT_COLOR"))
# the closing ripple of light that runs out from East Africa
tw0 = (S.TODAY_S + 9.4) * FPS
ax, ay = BS.px2w(*G["eaf"])[:2]
tw = T.math("SUBTRACT", frame, tw0)                                       # frames since it started
ring = T.math("MULTIPLY", tw, 3.30 / FPS)                                 # radius, BU (330 px/s)
P = T.node("GeometryNodeInputPosition").outputs[0]
sep = T.node("ShaderNodeSeparateXYZ"); T.link(P, sep.inputs[0])
dx = T.math("SUBTRACT", sep.outputs["X"], ax); dy = T.math("SUBTRACT", sep.outputs["Y"], ay)
dd = T.math("SQRT", T.math("ADD", T.math("MULTIPLY", dx, dx), T.math("MULTIPLY", dy, dy)))
q = T.math("DIVIDE", T.math("SUBTRACT", dd, ring), 0.6)
gauss = T.math("EXPONENT", T.math("MULTIPLY", T.math("MULTIPLY", q, q), -1.0))
mr = T.node("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
T.val(mr.inputs["Value"], T.math("SUBTRACT", tw0 + 6.5 * FPS, frame))
mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 1.5 * FPS
on = T.math("MULTIPLY", T.cmp("GREATER_EQUAL", tw, 0.0), mr.outputs["Result"])
wv = T.math("MULTIPLY", T.math("MULTIPLY", gauss, on), 0.85)
col = T.mix("RGBA", wv, col, (1.0, 0.92, 0.68, 1.0))
geo = T.store(spos.outputs["Geometry"], "col", "FLOAT_COLOR", col)
geo = T.store(geo, "emit", "FLOAT", wv)
setm = T.node("GeometryNodeSetMaterial")
T.link(geo, setm.inputs["Geometry"])
setm.inputs["Material"].default_value = sand_material("Sand", 2.5)
T.link(setm.outputs["Geometry"], go.inputs[0])
sand.modifiers.new("SandOverTime", "NODES").node_group = ng

# ================================================================ the flying sand
# One mesh holds every flight path (KS samples per grain) followed by the grains themselves (four points
# each: the grain and a short streak behind it). Geometry Nodes moves each grain along its path.
TRAIL = [(0.0, 1.0), (0.012, 0.8), (0.024, 0.6), (0.036, 0.45)]
npart = len(FL["t0"])
npath = npart * KS
rep = lambda a: np.repeat(np.asarray(a, np.float32), len(TRAIL), axis=0)
nfly = npart * len(TRAIL)
cat = lambda path_vals, fly_vals: np.concatenate([np.asarray(path_vals, np.float32), np.asarray(fly_vals, np.float32)])
fly = point_mesh("FlyingSand", np.concatenate([FL["pos"].reshape(-1, 3), np.zeros((nfly, 3), np.float32)]), dict(
    isfly=cat(np.zeros(npath), np.ones(nfly)),
    okseg=cat(FL["ok"].reshape(-1), np.zeros(nfly)),
    pid=cat(np.zeros(npath), rep(np.arange(npart))), t0=cat(np.zeros(npath), rep(FL["t0"])),
    dur=cat(np.ones(npath), rep(FL["dur"])),
    back=cat(np.zeros(npath), np.tile([b for b, _ in TRAIL], npart)),
    shrink=cat(np.zeros(npath), np.tile([k for _, k in TRAIL], npart)),
    col=np.concatenate([np.zeros((npath, 3), np.float32), rep(FL["col"])])))
ng, T, gi, go = geo_tree("FlightsOverTime")
frame = T.node("GeometryNodeInputSceneTime").outputs["Frame"]
pts = T.node("GeometryNodeMeshToPoints", mode="VERTICES")
T.link(gi.outputs[0], pts.inputs["Mesh"])
split = T.node("GeometryNodeSeparateGeometry", domain="POINT")
T.link(pts.outputs["Points"], split.inputs["Geometry"])
T.link(T.cmp("GREATER_THAN", T.attr("isfly"), 0.5), split.inputs["Selection"])
grains, path_geo = split.outputs["Selection"], split.outputs["Inverted"]
u = T.math("DIVIDE", T.math("SUBTRACT", frame, T.attr("t0")), T.attr("dur"))
uu = T.math("SUBTRACT", u, T.attr("back"), clamp=True)
kf = T.math("MULTIPLY", uu, KS - 1)
k0 = T.math("FLOOR", T.math("MINIMUM", kf, KS - 2))
frac = T.math("SUBTRACT", kf, k0)
i0 = T.math("ADD", T.math("MULTIPLY", T.attr("pid"), KS), k0)


def sample(value_sock, kind, index):
    n = T.node("GeometryNodeSampleIndex", data_type=kind, domain="POINT")
    T.link(path_geo, n.inputs["Geometry"])
    T.link(value_sock, sock(n.inputs, "Value"))
    T.val(n.inputs["Index"], index)
    return sock(n.outputs, "Value")


Pp = T.node("GeometryNodeInputPosition").outputs[0]
p0 = sample(Pp, "FLOAT_VECTOR", i0)
p1 = sample(Pp, "FLOAT_VECTOR", T.math("ADD", i0, 1.0))
okv = sample(T.attr("okseg"), "FLOAT", i0)
alive = T.boolean("AND", T.boolean("AND", T.cmp("GREATER_THAN", u, 0.0), T.cmp("LESS_THAN", u, 1.0)),
                  T.boolean("AND", T.cmp("GREATER_THAN", uu, 0.04), T.cmp("GREATER_THAN", okv, 0.5)))
fade = T.math("MINIMUM", T.math("DIVIDE", uu, 0.15),
              T.math("ADD", T.math("DIVIDE", T.math("SUBTRACT", 1.0, uu), 0.05), 0.3), clamp=True)
# evaluate the path samples while every grain is still present, then drop the grains not in flight
spos = T.node("GeometryNodeSetPosition")
T.link(grains, spos.inputs["Geometry"])
T.link(T.mix("VECTOR", frac, p0, p1), spos.inputs["Position"])
rad = T.math("MULTIPLY", T.math("MULTIPLY", T.attr("shrink"), 0.013), T.math("MULTIPLY_ADD", fade, 0.6, 0.4))
srad = T.node("GeometryNodeSetPointRadius")
T.link(spos.outputs["Geometry"], srad.inputs["Points"]); T.link(rad, srad.inputs["Radius"])
geo = T.store(srad.outputs["Points"], "emit", "FLOAT", fade)
geo = T.store(geo, "keep", "BOOLEAN", alive)
dele = T.node("GeometryNodeDeleteGeometry", domain="POINT")
T.link(geo, dele.inputs["Geometry"]); T.link(T.boolean("NOT", T.attr("keep", "BOOLEAN")), dele.inputs["Selection"])
setm = T.node("GeometryNodeSetMaterial")
T.link(dele.outputs["Geometry"], setm.inputs["Geometry"])
setm.inputs["Material"].default_value = sand_material("FlyingSand", 1.4)
T.link(setm.outputs["Geometry"], go.inputs[0])
fly.modifiers.new("FlightsOverTime", "NODES").node_group = ng

# ================================================================ the camera: still, then the finale
cam.rotation_mode = "QUATERNION"
cam_d.lens = CM.LENS
cam_d.dof.use_dof = True
cam_d.dof.aperture_fstop = 0.9
prev, q_prev = None, None
moving = [f for f in range(NFRAMES) if CM.pose_at(f / FPS) != CM.pose_at(max(f - 1, 0) / FPS)]
keyf = sorted(set([0, NFRAMES - 1] + moving + [f - 1 for f in moving] + [f + 1 for f in moving if f + 1 < NFRAMES]))
for f in keyf:
    pose = CM.pose_at(f / FPS)
    loc, r, uax, fwd = CM.basis(pose)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = r[i], uax[i], -fwd[i], loc[i]
    q = m.to_quaternion()
    if q_prev is not None and q.dot(q_prev) < 0:
        q.negate()
    q_prev = q
    cam.location = m.to_translation()
    cam.rotation_quaternion = q
    cam_d.dof.focus_distance = pose[3]
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_quaternion", frame=f)
    cam_d.dof.keyframe_insert("focus_distance", frame=f)
for ad in (cam.animation_data, cam_d.animation_data):
    for fc in ad.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

# ================================================================ render settings: quality per minute
scene.frame_start, scene.frame_end = 0, NFRAMES - 1
scene.render.fps = FPS
c = scene.cycles
c.device = "GPU"                     # falls back to the CPU when no GPU is enabled in Preferences
c.samples = 64
c.use_adaptive_sampling = True
c.adaptive_threshold = 0.04          # stop each pixel once it is clean enough
c.adaptive_min_samples = 0
c.use_denoising = True
c.denoiser = "OPENIMAGEDENOISE"
c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
c.denoising_prefilter = "ACCURATE"
c.denoising_use_gpu = True
c.max_bounces = 6
c.diffuse_bounces = 1
c.glossy_bounces = 3
c.transmission_bounces = 6           # enough to see sand through two glass walls
c.transparent_max_bounces = 4
c.volume_bounces = 0
c.caustics_reflective = False
c.caustics_refractive = False
c.blur_glossy = 1.0
c.sample_clamp_indirect = 3.0
c.seed = 0
c.use_animated_seed = False      # the same noise pattern every frame: no shimmer while the camera is still
c.use_auto_tile = True
scene.render.use_persistent_data = True       # keep the static table and glass between frames
scene.render.filter_size = 1.5
scene.render.use_motion_blur = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 15
scene.render.filepath = "//render/frame_"
scene.render.use_overwrite = False            # stop and restart any time: finished frames are kept
scene.render.use_placeholder = True           # several computers can share the job
scene.render.use_file_extension = True

# compositor: a soft bloom on the bright flying sand, and a vignette
scene.use_nodes = True
ct = Tree(scene.node_tree)
for n in list(scene.node_tree.nodes):
    if n.bl_idname not in ("CompositorNodeRLayers", "CompositorNodeComposite"):
        scene.node_tree.nodes.remove(n)
rl = scene.node_tree.nodes["Render Layers"]
comp = scene.node_tree.nodes["Composite"]
glare = ct.node("CompositorNodeGlare", glare_type="BLOOM", quality="HIGH", threshold=0.9, mix=-0.6, size=7)
ct.link(rl.outputs["Image"], glare.inputs["Image"])
ell = ct.node("CompositorNodeEllipseMask", width=0.98, height=1.0)
blur = ct.node("CompositorNodeBlur", filter_type="FAST_GAUSS", use_relative=True, factor_x=30, factor_y=30)
ct.link(ell.outputs["Mask"], blur.inputs["Image"])
vig = ct.node("CompositorNodeMixRGB", blend_type="MULTIPLY")
vig.inputs["Fac"].default_value = 0.32
ct.link(glare.outputs["Image"], vig.inputs[1]); ct.link(blur.outputs["Image"], vig.inputs[2])
ct.link(vig.outputs["Image"], comp.inputs["Image"])

BS.table.data.materials[0].node_tree.nodes["Image Texture"].image.pack()
os.makedirs(OUTD, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)

# ================================================================ Edit: frames + HUD + title -> MP4
ed = bpy.data.scenes.new("Edit")
ed.render.resolution_x, ed.render.resolution_y = 1920, 1080
ed.render.resolution_percentage = 100
ed.render.fps = FPS
ed.frame_start, ed.frame_end = 0, NFRAMES - 1
ed.view_settings.view_transform = "Standard"
ed.render.use_sequencer = True
ed.render.use_compositing = False
sed = ed.sequence_editor_create()
frames = sed.sequences.new_image("Render", os.path.join(OUTD, "render", "frame_0000.png"), channel=1, frame_start=0)
for f in range(1, NFRAMES):
    frames.elements.append(f"frame_{f:04d}.png")
hud = sed.sequences.new_movie("HUD", os.path.join(OUTD, "hud.mov"), channel=2, frame_start=0)
hud.blend_type = "ALPHA_OVER"
t_on, t_off = int(4.2 * FPS), int(5.8 * FPS)                  # the title fades between these frames
shade = sed.sequences.new_effect("TitleShade", "COLOR", channel=3, frame_start=0, frame_end=t_off + 1)
shade.color = (0, 0, 0)
shade.blend_type = "ALPHA_OVER"


def fade_out(strip, a0):
    strip.blend_alpha = a0
    strip.keyframe_insert("blend_alpha", frame=t_on)
    strip.blend_alpha = 0.0
    strip.keyframe_insert("blend_alpha", frame=t_off)


fade_out(shade, 0.55)
INK = (0xe8 / 255, 0xee / 255, 0xf8 / 255, 1.0)
MUTED = (0x93 / 255, 0xa1 / 255, 0xb8 / 255, 1.0)
for name, text, ch, fnt, size, y, colr in (
        ("Title", S.TITLE, 4, "InterDisplay-SemiBold.otf", 84, 0.565, INK),
        ("Subtitle", S.SUBTITLE, 5, "Inter-Regular.otf", 28, 0.494, MUTED)):
    t = sed.sequences.new_effect(name, "TEXT", channel=ch, frame_start=0, frame_end=t_off + 1)
    t.text = text
    t.font = bpy.data.fonts.load(os.path.join(FONT_DIR, fnt))
    t.font.pack()
    t.font_size = size
    t.color = colr
    t.location = (0.5, y)
    t.align_x, t.align_y = "CENTER", "CENTER"
    t.blend_type = "ALPHA_OVER"
    fade_out(t, 1.0)
ed.render.image_settings.file_format = "FFMPEG"
ed.render.ffmpeg.format = "MPEG4"
ed.render.ffmpeg.codec = "H264"
ed.render.ffmpeg.constant_rate_factor = "HIGH"
ed.render.ffmpeg.ffmpeg_preset = "GOOD"
ed.render.ffmpeg.audio_codec = "AAC"
ed.render.filepath = "//every_grain_of_us.mp4"
bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
print("saved", BLEND, "frames", NFRAMES, "grain records", len(G["tb"]), "flying grains", npart)
