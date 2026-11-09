"""Build the Cycles project for the film in 4K: cycles/two_thousand_years_of_catastrophe.blend.

    python export_cycles.py                                   # first: the cache (sand, lifted grains, 4K text)
    blender -b -P blender_cycles.py                           # then: build the .blend
    blender -b -P blender_cycles.py -- --test 6007 --percent 50   # + test frames (film frame numbers)

The render settings are the migration film's (ancestry_sand/blender_film.py, "Every grain of us"): 3840x2160,
Cycles on the GPU at 64 adaptive samples (threshold 0.04) with OpenImageDenoise on albedo and normals,
trimmed bounces and no caustics, a fixed noise seed, persistent data, motion blur with a half-frame
shutter, the AgX High Contrast grade at -0.15 exposure, bloom and a vignette in the compositor, an f/0.9
lens, and its richer sand (each grain a little different in hue, shade and shine). PNG frames go to
render4k/frame_0000.png and on.

The .blend holds two scenes:
  Film   the world in Cycles: the dark map table and its glass vessels (as blender_scene.py builds them for
         the painted film's plates), every visible grain of sand as a lit sphere, and the lifted grains
         rising out of the frame, glowing in their cause's colour, with motion blur. A script stored in the
         file ("sand_loader.py", registered to run when the file opens) fills the grains every frame from
         the cache/ folder beside the .blend. Blender frame N is film frame N.
  Edit   the sequencer: those frames, the film's 4K text layer over them (overlay_matte.mp4 multiplied,
         overlay_fill.mp4 added) and the narration. Renders the finished MP4.
Blender 4.2 or later. Paths are relative: keep the .blend, cache/, the overlay movies and narration.m4a together.
"""
import json, os, sys
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "cycles")
RES = (3840, 2160)
RENDER_DIR = "render4k"
NAME = "two_thousand_years_of_catastrophe"
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default, cast=str, n=1):
    if name not in argv:
        return default
    i = argv.index(name)
    vals = []
    for v in argv[i + 1:]:
        if v.startswith("--") or (n == 1 and vals):
            break
        vals.append(cast(v))
    return vals if n != 1 else vals[0]


SAMPLES = opt("--samples", 64, int)
TEST = opt("--test", [], int, n=0)
PERCENT = opt("--percent", 100, int)
EXPOSURE = opt("--exposure", -0.15, float)

sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]                      # blender_scene.py reads its own flags: none
import numpy as np
import blender_scene as BS                     # the world, the table, the glass, the lights, the camera

st = np.load(os.path.join(OUTD, "cache", "static.npz"))
NF = int(st["nframes"])
FPS = int(st["fps"])
# the finished film: the rendered frames, then the last one held under the closing words (story.HOLD)
_ej = os.path.join(OUTD, "cache", "edit.json")
NE = max(json.load(open(_ej))["frames"], NF) if os.path.exists(_ej) else NF

scene = bpy.context.scene
scene.name = "Film"
scene.frame_start, scene.frame_end = 0, NF - 1
scene.render.fps = FPS

# ---------------------------------------------------------------- render settings: the migration film's
c = scene.cycles
c.device = "GPU"                     # falls back to the CPU when no GPU is enabled in Preferences
c.samples = SAMPLES
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
c.use_animated_seed = False          # the same noise pattern every frame: no shimmer, the camera is still
c.use_auto_tile = True
scene.render.use_persistent_data = True       # keep the static table and glass between frames
scene.render.use_lock_interface = True        # the grains are refilled by a script each frame
scene.render.filter_size = 1.5
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.resolution_percentage = 100      # 50 gives a quick 1920x1080 draft
scene.view_settings.view_transform = "AgX"    # the grade: deeper blacks and richer colour, a touch darker
scene.view_settings.look = "AgX - High Contrast"
scene.view_settings.exposure = EXPOSURE
scene.render.use_motion_blur = True           # the rising grains streak (they carry a velocity)
scene.render.motion_blur_shutter = 0.5
c.motion_blur_position = "CENTER"
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.image_settings.compression = 15
scene.render.filepath = f"//{RENDER_DIR}/frame_"
scene.render.use_overwrite = False            # stop and restart any time: finished frames are kept
scene.render.use_placeholder = True           # several computers can share the job
scene.render.use_file_extension = True
# the lens: f/0.9 focused on the table, as the migration film's (only grains rising close to the camera blur)
BS.cam_d.dof.use_dof = True
BS.cam_d.dof.aperture_fstop = 0.9
BS.cam_d.dof.focus_distance = BS.dist


# ---------------------------------------------------------------- the grains: two point clouds
def points_object(name, material):
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ng = bpy.data.node_groups.new(f"{name}Points", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    rad = ng.nodes.new("GeometryNodeInputNamedAttribute")
    rad.data_type = "FLOAT"
    rad.inputs["Name"].default_value = "rad"
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = material
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
    ng.links.new(m2p.outputs["Points"], setm.inputs["Geometry"])
    ng.links.new(setm.outputs["Geometry"], go.inputs[0])
    gi.location, m2p.location, setm.location, go.location, rad.location = (-400, 0), (-150, 0), (80, 0), (300, 0), (-400, -150)
    ob.modifiers.new("points", "NODES").node_group = ng
    return ob


def attr(nt, name, kind="GEOMETRY"):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    a.attribute_type = kind
    return a


def sand_material(name):
    """The migration film's richer sand: the grain's colour from the simulation, each grain a little
    different in shade, hue and shine (a noise of its fixed position, so it never flickers), and slightly
    translucent."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    col = attr(nt, "col")
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Specular IOR Level"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.15
    pi = nt.nodes.new("ShaderNodePointInfo")
    sc = nt.nodes.new("ShaderNodeVectorMath"); sc.operation = "SCALE"; sc.inputs["Scale"].default_value = 61.7
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "3D"
    sp = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(pi.outputs["Position"], sc.inputs[0])
    nt.links.new(sc.outputs["Vector"], wn.inputs["Vector"])
    nt.links.new(wn.outputs["Color"], sp.inputs["Color"])

    def madd(x, a, c):
        n = nt.nodes.new("ShaderNodeMath"); n.operation = "MULTIPLY_ADD"
        nt.links.new(x, n.inputs[0]); n.inputs[1].default_value = a; n.inputs[2].default_value = c
        return n.outputs[0]

    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    nt.links.new(col.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(madd(sp.outputs["Red"], 0.04, 0.48), hsv.inputs["Hue"])          # hue +-0.02
    nt.links.new(madd(sp.outputs["Green"], 0.26, 0.87), hsv.inputs["Value"])     # brightness +-13%
    nt.links.new(madd(sp.outputs["Blue"], 0.18, 0.92), hsv.inputs["Saturation"])
    nt.links.new(hsv.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(madd(sp.outputs["Blue"], 0.3, 0.33), b.inputs["Roughness"])     # some grains shinier
    b.inputs["Subsurface Weight"].default_value = 0.08                          # a little light inside
    return m


sand_mat = sand_material("Sand grain")

lift_mat = bpy.data.materials.new("Lifted grain")
lift_mat.use_nodes = True
nt = lift_mat.node_tree
b = nt.nodes["Principled BSDF"]
sand_lin = np.where(st["sand_srgb"] <= 0.04045, st["sand_srgb"] / 12.92, ((st["sand_srgb"] + 0.055) / 1.055) ** 2.4)
mix = nt.nodes.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.inputs[6].default_value = (*[float(v) for v in sand_lin], 1.0)
nt.links.new(attr(nt, "glow").outputs["Fac"], mix.inputs["Factor"])
ecol = attr(nt, "ecol")
nt.links.new(ecol.outputs["Color"], mix.inputs[7])
nt.links.new(mix.outputs[2], b.inputs["Base Color"])
nt.links.new(ecol.outputs["Color"], b.inputs["Emission Color"])
nt.links.new(attr(nt, "estr").outputs["Fac"], b.inputs["Emission Strength"])
b.inputs["Roughness"].default_value = 0.5

sand_ob = points_object("Sand", sand_mat)
lift_ob = points_object("Lifted", lift_mat)
for ob, blur in ((sand_ob, False), (lift_ob, True)):
    try:
        ob.cycles.use_motion_blur = blur
        ob.cycles.use_deform_motion = blur
    except AttributeError:
        pass
    ob.visible_shadow = ob is sand_ob              # the rising grains glow; they cast no shadows

# ---------------------------------------------------------------- compositor: the migration film's bloom and vignette
scene.use_nodes = True
cn = scene.node_tree
for n in list(cn.nodes):
    if n.bl_idname not in ("CompositorNodeRLayers", "CompositorNodeComposite"):
        cn.nodes.remove(n)
rl = cn.nodes.get("Render Layers") or cn.nodes.new("CompositorNodeRLayers")
comp = cn.nodes.get("Composite") or cn.nodes.new("CompositorNodeComposite")
glare = cn.nodes.new("CompositorNodeGlare")
glare.glare_type, glare.quality, glare.threshold, glare.mix, glare.size = "BLOOM", "HIGH", 0.9, -0.6, 8
ell = cn.nodes.new("CompositorNodeEllipseMask")
ell.width, ell.height = 0.98, 1.0
blur = cn.nodes.new("CompositorNodeBlur")
blur.filter_type, blur.use_relative = "FAST_GAUSS", True
blur.factor_x = blur.factor_y = 30
vig = cn.nodes.new("CompositorNodeMixRGB")
vig.blend_type = "MULTIPLY"
vig.inputs["Fac"].default_value = 0.32
cn.links.new(rl.outputs["Image"], glare.inputs["Image"])
cn.links.new(ell.outputs["Mask"], blur.inputs["Image"])
cn.links.new(glare.outputs["Image"], vig.inputs[1])
cn.links.new(blur.outputs["Image"], vig.inputs[2])
cn.links.new(vig.outputs["Image"], comp.inputs["Image"])
for i, n in enumerate((rl, glare, vig, comp)):
    n.location = (i * 260, 0)
for i, n in enumerate((ell, blur)):
    n.location = (i * 260, -320)

# ---------------------------------------------------------------- the loader script, stored in the file
src = open(os.path.join(HERE, "grains.py")).read() + "\n\n" + open(os.path.join(HERE, "cycles_loader.py")).read()
txt = bpy.data.texts.new("sand_loader.py")
txt.write(src)
txt.use_module = True                         # runs when the file opens (allow scripts: see README)

# ---------------------------------------------------------------- Edit: frames + text + narration -> MP4
ed = bpy.data.scenes.new("Edit")
ed.frame_start, ed.frame_end = 0, NE - 1
ed.render.fps = FPS
ed.render.resolution_x, ed.render.resolution_y = RES
ed.render.resolution_percentage = 100
ed.view_settings.view_transform = "Standard"   # the frames are finished pictures: no second tone mapping
ed.view_settings.look = "None"
seq = ed.sequence_editor_create()
black = seq.sequences.new_effect("Black", "COLOR", 1, frame_start=0, frame_end=NE)
black.color = (0, 0, 0)
fr = seq.sequences.new_image("Cycles frames", f"//{RENDER_DIR}/frame_0000.png", 2, 0, fit_method="ORIGINAL")
for i in range(1, NF):
    fr.elements.append(f"frame_{i:04d}.png")
strips = [fr]
if NE > NF:                                   # the closing: the last rendered frame, held
    hold = seq.sequences.new_image("Closing (last frame held)", f"//{RENDER_DIR}/frame_{NF - 1:04d}.png", 2, NF,
                                   fit_method="ORIGINAL")
    hold.frame_final_duration = NE - NF
    strips.append(hold)
# frames rendered as a draft (Film scene at less than 100%) are scaled up to fill the picture
for strip in strips:
    for axis in ("scale_x", "scale_y"):
        dv = strip.transform.driver_add(axis).driver
        dv.type = "SCRIPTED"
        v = dv.variables.new()
        v.name = "pct"
        v.targets[0].id_type = "SCENE"
        v.targets[0].id = scene
        v.targets[0].data_path = "render.resolution_percentage"
        dv.expression = "100 / pct"
for ch, name, fn, blend in ((3, "Text matte", "overlay_matte.mp4", "MULTIPLY"),
                            (4, "Text", "overlay_fill.mp4", "ADD")):
    m = seq.sequences.new_movie(name, os.path.join(OUTD, fn), ch, 0, fit_method="FIT")
    m.blend_type = blend
nar = os.path.join(OUTD, "narration.m4a")
if os.path.exists(nar):
    seq.sequences.new_sound("Narration", nar, 5, 0)
ed.render.image_settings.file_format = "FFMPEG"
ed.render.ffmpeg.format = "MPEG4"
ed.render.ffmpeg.codec = "H264"
ed.render.ffmpeg.constant_rate_factor = "HIGH"
ed.render.ffmpeg.ffmpeg_preset = "GOOD"
ed.render.ffmpeg.audio_codec = "AAC"
ed.render.ffmpeg.audio_bitrate = 192
ed.render.filepath = f"//{NAME}_4k.mp4"
ed.render.use_file_extension = False
ed.render.use_sequencer = True
ed.render.use_compositing = False

# ---------------------------------------------------------------- save (images packed, paths relative)
for img in bpy.data.images:
    if img.source == "FILE" and img.filepath and not img.packed_file:
        img.pack()
blend = os.path.join(OUTD, f"{NAME}.blend")
if bpy.context.window_manager.windows:
    bpy.context.window_manager.windows[0].scene = scene
bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True, compress=True)
print("saved", blend)

# run the loader in this session too, for the test frames
exec(compile(src, "sand_loader.py", "exec"), {"__name__": "sand_loader"})

if TEST:
    os.makedirs(os.path.join(OUTD, "test"), exist_ok=True)
    scene.render.resolution_percentage = PERCENT
    import time
    for f in TEST:
        t0 = time.time()
        scene.frame_set(f)
        print(f"frame {f}: {len(sand_ob.data.vertices)} sand grains, {len(lift_ob.data.vertices)} rising", flush=True)
        scene.render.filepath = os.path.join(OUTD, "test", f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"TEST frame {f} rendered in {time.time() - t0:.0f}s", flush=True)
