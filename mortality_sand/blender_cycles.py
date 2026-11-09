"""Build the Cycles project for the film: cycles/film_cycles.blend, a scene to render in Blender's Cycles.

    python export_cycles.py                                   # first: the cache (sand, lifted grains, text)
    blender -b -P blender_cycles.py                           # then: build cycles/film_cycles.blend
    blender -b -P blender_cycles.py -- --test 1 6050 --samples 32 --percent 50   # + test frames

The .blend holds two scenes:
  Sand   the world in Cycles: the dark map table and its glass vessels (as blender_scene.py builds them for
         the painted film's plates), every visible grain of sand as a lit sphere, and the lifted grains
         rising out of the frame, glowing in their cause's colour, with motion blur. A script stored in the
         file ("sand_loader.py", registered to run when the file opens) fills the grains every frame from
         the cache/ folder beside the .blend. Renders PNG frames to frames/.
  Edit   the sequencer: those frames, the film's text layer over them (overlay_matte.mp4 multiplied,
         overlay_fill.mp4 added) and the narration. Renders the finished MP4.
Blender 4.2. Paths are relative: keep the .blend, cache/, the two overlay movies and narration.m4a together.
"""
import os, sys
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "cycles")
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


SAMPLES = opt("--samples", 128, int)
TEST = opt("--test", [], int, n=0)
PERCENT = opt("--percent", 100, int)
EXPOSURE = opt("--exposure", 0.4, float)

sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]                      # blender_scene.py reads its own flags: none
import numpy as np
import blender_scene as BS                     # the world, the table, the glass, the lights, the camera

st = np.load(os.path.join(OUTD, "cache", "static.npz"))
NF = int(st["nframes"])
FPS = int(st["fps"])

scene = bpy.context.scene
scene.name = "Sand"
scene.frame_start, scene.frame_end = 1, NF
scene.render.fps = FPS
scene.render.resolution_percentage = 100
scene.view_settings.exposure = EXPOSURE      # the lit sand as bright as the painted film's
scene.cycles.device = "GPU"                   # renders on the CPU where no GPU is set up in Preferences
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.02
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.render.use_persistent_data = True       # keep the glass between frames: only the grains change
scene.render.use_lock_interface = True        # the grains are refilled by a script each frame
scene.render.use_motion_blur = True           # streaks for the rising grains (they carry a velocity)
scene.render.motion_blur_shutter = 0.5
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
scene.render.filepath = "//frames/"
scene.render.use_overwrite = False            # stop and start again: finished frames are kept
scene.render.use_placeholder = True           # several machines can share the frames folder
scene.render.use_file_extension = True


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


sand_mat = bpy.data.materials.new("Sand grain")
sand_mat.use_nodes = True
nt = sand_mat.node_tree
b = nt.nodes["Principled BSDF"]
nt.links.new(attr(nt, "col").outputs["Color"], b.inputs["Base Color"])
b.inputs["Roughness"].default_value = 0.5
b.inputs["Specular IOR Level"].default_value = 0.45
b.inputs["Subsurface Weight"].default_value = 0.08
b.inputs["Coat Weight"].default_value = 0.15

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

# ---------------------------------------------------------------- compositor: glare and a soft vignette
scene.use_nodes = True
cn = scene.node_tree
for n in list(cn.nodes):
    cn.nodes.remove(n)
rl = cn.nodes.new("CompositorNodeRLayers")
glare = cn.nodes.new("CompositorNodeGlare")
glare.glare_type = "FOG_GLOW"
glare.quality = "HIGH"
glare.threshold = 1.0
glare.size = 7
glare.mix = -0.35
ell = cn.nodes.new("CompositorNodeEllipseMask")
ell.width, ell.height = 1.25, 1.3
blur = cn.nodes.new("CompositorNodeBlur")
blur.filter_type = "FAST_GAUSS"
blur.use_relative = True
blur.factor_x = blur.factor_y = 30.0
rng = cn.nodes.new("CompositorNodeMapRange")
rng.inputs["To Min"].default_value = 0.72
mul = cn.nodes.new("CompositorNodeMixRGB")
mul.blend_type = "MULTIPLY"
comp = cn.nodes.new("CompositorNodeComposite")
cn.links.new(rl.outputs["Image"], glare.inputs["Image"])
cn.links.new(ell.outputs["Mask"], blur.inputs["Image"])
cn.links.new(blur.outputs["Image"], rng.inputs["Value"])
cn.links.new(glare.outputs["Image"], mul.inputs[1])
cn.links.new(rng.outputs["Value"], mul.inputs[2])
cn.links.new(mul.outputs["Image"], comp.inputs["Image"])
for i, n in enumerate((rl, glare, mul, comp)):
    n.location = (i * 260, 0)
for i, n in enumerate((ell, blur, rng)):
    n.location = (i * 260, -320)

# ---------------------------------------------------------------- the loader script, stored in the file
src = open(os.path.join(HERE, "grains.py")).read() + "\n\n" + open(os.path.join(HERE, "cycles_loader.py")).read()
txt = bpy.data.texts.new("sand_loader.py")
txt.write(src)
txt.use_module = True                         # runs when the file opens (allow scripts: see README)

# ---------------------------------------------------------------- Edit: frames + text + narration -> MP4
ed = bpy.data.scenes.new("Edit")
ed.frame_start, ed.frame_end = 1, NF
ed.render.fps = FPS
ed.render.resolution_x, ed.render.resolution_y = 1920, 1080
ed.render.resolution_percentage = 100
ed.view_settings.view_transform = "Standard"   # the frames are finished pictures: no second tone mapping
ed.view_settings.look = "None"
seq = ed.sequence_editor_create()
black = seq.sequences.new_effect("Black", "COLOR", 1, frame_start=1, frame_end=NF + 1)
black.color = (0, 0, 0)
fr = seq.sequences.new_image("Cycles frames", "//frames/0001.png", 2, 1, fit_method="ORIGINAL")
for i in range(2, NF + 1):
    fr.elements.append(f"{i:04d}.png")
# frames rendered as a draft (Sand scene at less than 100%) are scaled up to fill the picture
for axis in ("scale_x", "scale_y"):
    dv = fr.transform.driver_add(axis).driver
    dv.type = "SCRIPTED"
    v = dv.variables.new()
    v.name = "pct"
    v.targets[0].id_type = "SCENE"
    v.targets[0].id = scene
    v.targets[0].data_path = "render.resolution_percentage"
    dv.expression = "100 / pct"
for ch, name, fn, blend in ((3, "Text matte", "overlay_matte.mp4", "MULTIPLY"),
                            (4, "Text", "overlay_fill.mp4", "ADD")):
    m = seq.sequences.new_movie(name, os.path.join(OUTD, fn), ch, 1, fit_method="FIT")
    m.blend_type = blend
nar = os.path.join(OUTD, "narration.m4a")
if os.path.exists(nar):
    seq.sequences.new_sound("Narration", nar, 5, 1)
ed.render.image_settings.file_format = "FFMPEG"
ed.render.ffmpeg.format = "MPEG4"
ed.render.ffmpeg.codec = "H264"
ed.render.ffmpeg.constant_rate_factor = "HIGH"
ed.render.ffmpeg.ffmpeg_preset = "GOOD"
ed.render.ffmpeg.audio_codec = "AAC"
ed.render.ffmpeg.audio_bitrate = 192
ed.render.filepath = "//two_thousand_years_of_catastrophe_cycles.mp4"
ed.render.use_file_extension = False
ed.render.use_sequencer = True
ed.render.use_compositing = False

# ---------------------------------------------------------------- save (images packed, paths relative)
for img in bpy.data.images:
    if img.source == "FILE" and img.filepath and not img.packed_file:
        img.pack()
blend = os.path.join(OUTD, "film_cycles.blend")
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
