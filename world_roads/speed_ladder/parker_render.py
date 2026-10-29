"""Blender renders for the Parker finale: the props with relighting passes (the fireball relights them in the
compositor), and the probe itself for the few replay frames it survives, its heat shield and then its whole body
glowing hotter as it boils away.
  bpy-python parker_render.py -> /home/user/lot/render/parker/static.npz, p####.npz (replay frames)"""
import bpy, sys, os, math, time
import numpy as np
from mathutils import Vector, Euler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, layout as L, aircraft as A, parker as P
from exr import save

OUT = "/home/user/lot/render/parker"
os.makedirs(OUT, exist_ok=True)
s = S.reset(samples=int(os.environ.get("SAMPLES", "48")))
vl = s.view_layers[0]
vl.use_pass_z = True; vl.use_pass_normal = True; vl.use_pass_position = True; vl.use_pass_diffuse_color = True
s.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
s.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
s.render.image_settings.color_depth = "16"
L.build()
root, _ = A.build("parker")
root.rotation_mode = "XYZ"

# heat: every probe surface emits as a blackbody; the temperature node values are set per frame
temps = []
done = set()
for o in bpy.data.objects:
    if o.type != "MESH" or o.parent is None: continue
    par = o
    while par is not None and par is not root: par = par.parent
    if par is None: continue
    for m in o.data.materials:
        if m is None or m.name in done or not m.use_nodes: continue
        done.add(m.name)
        nt = m.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None: continue
        N, Lk = nt.nodes.new, nt.links.new
        geo = N("ShaderNodeNewGeometry"); sep = N("ShaderNodeSeparateXYZ"); Lk(geo.outputs["Normal"], sep.inputs[0])
        f = N("ShaderNodeMath"); f.operation = "MAXIMUM"; f.inputs[1].default_value = 0.0; Lk(sep.outputs["X"], f.inputs[0])
        front = N("ShaderNodeValue"); side = N("ShaderNodeValue")        # K facing the flow, K side-on (set per frame)
        mix = N("ShaderNodeMapRange"); Lk(f.outputs[0], mix.inputs["Value"])
        Lk(side.outputs[0], mix.inputs["To Min"]); Lk(front.outputs[0], mix.inputs["To Max"])
        bb = N("ShaderNodeBlackbody"); Lk(mix.outputs["Result"], bb.inputs["Temperature"])
        # visible brightness of a blackbody against a sunlit white surface: 70 at 2,300 K, climbing as exp(-24000/T)
        dv = N("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[0].default_value = 24000.0; Lk(mix.outputs["Result"], dv.inputs[1])
        sub = N("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.inputs[0].default_value = 24000 / 2300; Lk(dv.outputs[0], sub.inputs[1])
        ex = N("ShaderNodeMath"); ex.operation = "EXPONENT"; Lk(sub.outputs[0], ex.inputs[0])
        cap = N("ShaderNodeMath"); cap.operation = "MINIMUM"; cap.inputs[1].default_value = 3000.0; Lk(ex.outputs[0], cap.inputs[0])
        st = N("ShaderNodeMath"); st.operation = "MULTIPLY"; st.inputs[1].default_value = 60.0; Lk(cap.outputs[0], st.inputs[0])
        Lk(bb.outputs["Color"], bsdf.inputs["Emission Color"]); Lk(st.outputs[0], bsdf.inputs["Emission Strength"])
        temps.append((front, side))


def heat(t):
    """K of the surfaces facing the flow and side-on, t after the probe met the air"""
    if t <= 0: return 300.0, 300.0
    u = min(t / P.T_GONE, 1.0)
    return 3500.0 + 3000.0 * min(t / P.T_SHIELD, 1.0), 300.0 + 3500.0 * u ** 0.7


def render(path, t):
    x = float(P.x_head(t)) - root["length"] / 2 if t > -1 else 0.0
    root.location = (x, P.PY, P.PZ) if t > -1 else (0, -500, -500)
    fr, sd = heat(t)
    for a, b in temps: a.outputs[0].default_value = fr; b.outputs[0].default_value = sd
    s.render.filepath = path.replace(".npz", ".exr")
    bpy.ops.render.render(write_still=True)
    save(path, s.render.filepath, passes=True)


if not os.path.exists(f"{OUT}/static.npz"):
    render(f"{OUT}/static.npz", -2.0)
for n, (t, slow) in enumerate(P.replay()):
    if t > P.T_GONE * 0.9: break
    path = f"{OUT}/p{n:04d}.npz"
    if os.path.exists(path): continue
    t0 = time.time()
    render(path, t)
    print(f"parker {n} t={t * 1e6:.1f} us {time.time() - t0:.1f}s", flush=True)
print("ALLDONE")
