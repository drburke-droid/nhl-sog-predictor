"""
Build the Blender scene for the powder-mixing conveyor animation from the
cache written by simulate.py, save it as a .blend, and optionally render.

Runs either inside Blender or with the `bpy` Python module:

    blender -b -P build_scene.py -- --save mixer.blend
    python build_scene.py --save mixer.blend --still 400 --res 3840x2160
    python build_scene.py --anim --res 1920x1080 --samples 48

Scene units are millimetres (1 unit = 1 mm) so the cache drops straight in.
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))

# ingredient colours (sRGB hex) - dried-powder palette
PALETTE = [
    ("Paprika",       "#D2362B"),
    ("Turmeric",      "#F2B31D"),
    ("Matcha",        "#62A33B"),
    ("Butterfly pea", "#2C6BD1"),
    ("Sugar",         "#F4EFE6"),
    ("Cocoa",         "#5B3420"),
    ("Beetroot",      "#B5266E"),
    ("Charcoal",      "#2B2B30"),
]


def srgb_to_lin(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_lin(h):
    h = h.lstrip("#")
    return srgb_to_lin([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def link(obj, coll=None):
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj


def new_collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def principled(name, color=(0.8, 0.8, 0.8), rough=0.5, metallic=0.0, **extra):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    for k, v in extra.items():
        b.inputs[k].default_value = v
    return m


def emission(name, color=(1, 1, 1), strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def box(name, lo, hi, mat=None, bevel=0.0, segs=3, coll=None, parent=None):
    """Axis aligned box from corner lo to corner hi, optional rounded edges."""
    lo, hi = Vector(lo), Vector(hi)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    size = hi - lo
    for v in bm.verts:
        v.co = Vector((v.co.x * size.x, v.co.y * size.y, v.co.z * size.z))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (lo + hi) / 2
    if bevel > 0:
        md = ob.modifiers.new("Bevel", "BEVEL")
        md.width = bevel
        md.segments = segs
        md.limit_method = "ANGLE"
        for p in me.polygons:
            p.use_smooth = True
    if mat:
        me.materials.append(mat)
    link(ob, coll)
    if parent:
        ob.parent = parent
    return ob


def cylinder(name, r, z0, z1, mat=None, segs=48, coll=None, parent=None, loc=(0, 0, 0), bevel=0.0):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r, depth=z1 - z0)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = len(p.vertices) == 4
    ob = bpy.data.objects.new(name, me)
    ob.location = (loc[0], loc[1], loc[2] + (z0 + z1) / 2)
    if bevel > 0:
        md = ob.modifiers.new("Bevel", "BEVEL")
        md.width = bevel
        md.segments = 3
        md.limit_method = "ANGLE"
    if mat:
        me.materials.append(mat)
    link(ob, coll)
    if parent:
        ob.parent = parent
    return ob


def rounded_rect(hx, hy, rc, n=14):
    """CCW outline of a rounded rectangle, list of (x, y)."""
    pts = []
    for cx, cy, a0 in ((hx - rc, hy - rc, 0), (-hx + rc, hy - rc, 90),
                       (-hx + rc, -hy + rc, 180), (hx - rc, -hy + rc, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + rc * math.cos(a), cy + rc * math.sin(a)))
    return pts


def keys(target, path, values, frames, index=-1):
    """Insert one keyframe per frame (linear)."""
    last = None
    n = len(values)
    for i, (f, v) in enumerate(zip(frames, values)):
        # skip the middle of constant runs to keep the curves light
        if last is not None and v == last and i + 1 < n and values[i + 1] == v:
            continue
        if index >= 0:
            cur = getattr(target, path)
            cur[index] = v
        else:
            setattr(target, path, v)
        target.keyframe_insert(path, index=index, frame=f)
        last = v


# ---------------------------------------------------------------------------
# scene pieces
# ---------------------------------------------------------------------------
def build_grains(cache_dir, meta, z, tray, n_ing, grain_scale):
    N, d = meta["n_grains"], meta["grain"]
    pc2 = os.path.join(cache_dir, "particles.pc2")
    first = np.fromfile(pc2, dtype="<f4", count=N * 3, offset=32)

    me = bpy.data.meshes.new("Grains")
    me.vertices.add(N)
    me.vertices.foreach_set("co", first)

    rnd = z["rand"]
    cols = np.array([hex_lin(PALETTE[i % len(PALETTE)][1]) for i in range(n_ing)])
    col = cols[z["ingredient"]]
    # natural variation: brightness +-12 %, a touch of hue drift
    col = col * (0.88 + 0.24 * rnd[:, 3:4]) * (1 + 0.06 * (rnd[:, :3] - 0.5))
    rgba = np.concatenate([np.clip(col, 0, 1), np.ones((N, 1))], 1).astype(np.float32)
    a = me.attributes.new("col", "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", rgba.ravel())
    a = me.attributes.new("rnd", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", rnd[:, :3].astype(np.float32).ravel())
    me.update()

    ob = link(bpy.data.objects.new("Grains", me))
    ob.parent = tray
    ob.location = (0, 0, TRAY_BASE)

    # 1) point cache  2) geometry nodes: a small cube on every point
    mc = ob.modifiers.new("Cache", "MESH_CACHE")
    mc.cache_format = "PC2"
    mc.filepath = pc2
    mc.interpolation = "LINEAR"
    mc.time_mode = "FRAME"
    mc.play_mode = "CUSTOM"
    mc.frame_start = 0
    mc.frame_scale = 1.0

    mat = bpy.data.materials.new("Powder")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = "col"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Specular IOR Level"].default_value = 0.3

    gn = bpy.data.node_groups.new("GrainInstancer", "GeometryNodeTree")
    gn.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    gn.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N_ = gn.nodes
    gin, gout = N_.new("NodeGroupInput"), N_.new("NodeGroupOutput")
    cube = N_.new("GeometryNodeMeshCube")
    s = d * grain_scale
    cube.inputs["Size"].default_value = (s, s, s)
    setm = N_.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = mat
    iop = N_.new("GeometryNodeInstanceOnPoints")
    na = N_.new("GeometryNodeInputNamedAttribute")
    na.data_type = "FLOAT_VECTOR"
    na.inputs["Name"].default_value = "rnd"
    pos = N_.new("GeometryNodeInputPosition")
    # rotation = random start + tumble as the grain travels
    m1 = N_.new("ShaderNodeVectorMath"); m1.operation = "SCALE"
    m1.inputs["Scale"].default_value = 2 * math.pi
    m2 = N_.new("ShaderNodeVectorMath"); m2.operation = "SCALE"
    m2.inputs["Scale"].default_value = 1.2 / d
    add = N_.new("ShaderNodeVectorMath"); add.operation = "ADD"
    e2r = N_.new("FunctionNodeEulerToRotation")
    rv = N_.new("FunctionNodeRandomValue"); rv.data_type = "FLOAT"
    rv.inputs["Min"].default_value = 0.82
    rv.inputs["Max"].default_value = 1.08
    L = gn.links.new
    L(na.outputs["Attribute"], m1.inputs[0])
    L(pos.outputs["Position"], m2.inputs[0])
    L(m1.outputs["Vector"], add.inputs[0])
    L(m2.outputs["Vector"], add.inputs[1])
    L(add.outputs["Vector"], e2r.inputs[0])
    L(cube.outputs["Mesh"], setm.inputs["Geometry"])
    L(gin.outputs[0], iop.inputs["Points"])
    L(setm.outputs["Geometry"], iop.inputs["Instance"])
    L(e2r.outputs[0], iop.inputs["Rotation"])
    L(rv.outputs["Value"], iop.inputs["Scale"])
    # realised into one mesh: a single BVH traces several times faster in
    # Cycles than tens of thousands of overlapping instances
    real = N_.new("GeometryNodeRealizeInstances")
    L(iop.outputs["Instances"], real.inputs["Geometry"])
    L(real.outputs["Geometry"], gout.inputs[0])
    md = ob.modifiers.new("Grains", "NODES")
    md.node_group = gn
    return ob, mc


def build_tray(geo, mat_glass):
    hx, hy, rc, H = geo["tray_lx"] / 2, geo["tray_ly"] / 2, geo["tray_rc"], geo["tray_h"]
    t = TRAY_WALL
    inner = rounded_rect(hx, hy, rc)
    outer = rounded_rect(hx + t, hy + t, rc + t)
    n = len(inner)
    bm = bmesh.new()
    vi0 = [bm.verts.new((x, y, TRAY_BASE)) for x, y in inner]
    vi1 = [bm.verts.new((x, y, TRAY_BASE + H)) for x, y in inner]
    vo0 = [bm.verts.new((x, y, 0)) for x, y in outer]
    vo1 = [bm.verts.new((x, y, TRAY_BASE + H)) for x, y in outer]
    bm.faces.new(vi0)                       # inside floor
    bm.faces.new(list(reversed(vo0)))       # underside
    for k in range(n):
        j = (k + 1) % n
        bm.faces.new((vi0[j], vi0[k], vi1[k], vi1[j]))     # inner wall
        bm.faces.new((vo0[k], vo0[j], vo1[j], vo1[k]))     # outer wall
        bm.faces.new((vi1[j], vi1[k], vo1[k], vo1[j]))     # rim
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Tray")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat_glass)
    ob = link(bpy.data.objects.new("TrayGlass", me))
    md = ob.modifiers.new("Bevel", "BEVEL")
    md.width = 0.7
    md.segments = 3
    md.limit_method = "ANGLE"
    md.angle_limit = math.radians(50)
    return ob


def paddle_outline(geo):
    """Outer contour + window holes of the flat gate paddle, in (r, z)."""
    R, H = geo["paddle_r"], geo["paddle_h"]
    bar, rib, spine = geo["paddle_bar"], geo["paddle_rib"], geo["paddle_spine"]
    rc = 7.0
    rci = rc - bar

    def arc(cx, cz, r, a0, a1, n=8):
        return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                 cz + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]

    outer = ([(-R, H)] + arc(-R + rc, rc, rc, 180, 270) + arc(R - rc, rc, rc, 270, 360) + [(R, H)])
    holes = []
    rib_c = 0.5 * (spine + (R - bar))
    for sgn in (1, -1):
        a0, a1 = spine, rib_c - rib / 2
        holes.append([(sgn * a0, bar), (sgn * a1, bar), (sgn * a1, H - bar), (sgn * a0, H - bar)])
        b0, b1 = rib_c + rib / 2, R - bar
        h = [(b0, bar)] + arc(b1 - rci, bar + rci, rci, 270, 360, 5) + [(b1, H - bar), (b0, H - bar)]
        holes.append([(sgn * x, zz) for x, zz in h])
    return outer, holes


def build_paddle(name, geo, mat_steel, parent, y):
    """Flat gate paddle + shaft, spinning about its local Z."""
    H = geo["paddle_h"]
    piv = link(bpy.data.objects.new(name, None))
    piv.empty_display_size = 10
    piv.parent = parent
    piv.location = (0, y, 0)

    cu = bpy.data.curves.new(name + "Blade", "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = 0.55
    cu.bevel_depth = 0.25
    cu.bevel_resolution = 2
    outer, holes = paddle_outline(geo)
    for pts in [outer] + holes:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, (x, zz) in zip(sp.points, pts):
            p.co = (x, zz, 0, 1)
        sp.use_cyclic_u = True
    cu.materials.append(mat_steel)
    blade = link(bpy.data.objects.new(name + "Blade", cu))
    blade.parent = piv
    blade.rotation_euler = (math.radians(90), 0, 0)      # curve XY -> world XZ

    cylinder(name + "Shaft", 3.0, H - 1, H + SHAFT_LEN, mat_steel, parent=piv)
    cylinder(name + "Hub", 5.2, H - 0.5, H + 5.5, mat_steel, parent=piv, bevel=0.8)
    cylinder(name + "Collar", 6.5, H + SHAFT_LEN - 7, H + SHAFT_LEN, mat_steel, parent=piv, bevel=1.0)
    return piv


def build_hud(meta, z, frames, cam, font):
    """Camera-locked overlay: pass counter, live % mixed and a progress bar."""
    coll = new_collection("HUD")
    white = emission("HUD_Ink", tuple(hex_lin("#23272E")), 1.0)
    dim = emission("HUD_Dim", tuple(hex_lin("#C9C4BB")), 1.0)
    accent = emission("HUD_Accent", tuple(hex_lin("#E8532B")), 1.0)

    D = cam.data.dof.focus_distance          # sits in the focal plane: always sharp
    sensor_w = cam.data.sensor_width
    half_w = D * sensor_w / (2 * cam.data.lens)
    r = bpy.context.scene.render
    half_h = half_w * r.resolution_y / r.resolution_x

    root = link(bpy.data.objects.new("HUD", None), coll)
    root.parent = cam
    root.location = (0, 0, -D)

    def no_rays(ob):
        ob.visible_shadow = False
        ob.visible_diffuse = False
        ob.visible_glossy = False
        ob.visible_transmission = False

    def text(name, body, size, x, y, mat, align="LEFT"):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size
        cu.align_x = align
        if font:
            cu.font = font
        cu.materials.append(mat)
        ob = link(bpy.data.objects.new(name, cu), coll)
        ob.parent = root
        ob.location = (x, y, 0)
        no_rays(ob)
        return ob

    def plane(name, x0, x1, y0, y1, mat, zoff=0.0):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0, y0, 0), (x1 - x0, y0, 0), (x1 - x0, y1, 0), (0, y1, 0)], [], [(0, 1, 2, 3)])
        me.materials.append(mat)
        ob = link(bpy.data.objects.new(name, me), coll)
        ob.parent = root
        ob.location = (x0, 0, zoff)
        no_rays(ob)
        return ob

    def show_when(ob, mask):
        """Animate render/viewport visibility from a boolean per-frame mask."""
        prev = None
        for f, m in zip(frames, mask):
            m = bool(m)
            if m != prev:
                ob.hide_render = not m
                ob.hide_viewport = not m
                ob.keyframe_insert("hide_render", frame=f)
                ob.keyframe_insert("hide_viewport", frame=f)
                prev = m

    u = half_h / 100.0                       # HUD unit = 1 % of half frame height
    mx = half_w - 9 * u
    top = half_h - 20 * u
    passes = meta["passes"]
    pass_no = z["pass_no"]
    text("PassLabel", "PASS", 7 * u, -mx, top + 3 * u, white)
    for p in range(passes + 1):
        ob = text(f"Pass{p:02d}", f"{p}", 22 * u, -mx, top - 17 * u, white)
        show_when(ob, pass_no == p)
    text("PassOf", f"of {passes}", 7 * u, -mx, top - 27 * u, white)

    pct = np.clip(np.round(z["mix"] * 100).astype(int), 0, 100)
    text("MixLabel", "MIXED", 7 * u, mx, top + 3 * u, white, "RIGHT")
    for k in np.unique(pct):
        ob = text(f"Pct{k:03d}", f"{k}%", 22 * u, mx, top - 17 * u, white, "RIGHT")
        show_when(ob, pct == k)
    bar_w = 62 * u
    plane("BarTrack", mx - bar_w, mx, top - 26 * u, top - 23 * u, dim, -0.01 * D / 100)
    fill = plane("BarFill", mx - bar_w, mx, top - 26 * u, top - 23 * u, accent)
    keys(fill, "scale", [max(float(v), 1e-3) for v in z["mix"]], frames, index=0)
    return coll


# ---------------------------------------------------------------------------
TRAY_BASE = 2.2      # glass base thickness
TRAY_WALL = 2.6
SHAFT_LEN = 64.0
FLOOR_Z = -150.0


def build(args):
    cache_dir = os.path.abspath(args.cache)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    z = np.load(os.path.join(cache_dir, "meta.npz"))
    geo = meta["geo"]
    n_frames = meta["n_frames"]
    frames = list(range(1, n_frames + 1))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = meta["fps"]
    scene.frame_start, scene.frame_end = 1, n_frames
    scene.unit_settings.scale_length = 0.001
    scene.unit_settings.length_unit = "MILLIMETERS"
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass
    W, Hres = (int(v) for v in args.res.lower().split("x"))
    scene.render.resolution_x, scene.render.resolution_y = W, Hres
    scene.render.resolution_percentage = 100

    # ---- materials -------------------------------------------------------
    m_glass = bpy.data.materials.new("Glass")
    m_glass.use_nodes = True
    nt = m_glass.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.93, 0.98, 0.97, 1)
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["IOR"].default_value = 1.47
    b.inputs["Transmission Weight"].default_value = 1.0
    # let light through to the powder: shadow rays see clear glass
    out = nt.nodes["Material Output"]
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (0.9, 0.96, 0.95, 1)
    lp = nt.nodes.new("ShaderNodeLightPath")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
    nt.links.new(b.outputs["BSDF"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    m_steel = principled("Steel", (0.82, 0.83, 0.85), rough=0.16, metallic=1.0)
    m_alu = principled("Aluminium", (0.74, 0.76, 0.78), rough=0.34, metallic=1.0)
    m_body = principled("MixerEnamel", tuple(hex_lin("#C8E6DC")), rough=0.18,
                        **{"Coat Weight": 0.6, "Coat Roughness": 0.05})
    m_dark = principled("DarkPaint", (0.035, 0.038, 0.045), rough=0.45)
    m_rubber = principled("Knob", (0.02, 0.02, 0.022), rough=0.35)
    m_floor = principled("Backdrop", tuple(hex_lin("#E9E4DC")), rough=0.7)

    # belt: dark rubber with fine transverse ribs that scroll with the belt
    m_belt = bpy.data.materials.new("Belt")
    m_belt.use_nodes = True
    nt = m_belt.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.62
    scroll = link(bpy.data.objects.new("BeltScroll", None))
    scroll.empty_display_size = 20
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = scroll
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])

    def math_node(op, a=None, bval=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        if a is not None:
            nt.links.new(a, n.inputs[0])
        if bval is not None:
            n.inputs[1].default_value = bval
        return n

    period = 12.0
    fr = math_node("FRACT", math_node("DIVIDE", sep.outputs["X"], period).outputs[0])
    tri = math_node("ABSOLUTE", math_node("SUBTRACT", fr.outputs[0], 0.5).outputs[0])   # 0..0.5
    groove = math_node("LESS_THAN", tri.outputs[0], 0.07)                               # thin rib
    # chevron tracking mark every 5 ribs, in the centre lane
    ramp = nt.nodes.new("ShaderNodeMixRGB")
    ramp.inputs["Color1"].default_value = (0.022, 0.024, 0.028, 1)
    ramp.inputs["Color2"].default_value = (0.05, 0.053, 0.06, 1)
    nt.links.new(groove.outputs[0], ramp.inputs["Fac"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.35
    noise.inputs["Detail"].default_value = 6
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    mixn = nt.nodes.new("ShaderNodeMixRGB")
    mixn.blend_type = "MULTIPLY"
    mixn.inputs["Fac"].default_value = 0.5
    nt.links.new(ramp.outputs[0], mixn.inputs["Color1"])
    nt.links.new(noise.outputs["Fac"], mixn.inputs["Color2"])
    nt.links.new(mixn.outputs[0], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 1.0
    nt.links.new(groove.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])

    # ---- conveyor --------------------------------------------------------
    belt_w = 112.0
    Lb = 1600.0
    box("Belt", (-Lb, -belt_w / 2, -4), (Lb, belt_w / 2, 0), m_belt)
    for sgn in (-1, 1):
        y0, y1 = sorted((sgn * (belt_w / 2 + 1), sgn * (belt_w / 2 + 13)))
        box(f"Rail{sgn:+d}", (-Lb, y0, -34), (Lb, y1, 7), m_alu, bevel=1.5)
        # slot line in the extrusion
        ys = sgn * (belt_w / 2 + 13.2)
        box(f"RailSlot{sgn:+d}", (-Lb, min(ys, ys + sgn * 0.6), -18), (Lb, max(ys, ys + sgn * 0.6), -12), m_dark)
    box("ConveyorBody", (-Lb, -belt_w / 2 - 8, -78), (Lb, belt_w / 2 + 8, -30), m_dark, bevel=2)
    for x in (-420, -140, 140, 420):
        for sgn in (-1, 1):
            y = sgn * (belt_w / 2 + 2)
            box(f"Leg{x}{sgn:+d}", (x - 9, y - 9, FLOOR_Z), (x + 9, y + 9, -70), m_alu, bevel=1.5)
            box(f"Foot{x}{sgn:+d}", (x - 14, y - 14, FLOOR_Z), (x + 14, y + 14, FLOOR_Z + 5), m_dark, bevel=1.5)
    box("Floor", (-6000, -6000, FLOOR_Z - 10), (6000, 6000, FLOOR_Z), m_floor)

    # ---- tray rig ----------------------------------------------------------
    tray = link(bpy.data.objects.new("Tray", None))
    tray.empty_display_size = 30
    glass = build_tray(geo, m_glass)
    glass.parent = tray
    grains, mc = build_grains(cache_dir, meta, z, tray, meta["ingredients"], args.grain_scale)

    # ---- mixer -------------------------------------------------------------
    col_y0 = belt_w / 2 + 26
    box("MixerBase", (-70, col_y0 - 14, FLOOR_Z), (70, col_y0 + 84, FLOOR_Z + 16), m_body, bevel=7, segs=5)
    box("MixerColumn", (-30, col_y0, FLOOR_Z + 6), (30, col_y0 + 56, 236), m_body, bevel=13, segs=6)
    box("ColumnTrack", (-9, col_y0 - 1.5, 30), (9, col_y0 + 3, 222), m_steel, bevel=1)

    head = link(bpy.data.objects.new("MixerHead", None))        # origin = paddle bottom plane
    head.empty_display_size = 30
    hb = geo["paddle_h"] + SHAFT_LEN                              # underside of the head body
    nose = -geo["paddle_sep"] / 2 - 19
    body = box("HeadBody", (-40, nose, hb), (40, col_y0 + 10, hb + 48), m_body, bevel=19, segs=8, parent=head)
    box("HeadCarriage", (-36, col_y0 - 12, hb - 6), (36, col_y0 + 4, hb + 58), m_body, bevel=6, segs=4, parent=head)
    box("HeadBand", (-40.6, 30, hb + 3), (40.6, 37, hb + 45), m_steel, bevel=1.2, parent=head)
    knob = cylinder("HeadKnob", 7.5, 0, 9, m_rubber, parent=head, bevel=1.5)
    knob.rotation_euler = (math.radians(90), 0, 0)
    knob.location = (0, nose - 3, hb + 24)
    cylinder("NoseRing", 11, 0, 2.2, m_steel, parent=head, bevel=0.6).rotation_euler = (math.radians(90), 0, 0)
    bpy.data.objects["NoseRing"].location = (0, nose - 0.6, hb + 24)
    ya, yb = -geo["paddle_sep"] / 2, geo["paddle_sep"] / 2
    for y in (ya, yb):
        cylinder(f"Chuck{y:+.0f}", 9.5, hb - 6, hb + 2, m_steel, parent=head, loc=(0, y, 0), bevel=1.2)
    pad_a = build_paddle("PaddleA", geo, m_steel, head, ya)
    pad_b = build_paddle("PaddleB", geo, m_steel, head, yb)

    # ---- animation channels --------------------------------------------------
    keys(tray, "location", [float(v) for v in z["tray_x"]], frames, index=0)
    keys(scroll, "location", [float(v) for v in z["belt_s"]], frames, index=0)
    head_z = TRAY_BASE + geo["paddle_clear"] + geo["lift"] * (1 - z["plunge"])
    keys(head, "location", [float(v) for v in head_z], frames, index=2)
    keys(pad_a, "rotation_euler", [float(v) for v in z["ang_a"]], frames, index=2)
    keys(pad_b, "rotation_euler", [float(v) for v in z["ang_b"]], frames, index=2)
    keys(mc, "eval_frame", [float(v) for v in z["cache_idx"]], frames)

    # wrap-around copy: the dish jumps from +x_wrap back to -x_wrap once per
    # cycle.  A second copy one belt-loop ahead carries on out of frame while
    # the original comes in on the other side, so nothing pops at the edge.
    wrap = link(bpy.data.objects.new("TrayWrap", None))
    wrap.parent = tray
    wrap.location = (2 * geo["x_wrap"], 0, 0)
    for src in (glass, grains):
        g = src.copy()                      # shares mesh, modifiers and animation
        g.name = src.name + "Wrap"
        link(g)
        g.parent = wrap

    # ---- camera ------------------------------------------------------------
    cam_d = bpy.data.cameras.new("Camera")
    cam_d.lens = 50
    cam_d.sensor_width = 36
    cam_d.clip_start, cam_d.clip_end = 10, 50000
    cam = link(bpy.data.objects.new("Camera", cam_d))
    target = Vector((0, 6, 56))
    elev, azim, dist = math.radians(args.elev), math.radians(args.azim), args.dist
    cam.location = target + Vector((dist * math.cos(elev) * math.sin(azim),
                                    -dist * math.cos(elev) * math.cos(azim),
                                    dist * math.sin(elev)))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    cam_d.dof.use_dof = not args.no_dof
    cam_d.dof.focus_distance = dist - 8
    # scene units are mm but Cycles reads the lens in metres: f/0.011 here gives
    # the blur of a ~4 mm aperture radius at this scale (gentle macro look)
    cam_d.dof.aperture_fstop = 0.05 / (2 * args.aperture)

    # ---- light ---------------------------------------------------------------
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world
    wn = world.node_tree
    bg = wn.nodes["Background"]
    bg.inputs["Color"].default_value = (0.75, 0.8, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.35

    def softbox(name, loc, size, strength, color=(1, 1, 1)):
        me = bpy.data.meshes.new(name)
        sx, sy = size[0] / 2, size[1] / 2
        me.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
        me.materials.append(emission(name, color, strength))
        ob = link(bpy.data.objects.new(name, me))
        ob.location = loc
        ob.rotation_euler = (Vector((0, 0, 30)) - Vector(loc)).to_track_quat("Z", "Y").to_euler()
        ob.visible_camera = False
        return ob

    softbox("KeyLight", (-380, -420, 620), (520, 380), 14.0, (1.0, 0.96, 0.9))
    softbox("FillLight", (520, -260, 380), (420, 420), 4.0, (0.86, 0.92, 1.0))
    softbox("RimLight", (140, 620, 420), (600, 240), 9.0, (1.0, 1.0, 1.0))

    # ---- overlay ---------------------------------------------------------------
    if not args.no_hud:
        font = None
        for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                   "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "C:/Windows/Fonts/arialbd.ttf"):
            if os.path.exists(fp):
                font = bpy.data.fonts.load(fp)
                break
        build_hud(meta, z, frames, cam, font)

    # ---- render settings ---------------------------------------------------------
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    cy.max_bounces = 8
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 4
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 8
    if args.fast:                       # preview quality: fewer light bounces
        cy.max_bounces = 4
        cy.diffuse_bounces = 1
        cy.glossy_bounces = 2
        cy.transmission_bounces = 4
        cy.adaptive_threshold = 0.05
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = args.motion_blur
    scene.render.motion_blur_shutter = 0.35
    if args.gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(dv.type == kind for dv in prefs.devices):
                    for dv in prefs.devices:
                        dv.use = True
                    cy.device = "GPU"
                    print("GPU rendering with", kind)
                    break
            except Exception:
                continue
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception as e:
        print("colour management:", e)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    return scene, meta


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--save", help="write a .blend here")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--still", type=int, nargs="*", help="render these frame numbers")
    ap.add_argument("--anim", action="store_true", help="render the animation frames")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "renders"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true", help="preview quality (fewer bounces)")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-dof", action="store_true")
    ap.add_argument("--motion-blur", action="store_true")
    ap.add_argument("--elev", type=float, default=37.0, help="camera elevation (deg)")
    ap.add_argument("--azim", type=float, default=-14.0, help="camera azimuth (deg)")
    ap.add_argument("--dist", type=float, default=560.0, help="camera distance (mm)")
    ap.add_argument("--aperture", type=float, default=2.5, help="DOF aperture radius (mm)")
    ap.add_argument("--grain-scale", type=float, default=0.84,
                    help="rendered cube edge as a fraction of the simulated grain size")
    args = ap.parse_args(argv)

    scene, meta = build(args)
    if args.threads:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    if args.save:
        path = os.path.abspath(args.save)
        # keep the cache path relative so the project folder can be moved
        bpy.ops.wm.save_as_mainfile(filepath=path)
        try:
            mc = bpy.data.objects["Grains"].modifiers["Cache"]
            mc.filepath = bpy.path.relpath(mc.filepath)
            bpy.ops.wm.save_mainfile()
        except Exception as e:
            print("relative cache path:", e)
        print("saved", path)
    os.makedirs(args.out, exist_ok=True)
    if args.still:
        for f in args.still:
            scene.frame_set(f)
            scene.render.filepath = os.path.join(args.out, f"still_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", scene.render.filepath)
    if args.anim:
        if args.start:
            scene.frame_start = args.start
        if args.end:
            scene.frame_end = args.end
        scene.frame_step = args.step
        scene.render.filepath = os.path.join(args.out, "frames", "f_")
        scene.render.use_overwrite = False       # resumable; several processes can share
        scene.render.use_placeholder = True
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
