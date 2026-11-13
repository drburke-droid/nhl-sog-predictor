"""Build the studio and the to-scale board in Blender (Cycles). Imported by the shot scripts.

Board coordinates come straight from board.py (the same geometry the physics uses). Locally the board is
built with X across, Z up (top pin row at z = 0, rows going down), and -Y toward the camera; a parent empty
then leans it back by the board tilt and lifts it so the slot floor sits 0.85 m above the stage.
"""
import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import board as B  # noqa: E402

W, H = B.SLOT_PITCH, B.ROW_SPACING
TILT = math.radians(B.BOARD_TILT_DEG)
R_CHIP = B.CHIP_DIAMETER / 2
T_CHIP = B.CHIP_THICKNESS
PIN_LEN = 0.044
COVER_Y = -0.046
FLOOR_Y = B.last_row_y() + B.FLOOR_BELOW_LAST_ROW       # board-down coordinate of the slot floor
DIV_Y = B.last_row_y() + B.DIVIDER_TOP_BELOW_LAST_ROW
SLOT_FLOOR_Z = 0.85                                      # stage height of the slot floor
RIG_Z = SLOT_FLOOR_Z + FLOOR_Y * math.cos(TILT)
FONT_DIR = "/usr/share/fonts/opentype/inter"
SLOT_LABELS = ["$100", "$500", "$1,000", "$0", "$10,000", "$0", "$1,000", "$500", "$100"]


# --------------------------------------------------------------------------------------------- helpers
def lx(x_board):
    """Board x (slot centres at 1..9 pitches) -> local X."""
    return x_board - 5 * W


def board_to_local(x, y, depth=0.0):
    """Board (x across, y down) plus depth toward the camera -> local coordinates."""
    return Vector((lx(x), -depth, -y))


def local_to_world(v):
    return rig_matrix() @ v


def rig_matrix():
    return Matrix.Translation((0, 0, RIG_Z)) @ Matrix.Rotation(-TILT, 4, "X")


def chip_world(x, y, ang):
    """World location and rotation (Euler XYZ) of a chip lying on the board at board coords (x, y)."""
    loc = local_to_world(board_to_local(x, y, T_CHIP / 2 + 0.0015))
    rot = (Matrix.Rotation(-TILT, 4, "X") @ Matrix.Rotation(math.pi / 2, 4, "X") @
           Matrix.Rotation(-ang, 4, "Z")).to_euler("XYZ")
    return loc, rot


def mat(name, color=(0.8, 0.8, 0.8), rough=0.5, metal=0.0, emit=None, strength=1.0, alpha=None, coat=0.0,
        aniso=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    if aniso:
        p.inputs["Anisotropic"].default_value = aniso
    if emit is not None:
        p.inputs["Emission Color"].default_value = (*emit, 1)
        p.inputs["Emission Strength"].default_value = strength
    if alpha is not None:
        p.inputs["Alpha"].default_value = alpha
    return m


def add_noise_bump(m, scale=400.0, strength=0.04):
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    bmp = nt.nodes.new("ShaderNodeBump")
    bmp.inputs["Strength"].default_value = strength
    nt.links.new(n.outputs["Fac"], bmp.inputs["Height"])
    nt.links.new(bmp.outputs["Normal"], p.inputs["Normal"])


def box(name, size, loc, material, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    o.data.materials.append(material)
    if parent:
        o.parent = parent
    return o


def mesh_object(name, verts, faces, material, parent=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(material)
    if parent:
        o.parent = parent
    for f in o.data.polygons:
        f.use_smooth = False
    return o


def text_obj(name, s, loc, size, material, parent=None, font="Inter-Bold.otf", align="CENTER", extrude=0.0,
             rot=(0, 0, 0)):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = s
    cu.font = bpy.data.fonts.load(os.path.join(FONT_DIR, font), check_existing=True)
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = extrude
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    o.data.materials.append(material)
    if parent:
        o.parent = parent
    return o


# --------------------------------------------------------------------------------------------- materials
def materials():
    M = {}
    M["backboard"] = mat("backboard", (0.62, 0.53, 0.40), rough=0.42, coat=0.15)
    add_noise_bump(M["backboard"], 900.0, 0.015)
    M["alu"] = mat("brushed_aluminium", (0.80, 0.81, 0.83), rough=0.28, metal=1.0, aniso=0.6)
    M["steel"] = mat("pin_steel", (0.55, 0.56, 0.58), rough=0.22, metal=1.0)
    M["led_off"] = mat("led_off", (0.012, 0.013, 0.016), rough=0.25, coat=0.8)
    M["acrylic_white"] = mat("wall_acrylic", (0.93, 0.93, 0.91), rough=0.18, coat=0.4)
    M["divider"] = mat("divider_steel", (0.55, 0.56, 0.58), rough=0.25, metal=1.0)
    M["fascia"] = mat("fascia", (0.025, 0.028, 0.035), rough=0.3, coat=0.5)
    M["pedestal"] = mat("pedestal", (0.035, 0.04, 0.05), rough=0.35, coat=0.6)
    M["floor"] = mat("stage_floor", (0.018, 0.02, 0.026), rough=0.16, coat=0.3)
    add_noise_bump(M["floor"], 60.0, 0.01)
    M["cyc"] = mat("cyclorama", (0.03, 0.035, 0.05), rough=0.9)
    M["steps"] = mat("steps", (0.025, 0.05, 0.07), rough=0.45, coat=0.3)
    M["nosing"] = mat("nosing", (0.85, 0.85, 0.85), rough=0.3)
    M["chrome"] = mat("chrome", (0.9, 0.9, 0.92), rough=0.06, metal=1.0)
    M["led_white"] = mat("led_white", (1, 1, 1), emit=(1.0, 0.93, 0.82), strength=6.0)
    M["led_gold"] = mat("led_gold", (1, 0.7, 0.2), emit=(1.0, 0.62, 0.12), strength=9.0)
    M["led_dim"] = mat("led_dim", (0.2, 0.2, 0.2), emit=(0.45, 0.42, 0.4), strength=1.2)
    M["bulb"] = mat("marquee_bulb", (1, 0.9, 0.7), emit=(1.0, 0.82, 0.55), strength=12.0)
    M["strip"] = mat("trim_strip", (0.3, 0.6, 1.0), emit=(0.25, 0.55, 1.0), strength=4.0)
    slot_cols = [(0.03, 0.06, 0.13), (0.03, 0.08, 0.19), (0.04, 0.12, 0.28), (0.012, 0.012, 0.014),
                 (0.30, 0.17, 0.01), (0.012, 0.012, 0.014), (0.04, 0.12, 0.28), (0.03, 0.08, 0.19),
                 (0.03, 0.06, 0.13)]
    M["slots"] = [mat(f"slot_{k}", c, rough=0.4) for k, c in enumerate(slot_cols)]
    # plexiglass cover: frosted bars where the mask is white, holes elsewhere
    m = bpy.data.materials.new("cover_plexi")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(HERE, "assets", "cover_mask.png"), check_existing=True)
    tex.image.colorspace_settings.name = "Non-Color"
    tex.interpolation = "Linear"
    bars = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bars.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1)
    bars.inputs["Roughness"].default_value = 0.35
    bars.inputs["Transmission Weight"].default_value = 0.0
    bars.inputs["Alpha"].default_value = 0.34
    bars.inputs["Coat Weight"].default_value = 0.6
    holes = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(holes.outputs[0], mix.inputs[1])
    nt.links.new(bars.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    M["cover"] = m
    # chip: printed face plus a polished metal rim
    m = bpy.data.materials.new("chip_face")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(HERE, "assets", "chip_face.png"), check_existing=True)
    nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.22
    p.inputs["Coat Weight"].default_value = 1.0
    p.inputs["Coat Roughness"].default_value = 0.05
    M["chip_face"] = m
    M["chip_rim"] = mat("chip_rim", (0.82, 0.83, 0.85), rough=0.18, metal=1.0)
    M["chip_blue_face"] = tinted_chip("chip_face_blue", (0.08, 0.30, 0.85))
    M["chip_gold_face"] = tinted_chip("chip_face_gold", (0.95, 0.62, 0.10))
    return M


def tinted_chip(name, rgb):
    """A chip face of another colour: the printed design multiplied onto a new base colour."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(HERE, "assets", "chip_face.png"), check_existing=True)
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*rgb, 1)
    mix.inputs[7].default_value = (0.93, 0.93, 0.91, 1)
    nt.links.new(sep.outputs["Green"], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.22
    p.inputs["Coat Weight"].default_value = 1.0
    return m


# --------------------------------------------------------------------------------------------- chip mesh
_chip_mesh = {}


def chip_mesh(face_mat, rim_mat):
    key = face_mat.name
    if key in _chip_mesh:
        return _chip_mesh[key]
    bm = bmesh.new()
    segs = 64
    r, t = R_CHIP, T_CHIP
    bevel = 0.0025
    # body: a closed cylinder (rim material); faces: two discs with UVs from position
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=t)
    for f in bm.faces:
        f.material_index = 1
    uv = bm.loops.layers.uv.new("UVMap")
    for z in (t / 2 + 0.0004, -t / 2 - 0.0004):
        verts = [bm.verts.new((math.cos(2 * math.pi * i / segs) * (r - bevel),
                               math.sin(2 * math.pi * i / segs) * (r - bevel), z)) for i in range(segs)]
        f = bm.faces.new(verts if z > 0 else verts[::-1])
        f.material_index = 0
        for loop in f.loops:
            co = loop.vert.co
            loop[uv].uv = (co.x / (2 * r) + 0.5, (co.y if z > 0 else -co.y) / (2 * r) + 0.5)
    me = bpy.data.meshes.new(f"chip_{key}")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(face_mat)
    me.materials.append(rim_mat)
    for p in me.polygons:
        p.use_smooth = p.material_index == 1
    _chip_mesh[key] = me
    return me


def new_chip(name, M, face="chip_face"):
    o = bpy.data.objects.new(name, chip_mesh(M[face], M["chip_rim"]))
    bpy.context.collection.objects.link(o)
    o.rotation_mode = "XYZ"
    return o


# --------------------------------------------------------------------------------------------- board
LED_ROWS = 20
LED_Z0, LED_Z1 = 0.035, 0.485


def led_cell(k, r):
    """World (x centre, bottom z, width, height) of LED cell r (0 = bottom) in column k (0..8)."""
    pitch = (LED_Z1 - LED_Z0) / LED_ROWS
    return lx((k + 1) * W), LED_Z0 + r * pitch, 0.62 * W, pitch * 0.78


def build_board(M):
    rig = bpy.data.objects.new("BoardRig", None)
    bpy.context.collection.objects.link(rig)
    rig.matrix_world = rig_matrix()
    top_z = 0.62 * W
    bot_z = -FLOOR_Y
    half = 4.5 * W
    side = 0.16
    # backboard
    box("backboard", (2 * (half + side), 0.05, top_z - bot_z + 0.12), (0, 0.025, (top_z + bot_z) / 2),
        M["backboard"], rig)
    # frame rails (aluminium) around the field
    box("rail_top", (2 * (half + side) + 0.02, 0.07, 0.05), (0, -0.01, top_z + 0.03), M["alu"], rig, 0.004)
    for s in (-1, 1):
        box(f"rail_side{s}", (0.06, 0.07, top_z - bot_z + 0.2), (s * (half + side - 0.03 + 0.01), -0.01,
            (top_z + bot_z) / 2), M["alu"], rig, 0.004)
    # sawtooth walls: extruded prisms between the outer edge and the tooth line
    for s in (-1, 1):
        poly = B.wall_polyline(s)
        outer = (half + side - 0.04) * s
        verts, faces = [], []
        pts = [(lx(x), -y) for (x, y) in poly]
        pts[0] = (pts[0][0], top_z)
        prof = [(outer, top_z)] + pts + [(outer, bot_z)]
        n = len(prof)
        for (x, z) in prof:
            verts.append((x, 0.0, z))
        for (x, z) in prof:
            verts.append((x, -PIN_LEN, z))
        for i in range(n):
            j = (i + 1) % n
            faces.append((i, j, j + n, i + n))
        faces.append(tuple(range(n))[::-1] if s > 0 else tuple(range(n)))
        faces.append(tuple(range(n, 2 * n)) if s > 0 else tuple(range(n, 2 * n))[::-1])
        o = mesh_object(f"wall{s}", verts, faces, M["acrylic_white"], rig)
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = 0.002
        mod.segments = 2
    # pins: one mesh, linked duplicates
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=B.PEG_DIAMETER / 2, depth=PIN_LEN)
    proto = bpy.context.active_object
    proto.name = "pin_proto"
    proto.data.materials.append(M["steel"])
    bpy.ops.object.shade_smooth()
    pins_col = bpy.data.collections.new("pins")
    bpy.context.scene.collection.children.link(pins_col)
    pts, _ = B.pegs()
    for i, (x, y) in enumerate(pts):
        o = bpy.data.objects.new(f"pin{i:03d}", proto.data)
        pins_col.objects.link(o)
        o.parent = rig
        o.location = board_to_local(x, y, PIN_LEN / 2)
        o.rotation_euler = (math.pi / 2, 0, 0)
    bpy.data.objects.remove(proto)
    # slot backs, dividers and floor tray
    for k in range(9):
        x0, x1 = lx((k + 0.5) * W), lx((k + 1.5) * W)
        box(f"slotback{k}", (x1 - x0, 0.004, FLOOR_Y - DIV_Y + 0.06), ((x0 + x1) / 2, -0.002,
            -(DIV_Y + FLOOR_Y) / 2 + 0.03), M["slots"][k], rig)
    for k in range(1, 9):
        xx = lx((k + 0.5) * W)
        box(f"divider{k}", (B.DIVIDER_THICKNESS, PIN_LEN, FLOOR_Y - DIV_Y), (xx, -PIN_LEN / 2,
            -(DIV_Y + FLOOR_Y) / 2), M["divider"], rig, 0.001)
    box("tray", (2 * half + 0.04, 0.09, 0.03), (0, -0.03, bot_z - 0.015), M["alu"], rig, 0.004)
    # plexiglass cover with triangular holes
    import make_textures as MT
    cx0, cx1 = lx(MT.X0), lx(MT.X1)
    cz0, cz1 = -MT.Y0, -MT.Y1
    verts = [(cx0, COVER_Y, cz0), (cx1, COVER_Y, cz0), (cx1, COVER_Y, cz1), (cx0, COVER_Y, cz1)]
    cov = mesh_object("cover", verts, [(0, 1, 2, 3)], M["cover"], rig)
    uv = cov.data.uv_layers.new(name="UVMap")
    for loop, (u, v) in zip(cov.data.loops, [(0, 1), (1, 1), (1, 0), (0, 0)]):
        uv.data[loop.index].uv = (u, v)
    # fascia with LED value signs under the slots, and the pedestal
    fz = bot_z - 0.03 - 0.16
    box("fascia", (2 * (half + side), 0.12, 0.30), (0, -0.04, fz), M["fascia"], rig, 0.01)
    signs = []
    for k in range(9):
        m = M["led_gold"] if k == 4 else M["led_white"]
        t = text_obj(f"sign{k}", SLOT_LABELS[k], (lx((k + 1) * W), -0.101, fz), 0.052 if k != 4 else 0.046,
                     m, rig, rot=(math.pi / 2, 0, 0))
        signs.append(t)
    rig_world = rig_matrix()
    base_top = (rig_world @ Vector((0, 0, fz - 0.15))).z
    box("pedestal", (2 * (half + side) + 0.3, 0.9, base_top), (0, 0.15, base_top / 2), M["pedestal"], None, 0.02)
    box("pedestal_strip", (2 * (half + side) + 0.32, 0.02, 0.012), (0, -0.30, 0.006), M["strip"], None)
    # LED bar-chart panel on the pedestal front: 9 columns under the slots, LED_ROWS cells each
    for k in range(9):
        for r in range(LED_ROWS):
            x, z0, w, h = led_cell(k, r)
            box(f"led_{k}_{r}", (w, 0.006, h), (x, -0.303, z0 + h / 2), M["led_off"], None, 0.0015)
    return rig, signs


# --------------------------------------------------------------------------------------------- studio
def build_studio(M):
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "stage_floor"
    fl.data.materials.append(M["floor"])
    # cyclorama: a big curved wall behind
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=7.5, depth=9.0, location=(0, -0.5, 4.5),
                                        end_fill_type="NOTHING")
    cyc = bpy.context.active_object
    cyc.name = "cyclorama"
    bm = bmesh.new()
    bm.from_mesh(cyc.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_center_median().y < -0.6], context="FACES")
    bm.to_mesh(cyc.data)
    bm.free()
    cyc.data.materials.append(M["cyc"])
    bpy.ops.object.shade_smooth()
    # marquee bulbs on two arcs behind the board (bokeh)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.035)
    bulb = bpy.context.active_object
    bulb.data.materials.append(M["bulb"])
    k = 0
    for (rad, zc, n) in ((4.2, 2.6, 34), (4.9, 2.6, 40)):
        for i in range(n):
            a = math.pi * (0.08 + 0.84 * i / (n - 1))
            o = bpy.data.objects.new(f"bulb{k}", bulb.data)
            bpy.context.collection.objects.link(o)
            o.location = (math.cos(a) * rad, 4.0, zc + math.sin(a) * rad * 0.62)
            k += 1
    bpy.data.objects.remove(bulb)
    # stairs and platform to the right of the board (a contestant climbs these to drop the chip)
    x0 = 4.5 * W + 0.45
    rise, run, nst = 0.19, 0.30, 9
    for i in range(nst):
        h = rise * (i + 1)
        box(f"step{i}", (1.1, run, h), (x0 + 0.55, -1.6 + i * run + run / 2, h / 2), M["steps"], None, 0.006)
        box(f"nose{i}", (1.1, 0.03, 0.025), (x0 + 0.55, -1.6 + i * run + 0.015, h - 0.0125), M["nosing"], None)
    plat_h = rise * nst
    box("platform", (1.3, 1.2, plat_h), (x0 + 0.6, -1.6 + nst * run + 0.6, plat_h / 2), M["steps"], None, 0.006)
    # railing: posts and a top rail
    for i in range(0, nst + 1, 3):
        h = rise * min(i + 1, nst)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.022, depth=0.95,
                                            location=(x0 + 1.07, -1.6 + i * run + 0.1, h + 0.47))
        o = bpy.context.active_object
        o.data.materials.append(M["chrome"])
    p0 = Vector((x0 + 1.07, -1.6 + 0.1, rise + 0.95))
    p1 = Vector((x0 + 1.07, -1.6 + nst * run + 0.1, plat_h + 0.95))
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.025, depth=d.length, location=(p0 + p1) / 2)
    o = bpy.context.active_object
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    o.data.materials.append(M["chrome"])


def lights():
    def area(name, loc, rot, size, power, color):
        bpy.ops.object.light_add(type="AREA", location=loc, rotation=rot)
        l = bpy.context.active_object
        l.name = name
        l.data.shape = "RECTANGLE"
        l.data.size, l.data.size_y = size
        l.data.energy = power
        l.data.color = color
        return l
    area("key", (-2.4, -3.6, 4.4), (math.radians(55), 0, math.radians(-32)), (2.4, 1.6), 420, (1.0, 0.92, 0.82))
    area("fill", (2.8, -3.4, 2.6), (math.radians(70), 0, math.radians(38)), (2.0, 2.0), 90, (0.80, 0.88, 1.0))
    area("top", (0, -0.9, 5.4), (0, 0, 0), (2.4, 1.2), 160, (1.0, 0.97, 0.92))
    area("cyc_left", (-3.5, 3.0, 0.3), (math.radians(-100), 0, math.radians(30)), (3, 1), 260, (0.95, 0.25, 0.55))
    area("cyc_right", (3.5, 3.0, 0.3), (math.radians(-100), 0, math.radians(-30)), (3, 1), 260, (0.20, 0.55, 1.0))
    w = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.008, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0


def camera(name="cam", lens=40, loc=(0, -6, 2), target=(0, 0, 2), fstop=4.0):
    cam = bpy.data.cameras.new(name)
    o = bpy.data.objects.new(name, cam)
    bpy.context.collection.objects.link(o)
    cam.lens = lens
    cam.sensor_width = 36
    cam.dof.use_dof = True
    cam.dof.aperture_fstop = fstop
    tgt = bpy.data.objects.new(name + "_target", None)
    bpy.context.collection.objects.link(tgt)
    tgt.location = target
    c = o.constraints.new("TRACK_TO")
    c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    cam.dof.focus_object = tgt
    o.location = loc
    bpy.context.scene.camera = o
    return o, tgt


def render_settings(samples=16, res=(1920, 1080), motion_blur=True, engine="CYCLES"):
    sc = bpy.context.scene
    if engine == "EEVEE":
        sc.render.engine = "BLENDER_EEVEE_NEXT"
        ee = sc.eevee
        ee.taa_render_samples = 32
        ee.use_raytracing = True
        ee.ray_tracing_options.resolution_scale = "2"
        ee.use_shadows = True
        ee.shadow_ray_count = 2
        ee.use_gtao = True
        sc.render.resolution_x, sc.render.resolution_y = res
        sc.render.resolution_percentage = 100
        sc.render.fps = 30
        sc.render.use_motion_blur = motion_blur
        sc.render.motion_blur_shutter = 0.5
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
        return
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.05
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 1
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 2
    sc.cycles.transparent_max_bounces = 6
    sc.cycles.use_fast_gi = True
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.fps = 30
    sc.render.use_motion_blur = motion_blur
    sc.render.motion_blur_shutter = 0.5
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.threads_mode = "AUTO"


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    M = materials()
    build_studio(M)
    rig, signs = build_board(M)
    lights()
    render_settings()
    return M, rig, signs


# --------------------------------------------------------------------------------------------- export
def camera_info(cam_obj):
    """Everything the compositor needs to project world points exactly as this camera does."""
    sc = bpy.context.scene
    cam = cam_obj.data
    return dict(matrix_world=[list(r) for r in cam_obj.matrix_world], lens=cam.lens, sensor_width=cam.sensor_width,
                shift_x=cam.shift_x, shift_y=cam.shift_y, res=[sc.render.resolution_x, sc.render.resolution_y],
                fstop=cam.dof.aperture_fstop, focus=list(cam.dof.focus_object.matrix_world.translation)
                if cam.dof.focus_object else None)


def world_points():
    """Named world points for overlays: pins, slot centres, drop gaps, LED cells, board corners."""
    pts = {}
    pg, rows = B.pegs()
    pts["pins"] = [list(local_to_world(board_to_local(x, y, PIN_LEN))) for (x, y) in pg]
    pts["slot_centres"] = [list(local_to_world(board_to_local((k + 1) * W, (DIV_Y + FLOOR_Y) / 2, 0.02)))
                           for k in range(9)]
    pts["drop_gaps"] = [list(local_to_world(board_to_local((k + 1) * W, 0.0, 0.02))) for k in range(9)]
    pts["wall_teeth"] = [list(local_to_world(board_to_local(x, y, PIN_LEN))) for s in (-1, 1)
                         for (x, y) in B.wall_polyline(s)[1:-1] if abs(x - 0.5 * W) > 1e-6 and abs(x - 9.5 * W) > 1e-6]
    pts["led_cells"] = [[[led_cell(k, r)[0] - led_cell(k, r)[2] / 2, -0.307, led_cell(k, r)[1]],
                         [led_cell(k, r)[0] + led_cell(k, r)[2] / 2, -0.307, led_cell(k, r)[1] + led_cell(k, r)[3]]]
                        for k in range(9) for r in range(LED_ROWS)]
    corners = {"field_tl": (0.5 * W, 0.0), "field_tr": (9.5 * W, 0.0), "field_bl": (0.5 * W, B.last_row_y()),
               "field_br": (9.5 * W, B.last_row_y()), "floor_l": (0.5 * W, FLOOR_Y), "floor_r": (9.5 * W, FLOOR_Y)}
    for k, (x, y) in corners.items():
        pts[k] = list(local_to_world(board_to_local(x, y, PIN_LEN)))
    return pts
