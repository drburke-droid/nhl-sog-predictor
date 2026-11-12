"""
Blender scene for the kitchen head-to-head: one or two round glass bowls, each
under a mixer head that carries a tool round the bowl, built from caches
written by bowl_sim.py.

    python bowl_scene.py --left cache/bowl/spoon --right cache/bowl/paddle_large --still 120
    python bowl_scene.py --left cache/bowl/whisk --anim --res 1920x1080 --samples 48
    blender -b -P bowl_scene.py -- --left cache/bowl/spoon --save spoon.blend

Scene units are millimetres, as in build_scene.py, whose conveyor, lights,
materials and render setup this reuses.
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_scene as B           # noqa: E402
from tools import KITCHENAID_COLOR, TOOL_COLOR, build_tool, fork_visual, mix_curve, spoon_dish      # noqa: E402

HC = 80.0          # chuck height above the tool's bottom edge
HEAD_R = 30.0      # radius of the cylindrical motor housing
KA_BLEND = os.path.join(HERE, "models", "kitchenaid", "kitchenaid.blend")   # CC0 model by swmo (Blend Swap 65088)
LETTERING_PNG = os.path.join(HERE, "models", "kitchenaid", "kitchen_inc_lettering.png")   # generic brand; make_lettering.py
KA_SCALE = 507.0   # mm per model unit.  True to the bowl would be 374; the mixer is drawn 1.36x
                   # too big so that its head clears the bowl and the powder stays in view.
KA_BASE, KA_SEAT, KA_HUB = -0.422, -0.388, -0.145     # model z of: underside, bowl seat, planetary hub


LETTER_ROUGH = {"brushed": 0.22, "chrome": 0.035}
LETTERING = True


def args_lettering():
    return LETTERING


def make_kitchenaid(tag, root, M, enamel, yaw):
    """Append the KitchenAid mixer body, standing on the belt with its bowl seat under the bowl."""
    with bpy.data.libraries.load(KA_BLEND, link=False) as (src, dst):
        dst.objects = [n for n in ("Mixer", "Armature") if n in src.objects]
    mixer = next(o for o in dst.objects if o.type == "MESH")
    for o in dst.objects:
        B.link(o)
    bpy.context.view_layer.update()
    mixer.data.transform(mixer.matrix_world)            # bake the rig's rest pose into the mesh
    mixer.parent = None
    mixer.matrix_world = Matrix.Identity(4)
    for md in list(mixer.modifiers):
        if md.type == "ARMATURE":
            mixer.modifiers.remove(md)
        elif md.type == "SUBSURF":
            md.levels = md.render_levels = 2
    for o in dst.objects:
        if o is not mixer:
            bpy.data.objects.remove(o)
    def lettered(key, base):
        """The trim band's lettering painted on our metal: a generic brand, "Kitchen Inc", laid
        out like the model's own texture (which names a real maker) so its UV mapping still fits."""
        m = bpy.data.materials.get(key)
        if m is None:
            if not args_lettering() or not os.path.exists(LETTERING_PNG):
                return M[base]
            img = bpy.data.images.get(os.path.basename(LETTERING_PNG)) or bpy.data.images.load(LETTERING_PNG)
            src = M[base].node_tree.nodes["Principled BSDF"]
            m = B.metal(key, tuple(src.inputs["Base Color"].default_value)[:3], LETTER_ROUGH[base],
                        aniso=src.inputs["Anisotropic"].default_value, tangent="uv" if base == "brushed" else None,
                        smudge=0.25, coords="world", scale=0.02, lettering=img)
        return m

    for slot in mixer.material_slots:                    # the file's own shaders are plain; use ours
        n = slot.material.name if slot.material else ""
        slot.material = (enamel if "Yellow" in n
                         else lettered("BandLettering", "brushed") if "BrushedChrome_Writing" in n
                         else lettered("CapLettering", "chrome") if "Chrome_Writing" in n
                         else M["brushed"] if "Brushed" in n else M["chrome"] if "Chrome" in n
                         else M["rubber"] if "Black" in n else M["white"])
    for mat in [mm for mm in bpy.data.materials if mm.name.startswith("Kitchenaid_") and not mm.users]:
        bpy.data.materials.remove(mat)                   # the model's own shaders, and with them its
    for img in [im for im in bpy.data.images if im.name.startswith("kitchenaid_writing") and not im.users]:
        bpy.data.images.remove(img)                      # branded lettering texture, stay out of the file
    for p in mixer.data.polygons:
        p.use_smooth = True
    mixer.name = tag + "Mixer"
    mixer.parent = root
    mixer.scale = (KA_SCALE,) * 3
    mixer.location = (0, 0, -KA_BASE * KA_SCALE)
    mixer.rotation_euler = (0, 0, yaw)
    return mixer


def make_bowl(name, bowl, mat):
    """Lathe a glass bowl: flat floor, filleted corner, straight wall, rounded rim."""
    R, rf, H, tg = bowl["R"], bowl["rf"], bowl["wall_h"], bowl["glass"]
    arc = lambda cx, cz, r, a0, a1, n: [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                                         cz + r * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]
    top = tg + H
    prof = ([(0.0, tg)] + arc(R - rf, tg + rf, rf, 270, 360, 10) + [(R, top)] +
            arc(R + tg / 2, top, tg / 2, 180, 0, 8) +
            arc(R - rf, tg + rf, rf + tg, 0, -90, 10) + [(0.0, 0.0)])
    N = 128
    bm = bmesh.new()
    rings = []
    for r, zz in prof:
        if r < 1e-6:
            rings.append([bm.verts.new((0, 0, zz))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * k / N), r * math.sin(2 * math.pi * k / N), zz))
                          for k in range(N)])
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(N):
            j = (k + 1) % N
            if len(a) == 1:
                bm.faces.new((a[0], b[k], b[j]))
            elif len(b) == 1:
                bm.faces.new((a[k], b[0], a[j]))
            else:
                bm.faces.new((a[k], b[k], b[j], a[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    return B.link(bpy.data.objects.new(name, me))


def make_part_mesh(name, part, M, wood, parent, bare=False, hc=HC):
    """Model of one tool part, in the frame the physics uses (x across, z up)."""
    g = part["geom"]
    H = g["H"]
    handle_mat = M["steel"]
    shaft_top = 400.0 if bare else hc - 5.0
    is_fork = part["shape"] == "fork"
    if g["kind"] == "outline":
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "2D"
        cu.fill_mode = "BOTH"
        cu.extrude = 0.55
        cu.bevel_depth = 0.25
        cu.bevel_resolution = 2
        outlines = [fork_visual(shaft_top)] if is_fork else [g["outer"]] + g["holes"]     # a fork gets its handle
        for pts in outlines:
            sp = cu.splines.new("POLY")
            sp.points.add(len(pts) - 1)
            for q, (x, zz) in zip(sp.points, pts):
                q.co = (x, zz, 0, 1)
            sp.use_cyclic_u = True
        cu.materials.append(M["steel"])
        ob = B.link(bpy.data.objects.new(name, cu))
        ob.rotation_euler = (math.radians(90), 0, 0)
    elif g["kind"] == "spoon":
        a, b, depth = g["a"], g["b"], g["depth"]
        bm = bmesh.new()
        nr, ns = 9, 48
        centre = bm.verts.new((0, float(spoon_dish(np.array(0.0), np.array(b), a, b, depth)), b))
        rings = []
        for i in range(1, nr + 1):
            rho = i / nr
            ring = []
            for k in range(ns):
                u = a * rho * math.cos(2 * math.pi * k / ns)
                w = b + b * rho * math.sin(2 * math.pi * k / ns)
                ring.append(bm.verts.new((u, float(spoon_dish(np.array(u), np.array(w), a, b, depth)), w)))
            rings.append(ring)
        for k in range(ns):
            bm.faces.new((centre, rings[0][k], rings[0][(k + 1) % ns]))
        for r0, r1 in zip(rings[:-1], rings[1:]):
            for k in range(ns):
                j = (k + 1) % ns
                bm.faces.new((r0[k], r1[k], r1[j], r0[j]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(wood)
        ob = B.link(bpy.data.objects.new(name, me))
        md = ob.modifiers.new("Thickness", "SOLIDIFY")
        md.thickness = 1.6
        md.offset = 0.0
        handle_mat = wood
    else:                                           # wires
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.7
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
        for wire in g["wires"]:
            sp = cu.splines.new("POLY")
            sp.points.add(len(wire) - 1)
            for q, (x, y, zz) in zip(sp.points, wire):
                q.co = (x, y, max(zz, 0.7), 1)
        cu.materials.append(M["steel"])
        ob = B.link(bpy.data.objects.new(name, cu))
    ob.parent = parent
    # handle / shaft up to the chuck
    r_h = 2.8 if g["kind"] in ("spoon",) else 2.4
    if not is_fork:
        B.cylinder(name + "Shaft", r_h, H - 2.0, shaft_top, handle_mat, parent=parent)
    if g["kind"] == "wires":
        B.cylinder(name + "Ferrule", 3.6, H - 3.0, H + 6.0, M["steel"], parent=parent, bevel=0.8)
    if not bare:
        B.cylinder(name + "Chuck", 6.2, hc - 7.0, hc, M["steel"], parent=parent, bevel=1.0)
    return ob


def make_station(tag, x0, cache_dir, M, wood, args, belt_w):
    """One bowl + grains + mixer head with its tool, animated from the cache."""
    cache_dir = os.path.abspath(cache_dir)
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    z = np.load(os.path.join(cache_dir, "meta.npz"))
    bowl = meta["bowl"]
    tg = bowl["glass"]
    n = meta["n_frames"]
    frames = list(range(1, n + 1))
    tool = build_tool(meta["tool"], meta["grain"], bowl)

    root = B.link(bpy.data.objects.new(tag + "Station", None))
    root.location = (x0, 0, 0)
    root.empty_display_size = 30
    ka = args.mixer == "kitchenaid" and not getattr(args, "bare", False)
    seat = (KA_SEAT - KA_BASE) * KA_SCALE if ka else 0.0      # the bowl stands on the mixer's base
    hc = 62.0 if ka else HC
    glass = make_bowl(tag + "Bowl", bowl, M["glass"])
    glass.parent = root
    glass.location = (0, 0, seat)
    grains, mc = B.build_grains(cache_dir, meta, z, root, meta["ingredients"], args.grain_scale)
    grains.name = tag + "Grains"
    grains.location = (0, 0, seat + tg)
    mc.play_mode = "SCENE"               # sample k is frame k + 1
    mc.frame_start = 1.0
    if getattr(args, "embed", None):     # pack the animation into the .blend instead of pointing at the cache
        os.makedirs(args.embed, exist_ok=True)
        B.embed_cache(grains, mc, grains.modifiers["Grains"].node_group, os.path.join(cache_dir, "particles.pc2"),
                      meta["n_grains"], os.path.join(args.embed, f"{tag}_{os.path.basename(cache_dir)}.png"))

    bare = getattr(args, "bare", False)          # tool and shaft only: nothing between camera and bowl
    M = dict(M)                                  # this station's mixer wears its contestant's colour
    paint = KITCHENAID_COLOR[meta["tool"]][1] if ka else TOOL_COLOR[meta["tool"]]
    M["body"] = B.principled(tag + "Enamel", tuple(B.hex_lin(paint)), rough=0.32,
                             **{"Coat Weight": 1.0, "Coat Roughness": 0.035})
    M["white"] = B.principled(tag + "White", (0.85, 0.85, 0.83), rough=0.35)
    if ka:
        make_kitchenaid(tag, root, M, M["body"], math.radians(args.yaw) * (1 if x0 <= 0 else -1))
    # --- mixer: column behind the belt, cylindrical motor housing on a carriage -------
    y_col = belt_w / 2 + 40
    top = tg + bowl["clear"] + bowl["lift"] + HC + 22 + 2 * HEAD_R + 16
    if not bare and not ka:
        B.cylinder(tag + "Base", 44, B.FLOOR_Z, B.FLOOR_Z + 14, M["body"], parent=root, loc=(0, y_col + 6, 0), bevel=5, segs=64)
        B.cylinder(tag + "Column", 15, B.FLOOR_Z + 8, top, M["steel"], parent=root, loc=(0, y_col, 0), segs=48)
        B.cylinder(tag + "ColumnCap", 17, top, top + 6, M["body"], parent=root, loc=(0, y_col, 0), bevel=2.5)

    head = B.link(bpy.data.objects.new(tag + "Head", None))      # origin: tool bottom, on the bowl axis
    head.parent = root
    if not bare and not ka:
        zc = HC + 22 + HEAD_R                                        # housing axis height above the tool bottom
        y0, y1 = -HEAD_R + 4, y_col + 24
        housing = B.cylinder(tag + "Housing", HEAD_R, -(y1 - y0) / 2, (y1 - y0) / 2, M["body"], parent=head, segs=96)
        housing.rotation_euler = (math.radians(90), 0, 0)
        housing.location = (0, (y0 + y1) / 2, zc)
        md = housing.modifiers.new("Dome", "BEVEL")                  # rounded ends: a capsule
        md.width = HEAD_R * 0.82
        md.segments = 12
        md.limit_method = "ANGLE"
        for pnt in housing.data.polygons:
            pnt.use_smooth = True
        for yy, w in ((y0 + HEAD_R * 0.95, 4.0), (y_col - 22, 3.0)):  # chrome trim bands
            ring = B.cylinder(tag + f"Band{yy:.0f}", HEAD_R + 0.5, -w / 2, w / 2, M["steel"], parent=head, segs=96, bevel=0.5)
            ring.rotation_euler = (math.radians(90), 0, 0)
            ring.location = (0, yy, zc)
        knob = B.cylinder(tag + "Knob", 6.5, 0, 8, M["rubber"], parent=head, bevel=1.4)
        knob.rotation_euler = (0, math.radians(90), 0)
        knob.location = (HEAD_R - 2, y0 + HEAD_R + 22, zc + 4)
        B.cylinder(tag + "Neck", 19, HC + 19, HC + 30, M["body"], parent=head, bevel=1.5, segs=64)

    rotor = B.link(bpy.data.objects.new(tag + "Rotor", None))
    rotor.parent = head
    dus = [p["du"] for p in tool["parts"]]
    reach = max(float(z["e"].max()) + max(dus) + 9, 16)
    mount = B.link(bpy.data.objects.new(tag + "Mount", None))
    mount.parent = rotor
    if not bare:
        B.cylinder(tag + "Hub", 22, hc + 9, hc + 20, M["turned"], parent=rotor, bevel=1.5, segs=64)
        B.box(tag + "Arm", (-10, -6.5, hc + 4.5), (reach, 6.5, hc + 9.5), M["steel"], bevel=1.2, parent=rotor)
        B.box(tag + "Carriage", (min(dus) - 9, -8.5, hc - 1), (max(dus) + 9, 8.5, hc + 4.5), M["dark"], bevel=1.2, parent=mount)
        if ka:        # drive shaft up into the mixer's planetary hub; it telescopes as the tool lifts
            B.cylinder(tag + "Drive", 7.5, hc + 18, hc + 18 + (KA_HUB - KA_SEAT) * KA_SCALE, M["turned"], parent=rotor, segs=48)
    pivots = []
    for k, part in enumerate(tool["parts"]):
        piv = B.link(bpy.data.objects.new(f"{tag}Part{k}", None))
        piv.parent = mount
        piv.location = (part["du"], 0, 0)
        make_part_mesh(f"{tag}Tool{k}", part, M, wood, piv, bare, hc)
        pivots.append(piv)

    # --- animation -------------------------------------------------------------------
    hz = seat + tg + bowl["clear"] + bowl["lift"] * (1 - z["plunge"])
    B.keys(head, "location", [float(v) for v in hz], frames, index=2)
    B.keys(rotor, "rotation_euler", [float(v) for v in z["phi"]], frames, index=2)
    B.keys(mount, "location", [float(v) for v in z["e"]], frames, index=0)
    for k, piv in enumerate(pivots):
        B.keys(piv, "rotation_euler", [float(v) for v in (z["angles"][:, k] - z["phi"])], frames, index=2)
    return dict(meta=meta, z=z, frames=frames, root=root)


def build_hud(stations, cam, font, scene):
    """Camera-locked overlay: each side's tool name, live % mixed and bar, plus a clock."""
    coll = B.new_collection("HUD")
    ink = B.emission("HUD_Ink", tuple(B.hex_lin("#23272E")), 1.0)
    dim = B.emission("HUD_Dim", tuple(B.hex_lin("#C9C4BB")), 1.0)
    accent = B.emission("HUD_Accent", tuple(B.hex_lin("#E8532B")), 1.0)
    D = cam.data.dof.focus_distance
    half_w = D * cam.data.sensor_width / (2 * cam.data.lens)
    half_h = half_w * scene.render.resolution_y / scene.render.resolution_x
    root = B.link(bpy.data.objects.new("HUD", None), coll)
    root.parent = cam
    root.location = (0, 0, -D)
    u = half_h / 100.0

    def quiet(ob):
        ob.visible_shadow = ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = False

    def text(name, body, size, x, y, mat, align="LEFT"):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body, cu.size, cu.align_x = body, size, align
        if font:
            cu.font = font
        cu.materials.append(mat)
        ob = B.link(bpy.data.objects.new(name, cu), coll)
        ob.parent = root
        ob.location = (x, y, 0)
        quiet(ob)
        return ob

    def plane(name, x0, x1, y0, y1, mat, zoff=0.0):
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0, y0, 0), (x1 - x0, y0, 0), (x1 - x0, y1, 0), (0, y1, 0)], [], [(0, 1, 2, 3)])
        me.materials.append(mat)
        ob = B.link(bpy.data.objects.new(name, me), coll)
        ob.parent = root
        ob.location = (x0, 0, zoff)
        quiet(ob)
        return ob

    def show_when(ob, mask, frames):
        prev = None
        for f, m in zip(frames, mask):
            m = bool(m)
            if m != prev:
                ob.hide_render = ob.hide_viewport = not m
                ob.keyframe_insert("hide_render", frame=f)
                ob.keyframe_insert("hide_viewport", frame=f)
                prev = m

    mx, top, bar_w = half_w - 9 * u, half_h - 14 * u, 62 * u
    for side, st in stations:
        tag = "L" if side < 0 else "R"
        align = "LEFT" if side < 0 else "RIGHT"
        x = side * mx
        frames, mixv = st["frames"], np.clip(mix_curve(st["z"]), 0, 1)
        text(tag + "Name", st["meta"]["label"].upper(), 5.6 * u, x, top, ink, align)
        pct = np.round(mixv * 100).astype(int)
        for k in np.unique(pct):
            show_when(text(f"{tag}Pct{k:03d}", f"{k}%", 20 * u, x, top - 21 * u, ink, align), pct == k, frames)
        x0 = x if side < 0 else x - bar_w
        plane(tag + "Track", x0, x0 + bar_w, top - 29 * u, top - 26 * u, dim, -0.0001 * D)
        fill = plane(tag + "Fill", x0, x0 + bar_w, top - 29 * u, top - 26 * u, accent)
        B.keys(fill, "scale", [max(float(v), 1e-3) for v in mixv], frames, index=0)
        cnt = np.floor(st["z"]["count"]).astype(int)
        unit = st["meta"]["unit"]
        for k in np.unique(cnt):
            word = unit[:-1] if k == 1 and unit.endswith("s") else unit
            show_when(text(f"{tag}Cnt{k:03d}", f"{k} {word}", 5.2 * u, x, top - 37 * u, ink, align), cnt == k, frames)
    return coll


def build(args):
    sides = [(s, p) for s, p in ((-1, args.left), (1, args.right)) if p]
    metas = [json.load(open(os.path.join(p, "meta.json"))) for _, p in sides]
    n_frames = min(m["n_frames"] for m in metas)
    scene = B.new_scene(args, metas[0]["fps"], n_frames)
    M = B.make_materials(args)
    wood = B.principled("Wood", tuple(B.hex_lin("#C89A5E")), rough=0.55)
    ka = args.mixer == "kitchenaid" and not args.bare
    belt_w = B.build_conveyor(M, belt_w=560.0 if ka else 140.0)       # room for the mixers' bases
    spacing = (args.spacing / 2 if ka else 82.0) if len(sides) == 2 else 0.0
    stations = [(s, make_station("L" if s < 0 else "R", s * spacing, p, M, wood, args, belt_w)) for s, p in sides]
    cam = B.make_camera(scene, args, target=tuple(float(v) for v in args.target.split(",")))
    B.setup_lights(scene)
    if not args.no_hud:
        build_hud(stations, cam, B.load_font(), scene)
    B.setup_render(scene, args)
    # Reusing scene data between frames once left a tool rendered solid black for
    # 116 consecutive frames of an animation (fine when rendered alone), so it is off here.
    scene.render.use_persistent_data = False
    return scene


def final_settings(scene, args):
    """Cycles settings for the final render on a GPU: open the file, press Render Animation."""
    cy = scene.cycles
    cy.device = "GPU"                        # used when a GPU is enabled in Preferences > System
    cy.samples = args.samples                # upper limit; adaptive sampling stops earlier where the picture is clean
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.025
    cy.adaptive_min_samples = 24
    cy.use_denoising = True
    for attr, val in (("denoiser", "OPENIMAGEDENOISE"), ("denoising_input_passes", "RGB_ALBEDO_NORMAL"),
                      ("denoising_prefilter", "ACCURATE"), ("denoising_use_gpu", True)):
        try:
            setattr(cy, attr, val)
        except Exception as e:
            print("cycles setting", attr, e)
    cy.max_bounces = 10
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 6                    # metal seen in metal
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 12          # two glass bowls, front and back walls
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    r = scene.render
    r.use_persistent_data = False            # see build(): reusing data once left a tool black
    r.filepath = args.final
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.image_settings.compression = 15
    r.use_overwrite = False                  # stop and restart freely: finished frames are kept
    r.use_placeholder = True
    r.use_file_extension = True
    scene.frame_set(scene.frame_start)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", help="cache folder for the left (or only) bowl")
    ap.add_argument("--right", help="cache folder for the right bowl")
    ap.add_argument("--save")
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--still", type=int, nargs="*")
    ap.add_argument("--anim", action="store_true")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "renders", "bowl"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--real-glass", action="store_true")
    ap.add_argument("--no-dof", action="store_true")
    ap.add_argument("--motion-blur", action="store_true")
    ap.add_argument("--elev", type=float, default=36.0)
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--dist", type=float)
    ap.add_argument("--aperture", type=float, default=2.0)
    ap.add_argument("--target", default="0,6,46", help="point the camera looks at, x,y,z in mm")
    ap.add_argument("--mixer", choices=("kitchenaid", "capsule"), default="kitchenaid",
                    help="the machine at each bowl: the KitchenAid model in models/, or the plain capsule rig")
    ap.add_argument("--yaw", type=float, default=25.0, help="how far each KitchenAid is turned towards the middle (deg)")
    ap.add_argument("--spacing", type=float, default=200.0, help="distance between the two bowls with KitchenAids (mm)")
    ap.add_argument("--bare", action="store_true",
                    help="leave out the mixer head and column: tool and shaft only, for steep camera angles")
    ap.add_argument("--grain-scale", type=float, default=0.84)
    ap.add_argument("--no-lettering", action="store_true", help="leave the brand lettering off the mixer's trim band")
    ap.add_argument("--final", metavar="PATH",
                    help="set the scene up for the final GPU render and store PATH (e.g. //frames/Q1/f_) as its output")
    ap.add_argument("--embed", metavar="DIR",
                    help="pack the grain animation into the saved .blend (data textures written to DIR), "
                         "so the file renders without the simulation caches")
    args = ap.parse_args(argv)
    if not args.left:
        ap.error("--left is required")
    if args.dist is None:
        args.dist = 570.0 if args.right else 430.0
    global LETTERING
    LETTERING = not args.no_lettering
    scene = build(args)
    if args.final:
        final_settings(scene, args)
    if args.save:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        for ob in bpy.data.objects:
            if ob.name.endswith("Grains") and "Cache" in ob.modifiers:
                ob.modifiers["Cache"].filepath = bpy.path.relpath(ob.modifiers["Cache"].filepath)
        bpy.ops.wm.save_mainfile()
        print("saved", args.save)
    os.makedirs(args.out, exist_ok=True)
    for f in args.still or []:
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
        scene.render.use_overwrite = False
        scene.render.use_placeholder = True
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
