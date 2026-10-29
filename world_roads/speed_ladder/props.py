"""The props on the road: traffic cones and a speed-limit sign built here, oil drums (Poly Haven's barrel_03, CC0) and a
car parked on the shoulder (the BMW 1M of Blender's benchmark, CC0). Each returns its root object, origin on the ground."""
import bpy, bmesh, math, os
from mathutils import Vector, Matrix
HERE = os.path.dirname(os.path.abspath(__file__))


def asset(rel, fallback):
    """an asset kept beside the code (assets/), else where the working copy keeps it"""
    p = os.path.join(HERE, "assets", rel)
    return p if os.path.exists(p) else fallback


DRUM = asset("barrel_03/barrel_03_1k.gltf", "/home/user/dl/ph/barrel_03/barrel_03_1k.gltf")
SIGN_TEX = asset("sign_100.png", "/home/user/lot/props/sign_100.png")


def mat(name, color, rough=0.5, metal=0.0, coat=0.0, spec=0.5):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if coat:
        p.inputs["Coat Weight"].default_value = coat
        p.inputs["Coat Roughness"].default_value = 0.04
    return m


def dusty(m, amount=0.35, height=0.6):
    """desert dust settled on the lower part of a material: mix toward tan with a noise, more near the ground"""
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    base = p.inputs["Base Color"].default_value[:]
    src = p.inputs["Base Color"].links[0].from_socket if p.inputs["Base Color"].is_linked else None
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.0, height
    ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = amount, 0.05
    nt.links.new(sep.outputs["Z"], ramp.inputs["Value"])
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 18
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(ramp.outputs[0], mul.inputs[0]); nt.links.new(noise.outputs["Fac"], mul.inputs[1])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs["A"].default_value = base
    if src is not None: nt.links.new(src, mix.inputs["A"])
    mix.inputs["B"].default_value = (0.42, 0.32, 0.22, 1)
    nt.links.new(mul.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], p.inputs["Base Color"])
    return m


def link(o):
    if o.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(o)


def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.active_object; o.name = name
    return o


# ---------------------------------------------------------------- traffic cone (0.71 m)
def cone(name="cone"):
    if "conemat" not in bpy.data.materials:
        m = bpy.data.materials.new("conemat"); m.use_nodes = True
        nt = m.node_tree; p = nt.nodes["Principled BSDF"]
        p.inputs["Roughness"].default_value = 0.42
        tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        cr = nt.nodes.new("ShaderNodeValToRGB"); cr.color_ramp.interpolation = "CONSTANT"
        e = cr.color_ramp.elements
        e[0].position, e[0].color = 0.0, (0.80, 0.16, 0.02, 1)
        e[1].position, e[1].color = 0.36, (0.80, 0.80, 0.78, 1)
        for pos, col in ((0.45, (0.80, 0.16, 0.02, 1)), (0.52, (0.80, 0.80, 0.78, 1)), (0.585, (0.80, 0.16, 0.02, 1))):
            x = e.new(pos); x.color = col
        nt.links.new(sep.outputs["Z"], cr.inputs["Fac"])
        nt.links.new(cr.outputs["Color"], p.inputs["Base Color"])
        dusty(m, 0.5, 0.35)
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.155, radius2=0.025, depth=0.68, location=(0, 0, 0.03 + 0.34))
    body = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True)
    body.data.materials.append(bpy.data.materials["conemat"])
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.015))
    base = bpy.context.active_object; base.scale = (0.36, 0.36, 0.03)
    bpy.ops.object.transform_apply(location=True, scale=True)
    bv = base.modifiers.new("bevel", "BEVEL"); bv.width = 0.02; bv.segments = 3
    bpy.ops.object.modifier_apply(modifier="bevel")
    base.data.materials.append(dusty(mat("conebase", (0.02, 0.02, 0.02), 0.85), 0.5, 0.05))
    o = join([body, base], name)
    return o


# ---------------------------------------------------------------- oil drum (Poly Haven barrel_03, 0.93 m)
def drum(name="drum"):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=DRUM)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in new:
        if o.parent: 
            mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
    o = join(meshes, name) if len(meshes) > 1 else meshes[0]
    o.name = name
    for x in new:
        if x != o and x.name in bpy.data.objects: bpy.data.objects.remove(x)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    # origin at the bottom centre
    zs = [v.co.z for v in o.data.vertices]; xs = [v.co.x for v in o.data.vertices]; ys = [v.co.y for v in o.data.vertices]
    off = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, min(zs)))
    o.data.transform(Matrix.Translation(-off))
    o.location = (0, 0, 0)
    return o


# ---------------------------------------------------------------- speed-limit sign
def sign_texture():
    if os.path.exists(SIGN_TEX): return SIGN_TEX
    from PIL import Image, ImageDraw, ImageFont
    S = 1024
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((4, 4, S - 4, S - 4), fill=(200, 24, 28, 255))
    d.ellipse((120, 120, S - 120, S - 120), fill=(238, 238, 232, 255))
    f = ImageFont.truetype("/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", 400)
    w = d.textlength("100", font=f)
    d.text(((S - w) / 2, S / 2 - 250), "100", font=f, fill=(20, 20, 20, 255))
    os.makedirs(os.path.dirname(SIGN_TEX), exist_ok=True)
    im.save(SIGN_TEX)
    return SIGN_TEX


def sign(name="sign", height=2.3, radius=0.38):
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.035, depth=height, location=(0, 0, height / 2))
    pole = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True)
    pole.data.materials.append(dusty(mat("galv", (0.55, 0.56, 0.57), 0.45, 1.0), 0.4, 0.5))
    bpy.ops.object.shade_smooth()
    # the disc faces -Y (toward the camera / oncoming traffic)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=0.012, location=(0, -0.045, height - radius + 0.02),
                                        rotation=(math.pi / 2, 0, 0))
    disc = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True)
    zc = height - radius + 0.02
    m = bpy.data.materials.new("signface"); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(sign_texture())
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    # object coords x, z in [-r, r] around the disc centre -> [0, 1]
    mp.inputs["Location"].default_value = (0.5, 0.5 - zc * 0.5 / radius, 0)
    mp.inputs["Scale"].default_value = (0.5 / radius, 0.5 / radius, 1)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); com = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], com.inputs["X"]); nt.links.new(sep.outputs["Z"], com.inputs["Y"])
    nt.links.new(com.outputs[0], mp.inputs["Vector"]); nt.links.new(mp.outputs[0], tex.inputs["Vector"])
    back = nt.nodes.new("ShaderNodeMix"); back.data_type = "RGBA"
    back.inputs["B"].default_value = (0.5, 0.5, 0.5, 1)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    # the face looks along -y: grey back where the normal points +y
    nsep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(geo.outputs["Normal"], nsep.inputs[0])
    gt = nt.nodes.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.0
    nt.links.new(nsep.outputs["Y"], gt.inputs[0])
    nt.links.new(gt.outputs[0], back.inputs["Factor"])
    nt.links.new(tex.outputs["Color"], back.inputs["A"])
    nt.links.new(back.outputs["Result"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.35
    disc.data.materials.append(m)
    o = join([pole, disc], name)
    return o


def shattered():
    """a blown-in tempered window: dark opening, a crazed rim of glass left in the frame"""
    m = bpy.data.materials.get("shattered")
    if m: return m
    m = bpy.data.materials.new("shattered"); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes["Principled BSDF"]
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.feature = "DISTANCE_TO_EDGE"; vor.inputs["Scale"].default_value = 60
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[0].color = 0.0, (0.35, 0.36, 0.37, 1)
    ramp.color_ramp.elements[1].position, ramp.color_ramp.elements[1].color = 0.06, (0.006, 0.006, 0.006, 1)
    nt.links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.5
    return m


# ---------------------------------------------------------------- the parked car: the BMW 1M from Blender's benchmark
BMW = asset("bmw27_cpu.blend", "/home/user/lot/props/bmw/bmw27_cpu.blend")   # "bmw27" by Mike Pan, Blender benchmark, CC0
BMW_LENGTH = 4.38                                     # a real 1M coupe, m


TYRE_SQUASH = 0.012                  # m each tyre is flattened where it meets the road


def car(name="car"):
    """the benchmark's BMW 1M, scaled to its real length, front along +y, wheels on the ground, under one empty;
    its glass is tagged so a blast can shatter it, its badges hidden"""
    with bpy.data.libraries.load(BMW, link=False) as (src, dst):
        dst.collections = ["1M"]
    col = bpy.data.collections["1M"]
    col.name = f"{name}_col"
    bpy.context.scene.collection.children.link(col)
    for o in list(col.objects):
        if o.type == "ARMATURE": bpy.data.objects.remove(o)
    # the benchmark's car is a rig (the wheels follow the body through constraints and drivers, and sit on a Floor
    # object by a Floor constraint): freeze it in its pose, so it moves and scales as one piece
    bpy.context.view_layer.update()
    depth = lambda o: 0 if o.parent is None else 1 + depth(o.parent)
    frozen = [(o, o.matrix_world.copy()) for o in sorted(col.objects, key=depth)]
    for o, mw in frozen:
        o.constraints.clear()
        if o.animation_data: o.animation_data_clear()
    for o, mw in frozen:
        o.matrix_world = mw
        bpy.context.view_layer.update()
    shell = bpy.data.objects["carShell"]
    for o in col.objects:
        if o.name.startswith("Roundel") or o.name == "MStreak": o.hide_render = True
        if o.type == "MESH" and any(m and m.name == "Glass" for m in o.data.materials): o["glass"] = True
    root = bpy.data.objects.new(name, None); link(root)
    bpy.context.view_layer.update()
    # its size and footprint from the evaluated meshes (mirror and subdivision applied, wheels instanced)
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH" or ob.name.startswith("Roundel"): continue
        if not (inst.is_instance or ob.original.name in col.objects): continue
        mw = inst.matrix_world
        me = ob.data
        step = max(1, len(me.vertices) // 400)
        pts += [mw @ me.vertices[i].co for i in range(0, len(me.vertices), step)]
    xs = [p.x for p in pts]; ys = [p.y for p in pts]
    length = max(xs) - min(xs)
    k = BMW_LENGTH / length
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    # the ground is where the tyres touch it: the lowest point of every tyre (all its vertices), less a little for the
    # weight of the car flattening each one onto the road
    tyres = [min((inst.matrix_world @ v.co).z for v in inst.object.data.vertices) for inst in dg.object_instances
             if inst.object.type == "MESH" and inst.object.original.name.startswith("TireRubber")]
    z0 = (sum(tyres) / len(tyres) if tyres else min(p.z for p in pts)) + TYRE_SQUASH / k
    shell.parent = root
    # put the shell so the car's centre is at the root and its tyres touch the ground, scaled, front to +y
    shell.matrix_parent_inverse = Matrix.Rotation(-math.pi / 2, 4, "Z") @ Matrix.Scale(k, 4) @ Matrix.Translation((-cx, -cy, -z0))
    root["length"], root["scale"] = BMW_LENGTH, k
    return root
