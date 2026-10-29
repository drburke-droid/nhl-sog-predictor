"""The five machines, built in Blender ready to fly: imported, cleaned (landing gear up, alternate parts hidden),
nose along +x (left to right across the frame), propellers on their own pivots so they can spin.

Models: FlightGear's Wright Flyer, Spitfire IX, Bell X-1 and X-15 (GPL), NASA's Parker Solar Probe (public domain).
FlightGear's models point their nose along -x (its x axis runs aft), y to the right, z up."""
import bpy, math, os, re, sys
from mathutils import Vector, Matrix, Quaternion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ac3d

FG = "/home/user/dl/fg/x"
NASA = "/home/user/nasa/nasa-3d-resources/3D Models"
_GEN = r"pilot_|cockpit|interior|panel|instrument|shadow|seat|yoke|stick|pedal|throttle|halo|flare|glow|beacon|strobe|" \
       r"smoke|contrail|flame|chute|light.?cone|procedural"

AIRCRAFT = {
    "wright": dict(
        title="Wright Flyer", year=1903, mph=30, speed=13.4, alt=3.0,
        files=[(f"{FG}/wrightFlyer1903/wrightFlyer1903/Models/WrightFlyer-pb-jw.ac", (0, 0, 0))],
        hide=_GEN + r"|Skid|Runner|Cradle", keep_people=True,
        props=[("LeftPropeller", "PropHubLeft", 35.0), ("RightPropeller", "PropHubRight", -35.0)]),
    "spitfire": dict(
        title="Spitfire", year=1943, mph=370, speed=165.4, alt=5.0,
        files=[(f"{FG}/spitfireIX/spitfireIX/Models/spitfire9.ac", (0, 0, 0)),
               (f"{FG}/spitfireIX/spitfireIX/Models/Propeller/propeller.ac", (0.83, 0.0, 0.0522))],
        hide=_GEN + r"|exttip|lftip|fightercanopy|interiour|tailgear|tailwheel",
        props=[("prop[1-4]|prophub", None, 142.0)]),
    "x1": dict(
        title="Bell X-1", year=1947, mph=807, mach=1.06, speed=1.06 * 343.0, alt=6.0,
        files=[(f"{FG}/Bell-X1/Bell-X1/Models/x1.ac", (0, 0, 0))],
        hide=_GEN + r"|roue|axe[ADG]", props=[]),
    "f35": dict(
        title="F-35 Lightning II", year=2016, mph=1228, speed=548.8, alt=6.0,
        files=[(f"{FG}/../f35/x/F-35B/Models/F-35B.ac", (0, 0, 0)), (f"{FG}/../f35/x/F-35B/Models/Engine.ac", (0, 0, 0)),
               (f"{FG}/../f35/x/F-35B/Models/Pilot.ac", (0, 0, 0))],
        hide=_GEN + r"|^(fan|box|case|lights|fuel.*|ReheatL)$", keep_people=False, props=[]),
    "x15": dict(
        title="X-15", year=1967, mph=4520, speed=2020.6, alt=6.0,
        files=[(f"{FG}/X15/X15/Models/X15.ac", (0, 0, 0))],
        hide=_GEN + r"|Tyre|Rim|NoseOleo|NoseStrut|NoseDoor|GearWell|LeftArm|RightArm|Skid|MainStrut|LowerStrut|"
                    r"UpperStrut|GearFulcrum", props=[]),
    "parker": dict(
        title="Parker Solar Probe", year=2024, mph=430000, speed=192227.0, alt=5.0,
        files=[(f"{NASA}/Parker Solar Probe/Parker Solar Probe.glb", (0, 0, 0))], hide=r"^_root$", props=[], glb=True,
        yaw=-math.pi / 2),   # its heat shield faces +y: turn it to lead
}


def _bounds(objs):
    pts = []
    for o in objs:
        if o.type == "MESH" and not o.hide_render:
            n = len(o.data.vertices)
            pts += [o.matrix_world @ o.data.vertices[i].co for i in range(0, n, max(1, n // 3000))]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi


def build(name):
    """-> root empty (origin at the machine's centre, nose along +x), [(pivot empty, rad/s)] for its propellers"""
    cfg = AIRCRAFT[name]
    hide = re.compile(cfg["hide"], re.I)
    objs = []
    for path, off in cfg["files"]:
        before = set(bpy.data.objects)
        if cfg.get("glb"):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            ac3d.load(path, prefix=name, skip=lambda n: bool(hide.search(n or "")) and not (
                cfg.get("keep_people") and "Orville" in (n or "")))
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            if hide.search(o.name):
                o.hide_render = o.hide_viewport = True
        for o in new:
            if o.parent is None and any(off):
                o.location = Vector(o.location) + Vector(off)
        objs += new
    bpy.context.view_layer.update()
    lo, hi = _bounds(objs)
    c = (lo + hi) / 2
    root = bpy.data.objects.new(name, None); bpy.context.scene.collection.objects.link(root)
    fix = Matrix.Rotation(cfg.get("yaw", math.pi), 4, "Z")   # nose to +x
    for o in objs:
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = fix @ Matrix.Translation(-c) @ mw
    bpy.context.view_layer.update()
    pivots = []
    for pat, hub, w in cfg["props"]:
        rx = re.compile(pat, re.I)
        blades = [o for o in objs if o.type == "MESH" and rx.search(o.name)]
        if not blades: continue
        ref = [o for o in objs if hub and hub in o.name] or blades
        lo_, hi_ = _bounds(ref)
        piv = bpy.data.objects.new(f"{name}_pivot{len(pivots)}", None)
        bpy.context.scene.collection.objects.link(piv)
        piv.parent = root
        piv.matrix_world = Matrix.Translation((lo_ + hi_) / 2) @ root.matrix_world.to_3x3().to_4x4()
        bpy.context.view_layer.update()
        for b in blades:
            mw = b.matrix_world.copy(); b.parent = piv; b.matrix_world = mw
        piv.rotation_mode = "XYZ"
        pivots.append((piv, w))
    size = hi - lo
    root["length"], root["span"] = max(size.x, size.y), min(size.x, size.y)
    return root, pivots


# rocket exhaust: the X-1's four alcohol/LOX chambers (a bright plume with shock diamonds), the X-15's ammonia/LOX
# XLR99 (fainter, longer); (object holding the nozzles, how many, plume length m, diamond spacing in exit diameters,
# colour near the nozzle, colour downstream, emission)
PLUMES = {
    "x1": ("sorties", 4, 3.2, 1.6, (1.0, 0.82, 0.62), (1.0, 0.42, 0.18), 14.0),
    "x15": ("Nozzle", 1, 11.0, 1.3, (1.0, 0.75, 0.62), (1.0, 0.45, 0.3), 0.9),
    # the F-35's F135 in full afterburner: a long yellow-orange flame with shock diamonds
    "f35": ("NozzlePetal", 1, 8.0, 1.2, (1.0, 0.72, 0.38), (1.0, 0.4, 0.12), 6.0),
}


def _nozzle_exits(objs, count):
    """[(centre, radius)] of the nozzle exits: the vertices at the aft end, split into `count` clusters"""
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    xmin = min(p.x for p in pts)
    aft = [p for p in pts if p.x < xmin + 0.04]
    c = sum(aft, Vector()) / len(aft)
    if count == 1:
        return [(c, max((Vector((0, p.y, p.z)) - Vector((0, c.y, c.z))).length for p in aft))]
    # four chambers in a diamond round the centre: group by angle about the axis
    groups = {}
    for p in aft:
        a = math.atan2(p.z - c.z, p.y - c.y)
        k = int(((a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))
        groups.setdefault(k, []).append(p)
    out = []
    for g in groups.values():
        cc = sum(g, Vector()) / len(g)
        out.append((cc, max((Vector((0, p.y, p.z)) - Vector((0, cc.y, cc.z))).length for p in g)))
    return out


def plume(root, name):
    """emissive volumes trailing from each nozzle exit, parented to the machine"""
    if name not in PLUMES: return []
    objname, count, length, lam_d, c0, c1, strength = PLUMES[name]
    src = [o for o in bpy.data.objects if o.type == "MESH" and objname in o.name and o.name.startswith(name + "_")
           and not o.hide_render and len(o.data.vertices) > 8]
    big = [o for o in src if len(o.data.vertices) > 50]
    out = []
    for i, (c, r0) in enumerate(_nozzle_exits(src if objname == "NozzlePetal" else big[:1], count)):
        r0 *= 0.92
        grow = 2.2 if name == "x15" else 1.8
        segs, rings = 24, 48
        verts, faces = [], []
        for j in range(rings + 1):                                           # a flared tube along +x (aft)
            u = j / rings
            rr = r0 * (1 + (grow - 1) * u ** 0.7)
            for k in range(segs):
                a = 2 * math.pi * k / segs
                verts.append((u * length, rr * math.cos(a), rr * math.sin(a)))
        for j in range(rings):
            for k in range(segs):
                a0, a1 = j * segs + k, j * segs + (k + 1) % segs
                faces.append((a0, a1, a1 + segs, a0 + segs))
        verts += [(0, 0, 0), (length, 0, 0)]
        for k in range(segs):                                                # caps
            faces.append((len(verts) - 2, (k + 1) % segs, k))
            faces.append((len(verts) - 1, rings * segs + k, rings * segs + (k + 1) % segs))
        me = bpy.data.meshes.new(f"{name}_plume{i}"); me.from_pydata(verts, [], faces)
        o = bpy.data.objects.new(f"{name}_plume{i}", me)
        bpy.context.scene.collection.objects.link(o)
        o.parent = root
        o.matrix_world = root.matrix_world @ Matrix.Translation(c) @ Matrix.Rotation(math.pi, 4, "Z")
        o.data.materials.append(_plume_volume(f"{name}_plumemat{i}", length, r0, grow, lam_d * 2 * r0, c0, c1, strength))
        o.visible_shadow = False
        out.append(o)
    return out


def _plume_volume(mname, length, r0, grow, lam, c0, c1, strength):
    """emission only (it adds light, absorbs none): bright at the nozzle, fading downstream, a hot core, and shock
    diamonds (bright knots a diamond-spacing apart) dying away in the first part"""
    m = bpy.data.materials.new(mname)
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    N, Lk = nt.nodes.new, nt.links.new

    def op(o, a, b=None):
        n = N("ShaderNodeMath"); n.operation = o
        for i, v in enumerate((a, b)):
            if v is None: continue
            if isinstance(v, (int, float)): n.inputs[i].default_value = v
            else: Lk(v, n.inputs[i])
        return n.outputs[0]
    tc = N("ShaderNodeTexCoord"); sep = N("ShaderNodeSeparateXYZ"); Lk(tc.outputs["Object"], sep.inputs[0])
    x = op("MAXIMUM", sep.outputs["X"], 0.0)
    u = op("DIVIDE", x, length)
    r = op("SQRT", op("ADD", op("MULTIPLY", sep.outputs["Y"], sep.outputs["Y"]), op("MULTIPLY", sep.outputs["Z"], sep.outputs["Z"])))
    rmax = op("MULTIPLY", r0, op("ADD", 1.0, op("MULTIPLY", grow - 1, op("POWER", u, 0.7))))
    rr = op("DIVIDE", r, rmax)                                                # 0 on the axis, 1 at the edge
    core = op("POWER", op("MAXIMUM", op("SUBTRACT", 1.0, op("MULTIPLY", rr, rr)), 0.0), 1.5)
    fade = op("EXPONENT", op("MULTIPLY", u, -2.2))
    cosd = op("COSINE", op("MULTIPLY", x, 2 * math.pi / lam))
    diam = op("POWER", op("MULTIPLY", op("ADD", cosd, 1.0), 0.5), 6.0)
    knots = op("MULTIPLY", diam, op("EXPONENT", op("MULTIPLY", u, -5.0)))
    noise = N("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 3.0 / max(r0, 0.05)
    noise.inputs["Detail"].default_value = 4.0
    Lk(tc.outputs["Object"], noise.inputs["Vector"])
    turb = op("ADD", 0.55, op("MULTIPLY", noise.outputs["Fac"], 0.9))
    dens = op("MULTIPLY", op("MULTIPLY", core, op("ADD", op("MULTIPLY", fade, 0.6), op("MULTIPLY", knots, 1.6))), turb)
    ramp = N("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c0, 1); ramp.color_ramp.elements[1].color = (*c1, 1)
    ramp.color_ramp.elements[1].position = 0.6
    Lk(u, ramp.inputs["Fac"])
    em = N("ShaderNodeEmission")
    Lk(ramp.outputs["Color"], em.inputs["Color"])
    Lk(op("MULTIPLY", dens, strength), em.inputs["Strength"])
    outn = N("ShaderNodeOutputMaterial")
    Lk(em.outputs[0], outn.inputs["Volume"])
    return m


def heat(name):
    """the X-15 at Mach 5.9 in sea-level air: the air stopped on its nose is at about 2,400 K, and its leading edges
    glow. The more a surface faces into the flow the hotter (900 K side-on to 2,300 K head-on); blackbody colour, and
    the visible brightness of a blackbody, which climbs as exp(-24,000 K / T): at 2,300 K some 70 times a sunlit white
    surface, at 1,500 K a dim glow, below 1,000 K nothing to see in daylight"""
    if name != "x15": return
    done = set()
    for o in bpy.data.objects:
        if o.type != "MESH" or not o.name.startswith(name + "_") or "plume" in o.name: continue
        for m in o.data.materials:
            if m is None or m.name in done or not m.use_nodes: continue
            done.add(m.name)
            nt = m.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None: continue
            N, Lk = nt.nodes.new, nt.links.new

            def op(o_, a_, b_=None):
                n_ = N("ShaderNodeMath"); n_.operation = o_
                for i, v in enumerate((a_, b_)):
                    if v is None: continue
                    if isinstance(v, (int, float)): n_.inputs[i].default_value = v
                    else: Lk(v, n_.inputs[i])
                return n_.outputs[0]
            geo = N("ShaderNodeNewGeometry")
            sep = N("ShaderNodeSeparateXYZ"); Lk(geo.outputs["Normal"], sep.inputs[0])
            f = op("MAXIMUM", sep.outputs["X"], 0.0)
            T = op("ADD", 900.0, op("MULTIPLY", 1400.0, op("POWER", f, 1.2)))
            bb = N("ShaderNodeBlackbody"); Lk(T, bb.inputs["Temperature"])
            st = op("MULTIPLY", 60.0, op("EXPONENT", op("SUBTRACT", 24000 / 2300, op("DIVIDE", 24000.0, T))))
            Lk(bb.outputs["Color"], bsdf.inputs["Emission Color"])
            Lk(st, bsdf.inputs["Emission Strength"])
