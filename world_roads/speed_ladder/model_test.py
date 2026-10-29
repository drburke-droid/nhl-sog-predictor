"""Render candidate aircraft over the plate, one still each, to judge the models:  bpy-python model_test.py name..."""
import bpy, sys, os, math, re, time
from mathutils import Vector, Matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene as S, ac3d

FG = "/home/user/dl/fg/x"
NASA = "/home/user/nasa/nasa-3d-resources/3D Models"
MODELS = {
    "wright": ("ac", f"{FG}/wrightFlyer1903/wrightFlyer1903/Models/WrightFlyer-pb-jw.ac"),
    "bleriot": ("ac", f"{FG}/bleriot-XI/bleriot-XI/Models/bleriot-XI.ac"),
    "spirit": ("ac", f"{FG}/Ryan-SoSL/Ryan-SoSL/Models/ryan-sosl.ac"),
    "p51": ("ac", f"{FG}/p51d/p51d/Models/P-51D-25NA.ac"),
    "spitfire": ("ac", f"{FG}/spitfireIX/spitfireIX/Models/spitfire9.ac"),
    "me262": ("ac", f"{FG}/ME-262/ME-262/Models/me262.ac"),
    "x1": ("ac", f"{FG}/Bell-X1/Bell-X1/Models/x1.ac"),
    "sr71": ("ac", f"{FG}/Lockheed-SR71/Lockheed-SR71/Models/sr71.ac"),
    "sr71b": ("ac", f"{FG}/SR71-BlackBird/SR71-BlackBird/Models/SR-71-Blackbird-Base.ac"),
    "xb70": ("ac", f"{FG}/XB-70/XB-70/Models/xb-70.ac"),
    "x15": ("ac", f"{FG}/X15/X15/Models/X15.ac"),
    "concorde": ("ac", f"{FG}/Concorde/Concorde/Models/concorde.ac"),
    "starship": ("ac", f"{FG}/Starship/Starship/Models/starship.ac"),
    "shuttle": ("glb", f"{NASA}/Space Shuttle (D)/Space Shuttle (D).glb"),
    "parker": ("glb", f"{NASA}/Parker Solar Probe/Parker Solar Probe.glb"),
}
SCALE = {"shuttle": 0.0254}
ROT = {}                                                 # extra heading (rad) where a model's nose isn't along -x
ROTX = {"shuttle": -math.pi / 2}
SKIP = re.compile(r"pilot|cockpit|interior|panel|instrument|shadow|seat|yoke|stick|pedal|throttle|halo|flare|disc|blur|"
                  r"glow|beacon|strobe|smoke|contrail|flame|afterburn|chute|reflect|landing.?light|light.?cone|procedural|"
                  r"head|hand|helmet|body_pilot|glare|canopy_frame_int", re.I)


def load(name):
    kind, path = MODELS[name]
    before = set(bpy.data.objects)
    if kind == "ac":
        ac3d.load(path, prefix=name, skip=lambda n: bool(SKIP.search(n or "")))
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(name, None); bpy.context.scene.collection.objects.link(root)
    for o in objs:
        if o.parent is None: o.parent = root
    bpy.context.view_layer.update()
    if name in SCALE:                                   # NASA's shuttle is in inches
        root.scale = (SCALE[name],) * 3
    if name in ROTX:                                    # ... and lies on its side
        root.rotation_euler.x = ROTX[name]
    bpy.context.view_layer.update()
    # drop strays (animation helpers parked far away): objects whose centre is far from the median centre
    import statistics
    cs = {o: o.matrix_world @ (sum((Vector(b) for b in o.bound_box), Vector()) / 8) for o in objs if o.type == "MESH"}
    med = Vector([statistics.median(c[i] for c in cs.values()) for i in range(3)])
    for o, c in cs.items():
        if (c - med).length > 40:
            o.hide_render = True
    pts = []
    for o in objs:
        if o.type == "MESH" and not o.hide_render:
            mw = o.matrix_world
            n = len(o.data.vertices)
            pts += [mw @ o.data.vertices[i].co for i in range(0, n, max(1, n // 2000))]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return root, objs, lo, hi


if __name__ == "__main__":
    names = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    os.makedirs("/home/user/lot/render/models", exist_ok=True)
    for name in names:
        s = S.reset(samples=20)
        root, objs, lo, hi = load(name)
        size = hi - lo
        print(f"{name}: {len(objs)} objects, size {size.x:.1f} x {size.y:.1f} x {size.z:.1f} m", flush=True)
        # FlightGear models point their nose to -x (x is aft); NASA's: guess the long axis. Fly left to right (+x).
        c = (lo + hi) / 2
        close = os.environ.get("CLOSE")
        L = max(size.x, size.y)
        dist = 1250 * L / 1100 if close else 30.0          # close: the plane spans most of the frame width
        z = 1.4 + 0.05 * dist if close else 6.0
        rot = Matrix.Rotation(math.pi + (ROT.get(name, 0.0)), 4, "Z")
        root.matrix_world = Matrix.Translation((0, dist, z)) @ rot @ Matrix.Translation(-c) @ root.matrix_world
        s.render.filepath = f"/home/user/lot/render/models/{name}{'_close' if close else ''}.png"
        t = time.time(); bpy.ops.render.render(write_still=True)
        print(f"{name}: rendered in {time.time() - t:.0f}s", flush=True)
