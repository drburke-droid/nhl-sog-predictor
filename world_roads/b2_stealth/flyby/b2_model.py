"""The B-2 Spirit, built in Blender ready to fly.

Model: FlightGear's B-2 Spirit by Markus Zojer (FGAddon, GPL v2+), from the Aircraft-2024 set on the FlightGear
mirrors (`fetch_assets.sh` puts it in /home/user/dl/fg/B-2). 52.2 m span, 21.2 m long, as the real aircraft.

FlightGear models point the nose along -x (x runs aft), y to the right, z up. The model rests with the gear up and
every door shut (its animations open them), so only the gear legs, tyres, lights, instruments and the cockpit
interior are left out. Its livery is a flight-sim one (white walkway outlines, "0088" serials, logos, credits): the
textures are toned to a few per cent of their contrast so the panel lines read only as faint differences in the gray,
as on the real aircraft.
"""
import bpy, math, os, re, sys
import numpy as np
from mathutils import Vector, Matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ac3d

B2_DIR = os.environ.get("B2_DIR", "/home/user/dl/fg/B-2")
AC = os.path.join(B2_DIR, "Models", "spirit.ac")
WORK = os.environ.get("B2_WORK", "/home/user/lot/b2")

SPAN_M, LENGTH_M = 52.4, 21.0
HIDE = re.compile(r"tire|^.*_gear(_\d)?$|front_gear|gear_hold|gearbox|la_light|light|pos_|refuel|instrument|nd_|pfd|"
                  r"glas_|clock|apmode|tcb|cons|chair|seat|pilot|ladder|agm|gust|^cockpit(_\d)?$|cockpit\.001", re.I)
# swatches in spirit_tex.rgb (flat colours the model samples for small parts): left as they are
SWATCH = {"spirit_tex": (400, 0, 512, 250)}
KEEP = 0.12          # share of each texture's contrast kept
LIFT = 1.15          # the real paint reads a touch lighter in daylight than the sim's


def clean_textures():
    """-> {texture name: cleaned png path}: each texture pulled toward its own base gray"""
    from PIL import Image
    os.makedirs(os.path.join(WORK, "tex"), exist_ok=True)
    out = {}
    for f in sorted(os.listdir(os.path.join(B2_DIR, "Models"))):
        if not f.endswith(".rgb"):
            continue
        stem = f[:-4]
        im = Image.open(os.path.join(B2_DIR, "Models", f)).convert("RGBA")
        a = np.asarray(im).astype(np.float32)
        rgb = a[..., :3]
        if stem.startswith("spirit"):
            lum = rgb.mean(axis=2)
            base = np.median(rgb[(lum > 20) & (lum < 120)], axis=0) if np.any((lum > 20) & (lum < 120)) else rgb.reshape(-1, 3).mean(0)
            new = base + (rgb - base) * KEEP
            if stem in SWATCH:
                x0, y0, x1, y1 = SWATCH[stem]
                new[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
            a[..., :3] = np.clip(new * LIFT, 0, 255)
        p = os.path.join(WORK, "tex", stem + ".png")
        Image.fromarray(a.astype(np.uint8)).save(p)
        out[f] = p
    return out


def _paint(mat):
    """the B-2's paint: a dull, slightly satin dark gray"""
    p = mat.node_tree.nodes.get("Principled BSDF")
    if p is None:
        return
    p.inputs["Roughness"].default_value = 0.52
    if "Specular IOR Level" in p.inputs:
        p.inputs["Specular IOR Level"].default_value = 0.4
    if "Coat Weight" in p.inputs:
        p.inputs["Coat Weight"].default_value = 0.0


def _glass(mat):
    p = mat.node_tree.nodes.get("Principled BSDF")
    nt = mat.node_tree
    for l in list(nt.links):
        if l.to_node == p and l.to_socket.name in ("Base Color", "Alpha"):
            nt.links.remove(l)
    p.inputs["Base Color"].default_value = (0.012, 0.016, 0.022, 1)
    p.inputs["Roughness"].default_value = 0.06
    p.inputs["Alpha"].default_value = 1.0
    if "Specular IOR Level" in p.inputs:
        p.inputs["Specular IOR Level"].default_value = 0.8


def _smooth_by_angle(me, angle):
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        e.smooth = not (len(e.link_faces) == 2 and e.calc_face_angle(0.0) > angle) and len(e.link_faces) == 2
    bm.to_mesh(me); bm.free()


def build(name="b2"):
    """-> root empty: origin at the aircraft's centre, nose along +x, left wing along +y, z up, real size"""
    tex = clean_textures()
    before = set(bpy.data.objects)
    ac3d.load(AC, prefix=name, skip=lambda n: bool(HIDE.search(n or "")))
    objs = [o for o in bpy.data.objects if o not in before]
    for o in objs:
        if HIDE.search(o.name.split(f"{name}_", 1)[-1]):
            o.hide_render = o.hide_viewport = True
    for img in bpy.data.images:
        src = os.path.basename(img.filepath)
        if src in tex:
            img.filepath = tex[src]; img.reload()
    for m in bpy.data.materials:
        if "glass" in m.name.lower() or "canopy" in m.name.lower():
            _glass(m)
        elif m.use_nodes:
            _paint(m)
    for o in objs:
        if o.type == "MESH" and "glass" in o.name.lower():
            for s in o.material_slots:
                if s.material: _glass(s.material)
        if o.type == "MESH" and len(o.data.polygons):
            _smooth_by_angle(o.data, math.radians(35))  # smooth within a surface, sharp where panels meet
    bpy.context.view_layer.update()
    vis = [o for o in objs if o.type == "MESH" and not o.hide_render]
    pts = [o.matrix_world @ v.co for o in vis for v in o.data.vertices]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    c = (lo + hi) / 2
    root = bpy.data.objects.new(name, None); bpy.context.scene.collection.objects.link(root)
    fix = Matrix.Rotation(math.pi, 4, "Z")
    for o in objs:
        if o.parent is None:
            mw = o.matrix_world.copy()
            o.parent = root; o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = fix @ Matrix.Translation(-c) @ mw
    root["span"], root["length"], root["height"] = hi.y - lo.y, hi.x - lo.x, hi.z - lo.z
    bpy.context.view_layer.update()
    return root


if __name__ == "__main__":
    # bpy-python b2_model.py OUT.blend : build it on its own and save it
    bpy.ops.wm.read_factory_settings(use_empty=True)
    r = build()
    print("span %.2f m, length %.2f m, height %.2f m" % (r["span"], r["length"], r["height"]))
    if len(sys.argv) > 1 and sys.argv[-1].endswith(".blend"):
        bpy.ops.wm.save_as_mainfile(filepath=sys.argv[-1])
