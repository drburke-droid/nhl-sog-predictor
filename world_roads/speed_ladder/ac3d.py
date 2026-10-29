"""Read AC3D (.ac) models, FlightGear's format, into Blender: the object tree with its transforms, polygons with UVs,
materials and textures (png, jpg or SGI .rgb). AC3D is y-up; Blender is z-up: (x, y, z) -> (x, -z, y)."""
import bpy, bmesh, os, math, re
from mathutils import Matrix, Vector


def _tok(line):
    out, cur, q = [], "", False
    for ch in line:
        if ch == '"': q = not q; continue
        if ch.isspace() and not q:
            if cur: out.append(cur); cur = ""
        else: cur += ch
    if cur: out.append(cur)
    return out


class Reader:
    def __init__(self, path):
        raw = open(path, encoding="latin-1", newline="").read().splitlines(keepends=True)
        self.lines = [l.rstrip("\r\n") for l in raw]
        self.lens = [len(l) for l in raw]                 # with line endings, for 'data' blocks
        self.i = 0
        self.dir = os.path.dirname(path)

    def next(self):
        while self.i < len(self.lines):
            l = self.lines[self.i]; self.i += 1
            if l.strip(): return l
        return None


def _find_tex(d, name):
    cands = [os.path.join(d, name), os.path.join(d, os.path.basename(name))]
    base, _ = os.path.splitext(name)
    for ext in (".png", ".rgb", ".jpg", ".dds", ".PNG", ".jpeg"):
        cands += [os.path.join(d, base + ext), os.path.join(d, os.path.basename(base) + ext)]
    up = os.path.dirname(d)
    cands += [os.path.join(up, "Textures", os.path.basename(name)), os.path.join(up, os.path.basename(name))]
    for c in cands:
        if os.path.exists(c): return c
    return None


def _material(cache, m, texpath, prefix):
    key = (id(m), texpath)
    if key in cache: return cache[key]
    name = f"{prefix}_{m['name']}" + (f"_{os.path.basename(texpath)}" if texpath else "")
    mat = bpy.data.materials.new(name[:60]); mat.use_nodes = True
    nt = mat.node_tree; p = nt.nodes["Principled BSDF"]
    r, g, b = m["rgb"]
    p.inputs["Base Color"].default_value = (r, g, b, 1)
    shi = m.get("shi", 0)
    p.inputs["Roughness"].default_value = max(0.12, 0.85 - 0.6 * min(shi, 128) / 128)
    trans = m.get("trans", 0.0)
    if texpath:
        tex = nt.nodes.new("ShaderNodeTexImage")
        try:
            tex.image = bpy.data.images.load(texpath, check_existing=True)
        except RuntimeError:
            tex.image = None
        if tex.image is not None:
            mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
            mul.inputs["Factor"].default_value = 1.0
            nt.links.new(tex.outputs["Color"], mul.inputs["A"])
            mul.inputs["B"].default_value = (min(1, r * 1.0 + 0.0), g, b, 1) if max(r, g, b) > 0.05 else (1, 1, 1, 1)
            nt.links.new(mul.outputs["Result"], p.inputs["Base Color"])
            if tex.image.depth in (32, 64, 128) or texpath.lower().endswith(".png"):
                nt.links.new(tex.outputs["Alpha"], p.inputs["Alpha"])
    if trans > 0.05:
        p.inputs["Alpha"].default_value = 1 - trans
        p.inputs["Roughness"].default_value = 0.05
    cache[key] = mat
    return mat


def load(path, prefix="ac", skip=None, collection=None):
    """import an .ac file; returns the created mesh objects. skip(name) -> True drops an object (and its children)"""
    R = Reader(path)
    head = R.next()
    assert head.startswith("AC3D"), head
    mats, objs, cache = [], [], {}
    col = collection or bpy.context.scene.collection

    def parse_object(parent_M, skipped):
        l0 = R.next()
        if l0 is None: return                       # some files promise more kids than they hold
        hdr = _tok(l0)
        kind = hdr[1] if len(hdr) > 1 else "poly"
        name, tex, texrep, texoff = kind, None, (1.0, 1.0), (0.0, 0.0)
        rot, loc = Matrix.Identity(3), Vector((0, 0, 0))
        verts, surfs = [], []
        nk = 0
        while True:
            l = R.next()
            if l is None: break
            t = _tok(l)
            k = t[0]
            if k == "name": name = t[1] if len(t) > 1 else ""
            elif k == "data":
                n = int(t[1]); got = 0
                while got < n and R.i < len(R.lines):
                    got += R.lens[R.i]; R.i += 1
            elif k == "texture": tex = t[1]
            elif k == "texrep": texrep = (float(t[1]), float(t[2]))
            elif k == "texoff": texoff = (float(t[1]), float(t[2]))
            elif k == "rot":
                v = [float(x) for x in t[1:10]]
                rot = Matrix(((v[0], v[1], v[2]), (v[3], v[4], v[5]), (v[6], v[7], v[8])))
            elif k == "loc": loc = Vector([float(x) for x in t[1:4]])
            elif k == "numvert":
                for _ in range(int(t[1])):
                    verts.append(Vector([float(x) for x in R.next().split()[:3]]))
            elif k == "numsurf":
                for _ in range(int(t[1])):
                    flags = int(_tok(R.next())[1], 0)
                    tm = _tok(R.next()); mi = int(tm[1]) if tm[0] == "mat" else 0
                    if tm[0] != "mat": R.i -= 1
                    nref = int(_tok(R.next())[1])
                    refs = []
                    for _ in range(nref):
                        a = R.next().split()
                        refs.append((int(a[0]), float(a[1]) if len(a) > 1 else 0.0, float(a[2]) if len(a) > 2 else 0.0))
                    surfs.append((flags, mi, refs))
            elif k == "kids":
                nk = int(t[1])
                break
        # AC3D: a child's vertices are in its parent's frame, moved by its own loc (rot rarely used)
        M = parent_M @ Matrix.Translation(loc) @ rot.to_4x4()
        drop = skipped or (skip is not None and skip(name))
        if verts and surfs and not drop:
            bm = bmesh.new()
            bv = [bm.verts.new(M @ v) for v in verts]
            uvl = bm.loops.layers.uv.new()
            texpath = _find_tex(R.dir, tex) if tex else None
            matlist = []
            for flags, mi, refs in surfs:
                if flags & 0x0F != 0 or len(refs) < 3: continue
                idx = [r[0] for r in refs]
                if len(set(idx)) < 3: continue
                try:
                    f = bm.faces.new([bv[i] for i in idx])
                except ValueError:
                    continue
                f.smooth = bool(flags & 0x10)
                key = (mi, texpath)
                if key not in matlist: matlist.append(key)
                f.material_index = matlist.index(key)
                for lp, r in zip(f.loops, refs):
                    lp[uvl].uv = (r[1] * texrep[0] + texoff[0], r[2] * texrep[1] + texoff[1])
            me = bpy.data.meshes.new(name or kind)
            bm.to_mesh(me); bm.free()
            # y-up to z-up
            me.transform(Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
            for mi, tp in matlist:
                m = mats[mi] if mi < len(mats) else {"name": "default", "rgb": (0.7, 0.7, 0.7)}
                me.materials.append(_material(cache, m, tp, prefix))
            o = bpy.data.objects.new(f"{prefix}_{name}"[:60], me)
            col.objects.link(o)
            objs.append(o)
        for _ in range(nk):
            parse_object(M, drop)

    while True:
        l = R.next()
        if l is None: break
        t = _tok(l)
        if t[0] == "MATERIAL":
            m = {"name": t[1], "rgb": (0.7, 0.7, 0.7)}           # (an empty name "" may vanish when tokenized)
            for j in range(1, len(t)):
                if t[j] == "rgb": m["rgb"] = tuple(float(x) for x in t[j + 1:j + 4])
                elif t[j] == "shi": m["shi"] = float(t[j + 1])
                elif t[j] == "trans": m["trans"] = float(t[j + 1])
            mats.append(m)
        elif t[0] == "OBJECT":
            R.i -= 1
            parse_object(Matrix.Identity(4), False)
    return objs
