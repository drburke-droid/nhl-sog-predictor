"""Real-world assets for the scale cards (all CC0, downloaded to /opt/seveneves_assets, not in the repository).

  human      Blender Studio "Human Base Meshes" (CC0): a sculpted realistic body. We lower its arms from the
             A-pose and dress it: shirt, trousers, shoes and hair are smoothed copies of the body's own regions
             (face sets), pushed out a little, so they read as cloth, not paint.
  trees      Poly Haven tree_small_02 and jacaranda_tree (CC0).
  facades    ambientCG Facade001/005/006/018A/019A/020A, Concrete049, PavingStones070 (CC0, PBR, 2K).
  skies      Poly Haven kloofendal_48d_partly_cloudy_puresky and kloofendal_43d_clear_puresky (CC0 HDRIs with
             an unclipped sun, so the sky lights the scene and casts real shadows).
  Everest    Copernicus GLO-30 DEM (30 m, (c) DLR/Airbus, free licence): the real massif.
"""
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector, noise

import look

A = '/opt/seveneves_assets'
HUMAN_BLEND = f'{A}/human/bundle/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
TEX = f'{A}/tex'
N = look.node


def img(path, colorspace='sRGB'):
    return look.img(path, colorspace)


# ------------------------------------------------------------------------------------ sky
def hdri_world(sc, name='kloofendal_48d_partly_cloudy_puresky_4k', sun_az=210.0, strength=1.0, exposure=-0.4):
    """An HDRI sky turned so its sun sits at azimuth sun_az (degrees from +x, counter-clockwise)."""
    info = json.load(open(f'{A}/hdri/sun.json'))[name]
    az0 = 180.0 - (info['u'] - 0.5) * 360.0
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground')
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = img(f'{A}/hdri/{name}.hdr', 'Linear Rec.709')
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Rotation'].default_value = (0, 0, math.radians(az0 - sun_az))
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
    nt.links.new(env.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = strength
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    sc.view_settings.exposure = exposure
    el = math.radians(info['elev'])
    a = math.radians(sun_az)
    return Vector((math.cos(a) * math.cos(el), math.sin(a) * math.cos(el), math.sin(el)))


# ------------------------------------------------------------------------------------ PBR textures
_PBR = {}


def pbr(name, tile, tile_h=None, roughness_scale=1.0, normal=1.0, tint=None, box=True, key=None):
    """An ambientCG material mapped in object space: tile = metres per texture repeat (x, y), tile_h along z."""
    key = key or f'{name}_{tile}_{tile_h}_{tint}'
    if key in _PBR:
        return _PBR[key]
    tile_h = tile_h or tile
    d = f'{TEX}/{name}/{name}_2K-JPG'
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    tc = N(nt, 'ShaderNodeTexCoord', (-1200, 0))
    oi = N(nt, 'ShaderNodeObjectInfo', (-1200, -300))
    mp = N(nt, 'ShaderNodeMapping', (-1000, 0))
    mp.inputs['Scale'].default_value = (1 / tile, 1 / tile, 1 / tile_h)
    off = N(nt, 'ShaderNodeVectorMath', (-1050, -250), operation='MULTIPLY')   # per-object random offset
    nt.links.new(oi.outputs['Random'], off.inputs[0])
    off.inputs[1].default_value = (37.0, 53.0, 0.0)
    snap = N(nt, 'ShaderNodeVectorMath', (-900, -250), operation='FLOOR')
    nt.links.new(off.outputs['Vector'], snap.inputs[0])
    add = N(nt, 'ShaderNodeVectorMath', (-800, 0), operation='ADD')
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], add.inputs[0])
    nt.links.new(snap.outputs['Vector'], add.inputs[1])

    def tex(suffix, cs, loc):
        p = f'{d}_{suffix}.jpg'
        if not os.path.exists(p):
            return None
        t = N(nt, 'ShaderNodeTexImage', loc)
        t.image = img(p, cs)
        if box:
            t.projection = 'BOX'
            t.projection_blend = 0.05
        nt.links.new(add.outputs['Vector'], t.inputs['Vector'])
        return t
    col = tex('Color', 'sRGB', (-500, 300))
    if tint:
        mix = N(nt, 'ShaderNodeMix', (-200, 300), data_type='RGBA', blend_type='MULTIPLY')
        mix.inputs['Factor'].default_value = 1.0
        nt.links.new(col.outputs['Color'], mix.inputs['A'])
        mix.inputs['B'].default_value = tuple(tint) + (1,)
        nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    else:
        nt.links.new(col.outputs['Color'], b.inputs['Base Color'])
    r = tex('Roughness', 'Non-Color', (-500, 0))
    if r:
        mr = N(nt, 'ShaderNodeMath', (-200, 0), operation='MULTIPLY')
        nt.links.new(r.outputs['Color'], mr.inputs[0])
        mr.inputs[1].default_value = roughness_scale
        nt.links.new(mr.outputs[0], b.inputs['Roughness'])
    mt = tex('Metalness', 'Non-Color', (-500, -250))
    if mt:
        nt.links.new(mt.outputs['Color'], b.inputs['Metallic'])
    nm = tex('NormalGL', 'Non-Color', (-500, -500))
    if nm and normal > 0:
        nmap = N(nt, 'ShaderNodeNormalMap', (-200, -500))
        nmap.inputs['Strength'].default_value = normal
        nt.links.new(nm.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], b.inputs['Normal'])
    _PBR[key] = m
    return m


def flat(name, col, rough=0.6, metal=0.0, spec=0.5):
    if name in _PBR:
        return _PBR[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = tuple(col) + (1,)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    b.inputs['Specular IOR Level'].default_value = spec
    _PBR[name] = m
    return m


# ------------------------------------------------------------------------------------ buildings
FACADES = {
    # name: (tile width m, tile height m), what it looks like
    'Facade006': (24.0, 24.0),     # white frame, green-grey glass: 8 storeys of 3 m per tile
    'Facade001': (32.0, 32.0),     # dark glass curtain wall
    'Facade005': (40.0, 40.0),     # fine blue-green glass grid
    'Facade018A': (22.4, 22.4),    # brick, punched windows
    'Facade019A': (17.0, 17.0),    # concrete and dark brick piers
    'Facade020A': (17.0, 17.0),    # brick piers, light spandrels
}


def box_mesh(name, w, d, h, roof_mat_index=1):
    """A w x d x h box in metres, origin at the centre of its base; the top face uses material slot 1."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * w, v.co.y * d, (v.co.z + 0.5) * h))
    for f in bm.faces:
        f.material_index = roof_mat_index if f.normal.z > 0.5 else 0
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


ROOF_TINTS = [(0.75, 0.75, 0.75), (0.55, 0.32, 0.24), (0.38, 0.38, 0.40), (0.62, 0.58, 0.52)]


def building(x, y, w, d, h, facade='Facade006', rot=0.0, roof=True):
    ob = box_mesh('Bldg', w, d, h)
    tw, th = FACADES[facade]
    ob.data.materials.append(pbr(facade, tw, th, roughness_scale=0.9))
    tint = ROOF_TINTS[int(abs(x * 3 + y * 7)) % len(ROOF_TINTS)]
    ob.data.materials.append(pbr('Concrete049', 6.0, tint=tint))
    ob.location = (x, y, 0)
    ob.rotation_euler = (0, 0, rot)
    if roof and h > 12:                               # plant rooms and a parapet line on the roof
        r = np.random.default_rng(int(abs(x * 7 + y * 13)) % 100000)
        for k in range(r.integers(1, 4)):
            pw, pd = r.uniform(0.15, 0.35) * w, r.uniform(0.15, 0.35) * d
            px, py = r.uniform(-0.3, 0.3) * w, r.uniform(-0.3, 0.3) * d
            pr = box_mesh('Plant', pw, pd, r.uniform(2.0, 4.0))
            pr.data.materials.append(flat('PlantRoom', (0.42, 0.42, 0.41), 0.7))
            pr.data.materials.append(flat('PlantRoom', (0.42, 0.42, 0.41), 0.7))
            pr.parent = ob
            pr.location = (px, py, h)
    return ob


def office_block(x, y, w=26.0, h=60.0):
    """A 20-storey office block: white frame and glass (3 m storeys), a dark lobby, roof plant."""
    ob = building(x, y, w, w, h, 'Facade006')
    lob = box_mesh('Lobby', w + 0.3, w + 0.3, 4.5)
    lob.data.materials.append(flat('LobbyGlass', (0.02, 0.025, 0.03), 0.05, 0.6))
    lob.data.materials.append(flat('LobbyGlass', (0.02, 0.025, 0.03), 0.05, 0.6))
    lob.location = (x, y, 0)
    return ob


def supertall(x, y, h=828.0):
    """A Y-plan supertall like the world's tallest building: three wings around a hexagonal core that step
    back in a spiral, a glass and steel skin, and a spire to 828 m."""
    glass = super_skin()
    steel = flat('SpireSteel', (0.75, 0.76, 0.78), 0.25, 1.0)
    occupied = 0.72 * h                                # ~600 m of floors, then the spire
    tiers = 26
    for k in range(tiers):
        z0, z1 = occupied * k / tiers, occupied * (k + 1) / tiers
        core = 30.0 * (1 - 0.55 * k / tiers)
        for wing in range(3):
            # each wing steps back in turn, so the setbacks spiral upwards
            step = (k * 3 + wing) / (tiers * 3)
            L = 82.0 * (1.0 - step) ** 1.15
            if L < core * 0.9:
                continue
            a = math.radians(90 + 120 * wing)
            ww = 30.0 * (1 - 0.35 * step)
            c = Vector((math.cos(a), math.sin(a), 0)) * (L / 2)
            ob = box_mesh('Wing', L, ww, z1 - z0)
            ob.data.materials.append(glass)
            ob.data.materials.append(glass)
            ob.location = (x + c.x, y + c.y, z0)
            ob.rotation_euler = (0, 0, a)
            cap = look_cyl('WingEnd', ww / 2, z1 - z0, glass)
            cap.location = (x + math.cos(a) * L, y + math.sin(a) * L, z0)
        hexo = look_cyl('Core', core, z1 - z0, glass, seg=6)
        hexo.location = (x, y, z0)
    # the spire
    sp = look_cyl('Spire', 9.0, h - occupied, steel, seg=16, top=0.08)
    sp.location = (x, y, occupied)


def super_skin():
    """Reflective glass with silver aluminium fins, from the Facade005 grid."""
    if 'SuperSkin' in _PBR:
        return _PBR['SuperSkin']
    m = pbr('Facade005', 10.0, 4.2 * 4, key='SuperSkinBase')
    m = m.copy()
    m.name = 'SuperSkin'
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    col = [n for n in nt.nodes if n.type == 'TEX_IMAGE' and 'Color' in n.image.name][0]
    bw = N(nt, 'ShaderNodeRGBToBW', (-250, 450))
    nt.links.new(col.outputs['Color'], bw.inputs['Color'])
    ramp = N(nt, 'ShaderNodeValToRGB', (-100, 450))
    ramp.color_ramp.elements[0].color = (0.22, 0.27, 0.32, 1)        # blue-grey glass
    ramp.color_ramp.elements[1].color = (0.75, 0.77, 0.80, 1)        # aluminium
    nt.links.new(bw.outputs['Val'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    for l in list(b.inputs['Metallic'].links) + list(b.inputs['Roughness'].links):
        nt.links.remove(l)
    b.inputs['Metallic'].default_value = 0.85
    b.inputs['Roughness'].default_value = 0.18
    _PBR['SuperSkin'] = m
    return m


def look_cyl(name, r, hgt, mat, seg=24, top=1.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r * top, depth=hgt)
    for v in bm.verts:
        v.co.z += hgt / 2
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def city(cx, cy, radius, rng, avoid=(), tall_frac=0.06, pitch=95.0):
    """Blocks on a jittered street grid: brick and concrete mid-rises, glass towers near the middle."""
    mids = ['Facade018A', 'Facade019A', 'Facade020A', 'Facade006']
    towers = ['Facade001', 'Facade005', 'Facade006']
    k = 0
    n = int(radius / pitch)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = cx + i * pitch + rng.uniform(-8, 8), cy + j * pitch + rng.uniform(-8, 8)
            r = math.hypot(x - cx, y - cy)
            if r > radius * rng.uniform(0.8, 1.0):
                continue
            if any(math.hypot(x - ax, y - ay) < ar for ax, ay, ar in avoid):
                continue
            tall = rng.random() < tall_frac * math.exp(-r / (radius * 0.4))
            if tall:
                h = rng.uniform(120, 330)
                w, d = rng.uniform(28, 45), rng.uniform(28, 45)
                fac = towers[rng.integers(len(towers))]
            else:
                h = float(np.clip(rng.lognormal(math.log(24), 0.45), 10, 90))
                w, d = rng.uniform(35, 70), rng.uniform(35, 70)
                fac = mids[rng.integers(len(mids))]
            building(x, y, w, d, h, fac)
            k += 1
    return k


def suburb(cx, cy, w, d, rng, pitch=60.0):
    """Low buildings (two to five storeys) on a grid inside a w x d rectangle."""
    fac = ['Facade018A', 'Facade019A', 'Facade020A']
    for i in range(int(w / pitch)):
        if i % 5 == 4:                                  # avenues
            continue
        for j in range(int(d / pitch)):
            if j % 4 == 3 or rng.random() < 0.2:
                continue
            x = cx - w / 2 + (i + 0.5) * pitch + rng.uniform(-8, 8)
            y = cy - d / 2 + (j + 0.5) * pitch + rng.uniform(-8, 8)
            building(x, y, rng.uniform(16, 48), rng.uniform(16, 48), float(rng.choice([7, 10, 13, 17, 22, 30])),
                     fac[rng.integers(3)], rot=rng.uniform(-0.05, 0.05), roof=False)


def ground_pbr(size, name, tile, tint=None):
    me = bpy.data.meshes.new('Ground')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('Ground', me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(pbr(name, tile, tint=tint, box=False))
    return ob


# ------------------------------------------------------------------------------------ trees
_TREES = {}


def tree_collection(name):
    """Append a Poly Haven tree once, into a hidden collection we can instance."""
    if name in _TREES:
        return _TREES[name]
    path = f'{A}/models/{name}/{name}_2k.blend'
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    col = bpy.data.collections.new(f'TREE_{name}')
    for ob in dst.objects:
        if ob is not None and ob.type == 'MESH':
            col.objects.link(ob)
    for im in bpy.data.images:
        if im.filepath.startswith('//textures'):
            im.filepath = f'{A}/models/{name}/' + im.filepath[2:]
    zmin = min((ob.matrix_world @ Vector(c)).z for ob in col.objects for c in ob.bound_box)
    zmax = max((ob.matrix_world @ Vector(c)).z for ob in col.objects for c in ob.bound_box)
    col['height'] = zmax - zmin
    col.instance_offset = (0, 0, zmin)
    _TREES[name] = col
    return col


def tree(x, y, height, name='tree_small_02', rot=0.0):
    col = tree_collection(name)
    e = bpy.data.objects.new(f'Tree_{name}', None)
    e.instance_type = 'COLLECTION'
    e.instance_collection = col
    s = height / col['height']
    e.scale = (s, s, s)
    e.location = (x, y, 0)
    e.rotation_euler = (0, 0, rot)
    bpy.context.collection.objects.link(e)
    return e


# ------------------------------------------------------------------------------------ people
ARMS = {20, 21, 11, 12, 9, 10} | set(range(64, 104))
HANDS = {9, 10} | set(range(64, 104))
SKIN_SETS = {17, 2, 3, 4, 5, 7, 8, 22, 9, 10} | set(range(64, 104))
SHIRT_SETS = {1, 20, 21, 11, 12}
TROUSER_SETS = {19, 18, 23, 24, 15, 16}
SHOE_SETS = {13, 14} | set(range(25, 64))

OUTFITS = [
    dict(shirt=(0.06, 0.12, 0.30), trousers=(0.03, 0.035, 0.05), shoes=(0.02, 0.015, 0.012), skin=(0.60, 0.40, 0.29),
         hair=(0.035, 0.025, 0.018)),
    dict(shirt=(0.45, 0.07, 0.06), trousers=(0.20, 0.17, 0.12), shoes=(0.05, 0.04, 0.03), skin=(0.36, 0.22, 0.14),
         hair=(0.015, 0.012, 0.01)),
    dict(shirt=(0.55, 0.55, 0.52), trousers=(0.05, 0.08, 0.16), shoes=(0.6, 0.6, 0.6), skin=(0.75, 0.55, 0.43),
         hair=(0.25, 0.15, 0.07)),
    dict(shirt=(0.10, 0.22, 0.12), trousers=(0.03, 0.03, 0.03), shoes=(0.03, 0.03, 0.03), skin=(0.28, 0.17, 0.10),
         hair=(0.01, 0.01, 0.01)),
]


def _cloth(name, col, rough=0.85, sheen=0.4, bump_scale=900.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = tuple(col) + (1,)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Sheen Weight'].default_value = sheen
    b.inputs['Specular IOR Level'].default_value = 0.3
    tc = N(nt, 'ShaderNodeTexCoord', (-700, -200))
    nz = N(nt, 'ShaderNodeTexNoise', (-500, -200))
    nz.inputs['Scale'].default_value = bump_scale
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    bump = N(nt, 'ShaderNodeBump', (-250, -200))
    bump.inputs['Strength'].default_value = 0.15
    nt.links.new(nz.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def _skin(name, col):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = tuple(col) + (1,)
    b.inputs['Roughness'].default_value = 0.5
    b.inputs['Subsurface Weight'].default_value = 0.15
    b.inputs['Subsurface Radius'].default_value = (0.012, 0.005, 0.003)
    b.inputs['Specular IOR Level'].default_value = 0.4
    return m


def _region_copy(src_me, faces, name, offset, smooth_iters, mat, subdiv=1):
    """A garment: the body's faces in `faces`, pushed out along the normals and smoothed (no muscles)."""
    bm = bmesh.new()
    bm.from_mesh(src_me)
    drop = [f for f in bm.faces if f.index not in faces]
    bmesh.ops.delete(bm, geom=drop, context='FACES')
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    me.materials.append(mat)
    if smooth_iters:
        sm = ob.modifiers.new('Smooth', 'SMOOTH')
        sm.factor = 0.8
        sm.iterations = smooth_iters
    disp = ob.modifiers.new('Out', 'DISPLACE')
    disp.mid_level = 0.0
    disp.strength = offset
    if subdiv:
        sd = ob.modifiers.new('Sub', 'SUBSURF')
        sd.levels = 1
        sd.render_levels = subdiv
    sol = ob.modifiers.new('Thick', 'SOLIDIFY')
    sol.thickness = 0.004
    sol.offset = -1.0
    return ob


def human(x, y, height=1.75, face_deg=0.0, kind='male', outfit=0, arms_down_deg=13.0):
    """A dressed adult standing at (x, y), facing face_deg (0 = facing -y)."""
    with bpy.data.libraries.load(HUMAN_BLEND, link=False) as (src, dst):
        dst.objects = [f'GEO-body_{kind}_realistic', f'GEO-body_{kind}_realistic.eye.L', f'GEO-body_{kind}_realistic.eye.R']
    body, eyeL, eyeR = dst.objects
    root = bpy.data.objects.new('Person', None)
    bpy.context.collection.objects.link(root)
    for ob in (body, eyeL, eyeR):
        bpy.context.collection.objects.link(ob)
    body.location = (0, 0, 0)                     # the bundle lays its bodies out side by side
    body.rotation_euler = (0, 0, 0)
    body.parent = root                            # the eyes stay parented to the body, as in the bundle
    me = body.data
    # --- lower the arms from the A-pose (rotate about the shoulders, blended near the joint)
    fs = np.zeros(len(me.polygons), int)
    me.attributes['.sculpt_face_set'].data.foreach_get('value', fs)
    arm_v = set()
    for p, f in zip(me.polygons, fs):
        if f in ARMS:
            arm_v.update(p.vertices)
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    top = co[:, 2].max()
    for side in (1, -1):
        sv = np.array([i for i in arm_v if co[i, 0] * side > 0.05])
        piv = np.array([side * 0.155 * top / 1.69, 0.0, 1.365 * top / 1.69])
        d = co[sv] - piv
        dist = np.linalg.norm(d, axis=1)
        w = np.clip((dist - 0.02) / 0.10, 0, 1)
        w = w * w * (3 - 2 * w)
        ang = np.radians(arms_down_deg) * w * side
        c, s = np.cos(ang), np.sin(ang)
        nx = d[:, 0] * c + d[:, 2] * s
        nz = -d[:, 0] * s + d[:, 2] * c
        co[sv, 0] = piv[0] + nx
        co[sv, 2] = piv[2] + nz
    me.vertices.foreach_set('co', co.ravel())
    me.update()
    for m in body.modifiers:
        if m.type == 'MULTIRES':
            m.render_levels = 2
            m.levels = 1
    o = OUTFITS[outfit % len(OUTFITS)]
    me.materials.clear()
    me.materials.append(_skin(f'Skin{outfit}', o['skin']))
    eye = flat('Eye', (0.10, 0.08, 0.07), 0.08)
    for e in (eyeL, eyeR):
        e.data.materials.clear()
        e.data.materials.append(eye)
    # --- clothes and hair: bands of the body by height (hands and head excluded), smoothed and pushed out
    faces = lambda sets: {i for i, f in enumerate(fs) if f in sets}
    zc = np.array([p.center.z for p in me.polygons]) / top
    hands = HANDS
    headset = {17, 2, 3, 4, 5, 7, 8, 22}
    shirt = {i for i in range(len(fs)) if 0.565 < zc[i] < 0.86 and fs[i] not in hands and fs[i] not in headset}
    trousers = {i for i in range(len(fs)) if 0.05 < zc[i] < 0.61 and fs[i] not in ARMS}
    shoes = {i for i in range(len(fs)) if zc[i] <= 0.06 and fs[i] not in ARMS}
    _region_copy(me, shirt, 'Shirt', 0.012, 12, _cloth(f'Shirt{outfit}', o['shirt']), 2).parent = body
    _region_copy(me, trousers, 'Trousers', 0.014, 12, _cloth(f'Trousers{outfit}', o['trousers'], 0.8, 0.2), 2).parent = body
    _region_copy(me, shoes, 'Shoes', 0.012, 6, flat(f'Shoes{outfit}', o['shoes'], 0.35), 1).parent = body
    # hide the body under the clothes (else its sculpted muscles poke through); keep 2 rings of overlap
    covered = shirt | trousers | shoes
    keep_v = set()
    for i, poly in enumerate(me.polygons):
        if i not in covered:
            keep_v.update(poly.vertices)
    for _ in range(2):
        grow = set()
        for poly in me.polygons:
            pv = set(poly.vertices)
            if pv & keep_v:
                grow |= pv
        keep_v |= grow
    vg = body.vertex_groups.new(name='skin')
    vg.add(sorted(keep_v), 1.0, 'REPLACE')
    mask = body.modifiers.new('HideCovered', 'MASK')
    mask.vertex_group = 'skin'
    head = faces({17})
    hair = set()
    for i in head:
        c = me.polygons[i].center
        if c.z > top - 0.085 or (c.z > top - 0.19 and c.y > -0.01):
            hair.add(i)
    hob = _region_copy(me, hair, 'Hair', 0.012, 4, _cloth(f'Hair{outfit}', o['hair'], 0.55, 0.6, 3000.0), 1)
    hob.parent = body
    tx = bpy.data.textures.get('HairTex') or bpy.data.textures.new('HairTex', 'CLOUDS')
    tx.noise_scale = 0.02
    hd = hob.modifiers.new('Tufts', 'DISPLACE')
    hd.texture = tx
    hd.strength = 0.008
    hd.mid_level = 0.3
    hob.modifiers.move(len(hob.modifiers) - 1, 1)
    s = height / top
    root.scale = (s, s, s)
    root.location = (x, y, 0)
    root.rotation_euler = (0, 0, math.radians(face_deg))
    return root


# ------------------------------------------------------------------------------------ Everest (real terrain)
EVEREST = (27.98903, 86.92569)          # summit in the DEM (its 30 m cells read 8,738 m; true 8,849 m)
R_E = 6371000.0


_DEM = []


def dem():
    if not _DEM:
        d = np.load(f'{A}/dem/everest.npz')
        _DEM.append((d['h'], d['lat'], d['lon']))
    return _DEM[0]


def enu(lat, lon):
    lat0, lon0 = EVEREST
    x = (lon - lon0) * math.radians(1) * R_E * math.cos(math.radians(lat0))
    y = (lat - lat0) * math.radians(1) * R_E
    return x, y


def terrain_mesh(name, step, box_km, hole_km=None, drop=0.0, centre=(0.0, 0.0)):
    """The DEM as a mesh (metres, origin at the summit, Earth's curvature included), every `step` cells,
    inside +-box_km around the summit; optionally with a lowered middle where a finer mesh sits on top."""
    h, lat, lon = dem()
    x, _ = enu(EVEREST[0], lon)
    _, y = enu(lat, EVEREST[1])
    cx, cy = centre
    xs = np.where(np.abs(x - cx) <= box_km * 1000)[0][::step]
    ys = np.where(np.abs(y - cy) <= box_km * 1000)[0][::step]
    X, Y = np.meshgrid(x[xs], y[ys])
    Z = h[np.ix_(ys, xs)].astype(np.float64)
    # the 30 m cells shave the summit: restore the surveyed 8,849 m with a small bump at the top
    r2 = X ** 2 + Y ** 2
    Z += 111.0 * np.exp(-r2 / (2 * 70.0 ** 2))
    Z -= r2 / (2 * R_E)
    if hole_km:
        inside = (np.abs(X - cx) < hole_km * 1000) & (np.abs(Y - cy) < hole_km * 1000)
        Z[inside] -= drop
    ny, nx = Z.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    ii = np.arange(ny * nx).reshape(ny, nx)
    a, b_, c, d = ii[:-1, :-1].ravel(), ii[:-1, 1:].ravel(), ii[1:, 1:].ravel(), ii[1:, :-1].ravel()
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set('co', verts.astype(np.float32).ravel())
    nq = len(a)
    me.loops.add(nq * 4)
    me.loops.foreach_set('vertex_index', np.stack([a, d, c, b_], 1).ravel().astype(np.int32))
    me.polygons.add(nq)
    me.polygons.foreach_set('loop_start', (np.arange(nq) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', np.ones(nq, bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def terrain_height(xm, ym):
    """DEM height (m) at local x, y (bilinear)."""
    h, lat, lon = dem()
    lat0, lon0 = EVEREST
    la = lat0 + ym / (math.radians(1) * R_E)
    lo = lon0 + xm / (math.radians(1) * R_E * math.cos(math.radians(lat0)))
    fi = (lat[0] - la) / (lat[0] - lat[1])
    fj = (lo - lon[0]) / (lon[1] - lon[0])
    i0, j0 = int(fi), int(fj)
    u, v = fi - i0, fj - j0
    return float(h[i0, j0] * (1 - u) * (1 - v) + h[i0 + 1, j0] * u * (1 - v) + h[i0, j0 + 1] * (1 - u) * v
                 + h[i0 + 1, j0 + 1] * u * v)


def himalaya_material():
    """Rock on steep faces, snow and ice above ~5.3 km where slopes hold it, debris-covered glacier tongues,
    brown alpine slopes, green forest low down in the south."""
    m = bpy.data.materials.new('Himalaya')
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    geo = N(nt, 'ShaderNodeNewGeometry', (-1500, 0))
    sep = N(nt, 'ShaderNodeSeparateXYZ', (-1300, 0))
    nt.links.new(geo.outputs['Position'], sep.inputs[0])
    sepn = N(nt, 'ShaderNodeSeparateXYZ', (-1300, -300))
    nt.links.new(geo.outputs['Normal'], sepn.inputs[0])
    n1 = N(nt, 'ShaderNodeTexNoise', (-1300, 400))
    n1.inputs['Scale'].default_value = 0.0009
    n1.inputs['Detail'].default_value = 10.0
    n1.inputs['Roughness'].default_value = 0.6
    nt.links.new(geo.outputs['Position'], n1.inputs['Vector'])
    n2 = N(nt, 'ShaderNodeTexNoise', (-1300, 700))
    n2.inputs['Scale'].default_value = 0.012
    n2.inputs['Detail'].default_value = 8.0
    nt.links.new(geo.outputs['Position'], n2.inputs['Vector'])

    def mr(src, a, b_, loc, clamp=True):
        r = N(nt, 'ShaderNodeMapRange', loc)
        nt.links.new(src, r.inputs['Value'])
        r.inputs['From Min'].default_value = a
        r.inputs['From Max'].default_value = b_
        r.clamp = clamp
        return r.outputs['Result']

    def math_(op, a, b_, loc):
        n = N(nt, 'ShaderNodeMath', loc, operation=op)
        for k, v in enumerate((a, b_)):
            if isinstance(v, (int, float)):
                n.inputs[k].default_value = v
            else:
                nt.links.new(v, n.inputs[k])
        return n.outputs[0]
    # jitter the height a little so lines (snowline, treeline) are ragged
    zj = math_('MULTIPLY_ADD', n1.outputs['Fac'], 700.0, (-1100, 200))
    zj = math_('ADD', zj, sep.outputs['Z'], (-950, 200))
    flat_ = sepn.outputs['Z']                                     # 1 flat, 0 vertical
    snow_h = mr(zj, 5300.0, 6000.0, (-750, 300))
    snow_s = mr(flat_, 0.58, 0.70, (-750, 100))                   # snow holds on slopes under ~50 degrees
    n4 = N(nt, 'ShaderNodeTexNoise', (-1300, 1000))               # fluting and wind-scoured patches
    n4.inputs['Scale'].default_value = 0.004
    n4.inputs['Detail'].default_value = 12.0
    n4.inputs['Roughness'].default_value = 0.7
    nt.links.new(geo.outputs['Position'], n4.inputs['Vector'])
    patch = mr(n4.outputs['Fac'], 0.36, 0.48, (-750, 500))
    snow = math_('MULTIPLY', math_('MULTIPLY', snow_h, snow_s, (-600, 250)), patch, (-550, 200))
    glac = math_('MULTIPLY', mr(flat_, 0.93, 0.985, (-750, -100)), mr(zj, 4700.0, 5200.0, (-750, -300)), (-550, -100))
    debris = math_('MULTIPLY', mr(flat_, 0.94, 0.99, (-750, -500)),
                   math_('SUBTRACT', 1.0, mr(zj, 5000.0, 5500.0, (-750, -700)), (-600, -650)), (-550, -500))
    debris = math_('MULTIPLY', debris, mr(zj, 4300.0, 4700.0, (-750, -900)), (-450, -600))
    veg = math_('SUBTRACT', 1.0, mr(zj, 3300.0, 4200.0, (-750, -1100)), (-550, -1100))

    def mix(fac, a, b_, loc):
        n = N(nt, 'ShaderNodeMix', loc, data_type='RGBA')
        for sock, v in (('A', a), ('B', b_)):
            if isinstance(v, tuple):
                n.inputs[sock].default_value = v + (1,)
            else:
                nt.links.new(v, n.inputs[sock])
        if isinstance(fac, float):
            n.inputs['Factor'].default_value = fac
        else:
            nt.links.new(fac, n.inputs['Factor'])
        return n.outputs['Result']
    rock = mix(n2.outputs['Fac'], (0.035, 0.032, 0.030), (0.12, 0.105, 0.09), (-300, 600))
    rock = mix(mr(n1.outputs['Fac'], 0.4, 0.7, (-500, 800)), rock, (0.10, 0.085, 0.07), (-150, 600))
    ground = mix(veg, rock, (0.05, 0.075, 0.03), (0, 450))
    alpine = mix(mr(zj, 4200.0, 4800.0, (-300, 300)), (0.11, 0.095, 0.07), ground, (100, 350))
    ground = mix(math_('MULTIPLY', mr(flat_, 0.7, 0.9, (-300, 100)), math_('SUBTRACT', 1.0, veg, (-300, -50)), (-150, 100)),
                 ground, alpine, (200, 250))
    ground = mix(debris, ground, (0.16, 0.15, 0.14), (350, 100))
    ground = mix(glac, ground, (0.46, 0.49, 0.53), (500, 0))
    col = mix(snow, ground, (0.86, 0.88, 0.93), (650, 0))
    nt.links.new(col, b.inputs['Base Color'])
    rough = mix(snow, (0.9, 0.9, 0.9), (0.45, 0.45, 0.45), (650, -250))
    sr = N(nt, 'ShaderNodeSeparateColor', (800, -250))
    nt.links.new(rough, sr.inputs[0])
    nt.links.new(sr.outputs[0], b.inputs['Roughness'])
    b.inputs['Specular IOR Level'].default_value = 0.3
    # fine relief the 30 m grid cannot hold
    n3 = N(nt, 'ShaderNodeTexNoise', (-400, -1400))
    n3.inputs['Scale'].default_value = 0.02
    n3.inputs['Detail'].default_value = 12.0
    n3.inputs['Roughness'].default_value = 0.65
    nt.links.new(geo.outputs['Position'], n3.inputs['Vector'])
    bump = N(nt, 'ShaderNodeBump', (650, -500))
    bump.inputs['Strength'].default_value = 0.35
    bump.inputs['Distance'].default_value = 25.0
    nt.links.new(n3.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m
