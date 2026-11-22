"""Scale cards: each impactor as a real-looking, slowly turning asteroid next to something familiar.

S1_1m     a 1 m boulder beside a person (an architectural scale figure), on grass
S2_60m    a 60 m asteroid beside a 20-storey office block, on a city plaza
S3_1km    a 1 km asteroid beside an 828 m supertall tower and its neighbours
S4_10km   a 10 km asteroid beside an 8.8 km Himalayan peak
S5_100km  a 100 km asteroid standing on the curved Earth (it pokes out of the atmosphere), with a 10 km
          one beside it

Units: metres for S1-S4 (Nishita sky, sun lamp, mist haze); 1 unit = 1,000 km for S5 (look.py's Earth).
Each frame also records where the labels go (screen fractions), for tv.py.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector, Euler, Quaternion, Matrix, noise
from bpy_extras.object_utils import world_to_camera_view

import look
import shots
from shots import shot, smooth, lerp, place_camera

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------ the asteroid
STYLES = {
    # cuts: flat fracture faces; craters; boulders per diameter; elongation (x, y, z); lumpiness
    'boulder': dict(cuts=16, craters=0, boulders=0, shape=(1.0, 0.90, 0.80), lumpy=0.10, crater_depth=0.0, top=0.0),
    'rubble': dict(cuts=4, craters=5, boulders=22, shape=(1.0, 0.80, 0.72), lumpy=0.16, crater_depth=0.18, top=0.0),
    'top': dict(cuts=0, craters=10, boulders=28, shape=(1.0, 1.0, 0.90), lumpy=0.05, crater_depth=0.18, top=0.16),
    'eros': dict(cuts=2, craters=30, boulders=40, shape=(1.0, 0.72, 0.64), lumpy=0.16, crater_depth=0.22, top=0.0),
    'big': dict(cuts=2, craters=34, boulders=0, shape=(1.0, 0.90, 0.82), lumpy=0.12, crater_depth=0.22, top=0.0),
}


def asteroid_mesh(name, seed=1, subdiv=6, style='big'):
    """A unit-diameter (mean) rock: fractured facets, potato shape, nested craters, boulders."""
    st = STYLES[style]
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=0.5)
    off = Vector(rng.uniform(-50, 50, 3))
    cr = []
    for k in range(st['craters']):
        c = Vector(rng.normal(size=3)).normalized()
        r = 0.04 + 0.30 * rng.random() ** 3
        cr.append((c, r))
    cuts = []
    for k in range(st['cuts']):
        n = Vector(rng.normal(size=3)).normalized()
        cuts.append((n, rng.uniform(0.30, 0.45)))
    sx, sy, sz = st['shape']
    nb = st['boulders']
    for v in bm.verts:
        p = v.co.normalized()
        h = st['lumpy'] * noise.fractal(p * 1.3 + off, 0.9, 2.0, 3)
        h += 0.05 * noise.fractal(p * 4.0 + off * 2, 0.7, 2.0, 4)
        h += 0.010 * noise.fractal(p * 18.0 + off * 3, 0.6, 2.0, 3)
        for c, r in cr:
            d = (p - c).length / r
            if d < 1.0:
                h += -st['crater_depth'] * r * (1 - d * d)
            elif d < 1.7:
                h += 0.09 * r * math.exp(-((d - 1.0) / 0.25) ** 2)
        if nb:
            dist, pts = noise.voronoi(p * nb + off, distance_metric='DISTANCE', exponent=2.5)
            rb = 0.42
            if dist[0] < rb:
                h += (0.55 / nb) * math.sqrt(max(0.0, 1 - (dist[0] / rb) ** 2))
        h += st['top'] * ((1 - abs(p.z)) ** 2 - 0.45)          # a spinning-top equatorial ridge
        q = Vector((p.x * sx, p.y * sy, p.z * sz)) * 0.5 * (1 + h)
        for n, dd in cuts:                       # fracture faces: clip to random planes
            e = q.dot(n) - dd
            if e > 0:
                q -= n * e * 0.92
        v.co = q
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for f in me.polygons:
        f.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(rock_material(name + 'Mat', seed=seed))
    co = np.array([v.co for v in me.vertices])
    dia = (co.max(0) - co.min(0)).mean()            # mean of length, width and height
    ob.scale = (1 / dia,) * 3
    bpy.context.view_layer.update()
    return ob


def rock_material(name, base=(0.10, 0.092, 0.082), seed=0):
    """Stony asteroid: dark grey-brown (albedo ~0.1-0.15), mottled, rough; crevices darker; fine bump."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    N = look.node
    tc = N(nt, 'ShaderNodeTexCoord', (-1100, 0))
    mp = N(nt, 'ShaderNodeMapping', (-950, 0))
    mp.inputs['Location'].default_value = (seed * 3.1, seed * 1.7, seed * 2.3)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    n1 = N(nt, 'ShaderNodeTexNoise', (-700, 250))
    n1.inputs['Scale'].default_value = 5.0
    n1.inputs['Detail'].default_value = 12.0
    n1.inputs['Roughness'].default_value = 0.65
    nt.links.new(mp.outputs['Vector'], n1.inputs['Vector'])
    ramp = N(nt, 'ShaderNodeValToRGB', (-450, 250))
    ramp.color_ramp.elements[0].position = 0.3
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[0].color = (base[0] * 0.45, base[1] * 0.45, base[2] * 0.45, 1)
    ramp.color_ramp.elements[1].color = (base[0] * 1.9, base[1] * 1.8, base[2] * 1.65, 1)
    nt.links.new(n1.outputs['Fac'], ramp.inputs['Fac'])
    geo = N(nt, 'ShaderNodeNewGeometry', (-700, -50))
    cav = N(nt, 'ShaderNodeMapRange', (-450, -50))
    nt.links.new(geo.outputs['Pointiness'], cav.inputs['Value'])
    cav.inputs['From Min'].default_value = 0.45
    cav.inputs['From Max'].default_value = 0.55
    cav.inputs['To Min'].default_value = 0.55
    cav.inputs['To Max'].default_value = 1.15
    mul = N(nt, 'ShaderNodeMix', (-200, 150), data_type='RGBA', blend_type='MULTIPLY')
    mul.inputs['Factor'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], mul.inputs['A'])
    nt.links.new(cav.outputs['Result'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.93
    bsdf.inputs['Specular IOR Level'].default_value = 0.25
    n2 = N(nt, 'ShaderNodeTexNoise', (-700, -300))
    n2.inputs['Scale'].default_value = 70.0
    n2.inputs['Detail'].default_value = 10.0
    n2.inputs['Roughness'].default_value = 0.7
    nt.links.new(mp.outputs['Vector'], n2.inputs['Vector'])
    vo = N(nt, 'ShaderNodeTexVoronoi', (-700, -550))
    vo.inputs['Scale'].default_value = 35.0
    nt.links.new(mp.outputs['Vector'], vo.inputs['Vector'])
    hb = N(nt, 'ShaderNodeMath', (-450, -400), operation='MULTIPLY_ADD')
    nt.links.new(vo.outputs['Distance'], hb.inputs[0])
    hb.inputs[1].default_value = 0.6
    nt.links.new(n2.outputs['Fac'], hb.inputs[2])
    bump = N(nt, 'ShaderNodeBump', (-250, -350))
    bump.inputs['Strength'].default_value = 0.7
    bump.inputs['Distance'].default_value = 0.004
    nt.links.new(hb.outputs[0], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return m


# ------------------------------------------------------------------------------------ environments
def sky(sc, elev_deg=28.0, rot_deg=40.0, strength=0.35):
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    bg = nt.nodes.new('ShaderNodeBackground')
    s = nt.nodes.new('ShaderNodeTexSky')
    s.sky_type = 'NISHITA'
    s.sun_disc = False
    s.sun_elevation = math.radians(elev_deg)
    s.sun_rotation = math.radians(rot_deg)
    s.altitude = 100.0
    s.air_density = 1.0
    s.dust_density = 1.5
    nt.links.new(s.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = strength
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    # matching sun lamp
    el, rot = math.radians(elev_deg), math.radians(rot_deg)
    d = Vector((math.sin(rot) * math.cos(el), -math.cos(rot) * math.cos(el), math.sin(el)))
    lamp = look.sun(tuple(d), 4.0)
    lamp.data.angle = math.radians(1.0)
    sc.view_settings.exposure = -0.7
    return lamp


def ground(size, kind='grass'):
    me = bpy.data.meshes.new('Ground')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new('Ground', me)
    bpy.context.collection.objects.link(ob)
    m = bpy.data.materials.new('Ground' + kind)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    N = look.node
    tc = N(nt, 'ShaderNodeTexCoord', (-900, 0))
    nz = N(nt, 'ShaderNodeTexNoise', (-650, 100))
    nz.inputs['Scale'].default_value = {'grass': 0.8, 'plaza': 0.05, 'city': 0.004}[kind]
    nz.inputs['Detail'].default_value = 12.0
    nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
    ramp = N(nt, 'ShaderNodeValToRGB', (-400, 100))
    cols = {'grass': ((0.06, 0.10, 0.03), (0.17, 0.22, 0.07)), 'plaza': ((0.70, 0.70, 0.68), (1.0, 0.98, 0.95)),
            'city': ((0.10, 0.10, 0.10), (0.24, 0.23, 0.21))}[kind]
    ramp.color_ramp.elements[0].color = cols[0] + (1,)
    ramp.color_ramp.elements[1].color = cols[1] + (1,)
    nt.links.new(nz.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.95
    bsdf.inputs['Specular IOR Level'].default_value = 0.12
    if kind == 'plaza':                                        # paving slabs, 0.6 x 0.3 m, varied tone
        br = N(nt, 'ShaderNodeTexBrick', (-650, -200))
        br.inputs['Scale'].default_value = 1.0
        br.inputs['Brick Width'].default_value = 0.6
        br.inputs['Row Height'].default_value = 0.3
        br.inputs['Mortar Size'].default_value = 0.008
        br.inputs['Color1'].default_value = (0.30, 0.29, 0.27, 1)
        br.inputs['Color2'].default_value = (0.42, 0.40, 0.37, 1)
        br.inputs['Mortar'].default_value = (0.12, 0.12, 0.11, 1)
        br.offset_frequency = 2
        br.squash_frequency = 2
        nt.links.new(tc.outputs['Object'], br.inputs['Vector'])
        mixc = N(nt, 'ShaderNodeMix', (-200, 100), data_type='RGBA', blend_type='MULTIPLY')
        mixc.inputs['Factor'].default_value = 0.6
        nt.links.new(br.outputs['Color'], mixc.inputs['A'])
        nt.links.new(ramp.outputs['Color'], mixc.inputs['B'])
        tone = N(nt, 'ShaderNodeMath', (-100, 250), operation='MULTIPLY')
        nt.links.new(mixc.outputs['Result'], bsdf.inputs['Base Color'])
        bump = N(nt, 'ShaderNodeBump', (-300, -200))
        bump.inputs['Strength'].default_value = 0.5
        bump.inputs['Distance'].default_value = 0.01
        nt.links.new(br.outputs['Fac'], bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    ob.data.materials.append(m)
    return ob


def matte(name, col, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = col + (1,)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    return m


def add_prim(kind, loc, scale, mat, rot=(0, 0, 0), **kw):
    bm = bmesh.new()
    if kind == 'sphere':
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
    elif kind == 'cyl':
        bmesh.ops.create_cone(bm, cap_ends=True, segments=kw.get('seg', 32), radius1=1.0, radius2=kw.get('top', 1.0), depth=2.0)
    else:
        bmesh.ops.create_cube(bm, size=2.0)
    me = bpy.data.meshes.new(kind)
    bm.to_mesh(me)
    bm.free()
    if kind != 'cube':
        for f in me.polygons:
            f.use_smooth = True
    ob = bpy.data.objects.new(kind, me)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    ob.scale = scale
    ob.rotation_euler = rot
    ob.data.materials.append(mat)
    return ob


_MATS = {}


def mat_cached(name, col, rough=0.6, metal=0.0, spec=0.5):
    if name not in _MATS:
        m = matte(name, col, rough, metal)
        m.node_tree.nodes['Principled BSDF'].inputs['Specular IOR Level'].default_value = spec
        _MATS[name] = m
    return _MATS[name]


def _mball(family, mat, res=0.012):
    mb = bpy.data.metaballs.new(family)
    mb.resolution = res
    mb.render_resolution = res * 0.5
    mb.threshold = 0.6
    ob = bpy.data.objects.new(family, mb)
    bpy.context.collection.objects.link(ob)
    mb.materials.append(mat)
    return ob, mb


def _cap(mb, a, b, r, stiff=2.0):
    """A capsule from point a to point b of radius r (metaball CAPSULE lies along local x)."""
    a, b = Vector(a), Vector(b)
    e = mb.elements.new()
    e.type = 'CAPSULE'
    e.co = (a + b) / 2
    e.size_x = (b - a).length / 2
    e.radius = r / 0.575
    e.stiffness = stiff
    e.rotation = Vector((1, 0, 0)).rotation_difference((b - a).normalized())
    return e


def _ell(mb, c, size, r, stiff=2.0):
    e = mb.elements.new()
    e.type = 'ELLIPSOID'
    e.co = Vector(c)
    e.size_x, e.size_y, e.size_z = size        # semi-axes
    e.radius = 1 / 0.575
    e.stiffness = stiff
    return e


def person(x, y, h=1.75, face=-90.0, palette=0, n=0):
    """A standing adult built from metaballs (smooth, organic): skin, hair, jacket, trousers, shoes."""
    s = h / 1.75
    jackets = [(0.10, 0.18, 0.40), (0.45, 0.07, 0.06), (0.20, 0.30, 0.18), (0.55, 0.50, 0.42)]
    skins = [(0.62, 0.42, 0.30), (0.40, 0.26, 0.17), (0.75, 0.55, 0.43), (0.30, 0.19, 0.12)]
    skin = mat_cached(f'Skin{palette}', skins[palette % 4], 0.5, spec=0.3)
    hair = mat_cached('Hair', (0.03, 0.025, 0.02), 0.45)
    jacket = mat_cached(f'Jacket{palette}', jackets[palette % 4], 0.8, spec=0.3)
    trousers = mat_cached('Trousers', (0.05, 0.06, 0.08), 0.75, spec=0.3)
    shoes = mat_cached('Shoes', (0.02, 0.02, 0.02), 0.35)
    root = bpy.data.objects.new(f'Person{n}', None)
    bpy.context.collection.objects.link(root)
    parts = []

    def P(px, py, pz):
        return (px * s, py * s, pz * s)
    ob, mb = _mball(f'FigSkin{n}x', skin)
    parts.append(ob)
    _ell(mb, P(0, 0.005, 1.625), (0.075, 0.085, 0.105), 0.02)                # head
    _cap(mb, P(0, 0, 1.47), P(0, 0, 1.53), 0.045)                               # neck
    for sg in (-1, 1):
        _ell(mb, P(sg * 0.245, -0.01, 0.80), (0.03, 0.02, 0.045), 0.012)       # hands
    ob, mb = _mball(f'FigHair{n}x', hair)
    parts.append(ob)
    _ell(mb, P(0, 0.02, 1.675), (0.080, 0.088, 0.068), 0.02)
    ob, mb = _mball(f'FigJacket{n}x', jacket)
    parts.append(ob)
    _ell(mb, P(0, 0, 1.28), (0.17, 0.10, 0.20), 0.03)                          # chest
    _ell(mb, P(0, 0, 1.05), (0.15, 0.095, 0.12), 0.03)                         # waist
    for sg in (-1, 1):
        _cap(mb, P(sg * 0.19, 0, 1.40), P(sg * 0.235, 0.0, 1.13), 0.045)       # upper arm
        _cap(mb, P(sg * 0.235, 0.0, 1.13), P(sg * 0.245, -0.03, 0.88), 0.038)  # forearm
    ob, mb = _mball(f'FigTrousers{n}x', trousers)
    parts.append(ob)
    _ell(mb, P(0, 0, 0.92), (0.16, 0.10, 0.09), 0.03)                          # hips
    for sg in (-1, 1):
        _cap(mb, P(sg * 0.09, 0, 0.88), P(sg * 0.10, 0.0, 0.50), 0.068)        # thigh
        _cap(mb, P(sg * 0.10, 0.0, 0.50), P(sg * 0.10, 0.02, 0.10), 0.052)     # shin
    ob, mb = _mball(f'FigShoes{n}x', shoes)
    parts.append(ob)
    for sg in (-1, 1):
        _cap(mb, P(sg * 0.10, 0.03, 0.045), P(sg * 0.10, -0.13, 0.045), 0.045)
    for p_ in parts:
        p_.parent = root
    root.location = (x, y, 0)
    root.rotation_euler = (0, 0, math.radians(face + 90))
    return root


def tree(x, y, h=9.0, seed=0):
    """A broadleaf tree: trunk plus a lumpy, displaced canopy."""
    rng = np.random.default_rng(seed)
    bark = mat_cached('Bark', (0.07, 0.05, 0.035), 0.9)
    leaf = mat_cached(f'Leaf{seed % 3}', [(0.035, 0.07, 0.02), (0.05, 0.085, 0.025), (0.03, 0.06, 0.025)][seed % 3], 0.8)
    add_prim('cyl', (x, y, h * 0.25), (h * 0.025, h * 0.025, h * 0.25), bark, top=0.7, seg=12)
    for k in range(5):
        c = Vector((x, y, h * 0.62)) + Vector(rng.normal(0, 1, 3)) * Vector((h * 0.14, h * 0.14, h * 0.08))
        r = h * rng.uniform(0.18, 0.26)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
        off = Vector(rng.uniform(-40, 40, 3))
        for v in bm.verts:
            v.co *= 1.0 + 0.25 * noise.fractal(v.co * 2.5 + off, 0.7, 2.0, 3)
        me = bpy.data.meshes.new('Canopy')
        bm.to_mesh(me)
        bm.free()
        for f in me.polygons:
            f.use_smooth = True
        ob = bpy.data.objects.new('Canopy', me)
        bpy.context.collection.objects.link(ob)
        ob.location = c
        ob.scale = (r, r, r * 0.85)
        ob.data.materials.append(leaf)


def facade(storeys_per_unit, cols_per_unit, concrete=(0.55, 0.53, 0.50), glass=(0.05, 0.07, 0.09), duty=(0.62, 0.78)):
    """Concrete frame and window grid in object coordinates (a cube of half-size 1)."""
    key = f'Facade{storeys_per_unit:.2f}_{cols_per_unit:.2f}_{concrete}_{glass}'
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    N = look.node
    tc = N(nt, 'ShaderNodeTexCoord', (-1000, 0))
    sep = N(nt, 'ShaderNodeSeparateXYZ', (-800, 0))
    nt.links.new(tc.outputs['Object'], sep.inputs[0])

    def band(src, n, dut):
        mul = N(nt, 'ShaderNodeMath', (-600, 0), operation='MULTIPLY')
        nt.links.new(src, mul.inputs[0])
        mul.inputs[1].default_value = n
        fr = N(nt, 'ShaderNodeMath', (-450, 0), operation='FRACT')
        nt.links.new(mul.outputs[0], fr.inputs[0])
        lt = N(nt, 'ShaderNodeMath', (-300, 0), operation='LESS_THAN')
        nt.links.new(fr.outputs[0], lt.inputs[0])
        lt.inputs[1].default_value = dut
        return lt.outputs[0]
    zf = band(sep.outputs['Z'], storeys_per_unit, duty[0])
    xy = N(nt, 'ShaderNodeMath', (-700, -200), operation='ADD')
    nt.links.new(sep.outputs['X'], xy.inputs[0])
    nt.links.new(sep.outputs['Y'], xy.inputs[1])
    cf = band(xy.outputs[0], cols_per_unit, duty[1])
    win = N(nt, 'ShaderNodeMath', (-150, -100), operation='MULTIPLY')
    nt.links.new(zf, win.inputs[0])
    nt.links.new(cf, win.inputs[1])
    # some windows a little lighter (blinds, reflections)
    wn = N(nt, 'ShaderNodeTexWhiteNoise', (-450, -400))
    wn.noise_dimensions = '3D'
    snap = N(nt, 'ShaderNodeVectorMath', (-650, -400), operation='SNAP')
    nt.links.new(tc.outputs['Object'], snap.inputs[0])
    snap.inputs[1].default_value = (1 / cols_per_unit, 1 / cols_per_unit, 1 / storeys_per_unit)
    nt.links.new(snap.outputs['Vector'], wn.inputs['Vector'])
    gl = N(nt, 'ShaderNodeMix', (-150, -400), data_type='RGBA')
    nt.links.new(wn.outputs['Value'], gl.inputs['Factor'])
    gl.inputs['A'].default_value = glass + (1,)
    gl.inputs['B'].default_value = tuple(min(1.0, g * 3.0 + 0.04) for g in glass) + (1,)
    mix = N(nt, 'ShaderNodeMix', (50, 100), data_type='RGBA')
    nt.links.new(win.outputs[0], mix.inputs['Factor'])
    mix.inputs['A'].default_value = concrete + (1,)
    nt.links.new(gl.outputs['Result'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    rough = N(nt, 'ShaderNodeMix', (50, -100), data_type='FLOAT')
    nt.links.new(win.outputs[0], rough.inputs['Factor'])
    rough.inputs['A'].default_value = 0.85
    rough.inputs['B'].default_value = 0.06
    nt.links.new(rough.outputs['Result'], b.inputs['Roughness'])
    _MATS[key] = m
    return m


def building(x, y, w, d, h, storey=3.2, bay=3.0, concrete=(0.36, 0.35, 0.33), glass=(0.05, 0.07, 0.09)):
    """A box building whose window grid is sized in metres (storey height, window bay)."""
    # object coords span -1..1, so storeys per unit = (h / storey) / 2; columns per unit ~ (w / bay) / 2
    m = facade(round((h / storey) / 2, 2), round((w / bay) / 2, 2), concrete, glass)
    return add_prim('cube', (x, y, h / 2), (w / 2, d / 2, h / 2), m)


def office_block(x, y, w=26.0, h=60.0):
    """A 20-storey office block (3 m storeys), with a darker ground floor and a roof plant room."""
    ob = building(x, y, w, w, h, storey=3.0, bay=3.25)
    add_prim('cube', (x, y, h + 1.5), (w * 0.3, w * 0.3, 1.5), mat_cached('Plant', (0.4, 0.4, 0.4), 0.7))
    return ob


def supertall(x, y, h=828.0):
    """A stepped, tapering tower like the world's tallest building (828 m), with a tall spire."""
    tiers = [(0.00, 0.12, 52), (0.12, 0.24, 44), (0.24, 0.36, 36), (0.36, 0.47, 29), (0.47, 0.57, 23),
             (0.57, 0.65, 17), (0.65, 0.71, 12), (0.71, 0.75, 8)]
    for z0, z1, r in tiers:
        zc = (z0 + z1) / 2 * h
        glass = facade(round((z1 - z0) * h / 4.2 / 2, 1), 5.0, (0.62, 0.64, 0.66), (0.10, 0.13, 0.17), (0.72, 0.55))
        ob = add_prim('cyl', (x, y, zc), (r, r, (z1 - z0) / 2 * h), glass, seg=6, top=0.9)
        for f in ob.data.polygons:
            f.use_smooth = False
    add_prim('cyl', (x, y, 0.75 * h + 0.125 * h), (5.0, 5.0, 0.125 * h), mat_cached('Steel', (0.7, 0.7, 0.72), 0.25, 1.0),
             top=0.08, seg=12)


def city(cx, cy, radius, rng, avoid=(), n=900, tall_frac=0.06):
    """Blocks of mid-rise buildings on a jittered street grid, with a few towers."""
    con = [(0.30, 0.29, 0.27), (0.38, 0.36, 0.33), (0.24, 0.23, 0.22), (0.42, 0.39, 0.34)]
    gls = [(0.05, 0.07, 0.09), (0.08, 0.10, 0.12), (0.06, 0.08, 0.07)]
    pitch = 95.0
    k = 0
    for i in range(-int(radius / pitch), int(radius / pitch) + 1):
        for j in range(-int(radius / pitch), int(radius / pitch) + 1):
            x, y = cx + i * pitch + rng.uniform(-10, 10), cy + j * pitch + rng.uniform(-10, 10)
            r = math.hypot(x - cx, y - cy)
            if r > radius * rng.uniform(0.8, 1.0):
                continue
            if any(math.hypot(x - ax, y - ay) < ar for ax, ay, ar in avoid):
                continue
            tall = rng.random() < tall_frac * math.exp(-r / (radius * 0.4))
            h = rng.uniform(120, 330) if tall else float(np.clip(rng.lognormal(math.log(22), 0.5), 8, 90))
            w = rng.uniform(25, 45) if tall else rng.uniform(30, 70)
            dd = rng.uniform(25, 45) if tall else rng.uniform(30, 70)
            building(x, y, w, dd, h, concrete=con[rng.integers(4)], glass=gls[rng.integers(3)])
            k += 1
    return k


PEAK = 8849.0
PLAIN_X = -16000.0
_OFF = Vector((13.1, 7.7, 0.0))


def _base_ridge(x, y, cx, cy):
    xr, yr = x - cx, y - cy
    r = math.hypot(xr, yr)
    p = Vector((xr, yr, 0.0))
    base = 3800.0 + 1600.0 * noise.ridged_multi_fractal(p / 8000.0 + _OFF, 1.0, 2.0, 7, 1.0, 2.0)
    fore = min(1.0, max(0.0, (-y - 12000.0) / 18000.0))         # foothills on the camera side
    base *= 1.0 - 0.55 * fore
    ridge = 900.0 * noise.fractal(p / 2200.0 + _OFF, 0.62, 2.0, 7) * (0.4 + 0.6 * math.exp(-r / 9000.0))
    # the plains: west of the range the land drops to ~300 m through ranks of foothills
    edge = PLAIN_X + 3000.0 * noise.fractal(Vector((0.0, y / 15000.0, 0.0)) + _OFF * 3, 0.5, 2.0, 3)
    u = min(1.0, max(0.0, (x - edge) / 14000.0))
    u = u * u * (3 - 2 * u)
    hills = 250.0 + 200.0 * noise.fractal(p / 5000.0 + _OFF * 2, 0.5, 2.0, 4)
    return hills * (1 - u) + (base + ridge) * u, r


def _pyramid(xr, yr, R=8000.0):
    """A three-faced pyramid (like Everest's) of unit height: 1 at the summit, 0 at distance ~R."""
    best = -1e9
    for k in range(3):
        a = math.radians(90 + 120 * k + 15)
        best = max(best, (xr * math.cos(a) + yr * math.sin(a)))
    return max(0.0, 1.0 - best / (R * 0.5)) ** 1.25 * math.exp(-(math.hypot(xr, yr) / (R * 1.4)) ** 2)


_A = {}


def terrain_h(x, y, cx=9000.0, cy=0.0):
    """Himalayan-style terrain height (m): plains in the west, a ridged range, one 8,849 m peak at (cx, cy)."""
    if (cx, cy) not in _A:
        _A[(cx, cy)] = PEAK - _base_ridge(cx, cy, cx, cy)[0]
    h, r = _base_ridge(x, y, cx, cy)
    return max(h + _A[(cx, cy)] * _pyramid(x - cx, y - cy), 0.0)


def mountains(cx=9000.0, n=420, half=160000.0):
    """The terrain on a grid that is fine near the peak (~90 m) and coarse at the horizon (~2 km)."""
    a = 0.12
    def warp(u):                                 # u in [-1, 1] -> metres
        return half * (a * u + (1 - a) * u ** 3)
    me = bpy.data.meshes.new('Range')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=n, y_segments=n, size=1.0)
    for v in bm.verts:
        x, y = warp(v.co.x) + cx, warp(v.co.y)
        v.co = Vector((x, y, terrain_h(x, y, cx)))
    bm.to_mesh(me)
    bm.free()
    for f in me.polygons:
        f.use_smooth = True
    ob = bpy.data.objects.new('Range', me)
    bpy.context.collection.objects.link(ob)
    m = bpy.data.materials.new('RockSnow')
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    N = look.node
    geo = N(nt, 'ShaderNodeNewGeometry', (-900, 0))
    sep = N(nt, 'ShaderNodeSeparateXYZ', (-700, 0))
    nt.links.new(geo.outputs['Position'], sep.inputs[0])
    sepn = N(nt, 'ShaderNodeSeparateXYZ', (-700, -250))
    nt.links.new(geo.outputs['Normal'], sepn.inputs[0])
    nz = N(nt, 'ShaderNodeTexNoise', (-900, -500))
    nz.inputs['Scale'].default_value = 0.0012
    nz.inputs['Detail'].default_value = 12.0
    nt.links.new(geo.outputs['Position'], nz.inputs['Vector'])
    zj = N(nt, 'ShaderNodeMath', (-600, 100), operation='MULTIPLY_ADD')       # jittered snow line
    nt.links.new(nz.outputs['Fac'], zj.inputs[0])
    zj.inputs[1].default_value = -1400.0
    nt.links.new(sep.outputs['Z'], zj.inputs[2])
    hz = N(nt, 'ShaderNodeMapRange', (-450, 0))
    nt.links.new(zj.outputs[0], hz.inputs['Value'])
    hz.inputs['From Min'].default_value = 4300.0
    hz.inputs['From Max'].default_value = 4900.0
    flat = N(nt, 'ShaderNodeMapRange', (-450, -250))
    nt.links.new(sepn.outputs['Z'], flat.inputs['Value'])
    flat.inputs['From Min'].default_value = 0.38
    flat.inputs['From Max'].default_value = 0.60
    snow = N(nt, 'ShaderNodeMath', (-300, -100), operation='MULTIPLY')
    nt.links.new(hz.outputs['Result'], snow.inputs[0])
    nt.links.new(flat.outputs['Result'], snow.inputs[1])
    rock = N(nt, 'ShaderNodeMix', (-250, 200), data_type='RGBA')
    nt.links.new(nz.outputs['Fac'], rock.inputs['Factor'])
    rock.inputs['A'].default_value = (0.06, 0.05, 0.045, 1)
    rock.inputs['B'].default_value = (0.16, 0.13, 0.10, 1)
    low = N(nt, 'ShaderNodeMapRange', (-450, 400))                   # green below ~2.5 km
    nt.links.new(zj.outputs[0], low.inputs['Value'])
    low.inputs['From Min'].default_value = 1500.0
    low.inputs['From Max'].default_value = 3200.0
    veg = N(nt, 'ShaderNodeMix', (-120, 300), data_type='RGBA')
    nt.links.new(low.outputs['Result'], veg.inputs['Factor'])
    veg.inputs['A'].default_value = (0.045, 0.06, 0.025, 1)
    nt.links.new(rock.outputs['Result'], veg.inputs['B'])
    mix = N(nt, 'ShaderNodeMix', (-100, 0), data_type='RGBA')
    nt.links.new(snow.outputs[0], mix.inputs['Factor'])
    nt.links.new(veg.outputs['Result'], mix.inputs['A'])
    mix.inputs['B'].default_value = (0.80, 0.82, 0.87, 1)
    nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.9
    ob.data.materials.append(m)
    return ob


def haze(sc, start, depth, color=(0.55, 0.65, 0.80)):
    """Aerial perspective in the compositor: mist pass mixed towards the sky colour."""
    sc.view_layers[0].use_pass_mist = True
    sc.world.mist_settings.start = start
    sc.world.mist_settings.depth = depth
    sc.world.mist_settings.falloff = 'QUADRATIC'
    ct = sc.node_tree
    rl = ct.nodes['Render Layers']
    comp = ct.nodes['Composite']
    src = comp.inputs['Image'].links[0].from_socket
    mix = ct.nodes.new('CompositorNodeMixRGB')
    mix.inputs[2].default_value = color + (1,)
    mul = ct.nodes.new('CompositorNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = 0.7
    ct.links.new(rl.outputs['Mist'], mul.inputs[0])
    ct.links.new(mul.outputs[0], mix.inputs['Fac'])
    ct.links.new(src, mix.inputs[1])
    ct.links.new(mix.outputs['Image'], comp.inputs['Image'])


# ------------------------------------------------------------------------------------ the cards


def aim(cam, loc, target, up=(0, 0, 1)):
    """Point the camera at target with 'up' kept up on screen."""
    loc, target, up = Vector(loc), Vector(target), Vector(up)
    fwd = (target - loc).normalized()
    right = fwd.cross(up).normalized()
    upc = right.cross(fwd)
    m = Matrix((right, upc, -fwd)).transposed()
    cam.location = loc
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = m.to_quaternion()


class ScaleShot:
    frames = 168
    spin_deg = 50.0           # how far the rock turns over the shot

    def label_at(self, key, world):
        sc = bpy.context.scene
        bpy.context.view_layer.update()
        p = world_to_camera_view(sc, self.cam, Vector(world))
        self.labels[key] = [round(p.x, 4), round(1 - p.y, 4)]

    def spin(self, ob, f, tilt=(0.35, 0.2), rate=1.0, phase=0.0):
        a = math.radians(self.spin_deg * rate) * f / (self.frames - 1) + phase
        ob.rotation_mode = 'QUATERNION'
        q_tilt = Euler((tilt[0], tilt[1], 0.0), 'XYZ').to_quaternion()
        ob.rotation_quaternion = q_tilt @ Euler((0.0, 0.0, a), 'XYZ').to_quaternion()


ROCK_STYLE = {1.0: ('boulder', 6), 60.0: ('rubble', 7), 1000.0: ('top', 7), 1e4: ('eros', 7), 1e5: ('big', 7)}


def place_rock(D, x, y, z_bottom, seed, gap=0.06):
    """A rock of mean diameter D hovering just clear of the ground (so it can turn)."""
    st, sub = ROCK_STYLE[D]
    ob = asteroid_mesh(f'Rock{D:g}', seed=seed, subdiv=sub, style=st)
    ob.scale = tuple(s * D for s in ob.scale)
    ob.location = (x, y, z_bottom + D * (0.5 + gap))
    return ob


@shot('S1_1m', 168)
class Scale1m(ScaleShot):
    def build(self, sc):
        self.labels = {}
        sky(sc, 34, 30, 0.33)
        ground(600, 'plaza')
        rng = np.random.default_rng(4)
        for k in range(14):
            tree(rng.uniform(-70, 70), rng.uniform(35, 90), rng.uniform(7, 13), seed=k)
        self.rock = place_rock(1.0, -0.75, 0.0, 0.0, 11, 0.12)
        person(0.75, 0.15, 1.75, face=-110, palette=0)
        self.cam = look.camera((0.0, -7.5, 1.2), (0.0, 0.0, 0.85), 45)
        self.cam.data.clip_start = 0.05
        haze(sc, 30.0, 400.0, (0.62, 0.70, 0.82))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (0.5 - 0.9 * x, -7.6 + 0.8 * x, 1.15), (0.0, 0.0, 0.85))
        self.spin(self.rock, f)
        self.label_at('rock', (-0.75, 0.0, 1.0 * 1.12 + 0.08))
        self.label_at('other', (0.75, 0.15, 1.85))
        return 0.0


@shot('S2_60m', 168)
class Scale60m(ScaleShot):
    def build(self, sc):
        self.labels = {}
        sky(sc, 36, 55, 0.33)
        ground(8000, 'plaza')
        self.rock = place_rock(60.0, -40.0, 0.0, 0.0, 21, 0.08)
        office_block(36.0, 0.0)
        rng = np.random.default_rng(7)
        city(0.0, 250.0, 1400.0, rng, avoid=((0.0, 0.0, 220.0), (0.0, -300.0, 260.0)), tall_frac=0.03)
        for k, (px, py) in enumerate([(-6.0, -40.0), (-4.5, -42.0), (8.0, -35.0), (14.0, -48.0)]):
            person(px, py, rng.uniform(1.6, 1.85), face=rng.uniform(-150, -30), palette=k, n=k)
        for k in range(6):
            tree(-110 + 40 * k, -60.0, rng.uniform(8, 11), seed=20 + k)
        self.cam = look.camera((10.0, -200.0, 7.0), (0.0, 0.0, 30.0), 35)
        haze(sc, 150.0, 3000.0)

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (16.0 - 14 * x, -205.0 + 25 * x, 7.0), (0.0, 0.0, 31.0))
        self.spin(self.rock, f)
        self.label_at('rock', (-40.0, 0.0, 60 * 1.08 + 3))
        self.label_at('other', (36.0, 0.0, 64.0))
        return 0.0


@shot('S3_1km', 168)
class Scale1km(ScaleShot):
    def build(self, sc):
        self.labels = {}
        sky(sc, 30, 60, 0.33)
        ground(120000, 'city')
        self.rock_xy = (-650.0, 250.0)
        self.rock = place_rock(1000.0, *self.rock_xy, 0.0, 31, 0.05)
        supertall(480.0, 250.0)
        rng = np.random.default_rng(3)
        city(0.0, 1800.0, 4200.0, rng, avoid=((-650.0, 250.0, 650.0), (480.0, 250.0, 120.0), (0.0, -2600.0, 1900.0)),
             tall_frac=0.15)
        self.cam = look.camera((-150.0, -3800.0, 150.0), (0.0, 0.0, 400.0), 32)
        self.cam.data.clip_end = 2e5
        haze(sc, 600.0, 26000.0)

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (-150.0 + 260 * x, -3800.0 + 350 * x, 150.0), (-60.0, 0.0, 420.0))
        self.spin(self.rock, f)
        self.label_at('rock', (self.rock_xy[0], self.rock_xy[1], 1000 * 1.05 + 40))
        self.label_at('other', (480.0, 250.0, 860.0))
        return 0.0


@shot('S4_10km', 168)
class Scale10km(ScaleShot):
    def build(self, sc):
        self.labels = {}
        sky(sc, 24, 75, 0.36)
        mountains()
        D = 10000.0
        self.rock_xy = (-19500.0, 3000.0)
        g = max(terrain_h(self.rock_xy[0] + dx, self.rock_xy[1] + dy) for dx in np.linspace(-6000, 6000, 13)
                for dy in np.linspace(-6000, 6000, 13))
        self.rock = place_rock(D, *self.rock_xy, g, 41, 0.04)
        self.top = self.rock.location.z + D / 2
        self.cam = look.camera((-5500.0, -60000.0, 4800.0), (-5500.0, 0.0, 5200.0), 40)
        self.cam.data.clip_end = 1e6
        haze(sc, 12000.0, 300000.0, (0.62, 0.70, 0.84))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (-5000.0 - 1500 * x, -62000.0 + 6000 * x, 4800.0), (-5500.0, 0.0, 5000.0))
        self.spin(self.rock, f)
        self.label_at('rock', (self.rock_xy[0], self.rock_xy[1], self.top + 500))
        self.label_at('other', (9000.0, 0.0, PEAK + 500))
        return 0.0


def air_layers(R, rho0=1.0, H=0.014, tops=(0.003, 0.006, 0.010, 0.015, 0.021, 0.028, 0.036, 0.045, 0.056, 0.070,
                                             0.086, 0.105)):
    """The air as nested shells of glowing gas, densest at the ground and thinning out with height (scale
    height H; 1 unit = 1,000 km). Each shell is homogeneous, so they render fast. Sight lines that skim the
    Earth cross much more air, so the limb glows; a rock sticking out of the air is not veiled."""
    prev = 0.0
    n = len(tops)
    for k, h in enumerate(tops):
        e = rho0 * (math.exp(-prev / H) - math.exp(-h / H))
        if k == n - 1:
            e = rho0 * math.exp(-prev / H)
        x = k / (n - 1)
        col = (0.62 - 0.32 * x, 0.76 - 0.24 * x, 1.0)
        ob = look.uv_sphere(f'Air{k}', R + h, 192, 96)
        m = bpy.data.materials.new(f'Air{k}')
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs['Color'].default_value = col + (1,)
        em.inputs['Strength'].default_value = e
        nt.links.new(em.outputs['Emission'], out.inputs['Volume'])
        ob.data.materials.append(m)
        ob.visible_shadow = False
        prev = h


@shot('S5_100km', 168)
class Scale100km(ScaleShot):
    """1 unit = 1,000 km. The rocks stand on the Earth, seen from orbit (~300 km up, ~1,000 km away). The big
    one's top sticks out above the air into black space; the 10 km one beside it barely shows."""

    def build(self, sc):
        self.labels = {}
        R = look.R_EARTH
        self.up = Vector((0.30, -0.25, 0.92)).normalized()
        self.t = self.up.cross(Vector((0, 0, 1))).normalized()          # horizontal, towards the camera
        self.side = self.up.cross(self.t).normalized()                  # horizontal, across the screen
        self.sun_dir = (self.up * 0.62 + self.t * 0.55 - self.side * 0.55).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        self.earth, self.shell = shots.make_earth(tuple(self.sun_dir), spin_deg=float(os.environ.get('S5_SPIN', 20)))
        self.shell.hide_render = True
        air_layers(R)
        self.P = self.up * R
        self.big = asteroid_mesh('Rock100km', seed=51, subdiv=7, style='big')
        self.big.scale = tuple(s * 0.100 for s in self.big.scale)
        self.big_c = self.P + self.up * 0.053 - self.side * 0.045
        self.big.location = self.big_c
        self.small = asteroid_mesh('Rock10km', seed=52, subdiv=6, style='eros')
        self.small.scale = tuple(s * 0.010 for s in self.small.scale)
        self.small_c = self.P + self.up * 0.0056 + self.side * 0.075
        self.small.location = self.small_c
        self.cam = look.camera((0, 0, 10), (0, 0, 0), 90)
        self.cam.data.clip_start = 0.001
        self.cam.data.clip_end = 100.0

    def frame(self, f):
        R = look.R_EARTH
        x = smooth(f / (self.frames - 1))
        loc = self.P + self.t * (1.06 - 0.10 * x) + self.up * 0.15
        aim(self.cam, loc, self.P + self.up * 0.040 + self.side * 0.012, self.up)
        self.spin(self.big, f, rate=0.5)
        self.spin(self.small, f, rate=1.5, phase=1.0)
        self.label_at('rock', tuple(self.big_c + self.up * 0.056))
        self.label_at('other', tuple(self.small_c + self.up * 0.0065))
        # the edge of space (100 km up) along the horizon, as a curve of points
        c = loc.normalized()
        w = (self.P - c * self.P.dot(c)).normalized()
        cphi = R / loc.length
        L0 = (c * cphi + w * math.sqrt(1 - cphi * cphi)) * (R + 0.100)
        for k, ang in enumerate(np.linspace(-0.09, 0.09, 9)):
            self.label_at(f'space{k}', tuple(Quaternion(c, float(ang)) @ L0))
        return 0.0


@shot('P_lineup', 144)
class Lineup:
    """The five rocks side by side (sizes not to scale), each turning, on a transparent background."""
    xs = [300, 620, 960, 1300, 1640]
    rad = [16, 30, 52, 92, 150]
    y = 600

    def build(self, sc):
        sc.render.resolution_percentage = 100
        sc.cycles.samples = max(sc.cycles.samples, 12)
        sc.render.film_transparent = True
        sc.render.image_settings.color_mode = 'RGBA'
        sc.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
        look.sun((-0.55, -0.6, 0.55), 6.0)
        cam = bpy.data.cameras.new('Ortho')
        cam.type = 'ORTHO'
        cam.ortho_scale = 1920.0
        self.cam = bpy.data.objects.new('Ortho', cam)
        bpy.context.collection.objects.link(self.cam)
        sc.camera = self.cam
        self.cam.location = (960.0, -2000.0, 540.0)
        self.cam.rotation_euler = (math.pi / 2, 0, 0)
        cam.clip_end = 5000.0
        self.rocks = []
        for i, (x, r) in enumerate(zip(self.xs, self.rad)):
            st, sub = ROCK_STYLE[[1.0, 60.0, 1000.0, 1e4, 1e5][i]]
            ob = asteroid_mesh(f'L{i}', seed=60 + i, subdiv=min(sub, 6 if r < 60 else 7), style=st)
            ob.scale = tuple(s * 2 * r for s in ob.scale)
            ob.location = (x, 0.0, 1080 - self.y)
            self.rocks.append(ob)

    def frame(self, f):
        for i, ob in enumerate(self.rocks):
            a = math.radians(70.0 * (1.0 - 0.12 * i)) * f / 143.0 + i
            ob.rotation_mode = 'QUATERNION'
            ob.rotation_quaternion = (Euler((0.4 + 0.2 * i, 0.3, 0.0), 'XYZ').to_quaternion() @
                                      Euler((0.0, 0.0, a), 'XYZ').to_quaternion())
        return 0.0
