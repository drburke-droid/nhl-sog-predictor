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

import assets
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
def sky(sc, elev_deg=28.0, rot_deg=40.0, strength=0.35, altitude=100.0, sun_strength=4.0):
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
    s.altitude = altitude
    s.air_density = 1.0
    s.dust_density = 1.5
    nt.links.new(s.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = strength
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    # matching sun lamp
    el, rot = math.radians(elev_deg), math.radians(rot_deg)
    d = Vector((math.sin(rot) * math.cos(el), -math.cos(rot) * math.cos(el), math.sin(el)))
    lamp = look.sun(tuple(d), sun_strength)
    lamp.data.angle = math.radians(1.0)
    sc.view_settings.exposure = -0.7
    return lamp


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
    """A 1 m rock beside a person on a cobbled square, trees behind."""

    def build(self, sc):
        self.labels = {}
        assets.hdri_world(sc, 'kloofendal_48d_partly_cloudy_puresky_4k', sun_az=232.0, exposure=-0.5)
        assets.ground_pbr(400, 'PavingStones070', 1.4)
        rng = np.random.default_rng(4)
        for k in range(9):
            assets.tree(rng.uniform(-45, 45), rng.uniform(28, 70), rng.uniform(6.5, 10.0),
                        'jacaranda_tree' if k % 3 == 0 else 'tree_small_02', rng.uniform(0, 6.3))
        self.rock = place_rock(1.0, -0.75, 0.0, 0.0, 11, 0.12)
        assets.human(0.78, 0.15, 1.75, face_deg=18.0, outfit=0)
        self.cam = look.camera((0.0, -7.5, 1.2), (0.0, 0.0, 0.85), 45)
        self.cam.data.clip_start = 0.05
        haze(sc, 40.0, 600.0, (0.70, 0.76, 0.85))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (0.5 - 0.9 * x, -7.6 + 0.8 * x, 1.15), (0.0, 0.0, 0.85))
        self.spin(self.rock, f)
        self.label_at('rock', (-0.75, 0.0, 1.0 * 1.12 + 0.08))
        self.label_at('other', (0.78, 0.15, 1.85))
        return 0.0


@shot('S2_60m', 168)
class Scale60m(ScaleShot):
    """A 60 m rock beside a 20-storey office block, on a plaza in a city."""

    def build(self, sc):
        self.labels = {}
        assets.hdri_world(sc, 'kloofendal_48d_partly_cloudy_puresky_4k', sun_az=240.0, exposure=-0.5)
        assets.ground_pbr(8000, 'Concrete049', 3.0, tint=(0.9, 0.9, 0.88))
        self.rock = place_rock(60.0, -40.0, 0.0, 0.0, 21, 0.08)
        assets.office_block(36.0, 0.0)
        rng = np.random.default_rng(7)
        assets.city(0.0, 250.0, 1400.0, rng, avoid=((0.0, 0.0, 220.0), (0.0, -300.0, 260.0)), tall_frac=0.04)
        for k, (px, py) in enumerate([(-6.0, -40.0), (-4.8, -41.5), (8.0, -35.0), (14.0, -48.0)]):
            assets.human(px, py, rng.uniform(1.62, 1.85), face_deg=rng.uniform(-60, 60),
                         kind='female' if k % 2 else 'male', outfit=k)
        for k in range(7):
            assets.tree(-120 + 40 * k, -62.0 + rng.uniform(-3, 3), rng.uniform(8, 11),
                        'jacaranda_tree' if k % 2 else 'tree_small_02', rng.uniform(0, 6.3))
        self.cam = look.camera((10.0, -200.0, 7.0), (0.0, 0.0, 30.0), 35)
        haze(sc, 200.0, 4000.0, (0.70, 0.76, 0.85))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (16.0 - 14 * x, -205.0 + 25 * x, 7.0), (0.0, 0.0, 31.0))
        self.spin(self.rock, f)
        self.label_at('rock', (-40.0, 0.0, 60 * 1.08 + 3))
        self.label_at('other', (36.0, 0.0, 64.0))
        return 0.0


@shot('S3_1km', 168)
class Scale1km(ScaleShot):
    """A 1 km rock beside an 828 m Y-plan supertall in a city of towers."""

    def build(self, sc):
        self.labels = {}
        assets.hdri_world(sc, 'kloofendal_48d_partly_cloudy_puresky_4k', sun_az=215.0, exposure=-0.25)
        assets.ground_pbr(120000, 'Asphalt031', 12.0, tint=(0.8, 0.8, 0.8))
        self.rock_xy = (-650.0, 250.0)
        self.rock = place_rock(1000.0, *self.rock_xy, 0.0, 31, 0.05)
        assets.supertall(480.0, 250.0)
        rng = np.random.default_rng(3)
        assets.city(0.0, 1800.0, 4200.0, rng, avoid=((-650.0, 250.0, 650.0), (480.0, 250.0, 150.0),
                                                     (0.0, -2600.0, 1900.0)), tall_frac=0.15)
        assets.suburb(0.0, -2000.0, 5200.0, 2600.0, rng)
        self.cam = look.camera((-150.0, -3800.0, 150.0), (0.0, 0.0, 400.0), 32)
        self.cam.data.clip_end = 2e5
        haze(sc, 800.0, 30000.0, (0.70, 0.76, 0.85))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (-150.0 + 260 * x, -3800.0 + 350 * x, 150.0), (-60.0, 0.0, 420.0))
        self.spin(self.rock, f)
        self.label_at('rock', (self.rock_xy[0], self.rock_xy[1], 1000 * 1.05 + 40))
        self.label_at('other', (480.0, 250.0, 860.0))
        return 0.0


@shot('S4_10km', 168)
class Scale10km(ScaleShot):
    """A 10 km rock beside the real Mount Everest (Copernicus 30 m terrain), seen from the west at 8.5 km."""

    def build(self, sc):
        self.labels = {}
        sky(sc, 26, 45, 0.30, altitude=8000.0, sun_strength=4.5)       # morning sun from the south-east
        m = assets.himalaya_material()
        c = (-6000.0, -3000.0)                   # finest terrain around the summit and the rock
        for name, step, box, hole, drop in (('TerrainNear', 1, 15.0, None, 0.0), ('TerrainMid', 3, 40.0, 14.9, 60.0),
                                            ('TerrainFar', 8, 120.0, 39.9, 150.0)):
            t = assets.terrain_mesh(name, step, box, hole, drop, centre=c)
            t.data.materials.append(m)
        D = 10000.0
        self.rock_xy = (-9900.0, 9900.0)
        # lowest centre height at which a 5 km-radius ball clears the terrain beneath it
        zc = 0.0
        for dx in np.linspace(-5000, 5000, 21):
            for dy in np.linspace(-5000, 5000, 21):
                r2 = dx * dx + dy * dy
                if r2 < 5000.0 ** 2:
                    hz = assets.terrain_height(self.rock_xy[0] + dx, self.rock_xy[1] + dy)
                    zc = max(zc, hz + math.sqrt(5000.0 ** 2 - r2))
        self.rock = place_rock(D, *self.rock_xy, zc - D * 0.5, 41, 0.03)
        self.top = self.rock.location.z + D / 2
        self.cam = look.camera((-27000.0, -27000.0, 8000.0), (-4950.0, 4950.0, 10600.0), 35)
        self.cam.data.clip_start = 5.0
        self.cam.data.clip_end = 6e5
        haze(sc, 6000.0, 160000.0, (0.64, 0.72, 0.86))

    def frame(self, f):
        x = smooth(f / (self.frames - 1))
        aim(self.cam, (-27500.0 + 2500 * x, -26500.0 + 1500 * x, 8000.0), (-4950.0, 4950.0, 10600.0))
        self.spin(self.rock, f)
        self.label_at('rock', (self.rock_xy[0], self.rock_xy[1], self.top + 500))
        self.label_at('other', (0.0, 0.0, 8849.0 + 300))
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
