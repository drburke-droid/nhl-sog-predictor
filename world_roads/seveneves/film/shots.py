"""The film's shots, driven frame by frame by the simulation (Blender 4.5, Cycles).

blender -b -P film/render.py -- <shot> <first> <last> <quality>

Every shot maps its frames to simulated time. Positions are geocentric, in the simulation's inertial
ecliptic frame (1 unit = 1,000 km); the Sun lamp follows the Sun's direction from the integrator.
Time-lapse shots freeze the Earth's spin (a spinning Earth strobes when one frame is hours long).
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector, Matrix, Quaternion

import look
import moon_pieces

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'out', 'film_data')
FPS = 24
HOUR, DAY, YEAR = 3600.0, 86400.0, 365.25 * 86400.0
TILT = math.radians(23.44)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


def logtime(f, f0, f1, t0, t1):
    x = min(max((f - f0) / (f1 - f0), 0.0), 1.0)
    return math.exp(math.log(t0) + x * (math.log(t1) - math.log(t0)))


class Data:
    _fine = None
    _rain = None

    @classmethod
    def fine(cls):
        if cls._fine is None:
            d = np.load(os.path.join(DATA, 'fine.npz'))
            cls._fine = {k: d[k] for k in d.files}
            cls._fine['merges'] = json.loads(str(d['merges']))
        return cls._fine

    @classmethod
    def rain(cls):
        if cls._rain is None:
            d = np.load(os.path.join(DATA, 'rain.npz'))
            cls._rain = {k: d[k] for k in d.files}
        return cls._rain


def interp_rows(t, ts, arr):
    """Linear interpolation of arr[k, ...] at time t (NaN where either side is NaN)."""
    k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    x = (t - ts[k]) / (ts[k + 1] - ts[k])
    return arr[k] * (1 - x) + arr[k + 1] * x


# --------------------------------------------------------------------------------- common pieces
def setup(quality):
    sc = look.reset_scene()
    q = dict(animatic=(0.4, 4, 4.0), preview=(0.5, 6, 3.0), draft=(0.5, 12, 2.0), final=(1.0, 16, 1.5))[quality]
    sc.render.resolution_percentage = int(q[0] * 100)
    sc.cycles.samples = q[1]
    sc.cycles.dicing_rate = q[2]
    sc.render.fps = FPS
    sc.render.use_persistent_data = True
    look.add_stars(0.0025)
    # a little bloom, so bright flashes and fireballs read as bright
    sc.use_nodes = True
    ct = sc.node_tree
    rl = ct.nodes.get('Render Layers') or ct.nodes.new('CompositorNodeRLayers')
    comp = ct.nodes.get('Composite') or ct.nodes.new('CompositorNodeComposite')
    gl = ct.nodes.new('CompositorNodeGlare')
    gl.glare_type = 'FOG_GLOW'
    gl.quality = 'HIGH'
    gl.threshold = 1.5
    gl.size = 7
    gl.inputs['Threshold'].default_value = 1.5             # Blender 4.5 reads the sockets
    ct.links.new(rl.outputs['Image'], gl.inputs['Image'])
    ct.links.new(gl.outputs['Image'], comp.inputs['Image'])
    return sc


def sun_lamp(direction):
    return look.sun(direction, 5.0)


def point_sun(ob, direction):
    d = Vector(direction).normalized()
    ob.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')


def make_earth(sun_dir, spin_deg=0.0):
    e = look.earth(tuple(sun_dir))
    look.add_clouds_to_surface(e, tuple(sun_dir))
    look.add_aerial_haze(e, tuple(sun_dir))
    shell = look.atmosphere_fast(tuple(sun_dir))
    add_sky_heat(shell)
    e.rotation_mode = 'XYZ'
    e.rotation_euler = (TILT, 0.0, math.radians(spin_deg))
    return e, shell


def add_steam(e, shell):
    """A pale, zonally banded steam-cloud deck (the oceans boiled into a 10-100 bar sky) and a thick
    white limb, blended in by the value node 'steam' (0..1)."""
    mat = e.data.materials[0]
    nt = mat.node_tree
    N = look.node
    bsdf = nt.nodes['Principled BSDF']
    src = bsdf.inputs['Base Color'].links[0].from_socket
    tc = N(nt, 'ShaderNodeTexCoord', (-600, 1500))
    mp = N(nt, 'ShaderNodeMapping', (-400, 1500))
    mp.inputs['Scale'].default_value = (1.2, 1.2, 14.0)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nz = N(nt, 'ShaderNodeTexNoise', (-200, 1500))
    nz.inputs['Scale'].default_value = 1.4
    nz.inputs['Detail'].default_value = 6.0
    nz.inputs['Distortion'].default_value = 0.35
    nt.links.new(mp.outputs['Vector'], nz.inputs['Vector'])
    band = N(nt, 'ShaderNodeMix', (0, 1500), data_type='RGBA')
    nt.links.new(nz.outputs['Fac'], band.inputs['Factor'])
    band.inputs['A'].default_value = (0.58, 0.56, 0.51, 1)
    band.inputs['B'].default_value = (0.82, 0.79, 0.72, 1)
    amt = N(nt, 'ShaderNodeValue', (0, 1700))
    amt.name = 'steam'
    amt.outputs[0].default_value = 0.0
    mix = N(nt, 'ShaderNodeMix', (300, 1300), data_type='RGBA')
    nt.links.new(amt.outputs[0], mix.inputs['Factor'])
    nt.links.new(src, mix.inputs['A'])
    nt.links.new(band.outputs['Result'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], bsdf.inputs['Base Color'])
    snt = shell.data.materials[0].node_tree
    snt.nodes['Emission'].inputs['Color'].default_value = (0.32, 0.55, 1.0, 1)


def set_steam(e, shell, x):
    e.data.materials[0].node_tree.nodes['steam'].outputs[0].default_value = x
    c = (lerp(0.32, 0.92, x), lerp(0.55, 0.90, x), lerp(1.0, 0.86, x), 1)
    snt = shell.data.materials[0].node_tree
    snt.nodes['Emission'].inputs['Color'].default_value = c
    snt.nodes[shell.data.materials[0]['strength_node']].inputs[1].default_value = lerp(1.6, 1.2, x)
    for n in snt.nodes:
        if n.bl_idname == 'ShaderNodeMath' and n.operation == 'POWER':
            n.inputs[1].default_value = lerp(3.0, 1.6, x)          # a thick sky glows further in


def set_sun_vectors(e, shell, sun_dir):
    for ob in (e, shell):
        nt = ob.data.materials[0].node_tree
        for n in nt.nodes:
            if n.bl_idname == 'ShaderNodeCombineXYZ' and n.label != 'keep':
                n.inputs[0].default_value, n.inputs[1].default_value, n.inputs[2].default_value = tuple(sun_dir)


def add_sky_heat(shell):
    """A glow of the whole upper atmosphere at the radiative temperature of the incoming heat."""
    nt = shell.data.materials[0].node_tree
    N = look.node
    temp = N(nt, 'ShaderNodeValue', (0, -800))
    temp.name = 'sky_T'
    temp.outputs[0].default_value = 300.0
    amt = N(nt, 'ShaderNodeValue', (0, -950))
    amt.name = 'sky_glow'
    amt.outputs[0].default_value = 0.0
    bb = N(nt, 'ShaderNodeBlackbody', (200, -800))
    nt.links.new(temp.outputs[0], bb.inputs['Temperature'])
    lw = N(nt, 'ShaderNodeLayerWeight', (0, -1100))
    lw.inputs['Blend'].default_value = 0.6
    lim = N(nt, 'ShaderNodeMath', (200, -1100), operation='MULTIPLY_ADD')
    nt.links.new(lw.outputs['Facing'], lim.inputs[0])
    lim.inputs[1].default_value = 1.5
    lim.inputs[2].default_value = 0.25
    geo = N(nt, 'ShaderNodeNewGeometry', (0, -1250))
    front = N(nt, 'ShaderNodeMath', (200, -1250), operation='SUBTRACT')
    front.inputs[0].default_value = 1.0
    nt.links.new(geo.outputs['Backfacing'], front.inputs[1])
    s1 = N(nt, 'ShaderNodeMath', (400, -1000), operation='MULTIPLY')
    nt.links.new(lim.outputs['Value'], s1.inputs[0])
    nt.links.new(amt.outputs[0], s1.inputs[1])
    s2 = N(nt, 'ShaderNodeMath', (550, -1000), operation='MULTIPLY')
    nt.links.new(s1.outputs['Value'], s2.inputs[0])
    nt.links.new(front.outputs['Value'], s2.inputs[1])
    em = N(nt, 'ShaderNodeEmission', (700, -800))
    nt.links.new(bb.outputs['Color'], em.inputs['Color'])
    nt.links.new(s2.outputs['Value'], em.inputs['Strength'])
    out = nt.nodes['Material Output']
    old = out.inputs['Surface'].links[0].from_socket
    add = N(nt, 'ShaderNodeAddShader', (900, -500))
    nt.links.new(old, add.inputs[0])
    nt.links.new(em.outputs['Emission'], add.inputs[1])
    nt.links.new(add.outputs['Shader'], out.inputs['Surface'])


def _vis_radiance(T):
    """Planck radiance weighted by the eye's photopic response (Gaussian stand-in, 555 nm, sigma 45 nm)."""
    lam = np.linspace(380e-9, 780e-9, 200)
    B = 2 * 6.626e-34 * 2.998e8 ** 2 / lam ** 5 / np.expm1(6.626e-34 * 2.998e8 / (lam * 1.381e-23 * T))
    V = np.exp(-0.5 * ((lam - 555e-9) / 45e-9) ** 2)
    return float(np.trapz(B * V, lam))


VIS_980 = _vis_radiance(980.0)


def sky_heat_from_power(P_wm2, f_space=0.5):
    """The heated upper atmosphere radiates the absorbed rain power up and down, 2 sigma T^4 = (1-f) P.
    Its visible glow is scaled by the eye-weighted Planck radiance, normalised so a 980 K layer (the
    30-day peak of the default run) is about as bright as city lights."""
    T = ((1 - f_space) * max(P_wm2, 1.0) / (2 * 5.670374e-8)) ** 0.25
    return T, 0.04 * _vis_radiance(T) / VIS_980


def debris_object(name='Debris', color=(0.30, 0.29, 0.27)):
    me = bpy.data.meshes.new(name)
    me.vertices.add(1)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    me.attributes.new('radius', 'FLOAT', 'POINT')
    me.attributes.new('bright', 'FLOAT', 'POINT')
    gn = bpy.data.node_groups.new(name + 'GN', 'GeometryNodeTree')
    gn.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    gn.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = gn.nodes.new('NodeGroupInput')
    go = gn.nodes.new('NodeGroupOutput')
    m2p = gn.nodes.new('GeometryNodeMeshToPoints')
    rad = gn.nodes.new('GeometryNodeInputNamedAttribute')
    rad.data_type = 'FLOAT'
    rad.inputs['Name'].default_value = 'radius'
    gn.links.new(gi.outputs[0], m2p.inputs['Mesh'])
    gn.links.new(rad.outputs['Attribute'], m2p.inputs['Radius'])
    mat = bpy.data.materials.new(name + 'Mat')
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'bright'
    attr.attribute_type = 'GEOMETRY'
    bb = nt.nodes.new('ShaderNodeBlackbody')
    bb.inputs['Temperature'].default_value = 4200.0
    nt.links.new(bb.outputs['Color'], bsdf.inputs['Emission Color'])
    nt.links.new(attr.outputs['Fac'], bsdf.inputs['Emission Strength'])
    mat.cycles.emission_sampling = 'NONE'
    sm = gn.nodes.new('GeometryNodeSetMaterial')
    sm.inputs['Material'].default_value = mat
    gn.links.new(m2p.outputs['Points'], sm.inputs['Geometry'])
    gn.links.new(sm.outputs['Geometry'], go.inputs[0])
    mod = ob.modifiers.new('points', 'NODES')
    mod.node_group = gn
    ob.visible_shadow = False            # each point stands for a diffuse dust swarm, not a boulder
    return ob


def set_points(ob, xyz, radius, bright=None):
    me = ob.data
    n = len(xyz)
    if n == 0:
        xyz = np.zeros((1, 3)) + 1e6
        radius = np.zeros(1)
        bright = None
        n = 1
    if len(me.vertices) != n:
        me.clear_geometry()
        me.vertices.add(n)
        me.attributes.new('radius', 'FLOAT', 'POINT')
        me.attributes.new('bright', 'FLOAT', 'POINT')
    me.vertices.foreach_set('co', np.asarray(xyz, np.float32).ravel())
    me.attributes['radius'].data.foreach_set('value', np.broadcast_to(np.asarray(radius, np.float32), (n,)).copy())
    b = np.zeros(n, np.float32) if bright is None else np.broadcast_to(np.asarray(bright, np.float32), (n,)).copy()
    me.attributes['bright'].data.foreach_set('value', b)
    me.update()


def place_camera(cam, loc, target, up=(0, 0, 1)):
    cam.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = d.to_track_quat('-Z', 'Y')


# --------------------------------------------------------------------------------- the broken Moon
class Fragments:
    """Seven meshes moved by the simulation, merging into rigid groups when the simulation merges them."""

    def __init__(self, seed=5):
        d = Data.fine()
        self.d = d
        p0 = d['pos'][0, 1:8]
        m0 = d['mass'][0, 1:8]
        self.com0 = (p0 * m0[:, None]).sum(0) / m0.sum()
        x = -self.com0 / np.linalg.norm(self.com0)                  # sub-Earth direction
        z = np.array([0.0, 0.0, 1.0])
        z = z - x * (z @ x)
        z /= np.linalg.norm(z)
        y = np.cross(z, x)
        tex = Matrix((tuple(x), tuple(y), tuple(z)))                # rows: texture axes in world coords
        self.tex = tex
        self.obs = moon_pieces.build_pieces(tex)
        c = moon_pieces.cells()
        cen = np.array(c['centroid']) * look.R_MOON
        fr_ = np.array(c['frac'])
        self.mean = (cen * fr_[:, None]).sum(0) / fr_.sum()
        rng = np.random.default_rng(seed)
        self.spin = []
        for k in range(7):
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            period = rng.uniform(30, 70) * HOUR
            self.spin.append((Vector(ax), 2 * math.pi / period))
        # carriers: piece k is carried by body id c with a fixed offset (and frozen orientation) after merges
        self.carrier = {k + 1: [(0.0, k + 1, None)] for k in range(7)}
        for mg in d['merges']:
            t, (a, b), new = mg['t'], mg['ids'], mg['new']
            lr = new[0]
            for k in range(1, 8):
                c = self.current(k, t - 1e-3)
                if c[1] in (a, b):
                    self.carrier[k].append((t, lr, (a, b)))
        self.small_ids = sorted({i for mg in d['merges'] for i in mg['new'][1:]})

    def current(self, k, t):
        c = self.carrier[k][0]
        for e in self.carrier[k]:
            if e[0] <= t:
                c = e
        return c

    def body_pos(self, i, t):
        p = interp_rows(t, self.d['t'], self.d['pos'][:, i])
        return p

    def intact_moon(self):
        """One seamless Moon where the pieces sit at t = 0 (separate pieces crack along their seams)."""
        m = look.moon(tuple(self.com(0.0) - self.mean), name='MoonIntact')
        m.rotation_mode = 'QUATERNION'
        m.rotation_quaternion = self.tex.transposed().to_quaternion()
        return m

    visible = True

    def show(self, on):
        self.visible = on
        for ob in self.obs:
            ob.hide_render = not on

    def update(self, t, glow, fracture=1.0):
        for k, ob in enumerate(self.obs, start=1):
            chain = [e for e in self.carrier[k] if e[0] <= t]
            # position: follow the last carrier, keeping the offset the piece had when it merged
            pos = self.body_pos(k, min(t, chain[1][0] - 1.0) if len(chain) > 1 else t)
            off = np.zeros(3)
            for j in range(1, len(chain)):
                tm, cid, _ = chain[j]
                prev = chain[j - 1][1]
                # member just before the merge, carrier just after it (they don't coexist at tm)
                pm = self.body_pos(k, tm - 1.0) if j == 1 else self.body_pos(prev, tm - 1.0) + off_prev
                off = pm - self.body_pos(cid, tm + 1.0)
                off_prev = off
            if len(chain) > 1:
                pos = self.body_pos(chain[-1][1], t) + off
            if np.any(np.isnan(pos)):
                ob.hide_render = True
                continue
            ob.hide_render = not self.visible
            ob.location = Vector(pos)
            ax, w = self.spin[k - 1]
            t_spin = t if len(chain) == 1 else chain[1][0]          # stop tumbling once stuck to another
            ob.rotation_mode = 'QUATERNION'
            ob.rotation_quaternion = Quaternion(ax, w * t_spin)
            nodes = ob.data.materials[0].node_tree.nodes
            nodes['glow'].outputs[0].default_value = glow
            nodes['fracture'].outputs[0].default_value = fracture
            # gravity pulls the shards into lumpy spheroids within a few hours (rock is far too weak)
            ob.data.shape_keys.key_blocks['slump'].value = 0.88 * smooth((t - 0.2 * HOUR) / (3.0 * HOUR))

    def com(self, t):
        ps, ms = [], []
        for i in range(1, 8):
            p = self.body_pos(i, t)
            if not np.any(np.isnan(p)):
                ps.append(p)
                ms.append(self.d['mass'][0, i])
        ps, ms = np.array(ps), np.array(ms)
        return (ps * ms[:, None]).sum(0) / ms.sum()


class DebrisCloud:
    """Debris parcels drawn as small clouds of sunlit grains (each parcel is a swarm, not one rock)."""

    def __init__(self, per=16, seed=3):
        d = Data.fine()
        self.d = d
        self.per = per
        rng = np.random.default_rng(seed)
        n = d['debris'].shape[1]
        self.jit = rng.normal(size=(n, per, 3)).astype(np.float32)
        self.size = rng.lognormal(0, 0.5, size=(n, per)).astype(np.float32)
        self.ob = debris_object()

    def update(self, t, cam_loc, px=0.00028):
        d = self.d
        P = interp_rows(t, d['debris_t'], d['debris'])
        ok = ~np.isnan(P[:, 0])
        if not ok.any():
            set_points(self.ob, np.zeros((0, 3)), 0)
            return
        age = np.maximum(t - d['debris_born'][ok], 0.0)
        spread = np.minimum(0.00004 * age, 3.0)                   # swarm spreads ~40 m/s
        pts = P[ok][:, None, :] + self.jit[ok] * spread[:, None, None]
        pts = pts.reshape(-1, 3)
        dist = np.linalg.norm(pts - np.array(cam_loc), axis=1)
        rad = px * dist * self.size[ok].reshape(-1)
        set_points(self.ob, pts, rad)


# --------------------------------------------------------------------------------- shots
SHOTS = {}


def shot(name, frames):
    def deco(cls):
        cls.name, cls.frames = name, frames
        SHOTS[name] = cls
        return cls
    return deco


@shot('A_moon', 120)
class MoonIntro:
    """The whole Moon, gently pushed in on. The pieces are in place and seamless. A flash at the end."""

    def build(self, sc):
        self.fr = Fragments()
        d = self.fr.d
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.com = self.fr.com(0.0)
        self.fr.update(0.0, 0.0, fracture=0.0)
        self.moon = self.fr.intact_moon()
        # camera: on the Earth side of the Moon (the familiar near side), Sun off to one side
        s = Vector(self.sun_dir)
        to_earth = -Vector(self.com).normalized()
        self.view = (to_earth * 0.45 + s * 0.9 + Vector((0, 0, 0.1))).normalized()
        self.cam = look.camera(tuple(Vector(self.com) + self.view * 9.0), tuple(self.com), 55)
    def frame(self, f):
        x = smooth(f / 119)
        dist = lerp(10.5, 7.2, x)
        place_camera(self.cam, Vector(self.com) + self.view * dist, self.com)
        # last 16 frames: the first 40 seconds after the breakup (seams open, glow shows through)
        t = 0.0 if f < 104 else 40.0 * (f - 103) / 16
        self.moon.hide_render = t > 0
        self.fr.show(t > 0)
        self.fr.update(t, 1.0 if t > 0 else 0.0, fracture=0.0 if t == 0 else min(1.0, t / 40.0))
        return t


@shot('B_breakup', 192)
class Breakup:
    """0 to 12 hours: the seven pieces fly apart, glowing where the interior is exposed."""

    def build(self, sc):
        self.fr = Fragments()
        self.deb = DebrisCloud()
        d = self.fr.d
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        s = Vector(self.sun_dir)
        to_earth = -Vector(self.fr.com(0.0)).normalized()
        self.view = (to_earth * 0.6 + s * 0.75 + Vector((0, 0, 0.2))).normalized()
        self.cam = look.camera((0, 0, 0), (1, 0, 0), 45)

    def frame(self, f):
        t = logtime(f, 0, 191, 40.0, 12 * HOUR)
        com = self.fr.com(t)
        glow = math.exp(-t / (5 * HOUR))
        self.fr.update(t, glow)
        dist = 8.0 + 3.4 * (t / 3600.0) ** 0.9
        loc = Vector(com) + self.view * dist
        place_camera(self.cam, loc, com)
        self.deb.update(t, loc)
        return t


@shot('C_merge', 144)
class Merge:
    """24 to 40 hours: two of the pieces fall back together at 1.25 km/s and stick."""

    def build(self, sc):
        self.fr = Fragments()
        self.deb = DebrisCloud()
        d = self.fr.d
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.cam = look.camera((0, 0, 0), (1, 0, 0), 50)
        s = Vector(self.sun_dir)
        side = s.cross(Vector((0, 0, 1))).normalized()
        self.view = (s * 0.75 + side * 0.55 + Vector((0, 0, 0.3))).normalized()

    def frame(self, f):
        t = lerp(28 * HOUR, 40 * HOUR, f / 143)
        self.fr.update(t, math.exp(-t / (5 * HOUR)))
        mid = (np.array(self.fr.obs[0].location) + np.array(self.fr.obs[6].location)) / 2
        loc = Vector(mid) + self.view * 11.0
        place_camera(self.cam, loc, mid)
        self.deb.update(t, loc)
        return t


@shot('D_wide', 168)
class Wide:
    """Day 0.5 to day 12, from behind the Earth: the broken Moon and its dust, debris starting to fall."""

    def build(self, sc):
        self.fr = Fragments()
        self.deb = DebrisCloud(per=6)
        d = self.fr.d
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.earth, self.shell = make_earth(self.sun_dir, spin_deg=200)
        com = Vector(self.fr.com(0.0))
        u = com.normalized()
        side = u.cross(Vector((0, 0, 1))).normalized()
        # beyond the Moon, looking back: the broken Moon in front, the Earth a small marble behind it
        self.loc = com * 1.13 + side * 9.0 + Vector((0, 0, 6.0))
        self.cam = look.camera(tuple(self.loc), (0, 0, 0), 28)

    def frame(self, f):
        t = logtime(f, 0, 167, 0.5 * DAY, 12 * DAY)
        com = Vector(self.fr.com(t))
        self.fr.update(t, max(0.0, math.exp(-t / (5 * HOUR))))
        # frame between the Earth and the cluster, drifting towards the Earth as the rain begins
        tgt = com.lerp(Vector((0, 0, 0)), lerp(0.25, 0.6, smooth(f / 167)))
        place_camera(self.cam, self.loc, tgt)
        self.deb.update(t, self.loc, px=0.0003)
        return t


@shot('E_rain', 240)
class Rain:
    """Day 2 to day 90 on the night side: meteors, the upper atmosphere heating to a red glow,
    the cloud deck thickening as the oceans start to boil."""

    def build(self, sc):
        r = Data.rain()
        self.r = r
        d = Data.fine()
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.earth, self.shell = make_earth(self.sun_dir, spin_deg=120)
        s = Vector(self.sun_dir)
        side = s.cross(Vector((0, 0, 1))).normalized()
        self.view = (-s * 0.75 + side * 0.62 + Vector((0, 0, 0.25))).normalized()
        self.cam = look.camera(tuple(self.view * 30.0), (0, 0, 0), 50)
        self.met = debris_object('Meteors', (0.2, 0.2, 0.2))
        self.rng = np.random.default_rng(7)

    def frame(self, f):
        r = self.r
        t = logtime(f, 0, 239, 2 * DAY, 90 * DAY)
        place_camera(self.cam, self.view * lerp(30.0, 26.0, smooth(f / 239)), (0, 0, 0))
        P = float(np.interp(t, r['series_t'], r['series_P']))
        ocean = float(np.interp(t, r['series_t'], r['series_ocean']))
        T, glow = sky_heat_from_power(P)
        nt = self.shell.data.materials[0].node_tree
        nt.nodes['sky_T'].outputs[0].default_value = T
        nt.nodes['sky_glow'].outputs[0].default_value = glow
        cover = min(1.0, (ocean / 0.01) ** 0.7)                   # ~1% of the ocean = a 2.6-bar steam sky
        em = self.earth.data.materials[0].node_tree
        em.nodes[self.earth.data.materials[0]['cloud_cover_node']].inputs['Factor'].default_value = cover
        # meteors: parcels arriving within +-15 days, each a stream hitting around its real entry point
        w = np.exp(-0.5 * ((r['t'] - t) / (6 * DAY)) ** 2)
        act = np.where(w > 0.05)[0]
        n = int(min(700, 25 + P / 300.0))
        if len(act):
            pick = self.rng.choice(act, size=n, p=w[act] / w[act].sum())
            base = r['dir'][pick]
            base = base + self.rng.normal(size=base.shape) * 0.35
        else:
            base = self.rng.normal(size=(n, 3))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        # a streak: 8 points along a short entry path, brightest at the end
        tang = np.cross(base, self.rng.normal(size=base.shape))
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        L = self.rng.uniform(0.05, 0.25, n)
        k = np.linspace(0, 1, 8)
        pts = (base[:, None, :] * (6.371 + 0.11 - 0.05 * k[None, :, None])
               + tang[:, None, :] * (L[:, None] * (1 - k))[:, :, None])
        br = (1.0 + 7.0 * k ** 2)[None, :] * self.rng.lognormal(0, 0.7, n)[:, None]
        set_points(self.met, pts.reshape(-1, 3), 0.0016, br.reshape(-1))
        return t


@shot('F_steam', 192)
class Steam:
    """Year 0.1 to year 300, day side: the Earth disappears under steam."""

    def build(self, sc):
        r = Data.rain()
        self.r = r
        d = Data.fine()
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.earth, self.shell = make_earth(self.sun_dir, spin_deg=300)
        add_steam(self.earth, self.shell)
        s = Vector(self.sun_dir)
        side = s.cross(Vector((0, 0, 1))).normalized()
        self.view = (s * 0.7 + side * 0.7 + Vector((0, 0, 0.15))).normalized()
        self.cam = look.camera(tuple(self.view * 22.0), (0, 0, 0), 50)

    def frame(self, f):
        r = self.r
        t = logtime(f, 0, 191, 0.02 * YEAR, 300 * YEAR)
        place_camera(self.cam, self.view * lerp(22.0, 20.0, f / 191), (0, 0, 0))
        ocean = float(np.interp(t, r['series_t'], r['series_ocean']))
        cover = min(1.0, (ocean / 0.01) ** 0.7)
        mat = self.earth.data.materials[0]
        mat.node_tree.nodes[mat['cloud_cover_node']].inputs['Factor'].default_value = cover
        set_steam(self.earth, self.shell, smooth(ocean / 0.05))       # >5% boiled: a 100-bar steam world
        return t


@shot('H_ladder', 480)
class Ladder:
    """The intact Moon, slowly turning: background for the impactor-size ladder (drawn in compositing)."""

    def build(self, sc):
        self.fr = Fragments()
        d = self.fr.d
        self.sun_dir = d['sun'][0] / np.linalg.norm(d['sun'][0])
        self.sun = sun_lamp(self.sun_dir)
        self.com = self.fr.com(0.0)
        self.fr.show(False)
        self.moon = self.fr.intact_moon()
        s = Vector(self.sun_dir)
        to_earth = -Vector(self.com).normalized()
        self.view = (to_earth * 0.45 + s * 0.9 + Vector((0, 0, 0.1))).normalized()
        self.cam = look.camera(tuple(Vector(self.com) + self.view * 7.4), tuple(self.com), 55)
        self.q0 = self.moon.rotation_quaternion.copy()

    def frame(self, f):
        self.moon.rotation_quaternion = Quaternion((0, 0, 1), math.radians(-0.05 * f)) @ self.q0
        return 0.0


def build(name, sc):
    s = SHOTS[name]()
    s.build(sc)
    return s
