"""The night sky from the ground, as a person standing in a field outside a small town would see it.

N1_1m     the Moon, and a brief star-like flash on its dark side (a 1 m rock hitting it)
N3_1km    a few nights later, real speed: ~30 shooting stars a second (11,000x normal)
N4_10km   real speed: ~30,000 a second (11 million x normal), as much light as a full Moon, and a fireball
          as bright as the Sun (about one a day, somewhere in view)
N5_100km  real speed: ~11 billion x normal: night as bright as an overcast day, fireballs every few minutes

Rates are the simulation's first-week averages (results/impact_ladder.json, storm_factor) applied to a
normal dark-sky rate of 10 visible meteors an hour. Debris from the Moon arrives at ~11 km/s, so the meteors are
slow and long, and they all radiate from one point in the sky (they share one direction of arrival).
Light: ~1% of the entry energy becomes visible light (meteor luminous efficiency), so the 10 km storm gives about
full-Moon light and the 100 km storm about overcast-daylight. The camera's exposure adapts like an eye (noted
on screen where it matters).
"""
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector, Quaternion, noise

import assets
import look
import shots
from shots import shot, smooth

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKY_R = 6000.0            # metres: the dome the streaks are drawn on (far beyond everything in the scene)
EYE = Vector((0.0, 0.0, 1.65))
NORMAL_PER_HOUR = 10.0


def ladder():
    rows = json.load(open(os.path.join(ROOT, 'results', 'impact_ladder.json')))['rows']
    return {r['D_m']: r for r in rows if r['material'] == 'rock'}


def sky_dir(az_deg, el_deg):
    """Unit vector for a compass azimuth (0 = north = +y, 90 = east = +x) and elevation."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))


# ------------------------------------------------------------------------------------ the place
def landscape(sc, town_glow=1.0):
    """A field on a gentle rise, trees, a few houses with lit windows, a town glow on the horizon."""
    me = bpy.data.meshes.new('Field')
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=160, y_segments=160, size=1.0)
    for v in bm.verts:
        u, w = v.co.x, v.co.y
        x = 3000.0 * u * abs(u) ** 0.6
        y = 3000.0 * w * abs(w) ** 0.6
        r = math.hypot(x, y)
        h = 14.0 * noise.fractal(Vector((x / 900.0, y / 900.0, 0.3)), 0.5, 2.0, 4) * min(1.0, r / 120.0)
        h -= r * r / (2 * 6371000.0)
        v.co = Vector((x, y, h - 0.4 * min(1.0, r / 60.0)))
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new('Field', me)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(assets.pbr('Grass004', 2.5, box=False, tint=(0.6, 0.65, 0.5)))
    rng = np.random.default_rng(12)
    # trees along hedgerows, silhouetted against the sky
    for k in range(40):
        az = rng.uniform(80, 260)
        d = rng.uniform(60, 400)
        x, y = d * math.sin(math.radians(az)), d * math.cos(math.radians(az))
        assets.tree(x, y, rng.uniform(7, 14), 'jacaranda_tree' if k % 3 == 0 else 'tree_small_02', rng.uniform(0, 6.3))
    # houses with a few lit windows
    lit = bpy.data.materials.new('LitWindow')
    lit.use_nodes = True
    em = lit.node_tree.nodes['Principled BSDF']
    em.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1)
    em.inputs['Emission Color'].default_value = (1.0, 0.62, 0.30, 1)
    em.inputs['Emission Strength'].default_value = 0.6
    lit.cycles.emission_sampling = 'NONE'
    for k in range(14):
        az = rng.uniform(110, 230)
        d = rng.uniform(250, 900)
        x, y = d * math.sin(math.radians(az)), d * math.cos(math.radians(az))
        hb = assets.building(x, y, rng.uniform(8, 14), rng.uniform(7, 11), rng.uniform(5, 8), 'Facade018A',
                             rot=rng.uniform(0, 3.1), roof=False)
        for j in range(rng.integers(1, 4)):
            w = assets.box_mesh('Win', 1.2, 0.2, 1.0)
            w.data.materials.append(lit)
            w.data.materials.append(lit)
            w.parent = hb
            w.location = (rng.uniform(-3, 3), -hb.dimensions.y / 2 - 0.05, rng.uniform(1.0, 4.0))
    return ob


def night_world(sc, stars=1.0, glow=0.0, glow_col=(0.62, 0.66, 0.78), town=0.25):
    """Stars (a real all-sky photograph), airglow and town glow near the horizon, and an optional all-sky
    glow (the light of unresolved meteors). Returns the node whose strength is the glow."""
    w = sc.world
    nt = w.node_tree
    nt.nodes.clear()
    N = look.node
    out = N(nt, 'ShaderNodeOutputWorld', (900, 0))
    tc = N(nt, 'ShaderNodeTexCoord', (-1100, 0))
    mp = N(nt, 'ShaderNodeMapping', (-900, 200))
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector'])
    env = N(nt, 'ShaderNodeTexEnvironment', (-700, 200))
    env.image = look.img(f'{look.ASSETS}/earth/8k_stars_milky_way.jpg')
    nt.links.new(mp.outputs['Vector'], env.inputs['Vector'])
    st = N(nt, 'ShaderNodeEmission', (-400, 300))
    nt.links.new(env.outputs['Color'], st.inputs['Color'])
    st.inputs['Strength'].default_value = 0.035 * stars
    # height in the sky: 0 at the horizon, 1 at the zenith
    sep = N(nt, 'ShaderNodeSeparateXYZ', (-900, -200))
    nt.links.new(tc.outputs['Generated'], sep.inputs[0])
    hz = N(nt, 'ShaderNodeMapRange', (-700, -200))
    nt.links.new(sep.outputs['Z'], hz.inputs['Value'])
    hz.inputs['From Min'].default_value = 0.0
    hz.inputs['From Max'].default_value = 0.35
    inv = N(nt, 'ShaderNodeMath', (-500, -200), operation='SUBTRACT')
    inv.inputs[0].default_value = 1.0
    nt.links.new(hz.outputs['Result'], inv.inputs[1])
    pw = N(nt, 'ShaderNodeMath', (-350, -200), operation='POWER')
    nt.links.new(inv.outputs[0], pw.inputs[0])
    pw.inputs[1].default_value = 2.5
    tg = N(nt, 'ShaderNodeEmission', (-150, -200))
    tg.inputs['Color'].default_value = (1.0, 0.55, 0.25, 1)
    tgm = N(nt, 'ShaderNodeMath', (-250, -350), operation='MULTIPLY')
    nt.links.new(pw.outputs[0], tgm.inputs[0])
    tgm.inputs[1].default_value = 0.010 * town
    nt.links.new(tgm.outputs[0], tg.inputs['Strength'])
    # the meteor glow: brighter towards the radiant side, otherwise even
    gl = N(nt, 'ShaderNodeEmission', (-150, -500))
    gl.inputs['Color'].default_value = tuple(glow_col) + (1,)
    gl.inputs['Strength'].default_value = glow + 0.0012          # + airglow
    a1 = N(nt, 'ShaderNodeAddShader', (200, 100))
    nt.links.new(st.outputs['Emission'], a1.inputs[0])
    nt.links.new(tg.outputs['Emission'], a1.inputs[1])
    a2 = N(nt, 'ShaderNodeAddShader', (450, 0))
    nt.links.new(a1.outputs['Shader'], a2.inputs[0])
    nt.links.new(gl.outputs['Emission'], a2.inputs[1])
    nt.links.new(a2.outputs['Shader'], out.inputs['Surface'])
    return gl, mp


# ------------------------------------------------------------------------------------ the meteors
def streak_material():
    m = bpy.data.materials.new('Streak')
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = look.node
    out = N(nt, 'ShaderNodeOutputMaterial', (800, 0))
    a_along = N(nt, 'ShaderNodeAttribute', (-800, 200))
    a_along.attribute_name = 'along'
    a_across = N(nt, 'ShaderNodeAttribute', (-800, 0))
    a_across.attribute_name = 'across'
    a_b = N(nt, 'ShaderNodeAttribute', (-800, -200))
    a_b.attribute_name = 'bright'
    a_col = N(nt, 'ShaderNodeAttribute', (-800, -400))
    a_col.attribute_name = 'tint'
    # head bright, tail fading; soft across the width
    p1 = N(nt, 'ShaderNodeMath', (-550, 200), operation='POWER')
    nt.links.new(a_along.outputs['Fac'], p1.inputs[0])
    p1.inputs[1].default_value = 2.2
    sq = N(nt, 'ShaderNodeMath', (-550, 0), operation='MULTIPLY')
    nt.links.new(a_across.outputs['Fac'], sq.inputs[0])
    nt.links.new(a_across.outputs['Fac'], sq.inputs[1])
    ex = N(nt, 'ShaderNodeMath', (-400, 0), operation='MULTIPLY')
    nt.links.new(sq.outputs[0], ex.inputs[0])
    ex.inputs[1].default_value = -3.5
    e2 = N(nt, 'ShaderNodeMath', (-250, 0), operation='EXPONENT')
    nt.links.new(ex.outputs[0], e2.inputs[0])
    m1 = N(nt, 'ShaderNodeMath', (-250, 200), operation='MULTIPLY')
    nt.links.new(p1.outputs[0], m1.inputs[0])
    nt.links.new(e2.outputs[0], m1.inputs[1])
    m2 = N(nt, 'ShaderNodeMath', (-100, 100), operation='MULTIPLY')
    nt.links.new(m1.outputs[0], m2.inputs[0])
    nt.links.new(a_b.outputs['Fac'], m2.inputs[1])
    # colour: white-green at the head, orange in the tail (slow meteors are orange)
    ramp = N(nt, 'ShaderNodeValToRGB', (-550, 450))
    ramp.color_ramp.elements[0].color = (1.0, 0.45, 0.15, 1)
    ramp.color_ramp.elements[1].color = (0.92, 1.0, 0.88, 1)
    nt.links.new(a_along.outputs['Fac'], ramp.inputs['Fac'])
    mix = N(nt, 'ShaderNodeMix', (-250, 450), data_type='RGBA', blend_type='MULTIPLY')
    mix.inputs['Factor'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], mix.inputs['A'])
    nt.links.new(a_col.outputs['Color'], mix.inputs['B'])
    em = N(nt, 'ShaderNodeEmission', (300, 100))
    nt.links.new(mix.outputs['Result'], em.inputs['Color'])
    nt.links.new(m2.outputs[0], em.inputs['Strength'])
    tr = N(nt, 'ShaderNodeBsdfTransparent', (300, -100))          # light only: the sky shows through
    add = N(nt, 'ShaderNodeAddShader', (550, 0))
    nt.links.new(em.outputs['Emission'], add.inputs[0])
    nt.links.new(tr.outputs['BSDF'], add.inputs[1])
    nt.links.new(add.outputs['Shader'], out.inputs['Surface'])
    m.cycles.emission_sampling = 'NONE'
    return m


class Meteors:
    """A deterministic list of meteors radiating from one point, drawn as soft streaks on a far dome."""

    def __init__(self, rate_per_s, seconds, radiant, seed=1, slow=1.0, bright_scale=1.0, sky_frac=1.0,
                 max_n=60000, frame_dir=None, frame_cone=None):
        rng = np.random.default_rng(seed)
        n = int(rate_per_s * seconds * sky_frac)
        self.scale = 1.0
        if n == 0:
            self.n = 0
            z = np.zeros(0)
            self.t0 = self.w = self.dur = self.b = self.tail = self.width = z
            self.p0 = self.m = self.tint = np.zeros((0, 3))
            return
        if n > max_n:                       # draw the brightest max_n; the rest go into the sky glow
            self.scale = n / max_n
            n = max_n
        self.n = n
        self.t0 = rng.uniform(-1.5, seconds, n)
        # start points over the visible sky (above ~5 degrees), favouring the camera's half of the sky
        dirs = []
        while len(dirs) < n:
            v = rng.normal(size=(n * 2, 3))
            v /= np.linalg.norm(v, axis=1)[:, None]
            ok = v[:, 2] > 0.06
            if frame_dir is not None:
                ok &= v @ np.array(frame_dir) > math.cos(frame_cone)
            dirs.extend(v[ok].tolist())
        p0 = np.array(dirs[:n])
        R = np.array(radiant)
        # motion: along the great circle away from the radiant
        m = p0 - R[None, :] * 0.0
        t = p0 - R[None, :]
        t -= (t * p0).sum(1)[:, None] * p0              # tangent at p0 pointing away from the radiant
        tn = np.linalg.norm(t, axis=1)
        t /= np.maximum(tn, 1e-6)[:, None]
        theta = np.arccos(np.clip(p0 @ R, -1, 1))       # angular distance from the radiant
        self.p0, self.m = p0, t
        # angular speed: slow (~11 km/s) meteors 100-200 km away: ~3-8 deg/s, foreshortened near the radiant
        self.w = np.radians(rng.uniform(3.0, 8.0, n)) * np.sin(theta) * slow
        self.dur = rng.uniform(0.6, 1.8, n)
        # brightness: power law (many faint, few bright)
        u = rng.uniform(1e-4, 1.0, n)
        self.b = np.minimum(u ** -0.9, 60.0) * bright_scale
        self.tail = rng.uniform(0.45, 0.95, n)
        self.width = np.radians(0.045 + 0.025 * np.log10(self.b / bright_scale + 1))
        tint = np.ones((n, 3))
        tint[:, 1] = rng.uniform(0.85, 1.0, n)
        tint[:, 2] = rng.uniform(0.7, 0.95, n)
        self.tint = tint

    def geometry(self, t, seg=5):
        """Vertices, quads and attributes for every meteor visible at time t."""
        if self.n == 0:
            return None
        a = (t >= self.t0) & (t <= self.t0 + self.dur + 0.4)
        idx = np.where(a)[0]
        if len(idx) == 0:
            return None
        tt = t - self.t0[idx]
        life = np.clip(tt / self.dur[idx], 0, 1)
        env = np.sin(np.pi * np.clip(life, 0, 1)) ** 0.7
        env[tt > self.dur[idx]] *= np.exp(-(tt[tt > self.dur[idx]] - self.dur[idx][tt > self.dur[idx]]) / 0.12)
        th_h = self.w[idx] * np.minimum(tt, self.dur[idx])
        th_t = np.maximum(0.0, th_h - self.w[idx] * self.tail[idx])
        p0, m = self.p0[idx], self.m[idx]
        k = len(idx)
        s = np.linspace(0, 1, seg + 1)
        th = th_t[:, None] + (th_h - th_t)[:, None] * s[None, :]            # k x (seg+1)
        P = np.cos(th)[..., None] * p0[:, None, :] + np.sin(th)[..., None] * m[:, None, :]
        side = np.cross(P, m[:, None, :])
        side /= np.maximum(np.linalg.norm(side, axis=2), 1e-9)[..., None]
        wdt = (self.width[idx] * SKY_R)[:, None, None] * (0.6 + 0.4 * s[None, :, None])
        V = np.concatenate([P * SKY_R + side * wdt, P * SKY_R - side * wdt], axis=1)     # k x 2(seg+1) x 3
        V = V + np.array(EYE)[None, None, :]
        along = np.concatenate([s, s])[None, :].repeat(k, 0)
        across = np.concatenate([np.ones(seg + 1), -np.ones(seg + 1)])[None, :].repeat(k, 0)
        bright = (self.b[idx] * env)[:, None].repeat(2 * (seg + 1), 1)
        tint = self.tint[idx][:, None, :].repeat(2 * (seg + 1), 1)
        base = (np.arange(k) * 2 * (seg + 1))[:, None]
        j = np.arange(seg)[None, :]
        q = np.stack([base + j, base + j + 1, base + seg + 1 + j + 1, base + seg + 1 + j], axis=2).reshape(-1, 4)
        return V.reshape(-1, 3), q, along.ravel(), across.ravel(), bright.ravel(), tint.reshape(-1, 3)


def streak_object(mat):
    me = bpy.data.meshes.new('Streaks')
    ob = bpy.data.objects.new('Streaks', me)
    bpy.context.collection.objects.link(ob)
    me.materials.append(mat)
    ob.visible_diffuse = False
    ob.visible_glossy = False
    ob.visible_shadow = False
    ob.visible_transmission = False
    ob.visible_volume_scatter = False
    return ob


def set_streaks(ob, geo):
    me = ob.data
    me.clear_geometry()
    for name in ('along', 'across', 'bright'):
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])
    if 'tint' in me.attributes:
        me.attributes.remove(me.attributes['tint'])
    if geo is None:
        me.update()
        return
    V, Q, along, across, bright, tint = geo
    me.vertices.add(len(V))
    me.vertices.foreach_set('co', V.astype(np.float32).ravel())
    me.loops.add(len(Q) * 4)
    me.loops.foreach_set('vertex_index', Q.astype(np.int32).ravel())
    me.polygons.add(len(Q))
    me.polygons.foreach_set('loop_start', (np.arange(len(Q)) * 4).astype(np.int32))
    me.update(calc_edges=True)
    for name, arr in (('along', along), ('across', across), ('bright', bright)):
        at = me.attributes.new(name, 'FLOAT', 'POINT')
        at.data.foreach_set('value', arr.astype(np.float32))
    at = me.attributes.new('tint', 'FLOAT_COLOR', 'POINT')
    at.data.foreach_set('color', np.concatenate([tint, np.ones((len(tint), 1))], 1).astype(np.float32).ravel())
    me.update()


# ------------------------------------------------------------------------------------ the shots
class NightShot:
    frames = 168
    look_az, look_el, lens = 150.0, 16.0, 20.0
    radiant = (125.0, 52.0)                 # azimuth, elevation of the radiant
    exposure = 4.2
    with_person = True

    def base(self, sc, glow=0.0, town=0.25, stars=1.0, moonlight=0.0):
        sc.cycles.samples = max(sc.cycles.samples, 8)
        gl = [n for n in sc.node_tree.nodes if n.type == 'GLARE']
        for g in gl:                                     # starlight-level points should not bloom
            g.inputs['Threshold'].default_value = 6.0
            g.inputs['Strength'].default_value = 0.6
        sc.view_settings.exposure = self.exposure
        self.glow, self.skymap = night_world(sc, stars=stars, glow=glow, town=town)
        landscape(sc)
        d = sky_dir(self.look_az, self.look_el)
        if self.with_person:
            h = Vector((d.x, d.y, 0)).normalized()
            left = Vector((-h.y, h.x, 0))
            pos = h * 2.4 + left * 0.85
            self.person = assets.human(pos.x, pos.y, 1.74, face_deg=180.0 - self.look_az, outfit=0)
        self.cam = look.camera(tuple(EYE), tuple(EYE + d * 10), self.lens)
        self.cam.data.clip_end = 20000.0
        if moonlight > 0:
            self.moon_lamp = look.sun(tuple(sky_dir(*self.moon_azel)), moonlight)
        self.streaks = streak_object(streak_material())

    def frame_cone(self):
        return tuple(sky_dir(self.look_az, self.look_el)), math.radians(62)


def moon_in_sky(az, el, sun_dir, name='NightMoon', dist=9000.0):
    """The Moon at its true angular size (0.52 degrees), lit by the Sun (below the horizon), with earthshine."""
    r = dist * math.tan(math.radians(0.26))
    d = sky_dir(az, el)
    ob = look.uv_sphere(name, r, 96, 48, loc=tuple(EYE + d * dist))
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    N = look.node
    out = N(nt, 'ShaderNodeOutputMaterial', (600, 0))
    geo = N(nt, 'ShaderNodeNewGeometry', (-800, 0))
    dot = N(nt, 'ShaderNodeVectorMath', (-600, 0), operation='DOT_PRODUCT')
    nt.links.new(geo.outputs['Normal'], dot.inputs[0])
    dot.inputs[1].default_value = tuple(Vector(sun_dir).normalized())
    lam = N(nt, 'ShaderNodeMath', (-400, 0), operation='MAXIMUM')
    nt.links.new(dot.outputs['Value'], lam.inputs[0])
    lam.inputs[1].default_value = 0.0
    tex = N(nt, 'ShaderNodeTexImage', (-600, 300))
    tex.image = look.img(f'{look.ASSETS}/moon/lroc_color_poles_8k.tif')
    uv = N(nt, 'ShaderNodeTexCoord', (-800, 300))
    nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
    lit = N(nt, 'ShaderNodeMath', (-250, 0), operation='MULTIPLY_ADD')
    nt.links.new(lam.outputs[0], lit.inputs[0])
    lit.inputs[1].default_value = 1.0
    lit.inputs[2].default_value = 0.004                      # earthshine on the dark side
    col = N(nt, 'ShaderNodeMix', (0, 200), data_type='RGBA', blend_type='MULTIPLY')
    col.inputs['Factor'].default_value = 1.0
    nt.links.new(tex.outputs['Color'], col.inputs['A'])
    col.inputs['B'].default_value = (1, 1, 1, 1)
    em = N(nt, 'ShaderNodeEmission', (300, 0))
    nt.links.new(col.outputs['Result'], em.inputs['Color'])
    s = N(nt, 'ShaderNodeMath', (100, -100), operation='MULTIPLY')
    nt.links.new(lit.outputs[0], s.inputs[0])
    s.inputs[1].default_value = 2.5
    nt.links.new(s.outputs[0], em.inputs['Strength'])
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    m.cycles.emission_sampling = 'NONE'
    ob.data.materials.append(m)
    ob.visible_diffuse = ob.visible_glossy = ob.visible_shadow = False
    # face the near side (the texture's centre) towards us
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = (-d).to_track_quat('X', 'Z')
    return ob, d, r


@shot('N1_1m', 144)
class Night1m(NightShot):
    """The Moon over the town, and the flash of a 1 m rock hitting its dark side (~mag 0, for ~0.1 s)."""
    frames = 144
    with_person = False
    look_az, look_el, lens = 160.0, 13.0, 40.0
    moon_azel = (165.0, 19.0)
    exposure = 3.2

    def build(self, sc):
        self.base(sc, town=0.35, moonlight=0.004)
        sun_dir = sky_dir(255.0, -12.0)
        self.moon, d, r = moon_in_sky(*self.moon_azel, sun_dir)
        # the flash: a tiny glowing point just above the dark side of the disc
        side = d.cross(Vector((0, 0, 1))).normalized()
        self.flash_pos = Vector(self.moon.location) - d * r * 1.02 - side * (r * 0.55) + Vector((0, 0, r * 0.2))
        self.flash = bpy.data.objects.new('Flash', bpy.data.meshes.new('Flash'))
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=r * 0.07)
        bm.to_mesh(self.flash.data)
        bm.free()
        bpy.context.collection.objects.link(self.flash)
        fm = bpy.data.materials.new('FlashMat')
        fm.use_nodes = True
        fm.node_tree.nodes.clear()
        e = fm.node_tree.nodes.new('ShaderNodeEmission')
        e.inputs['Color'].default_value = (1.0, 0.95, 0.85, 1)
        o = fm.node_tree.nodes.new('ShaderNodeOutputMaterial')
        fm.node_tree.links.new(e.outputs['Emission'], o.inputs['Surface'])
        fm.cycles.emission_sampling = 'NONE'
        self.flash.data.materials.append(fm)
        self.flash_em = e
        self.flash.location = self.flash_pos
        self.flash.visible_diffuse = self.flash.visible_glossy = self.flash.visible_shadow = False
        self.sporadic = Meteors(NORMAL_PER_HOUR / 3600.0, 7.0, tuple(sky_dir(60, 40)), seed=3)

    def frame(self, f):
        t = f / 24.0
        # the impact flash at t = 1.25 s: rises in a frame, fades in ~0.15 s
        k = f - 30
        self.flash_em.inputs['Strength'].default_value = 0.0 if k < 0 else 500.0 * math.exp(-k / 2.5)
        self.flash.hide_render = k < 0 or k > 14
        set_streaks(self.streaks, self.sporadic.geometry(t))
        return 0.0


class StormShot(NightShot):
    bright = 1.0
    rate = 0.0              # meteors per second over the whole sky
    seconds = 7.0
    speed = 1.0             # real time
    glow = 0.0
    seed = 7

    def build(self, sc):
        self.base(sc, glow=self.glow, town=0.25)
        cone = self.frame_cone()
        frac = 0.32                                      # share of the sky's meteors inside the wide view
        self.met = Meteors(self.rate * self.speed, self.seconds, tuple(sky_dir(*self.radiant)), seed=self.seed,
                           sky_frac=frac, frame_dir=cone[0], frame_cone=cone[1], bright_scale=self.bright,
                           slow=1.0)
        self.extra(sc)

    def extra(self, sc):
        pass

    def frame(self, f):
        t = f / 24.0
        set_streaks(self.streaks, self.met.geometry(t))
        self.more(f, t)
        return t * self.speed

    def more(self, f, t):
        pass


L = None


def rate(D):
    global L
    if L is None:
        L = ladder()
    return NORMAL_PER_HOUR / 3600.0 * L[D]['storm_factor']


@shot('N3_1km', 168)
class Night1km(StormShot):
    """Real speed, a few nights after a 1 km hit: ~30 shooting stars a second."""
    bright = 1.0

    def build(self, sc):
        self.rate = rate(1000.0)
        super().build(sc)


@shot('N4_10km', 168)
class Night10km(StormShot):
    """Real speed, after a 10 km hit: ~30,000 a second, about full-Moon light, and a fireball."""
    bright = 1.0
    glow = 0.006                              # the unresolved faint meteors: about full-Moon light in all

    def build(self, sc):
        self.rate = rate(1e4)
        super().build(sc)

    def extra(self, sc):
        # a fireball as bright as the Sun for a moment (Chelyabinsk class), crossing low from the left
        self.fb_lamp = look.sun(tuple(sky_dir(120, 20)), 0.0)
        self.fb_lamp.data.angle = math.radians(2.0)
        self.fb = Meteors(0, 1, tuple(sky_dir(*self.radiant)), seed=99)
        self.fb.n = 1
        p0 = sky_dir(100.0, 45.0)
        m0 = (sky_dir(140.0, 15.0) - p0)
        m0 = (m0 - p0 * m0.dot(p0)).normalized()
        self.fb.t0 = np.array([3.4])
        self.fb.p0, self.fb.m = np.array([tuple(p0)]), np.array([tuple(m0)])
        self.fb.w = np.array([math.radians(14.0)])
        self.fb.dur = np.array([2.2])
        self.fb.b = np.array([900.0])
        self.fb.tail = np.array([1.6])
        self.fb.width = np.array([math.radians(0.22)])
        self.fb.tint = np.array([[1.0, 1.0, 0.95]])
        self.fb.scale = 1.0
        self.fb_obj = streak_object(streak_material())

    def more(self, f, t):
        set_streaks(self.fb_obj, self.fb.geometry(t))
        x = (t - self.fb.t0[0]) / self.fb.dur[0]
        if 0 <= x <= 1.2:
            p = self.fb.p0[0] * math.cos(self.fb.w[0] * (t - self.fb.t0[0])) + \
                self.fb.m[0] * math.sin(self.fb.w[0] * (t - self.fb.t0[0]))
            shots.point_sun(self.fb_lamp, tuple(p))
            flare = math.sin(math.pi * min(x, 1.0)) ** 2 + (2.5 * math.exp(-((x - 0.85) / 0.05) ** 2))
            self.fb_lamp.data.energy = 0.012 * flare * (1.0 if x <= 1 else math.exp(-(x - 1) / 0.05))
        else:
            self.fb_lamp.data.energy = 0.0


@shot('N5_100km', 168)
class Night100km(StormShot):
    """Real speed, after a 100 km hit: night as bright as an overcast day (the eye adapts: 8 stops less
    exposure than the other night shots), streaks everywhere, fireballs every few minutes."""
    bright = 1.0
    exposure = 4.2 - 8.0                      # the eye adapts: 1,000x the light of the 10 km storm
    glow = 0.006 * 1000.0

    def build(self, sc):
        self.rate = rate(1e5)
        self.bright = 300.0
        super().build(sc)
