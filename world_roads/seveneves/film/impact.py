"""Impact ladder film: objects of increasing size hit the intact Moon (Blender 4.5, Cycles).

Physics from ../impact_ladder.py (results/impact_ladder.json): crater diameter and depth (Collins et al.
2005 with a strength cap), formation time, the ejecta speed distribution (Housen & Holsapple 2011).
Shots register into shots.SHOTS, so film/render.py renders them like the others.

Moon at the origin, object frame = texture frame (+x = sub-Earth point, +z = north). 1 unit = 1,000 km.
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector, Quaternion

import look
import shots
from shots import shot, smooth, lerp, place_camera, set_points, debris_object, make_earth, sky_heat_from_power

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = look.R_MOON
G_MOON = 1.62e-6           # units/s^2 (1.62 m/s^2)
V_ESC = 2.38e-3            # units/s
LADDER = json.load(open(os.path.join(ROOT, 'results', 'impact_ladder.json')))


def row(D):
    for r in LADDER['rows']:
        if r['material'] == 'rock' and abs(r['D_m'] - D) / D < 1e-6:
            return r
    raise KeyError(D)


def site_vec(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return Vector((math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)))


def impact_moon(site, name='Moon'):
    """The Moon with a crater term in its shader: radius/depth/rim/progress/melt/fresh driven by value nodes."""
    ob = look.moon((0, 0, 0), name=name)
    mat = ob.data.materials[0]
    nt = mat.node_tree
    N = look.node
    L = lambda a, b: nt.links.new(a, b)
    vals = {}
    for i, (k, v) in enumerate((('crater_R', 0.0), ('depth', 0.0), ('rim', 0.0), ('peak', 0.0),
                                ('melt', 0.0), ('fresh', 0.0), ('melt_T', 1800.0))):
        n = N(nt, 'ShaderNodeValue', (-1600, -900 - 120 * i))
        n.name = k
        n.outputs[0].default_value = v
        vals[k] = n.outputs[0]
    tc = nt.nodes['Texture Coordinate']
    nn = N(nt, 'ShaderNodeVectorMath', (-1400, -700), operation='NORMALIZE')
    L(tc.outputs['Object'], nn.inputs[0])
    sv = N(nt, 'ShaderNodeCombineXYZ', (-1400, -850))
    sv.label = 'keep'
    sv.inputs[0].default_value, sv.inputs[1].default_value, sv.inputs[2].default_value = tuple(site)
    dot = N(nt, 'ShaderNodeVectorMath', (-1200, -750), operation='DOT_PRODUCT')
    L(nn.outputs['Vector'], dot.inputs[0])
    L(sv.outputs[0], dot.inputs[1])
    cl = N(nt, 'ShaderNodeClamp', (-1050, -750))
    L(dot.outputs['Value'], cl.inputs['Value'])
    cl.inputs['Min'].default_value = -1.0
    ac = N(nt, 'ShaderNodeMath', (-900, -750), operation='ARCCOSINE')
    L(cl.outputs['Result'], ac.inputs[0])
    r = N(nt, 'ShaderNodeMath', (-750, -750), operation='MULTIPLY')
    L(ac.outputs['Value'], r.inputs[0])
    r.inputs[1].default_value = R
    x = N(nt, 'ShaderNodeMath', (-600, -750), operation='DIVIDE')       # r / crater radius
    L(r.outputs['Value'], x.inputs[0])
    safe = N(nt, 'ShaderNodeMath', (-750, -900), operation='MAXIMUM')
    L(vals['crater_R'], safe.inputs[0])
    safe.inputs[1].default_value = 1e-9
    L(safe.outputs['Value'], x.inputs[1])

    def M(op, a, b=None, loc=(0, 0)):
        n = N(nt, 'ShaderNodeMath', loc, operation=op)
        for i, s in enumerate((a, b)):
            if s is None:
                continue
            if isinstance(s, (int, float)):
                n.inputs[i].default_value = s
            else:
                L(s, n.inputs[i])
        return n.outputs['Value']
    x2 = M('MULTIPLY', x.outputs['Value'], x.outputs['Value'])
    # inside: paraboloid bowl from -depth at the centre to +rim at the edge, a flat floor for big ones,
    # and a central peak (complex craters)
    bowl = M('SUBTRACT', 1.0, x2)
    dr = M('ADD', vals['depth'], vals['rim'])
    hin = M('SUBTRACT', vals['rim'], M('MULTIPLY', dr, bowl))
    pk = M('MULTIPLY', vals['peak'], M('EXPONENT', M('MULTIPLY', x2, -40.0)))
    hin = M('ADD', hin, pk)
    # outside: the ejecta blanket thins as x^-3
    hout = M('MULTIPLY', vals['rim'], M('POWER', M('MAXIMUM', x.outputs['Value'], 1.0), -3.0))
    inside = M('LESS_THAN', x.outputs['Value'], 1.0)
    h = M('ADD', M('MULTIPLY', inside, hin), M('MULTIPLY', M('SUBTRACT', 1.0, inside), hout))
    # small craters below LOLA's ~2 km pixels: two scales of Voronoi bowls, faded in by 'detail'
    det = N(nt, 'ShaderNodeValue', (-1600, -1800))
    det.name = 'detail'
    det.outputs[0].default_value = 0.0
    hsmall = None
    for i, (sc_, amp) in enumerate(((300.0, 0.00028), (1200.0, 0.00008))):
        vo = N(nt, 'ShaderNodeTexVoronoi', (-600, -1900 - 200 * i))
        vo.feature = 'F1'
        vo.inputs['Scale'].default_value = sc_
        vo.inputs['Randomness'].default_value = 1.0
        L(tc.outputs['Object'], vo.inputs['Vector'])
        # each cell has a crater of random size (Voronoi colour), many cells none
        sepv = N(nt, 'ShaderNodeSeparateColor', (-450, -1900 - 200 * i))
        L(vo.outputs['Color'], sepv.inputs['Color'])
        rr = M('MULTIPLY', M('POWER', sepv.outputs[0], 2.5), 0.45)              # crater radius in cell units
        rr = M('MAXIMUM', rr, 1e-3)
        q = M('DIVIDE', vo.outputs['Distance'], rr)
        bowl_s = M('MAXIMUM', M('SUBTRACT', 1.0, M('MULTIPLY', q, q)), 0.0)
        rim_s = M('MULTIPLY', M('EXPONENT', M('MULTIPLY', M('MULTIPLY', M('SUBTRACT', q, 1.0), M('SUBTRACT', q, 1.0)), -25.0)), 0.25)
        term = M('MULTIPLY', M('MULTIPLY', M('SUBTRACT', rim_s, bowl_s), rr), amp / 0.2)
        hsmall = term if hsmall is None else M('ADD', hsmall, term)
    nz = N(nt, 'ShaderNodeTexNoise', (-600, -2350))
    nz.inputs['Scale'].default_value = 900.0
    nz.inputs['Detail'].default_value = 6.0
    L(tc.outputs['Object'], nz.inputs['Vector'])
    hsmall = M('ADD', hsmall, M('MULTIPLY', M('SUBTRACT', nz.outputs['Fac'], 0.5), 0.0003))
    # the new crater wipes out older small craters inside it
    keep = M('MINIMUM', M('MULTIPLY', x.outputs['Value'], 1.0), 1.0)
    h = M('ADD', h, M('MULTIPLY', M('MULTIPLY', hsmall, det.outputs[0]), keep))
    # add to the LOLA displacement already on the output
    out = nt.nodes['Material Output']
    disp = nt.nodes['Displacement']
    old_h = disp.inputs['Height'].links[0].from_socket
    hk = M('MULTIPLY', h, 1000.0)                  # the LOLA input is in km with scale 0.001
    L(M('ADD', old_h, hk), disp.inputs['Height'])
    # fresh ejecta: brighter near the crater, in rays; impact melt glowing on the floor
    e1 = Vector((0, 0, 1)).cross(site)
    if e1.length < 1e-6:
        e1 = Vector((1, 0, 0))
    e1.normalize()
    e2 = site.cross(e1)
    c1 = N(nt, 'ShaderNodeCombineXYZ', (-1400, -1500))
    c1.label = 'keep'
    c1.inputs[0].default_value, c1.inputs[1].default_value, c1.inputs[2].default_value = tuple(e1)
    c2 = N(nt, 'ShaderNodeCombineXYZ', (-1400, -1650))
    c2.label = 'keep'
    c2.inputs[0].default_value, c2.inputs[1].default_value, c2.inputs[2].default_value = tuple(e2)
    d1 = N(nt, 'ShaderNodeVectorMath', (-1200, -1500), operation='DOT_PRODUCT')
    L(nn.outputs['Vector'], d1.inputs[0])
    L(c1.outputs[0], d1.inputs[1])
    d2 = N(nt, 'ShaderNodeVectorMath', (-1200, -1650), operation='DOT_PRODUCT')
    L(nn.outputs['Vector'], d2.inputs[0])
    L(c2.outputs[0], d2.inputs[1])
    az = M('ARCTAN2', d2.outputs['Value'], d1.outputs['Value'])
    lx = M('LOGARITHM', M('MAXIMUM', x.outputs['Value'], 0.05), 2.718)
    cv = N(nt, 'ShaderNodeCombineXYZ', (-700, -1550))
    L(M('MULTIPLY', az, 9.0), cv.inputs['X'])
    L(M('MULTIPLY', lx, 0.6), cv.inputs['Y'])
    rays = N(nt, 'ShaderNodeTexNoise', (-500, -1550))
    rays.inputs['Scale'].default_value = 1.6
    rays.inputs['Detail'].default_value = 5.0
    L(cv.outputs['Vector'], rays.inputs['Vector'])
    rayk = M('POWER', M('MAXIMUM', M('SUBTRACT', rays.outputs['Fac'], 0.42), 0.0), 0.8)
    near = M('EXPONENT', M('MULTIPLY', M('ABSOLUTE', M('SUBTRACT', x.outputs['Value'], 1.0)), -3.0))   # rim zone
    outside = M('GREATER_THAN', x.outputs['Value'], 1.0)
    bright = M('ADD', M('MULTIPLY', near, 0.35),
               M('MULTIPLY', M('MULTIPLY', rayk, outside), M('MULTIPLY', M('POWER', M('MAXIMUM', x.outputs['Value'], 1.0), -1.0), 1.4)))
    bright = M('MULTIPLY', bright, vals['fresh'])
    dif = nt.nodes['Diffuse BSDF']
    src = dif.inputs['Color'].links[0].from_socket
    mixc = N(nt, 'ShaderNodeMix', (200, 600), data_type='RGBA', blend_type='SCREEN')
    L(M('MINIMUM', bright, 0.9), mixc.inputs['Factor'])
    L(src, mixc.inputs['A'])
    mixc.inputs['B'].default_value = (0.82, 0.80, 0.77, 1)
    L(mixc.outputs['Result'], dif.inputs['Color'])
    bb = N(nt, 'ShaderNodeBlackbody', (300, -1000))
    L(vals['melt_T'], bb.inputs['Temperature'])
    mn = N(nt, 'ShaderNodeTexNoise', (100, -1250))
    mn.inputs['Scale'].default_value = 140.0
    mn.inputs['Detail'].default_value = 8.0
    L(tc.outputs['Object'], mn.inputs['Vector'])
    patchy = M('POWER', M('MAXIMUM', M('SUBTRACT', mn.outputs['Fac'], 0.38), 0.0), 1.5)
    floor = M('MULTIPLY', M('SUBTRACT', 1.0, M('MINIMUM', M('MULTIPLY', x2, 3.0), 1.0)), vals['melt'])
    floor = M('MULTIPLY', floor, M('MULTIPLY', patchy, 6.0))
    em = N(nt, 'ShaderNodeEmission', (500, -1000))
    L(bb.outputs['Color'], em.inputs['Color'])
    L(floor, em.inputs['Strength'])
    add = N(nt, 'ShaderNodeAddShader', (700, 0))
    L(dif.outputs['BSDF'], add.inputs[0])
    L(em.outputs['Emission'], add.inputs[1])
    L(add.outputs['Shader'], out.inputs['Surface'])
    mat.cycles.emission_sampling = 'NONE'
    return ob, nt


def fireball(name='Fireball'):
    me = bpy.data.meshes.new(name)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    N = look.node
    out = N(nt, 'ShaderNodeOutputMaterial', (600, 0))
    T = N(nt, 'ShaderNodeValue', (-400, 100))
    T.name = 'T'
    S = N(nt, 'ShaderNodeValue', (-400, -100))
    S.name = 'S'
    bb = N(nt, 'ShaderNodeBlackbody', (-200, 100))
    nt.links.new(T.outputs[0], bb.inputs['Temperature'])
    lw = N(nt, 'ShaderNodeLayerWeight', (-400, -300))
    lw.inputs['Blend'].default_value = 0.5
    soft = N(nt, 'ShaderNodeMath', (-200, -300), operation='SUBTRACT')
    soft.inputs[0].default_value = 1.0
    nt.links.new(lw.outputs['Facing'], soft.inputs[1])
    soft3 = N(nt, 'ShaderNodeMath', (-100, -300), operation='POWER')      # soft, gaseous edge
    nt.links.new(soft.outputs['Value'], soft3.inputs[0])
    soft3.inputs[1].default_value = 3.0
    s2 = N(nt, 'ShaderNodeMath', (0, -200), operation='MULTIPLY')
    nt.links.new(soft3.outputs['Value'], s2.inputs[0])
    nt.links.new(S.outputs[0], s2.inputs[1])
    em = N(nt, 'ShaderNodeEmission', (200, 0))
    nt.links.new(bb.outputs['Color'], em.inputs['Color'])
    nt.links.new(s2.outputs['Value'], em.inputs['Strength'])
    tr = N(nt, 'ShaderNodeBsdfTransparent', (200, -200))
    add = N(nt, 'ShaderNodeAddShader', (400, 0))
    nt.links.new(em.outputs['Emission'], add.inputs[0])
    nt.links.new(tr.outputs['BSDF'], add.inputs[1])
    nt.links.new(add.outputs['Shader'], out.inputs['Surface'])
    mat.cycles.emission_sampling = 'NONE'
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    return ob


class Ejecta:
    """Ballistic ejecta curtain. Material leaves from radius x in the growing transient crater, at the
    Housen-Holsapple speed for that radius, 45 degrees outward; it then falls under lunar gravity
    (flat-Moon ballistics near the crater, straight lines for the fast escaping part)."""

    def __init__(self, site, a_units, rc_units, U_units, t_form, n=60000, seed=5):
        rng = np.random.default_rng(seed)
        mu, C1 = 0.55, 1.5
        self.site = np.array(site)
        e1 = np.cross([0, 0, 1.0], self.site)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.site, e1)
        # launch radius distributed by mass (most mass from far out, fastest from near the centre)
        u = rng.random(n)
        xr = a_units * (1 + u ** 0.6 * (rc_units / a_units - 1))
        v = np.minimum(C1 * U_units * (xr / a_units) ** (-1 / mu), 0.25 * U_units)
        az = rng.random(n) * 2 * np.pi
        tilt = np.radians(45 + rng.normal(0, 6, n))
        dirh = np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2
        self.p0 = self.site * R + xr[:, None] * dirh
        self.vel = v[:, None] * (np.cos(tilt)[:, None] * dirh + np.sin(tilt)[:, None] * self.site)
        self.t0 = t_form * (xr / rc_units) ** 2.0
        self.hot = np.clip((v - 1e-3) / 4e-3, 0, 1)            # fast ejecta = melt, glowing
        self.size = rng.lognormal(0, 0.5, n)
        self.ob = debris_object('Ejecta', (0.55, 0.53, 0.50))
        self.ob.visible_shadow = True

    def update(self, t, cam_loc, px=0.00045):
        dt = t - self.t0
        live = dt > 0
        if not live.any():
            set_points(self.ob, np.zeros((0, 3)), 0)
            return
        d = dt[live][:, None]
        p = self.p0[live] + self.vel[live] * d - 0.5 * G_MOON * self.site * d ** 2
        # landed: on or below the surface (radius < R) -> drop
        up = (p - self.site * R) @ self.site > -1e-4
        p = p[up]
        dist = np.linalg.norm(p - np.array(cam_loc), axis=1)
        rad = px * dist * self.size[live][up]
        heat = self.hot[live][up] * np.exp(-dt[live][up] / 60.0)
        set_points(self.ob, p, rad, 6.0 * heat)


class ImpactShot:
    """One impact, close up. Subclasses set D (m), site, sun, camera and time mapping."""
    D = 1000.0
    lat, lon = 12.0, -14.0
    sun_elev = 18.0
    cam_alt = 0.25            # units above the surface
    cam_back = 0.6            # units back from the site along the surface
    lens = 35
    t_end_factor = 2.5        # show this many formation times
    n_eject = 50000

    def build(self, sc):
        r = row(self.D)
        self.r = r
        self.site = site_vec(self.lat, self.lon)
        # Sun low in the east of the site
        east = Vector((0, 0, 1)).cross(self.site).normalized()
        el = math.radians(self.sun_elev)
        self.sun_dir = (self.site * math.sin(el) + east * math.cos(el)).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        self.moon, self.nt = impact_moon(self.site)
        self.Rc = r['crater_km'] / 2 / 1000.0
        self.depth = r['crater_depth_km'] / 1000.0
        self.tf = r['crater_time_s']
        a = self.D / 2 / 1e6
        self.ej = Ejecta(self.site, a, self.Rc / 1.25, 20e3 * 1e-6 * math.sin(math.radians(45)), self.tf,
                         n=self.n_eject)
        self.fb = fireball()
        north = self.site.cross(east)
        self.cam_loc = self.site * (R + self.cam_alt) - north * self.cam_back
        self.cam = look.camera(tuple(self.cam_loc), tuple(self.site * R), self.lens)
        self.nt.nodes['detail'].outputs[0].default_value = 1.0 if self.cam_alt < 0.5 else 0.0

    def time(self, f):
        x = f / (self.frames - 1)
        return self.tf * self.t_end_factor * x ** 1.6         # slow at first, then faster

    def frame(self, f):
        t = self.time(f)
        g = smooth(t / self.tf)
        nodes = self.nt.nodes
        nodes['crater_R'].outputs[0].default_value = self.Rc * max(g, 1e-3) ** 0.5
        nodes['depth'].outputs[0].default_value = self.depth * g
        nodes['rim'].outputs[0].default_value = 0.04 * 2 * self.Rc * g
        nodes['peak'].outputs[0].default_value = (0.25 * self.depth * g) if self.Rc * 2000 > 18 else 0.0
        nodes['fresh'].outputs[0].default_value = g
        nodes['melt'].outputs[0].default_value = 0.5 * g * math.exp(-t / (4 * self.tf))
        # fireball: vapour plume expanding at ~ a few km/s, cooling from ~10,000 K
        rb = min((self.D / 2e6) * 3 + 2.5e-3 * t, 1.6 * self.Rc)        # vapour plume, ~2.5 km/s
        self.fb.location = self.site * (R + 0.3 * rb)
        self.fb.scale = (rb, rb, rb)
        fm = self.fb.data.materials[0].node_tree.nodes
        fm['T'].outputs[0].default_value = 2000 + 7000 * math.exp(-t / (0.03 * self.tf))
        fm['S'].outputs[0].default_value = 25.0 * math.exp(-t / (0.012 * self.tf))
        self.fb.hide_render = fm['S'].outputs[0].default_value < 0.02
        self.ej.update(t, self.cam_loc)
        # a slow drift of the camera for life
        place_camera(self.cam, self.cam_loc + Vector(self.site) * 0.02 * f / self.frames, self.site * R)
        return t


@shot('I3_1km', 144)
class Impact1km(ImpactShot):
    D = 1000.0
    cam_alt, cam_back, lens = 0.035, 0.065, 30


@shot('I4_10km', 144)
class Impact10km(ImpactShot):
    D = 10000.0
    cam_alt, cam_back, lens = 0.25, 0.45, 30


@shot('I5_100km', 168)
class Impact100km(ImpactShot):
    D = 100000.0
    lat, lon = 8.0, -8.0
    cam_alt, cam_back, lens = 2.6, 2.4, 35
    n_eject = 70000


class FlashShot:
    """Small impacts seen from the Earth's distance: a flash on the Moon's night side."""
    D = 1.0
    peak = 1.0

    def build(self, sc):
        self.r = row(self.D)
        to_earth = Vector((1, 0, 0))
        # thin crescent: Sun nearly behind the Moon, a little to the side
        self.sun_dir = Vector((-0.75, 0.66, 0.05)).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        # earthshine: sunlight reflected off the Earth dimly lights the Moon's night side (as in long exposures)
        es = look.sun(tuple(to_earth), 0.15)
        es.data.color = (0.75, 0.85, 1.0)
        self.moon = look.moon((0, 0, 0))
        site = site_vec(18.0, -20.0)
        self.fb = fireball()
        self.fb.location = site * (R + 0.002)
        self.cam = look.camera(tuple(to_earth * 384.0), (0, 0, 0), 2400)    # a big telescope

    def frame(self, f):
        t = (f - 30) / 24.0                            # the impact at frame 30
        s = 0.0 if t < 0 else self.peak * math.exp(-t / 0.06)
        r = 0.006 * (1 + 4 * max(t, 0))
        self.fb.scale = (r, r, r)
        nodes = self.fb.data.materials[0].node_tree.nodes
        nodes['T'].outputs[0].default_value = 3500
        nodes['S'].outputs[0].default_value = s
        self.fb.hide_render = s < 1e-3
        return max(t, 0.0)


@shot('I1_1m', 96)
class Flash1m(FlashShot):
    D, peak = 1.0, 400.0


@shot('I2_60m', 96)
class Flash60m(FlashShot):
    D, peak = 60.0, 4000.0


class EarthAfter:
    """The Earth over the months after an impact on the Moon: meteors and fireballs where the ejecta
    arrive (the arrival-time distribution from the 6,000 simulated trajectories)."""
    D = 1000.0
    days = 60.0

    def build(self, sc):
        self.r = row(self.D)
        e = np.load(os.path.join(ROOT, 'out', 'lunar_impact', 'ejecta.npz'))
        # escaping-ejecta mass per log speed (rock, 20 km/s, 45 deg; as lunar_impact.speed_weights)
        Uv = 20e3 * math.sin(math.radians(45))
        vmax = 1.5 * Uv * 1.2 ** (-1 / 0.55)
        sys_w = np.where(e['v'] < vmax, (e['v'] / (1.5 * Uv)) ** (-3 * 0.55), 0.0)
        hit = e['fate'] == 1
        self.arr = (e['t_f'][hit] - e['t0'][hit])
        w = sys_w[hit]
        self.w = w / w.sum()
        self.M = self.r['to_earth_kg']
        self.sun_dir = Vector((1.0, -0.35, 0.12)).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        self.earth, self.shell = make_earth(tuple(self.sun_dir), spin_deg=120)
        view = (-self.sun_dir * 0.8 + Vector((0, 0, 1)).cross(self.sun_dir).normalized() * 0.55 + Vector((0, 0, 0.2))).normalized()
        self.cam = look.camera(tuple(view * 26.0), (0, 0, 0), 50)
        self.view = view
        self.met = debris_object('Meteors', (0.2, 0.2, 0.2))
        self.rng = np.random.default_rng(int(self.D) % 1000)

    def frame(self, f):
        t = shots.logtime(f, 0, self.frames - 1, 0.5 * 86400, self.days * 86400)
        # kg per day arriving now (+-25%)
        lo, hi = t * 0.8, t * 1.25
        sel = (self.arr > lo) & (self.arr < hi)
        rate = self.M * self.w[sel].sum() / ((hi - lo) / 86400)
        storm = rate / LADDER['background_kg_day']
        n = int(min(900, 6 * math.log10(1 + storm) ** 2.2))
        pts, br = np.zeros((0, 3)), np.zeros(0)
        if n:
            base = self.rng.normal(size=(n, 3))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            tang = np.cross(base, self.rng.normal(size=base.shape))
            tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            Ls = self.rng.uniform(0.04, 0.2, n)
            k = np.linspace(0, 1, 8)
            pts = (base[:, None, :] * (6.371 + 0.11 - 0.05 * k[None, :, None])
                   + tang[:, None, :] * (Ls[:, None] * (1 - k))[:, :, None]).reshape(-1, 3)
            big = self.rng.random(n) < min(0.05, 0.002 * math.log10(1 + storm))
            b0 = self.rng.lognormal(0, 0.7, n) * np.where(big, 25.0, 1.0)
            br = ((1.0 + 7.0 * k ** 2)[None, :] * b0[:, None]).reshape(-1)
        set_points(self.met, pts, 0.004, br * 6.0)
        P = rate * 0.5 * 11.1e3 ** 2 / 86400 / 5.1e14
        T, glow = sky_heat_from_power(max(P, 1.0))
        nt = self.shell.data.materials[0].node_tree
        nt.nodes['sky_T'].outputs[0].default_value = T
        nt.nodes['sky_glow'].outputs[0].default_value = glow
        place_camera(self.cam, self.view * lerp(26.0, 23.0, smooth(f / (self.frames - 1))), (0, 0, 0))
        return t


@shot('E3_1km', 120)
class After1km(EarthAfter):
    D, days = 1000.0, 120.0


@shot('E4_10km', 120)
class After10km(EarthAfter):
    D, days = 10000.0, 365.0


@shot('E5_100km', 120)
class After100km(EarthAfter):
    D, days = 100000.0, 365.0 * 3


@shot('J_ejecta', 240)
class EjectaFlow:
    """One impact's escaping ejecta, 1 hour to 120 days: the simulated trajectories (ejecta_viz.py),
    thrown from the Moon's trailing side. Most leaves the Earth-Moon system; some falls to the Earth."""

    def build(self, sc):
        d = np.load(os.path.join(ROOT, 'out', 'film_data', 'ejecta_paths.npz'))
        self.d = {k: d[k] for k in d.files}
        s0 = self.d['sun'][0]
        self.sun_dir = Vector(s0).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        self.earth, self.shell = make_earth(tuple(self.sun_dir), spin_deg=40)
        self.moon = look.moon((0, 0, 0), name='MoonFar')
        self.moon.rotation_mode = 'QUATERNION'
        self.pts = debris_object('EjectaFlow', (0.62, 0.60, 0.57))
        self.hot = debris_object('EjectaToEarth', (0.9, 0.55, 0.25))      # the ones the simulation sends to Earth
        self.to_earth = self.d['fate'] == 1
        m0 = Vector(self.d['moon'][0])
        mid = m0 * 0.55
        up = Vector((0, 0, 1))
        side = m0.normalized().cross(up).normalized()
        self.cam_loc = mid + up * 420.0 + side * 330.0
        self.cam = look.camera(tuple(self.cam_loc), tuple(mid), 32)
        self.mid = mid

    def frame(self, f):
        d = self.d
        t = shots.logtime(f, 0, self.frames - 1, 1800.0, d['t'][-1])
        k = int(np.clip(np.searchsorted(d['t'], t) - 1, 0, len(d['t']) - 2))
        x = (t - d['t'][k]) / (d['t'][k + 1] - d['t'][k])
        mp = d['moon'][k] * (1 - x) + d['moon'][k + 1] * x
        self.moon.location = Vector(mp)
        xh = Vector(-mp).normalized()
        zh = (Vector((0, 0, 1)) - xh * xh.z).normalized()
        yh = zh.cross(xh)
        from mathutils import Matrix
        self.moon.rotation_quaternion = Matrix((xh, yh, zh)).transposed().to_quaternion()
        # each particle with a short trail back along its path (8 samples over the last ~6 frames' time)
        tr_t = [t * (1 - 0.06 * j) for j in range(8)]
        cam = np.array(self.cam_loc)
        out = {False: ([], [], []), True: ([], [], [])}
        for j, tj in enumerate(tr_t):
            kj = int(np.clip(np.searchsorted(d['t'], tj) - 1, 0, len(d['t']) - 2))
            xj = (tj - d['t'][kj]) / (d['t'][kj + 1] - d['t'][kj])
            Pj = d['P'][kj] * (1 - xj) + d['P'][kj + 1] * xj
            ok = ~np.isnan(Pj[:, 0])
            for hot in (False, True):
                sel = ok & (self.to_earth == hot)
                p = Pj[sel]
                dist = np.linalg.norm(p - cam, axis=1)
                out[hot][0].append(p)
                out[hot][1].append(0.00045 * dist * (1 - 0.08 * j))
                out[hot][2].append(np.full(len(p), (0.0 if not hot else 2.5) * (1 - 0.1 * j)))
        for hot, ob in ((False, self.pts), (True, self.hot)):
            p = np.concatenate(out[hot][0]) if out[hot][0] else np.zeros((0, 3))
            set_points(ob, p, np.concatenate(out[hot][1]), np.concatenate(out[hot][2]))
        sd = Vector(d['sun'][k]).normalized()
        shots.point_sun(self.sun, sd)
        shots.set_sun_vectors(self.earth, self.shell, tuple(sd))
        return t


@shot('K_entry', 144)
class Entry:
    """A piece of Moon rock hitting the Earth's atmosphere at 11+ km/s, seen from 400 km up at night.
    It lights up ~100 km high, heats the air around it, and explodes in the lower atmosphere."""

    def build(self, sc):
        self.sun_dir = Vector((1.0, 0.2, 0.1)).normalized()
        self.sun = shots.sun_lamp(tuple(self.sun_dir))
        self.earth, self.shell = make_earth(tuple(self.sun_dir), spin_deg=95)
        # camera 400 km above a night-side point, looking towards the horizon
        below = (-self.sun_dir + Vector((0, 0, 0.55))).normalized()
        east = Vector((0, 0, 1)).cross(below).normalized()
        north = below.cross(east)
        self.below, self.east, self.north = below, east, north
        self.cam_loc = below * (look.R_EARTH + 0.40)
        tgt = below * (look.R_EARTH + 0.06) + north * 0.30 + east * 0.02
        self.cam = look.camera(tuple(self.cam_loc), tuple(tgt), 30)
        self.tgt = tgt
        self.trail = debris_object('Fireball', (0.1, 0.1, 0.1))
        self.trail.visible_shadow = False
        lamp = bpy.data.lights.new('FireLight', 'POINT')
        lamp.shadow_soft_size = 0.002
        self.lamp = bpy.data.objects.new('FireLight', lamp)
        bpy.context.collection.objects.link(self.lamp)
        # entry path: from 120 km up, 35 degrees below horizontal, 12 km/s, across the view
        self.start = below * look.R_EARTH + north * 0.30 - east * 0.10
        self.p0 = self.start.normalized() * (look.R_EARTH + 0.12)
        self.dirv = (east * math.cos(math.radians(35)) - self.p0.normalized() * math.sin(math.radians(35))).normalized()

    def frame(self, f):
        t = f / 24.0 - 0.6
        if t < 0:
            set_points(self.trail, np.zeros((0, 3)), 0)
            self.lamp.data.energy = 0.0
            return 0.0
        v = 0.012                                       # units/s = 12 km/s
        tt = np.linspace(max(0.0, t - 2.0), t, 120)
        pts = np.array([tuple(self.p0 + self.dirv * (v * s)) for s in tt])
        alt = np.linalg.norm(pts, axis=1) - look.R_EARTH
        head = alt[-1]
        # brighter as it goes deeper into thicker air; it bursts below ~30 km
        burst = head < 0.032
        age = t - tt
        b = np.exp(-age / 0.5) * np.clip((0.12 - alt) / 0.06, 0, 1) ** 2 * 60.0
        if burst:
            b *= np.exp(-(t - (0.12 - 0.032) / (v * math.sin(math.radians(35)))) / 0.4)
        rad = 0.0006 + 0.0015 * np.exp(-age / 0.3)
        set_points(self.trail, pts, rad, b)
        self.lamp.location = Vector(tuple(pts[-1]))
        self.lamp.data.energy = float(b[-1]) * 2e-5     # a big fireball lights the ground at ~1/450 of sunlight
        self.lamp.data.color = (1.0, 0.85, 0.65)
        place_camera(self.cam, self.cam_loc, self.tgt + self.east * 0.0004 * f)
        return t
