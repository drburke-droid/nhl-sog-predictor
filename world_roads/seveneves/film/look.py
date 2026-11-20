"""Blender (4.5) look-development for the Seveneves film: photoreal Earth and Moon in Cycles.

Units: 1 Blender unit = 1,000 km. Earth at the origin, radius 6.371. Sun light along SUN_DIR.
Assets (not in the repository; see film/README.md for where they come from):
  /opt/seveneves_assets/moon/lroc_color_poles_8k.tif, ldem_16.tif   (NASA SVS CGI Moon Kit, public domain)
  /opt/seveneves_assets/earth/8k_earth_*.{jpg,tif}                    (Solar System Scope, CC BY 4.0)
"""
import math
import os

import bpy
import bmesh
from mathutils import Vector

ASSETS = os.environ.get('SEVENEVES_ASSETS', '/opt/seveneves_assets')
R_EARTH = 6.371
R_MOON = 1.7374


def img(path, colorspace='sRGB'):
    name = os.path.basename(path)
    im = bpy.data.images.get(name) or bpy.data.images.load(path)
    im.colorspace_settings.name = colorspace
    return im


def node(nt, kind, loc=(0, 0), **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def link(nt, a, b):
    nt.links.new(a, b)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.feature_set = 'EXPERIMENTAL'          # adaptive subdivision for the Moon's real terrain
    sc.cycles.samples = 24
    sc.cycles.use_light_tree = False                # one sun; the tree only costs time
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.cycles.max_bounces = 3
    sc.cycles.diffuse_bounces = 1                   # one bounce: earthshine on the Moon's night side
    sc.cycles.glossy_bounces = 1
    sc.cycles.transmission_bounces = 2
    sc.cycles.volume_bounces = 1
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.volume_step_rate = 4.0
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.unit_settings.system = 'NONE'
    w = bpy.data.worlds.new('space')
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0, 0, 0, 1)
    bg.inputs['Strength'].default_value = 0.0
    sc.world = w
    return sc


def add_stars(strength=0.004):
    """Faint real star field (a camera exposed for a sunlit planet barely records stars)."""
    w = bpy.context.scene.world
    nt = w.node_tree
    tex = node(nt, 'ShaderNodeTexEnvironment', (-400, 300))
    tex.image = img(f'{ASSETS}/earth/8k_stars_milky_way.jpg')
    bg = nt.nodes['Background']
    link(nt, tex.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = strength


def sun(direction, strength=6.0):
    """direction: unit vector pointing from the scene towards the Sun."""
    d = Vector(direction).normalized()
    lamp = bpy.data.lights.new('Sun', 'SUN')
    lamp.energy = strength
    lamp.angle = math.radians(0.53)
    ob = bpy.data.objects.new('Sun', lamp)
    bpy.context.collection.objects.link(ob)
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = (-d).to_track_quat('-Z', 'Y')
    return ob


def uv_sphere(name, radius, seg=256, rings=128, loc=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm.loops.layers.uv.new('UVMap')                 # create_uvsphere only fills an existing UV layer
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=radius, calc_uvs=True)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    bpy.context.collection.objects.link(ob)
    return ob


def _sun_mask(nt, sun_dir, loc, lo=-0.12, hi=0.08, use_position=False):
    """0 on the day side, 1 on the night side, smooth across the terminator. use_position: take the
    direction from the planet centre (world origin) instead of the shading normal, which Cycles flips
    on back faces."""
    geo = node(nt, 'ShaderNodeNewGeometry', (loc[0] - 600, loc[1]))
    sd = node(nt, 'ShaderNodeCombineXYZ', (loc[0] - 600, loc[1] - 200))
    sd.inputs[0].default_value, sd.inputs[1].default_value, sd.inputs[2].default_value = sun_dir
    dot = node(nt, 'ShaderNodeVectorMath', (loc[0] - 400, loc[1]), operation='DOT_PRODUCT')
    if use_position:
        pn = node(nt, 'ShaderNodeVectorMath', (loc[0] - 500, loc[1] + 100), operation='NORMALIZE')
        link(nt, geo.outputs['Position'], pn.inputs[0])
        link(nt, pn.outputs['Vector'], dot.inputs[0])
    else:
        link(nt, geo.outputs['Normal'], dot.inputs[0])
    link(nt, sd.outputs[0], dot.inputs[1])
    mr = node(nt, 'ShaderNodeMapRange', (loc[0] - 200, loc[1]), interpolation_type='SMOOTHSTEP')
    mr.inputs['From Min'].default_value = hi
    mr.inputs['From Max'].default_value = lo
    link(nt, dot.outputs['Value'], mr.inputs['Value'])
    return mr.outputs['Result'], sd


def earth(sun_dir, name='Earth'):
    ob = uv_sphere(name, R_EARTH, 384, 192)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    bsdf = node(nt, 'ShaderNodeBsdfPrincipled', (500, 0))
    uv = node(nt, 'ShaderNodeTexCoord', (-1200, 0))
    day = node(nt, 'ShaderNodeTexImage', (-800, 300))
    day.image = img(f'{ASSETS}/earth/8k_earth_daymap.jpg')
    spec = node(nt, 'ShaderNodeTexImage', (-800, 0))
    spec.image = img(f'{ASSETS}/earth/8k_earth_specular_map.tif', 'Non-Color')
    nrm = node(nt, 'ShaderNodeTexImage', (-800, -300))
    nrm.image = img(f'{ASSETS}/earth/8k_earth_normal_map.tif', 'Non-Color')
    night = node(nt, 'ShaderNodeTexImage', (-800, -600))
    night.image = img(f'{ASSETS}/earth/8k_earth_nightmap.jpg')
    for t in (day, spec, nrm, night):
        link(nt, uv.outputs['UV'], t.inputs['Vector'])
    # slightly deepen the oceans and lift contrast like orbital photographs
    cc = node(nt, 'ShaderNodeBrightContrast', (-500, 300))
    cc.inputs['Contrast'].default_value = 0.08
    link(nt, day.outputs['Color'], cc.inputs['Color'])
    link(nt, cc.outputs['Color'], bsdf.inputs['Base Color'])
    # oceans glossy, land rough
    rough = node(nt, 'ShaderNodeMapRange', (-500, 0))
    rough.inputs['To Min'].default_value = 0.85
    rough.inputs['To Max'].default_value = 0.32
    sep = node(nt, 'ShaderNodeSeparateColor', (-650, 0))
    link(nt, spec.outputs['Color'], sep.inputs['Color'])
    link(nt, sep.outputs[0], rough.inputs['Value'])
    link(nt, rough.outputs['Result'], bsdf.inputs['Roughness'])
    nm = node(nt, 'ShaderNodeNormalMap', (-500, -300))
    nm.inputs['Strength'].default_value = 0.35
    link(nt, nrm.outputs['Color'], nm.inputs['Color'])
    link(nt, nm.outputs['Normal'], bsdf.inputs['Normal'])
    # city lights on the night side
    mask, sd = _sun_mask(nt, sun_dir, (0, -700))
    em = node(nt, 'ShaderNodeMath', (200, -600), operation='MULTIPLY')
    link(nt, mask, em.inputs[0])
    em.inputs[1].default_value = 1.6
    link(nt, night.outputs['Color'], bsdf.inputs['Emission Color'])
    link(nt, em.outputs['Value'], bsdf.inputs['Emission Strength'])
    link(nt, bsdf.outputs['BSDF'], out.inputs['Surface'])
    mat.cycles.emission_sampling = 'NONE'          # city lights must not become 70k mesh lights
    ob.data.materials.append(mat)
    ob['sun_vec_node'] = sd.name
    return ob


def clouds(cover_boost=0.0, name='Clouds'):
    """cover_boost 0..1 thickens the cloud deck towards a full overcast (a steam-shrouded Earth)."""
    ob = uv_sphere(name, R_EARTH + 0.012, 384, 192)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputMaterial', (800, 0))
    uv = node(nt, 'ShaderNodeTexCoord', (-900, 0))
    tex = node(nt, 'ShaderNodeTexImage', (-600, 0))
    tex.image = img(f'{ASSETS}/earth/8k_earth_clouds.jpg', 'Non-Color')
    link(nt, uv.outputs['UV'], tex.inputs['Vector'])
    boost = node(nt, 'ShaderNodeMapRange', (-300, 0))
    boost.inputs['From Min'].default_value = 0.0
    boost.inputs['From Max'].default_value = 1.0
    boost.inputs['To Min'].default_value = 0.0
    boost.inputs['To Max'].default_value = 1.0
    link(nt, tex.outputs['Color'], boost.inputs['Value'])
    fill = node(nt, 'ShaderNodeMix', (-100, 0), data_type='FLOAT')
    fill.inputs['B'].default_value = 1.0
    fill.inputs['Factor'].default_value = cover_boost
    link(nt, boost.outputs['Result'], fill.inputs['A'])
    bsdf = node(nt, 'ShaderNodeBsdfPrincipled', (200, 100))
    bsdf.inputs['Base Color'].default_value = (0.92, 0.92, 0.92, 1)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    tr = node(nt, 'ShaderNodeBsdfTransparent', (200, -150))
    mix = node(nt, 'ShaderNodeMixShader', (500, 0))
    link(nt, fill.outputs['Result'], mix.inputs['Fac'])
    link(nt, tr.outputs['BSDF'], mix.inputs[1])
    link(nt, bsdf.outputs['BSDF'], mix.inputs[2])
    link(nt, mix.outputs['Shader'], out.inputs['Surface'])
    ob.data.materials.append(mat)
    mat['cover_node'] = fill.name
    ob.visible_shadow = True
    return ob


def atmosphere(name='Atmosphere', thickness=0.10, scale_height=0.0085, density=27.0):
    """Rayleigh-coloured scattering shell, density falling off exponentially with height. Vertical optical
    depth = density * scale_height * colour = 0.23 at 440 nm, 0.094 at 550 nm (the real Rayleigh values)."""
    ob = uv_sphere(name, R_EARTH + thickness, 192, 96)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    tc = node(nt, 'ShaderNodeTexCoord', (-900, 0))
    ln = node(nt, 'ShaderNodeVectorMath', (-700, 0), operation='LENGTH')
    link(nt, tc.outputs['Object'], ln.inputs[0])
    h = node(nt, 'ShaderNodeMath', (-500, 0), operation='SUBTRACT')
    link(nt, ln.outputs['Value'], h.inputs[0])
    h.inputs[1].default_value = R_EARTH
    sc = node(nt, 'ShaderNodeMath', (-300, 0), operation='DIVIDE')
    link(nt, h.outputs['Value'], sc.inputs[0])
    sc.inputs[1].default_value = -scale_height
    ex = node(nt, 'ShaderNodeMath', (-100, 0), operation='EXPONENT')
    link(nt, sc.outputs['Value'], ex.inputs[0])
    den = node(nt, 'ShaderNodeMath', (100, 0), operation='MULTIPLY')
    link(nt, ex.outputs['Value'], den.inputs[0])
    den.inputs[1].default_value = density
    clamp = node(nt, 'ShaderNodeMath', (250, 0), operation='MINIMUM')
    link(nt, den.outputs['Value'], clamp.inputs[0])
    clamp.inputs[1].default_value = density
    vs = node(nt, 'ShaderNodeVolumeScatter', (500, 0))
    vs.inputs['Color'].default_value = (0.175, 0.41, 1.0, 1)      # 1/lambda^4 at 680/550/440 nm
    vs.inputs['Anisotropy'].default_value = 0.0
    link(nt, clamp.outputs['Value'], vs.inputs['Density'])
    link(nt, vs.outputs['Volume'], out.inputs['Volume'])
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    mat['density_node'] = den.name
    return ob


def moon_material(name='MoonSurface', exaggerate=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.cycles.displacement_method = 'BOTH'
    nt = mat.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    tc = node(nt, 'ShaderNodeTexCoord', (-1200, 0))
    # equirectangular lookup from the object-space direction (works for fragments too)
    nrm = node(nt, 'ShaderNodeVectorMath', (-1000, 0), operation='NORMALIZE')
    link(nt, tc.outputs['Object'], nrm.inputs[0])
    sepx = node(nt, 'ShaderNodeSeparateXYZ', (-850, 0))
    link(nt, nrm.outputs['Vector'], sepx.inputs[0])
    lon = node(nt, 'ShaderNodeMath', (-700, 100), operation='ARCTAN2')
    link(nt, sepx.outputs['Y'], lon.inputs[0])
    link(nt, sepx.outputs['X'], lon.inputs[1])
    u = node(nt, 'ShaderNodeMath', (-550, 100), operation='MULTIPLY_ADD')
    link(nt, lon.outputs['Value'], u.inputs[0])
    u.inputs[1].default_value = 1 / (2 * math.pi)
    u.inputs[2].default_value = 0.5
    lat = node(nt, 'ShaderNodeMath', (-700, -100), operation='ARCSINE')
    link(nt, sepx.outputs['Z'], lat.inputs[0])
    v = node(nt, 'ShaderNodeMath', (-550, -100), operation='MULTIPLY_ADD')
    link(nt, lat.outputs['Value'], v.inputs[0])
    v.inputs[1].default_value = 1 / math.pi
    v.inputs[2].default_value = 0.5
    uvv = node(nt, 'ShaderNodeCombineXYZ', (-400, 0))
    link(nt, u.outputs['Value'], uvv.inputs['X'])
    link(nt, v.outputs['Value'], uvv.inputs['Y'])
    col = node(nt, 'ShaderNodeTexImage', (-200, 200))
    col.image = img(f'{ASSETS}/moon/lroc_color_poles_8k.tif')
    col.interpolation = 'Cubic'
    dem = node(nt, 'ShaderNodeTexImage', (-200, -200))
    dem.image = img(f'{ASSETS}/moon/ldem_16.tif', 'Non-Color')
    dem.interpolation = 'Cubic'
    for t in (col, dem):
        link(nt, uvv.outputs['Vector'], t.inputs['Vector'])
    gam = node(nt, 'ShaderNodeGamma', (50, 200))
    gam.inputs['Gamma'].default_value = 1.15
    link(nt, col.outputs['Color'], gam.inputs['Color'])
    dif = node(nt, 'ShaderNodeBsdfDiffuse', (400, 100))
    dif.inputs['Roughness'].default_value = 1.0                 # Oren-Nayar: flat-looking lunar disc
    link(nt, gam.outputs['Color'], dif.inputs['Color'])
    link(nt, dif.outputs['BSDF'], out.inputs['Surface'])
    # LOLA heights are km relative to 1737.4 km; 1 unit = 1000 km
    disp = node(nt, 'ShaderNodeDisplacement', (500, -250))
    disp.inputs['Midlevel'].default_value = 0.0
    disp.inputs['Scale'].default_value = 0.001 * exaggerate
    sepc = node(nt, 'ShaderNodeSeparateColor', (300, -250))
    link(nt, dem.outputs['Color'], sepc.inputs['Color'])
    link(nt, sepc.outputs[0], disp.inputs['Height'])
    link(nt, disp.outputs['Displacement'], out.inputs['Displacement'])
    return mat


def moon(loc=(0, 0, 0), name='Moon', exaggerate=1.0):
    ob = uv_sphere(name, R_MOON, 256, 128, loc)
    ob.data.materials.append(moon_material(name + 'Mat', exaggerate))
    sub = ob.modifiers.new('Subdiv', 'SUBSURF')
    sub.levels = 1
    ob.cycles.use_adaptive_subdivision = True
    bpy.context.scene.cycles.dicing_rate = 1.0
    return ob


def camera(loc, target, lens=50.0, name='Cam'):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.01
    cam.clip_end = 1e5
    ob = bpy.data.objects.new(name, cam)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = d.to_track_quat('-Z', 'Y')
    bpy.context.scene.camera = ob
    return ob


def atmosphere_fast(sun_dir, name='AtmosphereShell', thickness=0.075, strength=1.6, power=3.0):
    """Cheap limb glow: an emissive shell whose brightness grows towards the limb (long sight lines through
    the air) and fades into the night side, plus a little twilight wrap. No volume, so ~free to render."""
    ob = uv_sphere(name, R_EARTH + thickness, 192, 96)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    nt = mat.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    lw = node(nt, 'ShaderNodeLayerWeight', (-600, 200))
    lw.inputs['Blend'].default_value = 0.5
    inv = node(nt, 'ShaderNodeMath', (-400, 200), operation='POWER')
    link(nt, lw.outputs['Facing'], inv.inputs[0])
    inv.inputs[1].default_value = power
    mask, sd = _sun_mask(nt, sun_dir, (-200, -200), lo=0.25, hi=-0.25, use_position=True)   # 1 on the day side
    geo = node(nt, 'ShaderNodeNewGeometry', (-400, -450))
    front = node(nt, 'ShaderNodeMath', (-200, -450), operation='SUBTRACT')
    front.inputs[0].default_value = 1.0
    link(nt, geo.outputs['Backfacing'], front.inputs[1])
    m2 = node(nt, 'ShaderNodeMath', (-100, -300), operation='MULTIPLY')
    link(nt, mask, m2.inputs[0])
    link(nt, front.outputs['Value'], m2.inputs[1])
    day = node(nt, 'ShaderNodeMath', (0, 0), operation='MULTIPLY')
    link(nt, inv.outputs['Value'], day.inputs[0])
    link(nt, m2.outputs['Value'], day.inputs[1])
    s = node(nt, 'ShaderNodeMath', (200, 0), operation='MULTIPLY')
    link(nt, day.outputs['Value'], s.inputs[0])
    s.inputs[1].default_value = strength
    em = node(nt, 'ShaderNodeEmission', (400, 100))
    em.inputs['Color'].default_value = (0.32, 0.55, 1.0, 1)
    link(nt, s.outputs['Value'], em.inputs['Strength'])
    tr = node(nt, 'ShaderNodeBsdfTransparent', (400, -100))
    add = node(nt, 'ShaderNodeAddShader', (650, 0))
    link(nt, em.outputs['Emission'], add.inputs[0])
    link(nt, tr.outputs['BSDF'], add.inputs[1])
    link(nt, add.outputs['Shader'], out.inputs['Surface'])
    mat.cycles.emission_sampling = 'NONE'
    ob.data.materials.append(mat)
    ob.visible_shadow = False
    ob.visible_glossy = False
    ob.visible_diffuse = False
    mat['strength_node'] = s.name
    return ob


def add_aerial_haze(earth_ob, sun_dir, amount=0.35):
    """Blue haze over the disc that thickens towards the limb (aerial perspective), on the day side."""
    nt = earth_ob.data.materials[0].node_tree
    bsdf = nt.nodes['Principled BSDF']
    src = bsdf.inputs['Base Color'].links[0].from_socket
    lw = node(nt, 'ShaderNodeLayerWeight', (-200, 600))
    lw.inputs['Blend'].default_value = 0.35
    pw = node(nt, 'ShaderNodeMath', (0, 600), operation='POWER')
    link(nt, lw.outputs['Facing'], pw.inputs[0])
    pw.inputs[1].default_value = 1.5
    f = node(nt, 'ShaderNodeMath', (150, 600), operation='MULTIPLY_ADD')
    link(nt, pw.outputs['Value'], f.inputs[0])
    f.inputs[1].default_value = amount
    f.inputs[2].default_value = 0.06
    mix = node(nt, 'ShaderNodeMix', (300, 400), data_type='RGBA', blend_type='SCREEN')
    link(nt, f.outputs['Value'], mix.inputs['Factor'])
    link(nt, src, mix.inputs['A'])
    mix.inputs['B'].default_value = (0.30, 0.50, 0.95, 1)
    link(nt, mix.outputs['Result'], bsdf.inputs['Base Color'])


def add_clouds_to_surface(earth_ob, sun_dir, cover_boost=0.0, shadow=0.35):
    """Clouds painted into the Earth's own shader (white, rough, over the ground), with a cheap shadow:
    the cloud map sampled a little towards the Sun darkens the ground. Much faster than a transparent shell."""
    mat = earth_ob.data.materials[0]
    nt = mat.node_tree
    bsdf = nt.nodes['Principled BSDF']
    uv = nt.nodes['Texture Coordinate']
    tex = node(nt, 'ShaderNodeTexImage', (-800, 900))
    tex.image = img(f'{ASSETS}/earth/8k_earth_clouds.jpg', 'Non-Color')
    link(nt, uv.outputs['UV'], tex.inputs['Vector'])
    cov = node(nt, 'ShaderNodeMix', (-550, 900), data_type='FLOAT')
    link(nt, tex.outputs['Color'], cov.inputs['A'])
    cov.inputs['B'].default_value = 1.0
    cov.inputs['Factor'].default_value = cover_boost
    # shadow: same map, shifted ~0.15 degrees in longitude (cheap stand-in for the sun-ward offset)
    off = node(nt, 'ShaderNodeVectorMath', (-1000, 1150), operation='ADD')
    link(nt, uv.outputs['UV'], off.inputs[0])
    off.inputs[1].default_value = (0.0006, 0.0, 0.0)
    tex2 = node(nt, 'ShaderNodeTexImage', (-800, 1150))
    tex2.image = tex.image
    link(nt, off.outputs['Vector'], tex2.inputs['Vector'])
    sh = node(nt, 'ShaderNodeMath', (-550, 1150), operation='MULTIPLY_ADD')
    link(nt, tex2.outputs['Color'], sh.inputs[0])
    sh.inputs[1].default_value = -shadow
    sh.inputs[2].default_value = 1.0
    src = bsdf.inputs['Base Color'].links[0].from_socket
    dark = node(nt, 'ShaderNodeMix', (-300, 1000), data_type='RGBA', blend_type='MULTIPLY')
    dark.inputs['Factor'].default_value = 1.0
    link(nt, src, dark.inputs['A'])
    comb = node(nt, 'ShaderNodeCombineColor', (-450, 1150))
    for i in range(3):
        link(nt, sh.outputs['Value'], comb.inputs[i])
    link(nt, comb.outputs['Color'], dark.inputs['B'])
    white = node(nt, 'ShaderNodeMix', (-100, 1000), data_type='RGBA')
    link(nt, cov.outputs['Result'], white.inputs['Factor'])
    link(nt, dark.outputs['Result'], white.inputs['A'])
    white.inputs['B'].default_value = (0.93, 0.93, 0.93, 1)
    link(nt, white.outputs['Result'], bsdf.inputs['Base Color'])
    # clouds are rough and hide ocean glint and city lights
    rsrc = bsdf.inputs['Roughness'].links[0].from_socket
    rmix = node(nt, 'ShaderNodeMix', (200, 800), data_type='FLOAT')
    link(nt, cov.outputs['Result'], rmix.inputs['Factor'])
    link(nt, rsrc, rmix.inputs['A'])
    rmix.inputs['B'].default_value = 1.0
    link(nt, rmix.outputs['Result'], bsdf.inputs['Roughness'])
    esrc = bsdf.inputs['Emission Strength'].links[0].from_socket
    emix = node(nt, 'ShaderNodeMath', (200, -800), operation='MULTIPLY')
    link(nt, esrc, emix.inputs[0])
    inv = node(nt, 'ShaderNodeMath', (50, -900), operation='SUBTRACT')
    inv.inputs[0].default_value = 1.0
    link(nt, cov.outputs['Result'], inv.inputs[1])
    link(nt, inv.outputs['Value'], emix.inputs[1])
    link(nt, emix.outputs['Value'], bsdf.inputs['Emission Strength'])
    nm = bsdf.inputs['Normal'].links[0].from_node
    link(nt, inv.outputs['Value'], nm.inputs['Strength']) if False else None
    mat['cloud_cover_node'] = cov.name
    return cov
