"""The seven pieces of the Moon as Blender objects, cut along the simulation's own power-diagram cells.

Each piece is a high-resolution icosphere clipped by the cell's half-spaces, with the cut faces filled.
One material per piece:
  * outer surface: LRO colour + LOLA heights (radial vector displacement, so outer and cut faces agree
    at the rim and nothing cracks);
  * cut faces: fresh mantle rock (dark olive-grey) with fracture roughness, glowing by temperature with
    depth (the Moon's interior is ~1,000-1,600 K below a few hundred km), fading as the skin cools.
Units: 1 = 1,000 km. Mesh origin = the piece's centroid, matching the N-body body position.
"""
import json
import math
import os

import bpy
import bmesh
from mathutils import Vector, Matrix

import look

HERE = os.path.dirname(os.path.abspath(__file__))
R = look.R_MOON


def cells():
    return json.load(open(os.path.join(HERE, 'data', 'cells.json')))


def _piece_bmesh(k, c, subdiv=7):
    seeds = [Vector(s) for s in c['seeds']]
    w = c['w']
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=R)
    cut_faces = set()
    sk = seeds[k]
    for j, sj in enumerate(seeds):
        if j == k:
            continue
        # cell k: |x - sk|^2 - wk <= |x - sj|^2 - wj  (x in units of the Moon's radius)
        #   <=> 2 x.(sj - sk) <= |sj|^2 - |sk|^2 - wj + wk
        n = (sj - sk)
        d = (sj.length_squared - sk.length_squared - w[j] + w[k]) / 2.0
        if n.length == 0:
            continue
        nn = n.normalized()
        co = nn * (d / n.length) * R            # a point on the plane, in Blender units
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=co, plane_no=nn,
                                     clear_outer=True, clear_inner=False)
        edges = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
        if edges:
            fill = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            for f in fill['faces']:
                cut_faces.add(f)
    # older fill faces may have been re-cut; mark every face whose normal is far from radial as cut
    bm.faces.ensure_lookup_table()
    lay = bm.faces.layers.int.new('cut')
    for f in bm.faces:
        cen = f.calc_center_median()
        radial = abs(f.normal.dot(cen.normalized())) if cen.length > 0 else 0
        f[lay] = 1 if (cen.length < R * 0.995 or radial < 0.9) else 0
    cuts = [f for f in bm.faces if f[lay] == 1]
    bmesh.ops.triangulate(bm, faces=cuts, quad_method='BEAUTY', ngon_method='BEAUTY')
    return bm


def piece_material(k, glow_name='glow'):
    """centre_local: the Moon's centre in the piece's local coordinates."""
    m = bpy.data.materials.new(f'Piece{k}')
    m.use_nodes = True
    m.cycles.displacement_method = 'BOTH'
    m.cycles.emission_sampling = 'NONE'
    nt = m.node_tree
    nt.nodes.clear()
    N = look.node
    L = lambda a, b: nt.links.new(a, b)
    out = N(nt, 'ShaderNodeOutputMaterial', (1400, 0))
    # --- undeformed attributes: texture-frame direction, radius from the old centre, original position
    at_t = N(nt, 'ShaderNodeAttribute', (-1600, 300))
    at_t.attribute_name = 'tex_dir'
    at_t.attribute_type = 'GEOMETRY'
    at_r = N(nt, 'ShaderNodeAttribute', (-1600, -400))
    at_r.attribute_name = 'rlen'
    at_r.attribute_type = 'GEOMETRY'
    at_o = N(nt, 'ShaderNodeAttribute', (-1600, -250))
    at_o.attribute_name = 'orig'
    at_o.attribute_type = 'GEOMETRY'
    tco = at_o                                      # noise textures use the undeformed position
    nrm = N(nt, 'ShaderNodeVectorMath', (-1400, 300), operation='NORMALIZE')
    L(at_t.outputs['Vector'], nrm.inputs[0])
    sx = N(nt, 'ShaderNodeSeparateXYZ', (-1250, 300))
    L(nrm.outputs['Vector'], sx.inputs[0])
    lon = N(nt, 'ShaderNodeMath', (-1100, 400), operation='ARCTAN2')
    L(sx.outputs['Y'], lon.inputs[0])
    L(sx.outputs['X'], lon.inputs[1])
    u = N(nt, 'ShaderNodeMath', (-950, 400), operation='MULTIPLY_ADD')
    L(lon.outputs['Value'], u.inputs[0])
    u.inputs[1].default_value = 1 / (2 * math.pi)
    u.inputs[2].default_value = 0.5
    lat = N(nt, 'ShaderNodeMath', (-1100, 200), operation='ARCSINE')
    L(sx.outputs['Z'], lat.inputs[0])
    v = N(nt, 'ShaderNodeMath', (-950, 200), operation='MULTIPLY_ADD')
    L(lat.outputs['Value'], v.inputs[0])
    v.inputs[1].default_value = 1 / math.pi
    v.inputs[2].default_value = 0.5
    uvv = N(nt, 'ShaderNodeCombineXYZ', (-800, 300))
    L(u.outputs['Value'], uvv.inputs['X'])
    L(v.outputs['Value'], uvv.inputs['Y'])
    col = N(nt, 'ShaderNodeTexImage', (-600, 500))
    col.image = look.img(f'{look.ASSETS}/moon/lroc_color_poles_8k.tif')
    col.interpolation = 'Cubic'
    dem = N(nt, 'ShaderNodeTexImage', (-600, 200))
    dem.image = look.img(f'{look.ASSETS}/moon/ldem_16.tif', 'Non-Color')
    dem.interpolation = 'Cubic'
    L(uvv.outputs['Vector'], col.inputs['Vector'])
    L(uvv.outputs['Vector'], dem.inputs['Vector'])
    # --- which faces are cut, and how deep below the old surface we are
    cut = N(nt, 'ShaderNodeAttribute', (-600, -100))
    cut.attribute_name = 'cut'
    cut.attribute_type = 'GEOMETRY'
    depth = N(nt, 'ShaderNodeMapRange', (-900, -500))      # 1 at the surface, 0 deeper than 6% of R
    depth.inputs['From Min'].default_value = 0.94 * R
    depth.inputs['From Max'].default_value = 0.995 * R
    L(at_r.outputs['Fac'], depth.inputs['Value'])
    # --- colour: outer = LRO; cut = mantle rock with mottling
    noise = N(nt, 'ShaderNodeTexNoise', (-600, -300))
    noise.inputs['Scale'].default_value = 9.0
    noise.inputs['Detail'].default_value = 10.0
    noise.inputs['Roughness'].default_value = 0.62
    L(tco.outputs['Vector'], noise.inputs['Vector'])
    rock = N(nt, 'ShaderNodeMix', (-350, -300), data_type='RGBA')
    L(noise.outputs['Fac'], rock.inputs['Factor'])
    rock.inputs['A'].default_value = (0.045, 0.048, 0.040, 1)
    rock.inputs['B'].default_value = (0.22, 0.21, 0.18, 1)
    gam = N(nt, 'ShaderNodeGamma', (-350, 500))
    gam.inputs['Gamma'].default_value = 1.05
    L(col.outputs['Color'], gam.inputs['Color'])
    cmix = N(nt, 'ShaderNodeMix', (0, 300), data_type='RGBA')
    L(cut.outputs['Fac'], cmix.inputs['Factor'])
    L(gam.outputs['Color'], cmix.inputs['A'])
    L(rock.outputs['Result'], cmix.inputs['B'])
    dif = N(nt, 'ShaderNodeBsdfDiffuse', (400, 300))
    dif.inputs['Roughness'].default_value = 1.0
    L(cmix.outputs['Result'], dif.inputs['Color'])
    # --- heat glow on cut faces: T from ~1600 K deep to ~600 K near the crust, times a cooling factor
    temp = N(nt, 'ShaderNodeMapRange', (-350, -700))
    L(at_r.outputs['Fac'], temp.inputs['Value'])
    temp.inputs['From Min'].default_value = 0.35 * R
    temp.inputs['From Max'].default_value = 0.92 * R
    temp.inputs['To Min'].default_value = 1650.0
    temp.inputs['To Max'].default_value = 650.0
    bb = N(nt, 'ShaderNodeBlackbody', (-100, -700))
    L(temp.outputs['Result'], bb.inputs['Temperature'])
    t4 = N(nt, 'ShaderNodeMath', (-100, -900), operation='POWER')
    q = N(nt, 'ShaderNodeMath', (-250, -900), operation='DIVIDE')
    L(temp.outputs['Result'], q.inputs[0])
    q.inputs[1].default_value = 1500.0
    L(q.outputs['Value'], t4.inputs[0])
    t4.inputs[1].default_value = 8.0             # visible-band brightness rises much faster than T^4
    vor = N(nt, 'ShaderNodeTexVoronoi', (-350, -1150))
    vor.feature = 'DISTANCE_TO_EDGE'
    vor.inputs['Scale'].default_value = 7.0
    vor.inputs['Randomness'].default_value = 1.0
    warp = N(nt, 'ShaderNodeVectorMath', (-500, -1150), operation='MULTIPLY_ADD')
    L(noise.outputs['Color'], warp.inputs[0])
    warp.inputs[1].default_value = (0.08, 0.08, 0.08)
    L(tco.outputs['Vector'], warp.inputs[2])
    L(warp.outputs['Vector'], vor.inputs['Vector'])
    cracks = N(nt, 'ShaderNodeMapRange', (-100, -1100))
    L(vor.outputs['Distance'], cracks.inputs['Value'])
    cracks.inputs['From Min'].default_value = 0.035
    cracks.inputs['From Max'].default_value = 0.0
    cracks.interpolation_type = 'SMOOTHSTEP'
    g1 = N(nt, 'ShaderNodeMath', (100, -900), operation='MULTIPLY')
    L(t4.outputs['Value'], g1.inputs[0])
    L(cracks.outputs['Result'], g1.inputs[1])
    g2 = N(nt, 'ShaderNodeMath', (250, -900), operation='MULTIPLY')
    L(g1.outputs['Value'], g2.inputs[0])
    L(cut.outputs['Fac'], g2.inputs[1])
    glow = N(nt, 'ShaderNodeValue', (100, -1100))
    glow.name = glow_name
    glow.outputs[0].default_value = 1.0
    g3 = N(nt, 'ShaderNodeMath', (400, -900), operation='MULTIPLY')
    L(g2.outputs['Value'], g3.inputs[0])
    L(glow.outputs['Value'], g3.inputs[1])
    em = N(nt, 'ShaderNodeEmission', (650, -700))
    L(bb.outputs['Color'], em.inputs['Color'])
    L(g3.outputs['Value'], em.inputs['Strength'])
    add = N(nt, 'ShaderNodeAddShader', (900, 0))
    L(dif.outputs['BSDF'], add.inputs[0])
    L(em.outputs['Emission'], add.inputs[1])
    L(add.outputs['Shader'], out.inputs['Surface'])
    # --- displacement: radial LOLA height near the old surface, fracture noise deeper on the cuts
    sep = N(nt, 'ShaderNodeSeparateColor', (-350, 150))
    L(dem.outputs['Color'], sep.inputs['Color'])
    hk = N(nt, 'ShaderNodeMath', (-150, 150), operation='MULTIPLY')
    L(sep.outputs[0], hk.inputs[0])
    hk.inputs[1].default_value = 0.001              # km -> 1000 km units
    surf_w = N(nt, 'ShaderNodeMath', (-150, -50), operation='MAXIMUM')   # outer faces always 1
    inv_cut = N(nt, 'ShaderNodeMath', (-350, -50), operation='SUBTRACT')
    inv_cut.inputs[0].default_value = 1.0
    L(cut.outputs['Fac'], inv_cut.inputs[1])
    L(inv_cut.outputs['Value'], surf_w.inputs[0])
    L(depth.outputs['Result'], surf_w.inputs[1])
    hrad = N(nt, 'ShaderNodeMath', (50, 100), operation='MULTIPLY')
    L(hk.outputs['Value'], hrad.inputs[0])
    L(surf_w.outputs['Value'], hrad.inputs[1])
    big = N(nt, 'ShaderNodeTexNoise', (-600, -600))
    big.inputs['Scale'].default_value = 1.6
    big.inputs['Detail'].default_value = 4.0
    L(tco.outputs['Vector'], big.inputs['Vector'])
    both = N(nt, 'ShaderNodeMath', (-150, -600), operation='MULTIPLY_ADD')
    L(big.outputs['Fac'], both.inputs[0])
    both.inputs[1].default_value = 2.2
    L(noise.outputs['Fac'], both.inputs[2])
    rough = N(nt, 'ShaderNodeMath', (50, -250), operation='MULTIPLY_ADD')
    L(both.outputs['Value'], rough.inputs[0])
    rough.inputs[1].default_value = 0.05
    rough.inputs[2].default_value = -0.08
    frac_amt = N(nt, 'ShaderNodeValue', (50, -150))
    frac_amt.name = 'fracture'
    frac_amt.outputs[0].default_value = 1.0
    rough2 = N(nt, 'ShaderNodeMath', (120, -250), operation='MULTIPLY')
    L(rough.outputs['Value'], rough2.inputs[0])
    L(frac_amt.outputs[0], rough2.inputs[1])
    rw = N(nt, 'ShaderNodeMath', (200, -250), operation='MULTIPLY')
    L(rough2.outputs['Value'], rw.inputs[0])
    deep = N(nt, 'ShaderNodeMath', (50, -400), operation='SUBTRACT')
    deep.inputs[0].default_value = 1.0
    L(surf_w.outputs['Value'], deep.inputs[1])
    L(deep.outputs['Value'], rw.inputs[1])
    hsum = N(nt, 'ShaderNodeMath', (600, -100), operation='ADD')
    L(hrad.outputs['Value'], hsum.inputs[0])
    L(rw.outputs['Value'], hsum.inputs[1])
    disp = N(nt, 'ShaderNodeDisplacement', (900, -150))
    disp.inputs['Midlevel'].default_value = 0.0
    disp.inputs['Scale'].default_value = 1.0
    L(hsum.outputs['Value'], disp.inputs['Height'])
    L(disp.outputs['Displacement'], out.inputs['Displacement'])
    return m


def build_pieces(tex_rotation=Matrix.Identity(3), subdiv=7):
    """Returns list of piece objects (index = simulation piece id - 1). tex_rotation maps the inertial
    frame to the Moon's texture frame (x = sub-Earth point, z = north)."""
    c = cells()
    cen = [Vector(x) * R for x in c['centroid']]
    frac = c['frac']
    mean = sum((cen[k] * frac[k] for k in range(len(cen))), Vector()) / sum(frac)
    obs = []
    for k in range(len(c['seeds'])):
        bm = _piece_bmesh(k, c, subdiv)
        offset = cen[k] - mean                       # the body's position relative to the Moon's centre
        bmesh.ops.translate(bm, verts=bm.verts, vec=-cen[k])
        me = bpy.data.meshes.new(f'Piece{k + 1}')
        bm.to_mesh(me)
        bm.free()
        lay = me.attributes.get('cut')
        if lay is not None and lay.data_type != 'FLOAT':
            vals = [d.value for d in lay.data]
            me.attributes.remove(lay)
            nl = me.attributes.new('cut', 'FLOAT', 'FACE')
            for d, x in zip(nl.data, vals):
                d.value = float(x)
        for p in me.polygons:
            p.use_smooth = True
        # undeformed attributes, so textures stick to the rock while the piece slumps
        nv = len(me.vertices)
        co = [v.co.copy() for v in me.vertices]
        rel = [p + cen[k] for p in co]                        # position from the Moon's old centre
        tdir = [(tex_rotation @ r).normalized() for r in rel]
        for nm_, ty in (('tex_dir', 'FLOAT_VECTOR'), ('rlen', 'FLOAT'), ('orig', 'FLOAT_VECTOR')):
            me.attributes.new(nm_, ty, 'POINT')
        # (fetch after creating all three: adding an attribute invalidates earlier references)
        me.attributes['tex_dir'].data.foreach_set('vector', [x for v in tdir for x in v])
        me.attributes['rlen'].data.foreach_set('value', [r.length for r in rel])
        me.attributes['orig'].data.foreach_set('vector', [x for v in co for x in v])
        ob = bpy.data.objects.new(f'Piece{k + 1}', me)
        bpy.context.collection.objects.link(ob)
        # slump: a body this size can't hold a wedge shape; it settles towards a lumpy spheroid
        r_eq = R * frac[k] ** (1 / 3)
        ob.shape_key_add(name='Basis')
        sk = ob.shape_key_add(name='slump')
        import random
        rnd = random.Random(k)
        bumps = [(Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1))).normalized(), rnd.uniform(0.04, 0.10))
                 for _ in range(9)]
        for i, p in enumerate(co):
            d = p.normalized() if p.length > 0 else Vector((0, 0, 1))
            lump = 1.0 + sum(a * max(d.dot(b), 0.0) ** 3 for b, a in bumps) - 0.04
            sk.data[i].co = d * r_eq * lump
        sk.value = 0.0
        ob.data.materials.append(piece_material(k + 1))
        sub = ob.modifiers.new('Subdiv', 'SUBSURF')
        sub.subdivision_type = 'SIMPLE'
        sub.levels = 0
        ob.cycles.use_adaptive_subdivision = True
        ob['offset'] = tuple(offset)
        ob['centroid'] = tuple(cen[k])
        obs.append(ob)
    return obs
