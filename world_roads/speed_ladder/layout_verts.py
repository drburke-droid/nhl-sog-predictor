import bpy, sys, os, numpy as np
sys.path.insert(0, "/home/user/lot/src")
import scene as S, aircraft as A
for name in ("wright", "spitfire", "x1", "f35", "x15"):
    S.reset(samples=1)
    root, piv = A.build(name)
    bpy.context.view_layer.update()
    pts = []
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render or o is root: continue
        p = o
        while p is not None and p is not root: p = p.parent
        if p is None: continue
        M = np.array(o.matrix_world); Mr = np.array(root.matrix_world)
        v = np.array([x.co[:] for x in o.data.vertices])
        if not len(v): continue
        v = np.c_[v, np.ones(len(v))] @ M.T
        v = v @ np.linalg.inv(Mr).T
        pts.append(v[:, :3])
    P = np.concatenate(pts)
    np.save(f"/home/user/lot/layout/verts_{name}.npy", P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)])
    print(name, len(P), P.min(0).round(2), P.max(0).round(2))
