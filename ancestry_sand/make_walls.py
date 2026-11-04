"""Glass wall outlines for Blender (data/walls.json): each vessel's outline, thickened into a wall that
straddles it (outer face 1.6 px outside, inner face 1.3 px inside). Screen-pixel coordinates."""
import json, os
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import story as S, vessels as V

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    G = V.build(S.VESSELS)
    out = {}
    for vid, g in G.items():
        polys = [Polygon(e, hs).buffer(0) for e, hs in g["rings"]]
        u = unary_union([p for p in polys if p.area > 2.0]).simplify(0.6)
        wall = u.buffer(1.6, 8).difference(u.buffer(-1.3, 8))
        rings = []
        for p in getattr(wall, "geoms", [wall]):
            if p.area < 1.5:
                continue
            rings.append([np.round(np.array(p.exterior.coords), 2).tolist()] +
                         [np.round(np.array(r.coords), 2).tolist() for r in p.interiors])
        out[vid] = rings
    json.dump(out, open(os.path.join(DATA, "walls.json"), "w"))
    print("vertices", sum(len(r) for v in out.values() for p in v for r in p))


if __name__ == "__main__":
    main()
