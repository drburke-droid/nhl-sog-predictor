"""Where the props stand. World: road along +Y, x to the right, ground z = 0; the ball's path crosses at y = 12."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# name: (kind, x, y, heading in degrees about z)
# a line of cones across both lanes, edge line to edge line, 1.5 m beyond the ball's line: the road closed for the test
ROW_Y = 13.5
_ROW_X = [-4.0, -2.96, -1.92, -0.88, 0.16, 1.2, 2.24, 3.28]
_JIT = [(0.03, 10), (-0.02, 35), (0.04, 5), (-0.03, 50), (0.02, 20), (-0.04, 65), (0.01, 15), (-0.02, 40)]

# name: (kind, x, y, heading in degrees about z)
LAYOUT = {f"cone{i + 1}": ("cone", x, ROW_Y + dy, h) for i, (x, (dy, h)) in enumerate(zip(_ROW_X, _JIT))}
LAYOUT.update({
    "drum1": ("drum", -6.2, 13.4, 10), "drum2": ("drum", -6.9, 14.0, 70), "drum3": ("drum", -5.4, 9.0, 30),
    "sign": ("sign", 4.5, 10.2, 0),
    "car": ("car", 5.6, 17.2, 0),
})


COM = {"cone": 0.15}                             # height of the centre of mass (the cone's base is heavy rubber)


def build():
    """every prop at rest; cones and drums have their origin at the centre of mass, for the physics"""
    from mathutils import Matrix
    import props as P
    made = {}
    for name, (kind, x, y, h) in LAYOUT.items():
        o = getattr(P, kind)(name)
        z = 0.0
        if kind in ("cone", "drum"):
            z = COM.get(kind) or max(v.co.z for v in o.data.vertices) / 2
            o.data.transform(Matrix.Translation((0, 0, -z)))
        o.location = (x, y, z)
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = (math.cos(math.radians(h) / 2), 0, 0, math.sin(math.radians(h) / 2))
        made[name] = o
    return made
