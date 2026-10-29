"""Where the props stand. World: road along +Y, x to the right, ground z = 0; the ball's path crosses at y = 12."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# name: (kind, x, y, heading in degrees about z)
# the car parked on the right shoulder with a warning cone set out behind it, and the 100 sign
LAYOUT = {
    "cone1": ("cone", 2.95, 13.2, 20),
    "sign": ("sign", 4.5, 10.2, 0),
    "car": ("car", 5.6, 17.2, 0),
}


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
