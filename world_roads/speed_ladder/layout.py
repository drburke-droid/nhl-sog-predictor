"""Where the props stand. World: road along +Y, x to the right, ground z = 0; the ball's path crosses at y = 12."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# name: (kind, x, y, heading in degrees about z)
LAYOUT = {
    "cone1": ("cone", -2.3, 12.9, 0), "cone2": ("cone", -0.6, 11.0, 20), "cone3": ("cone", 1.5, 13.2, 40),
    "cone4": ("cone", 2.6, 10.4, 10), "cone5": ("cone", -3.1, 8.6, 30), "cone6": ("cone", 0.9, 8.0, 5),
    "cone7": ("cone", -1.2, 17.0, 15), "cone8": ("cone", 2.2, 18.5, 25),
    "cone9": ("cone", 3.2, 12.05, 0),             # on the ball's line: the 50 mph ball clips it after its bounce
    "drum1": ("drum", -6.2, 13.4, 10), "drum2": ("drum", -6.9, 14.0, 70), "drum3": ("drum", -5.4, 9.0, 30),
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
