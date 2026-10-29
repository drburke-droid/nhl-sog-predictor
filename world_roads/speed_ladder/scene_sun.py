"""The sun's direction, shared by Blender (scene.py) and everything that casts shadows outside it."""
import math
SUN_AZ, SUN_EL = 50.0, 20.0         # degrees: to the right of the road direction, above the horizon


def direction():
    az, el = math.radians(SUN_AZ), math.radians(SUN_EL)
    return (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))     # toward the sun
