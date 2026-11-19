"""Map projection for the coast film: the roads film's Natural Earth projection (../proj.py), placed so the
whole world fits, Antarctica included, with room for a title above the Arctic. RES scales everything
(1 = 1920x1080, 2 = 3840x2160)."""
import os
import numpy as np

RES = int(os.environ.get("RES", "1"))
W, H = 1920 * RES, 1080 * RES
LON0 = 10.0                 # antimeridian cut at 170 W: the Bering Strait
SCALE = 330.0 * RES
CX = W / 2
CY = 572.0 * RES


def natural_earth(lon, lat):
    lam = np.radians(((np.asarray(lon, dtype=np.float64) - LON0 + 180.0) % 360.0) - 180.0)
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    p2 = phi * phi
    p4 = p2 * p2
    x = lam * (0.8707 - 0.131979 * p2 + p4 * (-0.013791 + p4 * p2 * (0.003971 - 0.001529 * p2)))
    y = phi * (1.007226 + p2 * (0.015085 + p4 * (-0.044475 + 0.028874 * p2 - 0.005916 * p4)))
    return x, y


def to_px(lon, lat, ss=1):
    """lon/lat (degrees) -> pixel coordinates on a canvas supersampled by `ss`"""
    x, y = natural_earth(lon, lat)
    return (CX + x * SCALE) * ss, (CY - y * SCALE) * ss
