"""Natural Earth projection (Savric et al. 2011), Atlantic-centred, cut at the Bering Strait."""
import numpy as np

W, H = 1920, 1080
LON0 = 10.0          # central meridian; the map edge (lon -170) falls in the Bering Strait
SCALE = 330.0        # px per projection unit
CX = W / 2
CY = 540.0           # pixel row of the equator


def natural_earth(lon, lat):
    lam = np.radians(((np.asarray(lon, dtype=np.float64) - LON0 + 180.0) % 360.0) - 180.0)
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    p2 = phi * phi
    p4 = p2 * p2
    x = lam * (0.8707 - 0.131979 * p2 + p4 * (-0.013791 + p4 * p2 * (0.003971 - 0.001529 * p2)))
    y = phi * (1.007226 + p2 * (0.015085 + p4 * (-0.044475 + 0.028874 * p2 - 0.005916 * p4)))
    return x, y


def to_px(lon, lat):
    x, y = natural_earth(lon, lat)
    return CX + x * SCALE, CY - y * SCALE
