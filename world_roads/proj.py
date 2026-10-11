"""Shared map projection (Natural Earth projection, Šavrič et al. 2011) and canvas geometry."""
import numpy as np

W, H = 1920, 1080          # output frame
LON0 = 10.0                # central meridian -> antimeridian cut falls in the Bering Strait / mid-Pacific
SCALE = 344.0              # px per projection unit at 1x
CX = W / 2 + 0.0
CY = 604.0                 # vertical placement of the equator (leaves room for the title)


def natural_earth(lon, lat):
    lam = np.radians(((np.asarray(lon, dtype=np.float64) - LON0 + 180.0) % 360.0) - 180.0)
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    p2 = phi * phi
    p4 = p2 * p2
    x = lam * (0.8707 - 0.131979 * p2 + p4 * (-0.013791 + p4 * p2 * (0.003971 - 0.001529 * p2)))
    y = phi * (1.007226 + p2 * (0.015085 + p4 * (-0.044475 + 0.028874 * p2 - 0.005916 * p4)))
    return x, y


def to_px(lon, lat, ss=1):
    """lon/lat (deg) -> pixel coordinates in a canvas supersampled by `ss`."""
    x, y = natural_earth(lon, lat)
    return (CX + x * SCALE) * ss, (CY - y * SCALE) * ss


def lam_wrapped(lon):
    return ((np.asarray(lon, dtype=np.float64) - LON0 + 180.0) % 360.0) - 180.0
