"""Prepare the downloaded assets (see film/README.md for the downloads).

python3 film/prep_assets.py dem             cut dem/everest.npz from the six Copernicus 30 m tiles
blender -b -P film/prep_assets.py -- sun    find each sky HDRI's sun (u, v, elevation) -> hdri/sun.json
"""
import json
import math
import sys

A = '/opt/seveneves_assets'


def dem():
    import numpy as np
    import tifffile

    def rd(n):
        with tifffile.TiffFile(f'{A}/dem/Copernicus_DSM_COG_10_{n}_DEM.tif') as tf:
            return tf.pages[0].asarray()
    top = np.hstack([rd(f'N28_00_E0{lon}_00') for lon in (86, 87, 88)])
    bot = np.hstack([rd(f'N27_00_E0{lon}_00') for lon in (86, 87, 88)])
    full = np.vstack([top, bot])                                   # rows 29 N -> 27 N, columns 86 E -> 89 E
    lat = 29 - (np.arange(full.shape[0]) + 0.5) / 3600.0
    lon = 86 + (np.arange(full.shape[1]) + 0.5) / 3600.0
    r0, r1 = np.searchsorted(-lat, -28.55), np.searchsorted(-lat, -27.42)
    c0, c1 = np.searchsorted(lon, 86.05), np.searchsorted(lon, 87.95)
    np.savez(f'{A}/dem/everest.npz', h=full[r0:r1, c0:c1].astype(np.float32), lat=lat[r0:r1], lon=lon[c0:c1])


def sun():
    import bpy
    import numpy as np
    out = {}
    for name in ('kloofendal_48d_partly_cloudy_puresky_4k', 'kloofendal_43d_clear_puresky_4k'):
        im = bpy.data.images.load(f'{A}/hdri/{name}.hdr')
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        lum = px.reshape(h, w, 4)[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        r, c = np.unravel_index(np.argmax(lum), lum.shape)
        u, v = (c + 0.5) / w, (r + 0.5) / h                       # Blender's rows count from the bottom
        out[name] = dict(u=float(u), v=float(v), elev=float((v - 0.5) * 180), peak=float(lum.max()))
    json.dump(out, open(f'{A}/hdri/sun.json', 'w'), indent=1)


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    {'dem': dem, 'sun': sun}[args[0]]()
