"""Fetch USDA NAIP aerial imagery (public domain) around a point, north up, in UTM metres.

python fetch_naip.py NAME LON LAT HALF_WIDTH_M [YEAR]
    -> $B2_WORK/naip/NAME.png and NAME.json (extent in UTM, metres per pixel, the source scenes, acquisition dates)

The scenes come from Microsoft Planetary Computer's copy of NAIP (cloud-optimized GeoTIFFs, read by window, so only
the square asked for is downloaded). The USGS National Map's NAIP image service has the same imagery but times out on
exports this size.
"""
import json, os, sys, urllib.request
import numpy as np
from PIL import Image

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
WORK = os.environ.get("B2_WORK", "/home/user/lot/b2")


def _json(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def fetch(name, lon, lat, half, year=None):
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    feats = _json(STAC, {"collections": ["naip"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                         "limit": 20})["features"]
    if year:
        feats = [f for f in feats if str(f["properties"].get("naip:year")) == str(year)]
    feats.sort(key=lambda f: (f["properties"]["datetime"], -float(f["properties"].get("gsd", 1))), reverse=True)
    item = feats[0]
    token = _json(SAS)["token"]
    href = item["assets"]["image"]["href"] + "?" + token
    with rasterio.open(href) as src:
        cx, cy = Transformer.from_crs(4326, src.crs, always_xy=True).transform(lon, lat)
        b = (cx - half, cy - half, cx + half, cy + half)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        rgb = src.read([1, 2, 3], window=win, boundless=False)
        mpp = abs(src.transform.a)
        epsg = src.crs.to_epsg()
    img = np.transpose(rgb, (1, 2, 0))
    os.makedirs(os.path.join(WORK, "naip"), exist_ok=True)
    Image.fromarray(img).save(os.path.join(WORK, "naip", name + ".png"))
    meta = dict(name=name, lon=lon, lat=lat, epsg=epsg, center_utm=[cx, cy], m_per_px=mpp, size_px=list(img.shape[1::-1]),
                extent_utm=list(b), item=item["id"], acquired=item["properties"]["datetime"],
                source="USDA NAIP (public domain), via Microsoft Planetary Computer")
    json.dump(meta, open(os.path.join(WORK, "naip", name + ".json"), "w"), indent=1)
    return meta


if __name__ == "__main__":
    a = sys.argv[1:]
    print(fetch(a[0], float(a[1]), float(a[2]), float(a[3]), a[4] if len(a) > 4 else None))
