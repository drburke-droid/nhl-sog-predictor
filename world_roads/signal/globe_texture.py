"""Night-globe texture (equirectangular 8192x4096): dark land with hillshaded relief from the same
Terrarium mosaic (18-84N), plain land elsewhere, near-black ocean.   python globe_texture.py"""
import os, json
import numpy as np
import cv2
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne", "ne_10m_land.geojson")
TW, TH = 8192, 4096


def main():
    meta = json.load(open(os.path.join(DATA, "mosaic_meta.json")))
    mos = np.load(os.path.join(DATA, "z7_mosaic.npy"), mmap_mode="r")
    W, row0 = meta["width"], meta["row0_px"]
    lat = 90 - (np.arange(TH) + 0.5) * 180 / TH
    elev = np.zeros((TH, TW), np.float32)
    band = np.nonzero((lat > 18.2) & (lat < 83.8))[0]
    s = np.sin(np.radians(lat[band]))
    my = 0.5 * np.log((1 + s) / (1 - s))
    rows = np.clip(((0.5 - my / (2 * np.pi)) * W - row0).astype(int), 0, mos.shape[0] - 1)
    for r, mr in zip(band, rows):
        elev[r] = np.asarray(mos[mr], np.float32).reshape(TW, -1).mean(1)
    fade = np.clip((lat - 18.2) / 7.0, 0, 1) * np.clip((83.8 - lat) / 2.0, 0, 1)
    elev *= (fade * fade * (3 - 2 * fade))[:, None].astype(np.float32)   # blend relief out at the band edges
    # land mask
    land = np.zeros((TH, TW), np.uint8)
    gj = json.load(open(NE))
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for k, ring in enumerate(poly):
                c = np.asarray(ring)
                pts = np.c_[(c[:, 0] + 180) / 360 * TW, (90 - c[:, 1]) / 180 * TH] * 16
                cv2.fillPoly(land, [pts.astype(np.int32)], 255 if k == 0 else 0, cv2.LINE_AA, shift=4)
    land = land.astype(np.float32) / 255
    # hillshade (light from the north-west), x spacing shrinks with latitude
    km_px = 40075 / TW
    dx = km_px * np.cos(np.radians(lat))[:, None] * 1000
    dy = km_px * 1000
    gx = (np.roll(elev, -1, 1) - np.roll(elev, 1, 1)) / (2 * np.maximum(dx, 50))
    gy = (np.roll(elev, -1, 0) - np.roll(elev, 1, 0)) / (2 * dy)
    az, alt = np.radians(315), np.radians(40)
    slope = np.arctan(np.hypot(gx, gy) * 3.0)
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    hnorm = np.clip(elev / 3000, 0, 1)
    base = np.array([0.075, 0.095, 0.13], np.float32)
    hi = np.array([0.20, 0.23, 0.28], np.float32)
    col = base + (hi - base) * hnorm[..., None] ** 0.7
    col = col * (0.55 + 0.75 * shade[..., None])
    ocean = np.array([0.012, 0.022, 0.045], np.float32)
    tex = ocean * (1 - land[..., None]) + col * land[..., None]
    coast = cv2.morphologyEx((land > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    tex += coast[..., None] * np.array([0.05, 0.07, 0.10], np.float32)
    cv2.imwrite(os.path.join(DATA, "globe_tex.png"), (np.clip(tex, 0, 1) * 255)[..., ::-1].astype(np.uint8))
    print("ok")


if __name__ == "__main__":
    main()
