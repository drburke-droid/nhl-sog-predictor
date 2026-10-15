"""World night-globe texture for the long video (equirectangular 16384x8192, uint8): hillshaded relief
from the global max-pooled mosaic (60S-84N, world.py), faint country borders, near-black ocean.

    python globe_texture_world.py   -> data/globe_tex_world.png
"""
import os, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NE = os.path.join(os.path.dirname(HERE), "data", "ne")
TW, TH = 16384, 8192


def rings(path):
    gj = json.load(open(path))
    for f in gj["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            for k, ring in enumerate(poly):
                yield k, np.asarray(ring)


def px(c):
    return (np.c_[(c[:, 0] + 180) / 360 * TW, (90 - c[:, 1]) / 180 * TH] * 16).astype(np.int32)


def main():
    meta = json.load(open(os.path.join(DATA, "mosaic_global_meta.json")))
    mos = np.load(os.path.join(DATA, "mosaic_global.npy"), mmap_mode="r")
    W, row0 = meta["width"], meta["row0_px"]
    lat = 90 - (np.arange(TH) + 0.5) * 180 / TH
    elev = np.zeros((TH, TW), np.float32)
    band = np.nonzero((lat > -59.8) & (lat < 83.8))[0]
    s = np.sin(np.radians(lat[band]))
    my = 0.5 * np.log((1 + s) / (1 - s))
    rows = np.clip(((0.5 - my / (2 * np.pi)) * W - row0).astype(int), 0, mos.shape[0] - 1)
    for r, mr in zip(band, rows):
        elev[r] = np.asarray(mos[mr], np.float32).reshape(TW, -1).mean(1)
    # Greenland's ice sheet comes from a blocky source (flat terraces, holes): smooth its interior so the
    # hillshade shows the dome, not the tiles
    ys0, ys1 = int((90 - 84) / 180 * TH), int((90 - 59) / 180 * TH)
    xs0, xs1 = int((180 - 75) / 360 * TW), int((180 - 10) / 360 * TW)
    box = elev[ys0:ys1, xs0:xs1]
    hole = (box < 1500) & (cv2.dilate((box > 2000).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & \
           (cv2.erode((box > 1000).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    box = np.where(hole, cv2.medianBlur(box, 5), box)
    sm = cv2.GaussianBlur(box, (0, 0), 10)
    w = np.clip((cv2.GaussianBlur((box > 1400).astype(np.float32), (0, 0), 12) - 0.5) * 2.5, 0, 1)
    elev[ys0:ys1, xs0:xs1] = box * (1 - w) + sm * w
    fade = np.clip((lat + 59.8) / 3.0, 0, 1) * np.clip((83.8 - lat) / 2.0, 0, 1)
    elev *= (fade * fade * (3 - 2 * fade))[:, None].astype(np.float32)
    land = np.zeros((TH, TW), np.uint8)
    for k, c in rings(os.path.join(NE, "ne_10m_land.geojson")):
        cv2.fillPoly(land, [px(c)], 255 if k == 0 else 0, cv2.LINE_AA, shift=4)
    land = land.astype(np.float32) / 255
    km_px = 40075 / TW
    dx = km_px * np.cos(np.radians(lat))[:, None] * 1000
    gx = (np.roll(elev, -1, 1) - np.roll(elev, 1, 1)) / (2 * np.maximum(dx, 50))
    gy = (np.roll(elev, -1, 0) - np.roll(elev, 1, 0)) / (2 * km_px * 1000)
    del dx
    az, alt = np.radians(315), np.radians(40)
    slope = np.arctan(np.hypot(gx, gy) * 3.0)
    aspect = np.arctan2(-gx, gy)
    del gx, gy
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    del slope, aspect
    hnorm = np.clip(elev / 3000, 0, 1) ** 0.7
    del elev
    tex = np.empty((TH, TW, 3), np.float32)
    base = np.array([0.075, 0.095, 0.13]); hi = np.array([0.20, 0.23, 0.28]); ocean = np.array([0.012, 0.022, 0.045])
    for ch in range(3):
        col = (base[ch] + (hi[ch] - base[ch]) * hnorm) * (0.55 + 0.75 * shade)
        tex[..., ch] = ocean[ch] * (1 - land) + col * land
    del hnorm, shade
    coast = cv2.morphologyEx((land > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32)
    tex += coast[..., None] * np.array([0.05, 0.07, 0.10], np.float32)
    border = np.zeros((TH, TW), np.uint8)
    for k, c in rings(os.path.join(NE, "ne_10m_admin_0_countries.geojson")):
        cv2.polylines(border, [px(c)], True, 255, 1, cv2.LINE_AA, shift=4)
    b = (border.astype(np.float32) / 255) * land
    tex += b[..., None] * np.array([0.06, 0.075, 0.10], np.float32)
    cv2.imwrite(os.path.join(DATA, "globe_tex_world.png"), (np.clip(tex, 0, 1) * 255)[..., ::-1].astype(np.uint8))
    print("ok")


if __name__ == "__main__":
    main()
