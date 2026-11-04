"""The table's printed map (data/table.png, 3840x2160 = 2x map pixels): ocean, land, coastlines,
graticule, and hairlines from the magnified island insets to their real places."""
import json, os
import numpy as np, cv2
from shapely.geometry import shape
from proj import W, H, to_px
import story as S, vessels as V

SS = 2
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def main():
    gj = json.load(open(os.path.join(DATA, "ne_50m_land.geojson")))
    rings, holes = [], []
    for f in gj["features"]:
        for p in V.split_cut(shape(f["geometry"])):
            if p.bounds[3] < -58:
                continue        # no Antarctica
            for ring, dst in [(p.exterior, rings)] + [(r, holes) for r in p.interiors]:
                xy = np.array(ring.coords)
                x, y = to_px(np.minimum(xy[:, 0], 189.999), np.maximum(xy[:, 1], -89.9))
                dst.append(np.round(np.stack([x, y], 1) * SS * 16).astype(np.int32))
    hh, ww = H * SS, W * SS
    ocean, land, coast = rgb("#060b15"), rgb("#121b2b"), rgb("#2c3f5e")
    img = np.ones((hh, ww, 3), np.float32) * ocean
    m = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(m, rings, 255, cv2.LINE_AA, shift=4)
    cv2.fillPoly(m, holes, 0, cv2.LINE_AA, shift=4)
    a = (m / 255.0)[..., None]
    img = img * (1 - a) + land * a
    # graticule, very faint
    g = np.zeros((hh, ww), np.uint8)
    for lon in range(-180, 181, 30):
        lat = np.linspace(-85, 85, 200)
        x, y = to_px(np.full_like(lat, lon if lon < 190 else lon - 360), lat)
        if lon in (-180, 180):
            continue
        cv2.polylines(g, [np.round(np.stack([x, y], 1) * SS * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
    for lat in range(-60, 90, 30):
        lon = np.linspace(-169.9, 189.9, 400)
        x, y = to_px(lon, np.full_like(lon, lat))
        cv2.polylines(g, [np.round(np.stack([x, y], 1) * SS * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
    ga = (g / 255.0)[..., None] * 0.12
    img = img * (1 - ga) + coast * ga
    c = np.zeros((hh, ww), np.uint8)
    cv2.polylines(c, rings, True, 255, SS, cv2.LINE_AA, shift=4)
    ca = (c / 255.0)[..., None] * 0.6
    img = img * (1 - ca) + coast * ca
    # inset hairlines
    G = V.build(S.VESSELS)
    for vid, gm in G.items():
        if gm["geo"] is None:
            continue
        bx0, by0, bx1, by1 = gm["bbox"]
        gx, gy = gm["geo"]
        qx, qy = min(max(gx, bx0 - 8), bx1 + 8), min(max(gy, by0 - 8), by1 + 8)
        l = np.zeros((hh, ww), np.uint8)
        cv2.line(l, (int(qx * SS * 16), int(qy * SS * 16)), (int(gx * SS * 16), int(gy * SS * 16)), 255, SS, cv2.LINE_AA, 4)
        cv2.circle(l, (int(gx * SS * 16), int(gy * SS * 16)), int(3 * SS * 16), 255, -1, cv2.LINE_AA, 4)
        la = (l / 255.0)[..., None] * 0.55
        img = img * (1 - la) + rgb("#8d9bb3") * la
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DATA, "table.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("wrote table.png")


if __name__ == "__main__":
    main()
