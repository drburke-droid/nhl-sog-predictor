"""Map of the anti-GPS 'prison' on the full LA street network: every street in grey, every
intersection reachable by wrong turns in gold.   python anti_gps_map.py -> out/anti_gps_pocket.png"""
import os, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "out")
FD = "/usr/share/fonts/opentype/inter"
W, H = 1920, 1080
X0, X1, Y0, Y1 = -119.35, -117.15, 33.45, 34.55            # view (lon/lat)
KX = np.cos(np.radians(34.0))
sc = min(W / ((X1 - X0) * KX), H / (Y1 - Y0))
ox = (W - (X1 - X0) * KX * sc) / 2; oy = (H - (Y1 - Y0) * sc) / 2
px = lambda lo, la: np.c_[ox + (lo - X0) * KX * sc, oy + (Y1 - la) * sc]
z = np.load(os.path.join(DATA, "socal_edges.npz"))
keep = z["c"] <= 7
a, b = z["a"][keep], z["b"][keep]
dec = lambda c: ((c // 10**9) / 2e5, (c % 10**9 - 20_000_000) / 2e5)
alo, ala = dec(a); blo, bla = dec(b)
inview = (alo > X0) & (alo < X1) & (ala > Y0) & (ala < Y1)
P = np.rint(np.stack([px(alo[inview], ala[inview]), px(blo[inview], bla[inview])], 1) * 4).astype(np.int32)
S = 2
img = np.zeros((H * S, W * S), np.uint8)
cv2.polylines(img, list(P * S), False, 255, 1, cv2.LINE_AA, shift=2)
streets = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
reach = np.load(os.path.join(DATA, "anti_gps_full_socal_8_reach.npy"))
rp = px(reach[:, 0].astype(float), reach[:, 1].astype(float))
heat = np.zeros((H, W), np.float32)
ix = np.clip(rp[:, 0].astype(int), 0, W - 1); iy = np.clip(rp[:, 1].astype(int), 0, H - 1)
np.add.at(heat, (iy, ix), 1.0)
heat = cv2.GaussianBlur(heat, (0, 0), 1.2)
gold = np.clip(heat * 1.6, 0, 1)
E = streets[..., None] * np.array([0.30, 0.42, 0.55], np.float32) * 1.3 + gold[..., None] * np.array([1.0, 0.62, 0.18], np.float32)
bloom = cv2.GaussianBlur(E, (0, 0), 6) * 0.5
out = 1 - np.exp(-(E + bloom) * 1.4)
out = np.maximum(out, np.array([4, 7, 13], np.float32) / 255)
im = Image.fromarray((out * 255).astype(np.uint8))
d = ImageDraw.Draw(im)
f = lambda n, s: ImageFont.truetype(os.path.join(FD, n), s)
cx, cy = px(np.array([-118.2427]), np.array([34.0537]))[0]
d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 255, 255), width=3)
d.text((cx + 14, cy - 34), "Start: LA City Hall", font=f("Inter-SemiBold.otf", 26), fill=(255, 255, 255))
d.text((60, 40), "Ignore your GPS at every turn", font=f("Inter-Bold.otf", 56), fill=(240, 244, 250))
d.text((62, 112), "Gold: every intersection you can ever reach. You never get out.", font=f("Inter-Medium.otf", 30), fill=(255, 186, 80))
d.text((62, 1030), "Every drivable street (OpenStreetMap via Overture Maps). At each intersection: any road except the one GPS suggests.",
       font=f("Inter-Regular.otf", 20), fill=(120, 140, 165))
d.text((W - 60, 520), "New York →\n3,935 km", font=f("Inter-SemiBold.otf", 30), fill=(160, 178, 200), anchor="ra", align="right")
os.makedirs(OUT, exist_ok=True)
im.save(os.path.join(OUT, "anti_gps_pocket.png"))
print("ok")
