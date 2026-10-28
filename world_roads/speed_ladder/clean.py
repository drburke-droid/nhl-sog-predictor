"""Paint the pickup on the far-left road out of every stabilized frame (it drives across frames 0-130) with the same
patch of the last frame, where it has gone; then crop to the region valid in every frame and scale to 1920x804.
-> /home/user/lot/plate/p###.png"""
import cv2, numpy as np, os
S, OUT = "/home/user/lot/stab", "/home/user/lot/plate"
CROP = (40, 20, 2700, 1134)                  # x0, y0, x1, y1 in reference-frame pixels (2.39:1)
SIZE = (1920, 804)
os.makedirs(OUT, exist_ok=True)
clean = cv2.imread(f"{S}/s148.png").astype(np.float32)
for i in range(149):
    s = cv2.imread(f"{S}/s{i:03d}.png").astype(np.float32)
    band = (slice(500, 570), slice(0, 760))
    d = np.abs(s[band] - clean[band]).max(axis=2)
    m = np.zeros(s.shape[:2], np.float32)
    m[band] = (d > 28).astype(np.float32)
    m = cv2.dilate(m, np.ones((15, 25), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 4)[..., None]
    s = s * (1 - m) + clean * m
    x0, y0, x1, y1 = CROP
    cv2.imwrite(f"{OUT}/p{i:03d}.png", cv2.resize(np.clip(s[y0:y1, x0:x1], 0, 255).astype(np.uint8), SIZE, interpolation=cv2.INTER_AREA))
print("done")
