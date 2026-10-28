"""Lock the drifting drone plate to one frame: warp every frame onto frame REF with the homography of the ground
(fits to 0.1 px; the sky follows within ~1 px). Writes stab/s###.png, the homographies and the region valid in all."""
import cv2, numpy as np, os, json
SRC, OUT = "/home/user/lot/frames", "/home/user/lot/stab"
REF = 74                                   # 0-based
os.makedirs(OUT, exist_ok=True)
F = [cv2.imread(f"{SRC}/f{i + 1:03d}.png") for i in range(149)]
G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F]
h, w = G[0].shape
m = np.zeros_like(G[REF]); m[560:] = 255
p0 = cv2.goodFeaturesToTrack(G[REF], 2000, 0.005, 6, mask=m)
Hs, valid = [], np.full((h, w), 255, np.uint8)
for i in range(149):
    q, st, _ = cv2.calcOpticalFlowPyrLK(G[REF], G[i], p0, None, winSize=(31, 31), maxLevel=4)
    ok = st[:, 0] == 1
    H, _ = cv2.findHomography(q[ok], p0[ok], cv2.RANSAC, 1.5)
    Hs.append(H.tolist())
    cv2.imwrite(f"{OUT}/s{i:03d}.png", cv2.warpPerspective(F[i], H, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT))
    if i >= 40:
        valid &= cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
ys, xs = np.where(valid == 255)
json.dump(dict(ref=REF, H=Hs, valid=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]), open(f"{OUT}/stab.json", "w"))
# the largest axis-aligned box inside the valid region
rows = valid.min(axis=1) == 255
cols = valid.min(axis=0) == 255
print("valid bbox", xs.min(), ys.min(), xs.max(), ys.max(), "full rows", rows.argmax(), h - rows[::-1].argmax(), "full cols", cols.argmax(), w - cols[::-1].argmax())
