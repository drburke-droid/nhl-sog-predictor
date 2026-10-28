import cv2, numpy as np, sys
sys.path.insert(0, "/home/user/lot/src")
import comp as C
D = "/home/user/lot/render/dust_blast"
fr = [int(a) for a in sys.argv[1:]]
rows = []
for f in fr:
    bg = C.plate_loop(60 + f)
    d = cv2.imread(f"{D}/d{f:04d}.png", cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 32767.5
    g = cv2.imread(f"{D}/g{f:04d}.png", cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535
    d = cv2.resize(d, (1920, 804)); g = cv2.resize(g, (1920, 804))
    gs, tr = g[..., 1:2], g[..., 0:1]
    img = bg * (1 - 0.45 * (1 - gs)) * tr + d
    rows.append(cv2.resize((C.srgb(img) * 255).astype(np.uint8)[..., ::-1], (960, 402)))
cv2.imwrite("/home/user/lot/look/dust_preview.png", np.vstack(rows))
