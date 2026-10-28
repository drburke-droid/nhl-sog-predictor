"""An environment map for reflections and fill light, built from the plate itself: every direction inside the camera's
view takes the plate's colour there; outside it, azimuth folds back into the view (mirror tiling) and elevation clamps
to the nearest row, with the sky fading to a deeper blue toward the zenith. Linear, equirectangular, 2048x1024."""
import cv2, numpy as np, math
import camera as C
plate = cv2.imread("/home/user/lot/plate/p074.png")[..., ::-1].astype(np.float32) / 255
lin = np.where(plate <= 0.04045, plate / 12.92, ((plate + 0.055) / 1.055) ** 2.4)
W, H = 2048, 1024
u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
az = (u - 0.5) * 2 * math.pi                     # Blender equirect: u=0.5 looks along -X? handled in scene via mapping
el = (0.5 - v) * math.pi
AZ, EL = np.meshgrid(az, el)
# direction in world: azimuth measured from the camera's forward (+Y), positive to the right (+X)
half = math.atan(C.W / 2 / C.F)
fold = np.abs(((AZ + half) % (4 * half)) - 2 * half) - half      # triangle wave into [-half, half]
top, bot = math.atan(C.CY / C.F), -math.atan((C.H_IMG - C.CY) / C.F)
elc = np.clip(EL, bot + 0.002, top - 0.002)
d = np.stack([np.sin(fold) * np.cos(elc), np.cos(fold) * np.cos(elc), np.sin(elc)], axis=-1)
c = d @ C.R().T
x = (C.CX + C.F * c[..., 0] / c[..., 2]).astype(np.float32)
y = (C.CY + C.F * c[..., 1] / c[..., 2]).astype(np.float32)
env = cv2.remap(np.ascontiguousarray(lin), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
env = cv2.GaussianBlur(env, (0, 0), 1.2)
# beyond the plate's top row the clamped rows streak: smear them along the azimuth instead, more with height
ext = np.clip((EL - top) / 0.35, 0, 1)[..., None]
smear = cv2.GaussianBlur(env, (0, 0), sigmaX=90, sigmaY=1)
env = env * (1 - ext) + smear * ext
zen = np.clip((EL - top) / (math.pi / 2 - top), 0, 1)[..., None]
deep = np.array([0.18, 0.32, 0.62], np.float32)
env = env * (1 - 0.7 * zen) + deep * 0.7 * zen
nadir = np.clip((bot - EL) / (math.pi / 2 + bot), 0, 1)[..., None]
env = env * (1 - 0.5 * nadir) + env[-1:].mean(axis=1, keepdims=True) * 0.5 * nadir
env = np.roll(env, W // 4, axis=1)        # Blender's equirect looks along -X at u = 0.5; forward (+Y) is u = 0.75
cv2.imwrite("/home/user/lot/env.hdr", env[..., ::-1].astype(np.float32))
cv2.imwrite("/home/user/lot/look/env_preview.png", (np.clip(env, 0, 1) ** (1 / 2.2) * 255)[..., ::-1].astype(np.uint8))
print("env", env.min(), env.max(), env.mean())
