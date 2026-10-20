"""Retinal images on the vernier film's cone mosaic: a Snellen E and a golf ball.

Retina coordinates are cone spacings (0.5′), as in ../vernier/retina.py. Images are worked in the
frequency domain: the object's spectrum times the eye's optical transfer function (Westheimer's, the one
the vernier film uses) and, for what each cone collects, times its aperture (a disk 0.5′ wide).

The micrograph shows each cone's light in true proportion (linear light, then sRGB): a cone that gets 18%
more light is drawn 18% brighter.
"""
import os, sys, math, functools
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VERNIER = os.path.join(os.path.dirname(HERE), "vernier")
sys.path.insert(0, VERNIER)
sys.path.insert(0, HERE)
import look as L                                       # noqa: E402  (the vernier film's look)
import retina as R                                     # noqa: E402
import physics as P                                    # noqa: E402

L.OUT = os.path.join(HERE, "out")
W, H = L.W, L.H
U = P.CONE                                             # arcmin per retina unit (one cone spacing)
APERTURE = 0.5                                         # a foveal cone's aperture, arcmin

CONE_LIN = L.to_lin(np.array([246, 218, 186], np.uint8)).astype(np.float32)   # a cone in full light
GAP_LIN = L.to_lin(np.array([44, 34, 38], np.uint8)).astype(np.float32)       # the membranes between cones


# ------------------------------------------------------------------ the E
E_GRID = 0.025                                         # retina units per sample
E_N = 1024                                             # samples a side (25.6 units: 12.8′)


@functools.lru_cache(None)
def e_mask(size=10.0):
    """a Snellen E, `size` retina units tall (10 = 5′ = 20/20), opening to the right, centred at 0; anti-
    aliased coverage on the E grid (0 = paper, 1 = ink)"""
    q = 8
    n = E_N * q
    img = np.zeros((n, n), np.uint8)
    c = n / 2
    s = size / 5 / E_GRID * q                           # one stroke, in fine samples
    x0, y0 = c - 2.5 * s, c - 2.5 * s
    cv2.rectangle(img, (int(round(x0)), int(round(y0))), (int(round(x0 + s)) - 1, int(round(y0 + 5 * s)) - 1), 255, -1)
    for k in (0, 2, 4):
        cv2.rectangle(img, (int(round(x0)), int(round(y0 + k * s))), (int(round(x0 + 5 * s)) - 1, int(round(y0 + (k + 1) * s)) - 1), 255, -1)
    return cv2.resize(img.astype(np.float32) / 255, (E_N, E_N), interpolation=cv2.INTER_AREA)


@functools.lru_cache(None)
def e_image(size=10.0, aperture=True, ink=0.94):
    """the E's image on the retina: light (1 = paper) on the E grid, blurred by the eye (and each cone's
    aperture)"""
    m = e_mask(size)
    f = np.fft.fftfreq(E_N, E_GRID * U)                  # cycles per arcmin
    rho = np.hypot(f[None, :], f[:, None])
    T = P.otf(rho)
    if aperture:
        T = T * P.jinc(APERTURE * rho)
    blur = np.real(np.fft.ifft2(np.fft.fft2(m) * T))
    return (1 - ink * blur).astype(np.float32)


def e_sample(img, u, v):
    """sample an E-grid image at retina coordinates (u, v)"""
    u, v = np.broadcast_arrays(np.asarray(u, np.float64), np.asarray(v, np.float64))
    shape = u.shape
    x = (u / E_GRID + E_N / 2 - 0.5).reshape(1, -1) if u.ndim < 2 else u / E_GRID + E_N / 2 - 0.5
    y = (v / E_GRID + E_N / 2 - 0.5).reshape(1, -1) if v.ndim < 2 else v / E_GRID + E_N / 2 - 0.5
    out = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(shape)


def e_sharp_at(u, v, size=10.0, ink=0.94):
    """'pixel logic': ink if the cone's centre is on the E, paper if not"""
    s = size / 5
    u = np.asarray(u, np.float64) + 2.5 * s
    v = np.asarray(v, np.float64) + 2.5 * s
    inside = (u >= 0) & (u < 5 * s) & (v >= 0) & (v < 5 * s)
    col = np.floor(u / s)
    row = np.floor(v / s)
    on = inside & ((col == 0) | (row % 2 == 0))
    return 1 - ink * on.astype(np.float64)


# ------------------------------------------------------------------ the ball
@functools.lru_cache(512)
def ball_profile(diam_arcmin, aperture=True):
    """the ball's image: change in light (as a share of the ball's Weber contrast) against radius in retina
    units, on a grid from 0 to 12 units"""
    r = np.linspace(0, 12, 1201)
    return r, P.disk_image(r * U, diam_arcmin, 1.0, APERTURE if aperture else 0.0)


def ball_at(u, v, yd, aperture=True, cu=0.0, cv=0.0):
    """the ball's image at retina points (u, v), per unit Weber contrast; the ball is centred at (cu, cv).
    yd may be a float; the profile is cached on a fine grid of diameters."""
    d = float(P.ball_arcmin(yd))
    d = round(d, 4)
    rr, pr = ball_profile(d, aperture)
    r = np.hypot(np.asarray(u, np.float64) - cu, np.asarray(v, np.float64) - cv)
    return np.interp(r, rr, pr, right=0.0)


def ball_sharp_at(u, v, yd, cu=0.0, cv=0.0):
    """'pixel logic': 1 if the cone's centre is on the ball, 0 if not"""
    rad = float(P.ball_arcmin(yd)) / 2 / U
    return (np.hypot(np.asarray(u, np.float64) - cu, np.asarray(v, np.float64) - cv) <= rad).astype(np.float64)


# ------------------------------------------------------------------ the micrograph
DIM_LIN = L.to_lin(np.array([66, 50, 52], np.uint8)).astype(np.float32)


def shade(lv, curve):
    """cone colours (linear light) for light levels lv. curve 'linear': in true proportion. curve 'print':
    the vernier film's tone curve (light ** 1.8 between a dim and a bright cone, in sRGB), which makes a
    black-on-white pattern read clearly."""
    if curve == "linear":
        return CONE_LIN[None, :] * lv[:, None]
    srgb = np.array([66, 50, 52], np.float32) + np.array([180, 168, 134], np.float32) * (np.clip(lv, 0, 1) ** 1.8)[:, None]
    return L.to_lin(np.clip(srgb, 0, 255).astype(np.uint8)).astype(np.float32)


def mosaic(scope, light, overlay=None, band=0.0, exposure=1.0, centre=None, radius=None, field=True, glow=None,
           curve="linear"):
    """the micrograph, uint8 RGB (H, W, 3).
    light(u, v): each cone's light (1 = full light), at the cones' centres (retina units).
    overlay(u, v): the image itself, drawn continuous over the cones with opacity `band`.
    exposure: scales all light (as a camera's exposure would).
    glow: optional per-cone highlight (array over the lattice, 0..1) in red pencil colour.
    curve: 'linear' (true proportion) or 'print' (see shade)."""
    Q = R.Q
    p, tone = R.lattice()
    w, h = W // Q, H // Q
    k = scope.k / Q
    ccx, ccy = centre if centre is not None else (W * R.FIELD_CX, H * R.FIELD_CY)
    ccx, ccy = ccx / Q, ccy / Q
    idx, label, cover, inner = R.layout(scope.u, scope.v, k, ccx, ccy)
    lv = np.clip(np.asarray(light(p[idx, 0], p[idx, 1]), np.float32) * exposure, 0, 1.4)
    cols = shade(lv, curve) * (1 + 0.04 * tone[idx])[:, None]
    if glow is not None:
        g = glow[idx][:, None]
        cols = cols * (1 - g) + L.to_lin(np.array([212, 88, 66], np.uint8))[None, :] * g
    cols = np.vstack([cols * 0.82, GAP_LIN[None, :]]).astype(np.float32)
    img = GAP_LIN + (cols[label] - GAP_LIN) * cover[..., None]
    img *= (1 + 0.22 * inner)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), max(0.6, k * 0.025))
    if band > 0 and overlay is not None:
        vs = (np.arange(h, dtype=np.float32) - ccy) / k + scope.v
        us = (np.arange(w, dtype=np.float32) - ccx) / k + scope.u
        Lm = np.clip(np.asarray(overlay(us[None, :].repeat(h, 0), vs[:, None].repeat(w, 1)), np.float32) * exposure, 0, 1.4)
        img = img * (1 - band) + shade(Lm.ravel(), curve).reshape(h, w, 3) * 0.82 * band
    out = L.to_srgb(img).astype(np.float32) + R.photo_grain()
    if field:
        out = R.microscope(out, (ccx, ccy, (radius or H * R.FIELD_R) / Q), Q)
    return cv2.resize(np.clip(out, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
