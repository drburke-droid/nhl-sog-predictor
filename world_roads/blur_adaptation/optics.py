"""Eye-optics simulation: Zernike wavefronts, point-spread functions, rotation, blurred stimuli.

Units: wavefront in microns over a 6 mm pupil (as in Artal et al. 2004), 550 nm light.
"""
import math
import numpy as np
from scipy.ndimage import zoom, map_coordinates

LAM_UM = 0.55
NP = 256                     # pupil samples across the 6 mm diameter
NFFT = 1024
PUPIL_MM = 6.0
ARCMIN_PER_PX = math.degrees(LAM_UM * 1e-6 / (NFFT * PUPIL_MM * 1e-3 / NP)) * 60   # ≈ 0.079

yy, xx = (np.mgrid[0:NP, 0:NP] + 0.5 - NP / 2) / (NP / 2)
RHO = np.hypot(xx, yy)
THETA = np.arctan2(yy, xx)
PUPIL = (RHO <= 1.0).astype(np.float32)


def radial(n, m, r):
    m = abs(m)
    out = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        out += ((-1) ** k * math.factorial(n - k) /
                (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))) * r ** (n - 2 * k)
    return out


def zernike(n, m, r=RHO, th=THETA):
    """OSA/ANSI normalized Zernike Z_n^m (RMS 1 over the unit pupil)"""
    norm = math.sqrt(2 * (n + 1)) if m != 0 else math.sqrt(n + 1)
    ang = np.cos(m * th) if m >= 0 else np.sin(-m * th)
    return norm * radial(n, m, r) * ang


# three illustrative "normal" eyes (higher-order terms, microns RMS per mode, 6 mm pupil)
EYES = {
    "A": {(3, -1): 0.16, (3, 1): -0.09, (3, -3): 0.13, (3, 3): 0.05, (4, 0): 0.10, (4, 2): 0.05, (4, -4): 0.05},
    "B": {(3, -1): -0.07, (3, 1): 0.17, (3, -3): -0.06, (3, 3): 0.14, (4, 0): 0.13, (4, -2): 0.06, (4, 4): -0.05},
    "C": {(3, -1): 0.05, (3, 1): 0.06, (3, -3): 0.18, (3, 3): -0.10, (4, 0): 0.06, (4, 2): -0.07, (5, -1): 0.05},
}
LOW = {(2, 0): 0.55, (2, 2): 0.32}           # defocus + astigmatism, the part glasses fix


def wavefront(coeffs, rot_deg=0.0, scale=1.0):
    th = THETA - math.radians(rot_deg)
    W = np.zeros_like(RHO)
    for (n, m), c in coeffs.items():
        W += c * zernike(n, m, RHO, th)
    return W * PUPIL * scale


def rms(W):
    v = W[PUPIL > 0]
    return float(np.sqrt(((v - v.mean()) ** 2).mean()))


def psf(W, crop_arcmin=40.0):
    """PSF (peak-normalized to the diffraction-limited peak) cropped to crop_arcmin square."""
    field = np.zeros((NFFT, NFFT), np.complex64)
    field[:NP, :NP] = PUPIL * np.exp(2j * np.pi * W / LAM_UM)
    I = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    I0 = PUPIL.sum() ** 2
    half = int(round(crop_arcmin / 2 / ARCMIN_PER_PX))
    c = NFFT // 2
    return (I[c - half:c + half, c - half:c + half] / I0).astype(np.float32)


def strehl(W):
    return float(psf(W, 4).max())


# ------------------------------------------------------------------ the stimulus (binary noise, as in the paper)
def binary_noise(n=600, seed=0, lo=0.04, hi=0.16):
    """annular-band-pass filtered uniform noise, thresholded to black/white (edges at all orientations)"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, n))
    f = np.fft.fftfreq(n)
    fr = np.hypot(*np.meshgrid(f, f))
    X = np.fft.fft2(x) * ((fr > lo) & (fr < hi))
    y = np.real(np.fft.ifft2(X))
    return (y > 0).astype(np.float32)


def blur_with(img, W, px_arcmin):
    """convolve an image (px_arcmin per pixel) with the PSF of wavefront W"""
    n = img.shape[0]
    P = psf(W, crop_arcmin=min(n * px_arcmin, 60.0))
    P = zoom(P, ARCMIN_PER_PX / px_arcmin, order=1)
    P = np.clip(P, 0, None)[:n, :n]
    P /= P.sum()
    K = np.zeros_like(img)
    h, w = P.shape
    K[:h, :w] = P
    K = np.roll(K, (-(h // 2), -(w // 2)), (0, 1))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(K))).astype(np.float32)


# ------------------------------------------------------------------ colour maps (Light Speed palette)
def heat(I, gamma=0.45, gain=1.0):
    """PSF intensity -> RGB: navy -> deep orange -> gold -> white-hot"""
    v = np.clip((I / (I.max() + 1e-12)) * gain, 0, 1) ** gamma
    stops = np.array([[0.0, 0.0, 0.0], [0.45, 0.13, 0.02], [1.0, 0.56, 0.13], [1.0, 0.85, 0.55], [1.0, 0.98, 0.92]],
                     np.float32)
    pos = np.array([0, 0.3, 0.6, 0.85, 1.0])
    out = np.empty(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, pos, stops[:, c])
    return out


def diverge(W, lim=None):
    """wavefront -> RGB: cyan (behind) / navy (flat) / gold (ahead), outside pupil transparent-black"""
    lim = lim or (np.abs(W).max() + 1e-9)
    v = np.clip(W / lim, -1, 1)
    navy = np.array([0.04, 0.07, 0.13], np.float32)
    cyan = np.array([0.16, 0.62, 0.98], np.float32)
    gold = np.array([1.0, 0.62, 0.18], np.float32)
    a = np.abs(v)[..., None] ** 0.8
    col = np.where(v[..., None] < 0, navy * (1 - a) + cyan * a, navy * (1 - a) + gold * a)
    return col * PUPIL[..., None]


def lenslet_spots(W, n=15, gain=1.0):
    """Hartmann-Shack: lenslet centres (pupil units) inside the pupil and spot displacements ~ local slope"""
    gy, gx = np.gradient(W)
    cs = (np.arange(n) + 0.5) / n * 2 - 1
    out = []
    for yv in cs:
        for xv in cs:
            if math.hypot(xv, yv) <= 1.0 - 0.5 / n:
                i = int((yv + 1) / 2 * NP)
                j = int((xv + 1) / 2 * NP)
                out.append((xv, yv, gx[i, j] * gain, gy[i, j] * gain))
    return out


def actuators(n=11, target=97):
    """deformable-mirror actuator positions (pupil units), a square grid clipped to a circle of ~97"""
    cs = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    pts = [(x, y) for y in cs for x in cs]
    pts.sort(key=lambda p: math.hypot(*p))
    return pts[:target]


def sample(W, x, y):
    i = np.clip(((np.asarray(y) + 1) / 2 * NP), 0, NP - 1)
    j = np.clip(((np.asarray(x) + 1) / 2 * NP), 0, NP - 1)
    return map_coordinates(W, [i, j], order=1)
