"""Optics of a subjective refraction.

Everything here is standard ophthalmic optics:

* Spherocylinders are added and subtracted as power vectors (M, J0, J45) (Thibos, Wheeler & Horner 1997).
  M is the spherical equivalent; J0 and J45 are the two Jackson cross-cylinder components.
* The residual error the eye sees through a trial lens L is R = E - L, where E is the eye's true
  correcting lens (both in power-vector form, same vertex).
* A positive residual M (artificial hyperopia, e.g. from too much minus) can be cleared by accommodating,
  up to the patient's amplitude. A negative residual M (too much plus, or not enough minus) cannot.
* Blur strength B = |(M_eff, J0, J45)| is the geometric blur of the circle of least confusion, in dioptres.
* Visual acuity: MAR = sqrt(MAR0^2 + (K * pupil * B)^2), a geometric blur-circle model. K is calibrated so a
  4 mm pupil reproduces the usual clinical rule of thumb for uncorrected myopia (-0.50 D ~ 20/36,
  -1.00 D ~ 20/63, -2.00 D ~ 20/120). See ACCURACY.md.

The patient's-eye view in the video is rendered separately with Fourier (wave) optics: a pupil with the
residual defocus and astigmatism as its wavefront, at 555 nm, with the Stiles-Crawford apodisation.
"""
import numpy as np

K_BLUR = 0.75            # arcmin of MAR per (mm of pupil x dioptre of blur)


# ------------------------------------------------------------------ power vectors
def to_pv(S, C, axis_deg):
    """Spherocylinder (any cylinder sign) -> power vector (M, J0, J45)."""
    a = np.deg2rad(axis_deg)
    M = S + C / 2.0
    J0 = -(C / 2.0) * np.cos(2 * a)
    J45 = -(C / 2.0) * np.sin(2 * a)
    return M, J0, J45


def from_pv(M, J0, J45):
    """Power vector -> minus-cylinder spherocylinder (S, C <= 0, axis in (0, 180])."""
    J = np.hypot(J0, J45)
    C = -2.0 * J
    S = M + J
    axis = np.rad2deg(0.5 * np.arctan2(J45, J0)) % 180.0
    axis = np.where(axis <= 0.0, axis + 180.0, axis)
    return S, C, axis


def pv_dist(a, b):
    """Dioptric distance between two power vectors (tuples of arrays)."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def rx_error(a, b, cyl_weight=2.0):
    """How far prescription a is from b, as one number in dioptres.

    The leftover error is a spherocylinder; this is the length of the vector (spherical-equivalent error,
    leftover cylinder), with the leftover cylinder counted in full dioptres: |C_res| = 2|dJ|. A 0.25 D
    cylinder error counts 0.25 D, and an axis error of d degrees on cylinder C counts 2|C|sin(d).
    cyl_weight=1 gives the plain power-vector distance, which counts cylinder at half weight
    (attempt 1 in the video was trained on that).
    """
    return np.sqrt((a[0] - b[0]) ** 2 + cyl_weight ** 2 * ((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


def jcc_pv(minus_axis_deg, power=0.25):
    """A +/-power Jackson cross cylinder with its minus axis at minus_axis_deg: M = 0, |J| = power."""
    a = np.deg2rad(minus_axis_deg)
    return 0.0, power * np.cos(2 * a), power * np.sin(2 * a)


# ------------------------------------------------------------------ accommodation, blur, acuity
def effective_residual(Mr, amp):
    """Accommodation the patient uses (to clear artificial hyperopia, up to their amplitude) and the M left."""
    acc = np.clip(Mr, 0.0, amp)
    return Mr - acc, acc


def blur_strength(Mr, J0r, J45r, amp):
    Me, _ = effective_residual(Mr, amp)
    return np.sqrt(Me ** 2 + J0r ** 2 + J45r ** 2)


def logmar(B, pupil_mm, mar0):
    """Continuous logMAR for blur strength B (D), pupil diameter (mm), and best-corrected MAR0 (arcmin)."""
    return 0.5 * np.log10(mar0 ** 2 + (K_BLUR * pupil_mm * B) ** 2)


def snellen(lm):
    """logMAR line -> '20/xx' text."""
    den = 20.0 * 10 ** lm
    for std in (10, 12.5, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 400):
        if abs(den - std) / std < 0.06:
            den = std
            break
    return f"20/{den:g}"


# ------------------------------------------------------------------ wave-optics point-spread function
LAMBDA = 555e-9


SCE_RHO = 0.05          # Stiles-Crawford effect: intensity 10^(-rho r^2), r in mm (Applegate & Lakshminarayanan 1993)


def psf(M, J0, J45, pupil_mm, px_arcmin, oversample=2):
    """Monochromatic PSF for a residual (M, J0, J45) in dioptres, in TABO axis notation.

    Returned image is in the patient's view of the world (x right, rows downward), sampled at px_arcmin,
    normalised to sum 1. TABO axes are counter-clockwise as the examiner faces the patient, so from
    behind the lens (the patient's view) an oblique axis is mirrored: J45 changes sign.
    """
    J45 = -J45
    th = np.deg2rad(px_arcmin / oversample / 60.0)            # angular sample of the oversampled PSF
    L = LAMBDA / th                                           # pupil-plane grid extent (m)
    r = pupil_mm * 1e-3 / 2
    pmax = abs(M) + np.hypot(J0, J45)                         # largest principal power
    blur_px = 3.44 * pupil_mm * pmax / (px_arcmin / oversample) + 64
    N = 256
    while N < 2 * blur_px or (pmax * r * (L / N) / LAMBDA) > 0.45:
        N *= 2
        if N >= 2048:
            break
    x = (np.arange(N) - N / 2) * (L / N)
    X, Y = np.meshgrid(x, -x)                                 # Y up
    P = (X ** 2 + Y ** 2) <= r * r
    W = 0.5 * (M * (X ** 2 + Y ** 2) + J0 * (X ** 2 - Y ** 2) + 2 * J45 * X * Y)   # metres
    amp = 10 ** (-SCE_RHO * ((X ** 2 + Y ** 2) * 1e6) / 2)    # amplitude = sqrt of the intensity apodisation
    field = P * amp * np.exp(2j * np.pi * W / LAMBDA)
    h = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))) ** 2
    if oversample > 1:
        n = N // oversample
        h = h[: n * oversample, : n * oversample].reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    h /= h.sum()
    # crop to where the energy is
    c = h.shape[0] // 2
    half = int(min(c, blur_px / oversample + 24))
    h = h[c - half: c + half + 1, c - half: c + half + 1]
    return h / h.sum()


if __name__ == "__main__":
    # sanity checks
    for S, C, ax in [(-2.25, -1.00, 170), (1.5, -0.5, 90), (0.0, -2.0, 45)]:
        pv = to_pv(S, C, ax)
        back = from_pv(*pv)
        print(f"{S:+.2f} {C:+.2f} x{ax:3d} -> M {pv[0]:+.3f} J0 {pv[1]:+.3f} J45 {pv[2]:+.3f} -> "
              f"{back[0]:+.2f} {back[1]:+.2f} x{back[2]:.0f}")
    print("uncorrected myopia, 4 mm pupil, MAR0 = 1':")
    for D in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        lm = logmar(D, 4.0, 1.0)
        print(f"  -{D:.2f} D -> logMAR {lm:.2f} (20/{20 * 10 ** lm:.0f})")
    print("1.00 DC mixed astigmatism (CLC on retina):", f"logMAR {logmar(0.5, 4.0, 1.0):.2f}")
    h = psf(-1.0, 0.0, 0.0, 4.0, 0.25)
    print("PSF 1 D, 4 mm:", h.shape)
