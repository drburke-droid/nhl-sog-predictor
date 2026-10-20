"""Every number in the golf-ball film, in one place.

    python physics.py          # prints the tables quoted in ACCURACY.md

Angles are in arcminutes (′) unless named otherwise. Distances are in yards (1 yd = 0.9144 m).

* The ball: 1.680 in (42.67 mm), the smallest diameter the Rules of Golf allow.
* The cones: 0.5′ apart at the very centre of the fovea (Curcio et al. 1990), as in the vernier film.
* The eye's blur: Westheimer's line spread function for a 3 mm pupil, the same one the vernier film uses.
  Its point spread function (PSF) follows exactly: a Gaussian term and a K0 (modified Bessel) term, the
  2D functions whose projections are the LSF's Gaussian and exponential terms.
* Detection of a target smaller than the eye's summation area (Ricco's law): threshold contrast × area is
  constant. The constant, and how it depends on the background luminance, is Crumey's photopic fit to
  Blackwell's data (Crumey 2014, MNRAS 442:2600, eqs. 24, 35-44), times a field factor F.
* The spatial frequency window: the "20/20 sample" contrast sensitivity curve of the Spatial Frequency
  Analyzer that goes with the blog post (log-parabola, peak 126 at 4.5 c/deg).
"""
import math
import numpy as np
from scipy.special import k0, j1

YD = 0.9144
BALL_MM = 42.67
BALL_R_M = BALL_MM / 2000
RAD_ARCMIN = 180 * 60 / math.pi                  # arcmin per radian (3437.75)
SR_ARCMIN2 = RAD_ARCMIN ** 2                     # arcmin² per steradian
CONE = 0.5                                       # foveal cone spacing, arcmin
UM_PER_ARCMIN = 4.85                             # on the retina (about 291 µm per degree)


# ------------------------------------------------------------------ geometry
def ball_arcmin(yd):
    """the ball's angular diameter at a distance (yards)"""
    return 2 * np.arctan(BALL_R_M / (np.asarray(yd, np.float64) * YD)) * RAD_ARCMIN


def yards_for(arcmin):
    """the distance (yards) at which the ball spans this angle"""
    return BALL_R_M / np.tan(np.asarray(arcmin, np.float64) / RAD_ARCMIN / 2) / YD


def ball_area(yd):
    """the ball's solid angle, arcmin²"""
    return math.pi * (ball_arcmin(yd) / 2) ** 2


# ------------------------------------------------------------------ the eye's blur (Westheimer, 3 mm pupil)
LSF_G, LSF_A, LSF_E, LSF_B = 0.47, 3.3, 0.53, 0.93            # L(x) = 0.47 e^(-3.3x²) + 0.53 e^(-0.93|x|)
_LSF_SUM = LSF_G * math.sqrt(math.pi / LSF_A) + LSF_E * 2 / LSF_B
PSF_G = LSF_G / _LSF_SUM / math.sqrt(math.pi / LSF_A)         # Gaussian term's height
PSF_K = LSF_E / _LSF_SUM * LSF_B / math.pi                    # K0 term's coefficient


def lsf(x):
    return LSF_G * np.exp(-LSF_A * x * x) + LSF_E * np.exp(-LSF_B * np.abs(x))


def psf(r):
    """the eye's point spread function (per arcmin², unit volume), r in arcmin. The K0 term is
    logarithmically infinite at r = 0; integrate it over an area rather than sampling it there."""
    r = np.maximum(np.asarray(r, np.float64), 1e-6)
    return PSF_G * np.exp(-LSF_A * r * r) + PSF_K * k0(LSF_B * r)


def otf(rho):
    """the eye's optical transfer function, rho in cycles per arcmin (the 1D Fourier transform of the LSF,
    normalised)"""
    rho = np.asarray(rho, np.float64)
    g = LSF_G * math.sqrt(math.pi / LSF_A) / _LSF_SUM
    return g * np.exp(-math.pi ** 2 * rho ** 2 / LSF_A) + (1 - g) * LSF_B ** 2 / (LSF_B ** 2 + 4 * math.pi ** 2 * rho ** 2)


def jinc(x):
    """2 J1(πx) / (πx): the Fourier transform of a disk of unit diameter, normalised to 1 at 0"""
    x = np.asarray(x, np.float64)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-9
    out[m] = 2 * j1(math.pi * x[m]) / (math.pi * x[m])
    return out


_RHO = np.linspace(0, 12, 24001)                               # cycles per arcmin (to 720 c/deg)


def disk_image(r, diam, contrast=1.0, aperture=0.0):
    """the retinal image of a uniform disk (angular diameter diam, Weber contrast) on a uniform background:
    the change in light at radius r (arcmin), as a fraction of the background. aperture: optionally also
    averaged over a cone's aperture (a disk of this diameter). Computed as a Hankel transform."""
    r = np.atleast_1d(np.asarray(r, np.float64))
    from scipy.special import j0
    spec = contrast * math.pi * diam ** 2 / 4 * jinc(diam * _RHO) * otf(_RHO)
    if aperture > 0:
        spec = spec * jinc(aperture * _RHO)
    w = 2 * math.pi * _RHO * (_RHO[1] - _RHO[0])
    w[0] *= 0.5; w[-1] *= 0.5
    return (j0(2 * math.pi * np.outer(r, _RHO)) * (spec * w)[None, :]).sum(1)


def fwhm(diam, aperture=0.0):
    """full width at half maximum of the disk's retinal image (arcmin)"""
    r = np.linspace(0, 4, 4001)
    p = disk_image(r, diam, aperture=aperture)
    k = int(np.argmax(p < p[0] / 2))
    return 2 * float(np.interp(p[0] / 2, [p[k], p[k - 1]], [r[k], r[k - 1]]))


def encircled(radius):
    """share of a point's light that lands within this radius (arcmin)"""
    from scipy.integrate import quad
    return quad(lambda r: psf(r) * 2 * math.pi * r, 0, radius, limit=200)[0]


# ------------------------------------------------------------------ the contrast sensitivity curve (the analyzer's)
CSF_G, CSF_F, CSF_B, CSF_D = 2.10, 4.5, 1.10, 0.80


def csf(cpd):
    """sensitivity (1 / threshold contrast) at spatial frequency cpd (cycles per degree)"""
    d = np.log10(np.maximum(np.asarray(cpd, np.float64), 0.05)) - math.log10(CSF_F)
    return 10 ** (CSF_G - CSF_B * d * d - np.where(d > 0, CSF_D * d ** 4, 0.0))


def csf_cutoff():
    from scipy.optimize import brentq
    return brentq(lambda f: math.log10(csf(f)), CSF_F, 200)


# ------------------------------------------------------------------ detection (Ricco's law with Weber's law)
def ricco_R(B):
    """Crumey's photopic Ricco constant: threshold Weber contrast × area for a small target, in arcmin²,
    at background luminance B (cd/m²). From Blackwell's 50% forced-choice data."""
    return (1.772e-4 * B ** -0.25 + 7.167e-5) ** 2 * SR_ARCMIN2


C_INF = 2.720e-3                                               # large-target threshold contrast (daylight)


def threshold(area, B, F=2.0):
    """threshold Weber contrast for a target of this area (arcmin²) on background B: Crumey eq. 41, times F"""
    q = 1.146 - 0.0885 * math.log10(B)
    return F * ((ricco_R(B) / area) ** q + C_INF ** q) ** (1 / q)


def max_yards(contrast, B, F=2.0):
    """the farthest the ball is seen (yards): where |contrast| meets the threshold for its size"""
    from scipy.optimize import brentq
    f = lambda yd: math.log(abs(contrast) / threshold(ball_area(yd), B, F))
    return brentq(f, 5, 1e6)


HORIZON = 8000.0                                               # horizon sky luminance on a clear day, cd/m²


def max_yards_haze(ball, bg, sky_bg=False, vis_km=40.0, F=2.0):
    """as max_yards, with the air's own light (Koschmieder): over a path of transmission T, the air adds
    HORIZON·(1 - T) to the ball and, unless the background is the sky itself, to the background too.
    vis_km: the meteorological visibility (T = 2% at that distance)."""
    from scipy.optimize import brentq

    def f(yd):
        T = math.exp(-3.912 * yd * YD / (vis_km * 1000))
        b = ball * T + HORIZON * (1 - T)
        B = bg if sky_bg else bg * T + HORIZON * (1 - T)
        return math.log(max(abs(weber(b, B)), 1e-12) / threshold(ball_area(yd), B, F))
    return brentq(f, 5, 5000)


# ------------------------------------------------------------------ the scenes
# luminances (cd/m²): examples for a clear, sunny day (ACCURACY.md §4). The ball is a white ball lit by the sun.
BALL_SUN = 15000.0
SCENES = [  # name, background luminance, the ball's luminance against it
    ("dark trees", 500.0, BALL_SUN),
    ("the fairway", 2500.0, BALL_SUN),
    ("blue sky", 6000.0, BALL_SUN),
    ("overcast sky", 4000.0, 2200.0),
]


def weber(ball, bg):
    return (ball - bg) / bg


if __name__ == "__main__":
    print("ball angle:")
    for yd in (50, 100, 160, 166, 200, 250, 300, 320, 400, 500, 640, 1000):
        print(f"  {yd:5d} yd  {ball_arcmin(yd):6.3f}′  {ball_arcmin(yd) * 60:6.1f}″  "
              f"{ball_arcmin(yd) / CONE:5.2f} cones  {ball_arcmin(yd) * UM_PER_ARCMIN:5.2f} µm")
    for a in (1.0, 0.5):
        print(f"  {a}′ at {yards_for(a):.1f} yd ({yards_for(a) * YD:.1f} m)")
    print("166.5 yd ->", ball_arcmin(166.5))
    print("\nblur:")
    xs = np.linspace(0, 2, 20001)
    L = lsf(xs)
    print(f"  LSF FWHM {2 * xs[np.argmax(L < 0.5)]:.3f}′")
    from scipy.integrate import quad
    print(f"  PSF volume {quad(lambda r: psf(r) * 2 * math.pi * r, 0, 60, limit=400)[0]:.4f}")
    for rr in (0.25, 0.5, 1.0, 2.0, 5.0):
        print(f"  encircled within r = {rr}′: {encircled(rr):.3f}")
    for d in (2.0, 1.5, 1.0, 0.5, 0.25, 0.1, 0.01):
        print(f"  disk {d:4.2f}′: image FWHM {fwhm(d):.3f}′, with a 0.5′ cone aperture {fwhm(d, 0.5):.3f}′, "
              f"peak (contrast 1) {disk_image(0, d)[0]:.4f}")
    print(f"  a point, through one cone's aperture: field FWHM {fwhm(0.001, 0.5):.3f}′")
    print("\nCSF: cutoff", round(csf_cutoff(), 1), "c/deg; peak", round(float(csf(CSF_F)), 1))
    print("\nRicco constant (F = 1):")
    for B in (300, 500, 1000, 2500, 4000, 6000, 10000):
        print(f"  B = {B:6.0f} cd/m²: R = {ricco_R(B):.4f} arcmin²")
    print("\nscenes (F = 2):")
    for name, bg, ball in SCENES:
        c = weber(ball, bg)
        print(f"  {name:13s} bg {bg:6.0f}  ball {ball:6.0f}  C = {c:+6.2f}  seen to {max_yards(c, bg):6.0f} yd"
              f"  (F=1: {max_yards(c, bg, 1.0):6.0f}, F=4: {max_yards(c, bg, 4.0):6.0f})")
    print("  a black ball (C = -1) on a 6000 cd/m² sky:", round(max_yards(-1, 6000.0)), "yd")
    print("\nwith the air's light (visibility 40 km / 20 km):")
    for name, bg, ball in SCENES:
        sky = "sky" in name
        print(f"  {name:13s} {max_yards_haze(ball, bg, sky):6.0f} / {max_yards_haze(ball, bg, sky, 20.0):6.0f} yd")
    print("\nthe rule of thumb: F·R against one cone's area (0.2165 arcmin²)")
    for B in (500, 2500, 4000, 6000):
        print(f"  B = {B}: F·R = {2 * ricco_R(B):.3f} arcmin²")
    print("\nthe ball's image peak (per unit contrast, through a cone's aperture):")
    for yd in (100, 160, 320, 400, 640, 1280):
        print(f"  {yd:5d} yd: {disk_image(0, float(ball_arcmin(yd)), aperture=0.5)[0]:.4f}")
