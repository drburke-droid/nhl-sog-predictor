"""A simulated clinic population: true prescriptions, eyes, response styles, and an autorefractor starting point.

All draws are vectorised: sample_patients(rng, n) returns a dict of length-n arrays.
The distributions are plausible for a general optometry clinic, not fitted to any dataset (see ACCURACY.md).
"""
import numpy as np
from optics import to_pv, from_pv

Q = 0.25   # phoropter sphere/cylinder step


def q(x, step=Q):
    return np.round(np.asarray(x) / step) * step


def wrap_axis(ax):
    """Integer axis in 1..180."""
    ax = np.round(ax) % 180
    return np.where(ax == 0, 180, ax).astype(np.float64)


def sample_patients(rng, n, age=None, start="autoref", hard=0.0):
    """start='autoref': the phoropter starts at a simulated autorefractor reading; 'plano': at 0.00 (no
    autorefractor); 'mix': half and half. hard (0..1, training only): bigger cylinders and worse autorefractor
    axes, for early practice."""
    age = rng.uniform(8, 80, n) if age is None else np.full(n, float(age))

    # --- true refractive error (the full distance correction, accommodation relaxed)
    M = rng.normal(-1.0, 2.0, n)
    old = age > 40
    M = np.where(old, rng.normal(-0.5, 2.0, n) + 0.02 * (age - 40), M)
    M = np.clip(M, -8.0, 6.0)
    cyl = np.minimum(rng.gamma(1.2, 0.5 + 0.7 * hard, n), 3.5)
    r = rng.uniform(size=n)
    wtr_p = np.where(age > 50, 0.30, 0.60)                  # with-the-rule falls with age, against-the-rule rises
    atr_p = np.where(age > 50, 0.50, 0.20)
    axis = np.where(r < wtr_p, rng.normal(180, 12, n),
                    np.where(r < wtr_p + atr_p, rng.normal(90, 12, n), rng.uniform(0, 180, n)))
    C = -q(cyl)
    S = q(M - C / 2)
    axis = wrap_axis(axis)

    # --- the eye and how this person answers
    lm0 = rng.normal(-0.08 + 0.004 * np.maximum(age - 40, 0), 0.06, n)
    reduced = rng.uniform(size=n) < 0.08                    # cataract, amblyopia, macular change...
    lm0 = np.where(reduced, rng.uniform(0.2, 0.5, n), np.clip(lm0, -0.2, 0.3))
    mar0 = 10 ** lm0
    pupil = np.clip(4.8 - 0.025 * (age - 20) + rng.normal(0, 0.4, n), 2.5, 7.0)
    amp = np.clip(18.5 - 0.30 * age + rng.normal(0, 1.0, n), 0.0, 16.0)     # Hofstetter's average amplitude
    sigma = rng.uniform(0.015, 0.05, n)                      # response noise (logMAR units)
    tau = rng.uniform(0.02, 0.05, n)                         # "they look the same" band
    bias = rng.uniform(0.0, 0.20, n)                         # minus looks "darker, crisper": logMAR per D
    lapse = rng.uniform(0.0, 0.05, n)                        # random answers

    # --- autorefractor reading (the phoropter's starting point)
    young = age < 40
    mu = np.where(young, -0.25, np.where(age < 55, -0.10, 0.0))   # instrument myopia: over-minus in the young
    bad = rng.uniform(size=n) < 0.12
    sM = np.where(bad, 0.9, 0.30)
    sJ = np.where(bad, 0.30, 0.10) * (1 + 2 * hard)
    EM, EJ0, EJ45 = to_pv(S, C, axis)
    S0, C0, ax0 = from_pv(EM + rng.normal(mu, sM), EJ0 + rng.normal(0, sJ), EJ45 + rng.normal(0, sJ))
    C0 = q(C0)
    S0 = q(S0)
    ax0 = wrap_axis(ax0)
    has = np.ones(n)
    if start in ("plano", "mix"):                            # same patients, same random draws; no reading
        none = np.ones(n, bool) if start == "plano" else rng.uniform(size=n) < 0.5
        S0, C0, ax0 = np.where(none, 0.0, S0), np.where(none, 0.0, C0), np.where(none, 180.0, ax0)
        has = (~none).astype(float)
    return dict(has_reading=has, age=age, S=S, C=C, axis=axis, mar0=mar0, pupil=pupil, amp=amp, sigma=sigma, tau=tau,
                bias=bias, lapse=lapse, S0=S0, C0=C0, ax0=ax0)


def fixed_patient(**kw):
    """A patient sampler that always returns the given patient (for the video's showcase exams)."""
    def fn(rng, n):
        base = sample_patients(rng, n, age=kw.get("age"), start=kw.get("start", "autoref"))
        for k, v in kw.items():
            if k != "start":
                base[k] = np.full(n, float(v))
        return base
    return fn


class TestSet:
    """A fixed, numbered set of patients served in order (for fair comparisons between examiners)."""

    def __init__(self, size, seed=12345, start="autoref"):
        self.size = size
        self.start = start
        self.pool = sample_patients(np.random.default_rng(seed), size, start=start)
        self.pool["pid"] = np.arange(size, dtype=np.float64)
        self.next = 0

    def sampler(self, rng, n):
        take = np.arange(self.next, self.next + n)
        self.next += n
        out = sample_patients(rng, n, start=self.start)
        out["pid"] = np.full(n, -1.0)
        real = take < self.size
        for k, v in self.pool.items():
            out[k][real] = v[take[real]]
        return out


# ---------------------------------------------------------------------------- two eyes (attempt 4)
def sample_pair(rng, n, hard=0.0, age=None):
    """Two eyes per patient. The right eye is drawn exactly as sample_patients; the left eye is correlated:
    spherical equivalent differs by N(0, 0.25) D (5% of patients: N(0, 1.0) D, anisometropia), cylinder by
    N(0, 0.25) D, and the axis is roughly the mirror image (180 - axis, +/- 8 deg). Best-corrected acuity is
    nearly the same in both eyes, except 5% with one weaker eye (+0.2..0.4 logMAR). Each eye gets its own
    autorefractor error; the instrument-myopia bias is shared. Pupil, accommodation and answering style are
    the person's, so shared."""
    p = sample_patients(rng, n, age=age, hard=hard)
    age = p["age"]
    M_od = p["S"] + p["C"] / 2
    aniso = np.where(rng.uniform(size=n) < 0.05, rng.normal(0, 1.0, n), rng.normal(0, 0.25, n))
    cyl = np.clip(-p["C"] + rng.normal(0, 0.25, n), 0, 3.5)
    C = -q(cyl)
    S = q(np.clip(M_od + aniso, -9, 7) - C / 2)
    axis = wrap_axis(180 - p["axis"] + rng.normal(0, 8, n))
    weak = rng.uniform(size=n) < 0.05
    mar0 = p["mar0"] * 10 ** rng.normal(0, 0.03, n)
    mar0 = np.where(weak, mar0 * 10 ** rng.uniform(0.2, 0.4, n), mar0)
    young = age < 40
    mu = np.where(young, -0.25, np.where(age < 55, -0.10, 0.0))
    # the right eye's autorefractor error relative to its truth, so the shared bias is the same draw
    E_od = to_pv(p["S"], p["C"], p["axis"])
    R_od = to_pv(p["S0"], p["C0"], p["ax0"])
    shared = R_od[0] - E_od[0]                                # the right eye's SE offset carries the person's bias
    bad = rng.uniform(size=n) < 0.12
    sM = np.where(bad, 0.9, 0.30)
    sJ = np.where(bad, 0.30, 0.10) * (1 + 2 * hard)
    E = to_pv(S, C, axis)
    dM = 0.5 * (shared - mu) + mu + rng.normal(0, sM * 0.85, n)   # correlated with the right eye's reading
    S0, C0, ax0 = from_pv(E[0] + dM, E[1] + rng.normal(0, sJ), E[2] + rng.normal(0, sJ))
    # proximal ("instrument") accommodation behind the phoropter with one eye occluded: a person-level extra
    # effort that fogging relaxes, and that mostly relaxes when both eyes are open (env2.py uses 30% of it)
    prox = rng.uniform(0, 0.5, n) * np.clip(p["amp"] / 4, 0, 1)
    dom = np.where(rng.uniform(size=n) < 0.7, 1.0, -1.0)      # +1 right-eye dominant (about 70% of people)
    p.update(S_os=S, C_os=C, axis_os=axis, mar0_os=mar0, S0_os=q(S0), C0_os=q(C0), ax0_os=wrap_axis(ax0), prox=prox,
             dom=dom)
    return p


class PairTestSet(TestSet):
    """A fixed, numbered set of two-eyed patients."""

    def __init__(self, size, seed=12345):
        self.size = size
        self.start = "autoref"
        self.pool = sample_pair(np.random.default_rng(seed), size)
        self.pool["pid"] = np.arange(size, dtype=np.float64)
        self.next = 0

    def sampler(self, rng, n):
        take = np.arange(self.next, self.next + n)
        self.next += n
        out = sample_pair(rng, n)
        out["pid"] = np.full(n, -1.0)
        real = take < self.size
        for k, v in self.pool.items():
            out[k][real] = v[take[real]]
        return out
