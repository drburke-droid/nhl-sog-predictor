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
