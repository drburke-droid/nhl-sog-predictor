"""What a prescription is worth to the clinic: the AI's only motivation in the clinic exam (env3.py).

The patient judges their new glasses mostly by their worse eye:
    patient error  e = 0.75 * (worse eye's error) + 0.25 * (better eye's error)
where an eye's error is optics.rx_error (spherical-equivalent error and leftover cylinder, in dioptres).

Each outcome has a value in clinic points:
    Happy:     sees clearly, comes back next year and refers a friend           +100
    Annoyed:   "not quite crisp"; may not come back                              +30
    Rx check:  comes back, an unpaid 15-minute re-check (60 points of chair
               time), and doesn't return                                         -80
    Review:    re-check, never returns, leaves a one-star review                 -150
and the chance of each outcome depends on the patient error (OUTCOME_P, interpolated linearly between the
knots). These numbers are a plausible made-up scorecard, not measured data; see ACCURACY.md.

If the exam runs past 15 minutes the patient walks out without glasses: -150.

Chair time: every minute of the exam costs 4 points (what a minute of a 25-minute appointment earns), so
the AI is paid to be quick as well as right. Action durations are in env3.SECS.

The AI is trained on the expected value of the prescription (less noisy than a sampled outcome). The video
samples an actual outcome per patient for the scoreboard.
"""
import numpy as np

OUTCOMES = ("happy", "annoyed", "recheck", "review")
LABELS = {"happy": "Happy: returns and refers", "annoyed": "Annoyed: may not return",
          "recheck": "Rx check: unpaid chair time", "review": "Rx check + bad review"}
POINTS = np.array([100.0, 30.0, -80.0, -150.0])
KNOTS = np.array([0.0, 0.13, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 3.0])
OUTCOME_P = np.array([            # rows: knots; columns: happy, annoyed, recheck, review
    [0.95, 0.05, 0.00, 0.00],
    [0.92, 0.08, 0.00, 0.00],
    [0.75, 0.23, 0.02, 0.00],
    [0.45, 0.45, 0.10, 0.00],
    [0.20, 0.50, 0.28, 0.02],
    [0.05, 0.30, 0.55, 0.10],
    [0.00, 0.10, 0.55, 0.35],
    [0.00, 0.00, 0.25, 0.75],
    [0.00, 0.00, 0.05, 0.95],
])
POINTS_PER_MIN = 4.0
WALKOUT = -150.0                  # the exam ran out of time (15 minutes): no glasses, a bad review
SCALE = 100.0                     # reward = points / SCALE


def patient_error(e_od, e_os):
    e_od, e_os = np.asarray(e_od, float), np.asarray(e_os, float)
    return 0.75 * np.maximum(e_od, e_os) + 0.25 * np.minimum(e_od, e_os)


def outcome_probs(e):
    """(n, 4) probabilities of happy / annoyed / recheck / review for patient error e."""
    e = np.clip(np.asarray(e, float), 0, KNOTS[-1])
    return np.stack([np.interp(e, KNOTS, OUTCOME_P[:, j]) for j in range(4)], -1)


def expected_points(e):
    return outcome_probs(e) @ POINTS


def sample_outcome(e, rng):
    p = outcome_probs(e)
    u = rng.uniform(size=p.shape[:-1])[..., None]
    return (u > np.cumsum(p, -1)).sum(-1)


if __name__ == "__main__":
    for e in (0.0, 0.13, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        p = outcome_probs(np.array([e]))[0]
        print(f"e {e:4.2f}: " + "  ".join(f"{o} {v:.2f}" for o, v in zip(OUTCOMES, p)) +
              f"   expected {expected_points(np.array([e]))[0]:+7.1f} points")
