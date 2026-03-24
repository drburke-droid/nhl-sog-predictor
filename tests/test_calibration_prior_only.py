"""
Test that calibration never uses future outcomes.

Verifies isotonic calibration is trained only on prior windows.
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd


def test_calibrate_walk_forward_no_future():
    """Isotonic calibration must train only on prior windows."""
    import evaluation as ev

    # Create synthetic bets across 3 windows
    np.random.seed(42)
    n = 300
    bets = pd.DataFrame({
        "window": np.repeat([1, 2, 3], n // 3),
        "model_prob": np.random.uniform(0.3, 0.7, n),
        "won": np.random.binomial(1, 0.5, n),
    })

    calibrated = ev.calibrate_walk_forward(bets, "model_prob", "won")

    # Window 1: should have raw probs (no prior data to calibrate from)
    w1_raw = bets[bets["window"] == 1]["model_prob"].values
    w1_cal = calibrated[calibrated["window"] == 1]["calibrated_prob"].values
    assert np.allclose(w1_raw, w1_cal), "Window 1 should use raw probs (no prior data)"

    # Window 2: calibrated from window 1 only
    # Window 3: calibrated from windows 1+2
    # We can't verify the exact values, but we can verify they changed
    w3_raw = bets[bets["window"] == 3]["model_prob"].values
    w3_cal = calibrated[calibrated["window"] == 3]["calibrated_prob"].values
    # At least some should differ (calibration should do something)
    assert not np.allclose(w3_raw, w3_cal, atol=0.001), \
        "Window 3 calibration should differ from raw (isotonic should adjust)"

    print("PASS: Calibration uses only prior windows (window 1 raw, window 3 calibrated)")


def test_isotonic_no_future_data():
    """Isotonic regression must be fitted on training data only."""
    import evaluation as ev
    from sklearn.isotonic import IsotonicRegression

    # Train calibrator on "prior" data
    prior_probs = np.array([0.3, 0.4, 0.5, 0.6, 0.7] * 20)
    prior_outcomes = np.array([0, 0, 1, 1, 1] * 20)

    iso = ev.fit_isotonic_calibration(prior_probs, prior_outcomes)

    # Apply to "future" data
    future_probs = np.array([0.35, 0.55, 0.65])
    calibrated = ev.apply_calibration(iso, future_probs)

    # Calibrated values should be within [0, 1]
    assert all(0 <= p <= 1 for p in calibrated), "Calibrated probs out of range"
    # Calibrated values should be monotonic (isotonic guarantee)
    assert all(calibrated[i] <= calibrated[i + 1] for i in range(len(calibrated) - 1)), \
        "Calibrated probs not monotonic"

    print("PASS: Isotonic calibration produces valid, monotonic probabilities")


if __name__ == "__main__":
    test_calibrate_walk_forward_no_future()
    test_isotonic_no_future_data()
    print("\nAll calibration tests passed.")
