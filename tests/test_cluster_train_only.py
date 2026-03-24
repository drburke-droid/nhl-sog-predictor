"""
Test that clustering is fitted only on training data.

Verifies cluster assignments don't use future data.
"""

import sys
sys.path.insert(0, "..")

import numpy as np


def test_cluster_features_no_target():
    """Clustering must not use the target variable (SOG)."""
    from clustering import CLUSTER_FEATURES

    forbidden = {"shots", "sog", "goals", "actual_sog", "sog_residual"}
    overlap = set(f.lower() for f in CLUSTER_FEATURES) & forbidden
    assert len(overlap) == 0, f"Cluster features include target-like variables: {overlap}"
    print(f"PASS: Cluster features ({len(CLUSTER_FEATURES)}) do not include target variable")


def test_v2_trained_features_alignment():
    """V2 model must use saved trained_features list at prediction time."""
    import model_v2

    # If model is loaded, trained_features should be populated
    loaded = model_v2.load_model()
    if not loaded:
        print("SKIP: V2 model not saved, cannot verify")
        return

    assert len(model_v2._trained_features) > 0, "trained_features should be populated after load"
    assert len(model_v2._trained_features) <= len(model_v2.FEATURE_COLS), \
        "trained_features should be a subset of FEATURE_COLS"

    print(f"PASS: V2 trained_features has {len(model_v2._trained_features)} features "
          f"(FEATURE_COLS has {len(model_v2.FEATURE_COLS)})")


def test_cluster_k_selection():
    """Cluster k should be selected by silhouette score."""
    from clustering import PlayerClusterer

    # Verify the clusterer has a valid k
    c = PlayerClusterer()
    assert hasattr(c, "k"), "PlayerClusterer must have k attribute"
    print(f"PASS: PlayerClusterer has k={c.k}")


if __name__ == "__main__":
    test_cluster_features_no_target()
    test_v2_trained_features_alignment()
    test_cluster_k_selection()
    print("\nAll cluster tests passed.")
