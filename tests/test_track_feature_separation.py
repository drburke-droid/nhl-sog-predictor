"""
Test that Track A, B, C feature separation is correct.

Track A: No market-derived features
Track B: Only market features
Track C: All features
"""

import sys
sys.path.insert(0, "..")


MARKET_FEATURES = {
    "game_total", "implied_team_total", "sog_prop_line", "sharp_consensus_prob"
}


def test_track_a_no_market_features():
    """Track A must exclude all market-derived features."""
    import model as nhl_model
    all_features = set(nhl_model.FEATURE_COLS)
    track_a = all_features - MARKET_FEATURES

    for f in track_a:
        assert f not in MARKET_FEATURES, f"Track A contains market feature: {f}"

    print(f"PASS: Track A has {len(track_a)} features, no market features")


def test_track_b_only_market_features():
    """Track B must only use market-derived features."""
    import model as nhl_model
    all_features = set(nhl_model.FEATURE_COLS)
    track_b = all_features & MARKET_FEATURES

    for f in track_b:
        assert f in MARKET_FEATURES, f"Track B contains non-market feature: {f}"

    print(f"PASS: Track B has {len(track_b)} features, all market-derived")


def test_track_c_is_superset():
    """Track C must be the full feature set (A union B)."""
    import model as nhl_model
    all_features = set(nhl_model.FEATURE_COLS)
    track_a = all_features - MARKET_FEATURES
    track_b = all_features & MARKET_FEATURES

    assert track_a | track_b == all_features, "Track A + B does not equal Track C"
    print(f"PASS: Track C = Track A ({len(track_a)}) + Track B ({len(track_b)}) = {len(all_features)}")


def test_no_overlap():
    """Track A and B must not share features."""
    import model as nhl_model
    all_features = set(nhl_model.FEATURE_COLS)
    track_a = all_features - MARKET_FEATURES
    track_b = all_features & MARKET_FEATURES

    overlap = track_a & track_b
    assert len(overlap) == 0, f"Tracks A and B overlap on: {overlap}"
    print("PASS: No feature overlap between Track A and Track B")


if __name__ == "__main__":
    test_track_a_no_market_features()
    test_track_b_only_market_features()
    test_track_c_is_superset()
    test_no_overlap()
    print("\nAll track separation tests passed.")
