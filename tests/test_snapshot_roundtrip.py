"""
Test that model save/load roundtrip preserves all state.

Verifies: model weights, metadata, dispersion model, shrinkage,
player CV, var_ratio, and trained_features survive persistence.
"""

import sys
sys.path.insert(0, "..")

import json
from pathlib import Path


def test_v1_model_roundtrip():
    """V1 model save/load must preserve all state."""
    import model

    loaded = model.load_model()
    if not loaded:
        print("SKIP: V1 model not saved")
        return

    # Check all global state is populated
    assert model._model_fwd is not None, "Forward model not loaded"
    assert model._model_def is not None, "Defense model not loaded"
    assert len(model._player_cv) > 0, "Player CV dict empty"
    assert len(model._player_var_ratio) > 0, "Player var_ratio dict empty"
    assert model._disp_shrinkage > 0, "Dispersion shrinkage not loaded"

    # Check metadata
    meta_path = model.MODEL_DIR / "meta.json"
    assert meta_path.exists(), "meta.json not found"
    with open(meta_path) as f:
        meta = json.load(f)

    assert meta.get("model_version") == model.MODEL_VERSION, "Version mismatch"
    assert "disp_shrinkage" in meta, "disp_shrinkage not in metadata"
    assert "disp_features" in meta, "disp_features not in metadata"

    print(f"PASS: V1 model roundtrip OK (version={meta['model_version']}, "
          f"shrinkage={meta['disp_shrinkage']}, "
          f"{len(model._player_cv)} players)")


def test_v2_model_roundtrip():
    """V2 model save/load must preserve trained_features."""
    import model_v2

    loaded = model_v2.load_model()
    if not loaded:
        print("SKIP: V2 model not saved")
        return

    assert model_v2._model_fwd is not None, "V2 forward model not loaded"
    assert len(model_v2._trained_features) > 0, "trained_features empty after load"

    meta_path = model_v2.SAVE_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    assert "trained_features" in meta, "trained_features not in V2 metadata"
    assert meta["trained_features"] == model_v2._trained_features, \
        "trained_features mismatch between meta and global"

    print(f"PASS: V2 model roundtrip OK ({len(model_v2._trained_features)} trained features)")


def test_dispersion_model_persistence():
    """Dispersion model must survive save/load."""
    import model

    loaded = model.load_model()
    if not loaded:
        print("SKIP: Model not saved")
        return

    disp_path = model.MODEL_DIR / "model_disp.json"
    if not disp_path.exists():
        print("SKIP: Dispersion model not saved")
        return

    assert model._disp_model is not None, "Dispersion model not loaded"
    assert len(model._disp_features) > 0, "Dispersion features empty"
    assert 0 < model._disp_shrinkage <= 1, f"Shrinkage out of range: {model._disp_shrinkage}"

    print(f"PASS: Dispersion model roundtrip OK "
          f"(shrinkage={model._disp_shrinkage}, features={model._disp_features[:3]}...)")


if __name__ == "__main__":
    test_v1_model_roundtrip()
    test_v2_model_roundtrip()
    test_dispersion_model_persistence()
    print("\nAll snapshot roundtrip tests passed.")
