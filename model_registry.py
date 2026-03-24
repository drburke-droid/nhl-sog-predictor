"""
Model and Experiment Registry — Sprint 3.

Tracks model versions, experiments, and promotion decisions
for reproducibility and governance.
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent / "registry"


def _ensure_registry():
    REGISTRY_DIR.mkdir(exist_ok=True)


def _load_registry(name):
    _ensure_registry()
    path = REGISTRY_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def _save_registry(name, data):
    _ensure_registry()
    path = REGISTRY_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def register_model(model_name, sport, market, track_type, feature_version,
                   hyperparameters, calibration_version, distribution_version,
                   training_window, status="research"):
    """Register a model version."""
    registry = _load_registry("models")

    hp_hash = hashlib.md5(json.dumps(hyperparameters, sort_keys=True).encode()).hexdigest()[:8]

    entry = {
        "model_name": model_name,
        "sport": sport,
        "market": market,
        "track_type": track_type,
        "feature_version": feature_version,
        "hyperparameter_hash": hp_hash,
        "calibration_version": calibration_version,
        "distribution_version": distribution_version,
        "training_window": training_window,
        "status": status,
        "created_at": datetime.now().isoformat(),
    }

    registry.append(entry)
    _save_registry("models", registry)
    logger.info("Registered model: %s (%s/%s) status=%s", model_name, sport, market, status)
    return entry


def update_model_status(model_name, new_status):
    """Update a model's status (research -> candidate -> production -> retired)."""
    registry = _load_registry("models")
    for entry in reversed(registry):
        if entry["model_name"] == model_name:
            entry["status"] = new_status
            entry["status_updated_at"] = datetime.now().isoformat()
            _save_registry("models", registry)
            logger.info("Updated %s status to %s", model_name, new_status)
            return entry
    return None


def list_models(status=None):
    """List registered models, optionally filtered by status."""
    registry = _load_registry("models")
    if status:
        registry = [e for e in registry if e.get("status") == status]
    return registry


# ---------------------------------------------------------------------------
# Experiment Registry
# ---------------------------------------------------------------------------

def log_experiment(hypothesis, change, date_range, metrics, decision, notes=""):
    """Log an experiment to the registry."""
    registry = _load_registry("experiments")

    entry = {
        "id": len(registry) + 1,
        "hypothesis": hypothesis,
        "change": change,
        "date_range": date_range,
        "metrics": metrics,
        "decision": decision,
        "notes": notes,
        "logged_at": datetime.now().isoformat(),
    }

    registry.append(entry)
    _save_registry("experiments", registry)
    logger.info("Logged experiment #%d: %s -> %s", entry["id"], hypothesis, decision)
    return entry


def list_experiments():
    """List all logged experiments."""
    return _load_registry("experiments")


# ---------------------------------------------------------------------------
# Promotion Gates
# ---------------------------------------------------------------------------

def check_preproduction_eligibility(model_name, metrics):
    """Check whether a model passes promotion gates.

    Args:
        metrics: dict with keys from evaluation reports

    Returns dict with gate results and overall eligibility.
    """
    gates = {
        "permutation_significant": metrics.get("permutation_p", 1.0) < 0.10,
        "locked_forward_positive": metrics.get("locked_forward_yield", 0) > 0,
        "multiple_cutoffs_acceptable": metrics.get("worst_cutoff_p_positive", 0) >= 0.50,
        "edge_monotonic": metrics.get("edge_monotonicity_score", 0) >= 0.5,
        "ablation_supports_complexity": metrics.get("ablation_harmful_groups", 1) == 0,
        "side_calibration_acceptable": metrics.get("max_side_brier_gap", 1) < 0.05,
        "exposure_controls_active": metrics.get("exposure_controls", False),
        "reproducibility_snapshots": metrics.get("snapshots_active", False),
    }

    passed = sum(1 for v in gates.values() if v)
    total = len(gates)
    eligible = passed >= total - 2  # Allow up to 2 failures for pre-production

    return {
        "model_name": model_name,
        "gates": gates,
        "passed": passed,
        "total": total,
        "eligible": eligible,
        "checked_at": datetime.now().isoformat(),
    }


def print_promotion_report(result):
    """Print formatted promotion gate report."""
    print(f"\n{'=' * 60}")
    print(f"  PROMOTION GATE REPORT: {result['model_name']}")
    print(f"{'=' * 60}")

    for gate, passed in result["gates"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status:4s}] {gate}")

    print(f"\n  Result: {result['passed']}/{result['total']} gates passed")
    verdict = "ELIGIBLE" if result["eligible"] else "NOT ELIGIBLE"
    print(f"  Verdict: {verdict}")
    print(f"{'=' * 60}")
