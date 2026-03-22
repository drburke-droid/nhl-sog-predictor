"""
Feature Registry — loads and validates feature definitions for V2 model.

Enforces coverage thresholds, null policies, and fallback hierarchies
from feature_registry.yaml. Prevents silent model degradation from
missing or mostly-null features.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent / "feature_registry.yaml"

_registry = None


def load_registry() -> dict:
    """Load and cache the feature registry from YAML."""
    global _registry
    if _registry is not None:
        return _registry
    with open(_REGISTRY_PATH) as f:
        raw = yaml.safe_load(f)
    _registry = raw.get("features", {})
    return _registry


def get_feature_config(name: str) -> dict:
    """Get config for a single feature."""
    reg = load_registry()
    return reg.get(name, {})


def get_features_by_group(group: str) -> list[str]:
    """Get all feature names belonging to a group."""
    reg = load_registry()
    return [name for name, cfg in reg.items() if cfg.get("group") == group]


def get_all_feature_names() -> list[str]:
    """Get all registered feature names."""
    return list(load_registry().keys())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_coverage(df: pd.DataFrame, features: list[str] = None,
                      strict: bool = True) -> dict:
    """Validate feature coverage against registry thresholds.

    Args:
        df: Training dataframe.
        features: Feature names to check (default: all registered).
        strict: If True, raise on required-tier failures.

    Returns:
        Coverage report dict: {feature: {coverage, threshold, tier, status}}
    """
    reg = load_registry()
    if features is None:
        features = [f for f in reg if f in df.columns]

    report = {}
    failures = []

    for feat in features:
        cfg = reg.get(feat, {})
        tier = cfg.get("tier", "optional")
        threshold = cfg.get("min_coverage", 0.70)

        if feat not in df.columns:
            coverage = 0.0
            status = "MISSING"
        else:
            coverage = float(df[feat].notna().mean())
            if coverage >= threshold:
                status = "OK"
            elif tier == "required":
                status = "FAIL"
                failures.append(feat)
            elif tier == "important":
                status = "WARN"
            else:
                status = "LOW"

        report[feat] = {
            "coverage": round(coverage, 4),
            "threshold": threshold,
            "tier": tier,
            "status": status,
        }

    # Log summary
    ok = sum(1 for r in report.values() if r["status"] == "OK")
    warn = sum(1 for r in report.values() if r["status"] in ("WARN", "LOW"))
    fail = sum(1 for r in report.values() if r["status"] in ("FAIL", "MISSING"))
    logger.info("Feature coverage: %d OK, %d warn, %d fail (of %d features)",
                ok, warn, fail, len(report))

    if failures and strict:
        msg = "Required features below coverage threshold: " + ", ".join(failures)
        for f in failures:
            r = report[f]
            logger.error("  %s: %.1f%% < %.1f%% (%s)",
                         f, r["coverage"] * 100, r["threshold"] * 100, r["tier"])
        raise ValueError(msg)

    return report


# ---------------------------------------------------------------------------
# Null policy enforcement
# ---------------------------------------------------------------------------

def apply_null_policies(df: pd.DataFrame, features: list[str] = None,
                        position_col: str = "position") -> pd.DataFrame:
    """Apply null policies from the registry to fill missing values.

    Does NOT use blanket fillna. Each feature is handled individually
    according to its registered policy.

    Args:
        df: Dataframe with features.
        features: Features to process (default: all registered in df).
        position_col: Column name for position grouping.

    Returns:
        DataFrame with nulls handled per policy. Modifies in place.
    """
    reg = load_registry()
    if features is None:
        features = [f for f in reg if f in df.columns]

    fill_counts = {}

    for feat in features:
        if feat not in df.columns:
            continue

        cfg = reg.get(feat, {})
        policy = cfg.get("null_policy", "fill_league")
        n_null_before = int(df[feat].isna().sum())

        if n_null_before == 0:
            continue

        if policy == "drop_row":
            # Don't drop here — caller decides. Just log.
            fill_counts[feat] = {"policy": "drop_row", "nulls": n_null_before}
            continue

        elif policy == "fill_default":
            default = cfg.get("default_value", 0)
            df[feat] = df[feat].fillna(default)

        elif policy == "fill_zero":
            df[feat] = df[feat].fillna(0)

        elif policy == "fill_position":
            if position_col in df.columns:
                pos_means = df.groupby(position_col)[feat].transform("mean")
                df[feat] = df[feat].fillna(pos_means)
            # If still null, fill with global mean
            df[feat] = df[feat].fillna(df[feat].mean())

        elif policy == "fill_cluster":
            if "cluster_id" in df.columns:
                cluster_means = df.groupby("cluster_id")[feat].transform("mean")
                df[feat] = df[feat].fillna(cluster_means)
            df[feat] = df[feat].fillna(df[feat].mean())

        elif policy == "fill_league":
            df[feat] = df[feat].fillna(df[feat].mean())

        elif policy == "fill_team":
            if "team" in df.columns:
                team_means = df.groupby("team")[feat].transform("mean")
                df[feat] = df[feat].fillna(team_means)
            df[feat] = df[feat].fillna(df[feat].mean())

        n_filled = n_null_before - int(df[feat].isna().sum())
        if n_filled > 0:
            fill_counts[feat] = {
                "policy": policy,
                "nulls": n_null_before,
                "filled": n_filled,
                "remaining": n_null_before - n_filled,
            }

    if fill_counts:
        total_filled = sum(v.get("filled", 0) for v in fill_counts.values())
        logger.info("Null policies applied: %d values filled across %d features",
                     total_filled, len(fill_counts))

    return df


# ---------------------------------------------------------------------------
# Coverage report generation
# ---------------------------------------------------------------------------

def generate_coverage_report(df: pd.DataFrame, features: list[str] = None) -> str:
    """Generate a markdown coverage report for logging/saving."""
    report = validate_coverage(df, features, strict=False)

    lines = [
        "# Feature Coverage Report",
        "",
        f"Total rows: {len(df)}",
        "",
        "| Feature | Group | Tier | Coverage | Threshold | Status |",
        "|---------|-------|------|----------|-----------|--------|",
    ]

    reg = load_registry()
    for feat, r in sorted(report.items(), key=lambda x: x[1]["status"]):
        group = reg.get(feat, {}).get("group", "?")
        lines.append(
            f"| {feat} | {group} | {r['tier']} | "
            f"{r['coverage']:.1%} | {r['threshold']:.0%} | {r['status']} |"
        )

    return "\n".join(lines)
