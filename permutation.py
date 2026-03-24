"""
Permutation Significance Testing — Sprint 3.

Shuffles bet outcomes within local time windows to test whether
observed ROI is significantly above chance. Preserves seasonal
structure, line distribution, and bet selection — only breaks
the prediction-outcome relationship.

If the real ROI is not meaningfully above the permutation
distribution, the edge may be noise or structural artifact.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def permute_outcomes_within_windows(bets_df, window_days=14, rng=None):
    """Shuffle the 'won' column within local time windows.

    Preserves:
    - approximate seasonal win rate per window
    - bet count and structure per window
    - odds, edges, model_prob (untouched)

    Only breaks: which specific bets won/lost.
    """
    df = bets_df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    min_date = df["_date"].min()

    # Assign window IDs
    df["_window"] = ((df["_date"] - min_date).dt.days // window_days).astype(int)

    # Shuffle within each window
    for _, grp in df.groupby("_window"):
        idx = grp.index
        shuffled = rng.permutation(df.loc[idx, "won"].values)
        df.loc[idx, "won"] = shuffled

    df.drop(columns=["_date", "_window"], inplace=True)
    return df


def compute_flat_roi(bets_df):
    """Compute flat-stake ROI from bets DataFrame.

    ROI = mean profit per unit wagered (flat $1 per bet).
    """
    dec_odds = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    profits = np.where(won, dec_odds - 1, -1)
    return float(np.mean(profits))


def run_permutation_test(bets_df, n_perms=500, window_days=14, seed=42):
    """Run permutation significance test on a filtered set of bets.

    Args:
        bets_df: DataFrame of bets (already filtered to production strategy)
        n_perms: number of permutation shuffles
        window_days: window size for local shuffling (preserves seasonal structure)
        seed: random seed

    Returns dict with real ROI, permutation distribution stats, and significance.
    """
    if bets_df.empty or len(bets_df) < 20:
        return {"error": "Too few bets for permutation test"}

    rng = np.random.default_rng(seed)

    # Real ROI
    real_roi = compute_flat_roi(bets_df)

    # Permutation distribution
    perm_rois = []
    for i in range(n_perms):
        perm_df = permute_outcomes_within_windows(bets_df, window_days, rng)
        perm_rois.append(compute_flat_roi(perm_df))

    perm_rois = np.array(perm_rois)

    # Statistics
    percentile_rank = float(np.searchsorted(np.sort(perm_rois), real_roi) / n_perms)
    empirical_p = float(np.mean(perm_rois >= real_roi))

    if empirical_p < 0.01:
        verdict = "HIGHLY SIGNIFICANT (p < 0.01)"
    elif empirical_p < 0.05:
        verdict = "SIGNIFICANT (p < 0.05)"
    elif empirical_p < 0.10:
        verdict = "MARGINAL (p < 0.10)"
    else:
        verdict = "NOT SIGNIFICANT"

    return {
        "n_bets": len(bets_df),
        "n_perms": n_perms,
        "window_days": window_days,
        "real_roi": round(real_roi * 100, 2),
        "perm_mean": round(float(np.mean(perm_rois)) * 100, 2),
        "perm_std": round(float(np.std(perm_rois)) * 100, 2),
        "perm_median": round(float(np.median(perm_rois)) * 100, 2),
        "perm_5th": round(float(np.percentile(perm_rois, 5)) * 100, 2),
        "perm_95th": round(float(np.percentile(perm_rois, 95)) * 100, 2),
        "percentile_rank": round(percentile_rank * 100, 1),
        "empirical_p_value": round(empirical_p, 4),
        "verdict": verdict,
    }


def print_permutation_report(result):
    """Print formatted permutation test results."""
    print(f"\n{'=' * 70}")
    print("  PERMUTATION SIGNIFICANCE TEST")
    print(f"{'=' * 70}")

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Bets: {result['n_bets']}, Permutations: {result['n_perms']}, "
          f"Window: {result['window_days']} days")
    print()
    print(f"  Real ROI:          {result['real_roi']:+.2f}%")
    print(f"  Permutation mean:  {result['perm_mean']:+.2f}% (std: {result['perm_std']:.2f}%)")
    print(f"  Permutation range: {result['perm_5th']:+.2f}% to {result['perm_95th']:+.2f}% (5th-95th)")
    print(f"  Percentile rank:   {result['percentile_rank']:.1f}%")
    print(f"  Empirical p-value: {result['empirical_p_value']:.4f}")
    print(f"  Verdict:           {result['verdict']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Run walk-forward to get bets, then test significance
    import nhl_walkforward as wf

    logger.info("Running walk-forward to generate bets...")
    wf_result = wf.run_walkforward(min_edge=0.03)
    bets_df = wf_result.get("bets_df", pd.DataFrame())

    if bets_df.empty:
        print("No bets generated")
    else:
        # Apply production filter: unders, has soft book, 5%+ blended edge
        has_blend = bets_df["blended_prob"].notna()
        has_soft = bets_df["has_soft"] == True
        is_under = bets_df["side"] == "UNDER"

        # Recompute blended edge vs soft
        prod = bets_df[has_blend & has_soft & is_under].copy()
        prod["_soft_imp"] = prod["soft_implied"].fillna(prod["implied_prob"])
        prod["_blend_edge"] = prod["blended_prob"] - prod["_soft_imp"]
        prod = prod[prod["_blend_edge"] >= 0.05]

        # Use blended prob as model_prob for ROI calculation
        prod["model_prob"] = prod["blended_prob"]

        logger.info("Production bets: %d", len(prod))

        result = run_permutation_test(prod, n_perms=500, window_days=14)
        print_permutation_report(result)
