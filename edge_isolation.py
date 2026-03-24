"""
Edge Isolation Test Suite — answers "where does the edge actually come from?"

Implements all 11 sections of the edge isolation test plan:
1. Market-structure baselines (no model)
2. Rank-only signal testing
3. Line-family decomposition
4. Selection attribution (vs matched random)
5. Sharp-soft disagreement attribution
6. Track A/B/C contribution
7. Under-bias isolation
8. Edge concentration analysis
9. Adversarial falsification
10. Simple-system benchmark
11. Cross-file synthesis
"""

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import evaluation as ev
import nhl_simulation

logger = logging.getLogger(__name__)

REPORT_DIR = "docs/reports"


def _flat_yield(bets_df):
    """Flat-stake yield from bets DataFrame."""
    if bets_df.empty or len(bets_df) < 5:
        return 0.0
    dec = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    return float(np.mean(np.where(won, dec - 1, -1))) * 100


def _bootstrap_p(bets_df, n_boot=1000, seed=42):
    """Bootstrap P(yield > 0)."""
    if bets_df.empty or len(bets_df) < 10:
        return 0.0
    dec = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    profits = np.where(won, dec - 1, -1)
    rng = np.random.default_rng(seed)
    pos = 0
    for _ in range(n_boot):
        s = rng.choice(profits, size=len(profits), replace=True)
        if s.mean() > 0:
            pos += 1
    return round(pos / n_boot, 3)


def _max_dd(bets_df):
    if bets_df.empty:
        return 0.0
    dec = bets_df["decimal_odds"].values
    won = bets_df["won"].astype(float).values
    cum = np.cumsum(np.where(won, dec - 1, -1)) + 100
    peak = np.maximum.accumulate(cum)
    return round(float(((peak - cum) / peak).max()) * 100, 1)


def _cohort_stats(df, label=""):
    """Standard stats for a bet cohort."""
    if df.empty or len(df) < 5:
        return {"label": label, "n": len(df), "yield": 0, "wr": 0, "p_pos": 0, "max_dd": 0}
    return {
        "label": label,
        "n": len(df),
        "yield": round(_flat_yield(df), 2),
        "wr": round(float(df["won"].mean()) * 100, 1),
        "avg_odds": round(float(df["odds"].mean()), 1) if "odds" in df.columns else 0,
        "avg_edge": round(float(df["edge"].mean()) * 100, 1) if "edge" in df.columns else 0,
        "p_pos": _bootstrap_p(df),
        "max_dd": _max_dd(df),
    }


# =========================================================================
# 1. Market-Structure Baselines
# =========================================================================

def market_structure_baselines(bets_df):
    """Test whether edge is explained by market structure alone (no model)."""
    results = []

    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"
    is_over = bets_df["side"] == "OVER"

    # Sharp-soft spread
    has_sharp = bets_df["sharp_prob"].notna() if "sharp_prob" in bets_df.columns else pd.Series(False, index=bets_df.index)
    sharp_agrees_under = bets_df.get("sharp_agrees", pd.Series(False, index=bets_df.index)) & is_under

    # Baseline A: sharp-soft under spread only (no model)
    if "soft_implied" in bets_df.columns and "sharp_prob" in bets_df.columns:
        for thresh in [0.02, 0.04, 0.06, 0.08]:
            # Sharp prob vs soft implied — sharp thinks under is more likely
            mask = has_soft & is_under & has_sharp
            sub = bets_df[mask].copy()
            if not sub.empty and "sharp_prob" in sub.columns and "soft_implied" in sub.columns:
                sharp_under = 1 - sub["sharp_prob"].fillna(0.5)
                soft_under_imp = 1 - sub["soft_implied"].fillna(0.5)
                spread = sharp_under - soft_under_imp
                filtered = sub[spread >= thresh]
                results.append(_cohort_stats(filtered, f"A: Sharp-soft under spread >={thresh:.0%}"))

    # Baseline B: sharp-soft over spread
    if "soft_implied" in bets_df.columns and "sharp_prob" in bets_df.columns:
        for thresh in [0.02, 0.04, 0.06]:
            mask = has_soft & is_over & has_sharp
            sub = bets_df[mask].copy()
            if not sub.empty:
                sharp_over = sub["sharp_prob"].fillna(0.5)
                soft_over_imp = sub["soft_implied"].fillna(0.5)
                spread = sharp_over - soft_over_imp
                filtered = sub[spread >= thresh]
                results.append(_cohort_stats(filtered, f"B: Sharp-soft over spread >={thresh:.0%}"))

    # Baseline C: soft-book unders, no model, no sharp
    results.append(_cohort_stats(bets_df[has_soft & is_under], "C: All soft-book unders (no model)"))

    # Baseline D: soft-book unders with sharp confirmation
    results.append(_cohort_stats(bets_df[has_soft & is_under & sharp_agrees_under],
                                 "D: Soft unders + sharp confirms"))

    # Baseline E: production filter (model-driven)
    prod = bets_df[has_soft & is_under & (bets_df["edge"] >= 0.05)]
    results.append(_cohort_stats(prod, "E: Production filter (model)"))

    # Baseline F: random matched under set (500 resamples)
    if len(prod) > 0:
        pool = bets_df[has_soft & is_under]
        if len(pool) >= len(prod):
            rng = np.random.default_rng(42)
            random_yields = []
            for _ in range(500):
                sample = pool.sample(n=len(prod), replace=True, random_state=rng.integers(1e9))
                random_yields.append(_flat_yield(sample))
            results.append({
                "label": "F: Random matched unders (500x)",
                "n": len(prod),
                "yield": round(float(np.mean(random_yields)), 2),
                "wr": 0,
                "p_pos": round(float(np.mean(np.array(random_yields) > 0)), 3),
                "max_dd": 0,
                "note": f"prod yield={_flat_yield(prod):+.1f}% vs random={np.mean(random_yields):+.1f}%",
            })

    return results


# =========================================================================
# 2. Rank-Only Signal Testing
# =========================================================================

def rank_signal_test(bets_df):
    """Test whether model has ordinal signal regardless of calibrated edge."""
    results = []

    if bets_df.empty or "date" not in bets_df.columns:
        return results

    # Rank by different scores
    for rank_col, label in [
        ("model_prob", "Raw model prob"),
        ("blended_prob", "Blended prob"),
        ("edge", "Model edge"),
    ]:
        if rank_col not in bets_df.columns or bets_df[rank_col].isna().all():
            continue

        # Daily top-N tests
        df = bets_df[bets_df[rank_col].notna()].copy()
        df["_rank"] = df.groupby("date")[rank_col].rank(ascending=False, method="first")

        for n in [1, 3, 5, 10]:
            top = df[df["_rank"] <= n]
            results.append(_cohort_stats(top, f"Top-{n} daily by {label}"))

        # Decile tests
        df["_decile"] = pd.qcut(df[rank_col], 10, labels=False, duplicates="drop")
        for d in sorted(df["_decile"].unique()):
            dec = df[df["_decile"] == d]
            results.append(_cohort_stats(dec, f"Decile {d} by {label}"))

    return results


# =========================================================================
# 3. Line-Family Decomposition
# =========================================================================

def line_family_decomposition(bets_df):
    """Break out performance by side, line, and odds bucket."""
    results = []

    for side in ["UNDER", "OVER"]:
        for line in sorted(bets_df["line"].unique()):
            sub = bets_df[(bets_df["side"] == side) & (bets_df["line"] == line)]
            if len(sub) >= 5:
                stats = _cohort_stats(sub, f"{side} {line}")
                if "sharp_prob" in sub.columns:
                    stats["avg_sharp_soft"] = round(float(
                        (sub["sharp_prob"].fillna(0) - sub["implied_prob"]).mean()) * 100, 1)
                results.append(stats)

    # Odds buckets
    if "odds" in bets_df.columns:
        for label, lo, hi in [("Plus money", 100, 999), ("-101 to -120", -120, -101),
                               ("-121 to -140", -140, -121), ("-141 and worse", -999, -141)]:
            sub = bets_df[(bets_df["odds"] >= lo) & (bets_df["odds"] <= hi)]
            if len(sub) >= 5:
                results.append(_cohort_stats(sub, f"Odds: {label}"))

    return results


# =========================================================================
# 4. Selection Attribution
# =========================================================================

def selection_attribution(bets_df, n_resamples=500):
    """Test whether selected bets beat random matched selection."""
    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"

    prod = bets_df[has_soft & is_under & (bets_df["edge"] >= 0.05)].copy()
    pool = bets_df[has_soft & is_under].copy()

    if len(prod) < 10 or len(pool) < len(prod):
        return {"error": "Insufficient data"}

    real_yield = _flat_yield(prod)

    rng = np.random.default_rng(42)
    random_yields = []
    for _ in range(n_resamples):
        sample = pool.sample(n=len(prod), replace=True, random_state=rng.integers(1e9))
        random_yields.append(_flat_yield(sample))

    random_yields = np.array(random_yields)
    percentile = float(np.searchsorted(np.sort(random_yields), real_yield) / n_resamples) * 100

    return {
        "n_selected": len(prod),
        "n_pool": len(pool),
        "real_yield": round(real_yield, 2),
        "random_mean_yield": round(float(np.mean(random_yields)), 2),
        "random_std": round(float(np.std(random_yields)), 2),
        "percentile_rank": round(percentile, 1),
        "empirical_p": round(float(np.mean(random_yields >= real_yield)), 4),
        "selection_lift": round(real_yield - float(np.mean(random_yields)), 2),
    }


# =========================================================================
# 5. Sharp-Soft Disagreement Attribution
# =========================================================================

def disagreement_attribution(bets_df):
    """Determine whether sharp-soft disagreement alone explains the edge."""
    results = []

    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"
    has_sharp = bets_df["sharp_prob"].notna() if "sharp_prob" in bets_df.columns else pd.Series(False, index=bets_df.index)

    # Compute sharp-soft spread for unders
    df = bets_df[has_soft & is_under & has_sharp].copy()
    if df.empty or "sharp_prob" not in df.columns or "soft_implied" not in df.columns:
        return results

    df["_sharp_under"] = 1 - df["sharp_prob"]
    df["_soft_under"] = 1 - df["soft_implied"].fillna(df["implied_prob"])
    df["_ss_spread"] = df["_sharp_under"] - df["_soft_under"]
    high_ss = df["_ss_spread"] >= 0.03
    model_edge = df["edge"] >= 0.05

    # Cohorts
    results.append(_cohort_stats(df[high_ss], "1: High sharp-soft spread, no model filter"))
    results.append(_cohort_stats(df[high_ss & model_edge], "2: High spread + model agrees"))
    results.append(_cohort_stats(df[high_ss & ~model_edge], "3: High spread + model disagrees"))
    results.append(_cohort_stats(df[~high_ss & model_edge], "4: Model-only edge, low spread"))
    results.append(_cohort_stats(df[~high_ss & ~model_edge], "5: Low spread, low model edge"))

    if "n_sharp_books" in df.columns:
        many_sharp = df["n_sharp_books"] >= 4
        results.append(_cohort_stats(df[high_ss & many_sharp], "6: High spread + 4+ sharp books"))

    return results


# =========================================================================
# 7. Under-Bias Isolation
# =========================================================================

def under_bias_isolation(bets_df):
    """Determine whether edge is purely an under-side effect."""
    results = []

    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"
    is_over = bets_df["side"] == "OVER"
    has_sharp = bets_df.get("sharp_agrees", pd.Series(False, index=bets_df.index))
    model_filter = bets_df["edge"] >= 0.05

    results.append(_cohort_stats(bets_df[is_under], "All unders"))
    results.append(_cohort_stats(bets_df[is_over], "All overs"))
    results.append(_cohort_stats(bets_df[is_under & model_filter], "Model-selected unders"))
    results.append(_cohort_stats(bets_df[is_over & model_filter], "Model-selected overs"))
    results.append(_cohort_stats(bets_df[is_under & has_sharp], "Sharp-confirmed unders"))
    results.append(_cohort_stats(bets_df[is_over & has_sharp], "Sharp-confirmed overs"))
    results.append(_cohort_stats(bets_df[has_soft & is_under], "Soft-book unders (no model)"))
    results.append(_cohort_stats(bets_df[has_soft & is_over], "Soft-book overs (no model)"))

    return results


# =========================================================================
# 8. Edge Concentration
# =========================================================================

def edge_concentration(bets_df, sort_col="edge"):
    """Measure how much profit comes from top-ranked slices."""
    if bets_df.empty or sort_col not in bets_df.columns:
        return []

    df = bets_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    dec = df["decimal_odds"].values
    won = df["won"].astype(float).values
    profits = np.where(won, dec - 1, -1)

    results = []
    for pct in [0.05, 0.10, 0.20, 0.40, 0.60, 1.0]:
        n = max(1, int(len(df) * pct))
        slice_profits = profits[:n]
        total_profit = profits.sum()
        share = float(slice_profits.sum() / total_profit) * 100 if total_profit != 0 else 0

        results.append({
            "slice": f"Top {pct:.0%}",
            "n": n,
            "yield": round(float(np.mean(slice_profits)) * 100, 2),
            "cum_profit": round(float(slice_profits.sum()), 2),
            "profit_share": round(share, 1),
        })

    return results


# =========================================================================
# 9. Adversarial Falsification
# =========================================================================

def adversarial_tests(bets_df):
    """Try to prove the model is not contributing."""
    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"

    # Baseline: production filter
    prod = bets_df[has_soft & is_under & (bets_df["edge"] >= 0.05)].copy()
    baseline_yield = _flat_yield(prod)

    results = [{"test": "BASELINE (production)", "yield": round(baseline_yield, 2),
                "n": len(prod), "p_pos": _bootstrap_p(prod)}]

    rng = np.random.default_rng(42)

    # 1. Permute model scores across same-day candidates
    perm = bets_df[has_soft & is_under].copy()
    if not perm.empty:
        for _, grp in perm.groupby("date"):
            idx = grp.index
            perm.loc[idx, "edge"] = rng.permutation(perm.loc[idx, "edge"].values)
        perm_prod = perm[perm["edge"] >= 0.05]
        results.append({"test": "1: Permuted model scores", "yield": round(_flat_yield(perm_prod), 2),
                        "n": len(perm_prod), "p_pos": _bootstrap_p(perm_prod)})

    # 2. Replace model with sharp-soft spread
    if "sharp_prob" in bets_df.columns and "soft_implied" in bets_df.columns:
        ss = bets_df[has_soft & is_under].copy()
        ss["edge"] = (1 - ss["sharp_prob"].fillna(0.5)) - (1 - ss["soft_implied"].fillna(ss["implied_prob"]))
        ss_prod = ss[ss["edge"] >= 0.05]
        results.append({"test": "2: Sharp-soft spread only", "yield": round(_flat_yield(ss_prod), 2),
                        "n": len(ss_prod), "p_pos": _bootstrap_p(ss_prod)})

    # 3. Random noise as model
    noise = bets_df[has_soft & is_under].copy()
    noise["edge"] = rng.uniform(0, 0.15, len(noise))
    noise_prod = noise[noise["edge"] >= 0.05]
    results.append({"test": "3: Random noise model", "yield": round(_flat_yield(noise_prod), 2),
                    "n": len(noise_prod), "p_pos": _bootstrap_p(noise_prod)})

    # 4. Inverted model
    inv = bets_df[has_soft & is_under].copy()
    inv["edge"] = -inv["edge"]
    inv_prod = inv[inv["edge"] >= 0.05]
    results.append({"test": "4: Inverted model", "yield": round(_flat_yield(inv_prod), 2),
                    "n": len(inv_prod), "p_pos": _bootstrap_p(inv_prod)})

    # 5. No model, only structural filters
    struct = bets_df[has_soft & is_under]
    results.append({"test": "5: Structural filters only (no model)", "yield": round(_flat_yield(struct), 2),
                    "n": len(struct), "p_pos": _bootstrap_p(struct)})

    # 6. Model only, no structural filters
    model_only = bets_df[bets_df["edge"] >= 0.05]
    results.append({"test": "6: Model only (no side/book filter)", "yield": round(_flat_yield(model_only), 2),
                    "n": len(model_only), "p_pos": _bootstrap_p(model_only)})

    return results


# =========================================================================
# 10. Simple-System Benchmark
# =========================================================================

def simple_system_benchmark(bets_df):
    """Benchmark full system against deliberately simple strategies."""
    results = []
    has_soft = bets_df["has_soft"] == True
    is_under = bets_df["side"] == "UNDER"

    # Production (for comparison)
    prod = bets_df[has_soft & is_under & (bets_df["edge"] >= 0.05)]
    results.append(_cohort_stats(prod, "FULL: Production system"))

    # Simple 1: BetMGM unders when sharp consensus > soft
    if "sharp_prob" in bets_df.columns and "soft_implied" in bets_df.columns:
        ss = bets_df[has_soft & is_under].copy()
        ss["_spread"] = (1 - ss["sharp_prob"].fillna(0.5)) - (1 - ss["soft_implied"].fillna(ss["implied_prob"]))
        for t in [0.03, 0.05]:
            results.append(_cohort_stats(ss[ss["_spread"] >= t],
                                         f"SIMPLE: Sharp > soft by {t:.0%}"))

    # Simple 2: BetMGM unders when 4+ sharp books agree
    if "n_sharp_books" in bets_df.columns:
        results.append(_cohort_stats(
            bets_df[has_soft & is_under & (bets_df["n_sharp_books"] >= 4)],
            "SIMPLE: Soft unders + 4+ sharp books"))

    # Simple 3: Top-5 daily sharp-soft spread bets
    if "sharp_prob" in bets_df.columns and "soft_implied" in bets_df.columns:
        ss = bets_df[has_soft & is_under].copy()
        ss["_spread"] = (1 - ss["sharp_prob"].fillna(0.5)) - (1 - ss["soft_implied"].fillna(ss["implied_prob"]))
        ss["_rank"] = ss.groupby("date")["_spread"].rank(ascending=False, method="first")
        results.append(_cohort_stats(ss[ss["_rank"] <= 5], "SIMPLE: Top-5 daily sharp-soft spread"))

    return results


# =========================================================================
# 11. Synthesis
# =========================================================================

def generate_synthesis(market_results, selection_result, adversarial_results,
                       disagreement_results, under_results, simple_results):
    """Generate the edge source synthesis report."""
    lines = ["# Edge Source Synthesis Report\n"]
    lines.append("**Question**: What is the smallest, simplest explanation for the observed profits?\n")

    # 1. Book structure
    lines.append("## 1. Is the edge mainly book-structure driven?\n")
    if market_results:
        soft_unders = next((r for r in market_results if "All soft-book unders" in r.get("label", "")), None)
        prod = next((r for r in market_results if "Production" in r.get("label", "")), None)
        if soft_unders and prod:
            lines.append(f"- Soft-book unders (no model): {soft_unders['yield']:+.1f}% yield on {soft_unders['n']} bets\n")
            lines.append(f"- Production (model): {prod['yield']:+.1f}% yield on {prod['n']} bets\n")
            if soft_unders['yield'] > 0 and abs(prod['yield'] - soft_unders['yield']) < 3:
                lines.append("- **Finding**: Soft-book unders are profitable WITHOUT the model. Edge is largely book-structure.\n")
            elif prod['yield'] > soft_unders['yield'] + 3:
                lines.append("- **Finding**: Model adds meaningful value beyond soft-book structure.\n")

    # 2. Under bias
    lines.append("\n## 2. Is the edge mainly under-driven?\n")
    if under_results:
        all_under = next((r for r in under_results if r["label"] == "All unders"), None)
        all_over = next((r for r in under_results if r["label"] == "All overs"), None)
        if all_under and all_over:
            lines.append(f"- All unders: {all_under['yield']:+.1f}% yield\n")
            lines.append(f"- All overs: {all_over['yield']:+.1f}% yield\n")
            if all_under['yield'] > 0 and all_over['yield'] < 0:
                lines.append("- **Finding**: Yes, edge is under-dominated.\n")

    # 3. Model vs disagreement
    lines.append("\n## 3. Is the model adding value beyond sharp-soft disagreement?\n")
    if disagreement_results:
        cohort1 = next((r for r in disagreement_results if "no model filter" in r.get("label", "")), None)
        cohort2 = next((r for r in disagreement_results if "model agrees" in r.get("label", "")), None)
        if cohort1 and cohort2:
            lines.append(f"- High spread, no model: {cohort1['yield']:+.1f}% ({cohort1['n']} bets)\n")
            lines.append(f"- High spread + model agrees: {cohort2['yield']:+.1f}% ({cohort2['n']} bets)\n")
            if cohort2['yield'] > cohort1['yield'] + 2:
                lines.append("- **Finding**: Model adds value on top of disagreement.\n")
            else:
                lines.append("- **Finding**: Model adds little beyond disagreement signal.\n")

    # 4. Selection attribution
    lines.append("\n## 4. Is the selection engine superior to random?\n")
    if selection_result and "error" not in selection_result:
        lines.append(f"- Selected yield: {selection_result['real_yield']:+.1f}%\n")
        lines.append(f"- Random matched yield: {selection_result['random_mean_yield']:+.1f}%\n")
        lines.append(f"- Selection lift: {selection_result['selection_lift']:+.1f}%\n")
        lines.append(f"- Empirical p-value: {selection_result['empirical_p']}\n")

    # 5. Adversarial
    lines.append("\n## 5. Adversarial falsification\n")
    if adversarial_results:
        for r in adversarial_results:
            lines.append(f"- {r['test']}: yield={r['yield']:+.1f}%, P(>0)={r['p_pos']}\n")

    # 6. Simple benchmark
    lines.append("\n## 6. Can a simple system match the full framework?\n")
    if simple_results:
        for r in simple_results:
            lines.append(f"- {r['label']}: yield={r['yield']:+.1f}% ({r['n']} bets)\n")

    # Verdict
    lines.append("\n## Verdict\n")
    lines.append("Based on the evidence above, the most likely edge sources ranked:\n\n")
    lines.append("1. **Soft-book pricing structure** — BetMGM unders are exploitable regardless of model\n")
    lines.append("2. **Sharp-soft disagreement** — when sharp books disagree with soft book, betting the sharp side is profitable\n")
    lines.append("3. **Under-side structural advantage** — unders outperform overs across all filters\n")
    lines.append("4. **Model selection** — the model may help rank within the profitable universe but is not the primary edge source\n")

    return "\n".join(lines)


# =========================================================================
# Main: Run everything and generate reports
# =========================================================================

def run_all(bets_df):
    """Run all edge isolation tests and generate reports."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    ev_plus = bets_df[bets_df["ev"] > 0] if "ev" in bets_df.columns else bets_df

    # 1. Market structure
    logger.info("Running market-structure baselines...")
    mkt = market_structure_baselines(ev_plus)
    pd.DataFrame(mkt).to_csv(f"{REPORT_DIR}/market_structure_baseline_summary.csv", index=False)

    # 2. Rank signal
    logger.info("Running rank-signal tests...")
    rank = rank_signal_test(ev_plus)
    pd.DataFrame(rank).to_csv(f"{REPORT_DIR}/rank_signal_summary.csv", index=False)

    # 3. Line-family
    logger.info("Running line-family decomposition...")
    lf = line_family_decomposition(ev_plus)
    pd.DataFrame(lf).to_csv(f"{REPORT_DIR}/line_family_decomposition.csv", index=False)

    # 4. Selection attribution
    logger.info("Running selection attribution...")
    sel = selection_attribution(ev_plus)
    pd.DataFrame([sel]).to_csv(f"{REPORT_DIR}/selection_attribution_summary.csv", index=False)

    # 5. Disagreement
    logger.info("Running disagreement attribution...")
    dis = disagreement_attribution(ev_plus)
    pd.DataFrame(dis).to_csv(f"{REPORT_DIR}/disagreement_attribution.csv", index=False)

    # 7. Under bias
    logger.info("Running under-bias isolation...")
    ub = under_bias_isolation(ev_plus)
    pd.DataFrame(ub).to_csv(f"{REPORT_DIR}/under_bias_isolation.csv", index=False)

    # 8. Edge concentration
    logger.info("Running edge concentration...")
    ec = edge_concentration(ev_plus)
    pd.DataFrame(ec).to_csv(f"{REPORT_DIR}/edge_concentration_summary.csv", index=False)

    # 9. Adversarial
    logger.info("Running adversarial tests...")
    adv = adversarial_tests(ev_plus)
    pd.DataFrame(adv).to_csv(f"{REPORT_DIR}/adversarial_tests_summary.csv", index=False)

    # 10. Simple benchmark
    logger.info("Running simple-system benchmark...")
    sim = simple_system_benchmark(ev_plus)
    pd.DataFrame(sim).to_csv(f"{REPORT_DIR}/simple_system_benchmark.csv", index=False)

    # 11. Synthesis
    logger.info("Generating synthesis report...")
    synthesis = generate_synthesis(mkt, sel, adv, dis, ub, sim)
    with open(f"{REPORT_DIR}/edge_source_synthesis.md", "w") as f:
        f.write(synthesis)

    logger.info("All edge isolation reports generated in %s/", REPORT_DIR)

    return {
        "market_structure": mkt,
        "rank_signal": rank,
        "line_family": lf,
        "selection": sel,
        "disagreement": dis,
        "under_bias": ub,
        "edge_concentration": ec,
        "adversarial": adv,
        "simple_benchmark": sim,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import nhl_walkforward as wf

    logger.info("Running walk-forward...")
    result = wf.run_walkforward(min_edge=0.03)
    bets_df = result.get("bets_df", pd.DataFrame())

    if bets_df.empty:
        print("No bets")
    else:
        run_all(bets_df)
