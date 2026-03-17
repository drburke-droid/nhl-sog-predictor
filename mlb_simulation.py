"""
MLB Pitcher Strikeout Distribution & Betting Layer.

Monte Carlo simulation to convert point predictions (BF, K/BF) into
full strikeout distributions, prop probabilities, and fair odds.
"""

import numpy as np
from typing import Optional

# --- Odds Conversion Utilities ---

def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5

def prob_to_american(prob: float) -> float:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0.0
    if prob > 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    else:
        return round(100.0 * (1.0 - prob) / prob)

def prob_to_fair_decimal(prob: float) -> float:
    """Convert probability to fair decimal odds (no vig)."""
    if prob <= 0:
        return 0.0
    return round(1.0 / prob, 3)

def compute_edge(model_prob: float, market_odds: float) -> dict:
    """Compute edge and expected value for a given side.

    Args:
        model_prob: Model's probability for this outcome
        market_odds: American odds from sportsbook

    Returns:
        dict with implied_prob, edge, expected_value
    """
    implied = american_to_implied_prob(market_odds)
    edge = model_prob - implied
    # EV per unit staked
    if market_odds > 0:
        payout = market_odds / 100.0
    else:
        payout = 100.0 / abs(market_odds)
    ev = model_prob * payout - (1 - model_prob) * 1.0
    return {
        "implied_prob": round(implied, 4),
        "edge": round(edge, 4),
        "expected_value": round(ev, 4),
    }


# --- Variance Estimation ---

def estimate_bf_variance(pred_bf: float, pitcher_bf_std: float = 0.0,
                         default_cv: float = 0.18) -> float:
    """Estimate std dev of batters faced for simulation.

    Uses pitcher-specific std if available, otherwise default CV.
    """
    if pitcher_bf_std > 0:
        return pitcher_bf_std
    return pred_bf * default_cv


def estimate_kbf_variance(pred_kbf: float, pitcher_kbf_std: float = 0.0,
                          default_cv: float = 0.20) -> float:
    """Estimate std dev of K/BF rate for simulation."""
    if pitcher_kbf_std > 0:
        return pitcher_kbf_std
    return pred_kbf * default_cv


# --- Monte Carlo Simulation ---

PROP_LINES = [3.5, 4.5, 5.5, 6.5, 7.5]

def simulate_strikeouts(pred_bf: float, pred_kbf: float,
                        bf_std: float = 0.0, kbf_std: float = 0.0,
                        n_sims: int = 10000, seed: int = 42) -> dict:
    """Run Monte Carlo simulation for strikeout distribution.

    For each simulation:
    1. Sample BF from Normal(pred_bf, bf_std), clipped to [5, 40]
    2. Sample K/BF from Normal(pred_kbf, kbf_std), clipped to [0.02, 0.55]
    3. For each sampled BF, draw K ~ Binomial(round(BF), K/BF)

    Args:
        pred_bf: Predicted batters faced
        pred_kbf: Predicted K rate per BF
        bf_std: Std dev for BF sampling (0 = use default CV)
        kbf_std: Std dev for K/BF sampling (0 = use default CV)
        n_sims: Number of simulations
        seed: Random seed for reproducibility

    Returns:
        dict with mean_k, median_k, std_k, prop probabilities, fair odds, distribution
    """
    rng = np.random.default_rng(seed)

    # Estimate variances if not provided
    if bf_std <= 0:
        bf_std = estimate_bf_variance(pred_bf)
    if kbf_std <= 0:
        kbf_std = estimate_kbf_variance(pred_kbf)

    # Sample BF
    bf_samples = rng.normal(pred_bf, bf_std, size=n_sims)
    bf_samples = np.clip(bf_samples, 5, 40).astype(int)

    # Sample K/BF rate (add uncertainty to the rate itself)
    kbf_samples = rng.normal(pred_kbf, kbf_std, size=n_sims)
    kbf_samples = np.clip(kbf_samples, 0.02, 0.55)

    # For each sim, draw K from Binomial(BF, K/BF)
    k_samples = rng.binomial(bf_samples, kbf_samples)

    # Compute distribution stats
    mean_k = float(np.mean(k_samples))
    median_k = float(np.median(k_samples))
    std_k = float(np.std(k_samples))

    # Prop line probabilities
    prop_probs = {}
    fair_odds = {}
    for line in PROP_LINES:
        p_over = float(np.mean(k_samples > line))
        p_under = 1.0 - p_over
        prop_probs[f"P_over_{line:.1f}"] = round(p_over, 4)
        prop_probs[f"P_under_{line:.1f}"] = round(p_under, 4)
        fair_odds[f"fair_over_{line:.1f}"] = prob_to_american(p_over)
        fair_odds[f"fair_under_{line:.1f}"] = prob_to_american(p_under)

    # Integer K probabilities (for detailed distribution)
    max_k = int(k_samples.max()) + 1
    k_dist = {}
    for k_val in range(0, min(max_k + 1, 16)):
        k_dist[k_val] = round(float(np.mean(k_samples == k_val)), 4)

    # Cumulative probabilities for 3+, 4+, 5+, 6+, 7+
    cum_probs = {}
    for threshold in [3, 4, 5, 6, 7]:
        cum_probs[f"P_{threshold}_plus"] = round(float(np.mean(k_samples >= threshold)), 4)

    return {
        "mean_k": round(mean_k, 2),
        "median_k": round(median_k, 1),
        "std_k": round(std_k, 2),
        "n_sims": n_sims,
        "pred_bf": round(pred_bf, 1),
        "pred_kbf": round(pred_kbf, 4),
        "bf_std": round(bf_std, 2),
        "kbf_std": round(kbf_std, 4),
        **prop_probs,
        **fair_odds,
        **cum_probs,
        "k_distribution": k_dist,
    }


def simulate_with_market(pred_bf: float, pred_kbf: float,
                         bf_std: float = 0.0, kbf_std: float = 0.0,
                         market_line: Optional[float] = None,
                         market_over_odds: Optional[float] = None,
                         market_under_odds: Optional[float] = None,
                         n_sims: int = 10000, seed: int = 42) -> dict:
    """Full simulation with optional market comparison.

    Runs Monte Carlo simulation and if market odds are provided,
    computes edge and expected value for the specific market line.
    """
    result = simulate_strikeouts(pred_bf, pred_kbf, bf_std, kbf_std, n_sims, seed)

    if market_line is not None:
        result["market_line"] = market_line
        # Re-simulate with same seed to get exact line prob
        rng = np.random.default_rng(seed)
        _bf_std = bf_std if bf_std > 0 else estimate_bf_variance(pred_bf)
        _kbf_std = kbf_std if kbf_std > 0 else estimate_kbf_variance(pred_kbf)
        bf_s = np.clip(rng.normal(pred_bf, _bf_std, size=n_sims), 5, 40).astype(int)
        kbf_s = np.clip(rng.normal(pred_kbf, _kbf_std, size=n_sims), 0.02, 0.55)
        k_s = rng.binomial(bf_s, kbf_s)

        model_prob_over = float(np.mean(k_s > market_line))
        model_prob_under = 1.0 - model_prob_over

        result["model_prob_over"] = round(model_prob_over, 4)
        result["model_prob_under"] = round(model_prob_under, 4)

        if market_over_odds is not None:
            result["market_over_odds"] = market_over_odds
            result["market_implied_prob_over"] = american_to_implied_prob(market_over_odds)
            edge_info = compute_edge(model_prob_over, market_over_odds)
            result["edge_over"] = edge_info["edge"]
            result["ev_over"] = edge_info["expected_value"]

        if market_under_odds is not None:
            result["market_under_odds"] = market_under_odds
            result["market_implied_prob_under"] = american_to_implied_prob(market_under_odds)
            edge_info = compute_edge(model_prob_under, market_under_odds)
            result["edge_under"] = edge_info["edge"]
            result["ev_under"] = edge_info["expected_value"]

    return result
