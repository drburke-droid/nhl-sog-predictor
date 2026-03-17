"""
NHL Player Shots on Goal Distribution & Betting Layer.

Monte Carlo simulation to convert point predictions into full SOG
distributions, prop probabilities, and fair odds.

Uses a Gamma-Poisson mixture (negative binomial) which is the natural
distribution for count data with overdispersion — exactly what SOG is.
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


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1.0 + odds / 100.0
    elif odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 2.0


def prob_to_american(prob: float) -> float:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0.0
    if prob > 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    else:
        return round(100.0 * (1.0 - prob) / prob)


def compute_edge(model_prob: float, market_odds: float) -> dict:
    """Compute edge and expected value for a given side.

    Args:
        model_prob: Model's probability for this outcome
        market_odds: American odds from sportsbook

    Returns dict with implied_prob, edge, expected_value.
    """
    implied = american_to_implied_prob(market_odds)
    decimal_odds = american_to_decimal(market_odds)
    edge = model_prob - implied
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
    return {
        "implied_prob": round(implied, 4),
        "edge": round(edge, 4),
        "expected_value": round(ev, 4),
    }


# --- Monte Carlo Simulation ---

PROP_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]


def simulate_sog(pred_sog: float, var_ratio: float = 1.0,
                 model_std: float = 0.3, n_sims: int = 10000,
                 seed: int = 42) -> dict:
    """Run Monte Carlo simulation for SOG distribution.

    Uses a Gamma-Poisson mixture (negative binomial) which naturally
    handles overdispersion in count data.

    Two uncertainty layers:
      1. Model uncertainty — pred_sog is an estimate with error
      2. Game-to-game variance — even a perfect prediction has randomness

    The Gamma-Poisson mixture captures both:
      rate ~ Gamma(alpha, scale)   => model + intrinsic uncertainty
      sog  ~ Poisson(rate)         => count randomness

    This yields Var[SOG] = mean * effective_var_ratio, matching
    the empirical overdispersion we observe in player SOG data.

    Args:
        pred_sog: Predicted SOG (mean of distribution)
        var_ratio: Player's variance/mean ratio (1.0 = Poisson, >1 = overdispersed)
        model_std: Additional model uncertainty std dev
        n_sims: Number of simulations
        seed: Random seed

    Returns dict with mean, std, P_over/under for each line, fair odds, distribution.
    """
    rng = np.random.default_rng(seed)
    pred_sog = max(pred_sog, 0.1)

    # Effective var ratio: player variance + model uncertainty
    effective_vr = max(var_ratio, 1.01) + (model_std ** 2) / pred_sog

    # Gamma-Poisson parameterization
    # E[rate] = pred_sog, Var[SOG] = pred_sog * effective_vr
    alpha = pred_sog / (effective_vr - 1)  # shape
    scale = effective_vr - 1               # scale = 1/beta

    # Sample rates from Gamma, then counts from Poisson
    rates = rng.gamma(shape=alpha, scale=scale, size=n_sims)
    rates = np.maximum(rates, 0.001)
    sog_samples = rng.poisson(rates)

    # Distribution stats
    mean_sog = float(np.mean(sog_samples))
    std_sog = float(np.std(sog_samples))

    # Prop line probabilities
    prop_probs = {}
    fair_odds = {}
    for line in PROP_LINES:
        p_over = float(np.mean(sog_samples > line))
        p_under = 1.0 - p_over
        prop_probs[f"P_over_{line}"] = round(p_over, 4)
        prop_probs[f"P_under_{line}"] = round(p_under, 4)
        if 0.01 < p_over < 0.99:
            fair_odds[f"fair_over_{line}"] = prob_to_american(p_over)
            fair_odds[f"fair_under_{line}"] = prob_to_american(p_under)

    # Integer SOG probabilities
    max_val = min(int(sog_samples.max()) + 1, 12)
    sog_dist = {}
    for k in range(max_val + 1):
        sog_dist[k] = round(float(np.mean(sog_samples == k)), 4)

    # Cumulative: P(1+), P(2+), ..., P(5+)
    cum_probs = {}
    for t in [1, 2, 3, 4, 5]:
        cum_probs[f"P_{t}_plus"] = round(float(np.mean(sog_samples >= t)), 4)

    return {
        "mean_sog": round(mean_sog, 2),
        "std_sog": round(std_sog, 2),
        "pred_sog": round(pred_sog, 2),
        "var_ratio": round(var_ratio, 3),
        "effective_var_ratio": round(effective_vr, 3),
        "n_sims": n_sims,
        **prop_probs,
        **fair_odds,
        **cum_probs,
        "sog_distribution": sog_dist,
    }


def simulate_with_market(pred_sog: float, var_ratio: float = 1.0,
                         model_std: float = 0.3,
                         market_line: Optional[float] = None,
                         market_over_odds: Optional[float] = None,
                         market_under_odds: Optional[float] = None,
                         n_sims: int = 10000, seed: int = 42) -> dict:
    """Full simulation with market comparison.

    Runs Monte Carlo simulation and computes edge/EV against market odds.
    """
    result = simulate_sog(pred_sog, var_ratio, model_std, n_sims, seed)

    if market_line is not None:
        result["market_line"] = market_line

        # Get model probability for the market's specific line
        p_over_key = f"P_over_{market_line}"
        if p_over_key in result:
            model_prob_over = result[p_over_key]
        else:
            # Re-simulate for non-standard line
            rng = np.random.default_rng(seed)
            effective_vr = max(var_ratio, 1.01) + (model_std ** 2) / max(pred_sog, 0.1)
            alpha = max(pred_sog, 0.1) / (effective_vr - 1)
            scale = effective_vr - 1
            rates = np.maximum(rng.gamma(alpha, scale, n_sims), 0.001)
            sog_s = rng.poisson(rates)
            model_prob_over = float(np.mean(sog_s > market_line))

        model_prob_under = 1.0 - model_prob_over
        result["model_prob_over"] = round(model_prob_over, 4)
        result["model_prob_under"] = round(model_prob_under, 4)

        if market_over_odds is not None:
            result["market_over_odds"] = market_over_odds
            edge_info = compute_edge(model_prob_over, market_over_odds)
            result["edge_over"] = edge_info["edge"]
            result["ev_over"] = edge_info["expected_value"]

        if market_under_odds is not None:
            result["market_under_odds"] = market_under_odds
            edge_info = compute_edge(model_prob_under, market_under_odds)
            result["edge_under"] = edge_info["edge"]
            result["ev_under"] = edge_info["expected_value"]

    return result
