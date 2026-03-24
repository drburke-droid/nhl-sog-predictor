"""
NHL SOG Distribution Model — predicts full probability distribution, not just mean.

Replaces mean-only regression + heuristic probability conversion with:
1. XGBoost predicts mean SOG (existing)
2. Separate dispersion submodel predicts player-game variance
3. Negative Binomial distribution fitted with (mean, dispersion) per prediction
4. Direct P(over line) and P(under line) from the fitted distribution

This gives calibrated probabilities at each discrete line (1.5, 2.5, 3.5, 4.5)
instead of converting a point estimate heuristically.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Negative Binomial distribution helpers
# ---------------------------------------------------------------------------

def negbin_params(mean, variance):
    """Convert mean and variance to Negative Binomial (r, p) parameters.

    NB parameterization: mean = r*(1-p)/p, var = r*(1-p)/p^2
    So: p = mean/variance, r = mean*p/(1-p) = mean^2/(variance - mean)
    Requires variance > mean (overdispersed).
    """
    if variance <= mean or mean <= 0:
        # Not overdispersed — fall back to Poisson-like
        variance = mean * 1.05  # slight overdispersion

    p = mean / variance
    r = mean * p / (1 - p)
    return max(r, 0.5), min(max(p, 0.01), 0.99)


def negbin_prob_over(mean, variance, line):
    """P(X > line) using Negative Binomial distribution."""
    r, p = negbin_params(mean, variance)
    try:
        # P(X > line) = 1 - P(X <= floor(line))
        k = int(np.floor(line))
        p_under = stats.nbinom.cdf(k, r, p)
        return float(1.0 - p_under)
    except Exception:
        return 0.5


def negbin_prob_under(mean, variance, line):
    """P(X <= line) using Negative Binomial distribution."""
    return 1.0 - negbin_prob_over(mean, variance, line)


def negbin_full_distribution(mean, variance, max_k=15):
    """Get full PMF from Negative Binomial.

    Returns dict: {k: P(X=k)} for k=0 to max_k
    """
    r, p = negbin_params(mean, variance)
    dist = {}
    for k in range(max_k + 1):
        try:
            dist[k] = float(stats.nbinom.pmf(k, r, p))
        except Exception:
            dist[k] = 0.0
    return dist


# ---------------------------------------------------------------------------
# Dispersion Submodel
# ---------------------------------------------------------------------------

# Features that predict variance (not just mean)
DISPERSION_FEATURES = [
    "player_cv",           # historical volatility
    "baseline_sog",        # high-volume players have higher absolute variance
    "avg_toi",             # more TOI = more variance opportunity
    "is_back_to_back",     # fatigue increases randomness
    "rest_days",           # schedule effects
    "opp_shots_allowed",   # opponent pace
    "avg_shift_length",    # shift pattern volatility
    "pct_games_3plus",     # boom-bust tendency
]


def train_dispersion_model(df, mean_predictions):
    """Train XGBoost to predict per-game SOG variance.

    Target: squared residual (actual - predicted)^2 as proxy for variance.
    """
    avail = [f for f in DISPERSION_FEATURES if f in df.columns]
    if len(avail) < 3 or len(df) < 100:
        return None

    # Target: squared error as variance proxy
    residuals_sq = (df["shots"].values - mean_predictions) ** 2

    model = XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=3.0, min_child_weight=20,
        random_state=42, verbosity=0,
    )
    model.fit(df[avail].values, residuals_sq)
    return model, avail


def predict_variance(disp_model, disp_features, df, mean_preds=None, shrinkage=0.0):
    """Predict per-game variance using dispersion submodel.

    Args:
        shrinkage: blend toward Poisson (variance=mean). 0 = full NegBin, 1 = pure Poisson.
            Optimal value is cross-validated via calibrate_shrinkage().
    """
    if disp_model is None:
        # Fallback: use player_cv squared * mean as variance estimate
        if "player_cv" in df.columns and "baseline_sog" in df.columns:
            cv = df["player_cv"].fillna(1.0).values
            base = df["baseline_sog"].fillna(2.0).values
            raw_var = np.maximum(cv ** 2 * base, base * 1.05)
        else:
            raw_var = np.full(len(df), 3.0)
    else:
        avail = [f for f in disp_features if f in df.columns]
        raw_var = disp_model.predict(df[avail].values)
        raw_var = np.maximum(raw_var, 1.0)

    # Shrink toward Poisson baseline (variance = mean)
    if shrinkage > 0 and mean_preds is not None:
        poisson_var = np.maximum(mean_preds, 0.5)
        raw_var = (1 - shrinkage) * raw_var + shrinkage * poisson_var
        raw_var = np.maximum(raw_var, poisson_var * 1.01)  # stay slightly overdispersed

    return raw_var


def calibrate_shrinkage(df, mean_preds, disp_model, disp_features,
                        lines=None, alphas=None):
    """Cross-validate shrinkage parameter to minimize Brier score.

    Tests blends of NegBin variance and Poisson (variance=mean) to find
    the optimal balance. Returns best alpha and Brier scores per alpha.
    """
    if lines is None:
        lines = [1.5, 2.5, 3.5, 4.5]
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    actuals = df["shots"].values
    results = {}

    for alpha in alphas:
        var_preds = predict_variance(disp_model, disp_features, df,
                                     mean_preds=mean_preds, shrinkage=alpha)
        total_brier = 0.0
        for line in lines:
            pred_p_over = np.array([
                negbin_prob_over(m, v, line)
                for m, v in zip(mean_preds, var_preds)
            ])
            actual_over = (actuals > line).astype(float)
            total_brier += float(np.mean((pred_p_over - actual_over) ** 2))

        results[alpha] = total_brier / len(lines)

    best_alpha = min(results, key=results.get)
    logger.info("Shrinkage calibration: best alpha=%.2f (avg Brier=%.4f)",
                best_alpha, results[best_alpha])
    return best_alpha, results


# ---------------------------------------------------------------------------
# Full distribution prediction pipeline
# ---------------------------------------------------------------------------

def predict_line_probabilities(mean_pred, variance_pred, lines=None):
    """Get P(over) and P(under) for each standard prop line.

    Args:
        mean_pred: predicted mean SOG
        variance_pred: predicted variance

    Returns dict: {line: {over_prob, under_prob}}
    """
    if lines is None:
        lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    result = {}
    for line in lines:
        p_over = negbin_prob_over(mean_pred, variance_pred, line)
        result[line] = {
            "over_prob": round(p_over, 4),
            "under_prob": round(1 - p_over, 4),
        }
    return result


# ---------------------------------------------------------------------------
# Calibration evaluation
# ---------------------------------------------------------------------------

def evaluate_distribution_calibration(df, mean_preds, variance_preds):
    """Evaluate how well the distribution model calibrates at each line.

    For each standard line, compute predicted P(over) vs actual over rate.
    """
    lines = [1.5, 2.5, 3.5, 4.5]
    actuals = df["shots"].values

    results = []
    for line in lines:
        pred_p_over = np.array([
            negbin_prob_over(m, v, line)
            for m, v in zip(mean_preds, variance_preds)
        ])
        actual_over = (actuals > line).astype(float)

        n = len(actuals)
        pred_rate = pred_p_over.mean()
        actual_rate = actual_over.mean()

        # Brier score at this line
        brier = float(np.mean((pred_p_over - actual_over) ** 2))

        results.append({
            "line": line,
            "n": n,
            "pred_over_rate": round(pred_rate, 4),
            "actual_over_rate": round(actual_rate, 4),
            "gap": round(actual_rate - pred_rate, 4),
            "brier": round(brier, 4),
        })

    return results


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Test with a few example players
    print("=== Negative Binomial Distribution Examples ===\n")

    for name, mean, cv in [
        ("Low-volume D-man", 1.2, 0.9),
        ("Average forward", 2.3, 0.7),
        ("High-volume shooter", 3.8, 0.5),
        ("Volatile boom-bust", 2.5, 1.3),
    ]:
        variance = (cv * mean) ** 2 / mean  # var = cv^2 * mean
        variance = max(variance, mean * 1.05)

        print(f"{name}: mean={mean}, cv={cv}, variance={variance:.2f}")
        probs = predict_line_probabilities(mean, variance)
        for line, p in probs.items():
            print(f"  Line {line}: P(over)={p['over_prob']:.3f}, P(under)={p['under_prob']:.3f}")

        pmf = negbin_full_distribution(mean, variance, max_k=10)
        print(f"  PMF: {' '.join(f'{k}:{p:.3f}' for k, p in pmf.items())}")
        print()

    # Test on actual data
    import model as nhl_model
    from datetime import timedelta

    df = nhl_model._build_feature_dataframe()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        cutoff = df["date"].max() - timedelta(days=14)
        train = df[df["date"] <= cutoff]
        test = df[df["date"] > cutoff]

        # Get mean predictions from existing model
        feat_cols = [c for c in nhl_model.FEATURE_COLS if c != "sog_prop_line"]
        avail = [f for f in feat_cols if f in train.columns]

        from xgboost import XGBRegressor
        mdl = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.04,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                           reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0)
        mdl.fit(train[avail].values, train["sog_residual"].values)

        train_pred = np.maximum(train["baseline_sog"].values + mdl.predict(train[avail].values), 0)
        test_pred = np.maximum(test["baseline_sog"].values + mdl.predict(test[avail].values), 0)

        # Train dispersion model
        result = train_dispersion_model(train, train_pred)
        if result is not None:
            disp_model, disp_feats = result
        else:
            disp_model, disp_feats = None, []

        # Calibrate shrinkage on training data
        print("=== Shrinkage Calibration (on training data) ===\n")
        best_alpha, alpha_results = calibrate_shrinkage(
            train, train_pred, disp_model, disp_feats)
        for alpha, brier in sorted(alpha_results.items()):
            tag = " <-- best" if alpha == best_alpha else ""
            print(f"  alpha={alpha:.1f}: avg Brier={brier:.4f}{tag}")

        # Evaluate on holdout with calibrated shrinkage
        test_var_raw = predict_variance(disp_model, disp_feats, test,
                                        mean_preds=test_pred, shrinkage=0.0)
        test_var_cal = predict_variance(disp_model, disp_feats, test,
                                        mean_preds=test_pred, shrinkage=best_alpha)

        print(f"\n=== Distribution Model Calibration on Holdout (alpha={best_alpha:.1f}) ===\n")

        # Raw NegBin (no shrinkage)
        print("NegBin (no shrinkage):")
        cal_raw = evaluate_distribution_calibration(test, test_pred, test_var_raw)
        for c in cal_raw:
            print(f"  Line {c['line']}: pred {c['pred_over_rate']:.3f}, "
                  f"actual {c['actual_over_rate']:.3f}, gap {c['gap']:+.3f}, "
                  f"Brier {c['brier']:.4f}")

        # Calibrated NegBin
        print(f"\nNegBin (shrinkage={best_alpha:.1f}):")
        cal_cal = evaluate_distribution_calibration(test, test_pred, test_var_cal)
        for c in cal_cal:
            print(f"  Line {c['line']}: pred {c['pred_over_rate']:.3f}, "
                  f"actual {c['actual_over_rate']:.3f}, gap {c['gap']:+.3f}, "
                  f"Brier {c['brier']:.4f}")

        # Poisson baseline
        print("\nPoisson (variance=mean):")
        cal_poisson = evaluate_distribution_calibration(test, test_pred, test_pred)
        for c in cal_poisson:
            print(f"  Line {c['line']}: pred {c['pred_over_rate']:.3f}, "
                  f"actual {c['actual_over_rate']:.3f}, gap {c['gap']:+.3f}, "
                  f"Brier {c['brier']:.4f}")

        # Brier comparison: all three
        print("\nBrier score comparison (lower = better):")
        for raw, cal, po in zip(cal_raw, cal_cal, cal_poisson):
            scores = {"NegBin_raw": raw["brier"], f"NegBin_a{best_alpha:.1f}": cal["brier"],
                      "Poisson": po["brier"]}
            best = min(scores, key=scores.get)
            print(f"  Line {raw['line']}: " +
                  " | ".join(f"{k}={v:.4f}" for k, v in scores.items()) +
                  f" -> {best}")
