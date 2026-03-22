"""Outlier / tail-event analysis — how well do models predict extreme outcomes?"""

import logging
import numpy as np
import pandas as pd
import nhl_walkforward_v2 as wf_v2
import nhl_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze(bets_df, label):
    print(f"\n{'=' * 90}")
    print(f"  {label}: OUTLIER / TAIL ANALYSIS")
    print(f"{'=' * 90}")

    pg = bets_df.drop_duplicates(subset=["player_id", "date"]).copy()
    actuals = pg["actual_sog"].values
    preds = pg["pred_sog"].values

    print(f"  Unique player-games with odds: {len(pg)}")
    print(f"\n  ACTUAL SOG DISTRIBUTION:")
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        n = (actuals == t).sum()
        pct = n / len(actuals) * 100
        bar = "#" * int(pct)
        print(f"    {t} SOG: {n:5d} ({pct:5.1f}%) {bar}")
    n8 = (actuals >= 8).sum()
    print(f"   8+ SOG: {n8:5d} ({n8/len(actuals)*100:5.1f}%)")

    # LINE-LEVEL CALIBRATION
    print(f"\n  LINE-LEVEL OVER CALIBRATION:")
    print(f"  {'Line':>6s} {'N':>6s} {'Actual O%':>10s} {'Model P(O)':>10s} "
          f"{'Gap':>8s} {'Avg Odds':>9s} {'Flat Yield':>11s}")
    print(f"  " + "-" * 70)
    for line in sorted(bets_df[bets_df["side"] == "OVER"]["line"].unique()):
        overs = bets_df[(bets_df["side"] == "OVER") & (bets_df["line"] == line)]
        if len(overs) < 20:
            continue
        act = overs["won"].mean()
        mod = overs["model_prob"].mean()
        gap = act - mod
        avg_dec = overs["decimal_odds"].mean()
        yld = act * avg_dec - 1
        avg_am = overs["odds"].mean()
        print(f"  {line:6.1f} {len(overs):6d} {act:9.1%} {mod:9.1%} "
              f"{gap:+7.1%} {avg_am:+8.0f} {yld*100:+10.1f}%")

    # UNDER CALIBRATION
    print(f"\n  LINE-LEVEL UNDER CALIBRATION:")
    print(f"  {'Line':>6s} {'N':>6s} {'Actual U%':>10s} {'Model P(U)':>10s} "
          f"{'Gap':>8s} {'Avg Odds':>9s} {'Flat Yield':>11s}")
    print(f"  " + "-" * 70)
    for line in sorted(bets_df[bets_df["side"] == "UNDER"]["line"].unique()):
        unders = bets_df[(bets_df["side"] == "UNDER") & (bets_df["line"] == line)]
        if len(unders) < 20:
            continue
        act = unders["won"].mean()
        mod = unders["model_prob"].mean()
        gap = act - mod
        avg_dec = unders["decimal_odds"].mean()
        yld = act * avg_dec - 1
        avg_am = unders["odds"].mean()
        print(f"  {line:6.1f} {len(unders):6d} {act:9.1%} {mod:9.1%} "
              f"{gap:+7.1%} {avg_am:+8.0f} {yld*100:+10.1f}%")

    # BLOWOUT GAMES
    print(f"\n  BLOWOUT GAMES (actual >= 6 SOG):")
    blowouts = pg[pg["actual_sog"] >= 6]
    print(f"  Count: {len(blowouts)} ({len(blowouts)/len(pg)*100:.1f}% of games)")
    if len(blowouts) > 0:
        print(f"  Avg prediction: {blowouts['pred_sog'].mean():.2f} (actual avg: {blowouts['actual_sog'].mean():.1f})")
        hi_pred = (blowouts["pred_sog"] >= 4).sum()
        print(f"  Model predicted >= 4: {hi_pred}/{len(blowouts)} ({hi_pred/len(blowouts)*100:.0f}%)")
        lo_pred = (blowouts["pred_sog"] < 3).sum()
        print(f"  Model predicted < 3:  {lo_pred}/{len(blowouts)} ({lo_pred/len(blowouts)*100:.0f}%) -- TRUE SURPRISES")

    # SHUTOUTS
    print(f"\n  SHUTOUTS (0 SOG):")
    shutouts = pg[pg["actual_sog"] == 0]
    print(f"  Count: {len(shutouts)} ({len(shutouts)/len(pg)*100:.1f}% of games)")
    if len(shutouts) > 0:
        print(f"  Avg prediction: {shutouts['pred_sog'].mean():.2f}")
        hi_pred = (shutouts["pred_sog"] >= 2.5).sum()
        print(f"  Model predicted >= 2.5: {hi_pred}/{len(shutouts)} ({hi_pred/len(shutouts)*100:.0f}%) -- MISSED SHUTOUTS")

    # === TAIL BET STRATEGIES ===
    print(f"\n  {'=' * 80}")
    print(f"  TAIL BET STRATEGIES")
    print(f"  {'=' * 80}")

    def _strat(name, mask, min_n=10):
        subset = bets_df[mask]
        if len(subset) < min_n:
            print(f"\n  {name}: {len(subset)} bets (below min {min_n})")
            return
        won = subset["won"].mean()
        avg_dec = subset["decimal_odds"].mean()
        avg_odds = subset["odds"].mean()
        yld = won * avg_dec - 1
        # Simulate flat $1 betting
        total_profit = sum(
            (row["decimal_odds"] - 1) if row["won"] else -1
            for _, row in subset.iterrows()
        )
        print(f"\n  {name}:")
        print(f"    Bets: {len(subset)}, Win: {won*100:.1f}%, Avg odds: {avg_odds:+.0f}")
        print(f"    Flat yield: {yld*100:+.1f}%, Flat P/L per $1: ${total_profit/len(subset):+.3f}")

    # Over 4.5 and 5.5 (long shots)
    _strat("OVER 4.5 (all)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 4.5))
    _strat("OVER 4.5 (model P >= 0.35)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 4.5) & (bets_df["model_prob"] >= 0.35))
    _strat("OVER 4.5 (model P >= 0.40)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 4.5) & (bets_df["model_prob"] >= 0.40))
    _strat("OVER 5.5 (all)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 5.5), min_n=5)

    # Over 3.5 with confidence
    _strat("OVER 3.5 (model P >= 0.45)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 3.5) & (bets_df["model_prob"] >= 0.45))
    _strat("OVER 3.5 (model P >= 0.50)",
           (bets_df["side"] == "OVER") & (bets_df["line"] == 3.5) & (bets_df["model_prob"] >= 0.50))

    # Under 1.5 and 0.5 (shutout/near-shutout bets)
    _strat("UNDER 1.5 (all)",
           (bets_df["side"] == "UNDER") & (bets_df["line"] == 1.5))
    _strat("UNDER 1.5 (model P >= 0.55)",
           (bets_df["side"] == "UNDER") & (bets_df["line"] == 1.5) & (bets_df["model_prob"] >= 0.55))
    _strat("UNDER 0.5 (all)",
           (bets_df["side"] == "UNDER") & (bets_df["line"] == 0.5), min_n=5)

    # HIGH VARIANCE ANGLE: volatile players going over high lines
    _strat("OVER 3.5+ (high variance, var_ratio > 1.5)",
           (bets_df["side"] == "OVER") & (bets_df["line"] >= 3.5) & (bets_df["var_ratio"] > 1.5))
    _strat("OVER 3.5+ (high variance + model P >= 0.40)",
           (bets_df["side"] == "OVER") & (bets_df["line"] >= 3.5) &
           (bets_df["var_ratio"] > 1.5) & (bets_df["model_prob"] >= 0.40))

    # BLENDED PROB on tails
    has_blend = bets_df["blended_prob"].notna()
    _strat("OVER 3.5 (blended P >= 0.45)",
           has_blend & (bets_df["side"] == "OVER") & (bets_df["line"] == 3.5) &
           (bets_df["blended_prob"] >= 0.45))
    _strat("OVER 4.5 (blended P >= 0.30)",
           has_blend & (bets_df["side"] == "OVER") & (bets_df["line"] == 4.5) &
           (bets_df["blended_prob"] >= 0.30))
    _strat("UNDER 1.5 (blended P >= 0.55)",
           has_blend & (bets_df["side"] == "UNDER") & (bets_df["line"] == 1.5) &
           (bets_df["blended_prob"] >= 0.55))

    # SHARP CONFIRMS on tails
    sharp_agrees = bets_df["sharp_agrees"] == True
    _strat("OVER 3.5 (sharp confirms + model P >= 0.40)",
           sharp_agrees & (bets_df["side"] == "OVER") & (bets_df["line"] == 3.5) &
           (bets_df["model_prob"] >= 0.40))
    _strat("OVER 4.5 (sharp confirms)",
           sharp_agrees & (bets_df["side"] == "OVER") & (bets_df["line"] == 4.5))

    # PREDICTION ERROR ANALYSIS: where do big misses happen?
    print(f"\n  {'=' * 80}")
    print(f"  PREDICTION ERROR BY ACTUAL SOG BUCKET")
    print(f"  {'=' * 80}")
    print(f"  {'Actual':>8s} {'N':>6s} {'Avg Pred':>10s} {'Avg Error':>10s} {'RMSE':>8s}")
    print(f"  " + "-" * 50)
    for lo, hi, lbl in [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"),
                         (4, 4, "4"), (5, 5, "5"), (6, 8, "6-8"), (8, 99, "8+")]:
        mask = (pg["actual_sog"] >= lo) & (pg["actual_sog"] <= hi)
        sub = pg[mask]
        if len(sub) < 5:
            continue
        avg_pred = sub["pred_sog"].mean()
        avg_err = (sub["pred_sog"] - sub["actual_sog"]).mean()
        rmse = np.sqrt(((sub["pred_sog"] - sub["actual_sog"]) ** 2).mean())
        print(f"  {lbl:>8s} {len(sub):6d} {avg_pred:10.2f} {avg_err:+9.2f} {rmse:8.2f}")


# Run
print("Running walk-forward for outlier analysis...")
result = wf_v2.run_walkforward(
    starting_bankroll=100.0, kelly_fraction=0.25,
    min_edge=0.0, min_train_days=60, test_window_days=14, step_days=14,
)

analyze(result["v1"]["bets_df"], "V1")
analyze(result["v2"]["bets_df"], "V2")
