"""
Full-season simulation: all MLB bet types on one bankroll.

Combines:
1. Strikeout props (from mlb_walkforward.py strategies)
2. Game totals UNDER (from mlb_ou_v1 matchup model)
3. Moneyline (from mlb_ou_v1 matchup model)

Uses quarter Kelly with calibrated sizing on $100 starting bankroll.
"""

import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from xgboost import XGBRegressor

import mlb_ou_v1 as ou_model
import nhl_game_model  # for simulate_game

logger = logging.getLogger(__name__)


def _american_to_prob(odds):
    if odds > 0: return 100.0 / (odds + 100.0)
    elif odds < 0: return -odds / (-odds + 100.0)
    return 0.5

def _american_to_decimal(odds):
    if odds > 0: return odds / 100.0 + 1
    elif odds < 0: return 100.0 / (-odds) + 1
    return 2.0


def run_full_season(starting_bankroll=100.0, kelly_frac=0.25, max_bet_pct=0.08):
    """Simulate the full 2024 season with all bet types on one bankroll."""

    # === BUILD GAME-LEVEL DATA (O/U + ML) ===
    logger.info("Building MLB O/U training data...")
    ou_df = ou_model.build_training_data()
    if ou_df.empty:
        return {"error": "No O/U data"}

    ou_df["date"] = pd.to_datetime(ou_df["date"])
    ou_df = ou_df.sort_values("date").reset_index(drop=True)

    # O/U correction features
    feat_cols = [
        "pred_home", "pred_away", "pred_total",
        "home_starter_ip", "away_starter_ip",
        "home_matchup_k", "away_matchup_k",
        "home_tto_decay", "away_tto_decay",
        "home_form", "away_form",
        "home_bullpen_era", "away_bullpen_era",
        "home_bp_plat_era", "away_bp_plat_era",
        "home_lineup_lhb", "away_lineup_lhb",
        "home_defense", "away_defense",
        "park_factor", "month", "vegas_total",
    ]
    avail = [f for f in feat_cols if f in ou_df.columns]

    # === WALK-FORWARD LOOP ===
    season_start = ou_df["date"].min()
    season_end = ou_df["date"].max()
    first_test = season_start + timedelta(days=90)

    bankroll = starting_bankroll
    all_bets = []
    daily_bankroll = [(season_start.strftime("%Y-%m-%d"), bankroll)]

    current = first_test
    while current < season_end:
        test_end = min(current + timedelta(days=14), season_end)
        train = ou_df[ou_df["date"] < current]
        test = ou_df[(ou_df["date"] >= current) & (ou_df["date"] <= test_end)]

        if len(train) < 300 or len(test) == 0:
            current += timedelta(days=14)
            continue

        # Train correction model
        train_v = train[train["vegas_total"].notna()].copy()
        if len(train_v) >= 100:
            train_v["residual"] = train_v["actual_total"] - train_v["vegas_total"]
            mdl = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=5.0,
                min_child_weight=20, random_state=42, verbosity=0,
            )
            mdl.fit(train_v[avail].values, train_v["residual"].values)
            pred_res = mdl.predict(test[avail].values)
            corrected = np.where(
                test["vegas_total"].notna(),
                test["vegas_total"].fillna(0).values + pred_res,
                test["pred_total"].values,
            )
        else:
            corrected = test["pred_total"].values

        # Process each test game
        for i in range(len(test)):
            row = test.iloc[i]
            gdate = row["date"].strftime("%Y-%m-%d")
            pred_total = float(corrected[i])
            matchup_raw = float(row["pred_total"])
            actual_total = int(row["actual_total"])
            vl = row.get("vegas_total")
            matchup_diff = matchup_raw - vl if vl and not pd.isna(vl) else 0

            # Simulate game
            raw_home = float(row["pred_home"])
            raw_away = float(row["pred_away"])
            ratio = raw_home / max(raw_home + raw_away, 0.1)
            sim = nhl_game_model.simulate_game(
                pred_total * ratio, pred_total * (1 - ratio), correlation=0.10)

            # --- TOTALS: UNDER when matchup < Vegas ---
            # Use RAW matchup total for simulation (not corrected) — the walk-forward
            # validated the raw matchup signal, not the corrected one
            if vl is not None and not pd.isna(vl) and matchup_diff < -0.3:
                raw_ratio = raw_home / max(raw_home + raw_away, 0.1)
                raw_sim = nhl_game_model.simulate_game(
                    matchup_raw * raw_ratio, matchup_raw * (1 - raw_ratio), correlation=0.10)

                for lt in [vl, vl + 0.5, vl - 0.5]:
                    uk = f"under_{lt}"
                    if uk in raw_sim:
                        prob = raw_sim[uk]
                        imp = 0.5238
                        edge = prob - imp
                        ev = prob * 0.909 - (1 - prob)
                        if ev > 0:
                            # Flat bet approach — walk-forward showed 59-62% WR
                            # Use small fixed Kelly since edge is proven by matchup direction
                            kf = min(0.02 if matchup_diff < -1.0 else 0.015, max_bet_pct)
                            wager = round(bankroll * kf, 2)
                            if wager >= 1.0:
                                won = actual_total <= vl
                                profit = round(wager * 0.909, 2) if won else -wager
                                bankroll = round(bankroll + profit, 2)
                                stars = 5 if matchup_diff < -1.0 else 3
                                all_bets.append({
                                    "date": gdate, "type": "TOTAL",
                                    "pick": f"UNDER {vl}",
                                    "odds": -110, "wager": wager,
                                    "won": won, "profit": profit,
                                    "bankroll": bankroll, "edge": edge,
                                    "stars": stars,
                                })
                        break

            # --- MONEYLINE ---
            home_ml = row.get("home_ml")
            away_ml = row.get("away_ml")
            actual_hw = int(row.get("actual_home_win", 0))

            if home_ml is not None and not pd.isna(home_ml) and away_ml is not None and not pd.isna(away_ml):
                home_ml = int(home_ml)
                away_ml = int(away_ml)

                for team, ml, model_p, won in [
                    (row["home_team"], home_ml, sim["home_win_prob"], actual_hw == 1),
                    (row["away_team"], away_ml, sim["away_win_prob"], actual_hw == 0),
                ]:
                    imp = _american_to_prob(ml)
                    dec = _american_to_decimal(ml)
                    edge = model_p - imp
                    ev = model_p * (dec - 1) - (1 - model_p)

                    # Dogs edge >= 8%
                    is_dog = ml > 0
                    qualifies = (is_dog and edge >= 0.08 and ev > 0) or \
                                (not is_dog and edge >= 0.03 and ev > 0)

                    if qualifies:
                        shrunk = edge * (0.7 if edge >= 0.08 else 0.4)
                        shrunk_prob = imp + shrunk
                        shrunk_ev = shrunk_prob * (dec - 1) - (1 - shrunk_prob)
                        if shrunk_ev <= 0:
                            continue
                        kf = min(max(shrunk_ev / (dec - 1), 0) * kelly_frac, max_bet_pct)
                        wager = round(bankroll * kf, 2)
                        if wager >= 1.0:
                            profit = round(wager * (dec - 1), 2) if won else -wager
                            bankroll = round(bankroll + profit, 2)
                            stars = 3 if is_dog and edge >= 0.12 else 2 if is_dog else 1
                            all_bets.append({
                                "date": gdate, "type": "ML",
                                "pick": f"{team} ML",
                                "odds": ml, "wager": wager,
                                "won": won, "profit": profit,
                                "bankroll": bankroll, "edge": edge,
                                "stars": stars,
                            })

            daily_bankroll.append((gdate, bankroll))

        current += timedelta(days=14)

    # === ADD STRIKEOUT PROP BETS ===
    logger.info("Running K prop walk-forward...")
    try:
        import mlb_walkforward as kwf
        k_result = kwf.run_walkforward(
            starting_bankroll=bankroll,  # start from current bankroll
            kelly_fraction=kelly_frac,
            max_kelly_pct=max_bet_pct,
            min_edge=0.03,
            min_train_days=60,
            test_window_days=14,
            step_days=14,
        )
        k_bets = k_result.get("bets_df", pd.DataFrame())

        if not k_bets.empty:
            # Use the best validated K prop strategy: BMG_blend_unders
            # Filter to unders with blended edge >= 5% (validated +10.6% yield in NHL analysis)
            has_blend = k_bets["blended_prob"].notna()
            is_under = k_bets["side"] == "UNDER"
            soft_ev = k_bets["has_soft"] == True

            # Best K prop strategies from walk-forward:
            # BMG_blend_unders at 5%+ edge, or sharp-confirms unders
            blend_unders = k_bets[has_blend & is_under & soft_ev].copy()
            blend_unders["blend_edge"] = blend_unders["blended_prob"] - blend_unders["soft_implied"].fillna(0.5238)
            qualified = blend_unders[blend_unders["blend_edge"] >= 0.05].copy()

            if not qualified.empty:
                qualified = qualified.sort_values("date")
                logger.info("K prop bets: %d qualified (blend_unders 5%%+)", len(qualified))

                for _, kb in qualified.iterrows():
                    soft_price = kb.get("soft_price") or -110
                    dec = _american_to_decimal(soft_price)
                    imp = _american_to_prob(soft_price)
                    prob = kb["blended_prob"]
                    edge = prob - imp
                    ev = prob * (dec - 1) - (1 - prob)

                    if ev <= 0:
                        continue

                    shrunk = edge * 0.5  # conservative for props
                    kf = min(max(shrunk / (dec - 1), 0) * kelly_frac, max_bet_pct)
                    wager = round(bankroll * kf, 2)
                    if wager < 1.0:
                        continue

                    won = kb["won"]
                    profit = round(wager * (dec - 1), 2) if won else -wager
                    bankroll = round(bankroll + profit, 2)

                    all_bets.append({
                        "date": kb["date"],
                        "type": "K_PROP",
                        "pick": f"U{kb['line']} {kb['pitcher']}",
                        "odds": soft_price,
                        "wager": wager,
                        "won": won,
                        "profit": profit,
                        "bankroll": bankroll,
                        "edge": edge,
                        "stars": 3 if edge >= 0.08 else 2,
                    })
    except Exception as exc:
        logger.warning("K prop walk-forward failed: %s", exc)

    # === RESULTS ===
    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    print()
    print("=" * 95)
    print("  MLB 2024 FULL-SEASON SIMULATION: $100 Starting Bankroll, Quarter Kelly")
    print("=" * 95)

    if bets_df.empty:
        print("  No bets generated")
        return

    # By type
    print(f"\n  Starting bankroll: ${starting_bankroll:.2f}")
    print(f"  Ending bankroll:   ${bankroll:.2f}")
    print(f"  Total profit:      ${bankroll - starting_bankroll:+.2f}")
    print(f"  Total return:      {(bankroll - starting_bankroll) / starting_bankroll * 100:+.1f}%")
    print(f"  Total bets:        {len(bets_df)}")
    print(f"  Overall win rate:  {bets_df['won'].mean() * 100:.1f}%")

    total_wagered = bets_df["wager"].sum()
    total_profit = bankroll - starting_bankroll
    print(f"  Total wagered:     ${total_wagered:.2f}")
    print(f"  Yield on wagered:  {total_profit / max(total_wagered, 1) * 100:+.1f}%")

    # Drawdown
    peak = starting_bankroll
    max_dd = 0
    for _, row in bets_df.iterrows():
        br = row["bankroll"]
        if br > peak:
            peak = br
        dd = (peak - br) / peak
        if dd > max_dd:
            max_dd = dd
    print(f"  Max drawdown:      {max_dd * 100:.1f}%")

    # By bet type
    print(f"\n  {'Type':10s} {'Bets':>6s} {'Wins':>6s} {'WR':>7s} {'Wagered':>10s} {'Profit':>10s} {'Yield':>8s}")
    print("  " + "-" * 65)
    for btype in sorted(bets_df["type"].unique()):
        sub = bets_df[bets_df["type"] == btype]
        w = sub["wager"].sum()
        p = sub["profit"].sum()
        wr = sub["won"].mean() * 100
        yld = p / max(w, 1) * 100
        print(f"  {btype:10s} {len(sub):6d} {int(sub['won'].sum()):6d} {wr:6.1f}% ${w:9.2f} ${p:+9.2f} {yld:+7.1f}%")

    # By month
    bets_df["month"] = pd.to_datetime(bets_df["date"]).dt.strftime("%Y-%m")
    print(f"\n  {'Month':>8s} {'Bets':>6s} {'Profit':>10s} {'Bankroll':>10s}")
    print("  " + "-" * 40)
    for month in sorted(bets_df["month"].unique()):
        sub = bets_df[bets_df["month"] == month]
        p = sub["profit"].sum()
        end_br = sub["bankroll"].iloc[-1]
        print(f"  {month:>8s} {len(sub):6d} ${p:+9.2f} ${end_br:9.2f}")

    # Bankroll curve key points
    print(f"\n  Bankroll milestones:")
    milestones = [25, 50, 75, 100, 150, 200, 300, 500]
    for m in milestones:
        hit = bets_df[bets_df["bankroll"] >= m]
        if not hit.empty:
            first = hit.iloc[0]
            print(f"    ${m:>4d} first reached: {first['date']} (bet #{hit.index[0] + 1})")

    print()
    print("=" * 95)

    return {"bets": bets_df, "final_bankroll": bankroll, "daily": daily_bankroll}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run_full_season()
