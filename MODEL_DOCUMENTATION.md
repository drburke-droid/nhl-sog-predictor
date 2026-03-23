# Complete Model Documentation

## Overview

This system contains four prediction models across two sports, each validated via walk-forward backtesting with real sportsbook odds. All models use XGBoost with time-based train/test splits (no random CV, no future data leakage). Profitable strategies are identified by blending model probabilities 50/50 with sharp book consensus, then comparing against soft book (BetMGM/PlayAlberta) odds.

---

## Model 1: NHL SOG V1 — Player Shots on Goal (Original)

### Architecture
- **Type**: Dual XGBoost regressors (separate forward/defense models)
- **Target**: Residual (actual SOG minus weighted baseline)
- **Baseline**: 55% season avg + 30% rolling 10 + 15% rolling 5
- **Training grain**: One row per player per game
- **Train/test**: Expanding window, 14-day holdout

### Features (18)

| Feature | Importance (Fwd) | Description | Source |
|---------|-----------------|-------------|--------|
| baseline_sog | 0.158 | Weighted avg SOG (season + rolling) | NHL API |
| pct_games_3plus | 0.093 | Fraction of games with 3+ SOG | NHL API |
| player_cv | 0.072 | Coefficient of variation of SOG | NHL API |
| toi_last_5 | 0.072 | Avg TOI last 5 games (minutes) | NHL API |
| avg_toi | 0.062 | Avg TOI last 20 games | NHL API |
| is_home | 0.060 | Home team indicator (1/0) | NHL API |
| opp_shots_allowed | 0.060 | Opponent avg SOG allowed/game (rolling 20) | NHL API |
| avg_shift_length | 0.057 | Avg shift length in seconds | NHL API |
| arena_bias | 0.055 | Scorekeeper generosity factor | NHL API |
| implied_team_total | 0.055 | Team expected goals from ML + total | Odds API |
| opp_shots_allowed_pos | 0.055 | Opponent SOG allowed to player's position | NHL API |
| rolling_pp_rate | — | PP goal rate (proxy for PP deployment) | NHL API |
| rest_days | 0.057 (D) | Days since last game (capped at 4) | NHL API |
| is_back_to_back | 0.067 (D) | Playing consecutive days (1/0) | NHL API |
| linemate_quality | — | Linemate SOG rates | NHL API |
| game_total | — | Vegas over/under total goals | Odds API |
| sog_prop_line | — | Market consensus player SOG line | Odds API |
| sharp_consensus_prob | — | Vig-free over prob from sharp books | Odds API |

### Hyperparameters
- n_estimators: 400, max_depth: 4, learning_rate: 0.04
- subsample: 0.8, colsample_bytree: 0.8
- min_child_weight: 10, reg_alpha: 1.0, reg_lambda: 3.0
- Sample weighting: volume (baseline >= 2.5: 1.5x) * recency (exp decay 120 days)

### Performance
- **MAE**: 1.00 (forward: 1.056, defense: 0.888)
- **Training data**: ~40,000 player-games (2025-26 season)
- **Holdout**: Last 14 days

### Walk-Forward Validated Strategies (6-book sharp consensus)
| Strategy | Bets | Win Rate | Yield |
|----------|------|----------|-------|
| Blend unders 8%+ edge | 85 | 60.0% | +22.1% |
| All bets 8%+ edge | 103 | 60.2% | +20.2% |
| Under 1.5 (blend 3%+) | 352 | 50.3% | +15.4% |
| Under 3.5 (blend 2%+) | 101 | 59.4% | +11.3% |
| Over 3.5 (blend 5%+) | 50 | 50.0% | +10.4% |
| Blend unders 5%+ edge | 238 | 55.0% | +10.1% |

### Sharp Book Consensus
6 books ranked by Brier score on 2025-26 NHL SOG props:
1. Coolbet (0.2421), 2. FanDuel (0.2440), 3. Caesars/WH (0.2445),
4. DraftKings (0.2447), 5. Pinnacle (0.2447), 6. BetOnline (0.2448)

Soft target: BetMGM (0.2449 Brier, 7.61% vig — highest of any major)

### Edge Calibration
- Edges below 5%: noise (no bet)
- 5-8% edge: ~30% of predicted edge is real
- 8-10% edge: ~70% real (walk-forward: 20%+ yield)
- 10%+ edge: fully real
- Kelly sizing: quarter Kelly, 8% cap, with edge-dependent shrinkage

---

## Model 2: NHL SOG V2 — Enhanced with MoneyPuck + Clustering

### Architecture
- Same dual XGBoost as V1, plus:
  - MoneyPuck shot-quality features from shot-level and game-by-game data
  - K-means player archetype clustering (k=4) on style/deployment features
  - Team-level clustering with matchup interaction features
  - Feature registry with null policies and coverage thresholds

### Features (35)

**Core Form (from V1):**
| Feature | Importance (Fwd) | Description |
|---------|-----------------|-------------|
| baseline_sog | 0.040 | Weighted baseline (same as V1) |
| player_cv | 0.046 | SOG coefficient of variation |
| pct_games_3plus | 0.032 | Rate of 3+ SOG games |
| rolling_sog_ewm | 0.036 | Exponentially weighted SOG (span=10) |

**Opportunity / Role:**
| Feature | Importance | Description |
|---------|-----------|-------------|
| avg_toi | 0.035 | Avg TOI last 20 games |
| toi_last_5 | — | Avg TOI last 5 |
| toi_trend | — | TOI last 5 / last 20 ratio |
| avg_shift_length | 0.037 (D) | Avg shift length (seconds) |
| rolling_pp_rate | 0.033 (D) | PP goal rate |
| pp_toi_pct | — | Fraction of TOI on power play (MoneyPuck) |
| ice_time_rank | 0.032 | TOI rank within team (MoneyPuck) |

**Shot Style (MoneyPuck — V2 exclusive):**
| Feature | Importance (Fwd) | Description | Source |
|---------|-----------------|-------------|--------|
| shots_per_60 | **0.062** (#1) | SOG per 60 min ice time (rolling 10 games) | MoneyPuck |
| slot_pct | 0.032 | Fraction of shots from slot zone | MoneyPuck shots |
| rebound_rate | — | Fraction of shots that are rebounds | MoneyPuck shots |
| rush_rate | — | Fraction of shots off the rush | MoneyPuck shots |
| avg_shot_distance | — | Avg shot distance in feet | MoneyPuck shots |
| xg_per_shot | 0.032 | Expected goals per shot attempt | MoneyPuck |
| high_danger_rate | 0.035 (D) | Fraction of HD shots | MoneyPuck |

**Opponent (expanded):**
| Feature | Importance | Description |
|---------|-----------|-------------|
| opp_shots_allowed | 0.039 | Opponent SOG allowed/game |
| opp_shots_allowed_pos | — | Opponent SOG allowed to position |
| opp_hd_shots_allowed | — | Opponent HD shots allowed (MoneyPuck) |
| opp_xg_allowed | 0.034 (D) | Opponent xGA (MoneyPuck) |
| opp_pace | 0.035 (D) | Opponent shot attempts/game |

**Game Context:**
| Feature | Description |
|---------|-------------|
| is_home | Home indicator |
| rest_days | Days since last game |
| is_back_to_back | Back-to-back indicator |
| game_total | Vegas O/U total |
| implied_team_total | Implied team goals |

**Player Cluster Features (V2 exclusive):**
| Feature | Importance | Description |
|---------|-----------|-------------|
| cluster_id | — | Player archetype (0-3) via K-means |
| cluster_distance | — | Distance to cluster centroid |
| cluster_mean_sog | 0.039 | Avg SOG for players in this cluster |
| cluster_x_opp_hd | 0.034 (D) | Cluster SOG * opponent HD allowed |

**Team Cluster Matchup Features (V2 exclusive, added 2026-03-23):**
| Feature | Description |
|---------|-------------|
| opp_team_cluster_ga | Opponent team archetype's avg goals allowed |
| opp_team_cluster_save | Opponent team archetype's save percentage |
| player_cluster_x_opp_team_cluster | Player archetype SOG * opponent team archetype interaction |

### Player Clustering
- **Method**: K-means, k=4 (selected by silhouette score, tested k=4-8)
- **Clustering features**: avg_toi, pp_toi_pct, slot_pct, avg_shot_distance, rebound_rate, rush_rate, shots_per_60, avg_shift_length
- **NOT clustered on**: the target variable (SOG)
- **Refitted**: at every training run

### Team Clustering
- **Method**: K-means, k=3-6 (varies by window)
- **Clustering features (11)**: gf_per_game, xgf, hd_chances_for, pp_goals_per_game, ga_per_game, xga, hd_chances_against, save_pct, pace, fivon5_xg_pct, hd_ratio
- **Goaltending**: Captured via save_pct in clustering features
- **Matchup matrix**: How each cluster archetype performs (goals scored) against each other archetype
- **Refitted**: monthly during walk-forward

### Performance
- **MAE**: 1.004 (forward: 1.056, defense: 0.888)
- **Ablation**: V2-only features without V1 are worse; the combination is what works
- **MoneyPuck features**: 27% of total feature importance

### Walk-Forward Validated Strategies (with team clusters)
| Strategy | Bets | Win Rate | Yield |
|----------|------|----------|-------|
| Under 4.5 | 15 | 86.7% | +51.9% |
| Under 3.5/4.5 | 148 | 65.5% | +10.9% |
| Blend unders 8%+ | 113 | 55.8% | +10.9% |
| Under 1.5 (blend 3%+) | 531 | 47.8% | +9.8% |
| Blend unders 5%+ | 324 | 55.6% | +9.2% |
| Under 3.5 standalone | 183 | 66.1% | +8.9% |
| Forward unders (3%+) | — | — | +5.3% |

### Data Sources
- **NHL API** (api-web.nhle.com/v1): schedules, boxscores, player game stats, shift data
- **MoneyPuck** (peter-tanner.com mirror): game-by-game player stats (118K rows), shot-level data (95K rows) with xG, shot distance, coordinates, rebound/rush flags
- **The Odds API** (api.the-odds-api.com/v4): moneyline, totals, spreads, player SOG props from 32 bookmakers (regions: us, eu)

### Feature Registry
All 35 features registered in `feature_registry.yaml` with:
- Null policy (drop_row, fill_position, fill_default, fill_zero, fill_league)
- Coverage threshold (required: 99%, important: 90%, optional: 70%)
- Fallback hierarchy
- Leakage safety flag

### Null Handling
- No blanket fillna(0)
- Position group means for shot style features when MoneyPuck unavailable
- Cluster means for cluster-dependent features
- League averages as final fallback
- XGBoost handles remaining NaN natively

---

## Model 3: NHL Game Totals — Team-Level Predictions

### Architecture
- **Totals model**: XGBRegressor predicting total goals
- **Win probability model**: XGBClassifier predicting home win (binary)
- **Score simulation**: Correlated Poisson (correlation=0.12) for over/under, spread, and ML probabilities
- **Training grain**: One row per completed game

### Features (25)

**Team Rolling Stats (last 10 games):**
| Feature | Importance (Totals) | Description |
|---------|-------------------|-------------|
| home_gf_10 | — | Home team goals for per game |
| home_ga_10 | 0.048 | Home team goals against per game |
| away_gf_10 | — | Away goals for |
| away_ga_10 | 0.050 | Away goals against |
| home_sog_for_10 | 0.047 | Home SOG per game |
| home_sog_against_10 | — | Home SOG against |
| away_sog_for_10 | — | Away SOG per game |
| away_sog_against_10 | 0.049 (Win) | Away SOG against |

**MoneyPuck Advanced (last 10):**
| Feature | Importance | Description |
|---------|-----------|-------------|
| home_xgf_10 | 0.053 | Home expected goals for (MoneyPuck) |
| home_xga_10 | — | Home expected goals against |
| away_xgf_10 | 0.047 | Away xGF |
| away_xga_10 | — | Away xGA |
| home_hdcf_10 | — | Home high-danger chances for |
| home_hdca_10 | — | Home HD chances against |
| away_hdcf_10 | — | Away HD chances for |
| away_hdca_10 | 0.051 | Away HD chances against |

**Schedule:**
| Feature | Importance (Win) | Description |
|---------|-----------------|-------------|
| home_rest_days | — | Home team rest days |
| away_rest_days | 0.059 | Away team rest days |

**Odds-Derived:**
| Feature | Importance | Description |
|---------|-----------|-------------|
| game_total_line | 0.056 (T) | Vegas over/under line |
| implied_home_total | 0.052 (W) | Implied home goals |
| implied_away_total | — | Implied away goals |
| home_win_prob_implied | 0.052 (T), 0.072 (W) | Market-implied home win % |

**Team Cluster Matchup (added 2026-03-23):**
| Feature | Description |
|---------|-------------|
| cluster_home_gf | How home team's archetype scores vs away's archetype |
| cluster_away_gf | How away's archetype scores vs home's archetype |
| cluster_total | Expected total from archetype matchup history |

### Hyperparameters
- n_estimators: 200, max_depth: 4, learning_rate: 0.05
- subsample: 0.8, colsample_bytree: 0.8
- reg_alpha: 0.5, reg_lambda: 2.0, min_child_weight: 10

### Performance
- **Totals MAE**: 1.893 goals
- **Win accuracy**: 52.7% (Brier: 0.2707)
- **Training data**: ~1,136 games

### Walk-Forward Results
| Strategy | Bets | Win Rate | Yield |
|----------|------|----------|-------|
| Totals edge >= 10% | 101 | 59.4% | +6.8% |
| Totals edge >= 8% | 145 | 54.5% | +1.4% |
| Pure unders | 14 | 71.4% | +22.5% |

### Investigated and Rejected
- **Moneyline betting**: all strategies negative (-7% to -23% yield). Market is too efficient for game winners.
- **Player-opponent collaborative filtering**: correlation -0.019 (mean reversion). Individual player-opponent SOG history does not predict future performance.

---

## Model 4: MLB Pitcher Strikeouts (K Props)

### Architecture
- **Layer 1 (BF model)**: XGBRegressor predicting batters faced per start
- **Layer 2 (K/BF model)**: XGBRegressor predicting strikeout rate per batter faced
- **Layer 3 (Monte Carlo)**: 10,000 simulations per prediction sampling BF ~ Normal and K/BF ~ Normal, drawing K ~ Binomial(BF, K/BF)
- **Final prediction**: pred_K = pred_BF * pred_K_per_BF
- **Training grain**: One row per starting pitcher appearance

### BF Model Features (20)
| Feature | Importance | Description |
|---------|-----------|-------------|
| baseline_bf | 0.114 | Weighted avg batters faced |
| days_rest | 0.085 | Days since last start |
| avg_pitch_count | 0.063 | Rolling avg pitches thrown |
| market_k_line | 0.052 | Sportsbook K line for this start |
| pitches_last | 0.051 | Pitches thrown in last start |
| is_home | 0.048 | Home indicator |
| rolling_3_pc | 0.047 | Rolling 3-game pitch count |
| innings_last | 0.046 | IP in last start |
| rolling_whip | 0.046 | Rolling WHIP |
| pitches_per_bf | 0.045 | Pitch efficiency |

### K/BF Model Features (37)
| Feature | Importance | Description |
|---------|-----------|-------------|
| baseline_k_rate | 0.086 | Season K rate per BF |
| market_k_line | 0.060 | Sportsbook K line |
| k_minus_bb_rate | 0.059 | K% minus BB% (command metric) |
| two_strike_putaway_rate | 0.056 | Putaway rate with 2 strikes |
| opp_k_rate | 0.052 | Opposing team K rate |
| opp_contact_rate | 0.036 | Opposing team contact rate |
| pitcher_cv | 0.033 | K count variance |
| is_home | 0.031 | Home indicator |
| ff_usage | 0.025 | Fastball usage rate |
| csw_rate | 0.024 | Called strike + whiff rate |
| days_rest | 0.023 | Rest days |
| matchup_k_rate | 0.023 | Weighted matchup K rate vs opponent |

Additional features include: rolling velocity, arsenal diversity, zone rate, chase rate, whiff rate, pitch type breakdowns (SL, CH, CU usage), park K factor, opponent BB rate, first pitch strike rate, TTO K decay.

### Hyperparameters
- n_estimators: 400, max_depth: 4, learning_rate: 0.04
- subsample: 0.8, colsample_bytree: 0.8
- min_child_weight: 8, reg_alpha: 0.5, reg_lambda: 2.0

### Performance
- **MAE**: 1.681 strikeouts (vs 1.79 season avg baseline)
- **RMSE**: 2.158
- **Mean bias**: -0.047
- **Model vs market MAE**: 1.681 vs 1.698 (model wins by 0.017)
- **Over/under accuracy vs market line**: 55.0%
- **Training**: 4,062 starts, Test: 327 starts (Sept 16-30 2024)

### Calibration
| K Range | N | Avg Predicted | Avg Actual |
|---------|---|---------------|------------|
| < 4 K | 64 | 3.79 | 3.67 |
| 4-6 K | 209 | 4.92 | 5.10 |
| 6-8 K | 54 | 6.23 | 5.94 |

### Threshold Accuracy
| Line | Accuracy |
|------|----------|
| Over 3.5 | 74.9% |
| Over 4.5 | 67.0% |
| Over 5.5 | 66.1% |
| Over 6.5 | 76.5% |
| Over 7.5 | 86.5% |

### Walk-Forward Betting Strategy
Best strategy: BMG_blend_unders at 5%+ edge (blended 50/50 model + sharp consensus, targeting BetMGM)
- 265 qualifying bets over 2024 season
- 53% win rate, +5.0% yield on props
- Combined with MLB O/U and ML: $100 -> $240.55 over 2024 season

---

## Model 5: MLB O/U v1 — Game Totals via Pitch-Level Matchup

### Architecture
- **Matchup engine**: Simulates each pitcher-batter plate appearance at the pitch type level
- **Correction layer**: XGBRegressor learns residuals off Vegas line using matchup signals
- **Score simulation**: Correlated Poisson (correlation=0.10) for over/under probabilities
- **Training grain**: One row per completed game

### Matchup Computation Flow
1. Get pitcher's arsenal (rolling 7 games): pitch types, usage rates, whiff rates, CSW, zone rates
2. Get each batter's performance vs each pitch type (rolling, date-filtered — no leakage)
3. Compute matchup K rate: geometric mean of pitcher whiff ability and batter K vulnerability per pitch type, weighted by usage
4. Apply platoon K rate adjustment (pitcher hand × batter side)
5. Convert matchup K rate to expected runs per PA using calibrated linear weights
6. Apply platoon ERA multiplier (from 2024 pitcher ERA splits by lineup composition)
7. Apply platoon HR multiplier (same-side pitchers allow 20% fewer HR)
8. Apply per-pitcher individual platoon ERA split (50% shrinkage toward league average)
9. Blend individual batter platoon K rate (40% shrinkage)
10. Apply pitcher recent form multiplier (30% weight, heavy regression)
11. Predict starter depth from matchup difficulty (bad matchup = early hook)
12. Model bullpen innings with team-specific LHP/RHP ERA splits
13. Apply bullpen platoon management (67% optimal matchup rate)
14. Model closer separately (1 IP, closer-specific ERA)
15. Apply park run factor (from actual game scores, range 0.74-1.29)
16. Apply team defensive BABIP adjustment (60% weight)
17. Apply season month scoring factor (range 0.96-1.15)

### Platoon Multipliers (from 2024 data)

**ERA-based run multipliers (normalized to league avg 4.55 ERA):**
| Pitcher Hand | vs LHB | vs RHB | vs Switch |
|-------------|--------|--------|-----------|
| LHP | 0.79 | 1.06 | 0.88 |
| RHP | 1.12 | 1.08 | 1.00 |

**K rate multipliers:**
| Pitcher Hand | vs LHB | vs RHB | vs Switch |
|-------------|--------|--------|-----------|
| LHP | 1.05 | 1.01 | 0.92 |
| RHP | 1.00 | 1.01 | 0.94 |

**HR rate multipliers:**
| Pitcher Hand | vs LHB | vs RHB | vs Switch |
|-------------|--------|--------|-----------|
| LHP | 0.80 | 1.00 | 0.85 |
| RHP | 1.03 | 1.03 | 1.00 |

### League Averages (2024 calibration)
- Runs per game: 4.55
- K rate: 0.224, BB rate: 0.082
- BABIP: 0.296, HR per fly ball: 0.12
- Starter avg IP: 5.4, Bullpen avg ERA: 3.95

### Season Month Scoring Factors
| Month | Factor |
|-------|--------|
| March | 1.15 |
| April | 0.98 |
| May | 0.97 |
| June | 1.02 |
| July | 1.04 |
| August | 1.03 |
| September | 0.96 |

### Correction Model Features (21)
pred_home, pred_away, pred_total, home_starter_ip, away_starter_ip, home_matchup_k, away_matchup_k, home_tto_decay, away_tto_decay, home_form, away_form, home_bullpen_era, away_bullpen_era, home_bp_plat_era, away_bp_plat_era, home_lineup_lhb, away_lineup_lhb, home_defense, away_defense, park_factor, month, vegas_total

### Performance
- **Raw matchup MAE**: 3.37 (improved from 4.02 through iterations)
- **Corrected MAE**: 3.22 (anchored to Vegas line)
- **2024 walk-forward**: 8 windows, June-September

### Walk-Forward Validated Strategies
| Strategy | Bets | Win Rate | Yield |
|----------|------|----------|-------|
| UNDER when matchup << Vegas | 60 | 61.7% | +17.7% |
| UNDER when matchup < Vegas | 79 | 59.5% | +13.6% |
| ML Underdogs edge >= 8% | 244 | 42.2% | +8.6% |
| ML Favorites edge >= 3% | 57 | 56.1% | +5.3% |

### 2025 Out-of-Sample Validation
- Tested on 1,788 games (model trained on 2024 only)
- Raw matchup MAE: 3.83 (slightly degraded from 2024's 3.37)
- Win prediction: 53.5% on strong picks (above 50% baseline)
- Absolute run calibration degraded (7.09 pred vs 8.92 actual — higher-scoring 2025 environment)
- Directional signal survived out-of-sample

### Data Leakage Audit (passed)
- `_get_batter_vs_pitch`: uses `game_date < ?` filter (fixed from season-level)
- `_get_arsenal`: rolling 7 games with `game_date < ?` (fallback to season-level only when <7 starts)
- All rolling features: strict date ordering
- Walk-forward results unchanged after fixing leakage (signal was real)

### Full 2024 Season Simulation (all MLB bet types combined)
- Starting bankroll: $100
- Ending bankroll: $240.55 (+140.6%)
- 599 total bets (264 K props, 299 ML, 36 totals)
- Yield on wagered: +5.2%
- Max drawdown: 40.2%

---

## Betting Infrastructure

### Edge Calibration
All models use edge-dependent shrinkage before Kelly sizing:
- < 5% blended edge: no bet (noise)
- 5-6%: 25% of predicted edge used
- 6-8%: 40% of predicted edge
- 8-10%: 70% of predicted edge
- 10%+: 100% of predicted edge

### Kelly Sizing
- Fraction: 25% Kelly (quarter Kelly)
- Maximum: 8% of bankroll per bet
- Blended probability: 50% model + 50% sharp book consensus

### Sharp Book Consensus (NHL)
6 books by Brier score: Coolbet, FanDuel, Caesars/WH, DraftKings, Pinnacle, BetOnline

### Soft Book Target
BetMGM / PlayAlberta (highest vig at 7.61%, worst calibration among majors)

### Odds Collection
- The Odds API: regions us,eu — markets h2h, totals, spreads, player_shots_on_goal
- API key rotation with automatic fallback on 401 errors
- Pinnacle included since 2026-03-22

---

## Key Findings

1. **The market is efficient for game winners** — moneyline betting is not profitable in NHL. Only totals at 10%+ edge are marginally profitable (+6.8% yield).

2. **Player prop markets are inefficient** — SOG props at BetMGM have consistent 5-22% yield when blended with sharp consensus at high edge thresholds.

3. **Under bets dominate** — across both NHL and MLB, under props are far more profitable than overs. The models are best at identifying when pitchers/defense suppress scoring.

4. **Blending with sharp consensus is essential** — model-only edges are unreliable. The 50/50 blend with 6 sharp books filters noise and only retains real edges.

5. **MoneyPuck shot-quality data adds value** — V2's shots_per_60 is its most important feature. But V1's simpler baseline is still better for peak yield. The models are complementary.

6. **Team clustering captures defensive archetypes** — teams with elite defense + goaltending systematically suppress shots/goals. This generalizes across player types.

7. **Platoon splits are huge in MLB** — LHP vs LHB-heavy lineups produce 0.79x the normal runs. The market doesn't fully price this.

8. **Player-opponent history is noise** — investigated collaborative filtering; correlation was -0.019. Individual player-opponent SOG patterns mean-revert, not persist.

9. **Calibrated Kelly sizing matters** — raw model edges are overconfident. Edge-dependent shrinkage prevents over-betting on thin edges while preserving full Kelly on proven high-edge situations.

10. **Walk-forward is non-negotiable** — every profitable strategy was validated out-of-sample with expanding training windows. No random cross-validation, no lookahead.
