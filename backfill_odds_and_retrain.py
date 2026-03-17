"""
Backfill historical NHL odds, retrain model, and compare holdout metrics.

Usage: python backfill_odds_and_retrain.py

Pulls ~2.5 months of historical odds data (Jan 1 - Mar 15, 2026),
retrains the model with odds features, and compares holdout MAE
before and after.
"""

import json
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta

import requests
import numpy as np

import nhl_odds_collector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 3, 15)


def check_credits():
    """Check remaining API credits."""
    r = requests.get(
        f"{nhl_odds_collector.BASE_URL}/sports/{nhl_odds_collector.SPORT}/odds",
        params={
            "apiKey": nhl_odds_collector.API_KEY,
            "regions": "us",
            "markets": "h2h",
        },
    )
    remaining = int(r.headers.get("x-requests-remaining", 0))
    used = int(r.headers.get("x-requests-used", 0))
    return remaining, used


def backfill():
    """Backfill historical odds for the target date range."""
    conn = nhl_odds_collector.get_db()
    nhl_odds_collector.create_odds_tables(conn)

    # Check what we already have
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT event_id FROM nhl_odds_events"
        ).fetchall()
    }
    logger.info("Existing odds events: %d", len(existing))

    current = START_DATE
    total_events = 0
    total_props = 0
    total_credits_start = check_credits()[0]

    while current <= END_DATE:
        game_date = current.strftime("%Y-%m-%d")

        # Get events for this date
        dt = f"{game_date}T18:00:00Z"
        data, _ = nhl_odds_collector._api_get(
            f"{nhl_odds_collector.BASE_URL}/historical/sports/{nhl_odds_collector.SPORT}/events",
            {"apiKey": nhl_odds_collector.API_KEY, "date": dt},
        )

        if not data:
            current += timedelta(days=1)
            continue

        events = data.get("data", [])
        # Filter to this date's games
        next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
        day_events = [
            ev for ev in events
            if ev.get("commence_time", "").startswith(game_date)
            or ev.get("commence_time", "").startswith(next_day)
        ]

        new_events = [e for e in day_events if e["id"] not in existing]
        if not new_events:
            logger.info("%s: %d events (all already in DB)", game_date, len(day_events))
            current += timedelta(days=1)
            continue

        logger.info("%s: %d new events to process", game_date, len(new_events))

        for ev in new_events:
            eid = ev["id"]
            commence = ev.get("commence_time", "")

            # Query ~3 hours before game for pre-game odds
            try:
                ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                query_time = (ct - timedelta(hours=3)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            except (ValueError, TypeError):
                query_time = dt

            odds_data = nhl_odds_collector.get_historical_event_odds(
                eid, query_time
            )
            if odds_data:
                nhl_odds_collector._store_historical_odds(
                    conn, ev, odds_data, game_date
                )
                existing.add(eid)
                total_events += 1

                # Check for SOG props
                bks = odds_data.get("data", {}).get("bookmakers", [])
                has_props = any(
                    m["key"] == "player_shots_on_goal"
                    for b in bks for m in b.get("markets", [])
                )
                if has_props:
                    total_props += 1

            time.sleep(0.15)

        conn.commit()

        if total_events % 25 == 0 and total_events > 0:
            credits_now = check_credits()[0]
            credits_used = total_credits_start - credits_now
            logger.info(
                "Progress: %d events (%d with SOG props), "
                "%d credits used, %d remaining",
                total_events, total_props, credits_used, credits_now,
            )

        current += timedelta(days=1)

    conn.commit()
    conn.close()

    credits_end = check_credits()[0]
    credits_used = total_credits_start - credits_end
    logger.info(
        "Backfill complete: %d events, %d with SOG props, %d credits used",
        total_events, total_props, credits_used,
    )
    return total_events, total_props


def retrain_and_evaluate():
    """Retrain model and compare holdout metrics with/without odds."""
    import model

    # Load current metrics for comparison
    try:
        with open("saved_model/meta.json") as f:
            old_meta = json.load(f)
        old_metrics = old_meta.get("metrics", {})
    except FileNotFoundError:
        old_metrics = {}

    logger.info("=" * 60)
    logger.info("BEFORE (no odds features):")
    logger.info("  Combined MAE: %s", old_metrics.get("mae"))
    logger.info(
        "  Forward MAE: %s",
        old_metrics.get("forward_model", {}).get("mae"),
    )
    logger.info(
        "  Defense MAE: %s",
        old_metrics.get("defense_model", {}).get("mae"),
    )
    logger.info("=" * 60)

    # Retrain with odds features
    logger.info("Retraining model with odds features...")
    new_metrics = model.train_model()

    logger.info("=" * 60)
    logger.info("AFTER (with odds features):")
    logger.info("  Combined MAE: %s", new_metrics.get("mae"))
    logger.info(
        "  Forward MAE: %s",
        new_metrics.get("forward_model", {}).get("mae"),
    )
    logger.info(
        "  Defense MAE: %s",
        new_metrics.get("defense_model", {}).get("mae"),
    )
    logger.info("=" * 60)

    # Improvement
    old_mae = old_metrics.get("mae")
    new_mae = new_metrics.get("mae")
    if old_mae and new_mae:
        delta = old_mae - new_mae
        pct = (delta / old_mae) * 100
        logger.info(
            "MAE improvement: %.4f -> %.4f (%.3f better, %.1f%%)",
            old_mae, new_mae, delta, pct,
        )

    # Show odds feature importance
    for label, key in [("Forward", "forward_model"), ("Defense", "defense_model")]:
        feats = new_metrics.get(key, {}).get("top_features", {})
        odds_feats = {
            k: v for k, v in feats.items()
            if k in ("game_total", "implied_team_total", "sog_prop_line")
        }
        if odds_feats:
            logger.info("%s odds feature importance: %s", label, odds_feats)

    # Show threshold accuracy comparison
    for label, key in [("Forward", "forward_model"), ("Defense", "defense_model")]:
        old_thresh = old_metrics.get(key, {}).get("threshold_accuracy", {})
        new_thresh = new_metrics.get(key, {}).get("threshold_accuracy", {})
        if old_thresh and new_thresh:
            logger.info(
                "%s threshold accuracy - old: %s, new: %s",
                label, old_thresh, new_thresh,
            )

    # Show calibration comparison
    for label, key in [("Forward", "forward_model"), ("Defense", "defense_model")]:
        new_cal = new_metrics.get(key, {}).get("calibration", {})
        if new_cal:
            logger.info("%s calibration: %s", label, new_cal)

    return old_metrics, new_metrics


def show_odds_coverage():
    """Show how much odds data we have in the training vs holdout periods."""
    conn = nhl_odds_collector.get_db()

    # Total odds events
    total = conn.execute(
        "SELECT COUNT(*) as n FROM nhl_odds_events"
    ).fetchone()["n"]

    # By period
    training = conn.execute(
        "SELECT COUNT(*) as n FROM nhl_odds_events WHERE game_date < '2026-02-27'"
    ).fetchone()["n"]
    holdout = conn.execute(
        "SELECT COUNT(*) as n FROM nhl_odds_events WHERE game_date >= '2026-02-27'"
    ).fetchone()["n"]

    # Player props coverage
    total_props = conn.execute(
        "SELECT COUNT(DISTINCT player_name || game_date) as n FROM nhl_player_props"
    ).fetchone()["n"]
    prop_events = conn.execute(
        "SELECT COUNT(DISTINCT event_id) as n FROM nhl_player_props"
    ).fetchone()["n"]

    conn.close()

    logger.info("Odds data coverage:")
    logger.info("  Total events with odds: %d", total)
    logger.info("  Training period (< Feb 27): %d events", training)
    logger.info("  Holdout period (>= Feb 27): %d events", holdout)
    logger.info(
        "  Player SOG props: %d unique player-dates across %d events",
        total_props, prop_events,
    )


if __name__ == "__main__":
    remaining, used = check_credits()
    logger.info("API credits: %d remaining, %d used", remaining, used)

    # Step 1: Backfill historical odds
    logger.info("=" * 60)
    logger.info("STEP 1: Backfilling odds from %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)
    n_events, n_props = backfill()

    # Step 2: Show coverage
    logger.info("")
    show_odds_coverage()

    # Step 3: Retrain and evaluate
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Retraining model and evaluating holdout")
    logger.info("=" * 60)
    old_metrics, new_metrics = retrain_and_evaluate()

    # Final credit check
    remaining, used = check_credits()
    logger.info("")
    logger.info("Final API credits: %d remaining, %d used", remaining, used)
