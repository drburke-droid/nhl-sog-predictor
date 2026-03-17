"""
One-time script to collect 2024 MLB season data and train the model.
Run this locally before the 2025 season starts.

Usage: python train_mlb_2024.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def progress(msg):
    logger.info(msg)


def main():
    import mlb_data_collector
    import mlb_model

    logger.info("=" * 60)
    logger.info("MLB 2024 Season Data Collection & Model Training")
    logger.info("=" * 60)

    # Step 1: Collect game logs + team stats for 2024
    logger.info("Step 1: Collecting 2024 season data...")
    mlb_data_collector.collect_season_data(
        progress_callback=progress,
        season=2024,
    )

    # Check what we got
    conn = mlb_data_collector.get_db()
    game_count = conn.execute("SELECT COUNT(*) FROM mlb_games").fetchone()[0]
    pgs_count = conn.execute("SELECT COUNT(*) FROM mlb_pitcher_game_stats").fetchone()[0]
    arsenal_count = conn.execute("SELECT COUNT(*) FROM mlb_pitcher_arsenal").fetchone()[0]
    batter_count = conn.execute("SELECT COUNT(*) FROM mlb_batter_stats").fetchone()[0]
    bvp_count = conn.execute("SELECT COUNT(*) FROM mlb_batter_vs_pitch").fetchone()[0]
    tto_count = conn.execute("SELECT COUNT(*) FROM mlb_pitcher_tto").fetchone()[0]
    statcast_count = conn.execute("SELECT COUNT(*) FROM mlb_statcast_pitcher").fetchone()[0]
    conn.close()

    logger.info("Data summary:")
    logger.info("  Games: %d", game_count)
    logger.info("  Pitcher game stats: %d", pgs_count)
    logger.info("  Pitcher arsenal profiles: %d", arsenal_count)
    logger.info("  Batter stats: %d", batter_count)
    logger.info("  Batter vs pitch type: %d", bvp_count)
    logger.info("  Pitcher TTO profiles: %d", tto_count)
    logger.info("  Statcast pitcher-game rows: %d", statcast_count)

    if pgs_count < 50:
        logger.error("Not enough pitcher game stats to train. Exiting.")
        return

    # Step 2: Train model
    logger.info("Step 2: Training MLB model...")
    metrics = mlb_model.train_model()
    logger.info("Model metrics: %s", metrics)

    logger.info("=" * 60)
    logger.info("Done! Model saved to saved_model_mlb/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
