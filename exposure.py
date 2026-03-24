"""
Correlation-Aware Exposure Controls — Sprint 3.

Prevents hidden concentration risk by capping exposure per
game, team, player, and day. Real portfolio risk is higher
when bets are correlated (same game, same team direction).
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default exposure caps (fraction of bankroll)
DEFAULT_CAPS = {
    "max_game_exposure": 0.15,      # 15% max per game
    "max_team_exposure": 0.10,      # 10% max per team
    "max_player_exposure": 0.05,    # 5% max per player
    "max_daily_exposure": 0.30,     # 30% max per day
}


def apply_exposure_caps(bets, bankroll, caps=None):
    """Apply exposure caps to a list of bet recommendations.

    Args:
        bets: list of dicts with keys: player, team, game (away@home),
              date, wager, side, line
        bankroll: current bankroll
        caps: dict of cap names -> fractions (or use DEFAULT_CAPS)

    Returns (accepted, reduced, rejected) lists of bets.
    """
    if caps is None:
        caps = DEFAULT_CAPS

    max_game = bankroll * caps.get("max_game_exposure", 0.15)
    max_team = bankroll * caps.get("max_team_exposure", 0.10)
    max_player = bankroll * caps.get("max_player_exposure", 0.05)
    max_daily = bankroll * caps.get("max_daily_exposure", 0.30)

    # Track running exposure
    game_exposure = defaultdict(float)
    team_exposure = defaultdict(float)
    player_exposure = defaultdict(float)
    daily_exposure = defaultdict(float)

    accepted = []
    reduced = []
    rejected = []

    # Sort by edge descending — best bets get first allocation
    sorted_bets = sorted(bets, key=lambda b: b.get("edge", 0), reverse=True)

    for bet in sorted_bets:
        wager = bet.get("wager", 0)
        if wager <= 0:
            continue

        game = bet.get("game", "unknown")
        team = bet.get("team", "unknown")
        player = bet.get("player", "unknown")
        date = bet.get("date", "unknown")

        # Compute remaining capacity under each cap
        remaining_game = max(0, max_game - game_exposure[game])
        remaining_team = max(0, max_team - team_exposure[team])
        remaining_player = max(0, max_player - player_exposure[player])
        remaining_daily = max(0, max_daily - daily_exposure[date])

        max_allowed = min(remaining_game, remaining_team,
                         remaining_player, remaining_daily)

        if max_allowed < 1.0:
            rejected.append({**bet, "reason": "exposure cap reached"})
            continue

        if wager > max_allowed:
            original = wager
            wager = round(max_allowed, 2)
            reduced.append({**bet, "wager": wager,
                           "original_wager": original,
                           "reason": f"reduced from ${original:.2f}"})
        else:
            accepted.append({**bet, "wager": wager})

        # Update running exposure
        game_exposure[game] += wager
        team_exposure[team] += wager
        player_exposure[player] += wager
        daily_exposure[date] += wager

    return accepted, reduced, rejected


def portfolio_exposure_report(bets, bankroll, caps=None):
    """Generate exposure report showing concentration risk."""
    if caps is None:
        caps = DEFAULT_CAPS

    accepted, reduced, rejected = apply_exposure_caps(bets, bankroll, caps)

    total_accepted = sum(b["wager"] for b in accepted)
    total_reduced = sum(b["wager"] for b in reduced)

    # Group by dimensions
    by_game = defaultdict(float)
    by_team = defaultdict(float)
    by_date = defaultdict(float)

    for b in accepted + reduced:
        by_game[b.get("game", "?")] += b["wager"]
        by_team[b.get("team", "?")] += b["wager"]
        by_date[b.get("date", "?")] += b["wager"]

    return {
        "bankroll": bankroll,
        "total_bets": len(bets),
        "accepted": len(accepted),
        "reduced": len(reduced),
        "rejected": len(rejected),
        "total_wagered": round(total_accepted + total_reduced, 2),
        "pct_of_bankroll": round((total_accepted + total_reduced) / max(bankroll, 1) * 100, 1),
        "max_game_concentration": round(max(by_game.values()) / max(bankroll, 1) * 100, 1) if by_game else 0,
        "max_team_concentration": round(max(by_team.values()) / max(bankroll, 1) * 100, 1) if by_team else 0,
        "max_daily_concentration": round(max(by_date.values()) / max(bankroll, 1) * 100, 1) if by_date else 0,
        "caps_used": caps,
    }


def print_exposure_report(report):
    """Print formatted exposure report."""
    print(f"\n{'=' * 60}")
    print("  EXPOSURE REPORT")
    print(f"{'=' * 60}")
    print(f"  Bankroll: ${report['bankroll']:.2f}")
    print(f"  Bets: {report['total_bets']} total -> "
          f"{report['accepted']} accepted, "
          f"{report['reduced']} reduced, "
          f"{report['rejected']} rejected")
    print(f"  Total wagered: ${report['total_wagered']:.2f} "
          f"({report['pct_of_bankroll']:.1f}% of bankroll)")
    print(f"  Max game concentration: {report['max_game_concentration']:.1f}%")
    print(f"  Max team concentration: {report['max_team_concentration']:.1f}%")
    print(f"  Max daily concentration: {report['max_daily_concentration']:.1f}%")
    print(f"{'=' * 60}")
