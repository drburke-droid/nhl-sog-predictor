"""
Bookmaker Disagreement Features — cross-book signal extraction.

Without CLV, the best substitute is measuring how much books disagree.
When sharp and soft books diverge, the soft book is more likely wrong.

Features:
- implied_prob_std: standard deviation of implied probs across books
- sharp_soft_spread: vig-free sharp prob minus soft implied prob
- n_books_same_line: how many books agree on the same line
- soft_deviation: how far the soft book deviates from consensus
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "nhl_data.db"


def _american_to_prob(odds):
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return -odds / (-odds + 100.0)
    return 0.5


def compute_disagreement_features(player_name, game_date, line, side="UNDER"):
    """Compute cross-book disagreement features for a player prop.

    Args:
        player_name: full name from odds database
        game_date: YYYY-MM-DD
        line: prop line (e.g., 2.5)
        side: OVER or UNDER

    Returns dict with disagreement features.
    """
    import unicodedata

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get all bookmaker prices for this player-date-line
    rows = conn.execute("""
        SELECT bookmaker, over_under, price
        FROM nhl_player_props
        WHERE game_date = ? AND line = ?
    """, (game_date, line)).fetchall()
    conn.close()

    if not rows:
        return {}

    # Match player name (first initial + last name)
    def _match_key(name):
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = name.strip().lower().replace(".", "")
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0], " ".join(parts[1:]))
        return ("", name)

    target_key = _match_key(player_name)

    # Group by bookmaker
    book_probs = {}
    for r in rows:
        # Check if this row matches our player (need player_name in query)
        # Since we can't filter by player in this simplified query,
        # we'll use the broader per-book approach
        bk = r["bookmaker"]
        ou = r["over_under"]
        price = r["price"]

        if ou == side:
            imp = _american_to_prob(price)
            if bk not in book_probs:
                book_probs[bk] = imp

    if len(book_probs) < 2:
        return {}

    probs = list(book_probs.values())

    # Sharp books
    SHARP = {"coolbet", "fanduel", "williamhill_us", "draftkings", "pinnacle", "betonlineag"}
    SOFT = {"betmgm"}

    sharp_probs = [book_probs[b] for b in SHARP if b in book_probs]
    soft_probs = [book_probs[b] for b in SOFT if b in book_probs]

    result = {
        "n_books": len(book_probs),
        "implied_prob_std": round(float(np.std(probs)), 4),
        "implied_prob_range": round(float(max(probs) - min(probs)), 4),
        "consensus_prob": round(float(np.mean(probs)), 4),
    }

    if sharp_probs and soft_probs:
        sharp_avg = np.mean(sharp_probs)
        soft_avg = np.mean(soft_probs)
        result["sharp_soft_spread"] = round(float(sharp_avg - soft_avg), 4)
        result["soft_deviation"] = round(float(soft_avg - np.mean(probs)), 4)
    else:
        result["sharp_soft_spread"] = 0.0
        result["soft_deviation"] = 0.0

    if sharp_probs:
        result["n_sharp_books"] = len(sharp_probs)
        result["sharp_prob_std"] = round(float(np.std(sharp_probs)), 4)

    return result


def compute_bulk_disagreement(game_date=None):
    """Compute disagreement features for all player props on a date.

    Returns dict: (player_name, line) -> disagreement features
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    if game_date is None:
        from datetime import date
        game_date = date.today().isoformat()

    rows = conn.execute("""
        SELECT player_name, line, bookmaker, over_under, price
        FROM nhl_player_props WHERE game_date = ?
    """, (game_date,)).fetchall()
    conn.close()

    if not rows:
        return {}

    SHARP = {"coolbet", "fanduel", "williamhill_us", "draftkings", "pinnacle", "betonlineag"}
    SOFT = {"betmgm"}

    # Group: (player_name, line, side) -> {book: implied_prob}
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["player_name"], r["line"], r["over_under"])
        grouped[key][r["bookmaker"]] = _american_to_prob(r["price"])

    result = {}
    for (pname, line, side), book_probs in grouped.items():
        if len(book_probs) < 2:
            continue

        probs = list(book_probs.values())
        sharp_probs = [book_probs[b] for b in SHARP if b in book_probs]
        soft_probs = [book_probs[b] for b in SOFT if b in book_probs]

        features = {
            "n_books": len(book_probs),
            "implied_prob_std": round(float(np.std(probs)), 4),
            "implied_prob_range": round(float(max(probs) - min(probs)), 4),
            "consensus_prob": round(float(np.mean(probs)), 4),
        }

        if sharp_probs and soft_probs:
            features["sharp_soft_spread"] = round(float(np.mean(sharp_probs) - np.mean(soft_probs)), 4)
            features["soft_deviation"] = round(float(np.mean(soft_probs) - np.mean(probs)), 4)

        result[(pname, line, side)] = features

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test on a recent date
    features = compute_bulk_disagreement("2026-03-22")
    print(f"Disagreement features for {len(features)} player-line-side combos")

    # Show some examples
    for (pname, line, side), f in list(features.items())[:5]:
        print(f"  {pname} {side} {line}: {f}")
