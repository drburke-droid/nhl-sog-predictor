"""Rank sportsbook sharpness from actual 2025-26 NHL results."""

import sqlite3
import unicodedata
import numpy as np
from collections import defaultdict

def match_key(name):
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.strip().lower().replace(".", "")
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0], " ".join(parts[1:]))
    return ("", name)

def american_to_prob(odds):
    if odds > 0: return 100.0 / (odds + 100.0)
    elif odds < 0: return -odds / (-odds + 100.0)
    return 0.5

conn = sqlite3.connect("nhl_data.db")
conn.row_factory = sqlite3.Row

# Actual SOG per player-date
actuals = {}
for r in conn.execute("""
    SELECT pgs.player_name, g.date, pgs.shots
    FROM player_game_stats pgs JOIN games g ON pgs.game_id = g.game_id
    WHERE g.status = 'FINAL'
""").fetchall():
    k = match_key(r["player_name"])
    actuals[(k[0], k[1], r["date"])] = r["shots"]

# Pair over/under props per book-player-date-line
props = conn.execute("SELECT * FROM nhl_player_props").fetchall()
paired = defaultdict(dict)
for r in props:
    k = (r["bookmaker"], r["player_name"], r["game_date"], r["line"])
    paired[k][r["over_under"]] = r["price"]

# Evaluate each book on player props
book_stats = defaultdict(lambda: {
    "n": 0, "vig_sum": 0, "brier_sum": 0, "logloss_sum": 0,
})

for (book, pname, gdate, line), prices in paired.items():
    if "Over" not in prices or "Under" not in prices:
        continue
    pk = match_key(pname)
    actual = actuals.get((pk[0], pk[1], gdate))
    if actual is None:
        continue

    imp_o = american_to_prob(prices["Over"])
    imp_u = american_to_prob(prices["Under"])
    total_imp = imp_o + imp_u
    if total_imp <= 0:
        continue

    vig = (total_imp - 1.0) * 100
    fair_o = imp_o / total_imp
    went_over = 1 if actual > line else 0
    brier = (fair_o - went_over) ** 2
    eps = 1e-6
    p = max(min(fair_o, 1 - eps), eps)
    ll = -(went_over * np.log(p) + (1 - went_over) * np.log(1 - p))

    s = book_stats[book]
    s["n"] += 1
    s["vig_sum"] += vig
    s["brier_sum"] += brier
    s["logloss_sum"] += ll

print("=" * 100)
print("  NHL SPORTSBOOK SHARPNESS RANKING — Player SOG Props (2025-26)")
print("=" * 100)
print()
print("  Lower Brier = better calibrated (most important)")
print("  Lower Vig = less juice taken from bettors")
print("  Lower LogLoss = better at assigning probabilities")
print()
print(f"  {'Rank':>4s}  {'Bookmaker':22s} {'Lines':>6s} {'Brier':>8s} {'LogLoss':>8s} {'Avg Vig':>8s}")
print("  " + "-" * 62)

results = []
for book, s in book_stats.items():
    if s["n"] < 50:
        continue
    n = s["n"]
    results.append((
        book, n,
        s["brier_sum"] / n,
        s["logloss_sum"] / n,
        s["vig_sum"] / n,
    ))

results.sort(key=lambda x: x[2])  # sort by Brier (best calibration)
for rank, (book, n, brier, ll, vig) in enumerate(results, 1):
    tag = ""
    if rank <= 3:
        tag = " <<<< SHARPEST"
    elif vig > 7:
        tag = " (soft)"
    print(f"  {rank:4d}  {book:22s} {n:6d} {brier:8.4f} {ll:8.4f} {vig:7.2f}%{tag}")

# Game-level odds (ML + totals)
print()
print("=" * 100)
print("  NHL SPORTSBOOK SHARPNESS RANKING — Game Odds (Moneyline + Totals)")
print("=" * 100)

game_results = {}
for r in conn.execute("SELECT date, home_team, away_team, home_score, away_score FROM games WHERE status = 'FINAL'").fetchall():
    # Key by (date, home_abbrev) since game_odds uses home_abbrev
    game_results[(r["date"], r["home_team"])] = {
        "total": (r["home_score"] or 0) + (r["away_score"] or 0),
        "home_win": 1 if (r["home_score"] or 0) > (r["away_score"] or 0) else 0,
    }
    # Games table uses abbreviations already

game_rows = conn.execute("SELECT * FROM nhl_game_odds").fetchall()

ml_pairs = defaultdict(dict)
total_pairs = defaultdict(dict)
for r in game_rows:
    bk = r["bookmaker"]
    key = (bk, r["game_date"], r["home_abbrev"])
    if r["market"] == "h2h":
        if r["outcome_name"] == r["home_team"]:
            ml_pairs[key]["home"] = r["outcome_price"]
        else:
            ml_pairs[key]["away"] = r["outcome_price"]
        ml_pairs[key]["date"] = r["game_date"]
        ml_pairs[key]["home_abbrev"] = r["home_abbrev"]
    elif r["market"] == "totals":
        if r["outcome_name"] == "Over":
            total_pairs[key]["over"] = r["outcome_price"]
        else:
            total_pairs[key]["under"] = r["outcome_price"]
        total_pairs[key]["line"] = r["outcome_point"]
        total_pairs[key]["date"] = r["game_date"]
        total_pairs[key]["home_abbrev"] = r["home_abbrev"]

gbook = defaultdict(lambda: {"ml_b": 0, "ml_n": 0, "tot_b": 0, "tot_n": 0, "vig": 0, "vig_n": 0})

for key, odds in ml_pairs.items():
    bk = key[0]
    if "home" not in odds or "away" not in odds:
        continue
    gk = (odds.get("date"), odds.get("home_abbrev"))
    res = game_results.get(gk)
    if not res:
        continue
    imp_h = american_to_prob(odds["home"])
    imp_a = american_to_prob(odds["away"])
    ti = imp_h + imp_a
    if ti <= 0:
        continue
    fair_h = imp_h / ti
    gbook[bk]["ml_b"] += (fair_h - res["home_win"]) ** 2
    gbook[bk]["ml_n"] += 1
    gbook[bk]["vig"] += (ti - 1) * 100
    gbook[bk]["vig_n"] += 1

for key, odds in total_pairs.items():
    bk = key[0]
    if "over" not in odds or "under" not in odds or "line" not in odds:
        continue
    gk = (odds.get("date"), odds.get("home_abbrev"))
    res = game_results.get(gk)
    if not res:
        continue
    imp_o = american_to_prob(odds["over"])
    imp_u = american_to_prob(odds["under"])
    ti = imp_o + imp_u
    if ti <= 0:
        continue
    fair_o = imp_o / ti
    went_over = 1 if res["total"] > odds["line"] else 0
    gbook[bk]["tot_b"] += (fair_o - went_over) ** 2
    gbook[bk]["tot_n"] += 1

print()
print(f"  {'Rank':>4s}  {'Bookmaker':22s} {'ML Games':>8s} {'ML Brier':>9s} {'Tot Games':>9s} {'Tot Brier':>10s} {'Avg Vig':>8s} {'Combined':>9s}")
print("  " + "-" * 90)

glist = []
for bk, s in gbook.items():
    if s["ml_n"] < 20:
        continue
    ml_b = s["ml_b"] / s["ml_n"]
    tot_b = s["tot_b"] / s["tot_n"] if s["tot_n"] > 0 else 0.30
    vig = s["vig"] / s["vig_n"] if s["vig_n"] > 0 else 0
    comb = 0.5 * ml_b + 0.5 * tot_b
    glist.append((bk, s["ml_n"], ml_b, s["tot_n"], tot_b, vig, comb))

glist.sort(key=lambda x: x[6])
for rank, (bk, ml_n, ml_b, tot_n, tot_b, vig, comb) in enumerate(glist, 1):
    tag = ""
    if rank <= 3:
        tag = " <<<< SHARPEST"
    print(f"  {rank:4d}  {bk:22s} {ml_n:8d} {ml_b:9.4f} {tot_n:9d} {tot_b:10.4f} {vig:7.2f}% {comb:9.4f}{tag}")

conn.close()
