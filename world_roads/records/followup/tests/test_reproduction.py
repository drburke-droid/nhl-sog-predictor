"""Automated checks against the original scores and the handoff's exploratory numbers.

    python tests/test_reproduction.py      # from followup/; exits non-zero on any failure

1. Spec() on the original inputs reproduces every pipeline score (windows 7, 10, 15, 20) to 1e-9.
2. The original athlete-season ranking (greatest_seasons.csv, top 200) is reproduced to 1e-3.
3. The handoff's exploratory script, run unchanged, still gives its documented z-scores.
4. Invariances: rescaling a category's units leaves z unchanged; reordering tied leaders under ties='all'
   leaves the scored set unchanged.
"""
import csv, math, os, sys, random, importlib

F = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(F, "analysis"))
import gsmodel as G                                            # noqa: E402

fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main():
    D = G.Data("original")
    P = defaultdict_list()
    for r in csv.DictReader(open(os.path.join(F, "data", "original", "pipeline_era.csv"), encoding="utf-8")):
        P[int(r["window"])][(r["category"], int(r["season"]))] = float(r["z"])
    for w in (7, 10, 15, 20):
        mine = {(r["cat"], r["year"]): r["z"] for r in G.score(D, G.Spec(window=w))}
        both = set(mine) & set(P[w])
        diff = max(abs(mine[k] - P[w][k]) for k in both)
        check(set(mine) == set(P[w]) and diff < 1e-9, f"window {w}: {len(mine)} scores, keys equal, max |dz| {diff:.1e}")
    ranked = G.athlete_seasons(G.score(D, G.Spec()))
    pub = list(csv.DictReader(open(os.path.join(F, "data", "original", "greatest_seasons.csv"), encoding="utf-8")))
    bad = [g for g in pub if abs(float(g["z"]) - ranked[int(g["rank"]) - 1]["z"]) > 1e-3]
    check(not bad, f"published greatest_seasons.csv: {len(pub)} rows, {len(bad)} mismatched")

    # 3. the handoff script, unchanged
    sys.path.insert(0, os.path.join(F, "handoff", "exploration"))
    m = importlib.import_module("era_field_adjustment")
    expected = {"raw": {"Wilt Chamberlain": 7.731371, "Wayne Gretzky": 7.499599, "Joe DiMaggio": 7.197192},
                "era": {"Wilt Chamberlain": 4.4778, "Wayne Gretzky": 5.1947},
                "era_field": {"Wilt Chamberlain": 6.3275, "Wayne Gretzky": 5.1693, "Babe Dye": 5.278}}
    for mode, ex in expected.items():
        _, _, d = m.get_results(mode, field_strength=1.0, window=15)
        for nm, z in ex.items():
            r = next((v for v in d if v["name"] == nm), None)
            check(r is not None and abs(r["z"] - z) < 0.002, f"handoff {mode} {nm}: {r and round(r['z'], 4)} vs {z}")

    # 4. invariances
    D2 = G.Data("original")
    for y in D2.seasons["nba_ppg"]:
        for e in D2.seasons["nba_ppg"][y]:
            e.perf *= 2.54
    a = {(r["cat"], r["year"]): r["z"] for r in G.score(D, G.Spec(), cats=["nba_ppg"])}
    b = {(r["cat"], r["year"]): r["z"] for r in G.score(D2, G.Spec(), cats=["nba_ppg"])}
    check(max(abs(a[k] - b[k]) for k in a) < 1e-9, "unit change leaves z unchanged (nba_ppg x 2.54)")
    D3 = G.Data("original")
    random.seed(1)
    for c in D3.seasons:
        for y, lst in D3.seasons[c].items():
            top = [e for e in lst if e.perf == lst[0].perf]
            random.shuffle(top)
            lst[:len(top)] = top
    s1 = {(r["cat"], r["year"], r["id"]) for r in G.score(D, G.Spec(ties="all"))}
    s3 = {(r["cat"], r["year"], r["id"]) for r in G.score(D3, G.Spec(ties="all"))}
    check(s1 == s3, f"ties='all' is invariant to the order of tied leaders ({len(s1)} scored)")
    print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILED"))
    sys.exit(1 if fails else 0)


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(dict)


if __name__ == "__main__":
    main()
