"""How the replacement ranks shape the board:  python compare.py  (needs data/ from vorp.py fetch)

For each set of ranks: each position's share of the top-12 places at the 26 season ends, the final top 12's make-up,
and the final top 12. The README's table comes from this.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vorp as VP                                     # noqa: E402

SCHEMES = [("first cut", dict(QB=13, RB=36, WR=36, TE=13)),
           ("", dict(QB=20, RB=36, WR=36, TE=14, K=13)),
           ("", dict(QB=20, RB=30, WR=42, TE=14, K=13)),
           ("used", dict(QB=16, RB=30, WR=42, TE=14, K=13)),
           ("", dict(QB=14, RB=30, WR=42, TE=14, K=13))]


def board(sea, R, n=12):
    s = sea[sea.pos.isin(R)].copy()
    s["rk"] = s.groupby(["season", "pos"]).points.rank(ascending=False, method="first")
    rep = s[s.rk == s.pos.map(R)].set_index(["season", "pos"]).points
    s["vorp"] = (s.points - [rep[y, p] for y, p in zip(s.season, s.pos)]).clip(lower=0)
    pos = s.sort_values("season").groupby("player_id").pos.last()
    share = {p: 0 for p in R}
    for y in range(VP.FIRST, VP.LAST + 1):
        for pid in s[s.season <= y].groupby("player_id").vorp.sum().nlargest(n).index:
            share[pos[pid]] += 1
    car = s.groupby("player_id").agg(player=("player", "last"), vorp=("vorp", "sum")).join(pos).nlargest(n, "vorp")
    return share, car


def main():
    wk = VP.load()
    sea = wk.groupby(["player_id", "season"]).agg(player=("player_display_name", "last"), pos=("pos", "last"),
                                                  points=("pts", "sum")).reset_index()
    for label, R in SCHEMES:
        share, car = board(sea, R)
        tot = sum(share.values())
        mix = car.pos.value_counts()
        print(" ".join(f"{p}{r}" for p, r in R.items()) + (f" ({label})" if label else ""))
        print("   share of the board:", "  ".join(f"{p} {100 * share[p] / tot:.0f}%" for p in R))
        print("   final top 12:", ", ".join(f"{v} {p}" for p, v in mix.items()))
        print("   " + "; ".join(f"{a} ({p}) {v:,.0f}" for a, p, v in zip(car.player, car.pos, car.vorp)))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
