"""How the replacement ranks shape the board:  python compare.py  (needs data/ from vorp.py fetch)

For each set of ranks: each position's share of the places on the all-time board (the 12 best single seasons so far)
at the 46 season ends, the final board's make-up, and the final board, with VORP counted as vorp.py counts it (by the
game). The README's table comes from this.
"""
import os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vorp as VP                                     # noqa: E402

SCHEMES = [("first cut", dict(QB=13, RB=36, WR=36, TE=13)),
           ("", dict(QB=20, RB=36, WR=36, TE=14)),
           ("used", dict(QB=16, RB=30, WR=42, TE=14)),
           ("", dict(QB=16, RB=24, WR=48, TE=14)),
           ("", dict(QB=24, RB=24, WR=48, TE=20))]


def main(n=12):
    wk = VP.load()
    for label, R in SCHEMES:
        s, _ = VP.seasons(wk, R)
        share = Counter()
        for y in range(VP.FIRST, VP.LAST + 1):
            share.update(s[s.season <= y].nlargest(n, "vorp").pos)
        top = s.nlargest(n, "vorp")
        tot = sum(share.values())
        print(" ".join(f"{p}{r}" for p, r in R.items()) + (f" ({label})" if label else ""))
        print("   share of the board:", "  ".join(f"{p} {100 * share[p] / tot:.0f}%" for p in R))
        print("   final top 12:", ", ".join(f"{v} {p}" for p, v in top.pos.value_counts().items()))
        print("   " + "; ".join(f"{a} {y} ({p}) {v:,.0f}" for a, y, p, v in zip(top.player, top.season, top.pos, top.vorp)))


if __name__ == "__main__":
    main()
