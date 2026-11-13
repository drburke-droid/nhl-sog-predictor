"""The textbook model: every peg is a fair coin flip.

A chip released over slot k meets 12 pegs (rows 1..12; row 0 only frames the nine drop gaps). Each peg sends
it half a slot left or right. At the odd rows the side walls carry a tooth where slot 1's (slot 9's) peg
would be, so a chip there can only go inward - the wall rule used by Hastings (1997) and Biesterfeld (2001).

Positions are tracked in half-slot units: u = 2x, x = 1..9 slot centres.
"""
import numpy as np
from board import SLOT_VALUES


def kernel(p_right=0.5):
    """9x9 matrix K[start-1, end-1] = P(lands in slot `end` | dropped over slot `start`)."""
    K = np.zeros((9, 9))
    for s in range(1, 10):
        dist = {2 * s: 1.0}
        for row in range(1, 13):
            new = {}
            for u, p in dist.items():
                x = u / 2
                if row % 2 == 1 and x == 1:      # left wall tooth
                    moves = [(u + 1, 1.0)]
                elif row % 2 == 1 and x == 9:    # right wall tooth
                    moves = [(u - 1, 1.0)]
                else:
                    moves = [(u - 1, 1 - p_right), (u + 1, p_right)]
                for v, q in moves:
                    new[v] = new.get(v, 0.0) + p * q
            dist = new
        for u, p in dist.items():
            K[s - 1, u // 2 - 1] += p
    return K


def ev_table(K, values=SLOT_VALUES):
    ev = K @ values
    sd = np.sqrt(K @ values ** 2 - ev ** 2)
    return ev, sd


if __name__ == "__main__":
    K = kernel()
    ev, sd = ev_table(K)
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(K)
    for s in range(9):
        print(f"drop {s+1}: EV ${ev[s]:8.2f}  SD ${sd[s]:8.2f}  P($10k) {K[s,4]:.4f}")
