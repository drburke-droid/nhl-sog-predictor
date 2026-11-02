"""Exact long-run square probabilities for one Monopoly token (US board, current card decks).

A Markov chain over turns: the state is where a turn ends (40 squares, or in Jail after 0, 1 or 2
failed escape rolls). Within a turn the token rolls, moves, resolves cards and Go To Jail, and rolls
again on doubles (third double in a turn goes to jail). Cards are treated as a fresh random draw
(the usual approximation; the real deck cycles, so long-run frequencies are the same).

Two jail policies: 'leave' pays $50 at once and rolls normally; 'stay' rolls for doubles for up to
three turns (doubles move the token but give no extra roll; after the third miss it pays and moves).

Outputs, per policy:
  end_of_turn  probability the token is on each square at the end of a turn (Collins' convention;
               index 10 is just visiting, index 40 is in jail)
  landings     expected landings per turn on each square (what earns rent: doubles land twice)
Writes results/markov.json.
"""
import json
import os

import numpy as np

CHANCE_MOVES = ['boardwalk', 'go', 'illinois', 'stcharles', 'rr', 'rr', 'util', 'back3', 'jail', 'reading']
CC_MOVES = ['go', 'jail']
CHANCE_SQ = {7, 22, 36}
CC_SQ = {2, 17, 33}
JAIL = 40  # pseudo-square for "in jail"
DICE = [(a, b) for a in range(1, 7) for b in range(1, 7)]


def resolve(sq):
    """Distribution over where the token ends after landing on sq (cards, Go To Jail)."""
    if sq == 30:
        return {JAIL: 1.0}
    if sq in CHANCE_SQ:
        out = {sq: 6 / 16}
        for c in CHANCE_MOVES:
            if c == 'boardwalk': t = 39
            elif c == 'go': t = 0
            elif c == 'illinois': t = 24
            elif c == 'stcharles': t = 11
            elif c == 'reading': t = 5
            elif c == 'rr': t = {7: 15, 22: 25, 36: 5}[sq]
            elif c == 'util': t = {7: 12, 22: 28, 36: 12}[sq]
            elif c == 'jail': t = JAIL
            elif c == 'back3':
                sub = resolve(sq - 3)  # from 36 this is Community Chest 33
                for k, v in sub.items():
                    out[k] = out.get(k, 0) + v / 16
                continue
            out[t] = out.get(t, 0) + 1 / 16
        return out
    if sq in CC_SQ:
        return {sq: 14 / 16, 0: 1 / 16, JAIL: 1 / 16}
    return {sq: 1.0}


def landing_squares(sq):
    """Squares the token lands on (each counts as a landing) when it arrives at sq."""
    # landing on sq itself, plus the square a card sends it to
    return sq


def turn_from(start, doubles_so_far):
    """For the rest of a turn starting at `start` with `doubles_so_far` doubles already rolled:
    (end-of-turn distribution, expected landings, expected resting square of each roll), each [41]."""
    end = np.zeros(41)
    land = np.zeros(41)
    rest = np.zeros(41)
    for a, b in DICE:
        p = 1 / 36
        if a == b and doubles_so_far == 2:
            end[JAIL] += p
            rest[JAIL] += p
            continue
        tgt = (start + a + b) % 40
        land[tgt] += p
        for t, q in resolve(tgt).items():
            rest[t] += p * q
            if t != tgt:
                land[t] += p * q  # card destination is also a landing (jail counted as 40)
            if t == JAIL:
                end[JAIL] += p * q
            elif a == b:
                e2, l2, r2 = turn_from(t, doubles_so_far + 1)
                end += p * q * e2
                land += p * q * l2
                rest += p * q * r2
            else:
                end[t] += p * q
    return end, land, rest


def build_chain(policy):
    # states 0..39 on a square (just visiting at 10), 40/41/42 in jail after 0/1/2 failed attempts
    n = 43
    T = np.zeros((n, n))
    L = np.zeros((n, 41))  # expected landings during a turn that starts in each state
    R = np.zeros((n, 41))  # expected resting square of each roll during that turn
    cache = {}
    for s in range(40):
        cache[s] = turn_from(s, 0)
    def place(row, end, scale=1.0, jail_state=40):
        for t in range(40):
            T[row, t] += scale * end[t]
        T[row, jail_state] += scale * end[JAIL]
    for s in range(40):
        e, l, r = cache[s]
        place(s, e)
        L[s] += l
        R[s] += r
    for k in range(3):
        row = 40 + k
        if policy == 'leave':
            e, l, r = cache[10]  # pay, then a normal turn from Jail
            place(row, e)
            L[row] += l
            R[row] += r
            continue
        for a, b in DICE:
            p = 1 / 36
            if a == b or k == 2:
                tgt = (10 + a + b) % 40
                L[row, tgt] += p
                for t, q in resolve(tgt).items():
                    R[row, t] += p * q
                    if t != tgt:
                        L[row, t] += p * q
                    if t == JAIL:
                        T[row, 40] += p * q
                    else:
                        T[row, t] += p * q
            else:
                T[row, 40 + k + 1] += p
                R[row, JAIL] += p
    return T, L, R


def stationary(T):
    w, v = np.linalg.eig(T.T)
    i = np.argmin(abs(w - 1))
    pi = np.real(v[:, i])
    return pi / pi.sum()


def main():
    out = {}
    for policy in ('leave', 'stay'):
        T, L, R = build_chain(policy)
        assert np.allclose(T.sum(1), 1), T.sum(1)
        pi = stationary(T)
        end = np.zeros(41)
        end[:40] = pi[:40]
        end[40] = pi[40:].sum()
        land = pi @ L
        rest = pi @ R
        per_roll = rest / rest.sum()
        out[policy] = {'end_of_turn': end.tolist(), 'landings_per_turn': land.tolist(),
                       'per_roll': per_roll.tolist(), 'rolls_per_turn': float(rest.sum())}
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    with open(os.path.join(here, 'results', 'markov.json'), 'w') as f:
        json.dump(out, f, indent=1)
    names = json.load(open(os.path.join(here, 'board.json')))['names'] if os.path.exists(os.path.join(here, 'board.json')) else [str(i) for i in range(40)]
    for policy in ('leave', 'stay'):
        e = out[policy]['per_roll']
        print(policy, 'per roll: jail(in) %.2f%% visiting %.2f%% combined %.2f%%, rolls/turn %.4f' % (100 * e[40], 100 * e[10], 100 * (e[40] + e[10]), out[policy]['rolls_per_turn']))
        top = sorted(range(40), key=lambda i: -e[i])[:8]
        print('  top per roll:', ', '.join('%s %.2f%%' % (names[i], 100 * e[i]) for i in top))


if __name__ == '__main__':
    main()
