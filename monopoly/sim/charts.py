"""Draw the storyboard charts from results/summary.json (run analyze.py first).

    python3 charts.py      # writes results/charts/*.png
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = os.path.join(RES, 'charts')
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(RES, 'summary.json')))
NAMES = json.load(open(os.path.join(HERE, 'board.json')))['names']

GROUPS = ['brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue']
# the board's own colours: the colour *is* the entity here, and every mark is also labelled by name
BOARD = {'brown': '#8a4b2a', 'lightblue': '#9fd6f2', 'pink': '#d63c95', 'orange': '#f39a1e', 'red': '#e2262b',
         'yellow': '#f5d90a', 'green': '#1e9e55', 'darkblue': '#1f5fa8', 'railroads': '#3a3a3a', 'utilities': '#9a9a9a'}
GNAME = {'brown': 'Brown', 'lightblue': 'Light blue', 'pink': 'Pink', 'orange': 'Orange', 'red': 'Red', 'yellow': 'Yellow',
         'green': 'Green', 'darkblue': 'Dark blue', 'railroads': 'Railroads', 'utilities': 'Utilities'}
SHORT = {
    'everything': 'Yes-Man', 'nothing': 'Scrooge', 'railroads': 'Railroad Tycoon', 'utilities': 'Utilities only',
    'infrastructure': 'Railroads + utilities', 'brown': 'Brown only', 'lightblue': 'Light blue only', 'pink': 'Pink only',
    'orange': 'Orange only', 'red': 'Red only', 'yellow': 'Yellow only', 'green': 'Green only',
    'darkblue': 'Boardwalk Dreamer', 'prudent': 'Accountant ($300 cushion)', 'threehouses': 'Three-House Rule',
    'houselock': 'House Hoarder', 'jailbird': 'Jailbird', 'smartjail': 'Jail: out early, in late', 'stubborn': 'Never trades',
    'shark': 'Shark trader', 'allin': 'All-In', 'champion': 'Textbook expert',
}
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'savefig.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def group_of(i):
    g = {1: 'brown', 3: 'brown', 6: 'lightblue', 8: 'lightblue', 9: 'lightblue', 11: 'pink', 13: 'pink', 14: 'pink',
         16: 'orange', 18: 'orange', 19: 'orange', 21: 'red', 23: 'red', 24: 'red', 26: 'yellow', 27: 'yellow', 29: 'yellow',
         31: 'green', 32: 'green', 34: 'green', 37: 'darkblue', 39: 'darkblue', 5: 'railroads', 15: 'railroads',
         25: 'railroads', 35: 'railroads', 12: 'utilities', 28: 'utilities'}
    return g.get(i)


# ---------------------------------------------------------------- board ring heat map
def board_cells():
    """(row, col) of each square on an 11x11 board, Go at bottom right, play runs anticlockwise."""
    cells = []
    for i in range(40):
        if i <= 10:
            cells.append((10, 10 - i))
        elif i <= 20:
            cells.append((10 - (i - 10), 0))
        elif i <= 30:
            cells.append((0, i - 20))
        else:
            cells.append((i - 30, 10))
    return cells


SHORTSQ = ['GO', 'Mediterr.', 'Comm.\nChest', 'Baltic', 'Income\nTax', 'Reading\nRR', 'Oriental', 'Chance', 'Vermont',
           'Connect.', 'JAIL', 'St.\nCharles', 'Electric', 'States', 'Virginia', 'Penn.\nRR', 'St.\nJames', 'Comm.\nChest',
           'Tennessee', 'New\nYork', 'Free\nParking', 'Kentucky', 'Chance', 'Indiana', 'Illinois', 'B&O\nRR', 'Atlantic',
           'Ventnor', 'Water\nWorks', 'Marvin\nGardens', 'Go To\nJail', 'Pacific', 'N.\nCarolina', 'Comm.\nChest',
           'Penn.\nAve', 'Short\nLine', 'Chance', 'Park\nPlace', 'Luxury\nTax', 'Board-\nwalk']


def chart_board(policy='leave'):
    pr = S['markov'][policy]['per_roll']
    vals = list(pr[:40])
    vals[10] = pr[10] + pr[40]
    seq = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#86b6ef', '#2a78d6', '#104281'])
    lo, hi = 0.008, 0.032
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.set_xlim(0, 11); ax.set_ylim(11, 0); ax.axis('off')
    for i, (r, c) in enumerate(board_cells()):
        v = vals[i]
        col = seq(min(1.0, max(0.0, (v - lo) / (hi - lo))))
        ax.add_patch(plt.Rectangle((c + 0.03, r + 0.03), 0.94, 0.94, color=col, ec=SURF, lw=2))
        g = group_of(i)
        if g in GROUPS:
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.03), 0.94, 0.16, color=BOARD[g], ec=SURF, lw=1))
        dark = (v - lo) / (hi - lo) > 0.55
        tc = '#ffffff' if dark else INK
        ax.text(c + 0.5, r + 0.5, SHORTSQ[i], ha='center', va='center', fontsize=7, color=tc)
        ax.text(c + 0.5, r + 0.85, '%.1f%%' % (100 * v), ha='center', va='center', fontsize=8.5, color=tc, weight='bold')
    j = S['markov'][policy]['per_roll']
    ax.text(5.5, 4.2, 'Where a token ends up', ha='center', fontsize=20, weight='bold', color=INK)
    ax.text(5.5, 4.9, 'Chance of each square per roll, exact Markov chain\n(%s)' %
            ('player pays to leave Jail at once' if policy == 'leave' else 'player stays in Jail as long as allowed'),
            ha='center', fontsize=11, color=INK2)
    ax.text(5.5, 6.2, 'Jail: %.1f%% (%.1f%% locked up + %.1f%% just visiting)\nIllinois Ave: %.2f%%   Park Place: %.2f%%' %
            (100 * (j[40] + j[10]), 100 * j[40], 100 * j[10], 100 * j[24], 100 * j[37]), ha='center', fontsize=11, color=INK)
    save(fig, 'board_probability_%s.png' % policy)


# ---------------------------------------------------------------- leaderboards
def chart_leaderboard(world, title, fname, exclude=()):
    rows = [r for r in S['table'][world] if r['strategy'] not in exclude]
    rows = sorted(rows, key=lambda r: r['win'])
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(rows) + 1.6))
    y = range(len(rows))
    cols = [BLUE if r['strategy'] != 'everything' else MUTED for r in rows]
    ax.barh(list(y), [r['win'] for r in rows], color=cols, height=0.66)
    ax.errorbar([r['win'] for r in rows], list(y), xerr=[r['ci'] for r in rows], fmt='none', ecolor=INK2, lw=1, capsize=2)
    ax.axvline(25, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(25.4, len(rows) - 0.4, 'fair share 25%', color=INK2, fontsize=9, va='center')
    ax.set_yticks(list(y)); ax.set_yticklabels([SHORT.get(r['strategy'], r['strategy']) for r in rows])
    for i, r in enumerate(rows):
        ax.text(r['win'] + r['ci'] + 0.6, i, '%.1f%%' % r['win'], va='center', fontsize=9, color=INK)
    ax.set_xlabel('Win rate against three Yes-Men (buy everything, build hotels at once)')
    ax.set_xlim(0, max(r['win'] for r in rows) * 1.18 + 2)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, fname)


def chart_worlds(fname):
    worlds = [('official', 'Official rules'), ('notrade', 'No trading'), ('family', 'Family rules\n(no auctions,\nFree Parking cash)')]
    worlds = [w for w in worlds if w[0] in S['table']]
    keep = ['railroads', 'darkblue', 'orange', 'nothing', 'stubborn', 'threehouses', 'houselock', 'jailbird', 'prudent', 'allin', 'champion', 'shark']
    data = {w: {r['strategy']: r['win'] for r in S['table'][w]} for w, _ in worlds}
    fig, ax = plt.subplots(figsize=(10, 7.5))
    x = list(range(len(worlds)))

    def spread(items, gap):
        # nudge label positions apart so none sit closer than `gap`, keeping their order
        items = sorted(items, key=lambda it: it[0])
        pos = [y for y, _ in items]
        for _ in range(200):
            moved = False
            for i in range(1, len(pos)):
                if pos[i] - pos[i - 1] < gap:
                    d = (gap - (pos[i] - pos[i - 1])) / 2
                    pos[i - 1] -= d; pos[i] += d; moved = True
            if not moved:
                break
        return [(p, lab) for p, (_, lab) in zip(pos, items)]

    right, left = [], []
    for t in keep:
        ys = [data[w].get(t) for w, _ in worlds]
        if None in ys:
            continue
        ax.plot(x, ys, color=MUTED, lw=1.6, marker='o', ms=6)
        right.append((ys[-1], SHORT[t]))
        left.append((ys[0], '%.1f%%' % ys[0]))
    for y, lab in spread(right, 1.05):
        ax.text(x[-1] + 0.08, y, lab, va='center', fontsize=9, color=INK)
    for y, lab in spread(left, 1.05):
        ax.text(x[0] - 0.08, y, lab, va='center', ha='right', fontsize=8, color=INK2)
    ax.axhline(25, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in worlds])
    ax.set_xlim(-0.5, len(worlds) - 0.3)
    ax.set_ylabel('Win rate vs three Yes-Men')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Take away trading and everyone stalls; house rules barely matter', loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, fname)


def chart_skip(world, fname):
    rows = S['skip'][world]
    base = next(r['win'] for r in S['table'][world] if r['strategy'] == 'everything')
    rows = sorted(rows, key=lambda r: r['win'] - base)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labels, vals, cols = [], [], []
    for r in rows:
        g = r['strategy'].replace('skip_', '')
        g = 'railroads' if g == 'rr' else g
        labels.append('Skip ' + GNAME[g].lower()); vals.append(r['win'] - base); cols.append(BOARD[g])
    ax.barh(range(len(rows)), vals, color=cols, height=0.66, ec=INK2, lw=0.4)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
    for i, v in enumerate(vals):
        ax.text(v + (0.25 if v >= 0 else -0.25), i, '%+.1f' % v, va='center', ha='left' if v >= 0 else 'right', fontsize=9, color=INK)
    ax.set_xlabel('Change in win rate (percentage points) vs buying everything (%.1f%%)' % base)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('What it costs to never buy one group', loc='left', fontsize=14, weight='bold', color=INK)
    lo = min(vals); hi = max(vals)
    ax.set_xlim(lo - 2, max(hi, 0) + 2)
    save(fig, fname)


# ---------------------------------------------------------------- matrices
DIV = LinearSegmentedColormap.from_list('div', [ORANGE, '#f6c9b3', '#ececea', '#b7d3f6', BLUE])


def matrix(M, names, labels, title, fname, note):
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.62 * n + 2))
    import numpy as np
    A = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a != b and '%s>%s' % (a, b) in M:
                A[i, j] = 100 * M['%s>%s' % (a, b)]
    ax.imshow(A, cmap=DIV, norm=TwoSlopeNorm(50, 0, 100))
    for i in range(n):
        for j in range(n):
            if not np.isnan(A[i, j]):
                v = A[i, j]
                ax.text(j, i, '%.0f' % v, ha='center', va='center', fontsize=8.5, color='#ffffff' if abs(v - 50) > 33 else INK)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.02, -0.02, note, fontsize=9, color=INK2)
    save(fig, fname)


def chart_duels():
    for b, sub in (('b5', 'hotels as fast as possible'), ('b3', 'stop at three houses')):
        M = S.get('duel', {}).get(b)
        if not M:
            continue
        avg = {a: sum(M.get('%s>%s' % (a, c), 0) for c in GROUPS if c != a) / 7 for a in GROUPS}
        names = sorted(GROUPS, key=lambda g: -avg[g])
        matrix(M, names, [GNAME[g] for g in names], 'Colour-set duels: row beats column (%%), %s' % sub,
               'duel_matrix_%s.png' % b, 'Two players, $1,500 each, one colour set each, nothing else for sale.')


def chart_h2h(world):
    M = S.get('h2h', {}).get(world)
    if not M:
        return
    names = sorted({k.split('>')[0] for k in M})
    score = {a: sum(M.get('%s>%s' % (a, b), 0) for b in names if b != a) for a in names}
    names.sort(key=lambda a: -score[a])
    matrix(M, names, [SHORT.get(a, a) for a in names], 'Two-player head to head, %s rules: row beats column (%%)' % world,
           'h2h_matrix_%s.png' % world, 'Draws (no winner after 1,000 rounds) make a row and its mirror add to less than 100.')


def chart_royale():
    for size in (3, 4, 8):
        sh = S.get('duel', {}).get('royale%d' % size)
        if not sh:
            continue
        gs = sorted(sh, key=lambda g: sh[g])
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.barh(range(len(gs)), [100 * sh[g] for g in gs], color=[BOARD[g] for g in gs], ec=INK2, lw=0.4, height=0.66)
        ax.axvline(100.0 / size, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(range(len(gs))); ax.set_yticklabels([GNAME[g] for g in gs])
        for i, g in enumerate(gs):
            ax.text(100 * sh[g] + 0.5, i, '%.1f%%' % (100 * sh[g]), va='center', fontsize=9, color=INK)
        ax.set_xlabel('Win share (dashed line: fair share %.1f%%)' % (100.0 / size))
        ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title('%d players, one colour set each: who wins?' % size, loc='left', fontsize=13, weight='bold', color=INK)
        save(fig, 'royale_%d.png' % size)


def chart_players():
    P = S.get('players')
    if not P:
        return
    highlight = ['orange', 'railroads', 'darkblue', 'stubborn', 'champion']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t, v in P.items():
        xs = sorted(int(n) for n in v)
        ys = [v[str(n)] if str(n) in v else v[n] for n in xs]
        hl = t in highlight
        ax.plot(xs, ys, color=BLUE if hl else GRID, lw=2 if hl else 1.2, marker='o' if hl else None, ms=5, zorder=3 if hl else 1)
        if hl:
            ax.text(xs[-1] + 0.12, ys[-1], SHORT[t], va='center', fontsize=9, color=INK)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel('Players at the table'); ax.set_ylabel('Win rate ÷ fair share (1 = average)')
    ax.set_xlim(1.8, 9.6)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Does the table size change the answer?', loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, 'players.png')


def chart_length():
    L = S.get('length')
    if not L:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, lab, col in (('official', 'Official rules + deals', BLUE), ('swaponly', 'Set-for-set swaps only', ORANGE), ('notrade', 'No trading', INK2)):
        if k not in L:
            continue
        h = L[k]['rounds_hist']
        tot = sum(h) / (1 - L[k]['draws']) if L[k]['draws'] < 1 else 1
        xs = [10 * i + 5 for i in range(30)]
        ax.plot(xs, [100 * x / tot for x in h[:30]], color=col, lw=2, label='%s (%.0f%% never end)' % (lab, 100 * L[k]['draws']))
    ax.set_xlabel('Rounds until one player is left (each player takes a turn per round)')
    ax.set_ylabel('Share of games (%)')
    ax.legend(frameon=False)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('How long a four-player game lasts', loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, 'game_length.png')


def chart_setwin():
    sw = S.get('set_win')
    if not sw:
        return
    gs = sorted(sw, key=lambda g: sw[g])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(range(len(gs)), [100 * sw[g] for g in gs], color=[BOARD[g] for g in gs], ec=INK2, lw=0.4, height=0.66)
    ax.set_yticks(range(len(gs))); ax.set_yticklabels([GNAME[g] for g in gs])
    for i, g in enumerate(gs):
        ax.text(100 * sw[g] + 0.6, i, '%.0f%%' % (100 * sw[g]), va='center', fontsize=9, color=INK)
    ax.axvline(25, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel('How often the player who completed this set went on to win (4 players)')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Complete this set and you usually win', loc='left', fontsize=13, weight='bold', color=INK)
    save(fig, 'set_completed_win.png')


OLD_ECO_NAMES = {'Champion': 'Champion (learned values)', 'Evolved': 'Evolved specialist', 'Evolved r1': 'Evolved generalist (r1)'}


def chart_ecology(key='ecology', fname='ecology.png', title='Let a population of Monopoly players evolve', rename=OLD_ECO_NAMES):
    E = S.get(key)
    if not E:
        return
    pool, traj = E['pool'], E['traj']
    pool = [rename.get(p, p) for p in pool]
    gens = [2 * i for i in range(len(traj))]
    final = traj[-1]
    order = sorted(range(len(pool)), key=lambda i: -max(t[i] for t in traj))
    hl = order[:5]
    pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i in range(len(pool)):
        if i in hl:
            continue
        ax.plot(gens, [100 * t[i] for t in traj], color=GRID, lw=1.2)
    for c, i in zip(pal, hl):
        ys = [100 * t[i] for t in traj]
        ax.plot(gens, ys, color=c, lw=2.2, label=pool[i])
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlabel('Generation'); ax.set_ylabel('Share of the population (%)')
    ax.legend(frameon=False, loc='upper left', ncol=1, fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=14, weight='bold', color=INK)
    fig.text(0.02, -0.03, 'Grey: %s.' % ', '.join(pool[i] for i in range(len(pool)) if i not in hl), fontsize=8.5, color=INK2)
    save(fig, fname)


def chart_evolution(key='evolve_history', fname='evolution.png', ylabel='Win rate vs three Yes-Men (%)', fair=25,
                    title='Evolving a Monopoly strategy', fair_label='fair share 25%'):
    H = S.get(key)
    if not H:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot([h[0] for h in H], [100 * h[1] for h in H], color=BLUE, lw=2.2, marker='o', ms=4, label='best strategy')
    ax.plot([h[0] for h in H], [100 * h[2] for h in H], color=MUTED, lw=1.6, label='median strategy')
    ax.axhline(fair, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(H[-1][0], fair + 0.5, fair_label, ha='right', fontsize=9, color=INK2)
    ax.set_xlabel('Generation'); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, fname)


def chart_ablation(exp='ablation', fname='ablation.png', title="Which of evolution's habits matter?"):
    import glob
    rows = []
    for p in glob.glob(os.path.join(RES, exp, '*.json')):
        d = json.load(open(p))
        rows.append((d['key'], 100 * d['players'][0]['wins'] / d['games']))
    if not rows:
        return
    base = dict(rows).get('evolved')
    lab = {'plus_railroads': 'also buys railroads', 'plus_yellow': 'also buys yellow', 'plus_green': 'also buys green',
           'plus_utilities': 'also buys utilities', 'buy_everything': 'buys everything', 'no_reserve': 'keeps no cash back',
           'no_blocking': 'never bids to block', 'list_price_bids': 'bids only list price', 'leave_jail_always': 'always leaves jail at once',
           'stay_jail_always': 'always stays in jail', 'fair_trader': 'trades at list price', 'no_mortgage_to_build': 'never mortgages to build',
           'three_houses': 'stops at three houses', 'cheap_buyouts': 'pays at most 2.5x for a set piece',
           'no_favourites': 'has no favourite colour', 'no_cash_reserve': 'keeps no cash back', 'hotels': 'builds hotels',
           'no_build_reserve': 'keeps no cash when building', 'fair_trading': 'evens up swaps at printed prices',
           'savvy_trading': 'evens up swaps by set value', 'never_trades': 'never trades'}
    rows = sorted([(k, v - base) for k, v in rows if k != 'evolved'], key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(range(len(rows)), [v for _, v in rows], color=[ORANGE if v < 0 else BLUE for _, v in rows], height=0.62)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab.get(k, k) for k, _ in rows])
    for i, (_, v) in enumerate(rows):
        ax.text(v + (-0.15 if v < 0 else 0.15), i, '%+.1f' % v, va='center', ha='right' if v < 0 else 'left', fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(min(min(v for _, v in rows), 0) - 1.8, max(max(v for _, v in rows), 0) + 1.2)
    ax.set_xlabel('Change in win rate (points) from the evolved strategy\'s %.1f%%' % base)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=14, weight='bold', color=INK)
    save(fig, fname)


def chart_building():
    Bd = S.get('building')
    if not Bd:
        return
    keeps = [r for r in (0, 50, 100, 150, 200, 300, 400, 500, 750, 1000) if 'game__keep_%d' % r in Bd]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, key, lab, col in ((axes[0], 'win', 'Wins per 100 games', BLUE), (axes[1], 'first_out', 'First player to go bankrupt (%)', ORANGE)):
        ys = [Bd['game__keep_%d' % r][key] for r in keeps]
        ax.plot(keeps, ys, color=col, lw=2.2, marker='o', ms=5)
        for x, y in zip(keeps, ys):
            if x in (0, 300, 1000):
                ax.text(x, y + (0.5 if key == 'win' else 0.15), '%.1f' % y, ha='center', fontsize=9, color=INK)
        ax.set_ylabel(lab); ax.set_xlabel('Cash kept in hand when building ($)')
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].axhline(Bd['game__keep_0']['win'], color=MUTED, lw=1, ls=(0, (4, 3)))
    axes[0].set_title('Saving cash costs wins...', loc='left', fontsize=13, weight='bold', color=INK)
    axes[1].set_title('...and does not make you safer', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.04, 'Realistic play: one player vs three typical players, 100,000 games per point. Everyone buys what they land on and '
             'trades only set-for-set.', fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, 'building_cash.png')

    rows = [('one house', 'game__stop_1'), ('two houses', 'game__stop_2'), ('three houses', 'game__stop_3'),
            ('four houses', 'game__stop_4'), ('hotels', 'game__keep_0')]
    rows = [(l, Bd[k]['win']) for l, k in rows if k in Bd]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(rows)), [v for _, v in rows], color=[MUTED] * (len(rows) - 1) + [BLUE], width=0.62)
    for i, (_, v) in enumerate(rows):
        ax.text(i, v + 0.4, '%.1f' % v, ha='center', fontsize=10, color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([l for l, _ in rows])
    ax.set_ylabel('Wins per 100 games')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Stop building at...', loc='left', fontsize=13, weight='bold', color=INK)
    save(fig, 'building_stop.png')


def chart_pairs(cash=1000):
    P = S.get('building_pairs', {}).get(str(cash)) or S.get('building_pairs', {}).get(cash)
    if not P:
        return
    import numpy as np
    order = ['lightblue', 'pink', 'orange', 'brown', 'red', 'yellow', 'darkblue', 'green']
    n = len(order)
    A = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if a == b:
                continue
            key = '%s+%s' % (a, b) if '%s+%s' % (a, b) in P else '%s+%s' % (b, a)
            A[i, j] = 100 * P[key]['%s first' % a]
    seq = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#86b6ef', '#2a78d6', '#104281'])
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    ax.imshow(A, cmap=seq, vmin=55, vmax=90)
    for i in range(n):
        for j in range(n):
            if not np.isnan(A[i, j]):
                better = A[i, j] >= A[j, i]
                ax.text(j, i, '%.0f%s' % (A[i, j], '' if better else ''), ha='center', va='center', fontsize=9.5,
                        color='#ffffff' if A[i, j] > 76 else INK, weight='bold' if better else 'normal')
    ax.set_xticks(range(n)); ax.set_xticklabels([GNAME[g] for g in order], rotation=40, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels([GNAME[g] for g in order])
    ax.set_ylabel('Build this set first'); ax.set_xlabel('...when your other set is')
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Two sets: which gets the houses first?', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.02, -0.02, 'Wins per 100 games. Bold = the better choice of the pair. Mid-game: every property already bought, '
             '$%d each, three typical opponents, 20,000 games per cell.' % cash, fontsize=8.5, color=INK2)
    save(fig, 'building_which_first_%d.png' % cash)

    # max one set vs three-each vs spread, averaged over all 28 pairs
    best_max, best_staged, spread, wrong = [], [], [], []
    for key, v in P.items():
        a, b = key.split('+')
        fa, fb = v['%s first' % a], v['%s first' % b]
        first, other = (a, b) if fa >= fb else (b, a)
        best_max.append(max(fa, fb)); wrong.append(min(fa, fb))
        best_staged.append(v['%s to 3, then %s' % (first, other)]); spread.append(v['spread'])
    rows = [('Better set first,\nall the way to hotels', best_max), ('Better set to 3 houses,\nthen the other, then hotels', best_staged),
            ('One house on every lot,\nthen two, then three...', spread), ('Wrong set first,\nall the way to hotels', wrong)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vals = [100 * sum(r) / len(r) for _, r in rows]
    ax.barh(range(len(rows))[::-1], vals, color=[BLUE, BLUE, ORANGE, MUTED], height=0.6)
    for i, v in zip(range(len(rows))[::-1], vals):
        ax.text(v + 0.4, i, '%.1f' % v, va='center', fontsize=10, color=INK)
    ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([l for l, _ in rows])
    ax.set_xlim(0, max(vals) + 9)
    ax.set_xlabel('Wins per 100 games, average over all 28 pairs of sets ($%d each)' % cash)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Max one set, or one house on everything?', loc='left', fontsize=13, weight='bold', color=INK)
    save(fig, 'building_max_vs_spread_%d.png' % cash)


def bars(rows, title, fname, base=21.8, xlabel='Wins per 100 games (typical player: %.1f)', note=None, highlight=None, extra=None):
    """Horizontal bars of wins per 100 against the typical player's baseline."""
    fig, ax = plt.subplots(figsize=(8.8, 0.52 * len(rows) + 1.5))
    ys = list(range(len(rows)))[::-1]
    cols = [(BLUE if v > base + 0.3 else ORANGE if v < base - 0.3 else MUTED) for _, v in rows]
    ax.barh(ys, [v for _, v in rows], color=cols, height=0.6)
    for y, (lab, v) in zip(ys, rows):
        ax.text(v + 0.3, y, '%.1f' % v + (extra.get(lab, '') if extra else ''), va='center', fontsize=9.5, color=INK)
    ax.axvline(base, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in rows])
    ax.set_xlim(0, max(v for _, v in rows) * 1.22)
    ax.set_xlabel(xlabel % base if '%' in xlabel else xlabel)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=13, weight='bold', color=INK)
    if note:
        fig.text(0.01, -0.06, note, fontsize=8.5, color=INK2)
    save(fig, fname)


NOTE = 'Realistic play: everyone buys what they land on; only set-for-set swaps. One player vs three typical players, 100,000 games per bar.'


def chart_controls():
    C = S.get('controls')
    if not C:
        return
    w = lambda k: C['official__' + k]['win']
    bars([('Always buy', w('buy_always')), ('Mortgage to buy a card\nthat completes or blocks a set', w('mortgage_to_buy_set_cards')),
          ('Pass if it would leave\nyou under $100', w('pass_if_left_under_100')), ('...under $200', w('pass_if_left_under_200')),
          ('...under $300', w('pass_if_left_under_300')), ('...under $500', w('pass_if_left_under_500'))],
         'Buy it or pass?', 'decision_buy.png', note=NOTE)
    bars([('Bid up to half the price', w('bid_half_price')), ('Bid up to 3/4 of the price', w('bid_3_quarters')),
          ('Bid up to the printed price', w('buy_always')), ('Bid up to 1.5x the price', w('bid_1_5x'))],
         'Someone passes: how high do you bid at the auction?', 'decision_auction.png', note=NOTE)
    bars([('Make every fair set-for-set swap', w('buy_always')), ('Only swap if your set\nis at least as good', w('picky_trader')),
          ('Only swap when you have\n$400 to build with', w('trade_only_with_400')), ('Haggle over which set\nis worth more', w('haggler')),
          ('Never trade', w('never_trade'))], 'Trading for a full set', 'decision_trade.png', note=NOTE)
    rr = [('Never buy railroads', 'never_buy_railroads'), ('Buy them when you land', 'buy_always'),
          ('Chase them (mortgage, bid double)', 'railroad_fan'), ('Also trade for the last one', 'railroad_trader'),
          ('Chase them and trade for them', 'railroad_collector')]
    bars([(lab, w(k)) for lab, k in rr], 'Should you go for all four railroads?', 'decision_railroads.png', note=NOTE,
         extra={lab: '   (owns all four in %.0f%% of games)' % C['official__' + k]['all_rr'] for lab, k in rr})
    jl = [('Always pay and leave', 'buy_always'), ('Try one roll, then pay', 'jail_roll_first'), ('Always stay and roll', 'jail_stay'),
          ('Stay if houses are just past jail', 'jail_zone'), ('Stay once 4 or fewer\nproperties are unsold', 'jail_unsold'),
          ('Stay once anyone has houses', 'jail_houses')]
    bars([(lab, w(k)) for lab, k in jl], 'Jail: pay to leave or stay and roll?', 'decision_jail.png', note=NOTE,
         extra={lab: '   (first to go broke %.1f%%)' % C['official__' + k]['first_out'] for lab, k in jl})


def chart_decision_values():
    C, Bd = S.get('controls'), S.get('building')
    if not C or not Bd:
        return
    w = lambda k: C['official__' + k]['win']
    b = lambda k: Bd[k]['win']
    rows = [('Trade: make fair swaps vs never trade', w('buy_always') - w('never_trade')),
            ('Build: spend your cash vs keep $300 back', b('game__keep_0') - b('game__keep_300')),
            ('Trade: even up at printed prices vs haggle', w('buy_always') - w('haggler')),
            ('Owe money: mortgage first vs sell houses first', w('buy_always') - w('sell_houses_before_mortgaging')),
            ('Build: hotels vs stop at three houses', b('game__keep_0') - b('game__stop_3')),
            ('Railroads: trade for all four vs just buy them', w('railroad_collector') - w('buy_always')),
            ('Auctions: bid to 1.5x vs printed price', w('bid_1_5x') - w('buy_always')),
            ('Build: mortgage spare property to build', b('game__mortgage_to_build') - b('game__keep_0')),
            ('Buy: always vs pass when under $300', w('buy_always') - w('pass_if_left_under_300')),
            ('Jail: stay once anyone has houses vs always leave', w('jail_houses') - w('buy_always')),
            ('Buy: mortgage to buy a set card', w('mortgage_to_buy_set_cards') - w('buy_always'))]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.4))
    ax.barh(range(len(rows)), [v for _, v in rows], color=BLUE, height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(v + 0.2, i, '+%.1f' % v, va='center', fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for l, _ in rows])
    ax.set_xlim(0, max(v for _, v in rows) * 1.15)
    ax.set_xlabel('Extra wins per 100 games from the better choice')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('What each decision is worth', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.03, NOTE + ' The two-set choices (which set first, max vs spread) are measured separately: up to 20 and about 9 wins per 100 when they come up.',
             fontsize=8, color=INK2, wrap=True)
    save(fig, 'decision_values.png')


def chart_playbook():
    P = S.get('playbook')
    if not P:
        return
    ns = [2, 3, 4, 5, 6]
    pbv = [P.get('official__playbook_n%d' % n, P['official__playbook'] if n == 4 else None) for n in ns]
    typ = [P.get('official__typical_n%d' % n) for n in ns]
    typ[2] = 21.8
    import numpy as np
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - 0.19, typ, width=0.36, color=MUTED, label='Typical player')
    ax.bar(x + 0.19, pbv, width=0.36, color=BLUE, label='Playbook player')
    for i in range(len(ns)):
        ax.text(x[i] - 0.19, typ[i] + 0.8, '%.1f' % typ[i], ha='center', fontsize=9, color=INK)
        ax.text(x[i] + 0.19, pbv[i] + 0.8, '%.1f' % pbv[i], ha='center', fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['%d players' % n for n in ns])
    ax.set_ylabel('Wins per 100 games')
    ax.legend(frameon=False)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Same dice, better decisions', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.03, 'Playbook player vs typical players at every other seat. With 5-6 players many games never finish under set-for-set trading.', fontsize=8.5, color=INK2)
    save(fig, 'playbook_by_table_size.png')


def chart_final():
    """The charts for the video: realistic play with endgame deals, with auctions (house-rule numbers in the plan)."""
    E = S.get('endgame', {}).get('with_auctions')
    if not E:
        return
    w = lambda k: E[k]['win']
    base = w('buy_always')
    note = ('Everyone buys what they land on; set-for-set swaps; endgame deals when the board locks up. '
            'One player vs three typical players, 100,000 games per bar.')
    bars([('Always buy', w('buy_always')), ('Mortgage to buy a card\nthat completes or blocks a set', w('mortgage_to_buy_set_cards')),
          ('Pass if it would leave\nyou under $100', w('pass_if_left_under_100')), ('...under $200', w('pass_if_left_under_200')),
          ('...under $300', w('pass_if_left_under_300')), ('...under $500', w('pass_if_left_under_500'))],
         'Buy it or pass?', 'final_buy.png', base=base, note=note)
    bars([('Bid up to half the price', w('bid_half_price')), ('Bid up to 3/4 of the price', w('bid_3_quarters')),
          ('Bid up to the printed price', w('buy_always')), ('Bid up to 1.5x the price', w('bid_1_5x'))],
         'Someone passes: how high do you bid at the auction?', 'final_auction.png', base=base, note=note)
    bars([('Make every fair set-for-set swap', w('buy_always')), ('Only swap if your set\nis at least as good', w('picky_trader')),
          ('Only swap when you have\n$400 to build with', w('trade_only_with_400')), ('Haggle over which set\nis worth more', w('haggler')),
          ('In a stalled game, never sell\nthe card that finishes a set', w('endgame_never_sell')),
          ('In a stalled game, pay up to\n5x for the card you need', w('endgame_pay_up')), ('Never trade', w('never_trade'))],
         'Trading for a full set', 'final_trade.png', base=base, note=note)
    rr = [('Never buy railroads', 'never_buy_railroads'), ('Buy them when you land', 'buy_always'),
          ('Chase them (mortgage, bid double)', 'railroad_fan'), ('Also trade for the last one', 'railroad_trader'),
          ('Chase them and trade for them', 'railroad_collector')]
    bars([(lab, w(k)) for lab, k in rr], 'Should you go for all four railroads?', 'final_railroads.png', base=base, note=note,
         extra={lab: '   (owns all four in %.0f%% of games)' % E[k]['all_rr'] for lab, k in rr})
    jl = [('Always pay and leave', 'buy_always'), ('Try one roll, then pay', 'jail_roll_first'), ('Always stay and roll', 'jail_stay'),
          ('Stay if houses are just past jail', 'jail_zone'), ('Stay once 4 or fewer\nproperties are unsold', 'jail_unsold'),
          ('Stay once anyone has houses', 'jail_houses')]
    bars([(lab, w(k)) for lab, k in jl], 'Jail: pay to leave or stay and roll?', 'final_jail.png', base=base, note=note,
         extra={lab: '   (first to go broke %.1f%%)' % E[k]['first_out'] for lab, k in jl})
    st = [('one house', 'stop_1'), ('two houses', 'stop_2'), ('three houses', 'stop_3'), ('four houses', 'stop_4'), ('hotels', 'buy_always')]
    bars([(lab, w(k)) for lab, k in st], 'Stop building at...', 'final_stop.png', base=base, note=note)
    # spend or save: two panels
    keeps = [0, 50, 100, 200, 300, 500, 1000]
    kk = lambda r: 'buy_always' if r == 0 else 'keep_%d' % r
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, key, lab, col in ((axes[0], 'win', 'Wins per 100 games', BLUE), (axes[1], 'first_out', 'First player to go broke (%)', ORANGE)):
        ys = [E[kk(r)][key] for r in keeps]
        ax.plot(keeps, ys, color=col, lw=2.2, marker='o', ms=5)
        for x, y in zip(keeps, ys):
            if x in (0, 300, 1000):
                ax.text(x, y + (0.5 if key == 'win' else 0.15), '%.1f' % y, ha='center', fontsize=9, color=INK)
        ax.set_ylabel(lab); ax.set_xlabel('Cash kept in hand when building ($)')
        ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_title('Saving cash costs wins...', loc='left', fontsize=13, weight='bold', color=INK)
    axes[1].set_title('...and does not make you safer', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.04, note, fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, 'final_building_cash.png')
    # what each decision is worth
    rows = [('Trade: make fair swaps vs never trade', base - w('never_trade')),
            ('Build: spend your cash vs keep $300 back', base - w('keep_300')),
            ('Trade: even up at printed prices vs haggle', base - w('haggler')),
            ('Owe money: mortgage first vs sell houses first', base - w('sell_houses_before_mortgaging')),
            ('Build: hotels vs stop at three houses', base - w('stop_3')),
            ('Railroads: chase and trade for all four vs just buy them', w('railroad_collector') - base),
            ('Auctions: bid to 1.5x vs printed price', w('bid_1_5x') - base),
            ('Build: mortgage spare property to build', w('mortgage_to_build') - base),
            ('Buy: always vs pass when under $300', base - w('pass_if_left_under_300')),
            ('Jail: stay once anyone has houses vs always leave', w('jail_houses') - base),
            ('Buy: mortgage to buy a set card', w('mortgage_to_buy_set_cards') - base)]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.4))
    ax.barh(range(len(rows)), [v for _, v in rows], color=[BLUE if v > 0 else ORANGE for _, v in rows], height=0.6)
    for i, (_, v) in enumerate(rows):
        ax.text(max(v, 0) + 0.2, i, '%+.1f' % v, va='center', fontsize=9.5, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for l, _ in rows])
    ax.set_xlim(min(0, min(v for _, v in rows)) - 0.5, max(v for _, v in rows) * 1.15)
    ax.set_xlabel('Extra wins per 100 games from the better choice')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('What each decision is worth', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.03, note, fontsize=8, color=INK2)
    save(fig, 'final_values.png')
    # playbook by table size
    ns = [2, 3, 4, 5, 6]
    pbv = [E['playbook'] ['win'] if n == 4 else E['playbook_n%d' % n]['win'] for n in ns]
    typ = [E['typical_n%d' % n]['win'] for n in ns]
    import numpy as np
    x = np.arange(len(ns))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - 0.19, typ, width=0.36, color=MUTED, label='Typical player')
    ax.bar(x + 0.19, pbv, width=0.36, color=BLUE, label='Playbook player')
    for i in range(len(ns)):
        ax.text(x[i] - 0.19, typ[i] + 0.8, '%.1f' % typ[i], ha='center', fontsize=9, color=INK)
        ax.text(x[i] + 0.19, pbv[i] + 0.8, '%.1f' % pbv[i], ha='center', fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['%d players' % n for n in ns])
    ax.set_ylabel('Wins per 100 games'); ax.legend(frameon=False)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Same dice, better decisions', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.03, 'Playbook player vs typical players at every other seat; typical bars = one typical player at an all-typical table.', fontsize=8.5, color=INK2)
    save(fig, 'final_playbook.png')
    # two-set scenario
    P = S.get('endgame_pairs')
    if P:
        order = ['lightblue', 'pink', 'orange', 'brown', 'red', 'yellow', 'darkblue', 'green']
        n = len(order)
        A = np.full((n, n), np.nan)
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if a != b:
                    key = '%s+%s' % (a, b) if '%s+%s' % (a, b) in P else '%s+%s' % (b, a)
                    A[i, j] = 100 * P[key]['%s first' % a]
        seq = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#86b6ef', '#2a78d6', '#104281'])
        fig, ax = plt.subplots(figsize=(8.4, 7.4))
        ax.imshow(A, cmap=seq, vmin=55, vmax=90)
        for i in range(n):
            for j in range(n):
                if not np.isnan(A[i, j]):
                    ax.text(j, i, '%.0f' % A[i, j], ha='center', va='center', fontsize=9.5,
                            color='#ffffff' if A[i, j] > 76 else INK, weight='bold' if A[i, j] >= A[j, i] else 'normal')
        ax.set_xticks(range(n)); ax.set_xticklabels([GNAME[g] for g in order], rotation=40, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels([GNAME[g] for g in order])
        ax.set_ylabel('Build this set first'); ax.set_xlabel('...when your other set is')
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title('Two sets: which gets the houses first?', loc='left', fontsize=13, weight='bold', color=INK)
        fig.text(0.02, -0.02, 'Wins per 100 games. Bold = the better choice of the pair. Mid-game: every property already bought, $1,000 each, '
                 'three typical opponents, endgame deals on, 20,000 games per cell.', fontsize=8.5, color=INK2)
        save(fig, 'final_which_first.png')
        best_max, best_staged, spread, wrong = [], [], [], []
        for key, v in P.items():
            a, b = key.split('+'); fa, fb = v['%s first' % a], v['%s first' % b]
            first, other = (a, b) if fa >= fb else (b, a)
            best_max.append(max(fa, fb)); wrong.append(min(fa, fb))
            best_staged.append(v['%s to 3, then %s' % (first, other)]); spread.append(v['spread'])
        rows = [('Better set first,\nall the way to hotels', best_max), ('Better set to 3 houses,\nthen the other, then hotels', best_staged),
                ('One house on every lot,\nthen two, then three...', spread), ('Wrong set first,\nall the way to hotels', wrong)]
        vals = [100 * sum(r) / len(r) for _, r in rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.barh(range(len(rows))[::-1], vals, color=[BLUE, BLUE, ORANGE, MUTED], height=0.6)
        for i, v in zip(range(len(rows))[::-1], vals):
            ax.text(v + 0.4, i, '%.1f' % v, va='center', fontsize=10, color=INK)
        ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([l for l, _ in rows])
        ax.set_xlim(0, max(vals) + 9)
        ax.set_xlabel('Wins per 100 games, average over all 28 pairs of sets ($1,000 each)')
        ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title('Max one set, or one house on everything?', loc='left', fontsize=13, weight='bold', color=INK)
        save(fig, 'final_max_vs_spread.png')


def chart_champion():
    F = S.get('framing')
    if not F:
        return
    sc = {int(k): v for k, v in F['season_champion'].items()}
    gs = sorted(sc)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(gs, [100 * sc[g][0] for g in gs], color=BLUE, lw=2.4, marker='o', ms=5, label='Playbook player')
    ax.plot(gs, [100 * sc[g][1] for g in gs], color=MUTED, lw=2.0, marker='o', ms=4, label='Typical player')
    for g in (10, 52):
        ax.text(g, 100 * sc[g][0] + 3, '%.0f%%' % (100 * sc[g][0]), ha='center', fontsize=10, color=INK)
        ax.text(g, 100 * sc[g][1] - 6, '%.0f%%' % (100 * sc[g][1]), ha='center', fontsize=10, color=INK2)
    ax.set_ylim(0, 80); ax.set_xlim(0, 55)
    ax.set_xlabel('Game nights in the season (the same four players each time)')
    ax.set_ylabel('Chance of finishing the season\nwith the most wins (%)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title('Small edges compound: who is the family champion?', loc='left', fontsize=13, weight='bold', color=INK)
    fig.text(0.01, -0.03, 'Per game: playbook player 32.6 wins in 100, each opponent 22.4. Ties for the most wins are not counted as a championship.',
             fontsize=8.5, color=INK2)
    save(fig, 'final_champion.png')


if __name__ == '__main__':
    chart_champion()
    chart_final()
    chart_controls(); chart_decision_values(); chart_playbook()
    chart_building(); chart_pairs(1000); chart_pairs(500)
    chart_ecology(); chart_evolution(); chart_ablation()
    real_names = {}
    lin = os.path.join(RES, 'evolve_real', 'stage2_lineage.json')
    if os.path.exists(lin):
        for l in json.load(open(lin)):
            favs = ' + '.join(GNAME.get(f, f).lower() for f in l['plain']['favourites']) or 'no favourite'
            real_names['Evolved r%d' % l['round']] = 'Evolved (%s fan)' % favs
    chart_ecology('real_ecology', 'ecology_realistic.png', 'Realistic players: which habits survive?', real_names)
    chart_evolution('real_evolve_history', 'evolution_realistic.png', 'Wins per 100 games vs three typical players', 21.8,
                    'Evolving a realistic Monopoly player', 'typical player 21.8')
    chart_ablation('real_ablation', 'ablation_realistic.png', "Which of the realistic evolved player's habits matter?")
    chart_board('leave'); chart_board('stay')
    if 'table' in S:
        for w, t in (('official', 'Official rules: one strategy vs three Yes-Men'), ('family', 'Family house rules: one strategy vs three Yes-Men'),
                     ('notrade', 'No trading: one strategy vs three Yes-Men'), ('swaponly', 'Set-for-set swaps only: one strategy vs three Yes-Men')):
            if w in S['table']:
                chart_leaderboard(w, t, 'leaderboard_%s.png' % w)
        chart_worlds('rules_change_winner.png')
    if 'skip' in S:
        for w in S['skip']:
            chart_skip(w, 'skip_one_group_%s.png' % w)
    chart_duels(); chart_royale(); chart_h2h('official'); chart_h2h('family'); chart_players(); chart_length(); chart_setwin()
