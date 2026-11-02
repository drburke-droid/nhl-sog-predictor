"""Run the strategy experiments with the C++ engine and save every result as JSON.

    python3 experiments.py [name ...]     # default: all experiments, skipping ones already saved

Each run is cached at results/<experiment>/<key>.json, so re-running only fills in what is missing.
"""
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'monopoly')
RES = os.path.join(HERE, 'results')

GROUPS = ['brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue']

# rule worlds: official rules with trading, and the ways people actually play
WORLDS = {
    'official': dict(auctions=1, trading=1, buyouts=1),       # official rules, swaps and cash deals
    'swaponly': dict(auctions=1, trading=1),                  # trades only when both complete a set
    'notrade': dict(auctions=1, trading=0),
    'noauction': dict(auctions=0, trading=1, buyouts=1),
    'family': dict(auctions=0, trading=1, buyouts=1, fp=1),   # no auctions + Free Parking jackpot
    'kitchen': dict(auctions=0, trading=1, buyouts=1, fp=1, go400=1),  # ... + $400 for landing on Go
}

# the cast: each faces three "Yes-Men" (buy everything, build hotels at once, leave jail at once)
ROSTER = [
    'everything', 'nothing', 'railroads', 'utilities', 'infrastructure',
    'brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue',
    'prudent', 'threehouses', 'houselock', 'jailbird', 'smartjail', 'stubborn', 'shark', 'allin', 'champion',
]

H2H = ['everything', 'nothing', 'railroads', 'darkblue', 'orange', 'prudent', 'threehouses',
       'houselock', 'jailbird', 'stubborn', 'allin', 'champion']


def run(exp, key, strats, games, seed=1, **cfg):
    path = os.path.join(RES, exp, key + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = [BIN, '--strats', '|'.join(strats), '--games', str(games), '--seed', str(seed)]
    for k, v in cfg.items():
        args += ['--' + k, str(v)]
    out = json.loads(subprocess.check_output(args))
    out['strats'] = strats
    out['key'] = key
    json.dump(out, open(path, 'w'))
    p0 = out['players'][0]
    print('%-12s %-40s win %.2f%%  draws %.1f%%' % (exp, key, 100 * p0['wins'] / out['games'], 100 * out['timeouts'] / out['games']), flush=True)
    return out


def exp_table(n_games=100000):
    for world, cfg in WORLDS.items():
        g = n_games if world in ('official', 'family') else n_games // 2
        for t in ROSTER:
            run('table', '%s__%s' % (world, t), [t] + ['everything'] * 3, g, **cfg)


def exp_skip(n_games=100000):
    for world in ('official', 'family'):
        for g in GROUPS + ['rr', 'utilities']:
            spec = 'everything;name=skip_%s;buy=all-%s' % (g, g)
            run('skip', '%s__skip_%s' % (world, g), [spec] + ['everything'] * 3, n_games, **WORLDS[world])


def exp_duel(n_games=40000):
    # two players, each handed one colour set, nothing else for sale; build at once
    for build in (5, 3):
        for a in GROUPS:
            for b in GROUPS:
                if a >= b and a != b:
                    continue
                s = 'nothing;name=%s;build=%d' % (a, build)
                t = 'nothing;name=%s;build=%d' % (b, build)
                run('duel', 'b%d__%s__%s' % (build, a, b), [s, t], n_games, give='0:%s,1:%s' % (a, b))
    # everyone gets a set: 8-player free-for-all
    specs = ['nothing;name=%s' % g for g in GROUPS]
    run('duel', 'royale8', specs, n_games * 5, give=','.join('%d:%s' % (i, g) for i, g in enumerate(GROUPS)))
    # every 4-set table
    for combo in itertools.combinations(GROUPS, 4):
        specs = ['nothing;name=%s' % g for g in combo]
        run('duel', 'royale4__' + '_'.join(combo), specs, n_games // 2, give=','.join('%d:%s' % (i, g) for i, g in enumerate(combo)))
    for combo in itertools.combinations(GROUPS, 3):
        specs = ['nothing;name=%s' % g for g in combo]
        run('duel', 'royale3__' + '_'.join(combo), specs, n_games // 2, give=','.join('%d:%s' % (i, g) for i, g in enumerate(combo)))


def exp_h2h(n_games=40000):
    for world in ('official', 'family'):
        for a, b in itertools.combinations(H2H, 2):
            run('h2h', '%s__%s__%s' % (world, a, b), [a, b], n_games, **WORLDS[world])


def exp_players(n_games=60000):
    tests = ['everything', 'railroads', 'darkblue', 'orange', 'green', 'stubborn', 'threehouses',
             'houselock', 'jailbird', 'allin', 'champion', 'shark']
    for n in range(2, 9):
        for t in tests:
            run('players', 'official__n%d__%s' % (n, t), [t] + ['everything'] * (n - 1), n_games, **WORLDS['official'])


def exp_length(n_games=200000):
    for world, cfg in WORLDS.items():
        run('length', world, ['everything'] * 4, n_games, **cfg)
    for n in range(2, 9):
        run('length', 'official_n%d' % n, ['everything'] * n, n_games // 2, **WORLDS['official'])


def learned_values():
    """Set values the simulation measured: points of win rate lost by never buying each set (official rules)."""
    base = json.load(open(os.path.join(RES, 'table', 'official__everything.json')))
    b = base['players'][0]['wins'] / base['games']
    tv = {}
    for g in GROUPS:
        d = json.load(open(os.path.join(RES, 'skip', 'official__skip_%s.json' % g)))
        tv[g] = round(10000 * (b - d['players'][0]['wins'] / d['games']))  # 1 point of win rate = $100
    return tv


def exp_shark(n_games=100000):
    tv = learned_values()
    spec_tv = ','.join('%s:%d' % kv for kv in tv.items())
    json.dump(tv, open(os.path.join(RES, 'learned_values.json'), 'w'), indent=1)
    w = WORLDS['official']
    run('shark', 'textbook', ['shark;name=shark_textbook'] + ['everything'] * 3, n_games, **w)
    run('shark', 'learned', ['shark;name=shark_learned'] + ['everything'] * 3, n_games, tv=spec_tv, **w)
    run('shark', 'champion_learned', ['champion;name=champion_learned'] + ['everything'] * 3, n_games, tv=spec_tv, **w)
    run('shark', 'allin_learned', ['allin;name=allin_shark;trade=shark'] + ['everything'] * 3, n_games, tv=spec_tv, **w)
    run('shark', 'four_learned_sharks', ['shark;name=shark_learned'] * 4, n_games, tv=spec_tv, **w)


def exp_ablation(n_games=100000):
    """Take the evolved champion and undo one of its habits at a time."""
    best = json.load(open(os.path.join(RES, 'evolve', 'stage1_final.json')))[0]['spec']
    tv = ','.join('%s:%d' % kv for kv in learned_values().items())
    w = dict(WORLDS['official'], tv=tv)
    base = best.replace('name=evolved', 'name=evolved')
    def with_(k, v):
        parts = [p for p in base.split(';') if not p.startswith(k + '=')]
        return ';'.join(parts + ['%s=%s' % (k, v), 'name=%s_%s' % (k, v)])
    buy = [p for p in base.split(';') if p.startswith('buy=')][0][4:]
    variants = {
        'evolved': base,
        'plus_railroads': with_('buy', buy + '+rr'),
        'plus_yellow': with_('buy', buy + '+yellow'),
        'plus_green': with_('buy', buy + '+green'),
        'plus_utilities': with_('buy', buy + '+utilities'),
        'buy_everything': with_('buy', 'all'),
        'no_reserve': with_('reserve', 0),
        'no_blocking': with_('blockbid', 0),
        'list_price_bids': with_('bid', 1.0),
        'leave_jail_always': with_('jail', 'leave'),
        'stay_jail_always': with_('jail', 'stay'),
        'fair_trader': with_('trade', 'fair'),
        'no_mortgage_to_build': with_('m2b', 0),
        'three_houses': with_('build', 3),
        'cheap_buyouts': with_('buyp', 2.5),
    }
    for k, sp in variants.items():
        run('ablation', k, [sp] + ['everything'] * 3, n_games, **w)


def ecology_pool():
    ev = os.path.join(RES, 'evolve')
    pool = [('Yes-Man', 'everything;name=Yes-Man'), ('All-In', 'allin;name=All-In'),
            ('Three-House', 'threehouses;name=Three-House'), ('Accountant', 'prudent;name=Accountant'),
            ('Never trades', 'stubborn;name=Never trades'), ('Railroad Tycoon', 'railroads;name=Railroad Tycoon'),
            ('Boardwalk Dreamer', 'darkblue;name=Boardwalk Dreamer'), ('Orange only', 'orange;name=Orange only'),
            ('Champion', 'champion;name=Champion')]
    s1 = json.load(open(os.path.join(ev, 'stage1_final.json')))[0]['spec']
    pool.append(('Evolved', s1.replace('name=evolved', 'name=Evolved')))
    lin = os.path.join(ev, 'stage2_lineage.json')
    if os.path.exists(lin):
        last = json.load(open(lin))[-1]
        pool.append(('Evolved r%d' % last['round'], last['spec'].replace('name=evolved', 'name=Evolved r%d' % last['round'])))
    return pool


def exp_ecology(n_games=5000):
    """Every 4-seat table (with repeats) drawn from the pool; payoffs feed replicator dynamics in analyze.py."""
    pool = ecology_pool()
    tv = ','.join('%s:%d' % kv for kv in learned_values().items())
    json.dump([n for n, _ in pool], open(os.path.join(RES, 'ecology_pool.json'), 'w'))
    for combo in itertools.combinations_with_replacement(range(len(pool)), 4):
        key = 'm' + '_'.join(str(i) for i in combo)
        run('ecology', key, [pool[i][1] for i in combo], n_games, tv=tv, **WORLDS['official'])


REAL_WORLDS = {
    'real_official': dict(auctions=1, trading=1),        # swaps only when both complete a set; no cash for a set card
    'real_family': dict(auctions=0, trading=1, fp=1),    # same trading, no auctions, Free Parking money
}
REAL_ROSTER = [
    ('typical', 'everything;name=Typical player'),
    ('fan_rr', 'everything;name=Railroad fan;fav=rr'),
    ('fan_utilities', 'everything;name=Utilities fan;fav=utilities'),
    ('fan_brown', 'everything;name=Brown fan;fav=brown'),
    ('fan_lightblue', 'everything;name=Light blue fan;fav=lightblue'),
    ('fan_pink', 'everything;name=Pink fan;fav=pink'),
    ('fan_orange', 'everything;name=Orange fan;fav=orange'),
    ('fan_red', 'everything;name=Red fan;fav=red'),
    ('fan_yellow', 'everything;name=Yellow fan;fav=yellow'),
    ('fan_green', 'everything;name=Green fan;fav=green'),
    ('fan_darkblue', 'everything;name=Boardwalk fan;fav=darkblue'),
    ('saver_darkblue', 'everything;name=Boardwalk saver;fav=darkblue;hold=400'),
    ('cautious', 'prudent;name=Keeps $300 back'),
    ('threehouses', 'threehouses;name=Stops at three houses'),
    ('houselock', 'houselock;name=Four houses, never hotels'),
    ('smartjail', 'smartjail;name=Out of jail early, in late'),
    ('jailbird', 'jailbird;name=Always stays in jail'),
    ('stubborn', 'stubborn;name=Never trades'),
    ('allin', 'allin;name=All-In'),
    ('savvy', 'shark;name=Savvy trader'),
    ('champion', 'champion;name=Expert playbook'),
]


def exp_realistic(n_games=100000):
    """Realistic play: everyone buys what they land on; trades only when both players complete a set."""
    tv = ','.join('%s:%d' % kv for kv in learned_values().items())
    for world, cfg in REAL_WORLDS.items():
        for key, spec in REAL_ROSTER:
            run('realistic', '%s__%s' % (world, key), [spec] + ['everything;name=Typical player'] * 3, n_games, tv=tv, **cfg)
        for g in GROUPS + ['rr', 'utilities']:
            spec = 'everything;name=Never buys %s;buy=all-%s' % (g, g)
            run('realistic', '%s__skip_%s' % (world, g), [spec] + ['everything;name=Typical player'] * 3, n_games, tv=tv, **cfg)
    for n in (2, 3, 4, 5, 6):
        run('realistic', 'real_official__length_n%d' % n, ['everything'] * n, n_games, **REAL_WORLDS['real_official'])


EXPERIMENTS = {'realistic': exp_realistic, 'ecology': exp_ecology, 'ablation': exp_ablation, 'shark': exp_shark, 'length': exp_length, 'duel': exp_duel, 'table': exp_table, 'skip': exp_skip,
               'h2h': exp_h2h, 'players': exp_players}

if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPERIMENTS)
    for n in names:
        EXPERIMENTS[n]()
