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


def exp_real_evolution(n_games=100000):
    """Follow-ups to evolve_real.py: ablation, mirror test, arms-race cross-checks, ecology (realistic play)."""
    import evolve_real as er
    ev = os.path.join(RES, 'evolve_real')
    best = json.load(open(os.path.join(ev, 'stage1_final.json')))[0]['genome']
    cfg = dict(REAL_WORLDS['real_official'], tv=','.join('%s:%d' % kv for kv in er.TV.items()))
    typ = er.TYPICAL
    # ablation: undo one habit at a time
    def variant(**kw):
        g = json.loads(json.dumps(best))
        g.update(kw)
        return g
    variants = {
        'evolved': best,
        'no_favourites': variant(fav=[False] * 10),
        'no_cash_reserve': variant(reserve=0, hold=0),
        'list_price_bids': variant(bid=1.0, setbid=1.0, blockbid=0.0, favbid=1.0),
        'hotels': variant(build=5),
        'three_houses': variant(build=3),
        'no_build_reserve': variant(breserve=0),
        'leave_jail_always': variant(jail=0),
        'stay_jail_always': variant(jail=1),
        'no_mortgage_to_build': variant(m2b=0),
        'fair_trading': variant(trade='fair'),
        'savvy_trading': variant(trade='shark'),
        'never_trades': variant(trade='none'),
    }
    for k, g in variants.items():
        if k != 'evolved' and g == best:
            continue
        run('real_ablation', k, [er.spec(g, 'Evolved')] + [typ] * 3, n_games, **cfg)
    # mirror test: one player against three copies of the evolved champion
    ev_spec = er.spec(best, 'Evolved')
    run('real_mirror', 'typical_vs_3_evolved', [typ] + [ev_spec] * 3, n_games, **cfg)
    run('real_mirror', 'allin_vs_3_evolved', ['allin;name=All-In'] + [ev_spec] * 3, n_games, **cfg)
    run('real_mirror', 'four_evolved', [ev_spec] * 4, n_games, **cfg)

def exp_real_race(n_games=100000):
    """Arms-race cross-checks and the realistic ecology (run after evolve_real.py stage2)."""
    import evolve_real as er
    ev = os.path.join(RES, 'evolve_real')
    best = json.load(open(os.path.join(ev, 'stage1_final.json')))[0]['genome']
    cfg = dict(REAL_WORLDS['real_official'], tv=','.join('%s:%d' % kv for kv in er.TV.items()))
    typ = er.TYPICAL
    # arms race cross-checks
    lin_path = os.path.join(ev, 'stage2_lineage.json')
    lin = json.load(open(lin_path)) if os.path.exists(lin_path) else []
    specs = {l['round']: er.spec(l['genome'], 'r%d' % l['round']) for l in lin}
    for a_, b_ in itertools.permutations(sorted(specs), 2):
        run('real_mirror', 'r%d_vs_3_r%d' % (a_, b_), [specs[a_]] + [specs[b_]] * 3, n_games // 2, **cfg)
    for r_, sp in specs.items():
        run('real_mirror', 'r%d_vs_3_typical' % r_, [sp] + [typ] * 3, n_games // 2, **cfg)
        run('real_mirror', 'typical_vs_3_r%d' % r_, [typ] + [sp] * 3, n_games // 2, **cfg)
    # ecology: every four-seat table from a pool of realistic players
    pool = [('Typical player', typ), ('All-In', 'allin;name=All-In'), ('Expert playbook', 'champion;name=Expert playbook'),
            ('Keeps $300 back', 'prudent;name=Keeps $300 back'), ('Stops at three houses', 'threehouses;name=Stops at three houses'),
            ('Never trades', 'stubborn;name=Never trades'), ('Railroad fan', 'everything;name=Railroad fan;fav=rr'),
            ('Orange fan', 'everything;name=Orange fan;fav=orange'),
            ('Boardwalk saver', 'everything;name=Boardwalk saver;fav=darkblue;hold=400')]
    for r_ in sorted(specs):
        pool.append(('Evolved r%d' % r_, er.spec(lin[r_]['genome'], 'Evolved r%d' % r_)))
    if not specs:
        pool.append(('Evolved r0', er.spec(best, 'Evolved r0')))
    json.dump([nm for nm, _ in pool], open(os.path.join(RES, 'real_ecology_pool.json'), 'w'))
    for combo in itertools.combinations_with_replacement(range(len(pool)), 4):
        run('real_ecology', 'm' + '_'.join(str(i) for i in combo), [pool[i][1] for i in combo], 4000, **cfg)


def exp_building(n_games=100000, n_scen=20000):
    """How to build, under realistic play: cash kept back, where to stop, which set first, focus vs spread."""
    import evolve_real as er
    cfg = dict(REAL_WORLDS['real_official'], tv=','.join('%s:%d' % kv for kv in er.TV.items()))
    typ = er.TYPICAL
    # A. full games: one player vs three typical players
    variants = [('keep_%d' % r, 'everything;breserve=%d' % r) for r in (0, 50, 100, 150, 200, 300, 400, 500, 750, 1000)]
    variants += [('danger_%s' % f, 'everything;danger=%s' % f) for f in ('0.5', '1', '1.5', '2')]
    variants += [('stop_%d' % b, 'everything;build=%d' % b) for b in (1, 2, 3, 4)]
    variants += [('mortgage_to_build', 'everything;m2b=1'), ('staged', 'everything;bpri=staged'),
                 ('staged_cheap', 'everything;bpri=staged;border=cheap')]
    variants += [('order_cheap', 'everything;border=cheap'), ('order_dear', 'everything;border=dear'),
                 ('spread', 'everything;bpri=spread'), ('spread_cheap', 'everything;bpri=spread;border=cheap'),
                 ('focus', 'everything;bpri=focus'), ('focus_cheap', 'everything;bpri=focus;border=cheap'),
                 ('focus_dear', 'everything;bpri=focus;border=dear')]
    for k, sp in variants:
        run('building', 'game__' + k, [sp.replace('everything;', 'everything;name=%s;' % k)] + [typ] * 3, n_games, **cfg)
    for k in ('keep_0', 'keep_100', 'keep_300', 'keep_500', 'danger_1', 'spread', 'order_cheap'):
        sp = dict(variants)[k]
        run('building', 'family__' + k, [sp.replace('everything;', 'everything;name=%s;' % k)] + [typ] * 3, n_games,
            **dict(REAL_WORLDS['real_family'], tv=cfg['tv']))
    # the evolved realistic player with different building habits
    best = json.load(open(os.path.join(RES, 'evolve_real', 'stage1_final.json')))[0]['genome']
    base = er.spec(best, 'Evolved')
    def ev(**kw):
        parts = [p_ for p_ in base.split(';') if p_.split('=')[0] not in kw]
        return ';'.join(parts + ['%s=%s' % kv for kv in kw.items()])
    for k, sp in (('evolved', base), ('evolved_keep_0', ev(breserve=0)), ('evolved_keep_300', ev(breserve=300)),
                  ('evolved_keep_500', ev(breserve=500)), ('evolved_danger_1', ev(danger=1)), ('evolved_spread', ev(bpri='spread')),
                  ('evolved_focus', ev(bpri='focus')), ('evolved_cheap', ev(border='cheap')), ('evolved_dear', ev(border='dear'))):
        run('building', 'game__' + k, [sp] + [typ] * 3, n_games, **cfg)
    # B. mid-game: you hold two complete sets, everything else already bought, everyone has the same cash
    for cash in (500, 1000):
        for a_, b_ in itertools.combinations(GROUPS, 2):
            give = '0:%s,0:%s' % (a_, b_)
            for k, sp in (('first_%s' % a_, 'everything;first=%s' % a_), ('first_%s' % b_, 'everything;first=%s' % b_),
                          ('spread', 'everything;bpri=spread'),
                          ('staged_%s' % a_, 'everything;bpri=staged;first=%s' % a_), ('staged_%s' % b_, 'everything;bpri=staged;first=%s' % b_)):
                run('building_pairs', 'c%d__%s__%s__%s' % (cash, a_, b_, k), [sp.replace('everything;', 'everything;name=%s;' % k)] + [typ] * 3,
                    n_scen, give=give, deal=1, cash=cash, **cfg)


CONTROL_VARIANTS = [
    # buy or pass
    ('buy_always', 'everything'),
    ('pass_if_left_under_100', 'everything;reserve=100'),
    ('pass_if_left_under_200', 'everything;reserve=200'),
    ('pass_if_left_under_300', 'everything;reserve=300'),
    ('pass_if_left_under_500', 'everything;reserve=500'),
    ('pass_if_under_300_except_set_cards', 'everything;reserve=300;stretch=1'),
    ('mortgage_to_buy_set_cards', 'everything;buymort=1'),
    ('mortgage_to_buy_anything', 'everything;buymort=2'),
    ('bid_half_price', 'everything;bid=0.5'), ('bid_3_quarters', 'everything;bid=0.75'),
    ('bid_1_5x', 'everything;bid=1.5'), ('bid_2x', 'everything;bid=2.0'),
    ('bid_2x_to_block', 'everything;blockbid=2.0'),
    # trading
    ('never_trade', 'everything;trade=none'), ('picky_trader', 'everything;trade=picky'),
    ('haggler', 'everything;trade=shark'), ('trade_only_with_200', 'everything;tcash=200'),
    ('trade_only_with_400', 'everything;tcash=400'),
    # railroads
    ('railroad_fan', 'everything;fav=rr'), ('railroad_trader', 'everything;rrtrade=1'),
    ('railroad_collector', 'everything;fav=rr;rrtrade=1;buymort=1'), ('never_buy_railroads', 'everything;buy=all-rr'),
    # cash when you owe
    ('sell_houses_before_mortgaging', 'everything;raise=sell'),
    # jail
    ('jail_stay', 'everything;jail=stay'), ('jail_smart', 'everything;jail=smart'), ('jail_zone', 'everything;jail=zone'),
    ('jail_houses', 'everything;jail=houses'), ('jail_unsold', 'everything;jail=unsold'), ('jail_roll_first', 'everything;jail=rollfirst'),
]


def exp_controls(n_games=100000):
    """The decisions a player controls, under realistic play: buy or pass, trading, railroads, raising cash, jail."""
    import evolve_real as er
    tv = ','.join('%s:%d' % kv for kv in er.TV.items())
    typ = er.TYPICAL
    for k, sp in CONTROL_VARIANTS:
        sp = sp.replace('everything', 'everything;name=%s' % k, 1)
        run('controls', 'official__' + k, [sp] + [typ] * 3, n_games, tv=tv, **REAL_WORLDS['real_official'])
    for k in ('buy_always', 'pass_if_left_under_300', 'mortgage_to_buy_set_cards', 'bid_2x', 'picky_trader', 'trade_only_with_400',
              'railroad_collector', 'sell_houses_before_mortgaging', 'jail_stay', 'jail_smart', 'jail_zone', 'jail_roll_first'):
        sp = dict(CONTROL_VARIANTS)[k].replace('everything', 'everything;name=%s' % k, 1)
        run('controls', 'family__' + k, [sp] + [typ] * 3, n_games, tv=tv, **REAL_WORLDS['real_family'])
    best = json.load(open(os.path.join(RES, 'evolve_real', 'stage1_final.json')))[0]['genome']
    base = er.spec(best, 'Evolved')
    def ev(**kw):
        parts = [p_ for p_ in base.split(';') if p_.split('=')[0] not in kw]
        return ';'.join(parts + ['%s=%s' % kv for kv in kw.items()])
    for k, sp in (('evolved', base), ('evolved_jail_leave', ev(jail='leave')), ('evolved_jail_stay', ev(jail='stay')),
                  ('evolved_jail_zone', ev(jail='zone')), ('evolved_jail_roll_first', ev(jail='rollfirst')),
                  ('evolved_railroad_trader', ev(rrtrade=1)), ('evolved_mortgage_to_buy_set_cards', ev(buymort=1)),
                  ('evolved_picky_trader', ev(trade='picky'))):
        run('controls', 'official__' + k, [sp] + [typ] * 3, n_games, tv=tv, **REAL_WORLDS['real_official'])


PLAYBOOK = 'everything;name=Playbook;bid=1.5;buymort=1;rrtrade=1;m2b=1;border=cheap;jail=houses'


def exp_playbook(n_games=100000):
    """Every best choice from the controls and building experiments in one player."""
    import evolve_real as er
    tv = ','.join('%s:%d' % kv for kv in er.TV.items())
    typ = er.TYPICAL
    off, fam = REAL_WORLDS['real_official'], REAL_WORLDS['real_family']
    run('playbook', 'official__playbook', [PLAYBOOK] + [typ] * 3, n_games, tv=tv, **off)
    run('playbook', 'official__playbook_keep_100', [PLAYBOOK + ';breserve=100'] + [typ] * 3, n_games, tv=tv, **off)
    run('playbook', 'official__playbook_railroad_fan', [PLAYBOOK + ';fav=rr'] + [typ] * 3, n_games, tv=tv, **off)
    run('playbook', 'family__playbook', [PLAYBOOK] + [typ] * 3, n_games, tv=tv, **fam)
    run('playbook', 'official__typical_vs_3_playbook', [typ] + [PLAYBOOK] * 3, n_games, tv=tv, **off)
    run('playbook', 'official__four_playbook', [PLAYBOOK] * 4, n_games, tv=tv, **off)
    for n in (2, 3, 5, 6):
        run('playbook', 'official__playbook_n%d' % n, [PLAYBOOK] + [typ] * (n - 1), n_games // 2, tv=tv, **off)
        run('playbook', 'official__typical_n%d' % n, [typ] * n, n_games // 2, tv=tv, **off)


END_WORLDS = {
    'with_auctions': dict(auctions=1, trading=1, endgame=10),     # official rules + endgame deals when the board locks up
    'no_auctions': dict(auctions=0, trading=1, endgame=10),       # same, but unbought property just stays unsold
}
END_EXTRA = [
    ('endgame_never_sell', 'everything;sell=0'),   # when the game stalls, never sells a set card for cash
    ('endgame_pay_up', 'everything;buyp=5'),       # when the game stalls, pays up to 5x for a card that finishes a set
]
END_BUILD = [('keep_%d' % r, 'everything;breserve=%d' % r) for r in (50, 100, 200, 300, 500, 1000)] + [
    ('danger_1', 'everything;danger=1'), ('stop_1', 'everything;build=1'), ('stop_2', 'everything;build=2'),
    ('stop_3', 'everything;build=3'), ('stop_4', 'everything;build=4'), ('mortgage_to_build', 'everything;m2b=1'),
    ('order_cheap', 'everything;border=cheap'), ('order_dear', 'everything;border=dear'), ('spread', 'everything;bpri=spread'),
    ('staged', 'everything;bpri=staged')]
AUCTION_ONLY = ('bid_half_price', 'bid_3_quarters', 'bid_1_5x', 'bid_2x', 'bid_2x_to_block')


def exp_endgame(n_games=100000, n_scen=20000):
    """The video's final numbers: realistic play plus endgame deals, with and without auctions."""
    import evolve_real as er
    tv = ','.join('%s:%d' % kv for kv in er.TV.items())
    typ = er.TYPICAL
    variants = CONTROL_VARIANTS + END_EXTRA + END_BUILD
    best = json.load(open(os.path.join(RES, 'evolve_real', 'stage1_final.json')))[0]['genome']
    for world, cfg in END_WORLDS.items():
        for k, sp in variants:
            if world == 'no_auctions' and k in AUCTION_ONLY:
                continue
            run('endgame', '%s__%s' % (world, k), [sp.replace('everything', 'everything;name=%s' % k, 1)] + [typ] * 3, n_games, tv=tv, **cfg)
        run('endgame', '%s__playbook' % world, [PLAYBOOK] + [typ] * 3, n_games, tv=tv, **cfg)
        run('endgame', '%s__playbook_keep_100' % world, [PLAYBOOK + ';breserve=100'] + [typ] * 3, n_games, tv=tv, **cfg)
        run('endgame', '%s__typical_vs_3_playbook' % world, [typ] + [PLAYBOOK] * 3, n_games, tv=tv, **cfg)
        run('endgame', '%s__evolved' % world, [er.spec(best, 'Evolved')] + [typ] * 3, n_games, tv=tv, **cfg)
        for n in (2, 3, 4, 5, 6):
            run('endgame', '%s__typical_n%d' % (world, n), [typ] * n, n_games // 2, tv=tv, **cfg)
            if n != 4:
                run('endgame', '%s__playbook_n%d' % (world, n), [PLAYBOOK] + [typ] * (n - 1), n_games // 2, tv=tv, **cfg)
    # two complete sets, mid-game, $1,000 each, with endgame deals
    for a_, b_ in itertools.combinations(GROUPS, 2):
        give = '0:%s,0:%s' % (a_, b_)
        for k, sp in (('first_%s' % a_, 'everything;first=%s' % a_), ('first_%s' % b_, 'everything;first=%s' % b_),
                      ('spread', 'everything;bpri=spread'),
                      ('staged_%s' % a_, 'everything;bpri=staged;first=%s' % a_), ('staged_%s' % b_, 'everything;bpri=staged;first=%s' % b_)):
            run('endgame_pairs', 'c1000__%s__%s__%s' % (a_, b_, k), [sp.replace('everything;', 'everything;name=%s;' % k)] + [typ] * 3,
                n_scen, give=give, deal=1, cash=1000, tv=tv, **END_WORLDS['with_auctions'])


def exp_framing(n_games=100000):
    """Price the playbook's edge in dollars: extra starting cash a typical player needs to match it."""
    import evolve_real as er
    tv = ','.join('%s:%d' % kv for kv in er.TV.items())
    typ = er.TYPICAL
    for b_ in (250, 500, 750, 1000, 1250, 1500, 2000, 3000):
        run('framing', 'bonus_%d' % b_, ['everything;name=bonus %d;bonus=%d' % (b_, b_)] + [typ] * 3, n_games, tv=tv, **END_WORLDS['with_auctions'])
    run('framing', 'playbook_with_bonus_minus_1000', [PLAYBOOK + ';bonus=-1000'] + [typ] * 3, n_games, tv=tv, **END_WORLDS['with_auctions'])
    # finer steps around the match, 200,000 games each, with and without auctions
    for b_ in (225, 250, 275, 300, 325, 350):
        run('framing', 'fine_bonus_%d' % b_, ['everything;name=bonus %d;bonus=%d' % (b_, b_)] + [typ] * 3, 2 * n_games, seed=5, tv=tv, **END_WORLDS['with_auctions'])
    for b_ in (100, 150, 200, 250, 300):
        run('framing', 'noauction_bonus_%d' % b_, ['everything;name=bonus %d;bonus=%d' % (b_, b_)] + [typ] * 3, 2 * n_games, seed=5, tv=tv, **END_WORLDS['no_auctions'])
    # the other direction: how much can the playbook player hand to the bank and still only tie a typical player?
    for b_ in (150, 200, 250, 300, 350, 400, 450):
        run('framing', 'playbook_minus_%d' % b_, [PLAYBOOK + ';bonus=-%d' % b_] + [typ] * 3, 2 * n_games, seed=5, tv=tv, **END_WORLDS['with_auctions'])
    for b_ in (100, 150, 200, 250, 300):
        run('framing', 'noauction_playbook_minus_%d' % b_, [PLAYBOOK + ';bonus=-%d' % b_] + [typ] * 3, 2 * n_games, seed=5, tv=tv, **END_WORLDS['no_auctions'])


EXPERIMENTS = {'framing': exp_framing, 'endgame': exp_endgame, 'playbook': exp_playbook, 'controls': exp_controls, 'building': exp_building, 'real_race': exp_real_race, 'real_evolution': exp_real_evolution, 'realistic': exp_realistic, 'ecology': exp_ecology, 'ablation': exp_ablation, 'shark': exp_shark, 'length': exp_length, 'duel': exp_duel, 'table': exp_table, 'skip': exp_skip,
               'h2h': exp_h2h, 'players': exp_players}

RR_PRICES = [200, 250, 300, 350, 400, 500, 600, 700, 800, 1000, 1200, 1500]
COLLECTOR = 'everything;fav=rr;rrtrade=1;buymort=1'


def exp_deepdive(n_games=100000):
    """Follow-ups from the script review: paying cash for railroads, and when to sell in a stalled game."""
    import evolve_real as er
    tv = ','.join('%s:%d' % kv for kv in er.TV.items())
    typ = er.TYPICAL
    cfg = END_WORLDS['with_auctions']
    # railroads: offer a fixed cash price for every railroad an opponent holds (owners assumed to accept)
    for p in RR_PRICES:
        run('deepdive', 'rr_cash_%d' % p, ['everything;name=rr_cash_%d;rroffer=%d' % (p, p)] + [typ] * 3, n_games, tv=tv, **cfg)
        run('deepdive', 'rr_all_out_%d' % p, [COLLECTOR + ';name=rr_all_out_%d;rroffer=%d' % (p, p)] + [typ] * 3, n_games, tv=tv, **cfg)
    # the stalled game: when to sell the card that finishes someone's set, and for how much
    for k, sp in [('sell_2.5x', 'everything'), ('never_sell', 'everything;sell=0'), ('sell_only_with_a_set', 'everything;sellset=1'),
                  ('sell_1.5x', 'everything;sell=1.5'), ('sell_2x', 'everything;sell=2'), ('sell_3x', 'everything;sell=3'),
                  ('sell_only_with_a_set_3x', 'everything;sellset=1;sell=3')]:
        run('deepdive', 'endgame_' + k, [sp.replace('everything', 'everything;name=%s' % k, 1)] + [typ] * 3, n_games, tv=tv, **cfg)
    # the collector, adding cash only for the last railroad it is missing
    for p in [250, 300, 400, 500, 600, 800, 1000]:
        run('deepdive', 'rr_last_%d' % p, [COLLECTOR + ';name=rr_last_%d;rroffer=%d;rrlast=1' % (p, p)] + [typ] * 3, n_games, tv=tv, **cfg)
    # the playbook, selling in a stalled game only with a set of its own
    run('deepdive', 'playbook_sellset', [PLAYBOOK + ';sellset=1'] + [typ] * 3, n_games, tv=tv, **cfg)
    run('deepdive', 'playbook', [PLAYBOOK] + [typ] * 3, n_games, tv=tv, **cfg)


EXPERIMENTS['deepdive'] = exp_deepdive

if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPERIMENTS)
    for n in names:
        EXPERIMENTS[n]()
