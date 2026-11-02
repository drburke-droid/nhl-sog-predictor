"""Evolve a Monopoly strategy under realistic play.

Realistic play: every player buys what it lands on (unless it is keeping cash back); the only trades are
swaps in which both players complete a colour set (cash evens up the printed prices); nobody ever sells a
set-completing card for cash. Evolution can only tune choices a real player makes: up to two favourite
groups (and how hard to chase them), cash kept back, auction bids, when to stop building, mortgaging to
build, jail timing, and how to price a set-for-set swap.

    python3 evolve_real.py stage1 [generations]   # beat three typical players
    python3 evolve_real.py stage2 [rounds]        # arms race: beat three copies of the last champion
Results go to results/evolve_real/.
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'monopoly')
RES = os.path.join(HERE, 'results')
OUT = os.path.join(RES, 'evolve_real')
os.makedirs(OUT, exist_ok=True)
GROUPS = ['brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue', 'rr', 'utilities']
TYPICAL = 'everything;name=Typical player'


def realistic_values():
    """Set values measured under realistic play: points of win rate lost by never buying the set, x $100."""
    path = os.path.join(RES, 'realistic_values.json')
    if os.path.exists(path):
        return json.load(open(path))
    base = json.load(open(os.path.join(RES, 'realistic', 'real_official__typical.json')))
    b = base['players'][0]['wins'] / base['games']
    tv = {}
    for g in GROUPS[:8]:
        d = json.load(open(os.path.join(RES, 'realistic', 'real_official__skip_%s.json' % g)))
        tv[g] = round(10000 * (b - d['players'][0]['wins'] / d['games']))
    json.dump(tv, open(path, 'w'), indent=1)
    return tv


TV = realistic_values()
WORLD = ['--auctions', '1', '--trading', '1', '--tv', ','.join('%s:%d' % kv for kv in TV.items())]

SPACE = {
    'hold': [0, 100, 200, 400],          # cash kept back on other purchases while a favourite is unsold
    'favbid': [1.0, 1.5, 2.0, 3.0],      # auction limit on a favourite, x list price
    'favmort': [0, 1],                   # mortgage other property to afford a favourite
    'reserve': [0, 25, 50, 100, 150, 200, 300, 400],
    'bid': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    'setbid': [1.0, 1.25, 1.5, 2.0, 3.0],
    'blockbid': [0.0, 0.8, 1.0, 1.5, 2.0, 3.0],
    'build': [2, 3, 4, 5],
    'breserve': [0, 25, 50, 100, 150, 200, 300],
    'bpri': [0, 1],
    'jail': [0, 1, 2],
    'm2b': [0, 1],
    'stretch': [0, 1],
    'trade': ['none', 'fair', 'shark'],
}


def clamp_favs(f, rng):
    on = [i for i, x in enumerate(f) if x]
    while len(on) > 2:
        f[on.pop(rng.randrange(len(on)))] = False
    return f


def random_genome(rng):
    g = {k: rng.choice(v) for k, v in SPACE.items()}
    g['fav'] = clamp_favs([rng.random() < 0.15 for _ in GROUPS], rng)
    return g


def spec(g, name='evolved'):
    parts = ['name=%s' % name, 'buy=all']
    favs = [x for x, on in zip(GROUPS, g['fav']) if on]
    if favs:
        parts.append('fav=%s' % '+'.join(favs))  # must precede favbid/favmort, which override its defaults
    for k in SPACE:
        if k in ('hold', 'favbid', 'favmort') and not favs:
            continue
        parts.append('%s=%s' % (k, g[k]))
    return ';'.join(parts)


def describe(g):
    favs = [x for x, on in zip(GROUPS, g['fav']) if on]
    return {'favourites': favs, **{k: g[k] for k in SPACE}}


def mutate(g, rng, rate=0.15):
    h = json.loads(json.dumps(g))
    for k, v in SPACE.items():
        if rng.random() < rate:
            i = v.index(h[k])
            h[k] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.7 else rng.choice(v)
    h['fav'] = clamp_favs([(not b) if rng.random() < rate / 2 else b for b in h['fav']], rng)
    return h


def cross(a, b, rng):
    c = {k: (a if rng.random() < 0.5 else b)[k] for k in SPACE}
    c['fav'] = clamp_favs([x if rng.random() < 0.5 else y for x, y in zip(a['fav'], b['fav'])], rng)
    return c


def play(strats, games, seed):
    out = subprocess.check_output([BIN, '--strats', '|'.join(strats), '--games', str(games), '--seed', str(seed)] + WORLD)
    return json.loads(out)


def fitness(g, opponents, games, seed):
    d = play([spec(g)] + opponents, games, seed)
    return d['players'][0]['wins'] / d['games']


def ga(opponents, tag, generations=25, pop_size=24, games=16000, seed=7, seeds=()):
    rng = random.Random(seed)
    pop = [json.loads(json.dumps(s)) for s in seeds] + [random_genome(rng) for _ in range(pop_size - len(seeds))]
    history, scored = [], []
    for gen in range(generations):
        scored = sorted(((fitness(g, opponents, games, 1000 * gen + i), g) for i, g in enumerate(pop)), key=lambda x: -x[0])
        best = scored[0]
        history.append({'gen': gen, 'best': best[0], 'median': scored[len(scored) // 2][0], 'genome': best[1]})
        print('%s gen %2d best %.3f median %.3f  %s' % (tag, gen, best[0], scored[len(scored) // 2][0], spec(best[1])), flush=True)
        json.dump(history, open(os.path.join(OUT, '%s_history.json' % tag), 'w'), indent=1)
        nxt = [json.loads(json.dumps(g)) for _, g in scored[:4]]
        while len(nxt) < pop_size:
            a = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            b = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            nxt.append(mutate(cross(a, b, rng), rng))
        pop = nxt
    final = []
    for _, g in scored[:5]:
        final.append({'fitness_200k': fitness(g, opponents, 200000, 99), 'genome': g, 'spec': spec(g), 'plain': describe(g)})
    final.sort(key=lambda x: -x['fitness_200k'])
    json.dump(final, open(os.path.join(OUT, '%s_final.json' % tag), 'w'), indent=1)
    print(tag, 'final', json.dumps(final[0]), flush=True)
    return final


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'stage1':
        ga([TYPICAL] * 3, 'stage1', generations=int(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif stage == 'stage2':
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        champ = json.load(open(os.path.join(OUT, 'stage1_final.json')))[0]['genome']
        lineage = [{'round': 0, 'genome': champ, 'spec': spec(champ), 'plain': describe(champ)}]
        for r in range(1, rounds + 1):
            opp = [spec(champ, 'champ%d' % (r - 1))] * 3
            final = ga(opp, 'stage2_r%d' % r, generations=12, seeds=[champ, mutate(champ, random.Random(r))], seed=100 + r)
            champ = final[0]['genome']
            lineage.append({'round': r, 'genome': champ, 'spec': spec(champ), 'plain': describe(champ),
                            'vs_previous_200k': final[0]['fitness_200k']})
            json.dump(lineage, open(os.path.join(OUT, 'stage2_lineage.json'), 'w'), indent=1)
