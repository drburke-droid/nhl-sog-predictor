"""Evolve a Monopoly strategy with a genetic algorithm, then let the winners evolve against each other.

Stage 1: fitness = win rate against three Yes-Men (official rules with deals).
Stage 2 (arms race): fitness = win rate against three copies of the current champion; the champion
is replaced whenever a challenger beats it, so the opponents keep getting stronger.

    python3 evolve.py stage1 [generations]
    python3 evolve.py stage2 [rounds]
Results go to results/evolve/.
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'monopoly')
OUT = os.path.join(HERE, 'results', 'evolve')
os.makedirs(OUT, exist_ok=True)
BUYGROUPS = ['brown', 'lightblue', 'pink', 'orange', 'red', 'yellow', 'green', 'darkblue', 'rr', 'utilities']
_TV = json.load(open(os.path.join(HERE, 'results', 'learned_values.json')))
# official rules with deals; sharks price sets with the values the simulation measured
WORLD = ['--auctions', '1', '--trading', '1', '--buyouts', '1', '--tv', ','.join('%s:%d' % kv for kv in _TV.items())]

SPACE = {
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
    'sell': [1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
    'buyp': [1.5, 2.0, 2.5, 3.0, 4.0, 6.0],
    'trade': ['fair', 'shark'],
}


def random_genome(rng):
    g = {k: rng.choice(v) for k, v in SPACE.items()}
    g['buy'] = [rng.random() < 0.7 for _ in BUYGROUPS]
    return g


def spec(g, name='evolved'):
    buy = '+'.join(b for b, on in zip(BUYGROUPS, g['buy']) if on) or 'none'
    parts = ['name=%s' % name, 'buy=%s' % buy]
    parts += ['%s=%s' % (k, g[k]) for k in SPACE]
    return ';'.join(parts)


def mutate(g, rng, rate=0.15):
    h = json.loads(json.dumps(g))
    for k, v in SPACE.items():
        if rng.random() < rate:
            i = v.index(h[k])
            h[k] = v[max(0, min(len(v) - 1, i + rng.choice([-1, 1])))] if rng.random() < 0.7 else rng.choice(v)
    h['buy'] = [(not b) if rng.random() < rate / 2 else b for b in h['buy']]
    return h


def cross(a, b, rng):
    c = {k: (a if rng.random() < 0.5 else b)[k] for k in SPACE}
    c['buy'] = [x if rng.random() < 0.5 else y for x, y in zip(a['buy'], b['buy'])]
    return c


def fitness(g, opponents, games, seed):
    out = subprocess.check_output([BIN, '--strats', '|'.join([spec(g)] + opponents), '--games', str(games), '--seed', str(seed)] + WORLD)
    d = json.loads(out)
    return d['players'][0]['wins'] / d['games']


def ga(opponents, tag, generations=25, pop_size=24, games=16000, seed=7, seeds=()):
    rng = random.Random(seed)
    pop = [json.loads(json.dumps(s)) for s in seeds] + [random_genome(rng) for _ in range(pop_size - len(seeds))]
    history = []
    scored = []
    for gen in range(generations):
        scored = sorted(((fitness(g, opponents, games, 1000 * gen + i), g) for i, g in enumerate(pop)), key=lambda x: -x[0])
        best = scored[0]
        history.append({'gen': gen, 'best': best[0], 'median': scored[len(scored) // 2][0], 'genome': best[1]})
        print('%s gen %2d best %.3f median %.3f  %s' % (tag, gen, best[0], scored[len(scored) // 2][0], spec(best[1])), flush=True)
        json.dump(history, open(os.path.join(OUT, '%s_history.json' % tag), 'w'), indent=1)
        elite = [g for _, g in scored[:4]]
        nxt = [json.loads(json.dumps(g)) for g in elite]
        while len(nxt) < pop_size:
            a = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            b = max(rng.sample(scored, 3), key=lambda x: x[0])[1]
            nxt.append(mutate(cross(a, b, rng), rng))
        pop = nxt
    # re-test the final top five on many more games
    final = []
    for f, g in scored[:5]:
        f2 = fitness(g, opponents, 200000, 99)
        final.append({'fitness_200k': f2, 'genome': g, 'spec': spec(g)})
    final.sort(key=lambda x: -x['fitness_200k'])
    json.dump(final, open(os.path.join(OUT, '%s_final.json' % tag), 'w'), indent=1)
    print(tag, 'final', json.dumps(final[0]), flush=True)
    return final


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'stage1':
        gens = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        ga(['everything'] * 3, 'stage1', generations=gens)
    elif stage == 'stage2':
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        champ = json.load(open(os.path.join(OUT, 'stage1_final.json')))[0]['genome']
        lineage = [{'round': 0, 'genome': champ, 'spec': spec(champ)}]
        for r in range(1, rounds + 1):
            opp = [spec(champ, 'champ%d' % (r - 1))] * 3
            final = ga(opp, 'stage2_r%d' % r, generations=12, seeds=[champ, mutate(champ, random.Random(r))], seed=100 + r)
            champ = final[0]['genome']
            lineage.append({'round': r, 'genome': champ, 'spec': spec(champ), 'vs_previous_200k': final[0]['fitness_200k']})
            json.dump(lineage, open(os.path.join(OUT, 'stage2_lineage.json'), 'w'), indent=1)
