"""Build every circuit, check it, count it, and build the page.

    python build.py    -> ../media/firecomputer/fire_computer.json (the numbers)
                          ../media/firecomputer/fire_computer.html (page.html with the circuits inlined)

Checks: the half adder and the 4-bit adder on every input; the multipliers on 200 random products
each; the cipher keeper by keeper on the four published PRESENT-128 test vectors and 20 random
blocks against plain software PRESENT, and a message scrambled and unscrambled through the field.
"""
import json
import math
import os
import random

from circuits import (adder, bits, counter, fma_parts, half_adder, multiplier, present_ctr, present_ref, present_run,
                      value)
from field import KEEPER_S, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(HERE, '..', 'media', 'firecomputer')
CAPITALS = os.path.join(HERE, '..', 'media', 'signal', 'signal_fires_capitals.json')

HOP_S = 10        # a lidded mountain fire: the next keeper sees the lid move and copies it (assumed)
SYMBOL_S = 30     # one hex digit on the mountain chain: shown for 20 s, then 10 s dark
NONCE_BASE = 1588 << 16   # block number = 1588 (the Armada beacons) * 65,536 + the block's position

PRESENT_TEST_VECTORS = [   # PRESENT-128: (plaintext, key, ciphertext)
    (0, 0, 0x96db702a2e6900af),
    (0, 2 ** 128 - 1, 0x13238c710272a5d8),
    (2 ** 64 - 1, 0, 0x3c6019e5e5edd563),
    (2 ** 64 - 1, 2 ** 128 - 1, 0x628d9fbd4218e5b4),
]


def settle(f, values, outs):
    """Run a field with no memory keeper by keeper until it stops changing; return (output, steps)."""
    s = f.start(values)
    for t in range(1, 10 ** 6):
        n = f.step(s)
        if n == s:
            return value(s, outs), t - 1
        s = n


# --- layout ----------------------------------------------------------------------------------------
def block(f, fires, width, depth, placed=None):
    """Lay out a group of fires in rows by depth (inputs and memory on top), wrapping each row at
    `width`, ordered so each fire sits under the fires it watches. Returns {fire: (x, y)}, w, h."""
    placed = placed if placed is not None else {}
    rows = {}
    for i in fires:
        rows.setdefault(0 if f.kind[i] != 'nand' else depth[i], []).append(i)
    pos, y, w = {}, 0, 0
    for d in sorted(rows):
        def key(i):
            if f.kind[i] == 'nand':
                xs = [(pos.get(s) or placed.get(s) or (None,))[0] for s in f.src[i]]
                xs = [x for x in xs if x is not None]
                if xs:
                    return sum(xs) / len(xs)
            return -i
        row = sorted(rows[d], key=key)
        for k in range(0, len(row), width):
            chunk = row[k:k + width]
            off = (width - len(chunk)) / 2
            for j, i in enumerate(chunk):
                pos[i] = (off + j, y)
            w = max(w, len(chunk))
            y += 1
    return pos, width, y


def place(pos, dx, dy, into):
    for i, (x, y) in pos.items():
        into[i] = (round(x + dx, 2), round(y + dy, 2))


def layout_half_adder(f):
    """A and B on top; the shared NAND fire; its two helpers; the twos and ones fires at the bottom."""
    a, b = 0, 1
    m = f.nand(a, b)
    # the two helpers sit wide, so neither A -> helper nor B -> helper points straight at the ones fire,
    # and the twos fire hangs below the shared keeper, clear of both helpers
    spot = {a: (0.2, 0), b: (3.8, 0), m: (2, 1), f.nand(a, m): (0.4, 2.1), f.nand(b, m): (3.9, 2.1),
            f.NOT(m): (1.6, 3.4), f.nand(f.nand(a, m), f.nand(b, m)): (3.0, 3.4)}
    assert len(spot) == len(f.kind)
    return [spot[i] for i in range(len(f.kind))]


def layout_adder(f, n):
    """Left to right in the order the keepers react; one band per bit, the highest bit on top."""
    depth = f.depth()
    out = {}
    for i in range(n):
        members = [k for k in range(len(f.kind)) if f.tag[k] == f'bit{i}']
        cols = {}
        for k in members:
            cols.setdefault(depth[k], []).append(k)
        for d, ks in cols.items():
            for j, k in enumerate(sorted(ks)):
                out[k] = (d, (n - 1 - i) * 3.2 + j + (2 - len(ks)) / 2 + 0.5)
    return [out[k] for k in range(len(f.kind))]


def layout_full_adder(f):
    """A, B and the carry on top; the 9 keepers in rows by depth; the sum and the carry out at the bottom."""
    spot = [(0, 0), (2, 0), (4.2, 0), (1, 1), (0.35, 2), (1.65, 2), (1, 3), (2.6, 4), (1.9, 5), (3.3, 5), (2.6, 6.1), (0.2, 6.1)]
    assert len(spot) == len(f.kind) and f.kind[10] == f.kind[11] == 'nand'
    return spot


def layout_adder_columns(f, n, outs):
    """The whole adding chain as written on paper: one column per place (1s on the right), the two
    numbers' fires on top, each column's keepers below them, and the answer along the bottom
    (the last carry becomes the answer's extra left-hand fire)."""
    depth = f.depth()
    W = 5.0
    cx = lambda i: (n - i) * W            # column i (the 2**i place); column n is the extra fire
    row = lambda d: 1.2 + d * 0.95        # the two numbers sit on rows 0 and 0.9, the keepers below
    pos = {}
    for i in range(n):
        rows = {}
        for k in range(len(f.kind)):
            if f.tag[k] == f'bit{i}' and f.kind[k] == 'nand' and k not in outs:
                rows.setdefault(depth[k], []).append(k)
        carry = f.carries[i]
        for d, ks in rows.items():
            ks = sorted(k for k in ks if k != carry)
            for j, k in enumerate(ks):
                pos[k] = (cx(i) + 0.4 + (j - (len(ks) - 1) / 2) * 1.2, row(d))
        if carry not in outs:                 # the carry out sits at the column's left edge, next to the column it feeds
            pos[carry] = (cx(i) - 1.9, row(depth[carry]))
        a, b = [k for k in range(len(f.kind)) if f.tag[k] == f'bit{i}' and f.kind[k] == 'in']
        pos[a], pos[b] = (cx(i), 0), (cx(i), 0.9)
    for i, o in enumerate(outs):
        pos[o] = (cx(i), row(max(depth)) + 1.2)
    return [pos[k] for k in range(len(f.kind))]


def layout_counter(f, n):
    """Memory fires in a row on top (the highest bit on the left), each bit's keepers below it."""
    depth = f.depth()
    out = {}
    for i in range(n):
        cols = {}
        for k in range(len(f.kind)):
            if f.tag[k] == f'bit{i}':
                cols.setdefault(0 if f.kind[k] == 'mem' else depth[k], []).append(k)
        for d, ks in cols.items():
            for j, k in enumerate(sorted(ks)):
                out[k] = ((n - 1 - i) * 3 + j - (len(ks) - 1) / 2, d * 1.1)
    return [out[k] for k in range(len(f.kind))]


def layout_cipher(f, msg, key, rnd):
    """Sixteen columns, one per 4 bits of the block (the highest on the left). Each column has its
    4 state fires on top, then the key mixing and the substitution, then the message and the
    output row. The key machinery is a block on the right."""
    depth = f.depth()
    W, GAP = 7, 1.6
    pos = {}
    tops = []
    for j in range(16):
        x0 = (15 - j) * (W + GAP)
        members = [k for k in range(len(f.kind)) if f.tag[k] == f'slice{j}']
        p, _, h = block(f, members, W, depth)
        place(p, x0, 0, pos)
        tops.append(h)
    y_out = max(tops) + 1.5
    for j in range(16):
        x0 = (15 - j) * (W + GAP)
        members = [k for k in range(len(f.kind)) if f.tag[k] == f'out{j}'] + msg[4 * j:4 * j + 4]
        p, _, _ = block(f, members, W, depth, placed=pos)
        # the output row reads straight down from the column above it
        rows = sorted({y for _, y in p.values()})
        place({k: (x, rows.index(y)) for k, (x, y) in p.items()}, x0, y_out, pos)
    x_key = 16 * (W + GAP) + 2
    kw = 16
    members = [k for k in range(len(f.kind)) if f.tag[k] in ('key', 'keywork', 'round')]
    p, _, _ = block(f, members, kw, depth)
    place(p, x_key, 0, pos)
    return [pos[k] for k in range(len(f.kind))]


# --- the mountain chain ----------------------------------------------------------------------------
def chain_to(capital):
    d = json.load(open(CAPITALS))
    fires = d['fires']
    c = next(c for c in d['capitals'] if c['capital'] == capital)
    path, i = [], c['fire']
    while i != -1:
        path.append(i)
        i = fires[i]['parent']
    path = path[::-1]
    pts, km = [], 0.0
    for a, b in zip([None] + path, path):
        if a is not None:
            la1, lo1, la2, lo2 = map(math.radians, (fires[a]['lat'], fires[a]['lon'], fires[b]['lat'], fires[b]['lon']))
            km += 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                                                 math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
        F = fires[b]
        pts.append({'lat': round(F['lat'], 4), 'lon': round(F['lon'], 4), 'h_m': round(F['h_m']), 'km': round(km, 1)})
    return pts


# --- the steps a person would write, doing the cipher by hand ---------------------------------------
def by_hand_steps():
    per_round = 64 + 16 + 64 + 128 + 2 + 5   # add key bits, look up 16 S-boxes, re-copy 64 bits in the
    #                                        shuffled order, re-copy the 128-bit key rotated, 2 key
    #                                        look-ups, 5 bits of round number
    return 31 * per_round + 64 + 64         # + the last round key + adding the keystream to the message


def fmt_time(s):
    s = round(s)
    if s < 60:
        return f'{s} s'
    if s < 3600:
        return f'{s // 60} min {s % 60:02d} s'
    return f'{s // 3600} h {s % 3600 // 60:02d} min'


def main():
    random.seed(1588)
    stats = {'keeper_s': KEEPER_S, 'hop_s': HOP_S, 'symbol_s': SYMBOL_S, 'circuits': []}

    def row(name, f, outs, extra=None):
        d = f.depth()
        r = {'name': name, 'fires': f.count(), 'nand': f.count('nand'), 'memory': f.count('mem'),
             'inputs': f.count('in'), 'layers': max(d[o] for o in outs)}
        r.update(extra or {})
        stats['circuits'].append(r)
        print(f"{name:34s} {r['fires']:6,d} fires ({r['nand']:,} NAND, {r['memory']} memory, {r['inputs']} input)  "
              f"{r['layers']} layers", (extra or {}).get('note', ''))
        return r

    # 1 + 1
    ha, ha_in, ha_out = half_adder()
    for a in (0, 1):
        for b in (0, 1):
            assert settle(ha, {ha_in[0]: a, ha_in[1]: b}, ha_out)[0] == a + b
    row('half adder (1 + 1)', ha, ha_out)

    # adding two 4-bit numbers
    ad, adA, adB, ad_out = adder(4)
    worst = 0
    for a in range(16):
        for b in range(16):
            v = dict(zip(adA, bits(a, 4)))
            v.update(zip(adB, bits(b, 4)))
            got, t = settle(ad, v, ad_out)
            assert got == a + b
            worst = max(worst, t)
    row('4-bit adder', ad, ad_out, {'worst_steps': worst})

    for n in (8, 16, 32):
        f, A, B, out = adder(n)
        for _ in range(200):
            a, b = random.randrange(2 ** n), random.randrange(2 ** n)
            v = dict(zip(A, bits(a, n)))
            v.update(zip(B, bits(b, n)))
            assert value(f.evaluate(f.start(v)), out) == a + b
        row(f'{n}-bit adder', f, out)
    for n in (8, 16, 30):
        f, A, B, out = multiplier(n)
        for _ in range(200):
            a, b = random.randrange(2 ** n), random.randrange(2 ** n)
            v = dict(zip(A, bits(a, n)))
            v.update(zip(B, bits(b, n)))
            assert value(f.evaluate(f.start(v)), out) == a * b
        row(f'{n}-bit x {n}-bit multiplier', f, out)

    # one column of adding: two bits and a carry
    fa = Field()
    fa_in = [fa.input('A'), fa.input('B'), fa.input('C')]
    fa_out = list(fa.full_adder(*fa_in))
    for v in range(8):
        assert settle(fa, dict(zip(fa_in, bits(v, 3))), fa_out)[0] == sum(bits(v, 3))
    row('full adder (one column, with carry)', fa, fa_out)

    # a 3-bit counter, keeper by keeper with the drum
    ct, ct_mem = counter(3)
    s = ct.start({})
    for beat in range(1, 17):
        for _ in range(ct.beat_steps() - 1):
            s = ct.step(s)
        s = ct.step(s, drum=True)
        assert value(s, ct_mem) == beat % 8
    row('3-bit counter', ct, ct_mem, {'beat_steps': ct.beat_steps()})

    # the cipher, keeper by keeper
    cf, msg, state, key, rnd, out = present_ctr()
    steps = None
    for pt, k, want in PRESENT_TEST_VECTORS:
        got, steps = present_run(cf, msg, state, key, rnd, out, 0, pt, k)
        assert got == want, (hex(got), hex(want))
    for _ in range(20):
        m, pt, k = random.getrandbits(64), random.getrandbits(64), random.getrandbits(128)
        got, _ = present_run(cf, msg, state, key, rnd, out, m, pt, k)
        assert got == m ^ present_ref(pt, k)
    sample_key = 0x0f1e2d3c4b5a69788796a5b4c3d2e1f0
    text = 'UNPLUG EVERYTHING AT DAWN'.ljust(32)
    blocks = [int.from_bytes(text[i:i + 8].encode(), 'big') for i in range(0, len(text), 8)]
    secret = [present_run(cf, msg, state, key, rnd, out, b, NONCE_BASE + i, sample_key)[0] for i, b in enumerate(blocks)]
    back = [present_run(cf, msg, state, key, rnd, out, c, NONCE_BASE + i, sample_key)[0] for i, c in enumerate(secret)]
    assert back == blocks
    beat = cf.beat_steps()
    row('PRESENT-128 cipher (8 letters)', cf, out,
        {'beat_steps': beat, 'beats': 31, 'block_steps': steps,
         'note': f'{beat} steps a beat ({beat * KEEPER_S:.0f} s); a block takes {steps} steps = {fmt_time(steps * KEEPER_S)}'})

    stats['circuits'].sort(key=lambda r: r['fires'])
    # how far to a supercomputer: one calculation (a multiply-and-add in double precision) in fires
    parts = fma_parts()
    nand_unit = sum(n for n, _ in parts.values())
    deep = parts['multiply the digits (long multiplication)'][1] + parts['add the digits'][1] + 2 * parts['line up the two numbers'][1]
    line_fires = 2 * nand_unit          # a factory line: a memory keeper behind every keeper
    stats['super'] = {
        'parts': [{'name': k, 'fires': v[0], 'deep': v[1]} for k, v in parts.items()],
        'unit_nand': nand_unit, 'unit_deep': deep, 'line_fires': line_fires,
        'line_note': 'memory keeper behind every keeper; a beat is 2 reactions; 2 calculations per beat',
        'world_pop': 8.2e9,
        'targets': [
            {'id': 'lineshine', 'name': 'LineShine', 'what': 'No. 1 on the June 2026 TOP500, Shenzhen', 'flops': 2.1984e18},
            {'id': 'elcap', 'name': 'El Capitan', 'what': 'No. 2, Lawrence Livermore', 'flops': 1.809e18},
            {'id': 'cray1', 'name': 'Cray-1 (1976)', 'what': 'the first famous supercomputer, at its peak', 'flops': 160e6},
            {'id': 'eniac', 'name': 'ENIAC (1946)', 'what': 'the first electronic general-purpose computer: about 385 multiplications of 10-digit numbers a second', 'flops': 385},
        ],
        'switches': [
            {'id': 'fire', 'name': 'fires', 'one': 'fire', 'react_s': 2.0, 'spacing_m': 5.0, 'person': True, 'what': 'a lidded fire and its keeper'},
            {'id': 'flag', 'name': 'flags', 'one': 'flag', 'react_s': 1.0, 'spacing_m': 1.0, 'person': True, 'what': 'a soldier with a flag, as in The Three-Body Problem'},
            {'id': 'torch', 'name': 'flashlights', 'one': 'flashlight', 'react_s': 0.5, 'spacing_m': 0.6, 'person': True, 'what': 'a person with a flashlight, packed like a stadium crowd'},
            {'id': 'relay', 'name': 'relays', 'one': 'relay', 'react_s': 0.01, 'spacing_m': 0.05, 'person': False, 'what': 'a 1940s telephone relay, clicking in about a hundredth of a second'},
            {'id': 'tube', 'name': 'vacuum tubes', 'one': 'vacuum tube', 'react_s': 1e-5, 'spacing_m': 0.1, 'person': False, 'what': "a vacuum tube, at ENIAC's 100,000 steps a second"},
        ],
    }
    top = stats['super']['targets'][0]['flops']
    n_fire = top * 2.0 * line_fires
    print(f"\nOne calculation: {nand_unit:,} NAND keepers, {deep} deep ({fmt_time(deep * KEEPER_S)} per answer); "
          f"factory line {line_fires:,} fires, 2 calculations per 4 s")
    print(f"To match LineShine (2.198e18/s): {n_fire:.2e} fires; everyone on Earth tending one fire = "
          f"{8.2e9 / (2.0 * line_fires):,.0f} calculations a second")
    hand = by_hand_steps()
    route = chain_to('Paris')
    stats['cipher'] = {
        'fires': cf.count(), 'beat_s': beat * KEEPER_S, 'block_s': steps * KEEPER_S,
        'by_hand_steps': hand, 'by_hand_s_at_1_per_s': hand,
        'wire_block_s': 16 * SYMBOL_S, 'route': 'Bletchley Park to Paris', 'hops': len(route) - 1,
        'route_km': route[-1]['km'], 'latency_s': (len(route) - 1) * HOP_S,
        'sample': {'text': text.rstrip(), 'key': f'{sample_key:032x}', 'nonce_base': NONCE_BASE,
                   'secret': [f'{c:016x}' for c in secret]},
        'test_vectors': [[f'{p:016x}', f'{k:032x}', f'{c:016x}'] for p, k, c in PRESENT_TEST_VECTORS],
    }
    print(f"\nBy hand: {hand:,} written steps per 8 letters ({fmt_time(hand)} at one step a second, no breaks)")
    print(f"Field: {fmt_time(steps * KEEPER_S)} per 8 letters; the mountain chain sends 8 letters in "
          f"{fmt_time(16 * SYMBOL_S)}; Bletchley Park to Paris is {len(route) - 1} hops, {route[-1]['km']:.0f} km, "
          f"{(len(route) - 1) * HOP_S} s for the first digit to arrive")
    print('Sample:', repr(text.rstrip()), '->', ' '.join(stats['cipher']['sample']['secret']))

    page_data = {
        'stats': stats,
        'act1': {'net': ha.export(layout_half_adder(ha)), 'inputs': ha_in, 'outputs': ha_out},
        'chain': {'net': ad.export(layout_adder_columns(ad, 4, ad_out)), 'A': adA, 'B': adB, 'outputs': ad_out, 'carries': ad.carries},
        'full': {'net': fa.export(layout_full_adder(fa)), 'inputs': fa_in, 'outputs': fa_out},
        'counter': {'net': ct.export(layout_counter(ct, 3)), 'mem': ct_mem},
        'act2': {'net': ad.export(layout_adder(ad, 4)), 'A': adA, 'B': adB, 'outputs': ad_out},
        'act3': {'net': cf.export(layout_cipher(cf, msg, key, rnd)), 'msg': msg, 'state': state,
                 'key': key, 'round': rnd, 'outputs': out, 'route': route},
    }
    os.makedirs(MEDIA, exist_ok=True)
    json.dump(stats, open(os.path.join(MEDIA, 'fire_computer.json'), 'w'), indent=1)
    page = open(os.path.join(HERE, 'page.html')).read()
    sim = open(os.path.join(HERE, 'fire_sim.js')).read()
    data = json.dumps(page_data, separators=(',', ':'))
    page = page.replace('/*FIRE_SIM*/', sim).replace('/*FIRE_DATA*/null', data)
    open(os.path.join(MEDIA, 'fire_computer.html'), 'w').write(page)
    print(f"\nWrote media/firecomputer/fire_computer.html ({len(page) / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
