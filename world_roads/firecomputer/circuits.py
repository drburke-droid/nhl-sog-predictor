"""The circuits, built from NAND fires and memory fires only.

* half_adder()    1 + 1 with 5 fires
* adder(n)        add two n-bit numbers (4 bits: 32 fires)
* multiplier(n)   long multiplication in binary (the fire computer note's 600 / 2,608 / 9,510 fires)
* present_ctr()   the PRESENT-128 block cipher in counter mode: scrambles (and unscrambles) 8 letters
"""
from field import Field

# PRESENT (Bogdanov et al., CHES 2007; ISO/IEC 29192-2)
SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


def P(i):
    """The bit shuffle: bit i of the state moves to bit P(i). In fires it costs nothing: it is only
    a choice of which fire each memory keeper watches."""
    return 63 if i == 63 else (i * 16) % 63


def bits(v, n):
    return [(v >> i) & 1 for i in range(n)]


def value(state, fires):
    return sum(state[f] << i for i, f in enumerate(fires))


# --- small things ---------------------------------------------------------------------------------
def half_adder():
    f = Field()
    f.group = 'in'
    a, b = f.input('A'), f.input('B')
    f.group = 'gates'
    s, c = f.half_adder(a, b)
    return f, [a, b], [s, c]          # outputs: ones place, twos place


def adder(n):
    f = Field()
    A, B = [], []
    for i in range(n):
        f.group = f'bit{i}'
        A.append(f.input(f'A{i}'))
        B.append(f.input(f'B{i}'))
    out, carry = [], None
    for i in range(n):
        f.group = f'bit{i}'
        if carry is None:
            s, carry = f.half_adder(A[i], B[i])
        else:
            s, carry = f.full_adder(A[i], B[i], carry)
        out.append(s)
    return f, A, B, out + [carry]


def multiplier(n):
    f = Field()
    f.group = 'in'
    A = [f.input(f'A{i}') for i in range(n)]
    B = [f.input(f'B{i}') for i in range(n)]
    f.group = 'gates'
    # one AND fire for each pair of bits, then rows of adders sum them
    partial = [[f.AND(A[i], B[j]) for i in range(n)] for j in range(n)]
    out, acc = [partial[0][0]], partial[0][1:]
    for j in range(1, n):
        carry, row = None, []
        for i in range(n):
            a, b = partial[j][i], acc[i] if i < len(acc) else None
            if b is None and carry is None:
                row.append(a)
                continue
            if b is None:
                s, carry = f.half_adder(a, carry)
            elif carry is None:
                s, carry = f.half_adder(a, b)
            else:
                s, carry = f.full_adder(a, b, carry)
            row.append(s)
        row.append(carry)
        out.append(row[0])
        acc = row[1:]
    return f, A, B, out + acc


# --- the cipher -----------------------------------------------------------------------------------
def _sbox_cover():
    """The smallest sum-of-products recipe for each of the S-box's 4 output bits: the fewest
    "if these bits are such-and-such" terms, then the fewest bits per term. Each term is a cube:
    4 entries (lowest bit first), each 0, 1 or None (don't care)."""
    import itertools
    cubes = list(itertools.product([0, 1, None], repeat=4))
    covers = lambda c, v: all(c[i] is None or ((v >> i) & 1) == c[i] for i in range(4))
    cover = []
    for ob in range(4):
        on = {v for v in range(16) if (SBOX[v] >> ob) & 1}
        terms = [c for c in cubes if all(v in on for v in range(16) if covers(c, v))]
        for k in range(1, 9):
            fits = [comb for comb in itertools.combinations(terms, k)
                    if all(any(covers(c, v) for c in comb) for v in on)]
            if fits:
                cover.append(min(fits, key=lambda comb: sum(x is not None for c in comb for x in c)))
                break
    return cover


SBOX_COVER = _sbox_cover()


def sbox(f, x):
    """PRESENT's 4-bit substitution from NAND fires (x and the result lowest bit first). Each output
    bit is an OR of a few ANDs: shallow (9 keepers deep), so a drum beat is short."""
    nx = [f.NOT(a) for a in x]
    lit = lambda i, v: x[i] if v else nx[i]
    out = []
    for cover in SBOX_COVER:
        level = []                                   # each entry is NOT(one term)
        for c in cover:
            parts = []
            for pair in ((3, 2), (1, 0)):            # AND the bits in pairs, so pairs are shared
                g = [lit(i, c[i]) for i in pair if c[i] is not None]
                if g:
                    parts.append(f.AND(*g) if len(g) == 2 else g[0])
            level.append(f.nand(*parts) if len(parts) == 2 else f.NOT(parts[0]))
        while True:                                  # OR the terms together, two at a time
            ors = [f.nand(level[i], level[i + 1]) if i + 1 < len(level) else f.NOT(level[i])
                   for i in range(0, len(level), 2)]
            if len(ors) == 1:
                break
            level = [f.NOT(o) for o in ors]
        out.append(ors[0])
    return out


def present_ctr():
    """PRESENT-128 in counter mode.

    Set by hand before the run: the key (128 memory fires), the block number (64 memory fires: the
    "counter"), the round number (5 memory fires, set to 1) and the message (64 input fires: 8 letters).
    After 31 drum beats the output row reads message XOR PRESENT(key, block number). Running the
    scrambled text through an identical field with the same key and block number gives the message back.
    """
    f = Field()
    f.group = 'message'
    msg = [f.input(f'm{i}') for i in range(64)]
    state = []
    for i in range(64):
        f.group = f'slice{i // 4}'
        state.append(f.memory(f's{i}'))
    f.group = 'key'
    key = [f.memory(f'k{i}') for i in range(128)]
    f.group = 'round'
    rnd = [f.memory(f'r{i}', init=int(i == 0)) for i in range(5)]

    # one round: add the round key (the key's top 64 bits), substitute every 4 bits, shuffle the bits
    mixed, subbed = [], []
    for j in range(16):
        f.group = f'slice{j}'
        mixed += [f.XOR(state[i], key[64 + i]) for i in range(4 * j, 4 * j + 4)]
        subbed += sbox(f, mixed[4 * j: 4 * j + 4])
    for i in range(64):
        f.wire_memory(state[P(i)], subbed[i])

    # the key for the next round: rotate 61 places, substitute the top 8 bits, add the round number
    f.group = 'keywork'
    rot = [key[(i - 61) % 128] for i in range(128)]
    rot[120:124] = sbox(f, rot[120:124])
    rot[124:128] = sbox(f, rot[124:128])
    for j in range(5):
        rot[62 + j] = f.XOR(rot[62 + j], rnd[j])
    for i in range(128):
        f.wire_memory(key[i], rot[i])

    # the round number counts up by one each beat
    f.group = 'round'
    carry = rnd[0]
    f.wire_memory(rnd[0], f.NOT(rnd[0]))
    for j in range(1, 5):
        s, carry = f.half_adder(rnd[j], carry)
        f.wire_memory(rnd[j], s)

    # after the last round, the scrambled block is the state plus the last round key ("mixed"),
    # and the output row adds it to the message
    out = []
    for i in range(64):
        f.group = f'out{i // 4}'
        out.append(f.XOR(mixed[i], msg[i]))
    return f, msg, state, key, rnd, out


def present_ref(pt, key, rounds=31, kbits=128):
    """Plain software PRESENT, to check the fires against."""
    s, k, M = pt, key, (1 << kbits) - 1
    for r in range(1, rounds + 2):
        s ^= k >> (kbits - 64)
        if r == rounds + 1:
            break
        s = sum(SBOX[(s >> (4 * j)) & 15] << (4 * j) for j in range(16))
        s = sum(((s >> i) & 1) << P(i) for i in range(64))
        k = ((k << 61) | (k >> (kbits - 61))) & M
        k = (k & ~(15 << (kbits - 4))) | (SBOX[k >> (kbits - 4)] << (kbits - 4))
        if kbits == 128:
            k = (k & ~(15 << 120)) | (SBOX[(k >> 120) & 15] << 120)
            k ^= r << 62
        else:
            k ^= r << 15
    return s


def present_run(f, msg, state, key, rnd, out, m, nonce, k, watch=None):
    """Run the cipher field keeper by keeper: 31 beats, then let the output row settle.
    Returns the output and the number of keeper steps it took."""
    steps_per_beat = f.beat_steps()
    v = {}
    for fires, val, n in ((msg, m, 64), (state, nonce, 64), (key, k, 128), (rnd, 1, 5)):
        v.update(zip(fires, bits(val, n)))
    s = f.start(v)
    t = 0
    for beat in range(31):
        for _ in range(steps_per_beat - 1):
            s = f.step(s); t += 1
        s = f.step(s, drum=True); t += 1
        if watch:
            watch(beat, s)
    d = f.depth()
    for _ in range(max(d[o] for o in out)):
        s = f.step(s); t += 1
    return value(s, out), t
