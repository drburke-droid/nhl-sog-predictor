"""A field of fires that computes.

Every fire has a keeper. There are three kinds of fire:

* input fires: set by hand before the run (the numbers, the message, the key);
* NAND fires: "keep yours lit unless both of your fires are lit";
* memory fires: "when the drum beats, copy your fire" (between beats they hold still).

A circuit is just a list of who watches whom. `Field` builds one, simulates it keeper by keeper
(every keeper takes one step, about 2 seconds, to react), and lays the fires out for drawing.
"""

KEEPER_S = 2.0   # seconds for a keeper to see a change, check their card and move the lid


class Field:
    def __init__(self):
        self.kind = []     # 'in', 'mem' or 'nand'
        self.src = []      # nand: (a, b); mem: (copied fire,); in: ()
        self.init = []     # starting state (inputs and memory are set by hand; nand fires start dark)
        self.tag = []      # which part of the circuit a fire belongs to (for layout)
        self.label = []
        self._nand = {}    # (a, b) -> fire, so the same rule on the same two fires is never built twice
        self._not = {}     # fire -> fire that is its NOT
        self.group = ''

    def _add(self, kind, src, init=0, label=''):
        self.kind.append(kind)
        self.src.append(src)
        self.init.append(init)
        self.tag.append(self.group)
        self.label.append(label)
        return len(self.kind) - 1

    # --- the three kinds of fire ---------------------------------------------------------------
    def input(self, label='', init=0):
        return self._add('in', (), init, label)

    def memory(self, label='', init=0):
        return self._add('mem', [None], init, label)   # what it copies is wired later

    def wire_memory(self, m, source):
        self.src[m][0] = source

    def nand(self, a, b):
        key = (min(a, b), max(a, b))
        if key not in self._nand:
            f = self._add('nand', key)
            self._nand[key] = f
            if a == b:
                self._not[f], self._not[a] = a, f
        return self._nand[key]

    # --- every other rule, made from NAND fires ------------------------------------------------
    def NOT(self, a):
        return self._not[a] if a in self._not else self.nand(a, a)

    def AND(self, a, b):
        return self.NOT(self.nand(a, b))

    def OR(self, a, b):
        return self.nand(self.NOT(a), self.NOT(b))

    def XOR(self, a, b):
        m = self.nand(a, b)
        return self.nand(self.nand(a, m), self.nand(b, m))

    def half_adder(self, a, b):                      # 5 fires: (sum, carry)
        m = self.nand(a, b)
        return self.nand(self.nand(a, m), self.nand(b, m)), self.NOT(m)

    def full_adder(self, a, b, c):                   # 9 fires: (sum, carry)
        m1 = self.nand(a, b)
        x = self.nand(self.nand(a, m1), self.nand(b, m1))
        m2 = self.nand(x, c)
        return self.nand(self.nand(x, m2), self.nand(c, m2)), self.nand(m1, m2)

    # --- counting ------------------------------------------------------------------------------
    def count(self, kind=None):
        return sum(1 for k in self.kind if kind is None or k == kind)

    def depth(self):
        """How many keepers deep each fire is, counting from the input and memory fires."""
        d = [0] * len(self.kind)
        for i, k in enumerate(self.kind):
            if k == 'nand':
                a, b = self.src[i]
                d[i] = 1 + max(d[a], d[b])
        return d

    def beat_steps(self):
        """Keeper steps per drum beat: the longest chain from a memory fire back to a memory fire,
        plus the step the memory keeper takes to copy."""
        d = self.depth()
        return 1 + max(d[s[0]] for k, s in zip(self.kind, self.src) if k == 'mem')

    # --- running it ----------------------------------------------------------------------------
    def evaluate(self, state):
        """Let every NAND keeper settle (inputs and memory held still). Fires are built in order,
        so one pass is enough."""
        for i, k in enumerate(self.kind):
            if k == 'nand':
                a, b = self.src[i]
                state[i] = 1 - (state[a] & state[b])
        return state

    def step(self, state, drum=False):
        """One keeper step (about 2 s): every keeper looks at the fires as they are now and sets
        their own. Changes therefore move one keeper per step, and fires can flicker on the way."""
        new = list(state)
        for i, k in enumerate(self.kind):
            if k == 'nand':
                a, b = self.src[i]
                new[i] = 1 - (state[a] & state[b])
            elif k == 'mem' and drum:
                new[i] = state[self.src[i][0]]
        return new

    def start(self, values):
        """values: {fire: 0/1} for the input and memory fires; everything else starts dark."""
        state = list(self.init)
        for f, v in values.items():
            state[f] = v
        return state

    # --- layout for drawing --------------------------------------------------------------------
    def layout(self, tag_order=None):
        """Columns by depth, one band of rows per part of the circuit. Returns (col, row) per fire."""
        d = self.depth()
        tags = tag_order or list(dict.fromkeys(self.tag))
        col_of = list(d)
        for i, k in enumerate(self.kind):
            if k == 'mem':
                col_of[i] = 0
        pos = [None] * len(self.kind)
        y0 = 0
        for t in tags:
            members = [i for i in range(len(self.kind)) if self.tag[i] == t]
            cols = {}
            for i in members:
                cols.setdefault(col_of[i], []).append(i)
            # order each column by the average row of the fires it watches (keeps lines short)
            height = 0
            row = {}
            for c in sorted(cols):
                def key(i):
                    if self.kind[i] == 'nand':
                        rs = [row[s] for s in self.src[i] if s in row]
                        if rs:
                            return sum(rs) / len(rs)
                    return i
                cols[c].sort(key=key)
                for r, i in enumerate(cols[c]):
                    row[i] = r
                height = max(height, len(cols[c]))
            for i in members:
                pos[i] = (col_of[i], y0 + row[i])
            y0 += height + 1
        return pos

    def export(self, pos=None):
        """The netlist in the compact form the page's simulator reads."""
        return {
            'kind': ''.join({'in': 'i', 'mem': 'm', 'nand': 'n'}[k] for k in self.kind),
            'src': [list(s) for s in self.src],
            'init': self.init,
            'tag': self.tag,
            'pos': pos or self.layout(),
        }
