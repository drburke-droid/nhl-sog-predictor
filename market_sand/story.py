"""Every company's sand through time, from the merged data (data/companies.csv, values_today.csv, events.csv).

    S = Story()
    S.grains(t)        # {id: grains at decimal year t}, one grain = GRAIN_USD_M million of today's dollars
    S.peak_grains[id]  # the most it ever holds
    radius(n)          # a pile's base radius in cells for n grains

Between data points a value moves log-linearly (steady growth), except across a gap of BRIDGE_MIN years or
more where the company's home stock market has an index: there it follows the index, tilted so it still meets
both known values exactly (so a company with no 1930s data still falls in 1930-32 the way its market did).
Indices: the Bank of England's spliced UK share price index (monthly, 1709-2017) for London, and for Paris,
Amsterdam and the rest of Europe before 1914; Shiller's S&P composite (monthly, 1871-) for New York.
A company's sand pours in over RAMP years
up to its first value, and a company that fails drains over RAMP years after its last; one that is bought
or merged hands its sand to the buyer at the deal (the buyer's own series already includes it).
"""
import csv, os
import numpy as np
import money

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("MARKET_SAND_DATA", os.path.join(HERE, "data"))
GRAIN_USD_M = 10.0      # one grain = $10 million in today's dollars
SLOPE = 1.0             # pile flanks: levels per cell (a ~38 degree sand slope)
RAMP = 0.4              # years to pour in a newcomer / drain a failure
HANDOVER = ("acquired", "merged", "broken_up")
BRIDGE_MIN = 3.0        # years: gaps this long or longer follow the home market's index
INDEX_OF = {            # district -> (index, valid until)
    "New York": ("us", 9999), "London": ("uk", 9999), "Johannesburg": ("uk", 9999),
    "Paris": ("uk", 1914), "Amsterdam": ("uk", 1914), "Europe": ("uk", 1914),
}


def _index(name, col, cur):
    """A monthly share index in today's-dollar terms (deflated the same way as the values): (t, log level)."""
    t, v = [], []
    with open(os.path.join(HERE, "data", name)) as f:
        for r in csv.DictReader(f):
            if r[col]:
                tt = int(r["month"][:4]) + (int(r["month"][5:7]) - 0.5) / 12
                t.append(tt); v.append(np.log(float(r[col]) * money.to_today(1.0, cur, tt)))
    return np.array(t), np.array(v)


INDICES = {"uk": _index("uk_share_index_monthly.csv", "uk_share_index_1962_100", "GBP"),
           "us": _index("us_share_index_monthly.csv", "sp_composite", "USD")}


INDEX_FOR = {"royal_dutch_shell": ("uk", 9999)}   # company overrides: Shell traded in London as well as Amsterdam


def bridge(T, V, district, cid=None):
    """Dense (t, value) through the known points, following the home index across long gaps."""
    ix = INDEX_FOR.get(cid) or INDEX_OF.get(district)
    if ix is None or len(T) < 2:
        return T, V
    IT, IV = INDICES[ix[0]]
    out_t, out_v = [T[0]], [V[0]]
    for a, b, va, vb in zip(T[:-1], T[1:], V[:-1], V[1:]):
        if b - a >= BRIDGE_MIN and IT[0] <= a and b <= min(IT[-1], ix[1]) and va > 0 and vb > 0:
            ts = np.arange(a, b, 1 / 12)[1:]
            ia, ib = np.interp(a, IT, IV), np.interp(b, IT, IV)
            lv = np.log(va) + (np.interp(ts, IT, IV) - ia) + (ts - a) / (b - a) * ((np.log(vb) - np.log(va)) - (ib - ia))
            out_t.extend(ts); out_v.extend(np.exp(lv))
        out_t.append(b); out_v.append(vb)
    return np.array(out_t), np.array(out_v)

INDUSTRIES = {   # id -> (legend label, colour); colours from the Sandglass palette
    "trade":         ("Trading companies & shipping", "#f29a46"),
    "finance":       ("Banks & finance",              "#f1cb5a"),
    "rail":          ("Railways & canals",            "#e8606b"),
    "oil":           ("Oil & gas",                    "#b97a4c"),
    "mining_metals": ("Steel & mining",               "#dfe5ee"),
    "autos":         ("Cars",                         "#c970c9"),
    "industrial":    ("Electrical & industrial",      "#9a6ae0"),
    "chem_pharma":   ("Chemicals & medicines",        "#62c486"),
    "consumer":      ("Shops, food & goods",          "#ec5e93"),
    "telecom_media": ("Telegraph, phones & media",    "#93dbe8"),
    "computing":     ("Computers & software",         "#4d94e8"),
    "internet":      ("Internet",                     "#38bfae"),
    "semis":         ("Chips",                        "#b4dc48"),
}


def radius(n):
    """Base radius (cells) of a cone of n grains with flanks of SLOPE levels per cell."""
    return (3.0 * np.maximum(n, 0) / (np.pi * SLOPE)) ** (1 / 3)


def _rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


class Story:
    def __init__(self, now=None):
        self.companies = {r["id"]: r for r in _rows("companies.csv")}
        pts = {}
        for r in _rows("values_today.csv"):
            pts.setdefault(r["id"], []).append((float(r["t"]), float(r["value_today_musd"])))
        self.T, self.V = {}, {}
        self.known = {}
        for k, p in pts.items():
            p.sort()
            T, V = np.array([a for a, _ in p]), np.array([b for _, b in p]) / GRAIN_USD_M
            self.known[k] = T
            self.T[k], self.V[k] = bridge(T, V, self.companies[k]["district"], k)
        self.events = [dict(r, t=float(r["t"])) for r in _rows("events.csv")] if os.path.exists(os.path.join(DATA, "events.csv")) else []
        self.now = now if now is not None else max(T[-1] for T in self.T.values())
        self.start, self.end = {}, {}
        for k, c in self.companies.items():
            self.start[k] = self.T[k][0]
            dead = c["fate"] not in ("alive", "")
            self.end[k] = self.T[k][-1] if dead else np.inf
        self.peak_grains = {k: float(self.V[k].max()) for k in self.T}
        # a company's sand goes to a buyer only if the buyer is on the table; otherwise it drains
        self.hands_over = {k for k, c in self.companies.items() if c["fate"] in HANDOVER
                           and any(s in self.T for s in c.get("successors", "").split(";") if s)}

    def grains_one(self, k, t):
        T, V = self.T[k], self.V[k]
        if t < T[0] - RAMP:
            return 0.0
        if t < T[0]:
            return V[0] * (t - (T[0] - RAMP)) / RAMP
        if t > self.end[k]:
            if k in self.hands_over:
                return 0.0
            return V[-1] * max(0.0, 1 - (t - self.end[k]) / RAMP)
        return float(np.exp(np.interp(t, T, np.log(np.maximum(V, 1e-9)))))

    def grains(self, t):
        return {k: self.grains_one(k, t) for k in self.T}

    def peak_until(self, k, t):
        """The most sand company k has held up to year t (its tide mark)."""
        T, V = self.T[k], self.V[k]
        m = T <= t
        v = V[m].max() if m.any() else 0.0
        return max(v, self.grains_one(k, t))
