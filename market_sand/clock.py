"""The film's clock: how many seconds each year gets on screen.

Quiet centuries pass quickly; bubbles, crashes and the last forty years slow down. The pace eases
between the knots (log-linear in seconds per year), so the clock never jerks. HOLDS stop the year for a
beat (the title, the finale).

    C = Clock()
    C.year(seconds)     # decimal year shown at film time `seconds`
    C.duration          # film length in seconds
"""
import numpy as np

KNOTS = [   # (year, seconds per year); the beats each slowdown is for
    (1602.0, 0.9), (1618, 0.6), (1625, 0.15),          # the VOC's first grains, the English East India Company beside it
    (1690, 0.15), (1696, 0.3), (1716, 0.3),             # the Bank of England (1694)
    (1719.0, 3.0), (1721.5, 3.0), (1724, 0.15),         # the Mississippi and South Sea bubbles
    (1795, 0.15), (1800, 0.3), (1805, 0.12),            # the VOC dies (1799)
    (1840, 0.15), (1845, 0.6), (1850, 0.6), (1853, 0.2),  # the railway mania
    (1898, 0.3), (1901, 0.9), (1912, 0.9), (1916, 0.45),  # Standard Oil, US Steel, the 1911 break-up
    (1927, 0.6), (1929.3, 3.0), (1932.5, 2.0), (1936, 0.45),  # the 1929 crash
    (1980, 0.45), (1983, 1.4), (1985, 1.0),             # AT&T broken into eight (1984)
    (1986, 1.6), (1991, 1.6),                           # Japan's bubble
    (1995, 1.8), (1998.5, 3.0), (2001, 3.0), (2003, 1.6),   # the dot-com bubble
    (2007, 2.2), (2009.5, 2.2), (2012, 1.6),            # PetroChina, the financial crisis
    (2019, 2.4), (2026.9, 3.0),                         # Aramco, Covid, the AI boom
]
HOLDS = [(1602.0, 6.0)]     # (year, seconds held) - the title over the first grains
FINALE = 30.0               # seconds after the last year (the flight)


class Clock:
    def __init__(self, end_year):
        k = np.array(KNOTS)
        self.y = np.linspace(k[0, 0], end_year, 40000)
        rate = np.exp(np.interp(self.y, k[:, 0], np.log(k[:, 1])))
        dt = np.diff(self.y) * 0.5 * (rate[1:] + rate[:-1])
        s = np.r_[0.0, np.cumsum(dt)]
        for hy, hs in HOLDS:
            s[self.y > hy] += hs
        self.s = s                      # a hold is a jump in s, so the year stands still across it
        self.end_year = end_year
        self.history = float(s[-1])
        self.duration = self.history + FINALE

    def year(self, sec):
        return float(np.interp(sec, self.s, self.y))

    def seconds(self, year):
        return float(np.interp(year, self.y, self.s))
