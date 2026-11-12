"""The film's clock: how many seconds each year gets on screen.

Quiet centuries pass quickly; bubbles, crashes and the last forty years slow down. The pace eases
between the knots (log-linear in seconds per year), so the clock never jerks. HOLDS stop the year for a
beat (the title, the finale).

    C = Clock()
    C.year(seconds)     # decimal year shown at film time `seconds`
    C.duration          # film length in seconds
"""
import numpy as np

KNOTS = [   # (year, seconds per year)
    (1602.0, 0.9), (1618, 0.6), (1625, 0.15), (1700, 0.15), (1716, 0.4), (1719.0, 3.0), (1721.5, 3.0),
    (1724, 0.15), (1800, 0.15), (1850, 0.25), (1898, 0.35), (1901, 0.9), (1912, 0.9), (1916, 0.45),
    (1927, 0.6), (1929.3, 3.0), (1932.5, 2.0), (1936, 0.45), (1980, 0.45), (1986, 1.6), (1991, 1.6),
    (1995, 2.0), (2026.9, 2.4),
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
