"""US nominal GDP through time, and the scale that turns a fortune of any year into today's money by the
size of the economy: value_today = value_then x GDP_now / GDP_then.

Sources: BEA via FRED (GDPA annual from 1929, GDP quarterly SAAR from 1947; data/gdpa.csv, data/gdpq.csv);
1900-1928 from data/us_gdp_1900_1928.csv (research; see its source column). GDP_now is the latest quarter.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_t, _v = [], []
pre = os.path.join(HERE, "data", "us_gdp_1900_1928.csv")
if os.path.exists(pre):
    for r in csv.DictReader(open(pre)):
        _t.append(int(r["year"]) + 0.5); _v.append(float(r["gdp_busd"]))
for r in csv.DictReader(open(os.path.join(HERE, "data", "gdpa.csv"))):
    y = int(r["observation_date"][:4])
    if y < 1947 and r["GDPA"]:
        _t.append(y + 0.5); _v.append(float(r["GDPA"]))
for r in csv.DictReader(open(os.path.join(HERE, "data", "gdpq.csv"))):
    if r["GDP"]:
        y, m = int(r["observation_date"][:4]), int(r["observation_date"][5:7])
        _t.append(y + (m + 0.5) / 12); _v.append(float(r["GDP"]))
T, V = np.array(_t), np.array(_v)
o = np.argsort(T); T, V = T[o], V[o]
NOW_T, NOW = T[-1], V[-1]


def gdp(t):
    """Nominal GDP ($B) at decimal year t (log-linear between data points; flat past the last one)."""
    return np.exp(np.interp(t, T, np.log(V)))


def scale(t):
    """Multiply a nominal fortune at decimal year t by this to get today's money (same share of the economy)."""
    return NOW / gdp(t)
