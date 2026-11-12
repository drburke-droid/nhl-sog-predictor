"""Today's US dollars for a market value of any date and currency.

    from money import to_today, NOW
    to_today(6440, "NLG", 1602.5)      # VOC's paid-in capital at par, in millions of today's dollars

The rule, the one most "in today's dollars" figures use:
- from 1800: convert to US dollars at that year's exchange rate, then adjust for US consumer prices
  (CPI) from that date to the latest month;
- before 1800 (there is no US price index): convert to pounds at that time's exchange rate, adjust for
  UK consumer prices to 1800, convert to dollars at the 1800 rate, then follow the US CPI to today.

Series (data/):
  us_cpi_monthly_1913.csv   BLS CPI-U, not seasonally adjusted (FRED CPIAUCNS), monthly from 1913
  us_cpi_annual_1800.csv    Minneapolis Fed, CPI 1800- (1967 = 100), annual; used before 1913
  uk_cpi.csv                Bank of England, "A millennium of macroeconomic data" v3.1, sheet A47: CPI 1209-2016
  usd_per_gbp_annual.csv    same dataset, sheet A33: $/£ 1791-2016 (Officer); later years from FIXED_FX below
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _read(name, conv):
    with open(os.path.join(DATA, name)) as f:
        return [conv(r) for r in csv.DictReader(f) if all(v.strip() for v in r.values())]   # skip gaps (e.g. Oct 2025)


# ---- US CPI: annual 1800-1912 (mid-year), monthly from 1913, on one scale ----
_ann = _read("us_cpi_annual_1800.csv", lambda r: (int(r["year"]) + 0.5, float(r["us_cpi_1967_100"])))
_mon = _read("us_cpi_monthly_1913.csv", lambda r: (int(r["observation_date"][:4]) + (int(r["observation_date"][5:7]) - 0.5) / 12,
                                                   float(r["CPIAUCNS"])))
_link = np.mean([v for t, v in _mon if 1913 <= t < 1914]) / dict(_ann)[1913.5]   # monthly scale / annual scale in 1913
US_T = np.array([t for t, _ in _ann if t < 1913] + [t for t, _ in _mon])
US_P = np.array([v * _link for t, v in _ann if t < 1913] + [v for _, v in _mon])
NOW = float(US_T[-1])          # latest CPI month (decimal year)
US_NOW = float(US_P[-1])

# ---- UK CPI (annual, mid-year) ----
_uk = _read("uk_cpi.csv", lambda r: (int(r["year"]) + 0.5, float(r["uk_cpi_2015_100"])))
UK_T, UK_P = np.array([t for t, _ in _uk]), np.array([v for _, v in _uk])

# ---- exchange rates ----
_gbp = _read("usd_per_gbp_annual.csv", lambda r: (int(r["year"]) + 0.5, float(r["usd_per_gbp"])))
GBP_T, GBP_V = np.array([t for t, _ in _gbp]), np.array([v for _, v in _gbp])

# Units of a currency per pound sterling before 1800 (and other fixed parities), with their sources.
PER_GBP_PRE1800 = {
    # Amsterdam on London, c. 33-37 schellingen Flemish per pound (1 pound Flemish = 6 guilders = 20 s),
    # i.e. about 10-11 guilders per pound through the 17th and 18th centuries (McCusker, Money and Exchange
    # in Europe and America 1600-1775). Replace with annual rates where the research gives them.
    "NLG": 10.8,
}
PER_USD_FIXED = {
    # gold-standard parities 1800-1914 (approximate; the guilder was on silver until 1875)
    "NLG": [(1800, 1914, 12.1 / 4.866)],
    "FRF": [(1803, 1914, 25.22 / 4.866)],      # germinal franc, 25.22 francs per pound
}


def _interp(t, T, V, log=True):
    if log:
        return float(np.exp(np.interp(t, T, np.log(V))))
    return float(np.interp(t, T, V))


def us_cpi(t):
    return _interp(t, US_T, US_P)


def uk_cpi(t):
    return _interp(t, UK_T, UK_P)


def usd_per_gbp(t):
    if t < GBP_T[0]:
        raise ValueError(f"no $/£ rate before {GBP_T[0]:.0f}")
    return _interp(t, GBP_T, GBP_V, log=False)


def usd_per_unit(cur, t):
    """US dollars for one unit of currency `cur` at decimal year t (from 1800)."""
    if cur == "USD":
        return 1.0
    if cur == "GBP":
        return usd_per_gbp(t)
    for a, b, per_usd in PER_USD_FIXED.get(cur, []):
        if a <= t < b:
            return 1.0 / per_usd
    raise ValueError(f"no exchange rate for {cur} at {t:.2f}")


def to_today(value, cur, t):
    """Value (any unit, e.g. millions) in currency `cur` at decimal year t, in today's US dollars (same unit)."""
    if t >= 1800:
        return value * usd_per_unit(cur, t) * US_NOW / us_cpi(t)
    gbp = value if cur == "GBP" else value / PER_GBP_PRE1800[cur]
    gbp_1800 = gbp * uk_cpi(1800.5) / uk_cpi(t)
    return gbp_1800 * usd_per_gbp(1800.5) * US_NOW / us_cpi(1800.5)


if __name__ == "__main__":
    print(f"today = {NOW:.2f}, US CPI {US_NOW:.1f}")
    for v, c, t, what in [(6.44, "NLG", 1602.5, "VOC paid-in capital at par, 1602"),
                          (78, "NLG", 1637.0, "the popular '78 million guilders' VOC, 1637"),
                          (1, "GBP", 1720.5, "one pound, 1720"),
                          (1400, "USD", 1901.2, "US Steel at formation, 1901"),
                          (1, "USD", 1929.7, "one dollar, 1929")]:
        print(f"  {what:45s} -> ${to_today(v, c, t):,.3f} today ({'millions' if v != 1 else 'dollars'})")
