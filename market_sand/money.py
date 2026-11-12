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
  usd_per_gbp_annual.csv    same dataset, sheet A33: $/£ 1791-2016 (Officer); used to 1970
  fx_usd_per_unit_monthly_1971.csv   FRED (Federal Reserve H.10/G.5) monthly rates from 1971: GBP, JPY, EUR,
                            CHF, CNY, KRW, TWD, HKD, BRL, AUD, CAD; FRF and NLG to 2001
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

# ---- before 1800: units of a currency per pound sterling (research/a_to1825/notes.md, section 2) ----
# guilders: Amsterdam on London was about 33-36 schellingen Flemish per pound all through 1600-1795
#   (1 pound Flemish = 6 guilders = 20 s), i.e. about 10.5 guilders per pound;
# livres tournois: 28 before the System; month by month through John Law's collapse (Yale 1720 database:
#   London on Paris and Amsterdam cross rates); 33 in 1721-25; 24.5 after the 1726 stabilisation; francs 25.2;
# silver: the pound sterling held 111.4 g of fine silver 1601-1816, so 1 kg = 8.98 pounds (used for the
#   Bazacle, whose prices are reconstructed in silver).
LIVRE_KNOTS = [(1600, 28), (1719.0, 28), (1719.88, 28), (1719.96, 28), (1720.04, 30), (1720.12, 34), (1720.21, 40),
               (1720.46, 48), (1720.54, 56), (1720.62, 76), (1720.71, 87), (1720.96, 60), (1721.5, 33), (1725.9, 33),
               (1726.4, 24.5), (1794.9, 24.5), (1795.5, 25.2), (1803, 25.2)]


def per_gbp_pre1800(cur, t):
    if cur == "NLG":
        return 10.5
    if cur == "FRF":
        k = np.array(LIVRE_KNOTS)
        return float(np.interp(t, k[:, 0], k[:, 1]))
    if cur == "XAG_KG":
        return 1 / 8.98
    if cur == "USD":
        return 4.44          # the Spanish/US dollar at 4s 6d sterling
    raise ValueError(f"no pre-1800 rate for {cur}")


# ---- from 1800 ----
PER_USD_FIXED = {
    # gold-standard parities before 1914 (approximate; the guilder was on silver until 1875)
    "NLG": [(1800, 1914, 12.1 / 4.866)],
    "FRF": [(1800, 1914, 25.22 / 4.866)],      # germinal franc, 25.22 francs per pound
    "JPY": [(1949, 1971.0, 360.0)],            # Bretton Woods peg
}
_fx = list(csv.DictReader(open(os.path.join(DATA, "fx_usd_per_unit_monthly_1971.csv"))))
FX_T = np.array([int(r["month"][:4]) + (int(r["month"][5:7]) - 0.5) / 12 for r in _fx])
FX = {c: np.array([float(r[c]) if r[c] else np.nan for r in _fx]) for c in _fx[0] if c != "month"}


def _fred(cur, t):
    v, T = FX[cur], FX_T
    ok = ~np.isnan(v)
    if not ok.any() or t < T[ok][0] - 0.05:
        return None
    return float(np.interp(t, T[ok], v[ok]))


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
    if t >= 1971.0:
        return _fred("GBP", t)
    return _interp(t, GBP_T, GBP_V, log=False)


def usd_per_unit(cur, t):
    """US dollars for one unit of currency `cur` at decimal year t (from 1800)."""
    if cur == "USD":
        return 1.0
    if cur == "GBP":
        return usd_per_gbp(t)
    if cur == "XAG_KG":
        return 8.98 * usd_per_gbp(t)
    if cur in FX and t >= 1971.0:
        v = _fred(cur, t)
        if v is not None:
            return v
    for a, b, per_usd in PER_USD_FIXED.get(cur, []):
        if a <= t < b:
            return 1.0 / per_usd
    raise ValueError(f"no exchange rate for {cur} at {t:.2f}")


def to_today(value, cur, t):
    """Value (any unit, e.g. millions) in currency `cur` at decimal year t, in today's US dollars (same unit)."""
    if t >= 1800:
        return value * usd_per_unit(cur, t) * US_NOW / us_cpi(t)
    gbp = value if cur == "GBP" else value / per_gbp_pre1800(cur, t)
    gbp_1800 = gbp * uk_cpi(1800.5) / uk_cpi(t)
    return gbp_1800 * usd_per_gbp(1800.5) * US_NOW / us_cpi(1800.5)


if __name__ == "__main__":
    print(f"today = {NOW:.2f}, US CPI {US_NOW:.1f}")
    for v, c, t, what in [(6.44, "NLG", 1602.5, "VOC paid-in capital at par, 1602"),
                          (78, "NLG", 1637.0, "the popular '78 million guilders' VOC, 1637"),
                          (77.7, "NLG", 1720.6, "the VOC's real peak, Aug 1720 (f77.7m)"),
                          (4960, "FRF", 1720.05, "Mississippi Co. 4.96bn livres, Jan 1720"),
                          (180, "GBP", 1720.5, "South Sea Co. c. GBP 180m, Jun-Jul 1720"),
                          (1, "GBP", 1720.5, "one pound, 1720"),
                          (1400, "USD", 1901.2, "US Steel at formation, 1901"),
                          (1, "USD", 1929.7, "one dollar, 1929")]:
        print(f"  {what:45s} -> ${to_today(v, c, t):,.3f} today ({'millions' if v != 1 else 'dollars'})")
