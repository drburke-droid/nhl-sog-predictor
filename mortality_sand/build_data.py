"""Build the film's data: every vessel's population, year by year from 1 CE to 2025, and every source of
excess deaths (the grains lifted off the table), with its years, place, size and cause.

    python build_data.py            # -> data/series.npz, data/lifts.json, data/clock.json, out/data_check.png

Two kinds of source, joined at 1950:

1. 1950-2023: UN World Population Prospects 2024 (annual deaths and population of every country). A
   country's normal death rate for a year is the smooth trend of its crude death rate, drawn through its
   ordinary years: an asymmetric least-squares smoother (Eilers & Boelens 2005) on the log death rate,
   which follows the slow fall (and, as populations age, slow rise) of the death rate but is not pulled
   up by crisis years. Excess deaths are deaths above that trend, year by year; years at or below it lift
   nothing. Each country-year is labelled with the cause of the named crisis it falls in
   (inputs/events_post1950.csv), or "other" (bad flu seasons, heat waves, the post-Soviet mortality
   crisis and other years above normal).

2. 1 CE-1949: no annual death counts exist for most of the world, so the film shows the catastrophes
   big enough to have published death-toll estimates (inputs/crises_pre1950.csv: central estimates,
   with ranges and sources). Their deaths are spread evenly over the years given.

Population: Our World in Data's long-run series (HYDE 3.3 before 1800, Gapminder to 1950) at fixed anchor
years, UN WPP every year from 1950. Between anchors, the catalogued losses are taken out as they happen and
grow back at a natural pace (an e-folding time of RECOVERY years), so a catastrophe shows as a dip that
refills, while the population still passes exactly through every anchor.
"""
import csv, json, math, os
import numpy as np
import pandas as pd

import story as S
import places as PL

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
INP = os.path.join(HERE, "inputs")
OUT = os.path.join(HERE, "out")
YEARS = np.arange(S.YEAR0, S.YEAR1 + 1)
VESSELS = list(S.VESSELS.keys())
VI = {v: i for i, v in enumerate(VESSELS)}

WPP_URL = ("https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"
           "WPP2024_Demographic_Indicators_Medium.csv.gz")
WPP = os.path.join(DATA, "WPP2024_Demographic_Indicators_Medium.csv.gz")
OWID_URL = "https://ourworldindata.org/grapher/population.csv?v=1&csvType=full&useColumnShortNames=true"
OWID = os.path.join(DATA, "owid_population.csv")

ALS_LAMBDA = 300      # stiffness of the normal-death-rate trend (annual data: bends over about a decade)
ALS_P = 0.3           # asymmetry: points above the trend count 0.3, below 0.7 (crises do not pull it up)
PANDEMIC = 2020       # from here the normal death rate is the 2012-19 trend, projected
LAST_DATA_YEAR = 2023  # WPP 2024 estimates end here; 2024-25 are projections (population only)
SLOW = 0.25           # video seconds added per million excess deaths in a year
RECOVERY = 30.0       # years: e-folding time for a population to regrow after a catastrophe
ANCHORS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
           1750, 1800, 1850, 1900]
# The collapse of the Americas after 1492 is driven by the catalogue, not by HYDE's 1600 values (which
# disagree with the death-toll literature); these vessels skip the 1600 anchor and re-sync in 1700.
SKIP_ANCHOR = {1600: {"MES", "CAR", "AMZ", "ADS", "NAM", "SCN"}}


def fetch(url, path):
    if not os.path.exists(path):
        import subprocess
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", path, url], check=True)


# ------------------------------------------------------------------ 1950-2023: UN WPP
def wpp_countries():
    fetch(WPP_URL, WPP)
    d = pd.read_csv(WPP, low_memory=False, usecols=["LocTypeName", "ISO3_code", "Location", "Time",
                                                     "TPopulation1July", "Deaths", "CDR"])
    d = d[(d.LocTypeName == "Country/Area") & (d.Time <= S.YEAR1)]
    return d.drop(columns="LocTypeName")


def als(y, lam=ALS_LAMBDA, p=ALS_P, niter=30):
    """Asymmetric least squares smoothing (Eilers & Boelens 2005): a smooth curve that points above it
    pull up only weakly (weight p) and points below pull down strongly (1 - p)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    DD = lam * D.dot(D.T)
    w = np.ones(L)
    z = y
    for _ in range(niter):
        z = spsolve((sparse.spdiags(w, 0, L, L) + DD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def wpp_excess(d):
    """Excess deaths per country and year, 1950-2023 (people), and each country's normal death rate."""
    rows = []
    for iso, g in d[d.Time <= LAST_DATA_YEAR].groupby("ISO3_code"):
        g = g.sort_values("Time")
        cdr = g.CDR.values.astype(float)
        t = g.Time.values.astype(float)
        y = np.log(np.maximum(cdr, 0.5))
        pre = t < PANDEMIC
        z = np.empty_like(y)
        z[pre] = als(y[pre])
        # the pandemic years against the pre-pandemic trend, projected (as in WHO's and The Economist's
        # excess-death models): a flexible trend fitted through 2020-23 would bend up into the pandemic
        k = pre & (t >= PANDEMIC - 8)
        a1, b1 = np.polyfit(t[k], z[k], 1)
        z[~pre] = a1 * t[~pre] + b1
        base = np.exp(z)
        ex = np.maximum(g.Deaths.values * 1000.0 - base / 1000.0 * g.TPopulation1July.values * 1000.0, 0.0)
        for t, e, c, b, deaths in zip(g.Time.values, ex, cdr, base, g.Deaths.values * 1000.0):
            rows.append(dict(iso=iso, year=int(t), excess=float(e), cdr=float(c), base=float(b), deaths=float(deaths)))
    return pd.DataFrame(rows)


def wpp_vessel_pop(d):
    """Population per vessel, every year 1950-2025 (people)."""
    out = np.zeros((len(VESSELS), S.YEAR1 - 1950 + 1))
    for iso, g in d.groupby("ISO3_code"):
        v = S.ISO_VESSEL.get(iso)
        if v is None:
            continue
        g = g.sort_values("Time")
        t = g.Time.values.astype(int) - 1950
        p = g.TPopulation1July.values * 1000.0
        if iso == "RUS":
            sh = np.array([S.siberia_share(y) for y in g.Time.values])
            out[VI["SIB"], t] += p * sh
            p = p * (1 - sh)
        out[VI[v], t] += p
    return out


# ------------------------------------------------------------------ before 1950: OWID long-run series
def owid_anchor_pop():
    """Population per vessel at the anchor years (people), from OWID (HYDE 3.3 / Gapminder)."""
    fetch(OWID_URL, OWID)
    d = pd.read_csv(OWID)
    d = d[d.code.notna() & ~d.code.astype(str).str.startswith("OWID")]
    out = np.zeros((len(VESSELS), len(ANCHORS)))
    for iso, g in d.groupby("code"):
        v = S.ISO_VESSEL.get(iso)
        if v is None:
            continue
        s = g.set_index("year").population_historical.sort_index()
        vals = np.exp(np.interp(ANCHORS, s.index.values, np.log(np.maximum(s.values, 1.0))))
        if iso == "RUS":
            sh = np.array([S.siberia_share(max(y, 1)) for y in ANCHORS])
            out[VI["SIB"]] += vals * sh
            vals = vals * (1 - sh)
        out[VI[v]] += vals
    return out


# ------------------------------------------------------------------ the crisis catalogue (before 1950)
def label_of(name):
    """The on-screen name of an event: its catalogue name without the region after a colon or the notes
    in brackets ("Second World War: Japan" -> "Second World War")."""
    import re
    n = name.split(":")[0]
    n = re.sub(r"\s*\([^)]*\)", "", n).strip()
    return n or name


def read_crises():
    path = os.path.join(INP, "crises_pre1950.csv")
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if not r.get("vessel") or r["vessel"] not in VI:
            print("  skipped (vessel):", r.get("event_id"), r.get("vessel"))
            continue
        y0, y1 = int(float(r["start_year"])), int(float(r["end_year"]))
        y1 = max(y1, y0)
        if y0 >= 1950:
            continue
        keep = (min(y1, 1949) - y0 + 1) / (y1 - y0 + 1)      # from 1950 the UN series takes over
        rows.append(dict(id=r["event_id"], name=r["event_name"], cause=r["cause"].strip().lower(),
                         vessel=r["vessel"], y0=max(y0, S.YEAR0), y1=min(y1, 1949),
                         lon=float(r["lon"]), lat=float(r["lat"]), sigma=float(r["sigma_deg"] or 4),
                         deaths=float(r["deaths_central"]) * keep, low=float(r["deaths_low"] or 0) * keep,
                         high=float(r["deaths_high"] or 0) * keep))
    return rows


def read_events_post1950():
    path = os.path.join(INP, "events_post1950.csv")
    ev = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            y0, y1 = int(float(r["start_year"])), int(float(r["end_year"] or r["start_year"]))
        except ValueError:
            continue
        ev.append(dict(id=r["event_id"], name=r["event_name"], cause=r["cause"].strip().lower(),
                       iso=r["iso3"].strip(), y0=y0, y1=max(y1, y0),
                       deaths=float(r["deaths_central"] or 0)))
    return ev


def deaths_by_vessel_year(lifts):
    """Deaths per vessel per calendar year, from the lifts before 1950 (places re-time and re-place them)."""
    D = np.zeros((len(VESSELS), len(YEARS)))
    for l in lifts:
        if l["y0"] < 1950:
            D[VI[l["vessel"]]] += PL.per_year(l, S.YEAR0, len(YEARS))
    return D


def population_series(anchor, wpp_pop, D):
    """Population per vessel per year: log-interpolated between anchors, minus the catalogued losses
    (regrowing with RECOVERY), corrected so the path passes through every anchor; WPP from 1950."""
    P = np.zeros((len(VESSELS), len(YEARS)))
    pre = YEARS < 1950
    for i, v in enumerate(VESSELS):
        ay = [a for a in ANCHORS if v not in SKIP_ANCHOR.get(a, ())] + [1950]
        av = [anchor[i, ANCHORS.index(a)] for a in ay[:-1]] + [wpp_pop[i, 0]]
        ay = np.array([max(a, 0) for a in ay], float)
        av = np.maximum(np.array(av), 1e3)
        I = np.exp(np.interp(YEARS, ay, np.log(av)))
        d = np.zeros(len(YEARS))
        k = math.exp(-1.0 / RECOVERY)
        for j in range(1, len(YEARS)):
            d[j] = d[j - 1] * k + D[i, j - 1]
        Ld = np.interp(YEARS, ay, np.interp(ay, YEARS, d))
        P[i] = np.maximum(I - (d - Ld), 0.25 * I)
        P[i, ~pre] = wpp_pop[i, YEARS[~pre] - 1950]
    return P


# ------------------------------------------------------------------ post-1950 lifts with causes
def post1950_lifts(ex, events):
    """One lift per country-year with excess deaths, split between the named crisis it falls in and "other".

    A country's own events come first (the shortest, i.e. the most specific in time: COVID-19 over HIV/AIDS
    in 2020-23). A crisis takes at most 1.5x its own published yearly toll (deaths_central spread over its
    years) out of the country-year's excess, so a small war in a big country (Chechnya, in Russia's post-
    Soviet mortality crisis) does not claim the whole excess; the rest is "other". Worldwide rows ("WLD":
    the 1957, 1968 and 2009 flu pandemics, COVID-19 in countries without a row of their own) then take a
    share of the excess still unexplained in their years: their published total over that excess, or all
    of it when no total is given."""
    by_iso = {}
    for ev in events:
        by_iso.setdefault(ev["iso"], []).append(ev)
    rows = []                         # [iso, year, vessel, [(event or None, deaths)]]
    for r in ex[ex.excess > 0].itertuples():
        v = S.ISO_VESSEL.get(r.iso)
        if v is None:
            continue
        parts, left = [], float(r.excess)
        cands = [ev for ev in by_iso.get(r.iso, []) if ev["y0"] <= r.year <= ev["y1"]]
        if cands:
            ev = min(cands, key=lambda ev: (ev["y1"] - ev["y0"], -ev["deaths"]))
            take = left if ev["deaths"] <= 0 else min(left, 1.5 * ev["deaths"] / (ev["y1"] - ev["y0"] + 1))
            parts.append((ev, take))
            left -= take
        rows.append([r.iso, int(r.year), v, parts, left])
    for ev in by_iso.get("WLD", []):
        pool = [row for row in rows if ev["y0"] <= row[1] <= ev["y1"] and row[4] > 0]
        tot = sum(row[4] for row in pool)
        share = 1.0 if ev["deaths"] <= 0 or tot <= 0 else min(1.0, ev["deaths"] / tot)
        for row in pool:
            take = row[4] * share
            row[3].append((ev, take))
            row[4] -= take
    lifts = []
    for iso, year, v, parts, left in rows:
        parts = parts + ([(None, left)] if left > 0 else [])
        split = [(v, 1.0)]
        if iso == "RUS":
            sh = S.siberia_share(year)
            split = [("STP", 1 - sh), ("SIB", sh)]
        for ev, d in parts:
            for vv, f in split:
                lifts.append(dict(kind="country", iso=iso, vessel=vv, y0=year, y1=year, deaths=float(d * f),
                                  cause=ev["cause"] if ev else "other", name=ev["name"] if ev else "",
                                  label=label_of(ev["name"]) if ev else "", id=ev["id"] if ev else ""))
    return lifts


# ------------------------------------------------------------------ the clock
def build_clock(total):
    """Video seconds per year: a steady base pace (faster in the long quiet first millennium) plus SLOW seconds
    for every million excess deaths in the world that year, so a catastrophe's time on screen grows with its
    toll and, while it lasts, sand lifts at a steady ~4 million deaths a second."""
    E = total / 1e6                                              # world excess deaths per year, millions
    Es = np.convolve(E, np.ones(3) / 3, mode="same")             # a little smoothing (no jerky jumps)
    base = np.interp(YEARS, [1, 1000, 1500, 1800, 1900, 2025], [0.004, 0.006, 0.02, 0.04, 0.10, 0.30])
    slow = SLOW * Es                     # in proportion: during catastrophes ~1/SLOW million deaths per second
    rate = base + slow
    rate = dwell(rate)
    s = np.concatenate([[0.0], np.cumsum(rate)])[:-1]
    return rate, s


def dwell(rate):
    """Make room for the voiceover (voice.py): where a line would not finish before the next one starts,
    the years between them run slower (in proportion to how much of each year lies between the two)."""
    import voice
    lines = sorted((L["at"], L["dur"]) for L in voice.load_lines() if not isinstance(L["at"], str))
    if not lines:
        return rate
    rate = rate.copy()
    ends = lines[1:] + [(float(YEARS[-1] + 1), 0.0)]
    for _ in range(12):
        sk = np.concatenate([[0.0], np.cumsum(rate)])           # s at the start of each year
        ys = np.concatenate([YEARS, [YEARS[-1] + 1]]).astype(float)
        short = 0.0
        for (a, d), (b, _) in zip(lines, ends):
            need = d + voice.GAP + voice.LEAD
            have = np.interp(b, ys, sk) - np.interp(a, ys, sk)
            if have >= need - 1e-3:
                continue
            w = np.clip(np.minimum(YEARS + 1, b) - np.maximum(YEARS, a), 0, None)
            c = (need - have) / (w * w).sum()
            rate += c * w
            short += need - have
        if short < 1e-3:
            break
    return rate


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    d = wpp_countries()
    print("WPP countries:", d.ISO3_code.nunique())
    ex = wpp_excess(d)
    ex.to_csv(os.path.join(DATA, "wpp_excess.csv"), index=False)
    wpop = wpp_vessel_pop(d)
    anchor = owid_anchor_pop()
    crises = read_crises()
    events = read_events_post1950()
    places = PL.read_places()
    spots = [dict(kind="spot", vessel=c["vessel"], y0=c["y0"], y1=c["y1"], deaths=c["deaths"], cause=c["cause"],
                  name=c["name"], label=label_of(c["name"]), id=c["id"], lon=c["lon"], lat=c["lat"],
                  sigma=c["sigma"]) for c in crises]
    pre, rep1 = PL.apply_pre1950(spots, places)
    post, rep2 = PL.apply_post1950(post1950_lifts(ex, events), places)
    lifts = pre + post
    D = deaths_by_vessel_year(lifts)
    P = population_series(anchor, wpop, D)
    total = np.zeros(len(YEARS))
    for L in lifts:
        total += PL.per_year(L, S.YEAR0, len(YEARS))
    places_report(places, lifts, rep1, rep2)
    rate, s = build_clock(total)
    intro, outro = 7.0, 14.0
    clock = dict(s=[float(intro + v) for v in s] + [float(intro + s[-1] + rate[-1])],
                 year=[int(y) for y in YEARS] + [int(YEARS[-1]) + 1], intro=intro, outro=outro)
    json.dump(clock, open(os.path.join(DATA, "clock.json"), "w"))
    json.dump(lifts, open(os.path.join(DATA, "lifts.json"), "w"))
    np.savez_compressed(os.path.join(DATA, "series.npz"), years=YEARS, pop=P, deaths=total,
                        vessels=np.array(VESSELS))
    # ---- report
    print(f"film: {clock['s'][-1]:.1f} s of history after a {intro:.0f} s title, {outro:.0f} s ending")
    print(f"world population: 1 CE {P[:, 0].sum() / 1e6:.0f} M, 1500 {P[:, 1499].sum() / 1e6:.0f} M, "
          f"1900 {P[:, 1899].sum() / 1e6:.0f} M, 2025 {P[:, -1].sum() / 1e9:.2f} B "
          f"({P[:, -1].sum() / S.GRAIN:,.0f} grains)")
    pre = sum(L["deaths"] for L in lifts if L["y0"] < 1950)
    post = sum(L["deaths"] for L in lifts if L["y0"] >= 1950)
    print(f"lifted: before 1950 {pre / 1e6:.0f} M ({pre / S.GRAIN:,.0f} grains), 1950-2023 {post / 1e6:.0f} M "
          f"({post / S.GRAIN:,.0f} grains)")
    for k in S.CAUSE_KEYS:
        print(f"  {k:9s} {sum(L['deaths'] for L in lifts if L['cause'] == k) / 1e6:7.1f} M")
    for y0, y1 in [(1, 1000), (1000, 1500), (1500, 1800), (1800, 1900), (1900, 1950), (1950, 2026)]:
        m = (YEARS >= y0) & (YEARS < y1)
        print(f"  {y0}-{y1 - 1}: {rate[m].sum():6.1f} s on screen")
    check_plot(P, total, rate)


def places_report(places, lifts, rep1, rep2):
    n_pl = sum(1 for l in lifts if l.get("place"))
    d_pl = sum(l["deaths"] for l in lifts if l.get("place"))
    d_cv = sum(l["deaths"] for l in lifts if l.get("kind") == "converge")
    print(f"places: {len(places)} rows -> {n_pl} place lifts, {d_pl / 1e6:.1f} M deaths placed "
          f"({d_cv / 1e6:.1f} M travel from home to where they died)")
    for what, rep in (("before 1950", rep1), ("after 1950", rep2)):
        if rep["scaled"]:
            print(f"  {what}: places scaled down to fit their rows: {len(rep['scaled'])}")
            for r in rep["scaled"][:12]:
                print("    ", r)
        seen = set()
        for r in rep["skipped"]:
            if (r[0], r[2]) in seen and len(seen) > 30:
                continue
            seen.add((r[0], r[2]))
            print(f"  {what}: skipped {r}")


def check_plot(P, total, rate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 11))
    for v in ["EUR", "EAS", "SAS", "MES", "STP", "LEV", "WAF", "NAM"]:
        ax[0].semilogy(YEARS, P[VI[v]] / 1e6, label=S.VESSELS[v]["label"])
    ax[0].legend(ncol=4, fontsize=8); ax[0].set_ylabel("population (M)"); ax[0].grid(alpha=0.3)
    ax[1].plot(YEARS, total / 1e6, lw=0.8); ax[1].set_ylabel("world excess deaths / yr (M)"); ax[1].grid(alpha=0.3)
    ax[2].plot(YEARS, rate, lw=0.8); ax[2].set_ylabel("video s per year"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "data_check.png"), dpi=90)


if __name__ == "__main__":
    main()
