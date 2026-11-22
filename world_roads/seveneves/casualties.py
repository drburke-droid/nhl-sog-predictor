"""What the Moon's debris would do to people: blasts, casualties, and disruption, by impactor size.

Fragments reaching the Earth (from impact_ladder.py): the arriving mass spread evenly over log(size) from
10 microns up to the largest fragment (0.3%, 1% or 3% of the impactor; spall scaling, Melosh 1984).
Each fragment arrives at ~11.1 km/s (falling from the Moon's distance), so its blast energy is
E = m v^2 / 2. A 20 m piece of lunar rock carries ~0.17 Mt, a third of Chelyabinsk's 0.5 Mt (which hit at
19 km/s); so blasts are counted by energy, not size:
    "Chelyabinsk-strength"  >= 0.5 Mt     (a lunar fragment >= ~29 m)
    "Tunguska-strength"     >= 10 Mt      (>= ~79 m)
Damage areas scale as E^(2/3) (blast radius ~ E^(1/3)), calibrated on the two real events:
    severe (homes flattened):  Tunguska, ~12 Mt -> 2,150 km^2 of forest flattened (R = 26 km)
    light (windows, injuries): Chelyabinsk, 0.5 Mt -> windows broken out to ~50 km
Small rocks burst high: light damage fades in from 0.05 to 0.5 Mt, severe damage from 1 to 10 Mt.
Casualties: severe zone (Tunguska's flattened forest is roughly the >= 2.5 psi zone) 20% killed, 45% injured, from
the U.S. Office of Technology Assessment (1979, The Effects of Nuclear War: 2-5 psi ~5% killed / 45% injured,
5-12 psi ~50% / 40%, the inner third of the zone being above 5 psi); light zone 0.07% injured (Chelyabinsk:
~1,500 of ~2 million). Overlapping blasts can't kill the same people twice: totals saturate at the population.
Where they land is random, so the expected number of people in a zone is its area times the world
population over the whole surface: 8.1 billion / 510 million km^2 = 15.9 per km^2 (oceans included).

python3 casualties.py -> results/casualties.json and a table
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V = 11.1e3
RHO = 2800.0
MT = 4.184e15
POP = 8.1e9
AREA = 5.1e8                    # km^2
DENS = POP / AREA
K_SEVERE = 26.0 / 12.0 ** (1 / 3)     # km per Mt^(1/3)
K_LIGHT = 50.0 / 0.5 ** (1 / 3)
ARRIVE = {'10% in': 9, 'half in': 400, '90% in': 2286}     # days, from the simulated trajectories


def mass(d):
    return math.pi / 6 * RHO * d ** 3


def energy_mt(d):
    return 0.5 * mass(d) * V ** 2 / MT


def d_for_mt(e):
    return (e * MT / (0.5 * V ** 2) * 6 / (math.pi * RHO)) ** (1 / 3)


def ramp(e, lo, hi):
    return min(max(math.log10(e / lo) / math.log10(hi / lo), 0.0), 1.0)


def bins(M_earth, d_max, d_min=1e-5, n=400):
    """Fragment counts in log-size bins: equal mass per log(size)."""
    edges = np.geomspace(1.0, d_max, n + 1)
    lam = math.log(d_max / d_min)
    k = 6 / (math.pi * RHO)
    counts = M_earth / lam * k * (edges[:-1] ** -3 - edges[1:] ** -3) / 3
    mids = np.sqrt(edges[:-1] * edges[1:])
    return mids, counts


def assess(M_earth, D, frac):
    d_max = frac * D
    out = dict(largest_m=d_max, chely=0.0, tung=0.0, regional=0.0, deaths=0.0, injuries=0.0, biggest_mt=energy_mt(d_max))
    if d_max <= 1.0:
        return out
    mids, counts = bins(M_earth, d_max)
    for d, n in zip(mids, counts):
        e = energy_mt(d)
        if e >= 0.5:
            out['chely'] += n
        if e >= 10:
            out['tung'] += n
        if e >= 100:
            out['regional'] += n
        sev = math.pi * (K_SEVERE * e ** (1 / 3)) ** 2 * ramp(e, 1, 10)
        lig = math.pi * (K_LIGHT * e ** (1 / 3)) ** 2 * ramp(e, 0.05, 0.5)
        out['deaths'] += n * DENS * sev * 0.20
        out['injuries'] += n * DENS * (sev * 0.45 + lig * 0.0007)
    out['deaths'] = POP * (1 - math.exp(-out['deaths'] / POP))
    out['injuries'] = (POP - out['deaths']) * (1 - math.exp(-out['injuries'] / POP))
    return out


def main():
    lad = json.load(open(os.path.join(HERE, 'results', 'impact_ladder.json')))
    rows = []
    for r in lad['rows']:
        if r['material'] != 'rock' or r['D_m'] not in (60.0, 140.0, 1000.0, 1e4, 3e4, 1e5):
            continue
        res = {}
        for frac in (0.003, 0.01, 0.03):
            res[frac] = assess(r['to_earth_kg'], r['D_m'], frac)
        c = res[0.01]
        # first year: half of the arrivals; a given city: within the light-damage radius of a
        # Chelyabinsk-strength blast (50 km), within the flattened zone of a Tunguska-strength one (26 km)
        p_city_chely = 1 - math.exp(-0.5 * c['chely'] * math.pi * 50 ** 2 / AREA)
        p_city_tung = 1 - math.exp(-0.5 * c['tung'] * math.pi * 26 ** 2 / AREA)
        peak_day = 0.1 * c['chely'] / 9
        rows.append(dict(D_m=r['D_m'], to_earth_kg=r['to_earth_kg'], storm=r['storm_factor'], warming=r['warming_C'],
                         satellite_years_in_a_week=r['storm_factor'] * 7 / 365.25,
                         central=c, low=res[0.003], high=res[0.03], chely_per_day_first_week=peak_day,
                         city_chely_first_year=p_city_chely, city_tung_first_year=p_city_tung))
        lo, hi = res[0.003], res[0.03]
        print(f"D={r['D_m']:>7.0f} m: largest piece {c['largest_m']:.0f} m ({c['biggest_mt']:.2g} Mt); "
              f"Chelyabinsk-strength {c['chely']:.3g} [{lo['chely']:.2g}-{hi['chely']:.2g}], "
              f"Tunguska-strength {c['tung']:.3g} [{lo['tung']:.2g}-{hi['tung']:.2g}], >100 Mt {c['regional']:.3g} [..{hi['regional']:.2g}]; "
              f"deaths {c['deaths']:.3g} [{lo['deaths']:.2g}-{hi['deaths']:.2g}], injuries {c['injuries']:.3g} "
              f"[{lo['injuries']:.2g}-{hi['injuries']:.2g}]; first week {peak_day:.0f} Chelyabinsks/day; "
              f"a city: {p_city_chely:.0%} Chelyabinsk nearby / {p_city_tung:.1%} flattened in year 1")
    out = dict(model=dict(v=V, rho=RHO, dens_per_km2=DENS, k_severe=K_SEVERE, k_light=K_LIGHT,
                          d_chely=d_for_mt(0.5), d_tung=d_for_mt(10)), rows=rows)
    with open(os.path.join(HERE, 'results', 'casualties.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'Chelyabinsk-strength needs a lunar fragment >= {d_for_mt(0.5):.0f} m; Tunguska-strength >= {d_for_mt(10):.0f} m')


if __name__ == '__main__':
    main()
