# Accuracy and sources: the Roomba photon

The question: a photon drives the road network at random, at the speed of light. How long until it has
driven every road at least once? Answered for Calgary, then for every road in the contiguous United
States. Every number in the video comes from `stats.py` (written to `data/stats.json`).

## 1. The rule

At every junction the photon takes one of the roads it didn't arrive on, each with equal odds. It never
turns back, except at a dead end, where it must. This is a *non-backtracking random walk*: like a
Roomba, it has no map, no memory and no plan. Bends in a road are followed; one-way rules are ignored,
since light doesn't obey traffic law (the same convention as the earlier light-speed films).

For comparison, the video also runs the *plain random walk*, which may turn back at any junction
(`uturn=True` in `cover.py`).

**Time.** Distance driven ÷ the speed of light in vacuum (299,792.458 km/s). The photon doesn't slow
down at junctions.

**When is it done?** When every road segment (a piece of road between two junctions) has been driven at
least once, in either direction. That is the *edge cover time* of the walk. The walk stops on the last
new segment.

## 2. The networks

**Calgary.** Overture Maps streets inside city limits, motorway down to residential and living streets
(the street set of the Calgary film, `calgary/city.py`). The walk uses the largest connected network,
7,583 km of street in 256,824 segments (98.9% of the 7,664 km inside city limits; the rest is a handful
of pieces not joined to the main network). It starts at the junction nearest downtown.

**The contiguous United States.** The full North American network of the lost-photon film
(`turns/full_photon.py`; Overture release 2026-09-23.1, motorway through residential and living streets;
no service roads, tracks or paths). It is cut to the lower 48 with the Natural Earth 1:10m border
(rasterised at 0.005°), keeping the largest connected network:
* **7,901,434 km of road** in **44,581,882 segments**, meeting at 38,602,911 junctions.
* The photon starts at the junction nearest the geographic centre of the lower 48 (39.8283° N,
  98.5795° W, near Lebanon, Kansas).

Driving each segment exactly once would take light 26.4 s. No route can do better, so 26.4 s is the
floor for any photon, planned or not.

The network is derived from OpenStreetMap, which maps each carriageway of a divided road as its own
line and includes ramps and private residential streets. That is why its length (7.9 million km) is
more than the roughly 6.7 million km of public road in FHWA's *Highway Statistics*.

## 3. The walks

Walks are run in numba (`cover.py`); each seed gives a different walk. A walk is reproducible from its
seed, and the video's walks were replayed step by step to check this (`walk_path`, `tail_walk`).

**Calgary, 200 walks of each kind** (`city.py many 200`):

| Rule | Median | 10th–90th percentile | Fastest – slowest |
|---|---:|---:|---:|
| Roomba photon (no U-turns) | 1.24 s | 1.00–1.70 s | 0.80–2.67 s |
| Plain random walk | 19.2 s | 14.5–30.5 s | 10.6–82.7 s |
| Planned route (Calgary film) | 37.1 ms | | |

The video shows seed 31, the walk nearest the median: 371,417 km = 1.24 s of light, 49× the network.
Its last street is an 81 m segment on the north-east edge of the city. 68% of the trip went to finding
the last 1% of street length.

The planned route is the route of the Calgary film: a T-join plus Euler tour on all 7,664 km inside
city limits, so it is a slightly larger network than the walks'. Random is therefore at least 33×
slower than a plan.

**USA, 16 walks** (`usa.py walk 4`, `usa.py more 4 3`): see the table in section 5.

## 4. The last road

The walk shown in the video (seed 0) drives 2,541,485,088 km: **2 h 21 min 17.48 s** of light, 322× the
network. Its last 152 new segments (11.4 km of road) are all on the south end of **Manasota Key**, a
barrier island at Englewood Beach, Florida. The photon reaches them only in its final 125 km of driving
(0.42 ms of light). The slow-motion sequence plays the last 129 km (the island, plus a few hundred
metres of the way in) in 5.2 seconds, about 12,000 times slower than light.

The very last segment is 20 m of **Beach Road**. Both of its ends are within 0.3 m of Overture's Beach
Road geometry (`where.py`, data/where_seed0_island.json).

**Float precision.** Per-road first-visit distances are stored as float32 for memory. At 2.5 billion km
float32 resolves only 256 km, which can't order the island's 152 roads. The end of the walk was
therefore replayed in float64 (`usa.py tail 0 20000`; the replay reproduces the recorded total to the
metre). The video uses those exact values.

**Where the other runs ended.** Each run's last segment was looked up the same way; see section 5.
Islands, peninsulas and bayous have few connections to the rest of the network, so a walk that
doesn't remember where it has been rarely finds its way in.

## 5. All 16 USA runs

STATS_TABLE

## 6. Comparisons

* **Sun to Saturn:** 9.537 AU = 1.43 billion km (79 min of light). **Sun to Uranus:** 19.19 AU = 2.87
  billion km (2 h 40 min). The 2.54-billion-km drive is longer than the first and shorter than the
  second. Semi-major axes from NASA's planetary fact sheets; 1 AU = 149,597,870.7 km.

## 7. Rendering

* **Maps.** Each road's length is binned into pixels: along the segment for Calgary, and at the
  segment's midpoint for the USA, where segments are much shorter than a pixel. A pixel's gold is the
  share of its road length driven so far, raised to the 4th power on the US map. A pixel stays blue until nearly all
  of its roads are done, so unvisited pockets stay visible. Below 80,000 roads left, each remaining
  road is also drawn as a blue spark.
* **Time.** The runs speed up as they go (video time ∝ distance^(1/3.3) for the USA, ^(1/3.4) for
  Calgary), so you see both the first second and the long tail.
* **The zoom** draws every road within the view at each scale, nearest-first from the final view.
