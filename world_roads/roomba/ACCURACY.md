# Accuracy and sources: how long would a Roomba take to drive every road?

The question: a Roomba with no map (a first-generation one) drives a road network at the speed of
light, picking roads at random. How long until it has driven every road at least once? It is answered
for New York City and then for every road in the contiguous United States, plus Calgary in a Short. It
drives down the roads only; cleaning them is another matter (section 7). Every number in the videos
comes from `stats.py` (`data/stats.json`) or the scripts named below.

## 1. The Roomba, and the rule

**First-generation Roombas** (from 2002) had no map. They cleaned almost entirely at random: drive
until the bumper hits something, turn a random way, carry on (with some spiral and wall-following
moves). Mapping arrived in 2015 with the Roomba 980 and its camera-based vSLAM navigation.
Sources: [Wikipedia: Roomba](https://en.wikipedia.org/wiki/Roomba);
[Explain That Stuff](https://www.explainthatstuff.com/how-roomba-works.html);
[MIT Technology Review, 2015](https://www.technologyreview.com/2015/09/16/247936/the-roomba-now-sees-and-maps-a-home/).
The cold open (`room.py`) is a simulation of that bump-and-turn rule: a 34 cm robot at 0.3 m/s in a
6 × 4.2 m room. It covers 69% of the floor in 5 minutes, 89% in 15 and 94% in 30.

**On roads** the rule becomes: at every intersection, take one of the roads you didn't arrive on, each
with equal odds; turn back only at a dead end. This is a *non-backtracking random walk*. Bends are
followed; one-way rules are ignored, as in the earlier light-speed films.

**Time.** Distance driven ÷ the speed of light in vacuum (299,792.458 km/s), with no slowing at
junctions.

**When is it done?** When every piece of road has been driven at least once, in either direction (the
walk's *edge cover time*).

**Blocks, not segments.** The map data splits roads at every bend as well as at intersections. On
screen, roads are counted as **blocks**: a stretch of road from one intersection (or dead end) to the
next, with the bends merged in (`cover.block_ids`). The walk can't turn at a bend, so it always drives a
whole block at once. The counts:

| Network | Segments in the data | Blocks |
|---|---:|---:|
| New York City | 224,519 | 91,022 |
| Contiguous US | 44,581,882 | 22,995,011 |
| Calgary | 256,824 | 56,375 |

## 2. The networks

All three come from Overture Maps (release 2026-09-23.1): motorway down to residential and living
streets, with no service roads, tracks or paths. Each keeps its largest connected network.

**New York City** (`nyc.py`). The city's land from Overture divisions, plus New York State water within
about 2.5 km of it, so the bridges between boroughs stay in. Roads into New Jersey, Nassau and
Westchester stop at the city line, so Far Rockaway's roads through Nassau County don't count. The
result is **10,979 km** of street, 99.9% of the 10,988 km inside. The Roomba starts in Times Square.

**The contiguous United States** (`usa.py`). The North American network of the lost-photon film, cut
to the lower 48 with the Natural Earth 1:10m border. That gives **7,901,434 km** of road meeting at
38,602,911 points. The Roomba starts at the junction nearest the geographic centre of the lower 48
(39.8283° N, 98.5795° W, near Lebanon, Kansas). Driving every road exactly once would take light
26.4 s, the floor for any route.

The network is derived from OpenStreetMap, which draws each carriageway of a divided road as its own
line and includes ramps and private residential streets. That is why it is longer than the roughly
6.7 million km of public road in FHWA's *Highway Statistics*.

**Calgary** (`city.py`, for the Short). Streets inside city limits (the Calgary film's street set):
7,583 km in the main network.

## 3. The runs

Walks run in numba (`cover.py`). Each seed gives a different walk, and a walk replays exactly from its
seed (`walk_path`, `tail_walk` were checked against the recorded totals).

| Network | Runs | Median | 10th–90th percentile | Fastest – slowest |
|---|---:|---:|---:|---:|
| New York City | 200 | **1.88 s** | 1.30–3.02 s | 0.95–5.14 s |
| Contiguous US | 16 | **2 h 23 min** | | 1 h 33 min – 3 h 48 min |
| Calgary | 200 | **1.24 s** | 1.00–1.70 s | 0.80–2.67 s |

Each video shows a run near its median. NYC is run 30 (1.91 s). The US is run 0 (2 h 21 min 17 s).
Calgary is run 31 (1.24 s).

Run 30 was picked from the eight NYC runs nearest the median for its ending: a public street, reached
after a long final search. Of the four nearest, one ends inside the gated Kingsborough Community College
campus, two on Staten Island and one in Hamilton Beach.

## 4. New York's last street

Run 30 drives 571,802 km (52× the network). Two blocks are left after 407,846 km. Finding them takes the
remaining 163,956 km, 29% of the whole trip, and both fall in its last 2.5 km. The last block is
128 m of **Beach 6th Street, Far Rockaway, Queens**, at the far end of the Rockaway peninsula.

Across all 200 NYC runs, the last street was on the Rockaway peninsula 80 times, on Staten Island 60
times and in Hamilton Beach 10 times. The other 50 were mostly in other waterside pockets: Gerritsen
Beach, Throgs Neck, Manhattan Beach.

## 5. America's last road

Run 0 drives 2,541,485,088 km: **2 h 21 min 17.48 s** of light, 322× the network. Its last **105
blocks** (11.4 km of road) are all on the south end of **Manasota Key**, a barrier island at Englewood
Beach, Florida. The Roomba reaches them only in its final 125 km of driving (0.42 ms of light). The slow
motion plays the last 129 km in 4.8 s, about 11,000 times slower than light. The last block is 379 m of
**Beach Road**. Both ends of its final piece lie within 0.3 m of Overture's Beach Road (`where.py`).

**Float precision.** Per-road first-visit distances are stored as float32. At 2.5 billion km, float32
resolves only 256 km, too coarse to order the island's roads. So the walk's ending was replayed in
float64 (`usa.py tail 0 20000`), reproducing the recorded total to the metre, and the video uses those
exact values.

| Run | Light time | Distance | × network | The last road |
|---:|---:|---:|---:|---|
| 0 | 2 h 21 min | 2.54 billion km | 322× | Beach Road, Manasota Key (Englewood Beach), FL: barrier island |
| 1 | 2 h 13 min | 2.39 billion km | 302× | Sheldrake Court, near Beaufort, NC: coast |
| 2 | 1 h 45 min | 1.89 billion km | 239× | Cowshed Road, Queenstown, VA: Northern Neck peninsula |
| 3 | 3 h 00 min | 3.23 billion km | 409× | Nellie Lane, near Port Angeles, WA: Strait of Juan de Fuca coast |
| 4 | 2 h 34 min | 2.77 billion km | 351× | Caratoke Highway, Point Harbor, NC: tip of the Currituck peninsula |
| 5 | 2 h 06 min | 2.27 billion km | 287× | off Eider Lane, St. George, ME: peninsula |
| 6 | 3 h 15 min | 3.51 billion km | 444× | Foster Lane, Buras, LA: Mississippi River delta |
| 7 | 3 h 48 min | 4.10 billion km | 519× | Factory Street, Lafitte, LA: bayou |
| 8 | 2 h 40 min | 2.87 billion km | 363× | Tupelo Trail, Edisto Beach, SC: sea island |
| 9 | 1 h 33 min | 1.67 billion km | 212× | Lone Oak, near Big Creek, TX: inland |
| 10 | 2 h 25 min | 2.61 billion km | 331× | off Scenic View Drive, near Rockaway Beach, OR: coast |
| 11 | 2 h 38 min | 2.84 billion km | 360× | Tarpon Street, Anna Maria Island, FL: barrier island |
| 12 | 2 h 07 min | 2.28 billion km | 288× | unnamed road south of Hoquiam, WA: inland of Grays Harbor |
| 13 | 1 h 35 min | 1.71 billion km | 216× | Maison, Aliso Viejo, CA: inland suburb |
| 14 | 2 h 03 min | 2.22 billion km | 281× | Courtney Way, Scotland Beach, MD: Point Lookout peninsula |
| 15 | 2 h 27 min | 2.64 billion km | 334× | 9th Street, Ocean City, MD: barrier island |

13 of the 16 last roads are on a coast. Islands, peninsulas and bayous have few connections to the
rest of the network, so a walker that doesn't remember where it has been rarely finds its way in.

Place names come from `where.py` (data/where_*.json): the nearest named Overture segment to the last
piece's midpoint, and the nearest locality. The named segment is within 15 m except in runs 5 and 10
("off": 64 m and 29 m, probably an unnamed spur) and run 12 (no named road within 25 m).

## 6. A real Roomba's speed

The payoff divides each median distance by **0.5 m/s**, the top drive speed a Roomba accepts. That is
the iRobot Create 2's ±500 mm/s limit; the Create 2 is built on the Roomba 600 series. Sources: the
[create2](https://www.npmjs.com/package/create2) and [pycreate2](https://pypi.org/project/pycreate2/0.7.2/)
drive-command documentation. A cleaning run is usually slower, so these are best cases.

| | Median distance | At 0.5 m/s |
|---|---:|---:|
| New York City | 565,100 km | 36 years |
| Contiguous US | 2.58 billion km | 163,000 years |
| Calgary | 371,300 km | 24 years |

## 7. Driving isn't cleaning

A Roomba is about 34 cm across. A two-lane road (two 3.66 m lanes, 7.3 m) is about 21 Roombas wide,
hence "about 20" on screen. Cleaning it in neat strips would take about 20 passes; at random, far more.

## 8. Comparisons

* **Sun to Saturn:** 9.537 AU = 1.43 billion km. The US run's 2.54 billion km is longer. Semi-major axes
  from NASA's planetary fact sheets; 1 AU = 149,597,870.7 km.

## 9. Rendering

* **Maps.** Each road's length is binned into pixels: along the segment for the cities, and at the
  segment's midpoint for the US, where segments are far shorter than a pixel. A pixel's gold is the
  share of its road length driven so far, raised to the 4th power on the US map. A pixel therefore stays
  blue until nearly all its roads are done. Below 80,000 segments left, each remaining one is also drawn
  as a blue spark.
* **The Roomba** is drawn at its true position on the walk for the cities and the slow-motion finale.
  On the full US map it is far smaller than a pixel, so it isn't drawn there.
* **Time.** Runs speed up as they go (video time ∝ distance^(1/3.4) for cities, ^(1/3.3) for the US), so
  you see both the first moments and the long tail.
* **The zoom** draws every road within the view at each scale.
