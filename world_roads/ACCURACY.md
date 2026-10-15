# Accuracy disclosure — "All world roads at light speed"

**This is a visualization of a thought experiment.** Light cannot carry an observer or a vehicle, no road
trip can be made at *c*, and the route shown is a heuristic, not a proven optimum. Every figure on
screen and in this file is an estimate.

## 1. What is real

| Item | Value | Source |
|---|---:|---|
| Road data | Overture Maps `transportation/segment`, release **2026-09-23.1** (derived from OpenStreetMap) | streamed from the public S3 bucket, ~35 GB of geometry |
| All mapped `road` segments, every class, incl. footways/paths | **93.4 M km** | exact haversine sum over full-resolution vertices |
| Vehicular roads (motorway … service + unknown), **excluding** tracks | **64.2 M km** | same |
| … plus unpaved `track` | 84.6 M km | same |
| Classes actually *routed* by the dot: motorway, trunk, primary, secondary, tertiary | 17.7 M km (19% of all mapped roads, 28% of vehicular) | same |

The 64.2 M km vehicular total from OpenStreetMap is a close match to the ~64.3 M km working figure in
the brief. That is partly a coincidence of definitions. OSM counts each mapped way once, so a
divided highway mapped as two carriageways counts twice, and a two-way street counts once. It also
ignores one-way rules, and OSM completeness varies by country.

Per-class lengths (M km): motorway 1.31 · trunk 2.05 · primary 2.53 · secondary 3.80 · tertiary 8.01 ·
unclassified 17.23 · residential 18.40 · living_street 0.34 · service 8.63 · track 20.41 · unknown 1.88 ·
footway 2.72 · path 5.46 · cycleway 0.46 · other non-motorized 0.15.

## 2. What the dot actually traverses

* **Routed network.** The routed network covers every motorway–tertiary segment worldwide except
  Antarctica, which has about 6,400 km of roads (0.007%) and is left off the map. Vertices are
  snapped to a grid of about 5 km at the equator so the route can be drawn at 1080p. This
  generalization turns the 17.7 M km of real roads into a graph of **2.56 M edges, 12.0 M km**:
  curves are straightened, and parallel carriageways merge onto the same grid cells. Distance is
  measured on the real geometry, and the generalized graph is used only for routing and drawing.
* **Gap bridges.** Filtering to major classes leaves short gaps where, for example, a tertiary road
  ends at a residential street. We bridged 2,596 such gaps of at most 2.5 grid cells (≲13 km), only
  where every sample point along the bridge lies on land, totalling 18,400 km (0.15%). No bridge
  crosses water, and no international road link was invented.
* **Coverage.** The dot drives **100% of the routed graph's edges**, which the router checks with an
  assertion.
* **Minor roads.** Residential, unclassified, service and track roads are **not individually routed**.
  They are drawn as a faint, length-weighted density texture: real OSM-derived data, aggregated per
  pixel. Each minor-road pixel turns gold when the dot reaches the nearest routed road. So "the dot
  visited every road" is true only for the major-road graph; the minor roads light up by proximity.

## 3. How the route was built (`route.py`)

1. **Macro order.** A hand-authored waypoint path gives every edge a scalar *sweep key*. The path runs
   Alaska, North America west to east, the Caribbean, Mexico and Central America, South America,
   Europe, Turkey and the Levant, Africa, Arabia, Asia, Indonesia, Australia and the Pacific, ending
   in New Zealand. This is art direction, and it is the only scripted part.
2. **Bands and pieces.** Edges are grouped into 1,500 km key bands, and each band is split into its
   connected pieces (2,134 pieces in total).
3. **Inside each piece:** an **exact Euler path**, built by Hierholzer's algorithm, runs over the
   piece's edges plus a **greedy T-join**. Odd-degree junctions are paired with their nearest odd
   partner by shortest road path (≤300 km), and those paths are driven twice. This is the standard
   Chinese-postman construction, but the pairing is greedy, so it is **not** the minimum-weight
   matching an exact solution would need. Odd nodes left unpaired get a zero-length hypothetical jump.
4. **Between pieces:** the dot takes the shortest road path (Dijkstra) to the nearest unfinished
   piece in its own connected network, and this repeated travel is counted. If its network has
   nothing left in the band, it makes a **hypothetical instantaneous jump** to the nearest unfinished
   piece elsewhere: an island, another continent, or a fragment the class filter disconnected.
5. **Result.** The walk covers 16.65 M km on a 12.0 M km network, so **38.7% is repeated travel**.
   That overhead covers parity fixes, links between pieces and bands, and the greedy pairing; an
   exact rural-postman solution would need less. There are 1,415 jumps; 431 of them move more than 20
   px on screen and are drawn as rings. Jumps cost zero distance and zero time.

The 30% backtracking allowance in the brief is plausible, but it is an assumption. Our 38.7% applies
only to this generalized major-road graph under these heuristics; the overhead for the full road
network (with its many dead ends) is unknown.

## 4. How time maps to the clock

* **Timing.** The video lasts exactly 280 s: 8,400 frames at 30 fps. Frame *f* shows time
  280·f/8399, so the clock starts at 0:00 and the last frame reads 4:40.
* **Pacing.** The dot moves at a constant rate *along the walk*: elapsed video time is proportional to
  cumulative walk distance, repeats included. One second of video is 1/280 of the walk, about
  59,500 km of sampled walk. Jumps are instantaneous. The regional timetable is therefore
  **distance-proportional**: Asia takes the most time because it has the most routed road length.
* **Distance readout.** The on-screen "≈ X M km hypothetical distance at c" is simply
  *t* × 299,792.458 km/s, reaching 83.94 M km at 4:40. That is the distance the brief's estimate
  implies: 64.3 M km × 1.3055. It is **not** a measured quantity.
* **Using our own figures instead.** 64.2 M km of vehicular road × (1 + 0.30 … 0.387) ≈ 83 … 89 M km,
  or about **4m 38s … 4m 57s** at *c*. The brief's 4m 40s sits inside that range, but both
  assumptions behind it (what counts as a road, and how much backtracking) shift the answer by
  tens of seconds.

## 5. Where the dot really is at the art-direction checkpoints

| Time | Brief's target | Actual (distance-proportional) |
|---|---|---|
| 0:00 | western North America | northern Alaska (Utqiaġvik area) |
| 0:40 | eastern North America | southern Ontario |
| 1:20 | south-eastern South America | Tierra del Fuego |
| 2:00 | Europe | European Russia (Volga region) |
| 2:40 | NE Africa / Middle East | Mozambique (south-eastern Africa) |
| 3:20 | near East Asia | Korea / NE China border |
| 4:00 | Australia | eastern Himalaya / Yunnan |
| 4:40 | New Zealand finale | New Zealand (North Island) |

Asia's large share of road length pushes the Australia leg into the last ~25 s.

## 6. Disconnected networks

Islands and continents connected only by sea or ferry are separate graph components. The dot never
draws a line across water. It blinks to the new network, where an expanding ring marks the arrival
(with a fainter ring at the departure point). Long jumps also show a "hypothetical jump · no road"
label. Ferries, bridges and tunnels that exist in OSM *as roads* (for example, bridges that are road
segments) are routed like any other road. Ferry routes themselves are not roads and are not used.

## 7. Map

The map uses the Natural Earth projection (Šavrič et al.) with central meridian 10°E, so the cut at
170°W falls in the Bering Strait and mid-Pacific. Land polygons come from Natural Earth 1:10m. There
are no country borders, and the camera is static.

## 8. "The lost photon" (`turns/lost_photon.py`, `turns/lost_photon_render.py`)

**Network.** The walk uses the world film's major-road graph: motorway to tertiary, generalized to a
~4 km vertex grid. It takes only the connected piece that contains Los Angeles, which runs from
Alaska and Canada to Panama; the Darién Gap ends it. That piece is 2.18 M km of road, with 349,197
nodes and 467,769 road pieces. "Intersection" means a node of this graph where 3+ pieces meet.
Residential streets are not included.

**Rule.** At each intersection the photon picks uniformly among the exits other than the one it
arrived on. It never U-turns, except at a dead end. Bends are followed, and one-way rules are ignored.

**Arrival.** The walk ends once the photon is within 15 km of New York City Hall.

**Simulation.** 2,000 walks were run with seeds 0–1999, and all 2,000 arrived.

| | Light time |
|---|---:|
| Median | 14.68 s |
| Mean | 19.8 s |
| 10th–90th percentile | 4.2–42.4 s |
| Fastest | 0.88 s |
| Slowest | 117.4 s |

**The walk shown.** The video shows seed 974, the walk closest to the median. It covers 4,401,992 km
over 940,972 road pieces, which takes light 14.683 s.

| Comparison | Distance | Light time |
|---|---:|---:|
| Shortest road route on the same graph (Dijkstra) | 4,929 km | 16.4 ms |
| Straight line | 3,936 km | 13.1 ms |

**Derived numbers.**

* 893× longer = 14.683 s ÷ 16.4 ms.
* Around the Earth uses the equator, 40,075 km.
* At 100 km/h nonstop, a car would take 5.0 years.

**"Drove through X".** This counts each entry into a 25 km circle around the city centre.

**Time on screen.**

* The walk plays in real time: 1 s of video is 1 s at c.
* The opening animation of the shortest route is *not* real time; it is drawn over 0.9 s.

**Caveat.** This is a simulation on a simplified network of highways and main roads. Every extra
side street gives a random walker more places to wander, so on the full street network the trip
would very likely take longer. That case was not computed.

## 9. "Every street in 48 cities" (`cities/`)

**Area.** Each city is every public street within 15 km of a conventional downtown landmark. The
coordinates are in `cities/cities.py`. A fixed circle makes cities comparable at the same scale,
where city limits vary enormously (Paris is 105 km², Calgary 825 km²). Water inside the circle
simply has no streets.

**Which streets.** The same as the Calgary film: motorway through living street, plus `unknown`.
Service roads, alleys, parking aisles, driveways, tracks and paths are excluded.

**The lap.** A lap is a coverage walk, built with the same method as the Calgary film: a greedy
T-join plus an Euler path in rings spreading out from downtown. It drives every street at least
once, and repeats are counted; backtracking adds 22–82%, with a median around 45%. Jumps between
disconnected pieces are instant and add no distance. These are streets cut by the circle, islands,
and mapping gaps.

**Lap times.** They run from 11.5 ms (Hong Kong) to 71.7 ms (Delhi), and total 1.692 s for all 48.
`cities/data/race_stats.json` has every city, and `media/cities/race_stats.json` is a copy.

**On screen.** Every lap is slowed by the same factor, 30×, so on-screen durations compare
honestly. The 0.1 s pause between cities is not counted in any clock.

**Data completeness.** OpenStreetMap coverage varies by city. A few cities have unusually low
residential shares: Hong Kong 14%, Moscow 18%, Shanghai 25%, Beijing 26%, against 50–70% in most
cities. There, local streets are often tagged as service roads or not mapped at all. Their lap times
are likely **underestimates**.

**Calgary here vs. the Calgary film.** Here Calgary takes 30.8 ms; in the Calgary film it takes 37.1
ms. The area differs: a 15 km circle here, the city limits there.

**Bug fixed during this work.** The vertex key used for snapping did not handle southern latitudes.
It was fixed before rendering. It never affected the Calgary or Los Angeles work, which are both in
the northern hemisphere.

## 10. "How many signal fires to circle the Earth?" (`signal/`)

**Elevation data.** The terrain comes from Terrarium tiles on AWS, the same source as the VSL Sightline
tool. North of 56°N the search uses zoom 10, because at lower zooms Greenland, arctic Canada and
Siberia come from a coarse source that flattens whole ranges; south of 56°N it uses zoom 9.

**Despiking.** Faults are cleaned before use, extending the Sightline tool's rule:
* spikes, plus row and column streaks;
* spike-and-pit pairs (pits are repaired too, because these faults come in pairs);
* anything more than 1,500 m off its 5×5 median;
* no-data pixels (32,767 m).

**Physics.** Earth radius 6,371 km and refraction k = 0.13 (an effective radius of R/0.87), as in the
Sightline tool. Each fire burns on a 10 m pyre on a summit. Two fires can see each other when the
straight sight line clears the terrain everywhere between them; the sea counts as sea level.

**Candidate sites.** The candidates are the highest summit in every ~12 km cell, 1.04 million in all.
The chain therefore gives an upper bound: a finer search could save a few fires.

**The search.** A breadth-first search from Plymouth finds the fewest fires to reach every summit. It
is deliberately conservative:
* terrain is max-filtered at ~500 m;
* sightlines need 25 m of clearance, ramping in from 1 km of each fire.

**Verification.** Every summit and every sightline of the final loop, ships included, is re-checked
against zoom-10 terrain sampled every 150 m with no margin. The fire goes on the true zoom-10 summit
within 600 m. The search and verification were repeated until everything passed: 0 bad summits and
0 blocked sightlines out of 163.

**Fire ships.** A ship carries its fire 30 m up the mast, so it can be seen 21 km away. A search over
island stepping stones (Iceland, the Faroes, Jan Mayen, Svalbard and Bear Island) minimizes the total
number of fires plus ships. Ships only sit on open water, and each one must be visible from its
shore fire.

**The North Atlantic gap.** The last land fire is on a 3,368 m summit in East Greenland. The nearest
Icelandic summit that could answer it is 350 km away, but the two horizons together reach only
317 km. Iceland to the Faroes and the Faroes to Britain are far wider.

**Assumptions.**
* The air is perfectly clear. Real beacon chains, such as England's in 1588, spaced fires roughly
  10–30 km apart.
* Five minutes per relay is an assumption, for the time to spot a fire and light the next.
