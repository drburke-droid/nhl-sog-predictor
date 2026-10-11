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
