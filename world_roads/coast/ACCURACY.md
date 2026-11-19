# Accuracy and sources: every shore on Earth at light speed

**A thought experiment, drawn from real data.** Light cannot carry a fish, and nothing swims at *c*. What is real
is the geometry, the rule and the arithmetic: every coastline, lake shore and river in OpenStreetMap, traced with
one rule, and the time light would take to cover the result. Every number on screen comes from the scripts below.

| Script | Produces |
|---|---|
| `coast.py` | every coastline ring at full resolution (OpenStreetMap land polygons) |
| `fetch_lakes.py` | every lake and reservoir over 5 km² (Overture base/water) |
| `../rivers/fetch2.py` | every OpenStreetMap river line at full resolution (shared with the rivers-and-pi film) |
| `network.py` | one planar network of coasts, lake shores, rivers and links; the right-hand-rule loops |
| `tour.py` | the joins between loops (river crossings, sea swims, leaps) and the whole journey, with the time of every pass |
| `render.py`, `zoom.py`, `labels.py`, `film.py` | the picture; `music.py` the score |

## 1. The rule

The fish keeps the land on its right. At a junction it leaves by the next edge anticlockwise from the one it
arrived on: going up a river that is the rightmost branch, and at a dead end (a river's mapped source) the only
edge is the one it came by, so it turns round and comes back down the other bank. This is the wall-follower rule
for mazes. Applied to a planar network it traces the boundary of one face (one piece of land) at a time:

* each coastline once, clockwise round the land;
* each lake shore once, anticlockwise round the water;
* each river twice, once up each bank;
* each short link (below) twice.

`network.py` checks that the rule is consistent everywhere. Each turn lands on the land side (no turn ever had to
skip onto the sea side of a shore), and every directed edge has exactly one successor, so the walks are closed
loops. That gives **1,321,345 loops** from 7,672,164 directed edges.

## 2. The data

**Coastlines.** OpenStreetMap land polygons, `land-polygons-complete-4326` from osmdata.openstreetmap.de, data of
2026-11-17. 835,314 rings (835,313 islands and continents, plus the Caspian, the one sea that is a hole in the
land), 79,148,560 vertices, **1,957,245 km** measured vertex to vertex (haversine). The file cuts polygons at the
180° meridian and closes Antarctica along the South Pole; those 641 artificial segments (3,790 km) are not coast
and are removed, leaving **1,953,455 km**.

Africa is its own ring, because OpenStreetMap draws the Suez Canal as coastline. The Americas are one ring: the
Panama Canal is not coastline.

**Rivers.** OpenStreetMap `waterway=river` lines via Overture Maps (release 2026-09-23.1, base/water, class
`river`): 2,009,429 lines, 182 million vertices, 9,740,723 km. Streams (`waterway=stream`) are not included, so a
river "starts" where OpenStreetMap starts calling it a river. Everything tagged as a river is kept, including the
few canalised stretches named "canal": the Rhine reaches the North Sea through the Bijlands Kanaal, and leaving
that out cuts the Rhine off. The rivers-and-pi film used a canal filter; this one does not.

* The parts of river lines that run inside a lake are cut away, because the fish follows the lake's shore
  instead: 31,506 lines cut, leaving 9,577,580 km.
* The same stretch mapped twice (same ends, same length) is walked once: 24,679 duplicates dropped.
* That leaves **9,575,379 km** of river, so **19,150,758 km** of riverbank.

**Lakes.** Overture base/water classes `lake` and `reservoir` over 5 km². That is 21,816 polygons, or 17,808 once
touching parts are merged (a reservoir split at a dam is one lake). Outlines are simplified to about 50 m, and
islands in lakes are not traced. A lake is part of the journey only if a river reaches it: 10,213 lakes,
**1,131,929 km** of shore. That figure is low next to full-resolution shorelines, for the same coastline-paradox
reason as below.

## 3. Joining the pieces

OpenStreetMap's river lines do not always meet. A river end (any end where every line meeting there starts there,
or every one ends there) is joined by a straight link to:

* the coast within 1 km;
* a lake shore within 300 m;
* a separate piece of the river network within 1 km. A gap at a dam, for example: the Ohio at Cannelton Locks
  and Dam is two pieces 200 m apart. A link never joins a piece to itself, so a parallel channel does not count.

Each end gets up to three links, the nearest point of each target. Separately, an end joins the nearest end of a
separate piece of **the same river** (same Wikidata item) within 150 km. This covers rivers whose mapping has a
hole where a reservoir is drawn only as water area: the upper Yangtze has a 30 km hole of this kind. There are
4,181 such bridges, 23,495 km in all.

In all there are **587,571 links, 204,255 km**, walked both ways: **408,510 km** (1.7% of the journey). Links are
straight lines and stand in for water the map does not draw.

## 4. Crossing, swimming, leaping

The loops are the shores of separate pieces of land. To get from one to the next the fish can:

* **Cross a river.** Free: the two banks of a channel in a braid, a delta or a lake chain belong to different
  loops, and the far bank is right there. Loops joined this way form one *water body*: **892,547** of them, of
  which **826,952** reach the open ocean (19.28 million km of shore).
* **Swim across open sea.** From one ocean coastline to another, typically to an island. Each swim is a straight
  line counted at full distance, out and back: **826,943 swims, 436,150 km each way, 872,299 km in all** (3.7%).
  Swims are chosen as a minimum spanning tree over coast points about 3 km apart (from a spherical Delaunay
  triangulation). That keeps them short, but it is not an optimal route, and a straight swim can clip a headland.
* **Leap.** **65,595 water bodies** hold 3.36 million km of shore (14%) that no water connects to the sea. Some
  of these are basins that genuinely never reach the ocean: Lake Chad, the Okavango, the Aral Sea (Amu and Syr
  Darya), Balkhash, the Tarim, the Helmand, Lake Eyre, Turkana, the Sahara and Arabian wadis, and the Caspian with
  the Volga and Ural. The rest are rivers whose mapping has gaps wider than the links above. The fish reaches each
  of these by a hypothetical leap: instantaneous, adding no distance, like the jumps between road networks in the
  roads film. Only 12 leaps are long enough to draw as rings at 1080p; most are short hops between river pieces a
  few kilometres apart.

All the joins together form one tree. Crossings are chosen first, then swims by length, then leaps by length. The
fish walks the tree depth first. Going round a loop with the land on its right, wherever a join leaves the loop it
goes over, does everything beyond, and comes back to carry on. So it swims out to each island as it passes the
nearest point of coast. The rule decides the order inside each loop; the tree decides when each island, delta
island and inland basin is visited.

## 5. The journey and the clock

| | |
|---|---:|
| Coastline | 1,953,455 km |
| Lake shores | 1,131,929 km |
| Riverbanks (2 × 9,575,379 km) | 19,150,758 km |
| Links across mapping gaps (both ways) | 408,510 km |
| Open-sea swims (both ways) | 872,299 km |
| **Total** | **23,516,948 km** |
| **At 299,792.458 km/s** | **78.44 s = 1 min 18.4 s** |

* **Start and finish.** The fish starts on the Bow River in downtown Calgary, heading downstream. The walk is
  closed, so it ends there too: last of all it comes up the South Saskatchewan and the Bow to Lake Louise and
  back down to Calgary. Its first right turn is into the Elbow River, 2.8 km from the start. The Little Elbow's
  source is at 135 km, the Elbow's at 183 km, and the fish is back on the Bow at 387 km.
* **The film's clock** shows light-time. During the opening demo and the closing approach the fish is shown in
  slow motion; the speed readout gives its pace, from under 1 km/s up to light speed. Between 32.5 s and
  108.9 s of video the clock runs in real time: 76.4 of the 78.44 light-seconds are shown at true speed.
* **The order of the world**, as the rule and the tree produce it:
  * 0–8 s: Calgary, Hudson Bay, eastern Canada, the Caribbean, the Mississippi;
  * 8–19 s: South America;
  * 19–22 s: western North America, then across the Bering Strait;
  * 22–37 s: east Asia, Southeast Asia, New Guinea, Australia;
  * 38–56 s: India, central Asia, the Middle East, Africa;
  * 57–75 s: Europe, Russia, Siberia;
  * 75–78 s: arctic Canada and home.
* **River names** under the clock name the longest river being swum in each half-second, by its whole
  mapped course (grouped by Wikidata item, only rivers over 1,200 km, only names in Latin script).

## 6. The coastline paradox

The CIA World Factbook gives the world's coastline as 356,000 km (1.19 light-seconds). Measured along every
vertex of OpenStreetMap's coastline it is 1.95 million km, more than five times as much. Neither number is
wrong: a coastline's length depends on the ruler. The finer the detail, the longer it gets (Richardson, Mandelbrot,
"How long is the coast of Britain?", *Science* 1967). This film's total is the length at OpenStreetMap's detail,
and every other number here has the same dependence: lake shores simplified to about 50 m, rivers as single lines
rather than their real banks.

## 7. What is drawn

* **The map.** Natural Earth projection, centred on 10° E, cut at 170° W (the Bering Strait), with the whole of
  Antarctica in view.
* **Colours.** Blue: not reached yet. Pale gold: one bank of a river done. Gold: coast, lake shore, or both
  banks.
* **Swims.** Faint gold lines across the sea, drawn for swims over 40 km.
* **The fish.** A small glowing sprite on the walk's path. In the close-ups a river is drawn as water between
  two banks and the fish swims along its right-hand bank. In the world view, light-speed motion is a streak.
* **Geometry.** Drawn simplified to about 90 m. Every length is measured on the full-resolution vertices.

## 8. Licences

* Coastlines, rivers and lakes © OpenStreetMap contributors, ODbL. Coastlines via osmdata.openstreetmap.de;
  rivers and lakes via Overture Maps (release 2026-09-23.1).
* Font: Inter (SIL OFL).
* Music: synthesised from scratch for this film (`music.py`, with the instruments of `../flights/score.py`).
