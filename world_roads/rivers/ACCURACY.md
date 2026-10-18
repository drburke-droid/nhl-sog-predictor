# Accuracy and sources: do rivers wind by π?

The claim: divide a river's length by the straight-line distance from its source to its mouth (its
*sinuosity*), and on average you get π. We measured the 1,000 longest river systems on Earth in
OpenStreetMap. Every number in the video comes from the scripts named below.

| Script | Produces |
|---|---|
| `fetch2.py` | every river line at full resolution, every lake and reservoir over 5 km² |
| `systems.py` | the river network, its links across lakes and mapping gaps, the 1,200 longest systems' main stems |
| `features.py` | quality checks, the first 1,000 that pass, and their features (`data/rivers.json`) |
| `analyze.py` | the statistics (`data/analysis.json`) |
| `meander.py` | the meandering-river model in the Stølum section |
| `videodata.py`, `video.py`, `music.py`, `assemble.py`, `thumbs.py` | the film |

## 1. The background

* **Einstein, 1926.** A. Einstein, "Die Ursache der Mäanderbildung der Flußläufe und des sogenannten
  Baerschen Gesetzes", *Die Naturwissenschaften* 14, 223–224 (1926). He explains why tea leaves gather
  in the middle of a stirred cup: friction slows the swirl near the bottom, and the water there drifts
  inwards. In a river bend, the same secondary flow erodes the outer bank and deposits on the inner one,
  so bends grow. The explainer's drawings are schematic.
* **Stølum, 1996.** H.-H. Stølum, "River Meandering as a Self-Organization Process", *Science* 271,
  1710–1713 (1996). In a model river meandering across a flat plain, bends grow until loops touch and
  cut off. Sinuosity rises and collapses, and settles around π. Summaries give his simulated range as
  about 2.7 to 3.5 ([Wikipedia: Sinuosity](https://en.wikipedia.org/wiki/Sinuosity);
  [Autodesk, 2023](https://www.autodesk.com/blogs/water/2023/09/05/can-you-really-calculate-a-rivers-path-using-pi/)).
  Wikipedia also notes that later measurements found averages below 2. We did not have the paper itself.

## 2. The rivers

**Source.** Overture Maps release 2026-09-23.1, base/water, class `river`: OpenStreetMap's
`waterway=river` lines. Streams (`waterway=stream`) are not included, so a river's mapped source is where
OpenStreetMap starts calling it a river. That is usually downstream of the farthest trickle used for
published lengths.

| | |
|---|---:|
| River lines | 2,009,429 |
| Left out: irrigation canals mapped as rivers (named "Canal", "Kanal", "канал", …; compound names such as "L'Yonne - Canal du Nivernais" are kept) | 1,636 |
| Vertices | 181,967,942 |
| River, end to end | 9,732,786 km |
| Lake and reservoir outlines over 5 km² | 21,816 (18,413 lakes) |

**The network** (`systems.py`). Lines meet at their ends, or where one line's end lies on another line.
That gives 2.71 million junctions and 2.62 million stretches, directed downstream as OpenStreetMap draws
them.

**Across lakes:**
* A river that ends in a lake joins the lake's outflow. Inflows and outflows must meet the lake within
  about 3 km of its outer shore. The outlines have no holes, so rivers on islands would otherwise count.
* An outflow must not climb more than 30 m away from the lake within 15 km. That catches inflows drawn
  the wrong way.
* A stub that stops within 20 km doesn't count where the lake has an outflow running at least twice as
  far.
* Lakes with no outlet have no outflow: the Caspian, Aral, Balkhash, Chad, Eyre and others, by name.
* Where the same river (by name or Wikidata item) both enters and leaves, that is the outflow. The
  crossing counts as a straight line.
* In a reservoir with the river drawn through it, tributaries join that line instead.

**Over gaps in the mapping:**
* A river that stops short of another (often at the bank of a wide river whose centre line runs
  mid-channel) joins the nearest river. The reach is 300 m near the coast, 10 km inland, and 25 km
  inside a reservoir.
* A link never points upstream, since that would make a loop. A link over 1 km may not point back the
  way the river came.
* Where two links would form a loop, the shorter is kept and the other is chosen again.
* A river with a hole in its own mapping joins the next start of the same Wikidata item up to 150 km
  ahead.

**Main stems.** Braided channels and lines drawn against the flow still leave loops in the network.
1,312 stretches (0.05%) are dropped to break them. The *main stem* is then the longest path
from any source to each mouth, which is how hydrologists define a river's length. Two mouths whose
stems mostly overlap (delta channels) count as one river. This picks the longer branch around every
braided island, so lengths in braided rivers run slightly long.

**The 1,000** (`features.py`). Of the 1,200 longest systems, 60 failed the quality checks, 51 of them
ahead of the 1,000th river kept:
* unnamed;
* more than 15% gap links;
* a fragment: stops inland within 15 km of a longer river's stem, where the mapping has a gap;
* a duplicate: more than 30% of its course lies within 2 km of a longer system's.

The 1,000 run from 7,096 km (Amazon) to 294 km. Lake crossings make up 7,838 km of their 735,919 km
(1.1%) and gap links 7,669 km (1.0%).

**Lengths compared** with commonly cited ones (e.g. Wikipedia's list of river systems by length).
OpenStreetMap's lines are drawn in fine detail, and the main stem takes the longest route through
braids, so measured lengths usually come out longer. Where OpenStreetMap's mapped source differs, the
length moves either way.

| River | Here | Commonly cited |
|---|---:|---:|
| Amazon | 7,096 km | 6,400–7,000 |
| Nile | 7,015 km | 6,650 |
| Yangtze | 6,487 km | 6,300 |
| Ob–Irtysh | 6,145 km | 5,410 |
| Mississippi–Missouri | 6,120 km | 6,275 |
| Yenisei (Angara–Selenga) | 5,951 km | 5,539 |
| Yellow River | 5,611 km | 5,464 |
| Paraná | 5,102 km | 4,880 |
| Congo | 4,325 km | 4,700 |

The Congo comes up short. OpenStreetMap does not map the Luvua's first stretch out of Lake Mweru, so
the Chambeshi headwaters don't connect. The main stem starts on the Lualaba instead.

**Straight line.** The great-circle distance between the main stem's source and its mouth (Earth radius
6,371.0 km).

## 3. The results (`analyze.py`)

| | |
|---|---:|
| Total length ÷ total straight line, all 1,000 ("all together" on screen) | **2.07** |
| Average of the 1,000 ratios | 2.16 |
| Median | 1.98 |
| Middle half | 1.70–2.41 |
| Within 0.2 of π (2.94–3.34) | 36 |
| Within 0.1 of π | 19 |
| Above π | 83 |

The running ratio on the gauge is the total length so far divided by the total straight line so far,
with rivers added longest first.

**The nearest to π** of all 1,000 is the **Brahmaputra**: 3,958 km along the river, 1,252 km in a
straight line, 3.16. Its main stem starts on the Yarlung Tsangpo's headwaters in Tibet.

## 4. What the near-π rivers share (`features.py`, `analyze.py`)

Per river:
* **Elevation** every 1 km along the stem, from the zoom-8 Terrarium mosaic (Mapzen/AWS elevation
  tiles, about 600 m pixels at the equator), the one built for the signal-fire film. That mosaic is
  max-pooled, so the lowest of the 3×3 pixels around each point is taken, to stay near the valley floor.
* **Gradient**: the drop from source to mouth, divided by the length.
* **Lowland share**: the share of the stem below 200 m.
* **Meanders and bends**: the stem measured with a 10 km ruler (the divider method). Meanders are the
  stem's full length ÷ its 10 km-ruler length: wiggles under 10 km. Bends are the 10 km-ruler length ÷
  the straight line: the course bending on larger scales. Their product is the sinuosity (to within
  1%).
* **Where**: the mouth's latitude, and its continent from Natural Earth (Russia east of 60°E counted
  as Asia).

**Near π (36) against the other 964** (medians; Mann–Whitney p):

| | Near π | Others | p |
|---|---:|---:|---:|
| Share of the course below 200 m | 75% | 51% | 0.01 |
| Median elevation | 108 m | 191 m | 0.03 |
| Fall per km | 0.96 m | 1.30 m | 0.10 |
| Distance of the mouth from the equator | 15.6° | 30.5° | 0.03 |
| Length | 474 km | 488 km | 0.78 (no difference) |

**By gradient**, all rivers in each band together:

| Fall per km | Rivers | Ratio |
|---|---:|---:|
| over 3 m | 181 | 1.90 |
| 1–3 m | 402 | 2.01 |
| 0.3–1 m | 293 | 2.11 |
| 0.1–0.3 m | 91 | 2.19 |
| under 0.1 m | 33 | 2.55 |

**By latitude of the mouth**: within 10° of the equator, 16 of 190 (8.4%) are near π. At 10–23.5°, 2.3%;
at 23.5–40°, 2.6%; at 40–60°, 1.4%; beyond 60°, 5.0%.

**Flat Arctic tundra rivers** (fall under 0.3 m/km, more than 80% of the course below 200 m, mouth
beyond 60°N): 22 rivers, median 2.73, the closest of any group. On screen this is "about 2.7".

**Two routes:** 18 of the 36 get most of their extra length from meanders and 18 from bends.

These are associations among 36 rivers, not a proof of cause.

## 5. The model river (`meander.py`)

This follows the Howard & Knutson (1984) centre-line model, the kind Stølum used:
* Each point moves sideways at a rate set by the curvature at that point (weight −1) and by the
  curvature upstream, weighted by exp(−0.3 × distance in channel widths) (weight 2.5).
* A neck closer than 1.2 widths is cut through, leaving the loop as an oxbow lake. Small random bank
  disturbances keep new bends starting.
* The river is 160 channel widths from end to end, run for 300,000 steps.
* Over the last two-thirds of the run, sinuosity averages **2.74** (standard deviation 0.62, range about
  1.5–4). The video says so ("This one averages 2.7").

We did not tune the parameters toward π. Stølum's own runs settled near π.

## 6. Rendering

* **The globe:** the signal-fire film's night globe (Natural Earth coasts and borders; relief from the
  same elevation mosaic). Close-ups of small rivers use sharper relief patches.
* **Rivers on the globe:** simplified to about 1 km. Close-ups draw the main stem at full detail
  (simplified to about 30 m).
* **Names:** OpenStreetMap's local names, transliterated, or the common English name for well-known
  rivers.
* **The cold open:** a frame of the model river where its ratio is 3.14, scaled so the straight line is
  100 km. It is an illustration, not a real river.
* **Music:** synthesized for the video. The melody plays the digits of π up a D minor pentatonic scale.

## 7. Credits

* River data © OpenStreetMap contributors (ODbL), via Overture Maps Foundation.
* Elevation: Terrarium tiles (Mapzen, hosted on AWS Open Data).
* Coasts and countries: Natural Earth.
