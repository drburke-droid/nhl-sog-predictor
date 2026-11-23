# Accuracy and sources: a robot vacuum at the speed of light cleans every floor on Earth

The question: a present-day robot vacuum (one that maps the room and cleans in neat rows) moves at the
speed of light. How long would it take to vacuum every floor of every building, first in New York City,
then on Earth? This is the sequel to the Roomba film (`../roomba/`), whose first-generation Roomba had no
map and drove at random. This film's robot is not called a Roomba: it has a map and never repeats a row.

Every number in the storyboard stills comes from the scripts here (`nyc_prep.py`, `nyc_route.py`,
`ghsl.py`, `globe.py`, `frames.py`) or the constants listed below.

## 1. The robot and its rules

* **Rows 25 cm apart.** Mapping robot vacuums clean in back-and-forth rows, then go round the edges. No
  manufacturer publishes its lane spacing, so 25 cm (a typical main brush is about 20 to 25 cm wide) is
  an assumption. Every time on screen scales with 1 / spacing: at 20 cm, every floor on Earth takes 1 h
  16 min of vacuuming instead of 1 h 01 min.
* **Distance = floor area / 0.25 m.** Edges and the turns at the ends of rows are not added.
* **Floors only.** Gross floor area as the city and the world's building statistics count it (inside the
  outer walls, every storey). Walls, furniture and stairs are not subtracted.
* **Light speed the whole way**, 299,792,458 m/s, including between buildings and up and down stairs.
* **Then and now (frame 05).** The 2002 Roomba is the earlier film's simulation (`roomba/room.py`: bump,
  turn at random, 0.3 m/s). The modern robot drives rows 25 cm apart across the same room's free floor,
  then an edge pass round the walls and furniture: 152 m in all, so 8.5 min at the same 0.3 m/s. By then
  the random Roomba has covered 82% of the floor; the modern one 99%.

## 2. New York City

* **Buildings:** NYC Building Footprints (data.cityofnewyork.us, dataset 5zhs-2jue): 1,082,928 polygons
  with roof height. Kept: buildings and garages (feature codes 2100 and 5110), 1,081,950 in all.
* **Floor area:** PLUTO (dataset 64uk-42ks), the city's tax-lot table; `bldgarea` is each lot's gross
  floor area. Footprints are joined to lots by `mappluto_bbl` (which links condominium buildings to their
  billing lot); 99.3% join. A lot's floor area is shared among its footprints in proportion to footprint
  area x floors. Footprints with no PLUTO floor area get footprint x floors (roof height / 3.4 m).
  * PLUTO total: **533.9 million m²**. Sum over footprints: **534.9 million m²** (used on screen).
  * Manhattan 170.9, Brooklyn 144.0, Queens 119.5, the Bronx 71.7, Staten Island 27.8 million m² (PLUTO).
* **Empire State Building:** PLUTO lot 1008350041, 2,812,739 sq ft = **261,312 m²**, 102 floors.
  1,045 km of rows, **3.49 ms**. The X-ray shape uses OpenStreetMap's building parts for it (setbacks at
  the 6th, 25th, 30th, 72nd, 81st and 86th floors, and the mast), from Overture Maps (release
  2026-09-23.1, buildings / building_part). The tower floor in frame 06 is OpenStreetMap's 2,062 m² part:
  8.1 km of rows, 27 µs.
* **Order:** from the Empire State Building, spiralling outward in 150 m rings (`nyc_route.py`).
* **Time:** 7.14 s of vacuuming + 0.25 s of travel (straight lines between buildings, plus up and down
  each building's height) = **7.38 s**. Rows: 2.14 million km = 5.6 x the mean Earth–Moon distance
  (384,400 km).

## 3. Every floor on Earth

* **World total: 273 billion m² (2024).** UNEP / GlobalABC, *Global Status Report for Buildings and
  Construction 2025–2026*. More than three-quarters of it is residential.
* **Where it is:** GHSL (European Commission JRC, Global Human Settlement Layer R2023A, GHS-BUILT-V, epoch
  2025, 30 arcsec), building volume per cell, summed to 5-arcminute cells. Each cell gets 273 billion m² x
  its share of the world's building volume. Checked on New York: in cells inside the city, GHSL's volume
  is within 9% of the city's footprints x roof heights.
* **Why not GHSL's own total?** GHSL's world building volume is 2.61 trillion m³. Divided by the storey
  height that matches New York (4.5 m of volume per m² of floor) it gives about 575 billion m², twice the
  published figure; at 3 m per storey, 870 billion. GHSL counts every built structure it can see from
  orbit (sheds, barns, warehouses, informal housing) and overestimates footprints in sparse areas (its
  built-up surface is 28% above the city's footprints even inside New York). The published total is
  floor area in the sense building statistics use. Using GHSL's own total would roughly double every
  world time.
* **Countries** (Natural Earth 1:50m borders over the scaled grid): China 55.9 billion m² (20%), the US
  34.5, India 18.0, Japan 11.3, Brazil 10.2.
* **Cities** (scaled grid within 40 km of the centre): Greater Tokyo 1.80 billion m² = **24 s**, Beijing
  1.13 (15 s), Shanghai 0.99 (13 s), Delhi 0.60 (8 s).
* **Buildings:** about 2.53 billion (Overture Maps release notes, August 2026).
* **Vacuuming:** 273 billion m² / 0.25 m = 1.09 billion km of rows = **1 h 00 min 43 s** of light; 7.30
  AU, farther than Jupiter (5.20 AU), short of Saturn (9.54 AU).
* **Getting there (≈ 7 min):** hops between buildings, estimated per 5-arcminute cell as a random-scatter
  tour (0.7124 √(N·A)) with each cell's share of the 2.53 billion buildings in proportion to GHSL
  built-up surface, then halved: on New York's real buildings that formula gives 26,800 km against 10,400
  km of nearest-neighbour spacing. Plus every building's height up and down (GHSL volume / surface). About
  123 million km, 410 s. Rounded on screen.
* **Total: about 1 h 08 min.**
* **Order:** New York first, then the hand-placed sweep of the world-roads film (North America, Latin
  America, Europe, Africa, Asia, Oceania), with Antarctica last. The South Pole station as the last floor
  is a choice, not a computed result.

## 4. The dustbin (brief, frame 14)

* **Dust per m²: 0.3 g**, roughly what a robot vacuum picks up from a floor cleaned weekly. This is the
  shakiest number in the film: indoor dust loading varies by orders of magnitude (deep carpet holds far
  more). x 273 billion m² = **82,000 tonnes**.
* **Bin: 150 g** of dust per emptying → **546 million** emptyings. At **10 s** per self-emptying dock
  cycle: **173 years**. Only the robot moves at light speed; the dock is ordinary.

## 5. The physics footnote (frames 15–16)

* Nothing with mass reaches light speed. At 99% of it, γ = 7.09, and a **4 kg** robot carries
  (γ − 1) m c² = 2.19 × 10¹⁸ J = 523 megatons of TNT = **10 Tsar Bombas** (50 Mt each). It has to shed
  that to stop at the end of a row.
* **U-turns:** with rows averaging 4 m (room-sized), 1.09 billion km of rows means about **273 billion**
  U-turns. 273 billion x 2.19 × 10¹⁸ J = 6.0 × 10²⁹ J, about **26 minutes of the Sun's entire output**
  (3.828 × 10²⁶ W).

## 6. Rendering

* **X-ray view (`xray.py`):** every floor of every building is a slab: the footprint (or OpenStreetMap
  part) at that floor's height, projected with an orthographic 3/4 camera. Slabs are summed, so dense
  blocks glow brighter. Blue slabs are not yet cleaned; gold ones are. Buildings beyond the city line (New
  Jersey, Nassau) are not in the data, so they are not drawn.
* **The robot** is drawn at true size (35 cm) only in the magnifier; everywhere else it is the light.
* **Globe (`globe.py`):** brightness is floor area per area of ground (1 − e^(−30·d))^0.8, from the scaled
  GHSL grid; land from Natural Earth.
* **Stills, not the film:** these are key frames for review. Motion and timing notes are in the storyboard.

## 7. The film (`video.py`, `cityfilm.py`, `assemble.py`)

* **Time on screen is the true light time.** Every stretch that isn't real time is labelled: SLOW MOTION
  (×1,000,000,000 when the robot is visible at its true 35 cm and crawls at walking pace, falling smoothly to
  ×5 over New York), TIME-LAPSE ×60 in the living room, SPED UP ×68 over the globe (60 s of film for the
  whole planet). The label is the instantaneous factor, rounded to two significant figures.
* **The hook replays the start of the run** (the Empire State Building, then outward), so Act 3 restarts the
  clock: "Back to the start, slowed right down."
* **The globe's light time** spreads the 7 minutes of travel evenly over the sweep, so the clock reads 1 h 08
  min when the last floor is done.
* **The lift-off** from New York to the globe tilts the 3/4 view to top-down (a 2D warp of the ground plane)
  and cross-fades to the globe at the same scale; for that second the globe interpolates its 5-arcminute
  floor grid.
* **Fast city frames:** with an orthographic camera, zoom and pan are 2D scales and shifts, so every
  building's slab density is rendered once at 2.5, 0.6 and 0.12 px/m and sampled per frame (area-averaged
  mip level, then a sub-pixel affine); buildings are added to a "clean" copy as the light reaches them.
* **Music:** the Roomba film's synthesized theme (`roomba/music.py`), re-cut to this film. No samples.
