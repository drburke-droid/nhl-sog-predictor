# All world roads at light speed

A 4 min 40 s (280 s) cinematic visualization of a thought experiment: one white-hot point of light
drives the world's major roads at the speed of light. Roads glow cyan until the dot reaches them,
then stay gold, so over the runtime the Earth turns from a cyan network into a golden one.

* **Real roads.** The road geometry is real: OpenStreetMap via Overture Maps, streamed from about
  35 GB of source.
* **Real route.** The route is a real walk computed over that graph, with Euler paths, a greedy
  T-join for repeated travel, and Dijkstra deadheads.
* **Honest jumps.** Every sea crossing is shown as a hypothetical jump, never as a road.
* **What's approximate.** See [ACCURACY.md](ACCURACY.md) for exactly what is and isn't real.

## Deliverables

Rendered copies are committed in [`media/`](media/) (`render.py` writes them to `out/`):

| File | What |
|---|---|
| `world_roads_light_speed.mp4` | Full film: 1920×1080, 30 fps, H.264 High / yuv420p, **8,400 frames = 280.0 s** |
| `world_roads_preview.mp4` | 15 s preview, **time-compressed 18.7×** (labelled "PREVIEW" on screen) |
| `still_*.png`, `checkpoints_contact_sheet.jpg` | Frames at the 40 s art-direction checkpoints |
| `data_stats.json`, `route_stats.json` | Measured road lengths and route statistics behind every number quoted |

`python3 verify.py` probes both videos with ffprobe. For the full film it asserts H.264, yuv420p,
1920×1080, 30/1 fps, exactly 8,400 decoded frames, and a container duration of 280.0 s.

## Run it

```bash
pip install -r requirements.txt        # plus ffmpeg with libx264, and the Inter font
./run_all.sh                           # everything below, end to end
```

Or step by step:

| Step | Command | Time (4 vCPU) | Output |
|---|---|---|---|
| 1. Stream & reduce Overture roads | `NPROC=8 python3 fetch_overture.py` | ~25 min, resumable | `data/overture/part-*.npz` |
| 2. Build routable graph + density rasters | `python3 build_graph.py` | ~25 s | `data/graph.npz`, `density.npz`, `land2x.png`, `stats.json` |
| 3. Compute the coverage walk | `python3 route.py` | ~25 s | `data/route.npz`, `route_stats.json` |
| 4. Bake render layers | `python3 render.py prepare` | ~20 s | `data/layers.npz` |
| 5. Preview / stills | `python3 render.py preview` · `python3 render.py still 0 40 120` | ~2 min | `out/` |
| 6. Full film | `python3 render.py video --workers 3` | ~25 min | `out/world_roads_light_speed.mp4` |
| 7. Verify | `python3 verify.py` | | ffprobe report |

Step 1 also needs `data/ne/ne_10m_land.geojson` (Natural Earth land); `run_all.sh` downloads it.
Everything is deterministic: there are no random choices in routing, and the render's random seeds
are fixed.

Tunables, set as environment variables for `route.py`:
* `BAND_KM` (default 1500): width of the sweep band. Smaller values follow the waypoint order more
  strictly at the cost of more repeated travel.
* `MATCH_KM` (default 300): search radius for the T-join pairing.
* `ALPHA` (default 150): turn-cost weight in the Euler path.

## How it works

* **`fetch_overture.py`** reads only `subtype`, `class` and `geometry` from each parquet row group
  over HTTP range requests and never holds the global dataset in memory. Per file it keeps:
  * exact haversine length per road class;
  * a length-per-pixel density raster on a 3840×2160 canvas;
  * for motorway–tertiary, graph edges snapped to a 7680×4320 grid.
* **`build_graph.py`** merges those parts, drops Antarctic roads (south of 60°S), inverts the
  projection to get edge lengths, and bridges short class-filter gaps (land-only, ≲13 km). It
  labels connected components and gives each edge a sweep key (`waypoints.py`).
* **`route.py`** splits edges into key bands, and each band into connected pieces. Within a piece it
  runs a greedy T-join and then Hierholzer, giving an exact Euler path that starts where the dot
  arrives. Between pieces it takes the shortest road path, or a hypothetical jump when the networks
  are disconnected. It asserts that the walk is continuous and covers 100% of edges.
* **`render.py`**:
  * Rasterizes every routed road on a 2× canvas with the time the dot first reaches it, and gives
    minor-road density pixels the time of their nearest routed road.
  * Sorts all lit pixels by that time, so turning roads gold is an incremental scatter-add per
    frame.
  * Composites, on each frame:
    * a white-hot wake on freshly reached roads;
    * the dot sprite (core, glow, halo and a faint anamorphic streak) moving at constant
      speed along the walk, with a short motion trail;
    * jump rings with labels, and a restrained city sparkle;
    * two-scale bloom, a filmic tone map, and the text overlays.
  * Frames stream straight into FFmpeg in parallel chunks, which are then concatenated without
    re-encoding.

## Data sources and licences

* **Roads:** © OpenStreetMap contributors (ODbL), via the Overture Maps Foundation transportation
  theme, release `2026-09-23.1`
  (`s3://overturemaps-us-west-2/release/2026-09-23.1/theme=transportation/type=segment/`).
* **Land polygons:** Natural Earth 1:10m land (public domain),
  via `github.com/nvkelso/natural-earth-vector`.
* **Font:** Inter (SIL OFL).

## Key numbers

The walk is distance-proportional and was not forced onto the checkpoint schedule. Even so, the dot
is in eastern North America at 0:40, Tierra del Fuego at 1:20, European Russia at 2:00, Korea at
3:20 and New Zealand at 4:40.

| | |
|---|---:|
| All mapped road segments (Overture/OSM) | 93.4 M km |
| Vehicular roads excl. tracks | 64.2 M km |
| Routed motorway–tertiary (real geometry) | 17.7 M km |
| Routed graph after generalization | 12.0 M km, 2.56 M edges, 1,100 components |
| Coverage walk (incl. repeats) | 16.65 M km (+38.7%), 1,415 hypothetical jumps |
| On-screen distance scale | t × c, reaching 83.94 M km at 4:40 (estimate from the brief) |

## Spin-off Shorts: the lost photon, and 48 cities

Both are vertical 1080×1920 / 30 fps Shorts. They're committed in `media/lost_photon/` and
`media/cities/`. Their accuracy notes are sections 8 and 9 of [ACCURACY.md](ACCURACY.md).

**The lost photon** (`turns/`). Light leaves Los Angeles and flips a coin at every intersection. How
long until it reaches New York?

```bash
cd turns
python lost_photon.py sim 2000     # 2,000 random walks -> data/lost_photon_sim.json (~2 min)
python lost_photon_render.py       # the median walk, in real time -> out/lost_photon.mp4
```

* The shortest drive takes light 16 ms.
* The median coin-flip photon takes **14.7 s** (893× longer) and drives 4.4 million km.
* Across 2,000 photons, the luckiest took 0.9 s and the unluckiest 1 min 57 s.

**48 cities** (`cities/`). Every street within 15 km of downtown in 48 cities, run one after
another at the speed of light.

```bash
cd cities
python fetch_cities.py      # Overture segments around each centre (~10 s per city)
python build_cities.py      # street graph + coverage walk per city (same method as Calgary)
python race.py              # -> out/city_race.mp4 (63 s) and out/city_race_board.png
```

* Laps run from 11.5 ms (Hong Kong) to 71.7 ms (Delhi).
* All 48 together take 1.69 s.

## Every flight on Earth (`flights/`)

The video has two parts.

* **One modeled day of air traffic** (108,996 flights, 24 hours in 12 s). The map starts empty, and
  each route is drawn when a flight on it lands, so the world's route network inks itself in.
* **One plane at the speed of light flies every flight, in real time.** It goes **route by route**:
  it shuttles each of the 17,343 routes as many times as the route was flown that day, then the
  route is erased from the map. The map empties region by region, with a "routes left" countdown
  and a "Now flying" caption for the bigger routes. That's 192.5 million km in 10 min 42 s,
  including 4,959 empty repositioning legs, which add about 2% to the distance.

```bash
cd flights
python schedule.py && python itinerary.py      # modeled schedule (OpenFlights routes) + route-by-route itinerary
python flights_render.py prepare
python flights_render.py video --workers 4    # -> out/every_flight_light_speed_v2.mp4 (1920x1080, 11:02)
python flights_short.py                       # -> out/every_flight_short.mp4 (1080x1920, 36.6 s)
python score.py full && python score.py short # original music, timed to each cut
python thumbnail.py                           # -> out/every_flight_thumbnail.png
```

**Music.** `score.py` synthesizes an original track from scratch, so there are no samples and no
licences or copyright claims to worry about.

* **Instruments:** D minor at 100 BPM, with detuned-saw pads, a pulsing bass, a plucked arpeggio
  through a ping-pong delay, and light drums.
* **Timing:** risers and impacts land on the picture's key moments (takeoff, halfway, "farther than
  the Sun", landing).
* **Lift and resolution:** after "farther than the Sun" the music rises a whole step, and it ends on
  a major chord at landing.
* **Loudness:** normalized to −14 LUFS for YouTube.

The YouTube master in `media/flights/` has the music muxed in and is two-pass encoded to stay under
GitHub's 100 MB file limit.

## Signal fires around the world (`signal/`)

How many mountaintop signal fires would it take to send a message around the Earth? The chain starts
in Plymouth, where beacon fires spread the news of the Spanish Armada in 1588, and goes east. It
crosses Europe and Asia, the Bering Strait, Alaska, arctic Canada and Greenland. There it stops: no
summit on either side of the North Atlantic can see the other. Fire ships close the loop through
Iceland and the Faroes and back to Plymouth.

```bash
cd signal
python fetch_dem2.py      # Terrarium z10 north of 56N, z9 south: despiked, max-filtered sightline grid + summits
python globe_texture.py   # night-globe texture for the animation
./iterate.sh              # fewest-fires search -> fire-ship crossing -> zoom-10 verification, repeated until clean
python render_fires.py    # -> out/signal_fires.mp4 (vertical Short, with music)
```

| | |
|---|---:|
| Mountain fires, Plymouth to East Greenland | **131** |
| Fire ships to close the loop | **14** (2 + 7 + 5) |
| Fires on Iceland, the Faroes and back down Britain | 14 more |
| Relays around the world | **159** (145 fires + 14 ships), 26,512 km |
| At 5 minutes per relay | ≈ 13.2 hours |
| Longest sightline in the chain | 433 km, Mount Foraker → Mount Blackburn (Alaska); clears the Talkeetna Mountains by 178 m |
| Highest fire | Muztagh Ata, China, about 7,480 m in the data |
| Bering Strait | Russia → Alaska in one 189 km jump |
| Greenland → Iceland | 356 km apart; their two horizons only reach 304 km |

*Corrected in October 2026.* The first version said 163 relays (149 fires + 14 ships) and a 350 km gap.
Terrarium shows bedrock instead of ice over much of Greenland, and that made two sightlines across the
ice sheet look clear when they are not. The search now uses Greenland's ice surface; see ACCURACY.md
section 10.

Physics follows the VSL Sightline tool (`drburke-droid/VSL`, `sightline/`): Terrarium elevation, Earth
radius 6,371 km, normal refraction (k = 0.13). Details and limitations are in ACCURACY.md section 10.

### Two long videos (16:9)

**Video 1, one fire around the world (5:56):** the history of beacons, a first-person beacon chain on real
terrain, the curvature explainer, and the loop above. **Video 2, the analog backup (3:27):** if an AI took
over every phone, cable, satellite and radio on Earth, how fast could bonfires warn every capital? The
first fire is lit at Bletchley Park; every fire that sees a lit fire lights its own, so the warning
spreads as a wave and branches. Masters, previews, thumbnails, the Video 2 Short and the YouTube copy are
in `media/signal/` (`YOUTUBE.md`).

```bash
cd signal
python fetch_dem2.py south        # terrain 60S-18N (zoom 9)
python fix_greenland.py           # Greenland's ice surface from zoom 11 (zoom 5-10 show bedrock under the ice)
python fix_greenland.py holes     # fill bedrock patches inside the ice sheet (also: neighbours, holes2)
python fill_holes.py              # other blank Arctic pixels from zoom 8
python world.py                   # global grid, 944,589 candidate summits with their countries
python edges.py                   # every fire-to-fire sightline in the world (23.2 million)
python border.py && python border.py gulf && python border.py flat   # 1 km border rule, flat coasts
python history_check.py           # Aeschylus's beacon chain through the same sightline test
python fpv_terrain.py && python fpv.py && python fpv.py alt   # first-person sequence (real terrain, raymarched)
python globe_texture_world.py     # night-globe texture with relief and borders
python longform.py physics        # the curvature explainer
python long_globe.py hook|history|rules|journey|outro         # Video 1 sections
python assemble.py v1             # cross-fades + score -> out/long/signal_fires_world.mp4
# Video 2
python capitals.py && python capitals.py start    # a beacon for every capital; the first fire at Bletchley Park
python capitals_tree.py           # fastest warning to every capital, zoom-10/12 verified -> data/capitals_tree.json
python capitals.py plains && python capitals_tree.py   # ground-fire corridors where a capital's plain is too flat
python long_globe.py cold|plan|wave|outro2
python assemble.py v2             # -> out/long/signal_fires_ai.mp4
python short_ai.py                # vertical Short -> out/signal_fires_ai_short.mp4
python thumbnail_long.py
```

| Warning every capital by fire (Video 2) | |
|---|---:|
| Capitals warned | **176** of 195 |
| Time to the last one | **14 h 15 min** (171 relays at 5 minutes), Buenos Aires |
| Half the capitals warned by | 4 h 20 min |
| Fires | **1,326** (51 on flat ground, where the "summit" was a data spike) |
| Sightlines | 181,778 km; longest 497 km, Muztagh Ata → the Kunlun |
| Highest fire | Kangchenjunga, 8,330 m in the data |
| Never reached | Sri Lanka and 18 island nations, from Iceland to Tuvalu |
| Aeschylus's chain (458 BC) | all 8 hops pass, Troy → Argos, 558 km |

An earlier draft of the long video used a single chain through as many countries as possible (176, with
787 fires; `iterate_tour.sh`, `data/tour.json`). It backtracked too much and was replaced by Video 2.

Method and limits: ACCURACY.md sections 11–13.

## Hyperacuity: your eyes see smaller than their own pixels (`hyperacuity/`)

A caption-only explainer and test. You can see one pair of bars misaligned by a fraction of a pixel,
finer than the spacing of the cones in your own retina. Six test rounds go from 4 pixels down to ⅛ pixel.
Then come the scale of things (cones, a 20/20 letter, a phone pixel), how a screen draws between pixels,
and a simulated retina that recovers a 5″ shift from its cones. After that: bends, squashed circles,
tilts and the oblique effect, the vernier caliper, and the Amsler grid ("if straight lines ever look
bent, see your optometrist").

```bash
cd hyperacuity
python codec_check.py        # do 1/2, 1/4, 1/8 px offsets survive downscaling + VP9/H.264/AV1? (yes)
python assemble.py render    # render all sections at 4K (4 at a time), then join + score -> out/hyperacuity.mp4
python thumbs.py             # three thumbnails for A/B/C testing
```

* **Rendering:** at 4K, with every offset in pixels of a 1080p frame. Stimuli are drawn with exact area
  coverage, in linear light.
* **Sources and methods:** [hyperacuity/ACCURACY.md](hyperacuity/ACCURACY.md). Master, thumbnails and YouTube copy: `media/hyperacuity/`.

## How long would a Roomba take to drive every road? (`roomba/`)

A first-generation Roomba has no map: it drives until it bumps into something, then turns a random way.
On roads that becomes a rule: at every intersection, take any road except the one you came in on, and
turn back only at a dead end. Run at the speed of light, how long until it has driven every road?
* **New York City:** **1.88 s** typically (200 runs). The run shown spends its last 29% hunting for two
  blocks in Far Rockaway.
* **The contiguous United States:** **2 h 23 min** typically (16 runs, 1 h 33 min to 3 h 48 min). The
  run shown ends on Manasota Key, a barrier island in Florida, after 2.54 billion km; 13 of the 16 runs
  end on a coast.
* **At a real Roomba's top speed (0.5 m/s):** 36 years for New York, 163,000 years for America.
* **Calgary** (a Short): 1.24 s.

```bash
cd roomba
python nyc.py boundary && python nyc.py build && python nyc.py many 200 && python nyc.py feature 30
python city.py 8 && python city.py many 200 && python city.py feature 31   # Calgary
python usa.py build && python usa.py walk 4 && python usa.py more 4 3        # USA graph + 16 walks (~1.5 h on 4 cores)
python usa.py tail 0 20000                                                   # exact replay of the featured walk's ending
python where.py LAT LON ...                                                  # name the last roads (Overture)
python stats.py                                                              # the USA and Calgary numbers
python assemble.py render                                                    # sections (resumable) -> out/roomba_photon.mp4
python short.py && python assemble.py short                                  # vertical Short: America
python calgary_short.py                                                      # vertical Short: Calgary
python thumbs.py                                                             # thumbnails A/B/C
```

* **Walks:** numba, non-backtracking random walk on the half-edge graph, reproducible from the seed
  (`cover.py`). Counts are in blocks (intersection to intersection): 91,022 in NYC, 23.0 million in the US.
* **Sources and methods:** [roomba/ACCURACY.md](roomba/ACCURACY.md). Master, Shorts, thumbnails and
  YouTube copy: `media/roomba/`.

## Fantasy football's most valuable seasons since 1980 (`fantasy/`)

This is a bar-chart race of the all-time best single seasons in a 12-team PPR league, 1980-2025, valued in points over
a replacement player (the 16th QB, 30th RB, 42nd WR and 14th TE each season, counted by the game). The board is a
record book of the 12 most valuable seasons so far; each season's bars grow as it is played. The data is Pro Football
Reference's season tables for 1980-1998 (via fantasydatapros/data) and nflverse's weekly stats from 1999. The film is
`media/fantasy/fantasy_best_seasons.mp4` (2 min 45 s). See [fantasy/README.md](fantasy/README.md).

* The record is Marshall Faulk's 2000 (329.7). LaDainian Tomlinson's 2006 is 0.1 behind and Christian McCaffrey's
  2019 is third.
* The record passed from Billy Sims '80 to William Andrews '81, Eric Dickerson '83, James Wilder '84, Emmitt Smith '92
  and '95, then Faulk.
* The final board is 10 RB seasons and 2 WR seasons (Cooper Kupp 2021, Jerry Rice 1995). The best QB season (Dan
  Marino 1984) and the best TE season (Todd Christensen 1983) were on the 1980s board.
