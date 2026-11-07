# Every Grain of Us

The film's look and sound are a named, reusable style, **Sandglass**: see `.claude/skills/sandglass/`
(a Claude Code skill: the style specification, how to adapt this code to a new subject, and helper scripts).

(Working title: Grains of Ancestry.) 200,000 years of human migration and mixing, told as coloured sand poured into glass vessels shaped like
the land each population lives on. 1920x1080, 30 fps, about 3 minutes, no audio.

```
python render.py report                 # final ancestry of every vessel (check against the sources below)
python render.py still 40 75 165        # PNG stills at those video times -> out/
python render.py video --workers 4      # the film -> out/grains_of_ancestry.mp4 (~10 min on 4 cores)
python render.py video --preview        # half size, fast
```

Needs numpy, scipy, shapely, opencv-python-headless, pillow, ffmpeg and the Inter font
(`/usr/share/fonts/opentype/inter`). Natural Earth 1:50m land and country outlines are downloaded into
`data/` on first run.

## Glass edition (Blender glass + sandpile)

The second version stands real glass, rendered in Blender, on a dark map table. Sand behaves like sand:
it heaps up where it is poured and avalanches across the land, and migrating sand blows through the air
between vessels.

```
python make_walls.py && python make_table.py                # glass outlines and the printed map on the table
python camera.py --write                                    # glass keyframes along the camera path
blender -b -P blender_scene.py -- --samples 32 --poses data/plate_poses.json   # ~1 min per keyframe on 4 cores
python film.py still 40 112 147                             # stills -> out/
python film.py video --workers 4                            # the film, no captions (SAND_CAPTIONS=1 burns them in)
python film.py script                                       # the narration as a timed .srt, for a voice-over
```

**The camera.** `camera.py` holds the camera path, shared by Blender and the compositor. By default
(`FINALE`) it opens on diorama close-ups (splines through `OPENING`: low over the first sparse grains in
Africa, a look north to Neanderthal Europe, back to East Africa for the exit), settles on the whole map by
0:45 and holds it until today, then comes down among the city mountains and flies low through the
world's great cities (`CITIES`, timed from their distances, looking ahead; `_clearance()` lifts the camera
over any mountain in its way, using the settled heights export_blender.py saves in `data/tour_heights.npy`),
whips across the Pacific (the map wraps between two frames), and ends in Africa before rising to reveal
the world (3:42 in all). `SAND_CAMERA=story` selects the earlier path that follows the story around the
world. For the composited film, Blender renders the glass at keyframes placed wherever reprojecting from the last one
would be more than 1 px off; in between, the two nearest keyframes are reprojected through the plane
halfway up the walls and cross-faded.

**Every grain is a ball.** Each visible grain is drawn as a lit sphere in 3D, as in the full Blender
renders: the top layers of every pile and every grain on a mountain's flank. Sand piles up where
people actually live, and the great cities rise as mountains above the glass. They cast long shadows, and
depth of field matches the Blender lens (f/0.9, focused on the camera's target).

Blender 4.2 LTS (Cycles, CPU) renders the static world once: the table, plus a thick bevelled glass wall
on every vessel outline under studio lighting. It renders two plates, with and without the glass; their
ratio gives the glass's tint and refraction darkening, and their difference gives its highlights. Path
tracing every frame (glass plus hundreds of thousands of grains) would take days without a GPU. So the
moving parts are composited per frame through the same camera:

- **sand.py** holds a sandpile simulation on a 3-px grid. Each cell is a stack of coloured grains. A pile
  two or more grains taller than its neighbour avalanches onto it. Founders pour in at a landing point on
  the side they came from and spread over the land as they multiply. Gene flow pours in around the
  landing point while as much sand makes way elsewhere, and stirring slowly blends the layers. Every
  vessel's grain counts still follow its exact gene pool, so `render.py report` describes this version too.
- **film.py** shades the sand surface in map space (pile lighting from the height field, ambient
  occlusion, cast shadows, a texture of individually lit grains of random size, glints), projects it
  into the Blender camera with the plane homography, and lays the glass over it. Flying sand follows 3D
  arcs with turbulence, is projected with the full camera matrix, and casts shadows on the table.

### Full Cycles render (Blender project)

The whole film can also be path traced in Blender, frame by frame, for the true Cycles look (real
refraction of the sand through the glass, real shadows and depth of field):

```
python export_blender.py --scale 2       # sand records, flight paths, 4K HUD overlay -> out/blender_film/
blender -b -P blender_film.py            # -> out/blender_film/every_grain_of_us.blend
```

`export_blender.py` runs the simulation once and writes every grain the film shows as a record (where it
sits, its radius, the frames it is visible, and its colour easing from one value to the next), every
flying grain with its path sampled at 40 points, and the HUD (year, legend, labels, closing highlights)
as a transparent QuickTime Animation video. `blender_film.py` builds on `blender_scene.py` (table, glass,
lights) and adds the sand as point clouds whose Geometry Nodes pick the records alive at the current
frame, ease their colours, swell new grains in, fly the airborne grains along their paths and run the
closing ripple, so the .blend renders with no Python at render time. Its `Edit` scene joins the
rendered frames, the HUD and the title (editable text strips) into an MP4, and is where a voice-over
goes. It renders in 4K with motion blur (the flying grains carry a velocity attribute), a golden-hour
sun and cool rim light that fade in for the finale, and sand whose every grain varies a little in
shade, hue and shine (noise of its fixed centre) with a touch of subsurface scattering. Render settings
are tuned for speed: adaptive sampling up to 64 samples, OpenImageDenoise, 6 bounces, no caustics, a
fixed noise seed so the still camera does not shimmer.
`blender_film_README.txt` (shipped next to the .blend) explains rendering it.

### City mountains

Height stands for city size. Every urban agglomeration of 1 million people or more in 2025 gets a
mountain of poured sand: the 66 largest by hand (`story.MEGACITIES`, about 5 million or more, rounded; the
camera's flight visits them) and 574 more as hills of the same kind (`story.CITY_HILLS`, read from
`data/wup2018_cities_1m.csv`, extracted from UN World Urbanization Prospects 2018, File 12; places within
0.3 degrees of a megacity are that megacity). A 1-million city is a low mound of 6.6 levels, Calgary 9;
no base is narrower than 2.2 cells (`CITY_MIN_R`), so small cities are mounds, not spikes. Its summit height is `story.city_height`: 22 grain levels for a 5-million city, growing as
population^0.8 (Tokyo 109, New York 64, Los Angeles 46, Toronto 27, Vancouver 13.5), so the order is always by size
without the giants turning into needles. Its flanks keep to a sand-pile slope (`CITY_SLOPE`, about 6
levels per cell, ~48 degrees), so a bigger city spreads a wider base; two to four lesser summits on its
shoulders, a few ridges and gullies, and grain-scale roughness keep it irregular. A coastal or island
city's mountain spills over the glass onto the sea beside it (`sand.SPILL`: those cells join the city's
vessel for the mountain alone), so Tokyo is a whole cone rather than a slab squeezed onto Honshu. City
mountains may hold at most 70% of a region's sand (`CITY_SHARE`); where they would need more (sparse
regions such as the Southern Cone or Mexico), that region's mountains stand steeper instead. They grow
with urbanization (`story.urban_frac`: a quarter of today's size in 1950, half by 1975). Where the
countryside is already high (Java, Japan), a city still rises at least 45% of its height above it
(`CITY_RELIEF`). Everyone else is spread by density (towns and smaller cities: Natural Earth populated places,
without those the city mountains stand for) and levelled off at 60 grains, so only cities rise above the
land. All the mountains are computed at once each frame (`sand.CF`: one entry per city cell).

- **Deep piles.** Density is uncapped and piles stack up to 600 grains, so crowded places can rise
  high (`SAND_TOWERS=0` caps density at 6x the mean instead).
- **Sand settles, then lies still.** Each cell's target is a whole number of grains (the density
  target, dithered once). Grains above target move to nearby cells below it, and once every cell holds
  its target nothing moves, except where people really arrive, grow or mix.

## How it works (first version: render.py)

- **story.py** is the whole story as data: the ancestry components (sand colours), the vessels (groups
  of countries), the event timeline, the clock (video seconds against years ago, log-scaled) and the captions.
- **vessels.py** turns each group of countries into a glass vessel: the union of their outlines, a
  supersampled glass sprite (rim, highlight, tint), and a hex lattice of grain slots inside the glass,
  ordered bottom-up so sand settles to the bottom of each shape. Island chains too thin to hold sand
  (the Caribbean, Hawai'i for Polynesia) are thickened just enough (`buffer`); vessels can still be drawn
  as magnified insets (`inset`), which the film no longer uses.
- **render.py** runs the simulation and paints the frames. Every vessel carries an exact gene-pool
  composition P; the grains are a physical picture of it:
  - *founding*: a handful of real grains flies from the source and the newcomers multiply by budding
    copies of themselves. A small random sample is how the founder effect shows up.
  - *gene flow*: grains stream along the route and land on top. A flying streak is a fifth of a grain
    (`story.FLY_PER_GRAIN` = 5 streaks per landing grain, the same for every flow), so streams stay in
    proportion to one another and the screen says what a streak stands for ("1 streak = N people", under
    the grain scale, until the flight through the cities). As many grains of the colours now over
    their share make way elsewhere in the vessel.
  - *jostling*: neighbouring grains keep swapping places, so a new layer slowly blends in.
  - *isolation and drift*: in an isolated vessel, grains of the ancestral colour turn one by one into a
    new colour (how a new "ancestry component" forms).
  - *archaic humans*: Neanderthals and Denisovans have no countries, so their sand starts inside the
    Eurasian vessels. When modern humans arrive, it is displaced except for the share that mixed in.

`python render.py report` prints where every vessel ends up. Today's values, rounded:

| Vessel | Result |
|---|---|
| Europe | Near Eastern farmer 43%, hunter-gatherer 18%, Iranian/Caucasus 17%, Ancient North Eurasian 14% (the last two arrive with the steppe migration), Neanderthal 2.5% |
| South Asia | South Asian forager (AASI) 46%, Iranian-related 38%, steppe-related about 10%, East Asian 4% |
| New Guinea | Papuan 87%, Denisovan 4.4%, Neanderthal 2.5%, Austronesian 6% |
| East Asia | East Asian 97%, Neanderthal 2.6%, Denisovan 0.2% |
| Island SE Asia | Austronesian (East Asian) 90%, Papuan-related 6.5% |
| Madagascar | African 58%, Austronesian 37%, Near Eastern 4% |
| Polynesia | East Asian (Austronesian) 65%, Papuan 18%, European 12%, Native American 1% |
| North Africa | Near Eastern/Iranian 62%, West African 24%, East African 12% |
| East Africa | East African 52%, Bantu-related 22%, Eurasian 23% |
| Southern Africa | Bantu-related 80%, Khoe-San 12%, European 6% |
| Siberia | European (Russian settlers) 89%, East Asian 8% |
| Mexico & C. America | Native American 59%, European 34%, African 4% |
| Brazil & north | European 60%, African 24%, Native American 14% |
| Andes | Native American 77%, European 17%, African 3% |
| Caribbean | African 53%, European 38%, Native American 7% |
| Southern Cone | European 61%, Native American 33%, African 3% |
| North America | European 73%, Native American 10%, African 9%, East Asian 5%, South Asian 2% |
| Greenland | Inuit (Native American-related, with Paleo-Inuit/Siberian) 75%, European 22% |
| Australia | European 80%, East Asian 10%, Aboriginal 4% |

## Simplifications, said plainly

- The components are the usual ancient-DNA ones (as in ADMIXTURE/qpAdm studies). They are labels for
  ancient populations, not races, and real ancestry is continuous. "Steppe" ancestry is shown as its
  parts (Ancient North Eurasian + European hunter-gatherer + Caucasus).
- Vessels group neighbouring countries. Each one shows the average ancestry of the people living there
  today, so colonised countries (Australia, North America) look mostly European; Indigenous peoples
  and their ancestry continue there, and the within-region hotspots show where. Polynesia is drawn as
  Hawai'i (settled about 1,000 years ago) but stands for Polynesians as a people, whose ancestors
  reached Tonga and Samoa about 2,900 years ago.
- Dates are mid-range estimates; many are debated, and the captions say so: the main exit from Africa
  (70,000-50,000 years ago; earlier ventures about 120,000 years ago left little trace), the settlement
  of Sahul (here 50,000 years ago; possibly 65,000), Neanderthal mixing (mostly 50,000-43,000 years ago,
  carried by every founder group whatever the founding date drawn), and the first arrival in the
  Americas (here 20,000 years ago; the White Sands footprints suggest 23,000-21,000).
- Some components are deliberately broad. "East African" (yellow) also stands for the deep African
  stem that North Africans carried before the Natufian-related and later flows; "South Asian forager"
  also colours the Hoabinhian ancestry of Southeast Asia.
- The African groups at the start are a simplification of a complicated, deeply structured history.
- Sand is proportional to estimated population (1 grain = a few people 200,000 years ago, about
  55,000 today). Before farming these numbers are guesses within a factor of about 3.

## Main sources

- Schlebusch et al. 2017, *Science*; Ragsdale et al. 2023, *Nature*: deep, weakly structured African history
- Hublin et al. 2017, *Nature*: Jebel Irhoud, *H. sapiens* in North Africa ~315,000 years ago
- Green et al. 2010; Prüfer et al. 2014; Iasi et al. 2024, *Science*; Sümer et al. 2025, *Nature*:
  Neanderthal ancestry (~2%) and its timing (~50,000-43,000 years ago); Chen et al. 2020, *Cell*:
  Neanderthal ancestry in Africans
- Reich et al. 2010; Browning et al. 2018, *Cell*; Jacobs et al. 2019, *Cell*: Denisovan ancestry
- O'Connell et al. 2018, *PNAS*; Clarkson et al. 2017, *Nature*: the settlement of Sahul
- Seguin-Orlando et al. 2014 and Sikora et al. 2017, *Science*: Kostenki and Sungir
- Lazaridis et al. 2014, 2016; Haak et al. 2015: European hunter-gatherers, farmers, the steppe migration
- Raghavan et al. 2014; Sikora et al. 2019: Mal'ta, Ancient Paleo-Siberians, Native American origins
- Bennett et al. 2021 and Pigati et al. 2023, *Science*: White Sands; Dillehay et al. 2015: Monte Verde
- Zeder 2011, *Current Anthropology*: the origins of farming
- Narasimhan et al. 2019 and Shinde et al. 2019: South Asia (AASI, Iranian-related, steppe)
- McColl et al. 2018; Lipson et al. 2014, 2018: Southeast Asia and Island Southeast Asia
- Skoglund et al. 2016; Posth et al. 2018; Ioannidis et al. 2020: Lapita, Remote Oceania, Native
  American contact in Polynesia; Kirch 2011 and Wilmshurst et al. 2011: dates of East Polynesia
- Pierron et al. 2017, *PNAS*: Madagascar
- van de Loosdrecht et al. 2018, *Science*: Taforalt (North Africa)
- Llorente et al. 2015 (with its 2016 erratum): Eurasian back-migration into East Africa
- Patin et al. 2017; Sengupta et al. 2021: the Bantu expansion and southern Africa
- Rasmussen et al. 2010; Moltke et al. 2015: Paleo-Inuit and Greenland
- Nägele et al. 2020; Fernandes et al. 2021: the peopling of the Caribbean
- Eltis & Richardson 2010, *Atlas of the Transatlantic Slave Trade*; Micheletti et al. 2020, *AJHG*:
  origins of the enslaved and their descendants
- Bryc et al. 2015; Moreno-Estrada et al. 2014; Kehdy et al. 2015; Montinaro et al. 2015: ancestry in
  the US, Mexico, Brazil and the Caribbean
- Olalde et al. 2019, *Science*: North African ancestry in Iberia; Almarri et al. 2021, *Cell*: the Middle East
- McEvedy & Jones 1978; HYDE 3.2; UN WPP 2024; Denevan 1992; Koch et al. 2019: population
