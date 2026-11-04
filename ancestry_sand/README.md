# Grains of Ancestry

200,000 years of human migration and mixing, told as coloured sand poured into glass vessels shaped like
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
python make_table.py                                        # the printed map on the table
blender -b -P blender_scene.py -- --samples 512             # glass plates + camera (about 15 min on 4 CPU cores)
python film.py still 40 133.5 163                           # stills -> out/
python film.py video --workers 4                            # the film -> out/grains_of_ancestry_glass.mp4
```

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

### Population, where people live, and ancestry within a region

- **Sand stands for people.** The total amount of sand follows long-run world population estimates, and
  each vessel's share follows historical estimates (1 CE, 1500, 1800, 1950, today). The on-screen
  "1 grain ≈ N people" line rises from a few people per grain to about 55,000 today.
- **Sand settles where people live.** Each cell's target height comes from a smoothed density map of
  Natural Earth populated places. Northern Canada, Siberia, the Sahara and inland Australia stay thinly
  covered, while the Nile, the Ganges plain and eastern China pile high.
- **Ancestry varies within a region.** `story.ANCESTRY_GEO` gives hotspots where a component is more
  common inside a vessel: for example, Native American ancestry in the US Southwest, Alaska and the
  Amazon, African ancestry in the US South and Bahia, and European ancestry in southern Brazil.
  Grains swap places towards their hotspots, and new grains grow there preferentially. These
  movements never change a vessel's totals.
- **Magnified islands.** Sardinia, the Andamans, the Caribbean, Hawai'i and Madagascar are drawn at
  true size in glass, plus a framed, labelled inset (×k) that holds their sand. They keep a minimum
  amount of sand so they stay visible.
- **Megatowers (experiment).** Running with `SAND_TOWERS=1 python export_sand.py 163` drops the
  density cap and lets piles stack up to 600 grains. That way, cities rise as towers. The Blender stills
  are in `videos/ancestry/blender_samples/megatowers_*.jpg`.

## How it works (first version: render.py)

- **story.py** is the whole story as data: the ancestry components (sand colours), the vessels (groups
  of countries), the event timeline, the clock (video seconds against years ago, log-scaled) and the captions.
- **vessels.py** turns each group of countries into a glass vessel: the union of their outlines, a
  supersampled glass sprite (rim, highlight, tint), and a hex lattice of grain slots inside the glass,
  ordered bottom-up so sand settles to the bottom of each shape. Islands too small to hold sand
  (Sardinia, the Andamans, the Caribbean, Hawai'i for Polynesia) are drawn as magnified insets joined
  to their real location by a hairline.
- **render.py** runs the simulation and paints the frames. Every vessel carries an exact gene-pool
  composition P; the grains are a physical picture of it:
  - *founding*: a handful of real grains flies from the source and the newcomers multiply by budding
    copies of themselves. A small random sample is how the founder effect shows up.
  - *gene flow*: grains stream along the route and land on top. Each visible grain stands for a share of
    the flow, so its colour pours into the surface layer, and some of the existing sand makes way.
  - *jostling*: neighbouring grains keep swapping places, so a new layer slowly blends in.
  - *isolation and drift*: in an isolated vessel, grains of the ancestral colour turn one by one into a
    new colour (how a new "ancestry component" forms).
  - *archaic humans*: Neanderthals and Denisovans have no countries, so their sand starts inside the
    Eurasian vessels. When modern humans arrive, it is displaced except for the share that mixed in.

`python render.py report` prints where every vessel ends up. Today's values, rounded:

| Vessel | Result |
|---|---|
| Europe | Near Eastern farmer 43%, hunter-gatherer 19%, Iranian/Caucasus 17%, Ancient North Eurasian 15% (the last two arrive with the steppe migration), Neanderthal 2.5% |
| Sardinia | farmer ancestry 78%, the "island" of early-farmer ancestry |
| South Asia | South Asian forager (AASI) 46%, Iranian-related 38%, steppe-related about 10% |
| New Guinea | Papuan 89%, Denisovan 4.6%, Neanderthal 2.5% |
| Andaman Islands | 97% their own ancestry |
| Madagascar | African 58%, Austronesian (East Asian + Papuan-related) 37%, Arab/Near Eastern 4% |
| Polynesia | East Asian (Austronesian) 65%, Papuan 19%, European 12%, Native American 1% |
| Mexico & C. America | Native American 59%, European 34%, African 4% |
| Brazil & north | European 61%, African 24%, Native American 14% |
| Andes | Native American 77%, European 17% |
| Caribbean | African 60%, European 29%, Native American 10% |
| North America | European 74%, Native American 10%, African 8%, East Asian 5% |
| Australia | European 80%, East Asian 10%, Aboriginal 4% |
| Southern Africa | West-Central African (Bantu) 60%, Khoe-San 25%, European 11% |

## Simplifications, said plainly

- The components are the usual ancient-DNA ones (as in ADMIXTURE/qpAdm studies). They are labels for
  ancient populations, not races, and real ancestry is continuous. "Steppe" ancestry is shown as its
  parts (Ancient North Eurasian + European hunter-gatherer + Caucasus).
- Vessels group neighbouring countries. Each one shows the average ancestry of the people living there
  today, so colonised countries (Australia, North America) look mostly European, although Indigenous
  peoples there remain genetically distinct. Polynesia is shown as Hawai'i, and it stands for
  Polynesians as a people.
- Dates are mid-range estimates; many are debated (the date people left Africa; the first arrival in the
  Americas, here about 20,000 years ago; when Neanderthals mixed in, here about 55,000-50,000 years ago).
- The African groups at the start are a simplification of a complicated, deeply structured history.
- How full a vessel is stands, very roughly, for population growth.

## Main sources

- Schlebusch et al. 2017, *Science*: Khoe-San divergence 260,000-350,000 years ago
- Green et al. 2010; Prüfer et al. 2014; Sümer et al. 2024 and Iasi et al. 2024: Neanderthal ancestry
  (~2%) and timing
- Reich et al. 2010; Jacobs et al. 2019: Denisovan ancestry in Papuans and Island Southeast Asia
- Lazaridis et al. 2014, 2016; Haak et al. 2015: European hunter-gatherers, early farmers, Ancient North
  Eurasians, Iranian/Levantine farmers, the steppe migration
- Raghavan et al. 2014; Sikora et al. 2019: Mal'ta, Ancient Paleo-Siberians, Native American origins
- Narasimhan et al. 2019, *Science*: South Asia (AASI, Iranian-related, steppe)
- McColl et al. 2018; Lipson et al. 2018: Southeast Asia
- Skoglund et al. 2016; Posth et al. 2018; Ioannidis et al. 2020: Lapita, Papuan admixture in Remote
  Oceania, Native American contact in Polynesia
- Pierron et al. 2017: Madagascar
- van de Loosdrecht et al. 2018: Taforalt (North Africa)
- Marcus et al. 2020: Sardinia
- Llorente et al. 2015: Mota and Eurasian back-migration into East Africa
- Patin et al. 2017: the Bantu expansion
- Bryc et al. 2015; Moreno-Estrada et al. 2014; Kehdy et al. 2015: ancestry in the US, Mexico and Brazil
- Mondal et al. 2016: Andamanese
