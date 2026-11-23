# Every floor on Earth at the speed of light

The sequel to the Roomba film (`../roomba/`): a present-day robot vacuum, one that maps each room and
cleans in rows 25 cm apart, moves at the speed of light. It cleans every floor of every building in New
York City, then every floor on Earth. Storyboard stills are in `../media/floors/storyboard/`. Sources,
method and every number are in [ACCURACY.md](ACCURACY.md).

| | Light time |
|---|---:|
| One Empire State Building (261,312 m²) | 3.49 ms |
| Every floor in New York (1,081,950 buildings, 535 million m²) | 7.38 s |
| Every floor on Earth (273 billion m², about 2.5 billion buildings) | about 1 h 08 min |

## Run it

```bash
pip install -r ../requirements.txt rasterio
# data (not committed): NYC footprints + PLUTO (NYC Open Data), GHSL GHS-BUILT-V/S 2025 30ss (JRC FTP),
# Natural Earth 50m land + countries -> data/nyc, data/ghsl, data/ne (see the docstrings for exact URLs)
python fetch_parts.py      # OpenStreetMap building parts for Manhattan (Overture) -> data/nyc/parts.npz
python nyc_prep.py         # footprints + PLUTO floor area -> data/nyc/buildings.npz, stats.json
python nyc_route.py        # cleaning order and light times -> data/nyc/route.npz, route_stats.json
python ghsl.py             # GHSL sums on a 5-arcminute grid -> data/ghsl/agg10.npz, totals.json
python globe.py            # world sweep order, country and city floor area -> data/ghsl/sweep.npz
python frames.py           # all 16 stills and 2 thumbnails -> out/frames/
```

The film (3:18): `../../videos/speed_of_light/every_floor_on_earth_at_light_speed_3m18s.mp4`.

```bash
for s in hook room rules esb nyc earth1 far earth2 dust uturn spotless; do python video.py render $s 3; done
python assemble.py         # cross-fades + score -> out/video/every_floor_light_speed.mp4 (master, CRF 17)
```

* `cityfilm.py`: fast city frames. Zoom and pan are 2D with this camera, so slab density is rendered once at
  three scales and sampled per frame; cleaned buildings are added as the light reaches them.
* `video.py`: the eleven sections, their timing (true light time, labelled slow motion) and captions.
* `xray.py`: the X-ray 3/4 renderer. Every floor slab of every building is projected and summed; blue is
  dirty, gold is clean.
* `globe.py`: the night globe, the GHSL floor map scaled to the published world total, and the sweep.
