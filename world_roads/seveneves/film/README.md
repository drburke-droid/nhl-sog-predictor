# The film: Seveneves in Blender (Cycles)

Everything on screen is driven by the simulation in the folder above.

| File | What it does |
|---|---|
| `look.py` | Photoreal Earth (8K day map, city lights on the night side, ocean glint, clouds and cloud shadows in the surface shader, limb atmosphere) and Moon (LRO colour with true LOLA displacement via adaptive subdivision) |
| `moon_pieces.py` | The seven pieces, cut on the simulation's power-diagram cells (`data/cells.json`). Each has LRO/LOLA outside, mantle rock with glowing crack networks inside, and a slump shape key |
| `data_prep.py` | Turns simulation output into per-shot arrays (`../out/film_data/`) |
| `shots.py` | The shots: A the Moon, B breakup, C merge, D wide, E the rain, F the steam world, H the impactor ladder |
| `impact.py` | The impact ladder: the intact Moon hit by 1 m to 100 km rocks (flash, crater, ejecta), the Earth afterwards, the ejecta's path to Earth, and a fireball entering the air |
| `scale.py` | The scale cards: each rock as a turning, procedurally built asteroid (fractured boulder, rubble pile, spinning top, cratered potato) next to something familiar at true scale: a person (1 m), a 20-storey office block (60 m), an 828 m supertall in a city (1 km), an Everest-height peak rising from the plains (10 km), and the curved Earth seen from orbit with the dinosaur-killer beside it (100 km). Also the line-up of all five rocks. Saves per-frame label positions |
| `render.py`, `render_all.sh`, `render_scale.sh` | Render frames (resumable). Quality is `animatic`, `preview`, `draft` or `final` (1920×1080, 16 samples, OIDN) |
| `assemble.py` | Captions, live counters from the simulation, title cards, ladder circles drawn to scale, and the encode |
| `tv.py` | The silent waiting-room cut (five rounds, verdict stamps, explainers, labels that follow the rendered objects) and its narration script `tv_script.md` |

## Assets (not in the repository)

```bash
mkdir -p /opt/seveneves_assets/moon /opt/seveneves_assets/earth
cd /opt/seveneves_assets/moon
curl -O https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_8k.tif
curl -O https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16.tif
cd ../earth
for f in 8k_earth_daymap.jpg 8k_earth_nightmap.jpg 8k_earth_clouds.jpg 8k_earth_normal_map.tif 8k_earth_specular_map.tif 8k_stars_milky_way.jpg; do
  curl -O https://www.solarsystemscope.com/textures/download/$f
done
```

Blender 4.5 LTS (`blender-4.5.16-linux-x64`). On 4 CPU cores a final frame takes about 30–90 s.

## Run

```bash
python3 film/data_prep.py                  # after nbody.py v150_film ... and tracers.py v150_film
film/render_all.sh animatic 2              # every other frame, ~40 min
python3 film/assemble.py animatic
film/render_all.sh final                   # the real thing, many hours; resumable
python3 film/assemble.py final
film/render_scale.sh animatic              # the scale cards and the line-up (~1.5 h)
python3 film/tv.py animatic                # the waiting-room cut (uses final frames where they exist)
```

The scale cards are honest about size: a rock's size is the mean of its length, width and height, and the
familiar objects are true size (person 1.75 m, 20 storeys of 3 m, a stepped 828 m tower, a summit at 8,849 m).
The rocks hover a little so they can turn. The 10 km card is a Himalaya-style procedural range, not a model of
Everest itself. In the 100 km card the air is drawn as glowing layers that thin out with height (scale height
14 km, a little thicker than the real 8 km so it shows).
