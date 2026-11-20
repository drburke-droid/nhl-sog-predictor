# The film: Seveneves in Blender (Cycles)

Everything on screen is driven by the simulation in the folder above.

| File | What it does |
|---|---|
| `look.py` | Photoreal Earth (8K day map, city lights on the night side, ocean glint, clouds and cloud shadows in the surface shader, limb atmosphere) and Moon (LRO colour with true LOLA displacement via adaptive subdivision) |
| `moon_pieces.py` | The seven pieces, cut on the simulation's power-diagram cells (`data/cells.json`). Each has LRO/LOLA outside, mantle rock with glowing crack networks inside, and a slump shape key |
| `data_prep.py` | Turns simulation output into per-shot arrays (`../out/film_data/`) |
| `shots.py` | The shots: A the Moon, B breakup, C merge, D wide, E the rain, F the steam world, H the impactor ladder |
| `render.py`, `render_all.sh` | Render frames (resumable). Quality is `animatic`, `preview`, `draft` or `final` (1920×1080, 16 samples, OIDN) |
| `assemble.py` | Captions, live counters from the simulation, title cards, ladder circles drawn to scale, and the encode |

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
```
