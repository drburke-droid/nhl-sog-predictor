# The film: Seveneves in Blender (Cycles)

Everything on screen is driven by the simulation in the folder above.

| File | What it does |
|---|---|
| `look.py` | Photoreal Earth (8K day map, city lights on the night side, ocean glint, clouds and cloud shadows in the surface shader, limb atmosphere) and Moon (LRO colour with true LOLA displacement via adaptive subdivision) |
| `moon_pieces.py` | The seven pieces, cut on the simulation's power-diagram cells (`data/cells.json`). Each has LRO/LOLA outside, mantle rock with glowing crack networks inside, and a slump shape key |
| `data_prep.py` | Turns simulation output into per-shot arrays (`../out/film_data/`) |
| `shots.py` | The shots: A the Moon, B breakup, C merge, D wide, E the rain, F the steam world, H the impactor ladder |
| `impact.py` | The impact ladder: the intact Moon hit by 1 m to 100 km rocks (flash, crater, ejecta), the Earth afterwards, the ejecta's path to Earth, and a fireball entering the air |
| `scale.py` | The scale cards: each rock as a turning, procedurally built asteroid (fractured boulder, rubble pile, spinning top, cratered potato) next to something familiar at true scale: a person on a cobbled square (1 m), a 20-storey office block in a city (60 m), an 828 m Y-plan supertall in a skyline (1 km), the real Mount Everest (10 km), and the curved Earth seen from orbit with the dinosaur-killer beside it (100 km). Also the line-up of all five rocks. Saves per-frame label positions |
| `assets.py` | The real-world assets: the CC0 human (posed and dressed), Poly Haven trees and HDR skies, ambientCG facades and ground textures, the Y-plan tower, and the Copernicus 30 m terrain of Everest with its snow, ice and rock material |
| `night.py` | The night sky from the ground for each round: the 1 m flash on the Moon, and the 1 km, 10 km and 100 km meteor storms at real speed, radiating from one point, with a fireball |
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

Free (CC0) assets for the scale cards and night views, and the Everest terrain:

```bash
cd /opt/seveneves_assets
mkdir -p human hdri tex models dem
curl -o human/hbm.zip https://download.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip
for h in kloofendal_48d_partly_cloudy_puresky kloofendal_43d_clear_puresky; do
  curl -o hdri/${h}_4k.hdr https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/${h}_4k.hdr; done
for t in Facade001 Facade005 Facade006 Facade018A Facade019A Facade020A Grass004 Asphalt031 Concrete049 PavingStones070; do
  curl -L -o tex/$t.zip "https://ambientcg.com/get?file=${t}_2K-JPG.zip"; done
# Poly Haven models tree_small_02 and jacaranda_tree (2k .blend + textures), via api.polyhaven.com/files/<id>
for t in N27_00_E086_00 N28_00_E086_00 N27_00_E087_00 N28_00_E087_00 N27_00_E088_00 N28_00_E088_00; do
  curl -o dem/$t.tif https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_${t}_DEM/Copernicus_DSM_COG_10_${t}_DEM.tif; done
```

Then unzip, and run `python3 film/prep_assets.py dem` (cuts `dem/everest.npz`, lat 27.42–28.55 N, lon
86.05–87.95 E) and `blender -b -P film/prep_assets.py -- sun` (finds each sky's sun, `hdri/sun.json`).

Blender 4.5 LTS (`blender-4.5.16-linux-x64`). On 4 CPU cores a final frame takes about 30–90 s.

## Run

```bash
python3 film/data_prep.py                  # after nbody.py v150_film ... and tracers.py v150_film
film/render_all.sh animatic 2              # every other frame, ~40 min
python3 film/assemble.py animatic
film/render_all.sh final                   # the real thing, many hours; resumable
python3 film/assemble.py final
film/render_scale.sh animatic P_lineup S1_1m S2_60m S3_1km S4_10km S5_100km N1_1m N3_1km N4_10km N5_100km
python3 film/tv.py animatic                # the waiting-room cut (uses final frames where they exist)
```

See ACCURACY.md section 4 for what the scale cards and night views get right and what they simplify.
