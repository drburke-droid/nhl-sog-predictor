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
