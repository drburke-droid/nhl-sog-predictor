# Seveneves simulator: what if the Moon broke into seven pieces?

In Neal Stephenson's *Seveneves*, an unknown "Agent" breaks the Moon into seven large pieces. They
collide and grind each other down, the debris cloud grows "exponentially", and about two years later
the **Hard Rain** begins: a bombardment that heats the sky, sterilises the surface and lasts
5,000 to 10,000 years.

This folder is a physics simulator built to test that scenario, with every dial exposed.

## The pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Breakup & fragments | `nbody.py` | Splits the Moon into 7 uneven pieces (a power-diagram cut of a uniform sphere). Gives them just enough energy to escape each other, plus a chosen leftover drift speed `v_disperse`. Then integrates the Sun, the Earth and every fragment above 80 km with REBOUND (IAS15, full mutual gravity). Collisions halt the integrator. Outcomes follow Leinhardt & Stewart (2012) and Kokubo & Genda (2010): merge, erode, shatter or hit-and-run. Debris is split into a power-law size distribution. |
| 2. Debris | `tracers.py` | Follows every debris parcel (the breakup's own debris and debris from fragment collisions) as a test particle. Each parcel feels the Earth, the Sun's tide and every fragment above 170 km. Integration is adaptive Dormand–Prince 5(4) (numba, parallel). Each parcel ends by hitting the Earth (below 100 km altitude), hitting a fragment, leaving Earth's sphere of influence, or still orbiting at the end. Crater ejecta from fragment hits (Housen & Holsapple 2011) become the next generation: this tests the novel's "exponential cascade" directly. |
| 3. Climate | `climate.py` | A global energy-balance model driven by the rain's kinetic energy. It uses real steam tables (the oceans must be boiled, metre by metre), the runaway-greenhouse limit on outgoing heat (~282 W/m², Goldblatt et al. 2013) and a deep ocean. |
| 4. Results | `analyze.py` | Mass budget (where the Moon went), rain power, temperature and fraction of the ocean boiled, out to 10,000 years. Can re-weight the breakup debris without re-running, because tracers are test particles. |
| Ensemble | `ensemble.py` | Fragments only, 6 random piece shapes × 5 dispersal speeds. |
| Picture | `render_preview.py` | A top-down preview animation of the first years. |

## Run it

```bash
pip install rebound numba scipy numpy pillow    # plus ffmpeg
python3 nbody.py v150 v_disperse=150 years=50   # ~15 s
python3 tracers.py v150 frame_years=10.0        # ~30 min on 4 cores
python3 analyze.py v150                         # prints the summary, writes out/v150/summary.json
python3 render_preview.py v150 years=3
python3 ensemble.py procs=2                     # 30 fragment-only runs
```

Main dials (`nbody.py` `DEFAULTS`): `v_disperse` (how violently the Moon breaks), `debris0` (how much
of the Moon the breakup turns straight into small debris, default 2%), `debris0_v_median` (how fast that debris
leaves), and the collision constants `cstar`, `mubar`, `size_slope`, `f_energy`. In `analyze.py`:
`f_space` (share of the entry heat radiated straight back to space, default 0.5).

`out/` holds the large intermediate files and is not committed. Its small summaries are copied into
`results/`.

See [ACCURACY.md](ACCURACY.md) for what is real, what is approximated, and what is left out.
