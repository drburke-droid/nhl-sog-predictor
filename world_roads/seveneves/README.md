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
| The Agent | `lunar_impact.py` | The question turned around. An ordinary asteroid hits the *intact* Moon. How big must it be to hurt the Earth? Crater ejecta faster than lunar escape speed are followed with the same tracer integrator, from 6,000 launch sites. The ladder runs from "a Tunguska's energy every day for a year" to "the Moon shatters". |
| The ladder | `impact_ladder.py` | Ever bigger objects hit the intact Moon (1 m to 1,000 km). For each: the crater, the ejecta reaching the Earth and when, the meteoroid storm (satellite hazard), big fireballs, warming, and how often objects that size hit the Moon and the Earth. Writes `results/impact_ladder.json`. |
| People | `casualties.py` | For each impactor: blasts as strong as Chelyabinsk and Tunguska, expected deaths and injuries (with a range), blasts per day, the risk to a given city, and satellite exposure. Writes `results/casualties.json`. |
| Picture | `render_preview.py` | A top-down preview animation of the first years. |
| Waiting-room film | `film/tv.py`, `film/scale.py` | A silent, self-explaining loop: five rounds, each opening with the rock rendered beside something familiar at true scale (a person, an office block, the tallest building, an Everest-height peak, the Earth from orbit), a size ladder on screen, verdict stamps, explainers, and a narration script (`film/tv_script.md`). |
| Film | `film/` | A photoreal Blender/Cycles film driven by the simulation; see [film/README.md](film/README.md). |
| Page | `results.py`, `build_page.py` | The findings page `results/seveneves.html`, built from `page_template.html` and `results/*.json`. |

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

## What it found (default: pieces drift apart at 150 m/s, 2% of the Moon made into debris by the breakup)

* **No runaway cascade.** Across 30 runs the fragments collided 66 times, and 64 of those were merges.
  Pieces 900 km across meet at about their mutual escape speed (1.2 km/s). Shattering them takes about
  4.4 km/s, faster than two bodies orbiting the Earth at the Moon's distance can ever meet (2.9 km/s).
  Debris hitting the fragments knocks loose 1/58,000 of its own mass. About half the Moon re-forms,
  and the rest is slung out of Earth orbit.
* **The rain starts in days.** The first debris from the breakup lands on day 2. 0.12% of the Moon
  lands (8.5×10¹⁹ kg). In the first year the planet receives on average ~90,000 W/m², about 370× the
  sunlight it absorbs.
* **The oceans boil.** 75% of the oceans become steam. The surface peaks at 363 °C and stays above
  100 °C for ~3,600 years, because a wet planet can't radiate more than ~282 W/m².
* **Every setting sterilises the surface.** Even with no breakup debris at all, collision debris boils
  17% of the oceans. In two-thirds of the fragment-only runs at least one whole fragment of 80 km or
  more hits the Earth within 50 years.
* **The ladder (the film).** Problems for the Earth start at ~1 km. Such an impact on the Moon gives a
  week of meteoroids ~10,000× the normal background, which would wreck satellites, and possibly up to
  ~100 Chelyabinsk-size fireballs. It happens to the Moon about once every 14 million years. A 10 km
  impactor on the Moon, every ~2 billion years, sends ~150,000 Chelyabinsk-size airbursts over a few
  years but warms the Earth by only 0.01 °C. Changing the climate takes ~100 km, and nothing that big
  crosses the Earth's orbit today.
* **The Agent.** At 20 km/s, an asteroid 80–190 km across hitting the Moon causes +10 °C on Earth.
  Starting to boil the oceans takes 175–405 km. Shattering the Moon takes a body ~3,000 km across.

See [ACCURACY.md](ACCURACY.md) for what is real, what is approximated, and what is left out.
