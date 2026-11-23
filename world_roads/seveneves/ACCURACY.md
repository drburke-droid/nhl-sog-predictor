# Accuracy and sources: the Seveneves simulator

What is computed from physics, what is a modelling choice, and what is left out.

## 1. What is real physics

* **Gravity.** The Sun, the Earth and every fragment above `m_cut` (1e-4 Moon masses, radius ~80 km)
  are integrated together with REBOUND's IAS15 integrator (Rein & Spiegel 2015), with full mutual
  gravity and adaptive steps. That covers the solar tide that stretches and tilts orbits at lunar
  distance, close flybys, and slingshots out of Earth orbit.
* **Debris orbits.** Debris parcels are test particles in that same field: the Earth, the solar tide
  (written exactly in the Earth's accelerating frame), and every fragment above 1e-3 Moon masses
  (radius ~170 km). Fragment positions are interpolated from samples every 0.1 day (cubic Hermite).
  Each parcel is integrated with adaptive Dormand–Prince 5(4) at relative tolerance 1e-9.
  `validate.py` checks this against REBOUND for 40 test particles over one year, half of them on
  Earth-grazing orbits. Median position difference after a year: **5 km** on orbits 384,000 km
  across. The worst case is a few thousand km, for particles that pass close to the fragment and go
  chaotic. The impact time of the Earth impact seen by both codes agrees.
* **Hitting the Earth.** Debris counts as arriving when it crosses 100 km altitude. Entry speeds come
  out of the integration: about 11 km/s, a little above escape speed, as expected for material
  falling from lunar distance.
* **Energy.** Each arrival brings ½mv². At 11 km/s that is ~6×10⁷ J per kg, about 15 times the
  energy of the same mass of TNT.
* **Boiling the oceans.** Water vapour in the sky is in equilibrium with the ocean surface (column
  = p_sat(T)/g, Wagner & Pruss 2002 saturation curve). The enthalpy of every kilogram of vapour comes
  from the IAPWS-IF97 saturated-steam table. Turning the whole ocean (1.335×10²¹ kg) into steam costs
  **5.6×10¹² J per m² of Earth**, about 2.9×10²⁷ J in total.
* **The runaway greenhouse.** While liquid water remains, a planet cannot radiate more than about
  282 W/m² (Goldblatt et al. 2013, Nature Geoscience 6:661; Kopparapu et al. 2013). Today Earth absorbs
  ~240 W/m² of sunlight, so once the oceans are steam the Earth can shed only ~42 W/m² of extra heat.
  That is why it takes thousands of years to cool down afterwards.

## 2. Modelling choices (the dials)

| Dial | Default | Why | Effect |
|---|---|---|---|
| Seven pieces | power-diagram cut of a uniform sphere, seed 7: 6–19% of the Moon each | The novel says seven big pieces, no sizes | `ensemble.py` repeats with 6 different cuts |
| `v_disperse` | 150 m/s | The pieces get exactly enough energy to escape each other, plus this leftover drift. The book's pieces stay in roughly lunar orbit, which needs a drift well below the Moon's 1 km/s orbital speed | Swept: 50 – 1,200 m/s |
| `debris0` | 2% of the Moon | How much the breakup itself turns straight into small debris. The book doesn't say | The single most important dial for Earth. `analyze.py` re-weights it without re-running |
| `debris0_v_median` | 0.25 × the Moon's escape speed (~600 m/s) beyond the expansion | Debris thrown out at a fraction of escape speed | Faster debris reaches Earth sooner and in larger share |
| Collision outcomes | Leinhardt & Stewart 2012 "universal law", c* = 5, μ̄ = 0.37 (their gravity-dominated planet values); Kokubo & Genda 2010 merge / hit-and-run boundary | Standard scaling laws for planet-sized bodies | Gentle collisions merge |
| Debris sizes | cumulative N(>m) ∝ m^-0.85, mass carried by the biggest pieces | Typical of collisional fragments | Sets how many "dinosaur-killers" a parcel holds |
| Debris speed | median 0.6 × mutual escape speed beyond escape, capped by half the impact energy | SPH collision studies put debris at about the escape speed | |
| Crater ejecta | Housen & Holsapple 2011, "water/competent rock" fit (μ = 0.55, C₁ = 1.5, k = 0.2) | The most generous of their fits to a cascade | Tests the novel's runaway cascade |
| `f_space` | 0.5 | Half the entry heat is radiated straight back to space from the hot upper atmosphere | Doubling or halving the rain gives the scale of this |

## 3. The lunar-impact ladder (`lunar_impact.py`)

* **Impactor:** stony (2,600 kg/m³), 45° impact angle, 12, 20 or 50 km/s.
* **Escaping mass:** Housen & Holsapple (2011) ejecta scaling,
  M(>v)/m = (3k/4π)[(v/(C₁U))^(-3μ) − n₁³], with U the vertical impact speed. "Rock" uses
  μ = 0.55, C₁ = 1.5, k = 0.2; "regolith" uses μ = 0.41, C₁ = 0.55, k = 0.3. For impactors big enough
  to erode the Moon as a whole, the Leinhardt & Stewart (2012) largest-remnant law takes over
  (whichever predicts more escaping mass). Above Q*_RD the Moon shatters. Basin-scale craters
  (D ≳ 300 km) are beyond the strict validity of crater scaling, so treat the upper rungs as order of
  magnitude.
* **Where it goes:** 6,000 test particles launched from random sites in a 45° cone around the local
  vertical, at random times over one lunar month. Speeds are spread evenly in log(v) from escape speed
  to 16 km/s, then re-weighted to each case's speed distribution. They are followed for 30 years with
  the Sun and the real Moon. The last fifth of the run is extrapolated as before. 10% of the
  Earth-bound mass arrives within 9 days, half within about 400 days, and 90% within 6 years.
* **On Earth:** the same climate model. The rungs use the first year's energy, the peak month's
  heating, the peak warming and the share of the ocean boiled. "A Tunguska's energy every day" is an
  energy equivalence. Most of that mass would arrive as small fireballs.

## 3b. The impact ladder and how often (`impact_ladder.py`)

* **Craters.** Collins, Melosh & Marcus (2005) transient-crater scaling with lunar gravity
  (1.62 m/s²), impactor 2,600 kg/m³, target 2,800 kg/m³. The simple-to-complex transition is at
  18 km, and complex depths follow Pike (1977). Small impacts are capped by target strength at
  ~20× the impactor's diameter; LRO's newly formed craters from ~1 m impactors are 10–20 m across.
  Formation time ≈ √(D_transient/g).
* **Meteoroid storm (the satellite hazard).** The densest 7 days of arriving lunar ejecta (kg/day),
  divided by the ~110 tonnes/day of meteoroids that reach the Earth normally (Love & Brownlee
  1993). This compares mass, and uses mass as a proxy for the number of satellite-damaging grains.
  Studies of asteroid 2024 YR4 hitting the Moon (December 2032, a few-percent chance as estimated in
  2025) find short spikes in small-particle flux that are larger than our week-averaged mass ratio.
* **Big fragments.** The size of the largest escaping fragments is not well known. We use ~1% of the
  impactor (range 0.3–3%, from spall scaling: Melosh 1984), with the arriving mass spread evenly over
  log(size) from 10 µm upwards. The counts of Chelyabinsk-class (≥20 m) and Tunguska-class (≥50 m)
  fragments are order of magnitude only. Below ~1 km impactors there are none at all, because the
  largest fragments are smaller than 20 m.
* **How often.** Below 100 m: the bolide flux of Brown et al. (2002), N(>E) = 3.7 E^-0.90 per year
  (E in kilotons), at 20 km/s. Above: Harris & Chodas (2021), with impacts of ≥1 km every ~700,000 years
  and ≥10 km every ~100 million years, extrapolated beyond. The Moon receives ~1/20 of the Earth's
  impacts (smaller target, weaker gravitational focusing; Le Feuvre & Wieczorek 2011). No known
  near-Earth asteroid is bigger than ~35 km. Above that, an "interval" longer than the age of the Solar
  System means it effectively doesn't happen today. It did happen ~3.9 billion years ago, when
  Imbrium and the other basins formed.

## 3c. Blasts, casualties and daily life (`casualties.py`)

* **Energy, not size.** Lunar debris reaches the Earth at ~11.1 km/s, slower than Chelyabinsk
  (19 km/s) or Tunguska, so a 20 m piece of Moon carries only ~0.17 Mt, a third of Chelyabinsk's
  0.5 Mt. Blasts are therefore counted by energy. "Chelyabinsk-strength" is ≥0.5 Mt, a lunar
  fragment ≥29 m. "Tunguska-strength" is ≥10 Mt, a fragment ≥77 m.
* **Damage areas** scale with energy^(2/3), calibrated on the two real events:
  * Severe (homes flattened): Tunguska's 2,150 km² of flattened forest at ~12 Mt. That is roughly the
    ≥2.5 psi zone.
  * Light (windows broken, injuries): Chelyabinsk's window damage out to ~50 km at 0.5 Mt.
  * Small rocks burst high, so light damage fades in between 0.05 and 0.5 Mt, and severe damage
    between 1 and 10 Mt.
* **Casualties.**
  * Severe zone: 20% killed and 45% injured, from the U.S. Office of Technology Assessment (1979),
    *The Effects of Nuclear War*. Its blast-only figures are ~5% killed at 2–5 psi and ~50% at
    5–12 psi; the inner third of the zone is above 5 psi.
  * Light zone: 0.07% injured (Chelyabinsk: ~1,500 of ~2 million people).
  * Blasts fall at random places, so the expected number of people in a zone is its area × 15.9 per km²
    (8.1 billion people over the whole surface, oceans included). Totals saturate at the world
    population.
* **The big uncertainty** is the size of the largest fragments (0.3–3% of the impactor). For a 10 km
  impactor the death estimate runs from ~0 (largest pieces ~30 m) through ~40 million (1%, ~100 m)
  to ~60 million (3%, ~300 m). The film quotes the middle value and says so.
* **City risk** is the chance that a Chelyabinsk-strength blast lands within 50 km of a given city in
  the first year, when half the debris arrives.
* **Satellites:** the meteoroid storm factor × 7 days = years of normal meteoroid exposure packed into
  the first week.
* **Not included:** tsunamis from ocean impacts of the largest fragments, fires, crop and economic
  effects, deaths from infrastructure loss (no GPS or satellites), and any warning or sheltering.

## 4. The film (`film/`)

* **Driven by the simulation.** Fragment positions, merges and debris are the simulation's own (the
  12-day opening comes from a 10-minute-cadence run). The Sun's direction comes from the integrator.
  Earth arrivals (time and entry point) come from the 50-year run. Rain power, temperature and ocean
  boiled come from the climate model.
* **The opening re-run.** For the film's first 12 days the breakup was re-run with the Moon at
  gibbous phase (`moon_f = 3.95`), so its familiar face is lit. At the main run's phase the Moon is a
  thin crescent seen from Earth. That re-run has one merge (hour 36) instead of the main run's two,
  which is the system's chaos, not a different physics. The long-term numbers and the findings page
  use the main run.
* **Shapes.** The seven pieces are cut on the simulation's own power-diagram cells. They slump towards
  lumpy spheroids within a few hours, because rock is about 100× too weak to hold a 900-km wedge
  against its own gravity (ρgh ≈ 800 MPa against ~10 MPa). The slumping is a visual morph, not a
  simulation.
* **Glow.** Exposed interior glows by depth (1,650 K near the centre falling to 650 K below the crust,
  roughly following lunar selenotherms) along crack networks, fading over ~5 h as the skin cools. The
  night-side sky glow is a blackbody at the radiative temperature of the heated upper atmosphere,
  2σT⁴ = (1 − f_space)·P. Its brightness is scaled by the eye-weighted Planck radiance, normalised so
  980 K (the 30-day peak) is about as bright as city lights.
* **Not to scale in time.** Time-lapse shots freeze the Earth's spin. Meteor streaks are
  representative: their number follows the rain rate, and they cluster around real entry points.
* **Textures.** Moon: NASA SVS CGI Moon Kit (LRO LROC WAC colour, LOLA elevation, public domain).
  Earth: Solar System Scope 8K maps (CC BY 4.0, derived from NASA Blue Marble and Black Marble).
* **Scale cards (`film/scale.py`, `film/assets.py`).** Sizes are true: a rock's size is the mean of its
  length, width and height. Beside it: a 1.75 m person (Blender Studio's CC0 realistic human base mesh, arms
  lowered from its A-pose, dressed with smoothed copies of its own body regions), a 60 m office block (20
  storeys of 3 m, a real facade texture), an 828 m Y-plan tower that steps back in a spiral like the world's
  tallest building, and the real Mount Everest: Copernicus GLO-30 elevation data (30 m) with the summit restored
  to the surveyed 8,849 m (30 m cells shave ~110 m off sharp peaks) and the Earth's curvature included. Snow,
  ice, rock and moraine are painted by height and slope, not from imagery. Skies are real HDR photographs whose
  sun lights the scene. The rocks are procedural (fractured boulder; rubble pile like Itokawa; spinning top like
  Bennu; elongated and cratered like Eros; cratered potato like Ida), dark grey-brown like stony asteroids, and
  hover slightly so they can turn. Around Everest the ground is 5.5–7 km high, so the floating 10 km rock towers
  over the summit: it would dwarf the mountain. In the 100 km card the air glows in layers that thin out with
  height (scale height 14 km, a little thicker than the real ~8 km so it reads on screen). The line-up of five
  rocks is not to scale, and says so.
* **The night sky from the ground (`film/night.py`).** Meteor rates are the simulation's first-week averages
  (`storm_factor`) times a normal dark-sky rate of 10 visible meteors an hour: ~30 a second
  (1 km), ~30,000 a second (10 km), ~3×10⁷ a second (100 km). Brightnesses follow a power law (many faint, few
  bright). Debris from the Moon arrives at ~11 km/s, so the meteors are slow, long and orange-tailed, and they
  radiate from one point (they share a direction of arrival). Where there are too many to draw (10 km and
  100 km), the brightest 60,000 are drawn and the rest become an even sky glow. Light: ~1% of the entry energy
  becomes visible light, so the 10 km storm gives roughly full-Moon light and the 100 km storm roughly overcast
  daylight; the 100 km shot is exposed 8 stops darker, like eyes adjusting to glare. A fireball "as bright as the
  Sun" (Chelyabinsk class, visible to ~500 km) shows up in one place's sky about once a day after the 10 km hit
  and every couple of minutes after the 100 km hit; each shot shows one. The 1 m impact flash is a brief,
  star-like point on the Moon's dark side. The 60 m round (~25 shooting stars an hour instead of ~10) has no
  ground view: at real speed you would barely notice it. The stars are a real all-sky photograph.

## 5. Left out

* **Debris hitting debris.** Debris parcels don't collide with each other. Grinding would turn debris
  into dust faster and could randomise its orbits (feeding the Earth-crossing ones) or damp them into
  a flat ring (starving them). `tracers.py` has an optional random-kick mode (`kick_dv`, `kick_tau_years`)
  to bound the first effect. It is not used in the headline runs.
* **Feedback of debris on fragments.** Debris that hits a fragment is recorded (and makes crater
  ejecta), but the fragment's mass and motion in the N-body run don't change. The debris is at most a
  few per cent of the Moon.
* **Small fragments' gravity on debris.** N-body fragments between 80 km and 170 km pull on each
  other but not on the debris, and debris doesn't collide with them.
* **"Escaped" is not "safe".** Pieces thrown out of Earth's sphere of influence go round the Sun on
  orbits close to Earth's. Over 10⁴–10⁶ years many of them would come back and hit the Earth (or the
  Moon's remnant). They are not followed.
* **Radiation pressure and Poynting–Robertson drag** on dust; **Earth's oblateness**; **atmospheric
  drag** on grazing orbits (anything below 100 km altitude simply counts as arrived).
* **Climate is one global column.** No geography, seasons, clouds, impact winter (dust and soot
  blocking sunlight), crust melting, or chemistry. Land under a sky radiating as hot as the rain
  would be far hotter than the global mean. The model answers one question: is there enough heat to
  boil the oceans, and how long does it take to get rid of it?
* **The Agent.** Whatever broke the Moon is not modelled. Only its results are set by the dials above.

## 6. Sources

* Collins, G. S., Melosh, H. J. & Marcus, R. A. (2005). Earth Impact Effects Program. MAPS 40:817.
* Brown, P. et al. (2002). The flux of small near-Earth objects colliding with the Earth. Nature 420:294.
* Harris, A. W. & Chodas, P. W. (2021). The population of near-earth asteroids revisited and updated. Icarus 365:114452.
* Love, S. G. & Brownlee, D. E. (1993). A direct measurement of the terrestrial mass accretion rate of cosmic dust. Science 262:550.
* Le Feuvre, M. & Wieczorek, M. A. (2011). Nonuniform cratering of the Moon and a revised crater chronology. Icarus 214:1.
* Melosh, H. J. (1984). Impact ejection, spallation, and the origin of meteorites. Icarus 59:234.
* U.S. Congress, Office of Technology Assessment (1979). *The Effects of Nuclear War*.
* Popova, O. P. et al. (2013). Chelyabinsk airburst, damage assessment, meteorite recovery, and characterization. Science 342:1069.
* Pike, R. J. (1977). Size-dependence in the shape of fresh impact craters on the Moon.
* Leinhardt, Z. M. & Stewart, S. T. (2012). Collisions between gravity-dominated bodies. I. ApJ 745:79.
* Kokubo, E. & Genda, H. (2010). Formation of terrestrial planets from protoplanets under a realistic accretion condition. ApJL 714:L21.
* Housen, K. R. & Holsapple, K. A. (2011). Ejecta from impact craters. Icarus 211:856–875.
* NASA Scientific Visualization Studio, CGI Moon Kit (svs.gsfc.nasa.gov/4720).
* Rein, H. & Spiegel, D. S. (2015). IAS15: a fast, adaptive, high-order integrator. MNRAS 446:1424.
* Goldblatt, C., Robinson, T. D., Zahnle, K. J. & Crisp, D. (2013). Low simulated radiation limit for runaway greenhouse climates. Nature Geoscience 6:661–667.
* Wagner, W. & Pruss, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance. J. Phys. Chem. Ref. Data 31:387.
* Stephenson, N. (2015). *Seveneves*. William Morrow.
