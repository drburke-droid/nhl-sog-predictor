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

* Leinhardt, Z. M. & Stewart, S. T. (2012). Collisions between gravity-dominated bodies. I. ApJ 745:79.
* Kokubo, E. & Genda, H. (2010). Formation of terrestrial planets from protoplanets under a realistic accretion condition. ApJL 714:L21.
* Housen, K. R. & Holsapple, K. A. (2011). Ejecta from impact craters. Icarus 211:856–875.
* NASA Scientific Visualization Studio, CGI Moon Kit (svs.gsfc.nasa.gov/4720).
* Rein, H. & Spiegel, D. S. (2015). IAS15: a fast, adaptive, high-order integrator. MNRAS 446:1424.
* Goldblatt, C., Robinson, T. D., Zahnle, K. J. & Crisp, D. (2013). Low simulated radiation limit for runaway greenhouse climates. Nature Geoscience 6:661–667.
* Wagner, W. & Pruss, A. (2002). The IAPWS formulation 1995 for the thermodynamic properties of ordinary water substance. J. Phys. Chem. Ref. Data 31:387.
* Stephenson, N. (2015). *Seveneves*. William Morrow.
