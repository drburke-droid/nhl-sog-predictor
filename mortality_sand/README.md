# 2,000 years of catastrophe (working title)

A Sandglass film (the style of `ancestry_sand/`, "Every Grain of Us"): the world's people as sand in
glass vessels on a dark table, one grain = 5,000 people (about 1.4 million grains by 2025), from 1 CE to 2025. Wherever and whenever deaths
ran above the normal level for that place and time, that many grains are lifted off the piles and drawn up
out of the frame: first glowing in the colour of their cause where they lay, then pulled into a turning
column over the place, faster and faster, toward the camera.

| Colour | Cause |
|---|---|
| red | war, conquest and mass killing (with the famine and disease a war directly caused) |
| green | epidemic |
| amber | famine |
| blue | earthquake, flood and storm |
| lilac | other years above normal (after 1950: bad flu seasons, heat waves, the post-Soviet mortality crisis and anything not tied to a named crisis) |

```
python build_data.py             # data: populations, excess deaths, the clock -> data/, out/data_check.png
python fit_glass.py              # packing for regions whose people cannot lie inside their glass -> data/pack.json
python make_walls.py && python make_table.py                     # glass outlines, the printed map
blender -b -P blender_scene.py -- --samples 128                  # glass plates for the wide shot (~10 min)
python film.py still 20 50 67 135                                # PNG stills at those video seconds -> out/
python film.py video --workers 4                                 # the film (no sound yet)
python film.py video --from 15 --to 40 --out out/clip.mp4        # a stretch of it
```

Needs numpy, scipy, pandas, shapely, opencv-python-headless, pillow, matplotlib, ffmpeg, the Inter font
(`/usr/share/fonts/opentype/inter`) and Blender 4.2 (only for the plates). Downloads on first run: UN WPP
2024, Our World in Data population, Natural Earth countries, land and populated places.

## What the film shows (this cut)

4:01, 1920x1080, 30 fps, no sound, one still wide shot (`out/two_thousand_years_of_catastrophe.mp4`).
641 million deaths above normal are lifted, about 128,000 grains: before 1950, 546 million from 123
catalogued catastrophes; from 1950, 95 million from the UN's country-by-country death counts. By cause:
war, conquest and mass killing 285 million; epidemics 219 million; famine 103 million; earthquakes, floods
and storms 4 million; other years above normal (after 1950) 31 million. The largest single lifts are the
Great Dying of the Americas after 1492 (53 million), the Second World War (about 68 million with the
Holocaust, China and the war famines), the Black Death (38 million), the Mongol conquests, the 1918
influenza, the Ming-Qing transition, the Taiping Rebellion, COVID-19 (27 million) and the Great Leap
Forward famine (15 million in the UN series).

Not yet done (the rest of the Sandglass workflow): diorama close-ups and a closing flight (glass plates
at camera keyframes, as in `ancestry_sand/camera.py`), the 4K Cycles package, the score and narration.

## What counts as "above normal"

**1950-2023: every country, every year.** UN World Population Prospects 2024 gives each country's deaths
and population every year. A country's *normal* death rate is the smooth trend of its crude death rate
drawn through its ordinary years: an asymmetric least-squares smoother on the log death rate (Eilers &
Boelens 2005; stiffness 300, asymmetry 0.3), which follows the slow fall of the death rate (and, as
populations age, its slow rise) over about a decade but is not pulled up by crisis years. For 2020-23 the
normal rate is the 2012-19 trend projected forward, as in WHO's and The Economist's excess-death models
(a flexible trend fitted through the pandemic would bend up into it). Excess deaths are the deaths above
the trend; a year at or below it lifts nothing. Checks: China 1959-61, 15 million (published estimates
15-45 million); Cambodia 1975-79, 1.6 million; Rwanda 1994, 0.9 million; Bangladesh 1971, 1.5 million;
COVID-19 years 2020-23, 27 million worldwide (The Economist's model: about 27 million; WHO: 14.9 million
for 2020-21 alone). Crises that last much longer than a decade are only partly counted, because the trend
slowly takes them for the new normal: HIV/AIDS in southern Africa, the post-Soviet mortality crisis.

Each country-year is labelled with the named crisis it falls in (`inputs/events_post1950.csv`, from the
research behind `inputs/NOTES.md`): the country's own events first (the most specific in time: COVID-19
over HIV/AIDS in 2020-23), taking at most 1.5 times that crisis's own published yearly toll, so a small
war in a big country does not claim all of its excess; then worldwide rows (the 1957, 1968 and 2009 flu
pandemics, in proportion to their published totals; COVID-19 in every country without a row of its own).
What is left is "other years above normal": bad flu seasons, heat waves, the post-Soviet mortality crisis.

**1 CE-1949: the catastrophes big enough to have been counted.** No annual death counts exist for most of
the world before the twentieth century, so this part of the film shows every war, epidemic, famine and
disaster with a published death-toll estimate of about half a million or more
(`inputs/crises_pre1950.csv`: central estimate, low-high range, sources, confidence). Each event's deaths
are spread evenly over its years. The many smaller bad years of the pre-modern world (a failed harvest
here, a local epidemic there) are not shown: before 1950 the film undercounts by design.

**Population.** Our World in Data's long-run series (HYDE 3.3 before 1800, Gapminder to 1950), summed
into the vessels at anchor years (every century to 1700, then 1750, 1800, 1850, 1900), and UN WPP 2024
every year from 1950. Between anchors, the catalogued losses are taken out as they happen and regrow
with a 30-year e-folding time, corrected so the population still passes through every anchor. The
Americas skip HYDE's 1600 anchor: their collapse after 1492 follows the death-toll literature (Koch et
al. 2019) and re-joins HYDE in 1700. Asian Russia (the Siberia vessel) is a share of Russia's population
(5% before 1800, a quarter today).

**The clock.** Time runs at a steady pace through quiet centuries and slows whenever the world's excess
deaths rise: seconds per year = a base pace + 0.25 x millions of excess deaths that year (3-year running
mean). So a catastrophe's time on screen grows with its toll, and while it lasts sand lifts at a steady
rate of about 4 million deaths (800 grains) a second. The year on screen always says where we are.

## Simplifications, said plainly

- One grain is 5,000 people, the same in every year, so lifted sand is directly comparable across 2,000
  years. The balls are two thirds the size of the migration film's (2-px cells instead of 3), so five
  times as many fit a landscape of the same shape; pile heights are scaled to match.
- Before 1950, deaths are central estimates of debated numbers (the ranges are in the catalogue); the
  Plague of Justinian, the An Lushan rebellion's census collapse and the Congo Free State are among the
  most uncertain. Events are spread evenly over their years.
- After 1950, the method is statistical and uniform; for countries with little vital registration the UN
  series are themselves modelled, so their "normal" years are smooth and only the crises the UN put in
  stand out.
- A grain is lifted from the piles where the catastrophe struck, in proportion to the people living
  there: from a country's land (after 1950), or around the place the catalogue gives (before). Soldiers
  count against their home region, not where they fell (the film takes grains out of a population, as
  the UN's death counts do), and Middle Passage deaths against the African region they were taken from.
- Population decline stands in for excess deaths in the Americas after 1492 (Koch et al. 2019; the
  regional split is ours), and several Chinese dynastic collapses rest on census drops that are partly
  people leaving the tax registers: those central values are a third to a half of the drop.
- All sand stays inside its region's glass: city mountains are not allowed to spill onto the sea
  (Tokyo, Osaka, Shanghai and Mumbai stand within their coasts). Three regions hold more people than
  their land can carry as sand at 5,000 a grain: their countryside would stand at its height limit
  everywhere, filling the Thar desert and Borneo, with walls at the coast. Their piles are packed
  (`fit_glass.py`: the largest share of grains at which, in 2025, no more than 15% of the countryside
  stands at the limit), the same factor in every year: South Asia 1 grain = 10,400 people, Island SE Asia
  11,600, the Caribbean 6,200 (said on screen). Every other region, East Asia and Japan included, is at
  5,000. Lifted grains are 5,000 people everywhere: in a packed region more grains rise than leave its
  piles, so the death streams compare across the world and each region's losses keep their share of its
  people. Whether to pack at all followed from the catastrophes being visible in the piles themselves
  (the Andes lose 80% of their people after 1525, Mexico 70%, Europe 47% in the Black Death), so the
  common scale had to hold everywhere it could.
- The resting sand is one colour: it is people, all alike.

## Main sources

- UN DESA, World Population Prospects 2024 (deaths, population, crude death rate, 1950-2023)
- Our World in Data, population (HYDE 3.3; Gapminder; UN WPP)
- Eilers & Boelens 2005, "Baseline correction with asymmetric least squares smoothing"
- The crisis catalogue's own sources: `inputs/crises_pre1950.csv` and `inputs/NOTES.md`
- Cities: UN World Urbanization Prospects 2018; Chandler 1987 as geocoded by Reba, Reitsma & Seto 2016
