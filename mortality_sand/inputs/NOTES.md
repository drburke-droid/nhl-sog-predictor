# Notes on the mortality-crisis catalogue

## Files
- `crises_pre1950.csv`: 233 rows, 123 events, years 9 to 1949. The central values sum to about 546M. Confidence is low for 141 rows, medium for 75 and high for 17.
- `events_post1950.csv`: 155 rows, 75 events, years 1946 to 2023. Use these rows only as labels for excess deaths computed from WPP.
- Every number is a count of people. Each row carries its own `deaths_low`, `deaths_high` and `deaths_central` for that region's share.

## How central values were chosen
1. When a careful scholarly estimate exists, it is the central value. Examples: Maharatna for India 1876-78 (8.2M), Rudnytskyi et al. for Ukraine 1932-33 (3.9M), Dyson & Maharatna for Bengal 1943 (2.1M), Koch et al. for the Americas, Hacker for the US Civil War, USHMM for the Holocaust.
2. When estimates spread widely, I took the commonly cited mid-range, never the maximum. Matthew White's medians (necrometrics.com and *The Great Big Book of Horrible Things*) served as a cross-check and often as the anchor.
3. Some totals rest on census drops or contemporary claims: the "11M" Indian famines, the "8-10M" for Dungan or Mahdist Sudan, Persian chronicle tolls. Here the central value sits well below the headline figure, the headline goes in `deaths_high`, and the row is marked `low`.
4. **Census-registration issue (China).** This applies to Xin/Red Eyebrows, the late Han and Three Kingdoms, An Lushan, the Mongol conquest of China, the fall of the Yuan and Ming-Qing. Registered population often fell by 50-70%, but much of that drop is people lost to tax registers, not deaths. Central values are about a third to a half of the census drop: White's 13M for An Lushan (he halves the 26M household-based loss), and 10M for the Three Kingdoms against a census "loss" of 34M.
5. **Regional splits.** Where a published split exists I used it: Erlikman's Soviet republics, provincial tolls for 1876-79 China, SlaveVoyages embarkation regions, country studies for 1918 influenza. Otherwise I split by rough population share and say so in the row's notes. That applies to the Antonine, Cyprian and Justinian plagues, the Crusades, Timur and the Black Death outside Europe.
6. **Americas.** Koch et al. (2019) estimate 60.5M people in 1492 (IQR 44.8-78.2M) and about 55M deaths by 1600, a 90% decline. They publish only hemisphere totals plus a regional discussion, so the split into Mexico 17M, Central America 5M, Caribbean 0.6M, Andes 13M, Colombia/Venezuela 3M, Amazonia 9M, Southern Cone 1.4M and North America 3.5M is mine. North America runs to 1900 because most of its decline came after 1600. Population decline is used as a stand-in for excess deaths.

## Where the grains come from (please check this choice)
The brief says "the region the deaths happened in". For civilians that is where they lived. For soldiers I used their **home region** instead, because the film removes grains from a population: a German soldier killed near Stalingrad was a grain in Germany, not in the steppe. This also matches the post-1950 file, where WPP counts deaths by country of residence. It matters mainly for WWI, WWII, the Napoleonic and Crimean wars and the Paraguayan War. The Middle Passage is handled the same way: those deaths are attributed to the African embarkation region. If you want place of death instead, the WWI and WWII rows are the ones to re-split.

## Avoiding double counting
- **The Holocaust.** It is a separate event (`holocaust`, 5.9M Jewish victims; USHMM country table, Hilberg 5.1M low). Every WWII row excludes Jewish victims: Erlikman's Soviet republic totals and Poland's IPN total were reduced accordingly. Roma, Soviet POWs, Polish civilians and disabled victims stay in the WWII rows. Polish citizens who died in the east are partly inside Erlikman's Belarus and Ukraine figures, so the Poland row uses about 2M ethnic Poles rather than the full IPN non-Jewish total of about 2.9M.
- **War-caused famines.** Bengal 1943, Henan 1942-43 and Vietnam 1944-45 are separate famine rows, and the WWII and Sino-Japanese rows exclude them. The Greek famine, the Java famine and the Leningrad siege sit inside the WWII rows. The Sino-Japanese war is about 15M in total: 12.9M in the main row, 0.6M for the 1938 Yellow River dike breach and 1.5M for the Henan famine.
- **1918 influenza and WWI.** WWI rows exclude influenza: the US row nets out roughly 45k soldiers who died of flu. The Iranian 1917-19 crisis (~2M mainstream) is split into 1M famine plus 1M flu. Russian flu (0.45M) and Civil War typhus (2-3M, inside the Civil War row) may overlap. The Mexican Revolution row may contain some flu deaths.
- **Russian Civil War and the 1921-22 famine.** These are separate rows that together come to about 9M, matching White's 9M.
- **Cholera and famine in India.** Much famine mortality was cholera. The two India cholera rows are therefore set to about two-thirds of the cited cumulative totals (15M for 1817-60, 23M for 1865-1917). Even so, these rows are the softest large numbers in the file, because cholera became endemic. Consider dropping them or down-weighting them visually.
- **Other overlaps handled in the rows:** the Congo Free State includes the Congo-basin sleeping sickness epidemic, the Algerian conquest row excludes the 1866-68 famine, and the first phase of the Chinese Civil War excludes the 1928-30 famine and the 1931 floods (White's 5M includes them). Soviet WWII rows include Gulag deaths for 1941-45, and the 1930s repression row stops at 1940. White's 4.6M for Aurangzeb's Deccan wars is split into the war plus the 1702-04 famine.
- **Rows that straddle 1950.** The First Indochina War has a 1946-49 row (0.15M) in the pre-1950 file and the whole war (0.5M) in the post-1950 file. China's land reform starts in 1949 and La Violencia in 1948.

## Most debated entries
- **Plague of Justinian.** Traditional estimates run 15-100M. Mordechai, Eisenberg et al. (PNAS 2019) argue the plague was nearly inconsequential. Central 15M.
- **Plague of Cyprian.** No scholarly total exists; the central value is an order-of-magnitude placeholder. The Antonine Plague is also debated (Duncan-Jones vs Bruun), with most estimates 5-10M.
- **An Lushan, Three Kingdoms, Xin, Mongol China, Ming-Qing.** All rest on census drops (see above).
- **Timur.** The popular 15-20M ("5% of the world") traces to chronicle claims. Central 9.5M.
- **Black Death.** Benedictow's 60% (about 50M in Europe) is the high end; the central value is 30M for Europe plus 8M for the steppe, Near East and North Africa. There is no defensible China component: Sussman 2011 doubts plague reached China at all.
- **Congo Free State.** Hochschild/Vansina give 10M by extrapolation, White 8M, Sanderson's demographic reconstruction 1-5M. Central 5M.
- **Mahdist Sudan and Menelik II's conquests.** The 4-6M figures rest on colonial population claims (Wingate, de Salviac). Centrals are 3M and 1M.
- **Bengal 1770.** Traditional 10M, Datta about 2M; Dyson says the toll is overstated. Central 3M. **Chalisa 1783 and Doji bara 1789-92:** the "11M" for each is untraceable or comes from a single study. Centrals 3M and 4M.
- **Dungan revolt.** Commonly 8-12M; Cao Shuji's 20.8M population loss includes migration. Central 8M.
- **Smaller but disputed figures.** The 1931 China floods (0.42M survey vs 2-4M), 1887 Yellow River (0.9-2M), Henan 1942 (Garnaut under 1M vs 3M), Persia 1917-19 (2M mainstream vs Majd's 8-10M), and the 1918 flu in China (1-9.5M) and Indonesia (1.5-4.4M).
- **1918 influenza overall.** My regional central values sum to about 25M, against a published range of 17.4M (Spreeuwenberg) to about 50M (Johnson & Mueller). I could not access Johnson & Mueller's regional table, so the regional rows are built from country studies.
- **Post-1950.** The most contested are DR Congo (IRC 5.4M vs the Human Security Report), Bangladesh 1971 (0.3-3M), Bangladesh 1974 (27k-1.5M), North Korea's famine, Tigray, and China's Covid toll (Economist 2.2M, 95% interval 0.5-6.4M).

## Missing because no defensible estimate exists (or the figure is not credible)
- **Not added: no defensible regional split.** The 1889-90 influenza (~1M worldwide) and encephalitis lethargica (~0.5M) have no published regional breakdown, so they were not added.
- **Not added: no defensible estimate.**
  - The trans-Saharan, Red Sea and Indian Ocean slave trades: White gives about 19M deaths at a ratio of 3 deaths per 2 delivered, a guess.
  - Seasoning deaths of enslaved Africans in the Americas.
  - Madagascar's Merina wars and 1947 revolt.
  - The French Congo concession regime.
  - Indigenous decline in Brazil after 1650 and in the 20th century (White: 0.8M).
  - Sui-Tang and Jin/Five Barbarians dynastic collapses.
  - Islamic conquests.
  - Sengoku Japan.
- **Excluded as not credible or as cumulative aggregates.** These appear in popular lists:
  - China famines of 1810/1811/1846/1849 (45M, 20M...).
  - The 1907 China famine at 20-25M (kept at 0.6M instead).
  - The 1333-37 China famine (6M).
  - The Reconquista (7M), the Fang La rebellion (2M), Akbar's and Khalji's conquests, and the Bahmani-Vijayanagara wars.
  - The 1772-73 "Persian plague" (2M), the 1201 Mediterranean earthquake (1.1M) and the 1737 Calcutta and 1881 Haiphong cyclone tolls.
- **Below about 300k, not listed.** Haitian Revolution, Java War, Aceh War, Herero/Nama, Maji Maji, Franco-Prussian War combat deaths, Russo-Turkish wars, Turkish War of Independence (partly inside the genocide rows), Greek Civil War, Tenpo famine (0.2-0.3M), Finnish famine 1866-68, 1876 Bengal cyclone, 1920 Haiyuan and 1923 Kanto earthquakes.
- **Kept for regional coverage although below the threshold, and flagged in their notes:** Hawaii/Polynesia, Aboriginal Australia, Maori, Uganda sleeping sickness, Irish famine of 1740, Cuban wars, Paraguayan War and Russian famine of 1891.

## Post-1950 file specifics
- **COVID-19.** Rows cover 42 countries with at least 95k excess deaths, about 22.3M of the 25.6M global total. The central value is The Economist's model to the end of 2023, downloaded from their GitHub. Low/high are the model's 95% interval, with the low raised to the WHO 2020-21 lower bound where that is higher. Each row's notes also give WHO's 2020-21 estimate (Msemburi et al. 2023; WHO world total 14.9M).
- **HIV/AIDS.** Rows are the sum of WHO Global Health Estimates 2000-2021, taken from OWID because the IHME series cannot be downloaded. Two parts are my approximations: 1990s deaths, taken as a multiple of year-2000 deaths (larger for early epidemics such as Uganda), and 2022-23. Read the pre-2000 part as order-of-magnitude. India's WHO figures are higher than India's national (NACO) estimates.
- **Global influenza rows.** These use `iso3 = WLD`, which is not a country code: the 1957, 1968 and 2009 pandemics have no published country split beyond the USA.
- **Russia-Ukraine.** Rows cover 2022-2023 only.

## Reproducibility
The download (Koch PDF, OWID WHO HIV and excess-death CSVs, Economist CSV) and the build scripts (`build_pre1950.py`, `build_post1950.py`) are in the parent scratchpad folder, outside this directory.
