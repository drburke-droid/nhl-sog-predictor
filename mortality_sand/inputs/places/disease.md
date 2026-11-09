# Places: epidemics, famines and disasters (pre-1950)

`places_disease.csv` has **159 rows**, all `kind = local`, across 34 parent events. Every row stays inside its parent row's region (vessel) and years, and no parent row is over-filled (see the table at the end).

The build script `build_places_disease.py` sits in the same folder. It holds the derivations (Black Death population x rate, Bengal district arithmetic, India flu factor) and the parent-row checks, so a change of central value can be re-run.

## Coverage

**What's covered.** Rows go where there is real place or time detail, and where that detail changes the picture: deaths move to a different place, or into a much narrower window than the parent row's even spread. The film's waves come from the months.

**Big events.**
- **Black Death (30 rows, 95% of 38M placed).** The wave front runs from the Lower Volga and Caffa (1346) to Constantinople, Sicily and Alexandria (1347). It then reaches Italy, southern France, Iberia, Egypt and Syria (1348); England, northern France, Germany and the Low Countries (1348-50); Scandinavia and East-central Europe (1349-51); and Rus' (1352-53).
  - Cities: Florence, Venice, Paris, Cairo, Damascus and Constantinople.
- **Great Dying (25 rows, 67% of 52.5M).**
  - Central Mexico is split into seven waves that add up to 98% of its 17M: Valley smallpox 1520, the siege of Tenochtitlan, provincial smallpox 1520-21, the gap 1522-44, cocoliztli 1545-48, the gap 1549-75, and cocoliztli 1576-80.
  - The Andes (92%), the Caribbean (93%) and Central America (88%) are split into dated waves.
  - North America has the five dated episodes asked for: New England 1616-19 and 1633-40, the Mississippian Southeast, Plains 1780-82 and Upper Missouri 1837.
  - Amazonia and Colombia/Venezuela get one dated row each. The rest of those rows is undated, so it stays in the parent.
- **1918 influenza (25 rows, 60% of 25.1M).**
  - India by province, with wave months.
  - Japan (two waves), Korea, a timing-only row for China, Java, Iran, Western Samoa and South Africa.
  - Three US regions, five European countries or pairs, Brazil and Mexico.

**India, China and the USSR.**
- **Indian famines (20 rows).**
  - 1876-78 by presidency and phase: Madras in two phases, Bombay, Mysore, Hyderabad, NWP.
  - 1896-97 and 1899-1900: the worst provinces.
  - Bengal 1770, Doji bara (Northern Circars) and Orissa 1866 (monsoon months).
  - Bengal 1943: six districts plus Calcutta.
- **Third plague (7 rows).** India: Bombay, Punjab in two periods, UP. China: Yunnan (origin), Canton 1894, Manchuria 1910-11.
- **Chinese famines and disasters (12 rows).** North China 1876-79 (the four provinces with their peak windows, Shanxi split south/centre), NW China 1928-30, Henan 1942-43, the 1931 floods, the 1887 Yellow River breach and the 1556 earthquake.
- **Soviet famines (17 rows).**
  - 1933 Ukraine: the seven oblasts of the time, plus the Moldavian ASSR.
  - 1933 Russia: Kuban, Saratov with the Volga Germans, Don/Stavropol, Central Black Earth.
  - The Kazakh steppe; 1921-22 Volga (three rows) and southern Ukraine; 1946-47.

**Plagues of antiquity (4 rows).** Antonine plague in Egypt (papyri), the Cyprian plague in Alexandria (Dionysius's grain-dole register), and the first Justinianic wave at Pelusium/Alexandria (541) and Constantinople (542). All are low confidence; none rests on a count.

**Others (19 rows).** Ireland 1846-50 (Connacht and Munster, before and after 1848), Persia 1917-18, France 1693-94, Italy 1629-31 (Milan, Venice, the rest of the north), Naples 1656, Seville 1649, Moscow 1601-03, Kyushu 735, Tonkin 1944-45, cholera in Europe (France 1832, Spain 1854-55, Spain 1885) and cholera in Russia (1848, 1892).

**Deliberately not given rows.** Either nothing defensible exists below the parent row, or the parent row already sits on the right place and years, so a row would add nothing:
- the Cyprian plague outside Alexandria, the Antonine plague outside Egypt, and the Justinianic recurrences 543-750;
- Kanki, Tenmei, the two Korean famines, Chalisa, Agra, Upper Doab, Rajputana, the Deccan 1630 and 1702 famines, and the Great Famine of 1315-17;
- France 1709, Spain 1596, Algeria 1866, the Brazilian Grande Seca, Ethiopia 1888-92, Persia 1870, the Uganda sleeping sickness, Hawaii, smallpox 1870-75, both India cholera rows and the Vietnam cholera of 1849;
- China 1906 and 1920-21, the Irish famine of 1740 and the Russian famine of 1891.

The two India cholera rows have no regional or year data that would not double-count the famine rows (see the catalogue's NOTES).

**Small iconic rows.** Several rows are below the ~50,000 visibility line but kept because they are wave origins or were asked for:
- Hispaniola smallpox 1518-19 (10k), Upper Missouri 1837 (17k), Western Samoa (8.5k);
- Calcutta 1943 (15k), Constantinople 1347 (35k).

## Main judgment calls

### Black Death: Europe
1. **How Europe's 30M is split.** Each region gets its population c.1347 times a regional mortality rate. The populations follow McEvedy & Jones and Benedictow. The rates follow Benedictow's pattern:
   - England, Italy, France, Iberia and Norway highest (his ~60-65%);
   - the Low Countries and East-central Europe lower.

   All rates are scaled by ~0.7, so the regions add to the catalogue's 30M (98% placed). Benedictow's own total is ~50M. For each regional row, low = 0.6x and high = 1.5x; the high is roughly Benedictow's own level.
2. **Cities sit inside their regions.** London (~35k) is folded into "Southern England", because it falls below the visibility line.

### Black Death: Near East, North Africa and the steppe
The Near East, North Africa and steppe rows are allocations of the catalogue's rows by rough population share. Their dates come from Dols, Benedictow and the chronicles.
- **City tolls are the commonly cited figures:** Cairo 200k, Damascus 50k, Florence 55k, Venice 60k and Paris 50k.
- **Constantinople's 35k is an assumption** of about a third of ~100k people. There is no count.

### Americas
**Population decline stands in for deaths**, as in the parent rows.
- **Central Mexico.** Borah & Cook's 1518-1605 decline is scaled to the catalogue's 17M. Waves were assigned within it:
  - smallpox 1520-21 at 4.5M (Valley 0.5M, rest 4.0M), plus the Tenochtitlan siege at 0.1M;
  - cocoliztli 1545-48 at 6.0M;
  - cocoliztli 1576-80 at 1.4M. This is above B&C's net 1568-80 decline, because births offset it; it is below the contemporary 2-2.5M.
  - Two labelled "between waves" rows (1522-44 and 1549-75) make the shares add up.
- **The Andes.** Cook's Peru decline was split by me into four periods. Assigning 2M to the 1525-28 smallpox is a judgment; there is no count.
- **Siege of Tenochtitlan.** No war row exists for the conquest of Mexico, so the siege (0.1M) is placed here as `local`.

### 1918 influenza
1. **India.** The 1921 Census's estimated influenza deaths by province, from the 7.1M British-India total, were multiplied by 1.3. That lets British provinces (8.7M) plus the princely states (~3M, left in the parent row) match the catalogue's 12M.
   - low = the census figure;
   - high = x1.95, the factor implied by Chandra et al.'s 13.88M.
   - Burma's 137k is excluded because it falls in SEA.
2. **Timing-only rows.** Where there is no sub-national data, a single row puts the autumn-wave share in its months: China 1.0M of 2.5M, Mexico, Brazil and Iran.
3. **US regions are rough.** They are regional excess rates times population. They cover only the September-December 1918 wave (0.45M of 0.72M); 1919-20 stays in the parent.

### India and China famines
- **Bengal 1943.** Maharatna's 1992 thesis gives rural death rates by district for 1940-41 and for July 1943-June 1944 (Tables 6.1, 6.6). Excess = the rise x the 1941 census population. All 26 districts were then scaled together to Dyson & Maharatna's 2.1M (factor ~1.78: under-registration plus months outside the peak year).
  - **Calcutta (15k) is a guess.** The dead were mostly rural destitutes from 24 Parganas and Midnapore. Because they came from within ~100 km, the row is kept `local`, not `converge`.
- **Famines of 1876-78, 1896-97 and 1899-1900.** Rows use Maharatna's provincial death-rate indices and census population changes (Mysore). Hyderabad 1876-78 (0.5M) and NWP 1896-97 are allocations.
- **North China 1876-79.** The four rows reproduce the catalogue's provincial rows exactly; they only add the peak windows. Shanxi is split south/centre following Edgerton-Tarpley's emphasis on the southern prefectures. That 3.3M/2.7M split is mine.

### Soviet famines
- **Ukraine.** Rows are 1933 only, from Wolowyna et al. 2016, Table 3, in 1932 oblast borders. Kharkiv oblast includes today's Poltava and Sumy; Kyiv oblast includes Zhytomyr and Cherkasy. 1932 deaths (250k) stay in the parent row.
- **Russia: the catalogue's row looks low.** Its 1.5M sits below Levchuk et al. 2020, who give **3.32M for RSFSR, 1932-34** (2.89M in 1933, excluding Kazakhstan).
  - Kuban (389.5k) and Saratov with the Volga Germans (359.5k) use Levchuk's 1933 figures directly.
  - Don/Stavropol and Central Black Earth were trimmed to fit the 1.5M.
  - **Consider raising that parent row** to ~2.5-3M. Andreev, Darsky & Kharkova and Kondrashin put Russia's 1933 excess deaths at 2.3-2.4M, also above the row (Davies & Wheatcroft's lower USSR total of 5.5-6.5M is the main dissent).
- **Volga 1921-22.** The Samara, Tatarstan/Bashkiria and Saratov rows are allocations by population and Poliakov's relative mortality. No published provincial death table was found.

### Other famines and epidemics
- **Ireland.** Connacht (0.32M) and Munster (0.35M) are Mokyr's county excess-death rates aggregated and scaled to the catalogue's 1M. About 45% falls in October 1846-1847 and 55% in 1848-50. Ulster and Leinster stay in the parent row.
- **Italy 1629-31.** The Milan, Venice and Naples figures are the standard city tolls. "Lombardy, Veneto and Emilia" (0.6M) is the remainder of Alfani's northern-Italy loss, kept within the 1M parent.

## Notes on the parent rows (for whoever revises the catalogue)
- `soviet_famine_1930` Russia row: 1.5M vs Levchuk et al.'s 3.3M (see above).
- `flu_1918` India: the catalogue's 12M is consistent with census-plus-states. Chandra et al.'s 13.88M is for British districts alone, which would put all India at ~17M (Mills 16.8M).
- `bengal_famine_1943`: Dyson & Maharatna's 2.1M runs into 1945 (0.15M in 1945); the parent row stops at 1944.
- `orissa_famine`: the parent's 1M is Hunter's "a third". Official 1867 estimates were nearer a quarter (~0.75-0.8M); my row uses 0.8M.
- `north_china_famine_1876`: the provincial rows already carry the detail; my rows only add months.

## Missing from the catalogue (in or near my scope)
**Plague.**
- **Pestis secunda (1360-63) and the later 14th-century plague waves.** Probably 10-20% of Europe again, i.e. several million deaths. This is the largest omission.
- **Great Plague of London 1665-66.** ~100k in London; ~0.2M in England.
- **Plague of Genoa 1656-57.** ~55-60k, about half the city. Outside the Kingdom-of-Naples parent.
- **Plague of Marseille and Provence 1720-22.** ~0.1M.
- **Moscow plague 1770-72.** ~50-100k in Moscow, more across Russia.

**Famines.**
- **Indian famine of 1907-08 (United Provinces).** Maharatna's thesis, Table 1.1, gives 2.1-3.2M excess deaths. It overlaps the 1907 plague peak and the 1908 Punjab malaria, but it is a major famine absent from the catalogue.
- **Finnish famine of 1695-97.** About a quarter to a third of Finland, ~0.15M. Estonia and Livonia lost ~0.07M in the same years.
- **Bohemian/Central European famine of 1770-72.** Bohemia ~0.25M.
- **Sichuan famine of 1936-37.** Tolls of 1-5M are claimed; poorly documented.
- **Year without a summer, 1816-17.** Tambora (Sumbawa/Lombok, ~0.07-0.09M direct, 1815) and the European subsistence crisis.
- **Mexican matlazahuatl of 1736-39.** ~0.2M; the Mexico Great Dying row stops at 1600.
- **Below threshold but notable:** the Guntur (Madras) famine of 1832-33 (~0.2M), the Mount Lebanon famine of 1915-18 (~0.1-0.2M; may sit in the WWI rows) and the Ruzagayura famine (Rwanda-Burundi, 1943-44).

## Placed vs parent (central values)

| parent row | rows | placed | parent |
|---|---|---|---|
| black_death EUR | 18 | 29.48M | 30.0M |
| black_death LEV / NAF / STP | 5 / 4 / 3 | 1.59M / 2.60M / 2.50M | 2.0M / 3.0M / 3.0M |
| americas_great_dying central Mexico | 7 | 16.6M | 17.0M |
| americas_great_dying Central America | 4 | 4.4M | 5.0M |
| americas_great_dying Andes | 4 | 12.0M | 13.0M |
| americas_great_dying Caribbean | 3 | 0.56M | 0.6M |
| americas_great_dying N. America / Brazil / Colombia | 5 / 1 / 1 | 1.07M / 0.15M / 0.40M | 3.5M / 9.0M / 3.0M |
| flu_1918 SAS | 7 | 8.70M | 12.0M |
| flu_1918 EUR / NAM / EAS (China; Japan-Korea) | 5 / 3 / 1+3 | 1.58M / 0.45M / 1.0M + 0.52M | 2.2M / 0.72M / 2.5M + 0.55M |
| flu_1918 ISE / IRN / KHS / MES / AMZ / POL | 1 each | 1.4M / 0.8M / 0.3M / 0.3M / 0.12M / 8.5k | 2.0M / 1.0M / 0.5M / 0.35M / 0.25M / 10k |
| soviet_famine_1930 Ukraine / Russia / Kazakhstan | 7 / 4 / 1 | 3.53M / 1.38M / 1.25M | 3.9M / 1.5M / 1.5M |
| russian_famine_1921 Volga / Ukraine | 3 / 1 | 1.85M / 0.70M | 3.5M / 1.0M |
| north_china_famine_1876 (4 provincial rows) | 5 | 11.0M | 11.0M |
| indian_famine_1876 | 6 | 6.3M | 8.2M |
| indian_famine_1896 / 1899 | 2 / 2 | 1.5M / 2.1M | 4.5M / 3.5M |
| bengal_famine_1943 | 7 | 0.93M | 2.1M |
| third_plague_pandemic SAS / EAS | 4 / 3 | 7.32M / 0.53M | 10.0M / 1.5M |
| nw_china_famine_1928 | 2 | 4.0M | 5.0M |
| others | 1-4 each | see `check_out.txt` | |

The full per-row check, regenerated by the script, is in `check_out.txt` next to this file.
