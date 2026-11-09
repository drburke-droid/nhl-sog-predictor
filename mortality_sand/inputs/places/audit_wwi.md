# WWI battle and front deaths: audit (`wwi` rows of `inputs/places/wars.csv`)

Scope: the 18 battle/front rows named in the brief. The epidemic and famine rows were not audited, but they
are counted in the parent-row check. Output: `audit_wwi.csv` (same header as the places CSVs plus a first
`old_place` column). Every row is revised; no rows were deleted or added. Coordinates, radii and month windows
are unchanged. All numbers are **deaths only**, built from components by side; wounded and surviving
prisoners are never counted.

## Method

**Western Front backbone (8 rows).** The Reichsarchiv table *Die Verluste an der Westfront nach
Hauptkampfzeiten* (Statistisches Jahrbuch fuer das Deutsche Reich 1921/1923/1924; scans fetched and read) gives,
for each fighting period from Aug 1914 to Nov 1918:
- German losses facing the French and the British fronts, split into killed, missing/captured and wounded;
- French killed+missing+captured;
- British dead and missing/captured.

All the Western Front rows are cut from this one table, so the year rows and the named battles use the same
counting and cannot overlap.
- **German:** the table's "Tot" counts only men killed on the field (footnote 3), not those who died later of
  wounds or men missing and dead. Totals: 789,400 field-killed, against about 1.5M German dead in the West
  (Western Front infobox 1,495,000; Sanitaetsbericht East 317,118 of 2.04M total). Each period's German deaths
  = field-killed x 1.3 (died of wounds and late reports; Sanitaetsbericht ratio about 0.37 died of wounds per
  killed) + missing not captured (missing less the Germans the Allies reported capturing). Applied to the whole
  war this gives about 1.3M, so it errs low, not high.
- **French:** the annual recorded dead (Huber 1931 via Prost: 1914 301k, 1915 349k, 1916 252k, 1917 164k,
  1918 235k) are split by period using the table's killed+missing+captured, less prisoners.
  - For battles, official battle figures are used: AFGG Verdun 61k killed / 101k missing / 216k wounded;
    Denizot for the Somme; Doughty and Pierrefeu for the Nivelle offensive.
- **British:** the table's British "dead" already include died of wounds and of disease. Missing presumed dead
  = missing less British POWs. The battle's share of British losses in France comes from the official
  casualty totals (e.g. Edmonds: 419,654 of 498,054 for the Somme).

**Eastern Front (5 rows).**
- Where official killed/wounded/missing and prisoner counts exist, deaths = killed + missing not taken
  prisoner + died of wounds. The prisoner counts are those announced by the capturing side. This applies to
  Gorlice-Tarnow and the Great Retreat (Wikipedia, after Nelipovich), the Sanitaetsbericht figures for German
  armies, and Der Weltkrieg for Brusilov.
- Otherwise deaths come from casualty totals less prisoners, at the killed:wounded ratios seen in those
  operations (low confidence).
- Russian and Austro-Hungarian prisoners are excluded. That includes those who later died in camps far away:
  Przemysl's 117k, Novogeorgievsk, the ~0.5M Russian POWs of summer 1915 and the >327k Austro-Hungarian POWs
  of the Brusilov offensive.
- Austro-Hungarian POWs who died on the march through Albania are counted at the Serbian retreat, where they
  died.

**Home-country splits (modern ISO3).**
- **German army:** 78% DEU, 18.5% POL (Posen, West Prussia, Silesia, Farther Pomerania, southern East
  Prussia), 3% FRA (Alsace-Lorraine), DNK, LTU. Kaliningrad (~1.7%) is folded into POL/DEU, because the film
  draws RUS grains from central Russia.
- **Imperial Russian army:** the conscript-liable population of 1897 by modern borders: RUS 0.54, UKR 0.23,
  POL 0.07, BLR 0.06, Baltic states, MDA, GEO, ARM, AZE, KAZ.
- **Austro-Hungarian army:** 1910 population by modern borders, weighted by the higher death rates of
  German-Austrians and Magyars. AUT 0.16, CZE 0.19, HUN 0.18, ROU 0.10, UKR 0.075, HRV 0.07, POL 0.06,
  SVK 0.05, BIH, SRB (Vojvodina), SVN, ITA (Trentino/Trieste). At the Isonzo, South Slavs are weighted up.
- **British Empire:** GBR 0.96 / IRL 0.04, with Dominion and Indian dead named separately.
- **French army:** North African (DZA/MAR/TUN) and West African (SEN/MLI/BFA/GIN/CIV) shares vary by year:
  West Africans about 0.6% in 1914-15, 10% in the Nivelle offensive.
- **Left out:** South Africans, and Malawian, Rhodesian, Mozambican and Congolese carriers, because KHS and
  CAH have no `wwi` row. Canadians are now included before 1917: `places._pick_row` falls back to the NAM row
  regardless of its 1917 start.

**Double-counting checks.**
- **Western Front:** the year rows (1914, 1915, 1918 spring, 1918 autumn) and the named battles (Verdun,
  Somme, Arras/Chemin des Dames, Passchendaele) cover different months or different sectors of the same
  period table.
- **1916:** French Verdun (2nd Army) and Somme (6th/10th Armies) figures are separate. German losses facing
  the French are split between the two by each battle's share of casualties.
- **Gallipoli:** French Dardanelles dead are taken out of the 1915 French total.
- **East Prussia and Galicia:** Austro-Hungarian dead at the Vistula and Lodz are in the East Prussia/Poland
  row; Galicia and the Carpathians are in their own row. Rauchensteiner's 189k Austro-Hungarian fallen by the
  end of 1914 covers both, plus Serbia.
- **1915 Russian front:** losses of 1-13 July, which Wikipedia includes in the Gorlice-Tarnow period, are moved
  to the Great Retreat row (Gorlice x0.85).
- **Influenza:** French, British and US influenza deaths of autumn 1918 are excluded, because `flu_1918` holds
  them.

**Ranges.** A row's low and high values sit halfway between the root-sum-square of the component ranges and
their straight sum, because the errors are partly shared: the same multiplier or the same national series.

## Rows

| Place | Old central (low-high) | New central (low-high) | Components (central, low-high) | Why it changed | Sources |
|---|---|---|---|---|---|
| Western Front 1914 | 470k (380k-560k) | **560k** (515k-615k) +19% | French army: 300k (290k-330k)<br>German army: 225k (190k-255k)<br>British Empire (BEF incl. Indian Corps): 24k (20k-30k)<br>Belgian army: 10k (7.0k-14k) | German dead were under-counted (145k): the Reichsarchiv's 116,750 'killed' Aug-Nov are field reports that leave out died of wounds and the missing who were dead; rebuilt at ~225k. French 300k kept (Huber/Prost). | [Prost, 'War losses (France)', 1914-1918-Online, after Huber 1931: French recorded dead 1914 301k, 1915 349k](https://encyclopedia.1914-1918-online.net/article/war_losses_france); [Wikipedia, Battle of the Frontiers / First Battle of the Marne](https://en.wikipedia.org/wiki/Battle_of_the_Frontiers); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf); [Wikipedia, Western Front (WWI) infobox: 1,495,000 German military dead](https://en.wikipedia.org/wiki/Western_Front_(World_War_I)); [Wikipedia, World War I casualties](https://en.wikipedia.org/wiki/World_War_I_casualties); [Wikipedia, Battle of the Yser (Belgians 3,500 killed)](https://en.wikipedia.org/wiki/Battle_of_the_Yser) |
| Western Front 1915 | 520k (420k-620k) | **640k** (585k-690k) +23% | French army: 330k (310k-350k)<br>German army: 225k (190k-255k)<br>British Empire: 82k (72k-92k)<br>Belgian army: 4.0k (2.0k-6.0k) | Same German correction (112k to 225k, from the Reichsarchiv period table); British raised to 82k from the British period totals incl. missing presumed dead. | [Prost, 'War losses (France)', 1914-1918-Online, after Huber 1931: French recorded dead 1914 301k, 1915 349k](https://encyclopedia.1914-1918-online.net/article/war_losses_france); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf); [Wikipedia, Western Front (WWI) infobox: 1,495,000 German military dead](https://en.wikipedia.org/wiki/Western_Front_(World_War_I)) |
| Verdun | 305k (260k-380k) | **220k** (190k-295k) -28% | French army: 125k (105k-163k)<br>German army: 95k (80k-143k) | Old row used 163k French + 143k German 'dead', but both are killed + missing incl. prisoners. Deaths: French 61k killed + died of wounds + missing who were not POWs; German ~100k killed/missing less ~21k POW plus died of wounds. | [fr.wikipedia, Bataille de Verdun, citing AFGG t.IV vol.3 p.509: 61,000 tues, 101,000 disparus, 216,000 blesses](https://fr.wikipedia.org/wiki/Bataille_de_Verdun); [Wikipedia, Battle of Verdun (Doughty 2005: 377,231 French casualties, 16% fatal, 28% missing; Philpott 2014; Wendt 336,831 German)](https://en.wikipedia.org/wiki/Battle_of_Verdun); [Memorial de Verdun / Historial de Peronne, 'Le bilan des batailles de Verdun et de la Somme' 2021](https://memorial-verdun.fr/files/6a80fff1/le_bilan_des_batailles_de_verdun_et_de_la_somme.pdf); [wereldoorlog1418.nl, Verdun casualties](https://wereldoorlog1418.nl/battleverdun/slachtoffers.htm); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf) |
| Somme | 297k (250k-340k) | **285k** (255k-370k) -4% | British Empire: 115k (100k-150k)<br>French army: 62k (50k-75k)<br>German army: 110k (95k-165k) | Small change: British raised (115k vs 87k) from British dead+missing in France x Somme share; German lowered (110k vs 160k: Gilbert's 160k+ incl. missing who were captured); French 62k. | [Wikipedia, Battle of the Somme](https://en.wikipedia.org/wiki/Battle_of_the_Somme); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf); [Memorial de Verdun / Historial de Peronne, 'Le bilan des batailles de Verdun et de la Somme' 2021](https://memorial-verdun.fr/files/6a80fff1/le_bilan_des_batailles_de_verdun_et_de_la_somme.pdf); [CWGC, 'The Somme in 2026' (more than 150,000 Commonwealth dead Jul-Nov 1916)](https://admin.cwgc.org/our-work/blog/the-somme-in-2026-why-what-happened-in-1916-still-matters/); [White, necrometrics.com/battles.htm](http://necrometrics.com/battles.htm) |
| Arras & Chemin des Dames | 140k (110k-170k) | **150k** (130k-180k) +7% | British Empire (Arras, 9 Apr-17 May): 43k (37k-50k)<br>French army (Nivelle offensive): 45k (35k-55k)<br>German army: 63k (50k-80k)<br>Russian Expeditionary Force: 1.0k (0.5k-1.5k) | Rebuilt by side from the period table; French Nivelle offensive 45k (vs 36k) and German 63k. | [Wikipedia, Battle of Arras (1917)](https://en.wikipedia.org/wiki/Battle_of_Arras_(1917)); [Wikipedia, Nivelle offensive / Second Battle of the Aisne](https://en.wikipedia.org/wiki/Nivelle_offensive); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf) |
| Passchendaele | 150k (110k-200k) | **120k** (105k-140k) -20% | British Empire: 70k (60k-80k)<br>French First Army: 2.5k (2.0k-3.5k)<br>German army: 47k (40k-60k) | German lowered (47k vs ~66k): Reichsarchiv field-killed vs British in Aug-Dec 1917 only supports ~45-50k; British 70k kept. | [Wikipedia, Battle of Passchendaele](https://en.wikipedia.org/wiki/Battle_of_Passchendaele); [White, necrometrics.com/battles.htm](http://necrometrics.com/battles.htm); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf) |
| Western Front 1918: German offensives | 280k (220k-380k) | **385k** (340k-440k) +38% | German army: 210k (180k-250k)<br>French army: 95k (80k-110k)<br>British Empire: 65k (55k-77k)<br>US Army/Marines: 10k (8.0k-12k)<br>Italian II Corps: 3.0k (2.0k-4.4k)<br>Belgian and Portuguese: 1.5k (1.0k-2.5k) | German dead raised to 210k (688k casualties Mar-Jun incl. 114k field-killed); British 65k and French 95k from the period table; US, Italian, Portuguese added. | [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf); [Wikipedia, German spring offensive / Operation Michael](https://en.wikipedia.org/wiki/Operation_Michael); [Wikipedia, Second Battle of the Marne](https://en.wikipedia.org/wiki/Second_Battle_of_the_Marne) |
| Western Front 1918: Hundred Days | 340k (270k-420k) | **360k** (315k-415k) +6% | German army: 140k (115k-170k)<br>French army: 95k (80k-115k)<br>British Empire: 80k (70k-92k)<br>US Army: 41k (38k-45k)<br>Belgian army: 4.0k (2.0k-6.0k) | Little change; influenza deaths taken out of French/British; German 140k vs Tucker's 100k+ killed. | [Wikipedia, Hundred Days Offensive](https://en.wikipedia.org/wiki/Hundred_Days_Offensive); [Reichsarchiv, 'Die Verluste an der Westfront nach Hauptkampfzeiten', Statistisches Jahrbuch fuer das Deutsche Reich 1924, p.26](https://digi.bib.uni-mannheim.de/fileadmin/statjahrb/514401303_1924/pdf/514401303_0075.pdf); [Wikipedia, Meuse-Argonne offensive](https://en.wikipedia.org/wiki/Meuse%E2%80%93Argonne_offensive) |
| Tannenberg & East Prussia/Poland 1914 | 130k (80k-180k) | **135k** (100k-190k) +4% | Russian army: 95k (70k-140k)<br>German army: 30k (20k-45k)<br>Austro-Hungarian 1st Army: 10k (6.0k-15k) | German share cut (Sanitaetsbericht shows only a few thousand German killed at Tannenberg/Masurian Lakes); Russian kept; A-H at the Vistula added. | [Wikipedia, Battle of Tannenberg / First Battle of the Masurian Lakes](https://en.wikipedia.org/wiki/Battle_of_Tannenberg); [Wikipedia, Battle of Lodz (1914)](https://en.wikipedia.org/wiki/Battle_of_%C5%81%C3%B3d%C5%BA_(1914)); [Wikipedia, Eastern Front (WWI) infobox: German 317,118 dead](https://en.wikipedia.org/wiki/Eastern_Front_(World_War_I)) |
| Galicia & the Carpathians | 390k (260k-550k) | **485k** (375k-615k) +24% | Austro-Hungarian army: 240k (190k-300k)<br>Russian army: 235k (160k-320k)<br>German army: 12k (8.0k-18k) | A-H 1914 dead per Herwig/Rauchensteiner higher than assumed; Russian Carpathian losses raised. | [Wikipedia, Battle of Galicia (Herwig: A-H 100,000 dead; Buttar: Russian 225,000 incl. 40,000 POW) and Siege of Przemysl](https://en.wikipedia.org/wiki/Battle_of_Galicia); [Rauchensteiner 2014, The First World War and the End of the Habsburg Monarchy, p.281](https://austria-forum.org/web-books/firstworldwar01en2014isds/000283); ru.wikipedia, Karpatskaya operatsiya (Russian ~1M, CP ~0.8M casualties incl. sick); Nelipovich 2022 (German 5,687 killed in the Carpathians) |
| Gorlice-Tarnow | 100k (60k-160k) | **235k** (175k-315k) +135% | Russian army: 140k (100k-195k)<br>Austro-Hungarian army: 68k (45k-105k)<br>German army: 26k (20k-32k) | Old 100k assumed '1/4 of killed+wounded'; the official 2 May-13 Jul data give 118k Russian and 68k CP killed in action alone plus large missing-not-captured. | [Wikipedia, Gorlice-Tarnow offensive](https://en.wikipedia.org/wiki/Gorlice%E2%80%93Tarn%C3%B3w_offensive) |
| Great Retreat 1915 | 250k (150k-400k) | **325k** (250k-415k) +30% | Russian army: 210k (150k-280k)<br>German army: 55k (45k-70k)<br>Austro-Hungarian army: 60k (40k-80k) | Russian 96,820 KIA in 13 Jul-28 Aug alone; German 67,290 killed and died; A-H incl. Rovno. | [Wikipedia, Great Retreat (Russia)](https://en.wikipedia.org/wiki/Great_Retreat_(Russia)); ru.wikipedia, Velikoye otstupleniye 1915 (German 67,290 killed and died; German East 1915: 92,131 killed, 33,634 missing); [Wikipedia, Gorlice-Tarnow offensive](https://en.wikipedia.org/wiki/Gorlice%E2%80%93Tarn%C3%B3w_offensive) |
| Brusilov Offensive | 250k (150k-400k) | **315k** (230k-440k) +26% | Russian army: 200k (140k-300k)<br>Austro-Hungarian army: 75k (50k-110k)<br>German army: 35k (25k-50k)<br>Ottoman XV Corps: 3.0k (2.0k-4.0k) | Russian losses revised upward (SW Front 1.45-1.65M by Oct per Nelipovich); A-H/German built from Der Weltkrieg losses less POWs. | [Wikipedia, Brusilov offensive (Russian archival 62,155 killed, 376,910 wounded, 59,802 missing to mid-July)](https://en.wikipedia.org/wiki/Brusilov_offensive); ru.wikipedia, Brusilovskiy proryv (Der Weltkrieg Bd.10 p.566: A-H 616k incl. >327k POW, German 148k incl. ~20k POW; Nelipovich 2006: SW Front 1,446,334 vs Austro-German 845,956); [Nelipovich 2026, Russkiy front Pervoy mirovoy voyny: poteri storon 1916](https://iriran.ru/sites/default/files/2026-08/%D0%9D%D0%B5%D0%BB%D0%B8%D0%BF%D0%BE%D0%B2%D0%B8%D1%87.pdf) |
| Isonzo (12 battles) & Caporetto | 260k (200k-350k) | **245k** (200k-325k) -6% | Italian army: 160k (130k-220k)<br>Austro-Hungarian army: 85k (60k-110k)<br>German 14th Army: 2.0k (1.0k-3.0k) | Kept near Scole's 228k battle deaths plus attrition; Italian/A-H split ~2:1. | [Wikipedia, Battles of the Isonzo, citing Scole, 'War losses](https://en.wikipedia.org/wiki/Battles_of_the_Isonzo); [Wikipedia, Battle of Caporetto](https://en.wikipedia.org/wiki/Battle_of_Caporetto) |
| Gallipoli | 130k (100k-140k) | **120k** (110k-145k) -8% | Ottoman army: 65k (57k-87k)<br>UK (incl. Royal Naval Division): 34k (32k-36k)<br>France: 9.8k (9.0k-10k)<br>Australia: 8.7k (8.7k-8.8k)<br>New Zealand: 2.8k (2.7k-2.8k)<br>India: 1.4k (1.3k-1.4k) | Ottoman central moved to Erickson's archival 56,643 (+ some missing) instead of 86,692; Allied dead from the per-country table. | [Wikipedia, Gallipoli campaign (Erickson 2001 from Ottoman archives: 56,643 died; Allied dead by country)](https://en.wikipedia.org/wiki/Gallipoli_campaign); [White, necrometrics.com/battles.htm](http://necrometrics.com/battles.htm) |
| Sarikamish | 65k (40k-90k) | **56k** (35k-82k) -14% | Ottoman Third Army: 50k (30k-75k)<br>Russian Caucasus Army: 6.0k (4.0k-8.0k) | Ottoman 50k from Cakmak's breakdown and Hafiz Hakki (60k was near the top). | [Wikipedia, Battle of Sarikamish](https://en.wikipedia.org/wiki/Battle_of_Sarikamish) |
| Serbian retreat through Albania | 120k (60k-280k) | **115k** (75k-265k) -4% | Serbian soldiers: 45k (25k-77k)<br>Serbian civilians (refugees): 40k (20k-160k)<br>Austro-Hungarian POWs: 30k (15k-47k)<br>Montenegrin army: 2.0k (1.0k-4.0k) | Recomposed: A-H POW deaths larger (30k vs 14k), Serbian soldiers smaller (the 77k official covers the whole 1915 campaign). | [Wikipedia, Great Retreat (Serbia)](https://en.wikipedia.org/wiki/Great_Retreat_(Serbia)) |
| German East Africa | 250k (100k-400k) | **260k** (180k-420k) +4% | GEA civilians (famine, Mtunya 1917-18): 150k (100k-300k)<br>German-side askari and porters: 15k (10k-25k)<br>British-side carriers: 83k (45k-95k)<br>British Empire soldiers: 7.5k (6.0k-9.0k)<br>Belgian-side porters from Ruanda-Urundi: 3.0k (2.0k-6.0k) | Recomposed by group; total little changed. | [Wikipedia, East African campaign](https://en.wikipedia.org/wiki/East_African_campaign_(World_War_I)) |

## Biggest changes

| Place | Old | New | Change |
|---|---|---|---|
| Gorlice-Tarnow | 100k | 235k | +135% |
| Western Front 1918: German offensives | 280k | 385k | +38% |
| Great Retreat 1915 | 250k | 325k | +30% |
| Verdun | 305k | 220k | -28% |
| Brusilov Offensive | 250k | 315k | +26% |
| Galicia & the Carpathians | 390k | 485k | +24% |
| Western Front 1915 | 520k | 640k | +23% |
| Passchendaele | 150k | 120k | -20% |
| Western Front 1914 | 470k | 560k | +19% |

- **Gorlice-Tarnow.** Official losses for 2 May-13 Jul 1915 show 118k Russians and 68k soldiers of the Central
  Powers killed in action. On top of that, about 170k Russian and 140k Central Powers missing were never
  reported as prisoners. The old row was built as "1/4 of killed+wounded".
- **Western Front 1918, German offensives.** German dead rise from about 155k to 210k: there were 688k German
  casualties in March-June alone, 114k of them killed in the field.
- **Great Retreat 1915.** 96,820 Russian killed in action in 13 Jul-28 Aug alone; Germans 67,290 killed and
  died.
- **Verdun.** The quoted 162k French and 143k German "dead" are both killed + missing including prisoners.
  Deaths only: 125k French + 95k German.
- **Brusilov.** Russian losses are revised upward (Nelipovich: the Southwestern Front lost 1.45-1.65M,
  including sick, by October).
- **Galicia & the Carpathians.** Rauchensteiner gives 189k Austro-Hungarian fallen by the end of 1914, and the
  Russian Carpathian losses are larger than assumed.
- **Western Front 1914 and 1915.** The German dead were under-counted by roughly half: the old rows used
  ~145k (1914) and ~112k (1915), against ~225k each from the Reichsarchiv period table.
- **Passchendaele.** The German share falls (66k to 47k).

Seven rows changed by more than 20%. Passchendaele changed by exactly 20% and Western Front 1914 by 19%.

## Popular figures and why they differ
- **Verdun "300,000 dead" / "700,000".** The 163k French and 143k German (Memorial de Verdun) are "morts et
  disparus" and include ~50k French and ~21k German prisoners. 700k+ includes the wounded.
- **Somme "over a million".** Casualties including 616k wounded and ~80k prisoners. The CWGC's ">150,000
  Commonwealth dead" is above the British period totals, so it is kept as the high end.
- **Passchendaele "500-600k".** Casualties. Deaths were about 120k.
- **Hundred Days.** "1.07M Allied casualties" and "785k German casualties" include the wounded and the 386k
  German prisoners.
- **Gallipoli "~87k Turks".** Waite's 1919 figure. Erickson's archival 56,643 dead of all causes is the
  better-founded base.
- **Serbian Golgotha "240k+".** The 1919 official 77,455 soldiers dead + 77,278 missing cover the whole 1915
  campaign. Hart's 160k civilians is uncorroborated.
- **Eastern Front.** Russian "losses" of 1M+ per operation include the sick and very large numbers of
  prisoners. Most Austro-Hungarian "missing" were prisoners, stragglers or deserters.

## Parent-row check
All 23 `wwi` places are included, using the audited values where revised. Each figure is the places' total
against the row's 95% cap:

| Row | Places | 95% cap | Share of cap |
|---|---|---|---|
| EUR | 4.39M | 8.55M | 51% |
| STP | 0.79M | 2.47M | 32% |
| LEV | 0.27M | 1.52M | 18% |
| EAF | 255k | 333k | 77% |
| NAM | 83k | 124k | 67% |
| SIB | 60k | 114k | 52% |
| AUS | 40k | 59k | 67% |
| **NAF** | **31.4k** | **33.2k** | **94.5%** |
| IRN | 28k | 95k | 30% |
| WAF | 18k | 28.5k | 63% |
| SAS | 12k | 70k | 17% |
| NZL | 9k | 17k | 54% |

**No overflow.**
- **NAF** is close to its cap. The row (35k) is right: Huber gives 35,900 North African dead. Any further
  North African place would overflow it.
- **WAF:** the old French template would also have pushed West Africans too high. 1914-15 shares are now small.
- **Missing parent rows:**
  - South African, Malawian, Rhodesian, Mozambican and Congolese military and carrier deaths have no `wwi`
    row (KHS and CAH). About 1k South Africans at the Somme, ~3.5k South Africans and Rhodesians in East
    Africa, ~12k southern-African carriers and ~15k Congolese porters are left out of the places.
  - The catalogue may lack a KHS/CAH `wwi` row of roughly 30-60k.

## Verification notes (what could not be checked)
- **1914-1918-Online is bot-blocked**, so these come from search snippets and Wikipedia citations, not the
  pages themselves:
  - Prost's French dead by year (1914 301k and 1915 349k were confirmed; 1916-18 are the existing values);
  - Scole's Isonzo deaths;
  - Overmans.
- **Unverified items, kept with low confidence:**
  - the Austrian official history's ~66k dead in the Carpathians, January-April 1915;
  - French prisoners at Verdun (~50k; a forum citation gives ~70k);
  - Dominion sub-totals (AUS/CAN/NZ per battle);
  - US battle deaths before August 1918.
- The German x1.3 + missing-not-captured method is a calibration, not a published series. The Sanitaetsbericht
  monthly Westheer tables (vol. III, 1934) would let it be replaced by direct counts; they were not reachable
  online.
- **Eastern rows are low confidence throughout.** Nelipovich's per-operation volumes (1914, 1915, 2022;
  1916, 2026) are the best current scholarship, and only their headline figures were reachable.
  - 1916: all armies on the Russian front had 926k dead.
  - Brusilov: 1,446,334 Russian vs 845,956 Austro-German losses.
  - These support the upward revisions here.
