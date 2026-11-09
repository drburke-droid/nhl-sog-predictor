# Audit: WWII in Europe (25 rows of `inputs/places/wwii.csv`, plus the two Stalingrad rows cross-checked)

Files (all in the scratchpad; nothing under `/home/user/nhl-sog-predictor` was edited):
- `audit_wwii_europe.csv`: the 25 scope rows, revised or re-verified, plus both `Stalingrad` rows (unchanged; the battle row's central is now written `950000`, not `950000.0`). Header is the places header with `old_place` first. Both Stalingrad rows carry `old_place` = `Stalingrad`, in file order (battle row first, prisoners row second).
- `euro_build.py`: builds every row from its components and recomputes `from`.
- `euro_merge.py` + `euro_check.py`: merge this audit and the Asia audit into a copy of `wwii.csv` and run the film's own `places.py` row-picking on it, read-only.
- `aud_dl/`: fetched sources (Wikipedia wikitext in en/ru/de, Krivosheev 2001 full text, Kantor 2018 PDF).

## Method
- **Deaths only, built by side.** Every row is a sum of components (each army, plus civilians), each with a central, a low-high range and a source. `from` is recomputed from the components. Row lows and highs are set by judgment, not by adding the component extremes.
- **Soviet soldiers.**
  - Start from Krivosheev's irrecoverable losses per operation and per front. Source: the full 2001 Russian edition, fetched from rus-sky.com.
  - Treat Krivosheev as the floor. In 1941 he leaves out 1,162,600 "unaccounted losses of the first months" from units that never reported (his table 120). His Kiev total (616k) is even lower than the prisoners the Germans claimed there.
  - For 1942-44, weigh the revisions:
    - Isaev's TsAMO totals and Gerasimova for Rzhev.
    - Lopukhovsky for Vyazma.
    - Töppel (+40%) and Sokolov for Kursk.
    - The 2004 MoD reference for Rzhev-Vyazma (+8%).
    - Field conscripts in liberated Ukraine who were not on unit rolls.
  - **Subtract all prisoners.** Soviet POW deaths are already in the six POW-camp rows (2.95M). Prisoner counts use German claims, discounted ~15% for civilians and reservists who were not on unit rolls.
- **German soldiers.**
  - The floor is killed + missing − captured, using Frieser (DRZW 8), the 10-day reports and Müller (DRZW 10/1).
  - Overmans' month-of-death totals run 1.3-2× higher (Wikipedia table, fetched), because the 10-day reports undercounted. The central is set between the two.
  - Germans captured are excluded. Their documented deaths in captivity are in `German POWs in Soviet camps` and the Stalingrad prisoners row.
  - Austrians are 5% of German dead, as before.
- **Civilians:** registered deaths (Leningrad ZAGS), commissions (Dresden, Hamburg police) or the best published estimate, with the popular high noted.
- **Republic split of Soviet dead:** the old row's split is kept for each row. Only the Soviet share as a whole changes.

## Rows
| place | old central (low-high) | new central (low-high) | components (central, low-high) | why it changed | sources |
|---|---|---|---|---|---|
| Leningrad: hunger winter | 450k (400k-650k) | **500k** (430k-600k), +11% | Civilians who died in the city Dec 1941-Apr 1942 (registered 446,388 + unregistered ~80k - normal deaths ~15k) [LOCAL] 500k (430k-600k) | Up. Registered deaths alone are 446,388 (ZAGS monthly certificate), already at the old central; Kovalchuk's 715-750k total in the city implies ~80k unregistered deaths this winter. | <https://ru.wikipedia.org/wiki/Материнство_и_детство_в_блокадном_Ленинграде> (ZAGS certificate of 4 Feb 1943, monthly deaths 1942); <https://ru.wikipedia.org/wiki/Блокада_Ленинграда> (Dec 1941 52,881 per Kislitsyn; Kovalchuk 715-750k; 632,253 official); <https://hist.spbiiran.ru/wp-content/uploads/2019/11/_4_2018_08_Kantor.pdf> (NKVD: Dec 52,612, Jan 96,751); <https://en.wikipedia.org/wiki/Siege_of_Leningrad>; Barber & Dzeniskevich 2005, Life and Death in Besieged Leningrad |
| Leningrad: spring-summer 1942 | 115k (100k-150k) | **115k** (105k-135k), +0% | Civilians who died in the city May-Aug 1942 (registered 113,772; unregistered ~ offset by normal deaths) [LOCAL] 115k (105k-135k) | Unchanged. Re-verified against the ZAGS monthly certificate (113,772 registered May-Aug). | <https://ru.wikipedia.org/wiki/Материнство_и_детство_в_блокадном_Ленинграде> (ZAGS certificate, monthly deaths 1942); <https://ru.wikipedia.org/wiki/Блокада_Ленинграда>; <https://en.wikipedia.org/wiki/Siege_of_Leningrad> |
| Leningrad and Volkhov fronts | 894k (650k-1150k) | **905k** (700k-1170k), +1% | Soviet soldiers and sailors from outside the city (killed, died of wounds or disease, missing not captured) [USSR] 680k (530k-870k); Leningrad militia (opolchenie) and city recruits [LOCAL] 70k (50k-90k); German 18th Army and attached (Aug 1941-Feb 1944) [DEU/AUT] 150k (110k-200k); Spanish Blue Division (whole service, Volkhov and Leningrad) [ESP] 5.0k (4.5k-5.0k) | Nearly unchanged. Re-derived from Krivosheev's front tables; German raised 140k -> 150k (Overmans scale); Blue Division 4.5k -> 5k. | <https://en.wikipedia.org/wiki/Siege_of_Leningrad> (Glantz 2001 p.179: 1,017,881; App. p.265: 579,985); <http://rus-sky.com/history/library/w/w07.htm> (Krivosheev 2001 front tables: Leningrad Front 467,525; Volkhov Front 298,623); <http://rus-sky.com/history/library/w/w06.htm> (operations: Leningrad def. 214,078; Lyuban 95,064; Sinyavino 40,085; Iskra 33,940; Leningrad-Novgorod 76,686); <https://en.wikipedia.org/wiki/Blue_Division>; <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Bialystok-Minsk and Smolensk | 320k (200k-450k) | **415k** (230k-680k), +30% | Soviet soldiers killed or died (not captured) [USSR] 330k (150k-600k); German Army Group Centre, 22 Jun-30 Sep 1941 [DEU/AUT] 85k (65k-100k) | Up. The old row ignored Krivosheev's own 1,162,600 unrecorded losses of the first months; adding a share of them and subtracting the German prisoner counts gives more killed. German raised to the Overmans scale. | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: 341,073; 486,171; 1,162,600 unaccounted, table 120); <https://en.wikipedia.org/wiki/Battle_of_Białystok–Minsk>; <https://en.wikipedia.org/wiki/Battle_of_Smolensk_(1941)> (German prisoner claims via Glantz 2010); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Kyiv and Uman encirclements | 175k (100k-280k) | **245k** (170k-380k), +40% | Soviet soldiers killed or died (not captured) [USSR] 200k (130k-330k); German 6th, 17th Armies, 1st Panzer Group, 2nd Army and 2nd Panzer Group [DEU/AUT] 45k (35k-55k) | Up. Same reason: the Kiev figure (616k) is below the 665k+103k prisoners claimed, so it is an undercount; German raised from 25k to 45k (10-day reports 27.7k killed+missing for three armies only). | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: Kiev defensive 616,304); <https://en.wikipedia.org/wiki/Battle_of_Kiev_(1941)> (Kirchubel 2013 p.283); <https://en.wikipedia.org/wiki/Battle_of_Uman>; <https://ru.wikipedia.org/wiki/Киевская_стратегическая_оборонительная_операция> (German 10-day reports); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Vyazma-Bryansk encirclements | 265k (160k-400k) | **248k** (185k-355k), -6% | Soviet soldiers killed or died (not captured) [USSR] 230k (170k-330k); German Army Group Centre, October 1941 (pockets only) [DEU/AUT] 18k (14k-25k) | Slightly down. Rebuilt from Lopukhovsky's 780k lost / 527k captured; German pocket losses ~18k. | <https://ru.wikipedia.org/wiki/Вяземская_операция> (Lopukhovsky 2008/2013: ~780k lost, 527k captured); <http://rus-sky.com/history/library/w/w07.htm> (Krivosheev 2001: Bryansk Front by year); <http://rus-sky.com/history/library/w/w06.htm> (Moscow defensive 514,338); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Moscow | 550k (400k-750k) | **585k** (480k-850k), +6% | Soviet soldiers killed or died (not captured) [USSR] 470k (400k-700k); German Army Group Centre, 16 Nov 1941-20 Apr 1942 [DEU/AUT] 115k (80k-150k) | Up. German side moved from ~100k to 115k (Overmans: 217k German dead Dec 1941-Apr 1942); Soviet slightly up (MoD 2004 reference is 8% above Krivosheev). | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: 514,338; 139,586; 272,320); <https://en.wikipedia.org/wiki/Battle_of_Moscow>; <https://ru.wikipedia.org/wiki/Ржевская_битва> (MoD 2004 reference: 835,881 vs 196,939); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Crimea: Sevastopol and Kerch | 180k (120k-260k) | **191k** (140k-280k), +6% | Soviet soldiers and sailors killed or died (not captured) [USSR] 150k (110k-230k); German 11th Army [DEU/AUT] 33k (25k-40k); Romanian Mountain Corps and others [ROU] 8.0k (5.0k-10k) | Slightly up: Axis dead ~41k (25k in the Sevastopol siege alone), Soviet unchanged. | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001 table 142); <https://en.wikipedia.org/wiki/Siege_of_Sevastopol_(1941–1942)>; <https://en.wikipedia.org/wiki/Battle_of_the_Kerch_Peninsula>; <https://en.wikipedia.org/wiki/Crimean_campaign> |
| Kharkiv and the Don steppe | 290k (180k-380k) | **275k** (205k-360k), -5% | Soviet soldiers killed or died (not captured) [USSR] 245k (180k-320k); German Army Group South / A and B, May-Jul 1942 [DEU/AUT] 30k (23k-40k) | Slightly down: prisoners of the June operations (~47,600) also subtracted. | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: 170,958; 370,522); <https://en.wikipedia.org/wiki/Second_Battle_of_Kharkov>; <https://ru.wikipedia.org/wiki/Харьковская_операция_(1942)>; <https://ru.wikipedia.org/wiki/Бои_за_Воронеж_(1942)> (German July losses); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Don front | 155k (120k-185k) | **130k** (110k-180k), -16% | Italian 8th Army: killed in Little Saturn and the January retreat [ITA] 25k (21k-30k); Italian prisoners who died on the marches and in camps by spring 1943 [ITA] 40k (30k-44k); Hungarian 2nd Army: killed, died of wounds, missing Jan-Feb 1943 [HUN] 42k (35k-45k); Hungarian prisoners who died by spring 1943 [HUN] 23k (15k-40k) | Down. The Italian 84,830 counts the captured too (Krinko & Medvedev), and ~20k of the captives survived or died after the window; Hungarian dead ~65k (42k killed/missing + captive deaths). | <https://en.wikipedia.org/wiki/Italian_participation_on_the_Eastern_Front>; Krinko & Medvedev 2018, doi:10.15688/jvolsu4.2018.1.9 (Hungary 96k: 42k dead/missing, 26k captured; Italy 84,830); <https://honvedelem.hu/hirek/we-suffered-the-fourth-greatest-loss.html>; <https://ceeol.com/search/article-detail?id=747975> (Szabo 2019); <https://en.wikipedia.org/wiki/Second_Army_(Hungary)> |
| Caucasus | 270k (180k-350k) | **270k** (200k-360k), +0% | Soviet soldiers killed or died (not captured) [USSR] 220k (170k-300k); German Army Group A [DEU/AUT] 40k (30k-55k); Romanian mountain and cavalry divisions [ROU] 10k (7.0k-15k) | Unchanged after re-verification. | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: 192,791; 69,627); <http://rus-sky.com/history/library/w/w07.htm> (Transcaucasian Front by year); <https://ru.wikipedia.org/wiki/Битва_за_Кавказ_(1942—1943)>; <https://en.wikipedia.org/wiki/Battle_of_the_Caucasus> |
| Rzhev | 270k (160k-400k) | **260k** (195k-385k), -4% | Soviet soldiers killed or died (not captured) [USSR] 210k (160k-320k); German 9th Army and neighbours [DEU/AUT] 50k (35k-65k) | Slightly down. Isaev's TsAMO totals show ~64k Soviet prisoners in Jul-Sep 1942, which the old row did not subtract; Gerasimova's between-offensive losses added back. | <https://ru.wikipedia.org/wiki/Ржевская_битва> (Isaev: 392,554 irrecoverable; 50,000 + 13,700 captured; German losses by operation); <http://actualhistory.ru/isaev-rzhev2>; <https://en.wikipedia.org/wiki/Battles_of_Rzhev> (Gerasimova); <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001 table 142); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Kursk | 290k (230k-350k) | **340k** (270k-500k), +17% | Soviet soldiers killed or died (not captured) [USSR] 280k (225k-420k); German 9th Army, 2nd Panzer Army, 4th Panzer Army, Kempf [DEU/AUT] 60k (46k-80k) | Up. Krivosheev is the floor; Toppel finds the official count ~40% low, so +20% central; German raised from Frieser's 46.5k toward Overmans. | <http://rus-sky.com/history/library/w/w06.htm> (Krivosheev 2001: 70,330; 112,529; 71,611); <https://en.wikipedia.org/wiki/Battle_of_Kursk> (Frieser 2007; Toppel; Sokolov); <https://en.wikipedia.org/wiki/Operation_Kutuzov>; <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Dnieper | 500k (380k-600k) | **545k** (430k-720k), +9% | Soviet soldiers killed or died (not captured) [USSR] 415k (330k-550k); German Army Group South / A [DEU/AUT] 130k (100k-170k) | Up. Field conscripts were under-recorded (+25%); German from 100k to 130k (Forczyk 102k is the floor, Overmans implies ~160k). | <http://rus-sky.com/history/library/w/w08.htm> (Krivosheev 2001: 102,957; 173,201; Kiev defensive 26,443); <http://rus-sky.com/history/library/w/w06.htm> (Lyutezh 24,422; Bukrin 6,498); <https://en.wikipedia.org/wiki/Battle_of_the_Dnieper> (Frieser; Forczyk); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Right-bank Ukraine | 340k (250k-420k) | **430k** (340k-600k), +26% | Soviet soldiers killed or died (not captured) [USSR] 310k (260k-420k); German Army Groups South and A (Korsun, Kamianets, Uman-Botosani, Odessa) [DEU/AUT] 120k (80k-180k) | Up. German from ~80k to 120k: Frieser's 93k killed+missing are 'incomplete' and Overmans implies far more; Soviet +15% for field conscripts. | <http://rus-sky.com/history/library/w/w08.htm> (Krivosheev 2001: 270,198); <https://en.wikipedia.org/wiki/Dnieper–Carpathian_offensive> (Frieser 2007; OKH); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Operation Bagration | 325k (250k-450k) | **380k** (305k-540k), +17% | Soviet soldiers killed or died [USSR] 198k (178k-258k); Polish 1st Army [POL] 1.5k (1.5k-1.5k); German Army Group Centre [DEU/AUT] 180k (125k-280k) | Up. German raised from ~150k to 180k (Frieser-based floor ~130k; Overmans-based ~250k+); Soviet +10% and Polish 1st Army added. | <http://rus-sky.com/history/library/w/w08.htm> (Krivosheev 2001: 178,507 + 1,533); <https://en.wikipedia.org/wiki/Operation_Bagration> (Frieser 2007 pp.593-594; Soviet 158,480 prisoners); <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Warsaw Uprising | 185k (140k-225k) | **187k** (170k-235k), +1% | Civilians of Warsaw [LOCAL] 160k (150k-200k); Home Army insurgents killed and missing [LOCAL] 16k (15k-17k); Polish 1st Army at the Vistula crossings (of 5,660 casualties) [POL] 2.0k (1.5k-2.5k); German forces [DEU/AUT] 9.0k (2.0k-17k) | Unchanged in total; Polish 1st Army (~2k) added and civilians trimmed to 160k. | <https://en.wikipedia.org/wiki/Warsaw_Uprising>; <https://en.wikipedia.org/wiki/Wola_massacre>; Borodziej 2001, The Warsaw Uprising of 1944 |
| Budapest | 124k (90k-170k) | **142k** (120k-185k), +15% | Soviet soldiers (encirclement, siege and Konrad relief battles) [USSR] 60k (50k-80k); Romanian 7th Corps (kept from the old row, unverified) [ROU] 5.0k (3.0k-8.0k); German garrison (30,000 killed) and relief attacks [DEU/AUT] 35k (30k-45k); Hungarian garrison (9,000 killed) and outside [HUN] 11k (9.0k-15k); Budapest civilians (38,000 dead less ~7,000 executed, left to the Holocaust) [LOCAL] 31k (25k-38k) | Up. The old row left out ~10k Axis dead outside the ring and used a smaller Soviet share; civilians kept net of the ~7k Arrow Cross victims. | <https://en.wikipedia.org/wiki/Siege_of_Budapest> (Ungvary 2005); <http://rus-sky.com/history/library/w/w09.htm> (Krivosheev 2001: Budapest operation 80,026) |
| Koenigsberg and East Prussia | 300k (200k-450k) | **335k** (260k-480k), +12% | Soviet soldiers [USSR] 135k (126k-160k); German 4th Army, 3rd Panzer Army and others [DEU/AUT] 150k (100k-220k); East Prussian civilians (flight, Soviet advance, Koenigsberg) [DEU] 50k (30k-100k) | Up. German soldiers from ~120k to 150k (Overmans' final-battles scale; Soviet claims run far higher); civilians trimmed from 60k to 50k (31,940 confirmed; Schieder 30k). | <http://rus-sky.com/history/library/w/w09.htm> (Krivosheev 2001: 126,464); <https://en.wikipedia.org/wiki/East_Prussian_offensive>; <https://en.wikipedia.org/wiki/Evacuation_of_East_Prussia>; <https://en.wikipedia.org/wiki/Battle_of_Königsberg>; <https://en.wikipedia.org/wiki/Heiligenbeil_Pocket>; <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Berlin | 195k (150k-250k) | **207k** (180k-320k), +6% | Soviet soldiers [USSR] 82k (78k-95k); Polish 1st and 2nd Armies [POL] 2.8k (2.8k-2.8k); German soldiers incl. Seelow and Halbe [DEU/AUT] 92k (80k-125k); Berlin and Halbe civilians [LOCAL] 30k (20k-125k) | Slightly up: civilians 22k -> 30k (incl. ~10k at Halbe); Soviet +5%. | <http://rus-sky.com/history/library/w/w09.htm> (Krivosheev 2001: 78,291 + 2,825); <https://en.wikipedia.org/wiki/Battle_of_Berlin> (Mueller 2008 p.673; Antill 2005); <https://de.wikipedia.org/wiki/Schlacht_um_Berlin>; <https://en.wikipedia.org/wiki/Battle_of_Halbe> |
| Poland 1939 | 156k (120k-250k) | **187k** (155k-320k), +20% | Polish soldiers [POL] 66k (63k-100k); Polish civilians (bombing, sieges, executions) [LOCAL] 100k (70k-200k); Ethnic German civilians and Selbstschutz militia [LOCAL] 3.2k (2.0k-5.0k); German soldiers (16,343 killed + 3,500 missing; Overmans 19k for 1939) [DEU/AUT] 18k (16k-20k) | Up. Civilians were 70k; Brzoza & Sowa give ~100,000 (150-200k at the high end); ethnic-German dead added. | <https://en.wikipedia.org/wiki/Invasion_of_Poland> (Brzoza & Sowa 2006 p.510); <https://de.wikipedia.org/wiki/Überfall_auf_Polen>; <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| France and the Low Countries 1940 | 147k (120k-180k) | **143k** (125k-185k), -3% | French soldiers [FRA] 62k (55k-85k); German soldiers (27,074 killed + 18,384 missing; 1944 recount 46,059 Heer + 3,200 Luftwaffe) [DEU/AUT] 46k (45k-50k); British BEF and RAF [GBR] 12k (11k-17k); Belgian soldiers [BEL] 6.1k (6.0k-6.5k); Dutch soldiers [NLD] 2.3k (2.3k-2.4k); Italian soldiers (Alps, June 1940) [ITA] 0.7k (0.6k-0.7k); French civilians (exodus strafing, bombing) [FRA] 12k (8.0k-20k); Dutch civilians (Rotterdam ~900 and others) [NLD] 2.0k (1.5k-2.5k) | Slightly down: French military 65k -> 62k (SHD's 58,829 combat deaths excl. navy), civilians 15k -> 12k (unverified national estimate ~10k). | <https://en.wikipedia.org/wiki/Battle_of_France> (SHD; Frieser; Groehler); <https://de.wikipedia.org/wiki/Westfeldzug>; <https://en.wikipedia.org/wiki/German_casualties_in_World_War_II#Statistical_study_by_Rüdiger_Overmans> |
| Normandy | 122k (100k-150k) | **127k** (105k-155k), +4% | US Army (20,668 killed + ~half of 10,128 missing) [USA] 26k (21k-31k); British Second Army [GBR] 14k (11k-18k); First Canadian Army (5,021 killed + missing dead) [CAN] 5.5k (5.0k-7.5k); Polish 1st Armoured Division (of 2,097 casualties) [POL] 0.6k (0.5k-0.7k); German Army Group B / OB West in Normandy [DEU/AUT] 65k (50k-80k); Norman civilians killed during the campaign [LOCAL] 16k (14k-20k) | Slightly up: German 55k -> 65k (80k buried in Normandy), US 29k -> 25.7k (20,668 killed + half the missing), civilians 20k -> 16.5k (13.6-19.9k). | <https://en.wikipedia.org/wiki/Operation_Overlord>; Zetterling 2000, Normandy 1944; Quellien 2010, Bilan des pertes civiles en Normandie |
| Hamburg firestorm | 37k (34k-43k) | **37k** (34k-43k), +0% | Hamburg residents, forced labourers and POWs [LOCAL] 37k (34k-43k) | Unchanged. 'Most accepted single number is now 37,000.' | <https://en.wikipedia.org/wiki/Bombing_of_Hamburg_in_World_War_II> (Lowe 2007); <https://de.wikipedia.org/wiki/Operation_Gomorrha> |
| Dresden | 23k (18k-25k) | **23k** (18k-25k), +0% | Dresden residents and refugees [LOCAL] 23k (18k-25k) | Unchanged. Historians' commission 2010. | <https://en.wikipedia.org/wiki/Bombing_of_Dresden>; <https://de.wikipedia.org/wiki/Luftangriffe_auf_Dresden>; Neutzner et al. 2010, Abschlussbericht der Historikerkommission |

## Biggest changes (old -> new central)
1. **Kyiv and Uman encirclements: 175k -> 245k (+40%).**
   - The Kiev figure of 616,304 irrecoverable is below the prisoners the Germans claimed (665k at Kyiv, 103k at Uman), so it is an undercount.
   - Kirchubel's ~480k prisoners give a floor of ~136k killed. Adding a share of Krivosheev's 1.16M unrecorded losses gives ~300k. The central is 200k.
   - German dead rise from 25k to 45k.
2. **Białystok-Minsk and Smolensk: 320k -> 415k (+30%).** Same reasoning. Krivosheev's 827,244 plus a ~40% share of the unrecorded 1.16M, less ~680k prisoners, gives 150-610k Soviet dead. The central is 330k; German dead are 85k.
3. **Right-bank Ukraine: 340k -> 430k (+26%).** German dead go from ~80k to 120k: Frieser calls his 93k killed + missing "incomplete", and Overmans implies 150-200k. Soviet dead get +15% for field conscripts.
4. **Poland 1939: 156k -> 187k (+20%).** Polish civilians go from 70k to 100k (Brzoza & Sowa: "about 100,000 or between 150,000 and 200,000").
5. **Kursk: 290k -> 340k (+17%).** Soviet dead rise 240k -> 280k (Krivosheev +20%, toward Töppel's +40%) and German dead 50k -> 60k.
6. **Bagration: 325k -> 380k (+17%).** German dead rise 150k -> 180k, between the Frieser-based ~130k and the Overmans-based ~250k.
7. **Don front: 155k -> 130k (-16%).**
   - The Italian 84,830 counts killed, missing **and captured** (Krinko & Medvedev). About 20.8k were killed and ~44k died as prisoners, so ~65k died in this window.
   - Hungarian dead are ~65k: 42k killed or missing, plus captive deaths.
8. **Budapest: 124k -> 142k (+15%); East Prussia: 300k -> 335k (+12%); Leningrad hunger winter: 450k -> 500k (+11%).**
   - Leningrad: the registered deaths alone (446,388) already equal the old central.

Rows changed by more than 20%: **4 of 25** (Kyiv-Uman, Białystok-Smolensk, Right-bank Ukraine, Poland 1939; Poland is at +20.2%).

Unchanged after verification (within ±5%): Leningrad spring-summer 1942, Leningrad and Volkhov fronts, Caucasus, Rzhev, Warsaw, France 1940, Normandy, Hamburg, Dresden.

## Stalingrad cross-check (no change)
- The Soviet 600k is consistent with the sources fetched.
  - Krivosheev's front tables give the Stalingrad (1st) and Don fronts 429,043 irrecoverable over 218 days, of which 299,130 were missing or captured (37.8% of all losses, Krinko & Medvedev).
  - Most of these were killed. German claims for the Don bend and Kalach, the only big bags, are of the order of the ~75k the row subtracts. This is not a clear error.
- The Romanian (60k) and civilian (100k) components are not contradicted. Krinko gives Romanian losses on the Don and Volga of 158,850 in all.
- **Overlap with this scope:**
  - The Italian and Hungarian armies are only in `Don front`.
  - `Kharkiv and the Don steppe` ends with Voronezh-Voroshilovgrad (to 24 Jul 1942); Krivosheev books the Stalingrad defensive from 17 Jul to different fronts.
  - There is no double count.
- **Fixed:** the battle row's `deaths_central` was written `950000.0` and is now `950000`.

## Overlap checks inside the scope
- **Leningrad fronts vs Leningrad civilians.**
  - The civilian rows are ZAGS-registered city deaths (plus unregistered) and include no soldiers.
  - The militia (~70k Leningraders) are `LOCAL` in the fronts row only.
  - Evacuee deaths (~400k) and the 1941/1943 civilian deaths stay in the parent row.
- **Vyazma vs Moscow.**
  - Vyazma-Bryansk is October 1941 only (Lopukhovsky's 28 Sep-18 Oct losses + the Bryansk pockets).
  - Moscow starts with the 16 Nov-5 Dec defensive phase (~80k, estimated separately from Krivosheev's 514,338 for the whole defensive operation) and runs to 20 Apr 1942.
- **Moscow vs Rzhev.**
  - Jan-Apr 1942 (Rzhev-Vyazma offensive, 272,320) is only in Moscow.
  - Rzhev starts in Jul 1942. Isaev's Jan-Apr 152,943 was removed from his 392,554.
- **Kharkiv/Don steppe vs Stalingrad.** See above.
- **Dnieper vs Kursk vs Right-bank.** These are disjoint Krivosheev operations:
  - Kursk: to 23 Aug 1943.
  - Dnieper: 26 Aug-22 Dec 1943. The Donbas operation (66,166) is left out.
  - Right-bank Ukraine: from 24 Dec 1943.
- **Bagration vs Warsaw.** Bagration ends 29 Aug 1944 and holds no Warsaw dead. The Polish 1st Army's Vistula crossing (Sep 1944) is only in Warsaw.
- **East Prussia vs Berlin vs the German POW row.** All captured (East Prussia ~220k, Berlin ~480k, Bagration ~158k claimed) are excluded from the battle rows.
- **Battle rows vs Soviet POW-camp rows.**
  - Every Soviet component excludes prisoners. In total ~3M captured are subtracted across these rows; they die (or survive) in the six POW-camp rows (2.95M deaths).
  - Prisoners who died on the march from the pocket (Vyazma, Kyiv) are left to the Dulag rows, which list those camps.
- **German missing vs the German POW row.**
  - The German central values lean on Overmans, whose "missing presumed dead" include men who died in Soviet custody unregistered.
  - The POW row's central (280k) uses only documented deaths (363k less Stalingrad), so the centrals do not overlap. Its high (900k) assumes those missing died in captivity, so the highs overlap and must not be added.
- **Holocaust.**
  - Budapest excludes the ~7k Arrow Cross killings, and the Don front excludes the Jewish labour servicemen.
  - Poland 1939 civilians include some thousands of Jewish civilians killed by bombing. The parent row is "non-Jewish Polish citizens", but they are not Holocaust victims, so they are kept here.

## Parent-row problems
- **No overflows.** The film's own row-picking (`euro_check.py`, with this audit and the Asia audit merged) leaves every `wwii` row under the 95% cap. Fullest rows:
  - Soviet Transcaucasia (IRN): 61%.
  - Siberia (SIB): 53%.
  - Central Asia: 45%.
  - Belarus/Baltic: 39%.
  - European Russia: 38%.
  - Germany and Austria: 30%.
- **Totals placed.**
  - Soviet military (battle rows) 4.85M + Soviet POW rows 2.95M = 7.8M. This is below Krivosheev's 8.67M demographic military loss and well below revisionist totals.
  - German military in the battle rows: 1.61M. Overmans' Eastern Front alone is ~3.6M to May 1945.
  - So the parent rows do not look too low for these places.
- **Austrian dead are charged to the wrong row.**
  - `AUT` shares (~87k across these rows) go to "rest of Europe", not "Germany and Austria". `_pick_row` picks the row whose spot is nearest in units of its spread, and Austria's centroid is closer to the wide rest-of-Europe spot.
  - The totals are right, but the leftovers are skewed. Coding Austrians as `DEU`, or matching rows by country, would fix it.
- **No parent for some deaths:**
  - Soviet dead in Poland 1939 (1,475) and the Winter War (Soviet rows start in 1941).
  - Jewish civilians killed in Sept 1939 (the Poland row is non-Jewish).
  - These are left out, as before.

## Confidence and unverified pieces
- **Low confidence:**
  - All 1941 Soviet components. The unrecorded-loss share and the prisoner discount are my judgment between a Krivosheev floor and an upper bound.
  - Crimea, Kharkiv, Caucasus and Rzhev.
  - German soldiers in East Prussia: no archival count exists; only Soviet claims and Overmans' aggregate.
- **Estimates of mine:**
  - Moscow 16 Nov-5 Dec phase (~80k irrecoverable).
  - The Soviet share of the Budapest operation near the city (60k of 80,026).
  - German relief-attack dead at Budapest (~5k).
  - Polish 1st Army dead at Warsaw (~2k of 5,660 casualties).
  - Polish dead at Falaise (~600 of 2,097 casualties).
  - The unregistered Leningrad deaths (~80k in the hunger winter).
- **Kept from the old row, not re-verified:**
  - Romanian dead at Budapest (5k).
  - The Soviet republic splits.
- **Unverified national estimates:**
  - French civilians in May-June 1940 (~10,000, from the exodus). Belgian civilians are not included.
  - Berlin civilians (central 30k; the popular 100-125k is the high).
- **Not included:**
  - Finnish dead on the Karelian Isthmus.
  - Hungarian dead at Voronezh in Jul 1942.
  - Romanian and Hungarian dead in Right-bank Ukraine.
  - Allied airmen over Normandy.
