# Audit: WWII in Asia and the Pacific (26 rows of `inputs/places/wwii.csv`)

Files:
- `audit_wwii_asia.csv`: all 26 rows, revised or re-verified. Header is the places header with `old_place` first.
- `asia_build.py`: builds each row from its components.
- `asia_check.py`: runs the film's own `places.py` row-picking, read-only, on the merged CSV.

Nothing under `/home/user/nhl-sog-predictor` was edited.

## Method
- **Deaths only, built by side.** Each row is built from components: each army, plus civilians. The `from` shares are recomputed from the components.
- **Chinese battles.** Reported casualties are killed + wounded (+ missing). Deaths are taken as the reported *killed* plus a third of the *missing*. Where only casualty totals exist, ~40-42% is used: the Nationalist army's war totals are 1.32M killed against 1.80M wounded and 0.12M missing (Taiwanese official figures, via Wikipedia).
  - Japanese claims of Chinese dead are not used.
  - Japanese dead in China are not in the Chinese rows. They come from a different parent (`wwii`) and sit in "Japanese army in China".
- **Japanese dead.** The source is the Ministry of Health, Labour and Welfare's current area tables (戦没者概数, Aug 2026 editions, fetched as PDFs). They succeed the MHW 1964 table that Dower used, and they include civilians.
  - **Philippines:** Luzon 272,500; Leyte 79,000; Mindanao 63,700; Cebu 11,700; Negros 11,200; Panay 1,200. Another 78,700, mostly lost at sea, have no island.
  - **Central Pacific:** Saipan 55,300; Tinian 15,500; Guam 20,000.
  - **Solomons:** Guadalcanal 22,000.
  - **New Guinea:** eastern New Guinea 127,600; West Irian 53,000.
  - **Burma and India:** Myanmar 137,000; India 30,000.
  - **China:** China proper 465,700, incl. 23,500 lost at sea; northeast China 245,400.
  - **Japan:** Okinawa 188,140; Iwo Jima 21,900.
  - **Other counts.** US body counts of Japanese dead are lower everywhere (they miss those who starved unseen), so they are used as the lows.
- **Allied dead** come from the battle articles and the US and Australian official figures they cite.
- **Wounded and surviving prisoners are never counted.** Japanese POWs who died in the USSR (MHLW 54,400) are not placed in this scope.

## Rows
| place | old central (low-high) | new central (low-high) | components (central, low-high) | why it changed | sources |
|---|---|---|---|---|---|
| Shanghai | 100k (60k-150k) | **100k** (65k-150k), +0% | Chinese soldiers [CHN] 90k (60k-130k); Shanghai civilians [LOCAL] 10k (5k-20k) | Unchanged. Re-derived: ~220k casualties x 42% killed (ROC war ratio) = ~90k soldiers; civilians unverified. | <https://en.wikipedia.org/wiki/Battle_of_Shanghai> <https://zh.wikipedia.org/wiki/淞滬會戰> <https://en.wikipedia.org/wiki/Second_Sino-Japanese_War> |
| Nanjing massacre | 200k (40k-300k) | **150k** (40k-300k), -25% | Civilians of the city and six counties [LOCAL] 85k (40k-150k); Surrendered/captured soldiers and men taken as soldiers [CHN] 65k (30k-100k) | Old 200k was the top of the newer-estimate range (100-200k). Central moved to its middle (Kasahara's 'over 100k, near 200k'; Wakabayashi 109k for December alone). Split ~57% civilians / 43% POWs and presumed soldiers. | <https://en.wikipedia.org/wiki/Nanjing_Massacre> Kasahara 1997, Nankin jiken Wakabayashi (ed.) 2007, The Nanking Atrocity 1937-38 |
| Xuzhou | 85k (60k-120k) | **105k** (65k-145k), +24% | Chinese soldiers [CHN] 95k (60k-115k); Civilians [LOCAL] 10k (5k-30k) | Old 76k soldiers was below the Fifth War Zone's own count of 85,324 killed. Now killed + a third of 27,968 missing. | <https://en.wikipedia.org/wiki/Battle_of_Xuzhou> Su Shengxiong 2018 (via Wikipedia) Dorn 1974, The Sino-Japanese War 1937-41 |
| Yellow River flood | 500k (400k-900k) | **500k** (400k-893k), +0% | Flood victims (drowning, then famine and epidemics) [LOCAL] 500k (400k-893k) | Unchanged. 400-500k is the scholarly range; 893,303 (1948) is discredited. | <https://en.wikipedia.org/wiki/1938_Yellow_River_flood> <https://en.wikipedia.org/wiki/Battle_of_Xuzhou> Lary 2001, War in History 8:191-207 |
| Wuhan | 150k (85k-300k) | **165k** (100k-290k), +10% | Chinese soldiers [CHN] 130k (85k-200k); Civilians (bombing, Jiujiang massacres) [LOCAL] 35k (15k-90k) | Soldiers rebuilt as ~40% of the casualty range (254,628 initial report to 400-500k postwar), up from 115k to 130k. | <https://en.wikipedia.org/wiki/Battle_of_Wuhan> MacKinnon 2008, Wuhan 1938 <https://en.wikipedia.org/wiki/Second_Sino-Japanese_War> |
| Changsha | 45k (25k-70k) | **45k** (37k-65k), +0% | Chinese soldiers [CHN] 40k (35k-50k); Civilians [LOCAL] 5k (2k-15k) | Unchanged. Verified: 35,435 killed + 14,040 missing in the 2nd and 3rd battles. | <https://en.wikipedia.org/wiki/Battle_of_Changsha_(1941)> <https://en.wikipedia.org/wiki/Battle_of_Changsha_(1942)> |
| Zhejiang-Jiangxi | 200k (100k-250k) | **130k** (75k-280k), -35% | Civilians (reprisals, Unit 731 cholera/plague/anthrax) [LOCAL] 100k (50k-250k); Chinese soldiers [CHN] 30k (24k-43k) | Old 200k leaned on the wartime 250,000 claim (Chennault, Fr. Smith). Provincial records are far lower (Zhejiang 66,526 killed, wounded or missing). Civilians now 100k; 30k soldiers added; row made converge. | <https://en.wikipedia.org/wiki/Zhejiang-Jiangxi_campaign> Scott 2015, Target Tokyo Chennault 1949, Way of a Fighter |
| Operation Ichi-Go | 250k (150k-500k) | **225k** (135k-450k), -10% | Chinese soldiers [CHN] 150k (104k-250k); Civilians (Henan, Hengyang, Guangxi flight) [LOCAL] 75k (30k-200k) | Soldiers now built from the 1944 MMA count (124,893 killed, 60,149 missing) rather than a disease-inflated 175k. | <https://en.wikipedia.org/wiki/Operation_Ichi-Go> Hsiung & Levine 1992, China's Bitter Victory Paine 2014, The Japanese Empire |
| Japanese army in China | 455k (400k-500k) | **442k** (396k-480k), -3% | Japanese dead in China proper incl. Hong Kong, on land [JPN] 442k (396k-480k) | Current MHLW 465,700 less 23,500 lost at sea (was MHW 1964 455,700). | <https://www.mhlw.go.jp/content/12103000/001223055.pdf> <https://en.wikipedia.org/wiki/Second_Sino-Japanese_War> Dower 1986, War Without Mercy |
| Guadalcanal | 32k (26k-38k) | **35k** (29k-38k), +9% | Japanese on the island (army 19,200, of whom 8,500 in combat; + navy ground) [JPN] 23k (19k-26k); Japanese sailors [JPN] 4k (3k-4k); Japanese aircrew [JPN] 1k (1k-2k); US ground, naval and air [USA] 7k (7k-7k); Australian (HMAS Canberra) [AUS] 0k (0k-0k) | Navy (3,543) and aircrew (1,200) added explicitly to ~23k on the island. | <https://www.mhlw.go.jp/content/12103000/001223067.pdf> <https://en.wikipedia.org/wiki/Guadalcanal_campaign> <https://www.newworldencyclopedia.org/entry/Guadalcanal_campaign> Frank 1990, Guadalcanal |
| New Guinea: Kokoda to Huon | 54k (40k-75k) | **54k** (40k-68k), +0% | Japanese [JPN] 45k (33k-55k); Australian [AUS] 4k (3k-4k); US [USA] 2k (2k-3k); Papuans [LOCAL] 3k (1k-6k) | Unchanged. Battle tolls sum to ~31k Japanese, plus rear-area disease. | <https://en.wikipedia.org/wiki/Kokoda_Track_campaign> <https://en.wikipedia.org/wiki/Battle_of_Buna%E2%80%93Gona> <https://en.wikipedia.org/wiki/Salamaua%E2%80%93Lae_campaign> <https://en.wikipedia.org/wiki/Huon_Peninsula_campaign> <https://ja.wikipedia.org/wiki/ニューギニアの戦い> |
| New Guinea: Wewak, Aitape, Hollandia | 140k (100k-160k) | **139k** (113k-154k), -1% | Japanese, eastern New Guinea 1944-45 and Dutch New Guinea [JPN] 136k (110k-150k); US [USA] 3k (2k-3k); Australian [AUS] 1k (0k-1k) | Total kept. Shares recomputed with fewer Allied dead (US 2.7k, Aus 0.7k). Centre moved to 139.0E and radius set to 650 km so the row covers Dutch New Guinea (Biak, Sarmi, Idore), whose 53k dead it holds. | <https://www.mhlw.go.jp/content/12103000/001223068.pdf> <https://www.mhlw.go.jp/content/12103000/001223070.pdf> <https://en.wikipedia.org/wiki/New_Guinea_campaign> <https://ja.wikipedia.org/wiki/ニューギニアの戦い> |
| Marianas: Saipan, Guam, Tinian | 73k (60k-85k) | **73k** (65k-97k), -0% | Japanese military: Saipan ~29,500 (25,144 buried + ~5,000 later) [JPN] 30k (25k-30k); Japanese military: Guam [JPN] 18k (18k-18k); Japanese military: Tinian [JPN] 8k (6k-8k); Japanese settler civilians (Saipan 8-10k, Tinian 2,610) [JPN] 10k (9k-13k); Korean labourers and settlers [KOR] 2k (0k-3k); US (Saipan ~3,540, Guam 1,760, Tinian ~380) [USA] 6k (6k-6k) | Unchanged within 1%. Components verified island by island. | <https://www.mhlw.go.jp/content/12103000/001223066.pdf> <https://en.wikipedia.org/wiki/Battle_of_Saipan> <https://en.wikipedia.org/wiki/Battle_of_Guam_(1944)> <https://en.wikipedia.org/wiki/Battle_of_Tinian> |
| Leyte | 91k (70k-105k) | **98k** (81k-106k), +7% | Japanese on Leyte [JPN] 79k (65k-80k); Japanese at Leyte Gulf [JPN] 12k (10k-12k); US ground (3,504 + 119 non-battle + 183 missing) [USA] 4k (4k-4k); US naval, Leyte Gulf [USA] 3k (3k-3k) | Old central left out part of its own components (79k + 12k Japanese alone = 91k). Now includes US ground and naval dead. | <https://www.mhlw.go.jp/content/12103000/001223069.pdf> <https://en.wikipedia.org/wiki/Battle_of_Leyte> <https://en.wikipedia.org/wiki/Battle_of_Leyte_Gulf> |
| Luzon | 253k (200k-290k) | **303k** (215k-330k), +20% | Japanese (MHLW 272,500 less 1941-42 and 1944 dead) [JPN] 265k (193k-272k); US Army killed and missing [USA] 8k (8k-8k); US Navy, Lingayen-Mindoro kamikazes [USA] 2k (2k-2k); Filipino civilians outside Manila [LOCAL] 25k (10k-40k); Filipino guerrillas [LOCAL] 3k (2k-5k) | Japanese from MHLW 272,500 less earlier deaths (was ~240k). ~28k Filipino civilians and guerrillas outside Manila added as LOCAL. | <https://www.mhlw.go.jp/content/12103000/001223069.pdf> <https://en.wikipedia.org/wiki/Battle_of_Luzon> <https://en.wikipedia.org/wiki/Philippines_campaign_(1941%E2%80%931942)> |
| Manila | 100k (80k-150k) | **100k** (80k-150k), +0% | Filipino civilians [LOCAL] 100k (80k-150k) | Unchanged. | <https://en.wikipedia.org/wiki/Battle_of_Manila_(1945)> Connaughton, Pimlott & Anderson 1995, The Battle for Manila |
| Mindanao and the Visayas | 82k (60k-95k) | **87k** (47k-90k), +5% | Japanese (MHLW Mindanao 63,700, Cebu 11,700, Negros 11,200, Panay 1,200) [JPN] 85k (45k-88k); US (Mindanao 820, Zamboanga 221, Visayas 835) [USA] 2k (2k-2k) | MHLW 87,800 less ~3k from 1942-44 (was 80k); US now 1,900 from Smith. | <https://www.mhlw.go.jp/content/12103000/001223069.pdf> <https://en.wikipedia.org/wiki/Battle_of_Mindanao> <https://en.wikipedia.org/wiki/Battle_of_the_Visayas> |
| Iwo Jima | 26k (24k-28k) | **27k** (25k-29k), +5% | Japanese garrison (21,060 less 1,083 captured) [JPN] 20k (18k-22k); US [USA] 7k (7k-7k) | Japanese = garrison less 1,083 captured (~20,000, was 18,800). | <https://en.wikipedia.org/wiki/Battle_of_Iwo_Jima> <https://www.mhlw.go.jp/content/12103000/001223072.pdf> |
| Okinawa | 200k (150k-240k) | **201k** (150k-240k), +0% | Okinawan civilians [LOCAL] 94k (40k-120k); Okinawan conscripts and Boeitai [LOCAL] 28k (28k-28k); Japanese soldiers from other prefectures [JPN] 66k (66k-66k); US [USA] 13k (13k-14k) | Unchanged (prefecture count, matches MHLW 188,140). | <https://www.mhlw.go.jp/content/12103000/001223072.pdf> <https://en.wikipedia.org/wiki/Battle_of_Okinawa> <https://ja.wikipedia.org/wiki/沖縄戦> |
| Imphal and Kohima | 34k (18k-45k) | **34k** (28k-41k), +2% | Japanese 15th Army [JPN] 30k (25k-35k); Indian Army [IND] 3k (2k-4k); British [GBR] 1k (1k-2k) | Japanese 30,500 (Japanese count; MHLW 30,000 in India); Allied ~4,000 (estimate). | <https://www.mhlw.go.jp/content/12103000/001223057.pdf> <https://ja.wikipedia.org/wiki/インパール作戦> <https://en.wikipedia.org/wiki/Battle_of_Imphal> <https://en.wikipedia.org/wiki/Battle_of_Kohima> <https://www.nam.ac.uk/explore/battle-imphal> |
| Burma 1944-45 | 90k (50k-110k) | **100k** (85k-122k), +11% | Japanese [JPN] 95k (80k-115k); Indian Army [IND] 4k (3k-5k); British [GBR] 2k (1k-2k) | Japanese raised to ~95k: 80k in 1945 alone (Hastings, Mitter) + Aug-Dec 1944. The old 85k residual was too low. | <https://en.wikipedia.org/wiki/Burma_campaign_(1944%E2%80%931945)> <https://www.mhlw.go.jp/content/12103000/001223058.pdf> Hastings 2012, All Hell Let Loose Mitter 2013, Forgotten Ally |
| Burma Railway | 100k (85k-130k) | **94k** (75k-120k), -6% | Malayan labourers (Tamils, Chinese, Malays) [MYS] 40k (33k-60k); Burmese labourers [MMR] 37k (30k-42k); Javanese labourers [IDN] 3k (1k-6k); British POWs [GBR] 7k (7k-7k); Dutch POWs [NLD] 3k (3k-3k); Australian POWs [AUS] 3k (3k-3k); US POWs [USA] 0k (0k-0k); Japanese and Korean railway troops [JPN] 1k (1k-1k) | Romusha central 80k (AWM: 'perhaps 75,000' by Oct 1943; 80-100k overall), was 85k; ~1,000 Japanese and Korean railway troops added. | <https://en.wikipedia.org/wiki/Burma_Railway> <https://www.awm.gov.au/visit/exhibitions/stolenyears/ww2/japan/burmathai> Kratoska 2006, The Thailand-Burma Railway 1942-1946 |
| Manchuria 1945 | 122k (60k-200k) | **122k** (80k-175k), +0% | Kwantung Army [JPN] 30k (21k-40k); Japanese settlers and refugees, Aug-Dec 1945 [JPN] 80k (50k-120k); Soviet (RUS) [RUS] 8k (6k-8k); Soviet (UKR) [UKR] 3k (2k-3k); Soviet (BLR) [BLR] 1k (1k-1k); Soviet (KAZ) [KAZ] 0k (0k-0k); Soviet (UZB) [UZB] 0k (0k-0k) | Unchanged total. Components re-verified (Krivosheev; Japanese medical records 21,389; Ran 2010: 80,000 settlers in the six months after 9 Aug). | <https://en.wikipedia.org/wiki/Soviet_invasion_of_Manchuria> <https://ja.wikipedia.org/wiki/満蒙開拓移民> <https://m.earticle.net/Article/A134035> <https://www.mhlw.go.jp/content/12103000/001223054.pdf> |
| Tokyo firebombing | 90k (80k-120k) | **95k** (84k-120k), +6% | Tokyo residents [LOCAL] 95k (84k-120k) | The Tokyo Fire Department figure is 97,000, not 83,793. Central moved to Frank's 90-100k midpoint. | <https://en.wikipedia.org/wiki/Bombing_of_Tokyo_(10_March_1945)> Frank 1999, Downfall |
| Hiroshima | 140k (90k-166k) | **140k** (90k-166k), +0% | City population incl. ~20k soldiers and Korean labourers [LOCAL] 140k (90k-166k) | Unchanged; verified with the City of Hiroshima (~140,000 by Dec 1945) and RERF. | <https://www.city.hiroshima.lg.jp/english/peace/1029875/1010074.html> <https://www.rerf.or.jp/en/faq/> <https://en.wikipedia.org/wiki/Atomic_bombings_of_Hiroshima_and_Nagasaki> |
| Nagasaki | 74k (60k-80k) | **74k** (60k-80k), +0% | City population [LOCAL] 74k (60k-80k) | Unchanged; verified (73,884 by Dec 1945, 1950 committee report). | <https://nagasakipeace.jp/en/reference/materials/kids/higai.html> <https://www.rerf.or.jp/en/faq/> |

## Biggest changes
1. **Zhejiang-Jiangxi: 200k -> 130k (-35%).**
   - The old figure rested on the wartime "250,000" of Chennault and Fr. Smith, the maximum.
   - Zhejiang's own 1942 records give 66,526 civilians killed, wounded or missing.
   - Now 100k civilians plus 30k soldiers (24,361 killed + a third of 18,640 missing). The row is now `converge`.
2. **Nanjing massacre: 200k -> 150k (-25%).** This is the middle of the newer-estimate range, 100-200k: Kasahara gives "over 100,000 and close to 200,000", and Wakabayashi 109,475 in December 1937 alone. 200k (IMTFE) and 300k (PRC) remain in the high.
3. **Xuzhou: 85k -> 105k (+24%).** The old row's 76k soldiers was below the Fifth War Zone's own 85,324 killed. A third of the 27,968 missing is now added.
4. **Luzon: 253k -> 303k (+20%, just under the threshold).**
   - Japanese go from ~240k to 265k: MHLW 272,500, less the 1941-42 dead.
   - ~28k Filipino civilians and guerrillas killed outside Manila are added. Wikipedia puts 120-140k civilian dead on Luzon, ~100k of them in Manila.
5. **Burma 1944-45: 90.5k -> 100.5k (+11%).** 80,000 Japanese died in 1945 alone (Hastings, Mitter).
6. **Wuhan: 150k -> 165k (+10%).** **Ichi-Go: 250k -> 225k (-10%).**
7. **Smaller changes:**
   - **Guadalcanal: 32k -> 35k.** Navy and aircrew are now explicit.
   - **Leyte: 91k -> 98k.** The old central did not add up its own components.
   - **Tokyo: 90k -> 95k.** The Tokyo Fire Department figure is 97,000, not 83,793 as the old note said.
   - **Iwo Jima: 25.6k -> 26.8k.**
   - **Burma Railway: 99.5k -> 93.6k.**
   - **Japanese army in China: 455k -> 442k.** This is current MHLW, less those lost at sea.
8. **Rows changed by more than 20%:** 3 of 26 (Zhejiang-Jiangxi, Nanjing, Xuzhou). Luzon is at +19.6%.

Unchanged after verification: Shanghai, the Yellow River flood, Changsha, both New Guinea rows (totals), the Marianas, Manila, Okinawa, Manchuria, Hiroshima and Nagasaki.

## Overlap checks (inside this scope)
- **Chinese rows and "Japanese army in China."** No overlap. The Chinese rows hold Chinese dead only (parent `sino_japanese_war`). All Japanese dead in China proper are in the one `wwii` row, which runs 1937-07 to 1945-08.
- **Manila and Luzon.** Manila is Filipino civilians only. The ~16k Japanese defenders and 1,010 Americans are in Luzon, and Luzon's LOCAL share excludes Manila.
- **Philippine islands.** Leyte, Luzon and Mindanao/Visayas follow MHLW's island split, with no double count. The 78,700 Japanese without an island (mostly lost at sea) are not placed. Leyte Gulf's sailors are inside the Leyte row.
- **The two New Guinea rows.** Both come from one total: MHLW eastern New Guinea 127,600 + West Irian 53,000, split 45k for 1942-43 and 135.6k for 1944-45.
- **Imphal and Burma 1944-45.**
  - Imphal holds the India-side dead to Aug 1944 (MHLW 30,000 in India).
  - Burma 1944-45 starts in Aug 1944. It holds the later retreat deaths inside Burma, which MHLW counts in Myanmar.
- **Burma Railway and Burma 1944-45.** The railway window ends Dec 1943, so later railway deaths stay in the parent rows.
- **Okinawa and the Japanese army.** Okinawa's 65,908 Japanese soldiers are only here. Okinawan conscripts are LOCAL, not JPN.
- **Hiroshima.** Its ~20k soldiers are inside the local row, not in any battle row.
- **Manchuria.** The row holds Aug-Dec 1945 only. Winter 1945-46 camp deaths and Siberian internment deaths are not placed.

## Left out on purpose (stay in parent rows or belong elsewhere)
- **Chinese NCAC/Y-Force dead in north Burma.** Their parent is `sino_japanese_war`. A `CHN` share on a `wwii` row would wrongly draw from the Japan or Korea row.
- **INA dead at Imphal.** Not verified.
- **~1,000 Chamorros on Guam and Saipan.** A `LOCAL` share there would be charged to the Japan row, because Guam and Saipan map to the `POL` vessel and `wwii` has no row there.
- **Koreans on Okinawa.** The Cornerstone of Peace lists 463 names. Higher estimates were not verified.
- **Combat deaths of Nanjing's defenders.** Only the massacre is counted.

## Parent-row problems
- **No overflows.** The film's row-picking was run on the merged CSV (`asia_check.py`). Every row stays under the 95% cap.
  - `wwii` Japan (EAS, 2.7M): 1.91M placed (71%), all of it from these rows.
  - Yellow River flood row (600k): 500k (83%).
  - China row (12.9M): 0.92M (7%).
  - Philippines (700k): 128k (18%).
  - Malaya (150k): 43k (29%).
  - USA+Canada (460k): 90k (20%).
  - PNG (15k): 3k (20%).
- **The Japan row looks too low.**
  - MHLW's official total is 3.1 million Japanese war dead: 2.3M military plus 0.8M civilians (MHLW 戦没者 page). The catalogue row is 2.7M.
  - These places take 1.91M. That leaves 0.79M for at least ~1.2M other MHLW-listed dead:
    - Solomons/Bismarcks beyond Guadalcanal: ~97k.
    - Central Pacific beyond the Marianas: ~156k.
    - About 300k lost at sea.
    - 54k in Soviet captivity.
    - About 190k killed in raids on other cities.
    - Taiwan, Korea, Indochina, Malaya, Indonesia and Borneo: ~170k.
  - Suggest a central of ~3.0-3.1M.
- **The Yellow River flood catalogue row (600k) is above the scholarly 400-500k.** With this place at 500k, the leftover 100k is spread over 1938. Suggest 500k.
- **The `sino_japanese_war` row note says "Nanjing massacre (median ~260,000)".** This no longer matches the place (150k) and should be updated.
- **The PNG row (15k) may be low.** Japanese Wikipedia gives 40-50k native deaths, unsourced. Not changed.
- **Javanese labourers on the Burma Railway (3k) are charged to the Malaya row, not the Dutch East Indies row.** `_pick_row` chooses by distance over spread, and Malaya's spread is wider. The effect is minor.

## Confidence and unverified pieces
- **Civilian components in the Chinese rows.** Only the casualty totals behind them were verified, so these are low confidence: Shanghai 10k, Xuzhou 10k, Wuhan 35k, Changsha 5k, Ichi-Go 75k, Zhejiang-Jiangxi 100k.
- **Estimates of mine:**
  - Allied dead at Imphal-Kohima: ~4,000, taken as ~24% of 16,667 casualties.
  - Burma Aug-Dec 1944: ~15k Japanese dead.
  - Japanese civilians dead in Manchuria by Dec 1945: 80k.
  - The 45k/135.6k New Guinea split.
- **Burma Railway romusha by nationality.** Ranges are from the fetched estimates, but Kratoska's table itself could not be fetched.
