# Places inside the pre-1950 wars (all except the WWII era)

`places_wars.csv`: 124 rows in 31 parent events. Together they hold 62.4M central deaths. Every row was checked against `crises_pre1950.csv` by the build script, `wars_build/build.py`. The checks were:
- Every `from` code and every local place maps to a vessel that has a parent row for that event.
- Every place's years fall inside that parent row's years.
- For each event and vessel, the placed low, central and high values do not exceed the parent's low, central and high.

Shares in `from` are rounded to 3 decimals and sum to exactly 1.

## Coverage

| parent_id | rows | placed (central) | parent total | share |
|---|---:|---:|---:|---:|
| taiping_rebellion | 6 | 16,200,000 | 20,000,000 | 81% |
| mongol_conquests | 17 | 11,850,000 | 28,200,000 | 42% |
| dungan_revolt | 2 | 7,400,000 | 8,000,000 | 92% |
| ming_qing_transition | 7 | 5,470,000 | 20,000,000 | 27% |
| wwi | 23 | 5,417,000 | 14,119,000 | 38% |
| an_lushan | 2 | 5,000,000 | 13,000,000 | 38% |
| french_rev_napoleonic_wars | 8 | 1,355,000 | 4,500,000 | 30% |
| late_han_three_kingdoms | 2 | 1,100,000 | 10,000,000 | 11% |
| stalin_repression_1930s | 2 | 870,000 | 1,500,000 | 58% |
| thirty_years_war | 6 | 810,000 | 6,000,000 | 14% |
| commonwealth_deluge | 2 | 800,000 | 2,000,000 | 40% |
| armenian_genocide | 4 | 770,000 | 1,000,000 | 77% |
| partition_india | 3 | 575,000 | 750,000 | 77% |
| wars_three_kingdoms_britain | 4 | 540,000 | 600,000 | 90% |
| us_civil_war | 3 | 502,920 | 700,000 | 72% |
| great_northern_war | 3 | 480,000 | 1,000,000 | 48% |
| russian_civil_war | 4 | 470,000 | 4,000,000 | 12% |
| jewish_roman_wars | 3 | 470,000 | 600,000 | 78% |
| caucasus_conquest | 2 | 350,000 | 800,000 | 44% |
| paraguayan_war | 2 | 280,000 | 420,000 | 67% |
| timur | 4 | 260,000 | 9,500,000 | 3% |
| balkan_wars | 2 | 220,000 | 400,000 | 55% |
| spanish_civil_war | 3 | 210,000 | 400,000 | 52% |
| crimean_war | 1 | 200,000 | 450,000 | 44% |
| ottoman_greek_assyrian_genocides | 2 | 180,000 | 600,000 | 30% |
| cuban_wars | 1 | 170,000 | 350,000 | 49% |
| chinese_civil_war_2 | 1 | 150,000 | 2,500,000 | 6% |
| crusades | 2 | 150,000 | 3,000,000 | 5% |
| mexican_revolution | 1 | 50,000 | 1,000,000 | 5% |
| warlord_era_china | 1 | 40,000 | 500,000 | 8% |
| italo_ethiopian_war | 1 | 19,000 | 400,000 | 5% |

### What is in it
- **WWI (23 rows).**
  - **Western Front:** seven rows covering every year:
    - 1914 and 1915;
    - Verdun and the Somme;
    - Arras and the Chemin des Dames;
    - Passchendaele;
    - the 1918 German offensives and the Hundred Days.
  - **Eastern Front:**
    - Tannenberg/East Prussia/Poland 1914;
    - Galicia and the Carpathians, Aug 1914-Apr 1915;
    - Gorlice-Tarnów;
    - the Great Retreat;
    - Brusilov.
  - **Other fronts:** the Isonzo battles with Caporetto, Gallipoli and Sarıkamış.
  - **Civilian disasters (local or mostly local):**
    - Serbian typhus and the Serbian retreat through Albania;
    - Moldavian typhus in 1917;
    - the Mount Lebanon famine;
    - the 1916 Urkün in Semirechye;
    - German blockade hunger;
    - German East Africa.
- **Armenian genocide (4 rows).** These are four phases:
  - massacres in the eastern vilayets (local);
  - death marches (converge from TUR);
  - camps along the railway and the Euphrates (converge);
  - Deir ez-Zor in Jul-Dec 1916 (converge).

  Two further rows belong to the Ottoman Greek/Assyrian event: the Pontus deportations and Smyrna in 1922.
- **Russian Civil War (4 rows).** The Ukrainian pogroms, Petrograd's hunger in 1918-20, Kolchak's collapse with typhus in the winter of 1919-20, and the famine in the Republic of Armenia in 1918-19.
- **Mongols (17 rows).** All ten requested sacks. Zhongdu, Kaifeng and Sichuan are separate rows, and North China is split into two periods. The other rows cover Khorasan/Afghanistan, north-east Rus' and Hungary.
- **Timur (4 rows).** Isfahan, Delhi, Aleppo with Damascus, and Baghdad 1401.
- **Taiping (6 rows).** Five provinces with their periods, plus the fall of Nanjing.
- **Ming-Qing (7 rows).** Shaanxi, Henan, the Kaifeng siege and flood of 1642, Sichuan, Yangzhou, Jiading and Guangzhou.
- **An Lushan (2 rows).** Chang'an/Luoyang and Hebei.
- **Late Han (2 rows).** The Yellow Turbans, and Luoyang, Chang'an and Xu in 190-195.
- **Thirty Years' War (6 rows).** Magdeburg, Württemberg, the Palatinate, Thuringia, Mecklenburg and Pomerania.
- **Napoleonic (8 rows).**
  - the Vendée;
  - Zaragoza, Spain, and Portugal 1810-11;
  - Russia 1812 in three rows (advance, Borodino, retreat);
  - Leipzig.
- **Others.**
  - US Civil War: Eastern and Western theaters, Andersonville.
  - Paraguayan War: the Humaitá front and Paraguay 1868-70.
  - Sevastopol.
  - Spanish Civil War: the two repressions, and Teruel with the Ebro.
  - Changchun 1948 and Xi'an 1926.
  - Dungan revolt: Shaanxi and Gansu.
  - Partition: Punjab, Jammu, and Calcutta/Noakhali/Bihar.
  - Crusades: the First Crusade and Acre.
  - Jewish-Roman wars: Galilee, Jerusalem 70 and Bar Kokhba.
  - Wars of the Three Kingdoms (Britain): four rows.
  - Great Northern War: three plague rows.
  - Commonwealth deluge: Belarus and Ukraine.
  - Caucasus: Circassia and the Ottoman ports.
  - Balkan Wars: Thrace and Macedonia.
  - Cuban reconcentration.
  - Italo-Ethiopian war: Addis Ababa 1937.
  - Mexican Revolution: Morelos.
  - Stalinist repression: kulak special settlements and the Great Terror.

### Not placed, and why
- **Atlantic slave trade.** Skipped as instructed. The parent is already split by embarkation region.
- **Parents that are already precise.** These rows already sit at the right province and span the right years, and I found no finer detail with numbers:
  - Miao (Guizhou), Panthay (Yunnan), Punti-Hakka (Pearl River delta);
  - the Maratha raids on Bengal, Dzungar (Ili), Tây Sơn.
- **Parents with no defensible regional or temporal numbers:**
  - Mfecane, Mughal-Maratha, Menelik, Mahdist Sudan;
  - Congo Free State (the rubber zone is known, but no count goes beyond the debated national total);
  - Algeria, the Hundred Years' War, the Nine Years' War;
  - the Spanish and Austrian Succession wars, the Seven Years' War;
  - Xin/Red Eyebrows, the fall of the Yuan;
  - Chinese Civil War phase 1: the Jiangxi soviet has no count, and the Central Plains War exists only as casualties;
  - the Huaihai campaign: casualties only;
  - Spanish American independence, the Philippine-American War (Batangas is documented only as a census gap of 54k for 1899-1903), the Indian rebellion of 1857, Aboriginal Australia and the Maori.
- **Timur.** Only 3% is placed. Beyond the four sacks, every Timur figure is a chronicle claim, and the parent rows are already split by theater.

## Method and judgment calls

1. **Deaths versus casualties.** Every row is deaths: killed, died of wounds, disease, hunger or captivity, and murdered. Where only casualties exist, I used these rules:
   - For WWI battles, I used White's battle-death estimates where they exist (necrometrics battles page).
   - Otherwise I took deaths as about 20-30% of killed plus wounded, and excluded prisoners who survived. This applies to Gorlice, the Great Retreat and Brusilov, where 0.4-1M prisoners are excluded.
   - Each row's notes give the casualty figure next to the deaths figure.
   - Population losses (Thirty Years' War, Taiping, Dungan, Morelos, Belarus) include flight and lost births. Deaths are taken as about 60-70% of the loss, or scaled to the parent's central value.
2. **Western Front by year.** French recorded dead per year (Prost: 301k, 349k, 252k, 164k, 235k) anchor each year. German and British shares are estimated from their national annual totals. The named battles (Verdun, the Somme, Arras/Chemin des Dames, Passchendaele) are separate rows, and the year rows exclude them. 1916 outside the two big battles and 1917 outside the named battles are left in the parent, about 0.2-0.3M. The eight Western Front rows place about 2.5M of roughly 3.5M Western Front dead.
3. **Home countries for converge rows**, by modern borders:
   - **German army:** DEU 81%, POL 16% (Silesia, Pomerania, Posen, West and East Prussia), FRA 3% (Alsace-Lorraine).
   - **Austro-Hungarian army:** split by the army's 1914 nationality shares onto AUT, CZE, HUN, SVK, POL, UKR, ROU, HRV, BIH, SRB and SVN.
   - **Russian Empire army:** split roughly by 1897 population, with Central Asia exempt from conscription:
     - RUS 55%, UKR 24%;
     - BLR, POL 6% each;
     - the Baltic states, MDA and the Caucasus 1-2% each.
   - **French army:** given small Algerian, Moroccan, Tunisian and West African shares. Together they fill 26k of the 35k NAF parent and 26k of the 30k WAF parent.
   - **Grande Armée 1812:** FRA 44%, DEU 20%, POL 15%, ITA 7%, plus small Dutch, Belgian, Swiss, Austrian, Croatian and Iberian shares.
   - **Russian army 1812:** RUS 72%, UKR 18%, with Baltic and Belarusian shares.
4. **Chronicle figures.**
   - **The Mongol cities.** Merv's 1.3M, Nishapur's 1.75M, Herat's 1.6-2.4M, Urgench's 1.2M and Baghdad's 0.8-2M are rejected. Each central value assumes a plausible city size plus refugees, about 100-250k. The chronicle number is quoted in the notes, and `deaths_high` is a plausible maximum, not the chronicle figure.
   - **Timur's tower counts.** Isfahan's 70k and Baghdad's 90k are treated as an order of magnitude; the central value sits below them.
   - **Other high claims kept only as `deaths_high`:** Yangzhou's 800k (high set to 300k), Josephus's 1.1M for Jerusalem (high set to Tacitus's 600k besieged), and Dio's 580k for Bar Kokhba.
5. **Chinese census-drop events.** The provincial rows share out the parent's deaths; they do not replace the census drop.
   - **Taiping:** split by Cao Shuji's provincial population losses, scaled to the parent's 20M. The five main provinces hold 16.1M, plus Nanjing.
   - **Dungan revolt:** Cao Shuji's Shaanxi 6.2M and Gansu 14.55M, scaled by 0.385.
   - **Mongol North China, Sichuan, An Lushan and Late Han:** shares are my assumptions, kept well below the parent (38-45%) and marked low confidence. Their main value is timing: in the parent rows the deaths are spread evenly over 50-90 years.
6. **Armenian genocide.** The 1M total is split by phase:
   - in-place massacres: 300k;
   - marches: 200k;
   - camps: 120k;
   - Deir ez-Zor: 150k.

   The split is my estimate after Kévorkian and Suny; only Deir ez-Zor has a near-direct figure, about 200k. The converge `from` is TUR:1. TUR sits in the same LEV vessel, so the film will draw the grains from all of Turkey, not only eastern Anatolia.
7. **Rejected numbers.**
   - **Romania 1917.** The often-quoted 350k soldiers and 450k civilians dead from typhus cannot be squared with about 500k cases. I used 150k (80-300k).
   - **Sarıkamış.** Larcher's 90k is used only as the high value; the central is 65k.
8. **Overlaps.** Within a parent, no two rows cover the same place and time. Province rows exclude the city rows inside them, and each such row says so in its notes:
   - Kaifeng is excluded from Mongol North China and from Ming Henan.
   - Merv, Nishapur and Herat are excluded from Khorasan.
   - Zaragoza is excluded from Spain.
9. **Two very wide local rows.**
   - **Great Terror executions:** 520k placed from the recorded 681,692. It is a local row with a radius of about 1,300 km, meant as a time pulse in 1937-38. Siberia's share, about 100k, stays in the SIB parent.
   - **Kolchak's collapse in Siberia:** radius 800 km.
10. **Dates.** Months are given wherever known. Years before 1000 are zero-padded: `0070-04`, `0184-03`, `0755-12`.

## Mismatches with the parent catalogue

1. **Canadian WWI dead, 1915-16.** About 3k at Second Ypres and 8k on the Somme are not placed, because the WWI NAM parent row runs 1917-1918 only. Canada lost about 20k men before 1917, so the NAM row should start in 1915.
2. **South Africans, Malawians and Mozambicans in WWI.** The WWI event has no KHS row. South African dead (7-12k) on the Somme and in East Africa, and the many Malawian and Mozambican carriers, are therefore missing. The same applies to carriers from the Belgian Congo: there is no CAH row.
3. **Spanish soldiers in Cuba.** 41-63k died of disease in 1895-98 (Tone; Thomas). They are not in `cuban_wars`, which has only a CAR row. An EUR row of about 50k is needed.
4. **Moroccan Regulares in the Spanish Civil War.** About 70-80k served and roughly 10k+ died. `spanish_civil_war` has no NAF row.
5. **Russian Civil War deaths in Belarus, and the Polish-Soviet War.**
   - `russian_civil_war` has no EUR row, so Belarusian pogroms and fighting there are not covered.
   - The Polish-Soviet War of 1919-21 is not covered either: about 47-60k Polish military dead and 16-18k Soviet prisoners who died in Polish camps.
6. **Stalinist repression outside Russia, Ukraine and Central Asia.** Great Terror executions in Belarus and Transcaucasia and Belarusian kulak deportees fall outside `stalin_repression_1930s`, which has only STP and SIB rows.
7. **Greek army dead, 1919-22.** The Greco-Turkish war of 1919-22, about 25-35k Greek army dead, is not in any parent. `ottoman_greek_assyrian_genocides` covers Ottoman Greek civilians only.
8. **Circassian expulsion after 1864.** The `caucasus_conquest` row ends in 1864, but deaths of Circassian exiles in the Ottoman ports continued into 1865.
9. **The French expedition to Saint-Domingue, 1802-03.** About 40-55k French soldiers and sailors died of yellow fever. They are inside the Napoleonic EUR row and could be a converge place. I left it out because the Haitian deaths, likely larger, are in no parent (the catalogue's notes list the Haitian Revolution as below threshold).
10. **Contested WWI Muslim deaths in eastern Anatolia.** Deaths among the Muslim population of eastern Anatolia in 1915-18 (refugees, typhus; McCarthy's figures are high and contested) sit inside the WWI LEV row. I did not place them.
