# e_lives_pre1925: whole-life fills for the 64 companies first seen before 1925

These files add rows only. Nothing in a_to1825, b_1825_1925, c_1925_1990 or d_1990_2026 was changed.

| file | rows | content |
|---|---|---|
| values.csv | 674 | new values for 35 worklist ids, nominal, millions of the source currency, arithmetic in `note` |
| events.csv | 30 | groupings, nationalisations, mergers, splits and takeovers that end or reshape these lives |
| companies.csv | 16 | full replacement rows only where the fate, end date or successor was wrong. Mostly 'alive' in the worklist although the company has since died; reason in each row's `notes` |

Rows whose id and month were already in an earlier era were dropped (20 rows), so nothing is double-counted.
Ids `lms` and `lner` in events.csv are the 1923 groups. They are not worklist companies.

## 1. Sources and method

1. **Investor's Monthly Manual (IMM), December issues, "Market prices in" table** (Yale ICF scans, https://depot.som.yale.edu/imm/).
   - Each December issue prints, for every security, the high and low of the four previous years and the closing price of the previous December.
   - Read from page images (300 dpi crops) for these issues:
     - 1900 (gives 1896-99)
     - 1913 (1909-12)
     - 1920 (1916-19)
     - 1925 (1921-24)
     - 1929 (1925-28; downloaded for this pass)
     - 1890 (Midland only, 1886-89)
   - `annual_avg` = capital x midpoint of the year's high and low; `year_end` = capital x closing price.
   - Capital (nominal stock, or share count) comes from the IMM main tables that era B read (1890, 1900, 1912, 1913, 1920) and from the 1925 main table (read here).
   - Between two capital readings the capital is interpolated; past the last reading it is held. The capital year is always written in the note.
   - Foreign shares quoted in London:
     - French railways and Suez: GBP per 500 fr (Nord: 400 fr) share.
     - US railroads and US Steel: London 'dollars' at the conventional $5/GBP. These are converted to New York equivalents with the year's average $/GBP divided by 5:
       - 0.9733 before 1914
       - 0.952 for 1916-18
       - 0.886 for 1919 and 1922
       - 0.770 for 1921
       - 0.916 for 1923
       - 0.884 for 1924
       - about 0.97 for 1925-28
     - December closes use the December rate.
2. **Yale ICF NYSE monthly prices 1815-1925** (Goetzmann-Ibbotson-Peng, rev. 2021).
   - Covers Union Pacific (common and preferred), Northern Pacific, Canadian Pacific, Illinois Central, Erie, NY Central and GE.
   - Value = December price x shares. Share counts are from IMM capital and are stated in each note.
   - GE's count (1,015,100 shares) is from Poor's Manual of Industrials 1916 ("issued and outstanding $101,510,000", unchanged since the 1912 stock dividend).
3. **The Bankers' Magazine (London), monthly table "Fluctuations in English and foreign stocks and railway shares"**.
   - archive.org `sim_bankers-insurance-managers-and-agents-magazine_YYYY-MM`.
   - Read from djvu OCR text; the Jan 1868 issue was also checked against the page image.
   - It fills the 1846-1868 railway hole for LNWR, Midland and the Nord (see section 3 for the capital problem).
4. **Capital in the 1840s**: Tuck, *The Railway Shareholder's Manual* (1847 ed., archive.org `railwaysharehol00tuckgoog`) gives LNWR, Midland and GWR capital accounts at 30 Jun 1846. The Bankers' Magazine of Jan 1849 gives a correspondent's LNWR capital analysis.
5. **companiesmarketcap.com** (fetched 12 Nov 2026), year-ends and latest values:
   - Union Pacific
   - Canadian Pacific (CP Ltd series before the Oct 2001 split)
   - BAT (from 1998 only)
   - Rio Tinto
   - US Steel (and Marathon Oil, to rebuild USX 1996-2001)
   - Imperial Brands (from 2012)
   - Coats Group (from 2015 only)
   - Navistar (International Harvester) 2001-20
6. **Shell 2014-2021**: shares from the Royal Dutch Shell 20-F cover pages (SEC EDGAR, A + B shares at each 31 Dec) x the RDS.A ADR year-end close (Yahoo; 1 ADR = 2 shares).
7. **Anchors**:
   - McGrattan & Prescott (2004), Table A2: CRSP values at end-Aug 1929 for UP, SP, Atchison, GN, NP, IC, CPR, Pullman, American Tobacco and International Harvester.
   - NYSE Fact Book 1983: Union Pacific at end-1982.
   - Hansard, 5 May 1958: Suez compensation.
   - Suez Lyonnaise des Eaux, Document de référence 1998: end-1997 and end-1998 market values.
   - FTC complaint and BP 20-F filings: Amoco and ARCO.
   - Press reports for SP, Santa Fe, IC, Navistar and De Beers.
   - Finaeon/GFD "350 Years of English Leadership" for Imperial Tobacco 1930/1940/1950.

## 2. What is now covered (new rows)

| company | added |
|---|---|
| LNWR | 1847-1867 (16 dates, Bankers' Mag.); 1896-99; 1909-11; 1916-19 |
| Midland | 1847-1867 (15 dates); 1886-89; 1897-99; 1916-19 |
| GWR | 1896-99; 1909-11; 1916-19; 1921-24 (still no 1846-72 and no 1925-47) |
| NER, L&Y | 1896-99; 1916-19 (NER), 1909-12 and 1916-19 (L&Y) |
| Bank of England | 1896-99; 1909-11; 1916-19; 1921-28 (gap now 1929-1945; nationalisation value is in era C) |
| Nord | Dec 1853-1867 (8 dates); 1896-99 |
| PLM, Est, Südbahn | 1896-99 (Est also 1909-12) |
| Suez | 1909-12; 1916-19; 1921-23; 1958 compensation; 1997-98 (Suez Lyonnaise des Eaux) |
| Union Pacific | every Dec 1899-1925 (common + preferred from 1907); Aug 1929; 1982; 1996-2026 |
| Canadian Pacific | every Dec 1885-1925; Aug 1929; CP Ltd 1997-2001; 2001-2026 |
| Northern Pacific | 1911-1925; 1929 |
| Illinois Central | 1881-1903; 1929; 1998 |
| Southern Pacific | 1916-28; 1929; 1996 |
| Great Northern (US) | 1916-28; 1929 |
| Atchison | 1921-28 (common only = est_low); 1929; 1995 |
| NY Central | 1881-91; 1910-13 |
| Pennsylvania RR | 1925-28 |
| US Steel | 1909-12 (common only); 1921-24 (common + pref); USX 1996-2001; steel company 2002-2024; 2025 takeover |
| Royal Dutch/Shell (group) | 1921-24 (Royal Dutch / 0.6, low confidence); 2014-2021 rebuilt from 20-F share counts |
| BAT, Imperial Tobacco, Rio Tinto | 1921-24 (Rio also 1909-12); to today from companiesmarketcap (Imperial 2012-, BAT 1998-, Rio 1996-); Imperial 1930/40/50 (GFD) |
| GE | Dec 1913 and Dec 1915 |
| Amoco (SO Indiana), ARCO (Atlantic Refining) | end values 1998 and 1999-2000 |
| International Harvester | 1929; Navistar 2001-2020; 2021 takeover |
| De Beers, Guinness, Pullman, American Tobacco | one or two anchors each |

## 3. Doubts, ranges and decisions for a human

1. **Railway capital 1846-1868 is interpolated.**
   - Prices are solid; capital is not.
   - LNWR runs straight-line from GBP10.7m paid up (Jun 1846) to about GBP15m productive ordinary capital (end-1848) to GBP29.9m (Dec 1869). Midland runs from GBP5.16m (1846 consolidated stocks) to GBP11.36m (1869).
   - Rows from 1849 are `est_low`/`est_high` at ±10% around the straight line. Partly paid 1840s shares are counted pro rata.
   - GWR, NER (with its York, Newcastle & Berwick and York & North Midland predecessors) and L&Y prices are in the Bankers' Magazine but were not valued, because their 1846-69 capital was not found. The Bankers' Magazine also prints the 1849 crash: GWR fell from 104-106 (Feb) to 48-50 (Oct), and York & North Midland from 58-60 to 16½-17.
2. **Capital held past its last reading.** L&Y 1909-19 uses the 1900 count. Atchison, Southern Pacific, Great Northern and Pennsylvania 1926-28 use the 1925 counts. All of these companies issued stock in the late 1920s, so those values are probably too low: Atchison has 2.0m shares in 1928, but about 2.4m are implied in 1929.
3. **US Steel and the USX split.**
   - From 2002 the rows follow the spun-off steel company: $1.6bn at the split, sold to Nippon Steel in 2025.
   - The legal heir of the 1901 corporation is Marathon Oil ($9.3bn at the split; bought by ConocoPhillips in 2024).
   - Which branch the island follows is a human decision.
4. **Canadian Pacific 2001.** The rows follow the railway: CP Ltd was $12.6bn in Aug 2001, and CPR alone was $3.1bn in Dec 2001. The legal parent became Fairmont Hotels.
5. **Imperial Tobacco and BAT "to today".**
   - Imperial Group was taken over by Hanson in 1986. The 1996 demerged Imperial Tobacco Group plc is a new company, but its values are under the same id as asked.
   - There are no values for 1987-2011, so the island would interpolate across Hanson's ownership.
   - BAT's 1998 holding company likewise replaces BAT Industries. There are no values for 1926-1997.
6. **Royal Dutch 1921-24.** These rows use the London-list count of 2,138,180 shares (era B already doubted it) and divide by 0.6 for the whole group, because they are written under `royal_dutch_shell` directly.
7. **Shell 2014-21.** companiesmarketcap counts only the A shares (about 55-60% of the group). The new rows use A + B shares, so the end-2017 value is $277bn rather than CMC's $153bn. The B shares are valued at the A price (they traded within about 1-3%).
8. **Rio Tinto.** CMC's series jumps 30% in a week in Jan 2026; the 2026 peak row is marked as unverified.
9. **Suez.**
   - The 1958 compensation (GBP(E)28.3m, about $81m at $2.8716 per Egyptian pound) is a floor, because the company kept its assets outside Egypt.
   - The 1958 EGP row and the 1997-98 FRF rows cannot be converted by money.py: it has no franc rate after 1914 and no EGP rate. USD rows at FRED's Dec-average franc rates are given next to them.
   - The London Suez quotes for 1918 (190/190) and 1924 (blank) look odd and are reproduced as printed.
10. **Guinness 1997** is an `est_low`/`est_high` pair at 45-50% of the $36bn combined Guinness + Grand Met value. The exact split was not verified.
11. **De Beers 2011** is a private sale between shareholders ($5.1bn for 40% = $12.75bn), not a market price.
12. **Successor companies.** For Southern Pacific (1988-96), Illinois Central (1989-98) and Santa Fe (1968-95), the late values are those of holding or successor companies, not the original legal railway.
13. **Myth check: GE.** GE ($142m in Dec 1913, $177m in Dec 1915) was smaller than Union Pacific, Canadian Pacific or the Pennsylvania, and far below US Steel.

## 4. Still missing (in priority order)

1. **Standard Oil (NJ) 1890-1906 and 1913-1925; Socony, SO Indiana, SO California and Atlantic 1912-1925.** These traded on the curb or over the counter and are not in the Yale NYSE file or the IMM. Poor's 1916 gives capital but not prices. Source: Commercial & Financial Chronicle curb tables; FRASER has the CFC only to 1869 online and Bank & Quotation Record from 1928.
2. **AT&T before 1922.** No share counts found. Time and the FCC report give 1922-25 only. Source: AT&T annual reports 1900-1921.
3. **GE 1914 and 1916-1925.** The share count after 1915 is unknown (stock dividends). The prices are in the Yale file.
4. **US railroads 1930-1995.** There is one value in Aug 1929 and then nothing until the 1990s mergers (UP: 1982 only). This also covers NY Central and the Pennsylvania 1930-68 (not in the 1926-90 gap-fill) and Anaconda 1930-77. The B&QR January texts already in the scratchpad hold the December prices; share counts are needed.
5. **French railways and the Banque de France.**
   - Railways: no values 1846-52, 1868-95, 1901-1937 (PLM, Orléans, Est; Nord 1868-95 apart from era B's points) or 1938 (SNCF terms). The IMM 1907, 1918 and 1929 issues have them; the 1904, 1918 and 1929 PDFs are already downloaded.
   - Banque de France: still only era B's scattered points.
6. **East India Company 1721-1765 and 1798-1825; South Sea 1721-1846; Compagnie des Indes 1724-1769; VOC 1699-1718.** Untouched. The Gentleman's Magazine monthly stock tables (1731 onward, archive.org) would fill the EIC and South Sea for a few images per decade.
7. **Further gaps:**
   - BoE 1929-45
   - GWR 1925-47
   - Imperial 1925-86 (apart from GFD 1930/40/50) and 1987-2011
   - BAT 1926-97
   - Coats 1926-2015
   - Rio Tinto 1925-95
   - De Beers 1913-2001
   - Guinness 1913-96
   - Pullman, Singer, American Tobacco and International Harvester 1913-1990s
   - Hudson's Bay after 1880
   - Krupp: never listed alone
   - Indian railways
   - Erie after 1882
8. **Unused IMM years.** Issues 1869, 1873, 1876, 1880, 1885, 1895, 1904, 1907 and 1918 would close the remaining 1866-1915 gaps for British and French railways, Suez and BoE (1874-95, 1900-08, 1913-15). The PDFs for 1876, 1904, 1918 and 1929 are in the scratchpad (e_downloads/imm).

## 5. Story beats

- **The Railway Mania crash was severe.**
  - Midland stock went from 113 (Oct 1847) to 42 (Oct 1849); its value fell by about 60%.
  - LNWR fell from 157 (Oct 1847) to 101 (Nov 1848), recovered to 143 (Feb 1849), and dropped to 104 (Oct 1849).
  - LNWR was worth about GBP15-35m in 1847-67. The Bank of England (GBP28-36m) stayed bigger until the mid-1860s, which fits GFD's "LNWR passed the Bank about 1865".
- **The First World War shrank British blue chips by half in real terms.** Bank of England stock fell from 279 (1909) to 164 (1921); LNWR from 151 (1911) to 86 (1919).
- **Union Pacific was a Harriman-era giant.** It was worth $520m in Dec 1909 (common + preferred), more than the Pennsylvania RR's $504m of Dec 1912, and it is still alive today ($180bn).
- **Splits moved the value away from the famous names.**
  - USX 2001: Marathon kept about 85%.
  - CP Ltd 2001: the railway kept about 25%.
- **Shell 2014-2021 was about 1.6-1.8x larger than the CMC-based picture**, peaking at $277bn at end-2017.
- **The Suez Canal Company outlived its canal by 52 years.** It was compensated for $81m in 1958, then worth $30bn as Suez Lyonnaise des Eaux in 1998, and was absorbed into GDF Suez (Engie) in 2008.
