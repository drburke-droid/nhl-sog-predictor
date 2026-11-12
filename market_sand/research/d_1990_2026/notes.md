# Era d: 1990 to 2026-11-12. Research notes

All values are **nominal US dollars, in millions**, unless the currency column says JPY. No inflation adjustment has been applied. "bn" in these notes means USD billions.

Files: `companies.csv` (82 companies), `values.csv` (about 13,200 rows), `events.csv` (50 events), `top10.csv` (a year-end top 10 for every year 1990 to 2025, plus 2026-11-12). The build script and all downloaded pages are kept outside the repo, in the scratchpad (`d_scripts/build.py`, `d_1990_2026_downloads/`).

## 1. Method by period

| Period | Main source | How the numbers were made | Confidence |
|---|---|---|---|
| 1990-1995 | Three anchor lists: Japan top 20 at 1989-12-29 (JPY), world top 50 at 1992-12-31 (USD), world top 10 at end-1995 (USD) | Values between anchors are **derived**: implied share count (anchor value / anchor price) x year-end price. Prices: Yahoo monthly closes (US), kabutan yearly closes (NTT, Toyota, TEPCO, Sumitomo Bank). JPY converted at Fed H.10 year-end rates. Each derived row shows its arithmetic in `note`. | medium for US, NTT, Toyota and Royal Dutch/Shell; **low** for Japanese banks other than Sumitomo |
| 1996-2005 | companiesmarketcap.com (CMC) year-end values (daily close x shares) | Used directly, after excluding CMC series that are clearly wrong (list in section 4). Added: NTT 1996-1999 (kabutan), Royal Dutch/Shell 1996-2001 (derived), AT&T Corp (10-K). 2005 top 10 from the Cabinet Office list. | medium-high |
| 2006-2025 | Wikipedia quarterly top-10 lists (FT Global 500 to 2016; YCharts/CMC after) | Q4 list used as the year-end top 10, with corrections: Saudi Aramco added from 2019 (Wikipedia excludes it for low free float); Google 2009-2014 and Facebook 2015-2016 replaced by full values including unlisted Class B shares, which the FT left out. | high |
| 2026-11-12 | CMC global ranking page and company pages | Latest value of each company. | high (as of that day) |

Monthly values: CMC month-end observations (weekly data from 2020, so "month-end" can be the last weekly print of the month; the note says so) for Jan 1999-Dec 2001, Jan 2007-Dec 2009, 2020, and Jan 2022-Nov 2026. Quarter-end CMC values are given for every year. Wikipedia quarterly values (2006 Q1 to 2026 Q3) are separate `point` rows. Peaks: the maximum of each CMC series (`peak`), plus daily nominal peaks from the Wikipedia trillion-dollar table where available; `trough` rows mark falls below half of the peak.

## 2. Sources (full references)

- companiesmarketcap.com, one page per company: `https://companiesmarketcap.com/<slug>/marketcap/` and the ranking page `https://companiesmarketcap.com/`, retrieved 2026-11-12. The embedded chart data give month-end values to 2019 and weekly values from 2020.
- Wikipedia, "List of public corporations by market capitalization", retrieved 2026-11-12. https://en.wikipedia.org/wiki/List_of_public_corporations_by_market_capitalization (quarterly top 10s 2006-2026; FT Global 500 lists for 2000-2005; trillion-dollar table).
- Wikipedia, "List of largest mergers and acquisitions", retrieved 2026-11-12 (deal values). Also the Wikipedia articles on AT&T, AT&T Corporation, Time Warner, Mannesmann, Vodafone, Alphabet Inc., Meta Platforms, Saudi Aramco, SpaceX, Lucent Technologies, GE Vernova, GE HealthCare, Altria, ExxonMobil.
- finance-gfp.com ("Financial Star"), Japanese tables (images transcribed by hand): world top 50 at 1992-12-31 (https://finance-gfp.com/?p=10552); world top 50 at 2000-03-31 and 2007-10-31 (https://finance-gfp.com/?p=3680); Japan top 20 at 1989-12-29 and 2000-02-29 (https://finance-gfp.com/?p=15169). The underlying database is not named.
- Cabinet Office of Japan, Council for Science, Technology and Innovation, reference material sanko3-6.pdf, p.70, "時価総額ランキングの変遷" (world top 10 for 1995, 2005, 2015), citing Mitsubishi UFJ Morgan Stanley Securities, "マーケットの歴史". https://www8.cao.go.jp/cstp/tyousakai/kihon6/6kai/sanko3-6.pdf
- america-kabu.com, "世界の時価総額ランキング、過去からの推移" (2018-05-11): world top 10 for end-1997, end-2007 and end-2017 (used only as a cross-check).
- kabutan.jp yearly price tables (split-adjusted closes) for codes 7203 Toyota, 9432 NTT, 9501 TEPCO, 8316 SMFG (history spliced from Sumitomo Bank).
- Yahoo Finance chart API, monthly split-adjusted closes, 1989-1996 for US tickers; SHEL history before 2005 is Royal Dutch Petroleum's New York shares (4-for-1 split June 1997); GOOGL 2004-2014.
- Board of Governors of the Federal Reserve, H.10 historical rates, year-end JPY per USD (1989 143.80, 1990 135.75, 1991 124.90, 1992 124.85, 1993 111.70, 1994 99.60, 1995 103.28, 1996 115.77, 1997 130.45, 1998 113.08, 1999 102.16, 2000 114.35).
- SEC EDGAR: AT&T Corp 10-K filings 1994-2005 (CIK 5907; cover-page market values, quarter-end closes, shares outstanding); Mobil Corp 10-Ks 1995-1999 (CIK 67182); Google Inc 10-K cover pages 2005-2014 (CIK 1288776; Class A and Class B share counts).
- CNBC, Paul Steiger, "What a difference 25 years makes", 2014-04-29 (MSCI 1989 list; NTT is not in it).
- Fortune, 1991-08-26, "Shell gets rich by beating risk" ("Shell ... market capitalization $75 billion").
- CNN Money 2000-01-10 "That's AOL folks"; CBS News "AOL, Time Warner to merge"; Fortune 2002-02-04 "AOL Time Warner's new math"; WSWS 2002-04-29. These were read through search-result summaries, not full pages; treat them as medium confidence.

## 3. Top 3 at the requested dates (USD bn)

| Date | 1 | 2 | 3 |
|---|---|---|---|
| end-1990 | NTT 113 (derived) | Royal Dutch/Shell 71 (derived) | IBM 64.4 (Exxon 64.3 is level) |
| end-1995 | NTT 128 | Royal Dutch/Shell ~127 (derived, not in the source list) | GE 120 |
| 2000 peak (31 Mar 2000) | Microsoft 533 | Cisco 533 | GE 511 |
| end-2002 | Microsoft 277 | GE 243 | ExxonMobil 234 |
| end-2007 | PetroChina 724 | ExxonMobil 512 | GE 375 |
| end-2009 | PetroChina 353 | ExxonMobil 324 | Microsoft 271 |
| end-2015 | Apple 598 | Alphabet 534 | Microsoft 440 |
| end-2020 | Apple 2,254 | Saudi Aramco 2,053 | Microsoft 1,682 |
| end-2024 | Apple 3,785 | Nvidia 3,289 | Microsoft 3,134 |
| 2026-11-12 | Apple 5,168 | Nvidia 4,195 | Microsoft 3,869 |

Note on the 2000 peak: at the end of 1999 Microsoft was 604, GE 510 and NTT DoCoMo 367. Cisco's month-end peak was 536 (31 Mar 2000), DoCoMo's daily peak 401 (30 Mar 2000), and GE's month-end peak 581 (31 Aug 2000).

## 4. Known problems in the sources, and what was done

**companiesmarketcap.com.** It is reliable for US companies, but some foreign series are wrong:
- Not used: Nestle 1998-2002 (about 3x too high); BP 1996; Shell 2005-2021 (about half of the real group value; it seems to count only some shares); Vodafone 1996-1999 (about half; it misses the AirTouch shares); ICBC 2006-2019 (1.4-1.6x the FT values); Roche 2012; BHP 2009; Mizuho 2008 (zero); Time Warner 2001.
- Flagged and kept: HSBC 2003-2006 (about 15% above FT; HSBC is left out of the 2003-2004 rankings); Toyota 2006-2007 (about 10% below FT); Samsung (common shares only); Bank of America 2005 (post-MBNA share count, so the CAO value is used).
- Around large stock mergers, CMC applies the next reported share count to earlier prices, which can inflate pre-merger values.
- CMC tracks ADRs for TSMC, Tencent and others. Wikipedia gives TSMC 565bn at end-2020 against CMC's 488bn.

**Wikipedia list page.**
- The "2000" to "2003" FT lists are misdated. The 2000 and 2001 lists match prices of early January 2000 and early January 2001 (Wikipedia says 31 March). The "2002" and "2003" lists match end-March 2002 and end-March 2003 (Wikipedia says December). They are stored with corrected, approximate dates.
- The **2008 Q2 and Q3 columns are copies of 2009 Q2 and Q3** (identical numbers) and are not used.
- The FT lists to 2016 exclude unlisted share classes (Google Class B, Facebook Class B). That is why Facebook is missing from the Q4 2015 and Q4 2016 lists although its full value ranked about #7-8, and why Google's full value puts it in the year-end top 10 in 2009, 2011 and 2012.
- The lists exclude companies with less than 15% free float, so Saudi Aramco is never shown.

**finance-gfp 1992 list.** US values check out exactly against price x shares (Exxon, Walmart, GE, Philip Morris, AT&T, IBM, Microsoft, Pfizer). It contains implausible entries that were **not used**: "Paribas" 54.5bn at #8 (French total market value was then about $350bn, so a $54bn Paribas is not credible), RELX 30.7bn, "BT Corp North America" 24.2bn and "TIS NV" 20.9bn. It omits TEPCO, which we derive at about 28bn, so it should rank about #11 in Japan. Royal Dutch Petroleum (43.6) and Shell Transport (28.2) are listed separately; we sum them as 71.8.

**Cabinet Office 1995 list.** Its #1 is "エスコム" (Eskom) at 135.7bn. Eskom is a South African state utility that is not listed. The list has no Royal Dutch/Shell, which by our derivation was about 127bn. We treat the Eskom row as an error, possibly mislabelled Royal Dutch/Shell (stored only as `est_high` 135.7 for royal_dutch_shell). The list also has no Japanese bank. Our proxy estimate for Mitsubishi Bank at end-1995 (78bn) would rank #9, so that slot is uncertain.

**america-kabu 1997 list** has an implausible "Aeroflex 124.1bn" and a pro-forma "Citigroup 123.9bn" for 1997 (before the merger). It is used only to cross-check NTT (136.3, against our 136.6) and Royal Dutch (117.8).

## 5. Ranges and doubts (human decisions)

1. **Japanese banks 1990-1995.** No price data were found for IBJ, Fuji, DKB, Mitsubishi or Sanwa (kabutan blocks the codes of delisted companies; Yahoo's Japanese data start in 2000). The 1990, 1991 and 1993-1995 values are **estimates**: the bank is assumed to move like Sumitomo Bank (real data). est_low and est_high come from the 1989 and the 1992 anchor. The ranges are wide; for example IBJ end-1990 is 47-64bn. In top10.csv the midpoint is used, and the source column says ESTIMATE. **Ranks 6-10 for 1990, 1991, 1993 and 1994 are uncertain.**
2. **Royal Dutch/Shell (one vessel).** Before 2002 the group value = Royal Dutch Petroleum value / 0.6 (Royal Dutch had a 60% interest). The check at end-1992 is 72.7 against the actual sum of 71.8. Royal Dutch Petroleum and Shell Transport were separate listed companies until 2005. If era c keeps them apart, they need splitting (about 60/40) or a merge event in 1990.
3. **Merger vessels.** Several legal survivors are not the obvious vessel: Travelers was legally renamed Citigroup (we show citicorp + travelers pouring into a new citigroup); Sakura Bank was the legal survivor of the 2001 merger with Sumitomo Bank (we keep sumitomo_bank as the SMFG vessel); AOL Time Warner was a new holding company (new vessel aol_time_warner). The Mitsubishi Bank lineage becomes MUFG; Sanwa becomes UFJ and then MUFG.
4. **Saudi Aramco** (1.5% float) and **SpaceX** (IPO 2026-06-12) are included in top10.csv, unlike Wikipedia for Aramco. Decide whether a mostly state-owned company with a tiny float belongs in the film as a full vessel.
5. **PetroChina "$1 trillion" (5 Nov 2007)** is a paper value: the first-day A-share price applied to all shares, while the Hong Kong H shares traded far lower. The FT year-end 2007 value is 724bn. Peak row: 1,200bn nominal (Wikipedia).
6. **AT&T Corp tracking stocks** (AT&T Wireless Group, Liberty Media Group, 1999-2001). The year-end 2000 value of 65.7bn counts AT&T common stock only; the tracking stocks were worth about 42bn more.
7. **ExxonMobil Holdings.** Wikipedia now names the parent "ExxonMobil Holdings Corporation". The date of this reorganisation was not established; it is treated as a rename of standard_oil_nj.
8. **Health insurer industry.** UnitedHealth is classed as consumer because the industry list has no health-services class.
9. **Nortel** (peak 252bn on 2000-07-26) may have been briefly in the top 10 in July-August 2000. It is not on any quarter-end list we hold, but it is included as a bubble husk (bankrupt 2009).

## 6. Gaps

- AT&T Corp: no year-end values for 1990-1991 (only MSCI 1989: 49.0bn) and 1998-1999 (only February values: 143.5bn in Feb 1999 and 157bn in Feb 2000). The SEC filings for those years do not carry the price table.
- Mobil: only 1992 year-end (25.2bn), 10-K cover values 1995-1999, and the Exxon deal value (77.2bn). 1990-1991 are missing.
- AOL and the old Time Warner: only the January 2000 announcement values and the 2000-03-31 list (CMC gives Time Warner 1997-2000). AOL Time Warner 2001-2017 has only scattered points (Feb 2002 under 135bn, Apr 2002 about 85bn, 2016 deal 85.4bn). Yahoo has no TWX history.
- Citicorp: only its deal value (70bn). Travelers: CMC 1996-1998. JPMorgan, Bank of America and Wells Fargo: no values before 1996 (their predecessor banks were not researched).
- No 1990 values for HSBC, Nestle (1992: 30.8bn), Roche (1995: 77.8bn), Nokia, Samsung (CMC starts 2007), Petrobras or BHP.
- Gazprom: FT 2006-2008, CMC from 2010.
- Intramonth peaks before 2020 are month-end values. Daily peaks exist only for some companies (Wikipedia trillion table; CMC daily data for DoCoMo, Lucent and Nortel).
- 1990-1995 US values outside the anchor years assume a smooth share count. Intel, Cisco, Apple, Berkshire, Lilly, UnitedHealth, SBC, BP and Vodafone assume a constant share count back from 1996/1997 (low confidence). Berkshire check: 1,152,547 shares x $6,675 = 7.69bn at end-1990, against our 8.2bn.

## 7. Myths and pitfalls

- "In 1990, seven of the world's top ten companies were Japanese" usually refers to the **1989** Business Week list (valued around mid-1989). By the end of 1990 the Nikkei had fallen 39%. On our estimates NTT was still #1 (113bn), with five Japanese companies in the top 10 on midpoint estimates (NTT, IBJ, Mitsubishi Bank, Sumitomo Bank, Toyota); the bank ranks are uncertain.
- "NTT became the first $100bn company in 1990" (Global Financial Data) is wrong. NTT was worth 160bn at the end of 1989 (JPY 22.9tn) and more in 1987.
- Global Financial Data says Exxon was the largest company in 1989 and 1991-1992, and GE from 1993 to 1997. Its lists appear to drop NTT after 1990. Our derivation has NTT ahead at end-1991 (92bn against Exxon's 76bn), and the Cabinet Office list has NTT ahead of GE at end-1995 (128.4 against 120.3). Exxon was #1 at end-1992 (75.9 against NTT 71.3).
- "IBJ was the most valuable company in 1989" is true only in lists without NTT (the MSCI list leaves NTT out).
- "AOL bought Time Warner for $350bn": $350bn was the combined value at announcement. AOL itself was worth about 163-164bn; the deal value is variously reported as 162-183bn.
- "PetroChina was the first trillion-dollar company": see doubt 5. Apple was the first to close above $1tn on a full-share basis (2 August 2018).
- "Microsoft $586bn in 2000" is the FT figure from early January 2000. The year-end 1999 value was 604bn.
- Wikipedia's 2017-2026 quarterly values (YCharts/CMC) and CMC year-ends differ by 1-3% (share-count timing); both are kept as rows.

## 8. Possible story beats

- **The Japanese deflation.** IBJ, Sumitomo, Fuji and DKB were four of the world's five most valuable companies in the 1989 MSCI list. The three banks that became Mizuho were worth JPY 34.2tn (about $238bn) at the end of 1989. In 2000 they merged into Mizuho, worth 83.7bn in 2006 and 32bn in 2020.
- **NTT's long glide.** NTT was worth 160bn (1989), 71bn (1992), 138bn (1994) and 272bn (end-1999, with the dot-com boom). Its child DoCoMo outgrew it (401bn, March 2000), and NTT took DoCoMo back in 2020.
- **The dot-com husks.** Lucent 259bn (Dec 1999) fell to 2.0bn (Oct 2002). Nortel 252bn (Jul 2000) went bankrupt. Cisco fell from 536bn to 76bn. Nokia fell from 267bn to 7.5bn (2012). AOL at 163bn bought a 100-year-old media group and was spun out at about 2.5bn in 2009.
- **2008-09.** Citigroup fell from 350bn (Aug 2000) to 8.5bn (Feb 2009). AIG fell from 230bn to 1.1bn (Feb 2009).
- **GE's arc.** GE was #1 at seven year-ends between 1996 and 2005 and peaked at 581bn (Aug 2000). It fell to 53bn (Sep 2020) and split into three companies (2023-2024).
- **AT&T eats its parent.** Southwestern Bell, a 1984 Baby Bell, bought AT&T Corp in 2005 for 16bn. AT&T Corp had been worth 157bn in 2000 and the world's #4 at the end of 1995 (103bn).
- **TEPCO and Fukushima.** TEPCO, a world top-10 company in 1989 (MSCI: 56bn), fell from 33bn at end-2010 to 3.8bn at end-2011.
- **Saudi Aramco** debuted at #1 in December 2019 (1.88tn) and had slipped to #10 by the end of 2025 (1.53tn).
- **Apple:** 5bn at end-2002, 5,168bn on 11 Nov 2026 (record), and #1 again after Nvidia fell from 5.65tn (3 Oct 2026, CMC) to 4.20tn (12 Nov 2026).
- **2026 surprises in the data (from CMC and Wikipedia; not checked elsewhere):** SpaceX listed on 12 June 2026 and peaked at 2.44tn a week later. A memory-chip boom took Samsung, SK Hynix and Micron above 1tn. Intel went from 188bn (January) to 673bn (June) and back to about 330bn (November).
- **Continuity.** Exxon was in our year-end top 10 every year from 1990 to 2016 (and again in 2022). Microsoft has been in it every year-end from 1996 to 2026 (31 straight). GE was in it every year from 1990 to 2008. Year-end #1s: NTT (1990-91, 1993-95), Exxon (1992, 2006, 2008, 2010-11), GE (1996-97, 2000-01, 2003-05), Microsoft (1998-99, 2002, 2018), PetroChina (2007, 2009), Apple (2012-17, 2020-24, Nov 2026), Aramco (2019), Nvidia (2025).
