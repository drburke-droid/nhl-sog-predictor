# f: whole lives of the post-1925 companies (afterlives, early years, gaps): notes

This folder adds rows only. Nothing in eras a-d was edited. Values are nominal and in millions of the stated
currency (USD, or JPY for the Japanese derivations). `values.csv` has 1,438 rows for 79 ids. `events.csv` has 41
deals, bankruptcies, renames and listings. `companies.csv` has 5 new companies and 17 corrected fates or
successors. Every derived row shows its arithmetic in `note`. The build script and all downloads are in the
scratchpad (`f_scripts/build_f.py`, `f_downloads/`), not in the repo.

I checked the merge with a copy of `build_data.py` writing to the scratchpad: 181 companies and no skipped
rows. The repo's `data/` was not touched.

## 1. What is now covered

Afterlives (from the last value to the end, or to 2026-11-12):

| id | rows | end |
|---|---|---|
| gm | 1991-94 NYSE year-ends; 10-K points 1994-2008 | bankrupt 2009-06-01, $457m on the last close. New GM is a new vessel, **gm_2010**, from the 2010-11-18 re-IPO (~$49.5bn at $33) to 2026 (CMC). |
| eastman_kodak | 1991-94 NYSE; 10-K points 1994-2011 | Chapter 11 2012-01-19 at $114-128m. The post-2013 Kodak is not a vessel. |
| sears | 1991-94 NYSE; Sears Roebuck 10-K points 1995-2004; Sears Holdings 2005-2017 (lower bounds) | Chapter 11 2018-10-15, $32m. |
| texaco | 1991-94 NYSE, 10-K 1994-97, CMC 1997-2001 | merged into Chevron (standard_oil_ca) 2001-10-09 at $38.4bn |
| dupont, dow_chemical | dupont: NYSE 1991-94, 10-K points 1994-2017. dow_chemical: 10-K year-ends 1995-2016, plus 1972-78 and 1986 to fill its gaps | merged into the new **dowdupont** 2017-08-31 (about $75bn each side at the merger) |
| dowdupont, dow_inc, corteva | CMC 2017-2026 | dowdupont spun off Dow Inc. (2019-04-01) and Corteva (2019-06-01) and was renamed DuPont de Nemours |
| union_carbide | 10-K 1995-2000 | Dow, 2001-02-06: ~$7.0bn at completion, $9.3bn at announcement |
| phillips_petroleum | NYSE 1981-84, derived 1985-95, CMC 1996-2002 | the new **conocophillips** (2002-08-30; CMC 2002-2026) |
| Bells | ameritech, nynex, pacific_telesis, bellsouth, us_west to their ends (SBC 1997/1999, Bell Atlantic 1997, AT&T Inc. 2006, Qwest 2000); bell_atlantic (Verizon) to 2026 | |
| mmm, xerox, schlumberger, ford, bristol_myers | to 2026 (NYSE 1991-94, derived 1995, CMC 1996-2026) | |
| con_ed, united_gas_improvement | CMC 1996-2026; upper-bound rows 1985-95 | |
| woolworth (Foot Locker) | CMC 1996-2025 | Dick's Sporting Goods, Sept 2025 |
| nomura, nippon_steel, matsushita, kansai_electric, hitachi | kabutan-derived year-ends 1990-1999, then CMC to 2026 | |
| tokai_bank | proxy estimates 1990-2000 | UFJ, i.e. sanwa_bank, 2001 |
| mitsui_bank (Sakura Bank) | proxy estimates 1990-2000 | sumitomo_bank (SMBC), 2001 |
| aol_time_warner | 10-K points 2002-2017 | |
| time_warner | NYSE 1993 and 10-K 1994-96 | |
| aol | 10-K points 1996-99 | |

Before the first value (growing in):

| id | rows |
|---|---|
| apple | IPO 1980 and 1980-92 (NASDAQ Fact Books) |
| microsoft | IPO 1986 and 1986-89 |
| intel | 1979-92 |
| cisco | IPO 1990 and 1992 |
| walmart | IPO 1970 and 1972-94 |
| berkshire | 1979-94 |
| eli_lilly, pfizer | 1972-94 |
| aig | 1973-94 |
| unitedhealth | 1984-89 |
| johnson_johnson | 1962-78 |
| ford | IPO 1956 and 1972-86 |
| bristol_myers | 1972-88 |
| jpmorgan (Chemical), bank_of_america (NationsBank), travelers | 1992/93-96 |
| wells_fargo (Norwest) | 1985-96 |
| citicorp | NYSE 1979 and 1982-85; 10-K Jan 1997 |
| rjr_nabisco (R.J. Reynolds Industries) | 1979-85 |
| getty_oil | 1981-82 |
| Japanese companies | 1970-1988 as upper bounds (see below) |
| tencent | IPO 2004-2011 |
| petrochina | 2000-03 |
| sinopec | 2000-01 |
| china_construction_bank | 2005-07 |
| samsung | 2000-06 |
| novartis | 1996-97 |
| nokia | 1994-95 |
| bhp | 1980-95 |
| vodafone | 1988-89 |
| deutsche_telekom | IPO month 1996 |
| mizuho | 2000-05 |
| google, amazon, visa, facebook | IPO-month points |

Gaps filled: dow_chemical 1972-78 and 1981-86; phillips_petroleum 1981-95; shell_oil_us not done (see section 5);
icbc 2015-19; tepco 1996-2005; sanwa_bank 1996-2004; ibj, fuji_bank and dai_ichi_kangyo 1996-2000;
sumitomo_bank 2001-02; nestle 1996-2002; roche 2003-05 and 2012; alcatel_lucent (Alcatel) 1997-2005;
anglo_persian (BP) 1987-89.

## 2. Methods and sources

1. **NYSE Fact Books, "Fifty leading stocks in market value"** (listed shares x 31 Dec price; includes treasury
   stock). Values were read by eye from page images.
   - Year-ends 1979 and 1981-1985 (the era-c page images, for companies era c had not transcribed).
   - 1990 (archive.org `micro_IA40706937_0391`).
   - 1991 (`micro_IA40706941_0042`, p.38), 1992 (`micro_IA40706944_0305`, p.35), and 1993 and 1994
     (`micro_IA40706951_0126`, pp.39 and 44). These are new for 1991-1994 and give GM, Kodak, Sears, Texaco,
     du Pont, Dow, the Bells, Ford, BMS, 3M, Schlumberger, Pfizer, Lilly, AIG, Walmart, Time Warner (1993) and
     Berkshire (1994).
2. **NASDAQ Fact Books, "50 market-value leaders"** (total shares outstanding x closing bid). Values were read by
   tesseract OCR and checked by eye.
   - Year-ends 1979 (`micro_IA40706901_0181`), 1982 (`micro_IA40706911_0059`), 1983 (`micro_IA40706914_0408`),
     1984 (`micro_IA40706917_0426`), 1985 (`micro_IA40706920_0415`), 1986 (`micro_IA40706924_0026`), 1987
     (`micro_IA40706927_0155`), 1988 (`micro_IA4070630_0329`), 1989 (`micro_IA40706934_0051`) and 1992
     (`micro_IA40706944_0143`).
   - These cover Apple, Intel, Microsoft, Berkshire (Nasdaq until 1988), AIG (Nasdaq until 1984) and Cisco.
   - Examples: Intel 1979 $1.37bn; Apple 1982 $1.72bn; Microsoft 1986 $1.24bn; Berkshire 1985 $2.83bn.
3. **Yahoo Finance chart API** (split-adjusted closes: monthly from 1985, and daily turned into month-ends before
   1985). Value = price x the share count implied at the anchors (anchor value / anchor price), interpolated
   between anchors.
   - Before the first anchor the count is held constant. The note says so, and the value is an upper bound if
     shares were issued.
   - The same method was used for HKD, KRW and TWD prices, converted to USD at the FRED monthly rate (the
     `data/fx` file).
4. **companiesmarketcap.com (CMC)**: year-end values for afterlives after 1996 (pages for Verizon, Xerox, 3M,
   SLB, Ford, BMS, ConocoPhillips, Con Edison, UGI, Foot Locker, Texaco, BellSouth, US West, GM, Dow, Corteva,
   DuPont de Nemours, Nomura Holdings, Nippon Steel, Hitachi, Panasonic, KEPCO and Alcatel-Lucent), retrieved
   2026-11-12.
5. **SEC EDGAR 10-K cover pages** ("aggregate market value of voting stock held by non-affiliates", with its
   date): GM, Kodak, Sears Roebuck, Sears Holdings, Texaco, Union Carbide, Dow, du Pont, Phillips, Ameritech,
   NYNEX, Pacific Telesis, BellSouth, U S WEST, Time Warner, Citicorp, AOL, AOL Time Warner, Chemical/Chase,
   NationsBank, Norwest and Travelers.
   - These are `point` rows at the stated date: Jan-Mar until 2003, then 30 June.
   - For widely held companies they are within a few per cent of the whole company. The exceptions are flagged:
     Sears Holdings (ESL holds about half), du Pont 1994-95 (Seagram's 24%), RJR Nabisco Holdings (KKR) and GM
     (tracking stocks).
6. **kabutan.jp yearly closes** (split-adjusted, back to 1949-1961) x implied shares between the 1989 values of
   eras c and d and the first CMC year-end. For Japanese years before 1989 the 1989 share count is held, so these
   are **est_high (upper bounds)**: Japanese companies kept issuing shares until 1989. They are given only from
   1970.
7. **Japanese banks without price data** (IBJ, Fuji, DKB, Sanwa/UFJ, Tokai, Mitsui/Sakura) are ESTIMATES, moved
   along the Sumitomo Bank/SMFG price path (kabutan 8316), as era d did for 1990-95. They are low confidence and
   flagged in every note.
8. **Deal values and end points**:
   - Wikipedia, "List of largest mergers and acquisitions" (retrieved 2026-11-12).
   - Reuters 2018-10-15 (Sears $32m).
   - Business Standard/Reuters 2009-06-02 (GM 75 cents, $457.2m).
   - Kodak 10-K 2011 with the 42-47 cent prices of Jan 2012.
   - CNN Money 1999-08-04 (Union Carbide).
   - Goldman Sachs history page (Microsoft IPO $777m).
   - Motley Fool 2019-01-16 (Ford IPO, 53.46m shares x $70.50).
   - Cult of Mac / TechCrunch (Apple IPO $1.778bn).
   - Motley Fool 2017 (Walmart IPO "$5m", very low confidence).

## 3. Errors found in other folders and sources (need a human decision)

- **Tencent 2007-2011 (era d) is about 5x too low.** CMC tracks the ADR with the pre-2014-split share count
  (implied 1.94bn shares, against ~10bn split-adjusted). Corrected values are given here: 2008 $12.0bn, 2009
  $40.4bn, 2010 $40.3bn, 2011 $37.2bn, not $7.8bn at end-2010. Because `build_data.py` takes the median of rows
  on the same date, **the era-d tencent rows for 2007-12-31 to 2011 should be dropped** (or f preferred),
  otherwise the two are averaged.
- **CMC "DuPont de Nemours" (DD) before Sept 2017 is Dow Chemical's history, not E. I. du Pont's.** Yahoo's DD
  series is too: May-Dec 2008 is -63% as for Dow; du Pont fell 47%. Era d did not use it. Here it is used as Dow
  Chemical (it matches the Dow 10-K covers, e.g. 2016 $64.2bn), and du Pont's own values come from its 10-K
  covers.
- **era d `ibj` 1999-08-20 = $40.1bn** is the Wikipedia deal value of the whole Fuji/DKB/IBJ combination, not IBJ
  alone. The proxy estimates for end-1999 are IBJ about JPY 4.5tn (~$44bn), and the three banks together about
  JPY 12.5tn (~$123bn), about 3x the 40.1bn. Treat the 40.1bn row with care.
- **era d `facebook` 2012-05-31 = $63.3bn** looks like Class A shares only. The IPO valued all shares at about
  $104bn (added here).
- **Sinopec**: CMC (and era d from Aug 2001) value the whole company at the Shanghai A-share price. The 2000-01
  rows here use the H-share price, so there is a step up in Aug 2001.
- **ICBC 2016-2019**: CMC is on the A price (~1.5x). These rows use the H price.
- **Dow 10-K covers for 2001 and 2002** print "$2.871 billion" and "$2.890 billion" (typos for 28.7 and 28.9).
  The rows recompute price x shares.

## 4. Decisions taken (please confirm)

1. **New GM is a separate vessel (`gm_2010`).** It is a different legal company (formed 2009, assets bought in
   the section 363 sale), and old GM's shareholders got nothing. So old GM's sand drains to the canal and new GM
   rises from the canal at its 2010 re-IPO.
2. **Kodak** ends at the 2012 bankruptcy. The re-emerged Kodak (2013, worth $0.3-1.5bn) is not a vessel.
3. **Sears** continues through Sears Holdings (2005, with Kmart) to the 2018 bankruptcy. A new vessel from 2005
   would also be defensible, because Kmart's holders had the majority.
   - The Sears Holdings 10-K values are lower bounds (`est_low`): ESL (Lampert) held about half.
   - Because `build_data` uses a lone `est_low` as the value, Sears looks about 2x too small in 2005-2017. The
     peak was about $30bn in April 2007 ($195 x ~155m shares; not sourced here).
4. **Dow + du Pont → DowDuPont** is a new vessel. Dow Inc. and Corteva are children (2019). DowDuPont continues
   as DuPont de Nemours.
5. **Phillips → ConocoPhillips** is a new vessel (Phillips holders had about 57%; CMC/Yahoo COP history is
   Phillips'). Calling it a rename would also be defensible.
6. **U S WEST**: in June 1998 the old legal company became MediaOne Group and the telephone company became a new
   U S WEST. The vessel follows the phone business to Qwest (2000). MediaOne is not tracked (bought by AT&T Corp.
   in 2000).
7. **Mitsui Bank** continues as Mitsui Taiyo Kobe/Sakura Bank (Taiyo Kobe poured in 1990, $22.8bn) and merges
   into sumitomo_bank in 2001. **Tokai** merges into sanwa_bank (UFJ) in 2001.
8. **RJR Nabisco Holdings** (re-listed 1991 by KKR, split 1999) was **not** created as a vessel. Its 10-K
   non-affiliate values, if wanted:
   - Jan 1994 $4.9bn and Jan 1995 $5.9bn (both exclude KKR's stake).
   - Jan 1996 $8.9bn, Jan 1997 $8.8bn, Feb 1998 $11.2bn and Feb 1999 $8.9bn.
   - Nabisco Group Holdings, Mar 2000: $3.1bn.

## 5. Still missing

- **No sources found**:
  - burmah_oil and courtaulds (the whole life after 1926).
  - anglo_persian (BP) 1926-1986. A pre-1987 series from the ADR price would need BP's share count, which was
    not found.
  - shell_oil_us 1955-1978.
  - rca 1930-1984 (only the 1985 deal, $6.28bn).
  - woolworth, con_ed and united_gas_improvement 1930-1984.
  - union_carbide 1991-1994.
  - citicorp 1986-1996.
  - time_warner 1990-1992.
  - getty_oil before 1979.
- **Pre-1926 years** of the companies in the 1926-1990 gap-fill were not done: GM 1917-28, Sears, Kodak, Texaco,
  du Pont, Union Carbide, IBM, Coca-Cola, P&G.
- **Before the first value, still missing**:
  - Intel 1971-78 (IPO Oct 1971 at $23.50, roughly $60m; not sourced).
  - Lilly and Pfizer before 1972; AIG 1969-72; J&J 1944-61; Ford 1957-71.
  - Japanese companies 1949-69 (only upper bounds possible).
  - Samsung 1975-99; TSMC 1994-97; China Mobile 1997-99; HSBC before 1996; Nortel before 1997; Petrobras
    before 2000; Gazprom 1996-2005 and 2009; BHP before 1980; Nokia before 1994; Lucent's 1996 IPO; AOL 1992-95.
- **Vodafone 1997-1999** (era d rejected CMC) is still missing.
- **Low-confidence rows to replace if better data turn up**:
  - The Japanese bank proxies.
  - UnitedHealth 1984-89.
  - Walmart 1972-82 and its IPO "$5m".
  - Wells Fargo (Norwest) 1985-93.
  - BHP 1980-95.

## 6. Story beats from the new data

- **Old GM's life ended at $457m** (2009), about 1% of its 1999 value ($55-60bn). New GM came back at about
  $50bn in 2010.
- **Kodak**: $28bn (Jan 1997), then $114-128m at its 2012 bankruptcy.
- **Sears**: $16-17bn in 1997, then $32m on its bankruptcy day in 2018.
- **Tencent was already about $40bn in 2009-10**, not $8bn, and was about $1bn at its 2004 IPO.
- **Apple grew in from $1.8bn** (IPO, Dec 1980). Intel was worth more than Apple at every year-end from 1980 to 1985 except 1981. Microsoft
  listed at $777m and passed Apple and Intel by 1992 ($23.2bn).
- **Walmart**: about $1bn in 1980, $8.9bn in 1985, $67.7bn in 1991 (NYSE #5).
- **The Bells were absorbed for $16-72bn each**, from Pacific Telesis (1997) to Ameritech (1999) and BellSouth
  ($67bn, 2006).
- **Dow and du Pont**: about $160bn combined in 2017, split three ways in 2019.
