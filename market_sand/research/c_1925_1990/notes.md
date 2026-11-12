# Era C, 1925-1990: notes

All values are **nominal** and in **millions of the stated currency** (USD unless marked; NTT also in JPY). Nothing is inflation-adjusted. Every derived value shows its arithmetic (shares x price) in the `note` column of `values.csv`.

## 1. What the files contain

| file | rows | content |
|---|---|---|
| companies.csv | 65 | every company that was (or plausibly was) in the world top 10 in 1925-1990, plus near-misses that recur in the sources and the 7 regional Bells |
| values.csv | 621 | market values. Where sources or share counts conflict there are `est_low`/`est_high` pairs |
| events.csv | 16 | renames, mergers, acquisitions, the AT&T break-up, the NTT listing, the Bank of England nationalisation |
| top10.csv | 208 | year-end rankings where evidence exists. **Most years are a US-only list** (see section 4) |

## 2. Method and sources, best first

1. **NYSE Fact Books 1980-1991** have the table "Fifty leading stocks in market value": listed shares x 31 Dec price for year-ends **1979-1990**. They are scanned in the Internet Archive microfiche collection (ids `micro_IA40706901_0335` ... `micro_IA40706937_0391`; Fact Book year N covers year-end N-1). The values were read by hand from the page images.
   - Caveat 1: "listed shares" includes treasury stock. This hardly matters for most companies, but **Exxon** bought back about 25% of its shares in 1983-89. Its NYSE values from 1983 on are therefore too high: end-1989 is $90.9bn listed, against $63.8bn on outstanding shares (MSCI). Both are given.
   - Caveat 2: for foreign companies such as Royal Dutch, only the New York-registered shares are counted. Those Royal Dutch rows are **not** company totals.
   - Caveat 3: foreign companies were excluded from the 1979 survey.
2. **CRSP end-August 1929** values for all 90 S&P index companies come from McGrattan & Prescott, "The 1929 Stock Market: Irving Fisher Was Right", *International Economic Review* 45 (2004), Table A2 (https://users.cla.umn.edu/~erm/data/sr294/ier/tex/ier.pdf). AT&T and du Pont were not S&P 90 members, so they are missing from this table.
3. **Dimensional Fund Advisors**, "Large and in charge? Giant firms atop market is nothing new" (2020), Exhibit 2 ("Big Board"). It gives the ranks of the 10 largest US stocks at the start of each decade (CRSP/Fama-French universe), read from the chart image. The "1930" column = year-end 1929, and so on. CRSP labels each firm with its *latest* name, so "Marathon Oil" is **US Steel** (via USX-Marathon), "Exxon" is Standard Oil NJ, "Altria" is Philip Morris, "Penn Central" is the Pennsylvania RR, "Con Edison" is Consolidated Gas, and "Chevron" is Socal. **"Philadelphia Co." (#8 at end-1929) is unidentified.** It may be United Gas Improvement (CRSP Aug-1929 $1.10bn); needs checking.
4. **Finaeon / Global Financial Data** (Bryan Taylor): "Two Centuries of American Leadership" (largest US firm and its market cap every 5 years, 1900-1995) and "Four Centuries of Global Leadership" (largest firm in the world every 10 years). The units are not printed. They are read as USD millions, because NTT 1990 = 119,000 = "$119 billion" in the text, and the IBM 1980/85/90 figures match the NYSE Fact Book to within 0.1%.
5. **The Exchange** (NYSE monthly magazine, Internet Archive `sim_exchange-nyse_*`):
   - top-20 NYSE common stocks by market value on 15 Sep 1953 (Oct 1953 issue);
   - top 20 on 1 Sep 1954 with share counts and prices, plus 1 Sep 1953 comparisons (Oct 1954 issue);
   - leaders at 30 Sep 1954 (Nov 1954 issue);
   - shares listed on 31 Jan of 1948, 1953, 1957 and 1958 for the 10 companies with the most shares (Apr 1958 issue).
6. **Bank and Quotation Record** (Commercial & Financial Chronicle, FRASER: https://fraser.stlouisfed.org/title/bank-quotation-record-5571):
   - monthly NYSE price tables (first price of the year, opening and close of the month, monthly and year-to-date high/low) were used for 1929-1933;
   - January issues (31 Dec close) were used for 1949-1974.
   - The OCR is poor: fractions mostly print as "%", so prices carry a +-1/2 point error, and many rows are unreadable. Several year-end rows were read from cropped page images.
   - **These derived values are medium/low confidence.**
7. **Yahoo Finance** split-adjusted monthly and daily closes were "un-split" with Yahoo's split events. This gives IBM and GE 1962-1978, 3M 1972-78, and the 1987 crash days, which were scaled to the NYSE end-1987 values. Checks: IBM 1972 = $402 and GE 1972 = $72 7/8, identical to the Bank and Quotation Record.
8. **AT&T annual reports** (memorial.bellsystem.com): 1974 and 1979 reports, for average shares in 1970-79 and year-end prices in 1975-79.
9. **Siegel & Schwartz**, "Long-term returns on the original S&P 500 companies", *Financial Analysts Journal* 62(1), 2006, Table 3. It gives ranks only for 1 March 1957.
10. **AT&T break-up**: CRS Report 83-221E (7 Dec 1983), Appendix E. It gives when-issued opening, weekly and 2 Dec 1983 prices of old AT&T, new AT&T and the 7 regional holding companies.
11. **Japan**:
    - BusinessWeek Global 1000 (17 Jul 1989), as reproduced by Diamond and quoted (rounded) at toshitime.com, covering NTT and the top 10 by mid-1989;
    - MSCI end-1989 world top 20 as printed by CNBC (29 Apr 2014). It excludes NTT; AT&T's value equals the NYSE end-1989 figure exactly, hence year-end;
    - NTT share-sale prices, first price and record high from Japanese Wikipedia (日本電信電話) and NTT IR;
    - AP/Deseret News 22 Mar 1990 for the NTT price in March 1990;
    - Washington Post 5 Jun 1988 (search snippet only) for NTT = $317bn;
    - NTT shares are 15.6m (¥780bn capital at ¥50,000 par), consistent with the cited ¥49.6tn peak at ¥3.18m.
12. **Bank of England**: Annual Report 1947 ("3% Treasury Stock ... 1st March 1946. Exchange for Bank of England Stock ... £58 million"; capital £14,553k). Stockholders received £400 of 3% Treasury Stock per £100 of Bank Stock.

## 3. Ranges, doubts and decisions for a human

- **AT&T share count 1929-30.** About 13.03m shares in mid-1929 (TIME, half-year report) against 17.65m implied at end-1930 (GFD value / price), because convertible bonds were converted and new stock issued in 1929-30. The Sept-1929 peak ($4.04bn) uses 13.03m and is therefore a *lower* bound. End-1929 is given as an est_low/est_high pair (2.9-3.9bn). AT&T's market cap may actually have peaked in **April 1930** (price 274 5/8 x ~15-17m shares ≈ $4.1-4.8bn), but this is not established.
- **1931-40 AT&T shares** are assumed to be ~18.66m (The Exchange shows 18,687,000 in 1943; GFD 1935/1940 caps imply ~18.7m).
- **Share interpolation.** These estimates use real prices but interpolated or assumed share counts:
  - GM 1959-1978 (281m in 1958 to 291m in 1979; error <1%);
  - Exxon 1971-75 (203.5m in 1958 to 226.6m pre-split in 1979; ~2%);
  - IBM 1962-69 (1970 GFD-implied count un-split with Yahoo split ratios; ignores stock dividends; up to ±10%);
  - IBM 1971-78 (between GFD 1970 and 1975 and NYSE 1979; ~2%);
  - GE 1962-78 (1954/58 trend plus 1971 split; the 1976 Utah International share issue is assumed; ±5%);
  - Kodak 1972-76 (count fixed at 161.6m; ±2%);
  - Sears and Texaco 1971-74 (est_low/est_high pairs, because split timing is unverified).
- **GFD 1940 AT&T**: 3,143.24 in one table and 3,134.24 in the other (typo in one of them).
- **GFD claims National City Bank (later Citicorp) was "the largest company in the world in 1925, 1927 and 1928".** This conflicts with GFD's own US table (AT&T 1925). Bank stocks then traded over the counter and are not in CRSP, so this cannot be checked against the CRSP-based lists. **No value was found.** It is a possible story beat, but it needs a primary source (NCB shares x OTC bid, 1925-29) before use.
- **Exxon 1983-1990**: use est_low (outstanding shares) wherever available. Only 1989 has one: MSCI $63.8bn. For other years the NYSE values are probably 15-30% too high.
- **1929 monthly granularity.** Requested; only partly delivered. The series for AT&T, GM, GE, SONJ and US Steel give roughly 10-20 dated points each in 1929-1932 (monthly closes where legible, plus each year's high and low and the year-ends). The rest of the monthly grid can be interpolated with the monthly Dow Jones Industrial Average, which is not supplied here.
- **Deal values** for Gulf (~$13.2bn), Getty (~$10.1bn), Shell Oil minority (~$5.7bn), RCA ($6.28bn) and RJR (~$25bn) are the widely reported figures, **not re-verified** in this session. Each event row says so.

## 4. Where only the US list is known (top10.csv)

- **1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975**: only rank 1 is known (GFD, US). Rank 1 is probably also the world #1 in these years.
- **1929, 1939, 1949, 1959, 1969**: full US top 10 (Dimensional ranks; values only where derived). Values exist only for the leaders, and non-US firms are not included.
- **1953** (15 Sep, NYSE only, The Exchange) and **1956** (1 Mar 1957, S&P 500, Siegel): ranks.
- **1972**: partial list, ranks 1-7, derived; AT&T's value is missing.
- **1979-1988**: NYSE top 10 (US-listed companies, Schlumberger included). Missing from this list:
  - **Royal Dutch Petroleum**, probably world top 10 throughout the 1980s (whole-company value found only for 1989: $41.0bn, plus $54.3bn for the Royal Dutch/Shell group in mid-1989);
  - possibly **Shell Transport, BP** and, from 1987, **NTT and the big Japanese banks**. In mid-1988 (31 May), 8 of the world top 10 were Japanese, 5 of them banks, and Exxon (#6) was the only US company (Roth 1989, quoted by Motley Fool 2017).
  - **Treat 1987 and 1988 as US lists, not world lists.**
- **1989**: world list = NTT (#1, no year-end value; $163.8bn in mid-1989) + MSCI ranks. Mitsubishi Bank ($59.2bn in mid-1989, not in the MSCI list) may belong in it.
- **1990**: NTT #1 ($119bn, GFD), then US companies. The Japanese banks, Toyota and TEPCO probably still belong in ranks 2-10 at end-1990; no values were found.

## 5. Gaps (not found)

- **Non-US companies before the 1980s** (Royal Dutch, Shell T&T, Anglo-Persian/BP, Unilever, ICI, Imperial Tobacco): no market values were found.
  - Leads: Wikipedia says Unilever was the largest British company by market value in 1930. An unsourced blog says Imperial Tobacco was "the second largest industrial company in the world by 1937", behind only GM.
  - L. Hannah's unpublished 1998 paper "The world's largest industrial firms, 1912, 1937, 1962 and 1987" (cited in Hannah 1999, NBER volume *Learning by Doing*) would settle this but was not obtained.
  - My reading of the US values: in 1929 the US #10 was ~$1bn, so UK and Dutch firms (each roughly $0.5-0.7bn) were probably outside the world top 10. At the 1932 trough the threshold fell to perhaps $0.3bn, so **Imperial Tobacco or Royal Dutch may have entered the world top 10 in 1932-37**. This is unverified.
- **du Pont** before 1953, **Union Carbide, Sears, Texaco, Socal and Gulf** before 1953, and **Kodak and Xerox** in the 1960s: ranks known, values not.
- **AT&T** 1926-29, 1931-34 year-ends, 1936-39, 1941-44, 1946-55, 1957-59, 1961-64 and 1966-74 (except GFD 5-year points and 1956): year-end values are missing. These could be filled from the Bank and Quotation Record January issues (price) and AT&T annual reports (shares). A 1975-79 series is included.
- **Japanese banks 1987-1990 quarterly**: only mid-1989 (BusinessWeek) and end-1989 (MSCI) values. **NTT**: dated points Feb 1987-Mar 1990 plus end-1990, but no quarter-end prices for 1988-89.
- **Polaroid** (Nifty Fifty icon): about 32-33m shares x 1972 high $149 1/2 ≈ $5bn. That is a rough estimate from an unverified share count, so it is not included as a value. Polaroid was never near the top 10. The same goes for **RCA in 1929** ($0.65bn at end-Aug; CRSP).

## 6. Myths to avoid

- "RCA was the biggest stock of 1929." It was the most famous bubble stock but was worth ~$0.65bn (CRSP) when GM was $3.1bn and AT&T ~$3.9bn.
- "IBM was the first $100bn company (1987)." NTT passed $100bn from its first trading day in Feb 1987 (¥25tn); IBM touched ~$108bn on 21 Aug 1987.
- "NTT was #1 all through 1987-1990": true for the BusinessWeek and GFD lists, but MSCI-based retellings (CNBC, Motley Fool 2024, Statista) put IBJ #1 in 1989 because MSCI's index excluded NTT.
- Popular lists in today's dollars, revenue-based lists (24/7 Wall St, Fortune 500) and AI-written pages (Grokipedia's 1980/1985 tables) were not used.
- finhacker.cz "top 20 S&P 500 by market cap 1989-" maps old tickers to today's companies (e.g. "AT&T" = SBC, "Verizon" = Bell Atlantic) and its GE 1989 figure is wrong. It was not used.

## 7. Story beats

- **AT&T's dominance grew in the crash.** At their 1932 lows, GM (~$0.33bn), GE (~$0.25bn), Standard Oil NJ (~$0.49bn) and US Steel (~$0.19bn) were together worth less than AT&T at its own low (~$1.30bn).
  - GE lost ~91% (Aug 1929 $2.85bn to May 1932 $0.25bn) and GM ~92% (Mar 1929 $4.0bn to Jun 1932 $0.33bn); AT&T lost ~68%.
- **GM peaked in March 1929, six months before the crash.** Standard Oil NJ and US Steel made their highs in April 1930, in the post-crash rally; SONJ's April 1930 high was above its 1929 peak.
- **The break-up made AT&T worth more.** The 8 pieces opened at $59.8bn (21 Nov 1983) against $58.8bn for old AT&T, and were worth $73.6bn a year later. BellSouth alone ranked #16 in the US at end-1984.
- **The 1980 oil sand-storm.** At end-1980, 7 of the 10 largest NYSE stocks were oil or oil-service companies. Schlumberger, a Franco-American oilfield-services firm incorporated in Curaçao, was #5.
- **The Nifty Fifty.** At end-1972 Kodak ($24.0bn) was worth more than GM ($23.3bn). By end-1974 Kodak was $10.2bn and GM $8.8bn. IBM fell from $46.6bn to $24.8bn.
- **NTT.** On 22 Apr 1987, eleven weeks after listing, NTT was worth ¥49.6tn (~$350bn), more than IBM, Exxon, GE, AT&T and GM combined (those five were worth ~$226bn at end-1986 on NYSE values; ~$250-260bn is a rough April-1987 estimate, not computed). By March 1990 it was down to ~$108bn.
- **The mid-1989 BusinessWeek top 10 held 7 Japanese companies.** In the MSCI year-end-1989 list, 6 of the top 10 were Japanese even without NTT, and IBJ was worth $104bn.
