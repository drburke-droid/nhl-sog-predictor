# Era C gap-fill (c2), 1926-1990: notes

New rows only; nothing in `c_1925_1990` was edited or duplicated (a company-year that already had a `year_end`/`est_*` row there was skipped). All values are **nominal USD millions** at 31 December, no inflation adjustment. Every derived row shows `shares x price = value` and the source of each in `note`.

| file | rows | content |
|---|---|---|
| values.csv | 715 | year-end values, 1928-1978 (mostly 1928-1968) |
| companies.csv | 1 | `international_nickel` (the only new company that plausibly reached a top 10) |
| events.csv | 3 | Indiana's 1957 distribution of its Jersey Standard shares; du Pont's 1962-65 GM divestiture; Royal Dutch 1958 rights issue |
| top10.csv | 290 | US top 10 for 1935-1963 (except years already complete in c: 1939, 1949, 1953, 1956, 1959), 1967, 1968, 1972 (completes c's partial list); ranks 1-4 only for 1930-1934 |

## 1. Method

value = shares outstanding (millions) x year-end price.

**Prices.**
1. *Bank and Quotation Record* (Commercial & Financial Chronicle), January issues 1929-1979 on FRASER (St. Louis Fed), NYSE "monthly and yearly record": the Dec-31 closing column for the year just ended. Rows were located with the FRASER ALTO OCR, then cropped from the full-resolution IIIF page image (de-skewed) and **read by eye**; the year's low/high are stored in the build data as a check. Each row's `source` gives the FRASER item id and URL.
   - Where the December row was unreadable, the *first price of the following year* (the "Price Jan. 2" column of the next issue) is used; the note says "next first-of-year price used as year-end proxy".
   - 1975 rows come from a regional-exchange table in the Jan-1976 issue (the NYSE section of that issue is not OCR'd); the note says so.
   - Reading error is typically ±1/8-1/4 point; rows marked "reading uncertain" are ±1-3%.
2. *Yahoo Finance chart API* daily closes 1962-1978 for XOM, KO, PG, MMM, MRK, MO, CVX, XRX, un-split with Yahoo's own split events. Checked against the NYSE Fact Book 31-Dec-1979 prices: all within 0.4%.
   - **Not used: Yahoo `DD` (du Pont).** Its pre-1990 series does not reproduce the NYSE 1979/1980 prices (ratio 1.26-1.31), its split list lacks the 1979 3-for-1, and it is distorted by the 2019 DowDuPont spin-off adjustments. du Pont prices are from the B&Q record throughout.

**Share counts.** For each company, dated anchor counts (below) are converted to split-adjusted units, interpolated linearly in time, and converted back. Split dates are taken from par-value changes and price breaks in the B&Q record (and Yahoo for 1962+). They are dated to the month at best; a year-end value is only affected if a split date falls in the wrong year. Beyond the last anchor the split-adjusted count is held flat, and the note says so.

## 2. Share-count anchors (sources in full)

- **NYSE, *The Exchange*** (Internet Archive `sim_exchange-nyse_*`):
  - Apr 1958, "10 companies with most common shares listed": listed shares on 31 Jan 1948, 1953, 1957 and 1958 for GM, SONJ, GE, Sears, AT&T, Socal, Texas Co, US Steel, Socony, du Pont.
  - Dec 1955, "blue chip" table: listed shares **and market value at 31 Dec 1954** for du Pont, SONJ, Kodak, IBM, Socony, Socal, Indiana Standard, Coca-Cola and Dow. These market values are used directly as 1954 `year_end` rows.
  - Jun 1957: shares outstanding of companies with over 20m shares.
  - Jul 1959: the 20 common stocks with the most shares listed. Two row names are illegible in the OCR: 60,015,665 is assigned to Texaco and 48,770,011 to Socony Mobil, by position and by continuity with 1958.
  - Apr 1958: Royal Dutch rights issue (see below).
- ***Forbes*** (Internet Archive `sim_forbes_*`):
  - 1 Jul 1927 (SONJ 20,695,902);
  - 15 Dec 1928 (Socal 12,594,098);
  - 15 May 1930 (SONJ 25,399,968);
  - 1 Jul 1930 (Kodak 2,261,320);
  - 1 Oct 1930 (Sears 4,682,341; Woolworth 9,750,000);
  - 1 Nov 1930 (Texas Corp 9,850,050);
  - 15 Nov 1930 (International Nickel 14,609,467);
  - 1 Feb 1931 (GM 43,500,000);
  - 1 Mar 1931 (Sears 4,797,924);
  - 15 Jun 1931 (AT&T 18,019,681);
  - 15 Dec 1931 (GE 28,845,928);
  - 15 Jun 1933 (AT&T 18,662,000);
  - 1 Nov 1933 (Sears 4,780,000);
  - 1 Nov 1934 (Consolidated Gas 11,477,000);
  - 15 Jun 1936 (Sears 4,832,447);
  - 15 Nov 1936 (GM 43,500,000);
  - 1 Aug 1937 (AT&T 18,685,093; Kodak 2,250,921; GE 28,845,927);
  - 15 Feb 1952 (SONJ 60,580,000);
  - **1 Jan 1956**: the 20 NYSE companies with the most shares "including recent splits and new issues", at about end-1955. The GE figure is missing from the OCR.
  - 15 Jun 1960 (SONJ 216,532,000).
- **TNEC Monograph 29** (1940), *The distribution of ownership in the 200 largest nonfinancial corporations*, issue headers (shares outstanding, Dec-1937 price): du Pont 11,065,708 (31 Dec 1937); Kodak 2,250,921; Consolidated Edison 11,476,527. Source: archive.org `investigationofc2902unit`.
- **AT&T annual reports** (memorial.bellsystem.com; files from the first pass): average shares 1951-53 (AR 1953), 1970-74 (AR 1974) and 1975-79 (AR 1979).
- **Implied counts**: GFD (Finaeon) values / B&Q price give AT&T 1945, 1950, 1960 and 1965. CRSP Aug-1929 values (McGrattan & Prescott 2004) / price give GM, SONJ and US Steel in 1929 (these were already in c).
- **NYSE Fact Book 1980** listed shares at 31 Dec 1979 (via c) are the closing anchor for 1960-78 interpolations.

## 3. Confidence and ranges (decisions for a human)

- **High** (two or more close anchors, clean price): AT&T 1929-1968, GM 1928-1966, GE 1928-1961, SONJ 1929-1963, US Steel 1929-1961, du Pont 1937-1968, Texaco 1930-1968, Socal 1928-1972, Kodak 1930-1968.
- **Medium**: 1960s-70s values interpolated over 20 years between the 1959 and 1979 counts (GM, SONJ, Kodak, Sears, Texaco, Gulf, Socal, Mobil, Indiana). The error is a few %, more where shares were issued for acquisitions.
- **Upper bounds (`est_high` only)**: P&G, Merck, Philip Morris, Xerox and 3M 1962-78. Only the 1979 NYSE count is known, held flat back in split-adjusted units, so any later issuance is counted. Xerox (the SDS merger in 1969) and Philip Morris are probably the most overstated (10-25%). AT&T 1928 (mid-1929 count) and US Steel 1928 (authorised shares) are also upper bounds.
- **Ranges (`est_low`/`est_high`)**:
  - Sears 1937-44: a Dec-1936 rights issue; the count lies between 4.83m and 5.91m.
  - du Pont 1930, 1932, 1933: 10.3m is an **unverified** low assumption; the high figure is the TNEC 1937 count.
  - Coca-Cola 1935-53: 4.0m (an unverified assumption of 1.0m shares before the 1935 4:1 split) to 4.28m (1954).
  - IBM 1955-61: the split history is uncertain, so the low uses the 1956/1957/1959 splits only; the high adds a 1955 5:4 split and 2.5% annual stock dividends. The range is wide (±20%).
- **Assumptions flagged in notes**:
  - Socony-Vacuum 1931-47: the Jan-1948 count is held flat.
  - Union Carbide 1929-47: the 1954 count / 3, held flat. This may overstate by up to ~8%.
  - Gulf 1943-50: the 1954 count / 2.
  - Indiana Standard 1951-53: the pre-1954-stock-dividend count.
  - Con Ed, Woolworth and Inco: one count, held flat.
  - Coca-Cola: the 1960 split is taken as **3-for-1** (company split history). With 2-for-1, 1962-76 values would be ~1/3 lower.
- **Royal Dutch 1954-58 (whole company)**: total shares are implied from *The Exchange* Apr 1958, where 14.158m NY shares = 21% of the total after a 1-for-8 rights issue. Values: 1954 $1.7bn, 1956 $2.6bn, 1958 $3.2bn. Low confidence, and it assumes the NY price equals the home-market price. Royal Dutch was not near the US top 10 in those years.
- **Possible error in c (for a human)**: c's IBM 1962-69 rows un-split the 1970 GFD count without stock dividends (30.4m in 1962). Rolling the 1954 *Exchange* count forward with known splits and 2.5%/yr stock dividends gives ~23-25m in 1962. **c's IBM 1962-67 values may be 5-20% too high.**
- **Exxon 1983-90 treasury stock**: not touched (no rows after 1978 for SONJ). The 1979 anchor is NYSE *listed* shares, but treasury stock was small before 1983.

## 4. Top-10 lists: scope and checks

- The universe is **NYSE-listed US companies**, as in CRSP/Dimensional before 1962. The candidate set is the ~30 companies researched.
- Check: for end-1939, end-1949 and end-1959 our values reproduce the Dimensional (CRSP) top 10 **exactly**, members and order (Sears and Coca-Cola at #9/#10 in 1939 are within their ranges). In the 1972 list, AT&T's value rests on the 2 Jan 1973 price (proxy).
- 1930-34: only ranks 1-4 are given. Railroads (Pennsylvania RR ~13.2m shares) and utility holding companies were not valued, and some may belong in ranks 5-10.
- **Not in the universe**: firms on the New York Curb or Chicago exchanges, chiefly **Gulf Oil before 1943** and **Standard Oil (Indiana) before ~1954**. Each was worth roughly $0.4-0.7bn in the late 1930s and would rank about #7-10 in an all-exchange list. Also excluded: private Ford (until 1956) and Aramco, and non-US firms (Inco, Royal Dutch, Shell Transport, Unilever, ICI).
- 1964-66, 1969-70 and 1973-78 have no list: AT&T, GM, Kodak, du Pont, Sears, Texaco or Mobil prices were unreadable in those B&Q issues (the Jan-1965, Jan-1971 and Jan-1974 issues have almost no usable OCR).

## 5. Gaps still open

- **1926-1927 for everyone**: the FRASER B&Q run starts in 1928, and no other year-end price source was found in time.
- IBM before 1954.
- US Steel and Union Carbide after 1966 (until 1979 for UCC; US Steel never re-enters the top 50).
- P&G, Merck, 3M, Philip Morris and Xerox before 1962.
- Coca-Cola 1926-34 and 1955-61.
- Royal Dutch 1926-53 and 1959-85.
- Indiana Standard 1926-50.
- Year gaps for the leaders: AT&T 1964 and 1969-70; GM 1969-70 and 1977-78; Kodak 1964, 1969-70 and 1977; du Pont 1929, 1931, 1956, 1964, 1969-70 and 1973-78; Sears 1929, 1964-66, 1970, 1973 and 1977; Texaco and Mobil several 1964-78 years.
- Leads:
  - FRASER pages can be read from the IIIF images where the OCR fails (the method is in `c2_scripts/bqrows.py`, `manualcrop.py`);
  - the Feb/Mar-1928 B&Q issues give Jan-1928 prices, a proxy for end-1927.

## 6. Story beats (new)

- **GM, not AT&T, was the largest US company at end-1928**: GM $3.55bn against AT&T at most $2.5bn. AT&T was #1 again by end-1929.
- **Con Ed was #3 at the 1932 trough**: AT&T $1.95bn, Jersey Standard $0.71bn, Consolidated Gas $0.68bn, GM $0.57bn.
- **du Pont was #2 in 1941**, ahead of GM, and #2 again at end-1955 ($10.5bn). At end-1955 GM, du Pont and Jersey Standard were all worth more than AT&T ($9.2bn), which fell to #4. AT&T was back at #1 by end-1957.
- **GM was #1 at end-1954, 1955 and 1956**, consistent with GFD.
- **Union Carbide was #5-6 every year 1935-1952.** **Inco, a Canadian nickel miner, was worth ~$0.93bn at end-1936, #7 among NYSE stocks** (it is excluded from US lists).
- **IBM rose from ~$1.5bn (1954) to ~$11bn (1960)**, level with GM. AT&T reached ~$34bn at end-1963, against GM's $22bn.
- **The Nifty Fifty in 1972**: IBM $46.6bn, AT&T $29.3bn, Kodak $24.0bn, GM $23.3bn, Exxon $19.1bn, Sears $18.4bn.

## 7. Share-anchor codes used in the `note` column

The `note` of each derived row names its share anchors by code and date (for example "interpolated EX5804 1953-01-31 -> EX5410 1954-09-30").

| code | source |
|---|---|
| TIME29 | TIME, AT&T half-year report 1929 (via c_1925_1990) |
| GFD | Finaeon/Global Financial Data "Two Centuries of American Leadership" value / B&Q year-end price (implied count) |
| F310615 | Forbes 15 Jun 1931 p. (AT&T "has outstanding 18,019,681 shares"), archive.org sim_forbes_1931-06-15 |
| F330615 | Forbes 15 Jun 1933 table of common shares (AT&T 18,662,000), archive.org sim_forbes_1933-06-15 |
| F370801 | Forbes 1 Aug 1937 table "No. of common shs." (AT&T 18,685,093; Kodak 2,250,921; GE 28,845,927), archive.org sim_forbes_1937-08-01 |
| EX43 | The Exchange (NYSE) 1943 (AT&T 18,687,000; via c_1925_1990) |
| EX5804 | The Exchange (NYSE) Apr 1958 p.5 "10 companies with most common shares listed", shares listed 31 Jan 1948/1953/1957/1958, archive.org sim_exchange-nyse_1958-04 |
| EX5410 | The Exchange (NYSE) Oct/Nov 1954, listed shares 1 Sep / 30 Sep 1954 (via c_1925_1990) |
| EX5512 | The Exchange (NYSE) Dec 1955 "blue chip" table: listed shares and market value 31 Dec 1954, archive.org sim_exchange-nyse_1955-12 |
| EX5706 | The Exchange (NYSE) Jun 1957 table "having over 20 million shares outstanding", archive.org sim_exchange-nyse_1957-06 |
| EX5907 | The Exchange (NYSE) Jul 1959 "20 common stocks which top all others in number of shares listed", archive.org sim_exchange-nyse_1959-07 |
| F560101 | Forbes 1 Jan 1956 letter: 20 NYSE corporations with most shares outstanding "including recent splits and new issues", archive.org sim_forbes_1956-01-01 |
| F600615 | Forbes 15 Jun 1960 (Standard Oil NJ 216,532,000 shares), archive.org sim_forbes_1960-06-15 |
| ATTAR53 | AT&T Annual Report 1953 (average shares 1951-53), memorial.bellsystem.com |
| ATTAR74 | AT&T Annual Report 1974 (average shares 1970-74) |
| ATTAR79 | AT&T Annual Report 1979 (average shares 1975-79) |
| NYSE79 | NYSE Fact Book 1980 "Fifty leading stocks in market value", listed shares 31 Dec 1979 (via c_1925_1990) |
| CRSP29 | CRSP end-Aug 1929 value (McGrattan & Prescott 2004 Table A2) / Aug-1929 price (implied; via c_1925_1990) |
| F300515 | Forbes 15 May 1930 table (Standard Oil NJ 25,399,968 shares), archive.org sim_forbes_1930-05-15 |
| F270701 | Forbes 1 Jul 1927 table (S.O. of N.J. common 20,695,902), archive.org sim_forbes_1927-07-01 |
| F520215 | Forbes 15 Feb 1952 (Standard Oil NJ 60,580,000 shares), archive.org sim_forbes_1952-02-15 |
| F311215 | Forbes 15 Dec 1931 (GE 28,845,928 common; Kennecott 9,393,150), archive.org sim_forbes_1931-12-15 |
| F310201 | Forbes 1 Feb 1931 table (GM 43,500,000; Chrysler 4,438,196), archive.org sim_forbes_1931-02-01 |
| F361115 | Forbes 15 Nov 1936 table (GM 43,500,000), archive.org sim_forbes_1936-11-15 |
| TNEC | TNEC Monograph 29 (1940) "Distribution of ownership in the 200 largest nonfinancial corporations", issue headers (shares outstanding, price Dec 1937), archive.org investigationofc2902unit |
| F301001 | Forbes 1 Oct 1930 table (Sears 4,682,341; Woolworth 9,750,000), archive.org sim_forbes_1930-10-01 |
| F310301 | Forbes 1 Mar 1931 (Sears 4,797,924), archive.org sim_forbes_1931-03-01 |
| F331101 | Forbes 1 Nov 1933 table (Sears 4,780,000; Chrysler 4,369,000), archive.org sim_forbes_1933-11-01 |
| F360615 | Forbes 15 Jun 1936 (Sears 4,832,447), archive.org sim_forbes_1936-06-15 |
| F300701 | Forbes 1 Jul 1930 table (Eastman Kodak 2,261,320), archive.org sim_forbes_1930-07-01 |
| F301101 | Forbes 1 Nov 1930 (Texas Corp 9,850,050 shares), archive.org sim_forbes_1930-11-01 |
| F281215 | Forbes 15 Dec 1928 (Standard of California 12,594,098 shares), archive.org sim_forbes_1928-12-15 |
| F301115 | Forbes 15 Nov 1930 (International Nickel 14,609,467 common), archive.org sim_forbes_1930-11-15 |
| F341101 | Forbes 1 Nov 1934 table (Consolidated Gas 11,477,000; Chrysler 4,361,000), archive.org sim_forbes_1934-11-01 |
| F280701 | Forbes 1 Jul 1928 (Chrysler 4,423,484 shares), archive.org sim_forbes_1928-07-01 |
| F580201 | Forbes 1 Feb 1958 (Kennecott capitalization ~10,800,000 shares), archive.org sim_forbes_1958-02-01 |
| EX5804RD | The Exchange Apr 1958: Royal Dutch NY shares 14,158,000 = ~21% of outstanding on 14 Feb 1958 after 1-for-8 rights issue (implied totals) |
