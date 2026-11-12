# Era B, 1825-1925: notes

Files: `companies.csv` (58 companies), `values.csv` (292 rows), `events.csv` (22), `top10.csv` (131 rows, 21 year-ends).
All values are NOMINAL millions in the currency the source quotes. No inflation or FX conversion has been applied, except where noted for London quotations of dollar shares (see Conventions). `top10.csv` ranks companies with rough FX: GBP 1 = USD 4.8665 (USD 3.49 for 1920), and 25.22 fr per GBP. The value and currency columns stay native.

## Method

1. **Capital x price.** Almost every value is shares (or nominal stock) outstanding x price at the date. The arithmetic is in the `note` column.
   British "stock" is quoted per GBP 100 nominal, so value = nominal stock (GBP m) x price / 100.
2. **Main primary source, 1869-1925: the Investor's Monthly Manual (IMM).** These are scans of the December issues held by the Yale ICF (https://depot.som.yale.edu/imm/).
   - The Yale data-download database is broken: its search returns HTTP 500. I downloaded the scanned PDFs instead, OCR'd them with tesseract to find the rows, and read every number used from the page image.
   - Each December issue lists, for each security: nominal capital, the month's open, high, low and latest prices, and the year's high and low. A second table gives high and low for the previous 3-4 years plus the previous December's closing price. "Latest" is used as the year-end price.
   - Years read: 1869, 1873, 1880 (with 1879 closes), 1890, 1900, 1907, 1912, 1913, 1920 and 1925.
3. **London quotations of foreign shares.**
   - French railways and Suez: the IMM gives capital as GBP nominal at 25 fr = GBP 1. Shares of 500 fr = GBP 20; Nord shares of 400 fr = GBP 16. Prices are in GBP per share, so those values are GBP.
   - American and Canadian shares were quoted in London in "dollars" at the conventional rate of USD 5 = GBP 1, which is about 2.7% above New York parity. I convert them to New York equivalents with these factors:
     - x 0.9733 (4.8665/5) up to 1914. Check: Union Pacific in Dec 1912 was 164.75 in London and 160.13 in New York, a ratio of 0.972.
     - x 0.698 for Dec 1920, when GBP 1 was about USD 3.49.
     - x 0.97 for Dec 1925.
   - Where the Yale NYSE price file has a reliable December price, I use it instead.
4. **What counts as equity.**
   - British railways: ordinary or consolidated stock only. Preference, guaranteed and debenture stocks are left out because they behaved like bonds and were often larger than the ordinary.
   - Midland: after 1897 its ordinary stock was split into preferred converted and deferred converted ordinary. Both are included.
   - US companies with large preferred issues (US Steel, Union Pacific, Atchison): the value is common plus preferred, which is how Hannah measures it. The common-only figure is always given in the note.
   - Great Northern (US) had only "preferred" stock, and it is counted.
   - **This needs a decision (see below):** GFD ranks on common stock only. That is why GFD calls the Pennsylvania RR the largest US company in 1912-15, while on Hannah's basis US Steel ($740m) is larger.
5. **Suez.** Value = capital shares plus actions de jouissance (amortised shares, priced at about the capital price less GBP 20 par; this is an assumption).
   - Left out: the founders' parts (10% of profits) and the parts civiles (the Egyptian 15%, sold to French interests in 1880). Including them would add about 35% to Suez.
6. **Banque de France.** I found no continuous price series because Gallica and HAL are blocked to this agent. There are only scattered prices from Neymarck (1893) and from JSFS (1884).
7. **Pre-1869 London.** The prices come from the *Gentleman's Magazine* "Daily price of stocks", read from page images of the December issues on archive.org:
   - Bank stock, India Stock and South Sea Stock for 1825, 1830, 1835, 1840, 1845, 1847, 1850, 1855, 1857, 1860 and 1865.
   - Most of these December issues print late-November prices, so their kind is `point`.
   - Bank stock annual highs and lows for 1825-46 come from Francis (1847); `annual_avg` rows are the midpoint of each year's high and low.
8. **Second Bank of the US.** Year-end New York quotes for 1825-1835 come from the *New York Shipping and Commercial List* (Yale ICF files). Capital: 350,000 shares of $100.
9. **US greenbacks.** US prices from 1862 to 1878 are in greenback dollars, which were below gold. This affects the GFD values for NYC in 1870 and 1875.

## Sources

- Investor's Monthly Manual, London, Dec issues 1869-1925. Yale ICF scans, https://depot.som.yale.edu/imm/ (pdfdownload.php).
- L. Hannah (1999), "Marshall's 'Trees' and the Global 'Forest': Were 'Giant Redwoods' Different?", in N. Lamoreaux, D. Raff and P. Temin (eds), *Learning by Doing in Markets, Firms and Countries*, NBER/Chicago. Table 7A.1: the 100 largest industrial firms of 1912 by equity market value, in $m at 1912 prices. https://www.nber.org/system/files/chapters/c10235/c10235.pdf
  - It covers industrials only: no railways, banks or utilities.
- J. Francis (1847-48), *History of the Bank of England*, vol. 2, appendix "The highest and lowest price of Bank stock in each year from 1732 to 1846". https://archive.org/details/historyofbankof02fran
- *Gentleman's Magazine*, "Daily Price of Stocks" (London brokers' lists), December issues 1825-1865. archive.org items `sim_gentlemans-magazine_YYYY-12_*`.
- Goetzmann, Ibbotson and Peng (2001), "A new historical database for the NYSE 1815 to 1925", *J. Financial Markets*. Data from Yale ICF (monthly prices, revised 2021) and the *New York Shipping & Commercial List* 1817-1835 files. https://som.yale.edu/faculty-research/our-centers-initiatives/international-center-finance/data/historical-newyork
  - Caution: some columns in the 2021 price file are mislabelled. Its "New York Central" column after 1875 is not NYC (it falls to 12 in 1877), so I did not use it for NYC before 1912.
- Global Financial Data / Finaeon (Bryan Taylor, 2018). These are commercial, not peer-reviewed, and are used as cross-checks or where nothing else was found:
  - "Four Centuries of Global Leadership"
  - "350 Years of English Leadership"
  - "Two Centuries of American Leadership"
  - "The First Billion-Dollar Company"
  - All at https://finaeon.com/.
  - Their LNWR figures match the IMM to within 0.3% (1890: 67.70 v 67.74; 1900: 76.05 v 76.26), and their Bank of England figures for 1830, 1850 and 1860 match the *Gentleman's Magazine* prices.
- "Le mouvement des actions de la Banque de France", *Journal de la Société de statistique de Paris* 25 (1884), p. 48. https://numdam.org/item/JSFS_1884__25__48_0/
- A. Neymarck (1893), "Une nouvelle évaluation du capital et du revenu des valeurs mobilières en France", *JSFS* 34, pp. 196-247. https://numdam.org/item/JSFS_1893__34__196_0/ It gives the 1848 and 1871 lows, the Jan 1882 crash, and the aggregate value of French railway shares at end-1891 (4,383m fr).
- A. Courtois, *Des opérations de bourse* (1859) and *Manuel des fonds publics* (c.1874 ed.), archive.org. Used for share counts of the Banque de France and Suez.
- Time magazine (1924-26) reports of AT&T results, giving average par stock for 1922-25.
- Not obtainable here:
  - Le Bris and Hautcoeur (2010), the Paris blue-chip index: HAL is behind an anti-bot wall.
  - Acheson, Hickson, Turner and Ye (2009), London 1825-70 data: on ESDS, not accessible.
  - Schmitz (1995).
  - Wardley (1991).
  - Gallica (blocked).
  - The FCC 1939 telephone report (OCR missing).

## Results in brief (top 3, approximate US$ m, see top10.csv)

| year | 1 | 2 | 3 | completeness |
|---|---|---|---|---|
| 1825 | Bank of England £30.9m | East India Co £14.9m | 2nd Bank of US $40.1m | Banque de France and Austrian National Bank not valued |
| 1830 | BoE £28.6m | EIC £13.2m | 2nd BUS $45.5m | same |
| 1850 | BoE £30.9m | EIC £16.1m | ? | **railways missing** |
| 1870 (Dec 1869) | LNWR £36.9m | BoE £34.4m | PLM £31.9m | London list, some missing |
| 1890 | LNWR £67.7m | BoE £48.4m | PLM £47.3m | BdF missing |
| 1900 | Standard Oil ~$803m | LNWR £76.1m | Suez £55.9m | fairly complete |
| 1910 | (GFD: Standard Oil $609m) | LNWR £59.2m (GFD) | ? | **not read; use 1907/1912** |
| 1912 | US Steel $740m ($346m common only) | Pennsylvania RR $504m | Canadian Pacific $476m | AT&T missing |
| 1913 | Canadian Pacific $535m | Pennsylvania RR $497m | Suez £77.7m | **partial** |
| 1925 | AT&T $1,319m | US Steel $1,145m ($690m common only) | Standard Oil NJ ~$1bn | GM, GE and du Pont missing |

## Gaps (biggest first)

1. **1846-1868: no railway values at all.** I did not find LNWR, Midland, GWR, Paris-Orléans, Nord, PLM or Paris-Lyon values for 1846-68. The world top 10 for 1845-1865 therefore lists only the Bank of England and the EIC (and NYC 1860), and is flagged PARTIAL.
   - GFD says the Bank of England stayed world No. 1 until about 1865, when LNWR passed it.
   - The French railways of the 1850s-60s were probably comparable to the Bank of England. In Dec 1869 PLM alone was £31.9m against the Bank's £34.4m.
   - Probable sources: IMM's predecessors, the *Course of the Exchange*, *The Economist*'s Railway Monitor, Courtois' *Tableaux des cours* (Gallica), and Acheson et al.'s ESDS dataset.
2. **Banque de France.** It is probably top 10 from 1825 to about 1890: about 1,500-2,000 fr x 67,900 shares in the 1830s, and 182,500 shares from 1857. Only 1848, 1871, 1881-82 and 1893 points were found.
   - It is NOT in the top10 lists after 1869. This probably understates it: for example, a January 1881 value of 698m fr (£27.7m) would rank about 8th in 1880.
3. **Other early candidates not valued:**
   - Austrian National Bank (privately owned, about 50,000 shares).
   - Bank of Ireland (capital £2.77m Irish; IMM 1880 price about 310).
   - Nederlandsche Handel-Maatschappij (1824, fl 37m) and Société Générale de Belgique (1822).
   - Crédit Mobilier (1852-67).
   - Crédit Foncier and Union Générale: the 1881 Paris boom, with Union Générale at 3,200 fr on 200,000 shares, about £25m at its peak, before bankruptcy in Feb 1882.
   - The Austrian Staatsbahn.
   - Prussian private railways before nationalisation (1879-84).
   - Russian railways (Grande Société).
   I think none of these exceeded about £15m, apart from the Banque de France and possibly the Austrian Staatsbahn in the 1860s. I have not verified this.
4. **United States:**
   - AT&T before 1922: share counts not found, and it is probably top 10 from about 1910.
   - GE, GM, du Pont, Sears and Woolworth for 1915-1925.
   - UP, SP, NYC and Atchison after 1913.
   - Western Union: never near the world top 10. Its $41m capital in 1866 and $80m in 1881 (both from memory, not verified) at prices of 45-100 give at most about $65m.
   - The components of US Steel in 1901 (Federal Steel, National Tube, American Steel & Wire and others) were not valued.
5. **Standard Oil before 1890 and 1913-1919.** Only GFD values are used (OTC and curb, not on the NYSE until 1920).
   - GFD dates some steps of the break-up to Sep 1912. In fact the Supreme Court decision was 15 May 1911 and the subsidiaries' shares were distributed from 1 Dec 1911. "Ex-subsidiary" trading began in Sep 1911.
   - GFD's statement that Standard Oil became the first $1bn company in 1913 "before the U.S. government split it up" is inconsistent with its own dates. I use its Aug 1912 figure of 1,100 x 983,383 = $1.08bn for the whole group.
6. **Annual coverage.** Values are dense for the Bank of England (annual 1825-46 and 1878-80, decade points otherwise) and for the Second BUS (1825-35).
   - For most other companies the coverage is the IMM benchmark years only (1869, 1873, 1879/80, 1890, 1900, 1907, 1912, 1913, 1920, 1925).
   - The IMM second tables could fill every year cheaply: each December issue gives 3-4 earlier years' highs and lows plus the previous year's close. The issues for 1885, 1895, 1905, 1910, 1916 and 1921 are already downloaded and OCR'd in the scratchpad (b_1825_1925_downloads/imm).
   - Thin spots: 1893 (no reading), 1901 (only Standard Oil and the US Steel formation), 1921 (none).
7. **Uncertain readings:**
   - Royal Dutch share count in the London list (2,138,180 shares of fl 100, unchanged 1920-25): suspicious.
   - Fractions of £1 share prices in 1920 and 1925: Imperial Tobacco, BAT and Coats could be out by about ±5%.
   - The Nord share count: IMM prints £9,274,000 nominal from 1890, which implies 579,625 shares rather than 525,000. That version is given as `est_high`.
   - The South Sea Stock capital of £3,662,784 is from memory, not verified. Treat the south_sea rows as low confidence.

## Where sources disagree

- **LNWR:** IMM and GFD agree for 1890 and 1900. GFD's 1880 table entry "26.25" is a misprint for about 260 ($m). The IMM gives £55.6m including partly paid new stock, which is $271m.
- **US Steel at formation:**
  - The famous "$1.4bn" is authorised par capital including bonds. Its 1901 equity market value was about $765m (GFD); Wikipedia cites a Bureau of Corporations valuation of about $700m.
  - Its common stock alone was worth about $140m at the end of the 1907 panic.
- **The largest company in 1900-1925:**
  - GFD (common only): Standard Oil 1900-1910, Pennsylvania RR 1912-15, Standard Oil NJ 1916 and 1920, AT&T 1922 onwards.
  - GFD also says that National City Bank was largest in 1925, 1927 and 1928. That looks implausible against AT&T ($1.3bn in 1925); I have not used it.
  - With preferred stock included, US Steel leads in 1912 and 1920.
- **Royal Dutch Shell:** Hannah's 1912 figure ($187m) is for the whole group, so I have not added Shell T&T separately for 1912.

## Myths to avoid

- "US Steel was the first billion-dollar company." That is true only of par value plus bonds. On market value the first was Standard Oil (whole group), in Aug 1912 (GFD), and US Steel's equity passed $1bn only in 1916.
- "The East India Company was a giant until 1874." From 1834 India Stock was a fixed 10.5% annuity charged on Indian revenues. Its value (£12-16m) reflects bond-like pricing, not trading profits.
- "Standard Oil was broken up in 1913." The decision was in 1911 and the distribution in Dec 1911.
- "Penn Central was the largest company in 1912-15." That is GFD's anachronistic label; the company was the Pennsylvania Railroad (Penn Central dates from 1968).

## Possible story beats

- **Panic of 1825.** Bank stock fell from a 1825 high of 299 to 196 on 14 Dec 1825, about a third of the Bank's value.
- **The Second Bank of the United States** was the world's No. 3 company (about $45m) in 1830. It lost its charter in the Bank War, and by 1841 it had failed. Its shares fetched under $2 in the 1849 liquidation.
- **The East India Company**, which ruled India until 1858, was worth less than half of the Bank of England. In 1874 it was simply bought out at twice par (£12m) and dissolved.
- **France in the 1870s.** About 1880, French railways held three of the world's top seven places (PLM was No. 2 at £47m). Two state-guaranteed Indian railways were also top 10 in the 1870s.
- **The Paris boom of 1881.** Banque de France shares went from 3,825 fr in Jan 1881 to 6,880 fr in Oct 1881, and Union Générale crashed in Jan 1882.
- **Suez.** In 1875 Disraeli bought 44% of Suez for £4m, implying £9m for the whole company. By 1912 it was worth about £94m ($456m), the world's No. 4. Its London price of £262 per 500-fr share was about 13 times par.
- **Standard Oil**, never listed on the NYSE, was worth about $800m in 1900, twice the LNWR. Break-up made it more valuable: about $590m in Sep 1911 against $1.08bn for the parts in Aug 1912.
- **1912.** A railway (Pennsylvania) and a Canadian railway (CPR) were Nos. 2 and 3 in the world, and the Suez canal was No. 4.
- **1920-25.** The tobacco, oil and rayon companies (Imperial Tobacco, BAT, Shell, Courtaulds) displace the British railways at the top of London.

## Decisions for a human

1. **Preferred stock:** include it (Hannah; this file's default) or use common only (GFD; modern convention)? It matters for US Steel, Union Pacific and Atchison: common-only values are in the notes.
2. **Suez founders' parts and parts civiles:** include them as part of Suez (about +35%)?
3. **EIC 1834-74:** show it as a company vessel, or as an annuity (husk) from 1834?
4. **1845-1868:** keep the partial top 10s, or borrow railway values from another source before animating? As it stands, the film would show the Bank of England and the EIC alone, which is misleading for 1846-68.
5. **GFD:** accept its numbers as medium-confidence fill-ins? They are used for Standard Oil, NYC 1860-80, PRR 1915, AT&T 1925 and Shell 1920.
