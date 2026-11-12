# Era A: everything up to 1825 — research notes

## 1. What is in the files

| file | rows | content |
|---|---|---|
| companies.csv | 17 | every company with at least one value (plus three banks needed for 1800/1825) |
| values.csv | ~1,000 | nominal market values in **millions** of the source currency, arithmetic in `note` |
| events.csv | 40 | listings, mergers, splits, restructurings, nationalisations, dissolutions |
| top10.csv | ~220 | ranking at 43 sampled year-ends, 1372-1825 (see section 4) |

The values are capital (shares outstanding, or nominal stock) × market price. Before 1825 almost every price
is quoted as a **percentage of nominal capital**, so value = nominal capital × price% / 100. Where only a
year's high and low exist (Scott, Francis, Mortimer), `kind = annual_avg` holds the **midpoint of high and low**,
which is not a true average. The note gives both figures.

Kinds used: `year_end` (last quote of the year), `annual_avg` (mean of monthly quotes or midpoint of high and low),
`point` (single dated quote, month-end in 1719-1720), `peak`, `trough`, `est_low` and `est_high` (where the
capital basis is uncertain). Dates in 1719-1720 are Gregorian (N.S.). Scott's English dates are Old Style
(11 days earlier).

## 2. Currencies and units

- **NLG**: Dutch guilders (bank guilders / current guilders not distinguished; the agio was about 4-5% in 1720).
- **GBP**: pounds sterling.
- **FRF**: **livres tournois** before 1795, francs (germinal, 1803) after. In 1719-1720 Mississippi prices are in
  livres that were increasingly paid in depreciating Banque Royale notes. **Convert each month with the exchange
  rate written in the note**, not with an annual livre rate. Monthly rates from the Yale 1720 database:
  London on Paris (pence sterling per écu of 3 livres): Nov 1719 25.6 (28 L/£), Mar 1720 17.9 (40 L/£), Jun 14.9 (48),
  Jul 12.9 (56), Aug 9.5 (76), Sep 8.3 (87). Amsterdam cross rates: Dec 1719 ≈ 28 L/£, Jan 1720 ≈ 30, Feb ≈ 34,
  Dec 1720 ≈ 60 L/£.
- **XAG_KG**: kilograms of fine silver (Bazacle only). The `value` column is in millions of kg, so 0.00053 = 531 kg.
  The source gives prices in silver and silver is the natural deflator before 1600. Conversion helpers:
  sterling 1601-1816 = 111.4 g fine silver per £ (so 1 kg ≈ £8.98); livre tournois 1726-1785 ≈ 4.45-4.5 g;
  franc germinal = 4.5 g.
- **USD**: US dollars (two US banks).

### Approximate FX used only to rank top10.csv
GBP 1 = NLG 10.5 (Amsterdam on London was about 33-36 schellingen Flemish per £ throughout 1600-1795).
Livres per £: 28 before 1719, monthly Yale rates in 1719-20, 33 in 1721-25, 24.5 in 1726-94; francs 25.2 per £ after 1795.
USD 4.44 per £. Silver £8.98 per kg. Rankings are robust to these choices except where noted.

## 3. Company notes, sources and doubts

### Bazacle (Toulouse, 1372-1946)
- Source: Le Bris, Goetzmann & Pouget, NBER w20199 (2014), published as "The present value relation over six
  centuries: the case of the Bazacle company", JFE 132(1), 2019. The paper prints no price table. I **extracted
  the year-end price per uchau (kg silver) from the vector data of Figure 3** in the NBER PDF: 333 price points,
  1372-1946. Calibration checks: 1709 = 0.874 kg (the paper says 878 g); the 1709 dividend = -3.3 kg (the paper
  says pariers paid 3-5 kg in disaster years). The text's 1816 low of 8.1 kg does not match the figure
  (1814 = 9.0 kg, 1816 = 22.5 kg), probably because the figure shows the last price of each year.
- Shares (App. 7.1): 80 (1372), 88 (1374), 96 (1384), 100 (1535), 128 (1714), 136 (1804), split 1:4 in 1888.
- Results: whole firm ≈ 531 kg silver at the 1372 merger; 4.4 t in 1589 (its pre-1825 high); 87 kg after the 1709
  ice flood; 3.7 t in 1822. At the 1601-1816 sterling silver standard that is ~£4.8k (1372) and ~£33k (1822):
  **tiny but measurable**.
- Gaps: 1451-1522 (none, apart from 1523 and 1527), 1823-1828 (register lost). The 1825 row uses the 1822 price.

### VOC (1602-1799)
- Capital f6,424,588 (Van Dillen 1958; Gaastra). Petram's thesis gives f6,429,588 (Den Heijer). The 0.08%
  difference does not matter. It never changed. Price = % of par of **Amsterdam-chamber** shares, applied to all
  chambers (Middelburg and the smaller chambers traded lower: a slight overstatement).
- 1602-1698: Petram (2011) Appendix A, 851 spot prices averaged by month (322 months), **ex-dividend**. Parsed
  from the thesis PDF. Missing years: 1606, 1612-13, 1615-16, 1620-21, 1624-31, 1645-46, 1654-56, 1662, 1690, 1693,
  1695-96.
- 1699-1718: **no data** (GFD's chart draws a straight line 1698→1719).
- Nov 1719-Dec 1720: monthly, Leydse Courant (Frehen, Goetzmann & Rouwenhorst database, column "OIC").
- 1721-1795: **read off the Global Financial Data chart (VOC1-AM)** published by Finaeon, by pixel digitising.
  The chart's time axis is ~0.65 yr early relative to the Yale data (its 1720 spike sits at early 1720), so I
  shifted it. The 1719 GFD values sit ~10% above Yale's Leydse Courant quotes. Treat as ±5-10% and ±1 year.
  GFD's 17th-century VOC series is on a different, unexplained scale (e.g. ~110 against Petram's ~525 in 1650)
  and is not used.
- **The real peak**: August 1720, 1,210% → **f77.7m** (≈ £7.4-7.8m). Petram's blog gives "~1,200, just over
  f77 million". The 17th-century high was Aug 1649 (565% → f36.3m). October 1637: 297% → f19.1m.
- End: nationalised 1 Mar 1796 by the Batavian Republic, charter lapsed 31 Dec 1799. Last GFD quote ~125%
  (f8m). No dividends after 1782.

### WIC: first company (1621-1674) and second company (1674-1791)
- Capital f7,108,161 by 1623. Repeated calls raised invested capital to "just over seventeen million" by 1639
  (Odegard 2019, Itinerario). The first was a 50% call in 1624. It is **not known whether quotes are % of the
  original or of the paid-in capital**, so 1629-1649 rows are est_low (f7.1m basis) and est_high
  (f10.66m to 1638, f17m from 1639).
- Prices 1628-1650: GFD chart WIC1-AM, digitised. The peak of ~205% follows Piet Heyn's capture of the silver
  fleet (Sep 1628). Prices fell to ~15% by 1649-50. No prices 1651-1674.
- Second WIC: nominal f6,332,362 (Schneeloch, final 1710; Den Heijer: f4,505,863 initial). It "never traded above
  par except during 1688 and 1720" (Odegard). 1720 monthly from Leydse Courant: peak 575% (Aug) → f36.4m,
  **excluding the two 1720 subscriptions** (amounts not found). 1721-1791 from the GFD chart (~15-90%).
- Doubt: the 1629 WIC > VOC result rests on a chart reading and an unclear capital basis. Even the low basis
  (f14.6m) beats the VOC's ~f11-12m (VOC had no quotes 1624-31; 1623 178%, 1632 177-186%).

### English East India Company (eic; and eic_english)
- 1601-1656: Scott vol. 2. Separate voyages (1601-1612), then terminable joint stocks.
  - First Voyage £68,373 (1601).
  - First Joint Stock £418,691 (1613): ~143% in 1615, 208-218 in 1616-18, but after large capital repayments.
  - **Second Joint Stock (1617) subscribed £1,629,040**: called in 8 × 12.5%, at least half paid by 1620.
    Quotes: 115 (1617), 110-116 (1618), 80 (1624-27), 70-80 (1628).
  - Third Joint Stock £420,700 (1632): 80 (1634-35), 72 (1639), 91-95 (1640, after 50% divided back).
  - Partly paid stock makes the basis ambiguous, so 1618, 1624 and 1628 are est_low (paid-in plus premium) and
    est_high (as if fully paid).
- 1657-1709 (Old Company): New General Stock £739,782 subscribed, 50% paid (£369,891). 100% bonus in 1682 →
  £739,782; £744,000 (1692); £1,488,000 (1693); £1,574,608 (1698-1708). Scott's year highs and lows are used; the
  1685 high of 500% gives **£3.7m**.
- English (New) Company 1698-1709: stock £1,662,000; 126-219% in 1700-1703. It was briefly worth more than the
  Old Company.
- United Company: £3,163,000 (1709), £3,194,080 (1718-1786). 1709-1719 Scott; 1719-20 monthly from Castaing (Yale).
  The 1720 high is 449% (Scott, 28 Jun O.S.); Castaing's daily high in the Yale file is 420.
- **1721-1765: no prices found (major gap).**
- 1766-1798: Mortimer, *Every Man His Own Broker* (eds. 1775, 1782, 1791, 1798), which prints yearly India-stock
  lists with "average price" and extreme variation. The OCR is poor: fractions are often illegible, so integers
  are used, and the 1767-1770 assignments are unclear and omitted.
  - Bengal-bubble peak 278% (June 1768) → £8.9m.
  - Low 140% (Dec 1773); Jan 1784 118½.
- Capital increases to £4m (1786), £5m (1789) and £6m (1794) are from standard histories (Bowen) and **not
  re-verified here, except £6m in 1817** (Fortune's Epitome).
- 1799-1825: no prices except **Feb-Mar 1826 = 226-234%** (Asiatic Journal daily lists), used as the 1825
  year-end proxy (£13.8m).

### Bank of England (1694-)
- 1694-1719: Scott vol. 3 summary table (year high and low, capital history). 1694-96 quotes are premiums on
  partly paid stock; their basis is uncertain. The 1697 low of 51 is a cash price in the bank-note crisis.
- Jul 1719-Dec 1720 monthly from Castaing (Yale). Peak 270% in July 1720 (Scott: 265 on 24 Jun O.S.).
- **1721-1731: gap.**
- 1732-1825: Francis's table of annual highest/lowest Bank Stock (Chronicles of the Stock Exchange 1855,
  Gutenberg text; the same table is in Francis 1847). It cross-checks against two OCR copies of the 1847 book.
  1810 is printed as 276-273 in all copies, probably a misprint (low perhaps 255), reproduced as printed.
- Capital: 1.2m (1694), 2,201,171 (1697), 4,402,343 (1708), 5,058,547 (1709), 5,559,995 (1710), 8,959,995 (1722),
  9.8m (1742; Clapham, implied by Francis's 10% call to 10.78m), 10.78m (1746), 11,642,400 (1782),
  14,553,000 (1816 bonus).
- 1825: high 299 (early-1825 boom), low 196 (presumably the December panic) → £28.5-43.5m.

### South Sea Company (1711-1853)
- Capital: £9,177,968 (1711), £10m (1715), £11,746,844 (1719).
- 1720 issues (Scott vols. 1 and 3):
  - First money subscription 14 Apr, £2.25m nominal at 300; second, £1.5m at 400.
  - Irredeemable annuities converted at 375 in May (~£3.5m stock). Scott: "in May 1720 there was issued
    18½ millions".
  - Third subscription 17 Jun, £5m nominal at 1000 with 10% down; fourth (Aug) £1.25m at 1000; redeemables and
    remaining annuities converted in Aug. All were re-priced at 400 in Sept.
  - Authorised/issuable stock £37.8m; about £32m in 1722 (Scott) and £33.8m by 1723 (standard figure).
- **Peak value is a human decision**:
  - 1,050% (Scott/Freke, 24 Jun O.S.) on fully-paid stock (£15.25m) = **£160m**.
  - On all stock issued by May (£19.0m) = **£199m**.
  - Castaing's high in the Yale data is only 950% → £145m.
  - Counting the 10%-paid third subscription at full price gives ~£250m (overstated).
  - I suggest **~£160-200m**. Popular "£420m" figures are not supported.
- Monthly 1720 rows are est_low (fully paid) and est_high (incl. partly paid first two subscriptions). Both
  understate Aug-Dec 1720, when later conversions added stock.
- **No prices after Dec 1720** (gap). Half the stock became annuities in 1723; from 1733 the trading stock was
  £3,662,784 (Fortune 1817). Thereafter it was in substance a government annuity paying 3½-4%.

### Mississippi Company / Compagnie des Indes
- Source for shares and capitalisation: **Velde (2003), "Government equity and money: John Law's System in 1720
  France", Chicago Fed WP 2003-31**. Daily prices to Nov 1720 are from the Yale database, which compiles Velde
  (Gazette d'Amsterdam 1718-19), Murphy (1997) and the Leydse Courant.
- Shares (all livres):
  - 200,000 mothers (1717-18, 500 L par, paid in billets d'État).
  - +50,000 daughters (Jun 1719, 550 L) → 250,000.
  - +50,000 granddaughters (Jul 1719, 1,000 L) → 300,000.
  - +300,000 *soumissions* (Sep-Oct 1719, 5,000 L, paid in instalments, option-like).
  - 600,000 shares at 12 Mar 1720 after conversion.
  - 200,000 by 3 Jun 1720, after buy-backs and the King's gift of his 100,000. Reverse split 3:2 → ~133,000.
  - 125,000 presented to the 1721 Visa and cut to 55,735.
- **Peak capitalisation (Velde fn. 32, citing du Hautchamp): 300,000 shares at the 9,925 L support price +
  300,000 subscriptions at 6,600 L = 4.96 bn L** (Dec 1719-Feb 1720), ≈ £150-170m at the month's exchange rates.
  The popular 624,000 × 10,000 L = 6.2 bn (or Scott's "500-600 millions sterling in paper") counts partly paid
  subscriptions, and the King's and Company's own holdings, at full price.
- The brief named "NBER w21693" as the source for overstated share counts. **w21693 is actually Goetzmann,
  "Bubble Investing: Learning from History" (2015)**, which only summarises Velde. The correction comes from Velde.
- After 1720: 1722-23 about 56,000 shares × 1,000-1,200 L (56-67m L).
- **1724-1769: no prices collected (major gap)**. Shares were quoted in Paris throughout; see Haudrère (2005) and
  the *Affiches de Paris* from 1745 (Velde & Weir 1992 used Paris bourse quotes for 1746-1793).

### Smaller 1690-1720 companies (Scott; Yale)
- Royal African Company: £111,100 (1672) → £1.1m (1697) → written down to £451,350 (1712). 1720 values use
  £451,350; any 1720 new stock is not included.
- Million Bank: £500,000.
- Hudson's Bay: £10,500 → £31,500 (1690) → £103,950 (1720); 1692-98 prices from Scott.
- Royal Exchange Assurance: quotes per £100 nominal; nominal £1.152m then £1.212m with 10% paid. Peak 225 (Aug)
  → £2.7m on £121k paid in.
- London Assurance: 19,740 shares × price; the share count before Sept 1720 is uncertain.

### US and French banks (only so 1800/1825 rankings are not empty)
- First BUS: par value only ($10m).
- Second BUS: 350,000 × ~$120 ≈ $42m in 1825 (MoAF: "vicinity of $120 for over a decade"; Jan 1836 $116). Approximate.
- Banque de France: par only (30m F in 1800). Prices: Manas (2019), *Banque de France's shareholders 1800-1945*.

## 4. top10.csv: how to read it
- 43 sampled year-ends (1372, 1400, 1450, 1550, 1600, 1602, 1610 ... 1825). For each, a company's value is its
  year_end; else a late-year point; else annual_avg; else the mid of est_low/est_high. If the company has no
  value that year, the nearest value within ±3 years is used and flagged ("nearest year used").
- Ranking uses the approximate FX above. The value column stays in source currency.
- **Before 1825 the universe of priced companies is tiny** (1-6 names in most years). "Top 10" here means
  "top of the companies with known prices". Before 1602 there is only the Bazacle.
- Rows with `rank = nr` mark giants alive in that year with **no price found**:
  - EIC 1730-1760 and 1810-1820.
  - South Sea 1730-1825.
  - Compagnie des Indes 1730-1760.
  - VOC 1705-1715.
  - first WIC 1660-1670.
  - Their absence makes 1730-1760 rankings incomplete: EIC (£3.19m capital × typically 150-200%) and the
    Compagnie des Indes were probably #2-#3 behind the Bank of England.
- Book values without a price (HBC 1720, RAC 1712) are not ranked.

### Top 3 at key dates (≈ £ at the FX above)
| year-end | 1 | 2 | 3 |
|---|---|---|---|
| 1610 | VOC f8.0m (£0.77m) | EIC voyages, book value £0.15m | Bazacle ~1.9 t silver (£17k) |
| 1640 | VOC f26.0m (£2.5m) | WIC f8-19m (£0.8-1.8m) | EIC Third Joint Stock £0.39m |
| 1700 | VOC ~f32.8m (1698 quote, £3.1m) | Bank of England £3.0m | English (New) EIC £2.3m (Old EIC £1.6m) |
| 1720 peak (Jan-Aug) | South Sea £160-200m (Jul) | Mississippi 4.96bn L ≈ £150-170m (Jan-Feb) | Bank of England £15.0m (Jul); EIC £14.3m (Jun-Jul); VOC f77.7m ≈ £7.5m (Aug) |
| 1720 / 1721 | South Sea ~£28-35m (Dec 1720) | Mississippi 728m L (Nov 1720, ≈ £12m) | Bank of England £7.9m (VOC £5.1m, EIC £5.1m) |
| 1750 | Bank of England £14.4m | VOC f27m (£2.6m) | (EIC, South Sea, Compagnie des Indes unpriced) |
| 1800 | Bank of England £19.2m | EIC ~£9.5m (1797 quote) | Banque de France 30m F at par |
| 1825 | Bank of England £28.5-43.5m | EIC ~£13.8m (Feb-Mar 1826 proxy) | Second BUS ~$42m (£9.5m) |

## 5. Story beats
1. **The VOC peaked in August 1720, not 1637.** f77.7m, about £7.5m (≈ 750-830 tonnes of silver).
   The "$7.9-8.3 trillion" figure (Motley Fool 2012, howmuch.net, lovemoney) applies 1720 prices to 1637 and then
   inflates them absurdly. Petram puts the real 1720 value at about €0.8-1bn today. In 1637 the VOC was worth f19m.
2. **At the 1720 peak the South Sea Company (£160-200m) and the Mississippi Company (~£150-170m) were each about
   20× the VOC's all-time high.** The Bank of England (£15m) and the EIC (£14m) were each about 2× the VOC.
   Even two marine insurers with £0.1m paid in (REA, London Assurance) briefly reached ~£2.7m each, a third of
   the VOC.
3. **The English EIC rivalled the VOC long before 1700**:
   - In 1617-1620 the Second Joint Stock's £1.63m subscription dwarfed the VOC's f6.4m (≈ £0.6m) capital;
     market values were comparable (£0.8-1.9m against £0.9-1.1m).
   - In 1685 the EIC at 500% (£3.7m) was worth more than the VOC (f29m ≈ £2.8m).
4. **1629: the West India Company may have been the world's most valuable company** for a few months after
   Piet Heyn captured the Spanish silver fleet: f15-22m against the VOC's ~f11-12m. Low confidence.
5. **The "New" English East India Company overtook the Old one (1700-1705)** before they merged in 1709.
6. **The Mississippi share-count myth**: 624,000 × 10,000 L = 6.2 bn overstates. Velde's 4.96 bn counts
   partly paid subscriptions at their own price. In real terms, Paris prices were paid in notes that lost ~90%
   by late 1720.
7. **Bazacle**: the oldest priced company. A 600-year-old firm worth only 0.5-4.4 tonnes of silver, it sank to
   87 kg when ice destroyed its dam in 1709. It would be a single grain next to the VOC.
8. **1730-1825: the Bank of England is the giant** (£12-36m). The EIC is second when priced; its Bengal-bubble
   peak is 278% (Jun 1768), followed by the 1769 and 1772-73 crashes.

## 6. Earlier than the Bazacle: shares without prices (narration only)
- **Roman publicani** (2nd-1st c. BC): tax-farming companies with *partes* (shares) held by investors. Cicero
  (In Vatinium 29) mentions *partes illo tempore carissimae*, "shares were very expensive at that time".
  No numbers survive.
- **Stora Kopparberg** (Falun, Sweden): a 1288 deed transfers a one-eighth share in the copper mountain. Often
  called the oldest share document. No price series.
- **Toulouse mill companies**: the Bazacle's predecessors (15 prices before 1372) and its rival the Château
  Narbonnais (Castel Narbonnais), whose share prices 1375-1450 moved with the Bazacle's (Sicard 1953).
- **Casa di San Giorgio** (Genoa, 1407-1805): its *luoghi* were traded and paid variable dividends (*paghe*);
  Goetzmann (w21693, Fig. 1, data from M. Fratianni) shows a 16th-17th-century price index. It is a corporation
  of state creditors, a debt vehicle rather than a business. **Human decision**: if counted, it would be by far
  the largest "company" before the VOC (tens of millions of lire of luoghi). I did not quantify it.
- **German mining Kuxe** (15th-16th c.): traded mining shares. Luther (1554): "Ich will kein Kuks haben!" No series.
- Other early joint stocks without price series: Muscovy Company (1555, 240 shares of £25), Mines Royal and
  Mineral & Battery Works (1568), Virginia Company (1606/1609), New River Company (1619; very valuable shares in
  the 18th-19th c.; Scott vol. 3 has prices to 1720, not extracted).

## 7. Gaps and how to fill them (priority order)
1. **Larry Neal's Course of the Exchange data (ICPSR 1008; also on the European State Finance Database at
   esfdb.websites.bbk.ac.uk)**: London prices 1698-1823 for the Bank of England, EIC and South Sea, plus
   Amsterdam 1723-1794. It would fill EIC 1721-1765 and 1799-1823, South Sea 1721-1823 and the Bank of England
   1721-1731, and give true monthly 1721-1722 values. ICPSR needs a login, and **esfdb.websites.bbk.ac.uk was not
   reachable from this session (DNS blocked by the network allowlist)**. Allow it in the session's network
   settings to retry.
2. Van Dillen (1931), "Effectenkoersen aan de Amsterdamsche Beurs 1723-1794" (Economisch-Historisch Jaarboek 17):
   proper VOC and WIC series to replace the GFD chart readings.
3. VOC 1699-1718: no source found (Amsterdam price currents; Gelderblom & Jonker).
4. Compagnie des Indes 1723-1793: Haudrère (2005); Velde & Weir (JEH 1992) Paris quotes from 1746.
5. 1780s-1825 candidates **not quantified**, which may belong in the 1780-1825 top 10:
   - Caisse d'Escompte (1776-1793): 5,000 shares of 3,000 L at founding; ~12,440 L in June 1787 per Wikipedia;
     share count changed in 1787-89.
   - Nouvelle Compagnie des Indes (1785-1794).
   - Banque de Saint-Charles (Madrid 1782; 300m reales nominal).
   - Danish Asiatic Company (1732-1840s); Swedish East India Company (1731-1813).
   - Bank of Ireland (1783; capital IR£3m by 1821).
   - London Dock (1800) and West India Dock (1799) companies.
   - British canals (e.g., Grand Junction).
   - Nederlandsche Handel-Maatschappij (1824).
   - 1824-25 mining bubble (Real del Monte, Anglo-Mexican).
   - Banque de France prices; First BUS prices (Sylla-Wilson-Wright database of early US securities prices).
6. Dutch 1720 IPOs (Stad Rotterdam, Middelburg companies) are in the Yale file but their share counts are
   unknown; not quantified.

## 8. Decisions for a human
- South Sea 1720 basis: fully paid stock only (£160m at peak) vs all issued stock (£199m) vs Castaing's 950 (£145m).
- Mississippi: use Velde's 4.96 bn L (recommended), or 300,000 originals × price as est_low, and which monthly FX.
- WIC 1629-1649: original (f7.1m) vs paid-in (f10.7-17m) capital basis.
- EIC 1617-1628: treatment of the partly paid Second Joint Stock.
- Whether to accept chart-read GFD values (VOC 1721-1795, WIC 1628-1650 and 1721-1791; low confidence) or show
  those years as gaps.
- Whether the Casa di San Giorgio (Genoa) should enter the film.
- Bazacle is in kg of silver; convert with a silver price or the sterling silver content.

## 9. Full references
- Le Bris, D., W. N. Goetzmann and S. Pouget (2014), "Testing asset pricing theory on six hundred years of stock
  returns: prices and dividends for the Bazacle company from 1372 to 1946", NBER WP 20199; published as "The
  present value relation over six centuries: the case of the Bazacle company", *Journal of Financial Economics*
  132(1), 2019, 248-265.
- Petram, L. O. (2011), *The World's First Stock Exchange: how the Amsterdam market for Dutch East India Company
  shares became a modern securities market, 1602-1700*, PhD thesis, University of Amsterdam (Appendix A monthly
  prices). Petram, "Was the VOC the most valuable company ever? (Answer: NO!)", worldsfirststockexchange.com, 2020.
- Frehen, R., W. N. Goetzmann and K. G. Rouwenhorst (2013), "New evidence on the first financial bubble",
  *JFE* 108(3), 585-607 (NBER WP 15332); data: Yale ICF "South Sea Bubble 1720" price database
  (1720PriceDatabaseOctober5_10version.xlsx).
- Goetzmann, W. N. (2015), "Bubble Investing: Learning from History", NBER WP 21693.
- Velde, F. R. (2003), "Government equity and money: John Law's System in 1720 France", Federal Reserve Bank of
  Chicago WP 2003-31.
- Scott, W. R. (1910-12), *The Constitution and Finance of English, Scottish and Irish Joint-Stock Companies to
  1720*, 3 vols, Cambridge (archive.org scans constitutionfina01/02/03scotuoft).
- Francis, J. (1847), *History of the Bank of England*, appendix table; Francis, J. (1855), *Chronicles and
  Characters of the Stock Exchange* (Project Gutenberg 70656).
- Mortimer, T., *Every Man His Own Broker*, eds. 1775, 1782, 1791, 1798 (ECCO scans on archive.org).
- Odegard, E. (2019), "Recapitalization or Reform? The Bankruptcy of the First Dutch West India Company and the
  Formation of the Second West India Company, 1674", *Itinerario* 43(1), 88-106.
- Global Financial Data / Finaeon (2018), "Data for Amsterdam stocks from the 1600s and 1700s added to GFD";
  "The First and the Greatest: the rise and fall of the VOC"; "An index for the first two centuries of stock
  markets" (charts digitised for VOC1-AM and WIC1-AM).
- *Fortune's Epitome of the Stocks and Public Funds* (1817 ed.); *Asiatic Journal and Monthly Register* vol. 21 (1826).
- Museum of American Finance, "Second Bank shares" exhibit.
- Not consulted but recommended: Neal, L. (1990), *The Rise of Financial Capitalism*, and ICPSR 1008; Van Dillen
  (1931); Haudrère (2005); Velde & Weir (1992), JEH 52(1); Gelderblom, de Jong & Jonker (2013), JEH 73(4);
  Dickson (1967) *The Financial Revolution in England*; Bowen (2006) *The Business of Empire*; Manas (2019) on
  Banque de France shareholders.
