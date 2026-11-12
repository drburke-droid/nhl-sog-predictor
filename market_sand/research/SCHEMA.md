# Research file format (all four eras use the same files)

Each era folder holds four files. UTF-8 CSV, comma-separated, quote fields that contain commas.

## companies.csv
id,name,aka,country,exchange,industry,first_value_date,end_date,fate,successors,top10_years,confidence,notes

- id: stable lowercase slug of the legal company at its first appearance (e.g. voc, eic, bank_of_england,
  south_sea, mississippi_co, standard_oil_nj, att, ibm, ntt, microsoft, nvidia). If the same legal
  company is later renamed, keep the first id and list the later names in aka ("Exxon (1972); ExxonMobil (1999)").
  Ids we already fixed (use exactly these if the company appears): voc, wic, eic, bank_of_england, south_sea,
  mississippi_co, hudsons_bay, bazacle, standard_oil_nj, socony (Standard Oil of New York, later Mobil),
  att, us_steel, ge, ibm, gm, ntt, microsoft, apple, nvidia, aramco.
- industry: exactly one of trade, finance, rail, oil, mining_metals, autos, industrial, chem_pharma, consumer,
  telecom_media, computing, internet, semis (primary industry at the company's peak; if it changed, give
  the change and its date in notes).
  trade = chartered trading companies and shipping lines; finance = banks, insurers, exchanges, holding
  companies like Berkshire; rail = railways and canals; industrial = electrical, machinery, utilities,
  conglomerates; consumer = retail, food, drink, tobacco, luxury, household goods; computing = hardware and
  software (IBM, Microsoft, Apple); internet = Google, Amazon, Meta, Tencent, Alibaba; semis = chips.
- country: home country of the listing; exchange: main exchange (Amsterdam, London, Paris, NYSE, Tokyo...).
- fate: alive, bankrupt, nationalised, dissolved, acquired, merged, broken_up, taken_private, delisted.
- successors: ids that received its value (semicolon-separated), if any.
- top10_years: years in which it was among the world's 10 most valuable listed companies (ranges like 1920-1935).
- confidence: high / medium / low for the values overall.

## values.csv
id,date,value,currency,kind,source,note

- date: YYYY-MM-DD, YYYY-MM or YYYY. value: market capitalisation in millions of the stated currency
  (shares outstanding x price). currency: USD, GBP, NLG (guilders), FRF (livres/francs, say which in note),
  JPY, EUR, CNY, ... Give the value in the currency the source uses; if the source also gives a USD
  figure, add a second row with currency USD and kind same.
- kind: year_end, annual_avg, peak, trough, point, est_low, est_high.
- At least one value a year while the company is in or near the world top 10 (more around bubbles and
  crashes: 1720, 1929, 1987-90, 2000, 2008, 2020, 2022, and monthly for wild years), plus its value at the
  start and end of its life, and at each merger, split or failure.
- source: short citation and URL where possible.

## events.csv
date,type,from_ids,to_ids,detail,source

- type: listing, rename, merger, acquisition, split, spinoff, bankruptcy, nationalisation, dissolution,
  taken_private, delisting.
- For mergers/acquisitions give the target's value at the deal; for splits and spin-offs give the
  children's first values (or shares of the parent) in detail.

## top10.csv
year,rank,id,value,currency,source

- The world's 10 most valuable listed companies at (or near) each year-end of the era, as far as the
  evidence allows. Where the world list is unknown, give what is known (e.g. the US or UK top 10) and say
  so in notes.md.

## notes.md
Method, sources (full references), gaps, ranges where sources disagree, myths to avoid, and anything
that needs a human decision.
