# Forbes-reported net worths, 1987–2001: notes

Files: `values.csv` (407 rows, 127 names), `corgis_billionaires.csv` (the CORGIS/PIIE data, unmodified).
Builder script: `../scripts/build_forbes87.py`. Every value is one I saw in a cited source. A blank year means no source was found.

## Conventions
- `list=world`: Forbes "World's Billionaires". The first edition was the 5 Oct 1987 issue (month 10). From 1988 to 2001 it ran in a July cover-dated issue, released 2–3 weeks earlier: 1998–2001 came out about 15–22 June. Stock prices were taken about a month before release (1999 cut-off: 30 Apr; 2001: 21 May).
- `list=forbes400`: issue cover-dated in October; values as of late August.
- `list=other`: Forbes figures published outside the annual lists:
  - the Forbes Global piece of 20 Mar 2000 on Son and Gates;
  - the Son peak quoted with the 2000 list;
  - Forbes's recalculation after 9/11 (to 24 Sep 2001) for Gates, Buffett and Allen.
- Family entries are kept as Forbes printed them ("Walton family", "Karl & Theo Albrecht", "Hans & Gad Rausing", "Kwok brothers"). Do not add a family row to its members' rows in the same year.
- Where a contemporaneous AP/UPI figure and the Wikipedia table disagree, I used the AP/UPI figure and put the other in the row's `NOTE`. Where one is just a rounding of the other, I used the more precise one.
- One derived value, flagged `DERIVED`: Ellison, Forbes 400 1999 = 13. Reuters said he rose $45bn to $58bn.

## Sources used, and how reliable they are
- Wikipedia "The World's Billionaires" top-10 tables. This is the only source here for 1987 and 1992–1994, and part of 1991, 1996 and 1998. It agrees with AP within about 0.1–0.5 in most years, but has a few mismatches (listed below).
- AP and UPI stories on each list, via the Deseret News, Spokesman-Review, CBS and UPI archives. Coverage: world lists 1989–1991, 1995, 1997, 1999–2001; Forbes 400 lists 1988, 1991, 1993–1997 and 1999–2001.
- Christian Science Monitor, 2 Oct 1997: the Forbes 400 top 40.
- Honolulu Star-Bulletin, 2002: a table with the 2001 Forbes 400 values.
- CORGIS billionaires CSV (Freund & Oliver, PIIE), years 1996 and 2001:
  - 2001 matches the June 2001 world list exactly. It agrees with AP and with the Kaggle "Forbes billionaires 1997–2023" set, whose rows come from Forbes's own site; no value differs, though the Kaggle set has only 335 rows.
  - 1996 is a hybrid. Non-US entries match the July 1996 world list. US entries match the Forbes 400 of 14 Oct 1996, not the July list: Gates 18.5 against 18.0 in July, Buffett 15.0 against 15.3. Allen 7.5, Kluge 7.2 and Ellison 6.0 all equal the AP figures for the Forbes 400. So I tagged CORGIS-1996 US rows `forbes400`, month 10.
  - CORGIS 1996 has no Prince Alwaleed.
  - CORGIS 2001 labels the Tetra Pak heir at 9.0 as "Kirsten Rausing"; the Kaggle set (from Forbes) says "Birgit Rausing & family", so the row is renamed.
- Not reachable: forbes.com (403) and web.archive.org (connection refused), so I did not read the original Forbes pages or the Forbes "History" articles. The Roanoke Times and Virginian-Pilot archive pages (scholar.lib.vt.edu) now return 410. For 1993 and 1994 I used their text as quoted in the search index.

## Coverage by person (w = world list, f = Forbes 400, o = other)
- Gates: f1987, f1990, f1991, w+f every year 1992–2001, o2000-03, o2001-09. Not found: world 1987–1991 (he was below the published top tens), f1988, f1989.
- Buffett: w1989, f1990, f1991, w+f 1993–1997, w1998, w+f 1999–2001. Not found: f1987–89, f1992, f1998.
- Paul Allen: f1991, f1995, f1996, w+f 1997, w1998, w+f 1999–2001.
- Ellison: f1995, f1996, f1997, w1999, f1999 (derived), w+f 2000, w+f 2001. No world value found 1996–1998.
- Ballmer: f1992, f1996, w+f 1997, w+f 1999, w+f 2001.
- Dell: f1996, w+f 1997, w1998, w+f 1999, w2000, w2001.
- Gordon Moore: f1996, f1997, w2001.
- Sam Walton (died April 1992): w1987 7.5, f1987 8.5, f1988 6.7, w1989 8.7, f1991 4.4. Forbes 1989/1990: AP says the fortune was "credited jointly to the family" in 1989, and Sam ranked No.11 in 1990, but no figure was found.
- Walton family, world list: 1990–1998, from 13.3 to 48.0.
- Walton heirs (S. Robson, John T., Jim C., Alice): f1991 4.4; f1996 about 4.7–4.8; f1997 6.3–6.5; w2001 18.5–18.8; f2001 17.5.
  - Helen: f1996, f1997, w2001, f2001.
  - S. Robson alone: w1999 15.8 and w2000 20.0. AP says Walton members Nos. 8–11 were at $16bn each in 1999, without naming them.
  - Christy Walton: nothing before 2002 (she inherited after 2005).
- Koch brothers: 1996 (2.0 each) and 2001 (3.2 each) only.
- Mars: family w1990, w1991, w1993, w1997, w1998. Individuals f1996 3.0, f1997 3.3, w2001 9.0 each.
- Kenneth Thomson: 1990–1992, 1995–1998, 2000, 2001.
- Albrechts (joint entry): 1994, 1996–2001. Rausings: 1987, 1990–1992, 1994–1997, 2001.
- Kamprad: 1996 1.7 and 2001 13.0. Forbes valued IKEA low until about 2000.
- Bettencourt: 1996, 1997, 1999, 2001.
- Li Ka-shing: 1995–1997, 1999, 2001. Lee Shau Kee: 1995–1998, 2001. Kwok brothers: 1996, 1997, 2001.
- Tsutsumi: every year 1987–1997, plus 2001 (2.8). Taikichiro Mori: 1987–1992 (he died Jan 1993). Minoru & Akira Mori: 1993. Akira Mori: 1996, 2001.
- Son: w1996, w1999, o2000-02, o2000-03, w2000, w2001. Alwaleed: 1997–2001.
- Slim: 1994, 1996, 2001. Ortega, Arnault, Pinault, Premji, Adelson, Jobs, Shari Arison: 2001 only.
- Persson, Otto, Johanna Quandt, Mittal, Plattner, Edward Johnson III, Micky Arison: 1996 and 2001. Ted Arison: 1996.
- Anschutz: 1996, 1997, 1999, 2001. Redstone: f1988, f1995–1997, w1999, w2001. Kluge: 13 values, 1988–2001.
- Cox sisters: joint 1990; f1996, f1997, w2001.
- Kerkorian, Murdoch, Perot, DeVos/Van Andel, Abigail Johnson: 3–5 values each.
- Also included because they were in the top 25 in a year with a usable list: du Pont, Reichmann, Shin Kyuk-ho, Watanabe, Nakajima, Itoyama, Kobayashi, Yoshimoto, Iwasaki, Osano, Mahfouz, Irving, Haub, Haniel, Sainsbury, Newhouse, Bass, Brenninkmeyer, Quandt family, Sacher/Oeri/Hoffmann, Tsai Wan-lin, Chung Ju-yung, Haas, Pritzker, Engelhorn, Cargill, Wonowidjojo, Nina Wang, Tan Yu, Mulliez, Kuok, Kwek, Cheng Yu-tung, Haefner, Kirch.
  - Not systematically covered: the full top 25 for 1987–1989, 1992–1996, 1998 and 2000. Only top tens, or fewer, were found for those years.

## Conflicts and how I resolved them
- Sam Walton 1987: the world list (Wikipedia) says 7.5; the Forbes 400 three weeks later says 8.5. Both are kept, on different lists.
- Walton family: 1990 AP 13.3 against Wikipedia 13.5 (used 13.3); 1991 UPI "$18bn" against Wikipedia and Tampa Bay 18.5 (used 18.5); 1995 AP 23.4 against Wikipedia 23.5 (used 23.4).
- Gates, world 1994: Wikipedia 8.2; the July 1995 AP said he "rose from $8.1bn". Kept 8.2 and noted it.
- Gates, Forbes 400 1998: Verdict says 58; AP 1999 says "up from $59bn a year ago". Kept 58. The often-quoted "$58.4bn" was not confirmed.
- Buffett, world 2000: AP 28.0 against Wikipedia 25.6. Used AP; AP 2001's "+$4bn to 32.3" supports it.
- Dell, world 2000: AP 17.8 against Wikipedia 19.1. Used AP.
- Li Ka-shing 1999: Irish Times 12.7 against Wikipedia 12.6.
- Ranking 1990–1992: Forbes ranked families separately from individuals. In 1991 Mori was the "richest individual" (No.1 at 15.0) although the Walton family was at 18.5. Wikipedia's tables mix the two.
- Kluge, Forbes 400 1994: 5.9 appears in the Oct-1994 AP story, in a sentence that also mentions his 1989–91 reign. Medium confidence.
- Tsutsumi 1989 is "at least $15bn", and 1991 is "more than $14bn" (Forbes said up to $35bn if he owned all of the holding company).
- Bettencourt 1999: the contemporary report gives 13.9. A 2013 Forbes retrospective claims "$30bn in 1999"; I treated that as an error and left it out.
- Tan Yu: Wikipedia says "7th richest in 1997, ~$7bn". The 1997 AP top 25 does not include him; CORGIS 1996 has him at 7.0 (No.16). Only the 1996 row is kept.

## Surprises worth knowing for the film
- Japan's land bubble: Tsutsumi was No.1 from 1987 to 1990 (20.0 → 18.9 → 15 → 16).
  - Taikichiro Mori was the No.1 individual in 1991 and 1992. Tsutsumi again led the individuals in 1993–94, at 9.0 and 8.5.
  - Tsutsumi fell to No.22 by 1997 (8.0) and to 2.8 by 2001.
  - Six of the 1987 top ten were Japanese land or transport tycoons.
- Gates's first world No.1 was July 1995 (12.9). The list peak was July 1999 at 90.0, priced 30 Apr 1999.
  - The famous "briefly over $100bn" was a press calculation for spring 1999, not a list figure. One 2004 recap says "$100bn in 1998".
  - In the Forbes 400 he peaked at 85 (Oct 1999).
- Son: 4.6 (1996) → 6.4 (1999 list) → Forbes Global put his holdings at $64bn (20 Mar 2000 issue, against Gates $82bn).
  - The July 2000 list gave 19.4 (No.8) and said he had peaked at $68.5bn in mid-February.
  - By 2001 he was at 5.6 (No.55). Forbes never put more than 19.4 on a list, but it did publish the $64bn and $68.5bn figures.
- Ellison went from 9.5 (No.30, world 1999) to 47.0 (world 2000) to 58.0 (Forbes 400, Oct 2000, 5bn behind Gates), then back to 26.0 and 21.9 in 2001.
- 2001 Forbes 400 values are as of 27 Aug 2001. Forbes then revalued them to 24 Sep: Gates 46.8, Buffett 30.4, Allen 24.2.
- Families are excluded from 1999–2000 working-rich rankings: in 1998 Forbes moved to "working rich" only. The list dropped the du Ponts, Mellons and Fords. Only S. Robson appears for the Waltons in 2000; Bettencourt sat in a separate "coupon clipper" list in 1999.
- Royals were always excluded: the Sultan of Brunei was put at $25bn (Fortune 1990), above $30bn (Forbes 1995) and $38bn (Forbes 1997). King Fahd was at $28–30bn in 1999–2000. None of these are in the CSV.

## Biggest gaps if more digging is possible
- Full world lists for 1988, 1989 and 1992–1994; only top-10 or partial lists were found.
- Forbes 400 1989, 1990 and 1992 beyond No.1–2.
- Forbes 400 1998 apart from Gates.
- Walton heirs individually 1992–1995 and 1998–2000.
- Ellison and Dell world values 1993–1996.
- The original Forbes issues (Oct 5 1987, July issues 1988–2001) on archive.org would close most of these.
