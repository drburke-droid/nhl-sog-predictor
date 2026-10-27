# YouTube copy: the best-paid seasons ever

A bar-chart race of the best-paid single seasons in MLB, the NBA, the NFL and the NHL, 1913-2025, every salary in 2025
dollars. The race itself (`paid/`, film `media/paid/best_paid_athletes.mp4`) lives on branch
`claude/wonderful-ritchie-uyd31i`; this folder holds what goes around it on YouTube.

**Upload `best_paid_athletes_march.mp4`:**
* this is that film as rendered at 084a68d, with the fantasy race's march under it (`../fantasy/march_best_paid.wav`,
  re-cut to this film's timeline);
* 2:23, 1920×1080, 30 fps, H.264, AAC 320 kbps;
* −14.0 LUFS, true peak −3.2 dBFS;
* if the film is rendered again, lay the march under the new render instead (`../../fantasy/README.md`).

**A/B/C test.** Each thumbnail is 1280×720 and built by `../../paid_youtube/thumbs.py` from the race's `season_top.csv`.

| | Thumbnail | Title |
|---|---|---|
| A | `best_paid_thumb_A.png`: Ruth and Jordan, "$1.8M" against "$65.4M", "PAID IN TODAY'S DOLLARS" | Babe Ruth vs. Michael Jordan: The Best-Paid Seasons Ever, in Today's Dollars |
| B | `best_paid_thumb_B.png`: Jordan, "STILL #1", over Prescott's and Burrow's seasons | Michael Jordan's 1998 Salary Is Still the Biggest in US Sports (Adjusted for Inflation) |
| C | `best_paid_thumb_C.png`: Ruth and Jordan, "?", "HIGHEST-PAID EVER", "MLB · NBA · NFL · NHL" | Who Had the Best-Paid Season in American Sports History? (Adjusted for Inflation) |

* **A** is the film's own ending: Ruth's best season against Jordan's, 37 times as much.
* **B** is the surprise: a 1998 salary still beats every quarterback contract of today. It shows Jordan alone, to test
  whether he carries the thumbnail without Ruth.
* **C** is the plain curiosity gap, with no figures.

The titles name the leagues (or say "US sports") because the race covers MLB, the NBA, the NFL and the NHL only. Soccer
and boxing have paid more than $65 million a year, so "highest-paid athlete ever" would be false.

**Description**

```
Who had the best-paid season in American sports? Every salary in MLB, the NBA, the NFL and the NHL since 1913, converted to 2025 dollars, so Babe Ruth's $80,000 and Dak Prescott's $60 million stand on one scale. The board keeps the ten best-paid seasons so far: a season stays until ten better ones come along. Under it, each league's revenue, in the same dollars.

Babe Ruth's $80,000 of 1930 topped President Hoover's $75,000, yet it is $1.5 million today: about twice today's MLB minimum salary. Red Grange's $100,000 of 1925 held the record for 41 years. After free agency, the record fell almost every year from 1982 to 1998, until Michael Jordan's $33.1 million of 1997-98. That is $65.4 million in 2025 dollars, and it is still No. 1.

How it's counted
• Team salaries only: no endorsements, boxing purses, prize money or leagues outside North America.
• MLB and NBA: the season's salary (an NBA season counts for the year it ends). NFL: the contract's average per year, from the season it was signed. NHL: the cap hit.
• Bonuses are spread over the contract. Deferred money counts at its present value: Shohei Ohtani's $70 million a year is $28.2 million.
• Rival leagues count: the AFL, AAFC, WFL, USFL, ABA and WHA.
• Before the 1980s the NBA, NFL and NHL boards hold the stars whose contracts were reported, each from a contract, a newspaper report or a biography.
• Inflation: the US consumer price index (CPI-U), annual averages.

Sources: Baseball Reference, the Lahman database, Basketball Reference, OverTheCap (via nflverse), CapWages and published contracts. League revenue: Congressional hearings, league statements, Financial World, Forbes, Sportico and CNBC, mostly via Rodney Fort's sports business data.

Chapters
0:00 1913: baseball's best-paid
0:13 Babe Ruth takes the record
0:39 Red Grange's record falls after 41 years
0:56 Free agency
1:13 The record falls almost every year
1:35 Ewing, then Jordan
1:46 Still No. 1
2:12 Ruth vs. Jordan

Music: an original orchestral march, synthesized for this video.
Thumbnail photos: Babe Ruth, 1921, Bain News Service, Library of Congress (no known restrictions). Michael Jordan, 1997, Steve Lipofsky, Basketballphoto.com (CC BY-SA 3.0).
```

**Tags**

```
highest paid athletes, best paid athletes of all time, athlete salaries, adjusted for inflation, Michael Jordan salary, Babe Ruth salary, sports salaries history, bar chart race, MLB, NBA, NFL, NHL, Dak Prescott, Patrick Mahomes, Joe Burrow, Stephen Curry, Shohei Ohtani, free agency, sports history, data visualization
```

**Pinned comment** (the closing board, from `season_top.csv`)

```
The 2025 board, in 2025 dollars (as paid in brackets):
1. Michael Jordan, 1997-98 (Bulls, NBA): $65.4M ($33.1M)
2. Dak Prescott, 2024 (Cowboys, NFL): $61.6M ($60.0M)
3. Michael Jordan, 1996-97 (Bulls, NBA): $60.4M ($30.1M)
4. Dak Prescott, 2025 (Cowboys, NFL): $60.0M
5. Joe Burrow, 2023 (Bengals, NFL): $58.1M ($55.0M)
6. Trevor Lawrence, 2024 (Jaguars, NFL): $56.4M ($55.0M)
7. Jordan Love, 2024 (Packers, NFL): $56.4M ($55.0M)
8. Joe Burrow, 2024 (Bengals, NFL): $56.4M ($55.0M)
9. Patrick Mahomes, 2020 (Chiefs, NFL): $56.0M ($45.0M)
10. Stephen Curry, 2024-25 (Warriors, NBA): $55.8M

Two of the top three are Jordan's, from the 1990s. Who passes him first?
```

**Notes on the upload.**
* **Chapters.** Each chapter starts at a caption in the film and lasts at least 10 s, as YouTube requires. The last,
  2:12, starts as the race ends and holds the closing Ruth-against-Jordan summary.
* **Photo credit.** The Jordan photo is CC BY-SA 3.0, so the credit line in the description is required, not a courtesy.
  The thumbnails made from it carry the same licence. Ruth's photo has no known restrictions.
* **Likeness.** The thumbnails show two real players to illustrate a factual video about their salaries, which is
  editorial use. They imply no endorsement; don't reuse them for advertising or merchandise.
