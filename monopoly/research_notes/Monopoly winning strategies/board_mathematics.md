# Mathematical Analysis of the Monopoly Board (US Atlantic City Edition): Markov Chains, Landing Probabilities, and ROI by Color Group

Conventions used in these notes:
- "Short jail" means the player leaves jail as soon as possible (pays $50 or uses a card). "Long jail" means the player stays in jail as long as the rules allow (rolls for doubles and pays on the third turn).
- "Per opponent roll" (Collins) counts each dice roll. A turn with doubles contains several rolls. "Per turn" figures (Ash & Bishop as reprinted) add up to about 1.19 per turn.
- Figures labelled "computed from Collins" are my own arithmetic on Truman Collins' published tables. They are inferences, not published claims.

## Q1. Which papers first modelled Monopoly as a Markov chain, and what did each find?

### Takeaway
The lineage runs as follows:
1. Ash & Bishop (Mathematics Magazine, 1972) built the first full Markov-chain model, using the long-jail assumption.
2. Ian Stewart's Scientific American columns (April and October 1996) first showed a toy model in which every square is equally likely, then synthesised readers' full models. Those readers included Abbott & Richey, the authors of the 1997 College Mathematics Journal paper.
3. Truman Collins' web pages (c. 1997) became the standard, widely cited set of exact tables.
4. Later teaching papers (Bilisoly 2008/2014, Gartland et al. 2014), a book chapter by Bewersdorff, and popular work by Hannah Fry, Thomas Oléron Evans and Matt Parker (2016-17) all reproduce the same main pattern. Jail is the most-occupied square, Illinois Ave is the top property, and orange (with red close behind) is the best group per dollar.

### Cited Findings

**Ash & Bishop (1972)**
- Citation: Robert B. Ash & Richard L. Bishop, "Monopoly as a Markov Process," *Mathematics Magazine* 45(1):26-29, January 1972 — [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107); [Semantic Scholar listing](https://www.semanticscholar.org/paper/Monopoly-as-a-Markov-Process-Ash-Bishop/08c973c8088ed9842b1516e52c01cb0846f0e0de)
- Abstract, as surfaced in search snippets: "Expected income from the bank and expected rents on properties are calculated from the limit frequencies, which allows a quantitative estimate of the relative value of the various properties." — [Semantic Scholar listing](https://www.semanticscholar.org/paper/Monopoly-as-a-Markov-Process-Ash-Bishop/08c973c8088ed9842b1516e52c01cb0846f0e0de). I could not read the original text. Only this snippet was seen.
- An operations-research textbook chapter (Ch. 17 "Markov Chains", Example 6 "Playing Monopoly", pp. 937-938) reprints the Ash & Bishop steady-state table "by permission". It states their assumption: "each Monopoly player who goes to Jail stays until he or she rolls doubles or has spent three turns in Jail", which is the long-jail policy — [UW course handout, Ch. 17](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)
- Ash & Bishop values as reprinted (selected): Jail (in jail) .1123; Illinois Ave .0355; Go .0346; B&O RR .0344; Free Parking .0336; Tennessee .0335; New York .0334; Reading RR .0332; St. James .0318; Water Works .0315; Pennsylvania RR .0313; Electric Co .0311; Park Place .0245; Boardwalk .0295; Mediterranean .0237; Chance 1 and Chance 3 .0096 each — [UW course handout, Ch. 17](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)
- Textbook example based on Ash & Bishop:
  - Hotels on orange cost $1,500 and earn "950(.0335) + 950(.0318) + 1,000(.0334) = $95.44" per dice roll, which is $0.064 per dollar invested per roll.
  - Hotels on green cost $3,000 and earn "1,275(.0294) + 1,275(.0300) + 1,400(.0279) = $114.80", which is $0.038 per dollar.
  - Conclusion: "the Orange monopoly is superior to the Green monopoly."
  - Source: [UW course handout, Ch. 17](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)

**Ian Stewart, Scientific American (1996)**
- "How Fair Is Monopoly?" (Mathematical Recreations, *Scientific American*, April 1996, pp. 104-105):
  - Uses a simplified model: a single die thrown twice ("ghost" move plus real move), with no cards and no jail.
  - Shows that the long-run probability of each square tends to 1/40. The second-largest eigenvalue has absolute value 0.964.
  - Concludes that "in the long run—no particular square is more or less likely to be landed on."
  - Source: [Stewart, Apr 1996 PDF](https://legacy-www.math.harvard.edu/archive/21b_fall_05/Articles/howfairismonopoly.pdf)
- "Monopoly Revisited" (*Scientific American*, October 1996, pp. 116-119) admits: "Monopoly enthusiasts were quick to point out that in the real game, the long-term distribution of probabilities is not even." It synthesises readers' full models:
  - William J. Butler (Pascal)
  - Thomas H. Friddell, a Boeing engineer (Mathcad)
  - Stephen Abbott of St. Olaf College with colleague Matt Richey (Maple)
  - Source: [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Stewart Oct 1996 results:
  - Jail is occupied 5.89% of the time, "almost twice as likely ... as any other" square.
  - Next: Illinois Ave 3.18% and GO 3.11% (third).
  - Railroads: B&O 3.06%, Reading 2.99%, Pennsylvania RR 2.91%, Short Line 2.44%. Short Line is low because "unlike the others, Short Line does not feature a CHANCE card."
  - Utilities: Water Works 2.81%, Electric Co 2.62%.
  - The third Chance square (0.87%) is the least likely square apart from Go To Jail (0%).
  - The most likely distance moved in one turn is 7. The maximum is 35 squares (6,6; 6,6; 6,5).
  - Source: [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Friddell's break-even findings, as reported by Stewart:
  - "Although it costs more to buy houses early, the break-even point will be reached more quickly if you do."
  - "With two houses or fewer, it typically takes around 20 moves or more to break even. Three houses produces a definite improvement."
  - New York Ave with three houses "breaks even in about 10 turns". It is the quickest of the properties between GO and Indiana Ave. Properties beyond Indiana were not evaluated.
  - Source: [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Reader Bruce Moskowitz described the orange ("tan-colored") St. James, Tennessee and New York as "common knowledge" valuable because of "a relatively high probability of landing on one of them when leaving Jail". Stewart notes all three "figure among the top 12" — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- David Weiblen argued the uneven probabilities do not make the game unfair "because all players face the same situation" — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Bibliographic details: Stewart 1996 "How Fair is Monopoly?" *Sci. Am.* 274(4):104-105; "Monopoly Revisited" *Sci. Am.* 275(4):116-119 — [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107). Collins instead says the topic appeared in "the April 1996 issue with additional information in the August 1996 and April 1997 issues" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml). The PDF footer shows October 1996 for "Monopoly Revisited". The August and April 1997 references may be Feedback notes; this is unverified.

**Abbott & Richey (1997)**
- Citation: Stephen D. Abbott & Matt Richey, "Take a Walk on the Boardwalk," *The College Mathematics Journal* 28(3):162-171 (1997) — [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/07468342.1997.11973857); [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107)
- Affiliation conflict: Stewart (1996) places Abbott at St. Olaf College; search metadata (ResearchGate) lists Middlebury College — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf). The full text is paywalled. See Gaps.

**Truman Collins (c. 1997)**
- Collins wrote a C simulation and then an exact Markov-matrix program. He estimated the doubles-history probability per square by simulating 32 billion rolls. He models both jail strategies and separates "visiting jail" from "in jail". His railroad income includes "the extra income from the two Chance cards that result in double railroad rent" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Wikipedia cites Collins (1997, "Monopoly Square Probabilities") for its landing-frequency claims — [Wikipedia: Monopoly (game)](https://en.wikipedia.org/wiki/Monopoly_(game))

**Later academic and popular work**
- Bilisoly (JSM 2008 Proceedings; arXiv 2014) gives a Mathematica teaching model with these assumptions:
  - Leave jail immediately.
  - Ignore the three-doubles rule.
  - Cards drawn with replacement, each with probability 1/16.
  - Chance on square 37 is modelled before Community Chest on 34, because of the "back 3" card.
  - Result: "Illinois Ave. has the highest probability (besides Jail), which is a well-known fact among Monopoly players."
  - Source: [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107)
- Bilisoly's reference list of the Monopoly-Markov literature includes Ash & Bishop 1972, Stewart 1996 (x2), Abbott & Richey 1997, Murrell 1999 ("The Statistics of Monopoly," *Chance* 12(4):36-40), Johnson 2003 ("Using Games to Teach Markov Chains," *PRIMUS*), and Bewersdorff 2005 (*Luck, Logic, and White Lies*, Ch. 16 "analyzes Monopoly with Markov Chains") — [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107)
- Bewersdorff's web supplement to his Ch. 16 says that "In long term the difference of both frequencies [Illinois vs Park Place] is 45%!" The exact chain is "a system of 121 linear equations including 120 variables". Results "are based on the American edition", and other editions differ slightly because of different card sets — [Bewersdorff](https://bewersdorff-online.de/amonopoly/)
- Gartland, Burson & Ferguson (Illinois, 2014) use a 3-states-per-square chain, with six states for the last Community Chest square and three states for in-jail. The second eigenvalue modulus is "about 0.87", so convergence is O(0.87^n) — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Stanford MS&E220 student project (2008) ignores cards. It finds light blue most-landed before jail is modelled and orange most-landed after. However, its model gives "Yellow and Green ... high expected earnings in the long-run". It recommends yellow and green if the game lasts more than about 30 throws — [Stanford MS&E220 2008](https://web.stanford.edu/class/msande220/monopoly2.pdf)
- Hannah Fry & Thomas Oléron Evans, *The Indisputable Existence of Santa Claus: The Mathematics of Christmas* (excerpt on TED Ideas):
  - One opponent: "go for the orange or light blue sets (both if you can)".
  - Two or three opponents: "it's orange and red that you want to be targeting".
  - More than three opponents: "green becomes your best shot".
  - "the average game of Monopoly takes about 30 turns per competitor".
  - "it's the third house that is really worth investing in quickly".
  - "utilities are completely pointless".
  - Railroads "are relatively affordable" but "it's unlikely they will get you anywhere fast".
  - Boardwalk "is the standout property on the board" individually; Park Place is "very expensive to buy and build on but is rarely visited".
  - Source: [TED Ideas (Fry & Oléron Evans)](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- Fry & Parker video "The Mathematics of Winning Monopoly" (Dec 2016). Inverse reports that "the orange block is the most valuable in the game — largely because of their choice position 6 to 9 squares away from Jail", verified by Fry's theoretical modelling and "Parker's simulation of a hundred-million dice rolls" — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math). Nerdist describes Parker's simulation as "a million rolls" and says the data and code were posted to Dropbox — [Nerdist](https://nerdist.com/article/the-math-of-winning-monopoly/). The two outlets disagree on the simulation size.

### Inferences
- The Ash & Bishop numbers as reprinted add up to 1.1907, not 1. This is consistent with per-turn landing frequencies, since doubles give an expected ~1.19 rolls per turn. To compare with Collins I divided each value by 1.1907 (my calculation):
  - In Jail becomes 9.43%, against Collins' long-jail 9.46%.
  - Illinois becomes 2.98%, against Collins' 2.99%.
  - Mean absolute deviation from Collins' long-jail table is 0.014 percentage points, excluding one outlier.
  - The outlier is Luxury Tax: .0295 reprinted, which normalises to 2.48%, against Collins' 2.05%. It is probably a transcription error in the textbook reprint, since it duplicates Boardwalk's .0295.
  - So the 1972 and 1997 results agree closely. The textbook's "$95.44 per dice roll" is effectively per opponent turn.
- Stewart's October 1996 figures (Jail 5.89, Illinois 3.18, GO 3.11, B&O 3.06, Reading 2.99, Penn RR 2.91, Short Line 2.44, Water Works 2.81, Electric 2.62, Chance-3 0.87) match Bilisoly's model to 3-4 significant figures (Jail .05896, Illinois .03188, ...). Bilisoly's model explicitly ignores the three-doubles-to-jail rule. Stewart's published table therefore likely reflects a model in which that rule has little or no effect, or is omitted.

### Gaps
- I could not access the full text of Ash & Bishop (1972) or Abbott & Richey (1997). Their own rent and ROI conclusions, beyond the textbook reprint, are unverified. Abbott & Richey's specific findings (for example any house-buying or break-even results) remain unknown.
- Murrell (1999, *Chance*) and Johnson (2003, *PRIMUS*): citations only, no content.
- I found no Monopoly example from a Durrett textbook. It may exist, but I could not verify it.
- I could not obtain Fry's own numeric tables from the Santa Claus book. The TED excerpt describes graphs but gives few numbers. Matt Parker's exact simulated percentages were not retrieved.

## Q2. Long-run landing probabilities for every square (short-jail vs long-jail policy)

### Takeaway
Under exact modelling, the full picture is:
- Jail is by far the most-occupied square: 6.22% combined (3.95% in jail plus 2.27% visiting) if you leave immediately, rising to 11.60% (9.46% in jail) if you stay.
- Illinois Ave is the most-landed property (3.19% / 2.99%), followed by GO and then B&O Railroad, the most-landed railroad.
- Orange squares average about 2.94%, which is 17.5% above a uniform 1/40 and 11.7% above the mean property.
- The orange group as a whole (8.81%) is landed on twice as often as brown (4.29%).

### Cited Findings
Full Collins table, percent chance of ending a roll on each square, from [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml):

| # | Square | Short jail % (rank) | Long jail % (rank) |
|---|---|---|---|
| 0 | Go | 3.0961 (3) | 2.9143 (3) |
| 1 | Mediterranean | 2.1314 (36) | 2.0073 (36) |
| 2 | Community Chest | 1.8849 (37) | 1.7750 (37) |
| 3 | Baltic | 2.1624 (35) | 2.0369 (35) |
| 4 | Income Tax | 2.3285 (28) | 2.1934 (27) |
| 5 | Reading RR | 2.9631 (6) | 2.8010 (8) |
| 6 | Oriental | 2.2621 (32) | 2.1317 (32) |
| 7 | Chance | 0.8650 (40) | 0.8152 (40) |
| 8 | Vermont | 2.3210 (29) | 2.1874 (28) |
| 9 | Connecticut | 2.3003 (30) | 2.1680 (30) |
| 10 | Visiting Jail | 2.2695 (31) | 2.1392 (31) |
| 11 | St. Charles | 2.7017 (15) | 2.5560 (15) |
| 12 | Electric Co | 2.6040 (20) | 2.6140 (13) |
| 13 | States | 2.3721 (26) | 2.1741 (29) |
| 14 | Virginia | 2.4649 (24) | 2.4255 (22) |
| 15 | Pennsylvania RR | 2.9200 (8) | 2.6354 (11) |
| 16 | St. James | 2.7924 (12) | 2.6802 (9) |
| 17 | Community Chest | 2.5945 (21) | 2.2957 (24) |
| 18 | Tennessee | 2.9356 (7) | 2.8210 (6) |
| 19 | New York | 3.0852 (4) | 2.8118 (7) |
| 20 | Free Parking | 2.8836 (9) | 2.8253 (5) |
| 21 | Kentucky | 2.8358 (10) | 2.6143 (12) |
| 22 | Chance | 1.0480 (38) | 1.0448 (38) |
| 23 | Indiana | 2.7357 (13) | 2.5671 (14) |
| 24 | Illinois | 3.1858 (2) | 2.9929 (2) |
| 25 | B&O RR | 3.0659 (5) | 2.8930 (4) |
| 26 | Atlantic | 2.7072 (14) | 2.5370 (16) |
| 27 | Ventnor | 2.6789 (16) | 2.5191 (18) |
| 28 | Water Works | 2.8074 (11) | 2.6507 (10) |
| 29 | Marvin Gardens | 2.5861 (22) | 2.4381 (21) |
| 30 | Go To Jail | 0.0000 (41) | 0.0000 (41) |
| 31 | Pacific | 2.6774 (17) | 2.5236 (17) |
| 32 | North Carolina | 2.6252 (19) | 2.4721 (20) |
| 33 | Community Chest | 2.3661 (27) | 2.2276 (26) |
| 34 | Pennsylvania Ave | 2.5006 (23) | 2.3531 (23) |
| 35 | Short Line | 2.4326 (25) | 2.2906 (25) |
| 36 | Chance | 0.8669 (39) | 0.8158 (39) |
| 37 | Park Place | 2.1864 (33) | 2.0595 (33) |
| 38 | Luxury Tax | 2.1799 (34) | 2.0521 (34) |
| 39 | Boardwalk | 2.6260 (18) | 2.4832 (19) |
| — | In Jail | 3.9499 (1) | 9.4569 (1) |

- Collins on why the Chance and Community Chest squares are low: the numbers are where you "end up", so a card that moves you is credited to the destination square — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins: "about 3.18% of the rolls will result in the player landing on the most probable non-jail square, Illinois Avenue" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins: "The properties between the Jail square and the Go To Jail square are landed on the most, because of the jump caused by landing on Go To Jail" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Wikipedia (citing Collins): the most-landed properties are Illinois Ave, New York Ave, B&O RR and Reading RR. The least-landed are Mediterranean, Baltic, Park Place and Oriental. "Among the property groups, the Railroads are most frequently landed upon, as no other group has four properties; Orange has the next highest frequency, followed by Red." — [Wikipedia: Monopoly (game)](https://en.wikipedia.org/wiki/Monopoly_(game))
- Bewersdorff: Illinois is visited 45% more often than Park Place in the long run — [Bewersdorff](https://bewersdorff-online.de/amonopoly/). This matches Collins: 3.1858 / 2.1864 = 1.457 (my calculation).
- Gartland et al. (long jail): Park Place 2.05%, Boardwalk 2.48%, Pennsylvania Ave 2.34%, Illinois 2.993% — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Bilisoly (no triple-doubles rule, short jail): Jail .05896, Illinois .03188, Go .03114, Reading .02993, B&O .03064 — [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107)
- Dartmouth lecture slides (Lola Thompson, 2010): "Jail is #1". The slides recommend buying "Illinois Avenue, B&O Railroad, Tennessee Avenue, New York Avenue, Reading Railroad, St. James Place, Water Works, Pennsylvania Railroad". Explanation given: distances from Jail, e.g. "Pennsylvania Railroad (5) • St. James Place (6) ..." — [Dartmouth slides](https://math.dartmouth.edu/archive/m20f10/public_html/mcpc.pdf)
- UK-edition undergraduate simulation (Leicester, 2021; 100,000 turns): "The most landed on square is jail at 6.4%" — [Leicester J. Phys. Special Topics](https://journals.le.ac.uk/index.php/pst/article/download/3991/3419/12120)
- Why squares 6-9 past Jail are favoured:
  - Inverse: "the orange block is the most valuable ... largely because of their choice position 6 to 9 squares away from Jail".
  - Inverse: 7 is the most common roll ("17 percent of rolls"), followed by 6 or 8 ("14 percent each").
  - Red follows because it is "the likely destination for people who just landed on an orange space or rolled doubles after leaving Jail".
  - Light blue sits "just as far from Go as the orange tiles are from Jail", and the Advance-to-Go card feeds it.
  - Source: [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)

### Inferences
Group landing figures computed from Collins (short jail; long jail in brackets). Group sum %, then per-square average, then the difference against the 28-property mean (2.631% short, 2.480% long):

| Group | Group sum % | Per-square % | vs. property mean |
|---|---|---|---|
| Brown | 4.29 [4.04] | 2.15 | -18.4% |
| Light blue | 6.88 [6.49] | 2.29 | -12.8% |
| Pink | 7.54 [7.16] | 2.51 | -4.5% |
| Orange | 8.81 [8.31] | 2.94 | +11.7% (and +17.5% vs uniform 2.5%) |
| Red | 8.76 [8.17] | 2.92 | +11.0% (+16.8% vs 2.5%) |
| Yellow | 7.97 [7.49] | 2.66 | +1.0% |
| Green | 7.80 [7.35] | 2.60 | -1.1% |
| Dark blue | 4.81 [4.54] | 2.41 | -8.5% |
| Railroads (4) | 11.38 [10.62] | 2.85 | +8.2% |
| Utilities | 5.41 [5.26] | 2.71 | +2.8% |

- Orange and red are essentially tied on landing frequency: orange 8.81% vs red 8.76% (short jail). Orange's lead widens under long jail, 8.31% vs 8.17%. Orange's edge over red is in ROI per dollar (Q3), not raw traffic.
- Orange squares sit 6, 8 and 9 squares past Jail. Combined 2-dice probabilities of rolling exactly 6, 8 or 9 are 5/36, 5/36 and 4/36. Jail is the most common starting point (Go To Jail, the cards and triple doubles all feed it), so orange gets a concentrated stream of traffic.
- Stewart's 5.89% for Jail and Collins' 6.22% (short jail, in plus visiting) disagree by about 0.33 percentage points. This is plausibly the three-doubles route to jail, which Bilisoly's matching model omits. 1/216 of turns end in triple doubles, roughly 0.39% per roll.

### Gaps
- Abbott & Richey's own probability table was not retrieved, so I cannot compare it with Collins.
- Fry/Parker's per-square numbers were not retrieved.

## Q3. Expected rent per opponent roll, payback times by group and building level, and published strategy claims

### Takeaway
Collins' payback tables show that the third house is the fastest-paying building on almost every property (New York Ave's third house repays in 8.5 opponent rolls, the best on the board). Orange gives the best return per dollar at 3 houses and at hotels. Railroads are strong cheap early income. Utilities and lone brown properties are the worst.

Dark blue's "overrated" reputation is only partly supported:
- Boardwalk alone is the best single property.
- Per dollar invested, a fully built dark blue set is mid-pack, comparable to red and yellow.
- Its total income trails green, yellow and red.

### Cited Findings

**Expected income per opponent roll** (short jail, $), from [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml). Columns are single property / whole block unimproved / 1H / 2H / 3H / 4H / hotel:
- Mediterranean 0.0426 / 0.0853 / 0.2131 / 0.6394 / 1.9182 / 3.4102 / 5.3284
- Baltic 0.0865 / 0.1730 / 0.4325 / 1.2974 / 3.8923 / 6.9197 / 9.7308
- Oriental 0.1357 / 0.2715 / 0.6786 / 2.0359 / 6.1078 / 9.0486 / 12.4418
- Vermont 0.1393 / 0.2785 / 0.6963 / 2.0889 / 6.2666 / 9.2839 / 12.7653
- Connecticut 0.1840 / 0.3681 / 0.9201 / 2.3003 / 6.9010 / 10.3515 / 13.8020
- St. Charles 0.2702 / 0.5403 / 1.3508 / 4.0525 / 12.1575 / 16.8854 / 20.2624
- States 0.2372 / 0.4744 / 1.1860 / 3.5581 / 10.6744 / 14.8256 / 17.7907
- Virginia 0.2958 / 0.5916 / 1.4789 / 4.4368 / 12.3245 / 17.2542 / 22.1840
- St. James 0.3909 / 0.7819 / 1.9547 / 5.5848 / 15.3583 / 20.9431 / 26.5279
- Tennessee 0.4110 / 0.8220 / 2.0549 / 5.8712 / 16.1457 / 22.0169 / 27.8880
- New York 0.4936 / 0.9873 / 2.4681 / 6.7874 / 18.5110 / 24.6814 / 30.8517
- Kentucky 0.5105 / 1.0209 / 2.5523 / 7.0896 / 19.8509 / 24.8137 / 29.7764
- Indiana 0.4924 / 0.9848 / 2.4621 / 6.8392 / 19.1498 / 23.9373 / 28.7248
- Illinois 0.6372 / 1.2743 / 3.1858 / 9.5573 / 23.8932 / 29.4683 / 35.0434
- Atlantic 0.5956 / 1.1912 / 2.9779 / 8.9338 / 21.6576 / 26.3952 / 31.1328
- Ventnor 0.5893 / 1.1787 / 2.9467 / 8.8402 / 21.4309 / 26.1189 / 30.8069
- Marvin Gardens 0.6207 / 1.2413 / 3.1033 / 9.3098 / 21.9814 / 26.5070 / 31.0326
- Pacific 0.6961 / 1.3922 / 3.4806 / 10.4417 / 24.0963 / 29.4511 / 34.1365
- North Carolina 0.6825 / 1.3651 / 3.4127 / 10.2382 / 23.6266 / 28.8769 / 33.4710
- Pennsylvania Ave 0.7002 / 1.4004 / 3.7509 / 11.2528 / 25.0063 / 30.0075 / 35.0088
- Park Place 0.7652 / 1.5305 / 3.8262 / 10.9320 / 24.0504 / 28.4232 / 32.7960
- Boardwalk 1.3130 / 2.6260 / 5.2519 / 15.7558 / 36.7635 / 44.6414 / 52.5193

**Railroads and utilities** (short jail, per railroad; includes the two double-rent Chance cards), from [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml). Columns are owning 1 / 2 / 3 / 4 railroads:
- Reading 0.8130 / 1.6261 / 3.2521 / 6.5043
- Pennsylvania RR 0.8021 / 1.6041 / 3.2083 / 6.4165
- B&O 0.8538 / 1.7076 / 3.4152 / 6.8304
- Short Line 0.6082 / 1.2163 / 2.4326 / 4.8653
- Utilities (one / both): Electric 0.7189 / 1.7972; Water Works 0.7939 / 1.9849
- Long-jail values are about 5-10% lower; e.g. Illinois hotel 32.92 and Boardwalk hotel 49.66.

**Payback: opponent rolls to recoup incremental cost** (short jail), from [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml):
- Buying a single property: Mediterranean 1407.5 (worst), Baltic 693.7, Oriental 736.8, New York 405.2, Illinois 376.7, Boardwalk 304.7 (best street).
- First house: Mediterranean 391.0, Baltic 192.7, New York 67.5, Boardwalk 76.2, Illinois 78.5.
- Second house: New York 23.2, Illinois 23.5, Boardwalk 19.0.
- Third house:
  - Orange: New York 8.53, Tennessee 9.73, St. James 10.23.
  - Boardwalk 9.52, Illinois 10.46, Connecticut 10.87.
  - Red and yellow: Kentucky 11.75, Indiana 12.18, Atlantic 11.79, Ventnor 11.91, Marvin 11.84.
  - Light blue and pink: Oriental 12.28, Vermont 11.97, St. Charles 12.34, States 14.05, Virginia 12.68.
  - Green and Park Place: Pacific 14.65, North Carolina 14.94, Pennsylvania Ave 14.54, Park Place 15.25.
  - Brown: Baltic 19.27, Mediterranean 39.10.
- Fourth house and hotel:
  - Orange: about 16-18 each.
  - Red: about 27-31.
  - Yellow: about 32-33.
  - Green: 4th house 37-40, hotel 40-44.
  - Park Place 45.7 each; Boardwalk 25.4 each.
- Railroads: first railroad 234-329 rolls (B&O 234.2 best, Short Line 328.9 worst). Second 82-98. Third 31-35. Fourth 12.7-13.6.
- Utilities: first 188.9 (Water Works) and 208.7 (Electric); second about 49-50.

**Collins' conclusions**, quoted from [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml):
- "The railroads are excellent investments, particularly when owned together, although in absolute income terms they don't keep up with heavily built on properties later in the game."
- "The best return on investment to be found is from putting a third house on New York Avenue. In fact, the third house has the fastest payoff of any building on almost all of the properties."
- "a hotel there [Illinois] will bring the most income other than a hotel on Boardwalk."
- "By far the worst individual investment is to buy Medeterranean Avenue without first owning Baltic."
- "The Orange ones have the biggest bang for the buck as far as building goes."
- Collins also warns that the figures are long-run averages and position matters. An opponent on Indiana Ave has "about a 14% chance of landing on Marvin Gardens on the next roll but no chance of landing on Illinois Avenue".

**Other expected-value results**
- Collins: average income per roll from other squares (short jail) is GO +$33.78, Income Tax -$4.66 (assumes $200 always paid), Luxury Tax -$1.63 — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Gartland et al. (long jail), expected $ per opponent roll for a whole monopoly. 3 houses: brown 5.50, light blue 18.17, pink 33.41, orange 47.81, red 58.41, yellow 61.53, green 68.22, dark blue 57.33. Hotels: 14.26, 36.77, 57.31, 81.52, 86.87, 87.91, 96.25, 80.43 — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Gartland et al. railroads (1/2/3/4 owned): 0.64 / 2.54 / 7.63 / 20.35. Utilities: Electric 0.61, Water Works 0.71, both 3.31. "Generally speaking, the utilities are not worth much." They also say that "In the early stages of a monopoly development, the additional rent gained by adding a house increases until three houses are erected. Thus, a 3-house develpoment is the most cost efficient." — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Gartland et al.: "a hotel on Pennsylvania Avenue is worth more than a hotel on Park Place". A third house on Pennsylvania Ave adds +$12.89 per opponent roll, against +$12.30 on Park Place, despite Park Place's larger rent jump ($600 vs $550) — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Gartland et al. trade examples. Green at 3 houses (68.22) beats yellow plus brown (61.53 + 5.50 = 67.03). Red plus brown (63.91) beats yellow (61.53) — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Ash & Bishop via textbook: orange hotels yield $0.064 per dollar per roll, against green's $0.038 — [UW course handout, Ch. 17](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)
- Dartmouth slides: "the orange group (St. James Place, Tennessee Ave. and New York Ave.) reaches break even point fastest, but the green group ... has the" (the text is cut off in the extracted file) — [Dartmouth slides](https://math.dartmouth.edu/archive/m20f10/public_html/mcpc.pdf)
- Fry & Oléron Evans:
  - "The first house takes a really long time to start paying for itself."
  - Mediterranean's earnings slope "is very shallow".
  - New York Ave has "its steep slope of $30 expected earnings per roll" (this matches Collins' NY hotel figure of $30.85; my comparison).
  - Oriental "is up there with the best-earning sets until about 25 rolls".
  - The dark ("navy") blue set "also has two rather than three squares" and is "just too rare an event to be worth the backing".
  - Source: [TED Ideas (Fry & Oléron Evans)](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- Inverse cites a Tim Darling simulation table showing "three houses are good across the board — especially on the orange properties", and that Darling "is also a fan of the railroads, if you can get all four." — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- Error in Inverse: it says "Each house placed on an orange property costs $50". The orange house cost is $100. Collins' payback math only works at $100: NY third house 100 / (18.511 - 6.787) = 8.53 rolls, matching his table. Inverse's rent figures for St. James ($14 → $70 → $200 → $550 → $950 hotel) are correct — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math); [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Dissenting result (Stanford MS&E220, no-cards model): "even though Orange and Light Blue have the highest probabilities of being landed on, their expected earnings in the long run are not the highest". The authors recommend yellow and green for games longer than about 30 throws — [Stanford MS&E220 2008](https://web.stanford.edu/class/msande220/monopoly2.pdf)
- UK edition (Leicester 2021): with no houses the Stations are the best set; at three houses orange is best. The abstract says "£3.4 per die roll", the body says "£3.50", "£0.90 above the average of £2.60", and "£4.20 per dice roll" with hotels. The metric is rent × probability / total cost × 100, so it is effectively per £100 invested. The paper is undergraduate and internally inconsistent (3.4 vs 3.50) — [Leicester J. Phys. Special Topics](https://journals.le.ac.uk/index.php/pst/article/download/3991/3419/12120)

### Inferences
Color-group ROI computed from Collins' short-jail income table, using standard US prices and house costs (brown and light blue $50, pink and orange $100, red and yellow $150, green and dark blue $200). Long jail lowers all of these by about 5-7% without changing the order much.

| Group | $/opp roll at 3H | $/opp roll at hotel | $/roll per $1,000 total invested (land + 3H) | Build-only payback, 3H (rolls) | $/roll per $1,000 total at hotels |
|---|---|---|---|---|---|
| Brown | 5.81 | 15.06 | 13.8 | 51.6 (on $300) | 24.3 |
| Light blue | 19.28 | 39.01 | 25.0 | 23.3 (on $450) | 36.5 |
| Pink | 35.16 | 60.24 | 26.2 | 25.6 (on $900) | 31.1 |
| **Orange** | **50.02** | **85.27** | **34.3** | **18.0** (on $900) | **41.4** |
| Red | 62.89 | 93.54 | 31.0 | 21.5 (on $1,350) | 31.9 |
| Yellow | 65.07 | 92.97 | 30.3 | 20.7 (on $1,350) | 30.5 |
| Green | 72.73 | 102.62 | 26.7 | 24.7 (on $1,800) | 26.2 |
| Dark blue | 60.81 | 85.32 | 31.2 | 19.7 (on $1,200) | 31.0 |

- Orange leads every per-dollar column. Green has the highest absolute income but the worst per-dollar return among the expensive groups. This reconciles the Stanford "green and yellow earn most" result (absolute income) with the "orange is best" consensus (ROI and speed).
- On land plus 3 houses, orange repays its whole outlay ($1,460) in about 29 opponent rolls, and green ($2,720) in about 37. With 3 opponents, each "round" gives about 3-3.6 opponent rolls, so orange recoups in about 8-10 rounds. Fry's "30 turns per competitor" game length leaves time for this.
- All four railroads together earn $24.62 per opponent roll (short jail; $23.02 long jail) for $800, which repays in about 32.5 rolls. That is $30.8 per roll per $1,000 invested, comparable to a 3-house monopoly, with no building cost and no need for a full color set. This supports "railroads are good early". Collins agrees that their absolute income lags heavily built sets later.
- Both utilities together earn $3.78 per opponent roll for $300, about 79 rolls to repay. One utility needs 189-209 rolls. Utilities are the weakest income asset besides unpaired brown, which supports "utilities are the worst" (Fry: "completely pointless").
- "Dark blue is overrated" is partly supported:
  - Supporting: the set has only two squares, Park Place is the 33rd most-landed square, Park Place's 4th house and hotel take 45.7 rolls each to repay, and its absolute income at hotels (85.32) is below red, yellow and green.
  - Against: Boardwalk alone is the single best street (fastest single-property payback, 304.7 rolls, and the highest hotel income, $52.52 per roll), and the set's per-dollar return is in line with red and yellow.
- "Third house is best" holds on 20 of 22 streets. The exceptions are the browns: Mediterranean's hotel (26.1) and 4th house (33.5) both repay faster than its 3rd (39.1), and Baltic's 4th house (16.5) and hotel (17.8) beat its 3rd (19.3). Orange is the one group where the 4th house and hotel are still quick (16-18 rolls).

### Gaps
- I found no published turn-count for the time-to-bankrupt-an-opponent by group. Only payback in opponent rolls is available.
- Collins' payback tables assume rent is always collected. They ignore mortgages, cash constraints and the time to complete a set.
- Fry's numeric tables and Darling's table values were not retrieved directly.

## Q4. The jail effect: how jail policy changes probabilities, and staying in jail late in the game

### Takeaway
Staying in jail ("long jail") raises time spent in jail from 3.95% to 9.46% of rolls. It cuts your landing rate on every other square by about 5-10%. That is why the math and the strategy writers advise leaving jail fast early, to buy property, and staying in late, to avoid rent. Jail is also the engine of orange and red traffic.

### Cited Findings
- Collins: "Early in the game it is often best to get out as soon as possible so as to have more opportunities to buy property. Later in the game, it is best to stay in jail as long as possible to avoid landing on an opponent's property. The choice of strategy changes the probabilities for all of the squares of the board because the more time you spend in jail, the less you will land on other squares." — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins figures. In Jail: 3.9499% short vs 9.4569% long. Visiting Jail: 2.2695% vs 2.1392%. Illinois: 3.1858% vs 2.9929%. New York: 3.0852% vs 2.8118% — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins: "Note the higher rank of the squares 2, 4, 6, 8, 10, and 12 in front of Jail caused by the increased likelihood of getting out of jail with doubles". Under long jail, Electric Co rises from rank 20 to 13, Free Parking from 9 to 5, and St. James from 12 to 9 — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins: long-jail expected income on all properties is lower. For example, Boardwalk hotel is $52.52 short vs $49.66 long, and orange 3-house payback is about 9.4-10.7 vs 8.5-10.2 rolls. This is because the long-jail tables describe opponents who sit in jail — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Gartland et al. model 3 in-jail states (roll each turn, leave on doubles, pay after the third failure). They cite advice "to start staying in jail once other players have monopolies with houses" — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Stewart: the jail choice makes the process "non-Markovian" because it "depend[s] on the player's psychology". Models handle it by treating Jail as "a Markov subprocess—a series of three (virtual) squares" ("Just in Jail", "In Jail One Turn Already", "Must Come Out of Jail Next Turn"). Players may "stay in and hope not to throw doubles!" — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Inverse/Fitzsimons: "the longer the game goes on, the smarter it is to wait in Jail for as long as possible". Use Get Out of Jail Free cards early, when properties are unclaimed. Late game, "try and get out of Jail by rolling doubles instead of paying bail" — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- US jail rules: pay $50, use a card, or roll doubles. After three failed turns you must pay and move. A player who rolls doubles to leave Jail "does not roll again". While in jail you still collect rent, build and trade — [Wikipedia: Monopoly (game)](https://en.wikipedia.org/wiki/Monopoly_(game))
- Ash & Bishop assumed the long-jail policy, with an In Jail probability of .1123 per turn — [UW course handout, Ch. 17](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)

### Inferences
- Combining jail states, a long-jail player sits on the Jail square 11.60% of the time, against 6.22% for a short-jail player. That is roughly 1 in 9 rolls versus 1 in 16.
- Late game, staying in jail reduces your own exposure by about 6% on Illinois (3.19% to 2.99%) and about 9% on New York (3.09% to 2.81%). Your opponents' properties earn correspondingly less from you, while you keep collecting rent.
- Because leaving jail by doubles gives only even moves (2-12), long-jail play favours squares an even distance past Jail: Electric Co (+2), Virginia (+4), St. James (+6), Tennessee (+8), Free Parking (+10) and the Chance square at +12. Odd-distance squares such as States (+3) fall in rank (26th to 29th). Of the orange group, this helps St. James and Tennessee (even distances 6 and 8) relative to New York (distance 9). In Collins' long-jail ranks New York falls from 4th to 7th while Tennessee rises from 7th to 6th.

### Gaps
- I found no published game-theoretic analysis of the optimal switch-over point (when exactly to start staying in jail). Sources give qualitative advice only.

## Q5. Effect of Chance/Community Chest card movement (Advance to Illinois, Go to Jail, nearest-railroad double rent)

### Takeaway
Card movement roughly halves the occupancy of the Chance squares (to about 0.87-1.05%). It boosts the card destinations: GO, Jail, Illinois, St. Charles, Reading, Boardwalk and the railroads near Chance squares. It is a main reason Illinois is the top property and Short Line the weakest railroad. Collins explicitly prices the two double-rent "nearest railroad" Chance cards into railroad income. No retrieved source isolates each card's effect.

### Cited Findings
- Bilisoly: of 16 Chance cards, 9 relocate the player, which makes the Chance squares dark (low probability). Community Chest has only 2 relocating cards (Advance to Go, Go to Jail). Cards are modelled with replacement at 1/16 — [Bilisoly, arXiv:1410.1107](https://arxiv.org/pdf/1410.1107)
- Gartland et al.: "of the 16 possible chance cards, nine of them send you to other spaces and of the 16 community chest cards, two of them send you to other spaces". The "go back three" card from the last Chance square lands on Community Chest and triggers a second draw, which needs extra states — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Collins: railroad income values "include the extra income from the two Chance cards that result in double railroad rent" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Collins: utility rent depends on the dice roll that lands you there. "the average roll when landing on a utility is a bit lower or higher than 7 depending on the utility and the jail strategy". His Chance and Community Chest income omits building-repair cards and the $50-per-player cards — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- Stewart: Short Line (2.44%) lags the other railroads (2.91-3.06%) because "Short Line does not feature a CHANCE card" — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Inverse: light blue is boosted "in part because of the chance card that sends players to the Go space" — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- Card edits over time: the "Advance to Illinois Avenue" card now says you collect $200 if you pass GO. Other card texts were renamed (e.g., "Xmas fund" became "holiday fund") — [Wikipedia: Monopoly (game)](https://en.wikipedia.org/wiki/Monopoly_(game))
- Edition differences:
  - Bewersdorff: results "are based on the American edition", and other editions such as the German one show "minor differences coming from other sets of 'Chance' and 'Community Chest' cards" — [Bewersdorff](https://bewersdorff-online.de/amonopoly/)
  - The UK Leicester paper models "a 2/16 or a 10/16 chance to jump" for Community Chest and Chance respectively — [Leicester J. Phys. Special Topics](https://journals.le.ac.uk/index.php/pst/article/download/3991/3419/12120)

### Inferences
- Possible discrepancy in the card count. Bilisoly and Gartland count 9 of 16 relocating Chance cards. Collins models two nearest-railroad cards, and the UK paper uses 10/16. From general knowledge of the classic US deck, which I did not verify against a source here, the Chance movement cards are: Advance to Go, Illinois, St. Charles, nearest Utility, nearest Railroad (x2), Go Back 3, Go to Jail, Reading Railroad, and Boardwalk. That is 10. The 9/16 models may treat only one nearest-railroad card, which would slightly understate railroad traffic. The script-writer should verify against a current US Chance deck before quoting "9" or "10".
- The low Chance-square figures (0.87%, 1.05%, 0.87%) against about 2.5% for normal squares imply that roughly 60% of arrivals at Chance are moved on: 9-10 of 16 cards. This matches the 9-10/16 counts.

### Gaps
- No retrieved source quantifies the isolated marginal effect of individual cards. For example, how much "Advance to Illinois" alone adds to Illinois, or how much the two nearest-railroad cards add to Reading, Pennsylvania RR and B&O.
- No side-by-side US vs UK card-deck comparison with numbers was found.

## Q6. House shortage (32 houses, 12 hotels) and the value of building to 3 houses

### Takeaway
The standard set has 32 houses and 12 hotels. The rules auction the last houses when demand exceeds supply and forbid building when none remain. Popular math sources (Fry, Inverse) recommend building three houses fast, or even four, partly to block opponents. No rigorous quantitative model of the shortage was found.

### Cited Findings
- "A standard Monopoly set contains 32 houses and 12 hotels." "If there are no houses left, players may not build any and must wait for other players with houses to return them to the bank." "If there is more demand for houses to be built than houses remaining in the bank, then an auction is conducted" — [Wikipedia: Monopoly (game)](https://en.wikipedia.org/wiki/Monopoly_(game))
- Fry & Oléron Evans: building three houses early and waiting to upgrade "has the added advantage of potentially blocking the building projects of other players" because "there are a limited number of houses in the game" — [TED Ideas (Fry & Oléron Evans)](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- Inverse: "there's an argument to be made for buying four, and the math behind it is as simple as counting to 32 ... get your four-house clusters early so they don't ever get to buy some of their own". Also: "For some properties, like the orange block, a fourth house and even a hotel can be worthwhile because your opponents are more likely to land there" — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- Gartland et al.: "the additional rent gained by adding a house increases until three houses are erected. Thus, a 3-house develpoment is the most cost efficient" — [Gartland et al. 2014](https://pi4math.web.illinois.edu/wp-content/uploads/2014/10/Gartland-Burson-Ferguson-Markovopoly.pdf)
- Friddell (via Stewart): "Three houses produces a definite improvement" over two or fewer, which take "around 20 moves or more to break even" — [Stewart, Oct 1996 PDF](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)
- Collins: "the third house has the fastest payoff of any building on almost all of the properties" — [Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)

### Inferences
- House arithmetic:
  - A 3-property set at 4 houses uses 12 houses, so two such sets use 24 and leave 8.
  - Three 3-property sets at 4 houses would need 36, more than the 32 supplied.
  - At 3 houses, three 3-property sets use 27 houses, leaving only 5 for everyone else.
  - Upgrading to hotels returns 4 houses per property to the bank, which frees supply for opponents.
  - So holding at 3-4 houses rather than building hotels can be a deliberate denial tactic. This logic follows from the rules; no math paper models it.
- The economics support pausing at 3 houses outside orange. For example, Collins' green 4th house takes 37-40 rolls against 14.5-14.9 for the third. On orange, the 4th house and hotel still repay in 16-18 rolls, so the shortage tactic and the ROI case both favour going to 4 houses on orange.

### Gaps
- I found no published Markov or simulation study that quantifies the win-probability effect of house-shortage blocking. Claims about the 32-house limit are rule-based and qualitative.
- I found no analysis of the house-auction mechanism's strategic value.
