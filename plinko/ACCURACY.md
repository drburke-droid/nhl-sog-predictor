# What's measured, what's estimated, and how sure we are

This page covers every number in the Plinko film, the Short and the interactive Drop Lab: where it
comes from and how much to trust it. Confidence levels: **solid** (published or counted),
**measured** (from broadcast frames, with an error bar), **estimated** (engineering judgement), and
**fitted** (tuned to the show's own results).

## 1. The board

No production drawing of the TV board is public. Searches of fan wikis, news features, prop-replica
listings and set-builder pages turned up no official dimensions. Retail "Price Is Right Plinko" boards
are scaled-down toys and say nothing about the set. So the layout is counted and the sizes are measured
from broadcast frames, using the chip as the ruler.

| Quantity | Value | Basis | Confidence |
|---|---|---|---|
| Slots | 9, valued (left to right) $100, $500, $1,000, $0, $10,000, $0, $1,000, $500, $100 | Price Is Right wiki; every frame | solid |
| Centre slot history | $5,000 from 3 Jan 1983; $10,000 permanent from 15 Oct 1998 | Price Is Right wiki | solid |
| Drop positions | 9, one over each slot | frames; Biesterfeld (2001) | solid |
| Pin rows | 13, alternating 8 pins (between slot centres) and 7 pins (over slots 2 to 8) | counted on 2017 and 2023 frames; Marasco and Assefa (2020) say "thirteen rows of pegs" and their Figure 1 draws 13 | solid |
| Pins | 98 (7 rows x 8 + 6 rows x 7) | follows from the rows | solid |
| Pins met on the way down | 12 (the top row only frames the drop gaps) | geometry; Biesterfeld (2001): "it will encounter 12 pegs" | solid |
| Side walls | sawtooth: 6 triangular teeth a side, each reaching to the slot 1 (slot 9) centre line on the 7-pin rows, where an end pin would otherwise be | 2017 close-up frame; the schematic in Marasco and Assefa (2020), Figure 1, shows the same 6 teeth a side in the same rows | solid |
| Slot dividers | 8 thin plates directly below the 8 pins of the bottom row | 2017 close-up frame | solid |
| Chip diameter | 13.0 cm (5.1 in) | 13.5 cm scaled against a model's hand in a 4K frame; replicas sold as 5 in | measured, +-0.5 cm |
| Chip thickness | about 19 mm (3/4 in) | replica listings | estimated (does not enter the 2D physics) |
| Chip-to-slot ratio | 0.73 | 95 px chip in a 129.7 px slot pitch, near-frontal 2017 close-up | measured, +-0.02 |
| Slot pitch (= pin spacing in a row) | 17.8 cm (7.0 in) | chip / 0.73 | measured, +-0.7 cm |
| Row spacing | 15.4 cm (6.1 in), equilateral triangles | the plexiglass cover is cut as triangles; the close-up shows 0.77 of the pitch, foreshortened by the lean of the board and the camera looking up, consistent with an equilateral 0.87 | estimated, 0.80 to 0.87 of the pitch |
| Pin diameter | 9.5 mm (3/8 in) steel | frames | estimated, 6 to 10 mm |
| Playfield width | 160 cm (63 in) wall to wall | 9 x pitch; consistent with the host (178 cm) standing beside the board | measured, +-6 cm |
| Top pin row to bottom pin row | 185 cm (73 in) | 12 x row spacing | estimated, 171 to 185 cm |
| Top pin row to slot floor | 207 cm (81.5 in) | plus 1.23 pitches measured to the slot floor | estimated |
| Lean of the board | 10 degrees back from vertical | side-on frames; the chip slides on the backboard | estimated, 5 to 20 degrees |

Cross-check: Drew Carey is about 178 cm (5 ft 10 in). In the 2026 wide shot he stands a little in front
of the board and the playfield measures about 140 cm wide at his depth, so the true width is somewhat more,
consistent with 160 cm. The whole set piece, with its sign and sunburst frame, is about 3 m (10 ft) tall,
which matches the one hobby source that gives a figure for the TV board.

## 2. The game

* One free chip plus up to four more, one per small prize. Each small prize shows a two-digit price with
  one digit wrong; the contestant says which digit is right (Price Is Right wiki). Biesterfeld (2001)
  describes the older higher/lower format; the digit format is the current one.
* A chip stuck on the board is knocked free and dropped again (since 2007, one stuck on the top row
  counts). The simulator re-drops stuck chips the same way.

## 3. The show's own results (the calibration target)

* **1,029 chips, seasons 44 to 54 (2015-16 to 2025-26, the last in progress)**, from tpirstats.com's
  per-season Plinko pages, downloaded 13 Nov 2026 (`data/real_outcomes.json`). Fan-compiled. Counts are
  by slot position, so special episodes with bigger values count by where the chip landed.
  * Left to right: 61, 123, 144, 141, **156**, 130, 132, 98, 44.
  * $10,000: 15.2%. Average: $1,902 a chip at the standard values.
  * Drop positions are **not** recorded anywhere I could find. A Golden Road forum thread titled
    "Plinko Chip Placement Statistics" may have them, but the forum blocks automated access.
* Bowen Kerins' tally for 2000 to 2011 (via WJBQ, 2016): 308 games, 1,227 chips, 176 in the $10,000
  (14.3%), $1,805 a chip. It agrees with the newer data.
* Small prizes, same seasons: in 1,040 items the last digit was the correct one 630 times (60.6%).
  Contestants were right 73.9% of the time overall: 81.6% when the last digit was correct, 62.2% when
  the first was.
* There is a mild left-right imbalance (469 chips left of centre, 404 right; about 2.2 standard errors).
  The simulator is symmetric and ignores it. It could come from where contestants like to drop, or from a
  board that is not perfectly level.

## 4. The two models

**Coin flips (the textbook).** Each pin sends the chip half a slot left or right with probability 1/2;
at a wall tooth the chip can only go inward. This is the model in Biesterfeld (2001) and Hastings (1997),
and `galton.py` reproduces their numbers exactly: a chip dropped in the middle wins $10,000 with
probability 924/4096 = 22.56% and averages $2,557.91; one slot off, $2,265.92. Marasco and Assefa (2020)
quote $2,555.32 for the same drop. The gap comes from their approximate "fold" at the walls (78 and 221 paths
to the $100 and $500 slots, where the exact chain gives 66 and 232) and from a typo (4069 for 4096 in two
terms). Their own 5,000-drop coin-flip simulation from the middle, 23.0% in the centre slot, agrees with the
exact 22.56% within its noise.

**Physics (`physics.py`).** A 2D rigid disc with spin, moving under gravity along the leaning board, with
sliding friction against the backboard. It makes impulse contacts with the round pins, the sawtooth walls
and the dividers, with a coefficient of restitution *e* and Coulomb friction (0.2) that couples slide and
spin. It integrates with 0.1 ms steps; a chip makes about 30 to 40 impacts on the way down, which takes
about 4 s. Real pins are not perfect, so each impact's contact normal is tilted by a random 3 degrees
(standard deviation) and its restitution varies by +-0.05. Without this, low-bounce chips lock into
unrealistic repeating channels. Releases scatter by 1 cm, 5 cm/s and 1 rad/s, like a hand.

Two settings are tuned rather than measured:
* **Backboard friction (mu = 0.10 at a 10 degree lean)** keeps the stuck-chip rate near 1%, which matches
  the occasional "stuck chip" on air. Higher friction left 5 to 20% of chips balanced on pin tops.
* **Bounciness e = 0.50 (fitted).** For each e from 0.30 to 0.575 the model dropped 6,000 chips from
  each position. Then the drop habits of contestants (a one-parameter family from "everyone drops in the
  middle" to "drops spread evenly"), and separately any symmetric mix of drop positions, were fitted to
  the 1,029 real chips (multinomial deviance over the five prize values, 4 degrees of freedom).

| Model | Best fit | Deviance | What it needs contestants to do |
|---|---|---|---|
| Physics, e = 0.50 | everyone drops in the middle | 0.6 | aim at the middle (what you see on the show) |
| Physics, e = 0.30 to 0.45 | spread-out drops | 8 to 29 | drop widely; fits poorly in the middle of that range |
| Coin flips | spread-out drops | 2.9 | only about 21% of drops in the middle and 27% at positions 1, 2, 8 or 9 |

**So the data alone cannot settle it.** If most contestants aim near the middle, the board must be about
as chaotic as e = 0.5. If they really spread their drops across the board, coin flips also fit. Physics
favours the first: hard plastic on steel pins at 1 to 2 m/s typically has e of 0.4 to 0.7, and e = 0.5
gives a 4 s descent with about 33 impacts. Video logs of drop positions would settle it; see section 3.
The film says which way it leans and why, and keeps the coin-flip numbers on screen as the other story.

## 5. Results and their noise

At the tuned bounciness (24,000 drops per position; standard error of an average about $22):

| Drop over | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Physics, average per chip | $1,617 | $1,596 | $1,735 | $1,828 | $1,829 | $1,828 | $1,735 | $1,596 | $1,617 |
| Physics, P($10,000) | 12.1% | 12.0% | 13.4% | 14.5% | 14.5% | 14.5% | 13.4% | 12.0% | 12.1% |
| Coin flips, average | $780 | $1,009 | $1,606 | $2,266 | $2,558 | $2,266 | $1,606 | $1,009 | $780 |
| Coin flips, P($10,000) | 3.2% | 5.7% | 12.1% | 19.3% | 22.6% | 19.3% | 12.1% | 5.7% | 3.2% |

* The middle is the best drop, or tied for best, in every model and every sensitivity run. On the tuned
  board positions 4, 5 and 6 are a statistical tie.
* Five chips, all dropped in the middle (tuned model, 200,000 simulated games): $9,137 on average, a 54%
  chance of at least one $10,000, and a 0.17% chance of $0 from the board.
* Sensitivity (`results/sensitivity.json`, 4,000 drops per position, so about +-$55 on each average):
  best drop was position 4, 5 or 6 in 16 of 17 runs; in the 17th (a 2 mm steady hand) position 3 came out $127 ahead, 1.7 standard errors, and the 24,000-drop technique test below puts that hand at $1,826 from the middle. The edge always paid less than the middle, by $18 to $857 (the smallest gaps are within the 4,000-drop noise of about $75 on a difference); the gap depends mostly on bounciness (e = 0.45 gives $857, e = 0.55 gives $318). Per run:

  * baseline: middle $1,750, edge $1,732, best position 4
  * e -0.05: middle $2,216, edge $1,359, best position 5
  * e +0.05: middle $1,859, edge $1,541, best position 5
  * pin jitter 1 deg: middle $1,754, edge $1,663, best position 6
  * pin jitter 6 deg: middle $1,804, edge $1,619, best position 5
  * peg friction 0.1: middle $1,744, edge $1,680, best position 5
  * peg friction 0.4: middle $1,729, edge $1,684, best position 6
  * tilt 5 deg: middle $1,792, edge $1,754, best position 5
  * tilt 20 deg: middle $1,854, edge $1,627, best position 5
  * backboard friction 0.05: middle $1,814, edge $1,732, best position 5
  * backboard friction 0.25: middle $1,951, edge $1,528, best position 5
  * uniform-disc chip (I=0.5mR^2): middle $1,856, edge $1,656, best position 5
  * rim-heavy chip (I=0.9mR^2): middle $1,803, edge $1,575, best position 5
  * sloppy hand (3 cm): middle $1,796, edge $1,659, best position 6
  * steady hand (2 mm): middle $1,728, edge $1,698, best position 3
  * tossed (0.5 m/s): middle $1,989, edge $1,672, best position 5
  * spun (10 rad/s): middle $1,868, edge $1,529, best position 5
* Release technique (`results/technique.json`, 24,000 drops each over the middle, +-$22): normal release (1 cm scatter) $1,792; steady hand (2 mm scatter) $1,826; sloppy hand (3 cm scatter) $1,833; thrown down at 1 m/s $1,832; spun hard (10 rad/s) $1,835; aimed 2 cm off centre $1,822; dropped from 15 cm higher $1,826.
* Chaos (`results/chaos.json`): two chips released 0.001 mm apart, with identical pin imperfections, land in the same slot 19% of the time; 1 mm apart, 14%. Two unrelated chips from the middle match 12% of the time. The film's twin pair is drawn with pin imperfections switched off, so only the 0.001 mm differs.

## 6. What is not modelled

The plexiglass cover (chips occasionally touch it), out-of-plane wobble, chip wear and weight differences
between the 10 official chips, pins bent over 40 years, and air. The pin-imperfection term stands in for
all of them together. The board is treated as perfectly level and symmetric.

**Hand releases past the top pins.** The hand-placement scatter (1 cm) puts 5.4% of simulated releases
further from the gap centre than the chip can physically fit (+-1.92 cm), where it starts overlapping a top
pin and is pushed out. Clipping those releases to the edge of the gap and re-running the same 40,000 drops
changes the middle's average by $4 ($1,822 to $1,817) and the edge's by $5 ($1,644 to $1,639), and the middle's
$10,000 chance from 14.25% to 14.20%. That is far inside the noise, so the strategy results stand; the chaos
experiments in section 7 use only valid releases.

## 7. Could perfect aim beat it? Chaos on the board (`chaos2.py`, `releasemap.py`)

The question: find a release point that wins $10,000, then replay it as precisely as you like. Does it win
again? To test it the board is made a perfect, repeatable machine: no pin imperfections, no random
restitution, the chip let go at rest with no spin. Only what is named changes.

**Corrections.** Two earlier claims were wrong, in opposite directions.

1. The first Blender cut said micron accuracy is "no better than random" (twins 1 um apart share a slot 14% of
   the time, against 12% for unrelated chips). Measured over every valid release point it is slightly better
   than random, not equal to it: 18% against 15% (below). The spirit of the claim holds.
2. The first chaos cut said a micron-accurate replay wins again 57% of the time and an atom-accurate one 71%.
   Those numbers came from a bug: the candidate releases were spread over the middle 6 cm, but the chip's
   centre can only be let go within +-1.92 cm of the gap centre (half the 17.8 cm pin pitch, less half a pin
   and half the 13 cm chip). Outside that, the simulated chip started overlapping a top pin and was shoved out
   the same way every time, which made replays look far more repeatable than they are. `chaos2.py` now draws
   releases only from where a chip can be let go (with 0.1 mm clearance), and every number below comes from
   those runs. (The middle-gap replay at 1 cm error could still start some replays overlapping a pin; the
   whole-top run redraws those, and the two agree.)

* **Where a player can let go.** On the show a player may let go anywhere across the top: right at the top
  pins or anywhere in the roughly 12 in (30 cm) above them, wall to wall. The simulated board has open space
  above the top row between its side walls, so `chaos2.py regionmap` and `region` use that whole area: chip
  centres between the walls (less a chip radius), from the top pins' centre line up to 30.48 cm above, and not
  overlapping a top pin. Releases are at rest. The bullets below marked "middle gap" use only the line through
  the middle gap at the pins (+-1.92 cm), which is one slice of that area.
* **The whole-top release map** (`chaos2.py regionmap`, the film's map). Every point of a 0.95 mm grid over that
  area (425,250 valid release points) dropped on the perfect board: 12.9% win $10,000, sprinkled
  over the whole top rather than gathered in the middle: 14.7% of the points over the middle
  (within half a pitch of position 5) win, 11.4% of those over positions 1 and 9. Height above the pins
  barely matters (11% to 15% in every band from 0 to 30 cm). 0.9% balance on a pin (counted as not
  winning). On this perfect board a release at rest carries little aim past the first few pins; the
  where-to-drop results (sections 4 and 5) come from the tuned model with real hands, where the middle's edge
  is also modest.
* **Replays, whole top** (`chaos2.py region`; the film's replay and temperature charts). 6,000 releases spread uniformly over the
  area: a fresh drop wins 13%. The first 120 winners are each replayed 300 times with a random
  release error of the given size in both directions (a replay that would start inside a pin or wall is
  redrawn). Win again: 1 cm 13%, 1 mm 14%, 0.1 mm 14%, 10 um 15%,
  1 um 16%, 100 nm 19%, 10 nm 23%, 1 nm 28%, 0.1 nm 39%,
  10 pm 55%, 1 pm 67%, 0.1 pm 79%, 10 fm 87%, 1 fm 95%.
  Fitted over 1 fm to 1 nm, misses scale as precision^0.19: each tenfold gain removes about
  35% of the remaining misses. The chip's own thermal motion: 49%. Within +-0.1 nm,
  13% of winners repeat more than 90% of the time (islands), 42% less than 30%. Board warmed by a
  fixed dT, all 779 winners replayed exactly: 1e-9 C 94%, 1e-6 C 64%, one drop's heat
  (5.7e-05 C) 39%, 1e-3 C 24%, 0.01 C 19%, 1 C 12%. All within a few points of
  the middle-gap results below (slightly more sensitive), so where you let go doesn't change the story.
* **Replays, middle gap** (`chaos2.py replay`). 2,000 candidate releases spread uniformly over the valid +-1.92 cm;
  the first 80 that win are each replayed 300 times with a Gaussian release error of the given size.
  A fresh drop anywhere in the gap wins 15% of the time. Replays win again:
  1 cm 17%, 1 mm 16%, 0.1 mm 16%, 10 um 17%, 1 um 18%,
  100 nm 19%, 10 nm 23%, 1 nm 31%, 0.1 nm (about one atom) 43%,
  10 pm 59%, 1 pm 71%, 0.1 pm 84%, 10 fm 92%, 1 fm 98%.
  At 1 um not one of the 80 winners repeats more than 27% of the time; at 0.1 nm they range from 0% to 100%.
  Better precision always helps, but slowly: fitted over 1 fm to 1 nm, the miss chance scales as
  precision^0.24, so each tenfold gain removes only about 42% of the remaining misses. This is
  final-state sensitivity, and the fitted power is what Grebogi, McDonald, Ott and Yorke (1983) call the
  uncertainty exponent.
* **Islands and knife edges, middle gap** (`chaos2.py film`). Probing 201 releases within +-0.1 nm of each winner:
  15% of winners repeat more than 90% of the time (islands), 45% less than 30%, the rest in between. The
  release strip in the film (20,001 points across the whole +-1.92 cm) is 15% gold. Release maps around the
  most robust winner (at +1.05 cm) and around the opening's winner (at -1.90 cm, which repeats 43% of the time
  within +-0.1 nm), 4,001 points per zoom over windows of 2 mm, 20 um, 200 nm, 2 nm and 20 pm: the island is
  14%, 17%, 19%, 88% and 100% gold, so it only turns solid within a few atoms; the knife edge is 11%, 11%, 13%,
  20% and 36% gold, mixed at every zoom. At any scale a hand could control, both look like a random drop. One
  win can't tell you which kind you found.
* **How fast differences grow** (`chaos2.py growth`, 156 twin pairs starting 1 pm apart). While the gap
  grows from 10 pm to 1 mm it multiplies by about 2.15 per bounce (middle half
  1.98 to 2.54), doubling every
  109 ms. A drop has about 36 bounces. Dividing a 1 cm
  tolerance by that factor per bounce gives the film's "how far ahead" bars (1 mm: 3 bounces, 1 um: 12, 1 nm:
  21, 1 atom: 24) and the 15 fm needed for a whole drop. These are typical
  values: islands grow slower, knife edges faster.
* **One variable: the board's temperature, middle gap** (`chaos2.py onevar`). 6,000 releases across the valid gap give
  892 winners (14.9%). Each is replayed exactly (the same release to the last bit,
  which wins every time on the same board) on a board warmed by a fixed dT: uniform expansion of 1e-5 per kelvin
  about the centre line, pins and chip unchanged. Share that still win: 1e-9 C 98%,
  1e-8 C 95%, 1e-7 C 88%, 1e-6 C 76%,
  1e-5 C 60%, 1e-4 C 44%, 1e-3 C 35%,
  0.01 C 23%, 0.1 C 17%, 1 C 17% (a fresh drop: 15%).
  The heat of one drop: the chip falls 1.97 m (vertically) from the top row to rest on the slot
  floor and stops, so its 4.8 J of potential energy all becomes heat. Spread evenly through an
  assumed 50 kg wooden board (1,700 J/kg/K), that is 5.7e-05 C, and the 2.5 m board grows by
  1.4 nm. On a board that much warmer 49% of winners win again. This is
  an upper bound on the board's share of the heat (some warms the chip, some leaves as sound) but uniform
  warming is also the gentlest way to spread it: the pins the chip actually hits warm far more, locally. The
  film's pair (release -0.93 cm from the gap centre) wins $10,000 on the cool board and
  lands in slot 6 ($0) on the warmed one, parting about 2.6 s into the drop.
* **The real world** (perfect release, one thing varied). The chip's own thermal motion (equipartition, for a
  0.25 kg chip: 0.13 nm/s and 2.5 nrad/s at random): 55%. A board 0.001 C warmer or cooler (expansion 1e-5
  per kelvin, about 20 nm over 2 m, scaled about the board's centre line): 38%; 0.0001 C: 52%; 0.01 C: 26%;
  0.1 C: 21%; 1 C: 18%. Uniform expansion is a smooth change, so a little memory survives even at 1 C.
* **The earlier release map** (`releasemap.py`, 20,001 points per zoom from the 9 cm gap down to 1 nm, centred
  on the edge of a winning stretch near dead centre) and **twin test** (4,000 pairs within 0.5 mm of centre)
  still stand for that region: same slot 95% of the time after a one-unit rounding nudge (so the computer's
  own rounding barely matters), 48% at 1 pm, 24% at 0.1 nm, 14% at 1 um.
* **Dead centre sticks.** On the perfect board a chip let go at rest within 0.12 mm of dead centre lands
  squarely on the first pin and balances there (backboard friction holds it). A real hand always adds some
  sideways motion or spin, so real chips almost never do this (the tuned model sticks about 1% of the time).
* **Why so sensitive.** Round pins defocus trajectories the way a convex mirror spreads light, as in the
  chaotic billiards studied since Sinai. Energy loss at each bounce lets some paths settle into stable
  channels, which is where the islands come from.
* **What this leaves out.** A 2D rigid body with a fixed 0.1 ms step; no air, no out-of-plane wobble, no chip
  wear. The chip mass (0.25 kg) is an estimate; at 0.1 kg the thermal speeds are 1.6 times larger.

## Sources

* Price Is Right wiki, "Plinko" (rules, values, history, records): https://priceisright.fandom.com/wiki/Plinko
* tpirstats.com, Pricing Game Stats: Plinko, seasons 44 to 54: http://tpirstats.com/Season48/Pricing/Plinko.htm (and Season44 ... Season54)
* Biesterfeld, A. (2001), "The Price (or Probability) Is Right", *Journal of Statistics Education* 9(3): https://jse.amstat.org/v9n3/biesterfeld.html
* Marasco, B. and Assefa, N. (2020), "So, You Want to Win Plinko?", *Journal of Research in Progress* 3, pp. 79-88, Howard Community College: https://pressbooks.howardcc.edu/jrip3/chapter/so-you-want-to-win-plinko/ (PDF of the published volume read in full)
* Parsons, J. (2016), "A Friend of Mine Confirmed The Best Place to Drop a Plinko Chip Using Math" (Bowen Kerins' 2000 to 2011 tally), WJBQ: https://wjbq.com/a-friend-of-mine-confirmed-the-best-place-to-drop-a-plinko-chip-using-math/
* Looper, "What You Probably Didn't Know About The Plinko Chips" (10 official chips, kept locked): https://looper.com/928265/what-you-probably-didnt-know-about-the-plinko-chips-on-the-price-is-right
* BroBible, "6 pieces of Plinko trivia" (metal pins, plexiglass cover with triangular holes): https://brobible.com/guyism/article/plinko-price-is-right-history/
* Replica chip listings (5 in diameter, 3/4 in thick): e.g. https://www.amazon.com/Price-Right-Plinko-Chips-diameter/dp/B0D1NHCBFJ
* CelebHeights, Drew Carey (5 ft 10 in): https://celebheights.com/s/Drew-Carey-1195.html
* Grebogi, C., McDonald, S. W., Ott, E. and Yorke, J. A. (1983), "Final state sensitivity: an obstruction to predictability", *Physics Letters A* 99(9), 415-418.
* Lorenz, E. N. (1963), "Deterministic nonperiodic flow", *Journal of the Atmospheric Sciences* 20(2), 130-141 (the 0.506 / 0.506127 rerun is from Lorenz's own account, *The Essence of Chaos*, 1993).
* Broadcast frames used for measurement: Price Is Right wiki galleries for the 9 Oct 2017, 27 Jan 2023 and
  3 Nov 2026 playings (not redistributed here).
