# Our own Monopoly simulations (original work for this project)

All numbers below come from our own simulator, written for this project (code and raw JSON in
`/home/user/nhl-sog-predictor/monopoly/sim/`; every table regenerates with `python3 analyze.py`, full tables in
`monopoly/sim/results/summary.md`, charts in `monopoly/sim/results/charts/`). About 36 million complete games
in the fixed experiments plus about 20 million inside the genetic-algorithm searches: roughly 56 million games.

## Method

**Engine** (`sim/monopoly.cpp`, C++, ~25 million turns per second on 4 cores). Standard US board and current
US card decks, official rules: doubles roll again and a third double goes to jail; both 16-card decks are
shuffled once and cycle (cards go back on the bottom; Get Out of Jail Free cards are held until used, so the
deck is 15 cards while one is out); jail = pay $50, use a card, or roll doubles for up to three turns, then pay;
rent doubled on unimproved full sets (even if another lot in the set is mortgaged); railroads 25/50/100/200;
utilities 4x/10x dice; the two "advance to nearest railroad" Chance cards pay double and the "nearest utility"
card pays 10x a fresh roll; even building; a bank of 32 houses and 12 hotels; mortgages at half price, lifted at
+10%; buildings sell back at half; bankruptcy to a player hands over everything, bankruptcy to the bank puts
the property back up for auction; **unbought property is auctioned** (official rule).
Options: no auctions, Free Parking jackpot (taxes and fees go to the middle, landing on Free Parking collects),
$400 for landing exactly on Go, trading on/off.

**Trading model.** (a) *Set-for-set swaps*: two players swap the pieces that complete a colour set for each of
them, with cash to balance by each side's valuation. "Fair" traders value by list price, so they accept any
swap balanced by list price; a "Shark" values by an estimate of what each set is worth. (b) *Cash buyouts*: a
player who owns part of a set buys the missing pieces from the one player who holds them. Fair players sell at
2.5x list price and pay up to 3x. Worlds:
- **official** = official rules + swaps + cash buyouts (main world; essentially every game finishes)
- **swaponly** = official rules + swaps only (the trading model Perfect Play used)
- **notrade**, **noauction**, **family** (no auctions + Free Parking jackpot), **kitchen** (family + $400 on Go)

Games end when one player is left; a game still going after 1,000 rounds is a draw. Seating is random every
game. Win rates are outright wins.

**The cast** (each faces three "Yes-Men" unless stated):
- Yes-Man = buys everything it lands on if it has the cash, builds hotels as fast as cash allows (no reserve),
  leaves jail at once, fair trader. This is our "typical player" and the opponent field.
- Scrooge (buys nothing); Railroad Tycoon (railroads only); Utilities only; Railroads + utilities;
  one-colour-only players (Brown only ... Boardwalk Dreamer = Park Place + Boardwalk only).
- Accountant (buys everything but keeps $300 back when buying and building).
- Three-House Rule (never builds past three houses); House Hoarder (four houses, never hotels: the
  house-shortage tactic).
- Jailbird (always stays in jail as long as allowed); "out early, in late" (leaves jail while property is still
  for sale, stays once any opponent has houses or four or fewer properties are unowned).
- Stubborn (never trades). Shark (trades using set values).
- All-In (Yes-Man + mortgages loose property to build + out of jail early/in late); this mirrors Perfect
  Play's "All In".
- Champion (our hand-written expert playbook: bids up to 1.1x list, about 2x list (1.1 x 1.8) for a lot that completes its own set, 1.4x
  to block an opponent's set, builds level by level across sets keeping $100, out-early/in-late jail,
  mortgages to build, Shark trading).

## Validation

- **Exact Markov chain** (`sim/markov.py`, independent of the C++ engine): per-roll probabilities reproduce
  Truman Collins' published table to two decimals. Leave-jail-at-once policy: in jail 3.95% (Collins 3.9499),
  just visiting 2.27% (2.2695), combined 6.22%; Illinois 3.19% (3.1858); Go 3.10% (3.0961); B&O 3.07% (3.0659);
  Reading 2.96% (2.9631). Stay-in-jail policy: in jail 9.39% (Collins 9.4569), Illinois 3.00% (2.9929).
  Rolls per turn 1.187 (leave) / 1.166 (stay). Park Place 2.19% vs Illinois 3.19% (Illinois is 46% more likely).
- **Engine vs chain**: the simulator's landings per turn agree with the exact chain to within 0.0002 on every
  property (20 million turns). The railroads run very slightly high in the simulator because a held Get Out of
  Jail Free card leaves 15 Chance cards, so the two nearest-railroad cards come up a little more often; that
  is a real effect of the physical deck, absent from textbook chains.
- **Rules invariants** checked every turn in a debug build (house/hotel counts always 32/12, no negative cash
  for live players, all 32 cards accounted for) across 120,000 games covering every rule option.
- **Replicates Perfect Play's headline result**: in the swap-only world, the never-trading player wins 6.4%
  against three normal players who win 21.7% each (Perfect Play reported 6.5% vs 21%). Draw rate there:
  13.1% of four-Yes-Man games never end (Perfect Play: about 1 in 6). Adding cash buyouts brings draws to
  0.005% (gamesdecoded's 200M-game hobby sim: 1 in 6,652 never ends).

## Results

### 1. The board (exact chain)
Top end-of-roll squares (leave jail at once): Jail 6.22% total, Illinois 3.19%, Go 3.10%, New York 3.09%,
B&O 3.07%, Reading 2.96%, Tennessee 2.94%, Pennsylvania RR 2.92%, Free Parking 2.88%. Least: Chance squares
0.87-1.05%, Community Chest 1 (1.88%), Mediterranean 2.13%, Baltic 2.16%, Park Place 2.19%.
Chart: `charts/board_probability_leave.png`, `charts/board_probability_stay.png`.

### 2. Who collects the rent (four Yes-Men, official rules, 200,000 games)
Share of all rent paid, by group: orange 17.0%, light blue 13.6%, dark blue 12.3%, pink 11.7%, red 10.3%,
brown 9.8%, yellow 9.2%, railroads 7.2%, green 6.5%, utilities 2.2%. By square: Boardwalk 6.6%, **Baltic 6.3%**
(cheap hotels go up fast: Baltic's hotel rent is $450 for $250 of buildings), St. James 5.8%, Park Place 5.7%,
Tennessee 5.6%, New York 5.6%.

### 3. Game length and turn order (four Yes-Men, 200,000 games per rule world)
| Rules | Never end (1,000 rounds) | Avg rounds to finish | Trades per game |
|---|---|---|---|
| official (swaps + cash deals) | 0.0% | 48.2 | 5.6 |
| swaps only | 13.1% | 53.9 | 2.4 |
| no trading | **67.2%** | 76.7 | 0 |
| no auctions | 0.0% | 48.6 | 5.2 |
| family (no auctions + Free Parking jackpot) | 0.0% | 47.3 | 5.4 |
| kitchen (family + $400 on Go) | 0.0% | 46.8 | 5.5 |
- Without trading, two games in three never end.
- Free Parking jackpot does not make games longer here (47.3 vs 48.2 rounds).
- First player to roll wins 28.3% of four-player games, last player 22.1% (first wins 28% more often). By table
  size the first seat wins 50.9% (2 players), 35.7% (3), 28.4% (4), 23.7% (5), 20.4% (6), 16.1% (8).
- Average game: 48 rounds at 4 players, 57 at 2, 42 at 8.

### 4. One strategy against three Yes-Men (official rules, 100,000 games each, 95% CI about ±0.3 pts; fair share 25%)
All-In 27.7%; out early/in late jail 26.7%; Jailbird 25.8%; Yes-Man 25.2% (control); Champion (textbook
Shark trading) 23.5%; House Hoarder 23.2%; Three-House Rule 19.6%; Accountant ($300 cushion) 18.7%; Shark
(textbook values) 18.3%; Orange only 9.8%; Pink only 7.0%; Boardwalk Dreamer 6.0%; Light blue only 5.7%;
Never trades 4.4%; Red only 3.7%; Yellow only 2.7%; Green only 0.9%; Brown only 0.3%; Railroads + utilities
0.2%; **Railroad Tycoon 0.18% (about 1 win in 550 games)**; Utilities only 0.01%; Scrooge 0.01%.
Chart: `charts/leaderboard_official.png`.

### 5. House rules barely change the ranking at four players
Same experiment in each world (win %, vs three Yes-Men):
| Strategy | official | swaps only | no trading | no auctions | family | kitchen |
|---|---|---|---|---|---|---|
| All-In | 27.7 | 24.2 | 8.4 | 27.9 | 27.9 | 27.5 |
| Yes-Man | 25.2 | 21.7 | 8.4 | 25.5 | 25.1 | 25.1 |
| Three-House Rule | 19.6 | 17.9 | 7.4 | 19.3 | 18.6 | 18.0 |
| Accountant | 18.7 | 15.3 | 5.6 | 18.2 | 18.8 | 19.6 |
| Orange only | 9.8 | 1.0 | 1.0 | 10.2 | 9.9 | 9.8 |
| Boardwalk Dreamer | 6.0 | 4.8 | **8.8** | 6.5 | 7.3 | 7.6 |
| Never trades | 4.4 | 6.4 | 8.4 | 4.6 | 4.4 | 4.4 |
| Railroad Tycoon | 0.2 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
- In the no-trading world the Boardwalk Dreamer (8.8%) does as well as buying everything (8.4%): with no trading,
  the only sets that get finished are ones you land on, and dark blue has only two lots.
- Orange-only collapses without cash buyouts (9.8% -> 1.0%): it can only finish the set by buying the last
  piece from whoever holds it.

### 6. Heads-up (two players), the rules DO matter: auctions punish selective buyers
Yes-Man vs X, 40,000 games each (X's win rate): official / family (no auctions + Free Parking)
- Orange only: 18.6% / **52.6%** (orange-only beats buy-everything heads-up when nothing is auctioned)
- Boardwalk Dreamer: 10.3% / 35.0%
- Accountant: 31.9% / 42.7%
- Railroad Tycoon: 0.0% / 0.4%
- Never trades: 43.5% / 41.9% (13-17% draws): heads-up, refusing to trade costs almost nothing
- All-In: 52.7% / 51.7%; Jailbird 51.3% / 50.3%; Three-House 46.9% / 41.4%
Reason: under official rules every lot you decline is auctioned, and with no other bidder your opponent gets it
for $10 (the minimum bid in the engine). Family rules leave declined lots unsold.
Full 12x12 matrices: `charts/h2h_matrix_official.png`, `charts/h2h_matrix_family.png`. Under family rules there
are rock-paper-scissors loops, e.g. Champion beats Orange-only (76%), Orange-only beats Yes-Man (53%), Yes-Man
beats Champion (53%); none under official rules.

### 7. Which group should you never buy? (Yes-Man that skips one group, vs three Yes-Men, 100,000 games each)
Official rules, change vs the Yes-Man's 25.2%: skip orange -5.9 pts; skip light blue -5.8; skip pink -4.1;
skip dark blue -2.5; skip brown -2.3; skip red -1.7; skip railroads -0.6; skip yellow -0.5; skip utilities +0.2;
**skip green +1.4** (the only group you are better off never buying). Family rules give the same order
(orange -4.9, light blue -3.4, pink -2.9, green +1.2).
Chart: `charts/skip_one_group_official.png`.

### 8. Colour-set duels (two players, $1,500 each, one set each, nothing else for sale; 40,000 games per pair)
Building hotels as fast as cash allows, average win rate against the other seven sets: orange 70.9%, pink 62.8%,
red 62.1%, yellow 58.6%, green 48.6%, dark blue 45.2%, light blue 45.1%, brown 0.4%.
Orange beats every other set: pink 56%, red 58%, yellow 62%, green 72%, dark blue 74%, light blue 75%, brown 100%.
Dark blue loses to orange 26-74, to red 30-70, to pink 33-67, to yellow 33-67 and to green 44-56.
Brown wins 0-1% against every set.
If both stop at three houses the order changes: red 69.5%, orange 68.2%, yellow 67.8%, dark blue 59.0%,
green 58.2%, pink 38.0%, light blue 0.9%, brown 0%: cheap sets need hotels to hurt anyone.
Every player gets one set, average win share over every table:
- 3 players (fair 33.3%): orange 59.8%, pink 45.8%, red 42.3%, yellow 38.1%, light blue 27.2%, green 26.0%,
  dark blue 25.3%, brown 1.2%.
- 4 players (fair 25%): orange 53.0%, pink 36.3%, red 33.1%, yellow 27.6%, light blue 17.4%, green 15.9%,
  dark blue 15.8%, brown 0.8%.
- 8 players, all eight sets at one table (fair 12.5%): orange 35.0%, red 19.6%, pink 15.4%, yellow 13.6%,
  light blue 5.7%, dark blue 5.5%, green 5.0%, brown 0.1%.
Charts: `charts/duel_matrix_b5.png`, `charts/duel_matrix_b3.png`, `charts/royale_4.png`, `charts/royale_8.png`.

### 9. Number of players (one strategy vs N-1 Yes-Men, official rules, 60,000 games each; win rate / fair share)
| Strategy | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| All-In | 1.06 | 1.06 | 1.11 | 1.12 | 1.12 | 1.08 | 1.09 |
| Never trades | 0.87 | 0.30 | 0.18 | 0.14 | 0.11 | 0.10 | 0.08 |
| Boardwalk Dreamer | 0.21 | 0.38 | 0.24 | 0.23 | 0.33 | 0.44 | 0.56 |
| Orange only | 0.38 | 0.60 | 0.39 | 0.29 | 0.24 | 0.20 | 0.17 |
| Three-House Rule | 0.95 | 0.84 | 0.79 | 0.75 | 0.75 | 0.73 | 0.74 |
| House Hoarder | 0.98 | 0.94 | 0.93 | 0.93 | 0.92 | 0.89 | 0.90 |
| Jailbird | 1.03 | 1.03 | 1.04 | 1.03 | 1.02 | 0.99 | 1.01 |
| Railroad Tycoon | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
- Refusing to trade is nearly free heads-up and ruinous at big tables (matches the oliverhutch2021 simulator:
  about 1.0x at 2 players, 0.20x at 4).
- Boardwalk-only gets better with more opponents (0.21x -> 0.56x), partly supporting Hannah Fry's point that
  expensive sets pay off with more players, but it never reaches fair share.

### 10. Teaching the Shark (official rules, 100,000 games each, vs three Yes-Men)
The textbook Shark values sets by expected rent at three houses over 25 opponent turns minus building cost
(dark blue $604, orange $582, yellow $579, red $515, green $361, pink $145, light blue $122, brown -$127).
It wins **18.3%**, worse than a naive list-price trader, because it sells the cheap sets (light blue, pink) that
actually win and pays up for dark blue and green.
"Learned" values = what the simulation measured (points of win rate lost by never buying each set, x $100):
orange $595, light blue $584, pink $410, dark blue $246, brown $233, red $168, yellow $46, green -$141.
Same Shark with learned values: **27.6%**. Champion playbook with learned values: **33.1%**. All-In with
learned-value trading: 31.5%. Four learned Sharks at one table: 25.1% each (no free lunch when everyone knows).

### 11. Evolution (genetic algorithm)
24 strategies per generation, 25 generations, fitness = win rate against three Yes-Men (16,000 games each).
Genes: which of 10 groups to buy; cash reserve; auction bid limits (normal, set-completing, blocking); max
building level; building reserve; build order; jail policy; mortgage-to-build; fair vs learned-value trading;
asking price to sell a set-completing lot; max price to buy one.
- Best of generation 0 (random strategies): 26.3%; median 12.8%. Best rose to 37% by generation 9, then plateaued.
- The final top five (re-tested on 200,000 games each: 36.6-36.7%) all agree:
  buy brown, light blue, pink, orange, red and dark blue; **never buy yellow, green, railroads or utilities**;
  keep about $300 when buying on landing but bid up to 1.2x list at auction, 2-3x for a lot that completes your
  set and 2-3x to stop an opponent completing one; build to hotels keeping only $50-100; mortgage loose property
  to build; leave jail early, stay late; trade with learned values; pay up to 6x list for the last piece of a set.
- Evolved champion: **36.7%** against three Yes-Men (fair share 25%: about 47% more wins than average).
- Ablation (undo one habit at a time; 100,000 games each; evolved = 36.9%):
  stop at three houses 25.2% (-11.7: hotels are the single most important habit); buy everything 32.5% (-4.4);
  bid only list price at auction 34.5% (-2.4); also buy green 34.6% (-2.3); always leave jail at once 34.8%
  (-2.2); never mortgage to build 35.7% (-1.2); always stay in jail 35.8% (-1.1); also buy yellow 35.9% (-1.0);
  fair trading 36.0% (-0.9); also buy railroads 36.1% (-0.8); also buy utilities 36.2% (-0.7); no cash reserve
  36.2% (-0.7); no blocking bids 36.4% (-0.5); cheaper buyout limit 36.9% (0).

### 12. The evolved strategy is a parasite (frequency dependence)
Put three copies of the evolved champion at a table with one plain Yes-Man (20,000 games, learned values):
the **Yes-Man wins 71.5%**; each champion 9-10%. An All-In wins 64.9% and even a never-trader wins 60.4%
against three champions. Four champions at one table split 25% each.
Mechanism: the champions all skip yellow, green, railroads and utilities, so the Yes-Man completes yellow in 87%
of games and green in 84%, and railroads take 16.8% of all rent (vs 7.2% normally). The evolved strategy only
works because normal players are there to overpay for the groups it ignores.

### 13. Arms race: the evolution goes in circles
Stage 2 of the genetic algorithm: each round evolves a new strategy (24 x 12 generations, seeded with the
previous champion) to beat a table of three copies of the previous champion. Check games: 50,000 per line.
- Round 0 = the evolved specialist from section 11 (skips yellow, green, railroads, utilities; hotels; bids hard).
- Round 1 winner = a **generalist house-hoarder**: buys everything, never builds past four houses (house-shortage
  tactic), stays in jail, keeps no reserve, will not overpay for a set piece, bids up to 3x list to block.
  It wins **75.4%** against three round-0 specialists, but only **24.4%** against three Yes-Men (average).
- Round 2 winner (search stopped after 10 of 12 generations; best fitness 33.2%) = back to a **specialist**:
  buys brown, light blue, pink, orange, red, yellow, dark blue and utilities, skips green and railroads, builds
  hotels, bids up to 2x list, out of jail early/in late, mortgages to build, keeps $200.
  It wins **33.1%** against three round-1 generalists and **36.0%** against three Yes-Men.
- But the round-1 generalist wins **66.0%** against three round-2 specialists, and a plain Yes-Man wins **61.0%**
  against three round-2 specialists (71.5% against three round-0 specialists).
- Specialists are roughly even with each other (round 0 vs three round-2: 25.9%).
Reading: there is no single best Monopoly strategy. Specialists that skip the overpriced groups beat ordinary
players, but they only work while somebody else is buying those groups; a table full of specialists is easy prey
for a generalist, and the generalist that beats specialists is only average against ordinary players.
Rock-paper-scissors at the level of whole strategies.

### 14. Ecology: let a whole population of players evolve
Pool of 11 strategies: Yes-Man, All-In, Three-House, Accountant, Never trades, Railroad Tycoon, Boardwalk Dreamer,
Orange only, Champion (hand-written expert playbook with learned set values: buys everything, bids up to 1.1x
list, about 2x list (1.1 x 1.8) for a lot that completes its own set, 1.4x to block; builds level by level across sets to hotels keeping
$100; out of jail early/in late; mortgages to build; trades with learned values), Evolved (round 0 specialist)
and Evolved r1 (round 1 generalist house-hoarder). Every possible four-seat table with repeats (1,001 tables,
5,000-10,000 games each, about 5.5 million games), official rules with deals. Replicator dynamics: each
generation every strategy's population share is multiplied by its average win rate at tables drawn at random
from the current population.
Win rate in an evenly mixed population (each seat filled at random): Champion 41.1%, Evolved 40.8%, All-In 35.3%,
Evolved r1 34.3%, Yes-Man 31.2%, Three-House 28.1%, Accountant 22.6%, Never trades 14.3%, Boardwalk Dreamer 10.2%,
Orange only 8.3%, Railroad Tycoon 0.3%.
Population over generations (share): start 9.1% each. Generation 5: Champion 27%, Evolved 27%, All-In 14%,
Evolved r1 14%, Yes-Man 9%. Generation 10: Champion 46%, Evolved 30%, Evolved r1 12%, All-In 9%, Yes-Man 3%.
Generation 20: Champion 63%, Evolved 28%, Evolved r1 6%, All-In 2%. Generation 50 onward: **Champion 71%,
Evolved 29%**, everything else extinct. Railroad Tycoon, Orange only, Boardwalk Dreamer and Never trades are
gone within a few generations; the Yes-Man (the typical family player) is gone by about generation 20.
The end state is a stable mix: the well-rounded expert (buy everything, smart auction bids, hotels, jail
timing, mortgage to build, know what each set is really worth when trading) makes up most of the population,
and the specialist that skips green/yellow/railroads/utilities survives as a minority that feeds off it.
Chart: `charts/ecology.png`.

### Charts available (monopoly/sim/results/charts/)
board_probability_leave.png, board_probability_stay.png, leaderboard_official.png (also _family, _notrade,
_swaponly), rules_change_winner.png, skip_one_group_official.png (and _family), duel_matrix_b5.png,
duel_matrix_b3.png, royale_3.png, royale_4.png, royale_8.png, h2h_matrix_official.png, h2h_matrix_family.png,
players.png, game_length.png, evolution.png, ablation.png, ecology.png, set_completed_win.png.
Sample turn-by-turn game logs for animation: monopoly/sim/results/sample_games/*.jsonl.

### 15. Realistic play (user's rules) and the evolution re-run
Realistic play: every player buys what it lands on; the only trades are set-for-set swaps (each player completes
a set, cash evens up the printed prices); nobody sells a set-completing card for cash. "Fans" buy everything too
but lean toward a favourite (mortgage to buy it, bid up to double for it, build it first, never trade it away).
Typical player vs three typical players: 21.8 wins per 100 games; 13 in 100 four-player games never finish
(57 in 100 six-player games) because a colour gets split three ways and no two-player swap can fix it.
One player vs three typical players (100,000 games each, official rules): Orange fan 24.7, All-In 24.3,
Light blue fan 23.7, Pink fan 23.2, out of jail early/in late 22.7, Boardwalk fan 22.1, Red fan 22.1,
Boardwalk saver ($400 held while dark blue unsold) 21.9, Railroad fan 21.8, Yellow fan 21.5, four houses never
hotels 20.4, Green fan 20.1, Brown fan 19.0, stops at three houses 17.9, keeps $300 back 15.3, never trades 6.3.
Never buying one group: dark blue -5.5, brown -4.4, orange -4.3, light blue -3.9, pink -3.3, red -2.2,
yellow -1.3, green 0.0, railroads +0.1, utilities +0.4. House rules (no auctions, Free Parking money) give the
same order of fans (orange best 24.4, brown worst 18.0).

Evolution under realistic play (`sim/evolve_real.py`): every player buys what it lands on; genes are only real
choices: up to two favourite groups and how hard to chase them, cash kept back, auction bids, when to stop
building, mortgaging to build, jail timing, and how to even up a swap (printed prices, set value, or never trade).
24 strategies x 25 generations, fitness = wins against three typical players (16,000 games each).
- Best rose from 26.1 (random) to about 31 by generation 7 and plateaued. All five finalists, re-tested on
  200,000 games each (30.1-31.0), agree: favourites **light blue and orange**; mortgage other property to buy a
  favourite and bid up to 2x list for one at auction; bid up to 1.5-2x list on other lots at auction (more to
  complete a set); keep only $50-100; build to hotels; mortgage loose property to build; out of jail early, in
  late; even up swaps at printed prices (do not haggle over set value).
- Evolved realistic player: **31.0 wins per 100** vs 21.8 for a typical player (about 42% more wins).
- Undo one habit (100,000 games each, evolved 31.3): never trades 8.3 (-23.0); evens up swaps by set value
  instead of printed price 21.5 (-9.8: haggling kills deals, so fewer sets get made); stops at three houses 26.0
  (-5.3); no favourite colours 26.7 (-4.5); never mortgages to build 27.4 (-3.8); bids only list price 28.7
  (-2.5); always leaves jail 29.8 (-1.4); always stays in jail 30.0 (-1.2); keeps no cash when building 30.8
  (-0.5); keeps no cash back when buying 31.5 (+0.2, no effect).
- **Not a parasite under realistic play**: a typical player seated with three copies of it wins 16.2 per 100
  (less than its usual 21.8); an All-In wins 17.8. Four copies at one table: 18.8 each, 25 in 100 never finish.
- Arms race (each round evolves against three copies of the last champion, 12 generations): round 1 = same
  playbook with favourites pink + orange (21.2 vs three round-0 players, where about 18.8 is par); round 2 =
  same playbook with favourites light blue + red (22.1 vs three round-1). No cycle: only the favourite pair
  drifts. Cross-checks (50,000 games each): against typical players round 0/1/2 win 31.1/30.6/28.1; a typical
  player against three of each wins 16.3/16.7/18.1; evolved variants against each other sit near par, with a
  small bonus for the odd one out whose favourites nobody else at the table is chasing (e.g. round 1 vs three
  round 2: 25.7, par about 20).
- Realistic ecology (12 players, all 1,365 four-seat tables, 4,000 games each): evenly mixed win rates:
  evolved pink+orange 26.8, evolved light blue+orange 26.5, evolved light blue+red 25.2, All-In 21.7, Orange fan
  21.7, Boardwalk saver 20.3, Railroad fan 20.0, typical 19.9, Expert playbook (haggles by set value) 18.0,
  stops at three houses 16.6, keeps $300 back 14.3, never trades 6.3. Population end state: the three evolved
  players only, 48% light blue+orange, 38% pink+orange, 14% light blue+red; everyone else extinct by about
  generation 25-50. Different favourite pairs coexist because players do better when their favourites are not
  contested. Charts: `charts/evolution_realistic.png`, `charts/ablation_realistic.png`,
  `charts/ecology_realistic.png`.
