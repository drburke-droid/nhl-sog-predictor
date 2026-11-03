# Video plan: how to actually win Monopoly (the way people really play)

This replaces the video plan in `Monopoly winning strategies.md`. Everything here comes from the realistic
simulations: every player buys what they land on, the only trades are swaps where both players complete a
set (cash evens up the printed prices), and nobody sells the card that completes someone else's set for cash.
Numbers are wins per 100 games against three "typical" players, from 100,000 simulated games per line unless
noted. Full tables: `sim/results/summary.md`; charts: `sim/results/charts/`.

## The pitch

Every Monopoly table has the same arguments. Do I spend all my money on houses or keep some back in case I
land on a hotel? I've got two monopolies, which one gets the houses? Do I put hotels on one, or a house on
everything? Should I trade? Is it worth chasing Boardwalk, or the railroads?

We built a simulator that plays Monopoly the way families do, ran 46 million games, and answered
each argument with a number. Then we let a computer evolve its own player under the same rules and asked
what habits it ended up with. Perfect Play's video (October 2026, about 2.7M views) ranked bot strategies;
this one answers the decisions a real player faces on their own turn, under the rules they actually play.

The three headline answers:

1. **Spend it.** Every dollar you hold back instead of building costs you wins, and it doesn't even make you
   safer: players who keep $300 back go bankrupt first *more* often (23.6%) than players who spend it all
   (21.8%). The only cushion worth keeping is about $100.
2. **Two sets? Build the one with the cheaper houses first** (orange or light blue beat everything; brown is
   the exception). Picking the wrong one costs up to 20 wins in 100.
3. **Never put one house on everything.** Spreading evenly was the best choice in 0 of 56 situations tested.
   Build one set up first. Cheap sets ($50 houses): straight to hotels. Expensive sets ($150-200 houses):
   three houses, then build up your other set, then hotels.

## How we model a real player (show this on screen in the first two minutes)

- Everyone buys what they land on if they can afford it. Unbought property goes to auction (official rule);
  every building result was re-checked with no auctions and Free Parking money and held up.
- You get the card you need by landing on it or by a swap: I give you the card that finishes your set only if
  you give me the card that finishes mine. Nobody sells a set-finishing card for cash.
- A "typical player" builds as soon as they have a set, as far as their cash allows, leaves jail straight
  away, and makes any fair swap. Against three of them, a typical player wins **21.8 games in 100**; **13 in
  100 never finish** (more on that below).
- Ground truth check: the board maths matches the published tables (Truman Collins) to two decimals.

## Running order (about 13 minutes)

| Time | Question on screen | Answer and numbers | Visual |
|---|---|---|---|
| 0:00-0:40 Cold open | "You've got two monopolies and $1,000. Hotels on one, or a house on everything?" | Don't answer yet. "We ran 46 million games to settle the arguments every family has." Tease: one choice wins 76 games in 100, the other 67 | A table mid-game, two sets highlighted; the two numbers flash |
| 0:40-2:00 The rules of the experiment | "Is this how people actually play?" | Buy what you land on, set-for-set swaps only, typical player wins 21.8 in 100. 13 in 100 four-player games never end: a colour gets split one card each between three players and no two-person swap can fix it (57 in 100 with six players). Going first: 24.6 wins vs 18.9 for last | Animated deal of a three-way split; "why your games never end" beat |
| 2:00-3:00 The board isn't fair | "Where do people actually land?" | Illinois Ave is landed on 46% more than Park Place (3.19% vs 2.19% of rolls). Jail is the busiest square. Orange sits 6, 8 and 9 squares past Jail | `board_probability_leave.png` as an animated board |
| 3:00-4:30 What to buy | "Should I skip anything? Chase a colour?" | Buy everything. Never buying dark blue costs 5.5 wins in 100, brown 4.4, orange 4.3, light blue 3.9, pink 3.3; green, railroads and utilities cost nothing either way. Favourite colours matter less than people think: orange fan +2.9, light blue +1.9, pink +1.4, Boardwalk fan +0.3, railroad fan 0, green fan -1.8, brown fan -2.8. Twist: Baltic Avenue collects more rent than any other square (10.4% of all rent) and railroads 17.9%, yet chasing either doesn't win games | Fan leaderboard; Baltic "top earner" board |
| 4:30-5:30 Trading | "Should I ever help someone finish a set?" | Yes, if you get yours. Refusing every trade drops you from 21.8 to 6.3 wins in 100. Haggling over which set is "worth more" instead of evening up at printed prices costs the best player 9.8 wins in 100: the other player walks away and neither of you gets a set | Two players at the table, the swap |
| 5:30-7:30 **Spend or save?** | "Should I keep cash back in case I land on a hotel?" | Win rate falls with every dollar kept back: $0 21.8, $100 20.0, $300 16.0, $500 12.5, $1,000 7.1. And it doesn't protect you: going bankrupt first rises from 21.8% to 23.6% ($300) and 27.0% ($1,000). Keeping enough to pay the worst rent on the board: 14.0. Why: houses earn the money that protects you; idle cash earns nothing; and in an emergency you can sell houses back and mortgage. Mortgaging loose property to pay for houses: 23.4 (+1.6). The evolved player keeps about $100 | `building_cash.png` (two panels: wins, and first to go bust) |
| 7:30-8:30 **How far to build** | "Is three houses enough?" | One house 0.1, two houses 4.9, three 17.9, four 20.4, hotels 21.8. The third house is the big jump, but stopping there throws away wins. Holding at four houses to starve others of houses: 20.4 | `building_stop.png` |
| 8:30-10:00 **Two sets: which first?** | "Light blue and green: where do the houses go?" | Mid-game test: you own two complete sets, every other property is already bought, everyone has $1,000. Light blue first 76 in 100, green first 61. Orange + green: 84 vs 65 (the biggest gap, 20 wins). Rule: the set with cheaper houses first; orange and light blue beat everything (and tie with each other); pink next; red = yellow; dark blue = green. Brown is the exception: weak rent, so it goes first only ahead of green and dark blue (and, when cash is short, ahead of red and yellow too) | `building_which_first_1000.png` |
| 10:00-11:00 **Max one set or spread?** | "Hotels on one, or a house on everything?" | One house on everything, then two, then three: 67.1 in 100 on average, worse even than building the wrong set first (68.8). Best in 0 of 56 situations. Build one set up first: 76.2. Middle path (first set to three houses, second to three, then hotels) ties on average (76.1); which is better depends on the set: light blue ($50 houses) go straight to hotels (+0.6 to +3.5); red, yellow, green, dark blue ($150-200 houses) three first, then the other set (+0.6 to +3.4); pink and orange either way | `building_max_vs_spread_1000.png` |
| 11:00-12:30 The evolved player | "What would a computer do?" | 24 random players, 25 generations, each bred from the winners. It settles on: two favourite colours (light blue and orange), mortgage to get them and bid up to double at auction; keep about $100; hotels; mortgage loose property to build; out of jail early, stay in late; make every fair swap. 31 wins in 100 vs 21.8 (about 40% more). Which habits matter: never trading -23, haggling -9.8, stopping at three -5.3, no favourites -4.5, not mortgaging to build -3.8. Put three of them at a table with one typical player and the typical player does worse than usual (16.2): it's genuinely better, not a trick | `evolution_realistic.png`, `ablation_realistic.png`, `ecology_realistic.png` (every other player type dies out) |
| 12:30-13:00 The playbook | "So what do I do on my turn?" | The six rules below. Even the best player loses 2 games in 3: these shift the odds, they don't remove the dice | End card |

## The playbook (end card, and pinned comment)

1. Buy everything you land on.
2. Pick a favourite cheap or middle set (orange, light blue, pink) and go hard for it.
3. Make every fair set-for-set trade. Don't haggle.
4. Spend your cash on houses. Keep about $100, and mortgage spare property to build.
5. Two sets? Cheaper houses first, built up, never one house on everything.
6. Build to hotels. On expensive sets, get three houses everywhere first.
(Bonus: leave jail early in the game, stay in once other players have houses.)

## Visuals

| Asset (in `sim/results/charts/`) | Act |
|---|---|
| `board_probability_leave.png` | The board isn't fair |
| `building_cash.png` | Spend or save (the key chart: both panels together) |
| `building_stop.png` | How far to build |
| `building_which_first_1000.png` (also `_500`) | Two sets: which first (read a row as "build this first", bold = better choice) |
| `building_max_vs_spread_1000.png` (also `_500`) | Max one or spread |
| `evolution_realistic.png`, `ablation_realistic.png`, `ecology_realistic.png` | The evolved player |
| `sample_games/*.jsonl` | Turn-by-turn logs for animating a real game in the cold open |

To build: a fan leaderboard (data in `summary.md`, "Realistic play"), the three-way-split deadlock animation,
a Baltic-vs-Boardwalk rent board, and the playbook end card.

## Titles and thumbnails

Lead with the question people already ask, not the game count.

| | Thumbnail | Title |
|---|---|---|
| A | Hand holding a fan of cash over a hotel, big "SPEND IT" | Should You Save Cash or Build Hotels in Monopoly? I Simulated 46 Million Games |
| B | Light blue and green sets side by side, an arrow to light blue, "76 vs 61" | You Have Two Monopolies. Which One Gets the Houses? (Simulated) |
| C | One house on every property crossed out | Hotels on One Set or a House on Everything? Monopoly, Settled With Math |

Shorts (one number each):
- "Should you keep cash in Monopoly?" Keeping $300 back makes you *more* likely to go bust first (23.6% vs 21.8%).
- "Two monopolies: which first?" Orange + green: orange first wins 84 in 100, green first 65.
- "One house on everything?" Best in 0 of 56 situations.
- "Is three houses enough?" Three: 17.9 wins in 100. Hotels: 21.8.
- "The top-earning square isn't Boardwalk." Baltic collects 10.4% of all rent.
- "Why your Monopoly games never end." 13 in 100 four-player games; 57 in 100 with six players.
- "Never trade?" 6.3 wins in 100 instead of 21.8.
- "Who should go first?" First player 24.6 wins, last 18.9.

## What to say carefully on camera

- **"Typical player" is a model.** It builds as fast as its cash allows and leaves jail at once. Real players
  vary; the results compare habits against that one kind of opponent.
- **The two-set scenario starts mid-game** with every property already bought and $500 or $1,000 each, so
  the owner of two sets wins most games anyway (roughly 55-89 in 100); the point is the gap between choices.
- **Trading is simplified**: two-player set-for-set swaps evened up at printed prices; no three-way deals, no
  cash for set cards, no bluffing. With that rule 13 in 100 four-player games deadlock. How real tables break
  those deadlocks (a cash sale, a three-way trade, calling the game) could shift some numbers.
- **Small differences are real but feel invisible**: 1-2 wins in 100 is about one extra win in 50-100 games.
  The building results (spend vs save, which set first, spread) are 5-20 wins in 100 and safe to lead with.
- **House rules**: the building results hold with no auctions and Free Parking money (keep $0: 21.7, $300:
  17.5, $500: 14.9; spread 21.2).
- **Published sources**: Perfect Play's figures and champion quotes in the research report are partly from
  secondhand summaries; quote them only after checking the originals (list in the report).

## Draft YouTube description (title A)

```
Should you spend all your money on houses in Monopoly, or keep some back in case you land on a hotel? If you
own two monopolies, which one gets the houses? And is a house on everything better than hotels on one set?

We simulated 46 million games of Monopoly played the way families play it: everyone buys what they
land on, and you only help someone finish a set if they help you finish yours.

🏠 Keep cash back? Every dollar costs you. Keeping $300 back drops you from 21.8 to 16.0 wins in 100, and you
   go bankrupt first MORE often, not less (23.6% vs 21.8%). Keep about $100.
🏠 Two sets? Build the cheaper houses first. Orange + green: orange first wins 84 games in 100, green first 65.
🏠 A house on everything? Never the best choice in any of 56 situations we tested.
🏠 Three houses or hotels? Three: 17.9 wins in 100. Hotels: 21.8.

How it's made: a simulator of the standard US board and current cards under the official rules (auctions, even
building, 32 houses and 12 hotels, mortgages); its board probabilities match Truman Collins' published tables.
Method, code and every result: https://github.com/drburke-droid/nhl-sog-predictor/tree/master/monopoly
```
