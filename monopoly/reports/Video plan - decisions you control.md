# Video plan for review: the Monopoly decisions you actually control

Status: draft for review, nothing filmed or rendered. Replaces `Video plan.md` (kept for reference).

## The niche

There are plenty of videos about which squares get landed on and which colour is best. At a real table you
can't choose where you land. What you *can* choose:

1. **Buy it or pass** when you land on an unowned property (and how much to bid if it goes to auction).
2. **When to trade**: in a real game, mainly a swap that finishes a set for both players.
3. **Whether to go for all four railroads.**
4. **When and where to build**: spend or save, how far, which set first, hotels on one set or a house on
   everything, and what to do when you owe money.
5. **Jail**: pay and leave, or stay and roll.

This video takes those five decisions one at a time, tests every common answer in simulated games, and
says how many games each choice wins or loses you. It finishes with a "playbook player" who makes every
best choice and never picks a colour.

## How the simulation plays (one slide, about 40 seconds)

- Standard US board, current cards, official rules; the board maths matches Truman Collins' published
  tables, which is the credibility line.
- Everyone buys what they land on if they can afford it.
- The only trades are swaps where both players finish a set ("I give you the card that completes yours if
  you give me the card that completes mine"), with cash evening up the printed prices. Nobody sells a
  set-finishing card for cash.
- To test a choice, one player makes that choice and the other three are "typical players". A typical player
  buys what it lands on, makes any fair swap, builds as far as its cash goes and pays to leave jail.
  **A typical player wins 21.8 games in 100.** About 13 in 100 four-player games never finish: a colour gets
  split three ways and no two-person swap can fix it.
- Each result below is 100,000 games. Every key result was also re-run with no auctions and Free Parking
  money (common house rules); the figures are noted where they differ.

## The five decisions: what we found

Wins per 100 games against three typical players (typical = 21.8).

### 1. Buy it or pass? (chart `decision_buy.png`, `decision_auction.png`)

| Choice | Wins |
|---|---|
| Always buy | 21.8 |
| Pass if buying would leave you under $100 | 21.5 |
| ...under $200 | 20.9 |
| ...under $300 | 20.1 |
| ...under $500 | 17.6 |
| Mortgage something to buy a card that completes or blocks a set | 22.3 |

- **Answer: always buy.** Passing to protect your cash costs games, and the more cautious you are the more
  it costs. Mortgaging to grab a set-finishing (or set-blocking) card helps a little (+0.5).
- **If someone else passes and it goes to auction, bid up to 1.5x the printed price**: 23.7 vs 21.8 for
  stopping at the printed price; lowballing at half price gives 20.5. (Only applies if you play auctions.)
- Why: every property is a future trading chip. You need cards to make swaps.

### 2. When to trade for a set (chart `decision_trade.png`)

| Choice | Wins |
|---|---|
| Make every fair set-for-set swap | 21.8 |
| Only swap if your set is at least as good as theirs | 22.0 |
| Only swap when you have $400 to build with straight away | 22.0 |
| Haggle: demand extra cash when their set is "worth more" | 16.4 |
| Never trade | 6.3 |

- **Answer: make the swap as soon as it's on the table, at printed prices.** Waiting until you can afford
  houses, or holding out for the better set, makes no real difference. Haggling costs 5.4 wins because the
  other player walks away and neither of you gets a set. Refusing to trade is the worst decision in the game
  (-15.6).
- For a strong player, being picky actually hurts: the best player we found drops from 31.3 to 26.4 if it
  refuses swaps that give the other player the better set.
- House rules: the same (picky 21.2, wait for $400 22.2).

### 3. Should you go for all four railroads? (chart `decision_railroads.png`)

| Choice | Wins | Owns all four railroads |
|---|---|---|
| Never buy railroads | 21.9 | 2% of games |
| Buy them when you land on them | 21.8 | 4% |
| Chase them: mortgage to buy, bid double at auction | 21.8 | 5% |
| Also trade for the last one (swap a colour card for the railroad that completes your four) | 23.6 | 25% |
| Chase them and trade for them | 24.2 | 26% |

- **Answer: only if you're willing to trade for them.** Buying railroads on your own almost never gets you
  all four, and it changes nothing. Trading for the fourth one (a fair swap: they finish a colour set, you
  finish your railroads) gets you all four in a quarter of games and adds 2.3 wins in 100. When the collector
  holds all four it wins 54 of those games. House rules: 23.0.
- Side note: skipping railroads entirely doesn't change your wins, but you're more likely to be the first to
  go broke (23.3% vs 21.8%). They are steady income, not a winning plan.

### 4. When and where to build (charts `building_cash.png`, `building_stop.png`, `building_which_first_1000.png`, `building_max_vs_spread_1000.png`)

**Spend it or keep some back?**

| Cash you keep in hand when building | Wins | First to go broke |
|---|---|---|
| $0 | 21.8 | 21.8% |
| $100 | 20.0 | 22.2% |
| $300 | 16.0 | 23.6% |
| $500 | 12.5 | 24.9% |
| $1,000 | 7.1 | 27.0% |
| Enough to pay the biggest rent on the board | 14.0 | 23.2% |

- **Answer: spend it.** Keeping cash back doesn't make you safer: the more you keep, the *more* likely you
  are to be the first to go broke. Houses bring the cash in; cash in hand earns nothing, and in a pinch you can
  mortgage or sell houses back. Mortgaging spare properties to pay for houses adds 1.6 (23.4).
- **How far to build:** one house 0.1, two 4.9, three 17.9, four 20.4, hotels 21.8. The third house is the big
  jump, but stopping there costs 3.9.
- **You own two sets: which first?** (mid-game test: every property already bought, $1,000 each, all 28 pairs
  of sets). Build the set with the cheaper houses first. Orange and light blue beat every other set (and tie
  with each other); pink next; red ties with yellow, and dark blue with green. Brown only goes first ahead of
  green and dark blue. Getting it wrong costs up to 20 wins in 100 (orange + green: orange first 84, green
  first 65).
- **Hotels on one set, or a house on everything?** Never spread one house across everything: it was the
  best choice in 0 of 56 situations, and on average (67.1) it did worse than building the *wrong* set first
  (68.8). Building one set up first: 76.2. Cheap sets (light blue, $50 houses): straight to hotels. Expensive
  sets ($150-200 houses): three houses on each set first, then hotels. Pink and orange: either way works.
- **You owe money you don't have: mortgage or sell houses?** Mortgage your spare properties first; sell
  houses last. Selling houses first drops you to 17.3 (-4.5) and makes you more likely to go broke first
  (24.3%). House rules: 18.3.

### 5. Jail: pay to leave, or stay and roll? (chart `decision_jail.png`)

| Choice | Wins | First to go broke |
|---|---|---|
| Always pay and leave | 21.8 | 21.8% |
| Try one roll for doubles, then pay | 21.6 | 21.2% |
| Always stay as long as you can | 21.7 | 20.4% |
| Stay if an opponent has houses on the stretch just past jail (St. Charles to Illinois) | 22.5 | 20.7% |
| Stay once four or fewer properties are unsold | 22.7 | 19.9% |
| Stay once anyone has houses | 22.9 | 19.6% |

- **Answer: get out early while there is property to buy; once anyone has houses, stay in.** It is the
  smallest of the five decisions (+1.1), but it also makes you less likely to go broke first (19.6% vs 21.8%).
  Always staying, or "try a roll first", gains nothing. House rules: same (22.5 for the early/late rule).

## The payoff: the playbook player (chart `playbook_by_table_size.png`, `decision_values.png`)

One player who makes every best choice above and never picks a favourite colour: always buys (mortgages for
a set card), bids up to 1.5x at auction, makes every fair swap including for railroads, spends its cash on
houses (mortgaging spare property), builds the cheaper set first to hotels, mortgages before selling houses,
and stays in jail once anyone has houses.

| Table | Typical player | Playbook player |
|---|---|---|
| 2 players | 50.4 | 58.6 |
| 3 players | 33.3 | 41.1 |
| 4 players | 21.8 | **28.8** (29.1 if it keeps about $100) |
| 5 players | 12.7 | 16.3 |
| 6 players | 7.4 | 9.1 |
| 4 players, house rules (no auctions, Free Parking money) | 21.7 | 25.0 |

Same dice, a third more wins. Sit a typical player down with three playbook players and the typical player
wins 19.2 instead of 21.8, so it isn't a trick that only works on weak players.

What each decision is worth (`decision_values.png`), better choice vs the common alternative:
trade at all +15.6; spend vs keep $300 +5.9; don't haggle +5.4; mortgage before selling houses +4.5; hotels
vs three houses +3.9; trade for railroads +2.3; bid to 1.5x at auction +1.9; always buy vs pass when under
$300 +1.7; mortgage to build +1.6; jail timing +1.1; mortgage to buy a set card +0.5. When you hold two sets:
which one first is worth up to 20, and building one up instead of spreading about 9.

## Running order (about 13 minutes)

| Time | Section | What's said and shown |
|---|---|---|
| 0:00-0:35 | Cold open | "Every Monopoly video tells you orange is the best colour. Great. You don't get to choose where you land. Here's what you do get to choose, and what each choice is worth." Flash the playbook number: 29 wins in 100 vs 22 |
| 0:35-1:15 | How we tested it | The rules slide above. A typical player wins 22 in 100; 13 in 100 games never end (the three-way split, briefly) |
| 1:15-2:45 | 1. Buy or pass | Buy-or-pass ladder (21.8 down to 17.6). Auction beat: bid to 1.5x. "Every card is a trading chip" |
| 2:45-4:30 | 2. Trading | The swap animation. Never trade 6.3. Haggling 16.4. Waiting or being picky doesn't help. "Make the deal" |
| 4:30-6:00 | 3. Railroads | "The railroad guy" setup. Buying them: nothing. Trading for the fourth: all four in 1 game in 4, +2.3. Never buying them: same wins, more likely to go broke first |
| 6:00-10:00 | 4. Building (the longest section) | Spend vs save (both panels: wins fall, risk rises). How far to build. Two sets: which first (the matrix, orange + green 84 vs 65). Max one set vs one house everywhere (0 of 56). Owe money: mortgage first, sell houses last |
| 10:00-11:15 | 5. Jail | Out early, in late; the "anyone has houses" trigger; risk of going broke first falls |
| 11:15-12:30 | The playbook player | Same dice, a third more wins; table sizes; the "what each decision is worth" ranking |
| 12:30-13:00 | End card | The seven-line playbook (below) |

## End card: the playbook

1. Land on it? Buy it. (Short on cash for a card that finishes a set? Mortgage something.)
2. At auction, bid up to one and a half times the price.
3. Make every fair set-for-set swap, at printed prices. Don't haggle.
4. Want the railroads? Trade for the last one, or don't bother.
5. Spend your cash on houses; mortgage spare property to build; go to hotels.
6. Two sets: cheaper houses first. Never one house on everything.
7. Out of jail early; stay in once anyone has houses.

## Titles and thumbnails

| | Thumbnail | Title |
|---|---|---|
| A | A player's hand hovering over "BUY" / "PASS", text "WHAT YOU CONTROL" | You Can't Control the Dice in Monopoly. Here's What You Can (Simulated) |
| B | Stack of cash vs a hotel, "SPEND IT" | Save Cash or Build Hotels? Every Monopoly Decision, Tested |
| C | Four railroad cards with a swap arrow, "ONLY IF YOU TRADE" | Should You Go for All Four Railroads? Monopoly Decisions, Settled |

## Shorts (one decision each)

- "Should you pass on a property to save cash?" Always buy: passing when under $300 costs 1.7 wins in 100.
- "Never trade in Monopoly?" 6.3 wins in 100 instead of 21.8.
- "Don't haggle." Even up at printed prices: haggling costs 5.4.
- "The railroad guy." Buying them does nothing; trading for the fourth: +2.3, all four in 1 game in 4.
- "Keep cash for emergencies?" Keeping $300 back makes you *more* likely to go broke first.
- "Owe rent? Don't sell your houses." Mortgage first: selling first costs 4.5.
- "Two monopolies, which first?" Orange + green: 84 vs 65.
- "A house on everything?" Best in 0 of 56 situations.
- "Stay in jail?" Only once someone has houses.

## Caveats to say on camera

- The "typical player" is a model of an ordinary player, not a real person; every result is "against players
  who play like this".
- Trading is simplified to fair two-player swaps. No three-way deals, no cash for set cards, no bluffing.
- Small effects (jail +1.1, mortgage to buy a set card +0.5) are real over 100,000 games but you'd need dozens
  of games to feel them. The big ones (trading, spending, which set first, max vs spread, mortgage vs sell)
  are 4-20 wins in 100.
- Auction advice only applies if you play auctions; everything else holds with house rules.

## Questions for your review

1. **Endless games.** Under the swap-only rule, 13 in 100 four-player games never finish. Do your games
   break those deadlocks somehow (a cash sale, a three-way trade, calling it), or do we say on camera that
   real games stall too?
2. **Drop the evolved computer player?** It relies on favourite colours, which this video deliberately skips.
   The playbook player now does its job. Suggest cutting it, or keeping it to a 20-second aside.
3. **Building section length.** It has five sub-questions. Keep all five in the main video, or move "which set
   first" and "max vs spread" to their own follow-up video and keep this one to spend vs save, how far, and
   mortgage vs sell?
4. **Auctions.** Many families never auction. Keep the auction beat (official rules) or drop it?
5. **Length.** About 13 minutes as planned; the earlier videos ran 3-11 minutes.

Data: `sim/results/summary.md` (sections "The decisions a player controls", "The playbook player", "Building",
"Two complete sets"); charts in `sim/results/charts/` (`decision_*.png`, `building_*.png`, `playbook_by_table_size.png`).
