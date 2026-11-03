# Video plan for review (revision 4): the Monopoly decisions you actually control

Status: draft for review, nothing filmed or rendered. Revision 3 adds the look (`Art direction.md`, style frames
in `art/frames/`), a 20-second lore cold open, and closes the open questions: endgame prices stay at 2.5-3x,
and the headline number is "33 vs 25" (with auctions). Revision 4 reframes that payoff so it lands: "three wins
for every two of theirs" per game, and "family champion two years in three" over a year of game nights (see
"Framing the payoff").

Changes since revision 1, following your review:
- **Endless games**: stalled games now go into an endgame where players start making deals that hand someone a
  full set (details below). Every simulated game now finishes, and every number in this plan was re-run with
  that endgame in place.
- **Computer-evolved player**: cut to one short, plain-English aside.
- **One video**: all five decisions stay in, building included.
- **Auctions**: kept as the main rules; every decision also lists what changes without auctions.
- **Length**: about 13 minutes.

## The niche

There are plenty of videos about which squares get landed on and which colour is best. At a real table you
can't choose where you land. What you *can* choose:

1. **Buy it or pass** when you land on an unowned property (and how much to bid if it goes to auction).
2. **When to trade**: in a real game, mostly a swap that finishes a set for both players, and later the
   endgame deals that break a stalled board.
3. **Whether to go for all four railroads.**
4. **When and where to build**: spend or save, how far, which set first, hotels on one set or a house on
   everything, and what to do when you owe money.
5. **Jail**: pay and leave, or stay and roll.

The video takes those five one at a time, tests each common answer over 100,000 simulated games, and says how
many games in 100 each choice wins or loses you. It ends with a "playbook player" who makes every best choice
and never picks a colour.

## How the simulation plays (one slide, about 40 seconds)

- Standard US board, current cards, official rules. The board maths matches Truman Collins' published tables,
  which is the credibility line.
- Everyone buys what they land on if they can afford it.
- Trades are swaps where both players finish a set: "I'll give you the card that completes yours if you give
  me the card that completes mine." Cash evens up the printed prices.
- **The endgame.** When nearly every property is owned and 10 rounds go by with no new set, trade or
  bankruptcy, the board has locked up. Players then start making deals that give someone a full set:
  - selling the card that finishes another player's set for cash (asking 2.5 times the printed price; buyers
    will pay up to 3 times);
  - buying up stray cards of a colour that's split between several players, so the colour can come together.

  A typical four-player game sees about 2 to 3 such deals. With this endgame every game finishes (it averages
  about 50 rounds); without it, 13 in 100 four-player games and 57 in 100 six-player games never ended.
- To test a choice, one player makes it and the other three are "typical players". A typical player buys what
  it lands on, makes any fair swap, builds as far as its cash goes and pays to leave jail.
  **A typical player wins 25.1 games in 100**, which is a fair share at a four-player table.
- Every result is 100,000 games, run twice: with auctions (official rules) and without them (an unbought
  property just stays unsold).

## The five decisions

Wins per 100 games against three typical players. A typical player wins 25.1 with auctions and 25.1 without.

### 1. Buy it or pass? (charts `final_buy.png`, `final_auction.png`)

| Choice | With auctions | Without auctions |
|---|---|---|
| Always buy | 25.1 | 25.1 |
| Pass if buying would leave you under $100 | 25.3 | 25.5 |
| ...under $200 | 25.2 | 25.6 |
| ...under $300 | 24.6 | 25.1 |
| ...under $500 | 22.7 | 23.2 |
| Pass if under $300, but always buy a card that finishes or blocks a set | 25.1 | 25.7 |
| Mortgage something to buy a card that finishes or blocks a set | 25.4 | 25.1 |

- **Answer: buy it, unless it would leave you almost broke.** Keeping a small cushion when *buying* (around
  $100-200) is harmless, and it makes you a little less likely to be first to go broke (24.0% vs 25.1% at
  $200). Passing to keep $500 back costs 2.4 games in 100. Never pass on a card that finishes or blocks a set.
- **Without auctions, passing is cheaper.** A property you pass on just sits unsold instead of going to an
  opponent at auction, so "keep about $300, but always buy set cards" is slightly ahead (25.7).
- **At auction, bid up to 1.5 times the printed price** (26.5 vs 25.1 at the printed price). Lowballing at
  half price gives 24.1. Bidding high only to block someone's set gives 25.8. Without auctions this choice
  never comes up.
- Why it works: every property is a trading chip. You need cards to make swaps.

### 2. When to trade (chart `final_trade.png`)

| Choice | With auctions | Without auctions |
|---|---|---|
| Make every fair set-for-set swap | 25.1 | 25.1 |
| Only swap if your set is at least as good as theirs | 25.6 | 25.4 |
| Only swap when you have $400 to build with straight away | 25.1 | 25.1 |
| Haggle: demand extra cash when their set is "worth more" | 21.4 | 20.8 |
| Endgame: never sell the card that finishes someone's set | 24.2 | 24.5 |
| Endgame: willing to pay up to 5x for the card you need | 25.1 | 25.1 |
| Never trade | 4.9 | 4.9 |

- **Answer: make the swap as soon as it's on the table, at printed prices.** Waiting until you can afford
  houses changes nothing. Turning down swaps that hand the other player the better set helps only a little
  (+0.5). Haggling costs about 4 games in 100: the other player walks away and neither of you gets a set.
  Refusing to trade at all is the worst decision in the game: **4.9 wins in 100, a fifth of a fair share.**
- **When the game stalls, take the deal.** Refusing to sell the card that finishes someone's set costs about
  a game in 100, and you become much more likely to go broke first (28.7% vs 25.1%). The cash lets you build
  on what you have. Paying above the going price doesn't help.
- Same with or without auctions.

### 3. Should you go for all four railroads? (chart `final_railroads.png`)

| Choice | With auctions | Without auctions | How often it owns all four |
|---|---|---|---|
| Never buy railroads | 24.8 | 24.9 | 2% of games |
| Buy them when you land on them | 25.1 | 25.1 | 4% |
| Chase them: mortgage to buy, bid double at auction | 25.1 | 25.0 | 4% |
| Also trade for the last one (a colour card for the railroad that completes your four) | 26.6 | 26.2 | 27% |
| Chase them and trade for them | 27.0 | 26.4 | 28% |

- **Answer: only if you're willing to trade for them.** Buying railroads on their own almost never gets you
  all four, and it changes nothing. Trading for the fourth (a fair swap: they finish a colour set, you finish
  your railroads) gets you all four in more than 1 game in 4 and adds about 2 games in 100. A player holding
  all four wins about 53 of those games.
- Skipping railroads entirely barely changes your wins, but you're more likely to be first to go broke (27.0%
  vs 25.1%). They're steady income, not a winning plan by themselves.

### 4. When and where to build (charts `final_building_cash.png`, `final_stop.png`, `final_which_first.png`, `final_max_vs_spread.png`)

**Spend it or keep some back when building?**

| Cash you keep in hand when building | Wins (with auctions) | First to go broke | Wins (without auctions) |
|---|---|---|---|
| $0 | 25.1 | 25.1% | 25.1 |
| $50 | 23.6 | 25.5% | 23.6 |
| $100 | 22.4 | 25.8% | 22.3 |
| $200 | 19.3 | 26.7% | 19.4 |
| $300 | 16.3 | 27.8% | 16.4 |
| $500 | 10.9 | 29.9% | 11.1 |
| $1,000 | 3.2 | 33.5% | 3.4 |
| Enough to pay the biggest rent on the board | 13.2 | 27.6% | 13.7 |

- **Answer: spend it on houses.** This is the biggest surprise in the video. Keeping cash back doesn't make
  you safer: the more you keep, the *more* likely you are to be first to go broke. Houses bring cash in, cash
  in hand earns nothing, and in a pinch you can mortgage or sell houses back.
- **Mortgage your spare properties to pay for houses**: 28.0 (27.8 without auctions), worth about 3 games in
  100.
- **(Contrast for the narration:** a small cushion is fine when *buying* a property, but not when *building*.
  The difference is that houses pay you back and a property on its own barely does.)
- **How far to build:** one house 0.1, two 4.4, three 19.6, four 23.2, hotels 25.1. The third house is the big
  jump, but stopping there costs 5.5. Same without auctions.
- **You own two sets: which one first?** Mid-game test: every property already bought, $1,000 each, all 28
  pairs of sets. Build the set with the cheaper houses first:
  - light blue and orange first (they tie with each other), then pink;
  - red ties with yellow, and dark blue with green;
  - brown goes first only ahead of dark blue or green, and only just.

  Picking wrong costs up to 21 games in 100: orange + green gives 84 with orange first, 63 with green first.
- **Hotels on one set, or a house on everything?** Never spread one house across everything. It was the
  best choice in 0 of 28 pairs, and on average (65.8) it did worse than building the *wrong* set first (67.6).
  Building one set up first: 75.7.
  - Cheap set first (light blue): go straight to hotels (+0.6 to +3.5 over the alternative).
  - Expensive set first (red, yellow, dark blue): three houses on both sets, then hotels (+1.0 to +3.9).
  - Pink or orange first: either way works.
- **You owe money you don't have: mortgage or sell houses?** Mortgage spare properties first and sell houses
  last. Selling houses first drops you to 19.1 (19.4 without auctions), a loss of 6 games in 100, and makes you
  more likely to go broke first (28.5%).

### 5. Jail: pay to leave, or stay and roll? (chart `final_jail.png`)

| Choice | With auctions | Without auctions | First to go broke |
|---|---|---|---|
| Always pay and leave | 25.1 | 25.1 | 25.1% |
| Try one roll for doubles, then pay | 25.0 | 25.1 | 24.4% |
| Always stay as long as you can | 25.1 | 24.9 | 23.8% |
| Stay if an opponent has houses on the stretch just past jail (St. Charles to Illinois) | 26.1 | 26.0 | 23.6% |
| Stay once four or fewer properties are unsold | 26.5 | 26.4 | 23.1% |
| Stay once anyone has houses | 26.5 | 26.5 | 23.0% |

- **Answer: get out early while there's property to buy; once anyone has houses, stay in.** It's worth about
  1.4 games in 100, and you're less likely to be first to go broke (23.0% vs 25.1%). Always staying, or trying
  one roll first, gains nothing. Same without auctions.

## Short aside: what if the computer designs its own player? (about 20 seconds)

Plain-English version for the narration: "We also let the computer invent a player by trial and error. It
started with 24 random players, kept the ones that won most, mixed and tweaked their habits, and repeated that
25 times. It ended up with the same habits as our playbook, plus one extra: it picked two favourite colours,
light blue and orange, and went hard for them. It wins about 36 games in 100 (33 without auctions)." Put this
just before the playbook reveal and don't dwell on how it works.

## The payoff: the playbook player (charts `final_playbook.png`, `final_values.png`)

This player makes every best choice above and never picks a favourite colour:
- always buys, and mortgages if needed to buy a card that finishes or blocks a set;
- bids up to 1.5 times the price at auction;
- makes every fair swap, railroads included;
- spends its cash on houses, mortgaging spare property to do it, and builds the cheaper set first to hotels;
- mortgages before selling houses;
- stays in jail once anyone has houses.

| Table | Typical player | Playbook player (with auctions) | Playbook player (without auctions) |
|---|---|---|---|
| 2 players | 50.5 | 58.5 | 52.6 |
| 3 players | 33.4 | 40.8 | 38.8 |
| 4 players | 25.2 | **32.6** | **30.2** |
| 5 players | 20.1 | 25.8 | 23.8 |
| 6 players | 16.8 | 20.5 | 19.3 |

- Same dice: at the four-player table the playbook player wins 32.6 games in 100 and each of its three
  opponents about 22.4. That's **about three wins for every two of theirs** (1.45 times as many). Without
  auctions: 30.2 against 23.3 each (1.3 times).
- Seat a typical player with three playbook players and it wins 20.7 instead of 25: the playbook isn't a trick
  that only works against weak players.
- Without auctions the playbook's edge shrinks, most of all heads-up (52.6 vs 50.5), because the auction
  bidding is one of its advantages.

What each decision is worth (`final_values.png`, with auctions, the better choice against the common
alternative):

| Decision | Gain (games in 100) |
|---|---|
| Making fair swaps instead of never trading | +20.2 |
| Spending your cash instead of keeping $300 back when building | +8.9 |
| Mortgaging before selling houses | +6.0 |
| Hotels instead of stopping at three houses | +5.5 |
| Not haggling | +3.8 |
| Mortgaging spare property to build | +2.9 |
| Trading for all four railroads | +1.9 |
| Jail timing | +1.4 |
| Bidding to 1.5 times at auction | +1.4 |
| Always buying instead of passing when under $300 | +0.5 |
| Mortgaging to buy a set card | +0.3 |

When you hold two sets, which one you build first is worth up to 21 games in 100, and building one up instead
of spreading is worth about 10.

## Framing the payoff

"33 wins in 100 instead of 25" is accurate, but it sounds small, because both numbers lose most games. What
people care about at game night is beating the people across the table, and bragging rights over a season.
All three framings below come from the same results and are exact.

**1. The price tag, in Monopoly's own currency:**
> "**Playing the typical way is like handing the bank $250 before you've rolled.**"

Set it up and pay it off: "We took the playbook player's money away until it was no better than a typical
player. It could give up $250 of its $1,500. Without auctions it's about $200, the same as passing Go."

(Measured directly in that direction: the playbook player starting with $250 less wins 25.10 games in 100,
against 25.13 for a typical player with the usual $1,500; $300 less gives 23.4. Match at about $249. Without
auctions: $200 less gives 24.99 against 25.07, a match at about $197. 200,000 games per step. Style frame 08.)

Why not $300: the other direction gives a different number. A typical player needs an extra $300 to match the
playbook player (+$275 gives 32.0, +$300 gives 32.7, playbook 32.6). Money has diminishing returns, so losing
$250 hurts as much as gaining $300 helps. If the narration wants both: "...and it would take $300 to buy that
edge back."

**2. Per game, against the people you're playing (the headline):**
> "Same dice. Same board. Make seven different decisions and you'll win **three games for every two** anyone
> else at the table wins."

(32.6 vs 22.3 / 22.5 / 22.5 wins per 100: 1.45 to 1.)

**3. Over a season of game nights (the emotional payoff; small edges compound):**
> "Play every week for a year, and you finish as the family champion **two years out of three**. Each of the
> people you're playing? About **one year in eleven**."

(Most wins outright over 52 games: 64.8% for the playbook player, 8.5-8.9% for each opponent, 8.8% a shared top
spot. Over 10 game nights: 35% vs about 13% each. A typical player at an all-typical table: 18% over 10 nights,
23% over a year. Chart: `final_champion.png`.)

**Plain-language backups, used once each:**
- "One game in three instead of one in four."
- "The playbook is worth more than twice the edge of rolling first." (Going first: +3.3 wins per 100;
  playbook: +7.6.)

**How to show it:**
- **The price tag** (style frame 08): three engraved banknotes of our own design ($100, $100, $50) pushed
  across the desk to the bank and stamped PAID TO THE BANK; "Playing the typical way is like handing the bank
  $250 before you've rolled"; the two ledger lines (playbook player with $250 handed over: 25.1; typical
  player with the usual $1,500: 25.1).
- **The game-night ledger** (style frame 07): tally marks after 100 game nights. YOU 33 in red; MOM 22, DAD 22,
  JORDAN 23 in ink. Stamp: FAMILY CHAMPION.
- **The trophy shelf:** run 100 simulated "years of Sundays" and put a small trophy on the shelf of whoever
  finished each year on top. About 65 trophies line up under YOU, about 9 under each of the others, and a few
  years are shared. One image that makes "two years in three" physical.
- **The race:** for one year, the four tally rows fill week by week. The rows run neck and neck for the first
  month, then YOU pulls away, which is exactly what "small edges compound" looks like.

Keep "32.6 vs 22.4 per 100 games" on screen as the footnote, so the claim can be checked.

## Running order (about 13 minutes)

| Time | Section | What's said and shown |
|---|---|---|
| 0:00-0:20 | Lore cold open | The 1904 patent drawing on a desk. "In 1904 a woman named Lizzie Magie patented a board game to show how landlords get rich. You know it as Monopoly." Push in on her corner, "NO TRESPASSING - GO TO JAIL" |
| 0:20-0:45 | The promise | Title card (frame 01). "Every Monopoly video tells you which colour is best. You don't get to choose where you land. But make seven different decisions, with the same dice, and you'll win three games for every two anyone else at the table wins. Play every week for a year, and you'll be the family champion two years out of three." Stamp: 3 WINS FOR EVERY 2 |
| 0:45-1:20 | How we tested it | The rules slide. Typical player wins 25 in 100. The endgame beat: "when the board locks up, people start dealing", 2-3 deals a game, and every game finishes |
| 1:20-2:50 | 1. Buy or pass | Buy-or-pass ladder: a small cushion is fine, passing to keep $500 costs 2.4. Never pass on a set card. Auction beat: bid to 1.5 times. Without auctions: passing is cheaper |
| 2:50-4:40 | 2. Trading | Swap animation. Never trade: 4.9. Haggling costs about 4. Waiting or being picky barely matters. Endgame: take the deal, because refusing makes you likelier to go broke first |
| 4:40-6:00 | 3. Railroads | "The railroad guy." Buying them does nothing; trading for the fourth gets all four in more than 1 game in 4, +2 |
| 6:00-10:00 | 4. Building (longest) | Spend vs save (both panels: wins fall, risk rises). Mortgage to build. How far to build. Two sets: which first (the matrix; orange + green 84 vs 63). Max one set vs one house everywhere (0 of 28). Owe money: mortgage first, sell houses last |
| 10:00-11:10 | 5. Jail | Out early, in once anyone has houses; less likely to go broke first |
| 11:10-11:30 | Aside | The computer-designed player, in plain English (above) |
| 11:30-12:30 | The playbook player | The price tag first (frame 08): "Playing the typical way is like handing the bank $250 before you've rolled", the notes pushed to the bank. Then the game-night ledger (frame 07): 33 tallies vs 22, 22, 23, "three for every two". The year-long race, then the trophy shelf: two years in three vs one in eleven. The "what each decision is worth" ranking. Without auctions: $200, a free trip past Go |
| 12:30-13:00 | End card | The seven-line playbook (below) |

## End card: the playbook

1. Land on it? Buy it, unless it would leave you nearly broke. Always buy a card that finishes or blocks a set.
2. At auction, bid up to one and a half times the price.
3. Make every fair set-for-set swap at printed prices. Don't haggle. When the game stalls, take the deal.
4. Want the railroads? Trade for the last one, or don't bother.
5. Spend your cash on houses, mortgage spare property to build, and go to hotels. Owe money? Mortgage before
   you sell houses.
6. Two sets: cheaper houses first. Never one house on everything.
7. Get out of jail early; stay in once anyone has houses.

## Titles and thumbnails

| | Thumbnail | Title |
|---|---|---|
| A | Magie's 1904 patent drawing with a red stamp across it: "WHAT YOU CONTROL" | You Can't Control the Dice in Monopoly. Here's What You Can (Simulated) |
| B | Frame 06: our board's corner with the brass pawn, "SAVE IT" crossed out, "SPEND IT" stamped, "3 wins for every 2" | Save Cash or Build Hotels? Every Monopoly Decision, Tested |
| D | Frame 07 cropped: tally marks YOU vs MOM, stamped FAMILY CHAMPION | How to Be Your Family's Monopoly Champion (Same Dice, Better Decisions) |
| F | Frame 08 cropped: the banknotes stamped PAID TO THE BANK and a big red "$250" | Playing Monopoly the Typical Way Costs You $250 Before You Roll |
| E | Same as D | Win 3 Monopoly Games for Every 2 Your Family Wins |
| C | The four railway tickets fanned out, stamped "ONLY IF YOU TRADE" | Should You Go for All Four Railroads? Monopoly Decisions, Settled |

All three use the art direction (no Hasbro artwork); the description and end card carry "Not affiliated with or
endorsed by Hasbro".

## Shorts (one decision each)

- "How to be the family Monopoly champion." Same dice; seven decisions; champion two years in three.
- "Playing the typical way is like handing the bank $250 before you've rolled." ($200 without auctions: the same as passing Go.)

- "Keep cash for emergencies?" Keeping $300 back when building: 16.3 wins in 100 instead of 25.1, and you're
  *more* likely to go broke first.
- "Never trade in Monopoly?" 4.9 wins in 100, a fifth of a fair share.
- "Owe rent? Don't sell your houses." Mortgage first: selling first costs 6 games in 100.
- "Two monopolies, which first?" Orange + green: 84 vs 63.
- "A house on everything?" Best in 0 of 28 pairs.
- "Is three houses enough?" Three: 19.6. Hotels: 25.1.
- "The railroad guy." Buying them does nothing; trading for the fourth does.
- "The game's stalled. Sell the card?" Yes. Refusing makes you likelier to go broke first.
- "Stay in jail?" Only once someone has houses.

## Caveats to say on camera

- The "typical player" is a model of an ordinary player, not a real person. Every result means "against
  players who play like this".
- Trading is simplified: fair two-player swaps, plus endgame cash deals for set cards once the board locks up.
  There are no bluffs and no three-way swaps. The endgame prices (2.5 to 3 times the printed price) are our
  assumption.
- Small effects (passing with a small cushion, mortgaging to buy a set card, jail timing) are real over
  100,000 games, but you'd need dozens of games to feel them. The big ones (trading at all, spending vs
  saving, mortgage vs sell, hotels, which set first, max vs spread) are 5 to 21 games in 100 and safe to lead
  with.
- Auctions: every decision was tested both ways. Only the auction bidding advice and the size of the passing
  penalty depend on them.

## Draft YouTube description (title A)

```
You can't choose where you land in Monopoly. You can choose whether to buy, when to trade, whether to chase
the railroads, how to build, and what to do in jail. We simulated each of those decisions over 100,000 games
of Monopoly played the way families play it: everyone buys what they land on, you only help someone finish a
set if they help you finish yours, and when the board locks up, people start dealing.

🎲 Never trading: 4.9 wins in 100 (a fair share is 25).
🏠 Keeping $300 back when building: 16.3 wins in 100, and you're MORE likely to go broke first.
🏠 Owe money? Mortgage before you sell houses (selling first costs 6 games in 100).
🏠 Two monopolies? Cheaper houses first. Orange + green: 84 vs 63.
🚂 Railroads: only worth it if you trade for the last one.
🔒 Jail: get out early; stay in once anyone has houses.
💵 Playing the typical way is like handing the bank $250 before you've rolled (about $200 without auctions).
📋 Do all of it and you win about 3 games for every 2 that anyone else at the table wins. Play weekly for a year and
   you finish as family champion two years in three. Same dice.

How it's made: a simulator of the standard US board and current cards under the official rules (auctions,
even building, 32 houses and 12 hotels, mortgages); its board probabilities match Truman Collins' published
tables. Every decision was also tested without auctions.
Method, code and every result: https://github.com/drburke-droid/nhl-sog-predictor/tree/master/monopoly
```

## Look and feel

Full art direction, palette, fonts, motion, sound and the Hasbro-free checklist: `Art direction.md`. In short:
the video is built from Monopoly's real paper trail. Lizzie Magie's public-domain 1904 patent drawing sets the
look of the board (our own drawing in her style), the real Atlantic City streets appear on period postcards
and maps, and the results arrive as 1930s paperwork: ledgers, rubber stamps, telegrams, railway tickets and a
certificate for the playbook. Style frames: `art/frames/01_title.png` (title card), `02_board.png` (the board),
`03_ledger_spend_or_save.png`, `04_railroads_telegram.png`, `05_end_card.png`, `06_thumbnail.png`.

On-screen game count: the numbers in this plan come from **13.4 million games** (100,000 per result); all the
realistic-play testing behind the project, including the evolution runs, totals about 65 million.

## Decided

- Endgame prices: 2.5x asking, up to 3x paid (as modelled).
- Headline: "three wins for every two of theirs" per game, and "family champion two years in three" over a
  year of weekly game nights (both with auctions). "33 vs 25" stays as the on-screen footnote. The no-auction
  figure (30.2, or 1.3 to 1) is mentioned once in the playbook section.
- The price tag is a featured beat, opening the playbook section, in the user's wording: "Playing the typical
  way is like handing the bank $250 before you've rolled" ($200 without auctions). $250 is measured in that
  direction; $300 is the other direction (what a typical player needs to catch up) and is used only as an
  optional follow-on line.

Data: `sim/results/summary.md` ("Final numbers for the video" and "Final: two complete sets"). Charts:
`sim/results/charts/final_*.png`.
