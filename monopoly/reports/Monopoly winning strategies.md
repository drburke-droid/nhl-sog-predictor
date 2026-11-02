# No Monopoly Strategy Beats Every Table

> **Update: re-run with realistic play.** Most results below come from a world where a player will sell the
> last card of a set for cash (about 2.5x its printed price), and several "players" are caricatures that only
> ever buy one colour. Real players buy what they land on and give an opponent a set only when they get a set
> back in the same trade. That realistic version is in `sim/results/summary.md` ("Realistic play"). What holds:
> never trading, keeping cash back and stopping at three houses all lose; building aggressively wins; orange is
> the best favourite (+2.9 wins per 100 games). What changes: a railroad or Boardwalk fan does as well as an
> ordinary player rather than collapsing; never buying green costs nothing rather than helping; never buying dark
> blue is the costliest gap (-5.5); and 13% of four-player games (57% of six-player games) never finish.
>
> **Evolution re-run under realistic play** (every player buys what it lands on; set-for-set swaps only): the
> evolved player wins **31 games in 100** against typical players' 21.8. Its playbook: pick two favourite cheap
> or middle sets (light blue and orange came out best) and go hard for them (mortgage to buy one, bid up to
> double at auction); bid above list at auctions generally; keep only $50-100 in hand; build to hotels,
> mortgaging loose property to do it; leave jail early and stay in late; make every fair set-for-set swap at
> printed prices. Never trading costs 23 wins per 100, haggling over set value 9.8, stopping at three houses
> 5.3, having no favourite 4.5. **The parasite, arms-race cycle and ecology findings below do not survive
> realistic play**: a typical player does worse, not better, against three copies of the evolved player (16.2),
> the arms race only shuffles the favourite colours, and the population ends up made entirely of evolved
> players with different favourite pairs. Details: `sim/results/summary.md` ("Realistic evolution") and
> `research_notes/.../our_simulations.md` section 15. The video plan below was built around the parasite story
> and needs a new angle if realistic play is the standard.

Under the official rules, the best way to win Monopoly is to buy almost everything you land on and trade for colour sets. Value those sets by what actually wins games (orange and light blue first, green last), bid hard at auctions, build past three houses to hotels with almost no cash held back, mortgage loose property to pay for the building, and leave jail early but stay in late. In our simulations (about 56 million games: 36 million in fixed experiments and 20 million inside a genetic-algorithm search), that playbook wins **33.1%** against three ordinary "buy everything" players, where a fair share is 25%. Evolution found a version that wins **36.7%** by also never buying yellow, green, railroads or utilities. Every popular shortcut loses: railroads only (**0.18%**, about one win in 550), Boardwalk and Park Place only (**6.0%**), orange only (**9.8%**), never trading (**4.4%**), keeping a $300 cushion (**18.7%**) and stopping at three houses (**19.6%**). The bigger finding is that no single best strategy exists. The 36.7% strategy is a parasite: it wins only because ordinary players overpay for the groups it skips. One ordinary player seated with three copies of it wins **71.5%**, and the next round of evolution produced a strategy that beats it but is only average against ordinary players. Published work supports the broad picture. Markov-chain analyses since 1972 rank orange best per dollar, and Perfect Play's October 2026 tournament found that buying broadly, trading and building aggressively win. No existing video covers the arms race, the population ecology, learned versus textbook property values, house rules, or table sizes from 2 to 8. That gap makes a strong video, but only if it leads with "what happens when strategies adapt" rather than with another ranked tournament. Perfect Play's tournament drew about 2.7M views three weeks ago, and a follow-up that only claimed a bigger game count drew about 10,000.

## Buying everything beats every named shortcut; only jail timing and aggressive building beat it

Every figure in this section comes from our C++ engine. It plays the standard US board with the current card decks and the official rules: auctions for unbought property, even building, a bank of 32 houses and 12 hotels, mortgages and bankruptcy. Each strategy played 100,000 games against three "Yes-Men", with seats randomised every game. A Yes-Man buys everything it can afford, builds hotels as fast as cash allows, leaves jail at once and trades at list-price fairness. The 95% interval on each win rate is about ±0.3 points ([our simulations](../sim/results/summary.md)). The engine is checked two ways. First, an independent exact Markov chain reproduces Truman Collins' published leave-jail table to two decimals: Illinois 3.19% against Collins' 3.1858%, and the jail square 6.22% combined ([Truman Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)). Second, the engine reproduces Perfect Play's headline result. Restricted to Perfect Play's swap-only trading, our never-trading player wins **6.4%** against **21.7%** for each normal player. Perfect Play reported 6.5% against 21% ([SozAI transcript of Perfect Play](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)).

The leaderboard splits into three tiers ([leaderboard chart](../sim/results/charts/leaderboard_official.png)). The top tier stays close to the Yes-Man baseline of **25.2%**:

- All-In: **27.7%**. It buys everything, mortgages loose lots to build, and leaves jail early but stays in late.
- Out-early/in-late jail: **26.7%**.
- Jailbird, which always stays in jail: **25.8%**.

The middle tier holds the cautious habits champions and tip videos recommend:

- House Hoarder: **23.2%**
- Three-House Rule: **19.6%**
- Accountant: **18.7%**

The bottom tier holds every "only buy X" strategy, led by orange only at **9.8%** and falling to **0.01%** for utilities only and for buying nothing at all. Picky strategies fail because you cannot choose where you land, so a restricted buyer rarely completes a set. Orange-only completes a set in **35%** of games and Railroad Tycoon in **1%**, against **90%** for the Yes-Man ([our simulations](../sim/results/summary.md)).

| Named strategy | Our win rate vs three Yes-Men (fair share 25%) | What published research says | Verdict |
|---|---|---|---|
| Buy only railroads | **0.18%** (railroads + utilities 0.2%); 0.00-0.01× fair share at every table size from 2 to 8 | Perfect Play's Railroad Baron (railroads + utilities, bids double) came last at 3% ([Kottke](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games)). Collins calls railroads "excellent investments" when owned together ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)). A two-player hill-climb ranked them first ([Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)) | The worst real strategy. As an add-on they are roughly neutral: skipping them costs 0.6 points |
| Wait for Boardwalk and Park Place | **6.0%**; rises from 0.21× fair share at 2 players to 0.56× at 8 | Boardwalk is the best single street, but Park Place is only the 33rd most-landed square ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)); Fry calls the set "just too rare" ([TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)) | Loses at every table size, although skipping dark blue entirely still costs 2.5 points |
| Buy everything you land on | **25.2%** (the control) | Perfect Play's Buy Everything placed second at 29% ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)); Hasbro's championship rulebook says "Buy everything you can afford early" ([Championship rules](https://www.monopolydocumentary.com/fileshare/Monopoly%20World%20Championship%20-%20Player%20Rules%20%28Final%29.doc)) | The right default |
| One colour (orange only) | **9.8%**; 1.0% with swap-only trading; **52.6%** heads-up under no-auction family rules | Orange has the best return per dollar ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)); Perfect Play's orange + red-only player won under 3% ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)) | Fight for orange, but don't limit yourself to it |
| Keep a cash cushion ($300) | **18.7%** | Perfect Play's Saver won 16% ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)); keeping £200 more cash costs 10-19% of wins ([oliverhutch2021](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)) | Loses. Hold cash only when buying loose lots, never when building |
| Stop at three houses | **19.6%**; reverting the evolved strategy to three houses costs **11.7 points**, the largest single habit | The third house has "the fastest payoff of any building on almost all of the properties" ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)); Perfect Play's Three Houses won 17% | Rush to three houses, then keep building |
| Hoard houses (four houses, never hotels) | **23.2%**, but **75.4%** as a counter-strategy against three evolved specialists | Hasbro: "The House Is King... hold other players at ransom!" ([Championship rules](https://www.monopolydocumentary.com/fileshare/Monopoly%20World%20Championship%20-%20Player%20Rules%20%28Final%29.doc)); a housing lock adds 5% at four players, almost all of it from cash-raising rather than withheld hotels ([oliverhutch2021](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)) | A counter-punch, not a default |
| Stay in jail | **25.8%** always staying; **26.7%** out early, in late | Collins: leave early, stay late ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)); the championship rulebook calls jail "the safest place" late in the game ([Championship rules](https://www.monopolydocumentary.com/fileshare/Monopoly%20World%20Championship%20-%20Player%20Rules%20%28Final%29.doc)) | A real but small edge of 0.6 to 1.5 points |
| Never trade | **4.4%** (0.18× fair share); 0.87× heads-up; 0.08× at 8 players | Perfect Play: 6.5% against 21% ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)); 0.20× at four players ([oliverhutch2021](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)) | The costliest habit at three or more players |
| Trade well | Shark with textbook values **18.3%**; Shark with learned values **27.6%**; Champion with learned values **33.1%** | "There's a three or four minute window..." when deals are made ([TNT Christopher, Pantograph Punch](https://pantograph-punch.com/posts/the-battleship-and-the-thimble-a-glimpse-into-the-monopoly-world-championships)) | What you value matters more than whether you trade |

The building results settle a long-running apparent conflict. Collins' payback tables show that New York's third house repays in **8.53 opponent rolls**, the best building on the board. Hannah Fry says "it's the third house that is really worth investing in quickly" ([TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)). The 2015 world champion Nicolò Falcone reportedly advises three houses per property ([Le Tribunal du Net, translating The Guardian](https://www.letribunaldunet.fr/famille/monopoly-nicolo-falcone-champion-monde-conseils-gagner.html)). All of that is about which purchase comes first, not about where to stop. Every full-game simulator, ours included, finds that stopping at three throws away the biggest rent jumps. Our ablation ranks the hotel habit above everything else ([ablation chart](../sim/results/charts/ablation.png)), and the colour duels show why: cheap sets need hotels to hurt anyone. With hotels, light blue wins 45.1% of its duels. Capped at three houses, it wins about 1%.

The cash results follow the same pattern. Idle cash loses, but the evolved strategy and the academic agents keep a cushion *when buying a loose lot*. The evolved strategy keeps about $300. The hybrid deep-RL agent of Bonjour, Haliem et al. buys only with "$200 more than the property price" unless the purchase completes a monopoly ([arXiv 2103.00683](https://arxiv.org/pdf/2103.00683)). Both then spend nearly everything on buildings. "Keep a cushion" is right for speculative purchases and wrong for development.

## Orange wins the board, every duel and the trade market; green is the one set to skip

The board maths is old and settled. Ash and Bishop modelled Monopoly as a Markov chain in 1972. Ian Stewart's 1996 Scientific American columns popularised it. Collins' exact tables became the standard reference ([UW course handout reprinting Ash & Bishop](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf); [Stewart, Oct 1996](https://abel.math.harvard.edu/archive/21b_fall_05/Articles/monopolyrevisited.pdf)). The jail square is occupied most. Illinois is the most-landed property, landed on **46% more often than Park Place** (3.19% against 2.19%, which matches Bewersdorff's "45%") ([Bewersdorff](https://bewersdorff-online.de/amonopoly/)). The orange lots sit 6, 8 and 9 squares past Jail, so they catch the steady stream of players leaving it. On Collins' numbers orange leads every return-per-dollar measure. The textbook reprint of Ash and Bishop puts orange hotels at $0.064 per dollar invested per roll, against green's $0.038 ([UW handout](https://sites.math.washington.edu//~perkins/381AWin14/handouts/chapter17.pdf)).

Rent collected does not predict winning, though. Among four Yes-Men, **Baltic Avenue collects 6.3% of all rent paid, more than Park Place (5.7%)** and second only to Boardwalk (6.6%), because brown hotels are cheap to build. A Baltic hotel charges $450 rent on $250 of buildings. Yet brown-only wins **0.3%** and brown loses 99-100% of its duels ([our simulations](../sim/results/summary.md)). The table below puts the classical measure (landing frequency) next to five full-game measures from our simulations.

| Group | Landings per roll, group total (Collins) | Two-player duel, average win | Four-player "one set each" share (fair 25%) | Points lost by never buying it | Learned value per set | Textbook value (3-house rent) | Completer goes on to win |
|---|---|---|---|---|---|---|---|
| Orange | 8.81% | **70.9%** | **53.0%** | **-5.9** | **$595** | $582 | **66.1%** |
| Light blue | 6.88% | 45.1% | 17.4% | -5.8 | $584 | $122 | 62.9% |
| Pink | 7.54% | 62.8% | 36.3% | -4.1 | $410 | $145 | 62.3% |
| Red | 8.76% | 62.1% | 33.1% | -1.7 | $168 | $515 | 62.4% |
| Yellow | 7.97% | 58.6% | 27.6% | -0.5 | $46 | $579 | 63.6% |
| Green | 7.80% | 48.6% | 15.9% | **+1.4** | **-$141** | $361 | 63.9% |
| Dark blue | 4.81% | 45.2% | 15.8% | -2.5 | $246 | **$604** | 56.2% |
| Brown | 4.29% | 0.4% | 0.8% | -2.3 | $233 | -$127 | 56.4% |
| Railroads | 11.38% | n/a | n/a | -0.6 | n/a | n/a | n/a |
| Utilities | 5.41% | n/a | n/a | +0.2 | n/a | n/a | n/a |

Sources for the table: landings from [Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml) (group sums computed in our notes); all other columns from [our simulations](../sim/results/summary.md).

Orange comes first on every measure. In two-player duels where each side owns one set and builds hotels as fast as it can, orange beats pink 56%, red 58%, yellow 62%, green 72%, dark blue 74% and light blue 75% ([duel matrix](../sim/results/charts/duel_matrix_b5.png)). When every player at a four-seat table holds one set, orange takes **53%** of wins, more than twice its fair share. It still takes 35% at an eight-seat table where the fair share is 12.5%. Dark blue loses to orange, red, pink, yellow and green.

The "points lost" column is our most practically useful new result. It comes from removing one group from the Yes-Man's shopping list and measuring the change in its win rate. **Green is the only group you are better off never buying** (+1.4 points), and the ordering holds under family rules too. Fry's claim that green becomes "your best shot" with more than three opponents ([TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)) is not supported. Green takes 5.0% of wins at an eight-player one-set table against a fair 12.5%, and a green-only player scores 0.03× fair share at six to eight players. Her advice to target orange holds at every table size we tested. Her other picks hold only partly: red ranks third rather than second when three players each hold one set, and light blue is strong in trade but weak in a one-on-one duel.

The two value columns explain the most surprising result in the tournament: a hand-written expert built from champions' advice finished **below** the naive Yes-Man. We first priced sets the textbook way, by expected rent at three houses over 25 opponent turns minus building cost. That puts dark blue first ($604) and light blue near the bottom ($122). The Shark trader using those prices wins **18.3%**, worse than a player who trades at list price. It sells the cheap sets that actually win and pays up for dark blue and green. We then repriced each set by what the simulation measured: the points of win rate a player loses by never buying it, times $100. The same Shark rises to **27.6%**, and the expert Champion playbook rises from **23.5%** to **33.1%** ([our simulations](../sim/results/summary.md)). An independent study arrived at almost the same order by a different route. A two-player hill-climb over 14.5 million strategy rankings found railroad, orange, light blue, brown, pink, utility, dark blue, red, yellow, green ([Eithun, Lamb & Young 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)). Apart from railroads, that matches our learned order closely. In the deep-RL study, the fixed bot prioritising railroads plus orange or light blue usually out-won the one prioritising railroads plus Boardwalk and Park Place ([arXiv 2103.00683](https://arxiv.org/pdf/2103.00683)). Hasbro's account of the 2009 final also fits. Bjørn Halvard Knappskog reportedly won by trading a red property for a light blue and building hotels at once ([Hasbro newsroom](https://newsroom.hasbro.com/node/15036)). By our learned values that trade swapped a $168 set for a $584 one.

## House rules barely matter at four players but flip heads-up games

Most families do not play by the rulebook. In Hasbro's 2014 survey, **68%** had never read all the rules and **49%** had made up their own, though the figures vary between outlets ([Hasbro newsroom](https://newsroom.hasbro.com/node/14166); [AP via KSL](https://www.ksl.com/article/29337113/hasbro-picks-5-house-rules-for-new-monopoly-set)). Under the official rules, an unbought property must be auctioned, Free Parking pays nothing, and jailed players still collect rent ([Hasbro rules](https://www.hasbro.com/common/instruct/00009.pdf)). The auction is older than Monopoly itself. An early folk variant, "Auction Monopoly", auctioned every lot, and the Atlantic City Quakers replaced that with fixed prices ([Wikipedia: History of Monopoly](https://en.wikipedia.org/wiki/History_of_Monopoly)). We ran the full tournament in six rule worlds, including "family" (no auctions plus a Free Parking jackpot) and "kitchen" (family plus $400 for landing on Go).

| Strategy (win %, vs three Yes-Men) | Official | Swaps only | No trading | No auctions | Family | Kitchen |
|---|---|---|---|---|---|---|
| All-In | 27.7 | 24.2 | 8.4 | 27.9 | 27.9 | 27.5 |
| Yes-Man | 25.2 | 21.7 | 8.4 | 25.5 | 25.1 | 25.1 |
| Three-House Rule | 19.6 | 17.9 | 7.4 | 19.3 | 18.6 | 18.0 |
| Accountant | 18.7 | 15.3 | 5.6 | 18.2 | 18.8 | 19.6 |
| Orange only | 9.8 | 1.0 | 1.0 | 10.2 | 9.9 | 9.8 |
| Boardwalk Dreamer | 6.0 | 4.8 | **8.8** | 6.5 | 7.3 | 7.6 |
| Never trades | 4.4 | 6.4 | 8.4 | 4.6 | 4.4 | 4.4 |
| Games that never end (four Yes-Men) | 0.0% | 13.1% | **67.2%** | 0.0% | 0.0% | 0.0% |

Source: [our simulations](../sim/results/summary.md); [rules chart](../sim/results/charts/rules_change_winner.png).

At four players, house rules barely change the ranking. The rule that matters is trading. Without it, **two games in three never end**, which matches the roughly 70% stalemate rate reported for b2studios' basic bots ([fomo.ai, snippet](https://learn.fomo.ai/p/what-happened-when-an-ai-played-11)). In that world the Boardwalk Dreamer (8.8%) does as well as buying everything (8.4%), because the only sets anyone finishes are ones they land on, and dark blue has only two lots. The swap-only column also explains a published result. Orange-only falls from 9.8% to 1.0% when it cannot buy the missing piece for cash, which is why Perfect Play's orange-and-red player won under 3% in a swap-only world. Contrary to folklore, the Free Parking jackpot did not lengthen our games (47.3 rounds against 48.2). That matches the shorter median game in gamesdecoded's 200-million-game simulator, which also found the jackpot makes rare endless games about five times more common ([gamesdecoded](https://github.com/gamesdecoded/monopoly-simulator)).

Heads-up is where rules flip outcomes. Under official rules every lot you decline goes to auction, and with no other bidder your opponent can win it for the engine's $10 minimum bid. Orange-only therefore wins **18.6%** against a Yes-Man under official rules but **52.6%** under family rules, where declined lots stay unsold. The Boardwalk Dreamer rises from 10.3% to 35.0% the same way. Family rules also create rock-paper-scissors loops: the Champion beats orange-only 76%, orange-only beats the Yes-Man 53%, and the Yes-Man beats the Champion 53%. Official rules produce no such loops ([head-to-head matrix](../sim/results/charts/h2h_matrix_family.png)).

| Strategy (win rate ÷ fair share) | 2 players | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| All-In | 1.06 | 1.06 | 1.11 | 1.12 | 1.12 | 1.08 | 1.09 |
| Never trades | **0.87** | 0.30 | 0.18 | 0.14 | 0.11 | 0.10 | **0.08** |
| Boardwalk Dreamer | 0.21 | 0.38 | 0.24 | 0.23 | 0.33 | 0.44 | 0.56 |
| Orange only | 0.38 | **0.60** | 0.39 | 0.29 | 0.24 | 0.20 | 0.17 |
| Three-House Rule | 0.95 | 0.84 | 0.79 | 0.75 | 0.75 | 0.73 | 0.74 |

Source: [our simulations](../sim/results/summary.md); [players chart](../sim/results/charts/players.png).

Table size changes the cost of habits, not the winner. All-In stays above average at every size. Refusing to trade is nearly free heads-up and ruinous at eight players, which matches oliverhutch2021's 0.98× at two players and 0.20× at four ([FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)). Turn order is a larger effect than most habits. The first player wins **28.3%** of four-player games and the last **22.1%**, so going first wins 28% more often. The first seat wins 50.9% heads-up and 16.1% at eight players. gamesdecoded measured a similar 27.3% against 22.6% ([gamesdecoded](https://github.com/gamesdecoded/monopoly-simulator)).

## Evolution finds a 37% strategy, then exposes it as a parasite

We let a genetic algorithm design its own player. Each of 25 generations held 24 strategies, and fitness was the win rate against three Yes-Men. The genes were: which of the ten groups to buy, cash reserves, three auction bid limits, maximum building level, build order, jail policy, mortgaging to build, trade valuation, and buy and sell prices for set-completing lots. The best random strategy won 26.3% and the median 12.8%. The best reached **37%** by generation 9 and then plateaued. The final top five, re-tested on 200,000 games each, all scored **36.6-36.7%** (about 47% more wins than a fair share) and agreed on one recipe ([evolution chart](../sim/results/charts/evolution.png)):

- Buy brown, light blue, pink, orange, red and dark blue; never buy yellow, green, railroads or utilities.
- Keep about $300 back when buying on landing.
- At auction, bid up to 1.2× list, and 2-3× to complete a set or to block one.
- Pay up to 6× list for the last piece of a set.
- Build to hotels keeping only $50-100, mortgaging loose property to fund it.
- Leave jail early and stay in late.
- Trade with learned values.

The 6× figure matches oliverhutch2021's finding that the third orange is worth up to almost six times its printed price, because two oranges win 6% of games without it and 90% with it ([oliverhutch2021 README](https://github.com/oliverhutch2021/monopoly-strategy)). Perfect Play's computer-tuned player differed in one telling way: it bought everything, and won 26% against normal players at 19% ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)). The ablation ranks which habits matter, by how much undoing each one costs: three houses instead of hotels **-11.7 points**, buying everything -4.4, bidding only list price -2.4, buying green -2.3, always leaving jail at once -2.2. Everything else costs about a point or less.

The twist came when we seated three copies of the evolved champion with one plain Yes-Man. **The Yes-Man won 71.5%** and each champion 9-10%. An All-In won 64.9% at the same table, and even a never-trader won 60.4%. The mechanism is visible in the game logs. All three champions skip yellow, green, railroads and utilities, so the Yes-Man completes yellow in 87% of games and green in 84%, and railroads collect 16.8% of all rent instead of the usual 7.2% ([our simulations](../sim/results/summary.md)). The evolved strategy wins only while ordinary players are around to overpay for the groups it ignores.

A second evolutionary stage made this an arms race: each round evolved a strategy to beat three copies of the previous champion.

- **Round 1** produced a generalist house-hoarder. It buys everything, never builds past four houses, stays in jail, keeps no reserve and bids up to 3× to block. It beat three round-0 specialists **75.4%** of the time but managed only **24.4%** against Yes-Men, an average result.
- **Round 2** went back to a specialist that skips green and railroads. It won 33.1% against the round-1 generalists and 36.0% against Yes-Men.
- **The cycle closed**: the round-1 generalist beats three round-2 specialists **66.0%**, and a plain Yes-Man beats them 61.0%.

Specialists exploit ordinary players, generalists exploit specialists, and ordinary players exploit tables full of specialists. Other studies saw this non-transitivity before, but at much smaller scale. Eithun et al. found two-player cycles of 58%, 55% and 56% ([Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)). oliverhutch2021 found that each strongest agent can be beaten by a counter-strategy ([README](https://github.com/oliverhutch2021/monopoly-strategy)). Bonjour, Haliem et al. cite earlier work that "no single strategy can maintain high win rates against all other strategies" ([arXiv 2103.00683](https://arxiv.org/pdf/2103.00683)).

The ecology experiment asks which strategy survives when everyone adapts. Eleven strategies played every possible four-seat table: 1,001 tables and about 5.5 million games. Each generation, a strategy's population share grew in proportion to its win rate at tables drawn from the current population. The Yes-Man, standing in for the typical family player, was extinct by about generation 20. Railroad Tycoon, orange-only, the Boardwalk Dreamer and the never-trader died out within a few generations. The population settled into a stable mix: **71% Champion** (the well-rounded expert with learned set values, which buys everything, bids smartly, builds hotels and times jail) and **29% evolved specialist** living off it ([ecology chart](../sim/results/charts/ecology.png)).

That end state is the honest answer to "what is the optimal strategy":

| Who else is at the table | Best play from our results | Expected win rate |
|---|---|---|
| Ordinary buy-everything players | Evolved specialist: skip yellow, green, railroads and utilities; hotels; bid hard; pay up to 6× for your last piece | 36.7% (fair share 25%) |
| Unknown or mixed table | Learned Champion: buy everything (green is optional), trade by learned values, hotels with a minimal reserve, out of jail early and in late | 33.1% against ordinary players; 41.1% in an evenly mixed population; survives evolution |
| Picky specialists | Generalist: buy everything they skip, hold at four houses to starve them of houses, bid to block | 75.4% against three specialists |
| Opponents as good as you | Anything sensible | 25% (four learned Sharks win 25.1% each) |

Even the best strategy loses about two games in three. A 1.5-point edge, such as good jail timing, adds roughly one extra win every 67 games. "How to win every time" does not exist.

## The video works if it sells the arms race, not the tournament

### Perfect Play owns the tournament premise, so the open lanes are adaptation, rules and table size

The audience is large and keeps coming back. b2studios' "AI Learns Insane Monopoly Strategies" has about **12.1M** views. Stand-up Maths with Hannah Fry has about **3.25M**. Short "how to win every time" videos from What's What and Way Too Simple drew about **6.9M** and **2.7M** this autumn ([RYD: b2studios](https://returnyoutubedislikeapi.com/votes?videoId=dkvFcYBznPI); [RYD: Stand-up Maths](https://returnyoutubedislikeapi.com/votes?videoId=ubQXz5RBBtU); [RYD: What's What](https://returnyoutubedislikeapi.com/votes?videoId=qu6GMpQfyUU)). The direct competitor is Perfect Play's "The Best Monopoly Strategy, Tested on 300 Million Games". It has about **2.67M** views in roughly three weeks and was picked up by Kottke, Neatorama and Hacker News ([RYD: Perfect Play](https://returnyoutubedislikeapi.com/votes?videoId=S_2eVAaw1UM); [Kottke](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games)). A bigger number alone does not work: Popperish's "We Played 950 Million Games" has about **10,000** views ([RYD: Popperish](https://returnyoutubedislikeapi.com/votes?videoId=pTyqhhWw614)).

Perfect Play's like-to-view ratio is about 0.6%, against about 2.5% for b2studios and 3% for What's What. Hacker News commenters criticised its AI narration ([Hacker News](https://news.ycombinator.com/item?id=50175732)). That points to room for a human presenter, though it is an inference rather than a measured cause. The same thread asked for realistic trading, house-shortage tactics, house rules, self-play learning and rock-paper-scissors dynamics. Perfect Play itself tested only four-player tables and no house rules. Our data answers most of those requests directly.

| Our finding | Headline number | Closest prior work | Genuinely new? |
|---|---|---|---|
| The evolved best strategy is a parasite | Yes-Man wins **71.5%** against three of it | None on YouTube | **Yes**, the strongest hook |
| Evolution goes in circles | 75.4% / 66.0% / 61.0% cycle | Two-player cycles of about 55-58% (Eithun); counter-strategies (oliverhutch2021) | **Yes** in scale and as a story |
| Population ecology end state | Champion 71% / specialist 29%; Yes-Man extinct | None found | **Yes** |
| Textbook valuation loses; learned valuation wins | 18.3% → 27.6%; expert 23.5% → 33.1% | Eithun's expected-value strategy ranked low | **Yes** for video; ties to Knappskog's trade |
| Green is the only group worth skipping | +1.4 points | Games Computers Play asked "ignore greens?" (about 186k views) | **Yes**, first full-game quantification found |
| Baltic out-earns Park Place in rent | 6.3% against 5.7% of all rent | None found | **Yes**, an ideal Short |
| House rules: stable at four players, flip heads-up | Orange-only 18.6% → 52.6% heads-up | gamesdecoded measured game length only | **Yes** |
| Table size 2 to 8 | Never-trading 0.87× → 0.08× | oliverhutch2021 covered 2-6 on GitHub | New to video |
| Colour duels and one-set royales | Orange beats every set; 53% share at four players | Perfect Play ran colour pairs (only partly transcribed) | Partly new |
| Hotels are the most important habit | -11.7 points | Perfect Play: three houses loses | Magnitude is new |
| Leaderboard: All-In top, railroads last, cushions and three houses lose | 27.7% top, 0.18% bottom | Perfect Play | **No**; use it as a fast replication beat |

### Angle: there is no best strategy, only a best reply to your table

The pitch is a sequel that answers back: "Perfect Play asked which strategy wins. We asked what happens when the strategies fight back." The video should credit Perfect Play in the first two minutes and replicate its headline in about 30 seconds (6.4% against its 6.5%), which buys credibility. Then it should set up the direct contradiction. Perfect Play's tuned bot bought everything; ours evolved to skip four groups and won more. Which one is right? Both, and explaining why is the video. That structure turns a well-trodden ranking into a mystery with a payoff. It answers the Hacker News complaints ("dull" strategies that never adapt, no self-play, no house rules), and it lets the presenter end with practical advice that changes with the table. Publishing the code is a second differentiator, since Perfect Play's methods are not available in text form.

A history beat fits the closing. Lizzie Magie designed the game in 1904 as "a practical demonstration of the present system of land-grabbing" ([NWHM](https://www.womenshistory.org/articles/monopolys-lost-female-inventor)), and our evolved population ends as nearly all full-board monopolists plus one parasite.

Timing: Stand-up Maths released its Monopoly video on 8 December 2016 ([Mental Floss](https://mentalfloss.com/article/90001/watch-mathematics-winning-monopoly)), and Fry's material comes from a Christmas maths book. An early-December release would catch holiday board-game season while Perfect Play's audience is still warm. This is a judgement, not a measured effect.

### Eight acts, each ending on a number

| Act (target time) | Question | Reveal and numbers | Main visual |
|---|---|---|---|
| Cold open (0:00-0:50) | Can the best strategy lose to a normal player? | Evolution bred a strategy that wins 36.7%. Seat three copies with one ordinary player and the ordinary player wins 71.5%. Credit Perfect Play and promise the explanation | Replay of `evolved_vs_yesmen` sample games; the 71.5% hard cut |
| 1. The board (0:50-3:00) | Where do tokens land? | Illinois is landed on 46% more often than Park Place; orange sits 6-9 squares past Jail; matches Collins' tables. Twist: Baltic collects more rent than Park Place, yet brown-only wins 0.3% | Animated board heat map; rent-share board |
| 2. The myth tournament (3:00-7:00) | Do the famous strategies work? | Countdown: Scrooge 0.01%, Railroad Tycoon 0.18% (1 in 550), Boardwalk Dreamer 6.0%, orange only 9.8%, never-trader 4.4%, $300 cushion 18.7%, three houses 19.6%, hoarder 23.2%, Yes-Man 25.2%, Jailbird 25.8%, All-In 27.7%. Replicate Perfect Play (6.4% against 6.5%). Plant the mystery: our champion-advice expert lost to the Yes-Man (23.5%) | `leaderboard_official.png`, revealed bottom-up |
| 3. Colour wars (7:00-10:00) | Which set is best? | Orange beats every set head-to-head and takes 53% at a one-set-each table; dark blue loses to five sets. Cap at three houses and red edges orange while light blue collapses to about 1%. Green is the only group you should never buy (+1.4) | Duel bracket; `skip_one_group_official.png`; `set_completed_win.png` |
| 4. Your house rules (10:00-12:30) | "I don't play official rules, so does this apply to me?" | 68% never read the rules. At four players the rankings barely move; with no trading, 67% of games never end. Heads-up, the auction rule flips orange-only from 18.6% to 52.6%. Never-trading is 0.87× at two players and 0.08× at eight. Going first wins 28% more | `rules_change_winner.png`; `h2h_matrix_family.png`; `players.png` |
| 5. What is a property worth? (12:30-14:30) | Why did the expert lose? | Textbook prices put dark blue first ($604) and light blue near the bottom ($122). Learned prices: orange $595, light blue $584, green -$141. Same bot, new prices: 18.3% → 27.6%; expert 23.5% → 33.1%. Knappskog's red-for-light-blue trade. Four learned Sharks win 25.1% each: no free lunch | New dumbbell chart of textbook against learned values |
| 6. Evolution (14:30-16:30) | What does a computer invent? | Median 12.8% → best 37% by generation 9. Recipe: skip yellow, green, railroads and utilities; hotels; bid hard; pay up to 6× for the last piece. Ablation: three houses costs 11.7 points | `evolution.png`; `ablation.png` |
| 7. The parasite and the arms race (16:30-19:00) | Why did it lose to a normal player? | The Yes-Man completes yellow in 87% and green in 84% of games, and railroads take 16.8% of rent. Round 1 generalist: 75.4% against specialists, 24.4% against Yes-Men. Round 2 specialist: 36.0% against Yes-Men but loses 66.0% to the generalist. Rock-paper-scissors | New triangle diagram; replayed sample games |
| 8. Ecology and the playbook (19:00-21:30) | What survives when everyone learns? | Yes-Man extinct by about generation 20; the stable mix is 71% Champion and 29% parasite. The playbook card, and why even the best loses two games in three. Magie callback | `ecology.png` animated; end-card checklist |

### Visuals: twenty-two charts rendered, six to build

| Asset (in `monopoly/sim/results/`) | Use | Fix before broadcast |
|---|---|---|
| `charts/board_probability_leave.png`, `board_probability_stay.png` | Act 1 board heat map; toggle between jail policies | Re-render onto a board image rather than a bar chart |
| `charts/leaderboard_official.png` (plus `_family`, `_notrade`, `_swaponly`) | Act 2 countdown; Act 4 comparisons Done: the textbook-valued bot is labelled "Textbook expert" in these charts, so it is not confused with the learned Champion in Acts 5 and 8 |
| `charts/duel_matrix_b5.png`, `duel_matrix_b3.png` | Act 3: matrix to bracket, then the three-house flip | Animate as a knockout bracket |
| `charts/royale_3.png`, `royale_4.png`, `royale_8.png` | Act 3 one-set free-for-alls | None |
| `charts/skip_one_group_official.png`, `_family.png` | Act 3 green reveal | None |
| `charts/set_completed_win.png` | Act 3: completing orange wins 66% | Note that completions can repeat when a set changes hands |
| `charts/rules_change_winner.png` | Act 4 slope chart | Done: labels spread apart, and retitled "Take away trading and everyone stalls; house rules barely matter" to match the finding |
| `charts/game_length.png` | Act 4: no-trade games never end | None |
| `charts/h2h_matrix_official.png`, `h2h_matrix_family.png` | Act 4 heads-up flip and loops | Highlight the 18.6% → 52.6% cell |
| `charts/players.png` | Act 4 table size | None |
| `charts/evolution.png`, `ablation.png` | Act 6 | None |
| `charts/ecology.png` | Act 8 | The x-axis is logarithmic; animate generation by generation instead |
| `sample_games/cast_seed*.jsonl`, `evolved_vs_yesmen_seed*.jsonl` | Animated real games for the cold open and Acts 2, 6 and 7 | Build a board-replay renderer |
| To build | Rent-share board (Baltic against Park Place); textbook-against-learned dumbbell chart; parasite table showing the "food" groups; arms-race triangle; first-seat bar chart; playbook end card | Data is in `summary.md` and `learned_values.json` |

### Titles and thumbnails should promise the twist, not the game count

The strongest title is **"We Evolved the Perfect Monopoly Strategy. A Normal Player Beat It."** It promises the parasite reveal, which must arrive in the cold open with the "three copies at one table" condition stated. Alternatives:

- "The Best Monopoly Strategy Is a Parasite" (maximum curiosity; needs fast explanation)
- "I Let Monopoly Strategies Evolve. It Turned Into Rock, Paper, Scissors."
- "Every Monopoly Strategy Ranked, Then Evolution Broke the Ranking"
- "Does Your House Rule Change How to Win Monopoly?" (for the large audience that plays house rules)

Avoid leading with "56 million games": Perfect Play's 313 million and Popperish's 950 million already own that framing, and the bigger number did not help Popperish.

Thumbnail concepts:

- **"71%"**: one ordinary token facing three identical robot tokens, with a huge 71% beside it.
- **Rock-paper-scissors**: three hands holding a house, a hotel and a railroad card.
- **Never buy this**: the green set with a red X.
- **Railroads**: "0.18%" stamped across the four railroads.

Use your own board photography rather than Hasbro character art.

### Shorts: twelve one-number reveals

| Short | Hook (first two seconds) | Payoff |
|---|---|---|
| Never buy green | "There's one colour you should never buy." | Skipping green is worth +1.4 points; it is the only such group |
| Railroad Tycoon | "Buying only railroads wins one game in..." | 550 (0.18%) |
| Baltic beats Park Place | "Baltic Avenue out-earns Park Place." | 6.3% against 5.7% of all rent, yet brown-only wins 0.3% |
| Orange against everyone | "Orange against every other colour, one on one." | Wins every pairing; brown wins 0-1% |
| Stop at three houses? | "The champions' favourite rule is the costliest habit." | -11.7 points |
| Go first | "Who rolls first matters more than most strategies." | 28.3% against 22.1%; 50.9% heads-up |
| Never trade | "Refusing every trade is fine... until a fifth person sits down." | 0.87× fair share heads-up, 0.08× at eight players |
| Free Parking jackpot | "Free Parking money makes games longer, right?" | 47.3 against 48.2 rounds (bots; caveat) |
| The parasite | "We built a super-strategy. Then this happened." | The normal player wins 71.5% |
| House rules, heads-up | "Your house rule makes orange unbeatable one on one." | 18.6% → 52.6% |
| Illinois against Park Place | "The most-landed property isn't the one you think." | 46% more landings than Park Place |
| Jail | "Should you stay in jail?" | Out early, in late: 26.7% against 25.2% |

### Claims to soften, verify or drop on camera

| Claim | Problem | Safe on-camera wording |
|---|---|---|
| Perfect Play's figures (43%, 29%, 17%, 3%, 6.5% against 21%) | Taken from a third-party transcript summary that contradicts itself (Three Houses appears as both "23" and "17%"); 312.9M games against "70 tables × 120,000"; its per-bot rates across mixed tables are not comparable with our vs-three-Yes-Men rates ([SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)) | Re-watch the video and quote only what it says; compare like with like (the swap-only replication) |
| Falcone: "three houses, stop short of hotels" and "jail is your enemy at the start... best friend towards the end" | Translated back from French and paraphrased by Mr Porter; the original Guardian text was not retrieved ([Le Tribunal du Net](https://www.letribunaldunet.fr/famille/monopoly-nicolo-falcone-champion-monde-conseils-gagner.html); [Mr Porter](https://www.mrporter.com/en-ie/journal/lifestyle/board-games-christmas-holidays-monopoly-scrabble-family-2022-23691802)) | "Falcone reportedly told The Guardian..." on screen as a paraphrase, or find the original |
| Knappskog's red-for-light-blue trade, "deprive your opponents" quote and "never buy utilities or railways" | Hasbro release and Motherboard tips were seen only as search snippets ([Hasbro](https://newsroom.hasbro.com/node/15036); [Motherboard](https://motherboard.vice.com/read/10-essential-tips-from-a-monopoly-world-champion)) | Verify before building Act 5 on the trade; otherwise say "according to Hasbro's announcement" |
| Hasbro TikTok "Don't rush the hotels. Four houses hit hard." | Taken from a TikTok discovery page, not a verified clip ([TikTok](https://www.tiktok.com/discover/monopoly-tips-and-tricks)) | Find and screen-record the actual clip, or drop it |
| b2studios' AI favoured railroads; 11.2M games | Secondhand (a university blog and a snippet); the game count is reported inconsistently ([UGA blog](https://ctlsites.uga.edu/mist5720/?p=1522)) | "Reportedly favoured railroads" |
| Stand-up Maths simulated 100M (or 1M) rolls | Outlets disagree ([Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math); [Nerdist](https://nerdist.com/article/the-math-of-winning-monopoly/)) | Don't quote a size |
| Frayn (2005) evolutionary study findings | Full text never read | Cite only as the first GA study of Monopoly |
| Chance deck has 9 or 10 movement cards | Sources disagree | Count a current US deck before saying a number |
| "Jail is the most-landed square" | True only if In Jail and Just Visiting are combined ([Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)) | "Illinois is the most-landed property" |
| "68% never read the rules" | Outlets report 66-70% and 34-49% for made-up rules; the figure is about reading, not finishing games ([Hasbro](https://newsroom.hasbro.com/node/14166)) | "About two in three Americans have never read all the rules" |
| "First player wins 50% more" (SumProduct) | Unverified and from a small simulation | Use our 28.3% against 22.1% and gamesdecoded's 27.3% against 22.6% |
| WWII escape-map sets with real money; "thousands escaped" | Disputed; the "744 airmen" figure covers all escape aids ([Mental Floss](https://www.mentalfloss.com/history/world-war-ii/how-brilliant-intelligence-officer-used-monopoly-free-wwii-pows)) | "MI9 hid escape aids in Monopoly sets," flagged as disputed |
| "Our chain matches Collins to two decimals" | True for the leave-jail table; for the stay-in-jail policy, jail occupancy is 9.39% against Collins' 9.46% | Say "matches the published table" for the leave-jail policy only |
| "56 million games" | 36M in fixed experiments plus 20M inside the GA search; never one tournament | State the split if quoted at all |
| "Champion" | Two versions: textbook values (23.5%) and learned values (33.1%; dominates the ecology) | Use distinct names on screen |
| "A normal player beats the perfect strategy" | True at a table of one Yes-Man and three evolved copies; the evolved strategy wins 36.7% against three Yes-Men | Say the table makeup every time |
| Learned set values | Fitted against Yes-Men, so partly in-sample | "Values learned against ordinary players" |

### What the simulator cannot tell you

Champions describe Monopoly as a negotiation game. TNT Christopher puts the decisive moment in "a three or four minute window" of deal-making ([Pantograph Punch](https://pantograph-punch.com/posts/the-battleship-and-the-thimble-a-glimpse-into-the-monopoly-world-championships)). Our bots do none of that, and several other limits matter:

- **Trading**: bots only make set-for-set swaps and cash buyouts at invented price rules (fair players sell at 2.5× list and pay up to 3×). They make no multi-party deals, no bluffs and no alliances. The trading results are only as good as those rules, and the swap-only world shows how sensitive game endings are to them: 13.1% of games never end there, against about 0% with cash deals.
- **Auctions**: the engine's $10 minimum bid drives the heads-up auction effect. Real players, and the championship rule that hands an unbid lot to the richest player for $1, would behave differently ([Championship FAQ](https://www.monopolydocumentary.com/fileshare/Monopoly%20World%20Championship%20FAQs%20%28Final%29.doc)).
- **Scoring**: we count outright wins and call a game a draw after 1,000 rounds. Tournaments score net worth at a time limit, which rewards different play.
- **Game version**: the engine plays the classic US game without the Speed Die, which was mandatory at the 2009 championships ([Wikipedia](https://en.wikipedia.org/wiki/Monopoly_%28game%29)).
- **Opponents**: the "typical player" is an assumption. The Yes-Man builds with no reserve, which may be bolder than real families.
- **Search scope**: evolution searched only the listed genes; the ecology pool held only 11 strategies; arms-race round 2 stopped after 10 of 12 generations.
- **Effect sizes**: differences under about a point are real in 100,000-game samples but invisible at a family table.

None of this is peer reviewed. Publishing the code and inviting replication is both the honest move and a selling point.

## Conclusion

The question "what is the best Monopoly strategy?" has a misleading premise. Monopoly behaves like an evolutionary game. Exploitative specialists beat ordinary players, generalists beat specialists, and a robust all-rounder that prices sets by their effect on winning dominates a population that keeps adapting, while a parasite survives alongside it. The classical maths was right about orange but mispriced everything else, because return per dollar ignores how often a set is completed and what it is worth in trade. Light blue is worth about as much as orange, and green is worth less than nothing. Champions' advice holds up on jail timing and on trading, and fails on stopping short of hotels.

For the video, the open lane is adaptation rather than another tournament. Lead with the parasite, pay it off with the arms race and the ecology, keep Perfect Play's ranking as a 30-second replication, and end with advice that depends on who is at the table. The natural next experiment would put humans or LLM players against these bots to measure the one skill the simulations leave out: negotiation.
