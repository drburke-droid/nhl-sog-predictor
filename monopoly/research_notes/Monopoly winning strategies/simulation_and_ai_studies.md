# Simulation, Evolutionary and Reinforcement-Learning Studies of Monopoly Strategy

Research note scope: what computer simulations, genetic/evolutionary searches, RL agents and LLM agents have found about which Monopoly strategies win. Figures are copied exactly from the sources. Where a figure comes only from a search-engine snippet or a third-party summary (I could not open the page), it is marked **[snippet only]** or **[secondhand]**.

Access problems in this session (reported so the writer knows what was not checked): the Frayn CIG 2005 PDF host (csapps.essex.ac.uk), learn.fomo.ai, v2.towardsdatascience.com and web.archive.org did not resolve from the session network. towardsdatascience.com, medium.com, sumproduct.com, semanticscholar.org and researchgate.net returned HTTP 403. The Bailis 2014, Bonjour/Haliem 2021-22, Eithun 2017, Gopalakrishnan 2021 and Stewart 1996 PDFs were downloaded and read in full text.

---

## 1. What did Colin Frayn's "An Evolutionary Approach to Strategies for the Game of Monopoly" (IEEE CIG 2005) find?

### Takeaway
The full text could not be retrieved, so its specific findings (which property groups the evolved agents value, jail and trading behaviour) are **unverified here**. What can be confirmed is its scope: a genetic algorithm evolved property valuations and portfolio-management rules. Later peer-reviewed work (Eithun, Lamb & Young 2017) built a simplified "discrete approximation" of Frayn's approach, and its results (railroads, orange and light blue ranked highest) are the closest verified stand-in.

### Cited Findings
- Bibliographic record: Colin Frayn (Computer Science, University of Birmingham), "An evolutionary approach to strategies for the game of Monopoly (TM)", IEEE Computational Intelligence in Games conference, 2005. The Birmingham repository lists it as "Contribution to conference (unpublished)" with no abstract or PDF. — [University of Birmingham record](https://research.birmingham.ac.uk/en/publications/an-evolutionary-approach-to-strategies-for-the-game-of-monopoly-t/)
- Venue: Proceedings of the 2005 IEEE Symposium on Computational Intelligence and Games (CIG05), Essex University, Colchester, UK, 4-6 April 2005. Abstract (as quoted in the search result): "the techniques of evolutionary computation are applied in order to evolve the most efficient strategy for property valuation and portfolio management in Monopoly." The PDF is reportedly at http://csapps.essex.ac.uk/cig/2005/papers/p1035.pdf, which did not resolve from this session. **[snippet only]** — [Semantic Scholar](https://www.semanticscholar.org/paper/An-Evolutionary-Approach-to-Strategies-for-the-Game-Frayn/524214efbfc703d11e449d7b14087ea2274f967c); [ResearchGate](https://www.researchgate.net/publication/221157585_An_Evolutionary_Approach_to_Strategies_for_the_Game_of_Monopoly)
- Citing papers describe Frayn's approach as using "good evolved estimates of the true value of each property", and as treating properties that are landed on more often as more valuable because they generate rent. **[snippet only; citing papers not identified by name in the result]** — [Semantic Scholar](https://www.semanticscholar.org/paper/An-Evolutionary-Approach-to-Strategies-for-the-Game-Frayn/524214efbfc703d11e449d7b14087ea2274f967c)
- Eithun, Lamb & Young (2017) say that "genetic algorithms have been used in Frayn [2005] and Mukhar [2013] to rank the properties and to approximate optimal strategy". They say their own static strategies "are discrete approximations of the strategies in Frayn [2005]": instead of numerical values per property they rank the 10 groups, which "allows us to replace the genetic algorithms used in Frayn [2005]" with hill-climbing. They also note that Mukhar [2013] encoded each strategy as "a chromosome of 160 real numbers". — [Eithun, Lamb & Young, UMAP Journal 38.1 (2017), PDF](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- A separate genetic-algorithm study of the Mega Edition reportedly found "no significant advantage in bidding more than 1.5 times the face value of the property". This is **not** Frayn's paper, and the Mega Edition board differs from the standard one. **[snippet only]** — [ResearchGate: Optimizing Strategies for Monopoly: The Mega Edition Using Genetic Algorithms & Simulations](https://www.researchgate.net/publication/315800283_Optimizing_Strategies_for_Monopoly_The_Mega_Edition_Using_Genetic_Algorithms_Simulations)

### Inferences
- For the video, Frayn (2005) is safe to cite as "the first evolutionary/GA study of Monopoly strategy, focused on property valuation and portfolio management". Any specific claim about which colours Frayn's agents preferred, or about jail and trading, should not be stated until someone reads the PDF.
- Eithun et al. (2017) explicitly position themselves as a simplified Frayn-style model, and they publish exact numbers. They are the best verified evolutionary/search-based source (see Section 3).

### Gaps
- Frayn's evolved property valuations, his treatment of trading, auctions and jail, the GA parameters (population, generations, games per evaluation) and any win rates: the full text was unreachable. Popular write-ups sometimes attribute claims such as "stay in jail late" or "orange is best" to Frayn. I could not confirm any of these, so they should be verified before use.
- Mukhar (2013) GA study: cited by Eithun et al. but not located.

---

## 2. What did RL and AI-agent papers find (Bailis et al. 2014; Haliem/Bonjour et al. 2021-22; DARPA SAIL-ON/GNOME; 2022-2026 LLM agents)?

### Takeaway
Learned agents beat hand-coded fixed policies by large margins: Bailis 2014 won 69.4% of 3-way games, Bonjour/Haliem's hybrid PPO won 91.65% against three fixed-policy bots, and a SAIL-ON planning agent won 76.48% before any novelty. The interpretable "strategies" that come out of these papers are about cash discipline, completing monopolies and trade acceptance rules, not about a single favourite colour. A bonus for the video: in the Bonjour/Haliem tables, the fixed-policy bot that prioritises orange/light blue plus railroads (FP-C) usually out-won the bot that prioritises Boardwalk/Park Place plus railroads (FP-B), which in turn usually out-won the equal-priority bot (FP-A).

### Cited Findings

**Bailis, Fachantidis & Vlahavas (2014), "Learning to play Monopoly: A Reinforcement Learning approach", AISB 50, London**
- Method: Monopoly modelled as a single-agent MDP. Q(λ)-learning with a 3-layer back-propagation neural network as function approximator. State = a 10×2 "area" matrix (holdings per colour group for agent vs opponents), board position, and a financial-status variable. Actions per colour group: Spend / Sell / Do nothing. — [Bailis et al. 2014 PDF](https://www.doc.gold.ac.uk/aisb50/AISB50-S02/AISB50-S2-Bailis-paper.pdf)
- Rules: auctions ("bidding contest") implemented "following the original rules"; "The special get out jail cards have been removed from the game". There is no trading action. Bonjour et al. confirm that Bailis "consider[s] a limited set of actions (buy, sell, do nothing)" with no trades. — [Bailis et al. 2014](https://www.doc.gold.ac.uk/aisb50/AISB50-S02/AISB50-S2-Bailis-paper.pdf); [Bonjour et al.](https://arxiv.org/pdf/2103.00683)
- Fixed-policy baseline: "He sells if he has less than 150, buys when he has more than 350 and does nothing between" (thresholds tuned for best performance). Fixed policy vs random: "97% to 3%" in 1000 games. — [Bailis et al. 2014](https://www.doc.gold.ac.uk/aisb50/AISB50-S02/AISB50-S2-Bailis-paper.pdf)
- RL training vs random: averaged over 25 trials, the RL agent reaches a "100% win ratio" after 1000 episodes. Hyperparameters: a = 0.2, γ = 0.95, NN learning rate α = 0.2, λ = 0.85. — [Bailis et al. 2014](https://www.doc.gold.ac.uk/aisb50/AISB50-S02/AISB50-S2-Bailis-paper.pdf)
- Final 1000-game test with Random, Fixed Policy and RL agent (Table 1; the table implies all three at one table): RL agent 694 wins (69.4%), Fixed Policy 286 (28.6%), Random 20 (2%). The agent "demonstrated several times an intelligent behaviour of sacrificing some of his temporary wealth in order to invest and secure a more prosperous future." Code: github.com/pmpailis/rlmonopoly. — [Bailis et al. 2014](https://www.doc.gold.ac.uk/aisb50/AISB50-S02/AISB50-S2-Bailis-paper.pdf)

**Haliem, Bonjour et al. (arXiv 2103.00683; v1 March 2021 "Learning Monopoly Gameplay: A Hybrid Model-Free Deep RL and Imitation Learning Approach"; later version "Decision Making in Monopoly using a Hybrid Deep Reinforcement Learning Approach", IEEE Trans. Emerging Topics in Computational Intelligence, 2022)**
- v1 (Haliem first author, 1 Mar 2021): the agent first imitates a rule-based agent designed to resemble human logic, then refines its policy with DRL. It reports "high win rates against different types of agent-players" but gives no number in the abstract. — [arXiv v1 abstract](https://export.arxiv.org/abs/2103.00683v1)
- Published version (Bonjour & Haliem equal first authors; with Alsalem, Thomas, Li, Aggarwal, Kejriwal, Bhargava; Purdue / USC-ISI): headline "hybrid PPO agent performs the best with a win rate of 91% against fixed-policy agents" and hybrid agents "outperform standard agents by 20% in the number of games won". — [arXiv PDF](https://arxiv.org/pdf/2103.00683); [IEEE Xplore](https://ieeexplore.ieee.org/document/9775710)
- Simulator and rules: an open-source four-player Python simulator (GNOME-p3, github.com/mayankkejriwal/GNOME-p3) with rules "similar to the US version of the game, barring some modifications". Doubles rules are not modelled ("a double can get a player out of jail... We treat them as any other dice roll"). Trading is allowed, but only of unimproved, unmortgaged properties. Offers can go to several players at once, and each player can have only one outstanding offer. Play is split into pre-roll, out-of-turn and post-roll phases. The state is 240-dimensional and the action space 2,922-dimensional. — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- Fixed-policy baselines, "based on successful tournament-level strategies adopted by human players" (the paper cites amnesta.net/monopoly and a Vice "10 essential tips from a Monopoly world champion" article):
  - All three "place the highest priority in acquiring a monopoly", trade (one-way and two-way, with or without cash), offer properties that are low-value to themselves but give the other player a monopoly, and build houses and hotels on their monopolies.
  - **FP-A** gives equal priority to all properties.
  - **FP-B** gives high priority to the four railroads and to Park Place and Boardwalk, and low priority to utilities.
  - **FP-C** gives high priority to the railroads and to the orange or sky-blue groups.
  - — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- Hybrid agent rules for rare decisions:
  - Buying: buy the property "if it results in a monopoly as long as it can afford it"; otherwise buy only "if the agent has $200 more than the property price".
  - Accepting a trade: accept if the trade "increases the number of monopolies". If the number of monopolies is unchanged, accept only if the offer's net worth nwo = (po + co) − (pr + cr) is positive.
  - — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- Evaluation: 5 runs × 2,000 games, with turn order randomised. Win rates:
  - Std PPO table: FP-A 7.69%, FP-B 8.79%, FP-C 13.57%, Std PPO 69.95%.
  - Hybrid PPO table: FP-A 2.07%, FP-B 2.88%, FP-C 3.40%, Hybrid PPO 91.65%.
  - Std DDQN table: FP-A 14.96%, FP-B 16.01%, FP-C 21.62%, Std DDQN 47.41%.
  - Hybrid DDQN table: FP-A 7.46%, FP-B 8.71%, FP-C 6.92%, Hybrid DDQN 76.91%.
  - — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- Mixed tables:
  - FP-B 2.78%, FP-C 2.53%, Std PPO 5.97%, Hybrid PPO 88.72%.
  - FP-B 14.59%, FP-C 14.68%, Std DDQN 13.87%, Hybrid DDQN 56.86%.
  - All four learners together: Std DDQN 0.9%, Hybrid DDQN 28.56%, Std PPO 1.48%, Hybrid PPO 69.06%.
  - Appendix tables: FP-A 2.93% / FP-B 3.64% / Std PPO 6.79% / Hybrid PPO 86.64%; FP-A 2.15% / FP-C 2.9% / Std PPO 6.24% / Hybrid PPO 88.71%; FP-A 12.01% / FP-B 15.92% / Std DDQN 13.52% / Hybrid DDQN 58.55%.
  - — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- The paper states: "A complete academic study on which strategies yield the highest probabilities of winning has been lacking." It cites Arun et al. (2019) as supporting "the idea that no single strategy can maintain high win rates against all other strategies." — [arXiv PDF](https://arxiv.org/pdf/2103.00683)
- Version discrepancy in headline claims: one arXiv version says hybrid agents improve on standard DRL by about 30%, while an earlier contribution list says "at least 25%" and "more than 20%". **[snippet only for the 30% figure]** — [arXiv listing](https://export.arxiv.org/abs/2103.00683)

**DARPA SAIL-ON / GNOME (Kejriwal et al.)**
- GNOME is a four-player Monopoly simulator built to test how AI agents cope with "novelty", that is, mid-tournament rule changes. Its baseline agents:
  - **Simple**: "desires neither to trade nor improve properties", and declares bankruptcy rather than mortgaging.
  - **H1**: builds on monopolies, trades, and mortgages or sells when short of cash.
  - **H2**: H1 plus more sophisticated trading, including two-way offers to multiple players at once.
  - **ML**: a hybrid agent that uses RL only for some decisions, such as whether to buy.
  - — [Kejriwal et al., arXiv 2507.03802](https://arxiv.org/html/2507.03802v1)
- About 45-50 novelties were used. Dice and "swap and extend" novelties "terminate within 200-500 board round-trips per player". With the Property Colour Change novelty, "the game tends to go on forever". The authors also note that "the average performance of H2 can decline when more than one instance is playing". No baseline win-rate table is published. — [arXiv 2507.03802](https://arxiv.org/html/2507.03802v1)
- Example novelties include a wealth tax and rent control. **[snippet only]** — [World Economic Forum / The Conversation](https://www.weforum.org/stories/emerging-technologies/reinforcement-learning-how-board-games-could-help-ai-handle-the-unexpected/)
- Gopalakrishnan, Soni, Thai, Lymperopoulos, Scheutz & Kambhampati (ASU/Tufts, 2021), an independent SAIL-ON evaluation:
  - Each trial is 100 games against 3 baseline agents (the "simple baseline agent" of Haliem et al.).
  - Their planning agent's pre-novelty win rate was **76.48%**, versus **63.61%** for the next-best team.
  - Two instances of their agent plus one baseline won **40.75%** and **39.43%** of games.
  - The agent's value function has a monopoly term that lets it "eschew buying new or bidding for properties if that amount can be used to complete and develop a monopoly". Its bankruptcy guards keep enough cash to survive the worst possible rent next round.
  - — [arXiv 2107.04303](https://arxiv.org/pdf/2107.04303)
- The same paper reports an earlier MCTS Monopoly agent (Sammul 2018, University of Edinburgh BSc thesis) reaching a 63% win rate, with significant simplifications: no out-of-turn actions, and access to the best adversary model. — [arXiv 2107.04303](https://arxiv.org/pdf/2107.04303)

**LLM agents (2022-2026)**
- AgentMonopoly (Wu, Li, Marsault, Huang & Cheng; arXiv 2610.29330v2, 22 Oct 2026). — [arXiv 2610.29330](https://arxiv.org/html/2610.29330v2)
  - Setup: 4 LLM agents per game on a 40-square board with Oldenburg street names. Trading, auctions, building and mortgages are all included, and owners may waive rent. Games stop at 150 steps, and the highest net worth then wins. 140 games in total (series of 40/40/40/20, seeds 42-46). Models are open-weight: Qwen3, DeepSeek-R1, Gemma4 and Mistral at two sizes each.
  - Cross-family wins out of 20 games: Gemma4-26B 14, Qwen3-32B 3, DeepSeek-R1-32B 2, Mistral-Small-24B 1. In the small-model series: Gemma4-12B 11, DeepSeek-R1-8B 6, Mistral-Nemo-12B 3, Qwen3-8B 0.
  - Mixed-size games: larger models won 18 of 40, smaller models 22.
  - Trade proposals per 100 rolls in same-model games: Qwen3-32B 110 (58% accepted); Gemma4-26B 33 (17% accepted); Gemma4-12B 1.5 (55% accepted). Rent-waiving rates ranged from 0% (Qwen3-8B) to 59% (DeepSeek-R1-8B).
  - A "Conservative" persona kept an average minimum cash of $688 versus $63 for opponents.
  - No rule-based or random baselines were run.
- I found no other peer-reviewed LLM-plays-Monopoly benchmark. General LLM game benchmarks (SmartPlay, LLMArena, Board Game Arena) did not show Monopoly in their abstracts. — [search results incl. SmartPlay ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/063d1fadbde557466bf5bcffe814a64e-Abstract-Conference.html)

### Inferences
- The fixed-policy colour preferences in Bonjour/Haliem form a ready-made mini-experiment for the video: "railroads + orange/light blue" (FP-C) vs "railroads + Boardwalk/Park Place" (FP-B) vs "everything equal" (FP-A). In 3 of the 4 main four-player tables, FP-C had the highest win share among the fixed bots, and FP-B beat FP-A in every table that included both. These are leftover shares in games dominated by a learning agent, not a designed head-to-head, so the gaps (often 1-6 points) are suggestive rather than conclusive.
- The rules the learned agents converge on all reward patience: Bailis's agent sacrificed short-term wealth to invest, the hybrid buys only with $200 to spare unless the property completes a monopoly, and the SAIL-ON agent skips purchases to fund a monopoly. This conflicts somewhat with the "All In / keep no cash" result from hobbyist sims (Section 3). The difference is likely about when cash is spent: hoarding cash idly loses, while holding enough to survive and then building aggressively wins.
- In the AgentMonopoly study, the most successful model (Gemma4-26B) traded relatively rarely and selectively (33 proposals per 100 rolls, 17% accepted) and built heavily after completing groups. With 20-game samples this is anecdotal.

### Gaps
- The exact player count for Bailis's Table 1 is not stated explicitly. A 3-player table is my inference from the three agents listed.
- Arun et al. (2019, "Monopoly using reinforcement learning") was not retrieved, so its win rates are unknown.
- SAIL-ON per-team results beyond the Gopalakrishnan table are not public.
- None of the RL papers publish the learned agents' property preferences (for example a colour heat map) or their jail behaviour. Their reported results are win rates only.
- The full-paper results of the v1 imitation-learning variant were not separately checked, beyond its being an earlier version of the same arXiv ID.

---

## 3. What have hobbyist/blog/GitHub Monte Carlo simulators found when pitting simple strategies against each other?

### Takeaway
Three large, recent, well-specified simulators broadly agree on the following:
- Buying broadly beats being picky.
- Idle cash reserves cost wins.
- Refusing to trade is a disaster at 3+ players, but nearly harmless heads-up.
- Building past 3 houses to hotels beats stopping at 3, even though the 3rd house has the best marginal return.
- Orange is the best colour group to fight for, while railroads-only or utilities-only play loses badly at four players.

These are the Perfect Play YouTube study (about 313M games), oliverhutch2021/monopoly-strategy, and gamesdecoded/monopoly-simulator (200M games). The one academic simulation, Eithun et al. 2017 (two-player), ranks railroads first, then orange and light blue. Railroads appear stronger heads-up than at a four-player table.

### Cited Findings

**Perfect Play, "The Best Monopoly Strategy, Tested on 300 Million Games" (YouTube, Oct 2026)**
- Creator: the Perfect Play channel; the presenter calls himself "Leo". Video: https://www.youtube.com/watch?v=S_2eVAaw1UM. "eight classic Monopoly strategies into computer players... played 312,898,560 games against each other". At a table of four, the best won 43% and the worst 3%. The gap came from "being picky about what you buy, saving your cash, stopping at three houses, and refusing to trade." — [kottke.org, 20 Oct 2026](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games)
- Rules, as summarised from the transcript: four players; official rules; 32 houses; declined properties go to auction; trading "limited to swaps that complete a color set for both sides"; Free Parking not mentioned. **[secondhand: transcript summary]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
- Eight-bot win rates at four-player tables:

  | Bot | Win rate |
  |---|---|
  | All In | 43% |
  | Buy Everything | 29% |
  | Three Houses | 17% |
  | Saver | 16% |
  | Never Trades | 14% |
  | Expensive Side (red to dark blue only) | 9% |
  | Cheap Side (brown to orange only) | 7% |
  | Railroad Baron (railroads + utilities only, bids double at auction) | 3% |

  These are per-bot win rates across mixed tables, so they do not sum to 100. **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
- Bot definitions:
  - **All In** never keeps cash, "mortgages its spare properties to build faster, and late in the game it would rather stay in Jail".
  - **Saver** keeps $300 after buying and $500 before building.
  - **Three Houses** "builds three houses on each property, and then it stops".
  - **Buy Everything** is the "normal player" baseline.
  - **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
- Baseline: a normal player wins about 21% at a four-player table, because "roughly one game in six never ends. So 21 is the number to beat." **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/); [kottke.org](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games)
- Single-habit tests (about 1M games each; colour pairs 200,000 games per pair). **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
  - Never Trades won 6.5% vs 21% for a normal player.
  - Skipping brown cost 5 points; skipping Boardwalk/Park Place cost 7 points; skipping utilities "did no harm". No single change gained more than half a point.
  - A player buying only orange and red won under 3%.
  - In single-colour tests, green and dark blue "finished near the bottom", and only brown did worse.
  - Orange with 3 houses pays back its cost after 29 opponent rolls.
  - Rent jumps by $56, then $130, then $350 per building step, so stopping at three houses gives up the biggest jump.
  - Smaller cash cushions won 13%.
- A computer-tuned player won 26% against three normal players (19% each). It "buys everything", "builds to three houses first, and then all the way to hotels", bids above the printed price, and "will go up to three times the price for a property that completes a set". The closing advice is "if you get to choose, fight for orange". **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
- Methodology caveat: the transcript summary also describes "70 tables of four with 120,000 games each" (about 8.4M), which does not match the 312.9M headline. The two figures probably describe different experiments, but I could not confirm this. Moneywise reports "more than 312 million". **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/); [Moneywise via search](https://moneywise.com/life/entertainment/how-to-win-at-monopoly)

**Eithun, Lamb & Young (Ripon College), "Long-Term vs. Short-Term Strategy in the Game of Monopoly", UMAP Journal 38.1 (2017)**
- Rules: **two-player**; Python model following the official 2008 Hasbro rules, "including trades and auctions". Trades involve no cash, are one-for-one, must give both players or neither a monopoly, never break a monopoly, and are searched only right after a purchase. Games still active after 1000 turns are ties (for example, 1% ties in Red vs Railroad). Code: github.com/RiponCollegeMaCS/Monopoly. — [Eithun et al. 2017 PDF](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- Strategy model: a ranking of the 10 groups plus a "maximum buying threshold" (MBT, the cash a player wants to keep after a purchase), scaled by rank. "static strategies do significantly better with a step-wise buying threshold than with a fixed buying threshold." — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- Hill-climbing over 4 × 10! = 14,515,200 strategies, scoring each in 10,000 games, ran about 18 hours and found 60 local maxima. The best won "just under 82%". The best ranking was **Railroad, Orange, Light Blue, Brown, Pink, Utility, Dark Blue, Red, Yellow, Green** with MBT $1500. — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- PageRank rankings:
  - Start of game: Railroad, Light Blue, Brown, Orange, Pink, Red, Utility, Dark Blue, Yellow, Green.
  - End of game (100,000 games): Railroad, Orange, Light Blue, Pink, Dark Blue, Red, Utility, Yellow, Green, Brown.
  - — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- "Success indicators" (win % over 100,000 games vs random static strategies): Hill Climb 79.6, PageRank Start 76.8, PageRank End 75.9, Frequency 65.8, Dynamic 61.0, Newbie 33.3, Expected Value 22.3. Newbie ranks the most expensive groups first with a fixed $500 threshold. "Expected Value" ranks Green, Yellow, Red, Dark Blue first. — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- Head-to-head (10,000 games per pair): Hill Climb beat Newbie 97% and the Dynamic (short-term expected-value) strategy 70%. Newbie won "just 3.1%" against Hill Climb. Strategies are non-transitive: B beats A 58%, A beats C 55%, C beats B 56%. — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- Conclusion: "focusing on acquiring the Railroads above all, followed by the Orange and Light Blue groups, and deprioritizing the Red, Yellow and Green groups". A composite player showed "no advantage to allowing the expectation of short-term gains to overrule long-term strategy". — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)

**oliverhutch2021/monopoly-strategy (GitHub; UK board and £; 2-6 players; evolved agents plus counter-strategy search; paired dice)**
- **Orange**: payback with hotels of 22.1 / 11.0 / 7.4 / 5.5 / 4.4 rounds at 2/3/4/5/6 players, versus 35.0 / 17.5 / 11.7 / 8.8 / 7.0 for green. "Orange pays back fastest" at every table size. On every set except brown, "the third house earns the best return of any building level". — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Last piece of a set**: in a 4-player test, owning two oranges gave a 6% win rate without the third and 90% with it. The piece is worth up to £1,144, almost six times its £200 printed price. — [README](https://github.com/oliverhutch2021/monopoly-strategy)
- **Building**: a hard three-house cap costs 0.79× against the final agent (official rules, 4 players, 20,000 games). Stopping at four houses costs 3% of wins at 2 players and 9-13% at 4-6 players. — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **House shortage**:
  - A "housing lock" (stopping at four houses while the bank is short) adds 5% wins at 4 players (1.049×, 40,000 games). Its gain comes from the cash-raising rule, not from holding back hotels, which alone gains nothing (0.99-1.00×).
  - Shortages occurred in 43% of 4-player games under championship rules versus 5% under official rules. House auctions moved no edge by more than 1%.
  - — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Cash**:
  - Keeping £200 more cash costs 10-11% of wins at 2-4 players, 15% at 5 and 19% at 6.
  - A Cautious bot (£400 buffer, 80% auction limit) scored 0.86× at 2 players falling to 0.63× at 6. An Aggressive bot (no buffer, 120% auction limit) scored 1.01× rising to 1.35×.
  - — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Jail**: players spend 11% of turns in Jail when they roll for doubles and 4.7% when they pay immediately. Always-leave versus always-stay costs at most 9% of wins, and staying is 3% better for the official 2-player agent. Flipping the "£10 fine or take a Chance" choice changes results by at most 1.4%. — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Auctions**:
  - The final agents' bid limits are 1.64 / 0.73 / 1.21 / 1.26 / 1.73 × face value at 2-6 players under official rules.
  - Bidding 25% more beats the official agents at 3, 5 and 6 players by 3-5%. For the 2-player official agent, bidding more is a mistake (0.82×).
  - — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Robustness**: no unbeatable strategy was found. Each of the strongest agents can be beaten by a counter-strategy scoring 3% to 51% more against it. — [README](https://github.com/oliverhutch2021/monopoly-strategy)

**gamescomputersplay/monopoly (GitHub, Python simulator)**
- Default bots buy everything they land on, build at the first opportunity, and leave jail on doubles. Trading is 1-on-1 and aimed at completing monopolies. Auctions are not implemented, and the Free Parking pot is optional. Default 1,000 games and a 1,000-turn cap. — [README](https://github.com/gamescomputersplay/monopoly)
- Results (survival rates, not wins): ignoring the Indigo (brown) properties lowers survival by about 10-12%; a $0 unspendable cash threshold beats $500 by about 15-20%. Free Parking "Not really" affects survival time, and a lower salary "very much" reduces draw games. No game counts are given per result. — [README](https://github.com/gamescomputersplay/monopoly)

**b2studios (YouTube, about 2022): NEAT neuroevolution**
- An evolved neural-network agent (NEAT, knockout tournaments). Code: github.com/b2developer/MonopolyNEAT. The README says the generation-162 brains are "quite good at monopoly (as far as I can tell)", with no data. — [MonopolyNEAT README](https://github.com/b2developer/MonopolyNEAT)
- The AI played 11.2 million games over two weeks. A heat map of its favourite and most-winning properties ranked railroads 1, orange 2, red 3 and purple 4. **[secondhand: university course blog]** — [UGA MIST 5720 blog](https://ctlsites.uga.edu/mist5720/?p=1522)

**Weaker / unverified hobbyist claims**
- TDS "A Data-Driven Tactics Simulation for Monopoly": being riskier with your buying strategy than your opponents increases expected average wealth. **[snippet only; page 403]** — [Towards Data Science](https://towardsdatascience.com/a-data-driven-tactics-simulation-for-monopoly-864e7cffe508/)
- TDS/Medium MATLAB study (1,000 games): no correlation between buying strategy and success rate in one analysis; "you can't win with just utilities or railroads". **[snippet only; page 403]** — [Medium/TDS](https://medium.com/data-science/simulating-monopoly-finding-the-best-properties-using-matlab-130fe557b1ae)
- Tomiwa-Adewumi/Monopoly_Strategy_Simulator (simplified Java rules): "Cautious saver" won 384 trials, "Disruptive" 264, "Aggressive buyer" 220, "Collector (green)" 132. Total trials and rules are unclear, and this conflicts with the larger sims on cash. **[snippet only]** — [GitHub](https://github.com/Tomiwa-Adewumi/Monopoly_Strategy_Simulator)
- wedg.github.io probability write-up: exit jail immediately early in the game, and stay in jail later in the game. This is a Markov/probability argument, not a head-to-head simulation. **[snippet only]** — [Simulating Monopoly (wedg)](https://wedg.github.io/monopoly/Monopoly.html)

### Inferences
Mapping the video's named strategies to the evidence:
- **"Buy only railroads"**: worst bot at four players (Railroad Baron 3%, with utilities included, Perfect Play). Yet railroads rank #1 in the two-player Eithun study and in b2studios' heat map. The best framing is "railroads are a great part of a portfolio, a terrible whole strategy", and they are relatively stronger heads-up.
- **"Wait for Boardwalk/Park Place" / expensive side**: weak. Expensive Side won 9%, and green and dark blue were near the bottom of the single-colour tests. Skipping dark blue entirely, however, cost 7 points. Dark blue ranks only 7th in Eithun's hill-climb result.
- **"Buy everything you land on"**: a strong baseline (29%; the "normal player"). It is beaten only by buying everything and also building aggressively (All In 43%; the tuned player 26% vs 19%).
- **"Orange only"**: no source isolates orange-only. Orange is consistently the best colour to own (fastest payback, top-2 or top-3 in every ranking), but a player restricted to orange + red won under 3%. Single-colour fixation fails because you cannot control where you land.
- **"Cash hoarder"**: loses (Saver 16%; keeping £200 extra costs 10-19% of wins). The academic agents' modest cushions (Eithun MBT $1500 scaled by rank; Bonjour's $200 margin) suggest cash should be held for survival, not hoarded.
- **"Three houses and stop"**: loses (17%; cap = 0.79×), even though the 3rd house is the best marginal buy. The right message is "rush to three houses, then keep going".
- **Never trade**: devastating at 4 players (0.20× to 0.31× of normal) but harmless at 2 players.

### Gaps
- No source isolated a pure "orange only" bot, a pure "utilities only" bot, or a pure "dark blue only" bot with published numbers. Perfect Play's single-colour test results are only partially transcribed.
- Perfect Play's code and methods are not published in text form, so its rule details (Free Parking, jail fine policy, how unfinished games are scored) are unknown.
- oliverhutch2021 uses the UK board and £ prices. Property positions and prices match the US board, but the rules list and agent-evolution details were not fully read (RULES.md not fetched).
- I found no r/dataisbeautiful or Kaggle Monopoly strategy-tournament post with retrievable numbers.

---

## 4. How long do simulated games last, how often do they never end, how much does going first matter, and how do house rules change outcomes?

### Takeaway
- Bot-vs-bot games are short: medians of roughly 136-155 player-turns at four players (gamesdecoded), or 35-63 rounds (oliverhutch2021).
- The share of games that never end depends heavily on the bots and their trading rules: about 1 in 6,652 in one sim, about 1 in 6 in Perfect Play, and about 70% with no trading at all (b2studios, secondhand).
- Going first is a real but modest edge at four players: seat 1 wins about 27% vs about 23% for seat 4, or 1.09× an equal share. The edge grows to about 1.2× at six players.
- The Free Parking jackpot slightly shortens the median game but makes runaway endless games about 5× more common. Removing auctions slightly lengthens games.

### Cited Findings
- **gamesdecoded/monopoly-simulator**: 40M games per variant (200M total), four identical heuristic bots, official rules otherwise. "Never-ending" means more than 1,500 turns. — [gamesdecoded README](https://github.com/gamesdecoded/monopoly-simulator)

  | Variant | Median game length | Never-ending rate | Seat 1/2/3/4 win % |
  |---|---|---|---|
  | Control | 147 turns | 1 in 6,652 (about 0.015%) | 27.3 / 25.8 / 24.2 / 22.6 |
  | Free Parking jackpot | 140 turns | 1 in 1,247 (about 0.080%) | 27.6 / 25.9 / 24.2 / 22.4 |
  | No auctions | 155 turns | 1 in 6,641 | 27.3 / 25.7 / 24.2 / 22.8 |
  | Double GO ($400) | 142 turns | 1 in 2,090 | 27.6 / 25.9 / 24.1 / 22.3 |
  | Snake Eyes ($500) | 136 turns | 1 in 805 | 27.8 / 26.0 / 24.0 / 22.2 |

- gamesdecoded bot logic and rule settings:
  - Buying: buy if cash after the purchase stays at or above $150.
  - Auctions: bid up to face value for set-completing properties, railroads or a second utility; otherwise up to 70-80% of face.
  - Building: build when cash exceeds cost + $200, and build hotels after four houses.
  - Trading: only a face-value cash offer for the missing piece of a set, which is auto-accepted.
  - Jail: pay $50 if cash exceeds $300, otherwise roll for doubles.
  - The Free Parking pot collects all taxes and fines and has no seed money.
  - The hard cap is 3,000 turns, after which the richest survivor wins.
  - — [gamesdecoded METHODS.md](https://raw.githubusercontent.com/gamesdecoded/monopoly-simulator/main/METHODS.md)
- **oliverhutch2021**:
  - Median length is 63 rounds at 2 players and 35 at 6 (126 and 212 total turns). The Free Parking jackpot "Shortens games at every size, most at 2 players (56.5 rounds)".
  - Seat order with identical bots (100,000 games per size): the first player gets 1.00× at 2 players (not significant), 1.09× at 4 and 1.20× at 6; the last player gets 0.81× at 6.
  - Meta-agent games stop at 150 rounds.
  - — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **Perfect Play**: "roughly one game in six never ends" at the four-player table, which drives the 21% normal-player baseline. With trading off, "more than half of games ended with no set completed". **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/); [kottke.org](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games)
- **b2studios**: a basic bot that buys everything, never trades and builds when it can produced stalemates in about 70% of four-player games. **[snippet only; blog not reachable]** — [learn.fomo.ai](https://learn.fomo.ai/p/what-happened-when-an-ai-played-11)
- **Eithun et al.**: two-player games over 1000 turns were scored as ties, about 1% in the Red vs Railroad example. The paper cites Friedman et al. (2009) for estimating "the probability that the game will never end" with Markov chains, Brownian motion and simulation. — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- **GNOME**: game length under novelties is "within 200-500 board round-trips per player", and some novelties make games "go on forever". — [arXiv 2507.03802](https://arxiv.org/html/2507.03802v1)
- **Ian Stewart**, Scientific American, "Mathematical Recreations", April 1996: the first player is most likely to land on the 7th square (Chance), or else on Oriental or Vermont, so "the first player has an excellent chance of securing one of these properties". He concludes that "the first player still has a small advantage, but this bonus is mitigated by the finiteness of his or her bank balance." — [Stewart 1996 (Harvard-hosted PDF)](https://legacy-www.math.harvard.edu/archive/21b_fall_05/Articles/howfairismonopoly.pdf)
- **SumProduct Excel/VBA simulation** (tens of thousands of games): in four-player games the first player won about 50% more often than the last. Advice: "convince people to let you roll first". **[snippet only; page 403]** — [SumProduct, Excel at Monopoly #5](https://sumproduct.com/blog/excel-at-monopoly-5-vba-simulation-and-review/)
- **Learning-agent papers neutralise seat order**: Bonjour et al. "randomize the turn order to remove any advantage one may get due to the player's position". — [arXiv 2103.00683](https://arxiv.org/pdf/2103.00683)

### Inferences
- First-player advantage at four players: gamesdecoded shows 27.3% vs 22.6%, so seat 1 wins about 1.21× as often as seat 4. That is a much smaller gap than SumProduct's reported "about 50% more often than last", which is unverified and from a smaller sim. The two larger, reproducible sims agree that seat 1 gets about +9% over a fair 25% share: gamesdecoded 27.3/25 ≈ 1.09×, and oliverhutch2021 1.09×. The video should randomise or rotate seats when comparing strategies.
- The Free Parking jackpot gives a counter-intuitive pair of results. The median game gets slightly shorter (140 vs 147 turns; and shorter at every table size in oliverhutch2021), but the long tail of endless games grows about 5× (1 in 1,247 vs 1 in 6,652). This partly contradicts the common claim that Free Parking "makes the game last forever". The pot also recirculates cash, making a stalled game harder to finish.
- The size of the "never ends" problem depends mostly on how freely bots trade: 70% with no trading, about 1 in 6 with limited mutual-set swaps, and about 1 in 6,652 with auto-accepted face-value buyouts.

### Gaps
- No source quantified the effect of the Free Parking jackpot or no-auction rule on which strategies win, only on game length and seat share.
- The SumProduct seat result and its rules could not be read directly. The SumProduct "#6 House Rules" post was also 403.
- I found no human-game dataset of real game lengths to compare with the simulations.

---

## 5. What is known about trading strategy in simulations (e.g., trading to complete monopolies, valuation models)?

### Takeaway
Every serious simulator finds that trades which complete monopolies are the engine of winning at 3+ players. The set-completing card is worth several times its printed price, and refusing all trades cuts a strong player's wins to about a fifth at four players. Academic agents use simple, explainable trade rules (accept if it adds a monopoly, otherwise only if net worth is positive; both-or-neither monopoly swaps). Learned and LLM agents show that trading volume alone does not predict success.

### Cited Findings
- **Bonjour/Haliem hybrid agents**:
  - Accept a trade if it "increases the number of monopolies". Otherwise accept only if net worth nwo = (po + co) − (pr + cr) is positive (property price plus cash offered, minus property price plus cash requested).
  - The fixed-policy bots offer "properties that hold a low value (for example, a solitary property) to the agent itself but may be of value (gives the other player a monopoly) to the other player", and roll out offers to multiple players simultaneously "to increase the probability of a successful trade".
  - Only unimproved, unmortgaged properties can be traded.
  - — [arXiv 2103.00683](https://arxiv.org/pdf/2103.00683)
- **Eithun et al.**:
  - Trade principles: no cash; one-for-one; "Any trade either gives both players a monopoly or neither player a monopoly"; both sides must think they gain according to their rankings; complete monopolies are never broken.
  - The dynamic strategy evaluates trades by expected rent change over the next six rolls (Markov chain, long jail stay assumed). Its "effective ranking" in trades was Railroad, Dark Blue, Utility, Green, Red, Yellow, Orange, Light Blue, Pink, Brown. Hill Climb beat it 70% head-to-head.
  - Short-term trade value should not override long-term group ranking.
  - — [Eithun et al. 2017](https://mitchelleithun.com/wp-content/uploads/2020/07/monopoly.pdf)
- **oliverhutch2021**:
  - Never trading scores 1.12× at 2 players (simple bots, 100,000 games) because "trades are zero-sum there", and 0.23× at 6. With the final agents: 0.98× at 2, 0.34× at 3, 0.20× at 4, 0.14× at 5, 0.11× at 6 (official rules, 40,000 games).
  - "Deal zones" (prices both sides should accept) for the set-completing Vine Street (an orange) against simple bots: none at 2 players, £662-£890 at 3, £594-£1,144 at 4 (20,000 rollouts per price).
  - Under championship rules at 5-6 players, the final agents refuse all trades, because accepting would cost them 19% and 29%.
  - — [FINDINGS.md](https://raw.githubusercontent.com/oliverhutch2021/monopoly-strategy/main/FINDINGS.md)
- **oliverhutch2021, the third orange**: two oranges give a 6% win rate without the third and 90% with it, so the card is worth up to £1,144 versus its £200 price. "Computer players usually sell it within a couple of turns for £400, so paying much more than that is a loss." — [README](https://github.com/oliverhutch2021/monopoly-strategy)
- **Perfect Play**:
  - Trading is limited to swaps that complete a set for both sides.
  - Never Trades won 6.5% vs 21% for a normal player.
  - With trading off, more than half of games ended with no set completed.
  - The Buy Everything bot "always takes the trade".
  - The tuned bot bids "up to three times the price for a property that completes a set", whether the set is its own or an opponent's (blocking).
  - **[secondhand]** — [SozAI transcript](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/)
- **gamesdecoded**: the only trade is a face-value cash offer for the single missing set piece, auto-accepted. This minimal rule alone keeps endless games to about 1 in 6,652. — [METHODS.md](https://raw.githubusercontent.com/gamesdecoded/monopoly-simulator/main/METHODS.md); [README](https://github.com/gamesdecoded/monopoly-simulator)
- **gamescomputersplay**: trades are 1-on-1 and aimed at completing a monopoly, with compensation owed for price differences; "too-unequal" trades are refused. — [README](https://github.com/gamescomputersplay/monopoly)
- **GNOME H2 baseline**: adds "two-way trade offers and offers to multiple players at once" on top of H1. H2's average performance "can decline when more than one instance is playing". — [arXiv 2507.03802](https://arxiv.org/html/2507.03802v1)
- **SAIL-ON planning agent** (Gopalakrishnan et al.): only its trade-related actions are rule-based. Its other decisions use a state-value function that includes a monopoly-completion gain. — [arXiv 2107.04303](https://arxiv.org/pdf/2107.04303)
- **AgentMonopoly LLM agents**: trade proposals ranged from 1.5 to 110 per 100 rolls, and acceptance rates from 17% to 86%. The top cross-family winner, Gemma4-26B (14/20 wins), proposed 33 trades per 100 rolls with 17% acceptance. — [arXiv 2610.29330](https://arxiv.org/html/2610.29330v2)
- **Mega Edition GA study**: "no significant advantage in bidding more than 1.5 times the face value of the property". **[snippet only]** — [ResearchGate](https://www.researchgate.net/publication/315800283_Optimizing_Strategies_for_Monopoly_The_Mega_Edition_Using_Genetic_Algorithms_Simulations)

### Inferences
- Trading is the single biggest strategic lever at a four-player table. Two independent sims give near-identical estimates: Never Trades at about 0.20× to 0.31× of a normal player's win rate (oliverhutch2021 0.20×; Perfect Play 6.5/21 ≈ 0.31×).
- The trade decision that matters most is whether to give an opponent a set. Accept only if you get a set back, or if you are paid well above face value: a set-completing card is worth up to about 3-6× its printed price, per Perfect Play's tuned bidder and oliverhutch2021's deal zones.
- For a video tournament, the trading rule must be fixed and disclosed, because it changes both who wins and whether games end.

### Gaps
- No source compared different trade-valuation models (for example net-worth rule vs expected-rent rule vs learned policy) head-to-head under identical conditions.
- No RL paper reported how often its learned agent proposed or accepted trades, or which colour groups it traded for.
- Frayn's (2005) treatment of trading is unknown (full text not retrieved).
