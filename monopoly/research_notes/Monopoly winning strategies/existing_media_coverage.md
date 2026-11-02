# Existing YouTube and Pop-Media Coverage of Monopoly Math and Strategy (as of 2 Nov 2026)

How the view counts were gathered: YouTube pages could not be fetched directly (they returned a Google "sorry" bot-check page). View and like counts therefore come from the Return YouTube Dislike (RYD) public API, and titles and channels come from YouTube's oEmbed endpoint. Both were queried on 2026-11-02. Keep these limits in mind:
- RYD counts are cached and may lag the live YouTube count. Treat every count as approximate (+/- a few percent, and more for fast-growing videos).
- RYD "likes" come from YouTube. "Dislikes" are an estimate built from extension users.
- RYD's `dateCreated` field records when RYD first logged a video, not when it was uploaded. For older videos it shows the 2022 database start date. For new 2026 videos it roughly matches the upload date, but it is not reliable (see the What's What note below).

## Q1. Which YouTube videos cover Monopoly math, strategy or simulations, and how many views do they have?

### Takeaway
Two kinds of video dominate this niche:
- **Simulation and AI videos.** b2studios' AI video has about 12.1M views, Stand-up Maths with Hannah Fry about 3.25M, and Perfect Play's Oct 2026 "300 Million Games" video about 2.67M in roughly three weeks.
- **Short "how to win every time" tip videos** from autumn 2026. What's What has about 6.9M plus 2.0M views, and Way Too Simple about 2.7M.

The "eight strategies fight across hundreds of millions of games" concept the creator is planning has already been made, and it went viral in Oct 2026 (Perfect Play). That is the most important competitive fact in these notes. I found no dedicated Monopoly-strategy videos from Vsauce, Veritasium, The Game Theorists, Tom Scott, Wendover or Jabrils.

### Cited Findings

**Videos with verified view counts (snapshot 2026-11-02, RYD API, approximate)**

| Title | Channel | ~Views | ~Likes | Date | URL |
|---|---|---|---|---|---|
| AI Learns Insane Monopoly Strategies | b2studios | 12,111,097 | 306,502 | c. Jan/Feb 2022 (Kottke covered it 2 Feb 2022) | https://www.youtube.com/watch?v=dkvFcYBznPI |
| HOW TO WIN MONOPOLY EVERY TIME | What's What (@whatswhat2022) | 6,934,615 | 209,507 | RYD first logged 11 Sep 2026; upload date unverified | https://www.youtube.com/watch?v=qu6GMpQfyUU |
| The Mathematics of Winning Monopoly (with Hannah Fry) | Stand-up Maths (Matt Parker) | 3,250,348 | 64,050 | 8 Dec 2016, about 19 min | https://www.youtube.com/watch?v=ubQXz5RBBtU |
| How to Win Monopoly Every Time | Way Too Simple (@waytoosimpleofficial) | 2,698,451 | 57,443 | RYD first logged 15 Sep 2026 | https://www.youtube.com/watch?v=CWfAlSn3u7A |
| The Best Monopoly Strategy, Tested on 300 Million Games | Perfect Play (@PerfectPlaySim) | 2,670,962 | 15,904 | RYD first logged 13 Oct 2026 | https://www.youtube.com/watch?v=S_2eVAaw1UM |
| ADVANCED VERSION - HOW TO WIN MONOPOLY EVERY TIME | What's What | 2,040,444 | 103,375 | RYD first logged 10 Sep 2026 | https://www.youtube.com/watch?v=s-r38R6jtgk |
| The Ultimate Monopoly Strategy Guide | Rempton Games | 221,847 | 4,576 | on or before Apr 2022 | https://www.youtube.com/watch?v=HJXwvJrbAqs |
| Testing Monopoly Tips with Python simulation. Should you really ignore greens? (and much more) | Games Computers Play | 186,466 | 7,685 | on or before Mar 2022 | https://www.youtube.com/watch?v=6EJrZeN0jNI |
| We Played 950 Million Games Of Monopoly... This is What We Found | Popperish | 10,399 | 74 | RYD first logged 23 Oct 2026 | https://www.youtube.com/watch?v=pTyqhhWw614 |
| HOW TO WIN MONOPOLY BOARD GAME EVERY TIME YOU PLAY | The Dice Treasures | 3,688 | 32 | RYD first logged May 2026 | https://www.youtube.com/watch?v=NTP4a1_DtFk |
| How to win Monopoly every time - the secret property percentages strategy | Spector Creative (features ex-Hasbro "toy guru" Scott Neitlich) | 3,569 | 204 | on or before Jan 2022 | https://www.youtube.com/watch?v=gaIhZHmc3jw |

Sources for the counts and channels (RYD API plus oEmbed, one per video):
[b2studios](https://returnyoutubedislikeapi.com/votes?videoId=dkvFcYBznPI), [What's What](https://returnyoutubedislikeapi.com/votes?videoId=qu6GMpQfyUU), [Stand-up Maths](https://returnyoutubedislikeapi.com/votes?videoId=ubQXz5RBBtU), [Way Too Simple](https://returnyoutubedislikeapi.com/votes?videoId=CWfAlSn3u7A), [Perfect Play](https://returnyoutubedislikeapi.com/votes?videoId=S_2eVAaw1UM), [What's What Advanced](https://returnyoutubedislikeapi.com/votes?videoId=s-r38R6jtgk), [Rempton](https://returnyoutubedislikeapi.com/votes?videoId=HJXwvJrbAqs), [Games Computers Play](https://returnyoutubedislikeapi.com/votes?videoId=6EJrZeN0jNI), [Popperish](https://returnyoutubedislikeapi.com/votes?videoId=pTyqhhWw614), [Dice Treasures](https://returnyoutubedislikeapi.com/votes?videoId=NTP4a1_DtFk), [Spector Creative](https://returnyoutubedislikeapi.com/votes?videoId=gaIhZHmc3jw).

Notes on the date and title sources:
- The Stand-up Maths release date (8 Dec 2016, about 19 min) and its description ("Hannah Fry visits Matt to compare their analyses") come from [IMDb Pro listing](https://pro.imdb.com/title/tt23028452) and [Mental Floss](https://mentalfloss.com/article/90001/watch-mathematics-winning-monopoly).
- The Spector Creative / Scott Neitlich attribution comes from search-result snippets only ([YouTube](https://www.youtube.com/watch?v=gaIhZHmc3jw)).

**Perfect Play, "The Best Monopoly Strategy, Tested on 300 Million Games" (Oct 2026). This is the closest direct competitor.**
- **Setup.** Eight classic strategies were made into bots and played 312,898,560 games under official rules: 32-house limit, auctions for unbought properties, mortgages, and both card decks. The tables were four-player: 70 table combinations at 120,000 games each. Dice were validated over 400M rolls against a 1996 Scientific American landing table (Illinois 3.19% vs 3.18%). — [SozAI transcript summary](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md); [YouTube search snippet](https://www.youtube.com/watch?v=S_2eVAaw1UM)
- **The eight strategies:**
  - Buy Everything
  - All In: never holds cash, mortgages to build, prefers to stay in Jail late in the game
  - Saver: keeps $300 after buying and $500 before building
  - Three Houses: stops at three
  - Cheap Side: brown to orange only
  - Expensive Side: red to dark blue only
  - Railroad Baron: railroads and utilities only, bids double at auction
  - Never Trades

  The only trade the bots know is a mutual set-completing swap. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- **Results.** All In won 43% at a four-player table and Railroad Baron won 3%. Never Trades won 6.5%, against 21% for a normal player in single-habit tests. When trading was switched off, more than half of games never completed a colour set. A final tuned bot (buy everything, three houses then hotels, aggressive auctions at up to 3x price for set-completing lots) won 26% against three normal bots at 19% each. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md); [Neatorama, 20 Oct 2026](https://www.neatorama.com/2026/10/20/The-Best-Monopoly-Strategy-as-Tested-on-300-Million-Games/)
- **Caveat.** The transcript's ranking numbers and its narrated percentages are internally inconsistent (for example, Three Houses appears as both "23" and "17%"). Exact figures should be checked against the video itself. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- **Coverage.** Kottke, Neatorama (20 Oct 2026), Hacker News (49 points, 45 comments) and Fark ("buy everything and stay in jail"). — [Kottke](https://kottke.org/26/10/the-best-monopoly-strategy-tested-on-300-million-games); [Neatorama](https://www.neatorama.com/2026/10/20/The-Best-Monopoly-Strategy-as-Tested-on-300-Million-Games/); [Hacker News](https://news.ycombinator.com/item?id=50175732); [Fark](https://www.fark.com/comments/14234526)
- **Criticism.** Hacker News commenters criticised the AI narration ("odd to use AI narration on such a labor-intensive video"), and one questioned how original the work was. — [Hacker News](https://news.ycombinator.com/item?id=50175732)

**b2studios, "AI Learns Insane Monopoly Strategies" (about 12.1M views, the most-viewed Monopoly-strategy video found)**
- **Method.** Neural networks were evolved with NEAT through self-play. Reports of the scale differ: one says 11.2M games, another says about 1M games against other bots. The baseline bot bought everything, never traded and built whenever it could. The code was open-sourced. — [mokslasplius summary](https://rf.mokslasplius.lt/b2studios-AI-learns-insane-monopoly-strategies/); [fomo.ai, via search snippet only; the site could not be fetched](https://learn.fomo.ai/p/what-happened-when-an-ai-played-11)
- **Reported findings.** Brown was an early favourite that was dropped once trading was available. The final favourites were orange, the railroads (especially the first) and red. A university class blog says the AI's most-won properties were railroads, then oranges, then reds and purples. — [fomo.ai snippet](https://learn.fomo.ai/p/what-happened-when-an-ai-played-11); [UGA MIST5720 blog](https://ctlsites.uga.edu/mist5720/?p=1522)
- **Kottke's description:** "game-playing AI bots are pitted against each other." — [Kottke, 2 Feb 2022](https://also.kottke.org/22/02/an-ai-discovers-the-best-strategy-for-monopoly)

**Stand-up Maths x Hannah Fry, "The Mathematics of Winning Monopoly" (2016, about 3.25M views)**
- Fry took a theoretical (Markov) approach and Parker a computational simulation, and the video compares their conclusions. — [Mental Floss](https://mentalfloss.com/article/90001/watch-mathematics-winning-monopoly); [Nerdist, 26 Dec 2016](https://nerdist.com/article/the-math-of-winning-monopoly/)
- The size of Parker's simulation is reported inconsistently: [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math) says 100 million dice rolls; [Nerdist](https://nerdist.com/article/the-math-of-winning-monopoly/) says one million rolls.
- A community GitHub gist reproduces the simulation in Python, with Chance cards and doubles. — [GitHub gist](https://gist.github.com/strategy/9b00ef3116dd3d09fafa09b4eaf82944)

**Other formats**
- **Popperish, "We Played 950 Million Games Of Monopoly..." (Oct 2026, about 10k views).** Its thesis, per search snippet: "the best strategy comes down to where tokens land, and Jail makes them land unevenly." It is a bigger-number follow-up to Perfect Play that did not take off. — [YouTube snippet](https://www.youtube.com/watch?v=pTyqhhWw614)
- **Games Computers Play (about 186k views).** Tests popular online tips with a Python simulation of hundreds of thousands of games. — [YouTube snippet](https://www.youtube.com/watch?v=6EJrZeN0jNI)
- **Vox, "The right way to play Monopoly" (about 2020).** Features Brian Valentine, US representative and 3rd place at the 2015 World Championship in Macau. Covers probability heat maps, orange near Jail, housing shortages, auctions and the social side of the game. View count not obtained. — [BetaSeries listing](https://www.betaseries.com/en/episode/vox/s01e1144); [Macau Daily Times](https://macaudailytimes.com.mo/?p=34242)
- **Great Big Story, Jason Bunn (1985 world champion), eight tips including "buy the oranges."** View count not obtained. — [Laughing Squid](https://laughingsquid.com/monopoly-champion-gives-tips-for-winning/)
- **LLM-vs-LLM Monopoly by JC BuenaVentura (29 Dec 2025).** GPT 5.2, Claude Opus 4.5, Gemini 3 Pro and Grok 4.1 played with custom rules (land-value levy, predatory bankruptcy auctions). A bug blocked house selling and the game ended early with no clear winner. View count not obtained. — [ReadRay/Lilys summary](https://readray.ai/notes/pt/grok-41-20260205/gpt-claude-grok-gemini-monopoly-battle)

### Inferences
- Perfect Play has already taken the obvious premise ("bots with named strategies, hundreds of millions of games, official rules, four-player tables, a ranked list") and drew about 2.7M views in about three weeks. Popperish's "950 million games" follow-up got about 10k views, which suggests a bigger game count alone does not set a video apart.
- The audience for this topic is large and recurring: 12M (2022), 3.25M (2016), about 2.7M (2026), and about 7M for short tip videos. A new video should start from Perfect Play's results and say how it differs, not repeat them.
- Perfect Play's like-to-view ratio (about 0.6%) is far below b2studios (about 2.5%) and What's What (about 3%). That fits the Hacker News complaints about AI narration and suggests room for a more personal, charismatic presentation. This is an inference, not a measured cause.

### Gaps
- I found no evidence of dedicated Monopoly-strategy videos from Vsauce/Vsauce2, Veritasium, The Game Theorists, Tom Scott, Wendover, Numberphile (standalone) or Jabrils. I could not prove they do not exist, because direct YouTube search was blocked.
- Upload dates for the What's What and Way Too Simple videos are unverified. RYD logged the "ADVANCED VERSION" (10 Sep 2026) a day before the original (11 Sep 2026), so `dateCreated` is not a reliable upload date. Whether these are Shorts is also unverified.
- View counts for Vox, Great Big Story, JC BuenaVentura's LLM game and any Hannah Fry TED or Royal Institution talk segments were not obtained.
- TikTok and Shorts view counts could not be retrieved. TikTok discovery pages are dominated by Monopoly GO (the mobile game) content.
- I could not access YouTube comment sections for any video.

## Q2. What claims recur, and are any contradicted by the math or simulations?

### Takeaway
The claims that recur most are:
- orange is the best set (helped by Jail);
- Jail is the most-visited square;
- the third house is the key purchase;
- utilities are pointless;
- railroads are disputed.

Simulations contradict several of them. Perfect Play found that being picky about colour groups, stopping at three houses, hoarding cash and specialising in railroads all lose. Brown and railroads get opposite verdicts from Fry, b2studios' AI and Perfect Play. Fry also says the best set depends on game length and number of opponents, which most tip videos ignore.

### Cited Findings

**Orange is best**
- Sources that say so:
  - Fry/Parker, attributing it to oranges sitting 6 to 9 squares past Jail — [Inverse, 7 Jul 2017](https://www.inverse.com/article/32781-win-monopoly-using-math)
  - Vox / Valentine — [BetaSeries/Vox](https://www.betaseries.com/en/episode/vox/s01e1144)
  - Jason Bunn — [Laughing Squid](https://laughingsquid.com/how-to-win-monopoly/)
  - Hasbro's official TikTok ("Buy orange + red") — [TikTok @monopoly discover page](https://www.tiktok.com/discover/monopoly-tips-and-tricks)
  - A 10M-turn MATLAB simulation — [Towards Data Science](https://towardsdatascience.com/simulating-monopoly-finding-the-best-properties-using-matlab-130fe557b1ae)
- **Nuance from Fry and Oléron Evans.** The best set depends on context:
  - One opponent: orange or light blue.
  - Two or three opponents in a longer game: orange and red.
  - More than three opponents: green.
  - Boardwalk is the best single property, but the dark-blue set is weaker because Park Place is rarely visited.
  - "No single set is the key to winning."

  — [TED Ideas, 16 Nov 2017](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- **Perfect Play's partial contradiction.** Targeting only some colour groups (Cheap Side or Expensive Side) did poorly, at about 7 to 9% in narration, and buying everything won. Orange with three houses pays back after about 29 opponent rolls. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

**Jail is the most landed-on square**
- Repeated by Inverse and tip videos. — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math); [Gistly summary of a "win every time" video](https://gist.ly/youtube-summarizer/mastering-monopoly-strategies-to-win-every-time)
- Probability analysis puts Illinois Avenue as the most-landed square after Jail, and finds the squares between Jail and Go To Jail are the busiest. — [Tkcs-Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- On the UK board, Trafalgar Square is the most-landed street. — [Plus Magazine (Haigh, Monte Carlo)](https://plus.maths.org/content/monte-carlo-monopoly); [Tcl wiki](https://wiki.tcl-lang.org/page/Monopoly)

**Build three houses**
- Fry: the third house is the one to rush. — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- Tim Darling's breakeven simulations, via Inverse. — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- UK champion Natalie Fitzsimons: "lots of houses, no hotels." — [Shortlist](https://www.shortlist.com/news/how-to-win-at-monopoly-lifehack)
- Hasbro TikTok: "Don't rush the hotels. Four houses hit hard." — [TikTok](https://www.tiktok.com/discover/monopoly-tips-and-tricks)
- **Perfect Play's contradiction.** Stopping at three houses underperforms because the jump to hotel rent is large. The tuned bot builds to three and then continues to hotels. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- These two positions are reconcilable. "The third house is the best marginal buy" is not the same claim as "stop at three."

**Railroads: underrated or useless?**
- Pro:
  - Valentine says don't dismiss railroads and utilities — [AOL](https://www.aol.com/finance/2016-07-24-monopoly-success-strategies-for-real-life-21438013.html)
  - b2studios' AI favoured railroads, which were its most-won properties — [UGA blog](https://ctlsites.uga.edu/mist5720/?p=1522)
- Con:
  - 2009 world champion Bjørn Halvard Knappskog: never buy utilities or railways — [Motherboard/Vice](https://motherboard.vice.com/read/10-essential-tips-from-a-monopoly-world-champion)
  - Fry: "unlikely they will get you anywhere fast" — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
  - MATLAB: railroads are only useful late in the game if you own three or four — [TDS](https://towardsdatascience.com/simulating-monopoly-finding-the-best-properties-using-matlab-130fe557b1ae)
  - Perfect Play: Railroad Baron came last at 3% — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

**Utilities**
- "Completely pointless" (Fry). — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- Consistent with Perfect Play, where skipping utilities caused no measurable harm. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

**Brown**
- Fry: "basically pointless" for houses. — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- b2studios' AI dropped brown once it could trade. — [fomo.ai snippet](https://learn.fomo.ai/p/what-happened-when-an-ai-played-11)
- **Contradiction.** In Perfect Play, skipping brown cost about 5 win-rate points. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

**Green**
- Fry: best with more than three opponents. — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- MATLAB: worst set for hotels. — [TDS](https://towardsdatascience.com/simulating-monopoly-finding-the-best-properties-using-matlab-130fe557b1ae)
- Perfect Play: green and dark blue finished near the bottom of the "picky" tests. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- Games Computers Play built a video around "should you really ignore greens?" — [YouTube](https://www.youtube.com/watch?v=6EJrZeN0jNI)

**Jail timing**
- Use Get Out of Jail Free early while properties are unowned, and stay in Jail (roll for doubles) once the board is built up (Fitzsimons). — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- Perfect Play's top bot also prefers to stay in Jail late in the game. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

**Cash reserves and trading**
- Perfect Play: the Saver strategy loses, and even a "smart saver" won only 13%.
- Refusing trades is catastrophic: "the worse half of a deal was still better than no deal at all."

— [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

### Inferences
- The clearest open conflicts a new video could settle head-on:
  - railroads (b2studios' AI favoured them; Perfect Play's specialist came last);
  - brown (Fry and b2studios dismissed it; Perfect Play found skipping it costs about 5 points);
  - green (best with many opponents per Fry; worst for hotels per MATLAB).
- Most of these conflicts likely come from differences in method: landing frequency versus ROI versus full-game win rate, number of players, game length, and whether trading exists. Showing why the sources disagree would be new.
- "Railroad Baron" in Perfect Play pairs railroads with utilities and bids double at auction. That tests "railroads plus overpaying," not "railroads are underrated" as an extra holding. A cleaner test of "add railroads on top of a normal strategy" appears not to have been done.

### Gaps
- I could not verify the specific claims in the What's What (about 6.9M) and Way Too Simple (about 2.7M) videos. Only AI-generated aggregator summaries were reachable, and the Gistly summary cited above could not be confirmed to describe either video.
- b2studios' exact findings come from secondary sources (fomo.ai and a UGA class blog), not from the video or its repository.

## Q3. Which formats did well, and what do audiences ask for that wasn't answered?

### Takeaway
The best-performing formats are:
- "AI learns" / evolving bots (b2studios, about 12M);
- the big-number strategy tournament (Perfect Play, about 2.7M in about three weeks);
- a math-celebrity collaboration (Stand-up Maths with Fry, about 3.25M);
- punchy "how to win every time" tip videos (about 2 to 7M each in autumn 2026).

What audiences say is missing: realistic trading and negotiation, house-shortage tactics, house rules (Free Parking, every-landing auctions), self-play / AlphaZero-style learning, and results at player counts other than four.

### Cited Findings
- Format performance, measured by view counts: see the Q1 table. — [RYD API, b2studios](https://returnyoutubedislikeapi.com/votes?videoId=dkvFcYBznPI); [RYD, Perfect Play](https://returnyoutubedislikeapi.com/votes?videoId=S_2eVAaw1UM); [RYD, Stand-up Maths](https://returnyoutubedislikeapi.com/votes?videoId=ubQXz5RBBtU); [RYD, What's What](https://returnyoutubedislikeapi.com/votes?videoId=qu6GMpQfyUU)
- Hacker News requests and critiques of Perfect Play — [Hacker News thread](https://news.ycombinator.com/item?id=50175732):
  - "Most simulations skip trading," which matters in competitive play. Others argued about win-win and compensation trades. Perfect Play's bots only make mutual set-completing swaps.
  - A question about AlphaZero-style self-play, with a suggestion to use sealed-bid auctions to simplify the action space.
  - The house-shortage tactic: "never upgrade to hotels" to exhaust the 32 houses, and waiting for the pay-per-house card.
  - Many commenters did not know the official auction rule. Others described house-rule variants: auction on every landing, random property deals, inherited starting properties, instant hotels.
  - The winning strategy is "dull" because it hardly adapts to opponents. Auctions are where the real strategy lies.
  - Player count barely came up.
- Perfect Play itself acknowledges or implies these limits:
  - Only four-player tables were tested.
  - Bots cannot negotiate like humans.
  - House rules and variants were not tested.
  - Jail was not studied in depth.
  - It asked viewers which game to test next.

  — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- The Neatorama reader comment joked that he has never played by the official rules, so the results don't apply to him. — [Neatorama](https://www.neatorama.com/2026/10/20/The-Best-Monopoly-Strategy-as-Tested-on-300-Million-Games/)
- The LLM-vs-LLM format (Dec 2025) shows a new trend of "AI personalities" negotiating and trash-talking. The game ended early because of a bug. — [ReadRay/Lilys](https://readray.ai/notes/pt/grok-41-20260205/gpt-claude-grok-gemini-monopoly-battle)

### Inferences
Angles that appear not to have been covered by the high-view videos found, and that build on Perfect Play's popularity:
1. **Player count from 2 to 6 players.** Fry's math says the best set changes with the number of opponents ([TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)), but Perfect Play only tested four-player tables.
2. **House rules most families play.** Free Parking jackpot, no auctions, no 32-house limit: do the "best strategies" flip? 49% of players make up rules and about 68% never read the rulebook ([Hasbro newsroom](https://newsroom.hasbro.com/node/14166)).
3. **Rich trading and negotiation.** Bots that accept lopsided trades, use cash sweeteners, or block opponents. Perfect Play's "worse half of a deal" finding invites this.
4. **The house-shortage tactic** (hoarding houses, never building hotels), which Hacker News readers specifically raised.
5. **Matchup matrix / rock-paper-scissors dynamics.** Which strategy beats which, and evolutionary "meta" population dynamics, rather than a single ranked list.
6. **Resolving the railroads and brown contradictions** between b2studios, Fry and Perfect Play.
7. **Human-readable "tip audits"** that test the viral "how to win every time" tips directly. Games Computers Play did a small version at about 186k views, so an updated and larger version, possibly built against the 2026 viral tip videos, has precedent.
8. **Simulation bots versus LLM players,** connecting with the LLM-Monopoly trend.

### Gaps
- No YouTube top-comment data could be retrieved for any video, because YouTube was blocked. Audience requests come only from Hacker News, Neatorama and Fark.
- No Reddit thread discussing Perfect Play or b2studios was found.

## Q4. Notable popular articles on Monopoly strategy and what they claim

### Takeaway
The main pop-media sources are:
- TED Ideas (Fry and Oléron Evans, 2017);
- Inverse (2017, combining Fry/Parker, Tim Darling and Fitzsimons);
- Business Insider (Walter Hickey, about 2014);
- Motherboard/Vice (Knappskog's ten tips);
- Towards Data Science MATLAB and Markov simulations;
- Plus Magazine (John Haigh);
- Ian Stewart's 1996 Scientific American landing-probability work, still used as the benchmark, including by Perfect Play.

I found no dedicated r/dataisbeautiful post, and no Guardian, Wired, Popular Mechanics or FiveThirtyEight piece.

### Cited Findings
- **TED Ideas, 16 Nov 2017 (Fry and Oléron Evans, from *The Indisputable Existence of Santa Claus*).**
  - Best set depends on game length (about 30 turns per player assumed) and number of opponents.
  - The third house is the key one.
  - Utilities are "completely pointless."
  - Railroads won't get you anywhere fast.

  — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- **Inverse, 7 Jul 2017 (Dan Robitzski).**
  - Orange is best (6 to 9 squares from Jail); red is second.
  - Build three houses (Darling).
  - Wait to buy houses until opponents are about 7 squares away.
  - Use Get Out of Jail Free early and stay in Jail late (Fitzsimons).
  - Trade Park Place for oranges.

  — [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- **Business Insider, "How To Use Math To Win At Monopoly"** (video based on Walter Hickey's analysis; secondary summaries stress purple, orange and red). — [Business Insider India](https://www.businessinsider.in/how-to-use-math-to-win-at-monopoly/articleshow/45577191.cms)
- **Motherboard/Vice, "10 essential tips from a Monopoly world champion"** (Knappskog, 2009 champion). Includes "never buy utility or railway companies" and a traffic heatmap. — [Motherboard](https://motherboard.vice.com/read/10-essential-tips-from-a-monopoly-world-champion)
- **Shortlist**, UK champion Natalie Fitzsimons: get lots of houses but no hotels. — [Shortlist](https://www.shortlist.com/news/how-to-win-at-monopoly-lifehack)
- **AOL** (Valentine): don't dismiss railroads and utilities; dice luck still matters a great deal. — [AOL](https://www.aol.com/finance/2016-07-24-monopoly-success-strategies-for-real-life-21438013.html)
- **Towards Data Science (MATLAB, 10M turns).**
  - Orange is best.
  - The oranges are all in the top five with three houses and are the top three for hotels.
  - Greens are the worst for hotels.
  - New York and Tennessee are the 3rd and 5th most-landed squares.

  — [TDS](https://towardsdatascience.com/simulating-monopoly-finding-the-best-properties-using-matlab-130fe557b1ae)
- **Ian Stewart's "How Fair Is Monopoly?"** (Scientific American, 1996, Markov chain). This is the canonical landing-probability reference. — [Harvard course handout](https://legacy-www.math.harvard.edu/archive/21b_fall_05/Articles/howfairismonopoly.pdf)
- **Plus Magazine, John Haigh's Monte Carlo work** (UK board; Trafalgar Square most landed). — [Plus Magazine](https://plus.maths.org/content/monte-carlo-monopoly)
- **Popular Science**, "How to destroy your friends and family in 4 popular board games," includes Monopoly. — [PopSci](https://www.popsci.com/story/science/head-trip-gaming-the-system)
- **Moneywise** covers how to win at Monopoly (content not reviewed). — [Moneywise](https://moneywise.com/life/entertainment/how-to-win-at-monopoly)

### Inferences
- The print and online canon is almost entirely about **landing probability and ROI per set**, dating from 1996 to 2017. Full-game win-rate simulation became mainstream only with b2studios (2022) and Perfect Play (2026).
- A new video can position itself as "testing the canon": checking claims from Fry, the champions and the viral 2026 tip videos against full-game results.

### Gaps
- I found no reliable Guardian, Wired, Popular Mechanics, FiveThirtyEight or Smithsonian articles on Monopoly *strategy*. They may exist, but they did not surface.
- I found no specific r/dataisbeautiful Monopoly post with an upvote count.

## Q5. Viral Monopoly "strategy" myths worth busting

### Takeaway
The best myths to bust are:
- the house rules that people believe are real rules (Free Parking jackpot, no auctions);
- "just buy orange," "only railroads" or "keep a cash cushion" (all losers in full-game simulation);
- "stop at three houses";
- "how to win every time" itself, given the role of dice luck.

### Cited Findings
- **Free Parking money is not an official rule.** Nothing happens on Free Parking under official rules; Hasbro offered the jackpot only as an optional house rule. — [Fatherly](https://www.fatherly.com/news/free-parking-truth); [Hasbro newsroom](https://newsroom.hasbro.com/node/14161)
- **Auctions are official.** An unbought property must go to auction. Auctioning on every landing is a variant. Many Hacker News commenters did not know the auction rule at all. — [Hasbro newsroom (2014 survey)](https://newsroom.hasbro.com/node/14166); [Hacker News](https://news.ycombinator.com/item?id=50175732)
- **Hasbro/Wakefield survey (2014).** About 68% of US players never read the full rules and 49% make up rules. Figures vary by outlet: 66 to 70% never read the rules, and 34 to 49% made up rules. — [Hasbro newsroom](https://newsroom.hasbro.com/node/14166); [Radio X](https://radiox.co.uk/news/music/youve-been-playing-monopoly-wrong-your-whole-life)
- **"Only buy orange" or other picky strategies lose to buying everything** in full-game simulation. — [SozAI / Perfect Play](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- **"Railroads are the secret"** is contradicted by Perfect Play's Railroad Baron at 3%, though b2studios' AI liked railroads. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md); [UGA blog](https://ctlsites.uga.edu/mist5720/?p=1522)
- **"Keep cash on hand to be safe"** is contradicted: the Saver strategy loses. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)
- **"Stop at three houses"** is contradicted by Perfect Play's results, though Fry's "rush the third house" is about priority, not stopping. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md); [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/)
- **"Boardwalk and Park Place are the best"** is contradicted: Boardwalk is the best single property, but the set is weak because Park Place is rarely landed on. — [TED Ideas](https://ideas.ted.com/heres-how-to-win-at-monopoly-according-to-math-experts/); [Inverse](https://www.inverse.com/article/32781-win-monopoly-using-math)
- **"Win every time."** Even the best bot won 43% at a four-player table, and Valentine stresses dice luck. — [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md); [AOL](https://www.aol.com/finance/2016-07-24-monopoly-success-strategies-for-real-life-21438013.html)
- **"Jail is the most-landed square"** is technically true only if Jail and Just Visiting are counted together. Illinois is the most-landed property. — [Tkcs-Collins](http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml)
- **"Never buy railroads or utilities"** (a world champion's advice) is half supported: utilities don't matter, while railroads are disputed. — [Motherboard](https://motherboard.vice.com/read/10-essential-tips-from-a-monopoly-world-champion); [SozAI](https://sozai.app/transcript/monopoly-strategy-tested-300-million-games/index.md)

### Inferences
- "How to win every time" titles have millions of views (What's What about 6.9M, Way Too Simple about 2.7M), so a simulation video that audits those exact tips could ride the same search demand while offering data the tip videos lack.
- House-rule myths are probably the most widely relatable hook, because most families play with them. Measuring how house rules change the optimal strategy appears untested in popular media.

### Gaps
- The specific tips in the 2026 viral "win every time" videos could not be retrieved, so the busting list cannot yet target their exact claims. Someone should watch [What's What](https://www.youtube.com/watch?v=qu6GMpQfyUU) and [Way Too Simple](https://www.youtube.com/watch?v=CWfAlSn3u7A) directly.
- No reliable data was found on the effect of house rules (for example, how Free Parking money changes game length or the winning strategy) from a popular source.
