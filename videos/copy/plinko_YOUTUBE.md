# YouTube copy: Plinko, where to drop the chip

**Upload `plinko/where_to_drop_a_plinko_chip_3m21s.mp4`:**
* 3:21, 1920x1080, 30 fps, H.264;
* on-screen captions, no narration, no music (add a bed under it if you like; nothing is timed to sound).

**Short: `plinko/where_to_drop_a_plinko_chip_SHORT_58s.mp4`**, 1080x1920, 0:58.

**A/B/C titles.**

| | Title |
|---|---|
| A | Where Should You Drop a Plinko Chip? I Rebuilt the Board to Find Out |
| B | Does It Matter Where You Drop a Plinko Chip? Less Than the Textbook Says |
| C | I Rebuilt Plinko in a Physics Engine. The Middle Matters Less Than You Think |

The full analysis (`analyze.py all`) simulates about 840,000 drops: 12 bounciness settings x 30,000,
120,000 at the tuned setting, 17 sensitivity runs x 20,000 and 18,000 twin-chip drops.

**Description**

```
Where should you drop a Plinko chip? The textbook answer treats every pin as a coin flip: drop it in the middle and you win $2,558 a chip on average, against $780 at the edge.

The show doesn't agree. Across 1,029 chips from 2015 to 2026, only 15.2% landed in the $10,000, not the 22.6% the coin flips predict, and the average was $1,902 a chip.

So I measured the board from broadcast frames (no blueprint is public): 9 slots 7 inches apart, 13 rows of steel pins, 98 pins in all, sawtooth side walls and a 5.1-inch chip. Then I rebuilt it in a physics engine: a spinning disc that bounces and slides down a board leaning back 10 degrees. The one thing I couldn't measure, how bouncy the chip is, I tuned until the simulator matched the show.

What I found:
- The middle is still the best drop, or tied for best, in every version of the model.
- On a board that behaves like the real one, the middle three are a tie at about $1,829 a chip; the edge pays about $1,617.
- Two chips released 0.001 mm apart end up in unrelated slots. Spin, toss, a steady hand: none of it changes the odds.
- What does pay is more chips. In the small-prize round, the last digit was the right one 61% of the time.

One open question: the show's numbers also fit a coin-flip board if contestants drop all over the place. Nobody logs where contestants drop. If you know of a dataset, tell me.

0:00 Where should you drop the chip?
0:09 The board, measured
0:31 The textbook answer
0:57 What really happens on the show
1:21 Rebuilding it in a physics engine
1:51 Why nobody can aim
2:13 Tuning the bounce to match the show
2:33 So where should you drop it?
3:11 The answer

How it's made: every frame is drawn in code from the simulator; there is no footage from the show. Show results from tpirstats.com (seasons 44 to 54). Coin-flip model as in Biesterfeld, Journal of Statistics Education 9(3), 2001. Method, sources and error bars: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/plinko/ACCURACY.md

Independent analysis; not affiliated with the show or the network.
```

**Tags:** plinko, price is right, plinko strategy, probability, physics simulation, chaos theory, galton
board, game show math, expected value
