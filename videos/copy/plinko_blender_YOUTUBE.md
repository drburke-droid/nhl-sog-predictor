# YouTube copy: Plinko, the Blender edition

**Upload `plinko/where_to_drop_a_plinko_chip_blender_2m55s.mp4`:**
* 2:55, 1920x1080, 30 fps, H.264;
* rendered in Blender (Cycles), on-screen captions, no narration, no music.

This replaces the flat 2D film as the main upload; the 2D cut stays in `plinko/` as a backup. The Blender
edition adds the micron question ("with perfect aim, could you hit $10,000 every time?").

**Titles.**

| | Title |
|---|---|
| A | Could a Robot Win Plinko Every Time? I Rebuilt the Board to Find Out |
| B | Where Should You Drop a Plinko Chip? (A Physics Simulation of the Real Board) |
| C | Two Plinko Chips, 0.001 mm Apart. One Wins $10,000. |

**Description**

```
Where should you drop a Plinko chip, and could perfect aim win it every time?

I measured the TV board from broadcast frames (no blueprint is public): 9 slots 7 inches apart, 13 rows of steel pins, 98 pins in all, sawtooth side walls and a 5.1-inch chip. Then I rebuilt it twice: in a physics engine, where a spinning chip bounces off every pin, and in Blender, to film it.

The textbook treats every pin as a coin flip and says the middle pays $2,558 a chip. The show disagrees: across 1,029 real chips from 2015 to 2026, only 15.2% landed in the $10,000 and the average was $1,902. Tuned to match the show, the simulator says the middle three drops tie at about $1,829 a chip, and the edge pays about $1,617.

Could a robot with micron accuracy pick the slot? No. Two chips released 0.001 mm apart split after a few pins: one wins $10,000, its twin wins $100. Twins one micron apart land in the same slot 14% of the time, about the same as two random chips (12%). Even a hundredth of an atom apart, they agree only half the time.

So: drop it in the middle, don't try to finesse the release, and fight for every chip in the small-prize round. The last digit was the right one 61% of the time.

0:00 Where should you drop a Plinko chip?
0:11 The board, measured
0:29 The textbook answer
0:49 What really happens on the show
1:05 Rebuilding it in a physics engine
1:35 Could micron accuracy win every time?
1:43 Two chips, 0.001 mm apart
1:57 Where you let go, and where it lands
2:21 So where should you drop it?
2:45 The answer

How it's made: the board is built in Blender from the same measurements the physics engine uses; every chip path comes from the simulator, not keyframed by hand. Show results from tpirstats.com (seasons 44 to 54). Coin-flip model as in Biesterfeld, Journal of Statistics Education 9(3), 2001. Method, sources and error bars: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/plinko/ACCURACY.md

Independent analysis; not affiliated with the show or the network. The board is a measured replica without the show's branding.
```

**Tags:** plinko, price is right, plinko strategy, blender, physics simulation, chaos theory, butterfly effect,
probability, game show math
