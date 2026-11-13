# YouTube copy: Plinko and chaos ("Same start, different finish")

**Upload `plinko/plinko_chaos_same_start_different_finish_narrated_3m30s.mp4`:**
* 3:30, 1920x1080, 30 fps, H.264, AAC stereo at -16 LUFS;
* narrated (ElevenLabs, "Lily" voice; script in `plinko_chaos_NARRATION.md`), on-screen captions, no music.

This is the main Plinko upload. It replaces the 2:54 chaos cut, whose replay numbers were inflated by
simulated releases the chip can't physically make (see `plinko/ACCURACY.md` section 7).

**Titles.**

| | Title |
|---|---|
| A | Same Drop, Different Prize: Chaos Theory on a Plinko Board |
| B | Could You Win Plinko Twice With Atomic Precision? |
| C | Why the Same Plinko Drop Doesn't Win Twice |

**Description**

```
Here's a Plinko drop that wins $10,000. Drop it again from the same spot, matched to the width of an atom, and it lands in $500.

There are no dice inside the board. Every bounce follows Newton's laws, so the same start should give the same finish. Plinko shows why that's not enough: it is chaotic. Two paths that start almost together are pushed apart at every round pin, about 2.2 times per bounce, so a tiny difference doubles every tenth of a second.

I measured the board from broadcast frames, rebuilt it in a physics engine and in Blender, and ran the experiment: find a release point that wins, then replay it as precisely as you like.

- A player can let go anywhere across the top, at the pins or up to 30 cm (12 in) above them. Find winning drops from all over that area and replay each from exactly the same spot.
- Replayed to within a micron, a winner wins again 16% of the time, barely better than a fresh drop (13%).
- Replayed to within an atom: 39%. Better odds, still worse than a coin flip, never a sure thing.
- Below a nanometre, each tenfold gain in precision removes only about 35% of the remaining misses. Even a hundred-thousandth of an atom misses 1 time in 22.
- Why: zoom in on the release points around two winning spots. At any scale a hand could control, both look like a random drop. One turns solid gold only within a few atoms (an island); the other stays mixed even at a fifth of an atom (a knife edge). Only 13 in 100 winning spots are islands, and one win can't tell you which you found.
- Make the start perfect and change just one thing: the board's temperature. The chip's fall turns about 5 joules into heat, warming the board by 0.00006 C. Replayed exactly on a board that much warmer, winning drops win again only 39% of the time. Even a billionth of a degree spoils 6 replays in 100.

One drop can't be predicted. Four hundred can, and the pile they make is why the middle still pays best.

0:00 Same start, different finish
0:17 No dice inside
0:28 Sensitive dependence on initial conditions
0:42 Why round pins make chaos
1:00 How far ahead can you see?
1:14 Replaying a winning drop
1:42 Every place you could let go: islands and knife edges
2:19 Perfect start, one thing changed: the board's temperature
2:36 How much warmer is too warm?
2:56 One drop vs four hundred
3:20 Deterministic, and unpredictable

How it's made: the board is built in Blender from measurements of the TV board; every chip path comes from the simulator. Show results from tpirstats.com (seasons 44 to 54). Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/plinko/ACCURACY.md

Independent analysis; not affiliated with the show or the network. The board is a measured replica without the show's branding.
```

**Tags:** chaos theory, butterfly effect, plinko, price is right, physics simulation, blender, sensitive
dependence on initial conditions, deterministic chaos, lyapunov exponent
