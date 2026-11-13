# YouTube copy: Plinko and chaos ("Same release point, different finish")

**Upload `plinko/plinko_chaos_same_release_point_different_finish_narrated_6m07s.mp4`:**
* 6:07, 1920x1080, 30 fps, H.264, AAC stereo at -16 LUFS;
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
Here's a Plinko drop that wins $10,000. Drop it again from the same release point, matched to the width of an atom, and it lands in $500.

Nothing on the board is random. Every bounce follows Newton's laws, so the same release point should always lead to the same slot. Plinko shows why that's not enough: it is chaotic.

Chaos theory studies systems like this: exact rules, no randomness, yet no long-range prediction. It began in 1961, when Edward Lorenz reran a weather simulation from a rounded number (0.506 instead of 0.506127) and got a completely different forecast: sensitive dependence on initial conditions, the butterfly effect.

On a Plinko board the culprit is the round pins. A flat wall faces the same way everywhere, so two chips a hair apart leave a hair apart. A round pin faces a slightly different way at every spot, so two chips that hit it a hair apart bounce off in different directions, and the next pin widens the gap again: about 2.2 times per bounce, doubling every tenth of a second.

I measured the board from broadcast frames, rebuilt it in a physics engine and in Blender, and ran the experiment: find a release point that wins, then replay it as precisely as you like.

- A player can let go anywhere across the top, at the pins or up to 30 cm (12 in) above them. Find winning drops from all over that area and replay each from exactly the same release point.
- Replayed to within a micron, a winner wins again 16% of the time, barely better than a fresh drop (13%).
- Replayed to within an atom: 39%. Better odds, still worse than a coin flip, never a sure thing.
- Below a nanometre, each tenfold gain in precision removes only about 35% of the remaining misses. Even a hundred-thousandth of an atom misses 1 time in 22.
- Why: zoom into the map of release points, first a square 4 cm across, then a single winning release point, a hundred times closer with each step. At any scale a hand could control, it looks like static. The gold turns solid only within a few atoms of it, and most winning release points never do (only 13 in 100 are solid gold even at the scale of one atom).
- Make the release point perfect and change just one thing: the board's temperature. Where would the heat come from? The chips themselves. A chip at the top holds about 5 joules from its height; as it falls that becomes speed, and every hit on a pin turns some of it into heat, until all 5 joules are heat. Spread through the board, that's 0.00006 C warmer, and warm wood and steel expand: the board grows about a nanometre, the width of ten atoms. Doubled by about twenty bounces, a nanometre becomes a centimetre: a different slot. Replayed exactly on a board that much warmer, winning drops win again only 39% of the time; even a billionth of a degree spoils 6 replays in 100.

One drop can't be predicted. Four hundred can, and the pile they make is why the middle still pays best.

0:00 Same release point, different finish
0:18 Nothing on the board is random
0:35 Two chips a thousandth of a millimetre apart
0:52 What is chaos theory? Lorenz and the butterfly effect
1:45 Why round pins make chaos
2:33 How far ahead can you see?
2:51 Replaying a winning drop
3:20 Every place you could let go
3:59 Where the heat comes from: one chip's energy
4:54 Same drop, warmer board
5:11 How much warmer is too warm?
5:31 One drop vs four hundred
5:55 Deterministic, and unpredictable

How it's made: the board is built in Blender from measurements of the TV board; every chip path comes from the simulator. Show results from tpirstats.com (seasons 44 to 54). Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/plinko/ACCURACY.md

Independent analysis; not affiliated with the show or the network. The board is a measured replica without the show's branding.
```

**Tags:** chaos theory, butterfly effect, plinko, price is right, physics simulation, blender, sensitive
dependence on initial conditions, deterministic chaos, lyapunov exponent
