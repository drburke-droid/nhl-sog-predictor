# YouTube copy: Plinko and chaos ("Same release point, different finish")

**Upload `plinko/plinko_chaos_same_release_point_different_finish_narrated_6m07s.mp4`:**
* 6:07, 1920x1080, 30 fps, H.264, AAC stereo at -16 LUFS;
* narrated (ElevenLabs, "Lily" voice; script in `plinko_chaos_NARRATION.md`), on-screen captions, no music.

This is the main Plinko upload. It replaces every earlier chaos cut (the 2:54 cut's replay numbers were inflated
by simulated releases the chip can't physically make; see `plinko/ACCURACY.md` section 7).

## Thumbnails

In `plinko/thumbnails/`, 1280x720, built by `plinko/thumbs_chaos.py`: every pairing of three lines of text with
three pictures, named `plinko_chaos_thumb_<text><picture>.png`. YouTube's thumbnail test takes up to three.

| | 1: the opening drop and its replay ($10,000 and $500) | 2: the warm-board pair ($10,000 and $0) | 3: a photo of a real Plinko board |
|---|---|---|---|
| **A** "Same release point. Different prize." | `A1` | `A2` | `A3` |
| **B** "Match release point to an atom, different result." | `B1` | `B2` | `B3` |
| **C** "0.00006 °C warmer: $10,000 became $0." | `C1` (prize pills left off: the chip shown lands in $500) | `C2` | `C3` |

The natural pairs are A1, B1 and C2, where the picture shows exactly what the text says; column 3 trades that for
instant recognition of a real board. Suggested test: A1, A3 and C2.

**If you use a thumbnail from column 3**, add this line to the end of the description (the photo's licence requires
the credit, and those thumbnails, as adaptations of it, are themselves CC BY-SA 4.0):

```
Thumbnail photo: "The Price Is Right Plinko Home Game" by TaurusEmerald, CC BY-SA 4.0, via Wikimedia Commons (https://commons.wikimedia.org/wiki/File:The_Price_Is_Right_Plinko_Home_Game.jpg). Thumbnail adapted from it, also CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
```

Other free Plinko photos on Commons (a library event, a county fair, a state agency booth) are homemade boards with
identifiable people, children among them, so they are not used.

## Titles

| | Title | Length |
|---|---|---|
| A | Same Drop, Different Prize: Chaos Theory on a Plinko Board | 58 |
| B | Could You Win Plinko Twice With Atomic Precision? | 49 |
| C | Why You Can't Repeat a Winning Plinko Drop | 42 |
| D | How 0.00006 °C Turns a $10,000 Plinko Drop Into $0 | 50 |
| E | Plinko Is Chaos Theory: The Butterfly Effect on The Price Is Right | 66 |

Recommended: title A with thumbnail A1 or A3; title B with B1; title D with C2. All are true to the film: the replay that "doesn't repeat" wins again
39% of the time at atomic precision, never every time.

## Description

```
This Plinko drop wins $10,000. Drop it again from the same release point, matched to the width of an atom, and it lands in $500. That's chaos.

Plinko, from The Price Is Right, is a perfect machine for chaos theory. Nothing on the board is random: every bounce follows Newton's laws, so the same release point should always lead to the same slot. It doesn't, and this film shows why.

Chaos theory studies systems like this: exact rules, no randomness, yet no long-range prediction. It began in 1961, when the meteorologist Edward Lorenz repeated a weather simulation from a rounded number (0.506 instead of 0.506127, about 1 part in 4,000) and two months into the simulated weather the forecasts had nothing in common: sensitive dependence on initial conditions, the butterfly effect.

On a Plinko board, the culprit is the round pins. A flat wall faces the same way everywhere, so two chips a hair apart bounce off a hair apart. A round pin faces a slightly different way at every spot, so two chips that hit it a hair apart bounce off in different directions, and the next pin widens the gap again: about 2.2 times per bounce, doubling every tenth of a second, nearly a trillion times over a whole drop.

I measured the board from broadcast frames, rebuilt it in a physics simulator and in Blender, and ran the experiment: find a release point that wins, then replay it as precisely as you like.

- A player can let go anywhere across the top, at the pins or up to 30 cm (12 in) above them. Of all those release points, 13% win $10,000.
- Replayed to within a micron, a winning drop wins again 16% of the time: barely better than a fresh drop.
- Replayed to within an atom: 39%. Still worse than a coin flip. Even at a hundred-thousandth of an atom, 1 replay in 22 misses.
- Zoom into the release points: at every scale a hand could control, gold and grey are mixed like static. Around one winning release point the gold turns solid only within a few atoms, and most winning release points never turn solid at all.
- Then make the release point perfect and change just one thing: the board's temperature. A chip at the top holds about 5 joules of energy from its height. As it falls that becomes speed, and every hit on a pin turns some of it into heat, until all 5 joules are heat. Spread through the board, that's 0.00006 °C warmer, and warm wood and steel expand: the board grows about a nanometre, the width of ten atoms. Doubled by about twenty bounces, a nanometre becomes a centimetre: a different slot. On a board warmed by one drop, winning drops win again only 39% of the time. Even a billionth of a degree spoils 6 replays in 100.

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

How it's made: the board is built in Blender from measurements of the TV board, and every chip path comes from a 2D rigid-body simulator tuned to 1,029 real chips from the show (tpirstats.com, seasons 44 to 54). Narration: ElevenLabs ("Lily"). Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/plinko/ACCURACY.md

Independent analysis; not affiliated with The Price Is Right, its producers or the network. The board is a measured replica without the show's branding.

#chaostheory #plinko #physics
```

## Tags

chaos theory, butterfly effect, plinko, price is right, plinko physics, lorenz attractor, edward lorenz,
sensitive dependence on initial conditions, deterministic chaos, exponential growth, lyapunov exponent,
thermal expansion, physics simulation, blender

## Pinned comment (optional)

```
Which would you rather have: aim accurate to a single atom, or a board held to a billionth of a degree? Neither gets you to 100%. (Chapters and every number are in the description.)
```
