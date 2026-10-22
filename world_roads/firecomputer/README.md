# The fire computer (`firecomputer/`)

A follow-up to the signal fires. A chain of fires carries a message; give each fire a keeper and a rule
card, and a field of them computes. It builds up in three steps, from 7 fires to 2,110, and the last
step does a job that is far too long to do by hand.

```bash
python build.py      # build every circuit, check it, count it -> ../media/firecomputer/fire_computer.{json,html}
node test_sim.js     # the page's simulator gives the same answers as Python
```

Interactive page: `../media/firecomputer/fire_computer.html` (also published as an artifact).

## The page: fourteen levels, from no maths to a supercomputer (and a cipher)

Each level adds one idea and has something to try; a bar shows how steep it is.

| Level | Idea | Try it |
|---|---|---|
| 1. One fire, one answer | lit = yes = 1, dark = no = 0 (a *bit*) | tap the fire |
| 2. Fires that count | fires worth 8, 4, 2, 1, like market weights (*binary*); each added fire doubles the reach | 1 to 10 fires, type a number, goals 13, 49, 100, 255, 1,000 |
| 3. The relay keeper is a wire | the signal-fire card, "copy the fire you watch"; 2 s per keeper | send a change down 6 relays |
| 4. Keepers that decide | NOT, AND, OR cards and their *truth tables* | pick a card, tap A and B |
| 5. One card is enough | NAND on its own; when its two fires match it shows the opposite, so a keeper watching one fire twice flips it; flip + combine rebuilds every card; the stand-ins for NOT (1 keeper), AND (2), OR (3) are an optional fold-out | tap A and B |
| 6. 1 + 1 | ones fire = "exactly one lit", twos fire = "both lit": 5 NAND keepers | the checked half adder |
| 7. Carrying the one | school column addition next to binary; carry in, sum and carry out of one column; the whole 4-column chain (32 keepers), each carry out feeding the next column | top and bottom numbers, examples 11 + 6, 5 + 2, 15 + 1; the written sum updates as the fires settle |
| 8. Sightlines are wires | village counts down ridges into a 32-keeper adder, total up to the mill | send two counts |
| 9. One real calculation | a supercomputer's calculation (16-digit scientific notation) built from the same fields: 35,505 keepers, 766 deep (25 min 32 s per answer); a factory line of 71,010 fires gives one answer every 2 s | the parts, counted |
| 10. The fastest computer on Earth | how many fires to match LineShine (No. 1, June 2026 TOP500, 2.198 × 10¹⁸ calculations a second), El Capitan, the Cray-1 or ENIAC; swap fires for flags, flashlights, relays or vacuum tubes | pick a computer and a switch; a powers-of-ten scale |
| 11. The drum | memory keepers copy on the beat (a computer's *clock*); a 3-bit counter | beat the drum |
| 12. Scrambling a message | the AI can see bonfires; mix in the key, swap by the table, shuffle | each move on its own |
| 13. The field at Bletchley Park | 2,110 fires make a *mask* from key + block number; one key fire flipped changes about half the mask | scramble a message |
| 14. Out along the beacon chain | hex digits to Paris; the same field unscrambles | send, then unscramble |

Pointing at (or tapping) any fire from Level 3 on explains it: what it watches, what the card says,
and whether the keeper has reacted yet. Levels 1 to 5 are drawn with the cards they name. From Level 6 on, every keeper comes from the checked
circuits in `circuits.py` (NAND and memory keepers only).

## The rules

* A **lit fire is a 1**, a **dark fire is a 0**. Every fire has a lid, so it can show or hide its light.
* **Every keeper carries the same card:** *keep your fire lit, unless both of the fires you watch are
  lit* (NAND). Every other rule is built from it. The "program" is simply who watches whom.
* **Memory keepers** have a second card: *when the drum beats, make your fire match the one you watch;
  between beats, leave it alone.*
* **Timing:** a field keeper takes about 2 s to see a change, check the card and move the lid. The
  simulation runs one keeper step at a time, so changes ripple through and fires flicker on the way,
  as they would in a real field. A mountain hop with lids is assumed to take 10 s.

## 1 + 1 (Level 6, 7 fires)

A and B on top; five NAND keepers below. Four of them make "lit if exactly one is lit" (the ones
place) and the fifth makes "lit if both" (the twos place). 1 + 1 = `10`: one two, no ones. It settles
in 3 keeper steps, about 6 s. One column with a carry (Level 7) takes 9 keepers.

## Sightlines are wires (Level 8, 40 fires, 3 ridges)

Two hill villages send their counts (0 to 15) down ridges of relay stations, 4 lidded fires per station
(8s, 4s, 2s, 1s). In the valley, 32 keepers (a half adder and three 9-fire full adders) add them, and
the 5-fire total climbs a third ridge to the mill. 11 + 6 = 17 reaches the mill a little over a minute after
the villages light their counts (6 hops at 10 s, plus about 16 s in the field); the slowest pair of numbers
needs 10 keeper steps in the field, because the carry passes from bit to bit.

## The drum (Level 11, 14 fires)

Three memory fires hold a number; 11 NAND keepers below work out the number plus one, and on each beat
the memory copies it. A beat is 6 keeper steps. Checked for 16 beats (it counts 0 to 7 and wraps).

## A message the sky can't read (Levels 12–14, 2,110 fires)

In Video 2's story an AI holds every phone, cable and satellite, and the warning goes out by bonfire.
But satellites can see bonfires. A private message needs a real cipher, and a real cipher is thousands of
tiny steps where one slip ruins the result: the job a field of rule-followers is good at.

* **The cipher:** PRESENT-128 (Bogdanov et al., CHES 2007; ISO/IEC 29192-2), a lightweight block cipher
  designed for tiny chips. 64-bit blocks (8 letters), a 128-bit key, 31 rounds.
* **The field at Bletchley Park** (under a roof, out of the sky's sight): 64 message fires, 64 memory
  fires for the block, 128 for the key, 5 for the round count, and 1,849 NAND keepers.
  * Each round mixes in the key (64 XORs, 4 fires each) and passes every 4 bits through the S-box (a
    shallow 72-fire recipe, 9 keepers deep). The bit shuffle costs **no fires at all**: it is only a
    choice of which fire each memory keeper watches.
  * One drum beat = 13 keeper steps = **26 s**. 31 beats plus reading the output = 409 steps =
    **13 min 38 s per 8 letters**.
* **Counter mode:** the field scrambles the block number (1588 × 65,536 + n) with the key and adds the
  result to the message. Paris builds the **identical field** with the same key; running the scrambled
  block through it gives the message back.
* **The wire:** Video 2's terrain-checked chain from Bletchley Park to Paris, 10 hops and 476 km. Each
  station shows one hex digit with 4 fires, plus a fifth that is lit while a digit shows. At 30 s per
  digit, 8 letters (16 digits) take 8 min, and the first digit reaches Paris 100 s after it leaves.
  (At 80 km the fires of a station need to stand about 30 m apart for the eye to tell them apart.)
* **By hand:** one block is about 8,777 written steps (per round: 64 key-bit additions, 16 S-box
  look-ups, 64 bits re-copied in shuffled order, the 128-bit key re-copied rotated, 2 key look-ups and
  5 round-number bits). That is 2 h 26 min at a perfect step every second; with checking, most of a day.

Example: "UNPLUG EVERYTHING AT DAWN" with key `0f1e2d3c4b5a69788796a5b4c3d2e1f0` →
`3c2d0a387b0a2429 cd461664a9c39989 e183ba5130011c1c 92d2bce86f8559e4`.

## How far to a supercomputer? (Levels 9 and 10)

Supercomputers are ranked by calculations a second, each a multiply or add of two double-precision
numbers (scientific notation, 53 binary digits). One multiply-and-add unit (2 calculations), built and
counted in `circuits.fma_parts()`:

| Part | NAND keepers | Keepers deep |
|---|---:|---:|
| multiply the digits (53 × 53 long multiplication) | 30,210 | 518 |
| line up the two numbers (106-bit shifter, checked) | 2,116 | 17 |
| tidy the result back into shape (same shifter) | 2,116 | 17 |
| add the digits (106-bit adder) | 950 | 214 |
| add the powers of ten (13-bit adder) | 113 | 28 |
| **one calculating field** | **35,505** | **about 766** |

It leaves out rounding, leading-zero counting and special cases, so it slightly undercounts.
Unpipelined, one answer takes 766 × 2 s = 25 min 32 s. As a factory line, with a holding (memory) keeper
behind every keeper, it is 71,010 fires, a beat is two reactions, and it finishes 2 calculations a beat:
**one calculation every reaction time per 71,010 switches.** So switches needed = calculations a second ×
reaction time × 71,010.

| To match LineShine (2.198 × 10¹⁸ a second) | Reaction | Spacing | Needed | Covers |
|---|---:|---:|---:|---|
| fires (lidded, one keeper each) | 2 s | 5 m | 3.1 × 10²³ | the Earth 15 billion times |
| flags (The Three-Body Problem) | 1 s | 1 m | 1.6 × 10²³ | the Earth 306 million times |
| flashlights (stadium crowd) | 0.5 s | 0.6 m | 7.8 × 10²² | the Earth 55 million times |
| telephone relays (1940s) | 0.01 s | 5 cm | 1.6 × 10²¹ | the Earth 7,652 times |
| vacuum tubes (ENIAC's 100 kHz) | 10 µs | 10 cm | 1.6 × 10¹⁸ | the Earth 31 times |

* Every person alive (8.2 billion) tending one fire: about 57,700 calculations a second, so 1.2 million
  years to do one second of LineShine's work. About 150 times ENIAC (1946, taken as 385 calculations a
  second, its 10-digit multiplication rate); matching ENIAC takes 55 million fire keepers.
* 3.1 × 10²³ fires is about the number of stars in the observable universe (rough estimate, ~2 × 10²³).
* Sources: TOP500 June 2026 (LineShine 2,198.40 PFlop/s Rmax; El Capitan 1,809.00); Cray-1 160 MFLOPS
  peak. Reaction times and spacings for flags, flashlights, relays and tubes are round assumptions.

## How big it gets

| Circuit | Fires | Keepers deep | Time at 2 s a keeper |
|---|---:|---:|---:|
| half adder (1 + 1) | 7 | 3 | 6 s |
| full adder (one column, with carry) | 12 | 6 | 12 s |
| 3-bit counter | 14 | 5 a beat | 12 s a beat |
| 4-bit adder | 40 | 10 | 20 s |
| 8-bit adder | 84 | 18 | 36 s |
| 16-bit adder | 172 | 34 | 1 min 08 s |
| 32-bit adder | 348 | 66 | 2 min 12 s |
| 8-bit × 8-bit multiplier | 616 | 68 | 2 min 16 s |
| 16-bit × 16-bit multiplier | 2,640 | 148 | 4 min 56 s |
| 30-bit × 30-bit multiplier | 9,570 | 288 | 9 min 36 s |
| **PRESENT-128 cipher, 8 letters** | **2,110** | 12 a beat, 31 beats | **13 min 38 s** |

Fire counts include the input fires (the earlier note counted NAND fires only: 600 / 2,608 / 9,510).

**When it's worth it.** For everyday arithmetic, the field loses: a person multiplies 2,400 × 365 in
about a minute, while the field needs 2,640 keepers and about five minutes. The cipher is
different because it is long, not because it is hard, and nobody in the field needs to know any arithmetic.

## Checks (`build.py`)

* half adder, full adder and 4-bit adder: every input, run keeper by keeper until the field stops changing;
* 3-bit counter: 16 drum beats, keeper by keeper;
* 8/16/32-bit adders and 8/16/30-bit multipliers: 200 random cases each;
* cipher, keeper by keeper with the drum: the four published PRESENT-128 test vectors (key and block all
  0s or all 1s), 20 random blocks against a plain software PRESENT, and a 4-block message scrambled and
  unscrambled. The software PRESENT also reproduces the four PRESENT-80 vectors in the original paper;
* `test_sim.js`: the page's JavaScript simulator, run on the netlist inlined in the page, gives the same
  test vectors and example as Python; the page itself re-runs the first test vector when it loads.

## Files

* `field.py`: fires, keepers, NAND and memory; the keeper-by-keeper simulation; layout helpers.
* `circuits.py`: half adder, adders, counter, multiplier, the S-box (exact smallest sum-of-products cover,
  shared pairs), PRESENT-128 in counter mode, and a plain software PRESENT.
* `build.py`: checks, counts, the Bletchley Park → Paris route from `media/signal/signal_fires_capitals.json`,
  and the page (`page.html` with `fire_sim.js` and the circuits inlined).

## For a video (Video 3)

The likely angle: **how powerful today's supercomputers are, shown with fire.** Explain very simply how a
computer works (Levels 1–8: a fire is a bit, fires count, keepers and cards, NAND, 1 + 1, carrying,
ridges as wires), build one real calculation (Level 9), then ask how many fires it would take to match
the fastest computer on Earth (Level 10: about 3 × 10²³, as many as the stars in the observable universe)
and swap fires for flags, flashlights, relays and vacuum tubes on the way to the transistor. The memory,
cipher and Paris levels (11–14) are optional extras or a follow-up. Closing point: computers aren't smart;
they are enormous numbers of very simple rule-followers, switching unimaginably fast. Precedents: the
human-army computer in Liu Cixin's *The Three-Body Problem*; Colossus at Bletchley Park; Polybius's torch
alphabet (2nd century BC); Matt Parker's 10,000-domino computer.
