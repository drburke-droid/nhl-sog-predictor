# YouTube copy: The Mix-Off

Eight ways of mixing dry ingredients in a knockout: four quarter-finals, two semi-finals, a final. Every
match is two simulated bowls of 23,273 grains each, and the bowl is loaded differently each round (pie
wedges, then stripes, then rings), so a winner has to cope with a new problem to go through.

**What is in the repository (`videos/powder_mixer/`):**
* `the_mix_off_PREVIEW_720p.mp4`: 3:20, 1280×720, 24 fps, H.264. On-screen captions, no narration,
  original synthesized score at −16 LUFS.
* This is a **preview**, not the upload. The 3-D picture was rendered on a CPU at 960×420 with 8 samples
  and scaled up; the text, charts and top-down views are drawn at full resolution. For the upload,
  re-render on a GPU machine and re-compose (commands in `videos/powder_mixer/README.md`).
* `same_spoon_two_ways_SHORT.mp4`: vertical 1080×1920, the first quarter-final on its own.

**A/B/C test.**

| | Thumbnail | Title |
|---|---|---|
| A | `thumbnails/mix_off_thumb_A.png`: the two bowls, "SPOON vs STAND MIXER", 96% against 78% | Spoon vs Stand Mixer: We Simulated 23,000 Grains to Settle It |
| B | `thumbnails/mix_off_thumb_B.png`: eight top-down bowls, "8 WAYS TO MIX. 1 WINNER." | 8 Ways to Mix Dry Ingredients, Ranked by Physics |
| C | `thumbnails/mix_off_thumb_C.png`: two top-down bowls, "SAME SPOON.", 98% against 17% | Same Spoon, Same Bowl: 98% Mixed vs 17% |

**Description**

```
What is the best way to mix dry ingredients: a spoon, a fork, a whisk, a stand mixer or a hand mixer? We simulated all of them, grain by grain, and ran a knockout.

Every bowl holds 23,273 grains in six colours. Each contestant gets the same bowl and 15 seconds. "% mixed" is measured, not judged: the bowl is cut into small scoops, and 100% means every scoop has the same share of each colour as a perfect shuffle.

Each round loads the bowl differently, so going through means solving a new problem: pie wedges in the quarter-finals, stripes in the semi-finals, rings in the final.

Results
Quarter-finals (pie wedges): spoon 98% beat a spoon twirled in the middle 17%. Stand mixer with large paddle 77% beat stand mixer with wire whip 72%. Whisk 98% beat hand mixer 60%. Fork 84% beat stand mixer with small paddle 66%.
Semi-finals (stripes): spoon 96% beat the stand mixer's large paddle 78%. Whisk 97% beat fork 79%.
Final (rings): whisk 94% beat spoon 88%.

What it shows
1. Coverage beats power. A tool only mixes where it sweeps. The same spoon scored 98% stirred in a spiral through the whole bowl and 17% twirled in the middle.
2. Count strokes, not seconds. A spoon or whisk needed 11 to 20 strokes to reach 90%, depending on how the bowl was loaded, and a spoon stirred twice as fast needed the same number of strokes.
3. How you load the bowl matters. Rings are the hard case, because powder has to move in and out, not just round. That is also where the stand mixer did its best work: 94% in a separate 20-second test.

What this is, and is not
This is a physics simulation of dry, free-flowing grains of equal size: no clumping, no dust cloud, no butter. The bowl is a half-scale model of a 22 cm kitchen bowl. The hand tools move at 1.5 strokes a second, the stand mixer at 0.8 orbits a second, the hand mixer's beaters at 5 turns a second, which is on the slow side for a real one. Change those speeds and the machines' scores change; the hand tools' strokes-to-mix does not. There is no published experiment on spoon or whisk strokes for dry powders to check the hand-tool results against.

0:00 The question
0:09 How it is scored, and who is playing
0:24 Quarter-final 1: spoon vs spoon twirled in the middle
0:47 Quarter-final 2: stand mixer, large paddle vs wire whip
1:06 Quarter-final 3: whisk vs hand mixer
1:24 Quarter-final 4: fork vs small paddle
1:47 Semi-final 1: spoon vs stand mixer, on stripes
2:10 Semi-final 2: whisk vs fork
2:32 The final: spoon vs whisk, on rings
2:55 The champion, and what it shows

How it's made: the grains are simulated with position-based dynamics with friction, at a 1 ms time step. The pictures are rendered in Blender from the simulated grain positions, and every number on screen is measured from the same grains. Code, data and method: https://github.com/drburke-droid/nhl-sog-predictor/tree/master/videos/powder_mixer

Sources: P.M.C. Lacey, J Appl Chem 4:257 (1954). D. Brone, A. Alexander & F.J. Muzzio, AIChE J 44:271 (1998). G.R. Chandratilleke et al., Ind Eng Chem Res 49:5467 (2010). V. Boonkanokwong et al., Powder Technol 302:333 (2016). B. Jadidi et al., Processes 11:733 (2023). D.J. Pine et al., Nature 438:997 (2005).

Music: an original score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 The question
0:09 How it is scored, and who is playing
0:24 Quarter-final 1: spoon vs spoon twirled in the middle
0:47 Quarter-final 2: stand mixer, large paddle vs wire whip
1:06 Quarter-final 3: whisk vs hand mixer
1:24 Quarter-final 4: fork vs small paddle
1:47 Semi-final 1: spoon vs stand mixer, on stripes
2:10 Semi-final 2: whisk vs fork
2:32 The final: spoon vs whisk, on rings
2:55 The champion, and what it shows
```

**Tags:** mixing, baking, dry ingredients, whisk, stand mixer, hand mixer, spoon, physics simulation,
granular flow, powder mixing, Blender, kitchen science, food science, simulation, satisfying

**Pinned comment idea:** "Which result surprised you most? For us it was the same spoon scoring 98% and
17% depending only on where it went."

**Before uploading.**
* The repository link in the description points at `master`; the work is on the branch
  `claude/charming-dijkstra-sd3q1f` until it is merged.
* The score was synthesized and checked for level (−16 LUFS) and clipping, but nobody has listened to it.
  Listen before publishing, or replace it.
