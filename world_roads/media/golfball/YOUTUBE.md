# YouTube copy: the golf-ball film

A companion to the vernier film, in the same paper-and-pencil look. It covers four things:
* past 160 yards, a golf ball is smaller than the finest detail you can resolve;
* past that point it doesn't look smaller, only fainter;
* why you can still see it, and why a dark background lets you see it so much farther;
* where a 20/20 E lands on your cones.

**Upload `golfball_4k_master.mp4`:**
* 7:04, 3840×2160, 30 fps, HEVC CRF 24, 166 MB;
* on-screen captions, no narration;
* felt-piano score at −16 LUFS.

The master is sent separately because it is over GitHub's 100 MB file limit. To rebuild it, run
`./render_all.sh` then `python assemble.py` in `world_roads/golfball`. Its SHA-256 is in `SHA256.txt`.

`golfball_small.mp4` (62 MB, HEVC CRF 28) is the same film, kept in the repository for reference.
**Don't upload it:** it keeps less of the faintest balls' light than the master does (see
`../../golfball/ACCURACY.md` §3).

**Notes on the upload.**
* **Upload the 4K file as it is.** The balls are drawn to scale in 1080p pixels (a ball at 800 yards is a
  third of a pixel), and a 4K upload gets YouTube's better encodes at every resolution.
* **Ask viewers** to watch full screen at 1080p or higher. See `../../golfball/ACCURACY.md` §3 for what the
  compression test found.

**A/B/C test.**

| | Thumbnail | Title |
|---|---|---|
| A | `golfball_thumb_A.png`: the dark-trees card under a magnifier, "A golf ball, 400 yards away." | You Shouldn't Be Able to See a Golf Ball This Far |
| B | `golfball_thumb_B.png`: the ball's blur spot on the cones, the ball itself a red ring one cone wide, "It doesn't get smaller. it gets fainter" | Past 160 Yards, a Golf Ball Doesn't Get Smaller. It Gets Fainter. |
| C | `golfball_thumb_C.png`: the 20/20 E on the cones with a golf ball drawn in one stroke, "160 yards: one stroke of the 20/20 E" | A Golf Ball at 160 Yards Is One Stroke of the 20/20 E |

**Description**

```
A golf ball is 1.68 inches across. At 160 yards it spans one arcminute: exactly the width of one stroke of a 20/20 letter, the finest detail normal vision is defined by. At 320 yards it's half an arcminute, one cone wide at the centre of your retina. By that logic you shouldn't be able to see it much past 160 yards. Yet against dark trees you can see a ball sitting still at 600 yards and more.

Here's why, exactly.

Your eye blurs every point of light into a spot about an arcminute wide, and each cone collects light from a patch of the world about that size. Past 160 yards the ball fits inside one cone's patch, so its image on your retina stops shrinking: it is the blur spot itself. It can only get fainter. Twice as far, a quarter of the light.

In the language of spatial frequencies, past about 300 yards the ball's recipe is flat across everything your eye can see. Nothing in it says how big the ball is, only how bright.

So whether you see it comes down to contrast. Your eye judges a spot by how much brighter or darker it is than its background, as a share of it. Lab thresholds give a simple rule: you can find a small spot as long as its contrast times its area adds up to about one cone. A dark speck can never be more than 100% darker than a bright sky, so a black ball is lost at about 310 yards. A sunlit white ball is about 30 times as bright as dark trees, so against them it holds out to roughly 1,000 yards on a clear day, if it's sitting still and you know where to look. That's why a drive can vanish against an overcast sky and turn up again on the fairway, and why you can see stars at all.

▶ Watch full screen, at 1080p or higher. The golf balls on the cards are drawn to scale for a phone held at reading distance, smaller than a pixel. Lower quality settings and small players blur them away.

0:00 A golf ball you shouldn't see
0:19 One arcminute at 160 yards
0:49 One stroke of the 20/20 E
1:24 If your cones were pixels
2:35 It doesn't get smaller. It gets fainter.
3:48 The ball's recipe of spatial frequencies
4:41 Why a dark background wins
5:56 Try it: four backgrounds, four distances
6:44 One blur spot

How it's made: every frame is rendered in code. The retina is a simulation: a cone mosaic 0.5 arcminutes apart, with the eye's own blur (Westheimer's line spread function, 3 mm pupil). The detection distances come from Blackwell's contrast thresholds as fitted by Crumey (2014), for a still ball on a clear day. Method, sources and assumptions: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/golfball/ACCURACY.md

Sources: H. R. Blackwell, J Opt Soc Am 36:624 (1946). A. Crumey, Mon Not R Astron Soc 442:2600 (2014). C. Curcio et al., J Comp Neurol 292:497 (1990). G. Westheimer, Handbook of Perception and Human Performance, vol. 1 (1986). W. Tuten et al., J Vis 18(8):6 (2018). H. Snellen, Probebuchstaben zur Bestimmung der Sehschärfe (1862).

Music: an original felt-piano score, synthesized for this video (no samples).
```

**Chapters** (from `out/timeline.json`)

```
0:00 A golf ball you shouldn't see
0:19 One arcminute at 160 yards
0:49 One stroke of the 20/20 E
1:24 If your cones were pixels
2:35 It doesn't get smaller. It gets fainter.
3:48 The ball's recipe of spatial frequencies
4:41 Why a dark background wins
5:56 Try it: four backgrounds, four distances
6:44 One blur spot
```

**Tags:** golf, golf ball, vision, eyes, retina, cones, fovea, visual acuity, 20/20, Snellen, eye chart,
contrast sensitivity, spatial frequency, Fourier, Ricco's law, Weber's law, optometry, perception

**Pinned comment idea:** "On the 'Try it' cards, which was the farthest ball you could see against each
background?"

**Link it with the vernier film.** Both films share the cone mosaic and the blur. Each can link the other
in its description and end screen.
