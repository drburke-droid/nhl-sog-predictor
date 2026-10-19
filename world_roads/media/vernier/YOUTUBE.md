# YouTube copy: the vernier film

This remake of the hyperacuity video has no quiz rounds. It shows the step, explains why you shouldn't
be able to see it, and lets you see it anyway.

Upload `vernier.mp4`:
* 6:05, 3840×2160, 30 fps, HEVC, 56 MB;
* on-screen captions, no narration;
* felt-piano score at −16 LUFS.

Ask viewers to watch full screen at 1080p or higher (see `../../vernier/ACCURACY.md` §2).

Upload the 4K file as it is. The steps are drawn in 1080p pixels, so each one is a 2×2 block of the 4K
frame. A 4K upload also gets YouTube's better encodes at every resolution. The file is HEVC because the
desk's moving film grain made H.264 about four times larger. Measured in the file itself, the plates
read 0.505, 0.251 and 0.137 px against the step-free plate 4 (drawn: ½, ¼, ⅛).

**A/B/C test.**

| | Thumbnail | Title |
|---|---|---|
| A | `vernier_thumb_A.png`: the card under a magnifier, "Half a pixel. and you can see it" | You Shouldn't Be Able to See This Line. But You Can. |
| B | `vernier_thumb_B.png`: the retina under a microscope with the stepped line traced in red, "Smaller than one cone. you still see it" | Your Eye Sees Steps Smaller Than Its Own Cones |
| C | `vernier_thumb_C.png`: retina and TV pixels side by side, "Your eye has pixels too." | Your Eye Has Pixels. So How Can You See Half of One? |

**Description**

```
There's a step in the middle of a line, half a pixel high. On your retina it's narrower than a single cone, the "pixels" your eye sees with. By that logic you shouldn't be able to see it. You can.

Then we make it smaller: a quarter of a pixel (about a third of a cone), an eighth (about a sixth of a cone), and a line with no step at all, for comparison.

How? Your eye blurs the line a little, so its shadow falls across several cones. Each cone reports how much light it gets, and the pattern of readings shifts when the line moves, even by a fraction of a cone. Your brain reads every column of cones along the line and pools the answers. Gerald Westheimer named this hyperacuity in 1975. Most people can see a step of 5 to 10 arcseconds, a sixth to a third of the gap between two cones.

It's why a vernier scale works, and why eye doctors use Amsler grids. If straight lines ever look bent or wavy to you, see your optometrist.

▶ Watch full screen, at 1080p or higher. The steps are drawn to a fraction of a pixel; lower quality settings and small players blur them away. Sizes on the retina assume a phone held at reading distance.

0:00 A step you shouldn't see
0:22 The line on your retina
1:11 Back to real size
1:34 Smaller and smaller
2:42 How you see it
3:47 How screens draw half a pixel
4:19 Hyperacuity, in numbers
4:54 Vernier scales and Amsler grids
5:46 How small you saw

How it's made: every frame is rendered in code. The lines are drawn with exact pixel coverage in linear light, and we tested that the steps survive YouTube's compression. The retina is a simulation: a cone mosaic 0.5 arcminutes apart, with the eye's own blur (Westheimer's line spread function). Method, sources and the compression test: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/vernier/ACCURACY.md

Sources: G. Westheimer, Invest Ophthalmol 14:570 (1975). G. Westheimer & S. McKee, Vision Res 17:941 (1977). C. Curcio et al., J Comp Neurol 292:497 (1990). M. Amsler, Ophthalmologica 114:248 (1947). P. Vernier, La construction, l'usage, et les propriétés du quadrant nouveau de mathématiques (1631).

Music: an original felt-piano score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 A step you shouldn't see
0:22 The line on your retina
1:11 Back to real size
1:34 Smaller and smaller
2:42 How you see it
3:47 How screens draw half a pixel
4:19 Hyperacuity, in numbers
4:54 Vernier scales and Amsler grids
5:46 How small you saw
```

**Tags:** hyperacuity, vernier acuity, vision, eyes, retina, cones, fovea, pixels, sub-pixel,
anti-aliasing, optometry, Amsler grid, macular degeneration, vernier scale, perception, Westheimer

**Pinned comment idea:** "Which was the smallest step you could see: plate 1, 2 or 3? (Plate 4 had
no step at all.)"

**The old video.** If the quiz version is still up, unlist it rather than delete it, and link this one
in its description, so existing links keep working.
