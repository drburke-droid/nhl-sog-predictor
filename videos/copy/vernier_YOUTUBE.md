# YouTube copy: the vernier film

This remake of the hyperacuity video has no quiz rounds. It shows the step, explains why you shouldn't
be able to see it, and lets you see it anyway.

**Upload `vernier_4k_master.mp4`:**
* 6:19, 3840×2160, 30 fps, HEVC CRF 24, 150 MB;
* on-screen captions, no narration;
* felt-piano score at −16 LUFS.

The master is sent separately because it is over GitHub's 100 MB file limit. To rebuild it, run
`python assemble.py` in `world_roads/vernier`. Its SHA-256 is in `SHA256.txt`.

`vernier_small.mp4` (53 MB, HEVC CRF 28) is the same film, kept in the repository for reference:
* it looks the same at 1:1;
* measured in the file itself, the plates read 0.511, 0.230 and 0.157 px against the step-free plate 4
  (drawn: ½, ¼, ⅛). The master reads 0.516, 0.240 and 0.143 px.

Don't upload `vernier_small.mp4`. When it was re-encoded to 1080p H.264 at a low bitrate, the ⅛ px step
sometimes grew to about 0.4 px. The CRF 24 master held it near the true size (see
`../../vernier/ACCURACY.md` §2).

**Notes on the upload.**
* **Upload the 4K file as it is.** The steps are drawn in 1080p pixels, so each one is a 2×2 block of
  the 4K frame. A 4K upload also gets YouTube's better encodes at every resolution.
* **Why HEVC.** The desk's moving film grain made H.264 about four times larger at the same quality.
* **Ask viewers** to watch full screen at 1080p or higher.

**A/B/C test.**

| | Thumbnail | Title |
|---|---|---|
| A | `vernier_thumb_A.png`: the card under a magnifier, "Half a pixel. and you can see it" | You Shouldn't Be Able to See This Line. But You Can. |
| B | `vernier_thumb_B.png`: the retina under a microscope with the stepped line traced in red, "Smaller than one cone. you still see it" | Your Eye Sees Steps Smaller Than Its Own Cones |
| C | `vernier_thumb_C.png`: retina and TV pixels side by side, "Your eye has pixels too." | Your Eye Has Pixels. So How Can You See Half of One? |

**Description**

```
There's a step in the middle of a line, half a pixel high. On your retina it's narrower than a single cone, the "pixels" your eye sees with. If each cone simply switched on or off like a pixel, the step would slip between the rows and the same cones would be dark on both sides of it. By that logic you shouldn't be able to see it. You can.

Then we make it smaller: a quarter of a pixel (about a third of a cone), an eighth (about a sixth of a cone), and a line with no step at all, for comparison.

How? Your eye isn't perfectly sharp: it blurs the line a little, so its shadow falls across several rows of cones. And a cone isn't an on-off pixel: it reports how much light it gets. The pattern of readings shifts when the line moves, even by a fraction of a cone. Your brain reads every column of cones along the line and pools the answers. Gerald Westheimer named this hyperacuity in 1975. Most people can see a step of 5 to 10 arcseconds, a sixth to a third of the gap between two cones.

It's why a vernier scale works, and why eye doctors use Amsler grids. If straight lines ever look bent or wavy to you, see your optometrist.

▶ Watch full screen, at 1080p or higher. The steps are drawn to a fraction of a pixel; lower quality settings and small players blur them away. Sizes on the retina assume a phone held at reading distance.

0:00 A step you shouldn't see
0:22 If your cones were pixels
1:20 Back to real size
1:43 Smaller and smaller
2:51 How you see it
4:01 How screens draw half a pixel
4:33 Hyperacuity, in numbers
5:08 Vernier scales and Amsler grids
6:00 How small you saw

How it's made: every frame is rendered in code. The lines are drawn with exact pixel coverage in linear light, and we tested that the steps survive YouTube's compression. The retina is a simulation: a cone mosaic 0.5 arcminutes apart, with the eye's own blur (Westheimer's line spread function). Method, sources and the compression test: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/vernier/ACCURACY.md

Sources: G. Westheimer, Invest Ophthalmol 14:570 (1975). G. Westheimer & S. McKee, Vision Res 17:941 (1977). C. Curcio et al., J Comp Neurol 292:497 (1990). M. Amsler, Ophthalmologica 114:248 (1947). P. Vernier, La construction, l'usage, et les propriétés du quadrant nouveau de mathématiques (1631).

Music: an original felt-piano score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 A step you shouldn't see
0:22 If your cones were pixels
1:20 Back to real size
1:43 Smaller and smaller
2:51 How you see it
4:01 How screens draw half a pixel
4:33 Hyperacuity, in numbers
5:08 Vernier scales and Amsler grids
6:00 How small you saw
```

**Tags:** hyperacuity, vernier acuity, vision, eyes, retina, cones, fovea, pixels, sub-pixel,
anti-aliasing, optometry, Amsler grid, macular degeneration, vernier scale, perception, Westheimer

**Pinned comment idea:** "Which was the smallest step you could see: plate 1, 2 or 3? (Plate 4 had
no step at all.)"

**The old video.** If the quiz version is still up, unlist it rather than delete it, and link this one
in its description, so existing links keep working.
