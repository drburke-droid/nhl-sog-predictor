# Accuracy and sources: the vernier film ("You shouldn't be able to see this line")

This is a remake of the hyperacuity video (`../hyperacuity`) without the quiz rounds. The facts and
sources are the same, and that video's `ACCURACY.md` has the full reference table.

## 1. The plates and what "a pixel" means

**The plates.** Each plate is a white card with a black line 2 px thick and 1,080 px long. The right
half of the line is lower than the left half by:

| Plate | Step | Arcseconds* | Cone spacings* | On the retina* |
|---|---:|---:|---:|---:|
| 1 | ½ px | 20.5″ | 0.68 | 1.7 µm |
| 2 | ¼ px | 10.3″ | 0.34 ("about a third") | 0.8 µm |
| 3 | ⅛ px | 5.1″ | 0.17 ("about a sixth") | 0.4 µm |
| 4 | none | 0 | 0 | 0 |

\* Assumes a phone held at reading distance (30 cm), with the video full screen. One 1080p pixel of
the video is then about 41″ (460 ppi phone, 1,179 px tall screen; see `../hyperacuity/ACCURACY.md` §2).
On screen this assumption is the footnote "Sizes for a phone held at reading distance." Other screens
give 31–95″ per pixel.

**Pixels.** All sizes are in pixels of a **1080p frame**. The film is rendered at 4K, so one pixel is a
2×2 block of the upload. Watched at 1080p, it is exactly one pixel.

**Drawing.** The line is drawn with exact area coverage (each screen pixel is shaded by how much of it
the line covers), blended in **linear light**, then encoded to sRGB. The card is printed flat:
1. no film grain or paper texture inside the card;
2. the camera holds still at exactly 1 unit = 1 pixel whenever you are asked to look.

## 2. Does the step survive YouTube? (`codec_check.py`)

The method:
1. Two seconds of each plate, exactly as the film shows it (moving film grain on the desk around the
   card), are encoded as the 4K master (HEVC, CRF 24, as uploaded).
2. Each is then scaled and re-encoded the ways YouTube might stream it, and decoded at six moments.
3. The line's dark centre is measured on each side of the step, in linear light, against each column's
   own background.
4. Plate 4 (no step) is the control: its reading, which comes from the lamp's shading on the card, is
   subtracted.

Medians (and the range over the six frames), in 1080p pixels:

| Encode | ½ px | ¼ px | ⅛ px |
|---|---:|---:|---:|
| As drawn (PNG, before any encoding) | 0.491 | 0.246 | 0.123 |
| **4K master (HEVC, CRF 24)** | **0.496** | **0.253** | **0.125** (0.125–0.128) |
| 4K VP9, 8 Mb/s | 0.496 (0.496–0.501) | 0.256 (0.256–0.265) | 0.118 (0.118–0.127) |
| 1080p VP9, 1.5 Mb/s | 0.468 (0.467–0.532) | 0.283 (0.148–0.298) | 0.183 (−0.065 to 0.192) |
| 1080p H.264, 2.5 Mb/s | 0.434 (0.434–0.546) | 0.190 (0.186–0.245) | 0.076 (0.072–0.119) |
| 1080p AV1, 1 Mb/s | 0.505 (0.505) | 0.248 (0.241–0.255) | 0.144 (0.139–0.144) |
| 720p VP9, 0.8 Mb/s | 0.272 (0.208–0.330) | 0.154 (0.018–0.556) | 0.262 (0.233–0.425) |
| 480p VP9, 0.4 Mb/s | 0.830 (0.760–0.831) | 0.431 (0.389–0.738) | 0.105 (0.049–0.448) |

The raw readings for every frame are in `out/codec_check.json`, written by `codec_check.py`.

**What it means.**
* **4K, and 1080p VP9 or AV1:** the steps survive, to within about 0.05 px in a typical frame.
* **1080p H.264:** reads the steps a little small (⅛ px as about 0.08 px), so plate 3 may be a touch
  harder on players that fall back to H.264.
* **Fast VP9 encodes:** can throw off a single frame by up to about 0.2 px. YouTube's own encodes are
  slower and better than these test settings.
* **720p and below:** the ¼ and ⅛ plates are unreliable.

The description therefore asks for full screen at 1080p or higher.

**Why the upload master is CRF 24.** The same test was run on masters made at HEVC CRF 26 and 28,
which come to about 90 MB and 56 MB. In their 1080p H.264 re-encodes at 2.5 Mb/s, the ⅛ px step
sometimes grew to 0.3–0.5 px. That would show a step bigger than the one the film names. With the
CRF 24 master it stayed near 0.1–0.2 px in every run.

**The final files.** Measured in the files themselves, against plate 4:
* the 150 MB CRF 24 master reads 0.516, 0.240 and 0.143 px;
* the 53 MB CRF 28 copy kept in the repository for reference (`media/vernier/vernier_small.mp4`) reads
  0.511, 0.230 and 0.157 px.

**A still frame.** In a still, grain-free test frame, x264 at 2.5 Mb/s gave the step-free line a false
step of about 0.3 px. The film never holds such a frame: its moving grain keeps the encoder refreshing
the card.

## 3. The numbers on screen

| Claim | Value used | Source |
|---|---|---|
| Foveal cone spacing | ≈ 0.5′ (30″), ≈ 2.4 µm at the very centre | Curcio et al. 1990, *J Comp Neurol* 292:497 |
| Retinal scale | ≈ 291 µm per degree (4.85 µm per arcminute) | Drasdo & Fowler 1974, *Br J Ophthalmol* 58:709 |
| Vernier thresholds | 5–10″ typical for attentive observers (2–5″ when practised) | Westheimer & McKee 1977, *Vision Res* 17:941 |
| "Hyperacuity" | coined by Gerald Westheimer in 1975 | Westheimer 1975, *Invest Ophthalmol* 14:570 |
| The eye's blur | line spread function L(x) = 0.47·e^(−3.3x²) + 0.53·e^(−0.93\|x\|), x in arcmin | Westheimer 1986; Wandell 1995, *Foundations of Vision* ch. 2 |
| A pencil lead | 0.5 mm at 20 m is 5.2″ | arithmetic: 0.5 mm ÷ 20 m × 206,265″ |
| The vernier scale | Pierre Vernier, Brussels, 1631 | *La construction, l'usage, et les propriétés du quadrant nouveau de mathématiques* |
| The Amsler grid | Marc Amsler, 1947 | *Ophthalmologica* 114:248 |
| Home monitors for AMD | preferential hyperacuity perimetry | FDA 510(k) K091579 (2009); Chew et al. 2014, *Ophthalmology* 121:535 |

**Ranges between people.**
* **Cone spacing:** peak foveal cone density varies about threefold between eyes, so spacing is
  roughly 0.4–0.7′. The "one cone: 2.4 µm" label and the cone counts use the 0.5′ average.
* **Thresholds:** these rise for untrained observers, low contrast, or viewing off-centre. That is why
  the film says "Some people can still see this one" for plate 3, not "everyone".

## 4. The drawings

**The eye.** A schematic, not to scale.
* The rays bend most at the cornea and a little more at the lens, about 2:1, as in a real eye, where
  the cornea supplies roughly two-thirds of the focusing power.
* The fovea is drawn on the axis.
* "The line" is a point because the sketch is a side view: the line runs into the page.

**The retina micrograph.** A simulation, not a photograph.
* **Mosaic:** cone centres on a hexagonal lattice, 0.5′ apart, with 4.5% positional jitter. Each cone
  is drawn as its Voronoi cell, with a membrane gap and a brighter inner segment.
* **Line image:** the printed line (2 px ≈ 82″ ≈ 2.7 cone spacings wide) blurred by the line spread
  function above (3 mm pupil).
* **Cone brightness:** each cone is shaded by the light at its centre, raised to the power 1.8 so the
  pattern reads clearly on screen. Cone apertures, colour types (L, M and S) and noise are left out.
* **Overlay:** the faint continuous band over the cones is the line's image itself, for reference.

**The paradox, in "pixel logic" (the retina section).** This is a deliberate hypothetical, stated on
screen as "Suppose each cone works like a pixel":
* the line's image has **no blur**: its sharp edges are 2.7 cone spacings apart;
* each cone is **on or off**: dark if the line covers its centre, light if it doesn't.

The line is placed so that the ½-pixel step (0.68 cone spacings) crosses no row of cone centres. The
rows are 0.87 spacings apart, so this is possible, and the margin is 0.024 spacings. The same three rows
are then dark on both sides of the step.
* **What decides a cone.** For this verdict a cone counts by its row's centre line. Each cone's own
  centre sits within about 0.05 spacings of it (the lattice jitter), too little to see.
* **Not every alignment hides the step.** Placed a little differently, the same step would flip a whole
  row on one side. Pixel logic would then show a full-cone step, also wrong.
* **The on-off screen** beside it uses the same rule: a screen row is off if the line covers its centre.
  The line is placed so both halves land on the same three rows, with a margin of 0.023 px.

The "How?" section then lets the real blur in: the sharp image softens into the line spread function's
image, and the cones go from on-off to graded readings.

**"A cone is a pixel of your eye."** This is an analogy for sampling. Each cone reports one number, the
light caught over its small aperture. It does not report where inside itself the light fell.

**The TV close-ups.** Illustrations of RGB subpixel stripes, one TV pixel to one cone, not a particular
screen.
* **The on-off screen** in the retina section is a hypothetical, as above.
* **The real screen** in "Screens…": each row's brightness is 1 minus the share of the row the line
  covers. The percentages on screen are that share, in linear light.
* The stripe intensities are scaled directly, not through the display's gamma, so the dimmed rows stay
  visible.

## 5. "How?" (the column of cones)

**The readings.** The dots on the pencil plot are the cones' simulated readings in one column across
the line: the light at each cone's centre.

**The "middle".** The line's middle is found by a least-squares fit of the blur profile to those
readings.
* Without noise, the fit recovers the true position exactly. Moving the line by ⅓ cone moves the fit
  by ⅓ cone (checked: 0.333).
* A simple balance point (centroid) of the readings gets within about 10%, because the blur's long
  tails run past the 13 cones in the column.

**Pooling.** Here each cone's reading has **illustrative noise** added (standard deviation 0.10, on a
0–1 scale). The noise makes each column's answer rough: its spread is about 0.22 cone. The averages of
the six columns either side of the step land within 0.011 cone of the truth.

**The brain's version.** Real observers pool along the length of the line in the visual cortex. Noise in
cones and neurons is what sets real thresholds at a few arcseconds rather than zero.

## 6. The clinical part

* **Amsler grid:** 20 × 20 squares and a centre dot, as printed. The bulge is simulated (a smooth
  radial displacement) and is labelled "simulated".
* **Home monitors:** no device is named. The line "Some home monitors for macular degeneration are
  built on hyperacuity" refers to preferential hyperacuity perimetry.
* **What the film advises:** "If straight lines ever look bent or wavy to you, see your optometrist."
  It is not a diagnosis.

## 7. The score

The music is synthesized for the film in `music.py`, with no samples:
* a felt piano, built from additive partials with string inharmonicity, two slightly detuned strings
  per note, felt damping and a soft hammer knock;
* a quiet pad;
* a synthetic room reverb.

It is normalised to −16 LUFS.
