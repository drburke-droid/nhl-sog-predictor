# Accuracy and sources: the golf-ball film ("You shouldn't be able to see a golf ball this far")

This film uses the look, the cone mosaic and the eye's blur from the vernier film (`../vernier`). Every
number on screen is computed in `physics.py`; `python physics.py` prints the tables below.

## 1. The ball and the angle

**The ball:** 1.680 in (42.67 mm) across. This is the smallest diameter the Rules of Golf allow (the
Equipment Rules' ball size: "not less than 1.680 inches (42.67 mm)").

**Its angle:** θ = 2·atan(r / d).

| Distance | Angle | Cone spacings* | On the retina |
|---:|---:|---:|---:|
| 100 yd | 1.60′ (96″) | 3.2 | 7.8 µm |
| **160.4 yd (146.7 m)** | **1.00′ (60″)** | **2.0** | **4.9 µm** |
| 200 yd | 0.80′ | 1.6 | 3.9 µm |
| 250 yd | 0.64′ | 1.3 | 3.1 µm |
| **320.8 yd (293.4 m)** | **0.50′ (30″)** | **1.0** | **2.4 µm** |
| 400 yd | 0.40′ | 0.8 | 1.9 µm |
| 640 yd | 0.25′ | 0.5 | 1.2 µm |

\* Foveal cone spacing 0.5′ (Curcio et al. 1990, *J Comp Neurol* 292:497, peak density about 199,000/mm²);
retinal scale 4.85 µm per arcminute (Drasdo & Fowler 1974). Both are the same as in the vernier film.

**Why 160 yards and not 166.** One arcminute is reached at 160.4 yards for a 1.680-inch ball. A distance of
166.5 yards corresponds to 0.963′. The film anchors the paradox on exactly 1′ because that is the stroke
width of a 20/20 letter. In the blur model below, one cone's patch of the world is 0.957′ wide at half
height, which the ball fits inside from 167.6 yards on. "About 160–170 yards" is therefore right either way.
The film says 160.

## 2. The 20/20 E

* **Size:** a 20/20 (6/6) letter is 5′ tall, with strokes and gaps of 1′ (Snellen 1862). The E is drawn on a
  5 × 5 grid: a spine and three arms, all one stroke wide.
* **Real size:** on the card it is 7.32 px tall (5′ ÷ 0.683′ per px; see §3). The golf ball beside it is
  1.47 px across: a golf ball at 160 yards.
* **On the retina:** the E is 10 cone spacings tall and each stroke is 2 ("1′: two cones", "5′: ten cones").

## 3. Sizes on screen ("drawn to scale")

All sizes assume a phone held at reading distance (30 cm) with the video full screen. One 1080p pixel of
the video is then 41″ = 0.683′; see `../hyperacuity/ACCURACY.md` §2. On screen this assumption is the
footnote "Sizes for a phone held at reading distance."

| Ball at | Diameter | Area |
|---:|---:|---:|
| 100 yd | 2.35 px | 4.33 px² |
| 250 yd | 0.94 px | 0.69 px² |
| 400 yd (the opening card) | 0.59 px | 0.27 px² |
| 500 yd | 0.47 px | 0.17 px² |
| 800 yd | 0.29 px | 0.068 px² |

**How the dots are drawn.** Each ball is a disk with exact area coverage: 12 × 12 samples per 4K pixel,
blended in linear light, then sRGB. A ball smaller than a pixel therefore lights its pixel by its share of
the area. On a screen too, a dot below one pixel gets fainter rather than smaller. The cards carry no
grain.

**The cards' contrast.** Each card has the luminance ratio of its scene (§5): the white ball has
Y = 0.88 in linear light. The background colours are chosen to have these luminances:

| Scene | Background Y | Ball Y | Weber contrast |
|---|---:|---:|---:|
| dark trees | 0.0293 | 0.88 | +29.0 |
| the fairway | 0.146 | 0.88 | +5.0 |
| blue sky | 0.349 | 0.88 | +1.5 |
| overcast sky | 0.801 | 0.44 | −0.45 |

**The ticks and crosses** under the demo cards are not about the screen. Each says whether a still ball at
that distance is seen on a real sunny course: its distance against the clear-day limit in §7 (980, 620,
380 and 200 yards). The distances (100, 250, 500, 800 yards) were chosen so that no ball is within 18% of a
limit.

**A screen is dimmer than a sunny day.** A phone shows the dark-trees card at about 10 cd/m², not 500.
At that light level the eye's threshold for small spots is roughly 2.5 times higher (§5, R at 10 cd/m²
vs 500 cd/m²). The faintest balls on the cards are therefore harder to see than the same balls on a real
course. The film does not claim that every ball on the cards is visible.

**Does it survive YouTube?** (`codec_check.py`)

**The method.**
1. Two seconds of the opening card and of each demo card, exactly as the film shows them (moving film grain
   on the desk around each card), are encoded as the 4K master.
2. Each is then scaled and re-encoded the ways YouTube might stream it:
   * two-pass VP9, like YouTube's own encodes;
   * a fast one-pass VP9, as a worst case;
   * H.264;
   * AV1.
3. Each is decoded at six moments.
4. Each ball is measured as the extra light in a 7×7-pixel window around it, against a ring around the
   window, and divided by what was drawn. 1.00 means the encode kept all of the ball's light.

**Caveats.** For the faintest balls on the mid-grey and pale cards, a reading can be off by ±0.5 from
encoder noise in the window, so readings above 1 occur. A reading of 0.00 means the encoder flattened the
ball.

Medians, for the balls the model marks as seen (✓). The opening card is the 400-yard ball against trees.

| Encode | Opening card | Trees 800 yd | Fairway 500 yd | Blue sky 250 yd | Overcast 100 yd |
|---|---:|---:|---:|---:|---:|
| 4K master (HEVC, CRF 24) | 0.95 | 0.69 | 1.87 | 0.71 | 0.95 |
| 4K VP9, two-pass, 12 Mb/s | 0.96 | 0.70 | 1.87 | 0.75 | 0.96 |
| 1080p VP9, two-pass, 2 Mb/s | 0.59 | 0.49 | 1.36 | 0.69 | 0.94 |
| 1080p VP9, two-pass, 1.2 Mb/s | 0.55 | **0.00** | 1.48 | 0.55 | 1.00 |
| 1080p VP9, fast one-pass, 1.5 Mb/s | 0.32 | **0.00** | **0.00** | 0.09 | 0.50 |
| 1080p H.264, 2.5 Mb/s | 0.74 | 0.50 | 1.33 | 0.57 | 1.00 |
| 1080p AV1, 1.2 Mb/s | 0.54 | 0.69 | 0.91 | 1.42 | 0.95 |
| 720p VP9, two-pass, 0.8 Mb/s | 0.18 | **0.00** | 1.48 | 0.85 | 1.03 |

Every frame and every ball is in `out/codec_check.json`.

**What it means.**
* **4K:** keeps the balls.
* **1080p with a good encoder:** keeps every ball the model says you'd see on a course, with about half of
  the faintest one's light.
* **Low bitrates, 720p, or a fast encoder:** can flatten the faintest balls (800 yards against trees, 500
  yards on the fairway).

The description therefore asks for full screen at 1080p or higher. The ticks and crosses do not depend on
any of this.

**The final files.** Measured in the files themselves, at three moments of each card:
* **The 166 MB master (CRF 24):** the opening ball keeps 0.93–1.06 of its light. Every ball the model
  marks as seen keeps 0.42–1.00 (the faintest is 800 yards against trees).
* **The 62 MB copy (CRF 28),** kept in the repository for reference (`media/golfball/golfball_small.mp4`):
  the opening ball keeps 0.71–0.74. In one frame of three, the 800-yard ball against trees reads 0.

Upload the master.

## 4. The eye's blur, and "one cone's patch"

**The blur.** Westheimer's line spread function for a 3 mm pupil, the same one the vernier film uses:
L(x) = 0.47·e^(−3.3x²) + 0.53·e^(−0.93|x|), x in arcmin (Westheimer 1986; Wandell 1995, *Foundations of
Vision*, ch. 2). Its point spread function follows exactly. The Gaussian term projects from a 2D Gaussian,
and the exponential term from a K₀ (modified Bessel) term:
PSF(r) = 0.301·e^(−3.3r²) + 0.0982·K₀(0.93r), with unit volume.

Images are computed in the frequency domain, as the object's spectrum times the optical transfer function
(the LSF's Fourier transform). The film calls this spot "about an arcminute wide": the LSF's full width at
half maximum is 1.08′.

**A cone's aperture.** A disk 0.5′ across. Foveal inner segments nearly fill the 0.5′ spacing, and Tuten et
al. 2018 (*J Vis* 18(8):6) use about 0.5′. Each cone's reading is the image averaged over its aperture.

**One cone's patch of the world.** A point seen through the optics and one cone's aperture gives a spot
0.957′ wide at half height. The film draws it as a 1′ circle labelled "about 1′".

**The image stops shrinking.** Here is the ball's image through a cone's aperture: its width at half
height, and its peak per unit contrast.

| Ball at | Ball | Image width | Image peak | Total light (vs 160 yd) |
|---:|---:|---:|---:|---:|
| 160 yd | 1.00′ | 1.27′ | 0.259 | 1 |
| 320 yd | 0.50′ | 1.04′ | 0.085 | ¼ |
| 640 yd | 0.25′ | 0.98′ | 0.023 | 1/16 |
| 1,280 yd | 0.13′ | 0.96′ | 0.0059 | 1/64 |
| (a point) | 0 | 0.957′ | → 0 | → 0 |

* **"Twice as far, a quarter of the light":** the total light is exactly proportional to the ball's area.
  The plot's labels (¼, 1/16, 1/64) are total light, not peak heights.
* **Peak heights:** these fall a little more slowly at first (0.33 from 160 to 320 yards), because the
  160-yard image is still a little wider.
* **"Hardly does":** the image narrows by 25% from 160 yards to infinity, and by 8% beyond 320 yards. The
  ball itself shrinks to nothing.

## 5. The micrographs

**The mosaic.** As in the vernier film: a hexagonal lattice 0.5′ apart with 4.5% jitter, each cone drawn as
its Voronoi cell.

**Pixel logic** (the retina section) is a stated hypothetical ("Suppose each cone works like a pixel"):
* each cone is on or off by whether the image (with no blur) covers its centre;
* the ball is placed 0.38 cone spacings from one cone. Its cone distances are 0.38, 0.65, 0.81, 0.87 and
  1.23 spacings, so it covers 4 cone centres at 160 yards (radius 1.003), 1 at 320 yards (0.501) and none
  at 640 yards (0.251). The margins are 0.12 spacings or more, larger than the lattice jitter.

**Brightness.**
* **The ball:** cones are drawn in true proportion to their light (linear light, then sRGB), with a fixed
  "exposure" of 2.8 for the trees' background. A cone that gets 18% more light is drawn 18% brighter.
* **The E:** drawn with the vernier film's tone curve (light^1.8 between a dim and a bright cone), so that a
  black-on-white pattern reads clearly. Its blurred image has about 16% modulation across the strokes.
* **"The same exposure"** (the three fields in the dark-background section): each field's exposure is set
  so the background reads the same. This mimics the eye's own light adaptation (Weber's law) and leaves
  only the change. The ball is at 400 yards in all three. The brightest cone's change is +146% against
  trees, +7.6% against blue sky and −2.3% against overcast sky. The ball sits 0.38 spacings off a cone's
  centre; at the spot's centre the change would be +164%, +8.5% and −2.5%.

## 6. The recipe (spatial frequencies)

* **The ball's spectrum:** the Fourier transform of a uniform disk, A·2J₁(πθf)/(πθf), with A its area.
  The plot shows its magnitude on log axes, relative to the 40-yard ball's at zero frequency. This is the
  object's recipe. The eye's optics are part of the contrast sensitivity curve, so they are not applied
  twice.
* **"What your eye can see":** frequencies up to 48.8 cycles per degree. That is where the "20/20 sample"
  contrast sensitivity curve of the blog's Spatial Frequency Analyzer (`csf/spatial-freq-analyzer.html`:
  log-parabola, peak 126 at 4.5 c/deg, b = 1.10, d = 0.80) falls to a sensitivity of 1.
* **The claims:**
  * **40 and 80 yards:** the spectrum's first zero is at 18 and 36 c/deg, inside the visible range ("the
    recipe has a shape there").
  * **160 yards:** the first zero is at 73 c/deg. At 48.8 c/deg the spectrum is still at 37% of its
    zero-frequency value ("slipping past the finest ripples you can see").
  * **320 yards:** 81% at the cut-off. **640 yards:** 95%. "Flat across everything your eye can see" means
    within 20% from 320 yards on.
  * **"A quarter as strong":** the strength is proportional to area, ¼ per doubling of distance. Lifted ×4,
    the 640-yard curve lies on the 320-yard curve to within the 15% difference at the cut-off.

## 7. Seeing it: Ricco's law, Weber's law, and the rule of thumb

**The threshold model.** Crumey 2014, *MNRAS* 442:2600 (eqs. 24, 35–44): a photopic fit to Blackwell
1946, *JOSA* 36:624, table 8. The data are a 0.595′ disc, 50% detection, eight-position forced choice, and
unlimited viewing time. The threshold Weber contrast for a target of area A on background B is
C = ((R/A)^q + C∞^q)^(1/q). Here R = (1.772×10⁻⁴·B^(−1/4) + 7.167×10⁻⁵)² sr, C∞ = 0.00272, and
q = 1.146 − 0.0885·log B.

**For a target this small, only the light counts.**
* For targets far smaller than the eye's summation area, (R/A) dominates. Threshold contrast × area is then
  constant (Ricco's law), and only the total light the ball adds matters.
* Foveal Ricco areas measured with ordinary optics are 2.4–3.0′ across (Tuten et al. 2018, at about
  8 cd/m²), and smaller in daylight. Beyond 160 yards the ball is 1′ or less.

**Field factor F = 2.** Blackwell's 50% forced-choice thresholds are multiplied by 2 for confident
detection. This is Crumey's notional "typical" value; Blackwell proposed ≥ 1.62 for 90% confident seeing,
and Blackwell & Blackwell 1971 proposed 2.4 for "common-sense seeing". The range F = 1–4 is given below.

**"Contrast × area ≥ about one cone."** F·R is 0.28, 0.22, 0.21 and 0.20 arcmin² at B = 500, 2,500, 4,000
and 6,000 cd/m². One cone's cell is 0.217 arcmin² (a hexagon 0.5′ across). The rule holds to within about
30% across these backgrounds.

**The scenes.** These are examples for a clear, sunny day, not measurements of a particular course.

| | Background | Ball | Contrast |
|---|---:|---:|---:|
| dark trees | 500 cd/m² | 15,000 cd/m² | +29 |
| the fairway | 2,500 | 15,000 | +5 |
| blue sky | 6,000 | 15,000 | +1.5 |
| overcast sky | 4,000 | 2,200 | −0.45 |

* **The ball:** a white ball (reflectance about 0.8) lit by about 90,000 lx of sunlight, seen with the sun
  behind the viewer. Its disk-average luminance is (2/3)·ρE/π ≈ 15,000 cd/m² (a Lambertian sphere at
  zero phase angle).
* **Foliage:** green leaves reflect about 5% at 400 nm, 15% at 550 nm and 5–6% at 675 nm (Morris 1951).
  Canopies are darker still from self-shadowing, and a tree line mixes sunlit and shaded foliage, which
  gives about 500 cd/m².
* **The fairway:** grass at about 8% in sunlight, about 2,500 cd/m².
* **Sky:** clear-sky luminance is about 8,000 cd/m² on average and about 10⁴ near the horizon. Overcast is
  about 2,000 cd/m², brighter when thin. A ball under an overcast sky is lit only by the sky and the ground,
  so it is darker than the sky behind it.

**How far a still ball is seen** (F = 2; you know where to look):

| Background | No haze | Clear day (40 km visibility) | Hazy (20 km) | F = 1 / F = 4, no haze |
|---|---:|---:|---:|---:|
| dark trees | 1,443 yd | **980** | 833 | 2,040 / 1,020 |
| the fairway | 675 | **620** | 587 | 955 / 476 |
| blue sky | 388 | **380** | 378 | 550 / 273 |
| overcast sky | 206 | **200** | 194 | 293 / 143 |
| a black ball, any sky (C = −1) | 316 | **310** | 304 | |

* **Haze:** Koschmieder. Over a path with transmission T, the air adds 8,000·(1 − T) cd/m² (a clear-day
  horizon sky) to the ball. It adds the same to the background, unless the background is the sky itself.
  The film shows the clear-day column, rounded to 10 yards.
* **What scales:** the farthest distance goes as √(contrast ÷ F·R). Twice the contrast is √2 farther, and a
  white ball against dark trees (29×) goes about 3× farther than a black ball against the sky.
* **What's left out:**
  * **Motion:** a ball in flight is moving. Thresholds for brief or moving targets, and for not knowing
    where to look, are higher.
  * **Lighting:** the sun's direction changes the ball's luminance.
  * **Colour:** contrast against a blue sky is partly colour, and the model is achromatic.

  The film therefore says "a ball sitting still, on a clear day, if you knew where to look".

**Stars.** Nearly all are thousands of times narrower than a cone: Sirius is about 6 milliarcseconds,
5,000 times narrower than a cone's 30″. Betelgeuse, one of the largest, is about 0.05″, 600 times narrower.
The dark night sky is about 2×10⁻⁴ cd/m².

## 8. The score

The vernier film's synthesized felt piano and pad (`../vernier/music.py`, no samples), on this film's chord
plan (`score.py`), normalised to −16 LUFS.
