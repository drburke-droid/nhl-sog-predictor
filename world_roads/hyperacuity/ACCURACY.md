# Accuracy and sources: "Your eyes see smaller than their own pixels" (hyperacuity)

## 1. What "1 pixel" means on screen

Every offset in the video is in pixels of a **1080p frame**. The video is rendered and uploaded at 4K
(3840×2160), so "1 pixel" is a 2×2 block of the upload. Watched at 1080p, it is exactly one pixel.

**Drawing.** Stimuli are drawn with exact area coverage (`ui.Patch.rect`) or a one-pixel signed-distance
ramp (`ui.Patch.sdf`, for curves and tilts). Coverage is blended in **linear light**, then encoded to
sRGB. A bar moved by ¼ px really has its light centred ¼ px over, on a screen that follows the sRGB
curve. A screen with a different gamma moves that centre slightly.

**YouTube's processing** (`codec_check.py`). YouTube scales the video to the player and re-encodes it.
To test whether sub-pixel offsets survive, a frame from rounds 4–6 was:
1. scaled from 4K to 1080p, 720p and 480p;
2. encoded at streaming-like bitrates;
3. decoded, with the light centre of each bar measured.

The aligned pairs read **≤ 0.004 px** in every case. The shifted pair read:

| Encode | ½ px | ¼ px | ⅛ px |
|---|---:|---:|---:|
| 4K master (x264, CRF 14) | 0.501 | 0.254 | 0.125 |
| 4K VP9, 8 Mb/s | 0.501 | 0.253 | 0.124 |
| 1080p VP9, 1.5 Mb/s | 0.496 | 0.245 | 0.133 |
| 1080p H.264, 2.5 Mb/s | 0.501 | 0.242 | 0.138 |
| 1080p AV1, 1 Mb/s | 0.501 | 0.242 | 0.138 |
| 720p VP9, 0.8 Mb/s | 0.484 | 0.246 | 0.150 |
| 480p VP9, 0.4 Mb/s | 0.537 | 0.285 | 0.171 |

The test can't model YouTube's own encoders exactly. Lower quality settings and small players shrink
each pixel on screen and make the later rounds harder, so the rules ask for full screen at 1080p or
higher.

## 2. The numbers on screen

**Arcseconds** (″) are 1/3600 of a degree. The angle a pixel covers is pitch ÷ viewing distance ×
206,265″.
* **A phone pixel:** a 460 ppi phone (55.2 µm pitch) at 30 cm gives 38″ per screen pixel.
* **A video pixel on that phone:** a full-screen 16:9 video on a 19.5:9 phone fills its 1,179-pixel
  height, so one 1080p video pixel is 1.09 phone pixels, about **41″**. The video says "about 40″", and
  the results ladder uses 40″ per pixel.
* **Other screens:** 31–95″ per pixel, for example a 1080p monitor at 60 cm or a 4K TV at 2.5 m.

| Claim | Value used | Source |
|---|---|---|
| Normal (20/20, 6/6) acuity | a letter 5′ tall with 1′ strokes and gaps | Snellen 1862 |
| Foveal cone spacing | ≈ 0.5′ (≈ 30″, ≈ 2.4 µm) at the very centre | Curcio et al. 1990, *J Comp Neurol* 292:497 |
| Vernier acuity | 5–10″ is typical of attentive observers; 2–5″ when practised | Westheimer & McKee 1977, *Vision Res* 17:941; Hu et al. 2021, *Front Neurosci* 15:714843 |
| The term "hyperacuity" | Gerald Westheimer, 1975 | Westheimer 1975, *Invest Ophthalmol* 14:570 |
| The eye's blur (line spread function) | L(x) = 0.47·e^(−3.3x²) + 0.53·e^(−0.93\|x\|), x in arcmin, 3 mm pupil | Westheimer 1986, *Handbook of Perception and Human Performance* Vol. 1; Wandell 1995, *Foundations of Vision* ch. 2 |
| Detecting tilt | ≈ 0.3–0.5° near horizontal or vertical; 1.5–4× worse on diagonals (the oblique effect) | Regan & Beverley 1985, *JOSA A* 2:147; Li, Peterson & Freeman 2003, *J Neurophysiol* 90:204; Appelle 1972 |
| Detecting a bend | thresholds of roughly 0.2–0.3° between curve and chord | Watt 1982, *Vision Res* 22:449; Wilson 1985, *JOSA A* 2:1191; Kramer et al. 1996 |
| The vernier scale | Pierre Vernier, *La construction, l'usage, et les propriétés du quadrant nouveau de mathématiques*, Brussels 1631 | MacTutor (St Andrews) |
| The Amsler grid | Marc Amsler, 1947, *Ophthalmologica* 114:248 | |
| Home monitoring for AMD | preferential hyperacuity perimetry, FDA-cleared (510(k) K091579, 2009); HOME study | Chew et al. 2014, *Ophthalmology* 121:535 |
| Amblyopia | in strabismic amblyopia, vernier acuity is lost far more than grating or letter acuity; in anisometropic amblyopia, roughly in proportion | Levi & Klein 1982, *Nature* 298:268; McKee, Levi & Movshon 2003, *J Vis* 3:380 |

**Ranges between people.**
* **Cone spacing:** peak foveal cone density averages about 199,000 per mm², but ranges 100,000–324,000
  between eyes. Spacing is therefore about 0.4–0.7′.
* **Vernier thresholds:** these rise with untrained observers (often 10–15″), wider gaps, low contrast,
  and viewing away from the centre of vision. The record is under 1″.

These values were checked against the papers' abstracts, reviews and textbooks listed above.

## 3. The test rounds and quizzes

**The six rounds.** Four bar pairs: bars 4 px wide and 110 px tall, with a 4 px gap. In three pairs the
bars line up exactly; in one, the lower bar is shifted by 4, 2, 1, ½, ¼ or ⅛ px. The answers (C, A, D,
B, D, A) and their directions are fixed.

**The ladder.** On a phone at 30 cm, the rounds span about 160″ down to 5″. The cone-spacing line (30″)
falls between round 3 (≈40″) and round 4 (≈20″).

**The quizzes.**

| Quiz | The odd one | What that means |
|---|---|---|
| Step | a 1 px step in a 1,320 px line | |
| Bend | a 2 px bow (sagitta) in a 1,300 px line | 0.35° between curve and chord at the ends, just above the published thresholds |
| Circle | a circle 1.5% wider than tall | 5.7 px on a 380 px diameter |
| Tilt | a 380 px line tilted 0.75° | its ends 2.5 px off |
| Tilt, diagonal | the same 0.75° tilt at 45° | |

**The magnifiers** show the real 1080p pixels, nearest-neighbour, with a pixel grid. Exaggerated
drawings (10×, 12×) are labelled as such.

## 4. The simulated retina

* **Mosaic:** a hexagonal lattice with 0.5′ spacing and 3% positional jitter.
* **Signal:** each cone's signal is the eye's line spread function (above) at its centre. Cone
  apertures and noise are left out.
* **Estimate:** each half-line's position is the least-squares fit of the line spread function to its
  cones' signals.

Without noise this recovers the shift exactly: the simulation shows that the **information is
there**. What limits real observers is noise in the photoreceptors and neurons, which is why
thresholds are a few arcseconds and not zero.

The caption "a little blur helps" refers to sampling: a line thinner than a cone could move inside a
single cone without changing any signal.

## 5. The clinical section

* **Amsler grid:** the distortion is simulated (a smooth local bulge).
* **Home monitors:** no device is named. "Some at-home monitors for macular degeneration are built on
  this same hyperacuity" refers to preferential hyperacuity perimetry.
* **What the video advises:** see an optometrist if straight lines look bent. It is not a diagnosis.
