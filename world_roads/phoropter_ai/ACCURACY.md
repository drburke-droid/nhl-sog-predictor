# Accuracy and sources: "I gave an AI a phoropter"

**This is a simulation.** Every patient in the video is a computer model. The AI has never examined a
real person. It is a demonstration of reinforcement learning, not a clinical tool, and none of its
numbers say anything about real optometrists. The numbers in the video come from `data/results_*.json`.
The tables below are filled from those files by `python report.py`.

## 1. What is real physics

| Piece | How it is done | Source |
|---|---|---|
| Adding and subtracting lenses | Power vectors: M = S + C/2, J0 = −(C/2)cos 2α, J45 = −(C/2)sin 2α | Thibos, Wheeler & Horner (1997), *Optom Vis Sci* 74:367 |
| Leftover error behind the phoropter | Residual = true correction − lens in the phoropter, as power vectors | same |
| Jackson cross-cylinder | ±0.25 D JCC = a pure J vector of length 0.25 D at its minus axis (M = 0) | same |
| Too much minus can still look clear | A positive residual M is cleared by accommodating, up to the patient's amplitude | standard optics |
| Blur size | Blur strength B = √(M² + J0² + J45²), the circle of least confusion in dioptres | Thibos et al. (1997) |
| What the patient sees (video only) | Fourier optics: circular pupil, wavefront = residual defocus and astigmatism, 555 nm, Stiles–Crawford apodisation ρ = 0.05 mm⁻². The PSF is convolved with a letter chart drawn at 0.22 arcmin per pixel. | Goodman, *Introduction to Fourier Optics*; Applegate & Lakshminarayanan (1993), *JOSA A* 10:1611 |
| Oblique axes on screen | TABO notation (counter-clockwise as the examiner faces the patient), mirrored for the patient's own view | TABO convention |

## 2. What is a model (assumptions, not fitted to data)

These choices were made before training and were not tuned to make the AI look good.

**Visual acuity.** MAR = √(MAR0² + (0.75 · pupil_mm · B)²) arcmin. This is a geometric blur-circle model.
The constant 0.75 was set so that a 4 mm pupil gives the usual clinical rule of thumb for uncorrected
myopia: −0.50 D ≈ 20/36, −1.00 D ≈ 20/63, −2.00 D ≈ 20/120. At large blur it is a little optimistic;
for example −3.00 D gives 20/180, where clinical tables often say 20/300.

**The rendered chart vs. the acuity model.** The patient's view in the video uses wave optics, while
the patient's answers use the acuity formula above. They agree to about one Snellen line. The
rendered view tends to look slightly blurrier, partly because it has no neural sharpening. The
rendered view also ignores higher-order aberrations and chromatic blur.

**The population** (`patients.py`):
* Age is uniform from 8 to 80.
* Spherical equivalent ~ N(−1.0, 2.0) D under 40, and N(−0.5, 2.0) + 0.02 D/yr over 40, clipped to −8…+6.
* Cylinder ~ Gamma(1.2, 0.5) D, capped at 3.5. With-the-rule is commoner when young and
  against-the-rule when older.
* True prescriptions are quantised to 0.25 D and 1°.
* Best-corrected acuity is logMAR ~ N(−0.08, 0.06), worsening slowly after 40. 8% have reduced
  potential (logMAR 0.2–0.5).
* Pupil: 4.8 mm at 20, 0.025 mm smaller per year (±0.4).
* Accommodation amplitude: Hofstetter's average, 18.5 − 0.3 × age (±1 D).

**The autorefractor starting point.**
* Spherical equivalent error ~ N(μ, 0.30) D, where μ = −0.25 D under 40 (instrument myopia),
  −0.10 D at 40–55, and 0 over 55.
* Each J component has error ~ N(0, 0.10) D.
* 12% are unreliable readings: SE σ = 0.9 D, J σ = 0.3 D.
* Rounded to 0.25 D and 1°.

**How patients answer** (`env.py`):
* "1 or 2?" compares the two acuities, plus noise σ ~ U(0.015, 0.05) logMAR.
* Inside a "they look the same" band τ ~ U(0.02, 0.05), they answer "the same".
* Some patients think extra minus looks crisper: up to 0.20 logMAR per dioptre in its favour.
* 0–5% of answers are random lapses.
* Chart reading adds noise σ = 0.05 logMAR and reports the nearest 0.1-logMAR line.
* Accommodation is instant and exact (up to the amplitude).

**The letter chart (attempt 3 and its scripts, `chart.py`).** Attempts 1–2 heard a whole line
("20/25"). From attempt 3 on, the chart is a projector chart read letter by letter.
* Lines run from logMAR 1.0 to −0.3 in 0.1 steps. Small lines have 5 letters; big lines have as many
  as fit 110 arcmin of screen, one letter-width apart (1–4).
* Each letter is recognised with probability 0.1 + 0.9 / (1 + exp(−(L − T)/0.04)). Here L is the
  line's logMAR, T is the patient's blur-limited acuity jittered by N(0, 0.02) per reading, and 0.1 is
  guessing among 10 Sloan letters. At their threshold they get 55%; one line bigger, 93%; one line
  smaller, 17%.
* "Read the smallest line you can": the patient reads down and stops at the first line where they get
  half or fewer. The answer is the line before it and the letters missed on it ("20/25, missed 2").
  The examiner then isolates that line.
* "Read this line" returns letters correct out of the number on the line.
* "Line smaller / larger" moves the screen one line. It costs the same chair time as a lens click.
* The letter-by-letter score is line + 0.02 per missed letter on it, which is what the scripts compare.
* The slope and the jitter are assumptions, chosen to give ≈ ±0.05 logMAR test–retest, as for a
  real letter chart.

**Not modelled.** Slow or lagging accommodation, latent hyperopia, fatigue, the duochrome test,
binocular balance, the second eye, near adds, ±0.50 D cross-cylinders for poor responders,
retinoscopy, and the patient's habitual glasses.

**"True Rx"** is the full distance correction with accommodation relaxed.

**The error measure** (`optics.rx_error`) is the length of the vector (spherical-equivalent error,
leftover cylinder): √(ΔM² + (2ΔJ0)² + (2ΔJ45)²).
* The leftover cylinder counts in full dioptres. A 0.25 D cylinder error counts 0.25 D, and an axis
  error of d° on cylinder C leaves 2·C·sin(d) dioptres of cylinder.
* "Exact" means under 0.13 D (half a click). "Within 0.25 D" means at most one click.
* Every number in the video and in RESULTS.md uses this measure, except the one scoreboard labelled
  "scored the way attempt 1 was trained".

**Attempt 1 was trained on the plain power-vector distance** √(ΔM² + ΔJ0² + ΔJ45²), the usual
"dioptric distance". It counts leftover cylinder at half weight (|ΔJ| = C_res/2). Under it:
* a 0.25 D cylinder error counted 0.125 D, which was still "exact";
* perfectly fixing every test patient's axis would have improved the average score by only 0.02 D.

The AI learned never to touch the axis. The video tells this story. Attempt 2 is identical except that
it is trained on the full-cylinder measure.

## 2b. The two-eye exam (attempt 4, `env2.py`, `clinicians2.py`)

The last comparison uses a fuller exam, prompted by the optometrist's own experience.

**Two eyes.** The left eye is correlated with the right:
* spherical equivalent differs by N(0, 0.25) D (5% of patients: N(0, 1.0) D);
* cylinder differs by N(0, 0.25) D;
* the axis is about the mirror image (180 − axis ± 8°).

Each eye has its own autorefractor error. The young-eye minus bias is shared (correlation 0.54).
Acuity is nearly equal in the two eyes, except 5% of patients with one weaker eye.

**Yoked accommodation.** With one eye occluded, the open eye accommodates for itself. With both
open, both accommodate by the smaller of the two eyes' needs.

**Proximal accommodation** (assumption, informed by instrument myopia). With one eye occluded behind
the phoropter, accommodation will not relax below a floor:
* 0–0.50 D, scaled down as amplitude falls below 4 D;
* fog relaxes the floor (gone at 0.50 D of fog);
* with both eyes open only 30% of it remains.

So monocular endpoints drift toward minus. This is the over-minus that binocular balancing and a
binocular plus check are meant to catch. Without it, the model gave no reason to balance at all.

**Line size and judgement** (assumption, informed by clinical experience).
* Patients see one projected line, not the whole chart.
* Each "1 or 2" answer is judged on that line. The real clarity difference is scaled by a gain that
  depends on δ, the line's size above the patient's threshold (logMAR):

  | Judgement | Gain peaks at | Fall-off width (bigger / smaller letters) | Example |
  |---|---|---|---|
  | Sphere and balance | δ = −0.05 | 0.35 / 0.20 | 20/200 with 20/63 vision: gain 0.08 |
  | Cross-cylinder | δ = +0.15 | 0.35 / 0.25 | ~1.5 lines bigger than threshold: gain 1.0 |

**Ocular dominance.** 70% of patients are right-eye dominant. In the prism-split balance the dominant
eye looks 0.02 logMAR clearer than it is.

**The two-eye scripts.**
* **Textbook.** For each eye in turn:
  1. Fog +1.00, then MPMVA by chart with a one-letter tolerance.
  2. Cylinder search if there is no cylinder.
  3. Cross-cylinder one line bigger than best: axis, power, axis.
  4. MPMVA re-check, then an isolated-line +0.25 check.

  Then the binocular steps:
  1. Both eyes +0.75 on the 20/40 line, prism-split.
  2. +0.25 to the clearer eye until equal. A flip takes the last +0.25 back.
  3. Prism out, then binocular MPMVA (stop at the first −0.25 that gains no letters).
* **The optometrist's routine.** As described by the optometrist (text in `clinicians2.py`). The one
  gap, a cylinder search when there is none, was filled the standard way.
  * At the optometrist's request it ends with a **binocular MPMVA**: after the balance, the +0.50 comes
    off both eyes a quarter at a time with a chart reading each time. It keeps the most plus lens that
    gives the most letters, using the optometrist's own rule (stop when no more letters come).
  * Versions tried on 3,000 two-eyed patients (within 0.25 D, per eye):

    | Ending | Within 0.25 D |
    |---|---|
    | −0.50 OU and a look at 20/20 | 32.9% |
    | −0.50 OU and a 20/20 check that adds minus if needed | 39.3% |
    | Binocular MPMVA (the version used) | 43.9% |
    | Binocular MPMVA with a one-letter tolerance | 45.4% |
    | Monocular MPMVA re-checks instead | 38.9% |
    | Textbook, for reference | 46.2% |

**Attempt 4** (`runs/d`, `--binocular`).
* It is trained on this two-eye exam, with the same fading training wheels as attempt 3 (harder
  practice patients and the warmer/colder hint, both gone by 30% of training).
* Settings: entropy bonus 0.015, 100M steps, up to 120 moves per exam.
* The projector starts on the 20/20 line.

A first run started on the 20/40 line (`runs/old_d_start_2040`, log only). It never learned to read
the chart. On 20/40, a 0.25 D sphere step is hard to judge for most patients, so its sphere questions
told it little, and by 1M exams it only shrank cylinders (24% of eyes within 0.25 D). It was stopped
there. The scripts always read the chart first, so where the projector starts does not affect them.

The restarted attempt 4 was stopped for good at 1,273,013 exams (23.6M steps). A second machine restart
ended it, and by then the optometrist had chosen to try attempt 5 instead. It had learned to read the
chart, ask sphere and cross-cylinder questions and test the left eye. But it had not found the
binocular steps, and it scored 29% of eyes within 0.25 D on its check set (textbook 46%). The video
shows it as that: the AI teaching itself the full exam, and falling short.

**Attempt 5** (`runs/e`): watch the textbook, then practise.
1. **Watching** (`imitate.py`, behaviour cloning). The two-eye textbook script examined 97,103
   simulated patients, about 10M moves. The network saw exactly what the AI sees (never the true
   prescription) and learned to predict the script's next move. It matched 99.2% of moves. On its own
   it scored 45.5% of eyes within 0.25 D on the check set, the same as the textbook.
2. **Practising** (`train.py --init`). Ordinary reinforcement learning on the fair end score, starting
   from that network:
   * no hints and no harder patients;
   * the first 30 updates train only the value estimate;
   * then learning rate 1e-4 and entropy bonus 0.002, for 70M steps.

   Attempt 5 is "learned from the textbook, then improved on it", not "taught itself".

   A third machine restart stopped the practice run at update 220 (14.4M steps). It resumed from its
   saved full state (weights and optimizer). To finish while the session was active, the run was
   shortened from 70M to 40M steps. The learning-rate and entropy schedules follow the shorter run from
   update 221.

## 3. The scripted examiners

They are scripts, not people. A skilled clinician adapts in ways they do not.

* **Textbook script** (`clinicians.py`):
  1. Fog +1.00 D over the autorefractor reading.
  2. Unfog in 0.25 D steps, reading the chart each time. Keep the most plus lens with the best line,
     stopping after two steps with no improvement (MPMVA).
  3. If there is no cylinder, check for it at four axes.
  4. Cross-cylinder axis by bracketing (15° steps, then 5°). Then cross-cylinder power, keeping the
     spherical equivalent (0.25 D sphere per 0.50 D cylinder). Then the axis again.
  5. Final MPMVA re-check from +0.25 D.

  This was the best of three scripted variants tried on the same patients (an earlier one pushed
  plus while the chart line held, and over-plussed older patients).
* **Patient-pleaser script.** Starts at the autorefractor reading and gives whatever the patient
  says is clearer. Then it runs the same cross-cylinder steps. No fog, and no push-plus check.
* **Autorefractor alone.** Prescribes the starting reading.
* **Textbook script on the letter chart** (`textbook_letters`, used against attempt 3). It runs the
  same sequence with two changes:
  * MPMVA keeps the most plus lens within one letter (0.02) of the best reading. Without that
    tolerance, a single lucky letter pulled the script toward minus.
  * At the end: isolate the smallest line, add +0.25, and keep it unless a letter is lost.

  This was the best of six letter-chart variants (57.7% within 0.25 D, versus 54.2% for the
  whole-line script on the same patients).
* **Textbook script with no autorefractor** (`textbook_scratch`), starting at plano:
  1. Read the chart.
  2. If they can't read the top line, compare −2.00 D with +2.00 D to choose a direction.
  3. Add −1.00 D steps while the line improves. Otherwise (or after only one step), add +1.00 D steps
     until the chart drops by more than a line.
  4. Then the textbook sequence: fog +1.00, MPMVA, cylinder search at four axes, cross-cylinder axis,
     then power twice (large cylinders need more clicks), then axis again, then the MPMVA re-check.

  Earlier versions were tried on the same patients and discarded:
  * the first gave up on very myopic patients who couldn't read the top line;
  * the second sent high hyperopes who couldn't read the top line toward −8 D;
  * the third fogged +2.00 D and stopped unfogging too early. This one was the best on the test patients (49% within 0.25 D, versus 54% for
  the same script given an autorefractor reading).

**A bug fixed along the way.** All three scripts' cross-cylinder power step first kept the spherical
equivalent by adding 0.25 D sphere on every second cylinder click, even when the two clicks cancelled
("more", then "less"). That left 0.25 D of extra minus on some exams. It now compensates for the *net*
cylinder change. Every script result in RESULTS.md and the video is from the fixed version:
* the one-eye textbook went from 54.2% to 55.2% within 0.25 D;
* the patient-pleaser went from 33.6% to 36.3%.

## 4. Training

* **Algorithm.** PPO (Schulman et al. 2017, arXiv:1707.06347).
* **Network.** Two 256-unit ReLU layers for the policy and two for the value.
* **Batches.** 1,024 simulated exam rooms in parallel, 64 moves per batch.
* **Discount and settings.** γ = 0.995, λ = 0.95, clip 0.2, learning rate 3e-4 decaying.
* **Hardware.** 4 CPU cores, no GPU.
* **What the AI sees.** Its own lens settings, the patient's age, the starting reading, its last 12
  moves with answers, and a summary of the latest answer to each question type. It never sees the
  true prescription.
* **Scoring.**
  * At DONE it gets 1 − 2 × error (floor −1), plus 0.25 if exact.
  * Each lens change costs 0.002 and each question 0.005.
  * Exams time out at 60 moves (−0.2).
  * So exact scores 1.25, 0.25 D off scores 0.5, and 0.50 D off scores 0. The video's "full / 40% /
    nothing" is this scale.
* **Attempt 1** (`runs/a`). Half-weight cylinder score (`--cyl_weight 1`). Stopped at about 14M steps
  once its check-set score had been flat for about 6M steps. The video uses its 1,002,464-exam
  checkpoint.
* **Attempt 2** (`runs/b`). Full-cylinder score (`--cyl_weight 2`, the default). Every other setting
  is identical, including the random seed. Stopped at its 2,002,626-exam checkpoint (24M steps), with
  its check-set score flat since about 1M exams. At both 1M and 2M exams it still never turned the
  axis or added cylinder.
* **Attempt 3** (`runs/c`). Full-cylinder score, plus three changes:
  * **Harder practice patients at first** (`--hard_frac 0.3`). Cylinder scale 0.5 → 1.2 D (Gamma
    shape 1.2) and autorefractor J noise ×3, both easing linearly back to the normal clinic by 30% of
    training.
  * **Potential-based shaping** (`--shape_frac 0.3`). After every move it gets
    F = γΦ(s′) − Φ(s), with Φ = −2 × (current fair error) (Ng, Harada & Russell 1999, ICML). The
    weight falls linearly from 1 to 0 over the same 30%. Shaping of this form provably leaves the best
    strategy unchanged. It changes only when credit arrives.
  * **No autorefractor for half its practice exams** (`--start mix`; those start at plano). An extra
    input tells it whether a reading exists (`--flag`). Exams may run to 80 moves (`--t_max 80`).
  * **The letter chart and its three buttons** (`--letters`; see above).

  **Interruption.** At 42.6M steps the machine restarted and the training process was lost.
  * Training resumed from the last saved checkpoint (1,000,488 exams, 33.4M steps), with fresh
    optimizer state and every schedule at the same point.
  * The log of the lost stretch is kept in `runs/c/logs/train_interrupted_run.csv`. The learning
    curve in the video uses only the resumed run after that point.
  * The trainer now also saves its full state every 20 updates (`ckpt/latest.pt`, `--resume`).

  A first version of attempt 3 without the letter chart (`runs/old_c_line_chart`, log only) was
  stopped after about 6M steps when the letter chart was added.

  The last 70% of its training is on normal patients with the fair end score only.
* **An abandoned run** (`runs/old_b_shaping_halfcyl`, log only). It kept the half-weight score and
  added potential-based "warmer/colder" shaping (Ng, Harada & Russell 1999) for the first 30% of
  training. At 1M exams it still never turned the axis. It was stopped when the cause turned out to
  be the score itself. The shaping option remains in `train.py` (`--shape_frac`) but is not used by
  either attempt in the video.
* **Testing.** Results use the greedy policy (always its top choice) on 5,000 test patients drawn
  with a different random seed from anything used in training. Every examiner saw the same patients.
  Answer noise differs between examiners because each makes different requests.

## 5. The exams shown in the video

* **The showcase patient** was fixed before any exam was filmed (`record.py`):
  * 24 years old, true Rx −2.25 −1.00 × 170.
  * Autorefractor reading −2.75 −0.75 × 004 (the extra minus typical of young eyes, and 14° off axis).
  * Mildly prefers minus.
* **The untrained exam** samples actions from the random starting network (its greedy choice just
  repeats one button). The recording seed is the first one, of 0–99, whose exam ended by itself
  within 10–20 moves, so the clip has a watchable length.
* **Every other exam** is the greedy policy of the named checkpoint, exactly as recorded.

## 6. Results

See `RESULTS.md` (written by `report.py` from `data/results_a.json` and `data/results_b.json`).
