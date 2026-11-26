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
