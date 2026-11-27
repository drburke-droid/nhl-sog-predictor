# Version 5: no mentor, a patient who talks back (the third video)

One AI, starting from nothing, every exam starting at plano: no autorefractor, no retinoscopy. Unlike the
earlier versions there is **no mentor** ("warmer/colder" hints) and **no curriculum** (no special practice
patients, no changes to the clinic between terms). Its only help is the patient: how sure they are, their
smile when the letters get clearer, and their complaint when the new glasses are still blurry. The question
for the video: can it work out a refraction, and how do its steps compare with the textbook's?

**This is a simulation.** Every patient is a computer model and the clinic scorecard is made up. The AI has
never examined a real person. It is a demonstration of reinforcement learning, not a clinical tool.

Everything in [CLINIC.md](CLINIC.md) still applies (the optics, the patient model, the scorecard, the seconds
per button) except what is listed here. Code: `env5.py`, `clinicians5.py`, `train.py --clinic --env5`,
`evaluate5.py`, `traces5.py`, `film5.py`.

## 1. What changed in the exam (`env5.py`)

| What | How |
|---|---|
| Graded answers | The patient says how sure they are: "One, much clearer" (the difference is beyond their "same" band by more than 0.06 logMAR), "One… a little clearer", or "They look the same". |
| Astigmatism check (the power-search method) | Two new questions: the cross-cylinder at 180 vs 90, and at 45 vs 135. With no cylinder in yet, a preference sets the axis dial to the preferred orientation. |
| Spin the axis | With some cylinder in, the patient watches the axis dial turn and says "stop" where the letters look sharpest (with their usual noise: 36 orientations, each judged with the same blur model as every other question). 6 seconds. |
| The cross-cylinder refinements | Unchanged but named plainly: **Cylinder AXIS: which way to turn?** and **Cylinder POWER: more or less?** |
| "It's still blurry" | On **Prescribe**, the patient reads with each eye through the new glasses. If either eye sees more than two lines (0.2 logMAR, with patient-to-patient noise SD 0.05) worse than it could through its true prescription, they say so ("Wait, my left eye is still blurry") and stay in the chair. Prescribe does not end the exam; each complaint costs 10 seconds. In effect the button is **"Final check: read with each eye. Both clear: these are the glasses"**, and the video labels it that way. The patient walks out at 15 minutes or 400 buttons (−150 points). |
| The occluder cycles | Right eye, left eye, both eyes, then back to the right eye, so either eye can be redone after a complaint. |
| What the AI sees | As before, plus the strength of each answer, which eye the last complaint was about, and how long ago. 30 buttons, 1,074 numbers. |

### The patient's smile (the only training help)

While practising, the AI earns a little when the letters get clearer for the patient and loses a little when
they get blurrier: 5 points per line of acuity (the average of the two eyes, each through its own lens and
focusing as it naturally would; `--smile 0.5` is per unit of logMAR in reward units of 100 points). A −3.00
myope who goes from 20/200 to 20/20 is worth about 50 points of smiles, against +100 for a happy patient. It is based only on what the patient sees, never on the true prescription.
Because it is a difference of the same quantity, over a whole exam it adds up to 5 points per line the
patient ends up seeing better than at the start: it nudges toward clear vision, which is what the clinic pays
for anyway, but cannot be farmed by going back and forth. The scores reported below are clinic points only,
without the smile.

## 2. The textbook script (`clinicians5.py`)

The textbook sequence of CLINIC.md section 3 (Borish, after Wilkinson 2016), with:

- **The power-search astigmatism check** (180/90, then 45/135) in place of the earlier four-orientation search.
- **A fixed trial-lens search.** Checking the complaint rule showed that the earlier script mishandled
  patients who cannot read the top line (high myopes, older hyperopes): if −2.00 and +2.00 were both still
  unreadable it drifted back toward plano. It now tries ±2, ±4 and ±6 D, starts from the lens that helps, and
  keeps adding minus while only the 20/200 letter is readable. **This bug also affected the textbook numbers
  in the first video (plano start).** The second video started at the autorefractor and was not affected.
- **The same complaint rule as the AI.** If the patient says an eye is still blurry, the script goes back to
  that eye, redoes it (trial lenses, best sphere, cylinder), then repeats the both-eyes steps and prescribes
  again; twice at most.

With these changes, on 2,000 test patients, 3.8% of patients leave the textbook exam with an eye more than two
lines worse than it could see (10% before the fix and the complaint rule). The remaining failures come from
the chart's guessing floor: a patient who can only see a blur sometimes names the single 20/200 letter by
chance, which fools the sphere search. Both examiners face the same chart.

## 3. Training

```bash
python train.py --clinic --env5 --smile 0.5 --gamma 1.0 --steps 200e6 --ent 0.01 --lr 3e-4 --eval_every 20 --out runs/h
# paused at update 448 (489,251 practice patients) while two other complaint rules were tried, then resumed:
python train.py ... --out runs/h --resume runs/h/ckpt/latest.pt      # from update 441 (476,675 patients)
```

One run from random weights. No mentor, no curriculum, no changes to the clinic during training. Its stdout is in
`runs/h/train_stdout_part1.log` and `runs_h.log`.

**What it learned first.** By 300,000 practice patients it skipped the one-eye steps and used the final check
as a question: −0.25 on both eyes, final check, "still blurry", more minus, final check again, until both eyes
passed. It ignored astigmatism. That looked like a loophole, so the run was paused and two stricter rules were
tried from scratch:

- **Charging for complaints (`runs/h1`).** 20 points for every complaint taught it never to press the button
  (every patient walked out at the 400-button cap; not kept). With the first complaint free and 20 points for
  each further one, it settled by 50,000 patients into a blind habit: −3.00 in both eyes for everyone, no
  questions. Young eyes focus through the extra minus, so most patients didn't complain, and the clinic outcome
  alone was too weak a signal.
- **Three strikes (`runs/h2`).** Complaints cost only time, but the third ends the visit (−150). It learned to
  press the button three times quickly: a hard patient walked out within a minute, which saves chair time, and
  a walkout costs no more than bad glasses. Stuck near −90 points at 190,000 patients.

Both stricter rules made learning worse. The original rule is also the realistic one: a final acuity check
with each eye is normal practice, and a patient who says "my left eye is still blurry" is simply answering it.
Checking often is unusual, not cheating; it costs 10 seconds each time. So the first run was resumed under
its own rule. Both examiners play by it. Under it the textbook script draws a complaint in 13% of exams and a
second one in 1.8%; 5.1% of its exams hit the 15-minute limit while redoing an eye; its net is −2.8 points
per patient (2,000 test patients).

A pilot with only the smile (no complaint rule) showed it escapes the trap that stopped the earlier no-mentor
attempt (prescribing plano at once, −107 points): −56 points by 122,000 practice patients. With the complaint
rule added, day one is worse (every random exam ends in a walkout), but the AI learned faster: −51 points on
held-out patients after 57,000 and −20 after 445,000.

**How it ended.** Training was stopped at update 1,800 (1,812,949 practice patients, 2 h 20 min after the
resume on 4 CPU cores) because the held-out score had been flat at about +65 clinic points (before chair time) for
300 updates. `runs/h/ckpt/final.pt` is that network (`latest.pt` at update 1,800, saved in half precision).

## 4. Results (`evaluate5.py` → `data/results_h.json`)

5,000 new simulated patients (seed 2026), every exam starting at plano. The AI picks its most likely button
at each step. "Plano" prescribes plano for everyone with the final check switched off (otherwise every blurry
patient refuses it and walks out): the do-nothing reference.

| | Plano | Textbook script | AI |
|---|---:|---:|---:|
| Time in the chair | 0:00 | 9:19 | **3:14** |
| Buttons per exam | 1 | 156 | 79 |
| Glasses within 0.25 D (power vector, every eye) | 3% | 43% | **46%** |
| Glasses within 0.50 D | 10% | 78% | **91%** |
| Spherical equivalent within 0.25 D | 12% | 75% | **91%** |
| Cylinder within 0.25 D | 42% | **82%** | 81% |
| Happy patients | 5% | 56% | **65%** |
| Clinic points (glasses) | −107 | +34 | **+64** |
| **Net points per patient** (glasses − 4/min) | −107 | −3 | **+51** |
| Walked out (15 min or 400 buttons) | — | 5.3% | 0.2% |
| Exams with a "still blurry" complaint | — | 10.8% | 13.6% |
| Two exams of the same eye agree within 0.25 D (2,000 patients, SE) | — | 64% | **87%** |

Astigmatism by orientation (within 0.25 D, power vector): with-the-rule 31% / 39%, against-the-rule 32% / 34%,
oblique 33% / 35% (textbook / AI): no orientation is left behind.

### How its exam differs from the textbook's (habits on the 5,000 patients)

| | Textbook script | AI |
|---|---|---|
| Order of occluder visits | right, left, both (89%) | **both, right, left, both (97.5%)** |
| Time per stage (right / left / both) | 3.9 / 3.9 / 1.6 min | 0.6 / 0.6 / 2.0 min |
| Chart readings per exam | 50 | 12 |
| Sphere "1 or 2?" | never | 3.5 per exam |
| Astigmatism check (180/90, 45/135) | every exam | **never** |
| Cylinder AXIS (cross-cylinder) | 96% of exams, 5.1 per exam | **never** |
| Spin the axis ("say stop") | never | **every exam, 4.0 per exam** |
| Cylinder POWER (cross-cylinder) | 96%, 6.7 per exam | every exam, 6.3 per exam |
| Prism balance | every exam | never (adjusts one eye with both open in 13%) |

In words: it opens both eyes, adds minus to both until the chart stops improving, and deliberately goes
past it. Then it covers each eye in turn, puts in a quarter of cylinder without asking anything, spins the
axis until the patient says stop, and alternates spin and cross-cylinder POWER until both settle, taking
minus off the sphere as the cylinder goes up. Finally it opens both eyes and takes the extra minus off
together, a quarter at a time, with the chart and "1 or 2?", and presses the final check.

**The overshoot depends on age** (`overshoot5.py` → `data/overshoot_h.json`, 1,500 new patients; extra minus
beyond the true spherical equivalent, both eyes averaged, when it first covers an eye):

| Age | under 20 | 20s | 30s | 40s | 50s | 60+ |
|---|---:|---:|---:|---:|---:|---:|
| Extra minus (mean) | 3.3 D | 2.6 D | 1.8 D | 1.4 D | 0.9 D | −0.1 D |

A young eye focuses through extra minus, so the letters stay sharp while it does the cylinder; an older eye
cannot, and the AI doesn't overshoot it. This may simply follow from "keep adding minus while the chart
improves or holds": young eyes hold for longer. In the final glasses the AI over-minuses young patients less
often than the textbook script: 31% against 46% of patients under 40 leave with the right eye over-minused by
0.25 D or more (`over_minus_young`).

### How it got there (`R["stages"]`, 2,000 patients per checkpoint, sampling buttons as it practised)

| Practice patients | Net points | Happy | Most common visit order | Spin used | Complaints per exam | Walked out |
|---:|---:|---:|---|---:|---:|---:|
| 0 | −174 | 2% | random | (random) | 10.8 | 68% |
| 10,114 | −123 | 8% | both only (16%) | 2% | 7.0 | 34% |
| 20,200 | −89 | 13% | both only (51%) | 1% | 10.8 | 23% |
| 100,240 | −65 | 18% | both only (53%) | 19% | 6.5 | 11% |
| 501,297 | −25 | 26% | both only (45%) | 30% | 3.7 | 2% |
| 751,332 | −10 | 33% | both, R, L, both (33%) | 44% | 2.9 | 1% |
| 1,000,094 | +41 | 58% | both, R, L, both (94%) | 100% | 0.5 | 1% |
| 1,812,949 | +48 | 63% | both, R, L, both (97%) | 100% | 0.3 | 1% |

From about 10,000 to 750,000 patients its usual exam never covered an eye: sphere only, both eyes open, using
the final check as a guide ("still blurry" → more minus → check again). Between 750,000 and 1,000,000 it began
covering each eye to put in cylinder, and the score jumped by 50 points. The training curve (sampled
buttons, smoothed) crosses the textbook script's −3 at about 800,000 patients.

## 5. The exams shown (`traces5.py`)

One showcase patient, 24 years old: OD −2.25 −1.00 × 170, OS −2.50 −0.75 × 10.

- Day one (`exams_0.pt`, sampling): 276 buttons, 21 "still blurry" complaints, walked out at 15 minutes.
- The textbook script: 193 buttons, 11:33, OD −2.50 −1.00 × 170, OS −3.00 −0.50 × 10; +42 points before chair time.
- The AI (`final.pt`): 95 buttons, 3:49, OD −2.00 −1.00 × 170, OS −2.50 −0.75 × 10 (left eye exact, right a
  quarter short); +88 points before chair time, +72 net.

The side-by-side groups each exam into named steps per occluder visit (`sequence5.py`). Two readability
merges apply to both columns: quarter-dioptre sphere clicks (no question) inside a cylinder-power or spin step
belong to it (keeping the spherical equivalent), and axis turns made while no cylinder is in (they do nothing)
belong to the "put in some cylinder" step that follows. Alternating spin and cylinder-power steps (three or more)
are one row, "Spin the axis, cylinder POWER, back and forth".

## 6. Limits

- The patients are a model. Real patients answer less consistently, tire, and have reasons for their answers
  the model doesn't have. The overshoot strategy relies on accommodation behaving as modelled; a real latent
  hyperope or a patient with accommodative spasm may not.
- The scorecard is made up, and the AI optimises it: a different scorecard would give a different exam.
- The textbook script is a script, not an optometrist: it reads the chart after almost every click, so it is
  slower than a clinician (about 5 minutes in published reports). Compare the AI with the script.
