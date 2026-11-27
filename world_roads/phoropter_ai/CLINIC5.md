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
