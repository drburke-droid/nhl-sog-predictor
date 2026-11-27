# Version 6: a more realistic patient, the Borish method (the fourth video)

One AI from nothing, every exam starting at plano, as in [CLINIC5.md](CLINIC5.md), with four changes asked for
after the third video:

1. **Spinning the axis is a rough answer.** In version 5 the patient stopped a spinning axis almost exactly
   where it was sharpest, so the AI never needed the cross-cylinder for the axis.
2. **No smile.** The training reward is the clinic's points and nothing else. Patients don't give running
   feedback; now and then they remark on a change they notice.
3. **Consistency on the same patient**, with a dial that makes patients more or less reliable.
4. **The textbook is named: the Borish method**, the standard subjective sequence taught in optometry school,
   with the maximum-plus endpoint check it calls for.

**This is a simulation.** Every patient is a computer model and the clinic scorecard is made up. The AI has
never examined a real person.

Code: `env6.py`, `clinicians5.py` (`borish`), `train.py --clinic --env6`, `evaluate6.py`, `traces6.py`, `film6.py`.

## 1. The patient (`env6.py`)

| What | How |
|---|---|
| Spin the axis | The patient watches the dial turn and compares what they see with what they saw a moment ago, not two lenses flipped side by side. Their judgement noise is 3× their noise on a "1 or 2?" question (`SPIN_K`), and "stop!" comes late: the dial turns on by N(6°, 4°) (never backwards) before it stops. A lapse stops it anywhere. |
| Remarks (instead of the smile) | After a lens change, if the change in what they see is at least a line (through their usual noise), the patient sometimes says so, each patient with their own chattiness (5–40% of noticeable changes): "Oh, that's clear!" (better, and within a line of the best they can see), "Better… but it's not that clear." (better, still more than a line off), "Hmm, that's not as clear." (worse). The AI sees the remark (which one, the lenses then, how long ago). In training each remark also counts: 5 points per line the patient's vision (both eyes, each through its own lens) improved since their previous remark, or lost. Summed over an exam that is the change from the start to the last remark, so going back and forth earns nothing. On average a remark comes on about 1 button in 50. |
| Only the clock | The patient walks out at 15 minutes. env5 also ended the exam at 400 buttons (a safety cap); at 1 second a lens click, that was a walkout after 7 minutes, cheaper than 15. |
| Reliability dial | Each patient's answer noise and lapse rate are multiplied by a factor: in practice drawn per patient, log-uniform between 0.5 and 2.5; in the tests fixed at 0.5 (reliable), 1 (typical: the version-5 patient) or 2 (unreliable). Chart reading and the final check are unchanged. |
| Everything else | As in version 5: graded answers ("much / a little / the same"), the power-search check, the final check ("my left eye is still blurry", 10 seconds, the exam goes on), the occluder cycling right → left → both, and at 15 minutes the patient walks out. |

How much rougher the spin is (`clinicians5` Borish script on 1,500 patients, eyes with at least 0.75 D of
cylinder; the axis step done by cross-cylinder, by one spin, or both):

| Patients | Axis step | Median axis error | Off by more than 10° | Net points |
|---|---|---:|---:|---:|
| env5 (v3) | cross-cylinder | 2° | 3% | −5.5 |
| env5 (v3) | one spin | 4° | 11% | −20.1 |
| env6, typical | cross-cylinder | 2° | 4% | −8.5 |
| env6, typical | one spin | 9° | 41% | −44.6 |
| env6, unreliable | cross-cylinder | 3° | 9% | −28.4 |
| env6, unreliable | one spin | 11° | 54% | −72.2 |

With the version-6 patient, refining the axis with the cross-cylinder is worth about 36 points a patient over
trusting the spin. (env6 rows: the cross-cylinder button turns the axis itself, by the same steps the script
used to press; section 3.)

## 2. The Borish script (`clinicians5.borish`)

The script of CLINIC5.md section 2 (right eye, left eye, both eyes; trial lenses, fog and clear, the
power-search check, cross-cylinder axis and power, binocular fog, prism balance and clear), plus:

- **The maximum-plus check at each endpoint.** After each eye's best sphere and after clearing the binocular
  fog: "+0.25: is 1 or 2 clearer?" If the extra plus is clearer, keep it and ask again; if it looks the same,
  keep it and stop; if the patient prefers the lens without it, stop.
- **Fog until it blurs.** If 0.75 D of fog and two more quarters haven't reduced the acuity (a young,
  over-minused eye focuses through it), add plus a dioptre at a time, up to 8 more steps. (The old script
  stopped after 1.50 D, so a young patient over-minused by 3 D from the trial lenses was never fogged and
  left with far too much minus.)

On 2,000 typical patients (env6, dial 1): net points −4.6 → +0.5, happy 55% → 60%, eyes under 40 over-minused
by 0.5 D or more 26% → 14% (by 0.25 D or more 49% → 36%); walkouts 5.0% → 9.1% (the exam is longer: 9.3 → 9.8
minutes).

**Why the third video's textbook over-minused the showcase patient's left eye** (OS −3.00 −0.50 × 10 against
−2.50 −0.75 × 10). Two things, both in the simulation on purpose:

1. *Instrument (proximal) myopia.* Behind the phoropter the patient focuses a little extra (this 24-year-old:
   0.35 D; relaxed by fog, partly back with both eyes open). Through their exact prescription they see
   slightly worse than with a quarter more minus, so every subjective question, the push-plus check included,
   prefers the extra quarter. No examiner using the patient's answers can see this; it is why the final
   glasses of both examiners tend to land 0.25 D on the minus side for young patients.
2. *The endpoint came from the chart alone.* A young eye reads the chart as well through extra minus as
   without it, and the chart's letter-by-letter noise picked an extra quarter. The old script had no
   "+0.25: same?" check to take it back. The maximum-plus check now does.

## 3. Training

### From nothing, with only the remarks: stuck (`runs/j`)

```bash
python train.py --clinic --env6 --remark 0.5 --gamma 1.0 --steps 200e6 --ent 0.01 --lr 3e-4 --eval_every 20 --out runs/j
```

Two settings were started side by side from random weights: `runs/k` with the remarks as information only
(the only score is the clinic's points at the end of the exam), and `runs/j` with the remarks also counting
as feedback (`--remark 0.5`). After 160 updates (10.5 million buttons each):

| Run | Exams | Practice points | Held-out points | Happy | Minutes |
|---|---:|---:|---:|---:|---:|
| `runs/k`, remarks as information | 18,524 | −119 | −150 | 0% | 15.0 (every patient walks out) |
| `runs/j`, remarks count | 33,047 | −61 | −48 | 18% | 3.7 |

`runs/k` was stopped there. `runs/j` trained on to 905,000 patients and stalled at a shortcut: both eyes open,
minus a quarter at a time while the chart improves, a little cylinder now and then, prescribe. Held-out points
went from −48 (200,000 patients) to −40 (900,000); happy patients stayed at 19%, prescriptions within 0.25 D
at 12%, and it never used the cross-cylinder. The third video's AI (`runs/h`, with the smile) was at −30 by
300,000 patients and passed zero at 700,000. The smile, a signal on every button, was doing much of the
teaching; a remark on about 1 button in 50 is not enough to find a full refraction from random buttons in a
reasonable time. (Checkpoint `runs/j/ckpt/exams_904847.pt`; `evaluate6.py` reports it as `scratch`.)

### From the third video's AI (`runs/m`)

```bash
python init6.py --nudge    # runs/h/ckpt/final.pt -> runs/m/init.pt (seven new inputs, zero weights; the nudge)
python train.py --clinic --env6 --remark 0.5 --gamma 1.0 --steps 100e6 --ent 0.01 --lr 3e-4 --eval_every 20 \
       --init runs/m/init.pt --critic_warmup 20 --out runs/m
```

The AI that practised on 1.8 million version-5 patients walks into the version-6 clinic: no smile, a rough
spin, the occasional remark, patients of every reliability. `init6.py` gives its first layer seven new input
columns (the remark) set to zero, so at the start it behaves as in the third video (checked: same choices,
same value estimates). The first 20 updates retrain only its value estimate (the smile it used to expect is
gone).

On the new patients, before any practice here (the training log's held-out check, 1,000 typical patients):
glasses points +41 (net of chair time about +27), happy 50%; on the version-5 patients it had +66 (net +51)
and 67%. (The log's "pts" is what the glasses earn; the net subtracts 4 points a minute.)

**Two changes, after two attempts.**

1. *Without help* (`runs/m0`, 195,000 patients) it changed little: points +41 → +42, happy 50% → 51%, the axis
   no closer (64% of astigmatic eyes within 10°, before and after). It spun the axis more and turned it by
   hand more (16 turns an exam instead of 2), but never tried the cross-cylinder for the axis: the third
   video's AI had dropped that button altogether (its chance of pressing it, wherever it is allowed: under 1
   in 100,000), so practice could never find out what it does now.
2. *With a nudge* (`runs/m1`, 187,000 patients): `init6.py --nudge` gives that one button the same outputs as
   the cross-cylinder *power* button, which the AI uses in every exam, at a quarter of its chance (about 5%
   wherever it is allowed, against 19% for power). Its most likely button never changes, so its exam before
   practice is the same. Practice dropped the button again within 50,000 patients (4.5% → 0.1% → 0.01%):
   an answer to "1 or 2?" only helps if the next button turns the axis the right way, and with points only at
   the end of the exam (and a remark now and then) it never learned that pairing. Each press was wasted time.
3. *The button turns the axis* (`runs/m`, with the nudge): in env6 "Axis? which way" turns the axis toward the
   side the patient picks, by the textbook's bracketing steps (15° then 5° for 0.50 D of cylinder or less, 5°
   for 0.75 D, 5° then 2° for 1.00 D or more), the way the spin sets the axis at "stop!". The AI still decides
   whether to use it, when, for which eyes and how many times. The Borish script uses the same button and
   stops where it stopped before (reversal on the smallest step); its results are unchanged within noise
   (textbook script on 1,500 patients: −8.4 with its own turns, −8.5 with the button).

`runs/m` was stopped at 585,322 patients (held-out points level since about 250,000).

## 4. Results (`evaluate6.py` → `data/results_m.json`)

5,000 new patients, typical reliability (dial 1), every exam starting at plano. "Day one" is the third
video's AI before any practice in the new clinic; "AI" is the same AI after 585,322 patients there.

| | Plano | Day one | Borish script | AI |
|---|---:|---:|---:|---:|
| Net points per patient | −107 | +23 | +1 | **+51** |
| … glasses | −107 | +37 | +41 | +67 |
| … chair time (4 a minute) | 0 | −14 | −39 | −16 |
| Time in the chair | 0:00 | 3:31 | 9:48 | 4:01 |
| Happy | 5% | 48% | 60% | **69%** |
| Walked out (15 minutes) | 0% | 0.8% | 9.1% | 0.3% |
| Spherical equivalent within 0.25 D | 12% | 86% | 79% | 91% |
| Whole prescription within 0.25 D (power vector) | 3% | 31% | 46% | 57% |
| Astigmatic eyes (≥ 0.75 D): axis within 10° | | 64% | 94% | 94% |
| … median axis error | | 8° | 2° | 2° |
| Eyes under 40 over-minused by 0.25 D or more | 30% | 39% | 36% | 15% |

**The Borish script's negative points are chair time.** Its glasses earn +41 a patient, a little more than
day one's; its 9:48 in the chair costs 39. That and 9% of patients walking out at 15 minutes (16% of
unreliable patients, 5% of reliable ones) leave it at +1.

**What practice changed.** The AI's exam was already "both eyes first, then each eye, then both eyes"; in
the new clinic it added the cross-cylinder for the axis after the spin, in every exam (2.3 spins, 6.4 axis
cross-cylinder and 6.7 power cross-cylinder questions an exam). Per eye, most often: a little cylinder in,
spin ("stop!"), the cross-cylinder for the axis two or three times, then the power; with the textbook's own
bracketing steps the axis ends as close as the Borish script's (94% within 10°). It also ends with less extra
minus than before: eyes under 40 over-minused 39% → 15%.

**Consistency: the same 2,000 patients examined twice** (fresh answers each time), how often the two
prescriptions agree:

| Patients (reliability dial) | Borish: SE within 0.25 D | AI: SE within 0.25 D | Borish: whole Rx within 0.25 D | AI: whole Rx within 0.25 D | Borish net | AI net |
|---|---:|---:|---:|---:|---:|---:|
| Reliable (0.5) | 75% | 91% | 64% | 65% | +16 | +57 |
| Typical (1) | 68% | 88% | 55% | 63% | +1 | +51 |
| Unreliable (2) | 61% | 83% | 46% | 58% | −27 | +34 |

Published test-retest for real optometrists, spherical equivalent within 0.25 D: about 80% (Goss &
Grosvenor 1996; Shah et al. 2009). Consistency alone is not accuracy: prescribing plano to everyone would
agree 100% of the time. Read it next to the happy patients.

**The AI from nothing** (`runs/j`, 905,000 patients, remarks only): net −48, happy 19%, whole prescription
within 0.25 D 12%; in half its exams it never covered an eye.

## 5. The exams shown (`traces6.py`)

The showcase patient of the earlier videos (24 years old, OD −2.25 −1.00 × 170, OS −2.50 −0.75 × 10,
0.35 D of instrument myopia), typical reliability:

| | Buttons | Time | Prescribed | Points |
|---|---:|---:|---|---:|
| Day one | 62 | 2:25 | OD −2.50 −1.00 × 174, OS −2.75 −0.75 × 7 | +73 |
| AI | 75 | 2:57 | OD −2.00 −1.00 × 169, OS −2.25 −0.75 × 13 | +78 |
| Borish script | 205 | 12:12 | OD −2.25 −1.00 × 170, OS −2.50 −0.75 × 10 | +97 |

The Borish script's push-plus checks took this patient's extra quarter of minus back: the prescription is
exact (in the third video's exam it left the left eye's sphere 0.50 D too minus).
