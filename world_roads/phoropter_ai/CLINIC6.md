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
| Remarks (instead of the smile) | After a lens change, if the change in what they see is at least a line (through their usual noise), the patient sometimes says so, each patient with their own chattiness (5–40% of noticeable changes): "Oh, that's clear!" (better, and within a line of the best they can see), "Better… but it's not that clear." (better, still more than a line off), "Hmm, that's not as clear." (worse). The AI sees the remark (which one, the lenses then, how long ago). It is never reward. |
| Reliability dial | Each patient's answer noise and lapse rate are multiplied by a factor: in practice drawn per patient, log-uniform between 0.5 and 2.5; in the tests fixed at 0.5 (reliable), 1 (typical: the version-5 patient) or 2 (unreliable). Chart reading and the final check are unchanged. |
| Everything else | As in version 5: graded answers ("much / a little / the same"), the power-search check, the final check ("my left eye is still blurry", 10 seconds, the exam goes on), the occluder cycling right → left → both, 15 minutes or 400 buttons and the patient walks out. |

How much rougher the spin is (`clinicians5` Borish script on 1,500 patients, eyes with at least 0.75 D of
cylinder; the axis step done by cross-cylinder, by one spin, or both):

| Patients | Axis step | Median axis error | Off by more than 10° | Net points |
|---|---|---:|---:|---:|
| env5 (v3) | cross-cylinder | 2° | 3% | −5.5 |
| env5 (v3) | one spin | 4° | 11% | −20.1 |
| env6, typical | cross-cylinder | 2° | 3% | −2.1 |
| env6, typical | one spin | 9° | 41% | −44.6 |
| env6, unreliable | cross-cylinder | 3° | 9% | −23.8 |
| env6, unreliable | one spin | 11° | 54% | −72.2 |

With the version-6 patient, refining the axis with the cross-cylinder is worth about 40 points a patient over
trusting the spin.

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

On 2,000 typical patients (env6, dial 1): net points −4.3 → +3.4, happy 55% → 60%, eyes under 40 over-minused
by 0.5 D or more 24% → 13%; walkouts 5.4% → 8.1% (the exam is longer: 9.7 minutes).

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

```bash
python train.py --clinic --env6 --gamma 1.0 --steps 200e6 --ent 0.01 --lr 3e-4 --eval_every 20 --out runs/i
```

One run from random weights. No mentor, no smile, no curriculum. The held-out checks during training use
typical patients (dial 1).
