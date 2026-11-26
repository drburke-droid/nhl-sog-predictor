# The clinic video: "Day one at AI Optometry"

One AI that knows nothing about eyes is given a phoropter, a projector chart and a clinic to run. Every exam
starts at plano: no autorefractor and no retinoscopy. The AI is paid in clinic points. A happy patient who
returns and refers is good; a re-check, a bad review or a long exam is bad. It practises on simulated
patients until it is good at the job. The video then compares how it examines, step by step, with the
textbook sequence.

**This is a simulation.** Every patient is a computer model, and the clinic scorecard is made up. The AI
has never examined a real person. It is a demonstration of reinforcement learning, not a clinical tool.

Everything the earlier video used still applies: the optics (section 1 of [ACCURACY.md](ACCURACY.md)), the
patient model (section 2), and the two-eye patient (section 2b). This file covers only what is new. The
numbers come from `data/results_f.json` (`evaluate3.py`).

## 1. The exam (`env3.py`)

| What | How |
|---|---|
| Start | Both eyes at plano (0.00 DS, axis 180). The projector shows 20/200. |
| Order | Right eye (left occluded), then left eye (right occluded), then both eyes open. The AI presses **Next eye** to move on and cannot go back. |
| Buttons (27) | Sphere ±0.25 and ±1.00; cylinder ±0.25; axis ±2°, ±5° and ±15°; "read the smallest line"; "read this line"; smaller or bigger line; "1 or 2?" with ±0.25 sphere; cross-cylinder "1 or 2?" for axis and for power; Next eye; both eyes ±0.25; prism-split balance "top or bottom?"; +0.25 right eye only; +0.25 left eye only; **Prescribe**. The ±2° axis buttons were added at the optometrist's request. |
| Switched-off buttons | Cylinder, axis and cross-cylinder work only with one eye covered. The both-eye buttons, the balance and the one-eye +0.25 buttons work only with both eyes open. The network never picks a switched-off button: those choices are masked. |
| Prescribe | Works at any time. It hands over whatever is in the phoropter, both eyes. |
| The patient | The two-eye patient of `env2.py`: yoked accommodation; proximal accommodation behind the phoropter (relaxed by fog; 30% of it remains with both eyes open); line-size-dependent judgement; ocular dominance in the prism split; noise, a "same" band, a preference for minus, and lapses. |
| What the AI sees | Age; the stage; the line on screen; button count and chair time so far; both eyes' lenses; its best chart reading per eye; the latest answer to each kind of question per stage, with the lenses it was asked at; the last balance answer; its last 16 buttons with their answers; which buttons work now. It never sees the true prescription. |

## 2. The clinic scorecard (`clinic.py`)

The patient judges the glasses mostly by the worse eye:
patient error = 0.75 × (worse eye's error) + 0.25 × (better eye's error). Each eye's error is `optics.rx_error`:
the length of (spherical-equivalent error, leftover cylinder in full dioptres).

| Outcome | Points | Meaning |
|---|---:|---|
| Happy | +100 | sees clearly, returns next year, refers a friend |
| Annoyed | +30 | "not quite crisp", may not return |
| Rx check | −80 | an unpaid 15-minute re-check (60 points of chair time), doesn't return |
| Rx check + bad review | −150 | re-check, never returns, one-star review |
| Walked out | −150 | the exam ran past 15 minutes (or 200 buttons): no glasses |
| Chair time | −4 per minute | what a minute of a 25-minute appointment earns |

Chance of each outcome by patient error (linear in between):

| Patient error (D) | 0 | 0.13 | 0.25 | 0.375 | 0.50 | 0.75 | 1.00 | 1.50 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Happy | 95% | 92% | 75% | 45% | 20% | 5% | 0 | 0 | 0 |
| Annoyed | 5% | 8% | 23% | 45% | 50% | 30% | 10% | 0 | 0 |
| Rx check | 0 | 0 | 2% | 10% | 28% | 55% | 55% | 25% | 5% |
| Bad review | 0 | 0 | 0 | 0 | 2% | 10% | 35% | 75% | 95% |

These numbers are a plausible made-up scorecard, not measured data. Real clinics see about 2% of glasses
come back (PUBLISHED.md §3), and real refractions mostly land within 0.25 D, so on real-world errors this
table would predict roughly 5%. The AI is trained on the expected points of each prescription, which is
less noisy than one sampled outcome. The video samples one outcome per patient for its scoreboards.

### Seconds per button (`env3.SECS`)

| Button | Seconds |
|---|---:|
| Lens click (sphere, cylinder, axis, both eyes, one eye) | 1 |
| Line smaller or bigger | 1 |
| "Read this line" | 4 |
| "1 or 2?" (sphere or cross-cylinder) | 4 |
| Balance "top or bottom?" | 5 |
| Next eye | 2 |
| "Read the smallest line you can" | 8 |

The timings are calibrated so the textbook script, starting from an objective reading, takes 6.3 minutes. The
published figures are 4:37 to 5:38, and 9.5 minutes for a full Borish routine (PUBLISHED.md §2).

## 3. The textbook script (`clinicians3.py`)

The standard sequence as taught (Wilkinson 2016, University of Iowa EyeRounds; the Borish sequence), with
the same buttons and the same patients:

1. **Starting sphere.** No objective start is available, so the script reads the chart. If the patient
   can't read 20/200, it compares −2.00 with +2.00. It then steps 1.00 D at a time while the chart improves
   (minus), or while it doesn't get clearly worse (plus).
2. **Monocular MPMVA.** Fog +0.75, adding more plus until the patient has lost at least 1.5 lines. Then
   reduce 0.25 D at a time, reading the chart. Stop after two readings with no improvement, but not while
   still two lines worse than before the fog. End on the most plus lens within two letters of the best
   reading.
3. **Cross-cylinder,** one line bigger than the best:
   - Search for cylinder at four orientations if there is none.
   - Axis first if the cylinder is 1.00 D or more, otherwise power first.
   - Axis steps narrow at each reversal: 15° → 5° for small cylinders, 5° → 2° for 1.00 D or more.
   - Power changes 0.25 D at a time, with 0.25 D of sphere per net 0.50 D of cylinder.
   - Power again if the axis moved 10° or more.
4. **Second MPMVA** if the cylinder changed by 0.50 D or more, or the axis by 10° or more.
5. **The left eye.**
6. **Binocular balance.** Read the chart with both eyes, then +0.75 on both eyes and the 20/40 line. Use the
   prism split and add +0.25 to the clearer eye until they look equal; stop if the answer flips.
7. **Binocular MPMVA.** Reduce plus on both eyes 0.25 at a time; no more minus without a better reading.
   Prescribe.

### Calibration against published repeatability

- Real refractions: two refractions of the same eye agree within 0.25 D (spherical equivalent) about 80%
  of the time (Goss & Grosvenor 1996; Shah et al. 2009: 81%).
- Run twice on the same simulated patients with fresh answer noise, starting from an objective reading, the
  first version of this script agreed with itself 64% of the time. Its 95% range was ±1.2 D.
- The tails came from the script stopping a fogged MPMVA too early: big blur makes chart readings coarse.
  Steps 2 and 7 above (fog check, plateau rule, the same patience with both eyes) fixed the tails.
- No single patient-noise setting moved the agreement much. That was tested for lapses, response noise,
  the "same" band, letter-reading jitter, the minus preference and proximal accommodation.
- Final version: 66% agreement from an objective start; [TBD]% from plano on the 5,000-patient test.
- Simulated patients are therefore harder to refract than real ones, or a rigid script does worse than a
  clinician's judgement, or both. Compare the AI with the script, not with real optometrists.

## 4. Training (`train.py --clinic`)

```bash
python train.py --clinic --gamma 1.0 --steps 200e6 --shape_frac 0.15 --hard_frac 0.3 --ent 0.01 --lr 3e-4 \
    --eval_every 25 --out runs/f
```

- PPO; actor and critic are separate MLPs, 2 × 256. It runs 1,024 exam rooms at once with 64-button
  rollouts.
- **No discount (γ = 1).** The AI is paid total points per patient. Two short test runs used γ = 0.995,
  and the AI learned never to prescribe: discounting makes a bad outcome 200 buttons away look smaller than
  one now. Every exam ran to the time limit.
- **The mentor (training wheels).** For the first 15% of practice, every button also earned "warmer/colder":
  2 × (how much closer the two eyes' lenses got to the truth, on average), fading linearly to zero. A first
  version used the textbook form of this hint (potential-based, zero at the end of the exam). That form pays
  a large bonus for ending an exam far from the truth, and the AI again learned to stall. The video says the
  mentor whispered for the first 15% of practice, then left.
- **Harder practice patients early.** Bigger cylinders, fading to normal by 30% of training, so cylinder
  matters while the AI is exploring.
- Checkpoints at milestone patient counts (`runs/f/ckpt/exams_*.pt`); `latest.pt` every 20 updates.

## 5. Evaluation (`evaluate3.py`)

- 5,000 new simulated patients (seed 2026), all starting at plano. The AI uses its best guess at each step
  (greedy); "Plano" prescribes immediately.
- Practice checkpoints are scored on 2,000 of them twice. Sampled is how it behaved while practising; greedy
  is its best guess. The practice curve in the video is from the training log: points per patient (glasses
  minus chair time) on the practice patients themselves, smoothed over 9 updates.
- Repeatability: the same 2,000 patients examined twice, with different random answers.
- Habits: button use per exam by group; minutes per stage; and switches between sphere work (sphere lenses,
  sphere "1 or 2?", chart readings) and cylinder work (cylinder, axis, cross-cylinder) within each eye.

## 6. The exams shown

- All on the showcase patient (`record3.SHOWCASE3`, the same 24-year-old as the earlier video):
  OD −2.25 −1.00 × 170, OS −2.50 −0.75 × 010; prefers minus; proximal accommodation 0.35 D; right-eye
  dominant. Chosen before any exam was filmed.
- Day one is the untrained network (`exams_0.pt`), sampling buttons as it does in practice; the
  recording's random seed was fixed before looking at it. The graduate exam is the final network, greedy.
- The patient's odds on screen are exact for the model: the normal-distribution chance of each answer,
  and the binomial chance of each letter count. Reading the smallest line is estimated from 4,000 simulated
  readings. A ✓ marks the answer that points toward the true prescription: the option whose lenses are
  closer to it.
- The 55-year-old in the "what the patient can say" demo is hypothetical (−2.75 DS, 1 D of accommodation,
  no proximal accommodation, a slight minus preference). It is asked on the smallest line they can just read.

## 7. Results

[TBD after the evaluation]
