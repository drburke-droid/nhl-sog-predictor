# I gave an AI a phoropter

A reinforcement-learning agent teaches itself subjective refraction on simulated patients, and the videos
of how it learned.

## The third video: no mentor, plano start (`runs/h`, [CLINIC5.md](CLINIC5.md))

One AI, starting from nothing, every exam at plano. No mentor hints and no curriculum: its only help is the
patient (how sure they are, a smile when the letters get clearer, and "it's still blurry" at the final
check). It can search for astigmatism the textbook way (cross-cylinder at 180/90 and 45/135) or put in some
cylinder and spin the axis until the patient says stop. After 1.8 million practice patients it works out its
own sequence (both eyes first, deliberately over-minused; then each eye: cylinder in, spin, cross-cylinder
power; then both eyes to take the extra minus off) and is compared with the textbook sequence side by side.
On 5,000 new patients: 3:14 in the chair against 9:19, 65% happy against 56%, +51 points against −3.

| File | What it does |
|---|---|
| `env5.py` | The exam: graded answers, the power-search check, spin, the final check ("still blurry"), cycling occluder |
| `clinicians5.py` | The textbook script for it (trial lenses ±2/±4/±6, power-search check, redo an eye after a complaint) |
| `evaluate5.py`, `overshoot5.py` | 5,000-patient comparison, habits, practice checkpoints; the overshoot by age |
| `record5.py`, `traces5.py`, `sequence5.py` | Record exams step by step; name the steps per occluder visit |
| `clinic_ui5.py`, `scenes5.py`, `film5.py`, `thumbs5.py` | The third video |

```bash
python train.py --clinic --env5 --smile 0.5 --gamma 1.0 --steps 200e6 --ent 0.01 --lr 3e-4 --eval_every 20 --out runs/h
python evaluate5.py && python overshoot5.py && python traces5.py
python film5.py all          # -> out/ai_optometry_v3.mp4
python thumbs5.py
```

## The second video: "Day one at AI Optometry" (`runs/f`, [CLINIC.md](CLINIC.md))

One AI that knows nothing about eyes opens an eye clinic:
* The phoropter is a real photograph (`assets/phoropter.webp`, supplied by the optometrist), its parts
  colour-coded to the AI's 27 buttons.
* Every exam starts at plano: no autorefractor, no retinoscopy. It runs right eye, left eye, then both.
* The AI is paid in clinic points: a happy patient who returns and refers +100; annoyed +30; an unpaid Rx
  check −80; a re-check with a bad review −150; −4 per minute of chair time.
* On day one it presses random buttons and hands out lenses close to plano. Then it practises: term 1 with
  the axis dial always at 180° (it found a shortcut and never turned the axis), term 2 with the dial wherever
  the last patient left it.
* The video compares its exam with the textbook sequence, step by step, and with published data on real
  refractions ([PUBLISHED.md](PUBLISHED.md)).

| File | What it does |
|---|---|
| `env3.py` | The clinic exam: plano start, stages, masked buttons, chair time, the patient's exact answer odds |
| `clinic.py` | The clinic scorecard: outcome chances by prescription error, points, chair time |
| `clinicians3.py` | The textbook script for the clinic exam (Wilkinson 2016 / Borish sequence) and a lock-step runner |
| `evaluate3.py`, `stages3.py` | 5,000-patient comparison, practice checkpoints, repeatability, habits |
| `record3.py`, `traces3.py` | Record exams step by step, with the patient's odds and the AI's odds for every button |
| `clinic_ui.py`, `scenes3.py`, `film3_intro.py`, `film3.py`, `thumbs3.py` | The second video |

```bash
python train.py --clinic --gamma 1.0 --steps 200e6 --shape_frac 0.15 --hard_frac 0.3 --ent 0.01 --lr 3e-4 --eval_every 25 --out runs/f   # term 1 (stopped at 1.63M patients)
python train.py --clinic --gamma 1.0 --init runs/f/ckpt/term1_full.pt --exams_offset 1630473 --steps_offset 100925440 --steps 70e6 --shape_frac 0.4 --hard_frac 0.4 --ent 0.01 --lr 2e-4 --critic_warmup 5 --eval_every 25 --out runs/f2   # term 2
python evaluate3.py && python stages3.py && python traces3.py
python film3.py all          # -> out/ai_optometry.mp4
python thumbs3.py
```

## The first video: "I gave an AI a phoropter" (attempts 1–5)

The AI sits at a phoropter with 16 buttons:
* change sphere, cylinder or axis;
* ask the patient to read the chart;
* ask "1 or 2?" (±0.25 D sphere, or a cross-cylinder flip for axis or power);
* press DONE.

It is never told the true prescription or what any answer means. It is scored only on how close its
final prescription is.

The patients are simulated with real ophthalmic optics:
* power vectors;
* accommodation that hides over-minus;
* a blur-to-acuity model;
* a wave-optics view of the chart.

They also answer like people do: noisily, sometimes "the same", and some prefer minus.

What is physics and what is assumption is spelled out in [ACCURACY.md](ACCURACY.md). Every number in the
video is in [RESULTS.md](RESULTS.md).

## Files

| File | What it does |
|---|---|
| `optics.py` | Power vectors, accommodation, blur → acuity, wave-optics PSF, the error measure (`rx_error`) |
| `chart.py` | The projector letter chart, read letter by letter (from attempt 3) |
| `patients.py` | The simulated clinic population and autorefractor readings; fixed test sets |
| `env.py` | The exam room: a vectorised RL environment (actions, patient answers, scoring) |
| `clinicians.py` | Scripted one-eye examiners: textbook (whole-line and letter chart), from-scratch, "patient-pleaser" |
| `env2.py` | The two-eye exam room (attempt 4): yoked and proximal accommodation, line-size-dependent judgement, prism balance |
| `clinicians2.py` | Scripted two-eye examiners: the textbook, and the optometrist's own routine |
| `policy.py` | The AI (actor-critic MLP) and a runner that scores it on a test set |
| `train.py` | PPO training; checkpoints at milestone exam counts; learning-curve log; `--init` / `--resume` |
| `imitate.py` | Attempt 5, stage 1: behaviour cloning of the textbook script ("watch the textbook") |
| `evaluate.py` | One-eye 5,000-patient comparison and strategy analysis → `data/results_*.json` |
| `evaluate2.py` | Two-eye 5,000-patient comparison → `data/results_d.json` |
| `record.py` | Records one exam move by move (the showcase patient is defined here) |
| `report.py` | Writes `RESULTS.md` from the JSON results |
| `traces.py` | Records every exam the video plays (`data/traces/`) |
| `ui.py`, `scenes.py`, `scenes2.py`, `results_scenes.py`, `film.py` | The video: patient's view, phoropter, charts, assembly and score |

* `runs/a/` holds attempt 1, trained on the plain power-vector distance. That distance counts leftover
  cylinder at half weight, so the AI learned never to turn the axis.
* `runs/b/` holds attempt 2, trained on the fair score: leftover cylinder in full dioptres. Nothing else
  changed.
* `runs/c/` holds attempt 3. It adds training wheels (harder practice patients and a warmer/colder hint,
  both faded out by 30% of training), the letter chart with line buttons, and half its practice with no
  autorefractor.
* `runs/d/` holds attempt 4, the two-eye exam with binocular balance, compared with the textbook and the
  optometrist's own routine. It taught itself and fell short (stopped at 1.27M exams).
* `runs/e/` holds attempt 5. It watched the textbook script examine about 97,000 patients
  (`ckpt/watched.pt`), then practised from there with reinforcement learning.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy pillow numba          # plus ffmpeg, and the Inter font for the video
python clinicians.py                          # score the scripted examiners (seconds)
python train.py --steps 150e6 --cyl_weight 1 --out runs/a   # attempt 1 (stopped at ~14M steps, flat by then)
python train.py --steps 120e6 --out runs/b                  # attempt 2 (stopped at 2M exams, flat by then)
python train.py --steps 120e6 --start mix --flag --letters --hard_frac 0.3 --shape_frac 0.3 --t_max 80 --out runs/c
python train.py --binocular --steps 120e6 --hard_frac 0.3 --shape_frac 0.3 --t_max 120 --eval_every 25 --out runs/d
python evaluate.py runs/a/ckpt/exams_1002464.pt a
python evaluate.py runs/b/ckpt/exams_2002626.pt b
python evaluate.py runs/b/ckpt/exams_2002626.pt b_plano plano
python evaluate.py runs/c/ckpt/final.pt c
python evaluate.py runs/c/ckpt/final.pt c_plano plano
python evaluate2.py runs/d/ckpt/exams_1273013.pt d
python imitate.py --samples 10e6 --out runs/e   # attempt 5, stage 1 (about 6 min)
python train.py --binocular --init runs/e/ckpt/watched.pt --critic_warmup 30 --steps 70e6 --lr 1e-4 --ent 0.002 --t_max 120 --eval_every 25 --out runs/e
python evaluate2.py runs/e/ckpt/final.pt e
python traces.py && python traces.py d        # record the exams the video plays
python report.py
python film.py all                            # renders every section and assembles out/phoropter_ai.mp4
```

The exam traces the video plays are in `data/traces/` (`traces.py`).
