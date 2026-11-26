# I gave an AI a phoropter

A reinforcement-learning agent teaches itself subjective refraction on simulated patients, and a video
of how it learned.

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
