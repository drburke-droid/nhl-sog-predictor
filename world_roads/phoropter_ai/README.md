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
| `optics.py` | Power vectors, accommodation, blur → acuity, wave-optics PSF |
| `patients.py` | The simulated clinic population and autorefractor readings; fixed test sets |
| `env.py` | The exam room: a vectorised RL environment (actions, patient answers, scoring) |
| `clinicians.py` | Scripted examiners: textbook technique and "patient-pleaser" |
| `policy.py` | The AI (actor-critic MLP) and a runner that scores it on a test set |
| `train.py` | PPO training; checkpoints at milestone exam counts; learning-curve log |
| `evaluate.py` | 5,000-patient comparison and strategy analysis → `data/results_*.json` |
| `record.py` | Records one exam move by move (the showcase patient is defined here) |
| `report.py` | Writes `RESULTS.md` from the JSON results |
| `ui.py`, `scenes.py`, `results_scenes.py`, `film.py` | The video: patient's view, phoropter, charts, assembly and score |

* `runs/a/` holds attempt 1, trained on the plain power-vector distance. That distance counts leftover
  cylinder at half weight, so the AI learned never to turn the axis.
* `runs/b/` holds attempt 2, trained on the fair score: leftover cylinder in full dioptres. Nothing else
  changed.

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy pillow numba          # plus ffmpeg, and the Inter font for the video
python clinicians.py                          # score the scripted examiners (seconds)
python train.py --steps 150e6 --cyl_weight 1 --out runs/a   # attempt 1 (stopped at ~14M steps, flat by then)
python train.py --steps 120e6 --out runs/b                  # attempt 2, ~1.2 h on 4 CPU cores
python evaluate.py runs/a/ckpt/exams_1002464.pt a
python evaluate.py runs/b/ckpt/final.pt b
python traces.py                              # record the exams the video plays
python report.py
python film.py all                            # renders every section and assembles out/phoropter_ai.mp4
```

The exam traces the video plays are in `data/traces/` (`traces.py`).
