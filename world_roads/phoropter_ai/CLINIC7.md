# Version 7: the full phoropter, a patient who only talks now and then, millions of practice patients

A retrain from random weights. Nothing about the exam is a hint: no smile, no spin-and-stop, no button that
turns the axis for the AI. What it gets is what an examiner gets: the phoropter, the chart, and a patient.

Code: `env7.py`, `train.py --clinic --env7`, `run_train7.sh`.

## 1. The exam (`env7.py`)

| What the AI can do | |
|---|---|
| Sphere | +0.25, −0.25 (and the dioptre dial, ±1.00), each eye or both eyes together |
| Cylinder | −0.25, +0.25 |
| Axis knob | ±1°, ±5°, ±15°: any axis, dialled in by the AI |
| Jackson cross-cylinder | axis "1 or 2?" (which way to turn), power "1 or 2?" (more or less), and at 180/90 and 45/135 to look for astigmatism. Questions only: nothing moves unless the AI moves it |
| Sphere "1 or 2?" | with +0.25 or −0.25 |
| Prism | split the chart: "top or bottom?" (top: right eye) |
| Occluder | right eye, left eye, both eyes, round again |
| Chart | "read the smallest line you can"; isolate a line ("read this line"); smaller or bigger letters |
| Prescribe | the final check: the patient reads with each eye; "my left eye is still blurry" keeps the exam going |

What the patient says, unasked, only now and then (each patient has their own chattiness, 5–40% of the changes
they notice; a change they notice is a line or more of acuity, through their usual noise):

| After a lens change | Remark |
|---|---|
| better, still more than 1.5 lines from the best they can see | "That's better." |
| better, within 1.5 lines of their best | "That's clear." |
| better, within half a line of their best | "That's really clear!" |
| worse, within 1.5 lines of their best | "That's worse." |
| worse, 1.5–4 lines from their best | "That's not clear." |
| worse, more than 4 lines from their best | "That's really blurry." |
| worse, more than 7 lines from their best (most patients say it) | "Is there something on this lens?" |

Patients differ in reliability (answer noise and lapse rate × a factor drawn log-uniform 0.5–2.5). The exam
starts at plano; at 15 minutes the patient walks out.

**The score**: the clinic's points at the end (happy +100, annoyed +30, re-check −80, bad review −150, walkout
−150, 4 points a minute in the chair), and each remark counts in proportion to how much clearer the patient's
vision got since their previous remark (`--remark 0.5`: 5 points a line; summed over an exam it is the change
from the start to the last remark, so going back and forth earns nothing). Reading the chart earns nothing by
itself: it is information, and the question for this run is whether the AI learns to use it.

## 2. Training that runs for days

```bash
OUT=runs/p ./run_train7.sh >> runs_p.log 2>&1 &
```

`run_train7.sh` runs `train.py --clinic --env7 --remark 0.5 --gamma 1.0 --steps 2e9 --ent 0.01 --lr 3e-4
--eval_every 40 --out runs/p` and keeps it going:

* `latest.pt` (weights, optimizer, counters) every 20 updates; if training stops for any reason the supervisor
  resumes from it after 10 seconds (tested: killed at update 21, resumed at update 21). The log is cut back to
  the resume point, so it never has gaps or duplicates.
* A non-finite loss or weight rolls the update back to `latest.pt` instead of crashing.
* Checkpoints at milestones (0, 2k, 5k … 1M, 1.5M … 4M, 5M, 6M, 7M, 8M, 10M, 12M, 15M, 20M, 25M, 30M patients),
  about 1.4 MB each; disk use stays small.
* 2 billion buttons is about 20 million patients once exams settle near 100 buttons; at roughly 10,000 buttons a
  second that is 2–3 days. It can be stopped earlier (`touch runs/p/STOP`) once the scores level off.
* A machine restart stops everything; the run is restarted from `latest.pt` with the same command.
