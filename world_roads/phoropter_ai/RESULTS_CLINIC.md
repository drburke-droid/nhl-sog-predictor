# Results: the clinic video (autorefractor start)

5,000 new simulated patients (seed 2026). Every exam starts at the patient's autorefractor reading (env3.ar_pair). The AI is the final network after 3,426,009 practice patients (`runs/f4/ckpt/final.pt`), choosing its most likely button at each step. The textbook script starts from the same reading. Net points = points for the glasses minus 4 per minute of chair time. See CLINIC.md.

| Examiner | Net points | Points for the glasses | Chair time | Buttons | SE within 0.25 D | Cylinder within 0.25 D | Whole Rx within 0.25 D | Exact (<0.13 D) | Happy | Rx check + review | Over-minused, under 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Autorefractor reading alone | +3.5 | +3.5 | 0:00 | 1 | 57.1% | 44.0% | 10.5% | 5.4% | 28.9% | 36.4% | 58.1% |
| Textbook script | +25.5 | +50.4 | 6:14 | 105 | 75.8% | 88.0% | 45.2% | 34.0% | 59.8% | 16.5% | 43.1% |
| AI | +69.0 | +80.0 | 2:45 | 51 | 96.8% | 85.3% | 65.9% | 51.6% | 77.9% | 3.9% | 12.5% |

Astigmatism of 0.75 D or more, eyes within 0.25 D by orientation:

| Examiner | With the rule | Against the rule | Oblique |
|---|---:|---:|---:|
| Autorefractor reading alone | 10.7% | 10.7% | 7.4% |
| Textbook script | 41.2% | 43.5% | 43.8% |
| AI | 64.7% | 57.2% | 54.3% |

Repeatability (the same 2,000 patients twice, fresh answers; spherical equivalent):

| Examiner | Agree within 0.25 D | Within 0.50 D |
|---|---:|---:|
| Textbook script | 64.5% | 84.6% |
| AI | 95.0% | 99.2% |

Habits per exam (mean):

| | AI | Textbook |
|---|---:|---:|
| sphere lens | 14.5 | 46.1 |
| sphere 1-or-2 | 1.9 | 0.0 |
| chart reading | 8.9 | 32.8 |
| line size | 2.0 | 4.8 |
| cylinder | 4.2 | 3.1 |
| axis | 3.1 | 3.3 |
| cross-cylinder | 10.9 | 10.0 |
| balance | 3.0 | 2.1 |
| next eye | 2.0 | 2.0 |
| cross-cylinder axis questions | 5.0 | 4.7 |

---

# Earlier: the plano-start comparison (after practice term 2)


5,000 new simulated patients (seed 2026), every exam starting at plano, the axis dial where the last patient left it. The AI is the final network after 2,697,295 practice patients (`runs/f2/ckpt/final.pt`), choosing its most likely button at each step. Error is optics.rx_error; SE is the spherical equivalent. Points are clinic points (clinic.py); net = points for the glasses minus 4 per minute of chair time. See CLINIC.md.

| Examiner | Net points | Points for the glasses | Chair time | Buttons | SE within 0.25 D | SE within 0.50 D | Cylinder within 0.25 D | Whole Rx within 0.25 D | Exact (<0.13 D) | Over-minused, under 40 | Over-plussed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plano for everyone | -107.4 | -107.4 | 0:00 | 1 | 12.1% | 21.0% | 41.5% | 2.8% | 0.9% | 29.9% | 54.6% |
| Textbook script | -8.9 | +27.5 | 9:07 | 157 | 70.9% | 84.4% | 78.5% | 39.9% | 29.4% | 45.5% | 23.4% |
| AI | +5.6 | +18.0 | 3:06 | 57 | 89.7% | 97.0% | 51.9% | 30.6% | 20.9% | 13.2% | 14.6% |

Outcome chances (mean over patients) and one sampled outcome per patient:

| Examiner | Happy | Annoyed | Rx check | Bad review | Walked out | Sampled: happy / annoyed / Rx check / review / walked out |
|---|---:|---:|---:|---:|---:|---|
| Plano for everyone | 4.6% | 8.0% | 23.6% | 63.7% | 0.0% | 228 / 396 / 1184 / 3192 / 0 |
| Textbook script | 51.2% | 22.1% | 13.9% | 12.8% | 0.0% | 2564 / 1110 / 696 / 630 / 0 |
| AI | 43.3% | 25.5% | 19.7% | 11.5% | 0.0% | 2156 / 1283 / 1004 / 555 / 2 |

## Repeatability (the same 2,000 patients twice, fresh answers; spherical equivalent, both eyes)

| Examiner | Agree within 0.25 D | Within 0.50 D | 95% range of the difference |
|---|---:|---:|---|
| Textbook script | 59.4% | 77.8% | -2.00 to +2.00 D |
| AI | 87.2% | 95.9% | -0.50 to +0.50 D |

Published, real optometrists: about 80% within 0.25 D (Goss & Grosvenor 1996; Shah et al. 2009: 81%). See PUBLISHED.md.

## Habits per exam (mean)

| | AI | Textbook |
|---|---:|---:|
| sphere lens | 18.5 | 66.2 |
| sphere 1-or-2 | 3.7 | 0.0 |
| chart reading | 11.2 | 47.3 |
| line size | 0.9 | 4.8 |
| cylinder | 2.8 | 6.2 |
| axis | 5.6 | 11.3 |
| cross-cylinder | 7.2 | 16.4 |
| balance | 4.3 | 2.2 |
| next eye | 2.0 | 2.0 |
| minutes: right eye | 1.34 | 3.84 |
| minutes: left eye | 0.86 | 3.83 |
| minutes: both eyes | 0.91 | 1.45 |
| switches between sphere and cylinder work | 8.6 | 5.6 |

## The shortcut (end of practice term 1, 1,630,473 patients)

| Clinic | Net points | SE within 0.25 D | Axis turns per exam | Cylinders at 180° | With-the-rule within 0.25 D | Against-the-rule | Oblique |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dial always at 180° (as practised) | +3.2 | 87.9% | 0.01 | 99.6% | 17.1% | 0.0% | 0.0% |
| Dial where the last patient left it | -31.5 | 69.1% | 0.21 | 0.8% | 1.5% | 0.3% | 0.7% |

## Practice checkpoints (600 test patients each, sampling buttons as it practised)

| Patients practised | Term | Net points | Chair time | Happy | SE within 0.25 D | Whole Rx within 0.25 D | Plano Rx | 1-or-2 | Chart | Cross-cyl | Axis | Balance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | -123.0 | 0:41 | 1.7% | 9.9% | 2.0% | 16.0% | 2.0 | 2.0 | 1.5 | 4.6 | 0.3 |
| 3,339 | 1 | -124.7 | 1:06 | 1.9% | 11.2% | 2.5% | 9.0% | 3.5 | 3.2 | 2.0 | 7.4 | 0.5 |
| 5,505 | 1 | -127.8 | 1:43 | 2.2% | 12.2% | 2.4% | 6.8% | 6.6 | 4.6 | 2.4 | 10.7 | 1.1 |
| 11,158 | 1 | -120.2 | 1:11 | 2.3% | 11.3% | 2.0% | 8.0% | 3.7 | 3.1 | 1.9 | 8.2 | 0.7 |
| 20,733 | 1 | -107.1 | 1:54 | 4.7% | 20.4% | 4.1% | 6.0% | 10.6 | 5.7 | 1.2 | 5.2 | 1.1 |
| 50,564 | 1 | -61.1 | 3:53 | 16.7% | 52.4% | 10.2% | 0.0% | 20.7 | 13.5 | 2.5 | 6.0 | 0.3 |
| 100,533 | 1 | -49.0 | 3:56 | 20.0% | 62.8% | 12.6% | 0.0% | 22.6 | 11.8 | 4.3 | 3.1 | 1.1 |
| 200,116 | 1 | -23.5 | 4:11 | 29.5% | 72.3% | 19.5% | 0.0% | 20.4 | 13.9 | 8.6 | 3.0 | 0.2 |
| 300,106 | 1 | -15.9 | 3:49 | 32.5% | 73.3% | 20.8% | 0.0% | 16.3 | 12.7 | 7.5 | 2.5 | 0.2 |
| 500,299 | 1 | -11.4 | 3:35 | 34.5% | 74.7% | 22.7% | 0.0% | 9.3 | 14.2 | 7.3 | 0.7 | 0.6 |
| 751,126 | 1 | -4.6 | 3:03 | 36.8% | 79.8% | 22.7% | 0.0% | 5.9 | 11.7 | 7.0 | 0.1 | 1.9 |
| 1,000,971 | 1 | -0.5 | 3:05 | 38.9% | 84.5% | 25.6% | 0.0% | 6.5 | 11.2 | 7.1 | 0.1 | 3.1 |
| 1,500,485 | 1 | +1.9 | 3:06 | 40.8% | 88.5% | 27.1% | 0.0% | 4.2 | 11.7 | 7.2 | 0.1 | 3.8 |
| 1,630,473 | 1 | +2.0 | 2:59 | 40.1% | 87.7% | 27.1% | 0.0% | 3.7 | 11.4 | 6.9 | 0.1 | 3.5 |
| 1,750,804 | 2 | -17.9 | 3:00 | 30.9% | 81.2% | 20.1% | 0.0% | 5.1 | 11.9 | 6.7 | 1.8 | 2.2 |
| 2,000,901 | 2 | +2.9 | 3:42 | 41.5% | 89.8% | 29.6% | 0.0% | 5.0 | 14.5 | 8.4 | 8.3 | 3.2 |
| 2,250,870 | 2 | +2.6 | 3:18 | 41.4% | 87.2% | 29.9% | 0.0% | 4.0 | 12.4 | 7.5 | 6.1 | 4.0 |
| 2,500,763 | 2 | +5.6 | 3:11 | 42.9% | 88.4% | 30.6% | 0.0% | 4.1 | 12.3 | 7.4 | 6.0 | 3.1 |
| 2,697,295 | 2 | +5.5 | 3:07 | 42.7% | 88.8% | 32.2% | 0.0% | 3.7 | 11.2 | 7.4 | 5.9 | 4.4 |
