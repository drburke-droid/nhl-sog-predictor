# Results

Simulated test patients drawn independently of training (seed 2026). Every examiner in a table saw the same patients. Error is optics.rx_error: the length of (spherical-equivalent error, leftover cylinder), with the leftover cylinder in full dioptres. Exact means under 0.13 D. Over-minused means the spherical equivalent is at least 0.25 D more minus than the truth. See ACCURACY.md for every assumption.

## One eye, whole-line chart, starting from the autorefractor (attempts 1 and 2)

| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | Over-plussed |
|---|---:|---:|---:|---:|---:|---:|
| Autorefractor reading alone | 10.6% | 5.1% | 0.498 | 0.0 | 58.0% | 18.7% |
| Script: patient-pleaser | 36.3% | 26.9% | 0.415 | 12.5 | 74.6% | 5.6% |
| Script: textbook | 55.2% | 45.4% | 0.244 | 33.2 | 25.8% | 24.6% |
| AI, attempt 1 (1,002,464 exams) | 52.7% | 37.0% | 0.227 | 14.2 | 7.8% | 5.8% |
| AI, attempt 2 (2,002,626 exams) | 54.1% | 38.4% | 0.220 | 14.4 | 8.7% | 3.7% |

Attempt 1 on the half-weight-cylinder score it was trained on (the plain power-vector distance):

| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | Over-plussed |
|---|---:|---:|---:|---:|---:|---:|
| Script: textbook | 64.8% | 48.6% | 0.214 | 33.2 | 25.8% | 24.6% |
| AI, attempt 1 | 85.0% | 51.3% | 0.144 | 14.2 | 7.8% | 5.8% |

## One eye, letter chart (attempt 3)

Starting from the autorefractor:

| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | Over-plussed |
|---|---:|---:|---:|---:|---:|---:|
| Autorefractor reading alone | 10.6% | 5.1% | 0.498 | 0.0 | 58.0% | 18.7% |
| Script: textbook (letter chart) | 59.0% | 50.0% | 0.224 | 39.0 | 26.3% | 19.6% |
| AI, attempt 3 | 87.4% | 79.7% | 0.092 | 24.3 | 6.4% | 3.5% |

No autorefractor (starting at plano, up to 80 moves):

| Examiner | Within 0.25 D | Exact | Mean error (D) | Moves per exam | Over-minused, under 40 | Over-plussed |
|---|---:|---:|---:|---:|---:|---:|
| Prescribe nothing (plano) | 2.2% | 0.8% | 1.893 | 0.0 | 29.7% | 54.4% |
| Script: textbook from scratch (letter chart) | 51.4% | 42.6% | 0.517 | 56.9 | 40.0% | 20.1% |
| AI, attempt 2 (never practised without one) | 11.9% | 10.2% | 0.997 | 31.8 | 15.8% | 19.8% |
| AI, attempt 3 | 84.5% | 77.7% | 0.176 | 41.6 | 7.8% | 4.0% |

## Habits, attempt 1

| Move | Mean uses per exam | Exams using it |
|---|---:|---:|
| Sphere +0.25 | 0.92 | 55.3% |
| Sphere -0.25 | 0.76 | 43.5% |
| Cylinder +0.25 | 0.82 | 50.2% |
| Read the chart | 0.97 | 95.5% |
| 1 or 2? (+0.25) | 5.28 | 100.0% |
| 1 or 2? (-0.25) | 1.53 | 71.2% |
| Cross-cyl: power | 2.92 | 100.0% |
| Done | 1.00 | 99.9% |

Checkpoints during training (2,000 of the test patients):

| Checkpoint | Practice exams | Within 0.25 D | Exact | Moves |
|---|---:|---:|---:|---:|
| exams_0.pt | 0 | 7.5% | 3.3% | 11.5 |
| exams_4150.pt | 4,150 | 10.1% | 5.3% | 60.0 |
| exams_10872.pt | 10,872 | 9.0% | 3.9% | 60.0 |
| exams_30064.pt | 30,064 | 10.2% | 5.5% | 1.0 |
| exams_107079.pt | 107,079 | 10.2% | 5.4% | 1.0 |
| exams_301936.pt | 301,936 | 16.6% | 10.6% | 3.3 |
| exams_1002464.pt | 1,002,464 | 51.7% | 36.9% | 14.2 |

## Habits, attempt 3

| Move | Mean uses per exam | Exams using it |
|---|---:|---:|
| Sphere +0.25 | 1.68 | 76.1% |
| Sphere -0.25 | 1.63 | 83.4% |
| Sphere +1.00 | 0.07 | 6.3% |
| Sphere -1.00 | 0.03 | 2.1% |
| Cylinder -0.25 | 0.52 | 37.4% |
| Cylinder +0.25 | 1.05 | 64.0% |
| Axis +5° | 0.89 | 44.9% |
| Axis -5° | 1.49 | 54.1% |
| Axis +15° | 0.23 | 13.7% |
| Read the chart | 2.57 | 100.0% |
| 1 or 2? (+0.25) | 3.19 | 100.0% |
| 1 or 2? (-0.25) | 2.37 | 97.9% |
| Cross-cyl: axis | 3.71 | 100.0% |
| Cross-cyl: power | 3.80 | 100.0% |
| Done | 0.99 | 99.0% |
| Read this line | 0.02 | 2.0% |

Checkpoints during training (2,000 of the test patients):

| Checkpoint | Practice exams | Within 0.25 D | Exact | Moves |
|---|---:|---:|---:|---:|
| exams_0.pt | 0 | 6.6% | 3.4% | 30.4 |
| exams_3498.pt | 3,498 | 10.2% | 5.5% | 1.0 |
| exams_14237.pt | 14,237 | 10.2% | 5.5% | 1.0 |
| exams_34926.pt | 34,926 | 10.2% | 5.5% | 1.0 |
| exams_102163.pt | 102,163 | 13.6% | 6.2% | 6.3 |
| exams_300805.pt | 300,805 | 17.2% | 8.4% | 11.3 |
| exams_1000488.pt | 1,000,488 | 83.5% | 75.2% | 25.9 |
| exams_2000177.pt | 2,000,177 | 85.0% | 76.8% | 24.7 |
| exams_3000418.pt | 3,000,418 | 87.5% | 79.2% | 24.5 |
| exams_3509820.pt | 3,509,820 | 87.0% | 79.5% | 24.5 |
