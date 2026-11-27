# YouTube copy: an AI's eye exam with more realistic patients, against the Borish method (v4)

Upload `ai_optometry_v4_11m23s.mp4`: 11:23, 1920×1080, 30 fps, caption-only, with an original synthesized score.
The third video's AI walks into a clinic with patients more like real ones (no smile, a rough spin, the
occasional remark, some unreliable) and practises there; it is compared with the Borish method, step by step
and on 5,000 new patients. Everything behind it is in `world_roads/phoropter_ai/`: CLINIC6.md (method and
results), README.md.

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

| | Thumbnail | Title |
|---|---|---|
| A | `ai_optometry_v4_thumb_A.png`: IT LEARNED THE CROSS-CYLINDER, astigmatism axis within 10°: 64% before, 94% after practice | Harder Patients Taught the AI the Cross-Cylinder |
| C | `ai_optometry_v4_thumb_C.png`: AI vs THE BORISH METHOD, 4:01 vs 9:48 in the chair, 69% vs 60% happy | AI vs the Borish Method: 4 Minutes vs 10 |
| D | `ai_optometry_v4_thumb_D.png`: AI vs BORISH, STEP BY STEP: the same patient's steps as coloured bars, 12:12 vs 2:57 | An AI's Eye Exam vs the Borish Method, Step by Step |

**Description**

```
Last time, an AI worked out its own eye exam. But its patients made it easy: they smiled whenever the letters got clearer, and stopped a spinning axis right where it was sharpest.

This time the patients are more like real ones:
• No smile. Now and then a patient says "Oh, that's clear!" or "Hmm, that's not as clear." That's all.
• "Stop!" on a spinning axis comes late and rough, as it does watching a turning dial.
• Some patients answer steadily and some don't: a reliability dial.

Why not start from nothing again? We tried. A new AI with only the occasional remark was stuck at a shortcut after 905,000 patients: minus with both eyes open, and prescribe (19% happy). The smile had been doing a lot of the teaching. So last video's AI, after 1.8 million patients, walks into the new clinic.

One button changed: the cross-cylinder for the axis. Last video's AI had stopped pressing it (its spin was good enough). Now the button also turns the axis toward the side the patient picks, by the textbook's bracketing steps, the way the spin sets the axis at "stop!". And at the start the AI was nudged to try it now and then. Practice decided whether it kept it.

It kept it. After 585,000 patients, every astigmatic eye gets: a little cylinder, a spin to get close, then the cross-cylinder for the axis, then the power. Astigmatism axis within 10°: 64% → 94%.

The Borish method (the standard subjective sequence taught in optometry school, written as a script, same buttons, same patients): right eye, left eye, then both. Trial lenses, fog and clear, the astigmatism check, cross-cylinder axis and power, "+0.25: 1 or 2?" at each endpoint; the other eye; fog both, balance, clear.

On 5,000 new patients, every exam starting at plano (Borish / AI):
• Time in the chair: 9:48 / 4:01
• Happy patients: 60% / 69%
• Points per patient: +1 / +51. The Borish script's glasses earn +41; its 9:48 in the chair costs 39 (every minute costs 4 points).
• Astigmatism axis within 10°: 94% / 94%
• The same patient examined twice, prescriptions within 0.25 D: reliable patients 75% / 91%, typical 68% / 88%, unreliable 61% / 83% (real optometrists: about 80%)

Our simulated patients are harder than real ones and our scorecard is made up, so compare the AI with the Borish script, not with your own optometrist. The script reads the chart after almost every click; a real refraction takes about 5 minutes.

0:00 Last time, an AI worked out its own eye exam
0:16 What changed in the clinic
0:45 What the AI can do
1:26 What the patient can say
2:10 What the AI wants
2:39 How practice works
3:19 Why not start from nothing again?
3:34 One button changed
3:52 Day one in the new clinic
4:49 Learning on the job
5:19 After 585,000 patients: the AI's exam
6:24 The Borish method
6:56 The Borish method's exam
8:37 Side by side: the Borish method's steps and the AI's
9:26 5,000 new patients
9:53 Where the points come from
10:13 Same patient, tested twice
10:38 And real optometrists?
11:08 The fine print

Every patient is simulated. The optics are real (power vectors, accommodation, a wave-optics model of the blur); the answers are a model. The AI is reinforcement learning (PPO) on a 4-core computer, no GPU. It has never examined a real person. This is a demonstration of machine learning, not a clinical tool, and nothing here is advice about your own eyes. If your vision changes, see your optometrist.

Sources:
Benjamin WJ, ed. (2006). Borish's Clinical Refraction, 2nd ed. Butterworth-Heinemann.
Thibos LN, Wheeler W, Horner D (1997). Power vectors. Optom Vis Sci 74(6): 367–375.
Schulman J et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Goss DA, Grosvenor T (1996). Reliability of refraction: a literature review. J Am Optom Assoc 67(10): 619–630.
Shah R, Edgar DF, Rabbetts R, Harle DE, Evans BJW (2009). Standardized patient methodology to assess refractive error reproducibility. Optom Vis Sci 86(5): 517–528.
Carracedo G et al. (2018). Transl Vis Sci Technol 7(4): 11. Otero C, Aldaba M, Pujol J (2019). J Optom 12(2): 74–83.

#optometry #eyeexam #AI #machinelearning #reinforcementlearning
```
