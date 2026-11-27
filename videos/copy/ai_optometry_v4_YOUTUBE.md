# YouTube copy: an AI's eye exam with more realistic patients, against the Borish method (v4)

Upload `ai_optometry_v4_LENGTH.mp4`: LENGTH, 1920×1080, 30 fps, caption-only, with an original synthesized score.
The third video's AI walks into a clinic with patients more like real ones (no smile, a rough spin, the
occasional remark, some unreliable) and practises there; it is compared with the Borish method, step by step
and on 5,000 new patients. Everything behind it is in `world_roads/phoropter_ai/`: CLINIC6.md (method and
results), README.md.

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

| | Thumbnail | Title |
|---|---|---|
| A | `ai_optometry_v4_thumb_A.png`: IT FOUND THE CROSS-CYLINDER, astigmatism axis within 10°: 64% before, 94% after practice | Harder Patients Taught the AI the Cross-Cylinder |
| C | `ai_optometry_v4_thumb_C.png`: AI vs THE BORISH METHOD, 4:01 vs 9:48 in the chair, 69% vs 60% happy | AI vs the Borish Method: 4 Minutes vs 10 |
| D | `ai_optometry_v4_thumb_D.png`: the Borish method's steps vs the AI's for the same patient, coloured bars | An AI's Eye Exam vs the Borish Method, Step by Step |

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

CHAPTERS

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
