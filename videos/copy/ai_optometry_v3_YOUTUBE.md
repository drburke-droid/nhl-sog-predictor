# YouTube copy: an AI works out its own eye exam (v3, no mentor, plano start)

Upload `ai_optometry_v3_10m17s.mp4`: 10:17, 1920×1080, 30 fps, caption-only, with an original synthesized
score. One AI, from nothing, every exam starting at plano: no autorefractor, no mentor hints, no curriculum.
Its only help is the patient. Everything behind it is in `world_roads/phoropter_ai/`: CLINIC5.md (method and
results), README.md.

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

**A correction for the first video** (`ai_optometry_12m48s.mp4`). While building this one, we found that the
textbook script in the first video mishandled patients who can't read the top line of the chart at plano
(high myopes, older hyperopes): if −2.00 and +2.00 were both still unreadable, it drifted back toward plano.
The script here tries ±2, ±4 and ±6 D first. The first video's textbook numbers were affected; the second
video (autorefractor start) was not. A pinned comment for the first video, if you want one:

```
Correction: the textbook script in this video mishandled patients who couldn't read the top line at plano (it could drift back toward plano when ±2.00 didn't help), which made its numbers look worse than they should. It's fixed in the follow-up, where the AI also learns with no mentor at all.
```

| | Thumbnail | Title |
|---|---|---|
| D | `ai_optometry_v3_thumb_D.png`: AN AI WORKED OUT ITS OWN EYE EXAM, the textbook's 17 steps vs the AI's 6, coloured bars | An AI Worked Out Its Own Eye Exam. It's Not the Textbook's. |
| C | `ai_optometry_v3_thumb_C.png`: AI vs THE TEXTBOOK, 9:19 vs 3:14 in the chair, 56% vs 65% happy | The AI's Eye Exam Took a Third of the Time |
| B | `ai_optometry_v3_thumb_B.png`: day one vs 1.8 million patients later, one square per patient | No Teacher, No Textbook: An AI Learns to Do Eye Exams |

**Description**

```
This AI has never seen an eye. It doesn't know what a phoropter is. Today it opens an eye clinic.

No autorefractor. No hints. Nobody shows it the textbook. It gets 30 buttons, including two ways to look for astigmatism: the cross-cylinder at 180/90 and 45/135, or put in some cylinder and spin the axis until the patient says "stop". Its only help is the patient: "1, much clearer", "they look the same", a smile when the letters get clearer, and, at the final check, "my left eye is still blurry". It's paid the way a clinic is: +100 for a happy patient, −150 for a bad review or a walkout, −4 for every minute in the chair.

Day one, it pressed buttons at random. Almost every patient walked out. Then it practised on 1.8 million simulated patients, and worked out its own exam:
1. Both eyes open first. Minus until the chart stops improving, and then some more. The younger the patient, the further it goes: about 3 D extra for teenagers, none over 60. A young eye focuses through it.
2. Right eye, left covered: a quarter of cylinder, spin the axis until the patient says stop, then the cross-cylinder for power. Back and forth.
3. Left eye, right covered: the same.
4. Both eyes open: take the extra minus off, a quarter at a time. Final check.
It took this route in 98% of exams. It never used the 180/90 check, the cross-cylinder for the axis, or the prism balance.

The textbook (written as a script, same buttons, same patients): right eye, left eye, then both. Find the sphere, fog and clear, astigmatism check, cross-cylinder axis and power, sphere again; the other eye; fog both, balance, clear.

On 5,000 new patients, every exam starting at plano (textbook / AI):
• Time in the chair: 9:19 / 3:14
• Glasses within 0.50 D of the true prescription: 78% / 91%
• Happy patients: 56% / 65%
• Points per patient: −3 / +51
• Two exams of the same eye agreeing within 0.25 D: 64% / 87% (real optometrists: about 80%)

Our simulated patients are harder than real ones and our scorecard is made up, so compare the AI with the textbook script, not with your own optometrist. The script reads the chart after almost every click; a real refraction takes about 5 minutes.

0:00 An AI opens an eye clinic
0:12 What the AI can do
0:53 What the patient can say
1:36 What the AI wants
2:05 How practice works
2:45 Day one
3:46 Learning on the job
4:15 After 1.8 million patients: the AI's exam
6:04 What the textbook says
6:35 The textbook's exam
8:02 Side by side: the textbook's steps and the AI's
8:51 5,000 new patients
9:27 And real optometrists?
10:02 The fine print

Every patient is simulated. The optics are real (power vectors, accommodation, a wave-optics model of the blur); the answers are a model. The AI is reinforcement learning (PPO) on a 4-core computer, no GPU. It has never examined a real person. This is a demonstration of machine learning, not a clinical tool, and nothing here is advice about your own eyes. If your vision changes, see your optometrist.

Sources:
Thibos LN, Wheeler W, Horner D (1997). Power vectors. Optom Vis Sci 74(6): 367–375.
Schulman J et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Wilkinson ME (2016). Minus Cylinder Subjective Refraction Techniques for Clinicians. University of Iowa, EyeRounds.org.
Goss DA, Grosvenor T (1996). Reliability of refraction: a literature review. J Am Optom Assoc 67(10): 619–630.
Shah R, Edgar DF, Rabbetts R, Harle DE, Evans BJW (2009). Standardized patient methodology to assess refractive error reproducibility. Optom Vis Sci 86(5): 517–528.
Carracedo G et al. (2018). Transl Vis Sci Technol 7(4): 11. Otero C, Aldaba M, Pujol J (2019). J Optom 12(2): 74–83.

#optometry #eyeexam #AI #machinelearning #reinforcementlearning
```
