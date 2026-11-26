# YouTube copy: Day one at AI Optometry

Upload `ai_optometry_{LEN}.mp4`: {LEN_COLON}, 1920×1080, 30 fps, caption-only, with an original synthesized score.
Everything behind it is in `world_roads/phoropter_ai/`: CLINIC.md, RESULTS_CLINIC.md and PUBLISHED.md. Every
number on screen is there too.

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

**A/B/C test.** Each thumbnail is paired with a title that doesn't repeat its words.

| | Thumbnail | Title |
|---|---|---|
| A | `ai_optometry_thumb_A.png`: the real phoropter, colour-coded; DAY ONE AT AI OPTOMETRY | An AI Opened an Eye Clinic. It Had Never Seen an Eye. |
| B | `ai_optometry_thumb_B.png`: day one vs 2.7 million patients later, one square per patient | I Paid an AI in Happy Patients Until It Learned to Do Eye Exams |
| C | `ai_optometry_thumb_C.png`: AI vs THE TEXTBOOK, 3:06 vs 9:07 in the chair, 43% vs 51% happy | The AI Was 3x Faster Than the Textbook Eye Exam. Was It Better? |

**Description**

```
This AI has never seen an eye. It doesn't know what a phoropter is. Today it opens an eye clinic.

It gets the phoropter's buttons (sphere, cylinder, axis, the cross-cylinder, the prism, the occluder, the projector chart, and "Prescribe") and its simulated patients' answers: "1 is clearer", "they look the same", "I can read down to 20/40". Nobody tells it what any of them mean. Every exam starts at plano: no autorefractor, no retinoscopy.

It's paid the way a clinic is:
• +100: happy, comes back, refers a friend
• +30: annoyed, might not come back
• −80: an unpaid Rx check
• −150: an Rx check and a one-star review
• −4 for every minute in the chair

Day one, it pressed buttons at random and handed out lenses close to plano. Then it practised on 2.7 million simulated patients.

Halfway through, it had found a shortcut. The axis dial always started at 180°, where most astigmatism sits, so it never turned it. We made the dial start wherever the last patient left it, as in a real practice, and it kept practising.

On 5,000 new patients, against a scripted textbook refraction (the Borish sequence: MPMVA, cross-cylinder, binocular balance):
• Clinic points per patient: AI +6, textbook −9
• Chair time: AI 3:06, textbook 9:07
• Spherical equivalent within 0.25 D: AI 90%, textbook 71%
• Whole prescription within 0.25 D: AI 31%, textbook 40%
• Happy patients: AI 43%, textbook 51%

What it worked out on its own: overshoot with minus, then come back with plus and leave each eye fogged. Find the sphere with both eyes open, balancing the two eyes while unfogging. Read the chart 11 times, not 47.

What it never learned: the cross-cylinder axis test. It turns the dial back toward 180° and hopes, so against-the-rule and oblique astigmatism go uncorrected. It wins on points because it's fast, which is exactly what we paid it for.

How does that compare with real optometrists? Two real refractions agree within 0.25 D about 80% of the time; ours: textbook script 59%, AI 87%. A real refraction takes about 5 minutes. About 2% of real glasses come back. Our simulated patients are harder than real ones, so compare the AI with the textbook script, not with your own optometrist.

{CHAPTERS}

How it was made: every patient is simulated. The optics are real: power vectors, accommodation, and a wave-optics model of the blur the patient sees. The patients' answers are a model, and the clinic scorecard is made up. The AI is reinforcement learning (PPO) on a 4-core CPU, no GPU.

This AI has never examined a real person. It is a demonstration of machine learning, not a clinical tool, and nothing here is advice about your own eyes. If your vision changes, see your optometrist.

Sources:
Thibos LN, Wheeler W, Horner D (1997). Power vectors. Optom Vis Sci 74(6): 367–375.
Schulman J et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Wilkinson ME (2016). Minus Cylinder Subjective Refraction Techniques for Clinicians. University of Iowa, EyeRounds.org.
Goss DA, Grosvenor T (1996). Reliability of refraction: a literature review. J Am Optom Assoc 67(10): 619–630.
Shah R, Edgar DF, Rabbetts R, Harle DE, Evans BJW (2009). Standardized patient methodology to assess refractive error reproducibility. Optom Vis Sci 86(5): 517–528.
Bullimore MA, Fusaro RE, Adams CW (1998). The repeatability of automated and clinician refraction. Optom Vis Sci 75(8): 617–622.
Carracedo G et al. (2018). Transl Vis Sci Technol 7(4): 11. Otero C, Aldaba M, Pujol J (2019). J Optom 12(2): 74–83. Kabat AG et al. (2019). Clin Ophthalmol 13: 1807–1813.
Freeman CE, Evans BJW (2010). Investigation of the causes of non-tolerance to optometric prescriptions for spectacles. Ophthalmic Physiol Opt 30(1): 1–11.
Hrynchak P (2006). Prescribing spectacles: reasons for failure of spectacle lens acceptance. Ophthalmic Physiol Opt 26(1): 111–115.
Bist J et al. (2021). Ophthalmic Physiol Opt 41(3): 610–622.

#optometry #eyeexam #AI #machinelearning #reinforcementlearning
```
