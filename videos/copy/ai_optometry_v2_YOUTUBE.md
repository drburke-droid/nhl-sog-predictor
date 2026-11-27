# YouTube copy: Day one at AI Optometry (v2, simpler, autorefractor start)

Upload `ai_optometry_v2_{LEN}.mp4`: {LEN_COLON}, 1920×1080, 30 fps, caption-only, with an original synthesized
score. This version replaces `ai_optometry_12m48s.mp4`. It is simpler (one idea per screen), and the AI has two
more practice terms: astigmatism at every angle, then exams starting from an autorefractor. Everything behind
it is in `world_roads/phoropter_ai/`: CLINIC.md, RESULTS_CLINIC.md and PUBLISHED.md.

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

| | Thumbnail | Title |
|---|---|---|
| A | `ai_optometry_thumb_A.png`: the real phoropter, colour-coded; DAY ONE AT AI OPTOMETRY | An AI Opened an Eye Clinic. It Had Never Seen an Eye. |
| B | `ai_optometry_v2_thumb_B.png`: day one vs 3.4 million patients later, one square per patient | I Paid an AI in Happy Patients Until It Learned to Do Eye Exams |
| C | `ai_optometry_v2_thumb_C.png`: AI vs THE TEXTBOOK, 2:45 vs 6:14 in the chair, 78% vs 60% happy | The AI's Eye Exam Took Half the Time. Its Patients Were Happier. |

**Description**

```
This AI has never seen an eye. It doesn't know what a phoropter is. Today it opens an eye clinic.

It gets the phoropter's buttons and its patients' answers: "1", "2", "they look the same", "I can read down to 20/40". Nobody tells it what any of them mean. It's paid the way a clinic is: +100 for a happy patient who comes back and refers a friend, +30 annoyed, −80 for an unpaid Rx check, −150 for an Rx check and a one-star review, and −4 for every minute in the chair.

Day one, it pressed buttons at random. Then it practised on 3.4 million simulated patients. Along the way the clinic changed three times:
1. It never turned the axis (the dial always started at 180°, where most astigmatism sits). Now the dial starts wherever the last patient left it.
2. It only turned the dial back to 180°. A run of practice patients with astigmatism at every angle taught it to use the cross-cylinder.
3. The clinic bought an autorefractor. Every exam now starts at its reading.

On 5,000 new patients, every exam starting at the autorefractor:
• Glasses within 0.25 D of the true prescription: autorefractor alone 10%, textbook 45%, AI 66%
• Happy patients: 29%, 60%, 78%
• Time in the chair: textbook 6:14, AI 2:45
• Points per patient: +4, +25, +69

How its exam differs from the textbook's: the textbook finds each eye's best sphere on its own, reading the chart after almost every quarter step (33 readings per patient). The AI reads the chart 9 times. It leaves each eye fogged and finds the sphere with both eyes open, balancing the two eyes as it unfogs them. Both refine the cylinder with the cross-cylinder.

How does that compare with real optometrists? Two real refractions agree within 0.25 D about 80% of the time; our textbook script 64%, the AI 95%. A real refraction takes about 5 minutes. Our simulated patients are harder than real ones and our scorecard is made up, so compare the AI with the textbook script, not with your own optometrist.

{CHAPTERS}

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
