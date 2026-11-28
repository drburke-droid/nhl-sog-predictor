# YouTube copy: an AI learns the eye exam, next to the textbook sequence

Upload `ai_eye_exam_10m40s.mp4`: 10:40, 1920×1080, 30 fps, caption-only, with an original synthesized score.
A standalone video: an optometrist hires an AI to do the refractions; the textbook sequence (based on Borish's
Clinical Refraction) as numbered steps and its published accuracy; patients who aren't perfect; the AI learning
from its mistakes, written in the same step numbers; its finished sequence next to the textbook's. Everything
behind it is in `world_roads/phoropter_ai/` (film7.py, montage7.py, steps7.py; method and numbers in CLINIC5.md
and CLINIC6.md).

The phoropter photo was supplied by the optometrist. Check you have the rights to publish it before uploading.

| | Thumbnail | Title |
|---|---|---|
| A | `ai_eye_exam_thumb_A.png`: AN AI LEARNED THE EYE EXAM, the textbook's step numbers over the AI's | An AI Learned the Eye Exam. Here's Its Sequence vs the Textbook. |
| B | `ai_eye_exam_thumb_B.png`: 2.4 MILLION PATIENTS LATER, 100 patients on day one (2 happy, 68 walk out) vs after practice (69 happy) | 2.4 Million Practice Patients: How an AI Learned "1 or 2?" |
| C | `ai_eye_exam_thumb_C.png`: "WHICH IS CLEARER, 1 OR 2?", two charts, the clearer one glowing | "Which Is Clearer, 1 or 2?" An AI Works Out Its Own Eye Exam |

**Description**

```
"Which is clearer: 1 or 2?" Behind that question is a precise sequence optometrists learn in school. Could an AI work out a sequence of its own?

An optometrist hires an AI to do the refractions, the part of the eye exam that measures your glasses prescription, leaving more time for the rest of the exam: the health of your eyes, how your two eyes work together, contact lenses and more. Every patient in this video is simulated; the AI has never examined a real person.

THE TEXTBOOK
In 1949 Dr. Irvin Borish published Clinical Refraction, written from his teaching notes and his years in practice. Optometry schools still teach a sequence built on it. In this video's step numbers: each eye on its own (cover the other eye, find the lens power and fog, check for astigmatism, cross-cylinder for the astigmatism angle and strength, "+0.25: 1 or 2?"), then both eyes together (balance, clear, prescribe).

It is very precise. Compared with expert benchmark prescriptions, optometrists' results were within ¼ dioptre for overall lens power 81% of the time and within ½ dioptre 97% of the time; astigmatism strength within ¼ dioptre 93% of the time (Shah et al. 2009).

PATIENTS AREN'T PERFECT
When two lenses look almost the same, people guess, and sometimes they point the examiner the wrong way. In the video a green glow shows how likely a patient is to pick each lens. The textbook and the AI face the same patients and the same guesses.

THE AI
It starts with nothing: it doesn't know what a lens does or what the questions mean. It has the same phoropter buttons, it hears the patient's answers, and after each patient it gets a score (a happy patient is good; a re-check, a bad review or a walkout is bad; every minute in the chair costs a little). Choices from an exam that went worse than usual become a little less likely next time; better than usual, a little more likely. That's all.

Patient #1: 276 random buttons, and after 15 minutes the patient walks out. Then 2.4 million practice patients. Its first 1.8 million practice patients smiled when the letters got clearer, a hint; after that the patients were realistic (no smiles, a rough "stop!"), we nudged it to try the cross-cylinder for the astigmatism angle, and made that button turn the angle toward the side the patient picks.

ITS OWN SEQUENCE
Both eyes open first, adding power while the chart improves. Then each eye: a little astigmatism lens, a spin of the angle ("say stop!"), then the textbook's own tools, the cross-cylinder for the angle and the strength. Then both eyes together, taking the extra power off. No astigmatism check, no fogging, no prism balance: same tools, its own order.

On 5,000 new simulated patients (where the true prescription is known): overall lens power within ¼ D 91%, within ½ D 98%; astigmatism angle within 10° for 94% of astigmatic eyes; 69 of 100 patients happy; about 4 minutes in the chair. Simulated patients aren't real ones: the real test of any refraction is a real patient.

Dr. Borish built his sequence from years of teaching and practice. The AI needed 2.4 million simulated patients to find one of its own, and it ended up using the textbook's own tools.

0:00 Which is clearer: 1 or 2?
0:15 An optometrist hires an AI
0:30 The phoropter
0:58 The textbook sequence (Borish)
1:39 How well the textbook works
2:01 Patients aren't perfect
2:42 The AI starts with nothing
2:53 Its score
3:22 How it learns
4:02 Patient #1
4:41 Learning from mistakes
6:15 After 2.4 million practice patients
7:21 The textbook, same patient
8:52 Side by side: the textbook's steps and the AI's
9:38 How close did the AI get?
10:01 When patients are less sure
10:23 Borish and the AI

The optics are real (power vectors, accommodation, a wave-optics model of the blur); the patients' answers are a model and the scorecard is made up. The AI is reinforcement learning (PPO) on a 4-core computer, no GPU. This is a demonstration of machine learning, not a clinical tool, and nothing here is advice about your own eyes. If your vision changes, see your optometrist.

Sources:
Borish IM (1949). Clinical Refraction. Chicago: Professional Press.
Benjamin WJ, ed. (2006). Borish's Clinical Refraction, 2nd ed. Butterworth-Heinemann.
Shah R, Edgar DF, Rabbetts R, Harle DE, Evans BJW (2009). Standardized patient methodology to assess refractive error reproducibility. Optom Vis Sci 86(5): 517–528.
Thibos LN, Wheeler W, Horner D (1997). Power vectors. Optom Vis Sci 74(6): 367–375.
Schulman J et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

#optometry #eyeexam #AI #machinelearning #reinforcementlearning
```
