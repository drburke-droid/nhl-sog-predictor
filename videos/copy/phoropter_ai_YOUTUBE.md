# YouTube copy: I gave an AI a phoropter

Upload `phoropter_ai_11m42s.mp4`: 11:42, 1920×1080, 30 fps, caption-only with an original synthesized
score. Everything behind it, and every number on screen, is in `world_roads/phoropter_ai/` (README,
ACCURACY, RESULTS).

**A/B/C test.** Each thumbnail is paired with a title that doesn't repeat its words.

| | Thumbnail | Title |
|---|---|---|
| A | `phoropter_ai_thumb_A.png`: the phoropter, cross-cylinder flipped in; I GAVE AN AI / A PHOROPTER | An AI Taught Itself to Do an Eye Exam. Here's What It Got Wrong (and Right). |
| B | `phoropter_ai_thumb_B.png`: AI vs MY EYE EXAM, ME 44% / AI 70% | I Trained an AI on 500,000 Eye Exams, Then Tested It Against My Own Technique |
| C | `phoropter_ai_thumb_C.png`: THE AI LEARNED THE CROSS-CYLINDER, blurry chart → sharp chart | Can an AI Learn Subjective Refraction? Five Attempts, One Phoropter |

**Description**

```
I gave an AI a phoropter. It had never seen an eye, a lens, or an eye chart, and nobody told it what "1 or 2?" means. It practised on simulated patients and was scored only on how close each final prescription was to the truth.

Attempt 1 learned to push plus and almost never over-minused, but it never once turned the axis. My score was the reason: it counted leftover cylinder at half weight. Scored fairly, every prescription is a vector: spherical equivalent, plus a cylinder arrow whose direction is twice the axis.

Attempt 2 had the fair score and still never turned the axis. Turning the axis only pays off as a pair of moves (ask the cross-cylinder, then turn the way the patient points), and trial and error never found the pair.

Attempt 3 got training wheels: harder practice patients and a warmer/colder hint, both faded out early. It learned the whole technique, cross-cylinder included. On 5,000 new simulated patients it got 87% within 0.25 D, against 59% for a scripted textbook refraction. With no autorefractor at all it still got 85%, against 51%.

Then the real exam: two eyes, one projected line at a time, binocular balance. Patients judge blur best on the right size of letters, and behind a phoropter a covered-up patient keeps focusing a little. I scripted my own routine too. On its own, the AI stalled at 28%. So attempt 5 first watched the textbook script examine 97,103 patients and copied its moves, then practised. On 5,000 two-eyed simulated patients:
• AI: 70% of eyes within 0.25 D, 58% exact, 71 moves
• Textbook script: 45%, 34%, 102 moves
• My routine: 44%, 33%, 66 moves

What it learned on its own: push plus early in young eyes, don't trust "minus looks clearer", follow the cross-cylinder, and use far fewer chart readings.

What the simulation taught me: binocular balance equalizes the two eyes but doesn't centre them. If both eyes end their one-eye testing equally over-minused, a fogged balance says "same". A binocular plus check (MPMVA with both eyes) is what catches it. Adding one to my routine raised it from 39% to 44%.

0:00 I gave an AI a phoropter
0:10 The rules
0:46 Attempt 1: practice, scored at the end
2:07 The learning curve
2:19 It beat the textbook?
2:32 It never turned the axis
2:46 Not so fast: my score
3:00 Every prescription is a vector
3:23 Scored fairly
3:35 Attempt 2: a fair score
3:58 Why it was still stuck
4:12 Attempt 3: training wheels
4:28 Exam: following the cross-cylinder
5:29 5,000 new patients
5:56 No autorefractor
7:42 Attempt 4: the real exam
7:58 My own routine
8:53 Teaching itself the full exam
9:04 Attempt 5: watch the textbook first
9:16 Exam: both eyes and binocular balance
10:18 The same patient, three examiners
10:42 5,000 two-eyed patients
11:10 What practice changed
11:26 The fine print

How it was made: every patient is simulated. The optics are real: power vectors, accommodation, and a wave-optics model of the blur the patient sees. The patients' answers are a model (noise, "they look the same", a preference for minus, letter-by-letter reading, proximal accommodation behind the phoropter). The AI is reinforcement learning (PPO) on a 4-core CPU, no GPU. Code, every assumption, and the numbers behind every chart are in the project's ACCURACY and RESULTS files. Voice: none. Music: original synthesized score.

This AI has never examined a real person. It is a demonstration of machine learning, not a clinical tool, and nothing here is advice about your own eyes. If your vision changes, see your optometrist.

Sources:
Thibos LN, Wheeler W, Horner D (1997). Power vectors: an application of Fourier analysis to the description and statistical analysis of refractive error. Optometry and Vision Science 74(6): 367–375.
Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Ng AY, Harada D, Russell S (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML 1999.
Applegate RA, Lakshminarayanan V (1993). Parametric representation of Stiles–Crawford functions. JOSA A 10(7): 1611–1623.

#optometry #eyeexam #AI #machinelearning #reinforcementlearning
```
