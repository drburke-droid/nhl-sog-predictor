# YouTube copy: Why 60? Why 24?

**Upload `why60_4k_master.mp4`:**
* 8:23, 3840×2160, 30 fps, H.264;
* on-screen captions, no narration;
* a synthesized score at 60 beats per minute, −16 LUFS.

At 8:23 the film is long enough for mid-roll ads, which YouTube allows from 8:00.

The master is over GitHub's 100 MB limit, so it is sent separately; its SHA-256 is in `SHA256.txt`. To rebuild
it, run `./render_all.sh` then `python assemble.py` in `world_roads/sixty`. The repository keeps a 1080p
reference copy, `videos/time/why_60_why_24_8m23s_1080p.mp4`. Upload the 4K file: thin gold lines and small type hold up much better in
YouTube's 4K-source encodes, even for viewers watching at 1080p.

**A/B/C test** (thumbnails are in this folder and in `videos/time/thumbnails/`)

| | Thumbnail | Title |
|---|---|---|
| A | `why60_thumb_A.png`: the clock, "WHY 60? WHY 24? and not 100?" | Why Is Time Counted in 60s and 24s, When Everything Else Is in 10s? |
| B | `why60_thumb_B.png`: the √2 tablet, "YOUR CLOCK RUNS ON BABYLONIAN MATH" | Your Clock Runs on 4,000-Year-Old Babylonian Math |
| C | `why60_thumb_C.png`: the French decimal clock at noon, "10 HOURS A DAY. France tried it in 1793." | France Tried a 10-Hour Day. Here's Why Time Never Went Metric. |

**Description**

```
Almost everything we measure goes in tens: 1,000 metres to a kilometre, 100 cents to a dollar. Time doesn't. A minute is 60 seconds, an hour 60 minutes, a day 24 hours. Why?

The sixties come from ancient Mesopotamia. Scribes there counted in base 60 and wrote numbers with just two wedge shapes pressed into clay: one for 1, one for 10. A school tablet from about 1700 BC (YBC 7289) gives the square root of 2 as 1;24,51,10 in base 60, correct to within six ten-millionths. Nobody knows for sure why they chose 60, but it splits evenly 12 ways: a third of an hour is 20 minutes, while a third of a metre is 33.333… cm.

The 24 comes from Egypt. Star tables split the night into 12 hours, each marked by a star group (a decan) rising, probably because about 12 of the 36 decans rose in the full dark of night. One ancient text counts 10 hours of sunlight on a shadow clock, plus an hour of twilight at each end: 12 more. These hours stretched and shrank with the seasons.

Greek astronomers put the two together. Hipparchus used 24 equal hours, and astronomers wrote fractions in sixtieths. In medieval Latin, the first sixtieth was the "pars minuta prima", the first small part: the minute. The next was the "pars minuta secunda", the second small part: the second. Around AD 1000, al-Bīrūnī went on to thirds and fourths.

Mechanical clocks brought equal hours to European towns from the 1300s. Huygens's pendulum clock (1656) made minute hands worth having. In 1793, revolutionary France tried a decimal day: 10 hours of 100 minutes of 100 seconds, with noon at 5 o'clock. It was suspended in April 1795, by the same law that brought in the metric system. Today the second is defined by the caesium-133 atom, and below a second, time is decimal after all.

0:00 Why 60? Why 24?
0:27 Counting in tens
0:49 Numbers pressed into clay
1:44 Why sixty?
2:31 The Babylonian day: 12 bēru, 360 UŠ
3:16 Egypt: 12 hours of night, 12 of day
4:11 Minutes and seconds
5:04 Clocks: one hand, then two, then three
5:40 France's decimal time
6:34 Other ways to cut a day
7:12 The atomic second
7:58 Every clock face is a fossil

How it's made: every frame is rendered in code. The cuneiform is pressed into a simulated clay surface and lit. The score is synthesized at exactly 60 beats per minute, so every beat is one second: the ticks land with the second hands on screen, and a small bell rings at each full minute of the film. Method, sources and what's simplified: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/sixty/ACCURACY.md

Sources: D. Fowler & E. Robson, Historia Mathematica 25:366 (1998). M. A. Powell, Visible Language 6:5 (1972). J. M. Steele, SCIAMVS 14:3 (2013). R. A. Parker, "Egyptian Astronomy, Astrology, and Calendrical Reckoning", Dictionary of Scientific Biography (1978). S. Symons & E. Tasker, Scientific American (Oct 2015). M. Lombardi, Scientific American (Mar 2007). Al-Bīrūnī, The Chronology of Ancient Nations, trans. E. Sachau (1879). G. Dohrn-van Rossum, History of the Hour (1996). M. Sôma, K. Kawabata & K. Tanikawa, PASJ 56:887 (2004). E. Burgess (trans.), Sūrya Siddhānta (1860). BIPM, The International System of Units, 9th ed. (2019).

Music: an original score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 Why 60? Why 24?
0:27 Counting in tens
0:49 Numbers pressed into clay
1:44 Why sixty?
2:31 The Babylonian day: 12 bēru, 360 UŠ
3:16 Egypt: 12 hours of night, 12 of day
4:11 Minutes and seconds
5:04 Clocks: one hand, then two, then three
5:40 France's decimal time
6:34 Other ways to cut a day
7:12 The atomic second
7:58 Every clock face is a fossil
```

**Tags:** time, why 60 minutes, why 24 hours, base 60, sexagesimal, Babylonian mathematics, cuneiform, YBC 7289,
square root of 2, ancient Egypt, decans, sundial, shadow clock, Hipparchus, minute, second, history of
timekeeping, pendulum clock, Huygens, decimal time, French Revolution, metric system, caesium, atomic clock

**Pinned comment idea:** "Would you switch to decimal time? Lunch at 5, home by 7.5 🕔"

**Shorts.** These cut cleanly from the film:
* the clay tablet (0:49 to 1:44): wedges for 1 and 10, then √2 to six ten-millionths;
* "pars minuta prima, pars minuta secunda" (4:35 to 4:53): why minutes and seconds are called that;
* French decimal time (5:40 to 6:34): noon at 5 o'clock.
