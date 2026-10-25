# YouTube copy: three routes, one ETA

Upload `three_routes_one_eta.mp4` (5:30, 1920×1080, 30 fps, caption-only, original music).

**The framing.** The video opens on the line *"You're not the only one who constantly gets 3 or more
completely different routes to your destination with the same ETA."* It shows three real routes from JFK
Airport to Times Square, all at 50 minutes, asks "Coincidence?", and spends the rest of the video
answering it.

**A/B/C test.** The thumbnails carry the hook. The titles are under 100 characters, and the first ~70 show
in search.

| | Thumbnail | Title |
|---|---|---|
| A | `thumb_A.png`: 3 ROUTES. / SAME ETA. beside the three routes into Manhattan and three 50 min pills | You're Not the Only One Who Always Gets 3 Different Routes With the Same ETA |
| B | `thumb_B.png`: NOT A / COINCIDENCE over three travel-time curves merging into one | Why Your Map App Keeps Showing 3 Completely Different Routes With the Same ETA |
| C | `thumb_C.png`: 3 A.M. 26 / 33 / 38 min vs RUSH HOUR 50 / 50 / 50 min | JFK to Times Square: 3 Completely Different Routes, Same ETA. It's Not a Coincidence |

**Description**

```
You're not the only one who constantly gets 3 or more completely different routes to your destination with the same ETA. It isn't a coincidence.

Take JFK Airport to Times Square. There are three very different ways in: straight through Queens and the Queens-Midtown Tunnel, around the north over the RFK Bridge, or around Brooklyn and through the Brooklyn-Battery Tunnel.

🚗 At 3 a.m., on empty roads at the speed limit, they are not equal: 26, 33 and 38 minutes. The Midtown Tunnel wins by seven minutes.
🚗 At rush hour (in our toy traffic model) all three show 50 minutes.

Why? Every car slows every other car. As more people make the trip, the fastest road fills up first and slows down, until it is no quicker than the next road. Then drivers spill onto that one too, and so on. Traffic settles where nobody can get there sooner by switching on their own. That is Wardrop's first principle (1952), a Nash equilibrium. Every route people actually use ends up with the same travel time.

Nobody solves the equation. Drivers find it by trial and error: if everyone jumps to yesterday's fastest route, the jam just moves around. If a few switch at a time, and nobody bothers to save under a minute, the times settle within a minute of each other. Round them to the minute and they look identical.

There's a second reason, too. Your app only shows you the fastest few routes out of hundreds, and the fastest few are always close together. Even at 3 a.m. the three quickest are within two minutes.

Is everyone choosing for themselves the best for everyone? Almost, here: a perfect traffic controller would save about 1%. But sometimes selfish choices are much worse. In Braess's paradox (1968), opening a free shortcut makes all 4,000 drivers 15 minutes slower.

Try it yourself, with every slider: https://drburke-droid.github.io/nhl-sog-predictor/traffic/

0:00 Three routes, one ETA
0:25 Light never hits traffic
0:40 3 a.m.: the routes aren't equal
1:05 Every car slows every other car
1:25 Rush hour: the times meet
2:15 Wardrop's principle (1952)
2:35 How drivers find it
3:20 The second reason: what your app shows
3:55 Selfish vs. best for everyone
4:20 Braess's paradox
5:05 You're not the only one

How it's made: road data from Overture Maps Foundation, which includes OpenStreetMap data (© OpenStreetMap contributors, ODbL). The three routes and their 3 a.m. times are real roads at OpenStreetMap's speed limits (New York's 25 mph default where none is mapped), with one-way rules but no traffic lights, turn restrictions or tolls. Everything about traffic is a toy model: made-up capacities, the standard BPR congestion curve (US Bureau of Public Roads, 1964), and the three routes treated as independent. It is not New York traffic data, and not how any particular app computes its ETAs.
Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/traffic/ACCURACY.md

Music: an original score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 Three routes, one ETA
0:25 Light never hits traffic
0:40 3 a.m.: the routes aren't equal
1:05 Every car slows every other car
1:25 Rush hour: the times meet
2:15 Wardrop's principle (1952)
2:35 How drivers find it
3:20 The second reason: what your app shows
3:55 Selfish vs. best for everyone
4:20 Braess's paradox
5:05 You're not the only one
```

**Tags:** traffic, Google Maps, ETA, navigation, New York City, NYC traffic, JFK Airport, Times Square,
Queens-Midtown Tunnel, RFK Bridge, Wardrop equilibrium, Nash equilibrium, Braess's paradox, price of
anarchy, game theory, data visualization, OpenStreetMap

**Pinned comment idea:** "Which way do you take from JFK, and does your app ever agree? The lab in the
description lets you find the demand at which each route starts to fill."

**Before publishing:** the lab link goes live once `docs/traffic/` is on the GitHub Pages branch (master).
Check that the link opens before the video goes public.
