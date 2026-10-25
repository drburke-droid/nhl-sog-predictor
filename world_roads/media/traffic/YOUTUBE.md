# YouTube copy: three routes, one ETA

Upload `three_routes_one_eta.mp4` (6:50, 1920×1080, 30 fps, caption-only, original music).

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

Why? Every car slows every other car. As more people make the trip, the fastest road fills up first and slows down, until it is no quicker than the next road. Then drivers spill onto that one too, and so on.

Why do the times end up exactly level? Because any gap undoes itself. If one route were quicker, drivers would switch to it, and their arrival would slow it down while the routes they left speed up. Switching only stops when no driver can get there sooner by changing route on their own. That is Wardrop's first principle (1952), a Nash equilibrium. Nobody plans it; it's what's left once everyone has picked the quickest route they know.

The fine print:
• It only applies to routes people actually use. At 3 a.m. everyone takes the Midtown Tunnel and the other two stay slower.
• It's about the same trip at the same time. Someone heading elsewhere, or leaving an hour earlier, faces a different balance.
• "Equal" means close enough. Nobody switches to save 30 seconds, ETAs are predictions, and apps round to the minute.
• It isn't the best outcome for everyone.

Nobody solves the equation. Drivers find it by trial and error: if everyone jumps to yesterday's fastest route, the jam just moves around. If a few switch at a time, and nobody bothers to save under a minute, the times settle within a minute of each other.

There's a second reason, too. Your app only shows you the fastest few routes out of hundreds, and the fastest few are always close together. Even at 3 a.m. the three quickest are within two minutes.

Is everyone choosing for themselves the best for everyone? Not quite. When you join a busy road you slow everyone behind you a little, and nobody counts that cost when picking a route. Here a perfect traffic controller would save only about 1%. But a new road can make the gap much bigger.

The mythical tunnel: a thought experiment on the same map. Take two routes, each with one bottleneck (the JFK Expressway on one, the Queens-Midtown Tunnel on the other). With 2,750 drivers an hour both take 38 minutes. Now dig an imaginary tunnel joining the end of the first bottleneck to the start of the second: 3 minutes, never any traffic. At 3 a.m. it's a real shortcut (12 minutes instead of 26), and at rush hour the first driver to try it arrives in 25 minutes instead of 38. So others follow, and every one of them goes through both bottlenecks. When the switching stops, every route takes 52 minutes, 14 minutes longer for everyone, and nobody can do better by going back. That's Braess's paradox (1968). It isn't the rule: with fewer drivers the same tunnel helps. The tunnel, its capacities and the driver numbers are all made up.

0:00 Three routes, one ETA
0:25 Light never hits traffic
0:40 3 a.m.: the routes aren't equal
1:05 Every car slows every other car
1:25 Rush hour: the times meet
2:15 Why the times end up equal
2:45 The fine print
3:15 How drivers find it
4:00 The second reason: what your app shows
4:35 Selfish vs. best for everyone
5:05 The mythical tunnel: Braess's paradox
6:25 You're not the only one

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
2:15 Why the times end up equal
2:45 The fine print
3:15 How drivers find it
4:00 The second reason: what your app shows
4:35 Selfish vs. best for everyone
5:05 The mythical tunnel: Braess's paradox
6:25 You're not the only one
```

**Tags:** traffic, Google Maps, ETA, navigation, New York City, NYC traffic, JFK Airport, Times Square,
Queens-Midtown Tunnel, RFK Bridge, JFK Expressway, Wardrop equilibrium, Nash equilibrium, Braess's paradox, price of
anarchy, game theory, data visualization, OpenStreetMap

**Pinned comment idea:** "Which way do you take from JFK, and does your app ever agree? And would you have
taken the mythical tunnel?"
