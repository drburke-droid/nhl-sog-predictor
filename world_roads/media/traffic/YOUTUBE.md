# YouTube copy: three routes, one ETA

Upload `three_routes_one_eta.mp4` (7:18, 1920×1080, 30 fps, caption-only, original music).

**The framing.** The video opens on the line *"You're not the only one who constantly gets 3 or more
completely different routes to your destination with the same ETA."* It shows three real routes from JFK
Airport to Times Square, all at 50 minutes, asks "Coincidence?", and spends the rest of the video
answering it.

**A/B/C test.** The thumbnails carry the hook. The titles are under 100 characters, and the first ~70 show
in search. D is the strongest click, because it is a real screenshot, but it needs yours (below).

| | Thumbnail | Title |
|---|---|---|
| A | `thumb_A.png`: 3 ROUTES. / SAME ETA. beside the three routes into Manhattan and three 50 min pills | You're Not the Only One Who Always Gets 3 Different Routes With the Same ETA |
| B | `thumb_B.png`: NOT A / COINCIDENCE over three travel-time curves merging into one | Why Your Map App Keeps Showing 3 Completely Different Routes With the Same ETA |
| C | `thumb_C.png`: 3 A.M. 26 / 33 / 38 min vs RUSH HOUR 50 / 50 / 50 min | JFK to Times Square: 3 Completely Different Routes, Same ETA. It's Not a Coincidence |
| D | `thumb_D.png`: your own Google Maps screenshot (three routes, one ETA) in a phone frame, beside 3 ROUTES. / SAME ETA. / WHY? | Why Does Google Maps Keep Giving Me 3 Different Routes With the Same ETA? |

**Thumbnail D needs a real screenshot.** Take it yourself in the Google Maps app, at a moment when it
offers three routes with the same time (rush hour is the best bet; any trip works, it needn't be JFK).
Save it as `media/traffic/gmaps_screenshot.png` and run `python thumbs.py D`. To crop it or ring the three
times, add `gmaps_screenshot.json`, e.g. `{"crop": [0, 300, 1179, 2100], "circles": [[620, 900, 70]]}`
(screenshot pixels). Until then `out/thumb_D_layout.png` shows the layout with an empty phone.
* Don't edit the screenshot's times or routes: the thumbnail promises a real one.
* Keep Google's logo and the map-data attribution inside the crop. Google allows Maps screenshots in
  online video with attribution (see its Geo guidelines), and doesn't allow implying it endorses the video.
* Crop out anything personal: home or work labels, saved places, your location.

**Description**

```
You're not the only one who constantly gets 3 or more completely different routes to your destination with the same ETA. It isn't a coincidence.

Take JFK Airport to Times Square. There are three very different ways in: straight through Queens and the Queens-Midtown Tunnel, around the north over the RFK Bridge, or around Brooklyn and through the Brooklyn-Battery Tunnel.

🚗 At 3 a.m., on empty roads at the speed limit, they are not equal: 26, 33 and 38 minutes.
🚗 At rush hour (in our toy traffic model) all three show 50 minutes.

Why? Every car slows every other car. As more people make the trip, the fastest road fills up first and slows down, until it is no quicker than the next road. Then drivers spill onto that one too, and so on.

Why do the times end up exactly level? Because any gap undoes itself: drivers switch to a quicker route, which slows it down while the routes they left speed up. Switching only stops when no driver can get there sooner by changing route on their own. That is Wardrop's first principle (1952), a Nash equilibrium. Nobody plans it; it's what's left once everyone has picked the quickest route they know.

The fine print:
• It only applies to routes people use: at 3 a.m. everyone takes the Midtown Tunnel.
• It's about one trip at one time. Leave an hour earlier and the balance differs.
• "Equal" means close enough. Nobody switches to save 30 seconds, ETAs are predictions, and apps round to the minute.
• It isn't the best outcome for everyone.

Nobody works this balance out, so how do drivers find it? Imagine a week when it hadn't settled. On Monday most people take the Midtown Tunnel, because it's quickest on empty roads, so it jams: 83 minutes, while the RFK Bridge takes 36. Drivers find out their route wasn't the quickest. If half of everyone switches to the RFK Bridge on Tuesday, it jams instead, and day after day the jam just moves from road to road. Real drivers are calmer: if only 1 in 50 tries another route each day, and nobody switches to save under a minute, the times creep together and settle within a minute in about two weeks.

There's a second reason, too. There are hundreds of ways from JFK to Times Square, taking 26 to 61 minutes at 3 a.m. Your app never shows you hundreds: it offers a few that are quick and genuinely different, like our three. The slow ones, and near-copies of quicker ones, never make the list, so what you see is already the close calls.

Is everyone choosing for themselves the best for everyone? Not quite: joining a busy road slows everyone behind you, and nobody counts that cost. Here a perfect traffic controller would save only about 1%. But a new road can make the gap much bigger.

The mythical tunnel: a thought experiment on the same map. A bottleneck is a place where the road narrows, say from 4 lanes to 2. In light traffic it copes; in heavy traffic more cars arrive than it can pass, a queue builds, and the wait keeps growing. Give each of two routes one bottleneck (the JFK Expressway on one, the Queens-Midtown Tunnel on the other), and say the rest never jams. With 2,750 drivers an hour both routes take 38 minutes. Now dig an imaginary tunnel joining the end of the first bottleneck to the start of the second: 3 minutes, never any traffic. At 3 a.m. it's a real shortcut (12 minutes instead of 26), and at rush hour the first driver to try it arrives in 25 minutes instead of 38. So others follow, and every one of them goes through both bottlenecks, so both queues grow (the JFK Expressway from 2 to 16 minutes, the Queens-Midtown Tunnel from 15 to 29). When the switching stops, every route takes 52 minutes, 14 minutes longer for everyone, and nobody can do better by going back. That's Braess's paradox (1968). It isn't the rule: with fewer drivers the same tunnel helps. The tunnel, its capacities and the driver numbers are all made up.

0:00 Three routes, one ETA
0:25 Light never hits traffic
0:40 3 a.m.: the routes aren't equal
1:05 Every car slows every other car
1:25 Rush hour: the times meet
2:15 Why the times end up equal
2:45 The fine print
3:15 What if it hadn't settled? How drivers find it
4:05 The second reason: what your app shows
4:42 Selfish vs. best for everyone
5:12 The mythical tunnel: Braess's paradox
6:52 You're not the only one

How it's made: road data from Overture Maps Foundation, which includes OpenStreetMap data (© OpenStreetMap contributors, ODbL). The three routes and their 3 a.m. times are real roads at the speed limit (one-way rules; no traffic lights, turns or tolls). Everything about traffic is a toy model (made-up capacities, the standard BPR congestion curve): not New York traffic data, and not how any app computes its ETAs.
Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/traffic/ACCURACY.md

Music: an original score, synthesized for this video (no samples).
```

If you use thumbnail D, add a line to the description: `Thumbnail: a Google Maps screenshot.` (it still fits under
YouTube's 5,000-character limit).

**Chapters**

```
0:00 Three routes, one ETA
0:25 Light never hits traffic
0:40 3 a.m.: the routes aren't equal
1:05 Every car slows every other car
1:25 Rush hour: the times meet
2:15 Why the times end up equal
2:45 The fine print
3:15 What if it hadn't settled? How drivers find it
4:05 The second reason: what your app shows
4:42 Selfish vs. best for everyone
5:12 The mythical tunnel: Braess's paradox
6:52 You're not the only one
```

**Tags:** traffic, Google Maps, ETA, navigation, New York City, NYC traffic, JFK Airport, Times Square,
Queens-Midtown Tunnel, RFK Bridge, JFK Expressway, Wardrop equilibrium, Nash equilibrium, Braess's paradox, price of
anarchy, game theory, data visualization, OpenStreetMap

**Pinned comment idea:** "Which way do you take from JFK, and does your app ever agree? And would you have
taken the mythical tunnel?"
