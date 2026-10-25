# YouTube copy: two routes home, one ETA

Upload `two_routes_one_eta.mp4` (7:18, 1920×1080, 30 fps, caption-only, original music).

**The framing.** The video opens on the question *"Ever noticed your drive home has two routes with almost
the exact same ETA, heading completely different ways?"* It answers *"You're not unique. And your house isn't
magically placed at some sweet spot in the traffic."* The example is a real evening drive home: from work near
Times Square to a house in Bayside, Queens. One way goes east under the river (14.9 mi); the other goes north
through the Bronx and back over the Throgs Neck Bridge (17.7 mi). Both show 29 minutes. The rest of the video
explains why, and why it's usually two routes rather than three.

**A/B/C test.** The thumbnails carry the hook. The titles are under 100 characters, and the first ~70 show
in search. D is the strongest click, because it is a real screenshot, but it needs yours (below).

| | Thumbnail | Title |
|---|---|---|
| A | `thumb_A.png`: 2 ROUTES / HOME. / SAME ETA. beside the two routes and two 29 min pills | Why Do 2 Completely Different Routes Home Have the Same ETA? |
| B | `thumb_B.png`: NOT A / COINCIDENCE over two travel-time curves merging into one | Your Commute Has 2 Routes With the Same ETA. It's Not a Coincidence |
| C | `thumb_C.png`: 3 A.M. 21 / 26 min vs RUSH HOUR 29 / 29 min | Two Ways Home, Same ETA: Why Traffic Does This |
| D | `thumb_D.png`: your own Google Maps screenshot (two or more routes, one ETA) in a phone frame, beside 2 ROUTES. / SAME ETA. / WHY? | Why Does Google Maps Keep Giving Me 2 Routes With the Same ETA? |

**Thumbnail D needs a real screenshot.** Take it yourself in the Google Maps app on your own drive home, at a
moment when two routes show the same time (the evening rush is the best bet). Save it as
`media/traffic/gmaps_screenshot.png` and run `python thumbs.py D`. To crop it or ring the times, add
`gmaps_screenshot.json`, e.g. `{"crop": [0, 300, 1179, 2100], "circles": [[620, 900, 70]]}` (screenshot
pixels). Until then `thumb_D_layout.png` shows the layout with an empty phone.
* Don't edit the screenshot's times or routes: the thumbnail promises a real one.
* Keep Google's logo and the map-data attribution inside the crop. Google allows Maps screenshots in
  online video with attribution (see its Geo guidelines), and doesn't allow implying it endorses the video.
* Crop out anything personal: your home address or label, saved places, your location.

**Description**

```
Ever noticed your drive home has two routes with almost the exact same ETA, heading completely different ways, one of them miles longer? You're not unique. And your house isn't magically placed at some sweet spot in the traffic. It happens on almost every busy trip.

Take an evening drive home from Times Square to a house in Bayside, Queens. One way goes east, under the river through the Queens-Midtown Tunnel (14.9 miles). The other goes north through the Bronx and back over the Throgs Neck Bridge (17.7 miles). A third goes south round Brooklyn and through the Battery Tunnel (23.6 miles).

🚗 At 3 a.m., on empty roads at the speed limit, they are not equal: 21, 26 and 38 minutes.
🚗 In the evening rush (in our toy traffic model) the first two both show 29 minutes.

Why? Every car slows every other car. As more people head home, the fastest road fills up first and slows down, until it is no quicker than the next road. Then drivers spill onto that one too, and the two slow down together.

Why do the times end up level? Because any gap undoes itself: drivers switch to a quicker route, which slows it down while the route they left speeds up. Switching only stops when no driver can get there sooner by changing route on their own. That is Wardrop's first principle (1952), a Nash equilibrium. Nobody plans it.

Why two and not three? It only applies to routes people actually use. The Battery Tunnel route stays at 38 minutes: slower, so nobody takes it, and nothing pulls it into line. On a heavier night it would fill too, and then all three would match.

The rest of the fine print: it's about one trip at one time; "equal" means close enough (nobody switches to save 30 seconds, and apps round to the minute); and it isn't the best outcome for everyone.

How do drivers find the balance? Imagine a week when it hadn't settled. On Monday most take the Midtown Tunnel, so it jams: 42 minutes, while the Throgs Neck route takes 26. If half of everyone switches on Tuesday, that one jams instead, and the jam just moves from road to road. Real drivers are calmer: if only 1 in 50 tries another route each day, the times settle within a minute in about two weeks.

There's a second reason, too. There are hundreds of ways home, taking 21 to 67 minutes at 3 a.m. Your app offers only a few that are quick and genuinely different, so what you see is already the close calls.

Is everyone choosing for themselves the best for everyone? Not quite: joining a busy road slows everyone behind you, and nobody counts that cost. Here a perfect traffic controller would save about 6%. And a new road can make it worse.

The mythical tunnel: a thought experiment on the same map. A bottleneck is a place where the road narrows, say from 4 lanes to 2: in heavy traffic a queue builds. Give each route one (the Queens-Midtown Tunnel and the Throgs Neck Bridge) and say the rest never jams. With 2,500 drivers an hour both routes take 27 minutes. Now dig an imaginary tunnel from the Queens-Midtown Tunnel straight to the Throgs Neck Bridge: 3 minutes, never any traffic. The first driver to try it saves 7 minutes, so others follow, and every one of them goes through both bottlenecks. When the switching stops, every route takes 34 minutes, 7 minutes longer for everyone. That's Braess's paradox (1968). It isn't the rule: with fewer drivers the same tunnel helps. The tunnel and its numbers are made up.

0:00 Two routes home, one ETA
0:30 Light never hits traffic
0:45 3 a.m.: the routes aren't equal
1:10 Every car slows every other car
1:30 The evening rush: two routes meet
2:15 Why the times end up equal
2:45 The fine print: why two, not three
3:15 What if it hadn't settled? How drivers find it
4:05 The second reason: what your app shows
4:42 Selfish vs. best for everyone
5:12 The mythical tunnel: Braess's paradox
6:52 You're not unique

How it's made: road data from Overture Maps Foundation, which includes OpenStreetMap data (© OpenStreetMap contributors, ODbL). The routes and their 3 a.m. times are real roads at the speed limit (one-way rules; no traffic lights, turns or tolls). Everything about traffic is a toy model (made-up capacities, the standard BPR congestion curve): not New York traffic data, and not how any app computes its ETAs.
Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/traffic/ACCURACY.md

Music: an original score, synthesized for this video (no samples).
```

If you use thumbnail D, add a line to the description: `Thumbnail: a Google Maps screenshot.`

**Chapters**

```
0:00 Two routes home, one ETA
0:30 Light never hits traffic
0:45 3 a.m.: the routes aren't equal
1:10 Every car slows every other car
1:30 The evening rush: two routes meet
2:15 Why the times end up equal
2:45 The fine print: why two, not three
3:15 What if it hadn't settled? How drivers find it
4:05 The second reason: what your app shows
4:42 Selfish vs. best for everyone
5:12 The mythical tunnel: Braess's paradox
6:52 You're not unique
```

**Tags:** traffic, commute, drive home, Google Maps, ETA, navigation, New York City, NYC traffic, Times Square,
Queens, Bayside, Queens-Midtown Tunnel, Throgs Neck Bridge, the Bronx, Wardrop equilibrium, Nash equilibrium,
Braess's paradox, price of anarchy, game theory, data visualization, OpenStreetMap

**Pinned comment idea:** "Do you have two routes home with the same ETA? Which way do you take, and would you
have taken the mythical tunnel?"
