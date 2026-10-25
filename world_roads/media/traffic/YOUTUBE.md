# YouTube copy: two routes home, one ETA

Upload `two_routes_one_eta.mp4` (7:40, 1920×1080, 30 fps): narrated by ElevenLabs' Lily voice, with
captions and original music.

**Narration.** `VOICEOVER.md` is the script; `traffic/lily.py` recorded it line by line with the Lily voice
(`media/traffic/voice/`), and `assemble.py` mixes each line in on its cue, with the music lowered under it. The
captions stay, so it still works with the sound off. The description credits the AI voice; also tick YouTube's
altered-content setting if it applies to you when you upload.

**The framing.** The video opens on the question *"Ever wonder why your maps app always offers two or more
completely different routes home, with pretty much the exact same time? Why isn't there just one no-brainer
route?"* Then a real one: your Apple Maps screenshot, Times Square to Bayside leaving at 6:01 p.m., with three
routes at 59 min, 1 hr and 1 hr 1 min. Then *"You're not alone. And your house isn't magically placed at some
sweet spot in the traffic."* The worked example is the same drive home in a toy traffic model: east under the
river (14.9 mi) or north through the Bronx and over the Throgs Neck Bridge (17.7 mi), 29 minutes either way. The
rest of the video explains why, and why it's often two routes rather than three.

**A/B test.** YouTube's Test & Compare takes up to three thumbnails at once; good first sets are D, E, G (the
real screenshot, the bottleneck, the real numbers) or A, E, F. F leads with the tunnel paradox rather than the
same-ETA question, so pair it with its own title. The thumbnails carry the hook. The titles are under 100 characters, and the first ~70 show
in search. D is the strongest click: it is your real screenshot.

| | Thumbnail | Title |
|---|---|---|
| A | `thumb_A.png`: 2 ROUTES / HOME. / SAME ETA. beside the two routes and two 29 min pills | Why Do 2 Completely Different Routes Home Have the Same ETA? |
| B | `thumb_B.png`: NOT A / COINCIDENCE over two travel-time curves merging into one | Your Commute Has 2 Routes With the Same ETA. It's Not a Coincidence |
| C | `thumb_C.png`: 3 A.M. 21 / 26 min vs RUSH HOUR 29 / 29 min | Two Ways Home, Same ETA: Why Traffic Does This |
| D | `thumb_D.png`: your Apple Maps screenshot (59 min, 1 hr, 1 hr 1 min, ringed in gold) beside 3 ROUTES. / SAME ETA. / WHY? | Why Does My Maps App Give Me 3 Routes Home With the Same ETA? |
| E | `thumb_E.png`: SAME ETA? / THIS IS WHY. over a glowing lane drop, 4 lanes into 2, with a red queue backed up behind the bottleneck | Why Your Two Routes Home Always Have the Same ETA |
| F | `thumb_F.png`: NEW ROAD. / EVERYONE / SLOWER. and +7 MIN beside the map with the mythical tunnel | Why a New Road Can Make Everyone's Drive Home Slower |
| G | `thumb_G.png`: 59 · 60 · 61 MIN in huge gold numerals over your blurred Apple Maps screenshot: 3 WAYS HOME / COINCIDENCE? | 59, 60, 61 Minutes: Why 3 Different Routes Home Have the Same ETA |

**Thumbnail D and the opening use your screenshot** (`maps_screenshot.png`, Apple Maps, Times Square to
Bayside, leave 6:01 p.m.). It is cropped and scaled, never edited; the gold rings and the caption are drawn
around it, and `maps_screenshot.json` sets the crop, the rings and the words. Apple doesn't publish a
screenshot permission as explicit as Google's, but showing a screenshot of the app to discuss it is common;
credit it in the description (the line is in it). To use another screenshot, replace the PNG, adjust the JSON
and run `python thumbs.py D` (and `python assemble.py render hook` for the opening).

**Description**

```
Ever wonder why, when you check your maps app for the best route home in the evening rush, there are always two or more completely different routes with pretty much the exact same time? Why isn't there just one no-brainer route that's by far the best? You're not alone. And your house isn't magically placed at some sweet spot in the traffic.

Here's a real one: Apple Maps, Times Square to Bayside, Queens, leaving at 6:01 p.m. Three completely different ways home: 59 minutes, 1 hour, 1 hour 1 minute.

Our toy traffic model uses the same drive home on real roads. One way goes east, under the river through the Queens-Midtown Tunnel (14.9 miles). The other goes north through the Bronx and back over the Throgs Neck Bridge (17.7 miles). A third goes south round Brooklyn and through the Battery Tunnel (23.6 miles).

🚗 At 3 a.m., on empty roads at the speed limit, they are not equal: 21, 26 and 38 minutes.
🚗 In the evening rush (in our toy traffic model) the first two both show 29 minutes.

Why? Every car slows every other car. As more people head home, the fastest road fills up first and slows down, until it is no quicker than the next road. Then drivers spill onto that one too, and the two slow down together.

Why do the times end up level? Because any gap undoes itself: drivers switch to a quicker route, which slows it down while the route they left speeds up. Switching only stops when no driver can get there sooner by changing route on their own. That is Wardrop's first principle (1952), a Nash equilibrium. Nobody plans it.

Why sometimes two, sometimes three? It only applies to routes people actually use. In our model the Battery Tunnel route stays at 38 minutes: slower, so nobody takes it, and nothing pulls it into line. On a heavier night it would fill too, and then all three would match, like the real screenshot.

The rest of the fine print: it's about one trip at one time; "equal" means close enough (nobody switches to save 30 seconds, and apps round to the minute); and it isn't the best outcome for everyone.

How do drivers find the balance? Imagine a week when it hadn't settled. On Monday most take the Midtown Tunnel, so it jams: 42 minutes, while the Throgs Neck route takes 26. If half of everyone switches on Tuesday, that one jams instead, and the jam just moves from road to road. Real drivers are calmer: if only 1 in 50 tries another route each day, the times settle within a minute in about two weeks.

There's a second reason, too. There are hundreds of ways home, taking 21 to 67 minutes at 3 a.m. Your app offers only a few that are quick and genuinely different, so what you see is already the close calls.

Is everyone choosing for themselves the best for everyone? Not quite: joining a busy road slows everyone behind you, and nobody counts that cost. Here a perfect traffic controller would save about 6%. And a new road can make it worse.

The mythical tunnel: a thought experiment on the same map. A bottleneck is a place where the road narrows, say from 4 lanes to 2: in heavy traffic a queue builds. Give each route one (the Queens-Midtown Tunnel and the Throgs Neck Bridge) and say the rest never jams. With 2,500 drivers an hour both routes take 27 minutes. Now dig an imaginary tunnel from the Queens-Midtown Tunnel straight to the Throgs Neck Bridge: 3 minutes, never any traffic. The first driver to try it saves 7 minutes, so others follow, and every one of them goes through both bottlenecks. When the switching stops, every route takes 34 minutes, 7 minutes longer for everyone. That's Braess's paradox (1968). It isn't the rule: with fewer drivers the same tunnel helps. The tunnel and its numbers are made up.

0:00 Two or more routes home, one ETA
0:50 Light never hits traffic
1:05 3 a.m.: the routes aren't equal
1:30 Every car slows every other car
1:50 The evening rush: two routes meet
2:35 Why the times end up equal
3:05 The fine print: sometimes two, sometimes three
3:35 What if it hadn't settled? How drivers find it
4:27 The second reason: what your app shows
5:05 Selfish vs. best for everyone
5:35 The mythical tunnel: Braess's paradox
7:15 You're not alone

How it's made: road data from Overture Maps Foundation, which includes OpenStreetMap data (© OpenStreetMap contributors, ODbL). The routes and their 3 a.m. times are real roads at the speed limit (one-way rules; no traffic lights, turns or tolls). Everything about traffic is a toy model (made-up capacities, the standard BPR congestion curve): not New York traffic data, and not how any app computes its ETAs.
Method, sources and every number: https://github.com/drburke-droid/nhl-sog-predictor/blob/master/world_roads/traffic/ACCURACY.md

Screenshot in the opening and thumbnail: Apple Maps, captured by the creator.
Narration: AI voice (ElevenLabs, "Lily").
Music: an original score, synthesized for this video (no samples).
```

**Chapters**

```
0:00 Two or more routes home, one ETA
0:50 Light never hits traffic
1:05 3 a.m.: the routes aren't equal
1:30 Every car slows every other car
1:50 The evening rush: two routes meet
2:35 Why the times end up equal
3:05 The fine print: sometimes two, sometimes three
3:35 What if it hadn't settled? How drivers find it
4:27 The second reason: what your app shows
5:05 Selfish vs. best for everyone
5:35 The mythical tunnel: Braess's paradox
7:15 You're not alone
```

**Tags:** traffic, commute, drive home, Google Maps, ETA, navigation, New York City, NYC traffic, Times Square,
Queens, Bayside, Apple Maps, Queens-Midtown Tunnel, Throgs Neck Bridge, the Bronx, Wardrop equilibrium, Nash equilibrium,
Braess's paradox, price of anarchy, game theory, data visualization, OpenStreetMap

**Pinned comment idea:** "Do you have two or more routes home with the same ETA? Screenshot it and tell us
which way you take, and whether you'd have taken the mythical tunnel."
