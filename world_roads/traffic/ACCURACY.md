# Accuracy and sources: two routes home, one ETA

The video asks why a drive home so often offers two completely different routes with almost the same ETA.
It answers with two separate effects. The first is traffic: drivers crowd the quicker roads until no one gains
by switching (Wardrop's user equilibrium). The second is selection: the app lists only a few quick, distinct
routes. It explains why it is usually two routes rather than three (a slower route nobody takes is never
pulled into line), the fine print of "the same ETA", the system optimum, the price of anarchy, and Braess's
paradox, shown on the same map with a mythical tunnel.

The trip is an evening drive home **from Times Square to a house in Bayside, Queens**.

**What is real, and what is a toy model:**
* **Real:** the road geometry, the three routes and their free-flow times, and the opening's maps-app screenshot.
* **Toy model:** everything about traffic. There are no New York traffic counts, no real app ETAs and no
  measured capacities anywhere in it.

**The real screenshot (opening and thumbnail D).** The creator's own Apple Maps screenshot
(`media/traffic/maps_screenshot.png`): directions from Times Square to Bayside with the departure set to
6:01 p.m. on 5 November. Apple Maps offers three routes: 59 min (east along the Long Island Expressway, like
route A; tagged "Avoids Sporting Event"), 1 hr (past LaGuardia) and 1 hr 1 min (north through the Bronx, like
route B). These are Apple's predictions with its own traffic data, shown cropped and scaled but otherwise
unedited; the gold rings and labels are drawn around them. They are not outputs of our model, whose times are
shorter because it uses speed-limit free-flow times and made-up capacities. The screenshot illustrates the
question; the model explains it.

Every number on screen comes from `scenario.py` (written to `media/traffic/video_numbers_nyc.json`), using the
tested functions in `model.py`.

The same pipeline also runs on a Calgary trip, Rocky Ridge to Seton (`TRAFFIC_CITY=calgary`). Its routes
are in `media/traffic/routes_calgary.json`.

## 1. The roads and the three routes (real data, idealized driving)

**Data.** Overture Maps Foundation transportation segments (release 2026-09-23.1, derived from
OpenStreetMap). They are fetched for a box that covers Queens, Brooklyn, Manhattan, the south Bronx and the
Hudson shore (`city_routes.py fetch`), with 330,709 segments. The routable street network covers motorway
down to residential and living streets, with no service roads, tracks or paths: 413,953 nodes and
12,932 km of road. City settings are in `cities.py`.

**Directions.** One-way rules are honoured. Overture's `access_restrictions` with `heading` give the
direction of travel. Segments closed to cars are dropped. Turn restrictions (Manhattan has many) are *not*
modelled.

**Speed.** Each road is timed at its speed limit.
* **Mapped limits:** OpenStreetMap's `maxspeed` where it is mapped. That is 33% of all road length, but
  47–63% of motorway, primary and secondary length, and 77–85% of the three routes.
* **Defaults elsewhere:**
  * New York's 25 mph default for city streets;
  * 50 mph on expressways, 40 mph on other limited-access roads, 30 mph on ramps.

**Free-flow time = length ÷ speed limit.** There is no traffic, no traffic-light delay and no time for
turns. Tolls (the Queens-Midtown Tunnel, the RFK and Throgs Neck bridges, the Battery Tunnel) are ignored,
and so is Manhattan's congestion charge. Real drivers weigh tolls as well as time, one more reason real ETAs
differ. This gives the time an ideal driver could make at 3 a.m.; real trips take longer.

**The trip.**
* **Start ("work"):** the nearest street junction to Times Square (40.7570° N, 73.9866° W).
* **Finish ("home"):** the nearest residential street junction to a point in Bayside, Queens
  (40.7686° N, 73.7771° W), near 35th Avenue. It stands for any house there; no real address is used.

**Finding alternatives.** We used the via-node method used by route planners:
1. Run a Dijkstra search forward from the start and backward from the finish.
2. Every junction *v* then defines a candidate route: fastest to *v*, fastest from *v*.
3. Reject candidates that double back on themselves.
4. Reject candidates whose stretch around *v* is not itself close to optimal (in the spirit of Abraham et
   al., "Alternative routes in road networks", 2013).
5. Keep the quickest 20 that share at most 80% of their road with each other.

Most of these are variations of route A: they leave Manhattan through the Queens-Midtown Tunnel and use the
Long Island Expressway.

**The three in the video** are therefore the fastest route home through each of three river crossings out
of Manhattan: east, north and south. Only nodes inside the bridge or tunnel count, so each route has to
actually cross it. Twelve candidate homes across Queens were compared before picking Bayside, where the two
quickest crossings share almost no road.

| Route | Free-flow time | Length | Roads (km on each) |
|---|---:|---:|---|
| A, Midtown Tunnel: east, under the river | 21.0 min | 23.9 km (14.9 mi) | Queens-Midtown Tunnel 2.2, Long Island Expressway 14.8, Clearview Expressway 2.2 |
| B, Throgs Neck Bridge: north, through the Bronx | 26.3 min | 28.5 km (17.7 mi) | FDR Drive 6.0, Robert F. Kennedy Bridge 1.0, Bruckner Expressway 6.4, Cross Bronx Expressway 2.4, Throgs Neck Bridge 3.4, Clearview Expressway 2.6 |
| C, Battery Tunnel: south, round Brooklyn | 37.7 min | 37.9 km (23.6 mi) | 7th Avenue 2.6, West Street 2.2, Brooklyn-Battery Tunnel 2.9, Brooklyn-Queens Expressway 10.4, Long Island Expressway 11.2, Clearview Expressway 2.2 |

**How much they share:**
* A and B share 2% (0.5 km): a few blocks at each end, near Times Square and in Bayside. Both use the
  Clearview Expressway, but in opposite directions (A comes north from the LIE, B comes south from the
  Throgs Neck), so they share none of it.
* C shares 63% of its road with A: the Long Island Expressway and the Clearview into Bayside. It is the
  long way round, and in the video it is the slower route nobody takes.

Route B is found as "the fastest route through the RFK Bridge". It uses the bridge's Manhattan–Bronx span,
then crosses the Bronx to the Throgs Neck Bridge, so the video names it after the Throgs Neck. The
Brooklyn-Battery Tunnel is officially the Hugh L. Carey Tunnel; the video uses the older name, which
OpenStreetMap carries.

## 2. The congestion curve (a standard simplification)

Each route's travel time follows the Bureau of Public Roads curve:

T(f) = T₀ · (1 + 0.15 · (f / c)⁴)

* **Variables:** *f* is the route's flow and *c* its capacity, both in vehicles per hour.
* **Source:** U.S. Bureau of Public Roads, *Traffic Assignment Manual*, 1964.
* **What it is:** a smooth planning curve. It is not a physical traffic model. It has no queues, no
  spillback, no stop-and-go waves and no gridlock. It keeps rising past capacity (f/c > 1) instead of
  breaking down.
* **Capacities: made up.** 2,000, 1,500 and 1,100 vehicles per hour for A, B and C. They are not
  measurements of these roads.
* **Demand: also made up.** The evening rush is 4,000 drivers per hour making this kind of trip. It was
  chosen so that two routes are in use and the third is not.

**The routes are treated as independent.** Real routes share roads with each other and with every other
trip in the city, which couples their times.

## 3. User equilibrium (exact)

* **Wardrop's first principle:** J. G. Wardrop, "Some theoretical aspects of road traffic research",
  *Proceedings of the Institution of Civil Engineers*, Part II, 1952.
* **The condition:** at equilibrium no driver can arrive sooner by switching routes alone. Every route
  that carries traffic has the same time λ, and an unused route is no faster.
* **Solution:** for parallel routes with increasing costs it is exact. Each route's flow is a function of
  λ:

  f(λ) = c · ((λ/T₀ − 1)/0.15)^(1/4) when λ > T₀ (else 0)

  We find the λ at which the flows add up to the demand, by bisection.
* **Numbers in the video:**
  * Route B starts carrying traffic at **2,277** drivers per hour. Route C would start at **4,986**.
  * At 4,000 per hour, A and B both take **29.6 min**, with flows of 2,568 and 1,432 per hour. C carries
    nobody and stays at its empty-road 37.7 min.
* **Why two, not three:** with fewer drivers only A is used; between 2,277 and 4,986 an hour exactly two
  routes are used, with equal times; above 4,986 all three are. So whether you see two or three equal ETAs
  depends on how busy the trip is, and a route that stays slower is a route nobody takes.
* **"Why do the times end up equal?"** The video starts from an unbalanced split of the same 4,000 drivers:
  2,121 on A (25.0 min) and 1,879 on B (36.0 min), with C unused. It then moves drivers linearly to the
  equilibrium split. That illustrates why a gap closes (arrivals slow the quicker route, departures speed up
  the other); it is not a dynamic model. The equilibrium is a Nash equilibrium of the routing game: no
  single driver can gain by switching. Nobody computes it; it is what remains once drivers stop finding
  quicker routes.

## 3a. The fine print (what "the same ETA" does and doesn't mean)

1. **Only routes in use.** Wardrop's condition says nothing equalizes the times of routes nobody uses. In
   the evening rush C stays at 37.7 min while A and B are at 29.6. On a heavier night (over 4,986 drivers
   an hour in this model) C fills too, and all three match.
2. **One trip, one moment.** The equilibrium is for one origin, destination and departure time. Other trips
   share the roads in reality (and are ignored here), and the balance shifts through the day.
3. **"Equal" is approximate.** Drivers don't switch for trivial savings, real ETAs are predictions with
   error, and apps round to the minute. In the learning run of section 4 the times end within a minute,
   not exactly equal.
4. **Equal is not best.** The equilibrium is what selfish choices produce, not the minimum total time
   (section 6).

"Your house isn't magically placed at some sweet spot in the traffic" is the same point from the other side:
the balance is a property of any trip with two or more busy routes, not of a particular address.

## 4. How drivers might find it (a behavioural sketch)

The day-to-day sequence is a model of learning, not a solver and not a model of any navigation app.

**The rule.** Each day:
1. A share of each route's drivers reconsider, using the times observed the previous day.
2. A reconsidering driver switches to the quickest route only if it saves more than a threshold.
3. Drivers move between routes, so no vehicles are created or lost and no flow goes negative.

**Day 1** (the video's "Monday"): 80% of drivers try the Midtown Tunnel, 15% the Throgs Neck Bridge and
5% the Battery Tunnel. That gives 42, 26 and 38 min. The video frames both runs as "what if it hadn't settled
yet?": a start that is deliberately far from equilibrium, to show how drivers could find it.

| Run | Who switches | Result |
|---|---|---|
| "The jam moves" | Half of all drivers jump to yesterday's quickest route, whatever the saving | Day 2 ("Tuesday"): half of everyone moves to the Throgs Neck Bridge, which jams (48 min). It never settles: the jam alternates between A and B, with B reaching 66 min. |
| "Settles" | 1 in 50 reconsider each day, and nobody switches to save a minute or less | A and B are within 1 minute of each other from day 13. After 40 days they are 29.12 and 29.13 min. C still carries 89 drivers an hour who have not yet reconsidered, which is why A and B sit a little below the exact 29.6. |

Rounded to the minute, the "settles" run's final times both show **29 min**: that is the opening and
closing display. They are not exactly equal; rounding makes them look it.

Real drivers differ far more than this. Many never switch. Some follow live directions. Times change
through the day. Trips with different origins and destinations share the roads.

## 5. What the app shows (selection, no traffic at all)

* **The 400 routes:** each one is forced through a different random junction on a major road within 18 km
  of the map centre (seeded). At 3 a.m. they take **21 to 67 min**, with a median of 31.5.
* **Our three on the histogram:** A, B and C (21.0, 26.3 and 37.7 min) are marked among the 400. They are
  the quickest route through each of three river crossings, which is the kind of choice an app makes: a
  few routes that are quick *and* genuinely different.
* **The point:** a selection effect. The app never lists the slow routes (up to 67 min here), or
  near-copies of a quicker route, so the routes you see are already the close calls. At rush hour traffic
  then evens out the ones people use.
* **Why "genuinely different" matters:** the three quickest of the 21 alternatives in section 1 take 21.0,
  22.6 and 23.4 min, but they are route A and two small variations of it (all through the Queens-Midtown
  Tunnel and along the LIE). The video therefore shows only A, B and C.
* **The filter is stylized:** "show a few quick, distinct routes". It is not any real app's algorithm,
  and real ETAs are predictions with error.

## 6. System optimum and the price of anarchy

* **Wardrop's second principle:** the system optimum minimizes total travel time. It equalizes each
  route's marginal social cost: the driver's own time plus the delay they add to everyone else,
  T₀ · (1 + 0.15 · 5 · (f/c)⁴) for BPR. It does not equalize the travel times.
* **Why selfish is not best:** a driver joining a busy road slows everyone behind them a little (the
  extra 0.15 · 4 · (f/c)⁴ · T₀ term). Nobody pays that cost when choosing, so an externality separates the
  two. This is what the video means by "nobody counts that cost".
* **In the evening rush:**

  | | A | B | C | Average |
  |---|---:|---:|---:|---:|
  | Flow (veh/h) | 2,083 | 1,358 | 559 | |
  | Trip time (min) | 24.7 | 29.0 | 38.0 | 28.0 |

  The controller sends 559 drivers an hour the long way round Brooklyn, a route nobody chooses for
  themselves. The average compares with 29.6 at equilibrium, so the price of anarchy (total time at user
  equilibrium ÷ total at system optimum) is **1.056**.
* **It peaks at 1.19**, at about 5,000 drivers per hour, just as route C would start to fill.
* **These are toy-model values.** Gaps like this depend entirely on the curves and capacities. For
  linear costs the price of anarchy can never exceed 4/3 (Roughgarden and Tardos, "How bad is selfish
  routing?", *J. ACM* 2002). For quartic BPR curves the bound is much higher. In the mythical-tunnel
  example below, total time with the tunnel is 33.6/26.5 = 1.27 times total time without it.

## 7. Braess's paradox on the map: the mythical tunnel (a thought experiment)

* **Source:** D. Braess, "Über ein Paradoxon aus der Verkehrsplanung", *Unternehmensforschung* 12,
  1968. English translation: Braess, Nagurney and Wakolbinger, *Transportation Science* 39, 2005.
* **The tunnel is imaginary.** There is no road like it. It would run about 15 km in a straight line, from
  Long Island City to the Bronx end of the Throgs Neck Bridge, so 3 minutes means over 300 km/h. It exists
  to show the paradox on familiar geometry.
* **What a bottleneck is (the lane-drop diagram):** a road narrowing from 4 lanes to 2. In light traffic
  fewer cars arrive than 2 lanes can pass, and nobody waits. In heavy traffic more arrive than it can pass,
  so a queue builds and the wait grows. The diagram is a schematic queue, not to scale (`video.py: Neck`).
  Cars follow the car ahead, and the merge lets one car through every 0.2 s, whichever has waited longest.
  Arrivals are 3 cars a second in light traffic and 9 in heavy. The lane counts are an illustration: in
  the model each bottleneck is a BPR link with a made-up capacity for this trip's drivers, which stands for
  what is left of a road's capacity after all other traffic. No real lane counts are claimed.
* **On the map**, each bottleneck's label shows the minutes it takes to get through, from its BPR time at
  the current flow. As drivers switch to the tunnel, the Queens-Midtown Tunnel's goes from 8 to 15 min and
  the Throgs Neck Bridge's from 3 to 10.
* **The network** (`scenario.py: myth_network`, settings in `cities.py`): routes A and B only, with C
  ignored. Each is cut into one bottleneck and stretches that are assumed never to slow down.
  * **Route A** (east, 21.0 min empty): the Queens-Midtown Tunnel, minutes 3.1–5.1 of the route, is a BPR
    bottleneck with a made-up capacity of 800 vehicles per hour. The rest is fixed.
  * **Route B** (north, 26.3 min empty): the Throgs Neck Bridge, minutes 20.7–23.3, is a BPR bottleneck
    with a made-up capacity of 1,000 vehicles per hour. The rest is fixed.
  * **The mythical tunnel** joins the Queens end of the Queens-Midtown Tunnel to the Bronx end of the
    Throgs Neck Bridge: 3 minutes, with no congestion. The route through it uses *both* bottlenecks.
  * **Demand:** 2,500 drivers an hour, made up.
* **Tunnel closed:** 1,641 drivers per hour take A and 859 take B, 26.5 min each.
* **Tunnel open:** at user equilibrium 1,615 take the tunnel route, 404 stay on A and 481 on B, and all three
  take **33.6 min**: 7.1 min (27%) more for everyone. A lone driver switching back to either old route
  would also take 33.6 min, so nobody gains by switching.
* **The first driver to try it** (with everyone else still on the old split) takes 19.4 min instead of
  26.5. That is why the switching starts.
* **At 3 a.m.** (no traffic) the tunnel route takes 13.7 min, against 21.0 on A and 26.3 on B.
* **It only happens at some demands.** With fewer than about 1,830 drivers an hour the tunnel helps
  (at 1,500 it saves 5.4 min). From 1,830 to 4,120 it makes everyone slower. Above about 4,120 nobody
  uses it: each bottleneck is then so slow that the tunnel route, which needs both, loses to the old
  routes. The video's chart is computed every 100 drivers per hour from 100 to 6,000.
* **Solver:** path equilibration in `model.py`, extended to BPR links and tested against the parallel-route
  solver.
* **The switching animation** moves drivers linearly from the closed equilibrium to the open one. It
  illustrates the move; it is not a dynamic model.
* **What it shows:** a new road *can* slow everyone down, when it sends traffic through bottlenecks it did
  not use before. It does not show that new roads usually do. Induced demand (new roads attracting new
  trips) is a different effect. `model.py` also still computes and tests Braess's original textbook
  network (4,000 drivers, 65 → 80 min).

## 8. Other material

* **The light-speed clip** is 15 s of this repository's own film, *All world roads at light speed*, from
  0:26. Its 4 min 40 s is that film's estimate for all vehicular roads (see `../ACCURACY.md`).
* **Music** is original, synthesized for this video by `music.py` on the instruments in
  `../flights/score.py`. There are no samples.
* **Fonts:** Inter (SIL OFL).
* **Thumbnail D** wraps the creator's Apple Maps screenshot, cropped and scaled, never edited (`thumbs.py`,
  settings in `media/traffic/maps_screenshot.json`).
* **Narration** is an AI voice: ElevenLabs' premade "Lily" voice (Eleven Multilingual v2), recorded line by line
  by `lily.py` from `media/traffic/VOICEOVER.md`. That script is generated by `voiceover.py` from the same scenario
  numbers (it refuses to write if they no longer match). A few clips were sped up by at most 12% to fit their
  slots. `voice.py` mixes each clip in on its cue, with the music 5 dB lower throughout and a further 10 dB lower
  under the voice.

## 9. Reproduce

```bash
cd traffic
python city_routes.py fetch && python city_routes.py build         # roads -> graph -> routes (~3 min); TRAFFIC_CITY=calgary for the other trip
python scenario.py                                                 # every number in the video
python -m pytest -q test_model.py                                  # 36 model tests
python assemble.py render                                          # sections (4 at a time) + score -> out/two_routes_one_eta.mp4
python thumbs.py                                                   # thumbnails A/B/C/D (D from maps_screenshot.png)
python voiceover.py                                                # the narration script -> media/traffic/VOICEOVER.md
```
