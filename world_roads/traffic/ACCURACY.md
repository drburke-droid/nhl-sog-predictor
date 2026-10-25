# Accuracy and sources: three routes, one ETA

The video asks why a map app so often offers three completely different routes with the same ETA. It
answers with two separate effects. The first is traffic: drivers crowd the quicker roads until no one gains
by switching (Wardrop's user equilibrium). The second is selection: the app lists only the quickest few
routes. The video then covers the fine print of "the same ETA", the system optimum, the price of anarchy,
and Braess's paradox, shown on the same map with a mythical tunnel.

The trip is **JFK Airport to Times Square**.

**What is real, and what is a toy model:**
* **Real:** the road geometry, the three routes and their free-flow times.
* **Toy model:** everything about traffic. There are no New York traffic counts, no real app ETAs and no
  measured capacities anywhere in it.

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
  47–63% of motorway, primary and secondary length, and 64–74% of the three routes.
* **Defaults elsewhere:**
  * New York's 25 mph default for city streets;
  * 50 mph on expressways, 40 mph on other limited-access roads, 30 mph on ramps.

**Free-flow time = length ÷ speed limit.** There is no traffic, no traffic-light delay and no time for
turns. The tolls on the RFK Bridge and both tunnels are ignored too, and so is Manhattan's congestion
charge. Real drivers weigh tolls as well as time, one more reason real ETAs differ. This gives the time
an ideal driver could make at 3 a.m.; real trips take longer.

**The trip.**
* **Start:** the road network nearest JFK's central terminal area (40.6447° N, 73.7812° W).
* **Finish:** the nearest street junction to Times Square (40.7570° N, 73.9866° W).

**Finding alternatives.** We used the via-node method used by route planners:
1. Run a Dijkstra search forward from the start and backward from the finish.
2. Every junction *v* then defines a candidate route: fastest to *v*, fastest from *v*.
3. Reject candidates that double back on themselves.
4. Reject candidates whose stretch around *v* is not itself close to optimal (in the spirit of Abraham et
   al., "Alternative routes in road networks", 2013).
5. Keep the quickest 20 that share at most 80% of their road with each other.

All of these leave JFK on the Van Wyck Expressway, and most also share the Long Island Expressway. So
they are variations of one route rather than completely different ones.

**The three in the video** are therefore the fastest route through each of three river crossings into
Manhattan, one in the north, one in the middle and one in the south. Only nodes inside the bridge or
tunnel count, so each route has to actually cross it.

| Route | Free-flow time | Length | Roads (km on each) |
|---|---:|---:|---|
| A, Midtown Tunnel, straight through Queens | 26.2 min | 29.0 km | Van Wyck Expressway 13.6, Long Island Expressway 7.5, Queens-Midtown Tunnel 2.1 |
| B, RFK Bridge, around the north | 33.1 min | 36.0 km | Van Wyck 9.8, Grand Central Parkway 11.7, Robert F. Kennedy Bridge 3.1, FDR Drive 6.0 |
| C, Battery Tunnel, around Brooklyn | 38.0 min | 34.7 km | JFK Expressway 2.1, Belt Parkway 3.0, Linden Boulevard 9.4, Prospect and Gowanus Expressways 2.8, Brooklyn-Battery Tunnel 3.0, West Street 3.2, 8th Avenue 2.8 |

**How much they share:**
* A and B share 36%, almost all of it the Van Wyck out of JFK.
* C shares about 1% with either.
* The Queensboro Bridge (27.7 min) and Williamsburg Bridge (32.1 min) routes were left out because they
  share 70–76% of their road with route A.

The Brooklyn-Battery Tunnel is officially the Hugh L. Carey Tunnel; the video uses the older name, which
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
* **Demand: also made up.** Rush hour is 6,500 drivers per hour making this kind of trip.

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
  * Route B starts carrying traffic at **2,303** drivers per hour and C at **4,121**.
  * At 6,500 per hour all three take **50.0 min**, with flows of 3,137, 2,037 and 1,326 per hour.
* **"Why do the times end up equal?"** The video starts from an unbalanced split of the same 6,500 drivers:
  2,957 on A (45.0 min) and 2,132 and 1,411 on B and C (53.4 min each). It then moves drivers linearly to
  the equilibrium split. That illustrates why a gap closes (arrivals slow the quicker route, departures
  speed up the others); it is not a dynamic model. The equilibrium is a Nash equilibrium of the routing
  game: no single driver can gain by switching. Nobody computes it; it is what remains once drivers stop
  finding quicker routes.

## 3a. The fine print (what "the same ETA" does and doesn't mean)

1. **Only routes in use.** Wardrop's condition says nothing equalizes the times of routes nobody uses. At
   3 a.m. (300 drivers an hour in the toy model) everyone takes A, and B and C stay slower: 26, 33 and
   38 min.
2. **One trip, one moment.** The equilibrium is for one origin, destination and departure time. Other trips
   share the roads in reality (and are ignored here), and the balance shifts through the day.
3. **"Equal" is approximate.** Drivers don't switch for trivial savings, real ETAs are predictions with
   error, and apps round to the minute. In the learning run of section 4 the times end within a minute,
   not exactly equal.
4. **Equal is not best.** The equilibrium is what selfish choices produce, not the minimum total time
   (section 6).

## 4. How drivers might find it (a behavioural sketch)

The day-to-day sequence is a model of learning, not a solver and not a model of any navigation app.

**The rule.** Each day:
1. A share of each route's drivers reconsider, using the times observed the previous day.
2. A reconsidering driver switches to the quickest route only if it saves more than a threshold.
3. Drivers move between routes, so no vehicles are created or lost and no flow goes negative.

**Day 1:** 60% of drivers try the Midtown Tunnel, 20% each try the other two.

| Run | Who switches | Result |
|---|---|---|
| "The jam moves" | Half of all drivers jump to yesterday's quickest route, whatever the saving | Never settles. The BPR curve makes the overloaded route absurdly slow, so the chart is cut off at 120 min and anything slower shows as "2 h+". |
| "Settles" | 1 in 50 reconsider each day, and nobody switches to save a minute or less | All three times are within 1 minute of each other from day 13. After 40 days they are 49.6, 50.2 and 50.4 min. |

Rounded to the minute, the "settles" run's final times all show **50 min**: that is the opening and closing
display. They are not exactly equal; rounding makes them look it.

Real drivers differ far more than this. Many never switch. Some follow live directions. Times change
through the day. Trips with different origins and destinations share the roads.

## 5. What the app shows (selection, no traffic at all)

* **The 400 routes:** each one is forced through a different random junction on a major road within 18 km
  of the map centre (seeded). At 3 a.m. they take **26 to 61 min**, with a median of 37.6.
* **The app's three:** the three quickest of the 22 alternatives in section 1, at **26.2, 27.7 and
  28.0 min**. That is within 2 minutes, with no traffic.
* **The point:** this is a pure selection effect. Picking the quickest few from many always picks routes
  close to each other.
* **The filter is stylized:** "show the fastest few distinct routes". It is not any real app's
  algorithm, and real ETAs are predictions with error.

## 6. System optimum and the price of anarchy

* **Wardrop's second principle:** the system optimum minimizes total travel time. It equalizes each
  route's marginal social cost: the driver's own time plus the delay they add to everyone else,
  T₀ · (1 + 0.15 · 5 · (f/c)⁴) for BPR. It does not equalize the travel times.
* **Why selfish is not best:** a driver joining a busy road slows everyone behind them a little (the
  extra 0.15 · 4 · (f/c)⁴ · T₀ term). Nobody pays that cost when choosing, so an externality separates the
  two. This is what the video means by "nobody counts that cost when choosing a route".
* **At rush hour:**

  | | A | B | C | Average |
  |---|---:|---:|---:|---:|
  | Flow (veh/h) | 2,982 | 2,071 | 1,447 | |
  | Trip time (min) | 45.6 | 51.2 | 55.1 | 49.5 |

  The average compares with 50.0 at equilibrium, so the price of anarchy (total time at user
  equilibrium ÷ total at system optimum) is **1.010**.
* **In lighter traffic** it peaks at **1.12**, around 2,300 drivers per hour, just as route B starts to
  fill.
* **These are toy-model values.** Gaps like this depend entirely on the curves and capacities. For
  linear costs the price of anarchy can never exceed 4/3 (Roughgarden and Tardos, "How bad is selfish
  routing?", *J. ACM* 2002). For quartic BPR curves the bound is much higher. In the mythical-tunnel
  example below, total time with the tunnel is 51.9/38.2 = 1.36 times total time without it.

## 7. Braess's paradox on the map: the mythical tunnel (a thought experiment)

* **Source:** D. Braess, "Über ein Paradoxon aus der Verkehrsplanung", *Unternehmensforschung* 12,
  1968. English translation: Braess, Nagurney and Wakolbinger, *Transportation Science* 39, 2005.
* **The tunnel is imaginary.** There is no road like it. It would run about 16 km in a straight line, so
  3 minutes means over 300 km/h. It exists to show the paradox on familiar geometry.
* **The network** (`scenario.py: myth_network`, settings in `cities.py`): routes A and C only, with B
  ignored. Each is cut into one bottleneck and stretches that are assumed never to slow down.
  * **Route C** (around Brooklyn, 38.0 min empty): the JFK Expressway, minutes 1.1–3.0 of the route, is a
    BPR bottleneck with a made-up capacity of 1,000 vehicles per hour. The rest is fixed.
  * **Route A** (through Queens, 26.2 min empty): the Queens-Midtown Tunnel, minutes 19.8–22.9, is a BPR
    bottleneck with a made-up capacity of 800 vehicles per hour. The rest is fixed.
  * **The mythical tunnel** joins the end of the JFK Expressway to the Queens-Midtown Tunnel's entrance:
    3 minutes, with no congestion. The route through it uses *both* bottlenecks.
  * **Demand:** 2,750 drivers an hour, made up.
* **Tunnel closed:** 942 drivers per hour take C and 1,808 take A, 38.2 min each.
* **Tunnel open:** at user equilibrium 2,107 take the tunnel route, 563 stay on C and 80 on A, and all three
  take **51.9 min**: 13.7 min (36%) more for everyone. A lone driver switching back to either old route
  would also take 51.9 min, so nobody gains by switching.
* **The first driver to try it** (with everyone else still on the old split) takes 24.5 min instead of
  38.2. That is why the switching starts.
* **At 3 a.m.** (no traffic) the tunnel route takes 12.3 min, against 26.2 on A and 38.0 on C.
* **It only happens at some demands.** With fewer than about 2,070 drivers an hour the tunnel helps
  (at 1,500 it saves 12.5 min). From 2,070 to 4,860 it makes everyone slower. Above about 4,860 nobody
  uses it: each bottleneck is then so slow that the tunnel route, which needs both, loses to the old routes. The video's chart
  is computed every 100 drivers per hour from 100 to 6,000.
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

## 9. Reproduce

```bash
cd traffic
python city_routes.py fetch && python city_routes.py build         # roads -> graph -> routes (~3 min); TRAFFIC_CITY=calgary for the other trip
python scenario.py                                                 # every number in the video
python -m pytest -q test_model.py                                  # 36 model tests
python assemble.py render                                          # sections (4 at a time) + score -> out/three_routes_one_eta.mp4
python thumbs.py                                                   # thumbnails A/B/C
```
