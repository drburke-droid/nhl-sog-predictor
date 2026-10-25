# Accuracy and sources: three routes, one ETA

The video asks why a map app so often offers three completely different routes with the same ETA. It
answers with two separate effects. The first is traffic: drivers crowd the quicker roads until no one gains
by switching (Wardrop's user equilibrium). The second is selection: the app lists only the quickest few
routes. The video then covers the system optimum, the price of anarchy and Braess's paradox.

**What is real, and what is a toy model:**
* **Real:** the road geometry, the three routes and their free-flow times.
* **Toy model:** everything about traffic. There are no Calgary traffic counts, no real app ETAs and no
  measured capacities anywhere in it.

Every number on screen comes from `scenario.py` (written to `media/traffic/video_numbers.json`), using the
tested functions in `model.py`.

## 1. The roads and the three routes (real data, idealized driving)

**Data.** Overture Maps Foundation transportation segments (release 2026-09-23.1, derived from
OpenStreetMap). They are fetched for a box around Calgary that includes the whole Stoney Trail ring road
(`calgary_routes.py fetch`), with 173,282 segments. The routable street network covers motorway down to
residential and living streets, with no service roads, tracks or paths: 264,311 nodes and 9,635 km of
road.

**Directions.** One-way rules are honoured. Overture's `access_restrictions` with `heading` give the
direction of travel. Segments closed to cars are dropped. Turn restrictions are *not* modelled.

**Speed.** Each road is timed at its speed limit.
* **Mapped limits:** OpenStreetMap's `maxspeed` where it is mapped (34% of all road length, but 55–83% of
  motorway, trunk, primary and secondary length).
* **Defaults elsewhere:**
  * motorway 100 km/h, trunk 80, primary 60, secondary and tertiary 50, ramps 60;
  * residential 40 km/h, Calgary's default for residential streets since 2021;
  * living streets 20.

**Free-flow time = length ÷ speed limit.** There is no traffic, no traffic-light or stop-sign delay and
no time for turns or acceleration: the time an ideal driver could make at 3 a.m. Real trips take longer.

**The trip.** From Rocky Ridge (51.1440° N, 114.2265° W, far northwest Calgary) to Seton (50.8810° N,
113.9535° W, near the South Health Campus).

**Finding alternatives.** We used the via-node method used by route planners:
1. Run a Dijkstra search forward from the origin and backward from the destination.
2. Every junction *v* then defines a candidate route: fastest to *v*, fastest from *v*.
3. Reject candidates that double back on themselves.
4. Reject candidates whose stretch around *v* is not itself close to optimal (a local-optimality test in
   the spirit of Abraham et al., "Alternative routes in road networks", 2013).
5. Keep the quickest 20 that share at most 80% of their length with each other.

The three routes in the video are the three of those within 15% of the fastest that share the least road:

| Route | Free-flow time | Length | Main roads |
|---|---:|---:|---|
| A, around the west | 33.0 min | 48.8 km | Stoney Trail NW, Tsuut'ina Trail, Stoney Trail SW |
| B, through the middle | 37.1 min | 45.4 km | Crowchild Trail NW/SW, Deerfoot Trail SE |
| C, up the east side | 37.8 min | 46.7 km | Deerfoot Trail NE, Barlow Trail SE, 52 Street SE |

They overlap by 4–11% of their length, mostly at the two ends. The video calls B "Crowchild & Deerfoot"
and C "Barlow & 52 St" for short.

## 2. The congestion curve (a standard simplification)

Each route's travel time follows the Bureau of Public Roads curve:

T(f) = T₀ · (1 + 0.15 · (f / c)⁴)

* **Variables:** *f* is the route's flow and *c* its capacity, both in vehicles per hour.
* **Source:** U.S. Bureau of Public Roads, *Traffic Assignment Manual*, 1964.
* **What it is:** a smooth planning curve. It is not a physical traffic model. It has no queues, no
  spillback, no stop-and-go waves and no gridlock. It keeps rising past capacity (f/c > 1) instead of
  breaking down.
* **Capacities: made up.** 2,000, 1,500 and 1,100 vehicles per hour for A, B and C. These are not
  measurements of Stoney Trail, Crowchild/Deerfoot or Barlow/52 St.
* **Demand: also made up.** Rush hour is 5,500 drivers per hour making this kind of trip.

**The routes are treated as independent** (Mode A of the lab). Real routes share roads with each other
and with every other trip in the city, which couples their times.

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
  * Route B starts carrying traffic at **1,914** drivers per hour and C at **2,860**.
  * At 5,500 per hour all three take **46.5 min**, with flows of 2,569, 1,707 and 1,224 per hour.

## 4. How drivers might find it (a behavioural sketch)

The day-to-day sequence is a model of learning, not a solver and not a model of any navigation app.

**The rule.** Each day:
1. A share of each route's drivers reconsider, using the times observed the previous day.
2. A reconsidering driver switches to the quickest route only if it saves more than a threshold.
3. Drivers move between routes, so no vehicles are created or lost and no flow goes negative.

**Day 1:** 60% of drivers try Stoney Trail, 20% each try the other two.

| Run | Who switches | Result |
|---|---|---|
| "The jam moves" | Half of all drivers jump to yesterday's quickest route, whatever the saving | Never settles. The BPR curve makes the overloaded route absurdly slow, so the chart is cut off at 120 min. |
| "Settles" | 1 in 20 reconsider each day, and nobody switches to save a minute or less | All three times are within 1 minute of each other from day 7. After 40 days they are 46.1, 46.7 and 46.5 min. |

The "settles" run's final times, rounded, are the **46, 47 and 47 min** shown in the opening and at the
end. They are close but not identical, because nobody bothers to switch for less than a minute.

Real drivers differ far more than this. Many never switch. Some follow live directions. Times change
through the day. Trips with different origins and destinations share the roads.

## 5. What the app shows (selection, no traffic at all)

* **The 400 routes:** each one is forced through a different random junction on a major road within
  22 km of downtown (seeded). At 3 a.m. they take **33 to 55 min**, with a median of 37.7.
* **The app's three:** the three quickest of the 20 alternatives in section 1, at **33.0, 33.8 and
  34.1 min**. That is within about a minute, with no traffic.
* **The point:** this is a pure selection effect. Picking the quickest few from many always picks routes
  close to each other.
* **The filter is stylized:** "show the fastest few distinct routes". It is not any real app's
  algorithm, and real ETAs are predictions with error.

## 6. System optimum and the price of anarchy

* **Wardrop's second principle:** the system optimum minimizes total travel time. It equalizes each
  route's marginal social cost: the driver's own time plus the delay they add to everyone else,
  T₀ · (1 + 0.15 · 5 · (f/c)⁴) for BPR. It does not equalize the travel times.
* **At rush hour:**

  | | A | B | C | Average |
  |---|---:|---:|---:|---:|
  | Flow (veh/h) | 2,461 | 1,759 | 1,280 | |
  | Trip time (min) | 44.3 | 47.6 | 48.2 | 46.3 |

  The average compares with 46.5 at equilibrium, so the price of anarchy (total time at user
  equilibrium ÷ total at system optimum) is **1.004**.
* **In lighter traffic** it peaks at **1.06**, around 1,950 drivers per hour.
* **These are toy-model values.** Gaps like this depend entirely on the curves and capacities. For
  linear costs the price of anarchy can never exceed 4/3 (Roughgarden and Tardos, "How bad is selfish
  routing?", *J. ACM* 2002). The Braess example below has 80/65 = 1.23.

## 7. Braess's paradox (exact, the textbook example)

* **Source:** D. Braess, "Über ein Paradoxon aus der Verkehrsplanung", *Unternehmensforschung* 12,
  1968. English translation: Braess, Nagurney and Wakolbinger, *Transportation Science* 39, 2005.
* **The network:** 4,000 drivers.
  * S→A and B→T take flow ÷ 100 minutes.
  * A→T and S→B take 45 minutes.
  * The connector A→B takes 0 minutes.
* **Connector closed:** 2,000 drivers per route, 65 min each.
* **Connector open:** at equilibrium all 4,000 take S-A-B-T, 80 min each. A lone driver switching to
  S-A-T or S-B-T would take 85 min, so nobody switches.
* **Solver:** path equilibration in `model.py`, tested.
* **The switching animation** moves drivers linearly from the closed equilibrium to the open one. It
  illustrates the move; it is not a dynamic model.
* **It only happens at some demands.** With 2,000 drivers the connector *helps* (55 → 40 min). At 3,000
  it changes nothing, and from 9,000 on nobody uses it. The paradox needs demand between 3,000 and 9,000
  (lab tab C). It shows that a new road *can* slow everyone down, not that new roads usually do. Induced
  demand (new roads attracting new trips) is a different effect.

## 8. Other material

* **The light-speed clip** is 15 s of this repository's own film, *All world roads at light speed*, from
  0:26. Its 4 min 40 s is that film's estimate for all vehicular roads (see `../ACCURACY.md`).
* **Music** is original, synthesized for this video by `music.py` on the instruments in
  `../flights/score.py`. There are no samples.
* **Fonts:** Inter (SIL OFL).

## 9. Reproduce

```bash
cd traffic
python calgary_routes.py fetch && python calgary_routes.py build   # roads -> graph -> routes (~2 min)
python scenario.py                                                 # every number in the video
python -m pytest -q test_model.py                                  # 35 model tests
python assemble.py render                                          # sections (4 at a time) + score -> out/three_routes_one_eta.mp4
python thumbs.py                                                   # thumbnails A/B/C
```
