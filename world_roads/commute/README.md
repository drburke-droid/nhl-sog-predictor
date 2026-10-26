# When should you leave work? (`commute/`)

A narrated video in the style of `../traffic/`: for the 20 largest North American metro areas and Vancouver, Calgary and Edmonton, how the time you
get home depends on when you leave the office, and whether there is a "best time to leave", where staying longer
at work barely makes you later home; and the same for the morning commute.

## The question, and what to expect

Arrival = departure + drive time. Leaving one minute later moves your arrival by 1 + (the change in drive time),
so:
* At the start of the rush, drive times climb: every minute you wait costs more than a minute.
* After the peak, drive times fall: every minute you wait costs *less* than a minute. That's the "discount
  window".
* Traffic is first-in, first-out: on the same road, leaving later can't get you there earlier, and only gets you
  there at the same time if nobody joins the queue behind you. So arrival time can flatten, but it can't stop or
  fall. Watch for exceptions in the data (a route switch, a prediction quirk) and report them as such.

## Status

**The routed numbers are in:** TomTom, fetched 26 Oct 2026, all 7,590 requests with historic traffic, in
`../media/commute/drive_times_tomtom.json`, `curves_tomtom.json` and `findings_tomtom_*.png`. The spot check:
Toronto, King and Bay to Richmond Hill leaving at 4:50 p.m. on Tue 17 Nov, 75 minutes (34 on empty roads), inside
the 1 h to 2 h Google Maps shows. No metro has a "magic" departure and none breaks first-in, first-out; the
closest is New York's evening (leave at 5:55 instead of 5:25, home 20.5 minutes later).

A first pass with city-wide averages (`model.py`, below) was too low: for that Toronto departure it gave 61
minutes. The route is the DVP and Highway 404, far more congested at 5 p.m. than Toronto's average road. City
averages can't be exact for any one commute, so its results are not used.

To run it again: no public source gives route-specific predicted drive times without a key (searched 26 Oct
2026), so the run needs one of:

* **TomTom (`tomtom_routes.py`, recommended):** a free account at developer.tomtom.com (no credit card); the key
  is on the dashboard. The Routing API's free allowance is 20,000 requests a month, and the whole run is 7,590.
  Add the key in the environment's settings (Network secrets where offered, otherwise the environment variable
  `TOMTOM_API_KEY`; `API_KEY` also works), then start a new session.
* **Google (`fetch.py`):** a Google Cloud project with billing on, the Routes API enabled and a key, as
  `GOOGLE_MAPS_API_KEY`. The same 7,590 requests; check the free usage for traffic-aware routes in the console.

Then, in a new session, from `commute/`:

```
python tomtom_routes.py --check     # one request: Toronto, Tue 17 Nov, 4:50 p.m.; compare with Google's 1-2 h
python tomtom_routes.py             # the 7,590 requests, about 32 min, resumable
python analyse.py tomtom            # -> ../media/commute/curves_tomtom.json, and a table of the findings
python findings.py tomtom           # -> ../media/commute/findings_tomtom_{evening,morning}_{half_hour,drive}.png
```

(`fetch.py`, `analyse.py google` and `findings.py google` for Google.)

## Method

| Step | Script | Output |
|---|---|---|
| The 23 metros and their downtown office points | `metros.py` | |
| A random suburb for each: GeoNames towns of 25,000+ people, 18-35 km from the office, seeded per metro | `homes.py` | `../media/commute/homes.json` |
| Up to three more, one per other compass direction: towns of 10,000+, 15-40 km, no ferry or detour round water (OSRM) | `homes.py --more` | `../media/commute/homes_more.json` |
| The departures to ask about, both commutes | `plan.py` | |
| Predicted drive for each departure, on the quickest route for it | `tomtom_routes.py` (or `fetch.py`) | `data/` (cache), `../media/commute/drive_times_<router>.json` |
| The same for the extra suburbs: Wed 18 Nov only, every 10 minutes (3,304 requests) | `tomtom_routes.py --more` | `../media/commute/drive_times_tomtom_more.json` |
| Findings per metro and commute, with checks | `analyse.py` | `../media/commute/curves_<router>.json` |
| Charts | `findings.py` | `../media/commute/findings_<router>_*.png` |
| When rush hour is in each metro, and why it differs (with the Traffic Index) | `rush.py` | `../media/commute/rush.json`, `rush_*.png` |

* **Commutes:** evening, office to home, leaving every 5 minutes from 3:00 to 8:00 p.m.; morning, home to office,
  every 5 minutes from 6:00 to 10:00 a.m.; local time, on Tue-Thu 17-19 Nov 2026 (after the clocks change on
  1 Nov (BC and Alberta keep daylight time from then on), clear of US Thanksgiving and of Mexico's Revolution Day holiday on Mon 16 Nov). 23 metros × (61 + 49)
  departures × 3 days = 7,590 requests.
* **TomTom:** Calculate Route, `routeType=fastest`, `traffic=true`, `departAt` the local departure,
  `computeTravelTimeFor=all`. The drive is `historicTrafficTravelTimeInSeconds`: TomTom's time-dependent historic
  traffic on that route for that weekday and time (weeks ahead, so no live traffic in it). The empty-roads time
  is `noTrafficTravelTimeInSeconds`. Each cached response keeps its summary, a thinned route line and a
  fingerprint of the full route, so route switches show. TomTom's time zones predate tzdata 2026e and still turn
  BC's and Alberta's clocks back on 1 Nov 2026, so for Vancouver, Calgary and Edmonton `departAt` is the local
  clock time at TomTom's offset (UTC-8, UTC-7), and every response's departure must match the clock time asked
  for (the first fetch, at the true instant, came back an hour early and was discarded).
* **Google:** computeRoutes, `TRAFFIC_AWARE_OPTIMAL`, `trafficModel: BEST_GUESS`, `departureTime`.
* **Findings (`analyse.py`), on the 5-minute departures as fetched, no interpolation:** the Tue-Thu average drive
  and its range over the three days; the worst departure and the rush window (within 5 minutes of the worst); the
  best and worst half hour (leave 30 minutes later: how much later do you arrive?); 5:00 vs 5:35 p.m. and 7:00 vs
  7:35 a.m.; the flattest 5-minute step; any first-in, first-out violation, on any day; and route switches.
* **Ranking:** Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona
  metropolitana; page as of 25 Sep 2026). #16 is the Inland Empire (office: downtown Riverside). #20 Santo
  Domingo has no TomTom Traffic Index data, so #21 Seattle takes its place (kept for the routed data too, so the
  list stays one source). Vancouver (#28), Calgary and Edmonton are added on request; Calgary and Edmonton are
  not on the list (it stops at #50, Havana, 2.14 million), so their 101 and 102 in `metros.py` are ids, not ranks.
  Their offices are OpenStreetMap points (Burrard Station; Bankers Hall, Stephen Avenue; Churchill Square). The
  city-average baseline below (`routes.py`, `model.py`) has the first 20 only; the Traffic Index file has all 23.
* **Suburbs:** GeoNames (CC BY 4.0). One GeoNames entry is excluded with a reason in `homes.py`.

## The city-average baseline (not used for findings)

`tomtom.py` (TomTom Traffic Index 2025 weekday-hour congestion, metro area and city centre), `routes.py` (OSRM
route on empty roads) and `model.py` (the route driven at the hourly average slowdown) need no key, and give
`../media/commute/curves.json`, evening only. Its shape is plausible (a worst departure around 4:30-5:00 p.m.,
a discount window after it), but its drive times are too low on commuter corridors (Toronto above), and its
hourly data smooths out sharp peaks. It stays as a comparison: how far a city's average is from its commutes.

## Rush hour (`rush.py`)

City-wide from the Traffic Index (2025, metro area), and on each routed commute:
* **US and Canada run on one clock:** the Tue-Thu evening peak is between 4:36 (Phoenix) and 5:21 p.m. (Bay Area)
  in all 20; the morning between 7:46 and 8:45. **Mexico's evenings are late:** 6:30-6:51 p.m. in Monterrey,
  Guadalajara and Mexico City, with mornings at the same time as everyone else's, so 10 h between the two peaks
  against 8 h 20 min-9 h 20 min. All three Mexican metros have a third peak at 2 p.m. (the time of the comida).
* **Not the sun:** the clock time of solar noon (2025 average) does not predict the peaks in the US and Canada
  (r = +0.12 evening, +0.23 morning); Mexico City and New York share a solar noon (12:37, 12:35) and their evening
  peaks are 1 h 43 min apart.
* **The evening is worse:** higher than the morning in all 23, and at least as long in 22 (Houston the exception).
* **The week:** Friday's evening peak comes 44 min earlier on average (all 20 US and Canadian metros; Boston 82
  min); Monday's morning peak is lower than Tue-Thu's in all 23 (16% on average). Mexico City's worst hour of the
  week is Friday 3-4 p.m. (+104%).
* **Trip length:** the worst time to leave comes earlier the longer the drive: over 71 US and Canadian routes
  (up to four per metro, Wed 18 Nov), 7 min (evening) and 5 min (morning) per 10 min of driving (r = -0.65,
  -0.68), and within each metro 6 and 7 min (r = -0.56, -0.73). New York's 100-min evening drive is worst leaving
  at 4:05, Detroit's 25-33-min ones at 4:46-4:53, while both cities peak around 5 p.m.
* **Rush hour by route:** at the worst departure the middle of the trip is close to the city-wide peak (median of
  metros 6 min before it in the evening, 14 in the morning), so the worst time to leave is about half an hour
  before the peak (29 min evening, 39 morning). Routes in one metro can differ by 40 minutes (Washington 4:28-5:11
  p.m., Montreal 4:08-4:45); Calgary's four agree within 15 (4:34-4:49 p.m., 7:38-7:50 a.m.). Vancouver's two
  (south and east, across the Fraser; north and west are water) are near their worst from about 3:10 to 4:30 p.m.
  and 7:10 to 8:00 a.m., some 50 minutes before its city-wide peaks.
* **TomTom's historic traffic is one profile per weekday:** the same routes on Wed 15 Sep 2027 (daylight time, when
  TomTom's time zones and the real ones agree) give exactly the drives of Wed 18 Nov 2026 as fetched, for Vancouver,
  Calgary and Seattle. So the drives are a typical weekday, not November's, and the BC and Alberta offset is right.
* **The costliest half hour:** leaving home in Mexico City at 6:50 instead of 6:20 a.m. gets you in 50 min later.

## Rush-hour map (`rush_map.py`)

`../media/commute/rush_map.mp4` (1920x1080, 30 fps, 80 s): the light-speed film's look (`../render.py`, `../na_wide.py`)
on a map of 45 metros: the 23 routed ones and 22 more for the map only (`metros.MAP_ONLY`: the rest of the same
Wikipedia list's top 50 that the Traffic Index covers and the map shows; not San José (Costa Rica), Guatemala City,
Port-au-Prince, San Salvador, Havana or Toluca, which have no Traffic Index page). `tomtom.py` fetches all 45. The clock runs from 5 a.m. to midnight at 4 s an hour, the same local time in every metro,
as if North America had one time zone, and each metro's highways glow by its city-wide congestion at that local time:
cyan while the roads are free, gold as they fill, white-hot at the worst (TomTom Traffic Index, Tue-Thu 2025
averages, metro area; hourly values joined by a monotone cubic). A board shows the five worst metros of the moment and
a line the average of the 40 US and Canadian metros through the day (the five Mexican metros are left out of it:
their evening peaks run about an hour and a half later). Base map: Natural Earth 10 m land, lakes, borders and roads
(`../data/ne`, fetched as `../run_all.sh` does, plus the roads and border layers from the same repository) in a
Lambert conformal conic projection. `python rush_map.py still 8.5 17.25` renders stills
(`rush_map_<hour>.png`).

## Traffic shapes (`shapes.py`)

Which metros have the same weekday traffic shape, apart from how bad it gets: each of the 45 metros' Tue-Thu
congestion by hour (Traffic Index, 2025), 4 a.m. to midnight, scaled to its own peak, and grouped with Ward's method.
Output: `../media/commute/shapes.json`, `shapes_buckets.png` (each group's shape and members), `shapes_axes.png` (the
ways shapes differ) and `shapes_map.png` (the map, metros by group).
* **Two clear groups:** the four big Mexican metros (Mexico City, Monterrey, Guadalajara, Puebla) and everyone else;
  Tijuana's day is US-shaped. Silhouette 0.53 (0.2-0.3 for any finer split); the same split with z-scored shapes,
  with average linkage, and on Monday, Tuesday, Wednesday or Thursday alone (not Friday, when Mexico City's worst
  hour moves to 3 p.m.).
* **Five shapes:** the split that each single day repeats best (86 % of metros, against 76-84 % for three, four or six
  groups); a metro stays in its group if at least two of the three days agree (41 of 45):
  * *Two sharp peaks* (quiet middays, weaker mornings; Texas and the Sun Belt): Dallas-Fort Worth, Houston, Austin, San
    Antonio, Atlanta, Charlotte, Tampa, Minneapolis-St. Paul, Indianapolis, Columbus;
  * *Early finishers* (evening peak about 4:54 p.m., the earliest): Calgary, Edmonton, Portland, St. Louis, Kansas
    City, Cincinnati;
  * *Early starters* (the strongest 6-8 a.m. traffic and mornings): Los Angeles, Inland Empire, Phoenix, Tijuana,
    Seattle, Chicago, Boston, Montreal, Washington, Baltimore, Miami;
  * *Busy all day* (middays at 62 % of the peak, busy late evenings): New York, Philadelphia, Toronto, Detroit,
    Cleveland, Pittsburgh, Vancouver, Bay Area, Las Vegas, Orlando;
  * *Late and long*: the four Mexican metros; *in between*: San Diego, Denver, Nashville, Sacramento.
* **How US-shaped days differ:** the first two principal components (52 % and 17 %) are midday traffic as a share of
  the peak (r = 0.96) and early-morning minus late-evening traffic (r = -0.96); how busy the middle of the day stays is
  unrelated to how congested a metro is. The early finishers stand apart on timing.
* **Busier metros peak later:** the evening peak comes about 3 minutes later per 10 points of peak congestion
  (r = 0.41, 41 US-shaped metros), as congestion spreads the peak.

## Next

- [x] A TomTom key in the environment, then the run above (26 Oct 2026)
- [x] Toronto 4:50 p.m. against Google Maps: 75 min, inside its 1 h to 2 h
- [x] Up to four suburbs per metro (82 routes), 26 Oct 2026
- [ ] More Google Maps spot checks, the three added Canadian metros among them (Vancouver's early peaks above all)
- [ ] Findings for both commutes, then the video: reuse `../traffic/look.py` and `../traffic/voice.py`;
      narration with `../traffic/lily.py`-style recording (needs `ELEVENLABS_API_KEY` in the environment)
