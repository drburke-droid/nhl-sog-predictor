# When should you leave work? (`commute/`)

A narrated video in the style of `../traffic/`: for the 20 largest North American metro areas, how the time you
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

**The numbers must come from route-specific predictions; none are in yet.** A first pass with city-wide
averages (`model.py`, below) was too low: for Toronto, King and Bay to Richmond Hill leaving at 4:50 p.m. on a
Tuesday, it gave 61 minutes, and Google Maps shows 1 h to 2 h for that departure (checked 26 Oct 2026). The route
is the DVP and Highway 404, far more congested at 5 p.m. than Toronto's average road. City averages can't be
exact for any one commute, so its results are not used.

No public source gives route-specific predicted drive times without a key (searched 26 Oct 2026). The run
needs one of:

* **TomTom (`tomtom_routes.py`, recommended):** a free account at developer.tomtom.com (no credit card); the key
  is on the dashboard. The Routing API's free allowance is 20,000 requests a month, and the whole run is 6,600.
  Add the key in the environment's settings (Network secrets where offered, otherwise the environment variable
  `TOMTOM_API_KEY`), then start a new session.
* **Google (`fetch.py`):** a Google Cloud project with billing on, the Routes API enabled and a key, as
  `GOOGLE_MAPS_API_KEY`. The same 6,600 requests; check the free usage for traffic-aware routes in the console.

Then, in a new session, from `commute/`:

```
python tomtom_routes.py --check     # one request: Toronto, Tue 17 Nov, 4:50 p.m.; compare with Google's 1-2 h
python tomtom_routes.py             # the 6,600 requests, about 30 min, resumable
python analyse.py tomtom            # -> ../media/commute/curves_tomtom.json, and a table of the findings
python findings.py tomtom           # -> ../media/commute/findings_tomtom_{evening,morning}_{half_hour,drive}.png
```

(`fetch.py`, `analyse.py google` and `findings.py google` for Google.)

## Method

| Step | Script | Output |
|---|---|---|
| The 20 metros and their downtown office points | `metros.py` | |
| A random suburb for each: GeoNames towns of 25,000+ people, 18-35 km from the office, seeded per metro | `homes.py` | `../media/commute/homes.json` |
| The departures to ask about, both commutes | `plan.py` | |
| Predicted drive for each departure, on the quickest route for it | `tomtom_routes.py` (or `fetch.py`) | `data/` (cache), `../media/commute/drive_times_<router>.json` |
| Findings per metro and commute, with checks | `analyse.py` | `../media/commute/curves_<router>.json` |
| Charts | `findings.py` | `../media/commute/findings_<router>_*.png` |

* **Commutes:** evening, office to home, leaving every 5 minutes from 3:00 to 8:00 p.m.; morning, home to office,
  every 5 minutes from 6:00 to 10:00 a.m.; local time, on Tue-Thu 17-19 Nov 2026 (after the clocks change on
  1 Nov, clear of US Thanksgiving and of Mexico's Revolution Day holiday on Mon 16 Nov). 20 metros × (61 + 49)
  departures × 3 days = 6,600 requests.
* **TomTom:** Calculate Route, `routeType=fastest`, `traffic=true`, `departAt` the local departure,
  `computeTravelTimeFor=all`. The drive is `historicTrafficTravelTimeInSeconds`: TomTom's time-dependent historic
  traffic on that route for that weekday and time (weeks ahead, so no live traffic in it). The empty-roads time
  is `noTrafficTravelTimeInSeconds`. Each cached response keeps its summary, a thinned route line and a
  fingerprint of the full route, so route switches show.
* **Google:** computeRoutes, `TRAFFIC_AWARE_OPTIMAL`, `trafficModel: BEST_GUESS`, `departureTime`.
* **Findings (`analyse.py`), on the 5-minute departures as fetched, no interpolation:** the Tue-Thu average drive
  and its range over the three days; the worst departure and the rush window (within 5 minutes of the worst); the
  best and worst half hour (leave 30 minutes later: how much later do you arrive?); 5:00 vs 5:35 p.m. and 7:00 vs
  7:35 a.m.; the flattest 5-minute step; any first-in, first-out violation, on any day; and route switches.
* **Ranking:** Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona
  metropolitana; page as of 25 Sep 2026). #16 is the Inland Empire (office: downtown Riverside). #20 Santo
  Domingo has no TomTom Traffic Index data, so #21 Seattle takes its place (kept for the routed data too, so the
  list stays one source).
* **Suburbs:** GeoNames (CC BY 4.0). One GeoNames entry is excluded with a reason in `homes.py`.

## The city-average baseline (not used for findings)

`tomtom.py` (TomTom Traffic Index 2025 weekday-hour congestion, metro area and city centre), `routes.py` (OSRM
route on empty roads) and `model.py` (the route driven at the hourly average slowdown) need no key, and give
`../media/commute/curves.json`, evening only. Its shape is plausible (a worst departure around 4:30-5:00 p.m.,
a discount window after it), but its drive times are too low on commuter corridors (Toronto above), and its
hourly data smooths out sharp peaks. It stays as a comparison: how far a city's average is from its commutes.

## Next

- [ ] A TomTom (or Google) key in the environment, then the run above
- [ ] Check the routed numbers against Google Maps spot checks (Toronto 4:50 p.m.: 1 h to 2 h)
- [ ] Findings for both commutes, then the video: reuse `../traffic/look.py` and `../traffic/voice.py`;
      narration with `../traffic/lily.py`-style recording (needs `ELEVENLABS_API_KEY` in the environment)
