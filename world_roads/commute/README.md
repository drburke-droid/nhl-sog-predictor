# When should you leave work? (`commute/`)

A narrated video in the style of `../traffic/`: for the 20 largest North American metro areas, how the time you
get home depends on when you leave the office, and whether there is a "best time to leave", where staying longer
at work barely makes you later home.

## The question, and what to expect

Arrival home = departure + drive time. Leaving one minute later moves your arrival by 1 + (the change in drive
time), so:
* At the start of the rush, drive times climb: every minute you stay costs more than a minute.
* After the peak, drive times fall: every minute you stay costs *less* than a minute. That's the "discount
  window".
* Traffic is first-in, first-out: on the same road, leaving later can't get you home earlier, and only gets you
  home at the same time if nobody joins the queue behind you. So arrival time can flatten, but it can't stop or
  fall.

## What the data says

`python model.py`, charts with `python findings.py` (`../media/commute/findings_*.png`):

* **There is no magic minute.** In every metro, staying 30 minutes longer gets you home at least 24.7 minutes
  later. The best any city does is Seattle: leave at 6:04 instead of 5:34 and you're home 24.7 minutes later,
  5.3 minutes less driving. Every metro's best half hour costs 24.7-27.9 minutes; in the US and Canada it starts
  between about 5:30 and 6:10 p.m.
* **The worst time to leave is earlier than you'd think:** 4:31-4:59 p.m. in the US and Canadian metros (you
  drive into the 5 p.m. peak), 6:05-6:17 p.m. in Mexico City, Monterrey and Guadalajara.
* **5:00 vs 5:35:** leaving 35 minutes later gets you home 30-33.4 minutes later in the US and Canada, and 38.8-39.3
  minutes later in the three Mexican metros, where 5-6 p.m. is still the build-up.
* **Rush hour is wide, not a spike:** the drive is within 5 minutes of its worst from about 3 to 6 p.m. in most
  metros. The flattest point, where time home rises slowest per minute you stay, is 0.82-0.93 minutes per minute
  (49-56 seconds) at about 5:45-6:30 p.m. (7:30-7:45 in Mexico).

The model can only see city-wide hourly averages, so it smooths out the sharp peaks a single freeway can have.
A specific commute can do better than these numbers, but no commute can get to zero (first-in, first-out).

## Method

| Step | Script | Output |
|---|---|---|
| The 20 metros and their downtown office points | `metros.py` | |
| A random suburb for each: GeoNames towns of 25,000+ people, 18-35 km from the office, seeded per metro | `homes.py` | `../media/commute/homes.json` |
| Hour-by-hour congestion, metro area and city centre, every weekday hour | `tomtom.py` | `../media/commute/tomtom_weekhour.json` |
| The route on empty roads, with every stretch's free-flow time | `routes.py` | `../media/commute/routes.json` |
| The drive and time home for every departure, 3-8 p.m. every minute | `model.py` | `../media/commute/curves.json` |
| Charts | `findings.py` | `../media/commute/findings_*.png` |

* **Ranking:** Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona
  metropolitana; page as of 25 Sep 2026). #16 is the Inland Empire (office: downtown Riverside). #20 Santo
  Domingo has no TomTom data, so #21 Seattle takes its place.
* **Congestion:** TomTom Traffic Index 2025, the public city pages (tomtom.com/traffic-index). For each
  metro, 168 weekday-hour values (local time) for the metro area and for the city centre: average speed,
  free-flow speed and congestion level c, the extra travel time over free flow. The model uses Tuesday-Thursday.
* **Routes:** OSRM's public demo server (router.project-osrm.org, car profile) on OpenStreetMap data
  (© OpenStreetMap contributors, ODbL): the quickest route on empty roads, 22-40 km and 25-41 minutes here.
* **Suburbs:** GeoNames (CC BY 4.0). One GeoNames entry is excluded with a reason in `homes.py`.
* **Model:** the car moves along its route at 1 / (1 + c / 100) of free-flow speed, for the hour it's in, using
  the city-centre profile within 5 km of the office and the metro profile beyond. Within the day the extra-time
  factor is a smooth curve whose average over each hour is exactly TomTom's figure (a periodic cubic spline
  through its running total). Every departure is integrated to the end of the route in 2 s steps, so cars keep
  their order: first-in, first-out holds by construction.
* **Checks:** TomTom's hours are local (every metro's morning peak is 7-8 a.m.); its c matches
  fv / v - 1 within rounding. The results barely change with a 3 km or 10 km city centre.

## Limits

* City-wide hourly averages, not your road: real corridors have sharper peaks, so a real route can show a
  bigger discount than these. Hourly data also puts each city's flattest point within about ±15 minutes.
* Typical days: averages over 2025. Any given evening (rain, a crash, a game) can be very different.
* One route on empty roads. A navigation app would switch to a different route in traffic, which only makes
  the drive shorter than the model's at the worst times.

## Optional: Google's predictions

`fetch.py` asks Google's Routes API for traffic-aware predictions on each route, every 5 minutes, 3:30-7:30 p.m.
on three weekdays (2,940 requests). It needs `GOOGLE_MAPS_API_KEY` in the environment (a Google Cloud project
with billing on and the Routes API enabled). It's not used now; with a key it would replace city averages with
route-specific predictions.

## Status and next steps

- [x] Metro list, office points, suburbs (`homes.json`)
- [x] TomTom hourly profiles, OSRM routes, the model and its findings
- [ ] Video: reuse `../traffic/look.py` and `../traffic/voice.py`; narration with `../traffic/lily.py`-style
      recording (needs `ELEVENLABS_API_KEY`)
