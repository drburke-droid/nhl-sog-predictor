# When should you leave work? (`commute/`)

A narrated video in the style of `../traffic/`: for the 20 largest North American metro areas, how the time you
get home depends on when you leave the office, and whether there is a "best time to leave", where staying longer
at work barely makes you later home.

## The question, and what to expect

Arrival home = departure + drive time. Leaving one minute later moves your arrival by 1 + (the change in drive
time), so:
* At the start of the rush, drive times climb: every minute you stay costs more than a minute.
* After the peak, drive times fall: every minute you stay costs *less* than a minute. That's the "discount
  window" the video is about (stay 25 more minutes, get home only 8 minutes later).
* Real traffic is first-in, first-out: on the same road, leaving later can't get you home earlier, and only
  gets you home at exactly the same time if nobody joins the queue behind you. So arrival time can flatten,
  but it shouldn't fall. Watch for exceptions in the data (route switches, prediction quirks) and report them
  as such.

The video's answer per city: the worst time to leave, the start of the discount window, and the "leave at X"
point (to define from the data: e.g. where the arrival curve's slope is lowest, or the latest departure that
gets you home within N minutes of the earliest).

## Data

| Step | Script | Output |
|---|---|---|
| The 20 metros and their downtown office points | `metros.py` | |
| A random suburb for each: GeoNames towns of 25,000+ people, 18–35 km from the office, seeded per metro | `homes.py` | `../media/commute/homes.json` |
| Drive times by departure, 3:30–7:30 p.m. every 5 min, Tue–Thu 17–19 Nov 2026 (2,940 requests) | `fetch.py` | `data/routes/` (cache), `../media/commute/travel_times.json` |

* **Ranking:** Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona
  metropolitana; page as of 25 Sep 2026). #16 is the Inland Empire (office: downtown Riverside) and #20 is Santo
  Domingo; both are kept as the ranking has them.
* **Drive times:** Google Maps Platform Routes API, `computeRoutes` with `TRAFFIC_AWARE_OPTIMAL` and
  `trafficModel: BEST_GUESS`: Google's prediction from its traffic history for each future departure, on the
  quickest route for that departure. These are predictions for typical weekdays, not measured trips.
* **Suburbs:** GeoNames (CC BY 4.0). One GeoNames entry is excluded with a reason in `homes.py`.

## Setup

1. In Google Cloud: a project with billing on, the **Routes API** enabled, and an API key restricted to it.
   Traffic-aware routes are a higher pricing tier than plain routes; check the free monthly usage for that tier
   in the console. 2,940 requests is the whole run.
2. Add the key to the environment as `GOOGLE_MAPS_API_KEY` (environment settings, not the chat), then start a
   new session.

## Status and next steps

- [x] Metro list, office points, suburbs (`homes.json`)
- [x] Request plan and fetcher (`python fetch.py --dry-run`)
- [ ] Fetch: `python fetch.py` (resumable)
- [ ] Analysis: arrival curves, discount windows, the "leave at" answer per city; check for FIFO violations
- [ ] Video: reuse `../traffic/look.py` and `../traffic/voice.py`; narration with `../traffic/lily.py`-style
      recording (needs `ELEVENLABS_API_KEY`)
