"""What to ask a traffic-aware router: the departures for both commutes, shared by tomtom_routes.py and fetch.py

* evening: office -> home, leaving every 5 minutes from 3:00 to 8:00 p.m. local time
* morning: home -> office, leaving every 5 minutes from 6:00 to 10:00 a.m. local time

on three ordinary weekdays ahead (Tue-Thu, 17-19 Nov 2026): after the clocks change on 1 Nov (BC and Alberta
keep daylight time from then on, tzdata 2026e), clear of US Thanksgiving and of Mexico's Revolution Day holiday
(Mon 16 Nov).

more_jobs: the extra suburbs of homes_more.json (homes.py --more), both commutes on Wednesday 18 Nov only, every
10 minutes; the three days differ by a few minutes at most on the main suburbs.
"""
import os, json
from datetime import datetime, date
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
DAYS = [date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
WINDOWS = {"evening": ((15, 0), (20, 0)), "morning": ((6, 0), (10, 0))}
STEP = 5                                                # minutes between departures
MORE_DAY, MORE_STEP = date(2026, 11, 18), 10


def departures(direction, step=STEP):
    (h0, m0), (h1, m1) = WINDOWS[direction]
    t = h0 * 60 + m0
    while t <= h1 * 60 + m1:
        yield t // 60, t % 60
        t += step


def job(m, place, direction, day, hh, mm, **extra):
    tz = ZoneInfo(m["tz"])
    office, home = (m["office"]["lat"], m["office"]["lon"]), (place["lat"], place["lon"])
    o, d = (office, home) if direction == "evening" else (home, office)
    local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
    return dict(rank=m["rank"], metro=m["metro"], tz=m["tz"], direction=direction, day=day.isoformat(),
                local=f"{hh:02d}:{mm:02d}", at=local.isoformat(timespec="seconds"),
                utc=local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"), o=o, d=d, **extra)


def jobs(only=()):
    """one dict per request: metro, direction, day, local departure, its UTC instant, origin and destination"""
    H = json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]
    return [job(m, m["home"], direction, day, hh, mm) for m in H if not only or m["rank"] in only
            for direction in WINDOWS for day in DAYS for hh, mm in departures(direction)]


def more_jobs(only=()):
    """the same for the extra suburbs (not the main one), with their sector and name"""
    H = json.load(open(os.path.join(MEDIA, "homes_more.json")))["metros"]
    return [job(m, h, direction, MORE_DAY, hh, mm, sector=h["sector"], home=h["name"])
            for m in H if not only or m["rank"] in only for h in m["homes"] if not h["main"]
            for direction in WINDOWS for hh, mm in departures(direction, MORE_STEP)]
