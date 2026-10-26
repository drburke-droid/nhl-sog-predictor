"""What to ask a traffic-aware router: the departures for both commutes, shared by tomtom_routes.py and fetch.py

* evening: office -> home, leaving every 5 minutes from 3:00 to 8:00 p.m. local time
* morning: home -> office, leaving every 5 minutes from 6:00 to 10:00 a.m. local time

on three ordinary weekdays ahead (Tue-Thu, 17-19 Nov 2026): after the clocks change on 1 Nov, clear of US
Thanksgiving and of Mexico's Revolution Day holiday (Mon 16 Nov).
"""
import os, json
from datetime import datetime, date
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = os.path.join(os.path.dirname(HERE), "media", "commute")
DAYS = [date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)]
WINDOWS = {"evening": ((15, 0), (20, 0)), "morning": ((6, 0), (10, 0))}
STEP = 5                                                # minutes between departures


def departures(direction):
    (h0, m0), (h1, m1) = WINDOWS[direction]
    t = h0 * 60 + m0
    while t <= h1 * 60 + m1:
        yield t // 60, t % 60
        t += STEP


def jobs(only=()):
    """one dict per request: metro, direction, day, local departure, its UTC instant, origin and destination"""
    H = json.load(open(os.path.join(MEDIA, "homes.json")))["metros"]
    out = []
    for m in H:
        if only and m["rank"] not in only:
            continue
        tz = ZoneInfo(m["tz"])
        office, home = (m["office"]["lat"], m["office"]["lon"]), (m["home"]["lat"], m["home"]["lon"])
        for direction in WINDOWS:
            o, d = (office, home) if direction == "evening" else (home, office)
            for day in DAYS:
                for hh, mm in departures(direction):
                    local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
                    out.append(dict(rank=m["rank"], metro=m["metro"], direction=direction, day=day.isoformat(),
                                    local=f"{hh:02d}:{mm:02d}", at=local.isoformat(timespec="seconds"),
                                    utc=local.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"), o=o, d=d))
    return out
