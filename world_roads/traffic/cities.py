"""The trips the video can be set on. Pick one with TRAFFIC_CITY=nyc|calgary (default nyc).

Each city: the Overture fetch box, a local projection centre, the trip's two ends (and the road classes to
snap them to), default speeds where OpenStreetMap has no maxspeed, and the short words the video uses for
its three routes (written after looking at what city_routes' search picked: see ACCURACY.md).
"""
import os

MPH = 1.609344
CITIES = {
    "nyc": dict(
        name="New York", trip="Times Square to Bayside, Queens",
        bbox=(-74.08, 40.56, -73.70, 40.90),                 # Queens, Brooklyn, Manhattan, the south Bronx, the Hudson shore
        centre=(-73.88, 40.71),
        # an evening drive home: from work near Times Square to a house in Bayside
        origin=(-73.9866, 40.7570, "Times Square"), dest=(-73.7771, 40.7686, "Bayside"),
        pin_names=("Work · Times Square", "Home · Bayside"),
        snap_origin=("primary", "secondary", "tertiary"),
        snap_dest=("residential", "tertiary", "secondary"),
        # New York's default limit is 25 mph unless posted; expressways are typically posted 45-50 mph
        default_kmh={"motorway": 50 * MPH, "trunk": 40 * MPH, "primary": 25 * MPH, "secondary": 25 * MPH,
                     "tertiary": 25 * MPH, "unclassified": 25 * MPH, "residential": 25 * MPH, "living_street": 10 * MPH},
        link_kmh=30 * MPH, via_radius_km=18.0,
        # the fastest way home through each of three river crossings out of Manhattan: east under the river, north
        # through the Bronx (the RFK Bridge, then the Throgs Neck), south round Brooklyn
        crossings=[["Queens-Midtown Tunnel"], ["Robert F. Kennedy Bridge"], ["Brooklyn-Battery Tunnel"]],
        names=["Midtown Tunnel", "Throgs Neck Bridge", "Battery Tunnel"],
        sides=["east, under the river", "north, through the Bronx", "south, round Brooklyn"], units="mph",
        d_rush=4000.0,                                       # toy-model drivers per hour in the evening rush: two routes in use
        calm=dict(replan=0.02, theta=1.0),                   # the learning run that settles: 1 in 50 rethink each day
        learn_start=[0.8, 0.15, 0.05],                       # "Monday": most drivers try the route that is fastest when empty
        names_in=["the Midtown Tunnel", "the Throgs Neck Bridge", "the Battery Tunnel"],   # as they read mid-sentence
        pin_labels=("b", "t"),                               # where the start / destination labels sit
        # Braess's paradox on the map: a mythical tunnel from the end of route A's bottleneck (the Queens-Midtown
        # Tunnel) to the start of route B's (the Throgs Neck Bridge). A thought experiment with its own made-up numbers.
        myth=dict(from_route=0, from_road="Queens-Midtown Tunnel", to_route=1, to_road="Throgs Neck Bridge", minutes=3.0,
                  cap_from=800.0, cap_to=1000.0, demand=2500.0)),
    "calgary": dict(
        name="Calgary", trip="Rocky Ridge to Seton",
        bbox=(-114.40, 50.79, -113.78, 51.25),              # Calgary with the whole Stoney Trail ring road
        centre=(-114.0631, 51.0461),
        origin=(-114.2265, 51.1440, "Rocky Ridge"), dest=(-113.9535, 50.8810, "Seton"),
        snap_origin=("residential", "tertiary", "secondary", "unclassified"),
        snap_dest=("residential", "tertiary", "secondary", "unclassified"),
        # Calgary's 40 km/h residential default (2021), otherwise typical limits
        default_kmh={"motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 50, "unclassified": 50,
                     "residential": 40, "living_street": 20},
        link_kmh=60, via_radius_km=22.0,
        names=["Stoney Trail", "Crowchild & Deerfoot", "Barlow & 52 St"],
        sides=["around the west", "through the middle", "up the east side"], units="km/h",
        d_rush=5500.0, calm=dict(replan=0.05, theta=1.0),
        names_in=["Stoney Trail", "Crowchild & Deerfoot", "Barlow & 52 St"], pin_labels=("l", "r")),
}
CITY = os.environ.get("TRAFFIC_CITY", "nyc")
C = CITIES[CITY]
