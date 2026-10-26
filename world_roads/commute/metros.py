"""The 20 largest North American metro areas, plus Vancouver, Calgary and Edmonton, and where their downtown office is.

Ranking: Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona metropolitana,
figures 2020-2025; page as of 25 Sep 2026). #20 there, Santo Domingo, has no TomTom Traffic Index data, so #21,
Seattle, takes its place. Vancouver (#28 there), Calgary and Edmonton are added on request. Calgary and Edmonton
are not on that list (it stops at #50, Havana, 2.14 million), so their 101 and 102 are ids that order them, not
ranks. The office is a well-known downtown or business-district point, set by hand (to within a block or two;
the three added ones from OpenStreetMap). The last field is the metro's city page in the TomTom Traffic Index.
"""
METROS = [
    # rank, name, country, time zone, office (name, lat, lon), TomTom Traffic Index city
    (1, "Mexico City", "MX", "America/Mexico_City", ("Ángel de la Independencia, Paseo de la Reforma", 19.42697, -99.16771), "mexico-city"),
    (2, "New York", "US", "America/New_York", ("Grand Central Terminal, Midtown Manhattan", 40.75274, -73.97723), "new-york-ny"),
    (3, "Los Angeles", "US", "America/Los_Angeles", ("Pershing Square, Downtown LA", 34.04848, -118.25270), "los-angeles-ca"),
    (4, "Chicago", "US", "America/Chicago", ("Willis Tower, the Loop", 41.87886, -87.63591), "chicago-il"),
    (5, "Dallas–Fort Worth", "US", "America/Chicago", ("Main Street District, Downtown Dallas", 32.78070, -96.79900), "dallas-tx"),
    (6, "Houston", "US", "America/Chicago", ("City Hall, Downtown Houston", 29.76026, -95.36925), "houston-tx"),
    (7, "Toronto", "CA", "America/Toronto", ("King and Bay, Financial District", 43.64860, -79.38110), "toronto"),
    (8, "Atlanta", "US", "America/New_York", ("Five Points, Downtown Atlanta", 33.75387, -84.39138), "atlanta-ga"),
    (9, "Washington", "US", "America/New_York", ("Farragut Square, Downtown DC", 38.90195, -77.03911), "washington-dc"),
    (10, "Miami", "US", "America/New_York", ("Brickell City Centre", 25.76680, -80.19300), "miami-fl"),
    (11, "Philadelphia", "US", "America/New_York", ("City Hall, Center City", 39.95238, -75.16362), "philadelphia-pa"),
    (12, "Monterrey", "MX", "America/Monterrey", ("Macroplaza, Centro", 25.66905, -100.30950), "monterrey"),
    (13, "Guadalajara", "MX", "America/Mexico_City", ("Catedral, Centro", 20.67680, -103.34720), "guadalajara"),
    (14, "Phoenix", "US", "America/Phoenix", ("Washington and Central, Downtown Phoenix", 33.44838, -112.07404), "phoenix-az"),
    (15, "Boston", "US", "America/New_York", ("Post Office Square, Financial District", 42.35630, -71.05560), "boston-ma"),
    (16, "Inland Empire", "US", "America/Los_Angeles", ("City Hall, Downtown Riverside", 33.98060, -117.37550), "riverside-ca"),
    (17, "San Francisco Bay Area", "US", "America/Los_Angeles", ("Montgomery and Market, Financial District", 37.78940, -122.40130), "san-francisco-ca"),
    (18, "Montreal", "CA", "America/Toronto", ("Place Ville Marie, Downtown", 45.50170, -73.56940), "montreal"),
    (19, "Detroit", "US", "America/Detroit", ("Campus Martius, Downtown", 42.33140, -83.04660), "detroit-mi"),
    (21, "Seattle", "US", "America/Los_Angeles", ("Westlake Center, Downtown Seattle", 47.61145, -122.33740), "seattle-wa"),
    (28, "Vancouver", "CA", "America/Vancouver", ("Burrard Station, Financial District", 49.28564, -123.11982), "vancouver"),
    (101, "Calgary", "CA", "America/Edmonton", ("Bankers Hall, Stephen Avenue, Downtown", 51.04524, -114.06911), "calgary"),
    (102, "Edmonton", "CA", "America/Edmonton", ("Sir Winston Churchill Square, Downtown", 53.54404, -113.48977), "edmonton"),
]
