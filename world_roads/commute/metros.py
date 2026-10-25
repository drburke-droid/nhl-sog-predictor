"""The 20 largest North American metro areas and where their downtown office is.

Ranking: Wikipedia, "List of North American metropolitan areas by population" (MSA / CMA / zona metropolitana,
figures 2020-2025; page as of 25 Sep 2026). The office is a well-known downtown or business-district point, set
by hand (to within a block or two); Google snaps it to the nearest road.
"""
METROS = [
    # rank, name, country, time zone, office (name, lat, lon)
    (1, "Mexico City", "MX", "America/Mexico_City", ("Ángel de la Independencia, Paseo de la Reforma", 19.42697, -99.16771)),
    (2, "New York", "US", "America/New_York", ("Grand Central Terminal, Midtown Manhattan", 40.75274, -73.97723)),
    (3, "Los Angeles", "US", "America/Los_Angeles", ("Pershing Square, Downtown LA", 34.04848, -118.25270)),
    (4, "Chicago", "US", "America/Chicago", ("Willis Tower, the Loop", 41.87886, -87.63591)),
    (5, "Dallas–Fort Worth", "US", "America/Chicago", ("Main Street District, Downtown Dallas", 32.78070, -96.79900)),
    (6, "Houston", "US", "America/Chicago", ("City Hall, Downtown Houston", 29.76026, -95.36925)),
    (7, "Toronto", "CA", "America/Toronto", ("King and Bay, Financial District", 43.64860, -79.38110)),
    (8, "Atlanta", "US", "America/New_York", ("Five Points, Downtown Atlanta", 33.75387, -84.39138)),
    (9, "Washington", "US", "America/New_York", ("Farragut Square, Downtown DC", 38.90195, -77.03911)),
    (10, "Miami", "US", "America/New_York", ("Brickell City Centre", 25.76680, -80.19300)),
    (11, "Philadelphia", "US", "America/New_York", ("City Hall, Center City", 39.95238, -75.16362)),
    (12, "Monterrey", "MX", "America/Monterrey", ("Macroplaza, Centro", 25.66905, -100.30950)),
    (13, "Guadalajara", "MX", "America/Mexico_City", ("Catedral, Centro", 20.67680, -103.34720)),
    (14, "Phoenix", "US", "America/Phoenix", ("Washington and Central, Downtown Phoenix", 33.44838, -112.07404)),
    (15, "Boston", "US", "America/New_York", ("Post Office Square, Financial District", 42.35630, -71.05560)),
    (16, "Inland Empire", "US", "America/Los_Angeles", ("City Hall, Downtown Riverside", 33.98060, -117.37550)),
    (17, "San Francisco Bay Area", "US", "America/Los_Angeles", ("Montgomery and Market, Financial District", 37.78940, -122.40130)),
    (18, "Montreal", "CA", "America/Toronto", ("Place Ville Marie, Downtown", 45.50170, -73.56940)),
    (19, "Detroit", "US", "America/Detroit", ("Campus Martius, Downtown", 42.33140, -83.04660)),
    (20, "Santo Domingo", "DO", "America/Santo_Domingo", ("Piantini business district", 18.47230, -69.93980)),
]
