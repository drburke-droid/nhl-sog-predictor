"""The story, as data: ancestry components (sand colours), populations (vessels), events, captions.

Times are in years before present (ybp). Fractions are the share of the destination's gene pool that
comes from the source by the end of the event. See README.md for the sources behind each number.
"""

TITLE = "200,000 years of human migration"
SUBTITLE = ""                    # none: the title says it

# ---------------------------------------------------------------- ancestry components (sand colours)
# key, legend label, colour
COMPONENTS = [
    ("KHS", "Southern African (Khoe-San)", "#e8606b"),
    ("CAH", "Central African forager", "#b97a4c"),
    ("WAF", "West & Central African", "#f29a46"),
    ("EAF", "East African", "#f1cb5a"),
    ("NEA", "Neanderthal", "#dfe5ee"),
    ("DEN", "Denisovan", "#93dbe8"),
    ("ANF", "Near Eastern farmer", "#38bfae"),
    ("IRN", "Iranian & Caucasus", "#62c486"),
    ("WHG", "European hunter-gatherer", "#4d94e8"),
    ("ANE", "Ancient North Eurasian", "#7173e0"),
    ("AAS", "South Asian forager (AASI)", "#c970c9"),
    ("EAS", "East Asian", "#b4dc48"),
    ("AUS", "Papuan & Aboriginal Australian", "#9a6ae0"),
    ("NAT", "Native American", "#ec5e93"),
]
KEYS = [c[0] for c in COMPONENTS]
ARCHAIC = {"NEA", "DEN"}

# ---------------------------------------------------------------- vessels
# Each vessel is the glass outline of a group of countries (Natural Earth ADM0_A3 codes).
#   bbox: keep only the polygons whose interior point lies in (lon0, lon1, lat0, lat1) -- drops overseas
#         territories (French Guiana from France) and carves islands out (Hawai'i from the USA)
#   clip: cut the countries listed in clip_codes (default: all) to this lon/lat box first
#         (European vs Asian Russia)
#   buffer: thicken island chains too small to hold sand at true size (degrees)
EUROPE = ["PRT", "ESP", "FRA", "ITA", "CHE", "AUT", "DEU", "BEL", "NLD", "LUX", "GBR", "IRL", "ISL", "NOR", "SWE",
          "FIN", "DNK", "POL", "CZE", "SVK", "HUN", "SVN", "HRV", "BIH", "SRB", "MNE", "KOS", "ALB", "MKD", "GRC",
          "BGR", "ROU", "BLR", "LTU", "LVA", "EST", "AND", "MCO", "LIE", "SMR", "VAT", "MLT", "FRO", "IMN"]
VESSELS = {
    "KHS": dict(label="Southern Africa", codes=["ZAF", "NAM", "BWA", "ZWE", "MOZ", "LSO", "SWZ", "ZMB", "MWI"]),
    "CAH": dict(label="Central Africa", codes=["COD", "COG", "CAF", "CMR", "GAB", "GNQ", "AGO"]),
    "EAF": dict(label="East Africa", codes=["ETH", "ERI", "DJI", "SOM", "SOL", "KEN", "UGA", "TZA", "RWA", "BDI",
                                            "SDS", "SDN"]),
    "WAF": dict(label="West Africa", codes=["NGA", "GHA", "CIV", "SEN", "MLI", "BFA", "NER", "GIN", "SLE", "LBR",
                                            "TGO", "BEN", "GMB", "GNB", "TCD"]),
    "NAF": dict(label="North Africa", codes=["MAR", "DZA", "TUN", "LBY", "EGY", "SAH", "MRT"]),
    "MDG": dict(label="Madagascar", codes=["MDG"]),
    "LEV": dict(label="Near East", codes=["TUR", "SYR", "LBN", "ISR", "PSX", "JOR", "IRQ", "SAU", "YEM", "OMN", "ARE",
                                          "QAT", "KWT", "BHR", "CYP", "CYN"]),
    "IRN": dict(label="Iran & Caucasus", codes=["IRN", "ARM", "AZE", "GEO", "AFG"]),
    "EUR": dict(label="Europe", codes=EUROPE, bbox=(-30, 45, 34, 72)),
    "STP": dict(label="Steppe & Central Asia", codes=["UKR", "MDA", "RUS", "KAZ", "UZB", "TKM", "KGZ", "TJK"],
                clip=(-180, 40, 60, 90), clip_codes=["RUS"], bbox=(20, 90, 35, 75)),
    "SIB": dict(label="Siberia", codes=["RUS"], clip=(60, 40, 190, 90), bbox=(60, 200, 40, 80)),
    "SAS": dict(label="South Asia", codes=["IND", "PAK", "BGD", "NPL", "LKA", "BTN"]),
    "SEA": dict(label="SE Asia", codes=["THA", "VNM", "LAO", "KHM", "MMR"]),
    "EAS": dict(label="East Asia", codes=["CHN", "MNG", "KOR", "PRK", "JPN", "TWN", "HKG", "MAC"]),
    "ISE": dict(label="Island SE Asia", codes=["PHL", "IDN", "MYS", "BRN", "TLS", "SGP"], bbox=(90, 134, -12, 22)),
    "PNG": dict(label="New Guinea", codes=["PNG", "IDN"], bbox=(134, 160, -12, 0)),
    "AUS": dict(label="Australia", codes=["AUS"]),
    "POL": dict(label="Polynesia", codes=["USA"], bbox=(-161, -154, 18, 23), note="(Hawai'i shown)",
                buffer=0.7),           # at true size, the islands thickened just enough to hold a few grains
    "NAM": dict(label="North America", codes=["USA", "CAN"], bbox=(-180, -50, 24, 85)),
    "MES": dict(label="Mexico & C. America", codes=["MEX", "GTM", "BLZ", "HND", "SLV", "NIC", "CRI", "PAN"]),
    "CAR": dict(label="Caribbean", codes=["CUB", "HTI", "DOM", "JAM", "PRI"], buffer=0.3),
    "AMZ": dict(label="Brazil & north", codes=["BRA", "COL", "VEN", "GUY", "SUR", "FRA"], bbox=(-80, -30, -35, 13)),
    "ADS": dict(label="Andes", codes=["PER", "BOL", "ECU"]),
    "GRL": dict(label="Greenland", codes=["GRL"]),
    "NZL": dict(label="New Zealand", codes=["NZL"], bbox=(160, 180, -50, -30)),
    "SCN": dict(label="Southern Cone", codes=["ARG", "CHL", "URY", "PRY"], bbox=(-80, -50, -56, -17)),
}

# Who is there 200,000 years ago: vessel -> (component, fill). Neanderthals and Denisovans have no
# countries, so their sand starts inside the Eurasian vessels and is displaced when modern humans arrive.
INITIAL = {
    "KHS": ("KHS", 0.50),
    "CAH": ("CAH", 0.50),
    "EAF": ("EAF", 0.55),
    "WAF": ("WAF", 0.45),
    "NAF": ("EAF", 0.22),
    "EUR": ("NEA", 1.0),            # archaic peoples: sand at the same density as people in Africa then
    "LEV": ("NEA", 1.0),            # (x this factor), so a grain means the same number of people
    "IRN": ("NEA", 1.0),
    "STP": ("NEA", 0.75),
    "SIB": ("DEN", 0.6),
    "EAS": ("DEN", 1.0),
    "SEA": ("DEN", 1.0),
    "ISE": ("DEN", 1.0),
}


# ---------------------------------------------------------------- population
# World population (years ago -> people), from the usual long-run estimates (Deevey; McEvedy & Jones;
# HYDE 3.2; UN World Population Prospects). Before farming these are guesses within a factor of ~3.
WORLD_POP = [(200_000, 1.0e5), (70_000, 2.0e5), (50_000, 5.0e5), (20_000, 1.0e6), (12_000, 4.0e6),
             (5_000, 1.4e7), (2_000, 2.3e8), (1_000, 3.1e8), (500, 4.6e8), (300, 6.0e8), (225, 9.8e8),
             (125, 1.65e9), (75, 2.5e9), (25, 6.1e9), (0, 8.1e9)]

# Each vessel's population, millions, at 1 CE, 1500, 1800, 1950 and today (approximate; McEvedy &
# Jones 1978, HYDE 3.2, UN WPP 2024, summed over each vessel's countries; the Americas in 1500 after
# Denevan 1992 and Koch et al. 2019, Australia in 1788 after the ABS minimum). Before farming (> 10,000
# years ago) people are spread by habitable area instead, blending into these by 1 CE.
POP_YEARS = [2_000, 525, 225, 75, 0]
POP = {
    "KHS": [1, 3, 5, 30, 164], "CAH": [3, 8, 13, 35, 186], "EAF": [5, 12, 18, 55, 407],
    "WAF": [8, 20, 30, 75, 456], "NAF": [10, 10, 13, 50, 221], "MDG": [0.01, 1, 2, 4, 31],
    "LEV": [15, 20, 25, 60, 289], "IRN": [5, 6, 9, 30, 149], "EUR": [30, 70, 150, 400, 590],
    "STP": [3, 10, 35, 120, 230], "SIB": [0.5, 1, 2, 25, 36],
    "SAS": [50, 110, 200, 450, 1907],
    "SEA": [5, 15, 28, 80, 251], "EAS": [60, 110, 340, 670, 1647], "ISE": [4, 12, 25, 110, 432],
    "PNG": [0.5, 1, 1.5, 2, 16], "AUS": [0.5, 0.75, 0.75, 8, 27], "POL": [0.01, 0.4, 0.3, 1, 2],
    "NAM": [1.5, 4, 7, 170, 375], "MES": [5, 22, 6, 35, 181], "CAR": [0.2, 3, 2.5, 15, 40],
    "AMZ": [2, 8, 4, 70, 298], "ADS": [4, 12, 3, 15, 64], "SCN": [0.5, 2, 1.5, 25, 76],
    "GRL": [0.001, 0.003, 0.008, 0.02, 0.056], "NZL": [0.0, 0.1, 0.1, 2, 5.2],
}
# The world's urban agglomerations of about 5 million people or more: (name, lon, lat, millions in 2025,
# rounded; UN World Urbanization Prospects 2018). Each is a mountain of poured sand: its summit height
# stands for its population (city_height), its flanks keep to a sand-pile slope, so bigger cities spread
# wider bases, and shoulders and ridges keep it irregular. The rest of a region's people are spread by
# density, levelled off below the mountains.
MEGACITIES = [
    ("Tokyo", 139.7, 35.7, 37.0), ("Delhi", 77.2, 28.6, 33.0), ("Shanghai", 121.5, 31.2, 29.5),
    ("Dhaka", 90.4, 23.8, 23.9), ("S\u00e3o Paulo", -46.6, -23.5, 22.8), ("Cairo", 31.2, 30.0, 22.6),
    ("Mexico City", -99.1, 19.4, 22.5), ("Beijing", 116.4, 39.9, 22.1), ("Mumbai", 72.9, 19.1, 21.7),
    ("Osaka", 135.5, 34.7, 18.9), ("New York", -74.0, 40.7, 18.9), ("Chongqing", 106.5, 29.6, 17.8),
    ("Karachi", 67.0, 24.9, 17.6), ("Kinshasa", 15.3, -4.3, 17.0), ("Lagos", 3.4, 6.5, 16.5),
    ("Istanbul", 29.0, 41.0, 16.0), ("Buenos Aires", -58.4, -34.6, 15.6), ("Kolkata", 88.4, 22.6, 15.5),
    ("Manila", 121.0, 14.6, 14.9), ("Guangzhou", 113.3, 23.1, 14.6), ("Lahore", 74.3, 31.5, 14.4),
    ("Tianjin", 117.2, 39.1, 14.2), ("Bangalore", 77.6, 13.0, 14.0), ("Rio de Janeiro", -43.2, -22.9, 13.8),
    ("Shenzhen", 114.1, 22.5, 13.5), ("Moscow", 37.6, 55.8, 12.7), ("Los Angeles", -118.2, 34.1, 12.5),
    ("Chennai", 80.3, 13.1, 12.3), ("Bogot\u00e1", -74.1, 4.7, 11.8), ("Jakarta", 106.8, -6.2, 11.4),
    ("Lima", -77.0, -12.0, 11.4), ("Paris", 2.35, 48.9, 11.3), ("Bangkok", 100.5, 13.8, 11.2),
    ("Hyderabad", 78.5, 17.4, 11.1), ("Seoul", 127.0, 37.6, 10.0), ("Chengdu", 104.1, 30.7, 9.9),
    ("London", -0.1, 51.5, 9.8), ("Nagoya", 136.9, 35.2, 9.6), ("Tehran", 51.4, 35.7, 9.5),
    ("Nanjing", 118.8, 32.1, 9.4), ("Ho Chi Minh City", 106.7, 10.8, 9.3), ("Luanda", 13.2, -8.8, 9.3),
    ("Chicago", -87.6, 41.9, 8.9), ("Wuhan", 114.3, 30.6, 8.6), ("Ahmedabad", 72.6, 23.0, 8.6),
    ("Kuala Lumpur", 101.7, 3.1, 8.6), ("Xi'an", 108.9, 34.3, 8.3), ("Dar es Salaam", 39.3, -6.8, 7.8),
    ("Hong Kong", 114.2, 22.3, 7.7), ("Baghdad", 44.4, 33.3, 7.7), ("Riyadh", 46.7, 24.7, 7.5),
    ("Santiago", -70.7, -33.4, 6.9), ("Madrid", -3.7, 40.4, 6.7), ("Toronto", -79.4, 43.7, 6.4),
    ("Johannesburg", 28.0, -26.2, 6.2), ("Khartoum", 32.5, 15.6, 6.2), ("Singapore", 103.8, 1.35, 6.0),
    ("Abidjan", -4.0, 5.3, 5.9), ("Yangon", 96.2, 16.8, 5.6), ("Addis Ababa", 38.7, 9.0, 5.5),
    ("Nairobi", 36.8, -1.3, 5.3), ("Melbourne", 145.0, -37.8, 5.2), ("Sydney", 151.2, -33.9, 5.1),
    ("Antananarivo", 47.5, -18.9, 4.0), ("Santo Domingo", -69.9, 18.5, 3.6), ("Port-au-Prince", -72.3, 18.5, 2.9),
]
import os as _os
# Migration streams: flying grains drawn per grain that lands. Every stream uses the same multiple, so streams
# stay in proportion to one another (a flow twice as large is twice as dense); >1 only makes them easier to see.
FLY_PER_GRAIN = int(_os.environ.get("SAND_FLY_PER_GRAIN", "3"))
CITY_H5 = 22.0          # summit height of a 5-million city, in grain levels
CITY_EXP = 0.8          # summit height grows as population^0.8: in order of size, without needles
CITY_SLOPE = 6.0        # mean flank slope, grain levels per cell (about 48 degrees): a pile, not a tower
CITY_MIN_R = 2.2        # cells: the narrowest base, so a small city is a low mound, not a spike
CITY_RELIEF = 0.45      # where the countryside is already high, a city still rises this share of its height above it
CITY_SHARE = 0.7        # city mountains hold at most this share of a region's sand (beyond it they steepen)
BACKGROUND_CAP = 60     # grains: the most the countryside piles up; only cities rise above it


def city_height(millions):
    """A city mountain's summit height in grain levels: 22 for 5 million, 2x population = 1.74x height
    (a 1-million city: 6.6; Calgary, 1.7 million: 9.2). Works on numbers or numpy arrays."""
    import numpy as np
    return CITY_H5 * (np.maximum(millions, 0.0) / 5.0) ** CITY_EXP


def _load_city_hills():
    """Every other urban agglomeration of 1 million people or more in 2025 (UN World Urbanization Prospects
    2018, File 12: data/wup2018_cities_1m.csv): (name, lon, lat, millions). Each is a hill of the same kind
    as the megacity mountains. Places within 0.3 degrees of a megacity are that megacity."""
    import csv, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wup2018_cities_1m.csv")
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        lat, lon, mil = float(r["lat"]), float(r["lon"]), float(r["pop2025_thousands"]) / 1000.0
        if any(abs(lat - la) < 0.3 and abs((lon - lo + 180.0) % 360.0 - 180.0) < 0.3 for _, lo, la, _ in MEGACITIES):
            continue
        out.append((r["name"], lon, lat, mil))
    return out


CITY_HILLS = _load_city_hills()


def urban_frac(year):
    """A megacity's population as a share of today's, by calendar year (world urban growth, roughly:
    UN WUP; Bairoch 1988): most were small until 1900 and grew fastest after 1950."""
    pts = [(1, 0.01), (1500, 0.02), (1800, 0.04), (1900, 0.10), (1950, 0.25), (1975, 0.45), (2000, 0.75),
           (2015, 0.92), (2025, 1.0)]
    if year <= pts[0][0]:
        return 0.0
    for (y0, f0), (y1, f1) in zip(pts, pts[1:]):
        if year <= y1:
            return f0 + (f1 - f0) * (year - y0) / (y1 - y0)
    return 1.0


# Before farming, people spread by habitable land: forager densities were lower in the Arctic and sub-Arctic,
# the great deserts and the high Andes (Binford 2001; Kelly 2013). Share of a vessel's area counted as
# habitable, for the pre-farming sand (1 elsewhere).
HABITABLE = {"SIB": 0.35, "NAM": 0.5, "STP": 0.6, "NAF": 0.5, "AUS": 0.45, "SCN": 0.7, "ADS": 0.75, "IRN": 0.8}

MIN_VISIBLE = 0.35      # magnified island peoples (Hawai'i, the Caribbean...) keep at least this many grains per cell

# ---------------------------------------------------------------- where each ancestry lives within a vessel
# (components, lon, lat, sigma in degrees, strength): grains of these components are drawn toward the
# spot (strength is a log-weight: 1.0 = about 2.7x more likely at the centre). Grains are only moved
# around inside a vessel, so each vessel's totals are unchanged. From: Bryc et al. 2015 (US),
# Moreno-Estrada et al. 2014 (Mexico), Kehdy et al. 2015 and Pena et al. 2011 (Brazil), Ruiz-Linares
# et al. 2014 (Latin America), Schlebusch et al. 2012 and Chimusa et al. 2015 (southern Africa),
# Pickrell et al. 2014 (Ethiopia), Narasimhan et al. 2019 (South Asia), Jeong et al. 2019 (Eurasian
# clines), Lipson et al. 2014 (Indonesia), Pierron et al. 2014 (Madagascar), census data for cities.
EU = "ANF IRN WHG ANE"
ANCESTRY_GEO = {
    "NAM": [("WAF", -86, 33, 7, 1.6), ("WAF", -76.5, 39, 3, 0.8), ("WAF", -90.5, 30.5, 3, 0.8),
            ("NAT", -108, 35, 6, 1.4), ("NAT", -99, 29, 5, 1.0), ("NAT", -118, 34, 3, 0.8),
            ("NAT", -150, 64, 8, 1.6), ("NAT", -100, 62, 12, 1.5), ("NAT", -97, 35.5, 3, 0.8),
            ("EAS", -122.3, 37.7, 3, 1.6), ("EAS", -118.2, 34, 2.5, 1.0), ("EAS", -123.1, 49.3, 2.5, 1.5),
            ("EAS", -74, 40.7, 2, 0.8), ("EAS", -79.4, 43.7, 2, 1.0),
            ("AAS", -74.2, 40.6, 2, 1.2), ("AAS", -79.6, 43.6, 2, 1.6), ("AAS", -122, 37.4, 2, 1.0),
            ("WHG ANE", -94, 46, 8, 0.4)],
    "MES": [("NAT", -94, 17, 4, 1.4), ("NAT", -89, 20, 2.5, 1.0), ("NAT", -91, 15, 2.5, 1.2),
            (EU, -106, 28, 6, 0.9), (EU, -84, 10, 2, 0.8),
            ("WAF", -96, 19, 2, 1.2), ("WAF", -87, 16, 2.5, 1.4), ("WAF", -79.5, 9, 1.5, 1.4)],
    "AMZ": [("WAF", -40, -12, 5, 1.6), ("WAF", -44, -3, 3, 0.8), ("WAF", -75.5, 9, 2.5, 1.3),
            ("WAF", -67, 10.5, 2.5, 0.8), ("WAF", -56, 5, 2, 1.2),
            (EU, -51, -27, 5, 1.3), (EU, -46.6, -23.5, 2.5, 0.7),
            ("NAT", -62, -3, 9, 1.4), ("NAT", -74, 4.5, 3, 0.6), ("WAF", -77, 5.5, 2, 1.2)],
    "SCN": [("NAT", -58, -23, 5, 1.3), ("NAT", -71, -38, 5, 1.0), ("NAT", -65, -24, 4, 1.0),
            (EU, -58.4, -34.6, 4, 1.0)],
    "ADS": [(EU, -77, -12, 2.5, 1.0), (EU, -79.9, -2.2, 2.5, 0.8), ("WAF", -79.5, 0.5, 2, 1.5),
            ("WAF", -76.5, -13, 2, 0.8), ("NAT", -70, -15, 5, 1.0)],
    "CAR": [("WAF", -72.5, 19, 1.5, 1.4), ("WAF", -77.3, 18.1, 1.0, 1.5), (EU, -80, 22.5, 2.5, 1.0),
            (EU, -66.5, 18.2, 1.0, 0.7), ("NAT", -66.5, 18.2, 1.0, 1.0)],
    "KHS": [("KHS", 20, -22, 4.5, 1.6), ("KHS", 19.5, -32, 3, 1.2), (EU, 18.5, -33.9, 2, 1.4),
            (EU, 28, -26, 2, 1.0), ("AAS", 31, -29.8, 1.5, 1.5)],
    "EAF": [("ANF IRN", 39, 11, 5, 1.2), ("ANF IRN", 45, 7, 5, 0.8), ("ANF IRN", 33, 17, 5, 1.0),
            ("WAF", 33, -5, 7, 1.3), ("WAF", 29.5, -2, 2, 1.0)],
    "NAF": [("WAF", -10, 18, 6, 1.4), ("WAF", 32, 23, 3, 0.8), ("WAF", 5, 25, 7, 0.6), ("IRN", 31, 29, 5, 0.6)],
    "CAH": [("CAH", 16, 1, 5, 1.5), ("CAH", 28, 1, 3, 1.0)],
    "SAS": [("IRN ANE WHG ANF", 72, 30, 6, 1.0), ("AAS", 79, 13, 6, 0.9), ("AAS", 83, 21, 4, 0.8),
            ("EAS", 92, 25, 3.5, 1.8), ("EAS", 85, 28, 3, 1.2)],
    "STP": [("EAS", 75, 46, 8, 1.4), ("EAS", 46, 46, 3, 0.6), ("ANF WHG", 31, 49, 6, 0.6)],
    "SIB": [("EAS", 130, 63, 12, 1.3), ("EAS", 108, 52, 5, 1.2), ("EAS", 94, 51.5, 3, 1.2),
            ("EAS", 170, 65, 10, 1.0), ("ANF IRN WHG", 83, 55, 9, 0.6)],
    "EUR": [("WHG ANE", 25, 60, 9, 0.8), ("ANF", 13, 41, 7, 0.9), ("ANF", -4, 40, 5, 0.6), ("IRN", 22, 40, 5, 0.9),
            ("WAF", 2.35, 48.85, 2, 1.4), ("WAF", -0.1, 51.5, 2, 1.2), ("AAS", -1.9, 52.5, 2, 1.4),
            ("EAS", -0.1, 51.5, 2, 0.6)],
    "LEV": [("EAS", 35, 39, 5, 1.0), ("IRN", 45, 32, 5, 0.8), ("ANF", 35.5, 33, 3, 0.5), ("WAF EAF", 44, 15, 4, 1.0)],
    "IRN": [("EAS", 66, 34, 4, 1.3), ("ANF", 43, 42, 3, 0.6)],
    "AUS": [("AUS", 133, -20, 8, 1.6), ("AUS", 130.8, -12.5, 3, 1.2), ("EAS", 151.2, -33.9, 1.5, 1.2),
            ("EAS", 145, -37.8, 1.5, 1.0), ("AAS", 145, -37.8, 1.5, 0.8)],
    "ISE": [("AUS", 125, -8, 5, 1.6), ("AUS", 129, -3, 4, 1.4), ("DEN", 123, -8, 5, 1.0), ("DEN", 120.3, 15.2, 2, 1.2)],
    "PNG": [("EAS", 151, -5, 4, 1.4), ("EAS", 146, -6, 2, 0.6)],
    "MDG": [("EAS AUS", 47.5, -19, 2.5, 1.2), ("WAF", 44, -22, 3, 1.0)],
    "NZL": [("EAS AUS", 175.5, -38, 2.5, 1.4)],
    "GRL": [(EU, -51.7, 64.2, 2, 1.5)],
    "SEA": [("AAS", 101, 6, 3, 0.8), ("AAS IRN", 104.9, 11.6, 2, 0.5)],
}

# ---------------------------------------------------------------- routes (lon, lat waypoints)
# longitudes beyond 180 continue eastwards across the Pacific (the map wraps at the Bering Strait)
ATL_N = [(-25, 47), (-60, 44)]
ATL_S = [(-20, 2)]
ROUTES = {
    ("EAF", "LEV"): [(43, 13), (44, 22), (39, 29)],
    ("LEV", "SAS"): [(50, 28), (63, 26)],
    ("SAS", "SEA"): [(90, 20)],
    ("SIB", "NAM"): [(140, 64), (175, 67), (-165, 65), (-140, 60)],
    ("EAS", "POL"): [(128, 18), (150, -4), (175, -12), (-165, -15)],
    ("PNG", "POL"): [(175, -12), (-165, -15)],
    ("ADS", "POL"): [(-110, -18)],
    ("ISE", "MDG"): [(95, -8), (70, -14)],
    ("ISE", "POL"): [(140, -4), (165, -12), (-178, -15), (-162, -10)],
    ("EAF", "MDG"): [(42, -12)],
    ("SIB", "GRL"): [(140, 66), (175, 68), (-165, 69), (-125, 72), (-80, 76)],
    ("CAH", "NAM"): [(-5, -8), (-35, 12), (-72, 30)],
    ("CAH", "CAR"): [(-10, -6), (-40, 12)],
    ("CAH", "MES"): [(-10, -6), (-45, 10), (-80, 16)],
    ("CAH", "ADS"): [(-10, -6), (-38, -2), (-75, 0)],
    ("CAH", "SCN"): [(-10, -12), (-45, -30)],
    ("EUR", "NAM"): ATL_N,
    ("EUR", "MES"): [(-25, 40), (-60, 28)],
    ("EUR", "CAR"): [(-25, 38), (-50, 26)],
    ("EUR", "AMZ"): [(-20, 25), (-30, 0)],
    ("EUR", "ADS"): [(-20, 25), (-40, 5), (-70, 2)],
    ("WAF", "AMZ"): ATL_S,
    ("WAF", "CAR"): [(-30, 14)],
    ("WAF", "NAM"): [(-35, 22), (-72, 32)],
    ("WAF", "MES"): [(-35, 16), (-80, 18)],
    ("WAF", "ADS"): [(-25, 6), (-55, 4), (-75, 2)],
    ("EUR", "AUS"): [(-12, 30), (0, -12), (20, -38), (75, -42), (115, -36)],
    ("EUR", "POL"): [(-40, 0), (-75, -45), (-120, -35)],
    ("EAS", "NAM"): [(150, 40), (175, 45), (-165, 47), (-130, 48)],
    ("SAS", "NAM"): [(110, 30), (160, 38), (175, 41), (-165, 44), (-140, 46)],
    ("SAS", "EUR"): [(55, 38), (30, 47)],
    ("SAS", "SAF"): [(60, -5), (40, -25)],
    ("STP", "SIB"): [(75, 58)],
    ("SIB", "STP"): [(75, 58)],
    ("EAS", "STP"): [(90, 48), (65, 50)],
    ("WAF", "MDG"): [(20, 0), (38, -12)],
    ("WAF", "KHS"): [(15, -2)],
    ("WAF", "EAF"): [(20, 10)],
    ("NEA", "LEV"): [(33, 44)],
    ("SEA", "ISE"): [(114, 8)],
    ("ISE", "PNG"): [(135, -5)],
    ("MES", "AMZ"): [(-80, 5), (-62, -3)],
    ("NAM", "MES"): [(-104, 33)],
    ("MES", "ADS"): [(-82, 3)],
    ("AMZ", "CAR"): [(-62, 9)],
    ("NAM", "GRL"): [(-100, 70), (-75, 77)],
    ("EUR", "GRL"): [(-12, 63)],
    ("POL", "NZL"): [(-165, -20), (178, -33)],
    ("EUR", "NZL"): [(-12, 30), (0, -12), (20, -38), (75, -44), (140, -46)],
}


# ---------------------------------------------------------------- events
def found(t, src, dst, n, only=None, keep=0.0, until=None):
    """A small group from `src` founds the population in `dst` (n grains travel; the rest are copies).
    `only`: the founders carry only these components (e.g. the out-of-Africa group).
    If dst already holds archaic sand, it is displaced by `until` (ybp) except for the share `keep`
    that mixes in (admixture)."""
    return dict(kind="found", t0=t, t1=t, src=src, dst=dst, n=n, only=only, keep=keep, until=until)


def flow(t0, t1, src, dst, f, only=None, land=None, spread=1.0):
    """Gene flow: by t1 a fraction f of dst's gene pool comes from src (replacing, not adding).
    `only`: the migrants carry only these components of the source (e.g. the Bantu-speaking ancestry of
    West-Central Africa, without its forager share).
    `land`: (lon, lat) where the migrants arrive (default: just inside the coast they approach from).
    `spread`: how far across the land the arrivals settle (1 = the whole region by the end; less keeps
    them near where they landed)."""
    return dict(kind="flow", t0=t0, t1=t1, src=src, dst=dst, f=f, only=only, land=land, spread=spread)


def restrict(P, only):
    """The part of composition P made of the components in `only` (renormalised); P if only is None."""
    import numpy as np
    P = np.asarray(P, float).copy()
    if only:
        P = P * np.array([k in only for k in KEYS], float)
    return P / P.sum()


def tint(t0, t1, jar, sources, target):
    """Isolation and drift: grains of `sources` colours in `jar` turn into a new component."""
    return dict(kind="tint", t0=t0, t1=t1, jar=jar, sources=sources, target=target)


def extinct(t0, t1, jar):
    return dict(kind="extinct", t0=t0, t1=t1, jar=jar)


EVENTS = [
    # --- deep African structure: groups already apart 200,000 years ago, with contact that never stopped
    # (Schlebusch et al. 2017; Ragsdale et al. 2023). North Africa is home to H. sapiens from the start
    # (Jebel Irhoud, Hublin et al. 2017).
    flow(120_000, 100_000, "EAF", "KHS", 0.05),
    flow(110_000, 95_000, "WAF", "CAH", 0.08),
    flow(100_000, 85_000, "CAH", "EAF", 0.04),
    flow(90_000, 70_000, "WAF", "NAF", 0.15),

    # --- out of Africa (main dispersal 70,000-50,000 years ago), the great dispersal, archaic admixture.
    # Neanderthal mixing ~50,000-43,000 (Iasi et al. 2024; Sumer et al. 2025): it is carried by every
    # founder group, whatever the founding date shown. Sahul by at least ~50,000 (O'Connell et al. 2018;
    # possibly ~65,000, Clarkson et al. 2017).
    found(62_000, "EAF", "LEV", 12, only=["EAF"], keep=0.026, until=44_000),
    found(56_000, "LEV", "SAS", 14),
    found(54_000, "SAS", "SEA", 12, keep=0.0012, until=40_000),
    found(52_500, "SEA", "ISE", 12, keep=0.045, until=46_000),
    found(51_000, "ISE", "PNG", 10),
    found(50_000, "PNG", "AUS", 10),
    found(46_000, "LEV", "EUR", 14, until=40_000),
    found(45_000, "LEV", "IRN", 14, until=40_000),
    found(44_500, "SEA", "EAS", 12, keep=0.0012, until=36_000),
    found(43_000, "LEV", "SIB", 12, until=36_000),
    found(41_000, "EUR", "STP", 12, until=37_000),           # Kostenki (~37 ka), Sungir (~34 ka)

    # --- isolation: each region's sand drifts into its own colour
    tint(44_000, 20_000, "LEV", ["EAF"], "ANF"),
    tint(44_000, 22_000, "IRN", ["EAF", "ANF"], "IRN"),
    tint(44_000, 20_000, "EUR", ["EAF", "ANF"], "WHG"),
    tint(42_000, 24_000, "SIB", ["EAF", "ANF"], "ANE"),
    tint(50_000, 26_000, "SAS", ["EAF"], "AAS"),
    tint(48_000, 26_000, "SEA", ["EAF"], "AAS"),
    tint(44_000, 26_000, "EAS", ["EAF", "AAS"], "EAS"),
    tint(48_000, 26_000, "ISE", ["EAF", "AAS"], "AUS"),
    tint(47_000, 26_000, "PNG", ["EAF", "AAS"], "AUS"),
    tint(46_000, 26_000, "AUS", ["EAF", "AAS"], "AUS"),

    # --- Ice Age North: Ancient North Eurasians, Eastern hunter-gatherers, the peopling of the Americas
    tint(36_000, 22_000, "STP", ["EAF", "ANF"], "ANE"),
    flow(28_000, 22_000, "SIB", "STP", 0.35),
    flow(26_000, 21_000, "EAS", "SIB", 0.62),
    found(20_000, "SIB", "NAM", 9),                           # by at least ~20 ka; White Sands 23-21 ka
    flow(18_000, 14_500, "EUR", "STP", 0.40),
    flow(20_000, 15_000, "LEV", "NAF", 0.60),                 # Taforalt (~15 ka): ~64% Natufian-related
    flow(17_000, 14_000, "WAF", "NAF", 0.30),                 # ...and ~36% sub-Saharan
    tint(19_000, 13_000, "NAM", ["ANE", "EAS"], "NAT"),
    found(16_000, "NAM", "MES", 10),
    found(15_200, "MES", "ADS", 9),
    found(14_800, "MES", "AMZ", 9),
    found(14_600, "ADS", "SCN", 8),                           # Monte Verde ~14.5 ka
    tint(14_600, 12_500, "SCN", ["ANE", "EAS"], "NAT"),
    tint(16_000, 12_500, "MES", ["ANE", "EAS"], "NAT"),
    tint(15_200, 12_500, "ADS", ["ANE", "EAS"], "NAT"),
    tint(14_800, 12_500, "AMZ", ["ANE", "EAS"], "NAT"),
    flow(11_000, 7_000, "EAS", "SIB", 0.45),

    # --- farming (from ~11,500 years ago) spreads, and mixes with the foragers it meets
    flow(9_500, 7_000, "LEV", "IRN", 0.22),
    flow(9_000, 5_000, "IRN", "SAS", 0.42),                   # Iranian-related ancestry (Shinde et al. 2019)
    flow(8_500, 6_200, "LEV", "EUR", 0.80),
    flow(7_500, 5_500, "LEV", "NAF", 0.32),
    flow(8_000, 4_200, "IRN", "LEV", 0.30),
    found(6_500, "MES", "CAR", 8),                            # Archaic Caribbean settlers ~6 ka
    flow(2_600, 1_800, "AMZ", "CAR", 0.70),                   # Ceramic-age migrants (Fernandes et al. 2021)
    flow(6_800, 5_400, "IRN", "STP", 0.45),
    found(4_500, "SIB", "GRL", 6),                            # Paleo-Inuit (Saqqaq, Rasmussen et al. 2010)

    # --- steppe herders, the Bantu expansion, Austronesian voyagers
    flow(5_000, 4_300, "STP", "EUR", 0.50),
    flow(5_000, 4_300, "STP", "SIB", 0.10),
    flow(5_000, 3_000, "WAF", "CAH", 0.78),
    flow(5_000, 3_500, "EAS", "SEA", 0.92),
    flow(4_500, 3_200, "EAS", "ISE", 0.93),
    flow(4_000, 3_000, "STP", "SAS", 0.16),
    flow(3_600, 2_600, "LEV", "EAF", 0.30),
    flow(3_500, 2_600, "ISE", "PNG", 0.08),
    found(2_900, "ISE", "POL", 10),                           # Lapita to Tonga/Samoa ~2,850 ybp
    flow(2_600, 1_600, "PNG", "POL", 0.24),
    flow(3_000, 1_000, "EAS", "SAS", 0.04),
    flow(2_600, 1_500, "WAF", "EAF", 0.22),
    flow(2_500, 700, "EAS", "STP", 0.18),
    flow(2_000, 1_600, "EAF", "KHS", 0.05),
    flow(2_000, 900, "CAH", "KHS", 0.86, only=["WAF"]),       # Bantu-speaking farmers reach southern Africa
    flow(2_000, 1_000, "SAS", "SEA", 0.05),
    found(1_500, "ISE", "MDG", 10),
    flow(1_300, 800, "EAF", "MDG", 0.60, only=["WAF"]),       # East African Bantu speakers (Pierron et al. 2017)
    flow(1_350, 950, "LEV", "NAF", 0.18),
    flow(1_300, 500, "NAF", "EUR", 0.012),                    # North African ancestry in Iberia (Olalde et al. 2019)
    flow(1_500, 100, "EAF", "LEV", 0.05),                     # Red Sea and Indian Ocean trade, incl. slavery
    flow(1_000, 600, "LEV", "MDG", 0.04),
    flow(1_000, 800, "EAS", "LEV", 0.05),
    flow(1_000, 600, "EAS", "IRN", 0.04),                     # Turkic and Mongol periods
    flow(900, 700, "ADS", "POL", 0.012),                      # Native American contact (Ioannidis et al. 2020)
    flow(800, 650, "NAM", "GRL", 0.90),                       # Thule (Inuit ancestors) replace the Paleo-Inuit
    found(750, "POL", "NZL", 8),
    flow(1_000, 400, "WAF", "NAF", 0.08),
    flow(3_500, 2_600, "STP", "IRN", 0.08),

    # --- 1500 CE onward: ships, colonization, the slave trade, modern migration. Enslaved Africans came
    # from West Africa and, in large numbers, West-Central Africa (Angola, Congo): ~70% of those taken to
    # Brazil (Eltis & Richardson 2010; Micheletti et al. 2020).
    flow(525, 430, "EUR", "CAR", 0.85),
    flow(505, 150, "EUR", "MES", 0.36),
    flow(490, 80, "EUR", "AMZ", 0.72),
    flow(490, 120, "EUR", "ADS", 0.18),
    flow(480, 60, "EUR", "SCN", 0.64),
    flow(450, 200, "WAF", "SCN", 0.02),
    flow(450, 200, "CAH", "SCN", 0.035, only=["WAF"]),
    flow(475, 165, "WAF", "AMZ", 0.17),
    flow(475, 165, "CAH", "AMZ", 0.40, only=["WAF"]),
    flow(470, 170, "WAF", "MES", 0.026),
    flow(470, 170, "CAH", "MES", 0.026, only=["WAF"]),
    flow(450, 170, "WAF", "ADS", 0.016),
    flow(450, 170, "CAH", "ADS", 0.016, only=["WAF"]),
    flow(430, 165, "WAF", "CAR", 0.36),
    flow(430, 165, "CAH", "CAR", 0.27, only=["WAF"]),
    flow(405, 105, "EUR", "NAM", 0.92),
    flow(400, 160, "WAF", "NAM", 0.30),
    flow(400, 160, "CAH", "NAM", 0.11, only=["WAF"]),
    flow(373, 120, "EUR", "KHS", 0.045),
    flow(380, 80, "STP", "SIB", 0.88, only=["ANF", "IRN", "WHG", "ANE", "NEA"]),   # Russian settlers
    flow(237, 70, "EUR", "AUS", 0.95),
    flow(220, 70, "EUR", "POL", 0.12),
    flow(200, 60, "EAS", "SEA", 0.05),
    flow(165, 110, "SAS", "KHS", 0.025),
    flow(70, 5, "MES", "NAM", 0.10),
    # across the Pacific to western North America: the Gold Rush and railroad era (cut short by the
    # exclusion laws of 1882-1924), then the large migrations after 1965 (US) and 1967 (Canada). Today
    # about 5% of the US and Canada has East or Southeast Asian ancestry and about 2% South Asian, most of
    # it in the West: California is ~16% Asian, Metro Vancouver ~50% (2020 US census, 2021 Canadian census)
    flow(175, 80, "EAS", "NAM", 0.006, land=(-121.5, 37.5), spread=0.12),
    flow(60, 5, "EAS", "NAM", 0.05, land=(-120.5, 37.0), spread=0.35),
    flow(55, 5, "SAS", "NAM", 0.02, land=(-121.0, 39.0), spread=0.45),
    flow(70, 5, "NAF", "EUR", 0.03),
    flow(65, 5, "LEV", "EUR", 0.03),
    flow(60, 5, "WAF", "EUR", 0.02),
    flow(60, 5, "SAS", "EUR", 0.02),
    flow(55, 5, "EAS", "AUS", 0.11),
    flow(180, 60, "EUR", "NZL", 0.80),
    flow(45, 5, "EAS", "NZL", 0.10),
    flow(45, 5, "SAS", "NZL", 0.04),
    flow(300, 60, "EUR", "GRL", 0.22),
    flow(40, 5, "SAS", "AUS", 0.05),
]

# ---------------------------------------------------------------- the clock: (video seconds, ybp)
# between keys, log(ybp + 200) is linear in video time
# A short caption set, for showing the film with music and no narration (video seconds, text). Each line
# under ten words, keeping the hedges of the full captions.
SHORT_CAPTIONS = [
    (7.0, 12.0, "200,000 years ago, our species lives only in Africa"),
    (13.5, 19.0, "Each colour of sand is a strand of ancestry"),
    (31.0, 37.0, "Out of Africa, between 70,000 and 50,000 years ago"),
    (44.0, 51.0, "Along the way, mixing with Neanderthals and Denisovans"),
    (71.5, 77.5, "At least 20,000 years ago: into the Americas"),
    (84.0, 91.0, "About 11,000 years ago, farming begins"),
    (99.0, 106.0, "Steppe herders and Bantu farmers spread far"),
    (109.0, 117.0, "Austronesian sailors cross two oceans"),
    (132.0, 140.0, "1500 CE: ships link every continent"),
    (161.5, 167.5, "Today: almost every vessel holds sand from many places"),
    (212.5, 220.0, "Every person alive descends from those first African grains"),
]

CLOCK = [
    (6.0, 200_000),
    (24.0, 70_000),
    (58.0, 38_000),
    (82.0, 12_000),
    (116.0, 2_000),
    (132.0, 500),
    (160.0, 0),
]
TODAY_S = CLOCK[-1][0]
DURATION = TODAY_S + 62.0      # today: a low flight through the world's cities, then the reveal


# ---------------------------------------------------------------- captions: (ybp, text) or (video s, text)
CAPTIONS = [
    ("s", 6.5, "200,000 years ago, our species lives only in Africa, already in deep-rooted groups. "
              "Neanderthals (white sand) and Denisovans (pale blue) live in Eurasia."),
    ("s", 12.2, "Each glass vessel is a population, shaped like its land. Each colour of sand is a strand of ancestry."),
    ("s", 16.6, "Separated for long enough, a group's sand drifts into a colour of its own."),
    ("s", 21.4, "Some groups venture out of Africa early, about 120,000 years ago, but leave little trace in people today."),
    (62_000, "Sometime between 70,000 and 50,000 years ago, a small band crosses out of Africa."),
    (55_000, "They spread fast, along the coasts to South Asia and, by boat, to New Guinea and Australia: "
             "by at least 50,000 years ago, possibly earlier."),
    (49_000, "Along the way they mix with Neanderthals, mostly 50,000 to 43,000 years ago. People outside "
             "Africa carry about 2% Neanderthal DNA (white grains); many Africans carry a little too."),
    (44_800, "In Island Southeast Asia and New Guinea they mix with Denisovans: Papuans and Aboriginal "
             "Australians carry 3-5% Denisovan DNA (pale blue)."),
    (40_500, "People reach Europe, East Asia and the steppe. The Neanderthals and Denisovans eventually "
             "disappear; a little of their sand lives on in us."),
    (33_000, "Tens of thousands of years apart, each region's sand drifts into its own colour."),
    (25_000, "In Ice Age Siberia, East Asian and Ancient North Eurasian ancestry mix."),
    (20_200, "By at least 20,000 years ago, perhaps earlier, a few founders cross Beringia. A handful of "
             "grains seeds two whole continents: the founder effect."),
    (15_500, "Largely separate for 15,000 years, the First Peoples of the Americas become genetically distinct."),
    (11_000, "About 11,000 years ago farming begins in the Fertile Crescent (and later, independently, "
             "elsewhere). Farmers spread and mix with the foragers they meet."),
    (5_000, "Herders from the steppe sweep west into Europe and later south into South Asia."),
    ("s", 103.4, "The Bantu expansion carries West-Central African ancestry across half a continent."),
    ("s", 108.2, "Austronesian sailors from Taiwan island-hop across two oceans..."),
    (2_000, "...to Tonga and Samoa about 2,900 years ago (Hawai'i, shown here for Polynesia, about "
            "1,000 years ago), and to Madagascar, where Asian and African ancestry meet."),
    (520, "1500 CE: ships link every continent. Epidemics, war and enslavement devastate Indigenous peoples "
          "of the Americas, and 12.5 million Africans are taken across the Atlantic in chains."),
    (230, "Settlers transform the makeup of Australia, Siberia and southern Africa."),
    (70, "Modern migration keeps stirring the sand, and people gather in cities."),
    ("s", TODAY_S + 1.0, "Today almost every vessel holds sand from many places. Islands of ancestry endure, "
                 "from New Guinea to the Andes."),
    ("s", TODAY_S + 9.0, "And underneath, it was always the same sand. Every person alive descends from "
                 "those first African grains."),
]
