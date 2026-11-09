"""The story, as data: what a grain is, the vessels (regions), the causes (colours), the clock and the cities.

Sand is people: one grain stands for GRAIN people, the same in every year, so a pile's size is its
population. Every period and place where deaths ran above the normal level for that place and time
(excess deaths) lifts that many grains off the table and out of the top of the frame, coloured by cause.
The data behind it is built by build_data.py (see README.md for sources and method).
"""
import json as _json, os as _os

HERE = _os.path.dirname(_os.path.abspath(__file__))
TITLE = "2,000 years of catastrophe"
SUBTITLE = "wars, plagues and famines, told in sand"

GRAIN = 5_000                   # people per grain, the whole film (constant: lifted grains are deaths)
# The sand grid: CELL map px per cell, one grain per level per cell. The migration film used 3-px cells at
# ~56,000 people per grain; here the balls are 2/3 that size (GS) so five times as many fit the same
# landscape, and pile heights in grain levels are HL times the migration film's.
CELL = 2
GS = CELL / 3.0
HL = 1.5                        # = 1/GS: piles and mountains as tall, in map px, as the migration film's
YEAR0, YEAR1 = 1, 2025          # the story's span (calendar years)

# ---------------------------------------------------------------- causes (the colours of lifted sand)
# key, legend label, colour
CAUSES = [
    ("war", "War, conquest & mass killing", "#ff5b4f"),
    ("epidemic", "Epidemic", "#b8e04a"),
    ("famine", "Famine", "#f5a33b"),
    ("disaster", "Earthquake, flood & storm", "#58a6ff"),
    ("other", "Other years above normal", "#c3b6e8"),
]
CAUSE_KEYS = [c[0] for c in CAUSES]
SAND_COLOUR = "#d8c6a0"         # resting sand: people, all one colour, so the lifted colours stand out

# ---------------------------------------------------------------- vessels (as in the migration film)
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


# UN World Population Prospects countries (ISO3) -> vessel. Small islands join the nearest vessel.
VESSEL_ISO = {
    "KHS": "ZAF NAM BWA ZWE MOZ LSO SWZ ZMB MWI",
    "CAH": "COD COG CAF CMR GAB GNQ AGO",
    "EAF": "ETH ERI DJI SOM KEN UGA TZA RWA BDI SSD SDN",
    "WAF": "NGA GHA CIV SEN MLI BFA NER GIN SLE LBR TGO BEN GMB GNB TCD CPV STP SHN",
    "NAF": "MAR DZA TUN LBY EGY ESH MRT",
    "MDG": "MDG MUS REU COM MYT SYC",
    "LEV": "TUR SYR LBN ISR PSE JOR IRQ SAU YEM OMN ARE QAT KWT BHR CYP",
    "IRN": "IRN ARM AZE GEO AFG",
    "EUR": ("PRT ESP FRA ITA CHE AUT DEU BEL NLD LUX GBR IRL ISL NOR SWE FIN DNK POL CZE SVK HUN SVN HRV BIH "
            "SRB MNE XKX ALB MKD GRC BGR ROU BLR LTU LVA EST AND MCO LIE SMR VAT MLT FRO IMN JEY GGY GIB"),
    "STP": "UKR MDA RUS KAZ UZB TKM KGZ TJK",
    "SIB": "",                  # Asian Russia: a share of RUS (siberia_share)
    "SAS": "IND PAK BGD NPL LKA BTN MDV",
    "SEA": "THA VNM LAO KHM MMR",
    "EAS": "CHN MNG KOR PRK JPN TWN HKG MAC",
    "ISE": "PHL IDN MYS BRN TLS SGP",
    "PNG": "PNG FJI SLB VUT NCL",
    "AUS": "AUS",
    "POL": "PYF WSM TON GUM KIR FSM ASM MNP MHL PLW COK NRU WLF TUV TKL NIU",
    "NAM": "USA CAN BMU SPM",
    "MES": "MEX GTM BLZ HND SLV NIC CRI PAN",
    "CAR": ("CUB HTI DOM JAM PRI TTO BHS GLP MTQ BRB CUW LCA GRD ABW VCT ATG VIR CYM DMA KNA TCA SXM VGB BES "
            "MAF AIA BLM MSR"),
    "AMZ": "BRA COL VEN GUY SUR GUF",
    "ADS": "PER BOL ECU",
    "GRL": "GRL",
    "NZL": "NZL",
    "SCN": "ARG CHL URY PRY FLK",
}
ISO_VESSEL = {i: v for v, s in VESSEL_ISO.items() for i in s.split()}
# Natural Earth ADM0_A3 codes that differ from ISO3 (for placing a country's grains on the map)
NE_TO_ISO = {"SDS": "SSD", "KOS": "XKX", "PSX": "PSE", "SAH": "ESH", "SOL": "SOM", "CYN": "CYP"}

# Asian Russia (east of 60 E, the Siberia vessel) as a share of Russia's people, by year: a few percent
# before the Russian settlement of Siberia, a quarter today (Russian censuses; Treadgold 1957).
SIB_SHARE = [(1, 0.08), (1500, 0.06), (1700, 0.05), (1800, 0.05), (1897, 0.09), (1926, 0.14), (1959, 0.22),
             (2025, 0.25)]


def siberia_share(year):
    import numpy as np
    y, s = zip(*SIB_SHARE)
    return float(np.interp(year, y, s))


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
CITY_H5 = 22.0 * HL          # summit height of a 5-million city, in grain levels
CITY_EXP = 0.8          # summit height grows as population^0.8: in order of size, without needles
CITY_SLOPE = 6.0        # mean flank slope, grain levels per cell (about 48 degrees): a pile, not a tower
CITY_MIN_R = 2.2 / GS        # cells: the narrowest base, so a small city is a low mound, not a spike
CITY_RELIEF = 0.45      # where the countryside is already high, a city still rises this share of its height above it
SPILL = False           # city mountains stay inside their vessel's glass (no spilling onto the sea)
CITY_SHARE = 0.7        # city mountains hold at most this share of a region's sand (beyond it they steepen)
BACKGROUND_CAP = 60 * HL     # grains: the most the countryside piles up; only cities rise above it


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




# ---------------------------------------------------------------- the clock (built by build_data.py)
# Video seconds against calendar years. Time runs at a steady pace through quiet centuries and slows
# whenever the world is dying faster than normal, so every catastrophe gets time on screen; the year
# on screen always says where we are.
def _load_clock():
    p = _os.path.join(HERE, "data", "clock.json")
    if _os.path.exists(p):
        return _json.load(open(p))
    return {"s": [0.0, 200.0], "year": [YEAR0, YEAR1], "intro": 7.0, "outro": 12.0}


CLOCK = _load_clock()
INTRO_S = CLOCK["intro"]        # the title over the empty-ish table, before year 1 starts to run
TODAY_S = CLOCK["s"][-1]
DURATION = TODAY_S + CLOCK["outro"]
# The closing: after the simulated film ends, its last picture holds for HOLD seconds while the closing words
# are spoken (the Cycles Edit scene holds the last rendered frame: no new frames to render).
HOLD = 23.0
FILM_END = DURATION + HOLD
TOTALS_OUT = 7.0                # seconds after the last year: the totals card fades, the closing words begin
CLOSING = [("Every one of your ancestors survived long enough to have a child.", 8.3),
           ("If a single one had not, you would not be here.", 20.6)]    # (text, seconds after the last year)
