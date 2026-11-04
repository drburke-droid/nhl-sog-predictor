"""The story, as data: ancestry components (sand colours), populations (vessels), events, captions.

Times are in years before present (ybp). Fractions are the share of the destination's gene pool that
comes from the source by the end of the event. See README.md for the sources behind each number.
"""

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
    ("AAS", "South Asian & Andamanese", "#c970c9"),
    ("EAS", "East Asian", "#b4dc48"),
    ("AUS", "Papuan & Aboriginal Australian", "#9a6ae0"),
    ("NAT", "Native American", "#ec5e93"),
]
KEYS = [c[0] for c in COMPONENTS]
ARCHAIC = {"NEA", "DEN"}

# ---------------------------------------------------------------- vessels
# Each vessel is the glass outline of a group of countries (Natural Earth ADM0_A3 codes).
#   bbox: keep only the polygons whose interior point lies in (lon0, lon1, lat0, lat1) -- drops overseas
#         territories (French Guiana from France) and carves islands out (Sardinia from Italy)
#   clip: cut the countries to this lon/lat box first (European vs Asian Russia)
#   inset: (magnification, (lon, lat) of the inset's centre) for islands too small to hold sand;
#          geo: the real location, joined to the inset by a hairline
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
    "MDG": dict(label="Madagascar", codes=["MDG"], inset=(1.5, (49, -19.5))),
    "LEV": dict(label="Near East", codes=["TUR", "SYR", "LBN", "ISR", "PSX", "JOR", "IRQ", "SAU", "YEM", "OMN", "ARE",
                                          "QAT", "KWT", "BHR", "CYP", "CYN"]),
    "IRN": dict(label="Iran & Caucasus", codes=["IRN", "ARM", "AZE", "GEO", "AFG"]),
    "EUR": dict(label="Europe", codes=EUROPE, bbox=(-30, 45, 34, 72),
                exclude=[(8.0, 10.0, 38.8, 41.4)]),
    "SAR": dict(label="Sardinia", codes=["ITA"], bbox=(8.0, 10.0, 38.8, 41.4), inset=(4.5, (-20, 40)),
                geo=(9.0, 40.0)),
    "STP": dict(label="Steppe & Central Asia", codes=["UKR", "MDA", "RUS", "KAZ", "UZB", "TKM", "KGZ", "TJK"],
                clip=(-180, 40, 60, 90), bbox=(20, 90, 35, 75)),
    "SIB": dict(label="Siberia", codes=["RUS"], clip=(60, 40, 190, 90), bbox=(60, 200, 40, 80)),
    "SAS": dict(label="South Asia", codes=["IND", "PAK", "BGD", "NPL", "LKA", "BTN"],
                exclude=[(91.5, 94.5, 6.0, 14.5)]),
    "AND": dict(label="Andaman Islands", codes=["IND"], bbox=(92.0, 93.5, 10.3, 14.0), inset=(4.5, (88.4, 10.8)), buffer=0.30,
                geo=(92.7, 11.7)),
    "SEA": dict(label="SE Asia", codes=["THA", "VNM", "LAO", "KHM", "MMR"]),
    "EAS": dict(label="East Asia", codes=["CHN", "MNG", "KOR", "PRK", "JPN", "TWN", "HKG", "MAC"]),
    "ISE": dict(label="Island SE Asia", codes=["PHL", "IDN", "MYS", "BRN", "TLS", "SGP"], bbox=(90, 134, -12, 22)),
    "PNG": dict(label="New Guinea", codes=["PNG", "IDN"], bbox=(134, 160, -12, 0)),
    "AUS": dict(label="Australia", codes=["AUS"]),
    "POL": dict(label="Polynesia", codes=["USA"], bbox=(-161, -154, 18, 23), inset=(4.2, (-150, 14)), buffer=0.30,
                note="(Hawai'i shown)"),
    "NAM": dict(label="North America", codes=["USA", "CAN"], bbox=(-180, -50, 24, 85)),
    "MES": dict(label="Mexico & C. America", codes=["MEX", "GTM", "BLZ", "HND", "SLV", "NIC", "CRI", "PAN"]),
    "CAR": dict(label="Caribbean", codes=["CUB", "HTI", "DOM", "JAM", "PRI"], inset=(2.0, (-39, 25)), buffer=0.25,
                geo=(-74, 20)),
    "AMZ": dict(label="Brazil & north", codes=["BRA", "COL", "VEN", "GUY", "SUR", "FRA"], bbox=(-80, -30, -35, 13)),
    "ADS": dict(label="Andes", codes=["PER", "BOL", "ECU"]),
    "SCN": dict(label="Southern Cone", codes=["ARG", "CHL", "URY", "PRY"], bbox=(-80, -50, -56, -17)),
}

# Who is there 200,000 years ago: vessel -> (component, fill). Neanderthals and Denisovans have no
# countries, so their sand starts inside the Eurasian vessels and is displaced when modern humans arrive.
INITIAL = {
    "KHS": ("KHS", 0.50),
    "CAH": ("CAH", 0.50),
    "EAF": ("EAF", 0.55),
    "EUR": ("NEA", 0.30),
    "LEV": ("NEA", 0.30),
    "IRN": ("NEA", 0.30),
    "STP": ("NEA", 0.22),
    "SIB": ("DEN", 0.18),
    "EAS": ("DEN", 0.25),
    "SEA": ("DEN", 0.25),
    "ISE": ("DEN", 0.25),
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
    ("SEA", "AND"): [(98, 9)],
    ("SEA", "ISE"): [(114, 8)],
    ("ISE", "PNG"): [(135, -5)],
    ("MES", "AMZ"): [(-80, 5), (-62, -3)],
    ("NAM", "MES"): [(-104, 33)],
    ("MES", "ADS"): [(-82, 3)],
    ("AMZ", "CAR"): [(-62, 9)],
}


# ---------------------------------------------------------------- events
def found(t, src, dst, n, only=None, keep=0.0, until=None):
    """A small group from `src` founds the population in `dst` (n grains travel; the rest are copies).
    `only`: the founders carry only these components (e.g. the out-of-Africa group).
    If dst already holds archaic sand, it is displaced by `until` (ybp) except for the share `keep`
    that mixes in (admixture)."""
    return dict(kind="found", t0=t, t1=t, src=src, dst=dst, n=n, only=only, keep=keep, until=until)


def flow(t0, t1, src, dst, f):
    """Gene flow: by t1 a fraction f of dst's gene pool comes from src (replacing, not adding)."""
    return dict(kind="flow", t0=t0, t1=t1, src=src, dst=dst, f=f)


def tint(t0, t1, jar, sources, target):
    """Isolation and drift: grains of `sources` colours in `jar` turn into a new component."""
    return dict(kind="tint", t0=t0, t1=t1, jar=jar, sources=sources, target=target)


def extinct(t0, t1, jar):
    return dict(kind="extinct", t0=t0, t1=t1, jar=jar)


EVENTS = [
    # --- deep African structure (contact between groups never stopped)
    found(165_000, "EAF", "WAF", 16),
    tint(155_000, 80_000, "WAF", ["EAF"], "WAF"),
    flow(120_000, 100_000, "EAF", "KHS", 0.05),
    flow(110_000, 95_000, "WAF", "CAH", 0.08),
    flow(100_000, 85_000, "CAH", "EAF", 0.04),

    # --- out of Africa, Neanderthals, the great dispersal
    found(65_000, "EAF", "LEV", 12, only=["EAF"], keep=0.026, until=52_000),
    found(50_000, "LEV", "SAS", 14),
    found(48_600, "SAS", "SEA", 12, keep=0.003, until=40_000),
    found(47_600, "SEA", "ISE", 12, keep=0.045, until=45_500),
    found(46_000, "LEV", "EUR", 14, until=40_000),
    found(44_800, "ISE", "PNG", 10),
    found(44_000, "PNG", "AUS", 10),
    found(45_000, "LEV", "IRN", 14, until=40_000),
    found(44_500, "SEA", "EAS", 12, keep=0.004, until=36_000),
    found(43_000, "LEV", "SIB", 12, until=36_000),
    extinct(42_000, 40_000, "STP"),

    # --- isolation: each region's sand drifts into its own colour
    tint(44_000, 20_000, "LEV", ["EAF"], "ANF"),
    tint(44_000, 22_000, "IRN", ["EAF", "ANF"], "IRN"),
    tint(44_000, 20_000, "EUR", ["EAF", "ANF"], "WHG"),
    tint(42_000, 24_000, "SIB", ["EAF", "ANF"], "ANE"),
    tint(47_000, 26_000, "SAS", ["EAF"], "AAS"),
    tint(46_000, 26_000, "SEA", ["EAF"], "AAS"),
    tint(44_000, 26_000, "EAS", ["EAF", "AAS"], "EAS"),
    tint(46_000, 26_000, "ISE", ["EAF", "AAS"], "AUS"),
    tint(45_000, 26_000, "PNG", ["EAF", "AAS"], "AUS"),
    tint(44_000, 26_000, "AUS", ["EAF", "AAS"], "AUS"),
    found(32_000, "SEA", "AND", 8),
    tint(32_000, 25_000, "AND", ["EAF"], "AAS"),

    # --- Ice Age North: Eastern hunter-gatherers, Ancient Paleo-Siberians, the peopling of the Americas
    found(28_000, "SIB", "STP", 12),
    tint(28_000, 22_000, "STP", ["EAF", "ANF"], "ANE"),
    flow(26_000, 21_000, "EAS", "SIB", 0.62),
    found(20_000, "SIB", "NAM", 9),
    flow(18_000, 14_500, "EUR", "STP", 0.40),
    found(19_000, "LEV", "NAF", 12),
    flow(17_000, 14_000, "WAF", "NAF", 0.35),
    tint(19_000, 13_000, "NAM", ["ANE", "EAS"], "NAT"),
    found(16_000, "NAM", "MES", 10),
    found(14_800, "MES", "ADS", 9),
    found(14_000, "MES", "AMZ", 9),
    found(13_500, "ADS", "SCN", 8),
    tint(13_500, 12_500, "SCN", ["ANE", "EAS"], "NAT"),
    tint(16_000, 12_500, "MES", ["ANE", "EAS"], "NAT"),
    tint(14_800, 12_500, "ADS", ["ANE", "EAS"], "NAT"),
    tint(14_000, 12_500, "AMZ", ["ANE", "EAS"], "NAT"),
    flow(11_000, 7_000, "EAS", "SIB", 0.45),

    # --- farming spreads, and mixes with the foragers it meets
    flow(9_500, 7_000, "LEV", "IRN", 0.22),
    flow(9_000, 5_000, "IRN", "SAS", 0.42),
    flow(8_500, 6_200, "LEV", "EUR", 0.80),
    found(7_800, "LEV", "SAR", 14),
    flow(7_200, 6_000, "EUR", "SAR", 0.18),
    flow(7_500, 5_500, "LEV", "NAF", 0.32),
    flow(8_000, 4_200, "IRN", "LEV", 0.30),
    found(6_500, "AMZ", "CAR", 8),
    flow(6_800, 5_400, "IRN", "STP", 0.45),

    # --- steppe herders, the Bantu expansion, Austronesian voyagers
    flow(5_000, 4_300, "STP", "EUR", 0.50),
    flow(5_000, 4_300, "STP", "SIB", 0.10),
    flow(5_000, 3_000, "WAF", "CAH", 0.78),
    flow(5_000, 3_500, "EAS", "SEA", 0.86),
    flow(4_500, 3_200, "EAS", "ISE", 0.78),
    flow(4_000, 3_000, "STP", "SAS", 0.16),
    flow(3_600, 2_600, "LEV", "EAF", 0.30),
    flow(3_500, 2_600, "ISE", "PNG", 0.08),
    found(3_000, "EAS", "POL", 10),
    flow(2_600, 1_600, "PNG", "POL", 0.24),
    flow(3_000, 1_000, "EAS", "SAS", 0.04),
    flow(2_600, 1_500, "WAF", "EAF", 0.22),
    flow(2_500, 700, "EAS", "STP", 0.18),
    flow(2_400, 1_800, "LEV", "SAR", 0.25),
    flow(2_000, 900, "WAF", "KHS", 0.68),
    flow(2_000, 1_600, "EAF", "KHS", 0.05),
    flow(2_000, 1_000, "SAS", "SEA", 0.05),
    found(1_500, "ISE", "MDG", 10),
    flow(1_300, 800, "WAF", "MDG", 0.60),
    flow(1_350, 950, "LEV", "NAF", 0.18),
    flow(1_000, 600, "LEV", "MDG", 0.04),
    flow(1_000, 800, "EAS", "LEV", 0.05),
    flow(900, 700, "ADS", "POL", 0.012),
    flow(1_000, 400, "WAF", "NAF", 0.08),
    flow(3_500, 2_600, "STP", "IRN", 0.08),
    flow(1_500, 600, "EUR", "SAR", 0.12),

    # --- 1500 CE onward: ships, colonization, the slave trade, modern migration
    flow(525, 430, "EUR", "CAR", 0.75),
    flow(505, 150, "EUR", "MES", 0.36),
    flow(490, 80, "EUR", "AMZ", 0.72),
    flow(490, 120, "EUR", "ADS", 0.18),
    flow(480, 60, "EUR", "SCN", 0.70),
    flow(450, 200, "WAF", "SCN", 0.02),
    flow(475, 165, "WAF", "AMZ", 0.50),
    flow(470, 170, "WAF", "MES", 0.05),
    flow(450, 170, "WAF", "ADS", 0.03),
    flow(430, 165, "WAF", "CAR", 0.60),
    flow(405, 105, "EUR", "NAM", 0.92),
    flow(400, 160, "WAF", "NAM", 0.36),
    flow(373, 120, "EUR", "KHS", 0.10),
    flow(380, 80, "STP", "SIB", 0.70),
    flow(237, 70, "EUR", "AUS", 0.95),
    flow(220, 70, "EUR", "POL", 0.12),
    flow(200, 60, "EAS", "SEA", 0.05),
    flow(165, 110, "SAS", "KHS", 0.025),
    flow(70, 5, "MES", "NAM", 0.10),
    flow(60, 5, "EAS", "NAM", 0.05),
    flow(55, 5, "SAS", "NAM", 0.015),
    flow(70, 5, "NAF", "EUR", 0.03),
    flow(65, 5, "LEV", "EUR", 0.03),
    flow(60, 5, "WAF", "EUR", 0.02),
    flow(60, 5, "SAS", "EUR", 0.02),
    flow(55, 5, "EAS", "AUS", 0.11),
    flow(40, 5, "SAS", "AUS", 0.05),
]

# ---------------------------------------------------------------- the clock: (video seconds, ybp)
# between keys, log(ybp + 200) is linear in video time
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
DURATION = TODAY_S + 18.0      # hold on today for the closing captions


# ---------------------------------------------------------------- captions: (ybp, text) or (video s, text)
CAPTIONS = [
    ("s", 6.5, "200,000 years ago, our species lives only in Africa, already in deep-rooted groups. "
              "Neanderthals (white sand) and Denisovans (pale blue) live in Eurasia."),
    ("s", 12.2, "Each glass vessel is a population, shaped like its land. Each colour of sand is a strand of ancestry."),
    ("s", 16.6, "Separated for long enough, a group's sand drifts into a colour of its own."),
    (65_000, "About 65,000 years ago, a small band crosses out of Africa."),
    (55_000, "They mix with Neanderthals. Everyone with ancestors outside Africa still carries "
             "about 2% Neanderthal DNA (the white grains)."),
    ("s", 43.4, "Within a few thousand years they reach South Asia, Europe, East Asia and, by boat, Australia and New Guinea."),
    ("s", 47.8, "Near Wallacea they mix with Denisovans: Papuans and Aboriginal Australians carry 3-5% "
             "Denisovan DNA."),
    ("s", 53.6, "The Neanderthals and Denisovans disappear. A little of their sand lives on in us."),
    (33_000, "Tens of thousands of years apart, each region's sand drifts into its own colour."),
    (25_000, "In Ice Age Siberia, East Asian and Ancient North Eurasian ancestry mix."),
    (20_200, "A few founders cross Beringia. A handful of grains seeds two whole continents: "
             "the founder effect."),
    (15_500, "Cut off for 15,000 years, the First Peoples of the Americas become genetically distinct."),
    (9_000, "Farming begins in the Fertile Crescent. Farmers spread and mix with the foragers they meet."),
    (7_400, "Sardinia is settled by farmers, and keeps much of their ancestry to this day."),
    (5_000, "Herders from the steppe sweep west into Europe and later south into India."),
    ("s", 103.4, "The Bantu expansion carries West-Central African ancestry across half a continent."),
    ("s", 108.2, "Austronesian sailors leave Taiwan and island-hop across two oceans..."),
    (1_300, "...to Polynesia and to Madagascar, where Asian and African ancestry meet."),
    (520, "1500 CE: ships link every continent. Colonization, and the forced migration of 12 million "
          "enslaved Africans, remake the Americas."),
    (230, "Settlers transform the makeup of Australia, Siberia and southern Africa."),
    (70, "Modern migration keeps stirring the sand."),
    ("s", TODAY_S + 1.0, "Today almost every vessel holds sand from many places. Islands of ancestry endure: "
                 "the Andaman Islanders, Papuans, Sardinians, the Andes."),
    ("s", TODAY_S + 9.0, "And underneath, it was always the same sand. Every person alive descends from "
                 "those first African grains."),
]
