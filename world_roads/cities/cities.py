"""48 cities for the street race. Each city = every public street within RADIUS_KM of its centre."""
RADIUS_KM = 15.0

# name, (lon, lat) of the centre (a conventional downtown landmark)
CITIES = [
    ("New York", (-73.9855, 40.7580)), ("Los Angeles", (-118.2427, 34.0537)), ("Chicago", (-87.6298, 41.8781)),
    ("Houston", (-95.3698, 29.7604)), ("Phoenix", (-112.0740, 33.4484)), ("San Francisco", (-122.4193, 37.7793)),
    ("Miami", (-80.1918, 25.7617)), ("Toronto", (-79.3832, 43.6532)), ("Montreal", (-73.5673, 45.5017)),
    ("Vancouver", (-123.1207, 49.2827)), ("Calgary", (-114.0631, 51.0461)), ("Edmonton", (-113.4938, 53.5461)),
    ("Mexico City", (-99.1332, 19.4326)),
    ("São Paulo", (-46.6333, -23.5505)), ("Buenos Aires", (-58.3816, -34.6037)), ("Rio de Janeiro", (-43.1729, -22.9068)),
    ("Bogotá", (-74.0758, 4.5981)), ("Lima", (-77.0428, -12.0464)),
    ("London", (-0.1276, 51.5073)), ("Paris", (2.3488, 48.8534)), ("Berlin", (13.4050, 52.5200)),
    ("Madrid", (-3.7038, 40.4168)), ("Rome", (12.4829, 41.8933)), ("Moscow", (37.6208, 55.7539)),
    ("Istanbul", (28.9700, 41.0170)), ("Amsterdam", (4.8926, 52.3731)), ("Barcelona", (2.1700, 41.3870)),
    ("Vienna", (16.3731, 48.2085)), ("Athens", (23.7348, 37.9755)),
    ("Cairo", (31.2357, 30.0444)), ("Lagos", (3.3947, 6.4541)), ("Johannesburg", (28.0473, -26.2041)),
    ("Nairobi", (36.8172, -1.2864)),
    ("Tokyo", (139.7671, 35.6812)), ("Seoul", (126.9779, 37.5663)), ("Beijing", (116.3975, 39.9087)),
    ("Shanghai", (121.4737, 31.2304)), ("Hong Kong", (114.1580, 22.2819)), ("Singapore", (103.8520, 1.2931)),
    ("Bangkok", (100.5018, 13.7563)), ("Jakarta", (106.8272, -6.1754)), ("Manila", (120.9787, 14.5826)),
    ("Mumbai", (72.8478, 19.0178)), ("Delhi", (77.2167, 28.6315)), ("Dubai", (55.2744, 25.1972)),
    ("Sydney", (151.2065, -33.8731)), ("Melbourne", (144.9631, -37.8136)), ("Auckland", (174.7633, -36.8485)),
]


def slug(name):
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return s.lower().replace(" ", "_")
