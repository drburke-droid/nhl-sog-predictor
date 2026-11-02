"""Download (and cache) the raw pages everything else is built from.

    python fetch.py olympics     # Olympedia: every Summer Games athletics result page for the events we track
    python fetch.py records      # Wikipedia: the world-record progression pages (parsed wikitext + HTML)

Pages are cached under cache/ and never re-downloaded. Olympedia is fetched one page at a time with a
pause between requests.
"""
import os, sys, time, json, re, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
UA = "RecordsVideo/1.0 (https://github.com/drburke-droid/nhl-sog-predictor; research for a data video) python-urllib"

# Olympedia event names (lower case, without the ", Men"/", Women" suffix) -> our event key
OLY_EVENTS = {
    "100 metres": "100m", "200 metres": "200m", "400 metres": "400m", "800 metres": "800m",
    "1,500 metres": "1500m", "5,000 metres": "5000m", "10,000 metres": "10000m", "marathon": "marathon",
    "110 metres hurdles": "110mH", "100 metres hurdles": "100mH", "80 metres hurdles": "80mH",
    "400 metres hurdles": "400mH", "3,000 metres steeplechase": "3000mSC",
    "high jump": "HJ", "pole vault": "PV", "long jump": "LJ", "triple jump": "TJ",
    "shot put": "SP", "discus throw": "DT", "javelin throw": "JT", "hammer throw": "HT",
    "100 / 80 metres hurdles": "W_sprint_hurdles", "100 / 80 metres hurdles (100 m)": "W_sprint_hurdles",
}
EDITIONS = {1896: 1, 1900: 2, 1904: 3, 1908: 5, 1912: 6, 1920: 7, 1924: 8, 1928: 9, 1932: 10, 1936: 11,
            1948: 12, 1952: 13, 1956: 14, 1960: 15, 1964: 16, 1968: 17, 1972: 18, 1976: 19, 1980: 20,
            1984: 21, 1988: 22, 1992: 23, 1996: 24, 2000: 25, 2004: 26, 2008: 53, 2012: 54, 2016: 59,
            2020: 61, 2024: 63}


def get(url, path, pause=0.7):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, encoding="utf-8").read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(7):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:                          # alltime-athletics pages are Latin-1
                txt = raw.decode("latin-1")
            break
        except Exception as e:                                   # noqa: BLE001
            if getattr(e, "code", None) == 404:
                raise RuntimeError(f"404: {url}") from e
            wait = 2 ** (attempt + 1)
            ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if getattr(e, "code", None) == 429:
                wait = max(wait * 5, int(ra) if ra and str(ra).isdigit() else 0)
            print("  retry in", wait, "s:", url[:90], e, flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    time.sleep(pause)
    return txt


def olympics():
    from bs4 import BeautifulSoup
    index = []
    for year, ed in EDITIONS.items():
        html = get(f"https://www.olympedia.org/editions/{ed}/sports/ATH",
                   os.path.join(CACHE, "olympedia", f"edition_{year}.html"))
        s = BeautifulSoup(html, "lxml")
        seen = set()
        for a in s.select('a[href^="/results/"]'):
            name = a.get_text(strip=True)
            m = re.match(r"(.+),\s*(Men|Women)$", name)
            if not m:
                continue
            ev, sex = m.group(1).strip().lower(), m.group(2)
            if ev not in OLY_EVENTS:
                continue
            rid = a["href"].split("/")[-1]
            if rid in seen:
                continue
            seen.add(rid)
            key = OLY_EVENTS[ev]
            if key == "W_sprint_hurdles":                      # Olympedia's name for the women's 80 m / 100 m hurdles
                key = "100mH" if year >= 1972 else "80mH"
            get(f"https://www.olympedia.org/results/{rid}", os.path.join(CACHE, "olympedia", f"result_{rid}.html"))
            index.append({"year": year, "sex": sex[0], "event": key, "name": name, "result_id": rid})
            print(year, sex, key, rid, flush=True)
    with open(os.path.join(CACHE, "olympedia", "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print("events:", len(index))


# Wikipedia world-record progression pages
WR_PAGES = {
    ("M", "100m"): "Men's 100 metres world record progression",
    ("W", "100m"): "Women's 100 metres world record progression",
    ("M", "200m"): "Men's 200 metres world record progression",
    ("W", "200m"): "Women's 200 metres world record progression",
    ("M", "400m"): "Men's 400 metres world record progression",
    ("W", "400m"): "Women's 400 metres world record progression",
    ("M", "800m"): "Men's 800 metres world record progression",
    ("W", "800m"): "Women's 800 metres world record progression",
    ("M", "1500m"): "Men's 1500 metres world record progression",
    ("W", "1500m"): "Women's 1500 metres world record progression",
    ("M", "5000m"): "Men's 5000 metres world record progression",
    ("W", "5000m"): "Women's 5000 metres world record progression",
    ("M", "10000m"): "Men's 10,000 metres world record progression",
    ("W", "10000m"): "Women's 10,000 metres world record progression",
    ("X", "marathon"): "Marathon world record progression",
    ("M", "110mH"): "Men's 110 metres hurdles world record progression",
    ("W", "100mH"): "Women's 100 metres hurdles world record progression",
    ("M", "400mH"): "Men's 400 metres hurdles world record progression",
    ("W", "400mH"): "Women's 400 metres hurdles world record progression",
    ("M", "3000mSC"): "Men's 3000 metres steeplechase world record progression",
    ("W", "3000mSC"): "Women's 3000 metres steeplechase world record progression",
    ("M", "HJ"): "Men's high jump world record progression",
    ("W", "HJ"): "Women's high jump world record progression",
    ("M", "PV"): "Men's pole vault world record progression",
    ("W", "PV"): "Women's pole vault world record progression",
    ("M", "LJ"): "Men's long jump world record progression",
    ("W", "LJ"): "Women's long jump world record progression",
    ("M", "TJ"): "Men's triple jump world record progression",
    ("W", "TJ"): "Women's triple jump world record progression",
    ("M", "SP"): "Men's shot put world record progression",
    ("W", "SP"): "Women's shot put world record progression",
    ("M", "DT"): "Men's discus throw world record progression",
    ("W", "DT"): "Women's discus throw world record progression",
    ("M", "JT"): "Men's javelin throw world record progression",
    ("W", "JT"): "Women's javelin throw world record progression",
    ("M", "HT"): "Men's hammer throw world record progression",
    ("W", "HT"): "Women's hammer throw world record progression",
}


def records():
    for (sex, ev), title in WR_PAGES.items():
        slug = f"{sex}_{ev}"
        q = urllib.parse.urlencode({"action": "parse", "page": title, "format": "json", "prop": "text|wikitext",
                                    "redirects": 1, "formatversion": 2})
        txt = get(f"https://en.wikipedia.org/w/api.php?{q}", os.path.join(CACHE, "wikipedia", f"{slug}.json"), pause=4)
        j = json.loads(txt)
        if "error" in j:
            print("MISSING", slug, title, j["error"].get("info"))
        else:
            print(slug, j["parse"]["title"], len(j["parse"]["text"]))


# Peter Larsson's all-time lists (alltime-athletics.com): every legal performance down to a deep cut-off,
# with wind, altitude flag, venue and date. "ok" pages are legal marks; "no" pages are wind-aided (or
# otherwise not legal) marks, fetched only where wind matters.
ALLTIME = {
    ("M", "100m"): "m_100ok", ("M", "200m"): "m_200ok", ("M", "400m"): "m_400ok", ("M", "800m"): "m_800ok",
    ("M", "1500m"): "m_1500ok", ("M", "5000m"): "m_5000ok", ("M", "10000m"): "m_10kok",
    ("M", "110mH"): "m_110hok", ("M", "400mH"): "m_400hok", ("M", "HJ"): "mhighok", ("M", "PV"): "mpoleok",
    ("M", "LJ"): "mlongok", ("M", "TJ"): "mtripok", ("M", "SP"): "mshotok", ("M", "DT"): "mdiscok",
    ("M", "JT"): "mjaveok", ("M", "JT_old"): "mjaveoldok", ("M", "HT"): "mhammok",
    ("W", "100m"): "w_100ok", ("W", "200m"): "w_200ok", ("W", "400m"): "w_400ok", ("W", "800m"): "w_800ok",
    ("W", "1500m"): "w_1500ok", ("W", "5000m"): "w_5000ok", ("W", "10000m"): "w_10kok",
    ("W", "100mH"): "w_100hok", ("W", "400mH"): "w_400hok", ("W", "HJ"): "whighok", ("W", "PV"): "wpoleok",
    ("W", "LJ"): "wlongok", ("W", "TJ"): "wtripleok", ("W", "SP"): "wshotok", ("W", "DT"): "wdiscok",
    ("W", "JT"): "wjaveok", ("W", "JT_old"): "wjaveoldok", ("W", "HT"): "whammok",
    # wind-aided lists
    ("M", "100m_w"): "m100mno", ("M", "200m_w"): "m_200no", ("M", "110mH_w"): "m_110hno",
    ("M", "LJ_w"): "mlongno", ("M", "TJ_w"): "mtripno",
    ("W", "100m_w"): "w_100no", ("W", "200m_w"): "w_200no", ("W", "100mH_w"): "w_100hno",
    ("W", "LJ_w"): "wlongno", ("W", "TJ_w"): "wtripleno",
}


def alltime():
    for (sex, ev), page in ALLTIME.items():
        txt = get(f"http://www.alltime-athletics.com/{page}.htm", os.path.join(CACHE, "alltime", f"{sex}_{ev}.htm"),
                  pause=1.5)
        print(sex, ev, page, len(txt), flush=True)


if __name__ == "__main__":
    {"olympics": olympics, "records": records, "alltime": alltime}[sys.argv[1]]()
