"""Modern fortune series, 2002-2026: Forbes annual lists (2002-2023) and Forbes real-time daily data
(late 2019 onwards), merged per person.

    series = load(dl_dir)    # {key: dict(name, points=[(decimal_year, nominal $M)], source, industry, country, ...)}

Daily values are smoothed with a 7-day median so one-day glitches in the real-time feed (a stock that spikes
for a day, a mis-keyed figure) do not become peaks. Names are matched across sources with ALIASES; a few
records known to be feed errors are dropped (DROP).
"""
import csv, glob, os, pickle, re, unicodedata
import numpy as np

ALIASES = {   # variant spelling -> canonical name
    "lawrence ellison": "larry ellison", "kashing li": "li kashing", "shau kee lee": "lee shau kee",
    "huateng ma": "ma huateng", "colin zheng huang": "colin huang", "william lei ding": "william ding",
    "mackenzie bezos": "mackenzie scott", "zeng robin": "robin zeng", "s robson walton": "rob walton",
    "jim c walton": "jim walton", "john t walton": "john walton", "christy walton": "christy walton",
    "alice l walton": "alice walton", "kwok thomas raymond": "thomas raymond kwok",
    "thomas raymond kwok": "thomas raymond kwok", "kwok family": "thomas raymond kwok",
    "lakshmi n mittal": "lakshmi mittal", "mukesh d ambani": "mukesh ambani", "carlos slim helu": "carlos slim helu",
    "jacqueline badger mars": "jacqueline mars", "john franklyn mars": "john mars",
    "francoise bettencourt meyers": "francoise bettencourt meyers", "gautam adani": "gautam adani",
    "prince alwaleed bin talal alsaud": "alwaleed bin talal", "alwaleed bin talal alsaud": "alwaleed bin talal",
    "karl albrecht jr": "karl albrecht jr", "beate heister karl albrecht jr": "karl albrecht jr",
    "sergey brin": "sergey brin", "steven ballmer": "steve ballmer", "michael bloomberg": "michael bloomberg",
}
DROP = {"rtb:sergey-jr-brin"}          # a duplicate of Sergey Brin in the real-time feed


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("& family", "").replace("&", " ").replace("-", "")
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def _median7(v):
    if len(v) < 7:
        return v
    p = np.pad(v, 3, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 7), axis=1)


def load(dl):
    S = {}

    def person(key, name):
        return S.setdefault(key, dict(name=name, annual={}, daily=None, industry=set(), country=None, deceased=False,
                                      sources=set(), family=False))

    for f in sorted(glob.glob(os.path.join(dl, "b*.csv"))) + sorted(glob.glob(os.path.join(dl, "u*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8", errors="ignore")):
            try:
                nw = float((r["net_worth"] or "").split()[0]) * 1000.0
                y = int(r["year"]); m = int(r["month"] or 3)
            except Exception:
                continue
            if y < 2002:            # the early years of this dataset are incomplete; 1987-2001 come from research
                continue
            p = person(norm(r["full_name"]), r["full_name"].replace(" & family", ""))
            p["annual"][y + (m - 0.5) / 12] = nw
            if r.get("business_category"):
                p["industry"].add(r["business_category"])
            p["country"] = p["country"] or r.get("country_of_citizenship")
            p["family"] |= "& family" in r["full_name"]
            p["sources"].add("Forbes World's Billionaires (annual)")
    R = pickle.load(open(os.path.join(dl, "rtb_compact.pkl"), "rb"))
    for uri, d in R.items():
        if "rtb:" + uri in DROP or not len(d["dates"]):
            continue
        info = d["info"]
        p = person(norm(info.get("name") or uri), (info.get("name") or uri).replace(" & family", ""))
        p.setdefault("_raw", []).append((str(d["dates"][-1]), d["dates"], d["nw"].astype(float)))
        p["industry"].update(info.get("industry") or [])
        p["country"] = p["country"] or (info.get("citizenship") or "").upper()
        p["deceased"] |= bool(info.get("deceased"))
        p["family"] |= bool(info.get("family"))
        p["rtb"] = uri
        p["bio"] = info.get("bio")
        p["source_of_wealth"] = info.get("source")
        p["sources"].add("Forbes Real-Time Billionaires (daily, via komed3/rtb-api)")
    for p in S.values():          # a person's profiles, joined end to end (Forbes renames profiles over time);
        raw = p.pop("_raw", None)  # where two overlap, the more recently updated one wins
        if not raw:
            continue
        val = {}
        for _, dates, nw in sorted(raw, key=lambda r: r[0]):
            val.update(zip((str(x) for x in dates), nw))
        dd = sorted(val)
        t = np.array([int(x[:4]) + (int(x[5:7]) - 1) / 12 + (int(x[8:10]) - 1) / 365 for x in dd])
        p["daily"] = (t, _median7(np.array([val[x] for x in dd])))
    return S
