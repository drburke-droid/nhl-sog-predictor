"""Choose the 100 largest private fortunes, 1900-2026, by peak share of the US economy (in today's money).

    python3 select100.py DL_DIR RESEARCH_DIR      ->  data/the100.csv

Rules:
- one person (or a couple/siblings Forbes lists as one fortune); clans ("du Pont family") are left out
- only the fortune's life from 1900 counts; a fortune already made before 1900 enters at its 1900 value
  (same share of the economy as its last earlier estimate); people who died before 1900 are out
- monarchs and rulers are not in the data (private fortunes only)
- lifetime figures that add gifts back (Klepper & Gunther) are not used as values (see series.py)
"""
import csv, os, sys
import numpy as np
import series, modern

DISPLAY = {"larry ellison": "Larry Ellison", "rob walton": "Rob Walton", "john walton": "John Walton", "helen walton": "Helen Walton", "ma huateng": "Ma Huateng (Pony Ma)",
           "colin huang": "Colin Huang", "carlos slim helu": "Carlos Slim", "li kashing": "Li Ka-shing",
           "lee shau kee": "Lee Shau Kee", "sir john ellerman st baronet": "John Ellerman",
           "alwaleed bin talal": "Alwaleed bin Talal", "francoise bettencourt meyers": "Françoise Bettencourt Meyers",
           "francois pinault": "François Pinault", "german larrea mota velasco": "Germán Larrea",
           "beate heister karl albrecht jr": "Beate Heister & Karl Albrecht Jr.", "e h harriman": "E. H. Harriman",
           "john t walton": "John Walton", "henry clay frick": "Henry Clay Frick", "andrew w mellon": "Andrew Mellon",
           "richard b mellon": "Richard B. Mellon", "j paul getty": "J. Paul Getty", "j p morgan": "J. P. Morgan",
           "william a clark": "William A. Clark", "j ogden armour": "J. Ogden Armour", "h l hunt": "H. L. Hunt",
           "george f baker": "George F. Baker", "henry h rogers": "Henry H. Rogers", "alfred g vanderbilt": "Alfred G. Vanderbilt",
           "lamon v harkness": "Lamon V. Harkness", "charles w harkness": "Charles W. Harkness", "edward s harkness": "Edward S. Harkness",
           "anna m harkness": "Anna M. Harkness", "joseph p kennedy": "Joseph P. Kennedy", "john d rockefeller": "John D. Rockefeller",
           "elon musk": "Elon Musk"}
EXTRA_ALIAS = {"prince alwaleed bin talal": "alwaleed bin talal", "sir john ellerman st baronet": "sir john ellerman st baronet"}
CLANS = ("family", "families")

ISLANDS = {   # industry island -> keywords matched against Forbes categories / research industry (first match wins)
    "Finance & crypto": ["finance", "investments", "investing", "banking", "lending", "crypto"],
    "Oil & energy": ["oil", "energy", "petroleum"],
    "Steel, rail & mining": ["steel", "rail", "metals", "mining", "copper", "coke", "shipping", "aluminum"],
    "Cars & rockets": ["automotive", "autos"],
    "Media & telecom": ["telecom", "media"],
    "Technology": ["technology", "software", "internet"],
    "Fashion & retail": ["fashion", "retail"],
    "Food & drink": ["food", "beverage", "meat"],
    "Property": ["real estate", "construction"],
    "Conglomerates": ["diversified", "conglomerate", "logistics", "manufacturing", "healthcare", "gaming", "gambling", "service"],
}
OVERRIDE = {  # where the keywords mislead or the source gives no industry
    "helen walton": "Fashion & retail", "john walton": "Fashion & retail", "taikichiro mori": "Property",
    "yoshiaki tsutsumi": "Property", "robin zeng": "Cars & rockets", "mukesh ambani": "Conglomerates",
    "andrew w mellon": "Finance & crypto", "richard b mellon": "Finance & crypto", "russell sage": "Finance & crypto",
    "michael bloomberg": "Media & telecom", "masayoshi son": "Technology", "paul allen": "Technology",
    "anil ambani": "Conglomerates", "zhang yiming": "Technology", "jeff bezos": "Technology", "philip anschutz": "Oil & energy",
    "joseph p kennedy": "Finance & crypto", "alwaleed bin talal": "Finance & crypto", "john ellerman": "Steel, rail & mining",
    "sir john ellerman st baronet": "Steel, rail & mining", "henry clay frick": "Steel, rail & mining",
    "klausmichael kuehne": "Steel, rail & mining", "wang wei": "Conglomerates",
}


def island(industries):
    s = " ".join(industries).lower()
    for name, keys in ISLANDS.items():
        if any(k in s for k in keys):
            return name
    return "Conglomerates"


def initials(name):
    parts = [p for p in name.replace("&", " ").replace("(", " ").split() if p[0].isupper() and p.rstrip(".") not in ("Jr", "Sr", "Sir", "II", "III", "IV")]
    if not parts:
        return name[:2].upper()
    ini = [p[0] for p in parts[:3]] if len(parts) <= 3 else [parts[0][0], parts[-1][0]]
    return "".join(ini)


def main(dl, research):
    P = series.build(dl, research)
    for a, b in EXTRA_ALIAS.items():
        if a in P and b in P and a != b:          # fold duplicate keys
            pa, pb = P.pop(a), P[b]
            t = np.r_[pa["t"], pb["t"]]; o = np.argsort(t)
            for k in ("t", "nominal", "today"):
                pb[k] = np.r_[pa[k], pb[k]][o]
            pb["src"] = list(np.array(pa["src"] + pb["src"], dtype=object)[o])
    gil = {modern.norm(r["name"]): r for r in csv.DictReader(open(os.path.join(research, "gilded", "people.csv")))}
    rows = []
    for k, p in P.items():
        if any(c in k.split() for c in CLANS) and k not in ("beate heister karl albrecht jr",):
            continue
        g = gil.get(k, {})
        died = int(g["died"]) if g.get("died", "").isdigit() else None
        if died and died < 1900:
            continue
        t, v = p["t"], p["today"]
        inwin = t >= 1900
        cand = list(v[inwin])
        if (~inwin).any():
            cand.append(v[~inwin][-1])          # entering 1900 at its last earlier share of the economy
        if not cand:
            continue
        i = int(np.argmax(v[inwin])) if inwin.any() else None
        peak = max(cand)
        when = float(t[inwin][i]) if inwin.any() and v[inwin][i] >= peak else 1900.0
        inds = list(p["industry"]) + ([g["industry"]] if g.get("industry") else [])
        name = DISPLAY.get(k, p["name"].replace(" & family", ""))
        isl = OVERRIDE.get(k) or island(inds)
        rows.append(dict(key=k, name=name, initials=initials(name), peak_today_busd=round(peak / 1000, 1), peak_year=round(when, 2),
                         island=isl, industry="; ".join(sorted(set(inds)))[:60], country=p.get("country") or g.get("nationality", ""),
                         died=died or "", points=len(t), first=round(float(t[0]), 1), last=round(float(t[-1]), 1),
                         kinds="; ".join(sorted(set(s.split(":")[0][:40] for s in p["src"])))[:120],
                         fate=g.get("fate_summary", "")[:200]))
    rows.sort(key=lambda r: -r["peak_today_busd"])
    top = rows[:100]
    os.makedirs("data", exist_ok=True)
    with open("data/the100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys())); w.writeheader(); w.writerows(top)
    with open("data/next20.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys())); w.writeheader(); w.writerows(rows[100:120])
    return top, rows[100:120]


if __name__ == "__main__":
    top, nxt = main(sys.argv[1], sys.argv[2])
    for i, r in enumerate(top):
        print(f"{i + 1:3d} {r['initials']:4s} {r['name'][:30]:30s} ${r['peak_today_busd']:7.1f}B {r['peak_year']:7.1f}  {r['island'][:20]:20s} pts {r['points']:3d} died {r['died']}")
    print("next:", [(r["name"], r["peak_today_busd"]) for r in nxt[:10]])
