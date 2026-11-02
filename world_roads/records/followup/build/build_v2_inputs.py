"""Build the follow-up audit's parallel input set (data/v2/) from the project's local source cache.

    python followup/build/build_v2_inputs.py        # from world_roads/records; needs cache/ (not shipped)

The original inputs (data/original/) are never modified. This script re-derives the same season lists with
the original pipeline (bigfour.categories()), then adds, without changing any original value:

* athlete_id   a stable identity where the source has one:
                 MLB/NBA/NHL   Reference player IDs from the leader-board links (e.g. chambwi01)
                 NFL           Pro Football Archives player links (to 1998), nflverse player_id (1999 on)
                 streaks       Retrosheet player IDs
                 swimming      name + nation (World Aquatics personIds include duplicate records of one person;
                               see the corrections log)
                 athletics     name + date of birth (all-time lists and World Athletics season lists)
               and "name:<normalised name>" where no ID could be matched (id_method says which).
* corrections  applied only in the v2 ranks/columns and listed in data/v2/corrections_log.csv:
                 C1 athletics names carrying a wind reading written with a decimal comma ("+1,7 Simon Ehammer")
                 C2 athletics marks dated after the November 2026 snapshot (source typos: 2027, 2109, 2202) are
                    quarantined, not re-dated
                 C3 swimming: one swimmer listed twice in a season under two World Aquatics personIds
                 C4 names compared with a transliteration-aware key (ł→l, ß→ss, ø→o ...) in name mode
* deeper lists the top 40 per season (the original export kept 10) so that removing a duplicate does not
               leave a hole, and so that rank-based tail diagnostics can use more than ten entries.

Output columns (season_lists_v2.csv): category, season, orig_rank (1-based rank in the original list, blank
if beyond it), v2_rank (rank after corrections, blank if dropped), athlete_id, id_method, name_raw,
name_clean, name_display, raw_value, performance, games_per_team, short_season, correction (codes).
"""
import os, sys, re, csv, json, glob, unicodedata, difflib
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import bigfour as B                                        # noqa: E402
from bs4 import BeautifulSoup                              # noqa: E402

OUT = os.path.join(ROOT, "followup", "data", "v2")
CACHE = os.path.join(ROOT, "cache")
SNAPSHOT_YEAR = 2026
DEPTH = 40
TRANS = str.maketrans({"ł": "l", "Ł": "L", "ß": "ss", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "æ": "ae",
                       "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D"})
WIND_PREFIX = re.compile(r"^[+\-±]?\d{1,2}[.,]\d{1,2}\s+")


def norm2(name):
    """transliteration-aware key: what bigfour.norm does, after mapping letters NFKD cannot decompose"""
    s = unicodedata.normalize("NFKD", name.translate(TRANS))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z ]", "", s.lower()).split())


# ---------------------------------------------------------------- identities
def pid_of(a):
    if a is None:
        return None
    if a.get("data-entry-id"):
        return a["data-entry-id"]
    m = re.search(r"/players/\w/([\w.]+?)\.s?html", a.get("href", ""))
    return m.group(1) if m else None


def sref_ids(div):
    """(name, value, pid) with the same name rule as bigfour.sref_entries"""
    out = []
    for e in div.select("div > div"):
        w, v = e.find(class_="who"), e.find(class_="value")
        if not w or not v:
            continue
        a = w.find("a")
        name = (a.get_text(strip=True) if a else w.get_text(" ", strip=True)).replace("*", "").strip()
        if a and a.get("title") and len(re.sub(r"\s*\([A-Z0-9]{2,4}\)$", "", a["title"])) > len(name):
            name = re.sub(r"\s*\([A-Z0-9]{2,4}\)$", "", a["title"]).strip()
        try:
            out.append((name, float(v.get_text(strip=True).replace(",", "")), pid_of(a)))
        except ValueError:
            pass
    return out


def league_id_index():
    """(sport, year, normalised name) -> {pid} from every cached leader board"""
    idx = defaultdict(set)

    def add(sport, y, ents):
        for n, _, p in ents:
            if p:
                idx[(sport, y, B.norm(n))].add(p)
    for f in glob.glob(os.path.join(CACHE, "sref", "mlb_*.html")):
        base = os.path.basename(f)
        s = BeautifulSoup(open(f, encoding="utf-8").read(), "lxml")
        m = re.match(r"mlb_(AL|NL)_(\d{4})_", base)
        if m:
            y = int(m.group(2))
            for div in s.select('div[id^="leaderboard_"]'):
                add("MLB", y, sref_ids(div))
        else:
            for div in s.select('div[id^="leaders_"]'):
                ym = re.search(r"_y(\d{4})$", div["id"])
                if ym:
                    add("MLB", int(ym.group(1)), sref_ids(div))
    for sport, pat in (("NBA", "nba_*.html"), ("NHL", "nhl_*.html")):
        for f in glob.glob(os.path.join(CACHE, "sref", pat)):
            ym = re.search(r"_(\d{4})\.html$", f)
            if not ym:
                continue
            s = BeautifulSoup(open(f, encoding="utf-8").read(), "lxml")
            for div in s.select('div[id^="leaders_"]'):
                add(sport, int(ym.group(1)), sref_ids(div))
    return idx


def nfl_id_index():
    idx = defaultdict(set)
    for tp in glob.glob(os.path.join(CACHE, "pfa", "[12][0-9][0-9][0-9][an]fl[a-z]*.html")):
        base = os.path.basename(tp)[:-5]
        if not re.fullmatch(r"\d{4}[an]fl[a-z]+", base):
            continue
        y = int(base[:4])
        for a in BeautifulSoup(open(tp, encoding="utf-8").read(), "lxml").select("table a[href]"):
            if "/" not in a["href"] and a["href"].endswith(".html"):
                pass
            nm = a.get_text(" ", strip=True)
            if nm:
                idx[(y, B.norm(nm))].add("pfa:" + a["href"].split("/")[-1].replace(".html", ""))
    for f in ("offense.csv", "defense.csv", "reg_2025.csv"):
        for r in csv.DictReader(open(os.path.join(CACHE, "nflverse", f))):
            nm = r.get("player_display_name") or r.get("player_name") or ""
            if r.get("player_id") and r.get("season", "").isdigit():
                idx[(int(r["season"]), B.norm(nm))].add("nflv:" + r["player_id"])
    return idx


def streak_id_index():
    """Retrosheet IDs, mapped to Baseball-Reference IDs through the Chadwick register (so a hitter's streak
    and his other categories share one identity)"""
    names, bbref = {}, {}
    for f in glob.glob(os.path.join(CACHE, "register", "people-*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["key_retro"]:
                names[r["key_retro"]] = f"{r['name_first']} {r['name_last']}".strip()
                if r.get("key_bbref"):
                    bbref[r["key_retro"]] = r["key_bbref"]
    idx = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(CACHE, "streaks.csv"))):
        k = ("ref:" + bbref[r["player"]]) if r["player"] in bbref else ("retro:" + r["player"])
        idx[(int(r["year"]), B.norm(names.get(r["player"], r["player"])))].add(k)
    return idx


def ath_dob_index():
    """(sex, event, normalised clean name) -> {dob as dd.mm.yy}, from the all-time lists and WA season lists;
    also (sex, event, year, name, mark) -> {dob} to resolve two people who share a name"""
    idx = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(CACHE, "alltime_parsed.csv"), encoding="utf-8")):
        nm = WIND_PREFIX.sub("", r["athlete"])
        if r["dob"]:
            idx[(r["sex"], r["event"], norm2(nm))].add((r["dob"].strip(), r["nat"].strip()))
            idx[(r["sex"], r["event"], int(r["date"][:4]), norm2(nm), round(float(r["mark"]), 2))].add((r["dob"].strip(), r["nat"].strip()))
    mon = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
    for p in glob.glob(os.path.join(CACHE, "wa", "*.html")):
        sex, ev, _ = os.path.basename(p)[:-5].rsplit("_", 2)
        t = BeautifulSoup(open(p, encoding="utf-8").read(), "lxml").find("table")
        for tr in (t.find_all("tr")[1:] if t else []):
            td = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(td) < 5:
                continue
            m = re.match(r"(\d{1,2}) ([A-Z]{3}) (\d{4})", td[4])
            if m and m.group(2) in mon:
                dob = (f"{int(m.group(1)):02d}.{mon[m.group(2)]:02d}.{m.group(3)[2:]}", td[5].strip() if len(td) > 5 else "")
                idx[(sex, ev, norm2(B.titlecase_wa(td[3])))].add(dob)
                try:
                    idx[(sex, ev, int(_), norm2(B.titlecase_wa(td[3])), round(B.deep_mark(td[1]), 2))].add(dob)
                except ValueError:
                    pass
    return idx


def byear(dob):
    yy = int(dob[-2:]) if dob[-2:].isdigit() else None
    return None if yy is None else (1900 + yy if yy > 26 else 2000 + yy)


def person_clusters(recs):
    """group (dob, nation) records of one name into people: the same person unless the records differ in both
    nation and date of birth, or the birth years are five or more apart. Label = earliest dob in the group."""
    recs = sorted(recs)
    parent = {r: r for r in recs}

    def find(r):
        while parent[r] != r:
            r = parent[r]
        return r
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            ya, yb = byear(a[0]), byear(b[0])
            far = ya is not None and yb is not None and abs(ya - yb) >= 5
            same = (a[0] == b[0]) or (not far and (a[1] == b[1] or not a[1] or not b[1]))
            if same:
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for r in recs:
        groups[find(r)].append(r)
    return {r: min(x[0] for x in groups[find(r)]) for r in recs}


def swim_raw_index():
    """(g, dist, stroke, year, swim_name) -> [(personId, country, time, dob, yob)], and a person clustering:
    World Aquatics sometimes lists one swimmer under two personIds, and two swimmers can share a name. Two
    personIds of one name are one person if they post an identical time in the same list, or if they share a
    nation and their birth dates agree, share day and month, are missing or a placeholder (a date shared by
    three or more different names), or the birth years differ by at most one. Different nations in the same
    season are always two people; a nation change between seasons is allowed."""
    idx = defaultdict(list)
    recs = defaultdict(dict)                      # name -> pid -> (nat set, dob, yob)
    same_list = defaultdict(list)                 # (file, name) -> [(pid, nat, time)]
    dob_names = defaultdict(set)
    for p in glob.glob(os.path.join(CACHE, "swim", "*.json")):
        g, dist, stroke, y = os.path.basename(p)[:-5].split("_")
        for r in json.load(open(p, encoding="utf-8")).get("swimmingWorldRankings", []):
            nm = B.swim_name(r)
            pid, nat, dob = r.get("personId"), r.get("participantCountryCode"), (r.get("dateOfBirth") or "")[:10]
            idx[(g, dist, stroke, int(y), nm)].append((pid, nat, r.get("time"), dob, r.get("yearOfBirth")))
            e = recs[nm].setdefault(pid, [set(), dob, r.get("yearOfBirth")])
            e[0].add(nat)
            same_list[(p, nm)].append((pid, nat, r.get("time"), int(y)))
            if dob:
                dob_names[dob].add(nm)
    placeholder = {d for d, ns in dob_names.items() if len(ns) >= 3}
    cluster = {}
    for nm, pids in recs.items():
        ps = sorted(pids)
        parent = {q: q for q in ps}

        def find(q):
            while parent[q] != q:
                q = parent[q]
            return q
        apart = set()
        for (fp, n2), lst in same_list.items():
            if n2 != nm:
                continue
            for i, a in enumerate(lst):
                for b in lst[i + 1:]:
                    if a[0] == b[0]:
                        continue
                    if a[2] == b[2]:
                        parent[find(b[0])] = find(a[0])               # the same swim listed twice
                    elif a[1] != b[1]:
                        apart.add(frozenset({a[0], b[0]}))           # two nations in one season
        for i, a in enumerate(ps):
            for b in ps[i + 1:]:
                if frozenset({a, b}) in apart:
                    continue
                na, da, ya = pids[a]
                nb, db, yb = pids[b]
                if not (na & nb):
                    continue
                da = "" if da in placeholder else da
                db = "" if db in placeholder else db
                ok = (not da or not db or da == db or da[5:] == db[5:]
                      or (ya and yb and abs(int(ya) - int(yb)) <= 1))
                if ok:
                    parent[find(b)] = find(a)
        labs = {q: find(q) for q in ps}
        multi = len(set(labs.values())) > 1
        for q in ps:
            cluster[(nm, q)] = (labs[q][:8] if multi else "")
    return idx, cluster


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    cats = B.categories()
    B.HOLDERS.update({B.norm(nm): nm for c in cats if c["sport"] == "Swimming" for _, _, nm in c["record_line"]})
    print("categories", len(cats), flush=True)
    LID, NID, SID, AID = league_id_index(), nfl_id_index(), streak_id_index(), ath_dob_index()
    WID, SWC = swim_raw_index()
    print("indexes built", flush=True)
    rows, log = [], []
    ath_alias = {}                     # (sex, ev, norm2 name, dob) for fuzzy merges across spellings
    for c in cats:
        cid, sport = c["id"], c["sport"]
        for y in sorted(c["seasons"]):
            ent = c["seasons"][y]
            g = c["games"](y) if c["counting"] and c["games"] else ""
            recs = []
            for i, (n, v) in enumerate(ent[:DEPTH + 5]):
                corr, clean = [], n
                if sport == "Athletics" and WIND_PREFIX.match(n):
                    clean = WIND_PREFIX.sub("", n)
                    corr.append("C1")
                aid, how = None, "name"
                if sport in ("MLB", "NBA", "NHL") and cid != "mlb_streak":
                    s = LID.get((sport, y, B.norm(n)), set())
                    if len(s) == 1:
                        aid, how = "ref:" + next(iter(s)), "reference_id"
                    elif len(s) > 1:
                        how = "ambiguous_reference_id"
                elif cid == "mlb_streak":
                    s = SID.get((y, B.norm(n)), set())
                    if len(s) == 1:
                        aid, how = next(iter(s)), "retrosheet_to_reference_id"
                elif sport == "NFL":
                    s = NID.get((y, B.norm(n)), set())
                    if len(s) == 1:
                        aid, how = next(iter(s)), "pfa_or_nflverse_id"
                    elif len(s) > 1:
                        how = "ambiguous_nfl_id"
                elif sport == "Swimming":
                    gg, dist, stroke = ("M" if c["sex"] == "M" else "F"), c["event"].split()[0], c["event"].split()[1]
                    raw = WID.get((gg, dist, stroke, y, n), [])
                    hit = [q for q in raw if q[2] and abs(B.swim_secs(q[2]) - v) < 1e-6] or raw
                    labs = {SWC.get((n, q[0]), "") for q in hit}
                    if len(labs) == 1:
                        lab = next(iter(labs))
                        aid, how = (f"swim:{norm2(n)}|{lab}" if lab else f"swim:{norm2(n)}"), "swimmer_cluster"
                    elif raw:
                        how = "ambiguous_swimmer"
                elif sport == "Athletics":
                    allrec = AID.get((c["sex"], c["event"], norm2(clean)), set())
                    cl = person_clusters(allrec)
                    if len(set(cl.values())) <= 1:
                        aid, how = f"ath:{norm2(clean)}", "name (one person)"
                    else:                            # two people with one name: use the dob/nation of this mark
                        d = AID.get((c["sex"], c["event"], y, norm2(clean), round(v, 2)), set())
                        labs = {cl[x] for x in d if x in cl}
                        if len(labs) == 1:
                            aid, how = f"ath:{norm2(clean)}|{next(iter(labs))}", "name_dob_nation"
                        else:
                            how = "ambiguous_dob"
                if aid is None:
                    aid = "name:" + norm2(clean)
                recs.append({"category": cid, "season": y, "orig_rank": i + 1 if i < 10 else "", "athlete_id": aid,
                             "id_method": how, "name_raw": n, "name_clean": clean,
                             "name_display": B.disp(clean, sport), "raw_value": v, "performance": float(c["perf"](v)),
                             "games_per_team": g, "short_season": y in c["short"], "correction": corr})
            # C2: marks dated after the snapshot (source typos) -> quarantine the whole phantom season
            if y > SNAPSHOT_YEAR:
                for r in recs:
                    r["correction"].append("C2")
                    log.append({"code": "C2", "category": cid, "season": y, "name": r["name_raw"], "value": r["raw_value"]})
            # C3 / C1 duplicates: one athlete twice in a season -> keep the best entry
            seen, keep = set(), []
            for r in recs:
                if "C2" in r["correction"]:
                    continue
                if r["athlete_id"] in seen:
                    r["correction"].append("C3" if sport == "Swimming" else "C1dup")
                    log.append({"code": r["correction"][-1], "category": cid, "season": y, "name": r["name_raw"],
                                "value": r["raw_value"]})
                    continue
                seen.add(r["athlete_id"])
                keep.append(r)
            for k, r in enumerate(keep):
                r["v2_rank"] = k + 1
            for r in recs:
                r.setdefault("v2_rank", "")
                if r["orig_rank"] or (r["v2_rank"] and r["v2_rank"] <= DEPTH):
                    r["correction"] = ";".join(r["correction"])
                    rows.append(r)
                if "C1" in r["correction"] and r["orig_rank"]:
                    log.append({"code": "C1", "category": cid, "season": y, "name": r["name_raw"], "value": r["raw_value"]})
    # C4 athletics: one person under two spellings (shared date of birth, similar names), and
    # C5: one person whose date of birth differs between sources (kept as one person by person_clusters)
    merge = {}
    TWINS = {frozenset({"alvin harrison", "calvin harrison"}),    # twin brothers, 400 m, both born 20.01.74
             frozenset({"tia clayton", "tina clayton"})}          # twin sisters, sprints, both born 17.08.04
    sim = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
    names_by_ev = defaultdict(lambda: defaultdict(set))
    for k, recs in AID.items():
        if len(k) == 3:
            names_by_ev[(k[0], k[1])][k[2]] |= {d for d, _ in recs}
    used = {(r["category"], r["athlete_id"]) for r in rows if r["athlete_id"].startswith("ath:")}
    for (sex, ev), nm in names_by_ev.items():
        cid = f"ath_{sex}_{ev}"
        by_dob = defaultdict(set)
        for n, ds in nm.items():
            if (cid, f"ath:{n}") in used:
                for d in ds:
                    by_dob[d].add(n)
            cl = person_clusters(AID[(sex, ev, n)])
            if (cid, f"ath:{n}") in used and len({d for d, _ in AID[(sex, ev, n)]}) > 1 and len(set(cl.values())) == 1:
                log.append({"code": "C5", "category": cid, "season": "", "value": "",
                            "name": f"{n}: dates of birth {sorted({d for d, _ in AID[(sex, ev, n)]})} in different "
                                    f"sources, treated as one person"})
        for d, names in by_dob.items():
            names = sorted(names)
            for a in names:
                for b in names:
                    if a >= b or frozenset({a, b}) in TWINS:
                        continue
                    if sim(a, b) >= 0.85 and sim(a.split()[0], b.split()[0]) >= 0.8:
                        merge[(cid, f"ath:{b}")] = f"ath:{a}"
                        log.append({"code": "C4", "category": cid, "season": "", "value": "",
                                    "name": f"{a} = {b} (born {d})"})
    for r in rows:
        r["athlete_id"] = merge.get((r["category"], r["athlete_id"]), r["athlete_id"])
    # C6 football: Pro Football Archives (to 1998) and nflverse (1999 on) IDs for one player across the boundary
    pfa_seen, nfl_seen = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["category"].startswith("nfl_"):
            k = B.norm(r["name_clean"])
            if r["athlete_id"].startswith("pfa:") and int(r["season"]) >= 1996:      # active up to the boundary
                pfa_seen[k].add(r["athlete_id"])
            if r["athlete_id"].startswith("nflv:") and int(r["season"]) <= 2001:     # and just after it
                nfl_seen[k].add(r["athlete_id"])
    cross = {}
    for k in set(pfa_seen) & set(nfl_seen):
        if len(pfa_seen[k]) == 1 and len(nfl_seen[k]) == 1:
            cross[next(iter(nfl_seen[k]))] = next(iter(pfa_seen[k]))
            log.append({"code": "C6", "category": "nfl", "season": "", "value": "",
                        "name": f"{k}: {next(iter(pfa_seen[k]))} = {next(iter(nfl_seen[k]))}"})
    for r in rows:
        r["athlete_id"] = cross.get(r["athlete_id"], r["athlete_id"])
    cols = ["category", "season", "orig_rank", "v2_rank", "athlete_id", "id_method", "name_raw", "name_clean",
            "name_display", "raw_value", "performance", "games_per_team", "short_season", "correction"]
    with open(os.path.join(OUT, "season_lists_v2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(r[k]) if k == "performance" else r[k]) for k in cols})
    with open(os.path.join(OUT, "corrections_applied.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "category", "season", "name", "value"])
        w.writeheader()
        w.writerows(log)
    from collections import Counter
    print("rows", len(rows), "id methods", Counter(r["id_method"] for r in rows if r["orig_rank"]))
    print("corrections", Counter(e["code"] for e in log))


if __name__ == "__main__":
    main()
