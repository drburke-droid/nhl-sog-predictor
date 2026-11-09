"""YouTube copy for the three companion films (titles, description with the countdown and chapters, full photo
credits), from each film's facts, timeline and credits, into ../../../videos/copy/.

    python copy3.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], "x"]
sys.path.insert(0, HERE)
import common                                               # noqa: E402
import afilm, bfilm, cfilm                                  # noqa: E402

COPY = os.path.normpath(os.path.join(HERE, "..", "..", "..", "videos", "copy"))
NAMES = {"a": "biggest_single_game_outliers", "b": "biggest_single_performances_athletics_swimming",
         "c": "biggest_season_outliers_team_sports"}


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def length_tag(total):
    return f"{int(total // 60)}m{int(round(total % 60)):02d}s"


def timeline(film):
    """section start times, without building any card (the three films share film.py's card state)"""
    import film as FM
    fixed = {"hook": getattr(film, "HOOK_DUR", FM.Hook.dur), "curve": getattr(film, "CURVE_DUR", FM.Curve.dur),
             "fair": getattr(film, "FAIR_DUR", FM.Fair.dur)}            # film A times its own (script_a.py)
    ds = [FM.Card.dur if s[0] == "c" and s[1:].isdigit() else fixed.get(s) or film.SECTIONS[s]().dur for s in film.ORDER]
    off = [0.0]
    for d in ds[:-1]:
        off.append(off[-1] + d - common.A.XF)
    return dict(zip(film.ORDER, off)), dict(zip(film.ORDER, ds)), off[-1] + ds[-1]


def chapters(film, labels):
    O, durs, total = timeline(film)
    out = []
    for s in film.ORDER:
        if s in labels:
            out.append(f"{mmss(O[s])} {labels[s]}")
    return out, total


def credit_lines(keys, credits):
    out = []
    for k in keys:
        cr = credits.get(k)
        if not cr:
            continue
        lic = cr["license"] if cr["license"] != "Public domain" else "public domain"
        url = f" ({cr['license_url']})" if cr.get("license_url") else ""
        out.append(f"{cr['athlete']}: \"{cr['file']}\" by {cr['author']}, {lic}{url}, {cr['page']}. Changes: {cr['changes']}.")
    return out


def short_url(u):
    from urllib.parse import unquote
    return unquote(u.replace("https://", "").replace("http://", "").replace("//", "", 1) if u.startswith("//") else
                   u.replace("https://", "").replace("http://", "")).replace(" ", "_")


def credit_lines_compact(keys, credits):
    """YouTube's description holds 5,000 characters: a full credit (file, author, licence, link) for each photo whose
    licence asks for attribution, and one line naming the public-domain and CC0 photos"""
    full, free = [], []
    for k in keys:
        cr = credits.get(k)
        if not cr:
            continue
        lic = cr["license"]
        if lic.lower() in ("public domain", "cc0") or lic.lower().startswith("cc0"):
            who = cr["author"] if cr["author"].lower() not in ("unknown author", "unknown", "uncredited") else None
            free.append(f"{cr['athlete']}" + (f" ({who})" if who else ""))
        else:
            lu = f" ({short_url(cr['license_url'])})" if cr.get("license_url") else ""
            full.append(f"{cr['athlete']}: \"{cr['file']}\" by {cr['author']}, {lic}{lu}, {short_url(cr['page'])}")
    out = ["Photos from Wikimedia Commons, cropped, backgrounds removed and toned in team colours:"] + full
    if free:
        out.append("Public domain or CC0: " + ", ".join(free) + ".")
    return out


def write(key, film, title, alt_titles, hook, method, countdown, labels, keys, sources, extra=(), rebuild=None,
          thumbs=(), hashtags=(), tags=(), compact=False):
    ch, total = chapters(film, labels)
    name = f"{NAMES[key]}_{length_tag(total)}"
    L = [f"# YouTube copy: \"{title}\"", "",
         f"**Upload `sports/{name}_voiced.mp4`** ({mmss(total)}, 1920×1080, 30 fps, H.264; narrated over the original orchestral "
         "score, kept at a steady low level under the voice with no ducking, −16 LUFS).", "",
         "To rebuild it: " + (rebuild or f"`python facts_{key}.py`, `./render.sh {key}`, `python {key}film.py assemble` in "
                              "`world_roads/records/three` (narration: `narrate.py` on `VO_" + key.upper() +
                              ".md`, cached by line)."), ""]
    if thumbs:
        L += ["**Thumbnails to test** (1280×720, in `sports/thumbnails/`)", "", "| | File | Idea |", "|---|---|---|"]
        L += [f"| {'ABC'[i]} | `{f}` | {idea} |" for i, (f, idea) in enumerate(thumbs)] + [""]
    L += ["**Titles to test**", "", "| | Title |", "|---|---|"]
    for i, t in enumerate([title] + list(alt_titles)):
        L.append(f"| {'ABC'[i]} | {t} |")
    if tags:
        L += ["", "**Tags** (YouTube's tags field)", "", ", ".join(tags)]
    L += ["", "**Description**", "", "```", hook, "", method, "", "The countdown:"] + countdown + [""] + list(extra) + \
         ["Chapters:"] + ch + ["", "Sources: " + sources, "",
                               "The information used here was obtained free of charge from and is copyrighted by "
                               "Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org." if "Retrosheet" in sources else "",
                               ] + (credit_lines_compact(keys, film.CREDITS) if compact else
                                    ["Photos (Wikimedia Commons):"] + credit_lines(keys, film.CREDITS)) + \
         ["Chart icons: BlenderKit models. Music: original score, FluidR3 GM soundfont (MIT).",
          "Scores are statistical distances from each era's curve, not probabilities or measures of ability."] + \
         ([""] + [" ".join(hashtags)] if hashtags else []) + ["```", ""]
    p = os.path.join(COPY, f"{NAMES[key]}_YOUTUBE.md")
    open(p, "w", encoding="utf-8").write("\n".join(x for x in L if x is not None) + "\n")
    print(p, name)
    return name


def main():
    out = {}
    # ---- A
    F = afilm.FA
    cd = [f"{c['rank']}. {c['athlete']}, {c['season']}: {c['achievement']} ({c['z_e3']:.2f})" for c in
          sorted(F["cards"], key=lambda c: -c["rank"])]
    lab = {"intro": "Comparing across eras: how the score works", "hook": "The biggest outliers", "curve": "Grading on a curve", "fair": "Four rules that keep it fair",
           "sure": "How sure is this list?", "close": "What this ranking is", "end": "Sources and photo credits"}
    lab.update({f"c{c['rank']}": f"#{c['rank']} {c['athlete']}" for c in F["cards"]})
    out["a"] = write("a", afilm, "The 10 Biggest Single-Game Outliers in Team Sports",
                     ["Wilt's 100, Van Brocklin's 554, Bottomley's 12 RBI: The Games That Broke the Curve",
                      "One Game, Off the Charts: Team Sports' 10 Biggest Outliers"],
                     "Wilt Chamberlain's 100 points. Norm Van Brocklin's 554 passing yards. Jim Bottomley's 12 RBI. Which "
                     "single game stood furthest above everyone else of its era?",
                     f"We took the best game of every season in {F['scale']['n_categories']} team-sport categories "
                     f"({F['scale']['n']:,} games), scaled every game to regulation length, and compared each with the five "
                     "best games by anyone else in each of the 15 seasons before and after, around a curved trend, "
                     "converted by the league's scoring rate. The score is how many standard deviations a game sits above that "
                     "curve. A game makes the list only if it stands at least one standard deviation clear of every other "
                     "game on its graph.", cd, lab, [c["photo"] for c in F["cards"]],
                     "NBA box scores (Eoin Moore, Historical NBA Data and Player Box Scores, CC0, via Hugging Face); "
                     "NHL_DB (Hugging Face, from NHL.com game data); Retrosheet; Pro Football Archives (NFL box scores "
                     "1950–1998); nflverse.")
    # ---- B
    F = bfilm.FB
    cd = [f"{c['rank']}. {c['athlete']}, {c['season']}: {c['achievement']} ({c['z_e3']:.2f})" for c in
          sorted(F["cards"], key=lambda c: -c["rank"])]
    lab = {"intro": "Comparing across eras: how the score works", "hook": "The biggest outliers", "curve": "Grading on a curve", "fair": "Four rules that keep it fair",
           "sure": "How sure is this list?", "close": "What this ranking is", "end": "Sources and photo credits"}
    lab.update({f"c{c['rank']}": f"#{c['rank']} {c['athlete']}" for c in F["cards"]})
    out["b"] = write("b", bfilm, "The 10 Biggest Single Performances in Athletics and Swimming",
                     ["Beamon, Železný, Bolt: The Marks That Broke the Curve",
                      "One Race, One Jump, One Throw: The 10 Biggest Outliers"],
                     "Bob Beamon's 8.90 m. Jan Železný's 98.48 m. Usain Bolt's 9.58. Which single performance stood "
                     "furthest above everyone else of its era?",
                     f"We took every season's best mark in {F['scale']['n_categories']} athletics and swimming events "
                     f"({F['scale']['n']:,} marks) and compared each with the best mark by anyone else in the 15 seasons "
                     "before and after, around a curved trend (races scored as speed; swimming's 2008–09 supersuit "
                     "seasons treated as a shift). The score is how many standard deviations a mark sits above that curve. A mark "
                     "makes the list only if it stands at least one step clear of every other mark on its graph.",
                     cd, lab, [c["photo"] for c in F["cards"]],
                     "World Athletics; World Aquatics; alltime-athletics.com (Peter Larsson).",
                     extra=["Gretchen Walsh has no freely licensed photo, so her card runs without one.", ""])
    # ---- C
    F = cfilm.FC
    cd = [f"{c['rank']}. {c['athlete']}, {c['season']}: {c['achievement']} ({c['z_e3']:.2f} standard deviations; top ten in "
          f"{c['specs_in_top10']} of 16 versions)" for c in sorted(F["cards"], key=lambda c: -c["rank"])]
    near = [f"#{k['rank_by_score']} by score: {k['athlete']}, {k['season']}, {k['achievement']} ({k['z']:.2f}; "
            f"{k['specs_in_top10']} of 16). {k['why']}" for k in F["considered"]]
    lab = {"intro": "Comparing across eras: how the score works", "hook": "The biggest outliers", "curve": "Grading on a curve", "fair": "Four rules that keep it fair",
           "sure": "How sure is this list?", "considered": "Five that came close", "close": "What this ranking is",
           "end": "Sources and photo credits"}
    lab.update({f"c{c['rank']}": f"#{c['rank']} {c['athlete']}" for c in F["cards"]})
    out["c"] = write("c", cfilm, "The 5 Most Dominant Seasons in Team Sports History",
                     ["Wilt, DiMaggio, Gretzky: The Seasons That Broke the Curve",
                      "5 Seasons Nobody Came Close To"],
                     "Wilt Chamberlain averaged 50.4 points a game; the next best scorer averaged 31.6. Joe DiMaggio hit "
                     "in 56 straight games; nobody else that year got past 24. Wayne Gretzky's 163 assists were more than "
                     "anyone else's points. Which seasons stood furthest above the best of their own era?",
                     "Athletes keep getting better and the games keep changing, so raw numbers can't be compared across "
                     f"eras. I took {F['scale']['n']:,} season-leading performances in {F['scale']['n_categories']} "
                     "categories (basketball, hockey, baseball and football) and judged each in two steps. First, every "
                     "other season is converted to the scoring level of the season being judged. Then it is compared "
                     "with only the best of everyone else, each season's leader, in the 15 seasons before and after: the "
                     "elite curve. The score is how many standard deviations a season sits above the elite curve. A "
                     "season makes the list only if it stands at least one standard deviation clear of every other "
                     "season on its graph, and the five here are in the top ten of at least 12 of 16 versions of the "
                     "method.",
                     cd, lab, [c["photo"] for c in F["cards"]] + [k["photo"] for k in F["considered"]],
                     "Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro Football Archives, nflverse, "
                     "Retrosheet.", extra=["Five that came close (the rest of the top ten by score):"] + near + [""],
                     rebuild="`python opening.py c`, `python script_c.py voice <chunk>` (opening, hook, cards, curve, "
                             "rules; each line cached), `python script_c.py`, `./render.sh c`, `python cfilm.py assemble` "
                             "in `world_roads/records/three`.",
                     thumbs=[("season_outliers_thumb_D.png", "NOT EVEN CLOSE: 2,602 other season-leading "
                                                             "performances piled up, and Wilt alone, 7 standard "
                                                             "deviations away"),
                             ("season_outliers_thumb_E.png", "NO ONE ELSE IN SIGHT: a running track from above; the pack "
                                                             "(the 153 other seasons on the five's graphs) is within "
                                                             "2.6 standard deviations of the elite curve, the five "
                                                             "lead around the turn at 5.3 to 7"),
                             ("season_outliers_thumb_F.png", "OFF THE BELL CURVE: the bell curve runs out by 3; the five "
                                                             "sit between 5 and 7")],
                     hashtags=["#sports", "#statistics", "#WiltChamberlain"], compact=True,
                     tags=["most dominant seasons", "greatest seasons in sports history", "sports statistics",
                           "standard deviation", "Wilt Chamberlain 50 points a game", "Joe DiMaggio 56 game hitting streak",
                           "Wayne Gretzky 163 assists", "Dennis Rodman rebounds", "Ichiro 262 hits", "NBA", "MLB", "NHL",
                           "NFL", "sports analytics", "era adjusted stats", "statistical outliers"])
    json.dump(out, open(os.path.join(HERE, "names.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
