"""Each film's opening, timed to its narration: the lines are voiced first (narrate.py, cached), and each scene lasts as
long as its lines take, so captions and animations land as each line starts.

    ELEVENLABS_API_KEY=... python opening.py [a b c]   # -> opening_<film>.json, and VO_<film>.md with the opening added

The rest of each script (from the hook on) is VO_<film>_body.md, timed without an opening, shifted by its length.
"""
import json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REC, "outliers"))
I = json.load(open(os.path.join(HERE, "intro.json"), encoding="utf-8"))
XF = 0.6
LEAD, GAP, TAIL = 1.0, 0.45, 1.3                     # before the first line, between lines, after the last


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


d, ex = I["improve"], I["improve"]["examples"]
IMPROVE = ("improve", 12, [
    "Athletes keep getting better. Since 1950, world records in athletics and swimming have improved by about "
    "twenty-three percent in the typical event.",
    "Science, nutrition, full-time training and better technique: every generation starts from a higher floor."], [
    ((0, 0), f"Since 1950, world records in athletics and swimming have risen by about {d['median'][-1]:.0f}% in the typical "
             f"event. The 100 m has gone from {ex['ath_M_100m']['y1950']:.1f} to {ex['ath_M_100m']['now']:.2f}, the pole "
             f"vault from {ex['ath_M_PV']['y1950']:.2f} m to {ex['ath_M_PV']['now']:.2f} m."),
    ((1, 0), "Science, nutrition, full-time training and better technique: each generation starts from a higher floor.")], {})
OLD50 = ("An athlete of the 1950s, with little weight training or nutrition science, can't be compared with today's "
         "full-time professionals without allowing for it.")
OLD50_CAP = ("And an athlete of the 1950s, with little weight training or sports science, can't be compared with a full-time "
             "professional of today without allowing for it.")
NH = I["c"]["nhl"]
RATE = ("rate", 10, [
    "So we look for the biggest statistical outliers in two steps. Step one: every performance is converted to the "
    "scoring level of the season being judged.",
    "In hockey's run-and-gun 1980s, teams scored almost four goals a game. In 2003–04, about two and a half."], [
    ((0, 0), "First, the game itself changes. When every team scores more, each goal is worth a little less, like money in "
             "an inflated economy. So every performance is first converted to the scoring level of the season being judged."),
    ((1, 0), f"NHL teams scored {NH['1986']:.2f} goals a game in 1985–86 and {NH['2004']:.2f} in 2003–04."),
    ((-1, -1.5), "Home runs, points and yards get the same treatment.")], {"band": (1, -0.5)})
SHARED = ("They trained and competed with much the same science, so together they set the standard for the best of "
          "their time: the curve.")
SD = ("The usual spread of the best around that curve is one standard deviation. The score is how many standard "
      "deviations a performance sits above it.")
MCAPS = ["Then each performance is compared with the best by everyone else in the 15 seasons before and after.",
         "They trained, ate and competed with much the same science and coaching, so together they set the standard for "
         "the best of that time: the curve.",
         "The usual spread of the best around that curve is one standard deviation.",
         "The score is that count: how many standard deviations above the best of its time."]


def method(lines, example_cap, chart_title, example, cap_at=None):
    caps = cap_at or [(0, 0), (1, 0), (2, 0), (3, 0), (-1, -1.0)]
    texts = [MCAPS[0], MCAPS[1], MCAPS[2], example_cap, MCAPS[3]]
    at = {"curve": (1, 2.0), "band": (2, 1.0), "example": (3, 0)} if len(lines) == 4 else \
         {"curve": (0, 7.0), "band": (1, 1.0), "example": (2, 0)}
    return ("method", 20, lines, list(zip(caps, texts)), at, {"example": example, "chart_title": chart_title})


bo = I["c"]["example"]["best_other"]
FILMS = {
    "c": [IMPROVE,                                       # film C's wording, from the review doc (6 November)
          ("tiles", 10, ["The games themselves also change. Sixty home runs in 1927, when teams hit one every three games, "
                         "is a much more impressive feat than hitting seventy in 1998, when teams were hitting nearly three "
                         "times as many.",
                         "An athlete of the 1950s, with little weight training or nutrition science, can't be compared with "
                         "today's full-time professionals without controlling for these changes."], [
              ((0, 0), "Sixty home runs in 1927, when a team hit one about every three games, is a much bigger feat than "
                       "seventy in 1998, when teams hit nearly three times as many."),
              ((1, 0), "And an athlete of the 1950s, with little weight training or sports science, can't be compared with "
                       "a full-time professional of today without controlling for these changes.")], {},
           {"title": "Sixty home runs isn't always sixty home runs",
            "tiles": [{"big": x["value"], "label": f"home runs · {x['who']} · {x['year']}",
                       "context": f"teams hit {x['rate']:.2f} a game"} for x in I["c"]["tiles"]]}),
          ("rate", 10, [
              "So I looked for the biggest statistical outliers in team sports history in two steps. Step one: every "
              "other performance is converted to the scoring level of the season being judged.",
              "For example, in hockey's run-and-gun 1980s, teams scored almost four goals a game. In 2003–2004, about "
              "two and a half."], [
              ((0, 0), "First, the game itself changes. When every team scores more, each goal is worth a little less, like "
                       "money in an inflated economy. So every other season is first converted to the scoring level of "
                       "the season being judged."),
              ((1, 0), f"NHL teams scored {NH['1986']:.2f} goals a game in 1985–86 and {NH['2004']:.2f} in 2003–04."),
              ((-1, -1.5), "Home runs, points and yards get the same treatment.")], {"band": (1, -0.5)}, {"nhl": NH}),
          ("method", 20, [
              "Step two: each season is compared with only the best of everyone else in the fifteen seasons before and "
              "after. Each of these players was in the top one percent of their league that season.",
              "They trained and competed with much the same science, so together they set the standard for the best of "
              "their time. Let's call that the elite curve.",
              "The usual spread of the best around that curve is one standard deviation. The score we assign our athletes "
              "is how many standard deviations a performance sits above it.",
              "Barry Bonds's seventy-three home runs are the record, but only two standard deviations above the elite "
              "curve: in his era, the other best hitters were that good."], [
              ((0, 0), "Then each performance is compared with only the best by everyone else in the 15 seasons before and "
                       "after: each season's leader, the top one percent of their league."),
              ((1, 0), "They trained, ate and competed with much the same science and coaching, so together they set the "
                       "standard for the best of that time: the elite curve."),
              ((2, 0), "The usual spread of the best around the elite curve is one standard deviation."),
              ((3, 0), f"Barry Bonds's 73 in 2001 is the record, but only {I['c']['example']['z']:.1f} standard deviations "
                       f"above the elite curve: {bo['name'].split()[-1]}'s {bo['value']} in {bo['year']} is worth about "
                       f"{bo['converted']:.0f} in 2001 terms."),
              ((-1, -1.0), "The score is that count: how many standard deviations above the elite curve.")],
           {"curve": (1, 2.0), "band": (2, 1.0), "example": (3, 0)},
           {"example": I["c"]["example"], "curve_label": "the elite curve",
            "chart_title": "Home runs in a season: the most by anyone else each season, in 2001 terms"})],
    "a": [("improve", 12, [                               # film A's wording, from the review doc (5 November)
              "Athletes keep getting better... Since 1950, world records in athletics and swimming have improved by "
              "about twenty-three percent.",
              "Science, nutrition, full-time training and better technique let every generation start from a higher "
              "floor."], IMPROVE[3], {}),
          ("tiles", 10, ["The games themselves also change... For example, four home runs by one player in a single "
                         "game in 1932, when whole teams were only hitting about one every other game, is much more "
                         "impressive than the same feat in 2017, when teams were hitting more than twice as many home runs.",
                         "In the same way, an athlete of the 1950s, with little weight training or nutrition science, "
                         "can't be fairly compared with today's full-time professionals, who build every part of their "
                         "day around their sport, unless we allow for it."], [
              ((0, 0), f"Four home runs in a game in 1932, when a team hit one about every other game, isn't the same feat "
                       f"as in 2017, when teams hit {I['a']['tiles'][2]['rate']:.2f} a game."), ((1, 0), OLD50_CAP)], {},
           {"title": "Four home runs isn't always four home runs",
            "tiles": [{"big": x["value"], "label": f"{x['who']} · {x['year']}", "context": f"teams hit {x['rate']:.2f} a game"}
                      for x in I["a"]["tiles"]]}),
          ("rate", 10, [RATE[2][0], RATE[2][1] + " We need to adjust the data to account for this."], RATE[3], RATE[4],
           {"nhl": NH}),
          ("method", 20, [
              "Step two: each game is compared with the five best games by everyone else, in each of the fifteen "
              "seasons before and after.",
              "Their peers in this group trained and competed with much the same science, so together they set the "
              "standard for the best of their time: the curve.",
              "The usual spread of the best around that curve is one standard deviation. The measure we use to find the "
              "greatest performances is how many standard deviations a single performance sits above it.",
              "On a bell curve, about sixty-eight percent of results fall within one standard deviation either side, "
              "ninety-five percent within two, and ninety-nine point seven percent within three.",
              "Adrian Peterson's record two hundred and ninety-six rushing yards stand almost four standard deviations "
              "above the curve. On a perfect bell curve, a result that high would come up about once in twenty thousand "
              "tries.",
              "Remember, the curve is the average of the best, not of every player. In 2007 the tenth-best rushing game "
              "by anyone else was a hundred and fifty-four yards; Peterson ran for nearly twice that. Measured against an "
              "ordinary running back's Sunday, he'd be off the chart. But we're not asking who beat the average player. "
              "We're asking who stood out most among the very best."], [
              ((0, 0), MCAPS[0]), ((1, 0), MCAPS[1]), ((2, 0), MCAPS[2]),
              ((3, 0), "On a bell curve, about 68% of results fall within one standard deviation either side, 95% within "
                       "two and 99.7% within three."),
              ((4, 0), f"Adrian Peterson's record 296 rushing yards in 2007 stand {I['a']['example']['z']:.1f} standard "
                       "deviations above the curve: on a perfect bell curve, about 1 in 20,000. Here each season's class "
                       "is its five best games."),
              ((5, 0), "The curve is the average of the best, not of every player. The dashed line is each season's "
                       "tenth-best game: in 2007, 154 yards to Peterson's 296. Against every player these games would be "
                       "off the chart; the question here is who stood out most among the very best.")],
           {"curve": (1, 2.0), "band": (2, 1.0), "example": (4, 0), "page": (4, 0), "field": (5, 0)},
           {"example": I["a"]["example"],
            "chart_title": "Rushing yards in a game: the five best by anyone else each season, in 2007 terms"})],
    "b": [IMPROVE,
          ("tiles", 10, ["A sprinter of 1950, with little weight training, sports science or modern technique, can't be "
                         "compared with a full-time professional of today without allowing for it.",
                         "In 1950 the hundred-metre world record was ten point two. Today it's nine fifty-eight.",
                         "In athletics and swimming a metre is a metre, so nothing is converted."], [
              ((0, 0), "A sprinter of 1950, with little weight training, sports science or modern technique, can't be "
                       "compared with a full-time professional of today without allowing for it."),
              ((2, 0), "In athletics and swimming a metre is a metre, so nothing is converted: the comparison does the "
                       "work.")], {},
           {"title": "The same event, a different sport",
            "tiles": [{"big": f"{ex['ath_M_100m']['now']:.2f}", "label": f"100 m world record · {ex['ath_M_100m']['now_year']}",
                       "context": f"1950: {ex['ath_M_100m']['y1950']:.1f}"},
                      {"big": f"{ex['ath_M_PV']['now']:.2f} m", "label": f"pole vault world record · {ex['ath_M_PV']['now_year']}",
                       "context": f"1950: {ex['ath_M_PV']['y1950']:.2f} m"},
                      {"big": mmss(ex['swim_M_1500_fr']['now']),
                       "label": f"1500 m freestyle record · {ex['swim_M_1500_fr']['now_year']}",
                       "context": f"1950: {mmss(ex['swim_M_1500_fr']['y1950'])}"}]}),
          method(["Instead, each mark is compared with the best by everyone else in the fifteen seasons before and after. "
                  + SHARED, SD,
                  "Katie Ledecky's eight oh four seventy-nine in 2016 is almost five standard deviations above the curve."],
                 f"Katie Ledecky's 8:04.79 in the 800 m freestyle in 2016 sits {I['b']['example']['z']:.1f} standard "
                 "deviations above the curve.",
                 "Women's 800 m freestyle: the fastest by anyone else each season · ▲ faster", I["b"]["example"],
                 cap_at=[(0, 0), (0, 6.5), (1, 0), (2, 0), (-1, -1.0)]),
          ("throws", 12, [
              "The method also handles a harder case. Two of the oldest world records are in the women's throws: the shot "
              "put from 1987 and the discus from 1988.",
              "In the 1980s, marks like these were routine at the top: eighteen shot puts of twenty-two metres or more, and "
              "fifty discus throws of seventy-two or more. No woman has put the shot twenty-two metres since 1988.",
              "So the curve for those years sits high, and the records stand only two point three and three point two "
              "standard deviations above it. The whole era is discounted, without singling anyone out."], None,
           {"band": (1, 0), "curve": (2, 0), "note": (2, 6.0)}, {})],
}


def throws_caps(top10_min):
    th = I["b"]["throws"]
    sp, dt = th["ath_W_SP"], th["ath_W_DT"]
    return [((0, 0), f"Two of the oldest world records are in the women's throws: the shot put's {sp['value']:.2f} m from "
                     f"{sp['year']} and the discus's {dt['value']:.2f} m from {dt['year']}."),
            ((1, 0), f"In the 1980s marks like these were routine at the top: {sp['marks_1980s']} shot puts of "
                     f"{sp['threshold']:.0f} m or more by {sp['athletes_1980s']} women, and {dt['marks_1980s']} discus throws "
                     f"of {dt['threshold']:.0f} m or more by {dt['athletes_1980s']}. No woman has put the shot "
                     f"{sp['threshold']:.0f} m since {sp['last_year_at_threshold']}."),
            ((2, 0), f"So the curve for those years sits high, and the records stand only {sp['z']:.1f} (shot) and "
                     f"{dt['z']:.1f} (discus) standard deviations above it. The top ten in this film starts at "
                     f"{top10_min:.1f}."),
            ((2, 6.0), "The method discounts the whole era, whatever lifted it, without singling anyone out.")]


SAY = json.load(open(os.path.join(HERE, "say.json"), encoding="utf-8"))   # respellings for the voice, by film


CHARS_PER_S = 16.45                                     # this voice's pace (film A's clips): for a timing estimate


def clip_len(text, cache):
    """a line's length when voiced; with NARRATE_ESTIMATE=1, a line not yet voiced is estimated from its length (for
    stills and layout checks only: the film is timed to the real clips)"""
    import narrate
    if os.environ.get("NARRATE_ESTIMATE") and not narrate.cached(text, cache):
        return round(len(narrate.say(text)) / CHARS_PER_S + 0.3, 2)
    mp3 = narrate.tts(text, cache)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", mp3],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def plan(f):
    import narrate
    narrate.SAY = SAY.get(f, {})
    cache = os.path.join(HERE, f"out_{f}", "voice")
    os.makedirs(cache, exist_ok=True)
    top10 = min(c["z_e3"] for c in json.load(open(os.path.join(HERE, f"facts_{f}.json"), encoding="utf-8"))["cards"])
    t0, scenes, cues = 0, [], []
    for sc in FILMS[f]:
        name, min_dur, lines, caps, at = sc[:5]
        extra = sc[5] if len(sc) > 5 else {}
        if caps is None:
            caps = throws_caps(top10)
        starts, ends, t = [], [], float(t0) + LEAD
        for ln in lines:
            t = math.ceil(t)
            n = clip_len(ln, cache)
            starts.append(t - t0)
            ends.append(t - t0 + 0.15 + n)
            cues.append((t, ln))
            t = t + 0.15 + n + GAP
        dur = max(min_dur, math.ceil(ends[-1] + TAIL))

        def resolve(a):
            i, off = a
            return (ends[-1] + off) if i == -1 else (starts[i] + off)
        spec = dict(extra, caps=[(round(resolve(a), 2), s) for a, s in caps],
                    at={k: round(resolve(v), 2) for k, v in at.items()})
        if name == "throws":
            spec["top10_min"] = top10
        scenes.append((name, dur, spec))
        t0 += dur
    json.dump({"scenes": scenes, "cues": cues, "dur": t0}, open(os.path.join(HERE, f"opening_{f}.json"), "w",
                                                                  encoding="utf-8"), indent=1, ensure_ascii=False)
    if f == "b":                                            # films A and C: script_a.py, script_c.py write the script
        script(f, t0, cues)
    print(f, "opening", t0, "s:", [(n, d_) for n, d_, _ in scenes])


def script(f, dur, cues):
    """VO_<f>.md: the opening's cues, then VO_<F>_body.md shifted by the opening's length"""
    old = open(os.path.join(HERE, f"VO_{f.upper()}_body.md"), encoding="utf-8").read()   # the script without an opening
    shift = dur - XF

    head, rest = old.split("\n## Hook", 1)
    rest = re.sub(r"\*\*(\d+):(\d\d)\*\*", lambda m: f"**{(lambda t: f'{t // 60}:{t % 60:02d}')(int(round(int(m.group(1)) * 60 + int(m.group(2)) + shift)))}**", rest)
    rest = re.sub(r"\((\d+):(\d\d)(–\d+:\d\d)?\)", lambda m: f"({(lambda t: f'{t // 60}:{t % 60:02d}')(int(round(int(m.group(1)) * 60 + int(m.group(2)) + shift)))})", rest)
    rest = rest.replace("How does the score work? Picture a class test.", "Here's the same idea as a class test.")
    tot = re.search(r"final cut \((\d+):(\d\d)\)", head)
    total = int(tot.group(1)) * 60 + int(tot.group(2)) + shift
    head = head.replace(tot.group(0), f"final cut ({int(total // 60)}:{int(round(total % 60)):02d})")
    op = "\n## Opening: why eras matter, and how the score allows for them (0:00)\n\n" + \
         "\n\n".join(f"**{t // 60}:{t % 60:02d}** {ln}" for t, ln in cues) + "\n"
    open(os.path.join(HERE, f"VO_{f.upper()}.md"), "w", encoding="utf-8").write(head + op + "\n## Hook" + rest)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["a", "b", "c"]:
        plan(f)
