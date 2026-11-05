"""Film A's narration, timed: the hook and "grading on a curve" are timed to their lines (voiced first, cached), the
other sections keep their length and their lines' offsets (VO_A_body.md), and the whole script is written out with
positions in the final cut.

    ELEVENLABS_API_KEY=... python opening.py a      # the opening (opening_a.json)
    ELEVENLABS_API_KEY=... python script_a.py       # timed_a.json and VO_A.md

The hook keeps its pictures and music; script_a.py only says when each part of it plays (knots: the hook's own
clock against the film's), so the five lines land with their captions and the faces and title keep their music.
"""
import json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "outliers"))
import opening                                              # noqa: E402  (clip_len, LEAD, GAP, TAIL, XF, SAY)

F = json.load(open(os.path.join(HERE, "facts_a.json"), encoding="utf-8"))
SC = F["scale"]

HOOK = [
    "I analysed four thousand three hundred and sixty-four season-leading performances in basketball, hockey, baseball "
    "and football.",
    "Each was graded against the best games of everyone else in its era: fifteen years before and, where possible, "
    "fifteen years after.",
    "Most performances land close to the curve. That makes sense: the curve is the typical level of the best in each era.",
    "Only six are more than five standard deviations above it. On a perfect bell curve, that's about one in three and a "
    "half million.",
    "Ten stood clear of the field. So without further introduction, these are the ten biggest single-game outliers in "
    "team sports, counting down from number ten."]
HOOK_WORDS = {"graded": "Each one graded against the best games of everyone else in its era, 15 seasons either side.",
              "most": "Most land close to the curve: the typical level of the best in each era.",
              "five": f"Only {SC['at_least_5']} of {SC['n']:,} are more than 5 standard deviations above it.",
              "axis": "score: standard deviations above the curve"}

CLASS = [
    "Here's the same idea, using a class test. One student scores ninety-five. Leave them out, and take the five best "
    "scores by everyone else in the class: they're all around eighty.",
    "Do the same for the fifteen classes before and the fifteen after. Their top fives land around eighty too, and the "
    "line through them is the curve.",
    "If one year's test was easier for everyone, its scores are scaled down first, just as every season is converted to "
    "the same scoring level.",
    "The top scores usually sit within about three points of the curve. That's one standard deviation.",
    "So the student's ninety-five is fifteen points above the curve: five standard deviations.",
    "But if the top fives in those years had scored around ninety-two, the same ninety-five would be just one standard "
    "deviation up.",
    "Same score, different era. So every game is graded against the best of everyone else in its own era."]
CLASS_CAPS = [
    "Picture a class test. One student scores 95. Leave them out, and take the five best scores by everyone else: "
    "about 80.",
    "Do the same for the 15 classes before and the 15 after. The line through all those top fives is the curve.",
    "A year with an easier test is scaled down first, just as every season is converted to the same scoring level.",
    "The top scores usually sit within about 3 points of the curve: one standard deviation.",
    "The 95 is 15 points above the curve: 15\u00a0÷\u00a03\u00a0=\u00a05 standard deviations. On our scale, a score "
    "of 5.",
    "If the top fives around it had scored about 92, the same 95 would be (95\u00a0−\u00a092)\u00a0÷\u00a03\u00a0=\u00a01 "
    "standard deviation up: a score of 1.",
    "Same score, different era. So every game in this film is graded against the best of everyone else in its own era."]

HOOK_FACES, HOOK_LEN = 17.4, 30.0                           # the hook's own clock: the faces arrive, the hook ends


def voice(lines, cache):
    """when each line starts and ends, from LEAD, with GAP between lines (whole tenths)"""
    starts, ends, t = [], [], opening.LEAD
    for ln in lines:
        t = math.ceil(t * 10) / 10
        n = opening.clip_len(ln, cache)
        starts.append(t)
        ends.append(t + 0.15 + n)
        t = t + 0.15 + n + opening.GAP
    return starts, ends


def at_word(line, phrase, start, end):
    """roughly when a phrase is spoken: its share of the line's characters"""
    return round(start + (end - start) * line.index(phrase) / len(line), 2)


def plan_hook(cache):
    s, e = voice(HOOK, cache)
    faces = s[4] - 0.6                                      # "Ten stood clear" lands as its caption does (faces + 0.6)
    dur = max(faces + HOOK_LEN - HOOK_FACES, math.ceil((e[4] + opening.TAIL) * 10) / 10)
    knots = [(0.0, 0.0), (7.2, s[1]), (10.0, s[2]), (12.4, s[3]), (15.0, faces - 2.4), (HOOK_FACES, faces),
             (HOOK_LEN, faces + HOOK_LEN - HOOK_FACES)]
    if dur > knots[-1][1] + 0.05:
        knots.append((HOOK_LEN + 0.5, dur))
    music = [(0.0, 0.0), (9.6, faces - (HOOK_FACES - 9.6)), (HOOK_LEN, faces + HOOK_LEN - HOOK_FACES),
             (HOOK_LEN + 30.0, faces + HOOK_LEN - HOOK_FACES + 30.0)]
    return {"dur": round(dur, 2), "knots": [[round(a, 2), round(b, 2)] for a, b in knots],
            "music": [[round(a, 2), round(b, 2)] for a, b in music], "starts": s, "ends": e, "words": HOOK_WORDS}


def plan_class(cache):
    s, e = voice(CLASS, cache)
    at = {"s95": at_word(CLASS[0], "One student", s[0], e[0]), "top5": at_word(CLASS[0], "Leave them out", s[0], e[0]),
          "others": s[1], "curve": at_word(CLASS[1], "and the", s[1], e[1]), "easy": s[2], "band": s[3], "star": s[4],
          "lift": s[5]}
    dur = math.ceil((e[-1] + opening.TAIL + 0.5) * 10) / 10
    events = (round(at["star"] + 0.8 + 3.5, 2), round(at["lift"] + 1.0 + 3.5, 2))   # the counter lands on 5, then on 1
    return {"dur": dur, "at": at, "caps": [[s[i], c] for i, c in enumerate(CLASS_CAPS)], "starts": s, "ends": e,
            "events": events}


def body():
    """{section: [(offset, text)]} and {section: heading} from VO_A_body.md"""
    s = open(os.path.join(HERE, "VO_A_body.md"), encoding="utf-8").read()
    out, heads = {}, {}
    for m in re.finditer(r"^## (\w+) · ([^\n]+)\n(.*?)(?=^## |\Z)", s, re.S | re.M):
        key = m.group(1)
        heads[key] = m.group(2).strip()
        out[key] = [(float(c.group(1)), " ".join(c.group(2).split()))
                    for c in re.finditer(r"\*\*\+(\d+(?:\.\d+)?)\*\*\s+(.+?)(?=\n\s*\n|\Z)", m.group(3), re.S)]
    return out, heads


def fit(cues, cache, length):
    """when each line really starts (a line that would run into the next pushes it, as narrate.build does), and when
    the last one ends"""
    free, real = 0.0, []
    for off, txt in cues:
        st = max(off + 0.15, free + 0.35)
        n = opening.clip_len(txt, cache)
        real.append(st)
        free = st + n
    return real, free


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    import narrate
    narrate.SAY = opening.SAY.get("a", {})
    cache = os.path.join(HERE, "out_a", "voice")
    os.makedirs(cache, exist_ok=True)
    op = json.load(open(os.path.join(HERE, "opening_a.json"), encoding="utf-8"))
    hook, cls = plan_hook(cache), plan_class(cache)
    B, heads = body()
    T = {"hook": hook, "curve": cls, "estimate": bool(os.environ.get("NARRATE_ESTIMATE"))}
    json.dump(T, open(os.path.join(HERE, "timed_a.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    sys.argv = [sys.argv[0], "x"]
    import afilm, common                                    # the timeline, with the new lengths
    O, durs, total = common.timeline(afilm)
    lines = [f"# Narration script: The 10 Biggest Single-Game Outliers in Team Sports", "",
             f"Timecodes are positions in the final cut ({round(total) // 60}:{round(total) % 60:02d}), in seconds "
             "and tenths. Written by script_a.py from opening.py (the opening), script_a.py (the hook and grading on a "
             "curve) and VO_A_body.md (the rest). Names are spelled normally here; say.json respells a few for the "
             "voice. Never say \"proved\", \"greatest athlete\" or \"certain\".", ""]
    if T["estimate"]:
        lines[2:2] = ["**Timing estimated**: some lines are not voiced yet, so they are timed by their length. Run "
                      "script_a.py again once they are voiced.", ""]
    lines += ["## Opening: why eras matter, and how the score allows for them (0:00)", ""]
    lines += [f"**{mmss(t)}** {ln}\n" for t, ln in op["cues"]]
    lines += [f"## Hook ({mmss(O['hook'])})", ""]
    lines += [f"**{mmss(O['hook'] + st)}** {ln}\n" for st, ln in zip(hook["starts"], HOOK)]
    report = []
    for k in afilm.ORDER:
        if k in B:
            lines += [f"## {heads[k]} ({mmss(O[k])})", ""]
            lines += [f"**{mmss(O[k] + off)}** {txt}\n" for off, txt in B[k]]
            real, end = fit(B[k], cache, durs[k])
            late = [round(r - off - 0.15, 1) for r, (off, _) in zip(real, B[k]) if r - off - 0.15 > 0.3]
            report.append((k, round(end, 1), durs[k], late))
        elif k == "curve":
            lines += [f"## How the score works: grading on a curve ({mmss(O[k])})", ""]
            lines += [f"**{mmss(O[k] + st)}** {ln}\n" for st, ln in zip(cls["starts"], CLASS)]
    open(os.path.join(HERE, "VO_A.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"total {total:.1f} s; hook {hook['dur']} s, curve {cls['dur']} s" +
          ("  (ESTIMATED: lines not yet voiced are timed by their length)" if T["estimate"] else ""))
    for k, end, d, late in report:
        flag = "  OVER" if end > d - 0.8 else ""
        print(f"  {k:6s} last line ends {end:5.1f} of {d:.0f} s{flag}" + (f"; pushed late {late}" if late else ""))
    return total


if __name__ == "__main__":
    main()
