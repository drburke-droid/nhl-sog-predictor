"""Film C's narration, timed (timing.py): the opening (opening.py), the hook timed to its lines, and the rest from
VO_C_body.md, in four chunks voiced on their own.

    python script_c.py status                        # what each chunk would cost to voice
    ELEVENLABS_API_KEY=... python script_c.py voice hook     # voice a chunk (opening hook cards rules) + previews
    python opening.py c && python script_c.py        # once all are voiced: timed_c.json and VO_C.md
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timing                                               # noqa: E402

SC = json.load(open(os.path.join(HERE, "facts_c.json"), encoding="utf-8"))["scale"]

HOOK = [
    "I found and isolated two thousand six hundred and seven season-leading performances, in basketball, hockey, "
    "baseball and football, and set out to determine which ones were the most impressive.",
    "Each one graded against the best of its era.",
    "Most land near zero: right on the elite curve. That makes sense: these are the league leaders, the best in their "
    "sport.",
    "But ten landed five or more standard deviations above it.",
    "And five of them stood well clear of everyone else.",
    "These are the five biggest season-long outliers in team sports, counting down from five."]

# grading on a curve: a class test built the way the film is (only the best score by everyone else counts, in this
# class and in each of the 15 classes before and after), timed to these lines (classtest.ClassTestBest)
CLASS = [
    "Here's the same idea as a class test. One student scores ninety-five. Leave them out, and look at the rest of the "
    "class: its best score is about eighty.",
    "That best score is the only one from this class that counts. Not the class average: the best by everyone else.",
    "Now do the same for the fifteen classes before and the fifteen after. From each class, keep only its best score.",
    "Those best scores all land around eighty, and the line through them is the elite curve.",
    "If one year's test was easier for everyone, its scores are scaled down first, just as every season is converted "
    "to the same scoring level.",
    "The best scores usually sit within about three points of the elite curve. That's one standard deviation.",
    "So the student's ninety-five is fifteen points above the elite curve: five standard deviations.",
    "But if the best scores in those years had been around ninety-two, the same ninety-five would be just one standard "
    "deviation up.",
    "Same score, different era. So every season is graded against the best of everyone else in its own era."]
CLASS_CAPS = [
    "Picture a class test. One student scores 95. Leave them out: the best score by everyone else is about 80.",
    "Only that best score counts, not the class average.",
    "Do the same for the 15 classes before and the 15 after: keep only each class's best score.",
    "The line through those best scores is the elite curve.",
    "A year with an easier test is scaled down first, just as every season is converted to the same scoring level.",
    "The best scores usually sit within about 3 points of the elite curve: one standard deviation.",
    "The 95 is 15 points above the elite curve: 15\u00a0÷\u00a03\u00a0=\u00a05 standard deviations. On our scale, a score "
    "of 5.",
    "If the best scores around it had been about 92, the same 95 would be (95\u00a0−\u00a092)\u00a0÷\u00a03\u00a0=\u00a01 "
    "standard deviation up: a score of 1.",
    "Same score, different era. So every season in this film is graded against the best of everyone else in its own era."]
CLS = {"lines": CLASS, "caps": CLASS_CAPS, "pages": [2, 6],
       "at": {"s95": (0, "One student"), "top5": (0, "Leave them out"), "avg": (1, "Not the class average"),
              "others": (2, None), "bests": (2, "From each class"), "curve": (3, "and the line"), "easy": (4, None),
              "band": (5, None), "star": (6, None), "lift": (7, None)},
       "extra": {"curve_word": "the elite curve"}}

FILM = timing.Film(
    "c", "The 5 Biggest Season-Long Outliers in Team Sports", HOOK,
    {7.2: (1, None), 10.0: (2, None), 12.4: (3, None), 18.0: (4, None)},
    {"graded": "Each one graded against the best of its era.",
     "most": "Most land near zero: right on the elite curve. These are the league leaders, the best in their sport.",
     "five": f"{SC['at_least_5']} of {SC['n']:,} land 5 or more standard deviations above it.",
     "ten": "Five stood well clear of everyone else.",
     "axis": "score: standard deviations above the elite curve"},
    {"opening": "opening", "hook": HOOK, "cards": ("body", [f"c{r}" for r in range(5, 0, -1)]), "curve": CLASS,
     "rules": ("body", ["fair", "sure", "considered", "close"])}, cls=CLS)

if __name__ == "__main__":
    timing.cli(FILM, "cfilm")
