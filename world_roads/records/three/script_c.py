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

FILM = timing.Film(
    "c", "The 5 Biggest Season-Long Outliers in Team Sports", HOOK,
    {7.2: (1, None), 10.0: (2, None), 12.4: (3, None), 18.0: (4, None)},
    {"graded": "Each one graded against the best of its era.",
     "most": "Most land near zero: right on the elite curve. These are the league leaders, the best in their sport.",
     "five": f"{SC['at_least_5']} of {SC['n']:,} land 5 or more standard deviations above it.",
     "ten": "Five stood well clear of everyone else.",
     "axis": "score: standard deviations above the elite curve"},
    {"opening": "opening", "hook": HOOK, "cards": ("body", [f"c{r}" for r in range(5, 0, -1)]),
     "rules": ("body", ["curve", "fair", "sure", "considered", "close"])})

if __name__ == "__main__":
    timing.cli(FILM, "cfilm")
