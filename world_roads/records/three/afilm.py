"""Film A: The 10 Biggest Single-Game Outliers in Team Sports.

    python afilm.py <section>       # hook c10 ... c1 curve fair sure close end -> out_a/<section>.mp4
    python afilm.py stills          # review stills -> out_a/stills/
    python afilm.py assemble        # out_a/film_a_final.mp4

The season film's machinery (../outliers/film.py, scenes.py) with this film's facts (facts_a.json, python facts_a.py).
Every card uses the top-five chart (scenes.TopK): each season's five best games by anyone else, converted into the
athlete's season, the curve through them, and the athlete's game measured from half a unit below its count.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REC)
sys.path.insert(0, os.path.join(REC, "outliers"))
import film as FM                                           # noqa: E402
import look                                                 # noqa: E402
import scenes as SC                                         # noqa: E402
import common                                               # noqa: E402

OUT = os.path.join(HERE, "out_a")
look.OUT = OUT
FA = json.load(open(os.path.join(HERE, "facts_a.json"), encoding="utf-8"))
CARDS = {c["rank"]: c for c in FA["cards"]}
SC.PALETTE.update({
    "rams": dict(team="Los Angeles Rams", bg="#0A2F6B", accent="#FFC72C", light="#FFF1C9"),
    "cardinals": dict(team="St. Louis Cardinals", bg="#6B0F1A", accent="#FEDB00", light="#F6E3E6"),
    "lions": dict(team="Detroit Lions", bg="#00508C", accent="#C9D1D6", light="#E3ECF3"),
    "canadiens": dict(team="Montreal Canadiens", bg="#1A2366", accent="#E03A3E", light="#F2E3E4"),
    "chiefs": dict(team="Kansas City Chiefs", bg="#7A0B17", accent="#FFB612", light="#FBEAD0"),
})
SC.FRAMING.update({"box": (600, 1560, 640), "vanbrocklin": (720, 1610, 760), "olmstead": (760, 1600, 820)})  # clear of the score
CREDITS = {}
for p in (os.path.join(REC, "outliers", "photos", "credits.json"), os.path.join(REC, "singlegame", "film", "photos", "credits.json"),
          os.path.join(HERE, "photos", "credits.json")):
    if os.path.exists(p):
        CREDITS.update(json.load(open(p, encoding="utf-8")))

sc = FA["scale"]
_ex = {e["athlete"]: e for e in FA["excluded"]}
_cj = _ex["Calvin Johnson"]
_and = FA["anderson"]
FA["hook_words"] = {"headline": f"{sc['n']:,} season-leading games.",
                    "sub": f"{sc['n_categories']} categories · basketball, hockey, baseball, football · "
                           f"{sc['first_season']} to {sc['last_season']}",
                    "graded": "Each one graded against the best games of everyone else in its era.",
                    "ten": "Ten stood clear of the field.", "title1": "The 10 Biggest", "title2": "Single-Game Outliers",
                    "title3": "in Team Sports", "countdown": "Counting down from ten"}
# the score in standard deviations on screen too (the narration says "standard deviations", not "steps")
FA["score_label"] = "standard deviations above the curve"
SC.TOPK_TICK = "each tick: one standard deviation,\nthe usual spread of the best"
for _c in FA["cards"]:
    _c["friendly"] = _c["friendly"].replace("one step about", "standard deviation about")
FA["fair_title"] = "Four rules that keep it fair"
FA["fair_rules"] = [
    (1.0, "Every game at regulation length",
     "Overtime and extra innings are extra chances, so every game is scaled to regulation length. Flipper Anderson's "
     f"record {float(_and['raw']):.0f} receiving yards came in overtime; they count as {_and['performance']:.0f}."),
    (9.0, "Adjusted for inflation",
     "When every team scores more, each point is worth a little less. So other seasons' games are converted into the "
     f"athlete's own season, by at most {FA['cap']:g} times either way. Wilt's 100 came when teams averaged "
     f"{CARDS[1]['league_rate']:.1f} points a game; Kobe's 81 when they averaged {CARDS[2]['league_rate']:.1f}."),
    (18.0, "The five best, not just the best",
     "The class is the five best games by anyone else in each of the 15 seasons before and after. And a count is a "
     "count: 10 points stands for anything from 9.5 to 10.5, so it's measured from 9.5."),
    (26.0, "Clear of the field",
     "A game counts only if it beats every other game on its own graph by at least one standard deviation. Calvin "
     f"Johnson's {float(_cj['value']):.0f} yards (2013) score {_cj['z']:.2f}, but {_cj['best_other']['name']}'s "
     f"{float(_cj['best_other']['value']):.0f} in {_cj['best_other']['year']} is about "
     f"{_cj['best_other']['converted']:.0f} in 2013 terms: within one standard deviation. It's left out.")]
FA.setdefault("extras", {})
FM.F = FA
FM.CARDS = CARDS
FM.CREDITS = CREDITS
FM.ruler_numbers = lambda c: ("", "", "")                   # every card carries its own arithmetic (with the half unit)
FM.RULER_UNIT.update({c["category"]: "" for c in FA["cards"]})


def card_lines(c):
    k, ru, nr, bo = c["photo"], c["runner_up"], c["next_record"], c["best_other"]
    rv = f"{float(ru['value']):.0f}"
    if k == "chamberlain":
        return ["Hershey, Pennsylvania, 2 March 1962, against the New York Knicks.",
                f"Next best in {c['season']}: {ru['name']}, {rv}. Teams averaged {c['league_rate']:.1f} points a game.",
                f"The closest since: {bo['name']}'s {float(bo['value']):.0f} ({bo['season']}), about {bo['converted']:.0f} "
                "in 1961–62 terms."]
    if k == "bryant":
        return ["Against Toronto, 22 January 2006: 55 of them in the second half.",
                "Only Wilt Chamberlain has scored more in an NBA game.",
                f"Next best in {c['season']}: {ru['name']}, {rv}. Teams averaged {c['league_rate']:.1f} points."]
    if k == "vanbrocklin":
        return ["Against the New York Yanks, 28 September 1951. It's still the NFL record.",
                f"Next best in {c['season']}: {ru['name']}, {rv}.",
                f"The closest on the graph: {bo['name']}'s {float(bo['value']):.0f} ({bo['year']}), about "
                f"{bo['converted']:.0f} in 1951 terms."]
    if k == "bottomley":
        return ["At Brooklyn, 16 September 1924: six hits, two of them home runs.",
                "Twelve RBI is still the major-league record. Mark Whiten matched it in 1993.",
                f"Next best in {c['season']}: {ru['name']}, {rv}."]
    if k == "box":
        return ["At Baltimore, 3 December 1950.",
                f"Next best in {c['season']}: {ru['name']}, {rv}.",
                f"The closest on the graph: {bo['name']}'s {float(bo['value']):.0f} ({bo['year']}), about "
                f"{bo['converted']:.0f} in 1950 terms. Football here starts in 1950."]
    if k == "cousy":
        return ["For Boston against the Minneapolis Lakers, 27 February 1959.",
                f"Next best in {c['season']}: {ru['name']}, {rv}.",
                f"{bo['name']} matched 28 in {bo['season']}, when teams passed more: about {bo['converted']:.0f} in 1958–59 "
                f"terms. {nr['holder']}'s {float(nr['value']):.0f} broke it in {nr['year']}."]
    if k == "whiten":
        return ["At Cincinnati, 7 September 1993, in the second game of a doubleheader: four home runs.",
                "It tied Jim Bottomley's record of 12, set in 1924.",
                f"Next best in {c['season']}: {ru['name']}, {rv}."]
    if k == "sittler":
        return ["Against Boston, 7 February 1976: six goals and four assists.",
                "It's still the NHL record.",
                f"Next best in {c['season']}: {ru['name']}, {rv}. The closest on the graph: {bo['name']}'s "
                f"{float(bo['value']):.0f} ({bo['season']})."]
    if k == "olmstead":
        return ["Montreal against Chicago, 9 January 1954: four goals and four assists.",
                f"It matched {bo['name']}'s record of 8 from {bo['season']}: about {bo['converted']:.1f} in 1953–54 terms.",
                f"Next best in {c['season']}: {ru['name']}, {rv}."]
    if k == "dthomas":
        return ["Against Seattle, 11 November 1990. It's still the NFL record.",
                f"Sacks have been counted officially since 1982. The closest on the graph: {bo['name']}'s "
                f"{float(bo['value']):.0f} ({bo['year']}).",
                f"Next best in {c['season']}: {ru['name']}, {rv}."]
    return []


FM.card_lines = card_lines


def surname(name):
    p = name.split()
    return " ".join(p[-2:]) if len(p) > 2 and p[-2].lower() in ("van", "von", "de", "del", "da", "di", "le", "la") else p[-1]


def board():
    rows = []
    for r in range(1, 11):
        c = CARDS[r]
        rows.append({"rank": r, "athlete": c["athlete"], "achievement": c["achievement"], "season": c["season"],
                     "z": c["z_e3"], "specs_in_top10": c["specs_in_top10"], "hi": c["specs_in_top10"] >= 12,
                     "face": SC.face_disc(c["photo"], c["palette"], 54)})
    all16 = [surname(c["athlete"]) for c in FA["cards"] if c["specs_in_top10"] == 16]
    many = sum(c["specs_in_top10"] >= 12 for c in FA["cards"])
    low = sorted(FA["cards"], key=lambda c: c["specs_in_top10"])[:2]
    caps = [(2.0, "We ranked every game 16 ways: comparison windows of 7, 10, 15 or 20 seasons, straight or curved trends, "
                  "with or without the league adjustment."),
            (9.0, "Each square is a version that put the game in its top ten."),
            (16.0, f"{', '.join(all16[:-1])} and {all16[-1]} made all 16. {many} made at least 12."),
            (24.0, f"{surname(low[1]['athlete'])}'s {low[1]['achievement'].split()[0]} and "
                   f"{surname(low[0]['athlete'])}'s {low[0]['achievement'].split()[0]} depend most on the choices: "
                   f"{low[1]['specs_in_top10']} and {low[0]['specs_in_top10']} of 16.")]
    return common.Board(rows, "HOW SURE IS THIS LIST?", "16 ways to rank", caps)


class End:
    dur = 26.0

    def draw(self, fig, t):
        common.credits_page(fig, t, [CARDS[r]["photo"] for r in range(1, 11)], CREDITS)
        common.sources_page(fig, t, [
            ("Data", "NBA: Eoin Moore, Historical NBA Data and Player Box Scores (CC0), via Hugging Face. NHL: NHL_DB "
                     "(Hugging Face), from NHL.com game data. MLB: Retrosheet. NFL: Pro Football Archives box scores "
                     "(1950–1998) and nflverse."),
            ("Model", f"{sc['n']:,} season-leading games in {sc['n_categories']} categories, each scaled to regulation "
                      "length and compared with the five best games by anyone else in each of the 15 seasons before and "
                      "after, around a curved trend, converted by the league's scoring rate (at most 1.5 times). Counts "
                      "are measured from half a unit below; a game must clear every other game on its graph by one standard "
                      "deviation."),
            ("Caveats", "NBA box scores before 1955–56 are incomplete and NFL box scores before 1950 unreliable, so those "
                        "seasons are left out. Old NFL overtimes count as 7.5 minutes. Scores are statistical distances, "
                        "not probabilities or measures of ability."),
            ("Retrosheet", "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
                           "Interested parties may contact Retrosheet at www.retrosheet.org."),
        ])


# ------------------------------------------------------------------ the opening: why eras matter, and how the score allows for it
import intro as IN                                          # noqa: E402
_OP = json.load(open(os.path.join(HERE, "opening_a.json"), encoding="utf-8"))   # python opening.py a: timed to the narration
INTRO = IN.Intro([(n, d, s) for n, d, s in _OP["scenes"]])

import classtest                                            # noqa: E402
import numpy as _np                                         # noqa: E402
_T = json.load(open(os.path.join(HERE, "timed_a.json"), encoding="utf-8"))  # python script_a.py
FA["hook_words"].update(_T["hook"]["words"])
HOOK_MUSIC = _T["hook"]["music"]                            # the hook's music on the film's clock (common.build_score)
CURVE_EVENTS = tuple(_T["curve"]["events"])                 # sparkles as the class test's counter lands on 5, then 1


class HookA(FM.Hook):
    """the season film's hook on a longer clock: each part plays as its line is spoken (knots: hook time, film time)"""

    def __init__(self):
        super().__init__()
        k = _T["hook"]["knots"]
        self.own, self.film = _np.array([a for a, _ in k]), _np.array([b for _, b in k])
        self.dur = _T["hook"]["dur"]

    def draw(self, fig, t):
        d, self.dur = self.dur, FM.Hook.dur
        try:
            super().draw(fig, float(_np.interp(t, self.film, self.own)))
        finally:
            self.dur = d


class FairA(FM.Fair):
    dur = 43.0                                              # three seconds longer: rule 4's line is longer


HOOK_DUR, CURVE_DUR, FAIR_DUR = _T["hook"]["dur"], _T["curve"]["dur"], FairA.dur    # for copy3.py's timeline

SECTIONS = {"intro": lambda: INTRO, "hook": HookA, "curve": lambda: classtest.ClassTest(_T["curve"]), "fair": FairA, "sure": board,
            "close": lambda: common.ClosingLine("one game", "Which would you choose?"), "end": End}
for _r in range(1, 11):
    SECTIONS[f"c{_r}"] = FM.SECTIONS[f"c{_r}"]
ORDER = ["intro", "hook"] + [f"c{r}" for r in range(10, 0, -1)] + ["curve", "fair", "sure", "close", "end"]
_hk, _ca = _T["hook"]["knots"], _T["curve"]["at"]
PICKS = {"hook": [round(float(_np.interp(x, [a for a, _ in _hk], [b for _, b in _hk])), 1) for x in (14.0, 19.5, 26.0)],
         "fair": [36.0], "sure": [26.0], "close": [12.0], "end": [8.0, 20.0],
         "curve": [round(_ca["others"] + 4.0, 1), round(_ca["easy"] + 1.5, 1), round(_ca["band"] + 3.0, 1),
                   round(_ca["star"] + 5.0, 1), round(_ca["lift"] + 5.5, 1)]}
FINAL = "film_a_final.mp4"

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "stills":
        common.stills(sys.modules[__name__], PICKS, OUT)
    elif which == "voiced":                                 # remix the voiced copy only (no re-render)
        common.voiced(OUT, FINAL)
    elif which == "assemble":
        common.assemble(sys.modules[__name__], OUT, FINAL, common.ClosingLine.FINALE)
    else:
        print(look.render_section(which, SECTIONS[which]()))
