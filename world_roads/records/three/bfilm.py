"""Film B: The 10 Biggest Single Performances in Athletics and Swimming.

    python bfilm.py <section>       # hook c10 ... c1 curve fair sure close end -> out_b/<section>.mp4
    python bfilm.py stills          # review stills -> out_b/stills/
    python bfilm.py assemble        # out_b/film_b_final.mp4

The season film's machinery (../outliers/film.py, scenes.py) with this film's facts (facts_b.json, python facts_b.py).
Seven of the ten marks are also in the single-day film and use its charts; Duplantis (crossbars), McIntosh (the pool)
and Bolt (the track) are new.
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

OUT = os.path.join(HERE, "out_b")
look.OUT = OUT
FB = json.load(open(os.path.join(HERE, "facts_b.json"), encoding="utf-8"))
CARDS = {c["rank"]: c for c in FB["cards"]}
SC.PALETTE.update({
    "sweden": dict(team="Sweden", bg="#004B87", accent="#FECC02", light="#FFF5CC"),
    "canada": dict(team="Canada", bg="#7A0C14", accent="#F2F2F2", light="#F6DADA"),
    "jamaica": dict(team="Jamaica", bg="#00552B", accent="#FED100", light="#E4F4E8"),
})
SC.FRAMING.update({"duplantis": (940, 1565, 1100), "mcintosh": (960, 1565, 1100), "bolt": (980, 1565, 1100)})
CREDITS = {}
for p in (os.path.join(REC, "outliers", "photos", "credits.json"), os.path.join(REC, "singlegame", "film", "photos", "credits.json"),
          os.path.join(HERE, "photos", "credits.json")):
    if os.path.exists(p):
        CREDITS.update(json.load(open(p, encoding="utf-8")))


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


sc = FB["scale"]
_ex = {e["athlete"]: e for e in FB["excluded"]}
_led = _ex["Katie Ledecky"]
FB["hook_words"] = {"headline": f"{sc['n']:,} season-best marks.",
                    "sub": f"{sc['n_categories']} events · athletics and swimming · {sc['first_season']} to {sc['last_season']}",
                    "graded": "Each one graded against the best mark of everyone else in its era.",
                    "ten": "Ten stood clear of the field.", "title1": "The 10 Biggest", "title2": "Single Performances",
                    "title3": "in Athletics and Swimming", "countdown": "Counting down from ten"}
FB["fair_title"] = "Four rules that keep it fair"
FB.setdefault("extras", {})
FB["fair_rules"] = [
    (1.0, "Graded against rivals, not yourself",
     "An athlete's own other seasons are left out of the class. Otherwise Bubka's 1993 would be graded against Bubka."),
    (9.0, "One entry each",
     "Each athlete appears once, with their highest-scoring mark. Every entry is one race, one jump or one throw."),
    (17.0, "The supersuit years",
     "In 2008 and 2009 swimmers raced in polyurethane suits, banned from 2010. The model treats those two seasons as a "
     "shift, so suit-era times don't set the bar for everyone after."),
    (26.0, "Clear of the field",
     "A mark counts only if it beats every other mark on its own graph by at least one step. Katie Ledecky's "
     f"{mmss(float(_led["value"]))} in the 800 m freestyle (2016) scores {_led['z']:.2f}, but "
     f"{_led['best_other']['name']} swam {mmss(float(_led['best_other']['value']))} in {_led['best_other']['year']}: "
     "within a step. It's left out.")]
FM.F = FB
FM.CARDS = CARDS
FM.CREDITS = CREDITS
FM.RULER_UNIT.update({"ath_M_JT": "m", "ath_M_PV": "m", "ath_W_TJ": "m", "ath_M_LJ": "cm", "swim_M_200_me": "s",
                      "swim_W_100_bu": "s", "swim_W_400_me": "s", "ath_M_100m": "s"})

for c in FB["cards"]:                                       # races: say it in seconds, rounded so the sum holds
    if "swim" in c:
        sw = c["swim"]
        tb, tx = round(sw["curve_time"], 2), round(sw["time"], 2)
        gap = tb - tx
        for d in (2, 3, 4):
            st = round(sw["step_s"], d)
            if abs(round(gap / st, 2) - c["z_e3"]) < 1e-9:
                break
        c["arith"] = f"({mmss(tb)} − {mmss(tx)}) ÷ {st} s = {c['z_e3']:.2f}"

_orig_nums = FM.ruler_numbers
FM.ruler_numbers = lambda c: ("", "", "") if c.get("arith") else _orig_nums(c)


def card_lines(c):
    k = c["photo"]
    ru, pr, nr = c["runner_up"], c["previous_record"], c["next_record"]
    if k == "bolt":
        return ["World Championships, Berlin, 16 August 2009.",
                f"It took {float(pr['value']) - 9.58:.2f} seconds off his own world record of {float(pr['value']):.2f}, "
                f"set in {pr['year']}. It still stands.",
                f"Next best in 2009: {ru['name']}, {float(ru['value']):.2f}."]
    if k == "mcintosh":
        return ["Canadian Trials, Victoria, June 2025: a world record.",
                f"It broke her own record of {mmss(float(pr['value']))}, from {pr['year']}.",
                f"Next best in 2025: {ru['name']}, {mmss(float(ru['value']))}. Only one later season to compare with."]
    if k == "duplantis":
        return [f"World Championships, Tokyo, September 2025: a world record.",
                f"His record going into 2025 was {float(pr['value']):.2f} m; he raised it to {float(nr['value']):.2f} m in "
                f"{nr['year']}.",
                f"Second in 2025: {ru['name']}, {float(ru['value']):.2f} m."]
    if k == "gretchen_walsh":
        rb = c["records_before"]
        first = next(i for i, p in enumerate(rb) if p["holder"] == "Gretchen Walsh")
        old = rb[first - 1]
        return [f"A world record in 2026, {float(pr['value']) - 54.33:.2f} seconds under her own {float(pr['value']):.2f} from "
                f"{pr['year']}.",
                f"Before her, {old['holder']}'s {float(old['value']):.2f} had stood since {old['year']}.",
                f"Next best in 2026: {ru['name']}, {float(ru['value']):.2f}. 2026 has only past seasons to compare with."]
    if k == "vetter":
        return [f"Next best in 2020: {ru['name']}, {float(ru['value']):.2f} m, {97.76 - float(ru['value']):.2f} m behind.",
                f"Only {pr['holder']}'s {float(pr['value']):.2f} m world record, from {pr['year']}, has gone further.",
                "Thrown in Chorzów, Poland, in September 2020."]
    if k == "marchand":
        return ["World Championships, Singapore, July 2025.",
                f"It broke {pr['holder']}'s world record of {mmss(float(pr['value']))}, which had stood since {pr['year']}.",
                f"Next best in 2025: {ru['name']}, {mmss(float(ru['value']))}. Only one later season to compare with."]
    if k == "bubka":
        return ["Indoors, in Donetsk, in February 1993.",
                f"It stood as the world record until {nr['holder']}'s {float(nr['value']):.2f} m in {nr['year']}.",
                f"Next best in 1993: {ru['name']}, {float(ru['value']):.2f} m."]
    if k == "rojas":
        return ["Tokyo Olympic final, August 2021.",
                f"It beat {pr['holder']}'s world record of {float(pr['value']):.2f} m, which had stood since {pr['year']}, "
                f"by {100 * (15.67 - float(pr['value'])):.0f} cm.",
                f"Next best in 2021: {ru['name']}, {float(ru['value']):.2f} m."]
    if k == "zelezny":
        return ["98.48 m with the javelin redesigned in 1986 — still the world record.",
                f"Next best in 1996: {ru['name']}, {float(ru['value']):.2f} m.",
                f"It broke his own record of {float(pr['value']):.2f} m, set in {pr['year']}."]
    if k == "beamon":
        return [f"One jump: {100 * (8.90 - float(pr['value'])):.0f} cm beyond the world record.",
                f"Next best that year: {float(ru['value']):.2f} m. The record stood until {nr['holder']}'s "
                f"{float(nr['value']):.2f} m in {nr['year']}.",
                "It came at 2,240 m altitude, with a +2.0 m/s tailwind, the legal maximum."]
    return []


FM.card_lines = card_lines


def board():
    R = FB["robustness"]
    em = next(b for b in R["beamon_out"] if b["window"] == 20)["best_other"]
    rows = []
    for r in range(1, 11):
        c = CARDS[r]
        rows.append({"rank": r, "athlete": c["athlete"], "achievement": c["achievement"].replace("individual medley", "medley"),
                     "season": c["season"], "z": c["z_e3"], "specs_in_top10": c["specs_in_top10"],
                     "hi": c["specs_in_top10"] >= 12, "face": SC.face_disc(c["photo"], c["palette"], 54)})
    n16 = [c["athlete"].split()[-1] for c in FB["cards"] if c["specs_in_top10"] == 16]
    caps = [(2.0, "We ranked every mark 16 ways: comparison windows of 7, 10, 15 or 20 seasons, straight or curved "
                  "trends."),
            (9.0, "Each square is a version that put the mark in its top ten."),
            (16.0, f"{' and '.join(n16)} made all 16. Five made at least 12 of 16: three versions in four."),
            (24.0, f"Beamon made {CARDS[1]['specs_in_top10']}: with the widest windows, {em['name']}'s "
                   f"{float(em['value']):.2f} m ({em['year']}) comes within a step of him.")]
    return common.Board(rows, "HOW SURE IS THIS LIST?", "16 ways to rank", caps)


class End:
    dur = 26.0

    def draw(self, fig, t):
        common.credits_page(fig, t, [CARDS[r]["photo"] for r in range(1, 11)], CREDITS)
        common.sources_page(fig, t, [
            ("Data", "World Athletics, World Aquatics, alltime-athletics.com (Peter Larsson): each event's season lists."),
            ("Model", f"{sc['n']:,} season-best marks in {sc['n_categories']} events. Each is compared with the best mark by "
                      "anyone else in the 15 seasons before and after, around a curved trend; swimming's 2008–09 supersuit "
                      "seasons are treated as a shift. A mark must clear every other mark on its own graph by at least "
                      "one step. Races are scored as speed."),
            ("Caveats", "Wind, altitude and equipment are not adjusted for. Scores are statistical distances, not "
                        "probabilities or measures of ability. Speed skating is left out: its season lists couldn't be "
                        "used."),
        ])


SECTIONS = {"hook": FM.SECTIONS["hook"], "curve": FM.SECTIONS["curve"], "fair": FM.SECTIONS["fair"], "sure": board,
            "close": lambda: common.ClosingLine("one performance", "Which would you choose?"), "end": End}
for _r in range(1, 11):
    SECTIONS[f"c{_r}"] = FM.SECTIONS[f"c{_r}"]
ORDER = ["hook"] + [f"c{r}" for r in range(10, 0, -1)] + ["curve", "fair", "sure", "close", "end"]
PICKS = {"hook": [14.0, 19.5, 26.0], "fair": [36.0], "sure": [26.0], "close": [12.0], "end": [8.0, 20.0], "curve": [24.0]}
FINAL = "film_b_final.mp4"

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
