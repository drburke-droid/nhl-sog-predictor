"""The 10 Biggest Single-Day Outliers in Sports History: the sections of the film.

    python sdfilm.py <section>       # hook c10 ... c1 sittler curve fair close end -> out/<section>.mp4
    python sdfilm.py stills          # review stills -> out/stills/

The season film's machinery (../../outliers/film.py, scenes.py) with this film's facts (facts_sg.json): every number on
screen comes from there. Cards use the team's or country's colours and sport-specific charts (a javelin sector, a
pool, crossbars, a shot-put sector, the jumping pit, stacked basketballs).
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REC, "outliers"))
import film as FM                                           # noqa: E402
import look                                                 # noqa: E402
import scenes as SC                                         # noqa: E402
from look import W, H, INK, INK2, MUTED, GOLD, fade, ease, clamp, text, title   # noqa: E402

look.OUT = os.path.join(HERE, "out")
FS = json.load(open(os.path.join(HERE, "facts_sg.json"), encoding="utf-8"))
CARDS = {c["rank"]: c for c in FS["cards"]}
SC_CREDITS = json.load(open(os.path.join(HERE, "photos", "credits.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(HERE, "photos", "credits.json")) else {}
CREDITS = dict(FM.CREDITS)
CREDITS.update(SC_CREDITS)

sc = FS["scale"]
FS["hook_words"] = {"headline": f"{sc['n']:,} single-day bests.",
                    "sub": f"{sc['n_categories']} categories · one game, one race, one jump at a time",
                    "graded": "Each one graded against the best day of everyone else in its era.",
                    "title2": "Single-Day Outliers"}
FS["fair_rules"] = [
    (1.0, "Adjusted for inflation",
     "When every team scores more, each point is worth a little less, like money in an inflated economy. So in team "
     f"sports, everyone else's games are converted into the athlete's own season before they're compared. Wilt's 100 "
     f"came when teams averaged {FS['extras']['nba_points_per_team_game']['1962']:.1f} points a game; Kobe's 81 when they "
     f"averaged {FS['extras']['nba_points_per_team_game']['2006']:.1f}."),
    (12.0, "Graded against rivals, not yourself",
     "An athlete's own other days are left out of the class. Otherwise Wilt's 100 would be graded against Wilt's 78."),
    (20.0, "One entry each, and only big numbers",
     "Each athlete appears once, with their highest-scoring day. Team stats that come in small whole numbers (goals, "
     "home runs, hits) are left out: one more of them can't be measured fairly.")]
FM.F = FS
FM.CARDS = CARDS
FM.CREDITS = CREDITS
FM.RULER_UNIT.update({"ath_M_JT": "m", "ath_M_SP": "m", "ath_M_PV": "m", "ath_W_TJ": "m", "ath_M_LJ": "cm",
                      "swim_W_800_fr": "s", "swim_M_200_me": "s", "sg_nba_pts": "points"})


def mmss(sec):
    m = int(sec // 60)
    return f"{m}:{sec - 60 * m:05.2f}" if m else f"{sec:.2f}"


for c in FS["cards"]:                                       # swimming: say it in seconds, rounded so the sum holds
    if "swim" in c:
        sw = c["swim"]
        tb, tx = round(sw["curve_time"], 2), round(sw["time"], 2)
        gap = tb - tx
        for d in (2, 3, 4):
            st = round(gap / c["z_e3"], d) if False else round(sw["step_s"], d)
            if abs(round(gap / st, 2) - c["z_e3"]) < 1e-9:
                break
        c["arith"] = f"({mmss(tb)} − {mmss(tx)}) ÷ {st} s = {c['z_e3']:.2f}"

_orig_nums = FM.ruler_numbers
FM.ruler_numbers = lambda c: ("", "", "") if c.get("arith") else _orig_nums(c)


def card_lines(c):
    r, ex = c["rank"], FS["extras"]
    ru, pr, nr = c["runner_up"], c["previous_record"], c["next_record"]
    if r == 10:
        return [f"Next best in 2020: {ru['name']}, {float(ru['value']):.2f} m, {97.76 - float(ru['value']):.2f} m behind.",
                f"Only {pr['holder']}'s {float(pr['value']):.2f} m world record, from {pr['year']}, has gone further.",
                "Thrown in Chorzów, Poland, in September 2020."]
    if r == 9:
        return ["World Championships, Singapore, July 2025.",
                f"It broke {pr['holder']}'s world record of {mmss(float(pr['value']))}, which had stood since {pr['year']}.",
                f"Next best in 2025: {ru['name']}, {mmss(float(ru['value']))}. 2025 has only past seasons to compare with."]
    if r == 8:
        return ["Indoors, in Donetsk, in February 1993.",
                f"It stood as the world record until {nr['holder']}'s {float(nr['value']):.2f} m in {nr['year']}.",
                f"Next best in 1993: {ru['name']}, {float(ru['value']):.2f} m."]
    if r == 7:
        return ["Thrown at a professional meet in El Paso, Texas, in May 1975: never ratified, as professionals' marks "
                "didn't count.",
                f"The official world record was {float(pr['value']):.2f} m ({pr['holder']}, {pr['year']}): "
                f"{22.86 - float(pr['value']):.2f} m shorter.",
                f"Next best in 1975: {ru['name']}, {float(ru['value']):.2f} m."]
    if r == 6:
        return ["Tokyo Olympic final, August 2021.",
                f"It beat {pr['holder']}'s world record of {float(pr['value']):.2f} m, which had stood since {pr['year']}, "
                f"by {100 * (15.67 - float(pr['value'])):.0f} cm.",
                f"Next best in 2021: {ru['name']}, {float(ru['value']):.2f} m."]
    if r == 5:
        return [f"Rio Olympic final, August 2016: {float(ru['value']) - 484.79:.2f} seconds ahead of the silver medallist.",
                f"It lowered her own world record, {mmss(486.68)} from January, by {486.68 - 484.79:.2f} seconds.",
                f"She broke it again in {nr['year']}: {mmss(float(nr['value']))}."]
    if r == 4:
        return ["Against Toronto, 22 January 2006: 55 of them in the second half.",
                "Only Wilt Chamberlain has scored more in an NBA game.",
                f"Next best that season: {ru['name']}, {float(ru['value']):.0f}. Teams averaged "
                f"{ex['nba_points_per_team_game']['2006']:.1f} points."]
    if r == 3:
        return ["98.48 m with the javelin redesigned in 1986 — still the world record.",
                f"Next best in 1996: {ru['name']}, {float(ru['value']):.2f} m.",
                f"It broke his own record of {float(pr['value']):.2f} m, set in {pr['year']}."]
    if r == 2:
        b, bd = ex["wilt_best_before_100"]
        return ["Hershey, Pennsylvania, 2 March 1962, against the New York Knicks.",
                f"Three months earlier he had set the record at {b:.0f}. No one else that season scored more than "
                f"{float(ru['value']):.0f}.",
                f"Teams averaged {ex['nba_points_per_team_game']['1962']:.1f} points; allowing for that, it still scores "
                f"{c['z_e3']:.2f}."]
    if r == 1:
        return [f"One jump: {100 * (8.90 - float(pr['value'])):.0f} cm beyond the world record.",
                f"Next best that year: {float(ru['value']):.2f} m. The record stood until {nr['holder']}'s "
                f"{float(nr['value']):.2f} m in {nr['year']}.",
                "It came at 2,240 m altitude, with a +2.0 m/s tailwind, the legal maximum."]
    return []


FM.card_lines = card_lines


# ------------------------------------------------------------------ why hockey and home runs aren't here
class Sittler:
    dur = 48.0

    def __init__(self):
        s = FS["sittler"]
        self.s = s
        c = {"rank": 0, "category": "sg_nhl_points", "year": 1976, "season": "1975–76", "athlete": "Darryl Sittler",
             "games_per_team": None, "z_e3": s["z"], "palette": "leafs",
             "ruler": {"window": s["window"], "x": 10.0, "base": s["base"], "step": s["step"], "trend": s["trend"],
                       "converted": True},
             "scene": dict(kind="stack", icon="puck", per=1, word="points in a game", key="one puck = 1 point")}
        self.scene = SC.make_scene(c)
        bg = SC.background("leafs").astype("float32")
        self.plate = (bg * 0.85 + 0.15 * 15).astype("uint8")

    def draw(self, fig, t):
        s, w = self.s, FS["whole_numbers"]
        al = fade(t, 0, self.dur, 0.6, 0.7)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.imshow(self.plate, extent=(0, W, H, 0), alpha=al, zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        text(fig, 96, 92, "WHY HOCKEY AND HOME RUNS AREN'T ON THIS LIST", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "When the numbers come in ones", size=58, alpha=al)
        self.scene.draw(fig, t, al)
        ev = ", ".join(n.split()[-1] for _, n in s["eights"][:-1]) + f" and {s['eights'][-1][1].split()[-1]}"
        caps = [(1.0, "Darryl Sittler, 7 February 1976: 10 points in one game, six goals and four assists. Still the NHL "
                      "record."),
                (7.0, f"But in the seasons around it, everyone else's best night was 7 points {s['sevens']} times, and 8 "
                      f"points {len(s['eights'])} times ({ev})."),
                (14.5, f"So the curve sits near {s['base']:.0f} points and one step is about {s['step']:.1f} points. Ten is "
                       f"{s['z']:.1f} steps: {s['rank_all']}th of everything, not top ten."),
                (22.0, "Goals, points, home runs and hits come in ones. When most seasons' best night is the same small "
                       "number, one more can look enormous, or like nothing at all."),
                (30.0, f"Johnny Burnett's 9 hits (in an 18-inning game) would score {w['burnett_z']:.1f}; Mike Schmidt's "
                       f"4 home runs {w['schmidt_z']:.1f}. We left those stats out."),
                (38.0, "What's left come in big numbers: points, rebounds, assists, strikeouts, yards, and every "
                       "athletics and swimming mark.")]
        y = 330
        for t0, msg in caps:
            a = al * fade(t, t0, 99, 0.8, 0)
            if a > 0:
                y = FM.para(fig, 1230, y, msg, width=42, size=24, color=INK if t0 == max(c[0] for c in caps if c[0] <= t) else INK2,
                            alpha=a) + 16


# ------------------------------------------------------------------ the close: a season against a day
class Close:
    dur = 50.0

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.8)
        a1 = fade(t, 0, 32.0, 0.6, 0.7) * al
        L = FS["lists"]
        if a1 > 0:
            text(fig, 96, 92, "THE SAME METHOD, TWO QUESTIONS", size=21, color=MUTED, weight="semibold", alpha=a1)
            title(fig, 96, 160, "Over a season, or on a single day", size=58, alpha=a1)
            for col, (key, head, x) in enumerate((("season", "Over a season", 96), ("day", "On a single day", 980))):
                text(fig, x, 250, head.upper(), size=21, color=GOLD, weight="semibold", alpha=a1)
                rows = sorted(L[key], key=lambda r: r[0])
                for i, (rk, name, ach, season, z) in enumerate(rows):
                    a = a1 * fade(t, 1.0 + 0.25 * i + 1.5 * col, 99, 0.5, 0)
                    yy = 300 + i * 50
                    shared = name in [r[1] for r in L["day" if key == "season" else "season"]]
                    text(fig, x, yy, f"{rk}", size=26, color=GOLD if shared else MUTED, weight="semibold", alpha=a)
                    text(fig, x + 50, yy, name, size=26, color=INK, weight="semibold" if shared else "regular", alpha=a)
                    text(fig, x + 360, yy, f"{ach.replace('individual medley', 'medley')}, {season}", size=22, color=INK2, alpha=a)
                    text(fig, x + 820, yy, f"{z:.2f}", size=22, color=MUTED, ha="right", alpha=a)
            FM.para(fig, 96, 850, "Over a season, consistency stands out: Gretzky, DiMaggio, Ruth, Curry. Over a single game, "
                                  "team sports are noisy, and only Wilt Chamberlain's 100 and Kobe Bryant's 81 clear the "
                                  "bar. Names in gold are on both lists.", width=110, size=27, color=INK,
                    alpha=a1 * fade(t, 8.0, 99, 0.8, 0))
            sp = FS["robustness"]["in_top10_specs"]
            FM.para(fig, 96, 960, f"The top four stayed in the top ten in at least {min(sp[n] for n in list(sp)[:4])} of 16 "
                                  "versions of the method; places 5 to 10 change with small choices.", width=120, size=23,
                    color=INK2, alpha=a1 * fade(t, 16.0, 99, 0.8, 0))
        a3 = fade(t, 33.0, self.dur, 0.8, 0.8) * al
        if a3 > 0:
            title(fig, W / 2, 400, "Not a ranking of the greatest athletes.", size=60, ha="center", alpha=a3)
            title(fig, W / 2, 500, "A ranking of how far one day stood", size=60, ha="center", alpha=a3 * fade(t, 34.5, 99, 0.8, 0))
            title(fig, W / 2, 580, "from everyone else of its time.", size=60, ha="center", color=GOLD,
                  alpha=a3 * fade(t, 34.5, 99, 0.8, 0))
            text(fig, W / 2, 720, "Which would you choose: the season or the day?", size=36, color=INK2, ha="center",
                 alpha=a3 * fade(t, 39.0, 99, 0.8, 0))


class End:
    dur = 26.0

    def draw(self, fig, t):
        self.credits(fig, t)
        al = fade(t, 0, 12.5, 0.8, 0.8)
        if al <= 0:
            return
        sc = FS["scale"]
        text(fig, 96, 120, "SOURCES AND METHOD", size=21, color=MUTED, weight="semibold", alpha=al)
        lines = [
            ("Data", "NBA box scores: Eoin Moore, Historical NBA Data and Player Box Scores (CC0), via Hugging Face. NHL: "
                     "NHL_DB (Hugging Face), from NHL.com game data. MLB: Retrosheet. NFL: nflverse. Athletics and "
                     "swimming: World Athletics, World Aquatics, alltime-athletics.com (Peter Larsson)."),
            ("Model", f"{sc['n']:,} single-day bests in {sc['n_categories']} categories. Each is compared with the best day "
                      "by anyone else in the 15 seasons before and after, around a curved trend, with a league-scoring "
                      "adjustment for team sports (estimated from the data). Team stats that come in small whole numbers "
                      "are left out."),
            ("Caveats", "Single games include overtime and extra innings. NBA box scores before 1955–56 are incomplete and "
                        "NFL games before 1999 aren't openly available, so those seasons are left out. Scores are "
                        "statistical distances, not probabilities or measures of ability."),
            ("Retrosheet", "The information used here was obtained free of charge from and is copyrighted by Retrosheet. "
                           "Interested parties may contact Retrosheet at www.retrosheet.org."),
        ]
        y = 200
        for i, (h, s) in enumerate(lines):
            a = al * fade(t, 0.4 + 0.5 * i, 99, 0.6, 0)
            text(fig, 96, y + 26, h, size=26, color=GOLD, weight="semibold", alpha=a)
            y = FM.para(fig, 330, y, s, width=92, size=25, color=INK2, alpha=a) + 30

    def credits(self, fig, t):
        al = fade(t, 13.0, self.dur, 0.8, 1.2)
        if al <= 0:
            return
        text(fig, 96, 120, "PHOTOGRAPHS", size=21, color=MUTED, weight="semibold", alpha=al)
        order = [CARDS[r]["photo"] for r in range(1, 11)]
        for i, k in enumerate(order):
            cr = CREDITS.get(k)
            if not cr:
                continue
            col, row = i % 2, i // 2
            x, y = 96 + col * 880, 200 + row * 130
            a = al * fade(t, 13.3 + 0.12 * i, 99, 0.5, 0)
            try:
                FM.photo(fig, k, x, y, x + 72, y + 90, alpha=a, credit=False)
            except (FileNotFoundError, KeyError):
                pass
            text(fig, x + 96, y + 30, cr["athlete"], size=24, color=INK, weight="semibold", alpha=a)
            lic = cr["license"] if cr["license"] != "Public domain" else "public domain"
            text(fig, x + 96, y + 62, f"{cr['author']} · {lic}", size=20, color=INK2, alpha=a)
            text(fig, x + 96, y + 88, cr["file"][:70], size=16, color=MUTED, alpha=a)
        FM.para(fig, 96, 900, "All from Wikimedia Commons (commons.wikimedia.org): cropped, backgrounds removed, toned in team "
                              "colours. Licences: creativecommons.org/licenses (CC BY, CC BY-SA, CC0) or public domain. Full "
                              "credits and links are in the description. Icons: BlenderKit models.", width=120, size=20,
                color=MUTED, alpha=al)


SECTIONS = dict(FM.SECTIONS)
SECTIONS["sittler"] = Sittler
SECTIONS["close"] = Close
SECTIONS["end"] = End
ORDER = ["hook"] + [f"c{r}" for r in range(10, 0, -1)] + ["sittler", "curve", "fair", "close", "end"]


def stills():
    os.makedirs(os.path.join(look.OUT, "stills"), exist_ok=True)
    picks = {"hook": [19.5], "sittler": [40.0], "close": [20.0, 42.0], "end": [8.0, 20.0], "fair": [34.0]}
    for name in ORDER:
        sec = SECTIONS[name]()
        for t in picks.get(name, [33.0] if name[0] == "c" and name[1:].isdigit() else [sec.dur / 2]):
            p = os.path.join(look.OUT, "stills", f"{ORDER.index(name):02d}_{name}_{t:05.1f}.png")
            look.still(sec, t, p)
            print(p, flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "stills":
        stills()
    else:
        print(look.render_section(which, SECTIONS[which]()))
