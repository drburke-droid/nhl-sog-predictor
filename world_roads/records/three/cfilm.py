"""Film C: The 5 Biggest Season-Long Outliers in Team Sports.

    python cfilm.py <section>       # hook c5 ... c1 curve fair sure considered close end -> out_c/<section>.mp4
    python cfilm.py stills          # review stills -> out_c/stills/
    python cfilm.py assemble        # out_c/film_c_final.mp4 (sections joined, score laid in)

The season film's machinery (../outliers/film.py, scenes.py) with this film's facts (facts_c.json, python facts_c.py).
The five cards are the season film's own cards for these seasons (same chart, ruler and score); the film adds the clear-
of-the-field rule, how sure the list is (16 versions of the method), and the five that came close.
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
from look import W, H, INK, INK2, MUTED, GOLD, BLUE, fade, ease, clamp, text, title   # noqa: E402

OUT = os.path.join(HERE, "out_c")
look.OUT = OUT
FC = json.load(open(os.path.join(HERE, "facts_c.json"), encoding="utf-8"))
CARDS = {c["rank"]: c for c in FC["cards"]}
N = FC["n_cards"]
SC.PALETTE.update({
    "indians": dict(team="Cleveland Indians", bg="#0C2340", accent="#E31937", light="#E6E9EE"),
    "dodgers": dict(team="Los Angeles Dodgers", bg="#005A9C", accent="#EF3E42", light="#E3EEF8"),
    "redsox": dict(team="Boston Red Sox", bg="#0C2340", accent="#BD3039", light="#ECE6E6"),
})
CREDITS = dict(FM.CREDITS)
_cp = os.path.join(HERE, "photos", "credits.json")
if os.path.exists(_cp):
    CREDITS.update(json.load(open(_cp, encoding="utf-8")))

sc = FC["scale"]
ex = FC["extras"]
cu = FC["curry"]
FC["hook_words"] = {"headline": f"{sc['n']:,} season-leading performances.",
                    "sub": f"{sc['n_categories']} categories · basketball, hockey, baseball, football · "
                           f"{sc['first_season']} to {sc['last_season']}",
                    "graded": "Each one graded against everyone else of its era.",
                    "ten": "Five stood clear of the field.", "title1": "The 5 Biggest", "title2": "Season-Long Outliers",
                    "title3": "in Team Sports", "countdown": "Counting down from five"}
FC["fair_title"] = "Four rules that keep it fair"
FC["fair_rules"] = [
    (1.0, "Adjusted for inflation",
     "When every team scores more, each point is worth a little less, like money in an inflated economy. So everyone "
     "else's seasons are converted into the athlete's own season before they're compared. Wilt's 50.4 came when teams "
     f"averaged {ex['nba_points_per_team_game_1962']:.1f} points a game. How much to convert is estimated from the data."),
    (12.0, "Graded against rivals, not yourself",
     "An athlete's own other seasons are left out of the class. Otherwise Gretzky's 1985–86 would be graded against "
     "Gretzky."),
    (20.0, "One entry each",
     "Each athlete appears once, with their highest-scoring season. DiMaggio's entry is a streak, not a season total."),
    (27.0, "Clear of the field",
     "A season counts only if it beats every other season on its own graph by at least one step. Stephen Curry's 402 "
     f"threes (2015–16) score {cu['z']:.2f}, but {cu['best_other']['name']}'s {cu['best_other']['value']} in "
     f"{cu['best_other']['season'].replace('-', '–')}, converted to 2015–16's three-point rate, is about "
     f"{cu['best_other']['converted']:.0f}: within a step. It's left out.")]
FM.F = FC
FM.CARDS = CARDS
FM.CREDITS = CREDITS

_season_lines = FM.card_lines


def card_lines(c):
    """the season film's three lines for this season (they are keyed by its rank there)"""
    return _season_lines(dict(c, rank=c["old_rank"]))


FM.card_lines = card_lines


def season_label(s):
    return s.replace("-", "–")


# ------------------------------------------------------------------ how sure: sixteen versions of the method
class Sure:
    dur = 32.0

    def __init__(self):
        self.board = FC["board"]
        self.faces = {}
        for b in self.board:
            c = next((c for c in FC["cards"] if c["athlete"] == b["athlete"]), None)
            k = next((k for k in FC["considered"] if k["athlete"] == b["athlete"]), None)
            key, pal = (c["photo"], c["palette"]) if c else (k["photo"], k["palette"])
            self.faces[b["athlete"]] = SC.face_disc(key, pal, 54)

    def draw(self, fig, t):
        import matplotlib
        al = fade(t, 0, self.dur, 0.6, 0.6)
        R = FC["robustness"]
        text(fig, 96, 92, "HOW SURE IS THIS LIST?", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, f"{R['n_specs']} ways to rank", size=58, alpha=al)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        text(fig, 1060, 236, "IN THE TOP TEN OF EACH VERSION", size=17, color=MUTED, weight="semibold", alpha=al)
        text(fig, 940, 236, "SCORE", size=17, color=MUTED, weight="semibold", ha="right", alpha=al)
        fill = clamp((t - 9.0) / 5.0)
        for i, b in enumerate(self.board):
            y = 280 + i * 60
            a = al * fade(t, 1.0 + 0.15 * i, 99, 0.5, 0)
            hi = b["in_film"] and t >= 16.0
            col = GOLD if hi else INK
            text(fig, 110, y + 12, f"{b['rank_by_score']}", size=24, color=MUTED, ha="right", alpha=a)
            ax.imshow(self.faces[b["athlete"]], extent=(130, 184, y + 27, y - 27), alpha=a, zorder=5)
            text(fig, 200, y + 10, b["athlete"], size=25, color=col, weight="semibold" if b["in_film"] else "regular", alpha=a)
            text(fig, 470, y + 10, f"{b['achievement']}, {season_label(b['season'])}", size=21, color=INK2, alpha=a)
            text(fig, 940, y + 10, f"{b['z']:.2f}", size=22, color=INK2, ha="right", alpha=a)
            k = b["specs_in_top10"]
            shown = int(round(k * fill))
            for j in range(R["n_specs"]):
                x0 = 1060 + j * 27
                on = j < shown
                ax.add_patch(matplotlib.patches.Rectangle((x0, y - 11), 22, 22, lw=0, zorder=4,
                                                          color=(GOLD if (on and hi) else BLUE) if on else "#2a2f36",
                                                          alpha=a * (0.95 if on else 0.8)))
            if fill > 0:
                text(fig, 1510, y + 10, f"{shown} of {R['n_specs']}", size=21, color=GOLD if hi else INK2, alpha=a * fill)
        caps = [(2.0, "We ranked every season 16 ways: comparison windows of 7, 10, 15 or 20 seasons, straight or curved "
                      "trends, with or without the league adjustment."),
                (9.0, "Each square is a version that put the season in its top ten."),
                (16.0, f"Three made all 16. The film's five made at least {R['agree']}: three versions in four."),
                (24.0, "Each of the other five rests on one choice of method.")]
        cur = [m for t0, m in caps if t0 <= t]
        if cur:
            y0 = 905
            for t0, m in caps:
                if t0 <= t and m == cur[-1]:
                    FM.para(fig, 96, y0, m, width=110, size=27, color=INK, alpha=al * fade(t, t0, 99, 0.6, 0))


# ------------------------------------------------------------------ the five that came close
class Considered:
    dur = 52.0
    STEP = 9.2

    def __init__(self):
        self.rows = FC["considered"]
        self.faces = {k["athlete"]: SC.face_disc(k["photo"], k["palette"], 108) for k in self.rows}

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.7)
        text(fig, 96, 92, "THE REST OF THE TOP TEN", size=21, color=MUTED, weight="semibold", alpha=al)
        title(fig, 96, 160, "Five that came close", size=58, alpha=al)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        for i, k in enumerate(self.rows):
            t0 = 2.0 + self.STEP * i
            y = 232 + i * 164
            a = al * fade(t, t0, 99, 0.7, 0)
            if a <= 0:
                continue
            now = t0 <= t < t0 + self.STEP or (i == len(self.rows) - 1 and t >= t0)
            dim = 1.0 if now else 0.62
            ax.imshow(self.faces[k["athlete"]], extent=(96, 204, y + 116, y + 8), alpha=a * dim, zorder=5)
            text(fig, 236, y + 44, k["athlete"], size=32, color=INK, weight="semibold", alpha=a * dim)
            text(fig, 520, y + 44,
                 f"{k['achievement']}, {season_label(k['season'])}", size=25, color=INK2, alpha=a * dim)
            text(fig, 1824, y + 44, f"#{k['rank_by_score']} by score · top ten in {k['specs_in_top10']} of 16",
                 size=22, color=GOLD if now else MUTED, ha="right", alpha=a * dim)
            FM.para(fig, 236, y + 72, k["why"], width=112, size=23, color=INK if now else INK2, alpha=a * dim)


# ------------------------------------------------------------------ the close
class Close:
    dur = 22.0
    FINALE = 1.5

    def draw(self, fig, t):
        al = fade(t, 0, self.dur, 0.6, 0.9)
        title(fig, W / 2, 400, "Not a ranking of the greatest athletes.", size=60, ha="center", alpha=al * fade(t, 1.3, 99, 0.8, 0))
        title(fig, W / 2, 500, "A ranking of how far one season stood", size=60, ha="center", alpha=al * fade(t, 3.0, 99, 0.8, 0))
        title(fig, W / 2, 580, "from everyone else of its time.", size=60, ha="center", color=GOLD,
              alpha=al * fade(t, 3.0, 99, 0.8, 0))
        text(fig, W / 2, 720, "Which season would you choose?", size=36, color=INK2, ha="center",
             alpha=al * fade(t, 7.5, 99, 0.8, 0))


class End:
    dur = 26.0

    def draw(self, fig, t):
        self.credits(fig, t)
        al = fade(t, 0, 12.5, 0.8, 0.8)
        if al <= 0:
            return
        text(fig, 96, 120, "SOURCES AND METHOD", size=21, color=MUTED, weight="semibold", alpha=al)
        lines = [
            ("Data", "Baseball-Reference, Basketball-Reference, Hockey-Reference, Pro Football Archives, nflverse, "
                     "Retrosheet."),
            ("Model", f"{sc['n']:,} season-leading performances in {sc['n_categories']} team-sport categories. Each is "
                      "compared with the best of everyone else in the 15 seasons before and after, around a curved trend, "
                      "with a league-scoring adjustment estimated from the data. A season must clear every other season on "
                      "its own graph by at least one step. Seasons are labelled by the year they end."),
            ("Caveats", "Scores are statistical distances, not probabilities or measures of ability. Player minutes, "
                        "possessions and ice time are not standardised. The five are the seasons in the top ten of at "
                        "least 12 of 16 versions of the method."),
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
        order = [CARDS[r]["photo"] for r in range(1, N + 1)] + [k["photo"] for k in FC["considered"]]
        i = 0
        for k in order:
            cr = CREDITS.get(k)
            if not cr:
                continue
            col, row = i % 2, i // 2
            x, y = 96 + col * 880, 200 + row * 128
            a = al * fade(t, 13.3 + 0.12 * i, 99, 0.5, 0)
            FM.photo(fig, k, x, y, x + 72, y + 90, alpha=a, credit=False)
            text(fig, x + 96, y + 30, cr["athlete"], size=24, color=INK, weight="semibold", alpha=a)
            lic = cr["license"] if cr["license"] != "Public domain" else "public domain"
            text(fig, x + 96, y + 62, f"{cr['author'][:60]} · {lic}", size=20, color=INK2, alpha=a)
            text(fig, x + 96, y + 88, cr["file"][:70], size=16, color=MUTED, alpha=a)
            i += 1
        FM.para(fig, 96, 900, "All from Wikimedia Commons (commons.wikimedia.org): cropped, backgrounds removed, toned in team "
                              "colours. Licences: creativecommons.org/licenses (CC BY, CC BY-SA, CC0) or public domain. Full "
                              "credits and links are in the description. Icons: BlenderKit models.", width=120, size=20,
                color=MUTED, alpha=al)


SECTIONS = {"hook": FM.SECTIONS["hook"], "curve": FM.SECTIONS["curve"], "fair": FM.SECTIONS["fair"],
            "sure": Sure, "considered": Considered, "close": Close, "end": End}
for _r in range(1, N + 1):
    SECTIONS[f"c{_r}"] = FM.SECTIONS[f"c{_r}"]
ORDER = ["hook"] + [f"c{r}" for r in range(N, 0, -1)] + ["curve", "fair", "sure", "considered", "close", "end"]
PICKS = {"hook": [14.0, 19.5, 26.0], "fair": [36.0], "sure": [20.0], "considered": [12.0, 48.0], "close": [12.0],
         "end": [8.0, 20.0], "curve": [24.0]}
FINAL = "film_c_final.mp4"

if __name__ == "__main__":
    which = sys.argv[1]
    import common
    if which == "stills":
        common.stills(sys.modules[__name__], PICKS, OUT)
    elif which == "voiced":                                 # remix the voiced copy only (no re-render)
        common.voiced(OUT, FINAL)
    elif which == "assemble":
        common.assemble(sys.modules[__name__], OUT, FINAL, Close.FINALE)
    else:
        print(look.render_section(which, SECTIONS[which]()))
