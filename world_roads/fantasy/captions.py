"""The race's captions and closing summary, from the data:  python captions.py  -> captions.json, summary.json

Every caption is found in the data, not written in by hand:
* each new No. 1 season after the first year, when it takes the top (if it holds it four weeks or to the end);
* a season that finishes within a point of the record, at its end;
* the best season at each position but the record's, at its end, with the place it then held;
* a season cut short (fewer than 12 games a team), at its start; the first 17-game season, at its start;
* the last season's best, at its end, with its all-time place.
A caption is anchored to a season and week; video.py turns that into film time and spaces them out.
"""
import os, sys, csv, json
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V                                     # noqa: E402
from vorp import team_games                           # noqa: E402

WORD = {"QB": "quarterback", "RB": "running back", "WR": "receiver", "TE": "tight end"}


def main():
    D = V.Data()
    N, F, L = V.N, V.FIRST, V.LAST
    e = D.e
    num = V.num
    who = lambda i: f"{e[i]['player']}'s {e[i]['season']}"
    marks = [(y, w, D.C[y][:, w]) for y in range(F, L + 1) for w in range(D.weeks[y] + 1)]
    caps = []

    def holds(seq, k):
        """how many week marks the place changed at mark k stays with its new holder (to the end: a large number)"""
        n = 1
        while k + n < len(seq) and seq[k + n][2] == seq[k][2]:
            n += 1
        return n if k + n < len(seq) else 99

    # each new No. 1 after the first year
    lead = [(y, w, int(np.argmax(v))) for y, w, v in marks if v.max() > 0]
    for k in range(1, len(lead)):
        y, w, i = lead[k]
        if i == lead[k - 1][2] or y == F or holds(lead, k) < 4:
            continue
        o = lead[k - 1][2]
        caps.append(dict(season=y, week=w, text=[f"{who(i)} passes {who(o)} for No. 1",
                                                 f"{V.statline(e[i])}: {e[i]['vorp']:,.1f} over replacement"]))
    record = lead[-1][2]

    # seasons within a point of the record
    for i in np.argsort([-r["vorp"] for r in e]):
        if i != record and e[record]["vorp"] - e[i]["vorp"] < 1.0:
            y = e[i]["season"]
            gap = e[record]["vorp"] - e[i]["vorp"]
            caps.append(dict(season=y, week=D.weeks[y], delay=0.2, text=[
                f"{e[i]['player']}, {y}: {V.statline(e[i])}",
                f"{e[i]['vorp']:,.1f} over replacement, {gap:.1f} short of {who(record)}" if y > e[record]["season"] else
                f"{e[i]['vorp']:,.1f} over replacement, {gap:.1f} behind the record still to come"]))

    # the best season at each other position, at its end, with the place it then held
    for P in WORD:
        if P == e[record]["pos"]:
            continue
        mine = [i for i in range(len(e)) if e[i]["pos"] == P]
        if not mine:
            continue
        i = max(mine, key=lambda i: e[i]["vorp"])
        y = e[i]["season"]
        place = D.place(e[i]["vorp"] + 1e-6, y, D.C[y][:, -1])
        caps.append(dict(season=y, week=D.weeks[y], delay=0.2, text=[
            f"{e[i]['player']}, {y}: the most valuable {WORD[P]} season",
            f"{V.statline(e[i])} · {e[i]['vorp']:,.0f} over replacement, No. {place} all-time when it ended"]))

    # short seasons, and the first 17-game season
    for y in range(F, L + 1):
        g = int(team_games(y))
        best = max((r["vorp"] for r in e if r["season"] == y), default=0)
        twelfth = sorted(D.C[y][:, 0], reverse=True)[N - 1]
        if g < 12:
            caps.append(dict(season=y, week=0, delay=0.3, text=[
                f"{y}: a players' strike cuts the season to {g} games",
                "no season this short reaches the board" if best < twelfth else "the season's best still reaches the board"]))
        if g == 17 and int(team_games(y - 1)) == 16:
            caps.append(dict(season=y, week=0, delay=0.3, text=[f"{y}: the season grows to 17 games",
                                                                "a 17th game is worth about 6% more over replacement"]))

    # the last season's best, and where it ranks
    mine = [i for i in range(len(e)) if e[i]["season"] == L]
    if mine:
        i = max(mine, key=lambda i: e[i]["vorp"])
        place = D.place(e[i]["vorp"] + 1e-6, L, D.C[L][:, -1])
        if place <= N:
            caps.append(dict(season=L, week=D.weeks[L], delay=0.2, text=[
                f"{e[i]['player']}, {L}: No. {place} all-time", f"{V.statline(e[i])} · {e[i]['vorp']:,.0f} over replacement"]))

    caps.sort(key=lambda c: (c["season"], c["week"]))
    json.dump(caps, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    for c in caps:
        print(c["season"], c["week"], " | ".join(c["text"]))

    # the closing summary: the top two, the board's make-up by position and decade, the best at the other positions
    v = D.C[L][:, -1]
    o = [i for i in np.argsort(-v, kind="stable")[:N]]
    card = lambda i: dict(player=e[i]["player"], pos=e[i]["pos"], team=e[i]["team"], season=e[i]["season"],
                          vorp=float(e[i]["vorp"]), line=V.statline(e[i]))
    a, b = card(o[0]), card(o[1])
    pos = Counter(e[i]["pos"] for i in o)
    dec = Counter(f"{e[i]['season'] // 10 * 10}s" for i in o)
    others = []
    for P in WORD:
        mine = [i for i in range(len(e)) if e[i]["pos"] == P]
        if mine and P not in pos:
            i = max(mine, key=lambda i: e[i]["vorp"])
            others.append(f"{P} {e[i]['player']} {e[i]['season']} ({e[i]['vorp']:,.0f})")
    gap = a["vorp"] - b["vorp"]
    summary = dict(first=a, second=b,
                   gap=f"{a['player'].split()[-1]}'s {a['season']} edges {b['player'].split()[-1]}'s {b['season']} by "
                       + (f"{gap:.1f}" if gap < 10 else num(gap)),
                   mix="The top 12: " + ", ".join(f"{n} {p}{'s' if n != 1 else ''}" for p, n in pos.most_common())
                       + " · " + ", ".join(f"{n} from the {d}" for d, n in sorted(dec.items())),
                   others=("Best of the rest: " + " · ".join(others)) if others else "")
    summary["example"] = example()
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(summary["gap"], "|", summary["mix"], "|", summary["others"])
    x = summary["example"]
    print("intro example:", x["season"], x["qb"]["player"], x["qb"]["points"], x["qb"]["free"], x["qb"]["value"], "|",
          x["rb"]["player"], x["rb"]["points"], x["rb"]["free"], x["rb"]["value"])


def example():
    """the intro's example: a season's top-scoring QB who outscored one of its three top-scoring RBs, both playing every
    game, while the RB was worth at least 30 more over replacement; of those, the QB who outscored his RB by the most"""
    rows = [dict(r, season=int(r["season"]), points=float(r["points"]), replacement=float(r["replacement"]),
                 vorp=float(r["vorp"]), games=int(r["games"])) for r in csv.DictReader(open(os.path.join(V.MEDIA, "seasons.csv")))]
    by = defaultdict(list)
    for r in rows:
        if r["games"] == team_games(r["season"]):
            by[r["season"], r["pos"]].append(r)
    best = None
    for (y, p), qbs in by.items():
        if p != "QB":
            continue
        q = max(qbs, key=lambda r: r["points"])
        if q["points"] < max(r["points"] for r in rows if r["season"] == y and r["pos"] == "QB"):
            continue                                  # the season's top-scoring QB must have played every game
        for b in sorted(by.get((y, "RB"), []), key=lambda r: -r["points"])[:3]:
            m = q["points"] - b["points"]
            if m > 0 and b["vorp"] - q["vorp"] >= 30 and (best is None or m > best[0]):
                best = (m, q, b)
    if best is None:
        return None
    side = lambda r: dict(player=r["player"], pos=r["pos"], team=r["team"], points=round(r["points"]),
                          free=round(r["replacement"]), value=round(r["points"]) - round(r["replacement"]))
    return dict(season=best[1]["season"], qb=side(best[1]), rb=side(best[2]))


if __name__ == "__main__":
    main()
