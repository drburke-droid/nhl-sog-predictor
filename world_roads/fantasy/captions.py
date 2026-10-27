"""The race's captions and closing summary, from the data:  python captions.py  -> captions.json, summary.json

Every caption is found in the data, not written in by hand:
* the most valuable season since 2000, at the end of that season, and any season that ties it (within a point);
* the last change at No. 1, when the leader takes it for good;
* the first player to 2,000 career points over replacement;
* the first quarterback to finish a season on the board, with his touchdown passes that season;
* a run of two or more seasons in which no one on the board adds a point;
* the first new name on the board after that run, and the year of the last new name before it;
* each new No. 2 after the leader retires (the highest an active player can then stand).
A caption is anchored to a season and week; video.py turns that into film time and spaces them out.
"""
import os, sys, json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V                                     # noqa: E402


def main():
    D = V.Data()
    N, F, L = V.N, V.FIRST, V.LAST
    sea = D.sea
    name = lambda i: D.name[D.pids[i]]
    pid = lambda i: D.pids[i]
    num = V.num
    # every week boundary in order: (season, week, values)
    marks = [(y, w, D.C[y][:, w]) for y in range(F, L + 1) for w in range(D.weeks[y] + 1)]
    top = lambda v: [i for i in np.argsort(-v, kind="stable")[:N] if v[i] > 0]
    caps = []

    # the most valuable season, and seasons tying it
    rows = sorted(sea.values(), key=lambda r: -r["vorp"])
    best = rows[0]
    ties = [r for r in rows[1:] if best["vorp"] - r["vorp"] < 1.0]
    for k, r in enumerate([best] + ties):
        td = int(float(r["rush_td"]) + float(r["rec_td"]))
        what = f"{int(float(r['pass_td']))} TD passes" if r["pos"] == "QB" else f"{td} touchdowns"
        if k == 0:
            tail = "no season since has topped it" if ties else "no season since has come close"
        else:
            tail = f"level with {best['player'].split()[-1]}'s {best['season']} as the most in a season"
        caps.append(dict(season=r["season"], week=D.weeks[r["season"]], delay=0.2,
                         text=[f"{r['player']}, {r['season']}: {what}, {num(r['ppr'])} PPR points",
                               f"{num(r['vorp'])} over replacement, {tail}"]))

    # No. 1 for good
    lead = [(y, w, int(np.argmax(v))) for y, w, v in marks if v.max() > 0]
    k = len(lead) - 1
    while k > 0 and lead[k - 1][2] == lead[-1][2]:
        k -= 1
    y, w, i = lead[k]
    prev = lead[k - 1][2]
    caps.append(dict(season=y, week=w, text=[f"{name(i)} passes {name(prev)} for No. 1",
                                             f"and holds it through {L}"]))
    leader = i

    # first to 2,000
    for y, w, v in marks:
        if v.max() >= 2000:
            i = int(np.argmax(v))
            others = sum(1 for j in range(len(D.pids)) if j != i and D.C[L][j, -1] >= 1500)
            caps.append(dict(season=y, week=w, text=[f"{name(i)}: the first to 2,000 points over replacement",
                                                     "no one else has reached 1,500" if not others else
                                                     f"{others} others have passed 1,500"]))
            break

    # first quarterback on the board
    for y in range(F, L + 1):
        qb = [i for i in top(D.C[y][:, -1]) if (pid(i), y) in sea and sea[pid(i), y]["pos"] == "QB"]
        if qb:
            i = qb[0]
            r = sea[pid(i), y]
            caps.append(dict(season=y, week=D.weeks[y], delay=0.2,
                             text=[f"{name(i)}, {y}: {int(float(r['pass_td']))} touchdown passes",
                                   "the first quarterback to finish a season in the top 12"]))
            break

    # the board stands still
    still = [y for y in range(F, L + 1) if D.board_gain(y) < 1.0]
    runs, cur = [], []
    for y in still:
        if cur and y != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(y)
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= 2]
    for r in runs:
        caps.append(dict(season=r[0], week=1, text=[f"{r[0]}–{r[-1]}: the board stands still",
                                                    "no one in the top 12 scores above replacement"]))
    # the first new name after the stillness, and the last one before it
    ever, first_in = set(), {}
    for y, w, v in marks:
        for i in top(v):
            if i not in ever:
                ever.add(i)
                first_in[i] = (y, w)
    if runs:
        after = sorted((t, i) for i, t in first_in.items() if t[0] > runs[-1][-1])
        if after:
            (y, w), i = after[0]
            (ly, lw), li = max((t, j) for j, t in first_in.items() if t[0] < runs[-1][0])
            caps.append(dict(season=y, week=w, text=[f"{name(i)}: the first new name in the top 12 since {ly}",
                                                     f"when {name(li)} broke in"]))

    # each new No. 2 after the leader retires: the highest an active player can stand
    gone = D.played[pid(leader)][-1]
    seq = [(y, w, top(v)[:2]) for y, w, v in marks]
    for k in range(1, len(seq)):
        y, w, o = seq[k]
        po = seq[k - 1][2]
        if y <= gone or len(o) < 2 or len(po) < 2 or o[1] == po[1]:
            continue
        i = o[1]
        then = None                                   # the last active player as high, before him
        for yy, ww, oo in seq[:k]:
            for j in oo:
                if (pid(j), yy) in D.wk and j != i:
                    then = (yy, name(j))
        caps.append(dict(season=y, week=w, text=[f"{name(i)} passes {name(po[1])} for No. 2",
                                                 f"no active player has stood higher since {then[1]} in {then[0]}" if then else
                                                 "the highest an active player has stood"]))
    caps.sort(key=lambda c: (c["season"], c["week"]))
    json.dump(caps, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    for c in caps:
        print(c["season"], c["week"], " | ".join(c["text"]))

    # the closing summary: No. 1 against No. 2 after the last season, and the board's make-up
    v = D.C[L][:, -1]
    o = top(v)

    def card(i):
        p = pid(i)
        ss = [sea[p, y] for y in D.played[p]]
        teams = []
        for r in ss:
            for w_, t, _ in sorted(D.wk.get((p, r["season"]), [])):
                t = V.TEAM.get(t, t)
                if t not in teams:
                    teams.append(t)
        top_s = max(ss, key=lambda r: r["vorp"])
        active = D.played[p][-1] == L
        span = f"{D.played[p][0]}–" if active else f"{D.played[p][0]}–{D.played[p][-1]}"
        above = sum(1 for r in ss if r["vorp"] > 0)
        return dict(player=name(i), pos=ss[-1]["pos"], teams=", ".join(teams[:3]), span=span, vorp=float(v[i]),
                    best=top_s["vorp"], best_year=top_s["season"],
                    note=f"{above} seasons above replacement · best {num(top_s['vorp'])} in {top_s['season']}")
    a, b = card(o[0]), card(o[1])
    gap = a["vorp"] - b["vorp"]
    mix = defaultdict(int)
    for i in o:
        p = pid(i)
        mix[sea[p, D.played[p][-1]]["pos"]] += 1
    word = {"QB": ("QB", "QBs"), "RB": ("RB", "RBs"), "WR": ("WR", "WRs"), "TE": ("TE", "TEs")}
    parts = [f"{mix[p]} {word[p][mix[p] != 1]}" for p in sorted(mix, key=lambda p: -mix[p])]
    summary = dict(lead=a, chase=b, gap=f"{a['player'].split()[-1]} leads by {num(gap)}, "
                                       f"{gap / b['best']:.1f} times {b['player'].split()[-1]}'s best season",
                   mix=f"The top 12 after {L}: " + ", ".join(parts))
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
