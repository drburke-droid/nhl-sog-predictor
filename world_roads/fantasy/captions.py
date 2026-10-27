"""The race's captions and closing summary, from the data:  python captions.py  -> captions.json, summary.json

Every caption is found in the data, not written in by hand:
* every change at No. 1 after the first season that lasts four weeks or more: the first leader at a new position,
  and the leader who keeps it;
* the most valuable season since 2000, at the end of that season, and any season within a point of it, in order;
* the first player to the highest 500-point milestone anyone reaches;
* the highest any quarterback climbs, the first time one gets there;
* each new No. 2 after the leader retires that lasts three weeks or more (or to the end);
* the last change of the top player at each other position, once the leader has No. 1 for good;
* a run of two or more seasons in which no one on the board adds a point, and the first new name after it;
* the highest-ranked active player at the end, when he climbs to that place.
A caption is anchored to a season and week; video.py turns that into film time and spaces them out.
"""
import os, sys, json
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video as V                                     # noqa: E402

WORD = {"QB": ("quarterback", "QB"), "RB": ("running back", "RB"), "WR": ("receiver", "WR"), "TE": ("tight end", "TE"),
        "K": ("kicker", "K")}


def main():
    D = V.Data()
    N, F, L = V.N, V.FIRST, V.LAST
    sea = D.sea
    name = lambda i: D.name[D.pids[i]]
    last = lambda i: name(i).split()[-1]
    pid = lambda i: D.pids[i]
    num = V.num
    active = lambda i, y: (pid(i), y) in D.wk

    def pos(i, y):
        p = pid(i)
        s = max((s for s in D.played[p] if s <= y), default=D.played[p][0])
        return sea[p, s]["pos"]

    # every week boundary in order: (season, week, values)
    marks = [(y, w, D.C[y][:, w]) for y in range(F, L + 1) for w in range(D.weeks[y] + 1)]
    top = lambda v: [i for i in np.argsort(-v, kind="stable")[:N] if v[i] > 0]
    caps = []

    def holds(seq, k):
        """how many week marks the place changed at mark k stays with its new holder (to the end: a large number)"""
        n = 1
        while k + n < len(seq) and seq[k + n][2] == seq[k][2]:
            n += 1
        return n if k + n < len(seq) else 99

    # changes at No. 1 after the first season that last four weeks or more
    lead = [(y, w, int(np.argmax(v))) for y, w, v in marks if v.max() > 0]
    since = {lead[0][2]: lead[0][0]}
    led = {pos(lead[0][2], lead[0][0])}
    changes = [k for k in range(1, len(lead)) if lead[k][2] != lead[k - 1][2] and lead[k][0] > F and holds(lead, k) >= 4]
    for n, k in enumerate(changes):
        y, w, i = lead[k]
        o = lead[changes[n - 1]][2] if n else lead[k - 1][2]
        p = pos(i, y)
        if n == len(changes) - 1:
            line = f"and holds it through {L}"
        elif p not in led:
            line = f"the first {WORD[p][0]} to lead"
        else:
            line = f"{last(o)} had led since {since[o]}"
        led.add(p)
        since[i] = y
        caps.append(dict(season=y, week=w, text=[f"{name(i)} passes {name(o)} for No. 1", line]))
    leader = lead[-1][2]
    settled = lead[changes[-1]][:2] if changes else (F, 0)

    # the most valuable season, and seasons within a point of it
    rows = sorted(sea.values(), key=lambda r: -r["vorp"])
    near = [r for r in rows if rows[0]["vorp"] - r["vorp"] < 1.0]
    nxt = rows[len(near)]
    top_so_far = None
    for r in sorted(near, key=lambda r: r["season"]):
        td = int(float(r["rush_td"]) + float(r["rec_td"]))
        what = f"{int(float(r['pass_td']))} TD passes" if r["pos"] == "QB" else f"{td} touchdowns"
        v = f"{r['vorp']:,.1f}" if len(near) > 1 else num(r["vorp"])
        if len(near) == 1:
            tail = f"the most in a season (next: {nxt['player'].split()[-1]}'s {nxt['season']}, {num(nxt['vorp'])})"
        elif top_so_far is None:
            tail = f"the most of any season since {F}" if r is rows[0] else "the most in a season so far"
        else:
            t = f"{top_so_far['player'].split()[-1]}'s {top_so_far['season']} ({top_so_far['vorp']:,.1f})"
            tail = (f"edging {t} for the most in a season" if r["vorp"] > top_so_far["vorp"] else
                    f"level with {t} for the most in a season" if r["vorp"] == top_so_far["vorp"] else f"just short of {t}")
        if top_so_far is None or r["vorp"] > top_so_far["vorp"]:
            top_so_far = r
        caps.append(dict(season=r["season"], week=D.weeks[r["season"]], delay=0.2,
                         text=[f"{r['player']}, {r['season']}: {what}, {num(r['points'])} PPR points",
                               f"{v} over replacement, {tail}"]))

    # the first to the highest 500-point milestone anyone reaches
    peak = D.C[L][:, -1].max()
    m = int(peak // 500 * 500)
    if m >= 1000:
        for y, w, v in marks:
            if v.max() >= m:
                i = int(np.argmax(v))
                others = sum(1 for j in range(len(D.pids)) if j != i and D.C[L][j, -1] >= m)
                caps.append(dict(season=y, week=w, text=[f"{name(i)}: the first to {num(m)} points over replacement",
                                                         f"no one else has reached {num(m)}" if not others else
                                                         f"{others} others have since"]))
                break

    # the highest any quarterback climbs (after the first season), the first time one gets there
    qbr = [(y, w, rk, i) for y, w, v in marks if y > F for rk, i in enumerate(top(v)) if pos(i, y) == "QB"]
    if qbr:
        hi = min(r[2] for r in qbr)
        y, w, rk, i = min((r for r in qbr if r[2] == hi), key=lambda r: (r[0], r[1]))
        caps.append(dict(season=y, week=w, text=[f"{name(i)} climbs to No. {rk + 1}",
                                                 "the highest any quarterback has stood"]))

    # each new No. 2 after the leader retires: the highest an active player can stand
    gone = D.played[pid(leader)][-1]
    seq = [(y, w, top(v)[:2]) for y, w, v in marks]
    sec = [(y, w, o[1] if len(o) > 1 else None) for y, w, o in seq]
    for k in range(1, len(seq)):
        y, w, o = seq[k]
        po = seq[k - 1][2]
        if y <= gone or len(o) < 2 or len(po) < 2 or o[1] == po[1] or holds(sec, k) < 3:
            continue
        i = o[1]
        then = None                                   # the last active player as high, before him
        for yy, ww, oo in seq[:k]:
            for j in oo:
                if active(j, yy) and j != i:
                    then = (yy, name(j))
        caps.append(dict(season=y, week=w, text=[f"{name(i)} passes {name(po[1])} for No. 2",
                                                 f"no active player has stood higher since {then[1]} in {then[0]}" if then else
                                                 "the highest an active player has stood"]))

    # the last change of the top player at each other position, once No. 1 is settled, if he is on the board
    for P in WORD:
        if P == pos(leader, L):
            continue
        prev, chg, start = None, None, None
        for y, w, v in marks:
            o = [i for i in np.argsort(-v, kind="stable")[:60] if v[i] > 0 and pos(i, y) == P]
            if o and o[0] != prev:
                if prev is not None:
                    chg = (y, w, o[0], prev, start)
                prev, start = o[0], y
        if chg and (chg[0], chg[1]) > settled:
            y, w, i, o, s = chg
            if i in top(D.C[y][:, w]):
                caps.append(dict(season=y, week=w, text=[f"{name(i)} passes {name(o)} as the top {WORD[P][1]}",
                                                         f"{last(o)} had been the top {WORD[P][0]} since {s}"]))

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
    first_in = {}
    for y, w, v in marks:
        for i in top(v):
            first_in.setdefault(i, (y, w))
    if runs:
        after = sorted((t, i) for i, t in first_in.items() if t[0] > runs[-1][-1])
        if after:
            (y, w), i = after[0]
            (ly, lw), li = max((t, j) for j, t in first_in.items() if t[0] < runs[-1][0])
            caps.append(dict(season=y, week=w, text=[f"{name(i)}: the first new name in the top 12 since {ly}",
                                                     f"when {name(li)} broke in"]))

    # the highest-ranked active player at the end, when he climbs to that place
    o = top(D.C[L][:, -1])
    act = [i for i in o[1:] if active(i, L)]
    if act:
        i = act[0]
        r = o.index(i)
        w = min(w for w in range(D.weeks[L] + 1) if all(top(D.C[L][:, x]).index(i) == r if i in top(D.C[L][:, x]) else False
                                                         for x in range(w, D.weeks[L] + 1)))
        if not any(c["season"] == L and c["week"] == w and c["text"][0].startswith(name(i)) for c in caps):
            caps.append(dict(season=L, week=w, text=[f"{name(i)} climbs to No. {r + 1}",
                                                     "the highest of any active player"]))

    caps.sort(key=lambda c: (c["season"], c["week"]))
    json.dump(caps, open(os.path.join(HERE, "captions.json"), "w"), indent=1)
    for c in caps:
        print(c["season"], c["week"], " | ".join(c["text"]))

    # the closing summary: No. 1 against the highest-ranked active player after the last season, and the board's make-up
    v = D.C[L][:, -1]

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
        span = f"{D.played[p][0]}–" if D.played[p][-1] == L else f"{D.played[p][0]}–{D.played[p][-1]}"
        above = sum(1 for r in ss if r["vorp"] > 0)
        return dict(player=name(i), pos=ss[-1]["pos"], teams=", ".join(teams[:3]), span=span, vorp=float(v[i]),
                    best=top_s["vorp"], best_year=top_s["season"], rank=o.index(i) + 1,
                    note=f"{above} seasons above replacement · best {num(top_s['vorp'])} in {top_s['season']}")
    a = card(o[0])
    b = card(act[0] if act else o[1])
    gap = a["vorp"] - b["vorp"]
    mix = defaultdict(int)
    for i in o:
        mix[pos(i, L)] += 1
    parts = [f"{mix[p]} {WORD[p][1]}{'s' if mix[p] != 1 else ''}" for p in sorted(mix, key=lambda p: -mix[p])]
    who = f"No. {b['rank']}, the top active player" if act else f"No. {b['rank']}"
    summary = dict(lead=a, chase=b, gap=f"{b['player'].split()[-1]} ({who}) trails by {num(gap)}, "
                                       f"{gap / b['best']:.1f} times his best season",
                   mix=f"The top 12 after {L}: " + ", ".join(parts))
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    print(summary["gap"], "|", summary["mix"])


if __name__ == "__main__":
    main()
