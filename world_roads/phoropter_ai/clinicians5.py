"""The textbook examiner for the clinic exam, version 5 (env5.py), and a runner for scripts and the AI.

Identical to clinicians3.py except the cylinder search uses the standard power-search buttons: a
cross-cylinder check at 180/90, then at 45/135 (env5 sets the dial to the preferred orientation).

The textbook sequence follows Wilkinson 2016 (University of Iowa EyeRounds; the standard Borish sequence), see
PUBLISHED.md section 4. Textbooks start from retinoscopy or an autorefractor. Here there is neither, so step 0
finds a starting sphere by trial, as a clinician would with a trial frame:

0. Starting sphere: read the smallest line. If they can't read 20/200, compare -2.00 with +2.00 (then -4.00
   with +4.00, then -6.00 with +6.00) and start from the lens that helps. Then 1.00 D steps while the chart
   improves (minus), or while it doesn't get worse (plus).
For each eye (right, then left):
1. Monocular MPMVA: fog +0.75, then -0.25 at a time reading the chart; keep the most plus lens within one
   letter of the best reading.
2. Cross-cylinder, one line bigger than the best: if there's no cylinder, search for it at four orientations
   45° apart; axis first if the cylinder is 1.00 D or more, otherwise power first; then the other; axis steps
   15° then 5°; cylinder 0.25 D at a time, with 0.25 D of sphere per net 0.50 D to keep the spherical equivalent.
3. Second MPMVA (cylinder changed by 0.50 D or more, or axis by 10° or more).
Then both eyes:
4. Binocular balance: +0.75 both eyes, 20/40 line, prism split (top = right eye), +0.25 to the clearer eye
   until they look equal; if the answer flips, stop there.
5. Binocular MPMVA: -0.25 both eyes at a time, reading the chart; no extra minus without a better reading.
Prescribe. If the patient says an eye is still blurry: redo that eye (steps 0-3), then 4-5 again (twice at most).

Each script is a generator: it yields a button and receives (code, va, letters right, letters on line, line).
"""
import numpy as np
TOL = 0.04                                               # two letters
from env5 import (ClinicEnv, N_ACT3, SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15, AXP2, AXM2,
                  READ, READ_LINE, LINE_SM, LINE_LG, CMP_P, CMP_M, JCC_AX, JCC_PW, CHK_180, CHK_45, SPIN, NEXT, OU_P,
                  OU_M, BALANCE, R_P, L_P, RX)


def to_line(env, i, target):
    for _ in range(14):
        cur = round(float(env.line[i]), 1)
        if abs(cur - target) < 0.01:
            return
        yield LINE_LG if cur < target else LINE_SM


def find_sphere(env, i):
    """No objective starting point: find a starting sphere by trial, 1.00 D at a time. If they can't read the top
    line, try -2.00 against +2.00, then -4.00 against +4.00, then -6.00 against +6.00, and start from the first
    lens that lets them read more; myopia first when it's a tie."""
    code, best, *_ = yield READ
    go = 0
    if best >= 0.95:
        for s in (2, 4, 6):
            for _ in range(s):
                yield SM100
            code, vm, *_ = yield READ
            for _ in range(2 * s):
                yield SP100
            code, vp, *_ = yield READ
            if min(vm, vp) < best - 0.05:
                if vm <= vp:
                    for _ in range(2 * s):
                        yield SM100
                    best, go = vm, -1
                else:
                    best, go = vp, 1
                break
            for _ in range(s):                           # back to plano, try further out
                yield SM100
        if go == 0:
            return                                       # nothing helped: leave it to the fog (very rare)
    if go <= 0:                                          # minus: keep going while it is no worse (2 tries)
        k, since, at_best = 0, 0, 0
        for _ in range(12):
            yield SM100
            k += 1
            code, v, *_ = yield READ
            if v < best - 1e-9:
                best, since, at_best = v, 0, k
            elif best < 0.95:                            # (while they only see the big E, keep going)
                since += 1
                if since >= 2:
                    break
        for _ in range(k - at_best):                     # back to the least minus with the best reading
            yield SP100
        if at_best >= 1 or go == -1:
            return
    for _ in range(9):                                   # plus: more plus until the chart clearly drops
        yield SP100
        code, v, *_ = yield READ
        if v <= best + 0.1 and v < 1.15:                 # (stop if they can't read anything)
            best = min(best, v)
        else:
            yield SM100
            break


def mpmva(env, i, fog=3, patience=2, tol=None, down=SM25, up=SP25, read_first=True):
    """Fog until the patient has lost at least 1.5 lines, then reduce the plus 0.25 at a time reading the chart.
    Stop after `patience` steps with no better reading, but never while the reading is still more than two
    lines worse than before the fog. End on the most plus lens within one letter of the best reading."""
    base = None
    if read_first:
        code, base, *_ = yield READ
    for _ in range(fog):
        yield up
    code, v0, *_ = yield READ
    extra = 0
    while base is not None and v0 < base + 0.15 and extra < 6 and v0 < 1.0:
        yield up                                         # not fogged yet (they were over-minused): more plus
        extra += 1
        code, v0, *_ = yield READ
    scores, k, since, best = [v0], 0, 0, v0
    floor = (base if base is not None else -1.0) + 0.2
    while k < 20 and (since < patience or best > floor):
        yield down
        k += 1
        code, v, *_ = yield READ
        scores.append(v)
        if v < best - 1e-9:
            best, since = v, 0
        else:
            since += 1
        if since >= 4:
            break
    tol = TOL if tol is None else tol
    target = min(j for j, sc in enumerate(scores) if sc <= best + tol + 1e-9)
    for _ in range(k - target):
        yield up


def cyl_search(env, i):
    """The power search: cross-cylinder at 180 vs 90, then 45 vs 135. A preference puts the dial there (env5);
    then add 0.25 D of cylinder."""
    for chk in (CHK_180, CHK_45):
        code, *_ = yield chk
        if code in (0, 1):
            yield CYL_MORE
            return True
    return False


def jcc_axis(env, i):
    """Bracket the axis: 15° steps for small cylinders, 5° for larger; narrow the step at each reversal
    (15 -> 5 -> 2 for cylinders of 1.00 D or more, where a few degrees matter); stop at 'same' or after the
    smallest step has reversed."""
    e = int(env.cur[i])
    big = abs(env.C[i, e]) >= 1.0
    steps = [15, 5, 2] if abs(env.C[i, e]) <= 0.5 else ([5, 2] if big else [5])
    if not big and steps[0] == 15:
        steps = [15, 5]
    si, last, moved = 0, 0, 0
    for _ in range(12):
        code, *_ = yield JCC_AX
        if code == 2:
            break
        d = 1 if code == 0 else -1
        if last and d != last:
            si += 1
            if si >= len(steps):
                break
        last = d
        st = steps[si]
        yield {15: AXP15, 5: AXP5, 2: AXP2}[st] if d > 0 else {15: AXM15, 5: AXM5, 2: AXM2}[st]
        moved += d * st
    return abs(moved)


def jcc_power(env, i, rounds=12):
    e = int(env.cur[i])
    net, comp, last = 0, 0, 0
    for _ in range(rounds):
        code, *_ = yield JCC_PW
        if code == 2:
            break
        d = -1 if code == 0 else 1
        if d == 1 and env.C[i, e] >= 0:
            break
        reverse = last and d != last
        yield CYL_MORE if d < 0 else CYL_LESS
        net += d
        if net % 2 == 0:
            need = -net // 2 - comp
            for _ in range(abs(need)):
                yield SP25 if need > 0 else SM25
            comp += need
        if reverse:
            break
        last = d
    return abs(net) * 0.25


def balance(env, i, max_rounds=6):
    last = None
    for _ in range(max_rounds):
        code, *_ = yield BALANCE
        if code == 2 or (last is not None and code != last):
            return
        last = code
        yield R_P if code == 0 else L_P


def mono(env, i, objective=False):
    """One eye, the other covered: starting sphere, best sphere, cylinder, best sphere again if it changed."""
    if not objective:
        yield from find_sphere(env, i)
    yield from mpmva(env, i)
    yield LINE_LG                                        # cross-cylinder one line bigger than the best
    e = int(env.cur[i])
    found = env.C[i, e] < 0 or (yield from cyl_search(env, i))
    changed = 0.25 if found else 0.0
    moved = 0
    if found:
        if abs(env.C[i, e]) >= 1.0:
            moved = yield from jcc_axis(env, i)
            changed += yield from jcc_power(env, i)
        else:
            changed += yield from jcc_power(env, i)
            moved = yield from jcc_axis(env, i)
        if moved >= 10:
            changed += yield from jcc_power(env, i)
    if changed >= 0.5 or (found and moved >= 10):
        yield from mpmva(env, i, fog=2, patience=1)


def bino(env, i):
    """Both eyes open: fog, balance with prism, then clear the fog together."""
    code, base, *_ = yield READ                          # binocular acuity before the fog
    for _ in range(3):
        yield OU_P
    yield from to_line(env, i, 0.3)
    yield from balance(env, i)
    yield from descend(env, i, base, OU_M, OU_P, patience=2)


def textbook3(env, i, objective=False):
    for eye in (0, 1):
        if eye == 1:
            yield NEXT
        yield from mono(env, i, objective)
    yield NEXT                                           # both eyes open
    yield from bino(env, i)
    for _ in range(2):
        code, *_ = yield RX
        # still here: "it's still blurry" (7 right eye, 8 left eye, 9 both). Redo that eye, then both eyes again.
        eyes = {7: (0,), 8: (1,)}.get(code, (0, 1))
        ph = 2
        for e in eyes:
            while ph != e:
                yield NEXT
                ph = (ph + 1) % 3
            yield from mono(env, i)
        while ph != 2:
            yield NEXT
            ph = (ph + 1) % 3
        yield from bino(env, i)


def descend(env, i, base, down, up, patience=2, tol=None):
    """Binocular MPMVA: reduce the plus on both eyes 0.25 at a time, reading the chart; stop after `patience`
    steps with no better reading (not while still two lines worse than before the fog); end on the most plus
    lens within one letter of the best."""
    code, v0, *_ = yield READ
    scores, k, since, best = [v0], 0, 0, v0
    while k < 12 and (since < patience or best > base + 0.2):
        yield down
        k += 1
        code, v, *_ = yield READ
        scores.append(v)
        if v < best - 1e-9:
            best, since = v, 0
        else:
            since += 1
        if since >= 4:
            break
    tol = TOL if tol is None else tol
    target = min(j for j, sc in enumerate(scores) if sc <= best + tol + 1e-9)
    for _ in range(k - target):
        yield up


def plano(env, i):
    yield RX


# ---------------------------------------------------------------- runner
def run3(make=None, test=None, n_par=500, seed=7, agent=None, greedy=True, record_actions=False, start="plano",
         rand_dial=True):
    """Run a script (make) or an AI (agent) on a test set in lock-step; returns per-patient arrays."""
    from clinicians import collect
    env = ClinicEnv(n_par, seed=seed, sampler=test.sampler, record=agent is None, start_mode=start, rand_dial=rand_dial)
    obs = env.obs()
    gens, acts = [None] * n_par, np.zeros(n_par, np.int64)
    if agent is None:
        for i in range(n_par):
            gens[i] = make(env, i)
            acts[i] = next(gens[i])
    out, n_real = {}, 0
    log, seqs = [[] for _ in range(n_par)], {}
    while n_real < test.size:
        if agent is not None:
            acts = agent.act(obs, greedy=greedy)
        pid_now = env.P["pid"].copy()
        obs, _, done, info = env.step(acts)
        if record_actions:
            for i in range(n_par):
                log[i].append(int(acts[i]))
                if done[i]:
                    if pid_now[i] >= 0:
                        seqs[int(pid_now[i])] = log[i]
                    log[i] = []
        if info:
            for k, v in info.items():
                out.setdefault(k, []).extend(v.tolist())
            n_real += int((info["pid"] >= 0).sum())
        if agent is None:
            last = env.last
            for i in range(n_par):
                if done[i]:
                    gens[i] = make(env, i)
                    acts[i] = next(gens[i])
                else:
                    try:
                        acts[i] = gens[i].send((int(last["code"][i]), float(last["va"][i]), int(last["k"][i]),
                                                int(last["nl"][i]), float(last["line"][i])))
                    except StopIteration:
                        acts[i] = RX
    res = collect(out, test.size)
    if record_actions:
        res["actions"] = [seqs[i] for i in range(test.size)]
    return res


def summary5(res):
    return summary3(res)


def summary3(res):
    from clinicians2 import summary2
    s = summary2(res)
    s.update(points=float(np.mean(res["points"])), minutes=float(np.mean(res["secs"])) / 60,
             happy=float(np.mean(res["p_happy"])), annoyed=float(np.mean(res["p_annoyed"])),
             recheck=float(np.mean(res["p_recheck"])), review=float(np.mean(res["p_review"])),
             timeout=float(np.mean(res["timeout"])), e_pt=float(np.mean(res["e_pt"])),
             net=float(np.mean(res["points"]) - clinic_time_cost(np.mean(res["secs"]))))
    return s


def clinic_time_cost(secs):
    import clinic
    return secs / 60 * clinic.POINTS_PER_MIN


if __name__ == "__main__":
    import time
    from env3 import PlanoPairTestSet
    for name, fn in (("plano", plano), ("textbook3", textbook3)):
        t0 = time.time()
        r = run3(fn, PlanoPairTestSet(2000, seed=2026))
        s = summary3(r)
        print(f"{name:10s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()),
              f"({time.time() - t0:.0f}s)")
