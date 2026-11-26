"""Scripted examiners to compare the AI against, plus a lock-step runner for any examiner on a fixed test set.

* textbook: fog +1.00, unfog in 0.25 steps reading the chart, and keep the most plus lens with the best line
  (MPMVA); cross-cylinder axis (bracketing), power (keeping the spherical equivalent), axis again;
  then a final MPMVA re-check of the sphere.
* pleaser: starts at the autorefractor and gives the patient whatever they say looks clearer
  (no fog, no push-plus check). Shows the over-minus trap.

Each examiner is a generator: it yields an action and receives (code, va) back, where code is
0 = "1", 1 = "2", 2 = "same", 3 = a chart reading (va = logMAR line), -1 = no answer (a lens change).
"""
import numpy as np
from env import RefractionEnv, READ, CMP_P, CMP_M, JCC_AX, JCC_PW, DONE, READ_LINE

SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15 = range(10)


def _jcc_axis(env, i):
    step, last_dir, reversals = (15 if abs(env.C[i]) <= 0.5 else 5), 0, 0
    for _ in range(10):
        code, _, *_ = yield JCC_AX
        if code == 2:
            return
        d = 1 if code == 0 else -1                     # follow the minus axis of the preferred flip
        if last_dir and d != last_dir:
            reversals += 1
            if step == 15:
                step = 5
            elif reversals >= 2:
                return
        last_dir = d
        yield (AXP15 if d > 0 else AXM15) if step == 15 else (AXP5 if d > 0 else AXM5)


def _jcc_power(env, i):
    clicks, last = 0, 0
    for _ in range(10):
        code, _, *_ = yield JCC_PW
        if code == 2:
            return
        d = -1 if code == 0 else 1                       # 1 = more minus cylinder, 2 = less
        if d == 1 and env.C[i] >= 0:
            return
        reverse = last and d != last
        yield CYL_MORE if d < 0 else CYL_LESS
        clicks += 1
        if clicks % 2 == 0:                              # keep the spherical equivalent: 0.25 sph per 0.50 cyl
            yield SP25 if d < 0 else SM25
        if reverse:
            return
        last = d


def _cyl_search(env, i):
    """No cylinder in the phoropter: test for it at four orientations 45 degrees apart."""
    for k in range(4):
        code, _, *_ = yield JCC_PW
        if code == 0:
            yield CYL_MORE
            return True
        if k < 3:
            for _ in range(3):
                yield AXP15
    return False


def _mpmva(env, i, patience=2, tol=0.0):
    """From a fogged lens, step -0.25 reading the chart each time; end on the most plus lens whose reading is
    within tol of the best (tol = 0.02 is one letter on the letter chart)."""
    code, v0, *_ = yield READ
    scores, k, since, best = [v0], 0, 0, v0
    while since < patience and k < 16:
        yield SM25
        k += 1
        code, v, *_ = yield READ
        scores.append(v)
        if v < best - 1e-9:
            best, since = v, 0
        else:
            since += 1
    target = min(j for j, sc in enumerate(scores) if sc <= best + tol + 1e-9)
    for _ in range(k - target):
        yield SP25


def _line_check(env, i):
    """Letter chart: isolate the smallest line they can read, add +0.25, and keep it unless they lose a letter."""
    code, v, k0, n0, line = yield READ
    yield SP25
    code, v, k1, n1, _ = yield READ_LINE
    if k1 < k0:
        yield SM25


def textbook(env, i):
    yield SP100                                          # fog +1.00 over the autorefractor
    yield from _mpmva(env, i)                            # maximum plus to maximum visual acuity
    found = env.C[i] < 0
    if not found:
        found = yield from _cyl_search(env, i)
    if found:
        yield from _jcc_axis(env, i)                     # refine axis, then power, then axis again
        yield from _jcc_power(env, i)
        yield from _jcc_axis(env, i)
    yield SP25                                           # re-check the sphere: slight fog, MPMVA again
    yield from _mpmva(env, i, patience=1)
    yield DONE


def textbook_scratch(env, i):
    """No autorefractor: the phoropter starts at plano. Find the sphere by chart in 1.00 D steps, then the
    textbook sequence (fog, MPMVA, cylinder search, cross-cylinder, MPMVA re-check)."""
    code, best, *_ = yield READ
    go = 0
    if best >= 0.95:                                     # can't read the top line: compare -2.00 with +2.00
        yield SM100
        yield SM100
        code, vm, *_ = yield READ
        for _ in range(4):
            yield SP100
        code, vp, *_ = yield READ
        yield SM100
        yield SM100
        go = -1 if vm <= vp else 1
    moved = 0
    if go <= 0:
        for _ in range(10):                              # myopia: more minus while the line improves
            yield SM100                                  # (or while a myope still can't read the top line)
            code, v, *_ = yield READ
            if v < best or (go == -1 and v <= best and best >= 0.95):
                best, moved = min(best, v), moved + 1
            else:
                yield SP100
                break
    if go == 1 or (go == 0 and moved <= 1):              # plus: more plus until the chart clearly drops
        for _ in range(9):
            yield SP100
            code, v, *_ = yield READ
            if v <= best + 0.1:
                best = min(best, v)
            else:
                yield SM100
                break
    tol = 0.02 if env.letters else 0.0
    yield SP100                                          # fog, then MPMVA
    yield from _mpmva(env, i, tol=tol)
    found = yield from _cyl_search(env, i)
    if found:
        yield from _jcc_axis(env, i)
        for _ in range(2):                               # large cylinders take several rounds
            yield from _jcc_power(env, i)
        yield from _jcc_axis(env, i)
    yield SP25
    yield from _mpmva(env, i, patience=1, tol=tol)
    if env.letters:
        yield from _line_check(env, i)
    yield DONE


def textbook_letters(env, i):
    """The textbook sequence on the letter chart: MPMVA with a one-letter tolerance, and a final isolated-line
    push-plus check ("add +0.25: did you lose any letters?")."""
    yield SP100
    yield from _mpmva(env, i, tol=0.02)
    found = env.C[i] < 0
    if not found:
        found = yield from _cyl_search(env, i)
    if found:
        yield from _jcc_axis(env, i)
        yield from _jcc_power(env, i)
        yield from _jcc_axis(env, i)
    yield SP25
    yield from _mpmva(env, i, patience=1, tol=0.02)
    yield from _line_check(env, i)
    yield DONE


def pleaser(env, i):
    for _ in range(16):
        code, _, *_ = yield CMP_M
        if code == 1:
            yield SM25
            continue
        code, _, *_ = yield CMP_P
        if code == 1:
            yield SP25
            continue
        break
    if env.C[i] < 0:
        yield from _jcc_axis(env, i)
        yield from _jcc_power(env, i)
    yield DONE


def run_scripted(make, test, n_par=500, seed=7, record_actions=False, t_max=60, letters=False):
    """Run a scripted examiner over a TestSet. Returns a dict of arrays indexed by patient id.
    Each examiner receives (code, va, letters correct, letters on the line, line) after every move."""
    env = RefractionEnv(n_par, seed=seed, sampler=test.sampler, record=True, letters=letters)
    env.t_max = t_max
    gens, acts = [], np.zeros(n_par, np.int64)
    for i in range(n_par):
        g = make(env, i)
        gens.append(g)
        acts[i] = next(g)
    out, n_real = {}, 0
    log, seqs = [[] for _ in range(n_par)], {}
    while n_real < test.size:
        pid_now = env.P["pid"].copy()
        _, _, done, info = env.step(acts)
        if record_actions:
            for i in range(n_par):
                log[i].append(int(acts[i]))
                if done[i]:
                    if pid_now[i] >= 0:
                        seqs[int(pid_now[i])] = log[i]
                    log[i] = []
        last = env.last
        if info:
            for k, v in info.items():
                out.setdefault(k, []).extend(v.tolist())
            n_real += int((info["pid"] >= 0).sum())
        for i in range(n_par):
            if done[i]:
                gens[i] = make(env, i)
                acts[i] = next(gens[i])
            else:
                try:
                    acts[i] = gens[i].send((int(last["code"][i]), float(last["va"][i]), int(last["k"][i]),
                                            int(last["nl"][i]), float(last["line"][i])))
                except StopIteration:
                    acts[i] = DONE
    res = collect(out, test.size)
    if record_actions:
        res["actions"] = [seqs[i] for i in range(test.size)]
    return res


def collect(out, size):
    pid = np.array(out["pid"])
    keep = pid >= 0
    res = {k: np.array(v)[keep] for k, v in out.items()}
    order = np.argsort(res["pid"])
    res = {k: v[order] for k, v in res.items()}
    assert len(res["pid"]) == size and (res["pid"] == np.arange(size)).all()
    return res


def summary(res, key="err"):
    """key="err": the full-cylinder score (optics.rx_error); key="err_half": plain power-vector distance."""
    young = res["age"] < 40
    e = res[key]
    return dict(
        exams=len(e),
        mean_err=float(np.mean(e)),
        within_025=float(np.mean(e <= 0.25 + 1e-9)),
        exact=float(np.mean(e < 0.13)),
        steps=float(np.mean(res["steps"])),
        over_minus_young=float(np.mean(res["dM"][young] <= -0.25 + 1e-9)),
        over_plus=float(np.mean(res["dM"] >= 0.25 - 1e-9)),
        start_within_025=float(np.mean(res["err0"] <= 0.25 + 1e-9)),
    )


if __name__ == "__main__":
    from patients import TestSet
    import time
    for name, fn in (("textbook", textbook), ("pleaser", pleaser)):
        t0 = time.time()
        r = run_scripted(fn, TestSet(5000))
        s = summary(r)
        print(f"{name:9s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()),
              f"({time.time() - t0:.0f}s)")
