"""Scripted two-eye examiners for the two-eye exam room (env2.py), and a runner.

* textbook2: each eye in turn: fog +1.00, unfog by chart (MPMVA, one-letter tolerance), cylinder search if
  none, cross-cylinder on a line one bigger than the best (axis, power, axis), MPMVA re-check and an
  isolated-line push-plus check. Then binocular balance: both eyes +0.75, 20/40 line, prism-split, +0.25 to
  the clearer eye until they match (if the answer flips, the last +0.25 comes back off); prism out;
  binocular MPMVA (reduce plus 0.25 at a time; the most plus within a letter of the best).
* mine: the channel's own routine, as described by the optometrist:
    autorefractor; read the smallest line; fog +0.50 and confirm it is blurrier; add minus 0.25 at a time
    while they read more or smaller letters, and stop when no more letters come; go two lines bigger (about
    20/40) for the cross-cylinder; power first, stop when unsure or they backtrack; then axis the same way;
    if the axis moved more than 10 degrees, power again; other eye. Binocular balance: prism-split, 20/40,
    both eyes +0.50, add +0.25 to the clearer eye until they are the same, prism out, take the +0.50 back
    off both eyes, and check 20/20 both eyes.
  On a flip in the balance answer, the right eye is left the slightly clearer one (the optometrist's rule:
  most people are right-eye dominant). Two details the description leaves open are filled the standard way: with no cylinder in the phoropter, cylinder is searched for at four axes; and if
  both eyes together do not read as well as the better eye did alone, -0.25 both eyes (up to twice).

Each examiner is a generator that yields actions and receives (code, va, letters correct, letters on line, line).
"""
import numpy as np
from env2 import (BinocularEnv, READ, CMP_P, CMP_M, JCC_AX, JCC_PW, DONE, LINE_SM, LINE_LG, READ_LINE, TEST_OD,
                  TEST_OS, BOTH, OU_P, OU_M, BALANCE)

SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15 = range(10)


# ---------------------------------------------------------------- building blocks (act on the tested eye)
def to_line(env, i, target):
    """Move the projector to a given line."""
    for _ in range(14):
        cur = round(float(env.line[i]), 1)
        if abs(cur - target) < 0.01:
            return
        yield LINE_LG if cur < target else LINE_SM


def bigger_lines(env, i, k):
    for _ in range(k):
        yield LINE_LG


def mpmva(env, i, patience=2, tol=0.02):
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


def cyl_search(env, i):
    for k in range(4):
        code, *_ = yield JCC_PW
        if code == 0:
            yield CYL_MORE
            return True
        if k < 3:
            for _ in range(3):
                yield AXP15
    return False


def jcc_axis(env, i, start_step=None):
    """Bracket the axis. Returns the total rotation (degrees)."""
    e = int(env.cur[i])
    step = start_step or (15 if abs(env.C[i, e]) <= 0.5 else 5)
    last, reversals, moved = 0, 0, 0
    for _ in range(10):
        code, *_ = yield JCC_AX
        if code == 2:
            break
        d = 1 if code == 0 else -1
        if last and d != last:
            reversals += 1
            if step == 15:
                step = 5
            elif reversals >= 2:
                break
        last = d
        yield (AXP15 if d > 0 else AXM15) if step == 15 else (AXP5 if d > 0 else AXM5)
        moved += d * step
    return abs(moved)


def jcc_power(env, i, rounds=10):
    e = int(env.cur[i])
    clicks, last = 0, 0
    for _ in range(rounds):
        code, *_ = yield JCC_PW
        if code == 2:
            return
        d = -1 if code == 0 else 1
        if d == 1 and env.C[i, e] >= 0:
            return
        reverse = last and d != last
        yield CYL_MORE if d < 0 else CYL_LESS
        clicks += 1
        if clicks % 2 == 0:
            yield SP25 if d < 0 else SM25
        if reverse:
            return
        last = d


def line_check(env, i):
    code, v, k0, n0, line = yield READ
    yield SP25
    code, v, k1, *_ = yield READ_LINE
    if k1 < k0:
        yield SM25


def balance(env, i, max_rounds=6, keep_right_clearer=False):
    """Prism-split; +0.25 to the clearer eye until they look the same. If the answer flips, the last +0.25
    overshot. Textbook: take it back off. keep_right_clearer (the optometrist's routine): end with the right
    eye the slightly clearer one: take the +0.25 back off the right eye, but leave it on the left eye."""
    last = None
    for _ in range(max_rounds):
        code, *_ = yield BALANCE
        if code == 2:
            return
        if last is not None and code != last:
            if not keep_right_clearer or last == 0:
                yield TEST_OD if last == 0 else TEST_OS
                yield SM25
            return
        last = code
        yield TEST_OD if code == 0 else TEST_OS              # select the clearer eye's dial
        yield SP25


def mpmva_ou(env, i, patience=2, tol=0.02, max_steps=8):
    """Binocular MPMVA: reduce plus on both eyes 0.25 at a time; end on the most plus within a letter of the best."""
    code, v0, *_ = yield READ
    scores, k, since, best = [v0], 0, 0, v0
    while since < patience and k < max_steps:
        yield OU_M
        k += 1
        code, v, *_ = yield READ
        scores.append(v)
        if v < best - 1e-9:
            best, since = v, 0
        else:
            since += 1
    target = min(j for j, sc in enumerate(scores) if sc <= best + tol + 1e-9)
    for _ in range(k - target):
        yield OU_P


# ---------------------------------------------------------------- the textbook
def textbook2(env, i):
    for eye in (TEST_OD, TEST_OS):
        yield eye
        e = int(env.cur[i])
        yield SP100
        yield from mpmva(env, i)
        found = env.C[i, e] < 0
        yield LINE_LG                                        # cross-cylinder one line bigger than the best
        if not found:
            found = yield from cyl_search(env, i)
        if found:
            yield from jcc_axis(env, i)
            yield from jcc_power(env, i)
            yield from jcc_axis(env, i)
        yield SP25
        yield from mpmva(env, i, patience=1)
        yield from line_check(env, i)
    yield BOTH                                               # binocular balance
    for _ in range(3):
        yield OU_P
    yield from to_line(env, i, 0.3)
    yield from balance(env, i)
    yield BOTH                                               # prism out; binocular MPMVA
    yield from mpmva_ou(env, i, patience=1)
    yield DONE


# ---------------------------------------------------------------- the optometrist's own routine
def mine(env, i):
    best_eye = []
    for eye in (TEST_OD, TEST_OS):
        yield eye
        e = int(env.cur[i])
        code, v0, *_ = yield READ                            # smallest line they can read
        for _ in range(3):                                   # fog +0.50 and confirm it is blurrier
            yield SP25
            yield SP25
            code, v, *_ = yield READ
            if v > v0 + 1e-9:
                break
        best = v
        for _ in range(24):                                  # add minus while they get more letters
            yield SM25
            code, v, *_ = yield READ
            if v < best - 1e-9:
                best = v
            else:
                yield SP25                                   # that lens gained nothing: back to the last one
                break
        best_eye.append(best)
        yield from bigger_lines(env, i, 2)                   # two lines bigger (about 20/40)
        found = env.C[i, e] < 0
        if not found:
            found = yield from cyl_search(env, i)
        if found:
            yield from jcc_power(env, i)                     # power first
            moved = yield from jcc_axis(env, i)              # then axis
            if moved > 10:
                yield from jcc_power(env, i)                 # axis moved a lot: power again
    yield BOTH                                               # binocular balance
    yield from to_line(env, i, 0.3)                          # 20/40
    yield OU_P
    yield OU_P                                               # +0.50 both eyes
    yield from balance(env, i, keep_right_clearer=True)
    yield BOTH                                               # prism out
    yield OU_M
    yield OU_M                                               # take the +0.50 back off
    code, v, *_ = yield READ                                 # check 20/20 both eyes
    for _ in range(2):                                       # not as good as the better eye alone: a little minus
        if v <= min(best_eye) + 0.02 + 1e-9:
            break
        yield OU_M
        code, v, *_ = yield READ
    yield DONE


# ---------------------------------------------------------------- runner
def run2(make, test, n_par=500, seed=7, record_actions=False, t_max=120):
    from clinicians import collect
    env = BinocularEnv(n_par, seed=seed, sampler=test.sampler, record=True)
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
        last = env.last
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


def summary2(res):
    """Both eyes pooled: every eye counts once."""
    e = np.concatenate([res["err"], res["err_os"]])
    dM = np.concatenate([res["dM"], res["dM_os"]])
    young = np.concatenate([res["age"], res["age"]]) < 40
    e0 = np.concatenate([res["err0"], res["err0_os"]])
    return dict(exams=len(res["err"]), eyes=len(e), mean_err=float(e.mean()),
                within_025=float(np.mean(e <= 0.25 + 1e-9)), exact=float(np.mean(e < 0.13)),
                both_within=float(np.mean((res["err"] <= 0.25 + 1e-9) & (res["err_os"] <= 0.25 + 1e-9))),
                steps=float(np.mean(res["steps"])), over_minus_young=float(np.mean(dM[young] <= -0.25 + 1e-9)),
                over_plus=float(np.mean(dM >= 0.25 - 1e-9)), start_within_025=float(np.mean(e0 <= 0.25 + 1e-9)))


if __name__ == "__main__":
    import time
    from patients import PairTestSet
    for name, fn in (("textbook2", textbook2), ("mine", mine)):
        t0 = time.time()
        r = run2(fn, PairTestSet(3000, seed=2026))
        s = summary2(r)
        print(f"{name:10s} " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()),
              f"({time.time() - t0:.0f}s)")
