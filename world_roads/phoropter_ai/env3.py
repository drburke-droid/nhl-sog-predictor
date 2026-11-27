"""The clinic exam (the video's AI): two eyes, every exam starts at plano, scored by the clinic.

The AI knows nothing about eyes. It has the buttons below and the patient's answers, and it is paid in
clinic points (clinic.py): a happy patient who returns and refers is worth +100; an annoyed one +30; an
Rx check -80; an Rx check with a bad review -150; and every minute of chair time costs 4 points.

The exam runs in the phoropter's natural order: right eye (left occluded), then left eye, then both eyes.
The AI decides when to press "Next" and when to prescribe. Buttons that do nothing in the current stage are
switched off (masked): cylinder, axis and cross-cylinder with both eyes open; the both-eyes buttons while one
eye is covered. "Prescribe" works at any time: it hands the patient whatever is in the phoropter.

The patient is the two-eye patient of env2.py (yoked accommodation, proximal accommodation behind the
phoropter, line-size-dependent judgement, ocular dominance in a prism-split balance, noise, a "same" band,
a preference for minus, lapses), with both eyes starting at plano. The projector starts on 20/200.
"""
import numpy as np
from optics import to_pv, rx_error, jcc_pv
from patients import sample_pair, PairTestSet, wrap_axis
from env2 import BinocularEnv, gain
import chart
import clinic

ACTS = ["Sphere +0.25", "Sphere -0.25", "Sphere +1.00", "Sphere -1.00", "Cylinder -0.25", "Cylinder +0.25",
        "Axis +5°", "Axis -5°", "Axis +15°", "Axis -15°", "Axis +2°", "Axis -2°",
        "Read the smallest line you can", "Read this line", "Line smaller", "Line larger",
        "1 or 2? (+0.25)", "1 or 2? (-0.25)", "Cross-cyl axis: 1 or 2?", "Cross-cyl power: 1 or 2?",
        "Next eye", "+0.25 both eyes", "-0.25 both eyes", "Balance: top or bottom?",
        "+0.25 right eye", "+0.25 left eye", "Prescribe"]
N_ACT3 = len(ACTS)
(SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15, AXP2, AXM2, READ, READ_LINE, LINE_SM,
 LINE_LG, CMP_P, CMP_M, JCC_AX, JCC_PW, NEXT, OU_P, OU_M, BALANCE, R_P, L_P, RX) = range(N_ACT3)

# seconds of chair time per button (a lens click is quick; a question takes a flip, "1... or 2?" and an answer)
SECS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # lens clicks
                 8.0, 4.0, 1.0, 1.0,                                        # smallest line; this line; line size
                 4.0, 4.0, 4.0, 4.0,                                        # "1 or 2?" questions
                 2.0, 1.0, 1.0, 5.0, 1.0, 1.0, 0.0])                        # next, both eyes, balance, prescribe
SPH = np.zeros(N_ACT3); SPH[[SP25, SM25, SP100, SM100]] = [0.25, -0.25, 1.0, -1.0]
CYL = np.zeros(N_ACT3); CYL[[CYL_MORE, CYL_LESS]] = [-0.25, 0.25]
AXD = np.zeros(N_ACT3); AXD[[AXP5, AXM5, AXP15, AXM15, AXP2, AXM2]] = [5, -5, 15, -15, 2, -2]
LND = np.zeros(N_ACT3); LND[[LINE_SM, LINE_LG]] = [-0.1, 0.1]
OUD = np.zeros(N_ACT3); OUD[[OU_P, OU_M]] = [0.25, -0.25]
RD = np.zeros(N_ACT3); RD[R_P] = 0.25
LD = np.zeros(N_ACT3); LD[L_P] = 0.25

MASK = np.zeros((3, N_ACT3), bool)
MASK[0, list(range(0, NEXT + 1)) + [RX]] = True              # one eye open: lenses, chart, questions, next, prescribe
MASK[1] = MASK[0]
MASK[2, [READ, READ_LINE, LINE_SM, LINE_LG, CMP_P, CMP_M, OU_P, OU_M, BALANCE, R_P, L_P, RX]] = True

QS = (READ, READ_LINE, CMP_P, CMP_M, JCC_AX, JCC_PW)          # answers remembered per stage
COMPARE = (CMP_P, CMP_M, JCC_AX, JCC_PW, BALANCE)
T_MAX3 = 400                                                  # button presses (a safety cap: the 15-minute limit is the real one; the AI trained with 200 and never came near it)
SECS_MAX = 900.0                                              # 15 minutes: the exam is cut short
H3 = 16
F3 = N_ACT3 + 5 + 3 + 5
N_OBS3 = 8 + 14 + 10 + 3 * len(QS) * 10 + 7 + H3 * F3 + N_ACT3

# what the patient can say
ANSWERS = {
    "compare": ("1 is clearer", "2 is clearer", "They look the same"),
    "balance": ("Top is clearer", "Bottom is clearer", "They look equal"),
    "line": ("I can read all of them", "I can read most of them", "I can read a few", "I can't read any"),
}


def plano_pair(rng, n, hard=0.0, axis_mix=0.0):
    """Two-eyed patients for the clinic: both eyes start at plano. The axis dial starts wherever the last
    patient left it (uniform 1-180), which with no cylinder in the phoropter changes nothing optically.
    axis_mix (practice term 3 only): this fraction of patients get their astigmatism at any angle (uniform),
    instead of mostly near 180 or 90; their left eye is the usual near-mirror image."""
    p = sample_pair(rng, n, hard=hard)
    if axis_mix > 0:
        m = rng.uniform(size=n) < axis_mix
        ax = rng.uniform(0, 180, n)
        p["axis"] = np.where(m, wrap_axis(ax), p["axis"])
        p["axis_os"] = np.where(m, wrap_axis(180 - ax + rng.normal(0, 8, n)), p["axis_os"])
    for k in ("S0", "C0", "S0_os", "C0_os"):
        p[k] = np.zeros(n)
    p["ax0"], p["ax0_os"] = np.full(n, 180.0), np.full(n, 180.0)
    p["dial0"] = rng.integers(1, 181, n).astype(float)
    p["dial0_os"] = rng.integers(1, 181, n).astype(float)
    return p


def ar_pair(rng, n, hard=0.0, axis_mix=0.0):
    """Two-eyed patients whose exam starts at their autorefractor readings. The same autorefractor model as
    patients.py: spherical-equivalent error N(mu, 0.30) with mu = -0.25 D under 40 (instrument myopia), -0.10 D
    to 55, 0 after; astigmatism error N(0, 0.10) per J component; 12% poor readings (0.9 D and 0.3 D). The left
    eye's error is correlated with the right's. Use with ClinicEnv(start_mode="objective")."""
    from optics import to_pv, from_pv
    from patients import q as quant
    p = plano_pair(rng, n, hard=hard, axis_mix=axis_mix)
    age = p["age"]
    mu = np.where(age < 40, -0.25, np.where(age < 55, -0.10, 0.0))
    errs = {}
    for suf in ("", "_os"):
        bad = rng.uniform(size=n) < 0.12
        sM, sJ = np.where(bad, 0.9, 0.30), np.where(bad, 0.30, 0.10)
        if suf == "":
            dM = rng.normal(mu, sM)
            errs["od"] = dM
        else:
            dM = 0.5 * (errs["od"] - mu) + mu + rng.normal(0, sM * 0.85)
        E = to_pv(p["S" + suf], p["C" + suf], p["axis" + suf])
        S0, C0, a0 = from_pv(E[0] + dM, E[1] + rng.normal(0, sJ), E[2] + rng.normal(0, sJ))
        p["S0" + suf], p["C0" + suf], p["ax0" + suf] = quant(S0), quant(C0), wrap_axis(a0)
    return p


class ARPairTestSet(PairTestSet):
    """A fixed set of two-eyed patients with autorefractor readings (exams start there)."""

    def __init__(self, size, seed=12345):
        self.size = size
        self.start = "objective"
        self.pool = ar_pair(np.random.default_rng(seed), size)
        self.pool["pid"] = np.arange(size, dtype=np.float64)
        self.next = 0

    def sampler(self, rng, n):
        take = np.arange(self.next, self.next + n)
        self.next += n
        out = ar_pair(rng, n)
        out["pid"] = np.full(n, -1.0)
        real = take < self.size
        for k, v in self.pool.items():
            out[k][real] = v[take[real]]
        return out


class PlanoPairTestSet(PairTestSet):
    def __init__(self, size, seed=12345):
        super().__init__(size, seed)
        for k in ("S0", "C0", "S0_os", "C0_os"):
            self.pool[k][:] = 0.0
        self.pool["ax0"][:], self.pool["ax0_os"][:] = 180.0, 180.0
        r = np.random.default_rng(seed + 7)
        self.pool["dial0"] = r.integers(1, 181, size).astype(float)
        self.pool["dial0_os"] = r.integers(1, 181, size).astype(float)

    def sampler(self, rng, n):
        take = np.arange(self.next, self.next + n)
        self.next += n
        out = plano_pair(rng, n)
        out["pid"] = np.full(n, -1.0)
        real = take < self.size
        for k, v in self.pool.items():
            out[k][real] = v[take[real]]
        return out


class ClinicEnv(BinocularEnv):
    def __init__(self, n, seed=0, sampler=plano_pair, record=False, start_mode="plano", rand_dial=True):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        self.record = record
        self.n_act, self.n_obs, self.letters, self.flag = N_ACT3, N_OBS3, True, False
        z = lambda *s: np.zeros((n,) + s)
        self.P = {}
        self.S, self.C, self.ax = z(2), z(2), z(2)
        self.phase = np.zeros(n, np.int64)
        self.cur = np.zeros(n, np.int64)
        self.bin = np.zeros(n)
        self.line = np.full(n, 1.0)
        self.t, self.secs = z(), z()
        self.hist_a = np.full((n, H3), -1, np.int64)
        self.hist_r, self.hist_m, self.hist_l = z(H3, 5), z(H3, 3), z(H3, 5)
        nq = len(QS)
        self.lq_ans, self.lq_on, self.lq_l, self.lq_t = z(3, nq, 3), z(3, nq), z(3, nq, 5), z(3, nq)
        self.bal_ans, self.bal_on, self.bal_m, self.bal_t = z(3), z(), z(2), z()
        self.best_va, self.best_on, self.best_l = z(2), z(2), z(2, 5)
        self.start_pv = z(2, 3)
        self.shaping, self.gamma, self.t_max, self.cyl_weight = 0.0, 0.995, T_MAX3, 2.0
        self.secs_max = SECS_MAX
        self.rand_dial = rand_dial                           # practice term 1 used False: the dial always at 180
        self.start_mode = start_mode                       # "objective": start at the sampler's S0/C0/ax0 (calibration)
        self.rows = np.arange(n)
        self._reset(np.arange(n))

    def _reset(self, idx):
        if len(idx) == 0:
            return
        p = self.sampler(self.rng, len(idx))
        for k, v in p.items():
            if k not in self.P:
                self.P[k] = np.zeros(self.n)
            self.P[k][idx] = v
        self.S[idx], self.C[idx], self.ax[idx] = 0.0, 0.0, 180.0          # plano, both eyes
        if self.rand_dial and "dial0" in p:                                 # the dial where the last patient left it
            self.ax[idx, 0], self.ax[idx, 1] = p["dial0"], p["dial0_os"]
        if self.start_mode == "objective":
            self.S[idx, 0], self.C[idx, 0], self.ax[idx, 0] = p["S0"], p["C0"], p["ax0"]
            self.S[idx, 1], self.C[idx, 1], self.ax[idx, 1] = p["S0_os"], p["C0_os"], p["ax0_os"]
        self.phase[idx], self.cur[idx], self.bin[idx], self.line[idx] = 0, 0, 0, 1.0
        self.t[idx], self.secs[idx] = 0, 0
        self.hist_a[idx] = -1
        for arr in (self.hist_r, self.hist_m, self.hist_l, self.lq_ans, self.lq_on, self.lq_l, self.lq_t, self.bal_ans,
                    self.bal_on, self.bal_m, self.bal_t, self.best_va, self.best_on, self.best_l, self.start_pv):
            arr[idx] = 0

    def mask(self):
        return MASK[self.phase]

    # ---------------------------------------------------------------- the patient's view of each question
    def _views(self):
        c, b = self.cur, self.bin > 0
        L0, L1 = self.lens_pv(0), self.lens_pv(1)
        Lc = tuple(np.where(c == 0, x0, x1) for x0, x1 in zip(L0, L1))
        Tm = np.where(c == 0, self.mono_T(0, Lc), self.mono_T(1, Lc))
        t0, t1, tb = self.bino(L0, L1)
        return c, b, L0, L1, Lc, Tm, t0, t1, tb

    def _mono(self, c, L):
        return np.where(c == 0, self.mono_T(0, L), self.mono_T(1, L))

    def _compare_terms(self, act, V):
        """For a '1 or 2?' question: thresholds through 1 and 2, the minus-preference terms, the kind of
        judgement, the dominance lean, whether the answer is forced to 'same', and the lenses compared
        (for working out which answer is truly right)."""
        c, b, L0, L1, Lc, Tm, t0, t1, tb = V
        r = self.rows
        n = self.n
        zero = np.zeros(n)
        if act in (CMP_P, CMP_M):
            ds = 0.25 if act == CMP_P else -0.25
            Lb = (Lc[0] + ds, Lc[1], Lc[2])
            _, _, qb_b = self.bino((L0[0] + ds, L0[1], L0[2]), (L1[0] + ds, L1[1], L1[2]))
            qa = np.where(b, tb, Tm)
            qb = np.where(b, qb_b, self._mono(c, Lb))
            ea = np.where(b, rx_error(L0, self.true_pv(0)) + rx_error(L1, self.true_pv(1)), rx_error(Lc, self._tpv(c)))
            eb = np.where(b, rx_error((L0[0] + ds, L0[1], L0[2]), self.true_pv(0)) +
                          rx_error((L1[0] + ds, L1[1], L1[2]), self.true_pv(1)), rx_error(Lb, self._tpv(c)))
            return qa, qb, Lc[0], Lc[0] + ds, "sph", zero, np.zeros(n, bool), ea, eb
        if act in (JCC_AX, JCC_PW):
            oa, ob = (45, -45) if act == JCC_AX else (0, 90)
            axc = self.ax[r, c]
            ja, jb = jcc_pv(axc + oa), jcc_pv(axc + ob)
            A_ = (Lc[0], Lc[1] + ja[1], Lc[2] + ja[2])
            B_ = (Lc[0], Lc[1] + jb[1], Lc[2] + jb[2])
            ea, eb = rx_error(A_, self._tpv(c)), rx_error(B_, self._tpv(c))
            return self._mono(c, A_), self._mono(c, B_), Lc[0], Lc[0], "jcc", zero, b.copy(), ea, eb
        # BALANCE: prism split, top = right eye, bottom = left eye; the dominant eye looks a little clearer
        return t0, t1, zero, zero, "sph", 0.02 * self.P["dom"], np.zeros(n, bool), t0, t1

    def _tpv(self, c):
        T0, T1 = self.true_pv(0), self.true_pv(1)
        return tuple(np.where(c == 0, x0, x1) for x0, x1 in zip(T0, T1))

    def compare_probs(self, act, V=None):
        """Exact chances of each answer (1, 2, same) to a '1 or 2?' question, and the answer that is truly
        right (0, 1 or 2 = they really are equally good)."""
        from scipy.stats import norm
        V = self._views() if V is None else V
        qa, qb, Ma, Mb, kind, lean, same, ea, eb = self._compare_terms(act, V)
        g = gain(self.line - 0.5 * (qa + qb), kind)
        mu = g * (qa - qb) + self.P["bias"] * (Ma - Mb) - lean
        s, tau, lap = self.P["sigma"], self.P["tau"], self.P["lapse"]
        p2 = 1 - norm.cdf((tau - mu) / s)
        p1 = norm.cdf((-tau - mu) / s)
        P = np.stack([p1, p2, 1 - p1 - p2], -1) * (1 - lap)[:, None] + lap[:, None] / 3
        P[same] = [0.0, 0.0, 1.0]
        tol = 0.01 if act == BALANCE else 0.02
        right = np.where(ea < eb - tol, 0, np.where(eb < ea - tol, 1, 2))
        return P, right

    def line_probs(self, V=None):
        """Chances of all / most / a few / none of the letters on the line on screen, through the lenses now."""
        V = self._views() if V is None else V
        c, b, *_ = V
        T = np.where(b, V[8], V[5])
        n = chart.n_letters(self.line)
        xs, ws = np.polynomial.hermite_e.hermegauss(9)
        ws = ws / ws.sum()
        out = np.zeros((self.n, 4))
        from scipy.stats import binom
        for x, w in zip(xs, ws):
            p = chart.p_correct(self.line, T + chart.JITTER * x)
            pk0 = binom.pmf(0, n, p)
            pall = binom.pmf(n, n, p)
            pmost = 1 - binom.cdf(n // 2, n, p) - pall        # more than half, not all
            few = 1 - pk0 - pall - np.maximum(pmost, 0)
            out += w * np.stack([pall, np.maximum(pmost, 0), few, pk0], -1)
        return out

    # ---------------------------------------------------------------- step
    def step(self, a):
        a = np.asarray(a, np.int64)
        n, r = self.n, self.rows
        ok = MASK[self.phase, a]
        a = np.where(ok, a, RX)                               # (the policy never picks a masked button)
        self.t += 1
        self.secs += SECS[a]
        rew = -SECS[a] * clinic.POINTS_PER_MIN / 60 / clinic.SCALE
        if self.shaping:
            phi0 = -2.0 * self.err(self.cyl_weight).mean(1)

        # lens and stage changes
        c = self.cur
        self.S[r, c] = np.clip(self.S[r, c] + SPH[a], -20, 20)
        self.C[r, c] = np.clip(self.C[r, c] + CYL[a], -6, 0)
        self.ax[r, c] = ((self.ax[r, c] + AXD[a] - 1) % 180) + 1
        self.S = np.clip(self.S + OUD[a][:, None], -20, 20)
        self.S[:, 0] = np.clip(self.S[:, 0] + RD[a], -20, 20)
        self.S[:, 1] = np.clip(self.S[:, 1] + LD[a], -20, 20)
        self.line = np.round(np.clip(self.line + LND[a], -0.3, 1.0), 1)
        self.phase = np.where(a == NEXT, np.minimum(self.phase + 1, 2), self.phase)
        self.cur = np.minimum(self.phase, 1)
        self.bin = (self.phase == 2).astype(float)
        V = self._views()
        c, b, L0, L1, Lc, Tm, t0, t1, tb = V
        Tview = np.where(b, tb, Tm)

        resp = np.zeros((n, 5))
        code = np.full(n, -1)
        kk, nn, line_read = np.zeros(n), np.ones(n), self.line.copy()
        va = np.zeros(n)
        rd, rl = a == READ, a == READ_LINE
        rec = {}
        if self.record and (rl.any() or np.isin(a, COMPARE).any()):
            if rl.any():
                rec["probs"] = self.line_probs(V)
            else:
                rec["probs"], rec["right"] = self.compare_probs(int(a[0]), V)
        if rd.any():
            if self.record:
                lines, _, _ = chart.read_all(np.repeat(Tview[rd][:1], 4000), self.rng)
                vals, cnt = np.unique(lines, return_counts=True)
                rec["read_dist"] = dict(zip(vals.tolist(), (cnt / cnt.sum()).tolist()))
            ln, k, m = chart.read_all(Tview[rd], self.rng)
            va[rd] = chart.score(ln, k, m)
            kk[rd], nn[rd], line_read[rd] = k, m, ln
            self.line[rd] = np.minimum(ln, 1.0)
        if rl.any():
            k, m = chart.read_line(self.line[rl], Tview[rl], self.rng)
            kk[rl], nn[rl] = k, m
            va[rl] = self.line[rl]
        resp[rd | rl, 0] = line_read[rd | rl]
        resp[rd | rl, 1] = (kk / nn)[rd | rl]
        code = np.where(rd, 3, np.where(rl, 4, code))

        ans = np.full(n, 2)
        for act in COMPARE:
            m = a == act
            if not m.any():
                continue
            qa, qb, Ma, Mb, kind, lean, same, _, _ = self._compare_terms(act, V)
            res = self.judge(qa, qb, Ma, Mb, kind, lean)
            ans = np.where(m, np.where(same, 2, res), ans)
        cm = np.isin(a, COMPARE)
        resp[cm, 2 + ans[cm]] = 1.0
        code = np.where(cm, ans, code)
        if self.record:
            self.last = dict(a=a.copy(), code=code, va=va, k=kk, nl=nn, line=line_read, screen=self.line.copy(),
                             cur=self.cur.copy(), bin=self.bin.copy(), phase=self.phase.copy(), T0=t0, T1=t1,
                             Tview=Tview, secs=self.secs.copy(), **rec)

        # memory: slot 0 right eye, 1 left eye, 2 both eyes
        slot = np.where(b, 2, c)
        cos2, sin2 = np.cos(np.deg2rad(2 * self.ax[r, c])), np.sin(np.deg2rad(2 * self.ax[r, c]))
        lf = np.where(b[:, None], np.stack([L0[0], L1[0], np.zeros(n), np.zeros(n), np.zeros(n)], -1),
                      np.stack([Lc[0], Lc[1], Lc[2], cos2, sin2], -1))
        for arr in (self.hist_a, self.hist_r, self.hist_m, self.hist_l):
            arr[:] = np.roll(arr, 1, 1)
        self.hist_a[:, 0], self.hist_r[:, 0] = a, resp
        self.hist_m[:, 0] = np.eye(3)[self.phase]
        self.hist_l[:, 0] = lf
        for k, act in enumerate(QS):
            m = a == act
            if m.any():
                v = np.stack([va, kk / nn, np.zeros(n)], -1) if act in (READ, READ_LINE) else resp[:, 2:]
                self.lq_ans[r[m], slot[m], k] = v[m]
                self.lq_on[r[m], slot[m], k] = 1
                self.lq_l[r[m], slot[m], k] = lf[m]
                self.lq_t[r[m], slot[m], k] = self.t[m]
        mb = a == BALANCE
        if mb.any():
            self.bal_ans[mb] = resp[mb, 2:]
            self.bal_on[mb] = 1
            self.bal_m[mb] = np.stack([L0[0], L1[0]], -1)[mb]
            self.bal_t[mb] = self.t[mb]
        better = rd & ~b & ((self.best_on[r, c] == 0) | (va <= self.best_va[r, c]))
        self.best_va[r[better], c[better]] = va[better]
        self.best_on[r[better], c[better]] = 1
        self.best_l[r[better], c[better]] = lf[better]

        # end of exam
        timeout = (self.t >= self.t_max) | (self.secs >= self.secs_max)
        done = (a == RX) | timeout
        if self.shaping:                                   # the mentor's "warmer / colder": closer lenses, a bonus
            phi1 = -2.0 * self.err(self.cyl_weight).mean(1)
            rew = rew + self.shaping * (phi1 - phi0)
        info = {}
        if done.any():
            idx = np.nonzero(done)[0]
            e = self.err()[idx]
            ept = clinic.patient_error(e[:, 0], e[:, 1])
            walked = a[idx] != RX                              # out of time: the patient walks out without glasses
            pts = np.where(walked, clinic.WALKOUT, clinic.expected_points(ept))
            rew[idx] += pts / clinic.SCALE
            E0, E1 = self.true_pv(0, idx), self.true_pv(1, idx)
            F0, F1 = self.lens_pv(0, idx), self.lens_pv(1, idx)
            zero = (np.zeros(len(idx)),) * 3
            pr = clinic.outcome_probs(ept)
            info = dict(idx=idx, err=e[:, 0], err_os=e[:, 1], e_pt=ept, points=pts, secs=self.secs[idx].copy(),
                        p_happy=pr[:, 0], p_annoyed=pr[:, 1], p_recheck=pr[:, 2], p_review=pr[:, 3],
                        phase_end=self.phase[idx].astype(float), timeout=timeout[idx].astype(float),
                        steps=self.t[idx].copy(), dM=F0[0] - E0[0], dM_os=F1[0] - E1[0], age=self.P["age"][idx].copy(),
                        S=self.S[idx, 0].copy(), C=self.C[idx, 0].copy(), ax=self.ax[idx, 0].copy(),
                        S_os=self.S[idx, 1].copy(), C_os=self.C[idx, 1].copy(), ax_os=self.ax[idx, 1].copy(),
                        tS=self.P["S"][idx].copy(), tC=self.P["C"][idx].copy(), tax=self.P["axis"][idx].copy(),
                        tS_os=self.P["S_os"][idx].copy(), tC_os=self.P["C_os"][idx].copy(),
                        tax_os=self.P["axis_os"][idx].copy(),
                        err0=rx_error(zero, E0), err0_os=rx_error(zero, E1), err_half=e[:, 0],
                        pid=self.P["pid"][idx].copy() if "pid" in self.P else np.full(len(idx), -1.0))
            self._reset(idx)
        return self.obs(), rew.astype(np.float32), done, info

    # ---------------------------------------------------------------- what the AI sees
    def obs(self):
        n, r, c = self.n, self.rows, self.cur
        glob = np.concatenate([np.stack([self.P["age"] / 80, self.line, self.t / 100, self.secs / 600,
                                         (self.C[r, c] < 0).astype(float)], 1), np.eye(3)[self.phase]], 1)
        eyes, cur_pv = [], []
        for e in (0, 1):
            M, J0, J45 = self.lens_pv(e)
            a2 = np.deg2rad(2 * self.ax[:, e])
            eyes.append(np.stack([self.S[:, e] / 6, self.C[:, e] / 3, M / 6, J0, J45, np.cos(a2), np.sin(a2)], 1))
            cur_pv.append(np.stack([M, J0, J45], 1))
        cur_pv = np.stack(cur_pv, 1)                                   # (n, 2, 3)
        both = np.stack([cur_pv[:, 0, 0], cur_pv[:, 1, 0], np.zeros(n)], 1)
        slot_pv = np.concatenate([cur_pv, both[:, None]], 1)           # (n, 3, 3)
        best = np.concatenate([np.stack([self.best_va[:, e], self.best_on[:, e]], 1) for e in (0, 1)] +
                              [(self.best_l[:, e, :3] - cur_pv[:, e]) * self.best_on[:, e, None] for e in (0, 1)], 1)
        on = self.lq_on[..., None]
        lq = np.concatenate([self.lq_ans, on, (self.lq_l[..., :3] - slot_pv[:, :, None, :]) * on, self.lq_l[..., 3:] * on,
                             ((self.t[:, None, None] - self.lq_t) / 60 * self.lq_on)[..., None]], 3).reshape(n, -1)
        bal = np.concatenate([self.bal_ans, self.bal_on[:, None],
                              (self.bal_m - cur_pv[:, :, 0]) * self.bal_on[:, None],
                              ((self.t - self.bal_t) / 60 * self.bal_on)[:, None]], 1)
        valid = (self.hist_a >= 0)[..., None]
        onehot = np.zeros((n, H3, N_ACT3))
        np.put_along_axis(onehot, np.maximum(self.hist_a, 0)[..., None], 1.0, 2)
        now = np.where((self.phase == 2)[:, None], both, cur_pv[r, c])
        hist = np.concatenate([onehot, self.hist_r, self.hist_m, self.hist_l[..., :3] - now[:, None],
                               self.hist_l[..., 3:]], 2)
        hist = (hist * valid).reshape(n, -1)
        o = np.concatenate([glob] + eyes + [best, lq, bal, hist, self.mask().astype(float)], 1)
        return np.clip(o, -6, 6).astype(np.float32)


if __name__ == "__main__":
    import time
    env = ClinicEnv(1024, seed=1)
    o = env.obs()
    print("obs", o.shape, N_OBS3)
    t0 = time.time()
    res = {}
    for i in range(300):
        m = env.mask()
        a = np.array([env.rng.choice(np.nonzero(row)[0]) for row in m])
        o, rr, d, info = env.step(a)
        for k in ("e_pt", "points", "secs", "steps", "phase_end"):
            if info:
                res.setdefault(k, []).extend(info[k].tolist())
    print(f"{300 * env.n / (time.time() - t0):,.0f} steps/s (incl. python mask sampling)")
    print("random:", {k: round(float(np.mean(v)), 2) for k, v in res.items()})
    # plano for everyone
    env = ClinicEnv(4096, seed=2)
    o, rr, d, info = env.step(np.full(4096, RX))
    print("plano for everyone: mean points", info["points"].mean().round(1), "outcomes",
          {k: round(float(info[k].mean()), 3) for k in ("p_happy", "p_annoyed", "p_recheck", "p_review")},
          "eyes within 0.25:", round(float(np.mean(np.r_[info["err"], info["err_os"]] <= 0.25)), 3))
    # perfect prescription instantly
    env = ClinicEnv(4096, seed=3)
    env.S[:, 0], env.C[:, 0], env.ax[:, 0] = env.P["S"], env.P["C"], env.P["axis"]
    env.S[:, 1], env.C[:, 1], env.ax[:, 1] = env.P["S_os"], env.P["C_os"], env.P["axis_os"]
    o, rr, d, info = env.step(np.full(4096, RX))
    print("perfect: mean points", info["points"].mean().round(1))
    env = ClinicEnv(1, seed=4, record=True)
    for a in (READ, SM100, READ, LINE_LG, CMP_M, JCC_PW, READ_LINE):
        env.step(np.array([a]))
        L = env.last
        print(ACTS[a], "code", L["code"][0], "probs", np.round(L.get("probs", np.zeros((1, 0)))[0], 2),
              "right", L.get("right", [None])[0], L.get("read_dist", ""))
