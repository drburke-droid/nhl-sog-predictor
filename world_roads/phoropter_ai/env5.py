"""The clinic exam, version 5: no mentor, more freedom, more talkative patients.

Like env3.py (both eyes start at plano, the axis dial wherever the last patient left it, clinic points), plus:
* The patient says how sure they are: "One, much clearer", "One, a little", "They look the same".
* Cylinder checks at fixed orientations (the "power search" method): a cross-cylinder at 180 vs 90, or 45 vs
  135. With no cylinder in yet, the axis dial is set to the orientation the patient preferred.
* "Spin the axis: say when it's clearest": with some cylinder in, the patient watches the dial turn and stops
  it where the letters look sharpest (with their usual noise).
The cross-cylinder refinements are still there: "cylinder axis: which way?" and "cylinder power: more or less?".
* "It's still blurry": before they leave, the patient reads the chart with each eye through the new glasses. If
  either eye sees clearly worse than it could (about two lines), they say so ("my left eye is still blurry") and
  the exam goes on: Prescribe doesn't end it. (env.complain = False turns this off.)
* The occluder cycles: right eye, left eye, both eyes, then back to the right eye, so either eye can be redone.

The patient's smile (optional, env.smile = weight): when the letters get clearer for either eye (the average
of the two eyes' acuity, each through its own lens, focusing as it naturally would), the patient smiles and
the AI earns smile x (lines of acuity gained) / 10; a frown when it gets blurrier. It is based only on what
the patient sees, never on the true prescription.
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
        "1 or 2? (+0.25)", "1 or 2? (-0.25)", "Cylinder axis: which way? (1 or 2)",
        "Cylinder power: more or less? (1 or 2)", "Cylinder check: 180 or 90? (1 or 2)",
        "Cylinder check: 45 or 135? (1 or 2)", "Spin the axis: say when clearest",
        "Next eye", "+0.25 both eyes", "-0.25 both eyes", "Balance: top or bottom?",
        "+0.25 right eye", "+0.25 left eye", "Prescribe"]
N_ACT3 = len(ACTS)
(SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15, AXP2, AXM2, READ, READ_LINE, LINE_SM,
 LINE_LG, CMP_P, CMP_M, JCC_AX, JCC_PW, CHK_180, CHK_45, SPIN, NEXT, OU_P, OU_M, BALANCE, R_P, L_P, RX) = range(N_ACT3)

# seconds of chair time per button (a lens click is quick; a question takes a flip, "1... or 2?" and an answer)
SECS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # lens clicks
                 8.0, 4.0, 1.0, 1.0,                                        # smallest line; this line; line size
                 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,                              # "1 or 2?" questions
                 6.0,                                                       # spin the axis
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
MASK[2, [READ, READ_LINE, LINE_SM, LINE_LG, CMP_P, CMP_M, NEXT, OU_P, OU_M, BALANCE, R_P, L_P, RX]] = True

QS = (READ, READ_LINE, CMP_P, CMP_M, JCC_AX, JCC_PW, CHK_180, CHK_45, SPIN)   # answers remembered per stage
COMPARE = (CMP_P, CMP_M, JCC_AX, JCC_PW, CHK_180, CHK_45, BALANCE)
CHK_ORIENT = {CHK_180: (180, 90), CHK_45: (45, 135)}
MUCH = 0.06                                                   # beyond the "same" band by this much: "much clearer"
T_MAX3 = 400                                                  # button presses (a safety cap: the 15-minute limit is the real one; the AI trained with 200 and never came near it)
SECS_MAX = 900.0                                              # 15 minutes: the exam is cut short
BLURRY = 0.2                                                  # "still blurry": two lines worse than the eye could see
BLUR_SD = 0.05                                                # (patients differ in what they notice)
SECS_COMPLAIN = 10.0                                          # reading with each eye, and the complaint
H3 = 16
F3 = N_ACT3 + 6 + 3 + 5
N_OBS3 = 8 + 14 + 10 + 3 * len(QS) * 11 + 8 + 3 + H3 * F3 + N_ACT3

# what the patient can say
ANSWERS = {
    "compare": ("1 is clearer", "2 is clearer", "They look the same"),
    "strength": ("", "a little", "much"),
    "balance": ("Top is clearer", "Bottom is clearer", "They look equal"),
    "line": ("I can read all of them", "I can read most of them", "I can read a few", "I can't read any"),
}


from env3 import plano_pair, PlanoPairTestSet


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
        self.hist_r, self.hist_m, self.hist_l = z(H3, 6), z(H3, 3), z(H3, 5)
        nq = len(QS)
        self.lq_ans, self.lq_on, self.lq_l, self.lq_t = z(3, nq, 4), z(3, nq), z(3, nq, 5), z(3, nq)
        self.bal_ans, self.bal_on, self.bal_m, self.bal_t = z(4), z(), z(2), z()
        self.comp_on, self.comp_t = z(2), z()
        self.complain = True
        self.best_va, self.best_on, self.best_l = z(2), z(2), z(2, 5)
        self.start_pv = z(2, 3)
        self.shaping, self.gamma, self.t_max, self.cyl_weight = 0.0, 0.995, T_MAX3, 2.0
        self.smile = 0.0
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
                    self.bal_on, self.bal_m, self.bal_t, self.best_va, self.best_on, self.best_l, self.start_pv,
                    self.comp_on, self.comp_t):
            arr[idx] = 0

    def clarity(self):
        """Minus the average acuity (logMAR) of the two eyes, each through its own lens, as it would naturally
        focus: what the patient experiences. Ten lines better = +1."""
        return -0.5 * (self.mono_T(0, self.lens_pv(0)) + self.mono_T(1, self.lens_pv(1)))

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
        if act in (JCC_AX, JCC_PW, CHK_180, CHK_45):
            if act in CHK_ORIENT:                              # fixed orientations: the power-search method
                oa, ob = CHK_ORIENT[act]
                ja, jb = jcc_pv(np.full(n, float(oa))), jcc_pv(np.full(n, float(ob)))
            else:
                oa, ob = (45, -45) if act == JCC_AX else (0, 90)
                axc = self.ax[r, c]
                ja, jb = jcc_pv(axc + oa), jcc_pv(axc + ob)
            A_ = (Lc[0], Lc[1] + ja[1], Lc[2] + ja[2])
            B_ = (Lc[0], Lc[1] + jb[1], Lc[2] + jb[2])
            ea, eb = rx_error(A_, self._tpv(c)), rx_error(B_, self._tpv(c))
            return self._mono(c, A_), self._mono(c, B_), Lc[0], Lc[0], "jcc", zero, b.copy(), ea, eb
        # BALANCE: prism split, top = right eye, bottom = left eye; the dominant eye looks a little clearer
        return t0, t1, zero, zero, "sph", 0.02 * self.P["dom"], np.zeros(n, bool), t0, t1

    def judge5(self, qa, qb, Ma, Mb, kind, lean=0.0):
        """'1 (a) or 2 (b)?': answer 0 = a, 1 = b, 2 = same; and how sure: 0 (same), 1 (a little), 2 (much)."""
        g = gain(self.line - 0.5 * (qa + qb), kind)
        d = g * (qa - qb) + self.P["bias"] * (Ma - Mb) - lean + self.rng.normal(0, 1, self.n) * self.P["sigma"]
        tau = self.P["tau"]
        ans = np.where(d > tau, 1, np.where(d < -tau, 0, 2))
        strength = np.where(np.abs(d) > tau + MUCH, 2, np.where(ans < 2, 1, 0))
        lapse = self.rng.uniform(size=self.n) < self.P["lapse"]
        rnd = self.rng.integers(0, 3, self.n)
        return np.where(lapse, rnd, ans), np.where(lapse, np.where(rnd < 2, 1, 0), strength)

    def spin(self, rows, c):
        """With cylinder in, the patient watches the axis dial turn and stops where the letters are sharpest:
        the clearest of 36 orientations 5 degrees apart, judged with their usual noise."""
        best_ax = self.ax[rows, c[rows]].copy()
        best_T = np.full(len(rows), np.inf)
        S, C = self.S[rows, c[rows]], self.C[rows, c[rows]]
        for a in range(5, 181, 5):
            L = to_pv(S, C, np.full(len(rows), float(a)))
            T0 = self._mono_rows(0, L, rows)
            T1 = self._mono_rows(1, L, rows)
            T = np.where(c[rows] == 0, T0, T1) + self.rng.normal(0, 1, len(rows)) * self.P["sigma"][rows]
            better = T < best_T
            best_T = np.where(better, T, best_T)
            best_ax = np.where(better, float(a), best_ax)
        return best_ax

    def _mono_rows(self, e, L, rows):
        """mono_T for a subset of rows (L given for those rows)."""
        k = "" if e == 0 else "_os"
        E = to_pv(self.P["S" + k][rows], self.P["C" + k][rows], self.P["axis" + k][rows])
        Mr, J0r, J45r = E[0] - L[0], E[1] - L[1], E[2] - L[2]
        prox = self.P["prox"][rows]
        fl = prox * np.clip(1 + Mr / 0.5, 0, 1)
        A = np.clip(np.maximum(Mr, fl), 0, self.P["amp"][rows])
        B = np.sqrt((Mr - A) ** 2 + J0r ** 2 + J45r ** 2)
        mar0 = self.P["mar0"][rows] if e == 0 else self.P["mar0_os"][rows]
        from optics import logmar
        return logmar(B, self.P["pupil"][rows], mar0)

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
        if self.smile:
            sm0 = self.clarity()

        # lens and stage changes
        c = self.cur
        self.S[r, c] = np.clip(self.S[r, c] + SPH[a], -20, 20)
        self.C[r, c] = np.clip(self.C[r, c] + CYL[a], -6, 0)
        self.ax[r, c] = ((self.ax[r, c] + AXD[a] - 1) % 180) + 1
        self.S = np.clip(self.S + OUD[a][:, None], -20, 20)
        self.S[:, 0] = np.clip(self.S[:, 0] + RD[a], -20, 20)
        self.S[:, 1] = np.clip(self.S[:, 1] + LD[a], -20, 20)
        self.line = np.round(np.clip(self.line + LND[a], -0.3, 1.0), 1)
        self.phase = np.where(a == NEXT, (self.phase + 1) % 3, self.phase)
        self.cur = np.minimum(self.phase, 1)
        self.bin = (self.phase == 2).astype(float)
        V = self._views()
        c, b, L0, L1, Lc, Tm, t0, t1, tb = V
        Tview = np.where(b, tb, Tm)

        resp = np.zeros((n, 6))
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
        strength = np.zeros(n, np.int64)
        for act in COMPARE:
            m = a == act
            if not m.any():
                continue
            qa, qb, Ma, Mb, kind, lean, same, _, _ = self._compare_terms(act, V)
            res, st = self.judge5(qa, qb, Ma, Mb, kind, lean)
            ans = np.where(m, np.where(same, 2, res), ans)
            strength = np.where(m, np.where(same, 0, st), strength)
        cm = np.isin(a, COMPARE)
        resp[cm, 2 + ans[cm]] = 1.0
        resp[cm, 5] = strength[cm] / 2
        code = np.where(cm, ans, code)
        # power search: with no cylinder in yet, the dial goes to the orientation the patient preferred
        for act, (oa, ob) in CHK_ORIENT.items():
            m = (a == act) & (self.C[r, c] >= 0) & (ans < 2)
            if m.any():
                self.ax[r[m], c[m]] = np.where(ans[m] == 0, float(oa), float(ob))
        # spin the axis
        spun = np.zeros(n, bool)
        spin_ax = np.zeros(n)
        ms = np.nonzero((a == SPIN) & (self.C[r, c] < 0))[0]
        if len(ms):
            new = self.spin(ms, c)
            self.ax[ms, c[ms]] = new
            spun[ms] = True
            spin_ax[ms] = new
        resp[a == SPIN, 2] = spun[a == SPIN].astype(float)
        code = np.where(a == SPIN, np.where(spun, 5, 6), code)
        if cm.any() or spun.any():
            V = self._views()
            c, b, L0, L1, Lc, Tm, t0, t1, tb = V
        # "It's still blurry": prescribing, the patient reads with each eye; two lines worse than it could see, and
        # they say so and stay in the chair
        blurry = np.zeros((n, 2), bool)
        rx = a == RX
        if self.complain and rx.any():
            lost = np.stack([self.mono_T(e, self.lens_pv(e)) - self.mono_T(e, self.true_pv(e)) for e in (0, 1)], 1)
            blurry = rx[:, None] & (lost + self.rng.normal(0, BLUR_SD, (n, 2)) > BLURRY)
        refused = blurry.any(1)
        if refused.any():
            self.secs[refused] += SECS_COMPLAIN
            rew = rew - refused * SECS_COMPLAIN * clinic.POINTS_PER_MIN / 60 / clinic.SCALE
            resp[refused, 2:4] = blurry[refused]
            code = np.where(refused, np.where(blurry[:, 0] & blurry[:, 1], 9, np.where(blurry[:, 0], 7, 8)), code)
            self.comp_on[refused] = blurry[refused]
            self.comp_t[refused] = self.t[refused]
        if self.record:
            self.last = dict(a=a.copy(), code=code, strength=strength, spin_ax=spin_ax, va=va, k=kk, nl=nn,
                             blurry=blurry,
                             line=line_read, screen=self.line.copy(),
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
                if act in (READ, READ_LINE):
                    v = np.stack([va, kk / nn, np.zeros(n), np.zeros(n)], -1)
                else:
                    v = resp[:, 2:6]
                self.lq_ans[r[m], slot[m], k] = v[m]
                self.lq_on[r[m], slot[m], k] = 1
                self.lq_l[r[m], slot[m], k] = lf[m]
                self.lq_t[r[m], slot[m], k] = self.t[m]
        mb = a == BALANCE
        if mb.any():
            self.bal_ans[mb] = resp[mb, 2:6]
            self.bal_on[mb] = 1
            self.bal_m[mb] = np.stack([L0[0], L1[0]], -1)[mb]
            self.bal_t[mb] = self.t[mb]
        better = rd & ~b & ((self.best_on[r, c] == 0) | (va <= self.best_va[r, c]))
        self.best_va[r[better], c[better]] = va[better]
        self.best_on[r[better], c[better]] = 1
        self.best_l[r[better], c[better]] = lf[better]

        # end of exam
        timeout = (self.t >= self.t_max) | (self.secs >= self.secs_max)
        accepted = rx & ~refused
        done = accepted | timeout
        if self.shaping:                                   # the mentor's "warmer / colder": closer lenses, a bonus
            phi1 = -2.0 * self.err(self.cyl_weight).mean(1)
            rew = rew + self.shaping * (phi1 - phi0)
        if self.smile:                                     # the patient smiles when the letters get clearer
            rew = rew + self.smile * (self.clarity() - sm0)
        info = {}
        if done.any():
            idx = np.nonzero(done)[0]
            e = self.err()[idx]
            ept = clinic.patient_error(e[:, 0], e[:, 1])
            walked = ~accepted[idx]                            # out of time: the patient walks out without glasses
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
        comp = np.concatenate([self.comp_on, ((self.t - self.comp_t) / 60 * self.comp_on.max(1))[:, None]], 1)
        valid = (self.hist_a >= 0)[..., None]
        onehot = np.zeros((n, H3, N_ACT3))
        np.put_along_axis(onehot, np.maximum(self.hist_a, 0)[..., None], 1.0, 2)
        now = np.where((self.phase == 2)[:, None], both, cur_pv[r, c])
        hist = np.concatenate([onehot, self.hist_r, self.hist_m, self.hist_l[..., :3] - now[:, None],
                               self.hist_l[..., 3:]], 2)
        hist = (hist * valid).reshape(n, -1)
        o = np.concatenate([glob] + eyes + [best, lq, bal, comp, hist, self.mask().astype(float)], 1)
        return np.clip(o, -6, 6).astype(np.float32)


if __name__ == "__main__" and False:
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


if __name__ == "__main__":
    import time
    env = ClinicEnv(1024, seed=1)
    o = env.obs()
    print("obs", o.shape, N_OBS3, "actions", N_ACT3)
    t0 = time.time()
    res = {}
    for i in range(200):
        m = env.mask()
        a = np.array([env.rng.choice(np.nonzero(row)[0]) for row in m])
        o, rr, d, info = env.step(a)
        if info:
            for k in ("points", "secs", "steps"):
                res.setdefault(k, []).extend(info[k].tolist())
    print(f"{200 * env.n / (time.time() - t0):,.0f} steps/s; random:", {k: round(float(np.mean(v)), 2) for k, v in res.items()})
    from record import fixed_pair
    from record3 import SHOWCASE3
    env = ClinicEnv(1, seed=4, sampler=fixed_pair(**SHOWCASE3), record=True)
    for a in (CHK_180, CYL_MORE, CYL_MORE, SPIN, JCC_AX, JCC_PW, CHK_45):
        env.step(np.array([a]))
        L = env.last
        print(ACTS[a], "code", L["code"][0], "strength", L["strength"][0], "axis now", env.ax[0, 0], "cyl", env.C[0, 0],
              "probs", np.round(L.get("probs", np.zeros((1, 0)))[0], 2))
