"""The two-eye exam room (attempt 4): refract each eye, then balance them binocularly.

What changes from env.py:
* Two eyes. Lens buttons act on the eye being tested. "Test right eye" / "Test left eye" occlude the other
  eye. "Both eyes open" removes the occluder. "+0.25 both eyes" / "-0.25 both eyes" fog or unfog together.
  "Balance" puts vertical prism in so the patient sees two copies of the line (top = right eye) and asks
  which is clearer.
* Accommodation is yoked: both eyes accommodate by the same amount. With one eye occluded, the open eye
  sets it (and can hide its own over-minus). With both open, it is the smaller of the two eyes' needs:
  accommodating more would blur the other eye. So an eye over-minused on its own stays over-minused, and
  a fogged balance shows it as the clearer eye.
* Proximal (instrument) accommodation: with one eye occluded behind the phoropter, a person's accommodation
  will not relax below a floor of 0-0.50 D (scaled down with age), so the clearest lens is a little too
  minus. Fog relaxes the floor (gone at 0.50 D of fog); with both eyes open only 30% of it remains. This is why binocular balancing and a binocular
  plus check exist.
* The projector shows one line (chart.py). Every judgement is made on that line, and how well the patient
  can judge depends on the line's size relative to their threshold T through the lenses being compared
  (delta = line - T, in logMAR; positive = letters bigger than they can just read):
    sphere and balance ("which is clearer?"): best at or just below threshold, fading on big letters,
        gain = exp(-((delta + 0.05)/w)^2), w = 0.35 above the peak, 0.20 below;
    cross-cylinder: best about 1.5 lines above threshold,
        gain = exp(-((delta - 0.15)/w)^2), w = 0.35 above, 0.25 below.
  The gain scales the real clarity difference before the patient's noise and "same" band are applied.
  At 20/200 with 20/63 vision (delta ~ +0.5) a 0.25 D sphere step is nearly invisible; at 20/40 it is clear.
* Ocular dominance: in the prism-split balance, the dominant eye's image looks 0.02 logMAR clearer than it
  is (70% of patients are right-eye dominant).
* Score: the mean over the two eyes of final_reward(rx_error).
"""
import numpy as np
from optics import to_pv, rx_error, blur_strength, logmar, jcc_pv
from patients import sample_pair
from env import final_reward, COST_LENS, COST_Q
import chart

ACTS = ["Sphere +0.25", "Sphere -0.25", "Sphere +1.00", "Sphere -1.00", "Cylinder -0.25", "Cylinder +0.25",
        "Axis +5°", "Axis -5°", "Axis +15°", "Axis -15°", "Read the chart",
        "1 or 2? (+0.25)", "1 or 2? (-0.25)", "Cross-cyl: axis", "Cross-cyl: power", "Done",
        "Line smaller", "Line larger", "Read this line",
        "Test right eye", "Test left eye", "Both eyes open", "+0.25 both eyes", "-0.25 both eyes", "Balance: top or bottom?"]
N_ACT2 = len(ACTS)
(READ, CMP_P, CMP_M, JCC_AX, JCC_PW, DONE, LINE_SM, LINE_LG, READ_LINE, TEST_OD, TEST_OS, BOTH, OU_P, OU_M,
 BALANCE) = 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
SPH = np.zeros(N_ACT2); SPH[[0, 1, 2, 3]] = [0.25, -0.25, 1.0, -1.0]
CYL = np.zeros(N_ACT2); CYL[[4, 5]] = [-0.25, 0.25]
AXD = np.zeros(N_ACT2); AXD[[6, 7, 8, 9]] = [5, -5, 15, -15]
LND = np.zeros(N_ACT2); LND[[LINE_SM, LINE_LG]] = [-0.1, 0.1]
OUD = np.zeros(N_ACT2); OUD[[OU_P, OU_M]] = [0.25, -0.25]
QUESTIONS2 = (READ, CMP_P, CMP_M, JCC_AX, JCC_PW, READ_LINE)
CHEAP = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, LINE_SM, LINE_LG, TEST_OD, TEST_OS, BOTH, OU_P, OU_M)
T_MAX2 = 120
H2 = 16
F2 = N_ACT2 + 5 + 2 + 3 + 2
N_OBS2 = 5 + 20 + 10 + 2 * len(QUESTIONS2) * 10 + 8 + H2 * F2


def gain(delta, kind):
    """How well the patient can judge a clarity difference on a line delta logMAR above threshold."""
    if kind == "jcc":
        mu, wa, wb = 0.15, 0.35, 0.25
    else:
        mu, wa, wb = -0.05, 0.35, 0.20
    w = np.where(delta >= mu, wa, wb)
    return np.exp(-((delta - mu) / w) ** 2)


class BinocularEnv:
    def __init__(self, n, seed=0, sampler=sample_pair, record=False):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        self.record = record
        self.n_act, self.n_obs, self.letters, self.flag = N_ACT2, N_OBS2, True, False
        z = lambda *s: np.zeros((n,) + s)
        self.P = {}
        self.S, self.C, self.ax = z(2), z(2), z(2)
        self.cur = np.zeros(n, np.int64)                  # 0 = right eye, 1 = left eye
        self.bin = np.zeros(n)                            # 1 = both eyes open
        self.line = np.full(n, 0.0)
        self.t = z()
        self.hist_a = np.full((n, H2), -1, np.int64)
        self.hist_r, self.hist_m, self.hist_l = z(H2, 5), z(H2, 2), z(H2, 5)
        nq = len(QUESTIONS2)
        self.lq_ans, self.lq_on, self.lq_l, self.lq_t = z(2, nq, 3), z(2, nq), z(2, nq, 5), z(2, nq)
        self.bal_ans, self.bal_on, self.bal_m, self.bal_t = z(3), z(), z(2), z()
        self.best_va, self.best_on, self.best_l = z(2), z(2), z(2, 5)
        self.start_pv = z(2, 3)
        self.shaping, self.gamma, self.t_max, self.cyl_weight = 0.0, 0.995, T_MAX2, 2.0
        self.rows = np.arange(n)
        self._reset(np.arange(n))

    # ---------------------------------------------------------------- state
    def lens_pv(self, e, idx=slice(None)):
        return to_pv(self.S[idx, e], self.C[idx, e], self.ax[idx, e])

    def true_pv(self, e, idx=slice(None)):
        k = "" if e == 0 else "_os"
        return to_pv(self.P["S" + k][idx], self.P["C" + k][idx], self.P["axis" + k][idx])

    def _reset(self, idx):
        if len(idx) == 0:
            return
        p = self.sampler(self.rng, len(idx))
        for k, v in p.items():
            if k not in self.P:
                self.P[k] = np.zeros(self.n)
            self.P[k][idx] = v
        self.S[idx, 0], self.C[idx, 0], self.ax[idx, 0] = p["S0"], p["C0"], p["ax0"]
        self.S[idx, 1], self.C[idx, 1], self.ax[idx, 1] = p["S0_os"], p["C0_os"], p["ax0_os"]
        self.cur[idx], self.bin[idx], self.line[idx], self.t[idx] = 0, 0, 0.0, 0     # projector starts on 20/20
        self.hist_a[idx] = -1
        for arr in (self.hist_r, self.hist_m, self.hist_l, self.lq_ans, self.lq_on, self.lq_l, self.lq_t, self.bal_ans,
                    self.bal_on, self.bal_m, self.bal_t, self.best_va, self.best_on, self.best_l):
            arr[idx] = 0
        for e in (0, 1):
            self.start_pv[idx, e] = np.stack(self.lens_pv(e, idx), -1)

    def err(self, w=2.0):
        """Per-eye error of what is in the phoropter now (n, 2)."""
        return np.stack([rx_error(self.lens_pv(e), self.true_pv(e), w) for e in (0, 1)], 1)

    # ---------------------------------------------------------------- the patient
    def _resid(self, e, L):
        E = self.true_pv(e)
        return E[0] - L[0], E[1] - L[1], E[2] - L[2]

    def _T(self, e, L, A):
        """Threshold (logMAR) of eye e through lens L with accommodation A."""
        Mr, J0r, J45r = self._resid(e, L)
        B = np.sqrt((Mr - A) ** 2 + J0r ** 2 + J45r ** 2)
        mar0 = self.P["mar0"] if e == 0 else self.P["mar0_os"]
        return logmar(B, self.P["pupil"], mar0)

    def mono_T(self, e, L):
        """Eye e alone (fellow occluded): it accommodates to clear its own artificial hyperopia, plus the person's
        proximal accommodation behind the phoropter, which fogging relaxes."""
        R = self._resid(e, L)[0]
        return self._T(e, L, self._acc(R, self.P["prox"]))

    def _acc(self, R, floor):
        """Accommodation: what clears the image (R > 0), but never relaxing below a floor (proximal accommodation),
        except that fog (R < 0) relaxes the floor away: gone at 0.50 D of fog."""
        fl = floor * np.clip(1 + R / 0.5, 0, 1)
        return np.clip(np.maximum(R, fl), 0, self.P["amp"])

    def bino(self, L0, L1):
        """Both eyes open, yoked accommodation: the smaller of the two eyes' needs, with only 30% of the proximal
        accommodation left. Returns (T_right, T_left, T_both)."""
        px = 0.3 * self.P["prox"]
        A = np.minimum(self._acc(self._resid(0, L0)[0], px), self._acc(self._resid(1, L1)[0], px))
        t0, t1 = self._T(0, L0, A), self._T(1, L1, A)
        return t0, t1, np.minimum(t0, t1) - 0.02          # a little binocular summation

    def judge(self, qa, qb, Ma, Mb, kind, lean=0.0):
        """'1 (a) or 2 (b)?' on the line on screen: 0 = a, 1 = b, 2 = same. lean > 0 favours a."""
        g = gain(self.line - 0.5 * (qa + qb), kind)
        d = g * (qa - qb) + self.P["bias"] * (Ma - Mb) - lean + self.rng.normal(0, 1, self.n) * self.P["sigma"]
        ans = np.where(d > self.P["tau"], 1, np.where(d < -self.P["tau"], 0, 2))
        lapse = self.rng.uniform(size=self.n) < self.P["lapse"]
        return np.where(lapse, self.rng.integers(0, 3, self.n), ans)

    # ---------------------------------------------------------------- step
    def step(self, a):
        a = np.asarray(a, np.int64)
        n, r = self.n, self.rows
        self.t += 1
        is_q = np.isin(a, QUESTIONS2 + (BALANCE,))
        rew = -np.where(np.isin(a, CHEAP), COST_LENS, np.where(is_q, COST_Q, 0.0))
        if self.shaping:
            phi0 = -2.0 * self.err(self.cyl_weight).mean(1)

        # lens and mode changes
        c = self.cur
        self.S[r, c] = np.clip(self.S[r, c] + SPH[a], -20, 20)
        self.C[r, c] = np.clip(self.C[r, c] + CYL[a], -6, 0)
        self.ax[r, c] = ((self.ax[r, c] + AXD[a] - 1) % 180) + 1
        self.S = np.clip(self.S + OUD[a][:, None], -20, 20)
        self.line = np.round(np.clip(self.line + LND[a], -0.3, 1.0), 1)
        self.cur = np.where(a == TEST_OD, 0, np.where(a == TEST_OS, 1, self.cur))
        self.bin = np.where((a == TEST_OD) | (a == TEST_OS), 0, np.where((a == BOTH) | (a == BALANCE), 1, self.bin))
        c = self.cur
        b = self.bin > 0

        # what the patient sees now
        L0, L1 = self.lens_pv(0), self.lens_pv(1)
        Lc = tuple(np.where(c == 0, x0, x1) for x0, x1 in zip(L0, L1))
        Tm = np.where(c == 0, self.mono_T(0, Lc), self.mono_T(1, Lc))
        t0, t1, tb = self.bino(L0, L1)
        Tview = np.where(b, tb, Tm)

        resp = np.zeros((n, 5))
        code = np.full(n, -1)
        kk, nn, line_read = np.zeros(n), np.ones(n), self.line.copy()
        va = np.zeros(n)
        rd, rl = a == READ, a == READ_LINE
        if rd.any():
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

        # 1-or-2 questions. Sphere: on the tested eye, or on both eyes when both are open.
        ans = np.full(n, 2)
        for act, ds in ((CMP_P, 0.25), (CMP_M, -0.25)):
            m = a == act
            if not m.any():
                continue
            Lb = (Lc[0] + ds, Lc[1], Lc[2])
            qa_m, qb_m = Tm, np.where(c == 0, self.mono_T(0, Lb), self.mono_T(1, Lb))
            _, _, qb_b = self.bino((L0[0] + ds, L0[1], L0[2]), (L1[0] + ds, L1[1], L1[2]))
            qa, qb = np.where(b, tb, qa_m), np.where(b, qb_b, qb_m)
            res = self.judge(qa, qb, Lc[0], Lc[0] + ds, "sph")
            ans = np.where(m, res, ans)
        for act, oa, ob in ((JCC_AX, 45, -45), (JCC_PW, 0, 90)):
            m = a == act
            if not m.any():
                continue
            axc = self.ax[r, c]
            ja, jb = jcc_pv(axc + oa), jcc_pv(axc + ob)
            A_ = (Lc[0], Lc[1] + ja[1], Lc[2] + ja[2])
            B_ = (Lc[0], Lc[1] + jb[1], Lc[2] + jb[2])
            qa = np.where(c == 0, self.mono_T(0, A_), self.mono_T(1, A_))
            qb = np.where(c == 0, self.mono_T(0, B_), self.mono_T(1, B_))
            res = self.judge(qa, qb, Lc[0], Lc[0], "jcc")
            res = np.where(b, 2, res)                       # with both eyes open a cross-cylinder tells nothing
            ans = np.where(m, res, ans)
        mb = a == BALANCE
        if mb.any():                                        # prism-split: top = right eye, bottom = left eye
            res = self.judge(t0, t1, 0.0, 0.0, "sph", lean=0.02 * self.P["dom"])   # the dominant eye looks a bit clearer
            ans = np.where(mb, res, ans)
        cm = np.isin(a, (CMP_P, CMP_M, JCC_AX, JCC_PW, BALANCE))
        resp[cm, 2 + ans[cm]] = 1.0
        code = np.where(cm, ans, code)
        if self.record:
            self.last = dict(a=a.copy(), code=code, va=va, k=kk, nl=nn, line=line_read, screen=self.line.copy(),
                             cur=self.cur.copy(), bin=self.bin.copy(), T0=t0, T1=t1)

        # memory
        lf = np.stack([Lc[0], Lc[1], Lc[2], np.cos(np.deg2rad(2 * self.ax[r, c])), np.sin(np.deg2rad(2 * self.ax[r, c]))], -1)
        self.hist_a = np.roll(self.hist_a, 1, 1)
        self.hist_r = np.roll(self.hist_r, 1, 1)
        self.hist_m = np.roll(self.hist_m, 1, 1)
        self.hist_l = np.roll(self.hist_l, 1, 1)
        self.hist_a[:, 0], self.hist_r[:, 0] = a, resp
        self.hist_m[:, 0] = np.stack([c, self.bin], -1)
        self.hist_l[:, 0] = lf
        for k, act in enumerate(QUESTIONS2):
            m = a == act
            if m.any():
                if act in (READ, READ_LINE):
                    v = np.stack([va, kk / nn, b.astype(float)], -1)
                else:
                    v = resp[:, 2:]
                self.lq_ans[r[m], c[m], k] = v[m]
                self.lq_on[r[m], c[m], k] = 1
                self.lq_l[r[m], c[m], k] = lf[m]
                self.lq_t[r[m], c[m], k] = self.t[m]
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
        done = (a == DONE) | (self.t >= self.t_max)
        if self.shaping:
            phi1 = np.where(done, 0.0, -2.0 * self.err(self.cyl_weight).mean(1))
            rew = rew + self.shaping * (self.gamma * phi1 - phi0)
        info = {}
        if done.any():
            idx = np.nonzero(done)[0]
            e = self.err()[idx]
            rew[idx] += 0.5 * (final_reward(e[:, 0]) + final_reward(e[:, 1])) - 0.2 * (a[idx] != DONE)
            E0, E1 = self.true_pv(0, idx), self.true_pv(1, idx)
            F0, F1 = self.lens_pv(0, idx), self.lens_pv(1, idx)
            st0 = tuple(self.start_pv[idx, 0].T)
            st1 = tuple(self.start_pv[idx, 1].T)
            info = dict(idx=idx, err=e[:, 0], err_os=e[:, 1], steps=self.t[idx].copy(), dM=F0[0] - E0[0],
                        dM_os=F1[0] - E1[0], age=self.P["age"][idx].copy(),
                        S=self.S[idx, 0].copy(), C=self.C[idx, 0].copy(), ax=self.ax[idx, 0].copy(),
                        S_os=self.S[idx, 1].copy(), C_os=self.C[idx, 1].copy(), ax_os=self.ax[idx, 1].copy(),
                        tS=self.P["S"][idx].copy(), tC=self.P["C"][idx].copy(), tax=self.P["axis"][idx].copy(),
                        tS_os=self.P["S_os"][idx].copy(), tC_os=self.P["C_os"][idx].copy(),
                        tax_os=self.P["axis_os"][idx].copy(),
                        err0=rx_error(st0, E0), err0_os=rx_error(st1, E1), err_half=e[:, 0],
                        pid=self.P["pid"][idx].copy() if "pid" in self.P else np.full(len(idx), -1.0))
            self._reset(idx)
        return self.obs(), rew.astype(np.float32), done, info

    # ---------------------------------------------------------------- what the AI sees
    def obs(self):
        n, r, c = self.n, self.rows, self.cur
        glob = np.stack([self.P["age"] / 80, (c == 0).astype(float), self.bin, self.line, self.t / 60], 1)
        eyes, cur_pv = [], []
        for e in (0, 1):
            M, J0, J45 = self.lens_pv(e)
            a2 = np.deg2rad(2 * self.ax[:, e])
            eyes.append(np.stack([self.S[:, e] / 6, self.C[:, e] / 3, M / 6, J0, J45, np.cos(a2), np.sin(a2),
                                  self.start_pv[:, e, 0] - M, self.start_pv[:, e, 1] - J0, self.start_pv[:, e, 2] - J45], 1))
            cur_pv.append(np.stack([M, J0, J45], 1))
        cur_pv = np.stack(cur_pv, 1)                        # (n, 2, 3)
        best = np.concatenate([np.stack([self.best_va[:, e], self.best_on[:, e]], 1) for e in (0, 1)] +
                              [(self.best_l[:, e, :3] - cur_pv[:, e]) * self.best_on[:, e, None] for e in (0, 1)], 1)
        on = self.lq_on[..., None]
        lq = np.concatenate([self.lq_ans, on, (self.lq_l[..., :3] - cur_pv[:, :, None, :]) * on, self.lq_l[..., 3:] * on,
                             ((self.t[:, None, None] - self.lq_t) / 60 * self.lq_on)[..., None]], 3).reshape(n, -1)
        bal = np.concatenate([self.bal_ans, self.bal_on[:, None],
                              (self.bal_m - cur_pv[:, :, 0]) * self.bal_on[:, None],
                              ((self.t - self.bal_t) / 60 * self.bal_on)[:, None], np.zeros((n, 1))], 1)
        valid = (self.hist_a >= 0)[..., None]
        onehot = np.zeros((n, H2, N_ACT2))
        np.put_along_axis(onehot, np.maximum(self.hist_a, 0)[..., None], 1.0, 2)
        cur_lens = cur_pv[r, c]                             # deltas relative to the eye being tested now
        hist = np.concatenate([onehot, self.hist_r, self.hist_m, self.hist_l[..., :3] - cur_lens[:, None],
                               self.hist_l[..., 3:]], 2)
        hist = (hist * valid).reshape(n, -1)
        o = np.concatenate([glob] + eyes + [best, lq, bal, hist], 1)
        return np.clip(o, -6, 6).astype(np.float32)


if __name__ == "__main__":
    import time
    env = BinocularEnv(1024, seed=1)
    o = env.obs()
    print("obs", o.shape, N_OBS2)
    t0 = time.time()
    errs, steps = [], []
    for i in range(300):
        o, rr, d, info = env.step(env.rng.integers(0, N_ACT2, env.n))
        if info:
            errs += list(info["err"]); steps += list(info["steps"])
    print(f"{300 * env.n / (time.time() - t0):,.0f} steps/s; random: mean err {np.mean(errs):.2f}, moves {np.mean(steps):.1f}")
    for dl in (-0.3, -0.1, 0.0, 0.2, 0.5, 0.8):
        print(f"delta {dl:+.1f}: sphere gain {float(gain(np.array(dl), 'sph')):.2f}  jcc gain {float(gain(np.array(dl), 'jcc')):.2f}")
