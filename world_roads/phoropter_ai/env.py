"""The exam room: a vectorised subjective-refraction environment.

The AI sits at the phoropter. Each turn it either changes a lens or asks the patient something.
It is never shown the true prescription; it only hears answers. When it says "done", it is scored on how
close its final prescription is to the patient's true one (optics.rx_error).

Actions
  0-3   sphere +0.25, -0.25, +1.00, -1.00
  4-5   cylinder -0.25 (more), +0.25 (less)
  6-9   axis +5, -5, +15, -15 degrees
  10    "read the lowest line you can"           -> a Snellen line
  11    "which is clearer, 1 or 2?"  1 = now, 2 = +0.25 sphere   -> 1 / 2 / same
  12    "which is clearer, 1 or 2?"  1 = now, 2 = -0.25 sphere   -> 1 / 2 / same
  13    cross-cylinder flip, straddling the axis (axis check)      -> 1 / 2 / same
  14    cross-cylinder flip, on the axis (power check)             -> 1 / 2 / same
  15    done: prescribe what is in the phoropter

The AI is not told what the cross-cylinder answers mean; it has to learn that too.
"""
import numpy as np
from optics import to_pv, from_pv, pv_dist, rx_error, blur_strength, logmar, jcc_pv
from patients import sample_patients
import chart

N_ACT = 16                   # attempts 1-2 (whole-line chart)
N_ACT_LETTERS = 19           # attempt 3 adds the projector buttons
SPH_D = np.array([0.25, -0.25, 1.0, -1.0] + [0] * 15, np.float64)
CYL_D = np.array([0] * 4 + [-0.25, 0.25] + [0] * 13, np.float64)
AX_D = np.array([0] * 6 + [5, -5, 15, -15] + [0] * 9, np.float64)
LINE_D = np.array([0] * 16 + [-0.1, 0.1, 0], np.float64)
READ, CMP_P, CMP_M, JCC_AX, JCC_PW, DONE = 10, 11, 12, 13, 14, 15
LINE_SM, LINE_LG, READ_LINE = 16, 17, 18
QUESTIONS = (READ, CMP_P, CMP_M, JCC_AX, JCC_PW)
ACT_NAME = ["Sphere +0.25", "Sphere -0.25", "Sphere +1.00", "Sphere -1.00", "Cylinder -0.25", "Cylinder +0.25",
            "Axis +5°", "Axis -5°", "Axis +15°", "Axis -15°", "Read the chart",
            "1 or 2? (+0.25)", "1 or 2? (-0.25)", "Cross-cyl: axis", "Cross-cyl: power", "Done",
            "Line smaller", "Line larger", "Read this line"]

COST_LENS, COST_Q = 0.002, 0.005
T_MAX = 60
H = 12                       # history window the policy sees
F_ENTRY = N_ACT + 4 + 3 + 2  # action one-hot, response (va, 1, 2, same), lens delta, axis cos/sin
N_OBS = 12 + 5 + 5 * 10 + H * F_ENTRY


def final_reward(err):
    return np.maximum(-1.0, 1.0 - 2.0 * err) + 0.25 * (err < 0.13)


class RefractionEnv:
    def __init__(self, n, seed=0, sampler=sample_patients, record=False, flag=False, letters=False):
        self.n = n
        self.flag = flag            # add a "has an autorefractor reading" input (attempt 3)
        self.letters = letters      # letter-by-letter projector chart with line buttons (attempt 3)
        self.n_act = N_ACT_LETTERS if letters else N_ACT
        self.questions = QUESTIONS + ((READ_LINE,) if letters else ())
        nq = len(self.questions)
        self.R = 5 if letters else 4                 # response: (line or va, letters fraction | -, 1, 2, same)
        self.n_obs = 12 + 5 + nq * 10 + H * (self.n_act + self.R + 5) + int(letters) + int(flag)
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        self.record = record
        z = lambda *s: np.zeros((n,) + s)
        self.P = {}
        self.S, self.C, self.ax, self.t = z(), z(), z(), z()
        self.hist_a = np.full((n, H), -1, np.int64)
        self.hist_r, self.hist_l = z(H, self.R), z(H, 5)
        self.lq_ans, self.lq_on, self.lq_l, self.lq_t = z(nq, 3), z(nq), z(nq, 5), z(nq)
        self.line = np.full(n, 0.3)                  # the line on the projector screen (logMAR)
        self.best_va, self.best_on, self.best_l = z(), z(), z(5)
        self.start_pv = z(3)
        self.shaping, self.gamma = 0.0, 0.995       # optional "warmer/colder" hint weight (training only)
        self.t_max = T_MAX                          # moves before the exam times out
        self.cyl_weight = 2.0                       # score: 2 = leftover cylinder in full dioptres; 1 = half (attempt 1)
        self._reset(np.arange(n))

    # ---------------------------------------------------------------- state helpers
    def lens_pv(self, idx=slice(None)):
        return to_pv(self.S[idx], self.C[idx], self.ax[idx])

    def true_pv(self, idx=slice(None)):
        return to_pv(self.P["S"][idx], self.P["C"][idx], self.P["axis"][idx])

    def _lens_feat(self):
        M, J0, J45 = self.lens_pv()
        a = np.deg2rad(self.ax) * 2
        return np.stack([M, J0, J45, np.cos(a), np.sin(a)], -1)

    def _reset(self, idx):
        if len(idx) == 0:
            return
        p = self.sampler(self.rng, len(idx))
        for k, v in p.items():
            if k not in self.P:
                self.P[k] = np.zeros(self.n)
            self.P[k][idx] = v
        self.S[idx], self.C[idx], self.ax[idx] = p["S0"], p["C0"], p["ax0"]
        self.t[idx] = 0
        self.hist_a[idx] = -1
        self.hist_r[idx] = 0
        self.hist_l[idx] = 0
        self.lq_ans[idx] = 0
        self.lq_on[idx] = 0
        self.lq_l[idx] = 0
        self.lq_t[idx] = 0
        self.best_va[idx] = 0
        self.best_on[idx] = 0
        self.best_l[idx] = 0
        self.line[idx] = 0.3
        self.start_pv[idx] = np.stack(to_pv(p["S0"], p["C0"], p["ax0"]), -1)

    # ---------------------------------------------------------------- the patient
    def quality(self, L, idx=slice(None)):
        """Continuous logMAR the patient gets through lens L (they accommodate if it helps)."""
        EM, EJ0, EJ45 = self.true_pv(idx)
        B = blur_strength(EM - L[0], EJ0 - L[1], EJ45 - L[2], self.P["amp"][idx])
        return logmar(B, self.P["pupil"][idx], self.P["mar0"][idx])

    def read_chart(self, L):
        lm = self.quality(L) + self.rng.normal(0, 0.05, self.n)
        return np.clip(np.round(lm * 10) / 10, -0.3, 1.0)

    def compare(self, A, B):
        """Patient's answer to '1 (A) or 2 (B)?': 0 = 1, 1 = 2, 2 = same."""
        b = self.P["bias"]
        qa = self.quality(A) + b * A[0]           # more minus looks a little darker/crisper to some patients
        qb = self.quality(B) + b * B[0]
        d = qa - qb + self.rng.normal(0, 1, self.n) * self.P["sigma"]
        ans = np.where(d > self.P["tau"], 1, np.where(d < -self.P["tau"], 0, 2))
        lapse = self.rng.uniform(size=self.n) < self.P["lapse"]
        return np.where(lapse, self.rng.integers(0, 3, self.n), ans)

    def presentations(self, a):
        """The two lens combinations shown for each question (power vectors), per env."""
        M, J0, J45 = self.lens_pv()
        z = np.zeros(self.n)
        dA = [z, z, z]
        dB = [z, z, z]
        for act, fa, fb in ((CMP_P, (0, 0, 0), (0.25, 0, 0)), (CMP_M, (0, 0, 0), (-0.25, 0, 0))):
            m = a == act
            dB = [np.where(m, fb[i], dB[i]) for i in range(3)]
        for act, oa, ob in ((JCC_AX, 45, -45), (JCC_PW, 0, 90)):
            m = a == act
            ja, jb = jcc_pv(self.ax + oa), jcc_pv(self.ax + ob)
            dA = [np.where(m, ja[i], dA[i]) for i in range(3)]
            dB = [np.where(m, jb[i], dB[i]) for i in range(3)]
        A = (M + dA[0], J0 + dA[1], J45 + dA[2])
        B = (M + dB[0], J0 + dB[1], J45 + dB[2])
        return A, B

    # ---------------------------------------------------------------- step
    def step(self, a):
        a = np.asarray(a, np.int64)
        n = self.n
        self.t += 1
        is_q = np.isin(a, self.questions)
        rew = -np.where((a < 10) | (a == LINE_SM) | (a == LINE_LG), COST_LENS, np.where(is_q, COST_Q, 0.0))
        if self.shaping:
            phi0 = -2.0 * rx_error(self.lens_pv(), self.true_pv(), self.cyl_weight)

        # lens changes
        self.S = np.clip(self.S + SPH_D[a], -20, 20)
        self.C = np.clip(self.C + CYL_D[a], -6, 0)
        self.ax = ((self.ax + AX_D[a] - 1) % 180) + 1
        self.line = np.round(np.clip(self.line + LINE_D[a], -0.3, 1.0), 1)

        # questions
        resp = np.zeros((n, self.R))
        L = self.lens_pv()
        rd = a == READ
        rl = a == READ_LINE
        cm = is_q & ~rd & ~rl
        kk, nn, line_read = np.zeros(n), np.ones(n), self.line.copy()
        if not self.letters:
            va = self.read_chart(L)                        # (random draws in the original order: chart, then 1-or-2)
        A, B = self.presentations(a)
        ans = self.compare(A, B)
        if not self.letters:
            resp[rd, 0] = va[rd]
            resp[cm, 1 + ans[cm]] = 1.0
        else:
            T = self.quality(L)
            va = np.zeros(n)
            if rd.any():                                   # "read the smallest line you can"
                ln, k, m = chart.read_all(T[rd], self.rng)
                va[rd] = chart.score(ln, k, m)
                kk[rd], nn[rd], line_read[rd] = k, m, ln
                self.line[rd] = np.minimum(ln, 1.0)        # the examiner isolates that line next
            if rl.any():                                   # "read this line"
                k, m = chart.read_line(self.line[rl], T[rl], self.rng)
                kk[rl], nn[rl] = k, m
                va[rl] = self.line[rl]
            resp[rd | rl, 0] = line_read[rd | rl]
            resp[rd | rl, 1] = (kk / nn)[rd | rl]
            resp[cm, 2 + ans[cm]] = 1.0
        code = np.where(rd, 3, np.where(rl, 4, np.where(cm, ans, -1)))
        if self.record:
            self.last = dict(a=a.copy(), code=code, va=va, A=A, B=B, L=L, k=kk, nl=nn, line=line_read,
                             screen=self.line.copy())

        # memory
        lf = self._lens_feat()
        self.hist_a = np.roll(self.hist_a, 1, 1)
        self.hist_r = np.roll(self.hist_r, 1, 1)
        self.hist_l = np.roll(self.hist_l, 1, 1)
        self.hist_a[:, 0] = a
        self.hist_r[:, 0] = resp
        self.hist_l[:, 0] = lf
        for k, act in enumerate(self.questions):
            m = a == act
            if m.any():
                if act in (READ, READ_LINE):
                    frac = (kk / nn)[m] if self.letters else 0 * va[m]
                    self.lq_ans[m, k] = np.stack([va[m], frac, 0 * va[m]], -1)
                else:
                    self.lq_ans[m, k] = resp[m, self.R - 3:]
                self.lq_on[m, k] = 1
                self.lq_l[m, k] = lf[m]
                self.lq_t[m, k] = self.t[m]
        better = rd & ((self.best_on == 0) | (va <= self.best_va))
        self.best_va[better] = va[better]
        self.best_on[better] = 1
        self.best_l[better] = lf[better]

        # end of exam
        done = (a == DONE) | (self.t >= self.t_max)
        if self.shaping:
            # potential-based shaping (Ng, Harada & Russell 1999): F = gamma*phi(s') - phi(s), phi(terminal) = 0.
            # It changes when credit arrives, not which strategy is best.
            phi1 = np.where(done, 0.0, -2.0 * rx_error(self.lens_pv(), self.true_pv(), self.cyl_weight))
            rew = rew + self.shaping * (self.gamma * phi1 - phi0)
        info = {}
        if done.any():
            idx = np.nonzero(done)[0]
            Lf = self.lens_pv(idx)
            E = self.true_pv(idx)
            err = rx_error(Lf, E)                                   # what we report: always the full-cylinder score
            train_err = rx_error(Lf, E, self.cyl_weight)            # what the AI is rewarded on
            rew[idx] += final_reward(train_err) - 0.2 * (a[idx] != DONE)
            info = dict(idx=idx, err=err, steps=self.t[idx].copy(), dM=Lf[0] - E[0], age=self.P["age"][idx].copy(),
                        S=self.S[idx].copy(), C=self.C[idx].copy(), ax=self.ax[idx].copy(),
                        tS=self.P["S"][idx].copy(), tC=self.P["C"][idx].copy(), tax=self.P["axis"][idx].copy(),
                        err0=rx_error(tuple(self.start_pv[idx].T), E), err_half=pv_dist(Lf, E),
                        pid=self.P["pid"][idx].copy() if "pid" in self.P else np.full(len(idx), -1.0))
            self._reset(idx)
        return self.obs(), rew.astype(np.float32), done, info

    # ---------------------------------------------------------------- what the AI sees
    def obs(self):
        n = self.n
        lf = self._lens_feat()
        cur = lf[:, :3]
        glob = np.concatenate([
            self.P["age"][:, None] / 80, self.S[:, None] / 6, self.C[:, None] / 3, lf[:, :1] / 6, lf[:, 1:],
            self.start_pv - cur, self.t[:, None] / T_MAX], 1)
        best = np.concatenate([self.best_va[:, None], self.best_on[:, None],
                               (self.best_l[:, :3] - cur) * self.best_on[:, None]], 1)
        on = self.lq_on[..., None]
        lq = np.concatenate([self.lq_ans, on, (self.lq_l[..., :3] - cur[:, None]) * on, self.lq_l[..., 3:] * on,
                             ((self.t[:, None] - self.lq_t) / T_MAX * self.lq_on)[..., None]], 2).reshape(n, -1)
        valid = (self.hist_a >= 0)[..., None]
        onehot = np.zeros((n, H, self.n_act))
        np.put_along_axis(onehot, np.maximum(self.hist_a, 0)[..., None], 1.0, 2)
        hist = np.concatenate([onehot, self.hist_r, self.hist_l[..., :3] - cur[:, None], self.hist_l[..., 3:]], 2)
        hist = (hist * valid).reshape(n, -1)
        parts = [glob, best, lq, hist]
        if self.letters:
            parts.append(self.line[:, None])
        if self.flag:
            parts.append(self.P["has_reading"][:, None])
        o = np.concatenate(parts, 1)
        return np.clip(o, -6, 6).astype(np.float32)


if __name__ == "__main__":
    import time
    env = RefractionEnv(1024, seed=1)
    o = env.obs()
    print("obs", o.shape, N_OBS)
    t0 = time.time()
    errs, steps, err0 = [], [], []
    for i in range(500):
        o, r, d, info = env.step(env.rng.integers(0, N_ACT, env.n))
        if info:
            errs += list(info["err"]); steps += list(info["steps"]); err0 += list(info["err0"])
    dt = time.time() - t0
    print(f"{500 * env.n / dt:,.0f} steps/s; random policy: {len(errs)} exams, mean err {np.mean(errs):.3f} D, "
          f"mean steps {np.mean(steps):.1f}; autorefractor err {np.mean(err0):.3f} D, "
          f"within 0.25 D: {np.mean(np.array(err0) <= 0.25):.1%}")
