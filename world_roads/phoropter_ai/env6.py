"""The clinic exam, version 6 (the fourth clinic video): env5 with a more realistic patient.

Everything in env5.py (plano start, graded answers, the power-search check, spin, the final check "it's still
blurry", the cycling occluder), except:

* **Spinning the axis is a rougher answer than the cross-cylinder.** Watching the dial turn, the patient
  compares what they see now with what they remember from a moment ago, not two lenses flipped side by side:
  their judgement noise is SPIN_K times larger than for a "1 or 2?" question, and "stop!" comes late, the dial
  having turned on by a few degrees (reaction time, SPIN_LATE). A random answer (their lapse rate) stops it
  anywhere.
* **No smile.** The patient gives no running feedback. Now and then, after a lens change they notice (at least
  a line of acuity, through their usual noise), they remark on it, each patient with their own chattiness
  (P_TALK):
    "Oh, that's clear!"                 (better, and now within a line of the best they can see)
    "Better... but it's not that clear." (better, still more than a line from their best)
    "Hmm, that's not as clear."          (worse)
  The AI sees the remarks. With `env.remark_w` > 0 (the fourth video's training: 0.5) they are also its only
  feedback during the exam: each remark counts in proportion to how much clearer the patient's vision got
  (both eyes, each through its own lens) since their previous remark (or since the start). Because each
  remark counts the change since the last one, over an exam they add up to the change between the start and
  the last remark: going back and forth earns nothing. (A first run with remarks as information only,
  `runs/i0`, learned to keep clicking lenses until the button limit and never prescribe.)
* **A reliability dial.** Each patient's answer noise and lapse rate are multiplied by a reliability factor:
  `dial=None` (practice) draws it per patient, log-uniform in DIAL; a number puts every patient at that
  setting (0.5 reliable, 1 typical as in env5, 2 unreliable).

What the AI sees: env5's 1,074 numbers, plus the last remark (which of the three), the lenses then relative
to now, and how long ago: 1,081.
"""
import numpy as np
import env5 as E5
from env5 import (N_ACT3, SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15, AXP2, AXM2,
                  OU_P, OU_M, R_P, L_P, SPIN, CHK_180, CHK_45, SPH, CYL, AXD, OUD, RD, LD)
from env3 import plano_pair, PlanoPairTestSet
from optics import to_pv

SPIN_K = 3.0                       # judgement noise watching a turning dial, relative to a side-by-side flip
SPIN_LATE = (6.0, 4.0)             # "stop!" comes late: the dial turns on by N(6, 4) degrees (never backwards)
REMARK = 0.1                       # a change of a line or more is noticeable
NEAR_BEST = 0.1                    # "that's clear": within a line of the best they can see
P_TALK = (0.05, 0.4)               # each patient's chance of remarking on a change they notice
DIAL = (0.5, 2.5)                  # practice: reliability factor, log-uniform
LAPSE_MAX = 0.3
REMARKS = ("“Oh, that's clear!”", "“Better… but it's not that clear.”", "“Hmm, that's not as clear.”")
N_EXTRA = 7
N_OBS6 = E5.N_OBS3 + N_EXTRA

LENS_ACTS = np.zeros(N_ACT3, bool)
LENS_ACTS[[SP25, SM25, SP100, SM100, CYL_MORE, CYL_LESS, AXP5, AXM5, AXP15, AXM15, AXP2, AXM2, OU_P, OU_M, R_P, L_P,
           SPIN]] = True


class ClinicEnv6(E5.ClinicEnv):
    def __init__(self, n, seed=0, sampler=plano_pair, record=False, start_mode="plano", rand_dial=True, dial=None):
        self.dial = dial
        self.rem_on, self.rem_l, self.rem_t = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
        self._tv0 = np.zeros(n)
        self.remark_w = 0.0
        self.rem_clar, self._rem_rew = np.zeros(n), np.zeros(n)
        super().__init__(n, seed=seed, sampler=sampler, record=record, start_mode=start_mode, rand_dial=rand_dial)
        self.n_obs = N_OBS6

    def _reset(self, idx):
        super()._reset(idx)
        if len(idx) == 0:
            return
        k = len(idx)
        if self.dial is None:
            rel = np.exp(self.rng.uniform(np.log(DIAL[0]), np.log(DIAL[1]), k))
        else:
            rel = np.full(k, float(self.dial))
        for key in ("rel", "talk"):
            if key not in self.P:
                self.P[key] = np.zeros(self.n)
        self.P["rel"][idx] = rel
        self.P["sigma"][idx] = self.P["sigma"][idx] * rel
        self.P["lapse"][idx] = np.minimum(self.P["lapse"][idx] * rel, LAPSE_MAX)
        self.P["talk"][idx] = self.rng.uniform(*P_TALK, k)
        self.rem_on[idx], self.rem_l[idx], self.rem_t[idx] = 0, 0, 0
        self.rem_clar[idx] = self.clarity()[idx]                 # what they see at the start (plano)

    # ------------------------------------------------------------ spinning the axis: rougher than a flip
    def spin(self, rows, c):
        m = len(rows)
        S, C = self.S[rows, c[rows]], self.C[rows, c[rows]]
        sig = self.P["sigma"][rows] * SPIN_K
        best_ax, best_T = self.ax[rows, c[rows]].copy(), np.full(m, np.inf)
        for a in range(5, 181, 5):
            L = to_pv(S, C, np.full(m, float(a)))
            T = np.where(c[rows] == 0, self._mono_rows(0, L, rows), self._mono_rows(1, L, rows))
            T = T + self.rng.normal(0, 1, m) * sig
            better = T < best_T
            best_T, best_ax = np.where(better, T, best_T), np.where(better, float(a), best_ax)
        late = np.maximum(self.rng.normal(*SPIN_LATE, m), 0) * self.rng.choice([-1.0, 1.0], m)
        ax = best_ax + late
        lapse = self.rng.uniform(size=m) < self.P["lapse"][rows]
        ax = np.where(lapse, self.rng.uniform(0, 180, m), ax)
        return np.round((ax - 1) % 180) + 1

    # ------------------------------------------------------------ occasional remarks
    def _view_T(self):
        c, b, L0, L1, Lc, Tm, t0, t1, tb = self._views()
        return np.where(b, tb, Tm)

    def _before(self, a):
        self._tv0 = self._view_T()

    def _after(self, a, resp, code, V):
        n, r = self.n, self.rows
        c, b, L0, L1, Lc, Tm, t0, t1, tb = V
        T1 = np.where(b, tb, Tm)
        changed = LENS_ACTS[a]
        if not changed.any():
            return resp, code, {}
        d = T1 - self._tv0 + self.rng.normal(0, 1, n) * self.P["sigma"] * np.sqrt(2)
        # the best they can see with the viewing eye(s): through their true prescription
        tpv = self._tpv(c)
        best_m = np.where(c == 0, self.mono_T(0, tpv), self.mono_T(1, tpv))
        best_b = self.bino(self.true_pv(0), self.true_pv(1))[2]
        best = np.where(b, best_b, best_m)
        talk = changed & (self.rng.uniform(size=n) < self.P["talk"])
        kind = np.full(n, -1)
        kind = np.where(talk & (d <= -REMARK) & (T1 <= best + NEAR_BEST), 0, kind)
        kind = np.where(talk & (d <= -REMARK) & (T1 > best + NEAR_BEST), 1, kind)
        kind = np.where(talk & (d >= REMARK), 2, kind)
        said = kind >= 0
        self._rem_rew = np.zeros(n)
        if said.any() and self.remark_w:
            cl = self.clarity()
            self._rem_rew = np.where(said, self.remark_w * (cl - self.rem_clar), 0.0)
            self.rem_clar = np.where(said, cl, self.rem_clar)
        if said.any():
            resp = resp.copy()
            resp[said, 2:5] = 0
            resp[said, 2 + kind[said]] = 1.0
            code = np.where(said, 10 + kind, code)
            self.rem_on[said] = np.eye(3)[kind[said]]
            now = self._slot_pv(c, b, L0, L1, Lc)
            self.rem_l[said] = now[said]
            self.rem_t[said] = self.t[said]
        return resp, code, dict(remark=kind)

    def step(self, a):
        self._rem_rew = np.zeros(self.n)
        o, rew, done, info = super().step(a)
        return o, (rew + self._rem_rew).astype(np.float32), done, info

    @staticmethod
    def _slot_pv(c, b, L0, L1, Lc):
        """The lenses in front of the viewing eye(s): one eye (M, J0, J45); both eyes (M right, M left, 0)."""
        mono = np.stack(Lc, 1)
        both = np.stack([L0[0], L1[0], np.zeros_like(L0[0])], 1)
        return np.where(b[:, None], both, mono)

    def obs(self):
        o = super().obs()
        c, b, L0, L1, Lc = self.cur, self.bin > 0, self.lens_pv(0), self.lens_pv(1), None
        Lc = tuple(np.where(c == 0, x0, x1) for x0, x1 in zip(L0, L1))
        on = self.rem_on.max(1, keepdims=True)
        extra = np.concatenate([self.rem_on, (self.rem_l - self._slot_pv(c, b, L0, L1, Lc)) * on,
                                ((self.t - self.rem_t) / 60)[:, None] * on], 1)
        return np.concatenate([o, np.clip(extra, -6, 6).astype(np.float32)], 1)


ClinicEnv = ClinicEnv6


if __name__ == "__main__":
    import time
    env = ClinicEnv6(1024, seed=1)
    o = env.obs()
    print("obs", o.shape, N_OBS6)
    t0 = time.time()
    res, rem = {}, np.zeros(4)
    for i in range(200):
        m = env.mask()
        a = np.array([env.rng.choice(np.nonzero(row)[0]) for row in m])
        env.record = False
        o, rr, d, info = env.step(a)
        if info:
            for k in ("points", "secs", "steps"):
                res.setdefault(k, []).extend(info[k].tolist())
    print(f"{200 * env.n / (time.time() - t0):,.0f} steps/s; random:", {k: round(float(np.mean(v)), 2) for k, v in res.items()})
