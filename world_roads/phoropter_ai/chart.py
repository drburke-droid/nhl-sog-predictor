"""A projected letter chart, read letter by letter (used from attempt 3 on).

* Lines run from logMAR 1.0 (20/200) to -0.3 (20/10) in 0.1 steps. Small lines carry 5 letters. Big lines
  carry fewer, as on a projected chart: as many as fit the screen one letter-width apart (at most 5).
* Each letter is recognised with probability
      p(L) = g + (1 - g) / (1 + exp(-(L - T) / s)),   g = 0.1 (10 Sloan letters), s = 0.04 logMAR,
  where L is the line's logMAR and T is the patient's acuity threshold through the current lens (the blur
  model in optics.py), jittered by N(0, 0.02) each time they read. At L = T they get 55% of letters;
  one line bigger, 93%; one line smaller, 17%.
* "Read the smallest line you can": the patient reads down the chart and stops at the first line where they
  get half or fewer right. They report the last line before that and how many of its letters they got.
* "Read this line": the patient reads only the line on the screen; the answer is letters correct of n.
"""
import numpy as np

LINES = np.round(np.arange(1.0, -0.301, -0.1), 1)          # top to bottom
GUESS, SLOPE, JITTER = 0.1, 0.04, 0.02
SCREEN_ARCMIN = 110.0                                       # projector width available for a line of letters


def n_letters(L):
    h = 5 * 10 ** np.asarray(L)                             # letter height in arcmin (5 x MAR)
    return np.clip(np.floor((SCREEN_ARCMIN / h + 1) / 2), 1, 5).astype(int)


N_PER_LINE = n_letters(LINES)


def p_correct(L, T):
    return GUESS + (1 - GUESS) / (1 + np.exp(-(L - T) / SLOPE))


def read_all(T, rng):
    """T: thresholds (n,). Returns (line logMAR reported, letters correct on it, letters on it).
    A patient who cannot read the top line reports line 1.1 with 0 of 1."""
    T = T + rng.normal(0, JITTER, T.shape)
    p = p_correct(LINES[None, :], T[:, None])               # (n, lines)
    k = rng.binomial(N_PER_LINE[None, :], p)
    ok = 2 * k > N_PER_LINE[None, :]                        # more than half right
    fail = ~ok
    first_fail = np.where(fail.any(1), fail.argmax(1), len(LINES))
    last_ok = first_fail - 1
    line = np.where(last_ok >= 0, LINES[np.maximum(last_ok, 0)], 1.1)
    kk = np.where(last_ok >= 0, k[np.arange(len(T)), np.maximum(last_ok, 0)], 0)
    nn = np.where(last_ok >= 0, N_PER_LINE[np.maximum(last_ok, 0)], 1)
    return line, kk, nn


def read_line(L, T, rng):
    T = T + rng.normal(0, JITTER, T.shape)
    n = n_letters(L)
    return rng.binomial(n, p_correct(L, T)), n


def score(line, k, n):
    """Letter-by-letter acuity from a 'smallest line' reading: each missed letter on that line adds 0.02."""
    return line + 0.1 * (n - k) / np.maximum(n, 1)


def snellen_of(L):
    den = {1.1: "worse than 20/200", 1.0: "20/200", 0.9: "20/160", 0.8: "20/125", 0.7: "20/100", 0.6: "20/80",
           0.5: "20/63", 0.4: "20/50", 0.3: "20/40", 0.2: "20/32", 0.1: "20/25", 0.0: "20/20", -0.1: "20/16",
           -0.2: "20/12.5", -0.3: "20/10"}
    return den[round(float(L), 1) + 0.0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("letters per line:", dict(zip(LINES.tolist(), N_PER_LINE.tolist())))
    for T0 in (0.0, 0.15, 0.4):
        line, k, n = read_all(np.full(4000, T0), rng)
        sc = score(line, k, n)
        print(f"threshold {T0:+.2f}: reported line mean {line.mean():+.3f} sd {line.std():.3f}; "
              f"letter score mean {sc.mean():+.3f} sd {sc.std():.3f}")
    for L in (0.1, 0.0, -0.1):
        k, n = read_line(np.full(4000, L), np.full(4000, 0.0), rng)
        print(f"threshold 0.0, line {L:+.1f}: mean letters {k.mean():.2f} of {n[0]}")
