"""Monte Carlo of the stage 2 decision rule (PLAN.md "Decision rule"), on synthetic numbers only.

No data, no model, no network. Adapted from the PR #41 reviewer's sketch (scratchpad review41/power.py) and updated
to the corrected rule: the noise-scaled threshold T applies to H (harm reproduced) only; N (not reproduced) uses a
separate margin, fixed (M1) or equivalence-style (M2); precedence H, then N, else I (inside the noise).

Assumptions (stated, speculative about the real experiment):
  - Y(c, s) = mu_c + sigma * eps, eps i.i.d. standard normal: training seeds independent, the same seed SD sigma in
    every configuration, evaluation noise folded into sigma, no pairing benefit from shared data order.
  - Scope: the four-configuration design (Open item b = no). With b = yes, the rule pools sigma_hat over six
    configurations (2 more degrees of freedom each), which this simulation does not model; its numbers are then a guide
    only.
  - Four configurations with k seeds each (none, dinov3, dinov3+siglip2, dinov3+c-radiov4); sigma_hat pooled over all
    four with k - 1 degrees of freedom each. Only mu(pair) - mu(dinov3) = delta matters for H and N; mu(dinov3) matters
    only for threshold option 2.
  - Options: T1 = 2.7; T2 = 0.032 x Ybar(dinov3); T3 = max(2.7, 2 sqrt(2/k) sigma_hat); T4 = dbar <= -2.7 and
    dbar + t_{0.95, k-1} s_d / sqrt(k) < 0. Margins: M1 = dbar > -1.35; M2 = dbar - 2 sqrt(2/k) sigma_hat > -2.7.
  - H always also needs every d_s < 0.
  - The calibration test C is not modelled (treated as passed). Under option C-a the real probability of a harm claim
    is P(H and C), so lower than shown.

Run: /opt/homebrew/bin/python3.11 power_sim.py > results/power-sim.json
"""

import json

import numpy as np

SEED = 0
R = 200_000
T_ONE_SIDED_95 = {2: 2.920, 4: 2.132}   # Student t, one-sided 95% (recalled table values)


def simulate(rng, sigma, delta, k, mu_d):
    Y = rng.standard_normal((R, 4, k)) * sigma
    Y[:, 1, :] += mu_d
    Y[:, 3, :] += mu_d + delta
    d = Y[:, 3, :] - Y[:, 1, :]
    dbar = d.mean(1)
    sd_d = d.std(1, ddof=1)
    resid = Y - Y.mean(2, keepdims=True)
    sh = np.sqrt((resid ** 2).sum((1, 2)) / (4 * (k - 1)))
    f = 2 * np.sqrt(2 / k)
    allneg = (d < 0).all(1)
    H = {"T1": allneg & (dbar <= -2.7),
         "T2": allneg & (dbar <= -0.032 * Y[:, 1, :].mean(1)),
         "T3": allneg & (dbar <= -np.maximum(2.7, f * sh)),
         "T4": allneg & (dbar <= -2.7) & (dbar + T_ONE_SIDED_95[k - 1] * sd_d / np.sqrt(k) < 0)}
    N = {"M1": dbar > -1.35, "M2": dbar - f * sh > -2.7}
    out = {}
    for tn, h in H.items():
        for mn, n in N.items():
            n_ = n & ~h
            out[f"{tn}+{mn}"] = {"H": round(float(h.mean()), 3), "N": round(float(n_.mean()), 3),
                                 "I": round(float(1 - h.mean() - n_.mean()), 3)}
    return out


def simulate_extension(rng, sigma, delta):
    """The pre-registered extension as written: dinov3 and the pair get 5 seeds, none and the helpful pair keep 3;
    sigma_hat pooled with n_c - 1 degrees of freedom (2 + 4 + 2 + 4 = 12). Unconditional (not only after an I at
    3 seeds). Sign condition: all five d_s < 0, or at least four of five."""
    n = (3, 5, 3, 5)
    Y = [rng.standard_normal((R, k)) * sigma for k in n]
    Y[3] = Y[3] + delta
    d = Y[3] - Y[1]
    dbar = d.mean(1)
    ss = sum(((y - y.mean(1, keepdims=True)) ** 2).sum(1) for y in Y)
    sh = np.sqrt(ss / sum(k - 1 for k in n))
    f = 2 * np.sqrt(2 / 5)
    neg = (d < 0).sum(1)
    out = {}
    for sign_name, sign in (("all5", neg == 5), ("4of5", neg >= 4)):
        for tn, T in (("T1", np.full(R, 2.7)), ("T3", np.maximum(2.7, f * sh))):
            h = sign & (dbar <= -T)
            n_ = (dbar - f * sh > -2.7) & ~h
            out[f"{tn}+M2,{sign_name}"] = {"H": round(float(h.mean()), 3), "N": round(float(n_.mean()), 3),
                                           "I": round(float(1 - h.mean() - n_.mean()), 3)}
    return out


def main():
    rng = np.random.default_rng(SEED)
    ext = []
    for sigma in (2.0, 3.0, 4.0):
        for delta in (-5.41, -2.7, 0.0):
            ext.append({"sigma": sigma, "delta": delta, "p": simulate_extension(rng, sigma, delta)})
    rows = []
    for k in (3, 5):
        for mu_d in (84.0, 30.0):
            for sigma in (1.0, 2.0, 3.0, 4.0):
                for delta in (-5.41, -2.7, 0.0):
                    rows.append({"k": k, "mu_dinov3": mu_d, "sigma": sigma, "delta": delta,
                                 "p": simulate(rng, sigma, delta, k, mu_d)})
    print(json.dumps({"what": "simulated outcome probabilities of the stage 2 rule (H, N, I) for the harmful-pair "
                              "comparison, synthetic numbers, assumptions in power_sim.py docstring",
                      "seed": SEED, "draws": R, "scope": "four-configuration design (Open item b = no); sigma_hat pooled over none, dinov3 and the two pairings only, in rows and extension; the calibration test C is not modelled (treated as passed)", "note": "k = 5 rows assume 5 seeds in every configuration (an upper "
                      "bound on what the 2-seed extension of dinov3 and dinov3+c-radiov4 buys)",
                      "rows": rows, "extension": ext}, indent=1))


if __name__ == "__main__":
    main()
