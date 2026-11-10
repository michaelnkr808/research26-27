"""Detection power of the locked 3-seed proxy outcome rule, from the first training-seed noise measurement.

Question: if D+R really is worse than D by some true drop, how often does the locked rule
(experiments/2026-10-27-proxy-calibration/PLAN.md, "Pre-registered outcome rule", item 2) say "substantial proxy
effect", given how much two training seeds of the same config disagree?

Plain version of what this does:
  1. Reads the two finished DINOv3 seeds (per-episode records, 10,030 LIBERO-Plus episodes each) and reproduces the
     seed-noise numbers (lp_eval.seednoise, LOG 2026-11-09 (t)).
  2. Turns the one between-seed gap into a point estimate of the training-seed sd (gap / sqrt 2), and shows how
     wide the plausible range is (one degree of freedom: very wide).
  3. Simulates the D+R vs D rule: per training seed, delta_s = -true_drop + training noise + episode-sampling noise;
     applies the rule's branch logic (same as lp_eval.verdict, lines 249-252); prints branch probabilities.

It imports lp_eval.py (importable without side effects: its main() is under __main__) for the thresholds, the record
reader and the seed-noise check. Run with:  python -I world-model-stacking-encoders/src/power_seeds.py [--selftest]

Model assumptions (all stated, none tested here): training noise is Gaussian and independent across (config, seed)
runs, so a same-seed D+R - D gap has training sd sqrt(2)*sd*sqrt(1-rho) with rho = 0 by default (PLAN says same-seed
pairing may correlate the noise; rho > 0 would make detection easier; speculative); episode-sampling noise of a
paired gap is sqrt(2) times the single-mean episode SE; D+R has the same noise level as D; calibration is assumed to
have passed (the rule's step 2 only applies then). The 5-seed rule is a hypothetical generalisation (all five deltas
negative and mean <= -2.7), NOT the locked rule.
"""
import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # python -I drops the script dir from sys.path
import lp_eval as L                     # noqa: E402

EXP = HERE.parent / "experiments" / "2026-10-27-proxy-calibration" / "results" / "full-benchmark"
RNG_SEED = 20261109
DROPS = (5.41, 2.7, 1.35, 0.0)          # paper's D+R - D drop, the locked harm line, the calibration margin, none
HIGH_MULTS = (8.0, 16.0, 32.0)           # extra rows, 3 seeds, paper-size drop only
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)      # grid on the training-seed sd, relative to the one-pair estimate
SEED_COUNTS = (3, 5)
N_SIM = 200_000

B_EFFECT = "effect"                      # lp_eval V_EFFECT
B_ANY_NONNEG = "no effect (some seed >= 0)"
B_MEAN_SHORT = "no effect (all < 0, mean above -2.7)"


# ---------------------------------------------------------------------------
# The locked D+R vs D branch logic, mirrored from lp_eval.verdict (lines 249-252):
#   elif all(x < 0 for x in exact["DR-D"][0]) and exact["DR-D"][1] <= HARM_MAX_MEAN: V_EFFECT   else V_NO_EFFECT
# The two "no effect" sub-branches are split here for information only; they are one rule branch in lp_eval.
# ---------------------------------------------------------------------------

def branch(deltas):
    """deltas: per-seed D+R - D in points (floats). Returns one of the three branch labels above."""
    harm = float(L.HARM_MAX_MEAN)
    all_neg = all(x < 0 for x in deltas)
    mean = sum(deltas) / len(deltas)
    if all_neg and mean <= harm:
        return B_EFFECT
    return B_ANY_NONNEG if not all_neg else B_MEAN_SHORT


def branch_probs(deltas):
    """deltas: (N, k) array. Vectorised version of branch(); returns the three probabilities."""
    harm = float(L.HARM_MAX_MEAN)
    all_neg = (deltas < 0).all(axis=1)
    eff = all_neg & (deltas.mean(axis=1) <= harm)
    return float(eff.mean()), float((~all_neg).mean()), float((all_neg & ~eff).mean())


def simulate(drop, sd_train, se_ep_mean, k, n, rng, rho=0.0):
    """(n, k) array of per-seed D+R - D gaps in points. Training noise: each run's mean ~ N(0, sd_train^2), so the gap
    has sd sqrt(2(1-rho)) sd_train. Episode noise of a paired gap: sqrt(2) * se_ep_mean."""
    tr = math.sqrt(2 * (1 - rho)) * sd_train
    ep = math.sqrt(2) * se_ep_mean
    return -drop + tr * rng.standard_normal((n, k)) + ep * rng.standard_normal((n, k))


# ---------------------------------------------------------------------------
# Measured numbers
# ---------------------------------------------------------------------------

def chi2_1_quantile(q):
    z = NormalDist().inv_cdf((1 + q) / 2)
    return z * z


def sd_interval(delta, level=0.95):
    """sd_train hat = |delta|/sqrt2. delta ~ N(0, 2 sd^2) => delta^2/(2 sd^2) ~ chi2 with 1 df, so
    sd in [ |delta|/sqrt2 / sqrt(q_hi), |delta|/sqrt2 / sqrt(q_lo) ]."""
    est = abs(delta) / math.sqrt(2)
    lo = est / math.sqrt(chi2_1_quantile((1 + level) / 2))
    hi = est / math.sqrt(chi2_1_quantile((1 - level) / 2))
    return est, lo, hi


def measure(dir_a, dir_b, subset_path):
    r = L.seednoise(dir_a, dir_b, subset_path)
    if r["problems"]:
        raise SystemExit("seed-noise inputs incomplete: " + "; ".join(r["problems"]))
    ea = L._sn_episodes(L._sn_load(dir_a)[1])
    eb = L._sn_episodes(L._sn_load(dir_b)[1])
    keys = sorted(set(ea) & set(eb))
    sa = np.array([sum(ea[k].values()) / len(ea[k]) for k in keys])      # per-episode score, 3 eval seeds averaged
    sb = np.array([sum(eb[k].values()) / len(eb[k]) for k in keys])
    n = len(keys)
    d = (sa - sb) * 100
    return {"r": r, "n": n,
            "mean_a": float(r["mean_a_pts"]), "mean_b": float(r["mean_b_pts"]), "delta": float(r["delta_pts"]),
            "se_a": float(sa.std(ddof=1) * 100 / math.sqrt(n)), "se_b": float(sb.std(ddof=1) * 100 / math.sqrt(n)),
            "binom_a": 100 * math.sqrt(sa.mean() * (1 - sa.mean()) / n),
            "binom_b": 100 * math.sqrt(sb.mean() * (1 - sb.mean()) / n),
            "d_sd": float(d.std(ddof=1)), "d_se": float(d.std(ddof=1) / math.sqrt(n)),
            "a_lower": r["a_lower"], "a_higher": r["a_higher"], "equal": r["equal"]}


def print_measured(m):
    print("== Measured: dinov3, training seed 1 (A) vs training seed 0 (B), %d LIBERO-Plus episodes" % m["n"])
    print(f"mean success A: {m['mean_a']:.3f} points   B: {m['mean_b']:.3f} points")
    print(f"between-seed delta A - B: {m['delta']:+.3f} points   (episodes A lower {m['a_lower']}, "
          f"A higher {m['a_higher']}, equal {m['equal']})")
    print(f"paired per-episode difference: sd {m['d_sd']:.2f} points, so episode-sampling SE of the mean gap "
          f"{m['d_se']:.3f} points")
    print(f"episode-sampling SE of one config's mean over {m['n']} episodes: A {m['se_a']:.3f}, B {m['se_b']:.3f} points "
          f"(empirical sd of the 3-eval-seed score / sqrt n); binary-success formula sqrt(p(1-p)/n): "
          f"A {m['binom_a']:.3f}, B {m['binom_b']:.3f}")
    est, lo, hi = sd_interval(m["delta"])
    print("== Training-seed sd from this one pair")
    print(f"point estimate |delta|/sqrt 2 = {est:.3f} points (includes a little episode-sampling noise; "
          f"expected sd of the gap from episodes alone is {m['d_se']:.2f})")
    print(f"95% chi-square (1 df) range: {lo:.3f} to {hi:.2f} points  ({lo / est:.2f}x to {hi / est:.1f}x the estimate)")
    print("One pair gives a point estimate with no useful confidence interval: the data fit a training-seed sd")
    print("from a few tenths of a point to more than twenty points. The grid below covers 0.25x to 4x only.")
    return est


# ---------------------------------------------------------------------------
# Power table
# ---------------------------------------------------------------------------

def power_table(est_sd, se_ep_mean, rho=0.0, n=N_SIM):
    rng = np.random.default_rng(RNG_SEED)
    lines = []
    for k in SEED_COUNTS:
        locked = "locked rule" if k == 3 else "hypothetical generalisation, not locked"
        lines.append(f"### {k} training seeds each of D and D+R ({locked}); {n} simulated rounds per cell")
        lines.append("| true drop (pts) | seed sd (pts) | multiple of estimate | P(effect) | P(no effect, some seed >= 0) "
                     "| P(no effect, all < 0 but mean above -2.7) |")
        lines.append("|---|---|---|---|---|---|")
        for drop in DROPS:
            for mult in MULTS:
                sd = est_sd * mult
                pe, pn, pm = branch_probs(simulate(drop, sd, se_ep_mean, k, n, rng, rho))
                lines.append(f"| {drop:g} | {sd:.2f} | {mult:g}x | {pe:.3f} | {pn:.3f} | {pm:.3f} |")
        lines.append("")
    return "\n".join(lines)


def high_noise_table(est_sd, se_ep_mean, rho=0.0, n=N_SIM):
    """3 seeds, paper-size drop only, at the upper part of the one-pair range (separate RNG stream, so the main
    tables above do not change)."""
    rng = np.random.default_rng(RNG_SEED + 1)
    lines = [f"### 3 training seeds (locked rule), true drop {DROPS[0]:g}, noise further up the one-pair range; "
             f"{n} simulated rounds per cell",
             "| true drop (pts) | seed sd (pts) | multiple of estimate | P(effect) | P(no effect, some seed >= 0) "
             "| P(no effect, all < 0 but mean above -2.7) |", "|---|---|---|---|---|---|"]
    for mult in HIGH_MULTS:
        sd = est_sd * mult
        pe, pn, pm = branch_probs(simulate(DROPS[0], sd, se_ep_mean, 3, n, rng, rho))
        lines.append(f"| {DROPS[0]:g} | {sd:.2f} | {mult:g}x | {pe:.3f} | {pn:.3f} | {pm:.3f} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _synth_tables(rng, n_ep, drop_level):
    """Tiny synthetic LIBERO-Plus table for 5 configs x 3 seeds so lp_eval.verdict can be run end to end."""
    episodes = [{"suite": "s", "id": i, "category": "c"} for i in range(n_ep)]
    subset = {"episodes": episodes}
    base = {L.NONE: 0.05, L.D: 0.25, L.DL: 0.25, L.DR: 0.25 - drop_level}
    table = {}
    for c, p in base.items():
        for s in L.TRAIN_SEEDS:
            table[(c, s)] = {("s", i): int(rng.binomial(3, min(max(p, 0), 1))) for i in range(n_ep)}
    return table, subset


def selftest():
    rng = np.random.default_rng(RNG_SEED)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (" " + info if info else ""))
        ok &= bool(cond)

    # 1. the estimator recovers a known training-seed sd (no episode noise: pure estimator check)
    true_sd = 1.7
    pairs = rng.normal(0, true_sd, (200_000, 2))
    delta = pairs[:, 0] - pairs[:, 1]
    est = math.sqrt(np.mean(delta ** 2) / 2)
    check("sd recovery from pairs (rms of delta/sqrt2)", abs(est - true_sd) / true_sd < 0.01, f"est {est:.3f} vs {true_sd}")

    # 1b. a single draw of |gap|/sqrt2 usually lands below the true sd: its median is 0.6745 x sd
    single = np.abs(delta) / math.sqrt(2)
    check("median of a single-draw estimate is about 0.6745 x sd", abs(np.median(single) / true_sd - 0.6745) < 0.01,
          f"{np.median(single) / true_sd:.4f}")
    check("a single draw is below the true sd about 68% of the time", abs(np.mean(single < true_sd) - 0.6827) < 0.01)

    # 2. one-pair interval has nominal coverage
    est1, lo, hi = zip(*[sd_interval(x) for x in delta[:20000]])
    cover = np.mean([(l <= true_sd <= h) for l, h in zip(lo, hi)])
    check("95% chi-square interval coverage", abs(cover - 0.95) < 0.01, f"coverage {cover:.3f}")

    # 3. simulator reproduces the requested mean and sd of a per-seed gap
    g = simulate(2.7, 1.0, 0.3, 3, 200_000, rng)
    want_sd = math.sqrt(2 * 1.0 ** 2 + 2 * 0.3 ** 2)
    check("simulated gap mean", abs(g.mean() + 2.7) < 0.01, f"{g.mean():.3f}")
    check("simulated gap sd", abs(g.std() - want_sd) / want_sd < 0.01, f"{g.std():.3f} vs {want_sd:.3f}")

    # 4. no noise: branch probabilities are exactly 0 or 1 at the thresholds' two sides
    z = simulate(2.71, 0.0, 0.0, 3, 10, rng)
    check("no noise, drop 2.71 -> effect", branch_probs(z)[0] == 1.0)
    z = simulate(2.69, 0.0, 0.0, 3, 10, rng)
    check("no noise, drop 2.69 -> mean short", branch_probs(z)[2] == 1.0)
    check("zero drop, small noise -> effect is rare", branch_probs(simulate(0, 0.1, 0.1, 3, 10000, rng))[0] == 0.0)

    # 5. branch() agrees with lp_eval.verdict on synthetic tables (the real rule code), several drop sizes
    agree = 0
    total = 0
    for drop_level in (0.0, 0.02, 0.04, 0.08, 0.15):
        for rep in range(6):
            table, subset = _synth_tables(rng, 400, drop_level)
            v = L.verdict(table, subset)
            if v["verdict"] == L.V_CAL_FAIL:
                continue
            deltas = v["pairs"]["DR-D"]["per_seed_pts"]
            mine = branch(deltas)
            real_effect = v["verdict"] == L.V_EFFECT
            total += 1
            agree += (mine == B_EFFECT) == real_effect
    check("branch() agrees with lp_eval.verdict", total >= 10 and agree == total, f"{agree}/{total} tables")
    print("selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--a-dir", default=str(EXP / "rollouts" / "eval-dinov3-s1"))
    ap.add_argument("--b-dir", default=str(EXP / "rollouts" / "eval-dinov3-s0"))
    ap.add_argument("--subset", default=str(EXP / "full-subset.json"))
    ap.add_argument("--rho", type=float, default=0.0, help="within-seed correlation of D and D+R training noise "
                    "(0 = independent; speculative that it is > 0)")
    ap.add_argument("--sims", type=int, default=N_SIM)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    m = measure(a.a_dir, a.b_dir, a.subset)
    est = print_measured(m)
    se = (m["se_a"] + m["se_b"]) / 2
    print()
    print(f"== Power table (rng seed {RNG_SEED}; seed sd grid = {MULTS} x {est:.3f} points; episode-sampling SE of one "
          f"mean = {se:.3f} points, of a paired gap sqrt2 x that = {math.sqrt(2) * se:.3f}; rho = {a.rho:g})")
    print("Locked rule, D+R vs D (lp_eval.verdict lines 249-252): effect iff every seed's gap < 0 and mean gap <= -2.7;")
    print("anything else is 'no substantial effect'. Calibration assumed passed.")
    print()
    print(power_table(est, se, a.rho, a.sims))
    print(high_noise_table(est, se, a.rho, a.sims))


if __name__ == "__main__":
    main()
