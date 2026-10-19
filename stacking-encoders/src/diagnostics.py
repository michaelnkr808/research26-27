"""Phase 1 diagnostics on cached frozen-encoder features (output of extract_features.py).

For every pair of encoders (A, B) this computes:
  - CKA(A, B): linear CKA, biased and debiased (Song et al. 2012 unbiased HSIC), per camera
    and feature type, on a fixed random frame subsample. Symmetric.
  - Action probes: ridge regression from features (both cams concatenated) to the 7-d action
    at the same timestep. Test R^2 for A, B and A+B (concatenation).
  - gain(B|A) = R^2(A+B) - R^2(A). Asymmetric. This is the probe-level analogue of a combo gain.
  - resid(B|A): R^2 of an action probe trained on the part of B that a ridge map from A cannot
    predict. Asymmetric "information in B not linearly available from A" (H1 in LOG 2026-10-15).
  - pred(A->B): R^2 of predicting B from A (how much of B is linearly redundant given A).
Plus a proprioception probe (joint + gripper states) as a reference point.

Splits are by demo, never by frame (frames within a demo are highly correlated): per task file,
demos sorted by index, the last 20% are test, the 20% before that are validation (ridge alpha
selection), the rest train. The final probe is refit on train+val at the selected alpha.

Example:
  python diagnostics.py --feats /persistent/stacking-encoders/features --suite libero_goal \
      --raw /work/data/libero_goal --encoders siglip2 c-radiov4 --out results.json
"""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

CAMS = ["agentview_rgb", "eye_in_hand_rgb"]
ALPHAS = [1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]


def git_commit():
    import os
    if os.environ.get("SRC_GIT_COMMIT"):
        return os.environ["SRC_GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def demo_split(demos, test_frac=0.2, val_frac=0.2):
    n = len(demos)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    split = {}
    for i, d in enumerate(demos):
        split[d] = "test" if i >= n - n_test else "val" if i >= n - n_test - n_val else "train"
    return split


def load(args):
    """Per-frame global/mean features, actions, proprio, split labels, and a frame index.

    Grid features are not loaded here (too wide for every frame); see load_grid.
    """
    raw_files = sorted(Path(args.raw).glob("*.hdf5"))
    assert raw_files, f"no LIBERO files in {args.raw}"
    feats = {e: {t: {c: [] for c in CAMS} for t in ("global", "mean")} for e in args.encoders}
    actions, proprio, split, index = [], [], [], []  # index: (file stem, demo, length) per demo
    meta = {}
    for raw in raw_files:
        fe = {e: h5py.File(Path(args.feats) / e / args.suite / (raw.stem + ".h5"), "r") for e in args.encoders}
        for e, f in fe.items():
            meta.setdefault(e, json.loads(f.attrs["meta"]))
        with h5py.File(raw, "r") as fr:
            demos = sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            demos = [d for d in demos if all(d in f for f in fe.values())]
            sp = demo_split(demos)
            for d in demos:
                a = fr["data"][d]["actions"][:]
                p = np.concatenate([fr["data"][d]["obs"]["joint_states"][:],
                                    fr["data"][d]["obs"]["gripper_states"][:]], axis=1)
                for e, f in fe.items():
                    for t in ("global", "mean"):
                        for c in CAMS:
                            x = f[d][c][t][:]
                            assert len(x) == len(a), f"{raw.name} {d} {e} {c}: {len(x)} feats vs {len(a)} actions"
                            feats[e][t][c].append(x)
                actions.append(a)
                proprio.append(p)
                split += [sp[d]] * len(a)
                index.append((raw.stem, d, len(a)))
        for f in fe.values():
            f.close()
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    feats = {e: {t: {c: cat(v) for c, v in tv.items()} for t, tv in ev.items()} for e, ev in feats.items()}
    return feats, cat(actions), cat(proprio), np.array(split), index, meta, [r.name for r in raw_files]


def load_grid(args, enc, cam, index, idx):
    """Flattened grid features for the global frame indices idx (sorted), read demo by demo."""
    out, start, j = [], 0, 0
    with_file = {}
    for stem, d, n in index:
        local = []
        while j < len(idx) and idx[j] < start + n:
            local.append(idx[j] - start)
            j += 1
        if local:
            if stem not in with_file:
                with_file[stem] = h5py.File(Path(args.feats) / enc / args.suite / (stem + ".h5"), "r")
            x = with_file[stem][d][cam]["grid"][local]
            out.append(x.reshape(len(local), -1))
        start += n
    for f in with_file.values():
        f.close()
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def _gram(x):
    x = x.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    return x @ x.T


def _hsic_unbiased(k, l):
    n = k.shape[0]
    k = k.copy(); l = l.copy()
    np.fill_diagonal(k, 0); np.fill_diagonal(l, 0)
    kl = k @ l
    return (np.trace(kl) + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * kl.sum() / (n - 2)) / (n * (n - 3))


def cka(x, y):
    k, l = _gram(x), _gram(y)
    n = k.shape[0]
    h = np.eye(n) - 1.0 / n
    kc, lc = h @ k @ h, h @ l @ h
    biased = float((kc * lc).sum() / np.sqrt((kc * kc).sum() * (lc * lc).sum()))
    debiased = float(_hsic_unbiased(k, l) / np.sqrt(_hsic_unbiased(k, k) * _hsic_unbiased(l, l)))
    return biased, debiased


# ---------------------------------------------------------------------------
# Ridge probes
# ---------------------------------------------------------------------------

def standardize(x, train):
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    return (x - mu) / sd


def r2(y, yhat):
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def ridge_fit(x, y, alpha):
    # Primal normal equations; x is standardized so an unpenalized intercept = mean(y).
    d = x.shape[1]
    w = np.linalg.solve(x.T @ x + alpha * len(x) * np.eye(d), x.T @ (y - y.mean(0)))
    return w, y.mean(0)


def ridge_select(x, y, split):
    """Pick alpha on val, refit on train+val, return (test R^2 per dim, alpha, predictor, val R^2 per alpha)."""
    tr, va, te = split == "train", split == "val", split == "test"
    best, val_curve = None, {}
    for a in ALPHAS:
        w, b = ridge_fit(x[tr], y[tr], a)
        score = r2(y[va], x[va] @ w + b).mean()
        val_curve[a] = float(score)
        if best is None or score > best[0]:
            best = (score, a)
    alpha = best[1]
    trva = tr | va
    w, b = ridge_fit(x[trva], y[trva], alpha)
    return r2(y[te], x[te] @ w + b), alpha, (w, b), val_curve


def probe(x, actions, split):
    per_dim, alpha, _, val_curve = ridge_select(x, actions, split)
    return {"r2_mean": float(per_dim.mean()), "r2_per_dim": [float(v) for v in per_dim], "alpha": alpha,
            "val_r2_by_alpha": val_curve}


def residualize(xb, xa, split, rel_tol=1e-8):
    """Part of B not linearly predictable from A (map fit on train+val, applied everywhere).

    Unregularized on purpose: a ridge-shrunk map leaves a scaled copy of the predictable part
    in the residual, which standardization then restores (caught by a synthetic test 2026-10-19).
    Solved as a float64 pseudo-inverse that drops eigen-directions of A'A below rel_tol * max, so a
    rank-deficient A can't blow the weights up (the first full run gave resid R^2 = 0.000 everywhere,
    with a float32 solve at alpha 1e-6; suspected cause, see LOG 2026-10-19).
    Needs n_train >> dim(A) (full libero_goal: ~51k frames vs <= 4.6k dims); returns None below 10x,
    where the map overfits (a 12-demo local run gave test R^2 in the -1000s).
    pred(A->B) is reported from a val-tuned ridge instead, so it is an honest test R^2.
    """
    trva = split != "test"
    pred_r2 = ridge_select(xa, xb, split)[0]
    if trva.sum() < 10 * xa.shape[1]:
        return None, float(pred_r2.mean()), {}
    xa64, xb64 = xa[trva].astype(np.float64), xb[trva].astype(np.float64)
    mu_a, mu_b = xa64.mean(0), xb64.mean(0)
    xa64 -= mu_a
    evals, evecs = np.linalg.eigh(xa64.T @ xa64)
    keep = evals > rel_tol * evals.max()
    v = evecs[:, keep] / evals[keep]
    w = v @ (evecs[:, keep].T @ (xa64.T @ (xb64 - mu_b)))
    resid = (xb.astype(np.float64) - mu_b - (xa.astype(np.float64) - mu_a) @ w).astype(np.float32)
    info = {"rank_kept": int(keep.sum()), "dim_A": int(xa.shape[1]),
            "eig_ratio_kept": float(evals[keep].min() / evals.max()),
            "max_abs_w": float(np.abs(w).max()),
            "resid_var_frac_trva": float(resid[trva].var(0).sum() / xb[trva].var(0).sum())}
    return resid, float(pred_r2.mean()), info


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="root written by extract_features.py --out")
    p.add_argument("--suite", required=True)
    p.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files (for actions)")
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--probe-types", nargs="+", default=["global", "mean"],
                   help="feature types for probes (grid is CKA-only: too wide for primal ridge)")
    p.add_argument("--cka-n", type=int, default=5000, help="frames subsampled for CKA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    feats, actions, proprio, split, index, meta, files = load(args)
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"[diag] {len(actions)} frames from {len(files)} files, split {counts}", flush=True)
    res = {"suite": args.suite, "files": files, "n_frames": len(actions), "split_frames": counts,
           "encoders": {e: meta[e] for e in args.encoders}, "git_commit": git_commit(),
           "alphas": ALPHAS, "seed": args.seed, "cka": {}, "probes": {}, "pairs": {}}

    # CKA on a fixed frame subsample (same frames for every pair).
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(actions), size=min(args.cka_n, len(actions)), replace=False))
    for a, b in itertools.combinations(args.encoders, 2):
        for t in ("global", "mean", "grid"):
            for c in CAMS:
                if t == "grid":
                    xa, xb = load_grid(args, a, c, index, idx), load_grid(args, b, c, index, idx)
                else:
                    xa, xb = feats[a][t][c][idx], feats[b][t][c][idx]
                bi, de = cka(xa, xb)
                res["cka"][f"{a}|{b}|{t}|{c}"] = {"biased": bi, "debiased": de}
                print(f"[diag] CKA {a} vs {b} {t:6s} {c:16s} biased {bi:.3f} debiased {de:.3f}", flush=True)
    # Sanity: CKA of an encoder with itself is 1; with a row-shuffled copy it is ~0 (debiased).
    e0 = args.encoders[0]
    x0 = feats[e0]["mean"][CAMS[0]][idx]
    res["cka_sanity"] = {"self": cka(x0, x0), "shuffled": cka(x0, x0[rng.permutation(len(x0))])}
    print(f"[diag] CKA sanity ({e0} mean): self {res['cka_sanity']['self']}, shuffled {res['cka_sanity']['shuffled']}", flush=True)

    train = split == "train"
    res["probes"]["proprio"] = probe(standardize(proprio, train), actions, split)
    print(f"[diag] probe proprio: R2 {res['probes']['proprio']['r2_mean']:.3f}", flush=True)
    for t in args.probe_types:
        xs = {e: standardize(np.concatenate([feats[e][t][c] for c in CAMS], axis=1), train) for e in args.encoders}
        for e in args.encoders:
            res["probes"][f"{e}|{t}"] = probe(xs[e], actions, split)
            print(f"[diag] probe {e} {t}: R2 {res['probes'][f'{e}|{t}']['r2_mean']:.3f}", flush=True)
        for a, b in itertools.permutations(args.encoders, 2):
            key = f"{a}|{b}|{t}"
            both_key = f"{min(a, b)}+{max(a, b)}|{t}"  # concatenation is symmetric; compute once per pair
            if both_key not in res["probes"]:
                res["probes"][both_key] = probe(np.concatenate([xs[a], xs[b]], axis=1), actions, split)
            both = res["probes"][both_key]
            resid, pred_r2, info = residualize(xs[b], xs[a], split)
            rp = probe(standardize(resid, train), actions, split) if resid is not None else None
            rp_raw = probe(resid, actions, split) if resid is not None else None  # unstandardized residual
            res["pairs"][key] = {
                "gain_B_given_A": both["r2_mean"] - res["probes"][f"{a}|{t}"]["r2_mean"],
                "resid_B_given_A_r2": rp and rp["r2_mean"],
                "resid_B_given_A_alpha": rp and rp["alpha"],
                "resid_B_given_A_val_curve": rp and rp["val_r2_by_alpha"],
                "resid_raw_B_given_A_r2": rp_raw and rp_raw["r2_mean"],
                "pred_A_to_B_r2": pred_r2,
                "resid_map": info,
            }
            rp_s = f"{rp['r2_mean']:.3f} (alpha {rp['alpha']:g}), raw {rp_raw['r2_mean']:.3f}" if rp else "skipped (n < 10 dim)"
            print(f"[diag] {t}: A={a} B={b} gain(B|A) {res['pairs'][key]['gain_B_given_A']:+.3f} "
                  f"resid(B|A) R2 {rp_s} pred(A->B) R2 {pred_r2:.3f} map {info}", flush=True)

    res["seconds"] = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[diag] wrote {args.out} in {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
