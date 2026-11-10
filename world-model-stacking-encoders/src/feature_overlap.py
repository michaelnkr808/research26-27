"""One-way feature overlap (containment) score for a pair of frozen encoders on cached features.

Binding spec: experiments/2026-11-10-feature-overlap/PLAN.md (signed off 2026-11-10, primary score P = S2). C(Y in X) is
the held-out score of a linear map from source X's features to target Y's features, both cameras concatenated.
For base D (DINOv3) and partner B:
  S1 = C(D in B)  (source B, target D),   S2 = C(B in D)  (source D, target B; the primary P),   S3 = S2 - S1.
For a context-matrix pair (X, Y) neither is DINOv3: the same code, --base X --partner Y, and the S1/S2/S3 labels read
relative to X.

One direction (source -> target):
  - The source columns are standardized with train-frame mean and std (diagnostics.standardize). The target columns are
    standardized on train frames, then projected on the top --m (64) principal components of the concatenated target,
    components from train frames only; coverage = share of the target's train variance those components hold.
  - Ridge from source to the m components, the formula of diagnostics.ridge_fit (w = (X'X + alpha n I)^-1 X'(y - ybar),
    b = ybar), float64 linear algebra, features float32. One eigendecomposition of X'X per source and fit set (train for
    choosing alpha, train+val for the final fit), reused for every alpha and every target (64 components, all dimensions,
    trivial baseline is a different source and gets its own). alpha from diagnostics.ALPHAS, chosen on validation demos
    only by mean validation R^2, then refit on train+val and scored once on test (as diagnostics.ridge_select).
  - Score = mean over the m components of held-out R^2 (test-set mean as baseline, as diagnostics.r2). Secondaries:
    pooled variance-weighted R^2 over the same components; the all-dimensions mean of per-dimension R^2 (no PCA, own
    alpha; the quantity diagnostics.py stores as pred_A_to_B_r2); a trivial-structure baseline for the same components
    (task one-hot, frame index over demo length, proprio; same ridge and alpha rule) and headroom = score - baseline.
  - Demo-level bootstrap over test demos, --n-boot paired draws (default 2,000; PLAN: at least 1,000), fitted maps held
    fixed, the same draws for every score, so differences (S3, headroom) are paired. --save-draws writes them so separate
    runs can be differenced.

Folds (--fold): per task file, demos in sorted order are cut into five blocks of 20%. fold 1 = tally split (test block 5,
validation block 4; equals diagnostics.demo_split), fold 2 = test block 1 / validation block 2, fold 3 = test block 3 /
validation block 4. Rest of the demos are train. Split is by demo, never by frame.

Test evaluations (_standards/07): each (map at the chosen alpha) scored on the test demos counts once: per direction and
feature type the 64-component map, the all-dimensions map and the trivial-baseline map. The count and the list go into the
result JSON.

Example:
  python feature_overlap.py --feats /work/feats --suite libero_goal --raw /work/data/libero_goal \
      --base dinov3 --partner c-radiov4 --fold 1 --out overlap-libero_goal-dinov3-c-radiov4-f1.json

Control partners (experiments/2026-11-09-control-partners/): --partner may be a synthetic spec generated in memory by
diagnostics.load (noise:4608, dup:dinov3, randproj:dinov3:4608, shuffle:dinov3), seeded by --synth-seed.
"""

import argparse
import hashlib
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import diagnostics as D

CHUNK = 4096
# fold -> (test block, validation block), blocks numbered 1..5 in sorted demo order per task file
FOLDS = {1: (5, 4), 2: (1, 2), 3: (3, 4)}
M_DEFAULT = 64
N_BOOT_MIN = 1000  # STANDARDS.md: demo-level bootstrap needs at least 1,000 draws


# ---------------------------------------------------------------------------
# Rotated demo-level splits
# ---------------------------------------------------------------------------

def fold_blocks(n, fold):
    """Positions (in the file's sorted demo order) of the test and validation demos for one fold; n demos in the file.
    Block sizes are those of diagnostics.demo_split. Block 5 is the last n_test demos, block 4 (validation, fold 1) the
    n_val before them, other blocks start at (block - 1) * size."""
    n_test, n_val = max(1, round(n * 0.2)), max(1, round(n * 0.2))
    tb, vb = FOLDS[fold]
    ts = n - n_test if tb == 5 else (tb - 1) * n_test
    vs = n - n_test - n_val if vb == 4 else (vb - 1) * n_val
    if fold == 3:  # block 4 starts at n - n_test - n_val, which reaches into test block 3 when n is not a multiple of 5
        vs = max(vs, ts + n_test)
    te, va = range(ts, ts + n_test), range(vs, vs + n_val)
    assert 0 <= ts and ts + n_test <= n and 0 <= vs and vs + n_val <= n and not set(te) & set(va), (n, fold, te, va)
    return te, va


def fold_split(index, fold):
    """Frame-level labels ('train', 'val', 'test') for one fold. index: (file stem, demo, length) per demo, as
    diagnostics.load returns it, demos of one file contiguous and in sorted order."""
    labels = []
    for _, grp in itertools.groupby(range(len(index)), key=lambda i: index[i][0]):
        ids = list(grp)
        te, va = fold_blocks(len(ids), fold)
        lab = ["test" if j in te else "val" if j in va else "train" for j in range(len(ids))]
        labels += [(lab[j], index[i][2]) for j, i in enumerate(ids)]
    return np.repeat(np.array([s for s, _ in labels]), [n for _, n in labels])


# ---------------------------------------------------------------------------
# Ridge through one eigendecomposition per source and fit set
# ---------------------------------------------------------------------------

class Fit:
    """Eigendecomposition of X'X on one set of rows. Ridge for any alpha and target is then a projection."""

    def __init__(self, x, rows, g, sx):
        s, self.q = np.linalg.eigh(g)
        self.s = np.maximum(s, 0.0)
        self.x, self.rows, self.n, self.sx = x, rows, len(rows), sx
        self._proj = {}

    def coef(self, y):
        """C0 = Q'X'(y - ybar) and ybar on the fit rows (y: rows-indexable array, one value per frame)."""
        xty, sy = 0.0, 0.0
        for i in range(0, len(self.rows), CHUNK):
            r = self.rows[i:i + CHUNK]
            yc = y[r].astype(np.float64)
            xty = xty + self.x[r].astype(np.float64).T @ yc
            sy = sy + yc.sum(0)
        ybar = sy / self.n
        return self.q.T @ (xty - np.outer(self.sx, ybar)), ybar

    def proj(self, name, rows):
        """X[rows] Q, float64, cached under name (the same rows are projected for every target)."""
        if name not in self._proj:
            self._proj[name] = np.concatenate(
                [self.x[rows[i:i + CHUNK]].astype(np.float64) @ self.q for i in range(0, len(rows), CHUNK)])
        return self._proj[name]

    def shrink(self, alpha):
        return 1.0 / (self.s + alpha * self.n)


def gram(x, rows):
    d = x.shape[1]
    g, sx = np.zeros((d, d)), np.zeros(d)
    for i in range(0, len(rows), CHUNK):
        c = x[rows[i:i + CHUNK]].astype(np.float64)
        g += c.T @ c
        sx += c.sum(0)
    return g, sx


def make_fits(x, tr, va):
    """(fit on train, fit on train+val): the Gram matrices are summed, so the rows are read once."""
    g_tr, s_tr = gram(x, tr)
    g_va, s_va = gram(x, va)
    return Fit(x, tr, g_tr, s_tr), Fit(x, np.concatenate([tr, va]), g_tr + g_va, s_tr + s_va)


def select_and_predict(fit_tr, fit_trva, y, va, te):
    """Alpha by mean validation R^2 over the columns of y (validation demos only; first maximum, as ridge_select), refit
    on train+val, predictions for the test frames. Returns (test predictions, alpha, validation curve)."""
    c_tr, yb_tr = fit_tr.coef(y)
    p_va = fit_tr.proj("va", va)
    y_va = y[va].astype(np.float64)
    curve = {a: float(D.r2(y_va, p_va @ (c_tr * fit_tr.shrink(a)[:, None]) + yb_tr).mean()) for a in D.ALPHAS}
    alpha = max(D.ALPHAS, key=lambda a: curve[a])
    c_trva, yb_trva = fit_trva.coef(y)
    p_te = fit_trva.proj("te", te)
    return p_te @ (c_trva * fit_trva.shrink(alpha)[:, None]) + yb_trva, alpha, curve


# ---------------------------------------------------------------------------
# Scores and bootstrap
# ---------------------------------------------------------------------------

def summaries(y, yhat):
    """Held-out R^2 per column (test-set mean as baseline), their mean, and the pooled variance-weighted R^2."""
    y = y.astype(np.float64)
    per = D.r2(y, yhat)
    pooled = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean(0)) ** 2).sum(), 1e-12)
    return per, float(per.mean()), float(pooled)


def boot_weights(k, n_boot, seed):
    """Demo resampling counts, built exactly as diagnostics.bootstrap_r2 builds them (same seed -> same draws)."""
    return np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)


def pooled_from_stats(w, n, sy, syy, sres):
    nn, s1, s2, sr = w @ n, w @ sy, w @ syy, w @ sres
    ss_tot = s2 - s1 ** 2 / nn[:, None]
    return 1 - sr.sum(1) / np.maximum(ss_tot.sum(1), 1e-12)


def boot_scores(y, yhat, groups, w, pooled=True):
    """Bootstrap draws of the mean-over-columns R^2 (and the pooled R^2) for fixed predictions, resampling demos."""
    st = D.demo_stats(y, yhat, groups)
    out = {"mean": D.r2_from_stats(w, *st)}
    if pooled:
        out["pooled"] = pooled_from_stats(w, *st)
    return out


# ---------------------------------------------------------------------------
# One direction: source -> target
# ---------------------------------------------------------------------------

def prepare_target(y_raw, train, m):
    """Target standardized on train frames, projected on the top-m train principal components. Returns the standardized
    target (float32), the components (n, m) float64 for every frame, V, coverage and the number of components used."""
    ys = D.standardize(y_raw, train)
    rows = np.flatnonzero(train)
    g, sy = gram(ys, rows)
    mu = sy / len(rows)
    ev, q = np.linalg.eigh((g - len(rows) * np.outer(mu, mu)) / len(rows))
    ev = np.maximum(ev, 0.0)
    m = min(m, ys.shape[1])
    v = np.ascontiguousarray(q[:, ::-1][:, :m])  # a reversed view makes later matmuls skip BLAS (very slow)
    y64 = np.concatenate([(ys[i:i + CHUNK].astype(np.float64) - mu) @ v for i in range(0, len(ys), CHUNK)])
    return ys, y64, v, float(ev[::-1][:m].sum() / max(ev.sum(), 1e-300)), m


def edge(alpha):
    return alpha in (D.ALPHAS[0], D.ALPHAS[-1])


def run_direction(x_raw, y_raw, split, groups, m, w, aux_raw=None):
    """C(target in source) for one direction. x_raw (n, d_src) and y_raw (n, d_tgt): raw features of every frame, split:
    'train'/'val'/'test' per frame, groups: demo id per frame, w: bootstrap weights over the test demos (boot_weights).
    aux_raw: (n, k) trivial-structure predictors; if given, the baseline and headroom are computed. Nothing from the test
    frames enters the standardization, the components or alpha. Returns (json-ready dict, draws dict)."""
    train = split == "train"
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    xs = D.standardize(x_raw, train)
    ys, y64, v, coverage, m = prepare_target(y_raw, train, m)
    fit_tr, fit_trva = make_fits(xs, tr, va)  # the only eigendecompositions of this source, reused below
    res = {"n_source_cols": int(xs.shape[1]), "n_target_cols": int(ys.shape[1]), "m": int(m), "coverage": coverage,
           "alpha": {}, "alpha_at_grid_edge": {}, "val_r2_by_alpha": {}}
    draws, gte = {}, groups[te]

    def finish(tag, y, yhat, pooled):
        per, mean, pool = summaries(y[te], yhat)
        d = boot_scores(y[te].astype(np.float64), yhat, gte, w, pooled)
        res[tag] = {"value": mean, "ci95": D.ci(d["mean"])}
        draws[tag] = d["mean"]
        if pooled:
            res[tag + "_pooled"] = {"value": pool, "ci95": D.ci(d["pooled"])}
            draws[tag + "_pooled"] = d["pooled"]
        return per

    yhat, a, curve = select_and_predict(fit_tr, fit_trva, y64, va, te)
    res["alpha"]["primary"], res["alpha_at_grid_edge"]["primary"], res["val_r2_by_alpha"]["primary"] = a, edge(a), curve
    per = finish("score", y64, yhat, True)
    res["per_component_r2"] = [float(p) for p in per]
    yhat, a, curve = select_and_predict(fit_tr, fit_trva, ys, va, te)
    res["alpha"]["alldims"], res["alpha_at_grid_edge"]["alldims"], res["val_r2_by_alpha"]["alldims"] = a, edge(a), curve
    finish("alldims", ys, yhat, False)
    n_eval = 2
    if aux_raw is not None:
        auxs = D.standardize(aux_raw, train)
        a_tr, a_trva = make_fits(auxs, tr, va)
        yhat, a, curve = select_and_predict(a_tr, a_trva, y64, va, te)
        res["alpha"]["baseline"], res["alpha_at_grid_edge"]["baseline"], res["val_r2_by_alpha"]["baseline"] = a, edge(a), curve
        finish("baseline", y64, yhat, False)
        res["headroom"] = {"value": res["score"]["value"] - res["baseline"]["value"],
                           "ci95": D.ci(draws["score"] - draws["baseline"])}
        n_eval += 1
    res["test_evaluations"] = n_eval
    return res, draws


# ---------------------------------------------------------------------------
# One feature type for a (base, partner) pair
# ---------------------------------------------------------------------------

def aux_features(index, proprio):
    """Trivial-structure predictors: task one-hot, frame index over demo length, proprio (joint + gripper states)."""
    stems = sorted({s for s, _, _ in index})
    task = np.concatenate([np.tile((np.array(stems) == s).astype(np.float32), (n, 1)) for s, _, n in index])
    frac = np.concatenate([np.arange(n) / n for _, _, n in index]).astype(np.float32)[:, None]
    return np.concatenate([task, frac, proprio.astype(np.float32)], axis=1)


def run_type(feats, split, index, groups, aux, base, partner, t, m, n_boot, seed, log, show_scores=True):
    """show_scores False (real partners): the log gets timing, alpha, grid-edge flags and coverage only, no scores
    (PLAN blindness: controls are read before any real-pair number is seen). The result JSON is the same either way."""
    x = {e: np.concatenate([feats[e][t][c] for c in D.CAMS], axis=1) for e in (base, partner)}
    te_groups = np.unique(groups[split == "test"])
    w = boot_weights(len(te_groups), n_boot, seed)
    out, draws = {"directions": {}}, {}
    for src, tgt in ((base, partner), (partner, base)):  # first is S2 = C(partner in base), second S1 = C(base in partner)
        tl = time.time()
        r, d = run_direction(x[src], x[tgt], split, groups, m, w, aux)
        r["source"], r["target"], r["seconds"] = src, tgt, time.time() - tl
        out["directions"][f"{src}->{tgt}"] = r
        draws.update({f"{t}|{src}->{tgt}|{k}": v for k, v in d.items()})
        edges = [k for k, hit in r["alpha_at_grid_edge"].items() if hit]
        if show_scores:
            log(f"{t} C({tgt} in {src}): score {r['score']['value']:+.4f} CI95 {r['score']['ci95'][0]:+.4f}..{r['score']['ci95'][1]:+.4f} "
                f"pooled {r['score_pooled']['value']:+.4f} alldims {r['alldims']['value']:+.4f} baseline {r['baseline']['value']:+.4f} "
                f"coverage {r['coverage']:.3f} alpha {r['alpha']['primary']} ({time.time() - tl:.0f}s)")
        else:
            log(f"{t} C({tgt} in {src}): scores withheld from the log; coverage {r['coverage']:.3f} alpha {r['alpha']} "
                f"grid-edge {edges or 'none'} ({time.time() - tl:.0f}s)")
    s2, s1 = f"{base}->{partner}", f"{partner}->{base}"
    sc = {"S2": out["directions"][s2]["score"]["value"], "S1": out["directions"][s1]["score"]["value"]}
    s3_draws = draws[f"{t}|{s2}|score"] - draws[f"{t}|{s1}|score"]
    sc["S3"] = sc["S2"] - sc["S1"]
    out["S1"] = {"definition": f"C({base} in {partner}): source {partner}, target {base}", "value": sc["S1"],
                 "ci95": out["directions"][s1]["score"]["ci95"]}
    out["S2"] = {"definition": f"C({partner} in {base}): source {base}, target {partner}", "value": sc["S2"],
                 "ci95": out["directions"][s2]["score"]["ci95"]}
    out["S3"] = {"definition": "S2 - S1", "value": sc["S3"], "ci95": D.ci(s3_draws)}
    draws[f"{t}|S3|score"] = s3_draws
    if base != "dinov3":
        out["note"] = f"context pair: the base {base} is not DINOv3, so no primary score P applies; S1/S2/S3 are relative to {base}"
    out["test_evaluations"] = sum(d["test_evaluations"] for d in out["directions"].values())
    if show_scores:
        log(f"{t} S1 {sc['S1']:+.4f} S2 {sc['S2']:+.4f} S3 {sc['S3']:+.4f} (CI95 S3 {out['S3']['ci95'][0]:+.4f}..{out['S3']['ci95'][1]:+.4f})")
    return out, draws


def test_demos(index, split):
    """Ordered ids of the test demos (file/demo) and the sha256 of that ordered list, to check two runs share a test set."""
    ids, start = [], 0
    for stem, d, n in index:
        if split[start] == "test":
            ids.append(f"{stem}/{d}")
        start += n
    return {"ids": ids, "sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="root written by extract_features.py --out")
    p.add_argument("--suite", required=True)
    p.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files (demo index, proprio)")
    p.add_argument("--base", default="dinov3", help="D; S2 = C(partner in base) is the primary score")
    p.add_argument("--partner", required=True, help="a real encoder, or a synthetic control spec (see module docstring)")
    p.add_argument("--fold", type=int, default=1, choices=sorted(FOLDS), help="1 = tally split; 2 and 3 rotate the test block")
    p.add_argument("--m", type=int, default=M_DEFAULT, help="target principal components scored")
    p.add_argument("--types", nargs="+", default=["global"], help="feature types, first is primary")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--synth-seed", type=int, default=None, help="seed for a synthetic partner only (default: --seed)")
    p.add_argument("--save-draws", default=None, help=".npz path: every bootstrap draw vector, so separate runs can be differenced")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    log = lambda msg: print(f"[overlap {time.time() - t0:6.0f}s] {msg}", flush=True)
    if args.n_boot < N_BOOT_MIN:
        log(f"WARNING: n_boot {args.n_boot} is below the {N_BOOT_MIN} the standards require; not a reportable run")
    args.encoders = [args.base, args.partner]
    res, all_draws, n_eval = None, {}, 0
    for t in args.types:
        tl = time.time()
        feats, _, proprio, split, index, meta, files = D.load(args, types=(t,))  # one feature type in memory at a time
        log(f"loaded {t} features in {time.time() - tl:.0f}s")
        fsplit = fold_split(index, args.fold)
        if args.fold == 1:
            assert np.array_equal(fsplit, split), "fold 1 must equal diagnostics.demo_split"
        groups = np.repeat(np.arange(len(index)), [n for _, _, n in index])  # demo id per frame
        counts = {s: int((fsplit == s).sum()) for s in ("train", "val", "test")}
        if res is None:
            log(f"{len(fsplit)} frames, {len(index)} demos from {len(files)} files, fold {args.fold}, split {counts}")
            res = {"suite": args.suite, "files": files, "base": args.base, "partner": args.partner,
                   "encoders": {e: meta[e] for e in args.encoders}, "git_commit": D.git_commit(),
                   "config": {"plan": "experiments/2026-11-10-feature-overlap/PLAN.md", "primary": ("S2 = C(partner in base)" if args.base == "dinov3" else
                                          "none: the base is not DINOv3 (context pair), no primary score P applies"),
                              "fold": args.fold, "fold_blocks (test, val)": FOLDS[args.fold], "m": args.m,
                              "types": args.types, "primary_type": args.types[0], "alphas": D.ALPHAS, "seed": args.seed,
                              "synth_seed": args.synth_seed, "n_boot": args.n_boot, "n_boot_meets_standard": args.n_boot >= N_BOOT_MIN,
                              "feature_source": "both cameras concatenated, standardized on train frames"},
                   "test_demos": test_demos(index, fsplit), "n_frames": len(fsplit), "split_frames": counts, "n_test_demos": int(len(np.unique(groups[fsplit == "test"]))),
                   "types": {}}
            split0, index0 = fsplit, index
        assert np.array_equal(fsplit, split0) and index == index0, f"{t}: demo index differs between loads"
        res["types"][t], d = run_type(feats, fsplit, index, groups, aux_features(index, proprio), args.base, args.partner,
                                      t, args.m, args.n_boot, args.seed, log, show_scores=D.is_synthetic(args.partner))
        all_draws.update(d)
        n_eval += res["types"][t]["test_evaluations"]
        del feats
    res["test_evaluations"] = {"count": n_eval, "unit": "map at the chosen alpha scored on the test demos: per direction and "
                               "type the m-component map, the all-dimensions map and the trivial-baseline map"}
    res["alpha_edge_hits"] = [f"{t}|{k}|{what}" for t, r in res["types"].items() for k, dr in r["directions"].items()
                              for what, hit in dr["alpha_at_grid_edge"].items() if hit]
    res["seconds"] = time.time() - t0
    # ru_maxrss is bytes on macOS and KiB on Linux; macOS undercounts a Linux run (project memory), so this is a floor
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    log(f"alpha at grid edge: {res['alpha_edge_hits'] or 'none'}; test evaluations {n_eval}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.save_draws:
        np.savez_compressed(args.save_draws, **all_draws)
    Path(args.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
