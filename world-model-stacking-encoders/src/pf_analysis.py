"""Analysis for the perturbed-frame experiment (experiments/2026-11-06-perturbed-frames/PLAN.md).

Plain idea: train an action probe on clean frames, then see how much it loses when the same demo
moments are re-rendered under four picture changes (camera, light, background, noise). The question
is whether adding C-RADIOv4 (R) to DINOv3 (D) loses more than adding SigLIP 2 (L), SAM 3 (S) or
VGGT-Omega (G). Nothing is trained on perturbed data.

What this file does (all rules are the ones signed off in PLAN.md; see the section names in brackets):
  1. Refit the clean probes from cached clean features: D alone and D+B for B in R, L, S, G (concatenated,
     standardized with clean-train statistics), ridge alpha taken from the tally's result JSONs, fit on
     train+val as diagnostics.py does ["Main measure: the probe", step 1]. Alpha and standardization are
     never re-tuned on perturbed data.
  2. Apply each fitted probe unchanged to the test features of every condition (clean re-render, camera,
     light, background, noise; plus the stored clean features as a pipeline reference, not used by any rule).
     Per-demo sufficient statistics (and, by default, the predictions) are saved so every rule and the
     bootstrap can be recomputed later without refitting.
  3. R2_c(X), drop_c(X) = R2_rerender(X) - R2_c(X), e_c(B) = drop_c(D+B) - drop_c(D),
     gain_c(B|D) = R2_c(D+B) - R2_c(D); pooled e(B), g(B) = equal-weight mean over the four types.
  4. One demo-level paired bootstrap (the same draws for every probe, condition and view), seed recorded.
  5. Rule P (decides), rule B (reported only, three readings of its last clause), the ordering by e(B).
  6. Feature drift per encoder ["Secondary measure: feature drift"].
  7. Matched-dimension control: PCA on clean-train frames only, applied unchanged to every condition;
     only encoders larger than the smallest are reduced (global: C-RADIOv4 and VGGT-Omega to 1,024 per
     camera). Reported as a separate "matched" config next to the "full" one.
  8. Every number is computed for three views: combined (both cameras), agentview, wrist (Q12).

Feature layout (same as diagnostics.py): <root>/<encoder>/<suite>/<file stem>.h5, with
<demo_k>/<cam>/global|mean of shape (T, D). Each condition has its own root with the same layout and holds
at least the test demos. A condition may hold T = n or T = n-1 frames per demo (PLAN Q14 drops each demo's
last frame); every condition and encoder must agree, and actions[:T] are the labels.

Never overwrites: the output dir must not exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import diagnostics as D

CAMS = D.CAMS
REF = "clean_rerender"      # reference for every drop and every drift
STORED = "clean_stored"     # stored clean features on the same frames: pipeline reference only
TYPES = ["camera", "light", "background", "noise"]
DINO = "dinov3"
PARTNERS = {"R": "c-radiov4", "L": "siglip2", "S": "sam3", "G": "vggt-omega"}
# Paper's pair deltas (D+B vs D, closed-loop success; PLAN "The stands out rule"): reported beside ours, no test.
PAPER_DELTA = {"L": 1.83, "S": 0.49, "G": 0.40, "R": -5.41}
PAPER_ORDER = sorted(PAPER_DELTA, key=lambda b: -PAPER_DELTA[b])
VIEWS = {"combined": list(CAMS), "agentview": [CAMS[0]], "wrist": [CAMS[1]]}
MIN_BOOT = 1000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_clean(feats_root, raw_dir, suite, encoders, ftype):
    """Clean stored features through diagnostics.load (same split, same frame order as the tally)."""
    ns = types.SimpleNamespace(raw=str(raw_dir), feats=str(feats_root), suite=suite, encoders=list(encoders))
    return D.load(ns, types=(ftype,))


def layout_test_demos(index, split):
    """Test demos: list of (stem, demo, n, start) in frame order."""
    starts = np.concatenate([[0], np.cumsum([n for _, _, n in index])[:-1]]).astype(int)
    return [(stem, d, n, int(s)) for (stem, d, n), s in zip(index, starts) if split[s] == "test"]


def cond_demo_lengths(root, enc, suite, ftype, demos):
    """Frames per test demo in a condition root; must be n or n-1 (PLAN Q14)."""
    out, handles = [], {}
    try:
        for stem, d, n, _ in demos:
            if stem not in handles:
                p = Path(root) / enc / suite / (stem + ".h5")
                if not p.exists():
                    raise FileNotFoundError(f"missing condition feature file {p}")
                import h5py
                handles[stem] = h5py.File(p, "r")
            t = handles[stem][d][CAMS[0]][ftype].shape[0]
            if t not in (n, n - 1):
                raise ValueError(f"{root} {enc} {stem}/{d}: {t} frames, expected {n} or {n - 1}")
            out.append(t)
    finally:
        for h in handles.values():
            h.close()
    return out


def load_condition(root, enc, suite, ftype, demos, lengths):
    """{cam: (sum lengths, D)} test features of one encoder in one condition, demos in test order."""
    import h5py
    out, handles = {c: [] for c in CAMS}, {}
    try:
        for (stem, d, n, _), t in zip(demos, lengths):
            if stem not in handles:
                p = Path(root) / enc / suite / (stem + ".h5")
                if not p.exists():
                    raise FileNotFoundError(f"missing condition feature file {p}")
                handles[stem] = h5py.File(p, "r")
            for c in CAMS:
                x = handles[stem][d][c][ftype][:]
                if len(x) != t:
                    raise ValueError(f"{root} {enc} {stem}/{d}/{c}: {len(x)} frames, expected {t}")
                out[c].append(x)
    finally:
        for h in handles.values():
            h.close()
    return {c: np.concatenate(v).astype(np.float32) for c, v in out.items()}


# ---------------------------------------------------------------------------
# Standardization and PCA (clean-train only)
# ---------------------------------------------------------------------------

def fit_pca(z, k):
    """Top-k principal axes of z (rows = frames). Returns (mean, axes (d, k), explained-variance ratio of the kept axes)."""
    n, d = z.shape
    pm = z.mean(0, dtype=np.float64)
    cov = np.zeros((d, d))
    for i in range(0, n, 8192):
        c = z[i:i + 8192].astype(np.float64) - pm
        cov += c.T @ c
    cov /= max(n - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:k]
    return pm.astype(np.float32), v[:, order].astype(np.float32), float(w[order].sum() / max(w.sum(), 1e-12))


class CamTransform:
    """Per-camera map: standardize with clean-train statistics, optionally project on clean-train PCA axes."""

    def __init__(self, x_train, reduce_to=None):
        self.mu = x_train.mean(0)
        self.sd = x_train.std(0) + 1e-6          # same formula as diagnostics.standardize
        self.pca = None
        self.dim_in = x_train.shape[1]
        if reduce_to is not None and reduce_to < self.dim_in:
            pm, axes, evr = fit_pca(self.standardize(x_train), reduce_to)
            self.pca, self.evr = (pm, axes), evr
        self.dim_out = self.dim_in if self.pca is None else self.pca[1].shape[1]

    def standardize(self, x):
        return (x - self.mu) / self.sd

    def __call__(self, x):
        z = self.standardize(x)
        return z if self.pca is None else (z - self.pca[0]) @ self.pca[1]


def matched_plan(dims, declared=None):
    """Encoders larger than the smallest are reduced to the smallest (Q10 global, Q23 mean). dims: {enc: per-camera size}.

    declared: optional {enc: size} claimed by the caller (e.g. from the smoke test); must match the data.
    """
    if declared is not None:
        for e, s in declared.items():
            if e in dims and dims[e] != s:
                raise ValueError(f"declared size {e}={s} but the feature files have {dims[e]}")
    target = min(dims.values())
    return target, sorted(e for e, s in dims.items() if s > target)


def tally_alphas(results_dir, ftype):
    """{probe key: alpha} from the tally JSONs; the same key in several files must agree."""
    files = sorted(Path(results_dir).glob("tally-*.json"))
    if not files:
        raise FileNotFoundError(f"no tally-*.json in {results_dir}")
    out = {}
    for f in files:
        for k, v in json.loads(f.read_text())["probes"].items():
            if k.endswith("|" + ftype):
                if k in out and out[k] != v["alpha"]:
                    raise ValueError(f"alpha for {k} differs between tally files")
                out[k] = v["alpha"]
    return out


def tune_alpha(x, y, split):
    """Pick alpha on clean validation demos (train fit), exactly diagnostics.ridge_select without the test step."""
    tr, va = split == "train", split == "val"
    best = None
    for a in D.ALPHAS:
        w, b = D.ridge_fit(x[tr], y[tr], a)
        s = D.r2(y[va], x[va] @ w + b).mean()
        if best is None or s > best[0]:
            best = (s, a)
    return best[1]


# ---------------------------------------------------------------------------
# Bootstrap, quantities, rules
# ---------------------------------------------------------------------------

def make_weights(k, n_boot, seed):
    """Row 0 = all ones (the point estimate); rows 1.. = demo resampling counts. One call serves every probe."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)
    return np.vstack([np.ones((1, k)), w])


def stat(arr):
    """Point (row 0) and 95% percentile CI over the draws (rows 1..)."""
    return {"point": float(arr[0]), "ci95": D.ci(arr[1:])}


def quantities(st, W):
    """st[cond][probe] = (n, sy, syy, sres) per demo. Returns arrays of shape (len(W),) in dicts.

    probes are "D" and "D+<B>". Needs REF and all four TYPES.
    """
    for c in [REF] + TYPES:
        if c not in st:
            raise KeyError(f"condition {c} missing")
    r2 = {c: {p: D.r2_from_stats(W, *s) for p, s in ps.items()} for c, ps in st.items()}
    partners = [p[2:] for p in st[REF] if p.startswith("D+")]
    drop = {c: {p: r2[REF][p] - r2[c][p] for p in r2[c]} for c in TYPES}
    e = {c: {b: drop[c]["D+" + b] - drop[c]["D"] for b in partners} for c in TYPES}
    g = {c: {b: r2[c]["D+" + b] - r2[c]["D"] for b in partners} for c in TYPES}
    pe = {b: np.mean([e[c][b] for c in TYPES], axis=0) for b in partners}
    pg = {b: np.mean([g[c][b] for c in TYPES], axis=0) for b in partners}
    return {"r2": r2, "drop": drop, "e": e, "g": g, "pooled_e": pe, "pooled_g": pg, "partners": partners}


def rule_p(e, target="R"):
    """Rule P (decides). e: {partner: pooled e draws (row 0 = point)}.

    Fires if e(R) is strictly the largest point estimate AND the lower end of the 95% CI of
    e(R) - e(next-largest) is above 0. The next-largest is picked from the point estimates.
    """
    others = [b for b in e if b != target]
    nxt = max(others, key=lambda b: e[b][0])
    diff = e[target] - e[nxt]
    is_largest = bool(all(e[target][0] > e[b][0] for b in others))
    ci = D.ci(diff[1:])
    return {"R_is_largest": is_largest, "next_largest": nxt, "diff_point": float(diff[0]), "diff_ci95": ci,
            "diff_vs_each_other": {b: {"point": float((e[target] - e[b])[0]), "ci95": D.ci((e[target] - e[b])[1:])}
                                   for b in others},
            "fires": bool(is_largest and ci[0] > 0)}


def rule_b(g, target="R"):
    """Rule B (reported only). R's clause: g(R) < 0 with the upper end of its 95% CI below 0.

    The clause "the other three are not below 0" is reported three ways (Q21):
      point_reading: each other point estimate >= 0 (as written);
      ci_reading_strict: the lower end of each other 95% CI >= 0 (clearly not below 0);
      ci_reading_lenient: the upper end of each other 95% CI >= 0 (not clearly below 0).
    The user's "stricter CI-based reading" is taken to mean ci_reading_strict; the lenient one is added so no
    reading of the wording is lost.
    """
    others = [b for b in g if b != target]
    r_ci = D.ci(g[target][1:])
    r_clause = bool(g[target][0] < 0 and r_ci[1] < 0)
    oc = {b: D.ci(g[b][1:]) for b in others}
    point = all(g[b][0] >= 0 for b in others)
    strict = all(oc[b][0] >= 0 for b in others)
    lenient = all(oc[b][1] >= 0 for b in others)
    return {"R_gain_point": float(g[target][0]), "R_gain_ci95": r_ci, "R_clause": r_clause,
            "others": {b: {"point": float(g[b][0]), "ci95": oc[b]} for b in others},
            "others_point_reading": bool(point), "others_ci_strict": bool(strict), "others_ci_lenient": bool(lenient),
            "fires_point_reading": bool(r_clause and point), "fires_ci_reading_strict": bool(r_clause and strict),
            "fires_ci_reading_lenient": bool(r_clause and lenient)}


def ordering(e):
    """All partners by pooled e(B) (point), largest first, beside the paper's order. No significance claim."""
    ours = sorted(e, key=lambda b: -e[b][0])
    return {"ours_by_e_point": ours, "ours_e": {b: float(e[b][0]) for b in ours},
            "paper_order": [b for b in PAPER_ORDER if b in e],
            "paper_delta": {b: PAPER_DELTA[b] for b in PAPER_ORDER if b in e},
            "note": "ordering only; 4 partners, no significance claim"}


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def drift_parts(z_ref, z_c, groups, k):
    """Per-demo sums: sum ||z_c - z_ref||^2 and sum ||z_ref - mean(z_ref over test frames)||^2."""
    mu = z_ref.mean(0, dtype=np.float64)
    num = ((z_c.astype(np.float64) - z_ref) ** 2).sum(1)
    den = ((z_ref.astype(np.float64) - mu) ** 2).sum(1)
    return np.bincount(groups, weights=num, minlength=k), np.bincount(groups, weights=den, minlength=k)


def drift_draws(num, den, W):
    """Ratio of the mean over resampled frames: (W @ num) / (W @ den); frame counts cancel."""
    return (W @ num) / np.maximum(W @ den, 1e-12)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(feats_root, raw_dir, suite, cond_roots, tally_dir, out_dir, *args, **kw):
    """Run the whole analysis and write results.json, stats.npz (and preds.npz) under out_dir (must not exist).

    The dir is claimed up front; if the run fails the (partial) dir is removed so a corrected retry can reuse the path.
    Arguments after out_dir are those of _run.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _run(feats_root, raw_dir, suite, cond_roots, tally_dir, out_dir, *args, **kw)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _run(feats_root, raw_dir, suite, cond_roots, tally_dir, out_dir, ftype="global", n_boot=1000, seed=0,
         sizes=None, other_alpha="tune", save_preds=True, min_boot=MIN_BOOT, quiet=False):
    t0 = time.time()
    if n_boot < min_boot:
        raise ValueError(f"n_boot={n_boot} < {min_boot}: the plan needs at least {MIN_BOOT} draws")
    if other_alpha not in ("tune", "reuse"):
        raise ValueError("other_alpha must be 'tune' or 'reuse'")
    need = [REF] + TYPES
    missing = [c for c in need if c not in cond_roots]
    if missing:
        raise KeyError(f"cond_roots missing {missing}")
    encoders = [DINO] + list(PARTNERS.values())
    alphas = tally_alphas(tally_dir, ftype)
    pair_key = lambda b: f"{min(DINO, PARTNERS[b])}+{max(DINO, PARTNERS[b])}|{ftype}"
    for key in [f"{DINO}|{ftype}"] + [pair_key(b) for b in PARTNERS]:
        if key not in alphas:
            raise KeyError(f"tally alpha for {key} not found in {tally_dir}")
    log = (lambda *a: None) if quiet else (lambda *a: print("[pf]", *a, flush=True))

    feats, actions, _, split, index, meta, files = load_clean(feats_root, raw_dir, suite, encoders, ftype)
    train, trva = split == "train", split != "test"
    demos = layout_test_demos(index, split)
    lengths = cond_demo_lengths(cond_roots[REF], DINO, suite, ftype, demos)
    test_sel = np.concatenate([s + np.arange(t) for (_, _, _, s), t in zip(demos, lengths)])
    groups = np.repeat(np.arange(len(demos)), lengths)
    k = len(demos)
    y_test = actions[test_sel]
    log(f"{k} test demos, {len(test_sel)} frames per condition")

    # Condition features, test frames only: cond[c][enc][cam]; the stored clean ones come from the clean load.
    conds = [STORED, REF] + TYPES
    cond = {STORED: {e: {c: feats[e][ftype][c][test_sel] for c in CAMS} for e in encoders}}
    for c in [REF] + TYPES:
        cond[c] = {}
        for e in encoders:
            if c != REF or e != DINO:
                if cond_demo_lengths(cond_roots[c], e, suite, ftype, demos) != lengths:
                    raise ValueError(f"condition {c} / {e}: frame counts differ from the clean re-render")
            cond[c][e] = load_condition(cond_roots[c], e, suite, ftype, demos, lengths)
    dims = {}
    for e in encoders:
        ds = {feats[e][ftype][c].shape[1] for c in CAMS}
        if len(ds) != 1:
            raise ValueError(f"{e}: cameras have different feature sizes {ds}")
        dims[e] = ds.pop()
    target, reduced = matched_plan(dims, sizes)
    log(f"per-camera sizes {dims}; matched target {target}, reduced {reduced}")

    W = make_weights(k, n_boot, seed)
    results = {"suite": suite, "ftype": ftype, "files": files, "encoders": {e: meta[e] for e in encoders},
               "git_commit": D.git_commit(), "seed": seed, "n_boot": n_boot, "n_test_demos": k,
               "n_test_frames": int(len(test_sel)), "frames_per_demo_dropped_last": [int(n - t) for (_, _, n, _), t in zip(demos, lengths)],
               "tally_dir": str(tally_dir), "tally_alphas": alphas, "cond_roots": {c: str(v) for c, v in cond_roots.items()},
               "per_camera_sizes": dims, "matched": {"target": target, "reduced": reduced, "declared_sizes": sizes},
               "other_alpha": other_alpha, "alphas_grid": D.ALPHAS, "configs": {}, "drift": {}}
    stats_out = {"test_demos": np.array([f"{s}/{d}" for s, d, _, _ in demos]), "frames_per_demo": np.array(lengths),
                 "boot_weights": W.astype(np.int16)}
    preds_out = {"y_test": y_test.astype(np.float32)}

    configs = {"full": []}
    if reduced:
        configs["matched"] = reduced
    else:
        results["matched"]["note"] = "all encoders already have the smallest size; the matched row would equal the full row and is skipped"

    for config, red in configs.items():
        results["configs"][config] = {}
        for view, cams in VIEWS.items():
            def inputs(enc, reduce_enc):
                """Probe inputs for one encoder: clean all-frames matrix and each condition's test matrix."""
                tfs = {c: CamTransform(feats[enc][ftype][c][train], target if reduce_enc else None) for c in cams}
                allx = np.concatenate([tfs[c](feats[enc][ftype][c]) for c in cams], 1)
                tests = {cn: np.concatenate([tfs[c](cond[cn][enc][c]) for c in cams], 1) for cn in conds}
                return allx, tests, sum(tf.dim_out for tf in tfs.values()), tfs

            probe_info, st = {}, {c: {} for c in conds}
            xd = inputs(DINO, DINO in red)
            for b in [None] + list(PARTNERS):
                name = "D" if b is None else "D+" + b
                if b is None:
                    allx, tests, dim = xd[0], xd[1], xd[2]
                    in_red = DINO in red
                    tkey = f"{DINO}|{ftype}"
                else:
                    xb = inputs(PARTNERS[b], PARTNERS[b] in red)
                    allx = np.concatenate([xd[0], xb[0]], 1)
                    tests = {cn: np.concatenate([xd[1][cn], xb[1][cn]], 1) for cn in conds}
                    dim = xd[2] + xb[2]
                    in_red = (DINO in red) or (PARTNERS[b] in red)
                    tkey = pair_key(b)
                    del xb
                same_as_tally = view == "combined" and not in_red
                if same_as_tally or other_alpha == "reuse":
                    alpha, src = alphas[tkey], ("tally" if same_as_tally else "tally-reused-on-changed-input")
                else:
                    alpha, src = tune_alpha(allx, actions, split), "tuned-on-clean-val"
                w, bias = D.ridge_fit(allx[trva], actions[trva], alpha)
                probe_info[name] = {"dim": int(dim), "alpha": alpha, "alpha_source": src}
                for cn in conds:
                    yhat = tests[cn] @ w + bias
                    st[cn][name] = D.demo_stats(y_test, yhat, groups)
                    if save_preds:
                        preds_out[f"{config}/{view}/{cn}/{name}"] = yhat.astype(np.float32)
                log(f"{config}/{view}/{name}: dim {dim}, alpha {alpha} ({src}), "
                    f"R2 rerender {float(D.r2_from_stats(W[:1], *st[REF][name])[0]):.3f}")
                del allx, tests
            for cn, ps in st.items():
                for p, (n_, sy, syy, sres) in ps.items():
                    for nm, arr in zip(("n", "sy", "syy", "sres"), (n_, sy, syy, sres)):
                        stats_out[f"{config}/{view}/{cn}/{p}/{nm}"] = arr

            q = quantities({c: st[c] for c in conds}, W)
            res = {"probes": probe_info,
                   "r2": {c: {p: stat(a) for p, a in ps.items()} for c, ps in q["r2"].items()},
                   "drop": {c: {p: stat(a) for p, a in ps.items()} for c, ps in q["drop"].items()},
                   "e": {c: {b: stat(a) for b, a in bs.items()} for c, bs in q["e"].items()},
                   "gain": {c: {b: stat(a) for b, a in bs.items()} for c, bs in q["g"].items()},
                   "pooled_e": {b: stat(a) for b, a in q["pooled_e"].items()},
                   "pooled_g": {b: stat(a) for b, a in q["pooled_g"].items()},
                   "rule_P": rule_p(q["pooled_e"]), "rule_B": rule_b(q["pooled_g"]),
                   "ordering": ordering(q["pooled_e"])}
            results["configs"][config][view] = res

    # Drift, single encoder, full-size standardized features (no PCA), clean re-render as the reference.
    for view, cams in VIEWS.items():
        results["drift"][view] = {}
        for e in encoders:
            tfs = {c: CamTransform(feats[e][ftype][c][train]) for c in cams}
            z = {cn: np.concatenate([tfs[c].standardize(cond[cn][e][c]) for c in cams], 1) for cn in conds}
            row = {}
            for cn in TYPES + [STORED]:
                num, den = drift_parts(z[REF], z[cn], groups, k)
                stats_out[f"drift/{view}/{e}/{cn}/num"], stats_out[f"drift/{view}/{e}/{cn}/den"] = num, den
                row[cn] = stat(drift_draws(num, den, W))
            # Pipeline reference: how far the clean re-render sits from the stored clean frame (reference = stored).
            num, den = drift_parts(z[STORED], z[REF], groups, k)
            stats_out[f"drift/{view}/{e}/rerender_vs_stored/num"], stats_out[f"drift/{view}/{e}/rerender_vs_stored/den"] = num, den
            row["rerender_vs_stored"] = stat(drift_draws(num, den, W))
            results["drift"][view][e] = row
    results["drift_note"] = ("drift_c(E) = mean ||f(x')-f(x)||^2 / mean ||f(x)-mean(f)||^2 over test frames, x = clean re-render, "
                             "mean(f) = mean over the test frames, clean-train-standardized features; entries for the stored "
                             "condition use the stored frame as x'; rerender_vs_stored uses the stored frame as x.")

    results["seconds"] = time.time() - t0
    np.savez_compressed(out_dir / "stats.npz", **stats_out)
    if save_preds:
        np.savez_compressed(out_dir / "preds.npz", **preds_out)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {out_dir} in {results['seconds']:.0f}s")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--feats", required=True, help="clean feature root (extract_features.py --out)")
    p.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files (actions, split)")
    p.add_argument("--suite", required=True)
    p.add_argument("--cond", action="append", required=True, metavar="NAME=ROOT",
                   help=f"feature root per condition; names: {REF} {' '.join(TYPES)} (repeat)")
    p.add_argument("--tally-results", required=True, help="dir with the tally-*.json result files")
    p.add_argument("--out", required=True, help="new output dir (refused if it exists)")
    p.add_argument("--ftype", default="global", choices=["global", "mean"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sizes", default=None, help="declared per-camera sizes enc=dim,... (checked against the files)")
    p.add_argument("--other-alpha", default="tune", choices=["tune", "reuse"],
                   help="alpha for probes whose input is not identical to a tally probe (matched control, single-camera views)")
    p.add_argument("--no-save-preds", action="store_true")
    a = p.parse_args()
    roots = dict(s.split("=", 1) for s in a.cond)
    sizes = {k: int(v) for k, v in (s.split("=") for s in a.sizes.split(","))} if a.sizes else None
    run(a.feats, a.raw, a.suite, roots, a.tally_results, a.out, a.ftype, a.n_boot, a.seed, sizes, a.other_alpha,
        not a.no_save_preds)


if __name__ == "__main__":
    main()
