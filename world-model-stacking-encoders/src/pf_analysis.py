"""Analysis for the perturbed-frame experiment (experiments/2026-11-06-perturbed-frames/PLAN.md,
Amendment 2026-11-09: every reader is trained on our own clean re-rendered frames).

Plain idea: train an action probe (a ridge "reader") on clean frames that we re-rendered ourselves, then see
how much it loses when the same demo moments are re-rendered under four picture changes (camera, light,
background, noise). The question is whether adding C-RADIOv4 (R) to DINOv3 (D) loses more than adding
SigLIP 2 (L), SAM 3 (S) or VGGT-Omega (G). Nothing is trained on perturbed data, and nothing is trained on the
stored LIBERO frames.

Two entries (subcommands), gate-first:
  dev    Phase A. Reads re-rendered clean TRAIN and VAL features only (a code-level guard raises if a test
         demo is opened). Tunes the ridge alpha of every probe on re-rendered val, writes a JSON report with
         the val R2 of every probe and the clean val gains gain(B|D), and runs the gate the user chose
         (Amendment 2026-11-09 section 4, check A): the val R2 (combined view, global features) of the
         train-only DINOv3 fit at its best alpha must be at least GATE_REF - GATE_MARGIN, one-sided.
  final  Phase B. Refuses to start unless it is given a dev report whose gate passed and whose train/val root
         matches the one given. Then:
  1. Refit the probes from re-rendered train+val features: D alone and D+B for B in R, L, S, G (concatenated,
     standardized with re-rendered-train statistics). Alpha is tuned on re-rendered val for EVERY probe (the
     tally's grid and procedure, tune_alpha), then the probe is refit on train+val ["Main measure: the probe",
     step 1, as amended]. Alpha and standardization are never re-tuned on perturbed data. No tally alpha is
     used anywhere on this path.
  2. Apply each fitted probe unchanged to the test features of every condition (clean re-render, camera,
     light, background, noise). Per-demo sufficient statistics (and, by default, the predictions) are saved so
     every rule and the bootstrap can be recomputed later without refitting. No stored frame is read for test.
  3. R2_c(X), drop_c(X) = R2_rerender(X) - R2_c(X), e_c(B) = drop_c(D+B) - drop_c(D),
     gain_c(B|D) = R2_c(D+B) - R2_c(D); pooled e(B), g(B) = equal-weight mean over the four types.
  4. One demo-level paired bootstrap (the same draws for every probe, condition and view), seed recorded.
  5. Rule P (decides), rule B (reported only, three readings of its last clause), the ordering by e(B).
  6. Feature drift per encoder ["Secondary measure: feature drift"].
  7. Matched-dimension control: PCA on re-rendered clean-train frames only (unwhitened), applied unchanged to
     every condition; only encoders larger than the smallest are reduced (global: C-RADIOv4 and VGGT-Omega to
     1,024 per camera). Reported as a separate "matched" config next to the "full" one.
  8. Every number is computed for three views: combined (both cameras), agentview, wrist (Q12).

Feature layout (same as diagnostics.py): <root>/<encoder>/<suite>/<file stem>.h5, with
<demo_k>/<cam>/global|mean of shape (T, D). Phase A (train+val) and phase B (clean test) sit in separate roots;
the loader merges per demo. Every re-rendered demo holds exactly T = n-1 frames (PLAN Q14: image k pairs with
state k+1, the last frame is dropped); actions[:T] are the labels. The split is computed from ALL demo names of
the raw file and must give val = demo_30..demo_39 and train = demo_0..demo_29 before anything is read
(diagnostics.load would silently give a 24/8/8 split on a file holding only the 40 train+val demos).
Expected frames per camera: 38,164 train, 12,316 val, 12,748 test.

Optional, report-only references (never gating, skipped with a note when their input is absent): the
frame-alignment (lag) check and the cross value against a stored-feature root (train and val demos only, never
test), and the tally's val gains beside ours.

Never overwrites: the output dir must not exist.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import diagnostics as D

CAMS = D.CAMS
REF = "clean_rerender"      # reference for every drop and every drift
TYPES = ["camera", "light", "background", "noise"]
DINO = "dinov3"
PARTNERS = {"R": "c-radiov4", "L": "siglip2", "S": "sam3", "G": "vggt-omega"}
# Paper's pair deltas (D+B vs D, closed-loop success; PLAN "The stands out rule"): reported beside ours, no test.
PAPER_DELTA = {"L": 1.83, "S": 0.49, "G": 0.40, "R": -5.41}
PAPER_ORDER = sorted(PAPER_DELTA, key=lambda b: -PAPER_DELTA[b])
LAST_PROBE = "D+" + list(PARTNERS)[-1]     # iter_probes yields D, then D+B for B in PARTNERS order
VIEWS = {"combined": list(CAMS), "agentview": [CAMS[0]], "wrist": [CAMS[1]]}
MIN_BOOT = 1000

# Gate (Amendment 2026-11-09 section 4, check A; fixed before any re-render-trained result existed).
GATE_REF = 0.6562       # experiments/2026-11-06-perturbed-frames/results/pf-val/20261108-181737/probe-report.json,
                        # val_r2.combined.val_r2_stored: stored-trained DINOv3 probe, alpha 0.1, stored val frames
GATE_MARGIN = 0.025     # user decision 2026-11-09 (AskUserQuestion label "0.025 (Recommended)")
# Expected re-rendered frames per camera (n-1 per demo; PLAN Amendment 2026-11-09 section 1).
EXPECTED_FRAMES = {"train": 38164, "val": 12316, "test": 12748}
TRAIN_DEMOS = [f"demo_{i}" for i in range(30)]
VAL_DEMOS = [f"demo_{i}" for i in range(30, 40)]
DEV_PHASES = ("train", "val")   # the development entry reads these and nothing else
DEV_KIND = "pf-dev-report"


class NoTestAccessError(RuntimeError):
    """A test demo was about to be opened where only train and val may be read."""


class GateError(RuntimeError):
    """The final run was started without a passing dev report that matches the given roots."""


def gate_threshold():
    return GATE_REF - GATE_MARGIN


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

META_KEYS = ["encoder", "hf_id", "resolution", "rotate180", "dtype", "grid", "frame_stride", "cams"]
PCA_WHITEN = False   # single switch: False keeps PCA scores unwhitened; True divides each score by its axis's std


def assert_finite(x, label):
    if not np.all(np.isfinite(x)):
        raise ValueError(f"non-finite values in {label}")


def _dnum(name):
    return int(name.split("_")[-1])


def file_split(names, label):
    """Split from ALL demo names of one raw file; val must be demo_30..39 and train demo_0..29, else raise.

    Returns (names sorted by demo index, {demo: 'train'|'val'|'test'}). Nothing is read from any feature file here.
    """
    names = sorted(names, key=_dnum)
    sp = D.demo_split(names)
    val = [d for d in names if sp[d] == "val"]
    train = [d for d in names if sp[d] == "train"]
    if val != VAL_DEMOS or train != TRAIN_DEMOS:
        raise ValueError(f"{label}: {len(names)} demos in the raw file give {len(train)} train demos "
                         f"({train[:1] + train[-1:]}) and val demos {val}; expected train demo_0..demo_29 and "
                         f"val demo_30..demo_39 (50 demos per file)")
    return names, sp


def check_meta(f, clean_meta, label):
    """The condition file's attrs['meta'] must match the clean cache on every key in META_KEYS (missing = fail)."""
    if "meta" not in f.attrs:
        raise ValueError(f"{label}: no attrs['meta']")
    m = json.loads(f.attrs["meta"])
    for k in META_KEYS:
        if k not in m or k not in clean_meta:
            raise ValueError(f"{label}: meta key '{k}' missing (condition has it: {k in m}, clean cache has it: {k in clean_meta})")
        if m[k] != clean_meta[k]:
            raise ValueError(f"{label}: meta '{k}' is {m[k]!r}, the clean cache has {clean_meta[k]!r}")


def _read_meta(f, label):
    if "meta" not in f.attrs:
        raise ValueError(f"{label}: no attrs['meta']")
    return json.loads(f.attrs["meta"])


def load_rendered(root, raw_dir, suite, encoders, ftypes, phases=DEV_PHASES, allow_test=False,
                  expected_frames=EXPECTED_FRAMES):
    """Re-rendered clean features of the demos in `phases`, with labels actions[:n-1] from the raw HDF5.

    Order of work: (1) the split of every raw file is computed from all its demo names and checked
    (val = demo_30..39, train = demo_0..29); (2) only then are features and labels read, and only for demos in
    `phases`. A test demo is never opened unless allow_test=True (the development entry never sets it).
    Every feature file must hold exactly n-1 frames per demo and its meta must match the first file's.

    Returns a namespace: feats[enc][ftype][cam] (rows in file order, demo order), actions, split (per frame),
    index [(stem, demo, T, part)], meta[enc], files, counts {part: frames per camera}.
    """
    import h5py
    phases = tuple(phases)
    ftypes = tuple(ftypes)
    unknown = set(phases) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown phases {sorted(unknown)}")
    if "test" in phases and not allow_test:
        raise NoTestAccessError("phase 'test' requested but test access is not allowed here")
    raw_files = sorted(Path(raw_dir).glob("*.hdf5"))
    assert raw_files, f"no LIBERO files in {raw_dir}"
    plans = []
    for raw in raw_files:                      # pass 1: names only
        with h5py.File(raw, "r") as fr:
            names, sp = file_split(list(fr["data"].keys()), raw.name)
        plans.append((raw, names, sp))
    feats = {e: {t: {c: [] for c in CAMS} for t in ftypes} for e in encoders}
    actions, split, index, meta = [], [], [], {}
    for raw, names, sp in plans:               # pass 2: features and labels of the allowed demos
        fe = {}
        for e in encoders:
            p = Path(root) / e / suite / (raw.stem + ".h5")
            if not p.exists():
                raise FileNotFoundError(f"missing re-rendered feature file {p}")
            fe[e] = h5py.File(p, "r")
        try:
            for e, f in fe.items():
                m = _read_meta(f, f"{root}/{e}/{raw.stem}")
                meta.setdefault(e, m)
                check_meta(f, meta[e], f"{root}/{e}/{raw.stem}")
            with h5py.File(raw, "r") as fr:
                for d in names:
                    if sp[d] not in phases:
                        continue
                    if sp[d] == "test" and not allow_test:
                        raise NoTestAccessError(f"{raw.name} {d}: refusing to open a test demo")
                    a = fr["data"][d]["actions"][:]
                    t = len(a) - 1
                    if t < 1:
                        raise ValueError(f"{raw.name} {d}: {len(a)} actions")
                    for e, f in fe.items():
                        if d not in f:
                            raise KeyError(f"{root}/{e}/{raw.stem}.h5 has no {d}")
                        for ft in ftypes:
                            for c in CAMS:
                                x = f[d][c][ft][:]
                                if len(x) != t:
                                    raise ValueError(f"{root} {e} {raw.stem}/{d}/{c}: {len(x)} frames, expected exactly n-1 = {t}")
                                feats[e][ft][c].append(x)
                    actions.append(a[:t])
                    split += [sp[d]] * t
                    index.append((raw.stem, d, t, sp[d]))
        finally:
            for f in fe.values():
                f.close()
    cat = lambda xs: np.concatenate(xs).astype(np.float32)
    feats = {e: {t: {c: cat(v) for c, v in tv.items()} for t, tv in ev.items()} for e, ev in feats.items()}
    actions, split = cat(actions), np.array(split)
    assert_finite(actions, "re-rendered-demo actions")
    for e in feats:
        for ft in ftypes:
            for c in CAMS:
                assert_finite(feats[e][ft][c], f"re-rendered features {root}/{e}/{c}/{ft}")
    counts = {p: int((split == p).sum()) for p in phases}
    if expected_frames is not None:
        for p in phases:
            if counts[p] != expected_frames[p]:
                raise ValueError(f"{p}: {counts[p]} re-rendered frames per camera, expected {expected_frames[p]}")
    return types.SimpleNamespace(feats=feats, actions=actions, split=split, index=index, meta=meta,
                                 files=[r.name for r in raw_files], counts=counts)


def load_test_layout(raw_dir, expected_frames=EXPECTED_FRAMES):
    """FINAL RUN ONLY. Test demos [(stem, demo, n, 0)] in frame order and their labels actions[:n-1].

    Reads the test demos' actions from the raw files (the labels of the final scoring); the dev entry never calls this.
    """
    import h5py
    demos, ys = [], []
    for raw in sorted(Path(raw_dir).glob("*.hdf5")):
        with h5py.File(raw, "r") as fr:
            names, sp = file_split(list(fr["data"].keys()), raw.name)
            for d in names:
                if sp[d] == "test":
                    a = fr["data"][d]["actions"][:]
                    demos.append((raw.stem, d, len(a), 0))
                    ys.append(a[:-1])
    assert demos, f"no LIBERO files / test demos in {raw_dir}"
    y = np.concatenate(ys).astype(np.float32)
    assert_finite(y, "test actions")
    if expected_frames is not None and len(y) != expected_frames["test"]:
        raise ValueError(f"test: {len(y)} frames per camera, expected {expected_frames['test']}")
    return demos, y


def _open_cond(handles, root, enc, suite, stem, clean_meta):
    if stem not in handles:
        import h5py
        p = Path(root) / enc / suite / (stem + ".h5")
        if not p.exists():
            raise FileNotFoundError(f"missing condition feature file {p}")
        handles[stem] = h5py.File(p, "r")
        check_meta(handles[stem], clean_meta, str(p))
    return handles[stem]


def cond_demo_lengths(root, enc, suite, ftype, demos, clean_meta):
    """Frames per test demo in a condition root; must be exactly n-1 (PLAN Q14: last frame dropped)."""
    out, handles = [], {}
    try:
        for stem, d, n, _ in demos:
            f = _open_cond(handles, root, enc, suite, stem, clean_meta)
            t = f[d][CAMS[0]][ftype].shape[0]
            if t != n - 1:
                raise ValueError(f"{root} {enc} {stem}/{d}: {t} frames, expected exactly n-1 = {n - 1}")
            out.append(t)
    finally:
        for h in handles.values():
            h.close()
    return out


def load_condition(root, enc, suite, ftype, demos, lengths, clean_meta):
    """{cam: (sum lengths, D)} test features of one encoder in one condition, demos in test order."""
    out, handles = {c: [] for c in CAMS}, {}
    try:
        for (stem, d, n, _), t in zip(demos, lengths):
            f = _open_cond(handles, root, enc, suite, stem, clean_meta)
            for c in CAMS:
                x = f[d][c][ftype][:]
                if len(x) != t:
                    raise ValueError(f"{root} {enc} {stem}/{d}/{c}: {len(x)} frames, expected {t}")
                out[c].append(x)
    finally:
        for h in handles.values():
            h.close()
    res = {c: np.concatenate(v).astype(np.float32) for c, v in out.items()}
    for c in CAMS:
        assert_finite(res[c], f"{root} {enc}/{c}/{ftype}")
    return res


def check_lag(z_rr, z_st, label):
    """Frame alignment for one demo and camera. z_rr: (T, d) re-render, z_st: (T+1, d) stored, both standardized.

    The mean distance between re-render frame j and stored frame j must be smaller than at lag -1 (stored j-1)
    and at lag +1 (stored j+1); otherwise the two are off by a frame.
    """
    t = len(z_rr)
    dist = lambda a, b: float(np.linalg.norm(a.astype(np.float64) - b, axis=1).mean())
    d0, dm, dp = dist(z_rr, z_st[:t]), dist(z_rr[1:], z_st[:t - 1]), dist(z_rr, z_st[1:t + 1])
    if not (d0 < dm and d0 < dp):
        raise ValueError(f"{label}: frame alignment check failed (mean distance lag 0 {d0:.4g}, lag -1 {dm:.4g}, lag +1 {dp:.4g})")


# ---------------------------------------------------------------------------
# Optional reference path: stored LIBERO features (train and val demos only, never test) and the tally
# ---------------------------------------------------------------------------

def load_stored(stored_root, suite, enc, ftype, index, clean_meta):
    """Stored features (n frames per demo) of the demos in `index` ([(stem, demo, T, part)]); never a test demo.

    Returns {cam: [array (T+1, d) per demo]}. Reference use only (lag check, cross value).
    """
    import h5py
    out, handles = {c: [] for c in CAMS}, {}
    try:
        for stem, d, t, part in index:
            if part not in ("train", "val"):
                raise NoTestAccessError(f"{stem}/{d}: refusing to read stored features of a {part} demo")
            if stem not in handles:
                p = Path(stored_root) / enc / suite / (stem + ".h5")
                if not p.exists():
                    raise FileNotFoundError(f"missing stored feature file {p}")
                handles[stem] = h5py.File(p, "r")
                check_meta(handles[stem], clean_meta, str(p))
            for c in CAMS:
                x = handles[stem][d][c][ftype][:]
                if len(x) != t + 1:
                    raise ValueError(f"{stored_root} {enc} {stem}/{d}/{c}: {len(x)} stored frames, expected n = {t + 1}")
                assert_finite(x, f"stored {enc} {stem}/{d}/{c}")
                out[c].append(x.astype(np.float32))
    finally:
        for h in handles.values():
            h.close()
    return out


def lag_check_stored(tr, stored_root, suite, ftype, whiten):
    """Lag check of the re-rendered DINOv3 features against the stored ones, train and val demos only."""
    st = load_stored(stored_root, suite, DINO, ftype, tr.index, tr.meta[DINO])
    off = np.concatenate([[0], np.cumsum([t for _, _, t, _ in tr.index])])
    train = tr.split == "train"
    for cam in CAMS:
        tf = CamTransform(tr.feats[DINO][ftype][cam][train], whiten=whiten)
        for i, (stem, d, t, part) in enumerate(tr.index):
            check_lag(tf.standardize(tr.feats[DINO][ftype][cam][off[i]:off[i + 1]]), tf.standardize(st[cam][i]),
                      f"{stem}/{d}/{cam}")
    return {"demos_checked": len(tr.index), "parts": sorted({p for *_, p in tr.index})}


def reference_tally_val_gains(results_dir, suite, ftype):
    """Optional reference only. The tally's val gains gain(B|D) = val R2(D+B) - val R2(D), each at its own tuned alpha.

    Read from the tally JSONs' val_r2_by_alpha (train-only fits, stored frames). Returns {partner: gain}.
    """
    val = {}
    for f in sorted(Path(results_dir).glob("tally-*.json")):
        j = json.loads(f.read_text())
        if j.get("suite") != suite:
            continue
        for k, v in j["probes"].items():
            if not k.endswith("|" + ftype) or "val_r2_by_alpha" not in v:
                continue
            at = [s for a_, s in ((float(a), s) for a, s in v["val_r2_by_alpha"].items()) if np.isclose(a_, v["alpha"])]
            if len(at) != 1:
                raise ValueError(f"{f.name}: {k}: alpha {v['alpha']} not found once in val_r2_by_alpha")
            if k in val and val[k] != at[0]:
                raise ValueError(f"val R2 for {k} differs between tally files")
            val[k] = at[0]
    base = val.get(f"{DINO}|{ftype}")
    if base is None:
        raise KeyError(f"no tally val curve for {DINO}|{ftype} (suite {suite}) in {results_dir}")
    out = {}
    for b, enc in PARTNERS.items():
        key = f"{min(DINO, enc)}+{max(DINO, enc)}|{ftype}"
        if key in val:
            out[b] = float(val[key] - base)
    return out


# ---------------------------------------------------------------------------
# Standardization and PCA (re-rendered clean-train only)
# ---------------------------------------------------------------------------

def fit_pca(z, k):
    """Top-k principal axes of z (rows = frames). Returns (mean, axes (d, k), kept eigenvalues, explained-variance ratio)."""
    n, d = z.shape
    pm = z.mean(0, dtype=np.float64)
    cov = np.zeros((d, d))
    for i in range(0, n, 8192):
        c = z[i:i + 8192].astype(np.float64) - pm
        cov += c.T @ c
    cov /= max(n - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:k]
    return pm.astype(np.float32), v[:, order].astype(np.float32), w[order], float(w[order].sum() / max(w.sum(), 1e-12))


class CamTransform:
    """Per-camera map: standardize with clean-train statistics, optionally project on clean-train PCA axes."""

    def __init__(self, x_train, reduce_to=None, whiten=PCA_WHITEN):
        self.mu = x_train.mean(0)
        self.sd = x_train.std(0) + 1e-6          # same formula as diagnostics.standardize
        self.pca = None
        self.evr = None
        self.dim_in = x_train.shape[1]
        if reduce_to is not None and reduce_to < self.dim_in:
            pm, axes, ev, self.evr = fit_pca(self.standardize(x_train), reduce_to)
            scale = (1.0 / np.sqrt(np.maximum(ev, 1e-12))).astype(np.float32) if whiten else np.ones(len(ev), np.float32)
            self.pca = (pm, axes * scale)
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
            if e not in dims:
                raise ValueError(f"declared size for unknown encoder '{e}' (known: {sorted(dims)})")
            if dims[e] != s:
                raise ValueError(f"declared size {e}={s} but the feature files have {dims[e]}")
    target = min(dims.values())
    return target, sorted(e for e, s in dims.items() if s > target)


def tune_alpha(x, y, split):
    """Pick alpha on clean validation demos (train fit), as diagnostics.ridge_select does without the test step.

    Returns (alpha, {alpha: val R2}). The train gram matrix is formed once; the solve matches ridge_fit.
    The score at the returned alpha is the val R2 of the train-only fit (the gate's scored value).
    """
    tr, va = split == "train", split == "val"
    xt, yt = x[tr], y[tr]
    gram, rhs = xt.T @ xt, xt.T @ (yt - yt.mean(0))
    best, curve = None, {}
    for a in D.ALPHAS:
        w = np.linalg.solve(gram + a * len(xt) * np.eye(x.shape[1], dtype=gram.dtype), rhs)
        s = float(D.r2(y[va], x[va] @ w + yt.mean(0)).mean())
        curve[a] = s
        if best is None or s > best[0]:
            best = (s, a)
    return best[1], curve


def config_plan(reduced):
    """Configs to run: 'full' always, 'matched' (the reduced encoders) only if some encoder is larger than the smallest."""
    configs = {"full": []}
    if reduced:
        configs["matched"] = reduced
    return configs


def iter_probes(feats, ftype, split, target, reduced, whiten):
    """Yield (config, view, name, x, parts, pca_log) for D and D+B probes over train+val rows.

    x: probe inputs on the train+val rows (standardized, optionally PCA-reduced; statistics and axes from the
    train rows only). parts: [(encoder, {cam: CamTransform})], to transform other data unchanged.
    pca_log is one dict per (config, view), filled as encoders are transformed.
    """
    train = split == "train"
    for config, red in config_plan(reduced).items():
        for view, cams in VIEWS.items():
            pca_log = {}

            def inputs(enc):
                tfs = {c: CamTransform(feats[enc][ftype][c][train], target if enc in red else None, whiten) for c in cams}
                for c, tf in tfs.items():
                    if tf.pca is not None:
                        pca_log.setdefault(enc, {})[c] = {"dim_in": tf.dim_in, "dim_out": tf.dim_out,
                                                          "explained_variance_ratio": tf.evr}
                return np.concatenate([tfs[c](feats[enc][ftype][c]) for c in cams], 1), tfs

            xd, tfd = inputs(DINO)
            for b in [None] + list(PARTNERS):
                if b is None:
                    yield config, view, "D", xd, [(DINO, tfd)], pca_log
                else:
                    xb, tfb = inputs(PARTNERS[b])
                    yield config, view, "D+" + b, np.concatenate([xd, xb], 1), [(DINO, tfd), (PARTNERS[b], tfb)], pca_log
                    del xb


def per_camera_sizes(feats, ftype):
    dims = {}
    for e in feats:
        ds = {feats[e][ftype][c].shape[1] for c in CAMS}
        if len(ds) != 1:
            raise ValueError(f"{e}: cameras have different feature sizes {ds}")
        dims[e] = ds.pop()
    return dims


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
    e(R) - e(next-largest) is above 0. The next-largest is picked from the point estimates; if several partners
    tie exactly, the one giving the lowest CI lower bound is used (the conservative choice; name breaks a further
    tie), so the result does not depend on dict order.
    """
    others = [b for b in e if b != target]
    top = max(e[b][0] for b in others)
    nxt = min((b for b in others if e[b][0] == top), key=lambda b: (D.ci((e[target] - e[b])[1:])[0], b))
    diff = e[target] - e[nxt]
    is_largest = bool(all(e[target][0] > e[b][0] for b in others))
    ci = D.ci(diff[1:])
    return {"R_is_largest": is_largest, "next_largest": nxt, "diff_point": float(diff[0]), "diff_ci95": ci,
            "diff_vs_each_other": {b: {"point": float((e[target] - e[b])[0]), "ci95": D.ci((e[target] - e[b])[1:])}
                                   for b in others},
            "fires": bool(is_largest and ci[0] > 0)}


def rule_b(g, target="R"):
    """Rule B (reported only). R's clause: g(R) < 0 with the upper end of its 95% CI below 0.

    The clause "the other three are not below 0" is reported two ways (Q21, decided by the user):
      point_reading: each other point estimate >= 0 (as written);
      ci_reading_strict: the lower end of each other 95% CI >= 0 (the stricter CI-based reading).
    """
    others = [b for b in g if b != target]
    r_ci = D.ci(g[target][1:])
    r_clause = bool(g[target][0] < 0 and r_ci[1] < 0)
    oc = {b: D.ci(g[b][1:]) for b in others}
    point = all(g[b][0] >= 0 for b in others)
    strict = all(oc[b][0] >= 0 for b in others)
    return {"R_gain_point": float(g[target][0]), "R_gain_ci95": r_ci, "R_clause": r_clause,
            "others": {b: {"point": float(g[b][0]), "ci95": oc[b]} for b in others},
            "others_point_reading": bool(point), "others_ci_strict": bool(strict),
            "fires_point_reading": bool(r_clause and point), "fires_ci_reading_strict": bool(r_clause and strict)}


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
# Development entry (phase A): train and val only, tunes every alpha, runs the gate
# ---------------------------------------------------------------------------

def _claim(out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def dev(trval_root, raw_dir, suite, out_dir, ftype="global", sizes=None, stored_root=None, tally_dir=None,
        pca_whiten=PCA_WHITEN, expected_frames=EXPECTED_FRAMES, quiet=False):
    """Development phase: reads re-rendered train and val features only. Writes <out_dir>/dev-report.json.

    The dir is claimed up front and removed again if the run fails. A failed gate is not an exception here: the
    report is written with gate.passed = False (the CLI then exits non-zero) and the final run refuses it.
    """
    out_dir = _claim(out_dir)
    try:
        return _dev(trval_root, raw_dir, suite, out_dir, ftype, sizes, stored_root, tally_dir, pca_whiten,
                    expected_frames, quiet)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _dev(trval_root, raw_dir, suite, out_dir, ftype, sizes, stored_root, tally_dir, pca_whiten, expected_frames, quiet):
    t0 = time.time()
    log = (lambda *a: None) if quiet else (lambda *a: print("[pf-dev]", *a, flush=True))
    encoders = [DINO] + list(PARTNERS.values())
    ftypes = sorted({"global", ftype})
    tr = load_rendered(trval_root, raw_dir, suite, encoders, ftypes, phases=DEV_PHASES, allow_test=False,
                       expected_frames=expected_frames)
    log(f"train+val: {tr.counts} frames per camera, {len(tr.index)} demos")
    train, val = tr.split == "train", tr.split == "val"
    notes = {}

    # Gate, check A: train-only DINOv3 fit, combined view, global features, best alpha on the tally grid;
    # the scored value is the tune_alpha curve value at that alpha (val R2), not the train+val refit.
    tfs_g = {c: CamTransform(tr.feats[DINO]["global"][c][train], None, pca_whiten) for c in CAMS}
    xg = np.concatenate([tfs_g[c](tr.feats[DINO]["global"][c]) for c in CAMS], 1)
    g_alpha, g_curve = tune_alpha(xg, tr.actions, tr.split)
    g_score = float(g_curve[g_alpha])
    thr = gate_threshold()
    gate = {"check": "A (PLAN Amendment 2026-11-09 section 4)",
            "scored": "val R2 (mean of 7 action dims) of the train-only DINOv3 fit at its best alpha on the tally grid, "
                      "combined view, global features, re-rendered train and val (the tune_alpha curve value, not the train+val refit)",
            "score": g_score, "alpha": g_alpha, "val_r2_by_alpha": {str(a): v for a, v in g_curve.items()},
            "reference": GATE_REF,
            "reference_source": "results/pf-val/20261108-181737/probe-report.json, val_r2.combined.val_r2_stored",
            "margin": GATE_MARGIN, "margin_source": "user decision 2026-11-09 (0.025, one-sided)",
            "threshold": thr, "passed": bool(g_score >= thr)}
    log(f"gate A: score {g_score:.4f}, threshold {thr:.4f}, passed {gate['passed']}")

    # Optional references, report only.
    if stored_root is None:
        notes["lag_check"] = notes["cross_value"] = "skipped: no stored-feature root given"
        lag = cross = None
    else:
        lag = lag_check_stored(tr, stored_root, suite, "global", pca_whiten)
        w_tr, b_tr = D.ridge_fit(xg[train], tr.actions[train], g_alpha)
        vidx = [e for e in tr.index if e[3] == "val"]
        sv = load_stored(stored_root, suite, DINO, "global", vidx, tr.meta[DINO])
        xs = np.concatenate([tfs_g[c](np.concatenate([a[:-1] for a in sv[c]])) for c in CAMS], 1)
        cross = {"val_r2_stored_features": float(D.r2(tr.actions[val], xs @ w_tr + b_tr).mean()),
                 "val_r2_rerender_features": g_score, "alpha": g_alpha,
                 "note": "re-render-trained DINOv3 probe (train-only fit at the gate's alpha) scored on stored val features, n-1 frames per demo; report only"}
    dims = per_camera_sizes(tr.feats, ftype)
    target, reduced = matched_plan(dims, sizes)
    log(f"per-camera sizes {dims}; matched target {target}, reduced {reduced}")

    report = {"kind": DEV_KIND, "suite": suite, "ftype": ftype, "trval_root": str(Path(trval_root).resolve()),
              "raw_dir": str(raw_dir), "files": tr.files, "git_commit": D.git_commit(), "pca_whiten": pca_whiten,
              "n_frames": tr.counts, "n_demos": len(tr.index), "expected_frames": expected_frames,
              "encoders": {e: tr.meta[e] for e in encoders}, "alphas_grid": D.ALPHAS, "gate": gate,
              "lag_check": lag, "cross_value": cross, "per_camera_sizes": dims,
              "matched": {"target": target, "reduced": reduced, "declared_sizes": sizes}, "configs": {}, "notes": notes}
    if not reduced:
        report["matched"]["note"] = "all encoders already have the smallest size; the matched row would equal the full row and is skipped"
    for config, view, name, x, parts, pca_log in iter_probes(tr.feats, ftype, tr.split, target, reduced, pca_whiten):
        cv = report["configs"].setdefault(config, {}).setdefault(view, {"probes": {}, "pca_explained_variance": pca_log})
        alpha, curve = tune_alpha(x, tr.actions, tr.split)
        cv["probes"][name] = {"dim": int(x.shape[1]), "alpha": alpha, "alpha_source": "tuned-on-rerendered-val",
                              "val_r2": float(curve[alpha]), "val_r2_by_alpha": {str(a): v for a, v in curve.items()}}
        log(f"{config}/{view}/{name}: dim {x.shape[1]}, alpha {alpha}, val R2 {curve[alpha]:.3f}")
    for config, views in report["configs"].items():
        for view, cv in views.items():
            pr = cv["probes"]
            cv["val_gain"] = {b: pr["D+" + b]["val_r2"] - pr["D"]["val_r2"] for b in PARTNERS}
            cv["val_gain_order"] = sorted(PARTNERS, key=lambda b: -cv["val_gain"][b])
    if tally_dir is None:
        notes["tally_reference"] = "skipped: no tally dir given"
    else:
        tg = reference_tally_val_gains(tally_dir, suite, ftype)
        ours = report["configs"]["full"]["combined"]["val_gain"]
        report["tally_reference"] = {"tally_val_gain": tg, "ours_val_gain": ours,
                                     "tally_order": sorted(tg, key=lambda b: -tg[b]),
                                     "ours_order": report["configs"]["full"]["combined"]["val_gain_order"],
                                     "note": "report only (PLAN T-a, words only); stored-trained tally vs re-render-trained ours; no test, no pass mark"}
    report["seconds"] = time.time() - t0
    (out_dir / "dev-report.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {out_dir / 'dev-report.json'} in {report['seconds']:.0f}s")
    return report


def check_dev_report(path, trval_root, suite, pca_whiten):
    """The final run's gate: a dev report whose gate passed, for the same train/val root, suite and PCA setting."""
    p = Path(path)
    if not p.is_file():
        raise GateError(f"dev report {p} not found; the final run needs a passing dev report")
    rep = json.loads(p.read_text())
    g = rep.get("gate") or {}
    if rep.get("kind") != DEV_KIND:
        raise GateError(f"{p} is not a dev report")
    if g.get("passed") is not True:
        raise GateError(f"{p}: the gate did not pass (score {g.get('score')}, threshold {g.get('threshold')})")
    if g.get("reference") != GATE_REF or g.get("margin") != GATE_MARGIN:
        raise GateError(f"{p}: gate constants ({g.get('reference')}, {g.get('margin')}) differ from this code's "
                        f"({GATE_REF}, {GATE_MARGIN})")
    if not isinstance(g.get("score"), float) or not g["score"] >= gate_threshold():
        raise GateError(f"{p}: gate score {g.get('score')} is below the threshold {gate_threshold()}")
    if rep.get("trval_root") != str(Path(trval_root).resolve()):
        raise GateError(f"{p}: dev ran on train/val root {rep.get('trval_root')}, the final run was given {Path(trval_root).resolve()}")
    if rep.get("suite") != suite:
        raise GateError(f"{p}: dev suite {rep.get('suite')} != {suite}")
    if rep.get("pca_whiten") != pca_whiten:
        raise GateError(f"{p}: dev pca_whiten {rep.get('pca_whiten')} != {pca_whiten}")
    return rep


# ---------------------------------------------------------------------------
# Final run (phase B)
# ---------------------------------------------------------------------------

def run(trval_root, raw_dir, suite, cond_roots, dev_report, out_dir, *args, **kw):
    """Run the whole analysis and write results.json, stats.npz (and preds.npz) under out_dir (must not exist).

    Refuses to start without a passing dev report for this train/val root. The dir is claimed up front; if the
    run fails the (partial) dir is removed so a corrected retry can reuse the path. Arguments after out_dir are those of _run.
    """
    out_dir = _claim(out_dir)
    try:
        return _run(trval_root, raw_dir, suite, cond_roots, dev_report, out_dir, *args, **kw)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _run(trval_root, raw_dir, suite, cond_roots, dev_report, out_dir, ftype="global", n_boot=1000, seed=0,
         sizes=None, save_preds=True, min_boot=MIN_BOOT, quiet=False, pca_whiten=PCA_WHITEN,
         expected_frames=EXPECTED_FRAMES, stored_root=None):
    t0 = time.time()
    if n_boot < min_boot:
        raise ValueError(f"n_boot={n_boot} < {min_boot}: the plan needs at least {MIN_BOOT} draws")
    dev_rep = check_dev_report(dev_report, trval_root, suite, pca_whiten)     # before anything is read
    need = [REF] + TYPES
    missing = [c for c in need if c not in cond_roots]
    if missing:
        raise KeyError(f"cond_roots missing {missing}")
    encoders = [DINO] + list(PARTNERS.values())
    log = (lambda *a: None) if quiet else (lambda *a: print("[pf]", *a, flush=True))

    tr = load_rendered(trval_root, raw_dir, suite, encoders, [ftype], phases=("train", "val"), allow_test=False,
                       expected_frames=expected_frames)
    if tr.counts != dev_rep["n_frames"]:
        raise GateError(f"train/val frame counts {tr.counts} differ from the dev report's {dev_rep['n_frames']}")
    feats, meta, split = tr.feats, tr.meta, tr.split
    trva = np.ones(len(split), bool)                       # every row loaded is train or val
    lag = None if stored_root is None else lag_check_stored(tr, stored_root, suite, ftype, pca_whiten)
    demos, y_test = load_test_layout(raw_dir, expected_frames)      # first read of test labels
    lengths = [n - 1 for _, _, n, _ in demos]
    groups = np.repeat(np.arange(len(demos)), lengths)
    k = len(demos)
    log(f"{k} test demos, {len(y_test)} frames per condition")

    # Condition features, test frames only: cond[c][enc][cam]. Every file must hold exactly n-1 frames per demo.
    conds = [REF] + TYPES
    cond = {}
    for c in conds:
        cond[c] = {}
        for e in encoders:
            if cond_demo_lengths(cond_roots[c], e, suite, ftype, demos, meta[e]) != lengths:
                raise ValueError(f"condition {c} / {e}: frame counts differ from n-1")
            cond[c][e] = load_condition(cond_roots[c], e, suite, ftype, demos, lengths, meta[e])
    dims = per_camera_sizes(feats, ftype)
    target, reduced = matched_plan(dims, sizes)
    log(f"per-camera sizes {dims}; matched target {target}, reduced {reduced}")

    W = make_weights(k, n_boot, seed)
    results = {"suite": suite, "ftype": ftype, "files": tr.files, "encoders": {e: meta[e] for e in encoders},
               "git_commit": D.git_commit(), "seed": seed, "n_boot": n_boot, "n_test_demos": k,
               "n_test_frames": int(len(y_test)), "n_trainval_frames": tr.counts,
               "frames_per_demo_dropped_last": [int(n - t) for (_, _, n, _), t in zip(demos, lengths)],
               "pca_whiten": pca_whiten, "pca_explained_variance": {},
               "trval_root": str(Path(trval_root).resolve()), "dev_report": str(dev_report), "dev_gate": dev_rep["gate"],
               "lag_check": lag if lag is not None else "skipped: no stored-feature root given",
               "cond_roots": {c: str(v) for c, v in cond_roots.items()},
               "per_camera_sizes": dims, "matched": {"target": target, "reduced": reduced, "declared_sizes": sizes},
               "alphas_grid": D.ALPHAS, "configs": {}, "drift": {}}
    stats_out = {"test_demos": np.array([f"{s}/{d}" for s, d, _, _ in demos]), "frames_per_demo": np.array(lengths),
                 "boot_weights": W.astype(np.int16)}
    preds_out = {"y_test": y_test.astype(np.float32)}
    if not reduced:
        results["matched"]["note"] = "all encoders already have the smallest size; the matched row would equal the full row and is skipped"

    st_all, alpha_mismatch = {}, []   # st_all[(config, view)][condition][probe] = per-demo stats
    for config, view, name, x, parts, pca_log in iter_probes(feats, ftype, split, target, reduced, pca_whiten):
        cams = VIEWS[view]
        results["pca_explained_variance"].setdefault(config, {})[view] = pca_log
        res_cv = results["configs"].setdefault(config, {}).setdefault(view, {"probes": {}})
        alpha, curve = tune_alpha(x, tr.actions, split)
        w, bias = D.ridge_fit(x[trva], tr.actions[trva], alpha)
        res_cv["probes"][name] = {"dim": int(x.shape[1]), "alpha": alpha, "alpha_source": "tuned-on-rerendered-val",
                                  "val_r2_by_alpha": {str(a_): v_ for a_, v_ in curve.items()}}
        if dev_rep["ftype"] == ftype:
            want = dev_rep["configs"].get(config, {}).get(view, {}).get("probes", {}).get(name, {}).get("alpha")
            if want != alpha:
                alpha_mismatch.append(f"{config}/{view}/{name}: dev {want}, final {alpha}")
        st = st_all.setdefault((config, view), {c: {} for c in conds})
        for cn in conds:
            xt = np.concatenate([np.concatenate([tfs[c](cond[cn][e][c]) for c in cams], 1) for e, tfs in parts], 1)
            yhat = xt @ w + bias
            st[cn][name] = D.demo_stats(y_test, yhat, groups)
            if save_preds:
                preds_out[f"{config}/{view}/{cn}/{name}"] = yhat.astype(np.float32)
        log(f"{config}/{view}/{name}: dim {x.shape[1]}, alpha {alpha}, R2 rerender {float(D.r2_from_stats(W[:1], *st[REF][name])[0]):.3f}")
        if name == LAST_PROBE:      # last probe of this (config, view): fold the statistics into the quantities
            for cn, ps in st.items():
                for p, (n_, sy, syy, sres) in ps.items():
                    for nm, arr in zip(("n", "sy", "syy", "sres"), (n_, sy, syy, sres)):
                        stats_out[f"{config}/{view}/{cn}/{p}/{nm}"] = arr
            q = quantities(st, W)
            res_cv.update({
                "r2": {c: {p: stat(a) for p, a in ps.items()} for c, ps in q["r2"].items()},
                "drop": {c: {p: stat(a) for p, a in ps.items()} for c, ps in q["drop"].items()},
                "e": {c: {b: stat(a) for b, a in bs.items()} for c, bs in q["e"].items()},
                "gain": {c: {b: stat(a) for b, a in bs.items()} for c, bs in q["g"].items()},
                "pooled_e": {b: stat(a) for b, a in q["pooled_e"].items()},
                "pooled_g": {b: stat(a) for b, a in q["pooled_g"].items()},
                "rule_P": rule_p(q["pooled_e"]), "rule_B": rule_b(q["pooled_g"]),
                "ordering": ordering(q["pooled_e"])})
    results["alphas_match_dev"] = (alpha_mismatch == [] if dev_rep["ftype"] == ftype else
                                   "not compared: dev ran on ftype " + dev_rep["ftype"])
    if alpha_mismatch:
        results["alphas_differ_from_dev"] = alpha_mismatch
        log("WARNING alphas differ from the dev report:", alpha_mismatch)

    # Drift (PLAN: global features only), single encoder, full-size standardized features (no PCA), clean re-render
    # as the reference, statistics from the re-rendered clean-train frames. Not defined in the PLAN for the mean
    # features, so it is skipped under ftype=mean.
    if ftype != "global":
        results["drift_note"] = "drift is defined in the PLAN for global features only; skipped for ftype=" + ftype
    else:
        train = split == "train"
        for view, cams in VIEWS.items():
            results["drift"][view] = {}
            for e in encoders:
                tfs = {c: CamTransform(feats[e][ftype][c][train]) for c in cams}
                z = {cn: np.concatenate([tfs[c].standardize(cond[cn][e][c]) for c in cams], 1) for cn in conds}
                row = {}
                for cn in TYPES:
                    num, den = drift_parts(z[REF], z[cn], groups, k)
                    stats_out[f"drift/{view}/{e}/{cn}/num"], stats_out[f"drift/{view}/{e}/{cn}/den"] = num, den
                    row[cn] = stat(drift_draws(num, den, W))
                results["drift"][view][e] = row
        results["drift_note"] = ("drift_c(E) = mean ||f(x')-f(x)||^2 / mean ||f(x)-mean(f)||^2 over test frames, x = clean re-render, "
                                 "mean(f) = mean over the test frames, features standardized with re-rendered clean-train statistics.")

    results["seconds"] = time.time() - t0
    np.savez_compressed(out_dir / "stats.npz", **stats_out)
    if save_preds:
        np.savez_compressed(out_dir / "preds.npz", **preds_out)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {out_dir} in {results['seconds']:.0f}s")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trval-feats", required=True, help="re-rendered clean train+val feature root (phase A)")
    common.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files (actions, split)")
    common.add_argument("--suite", required=True)
    common.add_argument("--out", required=True, help="new output dir (refused if it exists)")
    common.add_argument("--ftype", default="global", choices=["global", "mean"])
    common.add_argument("--sizes", default=None, help="declared per-camera sizes enc=dim,... (checked against the files)")
    common.add_argument("--pca-whiten", action="store_true", help="divide PCA scores by their axis std (default: unwhitened)")
    common.add_argument("--stored-feats", default=None,
                        help="optional reference: stored feature root (train and val demos only are read): lag check, cross value")
    d = sub.add_parser("dev", parents=[common], help="phase A: train and val only; tunes alphas, runs the gate")
    d.add_argument("--tally-results", default=None, help="optional reference: dir with tally-*.json (its val gains are shown beside ours)")
    f = sub.add_parser("final", parents=[common], help="phase B: needs a passing dev report")
    f.add_argument("--cond", action="append", required=True, metavar="NAME=ROOT",
                   help=f"test feature root per condition; names: {REF} {' '.join(TYPES)} (repeat)")
    f.add_argument("--dev-report", required=True, help="dev-report.json of a dev run whose gate passed (same --trval-feats)")
    f.add_argument("--n-boot", type=int, default=1000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--no-save-preds", action="store_true")
    a = p.parse_args()
    sizes = {k: int(v) for k, v in (s.split("=") for s in a.sizes.split(","))} if a.sizes else None
    if a.cmd == "dev":
        rep = dev(a.trval_feats, a.raw, a.suite, a.out, a.ftype, sizes, a.stored_feats, a.tally_results, a.pca_whiten)
        if not rep["gate"]["passed"]:
            print(f"[pf-dev] GATE FAILED: score {rep['gate']['score']:.4f} < threshold {rep['gate']['threshold']:.4f}; "
                  "stop and return to the user (Q8 rule)", file=sys.stderr)
            sys.exit(1)
    else:
        roots = dict(s.split("=", 1) for s in a.cond)
        run(a.trval_feats, a.raw, a.suite, roots, a.dev_report, a.out, a.ftype, a.n_boot, a.seed, sizes,
            not a.no_save_preds, pca_whiten=a.pca_whiten, stored_root=a.stored_feats)


if __name__ == "__main__":
    main()
