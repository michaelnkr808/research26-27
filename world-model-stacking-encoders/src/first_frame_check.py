"""First-frame feature check: are the encoder inputs of a rollout farther from training than the VAE inputs?

Stage 2 diagnosis of the pretrained-start pairings. The design is fixed in
experiments/2026-11-15-pretrained-start-pairings/diagnosis/2026-11-25-first-frame-check-plan.md (with its amendment of
2026-11-25, per-channel value ranges) and is implemented here as written; the Job steps and the map from each plan item
to the function that implements it are in that experiment's jobs/RUN-first-frame-check.md.

Four channel groups: the VAE (48 channels, raw, as the model sees them) and DINOv3, C-RADIOv4 and SigLIP 2 (each z-scored
with the feature_stats in the config of its own single-encoder seed 0 checkpoint, on both sides).
  live (plan Design 1): the 40 clean LIBERO tasks x init states 0 to 4, each frame built exactly as a rollout builds its
    first policy frame (rollout.first_frame: rollout.libero_env, reset, set_init_state, the 30 dummy steps,
    rollout.process_obs; global seeds = the eval seed 41 as rollout.run_episode sets them), in spawned simulator
    processes as rollouts run them, then rollout.Featurizer at batch 1 on the GPU (an L4, the evaluation card).
  training (Design 2): frame 0 of every training episode of the same task (task from <feats>/meta/<suite>.npz).
    Encoders: the cached <feats>/<encoder>/<suite>/ep<id>.npy[0], z-scored. VAE: frame 0 encoded alone, a one-frame clip
    through the training-side decode (fastwam_data.decode_episode, stopped after frame 0), fastwam_data.to_224 and
    fastwam_data.video_clips, then the Featurizer's own VAE (extract_fastwam.load_vae, bf16), rounded through float16.
  distance (Design 3): d(x, y) = ||x - y||_2 / sqrt(C x 14 x 28), on the flattened (C, 14, 28) arrays.
  per frame (Design 4): r(i, g) = d(i, nearest training frame 0 of its task) / ref(k, g), ref(k, g) = median over the
    task's training frame 0s of the distance to the nearest other one. A task with fewer than 2 training episodes or
    ref(k, g) = 0 for any group is left out of every statistic and named.
  per group (Design 5): m(g) = median of r(i, g) over all live frames pooled.
  statistic (Design 6): R(e) = m(e) / m(VAE), with a 95% percentile interval from 1,000 task-level bootstrap draws (seed
    0; tasks drawn with replacement, all 5 frames of each drawn task, ref not recomputed).
  reading rule: zones from the point estimate (<= 1.2 near, >= 1.5 clearly larger, else unclear), borderline wording when
    the interval crosses an edge, the DINOv3 channel 757 rule, and the outcome case (1, 2 or 3) in the plan's order.
  also reported (descriptive): median and 90th percentile of r per group per suite and overall; DINOv3 channel 757's live
    z-scores; the single-frame VAE latents against the cached clip-encoded ones (item 15); the per-task ratios for the
    Spearman correlation with the clean success drop (computed on the Mac by the correlate command, from the committed
    clean records, which are not on the cluster volume); the amendment's per-channel ranges, mean squares and flags.

Commands:
  run        the check on the cluster (needs torch, the simulator, the caches, the dataset, the checkpoints, a GPU).
             --dry-run: one task per suite; saves the live frame and the training frame 0 of each task side by side as
             PNG (both cameras), runs every step end to end, and writes no statistic of the check (only the pipeline
             checks, shapes and timings), so no number of the check exists before the full run.
  correlate  on the Mac, after the full run: Spearman rank correlation over tasks between each encoder's per-task ratio
             (from the result JSON) and its per-task clean success drop (from the committed clean records).

Exit codes (run): 0 done; 3 a pipeline check failed (a cache file missing or of the wrong length, a checkpoint config
that is not the expected single-encoder seed 0 run, an encoder revision mismatch, not an L4, the frame-0 decode check,
the training task list); 2 the script failed (an exception or bad arguments). 1 is not used by the script.

Pure functions (distance, nearest neighbours, r, m, R, the bootstrap, zones, ranges and flags, Spearman) need only numpy
and are tested by test_first_frame_check.py on the Mac. torch and the cluster modules are imported inside the functions
that need them.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ENCODERS = ("dinov3", "c-radiov4", "siglip2")
VAE = "vae"
GROUPS = (VAE,) + ENCODERS                  # the four channel groups of the design
DINO = "dinov3"
DINO_757 = 757                              # the near-constant DINOv3 channel (diagnosis note item 19)
NO757 = "dinov3_without_757"                # DINOv3 with channel 757 left out (reading rule, DINOv3 only)
INIT_STATES = (0, 1, 2, 3, 4)
EVAL_SEED = 41                              # the clean evaluation's eval seed (rollout.make_jobs clean_seed)
GRID = (14, 28)
# Reading rule (plan, "Reading rule"), fixed before the run.
NEAR_MAX, CLEAR_MIN = 1.2, 1.5
NEAR, UNCLEAR, CLEAR = "near", "unclear", "clearly larger"
N_BOOT, BOOT_SEED, CI = 1000, 0, (2.5, 97.5)
# Amendment 2026-11-25 (per-channel value ranges), fixed before the run.
SD_LO, SD_HI, ABS_MEAN_HI = 0.25, 4.0, 2.0  # test (a), set T
SD_RATIO = 4.0                              # test (b): sd_L below sd_T / 4 or above 4 sd_T
MEAN_SHIFT_SDS = 2.0                        # test (c): |mean_L - mean_T| above 2 sd_T
CONST_SD = 1e-6                             # sd_T below this: (b) and (c) not applied, flagged constant on set T
MS_FACTOR = 4.0                             # "in very different ranges": mean-square ratio above 4 or below 1/4
LP_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
PLAN = "experiments/2026-11-15-pretrained-start-pairings/diagnosis/2026-11-25-first-frame-check-plan.md"

OUTCOME_TEXT = {
    1: "Case 1: at least one encoder 'clearly larger'. The plan's next step: the planning lead proposes the settling run "
       "for that encoder and none (its own design by PR and the user's go); no fix to the rollout frame path is made "
       "from this check alone.",
    2: "Case 2: all three 'near'. B is not supported; no settling run is proposed from this check; what remains (A or "
       "seed variance) goes to the user with the choices already open.",
    3: "Case 3: no encoder 'clearly larger', at least one 'unclear'. Reported as it is; the user decides whether the "
       "settling run is worth doing anyway.",
}


class CheckFailed(Exception):
    """A pipeline check failed (exit 3)."""


# ---------------------------------------------------------------------------
# Distances, nearest neighbours, r, m, R (plan Design 3 to 6)
# ---------------------------------------------------------------------------

def flatten(x):
    """(n, C, 14, 28) (or any (n, ...)) -> (n, P) float64."""
    x = np.asarray(x)
    return x.reshape(len(x), -1).astype(np.float64)


def norm_dist(a, b):
    """Design 3: L2 distance between two same-shape arrays, flattened, divided by sqrt(number of values)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.sqrt(np.sum((a - b) ** 2)) / np.sqrt(a.size))


def nearest_dists(query, ref, exclude_self=False):
    """For each row of query (q, P), the normalized distance to its nearest row of ref (n, P) (exclude_self: query is
    ref and each row's own index is skipped). The nearest index is found from float64 Gram-matrix distances; the
    distance to it is then recomputed exactly by direct difference (norm_dist), so an exact duplicate gives exactly 0.
    Near-ties may pick either row: the plan needs no tie rule. Returns (distances (q,), indices (q,))."""
    Q, Rm = flatten(query), flatten(ref)
    sq = (Q * Q).sum(1)[:, None] + (Rm * Rm).sum(1)[None, :] - 2.0 * (Q @ Rm.T)
    sq = np.maximum(sq, 0.0)
    if exclude_self:
        assert len(Q) == len(Rm)
        np.fill_diagonal(sq, np.inf)
    idx = np.argmin(sq, axis=1)
    d = np.array([norm_dist(Q[i], Rm[j]) for i, j in enumerate(idx)])
    return d, idx


def task_ratios(train, live):
    """Design 4 for one task and one group. train (n, C, 14, 28) training frame 0s, live (q, C, 14, 28) live frames.
    ref = median over training frames of the distance to the nearest OTHER training frame (leave self out); d_live =
    distance of each live frame to its nearest training frame (the live frame is not in the training set, so nothing is
    left out); r = d_live / ref (None when n < 2 or ref == 0: the task is then left out by the caller)."""
    n = len(train)
    out = {"n_train": n}
    if n < 2:
        out.update(ref=None, ref_dists=None, d_live=None, nn_train=None, r=None)
        return out
    ref_d, _ = nearest_dists(train, train, exclude_self=True)
    ref = float(np.median(ref_d))
    d_live, nn = nearest_dists(live, train)
    out.update(ref=ref, ref_dists=ref_d.tolist(), d_live=d_live.tolist(), nn_train=nn.tolist(),
               r=(d_live / ref).tolist() if ref > 0 else None)
    return out


def drop_channel(x, ch, axis=1):
    """x with one channel removed along axis (the channel axis of (n, C, 14, 28))."""
    return np.delete(np.asarray(x), ch, axis=axis)


def pooled_median(per_task):
    """Design 5: median over all frames pooled, each counted once. per_task: list of per-task r lists."""
    v = np.concatenate([np.asarray(x, np.float64) for x in per_task])
    return float(np.median(v))


def ratio_R(m_enc, m_vae):
    """Design 6: R(e) = m(e) / m(VAE)."""
    return float(m_enc) / float(m_vae)


def bootstrap(r_by_group, enc_groups, n_boot=N_BOOT, seed=BOOT_SEED, ci=CI):
    """Design 6 uncertainty. r_by_group: {group: [per-task arrays of r]}, the tasks aligned across groups (same order);
    must hold VAE. Each draw: K task indices with replacement (rng = numpy default_rng(seed), rng.integers(0, K, K), one
    draw shared by every group), all frames of each drawn task (a task drawn twice counts twice), then m(g) for every
    group and R(e) = m(e) / m(VAE). The per-task ref is NOT recomputed (r values are reused as they are). Returns
    {e: {"draws": [...], "interval": [lo, hi]}} with the percentile interval (numpy percentile, linear)."""
    K = len(r_by_group[VAE])
    assert all(len(v) == K for v in r_by_group.values())
    rng = np.random.default_rng(seed)
    arrs = {g: [np.asarray(x, np.float64) for x in v] for g, v in r_by_group.items()}
    draws = {e: np.empty(n_boot) for e in enc_groups}
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        m_v = float(np.median(np.concatenate([arrs[VAE][k] for k in idx])))
        for e in enc_groups:
            draws[e][b] = float(np.median(np.concatenate([arrs[e][k] for k in idx]))) / m_v
    return {e: {"draws": d.tolist(), "interval": [float(x) for x in np.percentile(d, ci)]} for e, d in draws.items()}


# ---------------------------------------------------------------------------
# Reading rule and outcome case
# ---------------------------------------------------------------------------

def zone(R):
    """Reading rule zones, from a value of R."""
    if R <= NEAR_MAX:
        return NEAR
    if R >= CLEAR_MIN:
        return CLEAR
    return UNCLEAR


def crossed_edges(lo, hi):
    """Edges (1.2, 1.5) the interval [lo, hi] crosses: those with interval values in the zones on both sides of it
    (1.2: lo <= 1.2 < hi; 1.5: lo < 1.5 <= hi)."""
    edges = []
    if lo <= NEAR_MAX < hi:
        edges.append(NEAR_MAX)
    if lo < CLEAR_MIN <= hi:
        edges.append(CLEAR_MIN)
    return edges


def borderline_wording(zone_name, edges):
    if not edges:
        return f"{zone_name}"
    return f"{zone_name}, borderline: the bootstrap interval crosses " + " and ".join(f"{e:g}" for e in edges)


def read_encoder(R, interval, R_no757=None):
    """One encoder's reading from the point estimate (the interval changes neither zone nor next step). DINOv3 (R_no757
    given): if the zone with channel 757 left out differs from the zone with all channels, DINOv3 is read 'unclear' and
    the reading depends on that channel."""
    z = zone(R)
    edges = crossed_edges(*interval)
    out = {"R": R, "interval": list(interval), "zone": z, "borderline_edges": edges,
           "borderline": bool(edges), "wording": borderline_wording(z, edges), "reading": z}
    if R_no757 is not None:
        z7 = zone(R_no757)
        out["R_without_757"] = R_no757
        out["zone_without_757"] = z7
        out["depends_on_757"] = z7 != z
        if z7 != z:
            out["reading"] = UNCLEAR
            out["wording"] += (f"; read as 'unclear': the zone is '{z}' with all channels and '{z7}' with channel 757 "
                               f"left out, so the reading depends on that channel")
    return out


def outcome_case(readings):
    """The plan's three outcome cases, checked in order. readings: {encoder: reading}."""
    vals = list(readings.values())
    if any(v == CLEAR for v in vals):
        case = 1
    elif all(v == NEAR for v in vals):
        case = 2
    else:
        case = 3
    return {"case": case, "text": OUTCOME_TEXT[case],
            "clearly_larger": [e for e, v in readings.items() if v == CLEAR]}


# ---------------------------------------------------------------------------
# Descriptive items
# ---------------------------------------------------------------------------

def quantiles(v):
    v = np.asarray(v, np.float64)
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def r_summaries(frames, groups):
    """Median and 90th percentile of r per group, per suite and over all frames. frames: [{suite, r: {g: value}}]."""
    out = {}
    for g in groups:
        out[g] = {"all": quantiles([f["r"][g] for f in frames])}
        for s in SUITES:
            v = [f["r"][g] for f in frames if f["suite"] == s]
            if v:
                out[g][s] = quantiles(v)
    return out


def channel_757_live(z_values):
    """DINOv3 channel 757's live z-scores over all live frames and positions: median and largest absolute value."""
    v = np.asarray(z_values, np.float64).ravel()
    return {"n_values": int(v.size), "median": float(np.median(v)), "median_abs": float(np.median(np.abs(v))),
            "max_abs": float(np.max(np.abs(v)))}


def latent_diff(single, cached):
    """Item 15: single-frame latent against the cached clip-encoded one, same layout. (max abs diff, relative L2)."""
    a = np.asarray(single, np.float64).ravel()
    b = np.asarray(cached, np.float64).ravel()
    nb = np.linalg.norm(b)
    return float(np.max(np.abs(a - b))), (float(np.linalg.norm(a - b) / nb) if nb > 0 else None)


def latent_diff_summary(rows):
    """rows: [{suite, episode, max_abs_diff, rel_l2}] -> overall and per-suite summary."""
    def summ(rs):
        if not rs:
            return None
        ma = np.array([r["max_abs_diff"] for r in rs])
        rl = [r["rel_l2"] for r in rs if r["rel_l2"] is not None]       # None: a cached latent of all zeros
        return {"n_episodes": len(rs), "max_abs_diff_max": float(ma.max()), "max_abs_diff_median": float(np.median(ma)),
                "rel_l2_max": float(max(rl)) if rl else None, "rel_l2_median": float(np.median(rl)) if rl else None,
                "n_rel_l2_undefined": len(rs) - len(rl),
                "n_identical": int(sum(r["max_abs_diff"] == 0 for r in rs))}
    out = {"all": summ(rows)}
    for s in SUITES:
        out[s] = summ([r for r in rows if r["suite"] == s])
    return out


def task_level_ratios(per_task_r, enc_groups):
    """Per task: median r of each encoder group over the task's frames divided by the VAE's (the Spearman input)."""
    return {e: float(np.median(per_task_r[e])) / float(np.median(per_task_r[VAE])) for e in enc_groups}


def rankdata(v):
    """Ranks 1..n with ties given their average rank (as scipy.stats.rankdata 'average')."""
    v = np.asarray(v, np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation (Pearson correlation of average ranks); None when either side is constant."""
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else None


# ---------------------------------------------------------------------------
# Amendment 2026-11-25: per-channel value ranges and flags
# ---------------------------------------------------------------------------

class ChannelStats:
    """Per-channel pooled statistics over frames and the 14 x 28 positions, accumulated in float64 batch by batch.
    Shifted sums (shift = the first batch's channel means) keep the variance accurate; sum of squares is kept raw
    for the mean square."""

    def __init__(self, channels):
        self.C = channels
        self.n = 0
        self.shift = None
        self.s1 = np.zeros(channels)
        self.s2 = np.zeros(channels)
        self.sq = np.zeros(channels)
        self.maxabs = np.zeros(channels)

    def add(self, x):
        """x: (n, C, ...) values."""
        x = np.asarray(x)
        assert x.shape[1] == self.C, (x.shape, self.C)
        v = np.moveaxis(x, 1, 0).reshape(self.C, -1).astype(np.float64)
        if v.shape[1] == 0:
            return
        if self.shift is None:
            self.shift = v.mean(1)
        d = v - self.shift[:, None]
        self.s1 += d.sum(1)
        self.s2 += (d * d).sum(1)
        self.sq += (v * v).sum(1)
        self.maxabs = np.maximum(self.maxabs, np.abs(v).max(1))
        self.n += v.shape[1]

    def mean(self):
        return self.shift + self.s1 / self.n

    def sd(self):
        """Population standard deviation (divide by N)."""
        m1 = self.s1 / self.n
        return np.sqrt(np.maximum(self.s2 / self.n - m1 * m1, 0.0))

    def mean_square(self):
        """Overall mean square of the group: mean of squared values over all channels, frames and positions."""
        return float(self.sq.sum() / (self.n * self.C))

    def table(self):
        return {"n_values_per_channel": int(self.n), "mean": self.mean().tolist(), "sd": self.sd().tolist(),
                "max_abs": self.maxabs.tolist()}


def group_summary(st):
    """Per group and set: median, smallest and largest per-channel sd; median and largest absolute per-channel mean;
    largest absolute value; overall mean square."""
    sd, mu = st.sd(), st.mean()
    return {"n_channels": st.C, "n_values_per_channel": int(st.n),
            "sd_median": float(np.median(sd)), "sd_min": float(sd.min()), "sd_max": float(sd.max()),
            "abs_mean_median": float(np.median(np.abs(mu))), "abs_mean_max": float(np.abs(mu).max()),
            "max_abs": float(st.maxabs.max()), "mean_square": st.mean_square()}


def channel_flags(mean_T, sd_T, mean_L, sd_L):
    """The amendment's tests, per channel. Returns {test: [channel indices]}:
      a: on set T, sd below 0.25 or above 4, or |mean| above 2;
      b: sd on L below sd_T / 4 or above 4 sd_T (not applied when sd_T < 1e-6);
      c: |mean_L - mean_T| above 2 sd_T (not applied when sd_T < 1e-6);
      constant_T: sd_T < 1e-6 (the exception of b and c)."""
    mean_T, sd_T, mean_L, sd_L = (np.asarray(x, np.float64) for x in (mean_T, sd_T, mean_L, sd_L))
    const = sd_T < CONST_SD
    a = (sd_T < SD_LO) | (sd_T > SD_HI) | (np.abs(mean_T) > ABS_MEAN_HI)
    b = ~const & ((sd_L < sd_T / SD_RATIO) | (sd_L > SD_RATIO * sd_T))
    c = ~const & (np.abs(mean_L - mean_T) > MEAN_SHIFT_SDS * sd_T)
    return {k: np.flatnonzero(v).tolist() for k, v in (("a", a), ("b", b), ("c", c), ("constant_T", const))}


def flag_report(flags, channels):
    return {k: {"channels": v, "count": len(v), "share": len(v) / channels} for k, v in flags.items()}


def range_wording(ms_enc_T, ms_vae_T):
    """The amendment's one wording, from set T."""
    ratio = ms_enc_T / ms_vae_T
    if ratio > MS_FACTOR or ratio < 1.0 / MS_FACTOR:
        return ratio, f"in very different ranges (overall mean square on set T {ratio:.4g} times the VAE group's)"
    return ratio, f"the overall ranges are comparable (overall mean square on set T {ratio:.4g} times the VAE group's)"


def ranges_report(stats):
    """stats: {"T": {group: ChannelStats}, "L": {...}} -> the amendment's report, with the full per-channel table."""
    rep = {"summary": {s: {g: group_summary(st) for g, st in stats[s].items()} for s in ("T", "L")},
           "mean_square_ratio_to_vae": {s: {e: stats[s][e].mean_square() / stats[s][VAE].mean_square()
                                           for e in ENCODERS if e in stats[s]} for s in ("T", "L")},
           "wording": {}, "flags": {}, "per_channel": {}}
    for e in ENCODERS:
        if e in stats["T"]:
            ratio, text = range_wording(stats["T"][e].mean_square(), stats["T"][VAE].mean_square())
            rep["wording"][e] = {"ratio_T": ratio, "text": text}
    for g in stats["T"]:
        T, L = stats["T"][g], stats["L"][g]
        rep["flags"][g] = flag_report(channel_flags(T.mean(), T.sd(), L.mean(), L.sd()), T.C)
        rep["per_channel"][g] = {"T": T.table(), "L": L.table()}
    return rep


# ---------------------------------------------------------------------------
# Assembling the statistics from per-task results (pure; used by run and by the tests)
# ---------------------------------------------------------------------------

def assemble(tasks):
    """tasks: [{suite, task_index, name, ratios: {group: task_ratios(...)}, live_suite}] -> statistics dict.
    Leaves out tasks with fewer than 2 training episodes or ref == 0 in any of the four groups (Design 4); the
    DINOv3-without-757 variant uses the same tasks."""
    kept, left_out = [], []
    for t in tasks:
        bad = [g for g in GROUPS if t["ratios"][g]["n_train"] < 2 or not t["ratios"][g]["ref"]]
        if bad:
            left_out.append({"suite": t["suite"], "task_index": t["task_index"], "name": t["name"],
                             "n_train": t["ratios"][VAE]["n_train"], "groups": bad})
        else:
            kept.append(t)
    if not kept:
        raise CheckFailed("every task was left out")
    for t in kept:
        if t["ratios"][NO757]["r"] is None:
            raise CheckFailed(f"{t['suite']} task {t['task_index']}: ref = 0 for DINOv3 without channel 757")
    allg = GROUPS + (NO757,)
    per_task_r = {g: [t["ratios"][g]["r"] for t in kept] for g in allg}
    m = {g: pooled_median(per_task_r[g]) for g in allg}
    R = {e: ratio_R(m[e], m[VAE]) for e in ENCODERS}
    R_no757 = ratio_R(m[NO757], m[VAE])
    boot = bootstrap(per_task_r, list(ENCODERS) + [NO757])
    readings = {e: read_encoder(R[e], boot[e]["interval"], R_no757 if e == DINO else None) for e in ENCODERS}
    frames = [{"suite": t["suite"], "task_index": t["task_index"], "init_state": s,
               "r": {g: t["ratios"][g]["r"][j] for g in allg}} for t in kept for j, s in enumerate(t["init_states"])]
    return {
        "n_tasks": len(kept), "n_live_frames": len(frames), "left_out_tasks": left_out,
        "m": m, "R": R, "R_dinov3_without_757": R_no757,
        "bootstrap": {"n_draws": N_BOOT, "seed": BOOT_SEED, "percentiles": list(CI), "resampling": "tasks with "
                      "replacement, all live frames of each drawn task, ref not recomputed",
                      "interval": {e: boot[e]["interval"] for e in boot}, "draws": {e: boot[e]["draws"] for e in boot}},
        "reading": readings,
        "outcome": outcome_case({e: readings[e]["reading"] for e in ENCODERS}),
        "r_quantiles": r_summaries(frames, allg),
        "per_task_ratio": [{"suite": t["suite"], "task_index": t["task_index"], "name": t["name"],
                            "instruction": t.get("instruction"),
                            **task_level_ratios({g: t["ratios"][g]["r"] for g in allg}, list(ENCODERS) + [NO757])}
                           for t in kept],
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Correlate (Mac): Spearman over tasks against the committed clean records
# ---------------------------------------------------------------------------

DIR = Path(__file__).resolve().parents[1]           # world-model-stacking-encoders/
RECORDS = {
    "none": ("experiments/2026-11-15-pretrained-start-baseline/results/clean-eval/records",
             "clean-none-s0-shard*of4-try1.jsonl"),
    **{e: (f"experiments/2026-11-15-pretrained-start-pairings/results/clean/{e}-s0/records",
           "clean-shard*of4-try1.jsonl") for e in ENCODERS},
}


def task_success(files):
    """{(suite, task_index): (successes, trials)} from clean rollout records (episode_id = task_index * 1000 + trial)."""
    succ, n, seen, instr = {}, {}, set(), {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["suite"], int(r["episode_id"]), int(r["eval_seed"]))
            assert key not in seen, key
            seen.add(key)
            k = (r["suite"], int(r["episode_id"]) // 1000)
            succ[k] = succ.get(k, 0) + int(r["success"])
            n[k] = n.get(k, 0) + 1
            instr[k] = r["instruction"]
    return {k: (succ[k], n[k]) for k in n}, instr


def correlate(result, records=None):
    """Per encoder: Spearman over the included tasks between the per-task ratio and the clean success drop
    (none minus the encoder, success rates over the task's 50 trials)."""
    records = records or {c: sorted((DIR / d).glob(p)) for c, (d, p) in RECORDS.items()}
    succ, instr = {}, {}
    for c, files in records.items():
        if len(files) != 4:
            raise CheckFailed(f"{c}: expected 4 record files, found {len(files)}")
        succ[c], instr[c] = task_success(files)
    rows = []
    for t in result["statistics"]["per_task_ratio"]:
        k = (t["suite"], int(t["task_index"]))
        for c in records:
            if k not in succ[c]:
                raise CheckFailed(f"{c}: no records for {k}")
            if t.get("instruction") and instr[c][k] != t["instruction"]:
                raise CheckFailed(f"{c} {k}: record instruction {instr[c][k]!r} differs from {t['instruction']!r}")
        rate = {c: succ[c][k][0] / succ[c][k][1] for c in records}
        rows.append({"suite": k[0], "task_index": k[1], "name": t["name"],
                     "success": {c: list(succ[c][k]) for c in records},
                     **{f"ratio_{e}": t[e] for e in ENCODERS}, **{f"drop_{e}": rate["none"] - rate[e] for e in ENCODERS}})
    out = {e: {"spearman": spearman([r[f"ratio_{e}"] for r in rows], [r[f"drop_{e}"] for r in rows]),
               "n_tasks": len(rows)} for e in ENCODERS}
    return {"spearman": out, "rows": rows,
            "records": {c: [str(Path(f).relative_to(DIR)) if str(f).startswith(str(DIR)) else str(f) for f in fs]
                        for c, fs in records.items()}}


# ---------------------------------------------------------------------------
# Cluster side (torch, simulator, caches)
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean_jobs(tasks_file, dry_run):
    """The live frames' jobs: rollout.make_jobs on the clean task list (trial = init state 0 to 4, eval seed 41).
    Dry run: the first task of each suite."""
    import rollout
    clean = json.loads(Path(tasks_file).read_text())["clean"]
    if dry_run:
        clean = {s: clean[s][:1] for s in SUITES}
    else:
        if sorted(clean) != sorted(SUITES) or any(len(clean[s]) != 10 for s in SUITES):
            raise CheckFailed(f"{tasks_file}: expected 10 clean tasks in each of {SUITES}")
    jobs = rollout.make_jobs(None, len(INIT_STATES), clean, clean_seed=EVAL_SEED)
    full = json.loads(Path(tasks_file).read_text())["clean"]
    for j in jobs:
        j["task_index"] = full[j["suite"]].index(j["name"])       # = episode_id // 1000 in the clean records
    return jobs


def live_worker_init(mujoco_gl):
    """As rollout.worker_main: GL backend from the environment, one torch thread."""
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
        os.environ["PYOPENGL_PLATFORM"] = mujoco_gl
    import torch
    torch.set_num_threads(1)


def live_frame(job):
    """One live first frame, in a simulator process. Seeds as rollout.run_episode (random, numpy, torch = eval seed,
    before the env is built; rollout.first_frame seeds random and numpy itself), then rollout.first_frame: libero_env,
    reset, set_init_state, the 30 dummy steps, process_obs."""
    import torch
    import rollout
    t0 = time.time()
    torch.manual_seed(int(job["eval_seed"]))
    f = rollout.first_frame(job)
    return {"suite": job["suite"], "name": job["name"], "trial": int(job["trial"]), "agent": f["agent"],
            "wrist": f["wrist"], "instruction": f["instruction"], "wall_s": time.time() - t0}


def load_stats(ckpt, enc):
    """feature_stats of one encoder from its single-encoder seed 0 checkpoint's config (model.pt, read as
    rollout.Policy.load reads it; mmap so the weights are not read)."""
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["config"]
    got = {"config": cfg.get("config"), "seed": cfg.get("seed"), "sources": cfg.get("sources"),
           "git_commit": cfg.get("git_commit"), "feature_stats_keys": sorted(cfg.get("feature_stats", {}))}
    if cfg.get("config") != enc or int(cfg.get("seed", -1)) != 0 or list(cfg["sources"][1:]) != [enc] \
            or enc not in cfg["feature_stats"]:
        raise CheckFailed(f"{ckpt}: not the {enc} single-encoder seed 0 checkpoint ({got})")
    mu, sd = cfg["feature_stats"][enc]
    return np.asarray(mu, np.float32), np.asarray(sd, np.float32), got


def frame0_episode(ep, fps=20):
    """The episode dict with its video refs cut after frame 0, so fastwam_data.decode_episode decodes up to frame 0 and
    stops (same seek, same decoder, same frame-0 pixels; checked against a full decode by frame0_decode_check)."""
    return {**ep, "length": 1, "video": {c: {**v, "to": v["from"] + 1.0 / fps} for c, v in ep["video"].items()}}


def train_frame0(root, ep, feats, vae_dirs):
    """Training frame 0 of one episode: the 224 frames (to_224), the one-frame VAE clip, the uint8 frames the encoders
    saw, the cached encoder features [0] and the cached clip-encoded VAE latent [0, 0]."""
    import torch
    import fastwam_data as FD
    frames = FD.decode_episode(root, frame0_episode(ep))
    cams224 = {c: FD.to_224(frames[c]) for c in FD.CAMS}
    clip = FD.video_clips(cams224, np.array([0]), num_frames=1)            # (1, 3, 1, 224, 448) in [-1, 1]
    u8 = {c: (cams224[c] * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
          for c in FD.CAMS}
    s = ep["suite"]
    e = ep["episode_index"]
    enc = {}
    for g in ENCODERS:
        p = Path(feats) / g / s / f"ep{e:06d}.npy"
        if not p.exists():
            raise CheckFailed(f"missing cache {p}")
        a = np.load(p, mmap_mode="r")
        if a.shape[0] != ep["length"] or tuple(a.shape[1:3]) != GRID:
            raise CheckFailed(f"{p}: shape {a.shape}, episode length {ep['length']}")
        enc[g] = np.array(a[0])                                               # (14, 28, D) float16
    found = [d for d in vae_dirs if (Path(feats) / d / s / f"ep{e:06d}.npy").exists()]
    if not found:
        raise CheckFailed(f"no cached VAE latents for {s} episode {e} in {vae_dirs}")
    a = np.load(Path(feats) / found[0] / s / f"ep{e:06d}.npy", mmap_mode="r")
    if a.shape[0] != ep["length"] or tuple(a.shape[1:]) != (3,) + GRID + (48,):
        raise CheckFailed(f"cached VAE {s} ep {e}: shape {a.shape}, episode length {ep['length']}")
    return {"clip": clip, "u8": u8, "enc": enc, "vae_cached": np.array(a[0, 0]), "vae_dir": found[0]}


def frame0_decode_check(root, ep):
    """The frame-0 shortcut against the full training decode: decode_episode(frame 0 only) equals decode_episode()[0],
    and to_224 of the one frame equals to_224 of the whole episode at index 0, both bit for bit."""
    import torch
    import fastwam_data as FD
    full = FD.decode_episode(root, ep)
    one = FD.decode_episode(root, frame0_episode(ep))
    out = {}
    for c in FD.CAMS:
        pix = bool(np.array_equal(full[c][0], one[c][0]))
        r224 = bool(torch.equal(FD.to_224(full[c])[0], FD.to_224(one[c])[0]))
        out[c] = {"pixels_equal": pix, "to_224_equal": r224}
    return out


def save_side_by_side(path, live_agent, live_wrist, tr_agent, tr_wrist, top, bottom):
    """PNG: top row the live frame (agentview | wrist), bottom row the training frame 0, 224 px each, labelled."""
    from PIL import Image, ImageDraw
    pad = 18
    img = Image.new("RGB", (448, 2 * (224 + pad)), "white")
    d = ImageDraw.Draw(img)
    for row, (a, w, text) in enumerate(((live_agent, live_wrist, top), (tr_agent, tr_wrist, bottom))):
        y = row * (224 + pad)
        d.text((4, y + 3), text, fill="black")
        img.paste(Image.fromarray(np.ascontiguousarray(a)), (0, y + pad))
        img.paste(Image.fromarray(np.ascontiguousarray(w)), (224, y + pad))
    img.save(path)


def run(args):
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor
    import torch
    import fastwam_data as FD
    import proxy_wam as PW
    import rollout
    from extract_features import REVISIONS

    t_start = time.time()
    log = lambda m: print(f"[ffc {time.time() - t_start:7.0f}s] {m}", flush=True)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise CheckFailed(f"{out_dir} exists: nothing is overwritten; use a new --out-dir")
    out_dir.mkdir(parents=True)
    if args.dry_run:
        (out_dir / "frames").mkdir()
    ckpt = dict(kv.split("=", 1) for kv in args.ckpt)
    ckpt_sha = dict(kv.split("=", 1) for kv in args.ckpt_sha)
    if sorted(ckpt) != sorted(ENCODERS) or sorted(ckpt_sha) != sorted(ENCODERS):
        raise CheckFailed(f"--ckpt and --ckpt-sha need exactly {ENCODERS}")
    if args.lp_commit != LP_COMMIT:
        raise CheckFailed(f"LIBERO-Plus commit {args.lp_commit}, the plan pins {LP_COMMIT}")
    device = args.device
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    if "L4" not in gpu:
        raise CheckFailed(f"GPU {gpu!r}: the plan needs the evaluation card, an NVIDIA L4")
    timing, checks = {}, {}
    pins = {"plan": PLAN, "src_git_commit": os.environ.get("SRC_GIT_COMMIT"), "image": os.environ.get("IMAGE"),
            "libero_plus_commit": args.lp_commit, "data_root": args.data_root, "feats": args.feats,
            "tasks_file": args.tasks, "tasks_file_sha256": sha256_file(args.tasks), "vae_path": args.vae_path,
            "checkpoints": {e: {"path": ckpt[e], "sha256_checked_in_pod": ckpt_sha[e]} for e in ENCODERS},
            "encoder_revisions": {e: REVISIONS[e] for e in ENCODERS}, "enc_dtype": {e: str(rollout.ENC_DTYPE[e])
                                                                                   for e in ENCODERS},
            "gpu": gpu, "torch": torch.__version__, "eval_seed": EVAL_SEED, "init_states": list(INIT_STATES),
            "mode": "dry-run" if args.dry_run else "full", "python": sys.version.split()[0]}

    # -- statistics from each encoder's own single-encoder seed 0 checkpoint
    fstats = {}
    for e in ENCODERS:
        mu, sd, got = load_stats(ckpt[e], e)
        fstats[e] = (torch.tensor(mu, device=device), torch.tensor(sd, device=device))
        pins["checkpoints"][e]["config"] = got
        log(f"feature_stats {e}: {len(mu)} channels from {ckpt[e]} ({got['config']} seed {got['seed']}, "
            f"commit {got['git_commit']})")
    if DINO_757 >= len(fstats[DINO][0]):
        raise CheckFailed("DINOv3 has no channel 757")

    # -- step 1: live frames in simulator processes, then the Featurizer at batch 1
    jobs = clean_jobs(args.tasks, args.dry_run)
    log(f"{len(jobs)} live frames ({len({(j['suite'], j['name']) for j in jobs})} tasks x {len(INIT_STATES)} init "
        f"states) on {args.sim_workers} simulator processes")
    t = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.sim_workers, initializer=live_worker_init, initargs=(os.environ.get("MUJOCO_GL", "egl"),)) as pool:
        lives = list(pool.imap(live_frame, jobs, chunksize=1))
    timing["live_frames_s"] = time.time() - t
    timing["live_frame_wall_s"] = quantiles([f["wall_s"] for f in lives])
    log(f"live frames built in {timing['live_frames_s']:.0f}s (per frame median {timing['live_frame_wall_s']['median']:.1f}s)")
    for f, j in zip(lives, jobs):
        f["task_index"] = j["task_index"]
        if f["instruction"] != rollout.base_language(j["name"]):
            raise CheckFailed(f"live instruction {f['instruction']!r} for {j['name']}")

    t = time.time()
    feat = rollout.Featurizer(list(ENCODERS), args.vae_path, device)
    timing["featurizer_load_s"] = time.time() - t
    loaded = {s: getattr(enc, "loaded_commit", None) for s, enc in feat.encs}
    pins["encoder_loaded_commits"] = loaded
    if any(loaded[e] != REVISIONS[e] for e in ENCODERS):
        raise CheckFailed(f"loaded encoder commits {loaded} differ from the pinned {REVISIONS}")
    log(f"Featurizer loaded in {timing['featurizer_load_s']:.0f}s; encoder commits {loaded}")

    def zscore(f, e):              # as rollout.Policy.act and proxy_wam.to_device: float32 on the device
        mu, sd = fstats[e]
        return ((f - mu) / sd).permute(0, 3, 1, 2)[0].cpu().numpy()               # (D, 14, 28)

    t = time.time()
    live = {}                                          # (suite, task_index) -> {group: (5, C, 14, 28)}
    for f in lives:
        outs = feat(f["agent"][None], f["wrist"][None])                            # batch 1
        g = {VAE: outs[0][0, :, 0].cpu().numpy()}                                  # (48, 14, 28)
        for e, o in zip(ENCODERS, outs[1:]):
            g[e] = zscore(o, e)
        d = live.setdefault((f["suite"], f["task_index"]), {k: [] for k in GROUPS})
        for k in GROUPS:
            d[k].append(g[k])
    for d in live.values():
        for k in GROUPS:
            d[k] = np.stack(d[k])
    timing["live_featurize_s"] = time.time() - t
    shapes = {k: list(next(iter(live.values()))[k].shape[1:]) for k in GROUPS}
    checks["live_shapes"] = shapes
    log(f"live features at batch 1 in {timing['live_featurize_s']:.0f}s; per-frame shapes {shapes}")

    # -- step 2: training frame 0s of the same tasks
    tasks = sorted({(f["suite"], f["task_index"], f["name"]) for f in lives})
    per_task, lat_rows, stats = [], [], {"T": {}, "L": {}}
    for k in GROUPS:
        C = shapes[k][0]
        stats["T"][k], stats["L"][k] = ChannelStats(C), ChannelStats(C)
    live_757 = []
    checks["frame0_decode"] = {}
    checks["train_episodes"] = {}
    pool = ThreadPoolExecutor(max_workers=args.read_threads)
    t_train = time.time()
    n_eps = 0
    for s in SUITES:
        root = Path(args.data_root) / f"{s}_no_noops_lerobot"
        eps = FD.read_suite(root)
        meta = np.load(Path(args.feats) / "meta" / f"{s}.npz")
        by_id = {e["episode_index"]: e for e in eps}
        ids = [int(x) for x in meta["episode_ids"]]
        if sorted(ids) != sorted(by_id) or len(ids) != len(set(ids)):
            raise CheckFailed(f"{s}: meta episode ids differ from the dataset's")
        for i, task, n in zip(ids, meta["tasks"], meta["lengths"]):
            if str(task) != by_id[i]["task"] or int(n) != by_id[i]["length"]:
                raise CheckFailed(f"{s} ep {i}: meta task/length differ from the dataset's")
        checks["train_episodes"][s] = len(ids)
        suite_tasks = [x for x in tasks if x[0] == s]
        first = True
        for _, ti, name in suite_tasks:
            instr = rollout.base_language(name)
            tep = [{**by_id[i], "suite": s} for i, task in zip(ids, meta["tasks"]) if str(task) == instr]
            tep.sort(key=lambda e: e["episode_index"])
            if first and tep:
                chk = frame0_decode_check(root, tep[0])
                checks["frame0_decode"][f"{s} ep{tep[0]['episode_index']}"] = chk
                if not all(v["pixels_equal"] and v["to_224_equal"] for v in chk.values()):
                    raise CheckFailed(f"frame-0 decode differs from the full decode: {s} {chk}")
                first = False
            futs = [pool.submit(train_frame0, root, e, args.feats, PW.VAE_DIRS) for e in tep]
            tr = {k: [] for k in GROUPS}
            png_done = False
            for e, fu in zip(tep, futs):
                x = fu.result()
                with torch.no_grad():
                    z = feat.vae.model.encode(x["clip"].to(device, torch.bfloat16), feat.vae.scale)
                if tuple(z.shape[1:]) != (48, 1) + GRID:
                    raise CheckFailed(f"single-frame VAE latent shape {tuple(z.shape)}")
                v = z.float().half().float()[0, :, 0].cpu().numpy()                # (48, 14, 28)
                tr[VAE].append(v)
                ma, rl = latent_diff(v, np.transpose(x["vae_cached"].astype(np.float32), (2, 0, 1)))
                lat_rows.append({"suite": s, "episode": e["episode_index"], "vae_dir": x["vae_dir"],
                                 "max_abs_diff": ma, "rel_l2": rl})
                for g in ENCODERS:
                    tr[g].append(zscore(torch.from_numpy(x["enc"][g]).to(device).float()[None], g))
                if args.dry_run and not png_done:
                    lf = next(f for f in lives if f["suite"] == s and f["task_index"] == ti and f["trial"] == 0)
                    p = out_dir / "frames" / f"{s}-task{ti}-{name[:50]}.png"
                    save_side_by_side(p, lf["agent"], lf["wrist"], x["u8"][FD.CAMS[0]], x["u8"][FD.CAMS[1]],
                                      f"LIVE {s} task {ti} init 0: agentview | wrist (224, process_obs)",
                                      f"TRAINING {s} ep {e['episode_index']} frame 0: agentview | wrist (to_224, uint8)")
                    log(f"saved {p}")
                    png_done = True
                n_eps += 1
            tr = {k: np.stack(v) if v else np.zeros((0,) + tuple(shapes[k])) for k, v in tr.items()}
            lv = live[(s, ti)]
            ratios = {g: task_ratios(tr[g], lv[g]) for g in GROUPS}
            ratios[NO757] = task_ratios(drop_channel(tr[DINO], DINO_757), drop_channel(lv[DINO], DINO_757))
            entry = {"suite": s, "task_index": ti, "name": name, "instruction": instr,
                     "train_episodes": [e["episode_index"] for e in tep], "init_states": list(INIT_STATES),
                     "ratios": ratios}
            per_task.append(entry)
            included = all(ratios[g]["n_train"] >= 2 and ratios[g]["ref"] for g in GROUPS)
            if included:                                # a left-out task is left out of both range sets too
                for g in GROUPS:
                    stats["T"][g].add(tr[g])
                    stats["L"][g].add(lv[g])
                live_757.append(lv[DINO][:, DINO_757])
            log(f"{s} task {ti} ({name[:40]}): {len(tep)} training episodes; {n_eps} episodes so far "
                f"({time.time() - t_train:.0f}s)")
    pool.shutdown()
    timing["train_s"] = time.time() - t_train
    timing["train_episodes"] = n_eps
    checks["n_live_frames"] = len(lives)
    checks["n_tasks"] = len(per_task)

    # -- statistics
    st = assemble(per_task)
    st["dinov3_757_live_z"] = channel_757_live(np.concatenate(live_757)) if live_757 else None
    st["vae_single_vs_clip"] = {"summary": latent_diff_summary(lat_rows), "per_episode": lat_rows}
    st["ranges"] = ranges_report(stats)
    st["per_task"] = [{k: v for k, v in t.items()} for t in per_task]
    timing["total_s"] = time.time() - t_start
    result = {"check": "first-frame feature check", "pins": pins, "pipeline_checks": checks, "timing": timing,
              "statistics": st}
    text = json.dumps(result, allow_nan=False)
    if args.dry_run:      # no number of the check is written or printed: pipeline checks, shapes and timings only
        red = {"check": result["check"], "pins": pins, "pipeline_checks": checks, "timing": timing,
               "dry_run_note": "statistics computed end to end and withheld (plan Hand-off: no number of this check "
                               "is read before the full run)", "statistics_keys": sorted(st),
               "statistics_json_bytes": len(text)}
        text = json.dumps(red, indent=1, allow_nan=False)
    (out_dir / "result.json").write_text(text)
    log(f"wrote {out_dir / 'result.json'} ({len(text)} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    if not args.dry_run:
        print_summary(result)
    return 0


def print_summary(result):
    st = result["statistics"]
    print("=== SUMMARY (plan reading rule; descriptive items below it carry no decision)")
    print(f"tasks {st['n_tasks']}, live frames {st['n_live_frames']}, left out: {st['left_out_tasks'] or 'none'}")
    print("m: " + ", ".join(f"{g} {v:.4f}" for g, v in st["m"].items()))
    for e, rd in st["reading"].items():
        lo, hi = rd["interval"]
        extra = f"; without channel 757 R {rd['R_without_757']:.4f} ({rd['zone_without_757']})" if "R_without_757" in rd else ""
        print(f"R({e}) = {rd['R']:.4f} [{lo:.4f}, {hi:.4f}]: {rd['wording']}{extra}; reading '{rd['reading']}'")
    print(st["outcome"]["text"])
    z = st["dinov3_757_live_z"]
    if z:
        print(f"DINOv3 channel 757 live z: median {z['median']:.4g}, max |z| {z['max_abs']:.4g}")
    v = st["vae_single_vs_clip"]["summary"]["all"]
    print(f"VAE single-frame vs cached clip-encoded [0, 0]: max abs diff {v['max_abs_diff_max']:.4g}, rel L2 median "
          f"{v['rel_l2_median']:.4g} max {v['rel_l2_max']:.4g} over {v['n_episodes']} episodes "
          f"({v['n_identical']} identical)")
    for e, w in st["ranges"]["wording"].items():
        print(f"ranges {e}: {w['text']}")
    for g, fl in st["ranges"]["flags"].items():
        print(f"flags {g}: " + ", ".join(f"{k} {x['count']} ({x['share']:.1%})" for k, x in fl.items()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="Plan: " + PLAN + "; steps: experiments/2026-11-15-pretrained-start-pairings/"
                                       "jobs/RUN-first-frame-check.md")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="the check, on the cluster (L4)")
    r.add_argument("--data-root", required=True, help="lerobot_v30 folder with <suite>_no_noops_lerobot")
    r.add_argument("--feats", required=True, help="extract_fastwam cache root (meta/, vae/, vae_b/, <encoder>/)")
    r.add_argument("--tasks", required=True, help="rollout.py instructions --tasks-out file (its 'clean' task names)")
    r.add_argument("--vae-path", required=True)
    r.add_argument("--ckpt", action="append", required=True, metavar="ENC=PATH",
                   help="model.pt of the encoder's single-encoder seed 0 run, once per encoder")
    r.add_argument("--ckpt-sha", action="append", required=True, metavar="ENC=SHA256",
                   help="the sha256 the pod checked for that model.pt (recorded in the pins)")
    r.add_argument("--lp-commit", required=True, help="LIBERO-Plus commit of the environment (setup_sim.sh LP_COMMIT)")
    r.add_argument("--out-dir", required=True, help="new folder (refused if it exists)")
    r.add_argument("--dry-run", action="store_true", help="one task per suite, PNGs, no statistic written or printed")
    r.add_argument("--sim-workers", type=int, default=3)
    r.add_argument("--read-threads", type=int, default=2, help="threads decoding and reading training frame 0s")
    r.add_argument("--device", default="cuda")
    c = sp.add_parser("correlate", help="Mac: Spearman of per-task ratios against the committed clean records")
    c.add_argument("--result", required=True, help="result.json of the full run")
    c.add_argument("--out", default=None, help="write the correlation JSON here (default: print only)")
    args = p.parse_args(argv)
    try:
        if args.cmd == "run":
            return run(args)
        res = json.loads(Path(args.result).read_text())
        if res.get("pins", {}).get("mode") != "full":
            raise CheckFailed("correlate needs the full run's result.json")
        out = correlate(res)
        out["result_sha256"] = sha256_file(args.result)
        text = json.dumps(out, indent=1, allow_nan=False)
        if args.out:
            Path(args.out).write_text(text + "\n")
        for e, v in out["spearman"].items():
            print(f"Spearman {e}: per-task ratio vs clean success drop (none minus {e}), {v['n_tasks']} tasks: "
                  f"{v['spearman']}")
        return 0
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", flush=True)
        return 3
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
