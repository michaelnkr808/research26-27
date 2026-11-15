"""Sizing check for a pretrained-start world action model (experiments/2026-11-12-pretrained-start-sizing/PLAN.md,
Steps 4-6; loading facts from step2-reading.md in the same folder, which overrides the PLAN where they differ).

What it does: builds proxy_wam's FastWAMJoint (Fast-WAM 7faa711) at the full Wan 2.2 TI2V-5B width (video expert 3072
wide, FFN 14336, 24 heads of 128; action expert 1024 wide, FFN 4096) with only the kept blocks (default 0-7, the
user's choice, tagged "first 8, user's choice"), loads the Wan 2.2 DiT weights by name with its own safetensors
reader (only the shards that hold the kept blocks; Fast-WAM's loader can't take a subset or widen the input), widens
the patch embedding and the output head for an extra encoder's channels, and measures peak GPU memory and seconds per
step on real cached batches, under the proxy's recipe (fp32 weights, bf16 autocast, fp32 AdamW, grad clip 1.0).

Parts not loaded from Wan (Open item 5, the user's choice 2026-11-12; cost does not depend on it):
  - extra-channel slice of the widened patch embedding (input columns) and head (output rows, weight and bias): zero,
    so the widened model starts as the pretrained one on the VAE channels and predicts 0 for the extra channels;
  - action expert (blocks, embeddings, action encoder and head) and proprio encoder: fresh, as proxy_wam.build_model
    builds them (each module's own init, seeded). Fast-WAM itself derives its action expert from the Wan DiT by
    interpolation (step2-reading.md Q4); that is not done here.

Process layout (PLAN Step 6): every micro-batch size and the full-optimizer-step run is its own Python process. The
`launch` mode runs them in order and rewrites results/measure-<config>.json after each one ends:
  sweep m=1 (--from-wan): build, load Wan, write load-<config>.json and sanity-<config>.json, one optimizer step,
      save model + AdamW state to --state-dir (local disk), then measure;
  sweep m=2, 4, ... 128: load the saved state, one optimizer step at m, measure; stop at the first oom or error;
  optstep at m*: global batch 128 as 128/m* accumulated micro-batches, 1 warm-up + 2 timed optimizer steps;
      repeated once at m*/2 in a fresh process if it ends in oom or its peak is above 0.9 x total memory.
      m* is picked from the larger of each m's sweep peak and that process's own full-optimizer-step peak.
Each process ends as ok, oom (only torch.cuda.OutOfMemoryError) or error (type and message), never confused.

Example (inside the 3,600 s inner limit, as the Modal app runs it):
  timeout --signal=TERM --kill-after=60 3600 python pretrained_sizing.py launch --feats /vol/data/sizing-slice \
      --wan-dir /vol/wan/Wan2.2-TI2V-5B --configs none,dinov3 --out /vol/results/run1 --state-dir /tmp/sizing-state
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import proxy_wam as P
from diagnostics import git_commit

FASTWAM_COMMIT = "7faa71108368fbb3b6885649f112af607427a2d4"
WAN_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_INDEX = "diffusion_pytorch_model.safetensors.index.json"
WAN_CONFIG = "config.json"
WAN_LAYERS = 30
WAN_KW = {"hidden": 3072, "action_hidden": 1024, "heads": 24, "head_dim": 128}   # step2-reading.md Q3
KEEP_DEFAULT = "0-7"
KEEP_TAG = "first 8, user's choice"
SWEEP_MS = (1, 2, 4, 8, 16, 32, 64, 128)
GLOBAL_BATCH = 128            # proxy_wam.py --batch-size
BUFFER_EPISODES = 16          # proxy_wam.py --buffer-episodes
WARMUP, TIMED = 3, 10         # sweep micro-steps
OPT_WARMUP, OPT_TIMED = 1, 2  # full optimizer steps
FIT_FRACTION = 0.9            # PLAN "Decision rule": fits if peak reserved <= 0.9 x reported total memory
WIDENED = ("patch_embedding.weight", "head.head.weight", "head.head.bias")
EXPECTED_NEW = ("action_expert.", "proprio_encoder.")
INIT_NOTE = {
    "partly loaded": "VAE-channel slice (first 48 input channels of patch_embedding, first 48 output channels per "
                     "patch position of head.head) copied from Wan 2.2; extra-channel slice set to zero (user's "
                     "choice 2026-11-12, PLAN Open item 5)",
    "new": "fresh, as proxy_wam.build_model builds them (each module's own PyTorch init, after torch.manual_seed(seed)); "
           "Fast-WAM itself derives the action expert from the Wan DiT by linear interpolation with alpha scaling "
           "(step2-reading.md Q4), not done here; the cost does not depend on it",
}


class LoadError(RuntimeError):
    """The Wan weights did not load as intended (PLAN Step 4 stops here)."""


class SanityError(RuntimeError):
    """A sanity item failed (non-finite loss at start; PLAN Step 5 stops here)."""


# ---------------------------------------------------------------------------
# Model and loading (Step 4)
# ---------------------------------------------------------------------------

def parse_keep(s):
    """'0-7' -> [0..7]; '0,4,8' -> [0, 4, 8]."""
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    if len(set(out)) != len(out) or any(i < 0 for i in out):
        raise ValueError(f"bad --keep {s!r}")
    return out


def build(in_dim, device, n_layers, model_kw, seed):
    """proxy_wam.build_model at the given width with n_layers in both experts, parameters created on `device`."""
    torch.manual_seed(seed)
    with torch.device(device):
        return P.build_model(in_dim, device, layers=n_layers, **model_kw)


def source_name_map(ckpt_names, keep):
    """{checkpoint name: model name under video_expert, or None for a block that is not kept}. Wan's names equal
    WanVideoDiT's (step2-reading.md Q2); the only mapping is block index src -> position in `keep`."""
    pos = {src: dst for dst, src in enumerate(keep)}
    out = {}
    for n in ckpt_names:
        if n.startswith("blocks."):
            _, i, rest = n.split(".", 2)
            out[n] = f"blocks.{pos[int(i)]}.{rest}" if int(i) in pos else None
        else:
            out[n] = n
    return out


def _check_config(cfg, video, n_vae):
    want = {"dim": video.hidden_dim, "ffn_dim": video.blocks[0].ffn_dim, "freq_dim": video.freq_dim,
            "num_heads": video.num_heads, "in_dim": n_vae, "out_dim": n_vae}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise LoadError(f"checkpoint config.json differs from the built model (checkpoint, model): {bad}")


def _widened_views(p, name, src_shape, n_vae, patch):
    """(loaded slice of p, extra slice of p, shape to view the 48-channel source as) for a widened parameter.
    patch_embedding.weight: (H, C_in, 1, 2, 2), input channels on dim 1. head.head: rows ordered (x y z c), c innermost
    (WanVideoDiT.unpatchify), so rows view as (patch positions, C_out)."""
    src_shape = tuple(src_shape)
    if name == "patch_embedding.weight":
        if src_shape[1] != n_vae or p.shape[0] != src_shape[0] or tuple(p.shape[2:]) != src_shape[2:] \
                or p.shape[1] <= n_vae:
            raise LoadError(f"{name}: cannot widen source {src_shape} into {tuple(p.shape)}")
        return p[:, :n_vae], p[:, n_vae:], src_shape
    k = math.prod(patch)
    rest = tuple(p.shape[1:])
    if src_shape[0] != n_vae * k or p.shape[0] % k or p.shape[0] // k <= n_vae or src_shape[1:] != rest:
        raise LoadError(f"{name}: cannot widen source {src_shape} into {tuple(p.shape)}")
    pv = p.view((k, p.shape[0] // k) + rest)
    return pv[:, :n_vae], pv[:, n_vae:], (k, n_vae) + rest


@torch.no_grad()
def verify_against_file(model, wan_dir, entries, n_vae=P.VAE_DIM):
    """Re-read every loaded or partly loaded tensor from its safetensors file, cast as Fast-WAM's loader casts on read
    (value.to(torch_dtype), helpers/io.py:131-135, torch_dtype = the parameter's dtype), and compare with the parameter
    as it now is: max |difference| on the loaded part, max |value| on a widened parameter's extra slice. Writes both
    into `entries`; raises LoadError if any is nonzero."""
    from safetensors import safe_open
    wan_dir = Path(wan_dir)
    index = json.loads((wan_dir / WAN_INDEX).read_text())["weight_map"]
    params = dict(model.named_parameters())
    patch = tuple(model.video_expert.patch_size)
    by_shard = {}
    for n, e in entries.items():
        if e["tag"] in ("loaded", "partly loaded"):
            by_shard.setdefault(index[e["source"]], []).append(n)
    for sh, ns in sorted(by_shard.items()):
        with safe_open(str(wan_dir / sh), framework="pt", device="cpu") as f:
            for n in ns:
                e, p = entries[n], params[n].detach()
                ref = f.get_tensor(e["source"]).to(p.dtype)
                if e["tag"] == "loaded":
                    e["max_abs_diff"] = float((p - ref.to(p.device)).abs().max())
                else:
                    lv, xv, shape = _widened_views(p, n[len("video_expert."):], ref.shape, n_vae, patch)
                    e["max_abs_diff"] = float((lv - ref.view(shape).to(p.device)).abs().max())
                    e["extra_slice_max_abs"] = float(xv.abs().max())
                e["checked_against"] = "tensor re-read from the safetensors file, cast to the parameter dtype"
    bad = {n: e for n, e in entries.items() if e["tag"] != "new"
           and (e.get("max_abs_diff") != 0 or e.get("extra_slice_max_abs", 0) != 0)}
    if bad:
        raise LoadError(f"nonzero difference after loading: {list(bad.items())[:10]}")


@torch.no_grad()
def load_wan(model, wan_dir, keep, n_vae=P.VAE_DIM, log=print):
    """Load the Wan 2.2 DiT into model.video_expert by name, reading only the shards that hold the kept blocks and
    the non-block tensors. Returns the Step 4 report. Raises LoadError on: a checkpoint tensor (kept block or
    non-block) with no model parameter, a video-expert parameter with no checkpoint tensor, a shape mismatch other
    than the planned widening, a nonzero difference after copying, or an unexpected new parameter."""
    from safetensors import safe_open
    wan_dir = Path(wan_dir)
    index = json.loads((wan_dir / WAN_INDEX).read_text())["weight_map"]
    video = model.video_expert
    _check_config(json.loads((wan_dir / WAN_CONFIG).read_text()), video, n_vae)
    if len(keep) != len(video.blocks):
        raise LoadError(f"keep {keep} has {len(keep)} blocks, model has {len(video.blocks)}")
    n_src_blocks = 1 + max(int(n.split(".")[1]) for n in index if n.startswith("blocks."))
    if max(keep) >= n_src_blocks:
        raise LoadError(f"keep {keep} reaches past the checkpoint's {n_src_blocks} blocks")
    if dict(video.named_buffers()):
        raise LoadError(f"video expert has buffers the report does not cover: {list(dict(video.named_buffers()))}")
    params = dict(video.named_parameters())
    names = source_name_map(index, keep)
    used = {s: d for s, d in names.items() if d is not None}
    unexpected = sorted(s for s, d in used.items() if d not in params)
    if unexpected:
        raise LoadError(f"{len(unexpected)} checkpoint tensors have no model parameter: {unexpected[:20]}")
    missing = sorted(set(params) - set(used.values()))
    if missing:
        raise LoadError(f"{len(missing)} video-expert parameters have no checkpoint tensor: {missing[:20]}")
    shards = sorted({index[s] for s in used})
    for sh in shards:
        if not (wan_dir / sh).exists():
            raise LoadError(f"shard {sh} is needed for the kept blocks but is not in {wan_dir}")
    patch = tuple(video.patch_size)
    entries = {}
    for sh in shards:
        t0 = time.time()
        with safe_open(str(wan_dir / sh), framework="pt", device="cpu") as f:
            in_file = set(f.keys())
            for s in sorted(s for s in used if index[s] == sh):
                if s not in in_file:
                    raise LoadError(f"{s} is listed in the index for {sh} but is not in the file")
                src = f.get_tensor(s)
                d = used[s]
                p = params[d]
                # Fast-WAM's loader casts every tensor to the requested dtype on read (io.py:131-135); fp32 -> fp32
                # here, so this is an exact copy
                src_c = src.to(device=p.device, dtype=p.dtype)
                e = {"source": s, "shape": list(p.shape), "source_shape": list(src.shape),
                     "source_dtype": str(src.dtype).replace("torch.", ""), "dtype": str(p.dtype).replace("torch.", "")}
                if tuple(p.shape) == tuple(src.shape):
                    p.copy_(src_c)
                    e.update(tag="loaded")
                elif d in WIDENED:
                    lv, xv, shape = _widened_views(p, d, src.shape, n_vae, patch)
                    xv.zero_()
                    lv.copy_(src_c.view(shape))
                    e.update(tag="partly loaded", loaded_elements=src.numel(), new_elements=p.numel() - src.numel(),
                             init=INIT_NOTE["partly loaded"])
                else:
                    raise LoadError(f"shape mismatch: checkpoint {s} {tuple(src.shape)} vs model {d} {tuple(p.shape)}")
                entries["video_expert." + d] = e
                del src, src_c
        log(f"[load] read {sh} in {time.time() - t0:.1f}s")
    # independent check: re-read every used tensor from its file and compare with the parameter as loaded
    t0 = time.time()
    verify_against_file(model, wan_dir, entries, n_vae)
    log(f"[load] every loaded tensor re-read from its file and compared (max |diff| 0) in {time.time() - t0:.1f}s")
    for n, p in model.named_parameters():
        if n in entries:
            continue
        if not n.startswith(EXPECTED_NEW):
            raise LoadError(f"parameter {n} is neither loaded from Wan nor an expected new part {EXPECTED_NEW}")
        entries[n] = {"tag": "new", "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", ""),
                      "init": INIT_NOTE["new"]}
    skipped = [{"source": s, "tag": "skipped", "reason": f"block {s.split('.')[1]} not kept"}
               for s, d in sorted(names.items()) if d is None]
    plist = dict(model.named_parameters())
    counts = {t: {"tensors": 0, "elements": 0} for t in ("loaded", "partly loaded", "new")}
    for n, e in entries.items():
        counts[e["tag"]]["tensors"] += 1
        counts[e["tag"]]["elements"] += plist[n].numel()
    pl = [e for e in entries.values() if e["tag"] == "partly loaded"]
    counts["partly loaded"]["loaded_elements"] = sum(e["loaded_elements"] for e in pl)
    counts["partly loaded"]["new_elements"] = sum(e["new_elements"] for e in pl)
    counts["skipped"] = {"tensors": len(skipped)}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"wan_dir": str(wan_dir), "model_id": WAN_ID, "shards_read": shards,
            "shards_not_read": sorted(set(index.values()) - set(shards)), "keep": keep,
            "n_checkpoint_tensors": len(index), "counts": counts, "total_params": total,
            "trainable_params": trainable, "all_loaded_diffs_zero": True, "init": INIT_NOTE,
            "parameters": entries, "skipped": skipped}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(feats, config, t5=None, suites=None):
    feats = Path(feats)
    suites = suites or sorted(p.stem for p in (feats / "meta").glob("*.npz"))
    data = P.Data(feats, suites, P.config_sources(config), Path(t5) if t5 else feats / "t5")
    return data, suites


def read_all(data):
    """Every episode of the slice in RAM, as the Loader's reader threads cache them. Returns (cache, seconds)."""
    t = time.time()
    cache = {}
    for s, e, _, _ in data.episodes:
        cache[s, e] = {src: data.load_file((s, e), src) for src in [P.VAE] + data.sources}
    return cache, time.time() - t


def host_batches(data, cache, m, n, seed):
    """n host batches of m windows from the proxy's plan (seed, 16-episode buffers), assembled before any timing.
    Returns (batches, seconds per batch)."""
    lengths = [n_ for _, _, n_, _ in data.episodes]
    epochs = math.ceil(n * m / max(1, sum(lengths) - m)) + 1
    plan = P.make_plan(lengths, epochs, m, BUFFER_EPISODES, seed)[:n]
    if len(plan) < n:
        raise ValueError(f"slice too small for {n} batches of {m}")
    out, secs = [], []
    for b in plan:
        t = time.time()
        out.append(data.assemble(b, {data.episodes[ei][:2]: cache[data.episodes[ei][:2]] for ei in np.unique(b[:, 0])}))
        secs.append(time.time() - t)
    return out, secs


# ---------------------------------------------------------------------------
# Sanity (Step 5)
# ---------------------------------------------------------------------------

def _chan_stats(x):
    """Per-channel mean and std over (B, T, H, W) of x (B, C, T, H, W)."""
    f = x.float().transpose(0, 1).reshape(x.shape[1], -1)
    return f.mean(1), f.std(1)


@torch.no_grad()
def sanity(model, hb, bt, data, seed, amp):
    """Inputs inspected and loss at start vs the predict-zero baseline on the same batch and noise draws."""
    slices = data.channel_slices()
    rep = {"inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")} for k, v in bt.items()}}
    lat = bt["latents"]
    for src, sl in slices.items():
        mu, sd = _chan_stats(lat[:, sl])
        rep[f"channels/{src}"] = {"n": sl.stop - sl.start, "mean": mu.tolist(), "std": sd.tolist(),
                                  "mean_range": [float(mu.min()), float(mu.max())],
                                  "std_range": [float(sd.min()), float(sd.max())]}
    for k in ("action", "proprio"):
        rep[f"{k}_range"] = [float(bt[k].min()), float(bt[k].max())]
        rep[f"{k}_abs_gt1_fraction"] = float((bt[k].abs() > 1).float().mean())
    rep["padded_action_steps"] = int(bt["action_is_pad"].sum())
    rep["padded_latent_steps"] = int(bt["latent_is_pad"].sum())
    rep["tasks"] = [data.tasks[i] for i in hb["task"]]
    dev = lat.device.type
    losses = {}
    for name in ("model", "predict_zero"):
        if name == "predict_zero":
            model._joint_denoise_core = lambda **kw: (torch.zeros_like(kw["latents_video"]),
                                                      torch.zeros_like(kw["latents_action"]))
        try:
            torch.manual_seed(seed + 1)           # same noise and timesteps for both
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
                loss, parts = P.training_loss(model, bt, slices)
        finally:
            model.__dict__.pop("_joint_denoise_core", None)
        losses[name] = {"total": float(loss), **{k: float(v) for k, v in parts.items()}}
    rep["loss_at_start"] = losses
    rep["loss_parts_note"] = (f"'{P.VAE}' = video loss on the VAE channels; each source name = video loss on that "
                              f"source's channels; 'action' = action loss; predict_zero uses the same batch and draws")
    if not all(math.isfinite(v) for v in losses["model"].values()):
        raise SanityError(f"non-finite loss at start: {losses['model']}")
    return rep


# ---------------------------------------------------------------------------
# Measurement (Step 6)
# ---------------------------------------------------------------------------

def classify(e):
    """ok / oom / error for an exception (None = ok). Only torch.cuda.OutOfMemoryError counts as oom."""
    if e is None:
        return {"outcome": "ok"}
    out = {"outcome": "oom" if isinstance(e, torch.cuda.OutOfMemoryError) else "error",
           "error_type": type(e).__name__, "error_message": str(e)[:4000]}
    if out["outcome"] == "error":
        out["traceback"] = "".join(traceback.format_exception(e))[-8000:]
    return out


def m_star(rows, total_mem, global_batch=GLOBAL_BATCH):
    """Largest power of two m whose sweep row is ok, with the larger of its micro-step peak max_memory_reserved and its
    own full-optimizer-step peak (full_step_peak_reserved_bytes) <= 0.9 x total_mem (PLAN Step 6; the full-step peak
    added after the PR #31 review), and that divides the global batch. total_mem None (no CUDA, CPU smoke test only):
    largest ok m."""
    ok = [r for r in rows if r.get("outcome") == "ok" and r["m"] & (r["m"] - 1) == 0 and global_batch % r["m"] == 0]
    if total_mem is not None:
        ok = [r for r in ok if r.get("peak_reserved_bytes") is not None
              and max(r["peak_reserved_bytes"], r.get("full_step_peak_reserved_bytes") or 0)
              <= FIT_FRACTION * total_mem]
    return max((r["m"] for r in ok), default=None)


def needs_retry(opt_row, total_mem):
    """Whether the global-batch run at m* is repeated once at m*/2: it ended in oom, or it was ok but its peak
    max_memory_reserved is above 0.9 x total_mem (so m* does not fit and would leave no t_step)."""
    if opt_row.get("outcome") == "oom":
        return True
    pk = opt_row.get("peak_reserved_bytes")
    return (opt_row.get("outcome") == "ok" and total_mem is not None and pk is not None
            and pk > FIT_FRACTION * total_mem)


def fit_summary(rows, opt_rows, total_mem, mstar):
    """Fit list for the measured card (PLAN "Decision rule", with the PR #31 review's fixes).
    The peak at m is the largest of: the sweep's micro-step peak, the full optimizer step that every sweep process runs
    at m (full_step_peak_reserved_bytes), and the full-optimizer-step run at global batch if one ran at m. m fits only
    if that peak is <= 0.9 x M and no full-optimizer-step run at m ended in oom (so an oom at m* means m* does not fit,
    and the m*/2 retry decides for m*/2). t_step is reported only from an ok full-optimizer-step run at an m that fits;
    otherwise it is None and flagged. The linear fit uses sweep micro-step peaks only; the optimizer update's extra
    memory (full-step peak minus sweep peak at the same m) is a separate constant to add on top. Both are an
    extrapolation basis for cards not measured, not a measurement."""
    if total_mem is None:
        return {"note": "no CUDA memory figures"}
    limit = FIT_FRACTION * total_mem
    sweep = {r["m"]: r for r in rows if r.get("outcome") == "ok" and r.get("peak_reserved_bytes") is not None}
    opt_by_m = {}
    for r in opt_rows:
        opt_by_m.setdefault(r["m"], []).append(r)
    peaks, fits, why = {}, [], {}
    extra_sweep, extra_opt = {}, {}
    for m, r in sorted(sweep.items()):
        cand = {"sweep_micro_steps": r["peak_reserved_bytes"]}
        if r.get("full_step_peak_reserved_bytes") is not None:
            cand["sweep_full_step"] = r["full_step_peak_reserved_bytes"]
            extra_sweep[m] = r["full_step_peak_reserved_bytes"] - r["peak_reserved_bytes"]
        oom_at_m = False
        for o in opt_by_m.get(m, []):
            if o.get("outcome") == "ok" and o.get("peak_reserved_bytes") is not None:
                cand["global_batch_steps"] = max(cand.get("global_batch_steps", 0), o["peak_reserved_bytes"])
                extra_opt[m] = o["peak_reserved_bytes"] - r["peak_reserved_bytes"]
            elif o.get("outcome") == "oom":
                oom_at_m = True
        peaks[m] = {"peak_reserved_bytes": max(cand.values()), "from": cand}
        if oom_at_m:
            why[m] = "full optimizer steps at global batch ran out of memory at this m"
        elif peaks[m]["peak_reserved_bytes"] > limit:
            why[m] = "peak above 0.9 x total memory"
        else:
            fits.append(m)
    out = {"total_memory_bytes": total_mem, "limit_bytes": limit, "m_star": mstar, "peak_reserved_by_m": peaks,
           "does_not_fit": why, "largest_m_that_fits": max(fits, default=None), "fits_at_m1": bool(1 in fits)}
    timed = [o for m in fits for o in opt_by_m.get(m, []) if o.get("outcome") == "ok"]
    if timed:
        best = max(timed, key=lambda o: o["m"])
        out["t_step"] = {"m": best["m"], "sec_per_opt_step": best["sec_per_opt_step"]["mean"],
                         "global_batch": best.get("global_batch")}
    else:
        out["t_step"] = None
        out["t_step_flag"] = ("no full-optimizer-step time was measured at an m that fits; runs measured: "
                              + json.dumps({o["m"]: o.get("outcome") for o in opt_rows}))
    if len(sweep) >= 2:
        ms = sorted(sweep)
        slope, icpt = np.polyfit(ms, [sweep[m]["peak_reserved_bytes"] for m in ms], 1)
        out["linear_fit_sweep_peak_vs_m"] = {"bytes_per_sample": float(slope), "intercept_bytes": float(icpt),
                                             "label": "sweep micro-step peaks only; extrapolation basis for cards "
                                                      "not measured, not a measurement"}
    allx = list(extra_sweep.values()) + list(extra_opt.values())
    out["optimizer_update_extra_bytes"] = {
        "constant": max(0, max(allx)) if allx else None,
        "by_m_sweep_full_step": extra_sweep, "by_m_global_batch_steps": extra_opt,
        "label": "full-step peak minus sweep micro-step peak at the same m; the constant (largest seen) is added on "
                 "top of the linear fit when extrapolating, so it is an upper bound on the update's extra memory, "
                 "and an extrapolated fit using it is on the safe side"}
    return out


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _reset_peaks(device):
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem(device):
    if torch.device(device).type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def nvidia_smi(fields="memory.used,driver_version"):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")] if r.returncode == 0 else None
    except Exception:
        return None


def device_info(device):
    info = {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "fastwam_commit": FASTWAM_COMMIT, "repo_commit": git_commit()}
    if torch.device(device).type == "cuda":
        pr = torch.cuda.get_device_properties(0)
        smi = nvidia_smi("driver_version")
        info.update(card=pr.name, total_memory_bytes=pr.total_memory, driver=smi[0] if smi else None)
    return info


def make_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)


def micro_step(model, hb, data, device, amp, scale=1.0):
    """Forward and backward on one micro-batch; gradients accumulate. Returns the loss tensor."""
    bt = P.to_device(hb, data, device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
        loss, _ = P.training_loss(model, bt, data.channel_slices())
    (loss * scale).backward()
    return loss.detach()


def opt_update(model, opt):
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return gn


def state_path(state_dir, config):
    return Path(state_dir) / f"state-{config}.pt"


def save_state(path, model, opt):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, tmp)
    tmp.replace(path)


def load_state(path, model, opt):
    st = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    del st


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def model_kw_of(args):
    return {"hidden": args.hidden, "action_hidden": args.action_hidden, "heads": args.heads,
            "head_dim": args.head_dim}


def _child(args, row, phases, log):
    """One process: a sweep point (mode sweep) or the full optimizer steps (mode optstep). Fills row and phases."""
    dev, keep, out = args.device, parse_keep(args.keep), Path(args.out)
    amp = torch.device(dev).type == "cuda"
    t = time.time()
    data, suites = load_data(args.feats, args.config, args.t5, args.suites.split(",") if args.suites else None)
    data.context_dev = data.context.to(dev)
    cache, read_s = read_all(data)
    row.update(suites=suites, n_episodes=len(data.episodes), in_dim=data.in_dim, dims=data.dims,
               encoder=None if args.config == "none" else args.config, amp_bf16=amp, disk_read_s=read_s)
    m = args.m
    if args.mode == "sweep":
        n = 1 + WARMUP + TIMED
    else:
        k = args.global_batch // m
        if k * m != args.global_batch:
            raise ValueError(f"m {m} does not divide the global batch {args.global_batch}")
        n = (OPT_WARMUP + OPT_TIMED) * k
        row["accumulation"] = k
    hbs, asm = host_batches(data, cache, m, n, args.seed)
    sanity_hb = host_batches(data, cache, args.sanity_batch, 1, args.seed)[0][0] if args.from_wan else None
    del cache
    row["host_assembly_s_per_batch"] = {"median": float(np.median(asm)), "max": float(np.max(asm)), "n": len(asm)}
    phases["data"] = time.time() - t

    t = time.time()
    model = build(data.in_dim, dev, len(keep), model_kw_of(args), args.seed)
    model.train()
    opt = make_opt(model)
    phases["build"] = time.time() - t
    sp = state_path(args.state_dir, args.config)
    t = time.time()
    if args.from_wan:
        rep = load_wan(model, args.wan_dir, keep, log=log)
        rep.update(config=args.config, keep_tag=args.keep_tag, seed=args.seed, **device_info(dev))
        _write_json(out / f"load-{args.config}.json", rep)
        phases["load_wan"] = time.time() - t
        log(f"[load] {rep['counts']}")
        t = time.time()
        srep = sanity(model, sanity_hb, P.to_device(sanity_hb, data, dev), data, args.seed, amp)
        srep.update(config=args.config, keep=keep, keep_tag=args.keep_tag, seed=args.seed,
                    batch_size=args.sanity_batch)
        _write_json(out / f"sanity-{args.config}.json", srep)
        log(f"[sanity] loss at start {srep['loss_at_start']}")
        phases["sanity"] = time.time() - t
    else:
        if not sp.exists():
            raise FileNotFoundError(f"saved state {sp} missing (the first process did not finish)")
        load_state(sp, model, opt)
        phases["load_state"] = time.time() - t
    _reset_peaks(dev)

    if args.mode == "sweep":
        # one full optimizer step at m: allocates gradients (and AdamW states, unless loaded)
        t = time.time()
        loss = micro_step(model, hbs[0], data, dev, amp)
        opt_update(model, opt)
        _sync(dev)
        row["first_step_loss"] = float(loss)
        if not math.isfinite(row["first_step_loss"]):
            raise SanityError(f"non-finite loss {row['first_step_loss']}")
        phases["first_opt_step"] = time.time() - t
        # peak of this full optimizer step at m (weights, gradients, both AdamW moments and the update's own
        # temporaries), read before the reset below; peaks were reset just before the step
        fs = _mem(dev)
        row["full_step_peak_reserved_bytes"] = fs["peak_reserved_bytes"]
        row["full_step_peak_allocated_bytes"] = fs["peak_allocated_bytes"]
        if args.from_wan:
            t = time.time()
            save_state(sp, model, opt)
            phases["save_state"] = time.time() - t
        opt.zero_grad(set_to_none=False)
        _reset_peaks(dev)
        row["static_allocated_bytes"] = torch.cuda.memory_allocated() if amp else None
        t = time.time()
        for hb in hbs[1:1 + WARMUP]:
            micro_step(model, hb, data, dev, amp)
        _sync(dev)
        times = []
        for hb in hbs[1 + WARMUP:]:
            # host -> device copy outside the timed window (PLAN: step time is GPU time only)
            bt_dev = P.to_device(hb, data, dev)
            _sync(dev)
            t1 = time.perf_counter()
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=amp):
                loss, _ = P.training_loss(model, bt_dev, data.channel_slices())
            loss.backward()
            _sync(dev)
            times.append(time.perf_counter() - t1)
            del bt_dev, loss
        phases["measure"] = time.time() - t
        med = float(np.median(times))
        row.update(sec_per_micro_step={"median": med, "min": float(min(times)), "max": float(max(times)),
                                       "q25": float(np.percentile(times, 25)), "q75": float(np.percentile(times, 75)),
                                       "all": times},
                   sec_per_sample=med / m, warmup_micro_steps=WARMUP, timed_micro_steps=TIMED, **_mem(dev))
    else:
        t = time.time()
        steps, losses = [], []
        for s in range(OPT_WARMUP + OPT_TIMED):
            _sync(dev)
            t1 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            ls = [micro_step(model, hb, data, dev, amp, scale=1.0 / k) for hb in hbs[s * k:(s + 1) * k]]
            opt_update(model, opt)
            _sync(dev)
            steps.append(time.perf_counter() - t1)
            losses.append(float(torch.stack(ls).mean()))
        phases["measure"] = time.time() - t
        if not all(math.isfinite(x) for x in losses):
            raise SanityError(f"non-finite loss in optimizer steps: {losses}")
        timed = steps[OPT_WARMUP:]
        row.update(sec_per_opt_step={"timed": timed, "mean": float(np.mean(timed)), "warmup": steps[:OPT_WARMUP]},
                   global_batch=args.global_batch, losses=losses,
                   timing_note="each optimizer step includes the host->device copies of its micro-batches, "
                               "backward, grad clipping and the AdamW update; peaks since the state load",
                   **_mem(dev))
    smi = nvidia_smi("memory.used") if amp else None
    row["nvidia_smi_used_mib"] = float(smi[0]) if smi else None


def child_main(args):
    t0 = time.time()
    log = lambda msg: print(f"[{args.mode} {args.config} m={args.m} {time.time() - t0:6.0f}s] {msg}", flush=True)
    row = {"mode": args.mode, "config": args.config, "m": args.m, "from_wan": bool(args.from_wan),
           "keep": parse_keep(args.keep), "keep_tag": args.keep_tag, "seed": args.seed, "pid": os.getpid()}
    phases, err = {}, None
    try:
        row.update(device_info(args.device))
        _child(args, row, phases, log)
    except BaseException as e:          # noqa: BLE001 - every ending is classified and written
        err = e
    row.update(classify(err))
    row["phase_seconds"] = phases
    row["wall_s"] = time.time() - t0
    _write_json(args.row_file, row)
    log(f"outcome {row['outcome']}" + (f": {row.get('error_type')}: {row.get('error_message', '')[:300]}"
                                        if err else ""))
    return 0 if err is None else (3 if row["outcome"] == "oom" else 4)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

_current = {"proc": None, "stop": False}


def _on_term(signum, frame):
    _current["stop"] = True
    if _current["proc"] is not None and _current["proc"].poll() is None:
        _current["proc"].terminate()


def _child_cmd(args, mode, config, m, from_wan, row_file):
    cmd = [sys.executable, str(Path(__file__).resolve()), mode, "--config", config, "--m", str(m),
           "--row-file", str(row_file)]
    for k in ("feats", "t5", "suites", "wan_dir", "keep", "keep_tag", "seed", "out", "state_dir", "device",
              "hidden", "action_hidden", "heads", "head_dim", "global_batch", "sanity_batch"):
        v = getattr(args, k)
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd + (["--from-wan"] if from_wan else [])


def run_process(args, mode, config, m, from_wan=False):
    row_file = Path(args.out) / "rows" / f"{config}-{mode}-m{m}.json"
    if row_file.exists():
        row_file.unlink()
    t = time.time()
    print(f"=== launcher: {mode} config={config} m={m}{' (from Wan)' if from_wan else ''}", flush=True)
    proc = subprocess.Popen(_child_cmd(args, mode, config, m, from_wan, row_file))
    _current["proc"] = proc
    rc = proc.wait()
    _current["proc"] = None
    if row_file.exists():
        row = json.loads(row_file.read_text())
    else:
        row = {"mode": mode, "config": config, "m": m, "outcome": "error", "error_type": "ProcessDied",
               "error_message": f"exit code {rc} without writing a row"
                                + (" (stopped by the inner time limit)" if _current["stop"] else "")}
    row.update(returncode=rc, launcher_wall_s=time.time() - t)
    return row


def launch(args):
    signal.signal(signal.SIGTERM, _on_term)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    ms = [int(x) for x in args.ms.split(",")]
    for config in args.configs.split(","):
        res = {"config": config, "encoder": None if config == "none" else config, "keep": parse_keep(args.keep),
               "keep_tag": args.keep_tag, "seed": args.seed, "fastwam_commit": FASTWAM_COMMIT,
               "repo_commit": git_commit(), "model_id": WAN_ID, "model_kw": {**model_kw_of(args)},
               "global_batch": args.global_batch, "sweep_ms": ms, "fit_fraction": FIT_FRACTION,
               "rows": [], "optimizer_steps": [], "started_unix": time.time()}
        path = out / f"measure-{config}.json"
        # Never overwrite results (_standards/03): refuse an existing file
        if path.exists():
            raise FileExistsError(f"{path} exists; use a new --out")

        def write():
            for r in res["rows"] + res["optimizer_steps"]:
                for k in ("card", "total_memory_bytes", "driver", "cuda", "torch", "device"):
                    if r.get(k) is not None and res.get(k) is None:
                        res[k] = r[k]
            res["elapsed_s"] = time.time() - res["started_unix"]
            _write_json(path, res)

        for i, m in enumerate(ms):
            if _current["stop"]:
                break
            row = run_process(args, "sweep", config, m, from_wan=(i == 0))
            res["rows"].append(row)
            write()
            if row["outcome"] != "ok":
                res["sweep_stopped"] = {"m": m, "outcome": row["outcome"],
                                        "meaning": "does not fit" if row["outcome"] == "oom" else "error, not a memory limit"}
                write()
                break
        total = res.get("total_memory_bytes")
        mstar = m_star(res["rows"], total, args.global_batch)
        res["m_star"] = mstar
        res["m_star_rule"] = (f"largest power of two m with an ok sweep row whose larger of the micro-step peak and "
                              f"the full-optimizer-step peak (max_memory_reserved) is <= {FIT_FRACTION} x total "
                              f"memory; the global-batch run is retried once at m*/2 on oom or a peak above that"
                              if total is not None else
                              "no CUDA memory figures (CPU smoke test): largest ok m")
        write()
        first_ok = bool(res["rows"]) and res["rows"][0]["outcome"] == "ok"
        if mstar is not None and first_ok and not _current["stop"]:
            row = run_process(args, "optstep", config, mstar)
            res["optimizer_steps"].append(row)
            write()
            if needs_retry(row, total) and mstar > 1 and not _current["stop"]:
                res["optimizer_steps"].append(run_process(args, "optstep", config, mstar // 2))
                write()
        res["fit"] = fit_summary(res["rows"], res["optimizer_steps"], total, mstar)
        if _current["stop"]:
            res["stopped_by_inner_time_limit"] = True
        write()
        sp = state_path(args.state_dir, config)
        if sp.exists():
            sp.unlink()
        if _current["stop"]:
            break
    print(f"=== launcher done in {time.time() - t_all:.0f}s", flush=True)
    return 124 if _current["stop"] else 0


# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["launch", "sweep", "optstep"])
    p.add_argument("--feats", required=True, help="data slice root in the proxy's cache layout (vae, vae_b, <enc>, meta)")
    p.add_argument("--t5", default=None, help="T5 cache dir (default <feats>/t5)")
    p.add_argument("--suites", default=None, help="comma-separated; default every meta/*.npz in the slice")
    p.add_argument("--wan-dir", required=True, help="folder with Wan 2.2 TI2V-5B config.json, index json, needed shards")
    p.add_argument("--keep", default=KEEP_DEFAULT, help="kept Wan blocks, e.g. 0-7 or 0,4,8")
    p.add_argument("--keep-tag", default=KEEP_TAG)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="results folder")
    p.add_argument("--state-dir", required=True, help="local disk for the saved model + AdamW state")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden", type=int, default=WAN_KW["hidden"])
    p.add_argument("--action-hidden", type=int, default=WAN_KW["action_hidden"])
    p.add_argument("--heads", type=int, default=WAN_KW["heads"])
    p.add_argument("--head-dim", type=int, default=WAN_KW["head_dim"])
    p.add_argument("--global-batch", type=int, default=GLOBAL_BATCH)
    p.add_argument("--sanity-batch", type=int, default=16)
    # launcher
    p.add_argument("--configs", default="none,dinov3", help="launch: configs in order")
    p.add_argument("--ms", default=",".join(map(str, SWEEP_MS)), help="launch: sweep micro-batch sizes, ascending")
    # child
    p.add_argument("--config", default=None)
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--from-wan", action="store_true")
    p.add_argument("--row-file", default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "launch":
        return launch(args)
    return child_main(args)


if __name__ == "__main__":
    sys.exit(main())
