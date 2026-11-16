"""Training script for the stage 1 baseline: the 8-layer pretrained-start model, config `none`, clean LIBERO
(experiments/2026-11-15-pretrained-start-baseline/PLAN.md, Design; Fast-WAM facts from fastwam-reading.md there).

Model: pretrained_sizing.build + load_wan (FastWAMJoint at Wan 2.2 TI2V-5B width, video expert blocks 0-7 from the Wan
file), then action_expert_init.init_action_expert (the action expert's backbone derived from those blocks by Fast-WAM's
recipe; action encoder, head and proprio encoder fresh from the seeded build). The user's settled choices: all 1,712
episodes, no validation split; cached VAE latents and T5 (Fast-WAM's LIBERO training has no augmentation); action
timestep shift 1.0, video 5.0 (proxy_wam.build_model); fp32 weights + bf16 autocast + fp32 AdamW; one seed (0); the
paper's Table 6 settings including gradient clipping at 1.0.

Reused from proxy_wam: Data, make_plan (16-episode buffers), Loader, to_device, training_loss, lr_at, save_ckpt.

Global batch 128 by accumulation (PLAN departure 2): --micro-batch 8 (or 4) gives 16 (or 32) micro-batches per optimizer
step. The video and action noise and timesteps for all 128 samples are drawn once per optimizer step from a CPU
generator seeded by (seed, step) (draw_step), with the same calls, shapes and distributions training_loss uses for a
batch of 128 (Fast-WAM's sample_training_t takes no generator, so its two lines are applied to the generator's uniform
draws, t_from_u), and each micro-batch gets its slice. Each micro-batch loss is scaled by its share (m / 128) before
backward, so the accumulated gradient is that of the mean over 128 up to floating-point order. Clip 1.0 on the
accumulated gradient, then AdamW. The LR schedule (lr_at: 5% linear warmup, cosine 1e-4 to 1e-6) always runs over the
full plan, so a --max-steps run is exactly the first steps of the full run.

Checkpoints (PLAN "Checkpointing and resume"): every --ckpt-every optimizer steps to <out>/ckpt.pt (tmp + rename),
only between optimizer steps, with model, AdamW, step, RNG states, the micro-batch and its history. A restarted run
finds <out>/ckpt.pt and continues from its step; a micro-batch different from the checkpoint's is logged and recorded.
ckpt.pt is never deleted by this script. At the end: model.pt ({"model", "config"} with config["model_kw"] such that
rollout.Policy.load builds it with proxy_wam.build_model(cfg["in_dim"], device, **cfg["model_kw"])) and train.json.

Sanity modes (PLAN sanity checks 1-3):
  --overfit-steps N    a fixed batch of 8 windows (the first 8 of the plan's first batch), constant LR (--overfit-lr),
                       N optimizer steps with the draws of step 0 every step (fixed noise, as the proxy's check 9),
                       loss vs predict-zero on the same batch and draws; writes overfit.json, no checkpoint, no model.pt.
  --max-steps N        a short run: the first N optimizer steps of the full run.
  --first-batch-report the first batch of the plan (all 128 windows, step-0 draws): loss of the model and of
                       predict-zero, input shapes, ranges, padding counts and task strings; writes first-batch.json.
                       Skipped (and logged) if first-batch.json already exists, so a resume keeps the initial report.
A resume whose checkpoint is at or past the end of this run (--max-steps or the plan end) is refused. After the first
checkpoint write the projection is logged again with the measured write time times the number of writes.
A non-finite loss or gradient norm stops the run (exit 2) before the optimizer step; nothing is checkpointed from it.

Example (the full run):
  python pretrained_train.py --feats /mnt/proxy/feats --t5 /mnt/proxy/feats/t5 \
      --wan-dir /mnt/proxy/models/Wan-AI/Wan2.2-TI2V-5B --micro-batch 8 --ckpt-every 700 --first-batch-report \
      --out /mnt/proxy/runs/pretrained-none-s0
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import action_expert_init as AEI
import pretrained_sizing as S
import proxy_wam as P
from diagnostics import git_commit

SUITES = "libero_spatial,libero_object,libero_goal,libero_10"
MICRO_BATCHES = (8, 4)        # PLAN "Parameters set from measurements": the largest of 8 and 4 that fits
WATCHDOG_EXIT = 17            # exit code of the step watchdog (a hung step); the Job treats it as retryable
PROJECT_AFTER = 20            # optimizer steps before the projected total time is logged


class NonFiniteLoss(RuntimeError):
    """A non-finite loss or gradient norm (the run stops)."""


# ---------------------------------------------------------------------------
# Draws: once per optimizer step, keyed by (seed, step)
# ---------------------------------------------------------------------------

def step_generator(seed, step):
    """CPU generator seeded from (seed, step) only, so a step's draws depend on nothing else (not the micro-batch, not
    RNG state carried across a resume)."""
    s = int(np.random.SeedSequence([int(seed), int(step)]).generate_state(1, np.uint64)[0] & ((1 << 63) - 1))
    return torch.Generator(device="cpu").manual_seed(s)


def t_from_u(scheduler, u, dtype=torch.float32):
    """WanContinuousFlowMatchScheduler.sample_training_t (scheduler_continuous.py:31-37) given its uniform draws u."""
    sigma = scheduler._phi(u, scheduler.shift)
    return (sigma * float(scheduler.num_train_timesteps)).to(dtype=dtype)


def draw_step(model, seed, step, n, lat_shape, act_shape=(P.HORIZON, 7)):
    """(noise_v, tv, noise_a, ta) for n samples, in training_loss's order (video noise, video t, action noise,
    action t), fp32 on CPU. lat_shape: (C, LATENT_T, H, W) of one sample."""
    g = step_generator(seed, step)
    noise_v = torch.randn((n,) + tuple(lat_shape), generator=g, dtype=torch.float32)
    tv = t_from_u(model.train_video_scheduler, torch.rand((n,), generator=g, dtype=torch.float32))
    noise_a = torch.randn((n,) + tuple(act_shape), generator=g, dtype=torch.float32)
    ta = t_from_u(model.train_action_scheduler, torch.rand((n,), generator=g, dtype=torch.float32))
    return noise_v, tv, noise_a, ta


def lat_shape_of(data):
    return (data.in_dim, P.LATENT_T) + P.GRID


# ---------------------------------------------------------------------------
# One accumulated batch
# ---------------------------------------------------------------------------

def _amp(device):
    return torch.device(device).type == "cuda"


def host_slice(hb, sl):
    return {k: v[sl] for k, v in hb.items()}


def forward_backward(model, hb, data, draws, micro, device, backward=True):
    """Losses (and, if backward, accumulated gradients) for one host batch of B windows in micro-batches of `micro`,
    each micro-batch's loss scaled by its share micro / B. Returns (loss, parts) as floats, the share-weighted sums,
    i.e. the mean over the B samples."""
    B = len(hb["task"])
    if B % micro:
        raise ValueError(f"micro-batch {micro} does not divide the batch {B}")
    slices = data.channel_slices()
    tot, parts = 0.0, {}
    for j in range(B // micro):
        sl = slice(j * micro, (j + 1) * micro)
        bt = P.to_device(host_slice(hb, sl), data, device)
        d = tuple(x[sl].to(device, non_blocking=True) for x in draws)
        share = micro / B
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=_amp(device)):
            loss, pr = P.training_loss(model, bt, slices, draws=d)
        if backward:
            (loss * share).backward()
        # summed as fp64 tensors on the device; read back once per batch (no host sync per micro-batch)
        tot = tot + loss.detach().double() * share
        for k, v in pr.items():
            parts[k] = parts.get(k, 0.0) + v.detach().double() * share
        del bt, d, loss, pr
    return float(tot), {k: float(v) for k, v in parts.items()}


def optimizer_step(model, opt, hb, data, draws, micro, device, lr):
    """zero_grad, accumulate over micro-batches, check finite, clip 1.0, AdamW. Returns (loss, parts, grad norm)."""
    for gp in opt.param_groups:
        gp["lr"] = lr
    opt.zero_grad(set_to_none=True)
    loss, parts = forward_backward(model, hb, data, draws, micro, device)
    if not (math.isfinite(loss) and all(math.isfinite(v) for v in parts.values())):
        raise NonFiniteLoss(f"non-finite loss {loss} {parts}")
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not math.isfinite(gn):
        raise NonFiniteLoss(f"non-finite gradient norm {gn} (loss {loss})")
    opt.step()
    return loss, parts, gn


@torch.no_grad()
def predict_zero_loss(model, hb, data, draws, micro, device):
    """Loss of a model that predicts 0 for video and action, on the same batch and draws."""
    model._joint_denoise_core = lambda **kw: (torch.zeros_like(kw["latents_video"]),
                                              torch.zeros_like(kw["latents_action"]))
    try:
        return forward_backward(model, hb, data, draws, micro, device, backward=False)
    finally:
        model.__dict__.pop("_joint_denoise_core", None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_and_init(data, args, device, log):
    """Seeded build at the given width and depth, Wan blocks loaded, action expert derived. Returns (model, report)."""
    keep = S.parse_keep(args.keep)
    model = S.build(data.in_dim, device, len(keep), model_kw_of(args, keep)["_build"], args.seed)
    rep = S.load_wan(model, args.wan_dir, keep, log=log)
    arep = AEI.init_action_expert(model, log=log)
    # the sizing load report tags every action-expert tensor "new" (fresh); correct it for the derived ones
    for n, e in arep["parameters"].items():
        if e["kind"] != "fresh":
            rep["parameters"][n] = {"tag": "derived", **e}
    # load_wan's counts were taken before the derivation (every action-expert tensor "new"); keep them under a name
    # that says so and recompute the counts the run starts from
    rep["counts_before_derivation"] = rep.pop("counts")
    plist = dict(model.named_parameters())
    counts = {t: {"tensors": 0, "elements": 0} for t in ("loaded", "partly loaded", "derived", "fresh")}
    for n, e in rep["parameters"].items():
        t = "fresh" if e["tag"] == "new" else e["tag"]
        counts[t]["tensors"] += 1
        counts[t]["elements"] += plist[n].numel()
    for kind in ("copied", "resized"):
        counts["derived"][kind] = sum(1 for e in rep["parameters"].values()
                                      if e["tag"] == "derived" and e["kind"] == kind)
    counts["derived"]["source"] = "Wan 2.2 video blocks via the port of Fast-WAM's preprocess_action_dit_backbone.py"
    rep["counts"] = counts
    rep["counts_by_tag"] = {t: c["tensors"] for t, c in counts.items()}
    rep["action_expert_init"] = {k: v for k, v in arep.items() if k != "parameters"}
    rep["init"] = {**rep.get("init", {}),
                   "new": "fresh: action_expert.action_encoder, action_expert.head and proprio_encoder, each module's "
                          "own PyTorch init after torch.manual_seed(seed), as Fast-WAM leaves them",
                   "derived": arep["recipe"]}
    return model, rep


def model_kw_of(args, keep):
    kw = {"hidden": args.hidden, "action_hidden": args.action_hidden, "heads": args.heads, "head_dim": args.head_dim}
    return {"_build": kw, "model_kw": {**kw, "layers": len(keep)}}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def first_batch_report(model, hb, data, draws, micro, device):
    """Sanity checks 2 and 3 on one host batch: inputs (shapes, ranges, padding, tasks) and the loss of the model vs
    predict-zero on the same draws. Raises NonFiniteLoss if the model's loss is not finite."""
    B = len(hb["task"])
    rep = {"batch_size": B, "micro_batch": micro}
    lat = hb["latents"]                                          # (B, 3, 14, 28, C) float16, as cached
    rep["inputs"] = {"latents_host": {"shape": list(lat.shape), "dtype": str(lat.dtype)},
                     "latents_device_per_micro": [micro, data.in_dim, P.LATENT_T, *P.GRID],
                     "context_per_micro": [micro, *data.context.shape[1:]],
                     **{k: {"shape": list(hb[k].shape), "dtype": str(hb[k].dtype)}
                        for k in ("action", "action_is_pad", "latent_is_pad", "proprio", "task")}}
    f = lat.astype(np.float32).reshape(-1, lat.shape[-1])
    for src, sl in data.channel_slices().items():
        mu, sd = f[:, sl].mean(0), f[:, sl].std(0)
        rep[f"channels/{src}"] = {"n": sl.stop - sl.start, "mean_range": [float(mu.min()), float(mu.max())],
                                  "std_range": [float(sd.min()), float(sd.max())],
                                  "min": float(f[:, sl].min()), "max": float(f[:, sl].max())}
    for k in ("action", "proprio"):
        rep[f"{k}_range"] = [float(hb[k].min()), float(hb[k].max())]
        rep[f"{k}_abs_gt1_fraction"] = float((np.abs(hb[k]) > 1).mean())
    rep["padded_action_steps"] = int(hb["action_is_pad"].sum())
    rep["padded_latent_steps"] = int(hb["latent_is_pad"].sum())
    rep["windows_with_padding"] = int((hb["action_is_pad"].any(1) | hb["latent_is_pad"].any(1)).sum())
    rep["tasks"] = [data.tasks[i] for i in hb["task"]]
    rep["n_distinct_tasks"] = len(set(rep["tasks"]))
    rep["draws"] = {"tv_range": [float(draws[1].min()), float(draws[1].max())],
                    "ta_range": [float(draws[3].min()), float(draws[3].max())]}
    with torch.no_grad():
        loss, parts = forward_backward(model, hb, data, draws, micro, device, backward=False)
    zl, zp = predict_zero_loss(model, hb, data, draws, micro, device)
    rep["loss_at_start"] = {"model": {"total": loss, **parts}, "predict_zero": {"total": zl, **zp}}
    rep["loss_parts_note"] = ("'vae' = video loss on the VAE channels, 'action' = action loss; predict_zero = a model "
                              "that outputs 0, on the same batch and noise/timestep draws (step 0)")
    if not (math.isfinite(loss) and all(math.isfinite(v) for v in parts.values())):
        raise NonFiniteLoss(f"non-finite loss at start: {rep['loss_at_start']['model']}")
    return rep


def overfit(model, opt, hb, data, args, device, log):
    """Sanity check 1: a fixed batch, constant LR, the same draws every step (or fresh ones per step if asked)."""
    B = len(hb["task"])
    fixed = draw_step(model, args.seed, 0, B, lat_shape_of(data))
    zl, zp = predict_zero_loss(model, hb, data, fixed, min(args.micro_batch, B), device)
    hist = []
    for s in range(args.overfit_steps):
        draws = draw_step(model, args.seed, s, B, lat_shape_of(data)) if args.overfit_fresh_draws else fixed
        t = time.time()
        loss, parts, gn = optimizer_step(model, opt, hb, data, draws, min(args.micro_batch, B), device,
                                         args.overfit_lr)
        rec = {"step": s, "loss": loss, **{f"loss/{k}": v for k, v in parts.items()}, "grad_norm": gn,
               "s_step": time.time() - t}
        hist.append(rec)
        if s % 10 == 0 or s == args.overfit_steps - 1:
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
    k = min(20, max(1, len(hist) // 10))
    first = {key: float(np.mean([h[key] for h in hist[:k]])) for key in hist[0] if key.startswith("loss")}
    last = {key: float(np.mean([h[key] for h in hist[-k:]])) for key in first}
    return {"batch_size": B, "steps": args.overfit_steps, "lr": args.overfit_lr,
            "draws": "fresh per step (seed, step)" if args.overfit_fresh_draws else "fixed: the draws of step 0 "
                                                                                    "every step",
            "predict_zero": {"total": zl, **zp}, f"first_{k}_mean": first, f"last_{k}_mean": last,
            "history": hist}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _peak_reset(device):
    if _amp(device):
        torch.cuda.reset_peak_memory_stats()


def _peak(device):
    return torch.cuda.max_memory_reserved() if _amp(device) else None


def _watchdog_exit(code):
    os._exit(code)           # a hung main thread cannot be interrupted; tests replace this function


class StepWatchdog:
    """Ends the process (exit WATCHDOG_EXIT) when training stalls. A daemon thread with two deadlines: until the first
    optimizer step finishes, first_s seconds from construction (data index, model build, a resume's checkpoint load, the
    first step; None = not armed until the first step); after each step, limit_s seconds since the last beat (steps and
    checkpoint writes). It changes nothing in the training: no draws, no saved state. The checkpoint is only ever
    replaced by a rename, so ending the process during a write leaves the previous checkpoint intact. limit_s 0 disables
    the whole watchdog; first_s 0 or None disables the first deadline."""

    def __init__(self, limit_s, log, exit_fn=None, first_s=None):
        self.limit, self.first, self.log, self.exit_fn = limit_s, (first_s or None), log, exit_fn
        self.last, self.armed, self.fired, self.stepping = time.monotonic(), self.first is not None, False, False
        self.stop_ev = threading.Event()
        self.thread = None
        if limit_s and limit_s > 0:
            self.thread = threading.Thread(target=self._run, daemon=True, name="step-watchdog")
            self.thread.start()

    def beat(self):
        self.last = time.monotonic()
        self.armed = self.stepping = True

    def stop(self):
        self.stop_ev.set()

    def _run(self):
        poll = min(5.0, self.limit / 4, (self.first or self.limit) / 4)
        while not self.stop_ev.wait(poll):
            if not self.armed or self.fired:
                continue
            lim, phase = ((self.limit, "training steps") if self.stepping
                          else (self.first, "start-up before the first optimizer step (data index, model build, "
                                            "checkpoint load, first step)"))
            if time.monotonic() - self.last > lim:
                self.fired = True
                self.log(f"[watchdog] fired in the {phase} phase: no progress for {lim:.0f} s (a hung loader or a "
                         f"stalled CephFS read or write); exiting with code {WATCHDOG_EXIT}; the last ckpt.pt is "
                         f"intact and a retried pod resumes from it")
                (self.exit_fn or _watchdog_exit)(WATCHDOG_EXIT)


def train(model, opt, data, plan, args, device, log, ckpt, start, stop, micro_hist, hist, hist_file=None, wd=None):
    """Optimizer steps start..stop-1 of the plan. Returns (history of this process, finished). Each step's record is
    also appended to hist_file (history.jsonl in --out), which survives a lost pod."""
    total = len(plan)
    t_run = time.time()
    loader = P.Loader(data, plan[:stop], start, horizon=args.horizon, read_threads=args.read_threads,
                      assemble_threads=args.assemble_threads)
    times, ckpt_logged = [], False
    own_wd = wd is None
    wd = wd or StepWatchdog(args.step_watchdog_s, log, first_s=args.step_watchdog_first_s)
    try:
        return _train_loop(model, opt, data, plan, args, device, log, ckpt, start, stop, micro_hist, hist, hist_file,
                           total, t_run, loader, times, ckpt_logged, wd)
    finally:
        if own_wd:
            wd.stop()


def _train_loop(model, opt, data, plan, args, device, log, ckpt, start, stop, micro_hist, hist, hist_file, total,
                t_run, loader, times, ckpt_logged, wd):
    tw = time.time()
    for s, hb in zip(range(start, stop), loader):
        wait = time.time() - tw
        t = time.time()
        _peak_reset(device)
        draws = draw_step(model, args.seed, s, len(hb["task"]), lat_shape_of(data))
        lr = P.lr_at(s, total)
        loss, parts, gn = optimizer_step(model, opt, hb, data, draws, args.micro_batch, device, lr)
        if _amp(device):
            torch.cuda.synchronize()
        dt = time.time() - t
        times.append(dt)
        pk = _peak(device)
        rec = {"step": s, "lr": lr, "loss": loss, **{f"loss/{k}": v for k, v in parts.items()}, "grad_norm": gn,
               "s_step": dt, "data_wait_s": wait, "peak_reserved_bytes": pk, "micro_batch": args.micro_batch}
        hist.append(rec)
        log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
        if len(times) == PROJECT_AFTER:
            per = float(np.mean(times[1:]))           # the first step of a process carries warm-up costs
            log(f"[projection] after {PROJECT_AFTER} steps: {per:.3f} s/step (steps 2-{PROJECT_AFTER} of this "
                f"process, compute only; data wait so far {sum(h['data_wait_s'] for h in hist[-PROJECT_AFTER:]):.1f}"
                f" s); full run of {total} steps = {per * total / 3600:.2f} h; remaining {stop - s - 1} steps of this "
                f"run = {per * (stop - s - 1) / 3600:.2f} h")
        if hist_file is not None:
            with open(hist_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        wd.beat()
        if ckpt and args.ckpt_every and (s + 1) % args.ckpt_every == 0 and s + 1 < stop:
            wt = save(ckpt, model, opt, s + 1, args, micro_hist, log)
            wd.beat()
            if not ckpt_logged:
                ckpt_logged = True
                log(project_with_writes(times, wt, total, s + 1, stop, args.ckpt_every))
        if args.stop_after is not None and s + 1 >= args.stop_after:
            log(f"--stop-after {args.stop_after}: stopping as if the pod were lost (no model.pt)")
            return hist, False
        tw = time.time()
    log(f"steps {start}-{stop - 1} done in {time.time() - t_run:.0f}s; disk read wait {loader.read_s:.0f}s")
    return hist, True


def read_history(path):
    """Step records from history.jsonl, the last record of each step (a step run again after a resume replaces the
    record a lost pod wrote after its last checkpoint), in step order."""
    by = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by[r["step"]] = r
    return [by[k] for k in sorted(by)]


def n_writes(a, b, every):
    """Checkpoint writes after steps s with a < s < b, s a multiple of `every` (the loop writes at s + 1 < stop)."""
    return 0 if not every else max(0, (b - 1) // every - a // every)


def project_with_writes(times, write_s, total, done, stop, every):
    """The projection re-logged after the first checkpoint write: compute time per step (steps 2.. of this process)
    plus the checkpoint writes at the measured write time."""
    per = float(np.mean(times[1:] if len(times) > 1 else times))
    full_w, rem_w = n_writes(0, total, every), n_writes(done, stop, every)
    return (f"[projection] with checkpoint writes: first write took {write_s:.1f} s; {per:.3f} s/step over "
            f"{max(1, len(times) - 1)} steps; full run of {total} steps = {per * total / 3600:.2f} h compute + "
            f"{full_w} writes x {write_s:.1f} s = {(per * total + full_w * write_s) / 3600:.2f} h; remaining "
            f"{stop - done} steps of this run + {rem_w} writes = {(per * (stop - done) + rem_w * write_s) / 3600:.2f} h")


def save(ckpt, model, opt, step, args, micro_hist, log):
    """Write the checkpoint; returns the wall time of the write."""
    t = time.time()
    P.save_ckpt(ckpt, model, opt, step, micro_batch=args.micro_batch, global_batch=args.global_batch, seed=args.seed,
                micro_batch_history=micro_hist, run_identity=args.run_identity)
    dt = time.time() - t
    log(f"[ckpt] step {step} written to {ckpt} in {dt:.1f}s")
    return dt


# ---------------------------------------------------------------------------

def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--feats", required=True, help="root written by extract_fastwam.py (meta, vae, vae_b)")
    p.add_argument("--t5", required=True, help="dir written by t5_cache.py")
    p.add_argument("--suites", default=SUITES)
    p.add_argument("--config", default="none", help="none (stage 1); encoders joined by + for later stages")
    p.add_argument("--wan-dir", required=True, help="Wan 2.2 TI2V-5B folder: config.json, index json, shards")
    p.add_argument("--keep", default=S.KEEP_DEFAULT, help="kept Wan blocks (stage 1: 0-7, the user's choice)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--global-batch", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=8, help="8 or 4 on the real model (PLAN); any divisor in tests")
    p.add_argument("--buffer-episodes", type=int, default=16)
    p.add_argument("--read-threads", type=int, default=32)
    p.add_argument("--assemble-threads", type=int, default=3)
    p.add_argument("--ckpt-every", type=int, default=700, help="optimizer steps between checkpoints (PLAN: ~45 min)")
    p.add_argument("--step-watchdog-s", type=float, default=1800,
                   help=f"exit {WATCHDOG_EXIT} if no optimizer step finishes within this many seconds (armed after the "
                        f"first step; 0 = off)")
    p.add_argument("--step-watchdog-first-s", type=float, default=3600,
                   help="the watchdog's deadline from process start to the end of the first optimizer step (data "
                        "index, model build, a resume's checkpoint load); 0 = none until the first step")
    p.add_argument("--max-steps", type=int, default=None, help="run only the first N optimizer steps of the full run")
    p.add_argument("--stop-after", type=int, default=None, help="testing: stop after this step as if the pod were lost")
    p.add_argument("--first-batch-report", action="store_true", help="write first-batch.json before training")
    p.add_argument("--overfit-steps", type=int, default=0, help="sanity check 1: overfit a fixed batch of 8 windows")
    p.add_argument("--overfit-lr", type=float, default=1e-4)
    p.add_argument("--overfit-batch", type=int, default=8)
    p.add_argument("--overfit-fresh-draws", action="store_true")
    p.add_argument("--hidden", type=int, default=S.WAN_KW["hidden"])
    p.add_argument("--action-hidden", type=int, default=S.WAN_KW["action_hidden"])
    p.add_argument("--heads", type=int, default=S.WAN_KW["heads"])
    p.add_argument("--head-dim", type=int, default=S.WAN_KW["head_dim"])
    p.add_argument("--allow-any-micro-batch", action="store_true", help="tests: allow micro-batches other than 8, 4")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    return p


def run_identity(args, keep, data, plan, model_kw):
    """Everything that fixes the batches, the draws, the LR schedule and the model's shape. A resume must match it
    (--max-steps, the micro-batch, thread counts and paths may differ)."""
    return {"config": args.config, "suites": args.suites, "epochs": args.epochs, "buffer_episodes": args.buffer_episodes,
            "global_batch": args.global_batch, "seed": args.seed, "keep": keep, "model_kw": model_kw,
            "in_dim": data.in_dim, "n_episodes": len(data.episodes),
            "n_frames": int(sum(n for _, _, n, _ in data.episodes)), "steps": len(plan),
            "plan_sha256": hashlib.sha256(np.ascontiguousarray(plan).tobytes()).hexdigest()}


def _stop(out, name, e, log, **extra):
    """A non-finite loss: write the stop report and return exit code 2."""
    _write_json(out / name, {"stopped": f"NonFiniteLoss: {e}", **extra})
    log(f"STOP: {e}")
    return 2


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def main(argv=None):
    args = parser().parse_args(argv)
    t0 = time.time()
    log = lambda m: print(f"[train {time.time() - t0:7.0f}s] {m}", flush=True)
    wd = StepWatchdog(args.step_watchdog_s, log, first_s=args.step_watchdog_first_s)   # armed from here (start-up)
    try:
        return _main(args, t0, log, wd)
    finally:
        wd.stop()


def _main(args, t0, log, wd):
    if args.micro_batch not in MICRO_BATCHES and not args.allow_any_micro_batch:
        raise SystemExit(f"--micro-batch must be one of {MICRO_BATCHES}")
    if args.global_batch % args.micro_batch:
        raise SystemExit(f"--micro-batch {args.micro_batch} does not divide --global-batch {args.global_batch}")
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = S.parse_keep(args.keep)

    data = P.Data(args.feats, args.suites.split(","), P.config_sources(args.config), args.t5)
    data.context_dev = data.context.to(device)
    lengths = [n for _, _, n, _ in data.episodes]
    plan = P.make_plan(lengths, args.epochs, args.global_batch, args.buffer_episodes, args.seed)
    total = len(plan)
    stop = min(total, args.max_steps) if args.max_steps else total
    args.horizon = math.ceil(args.buffer_episodes * np.mean(lengths) / args.global_batch) + 1
    mk = model_kw_of(args, keep)
    cfg = {**{k: v for k, v in vars(args).items() if k != "horizon"}, "sources": [P.VAE] + data.sources,
           "dims": data.dims, "in_dim": data.in_dim, "n_episodes": len(data.episodes), "n_frames": int(sum(lengths)),
           "steps": total, "steps_this_run": stop, "tasks": data.tasks, "model_kw": mk["model_kw"],
           "git_commit": git_commit(), "torch": torch.__version__, "fastwam_commit": S.FASTWAM_COMMIT,
           "wan_model_id": S.WAN_ID, "keep": keep, "keep_tag": S.KEEP_TAG if args.keep == S.KEEP_DEFAULT else None,
           "precision": "fp32 weights, bf16 autocast on CUDA, fp32 AdamW",
           "timestep_shift": {"video": 5.0, "action": 1.0},
           "action_norm": data.a_norm.state(), "state_norm": data.s_norm.state(),
           "feature_stats": {k: [m.tolist(), s.tolist()] for k, (m, s) in data.fstats.items()}}
    log(f"{len(data.episodes)} episodes, {sum(lengths)} frames, {len(data.tasks)} tasks, in_dim {data.in_dim}; "
        f"plan {total} optimizer steps of {args.global_batch}, this run to step {stop}, micro-batch "
        f"{args.micro_batch} x {args.global_batch // args.micro_batch}")

    # ---- sanity check 1: overfit one batch (own output, no checkpoint) ----
    if args.overfit_steps:
        model, rep = build_and_init(data, args, device, log)
        model.train()
        opt = S.make_opt(model)
        w = plan[0][:args.overfit_batch]
        keys = {data.episodes[ei][:2] for ei in np.unique(w[:, 0])}
        hb = data.assemble(w, {k: {src: data.load_file(k, src) for src in [P.VAE] + data.sources} for k in keys})
        try:
            res = overfit(model, opt, hb, data, args, device, log)
        except NonFiniteLoss as e:
            return _stop(out, "overfit.json", e, log, windows=w.tolist(), config=cfg)
        res["windows"] = w.tolist()
        _write_json(out / "overfit.json", {**res, "config": cfg})
        log(f"[overfit] first {res[next(k for k in res if k.startswith('first_'))]}, "
            f"last {res[next(k for k in res if k.startswith('last_'))]}, predict-zero {res['predict_zero']}")
        return 0

    args.run_identity = cfg["run_identity"] = run_identity(args, keep, data, plan, mk["model_kw"])
    ckpt = out / "ckpt.pt"
    start, micro_hist = 0, [{"step": 0, "micro_batch": args.micro_batch}]
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        old = state.get("run_identity")
        if old != args.run_identity:
            diff = {k: (old.get(k) if old else None, v) for k, v in args.run_identity.items()
                    if not old or old.get(k) != v}
            raise SystemExit(f"{ckpt} was written by a different run (checkpoint, now): {diff}; use a new --out")
        start = state["step"]
        if start >= stop:
            raise SystemExit(f"{ckpt} is at step {start}, at or past the end of this run (step {stop}"
                             f"{', --max-steps' if args.max_steps and stop == args.max_steps else ', the plan end'}); "
                             f"nothing to train. Use a new --out or a larger --max-steps")
        # config.json of the first process is kept; each resume writes its own
        _write_json(out / f"config-resume-step{start}.json", cfg)
        micro_hist = list(state.get("micro_batch_history", micro_hist))
        model = S.build(data.in_dim, device, len(keep), mk["_build"], args.seed)
        model.load_state_dict(state["model"])
        opt = S.make_opt(model)
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
        log(f"resuming from {ckpt} at step {start}")
        if micro_hist[-1]["micro_batch"] != args.micro_batch:
            log(f"[resume] MICRO-BATCH CHANGED at step {start}: {micro_hist[-1]['micro_batch']} -> "
                f"{args.micro_batch} (global batch stays {args.global_batch}; same draws and per-sample losses, "
                f"gradients equal up to floating-point order)")
            micro_hist.append({"step": start, "micro_batch": args.micro_batch})
    else:
        _write_json(out / "config.json", cfg)
        model, rep = build_and_init(data, args, device, log)
        _write_json(out / "load.json", rep)
        log(f"[load] {rep['counts_by_tag']}")
        opt = S.make_opt(model)
    model.train()

    hist = []
    if args.first_batch_report and (out / "first-batch.json").exists():
        log(f"[first batch] skipped: {out / 'first-batch.json'} exists (the report on the initial model is kept)")
    elif args.first_batch_report:
        hb0 = data.assemble(plan[0], {k: {src: data.load_file(k, src) for src in [P.VAE] + data.sources}
                                      for k in {data.episodes[ei][:2] for ei in np.unique(plan[0][:, 0])}})
        try:
            rep = first_batch_report(model, hb0, data,
                                     draw_step(model, args.seed, 0, len(plan[0]), lat_shape_of(data)),
                                     args.micro_batch, device)
        except NonFiniteLoss as e:
            return _stop(out, "first-batch.json", e, log)
        rep["note"] = "model state at the time of the report: " + (f"resumed at step {start}" if start else "initial")
        _write_json(out / "first-batch.json", rep)
        log(f"[first batch] loss at start {rep['loss_at_start']}")

    try:
        hist, finished = train(model, opt, data, plan, args, device, log, ckpt, start, stop, micro_hist, hist,
                               out / "history.jsonl", wd=wd)
    except NonFiniteLoss as e:
        return _stop(out, "train.json", e, log, history=hist, micro_batch_history=micro_hist,
                     seconds=time.time() - t0)
    if not finished:
        return 3
    n_params = sum(q.numel() for q in model.parameters())
    tmp = out / "model.pt.tmp"
    torch.save({"model": model.state_dict(), "config": cfg}, tmp)
    tmp.replace(out / "model.pt")
    _write_json(out / "train.json", {"n_params": n_params, "start_step_this_process": start, "end_step": stop,
                                     "micro_batch_history": micro_hist, "history": read_history(out / "history.jsonl"),
                                     "seconds_this_process": time.time() - t0})
    # ckpt.pt is kept (PLAN: deleting it needs the user's OK after evaluation has loaded model.pt once)
    log(f"wrote {out / 'model.pt'} and train.json, {n_params} params; ckpt.pt kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
