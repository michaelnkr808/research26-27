"""Proxy world action model for the calibration round (decision 003; experiments/2026-10-27-proxy-calibration/PLAN.md).

This round is a proxy replication (decision 003 amendment, 2026-10-28): results from this model are about the proxy and
decide whether a higher-fidelity, multi-seed WAM replication is worth running; they can't show what happened in the
paper's model.

A small copy of the paper's default WAM (liu2026what Sec. 3.1, App. B.1 joint video-action denoising) built from
Fast-WAM's own modules (FastWAM 7faa711, MIT): FastWAMJoint = video expert (WanVideoDiT) + action expert (ActionDiT) in
a mixture of transformers with joint attention; video tokens attend to video only (first latent frame attends only to
itself), action tokens attend to all video and action tokens. T5 instruction tokens plus a proprio token by
cross-attention. Flow matching with Fast-WAM's schedulers (video shift 5, action shift 1, timestep-dependent loss
weights), the first latent frame kept clean. The experts start from random weights: the one gap to the paper we can't
close (it starts from Wan 2.2 and a pretrained action expert).

External features enter by channel-wise concatenation (App. B.2, E.1): per latent step and 14 x 28 grid cell, the
input is [Wan 2.2 VAE latent (48) | source 1 | source 2 ...], and the expanded patch embedding / head predict all of it.
Observed features (latent step 0) stay clean; future ones are noised and denoised with the VAE latents (shared
timestep). Feature time steps: step 0 = frame t, step 1 = mean of frames t+4..t+16, step 2 = mean of t+20..t+32
(the paper's REPA alignment averages the 8 future slices in two groups of four, App. B.2; that channel-wise
concatenation aligns time the same way is our reading). Loss = video FM (VAE channels) + sum over sources of that
source's FM loss (Eq. 9, weight 1 each) + action FM (weight 1.0, Table 6); each a per-sample mean over its own
channels and the valid future steps, then times the scheduler's weight.

Data: Fast-WAM's handling throughout (fastwam_data.py, extract_fastwam.py): 33-step windows, indices clamped to the
episode; padded actions get delta dims (ee pose, 6) zeroed and keep the clamped gripper; actions and proprio
min/max-normalized to [-1, 1] over all training frames, clamped to +-5; padded steps out of the loss. Feature channels
are z-scored with training-set statistics: the paper doesn't say how it scales sources (our choice, flagged in PLAN).

Batches come from a fixed plan: each epoch permutes the episodes (seed only, independent of the config), takes them
--buffer-episodes at a time, shuffles all their windows (plus the previous chunk's leftovers) and cuts batches. Each
episode is read from disk once per epoch, not once per window (PLAN: I/O ~9x lower). Batches mix windows from a few
dozen episodes rather than the whole dataset: a deviation from a full shuffle, logged in PLAN.

Example:
  python proxy_wam.py --feats /feats --suites libero_spatial,libero_object,libero_goal,libero_10 --t5 /feats/t5 \
      --config dinov3+c-radiov4 --seed 0 --out /runs/d+r-s0
"""

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT

import t5_cache
from diagnostics import git_commit

HORIZON = 32                  # actions per chunk (Table 6)
NUM_FRAMES = 9                # video frames per view, offsets 0, 4, ..., 32 (Table 6)
STRIDE = 4                    # action_video_freq_ratio
LATENT_T = 3                  # 1 + (NUM_FRAMES - 1) / 4
GRID = (14, 28)               # VAE latent grid of the 224 x 448 two-camera frame
PATCH = (1, 2, 2)             # Wan 2.2 TI2V-5B patch size
VAE, VAE_DIM = "vae", 48
TEXT_DIM, FREQ_DIM = 4096, 256
DELTA_DIMS = 6                # action dims 0-5 are deltas (Fast-WAM delta_action_dim_mask), 6 is the gripper
INFER_STEPS = 10              # App. E.3


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class MinMax:
    """Fast-WAM's LinearNormalizer, mode min/max: to [-1, 1], dims with range < 1e-4 shifted to 0, clamp to +-5."""

    def __init__(self, lo, hi):
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        rng = hi - lo
        flat = rng < 1e-4
        rng[flat] = 2.0
        self.scale = 2.0 / rng
        self.offset = -1.0 - self.scale * lo
        self.offset[flat] = 0.0 - lo[flat]
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        return np.clip(x * self.scale + self.offset, -5.0, 5.0).astype(np.float32)

    def inverse(self, y):
        return ((y - self.offset) / self.scale).astype(np.float32)

    def state(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def config_sources(config):
    """'none' -> []; 'dinov3+c-radiov4' -> ['dinov3', 'c-radiov4'] (also the channel order)."""
    return [] if config == "none" else config.split("+")


def action_window(action, t, n):
    """(B, HORIZON, 7) raw actions for windows starting at t in episodes of length n, and is_pad (B, HORIZON).
    Indices past the end are clamped; padded steps get the delta dims zeroed (Fast-WAM fastwam_processor.py)."""
    idx = t[:, None] + np.arange(HORIZON)[None]
    pad = idx >= n[:, None]
    a = action[np.minimum(idx, n[:, None] - 1)].copy()
    a[..., :DELTA_DIMS][pad] = 0.0
    return a, pad


def latent_pad(t, n):
    """(B, LATENT_T - 1): a future latent step is padding only if all 4 of its frames are (fastwam.py:420-457)."""
    f = t[:, None] + STRIDE * np.arange(1, NUM_FRAMES)[None]
    return (f >= n[:, None]).reshape(-1, LATENT_T - 1, STRIDE).all(2)


def feature_steps(feat, t):
    """(B, LATENT_T, 14, 28, D) float32 from per-frame features (n, 14, 28, D): frame t, then the mean of each group of
    four future frames (t+4..t+16, t+20..t+32), indices clamped to the episode (Fast-WAM pads with the last frame)."""
    n = len(feat)
    idx = np.minimum(t[:, None] + STRIDE * np.arange(NUM_FRAMES)[None], n - 1)       # (B, 9)
    out = [feat[idx[:, 0]].astype(np.float32)]
    for g in range(LATENT_T - 1):
        cols = idx[:, 1 + STRIDE * g: 1 + STRIDE * (g + 1)]
        out.append(feat[cols].astype(np.float32).mean(1))
    return np.stack(out, 1)


class Data:
    """Episodes of the given suites from the extract_fastwam.py cache (meta, VAE latents, each source's features)."""

    def __init__(self, feats, suites, sources, t5_dir):
        self.feats, self.sources = Path(feats), list(sources)
        self.episodes = []                                        # (suite, episode_id, length, task_id)
        acts, states, tasks = [], [], []
        for s in suites:
            m = np.load(self.feats / "meta" / f"{s}.npz")
            self._meta_check(m)
            off = 0
            for e, n, task in zip(m["episode_ids"], m["lengths"], m["tasks"]):
                if task not in tasks:
                    tasks.append(str(task))
                self.episodes.append((s, int(e), int(n), tasks.index(str(task))))
                acts.append(m["action"][off:off + n])
                states.append(m["state"][off:off + n])
                off += n
        self.tasks = tasks
        self.action = {k[:2]: a for k, a in zip(self.episodes, acts)}
        self.state = {k[:2]: a for k, a in zip(self.episodes, states)}
        all_a, all_s = np.concatenate(acts), np.concatenate(states)
        self.a_norm = MinMax(all_a.min(0), all_a.max(0))
        self.s_norm = MinMax(all_s.min(0), all_s.max(0))
        self.dims = {VAE: VAE_DIM}
        self.fstats = {}
        for src in self.sources:
            tot = None
            for s, e, n, _ in self.episodes:
                st = np.load(self._path(src, s, e).with_suffix(".stats.npy"))
                tot = st if tot is None else tot + st
            cnt, sm, sq = tot
            mu = sm / cnt
            sd = np.sqrt(np.maximum(sq / cnt - mu * mu, 0.0)) + 1e-6
            self.fstats[src] = (mu.astype(np.float32), sd.astype(np.float32))
            self.dims[src] = len(mu)
        self.context = torch.stack([t5_cache.load_context(t5_dir, t)[0] for t in tasks]) if t5_dir else None

    @staticmethod
    def _meta_check(m):
        assert m["action"].shape[1] == 7 and m["state"].shape[1] == 8, (m["action"].shape, m["state"].shape)
        assert m["lengths"].sum() == len(m["action"]) == len(m["episode_index"])

    def _path(self, src, suite, ep):
        return self.feats / src / suite / f"ep{ep:06d}.npy"

    @property
    def in_dim(self):
        return sum(self.dims.values())

    def channel_slices(self):
        """{source: slice} in the concatenation order: VAE first, then the config's sources."""
        out, c = {}, 0
        for k, d in self.dims.items():
            out[k] = slice(c, c + d)
            c += d
        return out

    def load_episode(self, key):
        suite, ep = key
        return {src: np.load(self._path(src, suite, ep)) for src in [VAE] + self.sources}

    def assemble(self, windows, cache):
        """Host-side batch for windows [(episode index, t)]: latents float16 (B, 3, 14, 28, C) un-normalized,
        normalized actions and proprio, masks, task ids."""
        B = len(windows)
        lat = np.empty((B, LATENT_T) + GRID + (self.in_dim,), np.float16)
        act = np.empty((B, HORIZON, 7), np.float32)
        apad = np.empty((B, HORIZON), bool)
        lpad = np.empty((B, LATENT_T - 1), bool)
        prop = np.empty((B, 8), np.float32)
        task = np.empty(B, np.int64)
        sl = self.channel_slices()
        windows = np.asarray(windows)
        for ei in np.unique(windows[:, 0]):
            rows = np.flatnonzero(windows[:, 0] == ei)
            s, e, n, ti = self.episodes[ei]
            t = windows[rows, 1]
            arr = cache[(s, e)]
            lat[rows, ..., sl[VAE]] = arr[VAE][t]
            for src in self.sources:
                lat[rows, ..., sl[src]] = feature_steps(arr[src], t)
            nn_ = np.full(len(t), n)
            a, p = action_window(self.action[(s, e)], t, nn_)
            act[rows], apad[rows] = self.a_norm(a), p
            lpad[rows] = latent_pad(t, nn_)
            prop[rows] = self.s_norm(self.state[(s, e)][t])
            task[rows] = ti
        return {"latents": lat, "action": act, "action_is_pad": apad, "latent_is_pad": lpad, "proprio": prop,
                "task": task}


def make_plan(lengths, epochs, bs, buffer_eps, seed):
    """(S, bs, 2) int64 batches of (episode index, t). Depends on the seed and the episode list only.
    Every window appears once per epoch; leftovers (< bs) carry into the next chunk, so all but < bs windows of the
    whole run are used."""
    rng = np.random.default_rng(seed)
    carry = np.empty((0, 2), np.int64)
    out = []
    for _ in range(epochs):
        order = rng.permutation(len(lengths))
        for c in range(0, len(order), buffer_eps):
            w = [np.stack([np.full(lengths[i], i), np.arange(lengths[i])], 1) for i in order[c:c + buffer_eps]]
            w = np.concatenate([carry] + w)
            w = w[rng.permutation(len(w))]
            k = len(w) // bs
            out.append(w[:k * bs].reshape(k, bs, 2))
            carry = w[k * bs:]
    return np.concatenate(out) if out else np.empty((0, bs, 2), np.int64)


class Loader:
    """Background thread that walks the plan from `start`, keeps the episodes it needs in RAM (reading each from disk
    when first needed and dropping it once no batch in the next `horizon` uses it), and queues assembled batches."""

    def __init__(self, data, plan, start=0, prefetch=3, horizon=None):
        self.data, self.plan, self.start = data, plan, start
        self.horizon = horizon or 1
        self.q = queue.Queue(maxsize=prefetch)
        self.read_s = 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        try:
            cache = {}
            for s in range(self.start, len(self.plan)):
                b = self.plan[s]
                for ei in np.unique(b[:, 0]):
                    key = self.data.episodes[ei][:2]
                    if key not in cache:
                        t = time.time()
                        cache[key] = self.data.load_episode(key)
                        self.read_s += time.time() - t
                self.q.put(self.data.assemble(b, cache))
                ahead = {self.data.episodes[ei][:2] for ei in np.unique(self.plan[s + 1:s + 1 + self.horizon, :, 0])}
                for key in [k for k in cache if k not in ahead]:
                    del cache[key]
            self.q.put(None)
        except BaseException as e:      # surface loader errors in the training thread
            self.q.put(e)

    def __iter__(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            if isinstance(b, BaseException):
                raise b
            yield b


def to_device(bt, data, device):
    """GPU tensors: latents (B, C, 3, 14, 28) float32 with source channels z-scored; context gathered by task."""
    lat = torch.from_numpy(bt["latents"]).to(device, non_blocking=True).float()
    for src, sl in data.channel_slices().items():
        if src != VAE:
            mu, sd = data.fstats[src]
            lat[..., sl] = (lat[..., sl] - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device)
    out = {"latents": lat.permute(0, 4, 1, 2, 3).contiguous(),
           **{k: torch.from_numpy(bt[k]).to(device) for k in ["action", "action_is_pad", "latent_is_pad", "proprio"]}}
    task = torch.from_numpy(bt["task"]).to(device)
    out["context"] = data.context_dev[task]
    out["context_mask"] = torch.ones(out["context"].shape[:2], dtype=torch.bool, device=device)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentsOnly(torch.nn.Module):
    """Stands in for FastWAM's VAE: latents come from the cache, so nothing is encoded. Carries the Wan 2.2 VAE's
    factors that FastWAM reads."""
    temporal_downsample_factor, upsampling_factor = 4, 16


def build_model(in_dim, device, hidden=384, action_hidden=128, layers=8, heads=3, head_dim=128):
    """FastWAMJoint at small size. Ratios as Fast-WAM's config: action width = video width / 3, FFN 14336/3072 x
    (video) and 4x (action), equal heads and head dim in both experts (MoT requires it). Head dim 128 as Wan; Wan's 3D
    RoPE splits it (d - 2 (d // 3), d // 3, d // 3) and needs d // 3 even."""
    assert (head_dim // 3) % 2 == 0, f"head_dim {head_dim}: Wan 3D RoPE needs head_dim // 3 even (e.g. 24, 48, 96, 128)"
    video = WanVideoDiT(hidden_dim=hidden, in_dim=in_dim, ffn_dim=round(hidden * 14336 / 3072), out_dim=in_dim,
                        text_dim=TEXT_DIM, freq_dim=FREQ_DIM, eps=1e-6, patch_size=PATCH, num_heads=heads,
                        attn_head_dim=head_dim, num_layers=layers, has_image_input=False, seperated_timestep=True,
                        require_vae_embedding=False, require_clip_embedding=False, fuse_vae_embedding_in_latents=True,
                        action_conditioned=False, video_attention_mask_mode="first_frame_causal")
    action = ActionDiT(hidden_dim=action_hidden, action_dim=7, ffn_dim=4 * action_hidden, text_dim=TEXT_DIM,
                       freq_dim=FREQ_DIM, eps=1e-6, num_heads=heads, attn_head_dim=head_dim, num_layers=layers)
    mot = MoT({"video": video, "action": action})
    return FastWAMJoint(video, action, mot, vae=LatentsOnly(), text_dim=TEXT_DIM, proprio_dim=8, device=device,
                        torch_dtype=torch.float32, video_train_shift=5.0, video_infer_shift=5.0,
                        action_train_shift=1.0, action_infer_shift=1.0, loss_lambda_video=1.0, loss_lambda_action=1.0)


def _mask(model, batch_size, device):
    tpf = (GRID[0] // PATCH[1]) * (GRID[1] // PATCH[2])
    return model._build_mot_attention_mask(video_seq_len=LATENT_T * tpf, action_seq_len=HORIZON,
                                           video_tokens_per_frame=tpf, device=device)


def training_loss(model, bt, slices):
    """FastWAM.training_loss on cached latents, with the video loss split by source. Same random draws in the same
    order (video noise, video t, action noise, action t), so with only VAE channels it equals Fast-WAM's loss."""
    x, act = bt["latents"], bt["action"]
    B = x.shape[0]
    noise_v = torch.randn_like(x)
    tv = model.train_video_scheduler.sample_training_t(batch_size=B, device=x.device, dtype=x.dtype)
    lat = model.train_video_scheduler.add_noise(x, noise_v, tv)
    target_v = model.train_video_scheduler.training_target(x, noise_v, tv)
    lat[:, :, 0:1] = x[:, :, 0:1]
    noise_a = torch.randn_like(act)
    ta = model.train_action_scheduler.sample_training_t(batch_size=B, device=x.device, dtype=act.dtype)
    noisy_a = model.train_action_scheduler.add_noise(act, noise_a, ta)
    target_a = model.train_action_scheduler.training_target(act, noise_a, ta)
    ctx, cmask = model._append_proprio_to_context(bt["context"], bt["context_mask"], bt["proprio"])
    pred_v, pred_a = model._joint_denoise_core(latents_video=lat, latents_action=noisy_a, timestep_video=tv,
                                               timestep_action=ta, context=ctx, context_mask=cmask,
                                               attention_mask=_mask(model, B, x.device),
                                               fuse_vae_embedding_in_latents=True, action_condition=act)
    pred_v, target_v = pred_v[:, :, 1:], target_v[:, :, 1:]
    valid_v = (~bt["latent_is_pad"]).float()
    w_v = model.train_video_scheduler.training_weight(tv)
    parts = {}
    for src, sl in slices.items():
        tok = F.mse_loss(pred_v[:, sl].float(), target_v[:, sl].float(), reduction="none").mean(dim=(1, 3, 4))
        per = (tok * valid_v).sum(1) / valid_v.sum(1).clamp(min=1.0)
        parts[src] = (per * w_v).mean()
    tok = F.mse_loss(pred_a.float(), target_a.float(), reduction="none").mean(dim=2)
    valid_a = (~bt["action_is_pad"]).float()
    per = (tok * valid_a).sum(1) / valid_a.sum(1).clamp(min=1.0)
    parts["action"] = (per * model.train_action_scheduler.training_weight(ta)).mean()
    return sum(parts.values()), parts


@torch.no_grad()
def infer_actions(model, current, context, context_mask, proprio, steps=INFER_STEPS, seed=None):
    """Normalized action chunks (B, HORIZON, 7), as FastWAMJoint.infer_action: video and actions start from noise and
    are denoised jointly over `steps` steps, the first latent step reset to the observation after every step.
    current: (B, C, 1, 14, 28) observed latents with source channels already z-scored."""
    B, C = current.shape[:2]
    dev = current.device
    g = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    lat = torch.randn((B, C, LATENT_T) + GRID, generator=g).to(dev)
    a = torch.randn((B, HORIZON, 7), generator=g).to(dev)
    lat[:, :, 0:1] = current
    ctx, cmask = model._append_proprio_to_context(context, context_mask, proprio)
    tv, dv = model.infer_video_scheduler.build_inference_schedule(steps, device=dev, dtype=lat.dtype)
    ta, da = model.infer_action_scheduler.build_inference_schedule(steps, device=dev, dtype=a.dtype)
    mask = _mask(model, B, dev)
    for t_v, d_v, t_a, d_a in zip(tv, dv, ta, da):
        pv, pa = model._joint_denoise_core(latents_video=lat, latents_action=a, timestep_video=t_v.expand(B),
                                           timestep_action=t_a.expand(B), context=ctx, context_mask=cmask,
                                           attention_mask=mask, fuse_vae_embedding_in_latents=True,
                                           action_condition=None)
        lat = model.infer_video_scheduler.step(pv, d_v, lat)
        a = model.infer_action_scheduler.step(pa, d_a, a)
        lat[:, :, 0:1] = current
    return a


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lr_at(step, total, base=1e-4, floor=1e-6, warmup=0.05):
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    q = (step - w) / max(1, total - w)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * q))


def train(data, plan, seed, device, log, model_kw=None, ckpt=None, ckpt_every=0, start=0, state=None, amp=True,
          horizon=None):
    torch.manual_seed(seed)
    model = build_model(data.in_dim, device, **(model_kw or {}))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    data.context_dev = data.context.to(device)
    slices = data.channel_slices()
    total = len(plan)
    use_amp = amp and torch.device(device).type == "cuda"
    hist, t0, wait = [], time.time(), 0.0
    loader = Loader(data, plan, start, horizon=horizon)
    tw = time.time()
    for s, hb in zip(range(start, total), loader):
        wait += time.time() - tw
        for gp in opt.param_groups:
            gp["lr"] = lr_at(s, total)
        bt = to_device(hb, data, device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_amp):
            loss, parts = training_loss(model, bt, slices)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % max(1, total // 50) == 0 or s == total - 1:
            rec = {"step": s, "lr": lr_at(s, total), "loss": float(loss),
                   **{f"loss/{k}": float(v) for k, v in parts.items()},
                   "sec": time.time() - t0, "data_wait_s": wait, "disk_read_s": loader.read_s}
            hist.append(rec)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
        if ckpt and ckpt_every and (s + 1) % ckpt_every == 0 and s + 1 < total:
            save_ckpt(ckpt, model, opt, s + 1)
        tw = time.time()
    return model, hist


def save_ckpt(path, model, opt, step):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="root written by extract_fastwam.py")
    p.add_argument("--suites", required=True, help="comma-separated, e.g. libero_spatial,libero_object,libero_goal,libero_10")
    p.add_argument("--t5", required=True, help="dir written by t5_cache.py")
    p.add_argument("--config", required=True, help="none, or encoders joined by +, e.g. dinov3+c-radiov4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=None, help="cap on steps (smoke tests and the pilot)")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    log = lambda m: print(f"[proxy {time.time() - t0:7.0f}s] {m}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(args.feats, args.suites.split(","), config_sources(args.config), args.t5)
    lengths = [n for _, _, n, _ in data.episodes]
    plan = make_plan(lengths, args.epochs, args.batch_size, args.buffer_episodes, args.seed)
    if args.max_steps:
        plan = plan[:args.max_steps]
    model_kw = {"hidden": args.hidden, "action_hidden": args.hidden // 3, "layers": args.layers, "heads": args.heads,
                "head_dim": args.head_dim}
    cfg = {**vars(args), "sources": [VAE] + data.sources, "dims": data.dims, "in_dim": data.in_dim,
           "n_episodes": len(data.episodes), "n_frames": int(sum(lengths)), "steps": len(plan), "tasks": data.tasks,
           "model_kw": model_kw, "git_commit": git_commit(), "torch": torch.__version__,
           "action_norm": data.a_norm.state(), "state_norm": data.s_norm.state(),
           "feature_stats": {k: [m.tolist(), s.tolist()] for k, (m, s) in data.fstats.items()}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    ckpt = out / "ckpt.pt"
    state, start = None, 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        start = state["step"]
        log(f"resuming at step {start}")
    log(f"{len(data.episodes)} episodes, {sum(lengths)} frames, {len(data.tasks)} tasks, in_dim {data.in_dim} "
        f"({data.dims}), {len(plan)} steps, loaded in {time.time() - t0:.0f}s")
    # look ahead about one chunk of batches before dropping an episode from RAM
    horizon = math.ceil(args.buffer_episodes * np.mean(lengths) / args.batch_size) + 1
    model, hist = train(data, plan, args.seed, args.device, log, model_kw, ckpt, args.ckpt_every, start, state,
                        horizon=horizon)
    n_params = sum(q.numel() for q in model.parameters())
    torch.save({"model": model.state_dict(), "config": cfg}, out / "model.pt")
    (out / "train.json").write_text(json.dumps({"n_params": n_params, "history": hist,
                                                "seconds": time.time() - t0}, indent=1))
    if ckpt.exists():
        ckpt.unlink()
    log(f"wrote {out}, {n_params} params")


if __name__ == "__main__":
    main()
