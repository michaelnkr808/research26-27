"""Cache the proxy policy's inputs from Fast-WAM's LIBERO data (decision 003, proxy-calibration PLAN step 4).

One run handles one source for one suite (shard with --shard i/n). Output, all float16:
  <out>/meta/<suite>.npz                 per frame: action (N, 7), state (N, 8), episode_index (N,);
                                         per episode: episode_ids, lengths, tasks
  <out>/vae/<suite>/ep<idx>.npy          (n, 3, 14, 28, 48)  Wan 2.2 VAE latents of the 9-frame clip that starts at
                                         each frame t (frames t, t+4, ..., t+32 clamped to the episode; 3 latent
                                         steps), exactly what Fast-WAM's training computes on the fly
  <out>/<encoder>/<suite>/ep<idx>.npy    (n, 14, 28, D)  final-layer patch tokens per frame, resized bilinearly to the
                                         VAE's 14 x 14 grid per camera, cameras side by side (agentview | wrist)
  <out>/<encoder>/<suite>/ep<idx>.stats.npy  (3, D) float64: count, sum, sum of squares per channel (z-score stats)
  <out>/<source>/<suite>/_meta.*.json    provenance

The encoders see Fast-WAM's 224 x 224 frame (rounded to uint8) and apply their own resize and normalization
(extract_features.py). The paper doesn't state encoder input resolution: our choice, logged in PLAN.

The VAE is Fast-WAM's own loader and class (FastWAM 7faa711 installed, MIT): Wan2.2_VAE weights, as Fast-WAM's
default `redirect_common_files` downloads them (ModelScope DiffSynth-Studio/Wan-Series-Converted-Safetensors).

Example:
  python extract_fastwam.py --root /data/lerobot_v30/libero_goal_no_noops_lerobot --suite libero_goal \
      --source vae --vae-path /models/Wan2.2_VAE.safetensors --out /feats --shard 0/4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import fastwam_data as FD
from diagnostics import git_commit

LATENT_GRID = (14, 28)   # 224 x 448 / 16


def save_npy(path, arr):
    """Atomic write, so a killed Job never leaves a truncated episode that a rerun would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def write_meta(eps, out, suite):
    path = Path(out) / "meta" / f"{suite}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, action=np.concatenate([e["action"] for e in eps]),
             state=np.concatenate([e["state"] for e in eps]),
             episode_index=np.concatenate([[e["episode_index"]] * e["length"] for e in eps]),
             episode_ids=np.array([e["episode_index"] for e in eps]),
             lengths=np.array([e["length"] for e in eps]), tasks=np.array([e["task"] for e in eps]))
    return path


def load_vae(path, device, dtype):
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    vae = _load_registered_model(path, "wan_video_vae", torch_dtype=dtype, device=device).eval()
    assert vae.model.z_dim == 48 and vae.upsampling_factor == 16 and vae.temporal_downsample_factor == 4
    return vae


@torch.no_grad()
def vae_episode(vae, cams224, batch, device, dtype):
    n = cams224[FD.CAMS[0]].shape[0]
    out = []
    for s in range(0, n, batch):
        clip = FD.video_clips(cams224, np.arange(s, min(n, s + batch))).to(device, dtype)
        z = vae.model.encode(clip, vae.scale)                       # (B, 48, 3, 14, 28)
        assert tuple(z.shape[2:]) == (3,) + LATENT_GRID, z.shape
        out.append(z.permute(0, 2, 3, 4, 1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


@torch.no_grad()
def encoder_episode(enc, cams224, batch):
    per_cam = []
    for c in FD.CAMS:
        u8 = (cams224[c] * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        g = []
        for s in range(0, len(u8), batch):
            _, tok = enc.encode(np.ascontiguousarray(u8[s:s + batch]))
            g.append(FD.grid_to_vae(tok.float(), (LATENT_GRID[0], LATENT_GRID[0])).cpu())
        per_cam.append(torch.cat(g))
    return torch.cat(per_cam, dim=2).numpy().astype(np.float16)    # (n, 14, 28, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="one suite of lerobot_v30 (e.g. .../libero_goal_no_noops_lerobot)")
    p.add_argument("--suite", required=True)
    p.add_argument("--source", required=True, help="meta, vae, or an encoder name from extract_features.ENCODERS")
    p.add_argument("--out", required=True)
    p.add_argument("--vae-path", default=None)
    p.add_argument("--shard", default="0/1", help="i/n: this run takes episodes i, i+n, ...")
    p.add_argument("--max-episodes", type=int, default=None, help="smoke tests")
    p.add_argument("--episodes", default=None, help="comma-separated episode ids (smoke tests)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[extract {time.time() - t0:7.0f}s] {m}", flush=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    eps = FD.read_suite(args.root)
    log(f"{args.suite}: {len(eps)} episodes, {sum(e['length'] for e in eps)} frames")
    if args.source == "meta":
        log(f"wrote {write_meta(eps, args.out, args.suite)}")
        return
    if args.episodes:
        keep = {int(x) for x in args.episodes.split(",")}
        eps = [e for e in eps if e["episode_index"] in keep]
    i, n = map(int, args.shard.split("/"))
    eps = eps[i::n][:args.max_episodes]

    out_dir = Path(args.out) / args.source / args.suite
    if args.source == "vae":
        assert args.vae_path, "--vae-path is required for the VAE"
        model = load_vae(args.vae_path, args.device, dtype)
        info = {"model": "Wan2.2_VAE (FastWAM loader)", "file": Path(args.vae_path).name}
        run = lambda c: vae_episode(model, c, args.batch, args.device, dtype)
    else:
        from extract_features import ENCODERS
        enc = ENCODERS[args.source](args.device, dtype)
        info = {"model": enc.hf_id, "resolution": enc.resolution}
        run = lambda c: encoder_episode(enc, c, args.batch)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"_meta.shard{i}of{n}.json").write_text(json.dumps({
        **info, "source": args.source, "suite": args.suite, "dtype": args.dtype, "device": args.device,
        "git_commit": git_commit(), "grid": list(LATENT_GRID), "cams": FD.CAMS, "frame_size": FD.SIZE,
        "torch": torch.__version__}, indent=1))

    done = 0
    for e in eps:
        path = out_dir / f"ep{e['episode_index']:06d}.npy"
        if path.exists():
            continue
        cams224 = {c: FD.to_224(f) for c, f in FD.decode_episode(args.root, e).items()}
        arr = run(cams224)
        assert len(arr) == e["length"], (len(arr), e["length"])
        if args.source != "vae":
            # per-channel count, sum, sum of squares over frames and positions: the proxy's feature z-score
            # statistics, without a second pass over the cache. Written before the features, so an episode file
            # never exists without its stats.
            x = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
            save_npy(path.with_suffix(".stats.npy"), np.stack([np.full(x.shape[1], len(x)), x.sum(0), (x * x).sum(0)]))
        save_npy(path, arr)
        done += e["length"]
        log(f"ep {e['episode_index']}: {arr.shape} ({done / (time.time() - t0):.1f} frames/s overall)")
    log("done")


if __name__ == "__main__":
    main()
