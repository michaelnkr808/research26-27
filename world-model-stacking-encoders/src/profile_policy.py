"""Where one rollout policy call spends its time, on a cluster GPU without the simulator (LOG 2026-11-03 (b), RUN stop 10).

The pilot rollout spent 85% of its wall time inside Policy.act at batch ~2, yet the GPU read only ~25%. This times the
pieces of Policy.act (rollout.py) on synthetic requests at several batch sizes, for each checkpoint:
  vae      stack the two cameras, Wan 2.2 VAE encode of the current frame
  enc:<s>  one source encoder on both cameras + resize to the 14 x 14 grid
  prep     z-score + concat, T5 context, proprio, starting noise (Policy.start_noise: drawn once per eval seed)
  denoise  infer_actions, 10 joint denoising steps (eager, or replayed from a CUDA graph with --cuda-graph)
  post     actions to CPU + to_exec
plus the whole Policy.act (the pieces should sum to about it), and, per batch size, a sustained loop of Policy.act with
nvidia-smi sampling GPU utilization. That loop is policy-only: in rollouts the GPU also waits while simulators step, so
it is an upper bound on rollout utilization at that batch. A torch.profiler table at batch 2 names the top ops by CUDA
and by CPU time. Torch threads = the pod's CPU quota (rollout.pod_cpus), as rollout.py run sets them.
With --cuda-graph, each checkpoint first compares Policy.act with the graph against the eager path on the same requests
(batch 1, 2, 8, 16; max abs difference of the executed action chunks, and whether they are identical).
Run 1 (LOG 2026-11-03 (d)): eager, CPU noise per call, 24 threads.

Synthetic inputs: random uint8 frames, mid-range proprio, one cached clean instruction. Timing doesn't depend on pixel
values; the outputs are not used.

Example:
  python profile_policy.py --runs /mnt/proxy/runs/pilot2-dinov3-s0 /mnt/proxy/runs/pilot2-dinov3+c-radiov4-s0 \
      --t5 /mnt/proxy/feats/t5 --vae-path /mnt/proxy/models/Wan2.2_VAE.safetensors --tasks /mnt/proxy/lp_tasks.json \
      --batches 1 2 4 8 16 32 --out /mnt/proxy/profile/policy.json
Prints one line starting with RESULT_JSON at the end.
"""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import fastwam_data as FD
import proxy_wam as PW
import rollout as RO


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


def make_reqs(policy, b, instruction, rng):
    lo, hi = (np.asarray(policy.cfg["state_norm"][k], np.float32) for k in ("lo", "hi"))
    return [{"agent": rng.integers(0, 256, (RO.SIZE, RO.SIZE, 3), dtype=np.uint8),
             "wrist": rng.integers(0, 256, (RO.SIZE, RO.SIZE, 3), dtype=np.uint8),
             "state": (lo + hi) / 2, "instruction": instruction, "eval_seed": 41 + i} for i in range(b)]


@torch.no_grad()
def act_pieces(policy, reqs):
    """Policy.act and Featurizer.__call__ (rollout.py) split into timed pieces; same ops in the same order."""
    dev, f, t = policy.device, policy.featurize, {}

    def mark(name, t0):
        sync(dev)
        t[name] = time.perf_counter() - t0
        return time.perf_counter()

    sync(dev)
    t0 = time.perf_counter()
    agent, wrist = np.stack([r["agent"] for r in reqs]), np.stack([r["wrist"] for r in reqs])
    rgb = torch.from_numpy(np.concatenate([agent, wrist], axis=2)).to(dev)
    x = (rgb.permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0)[:, :, None]
    z = f.vae.model.encode(x.to(torch.bfloat16), f.vae.scale)
    feats = [z.float().half().float()]
    t0 = mark("vae", t0)
    for s, enc in f.encs:
        g = []
        for cam in (agent, wrist):
            _, tok = enc.encode(np.ascontiguousarray(cam))
            g.append(FD.grid_to_vae(tok.float(), (PW.GRID[0], PW.GRID[0])))
        feats.append(torch.cat(g, dim=2).to(dev).half().float())
        t0 = mark(f"enc:{s}", t0)
    parts = [feats[0]]
    for src, ft in zip(policy.sources, feats[1:]):
        mu, sd = policy.fstats[src]
        parts.append(((ft - mu) / sd).permute(0, 3, 1, 2)[:, :, None])
    current = torch.cat(parts, 1)
    ctx = torch.stack([policy.context(r["instruction"]) for r in reqs]).to(dev)
    cmask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=dev)
    prop = torch.from_numpy(policy.s_norm(np.stack([r["state"] for r in reqs]))).to(dev)
    nz = [policy.start_noise(r["eval_seed"], current.shape[1]) for r in reqs]
    noise = (torch.stack([n[0] for n in nz]), torch.stack([n[1] for n in nz]))
    t0 = mark("prep", t0)
    if policy.graphed is not None:
        a = policy.graphed(current, ctx, cmask, prop, noise)
    else:
        use_amp = torch.device(dev).type == "cuda"
        with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=use_amp):
            a = PW.infer_actions(policy.model, current, ctx, cmask, prop, noise=noise)
    t0 = mark("denoise", t0)
    RO.to_exec(policy.a_norm.inverse(a.float().cpu().numpy()))
    mark("post", t0)
    return t


def breakdown(policy, reqs, reps):
    """Median seconds per piece and for the whole Policy.act over `reps` calls (after 2 warm-up calls)."""
    dev = policy.device
    for _ in range(2):
        act_pieces(policy, reqs)
        policy.act(reqs)
    rows, whole = [], []
    for _ in range(reps):
        rows.append(act_pieces(policy, reqs))
        sync(dev)
        t0 = time.perf_counter()
        policy.act(reqs)
        sync(dev)
        whole.append(time.perf_counter() - t0)
    out = {k: round(float(np.median([r[k] for r in rows])), 4) for k in rows[0]}
    out["act"] = round(float(np.median(whole)), 4)
    return out


def sustained(policy, reqs, seconds):
    """Policy.act back to back for `seconds`, with nvidia-smi sampling every 500 ms: calls/s and mean GPU util."""
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=subprocess.PIPE, text=True)
    t0, n = time.perf_counter(), 0
    try:
        while time.perf_counter() - t0 < seconds:
            policy.act(reqs)
            n += 1
        sync(policy.device)
        el = time.perf_counter() - t0
    finally:
        smi.terminate()
        lines = smi.communicate()[0].split()
    util = [float(u) for u in lines[2:] if u.replace(".", "", 1).isdigit()]   # skip the first samples (start-up)
    return {"calls": n, "s": round(el, 1), "calls_per_s": round(n / el, 3), "items_per_s": round(n * len(reqs) / el, 2),
            "gpu_util_mean": round(float(np.mean(util)), 1) if util else None, "gpu_util_samples": len(util),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}


def graph_vs_eager(policy, instruction, rng, batches=(1, 2, 8, 16)):
    """Executed action chunks from the CUDA-graph path vs the eager path on the same requests."""
    out, graphed = {}, policy.graphed
    for b in batches:
        reqs = make_reqs(policy, b, instruction, rng)
        a_graph = policy.act(reqs)
        policy.graphed = None
        try:
            a_eager = policy.act(reqs)
        finally:
            policy.graphed = graphed
        d = np.abs(a_graph - a_eager)
        out[b] = {"max_abs_diff": float(d.max()), "identical": bool(np.array_equal(a_graph, a_eager)),
                  "gripper_sign_flips": int((a_graph[..., -1] != a_eager[..., -1]).sum())}
    return out


def profiler_table(policy, reqs, rows=25):
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    for _ in range(2):
        policy.act(reqs)
    with profile(activities=acts) as prof:
        for _ in range(3):
            policy.act(reqs)
        sync(policy.device)
    ka = prof.key_averages()
    by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    n_launch = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    return (ka.table(sort_by=by, row_limit=rows), ka.table(sort_by="self_cpu_time_total", row_limit=rows),
            n_launch // 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="proxy_wam.py run dirs (model.pt)")
    p.add_argument("--t5", required=True)
    p.add_argument("--vae-path", required=True)
    p.add_argument("--tasks", required=True, help="lp_tasks.json (a clean instruction is taken from it)")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--sustain-s", type=float, default=45.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t_start = time.time()
    log = lambda m: print(f"[profile {time.time() - t_start:6.0f}s] {m}", flush=True)
    clean = json.loads(Path(args.tasks).read_text())["clean"]
    instruction = RO.base_language(sorted(clean["libero_goal"])[0])
    torch.set_num_threads(RO.pod_cpus())
    threads0 = torch.get_num_threads()
    log(f"device {args.device}, torch threads {threads0}, cuda graph {args.cuda_graph}, instruction '{instruction}'")
    if torch.cuda.is_available():
        log(f"GPU {torch.cuda.get_device_name(0)}")
    result = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch_threads": threads0,
              "cuda_graph": args.cuda_graph, "instruction": instruction, "runs": {}}
    for run in args.runs:
        name = Path(run).name
        res = result["runs"][name] = {"by_batch": {}}
        t0 = time.time()
        policy = RO.Policy.load(run, args.t5, args.vae_path, args.device, cuda_graph=args.cuda_graph)
        res["load_s"] = round(time.time() - t0, 1)
        res["config"] = policy.cfg["config"]
        log(f"{name}: config {policy.cfg['config']}, loaded in {res['load_s']}s")
        rng = np.random.default_rng(0)
        if args.cuda_graph:
            try:
                res["graph_vs_eager"] = graph_vs_eager(policy, instruction, rng)
                log(f"{name} graph vs eager: {res['graph_vs_eager']}")
            except Exception:
                res["graph_vs_eager"] = {"error": traceback.format_exc()}
                log(f"{name} graph vs eager FAILED:\n{traceback.format_exc()}")
        for b in args.batches:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                reqs = make_reqs(policy, b, instruction, rng)
                bd = breakdown(policy, reqs, args.reps)
                su = sustained(policy, reqs, args.sustain_s)
                res["by_batch"][b] = {"median_s": bd, **su}
                log(f"{name} batch {b}: " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in bd.items()) +
                    f" | sustained {su['items_per_s']} req/s, GPU {su['gpu_util_mean']}% "
                    f"({su['gpu_util_samples']} samples), peak {su['peak_mem_gb']} GB")
            except Exception:
                res["by_batch"][b] = {"error": traceback.format_exc()}
                log(f"{name} batch {b} FAILED:\n{traceback.format_exc()}")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        try:
            reqs = make_reqs(policy, 2, instruction, rng)
            cuda_tab, cpu_tab, launches = profiler_table(policy, reqs)
            res["batch2_kernel_launches_per_call"] = launches
            print(f"=== {name} batch 2: top ops by self CUDA time (3 calls)\n{cuda_tab}", flush=True)
            print(f"=== {name} batch 2: top ops by self CPU time (3 calls)\n{cpu_tab}", flush=True)
            log(f"{name} batch 2: ~{launches} kernel launches per call")
        except Exception:
            res["profiler_error"] = traceback.format_exc()
            log(f"{name} profiler FAILED:\n{traceback.format_exc()}")
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
