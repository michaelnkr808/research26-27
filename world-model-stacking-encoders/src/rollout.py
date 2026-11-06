"""Closed-loop rollouts of a trained proxy checkpoint in LIBERO and LIBERO-Plus (proxy-calibration RUN.md step 7).

This round is a proxy replication: success rates from this code are about the proxy (decision 003 amendment).

Follows Fast-WAM's LIBERO eval (FastWAM 7faa711 experiments/libero/eval_libero_single.py, libero_utils.py) and the
paper's composition evals (liu2026what App. E.3):
  - env: OffScreenRenderEnv rendered at 256 x 256; env.seed(eval seed) before reset; reset, then set_init_state;
    30 wait steps with the dummy action [0]*6 + [-1] (their `done` ignored, as Fast-WAM); then replan every 10 steps
    from a 32-step chunk, no action ensembling; max 400 steps (spatial, object, goal) / 700 (libero_10), success =
    the env's `done`. Global seeds (random, numpy, torch) = eval seed at the start of every episode, as Fast-WAM sets
    them once per process before its single episode per LIBERO-Plus task.
  - observation: both cameras rotated 180 deg ([::-1, ::-1]), center-crop-resized to 224 (PIL bilinear) as uint8;
    proprio = eef pos (3) + axis-angle of eef quat (3) + gripper qpos (2), min/max-normalized with the training stats.
  - model input, as training (proxy_wam.py, extract_fastwam.py): the Wan 2.2 VAE latent of the current two-camera frame
    [agentview | wrist] (224 x 448, x * 2/255 - 1), and each source's grid from the same uint8 frames, resized to
    14 x 14 per camera and z-scored with the training stats; both rounded through float16 as the training cache is.
    T5 context of the instruction from the t5_cache.py cache. 10 denoising steps.
  - noise: Fast-WAM reseeds both generators (video, action) with the eval seed at every inference call; done here per
    episode, so an episode's result doesn't depend on which other episodes share its GPU batch. Reseeding gives the same
    tensors every call, so Policy draws them once per (eval seed, channels) and keeps them on the device.
  - actions: denormalized, then the gripper as Fast-WAM: g -> 2g - 1, sign flipped, binarized (sign). In the dataset
    gripper 1 = open (fingers open ~10 steps later, checked 2026-10-28 on libero_goal); in the sim -1 = open.
LIBERO-Plus (github.com/sylvestf/LIBERO-plus @ 4976dc3): classification id k = task index k - 1 in the suite's default
task order (asserted by name); init state 0 (1 trial per task, LIBERO-Plus README). Instruction: for non-language
perturbations the env's language is the task name plus an appended perturbation description ("... table 1",
"... view 0 0 100 0 0 initstate 0 noise 45"), stripped back to the base task's instruction (App. C.2); rewritten
instructions (category Language Instructions, `_language_` in the name) are kept as the bddl states them.
Clean LIBERO: the 10 base tasks per suite (base bddl and init files, which LIBERO-Plus keeps), init state = trial.

Throughput (RUN.md "Rollouts and the 40% GPU floor"): --envs simulator worker processes step their own episodes on CPU
and send observations to the main process, which batches every pending inference call on the GPU. The eager denoising
loop was launch-bound (~17k small kernels per call, LOG 2026-11-03 (d)), so --cuda-graph replays it from one CUDA graph
per batch shape; the main process's torch threads are capped at the pod's CPU quota (24 host threads on a 4-CPU pod made
the CPU noise draw several times slower).

One JSON line per (episode, eval seed) via lp_eval.result_row (+ steps, instruction, wall_s), appended and flushed as
each episode ends; reruns skip records already in the file (resumable). Episodes that raise go to <out>.errors.jsonl and
are not recorded as failures (lp_eval then refuses the incomplete table).

Examples:
  python rollout.py instructions --out /mnt/proxy/lp_instructions.txt --tasks-out /mnt/proxy/lp_tasks.json
  python rollout.py run --run-dir /mnt/proxy/runs/dinov3-s0 --t5 /mnt/proxy/feats/t5 --vae-path Wan2.2_VAE.safetensors \
      --subset lp_subset.json --clean 0 --envs 16 --shard 0/4 --out /mnt/proxy/rollouts/dinov3-s0.0of4.jsonl
"""

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import lp_eval
import proxy_wam as PW

RENDER = 256                  # LIBERO_ENV_RESOLUTION
SIZE = 224
NUM_STEPS_WAIT = 30
REPLAN = 10
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700}
DUMMY = [0, 0, 0, 0, 0, 0, -1]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# perturbation suffixes of LIBERO-Plus task names: _table_1, _tb_10, _view_0_0_100_0_0_initstate_1, _light_1, _add_14,
# _language_1_view_..., _level1_sample4, _moved_level1_sample1 (libero_goal). A keyword counts only before a number, so
# "pick_up_the_black_bowl_from_table_center_..." keeps its "_table_center".
LP_SUFFIX = re.compile(r"_(?:(?:table|tb|view|light|add|language)_(?=[-\d])|(?:moved_)?level\d)")
ENC_DTYPE = {"dinov3": torch.float32, "c-radiov4": torch.float16, "siglip2": torch.float16}   # as the extraction


# ---------------------------------------------------------------------------
# Tasks and instructions
# ---------------------------------------------------------------------------

def base_name(name):
    """LIBERO-Plus task name -> its base LIBERO task name (the part before the perturbation suffix)."""
    m = LP_SUFFIX.search(name)
    return name[:m.start()] if m else name


def base_language(name):
    """LIBERO's instruction for a base task name (benchmark.grab_language_from_filename without the bddl lookup)."""
    if name[0].isupper():                                                   # libero_10: LIVING_ROOM_SCENE2_put_...
        return " ".join(name[name.find("SCENE") + (8 if "SCENE10" in name else 7):].split("_"))
    return " ".join(name.split("_"))


def lp_instruction(name, env_language):
    """Instruction the policy sees: the rewritten one for language perturbations, else the base task's (appended
    perturbation descriptions stripped, App. C.2)."""
    if "_language_" in name:
        return env_language.strip()
    base = base_language(base_name(name))
    assert env_language == base or env_language.startswith(base + " "), (name, env_language, base)
    return base


def request_noise(eval_seed, channels):
    """Starting noise of one inference call: as FastWAMJoint.infer_action with seed = eval seed, one generator each
    for video and action, reseeded every call."""
    gv = torch.Generator(device="cpu").manual_seed(int(eval_seed))
    ga = torch.Generator(device="cpu").manual_seed(int(eval_seed))
    return (torch.randn((1, channels, PW.LATENT_T) + PW.GRID, generator=gv)[0],
            torch.randn((1, PW.HORIZON, 7), generator=ga)[0])


def job_key(job):
    return (job["benchmark"], job["suite"], int(job["episode_id"]), int(job["eval_seed"]))


def make_jobs(subset=None, clean_per_task=0, clean_tasks=None, eval_seeds=lp_eval.EVAL_SEEDS, clean_seed=41):
    """Every LIBERO-Plus subset episode under every eval seed, plus `clean_per_task` trials of each clean task (one
    eval seed). Order: episode-major, so the eval seeds of an episode run close together."""
    jobs = []
    for e in (subset["episodes"] if subset else []):
        for s in eval_seeds:
            jobs.append({"benchmark": lp_eval.LP, "suite": e["suite"], "episode_id": int(e["id"]), "name": e["name"],
                         "category": e["category"], "eval_seed": int(s), "subset_id": subset["subset_id"]})
    for suite, names in sorted((clean_tasks or {}).items()):
        for ti, name in enumerate(names):
            for trial in range(clean_per_task):
                jobs.append({"benchmark": lp_eval.CLEAN, "suite": suite, "episode_id": ti * 1000 + trial,
                             "name": name, "trial": trial, "category": "", "eval_seed": int(clean_seed),
                             "subset_id": f"clean{clean_per_task}"})
    assert len({job_key(j) for j in jobs}) == len(jobs)
    return jobs


def order_jobs(jobs, order_seed):
    """Deterministic interleaving of the job list (LOG 2026-11-05: episodes of one perturbation category cost very
    different CPU per step, and in episode-major order all simulator workers hit the slow category together, so the
    GPU waited). A seeded shuffle of a copy: every pod computes the same full order from the same jobs and seed, so
    the `[i::n]` shards of it still partition the jobs. order_seed < 0 keeps the episode-major order (the pilot's).
    Results don't depend on job order (per-episode seeding, see the module docstring)."""
    jobs = list(jobs)
    if order_seed >= 0:
        random.Random(order_seed).shuffle(jobs)
    return jobs


def check_resume(jobs, out):
    """Refuse to resume a file that holds records outside this shard's job list. `done_keys` is per file, so a file
    written under another order seed, shard count or Python version would make this pod rerun jobs that other pods'
    files already hold (duplicates). Raises SystemExit; run_jobs is untouched."""
    stray = done_keys(out) - {job_key(j) for j in jobs}
    if stray:
        raise SystemExit(f"{out} holds {len(stray)} records that are not in this shard's {len(jobs)} jobs (example "
                         f"{sorted(stray)[0]}). Likely cause: a different --order-seed, --shard count, subset/--clean "
                         f"setting or Python version than the run that wrote the file. Use the same settings "
                         f"(--order-seed -1 reproduces the pre-shuffle pilot order) or a new --out.")


# ---------------------------------------------------------------------------
# Observation and action processing (Fast-WAM's eval path)
# ---------------------------------------------------------------------------

def center_crop_resize(image, width=SIZE, height=SIZE):
    """eval_libero_single.py _center_crop_resize (PIL bilinear), uint8 in and out."""
    pil = Image.fromarray(image)
    src_w, src_h = pil.size
    scale = max(width / src_w, height / src_h)
    resized = pil.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left, top = max((rw - width) // 2, 0), max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def quat2axisangle(quat):
    """robosuite transform_utils.quat2axisangle (x, y, z, w), as copied in Fast-WAM's libero_utils.py."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def process_obs(obs):
    """Raw sim observation -> (agentview 224 uint8, wrist 224 uint8, proprio (8,) float32)."""
    agent = center_crop_resize(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
    wrist = center_crop_resize(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    return agent, wrist, state


def to_exec(actions):
    """Denormalized chunk (..., 7) -> executed actions: gripper g -> 2g - 1, sign flipped, binarized (Fast-WAM)."""
    a = np.array(actions, dtype=np.float32, copy=True)
    a[..., -1] = -(a[..., -1] * 2 - 1)
    a[..., -1] = np.sign(a[..., -1])
    return a


# ---------------------------------------------------------------------------
# Policy (main process, GPU)
# ---------------------------------------------------------------------------

class Featurizer:
    """VAE latent and encoder grids of the current frame, as extract_fastwam.py computes them for training."""

    def __init__(self, sources, vae_path, device):
        from extract_fastwam import load_vae
        from extract_features import ENCODERS
        self.device = device
        self.vae = load_vae(vae_path, device, torch.bfloat16)
        self.encs = [(s, ENCODERS[s](device, ENC_DTYPE[s])) for s in sources]

    @torch.no_grad()
    def __call__(self, agent, wrist):
        """(B, 224, 224, 3) uint8 each -> [VAE (B, 48, 1, 14, 28)] + [source grid (B, 14, 28, D) ...], float32."""
        import fastwam_data as FD
        rgb = torch.from_numpy(np.concatenate([agent, wrist], axis=2)).to(self.device)        # (B, 224, 448, 3)
        x = (rgb.permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0)[:, :, None]               # (B, 3, 1, 224, 448)
        z = self.vae.model.encode(x.to(torch.bfloat16), self.vae.scale)
        assert tuple(z.shape[1:]) == (48, 1) + PW.GRID, z.shape
        out = [z.float().half().float()]
        for _, enc in self.encs:
            g = []
            for cam in (agent, wrist):
                _, tok = enc.encode(np.ascontiguousarray(cam))
                g.append(FD.grid_to_vae(tok.float(), (PW.GRID[0], PW.GRID[0])))
            out.append(torch.cat(g, dim=2).to(self.device).half().float())
        return out


class GraphedInfer:
    """PW.infer_actions (under bf16 autocast, as Policy.act) replayed from a CUDA graph, one per input shapes, captured
    on first use into one shared memory pool. Replay runs the captured kernels on the inputs copied into the graph's
    static tensors; profile_policy.py checks its actions against the eager path."""

    def __init__(self, model):
        self.model, self.graphs, self.pool = model, {}, torch.cuda.graph_pool_handle()

    def _run(self, current, ctx, cmask, prop, lat, a):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return PW.infer_actions(self.model, current, ctx, cmask, prop, noise=(lat, a))

    def _capture(self, args):
        static = [x.clone() for x in args]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                       # warm-up off the default stream, as torch.cuda.graph needs
            for _ in range(2):
                self._run(*static)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._run(*static)
        return g, static, out

    @torch.no_grad()
    def __call__(self, current, ctx, cmask, prop, noise):
        args = (current, ctx, cmask, prop, noise[0], noise[1])
        key = tuple((tuple(x.shape), x.dtype) for x in args)
        if key not in self.graphs:
            self.graphs[key] = self._capture(args)
        g, static, out = self.graphs[key]
        for st, x in zip(static, args):
            st.copy_(x)
        g.replay()
        return out.clone()


class Policy:
    """A trained proxy checkpoint (proxy_wam.py model.pt) with its training normalization."""

    def __init__(self, model, cfg, featurize, t5_dir, device, cuda_graph=False):
        self.model, self.cfg, self.featurize, self.t5_dir, self.device = model.eval(), cfg, featurize, t5_dir, device
        self.sources = cfg["sources"][1:]
        self.a_norm = PW.MinMax(cfg["action_norm"]["lo"], cfg["action_norm"]["hi"])
        self.s_norm = PW.MinMax(cfg["state_norm"]["lo"], cfg["state_norm"]["hi"])
        self.fstats = {s: (torch.tensor(m, device=device), torch.tensor(sd, device=device))
                       for s, (m, sd) in cfg["feature_stats"].items()}
        self.ctx, self.noise = {}, {}
        self.graphed = GraphedInfer(self.model) if cuda_graph else None

    @classmethod
    def load(cls, run_dir, t5_dir, vae_path, device, cuda_graph=False):
        ck = torch.load(Path(run_dir) / "model.pt", map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = PW.build_model(cfg["in_dim"], device, **cfg["model_kw"])
        model.load_state_dict(ck["model"])
        return cls(model.to(device), cfg, Featurizer(cfg["sources"][1:], vae_path, device), t5_dir, device, cuda_graph)

    def context(self, instruction):
        import t5_cache
        if instruction not in self.ctx:
            self.ctx[instruction] = t5_cache.load_context(self.t5_dir, instruction)[0]
        return self.ctx[instruction]

    def start_noise(self, eval_seed, channels):
        """request_noise(eval_seed, channels) on the device, drawn once (it is the same tensor every call)."""
        k = (int(eval_seed), int(channels))
        if k not in self.noise:
            self.noise[k] = tuple(t.to(self.device) for t in request_noise(eval_seed, channels))
        return self.noise[k]

    @torch.no_grad()
    def act(self, reqs):
        """reqs: [{agent, wrist, state, instruction, eval_seed}] -> (B, 32, 7) executed-form action chunks."""
        dev = self.device
        feats = self.featurize(np.stack([r["agent"] for r in reqs]), np.stack([r["wrist"] for r in reqs]))
        parts = [feats[0]]
        for src, f in zip(self.sources, feats[1:]):
            mu, sd = self.fstats[src]
            parts.append(((f - mu) / sd).permute(0, 3, 1, 2)[:, :, None])
        current = torch.cat(parts, 1)
        assert current.shape[1] == self.cfg["in_dim"], (current.shape, self.cfg["in_dim"])
        ctx = torch.stack([self.context(r["instruction"]) for r in reqs]).to(dev)
        cmask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=dev)
        prop = torch.from_numpy(self.s_norm(np.stack([r["state"] for r in reqs]))).to(dev)
        nz = [self.start_noise(r["eval_seed"], current.shape[1]) for r in reqs]
        noise = (torch.stack([n[0] for n in nz]), torch.stack([n[1] for n in nz]))
        if self.graphed is not None:
            a = self.graphed(current, ctx, cmask, prop, noise)
        else:
            use_amp = torch.device(dev).type == "cuda"
            with torch.autocast(device_type=torch.device(dev).type, dtype=torch.bfloat16, enabled=use_amp):
                a = PW.infer_actions(self.model, current, ctx, cmask, prop, noise=noise)
        return to_exec(self.a_norm.inverse(a.float().cpu().numpy()))


# ---------------------------------------------------------------------------
# Simulator side (worker processes)
# ---------------------------------------------------------------------------

def libero_env(job):
    """(env, init_state, instruction) for a job, from the installed LIBERO-Plus package (4976dc3)."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    if job["benchmark"] == lp_eval.LP:
        suite = benchmark.get_benchmark_dict()[job["suite"]]()
        i = int(job["episode_id"]) - 1
        task = suite.get_task(i)
        assert task.name == job["name"], (task.name, job["name"])
        # LIBERO-Plus calls torch.load() with the default, which torch >= 2.6 made weights_only=True; its init-state
        # pickles (numpy arrays, from the pinned repo) need the old behavior, as Fast-WAM's eval passes
        # weights_only=False for the same files. Scoped to this one call.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            init = suite.get_task_init_states(i)[0]
        finally:
            del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        language, instruction = task.language, lp_instruction(task.name, task.language)
    else:
        name = job["name"]
        bddl = Path(get_libero_path("bddl_files")) / job["suite"] / f"{name}.bddl"
        init = torch.load(Path(get_libero_path("init_states")) / job["suite"] / f"{name}.pruned_init",
                          weights_only=False)[int(job["trial"])]
        language = instruction = base_language(name)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RENDER, camera_widths=RENDER)
    env.seed(int(job["eval_seed"]))
    return env, init, instruction, language


def resolve(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_episode(job, env_factory, ask):
    """One closed-loop episode (Fast-WAM's run_single_episode without video saving). ask(request) -> chunk."""
    seed = int(job["eval_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    env, init, instruction, _ = env_factory(job)
    try:
        env.reset()
        obs = env.set_init_state(init)
        max_steps, t, done, pending = MAX_STEPS[job["suite"]], 0, False, []
        while t < max_steps + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, _, done, _ = env.step(DUMMY)
                t += 1
                continue
            if not pending:
                agent, wrist, state = process_obs(obs)
                chunk = ask({"agent": agent, "wrist": wrist, "state": state, "instruction": instruction,
                             "eval_seed": seed})
                pending = [list(map(float, a)) for a in chunk[:REPLAN]]
            obs, _, done, _ = env.step(pending.pop(0))
            if done:
                break
            t += 1
        final = process_obs(obs)[2]
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return {"success": bool(done), "steps": t, "instruction": instruction, "wall_s": round(time.time() - t0, 2),
            "final_state": [round(float(x), 6) for x in final], "worker_peak_rss_mb": peak_rss_mb()}


def peak_rss_mb():
    """This process's peak RSS (VmHWM) in MB, for sizing pod memory from measured use; None where /proc is missing."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024)
    except OSError:
        return None


def pod_cpus():
    """CPUs this container may use: the cgroup v2 quota (cpu.max), rounded down, else os.cpu_count()."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def cgroup_mem_gb():
    """(current, peak) memory of this container's cgroup in GB (cgroup v2), or None."""
    out = []
    for f in ("memory.current", "memory.peak"):
        try:
            out.append(round(int(Path("/sys/fs/cgroup", f).read_text()) / 1e9, 2))
        except (OSError, ValueError):
            out.append(None)
    return tuple(out)


def worker_main(conn, env_spec, mujoco_gl):
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = mujoco_gl
        os.environ["PYOPENGL_PLATFORM"] = mujoco_gl
    torch.set_num_threads(1)
    env_factory = resolve(env_spec)

    def ask(req):
        conn.send(("obs", req))
        return conn.recv()

    while True:
        conn.send(("ready", None))
        job = conn.recv()
        if job is None:
            return
        try:
            conn.send(("done", (job, run_episode(job, env_factory, ask))))
        except Exception:
            conn.send(("error", (job, traceback.format_exc())))


# ---------------------------------------------------------------------------
# Main process: job queue and batched inference
# ---------------------------------------------------------------------------

def record(job, res, config, train_seed, checkpoint):
    row = lp_eval.result_row(job["benchmark"], job["suite"], job["episode_id"], job["category"], config, train_seed,
                             job["eval_seed"], checkpoint, job["subset_id"], res["success"])
    return {**row, "steps": res["steps"], "instruction": res["instruction"], "wall_s": res["wall_s"],
            "final_state": res["final_state"], "worker_peak_rss_mb": res.get("worker_peak_rss_mb")}


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["benchmark"], r["suite"], int(r["episode_id"]), int(r["eval_seed"])))
    return keys


def run_jobs(jobs, act, n_envs, out, meta, env_spec="rollout:libero_env", mujoco_gl="egl", gather_s=0.05,
             log=print, log_every=60.0):
    """Run jobs on n_envs simulator processes; batch their inference requests through act(list) -> chunks.
    Appends one record per finished job to `out`; returns (records written, errors)."""
    todo = deque(j for j in jobs if job_key(j) not in done_keys(out))
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already in {out}, {len(todo)} to run on {n_envs} envs")
    if not todo:
        return 0, 0
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for _ in range(min(n_envs, len(todo))):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(b, env_spec, mujoco_gl), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)
    busy, waiting, live = set(), {}, set(conns)
    n_rec = n_err = 0
    stats = {"calls": 0, "items": 0, "act_s": 0.0}
    t_start = t_log = time.time()
    first_wait = None
    errf = Path(str(out) + ".errors.jsonl")
    with open(out, "a") as fo:
        while live:
            for c in wait(list(live), timeout=gather_s):
                try:
                    kind, payload = c.recv()
                except EOFError:
                    raise RuntimeError("a simulator worker died (see its traceback above)")
                if kind == "ready":
                    busy.discard(c)
                    if todo:
                        c.send(todo.popleft())
                        busy.add(c)
                    else:
                        c.send(None)
                        live.discard(c)
                elif kind == "obs":
                    waiting[c] = payload
                    first_wait = first_wait or time.time()
                elif kind == "done":
                    job, res = payload
                    fo.write(json.dumps(record(job, res, **meta)) + "\n")
                    fo.flush()
                    n_rec += 1
                elif kind == "error":
                    job, tb = payload
                    with open(errf, "a") as fe:
                        fe.write(json.dumps({"job": job, "traceback": tb}) + "\n")
                    log(f"ERROR in {job_key(job)}:\n{tb}")
                    n_err += 1
            # run a batch once every busy worker is waiting, or the oldest request has waited gather_s
            if waiting and (len(waiting) >= len(busy) or time.time() - first_wait >= gather_s):
                cs = list(waiting)
                t = time.time()
                chunks = act([waiting[c] for c in cs])
                stats["act_s"] += time.time() - t
                stats["calls"] += 1
                stats["items"] += len(cs)
                for c, ch in zip(cs, chunks):
                    c.send(ch)
                waiting.clear()
                first_wait = None
            if time.time() - t_log >= log_every:
                el = time.time() - t_start
                log(f"progress: {n_rec} done, {n_err} errors, {len(todo)} queued, {len(busy)} running; "
                    f"{3600 * n_rec / el:.0f} episodes/h; mean batch {stats['items'] / max(1, stats['calls']):.1f}; "
                    f"policy busy {stats['act_s'] / el:.0%}; cgroup memory now/peak GB {cgroup_mem_gb()}")
                t_log = time.time()
    for p in procs:
        p.join(timeout=30)
    el = time.time() - t_start
    log(f"finished: {n_rec} records, {n_err} errors in {el:.0f}s ({3600 * n_rec / max(el, 1e-9):.0f} episodes/h); "
        f"{stats['calls']} policy calls, mean batch {stats['items'] / max(1, stats['calls']):.1f}, "
        f"policy busy {stats['act_s'] / max(el, 1e-9):.0%}; cgroup memory now/peak GB {cgroup_mem_gb()}")
    return n_rec, n_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def lp_tasks():
    """All LIBERO-Plus tasks of the four suites from the installed package, with the policy's instruction."""
    from libero.libero import benchmark
    out = {}
    for s in SUITES:
        suite = benchmark.get_benchmark_dict()[s]()
        out[s] = [{"id": i + 1, "name": t.name, "language": t.language, "instruction": lp_instruction(t.name, t.language)}
                  for i, t in enumerate(suite.tasks)]
    return out


def hold_policy(reqs):
    """Smoke-test policy without a model: stay still, gripper open (executed form)."""
    return [np.tile(np.array(DUMMY, np.float32), (PW.HORIZON, 1)) for _ in reqs]


def first_frame(job, env_factory=libero_env):
    """The first frame the policy sees in an episode (after reset, set_init_state and the 30 wait steps), raw and
    through process_obs: for the sim-vs-training frame check (RUN.md step 8)."""
    random.seed(job["eval_seed"])
    np.random.seed(job["eval_seed"])
    env, init, instruction, _ = env_factory(job)
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(DUMMY)
    agent, wrist, state = process_obs(obs)
    raw = (obs["agentview_image"].copy(), obs["robot0_eye_in_hand_image"].copy())
    env.close()
    return {"agent": agent, "wrist": wrist, "state": state, "raw": raw, "instruction": instruction}


def smoke_jobs(classification, clean, per_cell=1, eval_seed=41):
    """The lowest-id episode(s) of every (suite, category) cell, plus trial 0 of the first clean task per suite."""
    eps = []
    for s in SUITES:
        by = {}
        for e in sorted(classification[s], key=lambda e: e["id"]):
            by.setdefault(e["category"], []).append(e)
        for c in sorted(by):
            eps += [{"suite": s, "id": e["id"], "name": e["name"], "category": c} for e in by[c][:per_cell]]
    sub = {"subset_id": "smoke", "episodes": eps}
    return make_jobs(sub, 1, {s: clean[s][:1] for s in SUITES}, eval_seeds=(eval_seed,), clean_seed=eval_seed)


def clean_tasks(tasks):
    """{suite: sorted base task names} from the LIBERO-Plus task names (10 per suite expected)."""
    out = {s: sorted({base_name(t["name"]) for t in v}) for s, v in tasks.items()}
    for s, names in out.items():
        assert len(names) == 10, (s, names)
    return out


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("instructions", help="list every LIBERO-Plus + clean instruction (for the T5 cache)")
    s1.add_argument("--out", required=True)
    s1.add_argument("--tasks-out", required=True, help="JSON: per suite, tasks with name, language, instruction, "
                                                        "and the clean base tasks")
    s1.add_argument("--classification", default=None, help="check names against task_classification.json")
    s1.add_argument("--meta", default=None, help="extract_fastwam meta dir: clean instructions must equal training's")
    s3 = sp.add_parser("smoke", help="simulator check without a model: one task per (suite, category) with a hold "
                                     "policy (timing, errors), a determinism rerun, and step-8 frames")
    s3.add_argument("--classification", required=True)
    s3.add_argument("--tasks", required=True)
    s3.add_argument("--out-dir", required=True)
    s3.add_argument("--envs", type=int, default=8)
    s3.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    s2 = sp.add_parser("run")
    s2.add_argument("--run-dir", required=True, help="proxy_wam.py output dir (model.pt)")
    s2.add_argument("--t5", required=True)
    s2.add_argument("--vae-path", required=True)
    s2.add_argument("--tasks", required=True, help="--tasks-out of the instructions step (clean base task names)")
    s2.add_argument("--subset", default=None, help="lp_eval subset file (LIBERO-Plus episodes)")
    s2.add_argument("--eval-seeds", default="41,42,43")
    s2.add_argument("--clean", type=int, default=0, help="clean LIBERO trials per task (one eval seed)")
    s2.add_argument("--clean-seed", type=int, default=41)
    s2.add_argument("--order-seed", type=int, default=0, help="seed of the job-order shuffle applied before the "
                    "shard split (same on every pod); -1 = episode-major order, as the pilot")
    s2.add_argument("--shard", default="0/1", help="i/n: this run takes jobs i, i+n, ...")
    s2.add_argument("--envs", type=int, default=8)
    s2.add_argument("--gather-s", type=float, default=0.05)
    s2.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s2.add_argument("--cuda-graph", action="store_true", help="replay the denoising loop from CUDA graphs")
    s2.add_argument("--torch-threads", type=int, default=pod_cpus(), help="main process (default: the pod's CPU quota)")
    s2.add_argument("--out", required=True, help="JSONL, one writer per file")
    args = p.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[rollout {time.time() - t0:7.0f}s] {m}", flush=True)

    if args.cmd == "instructions":
        tasks = lp_tasks()
        if args.classification:
            cls = json.loads(Path(args.classification).read_text())
            for s in SUITES:
                assert [(e["id"], e["name"]) for e in sorted(cls[s], key=lambda e: e["id"])] == \
                       [(t["id"], t["name"]) for t in tasks[s]], f"{s}: classification ids/names differ from the package"
        clean = clean_tasks(tasks)
        if args.meta:
            train = {str(t) for f in Path(args.meta).glob("*.npz") for t in np.load(f)["tasks"]}
            mine = {base_language(n) for v in clean.values() for n in v}
            assert train == mine, f"clean vs training instructions differ: {sorted(train ^ mine)}"
            log(f"clean instructions equal the {len(train)} training instructions")
        lines = sorted({t["instruction"] for v in tasks.values() for t in v} |
                       {base_language(n) for v in clean.values() for n in v})
        Path(args.out).write_text("\n".join(lines) + "\n")
        Path(args.tasks_out).write_text(json.dumps({"tasks": tasks, "clean": clean}, indent=1))
        n_lang = sum("_language_" in t["name"] for v in tasks.values() for t in v)
        log(f"{sum(map(len, tasks.values()))} LIBERO-Plus tasks ({n_lang} rewritten instructions), "
            f"{sum(map(len, clean.values()))} clean tasks -> {len(lines)} unique instructions in {args.out}")
        return

    if args.cmd == "smoke":
        import base64
        import io
        out = Path(args.out_dir)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        cls = json.loads(Path(args.classification).read_text())
        clean = json.loads(Path(args.tasks).read_text())["clean"]
        os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl
        # step-8 frames: the first frame the policy sees in the first clean task of each suite, plus libero_goal
        # "turn on the stove" (the task of training episode 432, checked by eye)
        fjobs = [j for j in make_jobs(None, 1, {s: clean[s] for s in SUITES})
                 if j["name"] in (clean[j["suite"]][0], "turn_on_the_stove")]
        for j in fjobs:
            t = time.time()
            f = first_frame(j)
            for k, img in [("agent224", f["agent"]), ("wrist224", f["wrist"]), ("agent256raw", f["raw"][0]),
                           ("wrist256raw", f["raw"][1])]:
                Image.fromarray(img).save(out / "frames" / f"{j['suite']}-{j['name'][:40]}-{k}.png")
                if k.endswith("224"):
                    buf = io.BytesIO()
                    Image.fromarray(img).resize((160, 160)).save(buf, "JPEG", quality=80)
                    print(f"THUMB {j['suite']}-{k} {base64.b64encode(buf.getvalue()).decode()}", flush=True)
            log(f"frame {j['suite']} {j['name']}: '{f['instruction']}', state {np.round(f['state'], 3).tolist()} "
                f"({time.time() - t:.1f}s incl. env build + 30 steps)")
        jobs = smoke_jobs(cls, clean)
        meta = {"config": "hold-policy-smoke", "train_seed": -1, "checkpoint": "none"}
        n1, e1 = run_jobs(jobs, hold_policy, args.envs, out / "smoke.jsonl", meta, mujoco_gl=args.mujoco_gl, log=log)
        n2, e2 = run_jobs(jobs[:6], hold_policy, 2, out / "smoke-repeat.jsonl", meta, mujoco_gl=args.mujoco_gl,
                          log=log)
        a = {job_key(r): r for r in map(json.loads, (out / "smoke.jsonl").read_text().splitlines())}
        b = {job_key(r): r for r in map(json.loads, (out / "smoke-repeat.jsonl").read_text().splitlines())}
        same = [k for k in b if k in a and (a[k]["final_state"], a[k]["steps"]) == (b[k]["final_state"], b[k]["steps"])]
        rows = list(a.values())
        steps = sum(r["steps"] + NUM_STEPS_WAIT for r in rows)
        log(f"smoke: {len(rows)} episodes, {e1 + e2} errors; repeat identical {len(same)}/{len(b)}; "
            f"{steps} sim steps in {sum(r['wall_s'] for r in rows):.0f} env-seconds = "
            f"{1000 * sum(r['wall_s'] for r in rows) / steps:.1f} ms per step per env (incl. env build)")
        for r in rows:
            log(f"  {r['benchmark']:<11} {r['suite']:<15} {r['category']:<22} id {r['episode_id']:<5} "
                f"steps {r['steps']:<4} {r['wall_s']:6.1f}s success {r['success']}  '{r['instruction'][:60]}'")
        if e1 + e2 or len(same) != len(b):
            raise SystemExit("smoke FAILED (errors or a non-deterministic repeat)")
        log("smoke ok")
        return

    i, n = map(int, args.shard.split("/"))
    sub = lp_eval.load_subset(args.subset) if args.subset else None
    clean = json.loads(Path(args.tasks).read_text())["clean"]
    jobs = order_jobs(make_jobs(sub, args.clean, clean, tuple(int(x) for x in args.eval_seeds.split(",")),
                                args.clean_seed), args.order_seed)[i::n]
    check_resume(jobs, args.out)
    torch.set_num_threads(args.torch_threads)
    policy = Policy.load(args.run_dir, args.t5, args.vae_path, args.device, cuda_graph=args.cuda_graph)
    cfg = policy.cfg
    log(f"checkpoint {args.run_dir}: config {cfg['config']} seed {cfg['seed']} commit {cfg['git_commit']}; "
        f"{len(jobs)} jobs in shard {i}/{n}, order seed {args.order_seed}; torch threads {args.torch_threads}, cuda graph {args.cuda_graph}")
    meta = {"config": cfg["config"], "train_seed": int(cfg["seed"]), "checkpoint": str(args.run_dir)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_rec, n_err = run_jobs(jobs, policy.act, args.envs, args.out, meta, log=log, gather_s=args.gather_s)
    if n_err:
        raise SystemExit(f"{n_err} episodes raised; see {args.out}.errors.jsonl")


if __name__ == "__main__":
    main()
