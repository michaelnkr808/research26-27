"""The 'none' invariance test (stage 2 PLAN, experiments/2026-11-15-pretrained-start-pairings, "What the implementation
session must add", item 6). Run from src/: python test_none_invariance.py

Stage 2 adds encoder sources to pretrained_train.py and the modules it uses. The configuration 'none' is stage 1's
baseline and is reused as stage 2's no-encoder arm, so its training path must stay bit-identical to the code the stage 1
runs used: commit STAGE1_COMMIT (the $C of the stage 1 runs). This test takes that commit's src/ tree with
"git archive" into a temporary folder and runs the same driver (this file, --driver) twice in fresh processes on the
same fake data (a tiny model, a fake Wan checkpoint, a fake cache with no sources): once importing the stage 1 tree,
once importing the current src/. Each process saves what it computed; the two are compared bit for bit (every float
tensor and array byte for byte, every Python float by its exact value; only wall times, the output folder and the
code's git hash are left out). What is compared, all for config 'none':
  draws    draw_step for several (seed, step) pairs, and the training plan (make_plan);
  data     the first host batch (Data.assemble) and its device tensors (to_device);
  build    the seeded build (pretrained_sizing.build), and the model after build_and_init (Wan load + action-expert
           derivation) with its load report;
  loss     training_loss with its own draws (old path) and with given draws; forward_backward on a batch of 128 at
           micro-batches 8 and 4 (loss, loss parts, every gradient); predict_zero_loss; one optimizer_step (loss,
           parts, gradient norm, weights and AdamW state after it);
  run      pretrained_train.main end to end: a run stopped after step 2 (ckpt.pt: weights, AdamW, step, RNG states,
           run identity, micro-batch history), then resumed to step 4 (model.pt, history losses, train.json), its
           config.json, load.json and first-batch.json, and the overfit mode (overfit.json).
Both processes use the same interpreter, the same FastWAM and the same thread count, so any difference comes from the
code. A comparator self-check (a one-ulp change in one weight is reported) keeps the comparison from passing vacuously.

Needs FastWAM 7faa711 importable (pip install --no-deps -e FastWAM), safetensors, and git with the commit available.
"""

import io
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

STAGE1_COMMIT = "2b9d39ce0551cd91df68a52389ee2d03cdc7cee5"
HERE = Path(__file__).resolve().parent
SUITES = "suite_a,suite_b"
LENGTHS = (200, 150, 250)
THREADS = 2
GIT_HASH = "none-invariance-test"     # SRC_GIT_COMMIT for both processes, so config["git_commit"] is equal
TIME_KEYS = {"s_step", "data_wait_s", "seconds_this_process", "seconds"}
MODULES = ("pretrained_train.py", "proxy_wam.py", "pretrained_sizing.py", "action_expert_init.py", "t5_cache.py",
           "diagnostics.py")


# ---------------------------------------------------------------------------
# Driver: runs inside a fresh process with the code under test first on sys.path
# ---------------------------------------------------------------------------

def driver(src, d, out):
    sys.path.insert(0, str(src))
    import contextlib

    import numpy as np
    import torch

    import pretrained_sizing as S
    import pretrained_train as T
    import proxy_wam as P
    torch.set_num_threads(THREADS)
    for m in (T, P, S):
        assert Path(m.__file__).resolve().parent == Path(src).resolve(), (m.__name__, m.__file__)
    d, out = Path(d), Path(out)
    res = {"files": {m: str(Path(sys.modules[m[:-3]].__file__).resolve()) for m in MODULES if m[:-3] in sys.modules}}

    def argv(name, *extra, micro=8, gb=32):
        return ["--feats", str(d), "--t5", str(d / "t5"), "--suites", SUITES, "--config", "none", "--wan-dir",
                str(d / "wan"), "--keep", "0-1", "--hidden", "48", "--action-hidden", "16", "--heads", "2",
                "--head-dim", "24", "--device", "cpu", "--global-batch", str(gb), "--micro-batch", str(micro),
                "--allow-any-micro-batch", "--buffer-episodes", "16", "--read-threads", "2", "--assemble-threads", "1",
                "--epochs", "1", "--step-watchdog-s", "0", "--out", str(out / name), *extra]

    def run(a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = T.main(a)
        return rc

    # data and plan
    data = P.Data(d, SUITES.split(","), P.config_sources("none"), d / "t5")
    data.context_dev = data.context
    assert data.sources == [] and data.in_dim == 48
    lengths = [n for _, _, n, _ in data.episodes]
    plan = P.make_plan(lengths, 2, 128, 16, 0)
    res["plan"] = plan
    res["plan_32"] = P.make_plan(lengths, 1, 32, 16, 0)
    res["channel_slices"] = {k: (v.start, v.stop) for k, v in data.channel_slices().items()}
    res["norms"] = {"action": data.a_norm.state(), "state": data.s_norm.state(), "tasks": data.tasks}
    hb = data.assemble(plan[0], {k: {P.VAE: data.load_file(k, P.VAE)} for k in
                                 {data.episodes[ei][:2] for ei in np.unique(plan[0][:, 0])}})
    res["host_batch"] = hb
    bt8 = P.to_device(T.host_slice(hb, slice(0, 8)), data, "cpu")
    res["device_batch_8"] = {k: v for k, v in bt8.items() if isinstance(v, torch.Tensor)}

    # build
    args = T.parser().parse_args(argv("unused"))
    res["seeded_build"] = S.build(data.in_dim, "cpu", 2, T.model_kw_of(args, S.parse_keep(args.keep))["_build"],
                                  args.seed).state_dict()
    model, rep = T.build_and_init(data, args, "cpu", lambda m: None)
    res["built"] = {k: v.clone() for k, v in model.state_dict().items()}
    res["load_report"] = rep

    # draws
    ls = T.lat_shape_of(data)
    res["lat_shape"] = ls
    res["draws"] = {f"{s}/{t}/{n}": T.draw_step(model, s, t, n, ls)
                    for s, t, n in ((0, 0, 128), (0, 1, 128), (0, 700, 128), (0, 2169, 128), (3, 5, 32), (1, 0, 8))}

    # losses
    sl = data.channel_slices()
    with torch.no_grad():
        torch.manual_seed(3)
        l_old, p_old = P.training_loss(model, bt8, sl)
        draws8 = T.draw_step(model, 0, 0, 8, ls)
        l_new, p_new = P.training_loss(model, bt8, sl, draws=draws8)
    res["training_loss"] = {"own_draws": (l_old, p_old), "given_draws": (l_new, p_new)}
    draws = T.draw_step(model, 0, 0, 128, ls)
    for m in (8, 4):
        model.zero_grad(set_to_none=True)
        loss, parts = T.forward_backward(model, hb, data, draws, m, "cpu")
        res[f"forward_backward_{m}"] = {"loss": loss, "parts": parts,
                                        "grads": {n: p.grad.clone() for n, p in model.named_parameters()}}
    model.zero_grad(set_to_none=True)
    res["predict_zero"] = T.predict_zero_loss(model, hb, data, draws, 8, "cpu")
    opt = S.make_opt(model)
    loss, parts, gn = T.optimizer_step(model, opt, hb, data, draws, 8, "cpu", P.lr_at(0, len(plan)))
    res["optimizer_step"] = {"loss": loss, "parts": parts, "grad_norm": gn, "lr": P.lr_at(0, len(plan)),
                             "lr_sched": [P.lr_at(s, len(plan)) for s in (0, 1, 5, 9, 17)],
                             "model": model.state_dict(), "opt": opt.state_dict()}
    del model, opt

    # end-to-end runs
    rc = run(argv("R", "--max-steps", "4", "--ckpt-every", "2", "--stop-after", "2", "--first-batch-report"))
    assert rc == 3, rc
    res["ckpt_step2"] = torch.load(out / "R" / "ckpt.pt", map_location="cpu", weights_only=False)
    rc = run(argv("R", "--max-steps", "4", "--ckpt-every", "2", "--first-batch-report"))
    assert rc == 0, rc
    res["model_pt"] = torch.load(out / "R" / "model.pt", map_location="cpu", weights_only=False)
    res["ckpt_final"] = torch.load(out / "R" / "ckpt.pt", map_location="cpu", weights_only=False)
    for f in ("config.json", "config-resume-step2.json", "load.json", "first-batch.json", "train.json"):
        res["R/" + f] = json.loads((out / "R" / f).read_text())
    rc = run(argv("O", "--overfit-steps", "5", "--overfit-lr", "1e-3"))
    assert rc == 0, rc
    res["O/overfit.json"] = json.loads((out / "O" / "overfit.json").read_text())
    torch.save(res, out / "results.pt")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(a, b, path="", skip=frozenset()):
    """List of differences between a and b, bit for bit. Dict keys in skip are left out at any depth."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return [f"{path}: tensor vs {type(b).__name__ if isinstance(a, torch.Tensor) else type(a).__name__}"]
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}"]
        ab = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bb = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return [] if torch.equal(ab, bb) else [f"{path}: tensor bytes differ"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return [f"{path}: array vs other"]
        if a.dtype != b.dtype or a.shape != b.shape or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            return [f"{path}: array differs ({a.dtype} {a.shape} vs {b.dtype} {b.shape})"]
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - skip, set(b) - skip
        out = [f"{path}: keys only in stage 1 {sorted(map(str, ka - kb))}, only now {sorted(map(str, kb - ka))}"] \
            if ka != kb else []
        for k in sorted(ka & kb, key=str):
            out += compare(a[k], b[k], f"{path}/{k}", skip)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare(x, y, f"{path}[{i}]", skip)
        return out
    if isinstance(a, float) and isinstance(b, float):
        return [] if struct.pack("<d", a) == struct.pack("<d", b) else [f"{path}: {a!r} vs {b!r}"]
    if type(a) is not type(b) or a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []


def stage1_tree(dest):
    """Extract STAGE1_COMMIT's src/ (the folder this file is in) into dest with git archive."""
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    top, prefix = git("rev-parse", "--show-toplevel"), git("rev-parse", "--show-prefix")
    # run from the top: in a subdirectory, git archive only takes that subdirectory's part of the tree
    r = subprocess.run(["git", "archive", "--format=tar", f"{STAGE1_COMMIT}:{prefix}"], cwd=top, capture_output=True)
    if r.returncode != 0:
        raise AssertionError(f"git archive of the stage 1 commit {STAGE1_COMMIT} failed: {r.stderr.decode()[-500:]}")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
        t.extractall(dest, filter="data")
    missing = [m for m in MODULES if not (Path(dest) / m).exists()]
    assert not missing, f"stage 1 tree lacks {missing}"


def run_driver(src, d, out):
    env = {**os.environ, "SRC_GIT_COMMIT": GIT_HASH, "OMP_NUM_THREADS": str(THREADS), "PYTHONHASHSEED": "0"}
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--driver", str(src), str(d), str(out)],
                       cwd=src, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"driver on {src} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"


def main():
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch

    from test_pretrained_sizing import fake_wan
    from test_proxy_wam import fake_cache
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        d, old_src = root / "data", root / "stage1-src"
        fake_cache(d, SUITES.split(","), [], np.random.default_rng(0), n_eps=3, lengths=LENGTHS)
        fake_wan(d / "wan", n_blocks=4)
        old_src.mkdir()
        stage1_tree(old_src)
        changed = [m for m in MODULES if (old_src / m).read_bytes() != (HERE / m).read_bytes()]
        print(f"stage 1 code: commit {STAGE1_COMMIT[:12]}; modules changed since then: {changed or 'none'}")
        outs = {}
        for name, src in (("stage1", old_src), ("current", HERE)):
            outs[name] = root / f"out-{name}"
            outs[name].mkdir()
            run_driver(src, d, outs[name])
        A = torch.load(outs["stage1"] / "results.pt", map_location="cpu", weights_only=False)
        B = torch.load(outs["current"] / "results.pt", map_location="cpu", weights_only=False)
        assert all(Path(p).parent == old_src.resolve() for p in A.pop("files").values())
        assert all(Path(p).parent == HERE for p in B.pop("files").values())

        # self-check: the comparator sees a one-ulp change in one weight and a changed loss value
        k = next(iter(A["model_pt"]["model"]))
        w = A["model_pt"]["model"][k].clone()
        w.view(-1)[0] = torch.nextafter(w.view(-1)[0], torch.tensor(float("inf")))
        assert compare(A["model_pt"]["model"][k], w), "comparator missed a one-ulp change"
        assert compare(1.0, 1.0 + 2 ** -52) and compare({"a": -0.0}, {"a": 0.0}) and not compare([1.5], [1.5])

        skip = frozenset(TIME_KEYS | {"out", "git_commit"})
        diffs = {}
        for key in sorted(A.keys() | B.keys()):
            if key not in A or key not in B:
                diffs[key] = [f"only in {'stage 1' if key in A else 'current'}"]
                continue
            dk = compare(A[key], B[key], key, skip)
            if dk:
                diffs[key] = dk
        if diffs:
            lines = [f"{k}: {len(v)} difference(s), e.g. {v[:3]}" for k, v in diffs.items()]
            raise AssertionError("the 'none' path differs from the stage 1 code:\n" + "\n".join(lines))
        # the run produced what it should (so equality is not between two empty results)
        assert A["ckpt_step2"]["step"] == 2 and len(A["R/train.json"]["history"]) == 4
        assert A["model_pt"]["config"]["git_commit"] == GIT_HASH and A["model_pt"]["config"]["in_dim"] == 48
        n_t = sum(isinstance(v, torch.Tensor) for v in A["model_pt"]["model"].values())
        print(f"ok: config 'none' is bit-identical to the stage 1 code in {len(A)} compared items: draws, plan, "
              f"batches, seeded build, build_and_init ({n_t} tensors) and load report, losses and gradients at "
              f"micro-batches 8 and 4, predict-zero, one optimizer step, ckpt.pt at step 2 and at the end, model.pt, "
              f"history, config/load/first-batch/train JSON, overfit mode")
    print("all checks passed")


def test_all():
    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--driver":
        driver(*sys.argv[2:5])
    else:
        main()
