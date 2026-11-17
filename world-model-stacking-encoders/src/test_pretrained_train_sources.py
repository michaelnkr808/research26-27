"""CPU checks for pretrained_train.py with encoder sources (stage 2 PLAN, experiments/2026-11-15-pretrained-start-pairings,
"What the implementation session must add", item 1). Run from src/: python test_pretrained_train_sources.py

The stage 1 test (test_pretrained_train.py) builds its fake cache with no sources. This one runs the same tiny model with
one fake source (config "dinov3") and with two (config "dinov3+c-radiov4"), fake feature widths 8 and 12, and checks:
  1. the widened load: every Wan tensor of the kept blocks is loaded unchanged; the three widened tensors
     (patch_embedding.weight, head.head.weight, head.head.bias) hold the fake Wan file's values on the 48 VAE channels
     and zeros on the extra channels; nothing else is partly loaded;
  2. the derived action expert is unchanged by the widening: its derived (copied or resized) tensors equal those of
     the 48-channel ("none") model bit for bit, and so does every video-expert tensor that is not widened. Not
     expected to match across configs at the same seed, and not checked: the fresh tensors (action encoder, action
     head, proprio encoder; the seeded build makes the wider patch embedding first, so it uses more random numbers
     before them), and the noise and timestep draws (draw_step's video-noise tensor has the extra channels in its
     shape, so every draw after it from the same generator, the timesteps and the action noise too, comes out
     different);
  3. one optimizer step's gradient with micro-batches 8 x 16 and 4 x 32 equals one batch of 128 with the same draws,
     relative L2 error <= 1e-5 over all parameters (the stage 1 tolerance, PR #39); the loss and each per-source loss
     part too; the extra input channels get a nonzero gradient (the check is not empty);
  4. resume: a run stopped after step 2 and resumed at the same micro-batch equals the straight 4-step run bit for bit
     (weights, losses); a resume with another --config is refused;
  5. rollout.Policy.load builds the widened model from model.pt (in_dim, widened shapes, every parameter equal, the
     sources in the saved config), and the per-source feature statistics it normalizes with equal the ones training
     used (Data.fstats, recomputed here from the cache's .stats.npy files, not read back from model.pt).

Needs FastWAM 7faa711 importable (pip install --no-deps -e FastWAM) and safetensors, as test_pretrained_train.py.
"""

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import pretrained_sizing as S
import pretrained_train as T
import proxy_wam as P
import rollout as R
from test_pretrained_sizing import SHARD1, SHARD2, fake_wan
from test_proxy_wam import fake_cache

TOL = 1e-5
SUITES = "suite_a,suite_b"
LENGTHS = (200, 150, 250)       # 600 windows per suite, 1,200 in all: 9 batches of 128 per epoch
DIMS = {"dinov3": 8, "c-radiov4": 12}
CONFIGS = ("dinov3", "dinov3+c-radiov4")
PATCH_K = 4                     # 1 x 2 x 2 patch positions per token (head rows are (positions, channels), c innermost)


def rel(a, b):
    a = torch.cat([x.reshape(-1).double() for x in a])
    b = torch.cat([x.reshape(-1).double() for x in b])
    return float((a - b).norm() / b.norm())


def argv(d, config, out, *extra, micro=8, gb=32):
    return ["--feats", str(d), "--t5", str(d / "t5"), "--suites", SUITES, "--config", config, "--wan-dir",
            str(d / "wan"), "--keep", "0-1", "--hidden", "48", "--action-hidden", "16", "--heads", "2", "--head-dim",
            "24", "--device", "cpu", "--global-batch", str(gb), "--micro-batch", str(micro), "--allow-any-micro-batch",
            "--buffer-episodes", "16", "--read-threads", "2", "--assemble-threads", "1", "--epochs", "1", "--out",
            str(d / out), *extra]


def run(a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = T.main(a)
    return rc, buf.getvalue()


def model_pt(d, out):
    return torch.load(d / out / "model.pt", map_location="cpu", weights_only=False)


def build(d, config):
    args = T.parser().parse_args(argv(d, config, "unused"))
    data = P.Data(d, SUITES.split(","), P.config_sources(config), d / "t5")
    data.context_dev = data.context
    model, rep = T.build_and_init(data, args, "cpu", lambda m: None)
    return data, model, rep


def check_widened_load(config, data, model, rep, wan):
    """1. Widened load: the VAE slice equals the fake Wan file, the extra slices are zero."""
    srcs = P.config_sources(config)
    extra = sum(DIMS[s] for s in srcs)
    assert data.sources == srcs and data.in_dim == 48 + extra, (data.sources, data.in_dim)
    assert list(data.channel_slices()) == [P.VAE] + srcs
    vs = model.video_expert.state_dict()
    kept = [k for k in wan if not k.startswith("blocks.") or int(k.split(".")[1]) < 2]
    assert set(kept) == set(vs), set(kept) ^ set(vs)
    for k in kept:
        if k in S.WIDENED:
            continue
        assert vs[k].shape == wan[k].shape and torch.equal(vs[k], wan[k]), k
    pw = vs["patch_embedding.weight"]
    assert pw.shape == (48, 48 + extra, 1, 2, 2), pw.shape
    assert torch.equal(pw[:, :48], wan["patch_embedding.weight"]) and bool((pw[:, 48:] == 0).all())
    for k in ("head.head.weight", "head.head.bias"):
        w = vs[k]
        assert w.shape[0] == PATCH_K * (48 + extra), (k, w.shape)
        wv = w.view((PATCH_K, 48 + extra) + tuple(w.shape[1:]))
        src = wan[k].view((PATCH_K, 48) + tuple(w.shape[1:]))
        assert torch.equal(wv[:, :48], src) and bool((wv[:, 48:] == 0).all()), k
    tags = {n: e["tag"] for n, e in rep["parameters"].items()}
    assert {n for n, t in tags.items() if t == "partly loaded"} == {"video_expert." + w for w in S.WIDENED}
    c = rep["counts_by_tag"]
    assert c == {"loaded": len(kept) - 3, "partly loaded": 3, "derived": 64, "fresh": 6}, c
    for n in S.WIDENED:
        e = rep["parameters"]["video_expert." + n]
        assert e["max_abs_diff"] == 0 and e["extra_slice_max_abs"] == 0, (n, e)
    print(f"  1 ok [{config}]: in_dim {data.in_dim}; {len(kept) - 3} tensors loaded unchanged; the 3 widened tensors "
          f"hold the Wan file on the 48 VAE channels and zeros on the {extra} extra channels")


def check_action_expert(config, model, rep, model0, rep0):
    """2. The derived action expert, and the video expert apart from the widened tensors, equal the 'none' model's."""
    derived = [n for n, e in rep["parameters"].items() if e["tag"] == "derived"]
    derived0 = [n for n, e in rep0["parameters"].items() if e["tag"] == "derived"]
    assert sorted(derived) == sorted(derived0) and len(derived) == 64
    assert all(n.startswith("action_expert.") for n in derived)
    sd, sd0 = model.state_dict(), model0.state_dict()
    bad = [n for n in derived if not torch.equal(sd[n], sd0[n])]
    assert not bad, bad
    assert rep["counts"]["derived"] == rep0["counts"]["derived"], (rep["counts"]["derived"], rep0["counts"]["derived"])
    assert all(e.get("source_shape") == rep0["parameters"][n].get("source_shape") for n, e in rep["parameters"].items()
               if e["tag"] == "derived")
    vid = [n for n in sd0 if n.startswith("video_expert.") and n[len("video_expert."):] not in S.WIDENED]
    assert all(torch.equal(sd[n], sd0[n]) for n in vid)
    fresh = [n for n, e in rep["parameters"].items() if e["tag"] == "new"]
    assert all(n.startswith(("action_expert.action_encoder.", "action_expert.head.", "proprio_encoder."))
               for n in fresh), fresh
    print(f"  2 ok [{config}]: the {len(derived)} derived action-expert tensors and the {len(vid)} non-widened "
          f"video-expert tensors equal the 'none' model's bit for bit")


def check_accumulation(config, data, model):
    """3. 8 x 16 and 4 x 32 against one batch of 128, same draws, within TOL; per-source loss parts too."""
    lengths = [n for _, _, n, _ in data.episodes]
    plan = P.make_plan(lengths, 1, 128, 16, 0)
    srcs = [P.VAE] + data.sources
    hb = data.assemble(plan[0], {k: {s: data.load_file(k, s) for s in srcs} for k in
                                 {data.episodes[ei][:2] for ei in np.unique(plan[0][:, 0])}})
    assert hb["latents"].shape == (128, P.LATENT_T, *P.GRID, data.in_dim)
    sl = data.channel_slices()
    for s in data.sources:                       # the extra channels hold features, not zeros or VAE copies
        x = hb["latents"][..., sl[s]].astype(np.float32)
        assert float(np.abs(x).max()) > 0 and float(x.std()) > 0, s
    draws = T.draw_step(model, 0, 0, 128, T.lat_shape_of(data))
    assert draws[0].shape == (128,) + T.lat_shape_of(data) and draws[0].shape[1] == data.in_dim
    grads, losses, parts = {}, {}, {}
    pe = dict(model.named_parameters())["video_expert.patch_embedding.weight"]
    for m in (128, 8, 4):
        model.zero_grad(set_to_none=True)
        losses[m], parts[m] = T.forward_backward(model, hb, data, draws, m, "cpu")
        grads[m] = [p.grad.clone() for p in model.parameters()]
        if m == 128:
            g_extra = float(pe.grad[:, 48:].abs().max())
    assert set(parts[128]) == {"vae", "action", *data.sources}, parts[128]
    assert g_extra > 0, "no gradient on the extra input channels"
    e8, e4 = rel(grads[8], grads[128]), rel(grads[4], grads[128])
    assert e8 <= TOL and e4 <= TOL, (e8, e4)
    for m in (8, 4):
        assert abs(losses[m] - losses[128]) <= TOL * abs(losses[128]), (m, losses)
        for k in parts[128]:
            assert abs(parts[m][k] - parts[128][k]) <= TOL * abs(parts[128][k]), (m, k, parts)
    model.zero_grad(set_to_none=True)
    print(f"  3 ok [{config}]: accumulated gradient equals the batch of 128 (rel. L2 error 8x16 {e8:.1e}, 4x32 "
          f"{e4:.1e}; tolerance {TOL:.0e}); loss parts {sorted(parts[128])} agree; extra-channel gradient nonzero")


def check_resume(d, config, tag):
    """4. Straight 4 steps vs stop at 2 + resume at the same micro-batch: bit for bit. Another --config is refused."""
    a, b = f"{tag}-A", f"{tag}-B"
    rc, logA = run(argv(d, config, a, "--max-steps", "4", "--ckpt-every", "2", "--first-batch-report"))
    assert rc == 0, logA[-2000:]
    rc, _ = run(argv(d, config, b, "--max-steps", "4", "--ckpt-every", "2", "--stop-after", "2"))
    assert rc == 3 and (d / b / "ckpt.pt").exists() and not (d / b / "model.pt").exists()
    ck = torch.load(d / b / "ckpt.pt", weights_only=False)
    assert ck["step"] == 2 and ck["run_identity"]["config"] == config and ck["run_identity"]["in_dim"] == \
        48 + sum(DIMS[s] for s in P.config_sources(config)), ck["run_identity"]
    rc, logB = run(argv(d, config, b, "--max-steps", "4", "--ckpt-every", "2"))
    assert rc == 0 and "resuming" in logB and "MICRO-BATCH CHANGED" not in logB, logB[-2000:]
    A, B = model_pt(d, a), model_pt(d, b)
    assert set(A["model"]) == set(B["model"]) and all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"])
    hA = json.loads((d / a / "train.json").read_text())["history"]
    hB = json.loads((d / b / "train.json").read_text())["history"]
    assert [h["step"] for h in hB] == [0, 1, 2, 3] and [h["loss"] for h in hA] == [h["loss"] for h in hB]
    assert all(f"loss/{s}" in hA[0] for s in P.config_sources(config)), hA[0]
    fb = json.loads((d / a / "first-batch.json").read_text())
    assert all(f"channels/{s}" in fb for s in P.config_sources(config))
    other = "dinov3" if config != "dinov3" else "dinov3+c-radiov4"
    try:
        run(argv(d, other, b, "--max-steps", "4", "--ckpt-every", "2"))
        raise AssertionError("a resume with another --config was accepted")
    except SystemExit as e:
        assert "different run" in str(e) and "config" in str(e) and "in_dim" in str(e), e
    print(f"  4 ok [{config}]: resume at the same micro-batch reproduces the straight run bit for bit; a resume with "
          f"--config {other} is refused")
    return A


def check_policy_load(d, config, tag, A):
    """5. rollout.Policy.load builds the widened model from model.pt."""
    srcs = P.config_sources(config)
    in_dim = 48 + sum(DIMS[s] for s in srcs)
    cfg = A["config"]
    assert cfg["in_dim"] == in_dim and cfg["sources"] == [P.VAE] + srcs and cfg["config"] == config
    assert set(cfg["feature_stats"]) == set(srcs)
    assert all(len(cfg["feature_stats"][s][0]) == len(cfg["feature_stats"][s][1]) == DIMS[s] for s in srcs)
    # the statistics training used: Data.fstats of the same cache and config (float32 mean, sd per channel)
    train_stats = P.Data(d, SUITES.split(","), srcs, None).fstats
    assert set(train_stats) == set(srcs)
    orig = R.Featurizer
    seen = []
    R.Featurizer = lambda sources, *a, **k: seen.append(list(sources))   # the encoders are not needed to build it
    try:
        pol = R.Policy.load(d / f"{tag}-A", d / "t5", None, "cpu")
    finally:
        R.Featurizer = orig
    assert seen == [srcs], seen
    assert pol.sources == srcs and set(pol.fstats) == set(srcs)
    for s in srcs:
        for i, what in enumerate(("mean", "sd")):
            ref = torch.from_numpy(train_stats[s][i])
            assert ref.dtype == torch.float32 and bool(ref.abs().max() > 0), (s, what)
            got = pol.fstats[s][i]
            assert got.dtype == ref.dtype and torch.equal(got, ref), \
                (s, what, float((got.float() - ref).abs().max()))
    sd = pol.model.state_dict()
    assert set(sd) == set(A["model"]) and all(torch.equal(sd[k], A["model"][k]) for k in sd)
    assert tuple(sd["video_expert.patch_embedding.weight"].shape) == (48, in_dim, 1, 2, 2)
    assert sd["video_expert.head.head.weight"].shape[0] == PATCH_K * in_dim
    assert sd["video_expert.head.head.bias"].shape[0] == PATCH_K * in_dim
    assert len(pol.model.video_expert.blocks) == len(pol.model.action_expert.blocks) == 2
    print(f"  5 ok [{config}]: rollout.Policy.load builds the widened model (in_dim {in_dim}) from model.pt, every "
          f"parameter equal; sources {srcs}; its feature statistics equal the ones training used (Data.fstats)")


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_cache(d, SUITES.split(","), list(DIMS), np.random.default_rng(0), n_eps=3, lengths=LENGTHS, dims=DIMS)
        fake_wan(d / "wan", n_blocks=4)
        wan = {**load_file(str(d / "wan" / SHARD1)), **load_file(str(d / "wan" / SHARD2))}
        _, model0, rep0 = build(d, "none")
        assert rep0["counts_by_tag"]["partly loaded"] == 0
        for i, config in enumerate(CONFIGS):
            print(f"config {config} ({len(P.config_sources(config))} source(s))")
            data, model, rep = build(d, config)
            check_widened_load(config, data, model, rep, wan)
            check_action_expert(config, model, rep, model0, rep0)
            check_accumulation(config, data, model)
            del model
            A = check_resume(d, config, f"c{i}")
            check_policy_load(d, config, f"c{i}", A)
    print("all checks passed")


def test_all():
    main()


if __name__ == "__main__":
    main()
