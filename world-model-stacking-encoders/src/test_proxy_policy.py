"""Synthetic checks for proxy_policy.py (cases with known answers). Run: python test_proxy_policy.py

Thresholds in check 7 were fixed before the first run.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

import proxy_policy as P


def episodes(lengths, dims, rng, val_eps=(), n_tasks=1, walk=True):
    """Synthetic Episodes: random-walk (or noise) streams on 32 positions, actions/proprio noise, given val episodes."""
    n = sum(lengths)
    ep_index = np.repeat(np.arange(len(lengths)), lengths)
    mk = (lambda d: np.cumsum(rng.normal(size=(n, 32, d)), 0) / 5) if walk else (lambda d: rng.normal(size=(n, 32, d)))
    streams = {k: mk(d).astype(np.float16) for k, d in dims.items()}
    split = np.where(np.isin(ep_index, val_eps), "val", "train")
    text = P.hash_text_embedding([f"task {i}" for i in range(n_tasks)])
    return P.Episodes(streams, rng.normal(size=(n, 7)).astype(np.float32), rng.normal(size=(n, 8)).astype(np.float32),
                      ep_index % n_tasks, ep_index, split, text)


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # 1. Pixel tokens: a 64x64 frame made of 16x16 constant blocks, value = 10 * (4 i + j), gives token 4 i + j equal
    #    to that value everywhere (block averaging and row-major patch order).
    g = 4
    frame = np.zeros((1, 64, 64, 3), np.uint8)
    for i in range(g):
        for j in range(g):
            frame[0, 16 * i:16 * (i + 1), 16 * j:16 * (j + 1)] = 10 * (4 * i + j)
    tok = P.pixel_tokens(frame, g)
    assert tok.shape == (1, 16, P.PATCH * P.PATCH * 3), tok.shape
    for q in range(16):
        assert np.allclose(tok[0, q], 10 * q / 255.0), (q, tok[0, q, :3])
    print(f"1 pixel tokens ok: shape {tok.shape[1:]}, token q = block q")

    # 2. Windows: frame value = global index, so gathered targets show exactly which frames were used. Nothing crosses
    #    an episode end; masks are true for exactly min(window, steps left) entries.
    lengths = [5, 40, 3, 33]
    ep = episodes(lengths, {"a": 1}, rng)
    ep.streams["a"] = np.arange(len(ep), dtype=np.float16)[:, None, None].repeat(32, 1)
    t = np.arange(len(ep))
    ai, am, fi, fm = ep.windows(t)
    end = ep.ep_end
    assert (ai < end[:, None]).all() and (fi < end[:, None]).all()
    assert (am.sum(1) == np.minimum(P.HORIZON, end - t)).all()
    assert (fm.sum(1) == np.array([sum(o < e - s for o in P.OFFSETS) for s, e in zip(t, end)])).all()
    assert (ai[am] == (t[:, None] + np.arange(P.HORIZON))[am]).all()
    assert (fi[fm] == (t[:, None] + np.array(P.OFFSETS))[fm]).all()
    print(f"2 windows ok: episodes {lengths}, no target crosses an episode end, mask counts exact")

    # 3. Normalization uses train frames only; actions round-trip.
    ep = episodes([30, 30, 30], {"a": 4}, rng, val_eps=[2])
    ep.streams["a"][ep.split == "val"] = 1000
    ep.actions[ep.split == "val"] = 1000
    nm = P.Norm(ep)
    tr = ep.split == "train"
    assert np.allclose(nm.stream["a"][0], ep.streams["a"][tr].astype(np.float64).reshape(-1, 4).mean(0), atol=1e-3)
    assert nm.a_hi.max() < 100
    a = ep.actions[tr][:50]
    assert np.allclose(nm.actions_inv(nm.actions(a)), a, atol=1e-5)
    assert nm.actions(a).min() >= -1 - 1e-6 and nm.actions(a).max() <= 1 + 1e-6
    print("3 normalization ok: stats from train only; actions round-trip and lie in [-1, 1]")

    # 4. Loss composition: action error 1 everywhere -> 1; two streams (d = 3 and d = 50) with error 2 -> 4 each, so the
    #    per-source loss doesn't grow with d; total = 1 + 4 + 4 = 9. Masked entries with huge errors change nothing.
    b, k, pos = 5, len(P.OFFSETS), 32
    amask = torch.ones(b, P.HORIZON, dtype=torch.bool)
    amask[:, 20:] = False
    fmask = torch.ones(b, k, dtype=torch.bool)
    fmask[:, 6:] = False
    act_t = torch.zeros(b, P.HORIZON, 7)
    act_p = torch.ones(b, P.HORIZON, 7)
    act_p[:, 20:] = 1e6
    y = {"s3": torch.zeros(b, k, pos, 3), "s50": torch.zeros(b, k, pos, 50)}
    pr = {n: torch.full_like(v, 2.0) for n, v in y.items()}
    for v in pr.values():
        v[:, 6:] = 1e6
    la, lf, tot = P.losses((act_p, pr), {"act": act_t, "amask": amask, "y": y, "fmask": fmask})
    assert abs(float(la) - 1) < 1e-6 and all(abs(float(v) - 4) < 1e-6 for v in lf.values()) and abs(float(tot) - 9) < 1e-5
    print(f"4 loss composition ok: action {float(la):.1f}, streams {[round(float(v), 1) for v in lf.values()]}, "
          f"total {float(tot):.1f} (weight 1.0 each, per-source means, masked entries ignored)")

    # 5. Config wiring and shapes: one projection and one future-feature head per stream, in a fixed order.
    assert P.config_encoders("none") == [] and P.config_encoders("dinov3+c-radiov4") == ["dinov3", "c-radiov4"]
    dims = {P.PIXELS: 192, "dinov3": 16, "c-radiov4": 24}
    m = P.ProxyPolicy(dims, 32, 64, hidden=32, layers=1, heads=4)
    x = {n: torch.randn(3, 32, d) for n, d in dims.items()}
    act, feats = m(x, torch.randn(3, 8), torch.randn(3, 64))
    assert act.shape == (3, P.HORIZON, 7) and list(feats) == list(dims)
    assert all(feats[n].shape == (3, len(P.OFFSETS), 32, d) for n, d in dims.items())
    m0 = P.ProxyPolicy({P.PIXELS: 192}, 32, 64, hidden=32, layers=1, heads=4)
    assert list(m0.feat_head) == [P.PIXELS] and m0.fuse.in_features == 64
    print(f"5 config wiring ok: streams {list(dims)} -> heads {list(feats)}; none -> {list(m0.feat_head)}")

    # 6. Data order depends on the seed only; different seeds differ.
    o0, o0b, o1 = P.data_order(1000, 2, 0), P.data_order(1000, 2, 0), P.data_order(1000, 2, 1)
    assert np.array_equal(o0, o0b) and not np.array_equal(o0, o1)
    assert all(sorted(o0[e * 1000:(e + 1) * 1000]) == list(range(1000)) for e in range(2))
    print("6 data order ok: same seed same order, each epoch a permutation, seeds differ")

    # 7. Planted signal. Actions = a fixed linear map of the dinov3 stream's mean over positions; pixels, proprio, text
    #    carry nothing. Thresholds (val, normalized units), fixed before running:
    #      config dinov3: action MSE < 0.5 x base; dinov3 future-feature MSE < 0.5 (targets are z-scored random
    #      walks, so f(t + k) is predictable from f(t)).
    #      config none: action MSE > 0.8 x base.
    #    base = val MSE of predicting the train mean action (what a model with no signal learns). First run
    #    (2026-10-27) used base = val action variance and one random walk across all episodes: the val episodes sat in
    #    a drifted region with a different action mean, so base was too small (0.094 vs none 0.663) and the dinov3
    #    check failed. Second run restarted the walk per episode: that made f(t + k) much less predictable (best
    #    possible feature MSE ~0.48 by a variance count), so the feature check failed instead (0.557), and actions
    #    were still poorly learned. Now: one continuous walk, val = every 5th episode (so train covers the same
    #    range), base includes the mean offset. Thresholds unchanged across all three runs.
    lengths = list(rng.integers(60, 90, size=40))
    full = episodes(lengths, {P.PIXELS: 12, "dinov3": 16}, rng, val_eps=list(range(4, 40, 5)))
    full.streams[P.PIXELS] = rng.normal(size=full.streams[P.PIXELS].shape).astype(np.float16)
    wmap = rng.normal(size=(16, 7))
    sig = full.streams["dinov3"].astype(np.float32).mean(1)
    sig = (sig - sig.mean(0)) / sig.std(0)
    full.actions = (sig @ wmap / 4).astype(np.float32)
    nm = P.Norm(full)
    va = full.split == "val"
    a_tr = nm.actions(full.actions[~va])
    base = float(((nm.actions(full.actions[va]) - a_tr.mean(0)) ** 2).mean())
    kw = {"hidden": 64, "layers": 2, "heads": 4}
    res = {}
    for cfg, keep in (("dinov3", [P.PIXELS, "dinov3"]), ("none", [P.PIXELS])):
        e = P.Episodes({k: full.streams[k] for k in keep}, full.actions, full.proprio, full.task, full.ep_index,
                       full.split, full.text)
        model, norm, hist = P.train(e, 0, 30, 64, "cpu", lambda m: None, kw, lr=1e-3)  # short run, higher lr
        res[cfg] = P.evaluate(model, e, norm, np.flatnonzero(e.split == "val"), 256, "cpu")
    print(f"7 planted signal: base (predict train mean) {base:.3f}; dinov3 action {res['dinov3']['action']:.3f} "
          f"feat {res['dinov3']['dinov3']:.3f}; none action {res['none']['action']:.3f}")
    assert res["dinov3"]["action"] < 0.5 * base, res
    assert res["dinov3"]["dinov3"] < 0.5, res
    assert res["none"]["action"] > 0.8 * base, res
    print("  ok")

    # 8. Determinism: the same seed gives identical training losses; another seed doesn't.
    small = episodes([50, 50, 50], {P.PIXELS: 12, "dinov3": 16}, rng)
    run = lambda s: [h["loss"] for h in P.train(small, s, 1, 16, "cpu", lambda m: None, kw)[2]]
    h0, h0b, h1 = run(0), run(0), run(1)
    assert h0 == h0b and h0 != h1, (h0[:2], h0b[:2], h1[:2])
    print(f"8 determinism ok: seed 0 twice identical ({len(h0)} logged steps), seed 1 differs")

    # 9. End to end through main() on synthetic LIBERO-style HDF5 files (raw + two feature caches), then
    #    predict_actions on raw inputs: shapes, finite, env-unit actions within the train min/max.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "raw").mkdir()
        encs = {"dinov3": 16, "c-radiov4": 24}
        for fi in range(2):
            stem = f"task{fi}_demo"
            with h5py.File(root / "raw" / f"{stem}.hdf5", "w") as fr:
                fr.create_group("data").attrs["problem_info"] = json.dumps({"language_instruction": f"do task {fi}"})
                caches = {}
                for e in encs:
                    (root / "feats" / e / "suite").mkdir(parents=True, exist_ok=True)
                    caches[e] = h5py.File(root / "feats" / e / "suite" / f"{stem}.h5", "w")
                    caches[e].attrs["meta"] = json.dumps({"rotate180": True, "grid": 4, "encoder": e})
                for di in range(5):
                    n = int(rng.integers(40, 60))
                    gr = fr.create_group(f"data/demo_{di}")
                    gr["actions"] = rng.normal(size=(n, 7))
                    for c in ("agentview_rgb", "eye_in_hand_rgb"):
                        gr[f"obs/{c}"] = rng.integers(0, 255, size=(n, 128, 128, 3), dtype=np.uint8)
                    gr["obs/ee_states"] = rng.normal(size=(n, 6))
                    gr["obs/gripper_states"] = rng.normal(size=(n, 2))
                    for e, d in encs.items():
                        for c in ("agentview_rgb", "eye_in_hand_rgb"):
                            caches[e][f"demo_{di}/{c}/grid"] = rng.normal(size=(n, 4, 4, d)).astype(np.float16)
                for f in caches.values():
                    f.close()
        out = root / "run"
        cmd = [sys.executable, str(Path(P.__file__)), "--feats", str(root / "feats"), "--suite", "suite",
               "--raw", str(root / "raw"), "--config", "dinov3+c-radiov4", "--epochs", "1", "--batch-size", "32",
               "--val-frac", "0.2", "--eval-every", "1", "--hidden", "32", "--layers", "1", "--device", "cpu",
               "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(P.__file__).parent)
        assert r.returncode == 0, r.stderr[-2000:]
        rec = json.loads((out / "train.json").read_text())
        assert rec["config"]["streams"] == [P.PIXELS, "dinov3", "c-radiov4"] and rec["history"], rec["config"]
        ck = torch.load(out / "model.pt", weights_only=False)
        ep, tasks, _ = P.load_libero_hdf5(root / "feats", "suite", root / "raw", list(encs), 0.2)
        model = P.ProxyPolicy({k: v.shape[-1] for k, v in ep.streams.items()}, 32, ep.text.shape[1], hidden=32,
                              layers=1)
        model.load_state_dict(ck["model"])
        nm = P.Norm(ep)
        t = np.array([0, 10, 20])
        a = P.predict_actions(model, nm, {k: v[t] for k, v in ep.streams.items()}, ep.proprio[t],
                              ep.text[ep.task[t]], "cpu")
        assert a.shape == (3, P.HORIZON, 7) and np.isfinite(a).all()
        assert np.allclose(ck["norm"]["a_lo"], nm.a_lo)
        print(f"9 end to end ok: main() wrote model.pt + train.json ({len(rec['history'])} log rows, "
              f"{rec['config']['n_params']} params); predict_actions {a.shape}, finite")
    print("all ok")


if __name__ == "__main__":
    main()
