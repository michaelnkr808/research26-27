"""Tests for check_cache_fresh.py with fakes (no real encoders, data or cluster).

A fake encoder (2x2-pixel average pool to a 14 x 14 grid, then a fixed 3 -> 8 linear map: its "weights") fills a fake
cache through the real extract_fastwam.encoder_episode. The check must pass on that cache, fail when the weights change,
pass small numeric noise, and report missing files, wrong shapes and provenance mismatches.

Run from src/: python test_check_cache_fresh.py   (or pytest)
"""

import argparse
import json
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import check_cache_fresh as CF
import extract_fastwam as X
import fastwam_data as FD

D = 8
REV = "0" * 40


def fake_encoder(name, weights, rev=REV, loaded=REV, hf_id=None):
    w = torch.from_numpy(weights)

    def encode(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0          # (N, 3, 224, 224)
        g = F.adaptive_avg_pool2d(x, 14).permute(0, 2, 3, 1)                       # (N, 14, 14, 3)
        tok = g @ w                                                                # (N, 14, 14, D)
        return tok.mean(dim=(1, 2)), tok

    return types.SimpleNamespace(name=name, hf_id=hf_id or f"fake/{name}", resolution=224, revision=rev,
                                 loaded_commit=loaded, encode=encode)


def episodes(suite, n_eps=5):
    return [{"episode_index": 100 * (CF.SUITES.index(suite) + 1) + i, "length": 3 + i, "task": "t"} for i in range(n_eps)]


def decode(root, ep):
    g = np.random.default_rng(ep["episode_index"])
    return {c: torch.from_numpy(g.random((ep["length"], 3, 224, 224), dtype=np.float32)) for c in FD.CAMS}


def make_cache(feats, weights, sources=("dinov3", "siglip2"), skip=(), dtype_override=None):
    for s in sources:
        enc = fake_encoder(s, weights[s])
        for suite in CF.SUITES:
            d = Path(feats) / s / suite
            d.mkdir(parents=True, exist_ok=True)
            (d / "_meta.shard0of1.json").write_text(json.dumps({"model": enc.hf_id, "dtype": dtype_override or CF.EXTRACT[s]}))
            for ep in episodes(suite):
                if (s, ep["episode_index"]) in skip:
                    continue
                np.save(d / f"ep{ep['episode_index']:06d}.npy", X.encoder_episode(enc, decode(None, ep), CF.BATCH))


def weights(seed=0, sources=("dinov3", "siglip2")):
    g = np.random.default_rng(seed)
    return {s: g.standard_normal((3, D)).astype(np.float32) for s in sources}


def args_for(feats, sources="dinov3,siglip2", **kw):
    a = dict(data_root="/nonexistent", feats=str(feats), sources=sources, suites=",".join(CF.SUITES),
             per_suite=2, seed=0, batch=CF.BATCH, cos_min=CF.COS_MIN, rel_max=CF.REL_MAX, rel_l2_max=CF.REL_L2_MAX, device="cpu",
             out_json=None)
    a.update(kw)
    return argparse.Namespace(**a)


def check(feats, enc_weights, encode_episode=None, **kw):
    load = lambda s, device, dtype: fake_encoder(s, enc_weights[s], **kw)
    return CF.run(args_for(feats), load_encoder=load, read_suite=lambda root: episodes(Path(root).name.split("_no_")[0]),
                  decode=decode, encode_episode=encode_episode, log=lambda *a: None)


def test_same_weights_pass():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w)
        s, ok = check(t, w)
        assert ok, s["problems"]
        assert len(s["episodes"]) == 2 * 4 * 2
        assert all(r["frac_bit_equal"] == 1.0 and r["cos_min"] > 0.999999 for r in s["episodes"])


def test_changed_weights_fail():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w)
        w2 = dict(w)
        w2["siglip2"] = w["siglip2"] + 0.3 * np.random.default_rng(9).standard_normal((3, D)).astype(np.float32)
        s, ok = check(t, w2)
        assert not ok
        assert s["per_source"]["siglip2"]["failed"] == 8 and s["per_source"]["dinov3"]["failed"] == 0


def test_small_numeric_noise_passes():
    """Noise of 1e-3 of the max abs value (the size of fp16 and GPU-kernel differences) stays inside the tolerance."""
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w)

        def noisy(enc, cams, batch):
            a = X.encoder_episode(enc, cams, batch).astype(np.float32)
            g = np.random.default_rng(1)
            return (a + 1e-3 * np.abs(a).max() * g.uniform(-1, 1, a.shape)).astype(np.float16)

        s, ok = check(t, w, encode_episode=noisy)
        assert ok, (s["problems"], s["per_source"])
        assert all(r["frac_bit_equal"] < 1.0 for r in s["episodes"])


def test_choice_fixed_by_seed_and_only_cached():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        skip = {("siglip2", 100), ("siglip2", 101), ("dinov3", 102)}        # libero_spatial: only 103, 104 complete
        make_cache(t, w, skip=skip)
        a, ok_a = check(t, w)
        b, _ = check(t, w)
        assert ok_a, a["problems"]
        pick = lambda s: [(r["source"], r["suite"], r["episode"]) for r in s["episodes"]]
        assert pick(a) == pick(b)
        spatial = sorted({r["episode"] for r in a["episodes"] if r["suite"] == "libero_spatial"})
        assert spatial == [103, 104], spatial
        for suite in CF.SUITES:      # the same episodes for every source
            eps = {src: [r["episode"] for r in a["episodes"] if r["suite"] == suite and r["source"] == src]
                   for src in ("dinov3", "siglip2")}
            assert eps["dinov3"] == eps["siglip2"], eps


def test_too_few_cached_is_a_problem():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w, skip={("dinov3", i) for i in range(400, 404)})       # libero_10: one episode left
        s, ok = check(t, w)
        assert not ok and any("libero_10: only 1" in p for p in s["problems"]), s["problems"]


def test_shape_mismatch_is_a_problem():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w)
        for p in (Path(t) / "dinov3").glob("*/ep*.npy"):
            np.save(p, np.load(p)[:-1])                                       # one frame short
        s, ok = check(t, w)
        assert not ok and sum("recomputed" in p for p in s["problems"]) == 8, s["problems"]


def test_provenance_mismatch_is_a_problem():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w, dtype_override="bf16")
        s, ok = check(t, w)
        assert not ok and sum("dtype 'bf16'" in p for p in s["problems"]) == 8, s["problems"]


def test_unpinned_or_wrong_loaded_commit_is_a_problem():
    with tempfile.TemporaryDirectory() as t:
        w = weights()
        make_cache(t, w)
        s, ok = check(t, w, loaded="1" * 40)
        assert not ok and sum("not what was loaded" in p for p in s["problems"]) == 2, s["problems"]
        s, ok = check(t, w, rev=None, loaded=None)
        assert not ok and sum("not pinned" in p for p in s["problems"]) == 2, s["problems"]


def test_compare_metrics():
    c = np.ones((2, 14, 28, D), np.float16)
    m = CF.compare(c, c)
    assert m["cos_min"] > 0.999999 and m["rel_max"] == 0 and m["frac_bit_equal"] == 1
    assert m["rel_l2"] == 0
    m = CF.compare(-c, c)
    assert m["cos_min"] < -0.999 and not CF.passes(m)
    # per-frame relative L2: one frame 3% too large -> rel_l2 0.03 on that frame; cosine is blind to the scale
    f = c.copy()
    f[1] = (c[1].astype(np.float32) * 1.03).astype(np.float16)
    m = CF.compare(f, c)
    assert abs(m["rel_l2"] - 0.03) < 2e-3 and m["cos_min"] > 0.999999, m
    assert not CF.passes(m) and CF.passes(m, rel_max=1, rel_l2_max=0.05)
    assert not CF.passes(dict(m, rel_max=0.0), rel_max=1)          # the L2 bar alone fails it


def test_main_exit_codes():
    orig = CF.run
    base = ["--data-root", "x", "--feats", "y", "--device", "cpu"]
    summ = lambda ok: ({"ok": ok, "problems": [], "episodes": [{"pass": ok}]}, ok)
    try:
        CF.run = lambda a: summ(True)
        assert CF.main(base) == 0
        CF.run = lambda a: summ(False)
        assert CF.main(base) == 3

        def boom(a):
            raise RuntimeError("no data")
        CF.run = boom
        assert CF.main(base) == 2
    finally:
        CF.run = orig
    try:
        CF.main(base + ["--sources", "nope"])
        raise AssertionError("bad source accepted")
    except SystemExit as e:
        assert e.code == 2


def test_defaults_match_extraction_jobs():
    """Dtype and batch of the extraction Jobs (job-extract-<source>.yaml) and the pinned revisions."""
    import extract_features as EF
    assert CF.EXTRACT == {"dinov3": "fp32", "siglip2": "fp16", "c-radiov4": "fp16"} and CF.BATCH == 32
    assert set(CF.SOURCES) <= set(EF.REVISIONS) and all(len(EF.REVISIONS[s]) == 40 for s in CF.SOURCES)
    assert (CF.COS_MIN, CF.REL_MAX, CF.REL_L2_MAX, CF.PER_SUITE, CF.SEED) == (0.999, 0.02, 0.02, 2, 0)


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k, flush=True)
    print(f"{n} tests passed")
