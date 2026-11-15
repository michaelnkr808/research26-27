"""Synthetic checks for make_sizing_slice.py. Run from src/: python test_make_sizing_slice.py

Uses test_proxy_wam.fake_cache and proxy_wam.Data, the real loader. proxy_wam imports Fast-WAM's model classes at the top,
but Data never touches them, so when `fastwam` is not installed (e.g. on the Mac) harmless stand-in modules are put in
sys.modules before the import; nothing in this test builds or runs a model.
"""

import hashlib
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

try:
    import fastwam  # noqa: F401
except ImportError:
    for name, attrs in {"fastwam": {}, "fastwam.models": {}, "fastwam.models.wan22": {},
                        "fastwam.models.wan22.action_dit": {"ActionDiT": object},
                        "fastwam.models.wan22.fastwam_joint": {"FastWAMJoint": object},
                        "fastwam.models.wan22.mot": {"MoT": object},
                        "fastwam.models.wan22.wan_video_dit": {"WanVideoDiT": object}}.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
    print("note: fastwam not installed, stand-in modules used (Data does not need them)")

import make_sizing_slice as M
import proxy_wam as P
import t5_cache
from test_proxy_wam import fake_cache

SUITES = ["suite_a", "suite_b", "suite_c", "suite_d"]
SOURCES = ["enc1", "enc2"]


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        feats = Path(d) / "feats"
        # 4 suites x 3 episodes = 12 training episodes; the last suite's VAE latents sit in vae_b, as in the real cache
        fake_cache(feats, SUITES, SOURCES, rng)
        (feats / "vae_b").mkdir()
        (feats / "vae" / "suite_d").rename(feats / "vae_b" / "suite_d")
        full = P.Data(feats, SUITES, SOURCES, feats / "t5")
        assert len(full.episodes) == 12

        # 0. the T5 file name logic equals t5_cache's
        for t in ("pick it", "place it", "x"):
            assert M.t5_name(t) == t5_cache.cache_path("", t5_cache.prompt_of(t)).name, t
        assert M.DEFAULT_PROMPT == t5_cache.DEFAULT_PROMPT
        print("0 ok: T5 file names equal t5_cache.cache_path")

        # 1. exactly n training episodes, no repeats, all from the full meta
        out = Path(d) / "slice"
        n = 5
        info = M.make_slice(feats, out, n=n, seed=0, encoders=SOURCES, suites=SUITES, log=lambda m: None)
        ids = [(e["suite"], e["episode_id"]) for e in info["episodes"]]
        assert len(ids) == n == len(set(ids)) and set(ids) <= {k[:2] for k in full.episodes}, ids
        print("1 ok: slice holds exactly", n, "distinct training episodes:", ids)

        # 2. the same seed gives the same ids (also from a fresh folder); another seed differs
        again = M.make_slice(feats, Path(d) / "slice2", n=n, seed=0, encoders=SOURCES, suites=SUITES, log=lambda m: None)
        assert [(e["suite"], e["episode_id"]) for e in again["episodes"]] == ids
        other = M.make_slice(feats, Path(d) / "slice3", n=n, seed=1, encoders=SOURCES, suites=SUITES, log=lambda m: None)
        assert [(e["suite"], e["episode_id"]) for e in other["episodes"]] != ids
        print("2 ok: same seed same ids, different seed different ids")

        # 3. proxy_wam's own loader reads the slice unchanged: same episodes, arrays equal the full cache's, z-score
        #    stats are the sum over the slice's episodes only, T5 context loads for every task
        sl = P.Data(out, info["suites_in_slice"], SOURCES, out / "t5")
        assert [k[:2] for k in sl.episodes] == ids, (sl.episodes, ids)
        assert sl.in_dim == full.in_dim and len(sl.episodes) == n
        for key in sl.episodes:
            k = key[:2]
            a, b = sl.load_episode(k), full.load_episode(k)
            assert a.keys() == b.keys() and all(np.array_equal(a[s], b[s]) for s in a), k
            assert np.array_equal(sl.action[k], full.action[k]) and np.array_equal(sl.state[k], full.state[k]), k
        for src in SOURCES:
            tot = sum(np.load(feats / src / s / f"ep{e:06d}.stats.npy") for s, e in ids)
            mu = tot[1] / tot[0]
            assert np.allclose(sl.fstats[src][0], mu.astype(np.float32))
        assert sl.context is not None and sl.context.shape[0] == len(sl.tasks)
        # a batch assembles through the real Data.assemble
        cache = {k[:2]: {s: sl.load_file(k[:2], s) for s in [P.VAE] + SOURCES} for k in sl.episodes}
        bt = sl.assemble([(0, 0), (1, 5)], cache)
        assert bt["latents"].shape == (2, P.LATENT_T) + P.GRID + (sl.in_dim,)
        print("3 ok: proxy_wam.Data loads the slice; arrays, actions and proprio equal the full cache's; batch assembles")

        # 4. manifest: a sha256 per file, hashes match the bytes, nothing missing or extra, copies equal the source files
        lines = (out / "MANIFEST.sha256").read_text().splitlines()
        listed = {}
        for ln in lines:
            h, rel = ln.split("  ", 1)
            listed[rel] = h
        on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"MANIFEST.sha256"}
        assert set(listed) == on_disk, set(listed) ^ on_disk
        for rel, h in listed.items():
            assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == h, rel
            if not rel.startswith("meta/") and rel != "slice.json":
                src = feats / rel
                if not src.exists():
                    src = feats / "vae_b" / rel.split("/", 1)[1]
                assert src.read_bytes() == (out / rel).read_bytes(), rel
        assert "meta/suite_a.npz" in listed or any(r.startswith("meta/") for r in listed)
        print("4 ok: manifest lists", len(listed), "files; hashes match; copied files are byte-identical")

        # 5. a non-empty output folder and a missing file are refused, not silently half done
        try:
            M.make_slice(feats, out, n=n, seed=0, encoders=SOURCES, suites=SUITES, log=lambda m: None)
            raise SystemExit("expected refusal for a non-empty folder")
        except AssertionError:
            pass
        (feats / "enc1" / ids[0][0] / f"ep{ids[0][1]:06d}.stats.npy").unlink()
        try:
            M.make_slice(feats, Path(d) / "slice4", n=n, seed=0, encoders=SOURCES, suites=SUITES, log=lambda m: None)
            raise SystemExit("expected refusal for a missing stats file")
        except AssertionError as e:
            assert "missing" in str(e)
            assert not (Path(d) / "slice4").exists()
        print("5 ok: refuses a non-empty folder and a missing source file before copying anything")
    print("all ok")


if __name__ == "__main__":
    main()
