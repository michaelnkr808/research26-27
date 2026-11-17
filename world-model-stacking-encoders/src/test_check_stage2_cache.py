"""Run: /opt/homebrew/bin/python3.11 -I test_check_stage2_cache.py  (needs numpy; fake cache, small D, no cluster)"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_stage2_cache as C

DIMS = {"dinov3": 4, "siglip2": 6, "c-radiov4": 6, "vae": 3}
SUITES = ["libero_goal", "libero_10"]
EPS = {"libero_goal": [(0, 5), (1, 7)], "libero_10": [(3, 4)]}


def put(feats, source, suite, ep, n, shape=None, dtype=np.float16, stats=True, folder=None):
    d = os.path.join(feats, folder or source, suite)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"ep{ep:06d}.npy"), np.zeros(shape or C.expected_shape(source, n, DIMS), dtype))
    if stats and source != "vae":
        np.save(os.path.join(d, f"ep{ep:06d}.stats.npy"),
                np.stack([np.full(DIMS[source], n * 14 * 28.0), np.ones(DIMS[source]), np.ones(DIMS[source])]))


def build(feats):
    for suite, eps in EPS.items():
        os.makedirs(os.path.join(feats, "meta"), exist_ok=True)
        np.savez(os.path.join(feats, "meta", f"{suite}.npz"), episode_ids=np.array([e for e, _ in eps]),
                 lengths=np.array([n for _, n in eps]))
        for source in C.SOURCES:
            for e, n in eps:
                put(feats, source, suite, e, n, folder="vae_b" if source == "vae" and e == 1 else None)
            d = os.path.join(feats, "vae" if source == "vae" else source, suite)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "_meta.shard0of1.json"), "w") as fh:
                json.dump({"model": f"m-{source}", "hf_id": f"org/{source}", "resolution": 224, "dtype": "fp16"}, fh)


class Check(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.feats = os.path.join(self.t.name, "feats")
        build(self.feats)
        self.lines = []

    def tearDown(self):
        self.t.cleanup()

    def run_check(self):
        return C.check_cache(self.feats, C.SOURCES, SUITES, DIMS, workers=2, out=self.lines.append)

    def test_complete_cache(self):
        gaps, cells = self.run_check()
        self.assertEqual(gaps, [])
        self.assertEqual(sum(n for _, _, n, _ in cells), 4 * 3)
        self.assertTrue(any("hf_id=org/dinov3" in x and "resolution=224" in x for x in self.lines))

    def test_main_exit_codes_and_summary(self):
        out = []
        os.remove(os.path.join(self.feats, "meta", "libero_10.npz"))
        rc = C.main(["--feats", self.feats, "--sources", "dinov3", "--suites", "libero_10"], out=out.append)
        self.assertEqual(rc, 1)
        self.assertTrue(any("meta/libero_10.npz missing" in x for x in out))
        self.assertTrue(any(x.startswith("=== FAIL") for x in out))

    def test_main_clean_exit_with_real_dims(self):
        feats = os.path.join(self.t.name, "real")
        os.makedirs(os.path.join(feats, "meta"))
        np.savez(os.path.join(feats, "meta", "libero_10.npz"), episode_ids=np.array([2]), lengths=np.array([2]))
        d = os.path.join(feats, "dinov3", "libero_10")
        os.makedirs(d)
        np.save(os.path.join(d, "ep000002.npy"), np.zeros((2, 14, 28, 1024), np.float16))
        np.save(os.path.join(d, "ep000002.stats.npy"), np.stack([np.full(1024, 784.0), np.ones(1024), np.ones(1024)]))
        with open(os.path.join(d, "_meta.shard0of1.json"), "w") as fh:
            json.dump({"model": "m", "hf_id": "org/dinov3", "resolution": 224}, fh)
        out = []
        rc = C.main(["--feats", feats, "--sources", "dinov3", "--suites", "libero_10"], out=out.append)
        self.assertEqual(rc, 0)
        self.assertTrue(any(x.startswith("=== OK") for x in out))
        self.assertTrue(any(x.startswith("volume:") for x in out))
        self.assertEqual(C.main(["--feats", feats, "--sources", "dinov3", "--suites", "libero_10",
                                 "--min-free-gib", "1e9"], out=out.append), 1)

    def test_missing_npy(self):
        os.remove(os.path.join(self.feats, "dinov3", "libero_goal", "ep000001.npy"))
        gaps, _ = self.run_check()
        self.assertEqual(gaps, ["dinov3/libero_goal/ep000001: missing .npy"])

    def test_missing_stats(self):
        os.remove(os.path.join(self.feats, "siglip2", "libero_10", "ep000003.stats.npy"))
        gaps, _ = self.run_check()
        self.assertEqual(gaps, ["siglip2/libero_10/ep000003: missing .stats.npy"])

    def test_wrong_shape(self):
        put(self.feats, "c-radiov4", "libero_goal", 0, 5, shape=(5, 14, 28, 5))
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1)
        self.assertIn("wrong shape", gaps[0])

    def test_wrong_length(self):
        put(self.feats, "dinov3", "libero_goal", 1, 7, shape=(6, 14, 28, 4))
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1)
        self.assertIn("wrong length", gaps[0])

    def test_wrong_dtype_and_vae_length(self):
        put(self.feats, "dinov3", "libero_goal", 0, 5, dtype=np.float32)
        put(self.feats, "vae", "libero_goal", 0, 5, shape=(4, 3, 14, 28, 3))
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 2)
        self.assertTrue(any("dtype float32" in g for g in gaps))
        self.assertTrue(any("vae/libero_goal/ep000000: wrong length" in g for g in gaps))

    def test_vae_found_in_vae_b(self):
        self.assertFalse(os.path.exists(os.path.join(self.feats, "vae", "libero_goal", "ep000001.npy")))
        gaps, _ = self.run_check()
        self.assertEqual(gaps, [])

    def test_bad_stats_count_and_missing_provenance(self):
        np.save(os.path.join(self.feats, "dinov3", "libero_10", "ep000003.stats.npy"), np.ones((3, 4)))
        os.remove(os.path.join(self.feats, "siglip2", "libero_10", "_meta.shard0of1.json"))
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 2)
        self.assertTrue(any("stats count" in g for g in gaps))
        self.assertTrue(any("no _meta.shard" in g for g in gaps))

    def test_no_writes(self):
        before = sorted(os.path.join(a, f) for a, _, fs in os.walk(self.feats) for f in fs)
        self.run_check()
        after = sorted(os.path.join(a, f) for a, _, fs in os.walk(self.feats) for f in fs)
        self.assertEqual(before, after)

    def test_default_dims(self):
        self.assertEqual(C.DIMS, {"dinov3": 1024, "siglip2": 1152, "c-radiov4": 1152, "vae": 48})


if __name__ == "__main__":
    unittest.main()
