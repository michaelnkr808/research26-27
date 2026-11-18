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

    def test_truncated_npy_is_a_gap(self):
        p = os.path.join(self.feats, "dinov3", "libero_goal", "ep000000.npy")
        with open(p, "rb") as fh:
            data = fh.read()
        with open(p, "wb") as fh:
            fh.write(data[:len(data) // 2])        # header intact, data cut short
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1)
        self.assertIn("dinov3/libero_goal/ep000000: cannot read .npy header", gaps[0])

    def test_unreadable_npy_is_a_gap(self):
        with open(os.path.join(self.feats, "siglip2", "libero_10", "ep000003.npy"), "wb") as fh:
            fh.write(b"not a numpy file at all")
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1)
        self.assertIn("siglip2/libero_10/ep000003: cannot read .npy header", gaps[0])

    def test_unreadable_stats_is_a_gap(self):
        with open(os.path.join(self.feats, "dinov3", "libero_10", "ep000003.stats.npy"), "wb") as fh:
            fh.write(b"junk")
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1)
        self.assertIn("cannot read .stats.npy", gaps[0])

    def test_bad_meta_json_is_a_gap(self):
        with open(os.path.join(self.feats, "c-radiov4", "libero_goal", "_meta.shard0of1.json"), "w") as fh:
            fh.write("{not json")
        with open(os.path.join(self.feats, "vae", "libero_10", "_meta.shard0of1.json"), "w") as fh:
            fh.write("[1, 2]")
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 2)
        self.assertTrue(any("c-radiov4/libero_goal: cannot read" in g and "JSONDecodeError" in g for g in gaps))
        self.assertTrue(any("vae/libero_10: cannot read" in g for g in gaps))

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


# ---- --values (value-level check). Random non-zero data and exact stats, so an intact cache passes. ----

def put_values(feats, source, suite, ep, n, dims=DIMS, folder=None, rng=None):
    """Writes ep<id>.npy with random float16 data and (encoders) its .stats.npy exactly as extract_fastwam.py does."""
    rng = rng or np.random.default_rng(1000 * ep + len(source) + len(suite))
    d = os.path.join(feats, folder or source, suite)
    os.makedirs(d, exist_ok=True)
    arr = (rng.standard_normal(C.expected_shape(source, n, dims)) + 0.3).astype(np.float16)
    np.save(os.path.join(d, f"ep{ep:06d}.npy"), arr)
    if source != "vae":
        x = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
        np.save(os.path.join(d, f"ep{ep:06d}.stats.npy"),
                np.stack([np.full(x.shape[1], len(x)), x.sum(0), (x * x).sum(0)]))
    return arr


def build_values(feats):
    build(feats)                                             # meta and provenance as above
    for suite, eps in EPS.items():
        for source in C.SOURCES:
            for e, n in eps:
                put_values(feats, source, suite, e, n, folder="vae_b" if source == "vae" and e == 1 else None)


def zero_bytes(path, start, length):
    with open(path, "r+b") as fh:
        fh.seek(start)
        fh.write(b"\0" * length)


class Values(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.feats = os.path.join(self.t.name, "feats")
        build_values(self.feats)
        self.lines = []

    def tearDown(self):
        self.t.cleanup()

    def run_check(self, values=True):
        return C.check_cache(self.feats, C.SOURCES, SUITES, DIMS, workers=2, out=self.lines.append, values=values)

    def p(self, source, suite, ep, stats=False, folder=None):
        return os.path.join(self.feats, folder or source, suite, f"ep{ep:06d}{'.stats' if stats else ''}.npy")

    def test_tolerance_constants(self):
        self.assertEqual((C.PAGE, C.CHUNK_BYTES, C.RTOL_SUMSQ, C.RTOL_SUM), (4096, 4 * 2**20, 1e-9, 1e-9))

    def test_intact_cache_passes_with_values(self):
        gaps, _ = self.run_check()
        self.assertEqual(gaps, [])
        self.assertIn("  values: 2 of 2 episodes value-checked", self.lines)
        self.assertIn("  values: 1 of 1 episodes value-checked", self.lines)

    def test_main_values_flag_and_summary(self):
        feats = os.path.join(self.t.name, "real")            # main() uses the real dims
        os.makedirs(os.path.join(feats, "meta"))
        np.savez(os.path.join(feats, "meta", "libero_10.npz"), episode_ids=np.array([2, 5]), lengths=np.array([2, 3]))
        for source in ("dinov3", "vae"):
            for e, n in ((2, 2), (5, 3)):
                put_values(feats, source, "libero_10", e, n, dims=C.DIMS)
        with open(os.path.join(feats, "dinov3", "libero_10", "_meta.shard0of1.json"), "w") as fh:
            json.dump({"model": "m", "hf_id": "org/dinov3", "resolution": 224}, fh)
        out = []
        rc = C.main(["--feats", feats, "--sources", "dinov3,vae", "--suites", "libero_10", "--values",
                     "--workers", "2"], out=out.append)
        self.assertEqual(rc, 0, out)
        self.assertIn("  values: 2 of 2 episodes value-checked", out)
        self.assertTrue(any(x.startswith("values: 4 episodes value-checked") and "GB read in" in x for x in out))
        self.assertIn("=== damaged episodes (every episode with a gap of any kind, uncapped): 0", out)
        self.assertTrue(any(x.startswith("=== OK") for x in out))

    def test_main_lists_every_damaged_episode_uncapped(self):
        feats = os.path.join(self.t.name, "real")
        os.makedirs(os.path.join(feats, "meta"))
        ids = list(range(C.MAX_SHOWN + 3))
        np.savez(os.path.join(feats, "meta", "libero_10.npz"), episode_ids=np.array(ids), lengths=np.array([1] * len(ids)))
        for e in ids:
            put_values(feats, "vae", "libero_10", e, 1, dims=C.DIMS)
            zero_bytes(os.path.join(feats, "vae", "libero_10", f"ep{e:06d}.npy"), 4096, 4096)
        out = []
        rc = C.main(["--feats", feats, "--sources", "vae", "--suites", "libero_10", "--values"], out=out.append)
        self.assertEqual(rc, 1)
        self.assertEqual([x for x in out if x.startswith("DAMAGED ")], [f"DAMAGED vae libero_10 {e}" for e in ids])
        self.assertIn("=== FAIL: 18 gap(s)", out)

    def test_zeroed_page_in_middle(self):
        path = self.p("siglip2", "libero_goal", 1)              # 7 frames: 128 + 7 * 392 * 6 * 2 = 33,056 bytes
        zero_bytes(path, 8192, 4096)
        self.assertIsNotNone(np.load(path, mmap_mode="r"))      # header still valid
        gaps, _ = self.run_check()
        self.assertTrue(gaps and all(g.startswith("siglip2/libero_goal/ep000001:") for g in gaps), gaps)
        self.assertTrue(any("1 all-zero 4096-byte page(s) in .npy, first at offset 8192" in g for g in gaps))
        self.assertTrue(any("sum of squares differs" in g for g in gaps))

    def test_damage_in_middle_passes_without_values(self):
        zero_bytes(self.p("siglip2", "libero_goal", 1), 8192, 4096)
        gaps, _ = self.run_check(values=False)
        self.assertEqual(gaps, [])                              # what the header check cannot see
        self.assertFalse(any("values:" in x for x in self.lines))

    def test_zeroed_tail(self):
        path = self.p("dinov3", "libero_goal", 0)
        size = os.path.getsize(path)
        zero_bytes(path, size - 1000, 1000)                     # under one page: only the stats comparison sees it
        gaps, _ = self.run_check()
        self.assertTrue(any(g.startswith("dinov3/libero_goal/ep000000: .stats.npy sum of squares differs") for g in gaps),
                        gaps)

    def test_all_zero_npy_and_stats_pair(self):
        path, sp = self.p("siglip2", "libero_10", 3), self.p("siglip2", "libero_10", 3, stats=True)
        for f in (path, sp):
            zero_bytes(f, 0, os.path.getsize(f))
        gaps, _ = self.run_check()
        self.assertTrue(any("cannot read .npy header" in g for g in gaps))
        self.assertTrue(any("all-zero 4096-byte page(s) in .npy, first at offset 0" in g for g in gaps))
        self.assertTrue(all(g.startswith("siglip2/libero_10/ep000003:") for g in gaps), gaps)
        self.assertIn("  values: 0 of 1 episodes value-checked", self.lines)

    def test_all_zero_real_sized_stats_file(self):
        # A 27,776-byte SigLIP 2 stats file of zeros next to a good .npy: the header check fails it, the scan reports
        # its zero pages, and nothing crashes.
        feats = os.path.join(self.t.name, "real")
        os.makedirs(os.path.join(feats, "meta"))
        np.savez(os.path.join(feats, "meta", "libero_10.npz"), episode_ids=np.array([2]), lengths=np.array([2]))
        put_values(feats, "siglip2", "libero_10", 2, 2, dims=C.DIMS)
        sp = os.path.join(feats, "siglip2", "libero_10", "ep000002.stats.npy")
        self.assertEqual(os.path.getsize(sp), 27776)
        zero_bytes(sp, 0, 27776)
        gaps, _ = C.check_cache(feats, ["siglip2"], ["libero_10"], C.DIMS, workers=1, out=self.lines.append,
                                values=True)
        self.assertTrue(any("cannot read .stats.npy" in g for g in gaps), gaps)
        self.assertTrue(any("6 all-zero 4096-byte page(s) in .stats.npy, first at offset 0" in g for g in gaps), gaps)

    def test_stats_from_other_data(self):
        sp = self.p("c-radiov4", "libero_goal", 0, stats=True)
        s = np.load(sp)
        other = (np.random.default_rng(7).standard_normal((5 * 392, 6)) + 0.3).astype(np.float16).astype(np.float64)
        s[1] = other.sum(0)                                     # sum row from different data, sum of squares intact
        np.save(sp, s)
        gaps, _ = self.run_check()
        self.assertEqual(len(gaps), 1, gaps)
        self.assertIn("c-radiov4/libero_goal/ep000000: .stats.npy sum differs", gaps[0])

    def test_near_zero_mean_channel_passes(self):
        # A channel whose sum is about 0: a relative tolerance on the sum would flag rounding noise here.
        sp, path = self.p("dinov3", "libero_10", 3, stats=True), self.p("dinov3", "libero_10", 3)
        a = np.load(path)
        a[..., 0] = np.where(np.arange(a[..., 0].size).reshape(a[..., 0].shape) % 2, 1.0009765625, -1.0009765625)
        np.save(path, a)                                        # 4 * 392 rows, an even count: channel 0 sums to 0
        x = a.reshape(-1, a.shape[-1]).astype(np.float64)
        s = np.stack([np.full(x.shape[1], len(x)), x.sum(0), (x * x).sum(0)])
        s[1, 0] += 1e-14                                        # rounding-sized noise on a near-zero sum
        np.save(sp, s)
        gaps, _ = self.run_check()
        self.assertEqual(gaps, [])

    def test_non_finite_value(self):
        path = self.p("dinov3", "libero_goal", 1)
        a = np.load(path)
        a[3, 2, 1, 0] = np.inf
        np.save(path, a)
        gaps, _ = self.run_check()
        self.assertEqual(gaps, ["dinov3/libero_goal/ep000001: 1 non-finite value(s) in .npy"])

    def test_vae_zeroed_page(self):
        path = self.p("vae", "libero_goal", 0)                  # 128 + 5 * 3 * 392 * 3 * 2 = 35,408 bytes
        zero_bytes(path, 4096 * 3, 4096)
        gaps, _ = self.run_check()
        self.assertEqual(gaps, ["vae/libero_goal/ep000000: 1 all-zero 4096-byte page(s) in .npy, first at offset 12288"])

    def test_unaligned_page_in_real_sized_file(self):
        # Sensitivity of the stats comparison alone: 4096 zero bytes that straddle two pages (so the zero-page scan
        # cannot see them) in a SigLIP 2 file of real width. 60 frames keeps the test small; the timing run in
        # RUN-cache-value-check.md repeats this on a 505-frame file.
        feats = os.path.join(self.t.name, "real")
        os.makedirs(os.path.join(feats, "meta"))
        np.savez(os.path.join(feats, "meta", "libero_10.npz"), episode_ids=np.array([2]), lengths=np.array([60]))
        put_values(feats, "siglip2", "libero_10", 2, 60, dims=C.DIMS)
        with open(os.path.join(feats, "siglip2", "libero_10", "_meta.shard0of1.json"), "w") as fh:
            json.dump({"model": "m", "hf_id": "org/siglip2", "resolution": 384}, fh)
        path = os.path.join(feats, "siglip2", "libero_10", "ep000002.npy")
        gaps, _ = C.check_cache(feats, ["siglip2"], ["libero_10"], C.DIMS, workers=1, out=self.lines.append,
                                values=True)
        self.assertEqual(gaps, [])
        zero_bytes(path, os.path.getsize(path) // 2 + 2048 + 128, 4096)
        gaps, _ = C.check_cache(feats, ["siglip2"], ["libero_10"], C.DIMS, workers=1, out=self.lines.append,
                                values=True)
        self.assertTrue(gaps and all("sum of squares differs" in g or "sum differs" in g for g in gaps), gaps)

    def test_values_no_writes(self):
        before = sorted((os.path.join(a, f), os.path.getmtime(os.path.join(a, f)))
                        for a, _, fs in os.walk(self.feats) for f in fs)
        self.run_check()
        after = sorted((os.path.join(a, f), os.path.getmtime(os.path.join(a, f)))
                       for a, _, fs in os.walk(self.feats) for f in fs)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
