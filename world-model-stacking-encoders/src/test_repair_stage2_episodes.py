"""Run: /opt/homebrew/bin/python3.11 -m pytest test_repair_stage2_episodes.py -q
Synthetic caches only (tiny width D = 6, a few frames, temporary folders); no cluster, no GPU, no real features.
The extractor is replaced by fake_extract, which does what extract_fastwam.py main() does for the repair: writes
_meta.shard0of1.json, skips an episode whose ep<id>.npy exists, writes stats then features for the others."""
import errno
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_stage2_cache as CS
import repair_stage2_episodes as R

DIMS = dict(CS.DIMS, siglip2=6)
EPS = {"libero_spatial": [(270, 3), (272, 4), (274, 2), (276, 3)], "libero_goal": [(5, 2), (6, 3)]}
OLD_META = {"model": "google/siglip2-so400m-patch14-384", "resolution": 384, "dtype": "fp16", "device": "cuda",
            "git_commit": "a" * 40}
COMMIT = "b" * 40


def healthy(d, ep, n, rng):
    x = rng.standard_normal((n, 14, 28, DIMS["siglip2"])).astype(np.float16)
    f = x.reshape(-1, x.shape[-1]).astype(np.float64)
    np.save(os.path.join(d, f"ep{ep:06d}.stats.npy"), np.stack([np.full(f.shape[1], len(f)), f.sum(0), (f * f).sum(0)]))
    np.save(os.path.join(d, f"ep{ep:06d}.npy"), x)


def zero_fill(d, ep):
    """What the storage failure left: full-size files of zero bytes, header included."""
    for n in R.episode_files(ep):
        p = os.path.join(d, n)
        size = os.path.getsize(p)
        with open(p, "wb") as f:
            f.write(b"\0" * size)


def build(feats, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(feats, "meta"), exist_ok=True)
    for suite, eps in EPS.items():
        np.savez(os.path.join(feats, "meta", f"{suite}.npz"), episode_ids=np.array([e for e, _ in eps]),
                 lengths=np.array([n for _, n in eps]))
        d = R.suite_dir(feats, suite)
        os.makedirs(d)
        for e, n in eps:
            healthy(d, e, n, rng)
        for m in R.OLD_METAS:
            with open(os.path.join(d, m), "w") as fh:
                json.dump(dict(OLD_META, shard=m), fh)


def fake_extract(feats, suite, ids, device="cuda", seed=1):
    d = R.suite_dir(feats, suite)
    with open(os.path.join(d, R.NEW_META), "w") as fh:
        json.dump(dict(OLD_META, device=device, git_commit=COMMIT), fh)
    rng = np.random.default_rng(seed)
    n = dict(EPS[suite])
    for e in ids:
        if not os.path.exists(os.path.join(d, f"ep{e:06d}.npy")):
            healthy(d, e, n[e], rng)


def tree(root):
    """{relative path: (size, mtime_ns)} of every file under root."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.feats = os.path.join(self.t.name, "feats")
        self.state = os.path.join(self.t.name, "logs", "siglip2-repair-uid")
        build(self.feats)
        self.sp = R.suite_dir(self.feats, "libero_spatial")
        self.lines = []

    def tearDown(self):
        self.t.cleanup()

    def write_list(self, text):
        p = os.path.join(self.t.name, "damaged-episodes.txt")
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def prepare(self, text):
        return R.prepare(self.write_list(text), self.feats, self.state, "node-x", DIMS, self.lines.append)

    def postcheck(self, text):
        return R.postcheck(self.write_list(text), self.feats, self.state, COMMIT, "node-x", "NVIDIA L4", DIMS,
                           self.lines.append)

    def damage(self, *eps):
        for e in eps:
            zero_fill(self.sp, e)


class ListParsing(Base):
    def refused(self, text):
        before = tree(self.feats)
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(text)
        self.assertEqual(cm.exception.code, R.EXIT_LIST)
        self.assertEqual(tree(self.feats), before)                 # nothing touched
        self.assertFalse(os.path.exists(self.state))
        return str(cm.exception)

    def test_other_source_refused(self):
        self.damage(272)
        self.assertIn("only 'siglip2'", self.refused("siglip2 libero_spatial 272\ndinov3 libero_spatial 272\n"))

    def test_malformed_lines_refused(self):
        self.damage(272)
        self.refused("siglip2 libero_spatial\n")
        self.refused("siglip2 libero_spatial 27a\n")
        self.refused("siglip2 libero_spatial -272\n")
        self.refused("siglip2 libero_nope 272\n")
        self.refused("siglip2 libero_spatial 272 extra\n")
        self.refused("siglip2 libero_spatial 272\n\n")

    def test_duplicate_refused(self):
        self.damage(272)
        self.assertIn("duplicate", self.refused("siglip2 libero_spatial 272\nsiglip2 libero_spatial 272\n"))

    def test_empty_list_refused(self):
        self.assertIn("empty", self.refused(""))

    def test_episode_not_in_meta_refused(self):
        self.assertIn("not in", self.refused("siglip2 libero_spatial 999\n"))

    def test_main_check_list(self):
        out = []
        self.assertEqual(R.main(["check-list", "--list", self.write_list("siglip2 libero_spatial 272\n")], out.append), 0)
        self.assertEqual(out[0], "libero_spatial 272")
        self.assertEqual(R.main(["check-list", "--list", self.write_list("vae libero_spatial 272\n")], out.append), 20)
        self.assertTrue(out[-1].startswith("REFUSE (exit 20)"))


class Prepare(Base):
    LIST = "siglip2 libero_spatial 272\nsiglip2 libero_spatial 276\n"

    def test_healthy_listed_episode_refuses_and_nothing_moves(self):
        self.damage(272)                                           # 276 stays healthy
        before = tree(self.feats)
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_HEALTHY)
        self.assertIn("libero_spatial 276", str(cm.exception))
        self.assertEqual(tree(self.feats), before)
        self.assertFalse(os.path.exists(os.path.join(self.state, R.SNAPSHOT)))
        self.assertFalse(os.path.exists(os.path.join(self.state, "libero_spatial")))

    def test_moves_both_files_of_each_damaged_episode_and_nothing_else(self):
        self.damage(272, 276)
        before = tree(self.feats)
        self.assertEqual(self.prepare(self.LIST), 0)
        after = tree(self.feats)
        moved = {os.path.join("siglip2", "libero_spatial", n) for e in (272, 276) for n in R.episode_files(e)}
        self.assertEqual(set(before) - set(after), moved)
        self.assertEqual(set(after) - set(before), set())
        for k in after:
            self.assertEqual(after[k], before[k])                  # every other file untouched
        aside = tree(os.path.join(self.state, "libero_spatial"))
        for e in (272, 276):
            for n in R.episode_files(e):
                self.assertEqual(aside[n], before[os.path.join("siglip2", "libero_spatial", n)])
        with open(os.path.join(self.state, R.PLAN)) as fh:
            self.assertEqual(fh.read(), "libero_spatial 272,276\n")
        self.assertTrue(os.path.exists(os.path.join(self.state, "libero_spatial", R.MARKER)))

    def test_rerun_after_partial_move_resumes(self):
        self.damage(272, 276)
        real = os.rename
        calls = []

        def die_after_one(a, b):
            if calls:
                raise KeyboardInterrupt("pod killed")              # not an OSError: like a SIGKILL mid-move
            calls.append(a)
            real(a, b)
        with mock.patch("os.rename", die_after_one):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare(self.LIST)
        self.assertEqual(len(calls), 1)
        self.assertTrue(os.path.exists(os.path.join(self.state, R.SNAPSHOT)))
        self.lines.clear()
        self.assertEqual(self.prepare(self.LIST), 0)               # the retried pod finishes the moves
        self.assertTrue(any(x.startswith("resume:") for x in self.lines))
        self.assertIn("moves: 3 file(s) renamed by this pod", self.lines)
        for e in (272, 276):
            for n in R.episode_files(e):
                self.assertFalse(os.path.exists(os.path.join(self.sp, n)))
                self.assertTrue(os.path.exists(os.path.join(self.state, "libero_spatial", n)))

    def test_rerun_after_partial_extraction_extracts_the_rest_and_skips_the_repaired(self):
        self.damage(272, 276)
        self.assertEqual(self.prepare(self.LIST), 0)
        fake_extract(self.feats, "libero_spatial", [272])         # the pod died after the first episode
        rep = os.stat(os.path.join(self.sp, "ep000272.npy")).st_mtime_ns
        self.lines.clear()
        self.assertEqual(self.prepare(self.LIST), 0)               # repaired 272 is not a "healthy" refusal
        self.assertTrue(any("ep000272: touched by this Job" in x for x in self.lines))
        self.assertTrue(any("ep000276: touched by this Job" in x for x in self.lines))
        self.assertIn("moves: 0 file(s) renamed by this pod", self.lines)
        with open(os.path.join(self.state, R.PLAN)) as fh:
            ids = [int(x) for x in fh.read().split()[1].split(",")]
        fake_extract(self.feats, "libero_spatial", ids)            # skips 272, writes 276
        self.assertEqual(os.stat(os.path.join(self.sp, "ep000272.npy")).st_mtime_ns, rep)
        self.assertEqual(self.postcheck(self.LIST), 0)

    def test_repaired_episode_with_copy_aside_is_skipped(self):
        self.damage(272, 276)
        self.assertEqual(self.prepare(self.LIST), 0)
        fake_extract(self.feats, "libero_spatial", [272, 276])
        before = tree(self.feats)
        aside = tree(self.state)
        self.lines.clear()
        self.assertEqual(self.prepare(self.LIST), 0)               # a retry after the extraction
        self.assertEqual(tree(self.feats), before)
        self.assertEqual({k: v for k, v in tree(self.state).items() if k.endswith(".npy")},
                         {k: v for k, v in aside.items() if k.endswith(".npy")})
        self.assertIn("moves: 0 file(s) renamed by this pod", self.lines)

    def test_foreign_new_meta_refuses(self):
        self.damage(272, 276)
        with open(os.path.join(self.sp, R.NEW_META), "w") as fh:
            json.dump(OLD_META, fh)
        before = tree(self.feats)
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_STATE)
        self.assertEqual(tree(self.feats), before)

    def test_missing_old_meta_refuses(self):
        self.damage(272, 276)
        os.remove(os.path.join(self.sp, R.OLD_METAS[1]))
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_STATE)

    def test_leftover_tmp_refuses(self):
        self.damage(272, 276)
        open(os.path.join(self.sp, ".ep000272.npy.tmp"), "wb").close()
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_STATE)

    def test_cross_device_rename_fails_without_copy(self):
        self.damage(272, 276)
        before = tree(self.feats)

        def exdev(a, b):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        with mock.patch("os.rename", exdev):
            with self.assertRaises(R.Refuse) as cm:
                self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_MOVE)
        self.assertIn("EXDEV", str(cm.exception))
        self.assertEqual(tree(self.feats), before)                 # nothing lost, nothing copied
        self.assertFalse(any(n.endswith(".npy") for n in tree(self.state)))

    def test_old_meta_changed_between_pods_refuses(self):
        self.damage(272, 276)
        self.assertEqual(self.prepare(self.LIST), 0)
        with open(os.path.join(self.sp, R.OLD_METAS[0]), "a") as fh:
            fh.write(" ")
        with self.assertRaises(R.Refuse) as cm:
            self.prepare(self.LIST)
        self.assertEqual(cm.exception.code, R.EXIT_META_CHANGED)

    def test_other_list_on_retry_refuses(self):
        self.damage(272, 276)
        self.assertEqual(self.prepare(self.LIST), 0)
        with self.assertRaises(R.Refuse) as cm:
            self.prepare("siglip2 libero_spatial 272\n")
        self.assertEqual(cm.exception.code, R.EXIT_STATE)

    def test_two_suites(self):
        self.damage(272)
        zero_fill(R.suite_dir(self.feats, "libero_goal"), 6)
        text = "siglip2 libero_spatial 272\nsiglip2 libero_goal 6\n"
        self.assertEqual(self.prepare(text), 0)
        with open(os.path.join(self.state, R.PLAN)) as fh:
            self.assertEqual(fh.read(), "libero_spatial 272\nlibero_goal 6\n")
        fake_extract(self.feats, "libero_spatial", [272])
        fake_extract(self.feats, "libero_goal", [6])
        self.assertEqual(self.postcheck(text), 0)


class Postcheck(Base):
    LIST = "siglip2 libero_spatial 272\nsiglip2 libero_spatial 276\n"

    def ready(self, ids=(272, 276), device="cuda"):
        self.damage(272, 276)
        self.assertEqual(self.prepare(self.LIST), 0)
        fake_extract(self.feats, "libero_spatial", list(ids), device)

    def record(self):
        with open(os.path.join(self.state, R.RECORD)) as fh:
            return json.load(fh)

    def test_ok_and_record(self):
        self.ready()
        self.assertEqual(self.postcheck(self.LIST), 0)
        r = self.record()
        self.assertTrue(r["ok"])
        self.assertEqual(r["episodes"], [["libero_spatial", 272], ["libero_spatial", 276]])
        self.assertEqual((r["source_commit"], r["node"], r["gpu"]), (COMMIT, "node-x", "NVIDIA L4"))
        self.assertEqual(r["new_meta"]["libero_spatial"]["device"], "cuda")
        n = len(EPS["libero_spatial"]) * 2 + 2
        self.assertEqual(r["files_before_after"]["libero_spatial"], [n, n + 1])
        self.assertTrue(r["start"] and r["end"])
        self.assertIn("=== repair post-check OK", self.lines)

    def test_still_missing_episode_fails(self):
        self.ready(ids=(272,))                                     # 276 never re-extracted
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_STILL_DAMAGED)
        self.assertFalse(self.record()["ok"])

    def test_still_damaged_episode_fails(self):
        self.ready()
        zero_fill(self.sp, 276)
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_STILL_DAMAGED)

    def test_old_meta_changed_fails(self):
        self.ready()
        with open(os.path.join(self.sp, R.OLD_METAS[1]), "w") as fh:
            json.dump(dict(OLD_META, device="cpu"), fh)
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_META_CHANGED)
        self.assertTrue(any("after: sha256" in x and R.OLD_METAS[1] in x for x in self.lines))

    def test_other_file_changed_fails(self):
        self.ready()
        p = os.path.join(self.sp, "ep000270.npy")
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_OTHER_CHANGED)

    def test_extra_file_fails(self):
        self.ready()
        open(os.path.join(self.sp, ".ep000270.npy.tmp"), "wb").close()
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_OTHER_CHANGED)

    def test_moved_copy_missing_fails(self):
        self.ready()
        os.remove(os.path.join(self.state, "libero_spatial", "ep000272.stats.npy"))
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_OTHER_CHANGED)

    def test_new_meta_not_cuda_fails(self):
        self.ready(device="cpu")
        self.assertEqual(self.postcheck(self.LIST), R.EXIT_NEW_META)

    def test_main_exit_code(self):
        self.ready()
        rc = R.main(["postcheck", "--list", self.write_list(self.LIST), "--feats", self.feats, "--state", self.state,
                     "--commit", COMMIT], self.lines.append)
        # main uses the real width (1152), so the tiny synthetic files fail the value check: exit 24, not 0
        self.assertEqual(rc, R.EXIT_STILL_DAMAGED)


class MetaReaders(Base):
    """The extra _meta.shard0of1.json is harmless to the readers of <feats>/<source>/<suite>/_meta.*.json."""

    def test_readers_accept_the_extra_meta(self):
        self.damage(272)
        self.assertEqual(self.prepare("siglip2 libero_spatial 272\n"), 0)
        fake_extract(self.feats, "libero_spatial", [272])
        gaps, lines = CS.provenance(self.feats, "siglip2", "libero_spatial")
        self.assertEqual(gaps, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("(3 file(s))", lines[0])
        import check_cache_fresh as CF
        self.assertEqual(CF.check_provenance(self.feats, "siglip2", "libero_spatial", OLD_META["model"], "fp16"), [])


if __name__ == "__main__":
    unittest.main()
