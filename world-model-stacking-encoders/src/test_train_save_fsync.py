"""Tests for the fsync before each rename in the training writers (LOG 2026-11-18: a CephFS pool-full event left
files of zeros under their final names; PR #64 fixed the feature writer the same way). The writers are
proxy_wam.save_ckpt (ckpt.pt), pretrained_train._write_json (config.json, train.json and the other reports) and the
model.pt save at the end of pretrained_train._main.

  1. Same content and bytes as the code before the change (commit OLD_COMMIT): the old src/ tree from "git archive"
     and the current src/ each run the same driver in a fresh process on the same fake data and the same output path
     (pretrained_train.main stopped after step 2, then resumed to step 4; plus one direct save_ckpt call). ckpt.pt,
     model.pt, the direct ckpt.pt, config.json and load.json are compared: loaded content bit for bit, and raw bytes.
  2. A refused write (os.fsync raising OSError(ENOSPC)) raises out of each writer and leaves the previous final file
     byte-identical.
  3. fsync is called on the file that is renamed onto the final name (same inode).
  4. The checkpoint write time that pretrained_train.save returns (logged and used for the projection) includes the
     fsync.
  5. Exit path: in a real process, a refused checkpoint write ends pretrained_train.py with exit code 1 (not one of
     the Job's FailJob codes, so the Job retries the pod), ckpt.pt keeps the last good step, and the next process
     resumes from it and finishes.

Run from src/ with the CPU torch venv: python -m pytest test_train_save_fsync.py -q
Needs FastWAM 7faa711 importable, safetensors, and git with OLD_COMMIT available.
"""

import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

OLD_COMMIT = "1b1fb13fe2c2cddfeefb858e5db0810cd831fb2a"     # the last commit before the fsync change
HERE = Path(__file__).resolve().parent
SUITES = "suite_a,suite_b"
LENGTHS = (200, 150, 250)
GIT_HASH = "train-save-fsync-test"       # SRC_GIT_COMMIT for both processes, so config["git_commit"] is equal
COMPARED = ("R/ckpt.pt", "R/model.pt", "direct/ckpt.pt", "R/config.json", "R/load.json")
ENOSPC = OSError(errno.ENOSPC, "No space left on device")


def argv(d, out, *extra):
    return ["--feats", str(d), "--t5", str(d / "t5"), "--suites", SUITES, "--config", "none", "--wan-dir",
            str(d / "wan"), "--keep", "0-1", "--hidden", "48", "--action-hidden", "16", "--heads", "2",
            "--head-dim", "24", "--device", "cpu", "--global-batch", "32", "--micro-batch", "8",
            "--allow-any-micro-batch", "--buffer-episodes", "16", "--read-threads", "2", "--assemble-threads", "1",
            "--epochs", "1", "--step-watchdog-s", "0", "--out", str(out), *extra]


def make_fixture(d):
    import numpy as np
    from test_pretrained_sizing import fake_wan
    from test_proxy_wam import fake_cache
    fake_cache(d, SUITES.split(","), [], np.random.default_rng(0), n_eps=3, lengths=LENGTHS)
    fake_wan(d / "wan", n_blocks=4)


def quiet_main(T, a):
    with contextlib.redirect_stdout(io.StringIO()):
        return T.main(a)


# ---------------------------------------------------------------------------
# Driver: runs inside a fresh process with the code under test first on sys.path
# ---------------------------------------------------------------------------

def driver(src, d, out):
    sys.path.insert(0, str(src))
    import torch

    import pretrained_train as T
    import proxy_wam as P
    for m in (T, P):
        assert Path(m.__file__).resolve().parent == Path(src).resolve(), (m.__name__, m.__file__)
    torch.set_num_threads(2)
    d, out = Path(d), Path(out)
    rc = quiet_main(T, argv(d, out / "R", "--max-steps", "4", "--ckpt-every", "2", "--stop-after", "2"))
    assert rc == 3, rc
    rc = quiet_main(T, argv(d, out / "R", "--max-steps", "4", "--ckpt-every", "2"))
    assert rc == 0, rc
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    (out / "direct").mkdir(parents=True)
    P.save_ckpt(out / "direct" / "ckpt.pt", m, opt, 7, note="direct")


def run_driver(src, d, out):
    env = {**os.environ, "SRC_GIT_COMMIT": GIT_HASH}
    r = subprocess.run([sys.executable, str(HERE / "test_train_save_fsync.py"), "--driver", str(src), str(d), str(out)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, (src, r.stdout[-3000:], r.stderr[-3000:])


def old_tree(dest):
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    rel = HERE.relative_to(Path(top).resolve()).as_posix()
    tar = subprocess.check_output(["git", "archive", OLD_COMMIT, rel], cwd=top)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest)
    return Path(dest) / rel


def diff(a, b, path=""):
    """Differences between two loaded objects: tensors and arrays bit for bit, everything else by ==."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return [f"{path}: tensor vs non-tensor"]
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}"]
        ab = a.detach().contiguous().reshape(-1).view(torch.uint8) if a.numel() else a
        bb = b.detach().contiguous().reshape(-1).view(torch.uint8) if b.numel() else b
        return [] if torch.equal(ab, bb) else [f"{path}: tensor bytes differ"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        ok = (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
              and a.tobytes() == b.tobytes())
        return [] if ok else [f"{path}: array differs"]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f"{path}: keys {list(a)} vs {list(b)}"]
        return [x for k in a for x in diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    if type(a) is not type(b):
        return [f"{path}: {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, float):
        return [] if a.hex() == b.hex() else [f"{path}: {a!r} vs {b!r}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def load(p):
    import torch
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class SameAsOldCode(unittest.TestCase):
    def test_old_and_new_writers_save_the_same(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            make_fixture(t / "data")
            old_src = old_tree(t / "old")
            self.assertTrue((old_src / "pretrained_train.py").exists())
            run_driver(old_src, t / "data", t / "run")
            (t / "run").rename(t / "run-old")
            run_driver(HERE, t / "data", t / "run")
            report = []
            for f in COMPARED:
                a, b = t / "run-old" / f, t / "run" / f
                self.assertEqual(diff(load(a), load(b)), [], f)
                report.append(f"{f}: content equal, raw bytes {'identical' if a.read_bytes() == b.read_bytes() else 'DIFFER'}"
                              f" ({a.stat().st_size} bytes)")
            sys.stderr.write("\n" + "\n".join(report) + "\n")
            for f in COMPARED:
                self.assertEqual((t / "run-old" / f).read_bytes(), (t / "run" / f).read_bytes(), f)
            leftovers = sorted(p.name for p in (t / "run").rglob("*.tmp"))
            self.assertEqual(leftovers, [])


class RefusedWrite(unittest.TestCase):
    """os.fsync raising ENOSPC: the writer raises, the older final file stays byte-identical."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)

    def tearDown(self):
        self.t.cleanup()

    def small(self, seed):
        import torch
        torch.manual_seed(seed)
        m = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        m(torch.randn(2, 4)).sum().backward()
        opt.step()
        return m, opt

    def test_save_ckpt(self):
        import proxy_wam as P
        p = self.root / "ckpt.pt"
        P.save_ckpt(p, *self.small(0), 2)
        good = p.read_bytes()
        with mock.patch("os.fsync", side_effect=ENOSPC):
            with self.assertRaises(OSError) as cm:
                P.save_ckpt(p, *self.small(1), 4)
        self.assertEqual(cm.exception.errno, errno.ENOSPC)
        self.assertEqual(p.read_bytes(), good)

    def test_write_json(self):
        import pretrained_train as T
        p = self.root / "train.json"
        T._write_json(p, {"step": 2})
        good = p.read_bytes()
        self.assertEqual(good, json.dumps({"step": 2}, indent=1).encode())     # what Path.write_text wrote before
        with mock.patch("os.fsync", side_effect=ENOSPC):
            with self.assertRaises(OSError) as cm:
                T._write_json(p, {"step": 4})
        self.assertEqual(cm.exception.errno, errno.ENOSPC)
        self.assertEqual(p.read_bytes(), good)

    def test_model_pt(self):
        import pretrained_train as T
        d = self.root / "data"
        make_fixture(d)
        out = self.root / "R"
        self.assertEqual(quiet_main(T, argv(d, out, "--max-steps", "4", "--ckpt-every", "2")), 0)
        good, good_ckpt, good_train = ((out / f).read_bytes() for f in ("model.pt", "ckpt.pt", "train.json"))
        # a second process on the same folder resumes from ckpt.pt (step 2) and writes model.pt again at step 4;
        # only the fsync of model.pt.tmp is refused, so the run gets as far as that write
        real = os.fsync

        def refuse_model_pt(fd):
            tmp = out / "model.pt.tmp"
            if tmp.exists() and os.fstat(fd).st_ino == tmp.stat().st_ino:
                raise ENOSPC
            return real(fd)

        with mock.patch("os.fsync", side_effect=refuse_model_pt):
            with self.assertRaises(OSError) as cm:
                quiet_main(T, argv(d, out, "--max-steps", "4", "--ckpt-every", "2"))
        self.assertEqual(cm.exception.errno, errno.ENOSPC)
        self.assertEqual((out / "model.pt").read_bytes(), good)
        self.assertEqual((out / "ckpt.pt").read_bytes(), good_ckpt)
        self.assertEqual((out / "train.json").read_bytes(), good_train)       # not reached


class FsyncTarget(unittest.TestCase):
    """fsync is called on the file that becomes the final file (a rename keeps the inode)."""

    def test_each_writer(self):
        import proxy_wam as P
        import pretrained_train as T
        import torch
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            real, seen = os.fsync, []

            def spy(fd):
                seen.append(os.fstat(fd).st_ino)
                return real(fd)

            m = torch.nn.Linear(4, 3)
            with mock.patch("os.fsync", side_effect=spy):
                P.save_ckpt(t / "ckpt.pt", m, torch.optim.AdamW(m.parameters()), 1)
            self.assertEqual(seen, [(t / "ckpt.pt").stat().st_ino])
            seen.clear()
            with mock.patch("os.fsync", side_effect=spy):
                T._write_json(t / "a.json", {"x": 1})
            self.assertEqual(seen, [(t / "a.json").stat().st_ino])
            make_fixture(t / "data")
            seen.clear()
            with mock.patch("os.fsync", side_effect=spy):
                self.assertEqual(quiet_main(T, argv(t / "data", t / "R", "--max-steps", "2")), 0)
            self.assertIn((t / "R" / "model.pt").stat().st_ino, seen)
            for f in ("model.pt", "config.json", "load.json", "train.json"):
                self.assertIn((t / "R" / f).stat().st_ino, seen, f)


class WriteTimeIncludesFsync(unittest.TestCase):
    def test_save_time(self):
        import pretrained_train as T
        import torch
        real = os.fsync

        def slow(fd):
            time.sleep(0.5)
            return real(fd)

        m = torch.nn.Linear(4, 3)
        args = T.parser().parse_args(["--feats", "x", "--t5", "x", "--wan-dir", "x", "--out", "x"])
        args.run_identity = {}
        logs = []
        with tempfile.TemporaryDirectory() as t:
            with mock.patch("os.fsync", side_effect=slow):
                dt = T.save(Path(t) / "ckpt.pt", m, torch.optim.AdamW(m.parameters()), 1, args, [], logs.append)
        self.assertGreaterEqual(dt, 0.5)
        self.assertTrue(logs and "[ckpt] step 1 written" in logs[0], logs)


EXIT_SCRIPT = """
import errno, os, sys
sys.path.insert(0, sys.argv[1])
def refuse(fd):
    raise OSError(errno.ENOSPC, "No space left on device")
real = os.fsync
import pretrained_train as T
if sys.argv[2] == "refuse-ckpt":
    def fs(fd, out=sys.argv[3]):
        tmp = os.path.join(out, "ckpt.pt.tmp")
        if os.path.exists(tmp) and os.fstat(fd).st_ino == os.stat(tmp).st_ino:
            refuse(fd)
        return real(fd)
    os.fsync = fs
sys.exit(T.main(sys.argv[4:]))
"""


class ExitPath(unittest.TestCase):
    def test_refused_ckpt_write_exits_1_and_the_retry_resumes(self):
        import torch
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            make_fixture(t / "data")
            out = t / "R"
            script = t / "run.py"
            script.write_text(EXIT_SCRIPT)

            def run(mode, *extra):
                return subprocess.run([sys.executable, str(script), str(HERE), mode, str(out),
                                       *argv(t / "data", out, "--max-steps", "6", "--ckpt-every", "2", *extra)],
                                      capture_output=True, text=True, env={**os.environ, "SRC_GIT_COMMIT": GIT_HASH})

            r = run("plain", "--stop-after", "2")
            self.assertEqual(r.returncode, 3, r.stderr[-2000:])
            good = (out / "ckpt.pt").read_bytes()
            r = run("refuse-ckpt")              # resumes at step 2; the write at step 4 is refused
            self.assertEqual(r.returncode, 1, (r.stdout[-2000:], r.stderr[-2000:]))
            self.assertIn("OSError: [Errno 28]", r.stderr)
            self.assertIn("save_ckpt", r.stderr)
            self.assertEqual((out / "ckpt.pt").read_bytes(), good)
            self.assertFalse((out / "model.pt").exists())
            r = run("plain")                     # the Job's retry: resumes from the step 2 checkpoint and finishes
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("resuming from", r.stdout)
            self.assertIn("at step 2", r.stdout)
            self.assertEqual(torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["step"], 4)
            self.assertTrue((out / "model.pt").exists())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--driver":
        driver(*sys.argv[2:5])
    else:
        unittest.main()
