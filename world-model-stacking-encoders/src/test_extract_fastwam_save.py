"""Tests for extract_fastwam.save_npy: the bytes are np.save's, and a refused flush (os.fsync raising) raises before
the rename, so no file appears under the final name (LOG 2026-11-18, zero-filled SigLIP 2 files).

Run from src/: /opt/homebrew/bin/python3.11 -m pytest test_extract_fastwam_save.py -q
"""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import extract_fastwam as X


def npsave_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


class SaveNpy(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        self.arr = (np.random.default_rng(0).standard_normal((3, 14, 28, 8)) + 0.3).astype(np.float16)
        self.stats = np.stack([np.full(8, 1176.0), np.arange(8.0), np.arange(8.0) ** 2])

    def tearDown(self):
        self.t.cleanup()

    def test_bytes_identical_direct(self):
        for a in (self.arr, self.stats):
            p = self.root / "out" / "suite" / "ep000001.npy"
            self.assertIsNone(X.save_npy(p, a))
            self.assertEqual(p.read_bytes(), npsave_bytes(a))
            self.assertFalse((p.parent / f".{p.name}.tmp").exists())

    def test_bytes_identical_staged(self):
        p = self.root / "out" / "suite" / "ep000002.npy"
        stage = self.root / "stage"
        cp = X.save_npy(p, self.arr, stage)
        self.assertIsInstance(cp, float)
        self.assertEqual(p.read_bytes(), npsave_bytes(self.arr))
        self.assertFalse((stage / p.name).exists())             # staged copy removed, as before
        self.assertFalse((p.parent / f".{p.name}.tmp").exists())

    def test_fsync_called_on_the_file_renamed(self):
        for stage in (None, self.root / "stage"):
            p = self.root / "out" / f"ep{0 if stage is None else 1}.npy"
            seen = []
            real = os.fsync

            def spy(fd):
                seen.append(os.fstat(fd).st_ino)
                return real(fd)

            with mock.patch("os.fsync", side_effect=spy):
                X.save_npy(p, self.arr, stage)
            self.assertEqual(seen, [p.stat().st_ino])               # the file that became path (a rename keeps the inode)

    def test_refused_flush_raises_and_leaves_no_final_file(self):
        for stage in (None, self.root / "stage"):
            p = self.root / "out" / "suite" / f"ep{3 if stage is None else 4:06d}.npy"
            with mock.patch("os.fsync", side_effect=OSError(28, "No space left on device")):
                with self.assertRaises(OSError):
                    X.save_npy(p, self.arr, stage)
            self.assertFalse(p.exists(), stage)


if __name__ == "__main__":
    unittest.main()
