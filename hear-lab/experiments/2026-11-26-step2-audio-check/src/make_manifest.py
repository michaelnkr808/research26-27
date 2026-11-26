#!/usr/bin/env python3
"""sha256 manifest of every file of a run folder, committed or not (hear-lab STANDARDS, "Results storage").

  python make_manifest.py <run-dir>            # writes <run-dir>/manifest.sha256
  python make_manifest.py <run-dir> --check    # verifies every listed file (exit 1 on any difference)

Line format: "<sha256>  <bytes>  <path relative to the run folder>", sorted by path; the manifest itself is not
listed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

NAME = "manifest.sha256"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(run: Path) -> str:
    files = sorted(p for p in run.rglob("*") if p.is_file() and p.name != NAME and p.name != ".DS_Store")
    return "".join(f"{sha256(p)}  {p.stat().st_size}  {p.relative_to(run).as_posix()}\n" for p in files)


def check(run: Path) -> list:
    bad = []
    listed = set()
    for line in (run / NAME).read_text(encoding="utf-8").splitlines():
        h, size, rel = line.split("  ", 2)
        listed.add(rel)
        p = run / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif p.stat().st_size != int(size) or sha256(p) != h:
            bad.append(f"differs {rel}")
    extra = [p.relative_to(run).as_posix() for p in run.rglob("*")
             if p.is_file() and p.name not in (NAME, ".DS_Store") and p.relative_to(run).as_posix() not in listed]
    bad += [f"not listed {x}" for x in sorted(extra)]
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = check(a.run_dir)
        print("\n".join(bad) if bad else "manifest OK")
        return 1 if bad else 0
    (a.run_dir / NAME).write_text(build(a.run_dir), encoding="utf-8")
    print(f"wrote {a.run_dir / NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
