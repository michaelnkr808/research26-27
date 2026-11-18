"""Read-only pre-check of the encoder and VAE feature caches for stage 2 (pairings PLAN, "What the implementation
session must add", item 3, and "Cache check, by reading only").

For each source (dinov3, siglip2, c-radiov4, vae) and each LIBERO suite it takes the episode list and lengths from
<feats>/meta/<suite>.npz (written by extract_fastwam.py write_meta) and checks that every episode has
  <feats>/<source>/<suite>/ep<id:06d>.npy        float16, shape (length, 14, 28, D)   (vae: (length, 3, 14, 28, 48))
  <feats>/<source>/<suite>/ep<id:06d>.stats.npy  float64, shape (3, D), count row = length * 14 * 28, finite
                                                 (encoders only; the VAE cache has no stats files)
The VAE is looked up in vae, then vae_b, as proxy_wam.py does. It also prints the model, hf_id, resolution and dtype
fields of every _meta.shard*.json, and the free space of the volume. Nothing is written: files are opened read-only,
.npy arrays are memory-mapped so only the header is read, and no directory is created.

Exit code 0: no gap. 1: at least one gap (missing meta, missing, truncated or unreadable file, wrong dtype, shape or
length, a missing, unreadable or inconsistent provenance file). Free space is printed only, never a gap. A summary is printed either way.

Run: python check_stage2_cache.py --feats /mnt/proxy/feats

--values (opt-in; without it the behaviour and output above are unchanged): after the header checks every episode file
is read in full, read-only, in one sequential pass, and three more kinds of gap are reported (same GAP lines, exit 1):
  a) encoders: per channel, count, sum and sum of squares are recomputed from the .npy the way extract_fastwam.py did
     (x = arr.reshape(-1, D).astype(float64); count, x.sum(0), (x * x).sum(0)) and compared with the .stats.npy.
  b) every source (the VAE too), and the .stats.npy files: any 4096-byte page at a file offset that is a multiple of
     4096 and holds only zero bytes. The count and the first offset are reported. A refused CephFS flush leaves pages
     of zeros (LOG 2026-11-18); in real features 2048 float16 values in a row that are all exactly +0 do not occur.
     Only whole pages count: a last partial page is not scanned (a zeroed tail is caught by a) instead).
  c) every source: any non-finite value (inf or nan) in the .npy.
With --values, every episode with any gap (header or value) is also listed after the summary, uncapped, one line each:
"DAMAGED <source> <suite> <episode id>" (the GAP lines stay capped at MAX_SHOWN per block).
The value part runs on a .npy whose header parses as C-order float16 with the expected last dimension and whose file
size matches the header; the zero-page scan runs on every file that can be opened, header or not.

Tolerance for a) (fixed: RTOL_SUMSQ, RTOL_SUM). The float16 values are exact in float64 and so are their squares
(11-bit significand, squared 22 bits), so the only difference between this pass and extraction is the summation order
(this pass adds frame chunks of CHUNK_BYTES; extraction summed all rows at once). For n terms, recursive summation
in float64 is off by at most (n - 1) * 2**-53 * sum|x_i| (Higham, Accuracy and Stability, ch. 4), so the two results
differ by at most 2 * n * 1.1e-16 * sum|x_i|. The longest episodes have 505 * 14 * 28 = 197,960 rows per channel, a
bound of 4.4e-11 * sum|x_i|.
  - sum of squares: all terms are non-negative, so sum|x_i| is the value itself: a gap when
    |recomputed - stored| > RTOL_SUMSQ * stored, RTOL_SUMSQ = 1e-9, more than 20 times the worst-case bound.
  - sum: a relative tolerance on the sum itself is wrong for a channel whose mean is near zero (the sum can be close
    to 0 while the rounding error scales with sum|x_i|). sum|x_i| <= sqrt(count * sum of squares) (Cauchy-Schwarz),
    so the gap test is |recomputed - stored| > RTOL_SUM * sqrt(count * stored sum of squares), RTOL_SUM = 1e-9.
  Sensitivity: zeroing one 4096-byte page removes 2048 values, about 1.8 rows of D = 1152, so 1 or 2 values of every
  channel. A value of typical size changes its channel's sum of squares by about 1 / rows, 5e-6 for the longest
  episode, more than 1000 times RTOL_SUMSQ; only channels whose lost values are all near 0 can hide, and not all of
  them do. (Checked on a synthetic 505-frame file with one unaligned page zeroed, see the test file.)
  - count: the stats count row must equal the number of rows read.

Memory per reader thread (--values): a CHUNK_BYTES = 4 MiB read buffer, the carried partial row joined to it (about
4 MiB), its float64 copy (16 MiB, squared in place), an isfinite mask (2 MiB) and two (D,) float64 accumulators:
about 26 MiB live, whatever the episode length (measured on the Mac, synthetic files: max RSS 112 MB with 1 thread,
434 MB with 8, so about 45 MB per thread with allocator slack on top of about 70 MB for Python and numpy) (a 505-frame SigLIP 2 file is 505 x 14 x 28 x 1152 float16 = 456 MB). The
reads go through the page cache; on Linux each file's cached pages are dropped after reading (posix_fadvise
DONTNEED, a hint, nothing is written). Bytes read and seconds taken are printed at the end.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
SOURCES = ["dinov3", "siglip2", "c-radiov4", "vae"]
DIMS = {"dinov3": 1024, "siglip2": 1152, "c-radiov4": 1152, "vae": 48}   # data/README.md section 4b
DIRS = {"vae": ["vae", "vae_b"]}                                          # proxy_wam.py VAE_DIRS order
GRID = (14, 28)
MAX_SHOWN = 15                                                            # gaps printed per source and suite
PAGE = 4096                                                               # zero-page scan unit and alignment (--values)
CHUNK_BYTES = 1024 * PAGE                                                 # 4 MiB read per step (--values)
RTOL_SUMSQ = 1e-9                                                         # docstring, "Tolerance for a)"
RTOL_SUM = 1e-9


def expected_shape(source, n, dims):
    return (n, 3, *GRID, dims[source]) if source == "vae" else (n, *GRID, dims[source])


def scan_file(path, data=None):
    """One sequential read of a file in CHUNK_BYTES steps (chunk starts are multiples of PAGE). Returns a dict with
    bytes read, the number of all-zero whole pages and the first one's offset. data = (offset, D): also accumulate,
    over the float16 rows from offset on, the row count, non-finite count, per-channel sum and sum of squares."""
    r = {"bytes": 0, "zero_pages": 0, "first_zero": None}
    if data is not None:
        off, d = data
        rowb = 2 * d
        s, q, rows, bad, carry = np.zeros(d), np.zeros(d), 0, 0, b""
    with open(path, "rb") as f:
        pos = 0
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                break
            nfull = len(buf) // PAGE
            if nfull:
                pages = np.frombuffer(buf, np.uint64, count=nfull * PAGE // 8).reshape(nfull, PAGE // 8)
                z = np.flatnonzero(~pages.any(1))
                if z.size:
                    if r["first_zero"] is None:
                        r["first_zero"] = pos + int(z[0]) * PAGE
                    r["zero_pages"] += int(z.size)
            if data is not None and pos + len(buf) > off:
                chunk = buf[max(0, off - pos):]
                if carry:
                    chunk = carry + chunk
                k = len(chunk) // rowb
                if k:
                    x = np.frombuffer(chunk, np.float16, count=k * d).reshape(k, d)
                    bad += x.size - int(np.count_nonzero(np.isfinite(x)))
                    x64 = x.astype(np.float64)
                    s += x64.sum(0)
                    np.multiply(x64, x64, out=x64)
                    q += x64.sum(0)
                    rows += k
                carry = chunk[k * rowb:]
            pos += len(buf)
        if hasattr(os, "posix_fadvise"):                     # Linux: drop this file's cached pages (a hint only)
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    r["bytes"] = pos
    if data is not None:
        r.update(rows=rows, nonfinite=bad, sum=s, sumsq=q)
    return r


def compare_stats(tag, stats, r):
    """Gap strings from comparing a (3, D) stats array with scan_file's accumulators (docstring, "Tolerance for a)")."""
    gaps = []
    if not (stats[0] == r["rows"]).all():
        gaps.append(f"{tag}: .stats.npy count {stats[0][0]:.0f} but the .npy has {r['rows']} rows per channel")
    sq = stats[2]
    dq = np.abs(r["sumsq"] - sq)
    badq = dq > RTOL_SUMSQ * np.abs(sq)
    if badq.any():
        rel = float(np.max(dq[badq] / np.maximum(np.abs(sq[badq]), np.finfo(np.float64).tiny)))
        gaps.append(f"{tag}: .stats.npy sum of squares differs from the data in {int(badq.sum())} of {len(sq)} "
                    f"channel(s) (largest relative difference {rel:.2e}, tolerance {RTOL_SUMSQ:g})")
    scale = np.sqrt(np.maximum(r["rows"] * np.maximum(sq, 0.0), 0.0))
    ds = np.abs(r["sum"] - stats[1])
    bads = ds > RTOL_SUM * scale
    if bads.any():
        rel = float(np.max(ds[bads] / np.maximum(scale[bads], np.finfo(np.float64).tiny)))
        gaps.append(f"{tag}: .stats.npy sum differs from the data in {int(bads.sum())} of {len(sq)} channel(s) "
                    f"(largest difference over sqrt(count * sum of squares) {rel:.2e}, tolerance {RTOL_SUM:g})")
    return gaps


def value_gaps(tag, path, stats_path, source, dims, tally):
    """Gap strings of the --values checks for one episode; adds bytes read and the value-checked count to tally."""
    gaps, data = [], None
    try:
        a = np.load(path, mmap_mode="r")                     # header only; fails on a file shorter than its header says
        if (a.dtype == np.float16 and a.ndim >= 2 and a.shape[-1] == dims[source] and a.flags.c_contiguous
                and os.path.getsize(path) == a.offset + a.nbytes):
            data = (a.offset, dims[source])
        del a
    except Exception:
        pass                                                 # the header check has reported it; still scan for zeros
    nbytes = 0
    try:
        r = scan_file(path, data)
        nbytes += r["bytes"]
        if r["zero_pages"]:
            gaps.append(f"{tag}: {r['zero_pages']} all-zero {PAGE}-byte page(s) in .npy, first at offset {r['first_zero']}")
    except Exception as e:
        r, data = None, None
        gaps.append(f"{tag}: cannot read .npy for the value check ({type(e).__name__}: {e})")
    if data is not None and r["nonfinite"]:
        gaps.append(f"{tag}: {r['nonfinite']} non-finite value(s) in .npy")
    if stats_path is not None and os.path.exists(stats_path):
        try:
            rs = scan_file(stats_path)
            nbytes += rs["bytes"]
            if rs["zero_pages"]:
                gaps.append(f"{tag}: {rs['zero_pages']} all-zero {PAGE}-byte page(s) in .stats.npy, "
                            f"first at offset {rs['first_zero']}")
        except Exception as e:
            gaps.append(f"{tag}: cannot read .stats.npy for the value check ({type(e).__name__}: {e})")
        if data is not None and not r["nonfinite"]:
            try:
                s = np.load(stats_path)
                ok = s.shape == (3, dims[source]) and s.dtype == np.float64 and np.isfinite(s).all()
            except Exception:
                ok = False                                   # the header check has reported it
            if ok:
                gaps += compare_stats(tag, s, r)
    with tally["lock"]:
        tally["bytes"] += nbytes
        if data is not None:
            tally["checked"][tag.rsplit("/", 1)[0]] = tally["checked"].get(tag.rsplit("/", 1)[0], 0) + 1
    return gaps


def new_tally():
    return {"lock": threading.Lock(), "bytes": 0, "checked": {}, "t0": time.time()}


def check_episode(feats, source, suite, ep, n, dims, values=False, tally=None):
    """Gap strings for one episode (empty list: fine). Reads headers (and the tiny stats file), never the arrays;
    with values=True it then reads every file in full (value_gaps)."""
    gaps = []
    tag = f"{source}/{suite}/ep{ep:06d}"
    found = [d for d in DIRS.get(source, [source]) if os.path.exists(os.path.join(feats, d, suite, f"ep{ep:06d}.npy"))]
    if not found:
        return [f"{tag}: missing .npy"]
    path = os.path.join(feats, found[0], suite, f"ep{ep:06d}.npy")
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as e:                                   # truncated or unreadable
        gaps = [f"{tag}: cannot read .npy header ({type(e).__name__}: {e})"]
        if values:                                           # the zero-page scan still runs, on the stats file too
            sp = None if source == "vae" else os.path.join(feats, found[0], suite, f"ep{ep:06d}.stats.npy")
            gaps += value_gaps(tag, path, sp, source, dims, tally)
        return gaps
    want = expected_shape(source, n, dims)
    if a.dtype != np.float16:
        gaps.append(f"{tag}: dtype {a.dtype}, want float16")
    if a.shape != want:
        kind = "length" if a.ndim == len(want) and a.shape[0] != n and a.shape[1:] == want[1:] else "shape"
        gaps.append(f"{tag}: wrong {kind}, got {a.shape}, want {want}")
    if source != "vae":
        sp = os.path.join(feats, found[0], suite, f"ep{ep:06d}.stats.npy")
        if not os.path.exists(sp):
            gaps.append(f"{tag}: missing .stats.npy")
        else:
            try:
                s = np.load(sp)
                if s.shape != (3, dims[source]) or s.dtype != np.float64:
                    gaps.append(f"{tag}: stats {s.dtype} {s.shape}, want float64 {(3, dims[source])}")
                elif not np.isfinite(s).all():
                    gaps.append(f"{tag}: stats not finite")
                elif not (s[0] == n * GRID[0] * GRID[1]).all():
                    gaps.append(f"{tag}: stats count {s[0][0]:.0f}, want {n * GRID[0] * GRID[1]}")
            except Exception as e:
                gaps.append(f"{tag}: cannot read .stats.npy ({type(e).__name__}: {e})")
    if values:
        gaps += value_gaps(tag, path, None if source == "vae" else sp, source, dims, tally)
    return gaps


def read_meta(feats, suite):
    path = os.path.join(feats, "meta", f"{suite}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as m:
        ids, lengths = m["episode_ids"].tolist(), m["lengths"].tolist()
    return list(zip(ids, lengths))


def provenance(feats, source, suite):
    """(gaps, lines) from the _meta.shard*.json files of every folder the source may live in."""
    files = sorted(f for d in DIRS.get(source, [source]) for f in glob.glob(os.path.join(feats, d, suite, "_meta.shard*.json")))
    if not files:
        return ([] if source == "vae" else [f"{source}/{suite}: no _meta.shard*.json"]), [f"  provenance: none found"]
    seen = {}
    bad = []
    for f in files:
        try:
            with open(f) as fh:
                j = json.load(fh)
            if not isinstance(j, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            bad.append(f"{source}/{suite}: cannot read {os.path.relpath(f, feats)} ({type(e).__name__}: {e})")
            continue
        key = (j.get("model"), j.get("hf_id"), str(j.get("resolution")), j.get("dtype"))
        seen.setdefault(key, []).append(os.path.relpath(f, feats))
    lines = [f"  provenance: model={k[0]} hf_id={k[1]} resolution={k[2]} compute_dtype={k[3]} ({len(v)} file(s))"
             for k, v in seen.items()]
    gaps = bad + ([f"{source}/{suite}: {len(seen)} different model/resolution/dtype values across shards"]
                  if len(seen) > 1 else [])
    return gaps, lines or ["  provenance: none readable"]


def check_cache(feats, sources=SOURCES, suites=SUITES, dims=DIMS, workers=16, out=print, values=False, tally=None):
    """Returns (all gaps, per-cell counts). Prints one block per source and suite. values=True adds the --values
    checks and a "values:" line per block; bytes read go to tally (new_tally())."""
    if values and tally is None:
        tally = new_tally()
    gaps_all, cells = [], []
    meta = {s: read_meta(feats, s) for s in suites}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for source in sources:
            for suite in suites:
                eps = meta[suite]
                if eps is None:
                    g = [f"{source}/{suite}: meta/{suite}.npz missing"]
                    out(f"{source} {suite}: no meta file\n  GAP {g[0]}")
                    gaps_all += g
                    cells.append((source, suite, 0, len(g)))
                    continue
                results = list(pool.map(lambda e: check_episode(feats, source, suite, e[0], e[1], dims, values, tally), eps))
                g = [x for r in results for x in r]
                if values:                                   # uncapped list for a repair Job (main prints DAMAGED lines)
                    tally.setdefault("damaged", []).extend((source, suite, e[0]) for e, r in zip(eps, results) if r)
                pg, plines = provenance(feats, source, suite)
                g += pg
                bad = sum(1 for r in results if r)
                out(f"{source} {suite}: {len(eps)} meta episodes, {len(eps) - bad} complete, {bad} with a gap")
                for line in plines:
                    out(line)
                if values:
                    out(f"  values: {tally['checked'].get(f'{source}/{suite}', 0)} of {len(eps)} episodes value-checked")
                for x in g[:MAX_SHOWN]:
                    out(f"  GAP {x}")
                if len(g) > MAX_SHOWN:
                    out(f"  ... {len(g) - MAX_SHOWN} more gaps in {source} {suite}")
                gaps_all += g
                cells.append((source, suite, len(eps), len(g)))
    return gaps_all, cells


def main(argv=None, out=print):
    p = argparse.ArgumentParser()
    p.add_argument("--feats", default="/mnt/proxy/feats")
    p.add_argument("--sources", default=",".join(SOURCES))
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--values", action="store_true",
                   help="also read every file in full: data against the .stats.npy, all-zero 4096-byte pages, "
                        "non-finite values (see the module docstring)")
    a = p.parse_args(argv)
    sources, suites = a.sources.split(","), a.suites.split(",")
    out(f"feats={a.feats} sources={sources} suites={suites}" + (" values=on" if a.values else ""))
    tally = new_tally() if a.values else None
    gaps, cells = check_cache(a.feats, sources, suites, DIMS, a.workers, out, a.values, tally)
    du = shutil.disk_usage(a.feats)
    free = du.free / 2**30
    out(f"volume: {du.used / 2**30:.0f} GiB used, {free:.0f} GiB free of {du.total / 2**30:.0f} GiB (report only, no threshold)")
    out("=== summary")
    for source, suite, n, g in cells:
        out(f"{source:10s} {suite:15s} episodes {n:4d}  gaps {g}")
    if a.values:
        sec = time.time() - tally["t0"]
        out(f"values: {sum(tally['checked'].values())} episodes value-checked, {tally['bytes'] / 1e9:.2f} GB read "
            f"in {sec:.0f} s ({tally['bytes'] / 1e6 / max(sec, 1e-9):.0f} MB/s, {a.workers} reader threads)")
        dmg = tally.get("damaged", [])
        out(f"=== damaged episodes (every episode with a gap of any kind, uncapped): {len(dmg)}")
        for source, suite, ep in dmg:
            out(f"DAMAGED {source} {suite} {ep}")
    out(f"=== {'FAIL' if gaps else 'OK'}: {len(gaps)} gap(s)")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
