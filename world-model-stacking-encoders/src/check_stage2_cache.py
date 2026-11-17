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
"""

import argparse
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
SOURCES = ["dinov3", "siglip2", "c-radiov4", "vae"]
DIMS = {"dinov3": 1024, "siglip2": 1152, "c-radiov4": 1152, "vae": 48}   # data/README.md section 4b
DIRS = {"vae": ["vae", "vae_b"]}                                          # proxy_wam.py VAE_DIRS order
GRID = (14, 28)
MAX_SHOWN = 15                                                            # gaps printed per source and suite


def expected_shape(source, n, dims):
    return (n, 3, *GRID, dims[source]) if source == "vae" else (n, *GRID, dims[source])


def check_episode(feats, source, suite, ep, n, dims):
    """Gap strings for one episode (empty list: fine). Reads headers (and the tiny stats file), never the arrays."""
    gaps = []
    tag = f"{source}/{suite}/ep{ep:06d}"
    found = [d for d in DIRS.get(source, [source]) if os.path.exists(os.path.join(feats, d, suite, f"ep{ep:06d}.npy"))]
    if not found:
        return [f"{tag}: missing .npy"]
    path = os.path.join(feats, found[0], suite, f"ep{ep:06d}.npy")
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as e:                                   # truncated or unreadable
        return [f"{tag}: cannot read .npy header ({type(e).__name__}: {e})"]
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


def check_cache(feats, sources=SOURCES, suites=SUITES, dims=DIMS, workers=16, out=print):
    """Returns (all gaps, per-cell counts). Prints one block per source and suite."""
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
                results = list(pool.map(lambda e: check_episode(feats, source, suite, e[0], e[1], dims), eps))
                g = [x for r in results for x in r]
                pg, plines = provenance(feats, source, suite)
                g += pg
                bad = sum(1 for r in results if r)
                out(f"{source} {suite}: {len(eps)} meta episodes, {len(eps) - bad} complete, {bad} with a gap")
                for line in plines:
                    out(line)
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
    a = p.parse_args(argv)
    sources, suites = a.sources.split(","), a.suites.split(",")
    out(f"feats={a.feats} sources={sources} suites={suites}")
    gaps, cells = check_cache(a.feats, sources, suites, DIMS, a.workers, out)
    du = shutil.disk_usage(a.feats)
    free = du.free / 2**30
    out(f"volume: {du.used / 2**30:.0f} GiB used, {free:.0f} GiB free of {du.total / 2**30:.0f} GiB (report only, no threshold)")
    out("=== summary")
    for source, suite, n, g in cells:
        out(f"{source:10s} {suite:15s} episodes {n:4d}  gaps {g}")
    out(f"=== {'FAIL' if gaps else 'OK'}: {len(gaps)} gap(s)")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
