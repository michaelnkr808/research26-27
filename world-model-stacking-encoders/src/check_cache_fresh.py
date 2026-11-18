"""Do the saved encoder feature caches still match features computed now from the pinned encoder revisions?

Stage 2 of the pretrained-start pairings (experiments/2026-11-15-pretrained-start-pairings, jobs/RUN-cache-fresh-check.md).
The DINOv3 and SigLIP 2 caches under /mnt/proxy/feats were built 2026-10-29 to 2026-10-31 from the repos' heads, before
their revisions were pinned in extract_features.REVISIONS (2026-11-17). Rollout computes features live, so if the loaded
weights ever differ from the ones that built the caches, training and rollout would see different features. This check
recomputes a few cached episodes and compares them to the .npy files.

For each suite, PER_SUITE cached episodes are chosen by a fixed seed (numpy default_rng(seed), the same episodes for
every source). Each episode's frames are decoded and resized exactly as extraction did (fastwam_data.decode_episode,
fastwam_data.to_224), and the features are recomputed with extract_fastwam.encoder_episode (same function, same batch
size, same compute dtype as the extraction Jobs, encoder from extract_features.ENCODERS with the pinned revision), then
rounded to float16 as stored. Compared per episode with the cached (n, 14, 28, D) float16 array:
  cos_min   the lowest per-frame cosine similarity (each frame's grid flattened)
  rel_max   max |fresh - cached| / max |cached| over the episode
An episode passes when cos_min >= COS_MIN and rel_max <= REL_MAX. The tolerances were fixed in RUN-cache-fresh-check.md
before any run, with their reasoning; do not change them after seeing results.

It also checks each source's provenance files (<feats>/<source>/<suite>/_meta.shard*.json): model id and compute dtype
must match what is recomputed here.

Writes nothing under --feats or --data-root. Prints one line per episode and a JSON summary; --out-json writes the
summary to a file as well (use a path off the PVC).

Exit codes: 0 every episode passes; 3 an episode fails the tolerance, a cached file is missing or has the wrong shape,
or provenance or the loaded revision does not match; 2 the script itself failed (an exception inside the check, bad
arguments). 1 is not used by the script, so a 1 means Python died outside main (for example an import error at startup).

Example (cluster):
  python check_cache_fresh.py --data-root /mnt/proxy/data/lerobot_v30 --feats /mnt/proxy/feats --device cuda
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SOURCES = ("dinov3", "siglip2", "c-radiov4")
# Compute dtype and batch of the extraction Jobs (experiments/2026-10-27-proxy-calibration/jobs/rendered/
# job-extract-<source>.yaml: --dtype and --batch 32; data/README.md section 4b).
EXTRACT = {"dinov3": "fp32", "siglip2": "fp16", "c-radiov4": "fp16"}
BATCH = 32
PER_SUITE = 2
SEED = 0
# Pass tolerances, fixed before any run (reasoning in RUN-cache-fresh-check.md).
COS_MIN = 0.999
REL_MAX = 0.02


def suite_root(data_root, suite):
    return Path(data_root) / f"{suite}_no_noops_lerobot"


def cache_path(feats, source, suite, ep_idx):
    return Path(feats) / source / suite / f"ep{ep_idx:06d}.npy"


def choose_episodes(eps, feats, sources, suite, per_suite, rng):
    """Episodes of one suite whose cache exists for every source; per_suite of them drawn by rng (sorted)."""
    have = [e for e in eps if all(cache_path(feats, s, suite, e["episode_index"]).exists() for s in sources)]
    if len(have) < per_suite:
        return [], len(have)
    pick = rng.choice(len(have), size=per_suite, replace=False)
    return sorted((have[i] for i in pick), key=lambda e: e["episode_index"]), len(have)


def compare(fresh, cached):
    """fresh, cached: (n, 14, 28, D) float16 -> metrics dict."""
    f = fresh.reshape(len(fresh), -1).astype(np.float64)
    c = cached.reshape(len(cached), -1).astype(np.float64)
    nf, nc = np.linalg.norm(f, axis=1), np.linalg.norm(c, axis=1)
    cos = (f * c).sum(1) / np.maximum(nf * nc, 1e-30)
    scale = float(np.abs(c).max())
    diff = np.abs(f - c)
    return {"cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
            "rel_max": float(diff.max() / scale) if scale > 0 else float("inf"),
            "max_abs_diff": float(diff.max()), "max_abs_cached": scale,
            "frac_bit_equal": float((fresh == cached).mean())}


def passes(m, cos_min=COS_MIN, rel_max=REL_MAX):
    return m["cos_min"] >= cos_min and m["rel_max"] <= rel_max


def check_provenance(feats, source, suite, hf_id, dtype_name):
    """List of problems with <feats>/<source>/<suite>/_meta.shard*.json (empty: fine)."""
    metas = sorted((Path(feats) / source / suite).glob("_meta.shard*.json"))
    if not metas:
        return [f"{source}/{suite}: no _meta.shard*.json"]
    bad = []
    for p in metas:
        try:
            m = json.loads(p.read_text())
        except Exception as e:
            bad.append(f"{p}: unreadable ({type(e).__name__}: {e})")
            continue
        if m.get("model") != hf_id:
            bad.append(f"{p}: model {m.get('model')!r}, recomputing with {hf_id!r}")
        if m.get("dtype") != dtype_name:
            bad.append(f"{p}: dtype {m.get('dtype')!r}, recomputing with {dtype_name!r}")
    return bad


def run(args, load_encoder=None, read_suite=None, decode=None, encode_episode=None, log=print):
    """Returns (summary dict, ok). The loader, reader, decoder and episode encoder are injectable for tests."""
    import torch
    if load_encoder is None:
        from extract_features import ENCODERS
        load_encoder = lambda s, device, dtype: ENCODERS[s](device, dtype)
    if read_suite is None or decode is None:
        import fastwam_data as FD
        read_suite = read_suite or FD.read_suite
        decode = decode or (lambda root, ep: {c: FD.to_224(f) for c, f in FD.decode_episode(root, ep).items()})
    if encode_episode is None:
        from extract_fastwam import encoder_episode as encode_episode
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    sources = [s for s in args.sources.split(",") if s]
    suites = [s for s in args.suites.split(",") if s]
    problems, rows = [], []
    t0 = time.time()
    encs = {}
    for s in sources:
        enc = load_encoder(s, args.device, dtypes[EXTRACT[s]])
        encs[s] = enc
        log(f"[load] {s}: {enc.hf_id} revision {getattr(enc, 'revision', None)} loaded {getattr(enc, 'loaded_commit', None)} "
            f"res {enc.resolution} dtype {EXTRACT[s]} device {args.device} ({time.time() - t0:.0f}s)")
        if getattr(enc, "revision", None) is None or getattr(enc, "loaded_commit", None) != enc.revision:
            problems.append(f"{s}: revision not pinned or not what was loaded "
                            f"(asked {getattr(enc, 'revision', None)}, loaded {getattr(enc, 'loaded_commit', None)})")
        for suite in suites:
            problems += check_provenance(args.feats, s, suite, enc.hf_id, EXTRACT[s])

    rng = np.random.default_rng(args.seed)
    for suite in suites:
        root = suite_root(args.data_root, suite)
        eps = read_suite(root)
        chosen, n_have = choose_episodes(eps, args.feats, sources, suite, args.per_suite, rng)
        log(f"[suite] {suite}: {len(eps)} episodes, {n_have} cached for every source, chosen "
            f"{[e['episode_index'] for e in chosen]}")
        if len(chosen) < args.per_suite:
            problems.append(f"{suite}: only {n_have} episodes cached for every source, need {args.per_suite}")
            continue
        for ep in chosen:
            cams = decode(root, ep)
            for s in sources:
                path = cache_path(args.feats, s, suite, ep["episode_index"])
                row = {"source": s, "suite": suite, "episode": ep["episode_index"], "length": ep["length"]}
                try:
                    cached = np.load(path)
                except Exception as e:
                    problems.append(f"{path}: unreadable ({type(e).__name__}: {e})")
                    continue
                t = time.time()
                fresh = np.asarray(encode_episode(encs[s], cams, args.batch)).astype(np.float16)
                row["seconds"] = round(time.time() - t, 1)
                if cached.dtype != np.float16 or cached.shape != fresh.shape or len(fresh) != ep["length"]:
                    problems.append(f"{path}: cached {cached.dtype} {cached.shape}, recomputed {fresh.shape}, "
                                    f"episode length {ep['length']}")
                    continue
                row.update(compare(fresh, cached))
                row["pass"] = passes(row, args.cos_min, args.rel_max)
                rows.append(row)
                log(f"[ep] {s} {suite} ep{ep['episode_index']:06d} n={ep['length']}: cos_min {row['cos_min']:.6f} "
                    f"cos_median {row['cos_median']:.6f} rel_max {row['rel_max']:.2e} "
                    f"(max abs diff {row['max_abs_diff']:.3g} of {row['max_abs_cached']:.3g}) "
                    f"bit-equal {100 * row['frac_bit_equal']:.1f}% {'PASS' if row['pass'] else 'FAIL'} ({row['seconds']}s)")

    failed = [r for r in rows if not r["pass"]]
    ok = not problems and not failed and len(rows) == len(sources) * len(suites) * args.per_suite
    summary = {
        "ok": ok, "tolerance": {"cos_min": args.cos_min, "rel_max": args.rel_max},
        "seed": args.seed, "per_suite": args.per_suite, "batch": args.batch, "device": args.device,
        "torch": torch.__version__,
        "encoders": {s: {"hf_id": e.hf_id, "revision": getattr(e, "revision", None),
                         "loaded_commit": getattr(e, "loaded_commit", None), "dtype": EXTRACT[s]}
                     for s, e in encs.items()},
        "per_source": {s: {"episodes": sum(r["source"] == s for r in rows),
                           "failed": sum(r["source"] == s and not r["pass"] for r in rows),
                           "worst_cos_min": min((r["cos_min"] for r in rows if r["source"] == s), default=None),
                           "worst_rel_max": max((r["rel_max"] for r in rows if r["source"] == s), default=None)}
                       for s in sources},
        "problems": problems, "episodes": rows, "seconds": round(time.time() - t0, 1)}
    return summary, ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", required=True, help="folder holding <suite>_no_noops_lerobot (lerobot_v30)")
    p.add_argument("--feats", required=True, help="the feature caches (/mnt/proxy/feats)")
    p.add_argument("--sources", default=",".join(SOURCES))
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--per-suite", type=int, default=PER_SUITE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--cos-min", type=float, default=COS_MIN)
    p.add_argument("--rel-max", type=float, default=REL_MAX)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)
    for s in args.sources.split(","):
        if s not in EXTRACT:
            p.error(f"unknown source {s!r}; known: {sorted(EXTRACT)}")
    try:
        summary, ok = run(args)
    except Exception:
        traceback.print_exc()
        print("=== ERROR: the check could not run (exit 2)", flush=True)
        return 2
    text = json.dumps(summary, indent=1)
    print("=== summary\n" + text, flush=True)
    if args.out_json:
        Path(args.out_json).write_text(text)
    for line in summary["problems"]:
        print(f"PROBLEM {line}", flush=True)
    print(f"=== {'OK' if ok else 'FAIL'}: {sum(not r['pass'] for r in summary['episodes'])} episode(s) out of tolerance, "
          f"{len(summary['problems'])} problem(s)", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
