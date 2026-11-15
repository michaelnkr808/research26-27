"""Copy a small, self-contained slice of the proxy's feature cache, for the pretrained-start sizing check
(experiments/2026-11-12-pretrained-start-sizing/PLAN.md, Step 3). Needs only numpy and the standard library, so it runs
in a plain CPU pod with the PVC `stacking-encoders-proxy` mounted at /mnt/proxy.

What it does: picks N episodes (default 16) with a seed (default 0) from the training episodes of the suites, and writes
a folder in the proxy's cache layout (extract_fastwam.py), so proxy_wam.Data(slice, suites, sources, slice/t5) reads it
unchanged:
  <out>/meta/<suite>.npz                 only the chosen episodes (same keys as the full meta, rows cut by episode)
  <out>/vae/<suite>/ep<idx>.npy          VAE latents (found in feats/vae or feats/vae_b, written into vae)
  <out>/<encoder>/<suite>/ep<idx>.npy    the chosen encoders' per-frame features
  <out>/<encoder>/<suite>/ep<idx>.stats.npy   their per-episode (count, sum, sum of squares) z-score statistics
  <out>/t5/<sha256(prompt)>.t5_len128.wan22ti2v5b.pt   one file per distinct task of the chosen episodes
  <out>/slice.json, <out>/MANIFEST.sha256   the chosen ids, settings and a sha256 per file
Files are copied byte for byte, never re-encoded.

"Training episodes": proxy_wam.Data trains on every episode listed in the full meta of the suites it is given; there is
no other split in the cache (evaluation is rollouts in the LIBERO-Plus simulator, not cached held-out episodes). So the
pool is every episode in feats/meta/<suite>.npz, exactly what proxy_wam.py's `--suites` loads.

Z-score statistics: proxy_wam.Data computes them by summing the per-episode `.stats.npy` files of the episodes it
loads. The slice carries those per-episode files unchanged, so loading the slice gives statistics of the slice's episodes
only (the accepted deviation in the sizing PLAN, Step 3), not of the whole training set. Shapes are unchanged.

Selection: the pool is the (suite, episode id) pairs in the order of --suites then the meta order; rng =
numpy default_rng(seed); n indices are drawn without replacement and the pairs sorted by (suite order, id). The same
seed and cache give the same episodes.

Example (in the pod):
  python make_sizing_slice.py --feats /mnt/proxy/feats --out /work/sizing-slice --encoders dinov3
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
VAE_DIRS = ("vae", "vae_b")        # proxy_wam.VAE_DIRS
CONTEXT_LEN, ENC_ID = 128, "wan22ti2v5b"     # t5_cache.py
# t5_cache.DEFAULT_PROMPT, copied so this script needs no torch; test_make_sizing_slice.py checks cache_path against t5_cache's
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def t5_name(task):
    """File name t5_cache.cache_path gives the task's prompt."""
    return f"{hashlib.sha256(DEFAULT_PROMPT.format(task=task).encode('utf-8')).hexdigest()}.t5_len{CONTEXT_LEN}.{ENC_ID}.pt"


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_pool(feats, suites):
    """[(suite, episode_id, length, task, offset)] for every episode in each suite's meta, in meta order."""
    pool = []
    for s in suites:
        m = np.load(Path(feats) / "meta" / f"{s}.npz")
        off = 0
        for e, n, task in zip(m["episode_ids"], m["lengths"], m["tasks"]):
            pool.append((s, int(e), int(n), str(task), off))
            off += int(n)
        assert off == len(m["action"]), (s, off, len(m["action"]))
    return pool


def choose(pool, n, seed, suites):
    """n episodes drawn without replacement with default_rng(seed), sorted by (suite order, episode id)."""
    assert 0 < n <= len(pool), (n, len(pool))
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    order = {s: i for i, s in enumerate(suites)}
    return sorted((pool[i] for i in idx), key=lambda r: (order[r[0]], r[1]))


def vae_source(feats, suite, ep):
    found = [Path(feats) / d / suite / f"ep{ep:06d}.npy" for d in VAE_DIRS]
    found = [p for p in found if p.exists()]
    assert found, f"no VAE latents for {suite} episode {ep} in {VAE_DIRS}"
    return found[0]


def write_meta(feats, out, chosen, suites):
    """One npz per suite that has chosen episodes, with the full meta's keys and only those episodes' rows."""
    for s in suites:
        mine = [r for r in chosen if r[0] == s]
        if not mine:
            continue
        m = np.load(Path(feats) / "meta" / f"{s}.npz")
        rows = lambda key: np.concatenate([m[key][o:o + n] for _, _, n, _, o in mine])
        dst = Path(out) / "meta" / f"{s}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, action=rows("action"), state=rows("state"), episode_index=rows("episode_index"),
                 episode_ids=np.array([e for _, e, _, _, _ in mine]), lengths=np.array([n for _, _, n, _, _ in mine]),
                 tasks=np.array([t for _, _, _, t, _ in mine]))


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def make_slice(feats, out, n=16, seed=0, encoders=("dinov3",), suites=SUITES, log=print):
    """Build the slice; returns the info dict also written to <out>/slice.json."""
    feats, out = Path(feats), Path(out)
    assert not out.exists() or not any(out.iterdir()), f"{out} exists and is not empty"
    pool = read_pool(feats, suites)
    chosen = choose(pool, n, seed, suites)
    # check every source file exists before copying anything
    for s, e, _, _, _ in chosen:
        vae_source(feats, s, e)
        for enc in encoders:
            for suffix in (".npy", ".stats.npy"):
                p = feats / enc / s / f"ep{e:06d}{suffix}"
                assert p.exists(), f"missing {p}"
    tasks = sorted({t for _, _, _, t, _ in chosen})
    for t in tasks:
        assert (feats / "t5" / t5_name(t)).exists(), f"missing T5 file for task {t!r}"
    write_meta(feats, out, chosen, suites)
    for s, e, _, _, _ in chosen:
        copy(vae_source(feats, s, e), out / "vae" / s / f"ep{e:06d}.npy")
        for enc in encoders:
            for suffix in (".npy", ".stats.npy"):
                copy(feats / enc / s / f"ep{e:06d}{suffix}", out / enc / s / f"ep{e:06d}{suffix}")
    for t in tasks:
        copy(feats / "t5" / t5_name(t), out / "t5" / t5_name(t))
    used = [s for s in suites if any(r[0] == s for r in chosen)]
    info = {"seed": seed, "n_episodes": n, "encoders": list(encoders), "suites_in_slice": used,
            "pool_size": len(pool), "episodes": [{"suite": s, "episode_id": e, "length": ln, "task": t}
                                                 for s, e, ln, t, _ in chosen],
            "frames": int(sum(r[2] for r in chosen)), "tasks": tasks,
            "stats": "per-episode .stats.npy copied unchanged; proxy_wam.Data sums them over the slice's episodes, so "
                     "z-score statistics are the slice's own, not the training set's (accepted deviation, PLAN Step 3)",
            "load_with": f"proxy_wam.Data(<slice>, {used}, {list(encoders)}, <slice>/t5)"}
    (out / "slice.json").write_text(json.dumps(info, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    lines = [f"{sha256_file(p)}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    size = sum(p.stat().st_size for p in files)
    info["bytes"] = size
    log(f"chosen episodes (suite, id, length): {[(s, e, ln) for s, e, ln, _, _ in chosen]}")
    log(f"suites in slice: {used}; frames {info['frames']}; tasks {len(tasks)}; files {len(files)}; "
        f"size {size / 1e9:.3f} GB ({size / 2**30:.3f} GiB)")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="the proxy's cache root, e.g. /mnt/proxy/feats")
    p.add_argument("--out", required=True, help="empty or new folder for the slice")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoders", default="dinov3", help="comma-separated encoder folders to copy")
    p.add_argument("--suites", default=",".join(SUITES))
    a = p.parse_args()
    make_slice(a.feats, a.out, a.n, a.seed, tuple(a.encoders.split(",")), tuple(a.suites.split(",")))


if __name__ == "__main__":
    main()
