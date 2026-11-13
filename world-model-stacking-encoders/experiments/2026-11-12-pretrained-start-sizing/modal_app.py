"""Modal app for the pretrained-start sizing check (PLAN.md Steps 4-6, one L40S). Steps and commands: RUN.md.

Nothing here runs on import. Each step is a separate `modal run` the user starts, after the go in chat:
  stage    CPU only: download from Hugging Face `Wan-AI/Wan2.2-TI2V-5B` (Apache-2.0) the index, config.json and only the
           DiT shards that hold the kept blocks and the non-block tensors (shard 1 for blocks 0-7, step2-reading.md Q7),
           check sizes, write sha256 and the resolved revision to a manifest on the volume.
  preflight CPU only, on the run image: imports Fast-WAM, opens the data slice for each config, checks the Wan files
           and the free local disk, so a broken image or volume is found before any GPU time.
  run      one L40S, function timeout 5,400 s (the PLAN's 90-minute outer limit on the function), the launcher under
           `timeout --signal=TERM --kill-after=60 3600` (the 60-minute inner limit); results and run.log are written to
           the volume as each process ends and committed every 2 minutes; then fetches them.
  fetch    downloads results/<run-id>/ from the volume into this folder's results/<run-id>/, never overwriting a file.

Volume `stacking-sizing` layout (mounted at /vol):
  data/sizing-slice/{meta,vae,vae_b,dinov3,t5}   the user's 16-episode slice, uploaded with `modal volume put`
  wan/Wan2.2-TI2V-5B/{config.json, diffusion_pytorch_model.safetensors.index.json, <needed shards>, manifest.json}
  results/<run-id>/{measure-*.json, load-*.json, sanity-*.json, rows/, run.log}
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal
from modal.volume import FileEntryType

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
REQS = HERE.parents[1] / "experiments" / "2026-10-27-proxy-calibration" / "requirements.txt"

VOLUME_NAME = "stacking-sizing"
MOUNT = "/vol"
SLICE_DIR = f"{MOUNT}/data/sizing-slice"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_DIR = f"{MOUNT}/wan/Wan2.2-TI2V-5B"
RESULTS = "results"                       # on the volume, and under this folder locally
STATE_DIR = "/tmp/sizing-state"           # container-local disk for the saved model + AdamW state
KEEP = "0-7"                              # the user's choice (2026-11-12): first 8 blocks
CONFIGS = "none,dinov3"                   # the user's choice: DINOv3 (1,024 channels) as the extra encoder
GPU = "L40S"
OUTER_TIMEOUT_S = 5400                    # PLAN "Cost and time cap", outer limit (Modal function timeout)
INNER_TIMEOUT_S = 3600                    # inner limit, on the launcher
STATE_DISK_MIN_GIB = 40                   # ~1.7B params x 12 bytes (fp32 weights + 2 AdamW moments) ~ 20 GB, x 2 margin

# Fast-WAM 7faa711 (MIT), the same pinned tarball and sha256 as the proxy Jobs' pip freeze
# (experiments/2026-10-27-proxy-calibration/results/logs/*.log) and src/setup_env.sh
FASTWAM_URL = ("https://github.com/yuantianyuan01/FastWAM/archive/7faa71108368fbb3b6885649f112af607427a2d4.tar.gz"
               "#sha256=43102a7c1d207f0a4a691991b6562d8b2dcfbc09b6b22b5a52ecc7c6b6f81520")
# Sizes of the Wan DiT files on Hugging Face, read 2026-11-12 (step2-reading.md Q1)
KNOWN_SIZES = {"diffusion_pytorch_model-00001-of-00003.safetensors": 9_825_014_472,
               "diffusion_pytorch_model-00002-of-00003.safetensors": 9_995_661_736,
               "diffusion_pytorch_model-00003-of-00003.safetensors": 178_558_176}
INDEX = "diffusion_pytorch_model.safetensors.index.json"
SRC_FILES = ("pretrained_sizing.py", "proxy_wam.py", "diagnostics.py", "t5_cache.py")


def _pins():
    """Pins from the proxy's requirements.txt (one source of truth), split into the torch wheels (CUDA 12.8 build, as
    Fast-WAM pins) and the rest. Read only on the user's machine; the container never rebuilds the image."""
    if not modal.is_local():
        return [], []
    lines = [ln.strip() for ln in REQS.read_text().splitlines()]
    pins = [ln for ln in lines if ln and not ln.startswith("#")]
    torch_pins = [p for p in pins if p.split("==")[0] in ("torch", "torchvision")]
    return torch_pins, [p for p in pins if p not in torch_pins]


TORCH_PINS, OTHER_PINS = _pins()

run_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*TORCH_PINS, index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(*OTHER_PINS)
    .pip_install(FASTWAM_URL, extra_options="--no-deps")
    .env({"PYTHONUNBUFFERED": "1"})
)
for _f in SRC_FILES:
    run_image = run_image.add_local_file(SRC / _f, f"/root/src/{_f}")

hf_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==0.36.2")  # requirements.txt pin

app = modal.App("stacking-encoders-sizing")
vol = modal.Volume.from_name(VOLUME_NAME)      # created by the user (RUN.md step 2); never created implicitly


def _keep_ids(keep):
    out = []
    for part in keep.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _needed_shards(weight_map, keep):
    ids = set(_keep_ids(keep))
    return sorted({sh for n, sh in weight_map.items()
                   if not n.startswith("blocks.") or int(n.split(".")[1]) in ids})


def _sha256(path, chunk=64 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@app.function(image=hf_image, volumes={MOUNT: vol}, cpu=2, memory=4096, timeout=3600)
def stage_weights(keep: str = KEEP, revision: str = "main"):
    """CPU only. Download the Wan files the kept blocks need into the volume, check sizes, write manifest.json."""
    from huggingface_hub import HfApi, hf_hub_download
    sha = HfApi().model_info(WAN_REPO, revision=revision).sha
    dst = Path(WAN_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in (INDEX, "config.json"):
        hf_hub_download(WAN_REPO, f, revision=sha, local_dir=dst)
    wm = json.loads((dst / INDEX).read_text())["weight_map"]
    shards = _needed_shards(wm, keep)
    print(f"revision {sha}; keep {keep} needs {shards} of {sorted(set(wm.values()))}", flush=True)
    for sh in shards:
        hf_hub_download(WAN_REPO, sh, revision=sha, local_dir=dst)
        print(f"downloaded {sh} ({(dst / sh).stat().st_size} bytes) at {time.time() - t0:.0f}s", flush=True)
    files = {}
    for f in [INDEX, "config.json"] + shards:
        size = (dst / f).stat().st_size
        if f in KNOWN_SIZES and size != KNOWN_SIZES[f]:
            raise RuntimeError(f"{f}: {size} bytes, expected {KNOWN_SIZES[f]} (step2-reading.md Q1)")
        files[f] = {"bytes": size, "sha256": _sha256(dst / f)}
    manifest = {"repo": WAN_REPO, "revision": sha, "requested_revision": revision, "license": "apache-2.0",
                "keep": keep, "shards": shards, "files": files, "staged_unix": time.time(),
                "download_s": time.time() - t0}
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    vol.commit()
    print(json.dumps(manifest, indent=1))
    return manifest


@app.function(image=run_image, volumes={MOUNT: vol}, cpu=4, memory=32768, timeout=1200)
def preflight(configs: str = CONFIGS, keep: str = KEEP):
    """CPU only, on the run image: Fast-WAM imports, the slice opens for each config, the Wan files are there."""
    import shutil
    sys.path.insert(0, "/root/src")
    import pretrained_sizing as S
    report = {"fastwam_commit": S.FASTWAM_COMMIT}
    wan = Path(WAN_DIR)
    manifest = json.loads((wan / "manifest.json").read_text())
    wm = json.loads((wan / INDEX).read_text())["weight_map"]
    missing = [sh for sh in _needed_shards(wm, keep) if not (wan / sh).exists()]
    report["wan"] = {"revision": manifest["revision"], "shards_needed": _needed_shards(wm, keep), "missing": missing}
    for c in configs.split(","):
        data, suites = S.load_data(SLICE_DIR, c)
        report[c] = {"suites": suites, "episodes": len(data.episodes), "frames": int(sum(n for _, _, n, _ in data.episodes)),
                     "dims": data.dims, "tasks": len(data.tasks)}
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    report["local_disk_free_gib"] = shutil.disk_usage(STATE_DIR).free / 2**30
    print(json.dumps(report, indent=1))
    if missing:
        raise RuntimeError(f"Wan shards missing on the volume: {missing}")
    return report


def _committer(stop, every=120):
    while not stop.wait(every):
        try:
            vol.commit()
        except Exception as e:          # noqa: BLE001 - a failed periodic commit must not kill the run
            print(f"[modal] periodic volume commit failed: {e!r}", flush=True)


@app.function(image=run_image, gpu=GPU, volumes={MOUNT: vol}, cpu=8, memory=49152, timeout=OUTER_TIMEOUT_S)
def run_sizing(run_id: str, repo_commit: str, configs: str = CONFIGS, keep: str = KEEP):
    """One L40S. Runs the launcher under the inner limit, writing results/<run_id>/ on the volume as it goes."""
    import shutil
    out = Path(MOUNT) / RESULTS / run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds results; use a new run id (never overwrite results)")
    out.mkdir(parents=True, exist_ok=True)
    Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STATE_DIR).free / 2**30
    log = open(out / "run.log", "w", buffering=1)

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    say(f"[modal] run {run_id} repo {repo_commit} configs {configs} keep {keep} start {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    say(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    say(f"[modal] local disk free for the saved state: {free:.0f} GiB")
    if free < STATE_DISK_MIN_GIB:
        say(f"[modal] stopping: needs >= {STATE_DISK_MIN_GIB} GiB")
        log.close()
        vol.commit()
        raise RuntimeError("not enough local disk for the saved state")
    cmd = ["timeout", "--signal=TERM", "--kill-after=60", str(INNER_TIMEOUT_S), sys.executable,
           "/root/src/pretrained_sizing.py", "launch", "--feats", SLICE_DIR, "--wan-dir", WAN_DIR, "--configs", configs,
           "--keep", keep, "--out", str(out), "--state-dir", STATE_DIR, "--device", "cuda", "--seed", "0"]
    say("[modal] " + " ".join(cmd))
    stop = threading.Event()
    th = threading.Thread(target=_committer, args=(stop,), daemon=True)
    th.start()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/root/src",
                            env={**os.environ, "SRC_GIT_COMMIT": repo_commit})
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = proc.wait()
    stop.set()
    say(f"[modal] launcher exit code {rc}{' (inner time limit)' if rc == 124 else ''} after {time.time() - t0:.0f}s")
    log.close()
    vol.commit()
    return rc


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=HERE).stdout.strip()


@app.local_entrypoint()
def stage(keep: str = KEEP, revision: str = "main"):
    print(json.dumps(stage_weights.remote(keep, revision), indent=1))


@app.local_entrypoint()
def check(configs: str = CONFIGS, keep: str = KEEP):
    preflight.remote(configs, keep)


@app.local_entrypoint()
def run(run_id: str, configs: str = CONFIGS, keep: str = KEEP):
    """Refuses uncommitted changes in the code the container runs, so results name a real commit."""
    dirty = _git("status", "--porcelain", "--", str(SRC), str(HERE / "modal_app.py"))
    if dirty:
        raise SystemExit(f"commit these first, the results must name the code's commit:\n{dirty}")
    commit = _git("rev-parse", "HEAD")
    rc = run_sizing.remote(run_id, commit, configs, keep)
    print(f"run_sizing returned {rc}")
    _fetch(run_id)


@app.local_entrypoint()
def fetch(run_id: str):
    """Download results/<run_id>/ from the volume into this folder; files that already exist locally are kept."""
    _fetch(run_id)


def _fetch(run_id):
    base = f"{RESULTS}/{run_id}"
    n = 0
    for e in vol.iterdir(base, recursive=True):
        if e.type != FileEntryType.FILE:
            continue
        dst = HERE / e.path
        if dst.exists():
            print(f"kept existing {dst} (not overwritten)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
        n += 1
    print(f"fetched {n} files into {HERE / base}")
