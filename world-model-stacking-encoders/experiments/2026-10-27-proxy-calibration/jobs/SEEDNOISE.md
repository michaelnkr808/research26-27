# Seed-noise read Job

Manifest: `job-seednoise-read.yaml` (Job `stacking-encoders-proxy-seednoise-read`, namespace `aria-world`). Written 2026-11-09, not submitted until the lead creates the ConfigMap and submits (user go 2026-11-09: "run the seednoise").

## What it does

CPU-only, read-only. It reads the 16 finished full-benchmark record files on the PVC (`/mnt/proxy/rollouts/eval-dinov3-s1/` = training seed 1 = A, `/mnt/proxy/rollouts/eval-dinov3-s0/` = training seed 0 = B) and, in order: prints date, sha256 of the two mounted scripts, nproc; runs `test_lp_eval.py` (must print "all checks passed"); gunzips the full subset and checks its sha256; prints `SHA256` and `LINES` for every `*of8.jsonl` plus any `.errors.jsonl` line counts (expected 3767 lines for shards 0 and 1, 3766 for shards 2 to 7, per seed); runs `lp_eval.py seednoise` (config `dinov3`) and prints its JSON between `=== BEGIN SEEDNOISE_JSON` and `=== END SEEDNOISE_JSON`; prints each record file as `BEGIN_GZ_B64 <name>` / gzip+base64 / `END_GZ_B64 <name>` (also if seednoise failed); ends with `=== done worst <code>`. Each stage prints `=== stage N exit C`; the Job exits with the worst code.

## ConfigMap (create before submitting)

The cluster ConfigMap `stacking-encoders-proxy-src` has an old `lp_eval.py` without the `seednoise` command, so this Job mounts a new one at `/lp`. Run from the repo root:

```
kubectl create configmap stacking-encoders-proxy-lpeval -n aria-world --request-timeout=30s --from-file=lp_eval.py=world-model-stacking-encoders/src/lp_eval.py --from-file=test_lp_eval.py=world-model-stacking-encoders/src/test_lp_eval.py
```

The log's first lines print the sha256 of both files; compare with `shasum -a 256` of the repo copies (`lp_eval.py` should start `6f7ed893bc82`).

## Recovering the files from the log

Save the pod log (`kubectl logs <pod> -n aria-world --request-timeout=60s > seednoise.log`), then:

```
python3 - <<'EOF'
import re, base64, gzip, hashlib, os
log = open("seednoise.log").read()
out = "world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/results/full-benchmark/rollouts"
os.makedirs(out, exist_ok=True)
for name, b64 in re.findall(r"BEGIN_GZ_B64 (\S+)\n(.*?)\nEND_GZ_B64 \1", log, re.S):
    data = gzip.decompress(base64.b64decode(b64))
    open(os.path.join(out, name), "wb").write(data)
    print(name, len(data.splitlines()), hashlib.sha256(data).hexdigest())
EOF
```

Check each printed line count and sha256 against the `LINES` and `SHA256` lines in the log. Note: both seeds' directories are separate on the PVC but file names differ (`dinov3-s1.Kof8.jsonl`, `dinov3-s0-a40.Kof8.jsonl`), so none collide.

Where things go (under `results/full-benchmark/` of this experiment):
- record files: `results/full-benchmark/rollouts/`
- seed-noise JSON: copy the text between `=== BEGIN SEEDNOISE_JSON` and `=== END SEEDNOISE_JSON` to `results/full-benchmark/seednoise.json`.

## Caveat

Read-only: the PVC is mounted `readOnly: true` and nothing is written to it; scratch output goes to an `emptyDir` at `/work` that disappears with the pod. The pod log is the only record, so save it within the 24 h TTL. Kubernetes log size limits could truncate the base64 output of 16 files; check every file's line count and sha256 after recovery, and re-read any shard that fails (a smaller read-Job) rather than trusting it.
