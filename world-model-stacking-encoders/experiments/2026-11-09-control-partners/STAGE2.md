# Control partners: stage 2 and fetch-draws runbook (for the lead)

Two CPU-only Jobs, written 2026-11-10, NOT submitted. The user's chat OK (2026-11-10): "sure go ahead with both cpu jobs". The lead's reading: (a) stage 2 = seeds 1 and 2 of ALL the random controls on both sides (the PLAN's "all of them ... standard 3 seeds" option), (b) fetch the stage 1 bootstrap-draw files off the volume. Per the memory rule, re-ask in the implementation session's window before applying (relayed approvals are not accepted there).

Files: `job-stage2.yaml` (Job `stacking-encoders-control-partners-s2`) and `job-fetch-draws.yaml` (Job `stacking-encoders-control-fetch-draws`).

## Pre-checks (both)

1. Main is clean and equal to origin/main, and the branch carrying these files is merged (chore, direct merge) or the files are copied from it.
2. The ConfigMap `stacking-encoders-control-src` still exists and its two keys have the sha256 recorded in `LOG.md` (2026-11-10 stage 1 submission). Read them back from the cluster and hash them:
   ```
   kubectl get configmap stacking-encoders-control-src -n aria-world -o json | /opt/homebrew/bin/python3.11 -I -c "import sys,json,hashlib; d=json.load(sys.stdin)['data']; [print(k, hashlib.sha256(v.encode()).hexdigest()) for k,v in sorted(d.items())]"
   ```
   Expected: `diagnostics.py 7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d` and `target_diagnostics.py 48bf7454d99cfe0f3a50205d1aa4bdb55d959042bd7af3f10cae42fe57fc46d7`. The scripts are unchanged since 5b81152 (verify with `git log --oneline -3 -- world-model-stacking-encoders/src/diagnostics.py world-model-stacking-encoders/src/target_diagnostics.py`). `SRC_GIT_COMMIT` is the current main commit at submission, one value (as stage 1 did), 40 lowercase hex characters.
3. No Job named `stacking-encoders-control-partners-s2` or `stacking-encoders-control-fetch-draws` exists (`kubectl get job -n aria-world`), and nothing else of ours is running that would hit the 4-pod cap.
4. Stage 2 only: `/mnt/features/stacking-encoders/control-partners-s2/` holds no entry at all, files or folders (the Job aborts with exit 3 if it does, nothing run). The folder not existing is fine. `/mnt/features/stacking-encoders/features/dinov3/libero_goal` is still on the PVC (the staging step fails fast if not).
5. Fetch only: `/mnt/features/stacking-encoders/control-partners/` still holds the stage 1 files (the Job lists them and exits 4 if there is no `input-draws-*.npz`).

## Render and apply

Run from `world-model-stacking-encoders/experiments/2026-11-09-control-partners/`. Replace `<SHA>` with the full 40-character commit. The sed edits only the `value:` line of the env var; the placeholder-guard in the script is spelled `SET-AT-*` so the render cannot disarm it.

```
mkdir -p rendered
sed 's/value: "SET-AT-SUBMISSION"/value: "<SHA>"/' job-stage2.yaml > rendered/job-stage2.yaml
diff job-stage2.yaml rendered/job-stage2.yaml
grep -c SET-AT-SUBMISSION rendered/job-stage2.yaml
kubectl apply --dry-run=server -n aria-world -f rendered/job-stage2.yaml
kubectl apply -n aria-world -f rendered/job-stage2.yaml
```

The diff must show only the `SRC_GIT_COMMIT` line, and the grep must print 0 (`grep -c` also returns status 1 when it prints 0, so that is the expected result, not a failure). The script itself refuses to run unless `SRC_GIT_COMMIT` is exactly 40 lowercase hex characters (exit 2). The fetch Job needs no render:

```
kubectl apply --dry-run=server -n aria-world -f job-fetch-draws.yaml
kubectl apply -n aria-world -f job-fetch-draws.yaml
```

## What normal output looks like

Fetch (a minute or two once scheduled): `=== files in ...` with an `ls -l` (the ten stage 1 JSONs and the `input-draws-*.npz` files), the sha256 list (regular files only, sorted), `=== input-draws npz files: N` (expected 5: dup-dinov3, noise-4608-s0, randproj-dinov3-4608-s0, shuffle-dinov3-s0, c-radiov4-rerun), `tarball bytes: ...`, a sha256, then the base64 between `=== BEGIN input-draws-s1.tgz base64` and `=== END input-draws-s1.tgz base64`, then `=== done`. If the tarball is 20 MB or more it prints an ERROR line and exits 5 (the sizes are above it); then decide how to move the files. The size of the npz files was not known when this was written (not guessed).

Stage 2 (about 11 h): the same log shape as stage 1 (`SRC_GIT_COMMIT=... SYNTH_SEEDS=1 2`, nproc, meminfo, pip freeze, `staged dinov3: 4.4G in ~150s`, the libero_goal download, then `=== INPUT SIDE`, 6 runs, `=== TARGET SIDE`, 8 runs with `--- target <spec> seed <n>` headers and `date -u` lines), then 14 result JSONs (6 input + 8 target) printed as `=== result json ...`, then `=== sha256 of every output file` (14 JSONs and 6 `input-draws-*-s1/s2.npz`), `tarball bytes`, and the base64 between `=== BEGIN input-draws-s2.tgz base64` and `=== END input-draws-s2.tgz base64`, then `=== done`. Expected warnings as in stage 1 (pip root warning, pip upgrade notice, "alpha at grid edge" once per target run). After the download there are `du -sh /work/data` and `df -h /work` lines (the measured size of the raw data and the disk left). Abort messages: `ABORT: SRC_GIT_COMMIT is not 40 lowercase hex characters` (exit 2), `ABORT: ... already holds entries` (exit 3). The script also closes its output and waits for the log writer on exit, so the last lines (including after a failure) reach the pod log; the exit code is the script's own. Memory: stage 1 took no RSS reading, so watch RSS on Prometheus in the first minutes against the 14Gi request (the two 4,608-dim controls are the big ones). Check GPU is not involved (none requested) and the pod landed on a CPU node.

## Decode the tarball locally

Save the log, cut the base64 out, decode, compare hashes. Replace `<JOB>` with `stacking-encoders-control-partners-s2` or `stacking-encoders-control-fetch-draws`, and `<TAG>` with `s2` or `s1`.

```
kubectl logs -n aria-world job/<JOB> > results/<JOB>.log
sed -n '/^=== BEGIN input-draws-<TAG>.tgz base64$/,/^=== END input-draws-<TAG>.tgz base64$/p' results/<JOB>.log | sed '1d;$d' | base64 -D > /tmp/input-draws-<TAG>.tgz
shasum -a 256 /tmp/input-draws-<TAG>.tgz
mkdir -p results/draws-<TAG> && tar -xzf /tmp/input-draws-<TAG>.tgz -C results/draws-<TAG>
shasum -a 256 results/draws-<TAG>/*.npz
```

`base64 -D` is the macOS flag (GNU is `-d`). The first `shasum` must equal the `sha256sum /work/input-draws-<TAG>.tgz` (stage 2) or `/tmp/input-draws-<TAG>.tgz` (fetch) line printed before the BEGIN marker. Each extracted `.npz` hash must equal the matching line in the `=== sha256 of every file` list (fetch) or `=== sha256 of every output file` list (stage 2). Save each log byte for byte and record its own sha256 in the log entry, as the stage 1 collection did. The repo `.gitignore` excludes `**/experiments/*/results/**/*.npz`, so a plain `git add` skips the decoded npz files under `results/`. The lead decides whether to keep them outside git with their sha256 recorded (their size was unknown, not guessed).

## Time estimate

From the stage 1 log timestamps: per seed, input 532 + 500 + 301 = 1,333 s; target 3,238 + 5,644 + 4,716 + 4,742 = 18,340 s; per seed 19,673 s; two seeds 39,346 s (10.9 h); setup about 300 s (pip, staging DINOv3 148 s, download 12 s, tar and hashes; the pip time was not logged). Expected about 11 h. With the x1.5 slow-node factor, (39,346 + 300) x 1.5 = 59,469 s (16.5 h). `activeDeadlineSeconds: 86400` (24 h) is the PLAN's figure for stage 2 (Cost section, the lead's decision); it counts from Job creation, so it covers queue time and a slower node. Fetch: minutes, deadline 30 min. Both Jobs use `backoffLimit: 0`, which deviates from `_standards/08` line 31 (set it above 0 so a Job survives node reboots): for stage 2 a retry would hit the non-empty-folder abort and run nothing, and a node reboot ends the run, so a rerun needs a new folder and a new Job name; the fetch is a short read-only Job where a failure should be seen.

## Job count against the direction gate

`world-model-stacking-encoders/STANDARDS.md` line 48: "Gate (stop and ask): more than 8 GPUs at once, or a sweep of more than 20 Jobs." It does not say whether CPU Jobs count, whether the count is cumulative or per sweep, or how an Indexed Job counts (`LOG.md` line 1443 records that open question). So the count is unclear and no number is claimed. Plain facts: these two add 2 Jobs, both CPU, and this experiment's stage 1 was 1 Job. This is not a sweep over a parameter grid; the user already approved both in chat. Neither Job uses a GPU, so the 8-GPU half of the gate is not touched. Concurrency: 2 pods, under the 4-pod cap.
