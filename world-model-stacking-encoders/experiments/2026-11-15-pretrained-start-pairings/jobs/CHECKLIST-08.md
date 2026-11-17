# _standards/08 rules mapped to the stage 2 Job templates

Written 2026-11-17 by a worker for the done check of the stage 2 manifests. Line numbers are in the templates as committed with this file (`job-train.template.yaml` = train, `job-smoke.template.yaml` = smoke, `job-clean-eval.template.yaml` = clean, `job-lp-eval.template.yaml` = lp); rendered files add 1 to 3 header lines on top. Found with grep, comment lines excluded. "RUN" points to `RUN-stage2-train.md` (RUN-T), `RUN-stage2-eval.md` (RUN-E) and `RUN-stage2-waves.md` (RUN-W).

## Workload type and Job rules (08 "Pick the right workload type")

| Rule (08) | Where | How |
|---|---|---|
| Every real training or eval run is a Job | `kind: Job`: train 69; smoke 63; clean 51; lp 61 | no bare pods, no Deployments |
| Never `sleep` in a Job; it must run a real command and exit | whole script | `grep -n -i sleep` on the templates finds no code line (comments only); each script ends after its python step and the exit trap. The RUN files' wait loops run on the Mac, not in a Job |
| `backoffLimit` > 0 so a Job survives node reboots | train 75; smoke 69; clean 60; lp 70 | train 3, smoke 2, clean / lp `backoffLimitPerIndex` 2 (per shard) with `maxFailedIndexes` 0 |
| `restartPolicy: Never` | train 93; smoke 87; clean 79; lp 89 | required by `podFailurePolicy` too (train 78; smoke 72; clean 64; lp 74) |
| Delete broken Jobs right away; delete finished Jobs yourself | RUN-T steps 8 and 11; RUN-E steps 11 and 17 | plus `ttlSecondsAfterFinished` 86,400 (train 77; smoke 71; clean 63; lp 73); deterministic failures end the Job at once through `podFailurePolicy` FailJob instead of spawning pods |
| Not more than 400 Jobs at once | RUN-W "Totals" | 58 Jobs for the whole stage, a handful at a time |
| A deadline that bounds the Job | Job `activeDeadlineSeconds` train 76; smoke 70; clean 62; lp 72; pod train 94; smoke 88; clean 80; lp 90; inner `timeout --kill-after` train 205, 213, 222, 248, 259; smoke 169, 184; clean 161, 174; lp 171, 186 | arithmetic in each header; render.sh computes them |

## Things that get you banned (08)

| Rule | Where | How |
|---|---|---|
| No `sleep` / never-ending Job | as above | |
| No tolerations without admin OK | none in any template | grep finds only header comments saying "no tolerations" |
| No under-used requests (namespace ban risk) | requests train 278; smoke 225; clean 209; lp 221 | sized from stage 1's measured use (train: 32Gi; L4 eval: 28Gi; CPU 4), pairs' memory from arithmetic in the train header; the exit trap prints `[cgroup]` peak and anon to resize from a measurement |
| Never force-delete pods | RUN-T step 6, RUN-E (stage 1's steps referenced) | "graceful, never `--force`" |

## GPU utilization (08 "GPU utilization: how it's actually checked")

| Rule | Where | How |
|---|---|---|
| GPU above 40% of the request; no more than 4 pods outside the ranges | in-pod sampler train 211; smoke 173; clean 165; lp 175; RUN-T step 8; RUN-E steps 8 and 16 | training ran at about 90% in stage 1; eval ENVS raised until 10-minute windows reach 40% (stage 1 rule), else stop for the user; long LIBERO-Plus pods get a 3 h trailing-mean check (the portal's warning path) |
| Delete pods as soon as the work is done | scripts exit on completion | the GPU sampler is killed in the exit trap |
| Start with 1 GPU | `nvidia.com/gpu: "1"` train 278, 279; smoke 225, 226; clean 209, 210; lp 221, 222 | one GPU per pod everywhere |
| Never leave an interactive GPU pod | none created | |
| Check the Violations page after launch (pre-flight item 8) | RUN-T step 8, RUN-E steps 8 and 16 (via stage 1's RUN steps) | |

## Requests and limits (08)

| Rule | Where | How |
|---|---|---|
| Limits within 20% of requests | requests / limits train 278; smoke 225; clean 209; lp 221 / train 279; smoke 226; clean 210; lp 222 | CPU 4 / 4800m (1.2x); memory 32Gi / 38Gi (1.19x), pairs 36Gi / 43Gi (1.19x), eval 28Gi / 33Gi (1.18x); ephemeral 38Gi / 44Gi (1.16x), 50Gi / 58Gi (1.16x) |
| Over 50 Gi of ephemeral scratch: request `ephemeral-storage` explicitly | train 278, 279; smoke 225, 226; clean 209, 210; lp 221, 222; `/work` sizeLimit train 288; smoke 235; clean 219; lp 232 | as stage 1 |
| `/dev/shm` as a memory emptyDir | train 284; smoke 231; clean 215; lp 227 | 2Gi train, 4Gi eval |

## Choosing GPUs (08)

| Rule | Where | How |
|---|---|---|
| Pin models through `nvidia.com/gpu.product` affinity, valid values only; special resources (A40, A6000) only by name | train 102; smoke 96; clean 88; lp 98 | train `[NVIDIA-L40, NVIDIA-L40S]` (stage 1's set), eval `[NVIDIA-L4]`; no A40/A6000 under `gpu.product`; no named resources |
| No `priorityClassName` other than opportunistic | none set | |
| Driver pin | `nvidia.com/cuda.driver.major Gt 569` on the line after train 102; smoke 96; clean 88; lp 98 | as stage 1 |

## Storage, PVC and CephFS (08 "Storage", "CephFS in detail")

| Rule | Where | How |
|---|---|---|
| Storage and compute in the same region | train 104; smoke 98; clean 90; lp 100 | `us-west` for `rook-cephfs` |
| CephFS: never write one file from several pods; one writer per file | out folder per run (train); one records file per shard and per pass (eval); logs named per pod with a timestamp | train: `$RUN_ROOT/pairings-<cfgk>-s<seed>`; RUN-T step 6: no second pod on one out folder before a resubmit |
| No pip / conda installs on CephFS | `/work` emptyDir train 288; smoke 235; clean 219; lp 232 | setup_env.sh installs into /work; the env bundle restores into /work |
| Not archival; back up final checkpoints | RUN-W "What the user must OK" | the Ceph S3 backup of each `model.pt` (the user's yes) precedes any `ckpt.pt` deletion |
| Agents never delete PVC data without asking | no `rm` in any template (grep); a stale folder is renamed, never deleted | RUN-T step 11: deletion only on the user's OK |
| PVC mounted under `/mnt/...`, not a home dir | claimName train 292, 296; smoke 239, 243; clean 223, 227; lp 238, 242 | `/mnt/proxy`; the second volume at `/mnt/runs` (eval mounts it read-only) |
| A new CephFS volume: `rook-cephfs`, ReadWriteMany, Bound before use; expansion cannot be undone | RUN-W | the second volume is a separate claim (the user's choice), created only on the user's go; not created by these files |
| Logs to the PVC | `exec > >(tee -a $LOG ...)` train 123; smoke 116; clean 107; lp 117 | full log per pod under `/mnt/proxy/logs/` |

## Reproducibility (08 pre-flight, and 03)

| Rule | Where | How |
|---|---|---|
| Pin the image tag | train 108; smoke 102; clean 94; lp 104 | `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` |
| Config and manifest committed before submitting; commit recorded | `SRC_GIT_COMMIT` train 263; smoke 202; clean 184; lp 196; `sha256sum -c` of the /src keys and pinned inputs train 130, 236; smoke 148, 197; clean 140; lp 150, 178 | render.sh --commit fills the commit and the key hashes; the rendered file is committed as submitted (RUN-T step 4) |
| Credentials only in Secrets | `secretKeyRef` train -; smoke 210; clean 192; lp 204 | HF_TOKEN from Secret `stacking-encoders-hf`; not mounted for config none |
| Sanity checks passed at small scale first (pre-flight 1) | stage 1 sanity checks and full run; stage 2 code PRs #54-#57 (CPU tests) and the cache pre-check (LOG 2026-11-17) | the first-minutes checks of every run (RUN-T step 8) |
| GPU-hours logged, estimated against actual | RUN-T step 10, RUN-E steps 8, 15 | |
| Gate: more than 8 GPUs or more than 20 Jobs needs the user | RUN-W | 58 Jobs: the user's sweep OK; GPU count before every submit (RUN-T step 5, RUN-E steps 7, 10, 13) |
