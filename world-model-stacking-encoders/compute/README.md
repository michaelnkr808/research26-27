# Nautilus Job times: how long each Job waited and how long it ran

Started 2026-11-19 at the user's request (their words: "start noting like the times its been taking ... i want to be able to measure the blocking effect nautilus has had on us and bring back to my PI how much paid compute would speed up the process"). Kept by the experiment lead. One row per pod, added when a Job ends (or expires, or is deleted).

## Files

- `nautilus-job-times.csv`: every `stacking-encoders*` pod in namespace `aria-world` that NRP's public Prometheus still had on 2026-11-19 08:33 UTC (14-day window; the oldest row is from 2026-11-11), and every Job since.
- `nautilus-job-times-before-2026-11-15.csv`: Jobs before 2026-11-15, collected from the experiment LOGs, same columns plus `source`; times there are only as good as what the LOG recorded (section at the end).

## Columns

- `created_utc`: when the pod was created, which is when it entered the queue (`kube_pod_created`).
- `container_started_utc`: when its container started (`kube_pod_container_state_started`). Empty: it never started.
- `ended_utc`: `kube_pod_completion_time`. Empty: no completion was recorded (the pod was deleted, was a bare pod, or is still there).
- `queue_wait_h`: created to container start, in hours. It includes the image pull and volume mount, usually seconds to a few minutes. For a pod that never started it is the time until the Job expired or was deleted, and the note says so.
- `run_h`: container start to end, in hours. It includes environment setup inside the pod.
- `gpus`: GPUs requested (0 = CPU only). `node`: where it ran. The `nautilus-it-gpu0*.fullerton.edu` nodes carry L4 cards (read from the node labels, 2026-11-19).
- `phase`: the pod's last phase. **`Failed` is not always an infrastructure failure**: a check Job that finds a problem exits non-zero on purpose (for example the cache checks of 2026-11-18). The LOG entry of that date says which.
- `note`: what the times do not show.

## What this does and does not measure

It measures queue wait and run time per pod (verified from Prometheus for the rows marked by this file's first commit). It does not by itself measure: time lost to reruns after a failure (count the rerun's rows), time when nothing was submitted because a decision or a code change was pending, or how much faster a run would be on another card. Those need the LOG beside it.

## Reading on 2026-11-19 (arithmetic on this file's rows, GPU pods created since 2026-11-15, 16 pods)

- All GPU pods: 20.8 h of queue wait for 40.9 h of run time.
- Training-card pods (L40 or L40S, 3 pods, one of them the 32.2 h stage 1 run): 2.6 h of waiting.
- L4 pods (13): 18.2 h of waiting for 8.2 h of work. That includes one Job that waited 6.0 h and expired without running and one still waiting (3.2 h so far when this was written).
- The SigLIP 2 re-extraction ran twice because NRP's storage filled during the first try: the second try's rows are a rerun.

## How to add a row

When a Job ends: `kubectl get pod <pod> -n aria-world -o jsonpath='{.metadata.creationTimestamp} {.status.containerStatuses[0].state.terminated.startedAt} {.status.containerStatuses[0].state.terminated.finishedAt}'`, or the three Prometheus series above for a pod that is already gone. Never estimate a time; leave the cell empty and say why in the note.

## Before 2026-11-15 (`nautilus-job-times-before-2026-11-15.csv`, added 2026-11-19)

Collected by a worker from the experiment LOGs and READMEs (132 rows, each with the LOG file and line in `source`); the lead spot-checked three rows against the LOG lines. It is per Job or per submit-try where the LOG does not name pods. A time is in a cell only if the LOG states it, so many cells are empty: 105 GPU rows, 54 with a recorded wait, 48 with a recorded run time. Extra `phase` values: `deleted`, `deleted_pending` (deleted while still waiting), `DeadlineExceeded`, `OOMKilled`, `BackoffLimitExceeded`.

Reading (arithmetic on the rows that have the number; a lower bound, since about half the GPU rows have no recorded wait):
- GPU rows with a recorded wait: 242.7 h of queue wait in total (54 rows). GPU rows with a recorded run time: 124.1 h (48 rows). These are sums over pods; pods that waited at the same time overlap in calendar time, so this is not days of calendar delay.
- 28 GPU rows never ran (deleted while Pending, or expired): 91.3 h of waiting with nothing to show.
- The longest single waits: 26.2 h, 26.0 h, 19.0 h, 13.9 h, 12.1 h, 12.0 h.
- A deleted-while-Pending row followed by a resubmit restarts the clock: to total the delay of one task, add its earlier rows.
- One overlap with the other file: `stacking-encoders-pfb-ext-sam3-background-xph77`. The LOG gives two submit times for it; the Prometheus file has the pod's creation time.
