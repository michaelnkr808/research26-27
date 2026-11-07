# 08 — Compute on NRP Nautilus

All GPU work runs on the NRP Nautilus Kubernetes cluster. These rules come from two sources, and each one is tagged with its source:

- **[NRP]**: the official docs at <https://nrp.ai/documentation/>, checked 2026-10-16 against the raw source in the site repo (<https://gitlab.nrp-nautilus.io/prp/nrp-site>, commit `f926199`, 2026-10-15). That covers the [Cluster Policies](https://nrp.ai/documentation/userdocs/start/policies) page, the [AUP PDF](https://nrp.ai/NRP-AUP.pdf), and the code for the portal's Violations page. The policies change, so recheck at least once a quarter and before any large run, and update this file when they change. Last recheck recorded in the repo: 2026-10-28, live Policies page (last updated 2026-10-26) and Jobs page (2026-10-25); wording unchanged on the numbers it checked: GPU above 40%, CPU 20-200%, RAM 20-150%, the 4-pod cap, limits within 20% of requests, no `sleep` in Jobs, and the exempt request size (it did not recheck the storage rules, such as the 6-month purge or the 10k-file rule) (`world-model-stacking-encoders/LOG.md` 2026-10-28 (g), policy check). The recheck did not change this file, and the pinned commit above is still the one checked on 2026-10-16. **Rechecked 2026-11-07 (GPU-utilization rule only) [verified]:** live Policies, Jobs, GPU Pods, Monitoring and Long Idle pages (all footed "Last updated: Oct 26, 2026"; footer dates are not proof a page is unchanged), the AUP PDF (Draft 0.1), <https://nrp.ai/llms.txt> (the file the Policies page points AI agents to), and the site repo (`main` at `57c9fb43`, 2026-11-03). The GPU wording matches this file. Repo history: `policies.mdx` changed after `f926199` in `f15598dc` (2026-10-27, LLM Usage section only, GPU text untouched); `jobs.mdx` was last committed 2026-09-01 (`87d59a73`); `src/components/vue/UserInfo.vue` (which holds the Violations-page text) is byte-identical at `f926199` and `57c9fb43`, last commit `b2f3545c` 2026-10-09. Beyond the 30-minute startup exemption and the 10-minute checks, no grace period, averaging window, warning, strike or automatic kill is documented anywhere in those sources (`world-model-stacking-encoders/LOG.md` 2026-11-07, NRP GPU-utilization rule rechecked).
- **[lab]**: lessons learned on this cluster during 2025–26 (see `../research25-26/LangSplat/README.md`, `../research25-26/spring26research_recap.md`, and `../research25-26/notes (mds)/degauss_full_notes.md`). These are field experience and are not in the official docs.

Where the two disagree, the official policy decides what's allowed. A lab lesson can only make a rule stricter.

## Our setup [lab]
- **Namespace:** `aria-world`. It is **shared with the lab**.
- **PVCs:**
  - `scenescript-persistent` (lab-shared, about 40 GB, `rook-ceph-block`, ReadWriteOnce) and `michael-persistent` (personal, 20 GB, `rook-ceph-block`, ReadWriteOnce). Both mount at `/persistent`. These sizes are from the older lab setup; the stacking LOG records only these later views: 5 PVCs, all `rook-ceph-block`, on 2026-10-19 (`world-model-stacking-encoders/LOG.md` 2026-10-19 (b)); `michael-persistent` as the 20 GB PVC (2026-10-18/19); and, on 2026-10-28, three 400Gi `vissam-*` PVCs from another project next to ours (`world-model-stacking-encoders/LOG.md` 2026-10-28 (b)). No entry rechecks `scenescript-persistent` or the sizes. Not rechecked since 2026-10-28.
  - `stacking-encoders-features` (CephFS `rook-cephfs`, 100Gi, ReadWriteMany): created and Bound (`world-model-stacking-encoders/LOG.md` 2026-10-19/20 (f); manifest `world-model-stacking-encoders/data/pvc-features-cephfs.yaml`). Mounted at `/mnt/features`.
  - `stacking-encoders-proxy` (CephFS `rook-cephfs`, 1Ti, ReadWriteMany): created and Bound (`world-model-stacking-encoders/LOG.md` 2026-10-28 (m); manifest `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/pvc-proxy.yaml`). Mounted at `/mnt/proxy`.
- **Manifests:** `../nautilus-scripts/` (a sibling folder of this repo, outside it) has `cp-pod.yml` (CPU-only staging pod), `scenescript-pod.yml` (GPU dev pod), and `pvc-template.yaml`. The stacking direction's Job manifests are under `world-model-stacking-encoders/experiments/` (see "Minimal Job template" below).
- **Base image:** `minarami/scenescript:latest` (PyTorch 2.4, CUDA 12.1) for the older lab pods. The stacking Jobs use `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` for GPU work and most of the proxy round's CPU-only Jobs (simulator setup, env builds, read tests, cache checks). `python:3.11.14-slim` is used only by the early diagnostics Jobs (`world-model-stacking-encoders/experiments/2026-10-19-diagnostics-libero-goal/job.yaml`, `world-model-stacking-encoders/experiments/2026-10-21-target-diagnostics/job.yaml`, the three `diag-*-job.yaml` in `world-model-stacking-encoders/experiments/2026-10-19-feasibility-tally/`) and the download Job (`world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/job-download.yaml`). Training and extraction Jobs install their Python packages inside the pod (`world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/job-train.template.yaml`). Rollout Jobs restore a prebuilt environment bundle instead (`world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/job-rollout.template.yaml`, `env_bundle.sh restore`).
- A direction's `STANDARDS.md` records which PVC, image, and GPU types it uses.

## Pick the right workload type [NRP]

| Type | Max lifetime | GPUs | Use for |
|---|---|---|---|
| Bare **Pod** (interactive) | **6 hours**, then destroyed | ≤ 2 (also ≤ 32 GB RAM, ≤ 16 CPU) | Debugging, short interactive work. `sleep` is allowed here. |
| **Job** | Runs to completion | ≤ 8 per node | **Every real training or eval run.** |
| **Deployment** | **2 weeks**, then deleted | **None allowed** | Long-lived idle shells or services. Requests should be minimal. |

- **Never put `sleep infinity` (or a script ending in `sleep`) in a Job. Doing so gets you banned from the cluster** [NRP]. A Job must run a meaningful command and exit.
- Prefer Jobs over interactive pods for anything reportable. A Job is also a record of exactly what ran, which `03` requires anyway.
- Set `backoffLimit` > 0 so a Job survives node reboots. Use `restartPolicy: Never` [NRP].
- Don't submit more than **400 jobs** at once. The cluster has no fair queueing, so for big sweeps use a work-queue pattern, such as Redis [NRP].
- Delete a broken Job right away, or it keeps spawning pods [NRP].
- Jobs that are left running are **not** cleaned up automatically. Delete finished Jobs yourself [NRP].

## Things that get you banned [NRP]
Each item below is quoted from or stated directly in the docs:
1. **A Job running `sleep` or anything else that never ends by itself.** *"Users running a Job with `sleep infinity` command or equivalent (script ending with "sleep") will be banned from using the cluster."*
2. **Tolerating a taint you aren't authorized for.** *"Tolerating a taint you're not explicitly allowed to tolerator by the cluster admins will result in being banned."* By default you may only use nodes with no taints. Never copy `tolerations:` out of someone else's manifest or a dumped live pod spec.
3. **Under-used requests** [verified 2026-11-07], at the namespace level: *"Admins monitor resource utilization and will ban namespaces with under-utilized resources requests."* Our namespace is shared, so one person's idle pod puts the whole lab at risk. The same page also says *"Namespaces with consistently underutilized requests risk being banned."* The page gives no number, no time window, and no process for this ban (checked 2026-11-07).
4. **Breaking the AUP** (see below): *"Violations of this AUP may result in loss of access to NRP resources."*

Also: **never force-delete pods** (*"Never force Delete Pods."*). A pod stuck in Terminating usually means its node is down or overloaded.

## GPU utilization: how it's actually checked
- Requested GPUs are reserved for you until the pod is gone, even while they sit idle. **Delete pods as soon as the work is done** [NRP].
- Policy text, verbatim: *"A user can not submit more than 4 pods, violating the following conditions: GPU utilization is more than 40% of the requested GPUs. CPU usage is within 20-200% of the requested CPUs. Memory (RAM) usage is within 20-150% of the requested memory."* The wording is awkward, but the portal's code settles it: a GPU pod is flagged when its utilization is **below** the minimum (`gpuMinPercent`, 40% per the docs). CPU and memory are flagged when they fall **outside** their ranges, on either side [NRP, `src/components/vue/userinfo/format.ts`, verified].
- **How the portal checks**, verbatim from the Violations page: *"The portal checks running pods every 10 minutes. Pods younger than 30 minutes are not checked yet."* *"Opportunistic (negative-priority) pods and JupyterHub pods are not checked."* CPU and memory are only checked on pods whose requests are above the exempt thresholds (the docs say 1 CPU and 2 GB; the live values are on the page) [NRP].
- **Your live status:** the [Violations page](https://nrp.ai/userinfo) shows any running pods that are outside the policy and the thresholds currently in force. Check it after every GPU launch.
- **What the rule is, in plain words** [verified 2026-11-07]. 40% is a threshold that marks a pod as outside the policy on the portal's Violations page. The rule itself is a limit on how many pods may be outside the ranges at once: *"The policy allows a user no more than 4 pods outside these ranges."* (`src/components/vue/UserInfo.vue` line 229, `57c9fb43`); *"A user may not run more than 4 pods outside these bounds"* (<https://nrp.ai/llms.txt>); *"A single user can’t submit more than 4 pods violating the following limits"* (Policies TL;DR). It is not a cap on your total pods while any pod violates. No source says what happens if you go over 4. The same UserInfo.vue sentence continues: *"Namespaces that keep requesting more than they use risk being banned."* And the Policies page says admins *"will ban namespaces with under-utilized resources requests"*. A pod below 40% is not documented to be killed, evicted, or shut down.
- **What the docs document, and what they don't** (checked 2026-10-16, rechecked 2026-11-07 [verified]). Documented timing: *"The portal checks running pods every 10 minutes. Pods younger than 30 minutes are not checked yet."* and *"Opportunistic (negative-priority) pods and JupyterHub pods are not checked."* (`UserInfo.vue` lines 222 and 227, identical at `f926199` and `57c9fb43`). Nothing else is documented: no averaging window, no further grace period, no warning, no kill. Where and how utilization is averaged is unknown: it is computed by the portal backend, which is not public (recorded 2026-10-16, not rechecked). The only backend file `format.ts` names is `usage.go`, as the thresholds map (`allowedUtilization`). Deletions the docs do describe, none tied to GPU utilization: a bare pod is destroyed after 6 hours (*"such pods will be destroyed in 6 hours unless you request an exception for your namespace"*), Deployments are deleted after 2 weeks, a pod over its memory limit is OOM-killed, pods writing over 50Gi of ephemeral scratch can be evicted, opportunistic pods can be preempted, volumes unaccessed for 6 months can be purged, and llms.txt says *"Pods are ephemeral and get deleted or evicted due lack of resources on the node or other node issues."* **There is no documented idle-GPU timeout, so don't rely on one, and don't assume a kill rule either.** A practical reading (speculative): from 30 minutes after startup, every 10-minute check can catch a GPU pod sitting below 40%.
- Keep GPU utilization **above 40%, and ideally near 100%**. Check it on the [namespace GPU dashboard](https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus). The Policies page says *"Check the GPU dashboard for your namespace to make sure the utilization is above 40%, and ideally is close to 100%."* [NRP, rechecked 2026-11-07]. It is guidance to check, not a stated kill line. llms.txt repeats it (*"GPU utilization above 40%"*, *"namespaces with under-utilized requests are banned"*); a reviewer searched it (keywords across the whole file, plus a reading of its rules and admin sections) and found no kill, strike or time-window text in it [verified 2026-11-07].
- **Idle-GPU strikes [lab-reported, not in NRP's published policy]:** the lab was told that idle GPU jobs get killed under a 3-strike rule, with suspension on strike 3, and this account was said to be at **1 of 3** (as of 2026-10-16, when this line was first written; source: user, 2026-11-06, relayed by the lead). The 2026-11-01 pilot held a GPU at about 0% for about 2 h, which the LOG calls an idle-GPU violation, so it may have added a strike; the Violations page was not read by an agent (`world-model-stacking-encoders/LOG.md` 2026-11-01 (f)). That strike system appears **nowhere** in the public docs, the AUP, llms.txt, or the portal code (checked 2026-10-16 and again 2026-11-07 [verified]: no GPU-utilization-related strike, warning or kill text; the Policies page does have OOM-kill, 6-hour and "3 notifications" text, but the 3 notifications are about deleting Deployments older than 2 weeks). It could be enforced by admins or a backend not in the repo (speculative). The user said 2026-11-06 to keep assuming 1 of 3, so treat it as real. Ask NRP support for the exact rule so this line can be replaced with the official one.
- Start with **1 GPU** on a subsample. Go above 1 only once you've shown the job is GPU-bound [NRP].
- Agents: **never** leave an interactive GPU pod running at the end of a session. Check with `kubectl get pods -n aria-world` and delete your own pods before you stop.

## Requests and limits [NRP]
- Limits must be **within 20% of requests**. Set requests near average usage and limits slightly above peak.
- Going over the memory limit gets the pod OOM-killed. Going over the CPU limit gets it throttled.
- Tune requests with the [CPU/memory dashboard](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods). Read **Memory Usage (RSS)**, not "Memory Usage", because the latter includes disk cache.
- More than 50 Gi of ephemeral scratch per container can get the pod evicted. If you need more, request `ephemeral-storage` explicitly.
- Kernel (`sy`) CPU load above about 15% usually means too many threads or too many small-file reads.
- Mount `/dev/shm` as a memory `emptyDir` for PyTorch DataLoaders. Without it, shared memory defaults to 64 MB.

## Choosing GPUs [NRP]
- Normal GPUs use `nvidia.com/gpu`. To pin a model, use node affinity on `nvidia.com/gpu.product` (for example `NVIDIA-GeForce-RTX-4090`). List current values with `kubectl get nodes -L nvidia.com/gpu.product`. The table in the docs is out of date.
- Some GPUs are **special resources** that `nvidia.com/gpu` won't match: A40 (`nvidia.com/a40`), RTX A6000 (`nvidia.com/rtxa6000`), A100 (`nvidia.com/a100`), H100/H200/GH200, and others. [lab] Two of our old manifests list A40/A6000 under `gpu.product` affinity. The current training GPU list does include A40 and A6000 under `gpu.product` with `nvidia.com/gpu` (`world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-06 (c); `world-model-stacking-encoders/LOG.md` 2026-11-02 (b), 2026-11-03 (i)). That entry cannot match (verified, 2026-11-06): every node that advertises `nvidia_com_a40` or `nvidia_com_rtxa6000` showed `nvidia_com_gpu` allocatable 0 or absent, so a pod requesting `nvidia.com/gpu` with A40 or A6000 in its `gpu.product` affinity can't land on those nodes (read-only Prometheus check, 2026-11-06 ~23:10 UTC; `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-06 (f), subsection "Lead's read-only Prometheus check"; `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/PLAN.md` "Amendment 2026-11-06 (b)"). A by-name request (`nvidia.com/a40` or `nvidia.com/rtxa6000`) is the only way to use these cards. Whether it schedules is untested: an A40-named Job was submitted but had not run when this was written.
- **A100:** the namespace quota is 0 by default, so it needs the A100 access form. **H100/H200/GH200:** not user-requestable (*"There is no access form."*). The RTX PRO 6000 Blackwell is reserved and not generally available.
- `priorityClassName: opportunistic` bypasses GPU quota (including A100/H100), **but the pod can be preempted at any time**. Use it only with Jobs that checkpoint often. Don't set any other priority class, because banned classes get rejected at admission.
- You can pin CUDA runtime or driver versions with the `nvidia.com/cuda.runtime.major/minor` and `nvidia.com/cuda.driver.major` labels.
- **CPU-only jobs:** keep them off GPU nodes with node anti-affinity on `feature.node.kubernetes.io/pci-10de.present NotIn ["true"]`. For large CPU jobs, also use `priorityClassName: opportunistic` [NRP].
- [lab] `../nautilus-scripts/pod-michael-pod.yaml` is a **dumped live pod** (it has `status`, `uid`, `nodeName`, and tolerations added by the system). Don't `kubectl apply` it. Use `scenescript-pod.yml` as the template.
- [lab] GPUs that worked for us: RTX 4090, RTX A5000, RTX 3090, TITAN RTX, RTX A4000, RTX 3080, V100, RTX 2080 Ti, A10. Stacking-direction Jobs have since run on TITAN RTX, RTX 2080 Ti, L4, L40, L40S, RTX 5000 Ada, RTX 4090, RTX 3090, and RTX A6000 (the 2026-10-16 step benchmark requested it as `nvidia.com/rtxa6000`, not through `gpu.product`) (`world-model-stacking-encoders/LOG.md` 2026-10-16 (d), 2026-10-17, 2026-10-18/19, 2026-10-19/20 (f), 2026-10-29 (e), 2026-10-29 (h), 2026-10-29 (i), 2026-11-01 (f), 2026-11-02 (c); `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-06 (c)). Its current training GPU list (3090, 4090, A5000, A6000, L40, L40S, A40) is in the same LOG, 2026-11-06 (c).

## Storage [NRP unless tagged]
- **RBD (`rook-ceph-block*`)**: ReadWriteOnce, one pod at a time. Fine for conda/pip envs and compiling. Recommended for data that doesn't need to be shared.
- **CephFS (`rook-cephfs*`)**: ReadWriteMany. Good for large files such as checkpoints. **Installing conda/pip on CephFS is prohibited.** It handles many small files badly, so don't write the same file from several pods at once. Details in the next section.
- **Ceph S3**: the right home for large datasets (over 100 GB) and the fastest way in and out of the cluster.
- **Local NVMe** (`emptyDir` + `ephemeral-storage` request): for I/O-heavy training, stage data from Ceph to local disk with an init container. It can be hundreds of times faster than Ceph.
- Pick storage and compute in the **same region**, via the `topology.kubernetes.io/region` affinity.
- **Not archival.** Volumes not accessed for **6 months can be purged without notice**. Keep anything that matters (results, final checkpoints) somewhere else too. `03` already requires results to live in the repo.
- Don't `rm -rf` folders with more than 10k files on Ceph. Use `find DIR -type f -delete`, then `find DIR/* -depth -type d -delete`.
- [lab] **The PVC is shared, so check ownership before deleting anything at the top level.** We once deleted a labmate's `/persistent/scenescript`. Agents: **never delete PVC data without asking**, per the verification gates in `05`.
- [lab] Because the volumes are ReadWriteOnce, a `cp-pod` can't attach while a GPU pod (or a labmate's pod) holds the volume. Tear one down first.
- [lab] Conda caches fill the PVC fast. Set `CONDA_PKGS_DIRS=/tmp/conda-pkgs` so the cache lands on the pod's overlay disk.

## CephFS in detail [NRP unless tagged]
Sources: `userdocs/storage/ceph.md`, `storage/intro.md`, `storage/purging.md`, `running/io-jobs.md` and `running/gui-desktop.md` in the site repo, commit `f926199`, checked 2026-10-19.

**What it is.** CephFS is a shared POSIX filesystem (ReadWriteMany) that many pods can mount at once. The docs call it *"the primary way of storing data in nautilus"* and say it has the largest storage pool. Most classes use spinning drives with NVMe metadata.

**Storage classes.** Pick the one in the same region as your compute.

| StorageClass | Region | Who can use it |
|---|---|---|
| `rook-cephfs` | US West | Anyone. Our default. |
| `rook-cephfs-central` | US Central | Anyone |
| `rook-cephfs-east` | US East (mixed drives) | Anyone |
| `rook-cephfs-south-east` | US South East | Anyone |
| `rook-cephfs-pacific` | Hawaii + Guam | Anyone |
| `rook-cephfs-fullerton` | US West (local) | Anyone (no restriction listed) |
| `rook-cephfs-haosu` | US West (local) | **Restricted:** Hao Su and Ravi cluster. Don't use. |
| `rook-cephfs-tide` | US West (local) | **Restricted:** SDSU Tide cluster. Don't use. |
| `rook-cephfs-ucsd` | US West (local), all NVMe | Has its own policy: *"very fast and small"*. Data is for running computation only and can be purged *"at admin's discretion without any notifications"*. Use it for scratch, never as the only copy. |

**When to use it.**
- It's good at large files and parallel streams. The intro page says it has *"a much higher read/write performance than RBD, especially if you are able to open multiple parallel streams to a file and aggregate many small files to a few large files."* Checkpoints and packed feature files (HDF5, tar shards) are the intended use.
- It's bad at many small files, because every file goes through the metadata servers. **No conda/pip installs and no code compilation on CephFS.** Put envs on RBD.
- RBD vs CephFS speed: `ceph.md` says RBD *"provides fastest access to the data, and is preferred for all datasets not needing shared access"*. `intro.md` explains the difference: RBD *"can provide higher IOPS than CephFS, but overall read/write performance tends to be slower than CephFS."* So RBD has quicker access (latency/IOPS), and CephFS has higher throughput on large files. Default to RBD for data only one pod needs. Choose CephFS when several pods need the data at once, or for large files read in parallel streams.
- Per-file size limit: 16 TB. The docs give no per-volume or per-namespace CephFS cap.
- For hot training loops, CephFS is still remote. Stage the data to local NVMe with an init container (see the RBD/Local NVMe notes above and `running/io-jobs.md`).

**Writing safely.** From `ceph.md`:
- **Never open the same file for writing from several pods in parallel.** It can lock the file for other clients and corrupt the data or the filesystem.
- Give each writer its own file, for example one shard per Job or rank, then merge in a single step.
- Close files as soon as you're done, rather than holding handles open.
- If several clients need to change the same data, work on a copy, or use versioned filenames or an explicit lock.

**Mounting.**
- Create the PVC with `storageClassName: rook-cephfs` and `accessModes: [ReadWriteMany]`, and wait for `kubectl get pvc <name>` to show `Bound`.
- Add region affinity so the pod runs near the storage: `topology.kubernetes.io/region In [us-west]` for `rook-cephfs`. List the labels with `kubectl get nodes -L topology.kubernetes.io/region`. As of 2026-10-19, 288 nodes carried `us-west` (`world-model-stacking-encoders/LOG.md` 2026-10-19 (c), the first entry with that heading); not rechecked since. [lab, speculative] The affinity also shrinks the GPU pool, so if a Job won't schedule, weigh latency against waiting.
- Don't mount CephFS as a home directory. The GUI-desktop page says *"NEVER MOUNT TO `/home/ubuntu`"*. Mount it under a subfolder such as `/mnt/persistent`.
- If the node holding a Ceph mount goes offline, the pod stays stuck until the node comes back. Ceph does not migrate it; only Linstor does. Combined with the "never force-delete pods" rule, a stuck pod means waiting.

**Sizing and cleanup.**
- Volumes created since December 2020 can be grown by editing `spec.resources.requests.storage` on the PVC. Start with a modest size and expand later. The docs say nothing about shrinking, so assume you can't.
- Same rules as other Ceph storage: not archival, purged after 6 months unaccessed, and use `find -delete` rather than `rm -rf` for folders with more than 10k files.
- Usage dashboard: [Ceph cluster on Grafana](https://grafana.nrp-nautilus.io/d/r6lloPJmz/ceph-cluster).

**Not in the docs** (checked 2026-10-19): CephFS quotas per namespace, expected throughput numbers, or any backup or snapshot guarantee. On our namespace, a read test on one node on 2026-11-01 gave about 30 MB/s with 1 thread and about 300 MB/s with 16 threads (`world-model-stacking-encoders/LOG.md` 2026-11-01 (g)). Separately, copies to CephFS from one node (ry-gpu-08) ran under 1 MB/s (`world-model-stacking-encoders/LOG.md` 2026-11-01 (c)). Both are single-node measurements, not a guarantee.

[lab] First use: `world-model-stacking-encoders/data/pvc-features-cephfs.yaml` (100Gi `rook-cephfs`, shared feature cache). It was drafted 2026-10-19 and created after the user's OK (`world-model-stacking-encoders/LOG.md` 2026-10-19/20 (f)). A second CephFS volume, `stacking-encoders-proxy` (1Ti), was created 2026-10-28 (`world-model-stacking-encoders/LOG.md` 2026-10-28 (m)).

## Moving data
- `kubectl cp` is **for small files only**. It goes through the API server, and more than a few MB can hurt the cluster [NRP]. [lab] In practice it drops streams above about 10 MB.
- For large data, have a **Job** pull it in (rclone, wget, scp, or S3), or use Ceph S3 [NRP]. For PVC-to-PVC copies, use `pv-migrate` or a `gsutil -m rsync` Job [NRP].
- [lab] To pull results out, tar them on the pod and copy back one small archive. Do the heavy work on the pod and only fetch the final artifacts.
- Credentials go in Kubernetes Secrets, never in images, manifests, or committed files [NRP; also `04`].

## Acceptable Use Policy (AUP) [NRP, `NRP-AUP.pdf`, Draft 0.1]
You accept the AUP when you get access, and again every 12 months. The parts that affect research here:
- **No protected data.** *"NRP does not provide services or resources that satisfy the security requirements for handling or processing HIPAA, PII, CUI, FERPA, or any other data protected by statute or data-use agreement."* Check this before uploading any dataset with people in it. Our ARIA egocentric recordings may contain faces or other PII, so ask before putting new recordings on the cluster (speculative risk, not yet assessed). Record the answer in that direction's `data/README.md` (see `06`).
- **Back up your own data.** *"You understand that you are responsible for backing up your data."*
- **Non-commercial use only**, and no use that breaks the license terms of software or data you bring in.
- **Acknowledge NRP in every paper** and other citable work that used it. Use this text verbatim:
  > This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners.
- Report suspected security breaches to security@sdsc.edu.

## Other services
- **JupyterHub** (the NRP-hosted one): *"Your Jupyter container will shut down 1hr after your browser disconnects from it."* Its pods aren't checked for violations, but it's only meant for testing [NRP].
- **Managed LLMs** (if agents call NRP-hosted models): the gateway enforces *"200,000 output tokens per minute"* per API token and model, and returns HTTP 429 above that. Per-user concurrency caps depend on the model, and any request using 35% or more of the context length is limited to 1 concurrent request. Scripts should retry with backoff [NRP, `ai/llm-managed/fair-use`]. Record the model ID and date in experiments, per `07`.

## kubectl gotchas [lab]
- Inside `kubectl exec`, **never** use `conda run -n <env>` (it hangs). Call the env's Python by its full path.
- **Never** run `sed -i` through `kubectl exec` (it hangs). Edit locally, then `kubectl cp` the file.
- **Never** paste multi-line Python into `exec`. Write it to a file, `kubectl cp` it, then run it.
- Set `MPLBACKEND=Agg` on pods, since they have no display.
- Tee long runs to the PVC, for example `... 2>&1 | tee /persistent/logs/run-$(date +%Y%m%d-%H%M%S).log`, so a dropped session doesn't lose the log.
- Large datasets can take minutes to load before printing any progress, so check `nvidia-smi` and the logs before deciding a job has hung.
- [lab] Cross-building CUDA images on the Mac (linux/amd64 under QEMU) has failed for us. Building the env on the pod (on RBD, not CephFS) is the fallback.

## Minimal Job template
This is adapted from the patterns in the NRP docs. The template itself (with `/persistent` and a conda env path) has not been run as written. Many Jobs built on the same pattern have run on our namespace since; for a working example see `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/job-train.template.yaml` and its rendered runs in `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/rendered/` (for instance `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/jobs/rendered/job-train-wave1-dinov3.yaml`, whose seed 1 completed: `world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md` 2026-11-06 (c)). Fill in the placeholders, check it into `experiments/<run>/`, and commit before submitting (per `03`).

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: <direction>-<run-name>        # unique, kebab-case
  namespace: aria-world
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: main
        image: <image:tag>            # pin a tag or digest, not :latest, for reportable runs
        command: ["bash", "-c"]
        args:
        - |
          set -euo pipefail
          cd /persistent/<code-dir> && git rev-parse HEAD
          /opt/conda/envs/<env>/bin/python <script>.py --config <config> \
            2>&1 | tee /persistent/logs/<run-name>-$(date +%Y%m%d-%H%M%S).log
        env:
        - name: MPLBACKEND
          value: Agg
        resources:                    # limits within 20% of requests
          requests: {cpu: "4", memory: 24Gi, nvidia.com/gpu: "1", ephemeral-storage: 20Gi}
          limits:   {cpu: "4", memory: 28Gi, nvidia.com/gpu: "1", ephemeral-storage: 20Gi}
        volumeMounts:
        - {name: persistent-vol, mountPath: /persistent}
        - {name: dshm, mountPath: /dev/shm}
      volumes:
      - name: persistent-vol
        persistentVolumeClaim: {claimName: <pvc>}
      - name: dshm
        emptyDir: {medium: Memory}
```

## Pre-flight checklist (before any GPU run)
1. The sanity checks in `07` passed at small scale.
2. The cost estimate fits the direction's compute budget, and the GPU-hours are logged. Above budget is a verification gate in `05`.
3. It's a **Job**, not a sleeping pod. 1 GPU unless you've shown the job is GPU-bound.
4. Requests and limits are within 20% of each other, and the GPU types are valid (not special-resource names under `gpu.product`).
5. The config and manifest are committed, and the commit hash goes into the experiment's `README.md`.
6. Logs and checkpoints go to the PVC, and the final results get copied back into the repo.
7. No `tolerations:` unless an admin authorized them, and no `sleep` anywhere in a Job's command.
8. About 30–40 minutes after launch, check the [Violations page](https://nrp.ai/userinfo) and the GPU dashboard.
9. Afterwards, delete the Job and pods, and check the GPU dashboard for idle time.

Writeups: any paper or report that used NRP compute includes the AUP acknowledgment above.
