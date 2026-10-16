# 08 — Compute on NRP Nautilus

All GPU work runs on the NRP Nautilus Kubernetes cluster. These rules come from two sources, and each one is tagged with its source:

- **[NRP]**: the official docs at <https://nrp.ai/documentation/>, read 2026-10-16. The policies change, so recheck the [Cluster Policies](https://nrp.ai/documentation/userdocs/start/policies) page at least once a quarter, and before any large run. Update this file when they change.
- **[lab]**: lessons learned on this cluster during 2025–26 (see `../research25-26/LangSplat/README.md`, `../research25-26/spring26research_recap.md`, and `../research25-26/notes (mds)/degauss_full_notes.md`). These are field experience and are not in the official docs.

Where the two disagree, the official policy decides what's allowed. A lab lesson can only make a rule stricter.

## Our setup [lab]
- **Namespace:** `aria-world`. It is **shared with the lab**.
- **PVCs:** `scenescript-persistent` (lab-shared, about 40 GB, `rook-ceph-block`, ReadWriteOnce) and `michael-persistent` (personal, 20 GB, `rook-ceph-block`, ReadWriteOnce). Both mount at `/persistent`.
- **Manifests:** `../nautilus-scripts/` has `cp-pod.yml` (CPU-only staging pod), `scenescript-pod.yml` (GPU dev pod), and `pvc-template.yaml`.
- **Base image:** `minarami/scenescript:latest` (PyTorch 2.4, CUDA 12.1).
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

## GPU utilization (where people get in trouble)
- Requested GPUs are reserved for you until the pod is gone, even while they sit idle. **Delete pods as soon as the work is done** [NRP].
- Keep GPU utilization **above 40%, and ideally near 100%**. Check it on the [namespace GPU dashboard](https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus) [NRP].
- Policy text, verbatim: *"A user can not submit more than 4 pods, violating the following conditions: GPU utilization is more than 40% of the requested GPUs. CPU usage is within 20-200% of the requested CPUs. Memory (RAM) usage is within 20-150% of the requested memory."* In other words, every pod is expected to meet these conditions, and you may have at most 4 that don't. Pods requesting ≤ 1 CPU and ≤ 2 GB are exempt [NRP]. Check the Violations page on the Nautilus portal.
- *"Admins … will ban namespaces with under-utilized resources requests"* [NRP]. Our namespace is shared, so one person's idle pod puts the whole lab at risk.
- **Idle-GPU strikes [lab]:** we've been told that idle GPU jobs get killed under a 3-strike rule, with suspension on strike 3, and this account is at **1 of 3**. That strike system is **not** described in the public docs (checked 2026-10-16). Treat it as real anyway.
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
- Some GPUs are **special resources** that `nvidia.com/gpu` won't match: A40 (`nvidia.com/a40`), RTX A6000 (`nvidia.com/rtxa6000`), A100 (`nvidia.com/a100`), H100/H200/GH200, and others. [lab] Two of our old manifests list A40/A6000 under `gpu.product` affinity. That entry probably never matches (speculative), and the last working list left them out.
- **A100:** the namespace quota is 0 by default, so it needs the A100 access form. **H100/H200/GH200:** not user-requestable.
- `priorityClassName: opportunistic` bypasses GPU quota (including A100/H100), **but the pod can be preempted at any time**. Use it only with Jobs that checkpoint often. Don't set any other priority class, because banned classes get rejected at admission.
- You can pin CUDA runtime or driver versions with the `nvidia.com/cuda.runtime.major/minor` and `nvidia.com/cuda.driver.major` labels.
- [lab] GPUs that worked for us: RTX 4090, RTX A5000, RTX 3090, TITAN RTX, RTX A4000, RTX 3080, V100, RTX 2080 Ti, A10.

## Storage [NRP unless tagged]
- **RBD (`rook-ceph-block*`)**: ReadWriteOnce, one pod at a time. Fine for conda/pip envs and compiling. Recommended for data that doesn't need to be shared.
- **CephFS (`rook-cephfs*`)**: ReadWriteMany. Good for large files such as checkpoints. **Installing conda/pip on CephFS is prohibited.** It handles many small files badly, so don't write the same file from several pods at once.
- **Ceph S3**: the right home for large datasets (over 100 GB) and the fastest way in and out of the cluster.
- **Local NVMe** (`emptyDir` + `ephemeral-storage` request): for I/O-heavy training, stage data from Ceph to local disk with an init container. It can be hundreds of times faster than Ceph.
- Pick storage and compute in the **same region**, via the `topology.kubernetes.io/region` affinity.
- **Not archival.** Volumes not accessed for **6 months can be purged without notice**. Keep anything that matters (results, final checkpoints) somewhere else too. `03` already requires results to live in the repo.
- Don't `rm -rf` folders with more than 10k files on Ceph. Use `find DIR -type f -delete`, then `find DIR/* -depth -type d -delete`.
- [lab] **The PVC is shared, so check ownership before deleting anything at the top level.** We once deleted a labmate's `/persistent/scenescript`. Agents: **never delete PVC data without asking**, per the verification gates in `05`.
- [lab] Because the volumes are ReadWriteOnce, a `cp-pod` can't attach while a GPU pod (or a labmate's pod) holds the volume. Tear one down first.
- [lab] Conda caches fill the PVC fast. Set `CONDA_PKGS_DIRS=/tmp/conda-pkgs` so the cache lands on the pod's overlay disk.

## Moving data
- `kubectl cp` is **for small files only**. It goes through the API server, and more than a few MB can hurt the cluster [NRP]. [lab] In practice it drops streams above about 10 MB.
- For large data, have a **Job** pull it in (rclone, wget, scp, or S3), or use Ceph S3 [NRP]. For PVC-to-PVC copies, use `pv-migrate` or a `gsutil -m rsync` Job [NRP].
- [lab] To pull results out, tar them on the pod and copy back one small archive. Do the heavy work on the pod and only fetch the final artifacts.
- Credentials go in Kubernetes Secrets, never in images, manifests, or committed files [NRP; also `04`].

## kubectl gotchas [lab]
- Inside `kubectl exec`, **never** use `conda run -n <env>` (it hangs). Call the env's Python by its full path.
- **Never** run `sed -i` through `kubectl exec` (it hangs). Edit locally, then `kubectl cp` the file.
- **Never** paste multi-line Python into `exec`. Write it to a file, `kubectl cp` it, then run it.
- Set `MPLBACKEND=Agg` on pods, since they have no display.
- Tee long runs to the PVC, for example `... 2>&1 | tee /persistent/logs/run-$(date +%Y%m%d-%H%M%S).log`, so a dropped session doesn't lose the log.
- Large datasets can take minutes to load before printing any progress, so check `nvidia-smi` and the logs before deciding a job has hung.
- [lab] Cross-building CUDA images on the Mac (linux/amd64 under QEMU) has failed for us. Building the env on the pod (on RBD, not CephFS) is the fallback.

## Minimal Job template
This is adapted from the patterns in the NRP docs and has **not yet been run** on our namespace. Fill in the placeholders, check it into `experiments/<run>/`, and commit before submitting (per `03`).

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
7. Afterwards, delete the Job and pods, and check the GPU dashboard for idle time.
