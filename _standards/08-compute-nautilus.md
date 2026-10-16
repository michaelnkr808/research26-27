# 08 — Compute on NRP Nautilus

All GPU work runs on the NRP Nautilus Kubernetes cluster. These rules come from two sources, and each one is tagged with its source:

- **[NRP]**: the official docs at <https://nrp.ai/documentation/>, checked 2026-10-16 against the raw source in the site repo (<https://gitlab.nrp-nautilus.io/prp/nrp-site>, commit `f926199`, 2026-10-15). That covers the [Cluster Policies](https://nrp.ai/documentation/userdocs/start/policies) page, the [AUP PDF](https://nrp.ai/NRP-AUP.pdf), and the code for the portal's Violations page. The policies change, so recheck at least once a quarter and before any large run, and update this file when they change.
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

## Things that get you banned [NRP]
Each item below is quoted from or stated directly in the docs:
1. **A Job running `sleep` or anything else that never ends by itself.** *"Users running a Job with `sleep infinity` command or equivalent (script ending with "sleep") will be banned from using the cluster."*
2. **Tolerating a taint you aren't authorized for.** *"Tolerating a taint you're not explicitly allowed to tolerator by the cluster admins will result in being banned."* By default you may only use nodes with no taints. Never copy `tolerations:` out of someone else's manifest or a dumped live pod spec.
3. **Under-used requests**, at the namespace level: *"Admins monitor resource utilization and will ban namespaces with under-utilized resources requests."* Our namespace is shared, so one person's idle pod puts the whole lab at risk.
4. **Breaking the AUP** (see below): *"Violations of this AUP may result in loss of access to NRP resources."*

Also: **never force-delete pods** (*"Never force Delete Pods."*). A pod stuck in Terminating usually means its node is down or overloaded.

## GPU utilization: how it's actually checked
- Requested GPUs are reserved for you until the pod is gone, even while they sit idle. **Delete pods as soon as the work is done** [NRP].
- Policy text, verbatim: *"A user can not submit more than 4 pods, violating the following conditions: GPU utilization is more than 40% of the requested GPUs. CPU usage is within 20-200% of the requested CPUs. Memory (RAM) usage is within 20-150% of the requested memory."* The wording is awkward, but the portal's code settles it: a GPU pod is flagged when its utilization is **below** the minimum (`gpuMinPercent`, 40% per the docs). CPU and memory are flagged when they fall **outside** their ranges, on either side [NRP, `src/components/vue/userinfo/format.ts`, verified].
- **How the portal checks**, verbatim from the Violations page: *"The portal checks running pods every 10 minutes. Pods younger than 30 minutes are not checked yet."* *"Opportunistic (negative-priority) pods and JupyterHub pods are not checked."* CPU and memory are only checked on pods whose requests are above the exempt thresholds (the docs say 1 CPU and 2 GB; the live values are on the page) [NRP].
- **Your live status:** the [Violations page](https://nrp.ai/userinfo) shows any running pods that are outside the policy and the thresholds currently in force. Check it after every GPU launch.
- **What the docs don't say** (checked 2026-10-16): how long a GPU can be idle before anything happens, over what window utilization is averaged (that's computed by the portal backend, which isn't in the public repo), or that flagged pods get killed automatically. Being flagged is a violation; the published consequences are the 4-pod cap and a possible namespace ban. **There is no documented idle-GPU timeout, so don't rely on one.** A practical reading (speculative): from 30 minutes after startup, every 10-minute check can catch a GPU pod sitting below 40%.
- Keep GPU utilization **above 40%, and ideally near 100%**. Check it on the [namespace GPU dashboard](https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus) [NRP].
- **Idle-GPU strikes [lab]:** we've been told that idle GPU jobs get killed under a 3-strike rule, with suspension on strike 3, and this account is at **1 of 3**. That strike system appears **nowhere** in the public docs, the AUP, or the portal code (all checked 2026-10-16). It's probably enforced by admins or a backend not in the repo (speculative). Treat it as real, and ask NRP support for the exact rule so this line can be replaced with the official one.
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
- **A100:** the namespace quota is 0 by default, so it needs the A100 access form. **H100/H200/GH200:** not user-requestable (*"There is no access form."*). The RTX PRO 6000 Blackwell is reserved and not generally available.
- `priorityClassName: opportunistic` bypasses GPU quota (including A100/H100), **but the pod can be preempted at any time**. Use it only with Jobs that checkpoint often. Don't set any other priority class, because banned classes get rejected at admission.
- You can pin CUDA runtime or driver versions with the `nvidia.com/cuda.runtime.major/minor` and `nvidia.com/cuda.driver.major` labels.
- **CPU-only jobs:** keep them off GPU nodes with node anti-affinity on `feature.node.kubernetes.io/pci-10de.present NotIn ["true"]`. For large CPU jobs, also use `priorityClassName: opportunistic` [NRP].
- [lab] `../nautilus-scripts/pod-michael-pod.yaml` is a **dumped live pod** (it has `status`, `uid`, `nodeName`, and tolerations added by the system). Don't `kubectl apply` it. Use `scenescript-pod.yml` as the template.
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
7. No `tolerations:` unless an admin authorized them, and no `sleep` anywhere in a Job's command.
8. About 30–40 minutes after launch, check the [Violations page](https://nrp.ai/userinfo) and the GPU dashboard.
9. Afterwards, delete the Job and pods, and check the GPU dashboard for idle time.

Writeups: any paper or report that used NRP compute includes the AUP acknowledgment above.
