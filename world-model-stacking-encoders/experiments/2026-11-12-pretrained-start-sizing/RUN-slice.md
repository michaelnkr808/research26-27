# RUN-slice: copy the 16-episode data slice off Nautilus (sizing check, PLAN Step 3)

Prepared 2026-11-12 by a worker. Nothing here has been run. Every `kubectl` line below changes the cluster except the read-only ones marked so; the user (or the implementation session with the user's OK in its window) runs them, per `memory: ask before new cluster work`. Paths are relative to `world-model-stacking-encoders/` unless absolute.

What it does, in plain words: a small CPU-only pod reads 16 training episodes out of the proxy's cache, copies them (VAE latents, DINOv3 features and their stats, T5 files, a meta file for only those episodes) into a folder on the pod's own disk, packs that into a tar cut into ~84 MB pieces with a checksum list, and waits 2 hours. You pull the pieces one at a time to your Mac, check each checksum, glue them back together and delete the pod. The pod only reads the PVC.

Facts the plan rests on (speculative until the run prints them): the slice is about 2.5 GB with DINOv3 (PLAN Step 3 arithmetic), so about 30 pieces. The script prints the chosen episode ids and the true size. Z-score stats: the slice carries each episode's own `.stats.npy`, so `proxy_wam.Data` computes them over these 16 episodes only (the accepted deviation in PLAN Step 3).

Encoder: the pod copies DINOv3 only (`ENCODERS=dinov3` in `pod-slice.yaml`; the Open item 3 choice of encoder is not made here). For another encoder, edit that env value in a rendered copy (comma-separated for several) and raise `ephemeral-storage` and the `emptyDir` size as the comment in the yaml says.

Why a Pod and not a Job: the copy needs the files to stay up after the work, and `_standards/08` bans `sleep` in a Job but allows it in a bare Pod (6 h cap; this one has a 2.5 h `activeDeadlineSeconds`).

## 0. Commit first
Commit the branch with `src/make_sizing_slice.py` and `pod-slice.yaml`, and note `git rev-parse HEAD` for the LOG. Run the local test once more: from `src/`, `/opt/homebrew/bin/python3.11 test_make_sizing_slice.py` (expects `all ok`).

## 1. Check (read-only)
```
kubectl get pods,pvc,configmaps -n aria-world
```
Expect PVC `stacking-encoders-proxy` Bound, and no pod `stacking-encoders-sizing-slice`. If kubectl hangs or shows an x509 error, the login expired (`memory: kubectl auth expiry`); log in again first.

## 2. ConfigMap with the script (cluster change)
```
kubectl create configmap stacking-encoders-sizing-slice-src -n aria-world --from-file=src/make_sizing_slice.py
```

## 3. Submit (cluster change)
```
kubectl apply -f experiments/2026-11-12-pretrained-start-sizing/pod-slice.yaml
```
(Add `--dry-run=client` first if you want a syntax check only.) Start-up is the image pull plus scheduling, about 8 minutes on Nautilus in past runs (recalled).

## 4. Watch (read-only, check by hand, no loop)
```
kubectl get pod stacking-encoders-sizing-slice -n aria-world
kubectl logs stacking-encoders-sizing-slice -n aria-world
```
Wait for the log line `=== READY ...`. Before it, the log shows the chosen episodes (`chosen episodes (suite, id, length): ...`), the suites in the slice, the frame count and the slice size. Save that log into the experiment's results now:
```
kubectl logs stacking-encoders-sizing-slice -n aria-world > experiments/2026-11-12-pretrained-start-sizing/results/slice-pod.log
```
(Create `results/` if it is not there.) Stop and report if the log shows an `AssertionError` (a missing file or a non-empty output folder) or the pod is `Error`/`OOMKilled`.

## 5. Copy the pieces, one at a time, each checked
`kubectl exec ... cat` streams one file; copies of one big file have broken at about 100 MB before (recalled, earlier sessions), which is why the pieces are ~84 MB and each is verified. Run in zsh on the Mac:
```
mkdir -p sizing-slice && cd sizing-slice
kubectl exec stacking-encoders-sizing-slice -n aria-world -- cat /work/chunks/SHA256SUMS > SHA256SUMS
kubectl exec stacking-encoders-sizing-slice -n aria-world -- cat /work/chunks/tar.sha256 > tar.sha256
for f in $(awk '{print $2}' SHA256SUMS); do
  for try in 1 2 3; do
    kubectl exec stacking-encoders-sizing-slice -n aria-world -- cat /work/chunks/$f > $f
    want=$(grep " $f\$" SHA256SUMS | awk '{print $1}')
    got=$(shasum -a 256 $f | awk '{print $1}')
    [ "$want" = "$got" ] && { echo "ok $f"; break; }
    echo "BAD $f try $try"
  done
done
shasum -a 256 -c SHA256SUMS
```
The last line must print `OK` for every piece. A piece that fails 3 tries: stop and report which one. Resume by rerunning the loop (it fetches everything again; to skip finished pieces, add `[ -f $f ] && continue` after the `for f` line only if you trust them, the final `shasum -c` re-checks all).

## 6. Reassemble and verify
```
cat slice.tar.part* > slice.tar
shasum -a 256 slice.tar            # must equal the hash inside tar.sha256
cat tar.sha256
mkdir slice && tar -xf slice.tar -C slice
(cd slice && shasum -a 256 -c MANIFEST.sha256)     # every file OK
cat slice/slice.json | head -c 1500
```
The two tar hashes must match, and `MANIFEST.sha256` lists the sha256 of every file in the slice (each one hashed on the pod right after it was copied). The slice loads as `proxy_wam.Data(slice, <suites_in_slice from slice.json>, ["dinov3"], slice/t5)`. Record the folder size (`du -sh slice`) and, for the LOG, the chosen episode ids from `slice.json`. Delete `slice.tar` and the pieces once `slice/` is verified if disk is tight.

## 7. Clean up (cluster change)
```
kubectl delete pod stacking-encoders-sizing-slice -n aria-world
kubectl delete configmap stacking-encoders-sizing-slice-src -n aria-world
kubectl get pods,configmaps -n aria-world
```
The pod also ends by itself after about 2 hours of sleep (and by the 2.5 h deadline), but delete it as soon as the copy is verified so it does not sit idle. Nothing was written to the PVC, so nothing there needs cleaning. Save the pod log before deleting (step 4); container logs vanish within about a day.

## If something goes wrong
- Pod Pending a long time: `kubectl describe pod stacking-encoders-sizing-slice -n aria-world` (read-only) and read the scheduling message; the `us-west` and no-GPU-node constraints are the only ones set.
- Evicted for ephemeral storage: the slice or chunks exceeded the limit (more or longer episodes than the arithmetic, or several encoders). Raise `ephemeral-storage` and the `emptyDir` `sizeLimit` in a rendered copy and resubmit; the choice of episodes does not change (same seed, same pool).
- Pod ended before the copy finished: resubmit; it rebuilds the same slice (same seed, same files).
