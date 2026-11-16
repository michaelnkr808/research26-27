# Sanity checks 1-3 on the cluster: submit steps for the lead

Written 2026-11-15 by a worker, Mac only. Nothing here has been submitted or dry-run, and no kubectl command was run to write it. The Job is `jobs/job-sanity.yaml` (Job `stacking-encoders-baseline-sanity`); its header has what it runs, every resource number's source and the deadline arithmetic. Paths are relative to this experiment folder unless they start with `src/` or `world-model-stacking-encoders/`. `<scratch>` is the lead's session scratchpad. `$C` is the source commit named in step 2.

**This Job is PLAN "Order of work" step 3. Submitting it starts the experiment, so the design locks** (PLAN header). Submit only with the user's go in the lead's window. It needs 1 GPU (L40 or L40S), about 0.5 h of pod time (speculative), and writes about 6.7 GB to the PVC (step 13).

## Before submitting

1. **Record the overfit pass mark in this folder's `LOG.md`** before the Job is created (PLAN sanity check 1: "Pass mark set by the implementation session before running it"). Proposed mark and its source are under "Pass and fail" below; the lead confirms or changes it, then writes it in the LOG, then submits.

2. **Name the source commit.** After this branch's PR is merged and pulled, on a clean `main`: `git rev-parse HEAD` gives `$C`. The pod reads only the ConfigMap, never git, so the ConfigMap must hold `$C`'s files exactly.

3. **Update the ConfigMap `stacking-encoders-proxy-src` in place** (replace, never delete: the pattern of proxy LOG 2026-11-06, line 455, which keeps keys only the ConfigMap holds, such as `pilot-subset.json`).
   The Job needs 7 keys (the import chain of `src/pretrained_train.py`, checked by grep: `pretrained_train` imports `action_expert_init`, `pretrained_sizing`, `proxy_wam`, `diagnostics`; `proxy_wam` imports `t5_cache`; plus `setup_env.sh` for the pip environment):
   - new keys: `pretrained_train.py`, `pretrained_sizing.py`, `action_expert_init.py`
   - existing keys: `proxy_wam.py`, `diagnostics.py`, `t5_cache.py`, `setup_env.sh`

   Steps:
   1. Who else mounts it: `kubectl get jobs,pods -n aria-world -o json > <scratch>/ns.json`, then list every Job or pod whose `spec.template.spec.volumes` (or pod `spec.volumes`) has `configMap.name == "stacking-encoders-proxy-src"` (for example `jq -r '.items[] | select([(.spec.template.spec.volumes // .spec.volumes)[]? | .configMap.name] | index("stacking-encoders-proxy-src")) | .kind + " " + .metadata.name + " " + (.status.phase // "")' <scratch>/ns.json`). Running pods see a replaced ConfigMap's files within about a minute (the kubelet syncs configMap volumes that are not mounted by subPath; recalled), and Pending pods start with the new files. The rollout Jobs import `proxy_wam.py` and `diagnostics.py` from `/src`, and the perturbed-frames Jobs use `diagnostics.py`, `extract_features.py` and the env-bundle scripts from it (`experiments/2026-11-06-perturbed-frames/jobs/PHASE-B.md` line 35). **If any Job or pod of ours that mounts it is Pending or Running, stop and ask** (the alternative is a separate ConfigMap for this Job, see the end of this file).
   2. Save the live one: `kubectl get configmap stacking-encoders-proxy-src -n aria-world -o json > <scratch>/proxy-src-before.json`.
   3. Write every key of it into an empty staging dir `<scratch>/proxy-src-stage/` (one file per `.data` key, bytes as stored; for example with a short Python script that reads the JSON and writes `data[k]` to `stage/k`; check `.binaryData` is absent or copy it too). Record the sha256 of every key (`shasum -a 256 <scratch>/proxy-src-stage/*`).
   4. Overwrite or add the 7 keys from `$C`: for each file `f` above, `git show $C:world-model-stacking-encoders/src/$f > <scratch>/proxy-src-stage/$f`. Do not touch any other key (in particular `extract_features.py`, `setup_sim.sh`, `env_bundle.sh`, `rollout.py`).
   5. Compare: sha256 of every key before and after. Expected changes: the 3 new keys, and `proxy_wam.py` (main's `proxy_wam.py` is not the file RETRAIN.md recorded at `cb9e329`, `7169332991...`, so the live copy is probably older; not checked on the cluster). `diagnostics.py`, `t5_cache.py`, `setup_env.sh` may or may not change; list each one that does in the LOG. **Two stop checks**, because other Jobs abort on them:
      - `cat setup_env.sh setup_sim.sh env_bundle.sh | shasum -a 256` in the stage dir must still start `9182f8131559` (the env-bundle key the rollout and perturbed-frames Jobs restore; at `6875f53` the repo's three files give `9182f8131559e453dc41c0c3408383e77d84029fd6737efad8c1e8d9595a2096`, verified on the Mac). If not, stop.
      - `extract_features.py` unchanged (its sha256 prefix `03734ba84a86` is what the perturbed-frames Jobs require; this procedure never writes it).
   6. Build and dry-run: `kubectl create configmap stacking-encoders-proxy-src -n aria-world --from-file=<scratch>/proxy-src-stage --dry-run=client -o yaml > <scratch>/proxy-src-new.yaml`; `kubectl replace --dry-run=server -f <scratch>/proxy-src-new.yaml`; then `kubectl replace -f <scratch>/proxy-src-new.yaml`. (Total size of the keys is far under the 1 MiB ConfigMap limit: the 16 `src/` files that have been in it sum to 291,568 bytes at `6875f53`.)
   7. Verify each of the 7 keys against `$C`: `kubectl get configmap stacking-encoders-proxy-src -n aria-world -o jsonpath='{.data.pretrained_train\.py}' | shasum -a 256` against `git show $C:world-model-stacking-encoders/src/pretrained_train.py | shasum -a 256`, and the same for the other six (escape the dot as `\.`). All 7 must match. For reference, main's files at `6875f53` (verified on the Mac, `shasum -a 256`; recompute at `$C`, they differ if `src/` changed since):
      - `pretrained_train.py` `e79ad083f8ac02af236e78de74c5b85af7271f0c86356d59086198973cfe6473`
      - `pretrained_sizing.py` `c14ccec63e66075c561573c315ed11a0abf3ce5e0fabd3751dbb599f64a52bb7`
      - `action_expert_init.py` `32e325315b766d601ac0fe60cb79e1c3cbf60caee96cd460682bf63c2431c3cf`
      - `proxy_wam.py` `e09543d5bf0bf8ce2a7d1db9048f8c7819366f97fb4ae0c552e802412e32fd0a`
      - `diagnostics.py` `7af1f7aa2d595d9e0b7f1fb462f7901c6fa9594d2da7895b0168f0ef3ea2d70d`
      - `t5_cache.py` `61181a74af8825eff0e0b8cb3a220f8483d9f1982642cd90b753cb9e5a87d708`
      - `setup_env.sh` `880c47cbe741b7fbcd98f44f85e7a1d14747d6c837e0493f4642e5ebbc18ace8`
      The pod also prints the sha256 of these 7 files as its first lines after `nvidia-smi`, so the record is in the log.

4. **Render** with the commit filled: `sed "s/FILL_AT_SUBMIT/$C/" jobs/job-sanity.yaml > jobs/rendered/job-sanity.yaml` (never edit the original for one run), and commit the rendered file as submitted. `TRY` stays `"1"` for the first submit; for a resubmit after a failed or deleted attempt, render with `TRY` set to `"2"`, `"3"`, ... (the pod aborts with exit 11 if `/mnt/proxy/runs/baseline-sanity-none-s0-try<TRY>` already exists, so an earlier attempt's partial files are never reused).

5. **GPU count:** the GPUs our Jobs hold or are Pending for, plus 1, must be at most 8 (`STANDARDS.md` line 49). The PVC needs about 7 GB free for `model.pt` (it had 220G free on 2026-11-15, `data/README.md`; the pod prints `df -h` at start).

6. **Server-side dry run, then submit:** `kubectl create --dry-run=server -f jobs/rendered/job-sanity.yaml`, then `kubectl create -f jobs/rendered/job-sanity.yaml`. Note the creation time in the LOG. The deadline (75,600 s, 21 h) counts from now, queue included.

## While it runs

7. **One blocking wait** (complete OR failed), started in the background, no polling loop. macOS bash 3.2 has no `wait -n`, so the two `kubectl wait` calls race through a FIFO and the loser is killed:
   ```
   bash -c 'J=job/stacking-encoders-baseline-sanity; T=76000s; D=$(mktemp -d); mkfifo "$D/f"
   ( kubectl wait "$J" -n aria-world --for=condition=complete --timeout=$T >/dev/null 2>&1; echo "complete rc=$?" > "$D/f" ) & A=$!
   ( kubectl wait "$J" -n aria-world --for=condition=failed --timeout=$T >/dev/null 2>&1; echo "failed rc=$?" > "$D/f" ) & B=$!
   read -r R < "$D/f"; echo "first: $R"; pkill -P $A; pkill -P $B; kill $A $B 2>/dev/null; rm -rf "$D"
   kubectl get "$J" -n aria-world -o jsonpath="{range .status.conditions[*]}{.type}={.status} {.reason}{\"\n\"}{end}"'
   ```
   `first: complete rc=0` = the Job completed; `first: failed rc=0` = it failed (look at the reason line: `BackoffLimitExceeded` = the pod failed, `DeadlineExceeded` = out of time). A non-zero `rc` means that `kubectl wait` itself stopped (timeout, or an expired OIDC login, which makes kubectl hang or error: re-login and rerun the wait). The timeout is the deadline plus about 7 min. A wait this long outlasts an idle session: if it will run more than about 50 min, write the handoff note first (user's global rule).

8. **The 15-minute GPU-use check** (`STANDARDS.md` line 48; the lesson "check GPU util 15 min into any new job type"). About 15 min after the pod is Running: `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-sanity -o name`, then `kubectl logs <pod> -n aria-world | grep -E '^=== PHASE|^\[gpu\]|^\[train' | tail -40`. What to expect (speculative): the `setup` and `stage-wan` phases at about 0% (pip install and the Wan copy, about 13 to 20 min), then high use in `first-batch` and `overfit`. What counts:
   - Still in `setup` after 30 min of Running: pip is stuck; look at the log, tell the user.
   - In `overfit`, the `[gpu]` samples should be well above 40%; the phase lasts only a few minutes. If they are under 40% there, note it in the LOG (it bears on the full run, which uses the same step code) and tell the user; the pod ends by itself, no action needed.
   - `device=cpu` or a CUDA error in the log: delete the Job (memory note: a GPU fault can make a script fall back to CPU).
   If the pod is still Running at 30 to 40 min, check the Violations page (`_standards/08` pre-flight item 8). The Prometheus alternative is the RETRAIN.md query with `pod=~"stacking-encoders-baseline-sanity-.*"`.

## After it ends

9. **Save the pod log right away** (container logs vanish within about a day; the Job's ttl is 24 h): `kubectl logs <pod> -n aria-world > results/sanity/<pod>.log` (the pod, not `job/`), and `kubectl get job stacking-encoders-baseline-sanity -n aria-world -o yaml > results/sanity/job-status.yaml`. The same log is on the PVC as `/mnt/proxy/logs/baseline-sanity-none-s0-try<TRY>-<date>.log`, the GPU samples as `...-gpu-<date>.csv`, the pip freeze as `/mnt/proxy/logs/pip-env-baseline-sanity-none-s0-try<TRY>.log`.

10. **Copy the result JSONs off the PVC** (small, so they travel in the pod log; no reader pod). The pod's EXIT trap prints each of `first-batch/config.json`, `first-batch/load.json`, `first-batch/first-batch.json`, `first-batch/train.json`, `overfit/overfit.json` (those that exist) between `=== FILE <name> sha256 <hex> bytes <n>` and `=== END <name>`. Extract each from the saved log and check it:
    ```
    L=results/sanity/<pod>.log
    for f in first-batch/config.json first-batch/load.json first-batch/first-batch.json first-batch/train.json overfit/overfit.json; do
      o=results/sanity/$(echo $f | tr / -)
      printf '%s' "$(awk -v f="$f" '$0 == "=== END " f {p=0} p {print} index($0, "=== FILE " f " ") == 1 {p=1}' $L)" > $o
      grep "^=== FILE $f " $L; shasum -a 256 $o; wc -c < $o
    done
    ```
    A file the pod never wrote has no `=== FILE` line and comes out empty; delete those. The sha256 and byte count of each extracted file must equal the `=== FILE` line's (the files are written without a trailing newline, and `printf '%s' "$(...)"` drops the one the dump adds; the recipe was tested on the Mac on a fake log in this format and the extract was byte-identical). If one differs, or the log was cut, ask the user before creating a CPU reader pod to copy the files. `model.pt` (about 6.7 GB) and `history.jsonl` stay on the PVC.

11. **Judge checks 1 to 3** against "Pass and fail" below and write the outcome in this folder's `LOG.md` (the lead's entry; the JSONs and the log are committed under `results/sanity/`).

12. **Delete the Job** once the log is saved and the JSONs are extracted: `kubectl delete job stacking-encoders-baseline-sanity -n aria-world` (graceful; never `--force`). Recheck `kubectl get pods -n aria-world -l job-name=stacking-encoders-baseline-sanity` shows nothing left.

13. **PVC files left behind:** `/mnt/proxy/runs/baseline-sanity-none-s0-try<TRY>/` holds `first-batch/model.pt` (about 6.7 GB, from the one optimizer step) plus the small files. Deleting it is data deletion and needs the user's OK (`../../../_standards/05` gates); say so in the LOG.

## Exit codes (pod log, last line `=== exit <rc>`)

| rc | Meaning | What to do |
|---|---|---|
| 0 | both phases ran | judge the checks |
| 2 | non-finite loss or gradient norm (`STOP:` line; `first-batch.json` or `overfit.json` holds `"stopped"`) | sanity fail; stop, tell the user |
| 1 | `setup_env.sh` failed (last 50 pip lines printed) or a Python error (traceback) | diagnose; resubmit with the next `TRY` after the user's OK |
| 10 | `SRC_GIT_COMMIT` not filled | re-render (step 4) |
| 11 | out dir exists | re-render with the next `TRY` |
| 12, 13 | Wan staging size or sha256 mismatch | stop; the PVC copy of the Wan files may be damaged; tell the user |

A Python exception other than the non-finite stop also exits 1 (traceback in the log); a pod `OOMKilled` (exit 137) means host memory: raise the memory request from the log's evidence and ask before resubmitting.

## Pass and fail

Any failed check is a stop: report to the user, no sanity check 4 until the user decides.

**Check 2, loss at start** (PLAN: "Repeated on the full data's first batch and logged; a non-finite loss stops"). Pass: `first-batch.json` exists with `loss_at_start.model` (`total`, `vae`, `action`) and `loss_at_start.predict_zero` all finite. The PLAN sets no numeric bar. Log them next to the sizing run's values on its batch of 16 (model total 3.735, VAE 2.634, action 1.101; predict-zero 2.526, VAE 1.492; sizing LOG line 38). They are not expected to match: this batch is 128 windows of the full data and the action expert is now derived from the Wan blocks, not fresh. A large gap is something to look at and tell the user, not a fail by the PLAN.

**Check 3, inputs inspected** (PLAN: "shapes, ranges, padding counts, task strings"). Pass when all hold:
- The first `[train ...]` line says `1712 episodes, 277713 frames, 40 tasks, in_dim 48; plan 21696 optimizer steps of 128, this run to step 1, micro-batch 8 x 16` (episodes, frames: `data/README.md` section 3; steps: PLAN recipe table; 40 = the paper's 40 tasks, Table 7, assuming the 40 instruction strings are all distinct, which is not checked here: the count is of distinct strings).
- `first-batch.json` `inputs`: `latents_host` shape `[128, 3, 14, 28, 48]` float16; `action` `[128, 32, 7]`; `action_is_pad` `[128, 32]`; `latent_is_pad` `[128, 2]`; `proprio` `[128, 8]`; `task` `[128]` (`proxy_wam.Data.assemble`).
- `action_range` and `proprio_range` inside [-1, 1] and both `*_abs_gt1_fraction` 0 (min/max normalization over the same 1,712 episodes maps every value into [-1, 1], `proxy_wam.MinMax`; anything outside is a normalization error).
- `channels/vae`: finite, `std_range` minimum above 0.
- `tasks`: real LIBERO instructions (read them), `n_distinct_tasks` above 1.
- Padding: `windows_with_padding` below 128 (some windows near episode ends are padded, not all).
- `load.json`: `all_loaded_diffs_zero` true; `counts_by_tag["loaded"]` 231 tensors (as the sizing run's `load-none.json`); the derived and fresh counts as the action-expert recipe says (`counts.derived`, `counts.fresh`).

**Check 1, overfit one batch. Proposed pass mark** (the lead records it in the LOG before submitting; PLAN sanity check 1 says it is set "from the proxy's check 9 (proxy PLAN line 137)"). From `overfit.json`, with `first_20_mean` and `last_20_mean` (means of the first and last 20 of the 300 steps, keys `loss`, `loss/vae`, `loss/action`) and `predict_zero` (keys `total`, `vae`, `action`, the same batch and draws):
1. `last loss/action < 0.1 x first loss/action`. Source: proxy check 9's action criterion, word for word (`src/test_proxy_wam.py`, check 9: "the action loss over the last 20 of 200 steps is below a tenth of the first 20").
2. Every term ends lower than it starts: `last[k] < first[k]` for `loss`, `loss/vae`, `loss/action`. Source: check 9's second criterion ("every other loss term ends lower than it starts").
3. `last loss < predict_zero total`, and `last loss < 0.5 x first loss`. Source: the PLAN's own words for this check ("the training loss falls well below its starting value and below the predict-zero baseline on that batch"); the 0.5 is the proxy's original check 9 criterion (total loss halves), which the proxy dropped only because its synthetic VAE latents were i.i.d. noise that a 0.46M-parameter test model could not memorize (check 9's history, proxy LOG line 500). Neither reason holds here (real latents, a 1.68B-parameter model; speculative that it then halves).
4. No `STOP:` (exit 2).
Pass = all four. A fail on 3 alone with 1 and 2 passing is still a fail under this mark; the lead may decide to weaken 3 before submitting (for example to "below predict-zero" only), but not after seeing the result (`../../../_standards/05`, no silent goalpost moves).

What is not tested here: whether the model learns on the full data (sanity check 4 and the full run), step time on an L40 (only the one optimizer step's time is in `history.jsonl`; the full run's first-minutes projection measures it), and rollouts.

## Alternative for step 3, if another Job mounts the ConfigMap

Create a separate ConfigMap with only the 7 keys from `$C` (for example `stacking-encoders-baseline-src`, `kubectl create configmap ... --from-file=<f>=<path>` for each), verify the 7 sha256 as in step 3.7, and change the one line `configMap: {name: stacking-encoders-proxy-src}` in the rendered Job to it. This leaves every other Job's `/src` untouched. It is a change to the brief (which names `stacking-encoders-proxy-src`), so the lead decides.
