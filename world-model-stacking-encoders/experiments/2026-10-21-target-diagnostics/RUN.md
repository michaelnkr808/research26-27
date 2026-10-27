# Handoff: run the target-side diagnostics (H-E) on the cluster

Planning agent, 2026-10-27. For the implementation agent. Builds on `PLAN.md` (same folder), which is still the task list. The **binding** spec is still `LOG.md` 2026-10-21 (i) as amended by (j). If anything here disagrees with those, LOG wins; stop and ask the user.

## Where things stand
- Code and tests are done (LOG 2026-10-21 (k)): `src/target_diagnostics.py`, `src/test_target_diagnostics.py`. All 7 synthetic checks passed on 2026-10-21.
- verified (2026-10-27, `git rev-parse`): `diagnostics.py`, `target_diagnostics.py` and `test_target_diagnostics.py` at HEAD are byte-identical to commit b68ba54. The folder rename (`stacking-encoders/` → `world-model-stacking-encoders/`) did not change them, so `SRC_GIT_COMMIT` b68ba54 in `job.yaml` is still correct.
- **Try 2 update (2026-10-27, LOG (h)):** try 1 was OOM-killed (LOG (f)). `src/` changed in 52d1226 (lower memory, outputs bit-identical on synthetic data), so `SRC_GIT_COMMIT` is now 52d1226 and the ConfigMap must be **deleted and recreated** from the files at that commit. `job.yaml` now stages features to local disk and requests 14Gi/16Gi memory and 22Gi/26Gi ephemeral storage. Steps below are otherwise unchanged; ask before each cluster step.
- `job.yaml` is drafted and **has never been submitted**. Nothing has run on real features. No target-side number exists yet.
- The user wants this run now (2026-10-27). The mentor's feedback today was to focus on the benchmark (LIBERO) and how stacking encoders relates to it. This run does not change because of that; do not widen scope.

## Read first
`CLAUDE.md`, `README.md`, every file in `../../../_standards/` (especially `05` and `08`), `STANDARDS.md`, `LOG.md` entries 2026-10-21 (i), (j), (k), and `PLAN.md` in this folder.

## Steps
Each step marked **[ask]** needs an explicit yes from the user for that step. A yes to one step is not a yes to the next.

1. **Re-run the tests locally.** `python src/test_target_diagnostics.py` and `python src/test_diagnostics.py`. Both must pass. If either fails, stop and report the output.
2. **kubectl auth check.** Run one kubectl call wrapped so it can't hang (macOS has no `timeout`): `(kubectl get pods -n aria-world & pid=$!; sleep 25; kill $pid)`. If it hangs, ask the user to run `! kubectl get pods -n aria-world` and finish the browser login.
3. **Read-only checks** (no ask needed):
   - Nothing of ours is running in `aria-world` (pods, Jobs, ConfigMaps).
   - `michael-persistent` is not mounted by any pod (it is ReadWriteOnce; the Job will hang in Pending otherwise).
   - PVC `stacking-encoders-features` is Bound and the features are still there (Ceph purge risk; LOG (compliance)). If the D, S or G features are gone, stop: re-extraction is GPU work and out of scope.
   - CPU capacity on us-west non-GPU nodes via public Prometheus (`https://prometheus.nrp-nautilus.io/api/v1/query`), including init containers in the per-pod request sum. Need a node with ≥ 2.4 CPU and ≥ 10 Gi free.
   - Report the results of all four to the user in plain words.
4. **[ask] Create the ConfigMap** `stacking-encoders-target-src` from `world-model-stacking-encoders/src/diagnostics.py` and `world-model-stacking-encoders/src/target_diagnostics.py`.
5. **[ask] Submit** `job.yaml`. Do not change resources, deadline or code unless the user agrees, and log any change before submitting.
6. **Watch.**
   - In the first ~10 minutes: pod phase, node, and memory against the 9 Gi request. If working-set memory passes ~9.5 Gi (limit 10 Gi), tell the user before it gets OOM-killed.
   - For the rest of the run use public Prometheus (`kube_pod_status_phase`, `container_memory_working_set_bytes`, `container_cpu_usage_seconds_total`). kubectl auth can expire after a few hours and then hangs.
   - Expected runtime (speculative, from local sizing): ~3 h, up to ~6 h on slower nodes. Hard cap 8 h.
7. **Collect, within 24 h of finish** (the Job's TTL deletes logs after that):
   - Save the pod log to `results/target-diag-<UTC timestamp>.log`.
   - Cut the 4 JSONs out of the log into `results/target-libero_goal-dinov3-<B>.json`. Check each parses and has the fields `PLAN.md` lists.
8. **[ask] Clean up:** delete the Job and the ConfigMap. Then confirm nothing of ours is left in `aria-world`.
9. **Write `README.md` in this folder.** Same shape as `../2026-10-19-feasibility-tally/README.md`:
   - config, the primary table (B, T0 loss share under a′, T1a, T2 primary interference [95% CI], paper Δ from Table 10), secondary tables, plain observations, caveats, resources (estimated vs actual CPU-hours and peak memory);
   - apply the outcome rule from LOG (i) word for word: R not the largest on T2 primary → H-E fails at this level; R largest → weak support only (a random measure picks R 1 time in 4). Do not argue past it.
10. **Explain the result to the user** in chat, in plain words, before anything goes into findings. The user is newer to this, so:
    - open with what the run tested in one or two sentences;
    - give the 4 primary numbers with CIs, and say which partner came out largest;
    - say plainly whether R was the largest and what the outcome rule says that means;
    - say that probe R² and the paper's success-rate points are different units, and only the orderings are compared.

## Not allowed
- Changing k, r, scalings, the primary measure or the outcome rule after seeing results. Any change is a dated LOG amendment that says it came after the results.
- Editing README key findings (verification gate in `05`). Update only the status line, and ask the user before anything goes into findings.
- Any GPU work, new extraction, a second Job, or reruns with new settings without asking.
- `sleep` in the Job, tolerations, or `priorityClassName`.

## Session end (`05`)
- `LOG.md` entry marked `[agent]`: what ran, every number labelled verified/recalled/speculative, CPU-hours, what was not done.
- README status line updated.
- Commit (`world-model-stacking-encoders: ...`), and leave a concrete Next step.
