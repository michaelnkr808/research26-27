# Check 1 follow-up: measured HRTF (MIT KEMAR)

What this answers, in one breath: in check 1 the left-ear-first test failed because the library's default head model changes the loudness between the ears but barely the arrival time. This run swaps in a measured head (MIT KEMAR dummy head, normal pinna, SOFA file) and asks whether the arrival-time difference appears and whether the check 1 left-ear test then passes.

Code: `hrtf_followup.py` (uses `ear_stats` and `fmt` from `diagnose_left_ear.py`, the pass test from `pass_tests.py`), `rlr_binding.py` (new `set_listener_hrtf`, calls `RLRA_SetListenerHRTF`, header line 342; never called for the default condition). Do not commit anything this run writes (see RUN.md).

Prerequisite: a rented Linux x64 machine set up as in RUN.md steps 1 to 4 (packages, repo at the commit the lead names, pinned library at `4fd446b`, venv, data). Shell in `/workspace`, venv active.

## 1. Get the SOFA file and record its hash

    mkdir -p /workspace/hrtf && cd /workspace/hrtf
    curl -fL -o mit_kemar_normal_pinna.sofa https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa
    ls -l mit_kemar_normal_pinna.sofa
    file mit_kemar_normal_pinna.sofa
    sha256sum mit_kemar_normal_pinna.sofa | tee /workspace/hrtf/sofa.sha256

The URL is `verified`: it downloaded on 2026-11-20 (1171764 bytes, HDF5, sha256 `e7035994f5fd754058424c061380ee92b1d5ed58fccef2887a4266916616acdf`); the downloaded file here should match that size and hash. `file` should say "Hierarchical Data Format (version 5)".

Terms: MIT's KEMAR page says the measurements are "made available to the research community"; no formal licence was found. Cite Gardner and Martin 1994. If it says HTML or the file is tiny, the download failed: stop and report. Paste the sha256 into the report; the script logs it again.

## 2. Rerun check 1 to regenerate the run directory (new output folder)

Exactly as RUN.md step 6, with a new folder name (`--out` must not exist):

    cd /workspace/research26-27/hear-lab/experiments/2026-11-20-route-a-checks/src/check1
    REPO_COMMIT=$(git -C /workspace/research26-27 rev-parse HEAD) \
    python run_check1.py --rlr-root /workspace/rlr-audio-propagation \
      --scenes-root /workspace/data/InteriorGS --parquet <path printed by find in RUN.md step 4> \
      --scene 0001_839920 --out ../../results/check1-run2 2>&1 | tee /workspace/check1-run2-console.txt

Expect the same as run 1: `FAIL left_ear_before_right_ear` in both settings sets (the default HRTF). If it now passes, say so: the run is not bit-identical across machines in every respect, and that is a finding.

## 3. Run the follow-up

    python hrtf_followup.py --help
    python hrtf_followup.py --rlr-root /workspace/rlr-audio-propagation \
      --scenes-root /workspace/data/InteriorGS --parquet <same parquet path> \
      --run-dir ../../results/check1-run2 --hrtf /workspace/hrtf/mit_kemar_normal_pinna.sofa \
      --out ../../results/hrtf-followup1 2>&1 | tee /workspace/hrtf-followup1-console.txt

CPU only. Per settings set it runs 4 simulations (default and kemar, each free-field sweep and left test); the sweeps are quick (no mesh, direct only), the left tests take as long as in check 1. `--out` must not exist.

## 4. Copy back before the machine is destroyed

`results/hrtf-followup1/` (log `hrtf_followup.log`, `hrtf_followup_summary.json`, the `.npy` impulse responses), `/workspace/hrtf-followup1-console.txt`, `/workspace/hrtf/sofa.sha256`, and `results/check1-run2/` if it differs from run 1.

## What normal output looks like

The log header lists the SOFA sha256, the library sha256, the config hash, every settings set and the reproduction-check note. Then, per settings set and condition, one `[variant]` line (with every configuration override and the HRTF used; for kemar a `[rlr] RLRA_SetListenerHRTF(...)` line before each simulation), then one line per impulse response in the `diagnose_left_ear.py` format (onsets per ear, peaks, `xcorr R-L` in ms, `ILD` in dB), each followed by a `[test]` or `[info]` PASS/FAIL line from the unchanged check 1 left-before-right function. `[test]` lines are `az090` (listener's left, expected left ear first) and `left_test` (the check 1 placement); `[info]` lines are the other azimuths, where the left ear is not expected first. A `[summary]` block ends the log. Any `RLRA_SetListenerHRTF` error is logged as `[variant] ... failed` and the other variants still run: report it as a finding.

- Default condition, free field: should reproduce the earlier diagnostic: a level difference, delays of largest magnitude 0.091 ms over all azimuths (that earlier result was measured in this project, check 1 diagnosis of 2026-11-20).
- KEMAR condition, free field: a physical head gives an interaural time difference of roughly 0.6 to 0.7 ms at +-90 degrees, with the left ear first for +90 (`recalled` textbook figure, not measured here), so `xcorr R-L` near +0.6 to +0.7 ms at `az090` and the opposite sign at `az270`. If the value is much smaller, the library may be ignoring or resampling the file: report the numbers, do not tune anything.
- The `left_test` line is the check 1 question: PASS means the left ear's first sample at 0.1 x its own peak comes before the right ear's with the measured head in the full scene.
- A KEMAR run is one measured head, not the library default; a PASS here does not change the check 1 result for the default HRTF, it only says whether switching HRTF is a fix.
