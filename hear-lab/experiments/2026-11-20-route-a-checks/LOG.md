# Experiment log: route A bring-up checks

Step-by-step record of running this experiment (rentals, errors, reruns, exact numbers). Append-only, newest at the bottom. The direction's `../../LOG.md` keeps short summaries that point here. Entry template: `_standards/04-documentation.md`.

## 2026-11-20 [agent, experiment lead]
**Goal:** set up the checks and start the check 1 and check 3 code.

**Did:**
- Read-only Vast.ai checks: account `HEAR Lab`, team, credit 200.0 USD; the saved key has `instance_read` and `instance_write` (rights_id 7648542, the team owner key); no instances (`verified`, 2026-11-20).
- `vastai create ssh-key` failed: "Team SSH keys are not supported. SSH keys can only be created in personal context." (error 400). SSH will be attached per instance with `vastai attach ssh`. Nothing changed on the account.
- InteriorGS access: HEAD on `0001_839920/occupancy.json` with the lead's saved login returned 302 (`verified`).
- Offer search (read-only, 2026-11-20): no 8x H100 offer matched; the cheapest 8-GPU offer was 8x H200 at 38.117 USD/h (offer 55940422, reliability 0.997); single H200 offers from 3.514 USD/h; single RTX 3090 from 0.153 USD/h.
- Wrote `PLAN.md` and `config.yaml`, committed before anything runs.

**Next:** workers write the check 1 and check 3 code; then ask the lead for the first rental.

## 2026-11-20 [agent, experiment lead] (worker code back)
- Check 1 code (worker, Opus, branch `hear-lab-route-a-checks-check1`, f81c5a5) and check 3 code (worker, Opus, branch `hear-lab-route-a-checks-check3`, 14afb87 and 577db04 after a Codex review) merged into this branch. Lead reran both done checks: check 1 `Ran 7 tests ... OK`, check 3 `Ran 22 tests ... OK`; all files compile with python3.11 (`verified`).
- Local numbers on scene `0001_839920` from those tests (Mac, small files only): wall share 0.8700 (value 0) and 0.9664 (not passable) over 1100 points; label_ins_id misses 0 of 38; goal position to goal box median about 5 m. Recorded before any rented run; see the PLAN amendment of 2026-11-20.
- test_unseen has 5000 episodes over 839 scenes (local parquet, `verified`).
- PLAN amendment of 2026-11-20 added (field of view, wall-share definition, goal distance, 3.0 m flags, outputs out of git).
