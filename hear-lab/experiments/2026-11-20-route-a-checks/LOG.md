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
