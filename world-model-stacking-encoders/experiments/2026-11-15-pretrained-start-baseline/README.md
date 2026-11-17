# 2026-11-15: stage 1 baseline, pretrained-start model with no extra encoder

Stage 1 of decision 006: train the 8-layer model that starts from the pretrained Wan 2.2 TI2V-5B video transformer (blocks 0-7), with no extra encoder, on the paper's clean LIBERO training data, and test it in the simulator on clean LIBERO against a pass bar fixed before training. It answers whether this model works well enough as a robot policy for the encoder-pairing experiments (stage 2) to be built on it.

Status 2026-11-17: all four sanity checks passed (2026-11-16). Full training run (PR #47) finished 2026-11-17 on a Nautilus L40: all 21,696 steps in about 32 h, GPU about 97% throughout, exit 0, no restarts; `model.pt` on the PVC (sha256 in `LOG.md`). The 2,000-rollout clean LIBERO evaluation (PR #48) was submitted 2026-11-17 17:20 UTC on 4 L4s; no outcome yet. Next: tally the evaluation and apply the locked pass bar. Record in `LOG.md`, design in `PLAN.md`.
