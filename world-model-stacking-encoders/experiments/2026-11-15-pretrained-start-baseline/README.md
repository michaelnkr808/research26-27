# 2026-11-15: stage 1 baseline, pretrained-start model with no extra encoder

Stage 1 of decision 006: train the 8-layer model that starts from the pretrained Wan 2.2 TI2V-5B video transformer (blocks 0-7), with no extra encoder, on the paper's clean LIBERO training data, and test it in the simulator on clean LIBERO against a pass bar fixed before training. It answers whether this model works well enough as a robot policy for the encoder-pairing experiments (stage 2) to be built on it.

Status 2026-11-16: sanity checks 1-3 passed on a Nautilus L40 (overfit one batch, loss at start, inputs), design locked at that run. Measured on the L40: 5.0 s per optimizer step at micro-batch 8 (about 30 L40-hours of steps for the full run), peak 35.8 GiB, fits. Next: sanity check 4 (short training run plus the timing-only 40-rollout smoke), on the user's go. Record in `LOG.md`, design in `PLAN.md`.
