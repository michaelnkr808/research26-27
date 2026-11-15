# 2026-11-15: stage 1 baseline, pretrained-start model with no extra encoder

Stage 1 of decision 006: train the 8-layer model that starts from the pretrained Wan 2.2 TI2V-5B video transformer (blocks 0-7), with no extra encoder, on the paper's clean LIBERO training data, and test it in the simulator on clean LIBERO against a pass bar fixed before training. It answers whether this model works well enough as a robot policy for the encoder-pairing experiments (stage 2) to be built on it.

Status 2026-11-15: design only, nothing run. Not signed off; the pass bar and the other items under "Open, for the user" are the user's decisions. Design in `PLAN.md`.
