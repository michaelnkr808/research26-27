# 2026-11-15: stage 1 baseline, pretrained-start model with no extra encoder

Stage 1 of decision 006: train the 8-layer model that starts from the pretrained Wan 2.2 TI2V-5B video transformer (blocks 0-7), with no extra encoder, on the paper's clean LIBERO training data, and test it in the simulator on clean LIBERO against a pass bar fixed before training. It answers whether this model works well enough as a robot policy for the encoder-pairing experiments (stage 2) to be built on it.

Status 2026-11-15: design only, nothing run. Signed off by the user 2026-11-15 (PR #34): pass bar option 4 (90 and above passes, below 36 fails, in between needs a LIBERO-Plus total of at least 27), action expert derived from the Wan 2.2 weights (Fast-WAM's recipe), training on Nautilus, one baseline seed now. The remaining open items are listed under "Amendments" in `PLAN.md` and are settled before the first cluster run. Design in `PLAN.md`.
