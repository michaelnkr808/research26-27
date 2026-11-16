# 2026-11-15: stage 1 baseline, pretrained-start model with no extra encoder

Stage 1 of decision 006: train the 8-layer model that starts from the pretrained Wan 2.2 TI2V-5B video transformer (blocks 0-7), with no extra encoder, on the paper's clean LIBERO training data, and test it in the simulator on clean LIBERO against a pass bar fixed before training. It answers whether this model works well enough as a robot policy for the encoder-pairing experiments (stage 2) to be built on it.

Status 2026-11-15 (later): code ready, nothing run. All open items answered (PLAN "Amendments"). Merged: proxy clean tally (PR #35), Fast-WAM reading (PR #36), PLAN amendment (PR #37), `rollout.py` timing-only mode (PR #38), training script `src/pretrained_train.py` with the Fast-WAM action-expert port (PR #39). Next: the sanity-check 1-3 Job on one Nautilus L40/L40S, prepared for the user's go; its first run locks the design. Step-by-step record in `LOG.md`, design in `PLAN.md`.
