# 2026-11-12: pretrained-start sizing check

A short, capped check of what one training run of the pretrained-start world action model costs: Fast-WAM's joint model cut to 8 of its 30 layers, loaded with the Wan 2.2 TI2V-5B weights, with the input layer widened for one encoder's features, run for a handful of training steps on real cached LIBERO batches on one card, to measure peak GPU memory and seconds per step and turn them into card-hours and dollars per full run. It produces a cost, not a research finding. Design in `PLAN.md`.

Design only. No code, no run, no results yet.
