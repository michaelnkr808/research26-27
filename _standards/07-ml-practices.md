# 07 — ML Practices

These are general ML rules that apply to every direction. Topic `STANDARDS.md` files add the specific baselines, metrics, and benchmarks for each direction.

## Evaluation hygiene
- **Keep the test set sacred.** Do all tuning and model selection on the validation set. Touch the test set only for final reporting, and log every time it's used.
- **Check for leakage and contamination.** Look for overlap between train and eval data, duplicates across splits, and features that encode the label. For pretrained or foundation models, consider whether the benchmark may be in their training data.
- **Give baselines a fair shot.** Tune baselines with the same budget and care as the proposed method. Include trivial baselines too: majority class, random, and the "do nothing" option.
- **Run multiple seeds.** Use at least 3 seeds for any reported comparison, and report mean ± std (or a CI). Don't claim one method beats another when the difference is within noise.
- **Run ablations** for every component you claim matters.
- **Use metrics with a reason.** State why each metric was chosen, and report the standard metrics for the benchmark even when you prefer another.

## Experiment tracking
- **Log every run.** Record config, commit hash, seed, data version, metrics, and wall-clock time, using a tracker (W&B, MLflow, or CSV/JSON logs) set in the topic `STANDARDS.md`.
- **Keep failed and killed runs** in the tracker. Tag them, don't delete them.
- **Make configs the source of truth.** Use one config file per run, and no unrecorded command-line overrides.
- **Save checkpoints with metadata**: the config, step, and metrics alongside them. Store them outside git and record their paths in the experiment's `README.md`.

## Sanity checks before scaling
Before any large run:
1. Overfit a single batch, or a tiny subset.
2. Check that the loss at initialization is roughly what you'd expect.
3. Visualize or inspect the inputs and labels actually fed to the model.
4. Run end-to-end on a small scale, including evaluation.

## Compute and cost
- Each direction sets a **compute budget** in its `STANDARDS.md`: GPU-hours, API dollars, or both.
- **Estimate before launching.** Log the estimated vs. actual cost for large runs.
- **Prefer small-scale signal first.** Scale up only once the small version shows the effect.

## Models, datasets, and APIs
- **Record exact versions:** model checkpoint or ID, dataset version or hash, and library versions. For hosted LLM APIs, record the model ID, date, and sampling parameters.
- **Check licenses** for models and datasets before use, and note them in `data/README.md`.
- **Cache API outputs** used in experiments, so results can be reproduced without calling the API again.
