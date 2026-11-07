# Plan: perturbed-frame test (frozen features, no policy training)

**Status:** DRAFT, not signed off by the user; nothing runs before sign-off.

Written 2026-11-06 by a planning agent from docs and source reading only. Labels: `verified` (checked in a repo file this session), `recalled` (from memory, not checked), `speculative` (a guess or a proposal). Direction agenda item: `../../PLAN.md` Next 3.

## Question
Under the picture changes that LIBERO-Plus applies, does adding C-RADIOv4 (R) to DINOv3 (D) help less, or hurt more, than adding SigLIP 2 (L), SAM 3 (S) or VGGT-Ω (G)? The paper's D+R drop is measured under LIBERO-Plus perturbations (`literature/liu2026what-table10-composition.csv`; `../../README.md`, Key findings), while our frozen-feature tests used clean frames only (verified, `../2026-10-19-feasibility-tally/README.md`). On clean frames R adds the most action information on top of D (gain +0.037 vs +0.014 to +0.033 for the others; verified, tally README). This test asks whether that changes under perturbation.

Why look at the pair and not each encoder alone: C-RADIOv4 alone holds up about as well as DINOv3 in the paper's per-category Table 1 (camera 62.6 vs 64.2, light 97.3 vs 98.5, background 86.2 vs 83.9, noise 86.6 vs 86.0; verified, `literature/liu2026what-table1-single-source.csv`). The paper gives no per-category numbers for compositions (verified, paper note). So the harm, if it is visible at all in frozen features, has to show up in the pair.

## Decided by the user (2026-11-06, in the lead session's chat)
- **Perturbation types: camera, lighting, background, sensor noise only.** User: "yeah lets just do the 4 picture types i guess". Left out, with the lead's reasons: robot initial state (changes the robot's state, so a frozen demo moment is no longer the same moment); language (the image does not change); object layout (moves the target or adds physical distractors; the distractor-only half was offered as an optional fifth and the user declined it).
- **Measurements: both, with the probe as the main result.** User: "yea with the probe as the main result".
  - Main: an action probe trained on clean frames and tested on perturbed re-renders of the same demo moments. The demo's recorded action stays the correct label because the robot state is unchanged. This mirrors the paper: train on clean LIBERO, evaluate on LIBERO-Plus.
  - Secondary sanity check: feature drift, meaning how far each encoder's features move between the clean and the perturbed version of the same frame, on a scale that can be compared across encoders.
- Context the lead gave the user: C-RADIOv4 alone holds up about as well as DINOv3 per category (above), so the test looks at the pair. With only 4 partners, D+R ranking last by chance is 1 in 4, so a hit is a hint, not proof (`../../PLAN.md` Next 3 says the same).

Everything below is a proposal and needs the user's OK.

## Proposed design (needs the user's OK)

### Frames: which data
Two candidates:
- **libero_goal HDF5** (`yifengzhu-hf/LIBERO-datasets`). The tally and diagnostics used it, so the clean probes, the demo-level split (100 test demos) and the cached D, L, R, S, G features already exist and the clean numbers are directly comparable (verified, `../../data/README.md` sections 1 and 4a). Images are 128x128.
- **Fast-WAM's `yuanty/LIBERO-fastwam`** (what the proxy round and the paper use). More faithful to the paper (512x512 frames, the paper's 1,712 episodes) but the frozen-feature probe results for it do not exist, and no D, L, R, S, G probe has been run on it (verified, data README sections 3 and 4b: only D, R and L caches exist there, no S or G).
- **Recommendation (speculative): libero_goal HDF5**, same train/val/test split as the tally. Reason: the whole test is a comparison against our own clean results, and S and G caches exist only for this data. Fidelity to the paper's data matters less here because no policy is trained.
- **Simulator states:** not settled by the repo for either dataset. The libero_goal contents list in the data README names images, actions, rewards and a language instruction per file, with no simulator states (verified, data README section 1, "Contents", checked on one file). Original LIBERO files are, from memory, supposed to carry per-step `states` and `robot_states` and a scene-file attribute (recalled, unverified). Fast-WAM's dataset lists images, an 8-d `observation.state` (end-effector pose 6 + gripper 2) and 7-d actions, and no simulator state (verified, proxy `PLAN.md` "Step 1 findings"). So this stays a feasibility question (below).

### Which demo moments
All frames of the 100 test demos (12,848 timesteps x 2 cameras = 25,696 frames per condition; verified, tally README, Config; the product is arithmetic). Training and validation stay on the clean stored frames, so nothing is re-rendered for them. Reason: the probe is trained on clean and tested on perturbed, as in the paper. (speculative: re-rendering train/val is not needed.)

### Conditions
Five test sets, same 100 demos in each: **clean re-render** (control), **camera**, **light**, **background**, **noise**. The clean re-render is the reference for "change under perturbation", so a perturbed-vs-clean gap cannot come from a difference between the stored images and our re-render. The stored-clean test set (the tally's) is reported beside it as a pipeline check: if stored-clean and re-rendered-clean disagree, stop and find out why before reading anything else.

### Perturbation settings per type
Use LIBERO-Plus's own sub-types and strengths, so "perturbed" means what the paper means (speculative as a choice). Sub-types as the lead read them through a summarizing fetch of the LIBERO-Plus paper (arXiv 2510.13626v3, Sec. 2.1 / App. A) and the repo README at `4976dc3` (**unverified; not read in primary text this session**):
- Camera: distance, spherical position (azimuth, elevation), orientation (yaw, pitch, roll); field of view per the README. The paper says only the viewpoint changes, "not the scene or robot state".
- Light: diffuse color, direction, specular, shadows on/off, reportedly through scene XML files.
- Background: scene theme textures and working-surface texture, reportedly through scene XML or new scene classes.
- Sensor noise: motion blur, Gaussian blur, zoom blur, fog, glass blur. Whether it is applied after rendering is not stated.

Sampling (proposal): one setting per (demo, type), drawn with a recorded seed and held fixed for every frame of that demo, because camera, light and background are properties of the scene, not of a frame. Sub-types uniformly, strengths from LIBERO-Plus's own difficulty levels where the source exposes them (`task_classification.json` carries a category and difficulty per episode; verified, proxy `PLAN.md` "Step 1 findings"). The exact parameter lists and ranges depend on the source check below; they get written into a config before any render and the config is the source of truth (`_standards/07`).

### Encoders
D plus R, L, S, G, with the same checkpoints and extraction settings as the tally (verified, `../../data/README.md` section 4a): DINOv3 `facebook/dinov3-vitl16-pretrain-lvd1689m` @ `ea8dc28`, 224 px, fp32; C-RADIOv4-SO400M (remote code `c0457f5`), 448 px, fp16 autocast; SigLIP 2 as in the pilot cache; SAM 3 @ `3c879f3`, 1008 px, ViT backbone before the neck; VGGT-Ω @ `1041e80`, 512 px, final aggregator layer. Same rotate-180 handling, both cameras, global features primary and mean secondary (as in the tally). No reason to change anything (a changed setting would break the comparison with the cached clean features). One note: the data README lists SigLIP 2 as `siglip2-large-patch16-384`, while the extractor's default is so400m; use whatever produced the pilot cache and record it (speculative that this is only a documentation mismatch; not checked).

### Main measure: the probe
1. Fit exactly as the tally does (`../../src/diagnostics.py`, verified): clean features, per-task-file demo-level split, ridge with alpha tuned on the clean validation demos, refit on train+val, R² on the test demos averaged over the 7 action dims, probes for D alone and D+B (concatenated, standardized with clean train statistics).
2. Apply the fitted probes, unchanged, to the test features of each condition. Alpha and standardization are never re-tuned on perturbed data (test data stays untouched in development, `_standards/07`).
3. Per condition c and probe X: R²_c(X). Define drop_c(X) = R²_clean-rerender(X) − R²_c(X).
4. Per-pair quantity: **extra drop** e_c(B) = drop_c(D+B) − drop_c(D). Positive means adding B makes the pair lose more under perturbation than DINOv3 alone. Equivalent forms: it is also minus the change in gain(B|D) from clean to perturbed. Report also gain_c(B|D) = R²_c(D+B) − R²_c(D) itself.
5. Pooled over types: e(B) = the mean of e_c(B) over the four perturbation types (each type weighted equally).
6. CIs: demo-level bootstrap over the 100 test demos, at least 1,000 draws, the same draws for every probe and condition so that differences between pairs are paired (as in the tally; `../../STANDARDS.md` requires demo-level CIs). The perturbation draw is fixed per demo, so its variation is part of the demo-level spread.

### Secondary measure: feature drift
Primary definition (proposal): for encoder E, drift_c(E) = mean over test frames of ||f(x') − f(x)||² divided by the mean over test frames of ||f(x) − mean(f)||², using clean-train standardized global features. Reads as "the fraction of how much this encoder's features normally vary between frames that the perturbation moves them". Dividing by the encoder's own spread is what makes it comparable across encoders with different feature sizes and scales. For a pair D+B, the same formula on the concatenated standardized features. Same bootstrap over demos.
Alternatives: (a) 1 − cosine between f(x) and f(x'), divided by the median 1 − cosine between two different random frames of the same encoder; (b) linear CKA between the clean and perturbed feature matrices of the same encoder (scale-free, but a single number per encoder, so no demo-level CI without extra work). Drift is a sanity check: a feature can move a lot without moving the action-relevant part of it (speculative), so drift does not decide anything on its own.

### The "stands out" rule (to be locked by the user before any result is seen)
Primary candidate (**P**): D+R stands out if (i) e(R) is the largest of the four partners' e(B), and (ii) the lower end of the 95% paired-bootstrap CI of e(R) minus the next-largest e(B) is above 0.
Alternatives:
- **A**: R has the largest e_c(B) in at least 3 of the 4 types separately. No CI test. A weaker, looser rule.
- **B**: sign-based: under perturbation, gain_c(R|D) is below 0 (adding R hurts the probe) pooled over types, with the upper end of its 95% CI below 0, while the other three partners' pooled gains are not below 0. Stricter on R's own behaviour, indifferent to the others' ranking.
Also reported either way: the full ordering of the four pairs by e(B) beside the paper's ordering L (+1.83) > S (+0.49) > G (+0.40) > R (−5.41) (verified, tally README), with no significance claim. With 4 partners a hit under any rule has a 1 in 4 chance by chance for "R is largest" (arithmetic), so a hit is a hint (`../../PLAN.md` Next 3).
Note a mismatch to keep in mind: the paper's number per pair is closed-loop success under perturbation; ours is frame-level action regression from a linear probe. They differ in kind (`../../README.md`, Key findings).

### Predictions (speculative; for the user to confirm or change before running)
- All pairs lose some probe accuracy under perturbation (e_c(B) is defined relative to D alone, so this is about drop_c(D) > 0). Confidence: fairly high.
- If the paper's harm is visible in frozen features at all, e(R) is largest and rule P fires. Confidence: low. The clean tests showed R's gain at the top of the four, and the target-side test did not flag R either, so a null result (R does not stand out) is at least as likely.
- A competing guess: because R alone is as robust as D in the paper (background, noise slightly better), R's e(B) could be the smallest, not the largest.
- Drift: no confident prediction on the encoder ordering. Weak guess: G (a geometry encoder) moves most under camera changes.
Any of these coming out the other way is reported, not smoothed over.

### Sample size and CIs
100 test demos (10 tasks x 10 demos; verified, tally README), one perturbation draw per (demo, type). Demo-level bootstrap, at least 1,000 draws, paired across probes and conditions. Optional (needs OK, doubles render and extraction cost): two perturbation draws per (demo, type) to damp the draw-to-draw noise. With 4 pairs, no significance claim for the ordering (`../../STANDARDS.md`).

### Cost (rough; built only from costs recorded in the repo)
- Frames per condition: 25,696 (above). Five conditions. Raw RGB at 128x128x3 is about 1.3 GB per condition (arithmetic), so storage is small.
- Extraction time per condition from recorded rates: SAM 3 at 5.5 frames/s on a 2080 Ti is about 1.3 GPU-h; VGGT-Ω at 12.5 frames/s about 0.6 GPU-h (tally README); SigLIP 2-L at about 142 frames/s about 3 min and C-RADIOv4 at about 62 frames/s about 7 min, both measured on a TITAN RTX (extraction README). The DINOv3 rate is not recorded. So roughly 2 GPU-h per condition, about 10 GPU-h for five (arithmetic from rates measured on different GPUs; speculative as a planning number).
- Probes and bootstrap: CPU. The tally's diagnostics took 30 to 49 CPU-min per pair at 2 cores (tally README); here no CKA and no probe refit per condition, so likely less (speculative).
- **Unknown:** the cost of re-rendering (depends on the feasibility answers below; simulator install per pod is part of it), and whether `diagnostics.py` can apply a fitted probe to new features without code changes (it keeps test predictions, not weights, in what the JSON stores; this is a reading of the code, not tested).

### Compute and gates
`../../STANDARDS.md` rules apply: Jobs that run a command and exit, never `sleep infinity`, start on 1 GPU, GPU utilization above 40%, requests and limits within 20%. Gate (stop and ask): more than 8 GPUs at once or more than 20 Jobs. 5 encoders x 5 conditions is 25 combinations, so extraction must be batched (for example one Job per encoder looping over the conditions, 5 extraction Jobs plus 1 render Job) to stay well under 20. The user's OK is needed before preparing or submitting any Job. Check GPU utilization about 15 minutes into the first extraction Job.

## Feasibility (result of the source check)
**Pending: LIBERO-Plus source not yet available.** The copy at commit `4976dc3` was expected at `/private/tmp/claude-501/-Users-michaelr808-Desktop-research-research26-27/783ff06c-cba7-4999-9fa7-738b18aab793/scratchpad/libero-plus-src` and was not there when this draft was written (2026-11-06). No web pages or summaries are substituted. Questions to answer from the source (cite file:line for each; read only, never run it):
1. **Camera:** where are the camera parameters set, and can an environment be built from an existing demo's scene, its simulator state set, and then rendered with a changed camera pose, distance or field of view?
2. **Light:** how is lighting changed (scene XML, or MuJoCo light fields at runtime), and can it be applied to a restored state?
3. **Background:** how are scene themes and working-surface textures swapped (XML, new scene classes), and can a restored state keep the same objects and robot configuration under a new texture?
4. **Noise:** is each sub-type an image-space function that could be called on any rendered frame, and where in the pipeline is it applied (before or after the policy's image preprocessing)?
5. **Same moments:** are the LIBERO-Plus libero_goal tasks built on the same scenes and object placements as the 10 libero_goal demo files, so a demo's state can be loaded into a perturbed scene? Where do the sub-type ranges and difficulty levels come from?
6. **Render path:** what image size and orientation does LIBERO-Plus render by default, and how do we render at 128x128 in the HDF5's convention (OpenGL, rotated 180 degrees on extraction)?

From the repo's own files (verified, this session): the libero_goal HDF5 contents recorded in `../../data/README.md` section 1 do not list per-step simulator states, so "stored in the HDF5" is **not recorded in the repo**. The inspection behind that list was one file (`turn_on_the_stove_demo.hdf5`). Further questions: (7) does the HDF5 actually contain `states`, `robot_states` and a scene-file attribute (a quick read of one downloaded file would settle it), and (8) if not, can a demo be replayed from its initial state by re-applying its recorded actions, and does the replay match the stored frames? Fast-WAM warns its data is MuJoCo-version sensitive (verified, proxy `PLAN.md` "Step 1 findings"), which makes replay drift a real risk (speculative).

If states or a faithful replay are not available, this design cannot produce "same demo moment, perturbed" frames, and the fallback is a design decision for the user (not made here).

## Open questions for the user
1. Data: libero_goal HDF5 (recommended) or Fast-WAM's dataset?
2. Is the clean re-render control, with the stored-clean test as a pipeline check, acceptable? (It adds one condition, about 20% more cost.)
3. One perturbation draw per (demo, type), or two (double the cost)?
4. Sampling: LIBERO-Plus's own sub-types and difficulty levels (recommended, depends on the source check), or one fixed moderate setting per type?
5. Which "stands out" rule do you lock: P (recommended), A, or B? Or P plus the others as reported-only?
6. Drift definition: the variance-normalized one (proposed), or one of the alternatives?
7. Predictions: confirm or change before any run.
8. If the feasibility check says a stored demo moment cannot be re-rendered under a new camera, light or texture, what then? (Options to discuss: replay from initial states; build perturbed episodes without the demo's actions, which loses the label; drop the type.)
9. Is it acceptable that the LIBERO-Plus sub-type list above stays unverified until the source copy is read?

## Handoff (after sign-off, to the implementation-agent session, high level)
1. Read the LIBERO-Plus source and answer the Feasibility questions (file:line); report back before building anything. If any of 1 to 8 comes back as "cannot", stop and return to the user.
2. Smoke test on one task and a few demos: render clean and one perturbation, look at the frames, confirm they match the stored ones in orientation and in scene content (`_standards/07`, sanity checks).
3. Write the config (perturbation lists, ranges, seeds) and commit it before the render Job.
4. Render Job, then extraction Jobs per encoder (batched, under the gate), then a CPU analysis that applies the clean-fit probes to the five test sets and writes result JSONs under `results/`. New folders for reruns; never overwrite.
5. Log estimated vs actual GPU-hours. Direction `LOG.md` gets a short summary; detail goes in this folder's `LOG.md`; results go in this folder's `README.md`. No result goes into the direction README's key findings without the user's sign-off.

## Amendments
<!-- ### YYYY-MM-DD: what changed, why, and whether any results had been seen yet -->
