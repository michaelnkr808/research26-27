# Research Log

## 2026-11-15 [agent, lead, planning] (direction folder set up and merged: PR #45; Vast.ai access working; RMS-AQA scoped for planning; Interspeech 2027 dates checked)
**Goal:** set up the hear-lab direction from the lead's description of the research, and get ready for the PI's first task (get the RMS-AQA benchmark running).

**Did:**
- Wrote the direction folder from `../_template/`: `README.md`, `STANDARDS.md`, `PLAN.md`, `literature/reading-list.md`, `literature/references.bib`. A fresh read-only reviewer fetched every source and found no wrong facts; it asked for softer wording in three places and for the benchmark reference facts to move out of `PLAN.md` (they are below). Fixed on the branch, then merged on the lead's "merge".
- Wrote the role prompt for the experiment lead session (`hear-lab-experiment`). It lives outside this repo, in the lead's agent tool. It says: Vast.ai first, Nautilus second and only with a reason, and the lead's OK before every rental or cluster Job.
- Vast.ai access: the lead installed the `vastai` tool (version 1.11.0) and saved an API key. `vastai show user` now reports the HEAR Lab team account with 200 dollars of credit (`verified`, read 2026-11-15). `vastai show instances` reports no instances (`verified`, read 2026-11-15). Nothing has been rented.
- Read the RMS-AQA paper (v1 web version), its code repository README, its Hugging Face dataset page, and the challenge site, for planning only.
- Read the official Interspeech 2027 dates page.

**PR:** #45 (direction folder). This entry and the small `PLAN.md` and `README.md` updates with it: direct.

**Found:**

RMS-AQA reference facts (all `verified` by reading the named source on 2026-11-15, unless marked):
- Paper: arXiv 2610.00935 (`chen2026rmsaqa`). Latest version is v2 (2026-10-26); the planning lead read v1, so anything v2 changed has not been read.
- Task: two-stage multiple-choice question answering on 10-second, four-channel, 24 kHz first-order Ambisonics clips. The training set is synthetic, the validation set mixes synthetic and recorded clips, and the hidden evaluation set is all recorded (code README).
- Metric: grounded Stage-2 accuracy (the paper's `A_GS2`).
- Baseline: a frozen audio language model with a trainable spatial branch and LoRA adapters (rank 16). Four backbones: Audio Flamingo 3, Audio Flamingo Next, MiDashengLM, Qwen3-Omni.
- Compute stated in the paper: one baseline training setup on four A100 GPUs, batch size 8 per GPU. Training duration is not stated, so the cost of reproducing it is not known yet.
- Code: https://github.com/rmsaqachallenge/rmsaqa-code. Last push 2026-10-21. No license file. Requirements pin torch 2.6.0 and transformers 4.46.3 for Audio Flamingo 3, with a separate environment (`requirements-transformers5.txt`) for the other three backbones. Scripts: `preprocess_audio_features.py`, `train_spatial_plugin.py`, `eval_spatial_plugin.py`, `score.py`. The code has not been run.
- Dataset: Hugging Face `PeacefulData/RMS-AQA`. The page is public; downloads are gated behind a form (full name, email, affiliation, country, intended use). The page says registration gets automatic approval. The lead said they will make the request.
- Challenge (ICASSP 2027 Grand Challenge) dates, which the challenge site marks as preliminary: recorded evaluation data released 2026-11-24, submission deadline 2026-12-10, results 2026-12-24, two-page papers due 2027-01-07.

Interspeech 2027 (`verified`, https://interspeech2027.org/important-dates/ read 2026-11-15):
- Paper submission 2027-02-09, paper update 2027-02-16, acceptance notification 2027-06-02, camera-ready 2027-06-09. Conference 2027-08-29 to 2027-09-02 in São Paulo, Brazil.
- This is earlier than the "end of February" the lead gave. The page does not state a time zone.
- The page lists the call for papers as "Soon", so the paper format and page limit could not be checked.
- `STANDARDS.md` still carries the unchecked "end of February" wording. It changes by pull request, so it is fixed with the next `STANDARDS.md` pull request (`PLAN.md`, Next).

Vast.ai setup note for later sessions: if `vastai show user` reports a personal account instead of HEAR Lab, a leftover two-factor session file in the tool's config folder is the first thing to check. The tool prefers that session over the saved key. The planning lead moved one such stale file aside on 2026-11-15.

From the lead, 2026-11-15:
- By "the cluster" the PI means Vast.ai.
- The 200 dollars is the whole compute budget for now.
- Experiments are run by the `hear-lab-experiment` session, not by the planning lead.
- The base model is open; Wan 2.2 is a candidate. The target is an Interspeech paper.

**Not verified:** anything in RMS-AQA v2 that differs from v1; whether the baseline code runs as published; how long the paper's training took; the Interspeech 2027 paper format.

**Next:** the lead requests dataset access on Hugging Face and asks the PI what "running" means. Once both are back, the planning lead writes the experiment `PLAN.md` and hands it to `hear-lab-experiment`.

**Open questions:**
- What "get RMS-AQA running" means: score an existing model on the validation set, reproduce the paper's trained baseline for one or more backbones, or enter the challenge.
- Whether the 200 dollar budget is meant to cover a training run.
- How RMS-AQA (question answering) connects to the world action model question, and what the Interspeech paper has to show.
- Whether to start the literature survey (waits for the lead's go).

## 2026-11-15 [agent, lead, planning] (addendum: dataset access requested, waiting for approval)
**Did:** the lead reported that the RMS-AQA dataset access request is made and is waiting for approval. The dataset page says approval is automatic, but the dataset's gate setting is manual, so it may need someone on the dataset side to approve it (`speculative` on how long that takes). `PLAN.md` updated.
**PR:** direct.
**Next:** unchanged. When access is granted and the PI has said what "running" means, the planning lead writes the experiment `PLAN.md` and hands it to `hear-lab-experiment`.
