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

## 2026-11-16 [agent, lead, planning] (dataset access still pending; what "the benchmark on the cluster" takes, read from paper v2 and the code)
**Goal:** answer the lead's question whether the dataset wait is over, and take in the lead's reading of what the PI wants from "get RMS-AQA running".

**Did:**
- The lead showed the Hugging Face dataset page: the access request is "awaiting a review from the repository authors".
- Tested the gate with the Hugging Face login saved on the lead's Mac: a header-only request for `dev_QA/dev_QA.tar` returns 403 with the login and 401 without it (`verified`, 2026-11-16). Downloads are still refused. Which Hugging Face account that saved login belongs to was not checked; it is assumed to be the account that made the request.
- Read, for planning only: the dataset's public metadata from the Hugging Face API (revision `9dd39d8`), the web version of the paper's v2 (training setup and dataset table), and the code repository at commit `387a518` (README, `eval_spatial_plugin.py`, the option lists of `train_spatial_plugin.py` and `preprocess_audio_features.py`, both requirements files). No code was run and nothing was rented.

**PR:** direct.

**Found** (all `verified` by reading the named source on 2026-11-16, unless marked):
- Dataset size: the Hugging Face API reports `usedStorage` of 317,147,090,135 bytes, about 317 GB, for 87 files: 61 training audio shards (`train_audio/train-00000` to `train-00060`, `.tar.zst`), 8 validation audio shards (`dev_audio/dev-00000` to `dev-00007`), one question-answer archive per split, and license files. Per-shard sizes were not read. The gate setting is `manual`.
- Split sizes (paper v2, Table 1): training 170K clips and 170K question-answer pairs from 1,348 rooms; validation 5K clips and 6K pairs from 61 rooms; test 2K clips and 3K pairs from 5 rooms.
- Training recipe (paper v2): three epochs, AdamW, gradient-norm clipping at 5.0, four A100 GPUs with batch size 8 per GPU, LoRA rank 16 with dropout 0.05 on all 28 language-model layers; the checkpoint with the best grounded Stage-2 accuracy on the validation set is kept. The code's defaults match what the paper states: `--epochs 3`, `--batch-size 8`, LoRA rank 16, dropout 0.05, 28 layers. The learning rate and weight decay values did not survive the text extraction of the paper page; the code defaults are `--lr 2e-4` and `--weight-decay 1e-2`, not checked against the paper. Default seed in the code: 20260327.
- Inference (paper v2): Stage 1 is decoded greedily, at most 48 new tokens, and the predicted Stage-1 answer is put into the Stage-2 prompt.
- Training time is still not stated: a search of the v2 text for "hours" finds nothing.
- Scoring needs a trained checkpoint. `eval_spatial_plugin.py` (the Audio Flamingo 3 evaluation script) has `--checkpoint` as a required argument, and the README does not name any released trained checkpoint. So with the released Audio Flamingo 3 scripts, no score can be produced without training first. Not checked: whether the evaluation scripts in the other three backbone folders can score a backbone that has not been adapted, and how the paper's unadapted ("zero-shot") rows were produced.
- Paper numbers the bring-up would be checked against (code README Table 1, validation set, grounded Stage-2 accuracy): Audio Flamingo 3 unadapted 15.78, with the spatial branch 39.10; the best adapted backbone is Qwen3-Omni at 48.93.
- Audio Flamingo 3 weights (`nvidia/audio-flamingo-3` on Hugging Face) are not gated; about 22 GB; the license field is "other" and the terms were not read.

From the lead, 2026-11-16: "i think he wants the benchmark on the cluster so i can run experiments through it". This is the lead's reading of the PI, not yet confirmed by the PI.

Planning lead's read of what that takes (`speculative`, a proposal, nothing decided):
- "Running" would then mean a working test bench on Vast.ai: the data, the feature preprocessing, the training script, and the scoring, shown to work by reproducing one row of the paper's table. Because no trained checkpoint is released, the faithful check is to train one baseline with the paper's recipe and compare its validation score with the paper's. Audio Flamingo 3 with the spatial branch is the natural first one: it is the original baseline and has its own pinned environment.
- That needs the whole dataset (about 317 GB), not only the validation part.
- Vast.ai keeps nothing between rentals unless the disk is paid for, so "on the cluster" has to mean a setup that can be rebuilt on a fresh machine from a script, with the small outputs (trained adapter checkpoint, results, logs) copied off before the machine is destroyed. Where those are kept is open (`STANDARDS.md`, checkpoint storage).
- The cost of the training run is unknown until a real batch is timed.

**Not verified:** the cost and duration of a baseline training run; whether the code runs as published; per-shard file sizes; the Audio Flamingo 3 license terms; anything in the three other backbone folders.

**Next:** the lead confirms with the PI what "running" means and asks whether the PI already has dataset access or can nudge the organizers. Once the lead agrees on the bring-up check, the planning lead writes the experiment `PLAN.md` (by pull request) so that `hear-lab-experiment` can start the day access is granted.

**Open questions:**
- Is the bring-up check "reproduce the Audio Flamingo 3 + spatial branch row" (the planning lead's recommendation), or something else?
- Does the PI also want a challenge entry (submission deadline 2026-12-10, preliminary)?
- Where trained checkpoints and results are kept between Vast.ai rentals.
- Still open from 2026-11-15: whether the 200 dollars is meant to cover a training run; how RMS-AQA connects to the world action model question; whether to start the literature survey.

## 2026-11-16 [agent, lead, planning] (addendum: the lead's approach in the lead's words; literature survey started)
**Did:**
- The lead stated the approach (2026-11-16): the question is about "the intersection of world action models and spatial audio", and "the idea is to add an audio encoder to the world action model so it can also make decisions based on audio input". Recorded here in the lead's words. It does not yet say which world action model, which audio encoder, or where spatial audio paired with robot data comes from; those stay open in `README.md`.
- The lead gave the go for the literature survey ("sure go for the literature survey"). Three workers were dispatched, one per topic, each writing one file under `literature/survey/` on its own branch from `main` at `665839d`: audio as an input to robot policies and vision-language-action models; spatial audio in embodied AI (agents, simulators and datasets, reusable spatial audio encoders); world action models and which could take an audio input (including how the term is defined). Each worker must open every paper's page before listing it and record how much of it was read.
- The planning lead suggested to the lead one more question for the PI: how the PI sees RMS-AQA (question answering) connecting to world action models.
**PR:** direct (this entry and the `PLAN.md` status move). The survey files will go by pull request.
**Not verified:** nothing from the survey yet; the workers had not reported when this was written.
**Next:** check the three survey files, have a fresh reviewer fact-check them against the sources, then write the summary of what exists and what is missing, and the `references.bib` entries.

## 2026-11-16 [agent, lead, planning] (literature survey done and merged: PR #49; the lead prefers Wan 2.2 as the base)
**Goal:** run the literature survey the lead approved, and report what exists and what is missing around "add an audio encoder to a world action model".

**Did:**
- Three workers each wrote one survey file under `literature/survey/`: audio in robot policies (17 papers), spatial audio in embodied AI (30 papers), world action models (19 papers). Each paper was opened by the worker, with read depth recorded per paper.
- Review, all posted on the pull request: three fact-check reviewers checked every entry's metadata against the arXiv API and the content against raw page text; a fourth reviewer tried to disprove the "I did not find" claims with its own searches; a second-round reviewer checked the fixes and the new entries. The workers fixed their own files. The lead read the Cosmos 3 model cards and the S2A2 seed sentence from raw text itself.
- Merged on the lead's "yeah merge the pr" (2026-11-16).

**PR:** #49 (the three survey files). This entry and the `PLAN.md` and `README.md` status moves: direct.

**Found** (what the reviewed survey files say; `verified` means read from the paper or page by a worker or reviewer on 2026-11-16, not reproduced; detail and quotes are in the files and the pull request comments):
- No world action model was found that was trained or evaluated with audio as an input to robot control (`verified` only as "not found in these searches").
- Cosmos 3 (NVIDIA, open weights) is a world action model family whose base model accepts and generates two-channel 48 kHz audio; its released robot policy takes cameras and proprioception only, and its audio input is capped at 0.5 s on the model card (`verified`, model cards read by the lead). Whether its two channels carry direction was not checked. The first survey pass missed it; the gap-claim reviewer found it.
- FLUX 3 Action (Black Forest Labs) saw audio in pretraining only (`verified` by a worker and a reviewer).
- Most open world action models read are built on the Wan 2.2 5B family (Fast-WAM, Motus, LingBot-VA, tau0-WM); Ovi and MOVA are joint audio-video generators built on Wan 2.2 (`verified` by a worker and a reviewer; the exact checkpoint is stated only for tau0-WM).
- Closest to the direction: "Audio-Visual World Models" (arXiv 2512.00883) predicts frames and binaural audio under actions, in simulated navigation only, with a small model, not a pretrained video model.
- Spatial audio in acting models is common in simulated navigation, including NaVLA2 (one of the papers the lead has read), which trains on its own 43.9K-episode set rendered with SoundSpaces 2.0 on Habitat and reports an ablation of a mono-downmix branch against a binaural branch. For real-robot manipulation only two were found: S2A2 (microphone arrays, preprint) and SoundAct (stereo, workshop project page).
- No publicly released real-robot dataset with calibrated spatial audio, video and robot actions was found. REASSEMBLE is the closest public one (three separately placed microphones, not an array). FEEL has a 7-microphone array with human, not robot, actions.
- Several spatial audio encoders and spatial audio language models exist that could be reused (BAT's SpatialAST, SPUR, JAEGER, OmniEcho, and the RMS-AQA baseline's spatial branch); release status differs and is recorded per entry.
- One reviewer finding was wrong and not applied (S2A2's real-robot seed count is stated in the paper: a single seed); one first-round finding was reversed (JAEGER has a public checkpoint repository). Both are recorded on the pull request.

From the lead, 2026-11-16:
- "id prefer to work with wan 2.2" as the base. This is the lead's preference, not yet a decision record; which Wan 2.2 world action model is not chosen.
- The lead expects datasets with spatial audio alongside video to exist, and asked what NaVLA2 trained on. Answer from the survey entry: simulator-rendered binaural audio (SoundSpaces 2.0 on Habitat) with navigation actions. So such data exists for simulated navigation; what was not found is the real-robot manipulation case.
- The lead asked what the Audio Flamingo 3 baseline is; the planning lead explained it in chat. Whether it is the first benchmark experiment is still not confirmed.

**Not verified:** nothing in the survey was reproduced. Items each file lists under "Could not verify" (among them the SoundAct OpenReview paper and the Preprints.org embodied-audio survey). Whether the MINav dataset and NaVLA2 weights are released. Section and table labels in entries read through the summarizing page reader.

**Next:** second pull request with the `references.bib` entries for the surveyed papers, a short summary of what exists and what is missing, and a read-first list in `reading-list.md`. Then the open choices for the lead and the PI: navigation in a simulator (data exists) or robot manipulation (no public spatial audio data found); which Wan 2.2 world action model; which spatial audio encoder.

**Open questions:**
- Navigation or manipulation as the setting, given where spatial audio data exists.
- Which Wan 2.2 based world action model, and what it costs to fine-tune on the 200 dollar budget.
- Still open: what the PI means by "running" for RMS-AQA and how it connects to world action models; whether the 200 dollars covers a training run; dataset access (pending).

## 2026-11-16 [agent, lead, planning] (addendum: the lead leans to navigation as the setting; follow-up surveys started; Audio Flamingo 3 size)
**Did:**
- From the lead, 2026-11-16: "why do we need a robot arm doing manipulation with spatial audio? i feel like that doesn't make sense for a task where audio would be useful ... i feel that adding audio to a WAM would help moreso with navigation over like the tasks that a robot arm would do". The lead asked for more datasets like the one NaVLA2 trained on. Recorded as the lead's lean toward navigation as the setting; not yet a decision record, and not yet discussed with the PI.
- From the lead, 2026-11-16, on reproducing the Audio Flamingo 3 baseline as the first benchmark experiment: "the audio flamingo 3 idea doesn't sound too bad", and asked how big it is. Not a go-ahead for an experiment plan yet.
- Dispatched two more survey workers, each writing one new file under `literature/survey/` from `main` at `10f5288`: datasets, benchmarks and simulators with spatial audio, egocentric video and navigation actions (how to get each, release status); and world models and world action models for navigation (what they are built on, whether any use Wan 2.2, whether any take audio). A third worker is writing the `references.bib` entries for the papers in PR #49.
- Read the Audio Flamingo 3 model page (`nvidia/audio-flamingo-3`, Hugging Face, raw model card and file listing, 2026-11-16).
**PR:** direct.
**Found** (`verified`, model card and file listing read 2026-11-16): Audio Flamingo 3 is built from the AF-Whisper audio encoder, an MLP audio adaptor, and a Qwen2.5-7B language model. The weight files total about 18 GB (language model about 15.2 GB in four files, audio encoder about 1.3 GB, plus adapter files). The card lists test hardware as an A100 80 GB. License: NVIDIA OneWay Noncommercial License, with parts under the Qwen Research License; the full terms were not read.
**Not verified:** how much GPU memory the RMS-AQA baseline training needs with this model, and how long it takes; that needs a timed batch on a rented machine.
**Next:** check and review the two new survey files and the bibliography, then bring the lead the navigation data options and the navigation base-model options.

## 2026-11-16 [agent, lead, planning] (survey follow-up merged: PR #50; navigation data and navigation world models; prior-work check started)
**Goal:** follow the lead's lean toward navigation: find data like NaVLA2's, find world models for navigation, and finish the survey outputs (bibliography, summary, read-first list).

**Did:**
- Two survey workers wrote `literature/survey/2026-11-16-spatial-audio-navigation-data.md` (20 datasets and simulators) and `literature/survey/2026-11-16-navigation-world-models.md` (23 papers). A third wrote `literature/references.bib` entries (95 entries, covering all 92 survey keys), with metadata from the arXiv API or each paper's own page. The planning lead wrote `literature/survey/2026-11-16-summary.md` and the read-first list in `literature/reading-list.md`.
- Review, posted on the pull request: one fact-check reviewer per new survey file, then a second-round reviewer for the fixes, the new entries, the summary, the read-first list and a 20-entry bibliography sample. The first round found two Wan 2.2 navigation models the worker had missed (WNM-3D, UniWAM); the second round found one wrong release field (WorldVLN) and nine statements in the summary that said more than the survey files support. All fixed; the resolution is recorded in a comment on the pull request.
- Merged on the lead's "merge pr 50" (2026-11-16).
- The planning lead ran a keyword search of the Hugging Face dataset index (soundspaces, binaural, ambisonic, "audio-visual navigation", audiogoal, "spatial audio", "audio visual world model") on 2026-11-16. It returned audio-only sets and the FAR corpora already in the survey, and no set with video and navigation actions (`verified` as a listing read; the datasets were not opened).

**PR:** #50. This entry and the `PLAN.md` and `README.md` status moves: direct.

**Found** (what the reviewed files say; `verified` means read from the paper or page by a worker or reviewer on 2026-11-16, nothing reproduced; detail in the files):
- Navigation world action models built on Wan2.2-TI2V-5B exist: WNM-3D (training scripts for two of three stages and checkpoints released; weights non-commercial; 16 H100), AstraNav-World (inference code and part of the weights; evaluated in Habitat; 96 H20), and for mobile manipulation UniWAM and ABot-M0.5. None mentions audio (keyword search of each paper's full text).
- WNM-3D's benchmarks are GN-Bench and VLN-CE R2R; the survey entry does not say which simulator runs them. The planning lead had told the lead in chat that WNM-3D is tested in the simulator family that renders spatial audio; that was not supported and was corrected in chat and in the summary.
- NaVLA2's MINav data and checkpoints are listed as unreleased in its repository. No release was found for the Audio-Visual World Models data (AVW-4k) or LH-AVLN. What is public is the means to render such data (SoundSpaces 2.0 on Habitat, with released sound libraries and episode definitions); the navigation data file's closing section lists what has to be obtained and run.
- Public real recordings nearest to the need: OmniEchoBench (first-order Ambisonics at grid points in 30 scanned rooms, frames rendered from meshes), and smart-glasses sets with a 7-microphone array and head trajectories but no navigation actions.
- Audio-Visual World Models (arXiv 2512.00883) uses its audio-visual world model in a planner to improve an audio-visual navigation agent in SoundSpaces 2.0. What exactly it compares (for example a vision-only world model) is not in the survey entry.

From the lead, 2026-11-16:
- Doubts that nobody has measured whether audio improves a world action model: "that seems like it may have already been done before".
- Base model: "ill decide on the base model in a bit".
- Data: "im sure we can find some dataset".
- Budget and base: "thats fine, we wanted to build of wan 2.2 anyway right", in reply to the point that the realistic form starts from released Wan 2.2 based weights and trains only an added part.

**Not verified:** whether a ready dataset exists that the searches missed; whether WNM-3D can be evaluated where SoundSpaces 2.0 renders audio; what fine-tuning from a released checkpoint costs; the answer to the lead's prior-work doubt.

**Next:** a worker is running a focused prior-work check on one question (has anyone measured whether audio improves a world action model or world model for acting), reading the closest papers in full and chasing their citations. Its file goes by pull request with a fact-check. Then the lead picks the base model; the planning lead writes the decision records.

**Open questions:**
- The prior-work check's answer.
- Base model (the lead will decide).
- Where the data comes from: a ready set if one is found, otherwise rendering with SoundSpaces 2.0.
- Still open with the PI: what "running" means for RMS-AQA and how it connects; whether the 200 dollars covers a training run; the navigation setting; what the Interspeech paper has to show. Dataset access for RMS-AQA is still pending.

## 2026-11-16 [agent, lead, planning] (prior-work check merged: PR #51; session close)
**Goal:** answer the lead's doubt that nobody has measured whether audio improves a world action model.

**Did:**
- One worker wrote `literature/survey/2026-11-16-prior-work-check-audio-in-world-action-models.md` (a verdict and nine ranked entries), reading the closest papers in full from raw text and chasing their citations.
- A fact-check reviewer checked the file against the sources, ran its own searches (14 further queries), and posted its findings on the pull request. Three required changes went back to the same worker and landed as a new commit. The planning lead added two bibliography entries (`ji2026dreaming`, `akin2025pixels`) and a paragraph plus pointer in `literature/survey/2026-11-16-summary.md`. Key check after the merge: 94 survey keys, 97 bibliography keys, none missing.
- Merged on the lead's "yeah merge 51" (2026-11-16). Worker worktree and branches removed.
- Filled the project brief the other hear-lab sessions read at start (outside the repo).

**PR:** #51. This entry and the `PLAN.md` move: direct.

**Found** (`verified` means read from the paper's raw text by the worker and the reviewer on 2026-11-16; nothing reproduced):
- Audio-Visual World Models (`wang2025audio`, arXiv 2512.00883), version 3 only, Table 4: navigation success over 500 episodes in SoundSpaces 2.0 with a fixed planner. Baseline without a world model 0.269; world model with its audio forecast removed 0.266; with its vision forecast removed 0.280; full audio-visual world model 0.381; oracle world model 0.423 (`verified`). So only the joint audio and video forecast beat having no world model. The table is absent from version 4 (`verified`). The paper does not state whether the agent's own audio input changed between rows; the file labels that reading as inferred. No mono variant, no pretrained video model, no code or data release found.
- `akin2025pixels` (arXiv 2511.01310), a multi-agent maze task built to need audio, reported a gain from audio but was withdrawn by its authors, who state the results are invalid (`verified` from the arXiv record). It is in the bibliography marked withdrawn and is not to be cited as evidence.
- Not found in the worker's search or the reviewer's own: (a) a comparison of mono against spatial audio inside any world model or world action model; (b) a large pretrained video world action model (Wan, Cosmos, LTX) given audio input and scored on task success. "Not found" is a search result, not proof of absence.
- The lead's doubt was partly right: "audio helps a world model act" has a small-scale precedent. The summary's "not found" item now carries this result and its limits.

Corrections the planning lead made in chat this session (also fixed in the files): WNM-3D is not confirmed to run in the simulator that renders spatial audio (the survey names its benchmarks, not the simulator); "the agent heard audio in every setup" in Audio-Visual World Models is an inference, not a statement in the paper, and removing the vision forecast hurt as much as removing the audio forecast.

Planning lead's suggestion, not a decision (`speculative`): frame the experiment as three versions of one Wan 2.2 navigation model (no audio, one-channel audio, spatial audio), since the spatial comparison is the part not found in prior work. The framing is the lead's and the PI's to decide.

**Not verified:** whether a paper the searches missed does (a) or (b); what Audio-Visual World Models' rows changed about the agent's own inputs.

**Next:** the lead picks the base model and talks to the PI. Then the planning lead writes the decision records (base model, setting, audio encoder, data source) and fills the open `STANDARDS.md` entries by pull request. No experiment is planned or approved. Good point to start a fresh session.

**Open questions:**
- Base model (the lead decides). Candidates in `literature/survey/2026-11-16-summary.md`: WNM-3D, AstraNav-World, UniWAM, or plain Wan 2.2 with an own action head. Open check: whether WNM-3D can be evaluated where SoundSpaces 2.0 renders audio.
- For the PI: what "running" RMS-AQA means (reproduce the Audio Flamingo 3 baseline by training it, or something lighter); how RMS-AQA connects to the world action model work; whether the 200 dollars covers a training run; challenge entry by 2026-12-10; the navigation setting; whether the PI has dataset access or knows a ready dataset.
- RMS-AQA dataset access: still pending on Hugging Face.

## 2026-11-17 [agent, lead, planning] (RMS-AQA access: the challenge has a separate team registration form)
**Goal:** find out why the dataset request is still pending and who to contact.

**Did:**
- Retested the download with the lead's saved Hugging Face login: a gated file returns 403 with the login and 401 without (`verified`, 2026-11-17). The lead's settings page shows the request as PENDING since Nov 15 (the lead's screenshot).
- Read the challenge website's page script (`rmsaqachallenge.github.io`, raw JavaScript bundle; the page itself is rendered by script, which is why the 2026-11-15 read missed this) and the public registration form.

**PR:** direct.

**Found** (`verified`, read 2026-11-17):
- The site's Resources section says: "Teams will register for the challenge by submitting the official registration form." and "The email address on your Hugging Face dataset access request must exactly match the Primary contact email used during challenge registration." The form is a Google Form titled "RMS-AQA Challenge at ICASSP 2027: Team Registration" (`https://forms.gle/vpWYdEp3i6kHX7hV8`). Its description says the primary contact "will receive challenge announcements, data-access instructions, and submission updates". Required fields: primary contact email and name, affiliation, team name, country or region, team members (name, email, affiliation per line), and an eligibility confirmation.
- Organizers' contact address on the site: `rmsaqa.2027@gmail.com`.
- The dataset's Community tab on Hugging Face had no discussions.
- Correction to the 2026-11-15 entry and to `PLAN.md`: access is not only the Hugging Face form. A team registration on the separate Google Form appears to be part of it.

**Not verified:** that the missing team registration is the reason the request is pending (`speculative`; it fits what the site says). Whether the organizers grant access for research use without a challenge entry.

From the lead, 2026-11-17: the direction as stated is too vague and needs to become a concrete question; does not see how Audio Flamingo 3 relates to the research. The planning lead proposed a candidate question in chat (does spatial audio help a Wan 2.2 navigation world action model more than one-channel audio when the target is out of sight); the lead has not chosen yet.

**Next:** the lead decides, with the PI, whether to register a team (which signals a challenge entry) or to ask the organizers for research-only access by email. No form was submitted and no email was sent from this session.

## 2026-11-18 [agent, lead, planning] (options for the research question, for the PI meeting on 2026-11-19)
**Goal:** turn the direction into a concrete question before the lead's weekly PI meeting (Thursdays, per the lead; the next one is 2026-11-19).

**Did:**
- The lead emailed the challenge organizers (`rmsaqa.2027@gmail.com`) from the lead's own account, asking whether dataset access is possible for research use without a team registration. The planning lead drafted the text; nothing was sent from this session, and the team registration form was not submitted.
- Put three candidate research questions to the lead in chat. The lead has not chosen; they go to the PI.

**PR:** direct.

**Found:** nothing new measured or read. The three options (`speculative`, the planning lead's proposals built on `literature/survey/2026-11-16-summary.md` and the prior-work check):
1. Does direction matter: when the target is out of sight, does a Wan 2.2 navigation world action model do better with spatial audio than with one-channel audio or no audio? Three versions of one model, success at reaching a sounding object, split by whether the target was visible at the start; second measure, whether audio improves the predicted frames for regions out of view. The planning lead's recommendation. Known neighbor: NaVLA2 compares one-channel and binaural audio in a navigation model that is not a world model.
2. Option 1 plus an encoder comparison: does a spatial audio encoder that scores well on RMS-AQA also help the world action model navigate? Ties the PI's first task to the paper; each extra encoder is another training run.
3. Does the model need to predict future sound, or is hearing it enough? Wan 2.2 does not generate audio, so this needs a new audio output path; proposed as future work only.

From the lead, 2026-11-18: "i just want the dataset" (not a decision to enter the challenge); wants a direction, or a few options, ready before the PI meeting.

**Not verified:** cost of any option (data must be rendered with SoundSpaces 2.0, and fine-tuning cost for a Wan 2.2 navigation model has not been timed); whether option 1's comparison exists in a paper the searches missed.

**Next:** the lead brings the options to the PI on 2026-11-19, with these questions: which option; what the PI wants taken from RMS-AQA into the world action model work (and whether reproducing the Audio Flamingo 3 baseline is needed at all); whether more compute is available if a timing run shows 200 dollars is not enough; whether to register a challenge team. After the meeting the planning lead writes the research question into `README.md` and a decision record (by pull request).

**Open questions:** the organizers' reply on dataset access; the base model (candidates in the summary; open check whether WNM-3D can run where SoundSpaces 2.0 renders audio).

## 2026-11-18 [agent, lead, planning] (addendum: the lead leans to WNM-3D as the base; simulator check)
**Goal:** record the lead's base model choice and run the open check on it (can WNM-3D run where SoundSpaces 2.0 renders audio).

**Did:**
- The lead asked whether to build on FastWAM (`yuan2026fast`) and then chose WNM-3D: "it makes more sense to do WNM-3D since its a navigation model" (2026-11-18). No decision record yet; the PI meeting is 2026-11-19.
- Read, as raw text on 2026-11-18: the WNM-3D repository's `README.md`, `docs/TRAINING.md`, `docs/DOWNLOADS.md` and its file tree (146 entries, GitHub trees API, branch `master`), and SoundSpaces' `INSTALLATION.md` (branch `main`). The code itself was not read.

**PR:** direct.

**Found** (`verified` as read from those files):
- WNM-3D's released training and evaluation run in its own simulator stack: GN-Bench from the companion GN0 repository, on InteriorGS Gaussian-splat scenes with GN-Matrix annotations. Training data is rendered there (`scripts/evaluation/render_dataset.sh` in GN0) and converted to LeRobot format (`scripts/data/convert_interiorgs_to_lerobot.py`); the trainer reads LeRobot-format data (`gammanav/vln/data/dataset/lerobot.py`). Evaluation talks to a GN-Bench WebSocket client. The words Habitat, R2R, RxR, Matterport and VLN-CE do not appear in the README, the two docs or any file path.
- SoundSpaces 2.0 installs on Habitat only: Python 3.9, habitat-sim on the `RLRAudioPropagationUpdate` branch built with `--audio`, habitat-lab `v0.2.2`.
- So the answer to the open check is no, not as released: WNM-3D's simulator and the audio simulator are different programs. Two things would have to be built to use WNM-3D with spatial audio: (1) audio-visual navigation data rendered in Habitat with SoundSpaces 2.0 and converted to the LeRobot format the trainer reads; (2) a closed-loop evaluation link between WNM-3D's inference server and Habitat.
- FastWAM for comparison (survey entry `yuan2026fast`): Wan2.2-5B plus an action expert, code and checkpoints released, robot-arm benchmarks (LIBERO, RoboTwin 2.0), no future video at test time.

**Not verified:** whether the released WNM-3D checkpoints were trained on any Habitat or Matterport3D data (the paper reports VLN-CE R2R; the repository does not mention it), so how well they transfer to Habitat scenes is unknown; how much work (1) and (2) are; whether AstraNav-World, which is evaluated in Habitat but has no released training script for its video part, would be less work overall (`speculative`: the planning lead thinks WNM-3D is still the better start because its training code exists and its data format is not tied to a simulator).

**Next:** the lead raises the base model and this finding with the PI on 2026-11-19. Then a decision record for the base model (by pull request).

## 2026-11-18 [agent, lead, planning] (addendum 2: the lead's positions before the PI meeting; RMS-AQA parked)
**Goal:** record what the lead decided to bring to the PI meeting on 2026-11-19.

**Did:**
- Gave the lead a status summary and the five questions planned for the PI. The lead answered them directly.
- Moved the RMS-AQA item in `PLAN.md` from "Now" to "Parked", and updated the `README.md` status and the project brief.

**PR:** direct.

**Found:** nothing new measured or read.

From the lead, 2026-11-18 (answers to the planning lead's numbered list):
- On what the PI wants from RMS-AQA and whether the Audio Flamingo 3 baseline is needed: "maybe just drop this for now". The planning lead reads this as parking the RMS-AQA task, not only the question (the lead's answer on the team registration supports that reading). The PI assigned the task and has not been told.
- On the base model: "WNM-3D seems fine but can u reexplain the simulator gap". The gap was explained again in chat (same content as the 2026-11-18 addendum above).
- On the budget: "training run and we can get more later".
- On registering a challenge team: "we can figure this out later and doesn't matter if we drop RMS-aqa".
- Asked for the research question options again; no option chosen.

Consequences the planning lead stated to the lead: with RMS-AQA parked, option 2 (the encoder comparison) loses its tie to the PI's first task; the challenge submission deadline (2026-12-10, preliminary) passes without an entry unless the task is picked up again; the pending access request and the email cost nothing to leave open.

**Not verified:** the PI's view on any of this; what a training run costs (no batch has been timed, and the data has to be rendered first), so whether 200 dollars covers one is unknown.

**Next:** the lead tells the PI on 2026-11-19 that RMS-AQA is parked, and brings the three options, WNM-3D with the simulator gap, and the budget position. After the meeting the planning lead writes the research question into `README.md` and the decision records (base model, setting, audio encoder, data source) by pull request.

**Open questions:** which research question option; the PI's answer on parking RMS-AQA; how much work the two WNM-3D pieces are (Habitat audio-visual data in LeRobot format, and a Habitat evaluation link); whether the released WNM-3D weights transfer to Habitat scenes.

## 2026-11-18 [agent, lead, planning] (addendum 3: novelty check and simulator gap, from the literature session, PR #68 open)
**Goal:** at the lead's ask, have the literature session check the novelty of all three candidate questions and look into the WNM-3D simulator gap, including the lead's question "is the reason we can't reuse the NaVLA paper's benchmark because its a vla?".

**Did:**
- Sent the request to the hear-lab literature session (the lead's words relayed verbatim). It ran two research workers and a fact-check reviewer with its own searches, and opened PR #68 (`literature/survey/2026-11-18-novelty-and-simulator-gap.md`). The reviewer reported no errors; two wording fixes were applied.
- Relayed the provisional verdicts, then the reviewed result, to the lead in chat. The planning lead read the file's verdict section on the PR branch, not the sources.

**PR:** #68 (the literature session's; open, merge waits for the lead's OK in that window). This entry and the `PLAN.md` notes: direct.

**Found** (as reported in the PR #68 file, `verified` there by its workers and reviewer on 2026-11-18; nothing reproduced; detail and sources in the file):
- Question 1 (spatial against one-channel against no audio in a Wan 2.2 navigation world action model, split by target visibility): not found. NaVLA2 remains the only spatial against one-channel comparison in an acting agent (a policy, two different encoders). No audio-visual navigation paper opened splits success by target visibility; in SoundSpaces AudioGoal the target has no visual presence by construction, so the split has to be built. The out-of-view frame measure was also not found.
- Question 2 (does an encoder that ranks better on a spatial audio benchmark navigate better): the claim not found. Nearest: ManiWAV compares three audio encoders by manipulation success, without spatial audio or an upstream benchmark.
- Question 3 (predict future sound, or only hear it): partly done in two other settings. HEAR (manipulation, one microphone, a vision-language-action policy) removes its audio prediction and keeps audio as input, and success drops (0.81 to 0.73 simulation average, as copied in the file). Audio-Visual World Models version 3 removes the audio part of the forecast in a small navigation world model. Nothing for a pretrained video world action model or with spatial audio.
- The lead's question: model type is not the obstacle. WNM-3D's server takes RGB frames, an optional instruction and pose and returns (dx, dy, dyaw) blocks, so it can drive any simulator given a client. MINav's obstacle is its release state (dataset, checkpoints and training scripts unreleased; two data-building scripts are stubs; the evaluation config points at episodes and a checkpoint that are not there), and MINav is a different task (instruction chains, discrete actions).
- No existing benchmark is a drop-in. To build: a Habitat client for WNM-3D's server, an action converter (described in the paper for VLN-CE, not released), a fixed instruction for audio tasks, and a target-visible-at-start flag (SAVi episodes store viewpoints from which the object is visible, so it can be computed). Closest fit by task: SoundSpaces SAVi. Closest by rendering stack: MINav's code (SoundSpaces 2.0 on HM3D), without episodes.
- WNM-3D's R2R result was run in Habitat (appendix B.2) after training on R2R and RxR, but no Habitat or R2R code is in the WNM-3D, GN0 or WNM-3D-RL repositories, and the released checkpoints were trained only on InteriorGS data.
- AstraNav-World is not closer (no training script for its video part, no video weights found). UniWAM (Wan2.2-5B, trained and evaluated in Habitat on HM3D, WebSocket policy server, code and gated checkpoints released) is the closest complete stack found; it has no audio, a different habitat-sim version from SoundSpaces 2.0's audio branch, and a different task.

Planning lead's suggestion to the lead, not a decision (`speculative`): keep WNM-3D as the position for the PI meeting and name UniWAM as the alternative; compare the two properly before any decision record.

**Not verified:** the literature session notes that the arXiv listing API was rate limited, so its sweep for papers newer than 2026-11-16 leaned on web search; "not found" is a search result. How much work the four pieces are. Whether UniWAM's habitat-sim version can be reconciled with SoundSpaces 2.0's.

**Next:** the lead's OK on PR #68 in the literature window; the PI meeting on 2026-11-19; then the research question and the decision records by pull request.

**Open questions:** WNM-3D or UniWAM as the base; which question; the PI's answer on parking RMS-AQA.

## 2026-11-18 [agent, lead, planning] (addendum 4: the lead narrows to two candidate questions; presentation prep handed to the presentations session)
**Goal:** record the lead's choice of candidate questions and hand the PI meeting presentation to the presentations session.

**Did:**
- The lead, 2026-11-18: "okay we'll mark down the two potential research directions as those top two novel ones, pass over to the presentations agent to help me prepare a presentation". The two are questions 1 and 2 of the 2026-11-18 options entry, the ones the novelty check (PR #68, open when this was written) reported as not found. Question 3 is set aside (partly done in other settings; needs a new audio output path).
- Sent the presentations session a hand-off: purpose, the two questions, where the checked facts are, the lead's positions, and the open points.
- Updated `PLAN.md` and the project brief.

**PR:** direct.

**Found:** nothing new measured or read.

Point the planning lead raised with the lead: question 2 needs an upstream spatial audio benchmark to rank the encoders by. RMS-AQA was that benchmark and is parked, so question 2 either brings RMS-AQA back or needs another benchmark (`speculative`: published encoder scores might serve; not checked).

**Not verified:** the PI's view; cost of either question (question 2 adds one training run per extra encoder).

**Next:** the PI meeting on 2026-11-19. Then the chosen question goes into `README.md` and a decision record by pull request, with the base model, setting, audio encoder and data source records.

**Open questions:** which of the two questions; WNM-3D or UniWAM as the base; what ranks the encoders in question 2 if RMS-AQA stays parked; the lead's OK on PR #68 (in the literature window).

## 2026-11-19 [agent, lead, planning] (second, adversarial novelty check on questions 1 and 2; PR #68 parts 3 and 4)
**Goal:** answer the lead's doubt, 2026-11-19: "are you sure nobody has done research on this? Like i feel like the obivious answer and we read the NaVLA paper too" and "can u double check nobody has done this also".

**Did:**
- Told the lead the planning lead is not sure: "not found" is a search result, and the expected answer to question 1 is already suggested by NaVLA2.
- Asked the literature session for a second check aimed at breaking the two verdicts (a worker and a reviewer who had not worked on PR #68; a direct arXiv sweep, forward citations of NaVLA2, Audio-Visual World Models, SoundSpaces 2.0, SAVi, BAT, ManiWAV and HEAR, OpenReview including ICLR 2027 submissions, GitHub), and for a statement of what is already known. It is now parts 3 and 4 of PR #68; a separate fact-check reviewer confirmed the verdicts and two corrections to minor entries were applied.
- Sent the presentations session the changed framing for question 1.

**PR:** #68 (the literature session's; open when this was written, merge waits for the lead's OK in that window). This entry and the `PLAN.md` note: direct.

**Found** (as reported by the literature session from PR #68 parts 3 and 4, `verified` there by its worker and reviewer; the planning lead did not read the sources; nothing reproduced):
- Question 1 still "not found" as stated: nothing gives a navigation world model spatial, one-channel and no audio as matched conditions, and nothing splits results by whether the target is visible at the start. The reviewer's own searches found no controlled mono against binaural ablation with the same encoder in any audio-visual navigation agent.
- Question 2 still "not found" as stated. Near misses: a public GitHub repository created about 2026-11-16 (adityakaduskar/acoustic-jepa; not a paper, echolocation only, no vision), where a ranking reversal between static tests and planning is shown only for a random encoder against the learned one; BCNav (arXiv 2609.37084; navigation success falls as sound-direction error grows, one estimator); ADAPT (arXiv 2610.15956; a spatial audio adapter raises success, no encoder ranking); HearVGo (ICLR 2027 submission; 14 omni models on a sound-guided walking benchmark, no world model, no mono condition).
- Already known: audio beating no audio in audio-goal navigation (SoundSpaces, Look Listen and Act, SAVi, Younes et al. 2022, NaVLA2). The only spatial against mono head-to-head is NaVLA2 Table 2, success rate: no audio 0.206, mono-downmix CLAP branch only 0.219, spatial SpatialAST branch only 0.258, both 0.272. It is not controlled: the branches use different encoders, no seeds or run count are stated.
- Still new, the literature session's reading (`speculative`): a controlled mono against spatial against none comparison with the encoder held fixed, inside a pretrained video world action model; the split by target visibility (SoundSpaces targets are invisible by design); predicted-frame quality out of view. For question 2: the encoder-rank against acting test with real spatial audio plus vision in a navigation world model.

Correction to what the planning lead told the lead in chat on 2026-11-19: the acoustic-jepa repository was described as hinting that a better-scoring encoder can be worse for planning; its reversal is only between a random encoder and the learned one, so it is a weak hint. Corrected in chat.

Planning lead's suggestion to the lead, not a decision (`speculative`): pitch question 1 as the first controlled test of how much direction adds, and when, inside a world action model; the headline direction (spatial beats mono) would not be surprising by itself.

**Not verified:** OpenReview blocked the PDFs, so some ICLR 2027 submissions (Hear-to-Act, SoundAct, HearVGo) were judged from abstracts only; a mono ablation could be in their full texts.

**Next:** the PI meeting today (2026-11-19); the lead's OK on PR #68 in the literature window; then the research question and decision records by pull request.

**Open questions:** whether a controlled comparison plus the visibility split is enough for the paper (for the PI); which of the two questions; WNM-3D or UniWAM; what ranks the encoders in question 2 if RMS-AQA stays parked.

## 2026-11-19 [agent, lead, planning] (PI feedback after the 2026-11-19 meeting; three checks sent to the literature session)
**Goal:** record the PI's feedback and start the follow-ups it names.

**Did:**
- The lead held the PI meeting and gave the planning lead a screenshot of the PI's written follow-up (chat messages, 2026-11-19, 8:15 PM to 8:33 PM). The screenshot starts partway through the thread; anything said earlier or in the meeting itself is not known to the planning lead.
- Identified the first linked paper from its arXiv abstract page (one web fetch). The second link (CVPR open access) refused the fetch (HTTP 403), so that paper is not identified beyond its file name.
- Asked the literature session for three checks (below).

**PR:** direct.

**Found:**
From the PI (quoted from the screenshot):
- "And if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0"
- A link: arxiv.org/pdf/2610.21588.
- A link: openaccess.thecvf.com/content/CVPR2026/papers/Ryu_Hear_you_are_Teaching_LLMs_Spatial_Reasoning_with_Vision_and_CVPR_2026_paper.pdf.
- "Also, it would be a good idea to pursue research in a direction that addresses aspects not covered in existing audio-VLA studies (e.g., long-form processing)" and "Is it good?"
From the lead, in reply to the PI: "Okay yeah I can definitely look into that too".

arXiv 2610.21588 (`verified` from the abstract page only, 2026-11-19; the PDF was too large for the fetch tool): "Hear, Localize, and Reason: Spatially Aware Scene Understanding for Audio-Visual LLMs", Kim Sung-Bin, Lee Jung-Mok, Jinwoo Jung, Oh Hyun-Bin, Hyeonggon Ryu, David Harwath, Tae-Hyun Oh; submitted 2026-10-15; comments field "NeurIPS 2026". Abstract: a benchmark (HLR-AVSceneQA) with a hear, localize, reason hierarchy, and a model (HLR-LLM) that adds a binaural spatial audio branch next to the monaural semantic audio pathway of an audio-visual foundation model. It is a question-answering model, not a navigation or acting model (from the abstract). How its data was made, and what is released, were not read.

The planning lead's reading (`speculative`): both links are about teaching an audio-visual language model to reason about where sounds are, so they look like method and data references for the spatial audio branch, not competitors on the navigation question. "Long-form processing" probably means handling long stretches of audio over time; the PI's meaning was not confirmed.

**Not verified:** whether GN-Bench can render any audio (only its documentation and file list were read on 2026-11-18); the content of the CVPR 2026 paper; what existing audio-VLA studies do about audio length; which of the two candidate questions the PI prefers, and the PI's answers on RMS-AQA, WNM-3D against UniWAM, and the budget.

**Next:** the literature session reports on (A) whether WNM-3D's own simulator can produce spatial audio and what it would take, (B) the two papers, (C) what existing audio-VLA studies cover and leave out, long-form audio first. The lead tells the planning lead what else the PI said. Then the research question and decision records by pull request. PR #68 was open when this was written.

**Open questions:** as under "Not verified"; whether the PI's long-form suggestion replaces the two candidate questions or adds to them.

## 2026-11-19 [agent, lead, planning] (PI follow-ups from the literature session: simulator audio, the two papers, audio-VLA gaps; PR #71)
**Goal:** get answers on the three things the PI's feedback names.

**Did:**
- The literature session ran three workers and a fact-check reviewer and opened PR #71 (`literature/survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md`). The planning lead relayed the provisional answers and then the reviewed changes to the lead in chat; the planning lead did not read the sources.

**PR:** #71 (the literature session's; open when this was written, merge waits for the lead's OK in that window). PR #68 was open. This entry and the `PLAN.md` notes: direct.

**Found** (as reported by the literature session from the PR #71 file, `verified` there by its workers and reviewer on 2026-11-19; nothing reproduced):
- A, WNM-3D's own simulator: possible with named extra work, not as released. GN0 / GN-Bench renders RGB and depth from Gaussian splats; the code was read at pinned commits and the reviewer's own search found no audio in GN0, GN-Bench or WNM-3D. InteriorGS ships a per-scene floor plan (walls with heights, door and window openings) and a labeled 3D box for every object. The collision meshes the same authors publish are convex-hull collision bodies, not the original triangle meshes, so walls and floors may have to come from the floor plan. No acoustic materials ship. Most direct engine: RLR Audio Propagation (the engine inside SoundSpaces 2.0), whose C interface takes meshes or boxes plus materials and returns mono, binaural or ambisonic impulse responses; Linux only, no ready Python wrapper. Fallback: pyroomacoustics built from the floor plan, lower fidelity. Precedent: pi-AVAS (ICCV 2025) builds a mesh from a Gaussian-splat or NeRF reconstruction and ray-traces sound through it. Nobody found doing this on InteriorGS, GN-Bench or a splat navigation simulator.
- B, the two papers the PI linked: "Hear You Are" (CVPR 2026, Ryu et al.; 1M question-answer pairs, 10 s clips, 360-degree panorama, Qwen2-7B) and HLR (arXiv 2610.21588; 175k train and 21.8k test question-answer pairs, Qwen2.5-Omni, three-stage training). Both are audio-visual question-answering language models, not acting models. Both make binaural audio with SoundSpaces 2.0 on Matterport-family scenes (32 kHz, one static viewpoint, one short clip per question) and build the spatial branch as a frozen Spatial-AST plus a small trainable projector. Nothing released by either. Hear You Are's "binaural helps much more out of view" holds on its direction and distance questions only, not on its out-of-view yes/no question. HLR has a control that feeds the left and right channels through the existing mono path.
- C, audio vision-language-action studies (about 20 models): the audio heard per decision is short, about 0.5 s (ADAPT), 1 s (S2A2), 2 s (ListenVLA), 2 to 3 s (ManiWAV), 4 s (HEAR); NaVLA2's paper gives no length and its released code uses 7 s mono and 10 s two-channel. RoboOmni takes 10 to 80 s dialogue clips as an instruction before acting, not a stream. Only HEAR carries an audio memory across decisions, and it is mono. ManiWAV has the only audio-length ablation (1 s and 10 s both worse than its default). "Long-form" in the audio literature means minutes to hours. The reviewer's own searches found no robot policy that takes long-form or streaming long audio.
- Not new: a sound that stops before the agent arrives. SAVN-CE (CVPR 2026) is built around targets that go quiet, with a memory model (MAGNet). The gap is narrowed to "in a vision-language-action or world action model".

Planning lead's suggestions to the lead, not decisions (`speculative`): the PI's long-form point and candidate question 1 could be merged (the same three audio versions, plus whether the model holds on to what it heard over an episode). Questions for the PI: what "long-form" means here (minutes of audio, or continuous listening with memory over a task), and whether using the RLR engine without Habitat counts as "not using SoundSpaces 2.0".

**Not verified:** how much work either simulator route is; whether the InteriorGS floor plans are enough for usable room acoustics; OpenReview blocked several submissions (SoundAct, Hear-to-Act, HearVGo), which rest on abstracts or project pages; the PI's meaning of "long-form"; the rest of the PI's meeting feedback (the lead has not relayed it yet).

**Next:** the lead's OK on PRs #68 and #71 in the literature window; the lead relays the rest of the PI's feedback and picks the simulator route with the PI; then the research question and decision records by pull request.

**Open questions:** stay in GN-Bench and build the audio, or move to Habitat; what the research question becomes after the long-form suggestion; RMS-AQA, base model and budget answers from the PI.

## 2026-11-19 [agent, lead, planning] (addendum: the PI's feedback is complete as recorded; budget position)
**Did:** asked the lead for the rest of the PI's feedback. The lead, 2026-11-19: "thats all the feedback i got, ignore the budget we have $200 before asking for more budget lets use what we have".

**PR:** direct.

**Found:** nothing new measured or read. What this settles and leaves open:
- The PI's feedback is the three points in the "PI feedback" entry above (WNM-3D's own simulator if it can make the data; two papers; pursue what audio-VLA studies do not cover, for example long-form processing). The PI gave no answer on which candidate question, on parking RMS-AQA, or on UniWAM. The PI's messages take WNM-3D as the base (the planning lead's reading).
- Budget (the lead): work within the 200 dollars first; asking for more comes later. Replaces "more asked for later if it is not enough" as a plan to raise with the PI now.
- RMS-AQA stays parked; the PI did not bring it back.

**Not verified:** what any option costs against 200 dollars (nothing timed).

**Next:** the lead chooses the simulator route and the research question (with the PI where needed); the planning lead then writes the question and the decision records by pull request. Before either simulator route is chosen it should be sized.

**Open questions:** simulator route; the research question after the long-form suggestion and what the PI means by "long-form"; the lead's OK on PRs #68 and #71.

## 2026-11-19 [agent, literature] (PRs #68 and #71 merged; NaVLA2 encoder check; reading list corrected)
**Goal:** answer the planning lead's literature requests of 2026-11-18 and 2026-11-19 (novelty of the candidate questions, the WNM-3D simulator gap, the adversarial second check, the PI follow-ups) and land them by pull request.

**Did:**
- PR #68 (`literature/survey/2026-11-18-novelty-and-simulator-gap.md`): parts 1 and 2 by two workers (2026-11-18), parts 3 and 4 by two more (2026-11-19), each round with a separate fact-check reviewer comment on the PR (own searches, raw sources); review fixes applied as new commits. Merged on the lead's OK ("merge", 2026-11-19).
- PR #71 (`literature/survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md`): parts A, B and C by three workers, one fact-check reviewer comment (one error and six imprecisions, all applied). Merged on the same OK.
- For the presentations session: read NaVLA2's AAAI-26 PDF (main text, `pdftotext`) for whether the same encoder ever gets one-channel and two-channel input. It does not: the only audio ablation (Table 2) switches a CLAP branch on the mono downmix and a SpatialAST branch on the binaural signal; no seeds or run count stated (`verified`, main text; the appendix is not in that PDF, so not checked).
- Reading list: the lead corrected the 2026-11-15 list (SAIL was not read) and has read WNM-3D (direct merge 5acb920).
- One script was refused by the permission classifier: a worker brief that told it to retry a refused (HTTP 403) PDF download with a browser-like request. The instruction was dropped; the paper then downloaded normally (HTTP 200).

**PR:** #68, #71.

**Found:** recorded in the planning lead's 2026-11-19 entries above and in the two survey files; not repeated here.

**Not verified:** OpenReview submissions behind HTTP 403 (SoundAct, Hear-to-Act, HearVGo and others) rest on abstracts or project pages; NaVLA2's appendix; whether InteriorGS's collision meshes include walls and floors.

**Next:** none for the literature session until the lead picks the simulator route and the research question.

## 2026-11-19 [agent, lead, planning] (sizing started: the lead's go; decision records 001 and 002 open as PR #72)
**Goal:** act on the lead's answers to the three open decisions.

**Did:**
- The planning lead had asked three things: whether to plan a sizing of the two simulator routes; what the research question becomes after the PI's long-form hint (suggestion: fold it into question 1, and ask the PI what "long-form" means); and the lead's OK on PRs #68 and #71. The lead, 2026-11-19: "1. Yeah plan that that sounds good 2. Yeah I agree, we can focus on long form tasks later 3. okay i answered". PRs #68 and #71 were merged by the literature session on the lead's word.
- Presented a plan. It stated the planning lead's reading of answer 2 ("Question 1 is the research question: does direction matter, with the three audio versions and the out-of-sight split. The long-form and memory angle comes later as an extension, and so does the encoder comparison. Correct me if that's off, because task 3 writes it down.") and four tasks: size route A (keep WNM-3D's rooms, add sound), size route B (move WNM-3D to Habitat with SoundSpaces 2.0), decision records for what is settled (the research question, WNM-3D as the base, navigation as the setting, with README and STANDARDS updates), and a comparison with a recommendation. The lead: "go".
- Downloaded read-only copies of eight public repositories at pinned commits to the session scratch directory for the workers (GN0 c927781, GN-Bench-Tools 0ce8367, WNM-3D 7fe2d5f, GN0-VLN-CE ac4ebed, rlr-audio-propagation 4fd446b, habitat-gs 8387881, NaVLA 436ad6b, sound-spaces 287184f). Nothing in them is run.
- Started two workers (Opus), one per route, each owning one memo in `writing/`.
- Wrote `decisions/001-research-question.md` and `decisions/002-base-model-and-setting.md` with matching `README.md` and `STANDARDS.md` updates, and opened PR #72. A reviewer is checking it.

**PR:** #72 (open). This entry and the `PLAN.md` Now items: direct.

**Found:** nothing measured or read for a result. A third way to get sound into WNM-3D's rooms came up in the literature file and is covered inside route A: Habitat-GS, a habitat-sim fork that renders Gaussian splats, lists InteriorGS scenes and keeps an audio build flag (`literature/survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md`, part A).

**Not verified:** anything about effort or cost for either route (that is what the memos estimate, and their figures will be estimates).

**Next:** the workers' reports; the comparison; one pull request for the three sizing files. The lead's merge OK on PR #72. Then the route choice (the lead and the PI), decision 003, and the experiment PLAN.

**Open questions:** the simulator route; what the PI means by "long-form"; the number of seeds against the 200 dollar budget (the general rule is at least 3 per reported comparison; three versions at three seeds each have not been costed).

## 2026-11-19 [agent, lead, planning] (sizing memos for the two simulator routes: PR #73 open and reviewed; PR #72 reviewed)
**Goal:** size the two ways to get audio-visual data and a closed-loop test for WNM-3D, and compare them.

**Did:**
- Two workers (Opus) each wrote one memo from the pinned code copies, nothing run: `writing/2026-11-19-sizing-route-a-audio-in-wnm3d-simulator.md` and `writing/2026-11-19-sizing-route-b-habitat-soundspaces.md`. The planning lead reran both done checks, wrote `writing/2026-11-19-sizing-comparison-simulator-routes.md`, and opened PR #73 with all three.
- A read-only reviewer (Opus) checked more than 12 code claims per memo against the pinned copies and posted 18 findings on PR #73 (4 must fix, 14 should fix). Each worker fixed its memo in a new commit after checking the findings in the code; the planning lead fixed the comparison. The resolution is a comment on the pull request.
- PR #72 (decisions 001 and 002): a read-only reviewer checked every quote against this LOG and every model fact against the survey files; one quote was out of order ("WNM-3D seems fine but can u reexplain the simulator gap" was said before the gap was explained again, not after) and several wordings were too strong (for example "the PI's preference" for a conditional sentence). Fixed in a second commit; resolution posted on the pull request.
- Chat with the lead on 2026-11-19 that the decision records cite and that was not logged before: the lead asked whether building on a Wan 2.2 world action model and adding an audio encoder was the wrong approach; the planning lead answered that adding inputs to a pretrained video model is how these models are built, named Cosmos 3 as the alternative with native two-channel audio, and recommended staying with the Wan 2.2 route. The lead also asked whether a simulator has to be built (answer: no, the simulators exist; what has to be built is the connection or the audio), and whether WNM-3D has its own simulator (answer: yes, GN-Bench, without audio).

**PR:** #73 (open), #72 (open, open). This entry and the `PLAN.md` Now item: direct.

**Found** (from the three files in PR #73; code claims `verified` there by the workers and sampled by the reviewer at the pinned commits; every effort figure is an estimate, `speculative`; nothing was run):
- Effort for the data and testing pieces, one person with coding agents: route A about 17 to 30 working days, at most about 24 to 42 with all add-ons; route B about 13 to 35, up to about 42 with a DAgger round in Habitat. The ranges overlap. Neither includes the audio encoder or any compute.
- Same in both routes: the audio encoder attaches in the same two places in WNM-3D's model code; RLR is the sound engine either way; WNM-3D's training frames are spaced by distance, not time, so seconds of audio per frame is a convention; the size of the not-visible group is unknown until computed; the trainer has a setting for one fixed instruction, so removing the target's name from the instruction is a configuration change; all three versions must share one pose source because the trainer derives action labels from poses.
- Training cost is the largest open item: the documented recipe is full fine-tuning on 8 GPUs with a per-device batch of 12; three versions at 3 seeds is 9 runs; the budget is 200 dollars; no training step has been timed.
- VGGT-Omega: the training launcher exits unless the checkpoint file exists (`scripts/train/wnm_3d_base.sh` lines 143 to 147), and the model is gated with manual approval, but when training starts from a released checkpoint the model code takes those weights from the checkpoint and does not open the standalone file. Ways to start without the file were read from the code and not tested. Requesting access stays the clean option; the choice is the lead's.
- Route A specifics: GN-Bench instructions name the target and often its room; the released checkpoint already trained on the train_seen episodes; geometry would come from floor plans and object boxes with assumed materials.
- Route B specifics: Matterport3D or HM3D access takes a request and calendar time; the released weights have never seen Habitat scenes; on HM3D the continuous SoundSpaces simulator cannot be used as is, because it needs per-scene metadata SoundSpaces ships only for its own scene sets.
- Sound through walls is on by default in the bare RLR library and in SoundSpaces' continuous simulator, and off in the Habitat-GS wrapper's documented defaults.

Correction to what the planning lead told the lead in chat: "training won't start without VGGT-Omega, so access has to be requested" was too strong; see the VGGT-Omega point above. Corrected in chat.

Planning lead's recommendation, not a decision (`speculative`): start with route A and commit to it only after real checks come back (the PI's answer on whether RLR outside Habitat counts as not using SoundSpaces 2.0; the InteriorGS and VGGT-Omega access steps; whether the engine runs on a rented Linux machine and the floor plans line up with the simulator's map; one timed training step; the visible and not-visible counts). Route B as the fallback.

**Not verified:** every effort figure; whether RLR runs on a current Linux image; what a training step costs; the InteriorGS terms on derived data; the launcher workarounds.

**Next:** the lead's merge OK on PR #72 and PR #73. Then the route choice with the PI (decision 003), and, if the lead wants them, the checks above planned for the experiment session, each with the lead's OK there.

**Open questions:** route A or B; seeds against the budget; request VGGT-Omega access or use a workaround; what the PI means by "long-form".

## 2026-11-19 [agent, lead, planning] (PRs #72 and #73 merged; route A first; access check)
**Goal:** act on the lead's answers to the four open decisions (merge PR #72, merge PR #73, the route, VGGT-Omega).

**Did:**
- The lead, 2026-11-19: "merge both prs 3. yeah thats fine 4. i think i might already have this token but give me the link to go check". Point 3 answered the planning lead's recommendation "to start with WNM-3D's own simulator and commit only after a few real checks come back".
- Changed the Status lines of `decisions/001` and `002` to accepted, then merged PR #72 and PR #73 (`gh pr merge --merge --delete-branch`). Removed the planning lead's two worktrees and both workers' worktrees and branches.
- Checked the lead's saved Hugging Face login with HEAD requests only (nothing downloaded): `facebook/VGGT-Omega`, file `vggt_omega_1b_512.pt`, returned HTTP 302 (a redirect to the file, so access is granted); `spatialverse/InteriorGS`, file `.gitattributes`, returned HTTP 403 (no access yet); the login itself answered 200 (`verified`, 2026-11-19).
- Rewrote `PLAN.md` "Now" and "Next" around the route decision. Items removed from "Now" because they are settled or replaced: "Scope the direction with the PI" (the meeting was held; decisions 001 and 002), "Choose the base model" (decision 002), "Find the data" (no ready set was found in three search rounds; the data is to be rendered, by route A or route B).

**PR:** #72 and #73, both merged. This entry and the `PLAN.md` rewrite: direct.

**Found:** nothing measured beyond the access check above.

**Decided by the lead:** route A first (WNM-3D's own simulator plus a sound engine), not committed until real checks come back; route B (Habitat with SoundSpaces 2.0) as the fallback. Decision `003` is written when the checks are back. No check is approved yet.

**Not verified:** that the 403 on InteriorGS means the terms are not accepted (it fits; `.gitattributes` may also be refused for another reason); everything listed under "Not verified" in the sizing entry above.

**Next:** the lead accepts the InteriorGS terms. With the lead's go, the planning lead hands the three checks to the experiment session as direction; each rental needs the lead's OK in that session. The PI questions go to the next Thursday meeting.

**Open questions:** whether the engine runs and the floor plans line up; what a training step costs; seeds against the budget; the PI's answers.

## 2026-11-19 [agent, lead, planning] (the three route A checks handed to the experiment session)
**Goal:** hand the checks that decide the simulator route to the experiment session.

**Did:**
- The planning lead asked "hand the checks to the experiment session?" with a recommendation of yes, stating that the experiment session would ask the lead in its own window before renting anything. The lead, 2026-11-19: "yes".
- Sent the hand-off to the experiment session (`hear-lab-experiment`), split into decided, the three checks, access, not decided, and not approved. The checks: (1) whether RLR-Audio-Propagation runs on a Linux machine and returns a one-channel and a two-ear impulse response for one real InteriorGS scene, and whether the floor plans line up with the simulator's occupancy map; (2) one timed real training step from the released WNM-3D checkpoint on the GPU type that would be rented; (3) counts of visible and not-visible starts on the real GN-Matrix episodes. Suggested order 1, 3, 2, with a stop and report if check 1 fails.
- Rechecked InteriorGS access with the lead's saved login (HEAD request only): HTTP 403 (`verified`, 2026-11-19).

**PR:** direct.

**Found:** nothing measured beyond the access recheck.

**Not approved:** any rental or spending; any training beyond the timed step; audio encoder work; route B work. Each rental needs the lead's OK in the experiment session.

**Next:** the lead accepts the InteriorGS terms if not done; the experiment session writes the checks' experiment `PLAN.md` and asks the lead before each rental; its results come back by pull request. Then decision `003`.

**Open questions:** as in the entry above.

## 2026-11-20 [agent, lead, planning] (status check; InteriorGS access granted; RMS-AQA organizers' reply; read-first list)
**Goal:** answer the lead's "whats the status", and record what arrived today.

**Did:**
- Rechecked InteriorGS access with the lead's saved Hugging Face login (HEAD request only, nothing downloaded): `spatialverse/InteriorGS`, file `.gitattributes`, returned HTTP 200 (`verified`, 2026-11-20; it was 403 on 2026-11-19). The experiment session reports the same from its side (a redirect on a scene file; its report, not rechecked here).
- Asked the experiment session (`hear-lab-experiment`) where the three route A checks stand. Its reply (its own report, not checked by the planning lead): hand-off received; a plan presented to the lead in its window (two rentals: a cheap single-GPU machine for checks 1 and 3, then a large card for check 2); nothing started, no folder or branch. It is waiting on the lead for: the go on the checks' `PLAN.md` and the check 1 and 3 code; the GPU type for check 2 (it recommends one H200); whether to keep the launcher's VGGT-Omega file check (it recommends yes, download the real file); an SSH key on the Vast.ai account. It also reports that the only 8-GPU offers on Vast.ai are 8x H200 at about 38 dollars an hour, so the documented 8-GPU recipe does not fit the 200 dollar budget (`speculative` here: its reading of the offer list, no price checked by the planning lead).
- The lead pasted the RMS-AQA organizers' answer to the lead's email of 2026-11-18 (a screenshot of the reply, received 2026-11-20, signed "RMS-AQA Organizing Committee"): "Due to licensing restrictions, the dataset is currently available only to registered challenge participants. Please submit the team registration form first to obtain access." So research-only access without a team registration is refused (`verified`, read from the screenshot).
- On the literature session's request (the lead asked it what to read before it matters), sent it an ordered read-first list of the papers whose methods the plan uses, each marked decided or open, and gave the lead the same list in chat. Order: SoundSpaces 2.0 (the sound engine, both routes), GN0 / GN-Bench (WNM-3D's simulator and episodes), BAT (Spatial-AST, the leading encoder candidate in the planning lead's reading; the encoder is open), the two papers the PI linked (Hear You Are; Hear, Localize, and Reason), DreamZero (WNM-3D's video-action design), Audio-Visual World Models (overlap, not a method used). Can wait: SoundSpaces and SAVi (task and metrics), VGGT-Omega, the Wan paper, pi-AVAS, other encoder candidates. The list was written from the survey files, the decision records and the sizing comparison; no paper was opened today. Several are not in `references.bib`; the Wan citation was given from memory (`recalled`). The literature session checks citations before the reading list changes.
- Updated the project brief (outside the repository) with the access change and the experiment session's state.

**PR:** direct.

**Found:** nothing measured. Access: InteriorGS and VGGT-Omega are both open on the lead's login, so no check is blocked on access.

**Not verified:** the Vast.ai prices and offer list; anything in the experiment session's plan; the citations marked not in the bibliography.

**Next:** the lead answers the experiment session's pending choices in its window. RMS-AQA stays parked; whether to register a challenge team is the lead's and the PI's call (a registration signals a challenge entry under the lab's name), and nothing in the current plan needs the dataset. Decision `003` after the checks.

**Open questions:** if the 8-GPU recipe cannot be rented within the budget, training on fewer GPUs is a change from the documented recipe that needs the lead's approval and a recorded note; the timed step comes first. The PI questions listed in `PLAN.md`.

## 2026-11-20 [agent, experiment] (route A checks 1 and 3 run; PR #75)
**Goal:** run the three route A bring-up checks handed over on 2026-11-19.

**Did:** wrote the experiment folder `experiments/2026-11-20-route-a-checks/` (PLAN before running, one dated amendment before any rented run); two Opus workers wrote the check 1 and check 3 code. On the lead's OKs in the experiment window, rented one RTX 3090 on Vast.ai (a first attempt failed to schedule and was destroyed at once), ran check 1, a diagnostic for its failed ear test, and the full check 3, rendered 48 training episodes for check 2, then destroyed the machine. Total spend 0.96 USD of the 200. Check 2 not run: its GPU and the VGGT-Omega question are with the lead.

**PR:** #75 (open, reviewer comment pending).

**Found** (detail and exact numbers in the experiment `LOG.md` and `README.md`; all `verified` this session):
- The sound engine (RLR-Audio-Propagation, bare library) runs on a current Linux image, returns one-channel and two-ear impulse responses for a real InteriorGS scene, with arrival times matching physics, and is fast (well under a second per response on one CPU thread).
- Its default head model, as configured, gives a level difference between the ears but almost no time difference. The pre-set left-ear-first test therefore failed. Not yet tried: a measured HRTF file or a higher spherical-harmonic order.
- Floor plans line up with GN-Bench's maps by eye in four scenes; the pre-set overlap score passed in two of four (misses on unknown border cells and openings). Check 1 is "not passed" by its own rule.
- About one start in five in test_unseen has the target in view at the start (1114 of 5000); about a third in the train_seen sample. The not-visible group is large.
- The GN-Matrix goal position is a median of about 1.3 m from the target object (test_unseen), with a quarter of episodes more than 3.6 m away: relevant to putting the sound on the object.

**Next:** the lead decides the check 2 GPU and the VGGT-Omega file check; reviewer comment on PR #75; the lead's call on whether check 1's misses block route A, and on trying a measured HRTF.

**Open questions:** does a level-only two-ear cue matter for the research question; InteriorGS terms (PDF) not yet read.

## 2026-11-20 [agent, experiment] (raw outputs lost and regenerated, measured-HRTF follow-up, check 3 rerun, check 2 designed and coded; PRs #75 to #79)
**Goal:** finish the route A checks the user approved on 2026-11-20.

**Did:** merged PR #75 on the user's OK. Removing its worktree deleted the gitignored raw outputs of checks 1 and 3 (the machine was already gone); recovered the overlays and review images from the session, regenerated the rest on two short Vast.ai rentals, and keep raw outputs in `~/Desktop/hear-lab-route-a-results/` from now on. Rental 3 (GTX 1660 Super, CPU work, about 0.004 USD by the hourly rate): check 1 rerun, identical numbers, and the measured-HRTF follow-up (PR #76, merged). Rental 4 (RTX 3090, 1.26 h, about 0.60 USD with downloads): check 3 rerun, every recorded count identical (PR #79, open). Wrote and reviewed the check 2 design (PR #77, merged, with the user's four decisions) and its code: timing and memory callback, start-weights key check, RUN.md (PR #78, merged). Account credit 198.40 USD of 200.

**Found** (exact numbers in the experiment `LOG.md`; `verified` this session):
- With a measured head file (MIT KEMAR) the sound engine gives a time difference between the ears of about 0.7 ms at 90 degrees, left ear first for a source on the left, and check 1's left-ear test passes. Check 1's recorded result (library default head) stays a FAIL. SoundSpaces 2.0 most likely uses that default head.
- The visible and not-visible counts of check 3 reproduce exactly on a second machine.

**Decided by the user:** check 2 on one H200 with the real VGGT-Omega file, no gradient accumulation, only the needed downloads, stage 3 release start, a key-count check; the token is typed by the user at a hidden prompt.

**Next:** the user's OK for the replacement H200 offer (Japan, about 3.94 USD/h, 2.5 to 4.5 h, hard stop 5 h); then run check 2 per `experiments/2026-11-20-route-a-checks/src/check2/RUN.md` (the user types the token about 20 to 40 minutes in). PR #79 waits for the user's merge OK. No instances exist.

## 2026-11-21 [agent, experiment] (check 2 run on one H200; PR #79 merged; PR #80 open)
**Goal:** run route A check 2 (one timed training step) and report.

**Did:** merged PR #79 on the user's OK. Rented one H200 on Vast.ai with the user's OK in the experiment window (Japan, 3.8048 USD/h, 2.15 h, 8.20 USD by the hourly rate; credit 198.39 to 190.11 USD), ran check 2 per its RUN.md, copied and checked all outputs, destroyed the machine. Results in the experiment `LOG.md` (2026-11-21) and `README.md`, PR #80 (reviewer comment pending). Total spent on the 200 USD so far: about 9.89 USD.

**Found** (`verified` this session; exact numbers in the experiment LOG):
- The released full fine-tune with its DeepSpeed ZeRO-2 config does not fit on one H200 at any batch size from 12 down to 1: it runs out of memory when the optimizer state is set up at the first step.
- With the PLAN's fallback (optimizer state offloaded to CPU, a deviation from the released config) batch 12 fits at about 35 s per step; the GPU is busy about a third of the time, most likely because the released launcher caps CPU threads at one (not profiled).
- Deviations: two library versions repaired (huggingface_hub, pyarrow), the saved Hugging Face token copied over on the user's request instead of typed, 46 of 48 episodes (the user's OK), key check run with the GPU visible.

**Next:** reviewer comment and the user's merge OK on PR #80. Planning decides how training should run given that one H200 cannot hold the released recipe (offload with more threads, a parameter-efficient set-up, or more GPUs); then decision `003`. No instances exist.
