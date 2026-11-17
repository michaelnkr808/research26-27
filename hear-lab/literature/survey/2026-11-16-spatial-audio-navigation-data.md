# Literature and data survey: spatial audio with egocentric video and navigation actions

Date: 2026-11-16. Direction: hear-lab (spatial audio for world action models). Question from the user: which datasets exist like the one NaVLA2 trained on (its own MINav set: about 43.9K navigation episodes, egocentric video, binaural audio rendered with SoundSpaces 2.0 on Habitat, discrete navigation actions). Scope: datasets, benchmarks and simulators that give spatial audio together with egocentric video and navigation actions, or egomotion that could stand in for actions, both simulated and real. Related papers on audio encoders and spatial audio language models are in `2026-11-16-spatial-audio-in-embodied-ai.md` and are not repeated here, except where one of its navigation entries is also a data source. That file was used for leads only; every release claim below was rechecked.

How the search was done: web searches (queries listed at the end) found candidates. Every item below was then opened on a primary page: arXiv metadata through the arXiv API (`export.arxiv.org/api/query?id_list=...`), paper text through the arXiv HTML page or a downloaded PDF read with `pdftotext`, repository READMEs through `raw.githubusercontent.com`, Hugging Face repositories through `huggingface.co/api/datasets/<id>` (file lists and gating flags, not downloads), and project or download pages through `curl -s -L` with the HTML stripped by a small script. Downloads were kept in a scratch directory outside the repository and treated as untrusted data; nothing was executed. No dataset payload was downloaded. For the SoundSpaces download links only the HTTP headers were fetched (status and content length). Where a number is quoted, the location in the source is named. Venues come only from the arXiv comments or journal-reference field or another primary page, and anything else is marked "preprint, not peer reviewed". The current date being 2026-11-16, several arXiv identifiers start with 26.

One important framing note first. The answer to "what is like MINav" is mostly: SoundSpaces-based simulated data, and in nearly every case the user gets it by running a simulator, not by downloading a finished set. For the exact MINav and AVW-4k setting (egocentric frames, binaural audio, discrete actions) no finished download was found (see the entries). The few finished downloads found are the pre-rendered SoundSpaces room impulse responses, the FAR latent corpora (gated, already encoded), and OmniEchoBench (real recorded first-order Ambisonics in scanned rooms, public). Real recordings with spatial audio plus camera motion exist (Ego-Exo4D, Aria Everyday Activities, Nymeria, StereoWalks), but they are people wearing glasses or phones, not navigation agents with an action space.

Labels: `verified` means seen on the page named in the entry this session.

## Simulators and simulated navigation sets

### chen2020soundspaces: SoundSpaces: Audio-Visual Navigation in 3D Environments (SoundSpaces 1.0)
- Paper and link: Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, Kristen Grauman. 2020. https://arxiv.org/abs/1912.11474 (v3). Venue: ECCV 2020 (Spotlight), per the arXiv comments field (verified via the arXiv API).
- Read depth: project page https://soundspaces.org/ (stats), repository README and `soundspaces/README.md` and `INSTALLATION.md` (raw), HTTP headers of the download links. The paper text itself was read in the earlier survey file (sections 3, 5, 6); not re-read here.
- What it contains: pre-rendered audio (room impulse responses between a grid of nodes in scanned scenes), per-scene metadata (node coordinates and a connectivity graph), episode datasets for AudioGoal navigation, and 102 mono sound files. The project page says: "85 scenes from Matterport3D, 18 scenes from Replica, 102 copyright-free sounds, Total 17.6M RIRs (16.7M for Matterport3D and 0.9M for Replica)".
- Spatial audio format: binaural, 2-channel RIRs, per agent heading angle (data README, "Data Folder Structure": `binaural_rirs ... [angle]/[receiver]-[source].wav`, "binaural RIRs of 2 channels"). A larger "full ambisonic" RIR set also exists (data README step 4). Sample rate: not stated on the pages read.
- Visual observations: not part of the RIR download. The agent's RGB and depth come from rendering the scene in habitat-sim; `scripts/cache_observations.py` caches them (needs habitat-sim and habitat-lab at v0.1.7 for the pretrained models, per `INSTALLATION.md`).
- Actions or motion: agent moves between nodes of a grid-like navigation graph; the AudioGoal task has an action set of stop, move forward, turn left, turn right (stated for the SoundSpaces setting in the AVLEN paper, Section 3, see the `paul2022avlen` entry). Continuous navigation agents are also supported (README, "[New] Training continuous navigation agent", which uses SoundSpaces 2.0).
- Size: from `soundspaces/README.md` Download section: "partial binaural RIRs (867G), metadata (1M), datasets (77M) and sound files (13M)". The HTTP headers for `binaural_rirs.tar` returned a content length of 931,213,506,560 bytes (about 867 GiB, consistent). Optional full RIRs: "full ambisonic (3.6T for Matterport) and binaural (682G for Matterport and 81G for Replica)".
- Simulated or real: simulated (geometric acoustic simulation in real 3D scans).
- How to get it: (1) download the five tar files with `wget http://dl.fbaipublicfiles.com/SoundSpaces/{binaural_rirs.tar, metadata.tar.xz, sounds.tar.xz, datasets.tar.xz, pretrained_weights.tar.xz}` (all five returned HTTP 200 this session); (2) separately obtain Matterport3D (official `download_mp.py --task habitat`, needs a terms-of-use agreement) and Replica; (3) install habitat-sim and habitat-lab at v0.1.7 and run `scripts/cache_observations.py`. README states: "Note that we do not open source the rendering code at this time" for SoundSpaces 1.0, so new scenes or sounds beyond the grid cannot be rendered with 1.0; use SoundSpaces 2.0 for that.
- Release status: https://github.com/facebookresearch/sound-spaces shows the repository as archived (GitHub API `archived: true`, last push 2023-09-29), with the data download instructions above intact. `verified`.
- License and prerequisites: code CC-BY-4.0 (README "License"). Per the README, "the trained models and the task datasets are considered data derived from the correspondent scene datasets": Matterport3D based ones are "distributed with Matterport3D Terms of Use and under CC BY-NC-SA 3.0 US", Replica based ones fall under the Replica license. So the Matterport3D terms (non-commercial) apply to anything derived from it, including world-model data.
- Used by: the SoundSpaces AudioGoal baselines in the same repository; AVLEN (`paul2022avlen`), LH-AVLN (`chen2026lhavln`), AVW-4k (`wang2025audio`), MINav (`fan2026navla`), the FAR corpora (`kim2026learning`), as stated in those papers' own text (see their entries).

### chen2022soundspaces2: SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning
- Paper and link: Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip W Robinson, Kristen Grauman. 2022. https://arxiv.org/abs/2206.08312 (v2). Venue: NeurIPS 2022 Datasets and Benchmarks, per the project page https://vision.cs.utexas.edu/projects/soundspaces2 ("NeurIPS 2022", read this session) and the repository's own citation block. The arXiv comment says only "Camera-ready version".
- Read depth: project page and repository docs (`SoundSpaces2.md`, `INSTALLATION.md`, README), arXiv HTML text grepped; the paper body was read in the earlier survey file.
- What it contains: not a dataset but a simulator. It renders room impulse responses on the fly for any 3D mesh, using the RLRAudioPropagation ray tracer inside habitat-sim, so the agent can move continuously and any source sound can be inserted.
- Spatial audio format: configurable channel layout: `SoundSpaces2.md` lists Mono, Binaural (2 channels, HRTF), Ambisonics, and others in the channel-layout table; default sample rate in the parameter table is 44100.
- Visual observations: whatever habitat-sim renders (RGB, depth, semantics) for the same agent pose.
- Actions or motion: whatever the habitat-lab task defines; the repository README includes training of a continuous AudioGoal navigation agent.
- Size: no fixed dataset. Cost is rendering time per step. `INSTALLATION.md` names open issues (audio rendering crashes with some semantic annotations; GLIBC below 2.29 needs extra build steps).
- Simulated or real: simulated.
- How to get it: follow `INSTALLATION.md`: conda environment (python 3.9), habitat-sim branch `RLRAudioPropagationUpdate` built with `--headless --audio`, habitat-lab v0.2.2, `pip install -e .` of sound-spaces, scene datasets placed in `data/scene_datasets` (Replica, Matterport3D, Gibson, HM3D) following habitat-sim `DATASETS.md`, and the material file `mp3d_material_config.json`. HM3D needs access through Matterport's HM3D page (habitat-sim `DATASETS.md`, "Getting access"); Matterport3D uses the official download script.
- Release status: same repository as above, archived, last push 2023-09-29 (GitHub API). `verified`. The project page https://vision.cs.utexas.edu/projects/soundspaces2 lists "[Website] [Code] [Paper] [Bibtex]". The installation file ties it to older habitat-sim and habitat-lab versions, which is the practical risk (AudioWorldSim below says the same).
- License and prerequisites: code CC-BY-4.0; scene dataset terms as for 1.0 (Matterport3D, HM3D, Replica, Gibson each have their own terms).
- Used by: MINav (`fan2026navla`, Section 3.4: "habitat simulation platform ... and SoundSpaces 2.0"), AVW-4k (`wang2025audio`), LH-AVLN (`chen2026lhavln`), FAR (`kim2026learning`), AudioWorldSim (`zerkowski2026audioworldsim`), and the sim-to-real paper `chen2024sim2real` (IROS 2024 per the arXiv comment; it built an acoustic field dataset on SoundSpaces 2.0 and a real Hello Robot platform with a binaural microphone; I did not find a released real-robot dataset in the text read).

### fan2026navla: NaVLA2 and the MINav dataset
- Paper and link: Jugang Fan, Peihao Chen, Changhao Li, Qing Du, Jian Chen, Mingkui Tan. 2026. AAAI-26, doi:10.1609/aaai.v40i22.38886; PDF https://tanmingkui.github.io/files/publications/NaVLA2.pdf (read with `pdftotext` this session). Code: https://github.com/felixfjg/NaVLA.
- Read depth: the PDF abstract, Sections 3.1 to 3.4 and 4 (grepped for the dataset, simulator and training-sample statements); repository README, file tree and two build scripts.
- What it contains: MINav, multimodal instruction navigation. Instructions mix object categories, images, language descriptions and auditory descriptions. The paper's abstract: "a large-scale dataset of 43.9K navigation episodes". Scenes are HM3DSem scenes (Section 3.2). Audio clips for objects were collected through the Freesound API with GPT-written descriptions and a BERT similarity filter of 0.5 (Section 3.2). A "test-mini subset of 360 episodes" (Section 3.3). NaVLA2 was finetuned on "379K training samples" built from navigation episodes (Section 4).
- Spatial audio format: binaural, rendered at each step by SoundSpaces 2.0: "add a pair of microphones at the agent's height to simulate binaural audio" (Section 3.4). The audio sensor is only enabled in episodes where an object has an auditory reference. Sample rate and channel handling beyond two channels: not stated in the sections read.
- Visual observations: egocentric RGB per step (Section 3.1).
- Actions or motion: discrete A = {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}; "FORWARD moves the agent forward by 25cm and the turns rotate it by 30 degrees"; success is stopping within 3 m of the goal (Section 3.1).
- Size: 43.9K episodes; training samples as above. Hours and scene count: not found in the sections read.
- Simulated or real: simulated.
- How to get it: the finished MINav data is not downloadable from what was found. The repository README's TODO list, copied as shown: "[x] Code Release", "[x] Data Collection: Released scripts for MINav data collection", "[x] Evaluation", then unchecked "[ ] Model Weights: Release pretrained NaVLA checkpoints.", "[ ] Dataset: Release the processed MINav dataset.", "[ ] Training: Release NaVLA training scripts." So one would have to regenerate it: install the benchmark environment (`install_env_bench.sh`, which pulls habitat and SoundSpaces under `minavbench/third_party`), obtain HM3D with semantics, and run the scripts in `minavbench/build_datasets`. Two of those scripts look like stubs in the repository as fetched today: `2_construct_qa_dataset.py` and `3_construct_nav_dataset.py` are each 99 bytes and contain only two functions whose bodies are `pass` (raw files fetched and read). Whether the full pipeline runs from the repository alone is therefore unconfirmed; the paper's episode generation (Section 3.3) would need to be re-implemented or requested from the authors.
- Release status: https://github.com/felixfjg/NaVLA: public, Apache-2.0 (GitHub API), created 2025-11-17, last push 2026-03-14. The README says it contains "the implementation of NaVLA2 and the MINav benchmark generation tools". A Hugging Face search for "MINav" returned one repository `minthwi2505/minav-packs` (public, created 2026-07-06) whose file list holds only `.gitattributes`; I could not connect it to these authors and it holds no data. A search for "NaVLA" among models returned nothing. `verified` as of 2026-11-16.
- License and prerequisites: code Apache-2.0; HM3D access agreement; Freesound clip terms are per clip (not checked).
- Used by: NaVLA2 itself. Table 1 of the paper compares against baselines (RL, IL, Navid, CA-Nav) trained or adapted on the same task.

### wang2025audio: AVW-4k (Audio-Visual World Models)
- Paper and link: Jiahua Wang, Leqi Zheng, Jialong Wu, Yaoxin Mao, Shijie Cheng. 2025 (v1 30 Nov 2025, v4 read). https://arxiv.org/abs/2512.00883. No venue on the arXiv page: preprint, not peer reviewed.
- Read depth: arXiv HTML text, sections "Formulation and Dataset" and "AVW-4k Dataset" in full; searched the whole text and the page links for a code, data or project URL.
- What it contains: "approximately 30 hours of audio-visual data, spanning 76 indoor scenes and 4,500 trajectories" (section AVW-4k Dataset). One sound source per scene playing a looping ringtone; trajectories sampled from source-approaching, source-receding and source-independent patterns; no more than 500 frames per sample.
- Spatial audio format: binaural, "0.15-second binaural audio segment sampled at 16 kHz", time-aligned with each frame; simulated with SoundSpaces 2.0 on Matterport3D.
- Visual observations: "egocentric RGB image with a resolution of 128 x 128 pixels" per frame.
- Actions or motion: four actions: "move forward by 0.15 m, turn left by 10 degrees, turn right by 10 degrees, and stop".
- Size: as above (the splits follow in the paper's next sentence; not copied).
- Simulated or real: simulated.
- How to get it: not stated. I found no download link, repository or project page in the paper text or its page links (the only links are arXiv's own feedback links). A GitHub repository search for "AVW-4k", "AV-CDiT" and "audio-visual world model binaural" and a Hugging Face dataset search for "AVW-4k" returned nothing relevant. To reproduce it, one would follow the data-collection description (Matterport3D plus SoundSpaces 2.0, looping ringtone, the action set above) with the simulator route described under `chen2022soundspaces2` or `zerkowski2026audioworldsim`.
- Release status: not found. The arXiv page names no code or data release (page text and links checked). This is a statement about what that page and my searches showed, not proof of absence.
- License and prerequisites: not stated; Matterport3D terms would apply to derived renderings.
- Used by: the paper's own AV-CDiT model (a diffusion transformer that jointly predicts video and binaural audio), which is also the one found model that predicts future video and spatial audio conditioned on actions.

### zerkowski2026audioworldsim: AudioWorldSim
- Paper and link: Luis Vitor Zerkowski, Luiz Velho. 2026 (21 Aug). https://arxiv.org/abs/2608.21075 (v1). Technical report, 7 pages, no venue: preprint, not peer reviewed. Code: https://github.com/Luizerko/AudioWorldSim.
- Read depth: arXiv abstract page; repository README in full; repository root listing.
- What it contains: a custom simulator on top of SoundSpaces 2.0 for automatic rollouts of random-agent navigation to a sound source, with a fix for clicking artifacts between simulation steps. Outputs per run: `action_list.txt`, `navigation.jpg`, `output.wav`; the audio processing script adds Mel and STFT arrays split per action (README, "Creating Your Own Dataset"). It produces audio and actions only; I saw no egocentric RGB output in the README.
- Spatial audio format: binaural; default `--sample_rate` 44100 Hz (README, "Configuration Arguments").
- Visual observations: none listed (navmesh plot only).
- Actions or motion: "1: forward, 2: turn left, 3: turn right"; a forward action moves 0.2 m and the default time step is 0.2 s.
- Size: README states a benchmark of "1000 simulations (yielding ~4 hours of audio) took ~3 hours" on a 36-core machine with 30 workers (~28 GB RAM), for one Matterport scene `17DRP5sb8fy`; about 1 percent of runs get stuck and about 5 percent crash.
- Simulated or real: simulated.
- How to get it: the dataset itself is not released. README: "Due to derivation distribution limitations on both 3D scene datasets, we cannot provide open access to the dataset itself. Instead, we are open-sourcing the simulator so you can build your own." One must request Matterport and/or Replica, install SoundSpaces as in `INSTALLATION.md`, then run `custom_simulator/simulator.py` and `generate_dataset.sh`.
- Release status: https://github.com/Luizerko/AudioWorldSim is public (GitHub API: created 2026-04-20, last push 2026-09-13). No license field in the GitHub API; the README tells users to check SoundSpaces' license. `verified`.
- License and prerequisites: no license stated for the repository (GitHub API `license: null`); Matterport and Replica terms; SoundSpaces CC-BY-4.0 base.
- Used by: its authors' BinauralVAE project (repository `Luizerko/audio-nav`, described in its README as built on this simulator; I read its README only).

### kim2026learning: FAR datasets (SoundSpaces corpora of audio-visual navigation episodes)
- Paper and link: Beomsu Kim, Chieh-Hsin Lai, Bac Nguyen, Amir Bar, Jong Chul Ye, Yuki Mitsufuji. 2026 (28 Sep). "Learning What to Recall: Adaptive Multi-Cue Episodic Memory for World Models". https://arxiv.org/abs/2609.34677 (v1). arXiv comment: "Preprint": not peer reviewed.
- Read depth: arXiv HTML text (dataset sections, Table 3, Appendix C.2 "SoundSpaces"); code repository README; Hugging Face API record of the dataset repository. The gated dataset card and files were not opened (access denied without login).
- What it contains: two paired corpora ("SoundSpaces v1" and "SoundSpaces v2") rendered in Matterport3D: "14,425 episodes over 81 eligible scenes with a scene-disjoint split of 57 training and 24 test scenes" (Appendix C.2). An agent traverses an ABA or ABCA loop with two to four static sound sources playing; the outbound path is memory and the return leg is the prediction target. Table 3 lists frame size 256 x 256 and "binaural audio" as the extra signal.
- Spatial audio format: "spatialized binaural audio rendered at 16 kHz" (Appendix C.2). In the released data the audio is stored as pre-computed latents, not as waveforms (see below).
- Visual observations: RGB frames at 256 x 256 (Table 3); v2 adds scans at region changes.
- Actions or motion: the paper says the world model is conditioned on selected frames "together with pose, time, and action information"; the specific action set was not found in the text read.
- Size: 14,425 episodes per corpus. The Hugging Face record of `1202kbs/FAR-Datasets-SoundSpaces` lists 34 files, including `soundspaces/train-000.tar` through `train-008.tar` and `soundspaces_v2/train-000.tar` through `train-009.tar` plus `test-000.tar` and `demo.tar` per corpus; the API field `usedStorage` is 422,845,808,640 bytes (about 423 GB; it may count repository history, so treat as an upper bound).
- Simulated or real: simulated.
- How to get it: README ("Downloads"): "Pre-computed latent corpora ... `bash bash_scripts/download.sh train soundspaces_v1` (also `soundspaces_v2`)", with the note that SoundSpaces data "is gated: accept the Matterport3D Terms of Use there, then run `hf auth login`". The dataset record's description says "Pre-computed latent corpora of audio-visual navigation episodes". So what can be downloaded is encoded latents (tokenized video and audio features) made for this model, not raw video and audio.
- Release status: https://huggingface.co/datasets/1202kbs/FAR-Datasets-SoundSpaces: public repository, "gated": "auto" (immediate approval after accepting the terms), created 2026-09-29. Code: https://github.com/sony/far (README with downloads and launchers; license badge CC BY-NC 4.0 on the README, repository license field not separately checked). `verified` via the API; contents behind the gate not seen.
- License and prerequisites: dataset card metadata: `license: other`, `license_name: matterport3d-terms-of-use`, with the prompt "provided for non-commercial academic research only, under the Matterport3D Terms of Use". Prerequisite: a Hugging Face account and acceptance of the gate.
- Used by: FAR (the paper's latent-diffusion world model with action-conditioned retrieval memory); checkpoints for these corpora are also on Hugging Face (README table).

### chen2026lhavln: LH-AVLN
- Paper and link: Rufeng Chen, Yue Chang, Zili Shao, Zhaofan Zhang, Li Chen, Hechang Chen, Hui Xiong, Sihong Xie. 2026 (4 Jul). https://arxiv.org/abs/2607.03920 (v2). arXiv comment "12 pages, 3 figures", no venue: preprint, not peer reviewed.
- Read depth: arXiv HTML text: abstract, Section 3 (task and construction), experimental setup, appendix A.2; searched the text and page links for a release statement.
- What it contains: long-horizon multi-goal navigation: missions of two to four goals specified by category, language description or reference image; each goal has a sound source that alternately emits cues. Evaluation on "the validation split with 550 full-goal episodes". Total episode count and hours: not found in the sections read.
- Spatial audio format: binaural, rendered according to the relative pose between agent and the active source; built on SoundSpaces 2.0 with Matterport3D scenes; sensors at 1.0 m height. Sample rate: not found.
- Visual observations: RGB-D plus pose.
- Actions or motion: A = {MoveForward, TurnLeft, TurnRight, Submit, Stop}; success if Submit or Stop within 2 m of the target.
- Simulated or real: simulated.
- How to get it: no download found. The benchmark is generated by sampling episodes in Matterport3D scenes with SoundSpaces 2.0 as described in Section 3.2; no episode files or code were located.
- Release status: not found. The arXiv HTML page and its links show no repository or data URL; a GitHub repository search for "LH-AVLN" and "PAG-Nav" and a Hugging Face dataset search for "LH-AVLN" returned nothing relevant. Statement is limited to those pages and searches.
- License and prerequisites: paper text CC BY 4.0 (arXiv); Matterport3D terms for any rendering.
- Used by: PAG-Nav (training-free agent) and MAV-Nav (learning-based policy, appendix A.1), both from the same paper.

### gan2020look: Visual-Audio-Room (Look, Listen, and Act)
- Paper and link: Chuang Gan, Yiwei Zhang, Jiajun Wu, Boqing Gong, Joshua B. Tenenbaum. 2020. https://arxiv.org/abs/1912.11684. ICRA 2020 per the arXiv comments field ("Accepted by ICRA 2020").
- Read depth: arXiv metadata via API; paper PDF text (environment and sound sections, grepped); project page http://avn.csail.mit.edu (raw HTML).
- What it contains: a simulated multi-room environment built on AI2-THOR with the Unity Resonance Audio SDK. Seven multi-room apartments made by concatenating AI2-THOR rooms, discretized into 0.5 m x 0.5 m grid cells (150 to 200 cells per apartment).
- Spatial audio format: "stereo sound frames" from Resonance Audio; the paper resamples the stereo sound ("re-sample the sound to 16 HZ", as printed; the unit looks like a typo) and uses STFT spectrograms.
- Visual observations: egocentric RGB from the AI2-THOR camera.
- Actions or motion: discrete moves on the grid (the exact set was not copied from the text).
- Simulated or real: simulated.
- How to get it: not available. The project page's Dataset section reads "Coming soon." (the text is visible in the raw HTML, followed by a commented-out block).
- Release status: project page http://avn.csail.mit.edu returned HTTP 200 on 2026-11-16 and shows "Dataset ... Coming soon." `verified`.
- License and prerequisites: not stated.
- Used by: the paper's own planner; later SoundSpaces work moved to Matterport3D and Replica.

### gao2023sonicverse: Sonicverse
- Paper and link: Ruohan Gao, Hao Li, Gokul Dharan, Zhuzhu Wang, Chengshu Li, Fei Xia, and others (9 authors per the arXiv API). 2023. https://arxiv.org/abs/2306.00923. ICRA 2023 per the arXiv comments field. Code: https://github.com/StanfordVL/sonicverse.
- Read depth: arXiv metadata and text (grepped); repository README (raw).
- What it contains: a simulator, built on iGibson and Resonance Audio, for continuous real-time audio rendering with a VR interface and audio-visual tasks such as AudioGoal navigation; README: "rendering frame rate of 10 Hz and an audio sampling rate of 48,000 Hz". The paper mentions a TurtleBot with a binaural microphone for the real-world counterpart.
- Spatial audio format: binaural (paper text, "binaural audio it receives").
- Visual observations: iGibson RGB (and other iGibson sensors).
- Actions or motion: robot control actions per timestep from an iGibson robot; the exact action space was not read.
- Simulated or real: simulated (with a real-robot sim-to-real demonstration mentioned in the paper; not read in depth).
- How to get it: install iGibson, build Resonance Audio into it (README build steps), download iGibson scenes through the iGibson documentation, and take the 102 sounds from the SoundSpaces `sounds.tar.xz`. No dataset of episodes is provided.
- Release status: https://github.com/StanfordVL/sonicverse: public, MIT (GitHub API), last push 2023-03-18, not archived. README's "Latest Updates" says audio simulation was being integrated into OmniGibson ("Updates will be released soon", dated 2/13/2023); I did not check whether that happened. `verified`.
- License and prerequisites: MIT for the code; iGibson scene and dataset terms apply.
- Used by: the paper's own semantic audio-visual navigation case study.

### paul2022avlen: AVLEN
- Paper and link: Sudipta Paul, Amit K. Roy-Chowdhury, Anoop Cherian. 2022. https://arxiv.org/abs/2210.07940. NeurIPS 2022 per the arXiv comments field ("Accepted at NeurIPS 2022").
- Read depth: arXiv HTML text, grepped for task, data and action text; searched for release statements (none found).
- What it contains: audio-visual-language embodied navigation: the agent navigates to a sound-emitting object, and may query an oracle for language instructions. Built on the Matterport3D simulator with SoundSpaces. It is a task and a method, not a new set of recordings.
- Spatial audio format: binaural microphone, "two channel spectrogram" (Section 3).
- Visual observations: egocentric RGBD.
- Actions or motion: A = {stop, move forward, turn right, turn left}, on SoundSpaces' 1 m grid.
- Simulated or real: simulated.
- How to get it: SoundSpaces data as above plus Matterport3D; the language instruction data was not located.
- Release status: no code or data statement found in the text read. I did not open a project page for it.
- License and prerequisites: not stated.
- Used by: its own query and navigation policies.

## Real recordings with spatial audio and camera or body motion

### liu2026omniecho: OmniEchoBench (QA and Nav tracks)
- Paper and link: Ruixun Liu, Yuxuan Wang, Jiacheng Xie, Yuhuan You, Donghua Cai, Junming Lin, Xiong-Hui Chen, Zhifang Guo, Yunfei Chu, Qize Yang, Xize Cheng, Jin Xu, Yiwu Zhong. 2026 (20 Sep, v2 read). https://arxiv.org/abs/2609.23407. No venue on the arXiv page: preprint, not peer reviewed. Code and data: https://github.com/PKU-VaLuE-Lab/OmniEcho and https://huggingface.co/datasets/PKU-VaLuE-Lab/OmniEchoBench.
- Read depth: arXiv HTML text for the benchmark description and appendix A.2.2; repository README in full; `docs/benchmark_data.md` (raw); Hugging Face API file list and metadata.
- What it contains: OmniEchoBench-QA (197 real first-person videos with FOA audio, 2,972 questions) and OmniEchoBench-Nav: "30 real-scanned indoor environments", "900 navigation samples" (30 per scene), each with one source at a known 3D position whose signal was captured at 12 to 17 receiver positions on about a 1 m grid at about 1 m height; the agent is served the recording of the receiver nearest its position, with horizontal directions rotated into its current heading. Visual observations for Nav are rendered from the scanned mesh (the data doc says "Navigation RGB observations are rendered from meshes"). So the Nav audio is a fixed field of recordings, not audio recorded by a moving agent.
- Spatial audio format: first-order Ambisonics, 4 channels, ACN/SN3D, ordered [W, Y, Z, X], 48 kHz (appendix A.2.2).
- Visual observations: first-person video (QA); rendered mesh views (Nav).
- Actions or motion: Nav is Habitat-based closed loop; the OmniEcho model outputs 8 future waypoints relative to the current pose (stated in the earlier survey from Section 4.1); success if within 1 m (README).
- Size: counts differ between sources: the README table says "12,390 unique recordings" and sources "489" sound events and "411" speech; the paper appendix says 12,900 receiver recordings and 505 and 395. The data doc's inventory (verified on October 18, 2026 by its authors) lists 12,390 unique Nav audio files and 14,872 objects with "208,431,607,935" total bytes. Hugging Face API: 14,875 files and `usedStorage` 197,525,924,519 bytes (about 198 GB); file types: 13,890 wav, 197 mp4, 60 glb, 30 navmesh, 600 png.
- Simulated or real: real recordings in real scanned rooms for the Nav audio field (paper: "real-world environments", "captured"); the video side of Nav is rendered. The QA videos are real recordings; the paper also builds a separate simulated training set by encoding mono audio into FOA with spherical harmonics (Section 3.3), which is not part of the benchmark download.
- How to get it: ready download. `huggingface.co/datasets/PKU-VaLuE-Lab/OmniEchoBench` is public and not gated (API: `gated: false`, `private: false`); it holds the media, scene meshes and navmeshes. The GitHub repository holds the annotation files, episode indices and the evaluation code, and its README says "Public media download links remain to be added" while its own table marks "Benchmark videos, FOA audio, and scene assets" as "Included"; the Hugging Face repository is the place the media were found. The Nav evaluation needs Habitat 0.2.4 and InternNav (README), so running the agent in the loop needs that environment (not tested).
- Release status: Hugging Face repository created 2026-11-06 (API `createdAt`), 1,822 downloads at the time of reading; GitHub last push 2026-11-09. `verified` through the API; no file was downloaded. Checkpoints: the README says model checkpoints are "distributed separately"; I did not find where.
- License and prerequisites: the README's License section says "released for non-commercial research use under CC BY-NC-SA 4.0". The Hugging Face metadata says `license: other` with a `LICENSE.md` (name "unspecified"); its content could not be read through the raw URL (empty response), so the README is the only license statement seen.
- Used by: OmniEcho (Qwen3-Omni-30B-A3B plus an FOA encoder) and the baselines in the paper (Seq2Seq, CMA, InternVLA-N1 for Nav; per the earlier survey file).

### grauman2023egoexo4d: Ego-Exo4D
- Paper and link: Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, and others (101 authors per the arXiv API). 2023. https://arxiv.org/abs/2311.18259 (v4). Venue: CVPR 2024, from the arXiv comment ("expanded manuscript compared to ... CVPR 2024 paper").
- Read depth: arXiv HTML text (grepped for audio and sensors); docs site https://docs.ego-exo4d-data.org/ (home, CLI downloader page).
- What it contains: time-synced first-person (Aria glasses) and third-person video of skilled activities. The docs home page says: "1,422 hours of video combined" for the first release and V2 "includes 1286.30 video hours (221.26 ego-hours) across 5035 takes", with "7-channel audio, IMU, eyegaze, two grayscale SLAM cameras, and 3D environment point clouds". A later paper (`adebi2025audio`, ECCV 2026 per the arXiv comment) says it used "all 132 hours of egocentric sequences for which there are audio recordings".
- Spatial audio format: 7-microphone array on the glasses ("An array of seven microphones recording spatial audio around the wearer", paper); Nymeria and AEA papers state 48 kHz for the same device, but the Ego-Exo4D page itself did not give a sample rate in the parts read.
- Visual observations: Aria RGB video, grayscale SLAM videos, plus exocentric cameras.
- Actions or motion: no navigation actions. Egomotion: Aria Machine Perception Services trajectories ("take_trajectory", 509.5 GB part on the downloader page), IMU, eye gaze.
- Size: CLI downloader page, part sizes in GB: `take_audio` 1056.907, `take_trajectory` 509.503, `take_vrs` 12301.458, `all` 38449.753, default set 12112.778.
- Simulated or real: real.
- How to get it: sign the Ego-Exo4D license agreement ("The License Agreement process takes 2 days"), set up the `ego4d` CLI and an AWS client, then download chosen parts, for example `take_audio` and `take_trajectory`, with filters (`--views ego`). The docs banner reads "Ego4D and Ego-Exo4D access has moved to Lab22", so the current sign-up path should be checked on the docs page before use.
- Release status: docs site live and shows V2 released (`verified`, 2026-11-16). The license PDF URL I tried (`ego4ddataset.com/ego-exo4d-license/`) returned a 404 page, so the license terms were not read.
- License and prerequisites: license agreement with approval wait; terms not read.
- Used by: the camera pose estimation paper `adebi2025audio` (arXiv 2512.12165, ECCV 2026 per its arXiv comment), which trains audio-visual relative camera pose estimation on the audio-bearing egocentric sequences; and the paper's own benchmarks.

### lv2024aria: Aria Everyday Activities (AEA)
- Paper and link: Zhaoyang Lv, Nicholas Charron, Pierre Moulon, Alexander Gamino, and others. 2024. https://arxiv.org/abs/2402.13349 (v2). No venue in the arXiv fields: preprint, not peer reviewed. Dataset page https://www.projectaria.com/datasets/aea/.
- Read depth: arXiv HTML text (abstract, Table 1, Appendix A on the microphone); Project Aria docs pages (overview and download); license page (start); Hugging Face API record.
- What it contains: "143 daily activity sequences recorded by multiple wearers in five geographically diverse indoor locations"; Table 1 total "1,091,621" image frames and "7.3" hours; sequences with one or two wearers. Includes high-frequency 6DoF trajectories (open loop and closed loop), semi-dense point clouds, eye gaze and speech transcripts.
- Spatial audio format: "There are 7 microphones in the glasses, five face the front plus one on each side. We captured spatial audio with 7 channels using a sample rate of 48kHz" (Appendix A).
- Visual observations: Aria RGB, mono SLAM cameras, eye cameras.
- Actions or motion: no actions. Egomotion from the closed-loop trajectory CSVs in a shared coordinate frame per location.
- Size: docs download page: "143 sequences ... about 353GB".
- Simulated or real: real.
- How to get it: sign up with an email on the AEA page, download the "AEA Download Links" file (links expire after 14 days), then use the `aea_dataset_downloader` command from `projectaria_tools` to fetch sequences or parts. The sample sequence is a small first download.
- Release status: docs page and project page live on 2026-11-16 (the docs site is labeled "Aria Gen 1 Docs"). `verified`. The Hugging Face record `projectaria/aria-everyday-activities` holds only a README and an image (the data is not there).
- License and prerequisites: Aria Everyday Activities Dataset License Agreement; the start of the license page says use is limited to non-commercial research in listed fields (action recognition, scene understanding, human-object interaction, human understanding, audio-visual diarization, action forecasting, scene reconstruction, camera localization and mapping) and the download links file is tied to the agreement. The rest of the agreement was not read.
- Used by: the AEA paper's own example applications (NeRF reconstruction, prompted segmentation); no spatial-audio model was found trained on it in the text read.

### ma2024nymeria: Nymeria and NymeriaPlus
- Paper and link: Lingni Ma, Yuting Ye, Fangzhou Hong, Vladimir Guzov, and others. 2024. "Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild". https://arxiv.org/abs/2406.09905 (v2). ECCV 2024 per the ECVA/ECCV page seen in the search (not opened); the arXiv fields give no venue, so treat the venue as unconfirmed here.
- Read depth: arXiv HTML text (sensor sections, grepped for audio); project page https://www.projectaria.com/datasets/nymeria/; download repository README (raw).
- What it contains: full-body motion capture ("XSens mocap suit") plus Project Aria glasses video and sensors from a participant, with a second observer device; the project page says "300 hours of daily activity, 3600 hours of video data, 1200 sequences, 264 participants, 50 indoor and outdoor locations".
- Spatial audio format: the paper's recording-profile text says the glasses record "48KHz 7-channel audio". Whether the downloadable original Nymeria files include that audio is not confirmed: the repository README lists "headset audio" among the "additional modalities" that NymeriaPlus adds, and its viewer list names video, trajectories, point clouds and motion but not audio.
- Visual observations: egocentric RGB and grayscale videos, eye tracking.
- Actions or motion: no navigation actions; ground-truth full-body motion and 6DoF device trajectories (the project page cites "400 Km traveling trajectory").
- Size: repository README: "Nymeria and NymeriaPlus datasets each provide 1100 sequences, which in total amounts to ~80 TB per dataset"; per-group downloads let one select sequences and artifacts.
- Simulated or real: real.
- How to get it: obtain a signed URL JSON file from https://explorer.projectaria.com/nymeria (or `/nymeria_plus`), then run the repository's download CLI (the README says to use the `nymeria_dataset_legacy` branch for the original Nymeria and `nymeriaplus-download` for NymeriaPlus).
- Release status: project page and repository live on 2026-11-16 (`verified`); the project page announces NymeriaPlus as the latest upgrade.
- License and prerequisites: project page: "released by Meta under the Creative Commons Attribution-NonCommercial 4.0 International License"; email submission required on the project page.
- Used by: the Nymeria paper's motion benchmarks (AvatarPoser, EgoEgo and others per its appendix); no spatial-audio model found.

### min2025supervising: StereoWalks
- Paper and link: Anna Min, Ziyang Chen, Hang Zhao, Andrew Owens. 2025. "Supervising Sound Localization by In-the-wild Egomotion". CVPR 2025, pp. 23936-23946 (CVF Open Access page https://openaccess.thecvf.com/content/CVPR2025/html/Min_Supervising_Sound_Localization_by_In-the-wild_Egomotion_CVPR_2025_paper.html).
- Read depth: CVF abstract page; paper PDF text, Section 3.1 and Table 1 (grepped for release statements).
- What it contains: in-the-wild videos with moving cameras and stereo sound. Section 3.1: internet video with camera motion and stereo sound, a larger set "YT-Stereo" and an iPhone subset "YT-Stereo-iPhone"; "obtaining 20 hours of video" after filtering; the validation and test sets are annotated with perceived angles. Two recorded subsets: Stereo-Fountain (iPhone 13 Pro stereo microphone) and Binaural-Fountain (Sennheiser AMBEO Smart Headset in-ear microphones).
- Spatial audio format: stereo and in-ear binaural; sample rate not found.
- Visual observations: handheld or head-worn video.
- Actions or motion: no actions. Camera egomotion is estimated by multi-view geometry and used as weak supervision.
- Size: 20 hours for the YouTube iPhone subset (Section 3.1); total not read.
- Simulated or real: real.
- How to get it: not found. The CVF page lists only pdf, supp and bibtex. A web search for a project page returned none; the one repository it surfaced (`IFICL/stereocrw`) belongs to an earlier paper by the same group. Because part of the data is YouTube video, redistribution terms are also unknown.
- Release status: I did not find a released dataset or code for StereoWalks. This is limited to the CVF page, the paper text and one web search.
- License and prerequisites: not found.
- Used by: the paper's own binaural localization model.

### guo2025mrsaudio: MRSAudio
- Paper and link: Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Xintong Hu, and others. 2025. https://arxiv.org/abs/2510.10396 (v3). arXiv page lists no venue; the Hugging Face card says "Our Paper is accepted by NeurIPS 2025!" (card text, not a conference page). Project page https://mrsaudio.github.io.
- Read depth: arXiv HTML text (abstract, Section 3, Appendix A.2 and A.5, grepped); project page; Hugging Face API records and card.
- What it contains: 484 hours (paper abstract-level statement: "484-hour") of recorded spatial audio in four parts. MRSLife (129 h) has "egocentric video and FOA audio" of daily activities; MRSSpeech (206 h) has binaural conversations with video and 3D source positions. The paper also lists "motion trajectories" and "3D pose tracking". Egocentric recording: Gopro video with a Zoom H3-VR first-order Ambisonics recorder, and the paper says "participants are asked to remain within the frontal field of view of the binaural recording head", so the head is mostly still: this is not a moving-agent dataset.
- Spatial audio format: binaural and first-order Ambisonics; "All audio is recorded in WAV format at a sampling rate of 48 kHz".
- Visual observations: video at "a minimum resolution of 1080p and 24 frames per second".
- Actions or motion: no navigation actions; positional geometry for sources; the walking in MRSSpeech is of the speaker, not the listener.
- Size: 484 hours per the paper (the Hugging Face card says 500 hours with MRSLife 129 h); the Hugging Face API for `verstar/MRSAudio` lists about 100,000 files (listing truncated) and `usedStorage` 284,572,634,706 bytes.
- Simulated or real: real.
- How to get it: ready download: `huggingface.co/datasets/verstar/MRSAudio` (public, not gated), and a "refreshed version" at `huggingface.co/datasets/MRSAudio/MRSAudio` (public, not gated, last modified 2026-08-29), per the first card's News section.
- Release status: both repositories exist and are public (`verified` through the API; no file downloaded).
- License and prerequisites: three statements disagree: the paper says CC BY-NC-SA 4.0, the `verstar/MRSAudio` card says CC-BY-4.0, and the `MRSAudio/MRSAudio` tags say CC-BY-SA-4.0. Resolve before use.
- Used by: the paper's own five tasks (audio spatialization, spatial text to speech, singing synthesis, music generation, sound event localization and detection).

### brunetto2023batvision: BatVision
- Paper and link: Amandine Brunetto, Sascha Hornauer, Stella X. Yu, Fabien Moutarde. 2023. https://arxiv.org/abs/2303.07257 (v3). IROS 2023, per the arXiv journal-reference field.
- Read depth: arXiv HTML text (dataset tables, grepped); project page https://amandinebtto.github.io/Batvision-Dataset/; repository README (raw).
- What it contains: a robot emitted chirps with a speaker and recorded the echoes with a binaural microphone, with synchronized RGB-D images, through corridors, offices and outdoor driveways. BV1 has 52,220 instances (UC Berkeley) and BV2 has 3,120 (Mines Paris), per the paper's data statement. It is active echolocation, not passive spatial hearing of external sources.
- Spatial audio format: binaural echoes, 44.1 kHz (README: "Binaural echoes are 0.5s long and sampled at 44,1kHz"; the paper table gives 72.5 ms and 0.45 s per instance for the two parts, so the README and paper differ on length).
- Visual observations: RGB and depth from a ZED camera (BV1) or an Intel RealSense (BV2).
- Actions or motion: no actions. Poses are not provided as odometry in the text read; the split is by recording time.
- Size: instance counts above; storage size not stated in the pages read.
- Simulated or real: real.
- How to get it: ready download from the project page's "Code/Dataset" link `https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M` (returned HTTP 200 this session) or the mirror DOI 10.57745/HYLZNL named in the repository README.
- Release status: repository https://github.com/AmandineBtto/Batvision-Dataset public with the dataset links. `verified` (links reachable; nothing downloaded).
- License and prerequisites: README: "CC-BY-SA-4.0".
- Used by: the paper's U-Net echo-to-depth baseline.

### evers2020locata: LOCATA corpus
- Paper and link: Christine Evers, Heinrich W. Loellmann, Heinrich Mellmann, Alexander Schmidt, Hendrik Barfuss, Patrick A. Naylor, Walter Kellermann. 2020. "The LOCATA Challenge: Acoustic Source Localization and Tracking", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1620-1643, doi:10.1109/TASLP.2020.2990485 (citation as printed in the Zenodo record; the paper itself was not opened). Data: https://zenodo.org/records/3630471.
- Read depth: Zenodo record (JSON API) in full; the paper was not read.
- What it contains: recordings with four arrays: a 32-channel Eigenmike, a 12-channel pseudo-spherical array on a NAO robot head, a 15-channel planar array and hearing aids on a head-torso simulator (4 channels), with OptiTrack ground truth for the positions and orientations of sources and arrays. Tasks 5 and 6 use moving microphone arrays.
- Spatial audio format: microphone arrays as listed; sample rate not read.
- Visual observations: none in the record (no video).
- Actions or motion: array motion with optical-tracking ground-truth pose; no actions.
- Size: Zenodo files `dev.zip` 6,207,354,195 bytes and `eval.zip` 13,045,105,034 bytes.
- Simulated or real: real.
- How to get it: ready download from Zenodo (open access).
- Release status: record public, published 2020-01-31 (`verified`).
- License and prerequisites: Open Data Commons Attribution (`odc-by`, Zenodo field).
- Used by: the IEEE-AASP LOCATA challenge submissions (not individually checked). Relevant here only as real moving-array audio with ground-truth pose and no video.

## Could not verify

- `minthwi2505/minav-packs` on Hugging Face (found by searching "MINav"): the repository exists and is public, but its file list holds only `.gitattributes`; I could not tie it to the MINav authors and it holds no data. Not treated as a release.
- A MINav episode download, NaVLA2 checkpoints, or any dataset page for NaVLA2: not found (the repository TODO list marks these unreleased; see the entry).
- AVW-4k and LH-AVLN: no code or data location found on their paper pages or by search. Absence of a link is not proof that none exists.
- StereoWalks: no data or code location found (see the entry).
- FAR's gated Hugging Face dataset card and the contents of its tar files: access requires login and acceptance of the Matterport3D terms, so only the API record was read. The audio and action layout of the latents is therefore unknown to me.
- OmniEchoBench `LICENSE.md` on Hugging Face: the raw URL returned an empty body, so the license terms there were not read.
- Ego-Exo4D license terms: the license URL I used returned a 404 page, and the docs say access moved to Lab22; I did not open that page.
- Nymeria's original-release audio: the paper states 7-channel audio is recorded, but I could not confirm that the original download includes it (see the entry).
- Visual-Audio-Room, AVLEN data: see their entries ("Coming soon" and "no release statement found").
- Matterport3D, HM3D and Replica terms and costs were not read in full; only the access routes named in habitat-sim's `DATASETS.md` and the SoundSpaces README were read.

## Searches run

Web searches (WebSearch tool, run 2026-11-16), exact queries:
1. real-world egocentric dataset binaural OR ambisonics OR microphone array audio with video and pose for audio-visual navigation
2. audio-visual navigation dataset SoundSpaces egocentric video binaural navigation episodes released benchmark 2026
3. Aria Everyday Activities dataset multichannel audio 7 microphones SLAM trajectory
4. robot navigation sound source localization dataset real recordings microphone array camera odometry released
5. cvpr 2025 poster 34012 binaural audio-visual ego-motion dataset real-world
6. vision-language-audio-action navigation dataset binaural audio instruction Habitat SoundSpaces 2026 arXiv
7. Nymeria dataset egocentric Aria audio body motion; Ego-Exo4D spatial audio
8. StereoWalks dataset "Supervising Sound Localization by In-the-wild Egomotion" project page code release
9. Aria Everyday Activities spatial audio 7 channel microphone projectaria docs AEA download license
10. 2026 arXiv audio-visual navigation benchmark spatial audio egocentric dataset trajectories released Hugging Face binaural Matterport HM3D (extended mode)
11. real robot audio-visual navigation dataset microphone array RGB-D odometry recorded sound source navigation sim-to-real 2025 2026

Direct lookups: arXiv API id lists for all papers above; GitHub repository search (`api.github.com/search/repositories`) for "AVW-4k", "AV-CDiT", "LH-AVLN", "PAG-Nav", "audio-visual world model binaural"; Hugging Face dataset search (`huggingface.co/api/datasets?search=`) for "MINav", "SoundSpaces", "AVW-4k", "LH-AVLN", "AudioWorldSim", "StereoWalks", "OmniEchoBench", "MRSAudio"; Hugging Face model search for "NaVLA".

Opened and set aside as not matching the brief (read only in part, listed for transparency): 360+x (https://x360dataset.github.io/, arXiv 2404.00989, CVPR 2024 Oral per its arXiv comment): real egocentric and 360 video with binaural-delay features and GPS, but no pose or action stream; SSLR (https://www.idiap.ch/dataset/sslr/): audio from a stationary Pepper robot with front-camera images for ground truth, no motion; RealMAN (arXiv 2406.19959, NeurIPS 2024 per its arXiv comment): real-recorded array data for localization, not read in depth; EgoSound and Ego4D appeared in results but were not opened.

## What exists and what I did not find

For navigation with spatial audio in simulation, SoundSpaces is the base of almost everything found. SoundSpaces 1.0 is a ready download of pre-rendered binaural impulse responses (about 867 GiB for the partial set, content length checked on the file server) on Matterport3D and Replica scenes, but the user still has to obtain Matterport3D or Replica separately and cache the visual frames with old habitat-sim versions, and the 1.0 rendering code is not open. SoundSpaces 2.0 renders on the fly for arbitrary meshes (binaural, Ambisonics and other layouts), so a MINav-like dataset is a generation job: scene data (HM3D access or Matterport3D), the archived SoundSpaces repository with its pinned habitat versions, a sound library, and render time. AudioWorldSim wraps that into a rollout generator but outputs only audio and actions, and states it cannot release its own data because of scene license limits.

I did not find a downloadable MINav dataset: its repository lists the dataset and checkpoints as not yet released, and two of the dataset-building scripts in the repository were stubs when fetched. I did not find a released AVW-4k or LH-AVLN set or generator: neither paper page names one, and my searches returned nothing. So of the simulated sets that pair egocentric frames, binaural audio and discrete actions, I found no finished download in the exact NaVLA2 or AVW-4k form. The closest finished download is the FAR corpora on Hugging Face (binaural audio at 16 kHz and 256 x 256 frames in Matterport3D navigation loops, about 14.4K episodes per corpus), which is gated behind the Matterport3D terms and is stored as pre-computed latents for one particular model, so it may not be usable as raw audio and video.

For real recordings, the one released benchmark that has navigation in it is OmniEchoBench-Nav, and its audio is a field of first-order Ambisonics recordings made at grid positions in 30 scanned rooms, with the visual frames rendered from meshes. It is public on Hugging Face without a gate (about 198 GB). It was not built from an agent moving and recording, so it fits an evaluation of "hear the target and move" more than world-model training on continuous egocentric video. Real egocentric recordings that pair a multi-microphone array with camera motion are Ego-Exo4D (7-channel audio on Aria glasses with trajectories, license agreement and approval wait required, audio part about 1 TB) and Aria Everyday Activities (7 channels at 48 kHz, 143 sequences, about 353 GB, license agreement). Nymeria also records the same device with full-body motion; I could not confirm that its original release contains the audio. None of these has a navigation action space; the egomotion is a head pose trajectory, which could stand in for an action label only through a conversion that none of the papers read does. StereoWalks (stereo and in-ear binaural video with camera egomotion) is the most direct "walking with ears" real dataset, but I did not find a release. MRSAudio is a large public real corpus with egocentric video and first-order Ambisonics, but the listener is mostly stationary. BatVision gives real binaural echoes with RGB-D from a moving robot, public, but it is active echolocation, not hearing external sources.

I did not find a real-robot dataset that gives passive spatial audio (Ambisonics, binaural or array), egocentric video, and the robot's own actions or odometry in one release. LOCATA has moving microphone arrays with optical-tracking pose but no video. The sim-to-real navigation paper (`chen2024sim2real`) deployed a policy on a real robot with a binaural microphone, but the text I read describes no released recordings. I did not search outside English-language sources or through conference proceedings beyond what the searches returned, so a dataset hidden in a venue I did not reach is possible.
