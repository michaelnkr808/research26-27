# Literature survey, topic B: spatial audio in embodied AI

Date: 2026-11-16. Direction: hear-lab (spatial audio for world action models). Scope: the four parts in the brief: (1) embodied agents that use spatial audio, (2) simulators and datasets that pair spatial audio with agent observations or actions, (3) spatial audio encoders and spatial audio language models that could be reused as an audio encoder, (4) measured comparisons of spatial versus single-channel audio for an agent's task. The four papers the user already read (`fan2026navla`, `lee2025deepasa`, `chen2026rmsaqa`, `luo2026sail`) are not re-surveyed. Topics A (non-spatial audio in robot policies) and C (world action models) are covered by other workers.

How the search was done: web searches (queries listed below) to find candidates, then every paper below was opened on its arXiv abstract page (https://arxiv.org/abs/<id>), which gave title, authors, dates, the comments field (where venues are stated) and the abstract. For the papers marked "abstract + sections", the arXiv HTML or ar5iv version was also fetched. Caveat on how pages were read: the fetch tool returns a model-written extract answering a question about the page, not the verbatim page, and for long papers it covered only the first ~100,000 characters. So "abstract + sections X" means those sections were read through such extracts, and numbers below are copied from those extracts with the table named. They have not been re-checked against the PDF. Verify any number against the paper before it goes into a writeup. Venues are taken only from the arXiv comments field or another primary page and are marked as such; anything else is "preprint, not peer reviewed". Several arXiv identifiers in this survey have 2026 prefixes (26xx), which is consistent with today's date.

Labels used: `verified` means seen on the page named in the entry this session.

## Part 1 and 2: embodied agents and simulators (foundational)

### chen2020soundspaces: SoundSpaces: Audio-Visual Navigation in 3D Environments
- Authors: Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, Kristen Grauman
- Year and venue: 2020. ECCV 2020 (Spotlight), per the arXiv comments field.
- Link: https://arxiv.org/abs/1912.11474 (v3 read)
- Read depth: abstract + sections 3 and 5 (audio data, sensory inputs), 6 (baselines, Table 2), 8.4 (Table 4), via ar5iv extract
- Part: 1, 2, 4
- What they do: Introduce audio-visual navigation (reach a sounding goal using sight and sound) and the SoundSpaces dataset of precomputed audio renderings on Matterport3D and Replica, with a Habitat extension for audio sensing.
- Spatial audio format: binaural (one channel per ear), precomputed room impulse responses at discrete grid points.
- How the audio is encoded and used: Each ear is converted to a log-magnitude spectrogram (STFT), the two stacked as a 2-channel image, encoded by a small CNN, concatenated with a vision CNN output, and fed to a GRU actor-critic trained with RL (Section 5).
- Tasks and data: AudioGoal and AudioPointGoal navigation on Replica and Matterport3D (Section 6).
- Spatial versus mono comparison: No mono versus binaural ablation (the extract states the paper reports none; it says methods could run on ambisonic inputs and that binaural was chosen so humans can interpret the videos). It does compare audio versus no audio (Section 6, Table 2: e.g. Replica Blind 0.480 without audio vs 0.681 with audio, as printed) and intensity-only versus spectrogram audio (Section 8.4, Table 4).
- Code, weights or data released: "Our audio platform and data is shared publicly" (Section 3); project page http://vision.cs.utexas.edu/projects/audio_visual_navigation.

### chen2022soundspaces2: SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning
- Authors: Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip W Robinson, Kristen Grauman
- Year and venue: 2022. The arXiv comments field says "Camera-ready version" and does not name the venue. NeurIPS 2022 Datasets and Benchmarks track was shown by a web search result (NeurIPS proceedings page), not opened here.
- Link: https://arxiv.org/abs/2206.08312 (v2 read)
- Read depth: abstract + sections 3 (platform, microphone types), 5 to 6 (validation, limits) via arXiv HTML extract
- Part: 2
- What they do: A geometry-based acoustic simulator that renders audio on the fly for arbitrary 3D meshes (Gibson, HM3D, Matterport3D), with configurable microphones and materials, plus the SoundSpaces-PanoIR dataset.
- Spatial audio format: mono, stereo, binaural (HRTF), quad, 5.1, 7.1, ambisonics, and user-defined arrays built from mono microphone positions (Section 3.3 and supplement).
- How the audio is encoded and used: It is a renderer, not a model: it outputs room impulse responses that are convolved with source audio at the agent's pose. Moving sources and listeners are handled by step-wise convolution with crossfading (Section 3.2).
- Tasks and data: navigation, mapping, source localization and separation, acoustic matching, far-field speech recognition (sim-to-real shown for the speech recognition part). SoundSpaces-PanoIR is about 10M panoramic image and impulse response pairs from 750 environments (Section 4 to 5).
- Spatial versus mono comparison: none.
- Code, weights or data released: stated publicly available on GitHub (soundspaces.org); the extract found no license statement. Matterport3D and similar scenes carry their own licenses.
- Relevance note: it renders sound for a moving agent in a scanned scene, but it has no manipulation or robot arm; the rendered audio is paired with navigation observations only.

### gan2020look: Look, Listen, and Act: Towards Audio-Visual Embodied Navigation
- Authors: Chuang Gan, Yiwei Zhang, Jiajun Wu, Boqing Gong, Joshua B. Tenenbaum
- Year and venue: 2020. ICRA 2020, per the arXiv comments field.
- Link: https://arxiv.org/abs/1912.11684
- Read depth: abstract only
- Part: 1
- What they do: Navigate indoors to a sound source using egocentric vision and audio, with a visual mapping module, a sound module that estimates the source's relative location, and a dynamic planner. Evaluated on a new simulated Visual-Audio-Room dataset.
- Spatial audio format: not stated in the abstract.
- How the audio is encoded and used: not read beyond the abstract (a sound module estimates relative source location).
- Tasks and data: Visual-Audio-Room dataset (simulated).
- Spatial versus mono comparison: not read.
- Code, weights or data released: not stated on the abstract page (project page link exists in the comments).

### chen2021semantic: Semantic Audio-Visual Navigation
- Authors: Changan Chen, Ziad Al-Halah, Kristen Grauman
- Year and venue: 2021. The arXiv page names no venue (only a project page). CVPR 2021 was shown by a search result (CVF open access page and DBLP), not opened here.
- Link: https://arxiv.org/abs/2012.11583 (v2)
- Read depth: abstract only
- Part: 1
- What they do: A navigation task where objects emit sounds tied to their meaning that are short or intermittent, and a transformer with persistent multimodal memory that reaches the goal after the sound stops. Extends SoundSpaces with semantic sounds.
- Spatial audio format: not stated in the abstract (SoundSpaces binaural per the original platform).
- How the audio is encoded and used: not read beyond the abstract.
- Tasks and data: semantic audio-visual navigation on Matterport3D via extended SoundSpaces.
- Spatial versus mono comparison: not read.
- Code, weights or data released: not stated on the abstract page.

### younes2023catch: Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds
- Authors: Abdelrahman Younes, Daniel Honerkamp, Tim Welschehold, Abhinav Valada
- Year and venue: 2021 first posted, v4 January 2023. Accepted to IEEE Robotics and Automation Letters, per the arXiv comments field.
- Link: https://arxiv.org/abs/2111.14843 (v4)
- Read depth: abstract only
- Part: 1
- What they do: A dynamic audio-visual navigation benchmark with moving, noisy and distractor sounds, and an RL policy fusing audio and visual features in a shared spatial feature space.
- Spatial audio format: not stated in the abstract.
- How the audio is encoded and used: not read beyond the abstract.
- Tasks and data: Matterport3D and Replica with SoundSpaces.
- Spatial versus mono comparison: not read.
- Code, weights or data released: not stated on the abstract page.

### gao2023sonicverse: Sonicverse: A Multisensory Simulation Platform for Embodied Household Agents that See and Hear
- Authors: Ruohan Gao, Hao Li, Gokul Dharan, Zhuzhu Wang, Chengshu Li, Fei Xia, Silvio Savarese, Li Fei-Fei, Jiajun Wu
- Year and venue: 2023. ICRA 2023, per the arXiv comments field.
- Link: https://arxiv.org/abs/2306.00923 (v2)
- Read depth: abstract + simulator, tasks, model inputs and sim-to-real sections via ar5iv extract
- Part: 2, 1
- What they do: A simulator built on iGibson 2.0 that renders continuous real-time audio (Resonance Audio SDK: distance attenuation, occlusion, early reflections, reverb) so household agents with physics and manipulation-capable simulation can also hear, with a VR interface.
- Spatial audio format: binaural via HRTFs. The extract says the paper does not describe mono or multichannel output modes.
- How the audio is encoded and used: two-channel spectrogram plus RGB-D and pose into a multi-task semantic audio-visual navigation model with an occupancy-map auxiliary task. Real robot: TurtleBot with a 3Dio FS binaural microphone.
- Tasks and data: semantic audio-visual navigation, audio-goal navigation, speaker following, voice-driven object retrieval. Sim-to-real in a real bedroom (10 trials per condition, per the extract).
- Spatial versus mono comparison: none (extract says no spatial versus mono ablation).
- Code, weights or data released: code at https://github.com/StanfordVL/Sonicverse (given on the arXiv page).

### gan2021threedworld: ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation
- Authors: Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James Traer, Julian De Freitas, Jonas Kubilius, Abhishek Bhandwaldar, Nick Haber, Megumi Sano, Kuno Kim, Elias Wang, Michael Lingelbach, Aidan Curtis, Kevin Feigelis, Daniel M. Bear, Dan Gutfreund, David Cox, Antonio Torralba, James J. DiCarlo, Joshua B. Tenenbaum, Josh H. McDermott, Daniel L.K. Yamins
- Year and venue: v1 2020, v2 December 2021. NeurIPS 2021 Datasets and Benchmarks Track (oral), per the arXiv comments field.
- Link: https://arxiv.org/abs/2007.04954 (v2)
- Read depth: abstract only. The notes on impact sounds (PyImpact) and Resonance Audio spatialization come from a web search snippet, not from a page read, so they are `recalled`-grade for this survey.
- Part: 2
- What they do: A physics-and-rendering platform with high-fidelity audio, customizable agents, VR interaction and physics for cloth, liquids and deformable objects.
- Spatial audio format: the abstract says "high-fidelity audio" only; the spatialization format was not confirmed on the abstract page.
- How the audio is encoded and used: not read.
- Tasks and data: physical scene understanding, dynamics prediction, multi-agent interaction (abstract).
- Spatial versus mono comparison: none seen.
- Code, weights or data released: project page threedworld.org (arXiv comments).
- Relevance note: one of few simulators whose abstract combines audio with physical interaction (contact), which is closer to manipulation than the navigation-only simulators. Whether it gives multi-channel spatial audio to an arm-mounted or head-mounted listener was not checked.

## Part 1 and 4: recent audio-visual navigation and manipulation with spatial cues

### li2025audio: Audio-Guided Dynamic Modality Fusion with Stereo-Aware Attention for Audio-Visual Navigation
- Authors: Jia Li, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng
- Year and venue: 2025. Accepted at ICONIP 2025, per the arXiv comments field.
- Link: https://arxiv.org/abs/2509.16924 (v1)
- Read depth: abstract + sections 3.2, 3.4, 4.1, 4.2 (Tables 2 and 3) via arXiv HTML extract
- Part: 1, 4 (indirect)
- What they do: An RL navigation agent with a Stereo-Aware Attention Module that uses the left-right channel disparity of binaural audio, and an audio-guided dynamic fusion module that rebalances audio and visual features.
- Spatial audio format: binaural (SoundSpaces).
- How the audio is encoded and used: left and right spectrograms stacked (65x65x2 log-Mel or 65x26x2 STFT), three conv layers, then bidirectional cross-attention between left and right halves of the feature map, then a GRU actor-critic with PPO (Sections 3.2, 3.4).
- Tasks and data: SoundSpaces audio-visual navigation on Replica and Matterport3D, heard and unheard sounds (Section 4.1).
- Spatial versus mono comparison: no stereo versus mono experiment (extract states none isolates it). Closest evidence is the module ablation in Table 3 (removing the stereo-aware module lowers Matterport3D "Same" success rate from 72.1 to 63.0 as printed in the extract), which tests the module, not mono input.
- Code, weights or data released: none stated on the arXiv page.

### zhou2026audio: Audio Spatially-Guided Fusion for Audio-Visual Navigation
- Authors: Xinyu Zhou, Yinfeng Yu
- Year and venue: 2026. Accepted to IJCNN 2026, per the arXiv comments field.
- Link: https://arxiv.org/abs/2604.02389 (v1)
- Read depth: abstract only
- Part: 1
- What they do: An audio spatial feature encoder with an intensity attention mechanism, and a fusion module guided by the audio spatial state, aimed at generalization to unheard sounds.
- Spatial audio format: not stated in the abstract (SoundSpaces on Replica and Matterport3D).
- How the audio is encoded and used: not read beyond the abstract.
- Tasks and data: Replica and Matterport3D audio-visual navigation.
- Spatial versus mono comparison: not read.
- Code, weights or data released: none stated.

### li2026generalizable: Generalizable Audio-Visual Navigation via Binaural Difference Attention and Action Transition Prediction
- Authors: Jia Li, Yinfeng Yu
- Year and venue: 2026. Accepted by IJCNN 2026, per the arXiv comments field.
- Link: https://arxiv.org/abs/2604.05007 (v1)
- Read depth: abstract only
- Part: 1
- What they do: A Binaural Difference Attention module that models interaural differences, plus an auxiliary action-prediction task, for generalization to unheard sounds and new scenes.
- Spatial audio format: binaural.
- How the audio is encoded and used: not read beyond the abstract.
- Tasks and data: Replica and Matterport3D navigation.
- Spatial versus mono comparison: not read (abstract does not mention a mono baseline).
- Code, weights or data released: none stated.

### kim2026soundact: SoundAct: Learning Spatial Sound Awareness for Egocentric Robot Manipulation with Stereo Audio
- Authors: Junsu Kim, Kyungdon Joo (UNIST), as listed on the project page
- Year and venue: 2026. The project page says "ICRA 2026 Workshop on Robot Manipulation". No arXiv page or OpenReview entry was found or opened, so the venue is not confirmed beyond the project page. Treat as workshop paper, not peer reviewed in the main-conference sense.
- Link: https://vision3d-lab.github.io/soundact/
- Read depth: project page, first 100,000 characters via fetch extract only. The page is very long and the experiments were not reached in a readable form.
- Part: 1
- What they do: A diffusion-policy manipulation framework on a Franka arm that fuses a wrist camera with stereo microphones to find a sound source (a ringing alarm clock) outside the camera's view.
- Spatial audio format: stereo (two microphones).
- How the audio is encoded and used: directional cues from stereo audio as magnitude spectrograms, fused with visual features through attention, with a spatial audio augmentation to handle audio distractors (project page description).
- Tasks and data: a "beyond-sight" ring-off task.
- Spatial versus mono comparison: not confirmed. I did not find it in the text read.
- Code, weights or data released: none linked on the visible page.

### hiratsuka2026s2a2: S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information
- Authors: Kaneyoshi Hiratsuka, Benjamin Yen, Ryosuke Kojima
- Year and venue: 2026 (v1 28 July, v2 29 September). Preprint, not peer reviewed (no venue on the arXiv page).
- Link: https://arxiv.org/abs/2607.26047 (v2)
- Read depth: abstract + method, tasks, tables 2 and 3 via arXiv HTML extract (first ~100,000 of ~155,000 characters; appendix and any release statement not reached)
- Part: 1, 4
- What they do: Imitation-learning tasks where a robot must use sound to find manipulation targets, and a framework that adds acoustic spatial and spectral features to existing visuomotor policies (ACT, Diffusion Policy, VQ-BeT, pi0).
- Spatial audio format: multiple microphone arrays. Simulation: four arrays on a 300 mm radius circle around the workspace (Pyroomacoustics for audio, Genesis for physics). Real robot: four 8-channel TAMAGO-03 arrays at 16 kHz around the workspace.
- How the audio is encoded and used: MUSIC direction-of-arrival scores per array projected to a 224x224 spatial map (one channel per array), plus a spectrogram of the sound separated at the strongest peak by spotforming; each goes through a ResNet-18 trained from scratch and is injected as tokens or conditioning into the policy.
- Tasks and data: simulated Localization, Identification, Localization-and-Identification, and Exploratory tasks (100 to 200 demonstrations each); real robot with 100 teleoperated demonstrations per task using ACT.
- Spatial versus mono comparison: the extract states the paper does not compare with single-channel audio. It compares against no audio (RGB-only baseline) and against raw 32-channel waveforms; Table 2 (simulation) and Table 3 (real, L&I task: baseline 5%, without spectral 18%, without spatial 3%, full 72%, as printed in the extract).
- Code, weights or data released: only a project page (https://azuma413.github.io/projects/s2a2) on the arXiv page; no release statement found in the part read.
- Relevance note: the closest paper found to "robot policy that takes microphone-array spatial audio plus vision and outputs actions." It converts the array into hand-built maps first, not a learned encoder.

## Part 2 and 1: audio-visual world models and data generators

### wang2025audio: Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics
- Authors: Jiahua Wang, Leqi Zheng, Jialong Wu, Yaoxin Mao, Shijie Cheng
- Year and venue: 2025 (v1 30 Nov 2025, current v4 4 Aug 2026). No venue on the arXiv page. Preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2512.00883 (v4)
- Read depth: abstract + method, data, navigation experiment and ablation sections via arXiv HTML extract
- Part: 1, 2, 4
- What they do: Define an audio-visual world model that predicts both video and binaural audio under action control, build the AVW-4k benchmark, and propose AV-CDiT, a diffusion transformer with modality experts; they use it to improve a pretrained navigation agent.
- Spatial audio format: binaural, 0.15 s segments at 16 kHz, rendered with SoundSpaces 2.0 on Matterport3D.
- How the audio is encoded and used: a SoundStream tokenizer (trained on AVW-4k) gives audio latents, a Stable Diffusion VAE gives visual latents, separate adaptors project both into a shared space and the tokens are concatenated; shared attention with per-modality feed-forward experts; initialized from a pretrained navigation world model, context of 4 frames.
- Tasks and data: AVW-4k is about 30 hours, 76 indoor scenes, 4,500 trajectories with four discrete actions (move forward, turn left, turn right, stop). Downstream: planning with the world model for continuous audio-visual navigation (Table 4).
- Spatial versus mono comparison: none. The extract states there is no binaural versus mono and no audio versus no audio ablation (Table 3 ablates modality experts and training stages).
- Code, weights or data released: no release statement found in the arXiv page or the extract read.
- Relevance note: this is the one paper found that jointly predicts future video and spatial audio conditioned on actions, which is the setup the hear-lab question describes. It uses a simulated, single looping-ringtone source and a navigation agent, not a robot arm.

### zerkowski2026audioworldsim: AudioWorldSim: Realistic Binaural Audio Datasets For World Models
- Authors: Luis Vitor Zerkowski, Luiz Velho
- Year and venue: 2026 (21 August). Technical report on arXiv, 7 pages; preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2608.21075
- Read depth: abstract only
- Part: 2
- What they do: An open-source platform extending SoundSpaces 2.0 with automated random-agent navigation rollouts and a fix to how continuous sound is composed, to generate binaural datasets for world-model research.
- Spatial audio format: binaural.
- How the audio is encoded and used: it is a data generator; no model.
- Tasks and data: dataset generation for audio-based world models. A web search result (not a page I read) says it records per-ear sound as the agent acts.
- Spatial versus mono comparison: none.
- Code, weights or data released: code at https://github.com/Luizerko/AudioWorldSim (given in the abstract). I did not open the repository.

### zerkowski2026binauralvae: BinauralVAE: Spatial Audio Reconstruction For World Models
- Authors: Luis Vitor Zerkowski, Luiz Velho
- Year and venue: 2026 (6 September). Technical report on arXiv, 17 pages; preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2609.06837
- Read depth: abstract only
- Part: 3, 2
- What they do: An open-source pipeline testing variational autoencoders, including complex-valued variants, to learn latent representations of binaural audio, as a step toward an audio-based world model that links navigation actions to acoustic effects.
- Spatial audio format: binaural, from AudioWorldSim.
- How the audio is encoded and used: VAE latents of binaural audio (details not read).
- Tasks and data: reconstruction of binaural audio from a simulated robot moving through an environment.
- Spatial versus mono comparison: not read.
- Code, weights or data released: code at https://github.com/Luizerko/BinauralVAE (given on the arXiv page).

## Part 3 and 4: spatial audio encoders and spatial audio language models

### zheng2024bat: BAT: Learning to Reason about Spatial Sounds with Large Language Models
- Authors: Zhisheng Zheng, Puyuan Peng, Ziyang Ma, Xie Chen, Eunsol Choi, David Harwath
- Year and venue: 2024 (v4 13 August 2026). "Accepted to ICML 2024" per the arXiv comments field.
- Link: https://arxiv.org/abs/2402.01591 (v4)
- Read depth: abstract + sections on Spatial-AST, LLM connection, training data, Tables 3 and 4 and Appendix D via arXiv HTML extract
- Part: 3, 4
- What they do: Pair a binaural audio encoder (Spatial-AST) with LLaMA-2 7B so the model can perceive and reason about the direction and distance of sounds, with the SpatialSoundQA dataset.
- Spatial audio format: binaural, 2 channels at 32 kHz.
- How the audio is encoded and used: input is 4 channels, namely left and right log-mel spectrograms plus cos and sin of the inter-channel phase difference (IPD) weighted by the mel filter bank; a conv layer, a 16x16 patch embedding and a 12-layer transformer initialized from AudioMAE with three [CLS] tokens, feeding three heads (sound class, distance in 0.5 m bins, azimuth and elevation in 1 degree bins). Its output tokens pass through a projection into LLaMA-2 7B, fine-tuned with LLaMA-adapter v2. Pretraining uses AudioSet clips convolved with SoundSpaces 2.0 impulse responses.
- Tasks and data: SpatialSoundQA (detection, direction, distance, reasoning questions), staged training on 278K, 514K and 872K samples (Table 2).
- Spatial versus mono comparison: yes. The paper has a monaural variant of the encoder (mono mel spectrogram, one input channel). Table 4: monaural BAT averages 54.53% and binaural BAT with prompt averages 76.89% (as printed in the extract). Table 3 compares mono and binaural encoder training for detection and localization. The extract's reading of Table 3 rows was inconsistent in one place, so check that table in the PDF before quoting it. This is question answering on audio clips, not an agent task.
- Code, weights or data released: the arXiv comments say demo, dataset, code and model weights are on https://zhishengzheng.com/bat. I did not open that page.

### sakshi2025spur: SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models
- Authors: S Sakshi, Vaibhavi Lokegaonkar, Neil Zhang, Ramani Duraiswami, Sreyan Ghosh, Dinesh Manocha, Lie Lu
- Year and venue: 2025 (10 Nov, v2 13 Nov). No venue on the arXiv page. Preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2511.06606 (v2)
- Read depth: abstract + encoder, results tables 1 and 2, appendix on licenses via arXiv HTML extract
- Part: 3, 4 (indirect)
- What they do: A lightweight add-on that gives existing audio-language models a first-order Ambisonics (FOA) encoder and adapter, with the SPUR-Set spatial QA dataset.
- Spatial audio format: first-order Ambisonics (W, X, Y, Z).
- How the audio is encoded and used: STFT, banded covariance matrix per mel band with learnable temporal smoothing, flattened to a real "Spectro-Spatial Covariance Vector", then 3D conv blocks, 16x16 patch tokens and a transformer encoder; a 2-layer MLP maps into the existing audio encoder's input space. The base audio encoder and LLM stay frozen; the SPUR encoder, MLP and rank-8 LoRA in the LLM are trained. Plugged into Audio Flamingo 3 and Qwen2.5-Omni.
- Tasks and data: spatial QA (relative direction, elevation, distance, overlap) from open FOA recordings plus controlled simulations; STARSS23 audio-visual QA in Appendix C.
- Spatial versus mono comparison: indirect only. The extract says there is no dedicated mono versus spatial ablation table in the text read; base models in their default single-channel path versus the same models with SPUR are compared (Table 1 average score for Audio Flamingo 3: 4.80 to 7.27, as printed). The extract notes the Qwen2.5-Omni base row is missing from Tables 1 and 2.
- Code, weights or data released: no explicit statement found; a project page https://sakshi113.github.io/spur/ is linked.

### jiang2025sciphi: Sci-Phi: A Large Language Model Spatial Audio Descriptor
- Authors: Xilin Jiang, Hannes Gamper, Sebastian Braun
- Year and venue: 2025 (7 October). The arXiv page lists no venue. A web search result lists IEEE Open Journal of Signal Processing, vol. 7, January 2026 (DOI 10.1109/ojsp.2026.3657300); I did not open that publisher page, so the venue is `unverified` here. Treat the arXiv version as a preprint.
- Link: https://arxiv.org/abs/2510.05542 (v1)
- Read depth: abstract only. The encoder description (four-channel mel spectrograms plus intensity vectors, frozen Phi-4 Multimodal audio encoder, Phi-4-Mini LLM) comes from a search snippet, not a page I read.
- Part: 3
- What they do: A spatial audio LLM with separate spatial and spectral encoders that lists up to four directional sources, background sounds and room properties in one pass.
- Spatial audio format: first-order Ambisonics.
- How the audio is encoded and used: see read-depth note. The abstract itself states "single-channel input fundamentally limits spatial understanding".
- Tasks and data: trained on over 4,000 hours of synthetic FOA recordings with metadata; generalizes to real room impulse responses with only minor loss (abstract).
- Spatial versus mono comparison: not read.
- Code, weights or data released: no code link on the arXiv page; a demo page https://sci-phi-audio.github.io/demo is linked.

### you2026world: The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models
- Authors: Yuhuan You, Lai Wei, Tianshu Qu
- Year and venue: 2026 (v1 6 Jan 2026, v4 7 Nov 2026). The arXiv comments say "NeurIPS 2026 camera-ready version."
- Link: https://arxiv.org/abs/2601.02954 (v4)
- Read depth: abstract + sections on the FOA encoder and Section 5.3 (Tables 2 and 3) via arXiv HTML extract (first ~100,000 of ~153,000 characters)
- Part: 3, 4
- What they do: Define spatial audio-language understanding as a three-level audio scene analysis problem and build TWNM, an FOA-based model trained with physically simulated supervision, a curriculum and preference optimization, with a controlled benchmark.
- Spatial audio format: first-order Ambisonics, 4-channel B-format at 44.1 kHz.
- How the audio is encoded and used: complex STFT features unfolded to an 8-channel spectro-temporal map; an 8-layer spatial encoder with an LSTM attractor giving up to 3 source slots, projected to 768 dimensions and aligned to a frozen Whisper-small semantic branch run on the W channel; decoder is an Audio-Flamingo-3-series model on Qwen2.5-7B with LoRA.
- Tasks and data: multiple-choice QA on localization, attribute binding, spatial comparison, scene abduction, counterfactual reasoning; STARSS23 as a limited real check.
- Spatial versus mono comparison: yes, indirect and labelled by the paper as diagnostic. Section 5.3, Table 2 lists mono systems (e.g. Qwen2.5-Omni-7B, Gemini-3.1-Pro Preview), a binaural system (BAT-SFT) and the FOA model; Table 3 splits spatial and semantic question families. Section 5.3 text: a trained model using only the W channel scores 49.3% and the FOA model 62.0% in a matched control. The paper says the external rows are not strict same-input baselines. Question answering on audio, not an agent task.
- Code, weights or data released: the extract says the paper does not commit to releasing code, weights or data; the supplement has a compact architecture prototype and benchmark metadata.

### liu2026omniecho: OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents
- Authors: Ruixun Liu, Yuxuan Wang, Jiacheng Xie, Yuhuan You, Donghua Cai, Junming Lin, Xiong-Hui Chen, Zhifang Guo, Yunfei Chu, Qize Yang, Xize Cheng, Jin Xu, Yiwu Zhong
- Year and venue: 2026 (v1 20 Sep, v2 23 Sep). No venue on the arXiv page. Preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2609.23407 (v2)
- Read depth: abstract + sections 4.1, 4.2, 5.2, 5.3 (Tables 1 and 4, Figure 6) via arXiv HTML extract (first ~100,000 of ~142,000 characters)
- Part: 1, 3, 4
- What they do: A benchmark (OmniEchoBench: spatial audio-visual QA plus audio-vision-language navigation, FOA recorded in real environments) and a model that adds an FOA spatial encoder to Qwen3-Omni-30B-A3B for perception and navigation.
- Spatial audio format: first-order Ambisonics.
- How the audio is encoded and used: encoder input is a 5-channel map: log-mel of the W channel, three active-intensity components and diffuseness. Trained in three stages (contrastive alignment with a CLIP text encoder on a synthetic FOA corpus, query-conditioned localization of azimuth, elevation and distance, then integration into the LLM). Spatial tokens are projected and inserted after the semantic audio tokens, which come from the frozen native audio tower fed the W channel as mono. Navigation: the model gets the instruction, camera frames and an FOA window and outputs 8 future waypoints relative to the current pose.
- Tasks and data: spatial QA and navigation, training on VLN datasets (ScaleVLN, RxR, R2R, VLN-PE) plus synthetic spatial data, evaluation on OmniEchoBench-QA (2,972 QA pairs, 197 real scenes) and -Nav (900 samples, 30 environments).
- Spatial versus mono comparison: yes. Table 1 (Section 5.2.1): the base model with single-channel audio versus OmniEcho with FOA, e.g. audio-plus-vision 18.5 versus 28.5 (as printed in the extract). Table 4 (Section 5.3): native audio only 19.8, with FOA encoder frozen 28.5. Figure 6 compares input feature representations for localization error. No no-audio ablation.
- Code, weights or data released: the abstract page says code and data "will be available" at https://github.com/PKU-VaLuE-Lab/OmniEcho (not yet confirmed as released).

### liu2026jaeger: JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments
- Authors: Zhan Liu, Changli Tang, Yuxin Wang, Zhiyuan Zhu, Youjun Chen, Yiwen Shao, Tianzi Wang, Lei Ke, Zengrui Jin, Chao Zhang
- Year and venue: 2026 (v1 20 Feb, v3 28 May). "Accepted to ICML 2026" per the arXiv comments field.
- Link: https://arxiv.org/abs/2602.18527 (v3)
- Read depth: abstract + sections 3.2, 4.1 to 4.4, 5.3 (Tables 2 and 5) and Appendices B and C (Tables 7 and 8) via arXiv HTML extract
- Part: 3, 4
- What they do: Extend audio-visual LLMs to 3D by combining RGB-D with 4-channel FOA through a learned "Neural Intensity Vector" and build the SpatialSceneQA benchmark (61K instruction samples from simulated environments).
- Spatial audio format: first-order Ambisonics, 4-channel.
- How the audio is encoded and used: dual path. The W channel carries semantic content; spatial cues come from a classical intensity vector (cross-spectra of W with X, Y, Z) or a learnable Neural IV (7-layer 1D CNN front end at 50 Hz, element-wise product of W and X/Y/Z latents, MLP). Initialized from Qwen2.5-Omni with LoRA.
- Tasks and data: single and overlapping source direction of arrival, 3D visual grounding of loudspeakers, multi-speaker matching (Sections 3.2, Table 1).
- Spatial versus mono comparison: yes, several. Table 7 (Appendix B): a "SimpleFuse" variant that flattens the four FOA channels into one waveform treated as mono scores 67.79% (one speaker) and 46.59% (two speakers) reasoning accuracy versus 99.5% and 99.2% for Neural IV, as printed in the extract. Table 5: removing the FOA encoder drops results to near the random baseline in Table 2. Table 8 (Appendix C): binaural (BAT-based) 85.0% versus FOA with Neural IV 99.2% on two-speaker reasoning. Simulated environments only.
- Code, weights or data released: the abstract states source code, checkpoints and datasets are available at https://github.com/liuzhan22/JAEGER (a search result said they would be released upon acceptance; I did not open the repository).

### oh2026spatiotemporal: Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources
- Authors: Oh Hyun-Bin, Kazuki Shimada, Yuhta Takida, Kim Sung-Bin, Toshimitsu Uesaka, Takashi Shibuya, Kyeongyoon Lee, Tae-Hyun Oh, Yuki Mitsufuji
- Year and venue: 2026 (v1 12 June, v2 27 October). "Accepted to EMNLP 2026 Main Conference" per the arXiv page.
- Link: https://arxiv.org/abs/2606.14141 (v2)
- Read depth: abstract only
- Part: 3
- What they do: ST-AudioQA (a QA benchmark from first-order Ambisonics renderings of static and moving sources), ST-Audio Encoder (event semantics plus time-resolved trajectories) and ST-AudioLM feeding these tokens to an LLM. Zero-shot transfer to real recordings is reported in the abstract.
- Spatial audio format: first-order Ambisonics.
- How the audio is encoded and used: not read beyond the abstract.
- Tasks and data: spatio-temporal audio QA (what sounds, where, how it moves).
- Spatial versus mono comparison: abstract says better than "static spatial and localization-only baselines"; no mono comparison seen.
- Code, weights or data released: none linked; project page https://st-audiolm.github.io/.

### sudarsanam2025towards: Towards Spatial Audio Understanding via Question Answering
- Authors: Parthasaarathy Sudarsanam, Archontis Politis
- Year and venue: 2025 (12 July). No venue on the arXiv page. Preprint, not peer reviewed.
- Link: https://arxiv.org/abs/2507.09195 (v1)
- Read depth: abstract only
- Part: 3
- What they do: A question answering framework for FOA audio on STARSS23 with rule-based spatio-temporal descriptions diversified by an LLM; a baseline takes FOA audio plus a question and classifies the answer.
- Spatial audio format: first-order Ambisonics.
- How the audio is encoded and used: not read beyond the abstract (the model is a classifier over answers, not an LLM).
- Tasks and data: QA on STARSS23 (presence, localization, spatial and temporal relations).
- Spatial versus mono comparison: none seen.
- Code, weights or data released: none stated.

## Part 1: sound source localization for robots (perception side)

### lee2025binaural: Binaural Sound Event Localization and Detection Neural Network based on HRTF Localization Cues for Humanoid Robots
- Authors: Gyeong-Tae Lee
- Year and venue: 2025 (6 August on arXiv). Listed on arXiv as a KAIST Ph.D. dissertation, 2024. Not peer reviewed as a paper.
- Link: https://arxiv.org/abs/2508.04333 (v1)
- Read depth: abstract only
- Part: 1, 3
- What they do: BiSELDnet, a two-ear sound event localization and detection network that uses HRTF cues to reduce elevation and front-back confusion on a humanoid robot's two-channel input, outputting direction vectors per sound class.
- Spatial audio format: binaural.
- How the audio is encoded and used: an eight-channel binaural feature set (mel spectrograms, V-maps, interaural time and level difference maps, spectral cue maps) (abstract).
- Tasks and data: SELD under urban background noise; not an agent task.
- Spatial versus mono comparison: not read.
- Code, weights or data released: none stated on the arXiv page.

### jalayer2025review: A Review on Sound Source Localization in Robotics: Focusing on Deep Learning Methods
- Authors: Reza Jalayer, Masoud Jalayer, Amirali Baniasadi
- Year and venue: 2025 (1 July). Journal reference on the arXiv page: Applied Sciences 2025, 15, 9354.
- Link: https://arxiv.org/abs/2507.01143 (v1)
- Read depth: abstract only
- Part: 1
- What they do: A review of sound source localization for robots, from classical methods (time difference of arrival, beamforming, steered response power, subspace) to CNN, CRNN and attention models, organized by robot type and application.
- Spatial audio format: not applicable (survey).
- How the audio is encoded and used: not applicable.
- Tasks and data: survey of training data and strategies and open challenges (noise, reverberation, multiple sources, compute).
- Spatial versus mono comparison: not applicable.
- Code, weights or data released: not applicable.
- Use: a starting point for the robot-side localization literature, which this survey sampled only lightly.

## Could not verify
- No paper that I tried to open failed outright. Items that could not be confirmed on a primary page, and are therefore not claimed:
  - SoundAct (`kim2026soundact`): the project page was readable but no arXiv or OpenReview page was found, and the experiments were not reached, so its venue and any mono comparison are unconfirmed.
  - `chen2021semantic` CVPR 2021 and `chen2022soundspaces2` NeurIPS 2022 Datasets and Benchmarks venues: shown only by search results, not by the arXiv pages I opened.
  - `jiang2025sciphi` IEEE OJSP venue: shown only by a search result.
  - Aria egocentric datasets (Aria Everyday Activities 2402.13349 and Aria Gen 2 Pilot 2510.16134, both opened): their abstract pages do not mention the glasses' multi-microphone audio. A search result said so but I did not confirm it from the papers, so they are not listed as sources of spatial audio here.
  - A search result mentioned a community "Kitchen Stereo4D" MuJoCo dataset with stereo video and binaural audio (10 episodes per the listing). I did not open it and do not list it.
- One whole direction was not found at all: see the last section.

## Searches run
1. `SoundSpaces audio-visual navigation 3D environments binaural arXiv`
2. `SoundSpaces 2.0 simulation platform visual-acoustic learning arXiv`
3. `BAT: Learning to Reason about Spatial Sounds with Large Language Models arXiv`
4. `spatial audio robot manipulation microphone array policy sound source localization 2025 arXiv`
5. `SoundAct stereo audio egocentric robot manipulation Kim Joo OpenReview`
6. `audio-visual navigation spatial audio embodied agent 2025 2026 arXiv binaural`
7. `spatial audio large language model first-order Ambisonics reasoning arXiv 2025 2026`
8. `simulator spatial audio robot manipulation microphone array dataset audio video actions`
9. `Sonicverse multisensory simulation platform embodied household agents audio`
10. `Semantic Audio-Visual Navigation Chen Al-Halah Grauman CVPR 2021 arXiv`
11. `sound source localization quadruped OR humanoid robot microphone array learned policy audio-visual 2026 arXiv`
12. `audio-visual navigation ablation mono versus binaural audio spatial cues agent success rate`
13. `JAEGER audio-visual large language model RGB-D 4-channel FOA Qwen2.5-Omni arXiv`
14. `egocentric dataset multichannel microphone array Aria glasses spatial audio video dataset arXiv`
15. `Sci-Phi spatial audio large language model dual encoders first-order ambisonics IEEE Open Journal of Signal Processing`
16. `ThreeDWorld multimodal physical simulation platform audio impact sounds arXiv Gan`
17. `robot learning spatial audio imitation learning binaural OR ambisonic sound localization manipulation navigation real robot dataset 2026` (extended mode)
18. `AudioWorldSim realistic binaural audio dataset world models`

## What exists and what I did not find
There is a solid body of work on agents that hear in simulation, almost all of it navigation: SoundSpaces and its follow-ups use binaural audio from a simulator, and a stream of 2025 to 2026 papers refine how the two ears are fused with vision. Spatial audio inside a manipulation policy is thin. I found two: S2A2 (four microphone arrays, hand-built direction maps, ACT, Diffusion Policy, VQ-BeT and pi0 policies) and SoundAct (stereo microphones on a wrist-camera arm, a workshop paper I could only read from its project page). Neither releases code that I could confirm.

On the world-model side, the only paper I found that predicts future video and spatial audio together conditioned on actions is Audio-Visual World Models (binaural, simulated navigation, no release statement found). AudioWorldSim and BinauralVAE are small technical reports toward audio-only world models. I did not find a spatial-audio model that is built on a large video-generation world action model, and I did not find a real robot dataset that pairs recorded binaural or Ambisonics audio with video and actions (the search was limited, so this could exist).

For encoders, there are several FOA and binaural spatial audio language models (BAT, SPUR, Sci-Phi, TWNM, JAEGER, OmniEcho, ST-AudioLM). They differ in input features: left and right mel plus phase difference (BAT), covariance vectors (SPUR), mel plus active intensity (OmniEcho), classical or learned intensity vectors (JAEGER). Several keep the omnidirectional channel on the ordinary audio path and add a spatial path next to it. Weights are stated released for BAT (project page) and JAEGER (abstract states a GitHub repository), not confirmed for the rest, and OmniEcho says "will be available."

Measured spatial versus single-channel comparisons exist only for question answering or perception, not for an agent acting: BAT (monaural versus binaural Bat), JAEGER (FOA collapsed to mono versus Neural IV), OmniEcho (single-channel base model versus FOA model) and TWNM (diagnostic only). For agents, SoundSpaces compares audio against no audio, and S2A2 compares with no audio and with raw multichannel waveforms, but I did not find a controlled mono versus binaural comparison for an agent's task. Most of the recent navigation papers I only read at abstract level, so one of them may contain such an ablation that I did not see.
