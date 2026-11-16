# Survey topic A: audio as an input to robot policies and vision-language-action models

Date: 2026-11-16. Written by a worker agent for the hear-lab direction. Topic A of 3 (topic B is spatial audio in embodied AI, topic C is world action models).

How the search was done: web searches (queries listed at the end) to find candidates, then each paper's arXiv abstract page (or the stated primary page) was opened to confirm title, authors, year, venue and identifier, then the HTML full text was fetched for method and experiment details. Page text was returned through a fetch tool that summarizes the page, so quoted numbers come from that returned text and the table or section named; a reader who needs a number for a writeup should re-check it in the paper itself. Search snippets were never used as evidence. Nothing here is from memory. All arXiv papers are marked "preprint, not peer reviewed" unless a venue was confirmed on the arXiv page or a primary page. A venue stated only in the arXiv comments or journal-ref field is recorded as "per arXiv page".

Label for every claim below: `verified` (read on the page named) unless the line says otherwise.

Short map of what the papers cover (ordered chronologically in the sections):
- Contact microphone on gripper or object, vision plus audio, behavior cloning: Li 2022, Du 2022, Mejia 2024, Liu 2024, Wei 2025.
- Speech or sound as an instruction to a VLA: Zhao 2025, Wang 2025 (RoboOmni).
- Airborne microphone or sound-conditioned policy: Wang 2025 (Sound of Simulation), Li 2026, Nie 2026 (HEAR), Zhang 2025 (audio world model).
- Spatial audio (stereo or microphone arrays): Kim 2026 (SoundAct), Hiratsuka 2026 (S2A2).

### li2022see: See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation
- Authors: Hao Li, Yizhi Zhang, Junzhe Zhu, Shaoxiong Wang, Michelle A. Lee, Huazhe Xu, Edward Adelson, Li Fei-Fei, Ruohan Gao, Jiajun Wu.
- Year and venue: 2022, CoRL 2022 (per arXiv comments; the PMLR proceedings page was not opened). arXiv 2212.03858.
- Link: https://arxiv.org/abs/2212.03858
- Read depth: abstract page + HTML full text, Secs. 3.2, 3.3, 4.1 to 4.3, Tables 1 to 3.
- What they do: a robot that fuses a camera, a contact microphone and a vision-based tactile sensor with a self-attention model, trained by imitation learning, on two tasks.
- Audio input and encoder: piezo contact microphone attached to the object of interest (HOYUJI TD-11, Sec. 4.1), 44.1 kHz resampled to 16 kHz, mel spectrogram (64 x 50 per 0.5 s). ResNet-18 features for all modalities (Sec. 3.2). The paper does not state whether the encoder is pretrained or frozen.
- Fusion: multi-head self-attention over the three modalities across time, then an MLP over a discrete action set (Sec. 3.2, "Multisensory Self-Attention Model").
- Tasks and data: dense packing (peg into a nearly full box) and pouring beads to a target weight (Sec. 3.3). Real Franka Panda. Demonstrations from a human expert choosing discrete actions by keyboard, about 10 per condition (Secs. 3.1, 4.2, 4.3).
- Benefit of audio (ablation): Table 1 (dense packing, 3D-printed objects), average success: vision only 0.25, vision + audio 0.60, all three 1.00. Table 3 (pouring, mean weight error in g): vision only 40.87, vision + audio 15.20. Text says audio helps tell hard from soft slanted bumps and helps decide when to stop pouring (Sec. 4.3).
- Code or data released: no statement found in the text read; a project page exists (https://ai.stanford.edu/~rhgao/see_hear_feel, seen only in a search result, not opened).

### du2022play: Play it by Ear: Learning Skills amidst Occlusion through Audio-Visual Imitation Learning
- Authors: Maximilian Du, Olivia Y. Lee, Suraj Nair, Chelsea Finn.
- Year and venue: 2022, RSS 2022 (per arXiv journal-ref field). arXiv 2205.14850.
- Link: https://arxiv.org/abs/2205.14850
- Read depth: abstract page + ar5iv HTML, Sec. III-B/III-C, Appendix A, Sec. IV-B and IV-E.
- What they do: adds a gripper-mounted microphone to a vision-based imitation policy so the robot can act when the camera cannot see the object (keys in a paper bag), with online fine-tuning from human interventions.
- Audio input and encoder: one microphone attached to the front of the Franka gripper, in contact with it (model not named); 2 s rolling buffer at 48 kHz, FFT spectrogram, small 1D CNN. The paper does not say pretrained or frozen; it is described as end-to-end (so likely trained from scratch, not stated outright).
- Fusion: vision (untrained ResNet-18), audio and proprioception embeddings concatenated into an MLP (early fusion), then a one-layer LSTM over 10 steps (late fusion) (Sec. III-B, III-C).
- Tasks and data: real Franka, keychain extraction from a bag (Easy and Hard) and a "mystery bag" sound-differentiation task; 50 teleoperated demonstrations (Oculus Quest 2) plus 40 online episodes with interventions. Simulated occluded grasp and pick-place tasks use average contact force as a stand-in for sound (Appendix A-B).
- Benefit of audio (ablation): Sec. IV-B, Fig. 5: full system about 70% (Easy) and about 50% (Hard); without audio about 10% (Easy) and consistently failed on Hard. Sim (Sec. IV-E, Fig. 7): removing audio lowers success by about 5%. The paper is inconsistent on how it states the comparison (abstract "50% higher", introduction "more than double", and Sec. IV-B text says a pure vision-based approach only gets 20% on the same task). Numbers are read from figures, so treat as approximate.
- Code or data released: no code statement found; project page for videos (https://sites.google.com/view/playitbyear).

### mejia2024hearing: Hearing Touch: Audio-Visual Pretraining for Contact-Rich Manipulation
- Authors: Jared Mejia, Victoria Dean, Tess Hellebrekers, Abhinav Gupta.
- Year and venue: 2024, ICRA 2024 (per arXiv page). arXiv 2405.08576.
- Link: https://arxiv.org/abs/2405.08576
- Read depth: abstract page + HTML full text, Secs. III-A to III-C, IV-A to IV-E, Table I.
- What they do: treats contact microphones as a tactile sensor and initializes the audio encoder from audio-visual pretraining so the policy needs fewer demonstrations.
- Audio input and encoder: four piezo contact microphones on the Franka gripper, 32 kHz, averaged and downsampled to 16 kHz, 2 s mel spectrogram clips. Encoder is AVID-pretrained on AudioSet and fine-tuned during policy learning, not frozen (Sec. III-B). Freezing it in the zero-shot ablation (Sec. IV-E1) caused only a slight drop. Visual encoder is R3M (ResNet-18).
- Fusion: learned positional embeddings, then a transformer block, then a 2-layer MLP predicting action chunks (Sec. III-C, Sec. IV-E4).
- Tasks and data: real Franka, flipping, scooping and zipping, with 40, 60 and 50 demonstrations (Sec. IV-B), teleoperated with an Oculus Quest headset.
- Benefit of audio (ablation): Table I, Sec. IV-D, success on flipping / scooping / zipping: proposed 50.0% / 78.1% / 88.9% versus vision-only 0.0% / 28.1% / 44.4%. Sec. IV-E3: vision-only drops about 60% from train to test on flipping versus about 20% for the proposed method.
- Code or data released: no statement in the text read; project page https://sites.google.com/view/hearing-touch.

### liu2024maniwav: ManiWAV: Learning Robot Manipulation from In-the-Wild Audio-Visual Data
- Authors: Zeyi Liu, Cheng Chi, Eric Cousineau, Naveen Kuppuswamy, Benjamin Burchfiel, Shuran Song.
- Year and venue: 2024, CoRL 2024 (per arXiv page; v2 dated 2024-11-04). arXiv 2406.19464.
- Link: https://arxiv.org/abs/2406.19464
- Read depth: abstract page + HTML v2 full text, Secs. 2, 3.1, 3.2, 4.1 to 4.4, Tables 1 to 4, Appendix A.2.
- What they do: a handheld gripper (UMI-style) with a contact microphone in one finger collects human demonstrations with synchronized audio and video, and a diffusion policy is trained on them, with audio augmentation for the human-to-robot gap.
- Audio input and encoder: piezo contact microphone embedded in a gripper finger under grip tape, 48 kHz (Sec. 3.1). Audio Spectrogram Transformer (AST) trained from scratch, end-to-end, not frozen (Sec. 3.2, 4.1, Appendix A.2); a directional "Env Mic" baseline is also tested (Sec. 4.4). Vision is CLIP ViT-B/16.
- Fusion: a one-layer, 8-head transformer encoder over vision and audio features, end-effector pose concatenated, diffusion policy (UNet) head (Sec. 3.2).
- Tasks and data: real UR5, four tasks (bagel flipping, whiteboard wiping, dice pouring, wire strapping with velcro). Human demonstrations via the handheld device: for example 283 in-lab and 274 in-the-wild for flipping, 119 wiping, 145 pouring, 193 taping (Sec. 4).
- Benefit of audio (ablation): wiping 85% versus vision-only 40% (Table 3, Sec. 4.2); flipping 90% versus 25% (Table 4, Sec. 4.1); pouring substeps Pour 100% versus 0% and Place 90% versus 8.3% (Table 1, Appendix B.2.1); taping Sense substep 90% versus 30% (Table 2, Sec. 4.4).
- Code or data released: no statement in the text read; project site maniwav.github.io (rollout videos).

### zhao2025vlas: VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation
- Authors: Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, Donglin Wang.
- Year and venue: 2025, ICLR 2025 (per arXiv page). arXiv 2502.13508.
- Link: https://arxiv.org/abs/2502.13508
- Read depth: abstract page + HTML v2, model, training and experiment sections, Tables 1 and 2, Appendix A. Note this paper is about speech as an instruction, not environmental sound.
- What they do: a VLA that takes spoken instructions directly through a speech encoder, with no separate speech recognizer, plus voice retrieval for user-specific tasks.
- Audio input and encoder: TTS-generated speech for training (ESPnet VITS), evaluation with novel TTS voices and real speech from 10 people. Whisper encoder, frozen (Stage I trains only the MLP projector; Stages II and III update everything except the image and speech encoders).
- Fusion: speech tokens and vision tokens each go through an MLP projector, concatenated with retrieved text tokens, fed to a Vicuna backbone (the "7B" size is read from the Table 4 row for VLAS-Base, not from the method text) that outputs discretized action tokens.
- Tasks and data: CALVIN (ABC/D and ABCD/D splits) and a customization benchmark, both simulation; a real UR5 demonstration is qualitative only. Data: SQA (185K speech question-answer samples from LLaVA data) and CSI (about 194K speech samples from CALVIN instructions).
- Benefit of audio (ablation): this is not an audio-versus-no-audio test but speech versus text instruction. Table 1, CALVIN ABCD/D average length: VLA with text 3.80, VLAS with speech 3.70, VLAS with real speech 3.61, VLA with speech through ASR 3.13. Table 2, customization average success: VLA with text 19.2%, VLAS with speech 86.5%, VLAS with speech and no retrieval 16.0%.
- Code or data released: "The model, data and code will be publicly available at https://github.com/whichwhichgone/VLAS" (stated as future; release not checked).

### wang2025sound: The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio
- Authors: Renhao Wang, Haoran Geng, Tingle Li, Feishi Wang, Gopala Anumanchipalli, Trevor Darrell, Boyi Li, Pieter Abbeel, Jitendra Malik, Alexei A. Efros.
- Year and venue: 2025, CoRL 2025 (per arXiv page). arXiv 2507.02864.
- Link: https://arxiv.org/abs/2507.02864
- Read depth: abstract page + HTML full text, method, Sec. 4.3, Table 1.
- What they do: MultiGen generates audio from simulated video with a fine-tuned video-to-audio model so a diffusion policy can train on audio-visual simulated trajectories and transfer zero-shot to real pouring.
- Audio input and encoder: omnidirectional USB lapel microphone at the end effector (24-bit, 192 kHz, downsampled to 16 kHz), log-mel spectrogram into an Audio Spectrogram Transformer. The paper does not state pretrained or frozen (policy described as trained from scratch). Simulated audio from MMAudio fine-tuned on pouring sounds (EPIC-Kitchens plus 1031 crawled clips).
- Fusion: diffusion policy conditioned on state, vision (CLIP ViT-B/16) and audio; the paper does not detail how they are combined.
- Tasks and data: pouring liquids into containers, trained in RoboVerse simulation with generated audio, tested zero-shot on a real Kinova Gen3.
- Benefit of audio (ablation): Table 1, Sec. 4.3, normalized mean absolute error (lower is better), vision-only versus vision + audio across eight liquid and container settings, for example 0.54 versus 0.44 and 0.54 versus 0.33; the paper reports an average 23.3% NMAE reduction.
- Code or data released: "Code, models and data available at: https://multigen-audio.github.io" (abstract; release not checked).

### wang2025roboomni: RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- Authors: Siyin Wang, Jinlan Fu, Feihong Liu, Xinzhe He, Huangxuan Wu, Junhao Shi, Kexin Huang, Zhaoye Fei, Jingjing Gong, Zuxuan Wu, Yu-Gang Jiang, See-Kiong Ng, Tat-Seng Chua, Xipeng Qiu.
- Year and venue: ICLR 2026 (confirmed on the ICLR proceedings page); arXiv v1 2025-10-27, v3 2025-11-01. arXiv 2510.23763.
- Link: https://arxiv.org/abs/2510.23763 ; ICLR page https://proceedings.iclr.cc/paper_files/paper/2026/hash/33610fba262d7b6fed0810b89f55e147-Abstract-Conference.html
- Read depth: ICLR page (metadata) + arXiv abstract page + HTML v3 first 100000 characters (method, data, main results); the last part (about 20000 characters) was not read.
- What they do: an end-to-end omni-modal LLM policy that infers a user's intent from speech, environmental sound and vision, confirms it, and acts.
- Audio input and encoder: speech and environmental sound (2,482 non-verbal sound events, 640 background soundscapes), 16 kHz. Audio pipeline follows Qwen2.5-Omni; frozen or trained is not stated. Mono or not is not stated. Simulation speech is TTS; one split uses real speech from 10 volunteers.
- Fusion: visual, audio and text embeddings go to one LLM backbone; actions are discrete FAST+ tokens (2,048, chunk size 6), decoded into 7-DoF deltas.
- Tasks and data: OmniAction dataset built from 74,645 Open X-Embodiment trajectories into 141,162 multimodal episodes (LLM-written dialogues, synthesized audio). Benchmarks OmniAction-LIBERO-TTS and OmniAction-LIBERO-Real (both simulation), plus qualitative real-robot cases on a WidowX 250S (no success rates).
- Benefit of audio (ablation): the paper has no matched ablation that removes audio from RoboOmni itself. The closest audio-removed comparison is text-only baselines given ASR transcripts or ground-truth text: Table 1 (OmniAction-LIBERO-TTS, Sec. 5.2) reports RoboOmni at 85.6% overall success versus 25.9% for the strongest baseline (NORA); the text says text-only models, "whether using ASR transcripts or ground-truth text", fail to capture paralinguistic cues, with a best score of 25.9% (textual baseline). These are different models, not one model with audio removed. Intent recognition: 88.9% versus 55.6% for ASR + GPT-4o (text after Fig. 6a).
- Code or data released: abstract lists a project page, GitHub repository and a Hugging Face collection (fnlp/roboomni); which artifacts are released was not checked.

### wei2025audio: Audio-VLA: Adding Contact Audio Perception to Vision-Language-Action Model for Robotic Manipulation
- Authors: Xiangyi Wei, Haotian Zhang, Xinyi Cao, Siyu Xie, Weifeng Ge, Yang Li, Changbo Wang.
- Year and venue: 2025, preprint, not peer reviewed (no venue on the arXiv page). arXiv 2511.09958.
- Link: https://arxiv.org/abs/2511.09958
- Read depth: abstract page + HTML v1 full text, Secs. III-A, III-C, IV-A, IV-C, Tables III and IV.
- What they do: adds contact audio to a Llama2-based VLA and adds collision-triggered sound to the RLBench and LIBERO simulators.
- Audio input and encoder: real piezo contact microphones on both sides of the gripper (Sec. IV-A). In simulation no sensor is simulated: sounds are retrieved from a library of real gripper-mic recordings indexed by material pair, interaction type and force, then scaled and pitch-shifted (Sec. III-C). AudioCLIP, pretrained, further trained on the ManiWAV dataset, with LoRA fine-tuning (Sec. III-A); whether base weights are frozen is not stated.
- Fusion: DINOv2 + SigLIP vision tokens (linear), audio tokens (3-layer MLP) and proprioception (2-layer MLP) are projected into a 7B Llama2 backbone with language tokens; an action head outputs continuous actions with L1 loss.
- Tasks and data: LIBERO (four suites) and RLBench (five tasks) in simulation with library-synthesized audio; real robot whiteboard erasing and oatmeal scooping with 40 teleoperated demonstrations each.
- Benefit of audio (ablation): RLBench (Table III, Sec. IV-C): vision-only average 48.0 versus full 55.1. Real world (Table IV): whiteboard erasing success 20 versus 60, oatmeal scooping success 10 versus 30. No audio ablation on LIBERO. The paper's text labels the whiteboard task "EAWM" and the tables "EAWS".
- Code or data released: "Both audio-enhanced simulation environments will be made open-source" (Sec. III-C); no model code or weights statement found.

### zhang2025learning: Learning Robot Manipulation from Audio World Models
- Authors: Fan Zhang, Michael Gienger.
- Year and venue: 2025, preprint, not peer reviewed (no venue on the arXiv page). arXiv 2512.08405.
- Link: https://arxiv.org/abs/2512.08405
- Read depth: abstract page + HTML full text (skim through the fetch summary; method and experiments).
- What they do: a flow-matching audio world model predicts future audio, and the robot policy conditions on current audio, the image and the predicted future audio.
- Audio input and encoder: omnidirectional USB lapel microphone (192 kHz recorded, 44.1 kHz processed; channel count not stated). AudioMAE encodes log-mel spectrograms, trained with a reconstruction loss in a first stage; frozen afterward is not stated.
- Fusion: a latent DiT flow-matching model generates future audio latents; the flow-matching policy takes current audio, current image and predicted future audio.
- Tasks and data: real water filling with a button-press dispenser (Kinova Gen3, teleoperated with a haptic device) and a simulated piano duet (the piano experiment uses MIDI and MusicVAE, not spectrogram audio, so the "audio world model" there is a MIDI generator).
- Benefit of audio (ablation): water filling 100% success over 30 trials; no no-lookahead or no-audio baseline numbers are reported in the text read. Piano: better than an RL baseline without lookahead, no figures in the text read.
- Code or data released: not mentioned in the text read.

### li2026hierarchical: Hierarchical Audio-Visual-Proprioceptive Fusion for Precise Robotic Manipulation
- Authors: Siyuan Li, Jiani Lu, Yu Song, Xianren Li, Bo An, Peng Liu.
- Year and venue: 2026, preprint, not peer reviewed (no venue on the arXiv page). arXiv 2602.13640 (submitted 2026-02-14).
- Link: https://arxiv.org/abs/2602.13640
- Read depth: abstract page + HTML full text, Secs. III, V-C, V-E, Tables I and III (via fetch summary).
- What they do: a hierarchical fusion module in which audio modulates point-cloud and proprioception features before a diffusion policy, for tasks where sound carries information vision lacks.
- Audio input and encoder: Audio-Technica AT2020 microphone with an audio interface; channel count not stated. Dual-path (temporal and spectral) encoder on log-mel spectrograms, pretrained with an autoencoding objective jointly with the proprioceptive encoder; text says all encoder parameters are optimized end-to-end (Sec. III-B).
- Fusion: Binary-Branched Fusion Module (audio-conditioned Hadamard modulation and FiLM) then an Interaction Modeling Module (cross-attention with an audio highway); the fused embedding conditions a diffusion policy.
- Tasks and data: real Franka Panda, liquid pouring and cabinet opening; expert trajectories (count and collection method not stated).
- Benefit of audio (ablation): Table I, Sec. V-C, mean over five trials. Pouring air-column height (lower is better): no audio 2.72 versus proposed 1.86; simple audio concatenation 3.08. Cabinet score (lower is better): no audio 7.58 versus proposed 4.50; simple concatenation 7.44. So naive concatenation did not help over no audio in these rows.
- Code or data released: not mentioned in the text read.

### nie2026towards: Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation
- Authors: Chang Nie, Tianchen Deng, Guangming Wang, Zhe Liu, Hesheng Wang.
- Year and venue: 2026; arXiv page states "Accepted by The International Journal of Robotics Research (IJRR 2026)" (v2 revised 2026-09-18; v1 2026-03-17). arXiv 2603.16086. The journal's own page was not opened, so the acceptance is per the arXiv page only.
- Link: https://arxiv.org/abs/2603.16086
- Read depth: abstract page + HTML full text, Secs. 5.1, 6.1, 6.2, 6.3, 6.4.1, Tables 2 to 4. The supplement was not read.
- What they do: a streaming-audio VLA with a causal audio memory, an omni-model reasoner (Qwen3-Omni), an audio world model that predicts near-future audio codes (training only), and a flow-matching action head, aimed at sound events that occur between action-chunk decisions.
- Audio input and encoder: "a 1D raw waveform vector sampled from a single microphone" (Sec. 6.1.1); real runs use one standard workspace microphone, and the authors state they avoided contact microphones. A small causal 1D convolutional patchifier (pretrained or frozen not stated); future-audio targets are codes from a causal streaming neural audio tokenizer; the reasoner is Qwen3-Omni with its own multimodal pretraining.
- Fusion: audio memory updated between decisions; the reasoner takes vision, instruction, proprioception, a 4 s audio window and the memory; a low-level Qwen3-0.6B reuses the KV cache to condition a flow-matching action head.
- Tasks and data: HEAR-Bench (simulation, seven tasks on RoboTwin2, event clips mixed into a background audio stream) and four real Franka Panda tasks. Pretraining set OpenX-Sound: Open X-Embodiment episodes with audio generated from video by video-to-audio models, with checks; the paper says it is not meant to replace real robot audio. Real-robot and benchmark data are scripted, not human: "rule-based controllers and motion planners produce sound-causal trajectories, while synchronized RGB, proprioception, actions, and microphone streams are recorded for imitation learning" (Sec. 6.1). So the real results say little about learning from human demonstrations.
- Benefit of audio (ablation): Tables 2 and 3 compare against audio-free base policies and audio variants. Simulation average success: OpenVLA 0.10, OpenVLA with raw waveform 0.53, OpenVLA with ASR 0.29, pi0.5 0.14, pi0.5 with waveform 0.61, HEAR 0.81 (Table 2). Real average: OpenVLA 0.06, pi0.5 0.13, pi0.5 with waveform 0.45, HEAR 0.70 (Table 3). Table 4 (component ablations, simulation average) varies HEAR's own parts (no Historizer 0.57, no OpenX-Sound pretraining 0.69) and has no audio-removed row. Note the base policies receive no audio and are different models, so Table 2 and 3 rows are the closest audio-versus-no-audio evidence, not a matched ablation of HEAR.
- Code or data released: "Code and videos are available at https://hear.irmv.top" (abstract); dataset "released through the project page and Hugging Face" (Sec. 5.1). Not checked.

### talwar2026youre: You're Pushing My Buttons: Instrumented Learning of Gentle Button Presses
- Authors: Raman Talwar, Remko Proesmans, Thomas Lips, Andreas Verleysen, Francis wyffels.
- Year and venue: 2026, ICRA 2026 workshop paper (per arXiv page; workshop, not a main-track paper). arXiv 2604.05954.
- Link: https://arxiv.org/abs/2604.05954
- Read depth: abstract page + HTML full text (via fetch summary; method and results).
- What they do: fine-tunes an audio encoder into a button-click detector using the button's own instrumented signal, then uses it in a Diffusion Policy so a vision-and-audio policy needs no force sensing at run time.
- Audio input and encoder: ICS-43434 microphone in a 3D-printed fingertip. Audio Spectrogram Transformer pretrained on AudioSet and fine-tuned as a click detector (10 epochs); frozen during policy training is not stated explicitly (future work mentions "partial AST unfreezing", which suggests frozen).
- Fusion: "Deep Fusion" (AST logits or embeddings fed to the policy during training) or "Soft Sensor" (policy trained on true button state, AST prediction at inference), with a wrist RGB ResNet-18.
- Tasks and data: real UR3e, approach, press, retract; automated data collection with randomized approach poses, no teleoperation.
- Benefit of audio (ablation): there is no vision-only baseline. Success rates 45% to 55% across models and statistically indistinguishable over 40 rollouts each; median peak vertical force was lower with fine-tuned audio fusion (5.87 N with embeddings, 5.95 N with classifier) than with the generic AST (6.98 N), expert 3.41 N (Fig. 3 and text).
- Code or data released: none found in the text read.

### hiratsuka2026s2a2: S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information
- Authors: Kaneyoshi Hiratsuka, Benjamin Yen, Ryosuke Kojima.
- Year and venue: 2026, preprint, not peer reviewed (no venue on the arXiv page; v1 2026-07-28, v2 2026-09-29). arXiv 2607.26047.
- Link: https://arxiv.org/abs/2607.26047
- Read depth: abstract page + HTML first 100000 characters (method, setup, Tables 2 and 3); the last part (about 56000 characters) was not read.
- What they do: gives imitation-learning policies (ACT, Diffusion Policy, VQ-BeT, pi0) an explicit sound-source position map computed from microphone arrays, alongside a spectrogram pipeline, for tasks where sound decides the target.
- Audio input and encoder: four circular microphone arrays around the workspace (real: TAMAGO-03, 8 channels each, 16 kHz). MUSIC direction-of-arrival is projected into a 224 x 224 workspace map per array. The spatial map and spectrogram encoders are ResNet-18s trained from scratch; a raw-audio baseline uses a 1D CNN, also from scratch.
- Fusion: policy-specific. ACT tokens, Diffusion Policy global conditioning, VQ-BeT GPT inputs, pi0 acoustic tokens in the PaliGemma prefix.
- Tasks and data: Localization, Identification, Localization & Identification, Exploratory (shake to find the sounding object). Simulation in Genesis with Pyroomacoustics and scripted demonstrations (100 or 200 episodes); real ALOHA-type robot, ACT only, 100 human teleoperated episodes per task.
- Benefit of audio (ablation): Table 2 (simulation, success %): ACT vision-only baseline 20.3 / 39.3 / 11.0 / 35.0 versus S2A2 73.0 / 76.7 / 78.7 / 77.3 on the four tasks. Removing the spatial map ("w/o Spat.") hurts the localization tasks (ACT Localization 15.3), so spatial information matters there; raw audio is a middle case. Table 3 (real, ACT): L&I 5 versus 72, Exploratory 12 versus 50.
- Code or data released: not mentioned in the text read; a project page link exists on the arXiv listing.

### kim2026soundact: SoundAct: Learning Spatial Sound Awareness for Egocentric Robot Manipulation with Stereo Audio
- Authors: Junsu Kim, Kyungdon Joo (UNIST, Ulsan, South Korea).
- Year and venue: 2026, ICRA 2026 Workshop on Robot Manipulation (per the project page; workshop paper, not main track). The page links an OpenReview forum at https://openreview.net/forum?id=Ia1lDHwoR5 (extension) and a PDF at https://openreview.net/pdf?id=yakI3P3Ib6; I did not open either from OpenReview, so the venue rests on the project page only. No arXiv id found.
- Link: https://vision3d-lab.github.io/soundact/
- Read depth: project page only, raw HTML fetched with curl and the full visible text (about 6.6 thousand characters) read; no paper PDF read.
- What they do: a wrist-camera Franka robot with a stereo microphone array learns a behavior-cloning diffusion policy that localizes and acts on a sound source that starts outside the camera view.
- Audio input and encoder: stereo microphone array on a wrist-mounted egocentric camera. Waveforms become magnitude spectrograms by STFT (phase dropped on purpose), encoded by a ResNet-18 and projected by an MLP to align with the visual features. The page does not say whether the ResNet-18 is pretrained. Vision is a CLIP-pretrained ViT ([CLS] tokens per frame).
- Fusion: a Transformer with self-attention over vision, audio and end-effector pose conditions a 1D-UNet diffusion policy that denoises 16-step relative action trajectories over 50 denoising steps. A spatial audio augmentation mixes ESC-50 distractor clips with independent per-channel scaling.
- Tasks and data: beyond-sight ring-off (turn off a ringing alarm clock that starts out of view). Trained on 60 teleoperated episodes with the target in a 20 cm x 20 cm region on either beyond-sight side; evaluated over 10 in-distribution trials and 5 trials per distractor type (music, conversation, typing, another beep, another clock alarm). Real robot.
- Benefit of audio (ablation): the page's table "Success rates on the beyond-sight ring-off task" (success / trials): vision-only 5 / 10, audio-only 7 / 10, mono audio (left only) 8 / 10, stereo audio (ours) 10 / 10; log-mel spectrogram 5 / 10, 4-channel magnitude-and-phase STFT 2 / 10, 2-channel magnitude STFT (ours) 10 / 10; distractors: no augmentation 0 / 5, background-audio augmentation 0 / 5, spatial augmentation (ours) 4 / 5. The page caption says vision-only "fails to localize the out-of-view source" and audio-only "fails to turn it off in time", but the table shows partial success for both. Trial counts are small (10 or 5). The stereo versus mono comparison (10 / 10 versus 8 / 10) is the closest thing in this survey to a direct test of whether direction information matters.
- Code or data released: not mentioned on the page.

### zhao2025polytouch: PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation Using Tactile-Diffusion Policies
- Authors: Jialiang Zhao, Naveen Kuppuswamy, Siyuan Feng, Benjamin Burchfiel, Edward Adelson.
- Year and venue: 2025; the arXiv comment reads "Nominated for the best paper award at ICRA 2025", which implies ICRA 2025 but is the only venue evidence seen. arXiv 2504.19341.
- Link: https://arxiv.org/abs/2504.19341
- Read depth: abstract page + HTML full text, the acoustic sensing, policy and Table II parts (via fetch summary).
- What they do: a robot finger combining camera-based tactile sensing, a contact microphone and peripheral vision, used in a tactile-diffusion policy. Audio is one component of a larger sensor, so this paper is a weak fit for the audio question.
- Audio input and encoder: piezo contact microphone on the back of the finger, 48 kHz; log-mel spectrogram into an Audio Spectrogram Transformer.
- Fusion: cross-attention (6 blocks, 12 heads) is applied between the tactile and peripheral-vision features and the scene-view features (Sec. IV-A). The output is pooled and combined with pooled or projected features from the other modalities, audio included, by a concatenation and projection layer, then fed to a U-Net diffusion policy. So audio goes through the concatenation path, not the cross-attention. Audio comes from the PolyTouch on each arm, so there are two audio streams.
- Tasks and data: bimanual real Franka tasks (Insert Wrench, Sort Fruit, Crack Egg, Serve Egg), teleoperated with SpaceMouse devices, about 70 to 200 demonstrations per task.
- Benefit of audio (ablation): audio is not ablated on its own. The "visuo-proprio" baseline zeroes both the tactile and audio inputs, so Table II gains (for example Serve Egg success 66% to 100% with all data) cannot be assigned to audio.
- Code or data released: not mentioned in the text read; project page polytouch.alanz.info.

### zhang2025vibecheck: VibeCheck: Using Active Acoustic Tactile Sensing for Contact-Rich Manipulation
- Authors: Kaidi Zhang, Do-Gon Kim, Eric T. Chang, Hua-Hsuan Liang, Zhanpeng He, Kathryn Lampo, Philippe Wu, Ioannis Kymissis, Matei Ciocarlie.
- Year and venue: 2025, IROS 2025 (per arXiv page; v2 revised 2026-06-18). arXiv 2504.15535.
- Link: https://arxiv.org/abs/2504.15535
- Read depth: abstract page + HTML full text, hardware, classifier and policy parts, Tables I and II (via fetch summary).
- What they do: a gripper with one piezo finger that emits a sweep and one that receives it (active acoustic sensing, not passive sound), used to classify contact state and drive a peg-insertion policy.
- Audio input and encoder: self-generated 20 Hz to 20 kHz sweep, received at 44.1 kHz; FFT then kernel PCA then MLP classifiers (no neural audio encoder, no pretrained audio model).
- Fusion: classifier output (contact type) feeds a small categorical imitation policy trained in simulation with noisy classifier observations sampled from the real confusion matrix.
- Tasks and data: peg insertion on a real UR5 with acoustic sensing as the only feedback; classifier data collected on the robot (for example 1,000 train and 200 test samples per contact type).
- Benefit of audio (ablation): the paper contains no experiment without acoustic sensing. Real UR5 insertion success was 6/10, 9/10 and 6/10 over three start conditions (Table II).
- Code or data released: no code statement found; project page roamlab.github.io/vibecheck.

### chang2023learning: Learning Visual-Audio Representations for Voice-Controlled Robots
- Authors: Peixin Chang, Shuijing Liu, D. Livingston McPherson, Katherine Driggs-Campbell.
- Year and venue: ICRA 2023 (per arXiv page; v1 2021-09-07, v3 2023-03-05). arXiv 2109.02823.
- Link: https://arxiv.org/abs/2109.02823
- Read depth: abstract page + HTML full text (via fetch summary; method, Sec. IV-C2, Table III).
- What they do: learns a joint visual-audio embedding so a robot can follow a spoken or sound command by relating the sound to the object it names, used inside a reinforcement-learning navigation and manipulation policy.
- Audio input and encoder: voice commands (Google Speech Commands, Fluent Speech Commands) and non-speech sounds (UrbanSound8K, NSynth), MFCC features; the sound encoder is not fixed ("any deep models ... can be used").
- Fusion: a three-branch Siamese network trained with a triplet loss on L2-normalized embeddings; the embedding goes into the policy input; the goal sound is not passed to the LSTM.
- Tasks and data: simulation (PyBullet TurtleBot and Kuka, AI2-THOR) plus a real Kinova Gen3 block-choosing task; 50,000 simulated audio-visual triplets, 993 real triplets for fine-tuning.
- Benefit of audio (ablation): Table III (iTHOR, FSC): "No current sound" 61.5 versus full method 71.0 success. This is a goal-sound command setting, not environmental or contact audio.
- Code or data released: not mentioned in the text read.

## Could not verify
- SoundAct's OpenReview pages (https://openreview.net/forum?id=Ia1lDHwoR5 and https://openreview.net/pdf?id=yakI3P3Ib6): the first attempt returned a browser verification screen, and the reviewer also got HTTP 403 for both. The venue is unconfirmed beyond the project page, which says ICRA 2026 Workshop.
- A "Embodied audio survey" listed on Preprints.org (manuscript 202607.2237) appeared in a search result about sound in embodied AI. It was not opened, so it is not described. Its relevance (a taxonomy that may cover contact audio and spoken interaction) is a lead only.
- NAC: Neural Action Codec (arXiv 2606.21372) appeared in a search. It tokenizes actions with audio-codec ideas and is not an audio-input paper, so it was not opened and is not listed.
- Two papers named in the HEAR tables as baselines (Play it by Ear, ManiWAV) are covered above. No other HEAR baselines were chased.

- Venue evidence for PolyTouch is only the arXiv comment about an ICRA 2025 best-paper nomination; inclusion in the ICRA 2025 proceedings is unconfirmed. Du 2022 success rates are read from Fig. 5 and the text (approximate). Section and table labels in the read-depth notes were checked for numbers and quotes, not every section label. "Code or data released" lines were not checked beyond the quoted sentences. ManiWAV's robot (UR5) is implied rather than stated outright in the raw text.
- Key `wang2025sound` is kept: the title is "The Sound of Simulation", and skipping the leading article, as is usual, gives first word "Sound". The standards file does not state an article rule, so this is a judgment call for the lead.

## Searches run
All were web searches on 2026-11-16.
1. audio vision-language-action model robot manipulation sound input arXiv
2. contact microphone robot manipulation policy audio vision imitation learning arXiv
3. in-hand microphone audio diffusion policy robot learning from sound
4. speech-conditioned vision-language-action model VLAS end-to-end speech robot manipulation
5. See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation
6. robot policy microphone array sound source localization manipulation vision audio policy 2025 2026 arXiv
7. omni-modal vision-audio-language-action model robot audio understanding 2026
8. audio-visual robot policy "sound" ablation vision-only baseline manipulation pretrained audio encoder AudioMAE OR CLAP OR Whisper OR AST
9. Hearing Touch: Audio-Visual Pretraining for Contact-Rich Manipulation
10. SoundAct stereo microphone wrist camera behavior cloning out-of-view sound source robot
11. vision-language-action model acoustic perception audio tokens action expert 2026 arXiv sound-aware VLA
12. "Play it by Ear" robot audio policy arXiv
13. robot learning audio-visual policy wearable microphone OR "audio" tactile dexterous manipulation vision-audio diffusion policy 2025
14. S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information arXiv

## What exists and what I did not find
Contact and in-hand microphones are the best-covered case: from 2022 to 2025 several groups showed that adding audio to a vision policy helps on contact-rich real-robot tasks (pouring, flipping, wiping, insertion, occluded grasping), and the ablations in those papers are mostly audio versus vision-only with a clear gap. Most of these use a small audio encoder trained from scratch or an AudioSet-style pretrained one (AST, AVID, AudioCLIP), fused by a transformer or concatenation into a behavior-cloning or diffusion policy. Audio-VLA and HEAR are the two papers I found that attach audio to a large VLA backbone; HEAR is the only one with streaming audio and a predicted-future-audio component, and its audio-free comparisons are against different base models, not a matched ablation.

Spatial audio is much thinner. I found two papers that use more than one microphone channel in a policy (SoundAct with stereo, a workshop paper known only from its project page, which does give a small real-robot table where stereo beat mono and vision-only, and S2A2 with microphone arrays and a computed source-position map, a July 2026 preprint). I did not find a peer-reviewed paper that feeds binaural or multichannel audio into a VLA or a video-prediction model. I did not find any paper that gives audio to a world action model (video plus action prediction) directly; the closest are the audio-world-model paper by Zhang and Gienger (predicts future audio, not video) and HEAR (predicts future audio codes). I did not run a systematic search for audio-visual navigation policies, which belong to topic B.

Things worth treating with care: many results rest on a small number of real-robot trials, several papers do not say whether the audio encoder is frozen, simulated audio in the VLA papers is synthesized (library retrieval, video-to-audio generation, TTS), and the HEAR numbers depend on a benchmark the same authors built. All of this is `verified` only as far as what each paper states; none of it has been reproduced.
