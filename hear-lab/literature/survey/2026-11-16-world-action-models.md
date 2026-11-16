# Literature survey C: world action models, and which could take audio (2026-11-16)

Topic: how the literature defines "world action model" (WAM) and close terms; the WAM / video-action models themselves; and any world model, WAM or video prediction model that takes or generates audio, plus joint audio-video generators a WAM could be built on.

How the search was done: web search (queries listed at the end), then every paper below was opened on its arXiv abstract page (title, authors, date, venue, identifier confirmed there) and, for most, its arXiv HTML full text. Code and weights claims come from the paper's own statement and, where marked "opened", from the GitHub repository page read on 2026-11-16. Status of everything below: `verified` means read on the page on 2026-11-16.

Caveats the user should know:
- The page-reading tool returns a model-written extract of each page, not the raw text. Quotes marked "(fetch extract)" came back that way and should be re-checked against the paper before they go into a writeup. The definition quotes from Wang et al. 2026 (2605.12090) were read from the PDF text directly and are exact.
- Long HTML pages were read only up to about the first 100,000 characters (noted per paper). Appendices at the end may be unread.
- Every arXiv paper below is "preprint, not peer reviewed" unless a venue is stated; a venue is stated only where a primary page (ICLR or CVPR site) listed it.
- Numbers are given only where copied from the paper, with the section or table named. Success rates are deliberately left out; the papers' own tables have them.

---

## Part 1: definitions of "world action model" and close terms

### wang2026world: World Action Models: The Next Frontier in Embodied AI
- Authors: Siyin Wang, Junhao Shi, Zhaoyang Fu, Xinzhe He, Feihong Liu, Chenchen Yang, Yikang Zhou, Zhaoye Fei, Jingjing Gong, Jinlan Fu, Mike Zheng Shou, Xuanjing Huang, Xipeng Qiu, Yu-Gang Jiang
- Year and venue: 2026 (v1, 12 May 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2605.12090
- Read depth: abstract + Sec. 1 (intro), 2.2, 5.1.2 and the open-challenges text, read from the PDF text; the rest not read
- Part: 1
- What they do: A survey that names and formalizes WAMs and splits them into Cascaded and Joint WAMs.
- Definition quote (exact, Sec. 1): "We formalize this family of approaches under the term World Action Models (WAMs): embodied foundation models that unify predictive state modeling with action generation, targeting a joint distribution p(o', a | o, l) over future states and actions rather than actions alone." Sec. 1 then says Cascaded WAM "explicitly factorizes the objective ... by first synthesizing representations of anticipated future states, from which actions are subsequently derived" and Joint WAM models the joint distribution "where state prediction and action generation are co-optimized within a shared representational space".
- Definition quote, Sec. 2.2 ("Video Action Models (VAMs)"): "VAMs often refer to the models that integrate video prediction with action generation, typically aligning action with synthesized visual futures. We define WAMs as a broader, modality-independent superset of predictive agents." Same section: WAMs "encompass models that utilize other predictive targets such as single-image state transitions, dense point clouds, or multi-sensory modalities like tactile and force feedback."
- Inputs: not a model. Sec. 2 defines the observation o as "encompassing visual inputs, proprioceptive signals, and any other sensory modalities".
- Built on: n/a (survey)
- Benchmarks: n/a (survey lists LIBERO, RoboTwin, RoboCasa and others in its evaluation taxonomy)
- Training compute stated: n/a
- Audio (input, output, or none mentioned): discussed. Sec. 5.1.2: "Berkeley UR5 [353] and OmniAction [135] introduced audio modalities, capturing acoustic signatures essential for material identification and impact verification." Open challenges: "Existing WAMs almost universally predict future world states in the RGB visual modality ... much of the physical information most critical for contact-rich manipulation is largely invisible in pixel space, including tactile distributions, contact forces, acoustic signatures, and material compliance." It calls extending WAMs to other modalities "an important and largely unexplored frontier" (its explicit examples are tactile, force, proprioceptive).
- Code and weights: n/a (survey)

### shen2026world: World Action Models: A Survey
- Authors: Qiuhong Shen, Shihua Zhang, Yue Liao, Qi Li, Zhenxiong Tan, Shizun Wang, Shuicheng Yan, Xinchao Wang
- Year and venue: 2026 (v1, 18 June 2026); arXiv preprint, not peer reviewed (comments: "57 pages, 6 figures")
- Link: https://arxiv.org/abs/2606.20781
- Read depth: abstract + Sec. 2.1 to 2.3, 3.3, 4.2.1 (first ~100,000 characters of the HTML, fetch extract; the remaining ~350,000 characters were not read)
- Part: 1 and 3
- What they do: A survey that separates WAMs from world models, video generators, action-grounded video world models and VLAs, and organizes methods by what they must generate and their predictive substrate.
- Definition quote (abstract, fetch extract): "World Action Models (WAMs) are embodied predictive-action models that make a forecast of the future available to action." Sec. 2.3 (fetch extract): "A World Action Model begins when the predicted future observation becomes part of how the action is obtained." Sec. 2.1: "It becomes a WAM only when a predicted future observation helps produce, choose, or check the action." Sec. 2.2: "Video generation is one visible instance of this broader idea, not its definition."
- Inputs: n/a (survey)
- Built on: n/a
- Benchmarks: n/a
- Training compute stated: n/a
- Audio (input, output, or none mentioned): discussed. Sec. 3.3 (fetch extract): Audio-WM "predicts future Mel latents that condition a robot policy when contact sound carries task evidence." Sec. 4.2.1: "Acoustic observation latents are rare in the current literature."
- Code and weights: n/a. Survey homepage listed on the arXiv page: https://world-action-models.github.io/ (not opened)

### li2025unified: Unified Video Action Model
- Authors: Shuang Li, Yihuai Gao, Dorsa Sadigh, Shuran Song
- Year and venue: 2025 (v1 28 Feb 2025, v3 24 Apr 2025); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2503.00200
- Read depth: abstract + Sec. III, V, X and the discussion (first ~100,000 characters of v3 HTML, fetch extract)
- Part: 1 and 2
- What they do: UVA learns one joint video-action latent and decodes video and actions with two light diffusion heads, so action inference can skip video generation. Masking lets one model act as policy, forward dynamics, inverse dynamics or video generator.
- Definition quote (abstract, fetch extract): UVA "jointly optimizes video and action predictions to achieve both high accuracy and efficient action inference". The paper does not use the term "world action model".
- Inputs: history image observations and action chunks; CLIP text encoder for language tasks (Sec. III).
- Built on: "Our model builds on the pretrained model (MAR-B) released by [27] but has undergone substantial modifications" (Sec. III); pretrained VAE (kl-f16).
- Benchmarks: PushT, Toolhang, PushT-M, LIBERO-10 (Sec. V-A); real-world Cup Arrangement, Towel Folding, Mouse Arrangement on an ARX X5 arm (Sec. V-B); also video generation (FVD), forward and inverse dynamics.
- Training compute stated: none stated (only a thanks for computing resources).
- Audio (input, output, or none mentioned): none in the model. Discussion (fetch extract): "our model can be naturally extended to predict modalities beyond video and action, such as sound and force."
- Code and weights: project website https://unified-video-action-model.github.io/ is linked; the paper says "Please check out our code for details" (Sec. X-A). I did not open a repository or see a weights statement.

### pai2025mimic: mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
- Authors: Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, Elvis Nava
- Year and venue: 2025 (v1 17 Dec 2025, v2 19 Dec 2025); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2512.15692
- Read depth: abstract + Sec. I, IV, V (v2 HTML, fetch extract)
- Part: 1 and 2
- What they do: Pairs a pretrained video model with a flow-matching action decoder that acts as an inverse dynamics model on the video model's intermediate features.
- Definition quote (Sec. I and IV, fetch extract): "establishing a new class of methods we term Video-Action Models (VAMs)"; "a generative Video-Action Model (VAM) capable of modeling the joint distribution of video and robot actions".
- Inputs: multiple RGB images, a language instruction, proprioceptive state (Sec. IV-B).
- Built on: Cosmos-Predict2, "an open-source 2B latent Diffusion Transformer (DiT)" (Sec. IV-C).
- Benchmarks: SIMPLER-Bridge, LIBERO (Goal, Object, Spatial), real bimanual dexterous manipulation (Sec. V).
- Training compute stated: no GPU count or hours; acknowledgments name the Swiss National Supercomputing Centre (CSCS) Alps infrastructure.
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: project page https://mimic-video.github.io linked; I saw no explicit code or weights statement on the paper page.

---

## Part 2: world action and video-action models

### ye2026world: World Action Models are Zero-shot Policies (DreamZero)
- Authors: Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, Joel Jang
- Year and venue: 2026 (v1, 17 Feb 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2602.15922
- Read depth: abstract + Sec. 1, 2.2, 3.1, 4.1, App. C, D.3 (first ~100,000 of ~147,000 characters of the HTML, fetch extract)
- Part: 1 and 2
- What they do: DreamZero is a 14B autoregressive video diffusion model that jointly denoises future video and action chunks; it is the paper that uses "World Action Model" as the headline term.
- Definition quote (Sec. 1, fetch extract): a WAM is "a foundation model designed to predict both actions and visual future states in an aligned manner." Abstract: "WAMs learn physical dynamics by predicting future world states and actions". Sec. 2.2: the name was chosen because "video is just one possible world modeling objective."
- Inputs: multi-view cameras concatenated into one frame, a text-encoded language instruction, robot state through a state encoder, VAE-encoded current and past observations (Sec. 3.1).
- Built on: "Wan2.1-I2V-14B-480P" (Sec. 4.1), an image-to-video diffusion backbone, trained with flow matching.
- Actions: video and actions jointly denoised, produced autoregressively in chunks with a KV cache (Sec. 3.1, App. C).
- Benchmarks: real robots AgiBot G1 and DROID-Franka, YAM cross-embodiment; the paper says it provides code for RoboArena, PolaRiS and Genie Sim 3.0. No LIBERO seen in the part I read.
- Training compute stated: steps and batch size only ("We train for 100K steps with a global batch size of 128"); no GPU count or GPU-hours seen in the part read. Inference on "2 GB200s" for 7Hz.
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: repository https://github.com/dreamzero0/dreamzero opened: Apache-2.0, training, LoRA and full fine-tuning code, checkpoints on Hugging Face (DreamZero-DROID, DreamZero-AgiBot). The AgiBot dataset is "plan[ned]" for later release (Sec. 1).

### yuan2026fast: Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- Authors: Tianyuan Yuan, Zibin Dong, Yicheng Liu, Hang Zhao
- Year and venue: 2026 (v1 17 Mar 2026, v2 23 Mar 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2603.16666
- Read depth: abstract + Sec. 1, 2, method and experiments (HTML v2, fetch extract)
- Part: 1 and 2
- What they do: Trains with video co-training but skips future-frame generation at inference, to test whether the benefit of WAMs comes at test time or from training.
- Definition quote (Sec. 1, fetch extract): "World Action Models (WAMs), which combine future visual prediction and action modeling in a unified framework." Abstract: WAMs "explicitly model how visual observations may evolve under action". "Video-action model" is not defined; it appears only in a reference title.
- Inputs: training uses current observation, language, noisy future video latents and action tokens; inference uses only the first-frame latent tokens plus language.
- Built on: Wan2.2-5B video DiT, T5 text encoder and video VAE, plus a 1B action-expert DiT (6B total).
- Actions: flow matching in the action expert, 10 denoising steps; no future video at test time.
- Benchmarks: LIBERO (Spatial, Object, Goal, Long), RoboTwin 2.0, real towel folding on Galaxea R1 Lite.
- Training compute stated: none (no GPU count or hours; only steps, 20k for LIBERO and 30k for RoboTwin and real). Inference latency measured on one RTX 5090D V2 32GB.
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: https://github.com/yuantianyuan01/FastWAM opened: official codebase, training scripts, checkpoints on Hugging Face (`yuanty/fastwam`: LIBERO and RoboTwin checkpoints). License type not shown in what I read.

### kim2026cosmos: Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning
- Authors: Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, Jinwei Gu
- Year and venue: 2026 (v1, 22 Jan 2026); ICLR 2026 poster (listed on https://iclr.cc/virtual/2026/poster/10006732, opened; the arXiv page itself lists no venue)
- Link: https://arxiv.org/abs/2601.16163
- Read depth: abstract + Sec. 3, 4.1, 4.3, 5.1, 5.3, 7, App. A (first ~100,000 of ~107,000 characters of the HTML, fetch extract)
- Part: 1 and 2
- What they do: Fine-tunes the pretrained Cosmos-Predict2 video model in one post-training stage, with no architecture change, to output actions, future state images and value as latent frames.
- Definition quote: none for "world action model". Abstract (fetch extract): actions are "encoded as latent frames within the model's diffusion process".
- Inputs: proprioception, multiple camera views, language (T5-XXL). ALOHA setup: "robot proprioceptive state (14 joint angles), three camera images, and task descriptions" (Sec. 5.1).
- Built on: Cosmos-Predict2-2B, fully fine-tuned (App. A.2).
- Actions: action chunk injected as latent frames and denoised jointly with future state and value (Sec. 4.1); best-of-N planning (Sec. 4.3).
- Benchmarks: LIBERO (Table 1), RoboCasa (Table 2), real ALOHA bimanual tasks (Table 3).
- Training compute stated (App. A.2.2 to A.2.4): LIBERO "64 H100 GPUs with global batch size 1920 (taking 48 hours total)"; RoboCasa 32 H100 GPUs, 48 hours; ALOHA 8 H100 GPUs, 48 hours.
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: https://github.com/NVlabs/cosmos-policy opened: Apache-2.0, fine-tuning and evaluation docs for LIBERO, RoboCasa and ALOHA, checkpoints in the Hugging Face collection `nvidia/cosmos-policy`. The paper's abstract says it releases code, models and training data.

### zhu2025unified: Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets
- Authors: Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, Abhishek Gupta
- Year and venue: 2025 (v1 3 Apr 2025, v3 23 May 2025); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2504.02792
- Read depth: abstract + Sec. III-B, IV, App. A (first ~100,000 of ~110,000 characters of the HTML, fetch extract)
- Part: 1 and 2
- What they do: One transformer runs action diffusion and video diffusion with separate timesteps per modality, so it can act as policy, forward or inverse dynamics, or video generator; action-free video can be used for pretraining.
- Definition quote (abstract, fetch extract): UWM is "a framework that allows for leveraging both video and action data for policy learning", with "independent diffusion timesteps govern[ing] each modality".
- Inputs: current multi-camera observations, noisy action chunk, noisy future observations, two timesteps (App. A-1).
- Built on: not a pretrained video diffusion model; the image encoder starts from ImageNet weights, and the model is pretrained on DROID subsets (App. A-1, IV-B1).
- Actions: reverse diffusion on the action variable, DDIM with 10 steps (Sec. III-B, App. A-2).
- Benchmarks: five real Franka tasks on the DROID platform; LIBERO (pretrain LIBERO-90, fine-tune LIBERO-10).
- Training compute stated (App. A-3): "Training a UWM on the DROID dataset for 100K gradient steps" "takes 24 hours on 4 NVIDIA A100 GPUs".
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: abstract points to https://weirdlabuw.github.io/uwm/ for "videos and code"; no weights statement seen; repository not opened.

### ma2026dit4dit: DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- Authors: Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, Shuo Yang
- Year and venue: 2026 (v1 11 Mar 2026, v2 22 Mar 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2603.10448
- Read depth: abstract + Sec. 4.2, 4.4, 5.4, release statement (first ~100,000 of ~111,000 characters of the HTML, fetch extract)
- Part: 1 and 2
- What they do: A cascaded system where an action DiT reads intermediate features of a video DiT (not decoded frames), trained end to end with a dual flow-matching loss.
- Definition quote (Sec. 4 intro, fetch extract): "an integrated Video-Action Model (VAM) designed for the joint optimization of Video and Action DiTs."
- Inputs: current frames and a language goal (embedded with Cosmos-Reason1); the action branch also gets proprioception (Sec. 4.2).
- Built on: Cosmos-Predict2.5-2B video backbone; action DiT adapted from GR00T-N1 (Sec. 4.2).
- Actions: hidden activations from the video DiT at a fixed flow timestep (default layer 18, Sec. 5.4) feed the action DiT by cross-attention.
- Benchmarks: LIBERO, RoboCasa-GR1 tabletop (24 tasks), real Unitree G1 (Sec. 5).
- Training compute stated: none (data size only, 241,450 pre-training episodes).
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: https://github.com/Mondo-Robotics/DiT4DiT opened: MIT license, training code, checkpoints on Hugging Face (`mondo-robotics/dit4dit-model`, LIBERO and RoboCasa-GR1). Abstract: "We release code and models".

### bi2025motus: Motus: A Unified Latent Action World Model
- Authors: Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu
- Year and venue: 2025 (v1 15 Dec 2025, v2 25 Dec 2025); CVPR 2026 poster (listed on https://cvpr.thecvf.com/virtual/2026/poster/38174, opened; the arXiv page lists no venue)
- Link: https://arxiv.org/abs/2512.13030
- Read depth: abstract + Sec. 3, 4.1, 9 (first ~100,000 of ~107,000 characters of v1 HTML, fetch extract)
- Part: 1 and 2
- What they do: A Mixture-of-Transformer with understanding, video-generation and action experts and a UniDiffuser-style scheduler that switches between world-model, VLA, inverse-dynamics and joint video-action modes; latent actions come from optical flow.
- Definition quote (Abstract, Sec. 4.1, fetch extract): "a unified latent action world model that leverages existing general pretrained models and rich, sharable motion information."
- Inputs: observation frame and language instruction (proprioception defined in Sec. 3 but not in the main equations).
- Built on: "we employ Wan 2.2 5B [42] as the video foundation model", Qwen3-VL-2B as understanding expert, new action expert (Sec. 4.1).
- Actions: joint rectified-flow denoising of video and action chunks, 10 flow-matching steps (Sec. 4.1, 9.1).
- Benchmarks: RoboTwin 2.0, real AC-One and Agilex-Aloha-2, LIBERO-Long, VLABench.
- Training compute stated: none (batch sizes and learning rates only, Sec. 9.2).
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: project page https://motus-robotics.github.io/motus is the only link in the paper; I did not see a code or weights statement. A GitHub URL I guessed returned 404, so I make no claim either way.

### li2026causal: Causal World Modeling for Robot Control (LingBot-VA)
- Authors: Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, Yinghao Xu
- Year and venue: 2026 (v1 29 Jan 2026, v2 22 Mar 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2601.21998
- Read depth: abstract + Sec. 3.2, 3.3, 4.2, 4.3, 6 (first ~100,000 of ~118,500 characters of the HTML, fetch extract)
- Part: 1 and 2
- What they do: An autoregressive diffusion model with a shared video-and-action token stream (Mixture-of-Transformers) and an asynchronous pipeline that predicts actions while the robot executes.
- Definition quote (Abstract, Sec. 3.2, fetch extract): "an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously"; the model "jointly models visual observations and robot actions within a single autoregressive process."
- Inputs: causal-VAE video latents, T5 language through cross-attention, action history (Sec. 3.2, 4.2).
- Built on: video stream initialized from Wan2.2-5B and its causal VAE (Sec. 3.3, 4.2).
- Actions: a separate action stream, flow matching, acts like an inverse dynamics model conditioned on predicted future latents (Sec. 3.2, 3.3).
- Benchmarks: RoboTwin 2.0 (Table 1), LIBERO (Table 2), six real tasks.
- Training compute stated: no GPU count or hours; "We pretrain LingBot-VA on the curated dataset for 1.4T tokens." (Sec. 4.2)
- Audio (input, output, or none mentioned): future work only, Sec. 6: "incorporating multi-modal sensory inputs (tactile, force, audio)" (fetch extract). Not in the model.
- Code and weights: https://github.com/robbyant/lingbot-va opened: Apache-2.0, post-training scripts for RoboTwin and LIBERO, checkpoints in the Hugging Face collection `robbyant/lingbot-va` (base, RoboTwin and LIBERO-Long post-trained).

### zhou2026tau0wm: tau0-WM: A Unified Video-Action World Model for Robotic Manipulation
- Authors: Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu, Yike Pan, Songen Gu, Kuanning Wang, Shufeng Nan, Xingyu Qiu, Chenhao Qiu, Pu Yang, Yunuo Cai, Jianxiong Gao, Yifan Li, Yanwei Fu, Xiangyu Yue, Zhi Chen, Jianlan Luo
- Year and venue: 2026 (v1 31 May 2026, v2 23 Aug 2026); arXiv preprint, not peer reviewed (the title uses the Greek letter tau-zero)
- Link: https://arxiv.org/abs/2606.01027
- Read depth: abstract + Sec. IV-A, IV-B, VII, App. A1 (HTML, fetch extract; page length not reported)
- Part: 1 and 2
- What they do: One video-diffusion backbone serves as a policy (video plus action chunks) and as an action-conditioned simulator used to score or correct candidate actions at inference.
- Definition quote (Abstract, fetch extract): "a unified video-action world model that integrates policy learning, video prediction, and action evaluation".
- Inputs: multi-view observations, language instructions, robot state; the simulator also takes a candidate action chunk (Sec. IV-A).
- Built on: "VAM is instantiated from Wan2.2-TI2V-5B" with a 0.5B action decoder, flow matching (Sec. IV-B).
- Actions: video action model jointly predicts future visual latents and continuous action chunks (Abstract, Sec. IV-B/C).
- Benchmarks: no public benchmark named; four real-robot tasks on AGIBOT-G01, ARX and dual-arm Franka, baselines pi0.5 and Fast-WAM (Sec. VII).
- Training compute stated: no GPU-hours; about 27.3K hours of data; global batch sizes 12,288 (pre-training) and 384 (post-training) (App. A1).
- Audio (input, output, or none mentioned): none mentioned.
- Code and weights: project page https://tau0-wm.github.io/ only; I saw no code or weights statement.

---

## Part 3: audio in world models and robots, and joint audio-video generators

### wang2025audio: Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics
- Authors: Jiahua Wang, Leqi Zheng, Jialong Wu, Yaoxin Mao, Shijie Cheng
- Year and venue: 2025 (v1 30 Nov 2025; v4 revised 4 Aug 2026 per the arXiv page); arXiv preprint, not peer reviewed. I read the v1 HTML, so details may differ in v4.
- Link: https://arxiv.org/abs/2512.00883
- Read depth: abstract + Sec. 3.1, 3.2, 4.1, 5.1, 5.4 (v1 HTML, fetch extract)
- Part: 3
- What they do: Casts audio-visual simulation under action control as a POMDP and builds AV-CDiT, a diffusion transformer that predicts the next binaural audio and next frame given an action, plus a reward; released with the AVW-4k action-annotated benchmark.
- Definition quote (Sec. 3.1, fetch extract): "An Audio-Visual World Model (AVWM) is designed to simulate an external environment containing sound signals."
- Inputs: 128x128 egocentric RGB; binaural audio in 0.15 s segments at 16 kHz; one of four discrete actions (forward 0.15 m, turn left 10 degrees, turn right 10 degrees, stop); preceding frames (Sec. 3.1, 3.2).
- Built on: CDiT from Navigation World Models (NWM), initialized from a pretrained base NWM; Stable Diffusion VAE for frames; a SoundStream tokenizer trained on AVW-4k for audio (Sec. 5.1). Not a large pretrained video diffusion model.
- Benchmarks: AVW-4k (Matterport3D scenes with SoundSpaces 2.0, 76 scenes, 4,500 trajectories, about 30 hours, Sec. 3.2); downstream continuous audio-visual navigation (Sec. 5.4). Simulated agent, no physical robot, no manipulation.
- Training compute stated (Sec. 5.1): "All experiments are conducted on a single machine equipped with eight NVIDIA A100 GPUs."
- Audio (input, output, or none mentioned): input and output, binaural (two channels; metrics per channel, Sec. 5.1). Closest thing found to a spatial-audio, action-conditioned world model.
- Code and weights: none stated; the arXiv page links no repository.

### zhang2025learning: Learning Robot Manipulation from Audio World Models
- Authors: Fan Zhang, Michael Gienger
- Year and venue: 2025 (v1, 9 Dec 2025); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2512.08405
- Read depth: abstract + Sec. 2, 3.1, 3.2 (HTML v1, fetch extract)
- Part: 3 (also relevant to topic A)
- What they do: A flow-matching model predicts future audio latents from recent audio, and a robot policy acts on the predicted audio (water filling, a simulated piano task).
- Definition quote (abstract, Sec. 2, fetch extract): "a generative latent flow matching model to anticipate future audio observations"; "flow matching as the backbone of our world model to generate a sequence of future audio latents".
- Inputs: the world model takes audio only (log Mel spectrogram, 128 Mel bins, 10 ms shift, about 1.28 s in, about 2.56 s out); the policy also takes 224x224 images (Sec. 3.1). The paper does not state microphone count or spatial layout; it names one "MAONO omnidirectional USB lapel microphone" (Sec. 3.1).
- Built on: AudioMAE autoencoder, DiT-style vector-field predictor, a flow-matching policy from the authors' earlier work (Sec. 2). Not a video model.
- Benchmarks: real water filling on a Kinova Gen3 (Sec. 3.1); simulated RoboPianist (Sec. 3.2).
- Training compute stated (Sec. 2): "requires approximately one day on a single NVIDIA H100 GPU" for 20,000 spectrograms.
- Audio (input, output, or none mentioned): input and predicted output; one omnidirectional microphone as far as I could see (not spatial). It predicts audio, not video.
- Code and weights: none stated; the arXiv page links no repository.

### hacohen2026ltx2: LTX-2: Efficient Joint Audio-Visual Foundation Model
- Authors: Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, Zeev Farbman
- Year and venue: 2026 (v1, 6 Jan 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2601.03233
- Read depth: abstract + architecture, audio and release text of the HTML (fetch extract); training and evaluation sections not read
- Part: 3
- What they do: Open joint audio-video generator: asymmetric dual-stream transformer, 14B video stream and 5B audio stream, bidirectional audio-video cross-attention.
- Definition quote: n/a
- Inputs: text prompts (frozen Gemma3-12B features); the paper describes V2A (audio for an existing video) and A2V (video driven by a given audio track) workflows. The paper extract did not describe image conditioning, but the repository lists image-to-video pipelines (see below).
- Built on: builds on the design of LTX-Video; the conclusion calls it an extension of "a pretrained 13B video diffusion transformer" with a "lightweight 3B audio stream" (the extract flagged this as inconsistent with the abstract's 14B and 5B). Not built on Wan.
- Benchmarks: generation quality and speed; no robot benchmark.
- Training compute stated: not stated in the part read.
- Audio (input, output, or none mentioned): output, stereo, two-channel at 24 kHz via a vocoder; audio can also condition video (A2V). Spatial (direction) properties of the audio were not examined in what I read.
- Code and weights: abstract: "All model weights and code are publicly released." https://github.com/Lightricks/LTX-2 opened: Apache-2.0 code, "ltx-trainer" covers LoRA, full fine-tuning and IC-LoRA, audio-to-video pipeline, text/image-to-video pipelines; weights on Hugging Face. The repository now lists newer releases (LTX-2.3, LTX-2.5) with their own community licenses, so the paper's model and the repository's current model are not necessarily the same version.

### low2025ovi: Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation
- Authors: Chetwin Low, Weimin Wang, Calder Katyal
- Year and venue: 2025 (v1, 30 Sep 2025); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2510.01284
- Read depth: abstract + Sec. 3.2, 4.1, 4.2.1, 4.3 (HTML v1, fetch extract)
- Part: 3
- What they do: Two matched DiT towers (video and audio) fused block by block with bidirectional cross-attention, generating audio and video together.
- Definition quote: n/a
- Inputs: text only, one frozen T5 on a combined video-caption and audio-description prompt (Sec. 4.1, 3.2); the paper extract describes no image conditioning (the repository does, see below).
- Built on: "The video branch is initialized from Wan2.2 5B, and an identical audio branch is trained from scratch." (Sec. 4.1)
- Benchmarks: generation quality; no robot benchmark. 5 s clips, 24 fps, 720x720.
- Training compute stated: none (steps, batch sizes, learning rates only, Sec. 4.3).
- Audio (input, output, or none mentioned): output, 16 kHz with a BigVGAN vocoder (Sec. 4.2.1); the paper does not state mono or stereo.
- Code and weights: abstract points to https://aaxwaz.github.io/Ovi. https://github.com/character-ai/Ovi opened: Apache-2.0, weights on Hugging Face (`chetwinlow1/Ovi`), text and image-to-audio-video modes, training and fine-tuning scripts listed as a to-do and not released.
- Relevance note: same Wan2.2 5B family that Fast-WAM, Motus, LingBot-VA and tau0-WM use for their video backbone, which makes it the nearest audio-video sibling of those WAMs.

### yu2026mova: MOVA: Towards Scalable and Synchronized Video-Audio Generation
- Authors: SII-OpenMOSS Team (Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng, Ruixiao Li, Tianyi Liang, Wenbo Zhang, Wenming Tu, Xiangyu Peng, Yang Gao, Yanru Huo, Ying Zhu, Yinze Luo, Yiyang Zhang, Yuerong Song, Zhe Xu, Zhiyu Zhang, Chenchen Yang, Cheng Chang, Chushu Zhou, Hanfu Chen, Hongnan Ma, Jiaxi Li, Jingqi Tong, Junxi Liu, Ke Chen, Shimin Li, Shiqi Jiang, Songlin Wang, Wei Jiang, Zhaoye Fei, Zhiyuan Ning, Chunguo Li, Chenhui Li, Ziwei He, Zengfeng Huang, Xie Chen, Xipeng Qiu)
- Year and venue: 2026 (v1 9 Feb 2026, v2 10 Feb 2026); arXiv preprint (technical report), not peer reviewed
- Link: https://arxiv.org/abs/2602.08794
- Read depth: abstract + Sec. 2.1, 2.2, 3.3, 4.1 to 4.3, 4.5 and the intro release sentence (first ~100,000 of ~157,000 characters of the v2 HTML, fetch extract)
- Part: 3
- What they do: Open Mixture-of-Experts video-audio generator (32B total, 18B active parameters per the abstract) with image-and-text-to-video-audio support.
- Definition quote: n/a
- Inputs: text prompt, optionally a first-frame image (Sec. 2.1); no audio input.
- Built on: video tower from Wan2.2 I2V A14B weights (Sec. 2.2, 4.1; the extract flagged that Sec. 2.1 cites the Wan2.1 VAE); audio tower uses a Wan2.1-1.3B architecture trained separately; DAC-style audio VAE from HunyuanVideo-Foley (Sec. 2.1, 4.2).
- Benchmarks: generation quality; no robot benchmark. About 8.05 s clips at 24 fps (193 frames, Sec. 3.3).
- Training compute stated (Sec. 4.5): "totaling approximately 43,000 GPU-days."
- Audio (input, output, or none mentioned): output, 48 kHz mono (Sec. 2.1), so not spatial.
- Code and weights: intro: "we release all model weights along with training, inference, and fine-tuning code." https://github.com/OpenMOSS/MOVA opened: Apache-2.0, weights on Hugging Face (MOVA-360p, MOVA-720p), LoRA fine-tuning documented (single-GPU and 8-GPU modes).

---

## Could not verify
- No paper was dropped for failing to open. Two page problems occurred and were worked around:
  - https://arxiv.org/html/2605.12090 returned 404; the survey was read from the arXiv PDF instead (text extracted locally).
  - https://github.com/Motus-Robotics/Motus (a URL I guessed, not taken from a page) returned 404, so Motus code and weights status is "not seen", not "not released".
- Not opened, only seen in search results or in survey lists, so they are not listed above: "World-Action Models for Robot Learning and Control: A Survey" (reported as arXiv 2609.16074; I did not open its arXiv page), GigaWorld-Policy, FlowWAM, Genie Envisioner (2508.05635), WALL-WM, VLA-JEPA, OmniAction, NAVA, JoVA, Wan2.2-S2V.
- The Cosmos Policy NVIDIA project URL redirects to https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/, which showed only a BibTeX block to the page reader. The release claim rests on the paper's abstract and the GitHub repository.

## Searches run
1. "world action model" video diffusion robot policy joint video and action prediction arXiv
2. video-action model unified video action diffusion robot policy pretrained video diffusion Wan LIBERO
3. joint audio-video generation model open-source diffusion transformer Ovi LTX-2 audio video
4. world model audio robot manipulation video prediction with sound action-conditioned audio-visual world model
5. Cosmos Policy video model robot policy; GigaWorld-Policy; LingBot-VA; Motus world action model arXiv
6. Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation arXiv
7. Motus unified latent action world model; LingBot-VA causal world modeling robot control video-action autoregressive; mimic-video arXiv
8. survey "world action models" embodied AI survey 2026 arXiv
9. vision-language-action model with audio input speech sound world model video prediction robot 2026
10. Audio-WM audio world model robot policy future Mel latents contact sound manipulation
11. LingBot-VA "Causal World Modeling for Robot Control" arXiv
12. mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs arXiv
13. joint audio-video diffusion Wan 2.2 audio branch UniVerse-1 OR JavisDiT OR "Universe-1" OR MM-Sonate audio video generation open weights

## What exists and what I did not find
The term "world action model" is used consistently for models that predict future observations and actions together. DreamZero (ye2026world) popularized the name, and the two 2026 surveys define it, with the broadest wording in Shen et al. (a forecast of the future is available to action) and Wang et al. (a joint distribution over future states and actions). The older name "video-action model" (mimic-video, UVA, DiT4DiT) is treated by Wang et al. as the video-only special case of a WAM. Most WAMs I read start from a pretrained video diffusion model. The Wan 2.2 5B family (Fast-WAM, Motus, LingBot-VA, tau0-WM) and Cosmos-Predict (Cosmos Policy, mimic-video, DiT4DiT) are the two common bases; DreamZero uses Wan 2.1 14B. Several have released code and weights (DreamZero, Fast-WAM, Cosmos Policy, DiT4DiT, LingBot-VA).

I did not find any world action model that takes audio as an input. Every one I read takes cameras, language and usually robot state; audio appears only as future work (UVA, LingBot-VA) or as a stated gap in the surveys (Wang et al. say future states are "almost universally" predicted in RGB; Shen et al. say acoustic latents "are rare"). The audio-using work I found is separate: Audio-WM (Zhang and Gienger) predicts future audio from one microphone for a manipulation policy but has no video, and the Audio-Visual World Model paper (Wang et al.) predicts binaural audio and frames under actions but only in a simulated navigation setting with a small diffusion transformer, not a pretrained video model and not a manipulator. I did not find a model that combines a pretrained video model, robot manipulation actions and spatial audio.

For building one, the open joint audio-video generators share a base with the WAMs. Ovi (Wan2.2 5B video branch plus an audio branch trained from scratch, no training scripts released) and MOVA (Wan2.2 I2V A14B video tower, 48 kHz mono audio, LoRA fine-tuning code) are the nearest to the Wan 2.2 WAMs; LTX-2 has stereo output, an audio-to-video pipeline and LoRA training code but is a different family. I did not see any of the three take multi-channel or direction-preserving audio as input, and I did not check how well any of them handles action conditioning. This survey is limited to what the web search returned on 2026-11-16, so a missing paper may exist; the reference lists of the two surveys (Shen et al., Wang et al.) are the best place to look for more WAMs.
