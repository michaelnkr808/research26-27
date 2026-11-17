# Literature survey D: world models and world action models for navigation (2026-11-16)

Topic: navigation world models and video-generation-based navigation policies (what base video model each starts from, inputs, actions, benchmarks, stated compute, code), whether the manipulation world action models in the reviewed file (`2026-11-16-world-action-models.md`) report navigation or mobile tasks, and whether any navigation world model takes or predicts audio.

How the search was done: web search to find candidates (queries listed at the end). Search results are leads only. Every paper below was then checked against the arXiv API (title, authors, date, identifier, comment field), and its arXiv HTML full text was downloaded as raw text with curl, stripped of tags (math replaced by its LaTeX alt text), and searched and read in the passages that matter. Code and weights claims come from the project page or the repository README and repository metadata read on 2026-11-16, not from search snippets. Status of everything below: `verified` means read on the page on 2026-11-16.

Caveats the user should know:
- How deep I read each paper is "raw HTML, keyword search plus targeted passages". I did not read every section of every paper. Where a fact is missing here, it means I did not see it in the passages read, not that the paper lacks it.
- Every arXiv paper is "preprint, not peer reviewed" unless a venue is noted. Where a venue is noted, it comes from the arXiv comment field or the authors' own project page or repository, and I did not check the conference's own site. Treat those venues as claimed, not confirmed.
- Numbers are only ones copied from the paper, with the section or table named. Success rates are left out on purpose; the papers' tables have them.
- Search found several papers that I did not open (listed under "Found but not opened"). They are not part of this survey's claims.
- Review round (PR #50 comment "Fact-check review: navigation world models"): every reviewer correction below was re-checked by me against the raw arXiv HTML, raw README, project page or repository metadata before editing. Six new entries were added (WNM-3D, NavGen, NavDreamer, WorldVLN in Part 1; UniWAM in Part 2; EchoWM in Part 3), each opened from raw arXiv HTML.

---

## Part 1: navigation world models and video-generation-based navigation policies

### bar2024navigation: Navigation World Models
- Authors: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun
- Year and venue: 2024 (v1 4 Dec 2024, v2 read); arXiv comment field says "CVPR 2025"; the project page and README also say CVPR 2025 (the README says Oral). Not checked on the CVPR site.
- Link: https://arxiv.org/abs/2412.03572
- Read depth: raw HTML v2, keyword search plus Sec. 3, 4.1 and the planning passages; not every section
- Part: 1
- What they do: A controllable video generator that predicts future egocentric frames from past frames and a navigation action (abstract: "a controllable video generation model that predicts future visual observations based on past observations and navigation actions"). The model is a Conditional Diffusion Transformer (CDiT). Planning scores simulated trajectories against a goal image.
- Built on: no pretrained video model found. Frames are encoded with a Stable Diffusion VAE tokenizer (Sec. 4.1, "We use the Stable Diffusion [4] VAE tokenizer, similar as in DiT"). The model is trained on robot navigation datasets, with unlabeled Ego4D video added in one experiment (Table 4).
- Inputs: past frames (context frames) in latent space, a navigation action, and for planning a goal image.
- Actions: a_i = (u, phi) with translation u in R^2 and a yaw rotation, extended to (u, phi, k) with a time shift k (Sec. 3). Planning: cross-entropy method (CEM) over candidate trajectories, formulated as model predictive control; the model can also rank trajectories sampled from NoMaD.
- Benchmarks and data: SCAND, TartanDrive, RECON, HuRoN for training; Go Stanford as an unknown environment; Ego4D unlabeled (Sec. 4.1). Offline evaluation on recorded datasets; no simulator and no closed-loop real robot seen in the passages read.
- Training compute stated: "XL sized model are trained on 8 H100 machines, each with 8 GPUs" (Sec. 4.1, implementation details). The default model is CDiT-XL with 1B parameters (same paragraph).
- Audio (input, output, or none mentioned): none mentioned (zero hits for audio, sound, acoustic or binaural in the full text).
- Code and weights: project page https://www.amirbar.net/nwm/ links code and models; repository https://github.com/facebookresearch/nwm opened: official PyTorch implementation with CDiT training code, models at https://huggingface.co/facebook/nwm. License: the GitHub API shows "NOASSERTION", but the README License section says "The code and model weights are licensed under Creative Commons Attribution-NonCommercial 4.0 International. See `LICENSE.txt` for details." So code and weights are CC BY-NC 4.0, non-commercial.

### chen2025astranav: AstraNav-World: World Model for Foresight Control and Consistency
- Authors: Jintao Chen, Junjun Hu, Haochen Bai, Minghua Luo, Xinda Xue, Botao Ren, Chengyu Bai, Shichao Xie, Ziyi Chen, Fei Liu, Zedong Chu, Xiaolong Wu, Mu Xu, Shanghang Zhang
- Year and venue: 2025 (v1 25 Dec 2025, v2 8 Apr 2026 read); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2512.21714
- Read depth: raw HTML v2, Sec. 3 (method) and 4.1, 4.2 in full, keyword search over the rest
- Part: 1
- What they do: A joint world-action model for embodied navigation. A VLM plans, a video generator predicts future frames, and a policy head predicts waypoints. Video and policy are trained together so predicted views and planned actions stay consistent. The video branch can be skipped at inference.
- Built on: Wan2.2-TI2V-5B. Sec. 3.1: "Our model is implemented using Qwen-2.5-VL-3B as the VLM backbone and Wan2.2-TI2V-5B as the foundation for the video generation and policy components." Sec. 4.2: the video generator is fine-tuned with LoRA; the VLM gets full-parameter fine-tuning. The VLM's embeddings replace the Wan text encoder via cross-attention (Sec. 3.3.2).
- Inputs: a language instruction, a history of front-view frames, and current left, front and right views (Sec. 3.3.3, 4.1).
- Actions: two policy heads. An "Action Former" (query transformer) outputs relative position displacement, a rotation-invariant heading, and a binary arrival flag; a diffusion policy head uses flow matching and can attend to the video latents (Sec. 3.4). Underlying actions are four discrete types (forward, left, right, stop) and continuous waypoints (Sec. 4.1). Fig. 2 caption: the model "predict[s] the next five visual frames while simultaneously outputting the corresponding 5-step waypoint sequence".
- Benchmarks and data: Habitat continuous VLN (R2R-CE, RxR-CE) on Matterport3D and open-vocabulary object navigation (HM3D-OVON) (Sec. 4.1, 4.3); zero-shot test on a physical robot with no real-world fine-tuning (Sec. 4.3.4).
- Training compute stated: "All experiments are conducted on a cluster equipped with 96 H20 GPUs." (Sec. 4.2, Implementation Details; the 96 sits in a math element in the HTML, which I confirmed in the raw HTML). No hours stated.
- Audio (input, output, or none mentioned): none mentioned (zero hits in the full text).
- Code and weights: project page https://astra-amap.github.io/AstraNav-World.github.io/ links code, model and demo. Repository https://github.com/amap-cvlab/AstraNav-World opened: Apache-2.0. README news has exactly two items: 2025-12-29 "We release the inference code of action former for the R2R, RxR, and OVON benchmarks." and 2026-01-21 "We release the inference checkpoints of action former." (ModelScope link, not opened). The README also has an "Install Training dependencies" step. The repository file tree (GitHub git trees API, 6,786 entries) has a `train_code/` directory that looks like a vendored ms-swift trainer for the VLM (it includes `train_code/data/waypoint_train_demo.json`, a demo of waypoint training data), and a `world/wan/` package (LoRA, action-former, future-frame and pipeline files for Wan2.2). So VLM training code is present. What I did not find: a training script for the Wan video generator, or any weight file for the fine-tuned video generator (no .safetensors, .pth, .bin or .ckpt in the tree). Whether the ModelScope action-former checkpoint also carries the video LoRA weights I could not check. Optional future-frame prediction needs a separate Wan2.2-TI2V-5B-Diffusers download.

### zhou2026uninav: UniNav: A Unified World-Action Diffusion Model for Visual Navigation
- Authors: Changqing Zhou, Yueru Luo, Zeyu Jiang, Changhao Chen
- Year and venue: 2026 (v1 4 Aug 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2608.03244
- Read depth: raw HTML v1, abstract, Sec. 1, 3, 4 (implementation details, datasets, Table 1 header); appendix not read
- Part: 1
- What they do: One diffusion transformer jointly denoises future egocentric video latents, waypoint tokens and camera-geometry tokens for image-goal navigation. Two variants: UniNav-Full (generates video and waypoints) and UniNav-Fast (drops the future-image tokens at inference). Video-only data without waypoint labels can be added.
- Built on: Wan2.1-1.3B. Sec. 1: "Built upon the pretrained Wan2.1-1.3B video diffusion model". Sec. 4: "We initialize UniNav from the pretrained Wan2.1-T2V-1.3B video diffusion transformer". The video VAE is frozen and the transformer is fine-tuned.
- Inputs: history frames and a goal image; camera tokens come from a frozen geometry foundation model.
- Actions: continuous local waypoint trajectory, each waypoint a relative translation and yaw change in the robot frame (Sec. 3.1, 3.2). Executed by predicting waypoints directly, with no external planner needed.
- Benchmarks and data: four real-robot image-goal datasets, RECON, SaCSoN, GO Stanford and SCAND (Sec. 4.1); metrics ATE and RPE. No simulator or closed-loop run seen in the passages read.
- Training compute stated: "Training is performed on four NVIDIA RTX 4090 GPUs for two epochs, with a batch size of 1 per GPU and gradient accumulation over 4 iterations" (Sec. 4, implementation details). Latency measured on a single RTX 4090.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: the abstract says "Code will be released." I found no repository link on the paper's page, so I could not check one.

### bagchi2026walk: Walk through Paintings: Egocentric World Models from Internet Priors (EgoWM)
- Authors: Anurag Bagchi, Zhipeng Bao, Homanga Bharadhwaj, Yu-Xiong Wang, Pavel Tokmakov, Martial Hebert
- Year and venue: 2026 (v1 21 Jan 2026, v2 read); the project page says "ECCV 2026"; the arXiv page lists no venue. Not checked on an ECCV site.
- Link: https://arxiv.org/abs/2601.15284
- Read depth: raw HTML v2, Sec. 1, 3, 4, 5, 6 via keyword search plus targeted passages; appendix only searched
- Part: 1
- What they do: A method that turns any pretrained video diffusion model into an action-conditioned world model by adding action embeddings through lightweight conditioning layers, for 3-DoF robot navigation and for a 25-DoF humanoid (navigation and manipulation). Planning uses CEM following NWM.
- Built on: three variants, SVD, Cosmos Predict2-2B and Wan2.1-14B (Experiments section: "three variants of our method built on top of the SVD, Cosmos (Predict2-2B) and Wan2.1-14B models"). Fine-tuned with the base model's own objective.
- Inputs: past frames plus per-frame actions; goal image for planning.
- Actions: 3-DoF navigation uses (delta x, delta y, delta phi); the humanoid uses a 25-DoF joint-space action (Sec. 3). Actions are projected and injected frame by frame, with 1D convolutions to match the temporal compression of Cosmos and Wan.
- Benchmarks and data: RECON, SCAND, Tartan Drive for navigation; 1X Humanoid dataset (Sec. 4). Metrics LPIPS, DreamSim and the authors' Structural Consistency Score; planning on RECON compared with NWM (Sec. 6, Table 4). Human assessment for manipulation success.
- Training compute stated: Experiments section: "NWM is trained on 64 H100 GPUs ... while our SVD and Cosmos variants use only 8 A100 GPUs". No compute stated for the Wan variant in the passage read.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: repository https://github.com/miccooper9/egowm opened. Its README TODO list marks the SVD navigation inference scripts (3-DoF and 25-DoF) "Done"; the Cosmos-2B train and infer scripts "Very Soon"; the SVD train and 25-DoF manipulation inference scripts "Very Soon"; the Wan2.1-14B train and infer scripts "Soon"; and the SCS metric scripts "Soon". So the Cosmos and Wan variants have no released scripts as of the README read (the README says fine-tuned checkpoints are at the Hugging Face page, which I did not open); weights page https://huggingface.co/anuragba/egowm/ is linked but I did not open it. No license shown in the repository metadata.

### zhang2025world: World-in-World: World Models in a Closed-Loop World
- Authors: Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal M. Patel, Paul Pu Liang, Daniel Khashabi, Cheng Peng, Rama Chellappa, Tianmin Shu, Alan Yuille, Yilun Du, Jieneng Chen
- Year and venue: 2025 (v1 20 Oct 2025, v2 read); arXiv comment field and the repository description say "ICLR 2026 Oral". Not checked on the ICLR site.
- Link: https://arxiv.org/abs/2510.18135
- Read depth: raw HTML v2, abstract, Sec. 2.1 to 2.4, 3 (table rows and conclusions), App. B and C (post-training configuration); not every appendix
- Part: 1
- What they do: A closed-loop benchmark that tests video world models by whether they help an embodied agent finish tasks, using a unified online planning loop and a unified action interface. It benchmarks off-the-shelf generators and post-trained (action-conditioned) versions.
- Built on: models evaluated include Wan2.1 (14B), Wan2.2 (5B and A14B), Cosmos-Predict2, LTX-Video, SVD and task-specific models (Table 1 and later tables). Post-trained variants start "from their official pretrained weights" (App. C.2).
- Inputs: panoramic observations for Habitat tasks, front-view for manipulation (App. C.2).
- Actions: Habitat tasks use four discrete primitives, "Forward 0.2 m, Turn_Left 22.5, Turn_Right 22.5, and Stop" (App. C.2); manipulation uses 7-DoF end-effector poses. The agent proposes actions, the world model imagines outcomes, and the agent picks among them.
- Benchmarks and data: Active Recognition, image-goal navigation and active embodied question answering in Habitat-Sim (HM3D, Matterport3D), plus manipulation in CoppeliaSim (Sec. 2.3, 2.4). Post-training on 40K sampled instances for one epoch (App. C.2).
- Training compute stated: not found in the passages read (only the instance count and one epoch).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: the abstract says "Code is available at github.com/World-In-World". Repository https://github.com/World-In-World/world-in-world opened: MIT license; README lists post-training and data collection instructions and a Hugging Face dataset link. Per-model repositories (Cosmos-Predict2, LTX-Video-Trainer, and others) are listed in App. Table 6.

### azuma2026navwam: NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation
- Authors: Daichi Azuma, Taiki Miyanishi, Koya Sakamoto, Shuhei Kurita, Yaonan Zhu, Petr Khrapchenkov, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo
- Year and venue: 2026 (v1 11 Jun 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2606.13494
- Read depth: raw HTML v1, abstract, Sec. 1, 3, 4 (backbone), App. B (training curriculum, datasets, robot) and C.1; not every section
- Part: 1
- What they do: Puts future egocentric views, a goal-progress value and an action chunk into one latent sequence of a video diffusion model, so one denoising pass produces the action with no external CEM planner. Following Cosmos Policy's latent-frame idea, for navigation.
- Built on: Cosmos Predict2 (a "2 B" model per App. C.1). Sec. 4: "We instantiate NavWAM with Cosmos Predict2". Three-phase curriculum: HM3D simulator pretrain, joint fine-tune on real-robot data, optional per-dataset fine-tune (App. B.2).
- Inputs: current egocentric frame, goal image, robot state (Sec. 3, 4).
- Actions: an action chunk of (delta x, delta y, delta psi) in the local frame, normalized per dataset (App. B.1), executed receding-horizon through ROS 2 cmd_vel on a wheeled robot (App. B.5).
- Benchmarks and data: GO Stanford (30-episode held-out split), a held-out "sit" image-goal protocol, training on HM3D simulator trajectories and the NWM-style RECON, SACSoN, SCAND splits (App. B.3); closed-loop real-robot runs on a Direct Drive Tech Diablo platform (App. B.5).
- Training compute stated: the supplementary training-hyperparameter table lists "Per-GPU batch size 8", "GPUs per training run 4 (RTX PRO 6000)" and "Effective batch 32". No hours. Inference cost is measured on a single RTX PRO 6000 (96 GB) (App. C.1).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: project page https://dachii-azm.github.io/navwam/ opened; it shows "Code (coming soon)". No weights seen. (The reviewer also found that the repository dachii-azm/navwam holds only the static project site.)

### chen2026imaginav: ImagiNav: Scalable Embodied Navigation via Generative Visual Prediction and Inverse Dynamics
- Authors: Jie Chen, Yuxin Cai, Yizhuo Wang, Ruofei Bai, Yuhong Cao, Jun Li, Wei-Yun Yau, Guillaume Sartoretti
- Year and venue: 2026 (v1 14 Mar 2026, v2 read); the project page says "IROS 2026". Not checked on an IROS site.
- Link: https://arxiv.org/abs/2603.13833
- Read depth: raw HTML v2, abstract, Sec. I, III, IV and the setup passages; results tables not read
- Part: 1
- What they do: Navigation in visual space. A VLM splits the instruction into subgoals, a fine-tuned video generator imagines the future egocentric video, and a geometric inverse dynamics step turns the video into waypoints. Trained on in-the-wild human walking videos with no robot data.
- Built on: LTX-Video 2B (not Wan or Cosmos). Sec. III: "pre-trained LTX-2B architecture"; the experiments section says it is fine-tuned with LoRA. VGGT is the inverse dynamics model; a VLM does subgoal decomposition.
- Inputs: current egocentric frame plus text subgoal.
- Actions: the video is decoded into relative waypoints (delta x, delta y, delta theta) by VGGT, then a heuristic proportional controller maps them to velocity commands (Sec. III-C and the implementation passage).
- Benchmarks and data: VLN-PE in NVIDIA Isaac Sim, 100 episodes from the InteriorNav dataset (experiments section); a pilot-scale hand-held iPhone video dataset (Sec. IV).
- Training compute stated: not found in the passages read.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: project page https://j1dan.github.io/ImagiNav lists code and a dataset; repository https://github.com/J1dan/ImagiNav-release opened: MIT license, README says it contains the pipeline, training configurations, and "checkpoint and dataset instructions"; dataset at https://huggingface.co/datasets/J1dan/imaginav-dataset (linked, not opened). The Hugging Face API file list for https://huggingface.co/J1dan/imaginav shows released weights: `LTX-2B-AC-MoE-LoRA/*` (LoRA files), `LTX-13B-LoRA/lora-weights-13b-85epoch.safetensors` and `LTX-2B-Full/model-weights-85epoch.safetensors`. So a 13B LTX LoRA is also released; the paper passages I read describe the 2B model only.

### liu2026imagineuav: ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning
- Authors: Xuchen Liu, Jiawei Huang, Shihao Xia, Bingxi Liu, Jinqiang Cui, Jiankun Yang
- Year and venue: 2026 (v1 published 31 May 2026, v2 updated 8 Jun 2026 per the API; v2 read); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2606.01205
- Read depth: raw HTML v2, Sec. I to IV (implementation) in the passages that matter; results not read
- Part: 1
- What they do: A drone (UAV) navigation pipeline in three steps: an instruction-conditioned video world model imagines the next egocentric rollout, a learned action extractor reads relative motion from the imagined frames, and a kinodynamic planner smooths the path. The imagination module is step-distilled for speed. The paper calls this a world action model for aerial vision-language navigation.
- Built on: Wan2.1. Sec. 4: "The backbone of our imagination module is Wan2.1-Fun-V1.1-1.3B-Control-Camera", with camera-control latents masked during training and testing so motion comes from language.
- Inputs: current observation and a language instruction.
- Actions: relative motion sequences, 6-DoF, extracted from the imagined video by an action extractor trained from scratch on video and trajectory pairs, then refined by the planner (Sec. III-D).
- Benchmarks and data: UAV-Flow benchmark and dataset; UE5 simulation plus real flights on a custom FPV quadrotor (Sec. IV).
- Training compute stated: "Training is conducted on 8 x NVIDIA PRO 6000 (96GB) GPUs" and "The model is trained for 2 epochs" at 256x256 resolution (Sec. IV, imagination module).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: no code or weights link in the paper text; the only links are a video demo and a hardware page.

### serpiva2026dreamtonav: DreamToNav: Generalizable Navigation for Robots via Generative Video Planning
- Authors: Valerii Serpiva, Jeffrin Sam, Chidera Simon, Hajira Amjad, Iana Zhura, Artem Lykov, Dzmitry Tsetserukou
- Year and venue: 2026 (v1 6 Mar 2026); arXiv comment: "Submitted to conference"; preprint, not peer reviewed
- Link: https://arxiv.org/abs/2603.06190
- Read depth: raw HTML v1, Sec. I to III and the experiment setup passages; results not read in detail
- Part: 1
- What they do: A user gives an image and a text command; a VLM rewrites the prompt, NVIDIA Cosmos 2.5 generates a video of the robot doing the task (plus a synthetic third-person view), and a detector plus tracking extracts a trajectory the robot follows. Tested on a wheeled robot and a quadruped.
- Built on: NVIDIA Cosmos 2.5 (Sec. III-B calls it "a world foundation model pre-trained on large-scale physical interaction data"). The paper does not say whether Cosmos is fine-tuned; no fine-tuning of Cosmos is described in the passages I read. (The phrase "without requiring any task-specific fine-tuning" in Sec. III-A refers to the Qwen 2.5-VL prompt-refinement step, not to Cosmos; I had misattributed it in the first version.) A small YOLO11n detector is trained on a robot dataset (Sec. III-C).
- Inputs: one image of the scene with the robot plus a text prompt.
- Actions: waypoints extracted from the generated video by detection (not an action head); executed by the robot's controller.
- Benchmarks and data: real-robot experiments on a wheeled UGV and a quadruped in a cluttered indoor lab; no public benchmark seen.
- Training compute stated: none stated.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: no code or weights link in the paper text.

### sam2026action: Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion
- Authors: Jeffrin Sam, Nguyen Khang, Yara Mahmoud, Miguel Altamirano Cabrera, Dzmitry Tsetserukou
- Year and venue: 2026 (v1 2 May 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2605.01477
- Read depth: raw HTML v1, abstract, Sec. I to V via keyword search plus the setup passages
- Part: 1
- What they do: Stage I is an LLM-orchestrated loop that picks a text-to-video model, refines prompts, and scores candidate first-person navigation videos. Stage II is a small policy (FlowDiT) that turns the validated video and the instruction into velocity commands.
- Built on: the Stage I model pool is WAN 2.2 or LTX-Video (Algorithm 1 lists the model pool as WAN and LTX; the Stage I description names "a text-to-video diffusion generator" with WAN 2.2 or LTX-Video as the options), used as generators; Cosmos-Reason1 is the validator. FlowDiT is a separate 43M-parameter model trained on RECON then fine-tuned on Unitree G1 simulation episodes.
- Inputs: language instruction, initial image, the validated reference video, optional live camera frame.
- Actions: continuous velocity commands (v_x, v_y, omega), action-space denoising diffusion, closed loop at 40 to 47 Hz (abstract-level description in Sec. I).
- Benchmarks and data: pretraining on RECON (11,830 outdoor episodes); fine-tuning on 203 Unitree G1 humanoid episodes in Isaac Sim (162 train, 41 validation); evaluated on a wheeled robot and a drone in Isaac Sim and open-loop on a real Unitree G1.
- Training compute stated: the FlowDiT training-hyperparameter table (Sec. IV) lists "Hardware NVIDIA RTX 5090 (32 GB)", "Batch Size 8 (FP16)" and about 43M trainable parameters. No hours. Inference timing is also on an RTX 5090.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: no code or weights link in the paper text.

### serpiva2025flight: FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video
- Authors: Valerii Serpiva, Artem Lykov, Faryal Batool, Vladislav Kozlovskiy, Miguel Altamirano Cabrera, Dzmitry Tsetserukou
- Year and venue: 2025 (v1 17 Sep 2025, v2 read); arXiv comment: "Submitted to conference"; preprint, not peer reviewed
- Link: https://arxiv.org/abs/2509.14082
- Read depth: raw HTML v2, Sec. I to V via keyword search plus targeted passages; evaluation not read in detail
- Part: 1
- What they do: A VLM turns a single drone camera frame and a task into a mission command, a pretrained image-to-video model generates a first-person flight video, and ORB-SLAM3 visual odometry extracts the trajectory and poses as action data to train a drone control policy.
- Built on: Wan 2.2 image-to-video, used off the shelf. The paper says "we selected the Wan 2.2 I2V Fast model" and cites a hosted (Replicate) endpoint for it. No training or fine-tuning of the video model seen. Gemini 2.5 Flash is the VLM.
- Inputs: one RGB image and a text mission.
- Actions: not a learned action head; pose and velocity are recovered from the generated video with ORB-SLAM3 (Sec. V).
- Benchmarks and data: simulated and real drone maneuver tests (Sec. VI-B, "Simulation vs. Real-World Maneuver Evaluation"); no public benchmark seen.
- Training compute stated: none stated.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: no code or weights link in the paper text.

### dong2025towards: Towards Unified World Models for Visual Navigation via Memory-Augmented Planning and Foresight (UniWM)
- Authors: Yifei Dong, Fengyi Wu, Guangyu Chen, Lingdong Kong, Qiyu Hu, Yuxuan Zhou, Xu Zhu, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G. Hauptmann, Zhi-Qi Cheng
- Year and venue: 2025 (v1 9 Oct 2025, v4 read); arXiv comment: "Accepted to ECCV 2026"; the repository title also says ECCV 2026. Not checked on an ECCV site.
- Link: https://arxiv.org/abs/2510.08713
- Read depth: raw HTML v4, Sec. 1, 3 (method), 4 (setup, implementation details); results tables not read
- Part: 1
- What they do: One multimodal autoregressive model does both navigation planning and next-observation imagination, with a hierarchical memory bank of cached attention keys and values. It is a token-based world model, not a diffusion video model.
- Built on: GAIR Anole-7B (built on Chameleon), "fine-tuned from GAIR Anole-7B ... all tokenizers frozen" with LoRA rank 16 (Sec. 4, implementation details). Not Wan, Cosmos or any video diffusion model.
- Inputs: start, goal and current observation images (as VQ tokens) with text tokens.
- Actions: (x, y, phi) per step, discretized into bins and predicted as classification tokens (Sec. 3.2).
- Benchmarks and data: Go Stanford, ReCon, SCAND, HuRoN for training and in-domain tests, TartanDrive zero-shot, 1X Humanoid navigation episodes (Sec. 4); metrics SR, ATE, RPE and image quality metrics.
- Training compute stated: "Training uses AdamW for 20 epochs (lr 2e-4, batch size 8) on 4 x A100-80GB GPUs" (Sec. 4, implementation details).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: repository https://github.com/UWMILab/UniWM opened (the paper's date line links it): official implementation; the GitHub API license field is empty but the README shows a "License-MIT" badge. The README links a dataset at https://huggingface.co/datasets/fly1113/UniWM_Dataset and describes training (`train.sh`) and evaluation (`eval.sh`) scripts. I did not check for released model weights.

### huang2026wnm: WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN
- Authors: Yuehao Huang, Yunzi Wu, Xiaotao Zhang, Xinhai Li, Jiankun Dong, Sunyao Zhou, Jiajun Lv, Chi Zhang, Chenjia Bai, Yong Liu, Xuelong Li
- Year and venue: 2026 (first published 7 Aug 2026; v3 read); arXiv preprint, not peer reviewed (no comment field)
- Link: https://arxiv.org/abs/2608.07267
- Read depth: raw HTML v3, abstract, Sec. 1 to 3 in the passages that matter, Sec. 4 setup and Tables 5 and training-hyperparameter appendix via keyword search; result tables not read
- Part: 1
- What they do: A generative world-action model for continuous vision-language navigation. A shared world-action diffusion transformer jointly predicts future egocentric views and aligned navigation actions. A frozen geometry encoder (VGGT-Omega) turns the RGB history into a fixed-length 3D scene prefix through a trainable "3D Scene-to-Token Adapter". WNM-2D is the same model without the 3D prefix. The joint video-action flow design comes from DreamZero (Sec. 3: "builds on the joint video-action flow backbone of DreamZero").
- Built on: Wan2.2-TI2V-5B. Experiment setup: "Both variants initialize from Wan2.2-TI2V-5B, use K=33 history frames, and predict B=4 blocks with N_a=8 actions each." Table 5 (architecture configuration) repeats "World-action initialization Wan2.2-TI2V-5B". How it is tuned: the experiment setup says "VGGT-Omega remains frozen while the Adapter and world-action backbone are optimized", and the video VAE is frozen (Sec. 3.3). The words LoRA and low-rank do not appear in the paper text, so I read this as full training of the backbone, but the paper does not say "full fine-tuning" in so many words. Training has three stages: offline A* supervised fine-tuning, DAgger-style supervised fine-tuning on policy-visited states, and Counterfactual DanceGRPO reinforcement learning.
- Inputs: a language instruction and K=33 monocular egocentric RGB history frames (RGB only at inference; Sec. 3.1); world-action input resolution 160x320 (Table 5).
- Actions: planar local-frame increments (delta x, delta y, delta psi) in blocks of N_a=8 actions, one action block per predicted visual block (Sec. 3.1).
- Benchmarks and data: GN-Bench (InteriorGS 3D Gaussian-splat scenes; 16K A* demonstrations for Stage I) and VLN-CE R2R Val-Unseen (R2R and RxR training splits); closed-loop deployment on a physical robot with a real-world-adapted checkpoint (Sec. 4).
- Training compute stated: the training-hyperparameter appendix table lists AdamW, learning rate 1e-5, 20 epochs for Stage I and 5 for Stage II, global batch 256 for WNM-2D and 192 for WNM-3D, and "GPUs 16 H100". No hours.
- Audio (input, output, or none mentioned): none mentioned (zero hits in the full text).
- Code and weights: repository https://github.com/TeleHuman/WNM-3D opened (default branch master; the paper header links it): Apache-2.0 for the original code. Releases: inference server scripts (`scripts/inference/wnm_3d_server.sh`), training scripts (`scripts/train/wnm_3d_stage1.sh`, `wnm_3d_stage2.sh` and distributed variants, plus `docs/TRAINING.md`) for Stage I and II, and Stage III reinforcement learning lives in a separate repository, https://github.com/TeleHuman/WNM-3D-RL (linked, not opened). Checkpoints: Hugging Face https://huggingface.co/TeleEmbodied/WNM-3D (opened) lists `wnm_3d_stage1_release`, `wnm_3d_stage2_release` and `wnm_3d_stage3_release`, each a self-contained inference checkpoint (about 25.25 GiB each by the Hugging Face API file sizes; the stage 3 one is the main policy). License caveat: the README says "WNM-3D as a complete distribution is not licensed solely under Apache-2.0", and the Hugging Face page lists the FAIR Noncommercial Research License and the DINOv3 license because the checkpoints contain VGGT-Omega and DINOv3 derived parameters. So the released weights are non-commercial research use.

### huang2026navgen: NavGen: Visual Generative Models as a Scalable Data Engine for Embodied 3D Navigation
- Authors: Xijie Huang, Yongyang Wan, Chengbin Dong, Zimo Ding, Mo Zhu, Yijin Wang, Zhiyang Liu, Fei Gao, Yuze Wu, Xin Zhou
- Year and venue: 2026 (v1 25 Sep 2026); arXiv preprint, not peer reviewed (comment: "8 pages,9 figures")
- Link: https://arxiv.org/abs/2609.30770
- Read depth: raw HTML v1, abstract, Sec. 1, III (data pipeline, supervised fine-tuning), IV (datasets, deployment) via keyword search plus targeted passages
- Part: 1 (a data engine for drone navigation plus a Wan2.2-5B world-action pipeline trained on it)
- What they do: Generate 400K navigation video episodes with video models, then train a Wan2.2 5B video model on that data and compare it with models trained on other navigation datasets. The video model plus a waypoint-decoding step forms a "world action model" for drones.
- Built on: for data generation, the "step-distilled Wan2.2 14B T2V" model ("the open-source, low-cost Wan2.2 T2V 14B", 300K episodes), a more powerful closed-source model for 10K long-tail samples expanded to 100K, and LTX-2.3 ICLoraPipeline for style variation. For the navigation model: "we adopt the video-prediction and action-decoding pipeline of NavDreamer with the lightweight Wan2.2 5B model as the base model" (Sec. III-D). Tuning of the deployed model is LoRA: the training table (the real-world deployment pipeline's three stages) lists "LoRA rank 64" and "LoRA alpha 64" for all three stages (SFT, sCM, DMD2 distillation), global batch 32, learning rates 1e-4, 1e-5 and 2e-5, and 12,500, 2,500 and 5,000 steps.
- Inputs: an RGB image plus a text instruction (rewritten by a VLM); a history frame buffer in deployment (Sec. IV).
- Actions: the generated video is decoded into waypoints with Pi3 camera estimation and metric scale correction (Sec. IV), then executed with Ego-Planner on a drone. No learned action head.
- Benchmarks and data: 400K generated episodes; a real-world navigation benchmark with 1,000 text-image pairs; real drone deployment (Sec. IV).
- Training compute stated: "16 NVIDIA A100" for each of the three stages (training table). Data generation: about 36 seconds per sample on an A100 for the 14B step-distilled model.
- Audio (input, output, or none mentioned): none mentioned. The only keyword hit is a reference title for LTX-2 ("joint audio-visual foundation model"), and LTX-2.3 is used only for video style conversion.
- Code and weights: the paper says "Our dataset, model weights, and code will be made publicly available upon acceptance." I found no repository.

### huang2026navdreamer: NavDreamer: Video Models as Zero-Shot 3D Navigators
- Authors: Xijie Huang, Weiqi Gai, Tianyue Wu, Congyu Wang, Zhiyang Liu, Xin Zhou, Yuze Wu, Fei Gao
- Year and venue: 2026 (v1 10 Feb 2026, v2 21 Sep 2026; v2 read); arXiv preprint, not peer reviewed (comment: "Work in the progress. 22 pages, 15 figures")
- Link: https://arxiv.org/abs/2602.09765
- Read depth: raw HTML v2, abstract, Sec. 1 to 3 and the benchmark and real-world passages via keyword search; tables not read
- Part: 1
- What they do: A zero-shot drone navigation pipeline. A video model generates candidate navigation videos from an image and an instruction, Qwen3-VL picks the best, an inverse dynamics step (Pi3 plus a metric-depth scale correction) turns it into waypoints, and Ego-Planner tracks them. They also build a benchmark of 15 tasks over five dimensions and compare video models.
- Built on: no training of a video model; it uses off-the-shelf generators. The benchmark table lists Wan 2.2 among the models compared, and the real-world deployment sends the image and prompt "to Wan 2.6 video model via an API". So Wan 2.2 is only a benchmarked model here, and the deployed one is Wan 2.6.
- Inputs: one image and a language instruction.
- Actions: waypoints decoded from the generated video; no action head.
- Benchmarks and data: the authors' 3D navigation benchmark (object navigation, precise navigation, spatial grounding, language control, scene reasoning); real drone flights.
- Training compute stated: none (no training).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: the project page https://xinjiu612.github.io/NavDreamer was opened and links only the arXiv paper; I saw no code or weights link in the paper text.

### zhao2026worldvln: WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation
- Authors: Baining Zhao, Jiacheng Xu, Weicheng Feng, Xin Zhang, Zhaolu Wang, Haoyang Wang, Shilong Ji, Ziyou Wang, Jianjie Fang, Zhiheng Zheng, Weichen Zhang, Yu Shang, Wei Wu, Chen Gao, Xinlei Chen, Yong Li
- Year and venue: 2026 (v1 15 May 2026); arXiv preprint, not peer reviewed (a NeurIPS 2026 poster page appeared in search results; I did not open it)
- Link: https://arxiv.org/abs/2605.15964
- Read depth: raw HTML v1, abstract, Sec. 1 to 4, implementation passage via keyword search; appendix not read
- Part: 1
- What they do: An autoregressive world action model for drone vision-language navigation. A pretrained latent autoregressive video transformer predicts short-horizon latent world transitions, an action decoder turns them into waypoints, and real observations are fed back into the context. Stage 1 is supervised training; Stage 2 is "Action-aware GRPO" reinforcement learning with online rollouts.
- Built on: not Wan 2.2. Sec. 5 (Experiments, opening paragraph): "WorldVLN uses InfinityStar as the latent autoregressive backbone, with the action decoder initialized from Wan VAE and TSformer-VO-style priors." (InfinityStar-8B appears in the licensing note.) Only the Wan VAE is borrowed.
- Inputs: instruction text and egocentric observation history.
- Actions: relative waypoint displacements decoded from predicted latents (Sec. 3).
- Benchmarks and data: UAV-Flow-Sim and IndoorUAV-VLA; zero-shot real drone test (Sec. 1, 4).
- Training compute stated: "Training is conducted on 8 NVIDIA A800 80GB GPUs".
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: released. The arXiv abstract ends "Demos and code are available at https://embodiedcity.github.io/WorldVLN/" (my first version said no link existed; that was wrong, I had searched the body text and not the abstract). The code repository https://github.com/EmbodiedCity/WorldVLN.code (raw README read): "reference implementation" with autoregressive inference (an online inference service and entry points) and the two-stage training pipelines, supervised backbone plus action-decoder training (`train/scripts/train_from_base.sh`, `train/TRAINING.md`) and action-aware GRPO. Weights are listed in the README at https://huggingface.co/EmbodiedCity/WorldVLN; the Hugging Face API shows the model public (not gated), about 73 GB, with sharded `.safetensors` backbone files, a `vae/model.safetensors`, and `WorldVLN_action_decoder.pt` for both the IndoorUAV and UAV-Flow settings. License: the repository's LICENSE file is Creative Commons Attribution 4.0 (the GitHub API license field could not be re-read because of the unauthenticated rate limit); I saw no separate weight-license statement in the README or the Hugging Face metadata.

---

## Part 2: manipulation world action models, and whether they report navigation or mobile tasks

I searched the full text of ten manipulation WAMs from the reviewed file (Fast-WAM, Motus, LingBot-VA, tau0-WM, DreamZero, Cosmos Policy, DiT4DiT, UVA, mimic-video, UWM) for navigat*, mobile, egocentric, locomotion, humanoid, legged, wheeled, driving, drone and UAV, and read the hits. A recount of the word "navigat" (case-insensitive) in the downloaded full texts gives 0 for Fast-WAM, Motus, LingBot-VA, tau0-WM, DiT4DiT, UVA, UWM and Cosmos Policy, 1 for mimic-video (a cited reference title, "One Policy for Manipulation, Navigation, Locomotion and Aviation") and 1 for DreamZero (the data-description sentence quoted below). The two with something relevant are below, followed by ABot-M0.5 and UniWAM, which are WAMs built for mobile manipulation.

### kim2026cosmos: Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning
- Authors: Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, Jinwei Gu
- Year and venue: 2026 (v1 22 Jan 2026); ICLR 2026 poster per the reviewed file (listed there from the ICLR site); arXiv page lists no venue
- Link: https://arxiv.org/abs/2601.16163
- Read depth: raw HTML, keyword search for navigation, mobile and locomotion terms plus the limitations passage; the rest was read earlier in the reviewed file, not re-read here
- Part: 2
- What they do: Fine-tunes Cosmos-Predict2-2B so one model outputs actions, future images and values as latent frames (details in the reviewed file).
- Built on: Cosmos-Predict2-2B (from the reviewed file).
- Inputs: see the reviewed file.
- Actions: manipulation action chunks.
- Benchmarks and data: LIBERO, RoboCasa, real ALOHA (from the reviewed file). No navigation task seen in the text.
- Training compute stated: see the reviewed file.
- Audio (input, output, or none mentioned): none mentioned.
- Navigation or locomotion statement: the only mention is a limitation, "such as difficulty in adapting the method to dynamic manipulation tasks or locomotion tasks" (limitations paragraph, in the context of slow inference). It does not say the method extends to navigation, and no navigation experiment appears.
- Code and weights: see the reviewed file (github.com/NVlabs/cosmos-policy).

### ye2026world: World Action Models are Zero-shot Policies (DreamZero)
- Authors: Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, Joel Jang
- Year and venue: 2026 (v1 17 Feb 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2602.15922
- Read depth: raw HTML, keyword search for navigation, mobile, locomotion and egocentric terms plus the Discussion and Future Work section (Sec. 6) in full; the rest was read earlier in the reviewed file, not re-read here
- Part: 2
- What they do: 14B autoregressive video-diffusion WAM built on Wan2.1-I2V-14B-480P (from the reviewed file).
- Built on: Wan2.1 (from the reviewed file).
- Inputs and actions: manipulation (see the reviewed file).
- Benchmarks and data: one of the two robots is "the AgiBot G1 mobile bimanual manipulator" (Sec. 5); about 500 hours of teleoperation data across 22 environments such as homes, restaurants and offices. The data description names navigation as a skill: "The skill distribution reflects real-world deployment requirements: navigation enables movement between workspaces, while torso adjustments allow interaction with objects at varying heights (shelves, cabinets)." That is the only navigation hit in the full text (case-insensitive count of 1, rechecked). It is a data description, not a navigation benchmark, and I found no navigation result. (My first version wrongly said keyword search found nothing; my earlier search output was cut off at the line width.)
- Training compute stated: see the reviewed file.
- Audio (input, output, or none mentioned): none mentioned.
- Navigation or locomotion statement: besides the data-description sentence above, Sec. 6 lists future work (scaling laws, in-the-wild human egocentric data, faster inference, long-horizon reasoning, high-precision tasks, embodiment design). It does not name navigation or locomotion. Egocentric human video is discussed only as manipulation-related training data.
- Code and weights: see the reviewed file (github.com/dreamzero0/dreamzero).

### chen2026abot: ABot-M0.5: Unified Mobility-and-Manipulation World Action Model
- Authors: Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, Tianlun Li, Mingxin Wang, Dekang Qi, Bin Hu, Wei Mei, Yuze Xuan, Haolong Yang, Yanqing Zhu, Mu Xu, Zhiheng Ma, Xinyuan Chang
- Year and venue: 2026 (v1 1 Jul 2026, v2 read); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2607.00678
- Read depth: raw HTML v2, abstract, Sec. 1, 2, the architecture and pretraining passages and keyword search; the experiments sections were not read in detail
- Part: 2 (a WAM for mobile manipulation, which includes base movement and a navigation phase)
- What they do: A WAM for tasks that combine moving the robot base and manipulating objects. It adds frame-level latent actions between video prediction and robot actions, a mixture-of-transformers that separates base-movement actions from arm actions, and "Dream Forcing" so action prediction is trained on the model's own predicted video.
- Built on: Wan2.2. Architecture section: "ABot-M0.5 is a video-action World Action Model built upon the Wan2.2 video diffusion backbone". Training section: "Initialized from the pretrained Wan2.2 5B weights", first trained as an action-unconditioned autoregressive video predictor with full-parameter fine-tuning.
- Inputs: language instruction and multi-view camera observations.
- Actions: executable robot actions, with the action space split into mobility (base) and manipulation (arm) subspaces (Sec. 2.2 and the architecture section). The Fig. 1 caption says real-world rollouts have navigation and manipulation phases.
- Benchmarks and data: RoboCasa365 (mobile manipulation), RoboTwin 2.0, LIBERO and LIBERO-Plus, and real-world long-horizon mobile manipulation (experiments). Pretraining data includes OXE.
- Training compute stated: not found in the passages read.
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: repository https://github.com/amap-cvlab/ABot-Manipulation opened: Apache-2.0; README news (2026-07-21) says the inference code, pre-trained weights and evaluation scripts for ABot-M0.5 are released; weights at https://huggingface.co/acvlab. The Hugging Face API model list for that organization shows `acvlab/ABot-M0.5-RoboCasa365` (and an `ABot-M0.6-RoboCasa365`); I did not see a general pretrained checkpoint, so the released weight is the RoboCasa365 one.

### xue2026uniwam: UniWAM Technical Report: Unified Mobile Manipulation via Mixed-Stream World-Action Modeling and Manipulation Anchor Pose Supervision
- Authors: Wei Xue, Keliang Liu, Mingzhang Cui, Jinhua Xie, Jinjie Wei, Jianan Hou, Jingcheng Lu, Lintao Wang, Kaixiang Qiu, Yizhou Liu, Xinghai Ye, Jinghang Han, Mingcheng Li, Jie Gu, Shunli Wang, Lihua Zhang, Dingkang Yang
- Year and venue: 2026 (v1 30 Sep 2026); arXiv preprint, not peer reviewed (comment: "UniWAM Technical Report")
- Link: https://arxiv.org/abs/2609.39388
- Read depth: raw HTML v1, abstract, Sec. 1 to 3 (method, implementation) and the experiment setup passages; result tables not read
- Part: 2 (mobile manipulation with a separate navigation stream; navigation-only evaluation on MAP-Bench)
- What they do: A world-action model with two independently sampled streams, navigation and manipulation, over a shared video DiT and action DiT. Navigation is supervised toward a "Manipulation Anchor Pose" (MAP), the base position and heading from which manipulation should start. They build MAP-Data (over 1.5M episodes) and a benchmark, MAP-Bench.
- Built on: Wan2.2-TI2V-5B. Sec. 3.2: "Both streams share a video DiT, initialized from Wan2.2-TI2V-5B, and an action DiT"; the experiment section: "Both variants start from Wan2.2-TI2V-5B without embodied pretraining." Tuning: "The video VAE and the text encoder stay frozen, and all other parameters are trained", so this is full training of the DiTs, not LoRA (Sec. 3.2 implementation, "Network and optimization"). The action DiT has 30 layers at hidden width 1024 (the video DiT is 3072); AdamW, learning rate 1e-5, cosine decay; inference uses ten Euler steps and no classifier-free guidance.
- Inputs: language instruction plus stream-specific camera views (a wide head camera for navigation, a downward head camera and wrist cameras for manipulation); MAP-Bench uses only the forward RGB camera.
- Actions: navigation predicts planar pose increments (delta x, delta y, delta theta) relative to the base pose at the start of the chunk (not velocities); manipulation has its own action encoder and head.
- Benchmarks and data: MAP-Bench (300 targets in 109 scenes, 35,454 training trajectories, 15K evaluation trajectories; Sec. 3.1) and 24 real-robot tasks on four embodiments (AgileX Cobot Magic, AgiBot G2, dual-arm Franka Research 3, an AgileX single-arm platform). Baselines include pi0.5, GR00T N1.7, Uni-NaVid, InternVLA-N1 and FastWAM.
- Training compute stated: "All models, including UniWAM, are trained on the MAP-Bench training split for 40,000 steps on 16 NVIDIA H200 GPUs." Real-robot settings have a separate table of steps and GPUs per model (not copied here).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: repository https://github.com/Fysics-AI/UniWAM opened: Apache-2.0 for original code (the end-effector tracking folder is MIT; the README License section defers to `docs/LICENSES.md`, which says "UniWAM-Base / UniWAM-Scaled: Apache-2.0 for the released UniWAM weights" with upstream Wan and FastWAM attribution preserved). The README release checklist ticks the paper and site, "UniWAM-Base and UniWAM-Scaled checkpoints for MAP-Bench", "UniWAM architecture, training and inference code" (`training/README.md`, `training/NAVIGATION.md`, with `training/scripts` and `training/src` in the tree), end-effector track label generation, MAP-Bench training and evaluation data, baseline rollouts, and the MAP-Data subset. Checkpoints are at https://huggingface.co/Fysics-AI/UniWAM under `weights/base` and `weights/scaled`; the README says these MAP-Bench checkpoints use "a single navigation expert and one forward RGB view" because MAP-Bench is navigation only, so they are not the full mobile-manipulation model. Data at https://huggingface.co/datasets/Fysics-AI/MAP-Bench and https://huggingface.co/datasets/Fysics-AI/MAP-Data. The Hugging Face model and the MAP-Data dataset are gated (API `gated: auto`, model card license "other" named uniwam-and-upstream-model-licenses), so I could not read the card text; I read only API metadata. `docs/LICENSES.md` says the public MAP-Data subset has 100,052 trajectories (507.52 hours) with HM3D-derived terms in the data card, so it is a subset of the 1.5M-episode MAP-Data.

---

## Part 3: navigation world models with audio

### wang2025audio: Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics
- Authors: Jiahua Wang, Leqi Zheng, Jialong Wu, Yaoxin Mao, Shijie Cheng
- Year and venue: 2025 (v1 30 Nov 2025; v4 of 4 Aug 2026 read; v1 was titled differently, per the arXiv page); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2512.00883
- Read depth: raw HTML v4, abstract, Sec. 1 to 5 and the supplementary materials (encoder configuration, training details, baselines) via targeted passages; the reviewed file also covers this paper
- Part: 3
- What they do: Defines audio-visual world models for embodied agents and proposes AV-CDiT, a conditional diffusion transformer that predicts both the next egocentric image and the next binaural audio segment, given a motion action. A reward-aware lookahead planner uses it to improve a pretrained audio-visual navigation agent.
- Built on: not a large video model. The model is initialized from a pretrained Navigation World Models CDiT "of base size", fine-tuned first on the visual part of AVW-4k, then extended with modality experts (supplementary, "Training Details"). Stable Diffusion VAE for images; a SoundStream tokenizer trained on AVW-4k for audio.
- Inputs: egocentric RGB frames (128x128) and 0.15 s binaural audio segments at 16 kHz; relative motion actions (dataset section).
- Actions: four discrete actions: forward 0.15 m, turn left 10 degrees, turn right 10 degrees, stop (dataset section). The planner samples candidate actions from an existing agent, rolls each through the model for k steps, and executes the first action of the best beam (navigation experiment).
- Benchmarks and data: AVW-4k, built in SoundSpaces 2.0 on Matterport3D: "approximately 30 hours" of audio-visual data across 76 indoor scenes and 4,500 trajectories. Downstream: continuous audio-visual navigation in SoundSpaces 2.0 where an episode succeeds if the agent issues stop within 1 m of the sound source. Simulation only.
- Training compute stated: "all experiments are conducted on eight NVIDIA A100 GPUs" (experiment setup); supplementary lists total batch sizes of 512, 768 and 128 for the three stages. No GPU-hours.
- Model size: not stated as a parameter count. The only size statement is "of base size" (the NWM CDiT base model) and "an AV-CDiT model of identical size" (supplementary). I did not find a number.
- Audio (input, output, or none mentioned): both. Input and output are binaural audio latents (SoundStream tokens); the model predicts future audio and image together, and is evaluated on audio fidelity and on interaural level difference and audio-visual lag metrics.
- Code and weights: I found none. The paper text has no code or project link, and the arXiv abstract page lists the code and data panels (alphaXiv, CatalyzeX, Hugging Face, Replicate) with no repository link shown. A web search for the model and dataset names also found no repository. I did not see a statement that AVW-4k is released.

### ke2026helixworld: HelixWorld: A Real-time Interactive Audio-Visual World Model
- Authors: Lei Ke, Jiahao Pan, Zeyue Tian, Jiaming Wang, Haoyuan Huang, Kam Man Wu, Pengjun Fang, Hongyu Liu, Chenyang Qi, Lin Wang, Ruibin Yuan, Weijia Chen, Fangneng Zhan, Qifeng Chen, Wei Xue, Yike Guo
- Year and venue: 2026 (v1 29 Sep 2026); arXiv preprint, not peer reviewed
- Link: https://arxiv.org/abs/2609.38123
- Read depth: raw HTML v1, abstract, Sec. 1 to 4 via keyword search and targeted passages, appendix B (hyperparameters) and C (dataset)
- Part: 3 (audio-generating, action-conditioned world model for exploring scenes; not a robot-navigation policy)
- What they do: A world model for interactive exploration where a user steers with camera trajectories and keyboard-style actions and the model generates video together with spatial stereo sound that follows the camera. A bidirectional teacher is distilled into a few-step causal student for real-time use.
- Built on: LTX-2.3, a joint audio-video model. Experiments section: "HelixWorld builds on open-source LTX-2.3". Not Wan. Student and distillation stages use LoRA (appendix table).
- Inputs: an image, a prompt, continuous 6-DoF camera trajectories (injected with PRoPE) and discrete user actions (added to the diffusion timestep embedding via AdaLN-Zero) (Sec. 1, 3.1).
- Actions: 81 discrete classes, "an 81-class vocabulary over a 9x9 grid of translation and view-rotation velocity bins" (Sec. 3.1), plus continuous camera poses.
- Benchmarks and data: a curated corpus of 4.1k raw hours filtered to 3.0k hours (2.1M synchronized clips) with true stereo audio and metric camera poses (Sec. 2); a new benchmark HelixBench with 1,015 human-verified cases including a Navigation subset (Sec. 4); visual comparison on the WBench navigation split (Table 3). Generated clips are 121 frames at 24 fps and 768x512 with two-channel 48 kHz audio (appendix C).
- Training compute stated: the appendix hyperparameter table for the distillation stage lists "Six nodes, each with eight H200 GPUs" (global batch size 48); I did not find the hardware for the teacher stage. Inference: 24 FPS on a single NVIDIA H800 (Sec. 1).
- Audio (input, output, or none mentioned): output. Stereo audio is generated jointly with video; audio is not an input to the model in the passages read (the inputs are image, prompt, camera and actions).
- Code and weights: repository https://github.com/NoizAI/HelixWorld opened: the README says "Code is Apache 2.0. The weight license is published with the Preview v1 checkpoint", so the Apache license covers the code and not necessarily the weights; README says it released inference code and a preview checkpoint on 2026-09-03 (https://huggingface.co/NoizAI/HelixWorld-preview) and the HelixBench dataset on 2026-10-12 (https://huggingface.co/datasets/NoizAI/Helixbench); both Hugging Face links were not opened.

### zhang2026echowm: EchoWM: Open and Enterable Omnimodal World Models
- Authors: Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang, Haoran Li, Xiaoxiao Ma, Yuming Li, Yijun Liu, Yaofeng Su, Yanwen Ma, Haoyu Wu, Zihan Su, Yue Ma, Lvmin Zhang, Haoyang Huang, Zeyue Xue, Anyi Rao, Nan Duan
- Year and venue: 2026 (v1 24 Aug 2026); arXiv preprint, not peer reviewed (comment: "42 pages, 24 figures")
- Link: https://arxiv.org/abs/2608.23189
- Read depth: raw HTML v1, abstract, Sec. 1 and the data and method sections via keyword search and targeted passages, and the WBench Navigation setup; result tables not read
- Part: 3 (audio-generating, camera-navigation-controlled world model; not a robot policy)
- What they do: A world model that responds to continuous navigation while jointly generating 720p video and sound. Abstract: "an omnimodal world model for enterable generative media that responds to continuous navigation while jointly generating 720p video, environmental sound, music and speech." Training is a progressive four-stage curriculum (audio-visual continued pretraining, action fine-tuning of the camera branch, joint fine-tuning, then distillation to a four-step causal "Flash" variant).
- Built on: LTX-2.3. Method: "We build on a pretrained joint audio-visual diffusion transformer", and later "our LTX-2.3-based audio-visual generator". Not Wan. Action fine-tuning trains only the camera-branch projections at first, and later stages update the generator.
- Inputs: a reference observation (first-frame image), a structured text world description, and a control sequence represented as a relative 6-DoF camera trajectory (discrete commands and continuous poses are both mapped to it). Audio is not an input in the passages I read.
- Actions: camera intent, not a robot action. Sec. 1: "interaction focuses on navigation and viewpoint evolution representable by continuous trajectories"; so "navigation" means moving the viewpoint of a generated scene, in first- or third-person, in game-like and real video, not a robot navigating a task. The paper evaluates on the 158-case WBench Navigation split.
- Audio (input, output, or none mentioned): output. It generates environmental sound, music and speech jointly with video. The channel count of the generated audio is not stated in the passages I read; the training data keeps the native stereo mix and "preserving stereo channels when present" (data section), so stereo is likely but not stated for the output. Teacher-forced and distilled generation produce "synchronized audio-visual chunks". I did not see a spatial or binaural claim for the output.
- Benchmarks and data: WBench Navigation (158 cases) and SANA-WM-Bench; training data from internally collected game recordings with native stereo mix, Unreal Engine rendered trajectories, internet gameplay and general internet video.
- Training compute stated: not found in the passages read.
- Code and weights: repository https://github.com/jd-opensource/JoyAI-Echo opened: README news "2026-08-26 Released Echo-WM"; `echo_wm/` has inference scripts (`inference_wm.py`, `inference_wm_causal.py`), a Gradio demo and bundled LTX code, and I found no training script in the file tree. Weights at https://huggingface.co/Echo-Team/Echo-WM (base and flash preview), listed in the README; the README says "Video + audio" outputs for both variants and that the Gemma 3 text encoder is downloaded separately. License: the README says the project is based on LTX-2 and "remains subject to the LTX-2 Community License Agreement", modified "for academic and research purposes only" and "not intended for commercial use"; the GitHub API license is NOASSERTION. The same repository has a UE simulation pipeline for the world data engine (README news 2026-09-04).

---

## Could not verify

- Nothing that I list as a section failed to open. A Hugging Face model page for RynnWorld-Teleop appeared in search results and was not opened; nothing above relies on it.
- Venue claims (CVPR 2025, ECCV 2026, ICLR 2026, IROS 2026, NeurIPS 2026) come from arXiv comment fields, project pages, repository titles or a search-result page. I did not open a conference page for any of them.
- Several Hugging Face weights pages (facebook/nwm, anuragba/egowm, NoizAI/HelixWorld-preview, Fysics-AI/UniWAM) and the ModelScope AstraNav action-former file were linked and not opened. I opened https://huggingface.co/TeleEmbodied/WNM-3D and used the Hugging Face API file lists for J1dan/imaginav and the acvlab organization. Whether the ModelScope action-former checkpoint also holds the video LoRA weights is unchecked.
- Whether WNM-3D's backbone is trained in full or with LoRA is not stated in so many words: the paper says the backbone is "optimized" and has no LoRA mention, so I read it as full training and say so in the entry.
- UniNav code: the paper says "Code will be released." with no link; I did not find a repository.

## Found but not opened (no claims made)

- Seen only as arXiv API metadata in the first round: NavForesee (arXiv 2512.01550), VISTAv2 (2512.00041), LatentPilot (2603.29165), PROSPECT (2603.03739), Language-Conditioned World Modeling for Visual Navigation (2603.26741).
- Partly read and left out of the sections: FutureNav (2606.30367; a VLM-based world-action model, no video generator; repository https://github.com/linglingxiansen/FutureNav shows weights and code badges) and Learning 3D Persistent Embodied World Models (2505.05495; CogVideoX-style backbone).
- Named in the PR #50 review, not opened by me: DECOWAM (2608.20114) and MobileWAM (2608.04657) (mobile manipulation WAMs); G2WAM, DiffWAM, ForeFly, DroneWAM, Hydra, RIWANav and AquaWAM (not Wan 2.2 builders by the reviewer's keyword scan, backbones not read); BinauralVAE (2609.06837) and AudioWorldSim (2608.21075) (audio reconstruction and dataset leads). Seen in search, driving rather than navigation, not opened: SimWAM and DriveWAM (reported in search snippets as Wan2.2-based driving models).

## Searches run

Web search, standard mode, 2026-11-16 (exact queries):
1. navigation world model egocentric video prediction navigation actions planning
2. navigation world action model Wan2.2 video generation navigation policy arXiv
3. video diffusion model embodied navigation policy Cosmos Habitat vision-language navigation world model 2026
4. audio-visual navigation world model sound future prediction arXiv
5. video generation model fine-tuned for vision-language navigation actions, pretrained video diffusion as navigation policy
6. world model for navigation Wan 2.1 fine-tune action-conditioned egocentric
7. mobile manipulation world action model navigation joint video action Wan2.2 base
8. sound-aware world model embodied agent audio prediction navigation SoundSpaces video generation
9. unified world model navigation memory-augmented egocentric visual navigation diffusion image-goal "navigation world model" 2026 arXiv
10. vision-and-language navigation world model imagination future observations video generation VLN-CE R2R 2026
11. Wan2.1 OR Wan2.2 fine-tuned navigation world model drone OR robot OR humanoid egocentric action-conditioned video 2026
12. AV-CDiT AVW-4k Audio-Visual World Models code repository project page
13. world model navigation audio prediction binaural OR "sound" action-conditioned world model robot 2026 -SoundSpaces agents "audio-visual world model"
14. audio-visual navigation video diffusion world model predicts future binaural audio and frames agent sound source localization generative
15. spatial audio conditioned video generation egocentric navigation world model sound source "world action model" audio
16. Helix World explorable video spatial stereo sound LTX-2.3 arXiv EchoWM audio-visual world model

Review round (my own additional searches, 2026-11-16, for Wan 2.2 based navigation world action models):
17. NavDreamer video models as zero-shot 3D navigators Wan2.2
18. world action model vision-language navigation initialized from Wan2.2-TI2V-5B joint video action 2026 R2R-CE
19. navigation world action model Wan2.2 drone OR UAV OR humanoid OR quadruped OR object-goal navigation video diffusion action head arXiv 2026

Other checks: arXiv API (export.arxiv.org, id_list, plus a title search for WorldVLN) for every identifier above; keyword counts for audio, sound, acoustic and binaural over the full text of every navigation and mobile-manipulation paper I downloaded (see the closing section for the list); keyword counts for navigat* over the ten manipulation WAM papers; direct reads of READMEs, the GitHub repository tree API, and Hugging Face model pages or API lists for the released-artifact claims. WNM-3D, UniWAM, NavGen and EchoWM came from the PR review, not from my searches; I then opened each from raw arXiv HTML. My three review-round searches found NavDreamer and WorldVLN but no further Wan 2.2 navigation model that I could open.

## What exists and what I did not find

Navigation world models: the line starts with Navigation World Models (2024), a 1B-parameter diffusion transformer trained from scratch on robot navigation video, planning with a search over candidate trajectories (code and weights CC BY-NC 4.0). Since then, papers have moved to start from a pretrained video model. In the papers I opened, the base models are Wan2.2-TI2V-5B (AstraNav-World, WNM-3D, UniWAM, NavGen), Wan2.1-1.3B (UniNav), Wan2.1-14B, Cosmos Predict2-2B and SVD (EgoWM), Wan2.1-Fun-1.3B (ImagineUAV, for drones), Cosmos Predict2 2B (NavWAM), LTX-Video 2B and 13B LoRA (ImagiNav), InfinityStar-8B (WorldVLN, drones), Anole-7B (UniWM, not a video diffusion model), and LTX-2.3 (HelixWorld, EchoWM). World-in-World does not propose a model but post-trains and compares Wan2.1, Wan2.2 (5B and A14B), Cosmos-Predict2, LTX-Video and SVD on navigation tasks in Habitat.

Navigation and mobile world action models built on Wan 2.2 (all checked in the paper text of each):
- AstraNav-World (2512.21714): Wan2.2-TI2V-5B video generator with LoRA, plus a Qwen2.5-VL-3B planner; joint waypoints and future frames; Habitat R2R-CE, RxR-CE, HM3D-OVON; 96 H20 GPUs. Releases: Apache-2.0 repository with inference code for the action-former variant, action-former checkpoints (ModelScope) and VLM training code; I did not find video-generator weights or a video-generator training script.
- WNM-3D (2608.07267): Wan2.2-TI2V-5B, backbone and adapter trained (no LoRA mentioned), VGGT-Omega frozen; local-frame (dx, dy, dpsi) action blocks; GN-Bench and VLN-CE; 16 H100. Releases: Apache-2.0 code with inference and Stage I and II training scripts (Stage III RL in a separate repository), Stage I, II and III checkpoints on Hugging Face (non-commercial research terms because of VGGT-Omega and DINOv3).
- UniWAM (2609.39388): Wan2.2-TI2V-5B for mobile manipulation with a navigation stream; all DiT parameters trained (VAE and text encoder frozen); MAP-Bench navigation plus 24 real tasks; 16 H200 for the MAP-Bench runs. Releases: Apache-2.0 code including training and inference, Base and Scaled MAP-Bench checkpoints (navigation-only), MAP-Bench and a 100,052-trajectory MAP-Data subset on Hugging Face (gated), with real-robot task checkpoints not included.
- NavGen (2609.30770): Wan2.2 5B with LoRA (rank 64) as the navigation video model, plus Wan2.2 14B T2V as a data generator; drones; 16 A100. Releases: none yet ("will be made publicly available upon acceptance").
- ABot-M0.5 (2607.00678): Wan2.2-5B, full-parameter fine-tune, mobile manipulation (base plus arm), RoboCasa365; releases Apache-2.0 inference code and a RoboCasa365 checkpoint.
- Wan 2.2 as an unchanged generator inside a pipeline: Action Agent (2605.01477, options WAN 2.2 or LTX-Video) and FlightDiffusion (2509.14082, Wan 2.2 I2V Fast through a hosted endpoint); neither reports training Wan. NavDreamer (2602.09765) benchmarks Wan 2.2 among several video models but deploys Wan 2.6 through an API.
- Wan2.2 variants in World-in-World (5B and A14B) are post-trained benchmark models for Habitat navigation (post-training on 40K instances for one epoch).
The first version of this survey said AstraNav-World was the one Wan 2.2 navigation model I found; that was wrong, as the list above shows. I did not find a navigation world model fine-tuned from the Wan2.2 A14B size other than World-in-World's post-trained benchmark variant, and I cannot rule out others because my searches were not exhaustive.

Audio and these Wan 2.2 navigation models: I checked this directly. A case-insensitive keyword count for audio, sound, acoustic and binaural over the full text of AstraNav-World, WNM-3D, UniWAM, NavGen, ABot-M0.5, Action Agent, FlightDiffusion, NavDreamer and World-in-World gives 0 hits, except one hit in NavGen that is only a reference title ("LTX-2: efficient joint audio-visual foundation model"). The same count gives 0 for the other Part 1 papers (NWM, UniNav, EgoWM, NavWAM, ImagiNav, ImagineUAV, DreamToNav, WorldVLN, UniWM) and FutureNav. So none of these takes or predicts audio, as far as keyword search of their full text shows. This is a text search, not a check of the released code.

Models that produce audio (three, not two): AVWM (2512.00883), built on a Navigation World Models CDiT base with a SoundStream audio tokenizer (not a video foundation model), predicts binaural audio and images in SoundSpaces with discrete robot-style actions, and its paper has no code link, no parameter count and no dataset release statement; HelixWorld (2609.38123), built on LTX-2.3, stereo audio with video under camera and keyboard actions, Apache-2.0 code and preview checkpoint (weight license separate); EchoWM (2608.23189), built on LTX-2.3, video with environmental sound, music and speech under a relative 6-DoF camera trajectory, inference code and weights released under the LTX-2 community license, output channel count not stated in the passages I read. HelixWorld and EchoWM are explorable generated worlds, not robot navigation policies. None of the three is built on Wan 2.2. I did not find an audio-producing or audio-taking navigation world model built on Wan 2.2.

Action spaces differ by setting. Image-goal and robot-dataset models use continuous waypoints or (delta x, delta y, yaw) (NWM, UniNav, NavWAM, WNM-3D, UniWAM, EgoWM). Habitat-style simulation uses four discrete moves (AstraNav-World, World-in-World, AVWM). Pipelines that extract actions from generated video use visual odometry or an inverse dynamics model (ImagiNav, ImagineUAV, DreamToNav, FlightDiffusion, NavDreamer, NavGen). Joint world-action designs that put actions in the same denoising pass as video are AstraNav-World, UniNav, NavWAM, WNM-3D and UniWAM.

Code and compute: see the entries for exact releases; the drone navigation models WorldVLN (InfinityStar backbone) and NavDreamer differ here: WorldVLN releases inference code, training code and weights (code under CC BY 4.0), NavDreamer none. Training hardware stated: NWM 8 machines of 8 H100 (64 GPUs); AstraNav-World 96 H20; UniNav four RTX 4090 for two epochs; ImagineUAV eight PRO 6000 for two epochs; UniWM four A100-80GB for 20 epochs; AVWM eight A100; EgoWM eight A100 for its SVD and Cosmos variants; NavWAM four RTX PRO 6000 per run; Action Agent one RTX 5090 for its 43M-parameter policy; WNM-3D 16 H100; UniWAM 16 H200; NavGen 16 A100; WorldVLN eight A800; HelixWorld six nodes of eight H200 for the distillation stage. I did not find compute for ImagiNav, ABot-M0.5 or EchoWM, and none is stated for the Wan variant of EgoWM.

Part 2: among the ten manipulation WAMs from the reviewed file, I found no navigation or locomotion experiment and no statement that the method extends to navigation, but the word "navigation" appears in two of them: DreamZero's data description lists navigation among the skills of its AgiBot G1 mobile manipulator data, and mimic-video cites a paper with "Navigation" in its title. Cosmos Policy names "locomotion tasks" once, as a limitation tied to slow inference. WAMs built for tasks that move the base exist: ABot-M0.5 and UniWAM, both on Wan2.2-5B, and two more titles (DECOWAM, MobileWAM) that the reviewer found and I did not open. The claim is therefore narrow: in the ten I searched, no navigation results; mobile-manipulation WAMs do exist.

Part 3 gaps: for AVWM, I did not find a parameter count, a code link, or a statement that AVW-4k is released. I did not find a navigation world action model, as opposed to an explorable generated world, that takes audio as an input to a policy. This is a statement about what I read and searched, not a claim that none exist.
