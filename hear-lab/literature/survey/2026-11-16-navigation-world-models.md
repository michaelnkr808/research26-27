# Literature survey D: world models and world action models for navigation (2026-11-16)

Topic: navigation world models and video-generation-based navigation policies (what base video model each starts from, inputs, actions, benchmarks, stated compute, code), whether the manipulation world action models in the reviewed file (`2026-11-16-world-action-models.md`) report navigation or mobile tasks, and whether any navigation world model takes or predicts audio.

How the search was done: web search to find candidates (queries listed at the end). Search results are leads only. Every paper below was then checked against the arXiv API (title, authors, date, identifier, comment field), and its arXiv HTML full text was downloaded as raw text with curl, stripped of tags (math replaced by its LaTeX alt text), and searched and read in the passages that matter. Code and weights claims come from the project page or the repository README and repository metadata read on 2026-11-16, not from search snippets. Status of everything below: `verified` means read on the page on 2026-11-16.

Caveats the user should know:
- How deep I read each paper is "raw HTML, keyword search plus targeted passages". I did not read every section of every paper. Where a fact is missing here, it means I did not see it in the passages read, not that the paper lacks it.
- Every arXiv paper is "preprint, not peer reviewed" unless a venue is noted. Where a venue is noted, it comes from the arXiv comment field or the authors' own project page or repository, and I did not check the conference's own site. Treat those venues as claimed, not confirmed.
- Numbers are only ones copied from the paper, with the section or table named. Success rates are left out on purpose; the papers' tables have them.
- Search found several papers that I did not open (listed under "Found but not opened"). They are not part of this survey's claims.

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
- Code and weights: project page https://www.amirbar.net/nwm/ links code and models; repository https://github.com/facebookresearch/nwm opened: official PyTorch implementation with CDiT training code, models at https://huggingface.co/facebook/nwm; the repository metadata shows license "NOASSERTION" (no recognized license text).

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
- Training compute stated: only "All experiments are conducted on a cluster equipped with H20 GPUs" (Sec. 4.2). No GPU count or hours seen.
- Audio (input, output, or none mentioned): none mentioned (zero hits in the full text).
- Code and weights: project page https://astra-amap.github.io/AstraNav-World.github.io/ links code, model and demo. Repository https://github.com/amap-cvlab/AstraNav-World opened: Apache-2.0. README news lists inference code (R2R, RxR, OVON) and "inference checkpoints of action former" (ModelScope link). Optional future-frame prediction needs a separate Wan2.2-TI2V-5B-Diffusers download. I did not see released weights for the fine-tuned video generator or a training script in the README.

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
- Code and weights: repository https://github.com/miccooper9/egowm opened. Its README TODO list marks the SVD navigation inference scripts "Done" and the Wan2.1-14B train and infer scripts as "Soon" (not yet released as of the README read); weights page https://huggingface.co/anuragba/egowm/ is linked but I did not open it. No license shown in the repository metadata.

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
- Training compute stated: none for training. Inference cost is measured on a single RTX PRO 6000 (96 GB) (App. C.1).
- Audio (input, output, or none mentioned): none mentioned (zero hits).
- Code and weights: project page https://dachii-azm.github.io/navwam/ opened; it shows "Code (coming soon)". No weights seen.

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
- Code and weights: project page https://j1dan.github.io/ImagiNav lists code and a dataset; repository https://github.com/J1dan/ImagiNav-release opened: MIT license, README says it contains the pipeline, training configurations, and "checkpoint and dataset instructions"; dataset at https://huggingface.co/datasets/J1dan/imaginav-dataset (linked, not opened).

### liu2026imagineuav: ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning
- Authors: Xuchen Liu, Jiawei Huang, Shihao Xia, Bingxi Liu, Jinqiang Cui, Jiankun Yang
- Year and venue: 2026 (v2 read, dated 31 May 2026 in the API); arXiv preprint, not peer reviewed
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
- Built on: NVIDIA Cosmos 2.5, used without training of the video model ("without requiring any task-specific fine-tuning"); Qwen 2.5-VL for prompt refinement; a small YOLO11n detector trained on a robot dataset (Sec. III-C).
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
- Training compute stated: not found for training; inference timing on an RTX 5090.
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
- Code and weights: repository https://github.com/UWMILab/UniWM opened (the paper's date line links it): official implementation; license not shown in the metadata. I did not check for released weights.

---

## Part 2: manipulation world action models, and whether they report navigation or mobile tasks

I searched the full text of ten manipulation WAMs from the reviewed file (Fast-WAM, Motus, LingBot-VA, tau0-WM, DreamZero, Cosmos Policy, DiT4DiT, UVA, mimic-video, UWM) for navigat*, mobile, egocentric, locomotion, humanoid, legged, wheeled, driving, drone and UAV, and read the hits. The two with something relevant are below, followed by ABot-M0.5, which is a WAM built for mobile manipulation.

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
- Benchmarks and data: one of the two robots is "the AgiBot G1 mobile bimanual manipulator" (Sec. 5); about 500 hours of teleoperation data across 22 environments such as homes, restaurants and offices. No navigation task or base-motion result was found by keyword search; tasks described are manipulation, plus free-form prompts such as "Press elevator button".
- Training compute stated: see the reviewed file.
- Audio (input, output, or none mentioned): none mentioned.
- Navigation or locomotion statement: Sec. 6 lists future work (scaling laws, in-the-wild human egocentric data, faster inference, long-horizon reasoning, high-precision tasks, embodiment design). It does not name navigation or locomotion. Egocentric human video is discussed only as manipulation-related training data.
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
- Code and weights: repository https://github.com/amap-cvlab/ABot-Manipulation opened: Apache-2.0; README news (2026-07-21) says the inference code, pre-trained weights and evaluation scripts for ABot-M0.5 are released; weights at https://huggingface.co/acvlab (linked, not opened).

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
- Code and weights: repository https://github.com/NoizAI/HelixWorld opened: Apache-2.0; README says it released inference code and a preview checkpoint on 2026-09-03 (https://huggingface.co/NoizAI/HelixWorld-preview) and the HelixBench dataset on 2026-10-12 (https://huggingface.co/datasets/NoizAI/Helixbench); both Hugging Face links were not opened.

---

## Could not verify

- Nothing that I listed failed to open. A Hugging Face model page for RynnWorld-Teleop appeared in search results and was not opened; nothing above relies on it.
- Venue claims (CVPR 2025, ECCV 2026, ICLR 2026, IROS 2026) come from arXiv comment fields, project pages or repository titles. I did not open a conference page for any of them.
- The Hugging Face weights pages (facebook/nwm, anuragba/egowm, acvlab, NoizAI/HelixWorld-preview, J1dan dataset) were linked from repositories and project pages and were not opened; I report only what the README or project page says.

## Found but not opened (arXiv metadata only, no claims made)

Confirmed to exist through the arXiv API, never read: NavForesee (arXiv 2512.01550), VISTAv2 (2512.00041), LatentPilot (2603.29165), PROSPECT (2603.03739), Language-Conditioned World Modeling for Visual Navigation (2603.26741). Opened and partly read but left out of the sections because it uses a vision-language model with no video generator: FutureNav (2606.30367, a "VLM-based unified world-action modeling framework" for vision-and-language navigation; the repository README at github.com/linglingxiansen/FutureNav lists weights and code; no audio hits). Learning 3D Persistent Embodied World Models (2505.05495) was opened and uses a CogVideoX-style backbone with 3D memory; I did not read it further and did not include it.

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

Other checks: arXiv API (export.arxiv.org, id_list) for every identifier above; keyword counts for audio, sound, acoustic and binaural over the full text of 14 navigation papers; keyword search for navigation, mobile, egocentric, locomotion, humanoid, legged, wheeled, driving, drone and UAV over ten manipulation WAM papers.

## What exists and what I did not find

Navigation world models: the line starts with Navigation World Models (2024), a 1B-parameter diffusion transformer trained from scratch on robot navigation video, planning with a search over candidate trajectories. Since then, papers have moved to start from a pretrained video model. In the papers I opened, the base models are Wan2.2-TI2V-5B (AstraNav-World), Wan2.1-1.3B (UniNav), Wan2.1-14B, Cosmos Predict2-2B and SVD (EgoWM), Wan2.1-Fun-1.3B (ImagineUAV, for drones), Cosmos Predict2 2B (NavWAM), LTX-Video 2B (ImagiNav), and Anole-7B (UniWM, which is not a video diffusion model). World-in-World does not propose a model but post-trains and compares Wan2.1, Wan2.2 (5B and A14B), Cosmos-Predict2, LTX-Video and SVD on navigation tasks in Habitat.

Wan 2.2 specifically: AstraNav-World is the one navigation model I found that is fine-tuned from Wan 2.2 (the 5B text-and-image-to-video model, with LoRA), and it also takes a language instruction and produces waypoints. Action Agent and FlightDiffusion use Wan 2.2 as an off-the-shelf generator inside a pipeline, with no training of Wan reported. ABot-M0.5 is built on Wan2.2-5B but targets mobile manipulation, not pure navigation. I did not find a navigation world model fine-tuned from the larger Wan2.2 A14B, other than World-in-World's post-trained benchmark variant.

Wan 2.1 and Cosmos: UniNav, ImagineUAV and EgoWM fine-tune Wan 2.1; NavWAM, EgoWM and DreamToNav (off the shelf, Cosmos 2.5) use Cosmos. I did not find a navigation model built from Wan 2.1 at the 14B size apart from EgoWM's variant, whose training scripts are not yet released per its README.

Action spaces differ by setting. Image-goal and robot-dataset models use continuous waypoints or (delta x, delta y, yaw) (NWM, UniNav, NavWAM, EgoWM). Habitat-style simulation uses four discrete moves (AstraNav-World, World-in-World, AVWM). Pipelines that extract actions from generated video use visual odometry or an inverse dynamics model (ImagiNav, ImagineUAV, DreamToNav, FlightDiffusion). Joint world-action designs that put actions in the same denoising pass as video are AstraNav-World, UniNav and NavWAM.

Code: I found released code or weights for NWM, AstraNav-World (inference code and action-former checkpoints, not the video generator), World-in-World, ImagiNav, UniWM (code), ABot-M0.5, and HelixWorld. EgoWM has a repository but its Wan scripts were still listed as coming. NavWAM says "Code (coming soon)", and UniNav says code will be released. I did not find a code link for ImagineUAV, DreamToNav, Action Agent or FlightDiffusion.

Training compute stated: NWM 64 H100 GPUs; UniNav four RTX 4090 GPUs for two epochs; ImagineUAV eight PRO 6000 GPUs for two epochs; UniWM four A100-80GB GPUs for 20 epochs; AVWM eight A100 GPUs; EgoWM eight A100 GPUs for its SVD and Cosmos variants. AstraNav-World says only "H20 GPUs". I did not find compute for NavWAM training, ImagiNav, Action Agent or ABot-M0.5.

Part 2: I did not find any of the ten manipulation WAMs from the reviewed file reporting a navigation, locomotion or mobile-base experiment, or stating that the method extends to navigation. Cosmos Policy names "locomotion tasks" once, as a limitation tied to slow inference. DreamZero trains on an AgiBot G1 described as a mobile bimanual manipulator, but I found only manipulation tasks. The WAM I found that does handle moving the base is ABot-M0.5 (mobile manipulation on RoboCasa365 and real robots, with the base and arm actions kept separate).

Part 3: apart from arXiv 2512.00883, the only other audio-producing world model I found is HelixWorld, which generates stereo sound together with video under camera and keyboard-style actions and is built on LTX-2.3. It is for interactive scene exploration, and I did not see a robot navigation task or an agent in it. I did not find any navigation world model or navigation world action model that takes audio as an input to the model, or that predicts audio, other than AVWM and HelixWorld. A keyword search over the full text of 14 navigation and mobile-manipulation papers I downloaded (the 12 Part 1 papers, ABot-M0.5, and FutureNav) found zero mentions of audio, sound, acoustic or binaural. For AVWM, I did not find a parameter count, a code link, or a statement that AVW-4k is released. This is a narrow statement about what I read and searched; it is not a claim that none exist.
