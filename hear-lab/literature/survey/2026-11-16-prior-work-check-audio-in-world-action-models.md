# Prior-work check: does giving audio to a world (action) model help an agent act? (2026-11-16)

Question checked: has anyone measured whether giving audio (any audio, and spatial audio in particular) to a world action model, world model, video-action model, world-model-based planner or a policy with a learned predictive model improves task success, by comparing the same or a matched system with and without audio, or with mono against spatial audio, in navigation or manipulation, in simulation or on a robot.

Method: every paper below was opened from raw text (arXiv HTML stripped with a local script, or `pdftotext` on the PDF) on 2026-11-16, and its metadata was confirmed with the arXiv API (except NaVLA2, an AAAI paper read from the authors' PDF, and the Cosmos 3 report, read from NVIDIA's PDF). Numbers are copied from the paper text; none were reproduced. `verified` below means read in the paper; `speculative` marks this file's own reading.

## Verdict
One paper has measured almost exactly the first half of the question. Audio-Visual World Models (arXiv 2512.00883), in its version 3 of 5 June 2026, ran a world model that predicts frames and binaural audio inside a VLM planner for simulated audio-visual navigation (SoundSpaces 2.0) and compared the full audio-visual world model with a "w/o audio" variant, a "w/o vision" variant and a no-world-model baseline over 500 episodes: success was 0.381 with audio-visual foresight against 0.266 without the audio part of the foresight and 0.269 with no world model (Table 4 of v3). That table is gone from the current version 4 (4 August 2026), whose only navigation table compares the agent with and without the world model, so the audio-ablation result exists only in a superseded version. In manipulation, a pipeline that uses a commercial joint video-and-audio generator as a planner (Dreaming the Sound of Contact, arXiv 2609.19137) showed on a real Franka that audio-derived force profiles beat kinematic-only and some constant-force execution, and Audio-WM (2512.08405) and HEAR (2603.16086) predict future audio from a single microphone for a policy but do not test spatial audio. No paper found compares mono against spatial (binaural or Ambisonics) audio inside any world model or world action model, and none gives spatial audio to a large pretrained video world action model (Wan, Cosmos or similar) and measures the effect on task success; the spatial-against-non-spatial comparisons that exist (NaVLA2's mono-downmix branch against its binaural branch, S2A2's ablation of its array-derived spatial map) are in policies with no predictive model.

## Closest work, ranked

### wang2025audio: Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics
- Link: https://arxiv.org/abs/2512.00883 (versions read: v4 of 4 Aug 2026 in full as HTML and PDF; v3 of 5 Jun 2026 as PDF, planning section and Appendix G in full; v1 of 30 Nov 2025 and v2 of 10 Mar 2026 as PDF, planning tables only). Version dates from the arXiv abstract page's submission history.
- Read depth: v4 full text (all sections and supplementary, including Table 4 numbers from the PDF because the HTML drops them); v3 Sec. 4.4, Table 4, Limitations and Appendix G in full; v1 Table 6 and v2 Table 8 (planning) and their surrounding text. Not read: v3 sections other than those named.
- Setting: navigation, simulated only (continuous audio-visual navigation in SoundSpaces 2.0 on Matterport3D; success = stop within 1 m of the sound source in v4, per "An episode succeeds if the agent issues a stop action within 1m of the target sound source").
- Predictive world model: yes. AV-CDiT, a conditional diffusion transformer initialised from a base-size Navigation World Models CDiT, predicts the next egocentric 128x128 RGB frame and the next 0.15 s binaural audio segment given relative motion (v4 "Training Configuration": "We initialize AV-CDiT from a pretrained NWM model ... fine-tuned on AVW-4k"). For planning, a variant also predicts a progress-reward token. Not a large pretrained video model.
- What audio: binaural, two channels, 16 kHz, 0.15 s per step, encoded by a SoundStream tokenizer trained on AVW-4k. The agent also receives binaural audio as an observation (v4: "using egocentric RGB observations and binaural audio").
- The audio comparison:
  - v3 (5 Jun 2026), Sec. 4.4 "World-Model-Assisted VLM Navigation", Table 4 "VLM-based AV-Nav with AVWM integration. Statistical significance validated via t-tests (p < 0.05)." Setup quote: "We compare four configurations: a Baseline Nav-Agent without a world model, the full AVWM Nav-Agent, two single-modality variants (w/o audio and w/o vision), and an Oracle WM Nav-Agent that uses simulator futures as an upper bound." Rows (Succ., SPL, SoftSPL, SNA, DTG): Baseline 0.269, 0.166, 0.239, 0.023, 10.36; w/o audio 0.266, 0.155, 0.214, 0.022, 10.71; w/o vision 0.280, 0.164, 0.209, 0.024, 10.72; AVWM (ours) 0.381, 0.207, 0.252, 0.030, 9.55; Oracle WM 0.423, 0.243, 0.294, 0.036, 8.98. Text: "Table 4 summarizes navigation performance over 500 randomized test episodes" and "Both single-modality variants (w/o audio and w/o vision) underperform the full agent by a wide margin, even falling below the baseline on SoftSPL and final distance-to-goal." The planner is Qwen3.6-35B-A3B choosing waypoints (Appendix G.5); world-model forecasts are injected into its prompt as a predicted RGB image and a "<future audio summary text>" for at most K = 3 candidates (Appendix G.4, G.6.3), and the prompt says to "Treat them only as tie-break evidence". The agent itself still hears in every configuration: an "acoustic prior map" built "by processing binaural waveforms into directional acoustic intensities" is part of the prompt (Appendix G.2). So "w/o audio" removes audio from the world model's forecast, not from the agent's input. The paper does not say whether the "w/o audio" world model was retrained or the audio output was only dropped from the prompt.
  - v4 (current), "Planning with an Audio-Visual World Model", Table 4: a different planner (beam-search lookahead on a pretrained AV-Nav policy). Rows compare only "AV-Nav Agent (baseline)" (SPL 45.08, SoftSPL 55.09, NA 332.817, SNA 6.44), "AVWM" and "Oracle WM" at beam widths B = 3, 4 and rollout lengths k = 4, 5; best AVWM row B = 3, k = 5: SPL 47.38, SoftSPL 56.53, NA 317.19, SNA 7.09. No success rate, no audio-ablated or vision-only world model row, no mono row. The same table appears in v2 (Table 8) and, with success rates, in v1 (Table 6: SR 55.30 baseline against 55.20 to 56.50 with AVWM).
  - v4 also compares world models on prediction quality only (Table 2: DIAMOND+AudioLDM, NWM+AudioLDM, AVDiT+action; Table 3: architecture ablations), not on navigation success. NWM is used there as a "visual-only" baseline for frame quality.
- What it does NOT test: mono against binaural (no mono variant in any version read); removing audio from the agent's own input; a large pretrained video model; manipulation; real robots. The audio ablation exists only in v3 and was dropped from v4, which the user should treat as a result the authors no longer report (`speculative`: the reason is not stated).
- How it differs from "a large pretrained video world action model given spatial audio, compared with the same model without audio and with mono audio": the world model is a small CDiT (NWM base size) trained on 30 hours of SoundSpaces data, not a Wan or Cosmos scale model; it is a world model used by a separate planner, not a world action model that outputs actions; its audio comparison (v3 only) ablates the audio forecast while the agent still hears binaural audio; there is no mono condition.

### ji2026dreaming: Dreaming the Sound of Contact: Leveraging Video and Audio Generation for Zero-Shot Force-Aware Manipulation and Data Generation
- Link: https://arxiv.org/abs/2609.19137 (v2, 17 Sep 2026; arXiv API title confirmed; authors Guanhua Ji, Tianyu Li, Dayoon Suh, Yuqian Zhang, Boyan Zhang, Nadia Figueroa)
- Read depth: raw HTML v2, abstract, Sec. I to IV-B in full; Sec. IV-C, IV-D and V not read.
- Setting: manipulation, real robot (Franka Panda), four contact tasks (whiteboard wiping, carrot peeling, chocolate box stacking, lamp button pressing).
- Predictive world model: yes in the broad sense: a commercial joint video-and-audio generator predicts the future of the task from the first image and a prompt ("We use Seedance 2.0 [3] to synthesize a robot-centric video with synchronized audio from the initial RGB image and the task prompt", Sec. III-B1). Actions are extracted from the generated video (trajectory) and the generated audio (force). It is a cascaded, zero-shot pipeline, not a trained world action model, and it is not conditioned on actions.
- What audio: generated audio only, used offline; "No audio is recorded or processed during execution" (Sec. III-C). Channel count not stated; the method uses loudness (LUFS) of a separated contact sound, so it does not use direction.
- The audio comparison: Table I (Sec. IV-A) "Zero-shot execution success: Force-Aware (Ours) vs. Kinematic-Only Baseline": base 0/10, 5/10, 3/10, 0/10; ours 10/10, 9/10, 9/10, 8/10 (wiping, peeling, stacking, pressing); "combined success rate of 36/40 (90.0%), compared with 8/40 (20.0%) for the baseline". The baseline "uses the same trajectories but sets the desired contact force to zero throughout execution", so it removes force control as a whole, not only audio. The cleaner audio test is Table II (Sec. IV-B), audio-shaped variable force against constant force on the same trajectories: wiping 7/10 and 10/10 for the two variable-force bounds against 8/10 (10 N) and 2/10 (15 N); peeling 9/10 and 9/10 against 8/10 and 6/10; the text says the audio-shaped profile "matches or exceeds constant-force execution in success while lowering peak force and impulse" (Sec. I). 10 trials per condition.
- What it does NOT test: audio as an input that the system perceives at run time; spatial or multi-channel audio; navigation; a trained world model.
- How it differs: closed generator used zero-shot, audio converted into a force heuristic, no closed-loop audio input, no mono-against-spatial comparison; manipulation rather than navigation.

### zhang2025learning: Learning Robot Manipulation from Audio World Models
- Link: https://arxiv.org/abs/2512.08405 (v1, 9 Dec 2025; arXiv API confirmed)
- Read depth: full text (PDF, five pages including references).
- Setting: manipulation; real (Kinova Gen3 water filling) and simulated (RoboPianist piano duet).
- Predictive world model: yes, an audio-only one: "a generative latent flow matching model to anticipate future audio observations" (abstract). It predicts about 2.56 s of future spectrogram from about 1.28 s of past spectrogram (Sec. 3.1); in the piano task MusicVAE predicts the next 64 MIDI steps (Sec. 3.2). It predicts no video. The policy takes current audio, predicted audio and a 224x224 image (Sec. 2).
- What audio: one "MAONO omnidirectional USB lapel microphone" (Sec. 3.1); log Mel spectrogram. Mono.
- The audio comparison: water filling reports only the full system: "Our method achieved a 100% success rate over 30 trials" (Sec. 3.1), with no no-audio or no-prediction row. Piano: "compared against a basic RL baseline without future lookahead. We observe improved performance when future goal states are incorporated into the observation" (Sec. 3.2), measured by F1 but with no number in the text. The comparison is with against without predicted future audio, not with against without audio.
- What it does NOT test: removing audio input; spatial audio; video prediction; navigation.
- How it differs: audio-only world model, mono microphone, small models trained in about a day on one H100 (Sec. 2), no video world model and no channel comparison.

### nie2026towards: Towards the Vision-Sound-Language-Action Paradigm: The HEAR Framework for Sound-Centric Manipulation
- Link: https://arxiv.org/abs/2603.16086 (v2, 18 Sep 2026; arXiv API confirmed)
- Read depth: PDF v2 via `pdftotext`: abstract, Sec. 1, method passages on Envisioner and Advancer, Tables 2, 3 and 4 with their discussion, Sec. 7.5 limitations. Supplementary sections S5.x not read.
- Setting: manipulation; simulated (HEAR-Bench, seven tasks, 100 trials each) and real robot (four tasks, 100 trials each).
- Predictive world model: partly. The "Advancer" is "a decoder-only audio world model [that] predicts near-future acoustic codes" from the policy's latent, used as a training signal; "HEAR does not need to explicitly generate the future audio" at deployment. No video prediction. The backbone builds on Qwen3-Omni.
- What audio: single microphone: "our hardware setup relies on a single workspace microphone. This configuration inherently limits the robot's ability to localize sound sources" (Sec. 7.5).
- The audio comparison: Table 2 (simulation, average success): OpenVLA 0.10 against OpenVLA-Waveform 0.53 and OpenVLA-ASR 0.29; pi0.5 0.14 against pi0.5-Waveform 0.61 and pi0.5-ASR 0.35; HEAR 0.81. Table 3 (real robot, average): OpenVLA 0.06, OpenVLA-Waveform 0.35; pi0.5 0.13, pi0.5-Waveform 0.45; HEAR 0.70. These are with-against-without-audio comparisons for VLAs with no world model ("Waveform" renders audio as an image). Table 4 ablates the predictive audio model: "HEAR full (OpenX-Sound pretrained) 0.81", "No Advancer 0.73".
- What it does NOT test: spatial audio; a video world model; navigation; a with-against-without-audio comparison of HEAR itself.
- How it differs: the predictive part forecasts mono audio codes as an auxiliary loss inside a VLA; no video prediction, no spatial audio, no mono-against-spatial test.

### nvidia2026cosmos: Cosmos 3: Omnimodal World Models for Physical AI
- Link: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf ; also listed by the arXiv API as 2606.02800 (v4) with the same title (arXiv listing seen in search results, arXiv text not read)
- Read depth: whole report converted with `pdftotext` and searched for "audio", "sound" and "acoustic" (213 lines with "audio"); every hit near policy, action, robot, navigation or control terms read in context; Sec. 3.2.3 (action data), Sec. 4.2.5 (robot policy post-training) and Sec. 6.2.5 (action generation evaluation, Tables 18 to 20) searched line by line for audio.
- Setting: manipulation policy (DROID, RoboLab, RoboArena, MolmoSpaces, LIBERO-10) plus forward and inverse dynamics in driving, camera motion, robotics and egocentric motion.
- Predictive world model: yes; one mixture-of-transformers that can act as a world action model.
- What audio: base model takes and generates 48 kHz stereo audio (Sec. 2.1.2).
- The audio comparison: none found. Sec. 4.2.5 and Sec. 6.2.5 contain no mention of audio; the policy input is "the current proprioceptive robot state and a three-view visual observation". In the mid-training mixture (Table 6), "Video + Audio" (T2(V+Audio), I2(V+Audio), V2(V+Audio)) and "Action" (forward dynamics, inverse dynamics, policy) are separate streams. The only sentence joining audio and policy is architectural (Sec. 7.4): "Cosmos 3 treats forward dynamics, inverse dynamics, and policy mode as conditioning patterns of one multimodal sequence model over video, audio, text, and action tokens."
- What it does NOT test: any action or policy experiment with audio input, any audio ablation for control.
- How it differs: has the pieces (stereo audio tokens and action tokens in one open model) but reports no experiment that uses audio for acting.

### fan2026navla: NaVLA2: A Vision-Language-Audio-Action Model for Multimodal Instruction Navigation
- Link: https://tanmingkui.github.io/files/publications/NaVLA2.pdf (AAAI-26 proceedings, DOI 10.1609/aaai.v40i22.38886)
- Read depth: full PDF via `pdftotext`, with attention to Sec. 4.1, 5.1, 5.2 and Tables 1 to 3.
- Setting: navigation, simulated (MINav, Habitat with SoundSpaces rendering).
- Predictive world model: no. A VLA (Vicuna-7B LLM) with a CoThinkAct head that outputs chain-of-thought and a multi-step action sequence; nothing predicts future observations.
- What audio: binaural; "the binaural audio is downmixed to mono and encoded by the semantic audio encoder ... In parallel, the original binaural signal is fed into a spatial audio encoder" (Sec. 4.1).
- The audio comparison: Table 2 "Ablation study of audio encoder design" (SR, SPL, NE): neither branch 0.206, 0.162, 6.677; semantic (mono downmix) only 0.219, 0.168, 6.631; spatial only 0.258, 0.201, 6.147; both 0.272, 0.196, 6.242. The mono and spatial branches use different encoders, so this is not channel count alone.
- What it does NOT test: a world model or world action model; video prediction.
- How it differs: the spatial-against-mono evidence is for a VLA policy, not a predictive model.

### hiratsuka2026s2a2: S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information
- Link: https://arxiv.org/abs/2607.26047 (v2, 29 Sep 2026; arXiv API confirmed)
- Read depth: PDF v2: Sec. 1, 3 (framework), 4.1 and 4.2 in full with Tables 2 and 3; appendix A.1.5 (raw-audio baseline). Other appendices not read.
- Setting: manipulation; simulated (Genesis plus Pyroomacoustics) and real (ILOHA arm).
- Predictive world model: no. Policies are ACT, Diffusion Policy, VQ-BeT and pi0.
- What audio: four 8-channel microphone arrays; MUSIC-based acoustic spatial map plus a spotformed spectrogram.
- The audio comparison: Table 2 (simulation, % success, 3 seeds, 100 trials): e.g. ACT localization task Baseline (RGB only) 20.3, S2A2 73.0, w/o Spec. 74.7, w/o Spat. 15.3, Raw (multi-channel waveforms) 43.3. Table 3 (real, 100 trials): L&I task Baseline 5, w/o Spec. 18, w/o Spat. 3, S2A2 72; exploratory 12, 33, 27, 50. The "w/o Spat." arm still uses the array for spotforming, so it is not stereo against mono.
- What it does NOT test: a predictive model of any kind; navigation.
- How it differs: spatial audio clearly helps, but in reactive imitation policies with hand-built localization, not in a world model.

### chen2026lhavln: LH-AVLN: A Benchmark for Long-Horizon Audio-Visual-Language Navigation
- Link: https://arxiv.org/abs/2607.03920 (v2; found by citation chasing from NaVLA2)
- Read depth: raw HTML v2: Sec. 1, task definition, Table 3 and Appendix A.1 (MAV-Nav) via targeted passages.
- Setting: navigation, simulated (SoundSpaces 2.0, Matterport3D), multi-goal.
- Predictive world model: no. MAV-Nav is "an audio-visual actor-critic policy for lifelong navigation" (Appendix A.1.1).
- What audio: binaural ("navigates with RGB-D observations, pose, and binaural audio").
- The audio comparison: Table 3 (550 validation episodes, ordered protocol SR / ISR / OSR / SPL / SoftSPL, %): MAV-Nav 2.2, 18.6, 45.9, 0.5, 11.9; "w/o Audio" 0.9, 8.0, 33.5, 0.4, 4.7.
- What it does NOT test: a world model; mono against binaural.
- How it differs: a with-against-without-audio navigation result for an RL policy, no predictive model.

## Checked and not relevant
- RoboOmni (arXiv 2510.23763): omni-modal VLA with speech and event sounds; compares end-to-end audio against text and ASR-transcribed prompts; no predictive world model, no spatial audio.
- ListenVLA (arXiv 2610.16313): audio-conditioned VLA on GR00T N1.7; Table III has "A is GR00T without audio"; a policy, no world model, no channel comparison found by keyword search.
- ADAPT (arXiv 2610.15956): RGB-free navigation with first-order Ambisonics and depth, trained with RL; a policy, no world model.
- MuSTMM (arXiv 2610.21429): sound-triggered mobile manipulation dataset with binaural RIRs and a navigation policy with spatial memory; no world model (abstract and keyword counts).
- Generalizable AV navigation with Binaural Difference Attention and Action Transition Prediction (arXiv 2604.05007): "action transition prediction" is an auxiliary action-prediction loss, not observation prediction; policy only (abstract).
- Multisensory Continual Learning / MuSe (arXiv 2606.30988): adds force-torque, not audio, to a pretrained world-action model (abstract; audio named only as a motivating modality).
- Survey of World Models Grounded in Acoustic Physical Information (arXiv 2506.13833): seen in the arXiv listing only; title suggests a survey, not an experiment; not opened, so no claim made.
- WorldSonus (arXiv 2610.08760), Audible World Models (2609.38444), HelixWorld (2609.38123), EchoWM (2608.23189), LiveFoley (2611.05988): generate sound for explorable or generated worlds; no agent acting and no task success (abstracts; LiveFoley text keyword-checked).
- BinauralVAE (2609.06837), AudioWorldSim (2608.21075): audio reconstruction and dataset reports for world models; no policy comparison (titles and survey file D notes).
- MLA (2509.26642), Dynin-Robotics (2609.13053), InternW0 (2609.27656), Dreaming when Necessary (2606.07089), From WAMs to Embodied Brains roadmap (2607.11689): zero or one hit for audio, sound, acoustic or binaural in the full text.
- OmniAct / "Advancing Omnimodal Embodied Agents" (2606.27251): audio appears only as an IoT "play_audio" tool.
- MuseVLA (2606.17598), ACE-Data-0 (2607.28625), MultiModal Action Conditioned Video Generation (2510.02287), Late Breaking Results: Audio-Language-Action Models (2610.33866): audio is a sensing or speech-instruction modality in a policy or dataset, or a minor mention; no world model with an audio ablation (keyword counts and abstracts).
- Say, Dream, and Act (2602.10717): video world model for instruction-driven manipulation; zero hits for audio, speech, sound or acoustic.
- World action model surveys: Wang et al. (2605.12090) cite only datasets with audio (Berkeley UR5, OmniAction) in Sec. 5.1.2; Shen et al. (2606.20781) list Audio-WM as the one acoustic case ("Acoustic observation latents are rare in the current literature"); the 2609.16074 survey cites only Audio-WM ("Existing work either incorporates these sensors directly [vtla, audio_wm] or predicts them from visual inputs [vt_wm]"). Each survey's full text was grepped for audio, acoustic, sound and auditory.

## Could not open
- Semantic Scholar citations for Audio-WM (arXiv:2512.08405): rate-limited (HTTP 429) on the first call and on the retry after a wait. Not chased.
- Semantic Scholar has no record for HEAR as arXiv:2603.16086 ("Paper with id arXiv:2603.16086 not found"); a title search was rate-limited. Not chased.
- OpenReview: the search API answered (one attempt per query, three queries), but the hits were review and comment notes without titles, and a batch title lookup by note id returned no notes. No OpenReview paper was opened.
- Seedance 2.0 (the generator in ji2026dreaming) and the SoundAct and Preprints.org leads named in the earlier survey files were not opened in this check.

## Searches run
Semantic Scholar citations API (`/graph/v1/paper/<id>/citations?fields=title,year,externalIds,abstract&limit=200`), citing papers scanned by title and abstract for audio, sound, acoustic, binaural, auditory, hearing or microphone:
- arXiv:2512.00883 (AVWM): 1 citing paper (LiveFoley), opened.
- DOI:10.1609/aaai.v40i22.38886 (NaVLA2): 2 citing papers (LH-AVLN, opened; Safe-Night VLA, thermal, not relevant).
- arXiv:2605.12090 (WAM survey, Wang et al.): 60 citing papers; none flagged by abstract; titles scanned; three opened and keyword-checked (2606.27251, 2606.07089, 2607.11689).
- arXiv:2609.16074 (WAM survey): 7 citing papers, none audio.
- arXiv:2607.26047 (S2A2): 0 citing papers.
- arXiv:2512.08405 (Audio-WM): rate-limited twice. arXiv:2603.16086 (HEAR): not in Semantic Scholar.

arXiv API listing search (`export.arxiv.org/api/query?search_query=...&max_results=50&sortBy=submittedDate`), results restricted to 2025 and 2026 (count = total results / kept 2025-26):
1. abs:"world model" AND abs:audio AND (abs:navigation OR abs:robot OR abs:manipulation OR abs:embodied): 14 / 14
2. abs:"world model" AND abs:binaural: 3 / 3
3. abs:"world model" AND abs:sound AND (abs:robot OR abs:navigation OR abs:agent): 9 / 7
4. abs:"world action model" AND (abs:audio OR abs:sound OR abs:acoustic): 2 / 2
5. abs:"world-action model" AND (abs:audio OR abs:sound OR abs:acoustic): 2 / 2
6. abs:"video prediction" AND (abs:audio OR abs:sound) AND (abs:robot OR abs:navigation OR abs:policy): 1 / 0
7. abs:"predictive model" AND (abs:audio OR abs:auditory) AND (abs:robot OR abs:navigation): 3 / 2
8. abs:"world model" AND abs:acoustic AND (abs:robot OR abs:navigation OR abs:embodied): 6 / 6
9. abs:"world model" AND abs:auditory: 6 / 3
10. abs:"audio-visual" AND abs:"world model": 11 / 11
11. abs:"spatial audio" AND (abs:policy OR abs:"world model" OR abs:navigation OR abs:robot): 21 / 14
12. abs:"video generation" AND (abs:audio OR abs:sound) AND (abs:"robot policy" OR abs:navigation OR abs:manipulation): 12 / 11
13. abs:"vision-language-action" AND (abs:audio OR abs:sound OR abs:binaural): 21 / 21
14. abs:"audio-visual navigation" AND (abs:imagination OR abs:predict OR abs:planning OR abs:"world model"): 8 / 6
15. abs:"model-based" AND (abs:audio OR abs:sound) AND (abs:navigation OR abs:robot) AND abs:reinforcement: 1 / 0
16. abs:"video action model" AND (abs:audio OR abs:sound): 0 / 0
17. abs:multisensory AND abs:"world model": 6 / 6
18. abs:omnimodal AND (abs:robot OR abs:policy OR abs:action): 16 / 16
(A first run of the same queries through a Python HTTP client returned nothing because of a client error; it was discarded and rerun with curl.)

Web searches (standard mode, 2026-11-16), each returning 10 links, scanned:
1. world model with audio improves navigation success ablation "without audio" world model planning agent 2026 arXiv
2. video world model robot policy audio input ablation spatial audio mono stereo world action model 2026
3. Dreamer world model audio observations sound navigation reinforcement learning imagination binaural

OpenReview API (`api2.openreview.net/notes/search?term=...`), one attempt each: "audio world model navigation", "audio-visual world model", "world action model audio" (25 notes each, titles not returned; see Could not open).

Survey audio sections: full-text grep of 2605.12090 (PDF), 2606.20781 (HTML) and 2609.16074 (HTML) for audio, acoustic, sound and auditory; the papers they cite there were opened (Audio-WM, RoboOmni) or are datasets (Berkeley UR5).

Direct reads: all eight papers in the ranked list from raw text to the depth stated in each entry, including all four arXiv versions of 2512.00883.
