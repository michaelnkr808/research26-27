# Novelty of the three candidate research questions, and the WNM-3D simulator gap (2026-11-18)

Direction: hear-lab (spatial audio for world action models). Requested by the planning lead on 2026-11-18, at the user's ask: a novelty check on the three candidate research questions in `LOG.md` (2026-11-18), and a look into the simulator gap for WNM-3D, including the user's question "is the reason we can't reuse the NaVLA paper's benchmark because its a vla?". Part 1 extends `2026-11-16-prior-work-check-audio-in-world-action-models.md` and does not redo it. Part 2 extends the WNM-3D, AstraNav-World and NaVLA2 entries in `2026-11-16-navigation-world-models.md` and `2026-11-16-spatial-audio-navigation-data.md`.

Written by two literature-session research workers (one per part) and assembled by the literature lead; each part keeps its own method note. Labels: `verified` = read in the named source this session; `recalled` = from an earlier survey file or memory, not reopened; `speculative` = this file's own reading. Downloads were kept in a scratch directory outside the repository, treated as untrusted, and never executed. Numbers are copied from sources; none were reproduced.

## Part 1: novelty of the three questions

Direction: hear-lab (spatial audio for world action models). Written by the literature agent. This file extends `2026-11-16-prior-work-check-audio-in-world-action-models.md` (called "the prior check" below) and does not redo it. Papers in the prior check are re-opened only for the specific points named. `verified` means read this session in the named source (arXiv HTML or PDF text, converted locally; downloads kept in the scratchpad, never executed). `recalled` means taken from the prior check or a survey file without re-reading the paper. `speculative` is this file's own reading. Numbers are copied from the source text; none were reproduced. The arXiv listing API was rate limited for most of the session (see "Searches run"), so the sweeps for new papers lean on web search plus opening each hit; a missing paper may exist.

### Verdicts

**Q1 (spatial vs mono vs no audio in a Wan 2.2 navigation world action model, split by whether the target is visible at the start; second measure: predicted frames for out-of-view regions): not found, as stated.** No paper compares spatial audio, one-channel audio and no audio inside any world model or world action model, and none gives audio to a large pretrained video model for navigation (the prior check's finding, and nothing new found since 2026-11-16). The only spatial-versus-one-channel comparison in an acting agent remains NaVLA2 (a vision-language-action policy, two different audio encoders, so not channel count alone). No audio-visual navigation paper opened reports success split by target visibility. In the original SoundSpaces the target has "no visual presence" by construction, so the split cannot exist there; the setting would have to be built (a visible sounding object, or a rendered one). The second measure is also open: the one navigation world model that predicts frames and audio (Multisensory Foresight, arXiv 2109.07561, also predicts future audio as a side task, but in robot manipulation, per the fact-check review on PR #68) (Audio-Visual World Models, v3) shows frame quality about the same as its vision-only base model, not split by visibility. The nearest "audio helps beyond the camera view" evidence is Audio-Visual Floorplan Reconstruction (audio plus images give a better top-down map of unseen areas than images alone), which is a map, not predicted frames, and has no mono comparison.

**Q2 (Q1 plus: does a spatial audio encoder that ranks better on a spatial audio benchmark also navigate better): partly done at the edges, the claim itself not found.** I found no work that ranks audio encoders by an acting task and relates that ranking to an upstream spatial-audio benchmark. What exists: ManiWAV compares three audio encoders by real-robot manipulation success (no spatial audio, no upstream benchmark); NaVLA2 switches its semantic and spatial audio branches on and off in navigation (not an encoder comparison with matched everything else); OmniEcho ablates its spatial (first-order Ambisonics) encoder, but on a question-answering benchmark, not on navigation; and a new probing benchmark for spatial audio encoders (SARL) states it deliberately avoids downstream tasks. The vision side has clear precedents (Burns et al., 2023; a 2026 probing study; a WAM study that swaps visual encoders under one framework), which show that upstream rank does not always predict robot success. Those are vision, not audio.

**Q3 (does the model need to predict future sound, or is hearing it enough): partly done, in two different settings, neither a navigation world action model.** In Audio-Visual World Models v3 (navigation, small world model, fixed VLM planner), removing the audio part of the world model's forecast took success back to the no-world-model level, so only the joint audio-visual forecast helped; the agent's own audio input is apparently unchanged (inferred, not stated). That row is gone from the current version 4. HEAR (manipulation, one microphone, a vision-language-action policy) removes its audio-prediction module and keeps audio as input: simulation average success fell from 0.81 to 0.73. Audio-WM claims prediction matters but only compares with and without future lookahead. So an input-only versus predicted-audio ablation exists in manipulation (HEAR) and in a different form in navigation (AVWM v3); none exists for a pretrained video world action model, and none with spatial audio.

### Q1. Does direction matter, and does audio help the unseen part of the scene

#### chen2020soundspaces: SoundSpaces: Audio-Visual Navigation in 3D Environments
- Link: https://arxiv.org/abs/1912.11474 (version read: the ar5iv rendering of the arXiv HTML, which follows the latest version; the arXiv abstract page lists v3, last revised 21 Aug 2020).
- Read depth: full text through ar5iv; keyword passes for visibility, mono, intensity, ablation.
- Setting: navigation, simulated (Replica, Matterport3D), AudioGoal and AudioPointGoal.
- What they did: binaural audio (ambisonics converted to binaural), spectrogram input, RL agent. Ablations: an "audio intensity baseline" where the audio CNN is replaced by the left and right waveform intensities; a modality-importance analysis that replaces each modality with its average training sample.
- Target visibility: `verified`. "A target has diameter 0.2 m and height 1.5 m, and, consistent with prior PointGoal work, has no visual presence. While the goal itself does not have a visible embodiment ... vision ... is essential to detect and avoid obstacles". The paper states that AudioGoal "captures scenarios where a target initially out of view makes itself known aurally". So the target is always unseen by construction, and no results are split by visibility.
- Did NOT do: mono-against-binaural; a visible-target condition; a world model.

#### chen2022soundspaces2: SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning
- Link: https://arxiv.org/abs/2206.08312 (v2, 23 Jan 2023, PDF read).
- Read depth: full text converted with pdftotext; keyword passes (mono, visible, occlu).
- What they did: simulator offering "mono, stereo, binaural, quad, surround_5_1, surround_7_1 and ambisonics" microphone types ("We provide seven built-in microphone types"). `verified`.
- Did NOT do (in the text read): any navigation comparison across channel layouts. The word "occluded" appears only for acoustic diffraction. Useful for the user's data pipeline: mono and binaural renderings of the same episodes are available from the simulator.

#### chen2021savi: Semantic Audio-Visual Navigation
- Link: https://arxiv.org/abs/2012.11583 (ar5iv rendering of the arXiv HTML read, text of the final version).
- Read depth: keyword passes only (visible, occlu, emit, mono).
- `verified`: it states that in earlier simulators "the sound emitting target has neither a visual embodiment nor any semantic context", and it adds semantically grounded objects in Matterport3D. No results split by whether the object is visible were found by keyword search; no mono comparison.

#### gan2019lla: Look, Listen, and Act: Towards Audio-Visual Embodied Navigation
- Link: https://arxiv.org/abs/1912.11684 (ar5iv rendering read; keyword pass only). It describes its goal as a sound source "that might also be non-line-of-sight in the virtual environment". No split by visibility and no mono comparison found by keyword search (visible, invisible, monaural, mono: no hits). `verified` for the absence of those words in the text read.

#### wang2025agvp: Audio-Guided Visual Perception for Audio-Visual Navigation
- Link: https://arxiv.org/abs/2510.11760 (v1, 13 Oct 2025, HTML read, keyword passes plus the ablation and qualitative sections).
- What they did: audio-guided attention over visual features; the paper motivates it with targets that are "occluded or invisible" and gives one qualitative example (apartment4, sound source occluded by walls). Table III ablates its own modules (self-attention, guided attention) on Replica and Matterport3D.
- Did NOT do: a quantitative visible versus occluded split (the only occluded case is one scene example); mono versus binaural. `verified`.

#### li2025agsa and the April 2026 pair (2604.02389, 2604.02390)
- Audio-Guided Dynamic Modality Fusion with Stereo-Aware Attention (arXiv 2509.16924 v1, 21 Sep 2025, HTML); Audio Spatially-Guided Fusion (2604.02389 v1, 2 Apr 2026, HTML); Spatial-Aware Conditioned Fusion (2604.02390 v1, 2 Apr 2026, HTML). Read: abstracts, introductions and ablation sections.
- What they did: new fusion modules for SoundSpaces AudioGoal. The first has a "Stereo-Aware Attention Module" that models the left-right channel difference, and reports an audio-only evaluation condition (agent without vision). Ablations remove their own modules.
- Did NOT do: a one-channel input condition (no hit for mono or single-channel in the texts read), or a split by target visibility. `verified` (keyword passes; ablation sections read).

#### fan2026navla (NaVLA2): re-checked for visibility
- Link: https://tanmingkui.github.io/files/publications/NaVLA2.pdf (AAAI-26; PDF re-read for keywords this session).
- `verified`: Table 2 ("Ablation study of audio encoder design") switches a semantic branch (binaural downmixed to mono, CLAP) and a spatial branch (SpatialAST) on and off; "spatial only" gives a larger gain than "semantic only" (SR 0.258 against 0.219, "neither" 0.206). All evaluation is "in previously unseen environments"; no results are split by whether the goal is in view (keyword passes for visible, out of view, line of sight, occlu found only generic uses). Not a world model. The two branches use different encoders, so the table is not channel count alone (as in the prior check).

#### liu2026omniecho: OmniEcho: Audio-Visual Spatial Understanding for Omni-Modal Embodied Agents (new to the survey files; named only as an unopened reviewer lead in the prior check)
- Link: https://arxiv.org/abs/2609.23407 (v2, 23 Sep 2026; HTML read: abstract, benchmark sections, Sec. 5.2.2 navigation, Sec. 5.3 ablation, appendix data generation).
- Setting: perception (OmniEchoBench-QA, 2,972 QA pairs) and real-scanned indoor navigation (OmniEchoBench-Nav, 30 scenes, 900 tasks) with first-order Ambisonics (FOA).
- What they did: an FOA spatial audio encoder added to Qwen3-Omni; the QA benchmark "covers audio-visual scenarios involving visible, occluded, and dynamically changing sound sources", with "2 to 4 off-screen sources" added to simulated clips and per-timestep visibility flags in the real recordings; tables label settings "A-single" (single-channel audio), "A-foa" and "V". For navigation it compares to text-guided VLN models and to "SoundSpaces with binaural audio".
- Table 4 (QA, not navigation): both encoders plus frozen FOA encoder best overall (28.5); the native-audio-only variant lower; removing the native encoder lowers overall to 26.2; training the FOA encoder in Stage 3 hurts.
- Did NOT do (in the sections read): a navigation ablation of single-channel against FOA audio; a world model; frame prediction; a navigation result split by whether the sound source is visible (the visible/occluded split exists in the perception benchmark's data design, and I did not find it reported as a navigation breakdown). `verified` for what is stated, `speculative` that no navigation split exists elsewhere in the appendix (appendix A.5 was only skimmed by its headings).
- Why it matters: the closest new work on "direction when the source is off screen", for perception and an LVLM navigator, not for a world model.

#### wang2025avwm: Audio-Visual World Models, re-checked for the frame-prediction measure
- Link: https://arxiv.org/abs/2512.00883 (v3, 5 Jun 2026, PDF re-read; the arXiv abstract page still lists v4 of 4 Aug 2026 as the latest, so nothing new since 2026-11-16).
- `verified` (v3 Table 2, "Comparison with modality-factorized baselines ... mean and standard deviation over three runs"): the vision-fidelity row shows LPIPS 0.382 for both NWM (visual-only) and AV-CDiT; the text says the comparison "should be interpreted together with the modeling scope: NWM is a visual-only world model, whereas AV-CDiT additionally introduces native synchronized audio". So the frame-quality gain from adding audio, in the one world model that has both, is not visible in this table, and no result is split by whether a region is in view.
- Did NOT do: out-of-view analysis; mono variant.

#### purushwalkam2021avmap: Audio-Visual Floorplan Reconstruction
- Link: https://arxiv.org/abs/2012.15470 (v1, HTML read: abstract, introduction, baselines and Table 1 passages).
- What they did: a few egocentric glimpses plus audio (device-generated echoes or ambient sounds from other rooms) are encoded to predict a top-down floorplan, including unseen rooms. The abstract states "Audio not only helps sense geometry outside the camera's field of view, but it also reveals the existence of distant freespace". Ablations: RGB-only, audio-only and audio-visual models; the audio-visual model is best on the interior-reconstruction metrics in the table read (first metric column "AP": 73.67 against 69.05 RGB-only and 70.44 audio-only).
- Did NOT do: predicted video or frames, actions, a world model, navigation success, a mono comparison (no hit for mono or monaural). `verified`.
- Why it matters: the best precedent for "audio improves what the model infers about regions outside the camera view". It is a 2020 map-prediction result, so it supports the second measure's plausibility, not its novelty.

#### xu2023avvp: Motion and Context-Aware Audio-Visual Conditioned Video Prediction
- Link: https://arxiv.org/abs/2212.04679 (v3, 20 Sep 2023; PDF text read: abstract, keyword pass).
- What it does: predicts future frames of a video from audio and visual context ("audio-motion correlation"). No hit for out of view, off-screen, invisible or occluded. Not embodied, no action conditioning, no spatial audio. `verified` (keyword pass). Sound2Sight (arXiv 2007.12130) appeared in the same search and was not opened; no claim is made.

#### Other items checked for Q1 and not relevant, or only motivating
- BinauralVAE (2609.06837, HTML read: motivation text): states that sound "conveys information about scene geometry, material properties, and out-of-view dynamic events" and motivates spatial audio for world models under occlusion or blackout; it is a reconstruction paper (as in the prior check), with no frame prediction or navigation comparison found in the keyword pass.
- RAO-Nav (2609.32224, HTML read: abstract, mono keyword pass): zero-shot semantic audio-visual navigation with an omni-language model; uses a binaural localization expert; no mono ablation found.
- Great Listeners, Poor Decision-Makers (2611.23847, 18 Nov 2026): voice agents' decision layer, not embodied; irrelevant.
- Not opened: HelixWorld (2609.38123), EchoWM, AudioWorldSim (as in the prior check, abstract-level only).

### Q2. Encoder comparison

#### nie2025maniwav: ManiWAV: Learning Robot Manipulation from In-the-Wild Audio-Visual Data
- Link: https://arxiv.org/abs/2406.19464 (v2, 4 Nov 2024; HTML read: audio encoder, baselines, ablation text).
- What they did: compare audio encoders by real-robot task success. Their own audio encoder is an AST trained from scratch; baselines are "ResNet: uses a ResNet18 encoder to encode the audio log-mel spectrograms" and "AVID: ... a 9-layer CNN audio encoder pre-trained on AudioSet using Audio-Visual Instance Discrimination". Their finding: "training a transformer encoder from scratch yields good performance compared to using a CNN-based encoder, as shown in the [ResNet] and [AVID] baseline". Tasks: flipping, wiping, pouring and taping (per-task tables).
- Did NOT do: spatial audio (the paper describes no multi-channel spatial setup in the passages read); any upstream audio benchmark; a rank correlation; navigation or a world model. `verified`.
- Why it matters: it shows the practice of comparing audio encoders by downstream acting exists, but only as a design ablation, not as "does upstream rank predict downstream success".

#### fan2026navla (NaVLA2) and liu2026omniecho: see Q1.
- NaVLA2 Table 2 compares branches, not encoders under one interface. OmniEcho Table 4 compares encoder configurations by QA score only, not acting.

#### chen2026sarl: Probing Spatial Structure in Pretrained Audio Representations (SARL)
- Link: https://arxiv.org/abs/2606.05544 (HTML read: abstract, introduction).
- What they did: probe frozen embeddings of pretrained audio encoders for azimuth, elevation, distance, class, and room factors. `verified`: they argue that "spatial representation evaluation should move beyond task-centric benchmarking toward probing-based analysis" and that probing "isolat[es] representation quality from downstream model design".
- Did NOT do: any acting task. It is an upstream benchmark of exactly the kind Q2 would relate to navigation success, and it chooses not to.

#### li2026listenvla: Listen Before Act (ListenVLA)
- Link: https://arxiv.org/abs/2610.16313 (HTML read: abstract, architecture, headings). Uses frozen CLAP embeddings plus waveform-derived cues; ablations are on its binding objective and audio modulation (abstract). No audio-encoder comparison found at the abstract and heading level; the full ablation table was not read. `verified` for the abstract and architecture statements, `speculative` for the absence of an encoder swap.

#### Vision-side precedents (vision, not audio)
- burns2023what: What makes pre-trained visual representations successful for robust manipulation? https://arxiv.org/abs/2312.12444 (v1, 3 Nov 2023, PDF read: abstract). They "compare the performance of 15 pre-trained vision models", and find that "emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models" and "more predictive than ... downstream ImageNet accuracy". `verified` (abstract).
- Capturing Visual Environment Structure Correlates with Control Performance, https://arxiv.org/abs/2602.04880 (HTML read: abstract): probing accuracy for decoding environment state "strongly correlates with downstream policy performance", "significantly outperforming prior metrics". `verified` (abstract).
- ALPS: Visual State Encoders for World Models, https://arxiv.org/abs/2610.30120 (HTML read: abstract): fifteen public encoders in "a unified evaluation" with state-probe error, then downstream policy success on LIBERO and world-model planning. Whether it states a rank correlation was not found by keyword search ("correlat" gave only ranking of motion windows). `verified` for the abstract.
- V-JEPA Policy, https://arxiv.org/abs/2609.37250 (HTML read: abstract): builds a world-action model on a frozen visual encoder and compares "visual foundations under the same downstream framework and training budget". The nearest vision precedent specific to world action models. `verified` (abstract).

#### Not opened (so no claim)
RMS-AQA, SPUR, BAT's SpatialSoundQA and STARSS were not opened this session; whether any of them ever reports a downstream acting result is therefore unchecked. RepWAM (2606.13674) was seen only in a search listing.

### Q3. Predict future sound, or hear it

#### wang2025avwm (v3): what exactly the Table 4 "w/o audio" row removes
- Link: https://arxiv.org/abs/2512.00883 (v3, 5 Jun 2026; PDF re-read: Sec. 4.4, Table 4, Appendix G prompts).
- `verified`: "We compare four configurations: a Baseline Nav-Agent without a world model, the full AVWM Nav-Agent, two single-modality variants (w/o audio and w/o vision), and an Oracle WM Nav-Agent". Table 4 success: baseline 0.269, w/o audio 0.266, w/o vision 0.280, AVWM 0.381, Oracle WM 0.423; "500 randomized test episodes". Forecasts reach the planner as an injected predicted RGB image and a "<future audio summary text>" in the prompt (Appendix G), for a small subset of candidate waypoints.
- What this means: "w/o audio" drops the audio part of the forecast. It does not say whether the world model was retrained without audio or only had its audio output dropped from the prompt, and it does not say what the Baseline, "w/o audio" and "w/o vision" agents keep as their own audio observation (the prior check inferred they keep binaural input; still not stated in the text re-read, `speculative`).
- Not the clean test of Q3: a model trained to predict audio is not isolated from a model that only hears; the comparison is forecast with both modalities against forecasts with one, inside a fixed VLM planner. No seed count; only "t-tests (p < 0.05)".
- Gone in v4 (prior check, `recalled`).

#### nie2026hear: HEAR (arXiv 2603.16086), re-checked for the Advancer ablation
- Link: https://arxiv.org/abs/2603.16086 (v2, PDF read again: Table 4 and the module descriptions).
- `verified`: Table 4, "Component ablations in HEAR-Bench ... average success rate across all seven simulation tasks", 100 trials per task per the text: HEAR full 0.81, "No Advancer" 0.73 (other rows: no pretraining 0.69, no Historizer 0.57). The Advancer is "a decoder-only transformer that predicts a near-future audio code sequence ... during training"; it "provides a predictive learning signal that shapes the latent representation"; "At deployment, HEAR does not need to explicitly generate the future audio codes". Audio stays an input in the "No Advancer" row.
- Did NOT do: spatial audio (single microphone, per the prior check), navigation, a video world model; no seeds or intervals found in the passages read; the ablation changes one auxiliary loss and the rest of the system is held fixed ("each ablation uses the same demonstrations ..."). This is the closest input-only versus predicted-audio ablation in an embodied agent. It is an auxiliary training signal, not a model that outputs audio at run time.

#### zhang2025audiowm: Audio-WM (arXiv 2512.08405 v1, 9 Dec 2025, PDF read)
- `verified`: it claims success "relies not merely on multi-modal input, but critically on the accurate prediction of future audio states" (abstract), but the evidence is "methods without future lookahead": the piano task compares the world-model framework "against a basic RL baseline without future lookahead" (F1, no number in the text), and water filling reports only the full system (as in the prior check). No row has audio as input only against audio as input plus predicted output. Mono microphone, manipulation.

#### Auxiliary prediction in navigation: Learning Semantic-Agnostic and Spatial-Aware Representation (arXiv 2304.10773)
- HTML read: abstract and introduction. Two auxiliary tasks for audio-visual navigation: an adversarial audio-category task and a task that predicts "the relative direction of a sound". Prediction of a direction label, not of future audio; results not read in detail. Mentioned only because it is an auxiliary prediction signal added to an audio-visual navigation agent; no claim about its effect size.

#### Vision-side analogues (vision, not audio)
- What Should Robot Policies Learn to Predict? (arXiv 2610.21915, HTML read: abstract): action-only training against added future-vision (FV) and latent-action targets under a fixed budget; FV "generally outperforms" latent action targets. DreamWAM (arXiv 2608.04996, abstract): beyond-RGB future-prediction heads used in training and "disabled at inference". These show the "predict extra futures only at training time" design in vision WAMs.

### Closest prior work, one line each
- Q1: NaVLA2 (spatial against mono, policy, different encoders); Audio-Visual World Models v3 (audio forecast in a small world model, no mono, no visibility split); Audio-Visual Floorplan Reconstruction (audio helps unseen regions, as a map); OmniEcho (off-screen sources in a perception benchmark, no navigation split).
- Q2: ManiWAV (three audio encoders compared by manipulation success); SARL (upstream probing only); vision: Burns et al., the probing-versus-control study, V-JEPA Policy.
- Q3: AVWM v3 Table 4; HEAR Table 4 ("No Advancer"); Audio-WM (with against without lookahead).

### Could not do or open
- arXiv listing API (export.arxiv.org): answered "Rate exceeded" for most calls from 2026-11-18 evening; the few that returned are listed below. Semantic Scholar was not tried again (429 in the prior check).
- A broad, dated sweep for papers newer than 2026-11-16 therefore rests on web search and on the IDs it surfaced (2611.19593 spatial audio-visual features from mono video, no acting task; 2611.23847 voice agents; both irrelevant to the three questions).
- The three RMS-AQA-type benchmarks, Sound2Sight, RepWAM, HelixWorld, EchoWM and AudioWorldSim were not opened.

### Searches run
arXiv API (export.arxiv.org), 2026-11-18/19; counts are totalResults when the call returned:
1. abs:"audio-visual" AND abs:navigation AND (abs:monaural OR abs:mono OR abs:"single channel"): 0 results.
2. abs:"sound source" AND abs:navigation AND (abs:occluded OR abs:"out of view" OR abs:"not visible" OR abs:invisible): 3 results (2509.16924 opened; SonicBoom and the disaster-victim paper not relevant).
3. abs:audio AND abs:"world model" AND (abs:ablation OR abs:"success rate"): 1 result (2609.18323, an omni-modal model evaluation; not opened beyond the title).
4. abs:"world action model" AND (abs:audio OR abs:sound OR abs:acoustic): rate limited on every attempt this session (the prior check's version of this query found 2 results).
5. abs:"world model" AND (abs:binaural OR abs:"spatial audio") AND (abs:navigation OR abs:embodied OR abs:robot): rate limited.
6. abs:"audio encoder" AND (abs:robot OR abs:navigation OR abs:manipulation) AND (abs:compare OR abs:comparison OR abs:benchmark): rate limited / no result returned.
7. Further queries on "audio-goal", SoundSpaces + binaural + ablation, out-of-view video prediction, audio-conditioned video prediction, "audio" + "world action model", "audio" + "world model" + navigation: rate limited, no result.
8. arxiv.org/search HTML page for "audio-visual navigation monaural binaural": returned no parsable results.

Web searches (each returned about 10 links; every relevant hit was then opened):
- audio-visual navigation SoundSpaces target visible vs not visible success rate audio goal analysis
- binaural vs monaural audio ablation audio-visual navigation SoundSpaces AudioGoal
- audio-visual world model navigation 2026 spatial audio video prediction out-of-view sound source
- audio-visual navigation spatial audio vs mono audio ablation embodied agent 2026 arXiv
- audio-visual navigation sound source visible versus occluded success rate breakdown
- audio-conditioned video prediction occluded or out-of-frame sound source future frames
- world action model audio spatial audio navigation Wan 2.2 arXiv November 2026
- comparing audio encoders downstream robot policy success pretrained audio representation manipulation benchmark
- which audio encoder for robot learning audio representation comparison CLAP BEATs AudioMAE policy success
- spatial audio encoder embodied navigation comparison SpatialAST BAT downstream navigation
- do benchmark rankings of pretrained encoders predict downstream robot manipulation success correlation visual representation
- future audio prediction ablation audio as input only versus predicted audio embodied agent
- audio-visual world model embodied agent predicts audio ablation without audio prediction 2026
- pretrained audio encoder comparison audio-visual navigation SoundSpaces CLAP AST BEATs wav2vec downstream success
- self-supervised audio representation embodied reinforcement learning benchmark comparison audio encoders agent
- audio representation for robotics benchmark evaluate audio encoders on robot tasks
- spatial audio representation learning evaluated on navigation downstream task
- audio encoders robot manipulation ablation AST versus CLAP versus BEATs ManiWAV follow-up
- visual encoder comparison for world action model downstream success vs perception benchmark rank
- audio-visual navigation sound source in view vs out of view agent performance analysis visible target
- SoundSpaces semantic audio-visual navigation object visible in view audio helps
- audio generation or prediction as auxiliary objective robot policy ablation predicting future audio tokens
- video world model conditioned on binaural audio navigation predicted frames behind the agent
- Wan 2.2 audio conditioning world model robot navigation sound
- world action model "audio" 2026 spatial hearing navigation arXiv November
- audio helps predict unseen regions of scene from sound out-of-view occupancy map prediction binaural echoes embodied
- sound-conditioned novel view synthesis or scene completion off-screen audio visual prediction embodied agent
- audio-visual world model predicts off-screen sound source events video generation off-screen
- audio-visual navigation "audio-visual" "world model" imagination planning binaural SoundSpaces 2026
- audio-visual navigation foundation model video generation pretrained spatial audio Wan Cosmos agent
(One search result page was the user's own repository PR page; ignored.)

Papers opened (text read, versions as stated above): SoundSpaces 1912.11474; SoundSpaces 2.0 2206.08312 v2; SAVi 2012.11583; Look, Listen and Act 1912.11684; AGVP 2510.11760 v1; 2509.16924 v1; 2604.02389 v1; 2604.02390 v1; OmniEcho 2609.23407 v2; BinauralVAE 2609.06837; RAO-Nav 2609.32224; AV-Map 2012.15470 v1; Xu et al. 2212.04679 v3; AVWM 2512.00883 v3 (and the v4 abstract page); HEAR 2603.16086 v2; Audio-WM 2512.08405 v1; NaVLA2 (AAAI-26 PDF); ManiWAV 2406.19464 v2; SARL 2606.05544; ListenVLA 2610.16313; SASR 2304.10773; Burns et al. 2312.12444 v1; 2602.04880; ALPS 2610.30120; V-JEPA Policy 2609.37250; 2610.21915; DreamWAM 2608.04996; 2611.19593 v1; 2611.23847 v1; 2609.21792 (AcousticDiffusion, abstract only; irrelevant).

## Part 2: the simulator gap

Date: 2026-11-18 (session date). Direction: hear-lab (spatial audio for world action models). Requested by the planning lead for the user's question "why can we not reuse the NaVLA2 benchmark (MINav)?". Read-only research; nothing in the repository was edited.
Labels: `verified` = read in the named source this session; `recalled` = from memory or from an earlier survey file and not reopened; `speculative` = my inference. No hour estimates are given. Downloads (repository tarballs, the NaVLA2 PDF, the WNM-3D arXiv HTML) were kept in the scratchpad `downloads` folder, treated as untrusted, and nothing in them was executed.

### Short answer

**a. Why not MINav?** The model type is not the obstacle (`verified`/`speculative`): WNM-3D's server only asks for RGB frames, an optional language string and an optional pose, and returns planar (dx, dy, dyaw) blocks, so it does not care whether the thing being replaced is a VLA or a world action model. The obstacle is what MINav's release contains. NaVLA2's repository (last pushed 2026-03-14, checked 2026-11-18) ships the Habitat + SoundSpaces 2.0 benchmark code and the evaluation script, but its own README still lists the processed MINav dataset, the pretrained checkpoints and the training scripts as unchecked; two of the data-building scripts are 99-byte stubs; and the evaluation config points at an episode folder (`data/datasets/diff_bench/{split}/{split}.json.gz`) and a checkpoint folder that are not in the repository. A Hugging Face search for MINav and NaVLA found nothing from these authors. So the MINav evaluation environment exists as code, but there are no released episodes to evaluate on. A second, smaller reason: MINav is a different task (an instruction chain mixing category, image, language and a sound description, discrete 25 cm and 30 degree actions, success within 3 m), so even with episodes, WNM-3D's continuous output would need a converter. The planning lead's provisional answer holds, with that addition.

**b. Running WNM-3D on an existing Habitat + SoundSpaces benchmark.** Nothing off the shelf is a drop-in. What exists: WNM-3D's server speaks a simulator-agnostic WebSocket protocol (RGB in, language optional, nav_delta out); a Habitat-side VLN-CE-style client pattern exists in the same authors' GN0-VLN-CE repository (Habitat v0.1.7, no WebSocket client, no WNM-3D); MINav's environment (habitat-sim v0.2.3 built with audio, binaural microphones, discrete actions, 512x512 RGB) exists as code; SoundSpaces AudioGoal and SAVi episodes exist as downloadable data (but on the SoundSpaces 1.0 grid, 128x128 RGB, discrete actions, no instruction). What would have to be built: a Habitat client for WNM-3D's server (frames in, nav_delta out), a converter from (dx, dy, dyaw) blocks to the benchmark's actions (the WNM-3D paper describes one for VLN-CE but no released repository contains it), a fixed or generated language instruction for audio tasks, and a start-of-episode target-visibility flag. No benchmark found records whether the target is visible at the start; SAVi's episodes carry `view_points` (positions from which the object is visible) and MINav builds per-object views, so visibility could be computed from the episode and scene, but I did not find a ready field. Closest fit to "walk to a sounding object, sometimes out of sight": SAVi episodes (an object category plus its sound, 1 m success) in task shape; MINav audio episodes in rendering stack (SoundSpaces 2.0 on HM3D with binaural audio). Neither is complete.

**c. VLN-CE R2R in WNM-3D.** The paper's R2R result is Habitat-based by construction (VLN-CE is a Habitat benchmark, `recalled`; the paper's appendix B.2 says the Stage-II oracle uses `habitat_sim.ShortestPath` and `pathfinder.find_path`, and that episode horizon and execution settings "follow the official VLN-CE protocol", `verified`). WNM-3D was trained for that benchmark on the R2R and RxR training splits (appendix B.2). No code for it is released: the WNM-3D, GN0 and WNM-3D-RL repositories contain no Habitat or Matterport3D code (tarball grep, below). The same authors' GN0-VLN-CE repository is a Habitat VLN-CE evaluation harness, but for their GN-BAE model, and the GN0 README ties it to GN-BAE only. The released WNM-3D checkpoints are the GN-Bench ones: the model card says the policy "was developed and evaluated with the InteriorGS scenes and GN-Matrix task annotations", and the checkpoint's own training config lists only the InteriorGS LeRobot data. So the released weights were not trained on Habitat or Matterport3D data (`verified` from the config and the model card; the R2R-trained variant is not released).

**d. Is AstraNav-World or another Wan 2.2 navigation model closer?** Partly. AstraNav-World (Wan2.2-TI2V-5B, `recalled` from the survey entry; evaluated in Habitat) releases Habitat inference code for R2R, RxR (habitat v0.1.7) and OVON (habitat v0.2.3 fork) and an action-former checkpoint (link on ModelScope, not verifiable by me), but no training script for the Wan video part and no video-generator weights that I found in the repository tree. So its Habitat side exists, its training side is partly missing; it has no WebSocket server. UniWAM (Fysics AI; Wan2.2-TI2V-5B; navigation expert on HM3D in Habitat with a WebSocket policy interface; training code, data and checkpoints on Hugging Face, gated by an automatic access agreement) is the closest complete stack I found for "Wan 2.2 world action model that is trained and evaluated against Habitat through a WebSocket policy server". It is instruction-conditioned navigation to a pose near an object, with Habitat-Sim 0.3.3 and HM3D, not audio. Its Habitat version differs from SoundSpaces 2.0's branch, so audio rendering would still need the same integration work. Summary: the gap is not smaller for AstraNav-World than for WNM-3D; UniWAM's gap is somewhat different (its harness is already Habitat + HM3D + WebSocket, and it is the same scene set MINav uses), but it is a mobile-manipulation paper whose navigation task is not "walk to a sounding object". WNM-3D's own training code is released and complete for Stage I and II, which is its advantage.

### Entries

#### fan2026navla: NaVLA2 and MINav, release state re-checked
- Authors: Jugang Fan, Peihao Chen, Changhao Li, Qing Du, Jian Chen, Mingkui Tan
- Year and venue: AAAI-26 (the AAAI proceedings page https://ojs.aaai.org/index.php/AAAI/article/view/38886 opened; it shows the title and DOI 10.1609/aaai.v40i22.38886). I found no arXiv version (web search only; not an arXiv listing check).
- Links: paper PDF https://tanmingkui.github.io/files/publications/NaVLA2.pdf (opened, `pdftotext`); code https://github.com/felixfjg/NaVLA
- Read depth: PDF: abstract, Sec. 3.1 to 3.4 and the metrics paragraph (text-searched for visible, line of sight, out of sight, occlusion: zero hits). Repository: README (raw), file tree (GitHub trees API, 513 entries, branch main), `eval.sh`, `install_env_bench.sh`, `minavbench/config/tasks/vla2.yaml`, `minavbench/third_party/minav-bench/config/tasks/minav_audio.yaml`, the third-party MINav README, tarball keyword search, issue list, release list.
- What the task is (`verified`, PDF Sec. 3.1): the agent "randomly initialized in an unseen indoor environment and receives a multimodal instruction"; references are object category, image, language description or auditory description; the last object mentioned is the goal and earlier ones are landmarks; at each step the agent gets "an egocentric RGB image and binaural spatial audio"; actions are `A = {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}`, forward 25 cm, turns 30 degrees; success is stopping within 3 m of the goal object. Start and end positions: same floor, distance between 5 and 30 m (Sec. 3.3). Audio: the retrieved clip is inserted at the center of the referenced object and the audio sensor is only enabled in episodes with an auditory reference (Sec. 3.4).
- Simulator stack (`verified`): `install_env_bench.sh` installs habitat-sim and habitat-lab at `v0.2.3`, builds habitat-sim with `--headless --audio`, installs goat-bench, and installs sound-spaces. `minav-bench/README.md` in the third-party folder is a copy of the GOAT-Bench README, so the MINav environment is a fork of GOAT-Bench. `minav_audio.yaml`: task actions stop, move_forward, turn_left, turn_right; RGB sensor 512 by 512, hfov 90; success distance 3; dataset type `Minav-audiogoal` with data path `data/datasets/diff_bench/{split}/{split}.json.gz`. `vla2.yaml` (the config `eval.sh` uses) adds look_up and look_down actions, `vla_instruction_sensor`, `vla_audio_sensor`, and the same dataset path pattern.
- Release state as of 2026-11-18 (`verified`):
  - README TODO list, copied: "[x] Code Release", "[x] Data Collection: Released scripts for MINav data collection (`minavbench/build_datasets`)", "[x] Evaluation: Released evaluation scripts.", then unchecked "[ ] Reorganization", "[ ] Model Weights: Release pretrained NaVLA checkpoints.", "[ ] Dataset: Release the processed MINav dataset.", "[ ] Training: Release NaVLA training scripts."
  - GitHub API: pushed_at 2026-03-14, no releases, one open issue ("Training Device and Time", 0 comments).
  - Tree: `minavbench/build_datasets/2_construct_qa_dataset.py` and `3_construct_nav_dataset.py` are 99 bytes each (same size as the earlier survey recorded; the fact-check reviewer on PR #68 opened both and found only two empty functions in each, so "stubs" is `verified`). The tree has no episode data files and no checkpoint files. `eval.sh` expects `ckpt/navla/navla_stage3_v6.1`, which is not in the repository.
  - Hugging Face: API search for datasets "MINav" returned only `minthwi2505/minav-packs` (not connectable to these authors, empty of data per the earlier survey); models search "NaVLA" returned [], "MINav" returned unrelated repositories.
- Visibility of the target at the start: no field found. The only "visible" hit in the build scripts is `render_keep_visible` in `1_annotate_object.py`, which selects representative views of an object for annotation, not an episode property.
- What they did not do: release the dataset, the weights or the training code (per their own README); the paper does not report any split by target visibility.

#### chen2022soundspaces2 and the SoundSpaces repository (AudioGoal, SAVi): what the existing benchmarks are
- Link: https://github.com/facebookresearch/sound-spaces (archived, pushed_at 2023-09-29, CC-BY-4.0, per the GitHub API). Tarball of branch main read; README, INSTALLATION already known from the earlier entry.
- Read depth: README, `soundspaces/README.md`, `configs/audionav/av_nav/mp3d/audiogoal.yaml`, `configs/semantic_audionav/savi/mp3d/semantic_audiogoal.yaml`, `soundspaces/tasks/semantic_audionav_task.py` (episode and goal classes), `soundspaces/datasets/audionav_dataset.py` (header), `ss_baselines/common/environments.py` (continuous flag), keyword search for visibility over `soundspaces/` and `ss_baselines/`.
- AudioGoal (`verified`): config `audiogoal.yaml`: `TYPE: "SoundSpacesSim"`, `ACTION_SPACE_CONFIG: "v0"`, `GRID_SIZE: 1.0`, RGB and depth sensors 128 by 128, success distance 1.0, task type `AudioNav`, sensor `SPECTROGRAM_SENSOR`, 500 steps. The README example splits are named `train_telephone` and so on. There is no instruction sensor. The README also documents "Training continuous navigation agent" with `CONTINUOUS True` and a `ContinuousSoundSpacesSim` class exists; I did not read the continuous action details beyond the config.
- SAVi / semantic AudioGoal (`verified`): `semantic_audiogoal.yaml`: task `SemanticAudioNav`, sensors include AUDIOGOAL, SPECTROGRAM, CATEGORY, POINTGOAL_WITH_GPS_COMPASS; RGB and depth 128 by 128; success distance 1.0; `DISTANCE_TO: VIEW_POINTS`; sound source directory `data/sounds/semantic_splits`. Episode class fields: `object_category`, `sound_id`, `distractor_sound_id`, `offset`, `duration`. Goal class: `view_points`, "navigable positions around the object with specified proximity of the object surface used for navigation metrics calculation. The object is visible from these positions."
- Start-of-episode visibility: not a stored field in either (keyword search found only the view_points docstrings above and a depth-map projection comment in `nav.py`). It could be computed from start pose, the object's semantic annotation and the scene (`speculative` on ease).
- Data needed: partial binaural RIRs for the SoundSpaces 1.0 grid (867G per the earlier entry), Matterport3D (terms of use); habitat-sim and habitat-lab v0.1.7 for the cached observations (`soundspaces/README.md` step 3).

#### huang2026wnm: WNM-3D, simulator and VLN-CE questions (extends the existing survey entry)
- Link: https://arxiv.org/abs/2608.07267, v3 (29 Oct 2026, per the arXiv HTML footer). Read depth for this file: raw HTML v3 text-searched for Habitat, R2R, Matterport/MP3D/HM3D (zero Matterport hits), VLN-CE; Sec. 4.1, Table 2 and appendix B.2 read in full.
- VLN-CE (`verified`, Sec. 4.1, 4.2.2, appendix B.2): "We evaluate WNM-2D and WNM-3D on GN-Bench Seen/Unseen and the VLN-CE R2R Val-Unseen split"; "VLN-CE uses R2R and RxR training splits with the same curriculum". Appendix B.2: Stage II's oracle "uses Habitat-Sim's navmesh shortest-path query ... The implementation uses habitat_sim.ShortestPath and pathfinder.find_path"; "Episode horizon and other environment-side execution settings follow the official VLN-CE protocol." The word Habitat occurs in the paper only in that passage (two hits). The paper does not name Habitat-Lab or a version. Whether the run used Habitat-Lab is `recalled` from VLN-CE being defined on Habitat (Krantz et al. 2020).
- Action adaptation (`verified`, B.2): WNM predicts eight local motion increments (dx, dy, dyaw) per block; "We accumulate the predicted (Δx, Δy) components ... to form a target XY polyline and track it with a map-free kinematic beam-search controller." Each expansion is zero or more same-direction turns followed by a 0.25 m forward move, turns discretized at 15 degrees on R2R and 30 degrees on RxR, beam width 32, up to four forward moves; the predicted yaw increment is not used in the cost; "Only the first four discrete primitives ... are executed before replanning"; STOP is handled separately from the predicted block's travel length. So the authors already converted continuous blocks to discrete VLN-CE actions, but this converter is described in the paper, not found in a released repository (below).
- Training and evaluation code searched (`verified`, tarballs of the default branches downloaded 2026-11-18 and searched case-insensitively for habitat, vln_ce, vlnce, r2r, rxr, matterport, mp3d, hm3d, soundspaces, vln-ce, excluding images): `TeleHuman/WNM-3D` (master, 146 tree entries): only two false hits (`cv2.COLOR_BGR2RGB`); `TeleHuman/WNM-3D-RL` (main, 252 entries): the same two false hits; `TeleHuman/GN0` (master, 79 entries): hits are GN-Bench documentation and code, see next entry. No R2R, RxR or Matterport3D file, config or script in any of the three. The `gammanav/vln` directory name is "vln" in the path only.
- Released checkpoints (`verified`): model card (raw `README.md` of https://huggingface.co/TeleEmbodied/WNM-3D and `docs/MODEL_CARD.md` in the repository): three checkpoints (stage 1, 2, 3); "The released policy was developed and evaluated with the InteriorGS scenes and GN-Matrix task annotations used by GN-Bench. Performance may not transfer to new simulators, sensors, scene distributions, languages, or physical robots without additional validation." The stage-3 release's `experiment_cfg/conf.yaml` lists `train_dataset` with `interiorgs: ./data/interiorgs_lerobot_seen` only. `experiment_cfg/metadata.json` gives raw video resolution 480 by 360 at about 3.33 frames per second and the `nav_delta` action statistics; the config sets `target_video_height: 160`, `target_video_width: 320`, `num_frames: 33`. The R2R-trained variant is not among the released checkpoints (the model card lists only the three GN-Bench stages).
- Inference server (`verified`, `scripts/inference/wnm_3d_server.py`, docstring and code): takes msgpack-numpy requests over WebSocket. Input: image(s) as "HWC uint8 single frame, or THWC uint8 frames" (keys include `video.rgb`, `image`, `rgb`, `observation/rgb`); optional `state.nav_pose` / `state` as [x, y, yaw] (defaults to zeros); optional language under `annotation.language.language_instruction`, `prompt`, `instruction`, `language` or `text` (defaults to an empty string); optional `session_id` (a change resets the frame buffer) and `reset`. The server keeps its own rolling RGB buffer, samples history to the model's 33 frames ("uniform" by default across the whole history; "recent" and "mixed" also exist), and raises an error if a frame is not at the checkpoint's raw resolution unless started with `--resize-input-to-checkpoint-resolution`. Output: `action.nav_delta` of shape (8, 3) physical [dx, dy, dyaw] (the first action block of the receding-horizon plan), plus a scaled copy, `action_type: "nav_delta"` and a unit field. It takes no audio and has no audio path, no map, no depth.

#### li2026gn0: GN0 / GN-Bench client and GN0-VLN-CE (same authors; read for the interface)
- Links: https://github.com/TeleHuman/GN0 (master, Apache-2.0, 79 entries), https://github.com/TeleHuman/GN0-VLN-CE (main, Apache-2.0, 86 entries). The GN0 README cites arXiv 2606.03682 (not opened in this session).
- Read depth: both READMEs, GN0 `configs/gn_bench/interiorgs/test_unseen.yaml`, `gn0/benchmark/config.py`, header of `scripts/evaluation/eval_remote.sh`, GN0-VLN-CE `docs/INSTALLATION.md`, headers of `agents/mpc.py`, `agents/habitat_adapter.py`, `env/habitat_env.py`, and a keyword search (wnm, websocket, nav_delta, remote) over the GN0-VLN-CE tarball.
- GN-Bench is Habitat-derived (`verified`): GN0 README acknowledgements: "GN-Bench-Tools is adapted from Habitat-Lab and extended for 3D Gaussian Splatting-based navigation"; the repository includes `LICENSES/HABITAT_LAB_MIT_LICENSE`; `GN-Bench-Tools` is a git submodule (not part of the tree I read). The config uses a Habitat-v0.1-style yacs layout (`TASK_CONFIG`, `TYPE: VLN-v0`, `POSSIBLE_ACTIONS: [STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]`, `FORWARD_STEP_SIZE: 0.25`, `TURN_ANGLE: 15`, RGB 480 by 360 with hfov 70, success distance 3.0, 500 steps). So the interface WNM-3D already uses is a VLN-CE-like Habitat interface backed by a Gaussian-splat renderer.
- Remote client protocol (`verified`, GN0 README "Remote Client"): the client sends `{"endpoint": "infer", "images": ..., "instruction": ..., "session_id": ...}` and accepts either `{"action_type": "discrete", "actions": [1, 1, 2, 3]}` (VLN-CE codes 0 stop, 1 forward, 2 left, 3 right) or `{"action_type": "nav_delta", "actions": [[dx, dy, dyaw], ...]}`, with translation in meters and yaw in radians. The client itself lives in GN0 and is wired to GN-Bench (`gn0/agents/remote.py`, 52,148 bytes); I did not read it in full.
- GN0-VLN-CE (`verified`): "provides standard Habitat VLN-CE evaluation and closed-loop DAgger collection for GN-BAE"; Matterport3D scenes plus R2R VLN-CE v1.3 annotations; installs habitat-sim and habitat-lab at v0.1.7 in a Python 3.9 environment. The keyword search found no websocket client, no nav_delta handling and no mention of WNM in this repository; `agents/mpc.py` is a "receding-horizon beam-search MPC to produce discrete actions" from a polyline, used by its DAgger agent. Whether this is the same controller as the WNM-3D paper's is not stated; I did not compare them.
- Not found: any repository from these authors that connects WNM-3D to Habitat.

#### chen2025astranav: AstraNav-World, Habitat evaluation and training code
- Link: https://arxiv.org/abs/2512.21714 (paper not re-read in this session; the paper-level facts come from the existing survey entry, `recalled`). Repository https://github.com/amap-cvlab/AstraNav-World (main, Apache-2.0, pushed_at 2026-09-22, 6,786 tree entries).
- Read depth: README (raw), file tree, `infer_r2r_rxr/eval_r2r.sh`, directory listings of `infer_ovon`, `infer_r2r_rxr`, `world/wan`, `train_code`; ModelScope page attempted.
- Habitat side (`verified`): README "Install habitat-sim and habitat-lab for inference": R2R and RxR use habitat-sim and habitat-lab `v0.1.7`; OVON uses habitat-sim `v0.2.3` and a habitat-lab fork (`chongchong2025/habitat-lab`, tag `v0.2.3_waypoint`). Released inference code: `infer_r2r_rxr/` (`run_infer.py`, `agent/waypoint_agent.py`, `eval_r2r.sh`, `eval_rxr.sh`, a vendored `VLN_CE` package) and `infer_ovon/` (`run_infer_ovon.py`, `agent/waypoint_agent_ovon.py`, a vendored `ovon` package). The agent runs the policy in-process (the script loads `--model-path`); I saw no WebSocket server in the tree.
- Training code (`verified` from the tree): `train_code/` appears to be a vendored ms-swift trainer plus `train_code/data/waypoint_train_demo.json` (a demo of waypoint data); `world/wan/` has LoRA, action-former, future-frame and pipeline code. I found no training script for the Wan video generator, and no weight files in the tree. README: "We release the inference checkpoints of action former" (2026-01-21), with a ModelScope link (https://modelscope.ai/models/chongchongjj/AstraNav_World_ActionFormer); the page is rendered by script, so I could not read its file list. Future-frame prediction needs a separate Wan2.2-TI2V-5B-Diffusers download. Whether the released checkpoint contains the video LoRA: not found.
- Audio: none (earlier survey, zero hits).

#### xue2026uniwam: UniWAM, the closest complete Wan 2.2 + Habitat + WebSocket stack found
- Link: https://arxiv.org/abs/2609.39388 (the repository README's paper badge; paper not re-read this session, paper-level facts from the existing survey entry, `recalled`). Repository https://github.com/Fysics-AI/UniWAM (main, Apache-2.0 for original code, pushed_at 2026-10-31, 501 tree entries); Hugging Face model https://huggingface.co/Fysics-AI/UniWAM, datasets `Fysics-AI/MAP-Bench` and `Fysics-AI/MAP-Data` (API: all three `gated: "auto"`, so a Hugging Face access agreement is required).
- Read depth: README, `docs/EVALUATION.md`, `docs/SETUP.md`, `POLICY_INTERFACE.md`, `DATA_INTERFACE.md`, `training/NAVIGATION.md`, `training/README.md`, `OPEN_SOURCE_CHECKLIST.md`, Hugging Face API file lists.
- What exists (`verified`): a navigation-only preset ("UniWAM-Base MAP-Bench training"), "built on FastWAM", initialized from Wan2.2-TI2V-5B, one forward RGB camera, 9-dimensional outputs (base waypoints, target-object boxes, image-plane MAP coordinates), training script `scripts/train.py` with Hydra configs; the data package carries RGB video at 15 Hz and instructions; closed-loop evaluation in Habitat on HM3D v0.2 scenes ("Habitat evaluation runs in a separate environment and connects to the policy through WebSocket"; `conda install habitat-sim=0.3.3`), with a documented WebSocket policy interface: request `{image: uint8 H x W x 3, task: str, navigation_state: float32[2], diffusion_seed: int}`, response `relative_se2_actions: float32[32, 3]` ([x_body_m, y_body_m, heading_rad]). Also released: model checkpoints (Base and Scaled), and raw rollout traces for five methods.
- What it is not: the navigation task is instruction-conditioned navigation to a "Manipulation Anchor Pose" near a target object, with one forward RGB view; no audio anywhere in the files read; it uses habitat-sim 0.3.3, not SoundSpaces 2.0's audio branch.
- Training scale stated: "The report's MAP-Bench runs use 16 GPUs" (`training/NAVIGATION.md`).

#### Other Wan 2.2 (or close) navigation models, from the existing survey file (not reopened beyond the ones above)
- `recalled`, from `2026-11-16-navigation-world-models.md`: NavGen (Wan2.2 5B, drone, no code found), EgoWM (Cosmos and Wan2.1 variants have no released scripts), UniNav (Wan2.1-1.3B, code "will be released", none found), ImagineUAV, DreamToNav, NavDreamer (no code or no training), ABot-M0.5 (Wan2.2, mobile manipulation, evaluated on RoboCasa; I opened its repository tree today: `evaluation/libero`, `evaluation/robocasa`, a `wam/server.py` and a `wam_client/websocket_client_policy.py`, no Habitat or navigation benchmark in the tree). None of these is evaluated against a Habitat audio simulator.

### What exists versus what would have to be built (WNM-3D on audio-visual navigation)
Exists (`verified` unless marked):
- WNM-3D server with a simulator-agnostic input (RGB frames, optional language, optional pose) and output (nav_delta blocks of 8 increments).
- WNM-3D Stage I and II training code and Stage III repository; the checkpoints (GN-Bench only; non-commercial licence caveat as in the survey).
- GN-Bench's remote client protocol, which already accepts both discrete VLN-CE actions and nav_delta.
- A Habitat VLN-CE harness from the same authors (GN0-VLN-CE, Habitat v0.1.7) and a beam-search polyline-to-discrete-action planner (`mpc.py`) in it.
- MINav's evaluation environment code (Habitat v0.2.3 + audio + SoundSpaces 2.0, binaural microphones, HM3D) and its episode-generation scripts (completeness unconfirmed).
- SoundSpaces AudioGoal and SAVi episode datasets and configs (grid simulator, 128 by 128, discrete actions; continuous mode documented).
- UniWAM's Habitat + HM3D + WebSocket harness and Wan2.2 navigation training code (no audio).
Would have to be built:
- A Habitat-side client that sends RGB frames and receives nav_delta (the WebSocket protocol makes this a client, not a model change).
- A nav_delta to discrete-action converter for whichever benchmark is chosen (the paper's appendix B.2 describes one; no released WNM-3D repository contains it).
- An instruction for audio tasks (AudioGoal has none; WNM-3D defaults to an empty string but was trained with language; whether it behaves with a fixed template or empty text is untested).
- An audio input path in the model (WNM-3D has none; this is the research contribution, not a simulator issue).
- Training data in WNM-3D's LeRobot format with continuous nav_delta labels, rendered in Habitat with SoundSpaces 2.0 (frame rate, resolution and action step differ from the checkpoint's 480 by 360, about 3.33 frames per second metadata).
- A start-of-episode target-visibility label (compute from start pose and scene; no benchmark stores it).
- A reconciliation of habitat-sim and habitat-lab versions: SoundSpaces 2.0 install uses the `RLRAudioPropagationUpdate` branch and habitat-lab v0.2.2 (earlier survey), MINav's script pins v0.2.3 with `--audio`, GN0-VLN-CE and AstraNav R2R use v0.1.7, UniWAM uses habitat-sim 0.3.3. I did not test whether any two of these coexist.

### Not verified
- Whether MINav's build scripts run end to end (the two numbered stubs are 99 bytes; others look substantial by size). Not run.
- Whether any released checkpoint of AstraNav-World includes the video generator (ModelScope page unreadable).
- Whether GN0's remote client could be pointed at a non-GN-Bench simulator without edits (`gn0/agents/remote.py` not read in full, `speculative` that it cannot).
- GN0 arXiv paper 2606.03682, UniWAM paper and AstraNav-World paper were not opened in this session.
- Whether SoundSpaces' continuous AudioGoal mode records episode start visibility: only keyword search was done.

### Searches run
- `gh api repos/<repo>` and `.../git/trees/<branch>?recursive=1` for TeleHuman/WNM-3D (master, 146 entries), TeleHuman/WNM-3D-RL (main, 252), TeleHuman/GN0 (master, 79), TeleHuman/GN0-VLN-CE (main, 86), felixfjg/NaVLA (main, 513), facebookresearch/sound-spaces (main, 290), amap-cvlab/AstraNav-World (main, 6786), amap-cvlab/ABot-Manipulation (111), Fysics-AI/UniWAM (501). Tree path filters: habitat, vln, r2r, rxr, matterport, mp3d, hm3d, sound, audio, ovon, eval, infer, train.
- GitHub code search (`search/code`, repo-restricted) for habitat, vln_ce, R2R, matterport on WNM-3D, GN0 and WNM-3D-RL: returned empty for habitat even where files mention Habitat (GN0), then rate-limited; so it was not relied on. Replaced by repository tarballs (`gh api repos/<o>/<r>/tarball/<branch>`) extracted in the scratchpad and `rg -i` for: `habitat|vln_ce|vlnce|r2r|rxr|matterport|mp3d|hm3d|soundspaces|vln-ce` (WNM-3D, GN0, WNM-3D-RL, GN0-VLN-CE) and `wnm|websocket|nav_delta|remote` (GN0-VLN-CE); `visib|line.of.sight|is_in_view|can_see|out of sight|occlu` (sound-spaces); `visib|line.of.sight|is_visible|in_view|raycast` (NaVLA).
- NaVLA2 PDF `pdftotext` searches: visible, line of sight, out of sight, occlu (zero hits); SoundSpaces; resolution, history frames, video frames.
- WNM-3D arXiv HTML v3 text searches: habitat; R2R; Matterport|MP3D|HM3D; VLN-CE.
- Hugging Face API: `/api/models?search=NaVLA`, `/api/models?search=MINav`, `/api/datasets?search=MINav`, `/api/datasets?search=NaVLA`; `/api/models/TeleEmbodied/WNM-3D` and its raw README and `experiment_cfg/{metadata.json,conf.yaml}`; `/api/models/Fysics-AI/UniWAM`, `/api/datasets/Fysics-AI/MAP-Bench`, `/api/datasets/Fysics-AI/MAP-Data`.
- WebSearch: "NaVLA2 Vision-Language-Audio-Action Model Multimodal Instruction Navigation MINav arXiv" (found the AAAI page and the repository; no arXiv listing); "MINav dataset release huggingface NaVLA felixfjg checkpoints" (found nothing newer than the repository README; a result was an unrelated MiniVLA-Nav dataset).
- WebFetch-equivalent `curl` of the AAAI page https://ojs.aaai.org/index.php/AAAI/article/view/38886 (title and DOI), ModelScope model page (script-rendered, unreadable), ModelScope file-list API (error "record not found").
- arXiv API for 2609.39388: rate limited (not retried), so the UniWAM title and version were not rechecked this session.

## Part 3: second, adversarial search on questions 1 and 2 (2026-11-19)

Requested by the planning lead on 2026-11-19 at the user's ask ("are you sure nobody has done research on this? ... can u double check nobody has done this also"). Run by a worker that did not work on Parts 1 and 2, told to try to break the two "not found" verdicts with the angles the first pass could not cover (arXiv API directly, forward citations, OpenReview, other wordings). Note: the SAVi arXiv id given in the lead's brief to that worker was wrong; the worker used SAVi's real id, 2012.11583.

Direction: hear-lab. Written by a research worker for the literature lead. Read-only on the repository. Task: try to break the two "not found" verdicts in `hear-lab/literature/survey/2026-11-18-novelty-and-simulator-gap.md` (PR #68, Part 1), assuming the first pass missed something. Labels: `verified` = read this session in the named source (section or table named); `recalled` = taken from the PR #68 or 2026-11-16 survey files without reopening; `speculative` = this file's own reading. Numbers are copied from the sources; none were reproduced. Downloads are in the session scratchpad (`downloads/`), untrusted, never executed.

Access limits that shape this file (stated up front, not smoothed):
- OpenReview's search API (`api2.openreview.net/notes/search`) answered with titles and abstracts this time, but every PDF and every `notes?id=` call returned HTTP 403 "ChallengeRequiredError" (a browser challenge). So the ICLR 2027 and workshop submissions below are judged from their OpenReview abstracts only.
- Semantic Scholar answered for most seeds after waits; the main SoundSpaces (1912.11474) record could not be resolved (the arXiv-id record has only 29 citers, and the title search was rate limited after six retries). HEAR (2603.16086) is still not in Semantic Scholar ("Paper with id arXiv:2603.16086 not found").
- The task brief's "SAVi (2105.09378 check)": that arXiv id is an MRI paper ("Robust partial Fourier reconstruction for diffusion-weighted imaging ...", Semantic Scholar record). SAVi is arXiv 2012.11583, which was used instead.

### Verdicts

**Q1 (spatial versus one-channel versus no audio inside a navigation world model or world action model, split by target visibility at the start): still not found, as stated. Not broken.** After about 50 arXiv API queries, forward citations of seven seed papers (about 450 citing records), 22 OpenReview searches (including ICLR 2027 submissions) and GitHub repository search, I found no paper or repository that gives a navigation world model (large or small) spatial audio, one-channel audio and no audio as matched conditions, and none that splits results by whether the target is visible at the start. Three things come closer than what PR #68 listed, and they should be added to the "nearest work" list. (1) A public, unreviewed GitHub repository, `adityakaduskar/acoustic-jepa` (created 2026-11-16), trains a small action-conditioned world model on simulated binaural echoes and uses it for planning (model predictive control) in navigation-like tasks. Its "binaural bearing versus range only" cue ablation is run on a model-free policy, not on the world model, and the agent hears its own echoes, not a sounding target, and has no vision. (2) BCNav (arXiv 2609.37084 v1) reports sound-direction accuracy separately for acoustic line of sight and non-line of sight (about occlusion of the sound path, not whether the target is visible to the camera), with a depth-plus-bearing policy, no world model. (3) HearVGo (ICLR 2027 submission, OpenReview abstract only) swaps the left and right audio channels as a spatial check on 14 omni models predicting walking waypoints, which is neither a world model nor a mono condition. None of these breaks Q1.

**Q2 (does an audio encoder that ranks better on a spatial-audio benchmark also give better navigation or acting success): the claim itself is still not found, but the "partly done at the edges" list is longer than PR #68 says.** No paper ranks several audio encoders on a spatial-audio benchmark and then on an acting task to see whether the rankings agree. New edge cases, from closest to farthest: (1) the same `acoustic-jepa` repository reports that, for binaural-echo representations, the ordering on static geometry probes (raw features and a random encoder beat the learned predictive encoder) reverses when the representation is used inside a world model for control (the random-encoder world model "FAILS, doing worse than random actions"). That is the Q2 logic (upstream probe rank against acting outcome), but it is unreviewed, toy-scale (box rooms, echolocation, no vision), compares only learned versus random encoders in control, and uses geometry probes, not a published spatial-audio benchmark. (2) BCNav measures how navigation success falls as injected bearing error grows and states "any DOA method with MAE below 15° enables near-perfect navigation", that is, a mapping from an upstream spatial-audio accuracy number to navigation success; but it tests one estimator (SRP-PHAT) and simulated noise, not a ranking of encoders. (3) ADAPT (arXiv 2610.15956 v1) switches a spatial-audio adapter on and off in an audio-plus-depth navigation agent (success goes up with it) and reports direction-of-arrival error and navigation success under the same noise conditions, but with one encoder, not a ranking. (4) HearVGo evaluates 14 omni models (different audio front ends) on one spatial-audio-guided acting benchmark, with no upstream spatial-audio benchmark comparison stated in the abstract. So Q2 is "partly, at the edges; the precise test is not found". PR #68's statement "I found no work that ranks audio encoders by an acting task and relates that ranking to an upstream spatial-audio benchmark" survives, with the caveat that an unreviewed repository has done the probe-versus-control comparison in a toy echolocation setting.

### Hits that come closest

#### acoustic-jepa (GitHub repository, not a paper; unreviewed, tier 3 to 4)
- Source: https://github.com/adityakaduskar/acoustic-jepa, default branch `master`, created 2026-11-16T02:15:56Z, last commit f9c1267 (2026-11-16), 0 stars. Read through `gh api`: `README.md`, `RESULTS.md` (section "Control test (latent-MPC navigation) — the dissociation" and "Robustness"), `docs/EchoRL_RESULTS.md` (sections 3.6, 4, 5, keyword passes). Author line in the README: "Aditya Kaduskar, Vanderelst Lab, University of Cincinnati". No paper found for it.
- What it did (`verified`, README "Main results"): simulated rooms rendered with Steam Audio; "An observation is a 256-bin matched-filter range profile plus a 2 x 64 x 48 binaural cochleagram." Two studies: Acoustic-JEPA (self-supervised world models on binaural echoes, geometry probes, then latent model predictive control) and EchoRL (PPO, DQN and model-based controllers that "navigate from binaural echoes alone").
- The Q2-like result (`verified`, RESULTS.md "Control test"): "This flips the probe result. The learned world model navigates at 87% of the analytic oracle, while the random-encoder world model — the encoder that *tied or beat* the learned one on every geometry probe — FAILS, doing worse than random actions". Five-seed rerun table: learned MPC "+0.798 ± 0.051" m, random-encoder MPC "−0.054 ± 0.138" m (nearest-wall-distance reduction). README summary: "Static decodability and dynamical predictability are different properties."
- The spatial-cue ablation (`verified`, EchoRL_RESULTS.md section 4 cue ablation table, fair evaluation, 3 seeds): doorway-finding success "cochleagram-only (bearing) 0.68 ± 0.03", "full 0.66 ± 0.03", "range-only (no bearing) 0.48 ± 0.05". This is a model-free PPO policy (section 3.2 and 4), not the world model. Section 5 compares the learned-world-model MPC ("0.46 ± 0.05") with PPO ("0.68") on the doorway task; I found no bearing ablation run on the world-model controller (keyword pass for MPC/WM near bearing/coch/range-only: no hits).
- Did NOT do (`verified` by the files read): no vision, no sounding target (the agent emits and hears its own echoes), no one-channel (mono) condition as such (the "range only" stream is a matched-filter range profile, "HRTF-free (no direction)"), no published spatial-audio benchmark, no target-visibility split, no large pretrained video model.
- Breaks Q1: no (closest small-world-model-plus-binaural case, but the spatial ablation is on the policy, not the world model). Breaks Q2: partly (same logic, toy setting, unreviewed, learned vs random encoder only).

#### BCNav: Bearing-Conditioned Depth Policies for Sound Source Navigation (arXiv 2609.37084 v1; also an OpenReview public-article record)
- Read: arXiv HTML (v1, per the arXiv API listing), sections IV-B to IV-E and Table I, II.
- What it did (`verified`): "a classical DOA estimator that converts microphone array signals into a scalar bearing, and a learned policy that maps depth images and the bearing to continuous velocity commands" (Sec. III). IV-B: with injected bearing noise, "SR reaches 96.5% for σ ≤ 15°" and "drops to ∼75% beyond 30°"; "any DOA method with MAE below 15° enables near-perfect navigation." Table I (SRP-PHAT, 8-channel array, SoundSpaces 2.0 renders): LOS MAE 11.4°, NLOS 37.3°, overall 21.7°. IV-D: closed loop with SRP-PHAT "achieves 71.1% SR on HM3D validation scenes".
- Did NOT do: compare several DOA estimators or audio encoders by navigation success; any world model; mono versus array; a split by whether the target is visible to the camera (the LOS/NLOS split is acoustic line of sight, and it is on DOA error, not on success).
- Breaks Q1: no. Breaks Q2: partly (it maps an upstream spatial-audio error to navigation success, but with one estimator and synthetic noise).

#### ADAPT: Audio-Depth Action Planning (arXiv 2610.15956 v1)
- Already listed in the 2026-11-16 prior check as "a policy, no world model" (`recalled`); reopened here for Q2.
- Read: arXiv HTML, Sec. 4.1, Tables 1 to 5, keyword passes.
- What it did (`verified`): the agent observes "a four-channel first-order ambisonics (FOA) observation and an egocentric depth map"; Sec. 4.1: "The audio pathways of existing LMMs are typically trained for mono audio and not optimized to recover directional information from FOA signals ... We therefore augment the audio encoder with an SAA from SPUR". Table 3(a) (SFT-only, LN setting): without SAA and without action history SR 52.07, with SAA only 53.27, with both 62.93; action history only 58.73. The rows are (SAA, action history) = (no, no) 52.07, (no, yes) 58.73, (yes, no) 53.27, (yes, yes) 62.93, reading the two check columns in header order "SAA Action Hist." (`verified` values; the column-to-component mapping follows the header order). Table 4 gives DOA error under conditions; Table 5 gives navigation success under fixed SNR. Zero-shot baselines Qwen 2.5-Omni and Qwen 3-Omni receive "synchronized FOA channels through separate mono-audio inputs" (Sec. 5 text).
- Did NOT do: rank several audio encoders on a spatial benchmark and on navigation; a mono-versus-FOA condition for its own trained agent; a world model.
- Breaks Q1: no. Breaks Q2: partly, at the edge (adapter on/off, one encoder).

#### HearVGo: Benchmarking Sound-Conditioned Trajectory Adaptation for Urban Walking (ICLR 2027 submission, OpenReview forum 0VS8HvxJyN)
- Read: OpenReview abstract via the search API only (PDF blocked by a 403 challenge); anonymous repository README (https://anonymous.4open.science/r/HearVGo, file README.md, fetched 2026-11-19).
- What it did (`verified`, abstract): CARLA with "1,288 environmental sound clips rendered as spatial audio", "536 trajectory instances", "We evaluate 14 omni models under a common zero-shot protocol and complement the main comparison with controlled audio and vision ablations. These results show model-dependent gains from audio and few reversals of predicted avoidance direction when the left and right audio channels are exchanged." README: "Every model predicts ten future two-dimensional waypoints".
- Did NOT do (as far as the abstract shows): a world model; a one-channel condition (the abstract names a left/right swap); an upstream spatial-audio benchmark ranking. Open-loop waypoint prediction, not closed-loop navigation (`verified` from README wording; whether there is any closed-loop part is not known).
- Breaks Q1: no. Breaks Q2: partly at most (14 audio-capable models compared by an acting-style task; no upstream ranking stated). Worth opening once the PDF is reachable.

#### Cosmos 3: Omnimodal World Models for Physical AI (arXiv 2606.02800 v4)
- The prior check read the NVIDIA PDF; here the arXiv HTML v4 was keyword-checked.
- `verified`: abstract, "omnimodal world models designed to jointly process and generate language, image, video, audio, and action sequences". Appendix E.3 "Audio Data in Pre-Training" compares video-only and joint video-audio continued pre-training on video-generation scores only (Table 30). E.4 is about action modes on PushT, with no audio.
- Did NOT do (in the sections read): use audio as an input to any robot policy or world-action evaluation; any spatial audio.
- Breaks Q1/Q2: no. It is the clearest example of a big pretrained world model that already carries an audio stream, which matters for the "audio in a large pretrained video model" framing.

#### Other papers opened this session, none breaking Q1 or Q2 (`verified` by keyword passes for mono/monaural/single-channel, visible/occluded/line of sight, world model)
- MuSTMM 2610.21429 v1: binaural RIRs, mobile manipulation, no mono or visibility split.
- Two-stage dynamic AVN 2610.14077 v1, CTAN 2609.17420 v1, token fusion 2609.17421 v1, Hybrid Mamba 2607.13110 v1, BDA-ATP 2604.05007 v1, RAGF 2604.02391 v1, SAVN-CE 2603.19660 v1, CRFN 2601.08868 v1, IRCAM 2509.25652 v1, DMTF 2509.21377 v1, multi-agent AVN 2509.22698 v1, CRONA 2605.06595: audio-visual navigation policies; zero hits for a mono condition; occlusion appears only as motivation or one qualitative scene.
- Sim2Real AVN 2405.02821 v2: "Since the simulator employs a mono receiver, the two-channel audio data we gather is transformed into mono format" (a data choice, not a comparison).
- Towards Generalisable Audio Representations for AVN 2206.00393 v1: regularises the binaural audio encoder (AFSO) and reports SPL gains; one encoder family, no upstream benchmark ranking.
- Self-supervised spatial audio representations 2206.00970 v1: compares "Ambisonics, mono, and stereo" input formats, evaluated on audio-visual spatial alignment and recognition tasks, not acting.
- Evidence of Absence 2608.14952 v1: microphone-array cue to sustain a "structured world-state" when vision is occluded (driving hazard warning, no learned world model of frames, no mono comparison).
- Dreaming the Sound of Contact 2609.19137 v2: Seedance 2.0 generates video plus audio for manipulation; audio loudness shapes force; no spatial audio, not navigation (already named in the prior check, `recalled`).
- Does Hearing Help Seeing 2512.02457 v2, MaineCoon 2606.17800 v1, Beyond Multimodal Alignment 2608.19492 v2, The Sound of Simulation 2507.02864 v2, Binaural sounding-instance AVS 2611.06074 v1: no acting comparison of spatial versus mono audio.
- OpenReview abstracts only (PDF blocked): D2AVN, TraSNav, DANCE/DyGMap (ICLR 2027 submissions, audio-visual navigation methods), ISAudio (ICLR 2027, action-conditioned binaural audio prediction "for world models", evaluated on audio quality, no agent success in the abstract), Acoustic World Models for Active Scene Reconstruction (ICLR 2027; an acoustic world model from room impulse responses used to choose sensing poses for mapping; no vision-plus-audio and no mono comparison in the abstract), Hear-to-Act (RSS 2026 workshop; binaural microphones on a humanoid and an arm, a "sound-conditioned controller"; no mono comparison in the abstract), SoundAct (ICRA 2026 workshop; stereo audio for "beyond-sight" manipulation; no mono comparison in the abstract), SARL / "Probing Spatial Structure in Pretrained Audio Representations" (already in PR #68, `recalled`), "Spectro-Temporal Interference Confounds Phase Encoding in Spatial Audio Foundation Models" (nine frozen audio models, a psychoacoustic benchmark, no acting), MM-World (video world model for egocentric navigation and manipulation, no audio in the abstract). Hear-to-Act and SoundAct could hide a mono ablation in the full text; I could not open them.
- GitHub: `huzjie/echoworld` ("Audio-visual world model inference & real-time interaction platform", description only) and `Luizerko/AudioWorldSim` (the dataset repo behind 2608.21075, already known) were seen in repository search; not opened beyond descriptions.

### Forward-citation screen (Semantic Scholar)
Citing records were screened by title and abstract (where Semantic Scholar had one) for mono/monaural/single-channel, world model, video prediction or generation, encoder or representation, benchmark, together with navigation/robot/policy/embodied/manipulation/agent; 107 of 448 unique records passed that filter and were read by title and abstract; the ones above were opened. Notable records not in PR #68: "Decoupling Embodied Active Audio-Visual Localization Via Offline Reinforcement Learning" (ACM MM 2026, DOI 10.1145/3840475.3841436, abstract only: viewpoint-control policy, no mono or world model in the abstract), MESH (2611.16787, a spatial-hearing benchmark across receivers, no acting), "Multimodal Video Generation Models with Audio: Present and Future" (survey, DOI 10.1145/3840475.3841440, abstract only). A Semantic Scholar record titled "Audio-Visual World Models: Grounding Multisensory Imagination for Embodied Agents" (no ids) and a DBLP/OpenReview record "Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound" carry the AVWM abstract wording; I read them as earlier titles of arXiv 2512.00883 (`speculative`).

### Search log
All on 2026-11-19. Counts are arXiv totalResults (all returned up to 100 were screened by title, and by abstract keyword filter for audio terms plus acting terms), Semantic Scholar citing records, or OpenReview notes returned (search source=forum, limit 100; titles screened). "FAILED" = rate limited or no feed after retries.

arXiv API (`export.arxiv.org/api/query?search_query=...&sortBy=submittedDate&sortOrder=descending&max_results=100`, all categories, all dates; 2025-06 onward and older screened) and OpenReview:
- arXiv API | `abs:binaural AND abs:navigation` | 2026-11-19 | 15
- arXiv API | `abs:binaural AND abs:navigation` | 2026-11-19 | 15
- arXiv API | `abs:binaural AND (abs:monaural OR abs:mono OR abs:"single-channel" OR abs:"single channel")` | 2026-11-19 | 65
- arXiv API | `(abs:monaural OR abs:mono) AND (abs:navigation OR abs:robot OR abs:embodied OR abs:policy)` | 2026-11-19 | 62
- arXiv API | `abs:"audio-visual navigation"` | 2026-11-19 | 33
- arXiv API | `abs:"audio-goal" OR abs:audiogoal OR abs:"audio goal navigation"` | 2026-11-19 | 4
- arXiv API | `abs:soundspaces` | 2026-11-19 | 21
- arXiv API | `abs:"world model" AND (abs:audio OR abs:sound OR abs:acoustic OR abs:auditory OR abs:binaural OR abs:ambisonic)` | 2026-11-19 | 77
- arXiv API | `abs:"video prediction" AND (abs:audio OR abs:sound OR abs:binaural)` | 2026-11-19 | 8
- arXiv API | `abs:"video generation" AND abs:audio AND (abs:navigation OR abs:robot OR abs:embodied OR abs:agent)` | 2026-11-19 | 37
- arXiv API | `abs:"audio-conditioned" AND (abs:video OR abs:"world model")` | 2026-11-19 | 49
- arXiv API | `abs:"sound-conditioned" OR abs:"audio-driven world"` | 2026-11-19 | 23
- arXiv API | `abs:ambisonic AND (abs:robot OR abs:navigation OR abs:embodied OR abs:agent OR abs:policy)` | 2026-11-19 | 8
- arXiv API | `abs:"spatial audio" AND (abs:encoder OR abs:representation) AND (abs:robot OR abs:navigation OR abs:embodied OR abs:downstream)` | 2026-11-19 | 21
- arXiv API | `abs:"audio encoder" AND (abs:robot OR abs:navigation OR abs:manipulation OR abs:embodied OR abs:policy)` | 2026-11-19 | 20
- arXiv API | `abs:"audio representation" AND (abs:robot OR abs:navigation OR abs:manipulation OR abs:embodied OR abs:policy OR abs:reinforcement)` | 2026-11-19 | 24
- arXiv API | `abs:"audio encoders" AND (abs:downstream OR abs:benchmark) AND (abs:agent OR abs:robot OR abs:embodied)` | 2026-11-19 | 2
- arXiv API | `abs:"audio" AND abs:"vision-language-action"` | 2026-11-19 | 17
- arXiv API | `abs:"audio" AND abs:"world action model"` | 2026-11-19 | 2
- arXiv API | `abs:"auditory" AND (abs:navigation OR abs:embodied) AND abs:2026` | 2026-11-19 | 0
- arXiv API | `(abs:"sound source" OR abs:"sounding object") AND (abs:"out of view" OR abs:occluded OR abs:invisible OR abs:"out-of-view" OR abs:"not visible" OR abs:"off-screen")` | 2026-11-19 | 33
- arXiv API | `abs:"spatial audio" AND abs:"large language model"` | 2026-11-19 | 23
- arXiv API | `abs:"spatial audio" AND (abs:benchmark) AND (abs:encoder OR abs:representation)` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"interaural" AND (abs:robot OR abs:navigation OR abs:agent)` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"acoustic" AND abs:"world model"` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"audio-visual" AND (abs:"embodied agent" OR abs:"embodied ai") AND abs:2026` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"audio-visual" AND abs:"video world model"` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"sound" AND abs:"video world model"` | 2026-11-19 | FAILED (rate limited / no feed)
- arXiv API | `abs:"omni" AND abs:"world model" AND abs:audio` | 2026-11-19 | 5
- arXiv API | `abs:"audio-visual" AND abs:"vision-language navigation"` | 2026-11-19 | 1
- arXiv API | `abs:"audio" AND abs:"embodied navigation" AND (abs:ablation OR abs:modality)` | 2026-11-19 | 9
- arXiv API | `abs:"spatial audio" AND (abs:benchmark) AND (abs:encoder OR abs:representation)` | 2026-11-19 | 25
- arXiv API | `abs:"interaural" AND (abs:robot OR abs:navigation OR abs:agent)` | 2026-11-19 | 7
- arXiv API | `abs:"audio-visual" AND abs:"video world model"` | 2026-11-19 | 0
- arXiv API | `abs:"sound" AND abs:"video world model"` | 2026-11-19 | 0
- arXiv API | `abs:"audio-visual" AND (abs:"embodied agent" OR abs:"embodied ai")` | 2026-11-19 | 15
- arXiv API | `abs:"stereo" AND abs:audio AND abs:navigation` | 2026-11-19 | 5
- arXiv API | `(abs:Dreamer OR abs:"model-based reinforcement") AND (abs:audio OR abs:sound OR abs:auditory)` | 2026-11-19 | 2
- arXiv API | `abs:"pretrained audio" AND (abs:robot OR abs:policy OR abs:navigation OR abs:embodied)` | 2026-11-19 | 6
- arXiv API | `abs:"audio representations" AND (abs:navigation OR abs:embodied OR abs:robot)` | 2026-11-19 | 6
- arXiv API | `abs:"audio" AND abs:"video diffusion" AND (abs:robot OR abs:policy OR abs:navigation)` | 2026-11-19 | 2
- arXiv API | `abs:"omnimodal" AND abs:"world model"` | 2026-11-19 | 5
- arXiv API | `abs:"audio-visual" AND abs:"world model"` | 2026-11-19 | 11
- arXiv API | `abs:"multisensory" AND (abs:"world model" OR abs:"video prediction")` | 2026-11-19 | 6
- arXiv API | `abs:"audio" AND abs:"action-conditioned" AND (abs:video OR abs:"world model")` | 2026-11-19 | 4
- arXiv API | `abs:"sound" AND abs:"navigation" AND abs:"video generation"` | 2026-11-19 | 0
- OpenReview api2 notes/search source=forum | `audio world model` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `binaural navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio-visual navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `spatial audio robot` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio-visual world model` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `spatial audio embodied` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `monaural binaural ablation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio encoder robot policy` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio-conditioned world model` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `acoustic world model` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `sound video world model navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio representation embodied agent` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `spatial audio navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `sound source navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `stereo audio robot manipulation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `binaural robot` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio world action model` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio encoders downstream robot` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `spatial audio representation probing` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio-visual video generation robot` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `audio vision-language-action navigation` | 2026-11-19 | 100 (count 10000)
- OpenReview api2 notes/search source=forum | `hearing world model` | 2026-11-19 | 100 (count 10000)
- arXiv API | `ti:HearVGo` | 2026-11-19 | 0
- arXiv API | `ti:Hear-to-Act` | 2026-11-19 | 1
- arXiv API | `ti:SoundAct` | 2026-11-19 | 0
- arXiv API | `ti:BCNav` | 2026-11-19 | 1
- arXiv API | `ti:D2AVN` | 2026-11-19 | 0
- arXiv API | `ti:"Dynamic Audio-Visual Navigation in Continuous"` | 2026-11-19 | 0
- arXiv API | `ti:TraSNav` | 2026-11-19 | 0
- arXiv API | `ti:ISAudio` | 2026-11-19 | 0
- arXiv API | `ti:"Acoustic World Models for Active"` | 2026-11-19 | 0
- arXiv API | `abs:"world model" AND abs:"spatial audio"` | 2026-11-19 | 3
- arXiv API | `abs:"audio-visual" AND abs:"video generation" AND (abs:robot OR abs:embodied OR abs:navigation)` | 2026-11-19 | 2
- arXiv API | `abs:egocentric AND abs:"spatial audio" AND (abs:action OR abs:navigation OR abs:policy)` | 2026-11-19 | 1
- arXiv API | `abs:"sound" AND abs:"world model" AND abs:navigation` | 2026-11-19 | 3
- arXiv API | `abs:"direction of arrival" AND abs:navigation AND (abs:policy OR abs:reinforcement OR abs:imitation)` | 2026-11-19 | 2
- arXiv API | `abs:"omni" AND abs:navigation AND abs:audio` | 2026-11-19 | 3
- arXiv API | `abs:"audio" AND abs:"navigation world model"` | 2026-11-19 | 0
- arXiv API | `abs:"spatial audio" AND abs:"downstream" AND (abs:localization) AND (abs:encoder OR abs:"self-supervised")` | 2026-11-19 | 3

Semantic Scholar forward citations:
- Semantic Scholar /graph/v1/paper/DOI:10.1609/aaai.v40i22.38886/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 2 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2206.08312/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 175 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2012.11583/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 146 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2105.09378/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 18 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2402.01591/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 72 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2512.00883/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 1 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2406.19464/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 85 citing records
- Semantic Scholar /graph/v1/paper/arXiv:2603.16086/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 0 citing records (Paper with id arXiv:2603.16086 not found)
- Semantic Scholar /graph/v1/paper/arXiv:2512.08405/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 3 citing records
- Semantic Scholar /graph/v1/paper/arXiv:1912.11474/citations (fields title, year, externalIds, abstract) | 2026-11-19 | 29 citing records
- (arXiv:2105.09378 is an unrelated MRI paper; its 18 citers were not screened.)
- Semantic Scholar title search for the main SoundSpaces record: rate limited after six retries (not screened beyond the 29 citers of the arXiv-id record).
- Semantic Scholar title search for HEAR: returned unrelated papers; HEAR has no record.

OpenReview PDFs and note pages (`openreview.net/pdf?id=`, `api2.openreview.net/pdf?id=`, `api2.openreview.net/notes?id=`) for HearVGo, Hear-to-Act, SoundAct, BCNav, D2AVN, DANCE, TraSNav, ISAudio, Acoustic World Models, MM-World, FAVEN: all HTTP 403 "ChallengeRequiredError".

arXiv title lookups (`ti:`) to find arXiv versions of those OpenReview papers: HearVGo, Hear-to-Act, SoundAct, D2AVN, "Dynamic Audio-Visual Navigation in Continuous", TraSNav, ISAudio, "Acoustic World Models for Active": none on arXiv; BCNav found (2609.37084).

GitHub (`gh api search/repositories`): "audio world model navigation" (3 results), "binaural world model" (5), "spatial audio navigation policy" (0). Repository read: adityakaduskar/acoustic-jepa (README, RESULTS.md, docs/EchoRL_RESULTS.md, tree, last five commits). Anonymous repository for HearVGo: README only.

Papers opened (arXiv HTML unless noted): 2610.21429, 2610.15956, 2610.14077, 2609.17420, 2609.17421, 2607.13110, 2604.05007, 2604.02391, 2603.19660, 2601.08868, 2509.25652, 2509.21377, 2509.22698, 2608.14952, 2606.02800, 2606.17800, 2609.19137, 2608.19492, 2507.02864, 2206.00393, 2206.00970, 2405.02821, 2308.00219, 2512.02457, 2611.06074, 2609.37084, 2605.06595. GRAM 2602.03307: arXiv HTML not available and the PDF request returned 404; not read.

## Part 4: what is already known outside world models (2026-11-19)

Requested at the same time, for the user's worry that the answer is obvious and NaVLA2 already shows it. Written by a separate worker.

Written 2026-11-19 by a read-only worker. Nothing here was edited in the repo or committed. Labels: verified = read in the named source this session; recalled = from memory or an earlier survey file, not reopened; speculative = my reading. Downloads (untrusted, never executed) are in the scratchpad downloads folder.

### Is the answer obvious already?

Partly yes. "Audio helps audio-goal navigation" is well established and the user should not present it as open: SoundSpaces (2019/2020), Look Listen and Act, SAVi, Younes et al. and NaVLA2 each show an agent with sound beating the same kind of agent without it (numbers below). "The sound has to be spatial for the agent to find the source" is also built into the task in these papers: the goal itself makes no visible appearance, so the agent has to use direction cues from two ears. What is NOT settled by a clean test: I found no paper in this lane that trains the same agent on one-channel audio and on two-channel audio with everything else fixed. The only head-to-head binaural versus mono-downmix result I found is NaVLA2's Table 2, and it is confounded (different encoders, one run per row, no seeds, split not named). Even there, the spatial branch gave a larger gain than the mono branch, so a reviewer will say "the direction is the obvious one". So the honest framing for the PI: the direction of the answer is expected; what is not published is a controlled version of it, inside a pretrained video world action model, and split by whether the target is on camera.

### 1. NaVLA2 binaural versus mono (verified)

Source: Fan, Chen, Li et al., "NaVLA2: A Vision-Language-Audio-Action Model for Multimodal Instruction Navigation", AAAI-26 proceedings PDF https://tanmingkui.github.io/files/publications/NaVLA2.pdf (read with pdftotext; the PDF has the main paper and references but no appendix, although the text points to one; no arXiv listing was found, per the PR #68 file's search log). Version read: the AAAI-26 camera-ready PDF as served 2026-11-19.

What the ablation is. Section 4.1 (Spatial-Semantic Audio Encoder) says: "Specifically, the binaural audio is downmixed to mono and encoded by the semantic audio encoder, followed by a learnable semantic audio adapter. In parallel, the original binaural signal is fed into a spatial audio encoder to extract directional cues, with its output passed through a spatial audio adapter. The resulting semantic and spatial embeddings are then concatenated." So "mono" there is a branch (mono downmix into CLAP), and "binaural" is another branch (binaural into SpatialAST). It is not the same encoder fed one or two channels. Section 4.2: "We use Vicuna-7B-v1.5 as the LLM ..., CLIP ViT-L/14 as the vision encoder, CLAP and SpatialAST as semantic and spatial audio encoders, respectively."

Table 2 (Section 5.2, "Effect of Spatial Audio Perception"), exact values, columns Spatial, Semantic, SR, SPL, NE:
- neither branch: SR 0.206, SPL 0.162, NE 6.677
- semantic (mono downmix, CLAP) only: SR 0.219, SPL 0.168, NE 6.631
- spatial (binaural, SpatialAST) only: SR 0.258, SPL 0.201, NE 6.147
- both: SR 0.272, SPL 0.196, NE 6.242
(The text extraction listed the columns as Spatial then Semantic; the Section 5.2 sentences agree with this reading: "Adding only the semantic branch yields a modest improvement in SR and SPL, while adding only the spatial branch leads to a larger gain across all metrics". Note that SPL for "both" (0.196) is lower than for spatial-only (0.201), so "across all metrics" is not true of the both row.) The full-model SR of 0.272 matches Table 1 (NaVLA2, SR 0.272, SPL 0.196).

Metric: success rate within 3 m of the goal (Section 5.1), plus SPL and navigation error. Split: Section 5.1 says "All evaluations are conducted in previously unseen environments." The paper does not say whether Table 2 uses the 2,628-episode test set (36 scenes) or the 360-episode test-mini subset (both described in Section 3.3). Runs or seeds: none stated anywhere; I searched the text for seed, variance and standard deviation and found nothing. One number per row, no error bars.

Everything that differs between the rows besides channel count:
- Different pretrained encoders (CLAP versus SpatialAST) and different adapters (two-layer MLPs, one per branch). Adding the spatial branch also adds a second set of audio tokens and a second encoder's capacity.
- Stage 1 alignment data differs per branch: 271K semantic audio-text pairs, and 813K spatial audio-text pairs made by convolving clips with room impulse responses and adding direction text (Section 4.2). Stage 2 instruction tuning (1M samples) includes data from BAT (spatial audio QA).
- Not stated: whether the ablation rows were retrained through all three stages or only re-finetuned, and whether training used the same budget.
- Audio exists only in some episodes: the audio sensor is "dynamically enable[d] ... only in episodes where an object is described via an auditory reference" (Section 3.4), and each object is shown through one randomly chosen modality (Section 3.3). So the success rate is averaged over all episodes, most of which may not have a sound source at all, and the paper gives no split by whether audio was present. Also, the instruction says where the sound-referenced object is in words ("finally ... draining sound", Fig. 3), not a sound-seeking goal as in SoundSpaces.
- The ablation does not report a vision-only agent in the sense of "no audio and no audio instruction": the no-audio row still gets the instruction text.

Size of the effect, in plain words: with no audio branch the agent succeeded about one time in five; with both branches about 27 times in 100 (difference 6.6 points, one run). Spatial-only was about 4 points above mono-only on SR. Without error bars a reviewer can call all of these within run noise (speculative).

### 2. Audio-visual navigation literature

#### SoundSpaces (arXiv 1912.11474 v3, ECCV 2020), verified
- Audio versus no audio: Table 2 ("Adding sound to sight and GPS sensing improves navigation performance significantly. Values are success rate normalized by path length (SPL)"). It compares PointGoal (GPS plus vision) with AudioPointGoal (GPS, vision and binaural sound). Replica, SPL: Blind 0.480 to 0.681, RGB 0.521 to 0.632, Depth 0.601 to 0.709. Matterport3D, SPL: Blind 0.426 to 0.473, RGB 0.466 to 0.521, Depth 0.541 to 0.581. (Column assignment read from the table text and cross-checked against Table 3, whose Replica Blind row shows PG 0.480 and APG 0.681 for the same sound.) Text, O.1 "Does audio help navigation?": "Audio improves accuracy significantly, showing the clear value in multi-modal perception for navigation."
- AudioGoal (sound only, no GPS): AudioGoal-Depth SPL 0.756 (Replica) and 0.552 (Matterport3D) (from the text of the "Audio as a learned spatial sensor" paragraph). Table 3 also gives AudioGoal Blind (no vision) 0.673 on Replica for the same sound, which is close to depth (0.756): a vision-free agent that can hear works well.
- Mono versus binaural: not tested. The platform renders binaural by design. The closest thing is a variant whose audio input is only the left and right intensities, with the audio CNN removed: "The performance of the AudioGoal-Depth agent drops to an SPL of 0.291 and 0.014" (Replica, Matterport3D), against 0.756 and 0.552 for the full spectrogram model. That removes spectral content, not the second channel (the intensity version still has two channels), so it is not a mono test.
- Variance: Table 3 says "Results are averaged over 7 test runs; all standard deviations are <= 0.01". Table 2 states no variance. Whether runs are training seeds or only test runs is not stated.

#### AV-WaN, Chen et al., ICLR 2021 (arXiv 2008.09622 v3), verified
- Audio versus no audio: not reported. Table 2 ablates the acoustic map A_t, the geometric map G_t and the waypoint predictor, all with audio present.
- Mono versus binaural: not reported. The appendix only says the direct-sound intensity is computed from the waveform "averaged between the left and right channels", as a feature, not as an ablation.
- Variance: Table 2 caption: "Results are averaged over 5 test runs; all standard deviations are <= 0.5."

#### SAVi, Chen et al., CVPR 2021 (arXiv 2012.11583), verified
- Audio versus no audio: Table 1 includes "ObjectGoal RL: An end-to-end RL policy with a GRU encoder and RGB-D inputs (no audio)", given the true goal category. Success: heard sounds 1.5 against SAVi 33.9; unheard sounds 1.5 against SAVi 24.8 (the 24.8 matches Table 3's full model). Caveat: this is a different task formulation (category-given), not a controlled audio ablation of one model.
- Mono versus binaural: not reported.
- Variance: none found (searched seed, standard deviation, test runs, variance).

#### Look, Listen and Act, Gan et al., ICRA 2020 (arXiv 1912.11684 v2), verified
- Audio versus no audio: Table I, success/SPL averaged over rooms and sounds: A3C (V) vision only: Ring 9/8.4, Alarm 8.4/7.6, Clock 8.8/8.0; A3C (V+A): Ring 18.4/14.5, Alarm 18/14.9, Clock 20.2/16.1. Audio about doubles success for that baseline. This is a small-room test environment of a different design than SoundSpaces.
- Mono versus binaural: not found. Variance: none found.

#### Move2Hear, Majumder et al., ICCV 2021 (arXiv 2105.07142), verified as to scope
It is active audio-visual source separation, not goal navigation (the agent succeeds if it separates the target sound, "not if it simply travels to where the sound ... is"). Its input is a binaural mixture; the output is monaural. No mono-versus-binaural or audio-versus-no-audio navigation table (searched mono/no audio terms; hits were only the task definition).

#### Younes et al., "Catch Me If You Hear Me" / Dynamical AV Navigation (arXiv 2111.14843 v4, RA-L), verified
- Audio ablation: supplementary Table S.5 ("Encoder ablation: w/o refers to the model without the specified encoder", static Audio Goal task, Matterport3D, complex scenarios). Multiple-heard SPL, SR, SNA: w/o Audio 55.6, 78.8, 41.5; w/o Audio-Spatial 56.3, 79.4, 41.7; full model 60.1, 82.2, 44.4 (SR column order read as 78.8, 79.4, 82.2 for the three rows). Unheard SPL, SR, SNA: 49.6, 72.1, 37.0; 48.8, 70.2, 35.6; 52.6, 72.5, 37.9. The supplement says "We find both the audio and the audio-spatial encoder to be important". The "Audio-Spatial" encoder maps the binaural spectrogram into a spatial feature space, but the ablation removes an encoder, it does not feed the same encoder one channel. "w/o Audio" removes the audio encoder; the paper's wording does not make clear if the agent is then fully deaf, so I do not call it a no-audio baseline.
- Seeds: stated in the main text: "We report the mean over three models trained with different random seeds." (Whether Table S.5 follows this is not said.)

#### Later work 2022-2026 (arXiv, mostly preprints, not peer-reviewed), verified by opening each PDF text
Opened and searched for mono/monaural/single-channel/no-audio/vision-only terms: 2201.04279 (thesis), 2210.01353, 2202.10910, 2206.00393, 2212.11345, 2306.04047 (CAVEN), 2308.00219, 2308.10306, 2405.02821, 2509.16924, 2509.21377, 2509.22698, 2509.25652, 2510.11760, 2601.08868, 2603.19660, 2604.02389, 2604.02390, 2604.02391, 2604.05007, 2607.13110, 2609.17420, 2609.17421, 2609.32224, 2610.14077. None reports a mono (one-channel) agent against a binaural agent with the rest fixed. What I did find:
- 2604.05007 (Binaural Difference Attention, v1): Table II ablation removes the binaural-difference module (BDA) and a transition-prediction loss on the Replica dataset. Text: "Without BDA, the Unheard SR decreases from 68.6% to 63.4%, showing that modeling interaural differences is crucial for generalizing to novel sound categories." (The table row for the full model shows 63.4 in the same column as printed, so text and table do not obviously agree; I did not reconcile them.) This removes a left-right fusion module, it does not remove a channel.
- 2509.16924 (stereo-aware attention, v1): Table 3 ablates its stereo attention module (SAM) and fusion module (AGDF). Again a module, not a channel. It also reports an audio-only (no vision) variant ("even without visual input ... SPL of 63.2" on a Matterport3D setting, per the text).
- 2405.02821 (sim2real, v2): its simulator uses "a mono receiver", and they average a real two-channel recording to mono (Sec. on real data). That is a design choice, not an experiment on mono versus binaural.
- SoundSpaces 2.0 (2206.08312 v2) lists "monaural single-channel audio, binaural ..., and ambisonics" as supported microphones, so a mono experiment is easy to run there, but the paper gives no such result.
- Seeds or variance in the newer ones: several report "averaged over 5 test runs" with a standard deviation (2202.10910, 2210.01353); I did not collect the rest.
Not read, only named in the PR #68 survey: AVWM 2512.00883 (its v3 and v4; the arXiv API now lists v4 under the title "LH-AVLN") and its audio/vision forecast ablation, labelled recalled here. See the 2026-11-16 prior-work file for that.

#### Not found, and where I looked
- arXiv API (export.arxiv.org): title search for "audio-visual navigation", "audio visual navigation", "audio-goal", "audio-visual embodied" (32 entries, listed above, newest 2026-10-08); abstract searches combining "audio-visual navigation" with binaural/monaural/mono/ablation (3 hits, none a mono experiment).
- Semantic Scholar graph API: four queries on binaural versus monaural navigation; the first returned one paper (DSSNav 2026, not opened), the other three were rate-limited after four retries each. So the Semantic Scholar side of the search is incomplete.
- Full-text grep of the 30 opened PDFs for mono, monaural, single-channel, one-channel, without audio, no audio, vision-only (hits listed above).
A mono-versus-binaural result may exist in a paper I did not open (speculative); say "I did not find" not "does not exist".

### 3a. What a skeptical reviewer would call already known

1. Audio helps goal navigation when the goal makes sound: SoundSpaces Table 2 and the AudioGoal results, Look Listen and Act Table I, SAVi Table 1 (no-audio ObjectGoal baseline), NaVLA2 Table 2 (no audio row versus both). verified
2. A blind agent that can hear can come close to a seeing agent in SoundSpaces (AudioGoal Blind 0.673 versus Depth 0.756 on Replica, Table 3), so audio alone carries most of the navigation signal there. verified
3. Two-channel (binaural) sound lets the agent read direction, and sound-only intensity cues are not enough in big houses: SoundSpaces' intensity-only variant fails on Matterport3D. verified
4. Spatial audio features beat or add to semantic-only features: NaVLA2 Table 2 (spatial-only 0.258 against semantic-only 0.219 SR, confounded) and Younes Table S.5 (both encoders needed). verified, with the confounds above.
5. Audio in a world model helps prediction: Audio-Visual World Models (2512.00883) and Multisensory Foresight; per the earlier survey files, recalled here.
Plain consequence for the PI meeting: "does audio help" and "does the directional channel help" are expected. A reviewer will not accept either as the finding by itself.

### 3b. What would still be new (speculative, my reading)

Question 1 (spatial versus mono versus no audio inside a Wan 2.2 navigation world action model, split by target visibility, plus predicted-frame quality out of view):
- A controlled one-channel against two-channel comparison with the same encoder and the same training, in any agent: I found none, so even outside world models this would be new. speculative, bounded by the search above.
- The same three-way comparison inside a pretrained video world model: none found in the survey files (recalled from the PR #51 and PR #68 searches, not rerun).
- The split by target visibility: no navigation paper opened reports it; SoundSpaces' target "does not have a visible embodiment", so the split cannot exist there (verified, SoundSpaces Section on tasks). Building a setting where the target can be visible or hidden is needed.
- Predicted-frame quality for regions out of view, as a function of audio type: not found.
- Caveat the reviewer will raise: a sensible prior says spatial helps most when the target is hidden and does not matter when it is visible, so the interesting result would be a surprise (for example mono matching spatial, or audio hurting prediction), not the expected direction.

Question 2 (does an encoder's rank on an upstream spatial-audio benchmark predict navigation success):
- I did not search this question in this task; it was covered by the PR #68 survey. I found nothing in the papers opened here that ties encoder benchmark rank to navigation success: the navigation papers compare fusion modules, not benchmark-ranked pretrained spatial encoders (NaVLA2 uses SpatialAST without comparing alternatives). speculative.
- Risk: with few encoders the rank correlation will have very few points, and one seed per configuration (the mentor's current advice in the user's notes) gives no noise estimate.

### Method note
Opened by me this session: NaVLA2 PDF; arXiv PDFs 1912.11474 v3, 2008.09622 v3, 2012.11583, 1912.11684 v2, 2105.07142, 2202.10910, 2210.01353 v2, 2111.14843, 2206.08312 v2 and the 24 later preprints named above; read by pdftotext, with table text taken from the extraction (column layouts for SoundSpaces Table 2, SAVi Table 1 and Younes Table S.5 were inferred from column order and checked against a second number in the same paper). Not checked: OpenReview versions of AV-WaN, and the camera-ready versus arXiv numbers for any paper.
