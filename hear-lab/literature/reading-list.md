# Reading list

What has been read for this direction, and by whom. Citation details live in `references.bib`; this file only tracks reading status. Paper notes are the lead's own and are not written by agents.

## Read by the lead
Reported as read by the lead on 2026-11-15, corrected by the lead on 2026-11-19 (SAIL had been listed by mistake and was removed; WNM-3D added). Titles, authors, and identifiers were checked on 2026-11-15 against the arXiv listing or the publisher page. The venue in the last column is as the lead gave it, and is marked where it was also confirmed from a primary page.

| Key | Paper | Identifier | Venue |
|---|---|---|---|
| `fan2026navla` | NaVLA²: A Vision-Language-Audio-Action Model for Multimodal Instruction Navigation | doi:10.1609/aaai.v40i22.38886 | AAAI 2026 as the lead gave it; the publisher page shows the AAAI proceedings, volume 40, dated 2026-03-14 |
| `lee2025deepasa` | DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis | arXiv:2509.17247 | NeurIPS 2025 (confirmed, arXiv comment field) |
| `chen2026rmsaqa` | RMS-AQA: A Two-Stage Spatial Audio Question Answering Benchmark for Real-World Domestic Environments | arXiv:2610.00935 | ICASSP 2027 Grand Challenge (confirmed, challenge site); preprint, not peer reviewed |
| `huang2026wnm` | WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN | arXiv:2608.07267 | arXiv preprint, not peer reviewed (survey file D) |

## Read by agents
- `chen2026rmsaqa`: the planning lead read the v1 web version and the baseline code repository's README on 2026-11-15, for planning the first task only (benchmark format, metric, baseline, compute). Not a full study of the paper.

- The papers in `survey/` (PRs #49 and #50, 2026-11-16) were opened by survey workers and reviewers, with the read depth recorded per paper in each survey file. Most were read in part, for the survey's questions only. None of them counts as read by the lead.

## To read
Read-first list for the lead, from the planning lead (hear-lab-94, 2026-11-20): papers whose methods the current plan uses, in reading order. Decided or open is as in the decision records; "leading candidate" is the planning lead's reading, not a decision. Identifiers marked "metadata checked" were confirmed against the arXiv API on 2026-11-20 (title, authors, comment field); no paper was re-read for this list. Why each paper matters beyond this table is in `survey/2026-11-16-summary.md` and `survey/2026-11-19-pi-followups-simulator-audio-papers-gaps.md`.

### Read now (in this order)
| # | Paper | Identifier | Part of the plan that uses it | Status |
|---|---|---|---|---|
| 1 | SoundSpaces 2.0 (`chen2022soundspaces2`) | arXiv 2206.08312 (metadata checked) | Data engine: its sound engine (RLR-Audio-Propagation) makes the audio in both simulator routes | Engine decided for both routes, pending check 1 (does it run on a rented Linux machine) |
| 2 | GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation (key not yet in bib; used as `li2026gn0` in the 2026-11-18 survey file) | arXiv 2606.03682 (metadata checked) | WNM-3D's own simulator and episodes (GN-Bench, GN-Matrix): the data and the closed-loop test under route A | Route A chosen first, not committed (decision 003 pending) |
| 3 | BAT (`zheng2024bat`) | arXiv 2402.01591, ICML 2024 (metadata checked) | Audio encoder: source of Spatial-AST, the spatial encoder NaVLA2 and both PI-linked papers use | OPEN; Spatial-AST is the leading candidate |
| 4 | Hear You Are: Teaching LLMs Spatial Reasoning with Vision and Spatial Sound (Ryu, Chung, Harwath; key not yet in bib) | CVPR 2026, open-access PDF on openaccess.thecvf.com (authors and title as read in the PR #71 survey file) | How to attach a spatial audio encoder to a big pretrained model (frozen Spatial-AST plus a small projector); nearest evidence for the visible / not-visible split. The PI linked it | Template only, OPEN; built for a language model, not a video model |
| 5 | Hear, Localize, and Reason: Spatially Aware Scene Understanding for Audio-Visual LLMs (key not yet in bib) | arXiv 2610.21588, NeurIPS 2026 per the comment field (metadata checked) | Same as 4; also its control that feeds left and right channels through the existing mono path, worth copying as a mono baseline. The PI linked it | OPEN |
| 6 | World Action Models are Zero-shot Policies, DreamZero (`ye2026world`) | arXiv 2602.15922 (metadata checked) | Base model design: WNM-3D builds on DreamZero's joint video-action flow backbone | Decided, comes with WNM-3D (decision 002) |
| 7 | Audio-Visual World Models (`wang2025audio`) | arXiv 2512.00883 (metadata checked) | Not a method we use; closest prior work, read for overlap | Reading only |

### Can wait
| # | Paper | Identifier | Part of the plan | Status |
|---|---|---|---|---|
| 8 | SoundSpaces (`chen2020soundspaces`) and Semantic Audio-Visual Navigation, SAVi (`chen2021semantic`) | arXiv 1912.11474 and 2012.11583 | Evaluation: define walking to a sounding object and its success measures; SAVi's episodes are what route B would use | Metrics OPEN (experiment PLAN not written) |
| 9 | VGGT-Omega | Citation NOT CHECKED: the paper behind the frozen 3D encoder in WNM-3D was not found by an arXiv title search on 2026-11-20. Take the reference from WNM-3D's reference list | The existing example of the pattern the audio encoder would follow (frozen encoder, trainable adapter, prefix tokens) | Decided, comes with WNM-3D |
| 10 | Wan: Open and Advanced Large-Scale Video Generative Models | arXiv 2503.20314 (title and authors checked on the arXiv API, 2026-11-20; no separate Wan 2.2 paper was looked for beyond the planning lead's recollection) | The video generator under WNM-3D; background | Background |
| 11 | pi-AVAS (ICCV 2025) | Citation NOT CHECKED: named in the PR #71 fact-check review, not opened by a worker, and an arXiv search on 2026-11-20 found nothing | Precedent for route A's pipeline (mesh from a Gaussian-splat scene, then ray-traced sound) | Read only if route A is committed |
| 12 | Other encoder candidates: OmniEcho (`liu2026omniecho`), SPUR (`sakshi2025spur`), JAEGER (`liu2026jaeger`) | in `references.bib` | Audio encoder, only if Spatial-AST is not chosen | OPEN |

Already read by the lead and still in use: WNM-3D (base model, decided), NaVLA2 (the uncontrolled spatial-versus-mono table we position against). DeepASA and RMS-AQA are not used by the current plan (RMS-AQA parked; its baseline's spatial branch is one more encoder candidate).

Dropped from read-first for now, since no method of theirs is in the plan (still fine as background): AstraNav-World (`chen2025astranav`), the Cosmos 3 technical report (`nvidia2026cosmos`), S2A2 (`hiratsuka2026s2a2`), the World Action Models survey (`wang2026world`).

File letters for the survey files: A `survey/2026-11-16-audio-in-robot-policies.md`, B `survey/2026-11-16-spatial-audio-in-embodied-ai.md`, C `survey/2026-11-16-world-action-models.md`, D `survey/2026-11-16-navigation-world-models.md`, E `survey/2026-11-16-spatial-audio-navigation-data.md`.
