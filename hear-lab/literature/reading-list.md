# Reading list

What has been read for this direction, and by whom. Citation details live in `references.bib`; this file only tracks reading status. Paper notes are the lead's own and are not written by agents.

## Read by the lead
Reported as read by the lead on 2026-11-15. Titles, authors, and identifiers were checked on 2026-11-15 against the arXiv listing or the publisher page. The venue in the last column is as the lead gave it, and is marked where it was also confirmed from a primary page.

| Key | Paper | Identifier | Venue |
|---|---|---|---|
| `fan2026navla` | NaVLA²: A Vision-Language-Audio-Action Model for Multimodal Instruction Navigation | doi:10.1609/aaai.v40i22.38886 | AAAI 2026 as the lead gave it; the publisher page shows the AAAI proceedings, volume 40, dated 2026-03-14 |
| `lee2025deepasa` | DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis | arXiv:2509.17247 | NeurIPS 2025 (confirmed, arXiv comment field) |
| `chen2026rmsaqa` | RMS-AQA: A Two-Stage Spatial Audio Question Answering Benchmark for Real-World Domestic Environments | arXiv:2610.00935 | ICASSP 2027 Grand Challenge (confirmed, challenge site); preprint, not peer reviewed |
| `luo2026sail` | SAIL: Spatial Audio Intelligence with Large Language Models via Disentangled Acoustic-Spatial Encoding and Dual-Stream Q-Former | arXiv:2609.34347 | Submitted to IEEE/ACM TASLP (as the lead gave it, not confirmed); preprint, not peer reviewed |

## Read by agents
- `chen2026rmsaqa`: the planning lead read the v1 web version and the baseline code repository's README on 2026-11-15, for planning the first task only (benchmark format, metric, baseline, compute). Not a full study of the paper.

- The papers in `survey/` (PRs #49 and #50, 2026-11-16) were opened by survey workers and reviewers, with the read depth recorded per paper in each survey file. Most were read in part, for the survey's questions only. None of them counts as read by the lead.

## To read
A read-first list for the lead, picked by the planning lead from the survey (2026-11-16). Why each one matters is in `survey/2026-11-16-summary.md`; the entry with link and read depth is in the survey file named. Order is the suggested reading order.

| Key | Paper | Why read it | Survey file |
|---|---|---|---|
| `wang2025audio` | Audio-Visual World Models (arXiv 2512.00883) | The closest prior work: predicts frames and binaural audio under navigation actions. Read for overlap with the direction. | C, D |
| `wang2026world` | World Action Models: The Next Frontier in Embodied AI (arXiv 2605.12090) | A survey that defines "world action model"; gives the vocabulary and the map. | C |
| `huang2026wnm` | WNM-3D (arXiv 2608.07267) | A navigation world action model on Wan 2.2 with training code and checkpoints; a candidate base. | D |
| `chen2025astranav` | AstraNav-World (arXiv 2512.21714) | The other Wan 2.2 navigation world action model; compare its design with WNM-3D. | D |
| `chen2022soundspaces2` | SoundSpaces 2.0 (arXiv 2206.08312) | The simulator that renders spatial audio in Habitat; how NaVLA2's data was made. | B, E |
| `zheng2024bat` | BAT (arXiv 2402.01591) | Source of SpatialAST, the spatial audio encoder NaVLA2 uses; a candidate encoder. | B |
| `liu2026omniecho` | OmniEcho (arXiv 2609.23407) | First-order Ambisonics encoder attached to an omni model, with a public real-recording navigation benchmark. | B, E |
| `nvidia2026cosmos` | Cosmos 3 technical report (audio and policy sections) | The one world action model family with an audio path; shows what is and is not tested. | C |
| `hiratsuka2026s2a2` | S2A2 (arXiv 2607.26047) | Spatial audio in a real-robot policy, with an ablation; background on what "spatial helps" evidence looks like. | A, B |

File letters: A `survey/2026-11-16-audio-in-robot-policies.md`, B `survey/2026-11-16-spatial-audio-in-embodied-ai.md`, C `survey/2026-11-16-world-action-models.md`, D `survey/2026-11-16-navigation-world-models.md`, E `survey/2026-11-16-spatial-audio-navigation-data.md`.
