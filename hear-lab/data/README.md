# Data, models and code pins used by hear-lab experiments

One entry per item: source, revision or version, access date, license, filtering. Rule: `_standards/03-reproducibility-and-data.md` and `_standards/07-ml-practices.md` (check licenses before use). Nothing listed here is stored in the repo: raw data, checkpoints and scene files stay on the rented machines and on the lead's Mac, and the repo keeps only pointers (revision, sha256, command).

Labels: `verified` means read from the source on the date given. License facts for the Hugging Face items were read by the lead from the Hugging Face API on 2026-11-24; license files of the GitHub repositories were read at the pinned commit through `gh api repos/<owner>/<repo>/license?ref=<commit>` on 2026-11-24 (`verified`). Where a license could not be checked, the entry says "license not verified".

## Rule for what may be committed from InteriorGS-derived outputs

InteriorGS Terms of Use (below) forbid redistributing the downloaded data, in whole or in part. Decision of the user, 2026-11-24 (the lead's recommendation: "commit scores and per-episode tables (success, distances, episode ids); keep scene images and rendered audio out of the repo, release them only as paper figures with a citation"; the repo is public). The user's answer, verbatim: "yes sure thats fine".

- May be committed: scores, summaries, per-episode result tables of our own runs (episode ids, success, distances), sha256 manifests, environment lists, run settings.
- Stays local even though it is a table: anything that carries InteriorGS labels, object or instance ids, scene coordinates or box-derived quantities. This covers the route A check 3 per-episode CSVs (`check3_InteriorGS_*.csv`: object label, `label_ins_id`, start and goal world coordinates, box-point counts). A stripped copy is not part of this PR (reviewer finding on PR #89, 2026-11-24).
- Stay out of the repo: anything holding content copied from InteriorGS itself (scene geometry, object boxes or labels, floor plans, meshes, images, rendered audio). Images and rendered audio may appear only as paper figures, with a citation.
- Enforced by each experiment's `results/.gitignore` (ignore everything, then an explicit allow list). Do not force-add.

## Datasets

### InteriorGS (scenes)
- Source: Hugging Face dataset `spatialverse/InteriorGS` (gated, automatic approval).
- Revision: `334dfeea4e0241033b4e5de97c01bc7c9c080530`.
- Access dates: 2026-11-20 (route A checks: 4 scenes for check 1 on rental 2 and 3; 939 scenes for check 3 on rentals 2 and 4); 2026-11-22 and 2026-11-23 (rung 1: the 839 scenes named in the test_unseen parquet, whole scene folders). Scene files live on the rented machines only; the lead's Mac holds only the small files of the 4 check 1 scenes (the LOG, 2026-11-20, says 4 scenes, small files only).
- License: "InteriorGS Terms of Use", https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf (PDF sha256 `508b0ec62ff1b5bee3c47524dd36f2402f5f70a09afbe62a4d0041078856356b`, read by the lead on 2026-11-24). Terms 1 to 3, quoted:
  1. "Researcher shall use the Database only for non-commercial research and educational purposes."
  2. "Researcher shall not redistribute the downloaded data, in whole or in part, through other media, but shall refer to the Database website (https://github.com/manycore-research/InteriorGS)."
  3. "Researcher shall cite the dataset appropriately in each publication based on Database."
- What may be committed or released (metrics, per-episode tables, images, rendered audio): decided by the user on 2026-11-24, see the rule above.
- Filtering: check 1 uses 4 test_unseen scenes (`0001_839920`, `0470_840871`, `0748_841603`, `0756_840012`); check 3 uses every scene named in the two parquet files it reads; rung 1 uses the scenes of the 5,000 test_unseen episodes (839 scenes, `config.yaml` `data.interiorgs`).

### GN-Matrix (episodes)
- Source: Hugging Face dataset `TeleEmbodied/GN-Matrix`.
- Revision: `50a68253b5436f0595779cd3f91baf65228a0238`.
- Access dates: 2026-11-20 (check 3, both parquet files; check 2 train_seen file on 2026-11-21); 2026-11-22 (rung 1 test_unseen file).
- License: cc-by-nc-4.0 (Hugging Face API, 2026-11-24).
- Filtering: files `InteriorGS/InteriorGS_test_unseen.parquet` (5,000 rows, sha256 `de4fd64a4e27ef552017eebf43aeb02992618b49c8800aa9bbc09dd30a0e1347`) and `InteriorGS/InteriorGS_train_seen.parquet`. Check 3 uses all 5,000 test_unseen episodes and a seeded sample of 1,000 train_seen episodes. Check 2 uses the first 48 train_seen episodes by trajectory_id (46 after conversion). Rung 1 uses the first 1,000 of a frozen random order of the 5,000 test_unseen episodes (`2026-11-21-rung1-wnm3d-baseline/config.yaml`).

### MIT KEMAR HRTF (SOFA file)
- Source: https://sofacoustics.org/data/database/mit/mit_kemar_normal_pinna.sofa (MIT KEMAR, normal pinna).
- Version: file of 1,171,764 bytes, sha256 `e7035994f5fd754058424c061380ee92b1d5ed58fccef2887a4266916616acdf`.
- Access date: 2026-11-20 (check 1 HRTF follow-up, rental 3). Not stored in the repo.
- License: **license not verified.** The experiment lead found no formal license on 2026-11-20; MIT's page says the measurements are "made available to the research community". Cite Gardner and Martin 1994.
- Filtering: none.

## Model checkpoints

### WNM-3D checkpoints
- Source: Hugging Face `TeleEmbodied/WNM-3D`, revision `5b8c08dd52028ef4a29e7b87966f27582032fb4f`, files `wnm_3d_stage3_release/**` (27,113,385,590 bytes; shard sha256 values in `2026-11-21-rung1-wnm3d-baseline/config.yaml`).
- Access dates: 2026-11-21 (check 2), 2026-11-22 and 2026-11-23 (rung 1).
- License: fair-noncommercial-research-license (Hugging Face API, 2026-11-24). The Hub repository at this revision has a `LICENSES/` folder with `APACHE-2.0`, `DINOV3_LICENSE.md` and `FAIR_NONCOMMERCIAL_RESEARCH_LICENSE` (listed through the Hub API on 2026-11-24), and a `THIRD_PARTY_NOTICES.md` file; the checkpoint license is the Hub card's `fair-noncommercial-research-license`.
- Filtering: only the stage 3 release folder.

### VGGT-Omega
- Source: Hugging Face `facebook/VGGT-Omega`, revision `1041e80fc0e911235d3426b0a3d9a81075111a53`, file `vggt_omega_1b_512.pt` (4,576,706,117 bytes, sha256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`). Gated, manual approval.
- Access date: 2026-11-21 (check 2).
- License: FAIR noncommercial research license, `LICENSE.txt` (Hugging Face API, 2026-11-24).
- Filtering: the one file.

### Wan2.2-TI2V-5B
- Source: `Wan-AI/Wan2.2-TI2V-5B`, revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, files `models_t5_umt5-xxl-enc-bf16.pth` and `Wan2.2_VAE.pth`.
- Access date: 2026-11-21 (check 2). License: apache-2.0 (Hugging Face API, 2026-11-24).

### Wan2.1-I2V-14B-480P
- Source: `Wan-AI/Wan2.1-I2V-14B-480P`, revision `6b73f84e66371cdfe870c72acd6826e1d61cf279`, file `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`.
- Access date: 2026-11-21 (check 2). License: apache-2.0 (Hugging Face API, 2026-11-24).

### google/umt5-xxl (tokenizer files only)
- Source: `google/umt5-xxl`, revision `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`, files `tokenizer_config.json`, `tokenizer.json`, `spiece.model`, `special_tokens_map.json`, `config.json`.
- Access dates: 2026-11-21 (check 2), 2026-11-22 (rung 1). License: apache-2.0 (Hugging Face API, 2026-11-24).

## Code pins (cloned on the rented machines, not in the repo)

| Item | Source | Pin | License (read at the pin, 2026-11-24) |
|---|---|---|---|
| RLR-Audio-Propagation | https://github.com/facebookresearch/rlr-audio-propagation | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` (`2026-11-20-route-a-checks/config.yaml`); `libRLRAudioPropagation.so` sha256 `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4` | `LICENSE` file is Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); GitHub's detector reports "Other" |
| GN0 | https://github.com/TeleHuman/GN0 | `c92778118542bf1dd7d437855d06e0d822e8361d` | Apache-2.0 |
| GN-Bench-Tools (GN0 submodule) | https://github.com/TeleHuman/GN-Bench-Tools | `0ce83677532591c582b0c8a2879ab431a1efff18` | MIT |
| WNM-3D code | https://github.com/TeleHuman/WNM-3D | `7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb` | `LICENSE` is Apache-2.0. The README "License" section (at the pin) says: "The vendored VGGT-Ω and DINOv3 source is governed by separate terms" and "WNM-3D as a complete distribution is not licensed solely under Apache-2.0. Use of the vendored VGGT-Ω components and released checkpoints is subject to the FAIR Noncommercial Research License and other applicable third-party terms." The repo's `LICENSES/` folder at the pin holds `FAIR_NONCOMMERCIAL_RESEARCH_LICENSE` and `DINOV3_LICENSE.md` |
| SoundSpaces (settings reference only, read-only, nothing run) | https://github.com/facebookresearch/sound-spaces | `287184fd7067a0385558492716355c54875500ee` | CC-BY-4.0 (`LICENSE` file at the pin) |

Access dates: RLR-Audio-Propagation 2026-11-20; GN0 and GN-Bench-Tools 2026-11-20 (check 3) and 2026-11-22 (rung 1); WNM-3D 2026-11-20 (check 2) and 2026-11-22 (rung 1); SoundSpaces: the access date is not recorded in the config or the LOG entries read, so none is given.
