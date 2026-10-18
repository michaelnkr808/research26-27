# Encoder weight availability (checked 2026-10-17, agent; user to verify)

Checked through the Hugging Face API and the official GitHub READMEs. "Variant" is a candidate only: **which size or variant liu2026what used is not confirmed.** Check the paper (App. / Sec. 3.2) and record it here.

| Paper name | Candidate source | Access | License tag | Loader in `src/extract_features.py` |
|---|---|---|---|---|
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` (also S/B/H/7B) | **gated (manual)** | other | written, untested (no HF token) |
| SigLIP 2 | `google/siglip2-so400m-patch14-384` (many sizes) | open | apache-2.0 | **smoke-tested** (base-224 variant) |
| C-RADIOv4 | `nvidia/C-RADIOv4-SO400M`, `nvidia/C-RADIOv4-H` | open, remote code (pinned) | other | **smoke-tested** (SO400M) |
| V-JEPA 2.1 | official `.pt` files in github.com/facebookresearch/vjepa2 README (ViT-B/L/g/G, 384) | open | (repo) | stub |
| UniTok | `FoundationVision/unitok_tokenizer` | open | apache-2.0 | stub (needs UniTok code) |
| SAM 3 | `facebook/sam3` | **gated (manual)** | other | stub |
| VGGT-Ω | `facebook/VGGT-Omega` | **gated (manual)** | other | stub |
| VC-1 | `facebook/vc1-large`, `vc1-base` | open | cc-by-nc-4.0 | – |
| VideoMAE V2 | `OpenGVLab/VideoMAEv2-{Base,Large,Huge,giant}` | open | cc-by-nc-4.0 | – |
| LingBot-Vision | `robbyant/lingbot-vision-vit-{small,base,large,giant}` | not checked for gating | – | – |
| Qwen3.5 | `Qwen/Qwen3.5-*` (vision tower of an LLM; which size?) | open | – | – |
| DynaFLIP | `jlee-larr/dynaflip-base` (uploader is the first author, per user check; model card says "more details coming soon") | open | apache-2.0 | – |
| HRP | https://www.cs.cmu.edu/~data4robotics/hrp/ (CMU project site, not HF; page returns HTTP 200, contents not checked by agent) | open | – | – |

Notes:
- Gated repos need the license accepted on HF, plus an `HF_TOKEN` stored as a Kubernetes Secret (never in manifests).
- Several licenses are non-commercial (cc-by-nc). That's fine for research, but record it.

## Update 2026-10-17 (user-supplied check, pasted; not re-verified by agent)
- HRP and DynaFLIP corrected above. HRP fine-tunes existing encoders, so there are several HRP checkpoints on different bases. Which base the paper used is unknown.
- VGGT-Ω: FAIR Noncommercial license, manually reviewed, and requests can be denied. **Request early.**
- VideoMAE V2: the largest checkpoints may require a request form (user recalls; unchecked).
- C-RADIOv4: NVIDIA Open Model License.

## Variants (paper App. A, p. 14, per user: "primarily L variants", Qwen3.5-4B, final-layer token features)
| Encoder | Likely variant | Basis |
|---|---|---|
| DINOv3 | ViT-L/16 | App. D.3: 14×14 grid × 1024-d at 224×224 |
| C-RADIOv4 | SO400M or H (no L exists) | release list |
| VGGT-Ω | 1B (`vggt_omega_1b_256_text.pt` or `_512.pt`) | release list |
| DynaFLIP | ViT-B (DINOv2-B init, 0.2B) | model card |
| Qwen3.5 | 4B | App. A |
| Others | L, exact checkpoint/resolution unknown | e.g. V-JEPA 2.1 has several ViT-L checkpoints |

Known-overlap sanity checks for the similarity metrics: C-RADIOv4's teachers include DINOv3-7B, SigLIP2-g, and SAM 3. DynaFLIP is initialized from DINOv2.
Open: exact checkpoints, input resolutions, and the HRP base. Candidate email to the authors.
