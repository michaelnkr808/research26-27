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
| DynaFLIP | `jlee-larr/dynaflip-base` (unofficial-looking uploader; verify) | open | apache-2.0 | – |
| HRP | **not found** on HF or by GitHub search | – | – | – |

Notes:
- Gated repos need the license accepted on HF, plus an `HF_TOKEN` stored as a Kubernetes Secret (never in manifests).
- Several licenses are non-commercial (cc-by-nc). That's fine for research, but record it.
