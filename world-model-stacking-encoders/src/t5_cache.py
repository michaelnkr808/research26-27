"""Precompute the T5 instruction embeddings the proxy cross-attends to, as Fast-WAM does (scripts/precompute_text_embeds.py
at FastWAM 7faa711): umT5-XXL encoder of Wan2.2-TI2V-5B (bf16), Wan2.1 tokenizer, 128 tokens, prompt template
DEFAULT_PROMPT, one file per prompt named sha256(prompt).t5_len128.wan22ti2v5b.pt holding `context` (128, 4096) bf16 and
`mask` (128,) bool. Same names and contents, so Fast-WAM's own cache and ours are interchangeable.

Instructions come from the training meta (extract_fastwam.py --source meta) and, for evaluation, from a text file with one
instruction per line (LIBERO-Plus instructions after removing the appended perturbation descriptions, App. C.2).

Example:
  python t5_cache.py --meta /feats/meta --extra lp_instructions.txt --out /feats/t5 \
      --text-encoder /models/models_t5_umt5-xxl-enc-bf16.safetensors --tokenizer /models/google/umt5-xxl
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CONTEXT_LEN = 128
ENC_ID = "wan22ti2v5b"           # _model_id_to_enc_id("Wan-AI/Wan2.2-TI2V-5B")
BATCH = 16                       # DEFAULT_BATCH_SIZE


# fastwam/datasets/lerobot/robot_video_dataset.py DEFAULT_PROMPT (copied: importing that module needs Fast-WAM's full
# dataset stack; test_proxy_wam.py checks the copy against the source)
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def prompt_of(task):
    return DEFAULT_PROMPT.format(task=task)


def cache_path(root, prompt):
    return Path(root) / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{CONTEXT_LEN}.{ENC_ID}.pt"


def load_context(root, task):
    """(context (128, 4096) float32, mask (128,) bool) for a task, after Fast-WAM's dataset step: padded positions
    zeroed and the mask set to all ones (robot_video_dataset.py: context[~mask] = 0; context_mask = ones)."""
    payload = torch.load(cache_path(root, prompt_of(task)), map_location="cpu")
    ctx, mask = payload["context"].float(), payload["mask"].bool()
    assert ctx.shape == (CONTEXT_LEN, 4096) and mask.shape == (CONTEXT_LEN,), (ctx.shape, mask.shape)
    ctx[~mask] = 0.0
    return ctx, torch.ones_like(mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="dir with <suite>.npz from extract_fastwam.py --source meta")
    p.add_argument("--extra", default=None, help="text file, one more instruction per line")
    p.add_argument("--out", required=True)
    p.add_argument("--text-encoder", required=True, help="models_t5_umt5-xxl-enc-bf16 weights file")
    p.add_argument("--tokenizer", required=True, help="google/umt5-xxl tokenizer dir (Wan2.1-T2V-1.3B repo)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    tasks = []
    for f in sorted(Path(args.meta).glob("*.npz")):
        tasks += np.load(f)["tasks"].tolist()
    if args.extra:
        tasks += [l.strip() for l in Path(args.extra).read_text().splitlines() if l.strip()]
    prompts = list(dict.fromkeys(prompt_of(t) for t in tasks))
    print(f"{len(tasks)} instructions -> {len(prompts)} unique prompts", flush=True)

    enc = _load_registered_model(args.text_encoder, "wan_video_text_encoder", torch_dtype=torch.bfloat16,
                                 device=args.device).eval()
    tok = HuggingfaceTokenizer(name=args.tokenizer, seq_len=CONTEXT_LEN, clean="whitespace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    over = 0
    with torch.no_grad():
        for s in range(0, len(prompts), BATCH):
            batch = prompts[s:s + BATCH]
            ids, mask = tok(batch, return_mask=True, add_special_tokens=True)
            ids, mask = ids.to(args.device), mask.to(args.device, dtype=torch.bool)
            over += int(mask.all(dim=1).sum())
            ctx = enc(ids, mask)
            for i, pr in enumerate(batch):
                torch.save({"context": ctx[i].to("cpu", torch.bfloat16).contiguous(),
                            "mask": mask[i].to("cpu", torch.bool).contiguous()}, cache_path(out, pr))
    (out / "_meta.json").write_text(json.dumps({"n_prompts": len(prompts), "over_length": over,
                                                "text_encoder": Path(args.text_encoder).name,
                                                "context_len": CONTEXT_LEN}, indent=1))
    print(f"wrote {len(prompts)} files to {out} ({over} over {CONTEXT_LEN} tokens)", flush=True)


if __name__ == "__main__":
    main()
