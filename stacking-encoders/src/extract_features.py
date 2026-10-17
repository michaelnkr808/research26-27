"""Extract frozen vision-encoder features from LIBERO demo HDF5 files.

Phase 1 of stacking-encoders: cache features once, then compute diagnostics
(CKA, residual probes) and train proxy policies from the cache.

Input: LIBERO HDF5 files (yifengzhu-hf/LIBERO-datasets layout:
data/demo_k/obs/<cam>). Use --inspect to print the real layout of a file
before trusting the defaults.

Output: one HDF5 per input file, at <out>/<encoder>/<suite>/<file>.h5, with
  demo_k/<cam>/global  (T, D)        encoder-native summary (CLS / pooler / RADIO summary)
  demo_k/<cam>/mean    (T, D)        mean over patch tokens
  demo_k/<cam>/grid    (T, g, g, D)  patch tokens average-pooled to g x g
All stored as float16. Run metadata goes in the file attributes.

Example:
  python extract_features.py --encoder siglip2 --out /data/feats \
      --hdf5 /data/LIBERO-datasets/libero_spatial/*.hdf5
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Encoder registry. Each loader returns an object with:
#   .name, .hf_id, .resolution
#   .encode(uint8 NHWC numpy) -> (global [N,D], tokens [N,h,w,D]) float tensors
# Model variants are defaults, NOT confirmed to match liu2026what's choices.
# ---------------------------------------------------------------------------

ENCODERS = {}


def register(name):
    def deco(fn):
        ENCODERS[name] = fn
        return fn
    return deco


def _to_float_nchw(frames, size, mean, std, device, dtype):
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    if mean is not None:
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
        x = (x - m) / s
    return x.to(dtype)


def _tokens_to_grid(tokens):
    n, t, d = tokens.shape
    side = int(round(t ** 0.5))
    assert side * side == t, f"patch tokens ({t}) are not a square grid"
    return tokens.reshape(n, side, side, d)


class _HFEncoder:
    def __init__(self, name, hf_id, resolution, device, dtype):
        self.name, self.hf_id, self.resolution = name, hf_id, resolution
        self.device, self.dtype = device, dtype


@register("dinov3")
def load_dinov3(device, dtype, variant=None):
    from transformers import AutoImageProcessor, AutoModel
    hf_id = variant or "facebook/dinov3-vitl16-pretrain-lvd1689m"  # gated: accept license on HF
    enc = _HFEncoder("dinov3", hf_id, 224, device, dtype)
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, dtype=dtype).to(device).eval()
    n_skip = 1 + getattr(model.config, "num_register_tokens", 0)  # CLS + registers

    @torch.no_grad()
    def encode(frames):
        x = _to_float_nchw(frames, enc.resolution, proc.image_mean, proc.image_std, device, dtype)
        out = model(pixel_values=x)
        h = out.last_hidden_state
        return h[:, 0], _tokens_to_grid(h[:, n_skip:])

    enc.encode = encode
    return enc


@register("siglip2")
def load_siglip2(device, dtype, variant=None):
    from transformers import AutoImageProcessor, AutoModel
    hf_id = variant or "google/siglip2-so400m-patch14-384"
    proc = AutoImageProcessor.from_pretrained(hf_id)
    res = proc.size["height"]
    enc = _HFEncoder("siglip2", hf_id, res, device, dtype)
    vision = AutoModel.from_pretrained(hf_id, dtype=dtype).vision_model.to(device).eval()

    @torch.no_grad()
    def encode(frames):
        x = _to_float_nchw(frames, res, proc.image_mean, proc.image_std, device, dtype)
        out = vision(pixel_values=x)
        return out.pooler_output, _tokens_to_grid(out.last_hidden_state)

    enc.encode = encode
    return enc


@register("c-radiov4")
def load_cradiov4(device, dtype, variant=None):
    from transformers import AutoModel
    hf_id = variant or "nvidia/C-RADIOv4-SO400M"
    enc = _HFEncoder("c-radiov4", hf_id, 448, device, dtype)
    # Pin the remote code: trust_remote_code runs files from the HF repo.
    revision = None if variant else "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"  # checked 2026-10-17
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, revision=revision).to(device).eval()
    model = model.to(dtype)

    @torch.no_grad()
    def encode(frames):
        # RADIO normalizes internally; it expects RGB in [0, 1].
        x = _to_float_nchw(frames, enc.resolution, None, None, device, dtype)
        summary, spatial = model(x)
        return summary, _tokens_to_grid(spatial)

    enc.encode = encode
    return enc


def _not_yet(name, how):
    def loader(device, dtype, variant=None):
        raise NotImplementedError(f"{name}: loader not written yet. Weights: {how}")
    ENCODERS[name] = loader


_not_yet("vjepa2.1", "official .pt files via github.com/facebookresearch/vjepa2 (needs its model code)")
_not_yet("vggt-omega", "facebook/VGGT-Omega (gated, needs vggt package)")
_not_yet("sam3", "facebook/sam3 (gated)")
_not_yet("unitok", "FoundationVision/unitok_tokenizer (needs UniTok repo code)")
_not_yet("hrp", "no public release found 2026-10-17")

# ---------------------------------------------------------------------------


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def inspect(path):
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"{path}: {len(demos)} demos")
        f["data"][demos[0]].visititems(
            lambda k, v: print(f"  {demos[0]}/{k}", getattr(v, "shape", ""), getattr(v, "dtype", "")))
        print("  data attrs:", dict(f["data"].attrs))


def pool(glob, grid, g):
    n, h, w, d = grid.shape
    x = grid.permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(x, g).permute(0, 2, 3, 1)
    return {"global": glob.float(), "mean": grid.float().mean(dim=(1, 2)), "grid": pooled}


def extract_file(enc, src, dst, args):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".h5.partial")
    n_frames = 0
    with h5py.File(src, "r") as fin, h5py.File(tmp, "w") as fout:
        demos = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.max_demos:
            demos = demos[: args.max_demos]
        for demo in demos:
            for cam in args.cams:
                frames = fin["data"][demo]["obs"][cam][:: args.frame_stride]
                if args.rotate180:
                    frames = frames[:, ::-1, ::-1]
                frames = np.ascontiguousarray(frames)
                outs = {"global": [], "mean": [], "grid": []}
                for i in range(0, len(frames), args.batch_size):
                    glob, grid = enc.encode(frames[i: i + args.batch_size])
                    for k, v in pool(glob, grid, args.grid).items():
                        outs[k].append(v.cpu().numpy().astype(np.float16))
                for k, v in outs.items():
                    fout.create_dataset(f"{demo}/{cam}/{k}", data=np.concatenate(v))
                n_frames += len(frames)
        fout.attrs["meta"] = json.dumps({
            "source_file": str(src), "encoder": enc.name, "hf_id": enc.hf_id,
            "resolution": enc.resolution, "cams": args.cams, "frame_stride": args.frame_stride,
            "rotate180": args.rotate180, "grid": args.grid, "dtype": args.dtype,
            "n_demos": len(demos), "git_commit": git_commit(),
        })
    os.replace(tmp, dst)
    return n_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", nargs="+", required=True)
    p.add_argument("--encoder", choices=sorted(ENCODERS))
    p.add_argument("--variant", default=None, help="override the HF model id")
    p.add_argument("--out", default="features")
    p.add_argument("--cams", nargs="+", default=["agentview_rgb", "eye_in_hand_rgb"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--max-demos", type=int, default=0, help="0 = all (smoke tests use small values)")
    p.add_argument("--rotate180", action=argparse.BooleanOptionalAction, default=True,
                   help="LIBERO renders upside down (recalled; check visually with --dump-frame)")
    p.add_argument("--dump-frame", default=None, help="save one raw + one processed frame as PNG and exit")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inspect", action="store_true", help="print HDF5 layout and exit")
    args = p.parse_args()

    if args.inspect:
        for path in args.hdf5:
            inspect(path)
        return
    if args.dump_frame:
        from PIL import Image
        with h5py.File(args.hdf5[0], "r") as f:
            demo = sorted(f["data"].keys())[0]
            for cam in args.cams:
                fr = f["data"][demo]["obs"][cam][0]
                Image.fromarray(fr).save(f"{args.dump_frame}_{cam}_raw.png")
                Image.fromarray(np.ascontiguousarray(fr[::-1, ::-1])).save(f"{args.dump_frame}_{cam}_rot180.png")
        return
    if not args.encoder:
        p.error("--encoder is required")

    dtype = getattr(torch, args.dtype)
    enc = ENCODERS[args.encoder](args.device, dtype, args.variant)
    print(f"[extract] encoder={enc.name} hf_id={enc.hf_id} res={enc.resolution} device={args.device}", flush=True)
    for src in map(Path, args.hdf5):
        dst = Path(args.out) / enc.name / src.parent.name / (src.stem + ".h5")
        if dst.exists():
            print(f"[extract] skip (exists) {dst}", flush=True)
            continue
        t0 = time.time()
        n = extract_file(enc, src, dst, args)
        dt = time.time() - t0
        print(f"[extract] {src.name}: {n} frames in {dt:.1f}s ({n / dt:.1f} frames/s)", flush=True)


if __name__ == "__main__":
    main()
