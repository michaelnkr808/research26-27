"""Training-versus-rollout feature parity (stage 2 PLAN, "What the implementation session must add", item 2).

The same uint8 frames go through extract_fastwam.encoder_episode (what filled the training caches) and through
rollout.Featurizer.__call__ (what the policy sees at rollout), for dinov3, siglip2 and c-radiov4. The two float16
(n, 14, 28, D) grids must match.

Each path builds its encoder the way it really does: extraction as extract_fastwam.main does
(ENCODERS[source](device, dtype), dtype = the compute dtype of the cache), rollout through the real
Featurizer.__init__ (ENCODERS[s](device, rollout.ENC_DTYPE[s])). Each side's variant (hf_id), input resolution, pinned
revision (C-RADIOv4) and dtype are asserted against the caches' description (data/README.md section 4b, table lines
71-76: DINOv3 computed fp32 with D 1024, C-RADIOv4 computed fp16 with D 1152, SigLIP 2 so400m computed fp16 with
D 1152; the variants are extract_features.py's defaults, which the extraction Jobs used because extract_fastwam.py
passes no variant). A variant or dtype mismatch fails the test.

Batch sizes 1, 2 and 16. Batch 16 uses 18 frames per camera, so extraction's chunk loop ends on a ragged chunk
(16 + 2); batch 1 and 2 use the first 4 frames (CPU cost). Rollout's Featurizer is called on the same chunks as the
extraction loop and the two must match exactly (TOL 0). The encoder alone is also not batch-invariant on CPU (small
fp16 differences between batch 1 or 2 and batch 16, measured first, then bounded by BATCH_REL times the max abs value);
that is a property of the encoders, the same for both paths, and is reported rather than treated as path parity.
Runs on CPU by default (FEATURE_PARITY_DEVICE=mps for the Mac GPU). Batch-dependent CUDA kernels are NOT covered:
the cluster extraction and rollout ran on GPUs whose kernels can pick different algorithms per batch size, so this
test shows the code paths agree, not that CUDA outputs are bit-identical across batch sizes.

Real encoder weights, from the local Hugging Face cache (DINOv3 is gated). A test skips when its weights cannot be
loaded. The Wan VAE is not part of this question: its loader is replaced by a stub returning zeros of the right shape
while Featurizer is built (the VAE weights are not on this Mac). The encoder branch of Featurizer is the real code.
(test_rollout.py and test_pretrained_train.py do not cover encoders: they use fake featurizers.)

Not covered: frame preparation before the encoder (decode + to_224 at cache time, process_obs at rollout). See the
report on PR #55: the two resizers differ by one uint8 level on about a fifth of the pixels.

Run from src/: python test_feature_parity.py
"""

import inspect
import os
import sys
import types

import numpy as np
import torch

import extract_fastwam as X
import extract_features as EF
import fastwam_data as FD
import rollout as R

DEVICE = os.environ.get("FEATURE_PARITY_DEVICE", "cpu")
N = 18                   # frames per camera
SMALL_N = 4              # frames for batch 1 and 2 (CPU cost)
TOL = 0.0                # same device, same batch, same code: expected bit-identical; fixed before the first run
BATCH_REL = 1e-2         # the encoder itself at another batch size: relative to the max abs value. Set AFTER a first run
                         # showed 1e-3 to 4e-3 on CPU (fp16 rounding differences from the batch-dependent GEMM blocking)
SOURCES = ("dinov3", "siglip2", "c-radiov4")

# What the caches used (data/README.md lines 71-76 and extract_features.py defaults, recorded in each _meta json).
CACHE = {
    "dinov3": {"hf_id": "facebook/dinov3-vitl16-pretrain-lvd1689m", "resolution": 224, "dtype": torch.float32, "D": 1024},
    "siglip2": {"hf_id": "google/siglip2-so400m-patch14-384", "resolution": 384, "dtype": torch.float16, "D": 1152},
    "c-radiov4": {"hf_id": "nvidia/C-RADIOv4-SO400M", "resolution": 448, "dtype": torch.float16, "D": 1152,
                  "revision": "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8"},
}


def stub_vae():
    def encode(x, scale):
        return torch.zeros(x.shape[0], 48, 1, *R.PW.GRID)
    return types.SimpleNamespace(model=types.SimpleNamespace(encode=encode), scale=None)


def frames(seed):
    """Smooth-ish uint8 frames (N, 224, 224, 3), a different pattern per camera, so the grid is not flat."""
    g = np.random.default_rng(seed)
    base = g.integers(0, 256, (N, 14, 14, 3)).astype(np.float32)
    big = torch.from_numpy(base).permute(0, 3, 1, 2)
    big = torch.nn.functional.interpolate(big, size=(224, 224), mode="bilinear", align_corners=False)
    return big.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def check_spec(side, source, enc):
    want = CACHE[source]
    assert enc.hf_id == want["hf_id"], (side, source, enc.hf_id, want["hf_id"])
    assert enc.resolution == want["resolution"], (side, source, enc.resolution, want["resolution"])
    assert enc.dtype == want["dtype"], (side, source, enc.dtype, want["dtype"])
    if "revision" in want:      # the loader pins it in code; the encoder object does not carry it
        assert want["revision"] in inspect.getsource(EF.load_cradiov4), (side, source, "revision pin changed")


def build_both(source):
    """(extraction encoder, rollout Featurizer), each built its own way."""
    ext = EF.ENCODERS[source](DEVICE, CACHE[source]["dtype"])                 # extract_fastwam.main, args.dtype per cache
    orig = X.load_vae
    X.load_vae = lambda *a, **k: stub_vae()
    try:
        feat = R.Featurizer([source], "unused-vae-path", DEVICE)             # the real __init__: R.ENC_DTYPE, own loader
    finally:
        X.load_vae = orig
    assert [s for s, _ in feat.encs] == [source]
    check_spec("extraction", source, ext)
    check_spec("rollout", source, feat.encs[0][1])
    return ext, feat


def to_f32(a):
    return a.astype(np.float32)


def parity(source, built=None):
    ext, feat = built or build_both(source)
    agent, wrist = frames(0), frames(1)
    cams = {c: torch.from_numpy(u).permute(0, 3, 1, 2).float() / 255.0 for c, u in zip(FD.CAMS, (agent, wrist))}
    D = CACHE[source]["D"]

    def train(n, b):
        out = X.encoder_episode(ext, {c: v[:n] for c, v in cams.items()}, b)
        assert out.dtype == np.float16 and out.shape == (n, 14, 28, D), (out.dtype, out.shape)
        return out

    def roll(n, b):
        """Featurizer on the same chunks the extraction loop uses."""
        outs = []
        for s in range(0, n, b):
            o = feat(agent[s:min(n, s + b)], wrist[s:min(n, s + b)])[1]
            assert o.dtype == torch.float32 and o.shape[1:] == (14, 28, D), (o.dtype, o.shape)
            outs.append(o.cpu().numpy().astype(np.float16))
        return np.concatenate(outs)

    def mad(x, y):
        return float(np.abs(to_f32(x) - to_f32(y)).max())

    ref = train(N, 16)                                   # 18 frames: chunks of 16 and 2
    assert np.isfinite(to_f32(ref)).all() and np.abs(to_f32(ref)).max() > 0
    scale = float(np.abs(to_f32(ref)).max())
    path = {"batch16": mad(ref, roll(N, 16))}            # path parity, same chunking
    sens = {}                                            # batch-size sensitivity of the encoder itself
    for b in (1, 2):
        t = train(SMALL_N, b)
        path[f"batch{b}"] = mad(t, roll(SMALL_N, b))
        sens[f"batch{b}_vs_16"] = mad(t, ref[:SMALL_N])
    print(f"{source}: {ext.hf_id} res {ext.resolution} dtype {ext.dtype} device {DEVICE}, grid {ref.shape} float16, "
          f"max abs value {scale:.3g}", flush=True)
    print("  max abs diff, extraction vs rollout at the same batch: " + ", ".join(f"{k} {v:.3g}" for k, v in path.items()),
          flush=True)
    print("  max abs diff, same path across batch sizes (1 or 2 vs 16, first %d frames): " % SMALL_N
          + ", ".join(f"{k} {v:.3g}" for k, v in sens.items()), flush=True)
    assert max(path.values()) <= TOL, (source, path)
    assert max(sens.values()) <= BATCH_REL * scale, (source, sens, BATCH_REL * scale)
    return path, sens


def _case(source):
    def test():
        try:
            import pytest
        except ImportError:
            pytest = None
        try:
            built = build_both(source)
        except AssertionError:
            raise
        except Exception as e:      # weights missing, gated, or no network: skip, never substitute
            if pytest:
                pytest.skip(f"{source} weights unavailable: {type(e).__name__}: {e}")
            raise
        parity(source, built)
    test.__name__ = "test_parity_" + source.replace("-", "_")
    return test


for _s in SOURCES:
    globals()["test_parity_" + _s.replace("-", "_")] = _case(_s)


if __name__ == "__main__":
    bad = 0
    for s in SOURCES:
        try:
            parity(s)
        except AssertionError as e:
            bad += 1
            print(f"{s}: FAILED {e}", file=sys.stderr)
        except Exception as e:
            bad += 1
            print(f"{s}: could not run ({type(e).__name__}: {e})", file=sys.stderr)
    sys.exit(1 if bad else 0)
