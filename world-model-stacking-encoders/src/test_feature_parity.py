"""Training-versus-rollout feature parity (stage 2 PLAN, "What the implementation session must add", item 2).

The same uint8 frames go through extract_fastwam.encoder_episode (what filled the training caches) and through
rollout.Featurizer.__call__ (what the policy sees at rollout), for dinov3, siglip2 and c-radiov4 with the dtypes the
caches used (rollout.ENC_DTYPE, data/README.md section 4b). The two float16 (n, 14, 28, D) grids must match.

Real encoder weights, from the local Hugging Face cache (DINOv3 is gated). A test skips when its weights cannot be
loaded. The Wan VAE is not part of this question: Featurizer is built without its __init__ (which loads the VAE) and
gets a stub VAE that returns zeros of the right shape; the encoder branch of Featurizer.__call__ is the real code.
(test_rollout.py and test_pretrained_train.py do not cover encoders: they use fake featurizers.)

Run from src/: python test_feature_parity.py   (CPU by default; FEATURE_PARITY_DEVICE=mps to use the Mac GPU)
"""

import os
import sys
import types

import numpy as np
import torch

import extract_fastwam as X
import fastwam_data as FD
import rollout as R

DEVICE = os.environ.get("FEATURE_PARITY_DEVICE", "cpu")
N = 2                    # frames per camera
TOL = 0.0                # same device, same batch, same code path: expected bit-identical; fixed before the first run


def stub_vae():
    def encode(x, scale):
        return torch.zeros(x.shape[0], 48, 1, *R.PW.GRID)
    return types.SimpleNamespace(model=types.SimpleNamespace(encode=encode), scale=None)


def frames(seed):
    """Smooth-ish uint8 frames (224, 224, 3), a different pattern per camera, so the grid is not flat."""
    g = np.random.default_rng(seed)
    base = g.integers(0, 256, (N, 14, 14, 3)).astype(np.float32)
    big = torch.from_numpy(base).permute(0, 3, 1, 2)
    big = torch.nn.functional.interpolate(big, size=(224, 224), mode="bilinear", align_corners=False)
    return big.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def parity(source, enc=None):
    from extract_features import ENCODERS
    agent, wrist = frames(0), frames(1)
    enc = enc or ENCODERS[source](DEVICE, R.ENC_DTYPE[source])
    # training path: encoder_episode takes float [0, 1] frames per camera, (n, 3, 224, 224), and rounds back to uint8
    cams = {c: torch.from_numpy(u).permute(0, 3, 1, 2).float() / 255.0 for c, u in zip(FD.CAMS, (agent, wrist))}
    train = X.encoder_episode(enc, cams, N)
    feat = R.Featurizer.__new__(R.Featurizer)
    feat.device, feat.vae, feat.encs = DEVICE, stub_vae(), [(source, enc)]
    roll = feat(agent, wrist)[1]
    assert roll.dtype == torch.float32 and train.dtype == np.float16
    assert train.shape == tuple(roll.shape) and train.shape[:3] == (N, 14, 28), (train.shape, roll.shape)
    roll16 = roll.cpu().numpy().astype(np.float16)
    assert np.isfinite(train.astype(np.float32)).all() and np.abs(train.astype(np.float32)).max() > 0
    diff = float(np.abs(train.astype(np.float32) - roll16.astype(np.float32)).max())
    print(f"{source}: grid {train.shape} {train.dtype}, dtype {R.ENC_DTYPE[source]}, device {DEVICE}, "
          f"max abs diff {diff:.3g}, max abs value {np.abs(train.astype(np.float32)).max():.3g}", flush=True)
    assert diff <= TOL, (source, diff)
    return diff


def _case(source):
    def test():
        try:
            import pytest
        except ImportError:
            pytest = None
        try:
            from extract_features import ENCODERS
            enc = ENCODERS[source](DEVICE, R.ENC_DTYPE[source])
        except Exception as e:      # weights missing, gated, or no network: skip, never substitute
            if pytest:
                pytest.skip(f"{source} weights unavailable: {type(e).__name__}: {e}")
            raise
        parity(source, enc)
    test.__name__ = "test_parity_" + source.replace("-", "_")
    return test


for _s in ("dinov3", "siglip2", "c-radiov4"):
    globals()["test_parity_" + _s.replace("-", "_")] = _case(_s)


if __name__ == "__main__":
    bad = 0
    for s in ("dinov3", "siglip2", "c-radiov4"):
        try:
            parity(s)
        except AssertionError:
            raise
        except Exception as e:
            bad += 1
            print(f"{s}: could not run ({type(e).__name__}: {e})", file=sys.stderr)
    sys.exit(1 if bad else 0)
