"""Read Fast-WAM's LIBERO data (HF yuanty/LIBERO-fastwam, LeRobot v3.0 layout) the way Fast-WAM's training does.

Copied from FastWAM 7faa711 (configs/data/libero_2cam*.yaml, datasets/lerobot/*):
  - cameras `observation.images.image` (agentview) and `observation.images.wrist_image`, stored 512x512, AV1, 20 fps.
    Frames are already upright (checked by eye on libero_goal episode 432, 2026-10-28): no rotation here.
    Fast-WAM's rollouts rotate the raw sim frame 180 deg to match.
  - per camera: uint8 -> float / 255 -> torchvision Resize([224, 224]) (bilinear, antialias); cameras concatenated
    horizontally to 224 x 448; normalized to [-1, 1] for the VAE.
  - proprio = `observation.state` (ee pose 6 + gripper 2); action = `action` (7: 6 delta ee + gripper).

Example:
  eps = read_suite("/data/lerobot_v30/libero_goal_no_noops_lerobot")
  frames = decode_episode(root, eps[0])          # {cam: (n, 512, 512, 3) uint8}
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CAMS = ["observation.images.image", "observation.images.wrist_image"]   # agentview, wrist; Fast-WAM's order
SIZE = 224                                                              # per camera, Fast-WAM shape_meta


def read_suite(root):
    """Episodes of one suite: index, length, task, action (n, 7), state (n, 8), and video refs per camera."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    meta = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))])
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("chunk-*/file-*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    assert (data["index"].to_numpy() == np.arange(len(data))).all(), "global frame index is not contiguous"
    eps = []
    for _, m in meta.sort_values("episode_index").iterrows():
        lo, hi = int(m["dataset_from_index"]), int(m["dataset_to_index"])
        d = data.iloc[lo:hi]
        assert (d["episode_index"] == m["episode_index"]).all() and len(d) == m["length"]
        assert (d["frame_index"].to_numpy() == np.arange(len(d))).all()
        assert len(m["tasks"]) == 1, m["tasks"]
        eps.append({
            "episode_index": int(m["episode_index"]), "length": int(m["length"]), "task": str(m["tasks"][0]),
            "action": np.stack(d["action"].to_numpy()).astype(np.float32),
            "state": np.stack(d["observation.state"].to_numpy()).astype(np.float32),
            "video": {c: {"path": info["video_path"].format(video_key=c,
                                                             chunk_index=int(m[f"videos/{c}/chunk_index"]),
                                                             file_index=int(m[f"videos/{c}/file_index"])),
                          "from": float(m[f"videos/{c}/from_timestamp"]), "to": float(m[f"videos/{c}/to_timestamp"])}
                      for c in CAMS},
        })
    return eps


def decode_episode(root, ep, fps=20):
    """{cam: (n, H, W, 3) uint8}: the episode's frames, matched by timestamp (one frame per 1 / fps)."""
    out = {}
    n = ep["length"]
    for c in CAMS:
        v = ep["video"][c]
        want = v["from"] + np.arange(n) / fps
        frames = [None] * n
        with av.open(str(Path(root) / v["path"])) as f:
            s = f.streams.video[0]
            f.seek(int(max(v["from"] - 1.0, 0) / s.time_base), stream=s, backward=True)
            for fr in f.decode(s):
                t = float(fr.pts * s.time_base)
                if t >= v["to"] - 0.5 / fps:
                    break
                i = int(round((t - v["from"]) * fps))
                if 0 <= i < n and abs(t - want[i]) < 0.5 / fps:
                    frames[i] = fr.to_ndarray(format="rgb24")
        missing = [i for i, x in enumerate(frames) if x is None]
        assert not missing, f"episode {ep['episode_index']} {c}: no frame for indices {missing[:5]}..."
        out[c] = np.stack(frames)
    return out


def to_224(frames):
    """(n, H, W, 3) uint8 -> (n, 3, 224, 224) float32 in [0, 1], as Fast-WAM's ToTensor + Resize([224, 224])."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    return TF.resize(x, [SIZE, SIZE], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


def video_clips(cams224, starts, num_frames=9, stride=4):
    """VAE input for clips starting at each t in `starts`: (B, 3, num_frames, 224, 448) in [-1, 1].

    Frames t, t + stride, ..., clamped to the last frame of the episode (Fast-WAM's index clamp; the matching
    `image_is_pad` flags are rebuilt by the loader). Cameras side by side in CAMS order (concat_multi_camera:
    horizontal), then Normalize(mean 0.5, std 0.5)."""
    n = cams224[CAMS[0]].shape[0]
    idx = np.minimum(np.asarray(starts)[:, None] + stride * np.arange(num_frames)[None], n - 1)  # (B, T)
    v = torch.cat([cams224[c][torch.from_numpy(idx)] for c in CAMS], dim=-1)                  # (B, T, 3, 224, 448)
    return ((v - 0.5) / 0.5).permute(0, 2, 1, 3, 4).contiguous()


def grid_to_vae(tokens, size=(14, 14)):
    """(n, h, w, D) encoder patch grid -> (n, 14, 14, D), bilinear (liu2026what App. B.2: spatial resizing is
    bilinear). A no-op for encoders whose 224 grid is already 14 x 14."""
    if tuple(tokens.shape[1:3]) == tuple(size):
        return tokens
    x = tokens.permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
